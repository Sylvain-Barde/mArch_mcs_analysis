# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 14:09:45 2025

This files runs the power analysis analysis for the fast implementation of the 
Model Confidence Set algorithm (fastMCS)

@author: Sylvain Barde, University of Kent
"""

import time
import sys
import os
import pickle
import zlib
import numpy as np
import multiprocessing as mp

from numpy.random import default_rng
from scipy.stats import norm
from fastMCS import mcs

def genLosses(lmbda, rho, phi, M, N, M0 = 1, seed = 0, shuffle = True):
    """
    Generate synthetic losses for the benchmarking exercise. Uses the 
    simulation experimet outlined in section 5.1 (page 477) of:
    
    Hansen, P.R., Lunde, A. and Nason, J.M., 2011. The model confidence set. 
    Econometrica, 79(2), pp.453-497.

    Parameters
    ----------
    lmbda : float
        Lambda parameter, controls the relative performance of different models
        (higher lambda -> models are easier to distinguish)
    rho : float
        Controls the correlation between model losses.
    phi : float
        Controls the level of conditional heteroskedasticity (GARCH effects) in
        model losses
    M : Int
        Number of models desired
    N : int
        Number of observations
    M0: int, optional
        Number of no-loss models desired. These models will form the set of 
        superior models. The default is 1
    seed : flaot, optional
        Seed for the random number generator. The default is 0.
    shuffle : boolean, optional
        Controls whether losses are returned ordered or shuffled.
        - If set to False, losses are returned 'as generated', ordered with 
          model 1 displayignt ht lowest average loss and model M the highest.
        - If set to True, columns of L are shuffled so that the average 
          performance of models is randomised
        The default is True.

    Returns
    -------
    A tuple containing:
    - L : ndarray
        2D ndarray of synthetic losses. Structure is N x 
    - perm : ndarray
        1D ndarray containing the permutation applied to the losses. 

    """
    
    rng = default_rng(seed = seed)

    # Concatenate M0 no-loss models and M-M0 models with positive loss
    theta0 = np.zeros(M0-1)
    theta1 = lmbda*np.arange(0,M - M0 + 1)/((M - M0)*N**0.5)
    theta = np.concatenate((theta0,theta1))

    S = np.ones([M,M])*rho + np.diag(np.ones(M) - rho)
    Sroot = np.linalg.cholesky(S)
    X = Sroot @ norm.ppf(rng.random((M,N)))   # Correlated shocks
    
    e = norm.ppf(rng.random(N+1))
    y = np.zeros(N+1)
    y[0] = -phi/(2*(1+phi)) + e[0]*phi**0.5
    for i in range(1,N+1):
        y[i] = -phi/(2*(1+phi)) + y[i-1]*phi + e[i]*phi**0.5
    
    a = np.exp(y[1:N+1])
    
    L = a[:,None]*X.transpose()
    L /= np.std(L,axis = 0)[None,:]
    L += theta[None,:]

    perm = np.arange(0,M)
    if shuffle:
        # generate column permutation
        rng.shuffle(perm)
        L = L[:,perm]
    
    return (L,perm)

def powerMCS(inputs):
    """
    Power analysis
    This function replicates the size and power analysis as displayed in table
    2, section 5.1 (page 477) of:
    
    Hansen, P.R., Lunde, A. and Nason, J.M., 2011. The model confidence set. 
    Econometrica, 79(2), pp.453-497.

    Using the fast MCS algorithm

    This function is parallelised over random seeds as part of  Monte Carlo 
    analysis.

    Parameters
    ----------
    inputs : list
        Inputs for the power analysis.
         inputs[0] contains vectors of variable parameters for the exercise
             inputs[0][0]     # Vector of Lambda values
             inputs[0][1]     # Vector of Rho values
             inputs[0][2]     # Vector of Phi values
             inputs[0][3]     # Vector of Model sizes

         inputs[1] contains a dict of common parameters
         inputs[2] contains a path for logging results
         inputs[3] contains a distinct seed for the RNG

    Returns
    -------
    runTime : float
        The running time for the function.
    returnDict : dict
        Dictionary containing the results of the analysis. Fields are:
        -'sizeMcIter': Array of size checks for MC iteration
        -'powerMcIter': Array of power checks for MC iteration
    """

    runTimeStart = time.time()

    # -- Unpack inputs and set hard-coded constants
    M0 = inputs[0][0]       # Size of superior model set
    numObs = inputs[0][1]   # Number of observations
    B = inputs[0][2]        # Number of bootstrap replications
    b = inputs[0][3]        # Width of bootstrap window
    alpha = inputs[0][4]    # significance level for MCS

    lmbdas = inputs[1][0]   # List of Lambda values
    rhos = inputs[1][1]     # List of Rho values
    phis = inputs[1][2]     # List of Phi values
    Nmods = inputs[1][3]    # List of model collectiion sizes

    log_path = inputs[2]    # Path to logs
    seed = inputs[3]        # MC iteration base seed
    
    # -- Prepare MC diagnostics return vectors
    sizeMcIter = np.zeros((len(lmbdas),
                            len(rhos),
                            len(phis),
                            len(Nmods)))
    
    powerMcIter = np.zeros((len(lmbdas),
                            len(rhos),
                            len(phis),
                            len(Nmods)))
    
    # -- Initialise log file, error file and display config settings
    file_name = 'mcIter_{:d}'.format(int(seed))
    sys.stdout = open(log_path + '//' + file_name + '.out', "w")
    
    print('Run {:d} initialised'.format(int(seed) ),
          flush=True)
    
    print('Monte Carlo run')
    print(u'\u2500' * 75)
    # print('-' * 75)
    for i, lmbda in enumerate(lmbdas):
        for j, rho in enumerate(rhos):
            for k, phi in enumerate(phis):
                for m, Nmod in enumerate(Nmods):
                    
                    tic = time.time()
                    seed += 1
                    print(' Iter: lmda={:d}, rho={:.2f}, phi={:.1f}, mods={:4d}'.format(
                        lmbda, rho, phi, Nmod), end="", flush=True)
                    
                    # Generate losses for iteration, get indices for M_0^*
                    lossesTuple = genLosses(lmbda, 
                                            rho, 
                                            phi, 
                                            Nmod, 
                                            numObs, 
                                            M0 = M0,
                                            seed = seed)                
                    losses = lossesTuple[0]
                    lossPerm = lossesTuple[1]
                    trueModOrder = np.argsort(lossPerm)
                    modsM0 = trueModOrder[0:M0]
                    
                    # Run 2-pass fastMCS, get alpha-level MCS
                    mcsEst = mcs(seed = seed, verbose = False)
                    mcsEst.addLosses(losses)
                    mcsEst.run(B,b, bootstrap = 'block', algorithm = '2-pass')
                    incl, excl = mcsEst.getMCS(alpha = alpha)
                    
                    # Check size/power stats, record in return variables
                    sizeMcIter[i,j,k,m] = set(modsM0).issubset(incl)
                    powerMcIter[i,j,k,m] = len(incl)
    
                    print(' - Done {:8.2f} secs.'.format(time.time() - tic))
    
    # Package return variables
    returnDict = {'sizeMcIter':sizeMcIter, 'powerMcIter':powerMcIter}
    
    print(u'\u2500' * 75)
    # print('-' * 75)
    runTime = time.time() - runTimeStart
    print( 'Total time: {:8.2f}'.format(runTime))
    return (runTime, returnDict)

#-----------------------------------------------------------------------------
if __name__ == '__main__':
    
    lmbdas = [5, 10, 20, 40]
    rhos = [0, 0.5, 0.75, 0.95]
    phis = [0, 0.5, 0.8]
    Nmods = [500, 1000, 2000]
    
    M0 = 10
    numObs = 250
    B = 1000
    b = 2
    alpha = 0.1
    
    numSeeds = 1000
    num_cores = 96
    
    # -- Create logging/saving directories
    runName = "power_N_{:d}_run_{:s}".format(numObs,
        time.strftime("%d-%b-%Y_%H-%M-%S",time.gmtime()))
    log_path = "logs//" + runName
    print('Saving logs to: {:s}'.format(log_path))
    os.makedirs(log_path,mode=0o777)
    
    save_path = "montecarlo//power_results_N_{:d}".format(numObs)
    if not os.path.exists(save_path):
        os.makedirs(save_path,mode=0o777)
        
    # -- Prepare parallel MC settings
    fixedParams = [M0,numObs,B,b,alpha]

    mcSettings = []
    incr = len(lmbdas)*len(rhos)*len(phis)*len(Nmods) # Number of combinations
    for i in range(numSeeds):
        mcSettings.append((fixedParams,
                           [lmbdas,rhos,phis,Nmods],
                           log_path,
                           i*incr))
        
    # -- Run parallel benchmarking exercise
    t_start = time.time()
    
    # -- Initialise Display
    print(' ')
    print('+------------------------------------------------------+')
    print('|            Parallel fast MCS power analysis          |')
    print('+------------------------------------------------------+')    
    print(' Number of cores: {:d} - Number of tasks: {:d}'.format(
                num_cores,numSeeds))
    print(' Number of observations: {:d}'.format(numObs))
    print('+------------------------------------------------------+')
    
    # -- Create pool and run parallel job
    pool = mp.Pool(processes=num_cores)
    res = pool.map(powerMCS,mcSettings)
    
    # Close pool when job is done
    pool.close()
    
    # -- Extract results, get timers, save output
    sum_time = 0
    fullResults = {}
    for i in range(numSeeds):
        res_i = res[i]
        sum_time = sum_time + res_i[0]
        fullResults[i] = res_i[1]
        
    fil = open(save_path+'//power_analysis.pkl','wb') 
    fil.write(zlib.compress(pickle.dumps(fullResults, protocol=2)))
    fil.close()     

    # Print timer/speedup diagnostics
    timer_1 = time.time() - t_start
    print('+------------------------------------------------------+')
    print(' Total running time:     {:10.4f} secs.'.format(timer_1))
    print(' Sum of iteration times: {:10.4f} secs.'.format(sum_time))
    print(' Mean iteration time:    {:10.4f} secs.'.format(
            sum_time/numSeeds))
    print(' Speed up:               {:10.4f}'.format(sum_time/timer_1))
    print('+------------------------------------------------------+')
#-----------------------------------------------------------------------------