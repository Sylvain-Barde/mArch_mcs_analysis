# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 12:40:27 2023

This files runs the benchmarking analysis for the fast implementation of the 
Model Confidence Set algorithm (fastMCS)

NOTE: Due to the time-consuming nature of the MC exercise the script uses the 
multiprocessing package to parallelize MC simulations over a set of 20 cores, 
with high memory usage (up to 24GB). Users should ensure they have access to 
the required compute resources before running the script, or set a small set of
parallel cores (line 395, variable 'numSeeds').

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
import sobol

#-----------------------------------------------------------------------------
def get_sobol_samples(num_samples, parameter_support, skips):
    """
    Draw multi-dimensional sobol samples

    Parameters
    ----------
    num_samples : int
        Number of samples to draw.
    parameter_support : ndarray
        A 2D set of bounds for each parameter. Structure is:
            2 x numParams
        with row 0 containing lower bounds, row 1 upper bounds.
    skips : int
        Number of draws to skip from the start of the sobol sequence.

    Returns
    -------
    sobol_samples : ndarray
        A 2D set of Sobol draws. Structure is:
            num_samples x num_param

    """
    params = np.transpose(parameter_support)
    sobol_samples = params[0,:] + sobol.sample(
                        dimension = parameter_support.shape[0], 
                        n_points = num_samples, 
                        skip = skips
                        )*(params[1,:]-params[0,:])
    
    return sobol_samples


def genLosses(lmbda, rho, phi, M, N, seed = 0, shuffle = True):
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
    L : ndarray
        2D ndarray of synthetic losses. Structure is N x M

    """
    
    rng = default_rng(seed = seed)
    
    theta = lmbda*np.arange(0,M)/((M-1)*N**0.5)

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
    if shuffle:
        # generate column permutation
        perm = np.arange(0,M)
        rng.shuffle(perm)
        L = L[:,perm]
    
    return L


def benchmarkMCS(inputs):
    """
    Main benchmarking function
    - Generates 32 collections of models, with collection sizes evenly spaced 
      between 2 and 4 in log_10 space (up to an integer). Parameters for losses
      are drawn from sobol sequences (unique to each replication)
    - Runs 3 MCS analyses on each capturing time and memory requirements:
        - The standard elimination algorithm (if |M| < 1000)
        - The 2-pass fast MCS analysis
        - The 1-pass fast MCS analysis
    - Runs comparisons to identify any discrepancies in MCS sizes, composition,
      t-scores and P-values

    This function is parallelised over random seeds as part of a Monte Carlo 
    analysis

    Parameters
    ----------
    inputs : list
        Inputs for the benchmarking exercise.
         inputs[0] contains a vector of parameters for the exercise
             inputs[0][0]     # Distinct skip value for the sobol sequence
             inputs[0][1]     # Number of observations in losses
             inputs[0][2]     # Number of bootstrap replications for MCS
             inputs[0][3]     # Bootstrap block width

         inputs[1] contains a path for logging results
         inputs[2] contains a distinct seed for the RNG

    Returns
    -------
    runTime : float
        The running time for the function.
    returnDict : dict
        Dictionary containing the results of the analysis. Fields are:
        -'t0': vector of times taken to run the elimination version
        -'t1': vector of times taken to run the 1-pass version
        -'t2': vector of times taken to run the 2-pass version
        -'mem0': vector of memory consumptions for the elimination version
        -'mem1': vector of memory consumptions for the 1-pass version
        -'mem2': vector of memory consumptions for the 2-pass version
        -'checkRank2': ranking check, 2-pass vs. elimination 
        -'checkRank1': ranking check, 2-pass vs. 1-pass 
        -'checkMcs2': MCS check, 2-pass vs. elimination 
        -'checkMcs1': MCS check, 2-pass vs. 1-pass 
        -'numMCSdiff2': diff. in MCS sizes, 2-pass vs. elimination 
        -'numMCSdiff1': diff. in MCS sizes, 2-pass vs. 1-pass
        -'checkTScores2': diff. in t-scores, 2-pass vs. elimination
        -'checkTScores1': diff. in t-scores, 2-pass vs. 1-pass
        -'numTScoreDiffs2': num. of diff. t-scores, 2-pass vs. elimination
        -'meanTScoreDiffs2': average t-scores diff., 2-pass vs. 1-pass
        -'checkPvals2': P-val check, 2-pass vs. elimination
        -'checkPvals1': P-val check, 2-pass vs. 1-pass
        -'numPvalDiffs2': num. of diff. P-vals, 2-pass vs. elimination
        -'meanPvalDiffs2': average P-val diff P-vals, 2-pass vs. elimination
        -'numPvalDiffs1': num. of diff. P-vals, 2-pass vs. 1-pass
        -'meanPvalDiffs1': average P-val diff P-vals, 2-pass vs. 1-pass
    """
    
    runTimeStart = time.time()
    
    # -- Unpack inputs and set hard-coded constants
    seed = inputs[2]
    log_path = inputs[1]

    baseSkip = inputs[0][0]     # Skip value for the sobol sequence
    N = inputs[0][1]            # Number of observations
    B = inputs[0][2]            # Number of bootstrap replications for MCS
    b = inputs[0][3]            # Bootstrap block width
    
    numMods = 32
    m = np.arange(2,4+2/(numMods-1),2/(numMods-1)) 
    mods = np.round(10**m).astype(np.int32)
    mods[np.where(np.log2(mods) % 1 == 0)] -= 1 # Control for exact powers of 2
    numElimination = np.sum(mods < 1000) # Number of models below 1000
    
    # -- Generate Sobol draws for the loss generation 
    # - These are different for all models collection sizes
    # - They are also different across MC replications
    skip = baseSkip + 50*seed
    parameterRange = np.asarray([[5,40],
                                 [0,0.95],
                                 [0,0.8]])
    params = get_sobol_samples(numMods, parameterRange, skip)
    
    # -- Prepare MC diagnostics return vectors
    t0 = np.zeros(numElimination)
    t1 = np.zeros(numMods)
    t2 = np.zeros(numMods)
    
    mem0 = np.zeros(numElimination)
    mem1 = np.zeros(numMods)
    mem2 = np.zeros(numMods)
    
    checkRank2 = np.zeros(numElimination, dtype=np.bool)
    checkRank1 = np.zeros(numMods, dtype=np.bool)
    
    checkMcs2 = np.zeros(numElimination, dtype=np.bool)
    checkMcs1 = np.zeros(numMods, dtype=np.bool)
    numMCSdiff2 = np.zeros(numElimination)
    numMCSdiff1 = np.zeros(numMods)
    
    checkTScores2 = np.zeros(numElimination, dtype=np.bool)
    checkTScores1 = np.zeros(numMods, dtype=np.bool)
    numTScoreDiffs2 = np.zeros(numElimination)
    meanTScoreDiffs2 = np.zeros(numElimination)
    
    checkPvals2 = np.zeros(numElimination, dtype=np.bool)
    checkPvals1 = np.zeros(numMods, dtype=np.bool)
    numPvalDiffs2 = np.zeros(numElimination)
    meanPvalDiffs2 = np.zeros(numElimination)
    numPvalDiffs1 = np.zeros(numMods)
    meanPvalDiffs1 = np.zeros(numMods)
    
    # -- Initialise log file, error file and display config settings
    file_name = 'mcIter_{:d}'.format(int(seed))
    sys.stdout = open(log_path + '//' + file_name + '.out', "w")
    
    print('Run {:d} initialised'.format(int(seed) ),
          flush=True)
    
    # Iterate over the different model collection sizes
    # Processing order is randomised to smooth memory consumption and avoid 
    # affecting time-complexity measurement through synchronisation of memory
    # IO across parallel runs of the functions
    rng = default_rng(seed = seed)
    perm = np.arange(0,len(mods))
    rng.shuffle(perm)
    mods = mods[perm]

    for i, M in zip(perm,mods):
   
        tStart = time.time()
    
        # -- Generate losses from sobol draw - dedicated seed per iteration
        lmbda,rho,phi = params[i]
        losses = genLosses(lmbda, rho, phi, M, N, seed = i + seed*50)
        
        # -- Run MCS with 2 fast algorithms - same seed for all
        mcsEst2 = mcs(seed = seed, verbose = False)
        mcsEst2.addLosses(losses)
        mcsEst2.run(B, b, bootstrap = 'block')
        incl2, excl2 = mcsEst2.getMCS(alpha = 0.1)
        rank2 = np.concatenate((excl2,incl2))
        tScore2 = mcsEst2.tScore
        pVals2 = mcsEst2.pVals
            
        mcsEst1 = mcs(seed = seed, verbose = False)
        mcsEst1.addLosses(losses)
        mcsEst1.run(B,b, bootstrap = 'block', algorithm = '1-pass')
        incl1, excl1 = mcsEst1.getMCS(alpha = 0.1)
        rank1 = np.concatenate((excl1,incl1))
        tScore1 = mcsEst1.tScore
        pVals1 = mcsEst1.pVals
        
        # -- Calculate diagnostics
        # Time and memory
        t1[i] = mcsEst1.stats[0]['time']
        t2[i] = mcsEst2.stats[0]['time']
        
        mem1[i] = mcsEst1.stats[0]['memory use']*10**-6
        mem2[i] = mcsEst2.stats[0]['memory use']*10**-6
        
        del mcsEst1
        del mcsEst2
        
        # Check overall rankings match - This should match in both cases
        checkRank1[i] = np.array_equal(rank2,rank1)
        
        # Check MCS lengths. 1-pass might be different
        checkMcs1[i] = set(incl2) == set(incl1)
        if not checkMcs1[i]:
            numMCSdiff1[i] = len(incl1) - len(incl2)
            
        # Check T-score differences - This should match in both cases
        # Note, using a tolerance due to floating point arithmetic differences
        checkTScores1[i] = np.allclose(tScore1,tScore2, rtol=1e-06, atol=1e-12)
        
        # Check p-value differences. 1-pass might be different
        checkPvals1[i] = np.array_equal(pVals1,pVals2)
        if not checkPvals1[i]:
            numPvalDiffs1[i] = np.count_nonzero(pVals1!=pVals2)
            meanPvalDiffs1[i] = sum(np.abs(pVals1-pVals2))/numPvalDiffs1[i]
        
        # If model collection size is below threshold, run elimination as well
        if M < 1000:
            mcsEst0 = mcs(seed = seed, verbose = False)
            mcsEst0.addLosses(losses)
            mcsEst0.run(B,b, bootstrap = 'block', algorithm = 'elimination')
            incl0, excl0 = mcsEst0.getMCS(alpha = 0.1)
            rank0 = np.concatenate((excl0,incl0))
            tScore0 = mcsEst0.tScore
            pVals0 = mcsEst0.pVals
        
            # -- Calculate diagnostics for elimination as well
            # time and memory
            t0[i] = mcsEst0.stats[0]['time']       
            mem0[i] = mcsEst0.stats[0]['memory use']*10**-6
            del mcsEst0
            
            # Check overall rankings match - This should match in both cases
            checkRank2[i] = np.array_equal(rank2,rank0)
            
            # Check MCS lengths. 2-pass should be fine
            checkMcs2[i] = set(incl2) == set(incl0)
            if not checkMcs2[i]:
                numMCSdiff2[i] = len(incl0) - len(incl2)
                
            # Check T-score differences - This should match in both cases
            # Note, using a tolerance due to floating point difference
            checkTScores2[i] = np.allclose(tScore0, tScore2, 
                                           rtol=1e-06, atol=1e-12)
            if not checkTScores2[i]:
                numTScoreDiffs2[i] = np.sum(1 -
                    np.isclose(tScore0, tScore2, rtol=1e-06, atol=1e-12))
                meanTScoreDiffs2[i] = sum(
                                    np.abs(tScore0-tScore2)
                                            )/numTScoreDiffs2[i]
            
            # Check p-value differences. 2-pass should be fine
            checkPvals2[i] = np.array_equal(pVals0,pVals2)
            if not checkPvals2[i]:
                numPvalDiffs2[i] = np.count_nonzero(pVals0!=pVals2)
                meanPvalDiffs2[i] = sum(np.abs(pVals0-pVals2))/numPvalDiffs2[i]
                
        print(' iteration {:d}, collection size = {:d}, time = {:f}'.format(
            i,M,time.time()-tStart), flush = True)
        
    # Package return variables
    returnDict = {'t0':t0,
                  't1':t1,
                  't2':t2,
                  'mem0':mem0,
                  'mem1':mem1,
                  'mem2':mem2,
                  'checkRank2':checkRank2,
                  'checkRank1':checkRank1,
                  'checkMcs2':checkMcs2,
                  'checkMcs1':checkMcs1,
                  'numMCSdiff2':numMCSdiff2,
                  'numMCSdiff1':numMCSdiff1,
                  'checkTScores2':checkTScores2,
                  'checkTScores1':checkTScores1,
                  'numTScoreDiffs2':numTScoreDiffs2,
                  'meanTScoreDiffs2':meanTScoreDiffs2,
                  'checkPvals2':checkPvals2,
                  'checkPvals1':checkPvals1,
                  'numPvalDiffs2':numPvalDiffs2,
                  'meanPvalDiffs2':meanPvalDiffs2,
                  'numPvalDiffs1':numPvalDiffs1,
                  'meanPvalDiffs1':meanPvalDiffs1}
    
    runTime = time.time() - runTimeStart
    print('Run {:d} finished, time: {:f}'.format(int(seed),runTime),
          flush=True)
    
    return (runTime,returnDict)


#-----------------------------------------------------------------------------
if __name__ == '__main__':
    
    mp.set_start_method('spawn')     # for 3.9, ensure compatibility with code.
    
    # Set benchmarking parameters here
    numSeeds = 20
    baseSeed = 180           # Set to [0, 20, 40, 60, 80]
    baseSkip = 1000        # Set to 1000 for N=250, 11000 for N=30
    N = 250                # Set to [30, 250]
    B = 1000
    b = 2
    
    # -- Create logging/saving directories
    runName = "benchmarking_N_{:d}_seed_{:d}_run_{:s}".format(N,baseSeed,
        time.strftime("%d-%b-%Y_%H-%M-%S",time.gmtime()))
    log_path = "logs//" + runName
    print('Saving logs to: {:s}'.format(log_path))
    os.makedirs(log_path,mode=0o777)
    
    save_path = "montecarlo//benchmarking_results_N_{:d}".format(N)
    if not os.path.exists(save_path):
        os.makedirs(save_path,mode=0o777)
    
    # -- Prepare parallel MC settings
    params = [baseSkip,N,B,b]

    mcSettings = []
    for i in range(numSeeds):
        mcSettings.append((params,
                           log_path,
                           baseSeed + i))
    
    # -- Run parallel benchmarking exercise
    num_cores = numSeeds # Core count is throttled to preserve RAM-per-CPU.
    t_start = time.time()
           
    # -- Initialise Display
    print(' ')
    print('+------------------------------------------------------+')
    print('|             Parallel fast MCS benchmarking           |')
    print('+------------------------------------------------------+')    
    print(' Number of cores: {:d} - Number of tasks: {:d}'.format(
                num_cores,numSeeds))
    print(' Base seed: {:d}'.format(baseSeed))
    print('+------------------------------------------------------+')
    
    # -- Create pool and run parallel job
    pool = mp.Pool(processes=num_cores)
    res = pool.map(benchmarkMCS,mcSettings)
    
    # Close pool when job is done
    pool.close()
    
    # -- Extract results, get timers, save output
    sum_time = 0
    fullResults = {}
    for i in range(numSeeds):
        res_i = res[i]
        sum_time = sum_time + res_i[0]
        fullResults[baseSeed + i] = res_i[1]


    fil = open(save_path+'//baseSeed_{:d}.pkl'.format(baseSeed),'wb') 
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