# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:19:07 2023

@author: Sylvain Barde

Implements the fast model confidence set algorithm

Requires the following packages:

    numpy

Classes:

    mcs

Utilities:

    blockBootstrap
    stationaryBootstrap

"""

import numpy as np
from numpy.random import default_rng
import time
import sys
import os
import pickle
import zlib


def blockBootstrap(rng,obs,B,b):
    """
    Generate boostrap indices using the block bootstrap method

    Parameters
    ----------
    rng : Instance of "numpy.random.default_rng"
        Random number generator for the .
    obs : int
        Number of observations to be resampled
    B : int
        Number of bootstrap replications required
    b : int
        Bootstrap window

    Returns
    -------
    obs x B ndarray
        Bootsraped indice for resampling.

    """
    
    numBlocks = int(np.ceil(obs/b))
    boostrapInds = np.zeros([numBlocks*b,B])       # Initialise index
    
    for i in range(numBlocks):
        boostrapInds[i*b,:] = np.ceil(obs*rng.random(B))  # Random selection
        for j in range(1,b):
            boostrapInds[i*b+j,:] = boostrapInds[i*b+j-1,:] + 1
    
    boostrapInds = boostrapInds[0:obs,:]
    boostrapInds[boostrapInds > obs-1] = (boostrapInds[boostrapInds > obs-1] 
                                          - obs)
    
    return boostrapInds.astype(int)


def stationaryBootstrap(rng, obs,B,b):
    """
    Generate boostrap indices using the stationary bootstrap method of Politis
    & Romano (1994)


    Parameters
    ----------
    rng : Instance of "numpy.random.default_rng"
        Random number generator for the .
    obs : int
        Number of observations to be resampled
    B : int
        Number of bootstrap replications required
    b : int
        Bootstrap window

    Returns
    -------
    obs x B ndarray
        Bootsraped indice for resampling.

    """
    
    boostrapInds = np.zeros([obs,B])       # Initialise index
    boostrapInds[0,:] = np.ceil(obs*rng.random(B))  # Random starting points
    jump = rng.random([obs,B]) < 1/b                # Random jump locations
    
    # Jump value    
    boostrapInds[np.where(jump)] = np.floor(rng.random([1,sum(sum(jump))])*obs)
    for i in range(1,obs):
        boostrapInds[i,np.where(~jump[i,:])] = (
            boostrapInds[i-1,np.where(~jump[i,:])] + 1)
    
    boostrapInds[boostrapInds > obs-1] = (boostrapInds[boostrapInds > obs-1] 
                                          - obs)
    
    return boostrapInds.astype(int)


class mcs:
    """
    Main module class for the package. See function-level help for more details.

    Attributes:
        verbose (Boolean):
            Flags whether the mcs object displays talkbacks or not
        isProcessed (Boolean):
            Flags whether the losses have already been processed or not
        Losses (ndarray):
            Model losses required for the MCS analysis
        boot (str):
            Bootstrap algorithm used in the analysis
        seed (float):
            Seed used for the bootstrap algorithm
        b (int):
            Bootstrap block size parameter
        bootInds (ndarray):
            Bootstrap indices
        bootMean (ndarray):
            Bootstrapped average losses (used to speed up calculation)
        tScore (ndarray):
            Vector of model eliminations t-statistics
        tBootDist (ndarray):
            Matrix of model Bootstrappedelimination t-statistics
        pVals (ndarray):
            Bootstrapped 
        exclMods (ndarray):
            model elimination sequence      
        stats (list of dicts)
            List of MCS run statistics (one entry per run)
            Each run dict contains the following fields:
                'run': The ID of the run
                'models processed': number of models processed in run
                'method': algorithm used to process the MCS
                'time': time taken in seconds
                'memory use': total memory used
                'memory alloc': detailed allocation of memory over variables
        
    Methods:
        __init__ :
            Initialises an empty instance of the MCS class
        save:
            Saves the current state of the MCS analysis to a pickle format
        load:
            Load a saved MCS analysis
        addLosses:
            Add model losses to the object
        run:
            Run the MCS analysis on the losses
        getMCS
            Retrieve lists of eliminated models and models included in the MCS

    """
    
    def __init__(self, seed = None, verbose = True):
        """
            Initialises an empty instance of the MCS class, with an optional 
            seed passed for the bootstrap algorithm

        Parameters
        ----------
        seed : float, optional
            Desired RNG seed. The default is None, in which case the current 
            time (using time.time() ) is set as the seed
        verbose : boolean, optional
            Set to False to hide normal operation talkbacks (error messages 
            will still be displayed). The defaut is True

        Returns
        -------
        None.

        """
        
        # Initialise empty fields
        self.verbose = verbose
        self.isProcessed = False
        self.Losses = None
        self.boot = None
        self.b = None
        self.bootInds = None
        self.bootMean = None
        self.tScore = None
        self.tBootDist = None
        self.pVals = None        
        self.exclMods = None
        self.stats = []
        
        # Set seed
        if seed is None:
            self.seed = time.time()     # set time as seed source
        else:
            self.seed = seed            # set provided seed
            
    def save(self, path, filename):
        """
        Saves the current state of the MCS analysis to a pickle format

        Parameters
        ----------
        path : str
            path to save directory.
        filename : str
            Filename to be used. File extension (.pkl) set by method

        Returns
        -------
        None.

        """
        
        if self.verbose:
            print(u'\u2500' * 75)
            print(' Saving MCS to: {:s}/{:s} '.format(path, filename), end="", 
                  flush=True)
        
        # Check saving directory exists
        if not os.path.exists(path):
            os.makedirs(path,mode=0o777)
        
        # Save fields to dict
        saveDict = {'isProcessed':self.isProcessed,
                    'Losses':self.Losses,
                    'boot':self.boot,
                    'b':self.b,
                    'bootInds':self.bootInds,
                    'bootMean':self.bootMean,
                    'tScore':self.tScore,
                    'tBootDist':self.tBootDist,
                    'pVals':self.pVals,
                    'exclMods':self.exclMods,
                    'stats':self.stats}

        with open(path + '/' + filename + '.pkl','wb') as f:
            f.write(zlib.compress(pickle.dumps(saveDict, protocol=2)))
        
        if self.verbose:
            print(' - Done')

    def load(self, path, filename):
        """
        Load a saved MCS analysis

        Parameters
        ----------
        path : str
            path to loading directory.
        filename : str
            Filename to be used. File extension (.pkl) set by method

        Returns
        -------
        None.

        """
        
        if self.verbose:
            print(u'\u2500' * 75)
            print(' Loading from: {:s}/{:s} '.format(path, filename), end="", 
                  flush=True)
        
        with open(path + '/' + filename + '.pkl','rb') as f:
            datas = zlib.decompress(f.read(),0)
            loadDict = pickle.loads(datas,encoding="bytes")
        
        # Allocate fields from load dict
        self.isProcessed = loadDict['isProcessed']
        self.Losses = loadDict['Losses']
        self.boot = loadDict['boot']
        self.b = loadDict['b']
        self.bootInds = loadDict['bootInds']
        self.bootMean = loadDict['bootMean']
        self.tScore = loadDict['tScore']
        self.tBootDist = loadDict['tBootDist']
        self.pVals = loadDict['pVals']
        self.exclMods = loadDict['exclMods']
        self.stats = loadDict['stats']
        
        if self.verbose:
            print(' - Done')
            
    def addLosses(self,losses):
        """
        Add model losses to the object. Several cases are possible:
            - The mcs.losses attribute is empty: losses are directly saved to 
              the empty mcs.losses attribute.
            - The mcs.losses attribute already contains loss data. In this 
              case, the method attempts to append the new losses to the 
              existing ones (e.g. to update the existing MCS analysis with new 
              models). This requires the number of observations to be the same
              in 'mcs.losses' and 'losses', the method fails if this is not
              the case.

        Parameters
        ----------
        losses : ndarray
            2D ndarray containing model losses. Structure is:
                
                num obs x num models

        Returns
        -------
        None.

        """
        
        if self.verbose:
            print(u'\u2500' * 75)
            
        if losses.ndim != 2:
            print('Error: Losses must be a 2-dimensional Numpy array')
        elif self.Losses is None:       # No pre-existing losses
            self.Losses = losses
            if self.verbose:
                print(' Added losses: {:d} obs x {:d} models'.format(
                    self.Losses.shape[0], self.Losses.shape[1]))
        elif self.Losses.shape[0] != losses.shape[0]:
            print('Error: Existing & additional losses have different' +
                  ' observations:')
            print(' - existing:   {.d}'.format(self.Losses.shape[0]))
            print(' - additional: {.d}'.format(losses.shape[0]))
        else:
            self.Losses = np.concatenate((self.Losses,losses),axis = 1)
            
    def run(self, B=1000, b=10, bootstrap='stationary', algorithm='2-pass'):
        """
        Run the MCS analysis on the model losses
        
        This requires specifying two settings:
            - The bootstrap method. This is set using the 'B', 'b' and 
              'bootstrap' arguments. These settings are ignored when updating 
              an existing mcs object, as this will use the existing bootstrap 
              settings.
            - The MCS algorithm. Three options are available (details below)
            
        Note: 
            - The .run() method will generate the elimination sequence for the 
              models in the loss data and the associated elimination test 
              statistics and bootstrapped distribution, but does not calculate 
              p-values and the alpha-level confidence set. This is done 
              seperately using the .getMCS() method, as any update of the 
              collection will require recalculating the MCS.
            - Because the 1-pass MCS algorithm allows for updating of existing
              MCS runs, the same MCS object can be run multiple times. 
              Additional model losses can be added to a collection using the
              .addLosses() method (see function help). Calling the .run() 
              method on the larger collection of losses will only process the
              additional models using the 1-pass algorithm.
            - THe 'elimination' version of the algorithm corresponds to the 
              original Hansen, Lunde and Nason (2011) algorithm. This is mainly
              provided for comparability/benchmarking, or cases where the size 
              of the model collection is small enough (~50-100) that the better
              performance of the fast MCS is not needed. CAUTION: due to high 
              memory requirements, this option should not be used on large
              collections (>~ 500).

        Parameters
        ----------
        B : int, optional
            Number of bootstrap replications required. Ignored when updating an
            existing instance of the mcs object.
            The default is 1000.
        b : int, optional
            Bootstrap window. Ignored when updating an existing instance of the
            mcs object.
            The default is 10.
        bootstrap : string, optional
            Bootstrap algorithm used in the analysis. Two options available:
                - 'block': a block bootstrap
                - 'stationary': Politis & Romano (1994) stationary bootstrap
            The default is 'stationary'. Ignored when updating an existing 
            instance of the mcs object.
        algorithm : string, optional
            Desired algorithm required for the analysis. Three options 
            available:
                - '2-pass': 2-pass fast MCS algorithm
                - '1-pass': 1-pass fast MCS algorithm, heuristic but allows 
                  updating
                - 'elimination': Original elimination implementation of Hansen,
                  Lunde and Nason (2011)
            The default is '2-pass', unless extra losses have been appended, in
            which case it forces use of 1-pass to enable updating.

        Returns
        -------
        None.

        """
        if self.verbose:
            print(u'\u2500' * 75)
        
        # Only run if losses have been added.
        if self.Losses is None:
            print(' Error - No losses to process')
            print(" Use the '.addLosses()' method to add model losses first")
            return
        
        tStart = time.time()        
        obs, n = self.Losses.shape

        # If a processed MCS already exists, force update with one-pass
        if self.isProcessed:
            # Only run if more losses have been added
            if self.tScore.shape[0] == n:
                print(' MCS processing already up-to-date')
                print('  No of existing models:   {:d}'.format(n))
                return
            
            n0 = self.tScore.shape[0]        # Starting point
            nNew = n-n0                      # Number of additional models

            if self.verbose:
                print(' Updating MCS')
                print('  No of existing models:   {:d}'.format(n0))
                print('  No of additional models: {:d}'.format(nNew))
                
            # Update object field sizes to larger 
            self.tScore = np.concatenate(       # t-statistic vector
                                (self.tScore,     
                                 np.zeros(nNew)),
                                        axis = 1) 
            self.bootMean = np.concatenate(	    # Mean t-statistic vector
                                 (self.bootMean,
                                  np.zeros([B,nNew])),
                                        axis = 1)
            self.tBootDist = np.concatenate(	# Bootstrapped distr. matrix
                                 (self.tBootDist,
                                  np.zeros([B,nNew])),
                                        axis = 1)               
            modRankOld = np.copy(np.flip(self.exclMods)) # Model rankings
            algorithm = '1-pass'             # Force use of 1-pass
            
        else:       # else start a fresh MCS analysis
            # Get bootstrap indices
            rng = default_rng(seed = self.seed)
            self.boot = bootstrap
            if bootstrap == 'stationary':
                self.bootInds = stationaryBootstrap(rng, obs, B, b)
            elif bootstrap == 'block':
                self.bootInds = blockBootstrap(rng, obs, B, b)
            else:
                print("Error: Incorrect bootstrap selection." 
                      + "Options are 'block' or 'stationary'")
                return
            
            # Initialise object fields
            self.tScore = np.zeros(n)        # t-statistic vector
            self.bootMean = np.zeros([B,n])	 # Mean t-statistic vector
            self.tBootDist = np.zeros([B,n]) # Bootstrapped distribution matrix
            modRankOld = 0                   # Model rankings
            n0 = 0                           # Starting point
            nNew = n                         # Number of additional models
    
        if algorithm == '1-pass' or algorithm == '2-pass':
            if self.verbose:
                print(' Running {:s} MCS analysis'.format(algorithm))
            
            # Pass 1: Sequential updating of model rankings
            for i in range(n0,n):
                
                # -- Calculate current (i^th) row of the t-statistic table
                dLvec = np.mean(self.Losses[:,0:i] - 
                                self.Losses[:,i][:,None], 
                                axis = 0)
                
                self.bootMean[:,i] = np.mean(self.Losses[:,i][self.bootInds],
                                             axis=0).transpose()
                Lboot = self.bootMean[:,0:i] - self.bootMean[:,i][:,None] 
                
                dLboot = Lboot - dLvec[None,:]
                varLvec = np.mean(dLboot**2,axis = 0)
                tVec = np.append(dLvec / (varLvec**0.5), 0)
                
                # Find score of current model with respect to past models
                tCurr = max(-tVec)
                self.tScore[i] = tCurr

                # Update past models on the basis of score against new model
                # Ensures only worse models are updated
                update = np.where((tVec > self.tScore[0:i+1]) &
                                  (tVec >= tCurr))
                self.tScore[update] = tVec[update]
                modRank = np.argsort(self.tScore[0:i+1]).astype(int)
                
                # Update bootstrapped distribution in same pass
                if algorithm == '1-pass' and i > 0 : 
                    
                    # Find where current model enters the existing rankings
                    loc = np.where(modRank == i)[0][0]
                    modRankOldUpdated = np.insert(modRankOld,loc,i)
                    
                    # Find range of any ranking swaps (will be in worse models)
                    swaps = modRankOldUpdated !=  modRank
                    swapsRange = np.where(swaps == True)[0]
                    if swapsRange.size > 0:
                        swaps[swapsRange[0]:swapsRange[-1]] = True
                    
                    # Get bootstrapped distribution for current model
                    tBoot = dLboot / (varLvec**0.5)
                    if loc == 0:    # Current model is the best
                        tMax = np.zeros(B)
                    else:
                        tMax = np.max(np.abs(tBoot[:,modRank[0:loc]]),
                                      axis = 1)
                        self.tBootDist[:,i] = np.maximum(
                                            tMax,
                                            self.tBootDist[:,modRank[loc-1]]
                                            )
                    
                    # Update bootstrapped distr. for models worse than current
                    # - swaps are dealt with via the midpoint of bounds
                    for j in range(loc+1,i+1):
                        tMax = np.maximum(tMax, np.abs(tBoot[:,modRank[j]]))
                        if swaps[j]:
                            lBound = np.maximum(tMax, 
                                            self.tBootDist[:,modRank[j-1]])
                            uBound = np.maximum(lBound, 
                                            self.tBootDist[:,modRank[j]])
                            self.tBootDist[:,modRank[j]] = 0.5*(lBound+uBound)
                        else:
                            self.tBootDist[:,modRank[j]] = np.maximum(tMax, 
                                              self.tBootDist[:,modRank[j]])
                    # Memorise current ranking for next iteration
                    modRankOld = modRank
                
            # Once models rankings are processed, get the model exclusion order
            self.exclMods = np.flip(modRank)
                    
            # Pass 2: Sequential calculation of the bootstrapped distribution
            # This is error-free, at the cost of not being updateable
            if algorithm == '2-pass':
                for i in range(1,n):
                    modRange = modRank[0:i]
                    k = modRank[i]
                    
                    dLvec = np.mean(self.Losses[:,modRange] -
                                    self.Losses[:,k][:,None], 
                                  axis = 0)
                    Lboot = (self.bootMean[:,modRange] - 
                             self.bootMean[:,k][:,None])
                    dLboot = Lboot - dLvec[None,:]
                
                    varLvec = np.mean(dLboot**2,axis = 0)
                    tBoot = dLboot / (varLvec**0.5)
                
                    tMax = np.max(np.abs(tBoot), axis = 1)
                    self.tBootDist[:,k] = np.maximum(
                                            tMax,
                                            self.tBootDist[:,modRank[i-1]])
        
        elif algorithm == 'elimination':
            # Original elimination implementation, provided for performance 
            # comparison. Not meant for large-scale application
            
            # Preprocess main and bootstrapped t-statistics 
            dL = np.mean(
                    np.concatenate([self.Losses[:,None,:]]*n,axis = 1) - 
                    np.concatenate([self.Losses[:,:,None]]*n,axis = 2),
                    axis = 0)
            LbootMean = np.mean(self.Losses[self.bootInds.astype(int)],
                                axis=0).transpose()
            Lboot = (np.concatenate([LbootMean[None,:,:]]*n,axis = 0) - 
                     np.concatenate([LbootMean[:,None,:]]*n,axis = 1))
            
            dLboot = Lboot - np.concatenate([dL[:,:,None]]*B,axis = 2)
            varL = np.sum(dLboot**2,axis = -1)/B + np.identity(n)
            t = dL / (varL**0.5)
            tBoot = dLboot / (varL[:,:,None]**0.5)
            
            # Iterate original MCS procedure
            exclMods = []
            inclMods = list(range(0,n))
            self.tScore = np.zeros(n)
            self.pVals = np.zeros(n)
            self.tBootDist = np.zeros([B,n])
    
            for i in range(n-1):
            
                # Locate worst performing model, save its T-statistic
                colMaxT = np.max(
                            t.take(inclMods,axis=0).take(inclMods,axis=1), 
                            axis = 0)
                ind = np.argmax(colMaxT)
                self.tScore[i] = colMaxT[ind]
            
                # Bootstrapped distribution, update p-value for worst model
                tBootDistIter = np.max(
                                 np.max(
                                     tBoot[np.ix_(inclMods,inclMods)], 
                                     axis = 0),
                                 axis=0)    
                self.tBootDist[:,i] = tBootDistIter
                
                # Move worst model to the excluded set
                exclMods.append(inclMods.pop(ind))
            
            # Append last model. t-score = 0 & pval = 1 by construction
            exclMods.append(inclMods[0])
            self.exclMods = np.asarray(exclMods)
            
            # Rearrange t-scores and bootstrapped distribution in model order
            # - This is so P-value extraction is compatible with fast MCS
            sortInd = np.argsort(exclMods).astype(int)
            self.tScore = self.tScore[sortInd]
            self.tBootDist = self.tBootDist[:,sortInd]
            
        else:
            print("Error: Incorrect algorithm selection." 
                  + "Options are '1-pass','2-pass' or 'elimination'")
            return
        
        # Get diagnostic statistics - time and memory use
        mem = {}
        memTot = 0
        selfVarList = [(name, value) for name, value in vars(self).items()]
        localVarList = [(name, value) for name, value in locals().items()]
        
        for name, value in selfVarList + localVarList:
            if isinstance(value, np.ndarray):
                mem[name] = value.nbytes
            else:
                mem[name] = sys.getsizeof(value)
            memTot += mem[name]
        
        tEnd = time.time()
        self.stats.append({'run':len(self.stats)+1,
                           'models processed':nNew,
                           'method':algorithm,
                           'time':tEnd - tStart,
                           'memory use':memTot,
                           'memory alloc':mem})
        
        # Set processed flag to True
        self.isProcessed = True

        
    def getMCS(self, alpha = 0.05):
        """
        Retrieve lists of eliminated models and models included in the MCS
        The method will calculate the p-values from the elimination test 
        statistics and the bootstrapped test statistics, ensure they are 
        monotonically increasing and then separate the elimination order when
        p>=alpha.
        
        Note: the full p-values associated with the lists of excluded and 
        included models are not returned but the function, as they can directly
        be accessed from the .pVals attribute of the MCS object.

        Parameters
        ----------
        alpha : float, optional
            The significance level desired for the MCS. The default is 0.05.

        Returns
        -------
        inclOut : ndarray
            List of models included in the MCS and significance level alpha
        exclOut : ndarray
            List of models excluded from the MCS and significance level alpha

        """
        
        # Get bootstrapped P-values
        self.pVals = np.mean(self.tBootDist[:,self.exclMods] 
                        >= self.tScore[self.exclMods], axis = 0)
        
        # Smooth P-values so they are non-decreasing
        maxpval = self.pVals[0]
        for i in range(1,len(self.pVals)):
            if self.pVals[i] < maxpval:
                self.pVals[i] = maxpval
            else:
                maxpval = self.pVals[i]
        
        # Locate alpha threshold, split the model set
        mcs_cut = np.where(self.pVals >= alpha)[0]
        inclOut = np.asarray(self.exclMods)[mcs_cut]
        exclOut = np.delete(np.asarray(self.exclMods),mcs_cut)
        
        return inclOut, exclOut
