# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 13:12:26 2023

This script is used to obtain the multivariate ARCH forecasts that form the
basis of the empirical application of the fast implementation of the Model 
Confidence Set algorithm (fastMCS)

Set the empirical sample (0 -> high or 1 -> low volatility) on line 345

NOTE: Due to the time required to estimate and forecast 4800 models, the 
script uses the multiprocessing package to parallelize the estimation and
forecasting over a set of 20 cores, with high memory usage (up to 20GB). Users 
should ensure they have access to the required compute resources before
running the script, or set a small set of parallel cores (line 395, variable 
'num_cores').

@author: Sylvain Barde, University of Kent
"""

import sys
import os
import time
import pickle
import zlib

import numpy as np
import pandas as pd
import multiprocessing as mp

from mArch import mArch
from copy import copy
from numpy.linalg import LinAlgError
from arch.univariate import (ARCH, GARCH, FIGARCH, EGARCH, HARCH,
                             MIDASHyperbolic, APARCH)

#-----------------------------------------------------------------------------
def archEstimation(inputs):
    """
    Main multivariate ARCH rolling window estimation/forecasting file.
    
    Note: The forecasts are not returned, instead they are directly saved to 
    the corresponding directories. Given the large size of the outputs, this
    minimise IO costs.

    Each batch will return an output dict containing 50 dicts, one per 
    univariate specification. The keys are given by the unique model ID.
    
    Each indvidual dict in 'outDict' has the following fields:
     'model': dict containing the full model specfiication estimated,
     'isEstimated':boolean,True if estimation sucessful
     'flags': 1D ndarray, flags initial condition issues for each 22-day window
         0 - Successful estimation with provided initial value
         1 - Successful estimation with built-in initial value
         2 - Unsuccessful estimation (results in isEstimated - False)
     'forecasts': dict, containing 5 forecasts, keyed using 'h.X' for the
         X-day ahead forecasts. Each forecast is a 3D ndarray structured as:
             numObs x numSeries x numSeries
     'time': dict, containing the breakdown of the total runtime:
         'setup': Time spent setting up estimation,
         'estimation': Time spent estimating the rolling windows
         'forecast': Time spent forecasting the rolling windows

    Parameters
    ----------
    inputs : list
        Inputs for the mARCH estimation/forecasting exercise.
         inputs[0] contains a list of paths for the exercise
             inputs[0][0]     # Path to empirical data
             inputs[0][1]     # Path to location for saving outputs
             inputs[0][2]     # Path to location for saving run logs

         inputs[1] contains a dict with multivariate ARCH settings. Fields are:
             - 'errors': Error distribution (Normal or Student)
             - 'multivar': Multivariate model (Naive, CCC, DCC or DCCA)
             - 'rvInd': Realised value measure (one of 12)
             - 'BaseID': Bade ID for the batch of 50 univariate ARCH 
                specification to be estimated.
         inputs[2] contains an index to the starting observation in the data

    Returns
    -------
    runTime : float
        The running time for the function.

    """
    
    runTimeStart = time.time()
    
    # -- Unpack inputs and set hard-coded constants
    split_obs = 1500    # Number of training observations
    w = 22              # Width of forecast window
    scale = 100         # Returns scaling parameter (for numerical stability)
    
    data_path = inputs[0][0]
    log_path = inputs[0][1]
    save_path = inputs[0][2]
    settings = inputs[1]
    start = inputs[2]    
    
    univarModParams = {'arch':{'p':[1,2]},
                       'garch':{'p':[1,2],'q':[1,2],'o':[0,1],'power':[1,2]},
                       'figarch':{'p':[0,1],'q':[0,1],'power':[1,2]},
                       'egarch':{'p':[1,2],'q':[1,2],'o':[0,1]},
                       'harch':{'lags':[[1,5],[1,5,44]]},
                       'midas':{'m':[5,22,44],'asym':[True,False]},
                       'aparch':{'p':[1,2],'q':[1,2],'o':[0,1]}}
    
    univarMods = {'arch':lambda **kwargs: ARCH(**kwargs),
                  'garch':lambda **kwargs: GARCH(**kwargs),
                  'figarch':lambda **kwargs: FIGARCH(**kwargs), 
                  'egarch':lambda **kwargs: EGARCH(**kwargs),
                  'harch':lambda **kwargs: HARCH(**kwargs),
                  'midas':lambda **kwargs: MIDASHyperbolic(**kwargs),
                  'aparch':lambda **kwargs: APARCH(**kwargs)}
    
    initVals = {
                'naive':np.empty(shape=(0)),
                'ccc':np.empty(shape=(0)),
                'dcc':np.asarray([0.01, 0.95]),
                'dcca':np.asarray([0.01, 0.95, 0.01]),
                'Normal':np.empty(shape=(0)),
                'Student':np.asarray([7])
                }
    
    # -- Generate universe of all possible arch models given the above
    archMods = []
    for mod in univarMods.keys():
        
        lst = []
        for key in univarModParams[mod].keys():
            lst.append(univarModParams[mod][key])
            
        if len(lst) == 1:
            paramArray = np.array(*lst, dtype=object).T.reshape(-1,len(lst))
        else:
            paramArray = np.array(np.meshgrid(*lst)).T.reshape(-1,len(lst))
    
        for params in paramArray:
            archMods.append({'func':univarMods[mod],
                             'type':mod,
                             'params':dict(zip(univarModParams[mod].keys(),
                                               params))})
        
    # -- Extract data for the returns and selected RV measure
    data = pd.read_csv(data_path, index_col=[0])
    
    indList = ['.SPX',
                '.IXIC',
                '.FCHI',
                '.FTSE',
                '.STOXX50E',
                '.AORD',
                '.HSI',
                '.N225',
                '.KS11',
                '.SSEC']

    rvInd = settings['rvInd']
    ind = indList.pop(0)
    mRt = pd.DataFrame(                             # daily returns (for sign)
            data[data.Symbol == ind]['open_to_close']
            ).rename(columns={"open_to_close": ind})
            

    mRv = pd.DataFrame(                             # volatility predictor
            data[data.Symbol == ind][rvInd]
            ).rename(columns={rvInd: ind})
    
    for ind in indList:
        mRt = mRt.merge(pd.DataFrame(
            data[data.Symbol == ind]['open_to_close']
            ).rename(columns={"open_to_close": ind}),
                        how='inner', 
                        left_index = True, 
                        right_index = True)

        mRv = mRv.merge(pd.DataFrame(
            data[data.Symbol == ind][rvInd]
            ).rename(columns={rvInd: ind}),
                        how='inner', 
                        left_index = True, 
                        right_index = True)
        
    # Scale, take root and assign sign to RV measure
    mRv = scale*np.sign(mRt)*mRv**0.5
    
    # -- Initialise log file, error file and display config settings
    file_name = 'batch_{:d}-{:d}'.format(int(settings['baseID']), 
                                        int(settings['baseID']+len(archMods)))
    sys.stdout = open(log_path + '//' + file_name + '.out', "w")
    sys.stderr = open(log_path + '//' + file_name + '_err.out', "w")
    
    print('Batch {:d}-{:d} initialised'.format(int(settings['baseID']), 
                                        int(settings['baseID']+len(archMods))),
          flush=True)
    print('Batch settings for arch models:', flush=True) 
    for item, value in settings.items():
        print('{}: {}'.format(item, value), flush=True)
    
    # -- Estimate for all arch model specifications
    outDict = {}
    for modBaseID, mod in enumerate(archMods):
        
        timeSetup = 0
        timeEst = 0
        timeForecast = 0
        flagVec = np.zeros(25)  # Set all to 0 initially (no estimation issues)
        modEstimated = True     # Set to true initially
        H_all = None            # Set to None initially
        forecastMethod = 'None' # Placeholder if all models fail to estimate
        modID = settings['baseID'] + modBaseID
        modelSpec = {**copy(mod), **copy(settings)}
        del modelSpec['func']
        del modelSpec['baseID']
        
        # Declare estimation start and iterate procedure over rolling windows
        print(u'\u2500' * 75)        
        print('Estimation for model {:d}'.format(int(modID)), flush=True)        
        for shift in range(0, 550, w):
            print(u'\u2500' * 75)
            print('Shift value {:d} started'.format(int(shift)), flush=True)
        
            tStart0 = time.time()
            mArchEst = mArch(mRv.iloc[start+shift : start+shift + split_obs+w])
            mArchEst.setArch(mod['func'](**mod['params']),
                             errors = settings['errors'], 
                             multivar = settings['multivar'])
            timeSetup += time.time() - tStart0
            
            #------------------------------------------------------------------
            # Estimate model for the window
            tStart1 = time.time()
            init_base = np.concatenate((initVals[settings['multivar']],
                                   initVals[settings['errors']]))   
            
            # Nested try/except structure to catch initialisation issues
            # Nesting as all errors are of the same type (linAlg, singular)
            try:
                # 1st try, use user-provided starting point
                mArchEst.fit(update_freq = 0, 
                             last_obs = split_obs,
                             init = init_base)
                if not mArchEst.checkBoundary():
                    raise LinAlgError('Matrix not positive definite')
                
            except BaseException as exp:
                print(type(exp))    # Print exception and details
                print(exp.args)           
                print('Switching to built-in initial condition')    
                
                try:
                    # 2nd try, toolbox default starting point, flag change if
                    # sucessful
                    mArchEst.fit(update_freq = 0, 
                                 last_obs = split_obs)
                    if not mArchEst.checkBoundary():
                        raise LinAlgError('Matrix not positive definite')
                    flagVec[int(shift/w)] = 1
                    
                except BaseException as exp:
                    # if both methods fail, give up and flag issue
                    print(type(exp))    # Print exception and details
                    print(exp.args)           
                    print('Cannot estimate model')

                    flagVec[int(shift/w)] = 2
                    modEstimated = False
                    sys.stdout.flush()
                    break
            
            sys.stdout.flush()
            timeEst += time.time() - tStart1
            
            #------------------------------------------------------------------
            # Generate forecasts for the window
            tStart2 = time.time()
            
            if shift == 0:
                # 1st window, try analytical, use simulation if not available
                # Sets forecast method for all other runs
                forecastMethod = 'analytic'

                try:
                    H_window = mArchEst.forecast(horizon=5, 
                                      start = split_obs, 
                                      method = forecastMethod)
                except:
                    forecastMethod = 'simulation'
                    H_window = mArchEst.forecast(horizon=5, 
                                      start = split_obs, 
                                      method = forecastMethod)
            else:
                # Just re-use what was set in 1st run
                H_window = mArchEst.forecast(horizon=5, 
                                      start = split_obs, 
                                      method = forecastMethod)
            
            # Save forecasts to forecast dict
            if shift == 0:
                H_all = copy(H_window)
            else:
                for key in H_window.keys():
                    H_all[key] = np.concatenate((H_all[key],
                                                 H_window[key]),
                                                axis = 0)
            
            timeForecast += time.time() - tStart2
        
            del mArchEst
            print('Shift value {:d} done, {:10.4f} secs'.format(
                int(shift), time.time() - tStart0
                ), flush=True)
            sys.stderr.flush()

        # Package forecasts and settings into the output dictionary
        modelSpec['forecastMethod'] = forecastMethod
        outDict[modID]={'model':modelSpec,
                        'isEstimated':modEstimated,
                        'flags':flagVec,
                        'forecasts':H_all,
                        'time':{'setup':timeSetup,
                                'estimation':timeEst,
                                'forecast':timeForecast}}

    # Save zipped output dictionary directly (not returned to save IO overhead)
    fil = open(save_path + '//' + file_name + '.pkl','wb') 
    fil.write(zlib.compress(pickle.dumps(outDict, protocol=2)))
    fil.close()     
        
    # Print completion time and return
    runTime = time.time() - runTimeStart
    print(u'\u2500' * 75, flush=True)
    print(' Batch complete - {:10.4f} secs.'.format(runTime), flush=True)
    print(u'\u2500' * 75, flush=True)
    sys.stdout.flush()
    
    return runTime

#-----------------------------------------------------------------------------
if __name__ == '__main__':
    mp.set_start_method('spawn')     # for 3.9, ensure compatibility with code.
    
    startList = [200, 1250]     # List of start points - both OK
    sample = 1                  # Sample 0 is the high-volatility test sample
                                # Sample 1 is the low-volatility test sample
                 
    data_path = 'data/oxfordmanrealizedvolatilityindices.csv'
    
    # Create logging/saving directory
    runName = "mArch_sample_{:d}_Run_{:s}".format(
        sample,
        time.strftime("%d-%b-%Y_%H-%M-%S",time.gmtime()))
    log_path = "logs//" + runName
    print('Saving logs to: {:s}'.format(log_path))
    os.makedirs(log_path,mode=0o777)
    save_path = "forecasts//" + runName
    os.makedirs(save_path,mode=0o777)
    
    rvMeasures = ['rv5',
                  'rv5_ss',
                  'rv10',
                  'rv10_ss',
                  'bv',
                  'bv_ss',
                  'rsv',
                  'rsv_ss',
                  'rk_twoscale',
                  'rk_parzen',
                  'rk_th2',
                  'medrv']
        
    # -- Generate all possible multivariate estimation settings
    start = startList[sample]
    errorLst = ['Normal','Student']
    multivarLst = ['naive','ccc','dcc','dcca']
    lst = [errorLst,multivarLst,rvMeasures]
    settingsArray = np.array(np.meshgrid(*lst)).T.reshape(-1,len(lst))
    multivarSettings = []
    baseID = 0
    for setting in settingsArray:
        settingsDict = dict(zip(['errors','multivar','rvInd'], setting))
        settingsDict['baseID'] = baseID
        multivarSettings.append(([data_path, log_path,save_path],
                                 settingsDict,
                                 start))
        baseID += 50
        
    # This results in 2 x 4 x 12 = 96 possible batch settings
    # Each setting will use 50 univariate volatility models, resulting in
    # 96 x 50 = 4800 forecast models, each indentified with a unique ID
    
    # -- Run parallel estimation and forecast
    num_tasks = len(multivarSettings)
    num_cores = 20 # Core count is throttled to preserve RAM-per-CPU.
    
    t_start = time.time()
           
    # Initialise Display
    print(' ')
    print('+------------------------------------------------------+')
    print('|         Parallel mARCH estimation / forecasting      |')
    print('+------------------------------------------------------+')    
    print(' Number of cores: ' + str(num_cores) + \
        ' - Number of tasks: ' + str(num_tasks))
    print('+------------------------------------------------------+')
    
    # Create pool and run parallel job
    pool = mp.Pool(processes=num_cores)
    res = pool.map(archEstimation,multivarSettings)
    
    # Close pool when job is done
    pool.close()
    
    # Extract timers from return (only timers are returned)
    sum_time = 0
    for i in range(num_tasks):
        res_i = res[i]
        sum_time = sum_time + res_i

    # Print timer/speedup diagnostics
    timer_1 = time.time() - t_start
    print('+------------------------------------------------------+')
    print(' Total running time:     {:10.4f} secs.'.format(timer_1))
    print(' Sum of iteration times: {:10.4f} secs.'.format(sum_time))
    print(' Mean iteration time:    {:10.4f} secs.'.format(
            sum_time/num_tasks))
    print(' Speed up:               {:10.4f}'.format(sum_time/timer_1))
    print('+------------------------------------------------------+')
