# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:11:54 2024

This script runs the MCS partition analysis on the multivariate ARCH forecast 
losses. This is parallelised over the 16 different combinations of bootstrap 
(Stationary/block), empirical sample (low/high volatility), volatility proxy 
(rv5/open-to-close) and forecast horizon (1/5 days).

@author: Sylvain Barde, University of Kent
"""

import sys
import os
import time
import pickle
import zlib

import numpy as np
import multiprocessing as mp

from fastMCS import mcs

def mcsPartition(inputs):
    
    runTimeStart = time.time()
    
    # Extract run settings
    (loadPath, logPath,savePath) = inputs[0]
    (sample, rvInd, btsrtp, horizon) = inputs[1]
    
    sample = int(sample)
    horizon = int(horizon)
    fileNameIn = 'sample_{:d}_proxy_{:s}'.format(sample,rvInd)
    fileNameOut = fileNameIn+'_horizon_{:d}_btsrtp_{:s}_mcs'.format(horizon, 
                                                                    btsrtp)
    
    # Format logging
    sys.stdout = open(logPath + '//' + fileNameOut + '.out', "w")
    
    # Load initial loss dict and extract relevant fields
    fil = open(loadPath + '/' + fileNameIn + '.pkl','rb')
    datas = zlib.decompress(fil.read(),0)
    fil.close()
    lossDict = pickle.loads(datas,encoding="bytes")
    
    losses = lossDict['losses'][:,:,horizon-1].squeeze()
    modList = np.asarray(lossDict['modList'])
    missing = list(lossDict['missingMods'])
    
    modPartition = np.zeros(4800)
    modPartition[missing] = np.nan
    counter = 0
    
    # Initialise display and header
    print(' Sample: {:d}'.format(sample))
    print(' Bootstrap: {:s}'.format(btsrtp))
    print(' RV proxy: {:s}'.format(rvInd))
    print(' Horizon: {:d}'.format(horizon))
    print(' ')
    print(' {:9s} | {:9s} | {:10s} | {:9s} '.format('Iteration',
                                                   'Num mods', 
                                                   'Time (sec)', 
                                                   'Mem (MB)' ))
    # Iterate MCS analysis until no models are left
    while len(modList) > 0:
        
        # Run MCS - not verbose, no saving
        mcsEst = mcs(seed = seed, verbose = False)
        mcsEst.addLosses(losses)
        mcsEst.run(B, b, bootstrap = btsrtp)
        
        # Identify MCS models, log iteration value in Partition
        inclInd, exclInd = mcsEst.getMCS(alpha = 0.1)
        inclMod = modList[inclInd]
        modPartition[inclMod] = counter
        
        # Display diagnostics
        print(' {:9d} | {:9d} | {:10.3f} | {:9.3f} '.format(
            counter,
            len(modList),
            mcsEst.stats[0]['time'],
            mcsEst.stats[0]['memory use']*10**-6))
        
        # Remove models from losses and list, increment counter
        losses = np.delete(losses,
                           inclInd, 
                           axis = 1)
        modList = np.delete(modList,
                            inclInd)
        counter += 1
    
    fil = open(savePath + '//' + fileNameOut + '.pkl','wb') 
    fil.write((pickle.dumps(modPartition, protocol=2)))
    fil.close()

    # Print completion time and return
    runTime = time.time() - runTimeStart
    print(u'\u2500' * 75, flush=True)
    print(' Setting complete - {:10.4f} secs.'.format(runTime), flush=True)
    print(u'\u2500' * 75, flush=True)
    sys.stdout.flush()
    
    return runTime

#-----------------------------------------------------------------------------
if __name__ == '__main__':
    loadPath = 'losses'
    logPath = "logs/partition_{:s}".format(
        time.strftime("%d-%b-%Y_%H-%M-%S",time.gmtime()))
    savePath = 'losses/partition'

    print('Saving logs to: {:s}'.format(logPath))
    os.makedirs(logPath,mode=0o777)

    # Bootstrap settings
    seed = 0            # RNG seed for bootstraps (ensures replicability)
    B = 1000            # Number of Bootstrap replications
    b = 10              # Size of bootstrap blocks
    
    if not os.path.exists(savePath):
        os.makedirs(savePath,mode=0o777)

    # Comparision settings and table labels
    settings = {'samples' : [0,1],
                'rvInds' : ['rv5','open_to_close'],
                'btsrtps' : ['block','stationary'],
                'horizons' : [1,5]}
    
    settingsList = []
    for key in settings.keys():
        settingsList.append(settings[key])
    mcsSettingsArray = np.array(
                np.meshgrid(*settingsList)).T.reshape(-1,len(settingsList))

    parallelRunSettings = []
    for setting in mcsSettingsArray:
        parallelRunSettings.append(([loadPath, logPath,savePath],
                                    setting))

    # -- Run parallel partition analysis
    num_tasks = len(parallelRunSettings)
    num_cores = num_tasks 
    
    t_start = time.time()
    
    # Initialise Display
    print(' ')
    print('+------------------------------------------------------+')
    print('|             Parallel MCS partition analysis          |')
    print('+------------------------------------------------------+')    
    print(' Number of cores: ' + str(num_cores) + \
        ' - Number of tasks: ' + str(num_tasks))
    print('+------------------------------------------------------+')
    
    # Create pool and run parallel job
    pool = mp.Pool(processes=num_cores)
    res = pool.map(mcsPartition,parallelRunSettings)
    
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
