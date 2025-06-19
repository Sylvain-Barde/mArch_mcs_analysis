# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 10:19:39 2023

This script produces the outputs for the fastMCS benchmarking exercise.
(figure 1, tables 1 and 7 in the paper).

@author: Sylvain Barde, University of Kent
"""
import os
import numpy as np
import zlib
import pickle
from matplotlib import pyplot as plt
from statsmodels.iolib.table import SimpleTable
from scipy.stats import mode

save = True     # Set to True to save output
color = True     # Set to True to generate color figures (B & W otherwise)
num_mods = 4482  # Use for projection in conclusion (from mArch)

numMcIter = 200
batchSize = 20

# Iterate over sample size runs (30 and 250)
for N in [30,250]:

    # Setup latex output and load/save folders
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"]})
    
    loadPath = 'montecarlo/benchmarking_results_N_{:d}'.format(N)
    savePath = 'montecarlo/outputs'
    if save is True:
        if not os.path.exists(savePath):
            os.makedirs(savePath,mode=0o777)
    
    # Load all seed runs into the same dict for processing
    mcDict = {}
    for baseSeed in range(0, numMcIter, batchSize):
    
        fileName = '/baseSeed_{:d}.pkl'.format(baseSeed)
    
        fil = open(loadPath + fileName,'rb')
        datas = zlib.decompress(fil.read(),0)
        fil.close()   
        mcDict.update(pickle.loads(datas,encoding="bytes"))
        
    # Prea-allocate result arrays
    numMcIter = len(mcDict)
    numMods = 32
    m = np.arange(2,4+2/(numMods-1),2/(numMods-1)) 
    mods = np.round(10**m)
    modsShort = mods[np.where(mods < 1000)]
    numElimination = np.sum(mods < 1000) # Number of models below 1000
    
    t0 = np.zeros((numElimination,numMcIter))
    t1 = np.zeros((numMods,numMcIter))
    t2 = np.zeros((numMods,numMcIter))
    
    mem0 = np.zeros((numElimination,numMcIter))
    mem1 = np.zeros((numMods,numMcIter))
    mem2 = np.zeros((numMods,numMcIter))
    
    checkRank2 = np.zeros((numElimination,numMcIter), dtype=bool)
    checkRank1 = np.zeros((numMods,numMcIter), dtype=bool)
    
    checkMcs2 = np.zeros((numElimination,numMcIter), dtype=bool)
    checkMcs1 = np.zeros((numMods,numMcIter), dtype=bool)
    numMCSdiff2 = np.zeros((numElimination,numMcIter))
    numMCSdiff1 = np.zeros((numMods,numMcIter))
    
    checkTScores2 = np.zeros((numElimination,numMcIter), dtype=bool)
    checkTScores1 = np.zeros((numMods,numMcIter), dtype=bool)
    numTScoreDiffs2 = np.zeros((numElimination,numMcIter))
    meanTScoreDiffs2 = np.zeros((numElimination,numMcIter))
    
    checkPvals2 = np.zeros((numElimination,numMcIter), dtype=bool)
    checkPvals1 = np.zeros((numMods,numMcIter), dtype=bool)
    numPvalDiffs2 = np.zeros((numElimination,numMcIter))
    meanPvalDiffs2 = np.zeros((numElimination,numMcIter))
    numPvalDiffs1 = np.zeros((numMods,numMcIter))
    meanPvalDiffs1 = np.zeros((numMods,numMcIter))
    
    # Process Monte Carlo dict into the results arrays
    for i in range(numMcIter):
        iterDict = mcDict[i]
        t0[:,i] = iterDict['t0']
        t1[:,i] = iterDict['t1']
        t2[:,i] = iterDict['t2']
        
        mem0[:,i] = iterDict['mem0']
        mem1[:,i] = iterDict['mem1']
        mem2[:,i] = iterDict['mem2']
        
        checkRank2[:,i] = iterDict['checkRank2']
        checkRank1[:,i] = iterDict['checkRank1']
        
        checkMcs2[:,i] = iterDict['checkMcs2']
        checkMcs1[:,i] = iterDict['checkMcs1']
        numMCSdiff2[:,i] = iterDict['numMCSdiff2']
        numMCSdiff1[:,i] = iterDict['numMCSdiff1']
        
        checkTScores2[:,i] = iterDict['checkTScores2']
        checkTScores1[:,i] = iterDict['checkTScores1']
        numTScoreDiffs2[:,i] = iterDict['numTScoreDiffs2']
        meanTScoreDiffs2[:,i] = iterDict['meanTScoreDiffs2']
        
        checkPvals2[:,i] = iterDict['checkPvals2']
        checkPvals1[:,i] = iterDict['checkPvals1']
        numPvalDiffs2[:,i] = iterDict['numPvalDiffs2']
        meanPvalDiffs2[:,i] = iterDict['meanPvalDiffs2']
        numPvalDiffs1[:,i] = iterDict['numPvalDiffs1']
        meanPvalDiffs1[:,i] = iterDict['meanPvalDiffs1']
        
    # Perform OLS regressions to get polynomial scaling parameters
    X_short_2 = np.concatenate((np.ones((numElimination,1)),
                        modsShort[:,None],
                        modsShort[:,None]**2),
                       axis = 1)
    
    X_short_3 = np.concatenate((np.ones((numElimination,1)),
                        modsShort[:,None],
                        modsShort[:,None]**2,
                        modsShort[:,None]**3),
                       axis = 1)
    
    X_long_1 = np.concatenate((np.ones((numMods,1)),
                        mods[:,None]),
                       axis = 1)
    
    X_long_2 = np.concatenate((np.ones((numMods,1)),
                        mods[:,None],
                        mods[:,None]**2),
                       axis = 1)
         
    X_long_3 = np.concatenate((np.ones((numMods,1)),
                        mods[:,None],
                        mods[:,None]**2,
                        mods[:,None]**3),
                        axis = 1)
    
    b_t0 = (np.linalg.inv(X_short_3.transpose() @ X_short_3) 
            @ (X_short_3.transpose() @ np.mean(t0,axis = 1)))
    b_t2 = (np.linalg.inv(X_long_3.transpose() @ X_long_3)
            @ (X_long_3.transpose() @ np.mean(t2,axis = 1)))
    b_mem0 = (np.linalg.inv(X_short_2.transpose() @ X_short_2)
              @ (X_short_2.transpose() @ np.mean(mem0,axis = 1)))
    b_mem1 = (np.linalg.inv(X_long_1.transpose() @ X_long_1)
               @ (X_long_1.transpose() @ np.mean(mem1,axis = 1)))
    
    
    # Calculate scaling plots and their position 
    y_pad = 0.10

    scale_t0 = np.log10(mods**3)
    scale_t1 = np.log10(mods**2)
    scale_mem0 = np.log10(mods**2)
    scale_mem1 = np.log10(mods)
    
    diff_t0 = np.min(
        np.log10(np.mean(t0, axis = 1)) - scale_t0[0:numElimination])
    diff_t1 = np.max(
        np.log10(np.mean(t2, axis = 1)) - scale_t1)
    diff_mem0 = np.min(
        np.log10(np.mean(mem0, axis = 1)) - scale_mem0[0:numElimination])
    diff_mem1 = np.min(
        np.log10(np.mean(mem1, axis = 1)) - scale_mem1)
    
    scale_t0 -= y_pad - diff_t0
    scale_t1 += y_pad + diff_t1
    scale_mem0 -= y_pad - diff_mem0
    scale_mem1 -= y_pad - diff_mem1
    
    #--------------------------------------------------------------------------
    # Generate Figures
    logMods = np.log10(mods)
    logModsShort = np.log10(modsShort)
    fontSize = 40
    y_max = 6.5
    y_min = -1
    xlim_left = min(logMods)
    xlim_right = max(logMods)+0.1
    if color:
        colorVec = ['b','r']
        colorSuffix = '_col'
    else:
        colorVec = ['k','k']    
        colorSuffix = '_bw'
    
    # Time scaling plot
    fig = plt.figure(figsize=(16,12))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(logModsShort,np.log10(np.mean(t0,axis = 1)),
            linestyle = 'solid', 
            linewidth=1.5, color = colorVec[0], alpha=1, 
            label = r'Elimination')
    ax.plot(logMods,np.log10(np.mean(t2,axis = 1)),
            linestyle = 'dashed', 
            linewidth=1.5, color = colorVec[1], alpha=1, 
            label = r'Two-pass')
    ax.plot(logMods,np.log10(np.mean(t1,axis = 1)),
            linestyle = 'dotted', 
            linewidth=1.5, color = colorVec[1], alpha=1, 
            label = r'One-pass')
    ax.plot(logMods,scale_t0,
            linestyle = 'dashdot', 
            linewidth=1.5, color = 'k', alpha=1, 
            label = r'$\propto M^3$')
    ax.plot(logMods,scale_t1,
            linestyle = (0, (3, 5, 1, 5)), # 'dashdotted' style
            linewidth=1.5, color = 'k', alpha=1, 
            label = r'$\propto M^2$')
    ax.set_xlabel(r'$\log_{10}(M)$', 
                  fontdict = {'fontsize': fontSize})
    ax.set_ylabel(r'$\log_{10}({\rm sec})$',
                  fontdict = {'fontsize': fontSize})
    ax.legend(loc='upper left', frameon=False, prop={'size':fontSize})
    
    ax.set_ylim(top = y_max, bottom = y_min)
    ax.set_xlim(left = xlim_left,right = xlim_right)
    ax.plot(xlim_right, y_min, ">k", ms=15, clip_on=False)
    ax.plot(xlim_left, y_max, "^k", ms=15, clip_on=False)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params(axis='x', pad=15, labelsize=fontSize)
    ax.tick_params(axis='y', pad=15, labelsize=fontSize)
    if save is True:
        plt.savefig(savePath + 
                    "/benchmark_time_N_{:d}{:s}.pdf".format(N, colorSuffix),
                    format = 'pdf', bbox_inches = 'tight')
    
    # Memory scaling plot
    fig = plt.figure(figsize=(16,12))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(logModsShort,np.log10(np.mean(mem0,axis = 1)),
            linestyle = 'solid', 
            linewidth=1.5, color = colorVec[0], alpha=1, 
            label = r'Elimination')
    ax.plot(logMods,np.log10(np.mean(mem2,axis = 1)),
            linestyle = 'dashed', 
            linewidth=1.5, color = colorVec[1], alpha=1, 
            label = r'Two-pass')
    ax.plot(logMods,scale_mem0,
            linestyle = 'dashdot', 
            linewidth=1.5, color = 'k', alpha=1, 
            label = r'$\propto M^2$')
    ax.plot(logMods,scale_mem1,
            linestyle = (0, (3, 5, 1, 5)), # 'dashdotted' style
            linewidth=1.5, color = 'k', alpha=1, 
            label = r'$\propto M$')
    ax.set_xlabel(r'$\log_{10}(M)$', 
                  fontdict = {'fontsize': fontSize})
    ax.set_ylabel(r'$\log_{10}({\rm MB})$',
                  fontdict = {'fontsize': fontSize})
    ax.legend(loc='upper left', frameon=False, prop={'size':fontSize})
    
    ax.set_ylim(top = y_max, bottom = y_min)
    ax.set_xlim(left = xlim_left,right = xlim_right)
    ax.plot(xlim_right, y_min, ">k", ms=15, clip_on=False)
    ax.plot(xlim_left, y_max, "^k", ms=15, clip_on=False)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params(axis='x', pad=15, labelsize=fontSize)
    ax.tick_params(axis='y', pad=15, labelsize=fontSize)
    if save is True:
        plt.savefig(savePath + 
                    "/benchmark_mem_N_{:d}{:s}.pdf".format(N, colorSuffix),
                    format = 'pdf', bbox_inches = 'tight')
    #--------------------------------------------------------------------------
    # Generate Tables
    
    # -- Calculate diagnostics table entries
    # Count differences in MCS, carry out diagnostics
    mcsDiff_0vs2 = 100*np.sum(checkMcs2 == False)/len(checkMcs2.flatten())
    mcsDiff_1vs2 = 100*np.sum(checkMcs1 == False)/len(checkMcs1.flatten())
    
    mcsDiffAbsSize_0vs2 = np.mean(np.abs(
                            numMCSdiff2[np.where(checkMcs2 == False)]))
    mcsMedianDiffAbsSize_0vs2 = np.median(np.abs(
                            numMCSdiff2[np.where(checkMcs2 == False)]))
    numMCSRelDiff2 = numMCSdiff2 / mods[0:numElimination,None]
    mscDiffRelSize_0vs2 = 100*np.mean(np.abs(
                            numMCSRelDiff2[np.where(checkMcs2 == False)]))
    mscMedianDiffRelSize_0vs2 = 100*np.median(np.abs(
                            numMCSRelDiff2[np.where(checkMcs2 == False)]))
    
    mcsDiffAbsSize_1vs2 = np.mean(np.abs(
                            numMCSdiff1[np.where(checkMcs1 == False)]))
    mcsMedianDiffAbsSize_1vs2 = np.median(np.abs(
                            numMCSdiff1[np.where(checkMcs1 == False)]))
    numMCSRelDiff1 = numMCSdiff1 / mods[:,None]
    mscDiffRelSize_1vs2 = 100*np.mean(np.abs(
                            numMCSRelDiff1[np.where(checkMcs1 == False)]))
    mscMedianDiffRelSize_1vs2 = 100*np.median(np.abs(
                            numMCSRelDiff1[np.where(checkMcs1 == False)]))
    mcsMedianDiffAbsSize_1vs2 = np.median(np.abs(
                            numMCSdiff1[np.where(checkMcs1 == False)]))
    
    mcsModeDiffAbsSize_1vs2 = mode(np.abs(
                            numMCSdiff1[np.where(checkMcs1 == False)]),
                                        keepdims = False)
    
    # Count differenes in Model rankings and T-scores
    rankDiff_0vs2 = 100*np.sum(checkRank2 == False)/len(checkRank2.flatten())
    rankDiff_1vs2 = 100*np.sum(checkRank1 == False)/len(checkRank1.flatten())
    
    tScoreDiff_0vs2 = 100*np.sum(
                        checkTScores2 == False)/len(checkTScores2.flatten())
    tScoreDiff_1vs2 = 100*np.sum(
                        checkTScores1 == False)/len(checkTScores1.flatten())
    
    # Count differenes in P-values, carry out diagnostics
    pvalsDiff_0vs_2 = 100*np.sum(
                        checkPvals2 == False)/len(checkPvals2.flatten())
    pvalsDiffAbsNum_0vs2 = np.mean(np.abs(
                            numPvalDiffs2[np.where(checkPvals2 == False)]))
    numPvalRefDiffs2 = numPvalDiffs2 / mods[0:numElimination,None]
    pvalsDiffRelNum_0vs2 = 100*np.mean(np.abs(
                            numPvalRefDiffs2[np.where(checkPvals2 == False)]))
    pvalsDiffAbsSize_0vs2 = 100*np.mean(np.abs(
                            meanPvalDiffs2[np.where(checkPvals2 == False)]))
    
    pvalsDiff_1vs_2 = 100*np.sum(
                        checkPvals1 == False)/len(checkPvals1.flatten())
    pvalsDiffAbsNum_1vs2 = np.mean(np.abs(
                            numPvalDiffs1[np.where(checkPvals1 == False)]))
    numPvalRefDiffs1 = numPvalDiffs1 / mods[:,None]
    pvalsDiffRelNum_1vs2 = 100*np.mean(np.abs(
                            numPvalRefDiffs1[np.where(checkPvals1 == False)]))
    pvalsDiffAbsSize_1vs2 = np.mean(np.abs(
                            meanPvalDiffs1[np.where(checkPvals1 == False)]))
    
    # Foramt and print diagnostic table
    labels = [r'\T\B Number of replications compared',
              r'\T\B \quad \emph{MCS size diagnostics}',
              r'\T % Replications with MCS size differences',
              r'Mean absolute difference in MCS size',
              r'\qquad % relative to collection size',
              r'Median absolute difference in MCS size',
              r'\B \qquad % relative to collection size',
              r'\T\B \quad \emph{Model ranking diagnostics}',
              r'\T % Replications with ranking differences',
              r'\B % Replications with t-score differences',
              r'\T\B \quad \emph{P-value diagnostics}',
              r'\T % Replications with P-value differences',
              r'Mean % share of models affected',
              r'\B Mean absolute difference in affected P-values',]
    
    header = ['vs. Elim.','vs. One-pass']
    
    values = np.asarray([[numElimination*numMcIter, numMods*numMcIter],
                        ['', ''],
                        [mcsDiff_0vs2, mcsDiff_1vs2],
                        [mcsDiffAbsSize_0vs2, mcsDiffAbsSize_1vs2],
                        [mscDiffRelSize_0vs2, mscDiffRelSize_1vs2],
                        [mcsMedianDiffAbsSize_0vs2, mcsMedianDiffAbsSize_1vs2],
                        [mscMedianDiffRelSize_0vs2, mscMedianDiffRelSize_1vs2],                    
                        ['', ''],
                        [rankDiff_0vs2, rankDiff_1vs2 ],
                        [tScoreDiff_0vs2, tScoreDiff_1vs2],
                        ['', ''],
                        [pvalsDiff_0vs_2, pvalsDiff_1vs_2],
                        [pvalsDiffRelNum_0vs2, pvalsDiffRelNum_1vs2],
                        [pvalsDiffAbsSize_0vs2, pvalsDiffAbsSize_1vs2]], 
                        dtype = str)
    
    valuesFormatted = []
    for row in values:
        rowFormatted = []
        for cellValue in row:
    
            if len(cellValue) == 0:
                cellValueFormatted = '{:s}'.format(cellValue)
            else:
                cellValueNum = cellValue.astype(float)
                if cellValueNum.is_integer():
                    cellValueFormatted = '{:d}'.format(int(cellValueNum))
                elif np.isnan(cellValueNum):
                    cellValueFormatted = '-'
                else:
                    cellValueFormatted = '{:3.3f}'.format(cellValueNum)            
    
            rowFormatted.append(cellValueFormatted)
        valuesFormatted.append(rowFormatted)
    
    table = SimpleTable(
            valuesFormatted,
            stubs=labels,
            headers=header,
            title=('Monte Carlo comparison of Two-pass fast MCS algorithm '+
                   'performance, N={:d}'.format(N)),
        )
    
    print(table)
    modeStr = (r'Note: the mode of the absolute difference in MCS size is' + 
               ' {:d}, occurring in {:3.2f}\% of cases'.format(
                   int(mcsModeDiffAbsSize_1vs2[0]),
                   100*mcsModeDiffAbsSize_1vs2[1]/np.sum(checkMcs1 == False)))
    print(modeStr)
    if save is True:
        with open(savePath+'/benchmark_diagnostics_N_{:d}.tex'.format(N),#
                  'w') as f:
            f.write(table.as_latex_tabular(header=r'%s',stub=r'%s'))
            f.write(modeStr)
    
    # -- Scaling performance table
    # Calculate scaling time/memory costs for table
    scalingValues = np.zeros([6,4])
    scalingLabels = []
    for i,M in enumerate([500, 1000, 2000, 5000, 10000, num_mods]):
        
        scalingLabels.append('$M = {:d}$'.format(M))
        scalingValues[i,0] = b_t0[0] + b_t0[1]*M + b_t0[2]*M**2 + b_t0[3]*M**3
        scalingValues[i,1] = b_mem0[0] + b_mem0[1]*M + b_mem0[2]*M**2
        scalingValues[i,2] = b_t2[0] + b_t2[1]*M + b_t2[2]*M**2 + b_t2[3]*M**3
        scalingValues[i,3] = b_mem1[0] + b_mem1[1]*M
    
    
    # Format and print table
    scalingValuesFormatted = []
    for row in scalingValues:
        rowFormatted = []
        for cellValue in row:
            rowFormatted.append('{:3.0f}'.format(cellValue))
            
        scalingValuesFormatted.append(rowFormatted)
    
    
    scalingHeader = ['Time (sec)', 'Memory (MB)','Time (sec)', 'Memory (MB)']
    scalingTable = SimpleTable(
            scalingValuesFormatted,
            stubs=scalingLabels,
            headers=scalingHeader,
            title='Scaling comparison of MCS algorithms',
        )
    
    print(scalingTable)
    if save is True:
        with open(savePath+'/benchmark_scaling_N_{:d}.tex'.format(N),'w') as f:
            f.write(scalingTable.as_latex_tabular(header=r'%s',stub=r'%s'))
