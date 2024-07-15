# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 06:52:12 2023

This script produces the outputs for the multivariate ARCH MCS comparison 
exercise. (tables 1, 2 and 3 in the paper).

NOTE: because these tables summarize both the MCS composition and the breakdown
of the partition, users need to run both 'parallel_mArch_main.py' and 
'parallel_mArch_mcs_partition.py' prior to running this script, so that both
sets of results are available.

@author: Sylvain Barde, University of Kent
"""

import numpy as np
import pandas as pd
import pickle
import zlib
import os
from fastMCS import mcs

from statsmodels.iolib.table import SimpleTable
#------------------------------------------------------------------------------
def format_int_table(tab, colorTab = None, maxIntLen = None):
    
    # Check if color tab is provided and matches
    if colorTab is None:
        colorTab = np.empty_like(tab.astype(float))
        colorTab[:] = np.nan
    elif ~np.all(colorTab.shape == tab.shape):
        print('Table formatting mismatch')
        return
    
    if maxIntLen is None:
        maxIntLen = np.ceil(
                        np.log10(
                            max(tab.flatten())
                                 )
                            ).astype(int)
        
    # Format table, including colored background if provided
    formattedTable = []
    for row, colorRow in zip(tab, colorTab):
        formattedRow = []
        for cell, colorCell in zip(row, colorRow):
            if np.isnan(colorCell):
                formattedCell = ''
            else:
                formattedCell = "\\C{{{:2d}}}".format(colorCell.astype(int))
            
            if cell < 0:
                formattedCell += "{:s}".format(' ')
            elif cell == 0:
                formattedCell += ("{:" + str(maxIntLen) + "s}- ").format(' ')
            else:
                formattedCell += ("{:" + str(maxIntLen) + "d} ").format(cell)
            formattedRow.append(formattedCell)
        formattedTable.append(formattedRow)
            
    return formattedTable
#------------------------------------------------------------------------------

loadPathMCS = 'losses'
loadPathPartition = 'losses/partition'
savePath = 'outputs'
save = False         # Flag: Set to 'True' to save outputs

# Bootstrap settings
seed = 0            # RNG seed for bootstraps (ensures replicability)
B = 1000            # Number of Bootstrap replications
b = 10              # Size of bootstrap blocks

if save is True:
    if not os.path.exists(savePath):
        os.makedirs(savePath,mode=0o777)

# Comparision settings and table labels
settings = {'samples' : [0,1],
            'rvInds' : ['rv5','open_to_close'],
            'btsrtps' : ['block','stationary'],
            'horizons' : [1,5]}

univarModsLabel = {'arch' : 'AR',
                  'garch':  'GA',
                  'figarch':'FI', 
                  'egarch': 'EG',
                  'harch':  'HA',
                  'midas':  'MI',
                  'aparch': 'AP'}

mArchLabels = {'naive':'Naïve',
               'ccc':'CCC',
               'dcc':'DCC', 
               'dcca':'DCCA'}

rvLabels = {'bv':'bv',
            'bv_ss':'bv_ss',
            'medrv':'medrv',
            'rk_parzen':'rk_pa',
            'rk_th2':'rk_th2',   
            'rk_twoscale':'rk_ts',
            'rsv':'rsv',
            'rsv_ss':'rsv_ss',
            'rv10':'rv10',
            'rv10_ss':'rv10_ss',
            'rv5':'rv5',
            'rv5_ss':'rv5_ss'}

univarInds = []

settingsList = []
for key in settings.keys():
    settingsList.append(settings[key])
mcsSettingsArray = np.array(
            np.meshgrid(*settingsList)).T.reshape(-1,len(settingsList))

# mcsSettingsArray = mcsSettingsArray[0][None,:]
mcsSize = []
mcsInclInds = []
mcsInclMods = []
t = []
mem = []
mods = []
tableList = []
fullTableList = []
tablePartList = []
fullTablePartList = []
numPartitionsList = []
for (sample, rvInd, btsrtp, horizon) in mcsSettingsArray:
    
    sample = int(sample)
    horizon = int(horizon)
    fileNameIn = 'sample_{:d}_proxy_{:s}'.format(sample,rvInd)
    fileNameOut = fileNameIn + '_horizon_{:d}_btsrtp_{:s}_mcs'.format(horizon, 
                                                                      btsrtp)
    
    fil = open(loadPathMCS + '/' + fileNameIn + '.pkl','rb')
    datas = zlib.decompress(fil.read(),0)
    fil.close()
    lossDict = pickle.loads(datas,encoding="bytes")
    univarType = lossDict['univarType']
    
    if os.path.isfile(loadPathMCS + '/' + fileNameOut + '.pkl'):
        print('MCS analysis already ran, loading results')
        mcsEst = mcs()
        mcsEst.load(loadPathMCS, fileNameOut)
        
    else:
        print('No pre-existing MCS analysis')
        losses = lossDict['losses'][:,:,horizon-1].squeeze()
        mcsEst = mcs(seed = seed)
        mcsEst.addLosses(losses)
        mcsEst.run(B, b, bootstrap = btsrtp)
        mcsEst.save(loadPathMCS, fileNameOut)
    
    modList = np.asarray(lossDict['modList'])
    inclInd, exclInd = mcsEst.getMCS(alpha = 0.1)
    inclMod = modList[inclInd]
    
    mcsSize.append(len(inclInd))
    mcsInclInds.append(inclInd)
    mcsInclMods.append(inclMod)
    
    # Pre-allocate empty tables
    modelSpecs = lossDict['modelSpecs']
    rows = np.unique(modelSpecs[:,3], return_counts=False)
    cols = np.unique(modelSpecs[:,2], return_counts=False)
    dists = np.unique(modelSpecs[:,1], return_counts=False)
    
    rowLabels = [rvLabels[key] for key in rvLabels.keys()]
    colLabels = [mArchLabels[key] for key in mArchLabels.keys()]
    univarLabels = ['-'] + [univarModsLabel[key] for key in univarModsLabel.keys()]
    
    # Summary table (top-level, only multivar & RV, in paper)
    table = pd.DataFrame(data=np.zeros((len(rows),len(cols))), 
                         index=rowLabels, 
                         columns=colLabels)
    
    # Full table (detailed, all 4 dimensions, for appendix)                            
    rowList = [rowLabels , dists]
    rowArray = [array for array in 
                np.array(np.meshgrid(*rowList)).T.reshape(-1,len(rowList)).T]
    
    colList = [colLabels, univarLabels]
    colArray = [array for array in 
                np.array(np.meshgrid(*colList)).T.reshape(-1,len(colList)).T]
    
    fullTable = pd.DataFrame(np.zeros((len(rows)*len(dists),
                                      len(cols)*(len(univarModsLabel)+1))),
                            index = rowArray,
                            columns = colArray)

    # Populate from each model in the included set
    for mod in inclMod:
        batch = int(np.floor(mod/50))
        
        rowInd = rvLabels[modelSpecs[batch,3]]
        colInd = mArchLabels[modelSpecs[batch,2]]
        distInd = modelSpecs[batch,1]
        archInd = univarModsLabel[univarType[mod]]
        
        table.loc[rowInd][colInd] += 1
        fullTable.loc[(rowInd,distInd),(colInd, archInd)] += 1
    
    # Set divider columns to negative integer for formatting
    for colInd in colLabels:
        fullTable.loc[:,(colInd, '-')] -= 1
    
    # Load Partition and generate shaded color tables
    if os.path.isfile(loadPathPartition + '/' + fileNameOut + '.pkl'):
        print('Partition analysis exists, loading results')
        fil = open(loadPathPartition + '/' + fileNameOut + '.pkl','rb')
        modPartition = pickle.loads(fil.read(),encoding="bytes")
        fil.close()
        
        tablePart = pd.DataFrame(data=np.zeros((len(rows),len(cols))), 
                                 index=rowLabels, 
                                 columns=colLabels)
        fullTablePart = pd.DataFrame(np.zeros((len(rows)*len(dists),
                                               len(cols)*(len(univarModsLabel)+1))),
                                     index = rowArray,
                                     columns = colArray)

        # Process partition into intensities (average) for summary table
        for mArchLabel in list(mArchLabels.keys()):
            for rvLabel in list(rvLabels.keys()):
                
                cond1 = modelSpecs[:,2] == mArchLabel
                cond2 = modelSpecs[:,3] == rvLabel
                modExtract = np.empty((0))
                for modInd in modelSpecs[np.where(cond1 & cond2),0].flatten():
                    modInd = int(modInd)
                    modExtract = np.concatenate(
                                    (modExtract,
                                     modPartition[modInd:modInd+50])
                                                )
                rowInd = rvLabels[rvLabel]
                colInd = mArchLabels[mArchLabel]
                tablePart.loc[rowInd][colInd] = np.nanmean(modExtract)

        # Process partition into intensities (average) for full table
        for (modIndBase, distInd, mArchLabel, rvLabel) in modelSpecs:
            modIndBase = int(modIndBase)
            
            # Populate univariate indices on 1st pass
            if univarInds == []:
                univarInds = {}
                univarBatch = np.asarray(
                    [univarType[key] for key in range(modIndBase, 
                                                      modIndBase+50)]
                                        )
                for univarLabel in list(univarModsLabel.keys()):
                    univarInds[univarModsLabel[univarLabel]] = np.where(
                        univarBatch == univarLabel)[0]
            
            for archInd, archIndLoc in univarInds.items():
                rowInd = rvLabels[rvLabel]
                colInd = mArchLabels[mArchLabel]
                modInd = modIndBase + archIndLoc
                
                fullTablePart.loc[(rowInd,distInd),
                                  (colInd, archInd)] = np.nanmean(
                                      modPartition[modInd])
                                      
        # Rescale entries
        numPartitions = int(max(modPartition))
        tablePart *= 75/max(modPartition)
        tablePart -= 75
        tablePart *= -1
        
        fullTablePart *= 75/max(modPartition)
        fullTablePart -= 75
        fullTablePart *= -1
        
        # Set NAN values to 0
        tablePart = tablePart.fillna(0)
        fullTablePart = fullTablePart.fillna(0)
        
        # Set divider columns to nan for formatting
        for colInd in colLabels:
            fullTablePart.loc[:,(colInd, '-')] *= np.nan
        
    else:   # If partition not available, return empty variables
        tablePart = None
        fullTablePart = None
        numPartitions = np.nan
        
    # Append to list of tables for output formatting later on
    tableList.append(table)
    fullTableList.append(fullTable)
    tablePartList.append(tablePart)
    fullTablePartList.append(fullTablePart)
    numPartitionsList.append(numPartitions)
    
    # Time/Memory diagnostics:
    t.append(mcsEst.stats[0]['time'])
    mem.append(mcsEst.stats[0]['memory use'])
    mods.append(mcsEst.stats[0]['models processed'])    
    print('Time taken: {:6.3f} secs - memory used: {:6.3f} MB'.format(
        t[-1],mem[-1]*10**-6))

print('Average collection size: {:6.3f} secs'.format(np.mean(mods)))    
print('Average time taken:      {:6.3f} secs'.format(np.mean(t)))
print('Average memory used:     {:6.3f} MB'.format(np.mean(mem)*10**-6))

mcsSizes = np.concatenate((mcsSettingsArray,
                                 np.asarray(mcsSize)[:,None]),
                                    axis = 1)
# Generate tables
for (setting, 
     tableDf, 
     fullTableDf, 
     tablePartDf, 
     fullTablePartDf, 
     numPartitions) in zip(mcsSettingsArray, 
                           tableList,
                           fullTableList,
                           tablePartList,
                           fullTablePartList,
                           numPartitionsList):
   
    print(setting[2])
    name = r'/sample_{:s}_pred_{:s}_{:s}_h_{:s}.tex'.format(
                                                      setting[0],
                                                      setting[1],
                                                      setting[2],
                                                      setting[3])
    nameFull = r'/sample_{:s}_pred_{:s}_{:s}_h_{:s}_full.tex'.format(
                                                      setting[0],
                                                      setting[1],
                                                      setting[2],
                                                      setting[3])
    if setting[0] == '0':
        vol = 'high'
    else:
        vol = 'low'
    
    if np.isnan(numPartitions):
        numPartitionsStr = 'No'
    else:
        numPartitionsStr = '{:d}'.format(numPartitions)
    
    title = 'MCS, {:s} bootstrap, {:s} volatility, {:s} proxy, {:s}-day horizon ({:s} classes)'.format(
        setting[2],
        vol,
        setting[1],
        setting[3],
        numPartitionsStr)
    tableDf_np = tableDf.to_numpy(dtype=int)
    tableWithTot = np.concatenate((tableDf_np, 
                                   np.sum(tableDf_np,axis = 0)[None,:]),
                                  axis = 0)
    if tablePartDf is None:
        tablePartWithTot = None
    else:
        tablePartDf_np = tablePartDf.to_numpy(dtype=int)
        tablePartWithTot = np.concatenate((tablePartDf_np, 
                                       np.nan*np.sum(tablePartDf_np,axis = 0)[None,:]),
                                      axis = 0)
    
    formattedTableWithTot = format_int_table(tableWithTot, tablePartWithTot)
    
    # Generate aggregated table from dataframe, print and save
    table = SimpleTable(
            formattedTableWithTot,
            stubs=list(tableDf.index.values) + ['Total'],
            headers=list(tableDf.columns),
            title=title
        )
    
    print(table)
    if save is True:
        with open(savePath + name,'w') as f:
            f.write(table.as_latex_tabular(header=r'%s',stub=r'%s'))

    # Generate full table from dataframe and save
    fullStubs = ['{:s}, {:s}'.format(*foo) for foo in list(fullTableDf.index)]
    fullHead = []
    for mgarchMod in ['N','C', 'D', 'A']:
        for label in univarLabels:
            fullHead.append('{:s}:{:s}'.format(mgarchMod,label))
    
    fullTableDf_np = fullTableDf.to_numpy(dtype=int)
    if fullTablePartDf is None:
        fullTablePartDf_np = None
    else:
        # fullTablePartDf_np = fullTablePartDf.to_numpy(dtype=int)
        fullTablePartDf_np = fullTablePartDf.to_numpy(dtype=float)
    
    formattedFullTable = format_int_table(fullTableDf_np, 
                                          fullTablePartDf_np, 
                                          maxIntLen = 2)
    
    fullTable = SimpleTable(
            formattedFullTable,
            stubs=fullStubs,
            headers=fullHead,
            title=title,
        )  
    if save is True:
        with open(savePath + nameFull,'w') as f:
            f.write(fullTable.as_latex_tabular(header=r'%s',stub=r'%s'))
