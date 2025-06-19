# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 09:27:26 2025

This script produces the outputs for the fastMCS power analysis.
(table 2 in the paper).

@author: Sylvain Barde, University of Kent
"""

import pickle
import zlib
import os

import numpy as np
from statsmodels.iolib.table import SimpleTable

def formatInnerTable(rawTable):
    innerTableFormatted = []
    for row in rawTable:
        
        rowFormatted = []
        for cellValue in row:
            rowTFormatted = '${:3.3f}$'.format(cellValue)
            rowFormatted.append(rowTFormatted)
        innerTableFormatted.append(rowFormatted)
    
    innerTableLabels = []
    for i in range(rawTable.shape[0]):
        innerTableLabels.append('{:g}'.format(lmbdas[i]))
            
    innerTableHeaders = []
    for i in range(rawTable.shape[1]):
        innerTableHeaders.append('{:g}'.format(rhos[i]))
        
    return (innerTableFormatted, innerTableLabels, innerTableHeaders)


#------------------------------------------------------------------------------
# Set Common parameters
lmbdas = [5, 10, 20, 40]
rhos = [0, 0.5, 0.75, 0.95]
phis = [0, 0.5, 0.8]
Nmods = [500, 1000, 2000]
numObs = 250
save = True

sizeTitle = ' Frequency at which $\mathcal{M}^* \subset \mathcal{M}^*_{90}$ (size)'
powerTitle = ' Average number of elements in $\mathcal{M}^*_{90}$ (power)'
tableTitle = ' MCS size and power analysis'

# Paths
loadFolder = 'montecarlo//power_results_N_{:d}'.format(numObs)
fileName = 'power_analysis.pkl'
save_path_tables = 'montecarlo//outputs'

loadPath = (loadFolder + '/' + fileName)

fil = open(loadPath,'rb')
datas = zlib.decompress(fil.read(),0)
fil.close()
mcData = pickle.loads(datas,encoding="bytes")

# -- Prepare MC table panels
sizeMcIter = np.zeros((len(lmbdas),
                        len(rhos),
                        len(phis),
                        len(Nmods)))

powerMcIter = np.zeros((len(lmbdas),
                        len(rhos),
                        len(phis),
                        len(Nmods)))

# Extract data from MC iterations, calculate averages.
numMcIter = len(mcData)
for iterationId, mcIteration in mcData.items():
    sizeMcIter += mcIteration['sizeMcIter']/numMcIter
    powerMcIter += mcIteration['powerMcIter']/numMcIter

# Iterate over table panels to build overall table
initFlagCol = True
for j, m in enumerate(Nmods):
    
    initFlagRow = True
    labelFlagRow = True
    for i, phi in enumerate(phis):
        rawSizeTable = sizeMcIter[:,:,i,j]
        rawPowerTable = powerMcIter[:,:,i,j]/m
        
        sizeTblFormatted, sizeTblLabels, sizeTblHeaders = formatInnerTable(
            rawSizeTable)
        powerTblFormatted, powerTblLabels, powerTblHeaders = formatInnerTable(
            rawPowerTable)
        
        sizeTblFormatted = np.asarray(sizeTblFormatted)
        powerTblFormatted = np.asanyarray(powerTblFormatted)
        
        if initFlagRow:
            hpad = np.asarray(['   ']*sizeTblFormatted.shape[1])[None,:]
            colFormatted = np.concatenate((hpad,
                                           hpad,
                                           sizeTblFormatted,
                                           hpad,
                                           powerTblFormatted),
                                           axis = 0)
            if labelFlagRow:
                tableLabels = [' Panel {:d}: $\phi={:g}$'.format(i,phi),
                               sizeTitle,
                               *sizeTblLabels,
                               powerTitle,
                               *powerTblLabels]
            initFlagRow = False
        else:            
            colFormatted = np.concatenate((colFormatted, 
                                           hpad,
                                           hpad,
                                           sizeTblFormatted,
                                           hpad,
                                           powerTblFormatted),
                                           axis = 0)
            if labelFlagRow:
                extraLabels = [' Panel {:d}: $\phi={:g}$'.format(i,phi),
                               sizeTitle,
                               *sizeTblLabels,
                               powerTitle,
                               *powerTblLabels]
                tableLabels = tableLabels + extraLabels
        
    labelFlagRow = False
    if initFlagCol:
        tableFormatted = colFormatted
        vpad = np.asarray(['   ']*tableFormatted.shape[0])[:,None]
        tableHeaders = sizeTblHeaders

        initFlagCol = False
    else:
        tableFormatted = np.concatenate((tableFormatted, 
                                       vpad,
                                       colFormatted),
                                       axis = 1)
        extraHeaders = ['   ', *sizeTblHeaders]
        tableHeaders = tableHeaders + extraHeaders
    colFormatted = None
    
table = SimpleTable(
            tableFormatted,
            stubs = tableLabels,
            headers = tableHeaders,
            title = tableTitle
    )

print(table)

if save is True:
    if not os.path.exists(save_path_tables):
        os.makedirs(save_path_tables,mode=0o777)
    
    with open(save_path_tables + '/power_analysis.tex', 'w') as f_out:
        f_out.write(table.as_latex_tabular(header=r'%s',stub=r'%s',
                                           replacements={"#": r"\#",
                                                         "$": r"$",
                                                         "%": r"\%",
                                                         "&": r"\&",
                                                         ">": r"$>$",
                                                         "_": r"_",
                                                         "|": r"$|$"}))