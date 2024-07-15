# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 09:22:23 2023

This script produces the empirical data plots for the mArch forecast comparison 
exercise (figure 2 in the paper).

@author: Sylvain Barde, University of Kent
"""

import os
import pandas as pd
from matplotlib import pyplot as plt

data_path = 'data/oxfordmanrealizedvolatilityindices.csv'
savePath = 'outputs'
save = True

# Set parameters 
indList = ['.SPX', '.FTSE', '.N225']        # Database indices
L1 = 1500                                   # Training obs
L2 = L1+550                                 # Total obs (train+test)
startList = [200, 1250]                     # List of start points

# Setup latex output and load/save folders
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})

if save is True:
    if not os.path.exists(savePath):
        os.makedirs(savePath,mode=0o777)

# Load dataset, extract indices, format data
data = pd.read_csv(data_path, index_col=[0])

for count,ind in enumerate(indList):
    if count == 0:
        mRt = pd.DataFrame(          # Daily returns (for sign)
                data[data.Symbol == ind]['open_to_close']
                ).rename(columns={"open_to_close": ind})

    else:
        mRt = mRt.merge(pd.DataFrame(
            data[data.Symbol == ind]['open_to_close']
            ).rename(columns={"open_to_close": ind}),
                        how='inner', 
                        left_index = True, 
                        right_index = True)
mRt *= 100                              # Rescale to % from raw log-returns
mRt.index = pd.to_datetime(mRt.index)   # Change index to datetime for plot

# Iterate over settings to make plots
fontSize = 40
y_min = -12
y_max = 12

# for ind, indLabel in zip(indList,indLabels):
for ind in indList:
    for start in startList:
        xlim_left = mRt.index[start]
        xlim_right = mRt.index[start + L2]
        
        if ind == indList[0]:
            print('Starting point: {:d}'.format(start))
            print(' first obs: {:s}'.format(str(xlim_left)))
            print(' test start obs: {:s}'.format(str(mRt.index[start + L1])))
            print(' last obs: {:s}'.format(str(xlim_right)))
        
        fig = plt.figure(figsize=(16,12))
        ax = fig.add_subplot(1, 1, 1)
        ax.fill([mRt.index[start+L1],xlim_right,xlim_right,mRt.index[start+L1]], 
                [y_min,y_min,y_max,y_max], 
                color = 'k', alpha=0.15, label = r'Testing sample')
        ax.plot(mRt.index[start:start+L2], mRt.iloc[start:start+L2][ind], 
                'b', linewidth=1,label = r'\% daily return')
        ax.set_ylim([y_min,y_max])
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
            plt.savefig(savePath + "/plot_{:s}_start_{:d}.pdf".format(
                        ind.replace('.',''),start), 
                        format = 'pdf', bbox_inches = 'tight')