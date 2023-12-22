'''
@File    :   calculation.py
@Time    :   2023/11/26 18:34:21
@Author  :   Qiuzi Chen 
@Version :   1.0
@Contact :   qiuzi.chen@outlook.com
@Desc    :   
'''


import numpy as np


def getDecelBinCount(traj, accCol):
    """
    Obtain counts for each brake deceleration bin.
    """
    interval = 0.1
    bins = np.arange(interval, 4.6, interval)
    count_list = [traj[(traj[accCol] > -bin) & (traj[accCol] <= -bin+interval)].shape[0] for bin in bins]
    return np.array(count_list)

def getBinCount(traj, binCol, bins):
    """
    Get counts for each deceleration bin.
    traj: trajectory data, DataFrame
    min_decel: the minimum deceleration (abs).
    max_decel: the maximum deceleration (abs).
    interval: interval of bins
    """

    # count
    count = [traj[(traj[binCol] >= minV) & (traj[binCol] < maxV)].shape[0] for minV, maxV in zip(bins[:-1], bins[1:])]
    
    return np.array(count)

def getOpModeCount(traj, OpModeCol):
    """
    
    """
    OpModeID_list = [0,1,11,12,13,14,15,16,21,22,23,24,25,27,28,29,30,33,35,37,38,39,40]
    value_counts = traj[OpModeCol].value_counts()
    count_list = [value_counts.loc[i] if (i in value_counts.index) else 0 for i in OpModeID_list]
    return np.array(count_list)