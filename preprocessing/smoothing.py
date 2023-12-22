'''
@File    :   smoothing.py
@Time    :   2023/09/06 14:23:29
@Author  :   Qiuzi Chen 
@Version :   1.0
@Contact :   qiuzi.chen@outlook.com
@Desc    :   
This module includes three smoothing methods for trajectory and indices:
- Kalman filter smooth
- exponential smooth
- double exponential smooth
'''


import numpy as np
from tqdm import tqdm


def KalmanSmooth(indices:list, Q=0.01, R=0.01):
    """
    Smooth the indices using Kalman Filter 1D.
    """
    n = len(indices)
    size = (n,)

    # parameters
    Q = Q  # process variance
    R = R  # measure variance
    
    # initialize arrays
    xhat_ = np.zeros(size)  # a priori estimate of x
    xhat = np.zeros(size)  # a posteriori estimate of x
    P_ = np.zeros(size)  # a priori error estimate
    P = np.zeros(size)  # a posteriori error estimate
    K = np.zeros(size)  # gain factor

    xhat[0] = indices[0]
    P[0] = 1

    for k in range(1, n):
        # time update
        xhat_[k] = xhat[k-1]
        P_[k] = P[k-1] + Q

        # measurement update
        K[k] = P_[k] / (P_[k] + R)
        xhat[k] = xhat_[k] + K[k] * (indices[k] - xhat_[k])
        P[k] = (1 - K[k]) * P_[k]

    return xhat

def expSmooth(indices:list, alpha=0.3):
    """
    Smooth the indices using exponential smoothing.
    alpha: smoothing parameter
    """
    result = [indices[0]]
    n = len(indices)
    index = list(range(1, n))
    for i in set(index):
        result.append(alpha * result[i-1] + (1 - alpha) * indices[i])
    
    return result

def doubleExpSmooth(indices:list, alpha=0.3, beta=0.1):
    """
    Smooth the indices using double exponential smoothing.
    alpha: level smoothing parameter
    beta: trend smoothing parameter
    """
    # initialize the result
    result = [indices[0]]
    n = len(indices)
    index = list(range(1, n))
    for i in set(index):
        if i == 1:
            level, trend = indices[0], indices[1] - indices[0]
            value = indices[0]
        else:
            value = indices[i]
        level_, level = level, alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - level_) + (1 - beta) * trend
        result.append(level + trend)
    
    return result


SMOOTH_DICT = {
    'exp': expSmooth,
    'dexp': doubleExpSmooth,
    'kal': KalmanSmooth
}

def smooth(
        traj,
        smoothCol=['speed[km/h]', 'acc[m/s2]', 'VSP[kW/t]'],
        tripIDCol="tripID",
        smoothFunc='exp',
        **params
):
    """
    Smooth the data sequence.
    smoothCol: list of columns that need smoothing.
    tripIDCol: column name of tripID. If None, perform smoothing for the whole traj.
    smoothFunc: smoothing method.
        `exp`: exponentail smoothing
        `dexp`: double exponential smoothing
        `kal`: Kalman filter smoothing 
    **params: smoothing parameter for each method.
        `exp`:[alpha]
        `dexp`: [alpha, beta]
        `Kalman`: [Q, R]
    """
    smooth_func = SMOOTH_DICT[smoothFunc]

    _traj = traj.copy()

    if tripIDCol:
        for id in tqdm(set(traj[tripIDCol].unique()), desc="Smoothing"):  
            # select trip from traj.df
            trip = traj[traj[tripIDCol] == id].copy()

            for col in smoothCol:
                _traj.loc[trip.index, col] = smooth_func(trip[col].to_list(), **params)
    
        return _traj
    
    else:
        for col in smoothCol:
            _traj.loc[:, col] = smooth_func(traj[col].to_list(), **params)
        
        return _traj
