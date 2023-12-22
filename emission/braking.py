'''
@File    :   braking.py
@Time    :   2023/10/15 14:10:11
@Author  :   Qiuzi Chen 
@Version :   1.0
@Contact :   qiuzi.chen@outlook.com
@Desc    :   braking detection and analysis toolkits.
'''


import numpy as np
import pandas as pd


# Coast down curve params for LDV from MOVES3.0
# y = -0.0001454 * x2 + 0.0000622 * x - O.1758­ (mph)
A = -0.0001454 / (1.609344 ** 2) / 3.6 * 1.609344
B = 0.0000622 / 1.609344 / 3.6 * 1.609344
C = -0.1758 / 3.6 * 1.609344

def PEREDetect(
        traj:pd.DataFrame,
        speedCol="speed[km/h]",
        accCol="acc[m/s2]",
        brakeColName="braking"
):
    """
    Determine whether braking event is happening according to PERE curve (MOVES).
    traj: trajectory DataFrame.
    speedCol: column name of speed.
    accCol: column name of acceleration.
    """
    traj = traj.copy()
    traj.loc[:, brakeColName] = traj.apply(lambda x: True if x[accCol] < coastDownDec(x[speedCol]) else False, axis=1)

    return traj

def coastDownDec(v):
    """
    Coasting deceleration value calculation according to the coast-down curve of 1497kg vehicles.
    v: speed, km/h
    return: dec, m/s^2
    """
    dec = A * v**2 + B * v + C
    return np.float32(dec)

