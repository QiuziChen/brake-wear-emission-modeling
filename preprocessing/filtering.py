'''
@File    :   filtering.py
@Time    :   2023/09/16 11:39:24
@Author  :   Qiuzi Chen 
@Version :   1.0
@Contact :   qiuzi.chen@outlook.com
@Desc    :   Provide data filtering tools to improve data quality.
'''

import pandas as pd
from preprocessing.calculation import calDistInterval, calTimeInterval, calSpeed, calAcc, calVSP

def filter(
        traj:pd.DataFrame,
        # lonCol='lon',
        # latCol='lat',
        intervalCol="interval[s]",
        speedCol='speed[km/h]',
        accCol='acc[m/s2]',
        VSPCol='VSP[kW/t]',
        minSpeed=0,
        maxSpeed=150,
        minAcc=-4.4704,
        maxAcc=6.25856,
        minVSP=-47.5,
        maxVSP=62.5
):
    """
    Filtering to keep data consistancy and remove redundancy.
    traj: traj dataframe
    redundancyFilter: True if want to perform redundancy filter
    speedFilter: True if want to perform speed filter
    lonCol: column name of longitude
    latCol: column name of latitude
    minSpeed: the minimum speed limit [km/h]
    maxSpeed: the maximum speed limit [km/h]
    minAcc: the minimum acceleration limit [m/s2]
    maxAcc: the maximum acceleration limit [m/s2]    
    minVSP: the minimum VSP limit [kW/t]
    maxVSP: the maximum VSP limit [kW/t]
    """
    print("Filtering...")

    length0 = traj.shape[0]

    # remove redundancyFilter
    traj.drop_duplicates(keep='first', inplace=True)
    
    # remove outliers
    traj = traj.loc[
        (traj[intervalCol] >= 0)
        & (traj[speedCol] <= maxSpeed)
        & (traj[speedCol] >= minSpeed)
        & (traj[accCol] >= minAcc)
        & (traj[accCol] <= maxAcc)
        & (traj[VSPCol] >= minVSP)
        & (traj[VSPCol] <= maxVSP)
    ].copy()

    # traj.reset_index(inplace=True, drop=True)

    # # distance interval
    # calDistInterval(
    #     traj,
    #     lonCol, latCol,
    # )
    
    # # time interval
    # calTimeInterval(
    #     traj,
    # )
    
    # # speed
    # calSpeed(
    #     traj,
    # )

    # # acc
    # calAcc(
    #     traj,
    # )

    # # VSP
    # calVSP(
    #     traj,
    #     gradeCol=None
    # )

    # # remove outliers
    # traj = traj.loc[
    #     (traj['interval[s]'] >= 0)
    #     & (traj['speed[km/h]'] <= maxSpeed)
    #     & (traj['speed[km/h]'] >= minSpeed)
    #     & (traj['acc[m/s2]'] >= minAcc)
    #     & (traj['acc[m/s2]'] <= maxAcc)
    #     & (traj['VSP[kW/t]'] >= minVSP)
    #     & (traj['VSP[kW/t]'] <= maxVSP)
    # ].copy()

    # traj.reset_index(inplace=True, drop=True)
    # traj.drop(['dist[km]', 'interval[s]', 'speed[km/h]', 'acc[m/s2]', 'VSP[kW/t]'], axis=1, inplace=True)

    length1 = traj.shape[0]
    print("- filtered length: %d;\n- filtered ratio: %.2f%%." % (length1, (length0-length1)/length0*100))

    return traj
