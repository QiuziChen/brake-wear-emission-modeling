'''
@File    :   segmentation.py
@Time    :   2023/09/19 17:59:05
@Author  :   Qiuzi Chen 
@Version :   1.0
@Contact :   qiuzi.chen@outlook.com
@Desc    :   Trip segmentation tools for trajectory.    
'''

import pandas as pd
from preprocessing.calculation import calTimeInterval


def segment(
        traj:pd.DataFrame,
        interval=True,
        # stayPoint=True,
        segRefCol='orderID',
        timeCol='time[s]',
        tripIDColName='tripID',
        maxInterval=6,
        # minDuration=1,
        # maxDuration=600,
):
    """
    criteria for segmentation
        - belong to the same vehicle/order
        - interval less than a value

    traj: trajectory data
    interval: use time interval as criteria if True
    segRefCol: sementation reference coloumn
    # stayPoint: use stay point as criteria if True
    timeCol: column name of time
    tripIDColName: column name of the newly generated tripID
    # maxInterval: the maximum time interval inside a trip [s]
    # maxDuration: the maximum duration of a trip [s]
    """
    print("Segmenting...")
    # calculate interval
    traj = calTimeInterval(
        traj,
        timeCol=timeCol,
        calDirect='backward'
    )

    segID = []

    # detect interval-based segment id
    if interval:
        segID = traj.loc[~((traj['interval[s]'] >= 0) & (traj['interval[s]'] <= maxInterval))].index.to_list()
        # _ = segID.pop(-1)  # drop the last segID with NaN interval
        traj.drop(['interval[s]'], axis=1, inplace=True)

    # detect refID change point
    traj.loc[:, 'change'] = traj[segRefCol].diff()
    segID += traj.loc[traj['change'] != 0].index.to_list()  # index of points whose refID is different from the last point's
    segID = sorted(list(set(segID)))
    traj.drop(['change'], axis=1, inplace=True)

    # max duration
    # segDur = [traj.loc[segID[i+1], timeCol] - traj.loc[id, timeCol] for i, id in enumerate(segID)]

    # generate tripID
    traj.loc[:, tripIDColName] = None
    traj.loc[segID, tripIDColName] = list(range(len(segID)))
    traj[tripIDColName].fillna(method='ffill', inplace=True)
    
    print("- segmented trip amount: %d." % len(segID))

    return traj