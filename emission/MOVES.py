'''
@File    :   MOVES.py
@Time    :   2023/10/15 14:25:45
@Author  :   Qiuzi Chen 
@Version :   1.0
@Contact :   qiuzi.chen@outlook.com
@Desc    :   A module containing toolkits and methods provided by MOVES.
'''


import pandas as pd
import numpy as np


def OpModeDetect(
        traj:pd.DataFrame,
        gradeCol='grade[D]',
        speedCol='speed[km/h]',
        accCol='acc[m/s2]',
        VSPCol='VSP[kW/t]',
        OpModeColName="OpModeID"
):
    """
    Detect Operating Mode for each traj point.
    traj: trajectory DataFrame.
    gradeCol: column name of grade.
    speedCol: column name of speed.
    accCol: column name of acc.
    VSPCol: column name of VSP.
    """
    traj = traj.copy()
    traj[OpModeColName] = None
    
    g = 9.8
    traj['acc_grade'] = traj[accCol] + g * np.sin(np.arctan(traj[gradeCol]))
    traj['acc_grade_t-1'] = traj['acc_grade'].shift(1)
    traj['acc_grade_t-2'] = traj['acc_grade'].shift(2)

    traj.loc[(traj['acc_grade']<=-0.894) | ((traj['acc_grade']<-0.447) & (traj['acc_grade_t-1']<-0.447) & (traj['acc_grade_t-2']<-0.447)), OpModeColName] = 0
    
    traj.loc[(traj['acc_grade']>-0.894) & (traj[speedCol]<1.609), OpModeColName] = 1

    traj.loc[(traj['acc_grade']>-0.894) & (traj[speedCol]>=1.609) & (traj[speedCol]<40.234) & (traj[VSPCol]<0), OpModeColName] = 11
    traj.loc[(traj['acc_grade']>-0.894) & (traj[speedCol]>=1.609) & (traj[speedCol]<40.234) & (traj[VSPCol]>=0) & (traj[VSPCol]<3), OpModeColName] = 12
    traj.loc[(traj['acc_grade']>-0.894) & (traj[speedCol]>=1.609) & (traj[speedCol]<40.234) & (traj[VSPCol]>=3) & (traj[VSPCol]<6), OpModeColName] = 13
    traj.loc[(traj['acc_grade']>-0.894) & (traj[speedCol]>=1.609) & (traj[speedCol]<40.234) & (traj[VSPCol]>=6) & (traj[VSPCol]<9), OpModeColName] = 14
    traj.loc[(traj['acc_grade']>-0.894) & (traj[speedCol]>=1.609) & (traj[speedCol]<40.234) & (traj[VSPCol]>=9) & (traj[VSPCol]<12), OpModeColName] = 15
    traj.loc[(traj['acc_grade']>-0.894) & (traj[speedCol]>=1.609) & (traj[speedCol]<40.234) & (traj[VSPCol]>=12), OpModeColName] = 16

    traj.loc[(traj['acc_grade']>-0.894) & (traj[speedCol]>=40.234) & (traj[speedCol]<80.467) & (traj[VSPCol]<0), OpModeColName] = 21
    traj.loc[(traj['acc_grade']>-0.894) & (traj[speedCol]>=40.234) & (traj[speedCol]<80.467) & (traj[VSPCol]>=0) & (traj[VSPCol]<3), OpModeColName] = 22
    traj.loc[(traj['acc_grade']>-0.894) & (traj[speedCol]>=40.234) & (traj[speedCol]<80.467) & (traj[VSPCol]>=3) & (traj[VSPCol]<6), OpModeColName] = 23
    traj.loc[(traj['acc_grade']>-0.894) & (traj[speedCol]>=40.234) & (traj[speedCol]<80.467) & (traj[VSPCol]>=6) & (traj[VSPCol]<9), OpModeColName] = 24
    traj.loc[(traj['acc_grade']>-0.894) & (traj[speedCol]>=40.234) & (traj[speedCol]<80.467) & (traj[VSPCol]>=9) & (traj[VSPCol]<12), OpModeColName] = 25
    traj.loc[(traj['acc_grade']>-0.894) & (traj[speedCol]>=40.234) & (traj[speedCol]<80.467) & (traj[VSPCol]>=12) & (traj[VSPCol]<18), OpModeColName] = 27
    traj.loc[(traj['acc_grade']>-0.894) & (traj[speedCol]>=40.234) & (traj[speedCol]<80.467) & (traj[VSPCol]>=18) & (traj[VSPCol]<24), OpModeColName] = 28
    traj.loc[(traj['acc_grade']>-0.894) & (traj[speedCol]>=40.234) & (traj[speedCol]<80.467) & (traj[VSPCol]>=24) & (traj[VSPCol]<30), OpModeColName] = 29
    traj.loc[(traj['acc_grade']>-0.894) & (traj[speedCol]>=40.234) & (traj[speedCol]<80.467) & (traj[VSPCol]>=30), OpModeColName] = 30

    traj.loc[(traj['acc_grade']>-0.894) & (traj[speedCol]>=80.467) & (traj[VSPCol]<6), OpModeColName] = 33
    traj.loc[(traj['acc_grade']>-0.894) & (traj[speedCol]>=80.467) & (traj[VSPCol]>=6) & (traj[VSPCol]<12), OpModeColName] = 35
    traj.loc[(traj['acc_grade']>-0.894) & (traj[speedCol]>=80.467) & (traj[VSPCol]>=12) & (traj[VSPCol]<18), OpModeColName] = 37
    traj.loc[(traj['acc_grade']>-0.894) & (traj[speedCol]>=80.467) & (traj[VSPCol]>=18) & (traj[VSPCol]<24), OpModeColName] = 38
    traj.loc[(traj['acc_grade']>-0.894) & (traj[speedCol]>=80.467) & (traj[VSPCol]>=24) & (traj[VSPCol]<30), OpModeColName] = 39
    traj.loc[(traj['acc_grade']>-0.894) & (traj[speedCol]>=80.467) & (traj[VSPCol]>=30), OpModeColName] = 40


    traj.drop(['acc_grade', 'acc_grade_t-1', 'acc_grade_t-2'], inplace=True, axis=1)

    return traj
