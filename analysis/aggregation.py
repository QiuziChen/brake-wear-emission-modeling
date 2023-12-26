'''
@File    :   aggregation.py
@Time    :   2023/11/22 22:26:06
@Author  :   Qiuzi Chen 
@Version :   1.0
@Contact :   qiuzi.chen@outlook.com
@Desc    :   Spatio-temporal aggregation methods.
'''

import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from analysis.calculation import getDecelBinCount, getBinCount, getOpModeCount


class Aggregator():
    """
    Contain methods for multi-level aggregation.
    """
    def __init__(self) -> None:
        
        self.SPEED_BIN = np.arange(0, 155, 5)
        self.ACC_BIN = np.arange(-5, 6.5, 0.5)
        self.VSP_BIN = np.arange(-47.5, 65, 2.5)
        self.BRAKE_DECEL_BIN = np.arange(0.1, 4.7, 0.1)
        self.BRAKE_DECEL_BIN_MPH = np.arange(-1, -15, -1) / 2.236936
        self.OPMODEID_LIST = [0,1,11,12,13,14,15,16,21,22,23,24,25,27,28,29,30,33,35,37,38,39,40]

    def overallAgg(
            self,
            traj:pd.DataFrame,
            vehIDCol='vehID',
            brakeCol='braking',
            distCol='dist[km]',
            speedCol='speed[km/h]',
            accCol='acc[m/s2]',
            VSPCol='VSP[kW/t]',
            OpModeCol='OpModeID'
    ):
        """
        Obtain basic information.
        """
        info = {
            "vehNum": [],
            "trajCount": [],
            "matchedCount": [],
            "brakeCount": [],
            "mileage": [],
            "speedBinCount":[],
            "accBinCount":[],
            "VSPBinCount":[],
            "brakeDecelBinCount": [],
            "brakeDecelBinMPHCount": [],
            "OpModeCount": []
        }
        info["vehNum"].append(len(traj[vehIDCol].unique()))
        info["trajCount"].append(traj.shape[0])
        info["matchedCount"].append(traj[~traj['osmid'].isna()].shape[0])
        info["brakeCount"].append(traj[traj[brakeCol]==True].shape[0])
        info["mileage"].append(traj[distCol].sum())
        info["speedBinCount"].append(getBinCount(traj, binCol=speedCol, bins=self.SPEED_BIN))
        info["accBinCount"].append(getBinCount(traj, binCol=accCol, bins=self.ACC_BIN))
        info["VSPBinCount"].append(getBinCount(traj, binCol=VSPCol, bins=self.VSP_BIN))
        info["brakeDecelBinCount"].append(getDecelBinCount(traj[traj[brakeCol]==True], accCol=accCol))
        info["brakeDecelBinMPHCount"].append(getBinCount(traj[traj[brakeCol]==True], binCol=accCol, bins=self.BRAKE_DECEL_BIN_MPH))
        info["OpModeCount"].append(getOpModeCount(traj, OpModeCol))
        
        return pd.DataFrame(info)

    def binAgg(
            self,
            traj:pd.DataFrame,
            refCol='hour',
            brakeCol='braking',
            distCol='dist[km]',
            speedCol='speed[km/h]',
            accCol='acc[m/s2]',
            VSPCol='VSP[kW/t]',
            OpModeCol='OpModeID',
    ):
        """
        Multi-level spatio-temporal aggregation.
        traj: trajectory file to aggregate, DataFrame.
        refCol: reference column for aggregation, 'hour', 'day', or 'weekday' etc. If None, aggregate all time periods.
        OpModeCol: column name of OpMode, if None, don't aggregate for each OpMode.
        """
        # generate index
        ref_list = sorted(traj[refCol].dropna().unique())
        
        # define agg file
        df_agg = pd.DataFrame(
            columns=['trajCount', 'brakeCount', 'mileage', 'speedBinCount', 'accBinCount', 'brakeDecelBinCount', 'VSPBinCount', 'OpModeCount'],
            index=ref_list
        )
        df_agg.loc[:,:] = 0

        # start aggregating
        for id in tqdm(set(ref_list), desc="Agg Pairs"):
            df = traj[
                (traj[refCol] == id)
            ].copy()

            df_agg.loc[id]['trajCount'] = df.shape[0]
            df_agg.loc[id]['brakeCount'] = df[df[brakeCol]==True].shape[0]
            df_agg.loc[id]['mileage'] = df[distCol].sum()
            df_agg.loc[id]['speedBinCount'] = getBinCount(df, binCol=speedCol, bins=self.SPEED_BIN)
            df_agg.loc[id]['accBinCount'] = getBinCount(df, binCol=accCol, bins=self.ACC_BIN)
            df_agg.loc[id]['brakeDecelBinCount'] = getDecelBinCount(df[df[brakeCol]==True], accCol=accCol)
            df_agg.loc[id]['VSPBinCount'] = getBinCount(df, binCol=VSPCol, bins=self.VSP_BIN)
            df_agg.loc[id]['OpModeCount'] = getOpModeCount(df, OpModeCol)

        return df_agg

    def statAgg(
            self,
            traj:pd.DataFrame,
            refCol='hour',
            brakeCol='braking',
            distCol='dist[km]',
            speedCol='speed[km/h]',
            accCol='acc[m/s2]',
            VSPCol='VSP[kW/t]',
            OpModeCol='OpModeID',
    ):
        """
        for simplify aggregation.
        """
        # generate index
        ref_list = sorted(traj[refCol].dropna().unique())
        
        # define agg file
        df_agg = pd.DataFrame(
            columns=['trajCount', 'brakeCount', 'mileage', 'speedMean', 'accMean', 'VSPMean', 'brakeDecelMean', 'OpModeCount'],
            index=ref_list
        )
        df_agg.loc[:,:] = 0

        # start aggregating
        for id in tqdm(set(ref_list), desc="Agg Pairs"):
            df = traj[
                (traj[refCol] == id)
            ].copy()

            df_agg.loc[id]['trajCount'] = df.shape[0]
            df_agg.loc[id]['brakeCount'] = df[df[brakeCol]==True].shape[0]
            df_agg.loc[id]['mileage'] = df[distCol].sum()

            df_agg.loc[id]['speedMean'] = df[speedCol].mean()
            df_agg.loc[id]['accMean'] = df[accCol].mean()
            df_agg.loc[id]['VSPMean'] = df[VSPCol].mean()
            df_agg.loc[id]['brakeDecelMean'] = df[df[brakeCol]==True][accCol].mean()

            df_agg.loc[id]['OpModeCount'] = getOpModeCount(df, OpModeCol)

        return df_agg

    def tripAgg(
            self,
            traj:pd.DataFrame,
            tripIDCol='tripID',
            vehIDCol='vehID',
            # maxDuration=30,
            maxMileage=0.5,
            hourCol='hour',
            brakeCol='braking',
            distCol='dist[km]',
            speedCol='speed[km/h]',
            accCol='acc[m/s2]',
            VSPCol='VSP[kW/t]',
            gradeCol='grade[D]',
            OpModeCol='OpModeID',
    ):
        """
        for trip-level aggregation.
        maxDuration: the maximum duration of a trip segment [s]
        maxMileage: the maximum mileage of a trip segment [km]
        """
        # generate index
        ref_list = sorted(traj[tripIDCol].dropna().unique())
        
        # define agg dict
        dict_agg = {
            'vehID':[],
            'startHour':[],
            'trajCount':[],
            'brakeCount':[],
            'idlingCount':[],
            'mileage':[],
            'speed_mean':[],
            'speed_cv':[],
            'acc_mean':[],
            'acc_cv':[],
            'VSP_mean':[],
            'VSP_cv':[],
            'brakeDecel_mean':[],
            'brakeDecel_cv':[],
            'grade_mean':[],
            'grade_cv':[],
            'OpModeCount':[]
        }
        warnings.filterwarnings("ignore")

        # start aggregating
        for id in tqdm(set(ref_list), desc="Agg Pairs"):
            df = traj[
                (traj[tripIDCol] == id)
            ].copy()

            if df[distCol].sum() <= maxMileage:
                dict_agg['vehID'].append(df.iloc[0][vehIDCol])
                dict_agg['startHour'].append(df.iloc[0][hourCol])
                dict_agg['trajCount'].append(df.shape[0])
                dict_agg['brakeCount'].append(df[df[brakeCol]==True].shape[0])
                dict_agg['idlingCount'].append(df[df[OpModeCol]==1].shape[0])
                dict_agg['mileage'].append(df[distCol].sum())

                dict_agg['speed_mean'].append(df[speedCol].mean())
                dict_agg['speed_cv'].append(df[speedCol].std()/df[speedCol].mean())
                dict_agg['acc_mean'].append(df[accCol].mean())
                dict_agg['acc_cv'].append(df[accCol].std()/df[accCol].mean())
                dict_agg['VSP_mean'].append(df[VSPCol].mean())
                dict_agg['VSP_cv'].append(df[VSPCol].std()/df[VSPCol].mean())
                dict_agg['brakeDecel_mean'].append(df[df[brakeCol]==True][accCol].mean())
                dict_agg['brakeDecel_cv'].append(df[df[brakeCol]==True][accCol].std() / df[df[brakeCol]==True][accCol].mean())
                dict_agg['grade_mean'].append(df[gradeCol].mean())
                dict_agg['grade_cv'].append(df[gradeCol].std()/df[gradeCol].mean())
                
                dict_agg['OpModeCount'].append(getOpModeCount(df, OpModeCol))
            else:
                # calculate cummulative distance
                df['dist_cum'] = df[distCol].cumsum()
                cumMileage = df['dist_cum'].iloc[-1]
                
                # re-segment
                segID = np.array([df[df['dist_cum'] >= maxM].index[0] for maxM in maxMileage * np.arange(0, cumMileage//maxMileage+1)])
                segID = np.append(segID, df.index[-1]+1)

                for id0, id1 in zip(segID[:-1], segID[1:]):
                    df_ = df.loc[id0:id1].copy()
                    dict_agg['vehID'].append(df_.iloc[0][vehIDCol])
                    dict_agg['startHour'].append(df_.iloc[0][hourCol])
                    dict_agg['trajCount'].append(df_.shape[0])
                    dict_agg['brakeCount'].append(df_[df_[brakeCol]==True].shape[0])
                    dict_agg['idlingCount'].append(df_[df_[OpModeCol]==1].shape[0])
                    dict_agg['mileage'].append(df_[distCol].sum())

                    dict_agg['speed_mean'].append(df_[speedCol].mean())
                    dict_agg['speed_cv'].append(df_[speedCol].std()/df_[speedCol].mean())
                    dict_agg['acc_mean'].append(df_[accCol].mean())
                    dict_agg['acc_cv'].append(df_[accCol].std()/df_[accCol].mean())
                    dict_agg['VSP_mean'].append(df_[VSPCol].mean())
                    dict_agg['VSP_cv'].append(df_[VSPCol].std()/df_[VSPCol].mean())
                    dict_agg['brakeDecel_mean'].append(df_[df_[brakeCol]==True][accCol].mean())
                    dict_agg['brakeDecel_cv'].append(df_[df_[brakeCol]==True][accCol].std() / df_[df_[brakeCol]==True][accCol].mean())
                    dict_agg['grade_mean'].append(df_[gradeCol].mean())
                    dict_agg['grade_cv'].append(df_[gradeCol].std()/df_[gradeCol].mean())

                    dict_agg['OpModeCount'].append(getOpModeCount(df_, OpModeCol))
        
        df_agg = pd.DataFrame(dict_agg)
        df_agg.fillna(0, inplace=True)
        df_agg['brakeFrac'] = df_agg['brakeCount'] / df_agg['trajCount']
        df_agg['idlingFrac'] = df_agg['idlingCount'] / df_agg['trajCount']

        return df_agg