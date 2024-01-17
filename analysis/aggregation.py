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

import geopandas as gpd
from shapely import Point

from analysis.calculation import getDecelBinCount, getBinCount, getOpModeCount


PI = 3.1415926535897932384626  # π
EARTH_RADIUS_M = 6371009


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
            columns=['trajCount', 'brakeCount', 'brakeEventNum', 'mileage', 'speedBinCount', 'accBinCount', 'brakeDecelBinCount', 'VSPBinCount', 'OpModeCount'],
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
            df_agg.loc[id]['brakeEventNum'] = df[brakeCol].diff().value_counts(normalize=False)[True] // 2
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
            columns=['trajCount', 'brakeCount', 'brakeEventNum', 'mileage', 'speedMean', 'accMean', 'VSPMean', 'brakeDecelMean', 'OpModeCount'],
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
            df_agg.loc[id]['brakeEventNum'] = df[brakeCol].diff().value_counts(normalize=False)[True] // 2
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
            'brakeEventNum':[],
            'mileage':[],
            'speed_mean':[],
            'speed_std':[],
            'acc_mean':[],
            'acc_std':[],
            'decel_mean':[],
            'decel_std':[],
            'VSP_mean':[],
            'VSP_std':[],
            'initSpeed_mean':[],
            'brakeDecel_mean':[],
            'brakeDecel_std':[],
            'grade_mean':[],
            'grade_std':[],
            'OpModeCount':[]
        }
        warnings.filterwarnings("ignore")

        # start aggregating
        for id in tqdm(set(ref_list), desc="Agg Pairs"):
            df = traj[
                (traj[tripIDCol] == id)
            ].copy()

            if df[distCol].sum() <= maxMileage:
                pass
                # dict_agg['vehID'].append(df.iloc[0][vehIDCol])
                # dict_agg['startHour'].append(df.iloc[0][hourCol])
                # dict_agg['trajCount'].append(df.shape[0])
                # dict_agg['brakeCount'].append(df[df[brakeCol]==True].shape[0])
                # dict_agg['idlingCount'].append(df[df[OpModeCol]==1].shape[0])
                # dict_agg['brakeEventNum'].append(df[brakeCol].diff().value_counts(normalize=False)[True] // 2)
                # dict_agg['mileage'].append(df[distCol].sum())

                # dict_agg['speed_mean'].append(df[speedCol].mean())
                # dict_agg['acc_mean'].append(df[accCol].mean())
                # dict_agg['decel_mean'].append(df[df[accCol] < 0][accCol].mean())
                # dict_agg['VSP_mean'].append(df[VSPCol].mean())
                # dict_agg['brakeDecel_mean'].append(df[df[brakeCol]==True][accCol].mean())
                # dict_agg['grade_mean'].append(df[gradeCol].mean())

                # # calculate braking status change
                # df['status'] = df[brakeCol].diff()
                # dict_agg['initSpeed_mean'].append(df[(df[brakeCol]==True) & (df['status']==True)][speedCol].mean())
                
                # dict_agg['speed_std'].append(df[speedCol].std())
                # dict_agg['acc_std'].append(df[accCol].std())
                # dict_agg['decel_std'].append(df[df[accCol] < 0][accCol].std())
                # dict_agg['VSP_std'].append(df[VSPCol].std())
                # dict_agg['brakeDecel_std'].append(df[df[brakeCol]==True][accCol].std())
                # dict_agg['grade_std'].append(df[gradeCol].std())

                # dict_agg['OpModeCount'].append(getOpModeCount(df, OpModeCol))
            else:
                # calculate cummulative distance
                df['dist_cum'] = df[distCol].cumsum()
                cumMileage = df['dist_cum'].iloc[-1]
                
                # re-segment
                segID = np.array([df[df['dist_cum'] >= maxM].index[0] for maxM in maxMileage * np.arange(0, cumMileage//maxMileage+1)])
                # segID = np.append(segID, df.index[-1]+1)

                for id0, id1 in zip(segID[:-1], segID[1:]):
                    df_ = df.loc[id0:id1].copy()
                    dict_agg['vehID'].append(df_.iloc[0][vehIDCol])
                    dict_agg['startHour'].append(df_.iloc[0][hourCol])
                    dict_agg['trajCount'].append(df_.shape[0])
                    dict_agg['brakeCount'].append(df_[df_[brakeCol]==True].shape[0])
                    dict_agg['idlingCount'].append(df_[df_[OpModeCol]==1].shape[0])
                    try:
                        dict_agg['brakeEventNum'].append(df_[brakeCol].diff().value_counts(normalize=False)[True] // 2)
                    except:
                        dict_agg['brakeEventNum'].append(0)
                    dict_agg['mileage'].append(df_[distCol].sum())

                    dict_agg['speed_mean'].append(df_[speedCol].mean())
                    dict_agg['acc_mean'].append(df_[accCol].mean())
                    dict_agg['decel_mean'].append(np.abs(df_[df_[accCol] < 0][accCol].mean()))
                    dict_agg['VSP_mean'].append(df_[VSPCol].mean())
                    dict_agg['brakeDecel_mean'].append(np.abs(df_[df_[brakeCol]==True][accCol].mean()))
                    dict_agg['grade_mean'].append(df_[gradeCol].mean())

                    # calculate braking status change
                    df_['status'] = df_[brakeCol].diff()
                    dict_agg['initSpeed_mean'].append(df_[(df_[brakeCol]==True) & (df_['status']==True)][speedCol].mean())
                
                    dict_agg['speed_std'].append(df_[speedCol].std())
                    dict_agg['acc_std'].append(df_[accCol].std())
                    dict_agg['decel_std'].append(np.abs(df_[df_[accCol] < 0][accCol].std()))
                    dict_agg['VSP_std'].append(df_[VSPCol].std())
                    dict_agg['brakeDecel_std'].append(np.abs(df_[df_[brakeCol]==True][accCol].std()))
                    dict_agg['grade_std'].append(df_[gradeCol].std())

                    dict_agg['OpModeCount'].append(getOpModeCount(df_, OpModeCol))
        
        df_agg = pd.DataFrame(dict_agg)
        df_agg.fillna(0, inplace=True)
        df_agg['brakeFrac'] = df_agg['brakeCount'] / df_agg['trajCount']
        df_agg['idlingFrac'] = df_agg['idlingCount'] / df_agg['trajCount']
        df_agg['brakingFreq'] = df_agg['brakeEventNum'] / df_agg['mileage']

        return df_agg
    
    def nodeAgg(
            self,
            traj:pd.DataFrame,
            nodes:gpd.GeoDataFrame,
            dist=20,
            lon='lon',
            lat='lat',
            brakeCol='braking',
            distCol='dist[km]',
            speedCol='speed[km/h]',
            accCol='acc[m/s2]',
            VSPCol='VSP[kW/t]',
            OpModeCol='OpModeID',
    ):
        """
        traj: trajectory data, DataFrame
        nodes: nodes data of the roadnet, GeoDataFrame
        dist: enlarge distance of the intersection area
        """
        warnings.filterwarnings("ignore")

        # intial geodataframe for traj
        traj['geometry'] = traj.apply(lambda x: Point(x[lon], x[lat]), axis=1)
        traj_gdf = gpd.GeoDataFrame(traj, geometry='geometry')
        
        # define agg file
        osmid = sorted(nodes['osmid'].dropna().unique())
        agg_node = pd.DataFrame(
            columns=['trajCount', 'brakeCount', 'brakeEventNum', 'mileage', 'speedMean', 'accMean', 'VSPMean', 'brakeDecelMean', 'OpModeCount'],
            index=osmid
        )
        agg_node.loc[:,:] = 0
        
        # clip
        for i in tqdm(set(nodes.index), desc="Agg nodes"):

            # filter points in buffer
            df = traj_gdf.clip(mask=nodes.loc[i].geometry.buffer(dist / EARTH_RADIUS_M * 180 / PI)).copy()
            id = nodes.loc[i, 'osmid']

            if df.shape[0] == 0:
                pass
            else:
                agg_node.loc[id]['trajCount'] = df.shape[0]
                agg_node.loc[id]['brakeCount'] = df[df[brakeCol]==True].shape[0]
                try:
                    agg_node.loc[id]['brakeEventNum'] = df[brakeCol].diff().value_counts(normalize=False)[True] // 2
                except:
                    pass
                agg_node.loc[id]['mileage'] = df[distCol].sum()

                agg_node.loc[id]['speedMean'] = df[speedCol].mean()
                agg_node.loc[id]['accMean'] = df[accCol].mean()
                agg_node.loc[id]['VSPMean'] = df[VSPCol].mean()
                agg_node.loc[id]['brakeDecelMean'] = df[df[brakeCol]==True][accCol].mean()

                agg_node.loc[id]['OpModeCount'] = getOpModeCount(df, OpModeCol)

        return agg_node
            