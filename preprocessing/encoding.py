'''
@File    :   preprocess.py
@Time    :   2023/09/06 14:27:39
@Author  :   Qiuzi Chen 
@Version :   1.0
@Contact :   qiuzi.chen@outlook.com
@Desc    :   
'''

import numpy as np
import pandas as pd
import time
from preprocessing.geo import gcj02_to_wgs84, bd09_to_wgs84


def encode(
        df:pd.DataFrame,
        colName=['vehID', 'orderID', 'time[s]', 'lon', 'lat'], 
        encodingCol=['vehID', 'orderID'],
        sortBy=['vehID', 'orderID', 'time[s]'],
        originCRS='GCJ02',
):
    """
    Encoding data.
    df: trajectory data
    colName: set column name
    encodingCol: columns need encoding
    sortBy: columns to sort by, list
    originCRS: coordinate reference system of raw data
    """
    print("Encoding...")

    if len(colName) != len(df.columns):
        raise ValueError("Input DataFrame has %d cols while %d colNames were given!" % (len(df.columns), len(colName)))
    
    df.columns = colName
    
    # vehicle id encoding
    for col in encodingCol:
        df[col] = df[col].squeeze().map({item:id for id, item in enumerate(df[col].squeeze().unique())})
    
    # CRS transferring
    if originCRS == 'WGS84':
        pass
    elif originCRS == 'GCJ02':
        df['lon'] = df.apply(lambda row: np.float64(gcj02_to_wgs84(row['lon'], row['lat'])[0]), axis=1)
        df['lat'] = df.apply(lambda row: np.float64(gcj02_to_wgs84(row['lon'], row['lat'])[1]), axis=1)
    elif originCRS == 'BD09':
        df['lon'] = df.apply(lambda row: np.float64(bd09_to_wgs84(row['lon'], row['lat'])[0]), axis=1)
        df['lat'] = df.apply(lambda row: np.float64(bd09_to_wgs84(row['lon'], row['lat'])[1]), axis=1)
    else:
        raise KeyError("Tranferring from %s to WGS84 is not available." % originCRS)
    
    # sort
    df = df.sort_values(by=sortBy)
    df.reset_index(inplace=True, drop=True)
    
    return df


def encodeTime(
        df:pd.DataFrame,
        timeCol='time[s]',
        date=True,
        weekday=True,
        hour=True
):
    """
    Encoding date and time.
    """
    if date:
        df['date'] = df[timeCol].squeeze().map(dateMap)
    if weekday:
        df['weekday'] = df[timeCol].squeeze().map(weekMap) 
    if hour:
        df['hour'] = df[timeCol].squeeze().map(hourMap)
    
    return df

# ---------

def dateMap(unixt):
    """
    Transfer unix time to local time.
    """
    t = time.localtime(unixt)
    return time.strftime("%Y/%m/%d", t)

def timeMap(unixt):
    """
    Transfer unix time to local time.
    """
    t = time.localtime(unixt)
    return time.strftime("%H:%M:%S", t)

def weekMap(unixt):
    """
    Transfer unix time to local time.
    """
    t = time.localtime(unixt)
    return time.strftime("%A", t)

def timeSecMap(unixt):
    """
    Transfer unix time to local time.
    """
    t = time.localtime(unixt)
    return t.tm_hour*3600 + t.tm_min*60 + t.tm_sec

def hourMap(unixt):
    """
    Transfer unix time to local time.
    """
    t = time.localtime(unixt)
    return t.tm_hour