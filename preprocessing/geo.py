'''
@File    :   geo.py
@Time    :   2023/09/06 12:01:06
@Author  :   Qiuzi Chen 
@Version :   1.0
@Contact :   qiuzi.chen@outlook.com
@Desc    :

geo module is developed for trajectory geographic calculation, including
- coordinate transform
- road network extraction
- road elevation matching
- road grade calculation
'''

import math
import numpy as np
import pandas as pd
import warnings

from osgeo import osr, gdal
from preprocessing.smoothing import expSmooth


"""
COORDINATE TRANSFORM
"""

x_PI = 3.14159265358979324 * 3000.0 / 180.0
PI = 3.1415926535897932384626  # π
a = 6378245.0  # major semi axispainxi
ee = 0.00669342162296594323  # eccentricity squared


def gcj02_to_bd09(lng, lat):
    """
    GCJ-02 to BD-09
    Googel, Gaode to Baidu
    """
    z = math.sqrt(lng * lng + lat * lat) + 0.00002 * math.sin(lat * x_PI)
    theta = math.atan2(lat, lng) + 0.000003 * math.cos(lng * x_PI)
    bd_lng = z * math.cos(theta) + 0.0065
    bd_lat = z * math.sin(theta) + 0.006
    return [bd_lng, bd_lat]


def bd09_to_gcj02(bd_lon, bd_lat):
    """
    BD-09 to GCJ-02
    Baidu to Google, Gaode
    """
    x = bd_lon - 0.0065
    y = bd_lat - 0.006
    z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * x_PI)
    theta = math.atan2(y, x) - 0.000003 * math.cos(x * x_PI)
    gg_lng = z * math.cos(theta)
    gg_lat = z * math.sin(theta)
    return [gg_lng, gg_lat]


def wgs84_to_gcj02(lng, lat):
    """
    WGS84 to GCJ02
    """
    if out_of_china(lng, lat):  # whether in range of China
        return [lng, lat]
    dlat = _transformlat(lng - 105.0, lat - 35.0)
    dlng = _transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * PI
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * PI)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * PI)
    mglat = lat + dlat
    mglng = lng + dlng
    return [mglng, mglat]


def gcj02_to_wgs84(lng, lat):
    """
    GCJ02 to GPS84
    """
    if out_of_china(lng, lat):
        return [lng, lat]
    dlat = _transformlat(lng - 105.0, lat - 35.0)
    dlng = _transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * PI
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * PI)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * PI)
    mglat = lat + dlat
    mglng = lng + dlng
    return [lng * 2 - mglng, lat * 2 - mglat]


def bd09_to_wgs84(bd_lon, bd_lat):
    lon, lat = bd09_to_gcj02(bd_lon, bd_lat)
    return gcj02_to_wgs84(lon, lat)


def wgs84_to_bd09(lon, lat):
    lon, lat = wgs84_to_gcj02(lon, lat)
    return gcj02_to_bd09(lon, lat)


def _transformlat(lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
          0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * PI) + 20.0 *
            math.sin(2.0 * lng * PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * PI) + 40.0 *
            math.sin(lat / 3.0 * PI)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * PI) + 320 *
            math.sin(lat * PI / 30.0)) * 2.0 / 3.0
    return ret


def _transformlng(lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
          0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * PI) + 20.0 *
            math.sin(2.0 * lng * PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * PI) + 40.0 *
            math.sin(lng / 3.0 * PI)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * PI) + 300.0 *
            math.sin(lng / 30.0 * PI)) * 2.0 / 3.0
    return ret


def out_of_china(lng, lat):
    """
    Determine whether the position is in china
    return True if yes
    """
    return not (lng > 73.66 and lng < 135.05 and lat > 3.86 and lat < 53.55)


"""
ELEVATION AND GRADE
"""


def grade2traj(
        traj:pd.DataFrame,
        dem=None,
        lonCol='lon',
        latCol='lat',
        distCol='dist[km]',
        gradeColName='grade[D]'
):
    """
    Match elevation to traj points and calculate the grade.
    """
    if dem:
        
        warnings.filterwarnings('ignore')

        # get elevation
        eleMap = dem2ele(dem)
        traj.loc[:, 'ele[m]'] = traj.apply(lambda x: eleMatch(x, dem, eleMap), axis=1)
        # elevation smoothing
        traj.loc[:, 'ele[m]'] = expSmooth(traj['ele[m]'].to_list(), alpha=0.7)
        traj.loc[:, 'lon_'] = traj[lonCol].shift(-1)
        traj.loc[:, 'lat_'] = traj[latCol].shift(-1)
        traj.drop(['lon_', 'lat_'], axis=1, inplace=True)
        a, b = traj['ele[m]'].diff(-1), (traj[distCol] * 1000)
        traj.loc[:, gradeColName] = np.divide(a, b, where=b!=0)  # avoid distance=0
        traj.loc[:, gradeColName] = traj[gradeColName].apply(np.arctan)  # theta=arctan(grade)

        # traj.drop(['ele[m]'], axis=1, inplace=True)
    else:
        traj.loc[:, gradeColName] = 0

    traj[gradeColName] = traj[gradeColName].astype('float32')

    return traj


def dem2ele(dem):
    """
    Obtain elevation from DEM.
    """
    band = dem.GetRasterBand(1)
    eleMap = band.ReadAsArray()
    return eleMap


def eleMatch(
        traj_point:pd.DataFrame,
        dem,
        eleMap,
        lonCol='lon',
        latCol='lat',
):
    """
    Match elevation to traj points.
    """
    lon, lat = traj_point[lonCol], traj_point[latCol]
    x, y = geo2xy(dem, lon, lat)
    return eleMap[x][y]

# -------------------------------------------------------

def geo2pro(dem, lon, lat):
    """
    Transfer geo longitude, latitude to projected x, y
    """
    # projected and geographic reference system
    psr = osr.SpatialReference()
    psr.ImportFromWkt(dem.GetProjection())
    gsr = psr.CloneGeogCS()
    
    # coordinate transformer
    ct = osr.CoordinateTransformation(gsr, psr)
    coords = ct.TransformPoint(lat, lon)
    return coords[0], coords[1]


def pro2xy(dem, x, y):
    """
    Transfer projected coordinate to row,col in rasters .
    """
    x0, dx, dxy, y0, dyx, dy = dem.GetGeoTransform()
    # if dxy == dyx == 0:
    #     col = (x - x0) / dx
    #     row = (y - y0) / dy
    a = np.array([[dx, dxy], [dyx, dy]])
    b = np.array([x - x0, y - y0])
    col, row = np.linalg.solve(a, b)
    return int(row), int(col)


def geo2xy(dem, lon, lat):
    """
    Transfer geo longitude, latitude to row, col in rasters.
    """
    x, y = geo2pro(dem, lon, lat)
    row, col = pro2xy(dem, x, y)
    return row, col
