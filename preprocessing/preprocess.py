'''
@File    :   traj.py
@Time    :   2023/09/21 20:02:10
@Author  :   Qiuzi Chen 
@Version :   1.0.1
@Contact :   qiuzi.chen@outlook.com
@Desc    :   Class Traj, packing preprocessing tools
'''


import pandas as pd
from preprocessing.encoding import encode, encodeTime
from preprocessing.filtering import filter
from preprocessing.segmentation import segment
from preprocessing.densification import densify, densifyUnit
from preprocessing.calculation import calParam, calParamUnit
from preprocessing.smoothing import smooth


class Preprocess():
    """
    Preprocess class contains functions for trajectory preprocessing and calculation.
    """
    def __init__(
            self,
            encodeParam={},
            filterParam={},
            segmentParam={},
            densifyParam={},
            calculateParam={},
            smoothParam={},
            encodeTimeParam={},
    ) -> None:
        """
        A DataFrame of trajectory is necessary for initialize a Preprocess object.
        """
        self.__encoded__ = False
        self.__filtered__ = False
        self.__segmented__ = False
        self.__densified__ = False
        self.__calculated__ = False
        self.__encodeParam__ = encodeParam
        self.__filterParam__ = filterParam
        self.__segmentParam__ = segmentParam
        self.__densifyParam__ = densifyParam
        self.__calculateParam__ = calculateParam
        self.__smoothParam__ = smoothParam
        self.__encodeTimeParam__ = encodeTimeParam

        # DEFAULT PARAMETERS
        self.DEFAULT_ENCODE_PARAM = {
            varname: value
            for varname, value in zip(encode.__code__.co_varnames[1:], encode.__defaults__)
        }

        self.DEFAULT_FILTER_PARAM = {
            varname: value
            for varname, value in zip(filter.__code__.co_varnames[1:], filter.__defaults__)
        }

        self.DEFAULT_SEGMENT_PARAM = {
            varname: value
            for varname, value in zip(segment.__code__.co_varnames[1:], segment.__defaults__)
        }

        self.DEFAULT_DENSIFY_PARAM = {
            varname: value
            for varname, value in zip(densify.__code__.co_varnames[1:], densify.__defaults__)
        }
        _DEFAULT_DENSIFYUNIT_PARAM = {
            varname: value
            for varname, value in zip(densifyUnit.__code__.co_varnames[1:], densifyUnit.__defaults__)
        }
        self.DEFAULT_DENSIFY_PARAM.update(_DEFAULT_DENSIFYUNIT_PARAM)

        self.DEFAULT_CALCULATE_PARAM = {
            varname: value
            for varname, value in zip(calParam.__code__.co_varnames[1:], calParam.__defaults__)
        }
        _DEFAULT_CALCULATEUNIT_PARAM = {
            varname: value
            for varname, value in zip(calParamUnit.__code__.co_varnames[1:], calParamUnit.__defaults__)
        }
        self.DEFAULT_CALCULATE_PARAM.update(_DEFAULT_CALCULATEUNIT_PARAM)


        self.DEFAULT_SMOOTH_PARAM = {
            varname: value
            for varname, value in zip(smooth.__code__.co_varnames[1:], smooth.__defaults__)
        }

        self.DEFAULT_ENCODETIME_PARAM = {
            varname: value
            for varname, value in zip(encodeTime.__code__.co_varnames[1:], encodeTime.__defaults__)
        }
    
    # def __getParam__(self, paramName, defaultParamDict, resetParamDict):
    #     """
    #     Get parameter from the input and default.
    #     """
    #     if paramName in resetParamDict.keys():
    #         param = resetParamDict['paramName']
    #     else:
    #         param = defaultParamDict['paramName']
    #     return param

    def setParam(self, funcName:str, paramDict):
        """
        Set parameter for each function.
        """
        if funcName == 'encode':
            self.__encodeParam__ = paramDict
        if funcName == 'filter':
            self.__filterParam__ = paramDict
        if funcName == 'segment':
            self.__segmentParam__ = paramDict
        if funcName == 'densify':
            self.__densifyParam__ = paramDict
        if funcName == 'calculate':
            self.__calculateParam__ = paramDict
        if funcName == 'smooth':
            self.__smoothParam__ = paramDict
        if funcName == 'encodeTime':
            self.__encodeTimeParam__ = paramDict
    
    def printDefaultParam(self, funcName:str):
        """
        Print default function parameters.
        funName: function name
        """
        def output(d:dict):
            for k,v in d.items():
                print("%s: %s" % (k,v))

        if funcName == 'encode':
            output(self.DEFAULT_ENCODE_PARAM)
        if funcName == 'filter':
            output(self.DEFAULT_FILTER_PARAM)
        if funcName == 'segment':
            output(self.DEFAULT_SEGMENT_PARAM)
        if funcName == 'densify':
            output(self.DEFAULT_DENSIFY_PARAM)
        if funcName == 'calculate':
            output(self.DEFAULT_CALCULATE_PARAM)
        if funcName == 'smooth':
            output(self.DEFAULT_SMOOTH_PARAM)
        if funcName == 'encodeTime':
            output(self.DEFAULT_ENCODETIME_PARAM)

    def encode(self, traj:pd.DataFrame):
        """
        Encoding.
        traj: trajectory DataFrame.
        """
        if self.__encoded__:
            raise Warning("The trajectory had already been encoded. Re-encoding could raise potential error.")
        self.__encoded__ = True
        
        return encode(traj, **self.__encodeParam__)

    def filter(self, traj:pd.DataFrame):
        """
        Filtering.
        traj: trajectory DataFrame.
        """
        if self.__filtered__:
            raise Warning("The trajectory had already been filtered. Re-filtering could raise potential error.")
        self.__filtered__ = True
        
        return filter(traj, **self.__filterParam__)

    def segment(self, traj:pd.DataFrame):
        """
        Trip segmentation.
        traj: trajectory DataFrame.
        """
        if self.__segmented__:
            raise Warning("The trajectory had already been segmented. Re-segmentation could raise potential error.")
        self.__segmented__ = True

        return segment(traj, **self.__segmentParam__)

    def densify(self, traj:pd.DataFrame):
        """
        Densify the trajectory of trips by interpolation.
        traj: trajectory DataFrame.
        """
        if self.__densified__:
            raise Warning("The trajectory had already been densified. Re-densification could raise potential error.")
        self.__densified__ = True

        return densify(traj, **self.__densifyParam__)

    def calParam(self, traj:pd.DataFrame):
        """
        Calculate parameters.
        traj: trajectory DataFrame.
        demPath: path of the DEM file
        """
        if self.__calculated__:
            raise Warning("The trajectory had already been calculated. Re-calculation could raise potential error.")
        self.__calculated__ = True

        return calParam(traj, **self.__calculateParam__)

    def smooth(self, traj:pd.DataFrame):
        """
        Smooth the data sequence.

        """
        return smooth(traj, **self.__smoothParam__)
        
    # TODO: adjusted encodeTime func for multi-time-format
    def encodeTime(self, traj:pd.DataFrame):
        """
        Encoding date and time.
        """
        return encodeTime(traj, **self.__encodeTimeParam__)

    def pipe(
            self,
            traj:pd.DataFrame,
            steps=['encode', 'filter', 'segment', 'densify', 'calculate', 'smooth', 'encodeTime']
    ):
        """
        A sequence of preprocessing steps.
        traj: trajectory dataframe.
        steps: sequence of preprocessing steps.
        """
        traj = traj.copy()

        self.funcNameDict = {
            'encode': self.encode,
            'filter': self.filter,
            'segment': self.segment,
            'densify': self.densify,
            'calculate': self.calParam,
            'smooth': self.smooth,
            'encodeTime': self.encodeTime,
        }
        
        traj = traj.copy()
        for step in steps:
            func = self.funcNameDict[step]
            traj = func(traj)
        
        return traj


