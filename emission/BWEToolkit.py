'''
@File    :   BWEToolkit.py
@Time    :   2023/11/16 11:53:15
@Author  :   Qiuzi Chen 
@Version :   1.0
@Contact :   qiuzi.chen@outlook.com
@Desc    :   Provide modelling and calculation toolkits for BWE.
'''


import pandas as pd
from emission.braking import PEREDetect
from emission.OpMode import OpModeDetect
from emission.emissionRate import ERCalculator


class BWETool():
    """
    Toolkits for brake wear emission modelling and calculation.
    """
    def __init__(self) -> None:
        """
        Initialization a BWE toolkit.
        """
        self.__BRAKE_DETECT_METHODS = {
            "PERE": PEREDetect
        }
        self.ER = ERCalculator()
        
    #TODO: braking detect method for different vehicle types
    def brakingDetect(
            self, 
            traj:pd.DataFrame,
            speedCol="speed[km/h]",
            accCol="acc[m/s2]",
            brakeColName="braking",
            method="PERE",
            # vehType=None
    ):
        """
        Determine whether braking event is happening according to braking detect models.
        f: trajectory DataFrame.
        speedCol: column name of speed.
        accCol: column name of acceleration.
        """
        model = self.__BRAKE_DETECT_METHODS[method]
        return model(traj, speedCol, accCol, brakeColName)
    
    def OpModeDetect(
            self,
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
        return OpModeDetect(traj, gradeCol, speedCol, accCol, VSPCol, OpModeColName)
    
    # TODO: intergrate ER calcualtion
    # def ERCal(
    #         self,
    #         decel, 
    #         material='avg', 
    #         pollutant='PM10'
    # ):
    #     pass
        