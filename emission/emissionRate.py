'''
@File    :   rate.py
@Time    :   2023/11/12 15:14:44
@Author  :   Qiuzi Chen 
@Version :   1.0
@Contact :   qiuzi.chen@outlook.com
@Desc    :   Emission rate generation and storage.
'''


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ER data path
PATH_PM10_REVIEW = r"emission/emission-data/BWER_PM10.xlsx"
PATH_LM_POW_PARAM = r"emission/emission-data/LM_PM10_pow_param.npy"
PATH_NAO_POW_PARAM = r"emission/emission-data/NAO_PM10_pow_param.npy"
PATH_SM_POW_PARAM = r"emission/emission-data/SM_PM10_pow_param.npy"
PATH_LM_POW_SCORE = r"emission/emission-data/LM_PM10_pow_score.npy"
PATH_NAO_POW_SCORE = r"emission/emission-data/NAO_PM10_pow_score.npy"
PATH_SM_POW_SCORE = r"emission/emission-data/SM_PM10_pow_score.npy"

# ER param
PM25_10_RATIO = 0.125  # derived from MOVES3.0

# colors
BLUE = '#2878B5'
LIGHTBLUE = '#9AC9DB'
ORANGE = '#F8AC8C'
RED = '#C82423'
PINK = '#FF8884'
GREY = '#9E9E9E'

# plotting params
LABEL_FONT = {
    "fontsize": 12,
    "weight": "bold",
}

class ERCalculator():

    """
    Emission rate calculator, contains brake wear emission factor or rate measurement results and calculation method.
    """
    def __init__(self) -> None:
        
        self.PM10_ERs = pd.read_excel(PATH_PM10_REVIEW)
        self.LM_PM10_ERs = self.PM10_ERs[self.PM10_ERs['Material'] == 'LM']
        self.NAO_PM10_ERs = self.PM10_ERs[self.PM10_ERs['Material'] == 'NAO']
        self.SM_PM10_ERs = self.PM10_ERs[self.PM10_ERs['Material'] == 'SM']

        self.LM_POW_PARAM = np.load(PATH_LM_POW_PARAM)
        self.NAO_POW_PARAM = np.load(PATH_NAO_POW_PARAM)
        self.SM_POW_PARAM = np.load(PATH_SM_POW_PARAM)
        self.MOVES_POW_PARAM = np.array([8 * 0.1872, 3.195])

        self.LM_POW_SCORE = np.load(PATH_LM_POW_SCORE)
        self.NAO_POW_SCORE = np.load(PATH_NAO_POW_SCORE)
        self.SM_POW_SCORE = np.load(PATH_SM_POW_SCORE)

        self._ER_DICT = {
            'LM': self.LM_PM10_ERs,
            'NAO': self.NAO_PM10_ERs,
            'SM': self.SM_PM10_ERs,
        }
        self.ER_PARAM_DICT = {
            'LM': self.LM_POW_PARAM,
            'NAO': self.NAO_POW_PARAM,
            'SM': self.SM_POW_PARAM,
            'MOVES': self.MOVES_POW_PARAM
        }
        self.ER_SCORE_DICT = {
            'LM': self.LM_POW_SCORE,
            'NAO': self.NAO_POW_SCORE,
            'SM': self.SM_POW_SCORE,
        }
        self.NAME_DICT = {
            'LM': "Low Metallic",
            'NAO': "NAO",
            'SM': "Semi-metallic",
            'MOVES': "MOVES",
            'mix': "mix"
        }
        self.NAME_DICT_CHN = {
            'LM': "低金属",
            'NAO': "NAO",
            'SM': "半金属",
            'MOVES': "MOVES",
            'mix': "混合"
        }

    def _pow(self, x, a, b):
        """
        Perform power function.
        """
        return a * np.power(x, b)
    
    def calER(
            self, decel, material='mix', pollutant='PM10', 
            prop_NAO_f = 0.7974, prop_LM_f = 0.1775, prop_SM_f = 0.0251,
            prop_NAO_r = 0.8186, prop_LM_r = 0.1755, prop_SM_r = 0.0058,
            prop_drum_f = 0, prop_drum_r = 0.2228,
            r_f_ratio = 0.5,  # brake force on front axle vs rear axle
            drum_disc_ratio = 0.3, # ratio of drum emissions to disc emissions 
    ):
        """
        Calculate emission rate according to deceleration.
        decel: deceleration value(s) number or ndarray.
        material: name of material,
            'mix': ER of a single vehicle considering local data.
        pollutant: PM10 or PM2.5,
            'PM10': PM10
            'PM2.5': ER of PM2.5 is calculated from PM10 with a PM2.5/10 ratio, default=0.125.
        """

        if material == 'mix':  # calculate the average ER
            # mix for disc brake
            ER_f_disc = (self._pow(decel, *self.NAO_POW_PARAM) * prop_NAO_f) \
                      + (self._pow(decel, *self.LM_POW_PARAM) * prop_LM_f) \
                      + (self._pow(decel, *self.SM_POW_PARAM) * prop_SM_f)
            ER_r_disc = (self._pow(decel, *self.NAO_POW_PARAM) * prop_NAO_r) \
                      + (self._pow(decel, *self.LM_POW_PARAM) * prop_LM_r) \
                      + (self._pow(decel, *self.SM_POW_PARAM) * prop_SM_r)
            ER_r_disc *= r_f_ratio
            
            # adjust for axles
            ER_f_drum = ER_f_disc * drum_disc_ratio
            ER_r_drum = ER_r_disc * drum_disc_ratio

            # mix for veh
            ER = (1 - prop_drum_f) * ER_f_disc \
               + prop_drum_f * ER_f_drum \
               + (1 - prop_drum_r) * ER_r_disc \
               + prop_drum_r * ER_r_drum
            
            ER *= 2
            
            return ER if pollutant == 'PM10' else ER * PM25_10_RATIO

        else:  # calculate the ER of each material or MOVES
            ER = self._pow(decel, *self.ER_PARAM_DICT[material])
            return ER if pollutant == 'PM10' else ER * PM25_10_RATIO
    

    def plotERCurve(
            self, material, pollutant='PM10',
            dpi=100, CHN=False
    ):
        """
        Plot Decel-ER curve.
        material: name of material, if None, plot all.
        """
        # matplotlib config
        plt.rcParams['figure.dpi'] = dpi
        if CHN:
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['font.sans-serif'] = ['SimSun']

        fig, ax = plt.subplots(1,1, figsize=(5,3.6))
        decel = np.arange(0, 8, 0.1)

        if material:  # plot only one curve

            pred = self.calER(decel, material, pollutant)
            
            # plot
            ax.plot(
                decel, pred, color=RED,
                label=self.NAME_DICT_CHN[material] if CHN else self.NAME_DICT[material],
                zorder=10
            )
            if (material != "MOVES") and (material != "mix"):
                ax.scatter(
                    self._ER_DICT[material]['Decel[m/s2]'], self._ER_DICT[material]['PM10ER[g/hr/brake]'],
                    color='k', marker='+', label="Measurement",
                    zorder=20
                )
                param = self.ER_PARAM_DICT[material]
                ax.text(
                    2.0, 0.5*max(pred),
                    "y=%.4fx$^{%.4f}$"%tuple(param) + "\nR$^2$=%.3f" % self.ER_SCORE_DICT[material],
                    color=RED, fontsize=10, weight="bold",
                    zorder=10
                ) 
            elif material == "MOVES":
                param = self.ER_PARAM_DICT[material]
                ax.text(
                    2.0, 0.5*max(pred),
                    "y=%.4fx$^{%.4f}$"%tuple(param),
                    color=RED, fontsize=10, weight="bold",
                    zorder=10
                )

        else:
            raise ValueError("A material name should be designated.")
        
        ax.set_xlabel(
            "减速度 [m/s$^2$]" if CHN else "Deceleration [m/s$^2$]" ,
            fontdict=LABEL_FONT
        )
        if (material != "MOVES") and (material != "mix"):
            ax.set_ylabel(
                "PM10排放率 [g/hr/brake]" if CHN else "PM10 Emission Rate[g/hr/brake]\n(single front brake)",
                fontdict=LABEL_FONT
            )
        else:
            ax.set_ylabel(
                "PM10排放率 [g/hr/veh]" if CHN else "PM10 Emission Rate[g/hr/veh]",
                fontdict=LABEL_FONT
            )
        plt.grid(axis='y', zorder=1)
        plt.legend()
        plt.show()
        