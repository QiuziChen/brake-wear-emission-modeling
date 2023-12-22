'''
@File    :   log.py
@Time    :   2023/10/13 22:36:59
@Author  :   Qiuzi Chen 
@Version :   1.0
@Contact :   qiuzi.chen@outlook.com
@Desc    :   
'''


import sys
import os


def blockprint(func):
    """
    Define a wrapper to block print of funcitons.
    """
    def wrapper(*args, **kwargs):
        sys.stdout = open(os.devnull, 'w')
        results = func(*args, **kwargs)
        sys.stdout = sys.__stdout__
        return results
    return wrapper