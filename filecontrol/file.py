'''
@File    :   file.py
@Time    :   2023/09/16 11:33:48
@Author  :   Qiuzi Chen 
@Version :   1.0
@Contact :   qiuzi.chen@outlook.com
@Desc    :   Provide file processing tools including spilt, merge and storage.

'''

import pandas as pd

def split(df:pd.DataFrame, refCol:str, maxNum=2000):
    """
    Split to avoid memory overflow.
    refCol: split according to which, e.g. id, vehicle id, order
    maxNum: maximum amount
    """
    # items to split
    items = df[refCol].squeeze().unique()  # unique item id

    # items at the split boundry
    if maxNum >= len(items):
        return df
    elif len(items) % maxNum == 0:  # can be divided exactly
        splitItems = [items[maxNum*i] for i in range(1, len(items)//maxNum)]
    else:
        splitItems = [items[maxNum*i] for i in range(1, (len(items)//maxNum)+1)]

    # first index of each boundry item
    itemsDF = [df[refCol] == item for item in splitItems]
    splitID = [0]
    for itemDF in itemsDF:
        splitID.append(df[itemDF.squeeze()].index[0])

    # split df according to boundry index
    df_split = [df.loc[splitID[i]:splitID[i+1]] for i in range(len(splitID)-1)]
    df_split.append(df.loc[splitID[-1]:])
    
    return df_split