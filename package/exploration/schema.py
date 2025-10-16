import numpy as np
import pandas as pd


def is_categorical(column):
    '''
    Check if a column is categorical or not
    :param column: pandas Series
    :return: bool
    '''
    categorical_dtypes = ['object', 'category', 'bool']
    if column.dtype.name in categorical_dtypes:
        return True
    else:
        return False