from kaggl_general.utils.general import process_timer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

def log_transform(frame, cols, verbose=1, merge_frame=True, drop_original=True):
    ''' Preform log transformation on selected columns of a dataframe
    
    :param frame: 
    :param cols: 
    :param verbose: 
    :param merge_frame: 
    :param drop_original: 
    :return: 
    '''
    with process_timer('Log transform', verbose):
        vals = frame[cols].applymap(lambda x: 0 if x == 0 else np.log(x + 2) if x == -1 else np.log(x + 1))
        old_names = vals.columns.values
        new_names = ['N-' + x + '-LogVal' for x in cols]
        vals.rename(columns=dict(zip(old_names, new_names)), inplace=True)

        zeros = frame[cols].applymap(lambda x: 1 if x == 0 else 0)
        old_names = zeros.columns.values
        new_names = ['N-' + x + '-LogZero' for x in cols]
        zeros.rename(columns=dict(zip(old_names, new_names)), inplace=True)

        if drop_original:
            frame.drop(cols, axis=1, inplace=True)

        if merge_frame:
            frame = pd.concat([frame, vals, zeros], axis=1)

    return frame

def rank_transform(frame, axis, rescale=True):
    ''' rank the data

    :param frame:
    :param axis:
    :param rescale:
    :return:
    '''
    frame = frame.rank(axis=axis)

    if rescale:
        scaler = MinMaxScaler()
        frame = scaler.fit_transform(frame)

    return frame

