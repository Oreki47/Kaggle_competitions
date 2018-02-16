import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def detect_binary_string_columns(frame_obj):
    ''' Detect String columns that only has two values with Boolean dtype

        Parameters
        ----------
        frame_obj: Pandas Dataframe to be modified

        Return
        ------
        frame_obj: Pandas Dataframe after modification

        Example
        -------
    '''
    for col in frame_obj.columns:
        if frame_obj[col].dtype == object:
            if len(set(frame_obj[col])) == 2:
                print frame_obj[col].name + ": ",
                print set(frame_obj)

def replace_binary_string_with_boolean(frame_obj):
    ''' Replace String columns that only has two values with Boolean dtype

        Parameters
        ----------
        frame_obj: Pandas Dataframe to be modified

        Return
        ------
        frame_obj: Pandas Dataframe after modification

        Example
        -------
    '''
    for col in frame_obj.columns:
        if frame_obj[col].dtype == object:
            if len(set(frame_obj[col])) == 2:
                frame_obj.loc[frame_obj[col] == -1, col] = 0
                frame_obj.loc[frame_obj[col] != -1, col] = 1
                frame_obj[col] = frame_obj[col].astype('bool')
    return frame_obj

def replace_string_with_boolean(frame_obj):
    ''' Replace String columns that has vals with 1 and the rest with 0

        Parameters
        ----------
        frame_obj: Pandas Dataframe to be modified

        Return
        ------
        frame_obj: Pandas Dataframe after modification

        Example
        -------
    '''
    for col in frame_obj.columns:
        if frame_obj[col].dtype == object:
            frame_obj.loc[frame_obj[col] == -1, col] = 0
            frame_obj.loc[frame_obj[col] != -1, col] = 1
            frame_obj[col] = frame_obj[col].astype('bool')
    return frame_obj

def label_encode(frame, id_features):
    for feature in frame.columns:
        if frame[feature].dtype == 'object':
            lbl = LabelEncoder()
            lbl.fit(list(frame[feature].values))
            frame[feature] = lbl.transform(list(frame[feature].values))
        if feature in id_features:
            lbl = LabelEncoder()
            lbl.fit(list(frame[feature].values))
            frame[feature] = lbl.transform(list(frame[feature].values))
            df_temp = pd.get_dummies(frame[feature])
            df_temp = df_temp.rename(columns=lambda x:feature+str(x))
            frame = pd.concat([frame, df_temp], axis=1)
    return frame
