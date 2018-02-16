import time
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_squared_log_error



def _process_timer(func):
    """
    Print the seconds that a function takes to execute.
    """
    def wrapper(*args, **kwargs):
        t0 = time.time()
        res = func(*args, **kwargs)
        print("function @{0} took {1:0.3f} seconds".format(func.__name__, time.time() - t0))
        return res
    return wrapper


def log1p_transform(df, cols, drop=False):
    for col in cols:
        df[col+'_log1p'] = np.log1p(df[col])

        if drop:
            df.drop(col, axis=1, inplace=True)

def lb_en_transform(df, cols, drop=True):
    lb = LabelEncoder()
    for col in cols:
        df[col+'_lb'] = lb.fit_transform(df[col].astype(str))

        if drop:
            df.drop(col, axis=1, inplace=True)


def sub_to_csv(sub, v06, v33, score_1, score_2, model_name):
    ''' Save submission with timestamp and score
        also save y_train_cv for blending

    '''
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    score1 = (np.round(score_1, 4)).astype('str')
    score2 = (np.round(score_2, 4)).astype('str')
    str_val = '_' + model_name +  '_' + score1 + '_' + score2
    sub.to_csv('../submission/sub{}.csv'.format(str_val), index=False)
    try:
        v06.to_csv('../valid/v06{}.csv'.format(str_val), index=False)
        v33.to_csv('../valid/v33{}.csv'.format(str_val), index=False)
    except: pass

def sub_to_csv_stacker(sub, score, model_name):
    ''' Save submission with timestamp and score
        also save y_train_cv for blending

    '''
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    score = (np.round(score, 4)).astype('str')
    str_val = '_' + model_name + '_' + score
    sub.to_csv('../submission/stacked/sub{}.csv'.format(str_val), index=False)



def rmse_xgb(y_pred, dtrain):
    y_true = dtrain.get_label()
    score = np.sqrt(mean_squared_log_error(y_true, y_pred))
    return [('RMSE', score)]


def score_valid(y_true, y_valid):
    # should only be called by stacker
    score = np.sqrt(mean_squared_error(y_true, y_valid  ))
    return score

# @_process_timer
def add_groupby_features(source_df, target_df, groupby_cols, method_dict, agg_on_target_df=True):
        '''
            add aggregation features:
            group by [N] columns
            compute stats on [M] columns
            N = len(groupby_cols)
            M = len(method_dit)
        '''

        for k, v in method_dict.items():
            assert type(v) == list, "function signature should be in 'list' type. " \
                                    "For instance, to aggregate on mean of TAEGET, " \
                                    "use method_dict = {TARGET: ['mean']}"

        n = target_df.shape[0]
        temp_df = source_df.copy()
        grouped_stats = temp_df.groupby(groupby_cols).agg(method_dict)
        grouped_stats.columns = ['GP_' + '_'.join(groupby_cols) + '_ON_' +
                                 '_'.join(x) for x in grouped_stats.columns.ravel()]
        grouped_stats.reset_index(inplace=True)
        if not agg_on_target_df:
            return grouped_stats
        else:
            temp = pd.merge(left=target_df, right=grouped_stats, on=groupby_cols, how='left')
            assert temp.shape[0] == n

            return temp


# How to deal with this
# agg_func = lambda x : ((x[1] * x[0]).sum() / x[1])
# agg_func.__name__ = "udf"
# method_dict = {['visitors', 'weight']: [agg_func]}
# groupby_cols = ['air_store_id', 'dow', 'holiday_flg']

def merge_numericals(df, merge_dict, drop=True):
    for col, method in merge_dict.items():
        cols = [col, col.replace('air', 'hpg')]
        name = col.replace("air", "all")

        nan_counts = []
        for c in cols:
            nan_counts.append(df[c].isnull().sum())

        if method == 'sum':
            df[name] = df[cols].sum(axis=1, min_count=1)
        if method == 'mean':
            df[name] = df[cols].mean(axis=1, skipna=True)
        if method == 'max':
            df[name] = df[cols].max(axis=1, skipna=True)
        if method == 'min':
            df[name] = df[cols].min(axis=1, skipna=True)
        if drop:
            df.drop(cols, axis=1, inplace=True)

        assert df[name].isnull().sum() <= np.min(nan_counts)


