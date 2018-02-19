import pandas as pd
import numpy as np

def load_all():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sub = pd.read_csv("data/sample_submission.csv")
    # train.drop([149161], axis=0, inplace=True)  # Remove outlier
    return train, test, sub


def load_train():
    train = pd.read_csv("data/train.csv")
    return train


def load_test():
    test = pd.read_csv("data/test.csv")
    return test


def load_sub():
    sub = pd.read_csv("data/sample_submission.csv")
    return sub


def general_downcast(frame):
    ''' Manual handling of the frame downcast

    :param frame:
    :return:
    '''
    # int8
    int8_cols = ['ps_ind_01', 'ps_ind_02_cat', 'ps_ind_03',
                 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_ind_14',
                 'ps_ind_15', 'ps_car_01_cat', 'ps_car_02_cat',
                 'ps_car_03_cat', 'ps_car_04_cat', 'ps_car_05_cat',
                 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat',
                 'ps_car_09_cat', 'ps_car_10_cat', 'ps_car_11_cat',
                 'ps_car_11', 'ps_calc_04', 'ps_calc_05',
                 'ps_calc_06', 'ps_calc_07', 'ps_calc_08',
                 'ps_calc_09', 'ps_calc_10', 'ps_calc_11',
                 'ps_calc_12', 'ps_calc_13', 'ps_calc_14',]
    for col in int8_cols:
        frame[col] = frame[col].astype(np.int8)

    # float32
    float32_cols = ['ps_reg_01', 'ps_reg_02', 'ps_reg_03',
                    'ps_car_12', 'ps_car_13', 'ps_car_14',
                    'ps_car_15', 'ps_calc_01', 'ps_calc_02',
                    'ps_calc_03']
    for col in float32_cols:
        frame[col] = frame[col].astype(np.float32)

    # Bool
    bool_cols = ['ps_ind_06_bin', 'ps_ind_07_bin',
                 'ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_10_bin',
                 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin',
                 'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin',
                 'ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin',
                 'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin',]
    for col in bool_cols:
        frame[col] = frame[col].astype(np.bool)

    return frame


def downcast(x_train, x_test):
    x_train = general_downcast(x_train)
    x_test = general_downcast(x_test)

    return x_train, x_test


def add_groupby_features_n_vs_1(frame, group_columns_list, target_columns_list, methods_list, keep_only_stats=True, verbose=1):
    '''Create statistical columns, group by [N columns] and compute stats on [1 column]

       Parameters
       ----------
       frame: pandas dataframe
          Features matrix
       group_columns_list: list_like
          List of columns you want to group with, could be multiple columns
       target_columns_list: list_like
          column you want to compute stats, need to be a list with only one element
       methods_list: list_like
          methods that you want to use, all methods that supported by groupby in Pandas
       keep_only_stats: boolean
          only keep stats or return both raw columns and stats
       verbose: int
          1 return tick_tock info 0 do not return any info
       Return
       ------
       new pandas dataframe with original columns and new added columns

       Example
       -------
    '''
    dicts = {"group_columns_list": group_columns_list , "target_columns_list": target_columns_list, "methods_list" :methods_list}

    for k, v in dicts.items():
        try:
            if type(v) == list:
                pass
            else:
                raise TypeError(k + " should be a list")
        except TypeError as e:
            print(e)
            raise

    grouped_name = ''.join(group_columns_list)
    target_name = ''.join(target_columns_list)
    combine_name = [[grouped_name] + [method_name] + [target_name] for method_name in methods_list]

    df_new = frame.copy()
    grouped = df_new.groupby(group_columns_list)

    the_stats = grouped[target_name].agg(methods_list).reset_index()
    the_stats.columns = [grouped_name] + \
                        ['_%s_%s_by_%s' % (grouped_name, method_name, target_name) \
                         for (grouped_name, method_name, target_name) in combine_name]
    if keep_only_stats:
        return the_stats
    else:
        df_new = pd.merge(left=df_new, right=the_stats, on=group_columns_list, how='left')
    return df_new


def log_transform(frame, cols, verbose=1, merge_frame=True, drop_original=True):
    ''' Preform log transformation on selected columns of a dataframe

    :param frame:
    :param cols:
    :param verbose:
    :param merge_frame:
    :param drop_original:
    :return:
    '''
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