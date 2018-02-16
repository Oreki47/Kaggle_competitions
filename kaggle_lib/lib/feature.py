from lib.utilities import process_timer
from lib.preprocess import add_groupby_features_n_vs_1, log_transform
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
import pandas as pd


def check_feature_feasible(frame):
    # Check if there is null values
    if frame.isnull().any().any():
        namelist = frame.columns[frame.isnull().any()]
        print "There are null values in the DataFrame:",
        print ", ".join(namelist)
    else:
        print "There is no null values in the DataFrame"
    # Check is there is object type
    if ('object' in frame.dtypes):
        namelist = frame.columns[frame.dtype == 'object']
        print "There are object types in the DataFrame:",
        print ", ".join(namelist)
    else:
        print "There is no object type in the DataFrame"


def remove_id(frame):
    frame.drop('id', axis=1, inplace=True)
    return frame


def feature_engineering_1(x_train, x_test, y_train):
    with process_timer('feature engineering'):
        x_train, x_test = cat_smooth_encode(x_train, x_test, y_train)
        x_train, x_test = ind_bin_smoothing(x_train, x_test, y_train)
        x_train, x_test = int_smoothing(x_train, x_test, y_train)
        x_train, x_test = combo_features(x_train, x_test)

        x_train = bin_counts(x_train)
        x_test = bin_counts(x_test)

        x_train = reconstruct_reg_13(x_train)
        x_test = reconstruct_reg_13(x_test)

        x_train = reverse_one_hot(x_train)
        x_test = reverse_one_hot(x_test)

        x_train = car_float_transformation(x_train)
        x_test = car_float_transformation(x_test)

        x_train = calc_bin_transformation(x_train)
        x_test = calc_bin_transformation(x_test)

        x_train = extract_na_features(x_train)
        x_test = extract_na_features(x_test)

        x_train = remove_id(x_train)
        x_test = remove_id(x_test)

    return x_train, x_test


def feature_engineering_2(x_train, x_test, y_train):
    with process_timer('feature engineering'):
        x_train = remove_id(x_train)
        x_test = remove_id(x_test)

        drop_cols = ['ps_ind_14', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin']
        drop_cols = drop_cols + list(x_train.columns[x_train.columns.str.startswith('ps_calc')])

        cat_cols = x_train.columns[x_train.columns.str.endswith('cat')]

        x_train = drop_columns(x_train, drop_cols)
        x_test = drop_columns(x_test, drop_cols)

        x_train = nan_counts_2(x_train)
        x_test = nan_counts_2(x_test)

        x_train = ohe_general(x_train, cat_cols)
        x_test = ohe_general(x_test, cat_cols)

    return x_train, x_test


def feature_engineering_3(x_train, x_test, y_train):
    with process_timer('feature engineering'):
        d_median = x_train.median(axis=0)
        d_mean = x_train.mean(axis=0)

        x_train = remove_id(x_train)
        x_test = remove_id(x_test)

        drop_cols = list(x_train.columns[x_train.columns.str.startswith('ps_calc')])
        x_train = drop_columns(x_train, drop_cols)
        x_test = drop_columns(x_test, drop_cols)

        cat_cols = [i for i in x_train.columns if len(x_train[i].unique()) < 7 and len(x_train[i].unique()) > 2]
        range_cols = [
            "ps_ind_01", "ps_ind_03", "ps_ind_15", "ps_reg_01",
            "ps_reg_02", "ps_reg_03", "ps_car_12",
            "ps_car_13", "ps_car_14", "ps_car_15",]

        x_train = ohe_general(x_train, cat_cols)
        x_test = ohe_general(x_test, cat_cols)


        x_train = reconstruct_reg_13(x_train)
        x_test = reconstruct_reg_13(x_test)

        x_train['N_car_13_reg_03'] = x_train['ps_car_13'] * x_train['ps_reg_03']
        x_test['N_car_13_reg_03'] = x_test['ps_car_13'] * x_test['ps_reg_03']


        x_train = ranges(x_train, range_cols, d_median, d_mean)
        x_test = ranges(x_test, range_cols, d_median, d_mean)

    return x_train, x_test


def feature_engineering_4(x_train, x_test, y_train):
    with process_timer('feature engineering'):
        x_train, x_test = cat_smooth_encode(x_train, x_test, y_train)

        x_train = nonlinear_features(x_train)
        x_test = nonlinear_features(x_test)

        drop_cols = [
            'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin',
            'ps_ind_14', 'ps_ind_18_bin', 'ps_car_10_cat', 'ps_calc_01',
            'ps_calc_02', 'ps_calc_03', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06',
            'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_11', 'ps_calc_12',
            'ps_calc_13',
        ]
        x_train = drop_columns(x_train, drop_cols)
        x_test = drop_columns(x_test, drop_cols)

        x_train = calc_bin_transformation(x_train)
        x_test = calc_bin_transformation(x_test)

        check_feature_feasible(x_train)
        check_feature_feasible(x_test)

    return x_train, x_test


def feature_engineering_5(x_train, x_test, y_train):
    with process_timer('feature engineering'):
        x_train = remove_id(x_train)
        x_test = remove_id(x_test)

        drop_cols = [
            'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin',
            'ps_ind_14', 'ps_ind_18_bin', 'ps_car_10_cat', 'ps_calc_01',
            'ps_calc_02', 'ps_calc_03', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06',
            'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_11', 'ps_calc_12',
            'ps_calc_13',
        ]
        x_train = drop_columns(x_train, drop_cols)
        x_test = drop_columns(x_test, drop_cols)

    return x_train, x_test


def feature_engineering_6(x_train, x_test, y_train):
    with process_timer('feature engineering'):
        x_train = remove_id(x_train)
        x_test = remove_id(x_test)

        drop_cols = [
            'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin',
            'ps_ind_14', 'ps_ind_18_bin', 'ps_car_10_cat', 'ps_calc_01',
            'ps_calc_02', 'ps_calc_03', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06',
            'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_11', 'ps_calc_12',
            'ps_calc_13',
        ]
        x_train = drop_columns(x_train, drop_cols)
        x_test = drop_columns(x_test, drop_cols)

        log_cols = [
            'ps_ind_01', 'ps_ind_03', 'ps_ind_15', 'ps_car_12',
            'ps_car_13', 'ps_car_14', 'ps_car_15', 'ps_reg_01',
            'ps_reg_02', 'ps_reg_03',
        ]

        cat_cols = x_train.columns[x_train.columns.str.endswith('cat')]

        x_train = log_transform(x_train, log_cols)
        x_test = log_transform(x_test, log_cols)

        x_train = ohe_general(x_train, cat_cols)
        x_test = ohe_general(x_test, cat_cols)

        check_feature_feasible(x_train)
        check_feature_feasible(x_test)

    return x_train, x_test

def feature_engineering_7(x_train, x_test, y_train):
    with process_timer('feature engineering'):
        x_train = remove_id(x_train)
        x_test = remove_id(x_test)

        drop_cols = [
            'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin',
            'ps_ind_14', 'ps_ind_18_bin', 'ps_car_10_cat', 'ps_calc_01',
            'ps_calc_02', 'ps_calc_03', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06',
            'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_11', 'ps_calc_12',
            'ps_calc_13',
        ]
        x_train = drop_columns(x_train, drop_cols)
        x_test = drop_columns(x_test, drop_cols)

        log_cols = [
            'ps_ind_01', 'ps_ind_03', 'ps_ind_15', 'ps_car_12',
            'ps_car_13', 'ps_car_14', 'ps_car_15', 'ps_reg_01',
            'ps_reg_02', 'ps_reg_03',
        ]

        cat_cols = x_train.columns[x_train.columns.str.endswith('cat')]

        x_train = log_transform(x_train, log_cols)
        x_test = log_transform(x_test, log_cols)

        x_train = ohe_general(x_train, cat_cols)
        x_test = ohe_general(x_test, cat_cols)

        check_feature_feasible(x_train)
        check_feature_feasible(x_test)

    return x_train, x_test


def nonlinear_features(frame):
    frame['v001'] = frame["ps_ind_03"] + frame["ps_ind_14"] + np.square(frame["ps_ind_15"])
    frame['v002'] = frame["ps_ind_03"] + frame["ps_ind_14"] + np.tanh(frame["ps_ind_15"])
    frame['v003'] = frame["ps_reg_01"] + frame["ps_reg_02"] ** 3 + frame["ps_reg_03"]
    frame['v004'] = frame["ps_reg_01"] ** 2.15 + np.tanh(frame["ps_reg_02"]) + frame["ps_reg_03"] ** 3.1
    frame['v005'] = frame["ps_calc_01"] + frame["ps_calc_13"] + np.tanh(frame["ps_calc_14"])
    frame['v006'] = frame["ps_car_13"] + np.tanh(frame["v003"])
    frame['v007'] = frame["ps_car_13"] + frame["v002"] ** 2.7
    frame['v008'] = frame["ps_car_13"] + frame["v003"] ** 3.4
    frame['v009'] = frame["ps_car_13"] + frame["v004"] ** 3.1
    frame['v010'] = frame["ps_car_13"] + frame["v005"] ** 2.3

    frame['v011'] = frame["ps_ind_03"] ** 2.1 + frame["ps_ind_14"] ** 0.45 + frame["ps_ind_15"] ** 2.4
    frame['v012'] = frame["ps_ind_03"] ** 2.56 + frame["ps_calc_13"] ** 2.15 + frame["ps_reg_01"] ** 2.3
    frame['v013'] = frame["v003"] ** 2.15 + frame["ps_reg_01"] ** 2.49 + frame["ps_ind_15"] ** 2.14
    frame['v014'] = frame["v009"] ** 2.36 + frame["ps_calc_01"] ** 2.25 + frame["ps_reg_01"] ** 2.36
    frame['v015'] = frame["v003"] ** 3.21 + 0.001 * np.tanh(frame["ps_reg_01"]) + frame["ps_ind_15"] ** 3.12
    frame['v016'] = frame["v009"] ** 2.13 + 0.001 * np.tanh(frame["ps_calc_01"]) + frame["ps_reg_01"] ** 2.13
    frame['v017'] = frame["v016"] ** 2 + frame["v001"] ** 2.1 + frame["v003"] ** 2.3

    frame.loc[frame['v004'].isnull(), 'v004'] = -1
    frame.loc[frame['v008'].isnull(), 'v008'] = -1
    frame.loc[frame['v009'].isnull(), 'v009'] = -1
    frame.loc[frame['v013'].isnull(), 'v013'] = -1
    frame.loc[frame['v014'].isnull(), 'v014'] = -1
    frame.loc[frame['v015'].isnull(), 'v015'] = -1
    frame.loc[frame['v016'].isnull(), 'v016'] = -1
    frame.loc[frame['v017'].isnull(), 'v017'] = -1
    return frame


def ranges(frame, range_cols, d_median, d_mean):
    for col in range_cols:
        print col
        frame[col + 'med_range'] = (frame[col] > d_median[col]).astype(np.int)
        frame[col + 'mean_range'] = (frame[col] > d_mean[col]).astype(np.int)
    return frame


def ohe_general(frame, cols, threshold=50):
    for col in cols:
        temp = pd.get_dummies(pd.Series(frame[col]), prefix=col)
        _abort_cols = []
        for c in temp.columns:
            if temp[c].sum() < threshold:
                _abort_cols.append(c)
        _remain_cols = [c for c in temp.columns if c not in _abort_cols]
        # check category number
        frame = pd.concat([frame, temp[_remain_cols]], axis=1)

    frame = frame.drop(cols, axis=1)

    return frame


def drop_columns(frame, cols):
    frame.drop(cols, axis=1, inplace=True)
    return frame


def nan_counts_2(frame):
    frame['nan_counts'] = MinMaxScaler().fit_transform((frame == -1).sum(axis=1).values.reshape(-1, 1))
    return frame


def combo_features(x_train, x_test):
    combs = [
        ('ps_reg_01', 'ps_car_02_cat'),
        ('ps_reg_01', 'ps_car_04_cat'),
    ]


    for n_c, (f1, f2) in enumerate(combs):
        name1 = f1 + "_plus_" + f2
        x_train[name1] = x_train[f1].apply(lambda x: str(x)) + "_" + x_train[f2].apply(lambda x: str(x))
        x_test[name1] = x_test[f1].apply(lambda x: str(x)) + "_" + x_test[f2].apply(lambda x: str(x))
        # Label Encode
        lbl = LabelEncoder()
        lbl.fit(list(x_train[name1].values) + list(x_test[name1].values))
        x_train[name1] = lbl.transform(list(x_train[name1].values))
        x_test[name1] = lbl.transform(list(x_test[name1].values))

    return x_train, x_test


def int_smoothing(x_train, x_test, y_train):

    x_train['N_car_age'] = np.square(x_train['ps_car_15']).astype(np.int8)
    x_train['N_car_age'] = 17 - x_train['N_car_age']

    x_test['N_car_age'] = np.square(x_test['ps_car_15']).astype(np.int8)
    x_test['N_car_age'] = 17 - x_test['N_car_age']

    cols = ['ps_ind_01', 'ps_ind_03', 'ps_ind_14', 'ps_ind_15', 'N_car_age']

    for cat in cols:
        x_train[cat + "_avg"], x_test[cat + "_avg"] = target_encode_smoothing_after(
            trn_series=x_train[cat],
            tst_series=x_test[cat],
            target=y_train,
            min_samples_leaf=200,
            smoothing=10,
            noise_level=0)

    return x_train, x_test


def ind_bin_smoothing(x_train, x_test, y_train):
    x_train = reverse_one_hot(x_train)
    x_test = reverse_one_hot(x_test)

    cols = ['N_reverse_0609', 'N_reverse_1619', 'ps_ind_10_bin']

    for cat in cols:
        x_train[cat + "_avg"], x_test[cat + "_avg"] = target_encode_smoothing_after(
            trn_series=x_train[cat],
            tst_series=x_test[cat],
            target=y_train,
            min_samples_leaf=200,
            smoothing=10,
            noise_level=0)

    return x_train, x_test


def reverse_one_hot(frame):
    frame['N_reverse_0609'] = frame['ps_ind_08_bin'].astype(np.int8)
    frame.loc[frame['ps_ind_09_bin'] == 1, 'N_reverse_0609'] = 2
    frame.loc[frame['ps_ind_07_bin'] == 1, 'N_reverse_0609'] = 3
    frame.loc[frame['ps_ind_08_bin'] == 1, 'N_reverse_0609'] = 4

    frame['ps_ind_19_bin'] = 1
    frame['ps_ind_19_bin'] = frame['ps_ind_19_bin'] - \
         frame[['ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin']].sum(axis=1)

    frame['N_reverse_1619'] = frame['ps_ind_19_bin'].astype(np.int8)
    frame.loc[frame['ps_ind_17_bin'] == 1, 'N_reverse_1619'] = 2
    frame.loc[frame['ps_ind_18_bin'] == 1, 'N_reverse_1619'] = 3
    frame.loc[frame['ps_ind_16_bin'] == 1, 'N_reverse_1619'] = 4

    return frame


def car_float_transformation(frame):
    frame['N_car_cc'] = (np.round(frame['ps_car_12'], 4) * 10000).astype(np.int32)  # Cylinder Capacity
    frame['N_car_price'] = (np.round(np.square(frame['ps_car_13']), 4) * 90000).astype(np.int32)  # Price/Value
    frame['N_car_aver_mil'] = np.round(np.square(frame['ps_car_14']) * 90000).astype(np.int32)  # Aver Mil per year

    frame['N_car_age'] = np.square(frame['ps_car_15']).astype(np.int8)  # temp placeholder
    frame['N_car_age'] = 17 - frame['N_car_age']  # Age of the car

    # frame['N-Car-Year-Log'] = boxcox(frame['N-Car-Year']).astype(np.float32) # not useful?

    return frame


def calc_bin_transformation(frame):
    temp = frame['ps_calc_15_bin'] * 32 + \
           frame['ps_calc_16_bin'] * 16 + \
           frame['ps_calc_17_bin'] * 8 + \
           frame['ps_calc_18_bin'] * 4 + \
           frame['ps_calc_19_bin'] * 2 + \
           frame['ps_calc_20_bin'] * 1
    temp2 = [5, 22, 9, 32, 13, 38, 20, 47, 2, 19, 8, 30, 10, 35, 17, 45, 1,
             15, 4, 24, 7, 29, 14, 40, 0, 12, 3, 21, 6, 26, 11, 36, 27, 52,
             37, 57, 42, 60, 51, 63, 23, 49, 34, 56, 39, 59, 48, 62, 18, 46,
             28, 53, 33, 55, 44, 61, 16, 43, 25, 50, 31, 54, 41, 58]
    temp2 = pd.Series(temp2)
    frame['N_calc_bin'] = temp.map(temp2)

    frame.drop(['ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin',
                'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin'], axis=1, inplace=True)

    return frame


def extract_na_features(frame):
    # if any of 02, 04, 05 ind is nan
    ind_cols = ['ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat']
    frame['N_nan_ind_any'] = (frame.loc[:, ind_cols] == -1).any(axis=1)  # max_corr =0.1
    frame['N_nan_ind_count'] = np.sum((frame.loc[:, ind_cols] == -1).astype(np.int8), axis=1)

    # if any of 01, 07, 09 car is nan
    car_cols_01 = ['ps_car_01_cat', 'ps_car_07_cat', 'ps_car_09_cat']
    frame['N_nan_car_any_01'] = (frame.loc[:, car_cols_01] == -1).any(axis=1)  # max_corr = 0.75
    frame['N_nan_car_count_01'] = np.sum((frame.loc[:, car_cols_01] == -1).astype(np.int8), axis=1)

    # if any of 03, 05 car is nan
    car_cols_02 = ['ps_car_03_cat', 'ps_car_05_cat']
    frame['N_nan_car_any_02'] = (frame.loc[:, car_cols_02] == -1).any(axis=1)  # max_corr = 0.9
    frame['N_nan_car_count_02'] = np.sum((frame.loc[:, car_cols_02] == -1).astype(np.int8), axis=1)

    # number of bad nan - good nan
    frame['N_nan_bad_minus_good'] = frame['N_nan_ind_count'] + \
                                    frame['N_nan_car_count_01'] - \
                                    frame['N_nan_car_count_02']

    return frame


def reconstruct_reg_13(frame):
    I = np.round((40 * frame['ps_reg_03']) ** 2)
    I = I.astype(int)
    M = (I - 1) // 27
    F = I - 27 * M
    frame['ps_reg_M'] = M
    frame['ps_reg_F'] = F

    frame.loc[frame['ps_reg_03'] == -1, ('ps_reg_F', 'ps_reg_M')] = -1

    return frame


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode_smoothing_after(trn_series, tst_series, target, min_samples_leaf=1, smoothing=1, noise_level=0):
    # This is a variation of likelihood/mean encoding which seems to be better by far
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name

    temp = pd.concat([trn_series, target], axis=1)
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    prior = target.mean()

    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)

    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    ft_trn_series.index = trn_series.index

    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    ft_tst_series.index = tst_series.index

    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)


def cat_smooth_encode(x_train, x_test, y_train):
    cat_cols = x_train.columns[x_train.columns.str.endswith('cat')]

    for cat in cat_cols:
        x_train[cat + "_avg"], x_test[cat + "_avg"] = target_encode_smoothing_after(
            trn_series=x_train[cat],
            tst_series=x_test[cat],
            target=y_train,
            min_samples_leaf=200,
            smoothing=10,
            noise_level=0)

    return x_train, x_test


def cat_mean_encode(x_train, x_test, y_train):
    cat_cols = x_train.columns[x_train.columns.str.endswith('cat')]
    x_train = pd.concat([x_train, y_train], axis=1)

    for cat in cat_cols:
        stats_cols = add_groupby_features_n_vs_1(x_train, [cat], ['target'], ['mean', 'count'])
        x_train = pd.merge(left=x_train, right=stats_cols, on=cat, how='left')
        x_test = pd.merge(left=x_test, right=stats_cols, on=cat, how='left')

    x_train.drop('target', axis=1, inplace=True)
    return x_train, x_test


def bin_counts(frame):
    ind_cols = frame.columns[np.logical_and(
        frame.columns.str.startswith('ps_ind'),
        frame.columns.str.endswith('_bin')
    )]
    calc_cols = frame.columns[np.logical_and(
        frame.columns.str.startswith('ps_calc'),
        frame.columns.str.endswith('_bin')
    )]

    bin_cols = frame.columns[frame.columns.str.endswith('_bin')]

    frame['N_Binary_Count_ind'] = frame[ind_cols].sum(axis=1)
    frame['N_Binary_Count_calc'] = frame[calc_cols].sum(axis=1)
    frame['N_Binary_Count'] = frame[bin_cols].sum(axis=1)

    return frame


def reg_mix(frame):

    frame['N_car_02_x_reg_01'] = frame['ps_car_02_cat'] * frame['ps_reg_01']
    frame['N_car_13_x_ps_reg_03'] = frame['ps_car_13'] * frame['ps_reg_03']
    frame['N_reg01_x_02'] = frame['ps_reg_01'] * frame['ps_reg_02']
    frame['N_reg01_2'] = np.square(frame['ps_reg_01'])
    frame['N_reg02_2'] = np.square(frame['ps_reg_02'])
    frame['N_reg01_x_03'] = frame['ps_reg_01'] * frame['ps_reg_03']
    frame['N_reg02_x_03'] = frame['ps_reg_02'] * frame['ps_reg_03']

    return frame


# def bin_diff_features(zip_data):
#     frame = zip_data[0]
#     ind_ref = zip_data[1]
#     calc_ref = zip_data[2]
#
#     ind_cols = frame.columns[np.logical_and(
#         frame.columns.str.startswith('ps_ind'),
#         frame.columns.str.endswith('_bin')
#     )]
#     calc_cols = frame.columns[np.logical_and(
#         frame.columns.str.startswith('ps_calc'),
#         frame.columns.str.endswith('_bin')
#     )]
#
#     bin_cols = ind_cols + calc_cols
#
#     frame['N-Binary-Diff-ind'] = frame[ind_cols].apply(lambda x: (x.astype(np.int8) - ind_ref).abs().sum(), axis=1)
#     frame['N-Binary-Diff-calc'] = frame[calc_cols].apply(lambda x: (x.astype(np.int8) - calc_ref).abs().sum(), axis=1)
#
#     return frame
#
#
# def multi_bin_diff_features(frame):
#     ind_cols = frame.columns[np.logical_and(
#         frame.columns.str.startswith('ps_ind'),
#         frame.columns.str.endswith('_bin')
#     )]
#     calc_cols = frame.columns[np.logical_and(
#         frame.columns.str.startswith('ps_calc'),
#         frame.columns.str.endswith('_bin')
#     )]
#
#     ind_ref = frame.loc[0, ind_cols]
#     calc_ref = frame.loc[0, calc_cols]
#
#     pool = Pool(cpu_count())
#     frame = pool.map(bin_diff_features,
#                      zip(np.array_split(frame, cpu_count()),
#                          itertools.repeat(ind_ref),
#                          itertools.repeat(calc_ref)))
#     frame = pd.concat(frame, axis=0, ignore_index=True).reset_index(drop=True)
#     pool.close()
#     pool.join()
#     return frame

unused = [
        'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin',
        'ps_ind_14', 'ps_ind_18_bin',

        'ps_car_10_cat'
        
        'ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_calc_04', 'ps_calc_05',
        'ps_calc_06', 'ps_calc_07', 'ps_calc_08', 'ps_calc_09',
        'ps_calc_11', 'ps_calc_12', 'ps_calc_13',

        # Smooth encode
        'ps_ind_02_cat_avg', 'ps_ind_04_cat_avg', 'ps_car_01_cat_avg', 'ps_ind_05_cat_avg',
        'ps_car_03_cat_avg', 'ps_car_04_cat_avg', 'ps_car_11_cat_avg',
        'ps_car_06_cat_avg', 'ps_car_07_cat_avg', 'ps_car_09_cat_avg',
        'ps_car_02_cat_avg', 'ps_car_05_cat_avg', 'ps_car_08_cat_avg', 'ps_car_10_cat_avg',  # low score

        # Mean encode
        '_ps_car_11_cat_count_by_target', '_ps_ind_02_cat_mean_by_target',
        '_ps_ind_02_cat_count_by_target', '_ps_ind_04_cat_mean_by_target',
        '_ps_ind_04_cat_count_by_target', '_ps_ind_05_cat_mean_by_target',
        '_ps_ind_05_cat_count_by_target', '_ps_car_01_cat_mean_by_target',
        '_ps_car_01_cat_count_by_target', '_ps_car_02_cat_mean_by_target',
        '_ps_car_02_cat_count_by_target', '_ps_car_03_cat_mean_by_target',
        '_ps_car_03_cat_count_by_target', '_ps_car_04_cat_mean_by_target',
        '_ps_car_04_cat_count_by_target', '_ps_car_05_cat_mean_by_target',
        '_ps_car_05_cat_count_by_target', '_ps_car_06_cat_mean_by_target',
        '_ps_car_06_cat_count_by_target', '_ps_car_07_cat_mean_by_target',
        '_ps_car_07_cat_count_by_target', '_ps_car_08_cat_mean_by_target',
        '_ps_car_08_cat_count_by_target', '_ps_car_09_cat_mean_by_target',
        '_ps_car_09_cat_count_by_target', '_ps_car_10_cat_mean_by_target',
        '_ps_car_10_cat_count_by_target', '_ps_car_11_cat_mean_by_target',

        'ps_reg_M', 'ps_reg_F',  # reg_03 reconstruction

        'N_nan_bad_minus_good',  # Nan transfer

        'N_Binary_Count', 'N_Binary_Count_ind',  # Binary count

        # Reverse one hot
        'N-reverse-LabelCount-1', 'N-reverse-LabelCount-2',
        'N-reverse-Target-1', 'N-reverse-Target-2', 'ps_ind_19_bin',  # low score

        # Car float
        'N_car_price', 'N_car_aver_mil',
        'N_car_age', 'N_car_cc',  # low score

        'N_nan_ind_any', 'N_nan_ind_count', 'N_nan_car_any_01',  # low score
        'N_nan_car_count_01', 'N_nan_car_count_02', 'N_nan_car_any_02',  # low score
]
