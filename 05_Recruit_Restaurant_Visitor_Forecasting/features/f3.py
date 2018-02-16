import gc
import sys
import numpy as np
import pandas as pd

sys.path.append("../")
from general.utilities import log1p_transform, lb_en_transform, add_groupby_features, merge_numericals


def prepare_data():
    data = {
        'ar': pd.read_csv('../data/air_reserve.csv'),
        'as': pd.read_csv('../data/air_store_info.csv'),
        'hs': pd.read_csv('../data/hpg_store_info.csv'),
        'trn': pd.read_csv('../data/air_visit_data.csv'),  # with the visitors column, which is the target
        'hr': pd.read_csv('../data/hpg_reserve.csv'),
        'id': pd.read_csv('../data/store_id_relation.csv'),
        'tst': pd.read_csv('../data/sample_submission.csv'),
        'hol': pd.read_csv('../data/date_info.csv').rename(columns={'calendar_date': 'visit_date'})  # advanced features
    }
    # TODO: with aggregation on visitors, do we need a fix?
    # basic_features/

    # Basic Operations
    ntrain = data['trn'].shape[0]
    ntest = data['tst'].shape[0]
    do_basics(data)

    # Add features Sequentially
    add_visit_time_features(data, ntrain)
    add_air_reserve_features(data)
    add_hpg_reserve_features(data)

    # store features
    add_air_store_feature(data)
    add_hpg_store_feature(data)

    # holiday
    add_holiday_feature(data)

    merge_frames(data)

    # add air/hpg combined features
    add_full_data_feature(data, ntrain)

    # lop1p transform
    log1p_cols = data['full'].columns[
        (data['full'].dtypes == 'float64').values &\
        (data['full'].columns.str.contains('_visitors') | data['full'].columns.str.contains('reserve_ahead'))
    ]
    log1p_transform(data['full'], log1p_cols, drop=True)

    # categorical label-encode
    cat_cols = ['air_genre_name', 'air_lv1', 'air_lv2', 'air_lv3',
            'hpg_genre_name', 'hpg_lv1', 'hpg_lv2', 'hpg_lv3']
    lb_en_transform(data['full'], cat_cols, drop=True)

    data['full'].drop('hpg_store_id', axis=1, inplace=True)
    data['full'].drop('air_area_name', axis=1, inplace=True)
    data['full'].drop('hpg_area_name', axis=1, inplace=True)

    # add log1p transform
    data['full']['visitors'] = np.log1p(data['full']['visitors'])
    frame = data['full']

    del data
    gc.collect()

    features = [f for f in frame if f not in ['id', 'air_store_id', 'visit_date','visitors']]
    cat_feats = ['day', 'month', 'year', 'dow', 'doy', 'woy', 'holiday_flg']
    bool_feats = ['is_weekend', 'is_golden_week', 'is_month_end', 'is_month_str']

    return frame, ntrain, ntest, features, cat_feats + bool_feats


def do_basics(data):
    data['trn']['id'] = data['trn']['air_store_id'] + "_" + data['trn']['visit_date']
    data['tst']['visitors'] = np.nan
    data['tst']['air_store_id'] = data['tst']['id'].apply(lambda x: "_".join(x.split('_')[:2]))
    data['tst']['visit_date'] = data['tst']['id'].apply(lambda x: x.split('_')[2])
    # Concat train/test
    data['full'] = pd.concat([data['trn'], data['tst']])


def add_visit_time_features(data: dict, ntrain):
    # Note : no null value generated from this operation

    # Add features
    data['full']['visit_date'] = pd.to_datetime(data['full']['visit_date'])
    data['full']['day'] = data['full']['visit_date'].dt.day
    data['full']['month'] = data['full']['visit_date'].dt.month
    data['full']['year'] = data['full']['visit_date'].dt.year
    data['full']['dow'] = data['full']['visit_date'].dt.dayofweek
    data['full']['doy'] = data['full']['visit_date'].dt.dayofyear  # required np.log1p
    data['full']['woy'] = data['full']['visit_date'].dt.weekofyear
    data['full']['is_weekend'] = data['full']['dow'].isin([4, 5, 6])  # Friday/Saturday/Sunday
    data['full']['is_golden_week'] = data['full']['visit_date'].between('04-29-2016', '05-07-2016') | \
                                     data['full']['visit_date'].between('04-29-2017', '05-07-2017')  # 4/29 - 5/7
    data['full']['is_month_end'] = data['full']['visit_date'].dt.is_month_end
    data['full']['is_month_str'] = data['full']['visit_date'].dt.is_month_start

    # aggregation by visitors on various time-related features (share the same method_dict)
    method_dict = {'visitors': ['mean', 'max', 'min', 'median', 'count']}

    groupby_cols = ['dow']
    data['full'] = add_groupby_features(data['full'][:ntrain], data['full'], groupby_cols, method_dict)

    groupby_cols = ['is_weekend']
    data['full'] = add_groupby_features(data['full'][:ntrain], data['full'], groupby_cols, method_dict)

    groupby_cols = ['is_golden_week']
    data['full'] = add_groupby_features(data['full'][:ntrain], data['full'], groupby_cols, method_dict)

    groupby_cols = ['is_month_str']
    data['full'] = add_groupby_features(data['full'][:ntrain], data['full'], groupby_cols, method_dict)

    # aggregation by visitors on store
    groupby_cols = ['air_store_id']
    data['full'] = add_groupby_features(data['full'][:ntrain], data['full'], groupby_cols, method_dict)

    # agg on both
    groupby_cols = ['dow', 'air_store_id']
    data['full'] = add_groupby_features(data['full'][:ntrain], data['full'], groupby_cols, method_dict)

    groupby_cols = ['is_golden_week', 'air_store_id']
    data['full'] = add_groupby_features(data['full'][:ntrain], data['full'], groupby_cols, method_dict)

    # cast to str
    data['full']['visit_date'] = data['full']['visit_date'].dt.date.apply(lambda x: str(x))


def add_air_store_feature(data):
    # air_store handle
    data['as']['air_lv1'] = data['as'].air_area_name.apply(lambda x: x.split(" ")[0])
    data['as']['air_lv2'] = data['as'].air_area_name.apply(lambda x: x.split(" ")[1])
    data['as']['air_lv3'] = data['as'].air_area_name.apply(lambda x: "".join(x.split(" ")[2:]))


    # Count aggregation on various geo-related features
    method_dict = {'air_store_id': ['count']}

    groupby_cols = ['latitude', 'longitude']
    data['as'] = add_groupby_features(data['as'], data['as'], groupby_cols, method_dict)

    groupby_cols = ['air_lv1']
    data['as'] = add_groupby_features(data['as'], data['as'], groupby_cols, method_dict)

    groupby_cols = ['air_lv1', 'air_lv2']
    data['as'] = add_groupby_features(data['as'], data['as'], groupby_cols, method_dict)

    groupby_cols = ['air_lv1', 'air_lv2', 'air_lv3']
    data['as'] = add_groupby_features(data['as'], data['as'], groupby_cols, method_dict)

    groupby_cols = ['air_genre_name']
    data['as'] = add_groupby_features(data['as'], data['as'], groupby_cols, method_dict)

    groupby_cols = ['air_genre_name', 'air_lv1']
    data['as'] = add_groupby_features(data['as'], data['as'], groupby_cols, method_dict)

    groupby_cols = ['air_genre_name', 'air_lv1', 'air_lv2']
    data['as'] = add_groupby_features(data['as'], data['as'], groupby_cols, method_dict)

    groupby_cols = ['air_genre_name', 'air_lv1', 'air_lv2', 'air_lv3']
    data['as'] = add_groupby_features(data['as'], data['as'], groupby_cols, method_dict)

    # group locations on different geo-levels
    method_dict = {'latitude': ['mean', 'max', 'min'],
                   'longitude': ['mean', 'max', 'min']}

    groupby_cols = ['air_lv1']
    data['as'] = add_groupby_features(data['as'], data['as'], groupby_cols, method_dict)

    groupby_cols = ['air_lv2']
    data['as'] = add_groupby_features(data['as'], data['as'], groupby_cols, method_dict)

    groupby_cols = ['air_lv3']
    data['as'] = add_groupby_features(data['as'], data['as'], groupby_cols, method_dict)


def add_hpg_store_feature(data):
    data['hs']['hpg_lv1'] = data['hs'].hpg_area_name.apply(lambda x: x.split(" ")[0])
    data['hs']['hpg_lv2'] = data['hs'].hpg_area_name.apply(lambda x: x.split(" ")[1])
    data['hs']['hpg_lv3'] = data['hs'].hpg_area_name.apply(lambda x: "".join(x.split(" ")[2:]))

    # Count aggregation on various geo-related features
    method_dict = {'hpg_store_id': ['count']}

    groupby_cols = ['latitude', 'longitude']
    data['hs'] = add_groupby_features(data['hs'], data['hs'], groupby_cols, method_dict)

    groupby_cols = ['hpg_lv1']
    data['hs'] = add_groupby_features(data['hs'], data['hs'], groupby_cols, method_dict)

    groupby_cols = ['hpg_lv1', 'hpg_lv2']
    data['hs'] = add_groupby_features(data['hs'], data['hs'], groupby_cols, method_dict)

    groupby_cols = ['hpg_lv1', 'hpg_lv2', 'hpg_lv3']
    data['hs'] = add_groupby_features(data['hs'], data['hs'], groupby_cols, method_dict)

    groupby_cols = ['hpg_genre_name']
    data['hs'] = add_groupby_features(data['hs'], data['hs'], groupby_cols, method_dict)

    groupby_cols = ['hpg_genre_name', 'hpg_lv1']
    data['hs'] = add_groupby_features(data['hs'], data['hs'], groupby_cols, method_dict)

    groupby_cols = ['hpg_genre_name', 'hpg_lv1', 'hpg_lv2']
    data['hs'] = add_groupby_features(data['hs'], data['hs'], groupby_cols, method_dict)

    groupby_cols = ['hpg_genre_name', 'hpg_lv1', 'hpg_lv2', 'hpg_lv3']
    data['hs'] = add_groupby_features(data['hs'], data['hs'], groupby_cols, method_dict)

    # group locations on different geo-levels
    method_dict = {'latitude': ['mean', 'max', 'min'],
                   'longitude': ['mean', 'max', 'min']}

    groupby_cols = ['hpg_lv1']
    data['hs'] = add_groupby_features(data['hs'], data['hs'], groupby_cols, method_dict)

    groupby_cols = ['hpg_lv2']
    data['hs'] = add_groupby_features(data['hs'], data['hs'], groupby_cols, method_dict)

    groupby_cols = ['hpg_lv3']
    data['hs'] = add_groupby_features(data['hs'], data['hs'], groupby_cols, method_dict)


def add_air_reserve_features(data):
    # reservation correlated to store | time, and store & time

    # reserve data handle
    data['ar']['visit_datetime'] = pd.to_datetime(data['ar']['visit_datetime'])
    data['ar']['reserve_datetime'] = pd.to_datetime(data['ar']['reserve_datetime'])
    data['ar']['visit_date'] = data['ar']['visit_datetime'].dt.date
    data['ar']['visit_time'] = data['ar']['visit_datetime'].dt.hour
    data['ar']['reserve_date'] = data['ar']['reserve_datetime'].dt.date
    data['ar']['reserve_time'] = data['ar']['reserve_datetime'].dt.hour
    data['ar']['reserve_to_visit_dow'] = data['ar']['visit_datetime'].dt.dayofweek
    data['ar']['reserve_ahead_in_days'] = (data['ar']['visit_date'] - data['ar']['reserve_date']).dt.days

    data['ar']['reserve_date'] = data['ar']['reserve_date'].apply(lambda x: str(x))
    data['ar']['visit_date'] = data['ar']['visit_date'].apply(lambda x: str(x))

    # reserve_visitors aggregate on store
    agg_func = lambda x: x.value_counts().index[0]
    agg_func.__name__ = 'mode'
    groupby_cols = ['air_store_id']
    method_dict = {'reserve_visitors': ['mean', 'max', 'min', 'count', 'sum'],
                   'reserve_ahead_in_days': ['mean', 'max', 'min'],
                   'reserve_to_visit_dow': [agg_func]}

    data['full'] = add_groupby_features(data['ar'], data['full'], groupby_cols, method_dict)

    # reserve_visitors aggregate on date
    groupby_cols = ['visit_date']
    method_dict = {'reserve_visitors': ['mean', 'max', 'min', 'count', 'sum'],
                   'reserve_ahead_in_days': ['mean', 'max', 'min'], }

    data['full'] = add_groupby_features(data['ar'], data['full'], groupby_cols, method_dict)

    # reserve_visitors aggregate on store & date
    groupby_cols = ['visit_date', 'air_store_id']
    method_dict = {'reserve_visitors': ['mean', 'max', 'min', 'count', 'sum'],
                   'reserve_ahead_in_days': ['mean', 'max', 'min'], }

    data['full'] = add_groupby_features(data['ar'], data['full'], groupby_cols, method_dict)


def add_hpg_reserve_features(data):
    # hpg reservation handle
    data['hr']['visit_datetime'] = pd.to_datetime(data['hr']['visit_datetime'])
    data['hr']['reserve_datetime'] = pd.to_datetime(data['hr']['reserve_datetime'])
    data['hr']['visit_date'] = data['hr']['visit_datetime'].dt.date
    data['hr']['visit_time'] = data['hr']['visit_datetime'].dt.hour
    data['hr']['reserve_date'] = data['hr']['reserve_datetime'].dt.date
    data['hr']['reserve_time'] = data['hr']['reserve_datetime'].dt.hour
    data['hr']['reserve_to_visit_dow'] = data['hr']['visit_datetime'].dt.dayofweek
    data['hr']['reserve_ahead_in_days'] = (data['hr']['visit_date'] - data['hr']['reserve_date']).dt.days

    data['hr']['reserve_date'] = data['hr']['reserve_date'].apply(lambda x: str(x))
    data['hr']['visit_date'] = data['hr']['visit_date'].apply(lambda x: str(x))

    data['hr_temp'] = data['hr'].groupby(['hpg_store_id', 'visit_date']). \
        agg({'reserve_visitors': 'count'}).reset_index()
    data['hr_temp'] = pd.merge(left=data['id'], right=data['hr_temp'], on='hpg_store_id', how='left')

    agg_func = lambda x: x.value_counts().index[0]
    agg_func.__name__ = 'mode'
    groupby_cols = ['hpg_store_id']
    method_dict = {'reserve_visitors': ['mean', 'max', 'min', 'count', 'sum'],
                   'reserve_ahead_in_days': ['mean', 'max', 'min'],
                   'reserve_to_visit_dow': [agg_func]}

    data['hr_temp'] = add_groupby_features(data['hr'], data['hr_temp'], groupby_cols, method_dict)

    # reserve_visitors aggregate on date
    groupby_cols = ['visit_date']
    method_dict = {'reserve_visitors': ['mean', 'max', 'min', 'count', 'sum'],
                   'reserve_ahead_in_days': ['mean', 'max', 'min'], }

    data['hr_temp'] = add_groupby_features(data['hr'], data['hr_temp'], groupby_cols, method_dict)

    # reserve_visitors aggregate on store & date
    groupby_cols = ['visit_date', 'hpg_store_id']
    method_dict = {'reserve_visitors': ['mean', 'max', 'min', 'count', 'sum'],
                   'reserve_ahead_in_days': ['mean', 'max', 'min'], }

    data['hr_temp'] = add_groupby_features(data['hr'], data['hr_temp'], groupby_cols, method_dict)


def add_holiday_feature(data):
    # holiday and weight of each day
    data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
    data['hol']['weight2'] = (data['hol'].index + 1) / len(data['hol']) ** 2
    data['hol']['weight3'] = (data['hol'].index + 1) / len(data['hol']) ** 3
    data['hol']['weight4'] = (data['hol'].index + 1) / len(data['hol']) ** 4
    data['hol']['weight5'] = (data['hol'].index + 1) / len(data['hol']) ** 5
    data['hol']['visit_date'] = data['hol']['visit_date'].dt.date
    data['hol']['visit_date'] = data['hol']['visit_date'].apply(lambda x: str(x))
    data['hol'].drop('day_of_week', inplace=True, axis=1)


def merge_frames(data):
    data['hs'] = pd.merge(left=data['id'], right=data['hs'], on='hpg_store_id', how='left')
    data['hs'].drop("hpg_store_id", axis=1, inplace=True)

    data['as'] = pd.merge(left=data['as'], right=data['hs'],
                          on='air_store_id', how='left', suffixes=('_air', '_hpg'))
    data['full'] = pd.merge(left=data['full'], right=data['hr_temp'],
                            on=['air_store_id', 'visit_date'], how='left', suffixes=('_air', '_hpg'))
    data['full'] = pd.merge(left=data['full'], right=data['as'], on=['air_store_id'], how='left')
    data['full'] = pd.merge(left=data['full'], right=data['hol'], on='visit_date', how='left')


def add_full_data_feature(data, ntrain):
    # More features
    merge_dict = {
        'GP_air_store_id_ON_reserve_visitors_mean': 'mean',
        'GP_air_store_id_ON_reserve_visitors_max': 'max',
        'GP_air_store_id_ON_reserve_visitors_min': 'min',
        'GP_air_store_id_ON_reserve_visitors_count': 'sum',
        'GP_air_store_id_ON_reserve_visitors_sum': 'sum',
        'GP_air_store_id_ON_reserve_ahead_in_days_mean': 'mean',
        'GP_air_store_id_ON_reserve_ahead_in_days_max': 'max',
        'GP_air_store_id_ON_reserve_ahead_in_days_min': 'min',
        'GP_visit_date_ON_reserve_visitors_mean_air': 'mean',
        'GP_visit_date_ON_reserve_visitors_max_air': 'max',
        'GP_visit_date_ON_reserve_visitors_min_air': 'min',
        'GP_visit_date_ON_reserve_visitors_count_air': 'sum',
        'GP_visit_date_ON_reserve_visitors_sum_air': 'sum',
        'GP_visit_date_ON_reserve_ahead_in_days_mean_air': 'mean',
        'GP_visit_date_ON_reserve_ahead_in_days_max_air': 'max',
        'GP_visit_date_ON_reserve_ahead_in_days_min_air': 'min',
        'GP_visit_date_air_store_id_ON_reserve_visitors_mean': 'mean',
        'GP_visit_date_air_store_id_ON_reserve_visitors_max': 'max',
        'GP_visit_date_air_store_id_ON_reserve_visitors_min': 'min',
        'GP_visit_date_air_store_id_ON_reserve_visitors_count': 'sum',
        'GP_visit_date_air_store_id_ON_reserve_visitors_sum': 'sum',
        'GP_visit_date_air_store_id_ON_reserve_ahead_in_days_mean': 'mean',
        'GP_visit_date_air_store_id_ON_reserve_ahead_in_days_max': 'max',
        'GP_visit_date_air_store_id_ON_reserve_ahead_in_days_min': 'min',
    }

    merge_numericals(data['full'], merge_dict, drop=True)

    data['full']['lon_plus_lat_air'] = data['full']['longitude_air'] + data['full']['latitude_air']

    data['full']['lat_to_mean_lat_air_lv1'] = abs(
        data['full']['latitude_air'] - data['full']['GP_air_lv1_ON_latitude_mean'])
    data['full']['lat_to_max_lat_air_lv1'] = data['full']['latitude_air'] - data['full']['GP_air_lv1_ON_latitude_max']
    data['full']['lat_to_min_lat_air_lv1'] = data['full']['latitude_air'] - data['full']['GP_air_lv1_ON_latitude_min']
    data['full']['lon_to_mean_lon_air_lv1'] = abs(
        data['full']['longitude_air'] - data['full']['GP_air_lv1_ON_longitude_mean'])
    data['full']['lon_to_max_lon_air_lv1'] = data['full']['longitude_air'] - data['full']['GP_air_lv1_ON_longitude_max']
    data['full']['lon_to_min_lon_air_lv1'] = data['full']['longitude_air'] - data['full']['GP_air_lv1_ON_longitude_min']
    data['full']['lat_to_mean_lat_air_lv2'] = abs(
        data['full']['latitude_air'] - data['full']['GP_air_lv2_ON_latitude_mean'])
    data['full']['lat_to_max_lat_air_lv2'] = data['full']['latitude_air'] - data['full']['GP_air_lv2_ON_latitude_max']
    data['full']['lat_to_min_lat_air_lv2'] = data['full']['latitude_air'] - data['full']['GP_air_lv2_ON_latitude_min']
    data['full']['lon_to_mean_lon_air_lv2'] = abs(
        data['full']['longitude_air'] - data['full']['GP_air_lv2_ON_longitude_mean'])
    data['full']['lon_to_max_lon_air_lv2'] = data['full']['longitude_air'] - data['full']['GP_air_lv2_ON_longitude_max']
    data['full']['lon_to_min_lon_air_lv2'] = data['full']['longitude_air'] - data['full']['GP_air_lv2_ON_longitude_min']

    data['full']['lat_to_mean_lat_hpg_lv1'] = abs(
        data['full']['latitude_hpg'] - data['full']['GP_hpg_lv1_ON_latitude_mean'])
    data['full']['lat_to_max_lat_hpg_lv1'] = data['full']['latitude_hpg'] - data['full']['GP_hpg_lv1_ON_latitude_max']
    data['full']['lat_to_min_lat_hpg_lv1'] = data['full']['latitude_hpg'] - data['full']['GP_hpg_lv1_ON_latitude_min']
    data['full']['lon_to_mean_lon_hpg_lv1'] = abs(
        data['full']['longitude_hpg'] - data['full']['GP_hpg_lv1_ON_longitude_mean'])
    data['full']['lon_to_max_lon_hpg_lv1'] = data['full']['longitude_hpg'] - data['full']['GP_hpg_lv1_ON_longitude_max']
    data['full']['lon_to_min_lon_hpg_lv1'] = data['full']['longitude_hpg'] - data['full']['GP_hpg_lv1_ON_longitude_min']
    data['full']['lat_to_mean_lat_hpg_lv2'] = abs(
        data['full']['latitude_hpg'] - data['full']['GP_hpg_lv2_ON_latitude_mean'])
    data['full']['lat_to_max_lat_hpg_lv2'] = data['full']['latitude_hpg'] - data['full']['GP_hpg_lv2_ON_latitude_max']
    data['full']['lat_to_min_lat_hpg_lv2'] = data['full']['latitude_hpg'] - data['full']['GP_hpg_lv2_ON_latitude_min']
    data['full']['lon_to_mean_lon_hpg_lv2'] = abs(
        data['full']['longitude_hpg'] - data['full']['GP_hpg_lv2_ON_longitude_mean'])
    data['full']['lon_to_max_lon_hpg_lv2'] = data['full']['longitude_hpg'] - data['full']['GP_hpg_lv2_ON_longitude_max']
    data['full']['lon_to_min_lon_hpg_lv2'] = data['full']['longitude_hpg'] - data['full']['GP_hpg_lv2_ON_longitude_min']

    method_dict = {'visitors': ['mean', 'max', 'min', 'median']}

    groupby_cols = ['air_store_id', 'dow', 'holiday_flg']
    data['full'] = add_groupby_features(data['full'][:ntrain], data['full'], groupby_cols, method_dict)

    groupby_cols = ['air_store_id', 'is_golden_week']
    data['full'] = add_groupby_features(data['full'][:ntrain], data['full'], groupby_cols, method_dict)

    agg_func = lambda x: ((x[1] * x[0]).sum() / x[1])


    # cast datetimes from object to datetime
    data['full'].visit_date = pd.to_datetime(data['full'].visit_date)
