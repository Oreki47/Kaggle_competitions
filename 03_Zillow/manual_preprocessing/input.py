import pandas as pd
import numpy as np
from manual_preprocessing.downcast import manual_downcast
from manual_preprocessing.dtype_string import replace_binary_string_with_boolean, label_encode
from manual_preprocessing.features import add_features, add_date_features
from kaggle_lib.Utils.KA_utils import process_tracker
from kaggle_lib.Preprocessing.predictor_checkers import check_feature_feasible
import gc


def load_data(year, verbose=False, nrows=None):
    # ===============================================================
    with process_tracker("Load files", verbose):
    # ===============================================================
        # Doing split process for now.
        if year == 2016:
            prop_dir = 'data/properties_2016.csv'
            train_dir = 'data/train_2016_v2.csv'
        else:
            prop_dir = 'data/properties_2017.csv'
            train_dir = 'data/train_2017.csv'
        prop = pd.read_csv(prop_dir, nrows=nrows)
        train = pd.read_csv(train_dir, nrows=nrows,
                            dtype={'parcelid': np.uint32,
                                   'logerror': np.float32},
                            parse_dates=['transactiondate'])
        sample = pd.read_csv('data/sample_submission.csv',
                             dtype={'parcelid': np.uint32,
                                    '201610': np.uint8,
                                    '201611': np.uint8,
                                    '201612': np.uint8,
                                    '201710': np.uint8,
                                    '201711': np.uint8,
                                    '201712': np.uint8,})
    # ===============================================================
    # Drop outliers
    # ===============================================================
    train = train[train.logerror > -0.3]
    train = train[train.logerror < 0.35]
    train = add_date_features(train)
    sample['transactiondate'] = pd.Timestamp('2016-12-01')  # Dummy
    sample = add_date_features(sample)
    return train, prop, sample


def add_additional_feature(add_feature, train, prop, sample, verbose=False):
    # ===============================================================
    with process_tracker("Preprocess", verbose):
    # ===============================================================
        prop = manual_downcast(prop)
        # Dealt with fireplaceflag, hashottuborspa and taxdelinquencyflag
        prop = replace_binary_string_with_boolean(frame_obj=prop)
        id_features = ['airconditioningtypeid','architecturalstyletypeid',
                       'buildingqualitytypeid', 'buildingqualitytypeid',
                       'decktypeid', 'heatingorsystemtypeid',
                       'pooltypeid7', 'pooltypeid2', 'pooltypeid10',
                       'propertylandusetypeid', 'storytypeid',
                       'typeconstructiontypeid', 'fips']
        prop = label_encode(prop, id_features)
        check_feature_feasible(prop)
    # ===============================================================
    with process_tracker("Add additional features", verbose):
    # ===============================================================
        df_train = train.merge(prop, how='left', on='parcelid')
        if add_feature:
            df_train = add_features(df_train)

        # Month
    # ===============================================================
    with process_tracker("Prepare train/test sets", verbose):
    # ===============================================================
        x_train = df_train.drop(['parcelid', 'logerror'], axis=1)
        y_train = df_train['logerror'].values

        sample['parcelid'] = sample['ParcelId']
        df_test = sample.merge(prop, on='parcelid', how='left')
        if add_feature:
            df_test = add_features(df_test)
        x_test = df_test[x_train.columns]
    if verbose:
        print('x_train shape: %s \ny_train shape: %s' % (x_train.shape, y_train.shape))
    del df_train; gc.collect()
    check_feature_feasible(x_train)
    return x_train, y_train, x_test
