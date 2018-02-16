import pandas as pd
import numpy as np
from kaggle_lib.Preprocessing.downcast import auto_downcast
from manual_preprocessing.dtype_string import replace_binary_string_with_boolean,\
    replace_string_with_boolean
from manual_preprocessing.features import  add_features
import lightgbm as lgb
import matplotlib.pyplot as plt
import gc

# ===============================================================
# Metadata
# ===============================================================
DROP_STRING_COLUMNS = 1
IF_TESTING = 1
IF_ADD_FEATURE = 0

# ===============================================================
print ('\nLoading data...')
# ===============================================================
prop = pd.read_csv('data/properties_2016.csv')
train = pd.read_csv('data/train_2016_v2.csv', dtype={'parcelid': np.uint32,
                                                     'logerror': np.float32,
                                                     'transactiondate': object})
sample = pd.read_csv('data/sample_submission.csv', dtype={'parcelid':np.uint32,
                                                          '201610': np.uint8,
                                                          '201611': np.uint8,
                                                          '201612': np.uint8,
                                                          '201710': np.uint8,
                                                          '201711': np.uint8,
                                                          '201712': np.uint8,})
# ===============================================================
print ('\nPerforming downcast...')
# ===============================================================
prop, nalist = auto_downcast(prop)
# Dealt with fireplaceflag, hashottuborspa and taxdelinquencyflag
prop = replace_binary_string_with_boolean(frame_obj=prop)
# Dealt with propertycountylandusecode, propertyzoningdesc
if DROP_STRING_COLUMNS == 1:
    prop.drop(['propertycountylandusecode', 'propertyzoningdesc'], axis=1, inplace=True)
else:
    prop = replace_string_with_boolean(frame_obj=prop)
# ===============================================================
print('Creating additional features set ...')
# ===============================================================
df_train = train.merge(prop, how='left', on='parcelid')
if IF_ADD_FEATURE:
    df_train = add_features(df_train)
# ===============================================================
print('Building train set...')
# ===============================================================
x_train = df_train.drop(['parcelid', 'transactiondate', 'logerror'], axis=1)
y_train = df_train['logerror'].values
print x_train.shape, y_train.shape

features = x_train.columns

del df_train; gc.collect()
# ===============================================================
print('Building ltrain...')
# ===============================================================
ltrain = lgb.Dataset(x_train, label=y_train)
# ===============================================================
print('Training ...')
# ===============================================================
lgb_params = {
    'metric': 'mae',
    'max_depth': 100,
    'num_leaves' : 32,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.95,
    'bagging_freq': 8,
    'learning_rate': 0.0025,
    'verbosity': 0,
}
lgb_model = lgb.train(lgb_params, ltrain, verbose_eval=0, num_boost_round=2930)
if IF_TESTING:
    # ===============================================================
    print('Building test set ...')
    # ===============================================================
    sample['parcelid'] = sample['ParcelId']
    df_test = sample.merge(prop, on='parcelid', how='left')
    if IF_ADD_FEATURE:
        df_test = add_features(df_test)

    del prop; gc.collect()

    x_test = df_test[features]
    del df_test, sample; gc.collect()
    # ===============================================================
    print('Predicting on test ...')
    # ===============================================================
    p_test = lgb_model.predict(x_test)

    del x_test; gc.collect()

    sub = pd.read_csv('data/sample_submission.csv', dtype={'parcelid':np.uint32,
                                                              '201610': np.float32,
                                                              '201611': np.float32,
                                                              '201612': np.float32,
                                                              '201710': np.float32,
                                                              '201711': np.float32,
                                                              '201712': np.float32,})
    for c in sub.columns[sub.columns != 'ParcelId']:
        sub[c] = p_test
    # ===============================================================
    print('Writing csv ...')
    # ===============================================================
    sub.to_csv('submission/lgb_starter.csv', index=False, float_format='%.4f')
    print sub.info()
