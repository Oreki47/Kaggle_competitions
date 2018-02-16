'''
    Main function that includes everything.
'''
from manual_preprocessing.input import load_data, add_additional_feature
from kaggle_lib.Ensemble.KA_stacking import ka_Stacking
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from datetime import datetime
import pandas as pd
import numpy as np
# ===============================================================
# Global
# ===============================================================
VERBOSE = 1
ADD_FEATURE = 1
SELECT_FEATURES = 1
# ===============================================================
# Features
# ===============================================================
data_features = ['transaction_month', 'transaction_day', 'transaction_quarter']
features = ["numberofstories", "N-taxdelinquencyyear-3", "N-YardProp-2", "regionidcity", "poolcnt",
            "regionidzip", "N-Avg-structuretaxvaluedollarcnt", "N-YardProp-1", "N-FirstFloorProp-2", "yearbuilt",
            "garagetotalsqft", "N-city_count", "taxdelinquencyflag", "N-Dev-structuretaxvaluedollarcnt", "N-ValueProp",
            "N-TotalRoomsSum", "heatingorsystemtypeid4", "propertylandusetypeid8", "structuretaxvaluedollarcnt",
            "airconditioningtypeid", "propertyzoningdesc", "censustractandblock-1", "N-TotalRoomsNotPriRes",
            "propertylandusetypeid5",
            "propertylandusetypeid6", "finishedsquarefeet6", "calculatedfinishedsquarefeet", "taxdelinquencyyear",
            "buildingqualitytypeid4",
            "fireplacecnt", "latitude", "roomcnt", "N-ValueRatio", "rawcensustractandblock-2",
            "landtaxvaluedollarcnt", "N-AvRoomSize", "hashottuborspa", "taxamount", "N-GarageProp",
            "pooltypeid7", "unitcnt", "buildingqualitytypeid", "yardbuildingsqft17", "buildingqualitytypeid7",
            "N-TotalRoomsProd", "propertylandusetypeid", "heatingorsystemtypeid2", "propertycountylandusecode",
            "regionidneighborhood",
            "bathroomcnt", "heatingorsystemtypeid", "propertylandusetypeid13", "N-TaxScore", "N-LivingAreaDiff",
            "propertylandusetypeid14", "N-location-2", "garagecarcnt", "N-ExtraSpace-1", "N-location",
            "bedroomcnt", "finishedsquarefeet50", "longitude", "N-LoftProp", "taxvaluedollarcnt",
            "finishedsquarefeet12", "N-zip_count", "finishedsquarefeet15", "lotsizesquarefeet",
            "N-structuretaxvaluedollarcnt-2",
            "N-structuretaxvaluedollarcnt-3", "N-ExternalSpaceProp", "censustractandblock-3", "N-ExternalSpaceSum",
            ]
features = features + data_features
# ===============================================================
# Params
# ===============================================================
# xgb params
xgb_params1 = {}
xgb_params1['n_estimators'] = 100
xgb_params1['min_child_weight'] = 12
xgb_params1['learning_rate'] = 0.27
xgb_params1['max_depth'] = 6
xgb_params1['subsample'] = 0.77
xgb_params1['reg_lambda'] = 0.8
xgb_params1['reg_alpha'] = 0.4
xgb_params1['base_score'] = 0
xgb_params1['silent'] = 1

xgb_params2 = {}
xgb_params2['n_estimators'] = 100
xgb_params2['min_child_weight'] = 8
xgb_params2['learning_rate'] = 0.37
xgb_params2['max_depth'] = 10
xgb_params2['subsample'] = 0.77
xgb_params2['reg_lambda'] = 0.8
xgb_params2['reg_alpha'] = 0.3
xgb_params2['base_score'] = 0
xgb_params2['silent'] = 1

# lgb params
lgb_params1 = {}
lgb_params1['n_estimators'] = 50
lgb_params1['max_bin'] = 10
lgb_params1['learning_rate'] = 0.36 # shrinkage_rate
lgb_params1['metric'] = 'l1'          # or 'mae'
lgb_params1['sub_feature'] = 0.34
lgb_params1['bagging_fraction'] = 0.85 # sub_row
lgb_params1['bagging_freq'] = 40
lgb_params1['num_leaves'] = 512        # num_leaf
lgb_params1['min_data'] = 500         # min_data_in_leaf
lgb_params1['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
lgb_params1['verbose'] = 0
lgb_params1['feature_fraction_seed'] = 2
lgb_params1['bagging_seed'] = 3

lgb_params2 = {}
lgb_params2['n_estimators'] = 100
lgb_params2['max_bin'] = 8
lgb_params2['learning_rate'] = 0.21 # shrinkage_rate
lgb_params2['metric'] = 'l1'          # or 'mae'
lgb_params2['sub_feature'] = 0.34
lgb_params2['bagging_fraction'] = 0.85 # sub_row
lgb_params2['bagging_freq'] = 40
lgb_params2['num_leaves'] = 512        # num_leaf
lgb_params2['min_data'] = 500         # min_data_in_leaf
lgb_params2['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
lgb_params2['verbose'] = 0
lgb_params2['feature_fraction_seed'] = 2
lgb_params2['bagging_seed'] = 3

# catboost params
cat_params = {}
cat_params['iterations'] = 200
cat_params['learning_rate'] = 0.3
cat_params['depth'] = 6
cat_params['l2_leaf_reg'] = 3
cat_params['loss_function'] = 'MAE'
cat_params['eval_metric'] = 'MAE'
cat_params['random_seed'] = 2017
# ===============================================================
# Stacking
# ===============================================================
# lgb model
lgb_model1 = lgb.LGBMRegressor(**lgb_params1)

lgb_model2 = lgb.LGBMRegressor(**lgb_params2)

# XGB model
xgb_model1 = xgb.XGBRegressor(**xgb_params1)

# XGB model
xgb_model2 = xgb.XGBRegressor(**xgb_params2)

# Catboost model
cat_model = CatBoostRegressor(**cat_params)

stack = ka_Stacking(n_splits=5,
                    stacker=LinearRegression(),
                    base_models=(cat_model,
                                 xgb_model1,
                                 lgb_model1))

# ===============================================================
# Train on 2016
# ===============================================================
train, prop, sample = load_data(2016, VERBOSE)
x_train, y_train, x_test = add_additional_feature(add_feature=ADD_FEATURE, train=train, prop=prop, sample=sample)

if SELECT_FEATURES:
    x_train = x_train[features]
    x_train = x_test[features]
y_test_2016 = stack.fit_predict(x_train, y_train, x_test)

# ===============================================================
# Train on 2017
# ===============================================================
train, prop, sample = load_data(2017, VERBOSE)
x_train, y_train, x_test = add_additional_feature(add_feature=ADD_FEATURE,
                                                  train=train,
                                                  prop=prop,
                                                  sample=sample)
if SELECT_FEATURES:
    x_train = x_train[features]
    x_train = x_test[features]
y_test_2017 = stack.fit_predict(x_train, y_train, x_test)

sub = pd.read_csv('data/sample_submission.csv', dtype={'parcelid': np.uint32,
                                                       '201610': np.float32,
                                                       '201611': np.float32,
                                                       '201612': np.float32,
                                                       '201710': np.float32,
                                                       '201711': np.float32,
                                                       '201712': np.float32,})


for c in ['201610', '201611', '201612']:
    sub[c] = y_test_2016
# CHANGE BACK TO 2017 FOR FINAL SUBMISSION
for c in ['201710', '201711', '201712']:
    sub[c] = y_test_2017
# ===============================================================
print('Writing csv ...')
# ===============================================================

sub.to_csv('submission/sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False, float_format='%.4f')
print sub.info()
