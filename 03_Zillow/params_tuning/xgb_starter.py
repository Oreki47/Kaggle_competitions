import sys
sys.path.append("../")

from manual_preprocessing.input import load_data, add_additional_feature
from xgboost.sklearn import XGBRegressor
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from kaggle_lib.Utils.KA_utils import process_tracker
# ===============================================================
# Global
# ===============================================================
VERBOSE = 1
ADD_FEATURE = 1
ENCODE_STRING = 1
# ===============================================================
# Get data
# ===============================================================
train, prop, sample = load_data(2016, VERBOSE)
x_train, y_train, x_test = add_additional_feature(add_feature=ADD_FEATURE, train=train, prop=prop, sample=sample,
                                                  verbose=VERBOSE)
# ===============================================================
# xgb model
# ===============================================================
xgb_params = {}
xgb_params['max_depth'] = 6
xgb_params['learning_rate'] = 0.20 # xgb's 'eta'
xgb_params['n_estimators'] = 100
xgb_params['silent'] = 1
xgb_params['nthread'] = 8 # as number of CPUs on the system
xgb_params['min_child_weight'] = 12
xgb_params['objective'] = "reg:linear"
xgb_params['subsample'] = 0.77
xgb_params['reg_lambda'] = 0.8
xgb_params['reg_alpha'] = 0.4
xgb_params['base_score'] = 0
# ===============================================================
# Tuning
# ===============================================================
for learning_rate in [i/100.0 for i in range(20, 41, 2)]:
    # ===============================================================
    with process_tracker("max_depth tuning", VERBOSE):
    # ===============================================================
        param_test1 = {
            'max_depth': [5, 6, 7, 8]
        }
        xgb_params1 = {key: value for key, value in xgb_params.iteritems() if key not in param_test1.keys()}
        xgb_model1 = XGBRegressor(**xgb_params1)
        gsearch1 = GridSearchCV(estimator=xgb_model1,
                                param_grid=param_test1,
                                scoring='neg_mean_absolute_error',
                                n_jobs=8,
                                iid=False,
                                cv=5)
        gsearch1.fit(x_train, y_train)
        xgb_params.update(gsearch1.best_params_)
    # # ===============================================================
    # with process_timer("gamma tuning", VERBOSE):
    # # ===============================================================
    #     param_test2 = {
    #         'gamma': [i / 10.0 for i in range(0, 5)]
    #     }
    #     xgb_params2 = {key: value for key, value in xgb_params.iteritems() if key not in param_test2.keys()}
    #     xgb_model2 = XGBRegressor(**xgb_params1)
    #     gsearch2 = GridSearchCV(estimator=xgb_model2,
    #                             param_grid=param_test2,
    #                             scoring='neg_mean_absolute_error',
    #                             n_jobs=-1,
    #                             iid=False,
    #                             cv=5)
    #     gsearch2.fit(x_train, y_train)
    #     xgb_params.update(gsearch2.best_params_)
    # # ===============================================================
    # with process_timer("subsample tuning", VERBOSE):
    # # ===============================================================
    #     param_test3 = {
    #         'subsample': [i / 10.0 for i in range(6, 10)],
    #         'colsample_bytree': [i / 10.0 for i in range(6, 10)]
    #     }
    #     xgb_params3 = {key: value for key, value in xgb_params.iteritems() if key not in param_test3.keys()}
    #     xgb_model3 = XGBRegressor(**xgb_params1)
    #     gsearch3 = GridSearchCV(estimator=xgb_model3,
    #                             param_grid=param_test3,
    #                             scoring='neg_mean_absolute_error',
    #                             n_jobs=-1,
    #                             iid=False,
    #                             cv=5)
    #     gsearch3.fit(x_train, y_train)
    #     xgb_params.update(gsearch3.best_params_)
    # # ===============================================================
    # with process_timer("regularization tuning", VERBOSE):
    # # ===============================================================
    #     param_test4 = {
    #         'reg_alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1],
    #         'reg_labmda': [1e-5, 1e-4, 1e-3, 1e-2, 1]
    #     }
    #     xgb_params4 = {key: value for key, value in xgb_params.iteritems() if key not in param_test4.keys()}
    #     xgb_model4 = XGBRegressor(**xgb_params1)
    #     gsearch4 = GridSearchCV(estimator=xgb_model4,
    #                             param_grid=param_test4,
    #                             scoring='neg_mean_absolute_error',
    #                             n_jobs=-1,
    #                             iid=False,
    #                             cv=5)
    #     gsearch4.fit(x_train, y_train)
    #     xgb_params.update(gsearch4.best_params_)

print xgb_params
