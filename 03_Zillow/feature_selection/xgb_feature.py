import sys
sys.path.append("../")

from manual_preprocessing.input import load_data, add_additional_feature
from xgboost.sklearn import XGBRegressor
from tqdm import tqdm
from sklearn.model_selection import cross_validate
from kaggle_lib.Utils.KA_utils import process_tracker
from sklearn.feature_selection import SelectFromModel
# ===============================================================
# Global
# ===============================================================
VERBOSE = 0
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

print "Validation starts ..."
xgb_model = XGBRegressor(**xgb_params)
scoring = 'neg_mean_absolute_error'
scores = cross_validate(xgb_model, x_train, y_train, 
    scoring=scoring, cv=5, n_jobs=-1)
print "Before:"
print scores

# remove some features
model = SelectFromModel(xgb_model, prefit=True)
x_new = model.transform(x_train)
scores = cross_validate(xgb_model, x_new, y_train, 
    scoring=scoring, cv=5, n_jobs=-1)
print "After"
print scores