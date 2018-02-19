import gc
import sys
import glob
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error

sys.path.append("../")
from general.utilities import sub_to_csv_stacker
from general.preprocess import data_preparation
from general.ClfWrappers import Stacker, SklearnWrapper, XgbWrapper
from stacker.utilities import search_model, score_valid
from features.f2 import features_set_f2

# starts here
full_data, ntrain, ntest = data_preparation()
features, cat_features = features_set_f2()
tgt = pd.read_csv('../data/air_visit_data.csv').visitors.values
trn_list = [x for x in glob.glob('../valid/*.csv') if 'cv4' in x]
tst_list = [x for x in glob.glob('../submission/*.csv') if 'cv4' in x]

trn_series = pd.DataFrame()
tst_series = pd.DataFrame()

for i, trn in enumerate(trn_list):
    temp = pd.read_csv(trn, index_col=['id']).rename(columns={'visitors': ('visitors_'+str(i))})
    trn_series = pd.concat([trn_series, temp], axis=1)

for i, tst in enumerate(tst_list):
    temp = pd.read_csv(tst, index_col=['id']).rename(columns={'visitors': ('visitors_'+str(i))})
    tst_series = pd.concat([tst_series, temp], axis=1)

# log1p all values
tgt = np.log1p(tgt)
trn = np.log1p(trn_series.values)
tst = np.log1p(tst_series.values)

assert(trn_series.shape[0] == tgt.shape[0])

# ridge Lv.2
param_grid = {"alpha": [0.001,0.01,0.1,1,10,30,100]}
ridge_clf = search_model(
    trn, tgt, Ridge(random_state=177),
    param_grid, n_jobs=1, cv=5, refit=True
)

ridge_clf_wrap = SklearnWrapper(ridge_clf, params={})
ridge_stacker = Stacker(5, ridge_clf_wrap)

ridge_pred, ridge_train, ridge_score = ridge_stacker.fit_predict(trn, np.expm1(tgt), tst)


# xgb Lv.2

x_trn, x_tst = full_data[:ntrain][features].values, full_data[ntrain:][features].values
x_trn = np.hstack([x_trn, trn])
x_tst = np.hstack([x_tst, tst])

del full_data
gc.collect()

xgb_stack_params = {}
xgb_stack_params['objective'] = 'reg:linear'
xgb_stack_params['eta'] = 0.1
xgb_stack_params['max_depth'] = 3
xgb_stack_params['min_child_weight'] = 1
xgb_stack_params['subsample'] = 0.9
xgb_stack_params['colsample_bytree'] = 0.2
xgb_stack_params['gamma'] = 0.1
xgb_stack_params['seed'] = 177
xgb_stack_params['silent']: True

cv_results = xgb.cv(xgb_stack_params, xgb.DMatrix(x_trn, tgt),
                    num_boost_round=1000, nfold=5,
                    metrics='rmse',
                    seed=177,
                    callbacks=[xgb.callback.early_stop(50)],
                    verbose_eval=False)


xgb_stack_params['nrounds'] = np.argmin(cv_results['test-rmse-mean'].values)+1
xgb_clf_wrap = XgbWrapper(seed=177, params=xgb_stack_params)
xgb_stacker = Stacker(5, xgb_clf_wrap)

xgb_pred, xgb_train, xgb_score = xgb_stacker.fit_predict(x_trn, tgt, x_tst)

# Linear Lv.3
lr_train = np.hstack([ridge_train.reshape(-1, 1), xgb_train.reshape(-1, 1)])
lr_test = np.hstack([ridge_pred.reshape(-1, 1), xgb_pred.reshape(-1, 1)])

lr_clf = LinearRegression()
lr_clf.fit(np.log1p(lr_train), tgt)
lr_score = score_valid(tgt, lr_clf.predict(np.log1p(lr_train)))

final_pred = np.expm1(lr_clf.predict(np.log1p(lr_test)))

tst_series['visitors'] = 0
sub = tst_series.reset_index()[['id', 'visitors']]
sub['visitors'] = final_pred

sub_to_csv_stacker(sub, lr_score, "ridge_xgb")
