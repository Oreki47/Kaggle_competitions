import sys, os

sys.path.append("../")
from general.preprocess import data_preparation
from general.ClfWrappers import XgbWrapper
from general.utilities import sub_to_csv
from features.f0 import features_set_f0
from cv.cv_02 import cross_validate


TARGET = 'visitors'
FEATURES = features_set_f0()
SEED = 177
print("Overfiting process initiating...")
xgb_params = dict()
xgb_params['objective'] = 'reg:linear'
xgb_params['eval_metric'] = 'rmse'
xgb_params['eta'] = 0.02
xgb_params['seed'] = SEED
xgb_params['silent'] = True  # does help
xgb_params['verbose_eval'] = False
xgb_params['nrounds'] = 5000
xgb_params['early_stopping_rounds'] = 100

xgb_params['max_depth'] = 6
xgb_params['min_child_weight'] = 1
xgb_params['colsample_bytree'] = 0.724
xgb_params['subsample'] = 0.925
xgb_params['gamma'] = 0.512
xgb_params['alpha'] = 8.6
xgb_params['lambda'] = 1

full_data, ntrain, ntest = data_preparation()
xgb_clf = XgbWrapper(seed=SEED, params=xgb_params)
results = cross_validate(
    full_data=full_data,
    clf=xgb_clf,
    seed=SEED,
    ntrain=ntrain,
    ntest=ntest,
    features=FEATURES,
    target=TARGET,
    nfolds=4,
)
sub, v06, v33, oof_score = results

sub_to_csv(sub, v06, v33, oof_score[0], oof_score[1], os.path.basename(sys.argv[0]))