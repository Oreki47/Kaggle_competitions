import sys, os

sys.path.append("../")
from general.preprocess import data_preparation
from general.ClfWrappers import LgbWrapper
from general.utilities import sub_to_csv
from features.f0 import features_set_f0
from cv.cv_01 import cross_validate


TARGET = 'visitors'
FEATURES = features_set_f0()
SEED = 177
print("Overfiting process initiating...")

lgb_params = dict ()
lgb_params['objective'] = 'regression_l2'
lgb_params['metric'] = 'l2_root'
lgb_params['learning_rate'] = 0.01
lgb_params['random_state'] = SEED
lgb_params['silent'] = True  # does help
lgb_params['verbose_eval'] = False

lgb_params['n_estimators'] = 2000
lgb_params['min_child_samples'] = 5
lgb_params['num_leaves'] = 32
lgb_params['max_bin'] = 100
lgb_params['subsample_freq'] = 1
lgb_params['colsample_bytree'] = 0.90
lgb_params['reg_alpha'] = 3.5
lgb_params['reg_lambda'] = 4.2
lgb_params['min_child_weight'] = 0.03
lgb_params['subsample'] = 0.93


full_data, ntrain, ntest = data_preparation()
lgb_clf= LgbWrapper (params=lgb_params)
results = cross_validate(
    full_data=full_data,
    clf=lgb_clf,
    seed=SEED,
    ntrain=ntrain,
    ntest=ntest,
    features=FEATURES,
    target=TARGET,
    nfolds=4,
)
sub, v06, v33, oof_score = results

sub_to_csv(sub, v06, v33, oof_score[0], oof_score[1], os.path.basename(sys.argv[0]))

print(lgb_clf.clf)
