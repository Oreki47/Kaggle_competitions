import sys, os

sys.path.append("../")
from general.preprocess import data_preparation
from general.ClfWrappers import LgbWrapper
from general.utilities import sub_to_csv
from features.f1 import features_set_f1
from cv.cv_03 import cross_validate


TARGET = 'visitors'
FEATURES, CAT_FEATURES = features_set_f1()
SEED = 177

full_data, ntrain, ntest = data_preparation()
cat_feats_lgb = [i for i, x in enumerate(full_data[FEATURES].columns) if x in CAT_FEATURES]
print("Overfiting process initiating...")

lgb_params = dict()
lgb_params['objective'] = 'regression_l2'
lgb_params['metric'] = 'l2_root'
lgb_params['learning_rate'] = 0.02
lgb_params['random_state'] = SEED
lgb_params['silent'] = True  # does help
lgb_params['verbose_eval'] = False

lgb_params['n_estimators'] = 2000

lgb_params['min_child_samples'] = 24
lgb_params['num_leaves'] = 31
lgb_params['subsample_freq'] = 5
lgb_params['colsample_bytree'] = 0.81
lgb_params['reg_alpha'] = 4.89
lgb_params['reg_lambda'] = 0.053
lgb_params['min_child_weight'] = 0.039
lgb_params['subsample'] = 0.99
lgb_params['cat_features'] = cat_feats_lgb

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
