import sys
sys.path.append("../")

from lib.preprocess import load_all, downcast
from lib.cv import Cross_Validate
from lib.utilities import sub_to_csv, plot_importance
from sklearn.model_selection import  KFold
from datetime import datetime
from lib.feature import feature_engineering_1
import gc
import os


def xgb01(x_train, y_train, x_test, folds, max_round, n_splits=5):
    params = {}
    params['max_depth'] = 4
    params['objective'] = "binary:logistic"
    params['eta'] = 0.07  # learning rate
    params['subsample'] = 0.8
    params['min_child_weight'] = 8
    params['colsample_bytree'] = 0.8
    params['scale_pos_weight'] = 1.6
    params['gamma'] = 2
    params['n_jobs'] = -1
    params['reg_alpha'] = 3
    params['reg_lambda'] = 1.3
    params['silent'] = 1

    # Additional processing of data
    x_train, x_test = feature_engineering_1(x_train, x_test, y_train)

    # Cross Validate
    cv = Cross_Validate(xgb01.__name__, n_splits, x_train.shape[0], x_test.shape[0], -1, params, max_round)
    cv.cross_validate_xgb(x_train, y_train, x_test, folds, verbose_eval=100)

    return cv.trn_gini, cv.y_trn, cv.y_tst, cv.fscore


if __name__ == '__main__':
    print "Level 0 model test..."
    os.chdir("..")  # set to parent dir

    # ===============================================================
    # Global params
    # ===============================================================
    MAX_ROUND = 1200
    # ===============================================================
    # Load and downcast
    # ===============================================================
    train, x_test, sub = load_all()
    x_train = train.drop('target', axis=1)
    y_train = train.target
    x_train, x_test = downcast(x_train, x_test)

    del train
    gc.collect()
    # ===============================================================
    # Run cv and bags
    # ===============================================================
    n_splits = 5
    folds = list(KFold(n_splits=n_splits, shuffle=True, random_state=47).split(x_train, y_train))
    trn_gini, y_trn, y_tst, fscore = xgb01(x_train, y_train, x_test, folds, MAX_ROUND, n_splits)

    # ===============================================================
    # Save submission and figure
    # ===============================================================
    if MAX_ROUND > 300:
        sub.target = y_tst
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        sub_to_csv(sub, y_trn, trn_gini, xgb01.__name__, current_time)
        plot_importance(fscore, trn_gini, xgb01.__name__, current_time)
