import sys
sys.path.append("../")

from lib.preprocess import load_all, downcast
from lib.feature_selection import Feature_Selection
from lib.utilities import sub_to_csv, plot_importance
from sklearn.model_selection import  KFold
from datetime import datetime
from lib.feature import feature_engineering_1, feature_subset_01
import gc
import os


def xgb_feature_selection(x_train, y_train, x_test, folds, max_round, cols, n_splits=5):
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
    x_train = feature_subset_01(x_train)
    x_test = feature_subset_01(x_test)

    # Cross Validate
    fc = Feature_Selection(n_splits, params, max_round)
    fc.backward_selection(x_train, y_train, x_test, folds, cols)

    return fc.cols, fc.current_best, fc.scores


if __name__ == '__main__':
    print "Level 0 model test..."
    os.chdir("..")  # set to parent dir

    # ===============================================================
    # Global params
    # ===============================================================
    MAX_ROUND = 800
    cols = [
        'ps_ind_04_cat_avg', 'N_reverse_0609',
        'N_reverse_1619'
    ]
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
    folds = list(KFold(n_splits=n_splits, shuffle=True, random_state=177).split(x_train, y_train))
    dropped_cols, best, scores = xgb_feature_selection(x_train, y_train, x_test, folds, MAX_ROUND, cols)
    print ("Columns dropped: "),
    print dropped_cols
    print ("best score: "),
    print best
    print ('round scores: '),
    print scores
