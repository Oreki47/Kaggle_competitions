import sys
sys.path.append("../")

from lib.preprocess import load_all, downcast
from lib.params_tuning import Tune_Params
from sklearn.model_selection import  KFold
from lib.feature import feature_engineering_1
from matplotlib.pyplot import savefig
from collections import OrderedDict
import numpy as np
import gc
import os


def params_tuning(x_train, y_train, x_test, folds, params_dict, max_round, n_splits=5):
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

    tp = Tune_Params(params_dict, params, max_round, n_splits)
    tp.tune_seq(x_train, y_train, x_test, folds)

    return tp



if __name__ == '__main__':
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

    params_dict = OrderedDict()
    params_dict['eta'] = np.arange(0.01, 0.06, 0.01)
    # params_dict['max_depth'] = np.arange(3, 5, 1)
    params_dict['scale_pos_weight'] = np.arange(1.2, 2.4, 0.2)
    params_dict['gamma'] = np.arange(1, 2.5, 0.25)
    params_dict['reg_alpha'] = np.arange(2, 3.5, 0.25)
    params_dict['reg_lambda'] = np.arange(0.7, 1.5, 0.2)
    # ===============================================================
    # Run cv and bags
    # ===============================================================
    n_splits = 5
    folds = list(KFold(n_splits=n_splits, shuffle=True, random_state=177).split(x_train, y_train))  # seed = 177
    tp = params_tuning(x_train, y_train, x_test, folds, params_dict, MAX_ROUND, n_splits)
    tp.sframe.sort_index().to_csv('trail6tune.csv')
    tp.sframe['score'].plot()
    savefig('trail6tune.pdf', dpi=800)
