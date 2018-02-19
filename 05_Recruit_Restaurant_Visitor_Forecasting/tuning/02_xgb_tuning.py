'''
    Quick Note:
    This search is by no means perfect and thorough. I have applied
    4 cv strategies and theoretically they should be having their
    own optimal parameters. However, that process is way too slow.
    Here is a quick computation. For this problem we need ~4 minutes
    to grow a tree. For the following code, a search with n_iter= 50
    will cost me roughly 4 hours. On the other hand, to find the best
    parameters using cross validation, each iteration takes 5*5 minutes,
    and I need to repeat 4 times, which results in 4*20 = 80+ hours. A
    huge commitment with imaginable minimal gain.

'''


import gc
import sys
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.model_selection import train_test_split

sys.path.append("../")
from general.preprocess import data_preparation
from general.ClfWrappers import XgbWrapper
from features.f1 import features_set_f1
from bayes_opt import BayesianOptimization


def xgb_evaluate(
        min_child_weight, colsample_bytree, max_depth,
        subsample, gamma, reg_alpha, reg_lambda, rate_drop
                 ):

    target = 'visitors'
    features, col_feats	 = features_set_f1()
    split = 0.33
    seed = 177
    full_data, ntrain, ntest = data_preparation()
    trn = full_data[:ntrain]
    x_train, x_valid, y_train, y_valid = train_test_split(
        trn[features].values, trn[target].values,
        test_size=split, random_state=seed
    )

    del full_data, trn
    gc.collect()

    xgb_params = dict()
    xgb_params['objective'] = 'reg:linear'
    xgb_params['eval_metric'] = 'rmse'
    xgb_params['eta'] = 0.1
    xgb_params['seed'] = seed
    xgb_params['silent'] = True  # does help
    xgb_params['verbose_eval'] = False
    xgb_params['nrounds'] = 500

    xgb_params['max_depth'] = int(np.round(max_depth))
    xgb_params['min_child_weight'] = int(np.round(min_child_weight))
    xgb_params['colsample_bytree'] = colsample_bytree
    xgb_params['subsample'] = subsample
    xgb_params['gamma'] = gamma
    xgb_params['alpha'] = reg_alpha
    xgb_params['lambda'] = reg_lambda
    xgb_params['rate_drop'] = rate_drop

    xgb_clf = XgbWrapper(seed=seed, params=xgb_params)
    xgb_clf.train(x_train, y_train, x_valid, y_valid)

    return xgb_clf.best_score


if __name__ == "__main__":
    print("Tuning process initiating...")
    str_time = datetime.now().replace(microsecond=0)

    search_space = {
        'max_depth': (5, 11),
        'min_child_weight': (1, 200),
        'colsample_bytree': (0.7, 1),
        'subsample': (0.7, 1),
        'gamma': (0, 10),
        'reg_alpha': (0, 10),
        'reg_lambda': (0, 10),
        'rate_drop': (0, 1),
    }

    xgb_BO = BayesianOptimization(
        xgb_evaluate,
        pbounds=search_space,
        )

    xgb_BO.maximize(init_points=10, n_iter=60)

    xgb_BO_scores = pd.DataFrame(xgb_BO.res['all']['params'])
    xgb_BO_scores['score'] = pd.DataFrame(xgb_BO.res['all']['values'])
    xgb_BO_scores = xgb_BO_scores.sort_values(by='score', ascending=True)
    xgb_BO_scores.to_csv('02_xgb_tuning.csv')

    end_time = datetime.now().replace(microsecond=0)
    print("Time used", (end_time-str_time))
