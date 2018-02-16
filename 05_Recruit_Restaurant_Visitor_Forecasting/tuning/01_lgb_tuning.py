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
from general.ClfWrappers import LgbWrapper
from features.f0 import features_set_f0
from bayes_opt import BayesianOptimization


def lgb_evaluate(
        min_child_sample, num_leaves, max_bin,
        min_child_weight, subsample, subsample_freq,
        colsample_bytree, reg_alpha, reg_lambda,
        feature_fraction, bagging_fraction
                 ):

    target = 'visitors'
    features = features_set_f0()
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

    lgb_params = dict()
    lgb_params['objective'] = 'regression_l2'
    lgb_params['metric'] = 'l2_root'
    lgb_params['learning_rate'] = 0.1
    lgb_params['random_state'] = seed
    lgb_params['silent'] = True  # does help
    lgb_params['verbose_eval'] = False

    lgb_params['n_estimators'] = 500


    lgb_params['min_child_samples'] = int(np.round(min_child_sample))
    lgb_params['num_leaves'] =  int(np.round(num_leaves))
    lgb_params['max_bin'] =  int(np.round(max_bin))
    lgb_params['subsample_freq'] =  int(np.round(subsample_freq))
    lgb_params['colsample_bytree'] = colsample_bytree
    lgb_params['reg_alpha'] = reg_alpha
    lgb_params['reg_lambda'] = reg_lambda
    lgb_params['min_child_weight'] = min_child_weight
    lgb_params['subsample'] = subsample
    lgb_params['feature_fraction'] = feature_fraction
    lgb_params['bagging_freq'] = int(np.round(bagging_fraction))


    lgb_clf = LgbWrapper(params=lgb_params)
    lgb_clf.train(x_train, y_train, x_valid, y_valid)

    return lgb_clf.best_score


if __name__ == "__main__":
    print("Tuning process initiating...")
    gp_params = {"alpha": 1e-5}
    str_time = datetime.now().replace(microsecond=0)

    search_space = {
        'min_child_sample': (20, 200),
        'num_leaves': (16, 32),
        'max_bin': (100, 255),
        'min_child_weight': (1e-3, 5e-2),
        'subsample': (0.8, 1),
        'subsample_freq': (1, 5),
        'colsample_bytree': (0.7, 1),
        'reg_alpha': (0, 5),
        'reg_lambda': (0, 5),
        'feature_fraction': (0, 1),
        'bagging_freq': (0, 10)
    }

    lgb_BO = BayesianOptimization(
        lgb_evaluate,
        pbounds=search_space,
        )

    lgb_BO.maximize(init_points=10, n_iter=80, )  # **gp_params

    lgb_BO_scores = pd.DataFrame(lgb_BO.res['all']['params'])
    lgb_BO_scores['score'] = pd.DataFrame(lgb_BO.res['all']['values'])
    lgb_BO_scores = lgb_BO_scores.sort_values(by='score', ascending=True)
    lgb_BO_scores.to_csv('01_lgb_tuning.csv')

    end_time = datetime.now().replace(microsecond=0)
    print("Time used", (end_time-str_time))
