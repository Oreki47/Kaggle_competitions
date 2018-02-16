import gc
import sys
import numpy as np
import lightgbm as lgb
import catboost as cat

sys.path.append("../")
from general.utilities import score_valid
from sklearn.model_selection import KFold

try:  # some error on one of the machine
    import xgboost as xgb
except:
    pass


# funtions and class
class SklearnWrapper(object):
    def __init__(self, clf, seed=177, params=None):
        params['random_state'] = seed
        self.clf = clf
        print(self.clf)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return np.clip(self.clf.predict(x), a_min=0, a_max=1000)


class LgbWrapper(object):
    def __init__(self, params=None):
        # TODO make several parameters int type
        self.params = params
        self.best_score = None
        self.metric = params.pop('metric', 'l2_root')
        self.verbose = params.pop('verbose_eval', False)
        self.early_stopping = params.pop('early_stopping_rounds', 50)
        self.cat_feats = params.pop('cat_features', "")

        self.clf = lgb.LGBMRegressor(**self.params)

    def train(self, x_train, y_train, x_valid, y_valid):
        if x_valid is None and y_valid is None:
            pass
        else:
            eval_set = (x_valid, y_valid)
            self.clf.fit(
                x_train, y_train,
                eval_set=eval_set, eval_names='valid',
                early_stopping_rounds=self.early_stopping,
                verbose=self.verbose,
                eval_metric=self.metric,
                categorical_feature=self.cat_feats
            )

            self.params['n_estimators'] = self.clf.best_iteration_  # update nrounds for this estimator
            self.best_score = self.clf.best_score_['valid']['rmse']  # assign best_score to return

        # retrain the model with best_iteration, or train the model
        self.clf = lgb.LGBMRegressor(**self.params)
        self.clf.fit(x_train, y_train, categorical_feature=self.cat_feats)

    def predict(self, x):
        return self.clf.predict(x)


class XgbWrapper(object):
    def __init__(self, seed=177, params=None):
        self.clf = None
        self.best_score = None
        self.params = params
        self.params['seed'] = seed
        self.nrounds = params.pop('nrounds', 500)
        self.verbose = params.pop('verbose_eval', 200)
        self.early_stopping = params.pop('early_stopping_rounds', 50)

    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        dtrain = xgb.DMatrix(x_train, label=y_train)

        if x_valid is None and y_valid is None:
            pass
        else:
            dvalid = xgb.DMatrix(x_valid, label=y_valid)
            watchlist = [(dvalid, "valid")]
            self.clf = xgb.train(
                params=self.params,
                dtrain=dtrain,
                num_boost_round=self.nrounds,
                evals=watchlist,
                verbose_eval=self.verbose,
                early_stopping_rounds=self.early_stopping
            )

            self.nrounds = self.clf.best_iteration  # update nrounds for this estimator
            self.best_score = self.clf.best_score  # assign best_score to return

        # retrain the model with best_iteration, or train the model
        self.clf = xgb.train(self.params, dtrain, self.nrounds, verbose_eval=self.verbose)

    def predict(self, x):
        return self.clf.predict(xgb.DMatrix(x))


class CatWrapper(object):
    def __init__(self, seed=177, params=None):
        params['random_seed'] = seed
        self.clf = cat.CatBoostRegressor(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return np.exp(self.clf.predict(x))


class Stacker(object):
    '''
        Seems redundant
    '''
    def __init__(self, nfolds, clf, random_state=177):
        self.nfolds = nfolds
        self.stacker = clf
        self.random_state = random_state

    def fit_predict(self, trn, tgt, tst):

        oof_trn = np.zeros((trn.shape[0],))
        oof_tst = np.zeros((tst.shape[0],))
        oof_tst_folds = np.empty((tst.shape[0], self.nfolds))

        folds = list(KFold(n_splits=self.nfolds, shuffle=True, random_state=self.random_state).split(trn, tgt))

        for i, (train_idx, valid_idx) in enumerate(folds):
            x_train = trn[train_idx]
            x_valid = trn[valid_idx]

            y_train = tgt[train_idx]

            self.stacker.train(x_train, y_train)

            oof_trn[valid_idx] = self.stacker.predict(x_valid)
            oof_tst_folds[:, i] = self.stacker.predict(tst)

            del x_train, x_valid, y_train
            gc.collect()

        oof_tst[:] = oof_tst_folds.mean(axis=1)
        oof_score = score_valid(tgt, oof_trn)

        oof_trn = np.clip(np.expm1(oof_trn.reshape(-1, 1)), a_min=0, a_max=1000)
        oof_tst = np.clip(np.expm1(oof_tst.reshape(-1, 1)), a_min=0, a_max=1000)
        return oof_tst, oof_trn, oof_score
