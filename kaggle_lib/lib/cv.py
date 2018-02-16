from lib.metrics import eval_gini_normalized, gini_xgb
from sklearn.model_selection import  KFold
from scipy.stats import hmean
from scipy.stats.mstats import gmean
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import xgboost as xgb
import gc


class Cross_Validate():
    ''' A class w

    '''

    def __init__(self, name, n_splits, len_trn, len_tst, clf, params, max_round, bags=5, upsam=False):
        self.match_fit = {
            'cat': self.cat_fit,
            'log': self.skl_fit,
            'etc': self.skl_fit,
            'xgb': None,}
        self.name = name[:3]
        self.fit_func = self.match_fit[self.name]
        self.n_splits = n_splits
        self.clf = clf
        self.len_trn = len_trn
        self.len_tst = len_tst
        self.upsam = upsam
        self.params = params
        self.max_round = max_round
        self.bags = bags
        self.y_trn = np.zeros(self.len_trn)
        self.y_tst = np.zeros(self.len_tst)
        self.y_tst_mrank = np.zeros(self.len_tst)
        self.y_tst_tmp = np.zeros((self.len_tst, self.n_splits))
        self.y_tst_tmp2 = np.zeros((self.len_tst, self.n_splits * self.bags))
        self.trn_gini = 0
        self.trn_gini_bags = np.zeros(self.bags)
        self.holdout_gini = np.zeros(n_splits)
        self.methods = [self.cross_validate, self.cross_validate_xgb]
        self.fscore = pd.DataFrame()


    def cross_validate_xgb(self, x_train, y_train, x_test, folds, verbose_eval=False):
        ''' cross validation for xgb not with sklearn api

        '''
        if int(verbose_eval) > 0:
            print("Validation starts")
            print(x_train.shape)

        for i, (train_idx, test_idx) in enumerate(folds):
            if int(verbose_eval) > 0:
                print("Fold %i" % (i + 1))
            x_trn = x_train.iloc[train_idx,]
            y_trn = y_train[train_idx]
            x_holdout = x_train.iloc[test_idx,]
            y_holdout = y_train[test_idx]

            if self.upsam:
                x_trn, y_trn = self.upsampling(x_trn, y_trn)

            d_trn = xgb.DMatrix(x_trn, label=y_trn)
            d_tst = xgb.DMatrix(x_test)
            d_holdout = xgb.DMatrix(x_holdout, label=y_holdout)
            watchlist = [(d_holdout, 'holdout')]

            del x_trn, y_trn, x_holdout
            gc.collect()

            self.clf = xgb.train(
                params=self.params,
                dtrain=d_trn,
                num_boost_round=self.max_round,
                evals=watchlist,
                feval=gini_xgb,
                maximize=True,
                verbose_eval=verbose_eval,
                early_stopping_rounds=50
            )

            self.y_trn[test_idx] = self.clf.predict(d_holdout)
            self.holdout_gini = eval_gini_normalized(y_holdout, self.y_trn[test_idx])
            self.y_tst_tmp[:, i] = self.clf.predict(d_tst)
            self.fscore = pd.concat([self.fscore, pd.Series(self.clf.get_fscore(), name=i)], axis=1)

            del d_trn, d_holdout, y_holdout
            gc.collect()

        self.trn_gini = eval_gini_normalized(y_train, self.y_trn)
        self.y_tst = np.mean(self.y_tst_tmp, axis=1)
        if int(verbose_eval) > 0:
            print ("CV score for train cv set: %f" % self.trn_gini)

    # print ("CV variance for %i folds: %f" % (self.n_splits, np.var(self.holdout_gini)))


    def cross_validate(self, x_train, y_train, x_test, folds, verbose_eval=False):
        if int(verbose_eval) > 0:
            print("Validation starts")
            print(x_train.shape)

        for i, (train_idx, test_idx) in enumerate(folds):
            print("Fold %i" % (i + 1))
            x_trn = x_train.iloc[train_idx,]
            y_trn = y_train[train_idx]
            x_holdout = x_train.iloc[test_idx,]
            y_holdout = y_train[test_idx]

            eval_set = [(x_holdout, y_holdout)]

            # match to a proper sklearn .fit function
            self.fit_func(x_trn, y_trn)

            self.y_trn[test_idx] = self.clf.predict_proba(x_holdout)[:, 1]
            self.holdout_gini = eval_gini_normalized(y_holdout, self.y_trn[test_idx])
            self.y_tst_tmp[:, i] = self.clf.predict_proba(x_test)[:, 1]
            gc.collect()

        self.trn_gini = eval_gini_normalized(y_train, self.y_trn)
        self.y_tst = np.mean(self.y_tst_tmp, axis=1)
        if int(verbose_eval) > 0:
            print ("CV score for train cv set: %f" % self.trn_gini)

    def bagging(self, x_train, y_train, x_test, idx, verbose_eval=False):
        random_seeds = [177, 47, 8243, 5210, 1]
        print("Bagging of %i" % self.bags)
        trn_series = np.zeros((self.len_trn, self.bags))
        tst_series = np.zeros((self.len_tst, self.bags))

        for i in range(self.bags):
            print("Bag %i" % (i + 1))
            folds = list(KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=random_seeds[i]
            ).split(x_train, y_train))
            self.methods[idx](x_train, y_train, x_test, folds, verbose_eval)
            trn_series[:, i] = self.y_trn
            tst_series[:, i] = self.y_tst

            self.y_tst_tmp2[:, (i*5):((i+1)*5)] = self.y_tst_tmp
            self.trn_gini_bags[i] = self.trn_gini
            # Reset all values (not necessary but)
            self.y_trn = np.zeros(self.len_trn)
            self.y_tst = np.zeros(self.len_tst)
            self.y_tst_tmp = np.zeros((self.len_tst, self.n_splits))

        self.y_trn = np.mean(trn_series, axis=1)
        self.y_tst = np.mean(tst_series, axis=1)
        self.trn_gini = eval_gini_normalized(y_train, self.y_trn)
        print ("CV score among different bags:\n")
        print ("CV score among different bags:\n")
        print self.trn_gini_bags
        print ("CV score var"),
        print np.var(self.trn_gini_bags)
        print ("\nCV score for %i aver cv set: %f" % (self.bags, self.trn_gini))

    def upsampling(self, x_trn, y_trn):
        pos = pd.Series(y_trn == 1)

        x_trn = pd.concat([x_trn, x_trn.loc[pos]], axis=0)
        y_trn = pd.concat([y_trn, y_trn.loc[pos]], axis=0)

        idx = np.arange(x_trn.shape[0])
        np.random.shuffle(idx)

        x_trn = x_trn[idx]
        y_trn = y_trn[idx]

        return x_trn, y_trn

    def cat_fit(self, x_trn, y_trn):
        cat_features = []
        temp = list(x_trn.columns.str.endswith('_cat'))
        for i in range(len(temp)):  # int index of cat columns
            if temp[i] == True:
                cat_features.append(i)

        self.clf.fit(
            X=x_trn,
            y=y_trn,
            cat_features = cat_features
        )

    def skl_fit(self, x_trn, y_trn):
        self.clf.fit(
            X=x_trn,
            y=y_trn,
        )



# funtions and class
class Cross_Validate_Reg():
    ''' A class wrapper

    '''

    def __init__(self, name, n_splits, len_trn, len_tst, clf, params, max_round, bags=5, upsam=False):
        self.match_fit = {
            'cat': self.cat_fit,
            'log': self.skl_fit,
            'etc': self.skl_fit,
            'xgb': None,}
        self.name = name[:3]
        self.fit_func = self.match_fit[self.name]
        self.n_splits = n_splits
        self.clf = clf
        self.len_trn = len_trn
        self.len_tst = len_tst
        self.upsam = upsam
        self.params = params
        self.max_round = max_round
        self.bags = bags
        self.y_trn = np.zeros(self.len_trn)
        self.y_tst = np.zeros(self.len_tst)
        self.y_tst_mrank = np.zeros(self.len_tst)
        self.y_tst_tmp = np.zeros((self.len_tst, self.n_splits))
        self.y_tst_tmp2 = np.zeros((self.len_tst, self.n_splits * self.bags))
        self.trn_gini = 0
        self.trn_gini_bags = np.zeros(self.bags)
        self.holdout_gini = np.zeros(n_splits)
        self.methods = [self.cross_validate, self.cross_validate_xgb]
        self.fscore = pd.DataFrame()


    def cross_validate_xgb(self, x_train, y_train, x_test, folds, verbose_eval=False):
        ''' cross validation for xgb not with sklearn api

        '''
        if int(verbose_eval) > 0:
            print("Validation starts")
            print(x_train.shape)

        for i, (train_idx, test_idx) in enumerate(folds):
            if int(verbose_eval) > 0:
                print("Fold %i" % (i + 1))
            x_trn = x_train.iloc[train_idx,]
            y_trn = y_train[train_idx]
            x_holdout = x_train.iloc[test_idx,]
            y_holdout = y_train[test_idx]

            if self.upsam:
                x_trn, y_trn = self.upsampling(x_trn, y_trn)

            d_trn = xgb.DMatrix(x_trn, label=y_trn)
            d_tst = xgb.DMatrix(x_test)
            d_holdout = xgb.DMatrix(x_holdout, label=y_holdout)
            watchlist = [(d_holdout, 'holdout')]

            del x_trn, y_trn, x_holdout
            gc.collect()

            self.clf = xgb.train(
                params=self.params,
                dtrain=d_trn,
                num_boost_round=self.max_round,
                evals=watchlist,
                feval=gini_xgb,
                maximize=True,
                verbose_eval=verbose_eval,
                early_stopping_rounds=50
            )

            self.y_trn[test_idx] = self.clf.predict(d_holdout)
            self.holdout_gini = mean_absolute_error(y_holdout, self.y_trn[test_idx])
            self.y_tst_tmp[:, i] = self.clf.predict(d_tst)
            self.fscore = pd.concat([self.fscore, pd.Series(self.clf.get_fscore(), name=i)], axis=1)

            del d_trn, d_holdout, y_holdout
            gc.collect()

        self.trn_gini = mean_absolute_error(y_train, self.y_trn)
        self.y_tst = np.mean(self.y_tst_tmp, axis=1)
        if int(verbose_eval) > 0:
            print ("CV score for train cv set: %f" % self.trn_gini)
            
    def cross_validate(self, x_train, y_train, x_test, folds, verbose_eval=False):
        if int(verbose_eval) > 0:
            print("Validation starts")
            print(x_train.shape)

        for i, (train_idx, test_idx) in enumerate(folds):
            print("Fold %i" % (i + 1))
            x_trn = x_train.iloc[train_idx,]
            y_trn = y_train[train_idx]
            x_holdout = x_train.iloc[test_idx,]
            y_holdout = y_train[test_idx]

            eval_set = [(x_holdout, y_holdout)]

            # match to a proper sklearn .fit function
            self.fit_func(x_trn, y_trn)

            self.y_trn[test_idx] = self.clf.predict(x_holdout)
            self.holdout_gini = mean_absolute_error(y_holdout, self.y_trn[test_idx])
            self.y_tst_tmp[:, i] = self.clf.predict(x_test)
            gc.collect()

        self.trn_gini = mean_absolute_error(y_train, self.y_trn)
        self.y_tst = np.mean(self.y_tst_tmp, axis=1)
        if int(verbose_eval) > 0:
            print ("CV score for train cv set: %f" % self.trn_gini)
            
    def cat_fit(self, x_trn, y_trn):
        cat_features = []
        temp = list(x_trn.columns.str.startswith('cat'))
        for i in range(len(temp)):  # int index of cat columns
            if temp[i] == True:
                cat_features.append(i)

        self.clf.fit(
            X=x_trn,
            y=y_trn,
            cat_features = cat_features
        )
        
    def skl_fit(self, x_trn, y_trn):
        self.clf.fit(
            X=x_trn,
            y=y_trn,
        )