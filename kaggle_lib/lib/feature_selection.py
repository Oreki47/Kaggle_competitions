from lib.cv import Cross_Validate
from sklearn.model_selection import  KFold
from scipy.stats import hmean
from scipy.stats.mstats import gmean
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import xgboost as xgb
import gc




class Feature_Selection():
    """ A class wrapper

    """

    def __init__(self, n_splits, params, max_round):
        self.n_splits = n_splits
        self.params = params
        self.max_round = max_round
        self.current_best = 0
        self.col_temp = 0
        self.scores = []
        self.cols = []

    def backward_selection(self, x_train, y_train, x_test, folds, cols, verbose_eval = False):
        cv = Cross_Validate(None, n_splits=self.n_splits, len_trn=x_train.shape[0], len_tst=x_test.shape[0], clf=-1,
                            params=self.params, max_round=self.max_round)

        cv.cross_validate_xgb(x_train, y_train, x_test, folds)
        self.current_best = cv.trn_gini
        self.scores.append(self.current_best)

        for i in range(len(cols)):
            print("Round %i" % (i+1))
            print("Shape of train"),
            print x_train.shape

            for col in cols:
                x_train_col = x_train[col]
                x_test_col = x_test[col]

                x_train.drop(col, axis=1, inplace=True)
                x_test.drop(col, axis=1, inplace=True)

                cv.cross_validate_xgb(x_train, y_train, x_test, folds)

                if cv.trn_gini > self.current_best:
                    self.current_best = cv.trn_gini
                    self.col_temp = col

                x_train = pd.concat([x_train, x_train_col], axis=1)
                x_test = pd.concat([x_test, x_test_col], axis=1)

            if self.col_temp != 0:
                cols.remove(self.col_temp)
                x_train.drop(self.col_temp, axis=1, inplace=True)
                x_test.drop(self.col_temp, axis=1, inplace=True)

                self.cols.append(self.col_temp)
                self.scores.append(self.current_best)
                self.col_temp = 0
            else:
                break


    def forward_selection(self, x_train, y_train, x_test, folds, cols):
        cv = Cross_Validate(None, n_splits=self.n_splits, len_trn=x_train.shape[0], len_tst=x_test.shape[0], clf=-1,
                            params=self.params, max_round=self.max_round)

        x_train_cols = x_train[cols]
        x_test_cols = x_test[cols]

        x_train.drop(cols, axis=1, inplace=True)
        x_test.drop(cols, axis=1, inplace=True)

        cv.cross_validate_xgb(x_train, y_train, x_test, folds)
        self.current_best = cv.trn_gini
        self.scores.append(self.current_best)


        for i in range(len(cols)):
            print("Round %i" % (i+1))
            print("Shape of train"),
            print x_train.shape

            for col in cols:

                x_train = pd.concat([x_train, x_train_cols[col]], axis=1)
                x_test = pd.concat([x_test, x_test_cols[col]], axis=1)

                cv.cross_validate_xgb(x_train, y_train, x_test, folds)

                if cv.trn_gini > self.current_best:
                    self.current_best = cv.trn_gini
                    self.col_temp = col

                x_train.drop(x_train_cols[col], axis=1, inplace=True)
                x_test.drop(x_test_cols[col], axis=1, inplace=True)

            if self.col_temp != 0:
                cols.remove(self.col_temp)
                x_train = pd.concat([x_train, x_train_cols[self.col_temp]], axis=1)
                x_test = pd.concat([x_test, x_test_cols[self.col_temp]], axis=1)

                self.cols.append(self.col_temp)
                self.scores.append(self.current_best)
                self.col_temp = 0
            else:
                break







