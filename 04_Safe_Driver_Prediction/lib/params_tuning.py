from lib.metrics import eval_gini_normalized, gini_xgb
from lib.cv import Cross_Validate
from sklearn.model_selection import StratifiedKFold, KFold
from scipy.stats import hmean
from scipy.stats.mstats import gmean
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import xgboost as xgb
import gc

class Tune_Params():
    """ A class wrapper

    """

    def __init__(self, params_dict, params, max_round=800, n_splits=5):
        self.params_dict = params_dict
        self.params = params
        self.params_temp = {}
        self.n_splits = n_splits
        self.max_round = max_round
        self.sframe = pd.DataFrame()
        self.max_item = 0
        self.max_score = 0



    def tune_seq(self, x_train, y_train, x_test, folds, verbose_eval=False):
        """ Tune parameters sequentially

        :return:
        """
        print("\ntuning starts...")


        for key in self.params_dict.keys():
            for item in self.params_dict[key]:
                print('Tuning for parameter %s with value %f' % (key, item))
                self.params_temp = self.params
                self.params_temp.update({key: item})
                cv = Cross_Validate(None, n_splits=self.n_splits, len_trn=x_train.shape[0], len_tst=x_test.shape[0],
                                    clf=-1, params=self.params, max_round=self.max_round)
                cv.cross_validate_xgb(x_train, y_train, x_test, folds, verbose_eval)
                self.params_temp.update({'score': cv.trn_gini})
                self.sframe = pd.concat([self.sframe, pd.Series(
                    self.params_temp.values(),
                    index=self.params_temp.keys())
                                         ], axis=1)

                if cv.trn_gini > self.max_score:
                    self.max_item = item
                    self.max_score = cv.trn_gini

            self.params.update({key: self.max_item})

            self.max_item = 0
            self.max_score = 0
        self.sframe = self.sframe.transpose().reset_index()


