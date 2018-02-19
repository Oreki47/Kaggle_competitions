from scipy.stats import hmean
from scipy.stats.mstats import gmean
from lib.metrics import eval_gini_normalized
from lib.utilities import sub_to_csv_blend
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import pandas as pd
import numpy as np
import glob

SELECT_MEAN = 1


def select_n_best(trn_list, tst_list, n):
    scores = [float(x[-12:-4]) for x in trn_list]
    scores.sort(reverse=True)
    scores = scores[:n]

    trn_list = [x for x in trn_list if float(x[-12:-4]) in scores]
    tst_list = [x for x in tst_list if float(x[-12:-4]) in scores]

    return trn_list, tst_list


def select_stack(trn_series, tst_series, y_train):
    methods = [mean, h_mean, g_mean, r_mean, log_reg]
    idx = -1
    score = 0
    y_tst = 0
    for i in range(len(methods)):
        score_temp, y_tst_temp = methods[i](trn_series, tst_series, y_train)
        print('Method %s score: %f ' % (methods[i].__name__, score_temp))
        if score_temp > score:
            score = score_temp
            y_tst = y_tst_temp
            idx = i
    print('Using method %s' % methods[idx].__name__)

    return y_tst, score

def g_mean(trn_series, tst_series, y_train):
    score = eval_gini_normalized(y_train, gmean(trn_series, axis=1))
    y_tst = gmean(tst_series, axis=1)
    return score, y_tst

def h_mean(trn_series, tst_series, y_train):
    score = eval_gini_normalized(y_train, hmean(trn_series, axis=1))
    y_tst = hmean(tst_series, axis=1)
    return score, y_tst

def r_mean(trn_series, tst_series, y_train):
    score = eval_gini_normalized(y_train, rmean(trn_series, axis=1))
    y_tst = rmean(tst_series, axis=1)
    return score, y_tst

def mean(trn_series, tst_series, y_train):
    score = eval_gini_normalized(y_train, np.mean(trn_series, axis=1))
    y_tst = np.mean(tst_series, axis=1)
    return score, y_tst


def rmean(y, axis):
    # compute the mean rank score of y
    y = pd.DataFrame(y)
    y = y.rank(axis=0)
    y = np.mean(y, axis=1)
    y -= y.min()
    y /= y.max()
    return y

def log_reg(trn_series, tst_series, y_train):
    clf = LogisticRegression()
    clf.fit(trn_series, y_train)
    score = eval_gini_normalized(y_train, clf.predict_proba(trn_series)[:, 1])
    y_tst = clf.predict_proba(tst_series)[:, 1]
    return score, y_tst


trn_list = glob.glob('train/blend/*.csv')
tst_list = glob.glob('submission/blend/*.csv')
y_train = pd.read_csv('data/train.csv')['target']
sub = pd.read_csv('data/sample_submission.csv', index_col='id')

# trn_list, tst_list = select_n_best(trn_list, tst_list, n)

trn_series = pd.DataFrame()
tst_series = pd.DataFrame()

for trn in trn_list:
    temp = pd.read_csv(trn, names=['Train'])
    trn_series = pd.concat([trn_series, temp], axis=1)

for tst in tst_list:
    temp = pd.read_csv(tst, index_col='id')
    tst_series = pd.concat([tst_series, temp], axis=1)

scaler = MinMaxScaler((0.01, 1))
trn_series = scaler.fit_transform(trn_series)
tst_series = scaler.fit_transform(tst_series)


y_tst, score = select_stack(trn_series, tst_series, y_train)

current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

sub.target = y_tst
sub.reset_index(inplace=True)

sub_to_csv_blend(sub, score, current_time)
