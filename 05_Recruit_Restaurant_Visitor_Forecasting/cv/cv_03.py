'''
    CV strategy 03 with time series cross validation,
    The key here is rolling, basically we gradually increase the training set,
    while maintaining a 6/39 split between public validation and private validation
'''

import gc
import sys
import numpy as np
import pandas as pd


sys.path.append("../")
from cv.utilities import score_valid


def cross_validate(full_data, clf, seed, ntrain, ntest, features, target, nfolds=5):
    # define
    lst_vld_date = pd.to_datetime('2017-4-22')  # one day before test
    lst_v06_date = lst_vld_date - pd.DateOffset(days=38)
    lst_v33_date = lst_vld_date - pd.DateOffset(days=32)
    step_by_days = 10
    # folds

    # split train/valid
    trn = full_data[:ntrain]
    tst = full_data[ntrain:]

    # setup train on test
    x_trn = trn[features].values
    y_trn = trn[target].values
    x_tst = tst[features].values

    # remove unnecessary data sets
    del full_data
    gc.collect()

    v06_scoring = np.zeros((nfolds,))
    v33_scoring = np.zeros((nfolds,))
    oof_tst = np.zeros((ntest,))
    oof_tst_fld = np.empty((ntest, nfolds))
    oof_score = [None] * 2

    for i in range(nfolds):
        # first date of each set
        v06_date = lst_v06_date - pd.DateOffset(days=(nfolds - i - 1) * step_by_days)
        v33_date = lst_v33_date - pd.DateOffset(days=(nfolds - i - 1) * step_by_days)

        # setup train on valid
        x_tvd = trn[trn.visit_date < v06_date][features].values
        y_tvd = trn[trn.visit_date < v06_date][target].values
        assert (x_tvd.shape[0] == y_tvd.shape[0])

        x_v06 = trn[trn.visit_date.between(v06_date, v33_date - pd.DateOffset(days=1))][features].values
        y_v06 = trn[trn.visit_date.between(v06_date, v33_date - pd.DateOffset(days=1))][target].values

        x_v33 = trn[trn.visit_date.between(v33_date, v33_date + pd.DateOffset(days=32))][features].values
        y_v33 = trn[trn.visit_date.between(v33_date, v33_date + pd.DateOffset(days=32))][target].values
        # train on train_valid set and predict on v06/v33 set
        clf.train(x_tvd, y_tvd, x_v33, y_v33)
        v06_scoring[i] = score_valid(y_v06, clf.predict(x_v06))
        v33_scoring[i] = score_valid(y_v33, clf.predict(x_v33))

        # train on train_full set and predict on test
        clf.train(x_trn, y_trn, None, None)

        oof_tst_fld[:, i] = clf.predict(x_tst)

        del x_tvd, y_tvd, x_v06, x_v33
        gc.collect()

    oof_tst[:] = oof_tst_fld.mean(axis=1)
    oof_score[0] = np.mean(v06_scoring)
    oof_score[1] = np.mean(v33_scoring)

    sub = tst[['id', target]]
    sub[target] = np.clip(np.expm1(oof_tst.reshape(-1, 1)), a_min=0, a_max=1000)

    v06 = None
    v33 = None

    return sub, v06, v33, oof_score

