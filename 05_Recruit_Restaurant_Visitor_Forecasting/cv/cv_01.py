''' Explanation

    cv strategy 01:

    The key here is to split stores into different folds

    Some problems worth concerning:
    what if the type of the stores are very different -- small NFOLDS
    some stores in the test set are not seen in the train

    This function still does too many things, i.e., it is decoupled

    This is by no means perfect.

'''

import sys
import gc
import numpy as np
import pandas as pd

sys.path.append("../")
from sklearn.model_selection import KFold
from cv.utilities import score_valid, get_store_ids


def cross_validate(full_data, clf, seed, ntrain, ntest, features, target, nfolds=5):
    # define
    lst_vld_date = pd.to_datetime('2017-4-22')  # one day before test
    v06_str = lst_vld_date - pd.DateOffset(days=38)
    v33_str = lst_vld_date - pd.DateOffset(days=32)

    # setup a validation set from 3/15 -3/20 and 3/21 - 4/22.
    trn = full_data[:ntrain]
    tst = full_data[ntrain:]
    v06 = trn[trn.visit_date.between(v06_str, v33_str - pd.DateOffset(days=1))]  # tvd removed to save memory
    v33 = trn[trn.visit_date.between(v33_str, lst_vld_date)]                     # v06/33 saved for assertions

    x_tst = tst[features].values

    # assertions
    assert (v33.visit_date.max() == lst_vld_date)
    assert ((v33.visit_date.min() - v06.visit_date.max()).days == 1)
    assert ((v06.visit_date.max() - v06.visit_date.min()).days == 5)
    assert ((v33.visit_date.max() - v33.visit_date.min()).days == 32)

    oof_v06 = np.zeros((v06.shape[0],))
    oof_v33 = np.zeros((v33.shape[0],))
    oof_tst = np.zeros((ntest,))
    oof_tst_fld = np.empty((ntest, nfolds))
    oof_score = [None] * 2

    # split stores
    store_ids = get_store_ids()
    folds = KFold(n_splits=nfolds, shuffle=True, random_state=seed).split(store_ids)

    for i, ids in enumerate(folds):
        # for predicting the test
        trn_idx = trn.air_store_id.isin(store_ids[ids[0]])

        x_trn = trn[trn_idx][features].values
        y_trn = trn[trn_idx][target].values

        # for predicting the valid
        tvd_idx = trn[trn.visit_date < v06_str].air_store_id.isin(store_ids[ids[0]])
        v06_idx = v06.air_store_id.isin(store_ids[ids[1]])
        v33_idx = v33.air_store_id.isin(store_ids[ids[1]])

        x_tvd = trn[trn.visit_date < v06_str][tvd_idx][features].values
        y_tvd = trn[trn.visit_date < v06_str][tvd_idx][target].values

        x_v06 = v06[v06_idx][features].values
        x_v33 = v33[v33_idx][features].values
        y_v33 = v33[v33_idx][target].values  # pass v33 set as early stopping sentinel

        # train on train_valid set and predict on v06/v33 set
        clf.train(x_tvd, y_tvd, x_v33, y_v33)
        oof_v06[v06_idx] = clf.predict(x_v06)
        oof_v33[v33_idx] = clf.predict(x_v33)

        # train on train_full set and predict on test
        clf.train(x_trn, y_trn, None, None)
        oof_tst_fld[:, i] = clf.predict(x_tst)

        del x_trn, y_trn, x_tvd, y_tvd, x_v06, x_v33
        gc.collect()

    oof_tst[:] = oof_tst_fld.mean(axis=1)
    oof_score[0] = score_valid(v06[target].values, oof_v06)
    oof_score[1] = score_valid(v33[target].values, oof_v33)

    sub = tst[['id', target]]
    v06 = v06[['id', target]]
    v33 = v33[['id', target]]

    sub[target] = oof_tst.reshape(-1, 1)
    v06[target] = oof_v06.reshape(-1, 1)
    v33[target] = oof_v33.reshape(-1, 1)

    return sub, v06, v33, oof_score
