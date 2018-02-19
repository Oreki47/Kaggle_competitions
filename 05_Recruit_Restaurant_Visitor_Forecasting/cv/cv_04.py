''' Explanation

    cv strategy 04: Regular K-fold cross validation

    problems:
    Essentially we might be using futures points to predict past points.
    In the perfect world where everything is not correlated, that is fine.
    However, consider such a scenario, I go to a restaurant today and it is full,
    so I decided to go tomorrow (past affecting the futures -- future carries
    information from the past)
    This may introduce some level of over-fitting, which results in local score
    lower than public score.
    This by no means says such cv strategy is not usable (it just makes less sense).


'''
import gc
import sys
import numpy as np

from sklearn.model_selection import KFold

sys.path.append("../")
from cv.utilities import score_valid

def cross_validate(full_data, clf, seed, ntrain, ntest, features, target, nfolds=5):
    trn = full_data[:ntrain]
    tst = full_data[ntrain:]

    x_tst = tst[features].values

    # define
    oof_trn = np.zeros((ntrain,))
    oof_tst = np.zeros((ntest,))
    oof_tst_fld = np.empty((ntest, nfolds))
    oof_score = [None] * 2

    folds = KFold(n_splits=nfolds, shuffle=True, random_state=seed).split(trn.id)

    for i, (train_idx, valid_idx) in enumerate(folds):
        x_trn = trn.loc[train_idx,][features].values
        y_trn = trn.loc[train_idx,][target].values

        x_vld = trn.loc[valid_idx,][features].values
        y_vld = trn.loc[valid_idx,][target].values

        clf.train(x_trn, y_trn, x_vld, y_vld)

        oof_trn[valid_idx] =clf.predict(x_vld)
        oof_tst_fld[:, i] = clf.predict(x_tst)

        del x_trn, y_trn, x_vld, y_vld
        gc.collect()

    oof_tst[:] = oof_tst_fld.mean(axis=1)
    oof_score[0] = score_valid(trn[target].values, oof_trn)
    oof_score[1] = 0.1234

    sub = tst[['id', target]]
    trn = trn[['id', target]]

    sub[target] = np.clip(np.expm1(oof_tst.reshape(-1, 1)), a_min=0, a_max=1000)
    trn[target] = np.clip(np.expm1(oof_trn.reshape(-1, 1)), a_min=0, a_max=1000)

    return sub, trn, None, oof_score