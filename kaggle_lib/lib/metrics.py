import numpy as np
from numba import jit


@jit
def eval_gini(y_true, y_pred):
    assert (len(y_true) == len(y_pred))
    sum_all = np.asarray(np.c_[y_true, y_pred, np.arange(len(y_true))], dtype=np.float)
    sum_all = sum_all[np.lexsort((sum_all[:, 2], -1 * sum_all[:, 1]))]
    total_losses = sum_all[:, 0].sum()
    gini_sum = sum_all[:, 0].cumsum().sum() / total_losses

    gini_sum -= (len(y_true) + 1) / 2.
    return gini_sum / len(y_true)


def eval_gini_normalized(y_true, y_pred):
    return eval_gini(y_true, y_pred) / eval_gini(y_true, y_true)

def gini_xgb(y_pred, dtrain):
    y_true = dtrain.get_label()
    gini_score = eval_gini_normalized(y_true, y_pred)
    return [('gini', gini_score)]
