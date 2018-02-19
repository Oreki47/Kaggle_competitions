import sys
sys.path.append("../")

from lib.preprocess import load_all, downcast
from lib.cv import Cross_Validate
from lib.utilities import sub_to_csv, plot_importance
from sklearn.model_selection import  KFold
from datetime import datetime
from lib.feature import feature_engineering_6
from sklearn.linear_model import LogisticRegression
import gc
import os


def log06(x_train, y_train, x_test, folds, max_round, n_splits=5):
    clf = LogisticRegression(
        penalty='l2',
        dual=False,
        tol=0.0001,
        C=0.005,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight='balanced',
        random_state=None,
        solver='sag',
        max_iter=200,
        multi_class='ovr',
        verbose=0,
        warm_start=False,
        n_jobs=4,
    )
    # Additional processing of data
    x_train, x_test = feature_engineering_6(x_train, x_test, y_train)


    # Cross Validate
    cv = Cross_Validate(log06.__name__, n_splits, x_train.shape[0], x_test.shape[0], clf, -1, -1)
    cv.cross_validate(x_train, y_train, x_test, folds, verbose_eval=True)

    return cv.trn_gini, cv.y_trn, cv.y_tst, cv.fscore


if __name__ == '__main__':
    print "Level 0 model test..."
    os.chdir("..")  # set to parent dir

    # ===============================================================
    # Global params
    # ===============================================================
    MAX_ROUND = 1200
    # ===============================================================
    # Load and downcast
    # ===============================================================
    train, x_test, sub = load_all()
    x_train = train.drop('target', axis=1)
    y_train = train.target
    x_train, x_test = downcast(x_train, x_test)

    del train
    gc.collect()
    # ===============================================================
    # Run cv and bags
    # ===============================================================
    n_splits = 5
    folds = list(KFold(n_splits=n_splits, shuffle=True, random_state=47).split(x_train, y_train))
    trn_gini, y_trn, y_tst, fscore = log06(x_train, y_train, x_test, folds, MAX_ROUND, n_splits)

    # ===============================================================
    # Save submission and figure
    # ===============================================================
    if MAX_ROUND > 300:
        sub.target = y_tst
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        sub_to_csv(sub, y_trn, trn_gini, log06.__name__, current_time)
        plot_importance(fscore, trn_gini, log06.__name__, current_time)