import sys
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer

sys.path.append("../")


def search_model(train_x, train_y, est, param_grid, n_jobs, cv, refit=False):
##Grid Search for the best model
    loss = make_scorer(score_valid)
    model = GridSearchCV(
        estimator=est,
        param_grid=param_grid,
        scoring=loss,
        verbose=False,
        n_jobs=n_jobs,
        iid =True,
        refit=refit,
        cv=cv
    )
    # Fit Grid Search Model
    model.fit(train_x, train_y)
    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:", model.best_params_)
    return model.best_estimator_



def score_valid(y_true, y_valid):
    # should only be called by stacker
    score = np.sqrt(mean_squared_error(y_true, y_valid))
    return score
