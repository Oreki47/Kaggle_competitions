from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.learning_curve import learning_curve as lc
import matplotlib.pyplot as plt
import numpy as np
import re

def load_file(file_name):
    # type: (string) -> dataframe
    dataframe = pd.read_csv("E:/OneDrive/Github/python/Kaggle/01_Titanic/%s" % file_name)
    return dataframe

def store_file(df, file_name):
    df.to_csv("E:/OneDrive/Github/python/Kaggle/01_Titanic/%s" % file_name, index = False)

def set_missing_ages(df):

    # float type features
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # Part age as known and unknown
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y as target age
    y = known_age[:, 0]
    # X as Feature value
    X = known_age[:, 1:]
    # Fit into RandomForestRegressor
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # Use the model to predict unknown age
    predictedAges = rfr.predict(unknown_age[:, 1::])
    # Use the prediction to make up missing values
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df, rfr

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

# Learning Curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    train_sizes, train_scores, test_scores = lc(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Sample Number")
        plt.ylabel("Score")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label="Test Error")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label="CV Error")

        plt.legend(loc="best")

        plt.draw()
        plt.show()
        plt.gca().invert_yaxis()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff


def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ""