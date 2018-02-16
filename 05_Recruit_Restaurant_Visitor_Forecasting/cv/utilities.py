import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error


def score_valid(y_true, y_valid):
    score = np.sqrt(mean_squared_error(y_true, y_valid))
    return score

def get_store_ids():
    df = pd.read_csv('../data/air_store_info.csv')
    return df['air_store_id']