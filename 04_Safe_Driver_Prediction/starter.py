import xgboost as xgb
from lib.preprocess import *
from lib.metrics import *
from kaggl_general.utils.general import *
from lib.feature import *
from sklearn.model_selection import train_test_split
import random
from datetime import datetime

import gc

# Global params
do_predict = 0
# xgb params
params = {}
params['eta'] = 0.02
params['objective'] = "binary:logistic"
params['eval_metric'] = 'auc'
params['max_depth'] = 4
params['silent'] = 1

train = load_train()
train = general_downcast(train)
train = handle_cat(train)

x = train.drop('target', axis=1)
y = train.target

del train; gc.collect()

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.33, random_state=random.seed(datetime.now()))

del x, y; gc.collect()

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

del x_train, x_valid; gc.collect()

watchlist = [(d_train, 'train'), (d_valid, 'eval')]
clf = xgb.train(params=params,
                dtrain=d_train,
                num_boost_round=300,
                evals=watchlist,
                verbose_eval=50,
                feval=gini_xgb)

del d_train, d_valid; gc.collect()



if do_predict:
    test = load_test()
    test = general_downcast(test)
    test = handle_cat(test)


    d_test = xgb.DMatrix(test)

    sub = load_sub()

    sub.target = clf.predict(d_test)

    save_submission_n_model(sub, clf)
