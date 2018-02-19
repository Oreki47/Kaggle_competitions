import pandas as pd
import numpy as np
from sklearn import linear_model as lm
from sklearn import cross_validation
import Tit_lib as tl
from sklearn.ensemble import BaggingRegressor


# Load rescaled dataframe
df = tl.load_file('train_rescaled.csv')

# Features to Numpy
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()
# Outcome: survive/dead
y = train_np[:, 0]
X = train_np[:, 1:]
# Generate Model
clf = lm.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
# Bagging Regressor
bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True,
                               bootstrap_features=False, n_jobs=-1)
if __name__ == '__main__':
    bagging_clf.fit(X, y)
# Check coefficient
#     tl.print_full(pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(bagging_clf.coef_.T)}))
#     print cross_validation.cross_val_score(clf, X, y, cv=5)
#     tl.plot_learning_curve(bagging_clf, "Learning Curve", X, y)

    data_test = tl.load_file("test.csv")
    df_test = tl.load_file('test_rescaled.csv')
    # Use model to generate result
    test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    predictions = bagging_clf.predict(test)
    result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
    tl.store_file(result, "predictions.csv")


# Cross-Validation and more
split_train, split_cv = cross_validation.train_test_split(df, test_size=0.3, random_state=0)
train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# Generate Model
clf = lm.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(train_df.as_matrix()[:,1:], train_df.as_matrix()[:,0])
# Predict
cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(cv_df.as_matrix()[:,1:])

# Pull bad cases
# origin_data_train = pd.read_csv("E:/OneDrive/Software Data/python/Kaggle/01_Titanic/train.csv")
# bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(split_cv[predictions != cv_df.as_matrix()[:,0]]['PassengerId'].values)]
# tl.print_full(bad_cases)


