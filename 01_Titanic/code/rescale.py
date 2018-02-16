import pandas as pd
import sklearn.preprocessing as preprocessing
import Tit_lib as tl

#Load Data and set missing ages
data_train = tl.load_file("train.csv")
data_train, rfr = tl.set_missing_ages(data_train)
data_train = tl.set_Cabin_type(data_train)

# factorization
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')
df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Cabin_No', 'Embarked_C', 'Embarked_Q', 'Pclass_2'], axis=1, inplace=True)

# Scaling
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'].reshape(-1, 1))
df['Age_scaled'] = scaler.fit_transform(df['Age'].reshape(-1, 1), age_scale_param)
fare_scale_param = scaler.fit(df['Fare'].reshape(-1, 1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].reshape(-1, 1), fare_scale_param)

# Store train data for later use
tl.store_file(df, "train_rescaled.csv")


# Load Train and perform same data processing
data_test = tl.load_file('test.csv')
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges

data_test = tl.set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')

df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Cabin_No', 'Embarked_C', 'Embarked_Q', 'Pclass_2'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].reshape(-1, 1), age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].reshape(-1, 1), fare_scale_param)

# Store test data for later use
tl.store_file(df_test, "test_rescaled.csv")