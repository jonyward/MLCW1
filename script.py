# import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC


df = pd.read_csv("data/high_diamond_ranked_10min.csv")
df.head()

# check the number of rows and columns in the dataset

df = df.drop(labels=['redWardsPlaced', 'redWardsDestroyed', 'redFirstBlood', 'redKills', 'redDeaths', 'redAssists', 'redEliteMonsters', 'redDragons', 'redHeralds', 'redTowersDestroyed', 'redTotalGold', 'redAvgLevel', 'redTotalExperience', 'redTotalMinionsKilled', 'redTotalJungleMinionsKilled', 'redGoldDiff', 'redExperienceDiff', 'redCSPerMin', 'redGoldPerMin'], axis=1)
df = df.drop(labels=['gameId'], axis=1)
print(df.shape)

df.head(20)

duplicate_rows_df = df[df.duplicated()]
print ("Duplicate rows: ", duplicate_rows_df.shape)
df.count()

df = df.drop_duplicates()
print(df.count())
print(df.isnull().sum())
print(df['blueWins'].value_counts())

X = df.drop(['blueWins'], axis = 1)
y = df['blueWins']

print(X.head())
print(y[0:5])

X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=1)
print("Size of training X:", X_train.shape)

X_validation, X_test, y_validation, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=1)
print("Size of the validation X:", X_validation.shape)
print("Size of the test X:", X_test.shape)

clf1 = SVC()
clf1 = clf1.fit(X_train, y_train)

y_pred1 = clf1.predict(X_validation)
print("Accuracy:", metrics.accuracy_score(y_validation, y_pred1))

from sklearn.model_selection import GridSearchCV

parameters = [{'kernel': ['linear'], 'gamma':[1, 0.1, 0.01], 'C': [1, 10, 100, 1000]}]

clf2 = SVC()
grid = GridSearchCV(clf2, parameters, cv=5, scoring='accuracy', verbose=10)
grid.fit(X_train, y_train)
print('Best Hyper-parameters: ', grid.best_params_)
print('Accuracy: ', grid.best_score_)

clf3 = SVC(kernel = 'linear', C=10)
clf3 = clf3.fit(X_train, y_train)
y_pred3 = clf3.predict(X_validation)
print("Accuracy:", metrics.accuracy_score(y_validation, y_pred3))

y_pred = clf3.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))