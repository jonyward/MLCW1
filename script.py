# import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

# Link the script to the dataset

df = pd.read_csv("data/high_diamond_ranked_10min.csv")
df.head()

# Drop columns which aren't needed for the model to work properly

df = df.drop(labels=['redWardsPlaced', 'redWardsDestroyed', 'redFirstBlood', 'redKills', 'redDeaths', 'redAssists', 'redEliteMonsters', 'redDragons', 'redHeralds', 'redTowersDestroyed', 'redTotalGold', 'redAvgLevel', 'redTotalExperience', 'redTotalMinionsKilled', 'redTotalJungleMinionsKilled', 'redGoldDiff', 'redExperienceDiff', 'redCSPerMin', 'redGoldPerMin'], axis=1)
df = df.drop(labels=['gameId'], axis=1)
print(df.shape)

# Show top 20 instances of the dataset

df.head(20)

# check for any duplicate rows in the dataset, and if so remove them

duplicate_rows_df = df[df.duplicated()]
print ("Duplicate rows: ", duplicate_rows_df.shape)
df.count()

df = df.drop_duplicates()
print(df.count())
print(df.isnull().sum())

# Check there are even amount of instances where blue team loses but also wins their games

print(df['blueWins'].value_counts())

# Specify that the variable 'blueWins' is the target varaible of the SVM model, while the rest of the dataset is represented by another variable

X = df.drop(['blueWins'], axis = 1)
y = df['blueWins']

print(X.head())
print(y[0:5])

# create variables used in the classification, specifying how much of the dataset is split into 3 individual datasets which are used for indiidual models

X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=1)
print("Size of training X:", X_train.shape)

X_validation, X_test, y_validation, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=1)
print("Size of the validation X:", X_validation.shape)
print("Size of the test X:", X_test.shape)

# First SVM classification is created, with its accuracy printed out

clf1 = SVC()
clf1 = clf1.fit(X_train, y_train)

y_pred1 = clf1.predict(X_validation)
print("Accuracy:", metrics.accuracy_score(y_validation, y_pred1))

# GridSearchCV tuning to perfect the previously made model to try increase its accuracy
# This also will find any hyperparameters which can be used for testing models using the validation and testing datasets previously declared

from sklearn.model_selection import GridSearchCV

parameters = [{'kernel': ['linear'], 'gamma':[1, 0.1, 0.01], 'C': [1, 10, 100, 1000]}]

clf2 = SVC()
grid = GridSearchCV(clf2, parameters, cv=5, scoring='accuracy', verbose=10)
grid.fit(X_train, y_train)
print('Best Hyper-parameters: ', grid.best_params_)
print('Accuracy: ', grid.best_score_)

# Hyperparameters found from GridSearchCV tuning used to increase accuracy of classification of validation data set

clf3 = SVC(kernel = 'linear', C=10)
clf3 = clf3.fit(X_train, y_train)
y_pred3 = clf3.predict(X_validation)
print("Accuracy:", metrics.accuracy_score(y_validation, y_pred3))

# Classification of testing data set

y_pred = clf3.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))