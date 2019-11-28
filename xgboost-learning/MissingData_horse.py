# -*- coding: utf-8 -*-

#####################################################################
# File Name:  MissingData_horse.py
# Author: shenming
# Created Time: Mon Nov 25 23:45:00 2019
#####################################################################

import os
import sys

# binary classification, missing data
from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import Imputer

# load data
dataframe = read_csv("./data/horse/horse-colic.data", delim_whitespace=True, header=None)
dataset = dataframe.values
# split data into X and y
X = dataset[:,0:27]
Y = dataset[:,27]
# set missing values to 0
#X[X == '?'] = 0  # got result 83.84%
#X[X == '?'] = 1  # got result 79.80%
X[X == '?'] = np.nan  # got result 85.86%
# convert to numeric
X = X.astype('float32')
# impute missing values as the mean  got result 79.80% 搭配 X[X == '?'] == np.nan
# Imputation transformer for completing missing values
# class sklearn.preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True)
# imputer = Imputer()
# imputed_x = imputer.fit_transform(X)
# X = imputed_x

# encode Y class values as integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(Y)
label_encoded_y = label_encoder.transform(Y)
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, label_encoded_y,
test_size=test_size, random_state=seed)

# fit model on training data
model = XGBClassifier()
model.fit(X_train, y_train)
print(model)
# make predictions for test data
predictions = model.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


if __name__ == "__main__":
    pass
