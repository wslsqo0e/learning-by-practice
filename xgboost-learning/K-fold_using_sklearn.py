# -*- coding: utf-8 -*-

#####################################################################
# File Name:  K-fold_using_sklearn.py
# Author: shenming
# Created Time: Fri Nov 29 00:36:16 2019
#####################################################################

import os
import sys

# k-fold cross validation evaluation of xgboost model
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# load data
dataset = loadtxt('./data/pima-indians-diabetes.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]
# CV model
model = XGBClassifier()
kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# stratified (分层的) k-fold cross validation evaluation of xgboost model
# 猜测划分k-fold时，尽量按照Y同分布进行划分
from sklearn.model_selection import StratifiedKFold
X = dataset[:,0:8]
Y = dataset[:,8]
# CV model
model = XGBClassifier()
kfold = StratifiedKFold(n_splits=10, random_state=7)
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))



if __name__ == "__main__":
    pass
