# -*- coding: utf-8 -*-

#####################################################################
# File Name:  BayesParams_optimization.py
# Author: shenming
# Created Time: Tue Nov 26 13:01:34 2019
#####################################################################

import os
import sys

from copy import deepcopy
import xgboost as xgb
from xgboost import DMatrix
from sklearn.datasets import load_svmlight_file
from bayes_opt import BayesianOptimization
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger

#TRAIN_FILENAME = "data/sm_data_ctc/bihuan_type12_mixture.libsvm"
TRAIN_FILENAME = "data/sm_data_ctc/bihuan_type12_mixture.libsvm.test"
BO_LOGNAME = "log/classificaton.log"
MODEL_NAME = "model.test"
NBR_CV = 500
ESR_CV = 150
#NBR_CV = 300
#ESR_CV = 50
BO_INIT_POINTS = 5
BO_N_ITER = 10
SEED = 2009
N_JOBS = 8

# load data
train_file = TRAIN_FILENAME
x_train, y_train = load_svmlight_file(train_file)
#print(x_train[0])
#print(y_train[0])
dtrain = DMatrix(x_train, y_train)
print("load data done", flush=True)

# verify train accuracy
PROB = 0.8
model_name = MODEL_NAME
test_model = xgb.Booster(model_file=model_name)
y_predict = test_model.predict(dtrain)
ll = sum(1 for i in range(len(y_predict)) if y_predict[i] > PROB)
print(ll)
print('correct=%f' % (sum(1 for i in range(len(y_predict)) if y_predict[i] > PROB and  y_train[i] == 1) / ll))
sys.exit(0)

def bo_probe(bo_opt):
    params_list = []
    p = {
        'eta': 0.01,
        'max_depth': 4,
        'gamma': 0,
        'min_child_weight': 0.1,
        'max_delta_step': 0,
        'subsample': 0.5,
        'colsample_bytree': 0.8,
        'alpha' : 0
    }
    #'max_leaves': 32,
    params_list.append(deepcopy(p))

    p = {
        'eta': 0.36,
        'max_depth': 8,
        'gamma': 0.63,
        'min_child_weight': 10.05,
        'max_delta_step': 9.8,
        'subsample': 0.84,
        'colsample_bytree': 1.0,
        'alpha' : 0.3
    }
    params_list.append(deepcopy(p))

    p = {
        'eta': 0.5,
        'max_depth': 6,
        'gamma': 1.0,
        'min_child_weight': 200.0,
        'max_delta_step': 10.0,
        'subsample': 1.0,
        'colsample_bytree': 0.3,
        'alpha' : 0.45
    }
    params_list.append(deepcopy(p))

    p = {
        'eta': 0.2,
        'max_depth': 10,
        'gamma': 1.0,
        'min_child_weight': 10.0,
        'max_delta_step': 10.0,
        'subsample': 0.2,
        'colsample_bytree': 0.6,
        'alpha' : 0.1
    }
    params_list.append(deepcopy(p))

    for param in params_list:
        bo_opt.probe(params=param, lazy=True)

    bo_opt.maximize(init_points=0, n_iter=0)

def legalise_param(param_map):
    if 'max_depth' in param_map:
        param_map['max_depth'] = int(param_map['max_depth'])
    if 'subsample' in param_map:
        param_map['subsample'] = max(min(param_map['subsample'], 1), 0)
    if 'colsample_bytree' in param_map:
        param_map['colsample_bytree'] = max(min(param_map['colsample_bytree'], 1), 0)
    if 'max_delta_step' in param_map:
        param_map['max_delta_step'] = int(param_map['max_delta_step'])
    if 'eta' in param_map:
        param_map['eta'] = max(min(param_map['eta'], 1), 0)



SACU_BEST = -100.0
BEST_ITER = -1
nbr_cv = NBR_CV
esr_cv = ESR_CV
def xgb_evaluate_rank(eta, max_depth, gamma, min_child_weight, max_delta_step,
                      subsample, colsample_bytree, alpha):
    global SACU_BEST
    global BEST_ITER
    # Define all XGBoost parameters
    paramt = {
        'booster' : 'gbtree',
        'max_depth' : int(max_depth),
        'gamma' : gamma,
        'eta' : max(min(eta, 1), 0),
        'objective' : 'binary:logistic',
        'n_jobs' : N_JOBS,
        'silent' : True,
        'subsample' : max(min(subsample, 1), 0),
        'colsample_bytree' : max(min(colsample_bytree, 1), 0),
        'min_child_weight' : min_child_weight,
        'max_delta_step' : int(max_delta_step),
        'alpha' : max(min(alpha, 1), 0),
        'seed' : SEED,
    }
    res = xgb.cv(paramt, dtrain, num_boost_round=nbr_cv, early_stopping_rounds=esr_cv, nfold=5, metrics='error')
    s_acu = 1. - res.iloc[[res.iloc[:, 0].size - 1]].values[0][0]
    if s_acu > SACU_BEST:
        print("in bo search, s_acu: %f [beat %f], best_iter: %d" % (s_acu, SACU_BEST, res.iloc[:,0].size), flush=True)
        SACU_BEST = s_acu
        BEST_ITER = res.iloc[:,0].size
    else:
        print("not beat best score [%f], s_acu: %f%%, best_iter: %d" % (SACU_BEST, s_acu, res.iloc[:,0].size), flush=True)
    return s_acu


# BayesianOptimization specifying a function to be optimized `f`
# and its parameters with their corresponding bounds `pbounds`
xgb_bo = BayesianOptimization(f = xgb_evaluate_rank,
                              pbounds = {
                                  'eta': (0.001, 1),
                                  'max_depth': (4, 15),
                                  'gamma': (0.001, 1.0),
                                  'min_child_weight': (10, 400),
                                  'max_delta_step': (0, 20),
                                  'subsample': (0.1, 1.0),
                                  'colsample_bytree' :(0.1, 1.0),
                                  'alpha' : (0.001, 0.5)
                              },
                              random_state = 2019)

# define optimization log
# everytime it probes the function and obtains a new parameter-target combination
# it will fire an `Events.OPTIMIZATION_STEP` event, which our logger will listen to.
bo_logname = BO_LOGNAME
bo_logger = JSONLogger(path=bo_logname)
xgb_bo.subscribe(Events.OPTMIZATION_STEP, bo_logger)

# explore
bo_probe(xgb_bo)

# search
print("start search", flush=True)
bo_init_points = BO_INIT_POINTS
bo_n_iter = BO_N_ITER
# init_points: How many steps of random exploraton you want to perform. at begining
# n_iter: How many steps of bayesian optimization you want to perform.
xgb_bo.maximize(init_points=bo_init_points, n_iter=bo_n_iter, acq='ei')

# done search and train
print("best parameters and target value:")
print(xgb_bo.max)

params_best = deepcopy(xgb_bo.max['params'])
params_best['objective'] = 'binary:logistic'
params_best['seed'] = SEED
params_best['n_jobs'] = N_JOBS
legalise_param(params_best)

full_model = xgb.train(params_best, dtrain, num_boost_round=BEST_ITER)

model_name = MODEL_NAME
full_model.save_model(model_name)
print("model saved as %s" % (model_name))

# verify train accuracy
test_model = xgb.Booster(model_file=model_name)
y_predict = test_model.predict(dtrain)
print('error=%f' % (sum(1 for i in range(len(y_predict)) if int(y_predict[i] > 0.5) != y_train[i]) / float(len(y_predict))))

if __name__ == "__main__":
    pass
