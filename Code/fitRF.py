import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import time

def fit_RF(Train_X, Train_Y, Test_X, Test_Y, Predefined_Split):
    
    #  1. Fitting
    # In article, 
    # - n_estimators: 120
    #     Reported number of trees is defined by 120.
    #   All other hyperparameters are not explored. Default values are chosen here.
    Fit_RF = GridSearchCV(\
        RandomForestRegressor(n_estimators = 120, n_jobs=-1), cv = Predefined_Split, \
        param_grid = {})
    Time0 = time.time()
    Fit_RF.fit(Train_X, Train_Y)
    Time1 = time.time()
    Time_Train = Time1 - Time0
    
    #  2. Prediction and Error
    Time0 = time.time()
    Predict_Y = Fit_RF.predict(Test_X)
    # debug
    #print "Predixt_Y = ", Predict_Y
    Err_MAD  = mean_absolute_error(Test_Y, Predict_Y)
    Err_RMSD = mean_squared_error (Test_Y, Predict_Y)
    Time1 = time.time()
    Time_Test = Time1 - Time0
    
    #  3. Final return
    return (Err_MAD, Err_RMSD, Time_Train, Time_Test)