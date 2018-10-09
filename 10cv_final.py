#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 13:50:00 2016   

@author: vikashsingh
"""
#Includes the proper way to get a single AUC curve from cross validation procedures 
import os 
os.chdir('/Users/vikashsingh/Desktop/Final Cedars Project/FinalCSVsDelongCI/')

import pandas as pd 
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier



def my_min_function(somelist): 
    min_value = None
    for value in somelist:
        if not min_value:
            min_value = value
        elif value < min_value:
            min_value = value
    return min_value
    
def my_max_function(someList):
    max_value=None
    for value in someList:
        if not max_value:
            max_value=value
        elif value>max_value:
            max_value=value
    return max_value

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
    for i in range(len(y_hat)): 
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==0:
           TN += 1
    for i in range(len(y_hat)): 
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1
    return(TP+FN, FP+TN)

seed = 7
np.random.seed(seed)
data = pd.read_csv('~/Desktop/Cedars.csv')
data=data.dropna()
X=data.iloc[0:715,0:31] 
Y=data['Outcome']
X=X.dropna()
Y=Y.dropna()
X=X.as_matrix()
Y=Y.as_matrix()


AUCave=[]
AUCmerged=[]
X=X.astype("float32")
Y=Y.astype("float32")



for x in range(0,100):
    kfold=StratifiedKFold(y=Y, n_folds=10, shuffle=True, random_state=x) 
    AUC=[]
    globpred=[]
    globy_test=[]
    predbin = []
    for i, (train,test) in enumerate(kfold):
        forest = RandomForestClassifier(n_estimators = 100, random_state = 1) 
        forest.fit(X[train], Y[train])
        predictions=forest.predict(X[test])
        predictionsproba=forest.predict_proba(X[test])[:,1]
        
        AUC.append(roc_auc_score(Y[test], predictionsproba))
        predlist=predictionsproba.tolist()
        truelist=Y[test].tolist()
        globy_test+=truelist
        globpred+=predlist
        predbin+=predictions.tolist()
        false_positive_rate, true_positive_rate, thresholds= roc_curve(Y[test], predictionsproba) 
        roc_auc = auc(false_positive_rate, true_positive_rate)

    print("The average ROC for the cross validation rounds is: ") 
    print(np.mean(AUC))
    AUCave.append(np.mean(AUC))  
    
    print("Concatenated ROC: " + str(x))
    print(roc_auc_score(globy_test, globpred))
    print(perf_measure(globy_test, predbin))
    AUCmerged.append(roc_auc_score(globy_test, globpred))
