#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 13:50:00 2016   

@author: vikashsingh
"""
import os 
os.chdir('/Users/vikashsingh/Desktop/Final Cedars Project/FinalCSVsDelongCI/')

import pandas as pd 
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import StratifiedKFold 
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




data = pd.read_csv('~/Desktop/Cedars.csv')
data=data.dropna()


X=data.iloc[0:715,0:31] 
#rint("X shape")

Y=data['Outcome']

X=X.dropna()
Y=Y.dropna()


X=X.as_matrix()
Y=Y.as_matrix()


AUCperseedave=[]
X=X.astype("float32")
Y=Y.astype("float32")

concatAUC=[]#This will be used to store all the concat AUCs for the 10 rounds of cross validation


for x in range(0,1000,10):#10 rounds of repeated cross validation, averaging the results of 10 seeds (each of which os averaged over the folds) 
    globepred=[]
    globy_test=[]
    predbin=[]
    for y in range (x,x+10):
        kfold=StratifiedKFold(y=Y, n_folds=10, shuffle=True, random_state=y) 
        AUC=[]
        for i, (train,test) in enumerate(kfold):#should be 100 concatenations added 
           forest = RandomForestClassifier(n_estimators = 100, random_state = 1) 
           forest.fit(X[train], Y[train])
           predictions=forest.predict(X[test]) 
           predictionsprob=forest.predict_proba(X[test])[:,1] 
           predlist=predictions.tolist() 
           truelist=Y[test].tolist()
           globepred+=predictionsprob.tolist()
           globy_test+=truelist
           predbin+=predlist
           AUC.append(roc_auc_score(Y[test], predictionsprob))  
          

    print(perf_measure(globy_test, predbin))
    AUCperseedave.append(np.mean(AUC))
    print(roc_auc_score(globy_test, globepred))  
    concatAUC.append(roc_auc_score(globy_test, globepred))

print("Minimum is: " )
print(my_min_function(AUCperseedave))

print("Maximum is:" )
print(my_max_function(AUCperseedave)) 

print("Minimum concat is: " )
print(my_min_function(concatAUC))

print("Maximum concat is:" )
print(my_max_function(concatAUC)) 

print("Average concat is: ")
print(np.mean(concatAUC))

