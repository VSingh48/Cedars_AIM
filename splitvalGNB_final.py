#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 16:55:25 2016

@author: vikashsingh
"""
import os 
os.chdir('/Users/vikashsingh/Desktop/Final Cedars Project/FinalCSVsDelongCI/')

import pandas as pd 
import numpy as np
import sklearn
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.cross_validation import StratifiedKFold
from sklearn import linear_model, datasets 
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KDTree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn import svm  
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

AUC=[]
X=X.astype("float32")
Y=Y.astype("float32") 


  

for x in range(0,100):
    X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.5,train_size=0.5, random_state=x) 
    forest = RandomForestClassifier(n_estimators = 100, random_state = 1) 
    forest.fit(X_train, y_train)
    predictions=forest.predict(X_test) 
    predictionsprob=forest.predict_proba(X_test)[:,1] 
    print(roc_auc_score(y_test,predictionsprob))  
    AUC.append(roc_auc_score(y_test,predictionsprob)) 
    print(perf_measure(y_test, predictions))






