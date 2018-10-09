#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 01:50:22 2017

@author: vikashsingh
"""

import pandas as pd 
import numpy as np
from sklearn.metrics import roc_auc_score
import random
from sklearn.ensemble import RandomForestClassifier



def Bootstrap(n, n_iter, random_state):
    if random_state:
        random.seed(random_state)
    for j in range(n_iter):
        bs = [random.randint(0, n-1) for i in range(n)]
        out_bs = list({i for i in range(n)} - set(bs))
        yield bs, out_bs
        
        
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

for y in range (36500,36700, 500):
    globpred=[]
    globy_test=[]
    globpred_np=[]
    AUC=[]
        
    for x in range(y,y+500):
        boot = Bootstrap(n=681, n_iter=1, random_state=x)  

        for train_idx, validation_idx in boot:
            forest = RandomForestClassifier(n_estimators = 100, random_state = 5) 
            forest.fit(X[train_idx], Y[train_idx])
            predictions = forest.predict(X[validation_idx])
            predictionsproba = forest.predict_proba(X[validation_idx])[:,1]
            
            AUC.append(roc_auc_score(Y[validation_idx], predictionsproba))

        print("Bootstrap AUC is: " + str(np.mean(AUC)))
        
        



        
 

