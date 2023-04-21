# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 16:19:17 2022

@author: wyx
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from lifelines import CoxPHFitter

min_max_scaler = preprocessing.MinMaxScaler() 
#simulation data 
TTE = pd.DataFrame(pd.read_excel("/.../TMB_simu.xlsx",sheet_name='Sheet2'))
cph = CoxPHFitter()
cph.fit(TTE, 't.simu', event_col='delta')
cph.print_summary()
pro_sur = np.copy(cph.predict_survival_function(TTE,6).T)

ORR = np.copy(pd.DataFrame(pd.read_excel("/.../TMB_simu.xlsx",sheet_name='Sheet3')))

lr = LogisticRegression(penalty="l2", C=0.8, solver="liblinear", multi_class='auto')
lr.fit(ORR[:,0:2], ORR[:,2])
cla_orr = lr.predict(ORR[:,0:2])

pro_orr = lr.predict_proba(ORR[:,0:2])[:,1]
print(accuracy_score(lr.predict(ORR[:,0:2]), ORR[:,2]))

Sample_simu = np.copy(pd.DataFrame(pd.read_excel("/.../TMB_simu.xlsx")))  
X_simu = Sample_simu[:,1:3]
y_simu_d1 = Sample_simu[:,3]
y_simu_d2 = np.column_stack((pro_orr,pro_sur))
y_simu_bag = Sample_simu[:,8]
#true data
TTE = pd.DataFrame(pd.read_excel("/.../TMB_syucc.xlsx",sheet_name='Sheet2'))

cph = CoxPHFitter()
cph.fit(TTE, 'PFS', event_col='Status')
cph.print_summary()
pro_sur2 = np.copy(cph.predict_survival_function(TTE,4).T)

ORR = np.copy(pd.DataFrame(pd.read_excel("/.../TMB_syucc.xlsx",sheet_name='Sheet3')))
lr = LogisticRegression(penalty="l2", C=0.1, solver="liblinear", multi_class='auto')
lr.fit(ORR[:,0:4], ORR[:,4])
cla_orr2 = lr.predict(ORR[:,0:4])
pro_orr2 = lr.predict_proba(ORR[:,0:4])[:,-1]+lr.predict_proba(ORR[:,0:4])[:,1]
print(accuracy_score(lr.predict(ORR[:,0:4]), ORR[:,4]))

Sample_syucc = np.copy(pd.DataFrame(pd.read_excel("/.../TMB_syucc.xlsx")))  
X_syucc = Sample_syucc[:,1:5]
y_syucc_d1 = Sample_syucc[:,7]
y_syucc_d2 = np.column_stack((pro_orr2,pro_sur2))

#input data
X = X_simu
y = np.column_stack((y_simu_d2,y_simu_bag))
X = min_max_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

#generate_bags
def generate_bags(X,y):           
    bag_label = np.unique(y[:, 2])   
    j = np.zeros(len(bag_label), dtype=int)
    for i in range(len(X)):  
        for k in range(len(bag_label)):
            if y[i, 2] == k:
                j[k] += 1
                
    bags = np.zeros((len(bag_label),max(j),len(X[0])))     
    bags_pro = np.zeros((len(bag_label),max(j), 2))
    bags_label2 = np.zeros((len(bag_label), 2))  
                    
    j = np.zeros(len(bag_label), dtype=int)
    
    for i in range(len(X)):  
        for k in range(len(bag_label)):
            if y[i, 2] == k:
                bags[k,j[k],:] = X[i]
                bags_pro[k,j[k],:] = y[i, 0:2]
                j[k] += 1
    
    for k in range(len(bag_label)):
        idx = np.argwhere(np.all(bags_pro[k]==0, axis=1))
        bags_label2[k] = np.mean(np.delete(bags_pro[k],idx,axis=0),axis = 0)
        
    return bags, bag_label, bags_label2

bags, bags_label, bags_label2 = generate_bags(X,y)
bags_train, bags_train_label, bags_train_label2 = generate_bags(X_train,y_train)
bags_num = len(bags_label)