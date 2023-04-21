#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 15:31:15 2023

@author: wangyixuan
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import chi2

X = X_simu
y = y_simu_d1
X = min_max_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

#training process
feature_num = len(X[0])
out_num = 1

class Net(nn.Module):
    def __init__(self,feature_num,out_num):
        super(Net,self).__init__()

        self.fc1 = nn.Linear(feature_num,50)
        self.fc2 = nn.Linear(50,20)
        self.fc3 = nn.Linear(20,out_num)
    def forward(self,x):
        x1 = self.fc1(x)
        x2 = self.fc2(x1)
        x3 = self.fc3(x2)
        x4 = torch.sigmoid(x3)
        return x4
    
class Loss_criterion(nn.Module):
    def __init__(self):
        super(Loss_criterion, self).__init__()

    def forward(self, x, y):
        hl_loss = torch.mean((((x-y)**2)/x) + (((x-y)**2)/(1-x)) )
        return hl_loss


criterion = Loss_criterion()
net = Net(feature_num = feature_num, out_num = out_num)
optimizer = optim.Adam(net.parameters(),lr =0.005)

for i in range(300):
    net.zero_grad()
    out = net(torch.tensor(X_train, dtype=torch.float32))
    target = torch.tensor(y_train, dtype=torch.float32)
    loss_train = criterion(out, target)
    print("loss_train",loss_train)    
    loss_train.backward(retain_graph=True)
    optimizer.step()
print("train loss",loss_train) 

out_test = net(torch.tensor(X_test,dtype=torch.float32))
target_test = torch.tensor(y_test)
loss_test = criterion(out_test, target_test)
print("test loss:",loss_test)

for i in range(300):
    net.zero_grad()
    out = net(torch.tensor(X, dtype=torch.float32))
    target = torch.tensor(y, dtype=torch.float32)
    loss = criterion(out, target)
    #print("loss",loss)    
    loss.backward(retain_graph=True)
    optimizer.step()
print("all loss",loss) 

#HL statistics
'''
    data: dataframe format, with ground_truth label name is y,
                                 prediction value column name is y_hat
'''

data=np.column_stack((out.detach().numpy(),y))
data=pd.DataFrame(data)
data.columns=['y_hat','y']
Q=20

data = data.sort_values('y_hat')

data['Q_group'] = pd.qcut(data['y_hat'], Q, duplicates='drop')
    
y_p = data['y'].groupby(data.Q_group).sum()
y_total = data['y'].groupby(data.Q_group).count()
y_n = y_total - y_p
    
y_hat_p = data['y_hat'].groupby(data.Q_group).sum()
y_hat_total = data['y_hat'].groupby(data.Q_group).count()
y_hat_n = y_hat_total - y_hat_p
    
hltest = (((y_p - y_hat_p)**2 / y_hat_p) + ((y_n - y_hat_n)**2 / y_hat_n)).sum()

pval = 1 - chi2.cdf(hltest, Q-2)

#goodness_of_fit

def __ssr(y_hat, y):
    """
    regression sum of squares
    :param y_hat: List[int] or array[int] 
    :param y: List[int] or array[int] 
    :return: SSR
    """
    y_mean = sum(y) / len(y)
    s_list =[(y - y_mean)**2 for y in y_hat]
    ssr = sum(s_list)
    return ssr


def __sst(y):
    y_mean = sum(y) / len(y)
    s_list =[(y - y_mean)**2 for y in y]
    sst = sum(s_list)
    return sst

def __sse(y_hat, y):
    s_list = [(y_hat[i] - y[i])**2 for i in range(len(y_hat))]
    sse = sum(s_list)
    return sse


def goodness_of_fit(y_hat, y):
    SSR = __sse(y_hat, y)
    SST = __sst(y)
    rr = SSR /SST
    return rr 
    
print('\nHL-chi2({}): {}, p-value: {}\n'.format(Q-2, hltest, pval))
R = goodness_of_fit(out.detach().numpy(), y)
print('goodness_of_fit',R)    
