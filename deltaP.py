# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 20:39:30 2018

@author: Andrew Selzler
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import train_test_split

def deltaP(c1,c2,c3,n1,n2,n3,arr):
    #n1,n2,n3<len(arr)
    model1=LinearRegression()
    t=np.array([i for i in range(len(arr))])
    # fit first order polynomial and calculate slope
    t1=t[len(arr)-1-n1:].reshape(-1,1)
    P1=arr[len(arr)-1-n1:].reshape(-1,1)
    model1.fit(t1,P1)
    slope=model1.coef_[0]
    # fit secord order polymomial with n2 points and find concavity
    model2=LinearRegression()
    t2=t[len(arr)-1-n2:]
    P2=arr[len(arr)-1-n2:]
    data=pd.DataFrame()
    data['t']=t2
    data['tsquared']=t2**2
    #print(data)
    #print(P2)
    model2.fit(data,P2)
    concavity=model2.coef_[1]
    # find moving average over n3 days and calculate difference between that and most recent value
    differential=arr[-1]-np.mean(arr[len(arr)-1-n3:])
    
    return c1*slope[0]+c2*concavity+c3*differential
    
def make_dataset(arr,n_l,n_s,n_ma):
    #n_l>n_s,n_ma
    short_slope=[]
    long_slope=[]
    short_concav=[]
    long_concav=[]
    differential=[]
    change=[]
    for i in range(n_l,len(arr)-1):
        model_s=LinearRegression()
        model_l=LinearRegression()
        data_s=pd.DataFrame()
        data_l=pd.DataFrame()
        t1_s=[j for j in range(i-n_s,i)]
        t2_s=[x**2 for x in t1_s]
        t1_l=[j for j in range(i-n_l,i)]
        t2_l=[x**2 for x in t1_l]
        data_s['t']=t1_s
        data_s['t2']=t2_s
        data_l['t']=t1_l
        data_l['t2']=t2_l
        price_s=np.array([arr[j] for j in range(i-n_s,i)])
        price_l=np.array([arr[j] for j in range(i-n_l,i)])
        model_s.fit(data_s,price_s)
        model_l.fit(data_l,price_l)
        short_slope.append(model_s.coef_[0])
        short_concav.append(model_s.coef_[1])
        long_slope.append(model_l.coef_[0])
        long_concav.append(model_l.coef_[1])
        differential.append(arr[i]-np.mean(arr[i-n_ma:i]))
        change.append(arr[i+1]-arr[i])
    data=pd.DataFrame()
    data['short_slope']=short_slope
    data['long_slope']=long_slope
    data['short_concav']=short_concav
    data['long_concav']=long_concav
    data['differential']=differential
    data['change']=change
    return data


name='C:/Users/Andrew Selzler/Desktop/SPY Historical Data (1).csv'
df=pd.read_csv(name)
df=df.drop([len(df)-2,len(df)-1],axis=0)
#df=df.iloc[::-1].reset_index()
df['Price']=df['Price'].apply(float)
plt.plot(df['Price'])
plt.show()

prices=[]
for i in range(len(df['Price'])):
    if i%30==0:
        prices.append(df['Price'][i])
plt.plot(prices)
plt.show()



data=make_dataset(prices,30,80,5)
print('dataset made')
model=GradientBoostingRegressor(learning_rate=0.001,n_estimators=10000,subsample=0.4,max_depth=5)
y=data['change']
X=data.drop(['change'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
