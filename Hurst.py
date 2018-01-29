# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 22:01:41 2018

@author: Andrew Selzler
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def split(arr,n_pieces):
    rem=len(arr)-len(arr)%n_pieces
    newarr=arr[:rem]
    result=[]
    for i in range(n_pieces):
        result.append(newarr[i*len(newarr)//n_pieces:(i+1)*len(newarr)//n_pieces])
    return result
    
def Z(arr):
    Y=np.array([arr[i]-np.mean(arr) for i in range(len(arr))])
    return np.cumsum(Y)
    
def E(arr,n):
    A=split(arr,len(arr)//n)
    result=0
    for subarr in A:
        R=max(Z(subarr))-min(Z(subarr))
        S=np.std(np.array(subarr))
        result+=R/S
    return result/len(A)
    
EX=[1]
for i in range(1,2000):
    if EX[i-1]==1:
        EX.append(np.random.choice([1,-1],p=[0.8,0.2]))
    else:
        EX.append(np.random.choice([1,-1],p=[0.2,0.8]))
EX=np.cumsum(np.array(EX))
plt.plot(EX)
plt.show()
    
name='C:/Users/Andrew Selzler/Desktop/CHF JPY Historical Data.csv'
df=pd.read_csv(name)
df=df.drop([len(df)-2,len(df)-1],axis=0)
df=df.iloc[::-1].reset_index()
df['Price']=df['Price'].apply(float)

f=[]
for i in range(4):
    f.append(E(np.array(df['Price']),len(df['Price'])//(2**(4-i))))
f=np.array(f)
x_axis=np.array([len(df['Price'])//(2**(4-i)) for i in range(4)])
plt.scatter(x_axis,f)
plt.show()
plt.scatter(np.log(x_axis),np.log(f))
plt.show()

model=LinearRegression()
model.fit(np.log(x_axis).reshape(-1,1),np.log(f).reshape(-1,1))
print('Hurst exponent is '+str(model.coef_[0][0]))


    