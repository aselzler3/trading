# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 21:25:18 2018

@author: Andrew Selzler
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as scs

def EDF(arr,binsize):
    A=sorted(arr)
    X=np.linspace(min(A),max(A),binsize)
    count=0
    result=[]
    for i in range(len(X)):
        while X[i]>=A[count]:
            count+=1
            if count>=len(A):
                break
        result.append(count/len(A))
    return result

def KStest(arr):
    A=sorted(arr)
    m=np.mean(A)
    s=np.std(A)
    binsize=5000
    edf=EDF(A,binsize)
    CDF=[scs.norm.cdf(x,loc=m,scale=s) for x in np.linspace(A[0],A[-1],binsize)]
    #plt.plot(edf)
    #plt.plot(CDF)
    #plt.show()
    diff=[edf[i]-CDF[i] for i in range(len(edf))]
    return abs(max(diff))
    
def bootstrap(arr):
    result=[]
    for _ in range(len(arr)):
        result.append(np.random.choice(arr))
    return result
    
def bootstrappedKS(arr):
    result=[]
    for i in range(40):
        result.append(KStest(bootstrap(arr)))
        if i%5==0:
            print(i)
    plt.hist(result)
    plt.show()
    return result
    
name='C:/Users/Andrew Selzler/Desktop/CHF JPY Historical Data.csv'
df=pd.read_csv(name)
df=df.drop([len(df)-2,len(df)-1],axis=0)
df=df.iloc[::-1].reset_index()
df['Price']=df['Price'].apply(float)
differential=[]
for i in range(1,len(df['Price'])):
    differential.append(df['Price'][i]-df['Price'][i-1])
K=bootstrappedKS(np.array(differential))
print('mean KS is '+str(np.mean(K)))
print('lower bount of 95% conf interval for KS is '+str(min(K)))
