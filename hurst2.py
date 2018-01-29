# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 00:26:04 2018

@author: Andrew Selzler
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def hurst(ts):
  # Returns the Hurst Exponent of the time series vector ts
  # Create the range of lag values
  lags = range(2, 100)

  # Calculate the array of the variances of the lagged differences
  tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
  plt.plot(np.log(lags),np.log(tau))
  plt.show()

  # Use a linear fit to estimate the Hurst Exponent
  poly = np.polyfit(np.log(lags), np.log(tau), 1)

  # Return the Hurst exponent from the polyfit output
  return poly[0]*2.0
  
EX=[1]
for i in range(1,2000):
    if EX[i-1]==1:
        EX.append(np.random.choice([1,-1],p=[0.9,0.1]))
    else:
        EX.append(np.random.choice([1,-1],p=[0.1,0.9]))
EX=np.cumsum(np.array(EX))
plt.plot(EX)
plt.show()
    
name='C:/Users/Andrew Selzler/Desktop/SPY Historical Data (1).csv'
df=pd.read_csv(name)
df=df.drop([len(df)-2,len(df)-1],axis=0)
#df=df.iloc[::-1].reset_index()
df['Price']=df['Price'].apply(float)

print(hurst(np.array(df['Price'])))
print(hurst(EX))
plt.plot(df['Price'])
plt.show()