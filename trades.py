# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 19:30:22 2017

@author: Andrew Selzler
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

name='C:/Users/Andrew Selzler/Desktop/NXRSdata.xlsx'
df=pd.read_excel(name, sheetname=1)
#print(df.head())
df=df.drop('Column19', axis=1)
df.columns=['date','start_time','end_time','duration','stock','type','start_price','end_price','num_shares','gross','f1','f2','f3','f4','f5','f6','f7','net']
#plt.hist(df['net'],bins=100)
#plt.show()
df['month']=[df['date'][i].month for i in range(len(df))]

def monthly_trades(ar,months):
    month_trades=[]
    for t in range(12000):
        month=np.random.choice(np.array(ar),size=len(ar)//months, replace=True)
        month_trades.append(sum(month))
    
    month_trades=np.array(month_trades)
    print('mean of net/month is '+str(np.mean(month_trades)))
    print('stdev of net/month is '+str(np.std(month_trades)))
    print('lower bound of 95% confidence interval is '+str(np.mean(month_trades)-1.96*np.std(month_trades)))

    plt.hist(month_trades, bins=50)
    plt.show()
    