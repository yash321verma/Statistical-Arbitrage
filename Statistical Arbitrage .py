# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 18:22:30 2019

@author: Yash
"""

import pandas as pd
import numpy as np
#read the dataset
stocks=pd.read_csv("../input/nse_data.csv")
stocks.head()
stocks.shape
stocks.describe




#List of columns of the stocks
columnindices=list(stocks.columns)
print(columnindices)


import datetime
TIMESTAMP=stocks.iloc[:,columnindices.index('TIMESTAMP')].values
print(type(TIMESTAMP[0]))
dates=[datetime.datetime.strptime(date,'%d/%m/%Y') for date in TIMESTAMP]
stocks[['TIMESTAMP']]=dates
from sklearn.preprocessing import LabelEncoder
stocks[['TIMESTAMP']]=LabelEncoder.fit_transform(stocks,stocks[['TIMESTAMP']])


print("!!!ENTER ALL LETTERS IN CAPITAL!!!")
print(" Input a stock on which you want to check arbitrage opportunities and its Series")
stock=input("Enter the stock name---->") 
series=input("Enter the series---->")
   
print(stock)
print(series)
if(stocks.loc[stocks.SYMBOL==stock,:].empty):
     print('NO')   
else:
    
             
    if( stocks.loc[(stocks.SYMBOL==stock)&(stocks.SERIES==series)&(stocks.index<409164),'HIGH'].any()):

     #Labelling the SERIES 
     #Labelling the Symbols 
     from sklearn.preprocessing import LabelEncoder
     
     stocks=stocks.replace(np.nan,'', regex=True)
     seriesfirstindex=stocks[stocks.SERIES==series].first_valid_index()
     stocksfirstindex=stocks[stocks.SYMBOL==stock].first_valid_index()
    
     stocks[['SERIES']]=LabelEncoder.fit_transform(stocks,stocks[['SERIES']])
     stocks[['SYMBOL']]=LabelEncoder.fit_transform(stocks,stocks[['SYMBOL']])
     
     #.....Prepare the training set .....
     #length of 2016 dataset is 409164
     stockstrain=stocks.loc[stocks.index<409164,:]
     stockstrain=stockstrain.drop(columns=['OPEN','LAST','PREVCLOSE','TOTTRDQTY','TOTTRDVAL','TOTALTRADES','ISIN'])
    
    
     #highs and lows of 2016 dataset
     stockshigh16=stockstrain[['HIGH']]
     stockslow16=stockstrain[['LOW']]
   
    
    
     stockstrain=pd.concat([stockstrain,pd.DataFrame(columns=['PREV1','PREV2','PREV3','PREV4','PREV5','PREV6','PREV7','PREV8','PREV9','PREV10'])],sort=False)
     stockstrain[['PREV1','PREV2','PREV3','PREV4','PREV5','PREV6','PREV7','PREV8','PREV9','PREV10']]=0 
     stockstrain.index=range(0,len(stockstrain))
     stockstrain[['TIMESTAMP']]=(pd.to_numeric(stockstrain["TIMESTAMP"]))
     stocks_list= stockstrain.SYMBOL.unique()
     print(stocks_list)
     stocks_list=list(stocks_list)
     num_stocks=len(stockstrain.SYMBOL.unique())
     stockstrainhigh=pd.DataFrame(columns=['SYMBOL','SERIES','TIMESTAMP','HIGH','PREV1','PREV2','PREV3','PREV4','PREV5','PREV6','PREV7','PREV8','PREV9','PREV10'])
     stockstrainlow=pd.DataFrame(columns=['SYMBOL','SERIES','TIMESTAMP','LOW','PREV1','PREV2','PREV3','PREV4','PREV5','PREV6','PREV7','PREV8','PREV9','PREV10'])
    
    
     #Fill in the previous 10 days value of the stock
     i=0
     for i in range(0,num_stocks):
         
         high=stockstrain.loc[stockstrain.SYMBOL==stocks_list.__getitem__(i),['HIGH']]
         low=stockstrain.loc[stockstrain.SYMBOL==stocks_list.__getitem__(i),['LOW']]
         l=len(high)
         temp_data_high=pd.DataFrame(index=range(0,l),columns=['SYMBOL','SERIES','TIMESTAMP','HIGH','PREV1','PREV2','PREV3','PREV4','PREV5','PREV6','PREV7','PREV8','PREV9','PREV10'])
         temp_data_low=pd.DataFrame(index=range(0,l),columns=['SYMBOL','SERIES','TIMESTAMP','LOW','PREV1','PREV2','PREV3','PREV4','PREV5','PREV6','PREV7','PREV8','PREV9','PREV10'])
   
         temp_data_high[['TIMESTAMP','HIGH','PREV1','PREV2','PREV3','PREV4','PREV5','PREV6','PREV7','PREV8','PREV9','PREV10']]=high.iloc[0:10,0].values.mean()
         
         temp_data_low[['TIMESTAMP','LOW','PREV1','PREV2','PREV3','PREV4','PREV5','PREV6','PREV7','PREV8','PREV9','PREV10']]=low.iloc[0:10,0].values.mean()
         
         tempstockhigh=stockstrain.loc[stockstrain.SYMBOL==stocks_list.__getitem__(i),['SYMBOL','SERIES','TIMESTAMP','HIGH']].values
         tempstockhigh =pd.DataFrame(tempstockhigh)
         tempstockdatalow=stockstrain.loc[stockstrain.SYMBOL==stocks_list.__getitem__(i),['SYMBOL','SERIES','TIMESTAMP','LOW']].values
         tempstockdatalow=pd.DataFrame(tempstockdatalow)
         
         temp_data_high[['SYMBOL','SERIES','TIMESTAMP','HIGH']]=tempstockhigh.iloc[:,0:].values
         temp_data_low[['SYMBOL','SERIES','TIMESTAMP','LOW']]=tempstockdatalow.iloc[:,0:].values
         temp_data_high.loc[1:,['PREV1']]=high.iloc[0:-1,0:1].values
         temp_data_low.loc[1:,['PREV1']]=low.iloc[0:-1,0:1].values
         
         temp_data_high.loc[2:,['PREV2']]=high.iloc[0:-2,0:1].values
         temp_data_low.loc[2:,['PREV2']]=low.iloc[0:-2,0:1].values
         temp_data_high.loc[3:,['PREV3']]=high.iloc[0:-3,0:1].values
         temp_data_low.loc[3:,['PREV3']]=low.iloc[0:-3,0:1].values
         temp_data_high.loc[4:,['PREV4']]=high.iloc[0:-4,0:1].values
         temp_data_low.loc[4:,['PREV4']]=low.iloc[0:-4,0:1].values 
         temp_data_high.loc[5:,['PREV5']]=high.iloc[0:-5,0:1].values
         temp_data_low.loc[5:,['PREV5']]=low.iloc[0:-5,0:1].values 
         temp_data_high.loc[6:,['PREV6']]=high.iloc[0:-6,0:1].values
         temp_data_low.loc[6:,['PREV6']]=low.iloc[0:-6,0:1].values 
         temp_data_high.loc[7:,['PREV7']]=high.iloc[0:-7,0:1].values
         temp_data_low.loc[7:,['PREV7']]=low.iloc[0:-7,0:1].values 
         temp_data_high.loc[8:,['PREV8']]=high.iloc[0:-8,0:1].values
         temp_data_low.loc[8:,['PREV8']]=low.iloc[0:-8,0:1].values 
         temp_data_high.loc[9:,['PREV9']]=high.iloc[0:-9,0:1].values
         temp_data_low.loc[9:,['PREV9']]=low.iloc[0:-9,0:1].values 
         temp_data_high.loc[10:,['PREV10']]=high.iloc[0:-10,0:1].values
         temp_data_low.loc[10:,['PREV10']]=low.iloc[0:-10,0:1].values 
         stockstrainhigh=pd.concat([stockstrainhigh,temp_data_high])
         stockstrainlow=pd.concat([stockstrainlow,temp_data_low])
         
     print(stockstrainhigh)
     print(stockstrainlow)
     #shuffle the training dataset to make parameters tune more accurate
     from sklearn.utils import shuffle
     stockstrainhigh=shuffle(stockstrainhigh)
     stockstrainlow=shuffle(stockstrainlow)
     
     seriesnumber=np.array(stocks.loc[0,'SERIES'])
     print(seriesnumber)
     stocknumber=np.array(stocks.loc[0,'SYMBOL'])
     
     stocks= stocks.loc[(stocks.SYMBOL==stocknumber)&(stocks.index<846406),:]

   
     stockdata2017=stocks.loc[(stocks.SYMBOL==stocknumber)&(stocks.SERIES==seriesnumber)&(stocks.index>=409164)&(stocks.index<846406),:]
     stockdata2017=stockdata2017.drop(columns=['OPEN','LAST','PREVCLOSE','TOTTRDQTY','TOTTRDVAL','TOTALTRADES','ISIN'])
     
     stockdata2017[['SYMBOL']]=stocknumber
     stockdata2017[['SERIES']]=seriesnumber
     
     
     from sklearn.preprocessing import LabelEncoder
     stockdata2017[['TIMESTAMP']]=LabelEncoder.fit_transform(stockdata2017,stockdata2017[['TIMESTAMP']])
      
   
    
    
     stockdata2016=stocks.loc[(stocks.SYMBOL==stocknumber)&(stocks.index<409164),:]
    
     print(stockdata2016)
     #change index according to stock size
     
     stockdata2016.index = range(0,len(stockdata2016.index)) 
     x=float('nan')
    
     if(stockdata2016.loc[stockdata2016.HIGH==x,'HIGH'].any()):
        stockdata2016.loc[stockdata2016.HIGH==x,'HIGH']=np.mean(stockdata2016[['HIGH']])
        print('Yes')
    
      
     if(stockdata2016.loc[stockdata2016.LOW==x,'LOW'].any()):
        stockdata2016.loc[stockdata2016.LOW==x,'LOW']=np.mean(stockdata2016[['LOW']])
      
     high2016=stockdata2016.loc[:,['HIGH']]
     
     low2016=stockdata2016.loc[:,['LOW']]
     
     print(high2016.shape)
     print(low2016)
  
    
     stockdata2016=stockdata2016.drop(columns=['HIGH','LOW','OPEN','CLOSE','LAST','PREVCLOSE','TOTTRDQTY','TOTTRDVAL','TOTALTRADES','ISIN'])
    
     stockhighdata2016=pd.concat([stockdata2016,pd.DataFrame(columns=['PREV1','PREV2','PREV3','PREV4','PREV5','PREV6','PREV7','PREV8','PREV9','PREV10','INDICES'])],sort=False)
     stockhighdata2016[['PREV1','PREV2','PREV3','PREV4','PREV5','PREV6','PREV7','PREV8','PREV9','PREV10','INDICES']]=0
     stockhighdata2016list=stockhighdata2016.columns.values.tolist()
     print(stockhighdata2016list)
 
     stocklowdata2016=pd.concat([stockdata2016,pd.DataFrame(columns=['PREV1','PREV2','PREV3','PREV4','PREV5','PREV6','PREV7','PREV8','PREV9','PREV10','INDICES'])],sort=False)
     stocklowdata2016[['PREV1','PREV2','PREV3','PREV4','PREV5','PREV6','PREV7','PREV8','PREV9','PREV10','INDICES']]=0 


     
     column2016=list(stocklowdata2016.columns)
     
     for i in range(0,len(stockhighdata2016)):
        
        if(i>0):
                  
               for j in range(0,10):
                 if((i-j)>0):
                  stockhighdata2016.loc[i,column2016.__getitem__(j+2)]=high2016.iloc[i-j-1,0]
                  stocklowdata2016.loc[i,column2016.__getitem__(j+2)]=low2016.iloc[i-j-1,0]
    
 
    
 #Putting the dataset to RandomFORestregressor
     from sklearn.ensemble import RandomForestRegressor
 #check for accuracy using different no. of trees  
 #checking with non random data for high
     stockhighdata2016=stockhighdata2016.drop(columns=['INDICES'])
     stocklowdata2016=stocklowdata2016.drop(columns=['INDICES'])
     print(stockhighdata2016)
     print(high2016)
     
     
     #Using random Forest model with trees =125 ,bootstrap....
     modelhigh=RandomForestRegressor(n_estimators=125,bootstrap.=True, criterion="mse",max_features="auto",random_state=0 ,min_samples_leaf=1)
     modellow=RandomForestRegressor(n_estimators=125,bootstrap=True, criterion="mse",max_features="auto",random_state=0 ,min_samples_leaf=1)
     
     hightrain=stockstrainhigh[['HIGH']]
     lowtrain=stockstrainlow[['LOW']]
     print(hightrain)
     import datetime
     print("Start Time")
     print(datetime.datetime.now())
     stockstrainhigh= stockstrainhigh.drop(columns=['HIGH'])
     stockstrainlow=stockstrainlow.drop(columns=['LOW'])
     modelhigh=modelhigh.fit(stockstrainhigh,hightrain)
     modellow=modellow.fit(stockstrainlow,lowtrain)
     datetime.datetime.now()

     print(modelhigh)
     print("End Time")
     print(datetime.datetime.now())
     
     #............Preparing the 2017 dataset.....
     high2017=stockdata2017.loc[stockdata2017.SERIES==seriesnumber,['HIGH']]
     stockdata2017.index=range(0,len(stockdata2017))
     low2017=stockdata2017.loc[stockdata2017.SERIES==seriesnumber,['LOW']]

     stockdata2017=stockdata2017.drop(columns=['HIGH','LOW','CLOSE'])
     stockdata2017.index = range(0,len(stockdata2017.index)) 
     stockhighdata2017=pd.concat([stockdata2017,pd.DataFrame(columns=['PREV1','PREV2','PREV3','PREV4','PREV5','PREV6','PREV7','PREV8','PREV9','PREV10'])],sort=False)
     
    
     stocklowdata2017=pd.concat([stockdata2017,pd.DataFrame(columns=['PREV1','PREV2','PREV3','PREV4','PREV5','PREV6','PREV7','PREV8','PREV9','PREV10'])],sort=False)
     stockhighdata2017[['PREV1','PREV2','PREV3','PREV4','PREV5','PREV6','PREV7','PREV8','PREV9','PREV10']]=0
     stocklowdata2017[['PREV1','PREV2','PREV3','PREV4','PREV5','PREV6','PREV7','PREV8','PREV9','PREV10']]=0
     column2017=list(stockhighdata2017.columns)
     
   
     high2017.index=range(0,len(stockdata2017))
     low2017.index=range(0,len(stockdata2017))

     for i in range(0,len(stockhighdata2017)):
        
        if(i>0):
                  
               for j in range(0,10):
                 if((i-j)>0):
                  stockhighdata2017.loc[i,column2017.__getitem__(j+2)]=high2017.iloc[i-j-1,0]
                  stocklowdata2017.loc[i,column2017.__getitem__(j+2)]=low2017.iloc[i-j-1,0]
     
            
     predictionshigh=modelhigh.predict(stockhighdata2017)
     predictionslow=modellow.predict(stocklowdata2017)
    
     errorhigh=abs(predictionshigh-np.array(high2017))
     mape=100*(np.mean(errorhigh/np.array(high2017)))
     accuracyhigh=100-mape
     print("......Accuracy of modelhigh..... ")
     print(accuracyhigh)
     
     errorlow=abs(predictionslow-np.array(low2017))
     mape=100*(np.mean(errorlow/np.array(low2017)))
     accuracylow=100-mape
     print("......Accuracy of modellow..... ")
     print(accuracylow)
     
    #length of a financial year 2017
     fin_year_len=len(high2017)
 



def prof_calc(high2017,low2017,highdata,lowdata,fin_year_len):
       max_high=np.zeros(shape=(1,))
       min_low=np.zeros(shape=(1,))
       timestamp_high=np.zeros(shape=(1,))
       timestamp_low=np.zeros(shape=(1,))
       second_high=np.zeros(shape=(1,))
       secondtimestamp_high=np.zeros(shape=(1,))
       profit=np.zeros(shape=(1,))
       print('..................333333333333........................')
       i=fin_year_len-2
       while(i>-1):
           print("111111111111")
           temp_high=highdata[0,i+1]
           temp_low=lowdata[0,i+1]
           print(".........TEMP LOW.............",temp_low)
           print(".........TEMP HIGH.............",temp_high)
           if(i==fin_year_len-2):
                timestamp_high=i+1
                timestamp_low=i
                max_high=highdata[0,i+1]
                min_low=lowdata[0,i]
                profit=max_high-min_low
           else:
              #if profitnew>profit
                if((temp_high-temp_low)>profit):
             #if this then max_high and min_low will change since,the previous bigger high is at a later date        
                    if(temp_high>max_high):
                    
                      max_high=temp_high
                      min_low=temp_low
                      profit=max_high-min_low
                      timestamp_high=i+1
                      timestamp_low=i
             #if only this case then temp_low will be then temp_low being at a lower date than previous low(min_low) will
             #maximiZe th eprofit
                    if(temp_low<min_low):
                     
                      min_low=temp_low
                      profit=max_high-min_low
                      timestamp_low=i
             #if both temphigh >  max_high an d temp_low<min_low    
                    if((temp_low<min_low) & (temp_high>max_high)):
                     
                      min_low=temp_low
                      max_high=temp_high
                      timestamp_high=i+1
                      timestamp_low=i
                      profit=max_high-min_low
 ##################################################################################################################      
                     
                if((temp_high-temp_low)==profit):
                           if(temp_high>max_high):
                                 max_high=temp_high
                                 min_low=temp_low
                                 timestamp_high=i+1
                                 timestamp_low=i
                     
                           if(temp_low<min_low):
                                 min_low=temp_low
                                 timestamp_low=i
                                 profit=max_high-min_low
#####################################################################################################################
                if((temp_high-temp_low)<profit):
                    if(temp_high>max_high):
                       if(temp_high>second_high): 
                         second_high=temp_high
                         secondtimestamp_high=i+1
                     
                    if((temp_low<min_low) & (second_high==0)):
                        min_low=temp_low
                        timestamp_low=i
                        profit=max_high-min_low
                     
                    if((temp_low<min_low) & (second_high!=0)):
                        max_high=second_high
                        min_low=temp_low
                        timestamp_high=secondtimestamp_high
                        timestamp_low=i
                        profit=max_high-min_low
           i=i-1         
            

       actual_high=high2017.iloc[timestamp_high,0]
       actual_low=low2017.iloc[timestamp_low,0]
    
       return max_high,min_low,actual_high,actual_low

max_high,min_low,actual_high,actual_low=prof_calc(high2017,low2017,predictionshigh,predictionslow,fin_year_len)


print("............Results of Statistical Arbitrage.........")
print(max_high)
print(min_low)
print(actual_high)
print(actual_low)

print("......Model Summary.......")
from sklearn.metrics import confusion_matrix
#confusion matrix for highpredictions
confusion_matrix(high2017, predictionshigh)

#confusion matrix for low predictions
confusion_matrix(low2017,predictionslow)

