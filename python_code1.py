#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib as mtb


# In[2]:


os.chdir(r"C:/Users/user/Documents")
os.getcwd()


# In[38]:


df=pd.read_csv(r"C:/Users/user/Downloads/DataScience2/train_cab.csv")
df1=pd.read_csv(r"C:/Users/user/Downloads/DataScience2/test.csv")


# In[51]:


df.iloc[1123,0]=16.9


# In[8]:


df1.columns


# In[9]:


type(df)
type(df1)


# In[52]:


miss_val=pd.DataFrame(df.isnull().sum())
miss_val


# In[73]:


df['pickup_datetime'] = pd.to_numeric(df['pickup_datetime'])


# In[70]:


df=df.fillna(df.mean())


# In[72]:


df.iloc[8,0]


# In[19]:


import seaborn as sns
corr = df.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


# In[111]:


df2=df[['fare_amount','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count']]


# In[84]:


import statsmodels.api as sm


# In[105]:


model=sm.OLS(df2.iloc[:,0],df2.iloc[:,1:]).fit()


# In[106]:


model


# In[97]:


df2


# In[103]:


df2 = df2.apply(pd.to_numeric)


# In[102]:


df2


# In[107]:


model.summary()


# In[113]:


pred=model.predict(df1.iloc[:,1:])


# In[114]:


pred


# In[119]:


def MAPE(y_true, y_pred): 
    return np.mean(np.abs((y_true-y_pred)/y_true)) 
MAPE(df2.iloc[:,0],pred)


# In[ ]:

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
from sklearn.ensemble import RandomForestClassifier
RF_model=RandomForestClassifier(n_estimate=500).fit(x_train,y_train)
RF_model
RF_pred=RF_model.predict(x_test)
CM=pd.crosstab(y_test,RF_pred)
TN=CM.iloc[0,0]
TP=CM.iloc[1,1]
FP=CM.iloc[0,1]
FN=CM.iloc[1,0]
((TP+TN*100)/(TP+TN+FN+FP))
(FN*100)/(FN+TP)



from sklearn.neighbors import KNeighborsClassifier
x1=df.values[0:,1:]
y1=df.values[0:,0]
from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.20)
KNN_Model=KNeighborsClassifier(n_neighbors=1)
KNN_Model.fit(X1_train,y1_train)
KNN_Model


# In[ ]:


KNN_pred=KNN_Model.predict(X1_test)
KNN_pred


# In[ ]:


CM=pd.crosstab(y1_test,KNN_pred)
CM


# In[ ]:


TN=CM.iloc[0,0]
FN=CM.iloc[1,0]
TP=CM.iloc[1,1]
FP=CM.iloc[0,1]
Accuracy=((TP+TN)*100)/(TP+TN+FN+FP)
FNR=(FN*100)/(FN+TP)
Recall=(TP*100)/(FN+TP)
Accuracy


# In[ ]:


FNR


# In[ ]:


Recall


df1.insert(0,'target',1)
df1['target']=KNN_pred
