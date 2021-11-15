#!/usr/bin/env python
# coding: utf-8

#    # PREDICTION USING SUPERVISED LEARNING
# 

# NAME: VARUN GAZALA
# DOMAIN: DATA SCIENCE AND BUSINESS ANALYTICS
# LANGUAGE:PYTHON
# PLATFORM: JUPYTER NOTEBOOK  
# Dataset Link:http://bit.ly/w-data

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as lr
from sklearn.cluster import KMeans


# In[4]:


url='http://bit.ly/w-data'
data=pd.read_csv(url)


# In[5]:


data.head(10)


# In[6]:


data.info()


# In[7]:


data.isnull().sum()


# In[8]:


data.describe()


# In[9]:


#PLOTTING THE DATASET


# In[10]:


data.plot(x='Hours',y='Scores',style='go')
plt.title('Prediction')
plt.xlabel('Hours_Studied')
plt.ylabel('Test_score')
plt.show()


# In[11]:


sns.boxplot(data=data[['Hours','Scores']])


# In[12]:


X=data.iloc[:,:-1].values
Y=data.iloc[:,1].values


# In[13]:


X


# In[14]:


Y


# In[15]:


#TRAINING SET AND TEST SET AND ALGO


# In[16]:


X_train,x_test,Y_train,y_test=tts(X,Y,test_size=0.20,random_state=0)


# In[17]:


Reg=lr()    


# In[18]:


Reg.fit(X_train,Y_train)


# In[41]:


LR=Reg.coef_*X+Reg.intercept_
data.plot.scatter(x='Hours',y='Scores')
plt.plot(X,LR)
plt.grid()
plt.show()


# In[29]:


y_prediction=Reg.predict(x_test)
print(y_prediction)


# In[30]:


#PREDICTION FOR STUDENT STUDYING 9.25 HRS/DAY


# In[42]:


comp=pd.DataFrame({'Actual':[y_test],'Predicted':[y_prediction]})
print(comp)


# In[54]:


Reg=lr()  
Reg.fit(X_train,Y_train)
hours=9.25
own_pred=Reg.predict([[hours]])
print("the predicted score if a person studies for",hours,"hours is",own_pred[0])


# In[35]:


#EVALUATING THE MODEL


# In[37]:


print("mean absolute error=",metrics.mean_absolute_error(y_test,y_prediction))


# In[ ]:




