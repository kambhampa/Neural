#!/usr/bin/env python
# coding: utf-8

# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import pandas as pd




# In[13]:


df=pd.read_csv('C:/Users/saibh/Downloads/glass.csv')


# In[14]:


df


# In[15]:


X = df.iloc[:,:-1]  
y = df.iloc[:, -1]


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# Training the model on training set
model = GaussianNB()  
model.fit(X_train, y_train)

# Making predictions on test set
y_pred = model.predict(X_test) 

# Accuracy of the model
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Score:", accuracy)

# Classification report of the model
print("Classification Report:\n", metrics.classification_report(y_test, y_pred))


# In[ ]:




