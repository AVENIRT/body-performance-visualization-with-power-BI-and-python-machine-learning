#!/usr/bin/env python
# coding: utf-8

# In[175]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


## For preprocessing 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import tree


# In[ ]:





# In[ ]:





# In[102]:


#Perform exploratory analysis.
#load data to a panda data frame

df = pd.read_csv('bodyPerformance[1].csv')


# In[103]:


#let's get deeper into data and get more infos

df.info()


# In[104]:


#perform data analysis
df.head()


# In[105]:


# The number of rows and columns

rows,columns =df.shape
print('Number of rows --> ', rows)
print('Number of columns --> ', columns)


# In[106]:


df.describe()


# In[ ]:





# In[ ]:





# In[42]:


sns.countplot(data = df , x = 'gender')


# In[80]:


sns.countplot(data = df , x = 'class')


# In[29]:


#analyse data distribution of height

sns.set()
plt.figure(figsize=(10, 5))
plt.title('height Distribution')
sns.distplot(df['height_cm'])
plt.show()


# In[31]:


# analyse data distribution of age

sns.set()
plt.figure(figsize=(10, 5))
plt.title('Age Distribution')
sns.distplot(df['age'])
plt.show()


# In[32]:


# analyse data distribution of 
sns.set()
plt.figure(figsize=(10, 5))
plt.title('weight Distribution')
sns.distplot(df['weight_kg'])
plt.show()


# In[33]:


# analyse data dsitribution of body fat

sns.set()
plt.figure(figsize=(10, 5))
plt.title('body fat Distribution')
sns.distplot(df['body fat_%'])
plt.show()


# In[34]:


# analyse data distribution of systolic

sns.set()
plt.figure(figsize=(10, 5))
plt.title('systolic Distribution')
sns.distplot(df['systolic'])
plt.show()


# In[35]:


# analyse data distribution of gripForce

sns.set()
plt.figure(figsize=(10, 5))
plt.title('gripForce Distribution')
sns.distplot(df['gripForce'])
plt.show()


# In[40]:


# analyse data distribution of sit and bend forward

sns.set()
plt.figure(figsize=(10, 5))
plt.title('sit and bend forward Distribution')
sns.distplot(df['sit and bend forward_cm'])
plt.show()


# In[78]:


# analyse data distribution of sits_up count

sns.set()
plt.figure(figsize=(10, 5))
plt.title('sits-up count')
sns.distplot(df['sit-ups counts'])
plt.show()


# In[47]:


# analyse data distribution of broad jump cm

sns.set()
plt.figure(figsize=(10, 5))
plt.title('broad of jump cm')
sns.distplot(df['broad jump_cm'])
plt.show()


# In[82]:


# lets uses the scatterplot to show relation between 2 variables
sns.scatterplot(data = df , x = 'class' , y = 'body fat_%' ,hue = 'gender')


# In[83]:


sns.scatterplot(data = df , x = 'class' , y = 'weight_kg' ,hue = 'gender')


# In[84]:


sns.scatterplot(data = df , x = 'weight_kg' , y = 'body fat_%',hue ='gender')


# In[86]:


sns.scatterplot(data = df , x = 'weight_kg' , y = 'height_cm',hue ='gender',alpha = 0.8)


# In[114]:


#I want to use heatmap to built the correlation matrix

plt.figure(figsize=(8,8))
sns.heatmap(df.corr(),annot=True)


# #                     DATA PREPROCESSING
# 

# In[112]:


df.replace("M", 0 , inplace = True)
df.replace("F", 1 , inplace = True)


# In[113]:


df.replace("A", 1 , inplace = True)
df.replace("B", 2 , inplace = True)
df.replace("C", 3 , inplace = True)
df.replace("D", 4 , inplace = True)


# In[114]:


df.head()


#                                            TRAIN DATA

# In[115]:


X=df.drop('class',axis=1).values
y=df['class'].values


# In[116]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)


# In[117]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[119]:


scale = StandardScaler ()
X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)


#                              ANALYZE DATA WITH LOGISTIC REGRESSION

# In[120]:


Lmodel=LogisticRegression()


# In[121]:


Lmodel.fit(X_train,y_train)                              


# In[122]:


Lmodel_score_train = Lmodel.score(X_train,y_train)


# In[124]:


Lmodel_score_train 


# In[125]:


Lmodel_score_test = Lmodel.score(X_test,y_test)


# In[156]:


Lmodel_score_test


# In[160]:


# evaluate the precision of our model

y_predict_Lmodel = Lmodel.fit(X_train, y_train).predict(X_test) 


# In[161]:


classification_report(y_test,predictions)


# In[197]:


Lmodel_cm = pd.DataFrame(confusion_matrix(y_test, y_predict_Lmodel))
sns.heatmap(Lmodel_cm, annot=True,cmap="mako_r")


# In[162]:


#check the confusion matrix
confusion_matrix(y_test,predictions)


# In[129]:


# let check the accuracy of our prediction
accuracy_score(y_test,predictions)


#                                      NAIVE BAYES 

# In[201]:


model =GaussianNB()


# In[202]:


model.fit(X_train,y_train)


# In[203]:


model_score_train = model.score(X_train,y_train)


# In[204]:


model_score_train


# In[205]:


model_score_test = model.score(X_test,y_test)


# In[207]:


model_score_test


# In[208]:


#EVALUATE SCORE PREDICTION
y_predict_model = model.fit(X_train, y_train).predict(X_test)


# In[209]:


classification_report(y_test,predictions)


# In[ ]:





# In[210]:


model_cm = pd.DataFrame(confusion_matrix(y_test, y_predict_Lmodel))
sns.heatmap(model_cm, annot=True,cmap="mako_r")


#              decision trees

# In[217]:


Tmodel =tree.DecisionTreeClassifier()


# In[218]:


Tmodel.fit(X_train,y_train)  


# In[219]:


Tmodel_score_train = Tmodel.score(X_train,y_train)


# In[220]:


Tmodel_score_train 


# In[221]:


Tmodel_score_test = Tmodel.score(X_test,y_test)


# In[222]:


Tmodel_score_test


# In[223]:


y_predict_Tmodel = Tmodel.fit(X_train, y_train).predict(X_test) 


# In[224]:


classification_report(y_test,predictions)


# In[225]:


Tmodel_cm = pd.DataFrame(confusion_matrix(y_test, y_predict_Lmodel))
sns.heatmap(Tmodel_cm, annot=True,cmap="mako_r")


# In[194]:


predictors_group = ('Logistic Regression','Naives bayes','Decision tree')
x_pos = np.arange(len(predictors_group))
accuracies = [Lmodel_score_test, model_score_test, Tmodel_score_test]
    
plt.bar(x_pos, accuracies, align='center', color='blue')
plt.xticks(x_pos, predictors_group, rotation='vertical')
plt.ylabel('Accuracy (%)',)
plt.title(' Accuracies')
plt.show()

