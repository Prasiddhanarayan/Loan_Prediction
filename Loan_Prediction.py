#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv('C:/Users/hp/Desktop/train.csv')
test = pd.read_csv('C:/Users/hp/Desktop/test.csv')


# In[6]:


print(test.shape)
print(train.shape)


# In[7]:


train.head()


# In[8]:


test.head()


# In[10]:


test.info()
train.info()


# In[14]:


print(train.describe())
print(test.describe())


# In[16]:


train.drop('Loan_ID',axis=1)
test.drop('Loan_ID',axis=1)


# In[21]:


catcol=train.select_dtypes(include='object').columns
numcol=train.select_dtypes(exclude='object').columns


# In[22]:


catcol


# In[25]:


catcol=catcol.delete(0)


# In[26]:


catcol


# In[40]:


for i in catcol:
    print(i)
    print(train[i].value_counts(normalize=True))
    print('\n')


# In[39]:


numcol


# In[41]:


sns.distplot(train['ApplicantIncome'])


# In[43]:


sns.boxplot(y=train['ApplicantIncome'])


# In[44]:


sns.distplot(train['CoapplicantIncome'])


# In[46]:


sns.boxplot(y=train['CoapplicantIncome'])


# In[48]:


sns.distplot(train['LoanAmount'].dropna())


# In[50]:


sns.boxplot(y=train['LoanAmount'].dropna())


# In[52]:


sns.distplot(train['Loan_Amount_Term'].dropna())


# In[54]:


sns.boxplot(y=train['Loan_Amount_Term'].dropna())


# In[55]:


plt.scatter('ApplicantIncome','LoanAmount',data=train)


# In[56]:


plt.scatter('ApplicantIncome','Loan_Amount_Term',data=train)


# In[62]:


sns.heatmap(train[numcol].drop(['Credit_History'], axis=1).corr(), annot=True)


# In[65]:


train.groupby('Credit_History').mean()


# In[67]:


train.columns


# In[98]:


y_train= train['Loan_Status']
train.drop('Loan_Status', axis=1, inplace=True)


# In[ ]:





# In[69]:


train.head()


# In[130]:


train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
test['Gender'].fillna(test['Gender'].mode()[0], inplace=True)
test['Married'].fillna(test['Married'].mode()[0], inplace=True)
test['Dependents'].fillna(test['Dependents'].mode()[0], inplace=True)
test['Credit_History'].fillna(test['Credit_History'].mode()[0], inplace=True)
test['Self_Employed'].fillna(test['Self_Employed'].mode()[0], inplace=True)


# In[131]:


test.isnull().sum()


# In[129]:


train.isnull().sum()


# In[132]:


train['LoanAmount'] = train.groupby(['Education', 'Self_Employed'])['LoanAmount'].apply(lambda x: x.fillna(x.median()))
test['LoanAmount'] = test.groupby(['Education', 'Self_Employed'])['LoanAmount'].apply(lambda x: x.fillna(x.median()))


# In[133]:


train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mode()[0], inplace=True)


# In[134]:


test.isnull().sum()


# In[135]:


train.isnull().sum()


# In[136]:


train.apply(lambda x: len(x.unique()))


# In[137]:


train=train.drop('Loan_ID',axis=1)


# In[138]:


train


# In[139]:


X=train


# In[140]:


Y=y_train


# In[141]:


X


# In[142]:


Y


# In[143]:


sns.pairplot(X)


# In[144]:


X=pd.get_dummies(X)


# In[145]:


X.head()


# In[147]:


Y


# In[148]:


train.head()


# In[ ]:





# In[149]:


X['TotalIncome']=X['ApplicantIncome']+X['CoapplicantIncome']
X['Loan/Income']=X['LoanAmount']/X['TotalIncome']
X['Loan/Term']=X['LoanAmount']/X['Loan_Amount_Term']
X['RepaymentRatio']=(X['Loan/Term']*1000)/X['TotalIncome']


# In[157]:


X=X.drop('LoanAmount', axis=1)


# In[158]:


X


# In[167]:


from sklearn.preprocessing import Imputer, StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score,GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


# In[168]:


cv=StratifiedKFold(n_splits=5,random_state=5)


# In[170]:


models = []
models.append(('CART', DecisionTreeClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('Naive Bayes', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('Random Forest', RandomForestClassifier()))
models.append(('Bagging', BaggingClassifier()))
models.append(('AdaBoost', AdaBoostClassifier()))
models.append(('Gradient Boosting', GradientBoostingClassifier()))
models.append(('Logistic Regression', LogisticRegression()))
models.append(('MLP', MLPClassifier ( max_iter=1000)))

results = []
names = []
tabel=[]
for name, model in models:
    akurasi=cross_val_score(model, X,Y,cv=cv)
    results.append(akurasi)
    names.append(name)
    hasil = "%s: %f" % (name, akurasi.mean())
    tabel.append(hasil)
tabel


# In[ ]:


# to Add : Feature Engineering


# In[ ]:




