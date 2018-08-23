
# coding: utf-8

# In[1]:


import sys
import numpy
import pandas
import matplotlib
import seaborn
import scipy
import sklearn

print('Python : {}'.format(sys.version))
print('Numpy : {}'.format(numpy.__version__))
print('Pandas : {}'.format(pandas.__version__))
print('Matplotlib : {}'.format(matplotlib.__version__))
print('Seaborn : {}'.format(seaborn.__version__))
print('Scipy : {}'.format(scipy.__version__))
print('Sklearn : {}'.format(sklearn.__version__))


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


data = pd.read_csv('datasets/creditcard.csv')
print(data.columns)


# In[5]:


print(data.shape)


# In[6]:


print(data.describe())


# In[7]:


# sample data set of small size
data = data.sample(frac = 0.1,random_state = 1)
print(data.shape)


# In[8]:


#plotting histogram for each catagory
data.hist(figsize = (20,20))
plt.show()


# In[10]:


# Determine number of fraud cases in dataset

Fraud = data[data['Class']==1]
Valid = data[data['Class']==0]

# Fraction of fraud cases
outlier_fraction = len(Fraud)/float(len(Valid))
print(outlier_fraction)
print('Fraud Cases : {}'.format(len(Fraud)))
print('Valid Cases : {}'.format(len(Valid)))

# precentage of Fraud cases
print(outlier_fraction*100)


# In[11]:


# Correlation Matrix for the data set
corr_matrix = data.corr()
fig = plt.figure(figsize = (12,12))

sns.heatmap(corr_matrix,vmax = 0.8,square = True)

plt.show()


# In[14]:


# get all the cloumns from the DataFrame

columns = data.columns.tolist()

#Filter the columns to remove data we do not want

columns = [c for c in columns if c not in ["Class"]]

# Store the variable we will be predicting on

target = "Class"

X = data[columns]
Y = data[target]

#Print the shape of X and Y

print(X.shape)
print(Y.shape)


# In[17]:


# algorithms for the project

from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

#define a random state

state = 1

# defien the outlier detection methods

classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(X),
                                       contamination = outlier_fraction,
                                       random_state = state),
    "Local Outlier Factor": LocalOutlierFactor(
    n_neighbors = 20,
    contamination = outlier_fraction)
}


# In[19]:


# Fit the model

n_outlier = len(Fraud)

for i,(clf_name, clf) in enumerate(classifiers.items()):
    
    #fit the data and tag outlier
    
    if clf_name =="Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
    
    # Reshape the prediction values to 0 fro valid and 1 for fraud
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    n_errors = (y_pred !=Y).sum()
    
    # Run classification metrics
    
    print('{} : {}'.format(clf_name,n_errors))
    print(accuracy_score(Y,y_pred))
    print(classification_report(Y,y_pred))
    

