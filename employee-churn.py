# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:17:37 2020

@author: addoda
"""

#importing libraries
import pandas as pd
from pandas.plotting import scatter_matrix
from pandas import ExcelWriter
from pandas import ExcelFile
from openpyxl import load_workbook
import numpy as np
from scipy.stats import norm, skew
from scipy import stats
import statsmodels.api as sm

# importing libraries for data visualisations
import seaborn as sns
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib
color = sns.color_palette()
from IPython.display import display
pd.options.display.max_columns = None

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# from imblearn.over_sampling import SMOTE  # SMOTE
# sklearn modules for ML model selection
from sklearn.model_selection import train_test_split  # import 'train_test_split'
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Libraries for data modelling
from sklearn import svm, tree, linear_model, neighbors
from sklearn import naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# Common sklearn Model Helpers
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
# from sklearn.datasets import make_classification

# sklearn modules for performance metrics
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.metrics import auc, roc_auc_score, roc_curve, recall_score, log_loss
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, make_scorer
from sklearn.metrics import average_precision_score


# reading data 
hr=pd.read_csv('employee_churn.csv')
hr.columns



#checking for missing data
hr.isnull().sum()
#no missing values
hr.info() # 1470 data sample

# check distribution of features
hr.hist(figsize=(50,50))
plt.show()

#simple description
hr.describe()

# remove certain variables that are not important
hr.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours','EducationField'],
        axis="columns", inplace=True)
hr.head()

# identifying the data types for the purpose of labeling
hr.columns.to_series().groupby(hr.dtypes).groups


#know more about my categorical data
hr['Attrition'].value_counts()
hr['BusinessTravel'].value_counts()
hr['Department'].value_counts()
hr['Gender'].value_counts()
hr['MaritalStatus'].value_counts()
hr['OverTime'].value_counts()

#correlation matrix
plt.figure(figsize=(25, 25)
sns.heatmap(hr.corr(), annot=True, cmap="RdYlGn", annot_kws={"size":15})

# Calculate correlations
corr = hr.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
# Heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(corr,
            vmax=1,
            mask=mask,
             annot=True, fmt='.1f',
            linewidths=.2, cmap="YlGnBu")

#preprocessing

# Label Encoding will be used for columns with 2 or less unique values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Create a label encoder object
lbe = LabelEncoder()
lbe_count = 0
for col in hr.columns[1:]:
    if hr[col].dtype == 'object':
        if len(list(hr[col].unique())) <= 2:
            lbe.fit(hr[col])
            hr[col] = lbe.transform(hr[col])
            lbe_count += 1
print('{} columns were label encoded.'.format(le_count))

hr = pd.get_dummies(hr, drop_first=True)
hr.shape
hr.columns
hr.head


# identifying the data types for the purpose of labeling
hr.columns.to_series().groupby(hr.dtypes).groups

##--------------
X = hr.drop(['Attrition'], axis=1)
y= hr['Attrition']


#get col names
names=X.columns

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scale= sc.fit_transform(X)
X_new=pd.DataFrame(X_scale,columns=names)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.2, random_state = 0)


# Instantiate the classfiers and make a list
classifiers = [LogisticRegression(random_state=1234), 
               GaussianNB(),
               svm.SVC(),  
               RandomForestClassifier(random_state=1234),
               XGBClassifier()]

# Define a result table as a DataFrame
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

# Train the models and record the results
for cls in classifiers:
    model = cls.fit(X_train, y_train)
    yproba = model.predict(X_test)
    
    fpr, tpr, _ = roc_curve(y_test,  yproba)
    auc = roc_auc_score(y_test, yproba)
    
    result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)

# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)
#Plot the figure
fig = plt.figure(figsize=(8,6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.2f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.show()

#----------------------------------------------------
#                    SMOTE METHOD
#---------------------------------------------------
(np.random.seed(1234))
from imblearn.over_sampling import SMOTE
sm = SMOTE()
X_smote, y_smote = sm.fit_sample(X_new, y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size = 0.2, random_state = 0)


# Instantiate the classfiers and make a list
classifiers = [LogisticRegression(random_state=1234), 
               GaussianNB(),
               svm.SVC(),  
               RandomForestClassifier(random_state=1234),
               XGBClassifier()]

# Define a result table as a DataFrame
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

# Train the models and record the results
for cls in classifiers:
    model = cls.fit(X_train, y_train)
    yproba = model.predict(X_test)
    
    fpr, tpr, _ = roc_curve(y_test,  yproba)
    auc = roc_auc_score(y_test, yproba)
    
    result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)

# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)
#Plot the figure
fig = plt.figure(figsize=(8,6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.2f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.show()

#----------------------------------------------------
#                    ADASYN METHOD
#---------------------------------------------------

from imblearn.over_sampling import ADASYN 
sa = ADASYN()
X_ad, y_ad = sa.fit_sample(X_new, y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_ad, y_ad, test_size = 0.2, random_state = 0)


# Instantiate the classfiers and make a list
classifiers = [LogisticRegression(random_state=1234), 
               GaussianNB(),
               svm.SVC(),  
               RandomForestClassifier(random_state=1234),
               XGBClassifier()]

# Define a result table as a DataFrame
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

# Train the models and record the results
for cls in classifiers:
    model = cls.fit(X_train, y_train)
    yproba = model.predict(X_test)
    
    fpr, tpr, _ = roc_curve(y_test,  yproba)
    auc = roc_auc_score(y_test, yproba)
    
    result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)

# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)
#Plot the figure
fig = plt.figure(figsize=(8,6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.2f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.show()

#----------------------------------------------
#            RandomForest
#-----------------------------------------------
(np.random.seed(1234))
from sklearn.ensemble import RandomForestClassifier
class_ad=RandomForestClassifier()
class_ad.fit(X_train, y_train)
# Predicting the Test set results
# Predicting the Test set results
y_pred = class_ad.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

#precision
from sklearn.metrics import precision_score
precision=precision_score(y_test, y_pred)
print(precision)

#recall
from sklearn.metrics import recall_score
recall=recall_score(y_test, y_pred)
print(recall)

#f1-score
from sklearn.metrics import f1_score
f1=f1_score(y_test, y_pred)
print(f1)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc_score(y_test, y_pred)

importances = class_ad.feature_importances_
indices = np.argsort(importances)[::-1] # Sort feature importances in descending order
names = [X_train.columns[i] for i in indices] # Rearrange feature names so they match the sorted feature importances
plt.figure(figsize=(15, 7)) # Create plot
plt.title("Feature Importance") # Create plot title
plt.bar(range(X_train.shape[1]), importances[indices]) # Add bars
plt.xticks(range(X_train.shape[1]), names, rotation=90) # Add feature names as x-axis labels
plt.show() # Show plot