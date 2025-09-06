#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 11:41:06 2023

@author: erangad
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import datetime
from sklearn.svm import SVC # "Support vector classifier"
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

dataset = pd.read_csv("dataset_preprocessed.csv")

X = dataset.iloc[:,1:]
y = dataset['is_not_survivor']


from imblearn.over_sampling import SMOTE
os = SMOTE()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

os_data_X,os_data_y=os.fit_resample(X_train, y_train)



# ##  Model train and optimising 
from sklearn import metrics


model = SVC(kernel='linear')

model.fit(os_data_X, os_data_y)

y_pred = model.predict(X_test)
print('Accuracy of logistic regression classifier on test set before optimising : {:.2f}'.format(model.score(X_test, y_test)))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

precision_score = metrics.precision_score(y_test, y_pred)
print("Sensitivity is :" + str(precision_score))


#ROC and AUC
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)

#create ROC curve
plt.figure(figsize=(10,8))
plt.plot(fpr,tpr,label="Logistic Regression AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

