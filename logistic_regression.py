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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

dataset = pd.read_csv("dataset_preprocessed.csv")

dataset = dataset.rename(columns={'Oxygen saturation in Arterial blood': 'Oxygen saturation'}  )
dataset = dataset.rename(columns={'Pain severity - 0-10 verbal numeric rating [Score] - Reported': 'Pain severity'} )

X = dataset.iloc[:,1:]
y = dataset['is_not_survivor']




from imblearn.over_sampling import SMOTE
os = SMOTE()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

os_data_X,os_data_y=os.fit_resample(X_train, y_train)



# ##  Model train and optimising 
from sklearn import metrics


logreg = LogisticRegression()

logreg.fit(os_data_X, os_data_y)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set before optimising : {:.2f}'.format(logreg.score(X_test, y_test)))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# Add constant term
X_lt_constant = sm.add_constant(os_data_X, prepend=False)
  
# Building model and fit the data (using statsmodels Logit)
logit_model = sm.GLM(os_data_y, X_lt_constant, family=sm.families.Binomial()).fit()

# Display summary results
print(logit_model.summary())

#remove one columns which are statistically insignificant p> 0.05
os_data_X.drop(['Body Mass Index','Diastolic Blood Pressure', 'Oxygen saturation', 'QOLS'], axis=1, inplace=True)
X_test.drop(['Body Mass Index','Diastolic Blood Pressure','Oxygen saturation','QOLS'], axis=1, inplace=True)


logreg.fit(os_data_X, os_data_y)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set after optimising: {:.2f}'.format(logreg.score(X_test, y_test)))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

precision_score = metrics.precision_score(y_test, y_pred)
print("Sensitivity is :" + str(precision_score))


# Add constant term
X_lt_constant = sm.add_constant(os_data_X, prepend=False)
  
# Building model and fit the data (using statsmodels Logit)
logit_model = sm.GLM(os_data_y, X_lt_constant, family=sm.families.Binomial()).fit()



# Display summary results
report = logit_model.summary()
print(report)


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

# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(logreg, X, y, cv=5)

abs_importance = np.abs(logreg.coef_[0])
total_importance = np.sum(abs_importance)

importance = (abs_importance /total_importance)*100
featurenames = logreg.feature_names_in_
# summarize feature importance
for i,v in enumerate(importance):
 print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.barh(featurenames, importance)
plt.show()


plt.pie(importance, labels = featurenames)
plt.show() 








