#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 11:21:22 2023

@author: erangad
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import datetime
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.svm import SVC # "Support vector classifier"
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv("dataset_preprocessed.csv")

X = dataset.iloc[:,1:]
y = dataset['is_not_survivor']


from imblearn.over_sampling import SMOTE
os = SMOTE()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

os_data_X,os_data_y=os.fit_resample(X_train, y_train)



# ##  Model train and optimising 
from sklearn import metrics

# Define a result table as a DataFrame
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])


#naive bayes

nb = GaussianNB()

nb.fit(os_data_X, os_data_y)

y_pred_nb = nb.predict(X_test)

# Train the models and record the results    
precision, recall, _  = metrics.precision_recall_curve(y_test,  y_pred_nb)
auc = metrics.auc(recall, precision)
model = "Naive Bayes"
result_table = result_table.append({'classifiers':model,
                                        'recall':recall, 
                                        'precision':precision, 
                                        'auc':auc}, ignore_index=True)

# svm

svm = SVC(kernel='linear')

svm.fit(os_data_X, os_data_y)

y_pred_svm = svm.predict(X_test)

# Train the models and record the results    
precision, recall, _ = metrics.precision_recall_curve(y_test,  y_pred_svm)
auc = metrics.auc(recall, precision)
model = "SVM"
result_table = result_table.append({'classifiers':model,
                                        'recall':recall, 
                                        'precision':precision, 
                                        'auc':auc}, ignore_index=True)


# ANN

model = Sequential()
model.add(Dense(12, input_shape=(12,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(os_data_X, os_data_y, epochs=10, batch_size=50)



y_pred = model.predict(X_test).ravel()


y_pred_ann = np.around(y_pred)

# Train the models and record the results    
precision, recall, _ = metrics.precision_recall_curve(y_test,  y_pred_ann)
auc = metrics.auc(recall, precision)
model = "ANN"
result_table = result_table.append({'classifiers':model,
                                        'recall':recall, 
                                        'precision':precision, 
                                        'auc':auc}, ignore_index=True)


# logistic regression


logreg = LogisticRegression()
#remove one columns which are statistically insignificant p> 0.05
os_data_X.drop(['Body Mass Index','Diastolic Blood Pressure', 'Oxygen saturation in Arterial blood', 'QOLS'], axis=1, inplace=True)
X_test.drop(['Body Mass Index','Diastolic Blood Pressure','Oxygen saturation in Arterial blood','QOLS'], axis=1, inplace=True)


logreg.fit(os_data_X, os_data_y)

y_pred_logreg = logreg.predict(X_test)


# Train the models and record the results    
precision, recall, _ = metrics.precision_recall_curve(y_test,  y_pred_logreg)
auc = metrics.auc(recall, precision)
model = "Logistic Regression"
result_table = result_table.append({'classifiers':model,
                                        'recall':recall, 
                                        'precision':precision, 
                                        'auc':auc}, ignore_index=True)

## draw the combined ROC
# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)

fig = plt.figure(figsize=(8,6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['recall'], 
              result_table.loc[i]['precision'], 
              label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
    
# plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("Recall", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("Precision", fontsize=15)

plt.title('Precision Recall Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower left')

plt.show()