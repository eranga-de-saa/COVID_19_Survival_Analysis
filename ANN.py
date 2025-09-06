#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 20:36:44 2023

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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

dataset = pd.read_csv("dataset_preprocessed.csv")

X = dataset.iloc[:,1:]
y = dataset['is_not_survivor']


from imblearn.over_sampling import SMOTE
os = SMOTE()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

os_data_X,os_data_y=os.fit_resample(X_train, y_train)


# ##  Model train and optimising 
from sklearn import metrics

model = Sequential()
model.add(Dense(12, input_shape=(12,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(os_data_X, os_data_y, epochs=10, batch_size=50)



y_pred = model.predict(X_test).ravel()


y_pred_binery = np.around(y_pred)

precision_score = metrics.precision_score(y_test, y_pred_binery)
print("Sensitivity is :" + str(precision_score))




from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_binery))


#ROC and AUC
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_binery)
auc = metrics.roc_auc_score(y_test, y_pred_binery)

#create ROC curve
plt.figure(figsize=(10,8))
plt.plot(fpr,tpr,label="ANN AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

