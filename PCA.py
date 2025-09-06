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
from sklearn.naive_bayes import GaussianNB
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


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(os_data_X)


# Apply PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=None)
pca.fit(X_scaled)


PC_values = np.arange(pca.n_components_) + 1


# Make the scree plot
plt.plot(PC_values,np.cumsum(pca.explained_variance_ratio_ * 100))
plt.xlabel("Number of components (Dimensions)")
plt.ylabel("Explained variance (%)")


pca_exp_var =  pca.explained_variance_ratio_ * 100
cum_pca_exp_var_ =  np.cumsum(pca.explained_variance_ratio_ * 100)

## with only 9 components

from sklearn.decomposition import PCA
pca = PCA(n_components=9)
X_pca = pca.fit_transform(X_scaled)

x_test_scale = sc.transform(X_test)
x_test_pca = pca.transform(x_test_scale)

# ##  Model train and optimising 
from sklearn import metrics



# naive bayes
nb = GaussianNB()

nb.fit(X_pca, os_data_y)

y_pred_nb = nb.predict(x_test_pca)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_nb))

precision_score = metrics.precision_score(y_test, y_pred_nb)
print("Senesitivity for NB is :" + str(precision_score))


#ANN

model = Sequential()
model.add(Dense(12, input_shape=(9,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_pca, os_data_y, epochs=10, batch_size=50)



y_pred = model.predict(x_test_pca).ravel()


y_pred_binery = np.around(y_pred)



from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_binery))

precision_score = metrics.precision_score(y_test, y_pred_binery)
print("Sensitivity for ANN :" + str(precision_score))




# # ##  Model train and optimising 
# from sklearn import metrics


# nb = GaussianNB()

# nb.fit(os_data_X, os_data_y)

# y_pred = nb.predict(X_test)
# print('Accuracy of logistic regression classifier on test set before optimising : {:.2f}'.format(nb.score(X_test, y_test)))

# from sklearn.metrics import classification_report
# print(classification_report(y_test, y_pred))


# #ROC and AUC
# fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
# auc = metrics.roc_auc_score(y_test, y_pred)

# #create ROC curve
# plt.figure(figsize=(10,8))
# plt.plot(fpr,tpr,label="Logistic Regression AUC="+str(auc))
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.legend(loc=4)
# plt.show()

