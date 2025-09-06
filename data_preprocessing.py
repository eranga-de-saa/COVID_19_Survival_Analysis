import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import datetime

conditions = pd.read_csv("conditions.csv")
patients = pd.read_csv("patients.csv")
observations = pd.read_csv("observations.csv")
care_plans = pd.read_csv("careplans.csv")
encounters = pd.read_csv("encounters.csv")

# Grab the IDs of patients that have been diagnosed with COVID-19
covid_patient_ids = conditions[conditions.CODE == 840539006].PATIENT.unique()

# This grabs every patient with a negative SARS-CoV-2 test. This will include patients who tested negative up front as well as patients that tested negative after leaving the hospital
negative_covid_patient_ids = observations[(observations.CODE == '94531-1') & (
    observations.VALUE == 'Not detected (qualifier value)')].PATIENT.unique()

# Grabs IDs for all patients that died in the simulation. This will be more than just COVID-19 deaths.
deceased_patients = patients[patients.DEATHDATE.notna()].Id

# Grabs IDs for patients that have completed the care plan for isolation at home.
completed_isolation_patients = care_plans[(care_plans.CODE == 736376001) & (
    care_plans.STOP.notna()) & (care_plans.REASONCODE == 840539006)].PATIENT

# Survivors are the union of those who have completed isolation at home or have a negative SARS-CoV-2 test.
survivor_ids = np.union1d(completed_isolation_patients,
                          negative_covid_patient_ids)

# Grab IDs for patients with admission due to COVID-19
inpatient_ids = encounters[(encounters.REASONCODE == 840539006) & (
    encounters.CODE == 1505002)].PATIENT


# The following code presents lab values taken for COVID-19 patients. Values are separated into survivors and non survivors.
# The first block of code selects lab values of interest from all observations in the simulation.


# Select COVID-19 conditions out of all conditions in the simulation
covid_conditions = conditions[conditions.CODE == 840539006]

# Merge the COVID-19 conditions with the patients
covid_patients = covid_conditions.merge(
    patients, how='left', left_on='PATIENT', right_on='Id')

# Add an attribute to the DataFrame indicating whether this is a survivor or not.
covid_patients['survivor'] = covid_patients.PATIENT.isin(survivor_ids)

# Reduce the columns on the DataFrame to ones needed
covid_patients = covid_patients[['START', 'PATIENT', 'survivor', 'CODE']]

# obervations
covid_patients_all_obs = covid_patients.merge(observations, on='PATIENT')
# This table shows all obervations of patients with COVID-19 since January 20, 2020.

covid_patients_all_obs["desc"] = covid_patients_all_obs["DESCRIPTION"]
covid_patients_all_obs = covid_patients_all_obs[pd.to_datetime(
    covid_patients_all_obs.START) > pd.to_datetime('2020-01-20')]
covid_patients_all_obs['VALUE'] = pd.to_numeric(
    covid_patients_all_obs['VALUE'], errors='coerce')

covid_patients_all_obs = covid_patients_all_obs.drop(
    ['START', 'CODE_x', 'DATE', 'ENCOUNTER', 'CODE_y', 'DESCRIPTION'], axis=1)

covid_patients_all_obs = covid_patients_all_obs.drop(['TYPE', 'UNITS'], axis=1)

covid_patients_all_obs = covid_patients_all_obs.pivot_table(
    values='VALUE', index=['PATIENT', 'survivor'], columns='desc', aggfunc=np.mean)

covid_patients_all_obs.reset_index()

number_of_missing_values = covid_patients_all_obs.isna().sum()
number_of_missing_values = number_of_missing_values.sort_values(ascending=True)

column_names_of_missing_value_columns = []
for idx, val in number_of_missing_values.items():
    if(val > 1600):
        column_names_of_missing_value_columns.append(idx)

covid_patients_all_obs = covid_patients_all_obs.drop(
    column_names_of_missing_value_columns, axis=1)

covid_patients_all_obs = covid_patients_all_obs.reset_index()

#patient isn't an observable
covid_patients_all_obs = covid_patients_all_obs.drop(['PATIENT'], axis=1)


# plotting the distribution
# fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(7, 15), sharex=False, sharey=False)
# axes = axes.ravel()  # array to 1D
# cols = covid_patients_all_obs.columns[2:]  # create a list of dataframe columns to use

# for col, ax in zip(cols, axes):
#     data = covid_patients_all_obs[[col, 'survivor']]  # select the data
#     sns.kdeplot(data=data, x=col, hue='survivor', shade=True, ax=ax)
#     ax.set(title=f'{col[0:20]}', xlabel=None)

# fig.delaxes(axes[13])  # delete the empty subplot
# fig.delaxes(axes[14])  # delete the empty subplot
# fig.tight_layout()
# plt.show()

# no of unique observations
# test_obs = observations[pd.to_datetime(
#     observations.DATE) > pd.to_datetime('2020-01-20')]
# test_obs_count = test_obs["CODE"].unique().shape

corr = covid_patients_all_obs.corr()


# handle missing values: MICE method

from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


lr = LinearRegression()
imp = IterativeImputer(estimator=lr, missing_values=np.nan,
                       max_iter=10, verbose=2, imputation_order='roman', random_state=0)
dataset_bp_array = imp.fit_transform(covid_patients_all_obs)
dataset = pd.DataFrame(data=dataset_bp_array[0:,0:], columns= covid_patients_all_obs.columns)

corr = dataset.corr()

# handle outliers
#check for outliers in predictors
summary = dataset.describe()

# plotting the distribution
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(7, 15), sharex=False, sharey=False)
axes = axes.ravel()  # array to 1D
cols = covid_patients_all_obs.columns[1:]  # create a list of dataframe columns to use

for col, ax in zip(cols, axes):
    sns.boxplot(data=dataset, x = col, ax=ax)
    ax.set(title=f'{col[0:20]}', xlabel=None)

fig.delaxes(axes[13])  # delete the empty subplot
fig.delaxes(axes[14])  # delete the empty subplot
fig.tight_layout()
plt.show()


#histplots to visualize outliers
# fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(7, 15), sharex=False, sharey=False)
# axes = axes.ravel()  # array to 1D
# cols = covid_patients_all_obs.columns[1:]  # create a list of dataframe columns to use
    
# for col, ax in zip(cols, axes):
#     sns.histplot(data=dataset, x = col, ax=ax)
#     ax.set(title=f'{col[0:20]}', xlabel=None)

# fig.delaxes(axes[13])  # delete the empty subplot
# fig.delaxes(axes[14])  # delete the empty subplot
# fig.tight_layout()
# plt.show()


# outliers present in body height, body mass index, body weight, DALY, Siast blood preasure, heart rate, pain severity, qaly, qols, respitory rate systolic blood preasure
# outliers not found in body temp, oxygen saturation

outlier_data_columns_iteration_1 = dataset.columns.difference(['survivor', 'Body temperature', 'Oxygen saturation in Arterial blood'])

# winsorize  technique to handle outliers

#Using scipy.stats.mstats.winsorize
from scipy.stats.mstats import winsorize 
# The 5% of the lowest value and the 5% of the highest values are replaced.


for col in outlier_data_columns_iteration_1:
    # The 5% of the lowest value and the 5% of the highest values are replaced.
    dataset[col] = pd.DataFrame(winsorize(dataset[col], limits = [0.05,0.05])) 
    
#verify outlier handling
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(7, 15), sharex=False, sharey=False)
axes = axes.ravel()  # array to 1D
cols = covid_patients_all_obs.columns[1:]  # create a list of dataframe columns to use
    
for col, ax in zip(cols, axes):
    sns.boxplot(data=dataset, x = col, ax=ax)
    ax.set(title=f'{col[0:20]}', xlabel=None)

fig.delaxes(axes[13])  # delete the empty subplot
fig.delaxes(axes[14])  # delete the empty subplot
fig.tight_layout()
plt.show()





# normalization // z score scaling 

dataset.iloc[:,1:] = dataset.iloc[:,1:].apply(lambda x: (x-x.mean())/ x.std(), axis=0)

# check for auto correclation among predictors

## check for multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor

#X = dataset.iloc[:,1:] 

vif_initial = pd.DataFrame()
vif_initial["features"] = dataset.iloc[:,1:].columns
vif_initial["VIF Factor"] = [variance_inflation_factor(dataset.iloc[:,1:].values, i) for i in range(dataset.iloc[:,1:].shape[1])]

#'Body Height', 'Body Mass Index', 'Body Weight' are have high multicollinearity vif>5

plt.figure(figsize=(20,16))
corr = dataset.iloc[:,1:].corr()
sns.heatmap(corr)
plt.show()

vif = vif_initial

# high_vif_index = vif[vif["VIF Factor"]>10]["VIF Factor"].idxmax()  #vif[vif["VIF Factor"]>5].tolist()  
while len(vif[vif["VIF Factor"]> 5]) > 0:
    high_vif_index = vif[vif["VIF Factor"] > 5]["VIF Factor"].idxmax()
    dataset.drop(vif['features'][high_vif_index], axis=1, inplace=True)
    del vif
    vif = pd.DataFrame()
    vif["features"] = dataset.iloc[:,1:].columns
    vif["VIF Factor"] = [variance_inflation_factor(dataset.iloc[:,1:].values, i) for i in range(dataset.iloc[:,1:].shape[1])]
    
 
    

# ## check for multicollinearity after removing highly correlated dimensions

vif_after = pd.DataFrame()
vif_after["features"] = dataset.iloc[:,1:].columns
vif_after["VIF Factor"] = [variance_inflation_factor(dataset.iloc[:,1:].values, i) for i in range(dataset.iloc[:,1:].shape[1])]
    
plt.figure(figsize=(20,16))
corr = dataset.iloc[:,1:].corr()
sns.heatmap(corr)
plt.show()

dataset['survivor'] = dataset['survivor'].apply(lambda x: False if x == True else True)
dataset = dataset.rename(columns={'survivor': 'is_not_survivor'})

dataset.to_csv('dataset_preprocessed.csv', index=False)


plt.figure()
ax = sns.boxplot(data=covid_patients_all_obs['Body Weight'])
ax.set(title= 'Body Weight', ylabel ='Kg')
plt.show()

plt.figure()
ax = sns.histplot(data=covid_patients_all_obs['Body Weight'])
ax.set(title= 'Body Weight', xlabel ='Kg')
plt.show()




    
    

