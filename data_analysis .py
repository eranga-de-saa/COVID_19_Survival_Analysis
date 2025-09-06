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
devices = pd.read_csv("devices.csv")
supplies = pd.read_csv('supplies.csv')
procedures = pd.read_csv("procedures.csv")
medications = pd.read_csv("medications.csv")

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

# The number of inpatient survivors
np.intersect1d(inpatient_ids, survivor_ids).shape

# The number of inpatient non-survivors
np.intersect1d(inpatient_ids, deceased_patients).shape

inpatient_ids.shape

# The following code presents lab values taken for COVID-19 patients. Values are separated into survivors and non survivors.
# The first block of code selects lab values of interest from all observations in the simulation.

lab_obs = observations[(observations.CODE == '48065-7') | (observations.CODE == '26881-3') |
                          (observations.CODE == '2276-4') | (observations.CODE == '89579-7') |
                          (observations.CODE == '2532-0') | (observations.CODE == '731-0') |
                          (observations.CODE == '14804-9')
                      ]

# Select COVID-19 conditions out of all conditions in the simulation
covid_conditions = conditions[conditions.CODE == 840539006]

# Merge the COVID-19 conditions with the patients
covid_patients = covid_conditions.merge(
    patients, how='left', left_on='PATIENT', right_on='Id')

# Add an attribute to the DataFrame indicating whether this is a survivor or not.
covid_patients['survivor'] = covid_patients.PATIENT.isin(survivor_ids)

# Reduce the columns on the DataFrame to ones needed
covid_patients = covid_patients[['START', 'PATIENT', 'survivor', 'CODE']]

# Calculate attributes needed to support the plot. Also coerce all lab values into a numeric data type.
covid_patients_obs = covid_patients.merge(lab_obs, on='PATIENT')
covid_patients_obs['START'] = pd.to_datetime(covid_patients_obs.START)
covid_patients_obs['DATE'] = pd.to_datetime(covid_patients_obs.DATE)
covid_patients_obs['lab_days'] = covid_patients_obs.DATE - \
    covid_patients_obs.START
covid_patients_obs['days'] = covid_patients_obs.lab_days / \
    np.timedelta64(1, 'D')
covid_patients_obs['VALUE'] = pd.to_numeric(
    covid_patients_obs['VALUE'], errors='coerce')

loinc_to_display = {'CODE_y = 48065-7': 'D-dimer', 'CODE_y = 2276-4': 'Serum Ferritin',
                    'CODE_y = 89579-7': 'High Sensitivity Cardiac Troponin I',
                    'CODE_y = 26881-3': 'IL-6', 'CODE_y = 731-0': 'Lymphocytes',
                    'CODE_y = 14804-9': 'Lactate dehydrogenase'}
catplt = sns.catplot(x="days", y="VALUE", hue="survivor", kind="box", col='CODE_y',
            col_wrap=2, sharey=False, sharex=False, data=covid_patients_obs, palette=["C1", "C0"])

for axis in catplt.fig.axes:
    axis.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    axis.xaxis.set_major_locator(ticker.MultipleLocator(base=4))
    axis.set_title(loinc_to_display[axis.title.get_text()])

plt.show()


loinc_to_display = {'CODE_y = 48065-7': 'D-dimer', 'CODE_y = 2276-4': 'Serum Ferritin',
                    'CODE_y = 89579-7': 'High Sensitivity Cardiac Troponin I',
                    'CODE_y = 26881-3': 'IL-6', 'CODE_y = 731-0': 'Lymphocytes',
                    'CODE_y = 14804-9': 'Lactate dehydrogenase'}
catplt = sns.catplot(x="days", y="VALUE", hue="survivor", kind="point", col='CODE_y',
            col_wrap=2, sharey=False, sharex=False, data=covid_patients_obs, palette=["C1", "C0"])

for axis in catplt.fig.axes:
    axis.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    axis.xaxis.set_major_locator(ticker.MultipleLocator(base=4))
    axis.set_title(loinc_to_display[axis.title.get_text()])

plt.show()

# Set up a new DataFrame with boolean columns representing various outcomes, like admit, recovery or death
cp = covid_conditions.merge(
    patients, how='left', left_on='PATIENT', right_on='Id')
isolation_ids = care_plans[(care_plans.CODE == 736376001) & (
    care_plans.REASONCODE == 840539006)].PATIENT
cp['isolation'] = cp.Id.isin(isolation_ids)
cp['admit'] = cp.Id.isin(inpatient_ids)
cp['recovered'] = cp.Id.isin(survivor_ids)
cp['death'] = cp.DEATHDATE.notna()
icu_ids = encounters[encounters.CODE == 305351004].PATIENT
cp['icu_admit'] = cp.Id.isin(icu_ids)
vent_ids = procedures[procedures.CODE == 26763009].PATIENT
cp['ventilated'] = cp.Id.isin(vent_ids)

# Outcomes for all COVID-19 Patients
# This code builds a new DataFrame for the purposes of display. The DataFrame contains the percentages of patients that experience a particular outcome. Percentages are then provided for only hospitalized patients, ICU admitted patients and ventilated patients.
hospitalized = (cp.admit == True)
icu = (cp.icu_admit == True)
vent = (cp.ventilated == True)
covid_count = cp.Id.size
row_filters = {'Home Isolation': (cp.isolation == True), 'Hospital Admission': hospitalized, 'ICU Admission': icu,
 'Ventilated': vent, 'Recovered': (cp.recovered == True), 'Death': (cp.death == True)}

table_rows = []
for category, row_filter in row_filters.items():
    row = {'Outcome': category}
    row['All Patients'] = cp[row_filter].Id.size / covid_count
    row['Hospitalized'] = cp[row_filter & hospitalized].Id.size / \
        hospitalized.value_counts()[True]
    row['ICU Admitted'] = cp[row_filter & icu].Id.size / icu.value_counts()[True]
    row['Required Ventilation'] = cp[row_filter &
        vent].Id.size / vent.value_counts()[True]
    table_rows.append(row)

pd.DataFrame.from_records(table_rows)


# Outcomes for ICU Admitted Patients
# Essentially a sub table from above, looking only at ICU patients.

icu_only = cp[cp.icu_admit == True]

vent = (icu_only.ventilated == True)
covid_count = icu_only.Id.size
row_filters = {'Ventilated': vent, 'Recovered': (
    icu_only.recovered == True), 'Death': (icu_only.death == True)}

table_rows = []
for category, row_filter in row_filters.items():
    row = {'Outcome': category}
    row['ICU Admitted'] = icu_only[row_filter].Id.size / covid_count
    row['Required Ventilation'] = icu_only[row_filter &
        vent].Id.size / vent.value_counts()[True]
    table_rows.append(row)

pd.DataFrame.from_records(table_rows)

# Start to build a DataFrame that we can use to look at other conditions in relation to COVID-19
covid_info = cp[['PATIENT', 'recovered', 'death', 'START',
    'DEATHDATE', 'BIRTHDATE', 'GENDER', 'admit', 'icu_admit']]
covid_info = covid_info.rename(columns={'START': 'covid_start'})

# Grab all of the conditions starting after January 20, 2020. This is a hack to get only conditions that are related to COVID-19. We will end up merging these with the COVID patients.
covid_related_conditions = conditions[pd.to_datetime(
    conditions.START) > pd.to_datetime('2020-01-20')]

# This DataFrame will contain all conditions for COVID-19 patients, where START can be compared to covid_start to see how long after the COVID-19 diagnosis something happened.
covid_patient_conditions = covid_info.merge(
    covid_related_conditions, on='PATIENT')

# Create a DataFrame with columns that show a condition's start and end in days relative to COVID-19 diagnosis. Also create a column that calculates the number of days between COVID-19 diagnosis and a person's death.
covid_patient_conditions['start_days'] = (pd.to_datetime(
    covid_patient_conditions.START) - pd.to_datetime(covid_patient_conditions.covid_start)) / np.timedelta64(1, 'D')
covid_patient_conditions['end_days'] = (pd.to_datetime(
    covid_patient_conditions.STOP) - pd.to_datetime(covid_patient_conditions.covid_start)) / np.timedelta64(1, 'D')
covid_patient_conditions['death_days'] = (pd.to_datetime(
    covid_patient_conditions.DEATHDATE) - pd.to_datetime(covid_patient_conditions.covid_start)) / np.timedelta64(1, 'D')

# Add an age column to the DataFrame for rows where the patient has died
covid_info.loc[covid_info.death == True, 'age'] = (pd.to_datetime(
    covid_info.DEATHDATE) - pd.to_datetime(covid_info.BIRTHDATE)) / np.timedelta64(1, 'Y')

# Populate ages for survivors based on the current date
covid_info.loc[covid_info.recovered == True, 'age'] = (datetime.datetime.now(
) - pd.to_datetime(covid_info.BIRTHDATE)) / np.timedelta64(1, 'Y')

# Create an age_range column that places individuals into 10 year age ranges, such as 0 - 10, 10 - 20, etc.
bins = list(range(0, 120, 10))
covid_info['age_range'] = pd.cut(covid_info.age, bins=bins)

# Mortality by Age and Sex
# A plot of deaths grouped by age range and gender.
chart = sns.catplot(x="age_range", kind="count", hue="GENDER",
                    data=covid_info[covid_info.death == True]);
for axes in chart.axes.flat:
    axes.set_xticklabels(axes.get_xticklabels(), rotation=90)
# A table view of the same information from above
covid_info[covid_info.death == True].groupby(
    ['age_range', 'GENDER']).count()[['PATIENT']]

# Build a DataFrame that shows the total count of a supply used on a given day
grouped_supplies = supplies.groupby(['DESCRIPTION', 'DATE']).sum()

# Supply Usage
# Small multiples plot of supply usage over time.
gs = grouped_supplies.reset_index()
gs['DATE'] = pd.to_datetime(gs.DATE)
g = sns.FacetGrid(gs, col="DESCRIPTION", col_wrap=3,
                  sharey=False, height=3, aspect=2)
g = g.map(sns.lineplot, "DATE", "QUANTITY", marker=".")
for axes in g.axes.flat:
    title = axes.get_title()
    if 'glove' in title:
        axes.set_title('Gloves')
    else:
        axes.set_title(title.replace("DESCRIPTION = ",
                       "").replace(" (physical object)", ""))
    for tick in axes.get_xticklabels():
        tick.set_rotation(90)


#  A table showing total supplies used over the entire simulation
supplies.groupby(['DESCRIPTION']).sum()[['QUANTITY']]

# Build a DataFrame that has cumulative case counts over time
case_counts = conditions[conditions.CODE == 840539006].groupby('START').count()[
                                                               ['PATIENT']]
case_counts['total'] = case_counts['PATIENT'].cumsum()
case_counts = case_counts.rename(columns={'PATIENT': 'daily'})
case_counts = case_counts.reset_index()
case_counts['START'] = pd.to_datetime(case_counts.START)

# Cumulative Case Count
# Show total cases over time

axes = sns.lineplot(x='START', y='total', data=case_counts)
plt.xticks(rotation=90)
plt.show()

# Medication Dispenses
# This table shows medications dispensed to patients with COVID-19 since January 20, 2020.
covid_meds = medications[pd.to_datetime(
    medications.START) > pd.to_datetime('2020-01-20')]
covid_meds = covid_info.merge(covid_meds, on='PATIENT')

covid_meds.groupby(['DESCRIPTION']).sum()[['DISPENSES']].sort_values(
    'DISPENSES', ascending=False).head(10)

# hospital stats
# For patients with COVID-19, calculate the average hospital length of stay as well as total hospital days for all COVID-19 patients. Provide the same information for ICU patients
device_codes = [448907002, 449071006, 36965003]
grouped_dev = devices[devices.CODE.isin(device_codes)].groupby(
    ['DESCRIPTION', 'START']).count()
grouped_dev = grouped_dev.reset_index()
grouped_dev['START'] = pd.to_datetime(grouped_dev.START)


# Device Usage
# Show the number of devices used to treat COVID-19 over time.
g = sns.FacetGrid(grouped_dev.reset_index(), col="DESCRIPTION",
                  col_wrap=3, sharey=False, height=3, aspect=2)
g = g.map(sns.lineplot, "START", "PATIENT", marker=".")
for axes in g.axes.flat:
    title = axes.get_title()
    axes.set_title(title.replace("DESCRIPTION = ",
                   "").replace(" (physical object)", ""))
    for tick in axes.get_xticklabels():
        tick.set_rotation(90)

# Medication Dispenses
# This table shows devices used to patients with COVID-19 since January 20, 2020.
covid_devices = devices[pd.to_datetime(
    devices.START) > pd.to_datetime(pd.to_datetime('2020-01-20'))]
covid_devices = covid_info.merge(covid_devices, on="PATIENT")
# covid_devices.groupby(['CODE']).sum()[['DESCRIPTION']].sort_values('DESCRIPTION', ascending=False).head(10)


# obervations
covid_patients_all_obs = covid_patients.merge(observations, on='PATIENT')
# This table shows all obervations of patients with COVID-19 since January 20, 2020.
# covid_patients_all_obs["CODE_DESCRIPTION"] = covid_patients_all_obs["CODE_y"].astype(
#     str) + '_' + covid_patients_all_obs["DESCRIPTION"]
covid_patients_all_obs["desc"] = covid_patients_all_obs["DESCRIPTION"]
covid_patients_all_obs = covid_patients_all_obs[pd.to_datetime(
    covid_patients_all_obs.START) > pd.to_datetime('2020-01-20')]
covid_patients_all_obs['VALUE'] = pd.to_numeric(
    covid_patients_all_obs['VALUE'], errors='coerce')

# covid_patients_all_obs_before = covid_patients_all_obs
# covid_patients_all_obs_before = covid_patients_all_obs_before['TYPE', 'UNITS']

covid_patients_all_obs = covid_patients_all_obs.drop(
    ['START', 'CODE_x', 'DATE', 'ENCOUNTER', 'CODE_y', 'DESCRIPTION'], axis=1)

covid_patients_all_obs = covid_patients_all_obs.drop(['TYPE', 'UNITS'], axis=1)

# defaoult numpy mean()
covid_patients_all_obs = covid_patients_all_obs.pivot_table(values='VALUE', index = ['PATIENT', 'survivor'], columns ='desc', aggfunc=np.mean)
# covid_patients_all_obs = covid_patients_all_obs.pivot_table(values='VALUE', index = ['PATIENT', 'survivor'], columns ='CODE_DESCRIPTION', aggfunc=np.mean)
covid_patients_all_obs.reset_index()

number_of_missing_values = covid_patients_all_obs.isna().sum()
number_of_missing_values = number_of_missing_values.sort_values(ascending =True)

column_names_of_missing_value_columns= []
for idx, val in number_of_missing_values.items():
    if(val > 1600):
        column_names_of_missing_value_columns.append(idx)

covid_patients_all_obs = covid_patients_all_obs.drop(column_names_of_missing_value_columns, axis =1)

covid_patients_all_obs = covid_patients_all_obs.reset_index()
# covid_patients_all_obs.groupby(['code'])observati

for i,column in enumerate(covid_patients_all_obs.columns):
    if(1>1):
        sns.displot(covid_patients_all_obs, x = column)
        plt.show()
        
        # swap the minority class label to 1
covid_patients_all_obs['survivor'] = covid_patients_all_obs['survivor'].apply(lambda x: False if x == True else True)
covid_patients_all_obs = covid_patients_all_obs.rename(columns={'survivor': 'is_not_survivor'})

covid_patients_all_obs = covid_patients_all_obs.rename(columns=[{'Oxygen saturation in Arterial blood': 'Oxygen saturation'}])
covid_patients_all_obs = covid_patients_all_obs.rename(columns=[{'Pain severity - 0-10 verbal numeric rating [Score] - Reported': 'Pain severity'} ])

fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 7), sharex=False, sharey=False)
axes = axes.ravel()  # array to 1D
cols = covid_patients_all_obs.columns[2:]  # create a list of dataframe columns to use

for col, ax in zip(cols, axes):
    data = covid_patients_all_obs[[col, 'is_not_survivor']]  # select the data
    sns.kdeplot(data=data, x=col, hue='is_not_survivor', shade=True, ax=ax)
    ax.set(title=f'{col[0:20]}', xlabel=None)
    
fig.delaxes(axes[13])  # delete the empty subplot
fig.delaxes(axes[14])  # delete the empty subplot
fig.tight_layout()
plt.show()

test_obs = observations[pd.to_datetime(
    observations.DATE) > pd.to_datetime('2020-01-20')]

test_obs_count = test_obs["CODE"].unique().shape




        
# fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 7), sharex=False, sharey=False)
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

# test_obs = observations[pd.to_datetime(
#     observations.DATE) > pd.to_datetime('2020-01-20')]

# test_obs_count = test_obs["CODE"].unique().shape
#covid_patients_all_obs['is_not_survivor'].value_counts()

