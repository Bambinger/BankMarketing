import os
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import seaborn as sns

# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
# load data and preprocess it

#####################
# CHANGE your path here
#####################
data = pd.read_csv('BankMarketing.csv')

# age
# Variant 1
# age_bins = [0, 21, 31, 41, 51, 61, 100]
# age_label = ['0-20', '21-30', '31-40', '41-50', '51-60', '>60']
# # just changing the row age here
# data['age'] = pd.cut(data['age'], bins=age_bins, labels=age_label)
# data['age'].describe()

# Variant 2 - equally distributed categories -> 10 ?
# data["age"] = pd.qcut(data["age"], q=10)


# #Variant 3
age_bins = [0, 25, 35, 45, 55, 65, 100]
age_label = ['0-24', '25-34', '35-44', '45-54', '55-64', '>60']
# just changing the row age here
data['age'] = pd.cut(data['age'], bins=age_bins, labels=age_label)
data['age'].describe()

lb_age = LabelEncoder()
data['age_encode'] = lb_age.fit_transform(data['age'])
data[['age', 'age_encode']].head()
data = data.drop('age', axis=1)

# job
# Varaint 1: binarise in the end with pandas

# Variant 2:
data = data.drop('job', axis=1)

# Variant 3: categorize higher and lower jobs


# marital
# Variant 1: unknown to single
data['marital'].unique()
data['marital'] = data['marital'].replace(['unknown'], ['single'])

# # Variant 2: assign unknown to most likely case
# data['marital'] = data['marital'].replace(['unknown'],['married'])

# education
# Variant 1: binarise

# # Variant 2: encoding to numbers
# data['education'].unique()
# data = data.loc[data['education']!='unknown']
# data['education'] = data['education'].replace(['basic.4y', 'basic.6y', 'basic.9y', 'illiterate', 'high.school',
#                                                'professional.course', 'university.degree'],[0, 0, 0, 0, 1, 1, 1])

# default -> drop yes
#data = data.drop('default', axis=1)
data = data.loc[data['default'] != 'yes']

# housing
# loan
# Variant 1:

# # Variant 2:
data = data.drop('loan', axis=1)


# contact

# month
# Variant 1 -> keep

# # Variant 2 -> label encode just from 0 -8 -> based on y-Ratio
# data['month'].unique()
# data['month'] = data['month'].replace(['may', 'jul', 'nov', 'aug', 'apr', 'oct', 'sep', 'dec', 'mar'],
#                                       [0, 1, 2, 3, 4, 5, 6, 7, 8])

# day_of_week
# Variant 1
data['day_of_week'] = data['day_of_week'].replace(['tue', 'wed', 'thu'], ['tue_to_thu', 'tue_to_thu', 'tue_to_thu'])

# # Variant 2
# data['day_of_week'] = data['day_of_week'].replace(['tue', 'wed', 'thu'],['tue_to_thu', 'tue_to_thu', 'tue_to_thu'])
# lb_day = LabelEncoder()
# data['day_of_week_encode'] = lb_day.fit_transform(data['day_of_week'])
# data[['day_of_week', 'day_of_week_encode']].head()
# data = data.drop('day_of_week', axis = 1)

# duration
# Variant 1 drop
data = data.drop('duration', axis=1)

# campaign
# Variant 1 Bins
campaign_bins = [-np.inf, 2, 3, 4, 6, np.inf]
campaign_label = ['exac_1', 'exac_2', 'exac_3', '4_to_5', 'more_5']
data['campaign'] = pd.cut(data['campaign'], bins=campaign_bins, labels=campaign_label)

# Variant 2 different Bins
# campaign_bins = [-np.inf, 2, 4, 100]
# campaign_label = ['exac_1', '2_to_3', 'more_3']
# data['campaign'] = pd.cut(data['campaign'], bins=campaign_bins, labels=campaign_label)


# pdays
# Variant 1 Bins
pdays_bins = [-np.inf, 21, 1000]
pdays_label = ['contact', 'no_contact']
data['pdays'] = pd.cut(data['pdays'], bins=pdays_bins, labels=pdays_label)

# previous
# Variant 1 bins
previous_bins = [-np.inf, 1, np.inf]
previous_label = ['no_contact', 'contact']
data['previous'] = pd.cut(data['previous'], bins=previous_bins, labels=previous_label)

# Variant 2 bins
# previous_bins = [-np.inf, 1, 2, 8]
# previous_label = ['no_contact', 'one_contact', 'more_than_one_contact']
# data['previous'] = pd.cut(data['previous'], bins=previous_bins, labels=previous_label)

# poutcome
# Variant 1 -> keep

# Variant 2 Label Encode
# data['poutcome'] = data['poutcome'].replace(['nonexistant', 'failure', 'success'],[0, 1, 2])

# emp.var.rate
# Variant 1 -> keep

# Variant 2
# data = data.drop('emp.var.rate', axis=1)


# cons.price.index -> keep

# cons.conf.index -> keep

# euribor3m

# Variant 1 -> keep

# Variant 2 -> drop
data = data.drop('euribor3m', axis=1)


# nr.employed
# Variant 1 -> keep

# Variant 2 -> drop
# data = data.drop('nr.employed', axis=1)


# name the CSV-File depending on the variante you chose
data.to_csv('data_final.csv', index=False)
