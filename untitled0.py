#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 10:38:55 2020

@author: TomMSchult
"""

import os
import pandas as pd
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

bankData = pd.read_csv('/Users/TomMSchult/Documents/FS/Master/Semester 1/Introduction to Data Analytics in Business/bankmarketing/BankMarketing.csv')

head = bankData.head()

info = bankData.info()
describe = bankData.describe()

bankData.dtypes

import seaborn as sns

catA = bankData.iloc[:,0:7]
catA.dtypes
headA = catA.head()

sns.distplot(bankData['age'],kde=False,color='slateblue')
sns.boxplot(bankData['age'],color='lightblue')

sns.boxplot(bankData['duration'],color='lightblue')

ax = sns.countplot(bankData['job'],color='lightblue')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()


axM = sns.countplot(bankData['marital'],color='lightblue')
axM.set_xticklabels(axM.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()

axE = sns.countplot(bankData['education'],color='lightblue')
axE.set_xticklabels(axE.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()

 

import seaborn as sns
df2 = bankData.iloc[:,4:7]

sns.countplot(x="variable", hue="value", data=pd.melt(df2),palette=['darkblue','blue','lightblue'])



sns.boxplot(data=bankData, orient="v", palette="Set2")

sns.jointplot(y="pdays",x="previous",data=bankData)
sns.jointplot(y="poutcome",x="previous",data=bankData)


sns.jointplot(y="cons.price.idx",x="y",data=bankData)

sns.jointplot(y="cons.conf.idx",x="y",data=bankData)
