# Arnold-Portfolio
 Data Science Portfolio

# ALCOHOL CONSUMPTION EDA

Visualized the data with countries as the Unique Identifier and incomeperperson curtailing Gross Domestic Product per capita in constant 2000 US$ with the inflation but not the differences in the cost of living between countries has been taken into account. alcconsumption as Alcohol consumption per adult (age 15+), litres Recorded and estimated average alcohol consumption, adult (15+) per capita consumption in litres pure alcohol. suicideper100TH as Suicide, age adjusted, per 100 000 Mortality due to self-inflicted injury, per 100 000 standard population, age adjusted.employrate as Total employees age 15+ (% of population) Percentage of total population, age above 15, that has been employed during the given year. urbanrate as Urban population (% of total) Urban population refers to people living in urban areas as defined by national statistical offices (calculated using World Bank population estimates and urban ratios from the United Nations World Urbanization Prospects.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as mons

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
df = pd.read_csv("../input/alcohol-consumption/gapminder_alcohol.csv")
alcohol_data = pd.read_csv("../input/alcohol-consumption/gapminder_alcohol.csv")
alcohol_data.columns

from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
df.head()
df.describe()
df.shape
df.info()
df.dropna(inplace=True)
plt.figure(figsize=(14,7))
plt.title("Relationship between Alcohol Consumption and Suicide Rate")
plt.xlabel("alcconsumption")
plt.ylabel("suicideper100")

# visualizing the data

sns.scatterplot(x=alcohol_data['alcconsumption'],y=alcohol_data['suicideper100th'])
sns.regplot(x=alcohol_data['alcconsumption'],y=alcohol_data['suicideper100th'])
sns.scatterplot(x=alcohol_data['alcconsumption'],y=alcohol_data['suicideper100th'],hue=alcohol_data['incomeperperson'])
mons.bar(alcohol_data)
top10=alcohol_data.sort_values("alcconsumption",ascending=False).head(10)
plt.figure(figsize=(25,12))
sns.barplot(data=top10,x='country', y='alcconsumption')
plt.xticks(rotation=90)
plt.show()
fig, ax1 = plt.subplots(1,1,figsize=(16,7), dpi= 80)
ax1.plot(top10['country'],top10['suicideper100th'],color='tab:red')
ax1.set_xlabel('Country', fontsize=20)
ax1.set_ylabel('Suicideper 100th', color='tab:red', fontsize=10)
ax1.tick_params(axis='x', rotation=90 )
ax2 = ax1.twinx()
ax2.plot(top10['country'],top10['incomeperperson'],color='tab:green')
ax2.set_xlabel('Country', fontsize=20)
ax2.set_ylabel('Income perperson', color='tab:green', fontsize=20)
ax2.tick_params(axis='x', rotation=90 )

plt.show()
fig, ax1 = plt.subplots(1,1,figsize=(16,7), dpi= 80)
ax1.plot(top10['country'],top10['suicideper100th'],color='tab:red')
ax1.set_xlabel('Country', fontsize=20)
ax1.set_ylabel('Suicideper 100th', color='tab:red', fontsize=20)
ax1.tick_params(axis='x', rotation=90 )
ax2 = ax1.twinx()
ax2.plot(top10['country'],top10['employrate'],color='tab:green')
ax2.set_xlabel('Country', fontsize=20)
ax2.set_ylabel('employ rate', color='tab:green', fontsize=20)
ax2.tick_params(axis='x', rotation=90 )

plt.show()
fig, ax1 = plt.subplots(1,1,figsize=(16,7), dpi= 80)
ax1.plot(top10['country'],top10['suicideper100th'],color='tab:red')
ax1.set_xlabel('Country', fontsize=20)
ax1.set_ylabel('Suicideper 100th', color='tab:red', fontsize=20)
ax1.tick_params(axis='x', rotation=90 )
ax2 = ax1.twinx()
ax2.plot(top10['country'],top10['urbanrate'],color='tab:green')
ax2.set_xlabel('Country', fontsize=20)
ax2.set_ylabel('Urban rate', color='tab:green', fontsize=20)
ax2.tick_params(axis='x', rotation=90 )


plt.show()
alcohol_data=alcohol_data.dropna(axis=0)
y=alcohol_data.alcconsumption
alcohol_features=['urbanrate', 'employrate', 'suicideper100th','incomeperperson']
X=alcohol_data[alcohol_features]
X.describe()
X.head()              
              
