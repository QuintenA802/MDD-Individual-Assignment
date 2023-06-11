#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: quintenachterberg
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score


# Importing the data
rawDF = pd.read_csv('https://raw.githubusercontent.com/QuintenA802/MDD-Individual-Assignment/main/AQI%20and%20Lat%20Long%20of%20Countries.csv')
rawDF.info()

pd.set_option('display.max_columns', 5000)
rawDF.head()



# Data preperation
# Dropping Country & City column, since the KNN analysis that I'm doing shouldn't make predictions based on the country but rather data such as Carbon, Ozone, Nitrogen and Air Quality data.
columns_to_drop = ['Country', 'City']
rawDF = rawDF.drop(columns_to_drop, axis=1)

# Renaming column titles
rawDF.rename(columns=({'AQI Value':'Air_Value','AQI Category':'Air_category','CO AQI Value':'Carbon_value','CO AQI Category':'Carbon_category','Ozone AQI Category':'Ozone_category','NO2 AQI Value':'Nitrogen_value','NO2 AQI Category':'Nitrogen_category','PM2.5 AQI Value':'Fine_particle','PM2.5 AQI Category':'Fine_particle_category','Ozone AQI Value':'Ozone_value'}),inplace = True)
rawDF.head()



# Replacing the catrgory names to numbers
rawDF['Air_category'].replace({'Hazardous':0,'Unhealthy for Sensitive Groups':1,'Very Unhealthy':2,'Unhealthy':3,'Moderate':4,'Good':5},inplace = True)
rawDF['Carbon_category'].replace({'Good':5,'Unhealthy for Sensitive Groups':1,'Moderate':4},inplace = True)
rawDF['Ozone_category'].replace({'Good':5,'Moderate':4,'Unhealthy':3,'Very Unhealthy':2,'Unhealthy for Sensitive Groups':1},inplace = True)
rawDF['Nitrogen_category'].replace({'Good':5,'Moderate':4},inplace = True)
rawDF['Fine_particle_category'].replace({'Hazardous':0,'Unhealthy for Sensitive Groups':1,'Very Unhealthy':2,'Unhealthy':3,'Moderate':4,'Good':5},inplace = True)

# The categories have the data type 'object' by default, so the following code will change these to integers.
rawDF['Air_category']= rawDF['Air_category'].astype(int)
rawDF['Carbon_category']= rawDF['Carbon_category'].astype(int)
rawDF['Ozone_category']= rawDF['Ozone_category'].astype(int)
rawDF['Nitrogen_category']= rawDF['Nitrogen_category'].astype(int)
rawDF['Fine_particle_category']= rawDF['Fine_particle_category'].astype(int)

# To check if the code has worked:
rawDF.info()



# To show the relation between Air_Value as a dependent variable and the four independent variables, the relationship is plotted (all within one grid)
fig,axes = plt.subplots(nrows=2,ncols = 2,figsize=(20,10));

sns.scatterplot(ax = axes[0,0],x='Air_Value',y='Fine_particle',data = rawDF,color = 'red');
sns.scatterplot(ax = axes[0,1],x='Air_Value',y='Carbon_value',data = rawDF,color = 'green');
sns.scatterplot(ax = axes[1,0],x='Air_Value',y='Nitrogen_value',data = rawDF,color = 'Fuchsia');
sns.scatterplot(ax = axes[1,1],x='Air_Value',y='Ozone_value',data = rawDF,color='orange');

axes[0,0].title.set_text('Relationship air value and fine particles') # Linear relationship
axes[0,1].title.set_text('Relationship between air value and carbon value') # Linear relationship
axes[1,0].title.set_text('Relationship between air value and nitrogen value') # Polynomial relationship
axes[1,1].title.set_text('Relationship between air value and ozone value') # Linear relationship
 
# For the Polynomial relationship
#rawDF['Nitrogen_value_Linear'] = pow(rawDF['Nitrogen_value'], 2)
#rawDF.drop('Nitrogen_value', axis=1, inplace=True)

# Draw a new scatterplot to analyse the difference
#sns.scatterplot(x='Air_Value', y='Nitrogen_value_Linear', data=rawDF, color='Fuchsia')
#plt.title("Relationship between air value and nitrogen value (linear)")
#plt.show()

# Renaming it back for less confusion in the modelling phase
#rawDF.rename(columns=({'Nitrogen_value_Linear':'Nitrogen_value'}),inplace = True)



# Looking closer to five different variables
rawDF[['Air_Value', 'Carbon_value', 'Ozone_value', 'Nitrogen_value', 'Fine_particle']].describe()
#description.to_excel('/Users/quintenachterberg/Library/Mobile Documents/com~apple~CloudDocs/HAN Arnhem/ISB year 3/IB Minor Data Driven Decision Making in Business/Individueel verslag/Visuals/description.xlsx', index=False)


# To do the normalization and to make the correlation heatmap, it is important to reverse the scale of the values.
    # Thanks to the categories it can be seen in the data that high values are 'not good' and low values are 'good'. It would be ideal if a 'good' value scores 100% when normalized in stead of 0%. 
reverse_scale_data = rawDF.copy()
reverse_scale_data = reverse_scale_data.drop(columns=['Air_category', 'Carbon_category', 'Ozone_category', 'Nitrogen_category', 'Fine_particle_category', 'lat', 'lng'])
reverse_scale_data.info()

# Create a DataFrame from the sample data
df = pd.DataFrame(reverse_scale_data)
df.info()

# Define the ranges for each column
ranges = {
    'Air_Value': (7, 500),
    'Carbon_value': (0, 133),
    'Ozone_value': (0, 222),
    'Nitrogen_value': (0, 91),
    'Fine_particle': (0, 500)
    
}

# Reverse the scale and normalize each column
for column, (min_val, max_val) in ranges.items():
    df[column] = max_val - df[column]  # Reverse the scale
    df[column] = df[column] / (max_val - min_val)  # Divide by the range
    df[column] = df[column] * 1  # Multiply by 1

# Rename before displaying changes
df.rename(columns={
    'Air_Value': 'Air_value_N',
    'Carbon_value': 'Carbon_value_N',
    'Ozone_value': 'Ozone_value_N',
    'Nitrogen_value': 'Nitrogen_value_N',
    'Fine_particle': 'Fine_particle_N'
}, inplace=True)

df.head()

# Display the normalized DataFrame. See that they have a max of 1
df[['Air_value_N', 'Carbon_value_N', 'Ozone_value_N', 'Nitrogen_value_N', 'Fine_particle_N']].describe()
#description_normalized.to_excel('/Users/quintenachterberg/Library/Mobile Documents/com~apple~CloudDocs/HAN Arnhem/ISB year 3/IB Minor Data Driven Decision Making in Business/Individueel verslag/Visuals/description_n.xlsx', index=False)

# Replacing the normalized and reversed scaled columns with the original non-normalized and non-reversed scaled columns
new_rawDF = rawDF.copy()

# Merge the datasets on all columns
merged_df = pd.merge(new_rawDF, df, left_index=True, right_index=True)

# Delete the old (non-normalized) columns
merged_df = merged_df.drop(columns=['Air_Value', 'Carbon_value', 'Ozone_value','Nitrogen_value','Fine_particle'])

# Display the merged dataset
merged_df.info()

#file_path = '/Users/quintenachterberg/Library/Mobile Documents/com~apple~CloudDocs/HAN Arnhem/ISB year 3/IB Minor Data Driven Decision Making in Business/Individueel verslag/output.csv'
#merged_df.to_csv(file_path, index=False)



# Doing correlation analysis
corr = merged_df.copy()

corr_matrix = corr.loc[:,['Air_value_N','Air_category','Carbon_value_N','Carbon_category','Ozone_value_N','Ozone_category','Nitrogen_value_N','Nitrogen_category','Fine_particle_N','Fine_particle_category']].corr()
print (corr_matrix)
#Create heatmap of the correlation matrix for clarification
sns.heatmap(corr_matrix, annot = True)

# Create a more clear heatmap (Run next two lines at once)
fig, ax = plt.subplots(figsize=(10,5)) 
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)


# Modelling
# Selecting the Y and X for testing
excluded = ['Air_category'] # list of columns to exclude
X = merged_df.loc[:, ~merged_df.columns.isin(excluded)]


y = merged_df['Air_category'] # 'Air_category' is now our y-axis and the rest of the columns the x-axis
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# make predictions on the test set
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=knn.classes_)
cm

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
disp.plot()

plt.show()

acc =accuracy_score(y_test, y_pred)

print("Accuracy of the model is", acc)

# Accuracy of the model is 0.9025753643441805








