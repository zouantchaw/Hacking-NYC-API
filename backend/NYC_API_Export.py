import numpy as np
import pandas as pd
import certifi 
import urllib3
from urllib3 import request
import json
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#Use certifi to handle certificate verication
http = urllib3.PoolManager(
       cert_reqs='CERT_REQUIRED',
       ca_certs=certifi.where())

#Get data from the NYC API endpoint
url = "https://data.cityofnewyork.us/resource/8wbx-tsch.json?$limit=10000"
r = http.request('GET', url)
r.status

#Decode JSON into a dict object
data = json.loads(r.data.decode('utf-8'))


#Normalize dict to pandas DataFrame
df = pd.json_normalize(data)
print(len(df))
print(df.columns)

#Explore the data
print(df['license_type'].value_counts())
print(df['base_type'].value_counts())
print(df['reason'].value_counts())
print(df['base_name'].value_counts().nlargest(10))

#Mappings
base_type_mapping = {'LIVERY': 1,
                     'BLACK-CAR': 2,
                     'LUXURY': 3}

#Strip last five characters of 'base address' column and convert to int to acquire zip code
df['zip_code'] = [int(x.strip()[-5:]) for x in df['base_address']]
zip_codes = df['zip_code'].value_counts()[:10].index
num_cars = df['zip_code'].value_counts()[:10]

#Use base type mapping to create numerical column in df for base type
df['base_code'] = df.base_type.map(base_type_mapping)

print(df.base_code.value_counts())


print(zip_codes)

print(df['base_address'].isnull().values.any())

#Sort 'vehicle_year" column in ascending order
df = df.sort_values('vehicle_year', ascending = True).reset_index(drop=True)

#Cleaning vehicle year and zip code columns for linear regression analysis
years_data = df[['vehicle_year', 'zip_code']]
years_features = ['zip_code']
'''years_data['years_corrected'] = years_data.vehicle_year.apply\
                            (lambda x: df.vehicle_year.mean() if (int(x) > 2020) else x)'''
years_data['years_corrected'] = years_data['vehicle_year']
years_data_corrected = years_data[['years_corrected', 'zip_code']].reset_index()
filtered_years = years_data_corrected.dropna().reset_index()
years_labels = filtered_years['years_corrected']
years_predictors = filtered_years[years_features]
print(filtered_years.head())

#Training and testing sets from vehicle year and zip code data
x_train, x_test, y_train, y_test = train_test_split(years_predictors, years_labels,\
                                                    test_size=0.2, random_state=1)
model = LinearRegression()
model.fit(x_train, y_train)

print('Multiple Linear Regression analysis of Vehicle Year predicted by Zip Code')
print('Train score:', model.score(x_train, y_train))
print("Test score:", model.score(x_test, y_test))
y_predict = model.predict(x_test)

#Scatter plot of actual vs predicted vehicle year using zip code
plt.scatter(sorted(y_test), y_predict, alpha=0.2)
plt.xlabel('Actual Vehicle Year')
plt.xticks(rotation=45)
plt.ylabel('Predicted Vehicle Year')
plt.title('Actual vs Predicted Vehicle Year: Linear Regression analysis using Zip Code')
plt.show()


#Plot bar chart with number of cars from the top ten zip codes in the DataFrame
fig1 = plt.figure(figsize=(8,6))
ax = fig1.add_subplot()
plt.bar(range(len(zip_codes)), num_cars)
plt.title('Top 10 Zip Codes')
ax.set_xticks(range(len(zip_codes)))
ax.set_xticklabels(zip_codes)
plt.xticks(rotation=45)
ax.set_xlabel('Zip Code')
ax.set_ylabel('Number of Cars')

#plt.show()