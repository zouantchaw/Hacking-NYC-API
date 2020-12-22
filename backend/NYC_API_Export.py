import numpy as np
import pandas as pd
import certifi 
import urllib3
from urllib3 import request
import json
import numpy as np
from matplotlib import pyplot as plt

http = urllib3.PoolManager(
       cert_reqs='CERT_REQUIRED',
       ca_certs=certifi.where())

url = "https://data.cityofnewyork.us/resource/8wbx-tsch.json?$limit=50000"
r = http.request('GET', url)
r.status

data = json.loads(r.data.decode('utf-8'))
#print(data[0:5])

df = pd.json_normalize(data)
print(len(df))
print(df.columns)

df['Zip Code'] = [x.strip()[-5:] for x in df['base_address']]
zip_codes = df['Zip Code'].value_counts()[:10].index
num_cars = df['Zip Code'].value_counts()[:10]
print(zip_codes)

print(df['base_address'].isnull().values.any())

fig1 = plt.figure(figsize=(8,6))
ax = fig1.add_subplot()
plt.bar(range(len(zip_codes)), num_cars)
plt.title('Top 10 Zip Codes')
ax.set_xticks(range(len(zip_codes)))
ax.set_xticklabels(zip_codes)
plt.xticks(rotation=45)
ax.set_xlabel('Zip Code')
ax.set_ylabel('Number of Cars')

plt.show()