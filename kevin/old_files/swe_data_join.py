#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 10:12:21 2022

@author: Kevin Xuan


FUNCTION
- merge SWE dataset with time series data on water gage
- join table based on lat, lon, and date
"""

# Import Packages
import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from datetime import datetime, date
from mpl_toolkits.basemap import Basemap



### 1. Read Data
# SWE Data
data = h5py.File('SN_SWE_WY1985.h5', 'r')
lat = data['lat'][0][::-1]
lon = data['lon'][:,0]

lats,lons = np.meshgrid(lat,lon)
# time series data
ts = pd.read_csv('gage_discharge.csv')



### 2. FUNCTION for finding nearest index

# ### Find Data value that's closest to 39.9927°, -120.8039°
def find_index(data,lon,lat):
    
    ### Find Lon
    maxx = 100
    lon_idx = 0
    for i in range(len(data['lon'])):
        dist = abs(data['lon'][i][0] - lon)
        if dist < maxx:
            maxx = dist
            lon_idx = i
    print('Longitude Index:',lon_idx)
    print('Longitude Value:',data['lon'][lon_idx][0])
    
    ### Find Lat
    maxx = 100
    lat_idx = 0
    for i in range(len(data['lat'][0])):
        dist = abs(data['lat'][0][i] - lat)
        if dist < maxx:
            maxx = dist
            lat_idx = i
    print('Latitude Index:',lat_idx)
    print('Latitude Value:',data['lat'][0][lat_idx])
    
    return lon_idx,lat_idx
# ####################


### find index
def index_finder(lon,lat):
    # Longtitude finder
    if lon < -123.3 or lon > -117.6:
        print('Longitude of Input is out of range! lon:',lon)
        return None
    elif lat < 35.4 or lat > 42:
        print('Latitude of Input is out of range! lat:',lat)
    else: #longtitude and latitude are within reasonable range
        lon_idx = round((lon + 123.3) * 1000)
        lat_idx = round((lat - 35.4) * 1000)
    
        return lon_idx,lat_idx




#### 3. Table joining ####
''' inputs of interested date'''
day = ts.iloc[10,:]


## Match date to swe data
date_format = "%Y/%m/%d"
d_date = datetime.strptime(day['Time'], date_format)
# extract year of date
d_year = d_date.year
# Extract number of days from SWE Data
num_days = d_date- datetime.strptime('{}/1/1'.format(d_year),date_format)
num_days = num_days.days


## Obtain swe data
swe_data = data['SWE'][num_days]
# flip over yaxis as lats are in a descending order --> need to change to ascending order
swe_data_flip = swe_data[:,::-1]

# Find closest idx to the lower left & upper right corner
ll_lon_idx,ll_lat_idx = index_finder(day['ll_lon'],day['ll_lat'])
tr_lon_idx,tr_lat_idx = index_finder(day['tr_lon'],day['tr_lat'])

### Obtain Value of Interested Region
region = swe_data_flip[ll_lon_idx:tr_lon_idx,ll_lat_idx:tr_lat_idx]
# change null values to null
region=region.astype('float')
region[region == -32768] = np.nan


# Convert to DataFrame
re = pd.DataFrame(region)
print(re)

# -----------------------------------

### Plot Graph
# Longitutude Range: [-123.3,-117.6]   -- Latitude Range: [35.4,42.0]
mp = Basemap(projection = 'merc',
             llcrnrlon = -120,
             llcrnrlat = 37,
             urcrnrlon = -119,
             urcrnrlat = 38,
             resolution = 'l')

x,y= mp(lons[ll_lon_idx:tr_lon_idx,ll_lat_idx:tr_lat_idx],
        lats[ll_lon_idx:tr_lon_idx,ll_lat_idx:tr_lat_idx])

cs = mp.pcolor(x,y,region,cmap= 'Blues')
cbar = mp.colorbar(cs,location = 'bottom',pad = '10%')

mp.drawcoastlines()
mp.drawstates()
mp.drawcountries()
mp.drawcounties()