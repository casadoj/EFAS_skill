#!/usr/bin/env python
# coding: utf-8

# # Forecast exceedance
# ***
# 
# **Author**: Chus Casado<br>
# **Date**: 23-02-2023<br>
# 
# **Introduction**:<br>
# 
# 
# **Questions**:<br>
# 
# 
# **Tasks to do**:<br>
# 
# **Interesting links**<br>
# [Pythonic way to perform statistics across multiple variables with Xarray](https://towardsdatascience.com/pythonic-way-to-perform-statistics-across-multiple-variables-with-xarray-d0221c78e34a)

# In[1]:


import os
path_root = os.getcwd()
import glob
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import time

import warnings
warnings.filterwarnings("ignore")

os.chdir('../py/')
from notifications import *
os.chdir(path_root)


# ### 1 Discharge forecast
# 
# #### List available data

# In[7]:


path_forecast = 'E:/casadje/Documents/skill_assessment/data/CDS/forecast/'

models = ['COS', 'DWD', 'EUD', 'EUE']

# list files
fore_files = {model: [] for model in models}
for year in [2020, 2021, 2022]:
    for month in range(1, 13):    
        # list files
        for model in models:
            fore_files[model] += glob.glob(f'{path_forecast}{model}/{year}/{month:02d}/*.nc')

# count files and check if all are avaible
n_files = pd.Series(data=[len(fore_files[model]) for model in models], index=models)

# list of forecast from the beginning to the end of the data
start, end = datetime(1900, 1, 1), datetime(2100, 1, 1)
for model in models:
    st, en = [datetime.strptime(fore_files[model][step][-13:-3], '%Y%m%d%H') for step in [0, -1]]
    start = max(st, start)
    end = min(en, end)
dates = pd.date_range(start, end, freq='12h')

# find missing files
if any(n_files != len(dates)):
    missing = {}
    for model in models:
        filedates = [datetime.strptime(file[-13:-3], '%Y%m%d%H') for file in fore_files[model]]    
        missing[model] = [date for date in dates if date not in filedates]
    print('mising files:', missing)

# trim files to the period where all models are available
for model in models:
    fore_files[model] = [file for file in fore_files[model] if start <= datetime.strptime(file[-13:-3], '%Y%m%d%H') <= end]
    print('{0}:\t{1} files'.format(model, len(fore_files[model])))


# ## 2 Analysis

# ### 2.1 Stations 

# In[9]:


# load selected points for all the catchments
stations = pd.DataFrame()
catchments = []
results_path = '../results/select_reporting_points/'
for folder in os.listdir(results_path):
    try:
        stn_cat = pd.read_csv(f'{results_path}{folder}/points_selected.csv', index_col='station_id')
        stations = pd.concat((stations, stn_cat))
        catchments.append(folder)
    except:
        continue
print('no. stations:\t\t\t{0}'.format(stations.shape[0]))


# ### 2.2 Reforecast data: exceedance probability
# 
# This section will iteratively (station by station) load all the available forecast and compute the probability of exceeding the discharge threshold for each of the meteorological forcings. The result will be a NetCDF file for each station that contains the exceedance probability. These files will be later used in the skill assessment.

# In[10]:


# export files station by station
path = f'../data/exceedance/forecast/'
if os.path.exists(path) is False:
    os.makedirs(path)

# select stations that haven't been processed before
files = glob.glob(f'{path}*.nc')
if len(files) > 0:
    old_stations = [int(file.split('\\')[-1].split('.')[0]) for file in files]
    new_stations = set(stations.index).difference(old_stations)
    stations = stations.loc[new_stations]
    print('no. new stations:\t\t\t{0}'.format(stations.shape[0]))

# generate a DataArray with the discharge threshold of the stations in the catchment
thresholds = xr.DataArray(stations.rl5, dims='id', coords={'id': stations.index.astype(str).tolist()})


# In[ ]:


start = time.perf_counter()

# compute exceedance
exceedance = compute_exceedance_2(fore_files, thresholds)

for stn in exceedance.id.data:
    file = f'{stn:>04}.nc'
    # file = f'{stn:04d}.nc'
    if file in os.listdir(path):
        print(f'File {file} already exists')
        continue
    else:
        exceedance.sel(id=stn).to_netcdf(f'{path}{file}')
        
end = time.perf_counter()

print('excecution time: {0:.1f} s'.format(end - start))

