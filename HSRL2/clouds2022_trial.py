#%%
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import datetime
import pathlib
import statistics
import glob
import os
import re
import math
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from scipy.integrate import quad
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D
from collections import Counter
from scipy.spatial import distance
import h5py
#%%
#testing one flight just to see the data structure and variables
file_path = '/home/disk/eos4/kathem24/activate/data/HSRL2/2022 HSRL-2/Cloud/ACTIVATE-HSRL2-Cloud_KingAir_20220111_R1_L1.h5'
with h5py.File(file_path, 'r') as f:
    print(list(f.keys()))
with h5py.File(file_path, 'r') as f:
    readme = f['000_Readme'][()]
    print(readme)
#%%
with h5py.File(file_path, 'r') as f:
    for var in [
        'DataProducts/cloud_height',
        'DataProducts/cloud_ext_average',
        'DataProducts/cloud_ext_prfl',
        'DataProducts/Altitude',
        'time',
        'z',
        'lat',
        'lon',
        'alt'
    ]:
        print("\n==============================")
        print(var)
        print("shape:", f[var].shape)
        print("attrs:")
        for key, val in f[var].attrs.items():
            print(key, ":", val)
#%%
with h5py.File(file_path, 'r') as f:
    time = f['time'][:]
    cloud_height = f['DataProducts/cloud_height'][:]
    cloud_ext = f['DataProducts/cloud_ext_average'][:]
plt.figure(figsize=(10,4))
plt.plot(time/3600, cloud_height)
plt.xlabel('UTC Time (hours)')
plt.ylabel('Cloud Top Height (m)')
plt.show()
#%%
print(np.nanmin(cloud_height))
print(np.nanmax(cloud_height))
print(np.nanmean(cloud_height))
print(np.nanmedian(cloud_height))
# %%
#import all cloud files from HSRL-2
# %%
# Import HSRL-2 Cloud files
clouds = []
successful_dates_clouds = []
dates_clouds = [
    '2022-01-11', '2022-01-12', '2022-01-15', '2022-01-18', 
    '2022-01-19', '2022-01-24', '2022-01-26', '2022-01-27',
    '2022-02-01', '2022-02-02', '2022-02-03', '2022-02-05', 
    '2022-02-15', '2022-02-16', '2022-02-19', '2022-02-22',
    '2022-02-26', '2022-03-03', '2022-03-04', '2022-03-13', 
    '2022-03-14', '2022-03-18', '2022-03-22', '2022-03-26',
    '2022-03-28', '2022-03-29', '2022-05-05', '2022-05-10',
    '2022-05-16', '2022-05-17', '2022-05-18', '2022-05-20', 
    '2022-05-21', '2022-05-31', 
    '2022-06-05', '2022-06-07', '2022-06-08', '2022-06-10',
    '2022-06-11', '2022-06-13', '2022-06-14', '2022-06-16',
    '2022-06-17', '2022-06-18'
]
for date in dates_clouds:
    datestr = date.replace('-', '')
    file_paths = sorted(
        glob.glob(
            f'/home/disk/eos4/kathem24/activate/data/HSRL2/2022 HSRL-2/Cloud/ACTIVATE-HSRL2-Cloud_KingAir_{datestr}_R*.h5'
        ),
        reverse=False
    )
    print(f"Processing {date}... Found files: {file_paths}")
    run = 1
    dfs_for_date = []
    for file_path in file_paths:
        try:
            with h5py.File(file_path, 'r') as f:
                df_cloud = pd.DataFrame({
                    'time': f['time'][:],
                    'lat': f['lat'][:],
                    'lon': f['lon'][:],
                    'alt': f['alt'][:],
                    'cloud_height': f['DataProducts/cloud_height'][:],
                    'cloud_ext_average': f['DataProducts/cloud_ext_average'][:],
                    'WindSpeedDerivedCM': f['DataProducts/WindSpeedDerivedCM'][:],
                    'WindSpeedDerivedHU': f['DataProducts/WindSpeedDerivedHU'][:],
                    'MultScatFrac': f['DataProducts/MultScatFrac'][:],
                    'Sc': f['DataProducts/Sc'][:]
                })
            df_cloud.replace(
                [-9999, -9999.0, -999, -999.0, -7777, -8888],
                np.NaN,
                inplace=True
            )
            for col in df_cloud.columns:
                df_cloud[col] = pd.to_numeric(df_cloud[col], errors='coerce')
            print(f"Columns for {file_path}: {df_cloud.columns[:10]}")
            print(f"Shape for {file_path}: {df_cloud.shape}")
            dfs_for_date.append(df_cloud)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    if len(dfs_for_date) == 2:
        df4, df5 = dfs_for_date[0], dfs_for_date[1]
        combined_df = pd.concat([df4, df5], ignore_index=True)
        clouds.append(combined_df)
        successful_dates_clouds.append(date)
        print(f"Combined Cloud DataFrame for {date} (first 5 rows):")
        print(combined_df.head())
    elif len(dfs_for_date) == 1:
        clouds.append(dfs_for_date[0])
        successful_dates_clouds.append(date)
        print(f"Single file Cloud DataFrame for {date} (first 5 rows):")
        print(dfs_for_date[0].head())
    else:
        print(f"No valid data for {date}")
print(f"Total dates requested: {len(dates_clouds)}")
print(f"Total dates processed: {len(clouds)}")
print("Successful cloud dates:")
print(successful_dates_clouds)
# %%
print(len(dates_clouds))
print(len(clouds))
missing_cloud_dates = [d for d in dates_clouds if d not in successful_dates_clouds]
print("Missing cloud dates:")
print(missing_cloud_dates)
for i in [0, 10, 20, 30, len(clouds)-1]:
    print(i, successful_dates_clouds[i], len(clouds[i]))
for date, df in zip(successful_dates_clouds, clouds):
    print(date)
    print("first time:", df['time'].iloc[0])
    print("last time:", df['time'].iloc[-1])
    print("time monotonic:", df['time'].is_monotonic_increasing)
    print("cloud height valid:", df['cloud_height'].count())
    print()
# %%
idx = successful_dates_clouds.index('2022-01-11')
df = clouds[idx]
diffs = df['time'].diff()
bad_jumps = diffs[diffs < 0]
print(bad_jumps)
print("Number of backward jumps:", len(bad_jumps))
for jump_index in bad_jumps.index:
    print("Backward jump at index:", jump_index)
    print(df.loc[jump_index-3:jump_index+3, ['time', 'cloud_height']])
# %%
idx = successful_dates_clouds.index('2022-01-11')
df = clouds[idx]
print("NaN times:", df['time'].isna().sum())
print("Duplicate times:", df['time'].duplicated().sum())
dupes = df[df['time'].duplicated(keep=False)]
print(dupes[['time', 'cloud_height']].head(20))
df_sorted = df.sort_values('time').reset_index(drop=True)
print("Sorted monotonic:", df_sorted['time'].is_monotonic_increasing)
print("Backward jumps after sorting:", (df_sorted['time'].diff() < 0).sum())
# %%
#looking at cloud top height for all flights
all_cloud_heights = []
for df in clouds:
    vals = df['cloud_height'].dropna().values
    all_cloud_heights.extend(vals)
all_cloud_heights = np.array(all_cloud_heights)
print("Mean:", np.mean(all_cloud_heights))
print("Median:", np.median(all_cloud_heights))
print("10th percentile:", np.percentile(all_cloud_heights,10))
print("90th percentile:", np.percentile(all_cloud_heights,90))
print("Min:", np.min(all_cloud_heights))
print("Max:", np.max(all_cloud_heights))
plt.figure(figsize=(8,5))
plt.hist(all_cloud_heights, bins=50)
plt.xlabel('Cloud Top Height (m)', fontsize=14, fontweight='bold')
plt.ylabel('Number of HSRL-2 Observations',fontsize=14,fontweight='bold')
plt.title('Distribution of HSRL-2 Cloud Top Heights\nJanuary-June 2022',fontsize=14,
          fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.show()
# %%
#by flight, looking at cloud top height for all flights
flight_mean_cloud_height = []
for df in clouds:
    flight_mean_cloud_height.append(
        np.nanmean(df['cloud_height'])
    )
plt.figure(figsize=(12,5))
plt.plot(successful_dates_clouds, flight_mean_cloud_height,marker='o')
plt.xticks(rotation=90,fontsize=10, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.ylabel('Mean Cloud Top Height (m)', fontsize=14, fontweight='bold')
plt.title('Flight Mean Cloud Top Height', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
print("Mean:", np.mean(all_cloud_heights))
print("Median:", np.median(all_cloud_heights))
print("10th:", np.percentile(all_cloud_heights,10))
print("90th:", np.percentile(all_cloud_heights,90))
# %%
for date, df in zip(successful_dates_clouds, clouds):
    pct_valid = (
        df['cloud_height'].count()
        / len(df)
        * 100
    )
    print(date,
          round(pct_valid,1))
# %%
cloud_coverage = []
for df in clouds:
    pct_valid = 100 * df['cloud_height'].count() / len(df)
    cloud_coverage.append(pct_valid)
plt.figure(figsize=(12,5))
plt.plot(successful_dates_clouds, cloud_coverage, marker='o', linewidth=2)
plt.xticks(rotation=90, fontsize=10, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.ylabel('Valid Cloud-Top Retrievals (%)', fontsize=14,fontweight='bold')
plt.xlabel('Flight Date', fontsize=14, fontweight='bold')
plt.title('HSRL-2 Cloud-Top Retrieval Coverage by Flight\nJanuary-June 2022', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# %%
with h5py.File(file_path,'r') as f:
    vars_to_check = [
        'DataProducts/Reflectance',
        'DataProducts/ReflectanceAvg',
        'DataProducts/SurfaceDepol',
        'DataProducts/WindSpeedDerivedCM',
        'DataProducts/WindSpeedDerivedHU',
        'DataProducts/MultScatFrac',
        'DataProducts/Sc'
    ]
    for var in vars_to_check:
        print("\n================")
        print(var)
        print("shape:", f[var].shape)
        for k,v in f[var].attrs.items():
            print(k,":",v)
# %%
#cloud extinction
all_extinction = []
for df in clouds:
    vals = df['cloud_ext_average'].dropna().values
    all_extinction.extend(vals)
all_extinction = np.array(all_extinction)
print("Mean extinction:", np.mean(all_extinction))
print("Median extinction:", np.median(all_extinction))
print("10th percentile:", np.percentile(all_extinction, 10))
print("90th percentile:", np.percentile(all_extinction, 90))
print("Min:", np.min(all_extinction))
print("Max:", np.max(all_extinction))
plt.figure(figsize=(8,5))
plt.hist(all_extinction, bins=50)
plt.xlabel('Cloud Extinction (km$^{-1}$)', fontsize=14, fontweight='bold')
plt.ylabel('Number of HSRL-2 Observations', fontsize=14, fontweight='bold')
plt.title('Distribution of HSRL-2 Cloud Extinction\nJanuary-June 2022',
          fontsize=14, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.show()

#by flight
flight_mean_extinction = []
flight_median_extinction = []
for df in clouds:
    vals = df['cloud_ext_average'].dropna()
    flight_mean_extinction.append(np.nanmean(vals))
    flight_median_extinction.append(np.nanmedian(vals))
plt.figure(figsize=(12,5))
plt.plot(successful_dates_clouds, flight_mean_extinction,
         marker='o', linewidth=2, label='Mean')
plt.plot(successful_dates_clouds, flight_median_extinction,
         marker='s', linewidth=2, label='Median')
plt.xticks(rotation=90, fontsize=10, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.xlabel('Flight Date', fontsize=14, fontweight='bold')
plt.ylabel('Cloud Extinction (km$^{-1}$)', fontsize=14, fontweight='bold')
plt.title('Flight Mean and Median HSRL-2 Cloud Extinction\nJanuary-June 2022',
          fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
# %%
all_h = []
all_e = []
for df in clouds:
    good = df[['cloud_height', 'cloud_ext_average']].dropna()
    all_h.extend(good['cloud_height'].values)
    all_e.extend(good['cloud_ext_average'].values)
all_h = np.array(all_h)
all_e = np.array(all_e)
plt.figure(figsize=(7,5))
plt.scatter(all_h, all_e, s=3, alpha=0.2)
plt.xlabel('Cloud Top Height (m)', fontsize=14, fontweight='bold')
plt.ylabel('Cloud Extinction (km$^{-1}$)', fontsize=14, fontweight='bold')
plt.title('HSRL-2 Cloud Extinction vs Cloud Top Height\nJanuary-June 2022',
          fontsize=14, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,5))
hb = plt.hexbin(all_h, all_e, gridsize=60,mincnt=1)
plt.colorbar(label='Counts')
plt.xlabel('Cloud Top Height (m)', fontsize=14, fontweight='bold')
plt.ylabel('Cloud Extinction (km$^{-1}$)',fontsize=14, fontweight='bold')
plt.title('HSRL-2 Cloud Extinction vs Cloud Top Height\nJanuary-June 2022',
          fontsize=14, fontweight='bold')
plt.show()
# %%
with h5py.File(file_path,'r') as f:
    vars_to_check = [
        'DataProducts/WindSpeedDerivedCM',
        'DataProducts/WindSpeedDerivedHU',
        'DataProducts/MultScatFrac',
        'DataProducts/Sc'
    ]
    for var in vars_to_check:
        print("\n================")
        print(var)
        print("shape:", f[var].shape)
        for k,v in f[var].attrs.items():
            print(k,":",v)
# %%
variables = [
    ('WindSpeedDerivedCM', 'HSRL-Derived Surface Wind Speed CM (m s$^{-1}$)'),
    ('WindSpeedDerivedHU', 'HSRL-Derived Surface Wind Speed HU (m s$^{-1}$)'),
    ('MultScatFrac', 'Multiple Scattering Fraction'),
    ('Sc', 'Cloud Lidar Ratio (sr)')
]

for var, label in variables:
    all_vals = []

    for df in clouds:
        vals = df[var].dropna().values
        all_vals.extend(vals)

    all_vals = np.array(all_vals)

    print('\n', var)
    print('Mean:', np.nanmean(all_vals))
    print('Median:', np.nanmedian(all_vals))
    print('10th:', np.nanpercentile(all_vals, 10))
    print('90th:', np.nanpercentile(all_vals, 90))
    print('Min:', np.nanmin(all_vals))
    print('Max:', np.nanmax(all_vals))
    plt.figure(figsize=(8,5))
    plt.hist(all_vals, bins=50)
    plt.xlabel(label, fontsize=14, fontweight='bold')
    plt.ylabel('Number of HSRL-2 Observations', fontsize=14, fontweight='bold')
    plt.title(f'Distribution of {var}\nJanuary-June 2022',
              fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()
# %%
plot_pairs = [
    ('cloud_ext_average', 'MultScatFrac',
     'Cloud Extinction (km$^{-1}$)', 'Multiple Scattering Fraction'),
    ('cloud_height', 'MultScatFrac',
     'Cloud Top Height (m)', 'Multiple Scattering Fraction'),
    ('cloud_ext_average', 'Sc',
     'Cloud Extinction (km$^{-1}$)', 'Cloud Lidar Ratio (sr)'),
    ('WindSpeedDerivedCM', 'WindSpeedDerivedHU',
     'CM Surface Wind Speed (m s$^{-1}$)', 'HU Surface Wind Speed (m s$^{-1}$)')
]
for xvar, yvar, xlabel, ylabel in plot_pairs:
    x_all = []
    y_all = []
    for df in clouds:
        good = df[[xvar, yvar]].dropna()
        x_all.extend(good[xvar].values)
        y_all.extend(good[yvar].values)
    plt.figure(figsize=(7,5))
    plt.scatter(x_all, y_all, s=3, alpha=0.2)
    plt.xlabel(xlabel, fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.title(f'{ylabel} vs {xlabel}\nJanuary-June 2022',
              fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()
# %%
