#%%
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import datetime
import pathlib
import statistics
import mputil
import shutil
import glob
import os
import re
import math
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from scipy.optimize import curve_fit
import seaborn as sns
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from scipy.integrate import quad
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D
from collections import Counter
from scipy.spatial import distance
# %%
#import mixed layer height files from HSRL-2
MLH = []
successful_dates = []
dates_MLH = [
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
for date in dates_MLH:
    datestr = date.replace('-', '')
    file_paths = sorted(
        glob.glob(
            f'/home/disk/eos4/kathem24/activate/data/HSRL2/2022 HSRL-2/MLH/ACTIVATE-HSRL2-mlh_KINGAIR_{datestr}_R*.ict'
        ), 
        reverse=False 
    )
    print(f"Processing {date}... Found files: {file_paths}")
    run = 1
    dfs_for_date = []
    for file_path in file_paths:
        header_row = None
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if line.startswith('Time_Start, Time_Stop'):
                    header_row = i
                    print(f"Detected header row for {file_path}: Line {header_row}")
                    print(f"Header content: {line.strip()}")
                    break
        if header_row is None:
            print(f"Error: Could not find header row in file {file_path}")
            continue
        try:
            df_MLH = pd.read_csv(
                file_path, 
                skiprows=header_row, 
                quoting=csv.QUOTE_NONE,
                engine='python'
            )
            df_MLH.columns = df_MLH.columns.str.strip().str.replace('"', '')
            print(f"Columns for {file_path}: {df_MLH.columns[:10]}")
            df_MLH.replace(
                [-9999, -9999.0, -999, -999.0, -7777, -8888],
                np.NaN,
                inplace=True
            )
            for col in df_MLH.select_dtypes(include=['object']).columns:
                df_MLH[col] = df_MLH[col].str.strip('"')
            for col in df_MLH.columns:
                df_MLH[col] = pd.to_numeric(df_MLH[col], errors='coerce')
            dfs_for_date.append(df_MLH)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    if len(dfs_for_date) == 2:
        df4, df5 = dfs_for_date[0], dfs_for_date[1]
        combined_df = pd.concat([df4, df5], ignore_index=True)
        MLH.append(combined_df)
        successful_dates.append(date)
        print(f"Combined MLH DataFrame for {date} (first 5 rows):")
        print(combined_df.head())
    elif len(dfs_for_date) == 1:
        MLH.append(dfs_for_date[0])
        successful_dates.append(date)
        print(f"Single file MLH DataFrame for {date} (first 5 rows):")
        print(dfs_for_date[0].head())
    else:
        print(f"No valid data for {date}")
print(f"Total dates requested: {len(dates_MLH)}")
print(f"Total dates processed: {len(MLH)}")
print("Successful MLH dates:")
print(successful_dates)
for i in [0, 10, 20, 30, 43]:
    print(i, dates_MLH[i], len(MLH[i]))
# %%
#understanding QA flags
# 0= excellent, 1=weak, 2=poor, NAN=missing
all_qa = pd.concat([df['MLH_qa'] for df in MLH])
print(all_qa.value_counts(dropna=False).sort_index())
good = []
weak=[]
bad = []
for df in MLH:
    good.extend(df.loc[df['MLH_qa'] == 0, 'MLH'].dropna())
    weak.extend(df.loc[df['MLH_qa'] == 1, 'MLH'].dropna())
    bad.extend(df.loc[df['MLH_qa'] == 2, 'MLH'].dropna())
print("QA=0 mean:", np.nanmean(good))
print("QA=1 mean:", np.nanmean(weak))
print("QA=2 mean:", np.nanmean(bad))
print("QA=0 min/max:", np.nanmin(good), np.nanmax(good))
print("QA=1 min/max:", np.nanmin(weak), np.nanmax(weak))
print("QA=2 min/max:", np.nanmin(bad), np.nanmax(bad))
#%%
all_good_mlh = np.concatenate([
    df.loc[df['MLH_qa'] == 0, 'MLH'].dropna().values
    for df in MLH
])
print("Mean MLH:", np.mean(all_good_mlh))
print("Median MLH:", np.median(all_good_mlh))
print("10th percentile:", np.percentile(all_good_mlh,10))
print("90th percentile:", np.percentile(all_good_mlh,90))
qa0 = []
qa1 = []
qa2 = []
for df in MLH:
    qa0.extend(df.loc[df['MLH_qa'] == 0, 'MLH'].dropna())
    qa1.extend(df.loc[df['MLH_qa'] == 1, 'MLH'].dropna())
    qa2.extend(df.loc[df['MLH_qa'] == 2, 'MLH'].dropna())
fig, axs = plt.subplots(1, 3, figsize=(15,5), sharey=True)
axs[0].hist(qa0, bins=40)
axs[0].set_title('QA = 0\nGood Detection',
                 fontsize=14, fontweight='bold')
axs[0].set_xlabel('Mixed Layer Height (m)',
                  fontsize=12, fontweight='bold')
axs[0].set_ylabel('Number of 10s HSRL-2 Observations',
                  fontsize=12, fontweight='bold')
axs[1].hist(qa1, bins=40)
axs[1].set_title('QA = 1\nWeak Detection',
                 fontsize=14, fontweight='bold')
axs[1].set_xlabel('Mixed Layer Height (m)',
                  fontsize=12, fontweight='bold')
axs[2].hist(qa2, bins=40)
axs[2].set_title('QA = 2\nPoor Detection',
                 fontsize=14, fontweight='bold')
axs[2].set_xlabel('Mixed Layer Height (m)',
                  fontsize=12, fontweight='bold')
axs[0].set_title(f'QA = 0\nGood Detection\nn={len(qa0):,}')
axs[1].set_title(f'QA = 1\nWeak Detection\nn={len(qa1):,}')
axs[2].set_title(f'QA = 2\nPoor Detection\nn={len(qa2):,}')
for ax in axs:
    ax.tick_params(axis='both', labelsize=12)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
plt.suptitle('Distribution of HSRL-2 Mixed Layer Heights\nACTIVATE 2022',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
# %%
#examining MLH by flight 
flight_mlh_counts = []
for date, df in zip(dates_MLH, MLH):
    good = df[df['MLH_qa'] == 0]
    flight_mlh_counts.append(good['MLH'].count())
plt.figure(figsize=(12,5))
plt.bar(dates_MLH, flight_mlh_counts)
plt.xticks(rotation=90, fontsize=10, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.ylabel('Number of QA=0 MLH Retrievals', fontsize=13, fontweight='bold')
plt.title('HSRL-2 MLH Availability by Flight Date', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

mean_mlh = []
median_mlh = []
for df in MLH:
    vals = df.loc[df['MLH_qa'] == 0, 'MLH'].dropna()
    mean_mlh.append(np.nanmean(vals))
    median_mlh.append(np.nanmedian(vals))
plt.figure(figsize=(12,5))
plt.plot(dates_MLH, mean_mlh, marker='o', label='Mean')
plt.plot(dates_MLH, median_mlh, marker='s', label='Median')
plt.xticks(rotation=90, fontsize=10, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.ylabel('MLH (m)', fontsize=13, fontweight='bold')
plt.title('Mean and Median HSRL-2 MLH \nper flight', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()
# %%
#testing time series of one flight 
date_to_plot = '2022-02-19'
idx = dates_MLH.index(date_to_plot)
df = MLH[idx]
good = df[df['MLH_qa'] == 0]
plt.figure(figsize=(10,4))
plt.scatter(good['Time_Mid']/3600, good['MLH'], s=10)
plt.xlabel('UTC Time (hours)', fontsize=13, fontweight='bold')
plt.ylabel('MLH (m)', fontsize=13, fontweight='bold')
plt.title(f'HSRL-2 MLH During Flight: {date_to_plot}', fontsize=14, fontweight='bold')
plt.show()
# %%
