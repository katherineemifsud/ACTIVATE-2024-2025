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
import h5py
#%%
#import generic labeled files from HSRL-2
dates_Lidar = [
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
Lidar = []
successful_dates_Lidar = []
missing_dates_Lidar = []
base_path = '/home/disk/eos4/kathem24/activate/data/HSRL2/2022 HSRL-2/H5'
bad_values = [-9999, -9999.0, -999, -999.0, -7777, -7777.0, -8888, -8888.0]
static_keys = [
    'z',
    'DataProducts_Altitude',
    'DataProducts_AltBinsize'
]
for date in dates_Lidar:
    datestr = date.replace('-', '')
    file_paths = sorted(
        glob.glob(
            f'{base_path}/ACTIVATE-HSRL2_KingAir_{datestr}_R*.h5'
        ),
        reverse=False
    )
    print(f"\nProcessing {date}... Found files: {file_paths}")
    files_for_date = []
    for file_path in file_paths:
        try:
            file_data = {}
            with h5py.File(file_path, 'r') as f:
                def read_dataset(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        arr = obj[()]
                        if isinstance(arr, bytes):
                            arr = arr.decode('utf-8')
                        if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.number):
                            arr = arr.astype(float)
                            arr[np.isin(arr, bad_values)] = np.nan
                        clean_name = name.replace('/', '_')
                        file_data[clean_name] = arr
                f.visititems(read_dataset)
            file_data['Date'] = date
            file_data['File'] = file_path.split('/')[-1]
            files_for_date.append(file_data)
            print(f"Loaded {file_path.split('/')[-1]} with {len(file_data)} variables")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    if len(files_for_date) == 2:
        combined = {}
        all_keys = set(files_for_date[0].keys()).union(set(files_for_date[1].keys()))
        for key in all_keys:
            v1 = files_for_date[0].get(key)
            v2 = files_for_date[1].get(key)
            if key in static_keys:
                combined[key] = v1
            elif isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
                try:
                    if v1.shape[1:] == v2.shape[1:]:
                        combined[key] = np.concatenate([v1, v2], axis=0)
                    else:
                        combined[key] = v1
                except:
                    combined[key] = v1

            else:
                combined[key] = v1
        combined['Date'] = date
        combined['Files'] = [files_for_date[0]['File'], files_for_date[1]['File']]
        Lidar.append(combined)
        successful_dates_Lidar.append(date)
        print(f"Combined Lidar data for {date}")
    elif len(files_for_date) == 1:
        Lidar.append(files_for_date[0])
        successful_dates_Lidar.append(date)
        print(f"Single file Lidar data for {date}")
    else:
        missing_dates_Lidar.append(date)
        print(f"No valid data for {date}")
print("\nDONE")
print(f"Total dates requested: {len(dates_Lidar)}")
print(f"Total dates processed: {len(Lidar)}")
print("Successful Lidar dates:")
print(successful_dates_Lidar)
print("Missing Lidar dates:")
print(missing_dates_Lidar)
#%%
file_path = '/home/disk/eos4/kathem24/activate/data/HSRL2/2022 HSRL-2/H5/ACTIVATE-HSRL2_KingAir_20220618_R3.h5'
with h5py.File(file_path, 'r') as f:
    def print_structure(name, obj):
        print(name)
    f.visititems(print_structure)
#%%
#examine readme file 
#dont have packages, do this on OLYMPUS directly
#%%
#look at aerosol ID
idx = successful_dates_Lidar.index('2022-01-11')
flight = Lidar[idx]
print(np.unique(flight['DataProducts_Aerosol_ID']))
file_path = '/home/disk/eos4/kathem24/activate/data/HSRL2/2022 HSRL-2/H5/ACTIVATE-HSRL2_KingAir_20220111_R3_L1.h5'
with h5py.File(file_path,'r') as f:
    dset = f['DataProducts/Aerosol_ID']
    print("Shape:", dset.shape)
    print("\nAttributes:")
    for k,v in dset.attrs.items():
        print(k, ":", v)
idx = successful_dates_Lidar.index('2022-01-11')
flight = Lidar[idx]
print(flight['DataProducts_Aerosol_ID'].shape)
#%%
aerosol_labels = {
    1: 'Ice',
    2: 'Dusty Mix',
    3: 'Marine',
    4: 'Urban/Pollution',
    5: 'Smoke',
    6: 'Fresh Smoke',
    7: 'Polluted Marine',
    8: 'Dust',
    9: 'Dry Marine',
    10: 'Untyped Ambiguous 10',
    11: 'Untyped Ambiguous 11'
}
aerosol_fraction_by_flight = []
for date, flight in zip(successful_dates_Lidar, Lidar):
    aerosol = flight['DataProducts_Aerosol_ID']
    valid = aerosol[~np.isnan(aerosol)]
    row = {'Date': date}
    for aerosol_id, label in aerosol_labels.items():
        row[label] = 100 * np.sum(valid == aerosol_id) / len(valid)
    aerosol_fraction_by_flight.append(row)
aerosol_frac_df = pd.DataFrame(aerosol_fraction_by_flight)
print(aerosol_frac_df.head())
# %%
overall_counts = {label: 0 for label in aerosol_labels.values()}
total_valid = 0
for flight in Lidar:
    aerosol = flight['DataProducts_Aerosol_ID']
    valid = aerosol[~np.isnan(aerosol)]
    total_valid += len(valid)
    for aerosol_id, label in aerosol_labels.items():
        overall_counts[label] += np.sum(valid == aerosol_id)
overall_percent = {
    label: 100 * count / total_valid
    for label, count in overall_counts.items()
}
overall_df = pd.DataFrame({
    'Aerosol Type': list(overall_percent.keys()),
    'Percent': list(overall_percent.values())
}).sort_values('Percent', ascending=False)
print(overall_df)
plt.figure(figsize=(10,5))
plt.bar(overall_df['Aerosol Type'], overall_df['Percent'])
plt.xticks(rotation=45, ha='right', fontsize=11, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.ylabel('Percent of Valid Aerosol-ID Pixels(%)', fontsize=12, fontweight='bold')
plt.title('Overall HSRL-2 Aerosol Classification\nJanuary-June 2022', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
# %%
file_path = '/home/disk/eos4/kathem24/activate/data/HSRL2/2022 HSRL-2/H5/ACTIVATE-HSRL2_KingAir_20220111_R3_L1.h5'
with h5py.File(file_path,'r') as f:
    dset = f['DataProducts/Aerosol_ID']
    for k,v in dset.attrs.items():
        print(k)
        print(v)
        print()
# %%
plot_cols = [
    'Marine',
    'Polluted Marine',
    'Dry Marine',
    'Urban/Pollution',
    'Dusty Mix',
    'Smoke',
    'Fresh Smoke',
    'Dust',
    'Ice',
    'Untyped Ambiguous 10',
    'Untyped Ambiguous 11'
]
plot_df = aerosol_frac_df.set_index('Date')[plot_cols]
plt.figure(figsize=(16,6))
bottom = np.zeros(len(plot_df))
for col in plot_cols:
    plt.bar(plot_df.index, plot_df[col], bottom=bottom, label=col)
    bottom += plot_df[col].values
plt.xticks(rotation=90, fontsize=9, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.ylabel('Percent of Valid Aerosol-ID Pixels(%)', fontsize=13, fontweight='bold')
plt.title('HSRL-2 Aerosol Classification\nJanuary-June 2022', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
plt.tight_layout()
plt.show()
# %%
for aerosol_type in ['Marine', 'Dry Marine', 'Polluted Marine']:
    top_df = aerosol_frac_df[['Date', aerosol_type]].sort_values(
        aerosol_type, ascending=False
    ).head(10)
    print(f"\nTop 10 {aerosol_type} flights:")
    print(top_df)
    plt.figure(figsize=(8,5))
    plt.bar(top_df['Date'], top_df[aerosol_type])
    plt.xticks(rotation=45, ha='right', fontsize=10, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.ylabel(f'{aerosol_type} Pixels (%)', fontsize=13, fontweight='bold')
    plt.title(f'Top 10 Flights by {aerosol_type} Classification', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
# %%
aerosol_frac_df['Date'] = pd.to_datetime(aerosol_frac_df['Date'])
aerosol_frac_df['Month'] = aerosol_frac_df['Date'].dt.strftime('%b')
monthly_df = aerosol_frac_df.groupby('Month').mean(numeric_only=True)
month_order = ['Jan', 'Feb', 'Mar', 'Apr''May', 'Jun']
monthly_df = monthly_df.reindex(
    [m for m in month_order if m in monthly_df.index]
)
print(monthly_df)
plot_cols = [
    'Marine',
    'Polluted Marine',
    'Dry Marine',
    'Urban/Pollution',
    'Dusty Mix',
    'Smoke',
    'Fresh Smoke',
    'Dust',
    'Ice'
]
plt.figure(figsize=(10,6))
bottom = np.zeros(len(monthly_df))
for col in plot_cols:
    plt.bar(monthly_df.index,
            monthly_df[col],
            bottom=bottom,
            label=col)
    bottom += monthly_df[col].values
plt.ylabel('Mean Fraction of Aerosol-ID Pixels (%)',
           fontsize=13,
           fontweight='bold')
plt.xlabel('Month',
           fontsize=13,
           fontweight='bold')
plt.title('Monthly Aerosol Classification',
          fontsize=14,
          fontweight='bold')
plt.legend(bbox_to_anchor=(1.02,1),
           loc='upper left')
plt.tight_layout()
plt.show()
# %%
plt.figure(figsize=(12,5))
plt.plot(
    aerosol_frac_df['Date'],
    aerosol_frac_df['Marine'],
    marker='o',
    label='Marine'
)
plt.plot(
    aerosol_frac_df['Date'],
    aerosol_frac_df['Polluted Marine'],
    marker='o',
    label='Polluted Marine'
)
plt.plot(
    aerosol_frac_df['Date'],
    aerosol_frac_df['Dry Marine'],
    marker='o',
    label='Dry Marine'
)
plt.ylabel('Fraction of Aerosol-ID Pixels (%)',
           fontsize=13,
           fontweight='bold')
plt.xlabel('Date',
           fontsize=13,
           fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()
# %%
#looking at depolarization 
flight = Lidar[0]
dep = flight['DataProducts_532_dep']
print(dep.shape)
print('Min:', np.nanmin(dep))
print('Max:', np.nanmax(dep))
print('Mean:', np.nanmean(dep))

print('Percentiles:')
print(np.nanpercentile(dep,[1,5,25,50,75,95,99]))
# %%
all_dep = []

for flight in Lidar:
    dep = flight['DataProducts_532_dep']
    dep = dep.flatten()
    dep = dep[~np.isnan(dep)]
    dep = dep[dep >= 0]
    all_dep.append(dep)
all_dep = np.concatenate(all_dep)
plt.figure(figsize=(8,5))

plt.hist(
    all_dep,
    bins=np.arange(0,0.25,0.002)
)
plt.xlabel('532 nm Depolarization', fontsize=13, fontweight='bold')
plt.ylabel('Pixel Count', fontsize=13, fontweight='bold')
#set y axis to logscale
plt.yscale('log')
plt.title('January-June 2022 Depolarization Distribution', fontsize=14, fontweight='bold')
plt.show()
# %%
all_dep = []
all_id = []
for flight in Lidar:
    dep = flight['DataProducts_532_dep']
    aid = flight['DataProducts_Aerosol_ID']
    mask = (
        ~np.isnan(dep)
        &
        ~np.isnan(aid)
        &
        (dep >= 0)
    )
    all_dep.append(dep[mask])
    all_id.append(aid[mask])
all_dep = np.concatenate(all_dep)
all_id = np.concatenate(all_id)
class_stats = []

for aerosol_id, label in aerosol_labels.items():
    vals = all_dep[all_id == aerosol_id]
    class_stats.append({
        'ID': aerosol_id,
        'Type': label,
        'Count': len(vals),
        'Mean': np.nanmean(vals),
        'Median': np.nanmedian(vals),
        'P95': np.nanpercentile(vals,95)
    })
class_df = pd.DataFrame(class_stats)
print(class_df.sort_values('Mean', ascending=False))
plt.figure(figsize=(10,5))
plot_df = class_df.sort_values('Mean', ascending=False)
plt.bar(
    plot_df['Type'],
    plot_df['Mean']
)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Mean 532 nm Depolarization')
plt.title('Mean Depolarization by Aerosol Type')
plt.tight_layout()
plt.show()
# %%
box_data = []
labels = []
for aerosol_id, label in aerosol_labels.items():
    vals = all_dep[all_id == aerosol_id]
    vals = vals[np.isfinite(vals)]
    if len(vals) > 100:
        box_data.append(vals)
        labels.append(label)
plt.figure(figsize=(12,6))
plt.boxplot(
    box_data,
    labels=labels,
    showfliers=False
)
plt.yticks(fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10, fontweight='bold')
plt.ylabel('532 nm Depolarization', fontsize=13, fontweight='bold')
plt.title('Depolarization Distribution by Aerosol Type', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
# %%
table_df = class_df.copy()

table_df['Count'] = table_df['Count'].astype(int)
table_df['Mean'] = table_df['Mean'].round(3)
table_df['Median'] = table_df['Median'].round(3)
table_df['P95'] = table_df['P95'].round(3)
table_df = table_df.sort_values(
    'Median',
    ascending=False
)
print(table_df)
# %%
fig, ax = plt.subplots(
    figsize=(10,5)
)
ax.axis('off')
table = ax.table(
    cellText=table_df.values,
    colLabels=table_df.columns,
    cellLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.5)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold')
        cell.set_fontsize(12)
plt.title(
    '532 nm Depolarization Statistics by Aerosol Classification',
    fontsize=14,
    fontweight='bold',
    pad=20
)
plt.tight_layout()
plt.savefig(
    'Depol_AerosolID_Table.png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()
# %%
table_df = class_df[['Type','Median','P95']].copy()
table_df['Median'] = table_df['Median'].round(3)
table_df['P95'] = table_df['P95'].round(3)
table_df = table_df.sort_values(
    'Median',
    ascending=False
)
# %%
#Angstrom
flight = Lidar[0]
ang = flight['DataProducts_Angstrom_532_355']
print(ang.shape)
print("Min:", np.nanmin(ang))
print("Max:", np.nanmax(ang))
print("Mean:", np.nanmean(ang))
print("Percentiles:")
print(np.nanpercentile(ang,[1,5,25,50,75,95,99]))
# %%
ang = flight['DataProducts_Angstrom_532_355']
valid = ang[~np.isnan(ang)]
print("Fraction between 0 and 3:",
      np.sum((valid >= 0) & (valid <= 3))/len(valid))
print("Fraction between -1 and 4:",
      np.sum((valid >= -1) & (valid <= 4))/len(valid))
# %%
file_path = '/home/disk/eos4/kathem24/activate/data/HSRL2/2022 HSRL-2/H5/ACTIVATE-HSRL2_KingAir_20220111_R3_L1.h5'
with h5py.File(file_path,'r') as f:
    dset = f['DataProducts/Angstrom_532_355']
    for k,v in dset.attrs.items():
        print(k)
        print(v)
        print()
# %%
bsc = flight['DataProducts_532_bsc']
print(bsc.shape)
print("Min:", np.nanmin(bsc))
print("Max:", np.nanmax(bsc))
print("Mean:", np.nanmean(bsc))
print(np.nanpercentile(
    bsc,
    [1,5,25,50,75,95,99]
))
# %%
bsc = flight['DataProducts_532_bsc']
valid = bsc[~np.isnan(bsc)]
print("% positive:",
      100*np.sum(valid > 0)/len(valid))
print("% > 0.001:",
      100*np.sum(valid > 0.001)/len(valid))
print("% > 0.01:",
      100*np.sum(valid > 0.01)/len(valid))
# %%
summary = []
all_dep = []
all_bsc = []
marine_ids = {
    3: 'Marine',
    7: 'Polluted Marine',
    9: 'Dry Marine'
}
all_ang = []
all_id = []
for flight in Lidar:
    dep = flight['DataProducts_532_dep']
    bsc = flight['DataProducts_532_bsc']
    ang = flight['DataProducts_Angstrom_532_355']
    aid = flight['DataProducts_Aerosol_ID']
    mask = (
        ~np.isnan(dep)
        &
        ~np.isnan(bsc)
        &
        ~np.isnan(ang)
        &
        ~np.isnan(aid)
    )
    all_dep.append(dep[mask])
    all_bsc.append(bsc[mask])
    all_ang.append(ang[mask])
    all_id.append(aid[mask])
all_dep = np.concatenate(all_dep)
all_bsc = np.concatenate(all_bsc)
all_ang = np.concatenate(all_ang)
all_id = np.concatenate(all_id)
for aerosol_id, label in marine_ids.items():
    mask = (all_id == aerosol_id)
    dep_vals = all_dep[mask]
    bsc_vals = all_bsc[mask]
    ang_vals = all_ang[mask]
    summary.append({
        'Type': label,
        'Count': len(dep_vals),
        'Depol_Median': np.nanmedian(dep_vals),
        'Depol_P95': np.nanpercentile(dep_vals,95),
        'BSC_Median': np.nanmedian(bsc_vals),
        'BSC_P95': np.nanpercentile(bsc_vals,95),
        'Ang_Median': np.nanmedian(ang_vals),
        'Ang_P95': np.nanpercentile(ang_vals,95)
    })
marine_summary_df = pd.DataFrame(summary)
print(marine_summary_df)
# %%
marine_summary_df = marine_summary_df.round({
    'Depol_Median':3,
    'Depol_P95':3,
    'BSC_Median':5,
    'BSC_P95':5,
    'Ang_Mean':2,
    'Ang_P95':2
})
print(marine_summary_df)
fig, ax = plt.subplots(figsize=(12,3))

ax.axis('off')
table = ax.table(
    cellText=marine_summary_df.values,
    colLabels=marine_summary_df.columns,
    loc='center',
    cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.3,1.6)
for (row,col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold')
plt.title(
    'Marine Aerosol Environment Comparison',
    fontsize=14,
    fontweight='bold',
    pad=20
)
plt.tight_layout()
plt.show()
# %%
plt.figure(figsize=(8,5))
plot_df = marine_summary_df.set_index('Type')
plt.bar(
    plot_df.index,
    plot_df['Depol_Median']
)
plt.xticks(rotation=45, ha='right', fontsize=10, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.ylabel('Median Depolarization', fontsize=13, fontweight='bold')
plt.title('Marine Aerosol Classes', fontsize=14, fontweight='bold')
plt.show()
# %%
plt.figure(figsize=(8,5))
plot_df = marine_summary_df.set_index('Type')
plt.bar(
    plot_df.index,
    plot_df['Ang_Median']
)
plt.xticks(rotation=45, ha='right', fontsize=10, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.ylabel('Median Angstrom Exponent', fontsize=13, fontweight='bold')
plt.title('Marine Aerosol Classes', fontsize=14, fontweight='bold')
plt.show()
# %%
plt.figure(figsize=(8,5))
plot_df = marine_summary_df.set_index('Type')
plt.bar(
    plot_df.index,
    plot_df['BSC_Median']
)
plt.xticks(rotation=45, ha='right', fontsize=10, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.ylabel('Median Backscatter Coefficient', fontsize=13, fontweight='bold')
plt.title('Marine Aerosol Classes', fontsize=14, fontweight='bold')
plt.show()
# %%
box_data = []
labels = []

for aerosol_id, label in aerosol_labels.items():
    vals = all_ang[all_id == aerosol_id]
    vals = vals[np.isfinite(vals)]
    if len(vals) > 100:
        box_data.append(vals)
        labels.append(label)
plt.figure(figsize=(12,6))
plt.boxplot(
    box_data,
    labels=labels,
    showfliers=False
)
plt.yticks(fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.ylabel('532-355 Ångström Exponent',
           fontsize=13,
           fontweight='bold')
plt.title('Ångström Exponent Distribution by Aerosol Type',
          fontsize=14,
          fontweight='bold')
plt.tight_layout()
plt.show()
# %%
marine_ids = {
    3: 'Marine',
    7: 'Polluted Marine',
    9: 'Dry Marine'
}
box_data = []
labels = []
for aerosol_id, label in marine_ids.items():
    vals = all_ang[all_id == aerosol_id]
    vals = vals[np.isfinite(vals)]
    vals = vals[(vals > -2) & (vals < 6)]
    box_data.append(vals)
    labels.append(label)
plt.figure(figsize=(8,6))
plt.boxplot(
    box_data,
    labels=labels,
    showfliers=False
)
plt.ylabel('532-355 Ångström Exponent',
           fontsize=13,
           fontweight='bold')
plt.title('Marine Aerosol Ångström Exponent',
          fontsize=14,
          fontweight='bold')
plt.tight_layout()
plt.show()
# %%
