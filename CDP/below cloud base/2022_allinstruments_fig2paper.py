#%%
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import datetime
import pathlib
import statistics
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy.ma as ma
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
import pickle
from scipy.integrate import quad
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D
from collections import Counter
from scipy.spatial import distance
#%%
## 2D-S Data Import for total checking number concentration to remove cloudy data from our
#clear sky analysis
bin_name = [
    'dNdlogD_total_003_2DS', 'dNdlogD_total_004_2DS', 
    'dNdlogD_total_005_2DS', 'dNdlogD_total_006_2DS'
]
twoDS = []
dates_twoDS = [
    '2022-01-11', '2022-01-12', '2022-01-15', '2022-01-18', 
    '2022-01-19', '2022-01-24', '2022-01-26', '2022-01-27',
    '2022-02-01', '2022-02-02', '2022-02-03', '2022-02-05', 
    '2022-02-15', '2022-02-16', '2022-02-19', '2022-02-22',
    '2022-02-26', '2022-03-03', '2022-03-04', '2022-03-13', 
    '2022-03-14', '2022-03-18', '2022-03-22', '2022-03-26',
    '2022-03-28', '2022-03-29', '2022-05-05', '2022-05-10',
    '2022-05-16', '2022-05-17', '2022-05-18', '2022-05-20', 
    '2022-05-21', '2022-05-31', '2022-06-02', '2022-06-03', 
    '2022-06-05', '2022-06-07', '2022-06-08', '2022-06-10',
    '2022-06-11', '2022-06-13', '2022-06-14', '2022-06-17', 
    '2022-06-18'
]

for date in dates_twoDS:
    datestr = date.replace('-', '')
    file_paths = sorted(
        glob.glob(f'/home/disk/eos4/kathem24/activate/data/twoDspectrometer/horizontal/csv/ACTIVATE-2DS-H-Arm_HU25_{datestr}_R*.csv'), 
        reverse=False 
    )

    print(f"Processing {date}... Found files: {file_paths}")

    run = 1
    dfs_for_date = []

    for file_path in file_paths:
        header_row = None
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if 'Time_Start' in line and 'LWC_2DS' in line:
                    header_row = i
                    print(f"Detected header row for {file_path}: Line {header_row}")
                    print(f"Header content: {line.strip()}")
                    break

        if header_row is None:
            print(f"Error: Could not find header row in file {file_path}")
            continue

        try:
            df_2DS = pd.read_csv(
                file_path, 
                skiprows=header_row, 
                quoting=csv.QUOTE_NONE,
                engine='python'
            )

            df_2DS.columns = df_2DS.columns.str.strip('"')
            print(f"Columns for {file_path}: {df_2DS.columns[:10]}")

            df_2DS.replace([-9999, -9999.0], 0, inplace=True)
            for col in df_2DS.select_dtypes(include=['object']).columns:
                df_2DS[col] = df_2DS[col].str.strip('"')

            dfs_for_date.append(df_2DS)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    if len(dfs_for_date) == 2:
        df4, df5 = dfs_for_date[0], dfs_for_date[1]
        combined_df = pd.concat([df4, df5], ignore_index=True)
        twoDS.append(combined_df)
        print(f"Combined DataFrame for {date} (first 5 rows):")
        print(combined_df.head())
    elif len(dfs_for_date) == 1:
        twoDS.append(dfs_for_date[0])
        print(f"Single file DataFrame for {date} (first 5 rows):")
        print(dfs_for_date[0].head())
    else:
        print(f"No valid data for {date}")
print(f"Total dates processed: {len(twoDS)}")
# %%
#Import humidity data. 
col_name_h20 = ['Time_Start', 'H2O_DLH', 'RHi_DLH', 'RHw_DLH']
h20=[]
dates_h20 = [
    '2022-01-11', '2022-01-12', '2022-01-15', '2022-01-18', 
    '2022-01-19', '2022-01-24', '2022-01-26', '2022-01-27',
    '2022-02-01', '2022-02-02', '2022-02-03', '2022-02-05', 
    '2022-02-15', '2022-02-16', '2022-02-19', '2022-02-22',
    '2022-02-26', '2022-03-03', '2022-03-04', '2022-03-13', 
    '2022-03-14', '2022-03-18', '2022-03-22', '2022-03-26',
    '2022-03-28', '2022-03-29', '2022-05-05', '2022-05-10',
    '2022-05-16', '2022-05-17', '2022-05-18', '2022-05-20', 
    '2022-05-21', '2022-05-31', '2022-06-02', '2022-06-03', 
    '2022-06-05', '2022-06-07', '2022-06-08', '2022-06-10',
    '2022-06-11', '2022-06-13', '2022-06-14', '2022-06-17', 
    '2022-06-18'
]
for date in dates_h20:
    datestr = date.replace('-', '')
    fname_h20 = sorted(glob.glob(f'/home/disk/eos4/kathem24/activate/data/DLH_H20/csv/ACTIVATE-DLH-H2O_HU25_{datestr}_R*.csv'))
    frames =[]
    for file_path in fname_h20:
        df_h20 = pd.read_csv(file_path, skiprows=36, quoting=csv.QUOTE_NONE)
        df_h20.columns = df_h20.columns.str.strip().str.replace('"', '')
        for col_ in col_name_h20:
            if col_ in df_h20.columns:
                df_h20[col_] = df_h20[col_].astype(str).str.strip().str.replace('"', '')
                df_h20[col_] = pd.to_numeric(df_h20[col_], errors='coerce')
                df_h20.replace([-9999, -9999.00], np.NaN, inplace=True)
        frames.append(df_h20)
    if len(frames) > 1:
        df_h20_combined = pd.concat(frames, ignore_index=True)

    else:
        df_h20_combined = frames[0]
    h20.append(df_h20_combined)
#%%
col_name = ['Time_mid', 'Latitude', 'Longitude', 'GPS_altitude', 'Pressure_Altitude',
             'Pitch', 'Roll', 'True_Heading', 'True_Air_Speed', 
             'Static_Air_Temp', 'IR_Surf_Temp', 'Static_Pressure',
             'Wind_Speed']
summary=[]
dates_sum = [
    '2022-01-11', '2022-01-12', '2022-01-15', '2022-01-18', 
    '2022-01-19', '2022-01-24', '2022-01-26', '2022-01-27',
    '2022-02-01', '2022-02-02', '2022-02-03', '2022-02-05', 
    '2022-02-15', '2022-02-16', '2022-02-19', '2022-02-22',
    '2022-02-26', '2022-03-03', '2022-03-04', '2022-03-13', 
    '2022-03-14', '2022-03-18', '2022-03-22', '2022-03-26',
    '2022-03-28', '2022-03-29', '2022-05-05', '2022-05-10',
    '2022-05-16', '2022-05-17', '2022-05-18', '2022-05-20', 
    '2022-05-21', '2022-05-31', '2022-06-02', '2022-06-03', 
    '2022-06-05', '2022-06-07', '2022-06-08', '2022-06-10',
    '2022-06-11', '2022-06-13', '2022-06-14', '2022-06-17', 
    '2022-06-18'
]

for date in dates_sum:
    datestr = date.replace('-', '')
    fname_sum = sorted(glob.glob(f'/home/disk/eos4/kathem24/activate/data/MET/2022/Summary/csv/ACTIVATE-SUMMARY_HU25_{datestr}_R*.csv'), reverse=True)
    run = 1
    for file_path in fname_sum: 
        num_file_paths = len(fname_sum)
        if date > '2022-01-12':
            df_sum = pd.read_csv(file_path, skiprows=47, quoting=csv.QUOTE_NONE)
        elif date == '2022-01-11':
            df_sum = pd.read_csv(file_path, skiprows=49, quoting=csv.QUOTE_NONE)
        elif date == '2022-01-12':
            df_sum = pd.read_csv(file_path, skiprows=48, quoting=csv.QUOTE_NONE)
        for col_ in col_name:
            if col_ in df_sum.columns:
                df_sum.columns = df_sum.columns.str.strip('"')
                df_sum[col_] = pd.to_numeric(df_sum[col_], errors='coerce')
                df_sum.replace([-9999, -9999.00], np.NaN, inplace=True)
        for col in ['Time_mid', 'Latitude', 'Longitude', 'GPS_altitude', 'Pressure_Altitude',
             'Pitch', 'Roll', 'True_Heading', 'True_Air_Speed', 
             'Static_Air_Temp', 'IR_Surf_Temp', 'Static_Pressure',
             'Wind_Speed']:
            if df_sum[col].dtype == 'O': 
                df_sum[col] = df_sum[col].str.strip('"')
        if num_file_paths==2:
            if run==1:
                df1 = df_sum 
            elif run==2:
                df2 = df_sum 
                frames = [df2,df1]
                df_sum = pd.concat(frames)
                summary.append(df_sum)
                break
        if num_file_paths ==1:
            summary.append(df_sum)
        run = run+1      
#%%
leg_data = []
leg_name=['Time_Start', '  Time_Stop', '  Julian_Day', 
          '  Date', '  LegIndex']
dates_legs= [
    '2022-01-11', '2022-01-12', '2022-01-15', '2022-01-18', 
    '2022-01-19', '2022-01-24', '2022-01-26', '2022-01-27',
    '2022-02-01', '2022-02-02', '2022-02-03', '2022-02-05', 
    '2022-02-15', '2022-02-16', '2022-02-19', '2022-02-22',
    '2022-02-26', '2022-03-03', '2022-03-04', '2022-03-13', 
    '2022-03-14', '2022-03-18', '2022-03-22', '2022-03-26',
    '2022-03-28', '2022-03-29', '2022-05-05', '2022-05-10',
    '2022-05-16', '2022-05-17', '2022-05-18', '2022-05-20', 
    '2022-05-21', '2022-05-31', '2022-06-02', '2022-06-03', 
    '2022-06-05', '2022-06-07', '2022-06-08', '2022-06-10',
    '2022-06-11', '2022-06-13', '2022-06-14', '2022-06-17', 
    '2022-06-18'
]
for date in dates_legs:
    datestr = date.replace('-', '')
    fname_legs = sorted(glob.glob(f'/home/disk/eos4/kathem24/activate/data/MET/2022/LegFLags/csv/ACTIVATE-LegFlags_HU25_{datestr}_R*.csv'), reverse=True)
    leg_dictionary = {
        'Date': date,
        'LegIndex_02': {'StartTimes': [], 'StopTimes': []},
        'LegIndex_06': {'StartTimes': [], 'StopTimes': []}
    }

    for file_path in fname_legs:
        if date <= '2022-01-19'or date == '2022-02-05':
            df_legs = pd.read_csv(file_path, skiprows=44, quoting=csv.QUOTE_NONE)
        elif date == '2022-01-24':
            df_legs = pd.read_csv(file_path, skiprows=45, quoting=csv.QUOTE_NONE)
        elif date > '2022-01-24' and date < '2022-02-02':
            df_legs = pd.read_csv(file_path, skiprows=44, quoting=csv.QUOTE_NONE)
        elif date >='2022-02-02' and date <= '2022-02-15':
            df_legs = pd.read_csv(file_path, skiprows=45, quoting=csv.QUOTE_NONE)
        elif date >= '2022-02-16': 
            df_legs = pd.read_csv(file_path, skiprows=44, quoting=csv.QUOTE_NONE)

        df_legs.columns = df_legs.columns.str.strip('"')

        for col in ['  LegIndex', 'Time_Start', '  Time_Stop']:
            if df_legs[col].dtype == 'O':  # 'O' stands for Object (usually string columns)
                df_legs[col] = df_legs[col].str.strip('"')
  
        df_legs['Time_Start'] = pd.to_numeric(df_legs['Time_Start'], errors='coerce')
        df_legs['  Time_Stop'] = pd.to_numeric(df_legs['  Time_Stop'], errors='coerce')
        df_legs['  LegIndex'] = pd.to_numeric(df_legs['  LegIndex'], errors='coerce')
 
        for leg_ in leg_name:
            if leg_ in df_legs.columns:
                df_legs.replace([-9999, -9999.00], np.NaN, inplace=True)
                df_legs.dropna(subset=['Time_Start', '  Time_Stop', '  LegIndex'], inplace=True)
     
        leg_index_02 = df_legs[df_legs['  LegIndex'] % 100 == 2]
        leg_index_06 = df_legs[df_legs['  LegIndex'] % 100 == 6]
        leg_dictionary['LegIndex_02']['StartTimes'].extend(leg_index_02['Time_Start'].tolist())
        leg_dictionary['LegIndex_02']['StopTimes'].extend(leg_index_02['  Time_Stop'].tolist())
        leg_dictionary['LegIndex_06']['StartTimes'].extend(leg_index_06['Time_Start'].tolist())
        leg_dictionary['LegIndex_06']['StopTimes'].extend(leg_index_06['  Time_Stop'].tolist())
    leg_data.append(leg_dictionary)
#%%
#Now CDP data
L_00c=math.log10(3)-math.log10(2)
L_01c=math.log10(4)-math.log10(3)
L_02c=math.log10(5)-math.log10(4)
L_03c=math.log10(6)-math.log10(5)
L_04c=math.log10(7)-math.log10(6)
L_05c=math.log10(8)-math.log10(7)
L_06c=math.log10(9)-math.log10(8)
L_07c=math.log10(10)-math.log10(9)
L_08c=math.log10(11)-math.log10(10)
L_09c=math.log10(12)-math.log10(11)
L_10c=math.log10(13)-math.log10(12)
L_11c=math.log10(14)-math.log10(13)
L_12c=math.log10(16)-math.log10(14)
L_13c=math.log10(18)-math.log10(16)
L_14c=math.log10(20)-math.log10(18)
L_15c=math.log10(22)-math.log10(20)
L_16c=math.log10(24)-math.log10(22)
L_17c=math.log10(26)-math.log10(24)
L_18c=math.log10(28)-math.log10(26)
L_19c=math.log10(30)-math.log10(28)
L_20c=math.log10(32)-math.log10(30)
L_21c=math.log10(34)-math.log10(32)
L_22c=math.log10(36)-math.log10(34)
L_23c=math.log10(38)-math.log10(36)
L_24c=math.log10(40)-math.log10(38)
L_25c=math.log10(42)-math.log10(40)
L_26c=math.log10(44)-math.log10(42)
L_27c=math.log10(46)-math.log10(44)
L_28c=math.log10(48)-math.log10(46)
L_29c=math.log10(50)-math.log10(48)



bin_log_CDP=[L_00c, L_01c, L_02c, L_03c, L_04c, L_05c, L_06c, L_07c, L_08c,
          L_09c, L_10c, L_11c,
          L_12c, L_13c, L_14c, L_15c, L_16c, 
        L_17c, L_18c, L_19c, L_20c, L_21c, L_22c, L_23c, 
        L_24c, L_25c, L_26c, L_27c, L_28c, L_29c]


P00c=(3-2)
P01c=(4-3)
P02c=(5-4)
P03c=(6-5)
P04c=(7-6)
P05c=(8-7)
P06c=(9-8)
P07c=(10-9)
P08c=(11-10)
P09c=(12-11)
P10c=(13-12)
P11c=(14-13)
P12c = (16-14)
P13c = (18-16)
P14c = (20-18)
P15c = (22-20)
P16c = (24-22)
P17c = (26-24)
P18c = (28-26)
P19c = (30-28)
P20c = (32-30)
P21c = (34-32)
P22c = (36-34)
P23c = (38-36)
P24c = (40-38)
P25c = (42-40)
P26c = (44-42)
P27c = (46-44)
P28c = (48-46)
P29c = (50-48)


J00c=(L_00c / P00c)
J01c=(L_01c / P01c)
J02c=(L_02c / P02c)
J03c=(L_03c / P03c)
J04c=(L_04c / P04c)
J05c=(L_05c / P05c)
J06c=(L_06c / P06c)
J07c=(L_07c / P07c)
J08c=(L_08c / P08c)
J09c=(L_09c / P09c)
J10c=(L_10c / P10c)
J11c=(L_11c / P11c)
J12c = (L_12c / P12c)
J13c = (L_13c / P13c)
J14c = (L_14c / P14c)
J15c = (L_15c / P15c)
J16c = (L_16c / P16c)
J17c = (L_17c / P17c)
J18c = (L_18c / P18c)
J19c = (L_19c / P19c)
J20c = (L_20c / P20c)
J21c = (L_21c / P21c)
J22c = (L_22c / P22c)
J23c = (L_23c / P23c)
J24c = (L_24c / P24c)
J25c = (L_25c / P25c)
J26c = (L_26c / P26c)
J27c = (L_27c / P27c)
J28c = (L_28c / P28c)
J29c = (L_29c / P29c)


Logg_CDP = [J00c, J01c, J02c, J03c, J04c, J05c, J06c, J07c, J08c, J09c, J10c, 
            J11c, J12c, J13c, J14c, J15c, J16c, J17c, J18c, J19c, J20c, J21c,
            J22c, J23c, J24c, J25c, J26c, J27c, J28c, J29c]

Logg_CDP = np.array(Logg_CDP)
 
bin_center_CDP=[2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 
            10.5, 11.5, 12.5, 13.5, 15, 17, 19, 
            21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49]
#%%
#Import the instrument data for the cloud droplet probe 

#Make sure to work with bins 0-30 for the coarse mode aerosol
bin_name_CDP = ['CDP_Bin00', 'CDP_Bin01', 'CDP_Bin02', 'CDP_Bin03', 
            'CDP_Bin04', 'CDP_Bin05', 'CDP_Bin06', 'CDP_Bin07', 
            'CDP_Bin08', 'CDP_Bin09', 'CDP_Bin11', 'CDP_Bin12',
            'CDP_Bin13', 'CDP_Bin14', 'CDP_Bin15', 'CDP_Bin16', 
            'CDP_Bin17', 'CDP_Bin18', 'CDP_Bin19', 'CDP_Bin20', 
            'CDP_Bin21', 'CDP_Bin22', 'CDP_Bin23', 'CDP_Bin24', 
            'CDP_Bin25', 'CDP_Bin26', 'CDP_Bin27',
            'CDP_Bin28', 'CDP_Bin29']

CDP_1Hz = []

dates_CDP = ['2022-01-11', '2022-01-12','2022-01-15', '2022-01-18', 
             '2022-01-19', '2022-01-24', '2022-01-26', '2022-01-27',
             '2022-02-01', '2022-02-02', '2022-02-03', '2022-02-05', 
             '2022-02-15', '2022-02-16', '2022-02-19', '2022-02-22',
             '2022-02-26', #'2022-03-02',
             '2022-03-03', '2022-03-04', 
             '2022-03-13', '2022-03-14', '2022-03-18', '2022-03-22',
             '2022-03-26', '2022-03-28', '2022-03-29', #'2022-05-03',
             '2022-05-05', '2022-05-10','2022-05-16', '2022-05-17',
             '2022-05-18',
             '2022-05-20','2022-05-21', '2022-05-31', '2022-06-02', 
             '2022-06-03', '2022-06-05','2022-06-07', '2022-06-08', 
             '2022-06-10','2022-06-11','2022-06-13', '2022-06-14',
             '2022-06-17', '2022-06-18']

for date in dates_CDP:
    dataset = {'Date': date, 'Clear Means': [], 'Cloud Means': []}

    file_path = f'/home/disk/eos4/kathem24/activate/data/CDP/2022/csv/CDP_1Hz_files/CDP_1Hz_{date}.csv'
    if not os.path.exists(file_path):
        print(f"File not found for date {date}: {file_path}")
        continue
    df_CDP = pd.read_csv(file_path)

    print(f"First rows for date {date}:")
    print(df_CDP.head())
#%%
for date in dates_CDP:
    file_path = f"/home/disk/eos4/kathem24/activate/data/CDP/2022/csv/CDP_1Hz_files/CDP_1Hz_{date}.csv"

    if not os.path.exists(file_path):
        print(f"File not found for date {date}: {file_path}")
        continue

    df_CDP = pd.read_csv(file_path)
    print(f"Loaded file for {date}:")
    print(df_CDP.head()) 
    CDP_1Hz.append(df_CDP)

#%%
print(f"Number of entries in CDP_1Hz: {len(CDP_1Hz)}")
if CDP_1Hz:
    print("Sample entry in CDP_1Hz:")
    print(CDP_1Hz[0].head())
#%%
for i, df in enumerate(CDP_1Hz):
    print(f"Date {dates_CDP[i]} matches file content with Date column:")
    print(df['Date'].unique())
#%%
#CAS 
C_12=math.log10(2.5)-math.log10(2)
C_13=math.log10(3)-math.log10(2.5)
C_14=math.log10(3.5)-math.log10(3)
C_15=math.log10(4)-math.log10(3.5)
C_16=math.log10(5)-math.log10(4)
C_17=math.log10(6.5)-math.log10(5)
C_18=math.log10(7.2)-math.log10(6.5)
C_19=math.log10(7.9)-math.log10(7.2)
C_20=math.log10(10.2)-math.log10(7.9)
C_21=math.log10(12.5)-math.log10(10.2)
C_22=math.log10(15)-math.log10(12.5)
C_23=math.log10(20)-math.log10(15)
C_24=math.log10(25)-math.log10(20)
C_25=math.log10(30)-math.log10(25)
C_26=math.log10(35)-math.log10(30)
C_27=math.log10(40)-math.log10(35)
C_28=math.log10(45)-math.log10(40)
C_29=math.log10(50)-math.log10(45)


bin_log=[C_12, C_13, C_14, C_15, C_16, 
        C_17, C_18, C_19, C_20, C_21, C_22, C_23, C_24, C_25, C_26, C_27, C_28, C_29]



D12 = (2.5-2)
D13 = (3-2.5)
D14 = (3.5-3)
D15 = (4-3.5)
D16 = (5-4)
D17 = (6.5-5)
D18 = (7.2-6.5)
D19 = (7.9-7.2)
D20 = (10.2-7.9)
D21 = (12.5-10.2)
D22 = (15-12.5)
D23 = (20-15)
D24 = (25-20)
D25 = (30-25)
D26 = (35-30)
D27 = (40-35)
D28 = (45-40)
D29 = (50-45)

F12 = (C_12 / D12)
F13 = (C_13 / D13)
F14 = (C_14 / D14)
F15 = (C_15 / D15)
F16 = (C_16 / D16)
F17 = (C_17 / D17)
F18 = (C_18 / D18)
F19 = (C_19 / D19)
F20 = (C_20 / D20)
F21 = (C_21 / D21)
F22 = (C_22 / D22)
F23 = (C_23 / D23)
F24 = (C_24 / D24)
F25 = (C_25 / D25)
F26 = (C_26 / D26)
F27 = (C_27 / D27)
F28 = (C_28 / D28)
F29 = (C_29 / D29)


Logg = [F12,
        F13, F14, F15, F16, F17, F18, F19, F20, F21, F22, F23, F24, F25,
        F26, F27, F28, F29]

Logg = np.array(Logg)
bin_center=[ 2.25, 2.75, 3.25, 3.75, 4.5, 5.75, 6.85, 7.55, 
            9.05, 11.4, 13.8, 17.5, 22.5, 27.5, 32.5, 
            37.5, 42.5, 47.5]
#%%
#Import the instrument data for the cloud-aerosol spectrometer

bin_name = ['CAS_Bin12' ,'CAS_Bin13', 'CAS_Bin14', 'CAS_Bin15', 
             'CAS_Bin16', 'CAS_Bin17', 
            'CAS_Bin18', 'CAS_Bin19', 'CAS_Bin20', 'CAS_Bin21', 'CAS_Bin22', 
             'CAS_Bin23', 'CAS_Bin24', 'CAS_Bin25', 'CAS_Bin26',
             'CAS_Bin27', 'CAS_Bin28', 'CAS_Bin29']

CAS = []

dates_CAS = [
    '2022-01-11', '2022-01-12', '2022-01-15', '2022-01-18', 
    '2022-01-19', '2022-01-24', '2022-01-26', '2022-01-27',
    '2022-02-01', '2022-02-02', '2022-02-03', '2022-02-05', 
    '2022-02-15', '2022-02-16', '2022-02-19', '2022-02-22',
    '2022-02-26', '2022-03-03', '2022-03-04', '2022-03-13', 
    '2022-03-14', '2022-03-18', '2022-03-22', '2022-03-26',
    '2022-03-28', '2022-03-29', '2022-05-05', '2022-05-10',
    '2022-05-16', '2022-05-17', '2022-05-18', '2022-05-20', 
    '2022-05-21', '2022-05-31', '2022-06-02', '2022-06-03', 
    '2022-06-05', '2022-06-07', '2022-06-08', '2022-06-10',
    '2022-06-11', '2022-06-13', '2022-06-14', '2022-06-17', 
    '2022-06-18'
]

for date in dates_CAS:

    dataset = {'Date': date, 'Clear Means': [], 'Cloud Means': []} 
    datestr = date.replace('-', '')
    fname_CAS = sorted(glob.glob(f'/home/disk/eos4/kathem24/activate/data/cloudaerospect/2022csv/ACTIVATE-LARGE-CAS_HU25_{datestr}_R*.csv'), reverse=True)
    
    run = 1
    for file_path in fname_CAS:
        nums_file_paths = len(fname_CAS)

        if date <= ('2022-03-29'):
            df_CAS = pd.read_csv(file_path, skiprows= 71, quoting=csv.QUOTE_NONE)
        elif date >= ('2022-05-05'):
            df_CAS = pd.read_csv(file_path, skiprows= 72, quoting=csv.QUOTE_NONE)
        
        
        for bin_ in bin_name:
            if bin_ in df_CAS.columns:
                df_CAS.columns = df_CAS.columns.str.strip('"')
                df_CAS[bin_] = pd.to_numeric(df_CAS[bin_], errors='coerce')
                df_CAS.replace([-9999, -9999.00], np.NaN, inplace=True)
        for col in ['Time_mid', 'LWC_CAS','CAS_Bin12', 'CAS_Bin13', 'CAS_Bin14', 
                    'CAS_Bin15', 'CAS_Bin16', 'CAS_Bin17', 
                    'CAS_Bin18', 'CAS_Bin19', 'CAS_Bin20', 
                    'CAS_Bin21', 'CAS_Bin22', 'CAS_Bin23', 
                    'CAS_Bin24', 'CAS_Bin25', 'CAS_Bin26',
                    'CAS_Bin27', 'CAS_Bin28', 'CAS_Bin29']:
            if df_CAS[col].dtype == 'O':  
                df_CAS[col] = df_CAS[col].str.strip('"')
        
        df_CAS['Time_mid']= pd.to_numeric(df_CAS['Time_mid'], errors='coerce')
        df_CAS['CAS_Bin12']= pd.to_numeric(df_CAS['CAS_Bin12'], errors='coerce')
        df_CAS['CAS_Bin13']= pd.to_numeric(df_CAS['CAS_Bin13'], errors='coerce')
        df_CAS['CAS_Bin14']= pd.to_numeric(df_CAS['CAS_Bin14'], errors='coerce')
        df_CAS['CAS_Bin15']= pd.to_numeric(df_CAS['CAS_Bin15'], errors='coerce')
        df_CAS['CAS_Bin16']= pd.to_numeric(df_CAS['CAS_Bin16'], errors='coerce')
        df_CAS['CAS_Bin17']= pd.to_numeric(df_CAS['CAS_Bin17'], errors='coerce')
        df_CAS['CAS_Bin18']= pd.to_numeric(df_CAS['CAS_Bin18'], errors='coerce')
        df_CAS['CAS_Bin19']= pd.to_numeric(df_CAS['CAS_Bin19'], errors='coerce')
        df_CAS['CAS_Bin20']= pd.to_numeric(df_CAS['CAS_Bin20'], errors='coerce')
        df_CAS['CAS_Bin21']= pd.to_numeric(df_CAS['CAS_Bin21'], errors='coerce')
        df_CAS['CAS_Bin22']= pd.to_numeric(df_CAS['CAS_Bin22'], errors='coerce')
        df_CAS['CAS_Bin23']= pd.to_numeric(df_CAS['CAS_Bin23'], errors='coerce')
        df_CAS['CAS_Bin24']= pd.to_numeric(df_CAS['CAS_Bin24'], errors='coerce')
        df_CAS['CAS_Bin25']= pd.to_numeric(df_CAS['CAS_Bin25'], errors='coerce')
        df_CAS['CAS_Bin26']= pd.to_numeric(df_CAS['CAS_Bin26'], errors='coerce')
        df_CAS['CAS_Bin27']= pd.to_numeric(df_CAS['CAS_Bin27'], errors='coerce')
        df_CAS['CAS_Bin28']= pd.to_numeric(df_CAS['CAS_Bin28'], errors='coerce')
        df_CAS['CAS_Bin29']= pd.to_numeric(df_CAS['CAS_Bin29'], errors='coerce')
        df_CAS['LWC_CAS']=pd.to_numeric(df_CAS['LWC_CAS'], errors='coerce')
        

        if nums_file_paths==2:
            if run==1:
                df4 = df_CAS 
            elif run==2:
                df5 = df_CAS 
                frames = [df5,df4]
                df_CAS = pd.concat(frames)
                CAS.append(df_CAS)
                break

        if nums_file_paths ==1:
            CAS.append(df_CAS)

        run = run+1 
#%%
#Now FCDP data
#This is how we will correct our droplet concentration units from 
#dN/dlogD to dN/dD
L_00f=math.log10(4.5)-math.log10(3)
L_01f=math.log10(6)-math.log10(4.5)
L_02f=math.log10(8)-math.log10(6)
L_03f=math.log10(10)-math.log10(8)
L_04f=math.log10(12)-math.log10(10)
L_05f=math.log10(14)-math.log10(12)
L_06f=math.log10(16)-math.log10(14)
L_07f=math.log10(18)-math.log10(16)
L_08f=math.log10(11)-math.log10(10)
L_09f=math.log10(12)-math.log10(11)
L_10f=math.log10(13)-math.log10(12)
L_11f=math.log10(14)-math.log10(13)
L_12f=math.log10(16)-math.log10(14)
L_13f=math.log10(18)-math.log10(16)
L_14f=math.log10(21)-math.log10(18)
L_15f=math.log10(24)-math.log10(21)
L_16f=math.log10(27)-math.log10(24)
L_17f=math.log10(30)-math.log10(27)
L_18f=math.log10(33)-math.log10(30)
L_19f=math.log10(36)-math.log10(33)
L_20f=math.log10(39)-math.log10(36)
L_21f=math.log10(42)-math.log10(39)
L_22f=math.log10(46)-math.log10(42)
L_23f=math.log10(50)-math.log10(46)


bin_log_FCDP=[L_00f, L_01f, L_02f, L_03f, L_04f, L_05f, L_06f, L_07f, L_08f,
          L_09f, L_10f, L_11f, L_12f, L_13f, L_14f, L_15f, L_16f, 
        L_17f, L_18f, L_19f, L_20f, L_21f, L_22f, L_23f]


P00f=(4.5-3)
P01f=(6-4.5)
P02f=(8-6)
P03f=(10-8)
P04f=(12-10)
P05f=(14-12)
P06f=(16-14)
P07f=(18-16)
P08f=(21-18)
P09f=(24-21)
P10f=(27-24)
P11f=(30-27)
P12f = (33-30)
P13f = (36-33)
P14f = (39-36)
P15f = (42-39)
P16f = (46-42)
P17f = (50-46)



J00f=(L_00f / P00f)
J01f=(L_01f / P01f)
J02f=(L_02f / P02f)
J03f=(L_03f / P03f)
J04f=(L_04f / P04f)
J05f=(L_05f / P05f)
J06f=(L_06f / P06f)
J07f=(L_07f / P07f)
J08f=(L_08f / P08f)
J09f=(L_09f / P09f)
J10f=(L_10f / P10f)
J11f=(L_11f / P11f)
J12f = (L_12f / P12f)
J13f = (L_13f / P13f)
J14f = (L_14f / P14f)
J15f = (L_15f / P15f)
J16f = (L_16f / P16f)
J17f = (L_17f / P17f)

Logg_FCDP = [J00f, J01f, J02f, J03f, J04f, J05f, J06f, J07f, J08f, J09f, J10f, 
            J11f, J12f, J13f, J14f, J15f, J16f, J17f]

Logg_FCDP = np.array(Logg_FCDP)
 
bin_center_FCDP = [
    3.75, 5.25, 7.00, 9.00, 11.00, 13.00,
    15.00, 17.00, 19.50, 22.50, 25.50,
    28.50, 31.50, 34.50, 37.50, 40.50,
    44.00, 48.00
]
# %%
#Import the instrument data for the fast cloud droplet probe 
bin_name_FCDP = [
    'dNdlogD_003_FCDP', 'dNdlogD_004_FCDP', 'dNdlogD_005_FCDP',
    'dNdlogD_006_FCDP', 'dNdlogD_007_FCDP', 'dNdlogD_008_FCDP',
    'dNdlogD_009_FCDP', 'dNdlogD_010_FCDP', 'dNdlogD_011_FCDP',
    'dNdlogD_012_FCDP', 'dNdlogD_013_FCDP', 'dNdlogD_014_FCDP',
    'dNdlogD_015_FCDP', 'dNdlogD_016_FCDP', 'dNdlogD_017_FCDP',
    'dNdlogD_018_FCDP', 'dNdlogD_019_FCDP', 'dNdlogD_020_FCDP'
]

dates_FCDP = [
    '2022-01-11', '2022-01-12', '2022-01-15', '2022-01-18',
    '2022-01-19', '2022-01-24', '2022-01-26', '2022-01-27',
    '2022-02-01', '2022-02-02', '2022-02-03', '2022-02-05',
    '2022-02-15', '2022-02-16', '2022-02-19', '2022-02-22',
    '2022-02-26', '2022-03-03', '2022-03-04', '2022-03-13',
    '2022-03-14', '2022-03-18', '2022-03-22', '2022-03-26',
    '2022-03-28', '2022-03-29', '2022-05-05', '2022-05-10',
    '2022-05-16', '2022-05-17', '2022-05-18', '2022-05-20',
    '2022-05-21', '2022-05-31', '2022-06-02', '2022-06-03',
    '2022-06-05', '2022-06-07', '2022-06-08', '2022-06-10',
    '2022-06-11', '2022-06-13', '2022-06-14', '2022-06-17',
    '2022-06-18'
]
FCDP = []
loaded_dates_FCDP = []
loaded_files_FCDP = []
input_dir = '/home/disk/eos4/kathem24/activate/data/2022/FCDP'

for date in dates_FCDP:

    file_paths = glob.glob(
        os.path.join(
            input_dir,
            f'*{date.replace("-", "")}*.ict'
        )
    )

    if len(file_paths) == 0:
        print(f"File not found for date {date}")
        continue
    def file_order(file_path):
        filename = os.path.basename(file_path).upper()

        if "L1" in filename:
            return 1
        elif "L2" in filename:
            return 2
        else:
            return 3

    file_paths = sorted(
        file_paths,
        key=file_order
    )

    flight_parts = []
    flight_filenames = []

    for file_path in file_paths:
        with open(file_path, "r", errors="replace") as file:
            first_line = file.readline().strip()

        number_header_lines = int(
            first_line.split(",")[0].strip()
        )

        df_FCDP_part = pd.read_csv(
            file_path,
            skiprows=number_header_lines - 1,
            header=0,
            quoting=csv.QUOTE_NONE,
            skipinitialspace=True
        )
        df_FCDP_part.columns = (
            df_FCDP_part.columns
            .astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
            .str.strip('"')
            .str.strip()
        )
        if "Time_Start" not in df_FCDP_part.columns:
            print(
                f"Skipping {os.path.basename(file_path)}: "
                f"Time_Start was not found."
            )
            print(df_FCDP_part.columns.tolist())
            continue
        df_FCDP_part.replace(
            [-9999, -9999.00],
            np.nan,
            inplace=True
        )
        object_columns = df_FCDP_part.select_dtypes(
            include=["object"]
        ).columns
        for column in object_columns:
            df_FCDP_part[column] = (
                df_FCDP_part[column]
                .astype(str)
                .str.strip()
                .str.strip('"')
                .str.strip()
                .replace("nan", np.nan)
            )
        df_FCDP_part["Time_Start"] = pd.to_numeric(
            df_FCDP_part["Time_Start"],
            errors="coerce"
        )
        df_FCDP_part = df_FCDP_part.dropna(
            subset=["Time_Start"]
        ).reset_index(drop=True)

        flight_parts.append(df_FCDP_part)
        flight_filenames.append(
            os.path.basename(file_path)
        )

        print(
            f"Loaded {date} part: "
            f"{os.path.basename(file_path)}"
        )

    if len(flight_parts) == 0:
        print(f"No valid FCDP files loaded for {date}")
        continue
    df_FCDP = pd.concat(
        flight_parts,
        ignore_index=True   )
    df_FCDP["Date"] = date
    time_is_ordered = df_FCDP["Time_Start"].is_monotonic_increasing
    print(f"\nCombined FCDP flight for {date}")
    print(f"Files: {flight_filenames}")
    print(f"Combined rows: {len(df_FCDP)}")
    print(
        f"Time range: "
        f"{df_FCDP['Time_Start'].min()} to "
        f"{df_FCDP['Time_Start'].max()}"
    )
    print(f"Time ordered: {time_is_ordered}")
    FCDP.append(df_FCDP)
    loaded_dates_FCDP.append(date)
    loaded_files_FCDP.append(flight_filenames)
#%%
#CAS average size distribution first 
master_CAS_BCB = []
leg_info = []
for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict = leg_data[i]
    flight_date = leg_dict['Date']
    BCB_start = leg_dict['LegIndex_02']['StartTimes']
    BCB_stop = leg_dict['LegIndex_02']['StopTimes']
    CAS_flight = CAS[i]
    twoDS_flight = twoDS[i]
    rh_flight = h20[i]
    CAS_flight['Time_mid'] = pd.to_numeric(CAS_flight['Time_mid'], errors='coerce')
    twoDS_flight['Time_Start'] = pd.to_numeric(twoDS_flight['Time_Start'], errors='coerce')
    rh_flight['Time_Start'] = pd.to_numeric(rh_flight['Time_Start'], errors='coerce')
    CAS_times = CAS_flight['Time_mid'].values
    CAS_lwc = CAS_flight['LWC_CAS'].values
    CAS_bins = {f'CAS_Bin{bin_label:02d}': CAS_flight[f'CAS_Bin{bin_label:02d}'].values for bin_label in range(12, 30)}
    TwoDS_times = twoDS_flight['Time_Start'].values
    TwoDS_N_total = twoDS_flight['N-total_2DS'].values

    rh_times = rh_flight['Time_Start'].values
    rh_values = rh_flight.RHw_DLH.values

    total_BCB_means = []

    for k in range(len(BCB_start)):
        start20 = BCB_start[k]
        end20 = BCB_stop[k]

        CAS_indices_in_range = np.where((CAS_times >= start20) & (CAS_times <= end20))[0]
        TwoDS_indices_in_range = np.where((TwoDS_times >= start20) & (TwoDS_times <= end20))[0]
        rh_indices_in_range = np.where((rh_times >= start20) & (rh_times <= end20))[0]
    
        if CAS_indices_in_range.size > 0 and TwoDS_indices_in_range.size > 0 and rh_indices_in_range.size > 0:
            data_labels = []
            BCB_means = []

            for CAS_idx, TwoDS_idx, rh_idx in zip(CAS_indices_in_range, TwoDS_indices_in_range, rh_indices_in_range):
                lwc_val = CAS_lwc[CAS_idx]
                N_val = TwoDS_N_total[TwoDS_idx]
                rh_val = rh_values[rh_idx]
                lwc_label = 'Y' if 0 <= lwc_val <= 0.0025 else 'N'
                N_label = 'Y' if 0 <= N_val <= 100 else 'N'
                rh_label = 'Y' if 0 <= rh_val <= 95 else 'N'
                label = 'Y' if lwc_label == 'Y' and N_label == 'Y' and rh_label == 'Y' else 'N'
                if label == 'Y' and rh_val > 95:
                    print(f"❗ RH violation: {rh_val:.2f} passed at time {CAS_times[CAS_idx]}")

                data_labels.append(label)    
                bin_values = [CAS_bins[f'CAS_Bin{bin_label:02d}'][CAS_idx] for bin_label in range(12, 30)]
                BCB_means.append(bin_values)

            if BCB_means:
                total_BCB_means.append(BCB_means)

            leg_info.append({
                'Date': date,
                'BCB_start': start20,
                'BCB_stop': end20,
                'Data_Labels': data_labels,
            })
    master_CAS_BCB.append(total_BCB_means)

for leg in leg_info:
    print(f"Date: {leg['Date']}, Start: {leg['BCB_start']}, Stop: {leg['BCB_stop']}, Data Labels: {leg['Data_Labels']}")
#%%
#Double check the number of legs associated with each date to compare across multiple instruments.  
leg_count = Counter([leg['Date'] for leg in leg_info])
print("Number of legs associated with each date:")
total_legs = 0
for date, count in sorted(leg_count.items()):
    print(f"Date: {date}, Number of Legs: {count}")
    total_legs += count
print(f"\nTotal number of legs: {total_legs}")
#%%
rh_Y_values = []
for leg in leg_info:
    date = leg['Date']
    start = leg['BCB_start']
    stop = leg['BCB_stop']
    
    flight_index = dates_legs.index(date)
    rh_flight = h20[flight_index]
    rh_times = rh_flight['Time_Start'].values
    rh_vals = rh_flight['RHw_DLH'].values

    rh_leg_indices = np.where((rh_times >= start) & (rh_times <= stop))[0]
    rh_leg_vals = rh_vals[rh_leg_indices]
    labels = leg['Data_Labels']
    for idx, label in enumerate(labels):
        if label == 'Y' and idx < len(rh_leg_vals):
            rh_Y_values.append(rh_leg_vals[idx])
plt.hist(rh_Y_values, bins=30, color='teal', edgecolor='black')
plt.axvline(95, color='red', linestyle='--', label='RH = 95% threshold')
plt.xlabel("RH (%)")
plt.ylabel("Count")
plt.title("RH Values for Seconds Labeled 'Y'")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
master_CAS_BCB = []
leg_info = []

for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict = leg_data[i]

    flight_date = leg_dict['Date']
    BCB_start = np.array(leg_dict['LegIndex_02']['StartTimes'], dtype=float)
    BCB_stop = np.array(leg_dict['LegIndex_02']['StopTimes'], dtype=float)

    CAS_flight = CAS[i]
    twoDS_flight = twoDS[i]
    rh_flight = h20[i]

    CAS_times = np.array(CAS_flight['Time_mid'], dtype=float)
    TwoDS_times = np.array(twoDS_flight['Time_Start'], dtype=float)
    rh_times = np.array(rh_flight['Time_Start'], dtype=float)

    lwc = np.array(CAS_flight['LWC_CAS'], dtype=float)
    N_total = np.array(twoDS_flight['N-total_2DS'], dtype=float)
    rh_total = np.array(rh_flight['RHw_DLH'], dtype=float)

    bins = {
        f'CAS_Bin{bin_label:02d}': np.array(CAS_flight[f'CAS_Bin{bin_label:02d}'], dtype=float)
        for bin_label in range(12, 30)
    }

    total_BCB_means = []

    for k in range(len(BCB_start)):
        start20 = BCB_start[k]
        end20 = BCB_stop[k]

        bin_means = {f'Bin{bin_label:02d}_Y_mean': [] for bin_label in range(12, 30)}
        bin_means.update({f'Bin{bin_label:02d}_N_mean': [] for bin_label in range(12, 30)})
        bin_means.update({'Date': date, 'BCB_start': start20, 'BCB_stop': end20})

        CAS_indices_in_range = np.where((CAS_times >= start20) & (CAS_times <= end20))[0]
        TwoDS_indices_in_range = np.where((TwoDS_times >= start20) & (TwoDS_times <= end20))[0]
        rh_indices_in_range = np.where((rh_times >= start20) & (rh_times <= end20))[0]

        if CAS_indices_in_range.size > 0 and TwoDS_indices_in_range.size > 0 and rh_indices_in_range.size >0:
            for cas_idx, twods_idx, rh_idx in zip(CAS_indices_in_range, TwoDS_indices_in_range, rh_indices_in_range):
                lwc_val = lwc[cas_idx]
                N_val = N_total[twods_idx]
                rh_val = rh_total[rh_idx]
                lwc_label = 'Y' if 0 <= lwc_val <= 0.0025 else 'N'
                N_label = 'Y' if 0 <= N_val <= 100 else 'N'
                rh_label = 'Y' if 0 <= rh_val <= 95 else 'N'
                label = 'Y' if lwc_label == 'Y' and N_label == 'Y' and rh_label == 'Y' else 'N'

                for bin_label in range(12, 30):
                    bin_key = f'Bin{bin_label:02d}_{label}_mean'
                    bin_means[bin_key].append(bins[f'CAS_Bin{bin_label:02d}'][cas_idx])

        
        for bin_label in range(12, 30):
            for label in ['Y', 'N']:
                bin_key = f'Bin{bin_label:02d}_{label}_mean'
                if bin_means[bin_key]:
                    bin_means[bin_key] = np.nanmean(bin_means[bin_key])
                else:
                    bin_means[bin_key] = np.nan

        total_BCB_means.append(bin_means)

    master_CAS_BCB.append(total_BCB_means)

for item in master_CAS_BCB:
    for bin_means in item:
        print(f"Date: {bin_means['Date']}, Start: {bin_means['BCB_start']}, Stop: {bin_means['BCB_stop']}")
        for bin_label in range(12, 30):
            for label in ['Y', 'N']:
                bin_key = f'Bin{bin_label:02d}_{label}_mean'
                print(f"   {bin_key}: {bin_means[bin_key]}")

#%%
# Count the total number of legs from master_CAS_BCB
total_legs = sum(len(item) for item in master_CAS_BCB)
print(f"Total number of legs: {total_legs}")

#%%
Y_BCB_calc = []
N_BCB_calc = []

for flight_data in master_CAS_BCB:
    for bin_means in flight_data:
        Y_calc = {'Date': bin_means['Date'], 'BCB_start': bin_means['BCB_start'], 'BCB_stop': bin_means['BCB_stop']}
        N_calc = {'Date': bin_means['Date'], 'BCB_start': bin_means['BCB_start'], 'BCB_stop': bin_means['BCB_stop']}
        
        for bin_label in range(12, 30):
            bin_key_Y = f'Bin{bin_label}_Y_mean'
            bin_key_N = f'Bin{bin_label}_N_mean'

            Y_calc[bin_key_Y] = np.nanmean(bin_means[bin_key_Y]) * Logg[bin_label - 12]
            N_calc[bin_key_N] = np.nanmean(bin_means[bin_key_N]) * Logg[bin_label - 12]

        Y_BCB_calc.append(Y_calc)
        N_BCB_calc.append(N_calc)
# %%
plt.figure(figsize=(8, 6))
for entry in Y_BCB_calc:
    bin_means = np.array([entry.get(f'Bin{i}_Y_mean', np.nan) for i in range(12, 30)], dtype=float)  
    valid_indices = ~np.isnan(bin_means)  
    bin_centers_valid = np.array(bin_center)[valid_indices]
    bin_means_valid = bin_means[valid_indices]

    plt.plot(bin_centers_valid, bin_means_valid, color='black', alpha=0.5)

plt.xlabel("Deliquesed Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r" CAS Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-7, 10**1)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base January - June 2022\n Raw Ambient Size Distributions", fontsize=14, fontweight="bold")
plt.show()
#%%
#Filtering the 0s
plt.figure(figsize=(8, 6))

for entry in Y_BCB_calc:
    bin_means = np.array([entry.get(f'Bin{i}_Y_mean', np.nan) for i in range(12, 30)], dtype=float)  
    bin_centers = np.array(bin_center)

    valid_indices = (bin_means > 0) & ~np.isnan(bin_means)  
    bin_centers_valid = bin_centers[valid_indices]
    bin_means_valid = bin_means[valid_indices]

    if len(bin_centers_valid) > 0:  
        plt.plot(bin_centers_valid, bin_means_valid, color='black', alpha=0.5)
plt.xlabel("Deliquesced Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"CAS Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-7, 10**1)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base January - June 2022\n Raw Ambient Size Distributions", fontsize=14, fontweight="bold")
plt.show()
#%%
#average ambient distribution
sum_bin_means = np.zeros(len(bin_center))
count_bin_means = np.zeros(len(bin_center))
for entry in Y_BCB_calc:
    bin_means = np.array([entry.get(f'Bin{i}_Y_mean', np.nan) for i in range(12, 30)], dtype=float)

    valid_indices = (bin_means > 0) & ~np.isnan(bin_means)

    sum_bin_means[valid_indices] += bin_means[valid_indices]
    count_bin_means[valid_indices] += 1

average_bin_means = np.divide(sum_bin_means, count_bin_means, where=count_bin_means > 0)
plt.figure(figsize=(8, 6))
plt.plot(bin_center, average_bin_means, color='red', linewidth=2, label='Average Size Distribution')
plt.xlabel("Deliquesced Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"CAS Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=19, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xlim(0, 45)
plt.xticks(fontweight="bold", fontsize=18)
plt.yticks(fontweight="bold", fontsize=18)
plt.title("CAS Average Ambient \nBelow Cloud Base Size Distribution\n January - June 2022", fontsize=19, fontweight="bold")
plt.show()
#%%
#Fitting an exponential to each size distribution
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)
ambient_fits = []
plt.figure(figsize=(8, 6))
for entry in Y_BCB_calc:
    bin_means = np.array([entry.get(f'Bin{i}_Y_mean', np.nan) for i in range(12, 30)], dtype=float)
    valid_indices = ~np.isnan(bin_means)
    bin_centers_valid = np.array(bin_center)[valid_indices]
    bin_means_valid = bin_means[valid_indices]
    if not valid_indices.any():
        print(f"No valid data for date {entry['Date']}")
        continue

    try:
        popt, _ = curve_fit(exponential, bin_centers_valid, bin_means_valid, p0=(1, 1), maxfev=5000)
        n0, D = popt

        ambient_fits.append({
            'Date': entry['Date'],
            'BCB_start': entry.get('BCB_start', np.nan),
            'BCB_stop': entry.get('BCB_stop', np.nan),
            'Intercept_n0': n0,
            'E_folding_D': D
        })

        x_fit = np.linspace(min(bin_centers_valid), max(bin_centers_valid), 100)
        y_fit = exponential(x_fit, *popt)

        plt.plot(x_fit, y_fit, color='black', alpha=0.2)

    except RuntimeError:
        print(f"Fit could not be performed for date {entry['Date']}")
plt.xlabel("Deliquesed Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"CAS Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base January - June 2022\n Exponential Fit to Ambient Size Distributions", fontsize=14, fontweight="bold")
plt.show()
#%%
#Trying to fit and stop at 10um 
bin_center=np.array(bin_center)
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

ambient_fits_10 = []

plt.figure(figsize=(8, 6))

for entry in Y_BCB_calc:
    bin_means = np.array([entry.get(f'Bin{i}_Y_mean', np.nan) for i in range(12, 30)], dtype=float)
    valid_indices = (bin_center <= 10) & ~np.isnan(bin_means) 
    bin_centers_valid = np.array(bin_center)[valid_indices]
    bin_means_valid = bin_means[valid_indices]

    if valid_indices.any():
        try:
            popt, _ = curve_fit(exponential, bin_centers_valid, bin_means_valid, 
                                p0=(1, 1), maxfev=5000, 
                                bounds=([0, 0.1], [np.inf, 20])) 
            n0, D = popt

            if D > 15:
                print(f"⚠️ High slope detected! Date: {entry['Date']}, D: {D:.2f}")

        except RuntimeError:
            print(f"❌ Fit failed for date {entry['Date']}")
            n0, D = np.nan, np.nan 
    else:
        n0, D = np.nan, np.nan 


    ambient_fits_10.append({
        'Date': entry['Date'],
        'BCB_start': entry.get('BCB_start', np.nan),
        'BCB_stop': entry.get('BCB_stop', np.nan),
        'Intercept_n0': n0,
        'E_folding_D': D
    })

    if not np.isnan(n0) and not np.isnan(D):
        x_fit = np.linspace(min(bin_centers_valid), 10, 100)
        y_fit = exponential(x_fit, *popt)
        plt.plot(x_fit, y_fit, color='black', alpha=0.2)
plt.xlabel("Deliquesed Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"CAS Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
# plt.ylim(10**-33, 10**1)
plt.ylim(10**-7, 10**1)
plt.xlim(0, 10)
plt.title(" CAS Below Cloud Base January - June 2022\n Exponential Fit Ambient Size Distributions (≤10 µm)", fontsize=14, fontweight="bold")
plt.show()
print(f"Total successful ambient exponential fits: {np.sum(~np.isnan([fit['E_folding_D'] for fit in ambient_fits]))}")
#%%
Y_BCB_calc_cm3 = []
N_BCB_calc_cm3 = []

for flight_data in master_CAS_BCB:
    for bin_means in flight_data:
        Y_calc = {'Date': bin_means['Date'], 'BCB_start': bin_means['BCB_start'], 'BCB_stop': bin_means['BCB_stop']}
        N_calc = {'Date': bin_means['Date'], 'BCB_start': bin_means['BCB_start'], 'BCB_stop': bin_means['BCB_stop']}
        
        for bin_label in range(12, 30):
            bin_key_Y = f'Bin{bin_label}_Y_mean'
            bin_key_N = f'Bin{bin_label}_N_mean'

            Y_calc[bin_key_Y] = np.nanmean(bin_means[bin_key_Y]) * bin_log[bin_label - 12]
            N_calc[bin_key_N] = np.nanmean(bin_means[bin_key_N]) * bin_log[bin_label - 12]

        Y_BCB_calc_cm3.append(Y_calc)
        N_BCB_calc_cm3.append(N_calc)
# %%
#Calculating total number concentration 
total_concentration_cm3 = []
for entry in Y_BCB_calc_cm3:
    total_Y_concentration = np.nansum([entry[f'Bin{i}_Y_mean'] for i in range(12, 30)])  # Sum all valid bin concentrations

    total_concentration_cm3.append({
        'Date': entry['Date'],
        'BCB_start': entry['BCB_start'],
        'BCB_stop': entry['BCB_stop'],
        'Total_Y_Concentration_cm3': total_Y_concentration
    })
#%%
total_Y_concentrations = [entry['Total_Y_Concentration_cm3'] for entry in total_concentration_cm3]
total_Y_concentrations = [conc for conc in total_Y_concentrations if not np.isnan(conc)]
mean_total_concentration = np.mean(total_Y_concentrations)
print(f"Mean Total Number Concentration: {mean_total_concentration:.2f} cm⁻³")
# %%
#Calculate the relative humidity of each leg.
master_BCB_RH = []
for i in range(len(dates_legs)):
    date = dates_legs[i]

    leg_dict = leg_data[i]

    flight_date = leg_dict['Date']  # Get date of flight from dictionary 
    BCB_start = leg_dict['LegIndex_02']['StartTimes']
    BCB_stop = leg_dict['LegIndex_02']['StopTimes']

    rh_flight = h20[i]
    times_rh = rh_flight.Time_Start.values
    rh_values = rh_flight.RHw_DLH.values

    all_BCB = []

    for j in range(len(BCB_start)):
        start = int(BCB_start[j])
        end = int(BCB_stop[j])

        rh_times = {
            'Date': date,
            'BCB_start': start,
            'BCB_stop': end,
            'Rh_mean': [],
        }

        
        index1_start = None
        for k in range(len(times_rh)):
            if times_rh[k] == start:
                index1_start = k
                break

        
        index1_end = None
        for k in range(len(times_rh)):
            if times_rh[k] == end:
                index1_end = k
                break

        if index1_start is None or index1_end is None:
            rh9_mean = np.nan
        else:
            rh9 = rh_values[index1_start:index1_end + 1]
            rh9 = rh9[(rh9 <= 95) & (rh9 > 0)]  # filter for RH ≤ 95 and ignore missing/bad (-999 or 0)
            rh9_mean = np.nanmean(rh9)


        rh_times['Rh_mean'].append(rh9_mean)
        all_BCB.append(rh_times)  # List that contains all the BCB wind/alt mean dictionaries for 1 flight

    master_BCB_RH.append(all_BCB)
for flight in master_BCB_RH:
    for leg in flight:
        rh_mean_list = leg['Rh_mean']
        leg['Rh_mean'] = [np.nan if value <=0 else value for value in rh_mean_list]

#%%
date_leg_set = set()

for entry in Y_BCB_calc: 
    date = entry['Date']
    BCB_start = entry.get('BCB_start', np.nan)
    BCB_stop = entry.get('BCB_stop', np.nan)
    date_leg_set.add((date, BCB_start, BCB_stop))

filtered_master_BCB_RH = []

for flight in master_BCB_RH:
    filtered_legs = []
    for leg in flight:
        date = leg['Date']
        BCB_start = leg['BCB_start']
        BCB_stop = leg['BCB_stop']
        if (date, BCB_start, BCB_stop) in date_leg_set:
            filtered_legs.append(leg)
    if filtered_legs:
        filtered_master_BCB_RH.append(filtered_legs)
#%%
#Make sure the leg counts 
total_entries_filtered_master_BCB_RH = sum(len(legs) for legs in filtered_master_BCB_RH)
print(f"Total entries in filtered_master_BCB_RH: {total_entries_filtered_master_BCB_RH}")
# %%
##Obtaining g(RH) = [1.7 / (1-RH)]^0.31 for all mean RH values 
## ie for every leg 

master_BCB_gRH = []
for flight in master_BCB_RH:
    flight_gRH = [] 
    
    for leg in flight:
        new_leg = leg.copy()  
        
        rh_mean = new_leg['Rh_mean'][0] / 100.0  
        if np.isnan(rh_mean) or rh_mean >= 1:
            gRH_value = np.nan
            print(f"Skipping calculation for Rh_mean = {new_leg['Rh_mean'][0]} as it results in division by zero or invalid value.")
        else:
            gRH_value = (1.7 / (1 - rh_mean)) ** 0.31
        new_leg['gRh_mean'] = [gRH_value]
        flight_gRH.append(new_leg)
    master_BCB_gRH.append(flight_gRH)
#%%
#Histogram of RH values
rh_values = [
    leg['Rh_mean'][0] for flight in filtered_master_BCB_RH for leg in flight if not np.isnan(leg['Rh_mean'][0])
]
plt.figure(figsize=(8, 6))
plt.hist(rh_values, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel('Relative Humidity (%)', fontsize=15, fontweight='bold')
plt.ylabel('Frequency of flight legs', fontsize=15, fontweight='bold')
plt.title('Leg average RH January - June 2022', fontweight='bold', fontsize=16)
plt.xticks(fontweight='bold', fontsize=14)
plt.yticks(fontweight='bold', fontsize=14)
plt.show()

# %%
#only the grh from filtered_master_BCB_RH
filtered_master_BCB_gRH = []
for flight in filtered_master_BCB_RH:
    flight_gRH = []
    
    for leg in flight:
        new_leg = leg.copy() 
        
        rh_mean = new_leg['Rh_mean'][0] / 100.0  # Convert percentage to a decimal
        
        if np.isnan(rh_mean) or rh_mean >= 1:
            gRH_value = np.nan
            print(f"Skipping calculation for Rh_mean = {new_leg['Rh_mean'][0]} as it results in division by zero or invalid value.")
        else:
            gRH_value = (1.7 / (1 - rh_mean)) ** 0.31
        new_leg['gRh_mean'] = [gRH_value]
        flight_gRH.append(new_leg)
    
    filtered_master_BCB_gRH.append(flight_gRH)
#%%
total_entries_filtered_master_BCB_gRH = sum(len(legs) for legs in filtered_master_BCB_gRH)
print(f"Total entries in filtered_master_BCB_gRH: {total_entries_filtered_master_BCB_gRH}")
# %%
filtered_master_BCB_ddry = []

for flight in filtered_master_BCB_gRH:
    for entry in flight:

        date = entry['Date']
        BCB_start = entry['BCB_start']
        BCB_stop = entry['BCB_stop']
        gRh_mean = entry['gRh_mean'][0]

        if gRh_mean > 0:
            ddry_values = np.array([
                D_amb / gRh_mean for D_amb in bin_center
            ])
        else:
            ddry_values = np.full(len(bin_center), np.nan)
            print(
                f"Skipping division for {date}, "
                f"{BCB_start}-{BCB_stop} due to invalid gRh_mean."
            )

        ddry_bin_widths = np.diff(
            ddry_values,
            append=np.nan
        )

        raw_concentrations = next(
            (
                leg for leg in Y_BCB_calc
                if leg['Date'] == date
                and leg['BCB_start'] == BCB_start
                and leg['BCB_stop'] == BCB_stop
            ),
            None
        )

        if raw_concentrations is not None:

            dN_dD_ambient = np.array([
                raw_concentrations.get(
                    f'Bin{i}_Y_mean',
                    np.nan
                )
                for i in range(12, 30)
            ], dtype=float)

            dN_dD_dry = np.where(
                (~np.isnan(dN_dD_ambient))
                & (~np.isnan(ddry_bin_widths))
                & (gRh_mean > 0),

                dN_dD_ambient
                * (np.array(bin_center) / ddry_values)
                * (
                    np.diff(bin_center, append=np.nan)
                    / ddry_bin_widths
                ),

                np.nan
            )

        else:
            dN_dD_dry = np.full(
                len(bin_center),
                np.nan
            )

            print(
                f"Missing raw size distribution for "
                f"{date}, {BCB_start}-{BCB_stop}"
            )

        filtered_master_BCB_ddry.append({
            'Date': date,
            'BCB_start': BCB_start,
            'BCB_stop': BCB_stop,
            'ddry': ddry_values.tolist(),
            'dN/dDdry': dN_dD_dry.tolist(),
            'ddry_bin_widths': ddry_bin_widths.tolist(),
            'gRh_mean': gRh_mean
        })

print(
    f"Length of filtered_master_BCB_ddry: "
    f"{len(filtered_master_BCB_ddry)}"
)
#%%

from scipy.interpolate import interp1d
common_bins = np.linspace(2, 25, 35)
plt.figure(figsize=(8, 6))
for entry in filtered_master_BCB_ddry:
    ddry_values = np.array(entry['ddry']) 
    dN_dD_dry = np.array(entry['dN/dDdry'])
    valid_indices = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry)
    if np.sum(valid_indices) < 2:
        continue  

    interp_func = interp1d(ddry_values[valid_indices], dN_dD_dry[valid_indices], 
                           kind='linear', bounds_error=False, fill_value=np.nan)
    interpolated_dN_dD_dry = interp_func(common_bins)
    plt.plot(common_bins, interpolated_dN_dD_dry, color='black', alpha=0.2)
plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base January - June 2022\n Raw Dry Size Distributions", fontsize=14, fontweight="bold")
plt.show()
#%%%
#Removing the 0s
common_bins = np.linspace(2, 25, 35) 
plt.figure(figsize=(8, 6))
for entry in filtered_master_BCB_ddry:
    ddry_values = np.array(entry['ddry'])  
    dN_dD_dry = np.array(entry['dN/dDdry']) 
   
    valid_indices = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry)
    if np.sum(valid_indices) < 2:
        continue 

   
    interp_func = interp1d(ddry_values[valid_indices], dN_dD_dry[valid_indices], 
                           kind='linear', bounds_error=False, fill_value=np.nan)
    interpolated_dN_dD_dry = interp_func(common_bins)

    valid_interpolated_indices = (interpolated_dN_dD_dry > 0) & ~np.isnan(interpolated_dN_dD_dry)
    filtered_bins = common_bins[valid_interpolated_indices]
    filtered_dN_dD_dry = interpolated_dN_dD_dry[valid_interpolated_indices]

    if len(filtered_bins) > 0: 
        plt.plot(filtered_bins, filtered_dN_dD_dry, color='purple', alpha=0.2)

plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=20, fontweight="bold")
plt.ylabel("CAS Number Concentration\n(cm$^{-3}$ μm$^{-1}$)", fontsize=20, fontweight="bold")
plt.yscale("log")
plt.xticks(fontweight="bold", fontsize=20)
plt.yticks(fontweight="bold", fontsize=20)
plt.ylim(10**-7, 10**1.5)
plt.xlim(0.5, 40)
plt.title("CAS Below Cloud Base\n January-June 2022\n Raw Dry Size Distributions", fontsize=20, fontweight="bold")
plt.show()

# %%
#average dry distribution ********
common_bins = np.linspace(2, 25, 35)  
sum_interpolated_dN_dD_dry = np.zeros_like(common_bins, dtype=float)
count_interpolated_dN_dD_dry = np.zeros_like(common_bins, dtype=int)
for entry in filtered_master_BCB_ddry:
    ddry_values = np.array(entry['ddry'])  
    dN_dD_dry = np.array(entry['dN/dDdry'])
    valid_indices = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry)
    if np.sum(valid_indices) < 2:
        continue 

    interp_func = interp1d(ddry_values[valid_indices], dN_dD_dry[valid_indices], 
                           kind='linear', bounds_error=False, fill_value=np.nan)
    interpolated_dN_dD_dry = interp_func(common_bins)

    valid_interpolated_indices = ~np.isnan(interpolated_dN_dD_dry)

    sum_interpolated_dN_dD_dry[valid_interpolated_indices] += interpolated_dN_dD_dry[valid_interpolated_indices]
    count_interpolated_dN_dD_dry[valid_interpolated_indices] += 1

average_dN_dD_dry = np.divide(sum_interpolated_dN_dD_dry, count_interpolated_dN_dD_dry, where=count_interpolated_dN_dD_dry > 0)

plt.figure(figsize=(8, 6))
plt.plot(common_bins, average_dN_dD_dry, color='black', linewidth=2, label='Average Dry Size Distribution')

plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=20, fontweight="bold")
plt.ylabel(r"CAS Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=19, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xlim(0, 25)
plt.xticks(fontweight="bold", fontsize=20)
plt.yticks(fontweight="bold", fontsize=20)
plt.title("CAS Average Below Cloud Base \nDry Size Distribution\n January - June 2022", fontsize=20, fontweight="bold")
plt.legend()
plt.show()
# %%
#add 2 standard errors to the average dry distribution for 95% confidence interval
two_sem_dN_dD_dry = 2 * sem_dN_dD_dry
plot_mask = (
    np.isfinite(average_dN_dD_dry) &
    np.isfinite(two_sem_dN_dD_dry) &
    (average_dN_dD_dry > 0) &
    (count_interpolated_dN_dD_dry > 1))

x_plot = common_bins[plot_mask]
mean_plot = average_dN_dD_dry[plot_mask]
two_sem_plot = two_sem_dN_dD_dry[plot_mask]
small_positive_value = 1e-12

lower_error = np.minimum(
    two_sem_plot,
    mean_plot - small_positive_value)

lower_error = np.clip(lower_error, 0, None)

upper_error = two_sem_plot
asymmetric_yerr = np.vstack([
    lower_error,
    upper_error
])
plt.figure(figsize=(8, 6))
plt.errorbar(
    x_plot,
    mean_plot,
    yerr=asymmetric_yerr,
    fmt='-',                 # line with no point markers
    color='black',
    ecolor='black',
    linewidth=2,
    elinewidth=1.5,
    capsize=4,
    capthick=1.5,
    label=r'CAS $\pm$2 SEM (~95% CI)')
plt.xlabel(
    "Dry Bin Center Diameter (μm)",
    fontsize=20,
    fontweight="bold")
plt.ylabel(
    r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)",
    fontsize=19,
    fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xlim(0, 25)
plt.xticks(fontweight="bold", fontsize=20)
plt.yticks(fontweight="bold", fontsize=20)
plt.title(
    "Average Dry Size Distributions",
    fontsize=20,
    fontweight="bold",)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()
#%%
# Preserve CAS results before another instrument overwrites them
common_bins_CAS = common_bins.copy()
average_dN_dD_dry_CAS = average_dN_dD_dry.copy()
std_dN_dD_dry_CAS = std_dN_dD_dry.copy()
sem_dN_dD_dry_CAS = sem_dN_dD_dry.copy()
two_sem_dN_dD_dry_CAS = two_sem_dN_dD_dry.copy()
count_interpolated_dN_dD_dry_CAS = (
    count_interpolated_dN_dD_dry.copy()
)
# %%
#save 
lower_2SEM = average_dN_dD_dry - two_sem_dN_dD_dry
upper_2SEM = average_dN_dD_dry + two_sem_dN_dD_dry
average_dry_distribution = pd.DataFrame({
    'Dry_Diameter_um': common_bins,
    'Average_dN_dD_dry': average_dN_dD_dry,
    'Standard_Deviation': std_dN_dD_dry,
    'SEM': sem_dN_dD_dry,
    'Two_SEM': two_sem_dN_dD_dry,
    'Lower_2SEM': lower_2SEM,
    'Upper_2SEM': upper_2SEM,
    'N_profiles': count_interpolated_dN_dD_dry
})
save_dir = "/home/disk/eos4/kathem24/activate/data/2021/CAS"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(
    save_dir,
    "Average_Dry_Size_Distribution_errorbars2022CAS.csv")
average_dry_distribution.to_csv(save_path, index=False)
print(f"Saved to: {save_path}")
print(average_dry_distribution.head())

# %%
#%%
#FCDP
master_FCDP_BCB = []
leg_info_FCDP = []
for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict_FCDP = leg_data[i]
    flight_date = leg_dict_FCDP['Date']
    BCB_start = leg_dict_FCDP['LegIndex_02']['StartTimes']
    BCB_stop = leg_dict_FCDP['LegIndex_02']['StopTimes']

    FCDP_flight = FCDP[i]
    twoDS_flight = twoDS[i]
    rh_flight = h20[i]
    FCDP_flight['Time_Start'] = pd.to_numeric(FCDP_flight['Time_Start'], errors='coerce')
    twoDS_flight['Time_Start'] = pd.to_numeric(twoDS_flight['Time_Start'], errors='coerce')
    rh_flight['Time_Start'] = pd.to_numeric(rh_flight['Time_Start'], errors='coerce')
    FCDP_times = FCDP_flight['Time_Start'].values
    FCDP_lwc = FCDP_flight['LWC_FCDP'].values
    FCDP_bins = {
    col: FCDP_flight[col].values
    for col in bin_name_FCDP
}    
    TwoDS_times = twoDS_flight['Time_Start'].values
    TwoDS_N_total = twoDS_flight['N-total_2DS'].values
    rh_times = rh_flight['Time_Start'].values
    rh_values = rh_flight.RHw_DLH.values
    total_BCB_means = []
    for k in range(len(BCB_start)):
        start20 = BCB_start[k]
        end20 = BCB_stop[k]
        FCDP_indices_in_range = np.where((FCDP_times >= start20) & (FCDP_times <= end20))[0]
        TwoDS_indices_in_range = np.where((TwoDS_times >= start20) & (TwoDS_times <= end20))[0]
        rh_indices_in_range = np.where((rh_times >= start20) & (rh_times <= end20))[0]
        if FCDP_indices_in_range.size > 0 and TwoDS_indices_in_range.size > 0 and rh_indices_in_range.size > 0:
            data_labels = []
            BCB_means = []
            for FCDP_idx, TwoDS_idx, rh_idx in zip(FCDP_indices_in_range, TwoDS_indices_in_range, rh_indices_in_range):
                lwc_val = FCDP_lwc[FCDP_idx]
                N_val = TwoDS_N_total[TwoDS_idx]
                rh_val = rh_values[rh_idx]
                lwc_label = 'Y' if 0 <= lwc_val <= 2.5e-6 else 'N'
                N_label   = 'Y' if 0 <= N_val   <= 100    else 'N'
                rh_label = 'Y' if 0 <= rh_val <= 95 else 'N'
                label = 'Y' if lwc_label == 'Y' and N_label == 'Y' and rh_label == 'Y' else 'N'
                if label == 'Y' and rh_val > 95:
                    print(f"RH violation: {rh_val:.2f} passed at time {FCDP_times[FCDP_idx]}")
                data_labels.append(label)
                bin_values = [FCDP_bins[col][FCDP_idx] for col in bin_name_FCDP]
                BCB_means.append(bin_values)
            if BCB_means:
                total_BCB_means.append(BCB_means)
            leg_info_FCDP.append({
                'Date': date,
                'BCB_start': start20,
                'BCB_stop': end20,
                'Data_Labels': data_labels,
            })
    master_FCDP_BCB.append(total_BCB_means)
for leg in leg_info_FCDP:
    print(f"Date: {leg['Date']}, Start: {leg['BCB_start']}, Stop: {leg['BCB_stop']}, Data Labels: {leg['Data_Labels']}")
#%%
#Double check the number of legs associated with each date to compare across multiple instruments.  
leg_count = Counter([leg['Date'] for leg in leg_info_FCDP])
print("Number of legs associated with each date:")
total_legs = 0
for date, count in sorted(leg_count.items()):
    print(f"Date: {date}, Number of Legs: {count}")
    total_legs += count
print(f"\nTotal number of legs: {total_legs}")
#%%
rh_Y_values_FCDP = []
for leg in leg_info_FCDP:
    date = leg['Date']
    start = leg['BCB_start']
    stop = leg['BCB_stop']
    flight_index = dates_legs.index(date)
    rh_flight = h20[flight_index]
    rh_times = rh_flight['Time_Start'].values
    rh_vals = rh_flight['RHw_DLH'].values

    rh_leg_indices = np.where((rh_times >= start) & (rh_times <= stop))[0]
    rh_leg_vals = rh_vals[rh_leg_indices]
    labels = leg['Data_Labels']
    for idx, label in enumerate(labels):
        if label == 'Y' and idx < len(rh_leg_vals):
            rh_Y_values_FCDP.append(rh_leg_vals[idx])
plt.hist(rh_Y_values_FCDP, bins=30, color='teal', edgecolor='black')
plt.axvline(95, color='red', linestyle='--', label='RH = 95% threshold')
plt.xlabel("RH (%)")
plt.ylabel("Count")
plt.title("RH Values for Seconds Labeled 'Y'")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
master_FCDP_BCB = []
leg_info_FCDP = []
fcdp_bin_labels = list(range(3, 21))
expected_FCDP_columns = [
    f"dNdlogD_{bin_label:03d}_FCDP"
    for bin_label in fcdp_bin_labels
]
print("Number of FCDP bin columns:", len(bin_name_FCDP))

if bin_name_FCDP != expected_FCDP_columns:
    print("WARNING: bin_name_FCDP does not match the expected order.")

    for expected, actual in zip(
        expected_FCDP_columns,
        bin_name_FCDP
    ):
        print(f"Expected: {expected} | Actual: {actual}")


for i in range(len(dates_legs)):

    date = dates_legs[i]
    leg_dict = leg_data[i]

    flight_date = leg_dict["Date"]

    BCB_start = np.array(
        leg_dict["LegIndex_02"]["StartTimes"],
        dtype=float
    )

    BCB_stop = np.array(
        leg_dict["LegIndex_02"]["StopTimes"],
        dtype=float
    )

    FCDP_flight = FCDP[i]
    twoDS_flight = twoDS[i]
    rh_flight = h20[i]

    FCDP_times = np.array(
        FCDP_flight["Time_Start"],
        dtype=float
    )

    TwoDS_times = np.array(
        twoDS_flight["Time_Start"],
        dtype=float
    )

    rh_times = np.array(
        rh_flight["Time_Start"],
        dtype=float
    )

    lwc = np.array(
        FCDP_flight["LWC_FCDP"],
        dtype=float
    )

    N_total = np.array(
        twoDS_flight["N-total_2DS"],
        dtype=float
    )

    rh_total = np.array(
        rh_flight["RHw_DLH"],
        dtype=float
    )
    bins = {
        f"FCDP_Bin{bin_label:03d}": np.array(
            FCDP_flight[column],
            dtype=float
        )
        for bin_label, column in zip(
            fcdp_bin_labels,
            bin_name_FCDP
        )
    }

    total_BCB_means = []

    for k in range(len(BCB_start)):

        start20 = BCB_start[k]
        end20 = BCB_stop[k]
        bin_means = {
            f"Bin{bin_label:03d}_Y_mean": []
            for bin_label in fcdp_bin_labels
        }

        bin_means.update({
            f"Bin{bin_label:03d}_N_mean": []
            for bin_label in fcdp_bin_labels
        })

        bin_means.update({
            "Date": date,
            "BCB_start": start20,
            "BCB_stop": end20
        })

        FCDP_indices_in_range = np.where(
            (FCDP_times >= start20)
            & (FCDP_times <= end20)
        )[0]

        TwoDS_indices_in_range = np.where(
            (TwoDS_times >= start20)
            & (TwoDS_times <= end20)
        )[0]

        rh_indices_in_range = np.where(
            (rh_times >= start20)
            & (rh_times <= end20)
        )[0]

        if (
            FCDP_indices_in_range.size > 0
            and TwoDS_indices_in_range.size > 0
            and rh_indices_in_range.size > 0
        ):

            data_labels = []

            for fcdp_idx, twods_idx, rh_idx in zip(
                FCDP_indices_in_range,
                TwoDS_indices_in_range,
                rh_indices_in_range
            ):

                lwc_val = lwc[fcdp_idx]
                N_val = N_total[twods_idx]
                rh_val = rh_total[rh_idx]

                lwc_label = (
                    "Y"
                    if 0 <= lwc_val <= 2.5e-6
                    else "N"
                )

                N_label = (
                    "Y"
                    if 0 <= N_val <= 100
                    else "N"
                )

                rh_label = (
                    "Y"
                    if 0 <= rh_val <= 95
                    else "N"
                )

                label = (
                    "Y"
                    if (
                        lwc_label == "Y"
                        and N_label == "Y"
                        and rh_label == "Y"
                    )
                    else "N"
                )

                data_labels.append(label)

                for bin_label in fcdp_bin_labels:

                    bin_key = (
                        f"Bin{bin_label:03d}_{label}_mean"
                    )

                    bin_means[bin_key].append(
                        bins[
                            f"FCDP_Bin{bin_label:03d}"
                        ][fcdp_idx]
                    )

            leg_info_FCDP.append({
                "Date": date,
                "BCB_start": start20,
                "BCB_stop": end20,
                "Data_Labels": data_labels
            })
        for bin_label in fcdp_bin_labels:

            for label in ["Y", "N"]:

                bin_key = (
                    f"Bin{bin_label:03d}_{label}_mean"
                )

                if len(bin_means[bin_key]) > 0:
                    bin_means[bin_key] = np.nanmean(
                        bin_means[bin_key]
                    )
                else:
                    bin_means[bin_key] = np.nan

        total_BCB_means.append(bin_means)
    master_FCDP_BCB.append(total_BCB_means)
for flight in master_FCDP_BCB:
    for bin_means in flight:
        print(
            f"Date: {bin_means['Date']}, "
            f"Start: {bin_means['BCB_start']}, "
            f"Stop: {bin_means['BCB_stop']}"
        )
        for bin_label in fcdp_bin_labels:

            for label in ["Y", "N"]:

                bin_key = (
                    f"Bin{bin_label:03d}_{label}_mean"
                )

                print(
                    f"   {bin_key}: "
                    f"{bin_means[bin_key]}"
                )
#%%
# Count the total number of legs from master_FCDP_BCB
total_legs_FCDP = sum(len(item) for item in master_FCDP_BCB)
print(f"Total number of legs: {total_legs_FCDP}")
#%%
Y_BCB_calc_FCDP = []
N_BCB_calc_FCDP = []

fcdp_bin_labels = range(3, 21)  # 003 through 020

for flight_data in master_FCDP_BCB:
    for bin_means_FCDP in flight_data:

        Y_calc = {
            "Date": bin_means_FCDP["Date"],
            "BCB_start": bin_means_FCDP["BCB_start"],
            "BCB_stop": bin_means_FCDP["BCB_stop"]
        }
        N_calc = {
            "Date": bin_means_FCDP["Date"],
            "BCB_start": bin_means_FCDP["BCB_start"],
            "BCB_stop": bin_means_FCDP["BCB_stop"]
        }
        # actual_bin_label = 003–020
        # width_index = 0–17 for Logg_FCDP
        for width_index, bin_label in enumerate(fcdp_bin_labels):

            bin_key_Y = f"Bin{bin_label:03d}_Y_mean"
            bin_key_N = f"Bin{bin_label:03d}_N_mean"

            Y_calc[bin_key_Y] = (
                bin_means_FCDP[bin_key_Y]
                * Logg_FCDP[width_index]
                / 1e6
            )

            N_calc[bin_key_N] = (
                bin_means_FCDP[bin_key_N]
                * Logg_FCDP[width_index]
                / 1e6
            )

        Y_BCB_calc_FCDP.append(Y_calc)
        N_BCB_calc_FCDP.append(N_calc)
# %%
plt.figure(figsize=(8, 6))
for entry in Y_BCB_calc_FCDP:
    bin_means = np.array([entry.get(f'Bin{i:03d}_Y_mean', np.nan) for i in fcdp_bin_labels], dtype=float)  
    valid_indices = ~np.isnan(bin_means)  
    bin_centers_valid = np.array(bin_center_FCDP)[valid_indices]
    bin_means_valid = bin_means[valid_indices]

    plt.plot(bin_centers_valid, bin_means_valid, color='black', alpha=0.5)
plt.xlabel("Deliquesed Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"FCDP Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-7, 10**1)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base January-June 2022\n Raw Ambient Size Distributions", fontsize=14, fontweight="bold")
plt.show()
#%%
#Filtering the 0s
plt.figure(figsize=(8, 6))
for entry in Y_BCB_calc_FCDP:
    bin_means = np.array([entry.get(f'Bin{i:03d}_Y_mean', np.nan) for i in fcdp_bin_labels], dtype=float)  
    bin_centers = np.array(bin_center_FCDP)
    valid_indices = (bin_means > 0) & ~np.isnan(bin_means)  
    bin_centers_valid = bin_centers[valid_indices]
    bin_means_valid = bin_means[valid_indices]
    if len(bin_centers_valid) > 0:  
        plt.plot(bin_centers_valid, bin_means_valid, color='black', alpha=0.5)
plt.xlabel("Deliquesced Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"FCDP Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-7, 10**1)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base January-June 2022\n Raw Ambient Size Distributions", fontsize=14, fontweight="bold")
plt.show()
#%%
#average distribution
sum_bin_means = np.zeros(len(bin_center_FCDP))
count_bin_means = np.zeros(len(bin_center_FCDP))
for entry in Y_BCB_calc_FCDP:
    bin_means = np.array([entry.get(f'Bin{i:03d}_Y_mean', np.nan) for i in fcdp_bin_labels], dtype=float)
    valid_indices = (bin_means > 0) & ~np.isnan(bin_means)
    sum_bin_means[valid_indices] += bin_means[valid_indices]
    count_bin_means[valid_indices] += 1
average_bin_means = np.divide(sum_bin_means, count_bin_means, where=count_bin_means > 0)
plt.figure(figsize=(8, 6))
plt.plot(bin_center_FCDP, average_bin_means, color='red', linewidth=2, label='Average Size Distribution')
plt.xlabel("Deliquesced Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"FCDP Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=19, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xlim(0, 45)
plt.xticks(fontweight="bold", fontsize=18)
plt.yticks(fontweight="bold", fontsize=18)
plt.title("FCDP Average Ambient \nBelow Cloud Base Size Distribution\n January-June 2022", fontsize=19, fontweight="bold")
plt.show()
#%%
#Fitting an exponential to each size distribution
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)
FCDP_fits = []
plt.figure(figsize=(8, 6))
for entry in Y_BCB_calc_FCDP:
    bin_means_FCDP = np.array([entry.get(f'Bin{i:03d}_Y_mean', np.nan) for i in fcdp_bin_labels], dtype=float)
    valid_indices = ~np.isnan(bin_means_FCDP)
    bin_centers_valid = np.array(bin_center_FCDP)[valid_indices]
    bin_means_valid = bin_means_FCDP[valid_indices]
    if not valid_indices.any():
        print(f"No valid data for date {entry['Date']}")
        continue
    try:
        popt, _ = curve_fit(exponential, bin_centers_valid, bin_means_valid, p0=(1, 1), maxfev=5000)
        n0, D = popt

        FCDP_fits.append({
            'Date': entry['Date'],
            'BCB_start': entry.get('BCB_start', np.nan),
            'BCB_stop': entry.get('BCB_stop', np.nan),
            'Intercept_n0': n0,
            'E_folding_D': D
        })

        x_fit = np.linspace(min(bin_centers_valid), max(bin_centers_valid), 100)
        y_fit = exponential(x_fit, *popt)

        plt.plot(x_fit, y_fit, color='black', alpha=0.2)

    except RuntimeError:
        print(f"Fit could not be performed for date {entry['Date']}")
plt.xlabel("Deliquesed Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"FCDP Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base January-June 2022\n Exponential Fit to Ambient Size Distributions", fontsize=14, fontweight="bold")
plt.show()
#%%
#find why we have so much empty data 
fcdp_bin_labels = list(range(3, 21))
FCDP_fit_input_diagnostics = []
for entry in Y_BCB_calc_FCDP:
    values = np.array(
        [entry.get(
                f"Bin{bin_label:03d}_Y_mean",
                np.nan)
            for bin_label in fcdp_bin_labels],
        dtype=float)
    finite_mask = np.isfinite(values)
    positive_mask = finite_mask & (values > 0)
    zero_mask = finite_mask & (values == 0)
    negative_mask = finite_mask & (values < 0)
    FCDP_fit_input_diagnostics.append({
        "Date": entry["Date"],
        "BCB_start": entry["BCB_start"],
        "BCB_stop": entry["BCB_stop"],
        "Finite_bins": int(finite_mask.sum()),
        "Positive_bins": int(positive_mask.sum()),
        "Zero_bins": int(zero_mask.sum()),
        "Negative_bins": int(negative_mask.sum()),
        "NaN_bins": int(np.isnan(values).sum()),
        "Minimum_positive": (
            np.min(values[positive_mask])
            if positive_mask.any()
            else np.nan ),
        "Maximum": (
            np.nanmax(values)
            if finite_mask.any()
            else np.nan)})
FCDP_fit_input_diagnostics_df = pd.DataFrame(
    FCDP_fit_input_diagnostics)
print(
    FCDP_fit_input_diagnostics_df[
        FCDP_fit_input_diagnostics_df["Finite_bins"] == 0
    ].to_string(index=False))
print("\nReason counts:")
print(
    FCDP_fit_input_diagnostics_df[
        [
            "Finite_bins",
            "Positive_bins",
            "Zero_bins",
            "NaN_bins"
        ]].describe())
#%%
#remove the NAN legs 
fcdp_bin_labels = list(range(3, 21))
filtered_master_FCDP_BCB = []
removed_empty_FCDP_legs = []
for flight_data in master_FCDP_BCB:
    filtered_flight = []
    for leg in flight_data:
        Y_values = np.array(
            [
                leg[f"Bin{bin_label:03d}_Y_mean"]
                for bin_label in fcdp_bin_labels],
            dtype=float)
        empty_leg = np.all(
            np.isnan(Y_values) | (Y_values == 0))

        if empty_leg:
            removed_empty_FCDP_legs.append({
                "Date": leg["Date"],
                "BCB_start": leg["BCB_start"],
                "BCB_stop": leg["BCB_stop"]})

            continue
        filtered_flight.append(leg)
    filtered_master_FCDP_BCB.append(filtered_flight)
#%%
master_FCDP_BCB = filtered_master_FCDP_BCB
#%%
#Trying to fit and stop at 10um 
bin_center_FCDPs=np.array(bin_center_FCDP)
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)
FCDP_fits_10 = []
plt.figure(figsize=(8, 6))
for entry in Y_BCB_calc_FCDP:
    bin_means_FCDP = np.array([entry.get(f'Bin{i:03d}_Y_mean', np.nan) for i in range(3, 21)], dtype=float)
    valid_indices = (bin_center_FCDPs <= 10) & ~np.isnan(bin_means_FCDP)
    bin_centers_valid = np.array(bin_center_FCDP)[valid_indices]
    bin_means_valid = bin_means_FCDP[valid_indices]
    if valid_indices.any():
        try:
            popt, _ = curve_fit(exponential, bin_centers_valid, bin_means_valid, 
                                p0=(1, 1), maxfev=5000, 
                                bounds=([0, 0.1], [np.inf, 20])) 
            n0, D = popt

            if D > 20:
                print(f"High slope detected! Date: {entry['Date']}, D: {D:.2f}")

        except RuntimeError:
            print(f"Fit failed for date {entry['Date']}")
            n0, D = np.nan, np.nan 
    else:
        n0, D = np.nan, np.nan 
    FCDP_fits_10.append({
        'Date': entry['Date'],
        'BCB_start': entry.get('BCB_start', np.nan),
        'BCB_stop': entry.get('BCB_stop', np.nan),
        'Intercept_n0': n0,
        'E_folding_D': D
    })

    if not np.isnan(n0) and not np.isnan(D):
        x_fit = np.linspace(min(bin_centers_valid), 10, 100)
        y_fit = exponential(x_fit, *popt)
        plt.plot(x_fit, y_fit, color='black', alpha=0.2)
plt.xlabel("Deliquesed Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"FCDP Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
# plt.ylim(10**-33, 10**1)
plt.ylim(10**-7, 10**1)
plt.xlim(0, 10)
plt.title("FCDP Below Cloud Base January-June 2022\n Exponential Fit Ambient Size Distributions (≤10 µm)", fontsize=14, fontweight="bold")
plt.show()
print(f"Total successful ambient exponential fits: {np.sum(~np.isnan([fit['E_folding_D'] for fit in FCDP_fits_10]))}")
#%%
#Calculating total number concentration 
Y_BCB_calc_cm3_FCDP = []
N_BCB_calc_cm3_FCDP = []

for flight_data in master_FCDP_BCB:
    for bin_means_FCDP in flight_data:
        Y_calc_FCDP = {
            'Date': bin_means_FCDP['Date'],
            'BCB_start': bin_means_FCDP['BCB_start'],
            'BCB_stop': bin_means_FCDP['BCB_stop']
        }
        N_calc_FCDP = {
            'Date': bin_means_FCDP['Date'],
            'BCB_start': bin_means_FCDP['BCB_start'],
            'BCB_stop': bin_means_FCDP['BCB_stop']
        }
        for log_index, bin_label in enumerate(range(3, 21)):
            bin_key_Y = f'Bin{bin_label:03d}_Y_mean'
            bin_key_N = f'Bin{bin_label:03d}_N_mean'
            Y_calc_FCDP[bin_key_Y] = (
                bin_means_FCDP[bin_key_Y]
                * bin_log_FCDP[log_index]
                / 1e6)
            N_calc_FCDP[bin_key_N] = (
                bin_means_FCDP[bin_key_N]
                * bin_log_FCDP[log_index]
                / 1e6)
        Y_BCB_calc_cm3_FCDP.append(Y_calc_FCDP)
        N_BCB_calc_cm3_FCDP.append(N_calc_FCDP)
# %%
#Calculating total number concentration 
total_concentration_cm3 = []
for entry in Y_BCB_calc_cm3_FCDP:
    total_Y_concentration_FCDP = np.nansum([entry[f'Bin{i:03d}_Y_mean'] for i in range(3,21)])  # Sum all valid bin concentrations
    total_concentration_cm3.append({
        'Date': entry['Date'],
        'BCB_start': entry['BCB_start'],
        'BCB_stop': entry['BCB_stop'],
        'Total_Y_Concentration_cm3': total_Y_concentration_FCDP
    })
#%%
total_Y_concentrations_FCDP = [entry['Total_Y_Concentration_cm3'] for entry in total_concentration_cm3]
total_Y_concentrations_FCDP = [conc for conc in total_Y_concentrations_FCDP if not np.isnan(conc)]
mean_total_concentration = np.mean(total_Y_concentrations_FCDP)
print(f"Mean Total Number Concentration: {mean_total_concentration:.2f} cm⁻³")
#%%
#Calculate the relative humidity of each leg.
master_BCB_RH = []
for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict = leg_data[i]
    flight_date = leg_dict['Date']  # Get date of flight from dictionary 
    BCB_start = leg_dict['LegIndex_02']['StartTimes']
    BCB_stop = leg_dict['LegIndex_02']['StopTimes']
    rh_flight = h20[i]
    times_rh = rh_flight.Time_Start.values
    rh_values = rh_flight.RHw_DLH.values
    all_BCB = []
    for j in range(len(BCB_start)):
        start = int(BCB_start[j])
        end = int(BCB_stop[j])
        rh_times = {
            'Date': date,
            'BCB_start': start,
            'BCB_stop': end,
            'Rh_mean': [],
        }
        index1_start = None
        for k in range(len(times_rh)):
            if times_rh[k] == start:
                index1_start = k
                break
        index1_end = None
        for k in range(len(times_rh)):
            if times_rh[k] == end:
                index1_end = k
                break
        if index1_start is None or index1_end is None:
            rh9_mean = np.nan
        else:
            rh9 = rh_values[index1_start:index1_end + 1]
            rh9 = rh9[(rh9 <= 95) & (rh9 > 0)]  # filter for RH ≤ 95 and ignore missing/bad (-999 or 0)
            rh9_mean = np.nanmean(rh9)
        rh_times['Rh_mean'].append(rh9_mean)
        all_BCB.append(rh_times)  # List that contains all the BCB wind/alt mean dictionaries for 1 flight
    master_BCB_RH.append(all_BCB)
for flight in master_BCB_RH:
    for leg in flight:
        rh_mean_list = leg['Rh_mean']
        leg['Rh_mean'] = [np.nan if value <=0 else value for value in rh_mean_list]

#%%
date_leg_set = set()
for flight in master_FCDP_BCB:  # this should now be the filtered master
    for leg in flight:
        date_leg_set.add((
            leg['Date'],
            float(leg['BCB_start']),
            float(leg['BCB_stop'])
        ))

filtered_master_BCB_RH_FCDP = []
for flight in master_BCB_RH:
    filtered_legs = []
    for leg in flight:
        date = leg['Date']
        BCB_start = leg['BCB_start']
        BCB_stop = leg['BCB_stop']
        if (date, BCB_start, BCB_stop) in date_leg_set:
            filtered_legs.append(leg)
    if filtered_legs:
        filtered_master_BCB_RH_FCDP.append(filtered_legs)
#%%
#Make sure the leg counts 
# Flatten the list of lists and count the total number of entries
total_entries_filtered_master_BCB_RH_FCDP = sum(len(legs) for legs in filtered_master_BCB_RH_FCDP)
print(f"Total entries in filtered_master_BCB_RH_FCDP: {total_entries_filtered_master_BCB_RH_FCDP}")
# %%
##Obtaining g(RH) = [1.7 / (1-RH)]^0.31 for all mean RH values 
master_BCB_gRH = []
removed_gRH_legs = []
for flight in filtered_master_BCB_RH_FCDP:
    flight_gRH = []
    for leg in flight:
        rh_value = leg["Rh_mean"][0]
        if not np.isfinite(rh_value):
            removed_gRH_legs.append({
                "Date": leg["Date"],
                "BCB_start": leg["BCB_start"],
                "BCB_stop": leg["BCB_stop"],
                "Rh_mean": rh_value})
            continue
        rh_fraction = rh_value / 100.0
        if rh_fraction >= 1:
            removed_gRH_legs.append({
                "Date": leg["Date"],
                "BCB_start": leg["BCB_start"],
                "BCB_stop": leg["BCB_stop"],
                "Rh_mean": rh_value})
            continue
        new_leg = leg.copy()
        gRH_value = (1.7 / (1 - rh_fraction)) ** 0.31
        new_leg["gRh_mean"] = [gRH_value]
        flight_gRH.append(new_leg)
    if flight_gRH:
        master_BCB_gRH.append(flight_gRH)
#%%
valid_gRH_leg_set = {
    (
        leg["Date"],
        float(leg["BCB_start"]),
        float(leg["BCB_stop"])
    )
    for flight in master_BCB_gRH
    for leg in flight
}

filtered_total_concentration_cm3 = [
    entry
    for entry in total_concentration_cm3
    if (
        entry["Date"],
        float(entry["BCB_start"]),
        float(entry["BCB_stop"])
    ) in valid_gRH_leg_set
]

print(
    "Concentration legs before RH removal:",
    len(total_concentration_cm3)
)

print(
    "Concentration legs after RH removal:",
    len(filtered_total_concentration_cm3)
)
# %%
#only the grh from filtered_master_BCB_RH
filtered_master_BCB_gRH_FCDP = []
for flight in filtered_master_BCB_RH_FCDP:
    flight_gRH = []
    for leg in flight:
        new_leg = leg.copy() 
        rh_mean = new_leg['Rh_mean'][0] / 100.0  # Convert percentage to a decimal
        if np.isnan(rh_mean) or rh_mean >= 1:
            gRH_value = np.nan
            print(f"Skipping calculation for Rh_mean = {new_leg['Rh_mean'][0]} as it results in division by zero or invalid value.")
        else:
            gRH_value = (1.7 / (1 - rh_mean)) ** 0.31
        new_leg['gRh_mean'] = [gRH_value]
        flight_gRH.append(new_leg)
    filtered_master_BCB_gRH_FCDP.append(flight_gRH)
#%%
total_entries_filtered_master_BCB_gRH_FCDP = sum(len(legs) for legs in filtered_master_BCB_gRH_FCDP)
print(f"Total entries in filtered_master_BCB_gRH_FCDP: {total_entries_filtered_master_BCB_gRH_FCDP}")
# %%
filtered_master_BCB_ddry_FCDP = []

for flight in filtered_master_BCB_gRH_FCDP:
    for entry in flight:

        date = entry['Date']
        BCB_start = entry['BCB_start']
        BCB_stop = entry['BCB_stop']
        gRh_mean = entry['gRh_mean'][0]

        if gRh_mean > 0:
            ddry_values = np.array([
                D_amb / gRh_mean for D_amb in bin_center_FCDP
            ])
        else:
            ddry_values = np.full(len(bin_center_FCDP), np.nan)
            print(
                f"Skipping division for {date}, "
                f"{BCB_start}-{BCB_stop} due to invalid gRh_mean."
            )

        ddry_bin_widths = np.diff(ddry_values, append=np.nan)

        raw_concentrations = next(
            (
                leg for leg in Y_BCB_calc_FCDP
                if leg['Date'] == date
                and leg['BCB_start'] == BCB_start
                and leg['BCB_stop'] == BCB_stop
            ),
            None
        )

        if raw_concentrations is not None:
            dN_dD_ambient = np.array([
                raw_concentrations.get(
                    f'Bin{i:03d}_Y_mean',
                    np.nan
                )
                for i in range(3, 21)
            ], dtype=float)

            dN_dD_dry = np.where(
                (~np.isnan(dN_dD_ambient))
                & (~np.isnan(ddry_bin_widths))
                & (gRh_mean > 0),

                dN_dD_ambient
                * (np.array(bin_center_FCDP) / ddry_values)
                * (
                    np.diff(bin_center_FCDP, append=np.nan)
                    / ddry_bin_widths
                ),

                np.nan
            )

        else:
            dN_dD_dry = np.full(
                len(bin_center_FCDP),
                np.nan
            )

            print(
                f"Missing raw size distribution for "
                f"{date}, {BCB_start}-{BCB_stop}"
            )

        filtered_master_BCB_ddry_FCDP.append({
            'Date': date,
            'BCB_start': BCB_start,
            'BCB_stop': BCB_stop,
            'ddry': ddry_values.tolist(),
            'dN/dDdry': dN_dD_dry.tolist(),
            'ddry_bin_widths': ddry_bin_widths.tolist(),
            'gRh_mean': gRh_mean
        })

print(
    "Length of filtered_master_BCB_ddry_FCDP:",
    len(filtered_master_BCB_ddry_FCDP)
)
#%%
from scipy.interpolate import interp1d
common_bins = np.linspace(2, 25, 35)
plt.figure(figsize=(8, 6))
for entry in filtered_master_BCB_ddry_FCDP:
    ddry_values = np.array(entry['ddry']) 
    dN_dD_dry = np.array(entry['dN/dDdry'])
    valid_indices = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry)
    if np.sum(valid_indices) < 2:
        continue  
    interp_func = interp1d(ddry_values[valid_indices], dN_dD_dry[valid_indices], 
                           kind='linear', bounds_error=False, fill_value=np.nan)
    interpolated_dN_dD_dry = interp_func(common_bins)
    plt.plot(common_bins, interpolated_dN_dD_dry, color='black', alpha=0.2)
plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base January-June 2022\n Raw Dry Size Distributions", fontsize=14, fontweight="bold")
plt.show()
#%%%
#Removing the 0s
common_bins = np.linspace(2, 25, 35) 
plt.figure(figsize=(8, 6))
for entry in filtered_master_BCB_ddry_FCDP:
    ddry_values = np.array(entry['ddry'])  
    dN_dD_dry = np.array(entry['dN/dDdry']) 
   
    valid_indices = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry)
    if np.sum(valid_indices) < 2:
        continue 
    interp_func = interp1d(ddry_values[valid_indices], dN_dD_dry[valid_indices], 
                           kind='linear', bounds_error=False, fill_value=np.nan)
    interpolated_dN_dD_dry = interp_func(common_bins)
    valid_interpolated_indices = (interpolated_dN_dD_dry > 0) & ~np.isnan(interpolated_dN_dD_dry)
    filtered_bins = common_bins[valid_interpolated_indices]
    filtered_dN_dD_dry = interpolated_dN_dD_dry[valid_interpolated_indices]
    if len(filtered_bins) > 0: 
        plt.plot(filtered_bins, filtered_dN_dD_dry, color='purple', alpha=0.2)
plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=20, fontweight="bold")
plt.ylabel("FCDP Number Concentration\n(cm$^{-3}$ μm$^{-1}$)", fontsize=20, fontweight="bold")
plt.yscale("log")
plt.xticks(fontweight="bold", fontsize=20)
plt.yticks(fontweight="bold", fontsize=20)
plt.ylim(10**-7, 10**1.5)
plt.xlim(0.5, 40)
plt.title("FCDP Below Cloud Base\n January-June 2022\n Raw Dry Size Distributions", fontsize=20, fontweight="bold")
plt.show()
#%%
#average dry distribution ********
common_bins = np.linspace(2, 25, 35)  
sum_interpolated_dN_dD_dry = np.zeros_like(common_bins, dtype=float)
count_interpolated_dN_dD_dry = np.zeros_like(common_bins, dtype=int)
for entry in filtered_master_BCB_ddry_FCDP:
    ddry_values = np.array(entry['ddry'])  
    dN_dD_dry = np.array(entry['dN/dDdry'])
    valid_indices = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry)
    if np.sum(valid_indices) < 2:
        continue 
    interp_func = interp1d(ddry_values[valid_indices], dN_dD_dry[valid_indices], 
                           kind='linear', bounds_error=False, fill_value=np.nan)
    interpolated_dN_dD_dry = interp_func(common_bins)
    valid_interpolated_indices = ~np.isnan(interpolated_dN_dD_dry)
    sum_interpolated_dN_dD_dry[valid_interpolated_indices] += interpolated_dN_dD_dry[valid_interpolated_indices]
    count_interpolated_dN_dD_dry[valid_interpolated_indices] += 1
average_dN_dD_dry = np.divide(sum_interpolated_dN_dD_dry, count_interpolated_dN_dD_dry, where=count_interpolated_dN_dD_dry > 0)
plt.figure(figsize=(8, 6))
plt.plot(common_bins, average_dN_dD_dry, color='red', linewidth=2, label='Average Dry Size Distribution')
plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=20, fontweight="bold")
plt.ylabel(r"FCDP Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=19, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xlim(0, 45)
plt.xticks(fontweight="bold", fontsize=20)
plt.yticks(fontweight="bold", fontsize=20)
plt.title("FCDP Average Below Cloud Base \nDry Size Distribution\n January-June 2022", fontsize=20, fontweight="bold")
plt.legend()
plt.show()
#%%
# Average FCDP dry distribution with ±2 SEM ********
common_bins = np.linspace(2, 25, 35)
sum_interpolated_dN_dD_dry = np.zeros_like(common_bins, dtype=float)
count_interpolated_dN_dD_dry = np.zeros_like(common_bins, dtype=int)
all_interpolated_dN_dD_dry = []

for entry in filtered_master_BCB_ddry_FCDP:

    ddry_values = np.asarray(entry['ddry'], dtype=float)
    dN_dD_dry = np.asarray(entry['dN/dDdry'], dtype=float)

    valid_indices = (
        np.isfinite(ddry_values) &
        np.isfinite(dN_dD_dry)
    )

    if np.sum(valid_indices) < 2:
        continue

    ddry_valid = ddry_values[valid_indices]
    dN_dD_dry_valid = dN_dD_dry[valid_indices]
    sort_idx = np.argsort(ddry_valid)

    ddry_valid = ddry_valid[sort_idx]
    dN_dD_dry_valid = dN_dD_dry_valid[sort_idx]

    interp_func = interp1d(
        ddry_valid,
        dN_dD_dry_valid,
        kind='linear',
        bounds_error=False,
        fill_value=np.nan
    )

    interpolated_dN_dD_dry = interp_func(common_bins)
    all_interpolated_dN_dD_dry.append(
        interpolated_dN_dD_dry
    )

    valid_interpolated_indices = np.isfinite(
        interpolated_dN_dD_dry
    )

    sum_interpolated_dN_dD_dry[
        valid_interpolated_indices
    ] += interpolated_dN_dD_dry[
        valid_interpolated_indices
    ]

    count_interpolated_dN_dD_dry[
        valid_interpolated_indices
    ] += 1
average_dN_dD_dry = np.full_like(
    common_bins,
    np.nan,
    dtype=float
)

np.divide(
    sum_interpolated_dN_dD_dry,
    count_interpolated_dN_dD_dry,
    out=average_dN_dD_dry,
    where=count_interpolated_dN_dD_dry > 0)
all_interpolated_dN_dD_dry = np.asarray(
    all_interpolated_dN_dD_dry,
    dtype=float)
std_dN_dD_dry = np.nanstd(
    all_interpolated_dN_dD_dry,
    axis=0,
    ddof=1)
sem_dN_dD_dry = np.full_like(
    common_bins,
    np.nan,
    dtype=float)
valid_sem = count_interpolated_dN_dD_dry > 1

sem_dN_dD_dry[valid_sem] = (
    std_dN_dD_dry[valid_sem] /
    np.sqrt(count_interpolated_dN_dD_dry[valid_sem]))
two_sem_dN_dD_dry = 2 * sem_dN_dD_dry
plot_mask = (
    np.isfinite(average_dN_dD_dry) &
    np.isfinite(two_sem_dN_dD_dry) &
    (average_dN_dD_dry > 0) &
    (count_interpolated_dN_dD_dry > 1))
x_plot = common_bins[plot_mask]
mean_plot = average_dN_dD_dry[plot_mask]
two_sem_plot = two_sem_dN_dD_dry[plot_mask]
small_positive_value = 1e-12

lower_error = np.minimum(
    two_sem_plot,
    mean_plot - small_positive_value)
lower_error = np.clip(
    lower_error,
    0,
    None)
upper_error = two_sem_plot

asymmetric_yerr = np.vstack([
    lower_error,
    upper_error
])
plt.figure(figsize=(8, 6))
plt.errorbar(
    x_plot,
    mean_plot,
    yerr=asymmetric_yerr,
    fmt='-',
    color='green',
    ecolor='green',
    linewidth=2,
    elinewidth=1.5,
    capsize=4,
    capthick=1.5,
    label=r'FCDP $\pm$2 SEM (~95% CI)')
plt.xlabel(
    "Dry Bin Center Diameter (μm)",
    fontsize=20,
    fontweight="bold")
plt.ylabel(
    r"FCDP Number Concentration "
    r"(cm$^{-3}$ $\mu$m$^{-1}$)",
    fontsize=19,
    fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xlim(0, 25)
plt.xticks(
    fontweight="bold",
    fontsize=20)
plt.yticks(
    fontweight="bold",
    fontsize=20)
plt.title(
    "FCDP Average Below Cloud Base\n"
    "Dry Size Distribution\n"
    "January–June 2022",
    fontsize=20,fontweight="bold")
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()
#%%
# Preserve FCDP results
common_bins_FCDP = common_bins.copy()
average_dN_dD_dry_FCDP = average_dN_dD_dry.copy()
std_dN_dD_dry_FCDP = std_dN_dD_dry.copy()
sem_dN_dD_dry_FCDP = sem_dN_dD_dry.copy()
two_sem_dN_dD_dry_FCDP = two_sem_dN_dD_dry.copy()
count_interpolated_dN_dD_dry_FCDP = (
    count_interpolated_dN_dD_dry.copy()
)
#%%
#save the average distribution
lower_2SEM = average_dN_dD_dry - two_sem_dN_dD_dry
upper_2SEM = average_dN_dD_dry + two_sem_dN_dD_dry
average_dry_distribution = pd.DataFrame({
    'Dry_Diameter_um': common_bins,
    'Average_dN_dD_dry': average_dN_dD_dry,
    'Standard_Deviation': std_dN_dD_dry,
    'SEM': sem_dN_dD_dry,
    'Two_SEM': two_sem_dN_dD_dry,
    'Lower_2SEM': lower_2SEM,
    'Upper_2SEM': upper_2SEM,
    'N_profiles': count_interpolated_dN_dD_dry
})
save_dir = "/home/disk/eos4/kathem24/activate/data/2022/FCDP"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(
    save_dir,
    "Average_Dry_Size_Distribution_errorbars2022FCDP.csv")
average_dry_distribution.to_csv(save_path, index=False)
print(f"Saved to: {save_path}")
print(average_dry_distribution.head())
# %%
#CDP
master_CDP_BCB = []
leg_info_CDP = []
for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict_CDP = leg_data[i]
    flight_date = leg_dict_CDP['Date']
    BCB_start = leg_dict_CDP['LegIndex_02']['StartTimes']
    BCB_stop = leg_dict_CDP['LegIndex_02']['StopTimes']
    CDP_flight = CDP_1Hz[i]
    twoDS_flight = twoDS[i]
    rh_flight = h20[i]
    CDP_flight['Time_Start'] = pd.to_numeric(CDP_flight['Time_Start'], errors='coerce')
    twoDS_flight['Time_Start'] = pd.to_numeric(twoDS_flight['Time_Start'], errors='coerce')
    rh_flight['Time_Start'] = pd.to_numeric(rh_flight['Time_Start'], errors='coerce')
    CDP_times = CDP_flight['Time_Start'].values
    CDP_lwc = CDP_flight['LWC_CDP'].values
    CDP_bins = {f'CDP_Bin{bin_label:02d}': CDP_flight[f'CDP_Bin{bin_label:02d}'].values for bin_label in range(30)}
    TwoDS_times = twoDS_flight['Time_Start'].values
    TwoDS_N_total = twoDS_flight['N-total_2DS'].values
    rh_times = rh_flight['Time_Start'].values
    rh_values = rh_flight.RHw_DLH.values
    total_BCB_means_CDP = []
    for k in range(len(BCB_start)):
        start20 = BCB_start[k]
        end20 = BCB_stop[k]

        CDP_indices_in_range = np.where((CDP_times >= start20) & (CDP_times <= end20))[0]
        TwoDS_indices_in_range = np.where((TwoDS_times >= start20) & (TwoDS_times <= end20))[0]
        rh_indices_in_range = np.where((rh_times >= start20) & (rh_times <= end20))[0]

        if CDP_indices_in_range.size > 0 and TwoDS_indices_in_range.size > 0 and rh_indices_in_range.size > 0:
            data_labels_CDP = []
            BCB_means_CDP = []

            for CDP_idx, TwoDS_idx, rh_idx in zip(CDP_indices_in_range, TwoDS_indices_in_range, rh_indices_in_range):
                lwc_val = CDP_lwc[CDP_idx]
                N_val = TwoDS_N_total[TwoDS_idx]
                rh_val = rh_values[rh_idx]

                lwc_label = 'Y' if 0 <= lwc_val <= 0.0025 else 'N'
                N_label = 'Y' if 0 <= N_val <= 100 else 'N'
                rh_label = 'Y' if 0 <= rh_val <= 95 else 'N'
                label = 'Y' if lwc_label == 'Y' and N_label == 'Y' and rh_label == 'Y' else 'N'
                if label == 'Y' and rh_val > 95:
                    print(f" RH violation: {rh_val:.2f} passed at time {CDP_times[CDP_idx]}")

                data_labels_CDP.append(label)

                bin_values_CDP = [CDP_bins[f'CDP_Bin{bin_label:02d}'][CDP_idx] for bin_label in range(30)]
                BCB_means_CDP.append(bin_values_CDP)

            if BCB_means_CDP:
                total_BCB_means_CDP.append(BCB_means_CDP)

            leg_info_CDP.append({
                'Date': date,
                'BCB_start': start20,
                'BCB_stop': end20,
                'Data_Labels': data_labels_CDP,
            })
    master_CDP_BCB.append(total_BCB_means_CDP)
for leg in leg_info_CDP:
    print(f"Date: {leg['Date']}, Start: {leg['BCB_start']}, Stop: {leg['BCB_stop']}, Data Labels: {leg['Data_Labels']}")
#%%
#Double check the number of legs associated with each date to compare with the CAS legs 
leg_count = Counter([leg['Date'] for leg in leg_info_CDP])
print("Number of legs associated with each date:")
for date, count in sorted(leg_count.items()):
    print(f"Date: {date}, Number of Legs: {count}")
#%%
master_CDP_BCB = []
leg_info_CDP = []
for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict_CDP = leg_data[i]
    flight_date = leg_dict_CDP['Date']
    BCB_start = leg_dict_CDP['LegIndex_02']['StartTimes']
    BCB_stop = leg_dict_CDP['LegIndex_02']['StopTimes']
    CDP_flight = CDP_1Hz[i]
    twoDS_flight = twoDS[i]
    rh_flight = h20[i]
    CDP_times = np.array(CDP_flight['Time_Start'], dtype=float)
    TwoDS_times = np.array(twoDS_flight['Time_Start'], dtype=float)
    rh_times = np.array(rh_flight['Time_Start'], dtype=float)
    lwc = np.array(CDP_flight['LWC_CDP'], dtype=float)
    N_total = np.array(twoDS_flight['N-total_2DS'], dtype=float)
    rh_total = np.array(rh_flight['RHw_DLH'], dtype=float)

    bins = {
        f'CDP_Bin{bin_label:02d}': np.array(CDP_flight[f'CDP_Bin{bin_label:02d}'], dtype=float)
        for bin_label in range(30)
    }

    total_BCB_means_CDP = []
    for k in range(len(BCB_start)):
        start20 = BCB_start[k]
        end20 = BCB_stop[k]

        bin_means_CDP = {f'Bin{bin_label:02d}_Y_mean': [] for bin_label in range(30)}
        bin_means_CDP.update({f'Bin{bin_label:02d}_N_mean': [] for bin_label in range(30)})
        bin_means_CDP.update({'Date': date, 'BCB_start': start20, 'BCB_stop': end20})

        CDP_indices_in_range = np.where((CDP_times >= start20) & (CDP_times <= end20))[0]
        TwoDS_indices_in_range = np.where((TwoDS_times >= start20) & (TwoDS_times <= end20))[0]
        rh_indices_in_range = np.where((rh_times >= start20) & (rh_times <= end20))[0]
        if CDP_indices_in_range.size > 0 and TwoDS_indices_in_range.size > 0 and rh_indices_in_range.size >0:
            for cdp_idx, twods_idx, rh_idx in zip(CDP_indices_in_range, TwoDS_indices_in_range, rh_indices_in_range):
                lwc_val = lwc[cdp_idx]
                N_val = N_total[twods_idx]
                rh_val = rh_total[rh_idx]
                lwc_label = 'Y' if 0 <= lwc_val <= 0.0025 else 'N'
                N_label = 'Y' if 0 <= N_val <= 100 else 'N'
                rh_label = 'Y' if 0 <= rh_val <= 95 else 'N'
                label = 'Y' if lwc_label == 'Y' and N_label == 'Y' and rh_label == 'Y' else 'N'

                for bin_label in range(30):
                    bin_key = f'Bin{bin_label:02d}_{label}_mean'
                    bin_means_CDP[bin_key].append(bins[f'CDP_Bin{bin_label:02d}'][cdp_idx])
        for bin_label in range(30):
            for label in ['Y', 'N']:
                bin_key = f'Bin{bin_label:02d}_{label}_mean'
                if bin_means_CDP[bin_key]:
                    bin_means_CDP[bin_key] = np.nanmean(bin_means_CDP[bin_key])
        total_BCB_means_CDP.append(bin_means_CDP)
    master_CDP_BCB.append(total_BCB_means_CDP)
for item in master_CDP_BCB:
    for bin_means_CDP in item:
        print(f"Date: {bin_means_CDP['Date']}, Start: {bin_means_CDP['BCB_start']}, Stop: {bin_means_CDP['BCB_stop']}")
        for bin_label in range(30):
            for label in ['Y', 'N']:
                bin_key = f'Bin{bin_label:02d}_{label}_mean'
                print(f"   {bin_key}: {bin_means_CDP[bin_key]}")
#%%
master_CDP_BCB = []
leg_info_CDP = []
for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict_CDP = leg_data[i]
    flight_date = leg_dict_CDP['Date']
    BCB_start = leg_dict_CDP['LegIndex_02']['StartTimes']
    BCB_stop = leg_dict_CDP['LegIndex_02']['StopTimes']
    CDP_flight = CDP_1Hz[i]
    twoDS_flight = twoDS[i]
    rh_flight = h20[i]
    CDP_times = np.array(CDP_flight['Time_Start'], dtype=float)
    TwoDS_times = np.array(twoDS_flight['Time_Start'], dtype=float)
    rh_times = np.array(rh_flight['Time_Start'], dtype=float)
    lwc = np.array(CDP_flight['LWC_CDP'], dtype=float)
    N_total = np.array(twoDS_flight['N-total_2DS'], dtype=float)
    rh_total = np.array(rh_flight['RHw_DLH'], dtype=float)
    bins = {
        f'CDP_Bin{bin_label:02d}': np.array(CDP_flight[f'CDP_Bin{bin_label:02d}'], dtype=float)
        for bin_label in range(30)
    }
    total_BCB_means_CDP = []
    for k in range(len(BCB_start)):
        start20 = BCB_start[k]
        end20 = BCB_stop[k]
        bin_means_CDP = {f'Bin{bin_label:02d}_Y_mean': [] for bin_label in range(30)}
        bin_means_CDP.update({f'Bin{bin_label:02d}_N_mean': [] for bin_label in range(30)})
        bin_means_CDP.update({'Date': date, 'BCB_start': start20, 'BCB_stop': end20})
        data_labels_CDP = []
        CDP_indices_in_range = np.where((CDP_times >= start20) & (CDP_times <= end20))[0]
        TwoDS_indices_in_range = np.where((TwoDS_times >= start20) & (TwoDS_times <= end20))[0]
        rh_indices_in_range = np.where((rh_times >= start20) & (rh_times <= end20))[0]
        if CDP_indices_in_range.size > 0 and TwoDS_indices_in_range.size > 0 and rh_indices_in_range.size > 0:
            for cdp_idx, twods_idx, rh_idx in zip(CDP_indices_in_range, TwoDS_indices_in_range, rh_indices_in_range):
                lwc_val = lwc[cdp_idx]
                N_val = N_total[twods_idx]
                rh_val = rh_total[rh_idx]
                lwc_label = 'Y' if 0 <= lwc_val <= 0.0025 else 'N'
                N_label   = 'Y' if 0 <= N_val <= 100 else 'N'
                rh_label  = 'Y' if 0 <= rh_val <= 95 else 'N'
                label     = 'Y' if lwc_label == 'Y' and N_label == 'Y' and rh_label == 'Y' else 'N'

                data_labels_CDP.append(label)

                for bin_label in range(30):
                    bin_key = f'Bin{bin_label:02d}_{label}_mean'
                    bin_means_CDP[bin_key].append(bins[f'CDP_Bin{bin_label:02d}'][cdp_idx])

        for bin_label in range(30):
            for label in ['Y', 'N']:
                bin_key = f'Bin{bin_label:02d}_{label}_mean'
                if bin_means_CDP[bin_key]:
                    bin_means_CDP[bin_key] = np.nanmean(bin_means_CDP[bin_key])
                else:
                    bin_means_CDP[bin_key] = np.nan

        total_BCB_means_CDP.append(bin_means_CDP)

        leg_info_CDP.append({
            'Date': date,
            'BCB_start': start20,
            'BCB_stop': end20,
            'Data_Labels': data_labels_CDP,
        })
    master_CDP_BCB.append(total_BCB_means_CDP)
for leg in leg_info_CDP[:3]:
    print(f"{leg['Date']} {leg['BCB_start']}-{leg['BCB_stop']}")
    print("First 20 labels:", leg['Data_Labels'][:20])
    print("Count Y:", leg['Data_Labels'].count('Y'))
    print("Count N:", leg['Data_Labels'].count('N'))
    print("----")

#%%
rh_Y_values_CDP = []
for leg in leg_info_CDP:
    date = leg['Date']
    start = leg['BCB_start']
    stop = leg['BCB_stop']
    flight_index = dates_legs.index(date)
    rh_flight = h20[flight_index]
    rh_times = rh_flight['Time_Start'].values
    rh_vals = rh_flight['RHw_DLH'].values
    rh_leg_indices = np.where((rh_times >= start) & (rh_times <= stop))[0]
    rh_leg_vals = rh_vals[rh_leg_indices]
    for label, rh in zip(leg['Data_Labels'], rh_leg_vals):
        if label == 'Y':
            rh_Y_values_CDP.append(rh)
plt.figure(figsize=(8,6))
plt.hist(rh_Y_values_CDP, bins=30, color='teal', edgecolor='black', alpha=0.7)
plt.axvline(95, color='red', linestyle='--', label='RH = 95% threshold')
plt.xlabel("Relative Humidity (%)")
plt.ylabel("Count")
plt.title("CDP: RH Values for Seconds Labeled 'Y'")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
Y_CDP_calc = []
N_CDP_calc = []

for flight_data in master_CDP_BCB:
    for bin_means_CDP in flight_data:
        Y_calc = {'Date': bin_means_CDP['Date'], 'BCB_start': bin_means_CDP['BCB_start'], 'BCB_stop': bin_means_CDP['BCB_stop']}
        N_calc = {'Date': bin_means_CDP['Date'], 'BCB_start': bin_means_CDP['BCB_start'], 'BCB_stop': bin_means_CDP['BCB_stop']}
        
        for bin_label in range(0, 30):
            bin_key_Y = f'Bin{bin_label:02d}_Y_mean'
            bin_key_N = f'Bin{bin_label:02d}_N_mean'

            Y_calc[bin_key_Y] = np.nanmean(bin_means_CDP[bin_key_Y]) * Logg_CDP[bin_label]
            N_calc[bin_key_N] = np.nanmean(bin_means_CDP[bin_key_N]) * Logg_CDP[bin_label]

        Y_CDP_calc.append(Y_calc)
        N_CDP_calc.append(N_calc)
# %%
# Ensure Y_BCB_calc exists before filtering
if 'Y_CDP_calc' in globals():
    BCB_leg_keys = {(entry['Date'], entry['BCB_start'], entry['BCB_stop']) for entry in Y_CDP_calc}

    filtered_Y_CDP_calc = [entry for entry in Y_CDP_calc if (entry['Date'], entry['BCB_start'], entry['BCB_stop']) in BCB_leg_keys]
    filtered_N_CDP_calc = [entry for entry in N_CDP_calc if (entry['Date'], entry['BCB_start'], entry['BCB_stop']) in BCB_leg_keys]
    print(f"Original Y_CDP_calc legs: {len(Y_CDP_calc)}")
    print(f"Filtered Y_CDP_calc legs: {len(filtered_Y_CDP_calc)}")
    print(f"Original N_CDP_calc legs: {len(N_CDP_calc)}")
    print(f"Filtered N_CDP_calc legs: {len(filtered_N_CDP_calc)}")
    Y_CDP_calc = filtered_Y_CDP_calc
    N_CDP_calc = filtered_N_CDP_calc
else:
    print("Error: Y_BCB_calc is not defined. Make sure it is loaded before running this script.")
# %%
#ambient size distribution
bin_center_CDP=np.array(bin_center_CDP)
plt.figure(figsize=(8, 6))
for entry in Y_CDP_calc:
    bin_means = np.array([entry.get(f'Bin{i:02d}_Y_mean', np.nan) for i in range(0, 30)], dtype=float)  
    valid_indices = ~np.isnan(bin_means)  
    bin_centers_valid = np.array(bin_center_CDP)[valid_indices]
    bin_means_valid = bin_means[valid_indices]
    plt.plot(bin_centers_valid, bin_means_valid, color='black', alpha=0.5)
plt.xlabel("Deliquesced Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"CDP Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-7, 10**1)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base January - June 2022\n Raw CDP Size Distributions", fontsize=14, fontweight="bold")
plt.show()
# %%
# Filtering out zero values before plotting
plt.figure(figsize=(8, 6))
for entry in Y_CDP_calc:
    bin_means = np.array([entry.get(f'Bin{i:02d}_Y_mean', np.nan) for i in range(0, 30)], dtype=float)  
    bin_centers = np.array(bin_center_CDP)
    valid_indices = (bin_means > 0) & ~np.isnan(bin_means)  
    bin_centers_valid = bin_centers[valid_indices]
    bin_means_valid = bin_means[valid_indices]
    if len(bin_centers_valid) > 0: 
        plt.plot(bin_centers_valid, bin_means_valid, color='black', alpha=0.5)
plt.xlabel("Deliquesced Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"CDP Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-7, 10**1)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base January - June 2022\n Filtered CDP Size Distributions", fontsize=14, fontweight="bold")
plt.show()
# %%
#average ambient CDP distribution
sum_bin_means_CDP = np.zeros(len(bin_center_CDP))
count_bin_means_CDP = np.zeros(len(bin_center_CDP))
for entry in Y_CDP_calc:
    bin_means_CDP = np.array([entry.get(f'Bin{i:02d}_Y_mean', np.nan) for i in range(0, 30)], dtype=float)
    valid_indices = (bin_means_CDP > 0) & ~np.isnan(bin_means_CDP)

    sum_bin_means_CDP[valid_indices] += bin_means_CDP[valid_indices]
    count_bin_means_CDP[valid_indices] += 1

average_bin_means_CDP = np.divide(sum_bin_means_CDP, count_bin_means_CDP, where=count_bin_means_CDP > 0)
plt.figure(figsize=(8, 6))
plt.plot(bin_center_CDP, average_bin_means_CDP, color='green', linewidth=2, label='Average CDP Size Distribution')
plt.xlabel("Deliquesced Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"CDP Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("CDP Average Ambient Below Cloud Base Size Distribution\n January - June 2022", fontsize=14, fontweight="bold")
plt.show()
# %%
#fitting an exponential to ambient 
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)
CDP_fits = []
plt.figure(figsize=(8, 6))
for entry in Y_CDP_calc:
    bin_means_CDP = np.array([entry.get(f'Bin{i:02d}_Y_mean', np.nan) for i in range(0, 30)], dtype=float)
    valid_indices = ~np.isnan(bin_means_CDP) 
    bin_centers_valid = np.array(bin_center_CDP)[valid_indices]
    bin_means_valid = bin_means_CDP[valid_indices]
    if not valid_indices.any():
        print(f"No valid data for date {entry['Date']}")
        continue
    try:
        popt, _ = curve_fit(exponential, bin_centers_valid, bin_means_valid, p0=(1, 1), maxfev=5000)
        n0, D = popt

        CDP_fits.append({
            'Date': entry['Date'],
            'BCB_start': entry.get('BCB_start', np.nan),
            'BCB_stop': entry.get('BCB_stop', np.nan),
            'Intercept_n0': n0,
            'E_folding_D': D
        })

        x_fit = np.linspace(min(bin_centers_valid), max(bin_centers_valid), 100)
        y_fit = exponential(x_fit, *popt)

        plt.plot(x_fit, y_fit, color='black', alpha=0.2)
    except RuntimeError:
        print(f"Fit could not be performed for date {entry['Date']}")
plt.xlabel("Deliquesced Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"CDP Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.ylim(10**-33, 10**1)
plt.title("Below Cloud Base January - June 2022\n Exponential Fit to CDP Size Distributions", fontsize=14, fontweight="bold")
plt.show()
# %%
#fit only to 10um diameter
bin_center_CDP = np.array(bin_center_CDP)
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)
CDP_fits_10 = []
plt.figure(figsize=(8, 6))
for entry in Y_CDP_calc:
    bin_means_CDP = np.array([entry.get(f'Bin{i:02d}_Y_mean', np.nan) for i in range(0, 30)], dtype=float)
    valid_indices = (bin_center_CDP <= 10) & ~np.isnan(bin_means_CDP)
    bin_centers_valid = np.array(bin_center_CDP)[valid_indices]
    bin_means_valid = bin_means_CDP[valid_indices]
    if valid_indices.any():
        try:
            popt, _ = curve_fit(exponential, bin_centers_valid, bin_means_valid, 
                                p0=(1, 1), maxfev=5000, 
                                bounds=([0, 0.1], [np.inf, 20])) 
            n0, D = popt

            if D > 15:
                print(f"⚠️ High slope detected! Date: {entry['Date']}, D: {D:.2f}")

        except RuntimeError:
            print(f" Fit failed for date {entry['Date']}")
            n0, D = np.nan, np.nan 
    else:
        n0, D = np.nan, np.nan  
    CDP_fits_10.append({
        'Date': entry['Date'],
        'BCB_start': entry.get('BCB_start', np.nan),
        'BCB_stop': entry.get('BCB_stop', np.nan),
        'Intercept_n0': n0,
        'E_folding_D': D
    })
    if not np.isnan(n0) and not np.isnan(D):
        x_fit = np.linspace(min(bin_centers_valid), 10, 100)
        y_fit = exponential(x_fit, *popt)
        plt.plot(x_fit, y_fit, color='black', alpha=0.2)
plt.xlabel("Deliquesced Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"CDP Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.ylim(10**-33, 10**1)
plt.title("Below Cloud Base January - June 2022\n Exponential Fit to CDP Size Distributions (≤10 µm)", fontsize=14, fontweight="bold")
plt.show()
print(f"Total successful CDP exponential fits: {np.sum(~np.isnan([fit['E_folding_D'] for fit in CDP_fits_10]))}")
# %%
#histogram for slope for 10um
ambient_slope_10_CDP = []
for fit in CDP_fits_10:
    if 'E_folding_D' in fit and not np.isnan(fit['E_folding_D']):
        ambient_slope_10_CDP.append(fit['E_folding_D'])
plt.figure(figsize=(8, 6))
plt.hist(ambient_slope_10_CDP, bins=20, color='blue', alpha=0.7)
plt.xlabel('Slope (um)', fontsize=14, fontweight="bold")
plt.ylabel('Frequency of flight legs', fontsize=14, fontweight="bold")
plt.title('CDP Fitted Ambient Size Distributions (≤10 µm)', fontsize=14, fontweight="bold")
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.show()
# %%
#Calculating total number concentration 
Y_BCB_calc_cm3_CDP = []
N_BCB_calc_cm3_CDP = []
for flight_data in master_CDP_BCB:
    for bin_means_CDP in flight_data:
        Y_calc_CDP = {'Date': bin_means_CDP['Date'], 'BCB_start': bin_means_CDP['BCB_start'], 'BCB_stop': bin_means_CDP['BCB_stop']}
        N_calc_CDP = {'Date': bin_means_CDP['Date'], 'BCB_start': bin_means_CDP['BCB_start'], 'BCB_stop': bin_means_CDP['BCB_stop']}
        for bin_label in range(0, 30):
            bin_key_Y = f'Bin{bin_label:02d}_Y_mean' 
            bin_key_N = f'Bin{bin_label:02d}_N_mean'
            Y_bin_value = np.nanmean(bin_means_CDP.get(bin_key_Y, np.nan))
            N_bin_value = np.nanmean(bin_means_CDP.get(bin_key_N, np.nan))
            Y_calc_CDP[bin_key_Y] = Y_bin_value * bin_log_CDP[bin_label]
            N_calc_CDP[bin_key_N] = N_bin_value * bin_log_CDP[bin_label]
        Y_BCB_calc_cm3_CDP.append(Y_calc_CDP)
        N_BCB_calc_cm3_CDP.append(N_calc_CDP)
print(f"Processed {len(Y_BCB_calc_cm3_CDP)} flight legs for CDP.")
# %%
#Calculating total number concentration 
total_concentration_cm3_CDP = []
for entry in Y_BCB_calc_cm3_CDP:
    total_Y_concentration = np.nansum([entry.get(f'Bin{i:02d}_Y_mean', 0) for i in range(0, 30)])
    total_concentration_cm3_CDP.append({
        'Date': entry['Date'],
        'BCB_start': entry['BCB_start'],
        'BCB_stop': entry['BCB_stop'],
        'Total_Y_Concentration_cm3': total_Y_concentration
    })
print(f"Processed {len(total_concentration_cm3_CDP)} flight legs for total CDP concentration.")
# %%
total_Y_concentrations_CDP = [entry['Total_Y_Concentration_cm3'] for entry in total_concentration_cm3_CDP]
total_Y_concentrations_CDP = [conc for conc in total_Y_concentrations_CDP if not np.isnan(conc)]
mean_total_concentration_CDP = np.mean(total_Y_concentrations_CDP)
print(f"Mean Total Number Concentration: {mean_total_concentration_CDP:.2f} cm⁻³")

#%%
master_BCB_RH = []
for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict = leg_data[i]
    flight_date = leg_dict['Date']  
    BCB_start = leg_dict['LegIndex_02']['StartTimes']
    BCB_stop = leg_dict['LegIndex_02']['StopTimes']

    rh_flight = h20[i]
    times_rh = rh_flight.Time_Start.values
    rh_values = rh_flight.RHw_DLH.values

    all_BCB = []

    for j in range(len(BCB_start)):
        start = int(BCB_start[j])
        end = int(BCB_stop[j])

        rh_times = {
            'Date': date,
            'BCB_start': start,
            'BCB_stop': end,
            'Rh_mean': [],
        }
        index1_start = None
        for k in range(len(times_rh)):
            if times_rh[k] == start:
                index1_start = k
                break
        index1_end = None
        for k in range(len(times_rh)):
            if times_rh[k] == end:
                index1_end = k
                break

        if index1_start is None or index1_end is None:
            rh9_mean = np.nan
        else:
            rh9 = rh_values[index1_start:index1_end + 1]
            rh9 = rh9[(rh9 <= 95) & (rh9 > 0)]  # filter for RH ≤ 95 and ignore missing/bad (-999 or 0)
            rh9_mean = np.nanmean(rh9)
        rh_times['Rh_mean'].append(rh9_mean)
        all_BCB.append(rh_times) 
    master_BCB_RH.append(all_BCB)
for flight in master_BCB_RH:
    for leg in flight:
        rh_mean_list = leg['Rh_mean']
        leg['Rh_mean'] = [np.nan if value <=0 else value for value in rh_mean_list]
#%%
relativy_humidity_values = []
#%%
#for only the legs present after LWC filtration and master_BCB_exponential 
date_leg_set = set()

for entry in Y_CDP_calc: 
    date = entry['Date']
    BCB_start = entry.get('BCB_start', np.nan)
    BCB_stop = entry.get('BCB_stop', np.nan)
    date_leg_set.add((date, BCB_start, BCB_stop))
filtered_master_BCB_RH_CDP = []

for flight in master_BCB_RH:
    filtered_legs = []
    for leg in flight:
        date = leg['Date']
        BCB_start = leg['BCB_start']
        BCB_stop = leg['BCB_stop']
        if (date, BCB_start, BCB_stop) in date_leg_set:
            filtered_legs.append(leg)
    if filtered_legs:
        filtered_master_BCB_RH_CDP.append(filtered_legs)
#%%
rh_values = [
    leg['Rh_mean'][0] for flight in filtered_master_BCB_RH_CDP for leg in flight if not np.isnan(leg['Rh_mean'][0])
]
plt.figure(figsize=(8, 6))
plt.hist(rh_values, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel('Relative Humidity (%)', fontsize=15, fontweight='bold')
plt.ylabel('Frequency of flight legs', fontsize=15, fontweight='bold')
plt.title('Leg average RH January - June 2022', fontweight='bold', fontsize=16)
plt.xticks(fontweight='bold', fontsize=14)
plt.yticks(fontweight='bold', fontsize=14)
plt.show()
# %%
#Make sure the leg counts 
total_entries_filtered_master_BCB_RH_CDP = sum(len(legs) for legs in filtered_master_BCB_RH_CDP)
print(f"Total entries in filtered_master_BCB_RH: {total_entries_filtered_master_BCB_RH_CDP}")
#%%
master_CDP_BCB = []
leg_info_CDP = []
for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict_CDP = leg_data[i]
    flight_date = leg_dict_CDP['Date']
    BCB_start = leg_dict_CDP['LegIndex_02']['StartTimes']
    BCB_stop = leg_dict_CDP['LegIndex_02']['StopTimes']
    CDP_flight = CDP_1Hz[i]
    twoDS_flight = twoDS[i]
    rh_flight = h20[i]
    CDP_times = np.array(CDP_flight['Time_Start'], dtype=float)
    TwoDS_times = np.array(twoDS_flight['Time_Start'], dtype=float)
    rh_times = np.array(rh_flight['Time_Start'], dtype=float)
    lwc = np.array(CDP_flight['LWC_CDP'], dtype=float)
    N_total = np.array(twoDS_flight['N-total_2DS'], dtype=float)
    rh_total = np.array(rh_flight['RHw_DLH'], dtype=float)

    bins = {
        f'CDP_Bin{bin_label:02d}': np.array(CDP_flight[f'CDP_Bin{bin_label:02d}'], dtype=float)
        for bin_label in range(30)
    }

    total_BCB_means_CDP = []

    for k in range(len(BCB_start)):
        start20 = BCB_start[k]
        end20 = BCB_stop[k]

        bin_means_CDP = {f'Bin{bin_label:02d}_Y_mean': [] for bin_label in range(30)}
        bin_means_CDP.update({f'Bin{bin_label:02d}_N_mean': [] for bin_label in range(30)})
        bin_means_CDP.update({
            'Date': date,
            'BCB_start': start20,
            'BCB_stop': end20,
            'LWC_mean': []
        })

        data_labels_CDP = []

        CDP_indices_in_range = np.where((CDP_times >= start20) & (CDP_times <= end20))[0]
        TwoDS_indices_in_range = np.where((TwoDS_times >= start20) & (TwoDS_times <= end20))[0]
        rh_indices_in_range = np.where((rh_times >= start20) & (rh_times <= end20))[0]

        if CDP_indices_in_range.size > 0 and TwoDS_indices_in_range.size > 0 and rh_indices_in_range.size > 0:
            for cdp_idx, twods_idx, rh_idx in zip(CDP_indices_in_range, TwoDS_indices_in_range, rh_indices_in_range):
                lwc_val = lwc[cdp_idx]
                N_val = N_total[twods_idx]
                rh_val = rh_total[rh_idx]

                lwc_label = 'Y' if 0 <= lwc_val <= 0.0025 else 'N'
                N_label   = 'Y' if 0 <= N_val <= 100 else 'N'
                rh_label  = 'Y' if 0 <= rh_val <= 95 else 'N'

                label = 'Y' if lwc_label == 'Y' and N_label == 'Y' and rh_label == 'Y' else 'N'

                data_labels_CDP.append(label)

                if label == 'Y':
                    bin_means_CDP['LWC_mean'].append(lwc_val)

                for bin_label in range(30):
                    bin_key = f'Bin{bin_label:02d}_{label}_mean'
                    bin_means_CDP[bin_key].append(bins[f'CDP_Bin{bin_label:02d}'][cdp_idx])

        for bin_label in range(30):
            for label in ['Y', 'N']:
                bin_key = f'Bin{bin_label:02d}_{label}_mean'
                if bin_means_CDP[bin_key]:
                    bin_means_CDP[bin_key] = np.nanmean(bin_means_CDP[bin_key])
                else:
                    bin_means_CDP[bin_key] = np.nan

        if bin_means_CDP['LWC_mean']:
            bin_means_CDP['LWC_mean'] = np.nanmean(bin_means_CDP['LWC_mean'])
        else:
            bin_means_CDP['LWC_mean'] = np.nan
        total_BCB_means_CDP.append(bin_means_CDP)

        leg_info_CDP.append({
            'Date': date,
            'BCB_start': start20,
            'BCB_stop': end20,
            'Data_Labels': data_labels_CDP,
        })

    master_CDP_BCB.append(total_BCB_means_CDP)
for leg in leg_info_CDP[:3]:
    print(f"{leg['Date']} {leg['BCB_start']}-{leg['BCB_stop']}")
    print("First 20 labels:", leg['Data_Labels'][:20])
    print("Count Y:", leg['Data_Labels'].count('Y'))
    print("Count N:", leg['Data_Labels'].count('N'))
    print("----")

#%%
Y_CDP_calc = []
N_CDP_calc = []
for flight_data in master_CDP_BCB:
    for bin_means_CDP in flight_data:
        Y_calc = {
            'Date': bin_means_CDP['Date'],
            'BCB_start': bin_means_CDP['BCB_start'],
            'BCB_stop': bin_means_CDP['BCB_stop'],
            'LWC_mean': bin_means_CDP['LWC_mean']
        }
        N_calc = {
            'Date': bin_means_CDP['Date'],
            'BCB_start': bin_means_CDP['BCB_start'],
            'BCB_stop': bin_means_CDP['BCB_stop']
        }
        
        for bin_label in range(0, 30):
            bin_key_Y = f'Bin{bin_label:02d}_Y_mean'
            bin_key_N = f'Bin{bin_label:02d}_N_mean'

            Y_calc[bin_key_Y] = np.nanmean(bin_means_CDP[bin_key_Y]) * Logg_CDP[bin_label]
            N_calc[bin_key_N] = np.nanmean(bin_means_CDP[bin_key_N]) * Logg_CDP[bin_label]

        Y_CDP_calc.append(Y_calc)
        N_CDP_calc.append(N_calc)
# LWC histogram for CDP filtered BCB legs
lwc_values = [
    entry['LWC_mean']
    for entry in Y_CDP_calc
    if np.isfinite(entry['LWC_mean'])
]

plt.figure(figsize=(8, 6))

plt.hist(
    lwc_values,
    bins=20,
    edgecolor='black',
    alpha=0.7
)

plt.xlabel('Mean LWC (g m$^{-3}$)', fontsize=15, fontweight='bold')
plt.ylabel('Frequency of flight legs', fontsize=15, fontweight='bold')
plt.title('Leg-average CDP LWC', fontweight='bold', fontsize=16)
plt.xticks(fontweight='bold', fontsize=14)
plt.yticks(fontweight='bold', fontsize=14)
# plt.savefig("LWC_CDP.pdf", dpi=300, bbox_inches='tight')
# plt.show()
print(f"Number of LWC legs plotted: {len(lwc_values)}")
print(f"Mean leg-average LWC: {np.nanmean(lwc_values):.6f} g m^-3")
print(f"Median leg-average LWC: {np.nanmedian(lwc_values):.6f} g m^-3")

# %%
##Obtaining g(RH) = [1.7 / (1-RH)]^0.31 for all mean RH values 
master_BCB_gRH_CDP = []
for flight in master_BCB_RH:
    flight_gRH = [] 
    
    for leg in flight:
        new_leg = leg.copy() 
        
        rh_mean = new_leg['Rh_mean'][0] / 100.0 
        
        if np.isnan(rh_mean) or rh_mean >= 1:
            gRH_value = np.nan
            print(f"Skipping calculation for Rh_mean = {new_leg['Rh_mean'][0]} as it results in division by zero or invalid value.")
        else:
            gRH_value = (1.7 / (1 - rh_mean)) ** 0.31
        
        new_leg['gRh_mean'] = [gRH_value]
        flight_gRH.append(new_leg)
    master_BCB_gRH_CDP.append(flight_gRH)
#%%
#only the grh from filtered_master_BCB_RH
filtered_master_BCB_gRH_CDP = []
for flight in filtered_master_BCB_RH_CDP:
    flight_gRH = [] 
    
    for leg in flight:
        new_leg = leg.copy() 
        
        rh_mean = new_leg['Rh_mean'][0] / 100.0 
        
        if np.isnan(rh_mean) or rh_mean >= 1:
            gRH_value = np.nan
            print(f"Skipping calculation for Rh_mean = {new_leg['Rh_mean'][0]} as it results in division by zero or invalid value.")
        else:
            gRH_value = (1.7 / (1 - rh_mean)) ** 0.31

        new_leg['gRh_mean'] = [gRH_value]
        flight_gRH.append(new_leg)
    filtered_master_BCB_gRH_CDP.append(flight_gRH)
#%%
total_entries_filtered_master_BCB_gRH_CDP= sum(len(legs) for legs in filtered_master_BCB_gRH_CDP)
print(f"Total entries in filtered_master_BCB_gRH: {total_entries_filtered_master_BCB_gRH_CDP}")
# %%
filtered_master_BCB_interceptdry_dict_CDP = {}
if isinstance(filtered_master_BCB_gRH_CDP[0], list):
    filtered_master_BCB_gRH_CDP = [item for sublist in filtered_master_BCB_gRH_CDP for item in sublist]
for entry in filtered_master_BCB_gRH_CDP:
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    gRh_mean = entry['gRh_mean'][0]  
    Rh_mean = entry['Rh_mean'][0]  
    if Rh_mean < 0:
        continue

    key = (date, BCB_start, BCB_stop)

    if key in CDP_fits_10:
        n0 = CDP_fits_10[key]['Intercept_n0'] 
        
        
        dryintercept = n0 / gRh_mean if gRh_mean > 0 else np.nan

        filtered_master_BCB_interceptdry_dict_CDP[key] = {
            'Date': date,
            'BCB_start': BCB_start,
            'BCB_stop': BCB_stop,
            'Rh_mean': entry['Rh_mean'],
            'gRh_mean': entry['gRh_mean'],
            'dry intercept': dryintercept
        }
filtered_master_BCB_dryintercept_CDP = list(filtered_master_BCB_interceptdry_dict_CDP.values())
print(f"Length of filtered_master_BCB_dryintercept: {len(filtered_master_BCB_dryintercept_CDP)}")
dryintercept_values_CDP = [
    leg['dry intercept'] for leg in filtered_master_BCB_dryintercept_CDP if not np.isnan(leg['dry intercept'])
]
plt.figure(figsize=(8, 6))
plt.hist(dryintercept_values_CDP, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel(r"$\mathbf{Dry\ intercept\ (cm^{-3}\ \mu m^{-1})}$", fontsize=15)
plt.ylabel('Frequency', fontsize=15, fontweight='bold')
plt.title('Dry intercept (gRH / N0)', fontweight='bold', fontsize=16)
plt.grid(True)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.show()
# %%
#dry size distributions
filtered_master_BCB_ddry_CDP = []
for entry in filtered_master_BCB_gRH_CDP:
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    gRh_mean = entry['gRh_mean'][0]  
    if gRh_mean > 0:
        ddry_values_CDP = np.array([D_amb / gRh_mean for D_amb in bin_center_CDP])
    else:
        ddry_values_CDP = np.full(len(bin_center_CDP), np.nan)
        print(f"Skipping division for {date}, {BCB_start}-{BCB_stop} due to invalid gRh_mean.")

    ddry_bin_widths_CDP = np.diff(ddry_values_CDP, append=np.nan)  

    raw_concentrations_CDP = next(
        (leg for leg in Y_CDP_calc if leg['Date'] == date and leg['BCB_start'] == BCB_start and leg['BCB_stop'] == BCB_stop),
        None
    )

    if raw_concentrations_CDP:
        dN_dD_ambient_CDP = np.array([raw_concentrations_CDP.get(f'Bin{i:02d}_Y_mean', np.nan) for i in range(0, 30)], dtype=float)

        dN_dD_dry_CDP = np.where(
            (~np.isnan(dN_dD_ambient_CDP)) & (~np.isnan(ddry_bin_widths_CDP)) & (gRh_mean > 0),
            dN_dD_ambient_CDP * (np.array(bin_center_CDP) / ddry_values_CDP) * (np.diff(bin_center_CDP, append=np.nan) / ddry_bin_widths_CDP),
            np.nan
        )
    else:
        dN_dD_dry_CDP = np.full(len(bin_center_CDP), np.nan)
        print(f"Missing raw size distribution for {date}, {BCB_start}-{BCB_stop}")

    filtered_master_BCB_ddry_CDP.append({
        'Date': date,
        'BCB_start': BCB_start,
        'BCB_stop': BCB_stop,
        'ddry': ddry_values_CDP.tolist(),
        'dN/dDdry': dN_dD_dry_CDP.tolist(),
        'ddry_bin_widths': ddry_bin_widths_CDP.tolist(),  
        'gRh_mean': gRh_mean
    })
print(f"Length of filtered_master_BCB_ddry_CDP: {len(filtered_master_BCB_ddry_CDP)}")
# %%
#Plotting dry size distributions
common_bins_CDP = np.linspace(2, 25, 35)  
plt.figure(figsize=(8, 6))
for entry in filtered_master_BCB_ddry_CDP:
    ddry_values_CDP = np.array(entry['ddry'])  
    dN_dD_dry_CDP = np.array(entry['dN/dDdry']) 

    valid_indices = ~np.isnan(ddry_values_CDP) & ~np.isnan(dN_dD_dry_CDP)
    if np.sum(valid_indices) < 2:
        continue  
    interp_func_CDP = interp1d(ddry_values_CDP[valid_indices], dN_dD_dry_CDP[valid_indices], 
                               kind='linear', bounds_error=False, fill_value=np.nan)
    interpolated_dN_dD_dry_CDP = interp_func_CDP(common_bins_CDP)

    plt.plot(common_bins_CDP, interpolated_dN_dD_dry_CDP, color='black', alpha=0.2)

plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base January - June 2022\n Raw Dry Size Distributions (CDP)", fontsize=14, fontweight="bold")
plt.show()

# %%
#removing the 0s
common_bins_CDP = np.linspace(2, 25, 35)
plt.figure(figsize=(8, 6))
for entry in filtered_master_BCB_ddry_CDP:
    ddry_values_CDP = np.array(entry['ddry']) 
    dN_dD_dry_CDP = np.array(entry['dN/dDdry']) 
    valid_indices = ~np.isnan(ddry_values_CDP) & ~np.isnan(dN_dD_dry_CDP)
    if np.sum(valid_indices) < 2:
        continue  
    interp_func_CDP = interp1d(ddry_values_CDP[valid_indices], dN_dD_dry_CDP[valid_indices], 
                               kind='linear', bounds_error=False, fill_value=np.nan)
    interpolated_dN_dD_dry_CDP = interp_func_CDP(common_bins_CDP)

    valid_interpolated_indices = (interpolated_dN_dD_dry_CDP > 0) & ~np.isnan(interpolated_dN_dD_dry_CDP)
    filtered_bins_CDP = common_bins_CDP[valid_interpolated_indices]
    filtered_dN_dD_dry_CDP = interpolated_dN_dD_dry_CDP[valid_interpolated_indices]

    if len(filtered_bins_CDP) > 0: 
        plt.plot(filtered_bins_CDP, filtered_dN_dD_dry_CDP, color='black', alpha=0.2)

plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base January - June 2022\n Raw Dry Size Distributions (CDP)", fontsize=14, fontweight="bold")
plt.show()
# %%
#Averaging the dry size distributions********
common_bins_CDP = np.linspace(2, 25, 35) 
sum_interpolated_dN_dD_dry_CDP = np.zeros_like(common_bins_CDP, dtype=float)
count_interpolated_dN_dD_dry_CDP = np.zeros_like(common_bins_CDP, dtype=int)
for entry in filtered_master_BCB_ddry_CDP:
    ddry_values_CDP = np.array(entry['ddry'])  
    dN_dD_dry_CDP = np.array(entry['dN/dDdry'])
    valid_indices = ~np.isnan(ddry_values_CDP) & ~np.isnan(dN_dD_dry_CDP)
    if np.sum(valid_indices) < 2:
        continue 
    interp_func_CDP = interp1d(ddry_values_CDP[valid_indices], dN_dD_dry_CDP[valid_indices], 
                               kind='linear', bounds_error=False, fill_value=np.nan)
    interpolated_dN_dD_dry_CDP = interp_func_CDP(common_bins_CDP)

    valid_interpolated_indices = (interpolated_dN_dD_dry_CDP > 0) & ~np.isnan(interpolated_dN_dD_dry_CDP)
    sum_interpolated_dN_dD_dry_CDP[valid_interpolated_indices] += interpolated_dN_dD_dry_CDP[valid_interpolated_indices]
    count_interpolated_dN_dD_dry_CDP[valid_interpolated_indices] += 1

average_dN_dD_dry_CDP = np.divide(sum_interpolated_dN_dD_dry_CDP, count_interpolated_dN_dD_dry_CDP, where=count_interpolated_dN_dD_dry_CDP > 0)
plt.figure(figsize=(8, 6))
plt.plot(common_bins_CDP, average_dN_dD_dry_CDP, color='blue', linewidth=2, label='Average Dry Size Distribution (CDP)')
plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"CDP Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("CDP Average Below Cloud Base Dry Size Distribution \n January - June 2022", fontsize=14, fontweight="bold")
plt.show()
# %%
# Averaging the CDP dry size distributions with ±2 SEM ********
common_bins_CDP = np.linspace(2, 25, 35)
sum_interpolated_dN_dD_dry_CDP = np.zeros_like(
    common_bins_CDP,
    dtype=float)
count_interpolated_dN_dD_dry_CDP = np.zeros_like(
    common_bins_CDP,
    dtype=int)
all_interpolated_dN_dD_dry_CDP = []
for entry in filtered_master_BCB_ddry_CDP:

    ddry_values_CDP = np.asarray(
        entry['ddry'],
        dtype=float
    )

    dN_dD_dry_CDP = np.asarray(
        entry['dN/dDdry'],
        dtype=float
    )

    valid_indices_CDP = (
        np.isfinite(ddry_values_CDP) &
        np.isfinite(dN_dD_dry_CDP)
    )

    if np.sum(valid_indices_CDP) < 2:
        continue
    ddry_valid_CDP = ddry_values_CDP[valid_indices_CDP]
    dN_dD_valid_CDP = dN_dD_dry_CDP[valid_indices_CDP]
    sort_idx_CDP = np.argsort(ddry_valid_CDP)

    ddry_valid_CDP = ddry_valid_CDP[sort_idx_CDP]
    dN_dD_valid_CDP = dN_dD_valid_CDP[sort_idx_CDP]


    interp_func_CDP = interp1d(
        ddry_valid_CDP,
        dN_dD_valid_CDP,
        kind='linear',
        bounds_error=False,
        fill_value=np.nan    )

    interpolated_dN_dD_dry_CDP = interp_func_CDP(
        common_bins_CDP    )
    valid_interpolated_indices_CDP = (
        np.isfinite(interpolated_dN_dD_dry_CDP) &
        (interpolated_dN_dD_dry_CDP > 0)    )
    interpolated_profile_CDP = np.full_like(
        common_bins_CDP,
        np.nan,
        dtype=float    )
    interpolated_profile_CDP[
        valid_interpolated_indices_CDP
    ] = interpolated_dN_dD_dry_CDP[
        valid_interpolated_indices_CDP    ]
    all_interpolated_dN_dD_dry_CDP.append(
        interpolated_profile_CDP    )
    sum_interpolated_dN_dD_dry_CDP[
        valid_interpolated_indices_CDP
    ] += interpolated_dN_dD_dry_CDP[
        valid_interpolated_indices_CDP    ]
    count_interpolated_dN_dD_dry_CDP[
        valid_interpolated_indices_CDP
    ] += 1
average_dN_dD_dry_CDP = np.full_like(
    common_bins_CDP,
    np.nan,
    dtype=float)

np.divide(
    sum_interpolated_dN_dD_dry_CDP,
    count_interpolated_dN_dD_dry_CDP,
    out=average_dN_dD_dry_CDP,
    where=count_interpolated_dN_dD_dry_CDP > 0)
all_interpolated_dN_dD_dry_CDP = np.asarray(
    all_interpolated_dN_dD_dry_CDP,
    dtype=float)
std_dN_dD_dry_CDP = np.nanstd(
    all_interpolated_dN_dD_dry_CDP,
    axis=0,
    ddof=1)
sem_dN_dD_dry_CDP = np.full_like(
    common_bins_CDP,
    np.nan,
    dtype=float)

valid_sem_CDP = count_interpolated_dN_dD_dry_CDP > 1

sem_dN_dD_dry_CDP[valid_sem_CDP] = (
    std_dN_dD_dry_CDP[valid_sem_CDP] /
    np.sqrt(count_interpolated_dN_dD_dry_CDP[valid_sem_CDP]))
two_sem_dN_dD_dry_CDP = 2 * sem_dN_dD_dry_CDP
plot_mask_CDP = (
    np.isfinite(average_dN_dD_dry_CDP) &
    np.isfinite(two_sem_dN_dD_dry_CDP) &
    (average_dN_dD_dry_CDP > 0) &
    (count_interpolated_dN_dD_dry_CDP > 1))

x_plot_CDP = common_bins_CDP[plot_mask_CDP]
mean_plot_CDP = average_dN_dD_dry_CDP[
    plot_mask_CDP]

two_sem_plot_CDP = two_sem_dN_dD_dry_CDP[
    plot_mask_CDP]
small_positive_value = 1e-12

lower_error_CDP = np.minimum(
    two_sem_plot_CDP,
    mean_plot_CDP - small_positive_value)

lower_error_CDP = np.clip(
    lower_error_CDP,
    0,
    None)

upper_error_CDP = two_sem_plot_CDP
asymmetric_yerr_CDP = np.vstack([
    lower_error_CDP,
    upper_error_CDP])
plt.figure(figsize=(8, 6))

plt.errorbar(
    x_plot_CDP,
    mean_plot_CDP,
    yerr=asymmetric_yerr_CDP,
    fmt='-',
    color='blue',
    ecolor='blue',
    linewidth=2,
    elinewidth=1.5,
    capsize=4,
    capthick=1.5,
    label=r'CDP $\pm$2 SEM (~95% CI)')
plt.xlabel(
    "Dry Bin Center Diameter (μm)",
    fontsize=14,
    fontweight="bold")
plt.ylabel(
    r"CDP Number Concentration "
    r"(cm$^{-3}$ $\mu$m$^{-1}$)",
    fontsize=14,
    fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xlim(0, 25)
plt.xticks(
    fontweight="bold",
    fontsize=14)
plt.yticks(
    fontweight="bold",
    fontsize=14)
plt.title(
    "CDP Average Below Cloud Base Dry Size Distribution\n"
    "January–June 2022",
    fontsize=14,
    fontweight="bold")
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
# %%
#save the average distribution
lower_2SEM_CDP = (
    average_dN_dD_dry_CDP -
    two_sem_dN_dD_dry_CDP
)

upper_2SEM_CDP = (
    average_dN_dD_dry_CDP +
    two_sem_dN_dD_dry_CDP
)

average_dry_distribution_CDP = pd.DataFrame({
    'Dry_Diameter_um': common_bins_CDP,
    'Average_dN_dD_dry': average_dN_dD_dry_CDP,
    'Standard_Deviation': std_dN_dD_dry_CDP,
    'SEM': sem_dN_dD_dry_CDP,
    'Two_SEM': two_sem_dN_dD_dry_CDP,
    'Lower_2SEM': lower_2SEM_CDP,
    'Upper_2SEM': upper_2SEM_CDP,
    'N_profiles': count_interpolated_dN_dD_dry_CDP
})

save_dir = "/home/disk/eos4/kathem24/activate/data/2022/FCDP"

os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(
    save_dir,
    "Average_Dry_Size_Distribution_errorbars2022CDP.csv"
)

average_dry_distribution_CDP.to_csv(
    save_path,
    index=False
)

print(f"Saved to: {save_path}")
print(average_dry_distribution_CDP.head())
print(f"File exists: {os.path.exists(save_path)}")
# %%
#plot all three together
small_positive_value = 1e-12
plot_mask_CAS = (
    np.isfinite(average_dN_dD_dry_CAS) &
    np.isfinite(two_sem_dN_dD_dry_CAS) &
    (average_dN_dD_dry_CAS > 0) &
    (count_interpolated_dN_dD_dry_CAS > 1))
x_plot_CAS = common_bins_CAS[plot_mask_CAS]
mean_plot_CAS = average_dN_dD_dry_CAS[
    plot_mask_CAS]
two_sem_plot_CAS = two_sem_dN_dD_dry_CAS[
    plot_mask_CAS]
lower_error_CAS = np.minimum(
    two_sem_plot_CAS,
    mean_plot_CAS - small_positive_value)
lower_error_CAS = np.clip(
    lower_error_CAS,
    0,
    None)
upper_error_CAS = two_sem_plot_CAS
asymmetric_yerr_CAS = np.vstack([
    lower_error_CAS,
    upper_error_CAS])
plot_mask_CDP = (
    np.isfinite(average_dN_dD_dry_CDP) &
    np.isfinite(two_sem_dN_dD_dry_CDP) &
    (average_dN_dD_dry_CDP > 0) &
    (count_interpolated_dN_dD_dry_CDP > 1))
x_plot_CDP = common_bins_CDP[plot_mask_CDP]
mean_plot_CDP = average_dN_dD_dry_CDP[
    plot_mask_CDP]
two_sem_plot_CDP = two_sem_dN_dD_dry_CDP[
    plot_mask_CDP]
lower_error_CDP = np.minimum(
    two_sem_plot_CDP,
    mean_plot_CDP - small_positive_value)
lower_error_CDP = np.clip(
    lower_error_CDP,
    0,
    None)
upper_error_CDP = two_sem_plot_CDP
asymmetric_yerr_CDP = np.vstack([
    lower_error_CDP,
    upper_error_CDP])
plot_mask_FCDP = (
    np.isfinite(average_dN_dD_dry_FCDP) &
    np.isfinite(two_sem_dN_dD_dry_FCDP) &
    (average_dN_dD_dry_FCDP > 0) &
    (count_interpolated_dN_dD_dry_FCDP > 1))
x_plot_FCDP = common_bins_FCDP[
    plot_mask_FCDP]
mean_plot_FCDP = average_dN_dD_dry_FCDP[
    plot_mask_FCDP]
two_sem_plot_FCDP = two_sem_dN_dD_dry_FCDP[
    plot_mask_FCDP]
lower_error_FCDP = np.minimum(
    two_sem_plot_FCDP,
    mean_plot_FCDP - small_positive_value)
lower_error_FCDP = np.clip(
    lower_error_FCDP,
    0,
    None)
upper_error_FCDP = two_sem_plot_FCDP
asymmetric_yerr_FCDP = np.vstack([
    lower_error_FCDP,
    upper_error_FCDP])
fig, ax = plt.subplots(figsize=(12, 7))
plt.errorbar(
    x_plot_CAS,
    mean_plot_CAS,
    yerr=asymmetric_yerr_CAS,
    fmt='-',
    color='black',
    ecolor='black',
    linewidth=2.5,
    elinewidth=1.3,
    capsize=4,
    capthick=1.3,
    label=r'CAS $\pm$2 SEM (~95% CI)',
    zorder=3)
plt.errorbar(
    x_plot_CDP,
    mean_plot_CDP,
    yerr=asymmetric_yerr_CDP,
    fmt='-',
    color='blue',
    ecolor='blue',
    linewidth=2.5,
    elinewidth=1.3,
    capsize=4,
    capthick=1.3,
    label=r'CDP $\pm$2 SEM (~95% CI)',
    zorder=2)
plt.errorbar(
    x_plot_FCDP,
    mean_plot_FCDP,
    yerr=asymmetric_yerr_FCDP,
    fmt='-',
    color='green',
    ecolor='green',
    linewidth=2.5,
    elinewidth=1.3,
    capsize=4,
    capthick=1.3,
    label=r'FCDP $\pm$2 SEM (~95% CI)',
    zorder=1)
plt.xlabel(
    "Dry Bin Center Diameter (μm)",
    fontsize=20,
    fontweight="bold")
plt.ylabel(
    r"Number Concentration "
    r"(cm$^{-3}$ $\mu$m$^{-1}$)",
    fontsize=19,
    fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xlim(0, 27)
plt.xticks(
    fontweight="bold",
    fontsize=18)
plt.yticks(
    fontweight="bold",
    fontsize=18)
plt.title(
    "Average Dry Size Distributions",
    fontsize=20,
    fontweight="bold")
plt.legend(    fontsize=14,    loc='center left',    bbox_to_anchor=(1.02, 0.5))
plt.tight_layout()
plt.show()
# %%
