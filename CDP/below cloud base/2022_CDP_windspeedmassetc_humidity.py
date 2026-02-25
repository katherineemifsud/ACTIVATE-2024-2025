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
from matplotlib.patches import Rectangle
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
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy.ma as ma
from scipy.interpolate import interp1d
#%%

#Now CDP data
#This is how we will correct our droplet concentration units from 
#dN/dlogD to dN/dD
#We will use the bin width to convert the units
L_00=math.log10(3)-math.log10(2)
L_01=math.log10(4)-math.log10(3)
L_02=math.log10(5)-math.log10(4)
L_03=math.log10(6)-math.log10(5)
L_04=math.log10(7)-math.log10(6)
L_05=math.log10(8)-math.log10(7)
L_06=math.log10(9)-math.log10(8)
L_07=math.log10(10)-math.log10(9)
L_08=math.log10(11)-math.log10(10)
L_09=math.log10(12)-math.log10(11)
L_10=math.log10(13)-math.log10(12)
L_11=math.log10(14)-math.log10(13)
L_12=math.log10(16)-math.log10(14)
L_13=math.log10(18)-math.log10(16)
L_14=math.log10(20)-math.log10(18)
L_15=math.log10(22)-math.log10(20)
L_16=math.log10(24)-math.log10(22)
L_17=math.log10(26)-math.log10(24)
L_18=math.log10(28)-math.log10(26)
L_19=math.log10(30)-math.log10(28)
L_20=math.log10(32)-math.log10(30)
L_21=math.log10(34)-math.log10(32)
L_22=math.log10(36)-math.log10(34)
L_23=math.log10(38)-math.log10(36)
L_24=math.log10(40)-math.log10(38)
L_25=math.log10(42)-math.log10(40)
L_26=math.log10(44)-math.log10(42)
L_27=math.log10(46)-math.log10(44)
L_28=math.log10(48)-math.log10(46)
L_29=math.log10(50)-math.log10(48)


bin_log_CDP=[L_00, L_01, L_02, L_03, L_04, L_05, L_06, L_07, L_08,
          L_09, L_10, L_11,
          L_12, L_13, L_14, L_15, L_16, 
        L_17, L_18, L_19, L_20, L_21, L_22, L_23, 
        L_24, L_25, L_26, L_27, L_28, L_29]


P00=(3-2)
P01=(4-3)
P02=(5-4)
P03=(6-5)
P04=(7-6)
P05=(8-7)
P06=(9-8)
P07=(10-9)
P08=(11-10)
P09=(12-11)
P10=(13-12)
P11=(14-13)
P12 = (16-14)
P13 = (18-16)
P14 = (20-18)
P15 = (22-20)
P16 = (24-22)
P17 = (26-24)
P18 = (28-26)
P19 = (30-28)
P20 = (32-30)
P21 = (34-32)
P22 = (36-34)
P23 = (38-36)
P24 = (40-38)
P25 = (42-40)
P26 = (44-42)
P27 = (46-44)
P28 = (48-46)
P29 = (50-48)


J00=(L_00 / P00)
J01=(L_01 / P01)
J02=(L_02 / P02)
J03=(L_03 / P03)
J04=(L_04 / P04)
J05=(L_05 / P05)
J06=(L_06 / P06)
J07=(L_07 / P07)
J08=(L_08 / P08)
J09=(L_09 / P09)
J10=(L_10 / P10)
J11=(L_11 / P11)
J12 = (L_12 / P12)
J13 = (L_13 / P13)
J14 = (L_14 / P14)
J15 = (L_15 / P15)
J16 = (L_16 / P16)
J17 = (L_17 / P17)
J18 = (L_18 / P18)
J19 = (L_19 / P19)
J20 = (L_20 / P20)
J21 = (L_21 / P21)
J22 = (L_22 / P22)
J23 = (L_23 / P23)
J24 = (L_24 / P24)
J25 = (L_25 / P25)
J26 = (L_26 / P26)
J27 = (L_27 / P27)
J28 = (L_28 / P28)
J29 = (L_29 / P29)


Logg_CDP = [J00, J01, J02, J03, J04, J05, J06, J07, J08, J09, J10, 
            J11, J12, J13, J14, J15, J16, J17, J18, J19, J20, J21,
            J22, J23, J24, J25, J26, J27, J28, J29]

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
                    print(f"❗ RH violation: {rh_val:.2f} passed at time {CDP_times[CDP_idx]}")

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
#%%
#as a heatmap

base_cmap = plt.cm.viridis
colors = base_cmap(np.linspace(0, 1, 256))
colors[:80] = np.linspace([1, 1, 1, 1], colors[80], 80)
fading_viridis = LinearSegmentedColormap.from_list("fading_viridis", colors)
common_bins_cdp = np.linspace(2, 40, 200)
all_interp_distributions_cdp = []

for entry in Y_CDP_calc:
    bin_means = np.array([entry.get(f'Bin{i:02d}_Y_mean', np.nan) for i in range(30)], dtype=float)
    valid = (bin_means > 0) & ~np.isnan(bin_means)
    if np.sum(valid) < 2:
        continue
    interp_func = interp1d(bin_center_CDP[valid], bin_means[valid], kind='linear',
                           bounds_error=False, fill_value=np.nan)
    interpolated = interp_func(common_bins_cdp)
    interpolated[(interpolated <= 0) | np.isnan(interpolated)] = np.nan
    all_interp_distributions_cdp.append(interpolated)

# Build histogram matrix
y_matrix_cdp = np.array(all_interp_distributions_cdp)
y_matrix_cdp[np.isnan(y_matrix_cdp)] = 0
y_bins_log = np.logspace(-7, 1.5, 150)

H, xedges, yedges = np.histogram2d(
    np.repeat(common_bins_cdp, y_matrix_cdp.shape[0]),
    y_matrix_cdp.T.flatten(),
    bins=[common_bins_cdp, y_bins_log]
)
H = H / y_matrix_cdp.shape[0]  # Normalize
H_masked = ma.masked_where(H == 0, H)

plt.figure(figsize=(9, 6))
norm = LogNorm(vmin=1e-4, vmax=1)
img = plt.pcolormesh(xedges, yedges, H_masked.T, shading='auto', cmap=fading_viridis, norm=norm)
plt.xlabel("Deliquesced Diameter (μm)", fontsize=19, fontweight="bold")
plt.ylabel("CDP Number Concentration\n(cm$^{-3}$ μm$^{-1}$)", fontsize=19, fontweight="bold")
plt.yscale("log")
plt.ylim(1e-7, 10**1)
plt.xlim(0, 40)
plt.xticks(fontweight="bold", fontsize=19)
plt.yticks(fontweight="bold", fontsize=19)
plt.title("Below Cloud Base\n January–June 2022\nAmbient CDP Size Distributions", fontsize=19, fontweight="bold")
cbar = plt.colorbar(img)
cbar.set_label("Fraction of Legs", fontsize=19)
cbar.ax.tick_params(labelsize=16)
cbar.set_ticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
cbar.set_ticklabels([r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$'])
plt.tight_layout()
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
#save total concentration to csv
# save_dir = "/home/disk/eos4/kathem24/activate/data/CDP/2022/csv"
# os.makedirs(save_dir, exist_ok=True)   # ensures directory exists
# save_path = os.path.join(save_dir, "total_Y_concentration_cm3_CDP.csv")
# total_concentration_df = pd.DataFrame(total_concentration_cm3_CDP)
# total_concentration_df.to_csv(save_path, index=False)
# print(f"Saved to: {save_path}")

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
#%%
#Histogram of gRH values
gRH_values_CDP = [
    leg['gRh_mean'][0] for flight in filtered_master_BCB_gRH_CDP for leg in flight if not np.isnan(leg['gRh_mean'][0])
]
plt.figure(figsize=(8, 6))
plt.hist(gRH_values_CDP, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel('Growth factor (gRH)', fontsize=15, fontweight='bold')
plt.ylabel('Frequency of flight legs', fontsize=15, fontweight='bold')
plt.title('Applying the growth factor equation to RH mean values', fontweight='bold', fontsize=15)
plt.xticks(fontweight='bold', fontsize=14)
plt.yticks(fontweight='bold', fontsize=14)
plt.show()

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
#%%
#Ambient histogram 
ambient_intercept_values_CDP = [
    fit['Intercept_n0'] for fit in CDP_fits if not np.isnan(fit['Intercept_n0'])
]
plt.figure(figsize=(8, 6))
plt.hist(ambient_intercept_values_CDP, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel(r"$\mathbf{Ambient\ Intercept\ (cm^{-3}\ \mu m^{-1})}$", fontsize=15)
plt.ylabel('Frequency', fontsize=15, fontweight='bold')
plt.title('Histogram of Ambient Intercepts (N0)', fontsize=16, fontweight='bold')
plt.grid(True)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.show()

# %%
#Scatter plot of ambient vs dry intercepts
ambient_n0_values_CDP = [fit['Intercept_n0'] for fit in CDP_fits_10.values() if not np.isnan(fit['Intercept_n0'])]
dry_intercept_values_CDP = [leg['dry intercept'] for leg in filtered_master_BCB_dryintercept_CDP if not np.isnan(leg['dry intercept'])]
if len(ambient_n0_values_CDP) == len(dry_intercept_values_CDP):
    plt.figure(figsize=(8, 6))
    plt.scatter(ambient_n0_values_CDP, dry_intercept_values_CDP, alpha=0.5, edgecolor='black')
    plt.xlabel(r'Ambient Intercept $N_0$ (cm$^{-3} \mu$m$^{-1}$)', fontsize=14)
    plt.ylabel(r'Dry Intercept (cm$^{-3} \mu$m$^{-1}$)', fontsize=14)
    plt.title('Scatter Plot: Ambient vs. Dry Intercept', fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
else:
    print(f"Mismatch in data lengths! Ambient: {len(ambient_n0_values_CDP)}, Dry: {len(dry_intercept_values_CDP)}")
plt.figure(figsize=(8, 6))
plt.hist(ambient_n0_values_CDP, bins=20, alpha=0.5, label="Ambient Intercept (N0)", edgecolor="black")
plt.hist(dry_intercept_values_CDP, bins=20, alpha=0.5, label="Dry Intercept (N0/gRH)", edgecolor="black")
plt.xlabel(r"Intercept Value (cm$^{-3} \mu$m$^{-1}$)", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.title("Comparison of Ambient vs. Dry Intercept Distributions", fontsize=16, fontweight='bold')
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
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
#%%
#as a heatmap 
base_cmap = plt.cm.viridis
colors = base_cmap(np.linspace(0, 1, 256))
colors[:80] = np.linspace([1, 1, 1, 1], colors[80], 80)
fading_viridis = LinearSegmentedColormap.from_list("fading_viridis", colors)
common_bins_dry = np.linspace(2, 25, 100)
all_interp_dry_distributions = []
for entry in filtered_master_BCB_ddry_CDP:
    ddry_values = np.array(entry['ddry'])
    dN_dD = np.array(entry['dN/dDdry'])

    valid = ~np.isnan(ddry_values) & ~np.isnan(dN_dD)
    if np.sum(valid) < 2:
        continue
    interp_func = interp1d(ddry_values[valid], dN_dD[valid], kind='linear', bounds_error=False, fill_value=np.nan)
    interpolated = interp_func(common_bins_dry)
    interpolated[(interpolated <= 0) | np.isnan(interpolated)] = np.nan
    all_interp_dry_distributions.append(interpolated)
y_matrix_dry = np.array(all_interp_dry_distributions)
y_matrix_dry[np.isnan(y_matrix_dry)] = 0
y_bins_dry_log = np.logspace(-7, 1.5, 150)

H, xedges, yedges = np.histogram2d(
    np.repeat(common_bins_dry, y_matrix_dry.shape[0]),
    y_matrix_dry.T.flatten(),
    bins=[common_bins_dry, y_bins_dry_log]
)
H = H / y_matrix_dry.shape[0]
H_masked = ma.masked_where(H == 0, H)
plt.figure(figsize=(9, 6))
norm = LogNorm(vmin=1e-4, vmax=1)
img = plt.pcolormesh(xedges, yedges, H_masked.T, shading='auto', cmap=fading_viridis, norm=norm)
plt.xlabel("Dry Diameter (μm)", fontsize=19, fontweight="bold")
plt.ylabel("CDP Number Concentration\n(cm$^{-3}$ μm$^{-1}$)", fontsize=19, fontweight="bold")
plt.yscale("log")
plt.ylim(1e-7, 10**1)
plt.xlim(0, 25)
plt.xticks(fontweight="bold", fontsize=19)
plt.yticks(fontweight="bold", fontsize=19)
plt.title("Below Cloud Base \nJanuary–June 2022\nCDP Dry Size Distributions", fontsize=19, fontweight="bold")
cbar = plt.colorbar(img)
cbar.set_label("Fraction of Legs", fontsize=19)
cbar.ax.tick_params(labelsize=16)
cbar.set_ticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
cbar.set_ticklabels([r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$'])
plt.tight_layout()
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
plt.plot(common_bins_CDP, average_dN_dD_dry_CDP, color='green', linewidth=2, label='Average Dry Size Distribution (CDP)')
plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"CDP Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("CDP Average Below Cloud Base Dry Size Distribution \n January - June 2022", fontsize=14, fontweight="bold")
plt.show()
#%%
#exponential fit to the averaged dry size distribution
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)
valid_fit_indices_CDP = common_bins_CDP <= 10
x_fit_CDP = common_bins_CDP[valid_fit_indices_CDP]
y_fit_CDP = average_dN_dD_dry_CDP[valid_fit_indices_CDP]
valid_data_indices_CDP = ~np.isnan(y_fit_CDP) & (y_fit_CDP > 0)
x_fit_CDP = x_fit_CDP[valid_data_indices_CDP]
y_fit_CDP = y_fit_CDP[valid_data_indices_CDP]
try:
    popt_CDP, pcov_CDP = curve_fit(exponential, x_fit_CDP, y_fit_CDP, p0=(1e-2, 2))  # Initial guess: (n0=0.01, D=2 μm)
    n0_fit_CDP, D_fit_CDP = popt_CDP 
except RuntimeError:
    print("Exponential fit failed for CDP.")
    n0_fit_CDP, D_fit_CDP = None, None
plt.figure(figsize=(8, 6))
plt.plot(common_bins_CDP, average_dN_dD_dry_CDP, color='green', linewidth=2, label='Average Dry Size Distribution (CDP)')
if n0_fit_CDP is not None and D_fit_CDP is not None:
    plt.plot(x_fit_CDP, exponential(x_fit_CDP, *popt_CDP), 'r--', linewidth=2, 
             label=f'Exponential Fit: $N_0$={n0_fit_CDP:.2e}, $D$={D_fit_CDP:.2f} μm')
plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"CDP Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("CDP Average Below Cloud Base Dry Size Distribution \n January - June 2022", fontsize=14, fontweight="bold")
plt.legend()
plt.show()
if n0_fit_CDP is not None and D_fit_CDP is not None:
    print(f"Fitted Parameters (CDP): N_0 = {n0_fit_CDP:.3e}, D = {D_fit_CDP:.3f} μm")
#%% 
common_bins = np.linspace(2, 25, 35)
# # Adjust bin range and count as needed
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)
valid_fit_indices_CAS = common_bins <= 10
x_fit_CAS = common_bins[valid_fit_indices_CAS]
y_fit_CAS = average_dN_dD_dry[valid_fit_indices_CAS]
valid_data_indices_CAS = ~np.isnan(y_fit_CAS) & (y_fit_CAS > 0)
x_fit_CAS = x_fit_CAS[valid_data_indices_CAS]
y_fit_CAS = y_fit_CAS[valid_data_indices_CAS]
try:
    popt_CAS, _ = curve_fit(exponential, x_fit_CAS, y_fit_CAS, p0=(1e-2, 2))  
    n0_fit_CAS, D_fit_CAS = popt_CAS
except RuntimeError:
    print("Exponential fit failed for CAS.")
    n0_fit_CAS, D_fit_CAS = None, None
valid_fit_indices_CDP = common_bins_CDP <= 10
x_fit_CDP = common_bins_CDP[valid_fit_indices_CDP]
y_fit_CDP = average_dN_dD_dry_CDP[valid_fit_indices_CDP]
valid_data_indices_CDP = ~np.isnan(y_fit_CDP) & (y_fit_CDP > 0)
x_fit_CDP = x_fit_CDP[valid_data_indices_CDP]
y_fit_CDP = y_fit_CDP[valid_data_indices_CDP]
try:
    popt_CDP, _ = curve_fit(exponential, x_fit_CDP, y_fit_CDP, p0=(1e-2, 2))  
    n0_fit_CDP, D_fit_CDP = popt_CDP
except RuntimeError:
    print("Exponential fit failed for CDP.")
    n0_fit_CDP, D_fit_CDP = None, None

plt.figure(figsize=(8, 6))
plt.plot(common_bins, average_dN_dD_dry, color='red', linewidth=3.5, label='CAS Average Dry Size Distribution')
plt.plot(common_bins_CDP, average_dN_dD_dry_CDP, color='green', linewidth=3.5, label='CDP Average Dry Size Distribution')

# # Plot CAS exponential fit
# if n0_fit_CAS is not None and D_fit_CAS is not None:
#     plt.plot(x_fit_CAS, exponential(x_fit_CAS, *popt_CAS), 'r--', linewidth=2,
#              label=f'CAS Exp Fit: $N_0$={n0_fit_CAS:.2e}, $D$={D_fit_CAS:.2f} μm')

# # Plot CDP exponential fit
# if n0_fit_CDP is not None and D_fit_CDP is not None:
#     plt.plot(x_fit_CDP, exponential(x_fit_CDP, *popt_CDP), 'b--', linewidth=2,
#              label=f'CDP Exp Fit: $N_0$={n0_fit_CDP:.2e}, $D$={D_fit_CDP:.2f} μm')

plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Average Below Cloud Base Dry Size Distributions\n January - June 2022", fontsize=14, fontweight="bold")
plt.legend()
plt.show()
if n0_fit_CAS is not None and D_fit_CAS is not None:
    print(f"CAS Fitted Parameters: N_0 = {n0_fit_CAS:.3e}, D = {D_fit_CAS:.3f} μm")

if n0_fit_CDP is not None and D_fit_CDP is not None:
    print(f"CDP Fitted Parameters: N_0 = {n0_fit_CDP:.3e}, D = {D_fit_CDP:.3f} μm")


# %%
#both ambient and dry 

common_bins_CDP = np.linspace(2, 40, 35) 

plt.figure(figsize=(8, 6))
for entry_ambient, entry_dry in zip(Y_CDP_calc, filtered_master_BCB_ddry_CDP):
    ambient_dd_CDP = np.array(bin_center_CDP)
    ambient_dN_dD_CDP = np.array([entry_ambient.get(f'Bin{i:02d}_Y_mean', np.nan) for i in range(0, 30)])

    dry_dd_CDP = np.array(entry_dry['ddry'])
    dry_dN_dD_CDP = np.array(entry_dry['dN/dDdry'])
    valid_ambient = ~np.isnan(ambient_dd_CDP) & ~np.isnan(ambient_dN_dD_CDP)
    valid_dry = ~np.isnan(dry_dd_CDP) & ~np.isnan(dry_dN_dD_CDP)

    if np.sum(valid_ambient) < 2 or np.sum(valid_dry) < 2:
        continue  
    interp_ambient_CDP = interp1d(ambient_dd_CDP[valid_ambient], ambient_dN_dD_CDP[valid_ambient], 
                                  kind='linear', bounds_error=False, fill_value=np.nan)
    interp_dry_CDP = interp1d(dry_dd_CDP[valid_dry], dry_dN_dD_CDP[valid_dry], 
                              kind='linear', bounds_error=False, fill_value=np.nan)

    interpolated_ambient_CDP = interp_ambient_CDP(common_bins_CDP)
    interpolated_dry_CDP = interp_dry_CDP(common_bins_CDP)
    valid_ambient_bins_CDP = (interpolated_ambient_CDP > 0) & ~np.isnan(interpolated_ambient_CDP)
    valid_dry_bins_CDP = (interpolated_dry_CDP > 0) & ~np.isnan(interpolated_dry_CDP)
    filtered_ambient_bins_CDP = common_bins_CDP[valid_ambient_bins_CDP]
    filtered_ambient_values_CDP = interpolated_ambient_CDP[valid_ambient_bins_CDP]

    filtered_dry_bins_CDP = common_bins_CDP[valid_dry_bins_CDP]
    filtered_dry_values_CDP = interpolated_dry_CDP[valid_dry_bins_CDP]
    if len(filtered_ambient_bins_CDP) > 0:
        plt.plot(filtered_ambient_bins_CDP, filtered_ambient_values_CDP, color='blue', alpha=0.3, label="Ambient" if 'Ambient' not in plt.gca().get_legend_handles_labels()[1] else "")

    if len(filtered_dry_bins_CDP) > 0:
        plt.plot(filtered_dry_bins_CDP, filtered_dry_values_CDP, color='red', alpha=0.3, label="Dry" if 'Dry' not in plt.gca().get_legend_handles_labels()[1] else "")
plt.xlabel("Bin Centers Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel("CDP Number Concentration\n(cm$^{-3}$ μm$^{-1}$)", fontsize=19, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-7, 10**1)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base January - June 2022\n Raw CDP Size Distributions", fontsize=14, fontweight="bold")
ambient_legend = mlines.Line2D([], [], color='blue', linewidth=6, label="Ambient")  # Thicker blue line
dry_legend = mlines.Line2D([], [], color='red', linewidth=6, label="Dry")  # Thicker red line
plt.legend(handles=[ambient_legend, dry_legend], fontsize=12, frameon=True)
plt.show()
#%%
#Fitting an exponential to the dry 

def exponential(x, n0, D):
    return n0 * np.exp(-x / D)
dry_exponential_fits_CDP = []

plt.figure(figsize=(8, 6))
for entry in filtered_master_BCB_ddry_CDP:
    ddry_values_CDP = np.array(entry['ddry'])
    dN_dD_dry_CDP = np.array(entry['dN/dDdry'])
    valid_indices = ~np.isnan(ddry_values_CDP) & ~np.isnan(dN_dD_dry_CDP) & (dN_dD_dry_CDP > 0)
    
    if np.sum(valid_indices) < 5:  
        continue 

    try:
        popt, _ = curve_fit(exponential, ddry_values_CDP[valid_indices], dN_dD_dry_CDP[valid_indices], 
                            p0=(1, 5), maxfev=5000)
        n0, D = popt

        dry_exponential_fits_CDP.append({
            'Date': entry['Date'],
            'BCB_start': entry['BCB_start'],
            'BCB_stop': entry['BCB_stop'],
            'Dry_Intercept_n0': n0,
            'Dry_E_folding_D': D
        })
        x_fit = np.linspace(min(ddry_values_CDP[valid_indices]), max(ddry_values_CDP[valid_indices]), 100)
        y_fit = exponential(x_fit, *popt)
        if np.all(y_fit > 1e-33):
            plt.plot(x_fit, y_fit, color='black', alpha=0.2)

    except RuntimeError:
        print(f"Fit could not be performed for date {entry['Date']}")
plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=19, fontweight="bold")
plt.ylabel("CDP Number Concentration\n(cm$^{-3}$ μm$^{-1}$)", fontsize=19, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-7, 10**1.5)
plt.xticks(fontweight="bold", fontsize=19)
plt.yticks(fontweight="bold", fontsize=19)
plt.title("Below Cloud Base\nJanuary-June 2022\n CDP Fitted Dry Size Distributions", fontsize=19, fontweight="bold")
plt.show()
print(f"Total successful dry exponential fits (CDP): {len(dry_exponential_fits_CDP)}")
#%%
#as a heatmap
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)
base_cmap = plt.cm.viridis
colors = base_cmap(np.linspace(0, 1, 256))
colors[:80] = np.linspace([1, 1, 1, 1], colors[80], 80)
fading_viridis = LinearSegmentedColormap.from_list("fading_viridis", colors)

common_bins_fit = np.linspace(2, 25, 100)
fitted_distributions = []
dry_exponential_fits_CDP = []
for entry in filtered_master_BCB_ddry_CDP:
    ddry_values = np.array(entry['ddry'])
    dN_dD = np.array(entry['dN/dDdry'])
    valid = ~np.isnan(ddry_values) & ~np.isnan(dN_dD) & (dN_dD > 0)

    if np.sum(valid) < 5:
        continue

    try:
        popt, _ = curve_fit(exponential, ddry_values[valid], dN_dD[valid], p0=(1, 5), maxfev=5000)
        n0, D = popt

        dry_exponential_fits_CDP.append({
            'Date': entry['Date'],
            'BCB_start': entry['BCB_start'],
            'BCB_stop': entry['BCB_stop'],
            'Dry_Intercept_n0': n0,
            'Dry_E_folding_D': D
        })

        x_fit = common_bins_fit
        y_fit = exponential(x_fit, n0, D)
        y_fit[(y_fit <= 0) | np.isnan(y_fit)] = np.nan
        fitted_distributions.append(y_fit)

    except RuntimeError:
        print(f"Fit could not be performed for date {entry['Date']}")
y_matrix_fit = np.array(fitted_distributions)
y_matrix_fit[np.isnan(y_matrix_fit)] = 0
y_bins_log_fit = np.logspace(-7, 1.5, 150)

H, xedges, yedges = np.histogram2d(
    np.repeat(common_bins_fit, y_matrix_fit.shape[0]),
    y_matrix_fit.T.flatten(),
    bins=[common_bins_fit, y_bins_log_fit]
)
H = H / y_matrix_fit.shape[0]
H_masked = ma.masked_where(H == 0, H)
plt.figure(figsize=(9, 6))
norm = LogNorm(vmin=1e-4, vmax=1)
img = plt.pcolormesh(xedges, yedges, H_masked.T, shading='auto', cmap=fading_viridis, norm=norm)
plt.xlabel("Dry Diameter (μm)", fontsize=19, fontweight="bold")
plt.ylabel("CDP Number Concentration\n(cm$^{-3}$ μm$^{-1}$)", fontsize=19, fontweight="bold")
plt.yscale("log")
plt.ylim(1e-7, 10**1.5)
plt.xlim(0, 25)
plt.xticks(fontweight="bold", fontsize=19)
plt.yticks(fontweight="bold", fontsize=19)
plt.title("Below Cloud Base\nJanuary–June 2022\nCDP Fitted Dry Size Distributions", fontsize=19, fontweight="bold")
cbar = plt.colorbar(img)
cbar.set_label("Fraction of Legs", fontsize=19)
cbar.ax.tick_params(labelsize=16)
cbar.set_ticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
cbar.set_ticklabels([r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$'])
plt.tight_layout()
plt.show()
print(f"Total successful dry exponential fits (CDP): {len(dry_exponential_fits_CDP)}")


# %%
#Only to 10 um 
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)
dry_exponential_fits_10_CDP = []

plt.figure(figsize=(8, 6))
for entry in filtered_master_BCB_ddry_CDP:
    ddry_values_CDP = np.array(entry['ddry'])
    dN_dD_dry_CDP = np.array(entry['dN/dDdry'])

    valid_indices = (ddry_values_CDP <= 10) & ~np.isnan(ddry_values_CDP) & ~np.isnan(dN_dD_dry_CDP)
    if np.sum(valid_indices) == 0:
        dry_exponential_fits_10_CDP.append({
            'Date': entry['Date'],
            'BCB_start': entry['BCB_start'],
            'BCB_stop': entry['BCB_stop'],
            'Dry_Intercept_n0': np.nan,
            'Dry_E_folding_D': np.nan
        })
        continue  

    try:
        popt, _ = curve_fit(exponential, ddry_values_CDP[valid_indices], dN_dD_dry_CDP[valid_indices], 
                            p0=(1, 5), maxfev=5000)
        n0, D = popt

        if D < 0.5 or D > 20:  
            raise RuntimeError("D value out of range")

    except RuntimeError:
        print(f"Fit failed for {entry['Date']} (D={D:.2f})")
        n0, D = np.nan, np.nan 

    dry_exponential_fits_10_CDP.append({
        'Date': entry['Date'],
        'BCB_start': entry['BCB_start'],
        'BCB_stop': entry['BCB_stop'],
        'Dry_Intercept_n0': n0,
        'Dry_E_folding_D': D
    })

    if not np.isnan(n0) and not np.isnan(D):
        x_fit = np.linspace(2, 10, 100) 
        y_fit = exponential(x_fit, n0, D)

        y_fit[y_fit < 1e-15] = np.nan  

        plt.plot(x_fit, y_fit, color='black', alpha=0.2)

plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=15, fontweight="bold")
plt.ylabel(r"CDP Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=15, fontweight="bold")
plt.yscale("log")
plt.xlim(0, 10)
plt.ylim(1e-7, 1e1)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("CDP Below Cloud Base January - June 2022\n Fitted Dry Size Distributions (≤10 µm)", fontsize=15, fontweight="bold")
plt.show()
print(f"Total successful dry exponential fits (CDP ≤10 µm): {len([fit for fit in dry_exponential_fits_10_CDP if not np.isnan(fit['Dry_Intercept_n0'])])}")
#%%
#as a heatmap

def exponential(x, n0, D):
    return n0 * np.exp(-x / D)
base_cmap = plt.cm.viridis
colors = base_cmap(np.linspace(0, 1, 256))
colors[:80] = np.linspace([1, 1, 1, 1], colors[80], 80)
fading_viridis = LinearSegmentedColormap.from_list("fading_viridis", colors)

common_bins_fit_10 = np.linspace(2, 10, 100)
fitted_distributions_10 = []
dry_exponential_fits_10_CDP = []
for entry in filtered_master_BCB_ddry_CDP:
    ddry = np.array(entry['ddry'])
    dN_dD = np.array(entry['dN/dDdry'])

    valid = (ddry <= 10) & ~np.isnan(ddry) & ~np.isnan(dN_dD)

    if np.sum(valid) == 0:
        dry_exponential_fits_10_CDP.append({
            'Date': entry['Date'],
            'BCB_start': entry['BCB_start'],
            'BCB_stop': entry['BCB_stop'],
            'Dry_Intercept_n0': np.nan,
            'Dry_E_folding_D': np.nan
        })
        continue

    try:
        popt, _ = curve_fit(exponential, ddry[valid], dN_dD[valid], p0=(1, 5), maxfev=5000)
        n0, D = popt
        if D < 0.5 or D > 20:
            raise RuntimeError("D out of range")

    except RuntimeError:
        print(f"Fit failed for {entry['Date']}")
        n0, D = np.nan, np.nan

    dry_exponential_fits_10_CDP.append({
        'Date': entry['Date'],
        'BCB_start': entry['BCB_start'],
        'BCB_stop': entry['BCB_stop'],
        'Dry_Intercept_n0': n0,
        'Dry_E_folding_D': D
    })

    if not np.isnan(n0) and not np.isnan(D):
        y_fit = exponential(common_bins_fit_10, n0, D)
        y_fit[(y_fit <= 0) | np.isnan(y_fit)] = np.nan
        fitted_distributions_10.append(y_fit)
y_matrix_10 = np.array(fitted_distributions_10)
y_matrix_10[np.isnan(y_matrix_10)] = 0
y_bins_log_10 = np.logspace(-7, 1.5, 150)

H, xedges, yedges = np.histogram2d(
    np.repeat(common_bins_fit_10, y_matrix_10.shape[0]),
    y_matrix_10.T.flatten(),
    bins=[common_bins_fit_10, y_bins_log_10]
)
H = H / y_matrix_10.shape[0]
H_masked = ma.masked_where(H == 0, H)
plt.figure(figsize=(9, 6))
norm = LogNorm(vmin=1e-4, vmax=1)
img = plt.pcolormesh(xedges, yedges, H_masked.T, shading='auto', cmap=fading_viridis, norm=norm)

plt.xlabel("Dry Diameter (μm)", fontsize=19, fontweight="bold")
plt.ylabel("CDP Concentration\n(cm$^{-3}$ μm$^{-1}$)", fontsize=19, fontweight="bold")
plt.yscale("log")
plt.ylim(1e-7, 10**1.5)
plt.xlim(0, 10)
plt.xticks(fontweight="bold", fontsize=19)
plt.yticks(fontweight="bold", fontsize=19)
plt.title("Below Cloud Base\nJanuary–June 2022\nCDP Fitted Dry Size Distributions (≤10 µm)", fontsize=16, fontweight="bold")
cbar = plt.colorbar(img)
cbar.set_label("Fraction of Legs", fontsize=19)
cbar.ax.tick_params(labelsize=16)
cbar.set_ticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
cbar.set_ticklabels([r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$'])
plt.tight_layout()
plt.show()
successful_fits_10 = len([fit for fit in dry_exponential_fits_10_CDP if not np.isnan(fit['Dry_Intercept_n0'])])
print(f"Total successful dry exponential fits (CDP ≤10 µm): {successful_fits_10}")

#%%
dry_slope_10_CDP=[fit['Dry_E_folding_D'] for fit in dry_exponential_fits_10_CDP if not np.isnan(fit['Dry_E_folding_D'])]
# %%
dry_intercept_10_CDP=[fit['Dry_Intercept_n0'] for fit in dry_exponential_fits_10_CDP if not np.isnan(fit['Dry_Intercept_n0'])]  
#%%
master_CDP = []
for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict = leg_data[i]
    flight_date = leg_dict['Date'] 
    BCB_start = leg_dict['LegIndex_02']['StartTimes']
    BCB_stop = leg_dict['LegIndex_02']['StopTimes']
    sum_flight = summary[i]

    times = sum_flight.Time_mid.values
    winds = sum_flight.Wind_Speed.values
    alts = sum_flight.GPS_altitude.values
    all_CDP_means = []
    for i in range(len(BCB_start)):
        index1_start=None
        index1_end=None  
        start = int(BCB_start[i])
        end = BCB_stop[i]

        wind_alt = {
            'Date': date,
            'BCB_start': start,
            'BCB_end': end,
            'Alts_mean': [],
            'Winds_mean': []
        }

        for i in range(len(times)):
            start9 = int(times[i][0:5])
            if start9 == start:
                index1_start = i
                break
        
    
        for i in range(len(times)):
            end9 = int(times[i][0:5])
            if end9 == end:
                index1_end = i
                break
        
        if index1_start == None:
               
            winds9_mean = np.nan
            alts9_mean = np.nan
        if index1_end == None:
               
            winds9_mean = np.nan
            alts9_mean = np.nan
            break
        else:
            winds9 = winds[index1_start:index1_end]
                
            winds9_mean = np.nanmean(winds9)

            alts9 = alts[index1_start:index1_end]
            alts9_mean = np.nanmean(alts9)
        wind_alt['Winds_mean'].append(winds9_mean)
        wind_alt['Alts_mean'].append(alts9_mean)
        all_CDP_means.append(wind_alt)
    master_CDP.append(all_CDP_means) 
#%%
Z0 = 0.02
Z10 = 10  

corrected_calc_bcb_CDP = {'Date': [], 'Corrected_bcb_windspeed_CDP': []}
for flight in master_CDP:
    for wind_alt in flight:
        date = wind_alt['Date']
        windspeed = wind_alt['Winds_mean']
        altitude = wind_alt['Alts_mean']

        for wind_mean, alt_mean in zip(windspeed, altitude):
           
            new_windspeed = wind_mean * (np.log(Z10/Z0) / np.log(alt_mean / Z0))

            corrected_calc_bcb_CDP['Date'].append(date)
            corrected_calc_bcb_CDP['Corrected_bcb_windspeed_CDP'].append(new_windspeed)
for date, wind_mean in zip(corrected_calc_bcb_CDP['Date'], corrected_calc_bcb_CDP['Corrected_bcb_windspeed_CDP']):
    print(f"Date: {date}, Corrected_bcb_windspeed_CDP: {wind_mean}")
#%%
#histogram of altitudes
def correct_windspeed(windspeed, altitude):
    return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))

combined_data_CDP = []

for i, flight in enumerate(master_CDP):
    for j, wind_alt in enumerate(flight):
        try:
            date = wind_alt['Date']
            wind_mean = wind_alt['Winds_mean'][0]
            alt_mean = wind_alt['Alts_mean'][0]
            BCB_start = wind_alt['BCB_start']
            BCB_stop = wind_alt['BCB_end']
            
            if not np.isnan(alt_mean) and alt_mean > 0:
                corrected_windspeed = correct_windspeed(wind_mean, alt_mean)
            else:
                corrected_windspeed = np.nan

            combined_data_CDP.append({
                'Date': date,
                'BCB_start': BCB_start,
                'BCB_stop': BCB_stop,
                'Windspeed': corrected_windspeed
            })

        except IndexError as e:
            print(f"Index error at i={i}, j={j}: {e}")
            continue
df_combined_CDP = pd.DataFrame(combined_data_CDP)
#%%
#monthly trend of corrected windspeed
df_wind_CDP = pd.DataFrame(combined_data_CDP).copy()
df_wind_CDP = df_wind_CDP[df_wind_CDP["Date"].astype(str).str.startswith("2022-")].copy()
df_wind_CDP["Month"] = df_wind_CDP["Date"].astype(str).str[5:7].astype(int)
df_wind_CDP = df_wind_CDP[df_wind_CDP["Month"].between(1, 6)].copy()
df_wind_sorted = df_wind_CDP.sort_values(["Date", "BCB_start"], kind="mergesort").reset_index(drop=True)
wind = df_wind_sorted["Windspeed"].astype(float).values
x = np.arange(len(df_wind_sorted))
plt.figure(figsize=(12, 4.8))
plt.plot(x, wind, '-')
plt.grid(alpha=0.3)
plt.xlabel("Leg index (sorted by Date, then BCB_start)", fontsize=13, fontweight="bold")
plt.ylabel("Corrected Wind Speed (m/s)", fontsize=13, fontweight="bold")
plt.title("Corrected Wind Speed Timeline (Jan–Jun 2022)\nLegs ordered by Date then BCB_start",
          fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
#%%
#color coded by month with
month_name = {1:"January", 2:"February", 3:"March", 5:"May", 6:"June"}
plt.figure(figsize=(12, 4.8))
for m in sorted(df_wind_sorted["Month"].unique()):
    if m not in month_name:
        continue
    m_mask = (df_wind_sorted["Month"].values == m)
    vals = wind[m_mask]
    good = np.isfinite(vals)
    mean_val = np.mean(vals[good]) if np.any(good) else np.nan
    plt.plot(
        x[m_mask],
        wind[m_mask],
        '-',
        label=f"{month_name[m]} (mean: {mean_val:.2f} m/s)"
    )
plt.grid(alpha=0.3)
plt.ylabel("Wind Speed (m/s)", fontsize=16, fontweight="bold")
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.title("BCB Wind Speed CDP\nJanuary–June 2022 Monthly Means",
          fontsize=18, fontweight="bold")
plt.legend(ncol=2, fontsize=10)
plt.yticks(fontsize=14, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
#%%
dfp = pd.DataFrame(combined_data_CDP).copy()
dfp = dfp[dfp["Date"].astype(str).str.startswith("2022-")].copy()
dfp["Month"] = dfp["Date"].astype(str).str[5:7].astype(int)
dfp = dfp[dfp["Month"].between(1, 6)].copy()
dfp["Date_dt"] = pd.to_datetime(dfp["Date"])
sort_cols = ["Date_dt"]
if "BCB_start" in dfp.columns:
    sort_cols.append("BCB_start")
elif "Min_start" in dfp.columns:
    sort_cols.append("Min_start")
dfp = dfp.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
x = np.arange(len(dfp))
wind_arr = pd.to_numeric(dfp["Windspeed"], errors="coerce").to_numpy()
date_first = dfp.groupby(dfp["Date_dt"].dt.date, sort=False).head(1)
tick_pos = date_first.index.to_numpy()
tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
month_name = {1:"January", 2:"February", 3:"March", 4:"April", 5:"May", 6:"June"}
fig_w = max(22, 0.55 * len(tick_pos))
fig, ax = plt.subplots(figsize=(fig_w, 6.2))

for m in sorted(dfp["Month"].unique()):
    if m not in month_name:
        continue
    m_mask = (dfp["Month"].values == m)
    mean_w = np.nanmean(wind_arr[m_mask])
    ax.plot(
        x[m_mask], wind_arr[m_mask],
        '-', linewidth=1.5,
        label=f"{month_name[m]} (mean: {mean_w:.2f} m/s)"
    )
for p in tick_pos:
    ax.axvline(p, color="k", alpha=0.06, linewidth=1)

ax.grid(alpha=0.3)
ax.set_ylabel("Corrected Wind Speed (m/s)", fontsize=16, fontweight="bold")
ax.set_xlabel("Flight Date", fontsize=16, fontweight="bold")
ax.set_title("BCB Corrected Wind Speed CDP January–June 2022\nMonthly Trend",
             fontsize=18, fontweight="bold")
ax.legend(ncol=2, fontsize=10, loc="upper right")
ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_lab, rotation=60, ha="right", fontsize=7, fontweight="bold")
labels = ax.get_xticklabels()
for i, lab in enumerate(labels):
    lab.set_text("\n" * (i % 4) + lab.get_text())
ax.set_xticklabels([lab.get_text() for lab in labels])
fig.subplots_adjust(bottom=0.40)
fig.tight_layout()
plt.show()
#%%
month_colors = {
    1: "tab:blue",    
    2: "tab:orange",  
    3: "tab:green",   
    5: "tab:red",     
    6: "tab:purple"   
}

dfp = df_wind_CDP.copy()
dfp["Date_dt"] = pd.to_datetime(dfp["Date"])
if "Month" not in dfp.columns:
    dfp["Month"] = dfp["Date"].astype(str).str[5:7].astype(int)
sort_cols = ["Date_dt"]
if "BCB_start" in dfp.columns:
    sort_cols.append("BCB_start")
elif "Min_start" in dfp.columns:
    sort_cols.append("Min_start")
dfp = dfp.sort_values(sort_cols).reset_index(drop=True)
x = np.arange(len(dfp))
wind_arr = pd.to_numeric(dfp["Windspeed"], errors="coerce").to_numpy()
date_first = dfp.groupby(dfp["Date_dt"].dt.date).head(1)
tick_pos = date_first.index.to_numpy()
tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
fig_w = max(22, 0.55 * len(tick_pos))
fig, ax = plt.subplots(figsize=(fig_w, 6.2))
legend_handles = []
month_name = {1:"January", 2:"February", 3:"March", 4:"April", 5:"May", 6:"June"}
for m in sorted(dfp["Month"].unique()):
    if m not in month_name:
        continue

    m_mask = (dfp["Month"].values == m)
    if not np.any(m_mask):
        continue
    c = month_colors.get(m, "k")

    mean_w   = np.nanmean(wind_arr[m_mask])
    median_w = np.nanmedian(wind_arr[m_mask])
    ax.plot(
        x[m_mask], wind_arr[m_mask],
        '-', linewidth=1.5, color=c
    )

    month_x = x[m_mask]
    mid_x = month_x[len(month_x) // 2]
    ax.plot(
        mid_x, mean_w,
        marker="^", linestyle="None",
        markersize=13,
        markerfacecolor=c,
        markeredgecolor=c,   
        markeredgewidth=2.2,
        zorder=6
    )
    ax.plot(
        mid_x + 5, median_w,
        marker="o", linestyle="None",
        markersize=12,
        markerfacecolor=c,
        markeredgecolor=c,  
        markeredgewidth=2.2,
        zorder=6
    )
    legend_handles.extend([
    Line2D([0], [0], color=c, lw=2, label=month_name[m]),
    Line2D([0], [0], marker="^", lw=0, markersize=10,
           markerfacecolor=c, markeredgecolor=c,
           label=f"{month_name[m]} mean = {mean_w:.2f} m/s"),
    Line2D([0], [0], marker="o", lw=0, markersize=10,
           markerfacecolor=c, markeredgecolor=c,
           label=f"{month_name[m]} median = {median_w:.2f} m/s"),
])

for p in tick_pos:
    ax.axvline(p, color="k", alpha=0.06, linewidth=1)
ax.grid(alpha=0.3)
plt.yticks(fontsize=16, fontweight="bold")
ax.set_ylabel("Corrected Wind Speed (m/s)", fontsize=20, fontweight="bold")
ax.set_xlabel("Flight Date", fontsize=20, fontweight="bold")
ax.set_title("BCB Corrected Wind Speed CDP January–June 2022\nMonthly Trend",
             fontsize=20, fontweight="bold")

ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_lab, rotation=60, ha="right", fontsize=7, fontweight="bold")
labels = ax.get_xticklabels()
for i, lab in enumerate(labels):
    base = lab.get_text()
    lab.set_text("\n" * (i % 4) + base)
ax.set_xticklabels([lab.get_text() for lab in labels])
target_dates = {"2022-06-05": 0, "2022-06-07": 3}
labels = ax.get_xticklabels()
for lab in labels:
    txt = lab.get_text().replace("\n", "")
    if txt in target_dates:
        lab.set_text("\n" * target_dates[txt] + txt)
ax.set_xticklabels([lab.get_text() for lab in labels])
ax.legend(
    handles=legend_handles,
    ncol=3,
    fontsize=9,
    loc="upper left",
    frameon=True
)
fig.subplots_adjust(bottom=0.40)
fig.tight_layout()
plt.show()
#%%

#mass to inf
rho_salt = 2200  # kg/m³

def calculate_mass(N0, D):
    N0_m4 = N0 * 10**6  # Convert cm⁻³µm⁻¹ to m⁻⁴
    integrand = lambda d: np.exp(-d / D) * (d * 1e-6)**3  # Convert µm³ → m³
    mass_integral, _ = quad(integrand, 2, np.inf)  # Integrate from 2µm to ∞
    return (np.pi / 6) * rho_salt * N0_m4 * mass_integral  

dry_mass_data_inf_CDP = []

for entry in dry_exponential_fits_CDP:
    date = entry['Date']
    dry_intercept = entry['Dry_Intercept_n0']
    dry_slope = entry['Dry_E_folding_D']

    if dry_slope > 0 and dry_intercept > 0:
        mass_value = calculate_mass(dry_intercept, dry_slope) * 1e9 
        dry_mass_data_inf_CDP.append({
        'Date': date,
        'BCB_start': entry['BCB_start'], 
        'BCB_stop': entry['BCB_stop'], 
        'Dry Slope (D)': dry_slope,
        'Dry Intercept (N0)': dry_intercept,
        'Dry Mass (µg/m³)': mass_value
    })


dry_slopes = np.array([entry['Dry Slope (D)'] for entry in dry_mass_data_inf_CDP])
dry_intercepts = np.array([entry['Dry Intercept (N0)'] for entry in dry_mass_data_inf_CDP])

min_slope_threshold = np.percentile(dry_slopes, 1)

filtered_slopes = dry_slopes[dry_slopes >= min_slope_threshold]
filtered_intercepts = dry_intercepts[dry_slopes >= min_slope_threshold]

x_min, x_max = np.percentile(filtered_slopes, [5, 95])
y_min, y_max = np.percentile(filtered_intercepts, [5, 95])

xgrid_adjusted = np.logspace(np.log10(x_min), np.log10(x_max), 200)
ygrid_adjusted = np.logspace(np.log10(y_min), np.log10(y_max), 200)
D_grid_adjusted, dryintercept_grid_adjusted = np.meshgrid(xgrid_adjusted, ygrid_adjusted)

mass_grid_adjusted = np.zeros_like(D_grid_adjusted)
for i in range(D_grid_adjusted.shape[0]):
    for j in range(D_grid_adjusted.shape[1]):
        mass_grid_adjusted[i, j] = calculate_mass(dryintercept_grid_adjusted[i, j], D_grid_adjusted[i, j]) * 1e9

mass_levels = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]

plt.figure(figsize=(10, 8))
plt.scatter(filtered_slopes, filtered_intercepts, c='blue', s=80, alpha=0.7, label="Dry Data Points")

contour_plot = plt.contour(D_grid_adjusted, dryintercept_grid_adjusted, mass_grid_adjusted, 
                           levels=mass_levels, colors='red', alpha=0.75, linewidths=1.5)

plt.clabel(contour_plot, inline=True, fontsize=13, fmt=lambda x: f"{int(x)} µg/m³", colors='black', inline_spacing=5)
for txt in contour_plot.labelTexts:
    txt.set_fontweight('bold')
    txt.set_rotation(15)

plt.xlabel(r'Dry Slope ($\mu$m)', fontsize=19, fontweight='bold')
plt.ylabel(r'Dry Intercept (cm$^{-3}$ $\mu$m$^{-1}$)', fontsize=19, fontweight='bold')
plt.title('CDP Below Cloud Base January - June 2022\nContours of Dry Mass', fontsize=19, fontweight='bold')
plt.xscale('log')
plt.yscale('log')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
mass_values_ug_inf_CDP = [entry['Dry Mass (µg/m³)'] for entry in dry_mass_data_inf_CDP]
#%%
mass_threshold = 100  # µg/m³
filtered_dry_mass_inf_CDP = [entry for entry in dry_mass_data_inf_CDP if (
    not np.isnan(entry['Dry Slope (D)']) and 
    not np.isnan(entry['Dry Intercept (N0)']) and 
    entry['Dry Mass (µg/m³)'] <= mass_threshold
)]
print(f"Filtered Dry Mass Entries: {len(filtered_dry_mass_inf_CDP)} (after removing masses > {mass_threshold} µg/m³)")
slope_array = np.array([entry['Dry Slope (D)'] for entry in filtered_dry_mass_inf_CDP]).reshape(-1, 1)
intercept_array = np.array([entry['Dry Intercept (N0)'] for entry in filtered_dry_mass_inf_CDP]).reshape(-1, 1)
data_points = np.column_stack((slope_array, intercept_array))
#%%
filtered_mass_values_ug_inf_CDP = [entry['Dry Mass (µg/m³)'] for entry in filtered_dry_mass_inf_CDP]
mean_mass_filtered_inf_CDP = np.mean(filtered_mass_values_ug_inf_CDP)
median_mass_filtered_inf_CDP = np.median(filtered_mass_values_ug_inf_CDP)
print(f"Filtered Mean Mass: {mean_mass_filtered_inf_CDP:.2f} µg/m³")
print(f"Filtered Median Mass: {median_mass_filtered_inf_CDP:.2f} µg/m³")
#%%
#removing corresponding concentration legs based on the mass threshold 
mass_threshold = 100.0  # µg/m^3
def make_leg_key(entry):
    return (entry["Date"], int(entry["BCB_start"]), int(entry["BCB_stop"]))
bad_leg_keys_CDP = {
    make_leg_key(e)
    for e in dry_mass_data_inf_CDP
    if (np.isfinite(e["Dry Mass (µg/m³)"]) and (e["Dry Mass (µg/m³)"] > mass_threshold))
}
print("CDP legs with mass > threshold:", len(bad_leg_keys_CDP))
good_mass_legs_CDP = [
    e for e in dry_mass_data_inf_CDP
    if (np.isfinite(e["Dry Slope (D)"]) and np.isfinite(e["Dry Intercept (N0)"])
        and np.isfinite(e["Dry Mass (µg/m³)"]) and (e["Dry Mass (µg/m³)"] <= mass_threshold))
]

print("CDP legs kept (<= threshold):", len(good_mass_legs_CDP))
#%%
df_cdp_mass = pd.DataFrame(good_mass_legs_CDP)
df_cdp_mass.to_csv("Dry_mass_CDP_legs_mass100.csv", index=False)
print("Saved:", "Dry_mass_CDP_legs_mass100.csv")
#%%
def make_leg_key_ddry(entry):
    return (entry["Date"], int(entry["BCB_start"]), int(entry["BCB_stop"]))

master_ddry_CDP_mass100 = [
    e for e in filtered_master_BCB_ddry_CDP
    if make_leg_key_ddry(e) not in bad_leg_keys_CDP
]
print("Original CDP ddry legs:", len(filtered_master_BCB_ddry_CDP))
print("CDP ddry legs after removing high mass:", len(master_ddry_CDP_mass100))
#%%
common_bins_CDP = np.linspace(2, 25, 35)

sum_interpolated_dN_dD_dry_CDP = np.zeros_like(common_bins_CDP, dtype=float)
count_interpolated_dN_dD_dry_CDP = np.zeros_like(common_bins_CDP, dtype=int)

for entry in master_ddry_CDP_mass100:
    ddry_values_CDP = np.array(entry["ddry"], dtype=float)
    dN_dD_dry_CDP   = np.array(entry["dN/dDdry"], dtype=float)

    valid_indices = np.isfinite(ddry_values_CDP) & np.isfinite(dN_dD_dry_CDP)
    if valid_indices.sum() < 2:
        continue

    interp_func_CDP = interp1d(
        ddry_values_CDP[valid_indices],
        dN_dD_dry_CDP[valid_indices],
        kind="linear",
        bounds_error=False,
        fill_value=np.nan
    )

    interpolated_dN_dD_dry_CDP = interp_func_CDP(common_bins_CDP)
    valid_interpolated_indices = np.isfinite(interpolated_dN_dD_dry_CDP) & (interpolated_dN_dD_dry_CDP > 0)

    sum_interpolated_dN_dD_dry_CDP[valid_interpolated_indices] += interpolated_dN_dD_dry_CDP[valid_interpolated_indices]
    count_interpolated_dN_dD_dry_CDP[valid_interpolated_indices] += 1
average_dN_dD_dry_CDP = np.divide(
    sum_interpolated_dN_dD_dry_CDP,
    count_interpolated_dN_dD_dry_CDP,
    where=count_interpolated_dN_dD_dry_CDP > 0
)
print("Original CDP legs:", len(filtered_master_BCB_ddry_CDP))
print("CDP legs after removing mass > 100:", len(master_ddry_CDP_mass100))
print("CDP legs actually used in avg (count>0 anywhere):", np.max(count_interpolated_dN_dD_dry_CDP))
plt.figure(figsize=(8, 6))
plt.plot(common_bins_CDP, average_dN_dD_dry_CDP, color="green", linewidth=2,
         label="Average Dry Size Distribution (CDP)")
plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"CDP Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("CDP Average Below Cloud Base Dry Size Distribution \n January - June 2022",
          fontsize=14, fontweight="bold")
plt.legend()
plt.show()
#%%
import pickle
with open("CDP_ddry_massLE100.pkl", "wb") as f:
    pickle.dump(master_ddry_CDP_mass100, f)

np.savez("CDP_average_drysize_mass100.npz",
         bins=common_bins_CDP,
         average=average_dN_dD_dry_CDP)
#%%
plt.figure(figsize=(8, 6))
plt.hist(filtered_mass_values_ug_inf_CDP, bins=20, color='blue', edgecolor='black', alpha=0.7)
bins = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 
                 16384, 32768, 65536, 131072])  

plt.xscale('log')  
plt.xlabel('Dry Mass (µg/m³)', fontsize=18, fontweight='bold')
plt.ylabel('Frequency', fontsize=18, fontweight='bold')
plt.title('CDP Dry mass\nBelow Cloud Base all legs', fontsize=18, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.tight_layout()
plt.legend()
plt.show()
#%%
#both CAS and CDP average ambient 

common_bins = np.linspace(2, 45, 55)  # Adjust range and count as needed
sum_bin_means_CDP = np.zeros(len(bin_center_CDP))
count_bin_means_CDP = np.zeros(len(bin_center_CDP))
for entry in Y_CDP_calc:
    bin_means_CDP = np.array([entry.get(f'Bin{i:02d}_Y_mean', np.nan) for i in range(0, 30)], dtype=float)
    valid_indices_CDP = (bin_means_CDP > 0) & ~np.isnan(bin_means_CDP)
    sum_bin_means_CDP[:len(valid_indices_CDP)][valid_indices_CDP] += bin_means_CDP[valid_indices_CDP]
    count_bin_means_CDP[:len(valid_indices_CDP)][valid_indices_CDP] += 1
average_bin_means_CDP = np.divide(sum_bin_means_CDP, count_bin_means_CDP, where=count_bin_means_CDP > 0)
interp_func_CDP = interp1d(bin_center_CDP, average_bin_means_CDP, kind='linear', bounds_error=False, fill_value=np.nan)
interpolated_bin_means_CDP = interp_func_CDP(common_bins)
sum_bin_means_CAS = np.zeros(len(bin_center))
count_bin_means_CAS = np.zeros(len(bin_center))
for entry in Y_BCB_calc:
    bin_means_CAS = np.array([entry.get(f'Bin{i}_Y_mean', np.nan) for i in range(12, 30)], dtype=float)

    valid_indices_CAS = (bin_means_CAS > 0) & ~np.isnan(bin_means_CAS)

    
    sum_bin_means_CAS[:len(valid_indices_CAS)][valid_indices_CAS] += bin_means_CAS[valid_indices_CAS]
    count_bin_means_CAS[:len(valid_indices_CAS)][valid_indices_CAS] += 1
average_bin_means_CAS = np.divide(sum_bin_means_CAS, count_bin_means_CAS, where=count_bin_means_CAS > 0)
interp_func_CAS = interp1d(bin_center, average_bin_means_CAS, kind='linear', bounds_error=False, fill_value=np.nan)
interpolated_bin_means_CAS = interp_func_CAS(common_bins)
plt.figure(figsize=(8, 6))
plt.plot(common_bins, interpolated_bin_means_CDP, color='blue', linewidth=2, label='CDP')
plt.plot(common_bins, interpolated_bin_means_CAS, color='black', linewidth=2, label='CAS')
plt.xlabel("Deliquesced Diameter (μm)", fontsize=19, fontweight="bold")
plt.ylabel("Number Concentration\n(cm$^{-3}$ μm$^{-1}$)", fontsize=19, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xticks(fontweight="bold", fontsize=19)
plt.yticks(fontweight="bold", fontsize=19)
plt.title("Below Cloud Base\nJanuary-June 2022\n Average Ambient Size Distribution", fontsize=19, fontweight="bold")
plt.legend()
plt.legend(fontsize=14, loc='upper right', frameon=False)
plt.show()
#%%
#both CAS and CDP average dry 
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
plt.plot(common_bins, average_dN_dD_dry, color='blue', linewidth=2, label='CAS Average Dry Size Distribution')
plt.plot(common_bins_CDP, average_dN_dD_dry_CDP, color='black', linewidth=2, label='CDP Average Dry Size Distribution')
plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=20, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=20, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xlim(0, 45)
plt.xticks(fontweight="bold", fontsize=20)
plt.yticks(fontweight="bold", fontsize=20)
plt.title("Average Below Cloud Base Dry Size Distributions\nJanuary - June 2022", fontsize=20, fontweight="bold")
plt.legend(fontsize=16)
plt.show()
#%%
#dry mass histograms both CAS and CDP
bins_CAS = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256]) 
bins_CDP = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 
                     16384, 32768, 65536, 131072])  

plt.figure(figsize=(10, 6))
plt.hist(filtered_mass_values_ug_inf, bins=bins_CAS, color='blue', alpha=0.6, 
         edgecolor='blue', density=False, label='CAS Dry Mass')
plt.hist(filtered_mass_values_ug_inf_CDP, bins=bins_CDP, color='black', alpha=0.6, 
         edgecolor='black', density=False, label='CDP Dry Mass')

plt.xscale('log')
plt.xlabel('Dry Mass (µg/m³)', fontsize=18, fontweight='bold')
plt.ylabel('Frequency', fontsize=18, fontweight='bold')
plt.title('Below Cloud Base\nJanuary-June 2022\nDry Mass Distributions', fontsize=18, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.tight_layout()
plt.legend(fontsize=16)
plt.show()

#%%

import random

cas_bin_counts = [78, 174, 62, 54]  # Exact CAS leg counts per bin
grouped_distributions_CDP = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds_CDP = {i: [] for i in range(len(windspeed_bins))}
cdp_bin_leg_indices = {i: [] for i in range(len(windspeed_bins))} 

missing_windspeed_count_CDP = 0
windspeed_bins = [(0, 5), (5.001, 7), (7.001, 9), (9.001, np.inf)]
common_bins=np.linspace(2, 10, 10)
for leg_idx, entry in enumerate(dry_exponential_fits_10_CDP):
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']

    n0 = entry['Dry_Intercept_n0']
    D = entry['Dry_E_folding_D']

   
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) & 
        (df_combined['BCB_start'] == BCB_start) & 
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    if windspeed_entry.empty or np.isnan(n0) or np.isnan(D):
        missing_windspeed_count_CDP += 1
        continue 

    windspeed = windspeed_entry['Windspeed'].values[0]

    size_dist_CDP = n0 * np.exp(-common_bins / D) 

    
    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:
            grouped_distributions_CDP[idx].append(size_dist_CDP)
            mean_windspeeds_CDP[idx].append(windspeed)
            cdp_bin_leg_indices[idx].append(leg_idx) 
            break

final_grouped_CDP = {i: [] for i in range(len(windspeed_bins))}
final_mean_windspeeds_CDP = {i: [] for i in range(len(windspeed_bins))}

for idx in range(len(windspeed_bins)):
    if len(grouped_distributions_CDP[idx]) >= cas_bin_counts[idx]:
        selected_indices = random.sample(range(len(grouped_distributions_CDP[idx])), cas_bin_counts[idx])
    else:
        selected_indices = list(range(len(grouped_distributions_CDP[idx])))  # Use all if not enough

    final_grouped_CDP[idx] = [grouped_distributions_CDP[idx][i] for i in selected_indices]
    final_mean_windspeeds_CDP[idx] = [mean_windspeeds_CDP[idx][i] for i in selected_indices]
print("\n✅ **Windspeed Verification:**")
for idx, (low, high) in enumerate(windspeed_bins):
    if final_mean_windspeeds_CDP[idx]:
        min_ws, max_ws = min(final_mean_windspeeds_CDP[idx]), max(final_mean_windspeeds_CDP[idx])
        print(f"Windspeed bin {idx} ({low} - {high} m/s): {len(final_grouped_CDP[idx])} legs")
        print(f"  - Selected windspeed range: {min_ws:.2f} to {max_ws:.2f} m/s")
plt.figure(figsize=(10, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if final_grouped_CDP[idx]:
        avg_distribution_CDP = np.mean(final_grouped_CDP[idx], axis=0) 
        avg_windspeed_CDP = np.mean(final_mean_windspeeds_CDP[idx])
        num_legs_CDP = len(final_grouped_CDP[idx])

        plt.plot(common_bins, avg_distribution_CDP, label=f"{avg_windspeed_CDP:.1f} m/s, n={num_legs_CDP} legs", linewidth=2.5)

plt.yscale('log')
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=16, fontweight="bold")
plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=16, fontweight="bold")
plt.title('Dry Size Distributions Binned by Average Wind Speed (CDP, Matched to CAS)', fontweight='bold', fontsize=17)
plt.legend(title=r"Average wind speed m s$^{-1}$")
plt.tight_layout()
plt.ylim(1e-4, 10**0)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.show()
final_total_legs = sum(len(group) for group in final_grouped_CDP.values())
print(f"\n🎯 **Final CDP legs actually plotted (should match CAS 368): {final_total_legs}**")

for idx, count in enumerate(cas_bin_counts):
    print(f"Windspeed bin {idx} ({windspeed_bins[idx]} m/s): {len(final_grouped_CDP[idx])} legs (CAS had {count})")










# %%
common_bins=np.linspace(2, 10, 25)
windspeed_bins = [
    (0, 2.5),
    (2.501, 3.5),
    (3.501, 5),
    (5.001, 7),
    (7.001, 9),
    (9.001, np.inf)
]
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
cas_bin_counts = [54, 46, 74, 93, 62, 39] 
grouped_distributions_CDP = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds_CDP = {i: [] for i in range(len(windspeed_bins))}
cdp_bin_leg_indices = {i: [] for i in range(len(windspeed_bins))} 

missing_windspeed_count_CDP = 0
common_bins = np.linspace(2, 10, 25)
for leg_idx, entry in enumerate(dry_exponential_fits_10_CDP):
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']

    n0 = entry['Dry_Intercept_n0']
    D = entry['Dry_E_folding_D']

    
    windspeed_entry = df_combined_CDP[
        (df_combined_CDP['Date'] == date) & 
        (df_combined_CDP['BCB_start'] == BCB_start) & 
        (df_combined_CDP['BCB_stop'] == BCB_stop)
    ]

    if windspeed_entry.empty or np.isnan(n0) or np.isnan(D):
        missing_windspeed_count_CDP += 1
        continue  

    windspeed = windspeed_entry['Windspeed'].values[0]
    size_dist_CDP = n0 * np.exp(-common_bins / D) 

    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:
            grouped_distributions_CDP[idx].append(size_dist_CDP)
            mean_windspeeds_CDP[idx].append(windspeed)
            cdp_bin_leg_indices[idx].append(leg_idx) 
            break
final_grouped_CDP = {i: [] for i in range(len(windspeed_bins))}
final_mean_windspeeds_CDP = {i: [] for i in range(len(windspeed_bins))}

for idx in range(len(windspeed_bins)):
    if len(grouped_distributions_CDP[idx]) >= cas_bin_counts[idx]:
        selected_indices = random.sample(range(len(grouped_distributions_CDP[idx])), cas_bin_counts[idx])
    else:
        selected_indices = list(range(len(grouped_distributions_CDP[idx]))) 

    final_grouped_CDP[idx] = [grouped_distributions_CDP[idx][i] for i in selected_indices]
    final_mean_windspeeds_CDP[idx] = [mean_windspeeds_CDP[idx][i] for i in selected_indices]
plt.figure(figsize=(10, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if final_grouped_CDP[idx]:
        avg_distribution_CDP = np.mean(final_grouped_CDP[idx], axis=0) 
        avg_windspeed_CDP = np.mean(final_mean_windspeeds_CDP[idx])
        num_legs_CDP = len(final_grouped_CDP[idx])

        plt.plot(common_bins, avg_distribution_CDP, label=f"{avg_windspeed_CDP:.1f} m/s, n={num_legs_CDP} legs", linewidth=2.5)

plt.yscale('log')
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=16, fontweight="bold")
plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=16, fontweight="bold")
plt.title('Dry Size Distributions Binned by Average Wind Speed (CDP)', fontweight='bold', fontsize=17)
plt.legend(title=r"Average wind speed m s$^{-1}$")
plt.tight_layout()
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)  
plt.ylim(1e-4, 10**0)
#%%
#adding error bars
windspeed_bins = [
    (0, 2.5),
    (2.501, 3.5),
    (3.501, 5),
    (5.001, 7),
    (7.001, 9),
    (9.001, np.inf)
]
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
cas_bin_counts = [54, 46, 74, 93, 62, 39] 
grouped_distributions_CDP = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds_CDP = {i: [] for i in range(len(windspeed_bins))}

missing_windspeed_count_CDP = 0
common_bins = np.linspace(2, 10, 25)
for leg_idx, entry in enumerate(dry_exponential_fits_10_CDP):
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']

    n0 = entry['Dry_Intercept_n0']
    D = entry['Dry_E_folding_D']
    windspeed_entry = df_combined_CDP[
        (df_combined_CDP['Date'] == date) & 
        (df_combined_CDP['BCB_start'] == BCB_start) & 
        (df_combined_CDP['BCB_stop'] == BCB_stop)
    ]

    if windspeed_entry.empty or np.isnan(n0) or np.isnan(D):
        missing_windspeed_count_CDP += 1
        continue 

    windspeed = windspeed_entry['Windspeed'].values[0]

    size_dist_CDP = n0 * np.exp(-common_bins / D) 

    
    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:
            grouped_distributions_CDP[idx].append(size_dist_CDP)
            mean_windspeeds_CDP[idx].append(windspeed)
            break
final_grouped_CDP = {i: [] for i in range(len(windspeed_bins))}
final_mean_windspeeds_CDP = {i: [] for i in range(len(windspeed_bins))}

for idx in range(len(windspeed_bins)):
    if len(grouped_distributions_CDP[idx]) >= cas_bin_counts[idx]:
        selected_indices = random.sample(range(len(grouped_distributions_CDP[idx])), cas_bin_counts[idx])
    else:
        selected_indices = list(range(len(grouped_distributions_CDP[idx])))  

    final_grouped_CDP[idx] = [grouped_distributions_CDP[idx][i] for i in selected_indices]
    final_mean_windspeeds_CDP[idx] = [mean_windspeeds_CDP[idx][i] for i in selected_indices]

plt.figure(figsize=(10, 8))
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']  
legend_entries = []

for idx, (low, high) in enumerate(windspeed_bins):
    if final_grouped_CDP[idx]:
        avg_distribution_CDP = np.mean(final_grouped_CDP[idx], axis=0) 
        std_dev_CDP = np.std(final_grouped_CDP[idx], axis=0)
        std_error_CDP = std_dev_CDP / np.sqrt(len(final_grouped_CDP[idx])) 

        avg_windspeed_CDP = np.mean(final_mean_windspeeds_CDP[idx])
        num_legs_CDP = len(final_grouped_CDP[idx])
        avg_se = np.mean(std_error_CDP) 
        plt.plot(common_bins, avg_distribution_CDP, 
                 label=f"{avg_windspeed_CDP:.1f} m/s, SE={avg_se:.3f}", 
                 linewidth=2.5, color=colors[idx])

        plt.errorbar(common_bins, avg_distribution_CDP, yerr=std_error_CDP, fmt='o', color=colors[idx],
                     capsize=3, capthick=1.5, markersize=5, label=None)
plt.yscale('log')
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=21, fontweight="bold")
plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=22, fontweight="bold")
plt.title('CDP Dry Size Distributions \nBinned by Average Wind Speed', fontweight='bold', fontsize=21)
plt.legend(title=r"Wind Speed & Errors (m s$^{-1}$)", title_fontsize=19, fontsize=18, frameon=True)
plt.xticks(fontweight="bold", fontsize=21)
plt.yticks(fontweight="bold", fontsize=21)
plt.ylim(1e-4, 10**0)
plt.tight_layout()
plt.show()
#%%
from scipy.stats import pearsonr

#%%
sample_area_cm2 = 0.323 / 100  # convert mm² to cm²
plane_speed_cm_s = 1.2e4       # 120 m/s
sampling_time_s = 198          # 3.3 minutes
sample_volume = sample_area_cm2 * plane_speed_cm_s * sampling_time_s
windspeed_bins = [
    (0, 2.5),
    (2.501, 3.5),
    (3.501, 5),
    (5.001, 7),
    (7.001, 9),
    (9.001, np.inf)
]
grouped_relative_total_errors_CDP = {i: [] for i in range(len(windspeed_bins))}
missing_windspeed_count_CDP = 0
common_bins = np.linspace(2, 10, 25)
for entry in dry_exponential_fits_10_CDP:
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    n0 = entry['Dry_Intercept_n0']
    D = entry['Dry_E_folding_D']

    windspeed_entry = df_combined_CDP[
        (df_combined_CDP['Date'] == date) &
        (df_combined_CDP['BCB_start'] == BCB_start) &
        (df_combined_CDP['BCB_stop'] == BCB_stop)
    ]

    if windspeed_entry.empty or np.isnan(n0) or np.isnan(D):
        missing_windspeed_count_CDP += 1
        continue

    windspeed = windspeed_entry['Windspeed'].values[0]
    size_dist = n0 * np.exp(-common_bins / D)
    N_counts = size_dist * sample_volume
    N_counts[N_counts <= 0] = np.nan

    total_N = np.nansum(N_counts)
    if total_N > 0:
        relative_total_error = 1 / np.sqrt(total_N)
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed < high:
                grouped_relative_total_errors_CDP[idx].append(relative_total_error)
                break

print("=== CDP Median Relative Total Counting Error by Windspeed Bin ===")
median_relative_errors_per_bin_CDP = []
for idx, errors in grouped_relative_total_errors_CDP.items():
    if errors:
        median_rel_err = np.nanmedian(errors)
    else:
        median_rel_err = np.nan
    median_relative_errors_per_bin_CDP.append(median_rel_err)
    print(f"Windspeed bin {idx} ({windspeed_bins[idx]}): median relative total error = {median_rel_err:.4f}")

print(f"\nTotal legs with missing windspeed data: {missing_windspeed_count_CDP}")
counting_errors_CDP = np.array(median_relative_errors_per_bin_CDP[:len(total_concentrations_CDP)]) * total_concentrations_CDP

#%%
#computing regression
def linear_model(x, m, b):
    return m * x + b
avg_windspeeds_CDP = []
total_concentrations_CDP = []
bin_edges_CDP = np.linspace(2, 10, 10)  # Ensure correct bin edges from 2 to 10 μm
bin_widths_CDP = np.diff(bin_edges_CDP)  # Compute widths between bin edges
for idx, (low, high) in enumerate(windspeed_bins):
    if final_grouped_CDP[idx]:  # Use only selected CDP legs
        avg_windspeed_CDP = np.mean(final_mean_windspeeds_CDP[idx])  # Average windspeed for this bin

        avg_concentration_per_leg_CDP = [np.sum(dist * bin_widths_CDP) for dist in final_grouped_CDP[idx]]
        avg_concentration_CDP = np.mean(avg_concentration_per_leg_CDP)  # Average over all legs in this bin

        avg_windspeeds_CDP.append(avg_windspeed_CDP)
        total_concentrations_CDP.append(avg_concentration_CDP)
windspeed_values_CDP = np.array(avg_windspeeds_CDP)
total_concentrations_CDP = np.array(total_concentrations_CDP)
popt_CDP, _ = curve_fit(linear_model, windspeed_values_CDP, total_concentrations_CDP)
m_fit_CDP, b_fit_CDP = popt_CDP
residuals_CDP = total_concentrations_CDP - linear_model(windspeed_values_CDP, *popt_CDP)
ss_res_CDP = np.sum(residuals_CDP**2)
ss_tot_CDP = np.sum((total_concentrations_CDP - np.mean(total_concentrations_CDP))**2)
r_squared_CDP = 1 - (ss_res_CDP / ss_tot_CDP)
plt.figure(figsize=(8, 6))
for idx in range(len(windspeed_bins)):
    plt.scatter(windspeed_values_CDP[idx], total_concentrations_CDP[idx], 
                color=colors[idx], s=100, edgecolor='black', zorder=3)
x_fit_CDP = np.linspace(min(windspeed_values_CDP), max(windspeed_values_CDP), 100)
y_fit_CDP = linear_model(x_fit_CDP, *popt_CDP)
plt.plot(x_fit_CDP, y_fit_CDP, 'r-', label=f'Fit: y = {m_fit_CDP:.3f}x + {b_fit_CDP:.3f}, R² = {r_squared_CDP:.2f}')
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=16, fontweight='bold')
plt.ylabel("Total Wind Speed Bin Concentration (cm$^{-3}$)", fontsize=16, fontweight='bold')
plt.title("Wind Speed and Total Concentration Correlation (CDP)", fontsize=16, fontweight='bold')
legend_labels_CDP = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[idx], markersize=10, 
                                 label=f"{windspeed_values_CDP[idx]:.1f} m/s") for idx in range(len(windspeed_bins))]

plt.legend(handles=legend_labels_CDP + 
           [plt.Line2D([0], [0], color='red', label=f'Fit: y = {m_fit_CDP:.3f}x + {b_fit_CDP:.3f}, R² = {r_squared_CDP:.2f}')], 
           title="Wind Speed Bins", title_fontsize=14, fontsize=14)

plt.tight_layout()
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.show()
print(f"Slope (m): {m_fit_CDP:.3f}")
print(f"Intercept (b): {b_fit_CDP:.3f}")
print(f"R² value: {r_squared_CDP:.2f}")

# %%
# Computing regression for CDP
def linear_model(x, m, b):
    return m * x + b
avg_windspeeds_CDP = []
total_concentrations_CDP = []
bin_edges_CDP = np.linspace(2, 10, len(common_bins) + 1) 
bin_widths_CDP = np.diff(bin_edges_CDP)  
for idx, (low, high) in enumerate(windspeed_bins):
    if final_grouped_CDP[idx]: 
        avg_windspeed_CDP = np.mean(final_mean_windspeeds_CDP[idx])

        avg_concentration_per_leg_CDP = [np.sum(dist[:len(bin_widths_CDP)] * bin_widths_CDP) for dist in final_grouped_CDP[idx]]
        avg_concentration_CDP = np.mean(avg_concentration_per_leg_CDP)  

        avg_windspeeds_CDP.append(avg_windspeed_CDP)
        total_concentrations_CDP.append(avg_concentration_CDP)
windspeed_values_CDP = np.array(avg_windspeeds_CDP)
total_concentrations_CDP = np.array(total_concentrations_CDP)
popt_CDP, _ = curve_fit(linear_model, windspeed_values_CDP, total_concentrations_CDP)
m_fit_CDP, b_fit_CDP = popt_CDP
residuals_CDP = total_concentrations_CDP - linear_model(windspeed_values_CDP, *popt_CDP)
ss_res_CDP = np.sum(residuals_CDP**2)
ss_tot_CDP = np.sum((total_concentrations_CDP - np.mean(total_concentrations_CDP))**2)
r_squared_CDP = 1 - (ss_res_CDP / ss_tot_CDP)
plt.figure(figsize=(8, 6))
for idx in range(len(windspeed_bins)):
    plt.scatter(windspeed_values_CDP[idx], total_concentrations_CDP[idx], 
                color="green", s=100, edgecolor='black', zorder=3)
x_fit_CDP = np.linspace(min(windspeed_values_CDP), max(windspeed_values_CDP), 100)
y_fit_CDP = linear_model(x_fit_CDP, *popt_CDP)
plt.plot(x_fit_CDP, y_fit_CDP, 'r-', label=f'Fit: y = {m_fit_CDP:.3f}x + {b_fit_CDP:.3f}, R² = {r_squared_CDP:.2f}')
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=16, fontweight='bold')
plt.ylabel("Total Wind Speed Bin Concentration (cm$^{-3}$)", fontsize=16, fontweight='bold')
plt.title("CDP Wind Speed and Total Concentration Correlation", fontsize=16, fontweight='bold')
legend_labels_CDP = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[idx], markersize=10, 
                                 label=f"{windspeed_values_CDP[idx]:.1f} m/s") for idx in range(len(windspeed_bins))]

plt.legend(handles=legend_labels_CDP + 
           [plt.Line2D([0], [0], color='black', label=f'Fit: y = {m_fit_CDP:.3f}x + {b_fit_CDP:.3f}, R² = {r_squared_CDP:.2f}')], 
           title="Wind Speed Bins", title_fontsize=14, fontsize=14)

plt.tight_layout()
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.ylim(0.1,0.7)
plt.xlim(0,10)
plt.show()
print(f"Slope (m): {m_fit_CDP:.3f}")
print(f"Intercept (b): {b_fit_CDP:.3f}")
print(f"R² value: {r_squared_CDP:.2f}")
#%%
#addingR value
def linear_model(x, m, b):
    return m * x + b
avg_windspeeds_CDP = []
total_concentrations_CDP = []
bin_edges_CDP = np.linspace(2, 10, len(common_bins) + 1)  
bin_widths_CDP = np.diff(bin_edges_CDP) 
for idx, (low, high) in enumerate(windspeed_bins):
    if final_grouped_CDP[idx]:  
        avg_windspeed_CDP = np.mean(final_mean_windspeeds_CDP[idx]) 

        avg_concentration_per_leg_CDP = [np.sum(dist[:len(bin_widths_CDP)] * bin_widths_CDP) for dist in final_grouped_CDP[idx]]
        avg_concentration_CDP = np.mean(avg_concentration_per_leg_CDP)

        avg_windspeeds_CDP.append(avg_windspeed_CDP)
        total_concentrations_CDP.append(avg_concentration_CDP)
windspeed_values_CDP = np.array(avg_windspeeds_CDP)
total_concentrations_CDP = np.array(total_concentrations_CDP)
popt_CDP, _ = curve_fit(linear_model, windspeed_values_CDP, total_concentrations_CDP)
m_fit_CDP, b_fit_CDP = popt_CDP
residuals_CDP = total_concentrations_CDP - linear_model(windspeed_values_CDP, *popt_CDP)
ss_res_CDP = np.sum(residuals_CDP**2)
ss_tot_CDP = np.sum((total_concentrations_CDP - np.mean(total_concentrations_CDP))**2)
r_squared_CDP = 1 - (ss_res_CDP / ss_tot_CDP)
pearson_corr_CDP, _ = pearsonr(windspeed_values_CDP, total_concentrations_CDP)
plt.figure(figsize=(8, 6))
plt.scatter(windspeed_values_CDP, total_concentrations_CDP, 
            color="green", s=100, edgecolor='black', zorder=3, label="CDP")
x_fit_CDP = np.linspace(min(windspeed_values_CDP), max(windspeed_values_CDP), 100)
y_fit_CDP = linear_model(x_fit_CDP, *popt_CDP)
plt.plot(x_fit_CDP, y_fit_CDP, 'black', linewidth=2, 
         label=f'Fit: y = {m_fit_CDP:.3f}x + {b_fit_CDP:.3f}\nR² = {r_squared_CDP:.2f}, R = {pearson_corr_CDP:.2f}')
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=16, fontweight='bold')
plt.ylabel("Total Wind Speed Bin Concentration (cm$^{-3}$)", fontsize=16, fontweight='bold')
plt.title("CDP Wind Speed and Total Concentration Correlation", fontsize=16, fontweight='bold')
plt.legend(fontsize=14, title_fontsize=14, loc='best', frameon=True)
plt.tight_layout()
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.ylim(0.1, 0.7)
plt.xlim(0, 10)
plt.show()
print(f"Slope (m): {m_fit_CDP:.3f}")
print(f"Intercept (b): {b_fit_CDP:.3f}")
print(f"R² value: {r_squared_CDP:.2f}")
print(f"R value (Pearson correlation): {pearson_corr_CDP:.3f}")
#%%
# Define linear regression function
def linear_model(x, m, b):
    return m * x + b
avg_windspeeds_CDP = []
total_concentrations_CDP = []
standard_errors_CDP = []
bin_edges_CDP = np.linspace(2, 10, len(common_bins) + 1) 
bin_widths_CDP = np.diff(bin_edges_CDP)  
for idx, (low, high) in enumerate(windspeed_bins):
    if final_grouped_CDP[idx]: 
        avg_windspeed_CDP = np.mean(final_mean_windspeeds_CDP[idx])

        avg_concentration_per_leg_CDP = [np.sum(dist[:len(bin_widths_CDP)] * bin_widths_CDP) for dist in final_grouped_CDP[idx]]
        avg_concentration_CDP = np.mean(avg_concentration_per_leg_CDP) 
        std_concentration_CDP = np.std(avg_concentration_per_leg_CDP, ddof=1) 
        N_CDP = len(avg_concentration_per_leg_CDP) 
        SE_concentration_CDP = std_concentration_CDP / np.sqrt(N_CDP) 

        avg_windspeeds_CDP.append(avg_windspeed_CDP)
        total_concentrations_CDP.append(avg_concentration_CDP)
        standard_errors_CDP.append(SE_concentration_CDP)
windspeed_values_CDP = np.array(avg_windspeeds_CDP)
total_concentrations_CDP = np.array(total_concentrations_CDP)
standard_errors_CDP = np.array(standard_errors_CDP)
popt_CDP, _ = curve_fit(linear_model, windspeed_values_CDP, total_concentrations_CDP)
m_fit_CDP, b_fit_CDP = popt_CDP
residuals_CDP = total_concentrations_CDP - linear_model(windspeed_values_CDP, *popt_CDP)
ss_res_CDP = np.sum(residuals_CDP**2)
ss_tot_CDP = np.sum((total_concentrations_CDP - np.mean(total_concentrations_CDP))**2)
r_squared_CDP = 1 - (ss_res_CDP / ss_tot_CDP)
pearson_corr_CDP, _ = pearsonr(windspeed_values_CDP, total_concentrations_CDP)
plt.figure(figsize=(8, 6))
plt.errorbar(windspeed_values_CDP, total_concentrations_CDP, 
             yerr=standard_errors_CDP, fmt='o', color='green', 
             markersize=10, capsize=5, capthick=2, label="CDP", 
             ecolor='black', elinewidth=1.5, zorder=3)
x_fit_CDP = np.linspace(min(windspeed_values_CDP), max(windspeed_values_CDP), 100)
y_fit_CDP = linear_model(x_fit_CDP, *popt_CDP)
plt.plot(x_fit_CDP, y_fit_CDP, 'black', linewidth=2, 
         label=f'Fit: y = {m_fit_CDP:.3f}x + {b_fit_CDP:.3f}\nR² = {r_squared_CDP:.2f}, R = {pearson_corr_CDP:.2f}')
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=19, fontweight='bold')
plt.ylabel("Total Wind Speed Bin \nConcentration (cm$^{-3}$)", fontsize=19, fontweight='bold')
plt.title("CDP Below Cloud Base \nJanuary-June 2022", fontsize=19, fontweight='bold')
plt.legend(fontsize=16, title_fontsize=14, loc='best', frameon=True)
plt.tight_layout()
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.ylim(0.0, 0.7)
plt.xlim(0, 12)
plt.show()
print(f"Slope (m): {m_fit_CDP:.3f}")
print(f"Intercept (b): {b_fit_CDP:.3f}")
print(f"R² value: {r_squared_CDP:.2f}")
print(f"R value (Pearson correlation): {pearson_corr_CDP:.3f}")
#%%
#total error in concentration
sample_area_cm2 = 0.323 / 100   # cm²  (converted from mm²)
plane_speed_cm_s = 1.2e4        # cm/s
sampling_time_s = 198           # s
sample_volume_cm3 = sample_area_cm2 * plane_speed_cm_s * sampling_time_s

print(f"CDP Sample Volume (cm³): {sample_volume_cm3:.2f}")
grouped_conc_errors_CDP = {i: [] for i in range(len(windspeed_bins))}

for idx, (low, high) in enumerate(windspeed_bins):
    if final_grouped_CDP[idx]:
        for dist in final_grouped_CDP[idx]:
            counts_per_bin = np.array(dist[:len(bin_widths_CDP)]) * sample_volume_cm3
            
            conc_error_bins = np.sqrt(counts_per_bin) / sample_volume_cm3  # cm⁻³
            
            conc_error_total = np.sqrt(np.sum(conc_error_bins**2))
            
            grouped_conc_errors_CDP[idx].append(conc_error_total)
median_conc_errors_CDP = [np.nanmedian(errs) if errs else np.nan 
                          for errs in grouped_conc_errors_CDP.values()]

print("\n✅ CDP Concentration Counting Errors per Wind Speed Bin:")
for idx, (bounds, err) in enumerate(zip(windspeed_bins, median_conc_errors_CDP)):
    label = f"{bounds[0]}–{bounds[1] if bounds[1] != np.inf else '∞'} m/s"
    if not np.isnan(err):
        print(f"  Bin {idx} ({label}): {err:.4f} cm⁻³")
    else:
        print(f"  Bin {idx} ({label}): NaN (no data)")
counting_errors_CDP = np.array(median_conc_errors_CDP)
#%%
def linear_model(x, m, b):
    return m * x + b

popt_CDP, pcov_CDP = curve_fit(linear_model, windspeed_values_CDP, total_concentrations_CDP)
m_fit_CDP, b_fit_CDP = popt_CDP
m_err_CDP, b_err_CDP = np.sqrt(np.diag(pcov_CDP))

residuals_CDP = total_concentrations_CDP - linear_model(windspeed_values_CDP, *popt_CDP)
ss_res_CDP = np.sum(residuals_CDP**2)
ss_tot_CDP = np.sum((total_concentrations_CDP - np.mean(total_concentrations_CDP))**2)
r_squared_CDP = 1 - (ss_res_CDP / ss_tot_CDP)
pearson_corr_CDP, _ = pearsonr(windspeed_values_CDP, total_concentrations_CDP)

x_fit_CDP = np.linspace(min(windspeed_values_CDP), max(windspeed_values_CDP), 100)
y_fit_CDP = linear_model(x_fit_CDP, *popt_CDP)
plt.figure(figsize=(8, 6))
plt.errorbar(windspeed_values_CDP, total_concentrations_CDP,
             yerr=standard_errors_CDP,
             fmt='o', color='blue',  
             markersize=10, capsize=5, capthick=2,
             ecolor='blue', elinewidth=1.5,
             label="CDP Standard Error", zorder=3)
plt.errorbar(windspeed_values_CDP + 0.4,  # horizontal offset so bars don't overlap
             total_concentrations_CDP,
             yerr=counting_errors_CDP,  # from your new calculation
             fmt='o', color='brown',  # brown markers
             markersize=10, capsize=5, capthick=2,
             ecolor='black', elinewidth=1.5,
             label="CDP Error in Total Concentration", zorder=2)

plt.plot(x_fit_CDP, y_fit_CDP, '-', color='blue', linewidth=2.5,
         label=f'Fit: y = ({m_fit_CDP:.3f}±{m_err_CDP:.3f})x + {b_fit_CDP:.3f}\n'
               f'R² = {r_squared_CDP:.2f}, R = {pearson_corr_CDP:.2f}')
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=19, fontweight='bold')
plt.ylabel("Total Wind Speed Bin \nConcentration (cm$^{-3}$)", fontsize=19, fontweight='bold')
plt.title("CDP Below Cloud Base \nJanuary–June 2022", fontsize=19, fontweight='bold')
plt.legend(fontsize=14, frameon=False, loc='upper left')
plt.tight_layout()
plt.xlim(0, 12)
plt.ylim(0, 0.7)
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')
plt.show()
print(f"Slope (m): {m_fit_CDP:.3f} ± {m_err_CDP:.3f}")
print(f"Intercept (b): {b_fit_CDP:.3f}")
print(f"R² value: {r_squared_CDP:.2f}")
print(f"R value (Pearson correlation): {pearson_corr_CDP:.3f}")
#%%
#overlaying CDP and CAS concentration correlation
def linear_model(x, m, b):
    return m * x + b
popt_CAS, pcov_CAS = curve_fit(linear_model, windspeed_values, total_concentrations)
m_fit_CAS, b_fit_CAS = popt_CAS
m_err_CAS, b_err_CAS = np.sqrt(np.diag(pcov_CAS))

residuals_CAS = total_concentrations - linear_model(windspeed_values, *popt_CAS)
ss_res_CAS = np.sum(residuals_CAS**2)
ss_tot_CAS = np.sum((total_concentrations - np.mean(total_concentrations))**2)
r_squared_CAS = 1 - (ss_res_CAS / ss_tot_CAS)
r_value_CAS = np.sign(m_fit_CAS) * np.sqrt(r_squared_CAS)

x_fit_CAS = np.linspace(min(windspeed_values), max(windspeed_values), 100)
y_fit_CAS = linear_model(x_fit_CAS, *popt_CAS)
popt_CDP, pcov_CDP = curve_fit(linear_model, windspeed_values_CDP, total_concentrations_CDP)
m_fit_CDP, b_fit_CDP = popt_CDP
m_err_CDP, b_err_CDP = np.sqrt(np.diag(pcov_CDP))

residuals_CDP = total_concentrations_CDP - linear_model(windspeed_values_CDP, *popt_CDP)
ss_res_CDP = np.sum(residuals_CDP**2)
ss_tot_CDP = np.sum((total_concentrations_CDP - np.mean(total_concentrations_CDP))**2)
r_squared_CDP = 1 - (ss_res_CDP / ss_tot_CDP)
pearson_corr_CDP, _ = pearsonr(windspeed_values_CDP, total_concentrations_CDP)

x_fit_CDP = np.linspace(min(windspeed_values_CDP), max(windspeed_values_CDP), 100)
y_fit_CDP = linear_model(x_fit_CDP, *popt_CDP)
plt.figure(figsize=(9, 7))
plt.errorbar(windspeed_values, total_concentrations,
             yerr=2*standard_errors, fmt='o', color='black',
             ecolor='black', elinewidth=1.5, capsize=5, capthick=2,
             label="2 CAS Standard Error", zorder=3)
plt.errorbar(windspeed_values - 0.4, total_concentrations,
             yerr=2*counting_errors_CAS,
             fmt='s', markersize=6,
             markerfacecolor='#8c510a',
             markeredgecolor='black',
             ecolor='black',
             elinewidth=3, capsize=6, capthick=2,
             label="2σ CAS Error in Total Concentration", zorder=2)
plt.plot(x_fit_CAS, y_fit_CAS, '-', color='black', linewidth=2.5,
         label=f"CAS Fit: y = ({m_fit_CAS:.3f}±{m_err_CAS:.3f})x + {b_fit_CAS:.3f}, "
               f"R² = {r_squared_CAS:.2f}, R = {r_value_CAS:.2f}")
plt.errorbar(windspeed_values_CDP, total_concentrations_CDP,
             yerr=2*standard_errors_CDP, fmt='o', color='blue',
             ecolor='blue', elinewidth=1.5, capsize=5, capthick=2,
             label="2 CDP Standard Error", zorder=3)
plt.errorbar(windspeed_values_CDP + 0.4, total_concentrations_CDP,
             yerr=2*counting_errors_CDP,
             fmt='o', color='brown', markersize=8,
             ecolor='black', elinewidth=1.5, capsize=5, capthick=2,
             label="2σ CDP Error in Total Concentration", zorder=2)
plt.plot(x_fit_CDP, y_fit_CDP, '-', color='blue', linewidth=2.5,
         label=f"CDP Fit: y = ({m_fit_CDP:.3f}±{m_err_CDP:.3f})x + {b_fit_CDP:.3f}, "
               f"R² = {r_squared_CDP:.2f}, R = {pearson_corr_CDP:.2f}")
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=20, fontweight='bold')
plt.ylabel("Total Wind Speed Bin \nConcentration (cm$^{-3}$)", fontsize=20, fontweight='bold')
plt.title("Below Cloud Base\nJanuary–June 2022", fontsize=20, fontweight='bold')
ax.legend(
    [box_cas, whisker, box_cdp, point],
    ['±2 SE (CAS)', '±2σ Error in Total Concentration (both)', '±2 SE (CDP)', 'Mean Total Concentration'],
    fontsize=13,
    frameon=False,
    loc='center left',
    bbox_to_anchor=(1.02, 0.5)
)
plt.tight_layout()
plt.xlim(0, 12)
plt.ylim(0, 1.0)
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')
plt.show()
#%%
#trying box and whisker style 
def linear_model(x, m, b):
    return m * x + b
popt_CAS, pcov_CAS = curve_fit(linear_model, windspeed_values, total_concentrations)
m_fit_CAS, b_fit_CAS = popt_CAS
m_err_CAS, b_err_CAS = np.sqrt(np.diag(pcov_CAS))
residuals_CAS = total_concentrations - linear_model(windspeed_values, *popt_CAS)
ss_res_CAS = np.sum(residuals_CAS**2)
ss_tot_CAS = np.sum((total_concentrations - np.mean(total_concentrations))**2)
r_squared_CAS = 1 - (ss_res_CAS / ss_tot_CAS)
r_value_CAS = np.sign(m_fit_CAS) * np.sqrt(r_squared_CAS)
x_fit_CAS = np.linspace(min(windspeed_values), max(windspeed_values), 100)
y_fit_CAS = linear_model(x_fit_CAS, *popt_CAS)
popt_CDP, pcov_CDP = curve_fit(linear_model, windspeed_values_CDP, total_concentrations_CDP)
m_fit_CDP, b_fit_CDP = popt_CDP
m_err_CDP, b_err_CDP = np.sqrt(np.diag(pcov_CDP))
residuals_CDP = total_concentrations_CDP - linear_model(windspeed_values_CDP, *popt_CDP)
ss_res_CDP = np.sum(residuals_CDP**2)
ss_tot_CDP = np.sum((total_concentrations_CDP - np.mean(total_concentrations_CDP))**2)
r_squared_CDP = 1 - (ss_res_CDP / ss_tot_CDP)
pearson_corr_CDP, _ = pearsonr(windspeed_values_CDP, total_concentrations_CDP)
x_fit_CDP = np.linspace(min(windspeed_values_CDP), max(windspeed_values_CDP), 100)
y_fit_CDP = linear_model(x_fit_CDP, *popt_CDP)
fig, ax = plt.subplots(figsize=(9, 7))
dx = 0.3  # horizontal half-width of each box
for x, y, se, ce in zip(windspeed_values, total_concentrations, standard_errors, counting_errors_CAS):
    rect = Rectangle((x - dx, y - 2*se), width=2*dx, height=4*se,
                     facecolor='lightgray', edgecolor='black', alpha=0.7, zorder=2)
    ax.add_patch(rect)
    ax.plot(x, y, marker='o', color='black', markersize=4, zorder=3)
    ax.vlines(x, y - 2*ce, y + 2*ce, colors='black', linewidth=3, zorder=3)
for x, y, se, ce in zip(windspeed_values_CDP, total_concentrations_CDP, standard_errors_CDP, counting_errors_CDP):
    rect = Rectangle((x - dx, y - 2*se), width=2*dx, height=4*se,
                     facecolor='lightblue', edgecolor='blue', alpha=0.6, zorder=2)
    ax.add_patch(rect)
    ax.plot(x, y, marker='o', color='blue', markersize=4, zorder=3)
    ax.vlines(x, y - 2*ce, y + 2*ce, colors='blue', linewidth=3, zorder=3)
ax.plot(x_fit_CAS, y_fit_CAS, '-', color='black', linewidth=2.5,
        label=f"CAS Fit: y = ({m_fit_CAS:.3f}±{m_err_CAS:.3f})x + {b_fit_CAS:.3f}, R² = {r_squared_CAS:.2f}, R = {r_value_CAS:.2f}")
ax.plot(x_fit_CDP, y_fit_CDP, '-', color='blue', linewidth=2.5,
        label=f"CDP Fit: y = ({m_fit_CDP:.3f}±{m_err_CDP:.3f})x + {b_fit_CDP:.3f}, R² = {r_squared_CDP:.2f}, R = {pearson_corr_CDP:.2f}")
ax.set_xlabel("Wind Speed (m s$^{-1}$)", fontsize=15, fontweight='bold')
ax.set_ylabel("Total Wind Speed Bin \nConcentration (cm$^{-3}$)", fontsize=15, fontweight='bold')
ax.set_title("Binned GCCN Concentration\n as a function of Wind Speed", fontsize=15, fontweight='bold')
ax.set_xlim(0, 12)
ax.set_ylim(0, 1.0)
ax.tick_params(axis='both', labelsize=15)

for tick in ax.get_xticklabels():
    tick.set_fontweight('bold')

for tick in ax.get_yticklabels():
    tick.set_fontweight('bold')

box_cas = Rectangle((0,0), 1, 1, facecolor='gray', edgecolor='black')
box_cdp = Rectangle((0,0), 1, 1, facecolor='lightblue', edgecolor='blue')
whisker = plt.Line2D([0], [0], color='black', linewidth=3)
point = plt.Line2D([0], [0], marker='o', color='black', linestyle='None')

ax.legend(
    [box_cas, whisker, box_cdp, point],
    ['CAS ±2 SEM (~95% CI)', '±2σ Error in Total Concentration (both)', 'CDP ±2 SEM (~95% CI)', 'Mean Total Concentration'],
    fontsize=11,
    frameon=False,
    loc='center left',
    bbox_to_anchor=(1.02, 0.5)
)
plt.tight_layout()
plt.show()
#save figure as a pdf
fig.savefig("wind_speed_concentration_correlation_cdpcas.pdf", bbox_inches='tight')
#%%
#adding standard error in slope 

def linear_model(x, m, b):
    return m * x + b
avg_windspeeds_CAS = []
total_concentrations_CAS = []
standard_errors_CAS = []
for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]: 
        avg_windspeed_CAS = np.mean(mean_windspeeds[idx])
        avg_concentration_per_leg_CAS = [np.sum(dist * bin_widths) for dist in grouped_distributions[idx]]
        avg_concentration_CAS = np.mean(avg_concentration_per_leg_CAS)
        std_concentration_CAS = np.std(avg_concentration_per_leg_CAS, ddof=1)
        N_CAS = len(avg_concentration_per_leg_CAS)
        SE_concentration_CAS = std_concentration_CAS / np.sqrt(N_CAS)

        avg_windspeeds_CAS.append(avg_windspeed_CAS)
        total_concentrations_CAS.append(avg_concentration_CAS)
        standard_errors_CAS.append(SE_concentration_CAS)

windspeed_values_CAS = np.array(avg_windspeeds_CAS)
total_concentrations_CAS = np.array(total_concentrations_CAS)
standard_errors_CAS = np.array(standard_errors_CAS)
popt_CAS, pcov_CAS = curve_fit(linear_model, windspeed_values_CAS, total_concentrations_CAS)
m_fit_CAS, b_fit_CAS = popt_CAS
m_err_CAS, b_err_CAS = np.sqrt(np.diag(pcov_CAS))

residuals_CAS = total_concentrations_CAS - linear_model(windspeed_values_CAS, *popt_CAS)
ss_res_CAS = np.sum(residuals_CAS**2)
ss_tot_CAS = np.sum((total_concentrations_CAS - np.mean(total_concentrations_CAS))**2)
r_squared_CAS = 1 - (ss_res_CAS / ss_tot_CAS)
pearson_corr_CAS, _ = pearsonr(windspeed_values_CAS, total_concentrations_CAS)
avg_windspeeds_CDP = []
total_concentrations_CDP = []
standard_errors_CDP = []

for idx, (low, high) in enumerate(windspeed_bins):
    if final_grouped_CDP[idx]:
        avg_windspeed_CDP = np.mean(final_mean_windspeeds_CDP[idx])
        avg_concentration_per_leg_CDP = [np.sum(dist[:len(bin_widths_CDP)] * bin_widths_CDP) for dist in final_grouped_CDP[idx]]
        avg_concentration_CDP = np.mean(avg_concentration_per_leg_CDP)
        std_concentration_CDP = np.std(avg_concentration_per_leg_CDP, ddof=1)
        N_CDP = len(avg_concentration_per_leg_CDP)
        SE_concentration_CDP = std_concentration_CDP / np.sqrt(N_CDP)

        avg_windspeeds_CDP.append(avg_windspeed_CDP)
        total_concentrations_CDP.append(avg_concentration_CDP)
        standard_errors_CDP.append(SE_concentration_CDP)

windspeed_values_CDP = np.array(avg_windspeeds_CDP)
total_concentrations_CDP = np.array(total_concentrations_CDP)
standard_errors_CDP = np.array(standard_errors_CDP)
popt_CDP, pcov_CDP = curve_fit(linear_model, windspeed_values_CDP, total_concentrations_CDP)
m_fit_CDP, b_fit_CDP = popt_CDP
m_err_CDP, b_err_CDP = np.sqrt(np.diag(pcov_CDP))

residuals_CDP = total_concentrations_CDP - linear_model(windspeed_values_CDP, *popt_CDP)
ss_res_CDP = np.sum(residuals_CDP**2)
ss_tot_CDP = np.sum((total_concentrations_CDP - np.mean(total_concentrations_CDP))**2)
r_squared_CDP = 1 - (ss_res_CDP / ss_tot_CDP)
pearson_corr_CDP, _ = pearsonr(windspeed_values_CDP, total_concentrations_CDP)
plt.figure(figsize=(8, 6))
plt.errorbar(windspeed_values_CAS, total_concentrations_CAS, 
             yerr=standard_errors_CAS, fmt='o', color='black', 
             markersize=10, capsize=5, capthick=2, label="CAS Standard Error", 
             ecolor='black', elinewidth=1.5, zorder=3)

x_fit_CAS = np.linspace(min(windspeed_values_CAS), max(windspeed_values_CAS), 100)
y_fit_CAS = linear_model(x_fit_CAS, *popt_CAS)
plt.plot(x_fit_CAS, y_fit_CAS, 'black', linewidth=2, 
         label=f'CAS Fit: y = ({m_fit_CAS:.3f}±{m_err_CAS:.3f})x + {b_fit_CAS:.3f}\nR² = {r_squared_CAS:.2f}, R = {pearson_corr_CAS:.2f}')
plt.errorbar(windspeed_values_CDP, total_concentrations_CDP, 
             yerr=standard_errors_CDP, fmt='o', color='blue', 
             markersize=10, capsize=5, capthick=2, label="CDP Standard Error", 
             ecolor='blue', elinewidth=1.5, zorder=3)

x_fit_CDP = np.linspace(min(windspeed_values_CDP), max(windspeed_values_CDP), 100)
y_fit_CDP = linear_model(x_fit_CDP, *popt_CDP)
plt.plot(x_fit_CDP, y_fit_CDP, 'blue', linewidth=2, 
         label=f'CDP Fit: y = ({m_fit_CDP:.3f}±{m_err_CDP:.3f})x + {b_fit_CDP:.3f}\nR² = {r_squared_CDP:.2f}, R = {pearson_corr_CDP:.2f}')
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=19, fontweight='bold')
plt.ylabel("Total Wind Speed Bin \n Concentration (cm$^{-3}$)", fontsize=19, fontweight='bold')
plt.title("Below Cloud Base\n January-June 2022", fontsize=19, fontweight='bold')
plt.legend(fontsize=12, title_fontsize=14, loc='best', frameon=True)
plt.tight_layout()
plt.xlim(0, 12)
plt.ylim(0, 1)
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.show()
print(f"CAS Slope: {m_fit_CAS:.3f} ± {m_err_CAS:.3f}, Intercept: {b_fit_CAS:.3f}, R² = {r_squared_CAS:.2f}, R = {pearson_corr_CAS:.2f}")
print(f"CDP Slope: {m_fit_CDP:.3f} ± {m_err_CDP:.3f}, Intercept: {b_fit_CDP:.3f}, R² = {r_squared_CDP:.2f}, R = {pearson_corr_CDP:.2f}")
# %%
#computing mass against wind speed regression
grouped_mass_values = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds_mass = {i: [] for i in range(len(windspeed_bins))}
for mass_entry in filtered_dry_mass_inf_CDP:
    date = mass_entry['Date']
    BCB_start = mass_entry['BCB_start']
    BCB_stop = mass_entry['BCB_stop']
    
    
    windspeed_entry = df_combined_CDP[
        (df_combined_CDP['Date'] == date) & 
        (df_combined_CDP['BCB_start'] == BCB_start) & 
        (df_combined_CDP['BCB_stop'] == BCB_stop)
    ]

    if windspeed_entry.empty:
        continue 

    windspeed = windspeed_entry['Windspeed'].values[0]
    mass_value = mass_entry['Dry Mass (µg/m³)']

    
    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:
            grouped_mass_values[idx].append(mass_value)
            mean_windspeeds_mass[idx].append(windspeed)
            break

# %%
# Compute average wind speed and mass per bin
avg_windspeeds_mass = []
total_mass_values = []

for idx, mass_list in grouped_mass_values.items():
    if mass_list:
        avg_windspeed = np.mean(mean_windspeeds_mass[idx]) 
        avg_mass = np.mean(mass_list) 

        avg_windspeeds_mass.append(avg_windspeed)
        total_mass_values.append(avg_mass)

windspeed_values_mass = np.array(avg_windspeeds_mass)
total_mass_values = np.array(total_mass_values)
popt_mass, _ = curve_fit(linear_model, windspeed_values_mass, total_mass_values)
m_fit_mass, b_fit_mass = popt_mass
residuals_mass = total_mass_values - linear_model(windspeed_values_mass, *popt_mass)
ss_res_mass = np.sum(residuals_mass**2)
ss_tot_mass = np.sum((total_mass_values - np.mean(total_mass_values))**2)
r_squared_mass = 1 - (ss_res_mass / ss_tot_mass)

# %%
plt.figure(figsize=(8, 6))
for idx in range(len(windspeed_bins)):
    plt.scatter(windspeed_values_mass[idx], total_mass_values[idx], 
                color=colors[idx], s=100, edgecolor='black', zorder=3)
x_fit_mass = np.linspace(min(windspeed_values_mass), max(windspeed_values_mass), 100)
y_fit_mass = linear_model(x_fit_mass, *popt_mass)
plt.plot(x_fit_mass, y_fit_mass, 'r-', label=f'Fit: y = {m_fit_mass:.3f}x + {b_fit_mass:.3f}, R² = {r_squared_mass:.2f}')
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=16, fontweight='bold')
plt.ylabel("Total Dry Mass (µg/m³)", fontsize=16, fontweight='bold')
plt.title("CDP Below Cloud Base January - June 2022", fontsize=16, fontweight='bold')
legend_labels_mass = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[idx], markersize=10, 
               label=f"{windspeed_values_mass[idx]:.1f} m/s") for idx in range(len(windspeed_bins))
]
plt.legend(handles=legend_labels_mass + [plt.Line2D([0], [0], color='red', 
               label=f'Fit: y = {m_fit_mass:.3f}x + {b_fit_mass:.3f}, R² = {r_squared_mass:.2f}')], 
           title="Wind Speed Bins", title_fontsize=14, fontsize=13)
plt.tight_layout()
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.ylim(0, 55)
plt.show()
print(f"Slope (m): {m_fit_mass:.3f}")
print(f"Intercept (b): {b_fit_mass:.3f}")
print(f"R² value: {r_squared_mass:.2f}")
#%%
#mass against wind speed 
def linear_model(x, m, b):
    return m * x + b
grouped_mass_values_CDP = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds_mass_CDP = {i: [] for i in range(len(windspeed_bins))}
for mass_entry in filtered_dry_mass_inf_CDP:
    date = mass_entry['Date']
    BCB_start = mass_entry['BCB_start']
    BCB_stop = mass_entry['BCB_stop']
    
    windspeed_entry = df_combined_CDP[
        (df_combined_CDP['Date'] == date) & 
        (df_combined_CDP['BCB_start'] == BCB_start) & 
        (df_combined_CDP['BCB_stop'] == BCB_stop)
    ]

    if windspeed_entry.empty:
        continue  

    windspeed = windspeed_entry['Windspeed'].values[0]
    mass_value = mass_entry['Dry Mass (µg/m³)']  
    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:
            grouped_mass_values_CDP[idx].append(mass_value)
            mean_windspeeds_mass_CDP[idx].append(windspeed)
            break
avg_windspeeds_mass_CDP = []
total_mass_values_CDP = []

for idx, mass_list in grouped_mass_values_CDP.items():
    if mass_list:
        avg_windspeed = np.mean(mean_windspeeds_mass_CDP[idx]) 
        avg_mass = np.mean(mass_list) 

        avg_windspeeds_mass_CDP.append(avg_windspeed)
        total_mass_values_CDP.append(avg_mass)
windspeed_values_mass_CDP = np.array(avg_windspeeds_mass_CDP)
total_mass_values_CDP = np.array(total_mass_values_CDP)
popt_mass_CDP, _ = curve_fit(linear_model, windspeed_values_mass_CDP, total_mass_values_CDP)
m_fit_mass_CDP, b_fit_mass_CDP = popt_mass_CDP
residuals_mass_CDP = total_mass_values_CDP - linear_model(windspeed_values_mass_CDP, *popt_mass_CDP)
ss_res_mass_CDP = np.sum(residuals_mass_CDP**2)
ss_tot_mass_CDP = np.sum((total_mass_values_CDP - np.mean(total_mass_values_CDP))**2)
r_squared_mass_CDP = 1 - (ss_res_mass_CDP / ss_tot_mass_CDP)
plt.figure(figsize=(8, 6))
for idx in range(len(windspeed_bins)):
    plt.scatter(windspeed_values_mass_CDP[idx], total_mass_values_CDP[idx], 
                color="green", s=100, edgecolor='black', zorder=3)
x_fit_mass_CDP = np.linspace(min(windspeed_values_mass_CDP), max(windspeed_values_mass_CDP), 100)
y_fit_mass_CDP = linear_model(x_fit_mass_CDP, *popt_mass_CDP)
plt.plot(x_fit_mass_CDP, y_fit_mass_CDP, 'r-', label=f'Fit: y = {m_fit_mass_CDP:.3f}x + {b_fit_mass_CDP:.3f}, R² = {r_squared_mass_CDP:.2f}')
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=16, fontweight='bold')
plt.ylabel("Total Dry Mass (µg/m³)", fontsize=16, fontweight='bold')
plt.title("Below Cloud Base January - June 2022", fontsize=16, fontweight='bold')
legend_labels_mass_CDP = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[idx], markersize=10, 
               label=f"{windspeed_values_mass_CDP[idx]:.1f} m/s") for idx in range(len(windspeed_bins))
]
plt.legend(handles=legend_labels_mass_CDP + [plt.Line2D([0], [0], color='black', 
               label=f'Fit: y = {m_fit_mass_CDP:.3f}x + {b_fit_mass_CDP:.3f}, R² = {r_squared_mass_CDP:.2f}')], 
           title="Wind Speed Bins", title_fontsize=14, fontsize=13)
plt.tight_layout()
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.ylim(0, 55)
plt.xlim(0, 12)
plt.show()
print(f"Slope (m): {m_fit_mass_CDP:.3f}")
print(f"Intercept (b): {b_fit_mass_CDP:.3f}")
print(f"R² value: {r_squared_mass_CDP:.2f}")

#%%
#adding the R value and error bars 
def linear_model(x, m, b):
    return m * x + b
grouped_mass_values_CDP = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds_mass_CDP = {i: [] for i in range(len(windspeed_bins))}
for mass_entry in filtered_dry_mass_inf_CDP:
    date = mass_entry['Date']
    BCB_start = mass_entry['BCB_start']
    BCB_stop = mass_entry['BCB_stop']
    windspeed_entry = df_combined_CDP[
        (df_combined_CDP['Date'] == date) & 
        (df_combined_CDP['BCB_start'] == BCB_start) & 
        (df_combined_CDP['BCB_stop'] == BCB_stop)
    ]

    if windspeed_entry.empty:
        continue 

    windspeed = windspeed_entry['Windspeed'].values[0]
    mass_value = mass_entry['Dry Mass (µg/m³)'] 

    
    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:
            grouped_mass_values_CDP[idx].append(mass_value)
            mean_windspeeds_mass_CDP[idx].append(windspeed)
            break
avg_windspeeds_mass_CDP = []
total_mass_values_CDP = []
standard_errors_mass_CDP = []

for idx, mass_list in grouped_mass_values_CDP.items():
    if mass_list:
        avg_windspeed = np.mean(mean_windspeeds_mass_CDP[idx])
        avg_mass = np.mean(mass_list) 
        std_mass = np.std(mass_list, ddof=1)  
        N_mass = len(mass_list)  
        SE_mass = std_mass / np.sqrt(N_mass)  

        avg_windspeeds_mass_CDP.append(avg_windspeed)
        total_mass_values_CDP.append(avg_mass)
        standard_errors_mass_CDP.append(SE_mass)

windspeed_values_mass_CDP = np.array(avg_windspeeds_mass_CDP)
total_mass_values_CDP = np.array(total_mass_values_CDP)
standard_errors_mass_CDP = np.array(standard_errors_mass_CDP)
popt_mass_CDP, _ = curve_fit(linear_model, windspeed_values_mass_CDP, total_mass_values_CDP)
m_fit_mass_CDP, b_fit_mass_CDP = popt_mass_CDP
residuals_mass_CDP = total_mass_values_CDP - linear_model(windspeed_values_mass_CDP, *popt_mass_CDP)
ss_res_mass_CDP = np.sum(residuals_mass_CDP**2)
ss_tot_mass_CDP = np.sum((total_mass_values_CDP - np.mean(total_mass_values_CDP))**2)
r_squared_mass_CDP = 1 - (ss_res_mass_CDP / ss_tot_mass_CDP)
pearson_corr_mass_CDP, _ = pearsonr(windspeed_values_mass_CDP, total_mass_values_CDP)
plt.figure(figsize=(8, 6))
plt.errorbar(windspeed_values_mass_CDP, total_mass_values_CDP, 
             yerr=standard_errors_mass_CDP, fmt='o', color='green', 
             markersize=10, capsize=5, capthick=2, label="CDP", 
             ecolor='black', elinewidth=1.5, zorder=3)

x_fit_mass_CDP = np.linspace(min(windspeed_values_mass_CDP), max(windspeed_values_mass_CDP), 100)
y_fit_mass_CDP = linear_model(x_fit_mass_CDP, *popt_mass_CDP)
plt.plot(x_fit_mass_CDP, y_fit_mass_CDP, 'black', linewidth=2, 
         label=f'Fit: y = {m_fit_mass_CDP:.3f}x + {b_fit_mass_CDP:.3f}\nR² = {r_squared_mass_CDP:.2f}, R = {pearson_corr_mass_CDP:.2f}')
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=16, fontweight='bold')
plt.ylabel("Total Dry Mass (µg/m³)", fontsize=16, fontweight='bold')
plt.title("CDP Wind Speed vs. Total Dry Mass Correlation", fontsize=16, fontweight='bold')
plt.legend(fontsize=13, title_fontsize=14, loc='best', frameon=True)
plt.tight_layout()
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.ylim(0, 55)
plt.xlim(0, 12)
plt.show()
print(f"Slope (m): {m_fit_mass_CDP:.3f}")
print(f"Intercept (b): {b_fit_mass_CDP:.3f}")
print(f"R² value: {r_squared_mass_CDP:.2f}")
print(f"R value (Pearson correlation): {pearson_corr_mass_CDP:.3f}")
#%%
#mass counting error calculation
sample_area_cm2_CDP = 0.323 / 100   # cm² (converted from mm²)
plane_speed_cm_s = 1.2e4           # cm/s
sampling_time_s = 198              # s
T = sampling_time_s
V = sample_area_cm2_CDP * plane_speed_cm_s  # cm³/s
bin_centers_um_CDP = np.array([2.5, 3.5, 4.5, 5.5, 6.5, 7.5,
                               8.5, 9.5, 10.5, 11.5, 12.5, 13.5,
                               15., 17., 19., 21., 23., 25.,
                               27., 29., 31., 33., 35., 37.,
                               39., 41., 43., 45., 47., 49.])

edges_um_CDP = np.empty(len(bin_centers_um_CDP) + 1)
edges_um_CDP[0] = bin_centers_um_CDP[0] - (bin_centers_um_CDP[1] - bin_centers_um_CDP[0]) / 2
edges_um_CDP[1:-1] = 0.5 * (bin_centers_um_CDP[:-1] + bin_centers_um_CDP[1:])
edges_um_CDP[-1] = bin_centers_um_CDP[-1] + (bin_centers_um_CDP[-1] - bin_centers_um_CDP[-2]) / 2
bin_widths_um_CDP = np.diff(edges_um_CDP)

radii_cm_CDP = (bin_centers_um_CDP / 2.0) * 1e-4  # µm → cm
rho_salt_ug_cm3 = 2.2e6                   # µg/cm³
eta_cm = (4.0/3.0) * np.pi * rho_salt_ug_cm3
eta_m = eta_cm * 1e6                      # convert to µg/m³
grouped_mass_errors_CDP = {i: [] for i in range(len(windspeed_bins))}

for entry in dry_mass_data_inf_CDP:   
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    n0 = entry.get('Dry Intercept (N0)')
    D  = entry.get('Dry Slope (D)')

    ws_row = df_combined_CDP[
        (df_combined_CDP['Date'] == date) &
        (df_combined_CDP['BCB_start'] == BCB_start) &
        (df_combined_CDP['BCB_stop'] == BCB_stop)
    ]
    if ws_row.empty or np.isnan(n0) or np.isnan(D):
        continue

    windspeed = ws_row['Windspeed'].values[0]

    n_i = n0 * np.exp(-bin_centers_um_CDP / D)   # cm⁻³ / µm

    term_cm3 = (n_i * bin_widths_um_CDP) * (radii_cm_CDP ** 6)  # cm^3
    summation_cm3 = np.nansum(term_cm3)

    if summation_cm3 <= 0 or not np.isfinite(summation_cm3):
        mass_err_ug_m3 = 0.0
    else:
        factor = np.sqrt(summation_cm3 / (T * V))
        mass_err_ug_m3 = eta_m * factor

    for idx, (lo, hi) in enumerate(windspeed_bins):
        if lo <= windspeed < hi:
            grouped_mass_errors_CDP[idx].append(mass_err_ug_m3)
            break
median_mass_errors_per_bin_CDP = [np.nanmedian(errs) if errs else np.nan
                                  for errs in grouped_mass_errors_CDP.values()]

print("\n CDP Mass Counting Errors per Wind Speed Bin:")
for idx, (bounds, err) in enumerate(zip(windspeed_bins, median_mass_errors_per_bin_CDP)):
    label = f"{bounds[0]}–{bounds[1] if bounds[1] != np.inf else '∞'} m/s"
    if not np.isnan(err):
        print(f"  Bin {idx} ({label}): {err:.4f} µg/m³")
    else:
        print(f"  Bin {idx} ({label}): NaN (no data)")
counting_errors_mass_CDP = np.array(median_mass_errors_per_bin_CDP)

#%%
plt.figure(figsize=(8, 6))
plt.errorbar(windspeed_values_mass_CDP, total_mass_values_CDP, 
             yerr=standard_errors_mass_CDP, 
             fmt='o', color='blue', markersize=10,
             capsize=5, capthick=2, ecolor='black', elinewidth=1.5,
             label="CDP Standard Error", zorder=3)
plt.errorbar(windspeed_values_mass_CDP + 0.4, total_mass_values_CDP, 
             yerr=counting_errors_mass_CDP, 
             fmt='o', color='brown', markersize=10,
             capsize=5, capthick=2, ecolor='black', elinewidth=1.5,
             label="CDP Error in Total Mass", zorder=2)
x_fit_mass_CDP = np.linspace(min(windspeed_values_mass_CDP), max(windspeed_values_mass_CDP), 100)
y_fit_mass_CDP = linear_model(x_fit_mass_CDP, *popt_mass_CDP)

plt.plot(x_fit_mass_CDP, y_fit_mass_CDP, 'blue', linewidth=2,
         label=f'Fit: y = {m_fit_mass_CDP:.3f}x + {b_fit_mass_CDP:.3f}\n'
               f'R² = {r_squared_mass_CDP:.2f}, R = {pearson_corr_mass_CDP:.2f}')
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=18, fontweight='bold')
plt.ylabel("Total Dry Mass (µg/m³)", fontsize=18, fontweight='bold')
plt.title("CDP Below Cloud Base \nJanuary–June 2022", fontsize=18, fontweight='bold')
plt.legend(fontsize=13, loc='upper left', frameon=False)
plt.tight_layout()
plt.xlim(0, 12)
plt.ylim(0, 50)
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.show()
print(f"Slope (m): {m_fit_mass_CDP:.3f}")
print(f"Intercept (b): {b_fit_mass_CDP:.3f}")
print(f"R² value: {r_squared_mass_CDP:.2f}")
print(f"R value (Pearson correlation): {pearson_corr_mass_CDP:.3f}")
#%%
#overlaying CAS and CDP
windspeed_bins = [(0,2.5),(2.501,3.5),(3.501,5),(5.001,7),(7.001,9),(9.001,np.inf)]

def which_bin(ws):
    for i,(lo,hi) in enumerate(windspeed_bins):
        if lo <= ws < hi:
            return i
    return None

def count_err_for_points(ws_array, per_bin_err):
    out = []
    for ws in ws_array:
        b = which_bin(ws)
        out.append(per_bin_err[b] if b is not None else np.nan)
    return np.array(out, float)

def linear_model(x,m,b): return m*x + b
cas_x  = np.asarray(avg_ws,   float)
cas_y  = np.asarray(avg_mass, float)
cas_se = np.asarray(se_mass,  float)
cas_ce = count_err_for_points(cas_x, counting_errors_mass)

cdp_x  = np.asarray(windspeed_values_mass_CDP,     float)
cdp_y  = np.asarray(total_mass_values_CDP,         float)
cdp_se = np.asarray(standard_errors_mass_CDP,      float)
cdp_ce = count_err_for_points(cdp_x, counting_errors_mass_CDP)
mask_cas = np.isfinite(cas_x) & np.isfinite(cas_y)
mask_cdp = np.isfinite(cdp_x) & np.isfinite(cdp_y)

cas_x, cas_y, cas_se, cas_ce = cas_x[mask_cas], cas_y[mask_cas], cas_se[mask_cas], cas_ce[mask_cas]
cdp_x, cdp_y, cdp_se, cdp_ce = cdp_x[mask_cdp], cdp_y[mask_cdp], cdp_se[mask_cdp], cdp_ce[mask_cdp]
popt_cas, pcov_cas = curve_fit(linear_model, cas_x, cas_y)
m_cas, b_cas = popt_cas
merr_cas, berr_cas = np.sqrt(np.diag(pcov_cas))
res_cas = cas_y - linear_model(cas_x, *popt_cas)
r2_cas  = 1 - (np.sum(res_cas**2)/np.sum((cas_y - cas_y.mean())**2))
R_cas   = np.sign(m_cas)*np.sqrt(max(r2_cas,0))

popt_cdp, pcov_cdp = curve_fit(linear_model, cdp_x, cdp_y)
m_cdp, b_cdp = popt_cdp
merr_cdp, berr_cdp = np.sqrt(np.diag(pcov_cdp))
res_cdp = cdp_y - linear_model(cdp_x, *popt_cdp)
r2_cdp  = 1 - (np.sum(res_cdp**2)/np.sum((cdp_y - cdp_y.mean())**2))
R_cdp   = np.sign(m_cdp)*np.sqrt(max(r2_cdp,0))

xfit_cas = np.linspace(cas_x.min(), cas_x.max(), 100)
yfit_cas = linear_model(xfit_cas, *popt_cas)
xfit_cdp = np.linspace(cdp_x.min(), cdp_x.max(), 100)
yfit_cdp = linear_model(xfit_cdp, *popt_cdp)
plt.figure(figsize=(8,6))
dx = 0.4
plt.errorbar(cas_x, cas_y, yerr=2*cas_se, fmt='o', color='black',
             ecolor='black', elinewidth=1.5, capsize=5, capthick=2,
             label="±2 Standard Errors (CAS)", zorder=4)
plt.errorbar(cas_x+dx,       cas_y, yerr=2*cas_ce, fmt='s', markersize=6,
             markerfacecolor='#8c510a', markeredgecolor='black',
             ecolor='black', elinewidth=1.5, capsize=5, capthick=2,
             label="±2σ Error in Total Mass (CAS)", zorder=3)
plt.plot(xfit_cas, yfit_cas, '-', color='black', linewidth=3,
         label=f"CAS Fit: y = ({m_cas:.2f}±{merr_cas:.2f})x + {b_cas:.2f}, "
               f"R² = {r2_cas:.2f}, R = {R_cas:.2f}")
plt.errorbar(cdp_x, cdp_y, yerr=2*cdp_se, fmt='o', color='blue',
             ecolor='blue', elinewidth=1.5, capsize=5, capthick=2,
             label="±2 Standard Errors (CDP)", zorder=4)
plt.errorbar(cdp_x+dx,       cdp_y, yerr=2*cdp_ce, fmt='s', markersize=6,
             markerfacecolor='brown', markeredgecolor='black',
             ecolor='black', elinewidth=1.5, capsize=5, capthick=2,
             label="±2σ Error in Total Mass (CDP)", zorder=3)
plt.plot(xfit_cdp, yfit_cdp, '-', color='blue', linewidth=3,
         label=f"CDP Fit: y = ({m_cdp:.2f}±{merr_cdp:.2f})x + {b_cdp:.2f}, "
               f"R² = {r2_cdp:.2f}, R = {R_cdp:.2f}")
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=20, fontweight='bold')
plt.ylabel("Total Dry Mass (µg/m³)", fontsize=20, fontweight='bold')
plt.title("Below Cloud Base \nJanuary-June 2022", fontsize=22, fontweight='bold')
plt.legend(fontsize=13, frameon=False, loc='upper left')
plt.xlim(0, 12)
plt.ylim(0, 65)
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n=== CAS Mass Fit ===")
print(f"Slope (m): {m_cas:.3f} ± {merr_cas:.3f}")
print(f"Intercept (b): {b_cas:.3f}")
print(f"R² value: {r2_cas:.2f}, R = {R_cas:.2f}")
print("\n=== CDP Mass Fit ===")
print(f"Slope (m): {m_cdp:.3f} ± {merr_cdp:.3f}")
print(f"Intercept (b): {b_cdp:.3f}")
print(f"R² value: {r2_cdp:.2f}, R = {R_cdp:.2f}")
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -------------------------
# BINS (same as you used)
# -------------------------
windspeed_bins = [(0,2.5),(2.501,3.5),(3.501,5),(5.001,7),(7.001,9),(9.001,np.inf)]

def which_bin(ws):
    for i,(lo,hi) in enumerate(windspeed_bins):
        if lo <= ws < hi:
            return i
    return None

def count_err_for_points(ws_array, per_bin_err):
    out = []
    for ws in ws_array:
        b = which_bin(ws)
        out.append(per_bin_err[b] if b is not None else np.nan)
    return np.asarray(out, dtype=float)

def linear_model(x, m, b):
    return m * x + b

def as1d(a):
    """Force any input into a 1-D float array (prevents 0-D indexing crashes)."""
    return np.atleast_1d(np.asarray(a, dtype=float))


# ============================================================
# INPUTS EXPECTED TO EXIST (from your prior CAS/CDP steps):
#   CAS: avg_ws, avg_mass, se_mass, counting_errors_mass
#   CDP: windspeed_values_mass_CDP, total_mass_values_CDP,
#        standard_errors_mass_CDP, counting_errors_mass_CDP
# ============================================================

# -------------------------
# Build 1-D arrays (robust)
# -------------------------
cas_x  = as1d(avg_ws)
cas_y  = as1d(avg_mass)
cas_se = as1d(se_mass)

cdp_x  = as1d(windspeed_values_mass_CDP)
cdp_y  = as1d(total_mass_values_CDP)
cdp_se = as1d(standard_errors_mass_CDP)

# Per-point counting errors from per-bin medians
cas_ce = count_err_for_points(cas_x, counting_errors_mass)
cdp_ce = count_err_for_points(cdp_x, counting_errors_mass_CDP)

# Make sure each instrument’s arrays are same length (crop to shortest)
ncas = min(len(cas_x), len(cas_y), len(cas_se), len(cas_ce))
cas_x, cas_y, cas_se, cas_ce = cas_x[:ncas], cas_y[:ncas], cas_se[:ncas], cas_ce[:ncas]

ncdp = min(len(cdp_x), len(cdp_y), len(cdp_se), len(cdp_ce))
cdp_x, cdp_y, cdp_se, cdp_ce = cdp_x[:ncdp], cdp_y[:ncdp], cdp_se[:ncdp], cdp_ce[:ncdp]

# Mask invalid
mask_cas = np.isfinite(cas_x) & np.isfinite(cas_y) & np.isfinite(cas_se)
mask_cdp = np.isfinite(cdp_x) & np.isfinite(cdp_y) & np.isfinite(cdp_se)

cas_x, cas_y, cas_se, cas_ce = cas_x[mask_cas], cas_y[mask_cas], cas_se[mask_cas], cas_ce[mask_cas]
cdp_x, cdp_y, cdp_se, cdp_ce = cdp_x[mask_cdp], cdp_y[mask_cdp], cdp_se[mask_cdp], cdp_ce[mask_cdp]

# Safety checks
if len(cas_x) < 2:
    raise ValueError(f"CAS fit needs >=2 points; got {len(cas_x)}. Check avg_ws/avg_mass/se_mass.")
if len(cdp_x) < 2:
    raise ValueError(f"CDP fit needs >=2 points; got {len(cdp_x)}. Check CDP arrays.")

# -------------------------
# Fits (CAS)
# -------------------------
popt_cas, pcov_cas = curve_fit(linear_model, cas_x, cas_y)
m_cas, b_cas = popt_cas
merr_cas, berr_cas = np.sqrt(np.diag(pcov_cas))
res_cas = cas_y - linear_model(cas_x, *popt_cas)
r2_cas  = 1 - (np.sum(res_cas**2) / np.sum((cas_y - cas_y.mean())**2))
R_cas   = np.sign(m_cas) * np.sqrt(max(r2_cas, 0))

# -------------------------
# Fits (CDP)
# -------------------------
popt_cdp, pcov_cdp = curve_fit(linear_model, cdp_x, cdp_y)
m_cdp, b_cdp = popt_cdp
merr_cdp, berr_cdp = np.sqrt(np.diag(pcov_cdp))
res_cdp = cdp_y - linear_model(cdp_x, *popt_cdp)
r2_cdp  = 1 - (np.sum(res_cdp**2) / np.sum((cdp_y - cdp_y.mean())**2))
R_cdp   = np.sign(m_cdp) * np.sqrt(max(r2_cdp, 0))

# Fit curves
xfit_cas = np.linspace(cas_x.min(), cas_x.max(), 200)
yfit_cas = linear_model(xfit_cas, *popt_cas)
xfit_cdp = np.linspace(cdp_x.min(), cdp_x.max(), 200)
yfit_cdp = linear_model(xfit_cdp, *popt_cdp)

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(8, 6))
dx = 0.4

# CAS
plt.errorbar(cas_x, cas_y, yerr=2*cas_se, fmt='o', color='black',
             ecolor='black', elinewidth=1.5, capsize=5, capthick=2,
             label="±2 SE (CAS)", zorder=4)

plt.errorbar(cas_x+dx, cas_y, yerr=2*cas_ce, fmt='s', markersize=6,
             markerfacecolor='#8c510a', markeredgecolor='black',
             ecolor='black', elinewidth=1.5, capsize=5, capthick=2,
             label="±2σ Error in Total Mass (CAS)", zorder=3)

plt.plot(xfit_cas, yfit_cas, '-', color='black', linewidth=3,
         label=f"CAS Fit: y = ({m_cas:.2f}±{merr_cas:.2f})x + {b_cas:.2f}, "
               f"R² = {r2_cas:.2f}, R = {R_cas:.2f}")

# CDP
plt.errorbar(cdp_x, cdp_y, yerr=2*cdp_se, fmt='o', color='blue',
             ecolor='blue', elinewidth=1.5, capsize=5, capthick=2,
             label="±2 SE (CDP)", zorder=4)

plt.errorbar(cdp_x+dx, cdp_y, yerr=2*cdp_ce, fmt='s', markersize=6,
             markerfacecolor='brown', markeredgecolor='black',
             ecolor='black', elinewidth=1.5, capsize=5, capthick=2,
             label="±2σ Error in Total Mass (CDP)", zorder=3)

plt.plot(xfit_cdp, yfit_cdp, '-', color='blue', linewidth=3,
         label=f"CDP Fit: y = ({m_cdp:.2f}±{merr_cdp:.2f})x + {b_cdp:.2f}, "
               f"R² = {r2_cdp:.2f}, R = {R_cdp:.2f}")

# Labels/format
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=20, fontweight='bold')
plt.ylabel("Total Dry Mass (µg/m³)", fontsize=20, fontweight='bold')
plt.title("Below Cloud Base \nJanuary–June 2022", fontsize=22, fontweight='bold')

plt.legend(fontsize=13, frameon=False, loc='upper left')
plt.xlim(0, 12)
plt.ylim(0, 65)

plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n=== CAS Mass Fit ===")
print(f"Slope (m): {m_cas:.3f} ± {merr_cas:.3f}")
print(f"Intercept (b): {b_cas:.3f} ± {berr_cas:.3f}")
print(f"R² value: {r2_cas:.2f}, R = {R_cas:.2f}")

print("\n=== CDP Mass Fit ===")
print(f"Slope (m): {m_cdp:.3f} ± {merr_cdp:.3f}")
print(f"Intercept (b): {b_cdp:.3f} ± {berr_cdp:.3f}")
print(f"R² value: {r2_cdp:.2f}, R = {R_cdp:.2f}")

# %%
#trying box and whisker style
from matplotlib.patches import Rectangle

windspeed_bins = [(0,2.5),(2.501,3.5),(3.501,5),(5.001,7),(7.001,9),(9.001,np.inf)]

def which_bin(ws):
    for i,(lo,hi) in enumerate(windspeed_bins):
        if lo <= ws < hi:
            return i
    return None

def count_err_for_points(ws_array, per_bin_err):
    out = []
    for ws in ws_array:
        b = which_bin(ws)
        out.append(per_bin_err[b] if b is not None else np.nan)
    return np.array(out, float)

def linear_model(x, m, b): return m * x + b
cas_x  = np.asarray(avg_ws,   float)
cas_y  = np.asarray(avg_mass, float)
cas_se = np.asarray(se_mass,  float)
cas_ce = count_err_for_points(cas_x, counting_errors_mass)

cdp_x  = np.asarray(windspeed_values_mass_CDP,     float)
cdp_y  = np.asarray(total_mass_values_CDP,         float)
cdp_se = np.asarray(standard_errors_mass_CDP,      float)
cdp_ce = count_err_for_points(cdp_x, counting_errors_mass_CDP)

mask_cas = np.isfinite(cas_x) & np.isfinite(cas_y)
mask_cdp = np.isfinite(cdp_x) & np.isfinite(cdp_y)

cas_x, cas_y, cas_se, cas_ce = cas_x[mask_cas], cas_y[mask_cas], cas_se[mask_cas], cas_ce[mask_cas]
cdp_x, cdp_y, cdp_se, cdp_ce = cdp_x[mask_cdp], cdp_y[mask_cdp], cdp_se[mask_cdp], cdp_ce[mask_cdp]
popt_cas, pcov_cas = curve_fit(linear_model, cas_x, cas_y)
m_cas, b_cas = popt_cas
merr_cas, berr_cas = np.sqrt(np.diag(pcov_cas))
res_cas = cas_y - linear_model(cas_x, *popt_cas)
r2_cas  = 1 - (np.sum(res_cas**2)/np.sum((cas_y - cas_y.mean())**2))
R_cas   = np.sign(m_cas)*np.sqrt(max(r2_cas, 0))

popt_cdp, pcov_cdp = curve_fit(linear_model, cdp_x, cdp_y)
m_cdp, b_cdp = popt_cdp
merr_cdp, berr_cdp = np.sqrt(np.diag(pcov_cdp))
res_cdp = cdp_y - linear_model(cdp_x, *popt_cdp)
r2_cdp  = 1 - (np.sum(res_cdp**2)/np.sum((cdp_y - cdp_y.mean())**2))
R_cdp   = np.sign(m_cdp)*np.sqrt(max(r2_cdp, 0))
fig, ax = plt.subplots(figsize=(8, 6))
dx = 0.35
for x, y, se, ce in zip(cas_x, cas_y, cas_se, cas_ce):
    box_bottom = y - 2 * se
    box_height = 4 * se
    rect = Rectangle((x - dx, box_bottom), width=2*dx, height=box_height,
                     facecolor='gray', edgecolor='black', alpha=0.7, zorder=2)
    ax.add_patch(rect)
    ax.plot(x, y, marker='o', color='black', markersize=4, zorder=3)
    ax.vlines(x, y - 2*ce, y + 2*ce, colors='black', linestyles='-', linewidth=2, zorder=3)
for x, y, se, ce in zip(cdp_x, cdp_y, cdp_se, cdp_ce):
    box_bottom = y - 2 * se
    box_height = 4 * se
    rect = Rectangle((x - dx, box_bottom), width=2*dx, height=box_height,
                     facecolor='lightblue', edgecolor='blue', alpha=0.6, zorder=2)
    ax.add_patch(rect)
    ax.plot(x, y, marker='o', color='blue', markersize=4, zorder=3)
    ax.vlines(x, y - 2*ce, y + 2*ce, colors='blue', linestyles='-', linewidth=2, zorder=3)
xfit_cas = np.linspace(min(cas_x), max(cas_x), 100)
yfit_cas = linear_model(xfit_cas, *popt_cas)
xfit_cdp = np.linspace(min(cdp_x), max(cdp_x), 100)
yfit_cdp = linear_model(xfit_cdp, *popt_cdp)
ax.plot(xfit_cas, yfit_cas, '-', color='black', linewidth=2.5,
        label=f"CAS Fit: y = ({m_cas:.2f}±{merr_cas:.2f})x + {b_cas:.2f}, R² = {r2_cas:.2f}, R = {R_cas:.2f}")
ax.plot(xfit_cdp, yfit_cdp, '-', color='blue', linewidth=2.5,
        label=f"CDP Fit: y = ({m_cdp:.2f}±{merr_cdp:.2f})x + {b_cdp:.2f}, R² = {r2_cdp:.2f}, R = {R_cdp:.2f}")
ax.set_xlabel("Wind Speed (m s$^{-1}$)", fontsize=15, fontweight='bold')
ax.set_ylabel("Total Dry Mass (µg m$^{-3}$)", fontsize=15, fontweight='bold')
ax.set_title("Binned GCCN Mass\n as a function of Wind Speed", fontsize=15, fontweight='bold')
ax.set_xlim(0, 12)
ax.set_ylim(0, 20)
ax.tick_params(axis='both', labelsize=15)

for tick in ax.get_xticklabels():
    tick.set_fontweight('bold')

for tick in ax.get_yticklabels():
    tick.set_fontweight('bold')

box_cas = Rectangle((0,0), 1, 1, facecolor='gray', edgecolor='black')
box_cdp = Rectangle((0,0), 1, 1, facecolor='blue', edgecolor='blue')
whisker = plt.Line2D([0], [0], color='gray', linewidth=3)
point = plt.Line2D([0], [0], marker='o', color='black', linestyle='None')

ax.legend(
    [box_cas, whisker, box_cdp, point],
    ['CAS ±2 SEM (~95% CI)', '±2 SEM Error in total Mass (both)', 'CDP ±2 SEM (~95% CI)', 'Mean Total Mass'],
   fontsize=11,
    frameon=False,
    loc='center left',
    bbox_to_anchor=(1.02, 0.5)
)

plt.tight_layout()
plt.show()
print("\n=== CAS Mass Fit ===")
print(f"Slope (m): {m_cas:.3f} ± {merr_cas:.3f}")
print(f"Intercept (b): {b_cas:.3f}")
print(f"R² value: {r2_cas:.2f}, R = {R_cas:.2f}")
print("\n=== CDP Mass Fit ===")
print(f"Slope (m): {m_cdp:.3f} ± {merr_cdp:.3f}")
print(f"Intercept (b): {b_cdp:.3f}")
print(f"R² value: {r2_cdp:.2f}, R = {R_cdp:.2f}")

# %%
#2 panel cas and cdp mass and concentration 
fig, (axR, axL) = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
dx = 0.3
for x, y, se, ce in zip(windspeed_values, total_concentrations, standard_errors, counting_errors_CAS):
    rect = Rectangle((x - dx, y - 2*se), 2*dx, 4*se,
                     facecolor='lightgray', edgecolor='black', alpha=0.7)
    axL.add_patch(rect)
    axL.plot(x, y, 'o', color='black', markersize=4)
    axL.vlines(x, y-2*ce, y+2*ce, colors='black', linewidth=3)

for x, y, se, ce in zip(windspeed_values_CDP, total_concentrations_CDP,
                        standard_errors_CDP, counting_errors_CDP):
    rect = Rectangle((x - dx, y - 2*se), 2*dx, 4*se,
                     facecolor='lightblue', edgecolor='blue', alpha=0.6)
    axL.add_patch(rect)
    axL.plot(x, y, 'o', color='blue', markersize=4)
    axL.vlines(x, y-2*ce, y+2*ce, colors='blue', linewidth=3)

axL.plot(x_fit_CAS, y_fit_CAS, '-', color='black', lw=2.5)
axL.plot(x_fit_CDP, y_fit_CDP, '-', color='blue', lw=2.5)

axL.set_title("GCCN Concentration as a function of wind speed",
              fontsize=15, fontweight='bold')
axL.set_xlabel("Wind Speed (m s$^{-1}$)", fontsize=15, fontweight='bold')
axL.set_ylabel("Wind Speed Binned\nConcentration (cm$^{-3}$)",
               fontsize=15, fontweight='bold')
axL.set_xlim(0, 12)
axL.set_ylim(0, 1.0)
for x, y, se, ce in zip(cas_x, cas_y, cas_se, cas_ce):
    rect = Rectangle((x - dx, y - 2*se), 2*dx, 4*se,
                     facecolor='gray', edgecolor='black', alpha=0.7)
    axR.add_patch(rect)
    axR.plot(x, y, 'o', color='black', markersize=4)
    axR.vlines(x, y-2*ce, y+2*ce, colors='black', linewidth=2)

for x, y, se, ce in zip(cdp_x, cdp_y, cdp_se, cdp_ce):
    rect = Rectangle((x - dx, y - 2*se), 2*dx, 4*se,
                     facecolor='lightblue', edgecolor='blue', alpha=0.6)
    axR.add_patch(rect)
    axR.plot(x, y, 'o', color='blue', markersize=4)
    axR.vlines(x, y-2*ce, y+2*ce, colors='blue', linewidth=2)

axR.plot(xfit_cas, yfit_cas, '-', color='black', lw=2.5)
axR.plot(xfit_cdp, yfit_cdp, '-', color='blue', lw=2.5)

axR.set_title("GCCN Mass as a function of wind speed",
              fontsize=15, fontweight='bold')
axR.set_xlabel("Wind Speed (m s$^{-1}$)", fontsize=15, fontweight='bold')
axR.set_ylabel("Wind Speed Binned \nMass (µg m$^{-3}$)",
               fontsize=15, fontweight='bold')
axR.set_xlim(0, 12)
axR.set_ylim(0, 20)
for ax in (axL, axR):
    ax.tick_params(axis='both', labelsize=14)
    for t in ax.get_xticklabels()+ax.get_yticklabels():
        t.set_fontweight('bold')

fig.subplots_adjust(wspace=0.25, right=0.85)
from matplotlib.lines import Line2D

box_cas = Rectangle((0,0), 1, 1, facecolor='gray', edgecolor='black')
box_cdp = Rectangle((0,0), 1, 1, facecolor='lightblue', edgecolor='blue')
whisker = Line2D([0], [0], color='black', linewidth=3)
point = Line2D([0], [0], marker='o', color='black', linestyle='None')

fig.legend(
    [box_cas, whisker, box_cdp, point],
    ['CAS ±2 SEM (~95% CI)',
     '±2 SEM Error in total (both)',
     'CDP ±2 SEM (~95% CI)',
     'Mean'],
    fontsize=11,
    frameon=False,
    loc='center left',
    bbox_to_anchor=(1.02, 0.5)
)
fig.subplots_adjust(right=0.75)
plt.tight_layout()
plt.show()
#save figure as a pdf
fig.savefig("Mass_Concentration_WindSpeed_paper.pdf", format='pdf', bbox_inches='tight')
# %%
