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
from matplotlib.colors import BoundaryNorm
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
import matplotlib.patheffects as path_effects
from scipy.interpolate import interp1d
import matplotlib.colors as mcolors
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

#Import the flight leg time stamps and leg lengths 

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
        'LegIndex_06': {'StartTimes': [], 'StopTimes': []}, 
        'LegIndex_03': {'StartTimes': [], 'StopTimes': []},
        'LegIndex_04': {'StartTimes': [], 'StopTimes': []},
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
            if df_legs[col].dtype == 'O': 
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
        leg_index_03 = df_legs[df_legs['  LegIndex'] % 100 == 3]    
        leg_index_04 = df_legs[df_legs['  LegIndex'] % 100 == 4]
        leg_dictionary['LegIndex_02']['StartTimes'].extend(leg_index_02['Time_Start'].tolist())
        leg_dictionary['LegIndex_02']['StopTimes'].extend(leg_index_02['  Time_Stop'].tolist())
        leg_dictionary['LegIndex_06']['StartTimes'].extend(leg_index_06['Time_Start'].tolist())
        leg_dictionary['LegIndex_06']['StopTimes'].extend(leg_index_06['  Time_Stop'].tolist())
        leg_dictionary['LegIndex_03']['StartTimes'].extend(leg_index_03['Time_Start'].tolist())
        leg_dictionary['LegIndex_03']['StopTimes'].extend(leg_index_03['  Time_Stop'].tolist())
        leg_dictionary['LegIndex_04']['StartTimes'].extend(leg_index_04['Time_Start'].tolist())
        leg_dictionary['LegIndex_04']['StopTimes'].extend(leg_index_04['  Time_Stop'].tolist())

    leg_data.append(leg_dictionary)
#%%
#in-cloud droplet concentrations, we need to adjust lwc threshold. 
#continue using the CDP for lwc and pull CDP concentrations but filter 2DS N also to get drizzle drops out 

# New list for in-cloud droplet concentrations
in_cloud_concentrations = []

for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict = leg_data[i]

    ACB_start = leg_dict['LegIndex_03']['StartTimes']
    ACB_stop = leg_dict['LegIndex_03']['StopTimes']
    BCT_start =leg_dict['LegIndex_04']['StartTimes']
    BCT_stop = leg_dict['LegIndex_04']['StopTimes']
    CDP_flight = CDP_1Hz[i]
    twoDS_flight = twoDS[i]

    CDP_flight['Time_Start'] = pd.to_numeric(CDP_flight['Time_Start'], errors='coerce')
    twoDS_flight['Time_Start'] = pd.to_numeric(twoDS_flight['Time_Start'], errors='coerce')

    CDP_times = CDP_flight['Time_Start'].values
    CDP_lwc = CDP_flight['LWC_CDP'].values
    CDP_bins = {f'CDP_Bin{bin_label:02d}': CDP_flight[f'CDP_Bin{bin_label:02d}'].values for bin_label in range(0, 30)}
    TwoDS_times = twoDS_flight['Time_Start'].values

    for k in range(len(ACB_start)):
        start_time = ACB_start[k]
        end_time = ACB_stop[k]

        CDP_indices_in_range = np.where((CDP_times >= start_time) & (CDP_times <= end_time))[0]

        for CDP_idx in zip(CDP_indices_in_range):
            lwc_val = CDP_lwc[CDP_idx]

            if lwc_val >= 0.01:
                calc_entry = {
                    'Date': date,
                    'Time': CDP_times[CDP_idx],
                    'BCB_start': start_time,
                    'BCB_stop': end_time,
                    'CWC': lwc_val,
                }

                for bin_label in range(0, 30):
                    bin_key = f'Bin{bin_label}_concentration'
                    calc_entry[bin_key] = CDP_bins[f'CDP_Bin{bin_label:02d}'][CDP_idx]

                in_cloud_concentrations.append(calc_entry)


#%%
#adding BCT and ACB legs together in a combined dictionary 
in_cloud_concentrations = []

for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict = leg_data[i]

    ACB_start = leg_dict['LegIndex_03']['StartTimes']
    ACB_stop = leg_dict['LegIndex_03']['StopTimes']
    BCT_start = leg_dict['LegIndex_04']['StartTimes']
    BCT_stop = leg_dict['LegIndex_04']['StopTimes']

    CDP_flight = CDP_1Hz[i]
    twoDS_flight = twoDS[i]

    CDP_flight['Time_Start'] = pd.to_numeric(CDP_flight['Time_Start'], errors='coerce')

    CDP_times = CDP_flight['Time_Start'].values
    CDP_lwc = CDP_flight['LWC_CDP'].values
    CDP_bins = {f'CDP_Bin{bin_label:02d}': CDP_flight[f'CDP_Bin{bin_label:02d}'].values for bin_label in range(0, 30)}

    combined_legs = [
        (ACB_start, ACB_stop),
        (BCT_start, BCT_stop)
    ]

    for leg_start, leg_stop in combined_legs:
        for k in range(len(leg_start)):
            start_time = leg_start[k]
            end_time = leg_stop[k]

            CDP_indices_in_range = np.where((CDP_times >= start_time) & (CDP_times <= end_time))[0]

            for CDP_idx in CDP_indices_in_range:
                lwc_val = CDP_lwc[CDP_idx]

                if lwc_val >= 0.01:  # Adjust LWC threshold as needed
                    calc_entry = {
                        'Date': date,
                        'Time': CDP_times[CDP_idx],
                        'Leg_start': start_time,
                        'Leg_stop': end_time,
                        'CWC': lwc_val  # Cloud water content
                    }

                    for bin_label in range(0, 30):
                        bin_key = f'Bin{bin_label}_concentration'
                        calc_entry[bin_key] = CDP_bins[f'CDP_Bin{bin_label:02d}'][CDP_idx]

                    in_cloud_concentrations.append(calc_entry)

print(f"Number of in-cloud entries: {len(in_cloud_concentrations)}")
print(f"First 5 entries: {in_cloud_concentrations[:5]}")


#%%
# This code calculates total concentration in cm³
in_cloud_concentrations = []

for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict = leg_data[i]

    ACB_start = leg_dict['LegIndex_03']['StartTimes']
    ACB_stop = leg_dict['LegIndex_03']['StopTimes']
    BCT_start = leg_dict['LegIndex_04']['StartTimes']
    BCT_stop = leg_dict['LegIndex_04']['StopTimes']

    CDP_flight = CDP_1Hz[i]

    CDP_flight['Time_Start'] = pd.to_numeric(CDP_flight['Time_Start'], errors='coerce')

    CDP_times = CDP_flight['Time_Start'].values
    CDP_lwc = CDP_flight['LWC_CDP'].values
    CDP_bins = {f'CDP_Bin{bin_label:02d}': CDP_flight[f'CDP_Bin{bin_label:02d}'].values for bin_label in range(0, 30)}

    bin_widths = [bin_log_CDP[bin_label] for bin_label in range(0, 30)]

    all_legs_start = ACB_start + BCT_start
    all_legs_stop = ACB_stop + BCT_stop

    for k in range(len(all_legs_start)):
        start_time = all_legs_start[k]
        end_time = all_legs_stop[k]

        CDP_indices_in_range = np.where((CDP_times >= start_time) & (CDP_times <= end_time))[0]

        for CDP_idx in CDP_indices_in_range:
            lwc_val = CDP_lwc[CDP_idx]

          
            if lwc_val >= 0.01:
                total_concentration = sum(
                    np.nan_to_num(CDP_bins[f'CDP_Bin{bin_label:02d}'][CDP_idx]) * bin_width
                    for bin_label, bin_width in zip(range(0, 30), bin_widths)
                )

                
                calc_entry = {
                    'Date': date,
                    'Time': CDP_times[CDP_idx],
                    'Leg_start': start_time,
                    'Leg_stop': end_time,
                    'CWC': lwc_val,
                    'Total_Concentration': total_concentration  # Units: cm³
                }

                
                in_cloud_concentrations.append(calc_entry)

print(f"Number of in-cloud entries: {len(in_cloud_concentrations)}")
print(f"Sample entries: {in_cloud_concentrations[:5]}")
#%%
Bin_Lower = [62.70, 74.10, 85.50, 96.90, 
             108.30, 119.70, 131.10, 142.50, 153.90, 165.30, 
             176.70, 188.10, 199.50, 210.90, 222.30, 233.70, 
             245.10, 256.50, 267.90, 279.30, 290.70, 302.10, 
             313.50, 324.90, 336.30, 347.70, 359.10, 370.50, 
             381.90, 393.30, 404.70, 416.10, 427.50, 438.90, 
             450.30, 461.70, 473.10, 484.50, 495.90, 507.30, 
             518.70, 530.10, 541.50, 552.90, 564.30, 575.70, 
             587.10, 598.50, 609.90, 621.30, 632.70, 644.10, 
             655.50, 666.90, 678.30, 689.70, 701.10, 712.50, 
             723.90, 735.30, 746.70, 758.10, 769.50, 780.90, 
             792.30, 803.70, 815.10, 826.50, 837.90, 849.30, 
             860.70, 872.10, 883.50, 894.90, 906.30, 917.70, 
             929.10, 940.50, 951.90, 963.30, 974.70, 986.10, 
             997.50, 1008.90, 1020.30, 1031.70, 1043.10, 
             1054.50, 1065.90, 1077.30, 1088.70, 1100.10, 
             1111.50, 1122.90, 1134.30, 1145.70, 1157.10, 
             1168.50, 1179.90, 1191.30, 1202.70, 1214.10, 
             1225.50, 1236.90, 1248.30, 1259.70, 1271.10, 
             1282.50, 1293.90, 1305.30, 1316.70, 1328.10, 
             1339.50, 1350.90, 1362.30, 1373.70, 1385.10, 
             1396.50, 1407.90, 1419.30, 1430.70, 1442.10, 1453.50]
Bin_Upper = [74.10, 85.50, 96.90, 108.30, 
             119.70, 131.10, 142.50, 153.90, 165.30, 176.70, 188.10, 
             199.50, 210.90, 222.30, 233.70, 245.10, 256.50, 267.90, 
             279.30, 290.70, 302.10, 313.50, 324.90, 336.30, 347.70, 
             359.10, 370.50, 381.90, 393.30, 404.70, 416.10, 427.50, 
             438.90, 450.30, 461.70, 473.10, 484.50, 495.90, 507.30, 
             518.70, 530.10, 541.50, 552.90, 564.30, 575.70, 587.10, 
             598.50, 609.90, 621.30, 632.70, 644.10, 655.50, 666.90, 
             678.30, 689.70, 701.10, 712.50, 723.90, 735.30, 746.70, 
             758.10, 769.50, 780.90, 792.30, 803.70, 815.10, 826.50, 
             837.90, 849.30, 860.70, 872.10, 883.50, 894.90, 906.30, 
             917.70, 929.10, 940.50, 951.90, 963.30, 974.70, 986.10, 
             997.50, 1008.90, 1020.30, 1031.70, 1043.10, 1054.50, 
             1065.90, 1077.30, 1088.70, 1100.10, 1111.50, 1122.90, 
             1134.30, 1145.70, 1157.10, 1168.50, 1179.90, 1191.30, 
             1202.70, 1214.10, 1225.50, 1236.90, 1248.30, 1259.70, 
             1271.10, 1282.50, 1293.90, 1305.30, 1316.70, 1328.10, 
             1339.50, 1350.90, 1362.30, 1373.70, 1385.10, 1396.50, 
             1407.90, 1419.30, 1430.70, 1442.10, 1453.50, 1464.90]
#%%
P_06=math.log10(74.10)-math.log10(62.70)
P_07=math.log10(85.50)-math.log10(74.10)
P_08=math.log10(96.90)-math.log10(85.50)
P_09=math.log10(108.30)-math.log10(96.90)
P_10=math.log10(119.70)-math.log10(108.30)
P_11=math.log10(131.10)-math.log10(119.70)
P_12=math.log10(142.50)-math.log10(131.10)
P_23=math.log10(153.90)-math.log10(142.50)
P_24=math.log10(165.30)-math.log10(153.90)
P_25=math.log10(176.70)-math.log10(165.30)
P_26=math.log10(188.10)-math.log10(176.70)
P_27=math.log10(199.50)-math.log10(188.10)
P_28=math.log10(210.90)-math.log10(199.50)
P_29=math.log10(222.30)-math.log10(210.90)
P_30=math.log10(233.70)-math.log10(222.30)
P_31=math.log10(245.10)-math.log10(233.70)
P_32=math.log10(256.50)-math.log10(245.10)
P_33=math.log10(267.90)-math.log10(256.50)
P_34=math.log10(279.30)-math.log10(267.90)  
P_35=math.log10(290.70)-math.log10(279.30)
P_36=math.log10(302.10)-math.log10(290.70)
P_37=math.log10(313.50)-math.log10(302.10)
P_38=math.log10(324.90)-math.log10(313.50)
P_39=math.log10(336.30)-math.log10(324.90)
P_40=math.log10(347.70)-math.log10(336.30)
P_41=math.log10(359.10)-math.log10(347.70)
P_42=math.log10(370.50)-math.log10(359.10)
P_43=math.log10(381.90)-math.log10(370.50)
P_44=math.log10(393.30)-math.log10(381.90)
P_45=math.log10(404.70)-math.log10(393.30)
P_46=math.log10(416.10)-math.log10(404.70)
P_47=math.log10(427.50)-math.log10(416.10)
P_48=math.log10(438.90)-math.log10(427.50)
P_49=math.log10(450.30)-math.log10(438.90)
P_50=math.log10(461.70)-math.log10(450.30)
P_51=math.log10(473.10)-math.log10(461.70)
P_52=math.log10(484.50)-math.log10(473.10)
P_53=math.log10(495.90)-math.log10(484.50)
P_54=math.log10(507.30)-math.log10(495.90)
P_55=math.log10(518.70)-math.log10(507.30)
P_56=math.log10(530.10)-math.log10(518.70)
P_57=math.log10(541.50)-math.log10(530.10)
P_58=math.log10(552.90)-math.log10(541.50)
P_59=math.log10(564.30)-math.log10(552.90)
P_60=math.log10(575.70)-math.log10(564.30)
P_61=math.log10(587.10)-math.log10(575.70)
P_62=math.log10(598.50)-math.log10(587.10)
P_63=math.log10(609.90)-math.log10(598.50)
P_64=math.log10(621.30)-math.log10(609.90)
P_65=math.log10(632.70)-math.log10(621.30)
P_66=math.log10(644.10)-math.log10(632.70)
P_67=math.log10(655.50)-math.log10(644.10)
P_68=math.log10(666.90)-math.log10(655.50)
P_69=math.log10(678.30)-math.log10(666.90)
P_70=math.log10(689.70)-math.log10(678.30)
P_71=math.log10(701.10)-math.log10(689.70)
P_72=math.log10(712.50)-math.log10(701.10)
P_73=math.log10(723.90)-math.log10(712.50)
P_74=math.log10(735.30)-math.log10(723.90)
P_75=math.log10(746.70)-math.log10(735.30)
P_76=math.log10(758.10)-math.log10(746.70)
P_77=math.log10(769.50)-math.log10(758.10)
P_78=math.log10(780.90)-math.log10(769.50)
P_79=math.log10(792.30)-math.log10(780.90)
P_80=math.log10(803.70)-math.log10(792.30)
P_81=math.log10(815.10)-math.log10(803.70)
P_82=math.log10(826.50)-math.log10(815.10)
P_83=math.log10(837.90)-math.log10(826.50)
P_84=math.log10(849.30)-math.log10(837.90)
P_85=math.log10(860.70)-math.log10(849.30)
P_86=math.log10(872.10)-math.log10(860.70)
P_87=math.log10(883.50)-math.log10(872.10)
P_88=math.log10(894.90)-math.log10(883.50)
P_89=math.log10(906.30)-math.log10(894.90)
P_90=math.log10(917.70)-math.log10(906.30)
P_91=math.log10(929.10)-math.log10(917.70)
P_92=math.log10(940.50)-math.log10(929.10)
P_93=math.log10(951.90)-math.log10(940.50)
P_94=math.log10(963.30)-math.log10(951.90)
P_95=math.log10(974.70)-math.log10(963.30)
P_96=math.log10(986.10)-math.log10(974.70)
P_97=math.log10(997.50)-math.log10(986.10)
P_98=math.log10(1008.90)-math.log10(997.50)
P_99=math.log10(1020.30)-math.log10(1008.90)
P_100=math.log10(1031.70)-math.log10(1020.30)
P_101=math.log10(1043.10)-math.log10(1031.70)
P_102=math.log10(1054.50)-math.log10(1043.10)
P_103=math.log10(1065.90)-math.log10(1054.50)
P_104=math.log10(1077.30)-math.log10(1065.90)
P_105=math.log10(1088.70)-math.log10(1077.30)
P_106=math.log10(1100.10)-math.log10(1088.70)
P_107=math.log10(1111.50)-math.log10(1100.10)
P_108=math.log10(1122.90)-math.log10(1111.50)
P_109=math.log10(1134.30)-math.log10(1122.90)
P_110=math.log10(1145.70)-math.log10(1134.30)
P_111=math.log10(1157.10)-math.log10(1145.70)
P_112=math.log10(1168.50)-math.log10(1157.10)
P_113=math.log10(1179.90)-math.log10(1168.50)
P_114=math.log10(1191.30)-math.log10(1179.90)
P_115=math.log10(1202.70)-math.log10(1191.30)
P_116=math.log10(1214.10)-math.log10(1202.70)
P_117=math.log10(1225.50)-math.log10(1214.10)
P_118=math.log10(1236.90)-math.log10(1225.50)
P_119=math.log10(1248.30)-math.log10(1236.90)
P_120=math.log10(1259.70)-math.log10(1248.30)
P_121=math.log10(1271.10)-math.log10(1259.70)
P_122=math.log10(1282.50)-math.log10(1271.10)
P_123=math.log10(1293.90)-math.log10(1282.50)
P_124=math.log10(1305.30)-math.log10(1293.90)
P_125=math.log10(1316.70)-math.log10(1305.30)
P_126=math.log10(1328.10)-math.log10(1316.70)
P_127=math.log10(1339.50)-math.log10(1328.10)
P_128=math.log10(1350.90)-math.log10(1339.50)
P_129=math.log10(1362.30)-math.log10(1350.90)
P_130=math.log10(1373.70)-math.log10(1362.30)
P_131=math.log10(1385.10)-math.log10(1373.70)
P_132=math.log10(1396.50)-math.log10(1385.10)
P_133=math.log10(1407.90)-math.log10(1396.50)
P_134=math.log10(1419.30)-math.log10(1407.90)
P_135=math.log10(1430.70)-math.log10(1419.30)
P_136=math.log10(1442.10)-math.log10(1430.70)
P_137=math.log10(1453.50)-math.log10(1442.10)
P_138=math.log10(1464.90)-math.log10(1453.50)

twoDS_logg=[P_06, P_07, P_08, P_09, P_10, P_11, P_12, P_23, P_24, P_25, 
            P_26, P_27, P_28, P_29, P_30, P_31, P_32, P_33, P_34, P_35,
            P_36, P_37, P_38, P_39, P_40, P_41, P_42, P_43, P_44, P_45,
            P_46, P_47, P_48, P_49, P_50, P_51, P_52, P_53, P_54, P_55, 
            P_56, P_57, P_58, P_59, P_60, P_61, P_62, P_63, P_64, P_65, 
            P_66, P_67, P_68, P_69, P_70, P_71, P_72, P_73, P_74, P_75, 
            P_76, P_77, P_78, P_79, P_80, P_81, P_82, P_83, P_84, P_85, 
            P_86, P_87, P_88, P_89, P_90, P_91, P_92, P_93, P_94, P_95, 
            P_96, P_97, P_98, P_99, P_100, P_101, P_102, P_103, P_104, 
            P_105, P_106, P_107, P_108, P_109, P_110, P_111, P_112, P_113, 
            P_114, P_115, P_116, P_117, P_118, P_119, P_120, P_121, P_122, 
            P_123, P_124, P_125, P_126, P_127, P_128, P_129, P_130,
            P_131, P_132, P_133, P_134, P_135, P_136, P_137, P_138] 
            

#%%
#combined ACB and BCT legs
rain_concentrations = []

for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict = leg_data[i]

    ACB_start = leg_dict['LegIndex_03']['StartTimes']
    ACB_stop = leg_dict['LegIndex_03']['StopTimes']
    BCT_start = leg_dict['LegIndex_04']['StartTimes']
    BCT_stop = leg_dict['LegIndex_04']['StopTimes']
    all_legs_start = ACB_start + BCT_start
    all_legs_stop = ACB_stop + BCT_stop

    twoDS_flight = twoDS[i]
    twoDS_flight['Time_Start'] = pd.to_numeric(twoDS_flight['Time_Start'], errors='coerce')

    twoDS_times = twoDS_flight['Time_Start'].values
    twoDS_lwc = twoDS_flight['LWC_2DS'].values
    twoDS_bins = {f'dNdlogD_liquid_{bin_label:03d}_2DS': twoDS_flight[f'dNdlogD_liquid_{bin_label:03d}_2DS'].values
                  for bin_label in range(6, 129)}

    for k in range(len(all_legs_start)):
        start_time = all_legs_start[k]
        end_time = all_legs_stop[k]

        twoDS_indices_in_range = np.where((twoDS_times >= start_time) & (twoDS_times <= end_time))[0]

        for twoDS_idx in twoDS_indices_in_range:
            lwc_val = twoDS_lwc[twoDS_idx]

            if lwc_val >= 0.00001:  # LWC threshold (0.01 g/m³ = 1e-5 kg/m³)
                
                total_concentration = sum(
                    np.nan_to_num(twoDS_bins[f'dNdlogD_liquid_{bin_label:03d}_2DS'][twoDS_idx]) * log_width
                    for bin_label, log_width in zip(range(6, 129), twoDS_logg)
                )


                total_concentration /= 1e6  # /m³ to /cm³


                rain_entry = {
                    'Date': date,
                    'Time': twoDS_times[twoDS_idx],
                    'Leg_start': start_time,
                    'Leg_stop': end_time,
                    'LWC': lwc_val,  
                    'Total_Concentration': total_concentration 
                }

                rain_concentrations.append(rain_entry)

print(f"Number of rain entries: {len(rain_concentrations)}")
print(f"First 5 entries: {rain_concentrations[:5]}")


# %%
# Convert LWC to g/m³ and N_liquid to /cm³
for entry in rain_concentrations:
    entry['LWC'] = entry['LWC'] * 1e3  # kg/m³ to g/m³

print("Sample entries after unit conversion:")
for sample in rain_concentrations[:5]:
    print(sample)

# %%
# Convert Bin_Lower and Bin_Upper from µm to meters (once, since they are constant)
Bin_Lower_m = [lower / 1e6 for lower in Bin_Lower]  # Convert µm to m
Bin_Upper_m = [upper / 1e6 for upper in Bin_Upper]  # Convert µm to m
Bin_Centers_m = [(lower + upper) / 2 for lower, upper in zip(Bin_Lower_m, Bin_Upper_m)]  # Bin centers in meters
Bin_Centers_Cubed = [center**3 for center in Bin_Centers_m] 
print("Cubed Bin Centers (in m³):")
for i, (center, cubed) in enumerate(zip(Bin_Centers_m, Bin_Centers_Cubed), start=1):
    print(f"Bin {i}: Center = {center:.6e} m, Center³ = {cubed:.6e} m³")



# %%
#calculating rain water content 
rho_water = 1e3 # Density of water in g/m³
pi_over_6 = np.pi / 6
rain_water_content = []

for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict = leg_data[i]

    ACB_start = leg_dict['LegIndex_03']['StartTimes']
    ACB_stop = leg_dict['LegIndex_03']['StopTimes']
    BCT_start = leg_dict['LegIndex_04']['StartTimes']
    BCT_stop = leg_dict['LegIndex_04']['StopTimes']
    all_legs_start = ACB_start + BCT_start
    all_legs_stop = ACB_stop + BCT_stop

    twoDS_flight = twoDS[i]
    twoDS_flight['Time_Start'] = pd.to_numeric(twoDS_flight['Time_Start'], errors='coerce')

    twoDS_times = twoDS_flight['Time_Start'].values
    twoDS_bins = {f'dNdlogD_liquid_{bin_label:03d}_2DS': twoDS_flight[f'dNdlogD_liquid_{bin_label:03d}_2DS'].values
                  for bin_label in range(6, 129)}

    for k in range(len(all_legs_start)):
        start_time = all_legs_start[k]
        end_time = all_legs_stop[k]

        twoDS_indices_in_range = np.where((twoDS_times >= start_time) & (twoDS_times <= end_time))[0]

        for twoDS_idx in twoDS_indices_in_range:
            lwc_val = twoDS_flight['LWC_2DS'].iloc[twoDS_idx]
            N_liquid_total = 0

            if lwc_val >= 0.00001:  # LWC threshold (0.01 g/m³ = 1e-5 kg/m³)
            
                for bin_label in (range(6, 129)):
                    bin_column = f'dNdlogD_liquid_{bin_label:03d}_2DS'
                    if bin_column in twoDS_flight.columns:
                        N_bin = twoDS_flight[bin_column].iloc[twoDS_idx]  # Raw bin value in /m³
                        
                        N_dD = (N_bin * twoDS_logg[bin_label - 6])
                        
                        N_liquid_total += N_dD * Bin_Centers_Cubed[bin_label - 6]


                RWC = pi_over_6 * rho_water * N_liquid_total # kg/m³

             

                rain_water_content.append({
                    'Date': date,
                    'Time': twoDS_times[twoDS_idx],
                    'Leg_start': start_time,
                    'Leg_stop': end_time,
                    'LWC': lwc_val,
                    'RWC': RWC

                })

print(f"Number of RWC entries: {len(rain_water_content)}")
print(f"First 5 entries: {rain_water_content[:5]}")
#%%
# convert RWC to g/m³ 
for entry in rain_water_content:
    entry['RWC'] = entry['RWC'] * 1e3  # kg/m³ to g/m³
    entry['LWC'] = entry['LWC'] * 1e3  # kg/m³ to g/m³
#%%
#add RWC and CWC for total LWC
total_liquid_water = []

for rwc_entry in rain_water_content:  
    matching_time = rwc_entry['Time']
    matching_date = rwc_entry['Date']

    matching_cwc = next((entry for entry in in_cloud_concentrations if entry['Time'] == matching_time and entry['Date'] == matching_date), None)

    if matching_cwc:
        cwc_val = matching_cwc['CWC'] 
        rwc_val = rwc_entry['RWC'] 
        total_liquid = cwc_val + rwc_val

        total_liquid_water.append({
            'Date': matching_date,
            'Time': matching_time,
            'Leg_start': rwc_entry['Leg_start'],
            'Leg_stop': rwc_entry['Leg_stop'],
            'CWC': cwc_val,
            'RWC': rwc_val,
            'Total_Liquid_Water': total_liquid  
        })

print(f"Number of total liquid water entries: {len(total_liquid_water)}")
print(f"First 5 entries: {total_liquid_water[:5]}")

#%%
#add the Nc + Nr for total concentration
total_combined_concentration = []

for in_cloud_entry in in_cloud_concentrations: 
    matching_time = in_cloud_entry['Time']
    matching_date = in_cloud_entry['Date']

    matching_rain = next((entry for entry in rain_concentrations if entry['Time'] == matching_time and entry['Date'] == matching_date), None)
    
    
    if matching_rain:
        rain_val = matching_rain['Total_Concentration']
        inc_val = in_cloud_entry['Total_Concentration'] 
        combined_conc = inc_val + rain_val

        total_combined_concentration.append({
            'Date': matching_date,
            'Time': matching_time,
            'Leg_start': matching_rain['Leg_start'],
            'Leg_stop': matching_rain['Leg_stop'],
            'In_Cloud_Concentration': inc_val,
            'Rain_Concentration': rain_val,
            'Total_Combined_Concentration': combined_conc 
        })

print(f"Number of total combined concentration entries: {len(total_combined_concentration)}")
print(f"First 5 entries: {total_combined_concentration[:5]}")

#%% 
concentration = [entry['Total_Combined_Concentration'] for entry in total_combined_concentration]
total_liquid_water_values = [entry['Total_Liquid_Water'] for entry in total_liquid_water]  
rain_water_content_values = [entry['RWC'] for entry in total_liquid_water]  

rwc_percentage = []
for rwc, total in zip(rain_water_content_values, total_liquid_water_values):
    if total > 0:
        rwc_percentage.append((rwc / total) * 100)
    else:
        rwc_percentage.append(0) 
bins = 100  
plt.figure(figsize=(8, 6))
hist, xedges, yedges, img = plt.hist2d(concentration, total_liquid_water_values, bins=bins, 
                                       weights=rwc_percentage, cmap='RdBu_r', cmin=1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc /cm³ (log scale)', fontsize=16, fontweight='bold')
plt.ylabel('LWC g/m³ (log scale)', fontsize=16, fontweight='bold')
plt.title('RWC Percentage of Total Liquid Water', fontsize=18, fontweight='bold')
cbar = plt.colorbar(img)
cbar.set_label("Rainwater % of Total Liquid Water", fontsize=14)  
plt.grid(which="both", linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.show()

# %%
# Create histogram bins
bins = 100
counts, xedges, yedges = np.histogram2d(concentration, total_liquid_water_values, bins=bins)
sum_rwc, _, _ = np.histogram2d(concentration, total_liquid_water_values, bins=bins, weights=rwc_percentage)
mean_rwc = np.divide(sum_rwc, counts, out=np.zeros_like(sum_rwc), where=counts > 0)

plt.figure(figsize=(8, 6))
img = plt.pcolormesh(xedges, yedges, mean_rwc.T, cmap='RdBu_r', vmin=1, vmax=100)
cbar = plt.colorbar(img)
cbar.set_label("RWC/LWC (%)", fontsize=14)
plt.ylim(10**-2, 10**0.2) 
plt.xscale('log')
plt.yscale('log')
plt.tick_params(axis='both', which='major', labelsize=16, width=3, length=8)  # Major ticks
plt.tick_params(axis='both', which='minor', labelsize=14, width=2, length=5)
plt.xlabel('Nr+Nc /cm³', fontsize=16, fontweight='bold')
plt.ylabel('LWC g/m³', fontsize=16, fontweight='bold')
plt.title('CDP in-cloud January - June 2022', fontsize=18, fontweight='bold')
plt.grid(which="both", linestyle='--', linewidth=0.5, alpha=0.7)
plt.show()

#%%
concentration = np.array([entry['Total_Combined_Concentration'] for entry in total_combined_concentration])
total_liquid_water_values = np.array([entry['Total_Liquid_Water'] for entry in total_liquid_water])
rain_water_content_values = np.array([entry['RWC'] for entry in total_liquid_water])
rwc_percentage = np.divide(rain_water_content_values, total_liquid_water_values, 
                           out=np.full_like(rain_water_content_values, np.nan), where=total_liquid_water_values > 0) * 100  
num_bins = 5
x_bins = np.logspace(np.log10(min(concentration)), np.log10(max(concentration)), num_bins)
y_bins = np.logspace(np.log10(min(total_liquid_water_values)), np.log10(max(total_liquid_water_values)), num_bins)
counts, xedges, yedges = np.histogram2d(concentration, total_liquid_water_values, bins=[x_bins, y_bins])
sum_rwc, _, _ = np.histogram2d(concentration, total_liquid_water_values, bins=[x_bins, y_bins], weights=rwc_percentage)
mean_rwc = np.divide(sum_rwc, counts, out=np.full_like(sum_rwc, np.nan), where=counts > 0)  
masked_rwc = np.ma.masked_where(np.isnan(mean_rwc), mean_rwc)
cmap = plt.get_cmap('RdBu_r')
cmap.set_bad(color='gray') 
plt.figure(figsize=(8, 6))
norm = mcolors.Normalize(vmin=1, vmax=100)
img = plt.pcolormesh(xedges, yedges, masked_rwc.T, cmap=cmap, norm=norm, shading='auto')
cbar = plt.colorbar(img)
cbar.set_label("RWC/LWC %", fontsize=20, fontweight='bold')
cbar.ax.tick_params(labelsize=15, width=2, length=5) 
for t in cbar.ax.get_yticklabels():  
    t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=20, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=20, fontweight='bold')
plt.title('CDP (in cloud) \nJanuary-June 2022', fontsize=20, fontweight='bold')
plt.tight_layout()
plt.show()
# %%
#average RWC divided by average LWC in each bin
concentration = np.array([entry['Total_Combined_Concentration'] for entry in total_combined_concentration])
total_liquid_water_values = np.array([entry['Total_Liquid_Water'] for entry in total_liquid_water])
rain_water_content_values = np.array([entry['RWC'] for entry in total_liquid_water])
num_bins = 5
x_bins = np.logspace(np.log10(1), np.log10(max(concentration)), num_bins)
y_bins = np.logspace(np.log10(min(total_liquid_water_values)), np.log10(max(total_liquid_water_values)), num_bins)
sum_rwc, xedges, yedges = np.histogram2d(concentration, total_liquid_water_values, bins=[x_bins, y_bins], weights=rain_water_content_values)
sum_lwc, _, _ = np.histogram2d(concentration, total_liquid_water_values, bins=[x_bins, y_bins], weights=total_liquid_water_values)
counts, _, _ = np.histogram2d(concentration, total_liquid_water_values, bins=[x_bins, y_bins])
avg_rwc = np.divide(sum_rwc, counts, out=np.full_like(sum_rwc, np.nan), where=counts > 0)  # Average RWC per bin
avg_lwc = np.divide(sum_lwc, counts, out=np.full_like(sum_lwc, np.nan), where=counts > 0)  # Average LWC per bin
rwc_lwc_ratio = np.divide(avg_rwc, avg_lwc, out=np.full_like(avg_rwc, np.nan), where=avg_lwc > 0) * 100  # RWC / LWC * 100
masked_rwc_lwc_ratio = np.ma.masked_where(np.isnan(rwc_lwc_ratio), rwc_lwc_ratio)
plt.figure(figsize=(8, 6))
norm = mcolors.Normalize(vmin=1, vmax=100)
img = plt.pcolormesh(xedges, yedges, masked_rwc_lwc_ratio.T, cmap="RdBu_r", norm=norm, shading='auto')
gray_mask = np.isnan(rwc_lwc_ratio)  
gray_values = np.full_like(rwc_lwc_ratio, np.nan)
gray_values[gray_mask] = 1  
plt.pcolormesh(xedges, yedges, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)
cbar = plt.colorbar(img)
cbar.set_label("RWC/LWC (%)", fontsize=18, fontweight='bold') 
cbar.ax.tick_params(labelsize=18, width=2, length=5) 
for t in cbar.ax.get_yticklabels():  
    t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.xlabel('Nr+Nc /cm³', fontsize=19, fontweight='bold')
plt.ylabel('LWC g/m³', fontsize=19, fontweight='bold')
plt.title('CDP (in cloud)\n January-June 2022\n RWC as a function of number concentration', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
#trying to seperate LWC and RWC 
masked_avg_rwc = np.ma.masked_where(np.isnan(avg_rwc), avg_rwc)

plt.figure(figsize=(8, 6))
img = plt.pcolormesh(
    xedges,
    yedges,
    masked_avg_rwc.T,
    cmap="viridis",
    shading='auto',
    vmin=0, vmax=1
)
gray_mask = np.isnan(avg_rwc)
gray_values = np.full_like(avg_rwc, np.nan)
gray_values[gray_mask] = 1
plt.pcolormesh(
    xedges,
    yedges,
    gray_values.T,
    cmap=mcolors.ListedColormap(["gray"]),
    shading='auto',
    alpha=0.6
)
cbar = plt.colorbar(img)
cbar.set_label("Mean RWC (g m$^{-3}$)", fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=18, width=2, length=5)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight('bold')

plt.xscale('log')
plt.yscale('log')
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.xlabel('Nr+Nc /cm³', fontsize=19, fontweight='bold')
plt.ylabel('LWC g/m³', fontsize=19, fontweight='bold')
plt.title('CDP (in cloud)\nMean RWC\nJanuary–June 2022', fontsize=18, fontweight='bold')
plt.tight_layout()
fig.savefig("RWC_CDP.pdf", dpi=300, bbox_inches='tight')
plt.show()
fig.savefig("RWC_RWC_CDP.pdf", dpi=300, bbox_inches='tight')
#%%
#fixing for paper 
from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors

min_points = 100
valid_bins = counts >= min_points

masked_avg_rwc = np.ma.masked_where(
    np.isnan(avg_rwc) | (avg_rwc <= 0) | (~valid_bins),
    avg_rwc
)

cmap = plt.cm.viridis.copy()
cmap.set_bad(color='gray')

fig = plt.figure(figsize=(8, 6))
img = plt.pcolormesh(
    xedges,
    yedges,
    masked_avg_rwc.T,
    cmap=cmap,
    shading='auto',
    norm=LogNorm(vmin=0.01, vmax=0.5)
)

cbar = plt.colorbar(img, ticks=[0.01, 0.03, 0.1, 0.3, 0.5])
cbar.set_label("Mean RWC (g m$^{-3}$)", fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=18, width=2, length=5)
cbar.ax.set_yticklabels(["0.01", "0.03", "0.1", "0.3", "0.5"])
for t in cbar.ax.get_yticklabels():
    t.set_fontweight('bold')

plt.xscale('log')
plt.yscale('log')
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.xlabel(r"$N_d$ (cm$^{-3}$)", fontsize=19, fontweight='bold')
plt.ylabel(r"LWC (g m$^{-3}$)", fontsize=19, fontweight='bold')
plt.title('CDP (in cloud)\nMean RWC\nJanuary–June 2022', fontsize=18, fontweight='bold')
plt.tight_layout()
fig.savefig("RWC_RWC_CDP.pdf", dpi=300, bbox_inches='tight')
plt.show()
#%%
masked_avg_lwc = np.ma.masked_where(np.isnan(avg_lwc), avg_lwc)

plt.figure(figsize=(8, 6))
img = plt.pcolormesh(xedges, yedges, masked_avg_lwc.T, cmap="plasma", shading='auto')

gray_mask = np.isnan(avg_lwc)
gray_values = np.full_like(avg_lwc, np.nan)
gray_values[gray_mask] = 1
plt.pcolormesh(xedges, yedges, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)

cbar = plt.colorbar(img)
cbar.set_label("Mean LWC (g m$^{-3}$)", fontsize=18, fontweight='bold') 
cbar.ax.tick_params(labelsize=18, width=2, length=5)
for t in cbar.ax.get_yticklabels():  
    t.set_fontweight('bold')

plt.xscale('log')
plt.yscale('log')
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.xlabel('Nr+Nc /cm³', fontsize=19, fontweight='bold')
plt.ylabel('LWC g/m³', fontsize=19, fontweight='bold')
plt.title('Mean LWC\nJanuary–June 2022 (CDP in cloud)', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
#%%
num_bins = 5
x_bins = np.logspace(np.log10(1), np.log10(np.max(concentration)), num_bins)
y_bins = np.logspace(np.log10(np.min(total_liquid_water_values)), np.log10(np.max(total_liquid_water_values)), num_bins)
density_counts, xedges, yedges = np.histogram2d(
    concentration, 
    total_liquid_water_values, 
    bins=[x_bins, y_bins]
)

plt.figure(figsize=(8, 6))
img = plt.pcolormesh(
    xedges, yedges, density_counts.T,
    cmap="plasma", 
    shading='auto',
    norm=mcolors.LogNorm(vmax=np.max(density_counts) * 1.1)
)
cbar = plt.colorbar(img)
cbar.set_label("Density of Observations", fontsize=14, fontweight='bold')
cbar.ax.tick_params(labelsize=12, width=2, length=5)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=16, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=16, fontweight='bold')
plt.title('CAS in-cloud January-June 2022', fontsize=18, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=12, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=12, width=2, length=5)
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
plt.hist(concentration, bins=x_bins, color="darkred", alpha=0.7, log=True)
plt.xscale('log')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=16, fontweight='bold')
plt.ylabel('Frequency', fontsize=16, fontweight='bold')
plt.title('CDP in-cloud January-June 2022', fontsize=18, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=12, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=12, width=2, length=5)
plt.xlim(10**1, 10**np.ceil(np.log10(np.max(concentration))))
plt.tight_layout()
plt.show()
#%%
cmap = plt.cm.plasma.copy()
cmap.set_bad(color="gray")
masked_counts = np.ma.masked_where(density_counts == 0, density_counts)

plt.figure(figsize=(8, 6))
img = plt.pcolormesh(
    xedges,
    yedges,
    masked_counts.T,
    cmap=cmap,
    shading="auto",
    norm=mcolors.LogNorm(vmax=np.max(density_counts) * 1.1)
)
cbar = plt.colorbar(img)
cbar.set_label("Density of Observations", fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=18, width=2, length=5)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight('bold')

plt.xscale('log')
plt.yscale('log')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.title('CDP (in-cloud)\n January-June 2022)', fontsize=19, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=12, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=12, width=2, length=5)
plt.tight_layout()
plt.show()
#%%
#printing obs in each box
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(
    xedges,
    yedges,
    masked_counts.T,
    cmap=cmap,
    shading="auto",
    norm=mcolors.LogNorm(vmax=np.max(density_counts) * 1.1)
)
cbar = plt.colorbar(img)
cbar.set_label("Density of Observations", fontsize=20, fontweight='bold')
cbar.ax.tick_params(labelsize=20, width=2, length=5)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight('bold')

plt.xscale('log')
plt.yscale('log')
plt.xticks(fontsize=20, fontweight='bold')
plt.yticks(fontsize=20, fontweight='bold')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=20, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=20, fontweight='bold')
plt.title('CDP(in-cloud)\nJanuary-June 2022', fontsize=19, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=16, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=16, width=2, length=5)
x_centers = 0.5 * (xedges[:-1] + xedges[1:])
y_centers = 0.5 * (yedges[:-1] + yedges[1:])

for i, xc in enumerate(x_centers):
    for j, yc in enumerate(y_centers):
        val = density_counts[i, j]
        if val > 0:  
            plt.text(
                xc, yc, int(val),
                ha='center', va='center',
                color='black', fontsize=16, fontweight='bold'
            )

plt.tight_layout()
plt.show()

#%%
#adding the black box to selected region

concentration = np.array([entry['Total_Combined_Concentration'] for entry in total_combined_concentration])
total_liquid_water_values = np.array([entry['Total_Liquid_Water'] for entry in total_liquid_water])
rain_water_content_values = np.array([entry['RWC'] for entry in total_liquid_water])
num_bins = 5
x_bins = np.logspace(np.log10(1), np.log10(max(concentration)), num_bins)
y_bins = np.logspace(np.log10(min(total_liquid_water_values)), np.log10(max(total_liquid_water_values)), num_bins)
sum_rwc, xedges, yedges = np.histogram2d(concentration, total_liquid_water_values, bins=[x_bins, y_bins], weights=rain_water_content_values)
sum_lwc, _, _ = np.histogram2d(concentration, total_liquid_water_values, bins=[x_bins, y_bins], weights=total_liquid_water_values)
counts, _, _ = np.histogram2d(concentration, total_liquid_water_values, bins=[x_bins, y_bins])
avg_rwc = np.divide(sum_rwc, counts, out=np.full_like(sum_rwc, np.nan), where=counts > 0)  
avg_lwc = np.divide(sum_lwc, counts, out=np.full_like(sum_rwc, np.nan), where=counts > 0)  
rwc_lwc_ratio = np.divide(avg_rwc, avg_lwc, out=np.full_like(avg_rwc, np.nan), where=avg_lwc > 0) * 100  
masked_rwc_lwc_ratio = np.ma.masked_where(np.isnan(rwc_lwc_ratio), rwc_lwc_ratio)
plt.figure(figsize=(8, 6))
norm = mcolors.Normalize(vmin=1, vmax=100)
img = plt.pcolormesh(xedges, yedges, masked_rwc_lwc_ratio.T, cmap="RdBu_r", norm=norm, shading='auto')
gray_mask = np.isnan(rwc_lwc_ratio)  
gray_values = np.full_like(rwc_lwc_ratio, np.nan)
gray_values[gray_mask] = 1  
plt.pcolormesh(xedges, yedges, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)
cbar = plt.colorbar(img)
cbar.set_label("RWC / LWC (%)", fontsize=17, fontweight='bold') 
cbar.ax.tick_params(labelsize=19, width=2, length=5) 
for t in cbar.ax.get_yticklabels():  
    t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.title('CDP (in cloud)\n January-June 2022\n RWC as a function of number concentration', fontsize=18, fontweight='bold')
box_x_min, box_x_max = 25.182, 634.143  # Nr+Nc range
box_y_min, box_y_max = 0.041, 0.580 
plt.plot([box_x_min, box_x_max, box_x_max, box_x_min, box_x_min],
         [box_y_min, box_y_min, box_y_max, box_y_max, box_y_min], 
         color='black', linewidth=3) 

plt.tight_layout()
plt.show()
#%%
num_bins = 5
x_bins = np.logspace(np.log10(1), np.log10(max(concentration)), num_bins)
y_bins = np.logspace(np.log10(min(total_liquid_water_values)), np.log10(max(total_liquid_water_values)), num_bins)
sum_rwc, xedges, yedges = np.histogram2d(
    concentration,
    total_liquid_water_values,
    bins=[x_bins, y_bins],
    weights=rain_water_content_values
)
sum_lwc, _, _ = np.histogram2d(
    concentration,
    total_liquid_water_values,
    bins=[x_bins, y_bins],
    weights=total_liquid_water_values
)
counts, _, _ = np.histogram2d(
    concentration,
    total_liquid_water_values,
    bins=[x_bins, y_bins]
)

avg_rwc = np.divide(sum_rwc, counts, out=np.full_like(sum_rwc, np.nan), where=counts > 0)
avg_lwc = np.divide(sum_lwc, counts, out=np.full_like(sum_lwc, np.nan), where=counts > 0)
rwc_lwc_ratio = np.divide(avg_rwc, avg_lwc, out=np.full_like(avg_rwc, np.nan), where=avg_lwc > 0) * 100
masked_rwc_lwc_ratio = np.ma.masked_invalid(rwc_lwc_ratio)
cmap = plt.cm.plasma.copy()
cmap.set_bad(color="gray")
plt.figure(figsize=(8, 6))
norm = mcolors.Normalize(vmin=1, vmax=100)

img = plt.pcolormesh(
    xedges,
    yedges,
    masked_rwc_lwc_ratio.T,
    cmap=cmap,
    norm=norm,
    shading='auto'
)

cbar = plt.colorbar(img)
cbar.set_label("RWC / LWC (%)", fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=19, width=2, length=5)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight('bold')

plt.xscale('log')
plt.yscale('log')
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.title('CDP (in-cloud)\n January-June 2022\n RWC as a function of number concentration', fontsize=18, fontweight='bold')
box_x_min, box_x_max = 25.182, 634.143  # Nr+Nc range
box_y_min, box_y_max = 0.041, 0.580
plt.plot(
    [box_x_min, box_x_max, box_x_max, box_x_min, box_x_min],
    [box_y_min, box_y_min, box_y_max, box_y_max, box_y_min],
    color='black',
    linewidth=3
)

plt.tight_layout()
plt.show()
#%%
#creating my own colorbar
from matplotlib.cm import get_cmap
num_bins = 5
x_bins = np.logspace(np.log10(1), np.log10(max(concentration)), num_bins)
y_bins = np.logspace(np.log10(min(total_liquid_water_values)), np.log10(max(total_liquid_water_values)), num_bins)
sum_rwc, xedges, yedges = np.histogram2d(
    concentration,
    total_liquid_water_values,
    bins=[x_bins, y_bins],
    weights=rain_water_content_values
)
sum_lwc, _, _ = np.histogram2d(
    concentration,
    total_liquid_water_values,
    bins=[x_bins, y_bins],
    weights=total_liquid_water_values
)
counts, _, _ = np.histogram2d(
    concentration,
    total_liquid_water_values,
    bins=[x_bins, y_bins]
)

avg_rwc = np.divide(sum_rwc, counts, out=np.full_like(sum_rwc, np.nan), where=counts > 0)
avg_lwc = np.divide(sum_lwc, counts, out=np.full_like(sum_lwc, np.nan), where=counts > 0)
rwc_lwc_ratio = np.divide(avg_rwc, avg_lwc, out=np.full_like(avg_rwc, np.nan), where=avg_lwc > 0) * 100
masked_rwc_lwc_ratio = np.ma.masked_invalid(rwc_lwc_ratio)
valid_data = rwc_lwc_ratio[~np.isnan(rwc_lwc_ratio)].flatten()
bounds = [0, 2, 5, 10, 20, 40, 60, 80, 100] 
cmap = plt.cm.plasma.copy()
cmap.set_bad(color="gray")
norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N, extend='neither')
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(
    xedges,
    yedges,
    masked_rwc_lwc_ratio.T,
    cmap=cmap,
    norm=norm,
    shading='auto'
)
cbar = plt.colorbar(img, ticks=bounds)
cbar.set_label("RWC / LWC (%)", fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=19, width=2, length=5)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.title('CDP (in-cloud)\nJanuary–June 2022\nRWC as a function of number concentration',
          fontsize=18, fontweight='bold')
x_start_idx = 2
x_end_idx = 4
y_start_idx = 1
y_end_idx = 3

box_x_min = xedges[x_start_idx]
box_x_max = xedges[x_end_idx]
box_y_min = yedges[y_start_idx]
box_y_max = yedges[y_end_idx]

plt.plot(
    [box_x_min, box_x_max, box_x_max, box_x_min, box_x_min],
    [box_y_min, box_y_min, box_y_max, box_y_max, box_y_min],
    color='black',
    linewidth=3
)
plt.tight_layout()
plt.show()
print(f"Black box x-range: {box_x_min:.3f} to {box_x_max:.3f}")
print(f"Black box y-range: {box_y_min:.3f} to {box_y_max:.3f}")

# %%
#We need to figure out how to incporate GCCN into this RWC vs concentration relationship
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
#Now we need to pull the droplet concentration from each bin for each flight leg and calculate the bin
#mean concentration for each leg. You should end up with 18 mean values, 1 for each bin, for each leg. 


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
#Sum across all bins for each leg in dN for a total concentration per leg of GCCN
for entry in Y_CDP_calc:
    bin_keys = [f'Bin{bin_label}_Y_mean' for bin_label in range(0, 30)]
    entry['Total_GCCN_Concentration'] = np.nansum([entry[key] for key in bin_keys if key in entry])
#%%
# %%
from collections import defaultdict

GCCN_flight_totals = defaultdict(lambda: {'Legs': [], 'Total_GCCN_Concentration': 0, 'Leg_Count': 0})

for entry in Y_CDP_calc:
    date = entry['Date']
    start_time = entry['BCB_start']
    stop_time = entry['BCB_stop']
    total_gccn_leg = entry['Total_GCCN_Concentration']

    GCCN_flight_totals[date]['Legs'].append({
        'Leg_start': start_time,
        'Leg_stop': stop_time,
        'Leg_GCCN_Concentration': total_gccn_leg
    })

    
    GCCN_flight_totals[date]['Total_GCCN_Concentration'] += total_gccn_leg
    GCCN_flight_totals[date]['Leg_Count'] += 1  

GCCN_flight_totals = dict(GCCN_flight_totals)

for date, flight_data in GCCN_flight_totals.items():
    print(f"Date: {date}, Total GCCN: {flight_data['Total_GCCN_Concentration']}")
    for leg in flight_data['Legs']:
        print(f"   Leg Start: {leg['Leg_start']}, Leg Stop: {leg['Leg_stop']}, Leg GCCN: {leg['Leg_GCCN_Concentration']}")

#%%
average_gccn_per_flight = {}

for date, flight_data in GCCN_flight_totals.items():
    if flight_data['Leg_Count'] > 0:
        average_gccn_per_flight[date] = flight_data['Total_GCCN_Concentration'] / flight_data['Leg_Count']
    else:
        average_gccn_per_flight[date] = np.nan 

print("Average GCCN per Flight Dictionary:")
for date, avg_gccn in average_gccn_per_flight.items():
    print(f"Date: {date}, Average GCCN: {avg_gccn:.4f}")

#%%
#splitting flights based on high and low average GCCN
gccn_values = np.array(list(average_gccn_per_flight.values()))
#%%
threshold = np.percentile(gccn_values, 50)

high_GCCN_concentrations = {}
low_GCCN_concentrations = {}

for date, avg_gccn in average_gccn_per_flight.items():
    if avg_gccn >= threshold:
        high_GCCN_concentrations[date] = avg_gccn  
    else:
        low_GCCN_concentrations[date] = avg_gccn 

print(f"GCCN Threshold: Low < {threshold:.4f} cm⁻³, High ≥ {threshold:.4f} cm⁻³")
print(f"High GCCN Flights: {len(high_GCCN_concentrations)}")
print(f"Low GCCN Flights: {len(low_GCCN_concentrations)}")

print("\n**Low GCCN Flights:**")
for date, gccn in sorted(low_GCCN_concentrations.items(), key=lambda x: x[1]):
    print(f"Date: {date}, Total GCCN: {gccn:.4f} cm⁻³")

print("\n**High GCCN Flights:**")
for date, gccn in sorted(high_GCCN_concentrations.items(), key=lambda x: x[1], reverse=True):
    print(f"Date: {date}, Total GCCN: {gccn:.4f} cm⁻³")

#%%
#GCCN average concentrations per flight 
gccn_values = np.array(list(average_gccn_per_flight.values()))  
high_gccn_values = np.array(list(high_GCCN_concentrations.values()))
low_gccn_values = np.array(list(low_GCCN_concentrations.values()))
df_gccn = pd.DataFrame({
    "GCCN Concentration (cm⁻³)": np.concatenate([high_gccn_values, low_gccn_values]),
    "Flight Type": ["High GCCN"] * len(high_gccn_values) + ["Low GCCN"] * len(low_gccn_values)
})

plt.figure(figsize=(8, 6))
sns.violinplot(x="Flight Type", y="GCCN Concentration (cm⁻³)", data=df_gccn, inner="box", palette=["mediumpurple", "rebeccapurple"], scale="width")
plt.yscale('log')
plt.ylabel("GCCN Concentration (cm⁻³)", fontsize=20, fontweight="bold")
plt.xlabel("GCCN Flight Category", fontsize=20, fontweight="bold")
plt.title("Comparison of High & Low GCCN Flight Categories", fontsize=16, fontweight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tick_params(axis="both", which="major", labelsize=18, width=3, length=8)
plt.tick_params(axis="both", which="minor", labelsize=18, width=2, length=5)
plt.show()
#%%
#Average concentration stats
avg_high_gccn = np.mean(high_gccn_values)
avg_low_gccn = np.mean(low_gccn_values)
num_high_flights = len(high_gccn_values)
num_low_flights = len(low_gccn_values)
print(f"Average High GCCN Flight Concentration: {avg_high_gccn:.4f} cm⁻³")
print(f"Number of High GCCN Flights: {num_high_flights}")

print(f"Average Low GCCN Flight Concentration: {avg_low_gccn:.4f} cm⁻³")
print(f"Number of Low GCCN Flights: {num_low_flights}")

#%%
#Splitting the RWC plots based on which flights are categorized as high and low GCCN

high_gccn_data = [entry for entry in total_combined_concentration if entry['Date'] in high_GCCN_concentrations]
low_gccn_data = [entry for entry in total_combined_concentration if entry['Date'] in low_GCCN_concentrations]

high_concentration = np.array([entry['Total_Combined_Concentration'] for entry in high_gccn_data])
high_lwc = np.array([entry['Total_Liquid_Water'] for entry in total_liquid_water if entry['Date'] in high_GCCN_concentrations])
high_rwc = np.array([entry['RWC'] for entry in total_liquid_water if entry['Date'] in high_GCCN_concentrations])

low_concentration = np.array([entry['Total_Combined_Concentration'] for entry in low_gccn_data])
low_lwc = np.array([entry['Total_Liquid_Water'] for entry in total_liquid_water if entry['Date'] in low_GCCN_concentrations])
low_rwc = np.array([entry['RWC'] for entry in total_liquid_water if entry['Date'] in low_GCCN_concentrations])
num_bins = 5
all_concentration = np.concatenate([high_concentration, low_concentration])
all_lwc = np.concatenate([high_lwc, low_lwc])

x_bins = np.logspace(np.log10(np.nanmin(all_concentration)), np.log10(np.nanmax(all_concentration)), num_bins)
y_bins = np.logspace(np.log10(np.nanmin(all_lwc)), np.log10(np.nanmax(all_lwc)), num_bins)
sum_rwc_high, xedges, yedges = np.histogram2d(high_concentration, high_lwc, bins=[x_bins, y_bins], weights=high_rwc)
sum_lwc_high, _, _ = np.histogram2d(high_concentration, high_lwc, bins=[x_bins, y_bins], weights=high_lwc)
counts_high, _, _ = np.histogram2d(high_concentration, high_lwc, bins=[x_bins, y_bins])

avg_rwc_high = np.divide(sum_rwc_high, counts_high, out=np.full_like(sum_rwc_high, np.nan), where=counts_high > 0)
avg_lwc_high = np.divide(sum_lwc_high, counts_high, out=np.full_like(sum_lwc_high, np.nan), where=counts_high > 0)
rwc_lwc_ratio_high = np.divide(avg_rwc_high, avg_lwc_high, out=np.full_like(avg_rwc_high, np.nan), where=avg_lwc_high > 0) * 100
masked_rwc_high = np.ma.masked_where(np.isnan(rwc_lwc_ratio_high), rwc_lwc_ratio_high)
sum_rwc_low, _, _ = np.histogram2d(low_concentration, low_lwc, bins=[x_bins, y_bins], weights=low_rwc)
sum_lwc_low, _, _ = np.histogram2d(low_concentration, low_lwc, bins=[x_bins, y_bins], weights=low_lwc)
counts_low, _, _ = np.histogram2d(low_concentration, low_lwc, bins=[x_bins, y_bins])
counts_cdp_high_conc = counts_high.copy()
counts_cdp_low_conc  = counts_low.copy()
avg_rwc_low = np.divide(sum_rwc_low, counts_low, out=np.full_like(sum_rwc_low, np.nan), where=counts_low > 0)
avg_lwc_low = np.divide(sum_lwc_low, counts_low, out=np.full_like(sum_lwc_low, np.nan), where=counts_low > 0)
rwc_lwc_ratio_low = np.divide(avg_rwc_low, avg_lwc_low, out=np.full_like(avg_rwc_low, np.nan), where=avg_lwc_low > 0) * 100
masked_rwc_low = np.ma.masked_where(np.isnan(rwc_lwc_ratio_low), rwc_lwc_ratio_low)
plt.figure(figsize=(8, 6))
norm = mcolors.Normalize(vmin=1, vmax=100)
plt.pcolormesh(xedges, yedges, masked_rwc_high.T, cmap="RdBu_r", norm=norm, shading='auto')
plt.colorbar(label="RWC / LWC (%)")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc /cm³', fontsize=19, fontweight='bold')
plt.ylabel('LWC g/m³', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('High GCCN Flights CDP January-June 2022', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
plt.pcolormesh(xedges, yedges, masked_rwc_low.T, cmap="RdBu_r", norm=norm, shading='auto')
plt.colorbar(label="RWC / LWC (%)")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc /cm³', fontsize=19, fontweight='bold')
plt.ylabel('LWC g/m³', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Low GCCN Flights CDP January-June 2022', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()

#%%
gray_mask_high = np.isnan(rwc_lwc_ratio_high)
gray_values_high = np.full_like(rwc_lwc_ratio_high, np.nan)
gray_values_high[gray_mask_high] = 1 
gray_mask_low = np.isnan(rwc_lwc_ratio_low)
gray_values_low = np.full_like(rwc_lwc_ratio_low, np.nan)
gray_values_low[gray_mask_low] = 1 
plt.figure(figsize=(8, 6))
norm = mcolors.Normalize(vmin=1, vmax=100)
img = plt.pcolormesh(xedges, yedges, masked_rwc_high.T, cmap="RdBu_r", norm=norm, shading='auto')
plt.pcolormesh(xedges, yedges, gray_values_high.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)
cbar = plt.colorbar(img)
cbar.set_label("RWC / LWC (%)", fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc /cm³', fontsize=16, fontweight='bold')
plt.ylabel('LWC g/m³', fontsize=16, fontweight='bold')
plt.title('High GCCN CDP Flights January-June 2022', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(xedges, yedges, masked_rwc_low.T, cmap="RdBu_r", norm=norm, shading='auto')
plt.pcolormesh(xedges, yedges, gray_values_low.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)
cbar = plt.colorbar(img)
cbar.set_label("RWC / LWC (%)", fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc /cm³', fontsize=16, fontweight='bold')
plt.ylabel('LWC g/m³', fontsize=16, fontweight='bold')
plt.title('Low GCCN CDP Flights January-June 2022', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
#separating rwc and lwc 
masked_avg_rwc_high = np.ma.masked_where(np.isnan(avg_rwc_high), avg_rwc_high)
masked_avg_rwc_low = np.ma.masked_where(np.isnan(avg_rwc_low), avg_rwc_low)
vmin = 0
vmax = 1
plt.figure(figsize=(8, 6))
plt.pcolormesh(xedges, yedges, masked_avg_rwc_high.T, cmap="viridis", shading='auto', vmin=vmin, vmax=vmax)
plt.colorbar(label="Mean RWC (g m$^{-3}$)")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Mean RWC — High GCCN CDP Flights \nJanuary–June 2022)', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
plt.pcolormesh(xedges, yedges, masked_avg_rwc_low.T, cmap="viridis", shading='auto', vmin=vmin, vmax=vmax)
plt.colorbar(label="Mean RWC (g m$^{-3}$)")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Mean RWC — Low GCCN CDP Flights January–June 2022', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
#fixing gray NANs
masked_avg_rwc_high = np.ma.masked_where(np.isnan(avg_rwc_high), avg_rwc_high)
masked_avg_rwc_low = np.ma.masked_where(np.isnan(avg_rwc_low), avg_rwc_low)
vmin = 0
vmax = 1
cmap = plt.cm.viridis.copy()
cmap.set_bad(color='gray')  
plt.figure(figsize=(8, 6))
img_high = plt.pcolormesh(
    xedges, yedges, masked_avg_rwc_high.T,
    cmap=cmap, shading='auto', vmin=vmin, vmax=vmax
)
cbar_high = plt.colorbar(img_high)
cbar_high.set_label("Mean RWC (g m$^{-3}$)", fontsize=20, fontweight='bold')
cbar_high.ax.tick_params(labelsize=20, width=2, length=5)
for t in cbar_high.ax.get_yticklabels():
    t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Mean RWC\nHigh CDP GCCN Flights\nJanuary–June 2022', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
img_low = plt.pcolormesh(
    xedges, yedges, masked_avg_rwc_low.T,
    cmap=cmap, shading='auto', vmin=vmin, vmax=vmax
)
cbar_low = plt.colorbar(img_low)
cbar_low.set_label("Mean RWC (g m$^{-3}$)", fontsize=20, fontweight='bold')
cbar_low.ax.tick_params(labelsize=20, width=2, length=5)
for t in cbar_low.ax.get_yticklabels():
    t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Mean RWC\nLow GCCN CDP Flights\nJanuary–June 2022', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
masked_avg_lwc_high = np.ma.masked_where(np.isnan(avg_lwc_high), avg_lwc_high)
plt.figure(figsize=(8, 6))
plt.pcolormesh(xedges, yedges, masked_avg_lwc_high.T, cmap="plasma", shading='auto')
plt.colorbar(label="Mean LWC (g m$^{-3}$)")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Mean LWC — High GCCN CDP Flights January–June 2022', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
masked_avg_lwc_low = np.ma.masked_where(np.isnan(avg_lwc_low), avg_lwc_low)
plt.figure(figsize=(8, 6))
plt.pcolormesh(xedges, yedges, masked_avg_lwc_low.T, cmap="plasma", shading='auto')
plt.colorbar(label="Mean LWC (g m$^{-3}$)")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Mean LWC — Low GCCN CDP Flights January–June 2022', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
#creating our own color bar for RWC
valid_data_high = avg_rwc_high[~np.isnan(avg_rwc_high)].flatten()
valid_data_low = avg_rwc_low[~np.isnan(avg_rwc_low)].flatten()
all_valid_rwc = np.concatenate([valid_data_high, valid_data_low])
bounds = [0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8]
#for 2x2 binning scheme
# bounds = [0, 0.005, 0.01, 0.01, 0.02, 0.028, 0.03, 0.038, 0.05]
cmap = plt.cm.viridis.copy()
cmap.set_bad(color='gray') 
norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N, extend='neither')
masked_avg_rwc_high = np.ma.masked_where(np.isnan(avg_rwc_high), avg_rwc_high)
masked_avg_rwc_low = np.ma.masked_where(np.isnan(avg_rwc_low), avg_rwc_low)
plt.figure(figsize=(8, 6))
img_high = plt.pcolormesh(
    xedges, yedges, masked_avg_rwc_high.T,
    cmap=cmap, norm=norm, shading='auto'
)
cbar_high = plt.colorbar(img_high, ticks=bounds)
cbar_high.set_label("Mean RWC (g m$^{-3}$)", fontsize=20, fontweight='bold')
cbar_high.ax.tick_params(labelsize=20, width=2, length=5)
for t in cbar_high.ax.get_yticklabels():
    t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Mean RWC\nHigh CDP GCCN Flights\nJanuary–June 2022', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
img_low = plt.pcolormesh(
    xedges, yedges, masked_avg_rwc_low.T,
    cmap=cmap, norm=norm, shading='auto'
)
cbar_low = plt.colorbar(img_low, ticks=bounds)
cbar_low.set_label("Mean RWC (g m$^{-3}$)", fontsize=20, fontweight='bold')
cbar_low.ax.tick_params(labelsize=20, width=2, length=5)
for t in cbar_low.ax.get_yticklabels():
    t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Mean RWC\nLow CDP GCCN Flights\nJanuary–June 2022', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
#%%
# Creating custom colormap and bin bounds for RWC
valid_data_high = avg_rwc_high[~np.isnan(avg_rwc_high)].flatten()
valid_data_low = avg_rwc_low[~np.isnan(avg_rwc_low)].flatten()
all_valid_rwc = np.concatenate([valid_data_high, valid_data_low])
bounds = [0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8]

cmap = plt.cm.viridis.copy()
cmap.set_bad(color='gray') 
norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N, extend='neither')

masked_avg_rwc_high = np.ma.masked_where(np.isnan(avg_rwc_high), avg_rwc_high)
masked_avg_rwc_low = np.ma.masked_where(np.isnan(avg_rwc_low), avg_rwc_low)
x_centers = 0.5 * (xedges[:-1] + xedges[1:])
y_centers = 0.5 * (yedges[:-1] + yedges[1:])
plt.figure(figsize=(8, 6))
img_high = plt.pcolormesh(
    xedges, yedges, masked_avg_rwc_high.T,
    cmap=cmap, norm=norm, shading='auto'
)
cbar_high = plt.colorbar(img_high, ticks=bounds)
cbar_high.set_label("Mean RWC (g m$^{-3}$)", fontsize=20, fontweight='bold')
cbar_high.ax.tick_params(labelsize=20, width=2, length=5)
for t in cbar_high.ax.get_yticklabels():
    t.set_fontweight('bold')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Mean RWC\nHigh CDP GCCN Flights\nJanuary–June 2022', fontsize=19, fontweight='bold')
for i, xc in enumerate(x_centers):
    for j, yc in enumerate(y_centers):
        count = len(rwc_bins_high[i][j]) 
        if count > 0:
            plt.text(xc, yc, str(count),
                     ha='center', va='center',
                     color='black', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
img_low = plt.pcolormesh(
    xedges, yedges, masked_avg_rwc_low.T,
    cmap=cmap, norm=norm, shading='auto'
)
cbar_low = plt.colorbar(img_low, ticks=bounds)
cbar_low.set_label("Mean RWC (g m$^{-3}$)", fontsize=20, fontweight='bold')
cbar_low.ax.tick_params(labelsize=20, width=2, length=5)
for t in cbar_low.ax.get_yticklabels():
    t.set_fontweight('bold')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Mean RWC\nLow CDP GCCN Flights\nJanuary–June 2022', fontsize=19, fontweight='bold')
for i, xc in enumerate(x_centers):
    for j, yc in enumerate(y_centers):
        count = len(rwc_bins_low[i][j]) 
        if count > 0:
            plt.text(xc, yc, str(count),
                     ha='center', va='center',
                     color='black', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()
#%%
#creating this as a two panel figure with one color bar 

#%%
#creating our own colorbar but for LWC 
masked_avg_lwc_high = np.ma.masked_where(np.isnan(avg_lwc_high), avg_lwc_high)
masked_avg_lwc_low = np.ma.masked_where(np.isnan(avg_lwc_low), avg_lwc_low)
valid_data_high = avg_lwc_high[~np.isnan(avg_lwc_high)].flatten()
valid_data_low = avg_lwc_low[~np.isnan(avg_lwc_low)].flatten()
all_valid_lwc = np.concatenate([valid_data_high, valid_data_low])
bounds = [0, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0] 
#bounds for 2x2 binning scheme 
# bounds = [0, 0.04, 0.06, 0.07, 0.09, 0.1, 0.15, 0.18, 0.2, 0.25, 0.3, 0.35, 0.38, 0.4] 
cmap = plt.cm.plasma.copy()
cmap.set_bad(color='gray')
norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N, extend='neither')
plt.figure(figsize=(8, 6))
img_high = plt.pcolormesh(
    xedges, yedges, masked_avg_lwc_high.T,
    cmap=cmap, norm=norm, shading='auto'
)
cbar_high = plt.colorbar(img_high, ticks=bounds)
cbar_high.set_label("Mean LWC (g m$^{-3}$)", fontsize=20, fontweight='bold')
cbar_high.ax.tick_params(labelsize=20, width=2, length=5)
for t in cbar_high.ax.get_yticklabels():
    t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Mean LWC\nHigh CDP GCCN Flights\nJanuary–June 2022', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
img_low = plt.pcolormesh(
    xedges, yedges, masked_avg_lwc_low.T,
    cmap=cmap, norm=norm, shading='auto'
)
cbar_low = plt.colorbar(img_low, ticks=bounds)
cbar_low.set_label("Mean LWC (g m$^{-3}$)", fontsize=20, fontweight='bold')
cbar_low.ax.tick_params(labelsize=20, width=2, length=5)
for t in cbar_low.ax.get_yticklabels():
    t.set_fontweight('bold')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Mean LWC\nLow GCCN CDP Flights\nJanuary–June 2022', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
#using ratio of high lwc/low lwc and high rwc/low rwc 
ratio_rwc = np.divide(
    avg_rwc_high,
    avg_rwc_low,
    out=np.full_like(avg_rwc_high, np.nan),
    where=avg_rwc_low > 0
)
ratio_lwc = np.divide(
    avg_lwc_high,
    avg_lwc_low,
    out=np.full_like(avg_lwc_high, np.nan),
    where=avg_lwc_low > 0
)
masked_ratio_rwc = np.ma.masked_where(np.isnan(ratio_rwc), ratio_rwc)
masked_ratio_lwc = np.ma.masked_where(np.isnan(ratio_lwc), ratio_lwc)
norm = mcolors.Normalize(vmin=0, vmax=2)
plt.figure(figsize=(8,6))
img = plt.pcolormesh(
    xedges, yedges,
    masked_ratio_rwc.T,
    cmap="RdBu_r",
    norm=norm,
    shading="auto"
)
cbar = plt.colorbar(img)
cbar.set_label("RWC Ratio (High / Low)", fontsize=18, fontweight="bold")
cbar.ax.tick_params(labelsize=16)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"Nr+Nc (cm$^{-3}$)", fontsize=19, fontweight="bold")
plt.ylabel(r"LWC (g m$^{-3}$)", fontsize=19, fontweight="bold")
plt.title("CDP RWC Ratio — High / Low GCCN Flights", fontsize=19, fontweight="bold")
plt.tight_layout()
plt.show()
plt.figure(figsize=(8,6))
img = plt.pcolormesh(
    xedges, yedges,
    masked_ratio_lwc.T,
    cmap="RdBu_r",
    norm=norm,
    shading="auto"
)
cbar = plt.colorbar(img)
cbar.set_label("CDP LWC Ratio (High / Low)", fontsize=18, fontweight="bold")
cbar.ax.tick_params(labelsize=16)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"Nr+Nc (cm$^{-3}$)", fontsize=19, fontweight="bold")
plt.ylabel(r"LWC (g m$^{-3}$)", fontsize=19, fontweight="bold")
plt.title("CDP Ratio — High / Low GCCN Flights", fontsize=19, fontweight="bold")
plt.tight_layout()
plt.show()
#%%
masked_avg_rwc_low = np.ma.masked_where(np.isnan(avg_rwc_low), avg_rwc_low)
masked_avg_lwc_low = np.ma.masked_where(np.isnan(avg_lwc_low), avg_lwc_low)
norm = mcolors.Normalize(vmin=0, vmax=2)
cmap_rwc = plt.cm.viridis.copy()
cmap_rwc.set_bad(color='gray')
cmap_lwc = plt.cm.plasma.copy()
cmap_lwc.set_bad(color='gray')
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(
    xedges, yedges,
    masked_avg_rwc_low.T,
    cmap=cmap_rwc,
    shading='auto',
    norm=norm 
)
cbar = plt.colorbar(img)
cbar.set_label("Mean RWC (g m$^{-3}$)", fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=19, width=2, length=5)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.title('CDP Low GCCN Flights — Mean RWC', fontsize=19, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(
    xedges, yedges,
    masked_avg_lwc_low.T,
    cmap=cmap_lwc,
    shading='auto',
    norm=norm 
)
cbar = plt.colorbar(img)
cbar.set_label("Mean LWC (g m$^{-3}$)", fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=19, width=2, length=5)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.title('CDP Low GCCN Flights — Mean LWC', fontsize=19, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.tight_layout()
plt.show()
#%%
#trying to fix color scale 
ratio_rwc = np.divide(
    avg_rwc_high,
    avg_rwc_low,
    out=np.full_like(avg_rwc_high, np.nan),
    where=avg_rwc_low > 0
)
masked_ratio_rwc = np.ma.masked_where(np.isnan(ratio_rwc), ratio_rwc)
# custom_bounds = [1.2, 1.4, 1.6, 1.7, 2.0, 2.2, 2.4, 3.5, 3.8, 4.0, 4.3, 7.0]
custom_bounds = [0, 0.2, 0.5, 1.0, 1.1, 1.2, 1.3, 1.4, 2.0, 2.2, 2.4, 2.7, 3.0, 3.5, 3.6, 7, 8.5]
cmap = plt.cm.viridis.copy()
cmap.set_bad(color='gray')
norm = BoundaryNorm(boundaries=custom_bounds, ncolors=cmap.N)
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(
    xedges, yedges,
    masked_ratio_rwc.T,
    cmap=cmap,
    norm=norm,
    shading="auto"
)
cbar = plt.colorbar(img, ticks=custom_bounds)
cbar.set_label("RWC Ratio (High / Low)", fontsize=19, fontweight="bold")
cbar.ax.tick_params(labelsize=19)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
plt.xscale("log")
plt.yscale("log")
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.xlabel(r"Nr+Nc (cm$^{-3}$)", fontsize=19, fontweight="bold")
plt.ylabel(r"LWC (g m$^{-3}$)", fontsize=19, fontweight="bold")
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.title("CDP (in cloud)\nRWC Ratio High / Low GCCN Flights\nJanuary–June 2022", fontsize=19, fontweight="bold")
plt.tight_layout()
plt.show()
#%%
#masking bins with less than 100 samples
num_bins = 5
x_bins = np.logspace(np.log10(min(np.concatenate([high_concentration, low_concentration]))),
                     np.log10(max(np.concatenate([high_concentration, low_concentration]))),
                     num_bins)
y_bins = np.logspace(np.log10(min(np.concatenate([high_lwc, low_lwc]))),
                     np.log10(max(np.concatenate([high_lwc, low_lwc]))),
                     num_bins)
xedges, yedges = x_bins, y_bins
def make_empty_bins():
    return [[[] for _ in range(len(y_bins)-1)] for _ in range(len(x_bins)-1)]

rwc_bins_high = make_empty_bins()
rwc_bins_low  = make_empty_bins()
lwc_bins_high = make_empty_bins()
lwc_bins_low  = make_empty_bins()
for conc, lwc, rwc in zip(high_concentration, high_lwc, high_rwc):
    i = np.searchsorted(x_bins, conc, side='right') - 1
    j = np.searchsorted(y_bins, lwc,  side='right') - 1
    if 0 <= i < len(x_bins)-1 and 0 <= j < len(y_bins)-1 and not np.isnan(rwc) and not np.isnan(lwc):
        rwc_bins_high[i][j].append(rwc)
        lwc_bins_high[i][j].append(lwc)

for conc, lwc, rwc in zip(low_concentration, low_lwc, low_rwc):
    i = np.searchsorted(x_bins, conc, side='right') - 1
    j = np.searchsorted(y_bins, lwc,  side='right') - 1
    if 0 <= i < len(x_bins)-1 and 0 <= j < len(y_bins)-1 and not np.isnan(rwc) and not np.isnan(lwc):
        rwc_bins_low[i][j].append(rwc)
        lwc_bins_low[i][j].append(lwc)
avg_rwc_high = np.full((len(x_bins)-1, len(y_bins)-1), np.nan)
avg_rwc_low  = np.full((len(x_bins)-1, len(y_bins)-1), np.nan)
avg_lwc_high = np.full((len(x_bins)-1, len(y_bins)-1), np.nan)
avg_lwc_low  = np.full((len(x_bins)-1, len(y_bins)-1), np.nan)

for i in range(len(x_bins)-1):
    for j in range(len(y_bins)-1):
        if rwc_bins_high[i][j]:
            avg_rwc_high[i, j] = np.mean(rwc_bins_high[i][j])
        if rwc_bins_low[i][j]:
            avg_rwc_low[i, j]  = np.mean(rwc_bins_low[i][j])
        if lwc_bins_high[i][j]:
            avg_lwc_high[i, j] = np.mean(lwc_bins_high[i][j])
        if lwc_bins_low[i][j]:
            avg_lwc_low[i, j]  = np.mean(lwc_bins_low[i][j])
min_samples = 100
valid_bins = np.full((len(x_bins)-1, len(y_bins)-1), False)
for i in range(len(x_bins)-1):
    for j in range(len(y_bins)-1):
        total = len(rwc_bins_high[i][j]) + len(rwc_bins_low[i][j])
        if total >= min_samples:
            valid_bins[i, j] = True

avg_rwc_high[~valid_bins] = np.nan
avg_rwc_low[~valid_bins]  = np.nan
avg_lwc_high[~valid_bins] = np.nan
avg_lwc_low[~valid_bins]  = np.nan
ratio_rwc = np.divide(
    avg_rwc_high,
    avg_rwc_low,
    out=np.full_like(avg_rwc_high, np.nan),
    where=avg_rwc_low > 0
)
ratio_lwc = np.divide(
    avg_lwc_high,
    avg_lwc_low,
    out=np.full_like(avg_lwc_high, np.nan),
    where=avg_lwc_low > 0
)

masked_ratio_rwc = np.ma.masked_where(np.isnan(ratio_rwc), ratio_rwc)
masked_ratio_lwc = np.ma.masked_where(np.isnan(ratio_lwc), ratio_lwc)
norm = mcolors.Normalize(vmin=0, vmax=2)
plt.figure(figsize=(8,6))
img = plt.pcolormesh(
    xedges, yedges,
    masked_ratio_rwc.T,
    cmap="RdBu_r",
    norm=norm,
    shading="auto"
)
cbar = plt.colorbar(img)
cbar.set_label("RWC Ratio (High / Low)", fontsize=18, fontweight="bold")
cbar.ax.tick_params(labelsize=16)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"Nr+Nc (cm$^{-3}$)", fontsize=19, fontweight="bold")
plt.ylabel(r"LWC (g m$^{-3}$)", fontsize=19, fontweight="bold")
plt.title("CDP RWC Ratio — High / Low GCCN Mass Flights", fontsize=19, fontweight="bold")
plt.tight_layout()
plt.show()
plt.figure(figsize=(8,6))
img = plt.pcolormesh(
    xedges, yedges,
    masked_ratio_lwc.T,
    cmap="RdBu_r",
    norm=norm,
    shading="auto"
)
cbar = plt.colorbar(img)
cbar.set_label("CDP LWC Ratio (High / Low)", fontsize=18, fontweight="bold")
cbar.ax.tick_params(labelsize=16)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"Nr+Nc (cm$^{-3}$)", fontsize=19, fontweight="bold")
plt.ylabel(r"LWC (g m$^{-3}$)", fontsize=19, fontweight="bold")
plt.title("LWC Ratio — High / Low GCCN Flights", fontsize=19, fontweight="bold")
plt.tight_layout()
plt.show()
#%%

custom_bounds = [0, 0.2, 0.5, 1.0, 1.1, 1.2, 1.3, 1.4, 2.0, 2.2, 2.4, 2.7, 3.0, 3.5, 3.6, 7, 8.5]
cmap = plt.cm.viridis.copy()
cmap.set_bad(color='gray')
norm = BoundaryNorm(boundaries=custom_bounds, ncolors=cmap.N)

plt.figure(figsize=(8, 6))
img = plt.pcolormesh(
    xedges, yedges,
    masked_ratio_rwc.T,
    cmap=cmap,
    norm=norm,
    shading="auto"
)

cbar = plt.colorbar(img, ticks=custom_bounds)
cbar.set_label("RWC Ratio (High / Low)", fontsize=19, fontweight="bold")
cbar.ax.tick_params(labelsize=19)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
plt.xscale("log")
plt.yscale("log")
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.xlabel(r"Nr+Nc (cm$^{-3}$)", fontsize=19, fontweight="bold")
plt.ylabel(r"LWC (g m$^{-3}$)", fontsize=19, fontweight="bold")
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.title("CDP (in cloud)\nRWC Ratio High / Low GCCN Flights\nJanuary–June 2022", fontsize=19, fontweight="bold")
plt.tight_layout()
plt.show()
#%%
# Merge concentration and liquid water lists into a unified dataset
combined_dataset = []

for conc_entry in total_combined_concentration:
    date = conc_entry['Date']
    time = conc_entry['Time']
    matching_lwc = next((e for e in total_liquid_water 
                         if e['Date'] == date and e['Time'] == time), None)

    if matching_lwc:
        combined_dataset.append({
            'Date': date,
            'Time': time,
            'Leg_start': conc_entry['Leg_start'],
            'Leg_stop': conc_entry['Leg_stop'],
            'Total_Combined_Concentration': conc_entry['Total_Combined_Concentration'],
            'Rain_Concentration': conc_entry['Rain_Concentration'],
            'Total_Liquid_Water': matching_lwc['Total_Liquid_Water']
        })

#%%
#trying histograms for ratio of high/low rwc 
x_min = np.nanmin([entry['Total_Combined_Concentration'] for entry in combined_dataset])
x_max = np.nanmax([entry['Total_Combined_Concentration'] for entry in combined_dataset])
y_min = np.nanmin([entry['Total_Liquid_Water'] for entry in combined_dataset])
y_max = np.nanmax([entry['Total_Liquid_Water'] for entry in combined_dataset])
num_bins = 5
x_bins = np.logspace(np.log10(x_min), np.log10(x_max), num_bins)
y_bins = np.logspace(np.log10(y_min), np.log10(y_max), num_bins)
n_bootstrap = 10000
confidence_level = 0.90
lower_percentile = (1 - confidence_level) / 2 * 100
upper_percentile = (1 + confidence_level) / 2 * 100
def group_by_flight(data):
    flights = defaultdict(list)
    for entry in data:
        flights[entry['Date']].append(entry)
    return flights
min_samples = 100  # require at least this many raw points per bin before bootstrapping

def compute_flight_bin_means_RWC(flight_data):
    bin_means = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    bin_counts = np.zeros((len(x_bins) - 1, len(y_bins) - 1), dtype=int)

    for flight in flight_data.values():
        conc = np.array([e['Total_Combined_Concentration'] for e in flight])
        lwc = np.array([e['Total_Liquid_Water'] for e in flight])
        rwc = np.array([e['Rain_Concentration'] for e in flight])

        for i in range(len(x_bins) - 1):
            for j in range(len(y_bins) - 1):
                mask = (conc >= x_bins[i]) & (conc < x_bins[i + 1]) & \
                       (lwc >= y_bins[j]) & (lwc < y_bins[j + 1])
                if np.any(mask):
                    vals = rwc[mask]
                    vals = vals[~np.isnan(vals)]
                    if len(vals) > 0:
                        bin_means[i][j].extend(vals.tolist())
                        bin_counts[i, j] += len(vals)

    return bin_means, bin_counts


def bootstrap_ratio_distributions(bin_high, bin_low, counts_high, counts_low, min_samples=100):
    boot_ratios = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            if (counts_high[i, j] + counts_low[i, j]) >= min_samples:
                high_vals = bin_high[i][j]
                low_vals = bin_low[i][j]
                if len(high_vals) > 1 and len(low_vals) > 1:
                    boot_sample_ratios = []
                    for _ in range(n_bootstrap):
                        sampled_high = np.random.choice(high_vals, len(high_vals), replace=True)
                        sampled_low = np.random.choice(low_vals, len(low_vals), replace=True)
                        sampled_low = np.where(sampled_low == 0, np.nan, sampled_low)
                        mean_low = np.nanmean(sampled_low)
                        if mean_low > 0 and not np.isnan(mean_low):
                            ratio = np.nanmean(sampled_high) / mean_low
                            boot_sample_ratios.append(ratio)
                    boot_ratios[i][j] = np.array(boot_sample_ratios)
    return boot_ratios


def plot_histograms_with_percentage_ratio(boot_dists):
    fig, axes = plt.subplots(
        nrows=len(x_bins) - 1,
        ncols=len(y_bins) - 1,
        figsize=(14, 10),
        sharex=True,
        sharey=True
    )
    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            ax = axes[i][j]
            dist = boot_dists[i][j]
            if len(dist) > 0:
                clipped_ratio = np.clip(dist, 0, 10)
                bin_edges = np.arange(0, 10, 1)

                ax.hist(clipped_ratio, bins=bin_edges, color='skyblue', edgecolor='black')

                ax.axvline(1, color='red', linestyle='--')
                lower = np.percentile(dist, lower_percentile)
                upper = np.percentile(dist, upper_percentile)
                ax.axvline(lower, color='black', linestyle=':', linewidth=1)
                ax.axvline(upper, color='black', linestyle=':', linewidth=1)

                percent_above_one = np.sum(dist > 1) / len(dist) * 100
                mean_val = np.nanmean(dist)
                std_val = np.nanstd(dist)
                annotation = f"{percent_above_one:.1f}% > 1\nμ = {mean_val:.2f}, σ = {std_val:.2f}"
                ax.text(0.98, 0.95, annotation, transform=ax.transAxes,
                        ha='right', va='top', fontsize=14,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

                ax.set_xlim(0, 10)
                ax.tick_params(axis='both', labelsize=18)
            else:
                ax.set_visible(False)

   
    fig.suptitle("CDP (in cloud)\nBootstrapped RWC Ratio (High/Low GCCN)\n January–June 2022", fontsize=20, fontweight='bold')
    fig.supxlabel("RWC Ratio (High / Low)", fontsize=20, fontweight='bold')
    fig.supylabel("Count", fontsize=20, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    plt.show()
gccn_values = np.array(list(average_gccn_per_flight.values()))
threshold = np.percentile(gccn_values, 50)
high_dates = {date for date, val in average_gccn_per_flight.items() if val >= threshold}
low_dates = {date for date, val in average_gccn_per_flight.items() if val < threshold}
high_data = [entry for entry in combined_dataset if entry['Date'] in high_dates]
low_data = [entry for entry in combined_dataset if entry['Date'] in low_dates]
grouped_high = group_by_flight(high_data)
grouped_low = group_by_flight(low_data)
bin_means_high, counts_high = compute_flight_bin_means_RWC(grouped_high)
bin_means_low, counts_low   = compute_flight_bin_means_RWC(grouped_low)

boot_ratio_distributions = bootstrap_ratio_distributions(
    bin_means_high, bin_means_low, counts_high, counts_low, min_samples=100
)

plot_histograms_with_percentage_ratio(boot_ratio_distributions)
#%%
#fixing ranges 
conc_all = np.array([e['Total_Combined_Concentration'] for e in combined_dataset])
lwc_all  = np.array([e['Total_Liquid_Water'] for e in combined_dataset])
x_min = np.nanmin(conc_all[conc_all > 0])
x_max = np.nanmax(conc_all)
y_min = np.nanmin(lwc_all[lwc_all > 0])
y_max = np.nanmax(lwc_all)

print("Concentration range (Nr+Nc):", x_min, x_max)
print("LWC range:", y_min, y_max)

num_bins = 5
x_bins = np.logspace(np.log10(x_min), np.log10(x_max), num_bins)
y_bins = np.logspace(np.log10(y_min), np.log10(y_max), num_bins)

n_bootstrap = 10000
confidence_level = 0.90
lower_percentile = (1 - confidence_level) / 2 * 100
upper_percentile = (1 + confidence_level) / 2 * 100
min_samples = 100 

def group_by_flight(data):
    flights = defaultdict(list)
    for entry in data:
        flights[entry['Date']].append(entry)
    return flights

def compute_flight_bin_means_RWC(flight_data):
    bin_means = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    bin_counts = np.zeros((len(x_bins) - 1, len(y_bins) - 1), dtype=int)

    for flight in flight_data.values():
        conc = np.array([e['Total_Combined_Concentration'] for e in flight])
        lwc = np.array([e['Total_Liquid_Water'] for e in flight])
        rwc = np.array([e['Rain_Concentration'] for e in flight])

        for i in range(len(x_bins) - 1):
            for j in range(len(y_bins) - 1):
                mask = (conc >= x_bins[i]) & (conc < x_bins[i + 1]) & \
                       (lwc >= y_bins[j]) & (lwc < y_bins[j + 1])
                if np.any(mask):
                    vals = rwc[mask]
                    vals = vals[~np.isnan(vals)]
                    if len(vals) > 0:
                        bin_means[i][j].extend(vals.tolist())
                        bin_counts[i, j] += len(vals)

    return bin_means, bin_counts

def bootstrap_ratio_distributions(bin_high, bin_low, counts_high, counts_low, min_samples=100):
    boot_ratios = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            if (counts_high[i, j] + counts_low[i, j]) >= min_samples:
                high_vals = bin_high[i][j]
                low_vals = bin_low[i][j]
                if len(high_vals) > 1 and len(low_vals) > 1:
                    boot_sample_ratios = []
                    for _ in range(n_bootstrap):
                        sampled_high = np.random.choice(high_vals, len(high_vals), replace=True)
                        sampled_low = np.random.choice(low_vals, len(low_vals), replace=True)

                        mean_high = np.nanmean(sampled_high)
                        mean_low = np.nanmean(sampled_low)

                        if np.isfinite(mean_high) and np.isfinite(mean_low) and mean_low > 0:
                            ratio = mean_high / mean_low
                            boot_sample_ratios.append(ratio)
                        else:
                            boot_sample_ratios.append(np.nan)  # optional: could skip instead

                    boot_ratios[i][j] = np.array(boot_sample_ratios)
    return boot_ratios

def plot_histograms_with_percentage_ratio(boot_dists):
    fig, axes = plt.subplots(
        nrows=len(x_bins) - 1,
        ncols=len(y_bins) - 1,
        figsize=(14, 10),
        sharex=True,
        sharey=True
    )
    ratio_min, ratio_max = np.inf, -np.inf
    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            if len(boot_dists[i][j]) > 0:
                ratio_min = min(ratio_min, np.nanmin(boot_dists[i][j]))
                ratio_max = max(ratio_max, np.nanmax(boot_dists[i][j]))
    ratio_min = max(0, ratio_min)
    ratio_max = min(10, ratio_max)

    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            ax = axes[i][j]
            dist = boot_dists[i][j]
            if len(dist) > 0:
                ax.hist(dist, bins=30, color='skyblue', edgecolor='black')

                ax.axvline(1, color='red', linestyle='--')
                lower = np.percentile(dist, lower_percentile)
                upper = np.percentile(dist, upper_percentile)
                ax.axvline(lower, color='black', linestyle=':', linewidth=1)
                ax.axvline(upper, color='black', linestyle=':', linewidth=1)

                percent_above_one = np.sum(dist > 1) / len(dist) * 100
                mean_val = np.nanmean(dist)
                std_val = np.nanstd(dist)
                annotation = f"{percent_above_one:.1f}% > 1\nμ = {mean_val:.2f}, σ = {std_val:.2f}"
                ax.text(0.98, 0.95, annotation, transform=ax.transAxes,
                        ha='right', va='top', fontsize=12,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

                ax.set_xlim(ratio_min, ratio_max)
                ax.tick_params(axis='both', labelsize=10)
            else:
                ax.set_visible(False)

    fig.suptitle("CAS (in cloud)\nBootstrapped RWC Ratio (High/Low GCCN)\nJanuary–June 2022",
                 fontsize=18, fontweight='bold')
    fig.supxlabel("RWC Ratio (High / Low)", fontsize=16, fontweight='bold')
    fig.supylabel("Count", fontsize=16, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    plt.show()
gccn_values = np.array(list(average_gccn_per_flight.values()))
threshold = np.percentile(gccn_values, 50)
high_dates = {date for date, val in average_gccn_per_flight.items() if val >= threshold}
low_dates  = {date for date, val in average_gccn_per_flight.items() if val < threshold}

high_data = [entry for entry in combined_dataset if entry['Date'] in high_dates]
low_data  = [entry for entry in combined_dataset if entry['Date'] in low_dates]

grouped_high = group_by_flight(high_data)
grouped_low  = group_by_flight(low_data)

bin_means_high, counts_high = compute_flight_bin_means_RWC(grouped_high)
bin_means_low,  counts_low  = compute_flight_bin_means_RWC(grouped_low)

boot_ratio_distributions = bootstrap_ratio_distributions(
    bin_means_high, bin_means_low, counts_high, counts_low, min_samples=100
)

plot_histograms_with_percentage_ratio(boot_ratio_distributions)

# %%
#boostrapped heatmap
heatmap_data = np.full((len(x_bins) - 1, len(y_bins) - 1), np.nan)
min_rwc_threshold = 0.005          
for i in range(len(x_bins) - 1): 
    for j in range(len(y_bins) - 1): 
        dist = boot_ratio_distributions[j][i]
        if len(dist) > 0:
            dist = dist[np.isfinite(dist)]
            dist = dist[dist <= ratio_cap]
            if len(dist) > 0:
                heatmap_data[j][i] = np.nanmean(dist)  
masked_ratio_rwc = np.ma.masked_where(np.isnan(heatmap_data), heatmap_data)
custom_bounds = [0.1, 0.2, 0.4, 0.7, 1.0, 1.1, 1.4, 1.6, 1.8, 2.0, 2.5]
cmap = plt.cm.viridis.copy()
cmap.set_bad(color='gray')
norm = BoundaryNorm(boundaries=custom_bounds, ncolors=cmap.N)
fig, ax = plt.subplots(figsize=(8, 6))
img = ax.pcolormesh(x_bins, y_bins, masked_ratio_rwc.T,
                    cmap=cmap, norm=norm, shading="auto")
cbar = plt.colorbar(img, ticks=custom_bounds)
cbar.set_label("Bootstrapped RWC Ratio (High / Low)", fontsize=19, fontweight="bold")
cbar.ax.tick_params(labelsize=19)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Nr+Nc (cm$^{-3}$)", fontsize=19, fontweight="bold")
ax.set_ylabel(r"LWC (g m$^{-3}$)", fontsize=19, fontweight="bold")
ax.set_title("CDP (in cloud)\nRWC Ratio High / Low GCCN Flights\nJanuary–June 2022",
             fontsize=19, fontweight="bold")
ax.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
ax.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
for i in range(len(y_bins) - 1):
    for j in range(len(x_bins) - 1):
        dist = boot_ratio_distributions[i][j]
        
        if (counts_high[j][i] + counts_low[j][i]) >= min_samples:
            
            if len(dist) > 0:
                dist = dist[np.isfinite(dist)]
                dist = dist[dist <= ratio_cap]

                if len(dist) == 0:
                    continue

                percent_above = np.sum(dist > 1) / len(dist) * 100
                mean_val = np.nanmean(dist)
                std_val = np.nanstd(dist)

                label = f"{percent_above:.1f}% > 1\nμ={mean_val:.2f}, σ={std_val:.2f}"
                x_center = 10 ** ((np.log10(x_bins[j]) + np.log10(x_bins[j + 1])) / 2)
                y_center = 10 ** ((np.log10(y_bins[i]) + np.log10(y_bins[i + 1])) / 2)

                ax.text(
                    x_center, y_center, label,
                    ha='center', va='center',
                    fontsize=9, fontweight='bold', linespacing=1.1
                )

plt.tight_layout()
plt.show()
#%%
heatmap_data = np.full((len(x_bins) - 1, len(y_bins) - 1), np.nan)
min_samples = 100
for i in range(len(x_bins) - 1): 
    for j in range(len(y_bins) - 1): 
        dist = boot_ratio_distributions[j][i]
        if len(dist) > 0:
            dist = dist[np.isfinite(dist)]
            dist = dist[dist <= ratio_cap]
            if len(dist) > 0:
                heatmap_data[j][i] = np.nanmean(dist)  

masked_ratio_rwc = np.ma.masked_where(np.isnan(heatmap_data), heatmap_data)
custom_bounds = [0.1, 0.2, 0.4, 0.7, 1.0, 1.1, 1.4, 1.6, 1.8, 2.0, 2.5]
cmap = plt.cm.viridis.copy()
cmap.set_bad(color='gray')
norm = BoundaryNorm(boundaries=custom_bounds, ncolors=cmap.N)

fig, ax = plt.subplots(figsize=(8, 6))
img = ax.pcolormesh(x_bins, y_bins, masked_ratio_rwc.T,
                    cmap=cmap, norm=norm, shading="auto")

cbar = plt.colorbar(img, ticks=custom_bounds)
cbar.set_label("Bootstrapped RWC Ratio (High / Low)", fontsize=19, fontweight="bold")
cbar.ax.tick_params(labelsize=19)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Nr+Nc (cm$^{-3}$)", fontsize=19, fontweight="bold")
ax.set_ylabel(r"LWC (g m$^{-3}$)", fontsize=19, fontweight="bold")
ax.set_title("CDP (in cloud)\nRWC Ratio High / Low GCCN Flights\nJanuary–June 2022",
             fontsize=19, fontweight="bold")
ax.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
ax.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
for i in range(len(x_bins) - 1):
    for j in range(len(y_bins) - 1):
        dist = boot_ratio_distributions[j][i]
        if (counts_high[j][i] + counts_low[j][i]) >= min_samples:
            dist = dist[np.isfinite(dist)]
            dist = dist[dist <= ratio_cap]

            if len(dist) == 0:
                continue

            percent_above = np.sum(dist > 1) / len(dist) * 100
            mean_val = np.nanmean(dist)
            std_val = np.nanstd(dist)
            if mean_val < 0.75 or mean_val > 2.5:
                print(f"⚠️  Suspicious bin at Nr+Nc bin {i}, LWC bin {j} → μ = {mean_val:.2f}, σ = {std_val:.2f}, {percent_above:.1f}% > 1")

            label = f"{percent_above:.1f}% > 1\nμ={mean_val:.2f}, σ={std_val:.2f}"
            x_center = 10 ** ((np.log10(x_bins[i]) + np.log10(x_bins[i + 1])) / 2)
            y_center = 10 ** ((np.log10(y_bins[j]) + np.log10(y_bins[j + 1])) / 2)

            ax.text(
                x_center, y_center, label,
                ha='center', va='center',
                fontsize=9, fontweight='bold', linespacing=1.1
            )

plt.tight_layout()
plt.show()

# %%
#making sure we only use bins greater than 100 samples and adding uncertainty
heatmap_data = np.full((len(x_bins) - 1, len(y_bins) - 1), np.nan)
min_rwc_threshold = 0.005 
ratio_cap = 5.0     
min_samples = 100            
for i in range(len(x_bins) - 1): 
    for j in range(len(y_bins) - 1): 
        dist = boot_ratio_distributions[i][j]
        if len(dist) >= min_samples:
            dist = dist[np.isfinite(dist)]
            dist = dist[dist <= ratio_cap]
            if len(dist) > 0:
                heatmap_data[i][j] = np.nanmean(dist)  
masked_ratio_rwc = np.ma.masked_where(np.isnan(heatmap_data), heatmap_data)
custom_bounds = [0, 0.2, 0.5, 1.0, 1.1, 1.2, 1.3, 1.4, 2.0, 2.2, 2.4, 2.7, 3.0, 3.5, 3.6, 7, 9]
cmap = plt.cm.viridis.copy()
cmap.set_bad(color='gray')
norm = BoundaryNorm(boundaries=custom_bounds, ncolors=cmap.N)
fig, ax = plt.subplots(figsize=(8, 6))
img = ax.pcolormesh(
    x_bins, y_bins, masked_ratio_rwc.T,
    cmap=cmap, norm=norm, shading="auto"
)
cbar = plt.colorbar(img, ticks=custom_bounds)
cbar.set_label("Bootstrapped RWC Ratio (High / Low)", fontsize=19, fontweight="bold")
cbar.ax.tick_params(labelsize=19)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Nr+Nc (cm$^{-3}$)", fontsize=19, fontweight="bold")
ax.set_ylabel(r"LWC (g m$^{-3}$)", fontsize=19, fontweight="bold")
ax.set_title("CDP (in cloud)\nRWC Ratio High / Low GCCN Flights\nJanuary–June 2022",
             fontsize=19, fontweight="bold")
ax.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
ax.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
for i in range(len(y_bins) - 1):
    for j in range(len(x_bins) - 1):
        dist = boot_ratio_distributions[j][i]
        if len(dist) >= min_samples:
            dist = dist[np.isfinite(dist)]
            dist = dist[dist <= ratio_cap]
            if len(dist) == 0:
                continue
            percent_above = np.sum(dist > 1) / len(dist) * 100
            mean_val = np.nanmean(dist)
            std_val = np.nanstd(dist)
            sem_val = std_val / np.sqrt(len(dist)) 
            ci_lower = np.percentile(dist, lower_percentile)
            ci_upper = np.percentile(dist, upper_percentile)

            label = (f"{percent_above:.1f}% > 1\n"
                     f"μ={mean_val:.2f} ± {sem_val:.2f} (SEM)\n"
                     f"90% CI [{ci_lower:.2f}, {ci_upper:.2f}]")
            x_center = 10 ** ((np.log10(x_bins[j]) + np.log10(x_bins[j + 1])) / 2)
            y_center = 10 ** ((np.log10(y_bins[i]) + np.log10(y_bins[i + 1])) / 2)

            ax.text(
                x_center, y_center, label,
                ha='center', va='center',
                fontsize=7, fontweight='bold', linespacing=1.2
            )

plt.tight_layout()
plt.show()
#%%
heatmap_data = np.full((len(x_bins) - 1, len(y_bins) - 1), np.nan)
min_samples = 100


for i in range(len(x_bins) - 1): 
    for j in range(len(y_bins) - 1): 
        dist = boot_ratio_distributions[i][j]
        if len(dist) >= min_samples:
            dist = dist[np.isfinite(dist)]
            dist = dist[dist <= ratio_cap]
            if len(dist) > 0:
                heatmap_data[i][j] = np.nanmean(dist)

masked_ratio_rwc = np.ma.masked_where(np.isnan(heatmap_data), heatmap_data)
custom_bounds = [0, 0.2, 0.5, 1.0, 1.1, 1.2, 1.3, 1.4, 2.0, 2.2, 2.4, 2.7, 3.0, 3.5, 3.6, 7, 9]
cmap = plt.cm.viridis.copy()
cmap.set_bad(color='gray')
norm = BoundaryNorm(boundaries=custom_bounds, ncolors=cmap.N)

fig, ax = plt.subplots(figsize=(8, 6))
img = ax.pcolormesh(
    x_bins, y_bins, masked_ratio_rwc.T,
    cmap=cmap, norm=norm, shading="auto"
)

cbar = plt.colorbar(img, ticks=custom_bounds)
cbar.set_label("Bootstrapped RWC Ratio (High / Low)", fontsize=19, fontweight="bold")
cbar.ax.tick_params(labelsize=19)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Nr+Nc (cm$^{-3}$)", fontsize=19, fontweight="bold")
ax.set_ylabel(r"LWC (g m$^{-3}$)", fontsize=19, fontweight="bold")
ax.set_title("CDP (in cloud)\nRWC Ratio High / Low GCCN Flights\nJanuary–June 2022",
             fontsize=19, fontweight="bold")
ax.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
ax.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
for i in range(len(x_bins) - 1):
    for j in range(len(y_bins) - 1):
        dist = boot_ratio_distributions[i][j]
        if len(dist) >= min_samples:
            dist = dist[np.isfinite(dist)]
            dist = dist[dist <= ratio_cap]
            if len(dist) == 0:
                continue

            percent_above = np.sum(dist > 1) / len(dist) * 100
            mean_val = np.nanmean(dist)
            std_val = np.nanstd(dist)
            sem_val = std_val / np.sqrt(len(dist))
            ci_lower = np.percentile(dist, lower_percentile)
            ci_upper = np.percentile(dist, upper_percentile)

            label = (f"{percent_above:.1f}% > 1\n"
                     f"μ={mean_val:.2f} ± {sem_val:.2f} (SEM)\n"
                     f"90% CI [{ci_lower:.2f}, {ci_upper:.2f}]")

            x_center = 10 ** ((np.log10(x_bins[i]) + np.log10(x_bins[i + 1])) / 2)
            y_center = 10 ** ((np.log10(y_bins[j]) + np.log10(y_bins[j + 1])) / 2)

            ax.text(
                x_center, y_center, label,
                ha='center', va='center',
                fontsize=7, fontweight='bold', linespacing=1.2
            )
plt.tight_layout()
plt.show()
#%%
n_bootstrap = 10000
confidence_level = 0.90
lower_percentile = (1 - confidence_level) / 2 * 100
upper_percentile = (1 + confidence_level) / 2 * 100
valid_bins = np.full((len(x_bins)-1, len(y_bins)-1), False)
for i in range(len(x_bins)-1):
    for j in range(len(y_bins)-1):
        total = len(rwc_bins_high[i][j]) + len(rwc_bins_low[i][j])
        if total >= 100:
            valid_bins[i, j] = True
boot_ratio_distributions = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
heatmap_data = np.full((len(x_bins) - 1, len(y_bins) - 1), np.nan)

for i in range(len(x_bins) - 1):
    for j in range(len(y_bins) - 1):
        if valid_bins[i, j]:
            high_vals = np.array(rwc_bins_high[i][j])
            low_vals  = np.array(rwc_bins_low[i][j])
            
            if len(high_vals) > 0 and len(low_vals) > 0:
                ratios = []
                for _ in range(n_bootstrap):
                    sample_high = np.random.choice(high_vals, size=len(high_vals), replace=True)
                    sample_low  = np.random.choice(low_vals,  size=len(low_vals),  replace=True)

                    mean_high = np.mean(sample_high)
                    mean_low  = np.mean(sample_low)

                    if mean_low > 0:
                        ratios.append(mean_high / mean_low)

                boot_ratio_distributions[i][j] = ratios
                heatmap_data[i][j] = np.nanmean(ratios)

masked_ratio_rwc = np.ma.masked_where(np.isnan(heatmap_data), heatmap_data)
custom_bounds = [0, 0.2, 0.5, 1.0, 1.1, 1.2, 1.3, 1.4, 2.0, 2.2, 2.4, 2.7, 3.0, 3.5, 3.6, 7, 9]
cmap = plt.cm.viridis.copy()
cmap.set_bad(color='gray')
norm = BoundaryNorm(boundaries=custom_bounds, ncolors=cmap.N)

fig, ax = plt.subplots(figsize=(8, 6))
img = ax.pcolormesh(
    x_bins, y_bins, masked_ratio_rwc.T,
    cmap=cmap, norm=norm, shading="auto"
)

cbar = plt.colorbar(img, ticks=custom_bounds)
cbar.set_label("Bootstrapped RWC Ratio (High / Low)", fontsize=19, fontweight="bold")
cbar.ax.tick_params(labelsize=19)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Nr+Nc (cm$^{-3}$)", fontsize=19, fontweight="bold")
ax.set_ylabel(r"LWC (g m$^{-3}$)", fontsize=19, fontweight="bold")
ax.set_title("CDP (in cloud)\nRWC Ratio High / Low GCCN Flights\nJanuary–June 2022",
             fontsize=19, fontweight="bold")
ax.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
ax.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
for i in range(len(x_bins) - 1):
    for j in range(len(y_bins) - 1):
        dist = boot_ratio_distributions[i][j]
        if valid_bins[i][j] and len(dist) > 0:
            percent_above = np.sum(np.array(dist) > 1) / len(dist) * 100
            mean_val = np.nanmean(dist)
            std_val = np.nanstd(dist)
            sem_val = std_val / np.sqrt(len(dist))
            ci_lower = np.percentile(dist, lower_percentile)
            ci_upper = np.percentile(dist, upper_percentile)

            label = (f"{percent_above:.1f}% > 1\n"
                     f"μ={mean_val:.2f} ± {sem_val:.2f} (SEM)\n"
                     f"90% CI [{ci_lower:.2f}, {ci_upper:.2f}]")

            x_center = 10 ** ((np.log10(x_bins[i]) + np.log10(x_bins[i + 1])) / 2)
            y_center = 10 ** ((np.log10(y_bins[j]) + np.log10(y_bins[j + 1])) / 2)

            ax.text(
                x_center, y_center, label,
                ha='center', va='center',
                fontsize=7, fontweight='bold', linespacing=1.2
            )

plt.tight_layout()
plt.show()
#%%
i_check = 2  # replace with the actual bin index for x
j_check = 1  # replace with the actual bin index for y

high_vals = rwc_bins_high[i_check][j_check]
low_vals = rwc_bins_low[i_check][j_check]

print(f"Bin ({i_check}, {j_check}):")
print(f"High RWC values ({len(high_vals)}): {high_vals}")
print(f"Low  RWC values ({len(low_vals)}): {low_vals}")

if len(high_vals) > 0 and len(low_vals) > 0:
    ratio_of_means = np.mean(high_vals) / np.mean(low_vals)
    print(f"→ Ratio of means: {ratio_of_means:.2f}")
else:
    print("→ Not enough data for ratio of means")

# %%
#using median concentration 
n_bootstrap = 10000
confidence_level = 0.90
lower_percentile = (1 - confidence_level) / 2 * 100
upper_percentile = (1 + confidence_level) / 2 * 100

valid_bins = np.full((len(x_bins)-1, len(y_bins)-1), False)
for i in range(len(x_bins)-1):
    for j in range(len(y_bins)-1):
        total = len(rwc_bins_high[i][j]) + len(rwc_bins_low[i][j])
        if total >= 100:
            valid_bins[i, j] = True

boot_ratio_distributions = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
heatmap_data = np.full((len(x_bins) - 1, len(y_bins) - 1), np.nan)
for i in range(len(x_bins) - 1):
    for j in range(len(y_bins) - 1):
        if valid_bins[i, j]:
            high_vals = np.array(rwc_bins_high[i][j])
            low_vals  = np.array(rwc_bins_low[i][j])

            if len(high_vals) > 0 and len(low_vals) > 0:
                ratios = []
                for _ in range(n_bootstrap):
                    sample_high = np.random.choice(high_vals, size=len(high_vals), replace=True)
                    sample_low  = np.random.choice(low_vals,  size=len(low_vals),  replace=True)

                    mean_high = np.mean(sample_high)
                    mean_low  = np.mean(sample_low)

                    if mean_low > 0:
                        ratios.append(mean_high / mean_low)

                boot_ratio_distributions[i][j] = ratios
                heatmap_data[i][j] = np.nanmean(ratios)

masked_ratio_rwc = ma.masked_where(np.isnan(heatmap_data), heatmap_data)

custom_bounds = [1.0, 1.1, 1.2, 1.3, 1.4, 2.0, 2.2, 2.4, 2.7,
                 3.0, 3.5, 3.6, 7, 9]
cmap = plt.cm.viridis.copy()
cmap.set_bad(color='lightgray') 
norm = BoundaryNorm(boundaries=custom_bounds, ncolors=cmap.N)

fig, ax = plt.subplots(figsize=(8, 6))
img = ax.pcolormesh(
    x_bins, y_bins, masked_ratio_rwc.T,
    cmap=cmap, norm=norm, shading="auto"
)
for i in range(len(x_bins)-1):
    for j in range(len(y_bins)-1):
        val = heatmap_data[i, j]
        if np.isfinite(val) and val < 1:
            ax.add_patch(Rectangle(
                (x_bins[i], y_bins[j]),                     
                x_bins[i+1]-x_bins[i],                      
                y_bins[j+1]-y_bins[j],                      
                facecolor='dimgray', edgecolor='none', zorder=3
            ))
cbar = plt.colorbar(img, ticks=custom_bounds)
cbar.set_label("Bootstrapped RWC Ratio (High / Low)", fontsize=19, fontweight="bold")
cbar.ax.tick_params(labelsize=19)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Nr+Nc (cm$^{-3}$)", fontsize=19, fontweight="bold")
ax.set_ylabel(r"LWC (g m$^{-3}$)", fontsize=19, fontweight="bold")
ax.set_title("CAS (in cloud)\nRWC Ratio High / Low GCCN Flights\nJanuary–June 2022",
             fontsize=19, fontweight="bold")
ax.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
ax.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')

plt.tight_layout()
plt.show()

# %%
masked_ratio_rwc = ma.masked_where(np.isnan(heatmap_data), heatmap_data)

custom_bounds = [1.0, 1.1, 1.2, 1.3, 1.4,
                 2.0, 2.2, 2.4, 2.7, 3.0, 3.5, 3.6, 7, 9]
cmap = plt.cm.viridis.copy()
cmap.set_bad(color='lightgray')      
norm = BoundaryNorm(boundaries=custom_bounds, ncolors=cmap.N)

fig, ax = plt.subplots(figsize=(8, 6))
img = ax.pcolormesh(x_bins, y_bins, masked_ratio_rwc.T,
                    cmap=cmap, norm=norm, shading="auto")
for i in range(len(x_bins)-1):
    for j in range(len(y_bins)-1):
        dist = boot_ratio_distributions[i][j]
        if valid_bins[i, j] and len(dist) > 0:
            dist = np.asarray(dist)
            mean_val   = np.nanmean(dist)
            median_val = np.nanmedian(dist)
            std_val    = np.nanstd(dist)
            sem_val    = std_val / np.sqrt(len(dist))
            ci_lower   = np.percentile(dist, lower_percentile)
            ci_upper   = np.percentile(dist, upper_percentile)
            pct_above1 = np.sum(dist > 1) / len(dist) * 100

            if mean_val < 1:
                ax.add_patch(Rectangle(
                    (x_bins[i], y_bins[j]),
                    x_bins[i+1] - x_bins[i],
                    y_bins[j+1] - y_bins[j],
                    facecolor='dimgray', edgecolor='none',
                    zorder=3  
                ))

            label = (f"{pct_above1:.1f}% > 1\n"
                     f"μ={mean_val:.2f} ± {sem_val:.2f} (SEM)\n"
                     f"median={median_val:.2f}\n"
                     f"90% CI [{ci_lower:.2f}, {ci_upper:.2f}]")
            x_center = 10 ** ((np.log10(x_bins[i]) + np.log10(x_bins[i+1])) / 2)
            y_center = 10 ** ((np.log10(y_bins[j]) + np.log10(y_bins[j+1])) / 2)

            ax.text(x_center, y_center, label,
                    ha='center', va='center',
                    fontsize=7, fontweight='bold',
                    linespacing=1.2, zorder=4)
cbar = plt.colorbar(img, ticks=custom_bounds)
cbar.set_label("Bootstrapped RWC Ratio (High / Low)",
               fontsize=17, fontweight="bold")
cbar.ax.tick_params(labelsize=19)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Nr+Nc (cm$^{-3}$)", fontsize=19, fontweight="bold")
ax.set_ylabel(r"LWC (g m$^{-3}$)", fontsize=19, fontweight="bold")
ax.set_title("CDP (in cloud)\nRWC Ratio High / Low GCCN Concentration Flights\nJanuary–June 2022",
             fontsize=18, fontweight="bold")
ax.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
ax.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
#saving CDP
heatmap_data_CDP = heatmap_data.copy()
valid_bins_CDP = valid_bins.copy()
boot_ratio_distributions_CDP = boot_ratio_distributions
x_bins_CDP = np.array(x_bins, copy=True)
y_bins_CDP = np.array(y_bins, copy=True)

# %%
#plotting CAS and CDP as a 2 panel paper figure 
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
axL, axR = axes
masked_ratio_rwc_CAS = ma.masked_where(np.isnan(heatmap_data_CAS), heatmap_data_CAS)

img = axL.pcolormesh(
    x_bins_CAS, y_bins_CAS, masked_ratio_rwc_CAS.T,
    cmap=cmap, norm=norm, shading="auto"
)

for i in range(len(x_bins_CAS) - 1):
    for j in range(len(y_bins_CAS) - 1):
        dist = boot_ratio_distributions_CAS[i][j]
        if valid_bins_CAS[i, j] and len(dist) > 0:
            dist = np.asarray(dist)

            mean_val   = np.nanmean(dist)
            median_val = np.nanmedian(dist)
            std_val    = np.nanstd(dist)
            sem_val    = std_val / np.sqrt(len(dist))
            ci_lower   = np.percentile(dist, lower_percentile)
            ci_upper   = np.percentile(dist, upper_percentile)
            pct_above1 = np.sum(dist > 1) / len(dist) * 100

            if mean_val < 1:
                axL.add_patch(Rectangle(
                    (x_bins_CAS[i], y_bins_CAS[j]),
                    x_bins_CAS[i+1] - x_bins_CAS[i],
                    y_bins_CAS[j+1] - y_bins_CAS[j],
                    facecolor='dimgray', edgecolor='none', zorder=3
                ))

            label = (f"{pct_above1:.1f}% > 1\n"
                     f"μ={mean_val:.2f} ± {sem_val:.2f}\n"
                     f"med={median_val:.2f}\n"
                     f"[{ci_lower:.2f}, {ci_upper:.2f}]")

            x_center = 10 ** ((np.log10(x_bins_CAS[i]) + np.log10(x_bins_CAS[i+1])) / 2)
            y_center = 10 ** ((np.log10(y_bins_CAS[j]) + np.log10(y_bins_CAS[j+1])) / 2)

            axL.text(x_center, y_center, label,
                     ha='center', va='center',
                     fontsize=11, fontweight='bold',
                     linespacing=1.2, zorder=4)

axL.set_xscale("log")
axL.set_yscale("log")
axL.set_xlabel(r"Nr+Nc (cm$^{-3}$)", fontsize=15, fontweight="bold")
axL.set_ylabel(r"LWC (g m$^{-3}$)", fontsize=15, fontweight="bold")
axL.set_title("CAS GCCN Concentration",
              fontsize=15, fontweight="bold")
axL.tick_params(axis='both', which='major', labelsize=15, width=3, length=8)
axL.tick_params(axis='both', which='minor', width=2, length=5)
plt.setp(axL.get_xticklabels(), fontweight='bold')
plt.setp(axL.get_yticklabels(), fontweight='bold')
masked_ratio_rwc_CDP = ma.masked_where(np.isnan(heatmap_data_CDP), heatmap_data_CDP)

img2 = axR.pcolormesh(
    x_bins_CDP, y_bins_CDP, masked_ratio_rwc_CDP.T,
    cmap=cmap, norm=norm, shading="auto"
)

for i in range(len(x_bins_CDP) - 1):
    for j in range(len(y_bins_CDP) - 1):
        dist = boot_ratio_distributions_CDP[i][j]
        if valid_bins_CDP[i, j] and len(dist) > 0:
            dist = np.asarray(dist)

            mean_val   = np.nanmean(dist)
            median_val = np.nanmedian(dist)
            std_val    = np.nanstd(dist)
            sem_val    = std_val / np.sqrt(len(dist))
            ci_lower   = np.percentile(dist, lower_percentile)
            ci_upper   = np.percentile(dist, upper_percentile)
            pct_above1 = np.sum(dist > 1) / len(dist) * 100

            if mean_val < 1:
                axR.add_patch(Rectangle(
                    (x_bins_CDP[i], y_bins_CDP[j]),
                    x_bins_CDP[i+1] - x_bins_CDP[i],
                    y_bins_CDP[j+1] - y_bins_CDP[j],
                    facecolor='dimgray', edgecolor='none', zorder=3
                ))

            label = (f"{pct_above1:.1f}% > 1\n"
                     f"μ={mean_val:.2f} ± {sem_val:.2f}\n"
                     f"med={median_val:.2f}\n"
                     f"[{ci_lower:.2f}, {ci_upper:.2f}]")

            x_center = 10 ** ((np.log10(x_bins_CDP[i]) + np.log10(x_bins_CDP[i+1])) / 2)
            y_center = 10 ** ((np.log10(y_bins_CDP[j]) + np.log10(y_bins_CDP[j+1])) / 2)

            axR.text(x_center, y_center, label,
                     ha='center', va='center',
                     fontsize=11, fontweight='bold',
                     linespacing=1.2, zorder=4)

axR.set_xscale("log")
axR.set_yscale("log")
axR.set_xlabel(r"Nr+Nc (cm$^{-3}$)", fontsize=15, fontweight="bold")
axR.set_title("CDP GCCN Concentration",
              fontsize=15, fontweight="bold")
axR.tick_params(axis='both', which='major', labelsize=15, width=3, length=8)
axR.tick_params(axis='both', which='minor', width=2, length=5)
plt.setp(axR.get_xticklabels(), fontweight='bold')
plt.setp(axR.get_yticklabels(), fontweight='bold')
axL.set_ylim(y_bins_CAS.min(), y_bins_CAS.max())
axR.set_ylim(y_bins_CDP.min(), y_bins_CDP.max())
fig.subplots_adjust(right=0.86, wspace=0.14)
cax = fig.add_axes([0.88, 0.12, 0.02, 0.76])  # [left, bottom, width, height] in figure coords
cbar = fig.colorbar(img, cax=cax, ticks=custom_bounds)
cbar.set_label("Bootstrapped RWC (High/Low)", fontsize=15, fontweight="bold")
cbar.ax.tick_params(labelsize=15)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")

plt.show()
#save figure as pdf
fig.savefig("RWC_Ratio_CAS_CDP_Concentration.pdf", dpi=300, bbox_inches='tight')
# %%
#combined count figure 
plt.figure(figsize=(8, 6))
blank = np.zeros((len(xedges)-1, len(yedges)-1))
plt.pcolormesh(
    xedges, yedges, blank.T,
    cmap=mcolors.ListedColormap(["white"]),
    shading="auto",
    edgecolors="black",
    linewidth=0.8
)
x_centers = 0.5 * (xedges[:-1] + xedges[1:])
y_centers = 0.5 * (yedges[:-1] + yedges[1:])
for i in range(len(x_centers)):
    for j in range(len(y_centers)):

        cas_h = int(counts_cas_high_conc[i, j])
        cdp_h = int(counts_cdp_high_conc[i, j])
        cas_l = int(counts_cas_low_conc[i, j])
        cdp_l = int(counts_cdp_low_conc[i, j])

        if (cas_h + cdp_h + cas_l + cdp_l) > 0:

            label = (
                f"CAS H: {cas_h}\n"
                f"CDP H: {cdp_h}\n"
                f"CAS L: {cas_l}\n"
                f"CDP L: {cdp_l}"
            )

            plt.text(
                x_centers[i], y_centers[j], label,
                ha="center", va="center",
                fontsize=10,
                fontweight="bold",
                color="black",
                linespacing=1.05
            )

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Nr+Nc (cm$^{-3}$)", fontsize=15, fontweight="bold")
plt.ylabel("LWC (g m$^{-3}$)", fontsize=15, fontweight="bold")
plt.xticks(fontsize=15, fontweight="bold")
plt.yticks(fontsize=15, fontweight="bold")

plt.title(
    "Concentration Bin Counts",
    fontsize=15,
    fontweight="bold"
)
plt.tight_layout()
plt.show()
# %%
