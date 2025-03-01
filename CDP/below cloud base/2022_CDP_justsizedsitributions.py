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

# Loop through each date
for date in dates_CDP:
    dataset = {'Date': date, 'Clear Means': [], 'Cloud Means': []}  # Initialize a dataset dictionary

    # Construct the filename for the current date
    file_path = f'/home/disk/eos4/kathem24/activate/data/CDP/2022/csv/CDP_1Hz_files/CDP_1Hz_{date}.csv'

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found for date {date}: {file_path}")
        continue

    # Read the CSV file for the current date
    df_CDP = pd.read_csv(file_path)

    # Perform your operations on df_CDP
    # Example: Print the first few rows to ensure the file is read correctly
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
    print(df_CDP.head())  # Print first few rows
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
        reverse=False  # Ensure L1 flights come before L2 flights
    )

    print(f"Processing {date}... Found files: {file_paths}")

    run = 1
    dfs_for_date = []

    for file_path in file_paths:
        # Detect the correct header row dynamically without explicity skipping rows
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
            # Read the file using the identified header row
            df_2DS = pd.read_csv(
                file_path, 
                skiprows=header_row, 
                quoting=csv.QUOTE_NONE,
                engine='python'
            )

            # Clean up column names
            df_2DS.columns = df_2DS.columns.str.strip('"')
            print(f"Columns for {file_path}: {df_2DS.columns[:10]}")

            # Replace -9999 values with 0
            df_2DS.replace([-9999, -9999.0], 0, inplace=True)
            for col in df_2DS.select_dtypes(include=['object']).columns:
                df_2DS[col] = df_2DS[col].str.strip('"')

            # Append the cleaned DataFrame to the list
            dfs_for_date.append(df_2DS)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")


    # Combine DataFrames for this date, if there are multiple files
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

# Final check
print(f"Total dates processed: {len(twoDS)}")
# %%
#Import humidity data. 
#This is necessary when converting from ambient distributions to dry distributions. 
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
    # print(fname_h20)
    for file_path in fname_h20:
        # print(f"Reading file: {file_path}")
        df_h20 = pd.read_csv(file_path, skiprows=36, quoting=csv.QUOTE_NONE)

        
        # print(f"Raw data from {file_path}:")
        # print(df_h20.head(10))  # Print more rows to see more of the data

        # Strip quotes from column names and trim any whitespace
        df_h20.columns = df_h20.columns.str.strip().str.replace('"', '')

        
        # print(f"Cleaned column names: {df_h20.columns}")

        # Ensure each column is treated as a string, then strip quotes and convert to numeric
        for col_ in col_name_h20:
            if col_ in df_h20.columns:
                df_h20[col_] = df_h20[col_].astype(str).str.strip().str.replace('"', '')
                df_h20[col_] = pd.to_numeric(df_h20[col_], errors='coerce')
                df_h20.replace([-9999, -9999.00], np.NaN, inplace=True)

       
        # print(f"Processed data from {file_path}:")
        # print(df_h20.head(10))  # Print more rows to see more of the data
        frames.append(df_h20)
    if len(frames) > 1:
        df_h20_combined = pd.concat(frames, ignore_index=True)
        # print(f"Combined {len(frames)} files for date {date}")

    else:
        df_h20_combined = frames[0]
        # print(f"Only one file found for date {date}")
    h20.append(df_h20_combined)
    # print(df_h20_combined.head(10))  
    # print(df_h20_combined.tail(10))
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
    # print(date)
    # print(fname_sum)

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
            if df_sum[col].dtype == 'O':  # 'O' stands for Object (usually string columns)
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
            # print("YESSSSSSSSSSSSSSSSSSSSSSSSSSSSs")
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
##Checking where or loc instead of align
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

    # Convert Time_Start to numeric
    CDP_flight['Time_Start'] = pd.to_numeric(CDP_flight['Time_Start'], errors='coerce')
    twoDS_flight['Time_Start'] = pd.to_numeric(twoDS_flight['Time_Start'], errors='coerce')

    # Extract required columns
    CDP_times = CDP_flight['Time_Start'].values
    CDP_lwc = CDP_flight['LWC_CDP'].values
    CDP_bins = {f'CDP_Bin{bin_label:02d}': CDP_flight[f'CDP_Bin{bin_label:02d}'].values for bin_label in range(30)}

    TwoDS_times = twoDS_flight['Time_Start'].values
    TwoDS_N_total = twoDS_flight['N-total_2DS'].values

    total_BCB_means_CDP = []

    for k in range(len(BCB_start)):
        start20 = BCB_start[k]
        end20 = BCB_stop[k]

        # Find indices in the BCB range for CDP and TwoDS
        CDP_indices_in_range = np.where((CDP_times >= start20) & (CDP_times <= end20))[0]
        TwoDS_indices_in_range = np.where((TwoDS_times >= start20) & (TwoDS_times <= end20))[0]

        if CDP_indices_in_range.size > 0 and TwoDS_indices_in_range.size > 0:
            data_labels_CDP = []
            BCB_means_CDP = []

            for CDP_idx, TwoDS_idx in zip(CDP_indices_in_range, TwoDS_indices_in_range):
                lwc_val = CDP_lwc[CDP_idx]
                N_val = TwoDS_N_total[TwoDS_idx]

                # Assign labels
                lwc_label = 'Y' if 0 <= lwc_val <= 0.0025 else 'N'
                N_label = 'Y' if 0 <= N_val <= 100 else 'N'
                label = 'Y' if lwc_label == 'Y' and N_label == 'Y' else 'N'
                data_labels_CDP.append(label)

                # Collect bin values
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

# Print leg_info or use it as needed
for leg in leg_info_CDP:
    print(f"Date: {leg['Date']}, Start: {leg['BCB_start']}, Stop: {leg['BCB_stop']}, Data Labels: {leg['Data_Labels']}")
#%%
#Double check the number of legs associated with each date to compare with the CAS legs 

# Count the number of legs for each date
leg_count = Counter([leg['Date'] for leg in leg_info_CDP])

# Print the count for each date
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

    # Convert times to numeric arrays for filtering
    CDP_times = np.array(CDP_flight['Time_Start'], dtype=float)
    TwoDS_times = np.array(twoDS_flight['Time_Start'], dtype=float)

    lwc = np.array(CDP_flight['LWC_CDP'], dtype=float)
    N_total = np.array(twoDS_flight['N-total_2DS'], dtype=float)

    # Pre-fetch all bin values
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

        # Use np.where to find indices within the BCB interval
        CDP_indices_in_range = np.where((CDP_times >= start20) & (CDP_times <= end20))[0]
        TwoDS_indices_in_range = np.where((TwoDS_times >= start20) & (TwoDS_times <= end20))[0]

        if CDP_indices_in_range.size > 0 and TwoDS_indices_in_range.size > 0:
            for cdp_idx, twods_idx in zip(CDP_indices_in_range, TwoDS_indices_in_range):
                lwc_val = lwc[cdp_idx]
                N_val = N_total[twods_idx]
                lwc_label = 'Y' if 0 <= lwc_val <= 0.0025 else 'N'
                N_label = 'Y' if 0 <= N_val <= 100 else 'N'
                label = 'Y' if lwc_label == 'Y' and N_label == 'Y' else 'N'

                for bin_label in range(30):
                    bin_key = f'Bin{bin_label:02d}_{label}_mean'
                    bin_means_CDP[bin_key].append(bins[f'CDP_Bin{bin_label:02d}'][cdp_idx])

        # Calculate means
        for bin_label in range(30):
            for label in ['Y', 'N']:
                bin_key = f'Bin{bin_label:02d}_{label}_mean'
                if bin_means_CDP[bin_key]:
                    bin_means_CDP[bin_key] = np.nanmean(bin_means_CDP[bin_key])

        total_BCB_means_CDP.append(bin_means_CDP)

    master_CDP_BCB.append(total_BCB_means_CDP)

# Print or use master_CDP_BCB as needed
for item in master_CDP_BCB:
    for bin_means_CDP in item:
        print(f"Date: {bin_means_CDP['Date']}, Start: {bin_means_CDP['BCB_start']}, Stop: {bin_means_CDP['BCB_stop']}")
        for bin_label in range(30):
            for label in ['Y', 'N']:
                bin_key = f'Bin{bin_label:02d}_{label}_mean'
                print(f"   {bin_key}: {bin_means_CDP[bin_key]}")
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
if 'Y_BCB_calc' in globals():
    # Create a set of unique (Date, BCB_start, BCB_stop) identifiers from Y_BCB_calc
    BCB_leg_keys = {(entry['Date'], entry['BCB_start'], entry['BCB_stop']) for entry in Y_BCB_calc}

    # Filter Y_CDP_calc and N_CDP_calc to retain only legs present in Y_BCB_calc
    filtered_Y_CDP_calc = [entry for entry in Y_CDP_calc if (entry['Date'], entry['BCB_start'], entry['BCB_stop']) in BCB_leg_keys]
    filtered_N_CDP_calc = [entry for entry in N_CDP_calc if (entry['Date'], entry['BCB_start'], entry['BCB_stop']) in BCB_leg_keys]

    # Print verification
    print(f"Original Y_CDP_calc legs: {len(Y_CDP_calc)}")
    print(f"Filtered Y_CDP_calc legs: {len(filtered_Y_CDP_calc)}")
    print(f"Original N_CDP_calc legs: {len(N_CDP_calc)}")
    print(f"Filtered N_CDP_calc legs: {len(filtered_N_CDP_calc)}")

    # Assign back the filtered lists if needed
    Y_CDP_calc = filtered_Y_CDP_calc
    N_CDP_calc = filtered_N_CDP_calc
else:
    print("Error: Y_BCB_calc is not defined. Make sure it is loaded before running this script.")


# %%
#ambient size distribution
bin_center_CDP=np.array(bin_center_CDP)

# Plot Raw CDP Size Distributions
plt.figure(figsize=(8, 6))

for entry in Y_CDP_calc:
    bin_means = np.array([entry.get(f'Bin{i:02d}_Y_mean', np.nan) for i in range(0, 30)], dtype=float)  
    valid_indices = ~np.isnan(bin_means)  
    bin_centers_valid = np.array(bin_center_CDP)[valid_indices]
    bin_means_valid = bin_means[valid_indices]

    plt.plot(bin_centers_valid, bin_means_valid, color='black', alpha=0.5)

# Labels and formatting
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

    # Mask: Remove NaNs and zero values
    valid_indices = (bin_means > 0) & ~np.isnan(bin_means)  
    bin_centers_valid = bin_centers[valid_indices]
    bin_means_valid = bin_means[valid_indices]

    if len(bin_centers_valid) > 0:  # Only plot if valid data exists
        plt.plot(bin_centers_valid, bin_means_valid, color='black', alpha=0.5)

# Labels and formatting
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
# Initialize arrays for averaging for Y_CDP_calc
sum_bin_means_CDP = np.zeros(len(bin_center_CDP))
count_bin_means_CDP = np.zeros(len(bin_center_CDP))

# Iterate through all size distributions in Y_CDP_calc
for entry in Y_CDP_calc:
    bin_means_CDP = np.array([entry.get(f'Bin{i:02d}_Y_mean', np.nan) for i in range(0, 30)], dtype=float)

    # Mask: Remove NaNs and zero values
    valid_indices = (bin_means_CDP > 0) & ~np.isnan(bin_means_CDP)

    sum_bin_means_CDP[valid_indices] += bin_means_CDP[valid_indices]
    count_bin_means_CDP[valid_indices] += 1

average_bin_means_CDP = np.divide(sum_bin_means_CDP, count_bin_means_CDP, where=count_bin_means_CDP > 0)

plt.figure(figsize=(8, 6))
plt.plot(bin_center_CDP, average_bin_means_CDP, color='black', linewidth=2, label='Average CDP Size Distribution')

plt.xlabel("Deliquesced Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"CDP Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Average Ambient Below Cloud Base CDP Size Distribution\n January - June 2022", fontsize=14, fontweight="bold")
plt.show()
#%%
#Plotting CAS and CDP ambient averages together 

# %%
#fitting an exponential to ambient 


# Define the exponential function
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

# Dictionary to store fitted parameters for CDP data
CDP_fits = []

# Create figure for plotting
plt.figure(figsize=(8, 6))

# Process and fit each leg's data in Y_CDP_calc
for entry in Y_CDP_calc:
    bin_means_CDP = np.array([entry.get(f'Bin{i:02d}_Y_mean', np.nan) for i in range(0, 30)], dtype=float)
    valid_indices = ~np.isnan(bin_means_CDP)  # Mask for valid (non-NaN) values
    bin_centers_valid = np.array(bin_center_CDP)[valid_indices]
    bin_means_valid = bin_means_CDP[valid_indices]

    # Skip if no valid data
    if not valid_indices.any():
        print(f"No valid data for date {entry['Date']}")
        continue

    # Fit exponential function
    try:
        popt, _ = curve_fit(exponential, bin_centers_valid, bin_means_valid, p0=(1, 1), maxfev=5000)
        n0, D = popt

        # Store fitted parameters
        CDP_fits.append({
            'Date': entry['Date'],
            'BCB_start': entry.get('BCB_start', np.nan),
            'BCB_stop': entry.get('BCB_stop', np.nan),
            'Intercept_n0': n0,
            'E_folding_D': D
        })

        # Generate fitted curve
        x_fit = np.linspace(min(bin_centers_valid), max(bin_centers_valid), 100)
        y_fit = exponential(x_fit, *popt)

        # Plot the fitted curve as a black line
        plt.plot(x_fit, y_fit, color='black', alpha=0.2)

    except RuntimeError:
        print(f"Fit could not be performed for date {entry['Date']}")

# Formatting and labels
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


# Ensure bin_center is an array
bin_center_CDP = np.array(bin_center_CDP)

# Define the exponential function
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

# Dictionary to store fitted parameters for CDP data
CDP_fits_10 = []

# Create figure for plotting
plt.figure(figsize=(8, 6))

# Process and fit each leg's data in Y_CDP_calc
for entry in Y_CDP_calc:
    bin_means_CDP = np.array([entry.get(f'Bin{i:02d}_Y_mean', np.nan) for i in range(0, 30)], dtype=float)
    
    # **Only keep data where bin center ≤ 10 µm**
    valid_indices = (bin_center_CDP <= 10) & ~np.isnan(bin_means_CDP)
    bin_centers_valid = np.array(bin_center_CDP)[valid_indices]
    bin_means_valid = bin_means_CDP[valid_indices]

    # Skip if no valid data
    if valid_indices.any():
        try:
            # Fit exponential function with constraints
            popt, _ = curve_fit(exponential, bin_centers_valid, bin_means_valid, 
                                p0=(1, 1), maxfev=5000, 
                                bounds=([0, 0.1], [np.inf, 20]))  # Constrain D
            n0, D = popt

            # Detect extreme slopes (optional warning for high values of D)
            if D > 15:
                print(f"⚠️ High slope detected! Date: {entry['Date']}, D: {D:.2f}")

        except RuntimeError:
            print(f"❌ Fit failed for date {entry['Date']}")
            n0, D = np.nan, np.nan  # Assign NaN if fitting fails
    else:
        n0, D = np.nan, np.nan  # No valid data, store NaN values

    # Store fitted parameters
    CDP_fits_10.append({
        'Date': entry['Date'],
        'BCB_start': entry.get('BCB_start', np.nan),
        'BCB_stop': entry.get('BCB_stop', np.nan),
        'Intercept_n0': n0,
        'E_folding_D': D
    })

    # **Plot the fitted curve only within the valid range (≤ 10 µm)**
    if not np.isnan(n0) and not np.isnan(D):
        x_fit = np.linspace(min(bin_centers_valid), 10, 100)
        y_fit = exponential(x_fit, *popt)
        plt.plot(x_fit, y_fit, color='black', alpha=0.2)

# Formatting and labels
plt.xlabel("Deliquesced Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"CDP Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.ylim(10**-33, 10**1)
plt.title("Below Cloud Base January - June 2022\n Exponential Fit to CDP Size Distributions (≤10 µm)", fontsize=14, fontweight="bold")
plt.show()

# Print number of successful fits
print(f"Total successful CDP exponential fits: {np.sum(~np.isnan([fit['E_folding_D'] for fit in CDP_fits_10]))}")

# %%
#histogram for slope for 10um
ambient_slope_10_CDP = []
for fit in CDP_fits_10:
    if 'E_folding_D' in fit and not np.isnan(fit['E_folding_D']):
        ambient_slope_10_CDP.append(fit['E_folding_D'])
# Plot histogram of slopes
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
    for bin_means_CDP in flight_data:  # ✅ Ensure we correctly reference `bin_means_CDP`
        # Create dictionaries for each flight leg
        Y_calc_CDP = {'Date': bin_means_CDP['Date'], 'BCB_start': bin_means_CDP['BCB_start'], 'BCB_stop': bin_means_CDP['BCB_stop']}
        N_calc_CDP = {'Date': bin_means_CDP['Date'], 'BCB_start': bin_means_CDP['BCB_start'], 'BCB_stop': bin_means_CDP['BCB_stop']}

        for bin_label in range(0, 30):
            bin_key_Y = f'Bin{bin_label:02d}_Y_mean'  # ✅ Ensuring zero-padded bin keys
            bin_key_N = f'Bin{bin_label:02d}_N_mean'

            # ✅ Use `.get()` to avoid KeyErrors if a bin is missing
            Y_bin_value = np.nanmean(bin_means_CDP.get(bin_key_Y, np.nan))
            N_bin_value = np.nanmean(bin_means_CDP.get(bin_key_N, np.nan))

            # ✅ Multiply by logarithmic bin width factor
            Y_calc_CDP[bin_key_Y] = Y_bin_value * bin_log_CDP[bin_label]
            N_calc_CDP[bin_key_N] = N_bin_value * bin_log_CDP[bin_label]

        # ✅ Append processed data to lists
        Y_BCB_calc_cm3_CDP.append(Y_calc_CDP)
        N_BCB_calc_cm3_CDP.append(N_calc_CDP)

# ✅ Check results
print(f"Processed {len(Y_BCB_calc_cm3_CDP)} flight legs for CDP.")

# %%
#Calculating total number concentration 
# Dictionary to store total concentrations for each leg
total_concentration_cm3_CDP = []

# Process total concentration for each leg
for entry in Y_BCB_calc_cm3_CDP:
    # ✅ Use .get() to avoid KeyErrors if a bin is missing
    total_Y_concentration = np.nansum([entry.get(f'Bin{i:02d}_Y_mean', 0) for i in range(0, 30)])  # ✅ Default to 0 if missing

    total_concentration_cm3_CDP.append({
        'Date': entry['Date'],
        'BCB_start': entry['BCB_start'],
        'BCB_stop': entry['BCB_stop'],
        'Total_Y_Concentration_cm3': total_Y_concentration
    })

# ✅ Check results
print(f"Processed {len(total_concentration_cm3_CDP)} flight legs for total CDP concentration.")

# %%
total_Y_concentrations_CDP = [entry['Total_Y_Concentration_cm3'] for entry in total_concentration_cm3_CDP]

# Remove NaN values if any
total_Y_concentrations_CDP = [conc for conc in total_Y_concentrations_CDP if not np.isnan(conc)]

# Calculate mean total concentration
mean_total_concentration_CDP = np.mean(total_Y_concentrations_CDP)

# Print result
print(f"Mean Total Number Concentration: {mean_total_concentration_CDP:.2f} cm⁻³")
# %%
#CR 1a

wind_speed_dict_CDP = {
    (row['Date'], row['BCB_start'], row['BCB_stop']): row['Windspeed']
    for _, row in df_combined.iterrows()
}

# ✅ Lists to store matched wind speeds and total concentrations
matched_wind_speeds = []
matched_total_concentrations = []

# ✅ Match each total concentration with its corrected wind speed
for entry in total_concentration_cm3_CDP:
    key = (entry['Date'], entry['BCB_start'], entry['BCB_stop'])

    if key in wind_speed_dict_CDP:
        matched_total_concentrations.append(entry['Total_Y_Concentration_cm3'])
        matched_wind_speeds.append(wind_speed_dict_CDP[key])

# ✅ Convert lists to NumPy arrays
matched_wind_speeds = np.array(matched_wind_speeds, dtype=np.float64)
matched_total_concentrations = np.array(matched_total_concentrations, dtype=np.float64)

# ✅ Handle NaN values (ensure no NaN before plotting)
valid_indices = ~np.isnan(matched_wind_speeds) & ~np.isnan(matched_total_concentrations)
matched_wind_speeds = matched_wind_speeds[valid_indices]
matched_total_concentrations = matched_total_concentrations[valid_indices]

# ✅ Scatter plot: Corrected Wind Speed vs. Total Concentration
plt.figure(figsize=(8, 6))
plt.scatter(matched_wind_speeds, matched_total_concentrations, edgecolors='black', facecolors='none', marker='o')

# ✅ Formatting to match scientific standards
plt.xlabel("Corrected Wind Speed (m/s)", fontsize=14, fontweight='bold')
plt.ylabel("Total Concentration (cm$^{-3}$)", fontsize=14, fontweight='bold')
plt.title("Total Concentration vs. Corrected Wind Speed CDP", fontsize=14, fontweight='bold')

# ✅ Add reference lines if needed (adjust values based on expectations)
plt.axhline(0.05, color='red', linestyle='--', label="Reference Min (0.05 cm⁻³)")
plt.axhline(0.3, color='blue', linestyle='--', label="Reference Max (0.3 cm⁻³)")
plt.legend()

plt.tight_layout()
plt.show()
#%%
#Calculating correlation and fitting a line
from scipy.stats import pearsonr, spearmanr


# ✅ Compute Pearson & Spearman correlation coefficients
pearson_corr, pearson_p = pearsonr(matched_wind_speeds, matched_total_concentrations)
spearman_corr, spearman_p = spearmanr(matched_wind_speeds, matched_total_concentrations)

print(f"Pearson Correlation: {pearson_corr:.3f} (p = {pearson_p:.3e})")
print(f"Spearman Correlation: {spearman_corr:.3f} (p = {spearman_p:.3e})")

# ✅ Define a linear function for regression
def linear_model(x, m, b):
    return m * x + b

# ✅ Fit a linear regression model
popt, pcov = curve_fit(linear_model, matched_wind_speeds, matched_total_concentrations)
m_fit, b_fit = popt  # Extract slope and intercept

# ✅ Compute R² value
residuals = matched_total_concentrations - linear_model(matched_wind_speeds, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((matched_total_concentrations - np.mean(matched_total_concentrations))**2)
r_squared = 1 - (ss_res / ss_tot)

print(f"Linear Fit: y = {m_fit:.3f}x + {b_fit:.3f} (R² = {r_squared:.3f})")

# ✅ Scatter plot with linear trend line
plt.figure(figsize=(8, 6))
plt.scatter(matched_wind_speeds, matched_total_concentrations, edgecolors='black', facecolors='none', marker='o')
plt.plot(matched_wind_speeds, linear_model(matched_wind_speeds, *popt), color='red', linewidth=2, 
         label=f'Fit: y = {m_fit:.3f}x + {b_fit:.3f}, R² = {r_squared:.2f}')

# ✅ Formatting
plt.xlabel("Corrected Wind Speed (m s$^{-1}$)", fontsize=16, fontweight='bold')
plt.ylabel("Total Concentration (cm$^{-3}$)", fontsize=16, fontweight='bold')
plt.title("Ambient Concentration vs. 10m Wind Speed CDP", fontsize=16, fontweight='bold')

# ✅ Optional: Add reference lines if applicable
plt.axhline(0.05, color='red', linestyle='--', label="Reference Min (0.05 cm⁻³)")
plt.axhline(0.3, color='blue', linestyle='--', label="Reference Max (0.3 cm⁻³)")
plt.legend()  # Display legend with equation
plt.xticks(fontweight='bold', fontsize=14)
plt.yticks(fontweight='bold', fontsize=14)  
plt.tight_layout()
plt.show()
#%%
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
            rh9_mean = np.nanmean(rh9)

        rh_times['Rh_mean'].append(rh9_mean)
        all_BCB.append(rh_times)  # List that contains all the BCB wind/alt mean dictionaries for 1 flight

    master_BCB_RH.append(all_BCB)

# Step 2: Replace -999 with NaN in master_BCB_RH
for flight in master_BCB_RH:
    for leg in flight:
        rh_mean_list = leg['Rh_mean']
        leg['Rh_mean'] = [np.nan if value <=0 else value for value in rh_mean_list]

#%%
#for only the legs present after LWC filtration and master_BCB_exponential 
#Filtering master_BCB_RH because some legs may not contain humidity data. So we need to remove those legs. 
#Do not linearly interpolate the data. We are working with observations and linearly interpolating the data would no longer represent observation. 
# Extract the (date, BCB_start, BCB_stop) from master_BCB_exponential
# Extract the (date, BCB_start, BCB_stop) from Y_BCB_calc
date_leg_set = set()

for entry in Y_CDP_calc:  # Iterate over the list directly
    date = entry['Date']
    BCB_start = entry.get('BCB_start', np.nan)
    BCB_stop = entry.get('BCB_stop', np.nan)
    date_leg_set.add((date, BCB_start, BCB_stop))


# Filter master_BCB_RH
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
# %%
#Make sure the leg counts 
# Flatten the list of lists and count the total number of entries
total_entries_filtered_master_BCB_RH_CDP = sum(len(legs) for legs in filtered_master_BCB_RH_CDP)
print(f"Total entries in filtered_master_BCB_RH: {total_entries_filtered_master_BCB_RH_CDP}")
# %%
##Obtaining g(RH) = [1.7 / (1-RH)]^0.31 for all mean RH values 
## ie for every leg 

master_BCB_gRH_CDP = []

# Iterate over each flight in master_BCB_RH
for flight in master_BCB_RH:
    flight_gRH = []  # To store the modified data for each flight
    
    for leg in flight:
        new_leg = leg.copy()  # Copy the dictionary to preserve the structure
        
        # Access the single Rh_mean value (it's in a list)
        rh_mean = new_leg['Rh_mean'][0] / 100.0  # Convert percentage to a decimal
        
        # Apply the equation to Rh_mean and store the result
        if np.isnan(rh_mean) or rh_mean >= 1:
            # If Rh_mean is NaN or greater than or equal to 1, set gRH_value to NaN
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

# Iterate over each flight in master_BCB_RH
for flight in filtered_master_BCB_RH_CDP:
    flight_gRH = []  # To store the modified data for each flight
    
    for leg in flight:
        new_leg = leg.copy()  # Copy the dictionary to preserve the structure
        
        # Access the single Rh_mean value (it's in a list)
        rh_mean = new_leg['Rh_mean'][0] / 100.0  # Convert percentage to a decimal
        
        # Apply the equation to Rh_mean and store the result
        if np.isnan(rh_mean) or rh_mean >= 1:
            # If Rh_mean is NaN or greater than or equal to 1, set gRH_value to NaN
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

# Create histogram
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

# Flatten master_min_gRH if it's a list of lists
if isinstance(filtered_master_BCB_gRH_CDP[0], list):
    filtered_master_BCB_gRH_CDP = [item for sublist in filtered_master_BCB_gRH_CDP for item in sublist]

# Loop through each entry in filtered_master_BCB_gRH
for entry in filtered_master_BCB_gRH_CDP:
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    gRh_mean = entry['gRh_mean'][0]  # Assuming gRh_mean is a list with one value
    Rh_mean = entry['Rh_mean'][0]  # Assuming Rh_mean is a list with one value

    # Skip entries with invalid Rh_mean values
    if Rh_mean < 0:
        continue

    # Create a unique key for this entry
    key = (date, BCB_start, BCB_stop)

    # Find corresponding ambient exponential parameters
    if key in CDP_fits_10:
        n0 = CDP_fits_10[key]['Intercept_n0']  # Extract n0 from ambient_fits
        
        # Calculate dry intercept
        dryintercept = n0 / gRh_mean if gRh_mean > 0 else np.nan

        # Store the result in the dictionary
        filtered_master_BCB_interceptdry_dict_CDP[key] = {
            'Date': date,
            'BCB_start': BCB_start,
            'BCB_stop': BCB_stop,
            'Rh_mean': entry['Rh_mean'],
            'gRh_mean': entry['gRh_mean'],
            'dry intercept': dryintercept
        }

# Convert dictionary to a list
filtered_master_BCB_dryintercept_CDP = list(filtered_master_BCB_interceptdry_dict_CDP.values())

print(f"Length of filtered_master_BCB_dryintercept: {len(filtered_master_BCB_dryintercept_CDP)}")

# Histogram of dry intercept values
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

# Plot histogram of ambient intercepts
plt.figure(figsize=(8, 6))
plt.hist(ambient_intercept_values_CDP, bins=20, edgecolor='black', alpha=0.7)

# Labels and formatting
plt.xlabel(r"$\mathbf{Ambient\ Intercept\ (cm^{-3}\ \mu m^{-1})}$", fontsize=15)
plt.ylabel('Frequency', fontsize=15, fontweight='bold')
plt.title('Histogram of Ambient Intercepts (N0)', fontsize=16, fontweight='bold')
plt.grid(True)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')

# Show plot
plt.show()


# %%
#Scatter plot of ambient vs dry intercepts
import numpy as np
import matplotlib.pyplot as plt

# ✅ Extract only the values (not the keys) when iterating over CDP_fits_10
ambient_n0_values_CDP = [fit['Intercept_n0'] for fit in CDP_fits_10.values() if not np.isnan(fit['Intercept_n0'])]

# ✅ Extract dry intercept values correctly
dry_intercept_values_CDP = [leg['dry intercept'] for leg in filtered_master_BCB_dryintercept_CDP if not np.isnan(leg['dry intercept'])]

# ✅ Ensure matching lengths before plotting scatter
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

# ✅ Overlaid Histogram of Ambient and Dry Intercepts
plt.figure(figsize=(8, 6))
plt.hist(ambient_n0_values_CDP, bins=20, alpha=0.5, label="Ambient Intercept (N0)", edgecolor="black")
plt.hist(dry_intercept_values_CDP, bins=20, alpha=0.5, label="Dry Intercept (N0/gRH)", edgecolor="black")

# Labels and Formatting
plt.xlabel(r"Intercept Value (cm$^{-3} \mu$m$^{-1}$)", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.title("Comparison of Ambient vs. Dry Intercept Distributions", fontsize=16, fontweight='bold')
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# %%
#dry size distributions

import numpy as np

# Initialize the list for storing CDP dry size distributions
filtered_master_BCB_ddry_CDP = []

# Iterate over each entry in filtered_master_BCB_gRH_CDP
for entry in filtered_master_BCB_gRH_CDP:
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    gRh_mean = entry['gRh_mean'][0]  # Assuming it's stored as a list

    # Compute dry bin centers for CDP
    if gRh_mean > 0:
        ddry_values_CDP = np.array([D_amb / gRh_mean for D_amb in bin_center_CDP])
    else:
        ddry_values_CDP = np.full(len(bin_center_CDP), np.nan)
        print(f"Skipping division for {date}, {BCB_start}-{BCB_stop} due to invalid gRh_mean.")

    # Compute bin widths for dry size distribution (∆Ddry = Ddry[i+1] - Ddry[i])
    ddry_bin_widths_CDP = np.diff(ddry_values_CDP, append=np.nan)  # Append NaN to keep array size consistent

    # Find the corresponding ambient size distribution in CDP
    raw_concentrations_CDP = next(
        (leg for leg in Y_CDP_calc if leg['Date'] == date and leg['BCB_start'] == BCB_start and leg['BCB_stop'] == BCB_stop),
        None
    )

    if raw_concentrations_CDP:
        # Extract raw bin concentrations (dN/dDambient for CDP)
        dN_dD_ambient_CDP = np.array([raw_concentrations_CDP.get(f'Bin{i:02d}_Y_mean', np.nan) for i in range(0, 30)], dtype=float)

        # Apply the transformation using the correct bin widths
        dN_dD_dry_CDP = np.where(
            (~np.isnan(dN_dD_ambient_CDP)) & (~np.isnan(ddry_bin_widths_CDP)) & (gRh_mean > 0),
            dN_dD_ambient_CDP * (np.array(bin_center_CDP) / ddry_values_CDP) * (np.diff(bin_center_CDP, append=np.nan) / ddry_bin_widths_CDP),
            np.nan
        )
    else:
        dN_dD_dry_CDP = np.full(len(bin_center_CDP), np.nan)
        print(f"Missing raw size distribution for {date}, {BCB_start}-{BCB_stop}")

    # Store the raw dry size distribution for CDP
    filtered_master_BCB_ddry_CDP.append({
        'Date': date,
        'BCB_start': BCB_start,
        'BCB_stop': BCB_stop,
        'ddry': ddry_values_CDP.tolist(),
        'dN/dDdry': dN_dD_dry_CDP.tolist(),
        'ddry_bin_widths': ddry_bin_widths_CDP.tolist(),  # Store bin widths separately
        'gRh_mean': gRh_mean
    })

print(f"Length of filtered_master_BCB_ddry_CDP: {len(filtered_master_BCB_ddry_CDP)}")

# %%
#Plotting raw dry size distributions

# Define common bin centers for interpolation (adjust as needed)
common_bins_CDP = np.linspace(2, 25, 35)  

plt.figure(figsize=(8, 6))

# Loop through each dry size distribution in CDP dataset
for entry in filtered_master_BCB_ddry_CDP:
    ddry_values_CDP = np.array(entry['ddry'])  # The unique dry bin centers for this leg
    dN_dD_dry_CDP = np.array(entry['dN/dDdry'])  # The corresponding concentration values

    # Remove NaN values before interpolation
    valid_indices = ~np.isnan(ddry_values_CDP) & ~np.isnan(dN_dD_dry_CDP)
    if np.sum(valid_indices) < 2:
        continue  # Skip if not enough valid points for interpolation

    # Interpolate onto the common bins
    interp_func_CDP = interp1d(ddry_values_CDP[valid_indices], dN_dD_dry_CDP[valid_indices], 
                               kind='linear', bounds_error=False, fill_value=np.nan)
    interpolated_dN_dD_dry_CDP = interp_func_CDP(common_bins_CDP)

    # Plot the interpolated dry size distribution
    plt.plot(common_bins_CDP, interpolated_dN_dD_dry_CDP, color='black', alpha=0.2)

# Formatting and labels
plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base January - June 2022\n Raw Dry Size Distributions (CDP)", fontsize=14, fontweight="bold")
plt.show()

# %%
#removing the 0s

# Define common bin centers for interpolation
common_bins_CDP = np.linspace(2, 25, 35)  # Adjust bin range and count as needed

plt.figure(figsize=(8, 6))

# Loop through each dry size distribution in CDP dataset
for entry in filtered_master_BCB_ddry_CDP:
    ddry_values_CDP = np.array(entry['ddry'])  # Unique dry bin centers for this leg
    dN_dD_dry_CDP = np.array(entry['dN/dDdry'])  # Corresponding concentration values

    # Remove NaN values before interpolation
    valid_indices = ~np.isnan(ddry_values_CDP) & ~np.isnan(dN_dD_dry_CDP)
    if np.sum(valid_indices) < 2:
        continue  # Skip if not enough valid points for interpolation

    # Interpolate onto the common bins
    interp_func_CDP = interp1d(ddry_values_CDP[valid_indices], dN_dD_dry_CDP[valid_indices], 
                               kind='linear', bounds_error=False, fill_value=np.nan)
    interpolated_dN_dD_dry_CDP = interp_func_CDP(common_bins_CDP)

    # Mask: Remove NaNs and zero values
    valid_interpolated_indices = (interpolated_dN_dD_dry_CDP > 0) & ~np.isnan(interpolated_dN_dD_dry_CDP)
    filtered_bins_CDP = common_bins_CDP[valid_interpolated_indices]
    filtered_dN_dD_dry_CDP = interpolated_dN_dD_dry_CDP[valid_interpolated_indices]

    if len(filtered_bins_CDP) > 0:  # Only plot if valid data exists
        plt.plot(filtered_bins_CDP, filtered_dN_dD_dry_CDP, color='black', alpha=0.2)

# Formatting and labels
plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base January - June 2022\n Raw Dry Size Distributions (CDP)", fontsize=14, fontweight="bold")
plt.show()

# %%
#Averaging the dry size distributions

# Define common bin centers for interpolation
common_bins_CDP = np.linspace(2, 25, 35)  # Adjust bin range and count as needed

# Initialize sum and count arrays for averaging
sum_interpolated_dN_dD_dry_CDP = np.zeros_like(common_bins_CDP, dtype=float)
count_interpolated_dN_dD_dry_CDP = np.zeros_like(common_bins_CDP, dtype=int)

# Loop through each dry size distribution in CDP dataset
for entry in filtered_master_BCB_ddry_CDP:
    ddry_values_CDP = np.array(entry['ddry'])  # Unique dry bin centers for this leg
    dN_dD_dry_CDP = np.array(entry['dN/dDdry'])  # Corresponding concentration values

    # Remove NaN values before interpolation
    valid_indices = ~np.isnan(ddry_values_CDP) & ~np.isnan(dN_dD_dry_CDP)
    if np.sum(valid_indices) < 2:
        continue  # Skip if not enough valid points for interpolation

    # Interpolate onto the common bins
    interp_func_CDP = interp1d(ddry_values_CDP[valid_indices], dN_dD_dry_CDP[valid_indices], 
                               kind='linear', bounds_error=False, fill_value=np.nan)
    interpolated_dN_dD_dry_CDP = interp_func_CDP(common_bins_CDP)

    # Mask: Remove NaNs and zero values
    valid_interpolated_indices = (interpolated_dN_dD_dry_CDP > 0) & ~np.isnan(interpolated_dN_dD_dry_CDP)

    # Accumulate sum and count for averaging
    sum_interpolated_dN_dD_dry_CDP[valid_interpolated_indices] += interpolated_dN_dD_dry_CDP[valid_interpolated_indices]
    count_interpolated_dN_dD_dry_CDP[valid_interpolated_indices] += 1

# Compute average dry size distribution (avoid division by zero)
average_dN_dD_dry_CDP = np.divide(sum_interpolated_dN_dD_dry_CDP, count_interpolated_dN_dD_dry_CDP, where=count_interpolated_dN_dD_dry_CDP > 0)

# Plot the averaged dry size distribution
plt.figure(figsize=(8, 6))
plt.plot(common_bins_CDP, average_dN_dD_dry_CDP, color='black', linewidth=2, label='Average Dry Size Distribution (CDP)')

# Formatting and labels
plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Average Below Cloud Base Dry Size Distribution (CDP)\n January - June 2022", fontsize=14, fontweight="bold")
# Show plot
plt.show()

# %%
#both ambient and dry 

common_bins_CDP = np.linspace(2, 40, 35)  # Adjust bin range as needed

plt.figure(figsize=(8, 6))

# Plot both ambient and dry size distributions for comparison
for entry_ambient, entry_dry in zip(Y_CDP_calc, filtered_master_BCB_ddry_CDP):
    # Extract ambient data (CAS bin centers for CDP)
    ambient_dd_CDP = np.array(bin_center_CDP)
    ambient_dN_dD_CDP = np.array([entry_ambient.get(f'Bin{i:02d}_Y_mean', np.nan) for i in range(0, 30)])

    # Extract dry data
    dry_dd_CDP = np.array(entry_dry['ddry'])
    dry_dN_dD_CDP = np.array(entry_dry['dN/dDdry'])

    # Remove NaN values before interpolation
    valid_ambient = ~np.isnan(ambient_dd_CDP) & ~np.isnan(ambient_dN_dD_CDP)
    valid_dry = ~np.isnan(dry_dd_CDP) & ~np.isnan(dry_dN_dD_CDP)

    if np.sum(valid_ambient) < 2 or np.sum(valid_dry) < 2:
        continue  # Skip if not enough valid points

    # Interpolate onto common bins
    interp_ambient_CDP = interp1d(ambient_dd_CDP[valid_ambient], ambient_dN_dD_CDP[valid_ambient], 
                                  kind='linear', bounds_error=False, fill_value=np.nan)
    interp_dry_CDP = interp1d(dry_dd_CDP[valid_dry], dry_dN_dD_CDP[valid_dry], 
                              kind='linear', bounds_error=False, fill_value=np.nan)

    interpolated_ambient_CDP = interp_ambient_CDP(common_bins_CDP)
    interpolated_dry_CDP = interp_dry_CDP(common_bins_CDP)

    # Filter out zero and NaN values
    valid_ambient_bins_CDP = (interpolated_ambient_CDP > 0) & ~np.isnan(interpolated_ambient_CDP)
    valid_dry_bins_CDP = (interpolated_dry_CDP > 0) & ~np.isnan(interpolated_dry_CDP)

    # Apply filters to remove zero values before plotting
    filtered_ambient_bins_CDP = common_bins_CDP[valid_ambient_bins_CDP]
    filtered_ambient_values_CDP = interpolated_ambient_CDP[valid_ambient_bins_CDP]

    filtered_dry_bins_CDP = common_bins_CDP[valid_dry_bins_CDP]
    filtered_dry_values_CDP = interpolated_dry_CDP[valid_dry_bins_CDP]

    # Plot ambient in blue, dry in red
    if len(filtered_ambient_bins_CDP) > 0:
        plt.plot(filtered_ambient_bins_CDP, filtered_ambient_values_CDP, color='blue', alpha=0.3, label="Ambient" if 'Ambient' not in plt.gca().get_legend_handles_labels()[1] else "")

    if len(filtered_dry_bins_CDP) > 0:
        plt.plot(filtered_dry_bins_CDP, filtered_dry_values_CDP, color='red', alpha=0.3, label="Dry" if 'Dry' not in plt.gca().get_legend_handles_labels()[1] else "")

# Formatting
plt.xlabel("Bin Centers Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-7, 10**1)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base January - June 2022\n Raw CDP Size Distributions", fontsize=14, fontweight="bold")

# Create custom legend handles with thicker lines
ambient_legend = mlines.Line2D([], [], color='blue', linewidth=6, label="Ambient")  # Thicker blue line
dry_legend = mlines.Line2D([], [], color='red', linewidth=6, label="Dry")  # Thicker red line

# Apply the custom legend with thicker lines
plt.legend(handles=[ambient_legend, dry_legend], fontsize=12, frameon=True)

plt.show()
#%%
#Fitting an exponential to the dry 


def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

# Initialize a list to store fitted parameters
dry_exponential_fits_CDP = []

plt.figure(figsize=(8, 6))

# Process and fit each dry size distribution in CDP dataset
for entry in filtered_master_BCB_ddry_CDP:
    ddry_values_CDP = np.array(entry['ddry'])
    dN_dD_dry_CDP = np.array(entry['dN/dDdry'])

    # Ensure valid data points
    valid_indices = ~np.isnan(ddry_values_CDP) & ~np.isnan(dN_dD_dry_CDP) & (dN_dD_dry_CDP > 0)
    
    if np.sum(valid_indices) < 5:  
        continue  # Skip if not enough valid points

    try:
        # Fit the exponential function
        popt, _ = curve_fit(exponential, ddry_values_CDP[valid_indices], dN_dD_dry_CDP[valid_indices], 
                            p0=(1, 5), maxfev=5000)
        n0, D = popt

        # Store fitted parameters
        dry_exponential_fits_CDP.append({
            'Date': entry['Date'],
            'BCB_start': entry['BCB_start'],
            'BCB_stop': entry['BCB_stop'],
            'Dry_Intercept_n0': n0,
            'Dry_E_folding_D': D
        })

        # Generate fitted curve
        x_fit = np.linspace(min(ddry_values_CDP[valid_indices]), max(ddry_values_CDP[valid_indices]), 100)
        y_fit = exponential(x_fit, *popt)

        # Ensure reasonable y-values before plotting
        if np.all(y_fit > 1e-33):
            plt.plot(x_fit, y_fit, color='black', alpha=0.2)

    except RuntimeError:
        print(f"Fit could not be performed for date {entry['Date']}")

# Formatting
plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-33, 10**1.5)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base January - June 2022\n Exponential Fitted Dry Size Distributions (CDP)", fontsize=14, fontweight="bold")

plt.show()

print(f"Total successful dry exponential fits (CDP): {len(dry_exponential_fits_CDP)}")
#%%
#average exponential fit 
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

# Define the exponential function
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

# Define common bin centers for interpolation
common_bins_CDP = np.linspace(2, 25, 35)  # Adjust bin range and count as needed

# Initialize sum and count arrays for averaging
sum_interpolated_dN_dD_dry_CDP = np.zeros_like(common_bins_CDP, dtype=float)
count_interpolated_dN_dD_dry_CDP = np.zeros_like(common_bins_CDP, dtype=int)

# Loop through each dry size distribution and accumulate values
for entry in filtered_master_BCB_ddry_CDP:
    ddry_values_CDP = np.array(entry['ddry'])
    dN_dD_dry_CDP = np.array(entry['dN/dDdry'])

    # Ensure valid data points
    valid_indices = ~np.isnan(ddry_values_CDP) & ~np.isnan(dN_dD_dry_CDP) & (dN_dD_dry_CDP > 0)
    if np.sum(valid_indices) < 2:  
        continue

    # Interpolation onto common bins
    interp_func_CDP = interp1d(ddry_values_CDP[valid_indices], dN_dD_dry_CDP[valid_indices], 
                               kind='linear', bounds_error=False, fill_value=np.nan)
    interpolated_dN_dD_dry_CDP = interp_func_CDP(common_bins_CDP)

    # Mask: Remove NaNs and zero values
    valid_interpolated_indices = (interpolated_dN_dD_dry_CDP > 0) & ~np.isnan(interpolated_dN_dD_dry_CDP)

    # Accumulate sum and count for averaging
    sum_interpolated_dN_dD_dry_CDP[valid_interpolated_indices] += interpolated_dN_dD_dry_CDP[valid_interpolated_indices]
    count_interpolated_dN_dD_dry_CDP[valid_interpolated_indices] += 1

# Compute the average dry size distribution
average_dN_dD_dry_CDP = np.divide(sum_interpolated_dN_dD_dry_CDP, count_interpolated_dN_dD_dry_CDP, where=count_interpolated_dN_dD_dry_CDP > 0)

# Fit an exponential function to the averaged size distribution
valid_fit_indices_CDP = ~np.isnan(average_dN_dD_dry_CDP) & (average_dN_dD_dry_CDP > 0)
fit_bins_CDP = common_bins_CDP[valid_fit_indices_CDP]
fit_values_CDP = average_dN_dD_dry_CDP[valid_fit_indices_CDP]

try:
    popt, _ = curve_fit(exponential, fit_bins_CDP, fit_values_CDP, p0=(1, 5), maxfev=5000)
    n0_fit_CDP, D_fit_CDP = popt

    # Generate exponential fit curve
    x_fit_CDP = np.linspace(min(fit_bins_CDP), max(fit_bins_CDP), 100)
    y_fit_CDP = exponential(x_fit_CDP, *popt)

except RuntimeError:
    print("Exponential fit could not be performed.")
    n0_fit_CDP, D_fit_CDP = np.nan, np.nan
    x_fit_CDP, y_fit_CDP = [], []

# Plot the averaged dry size distribution (Raw Data)
plt.figure(figsize=(8, 6))
plt.plot(common_bins_CDP, average_dN_dD_dry_CDP, color='black', linewidth=2, label='Average Dry Size Distribution (CDP)')

# Plot the Exponential Fit
if len(x_fit_CDP) > 0:
    plt.plot(x_fit_CDP, y_fit_CDP, 'r--', linewidth=2, label=f'≤10 µm Fit: n0={n0_fit_CDP:.2e}, D={D_fit_CDP:.2f} µm')

# Formatting
plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Average Below Cloud Base Exponential Fitted Dry Size Distribution (CDP)\n January - June 2022", fontsize=14, fontweight="bold")
plt.legend()

# Show plot
plt.show()

print(f"Final Exponential Fit Parameters for CDP: n0 = {n0_fit_CDP:.2e}, D = {D_fit_CDP:.2f} µm")


# %%
#Only to 10 um 

# Define the exponential function
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

# Initialize a list to store fitted parameters
dry_exponential_fits_10_CDP = []

plt.figure(figsize=(8, 6))

# Process and fit each dry size distribution in CDP dataset
for entry in filtered_master_BCB_ddry_CDP:
    ddry_values_CDP = np.array(entry['ddry'])
    dN_dD_dry_CDP = np.array(entry['dN/dDdry'])

    # Filter data to only include bins ≤ 10 µm
    valid_indices = (ddry_values_CDP <= 10) & ~np.isnan(ddry_values_CDP) & ~np.isnan(dN_dD_dry_CDP)

    # If no valid points within ≤ 10 µm, store NaNs but do NOT skip
    if np.sum(valid_indices) == 0:
        dry_exponential_fits_10_CDP.append({
            'Date': entry['Date'],
            'BCB_start': entry['BCB_start'],
            'BCB_stop': entry['BCB_stop'],
            'Dry_Intercept_n0': np.nan,
            'Dry_E_folding_D': np.nan
        })
        continue  # Move to next entry but store NaNs instead of skipping

    try:
        # Fit the exponential only using data up to 10 µm
        popt, _ = curve_fit(exponential, ddry_values_CDP[valid_indices], dN_dD_dry_CDP[valid_indices], 
                            p0=(1, 5), maxfev=5000)
        n0, D = popt

        # **Sanity check: Only accept reasonable D values**
        if D < 0.5 or D > 20:  # Arbitrary threshold, can adjust
            raise RuntimeError("D value out of range")

    except RuntimeError:
        print(f"Fit failed for {entry['Date']} (D={D:.2f})")
        n0, D = np.nan, np.nan  # Store NaN if fitting fails

    # Store fitted parameters (including NaNs for failed fits)
    dry_exponential_fits_10_CDP.append({
        'Date': entry['Date'],
        'BCB_start': entry['BCB_start'],
        'BCB_stop': entry['BCB_stop'],
        'Dry_Intercept_n0': n0,
        'Dry_E_folding_D': D
    })

    # Generate fitted curve only up to 10 µm if fit was successful
    if not np.isnan(n0) and not np.isnan(D):
        x_fit = np.linspace(2, 10, 100)  # Start at 2 µm to avoid extreme behavior near zero
        y_fit = exponential(x_fit, n0, D)

        # **Exclude extreme values to prevent weird downward lines**
        y_fit[y_fit < 1e-15] = np.nan  # Replace very small values with NaN

        plt.plot(x_fit, y_fit, color='black', alpha=0.2)

# Formatting
plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=15, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=15, fontweight="bold")
plt.yscale("log")
plt.ylim(1e-33, 1e1)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base January - June 2022\n Fitted CDP Dry Size Distributions (≤10 µm)", fontsize=15, fontweight="bold")

plt.show()

print(f"Total successful dry exponential fits (CDP ≤10 µm): {len([fit for fit in dry_exponential_fits_10_CDP if not np.isnan(fit['Dry_Intercept_n0'])])}")

# %%
master_BCB = []


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
    
    all_BCB_means = []


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
                #print(times[i][0:5])
            if start9 == start:
                index1_start = i
                break
        
    
        for i in range(len(times)):
            end9 = int(times[i][0:5])
            if end9 == end:
                index1_end = i
                break
        
        if index1_start == None:
                # print(date)
                # print('Did not find start time in Summary')
            winds9_mean = np.nan
            alts9_mean = np.nan
        if index1_end == None:
                # print(date)
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

        all_BCB_means.append(wind_alt) #List that contains all the BCB wind/alt mean dictionaries for 1 flight
        
    master_BCB.append(all_BCB_means) #List that contains all BCB flights  
#%%
Z0 = 0.02  # meters (typical value for open ocean)
Z10 = 10  # target height m

corrected_calc_bcb = {'Date': [], 'Corrected_bcb_windspeed': []}

for flight in master_BCB:
    for wind_alt in flight:
        date = wind_alt['Date']
        windspeed = wind_alt['Winds_mean']
        altitude = wind_alt['Alts_mean']

        for wind_mean, alt_mean in zip(windspeed, altitude):
            # Apply the formula
            new_windspeed = wind_mean * (np.log(Z10/Z0) / np.log(alt_mean / Z0))

            corrected_calc_bcb['Date'].append(date)
            corrected_calc_bcb['Corrected_bcb_windspeed'].append(new_windspeed)
for date, wind_mean in zip(corrected_calc_bcb['Date'], corrected_calc_bcb['Corrected_bcb_windspeed']):
    print(f"Date: {date}, Corrected_bcb_windspeed: {wind_mean}")
#%%
#histogram of altitudes
#Use a dictionary of windspeeds 
def correct_windspeed(windspeed, altitude):
    return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))

combined_data = []

for i, flight in enumerate(master_BCB):
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

            combined_data.append({
                'Date': date,
                'BCB_start': BCB_start,
                'BCB_stop': BCB_stop,
                'Windspeed': corrected_windspeed
            })

        except IndexError as e:
            print(f"Index error at i={i}, j={j}: {e}")
            continue
df_combined = pd.DataFrame(combined_data)
#%%
#both CAS and CDP average ambient 

common_bins = np.linspace(2, 45, 55)  # Adjust range and count as needed

### --- PROCESSING CDP DATA --- ###
# Initialize arrays for averaging for Y_CDP_calc (30 bins)
sum_bin_means_CDP = np.zeros(len(bin_center_CDP))
count_bin_means_CDP = np.zeros(len(bin_center_CDP))

# Iterate through all size distributions in Y_CDP_calc
for entry in Y_CDP_calc:
    bin_means_CDP = np.array([entry.get(f'Bin{i:02d}_Y_mean', np.nan) for i in range(0, 30)], dtype=float)

    # Mask: Remove NaNs and zero values (ensure the length matches bin_means_CDP)
    valid_indices_CDP = (bin_means_CDP > 0) & ~np.isnan(bin_means_CDP)

    # Ensure indexing is valid
    sum_bin_means_CDP[:len(valid_indices_CDP)][valid_indices_CDP] += bin_means_CDP[valid_indices_CDP]
    count_bin_means_CDP[:len(valid_indices_CDP)][valid_indices_CDP] += 1

# Compute average CDP size distribution
average_bin_means_CDP = np.divide(sum_bin_means_CDP, count_bin_means_CDP, where=count_bin_means_CDP > 0)

# Interpolate CDP data onto common bins
interp_func_CDP = interp1d(bin_center_CDP, average_bin_means_CDP, kind='linear', bounds_error=False, fill_value=np.nan)
interpolated_bin_means_CDP = interp_func_CDP(common_bins)

### --- PROCESSING CAS DATA --- ###
# Initialize arrays for averaging for Y_BCB_calc (18 bins)
sum_bin_means_CAS = np.zeros(len(bin_center))
count_bin_means_CAS = np.zeros(len(bin_center))

# Iterate through all size distributions in Y_BCB_calc
for entry in Y_BCB_calc:
    bin_means_CAS = np.array([entry.get(f'Bin{i}_Y_mean', np.nan) for i in range(12, 30)], dtype=float)

    # Mask: Remove NaNs and zero values
    valid_indices_CAS = (bin_means_CAS > 0) & ~np.isnan(bin_means_CAS)

    # Ensure indexing is valid
    sum_bin_means_CAS[:len(valid_indices_CAS)][valid_indices_CAS] += bin_means_CAS[valid_indices_CAS]
    count_bin_means_CAS[:len(valid_indices_CAS)][valid_indices_CAS] += 1

# Compute average CAS size distribution
average_bin_means_CAS = np.divide(sum_bin_means_CAS, count_bin_means_CAS, where=count_bin_means_CAS > 0)

# Interpolate CAS data onto common bins
interp_func_CAS = interp1d(bin_center, average_bin_means_CAS, kind='linear', bounds_error=False, fill_value=np.nan)
interpolated_bin_means_CAS = interp_func_CAS(common_bins)

### --- PLOTTING BOTH DISTRIBUTIONS TOGETHER --- ###
plt.figure(figsize=(8, 6))
plt.plot(common_bins, interpolated_bin_means_CDP, color='blue', linewidth=2, label='Average CDP Size Distribution')
plt.plot(common_bins, interpolated_bin_means_CAS, color='black', linewidth=2, label='Average CAS Size Distribution')

# Labels and formatting
plt.xlabel("Deliquesced Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Average Ambient Below Cloud Base Size Distributions\n January - June 2022", fontsize=14, fontweight="bold")
plt.legend()

# Show plot
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
import random

# Define wind speed bins (same as CAS)
windspeed_bins = [(0, 3), (3.001, 6.5), (6.501, 8.5), (8.501, np.inf)]
cas_bin_counts = [78, 174, 62, 54]  # Exact CAS leg counts per bin

# Store binned distributions for CDP
grouped_distributions_CDP = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds_CDP = {i: [] for i in range(len(windspeed_bins))}
cdp_bin_leg_indices = {i: [] for i in range(len(windspeed_bins))}  # Store leg indices

missing_windspeed_count_CDP = 0

# Define common bins (10 bin centers between 2 and 10 µm)
common_bins = np.linspace(2, 10, 10)

# **Step 1: Collect CDP legs into bins**
for leg_idx, entry in enumerate(dry_exponential_fits_10_CDP):
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']

    n0 = entry['Dry_Intercept_n0']
    D = entry['Dry_E_folding_D']

    # Match windspeed
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) & 
        (df_combined['BCB_start'] == BCB_start) & 
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    if windspeed_entry.empty or np.isnan(n0) or np.isnan(D):
        missing_windspeed_count_CDP += 1
        continue  # Skip legs with missing windspeed

    windspeed = windspeed_entry['Windspeed'].values[0]

    # Use the already fitted size distribution (DO NOT FIT AGAIN)
    size_dist_CDP = n0 * np.exp(-common_bins / D)  # Use existing (n0, D)

    # Bin the distribution by windspeed
    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:
            grouped_distributions_CDP[idx].append(size_dist_CDP)
            mean_windspeeds_CDP[idx].append(windspeed)
            cdp_bin_leg_indices[idx].append(leg_idx)  # Track index for selection
            break

# **Step 2: Randomly select legs to match CAS counts**
final_grouped_CDP = {i: [] for i in range(len(windspeed_bins))}
final_mean_windspeeds_CDP = {i: [] for i in range(len(windspeed_bins))}

for idx in range(len(windspeed_bins)):
    if len(grouped_distributions_CDP[idx]) >= cas_bin_counts[idx]:
        selected_indices = random.sample(range(len(grouped_distributions_CDP[idx])), cas_bin_counts[idx])
    else:
        selected_indices = list(range(len(grouped_distributions_CDP[idx])))  # Use all if not enough

    final_grouped_CDP[idx] = [grouped_distributions_CDP[idx][i] for i in selected_indices]
    final_mean_windspeeds_CDP[idx] = [mean_windspeeds_CDP[idx][i] for i in selected_indices]

# **Step 3: Print verification of windspeed ranges**
print("\n✅ **Windspeed Verification:**")
for idx, (low, high) in enumerate(windspeed_bins):
    if final_mean_windspeeds_CDP[idx]:
        min_ws, max_ws = min(final_mean_windspeeds_CDP[idx]), max(final_mean_windspeeds_CDP[idx])
        print(f"Windspeed bin {idx} ({low} - {high} m/s): {len(final_grouped_CDP[idx])} legs")
        print(f"  - Selected windspeed range: {min_ws:.2f} to {max_ws:.2f} m/s")

# **Step 4: Plot CDP with exact CAS-matching bin counts**
plt.figure(figsize=(10, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if final_grouped_CDP[idx]:
        avg_distribution_CDP = np.mean(final_grouped_CDP[idx], axis=0)  # Average size distribution
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

# **Step 5: Print final leg counts**
final_total_legs = sum(len(group) for group in final_grouped_CDP.values())
print(f"\n🎯 **Final CDP legs actually plotted (should match CAS 368): {final_total_legs}**")

for idx, count in enumerate(cas_bin_counts):
    print(f"Windspeed bin {idx} ({windspeed_bins[idx]} m/s): {len(final_grouped_CDP[idx])} legs (CAS had {count})")



# %%
common_bins=np.linspace(2, 10, 25)


# Define wind speed bins (same as CAS)
windspeed_bins = [(0, 3), (3.001, 6.5), (6.501, 8.5), (8.501, np.inf)]
cas_bin_counts = [78, 174, 62, 54]  # Exact CAS leg counts per bin

# Store binned distributions for CDP
grouped_distributions_CDP = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds_CDP = {i: [] for i in range(len(windspeed_bins))}
cdp_bin_leg_indices = {i: [] for i in range(len(windspeed_bins))}  # Store leg indices

missing_windspeed_count_CDP = 0

# Define common bins (10 bin centers between 2 and 10 µm)
common_bins = np.linspace(2, 10, 25)

# **Step 1: Collect CDP legs into bins**
for leg_idx, entry in enumerate(dry_exponential_fits_10_CDP):
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']

    n0 = entry['Dry_Intercept_n0']
    D = entry['Dry_E_folding_D']

    # Match windspeed
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) & 
        (df_combined['BCB_start'] == BCB_start) & 
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    if windspeed_entry.empty or np.isnan(n0) or np.isnan(D):
        missing_windspeed_count_CDP += 1
        continue  # Skip legs with missing windspeed

    windspeed = windspeed_entry['Windspeed'].values[0]

    # Use the already fitted size distribution (DO NOT FIT AGAIN)
    size_dist_CDP = n0 * np.exp(-common_bins / D)  # Use existing (n0, D)

    # Bin the distribution by windspeed
    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:
            grouped_distributions_CDP[idx].append(size_dist_CDP)
            mean_windspeeds_CDP[idx].append(windspeed)
            cdp_bin_leg_indices[idx].append(leg_idx)  # Track index for selection
            break

# **Step 2: Randomly select legs to match CAS counts**
final_grouped_CDP = {i: [] for i in range(len(windspeed_bins))}
final_mean_windspeeds_CDP = {i: [] for i in range(len(windspeed_bins))}

for idx in range(len(windspeed_bins)):
    if len(grouped_distributions_CDP[idx]) >= cas_bin_counts[idx]:
        selected_indices = random.sample(range(len(grouped_distributions_CDP[idx])), cas_bin_counts[idx])
    else:
        selected_indices = list(range(len(grouped_distributions_CDP[idx])))  # Use all if not enough

    final_grouped_CDP[idx] = [grouped_distributions_CDP[idx][i] for i in selected_indices]
    final_mean_windspeeds_CDP[idx] = [mean_windspeeds_CDP[idx][i] for i in selected_indices]

# **Step 3: Plot CDP with exact CAS-matching bin counts**
plt.figure(figsize=(10, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if final_grouped_CDP[idx]:
        avg_distribution_CDP = np.mean(final_grouped_CDP[idx], axis=0)  # Average size distribution
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


# %%
#adding uncertainty

plt.figure(figsize=(10, 8))

# Choose whether to use Standard Error (SE) or Interquartile Range (IQR) for uncertainty
use_standard_error = True  # Set to False to use IQR instead

for idx, (low, high) in enumerate(windspeed_bins):
    if final_grouped_CDP[idx]:  # Ensure the bin is not empty
        avg_distribution_CDP = np.mean(final_grouped_CDP[idx], axis=0)  # Mean size distribution
        num_legs_CDP = len(final_grouped_CDP[idx])
        avg_windspeed_CDP = np.mean(final_mean_windspeeds_CDP[idx])

        # Standard Error (SE) Option
        if use_standard_error:
            std_error_CDP = np.std(final_grouped_CDP[idx], axis=0, ddof=1) / np.sqrt(num_legs_CDP)
            lower_bound_CDP = avg_distribution_CDP - std_error_CDP
            upper_bound_CDP = avg_distribution_CDP + std_error_CDP

        # Interquartile Range (IQR) Option
        else:
            lower_bound_CDP = np.percentile(final_grouped_CDP[idx], 25, axis=0)  # 25th percentile
            upper_bound_CDP = np.percentile(final_grouped_CDP[idx], 75, axis=0)  # 75th percentile

        # Plot the mean size distribution for CDP
        plt.plot(common_bins, avg_distribution_CDP, label=f"{avg_windspeed_CDP:.1f} m/s, n={num_legs_CDP} legs", linewidth=2.5)

        # Plot uncertainty as shaded region
        plt.fill_between(common_bins, lower_bound_CDP, upper_bound_CDP, alpha=0.3)

# Set Y-axis scaling
plt.yscale('log')  # Change to 'linear' if needed for checking uncertainty

# Labels and formatting
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=16, fontweight="bold")
plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=16, fontweight="bold")
plt.title('CDP Dry Size Distributions Binned by Average Wind Speed CDP', fontweight='bold', fontsize=17)
plt.legend(title=r"Average wind speed m s$^{-1}$")
plt.tight_layout()
plt.ylim(1e-4, 10**0)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')

# Show the plot
plt.show()

# Print sample sizes per bin for verification
for idx, group in final_grouped_CDP.items():
    print(f"Windspeed bin {idx} ({windspeed_bins[idx]} m/s): {len(group)} legs")
#%%
# Define the exponential function
def fit_function(x, n0, D):
    return n0 * np.exp(-x / D)

# Store fitted results for each windspeed bin
fit_results_CDP = {}

plt.figure(figsize=(10, 8))

# Colors must match the bins
windspeed_colors = ['blue', 'orange', 'green', 'red']

for idx, (low, high) in enumerate(windspeed_bins):
    if final_grouped_CDP[idx]:  # Use only the legs selected to match CAS
        concentrations_array = np.array(final_grouped_CDP[idx])
        
        # Compute average concentration for each bin
        avg_concentration_CDP = np.mean(concentrations_array, axis=0)
        
        # Avoid fitting negative or zero values
        avg_concentration_CDP = np.where(avg_concentration_CDP <= 0, 1e-10, avg_concentration_CDP)

        avg_windspeed_CDP = np.mean(final_mean_windspeeds_CDP[idx])
        num_legs_CDP = len(final_grouped_CDP[idx])

        # Perform exponential fit only after selection
        try:
            popt, _ = curve_fit(fit_function, common_bins, avg_concentration_CDP, p0=(1, 5), maxfev=5000)
            n0_fit, D_fit = popt
            fit_results_CDP[idx] = {'n0': n0_fit, 'D': D_fit, 'avg_windspeed': avg_windspeed_CDP, 'num_legs': num_legs_CDP}

            # Generate fitted curve
            x_fit = np.linspace(min(common_bins), max(common_bins), 10)
            y_fit = fit_function(x_fit, *popt)

            # Plot fitted lines
            plt.plot(x_fit, y_fit, color=windspeed_colors[idx], linewidth=3, linestyle='-',
                     label=f"{avg_windspeed_CDP:.1f} m/s, n={num_legs_CDP} legs")

        except RuntimeError:
            print(f"Exponential fit failed for windspeed bin {avg_windspeed_CDP:.1f} m/s")

# Plot Formatting
plt.ylabel(r'Number concentration (cm$^{-3}$ µm$^{-1}$)', fontweight='bold', fontsize=18)
plt.xlabel('Bin diameter (µm)', fontweight='bold', fontsize=18)
plt.yscale('log')
plt.ylim(10**-4, 10**0)
plt.title('Below Cloud Base January - June 2022\nFitted Dry Size Distributions Binned by Average Windspeed (CDP)', fontweight='bold', fontsize=16)
plt.legend(title=r"Wind speed bins (m s$^{-1}$)", title_fontsize=15, fontsize=13, frameon=True, prop={'weight': 'bold'})
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
# %%

table_df_CDP = pd.DataFrame.from_dict(fit_results_CDP, orient='index')

# Add wind speed bin ranges and format column names
table_df_CDP.insert(0, "Wind Speed Bin (m/s)", [f"{windspeed_bins[idx][0]} - {windspeed_bins[idx][1]}" for idx in table_df_CDP.index])
table_df_CDP.rename(columns={
    "avg_windspeed": "Avg. Wind Speed (m/s)",
    "n0": "n₀ (cm⁻³ µm⁻¹)",
    "D": "D (µm)",
    "num_legs": "Number of Legs"
}, inplace=True)

# Format numbers for readability
table_df_CDP["n₀ (cm⁻³ µm⁻¹)"] = table_df_CDP["n₀ (cm⁻³ µm⁻¹)"].apply(lambda x: f"{x:.3e}")
table_df_CDP["D (µm)"] = table_df_CDP["D (µm)"].apply(lambda x: f"{x:.3f}")
table_df_CDP["Avg. Wind Speed (m/s)"] = table_df_CDP["Avg. Wind Speed (m/s)"].apply(lambda x: f"{x:.2f}")

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('tight')
ax.axis('off')

# Create table in the plot
table = ax.table(cellText=table_df_CDP.values, colLabels=table_df_CDP.columns, cellLoc='center', loc='center')

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width([0, 1, 2, 3, 4])

# Add title
plt.title("Below Cloud Base Wind Speed Dry Size Distributions (CDP)", fontsize=12, fontweight="bold")

# Show the table
plt.show()
#%%
#computing regression
# Computing regression for CDP
colors = ['navy', 'orange', 'purple', 'darkgreen']

# Define linear regression function
def linear_model(x, m, b):
    return m * x + b

# Extract average wind speeds and total concentrations from binned distributions
avg_windspeeds_CDP = []
total_concentrations_CDP = []

# Compute bin widths for integration
bin_edges_CDP = np.linspace(2, 10, 10)  # Ensure correct bin edges from 2 to 10 μm
bin_widths_CDP = np.diff(bin_edges_CDP)  # Compute widths between bin edges

# Convert size distributions from cm⁻³ μm⁻¹ to total concentration (cm⁻³)
for idx, (low, high) in enumerate(windspeed_bins):
    if final_grouped_CDP[idx]:  # Use only selected CDP legs
        avg_windspeed_CDP = np.mean(final_mean_windspeeds_CDP[idx])  # Average windspeed for this bin

        # Convert using correct bin widths
        avg_concentration_per_leg_CDP = [np.sum(dist * bin_widths_CDP) for dist in final_grouped_CDP[idx]]
        avg_concentration_CDP = np.mean(avg_concentration_per_leg_CDP)  # Average over all legs in this bin

        avg_windspeeds_CDP.append(avg_windspeed_CDP)
        total_concentrations_CDP.append(avg_concentration_CDP)

# Convert to numpy arrays for fitting
windspeed_values_CDP = np.array(avg_windspeeds_CDP)
total_concentrations_CDP = np.array(total_concentrations_CDP)

# Perform linear regression
popt_CDP, _ = curve_fit(linear_model, windspeed_values_CDP, total_concentrations_CDP)
m_fit_CDP, b_fit_CDP = popt_CDP

# Compute R² value
residuals_CDP = total_concentrations_CDP - linear_model(windspeed_values_CDP, *popt_CDP)
ss_res_CDP = np.sum(residuals_CDP**2)
ss_tot_CDP = np.sum((total_concentrations_CDP - np.mean(total_concentrations_CDP))**2)
r_squared_CDP = 1 - (ss_res_CDP / ss_tot_CDP)

# Plot Wind Speed vs. Total Droplet Concentration for CDP
plt.figure(figsize=(8, 6))
for idx in range(len(windspeed_bins)):
    plt.scatter(windspeed_values_CDP[idx], total_concentrations_CDP[idx], 
                color=colors[idx], s=100, edgecolor='black', zorder=3)

# Fit line
x_fit_CDP = np.linspace(min(windspeed_values_CDP), max(windspeed_values_CDP), 100)
y_fit_CDP = linear_model(x_fit_CDP, *popt_CDP)
plt.plot(x_fit_CDP, y_fit_CDP, 'r-', label=f'Fit: y = {m_fit_CDP:.3f}x + {b_fit_CDP:.3f}, R² = {r_squared_CDP:.2f}')

# Labels and formatting
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=16, fontweight='bold')
plt.ylabel("Total Wind Speed Bin Concentration (cm$^{-3}$)", fontsize=16, fontweight='bold')
plt.title("Wind Speed and Total Concentration Correlation (CDP)", fontsize=16, fontweight='bold')

# Legend
legend_labels_CDP = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[idx], markersize=10, 
                                 label=f"{windspeed_values_CDP[idx]:.1f} m/s") for idx in range(len(windspeed_bins))]

plt.legend(handles=legend_labels_CDP + 
           [plt.Line2D([0], [0], color='red', label=f'Fit: y = {m_fit_CDP:.3f}x + {b_fit_CDP:.3f}, R² = {r_squared_CDP:.2f}')], 
           title="Wind Speed Bins", title_fontsize=14, fontsize=14)

plt.tight_layout()
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.show()

# Print regression results
print(f"Slope (m): {m_fit_CDP:.3f}")
print(f"Intercept (b): {b_fit_CDP:.3f}")
print(f"R² value: {r_squared_CDP:.2f}")

# %%
# Computing regression for CDP
colors = ['navy', 'orange', 'purple', 'darkgreen']

# Define linear regression function
def linear_model(x, m, b):
    return m * x + b

# Extract average wind speeds and total concentrations from binned distributions
avg_windspeeds_CDP = []
total_concentrations_CDP = []

# Ensure bin edges match the common_bins used for integration
bin_edges_CDP = np.linspace(2, 10, len(common_bins) + 1)  # +1 ensures correct edges
bin_widths_CDP = np.diff(bin_edges_CDP)  # Compute widths between bin edges

# Convert size distributions from cm⁻³ μm⁻¹ to total concentration (cm⁻³)
for idx, (low, high) in enumerate(windspeed_bins):
    if final_grouped_CDP[idx]:  # Use only selected CDP legs
        avg_windspeed_CDP = np.mean(final_mean_windspeeds_CDP[idx])  # Average windspeed for this bin

        # Convert using correct bin widths (ensuring dimensions match)
        avg_concentration_per_leg_CDP = [np.sum(dist[:len(bin_widths_CDP)] * bin_widths_CDP) for dist in final_grouped_CDP[idx]]
        avg_concentration_CDP = np.mean(avg_concentration_per_leg_CDP)  # Average over all legs in this bin

        avg_windspeeds_CDP.append(avg_windspeed_CDP)
        total_concentrations_CDP.append(avg_concentration_CDP)

# Convert to numpy arrays for fitting
windspeed_values_CDP = np.array(avg_windspeeds_CDP)
total_concentrations_CDP = np.array(total_concentrations_CDP)

# Perform linear regression
popt_CDP, _ = curve_fit(linear_model, windspeed_values_CDP, total_concentrations_CDP)
m_fit_CDP, b_fit_CDP = popt_CDP

# Compute R² value
residuals_CDP = total_concentrations_CDP - linear_model(windspeed_values_CDP, *popt_CDP)
ss_res_CDP = np.sum(residuals_CDP**2)
ss_tot_CDP = np.sum((total_concentrations_CDP - np.mean(total_concentrations_CDP))**2)
r_squared_CDP = 1 - (ss_res_CDP / ss_tot_CDP)

# Plot Wind Speed vs. Total Droplet Concentration for CDP
plt.figure(figsize=(8, 6))
for idx in range(len(windspeed_bins)):
    plt.scatter(windspeed_values_CDP[idx], total_concentrations_CDP[idx], 
                color=colors[idx], s=100, edgecolor='black', zorder=3)

# Fit line
x_fit_CDP = np.linspace(min(windspeed_values_CDP), max(windspeed_values_CDP), 100)
y_fit_CDP = linear_model(x_fit_CDP, *popt_CDP)
plt.plot(x_fit_CDP, y_fit_CDP, 'r-', label=f'Fit: y = {m_fit_CDP:.3f}x + {b_fit_CDP:.3f}, R² = {r_squared_CDP:.2f}')

# Labels and formatting
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=16, fontweight='bold')
plt.ylabel("Total Wind Speed Bin Concentration (cm$^{-3}$)", fontsize=16, fontweight='bold')
plt.title("Wind Speed and Total Concentration Correlation (CDP)", fontsize=16, fontweight='bold')

# Legend
legend_labels_CDP = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[idx], markersize=10, 
                                 label=f"{windspeed_values_CDP[idx]:.1f} m/s") for idx in range(len(windspeed_bins))]

plt.legend(handles=legend_labels_CDP + 
           [plt.Line2D([0], [0], color='red', label=f'Fit: y = {m_fit_CDP:.3f}x + {b_fit_CDP:.3f}, R² = {r_squared_CDP:.2f}')], 
           title="Wind Speed Bins", title_fontsize=14, fontsize=14)

plt.tight_layout()
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.show()

# Print regression results
print(f"Slope (m): {m_fit_CDP:.3f}")
print(f"Intercept (b): {b_fit_CDP:.3f}")
print(f"R² value: {r_squared_CDP:.2f}")

# %%
# Define the exponential function
def fit_function(x, n0, D):
    return n0 * np.exp(-x / D)

# Define windspeed colors (same for CAS and CDP)
windspeed_colors = ['blue', 'orange', 'green', 'red']

plt.figure(figsize=(10, 8))

# Plot CAS Fitted Distributions (Solid Lines)
for idx, (low, high) in enumerate(windspeed_bins):
    if idx in fit_results:
        n0_fit_CAS, D_fit_CAS = fit_results[idx]['n0'], fit_results[idx]['D']
        avg_windspeed_CAS = fit_results[idx]['avg_windspeed']
        num_legs_CAS = fit_results[idx]['num_legs']

        x_fit = np.linspace(min(common_bins), max(common_bins), 10)
        y_fit_CAS = fit_function(x_fit, n0_fit_CAS, D_fit_CAS)

        plt.plot(x_fit, y_fit_CAS, color=windspeed_colors[idx], linewidth=3, linestyle='-',
                 label=f"CAS {avg_windspeed_CAS:.1f} m/s, n={num_legs_CAS}")

# Plot CDP Fitted Distributions (Dashed Lines)
for idx, (low, high) in enumerate(windspeed_bins):
    if idx in fit_results_CDP:
        n0_fit_CDP, D_fit_CDP = fit_results_CDP[idx]['n0'], fit_results_CDP[idx]['D']
        avg_windspeed_CDP = fit_results_CDP[idx]['avg_windspeed']
        num_legs_CDP = fit_results_CDP[idx]['num_legs']

        x_fit = np.linspace(min(common_bins), max(common_bins), 10)
        y_fit_CDP = fit_function(x_fit, n0_fit_CDP, D_fit_CDP)

        plt.plot(x_fit, y_fit_CDP, color=windspeed_colors[idx], linewidth=3, linestyle='--',
                 label=f"CDP {avg_windspeed_CDP:.1f} m/s, n={num_legs_CDP}")

# Formatting
plt.ylabel(r'Number concentration (cm$^{-3}$ µm$^{-1}$)', fontweight='bold', fontsize=18)
plt.xlabel('Bin diameter (µm)', fontweight='bold', fontsize=18)
plt.yscale('log')
plt.ylim(10**-4, 10**0)
plt.title('CAS vs. CDP: Fitted Dry Size Distributions Binned by Average Windspeed', fontweight='bold', fontsize=18)

# Legend
plt.legend(title=r"Wind speed bins (m s$^{-1}$)", title_fontsize=15, fontsize=13, frameon=True, prop={'weight': 'bold'})

plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# %%
