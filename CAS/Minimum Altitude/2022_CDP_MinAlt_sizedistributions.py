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
master_CDP_Min = []
leg_info_CDP = []

for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict_CDP = leg_data[i]

    flight_date = leg_dict_CDP['Date']
    Min_start = leg_dict_CDP['LegIndex_06']['StartTimes']
    Min_stop = leg_dict_CDP['LegIndex_06']['StopTimes']

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

    total_Min_means_CDP = []

    for k in range(len(Min_start)):
        start20 = Min_start[k]
        end20 = Min_stop[k]

        # Find indices in the BCB range for CDP and TwoDS
        CDP_indices_in_range = np.where((CDP_times >= start20) & (CDP_times <= end20))[0]
        TwoDS_indices_in_range = np.where((TwoDS_times >= start20) & (TwoDS_times <= end20))[0]

        if CDP_indices_in_range.size > 0 and TwoDS_indices_in_range.size > 0:
            data_labels_CDP = []
            Min_means_CDP = []

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
                Min_means_CDP.append(bin_values_CDP)

            if Min_means_CDP:
                total_Min_means_CDP.append(Min_means_CDP)

            leg_info_CDP.append({
                'Date': date,
                'Min_start': start20,
                'Min_stop': end20,
                'Data_Labels': data_labels_CDP,
            })

    master_CDP_Min.append(total_Min_means_CDP)

# Print leg_info or use it as needed
for leg in leg_info_CDP:
    print(f"Date: {leg['Date']}, Start: {leg['Min_start']}, Stop: {leg['Min_stop']}, Data Labels: {leg['Data_Labels']}")
#%%
#Double check the number of legs associated with each date to compare with the CAS legs 

# Count the number of legs for each date
leg_count = Counter([leg['Date'] for leg in leg_info_CDP])

# Print the count for each date
print("Number of legs associated with each date:")
for date, count in sorted(leg_count.items()):
    print(f"Date: {date}, Number of Legs: {count}")
#%%
master_CDP_Min = []
leg_info_CDP = []

for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict_CDP = leg_data[i]

    flight_date = leg_dict_CDP['Date']
    Min_start = leg_dict_CDP['LegIndex_06']['StartTimes']
    Min_stop = leg_dict_CDP['LegIndex_06']['StopTimes']

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

    total_Min_means_CDP = []

    for k in range(len(Min_start)):
        start20 = Min_start[k]
        end20 = Min_stop[k]

        bin_means_CDP = {f'Bin{bin_label:02d}_Y_mean': [] for bin_label in range(30)}
        bin_means_CDP.update({f'Bin{bin_label:02d}_N_mean': [] for bin_label in range(30)})
        bin_means_CDP.update({'Date': date, 'Min_start': start20, 'Min_stop': end20})

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

        total_Min_means_CDP.append(bin_means_CDP)

    master_CDP_Min.append(total_Min_means_CDP)

# Print or use master_CDP_BCB as needed
for item in master_CDP_Min:
    for bin_means_CDP in item:
        print(f"Date: {bin_means_CDP['Date']}, Start: {bin_means_CDP['Min_start']}, Stop: {bin_means_CDP['Min_stop']}")
        for bin_label in range(30):
            for label in ['Y', 'N']:
                bin_key = f'Bin{bin_label:02d}_{label}_mean'
                print(f"   {bin_key}: {bin_means_CDP[bin_key]}")
#%%
Y_CDP_calc_Min = []
N_CDP_calc_Min = []

for flight_data in master_CDP_Min:
    for bin_means_CDP in flight_data:
        Y_calc = {'Date': bin_means_CDP['Date'], 'Min_start': bin_means_CDP['Min_start'], 'Min_stop': bin_means_CDP['Min_stop']}
        N_calc = {'Date': bin_means_CDP['Date'], 'Min_start': bin_means_CDP['Min_start'], 'Min_stop': bin_means_CDP['Min_stop']}
        
        for bin_label in range(0, 30):
            bin_key_Y = f'Bin{bin_label:02d}_Y_mean'
            bin_key_N = f'Bin{bin_label:02d}_N_mean'

            Y_calc[bin_key_Y] = np.nanmean(bin_means_CDP[bin_key_Y]) * Logg_CDP[bin_label]
            N_calc[bin_key_N] = np.nanmean(bin_means_CDP[bin_key_N]) * Logg_CDP[bin_label]

        Y_CDP_calc_Min.append(Y_calc)
        N_CDP_calc_Min.append(N_calc)

# %%
# Ensure Y_BCB_calc exists before filtering
if 'Y_Min_calc' in globals():
    # Create a set of unique (Date, BCB_start, BCB_stop) identifiers from Y_BCB_calc
    Min_leg_keys = {(entry['Date'], entry['Min_start'], entry['Min_stop']) for entry in Y_CDP_calc_Min}

    # Filter Y_CDP_calc and N_CDP_calc to retain only legs present in Y_BCB_calc
    filtered_Y_CDP_calc = [entry for entry in Y_CDP_calc_Min if (entry['Date'], entry['Min_start'], entry['Min_stop']) in Min_leg_keys]
    filtered_N_CDP_calc = [entry for entry in N_CDP_calc_Min if (entry['Date'], entry['Min_start'], entry['Min_stop']) in Min_leg_keys]

    # Print verification
    print(f"Original Y_CDP_calc legs: {len(Y_CDP_calc_Min)}")
    print(f"Filtered Y_CDP_calc legs: {len(filtered_Y_CDP_calc)}")
    print(f"Original N_CDP_calc legs: {len(N_CDP_calc_Min)}")
    print(f"Filtered N_CDP_calc legs: {len(filtered_N_CDP_calc)}")

    # Assign back the filtered lists if needed
    Y_CDP_calc_Min = filtered_Y_CDP_calc
    N_CDP_calc_Min = filtered_N_CDP_calc
else:
    print("Error: Y_BCB_calc is not defined. Make sure it is loaded before running this script.")


# %%
#ambient size distribution
bin_center_CDP=np.array(bin_center_CDP)

# Plot Raw CDP Size Distributions
plt.figure(figsize=(8, 6))

for entry in Y_CDP_calc_Min:
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

for entry in Y_CDP_calc_Min:
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
for entry in Y_CDP_calc_Min:
    bin_means_CDP = np.array([entry.get(f'Bin{i:02d}_Y_mean', np.nan) for i in range(0, 30)], dtype=float)

    # Mask: Remove NaNs and zero values
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
plt.title("CDP Average Ambient Minimum Altitude Size Distribution\n January - June 2022", fontsize=14, fontweight="bold")
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
for entry in Y_CDP_calc_Min:
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
            'Min_start': entry.get('Min_start', np.nan),
            'Min_stop': entry.get('Min_stop', np.nan),
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
plt.title("Minimum Altitude January - June 2022\n Exponential Fit to CDP Size Distributions", fontsize=14, fontweight="bold")
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
for entry in Y_CDP_calc_Min:
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
        'Min_start': entry.get('Min_start', np.nan),
        'Min_stop': entry.get('Min_stop', np.nan),
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
plt.title("Minimum Altitude January - June 2022\n Exponential Fit to CDP Size Distributions (≤10 µm)", fontsize=14, fontweight="bold")
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



Y_BCB_calc_cm3_CDP_Min = []
N_BCB_calc_cm3_CDP_Min = []

for flight_data in master_CDP_Min:
    for bin_means_CDP in flight_data:  # ✅ Ensure we correctly reference `bin_means_CDP`
        # Create dictionaries for each flight leg
        Y_calc_CDP_Min = {'Date': bin_means_CDP['Date'], 'Min_start': bin_means_CDP['Min_start'], 'Min_stop': bin_means_CDP['Min_stop']}
        N_calc_CDP_Min = {'Date': bin_means_CDP['Date'], 'Min_start': bin_means_CDP['Min_start'], 'Min_stop': bin_means_CDP['Min_stop']}

        for bin_label in range(0, 30):
            bin_key_Y = f'Bin{bin_label:02d}_Y_mean'  # ✅ Ensuring zero-padded bin keys
            bin_key_N = f'Bin{bin_label:02d}_N_mean'

            # ✅ Use `.get()` to avoid KeyErrors if a bin is missing
            Y_bin_value = np.nanmean(bin_means_CDP.get(bin_key_Y, np.nan))
            N_bin_value = np.nanmean(bin_means_CDP.get(bin_key_N, np.nan))

            # ✅ Multiply by logarithmic bin width factor
            Y_calc_CDP_Min[bin_key_Y] = Y_bin_value * bin_log_CDP[bin_label]
            N_calc_CDP_Min[bin_key_N] = N_bin_value * bin_log_CDP[bin_label]

        # ✅ Append processed data to lists
        Y_BCB_calc_cm3_CDP_Min.append(Y_calc_CDP_Min)
        N_BCB_calc_cm3_CDP_Min.append(N_calc_CDP_Min)

# ✅ Check results
print(f"Processed {len(Y_BCB_calc_cm3_CDP_Min)} flight legs for CDP.")

# %%
#Calculating total number concentration 
# Dictionary to store total concentrations for each leg
total_concentration_cm3_CDP_Min = []

# Process total concentration for each leg
for entry in Y_BCB_calc_cm3_CDP_Min:
    # ✅ Use .get() to avoid KeyErrors if a bin is missing
    total_Y_concentration_Min = np.nansum([entry.get(f'Bin{i:02d}_Y_mean', 0) for i in range(0, 30)])  # ✅ Default to 0 if missing

    total_concentration_cm3_CDP_Min.append({
        'Date': entry['Date'],
        'Min_start': entry['Min_start'],
        'Min_stop': entry['Min_stop'],
        'Total_Y_Concentration_cm3': total_Y_concentration_Min
    })

# ✅ Check results
print(f"Processed {len(total_concentration_cm3_CDP_Min)} flight legs for total CDP concentration.")

# %%
total_Y_concentrations_CDP_Min = [entry['Total_Y_Concentration_cm3'] for entry in total_concentration_cm3_CDP_Min]

# Remove NaN values if any
total_Y_concentrations_CDP_Min = [conc for conc in total_Y_concentrations_CDP_Min if not np.isnan(conc)]

# Calculate mean total concentration
mean_total_concentration_CDP = np.mean(total_Y_concentrations_CDP_Min)

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
plt.xlabel("Corrected Wind Speed (m s$^{-1}$)", fontsize=19, fontweight='bold')
plt.ylabel("Total Concentration (cm$^{-3}$)", fontsize=19, fontweight='bold')
plt.title("CDP Below Cloud Base January - June 2022", fontsize=19, fontweight='bold')

# ✅ Optional: Add reference lines if applicable
# plt.axhline(0.05, color='red', linestyle='--', label="Reference Min (0.05 cm⁻³)")
# plt.axhline(0.3, color='blue', linestyle='--', label="Reference Max (0.3 cm⁻³)")
plt.legend(fontsize=16, title_fontsize=21, loc='upper right', frameon=True)
plt.xticks(fontweight='bold', fontsize=19)
plt.yticks(fontweight='bold', fontsize=19)  
plt.tight_layout()
plt.show()
#%%
#adding an R value 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit

# ✅ Compute Pearson & Spearman correlation coefficients
pearson_corr, pearson_p = pearsonr(matched_wind_speeds, matched_total_concentrations)
spearman_corr, spearman_p = spearmanr(matched_wind_speeds, matched_total_concentrations)

print(f"Pearson Correlation: {pearson_corr:.3f} (p = {pearson_p:.3e})")
print(f"Spearman Correlation: {spearman_corr:.3f} (p = {spearman_p:.3e})")

# ✅ Define a linear function for regression
def linear_model(x, m, b):
    return m * x + b

# ✅ Fit a linear regression model
popt, _ = curve_fit(linear_model, matched_wind_speeds, matched_total_concentrations)
m_fit, b_fit = popt  # Extract slope and intercept

# ✅ Compute R² value
residuals = matched_total_concentrations - linear_model(matched_wind_speeds, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((matched_total_concentrations - np.mean(matched_total_concentrations))**2)
r_squared = 1 - (ss_res / ss_tot)

print(f"Linear Fit: y = {m_fit:.3f}x + {b_fit:.3f} (R² = {r_squared:.3f}, R = {pearson_corr:.3f})")

# ✅ Scatter plot with linear trend line
plt.figure(figsize=(8, 6))
plt.scatter(matched_wind_speeds, matched_total_concentrations, edgecolors='black', facecolors='none', marker='o')
plt.plot(matched_wind_speeds, linear_model(matched_wind_speeds, *popt), color='red', linewidth=2, 
         label=f'Fit: y = {m_fit:.3f}x + {b_fit:.3f}\nR² = {r_squared:.2f}, R = {pearson_corr:.2f}')

# ✅ Formatting
plt.xlabel("10m Wind Speed (m s$^{-1}$)", fontsize=19, fontweight='bold')
plt.ylabel("Total Concentration (cm$^{-3}$)", fontsize=19, fontweight='bold')
plt.title("CDP Below Cloud Base January - June 2022", fontsize=19, fontweight='bold')

# ✅ Legend: Only show fit equation, R², and R
plt.legend(fontsize=16, title_fontsize=21, loc='upper right', frameon=True)

# ✅ Final Formatting
plt.xticks(fontweight='bold', fontsize=19)
plt.yticks(fontweight='bold', fontsize=19)  
plt.tight_layout()
plt.show()

#%%
master_Min_RH = []

for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict = leg_data[i]

    flight_date = leg_dict['Date']  # Get date of flight from dictionary 
    Min_start = leg_dict['LegIndex_06']['StartTimes']
    Min_stop = leg_dict['LegIndex_06']['StopTimes']

    rh_flight = h20[i]
    times_rh = rh_flight.Time_Start.values
    rh_values = rh_flight.RHw_DLH.values

    all_Min= []

    for j in range(len(Min_start)):
        start = int(Min_start[j])
        end = int(Min_stop[j])

        rh_times = {
            'Date': date,
            'Min_start': start,
            'Min_stop': end,
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
        all_Min.append(rh_times)  # List that contains all the BCB wind/alt mean dictionaries for 1 flight

    master_Min_RH.append(all_Min)

# Step 2: Replace -999 with NaN in master_BCB_RH
for flight in master_Min_RH:
    for leg in flight:
        rh_mean_list = leg['Rh_mean']
        leg['Rh_mean'] = [np.nan if value <=0 else value for value in rh_mean_list]
#%%
#average relative humidity
relativy_humidity_values = []
#%%
#for only the legs present after LWC filtration and master_BCB_exponential 
#Filtering master_BCB_RH because some legs may not contain humidity data. So we need to remove those legs. 
#Do not linearly interpolate the data. We are working with observations and linearly interpolating the data would no longer represent observation. 
# Extract the (date, BCB_start, BCB_stop) from master_BCB_exponential
# Extract the (date, BCB_start, BCB_stop) from Y_BCB_calc
date_leg_set = set()

for entry in Y_CDP_calc_Min:  # Iterate over the list directly
    date = entry['Date']
    Min_start = entry.get('Min_start', np.nan)
    Min_stop = entry.get('Min_stop', np.nan)
    date_leg_set.add((date, Min_start, Min_stop))


# Filter master_BCB_RH
filtered_master_Min_RH_CDP = []

for flight in master_Min_RH:
    filtered_legs = []
    for leg in flight:
        date = leg['Date']
        Min_start = leg['Min_start']
        Min_stop = leg['Min_stop']
        if (date, Min_start, Min_stop) in date_leg_set:
            filtered_legs.append(leg)
    if filtered_legs:
        filtered_master_Min_RH_CDP.append(filtered_legs)
# %%
#Make sure the leg counts 
# Flatten the list of lists and count the total number of entries
total_entries_filtered_master_Min_RH_CDP = sum(len(legs) for legs in filtered_master_Min_RH_CDP)
print(f"Total entries in filtered_master_BCB_RH: {total_entries_filtered_master_Min_RH_CDP}")
# %%
##Obtaining g(RH) = [1.7 / (1-RH)]^0.31 for all mean RH values 
## ie for every leg 

master_Min_gRH_CDP = []

# Iterate over each flight in master_BCB_RH
for flight in master_Min_RH:
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
    
    master_Min_gRH_CDP.append(flight_gRH)
#%%
#only the grh from filtered_master_BCB_RH
filtered_master_Min_gRH_CDP = []

# Iterate over each flight in master_BCB_RH
for flight in filtered_master_Min_RH_CDP:
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
    
    filtered_master_Min_gRH_CDP.append(flight_gRH)
#%%
total_entries_filtered_master_Min_gRH_CDP= sum(len(legs) for legs in filtered_master_Min_gRH_CDP)
print(f"Total entries in filtered_master_BCB_gRH: {total_entries_filtered_master_Min_gRH_CDP}")
#%%
#Histogram of gRH values
gRH_values_CDP = [
    leg['gRh_mean'][0] for flight in filtered_master_Min_gRH_CDP for leg in flight if not np.isnan(leg['gRh_mean'][0])
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

filtered_master_Min_interceptdry_dict_CDP = {}

# Flatten master_min_gRH if it's a list of lists
if isinstance(filtered_master_Min_gRH_CDP[0], list):
    filtered_master_Min_gRH_CDP = [item for sublist in filtered_master_Min_gRH_CDP for item in sublist]

# Loop through each entry in filtered_master_BCB_gRH
for entry in filtered_master_Min_gRH_CDP:
    date = entry['Date']
    Min_start = entry['Min_start']
    Min_stop = entry['Min_stop']
    gRh_mean = entry['gRh_mean'][0]  # Assuming gRh_mean is a list with one value
    Rh_mean = entry['Rh_mean'][0]  # Assuming Rh_mean is a list with one value

    # Skip entries with invalid Rh_mean values
    if Rh_mean < 0:
        continue

    # Create a unique key for this entry
    key = (date, Min_start, Min_stop)

    # Find corresponding ambient exponential parameters
    if key in CDP_fits_10:
        n0 = CDP_fits_10[key]['Intercept_n0']  # Extract n0 from ambient_fits
        
        # Calculate dry intercept
        dryintercept = n0 / gRh_mean if gRh_mean > 0 else np.nan

        # Store the result in the dictionary
        filtered_master_Min_interceptdry_dict_CDP[key] = {
            'Date': date,
            'Min_start': Min_start,
            'Min_stop': Min_stop,
            'Rh_mean': entry['Rh_mean'],
            'gRh_mean': entry['gRh_mean'],
            'dry intercept': dryintercept
        }

# Convert dictionary to a list
filtered_master_Min_dryintercept_CDP = list(filtered_master_Min_interceptdry_dict_CDP.values())

print(f"Length of filtered_master_Min_dryintercept: {len(filtered_master_Min_dryintercept_CDP)}")

# Histogram of dry intercept values
dryintercept_values_CDP = [
    leg['dry intercept'] for leg in filtered_master_Min_dryintercept_CDP if not np.isnan(leg['dry intercept'])
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
dry_intercept_values_CDP = [leg['dry intercept'] for leg in filtered_master_Min_dryintercept_CDP if not np.isnan(leg['dry intercept'])]

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


# Initialize the list for storing CDP dry size distributions
filtered_master_Min_ddry_CDP = []

# Iterate over each entry in filtered_master_BCB_gRH_CDP
for entry in filtered_master_Min_gRH_CDP:
    date = entry['Date']
    Min_start = entry['Min_start']
    Min_stop = entry['Min_stop']
    gRh_mean = entry['gRh_mean'][0]  # Assuming it's stored as a list

    # Compute dry bin centers for CDP
    if gRh_mean > 0:
        ddry_values_CDP = np.array([D_amb / gRh_mean for D_amb in bin_center_CDP])
    else:
        ddry_values_CDP = np.full(len(bin_center_CDP), np.nan)
        print(f"Skipping division for {date}, {Min_start}-{Min_stop} due to invalid gRh_mean.")

    # Compute bin widths for dry size distribution (∆Ddry = Ddry[i+1] - Ddry[i])
    ddry_bin_widths_CDP = np.diff(ddry_values_CDP, append=np.nan)  # Append NaN to keep array size consistent

    # Find the corresponding ambient size distribution in CDP
    raw_concentrations_CDP = next(
        (leg for leg in Y_CDP_calc_Min if leg['Date'] == date and leg['Min_start'] == Min_start and leg['Min_stop'] == Min_stop),
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
        print(f"Missing raw size distribution for {date}, {Min_start}-{Min_stop}")

    # Store the raw dry size distribution for CDP
    filtered_master_Min_ddry_CDP.append({
        'Date': date,
        'BCB_start': Min_start,
        'BCB_stop': Min_stop,
        'ddry': ddry_values_CDP.tolist(),
        'dN/dDdry': dN_dD_dry_CDP.tolist(),
        'ddry_bin_widths': ddry_bin_widths_CDP.tolist(),  # Store bin widths separately
        'gRh_mean': gRh_mean
    })

print(f"Length of filtered_master_Min_ddry_CDP: {len(filtered_master_Min_ddry_CDP)}")

# %%
#Plotting raw dry size distributions

# Define common bin centers for interpolation (adjust as needed)
common_bins_CDP = np.linspace(2, 25, 35)  

plt.figure(figsize=(8, 6))

# Loop through each dry size distribution in CDP dataset
for entry in filtered_master_Min_ddry_CDP:
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
plt.title("Minimum Altitude January - June 2022\n Raw Dry Size Distributions (CDP)", fontsize=14, fontweight="bold")
plt.show()

# %%
#removing the 0s

# Define common bin centers for interpolation
common_bins_CDP = np.linspace(2, 25, 35)  # Adjust bin range and count as needed

plt.figure(figsize=(8, 6))

# Loop through each dry size distribution in CDP dataset
for entry in filtered_master_Min_ddry_CDP:
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
plt.title("Minimum Altitude January - June 2022\n Raw Dry Size Distributions (CDP)", fontsize=14, fontweight="bold")
plt.show()

# %%
#Averaging the dry size distributions

# Define common bin centers for interpolation
common_bins_CDP = np.linspace(2, 25, 35)  # Adjust bin range and count as needed

# Initialize sum and count arrays for averaging
sum_interpolated_dN_dD_dry_CDP = np.zeros_like(common_bins_CDP, dtype=float)
count_interpolated_dN_dD_dry_CDP = np.zeros_like(common_bins_CDP, dtype=int)

# Loop through each dry size distribution in CDP dataset
for entry in filtered_master_Min_ddry_CDP:
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
plt.plot(common_bins_CDP, average_dN_dD_dry_CDP, color='green', linewidth=2, label='Average Dry Size Distribution (CDP)')

# Formatting and labels
plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"CDP Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("CDP Average Minimum Altitude Dry Size Distribution \n January - June 2022", fontsize=14, fontweight="bold")
# Show plot
plt.show()
#%%
#exponential fit to the averaged dry size distribution

# Define the exponential function
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

# Filter bins for fitting (only include x ≤ 10 μm)
valid_fit_indices_CDP = common_bins_CDP <= 10
x_fit_CDP = common_bins_CDP[valid_fit_indices_CDP]
y_fit_CDP = average_dN_dD_dry_CDP[valid_fit_indices_CDP]

# Remove NaNs or zeros to avoid fitting issues
valid_data_indices_CDP = ~np.isnan(y_fit_CDP) & (y_fit_CDP > 0)
x_fit_CDP = x_fit_CDP[valid_data_indices_CDP]
y_fit_CDP = y_fit_CDP[valid_data_indices_CDP]

# Perform curve fitting
try:
    popt_CDP, pcov_CDP = curve_fit(exponential, x_fit_CDP, y_fit_CDP, p0=(1e-2, 2))  # Initial guess: (n0=0.01, D=2 μm)
    n0_fit_CDP, D_fit_CDP = popt_CDP  # Extract fitted parameters
except RuntimeError:
    print("Exponential fit failed for CDP.")
    n0_fit_CDP, D_fit_CDP = None, None

# Plot the averaged dry size distribution
plt.figure(figsize=(8, 6))
plt.plot(common_bins_CDP, average_dN_dD_dry_CDP, color='green', linewidth=2, label='Average Dry Size Distribution (CDP)')

# Overlay the exponential fit if successful
if n0_fit_CDP is not None and D_fit_CDP is not None:
    plt.plot(x_fit_CDP, exponential(x_fit_CDP, *popt_CDP), 'r--', linewidth=2, 
             label=f'Exponential Fit: $N_0$={n0_fit_CDP:.2e}, $D$={D_fit_CDP:.2f} μm')

# Formatting and labels
plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"CDP Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("CDP Average Minimum Altitude Dry Size Distribution \n January - June 2022", fontsize=14, fontweight="bold")
plt.legend()

# Show plot
plt.show()

# Print the fit parameters
if n0_fit_CDP is not None and D_fit_CDP is not None:
    print(f"Fitted Parameters (CDP): N_0 = {n0_fit_CDP:.3e}, D = {D_fit_CDP:.3f} μm")
#%%
common_bins = np.linspace(2, 25, 35) 
#%%
# # Adjust bin range and count as needed

# Define the exponential function
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

### --- CAS Data Fitting --- ###
# Filter bins for fitting (only include x ≤ 10 μm)
valid_fit_indices_CAS = common_bins <= 10
x_fit_CAS = common_bins[valid_fit_indices_CAS]
y_fit_CAS = average_dN_dD_dry[valid_fit_indices_CAS]

# Remove NaNs or zeros to avoid fitting issues
valid_data_indices_CAS = ~np.isnan(y_fit_CAS) & (y_fit_CAS > 0)
x_fit_CAS = x_fit_CAS[valid_data_indices_CAS]
y_fit_CAS = y_fit_CAS[valid_data_indices_CAS]

# Perform curve fitting for CAS
try:
    popt_CAS, _ = curve_fit(exponential, x_fit_CAS, y_fit_CAS, p0=(1e-2, 2))  
    n0_fit_CAS, D_fit_CAS = popt_CAS
except RuntimeError:
    print("Exponential fit failed for CAS.")
    n0_fit_CAS, D_fit_CAS = None, None

### --- CDP Data Fitting --- ###
# Filter bins for fitting (only include x ≤ 10 μm)
valid_fit_indices_CDP = common_bins_CDP <= 10
x_fit_CDP = common_bins_CDP[valid_fit_indices_CDP]
y_fit_CDP = average_dN_dD_dry_CDP[valid_fit_indices_CDP]

# Remove NaNs or zeros to avoid fitting issues
valid_data_indices_CDP = ~np.isnan(y_fit_CDP) & (y_fit_CDP > 0)
x_fit_CDP = x_fit_CDP[valid_data_indices_CDP]
y_fit_CDP = y_fit_CDP[valid_data_indices_CDP]

# Perform curve fitting for CDP
try:
    popt_CDP, _ = curve_fit(exponential, x_fit_CDP, y_fit_CDP, p0=(1e-2, 2))  
    n0_fit_CDP, D_fit_CDP = popt_CDP
except RuntimeError:
    print("Exponential fit failed for CDP.")
    n0_fit_CDP, D_fit_CDP = None, None

### --- Plot Both Distributions and Fits --- ###
plt.figure(figsize=(8, 6))

# Plot CAS distribution
plt.plot(common_bins, average_dN_dD_dry, color='red', linewidth=3.5, label='CAS Average Dry Size Distribution')

# Plot CDP distribution
plt.plot(common_bins_CDP, average_dN_dD_dry_CDP, color='green', linewidth=3.5, label='CDP Average Dry Size Distribution')

# # Plot CAS exponential fit
# if n0_fit_CAS is not None and D_fit_CAS is not None:
#     plt.plot(x_fit_CAS, exponential(x_fit_CAS, *popt_CAS), 'r--', linewidth=2,
#              label=f'CAS Exp Fit: $N_0$={n0_fit_CAS:.2e}, $D$={D_fit_CAS:.2f} μm')

# # Plot CDP exponential fit
# if n0_fit_CDP is not None and D_fit_CDP is not None:
#     plt.plot(x_fit_CDP, exponential(x_fit_CDP, *popt_CDP), 'b--', linewidth=2,
#              label=f'CDP Exp Fit: $N_0$={n0_fit_CDP:.2e}, $D$={D_fit_CDP:.2f} μm')

# Formatting and labels
plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Average Below Cloud Base Dry Size Distributions\n January - June 2022", fontsize=14, fontweight="bold")
plt.legend()

# Show plot
plt.show()

# Print fitted parameters
if n0_fit_CAS is not None and D_fit_CAS is not None:
    print(f"CAS Fitted Parameters: N_0 = {n0_fit_CAS:.3e}, D = {D_fit_CAS:.3f} μm")

if n0_fit_CDP is not None and D_fit_CDP is not None:
    print(f"CDP Fitted Parameters: N_0 = {n0_fit_CDP:.3e}, D = {D_fit_CDP:.3f} μm")


# %%
#both ambient and dry 

common_bins_CDP = np.linspace(2, 40, 35)  # Adjust bin range as needed

plt.figure(figsize=(8, 6))

# Plot both ambient and dry size distributions for comparison
for entry_ambient, entry_dry in zip(Y_CDP_calc_Min, filtered_master_Min_ddry_CDP):
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

#%%
#Fitting an exponential to the dry 


def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

# Initialize a list to store fitted parameters
dry_exponential_fits_CDP = []

plt.figure(figsize=(8, 6))

# Process and fit each dry size distribution in CDP dataset
for entry in filtered_master_Min_ddry_CDP:
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
            'Min_start': entry['Min_start'],
            'Min_stop': entry['Min_stop'],
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
plt.title("Minimum Altitude January - June 2022\n Exponential Fitted Dry Size Distributions (CDP)", fontsize=14, fontweight="bold")

plt.show()

print(f"Total successful dry exponential fits (CDP): {len(dry_exponential_fits_CDP)}")
#%%

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
dry_slope_10_CDP=[fit['Dry_E_folding_D'] for fit in dry_exponential_fits_10_CDP if not np.isnan(fit['Dry_E_folding_D'])]
# %%
dry_intercept_10_CDP=[fit['Dry_Intercept_n0'] for fit in dry_exponential_fits_10_CDP if not np.isnan(fit['Dry_Intercept_n0'])]  
#%%

#%%
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
        mass_value = calculate_mass(dry_intercept, dry_slope) * 1e9  # Convert kg/m³ to µg/m³
        dry_mass_data_inf_CDP.append({
        'Date': date,
        'BCB_start': entry['BCB_start'],  # Add start time
        'BCB_stop': entry['BCB_stop'],  # Add stop time
        'Dry Slope (D)': dry_slope,
        'Dry Intercept (N0)': dry_intercept,
        'Dry Mass (µg/m³)': mass_value
    })


dry_slopes = np.array([entry['Dry Slope (D)'] for entry in dry_mass_data_inf_CDP])
dry_intercepts = np.array([entry['Dry Intercept (N0)'] for entry in dry_mass_data_inf_CDP])

min_slope_threshold = np.percentile(dry_slopes, 1)  # Remove the lowest 1% of slopes

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
#histogram of mass values
#%%
mass_threshold = 300  # µg/m³


filtered_dry_mass_inf_CDP = [entry for entry in dry_mass_data_inf_CDP if (
    not np.isnan(entry['Dry Slope (D)']) and 
    not np.isnan(entry['Dry Intercept (N0)']) and 
    entry['Dry Mass (µg/m³)'] <= mass_threshold
)]


print(f"Filtered Dry Mass Entries: {len(filtered_dry_mass_inf_CDP)} (after removing masses > {mass_threshold} µg/m³)")

# Extract slope and intercept as NumPy arrays after filtering
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
plt.figure(figsize=(10, 8))
plt.hist(filtered_mass_values_ug_inf_CDP, bins=20, color='blue', edgecolor='black', alpha=0.7)
bins = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 
                 16384, 32768, 65536, 131072])  


plt.xscale('log')  
# plt.yscale('log')
plt.xlabel('Dry Mass (µg/m³)', fontsize=16, fontweight='bold')
plt.ylabel('Frequency', fontsize=16, fontweight='bold')
plt.title('CDP dry mass', fontsize=18, fontweight='bold')
plt.legend(fontsize=14)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.tight_layout()
plt.legend()
plt.show()
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
# windspeed_bins = [(0, 3), (3.001, 6.5), (6.501, 8.5), (8.501, np.inf)]
cas_bin_counts = [78, 174, 62, 54]  # Exact CAS leg counts per bin

# Store binned distributions for CDP
grouped_distributions_CDP = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds_CDP = {i: [] for i in range(len(windspeed_bins))}
cdp_bin_leg_indices = {i: [] for i in range(len(windspeed_bins))}  # Store leg indices

missing_windspeed_count_CDP = 0

# Define common bins (10 bin centers between 2 and 10 µm)
# common_bins = np.linspace(2, 10, 10)
windspeed_bins = [(0, 5), (5.001, 7), (7.001, 9), (9.001, np.inf)]
common_bins=np.linspace(2, 10, 10)

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
#%%
# #Lewis and Schwartz
# windspeed_bins = [(0, 5), (5.001, 7), (7.001, 9), (9.001, np.inf)]
# common_bins=np.linspace(2, 10, 10)

# # # Define wind speed bins
# # windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]

# # Store binned distributions
# grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
# mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# missing_windspeed_count = 0

# # Define common bins (10 bin centers between 2 and 10 µm)
# common_bins=np.linspace(2, 100, 100)

# # Use already fitted exponential size distributions
# for entry in CDP_fits_10:
#     date = entry['Date']
#     BCB_start = entry['BCB_start']
#     BCB_stop = entry['BCB_stop']
    
#     n0 = entry['Dry_Intercept_n0']
#     D = entry['Dry_E_folding_D']

#     # Match windspeed
#     windspeed_entry = df_combined[
#         (df_combined['Date'] == date) & 
#         (df_combined['BCB_start'] == BCB_start) & 
#         (df_combined['BCB_stop'] == BCB_stop)
#     ]

#     if windspeed_entry.empty or np.isnan(n0) or np.isnan(D):
#         missing_windspeed_count += 1
#         continue

#     windspeed = windspeed_entry['Windspeed'].values[0]

#     # Use the already fitted size distribution (DO NOT FIT AGAIN)
#     size_dist = n0 * np.exp(-common_bins / D)  # Use existing (n0, D)

#     # Bin the distribution by windspeed
#     for idx, (low, high) in enumerate(windspeed_bins):
#         if low <= windspeed < high:
#             grouped_distributions[idx].append(size_dist)
#             mean_windspeeds[idx].append(windspeed)
#             break

# # Print summary
# for idx, group in grouped_distributions.items():
#     print(f"Windspeed bin {idx} ({windspeed_bins[idx]} m/s): {len(group)} legs")

# print(f"Total legs with missing windspeed data: {missing_windspeed_count}")

# # Step 3: Plot binned size distributions
# plt.figure(figsize=(10, 8))

# for idx, (low, high) in enumerate(windspeed_bins):
#     if grouped_distributions[idx]:
#         avg_distribution = np.mean(grouped_distributions[idx], axis=0)  # Average size distribution
#         avg_windspeed = np.mean(mean_windspeeds[idx])
#         num_legs = len(grouped_distributions[idx])

#         plt.plot(common_bins, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linewidth=2.5)

# plt.yscale('log')
# plt.ylabel(r"CDP Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=16, fontweight="bold")
# plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=16, fontweight="bold")
# plt.title('CDP Dry Size Distributions Binned by Average Wind Speed', fontweight='bold', fontsize=17)
# plt.legend(title=r"Average wind speed m s$^{-1}$")
# plt.tight_layout()
# plt.ylim(1e-4, 10**0)

# plt.xticks(fontsize=14, fontweight='bold')
# plt.yticks(fontsize=14, fontweight='bold')
# plt.show()

# total_legs = sum(len(group) for group in grouped_distributions.values())
# print(f"Total number of legs plotted: {total_legs}")
#%%
# #trying to fix 

# # Define wind speed bins based on Lewis & Schwartz Fig. 22
# windspeed_bins = [(0, 5), (5.001, 7), (7.001, 9), (9.001, np.inf)]

# # Define common bins (log-spaced to match Lewis & Schwartz)
# common_bins = np.logspace(np.log10(0.1), np.log10(100), 50)  # 50 log-spaced bins
# bin_widths = np.diff(common_bins)  # Compute bin widths

# # Store binned distributions
# grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
# mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# missing_windspeed_count = 0

# # Use already fitted exponential size distributions
# for entry in dry_exponential_fits_10_CDP:
#     date = entry['Date']
#     BCB_start = entry['BCB_start']
#     BCB_stop = entry['BCB_stop']
    
#     n0 = entry['Dry_Intercept_n0']
#     D = entry['Dry_E_folding_D']

#     # Match windspeed
#     windspeed_entry = df_combined[
#         (df_combined['Date'] == date) & 
#         (df_combined['BCB_start'] == BCB_start) & 
#         (df_combined['BCB_stop'] == BCB_stop)
#     ]

#     if windspeed_entry.empty or np.isnan(n0) or np.isnan(D):
#         missing_windspeed_count += 1
#         continue

#     windspeed = windspeed_entry['Windspeed'].values[0]

#     # Use the already fitted size distribution (DO NOT FIT AGAIN)
#     size_dist = n0 * np.exp(-common_bins / D)  # Use existing (n0, D)

#     # Bin the distribution by windspeed
#     for idx, (low, high) in enumerate(windspeed_bins):
#         if low <= windspeed < high:
#             grouped_distributions[idx].append(size_dist)
#             mean_windspeeds[idx].append(windspeed)
#             break

# # Print summary
# for idx, group in grouped_distributions.items():
#     print(f"Windspeed bin {idx} ({windspeed_bins[idx]} m/s): {len(group)} legs")

# print(f"Total legs with missing windspeed data: {missing_windspeed_count}")

# # Step 3: Plot binned size distributions
# plt.figure(figsize=(10, 8))

# for idx, (low, high) in enumerate(windspeed_bins):
#     if grouped_distributions[idx]:
#         avg_distribution = np.mean(grouped_distributions[idx], axis=0)  # Average size distribution
#         avg_distribution_total = avg_distribution[:-1] * bin_widths  # Convert to cm⁻³

#         avg_windspeed = np.mean(mean_windspeeds[idx])
#         num_legs = len(grouped_distributions[idx])

#         plt.plot(common_bins[:-1], avg_distribution_total, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linewidth=2.5)

# # Set log scale for x-axis (diameter) and y-axis (concentration)
# plt.xscale('log')  # Match Lewis & Schwartz log scale
# plt.yscale('log')
# plt.ylabel(r"Number Concentration (cm$^{-3}$)", fontsize=16, fontweight="bold")  
# plt.xlabel("Dry Bin Centers Diameter (µm)", fontsize=16, fontweight="bold")
# plt.title('Dry Size Distributions Binned by Average Wind Speed', fontweight='bold', fontsize=17)
# plt.legend(title=r"Average wind speed (m s$^{-1}$)")
# plt.tight_layout()
# plt.ylim(1e-4, 10**2)  # Adjust y-axis to match Lewis & Schwartz

# plt.xticks(fontsize=14, fontweight='bold')
# plt.yticks(fontsize=14, fontweight='bold')
# plt.show()

# total_legs = sum(len(group) for group in grouped_distributions.values())
# print(f"Total number of legs plotted: {total_legs}")
#%%
# common_bins = np.linspace(2, 100, 100)  # Creates 10 bin centers between 2 and 10 µm

# # Define the exponential function
# def exponential(x, n0, D):
#     return n0 * np.exp(-x / D)


# windspeed_bins = [(0, 5), (5.001, 7), (7.001, 9), (9.001, np.inf)]

# # Store binned distributions
# grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
# mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# missing_windspeed_count = 0

# # Use dry_exponential_fits_10 instead of dry_exponential_fits
# for entry in dry_exponential_fits_10_CDP:
#     date = entry['Date']
#     BCB_start = entry['BCB_start']
#     BCB_stop = entry['BCB_stop']
    
#     n0 = entry['Dry_Intercept_n0']
#     D = entry['Dry_E_folding_D']

#     # Match windspeed
#     windspeed_entry = df_combined[
#         (df_combined['Date'] == date) & 
#         (df_combined['BCB_start'] == BCB_start) & 
#         (df_combined['BCB_stop'] == BCB_stop)
#     ]

#     if windspeed_entry.empty or np.isnan(n0) or np.isnan(D):
#         missing_windspeed_count += 1
#         continue

#     windspeed = windspeed_entry['Windspeed'].values[0]

#     # Generate size distribution from exponential fit
#     ddry_values = np.array(common_bins)
#     size_dist = n0 * np.exp(-ddry_values / D)

#     # Bin the distribution by windspeed
#     for idx, (low, high) in enumerate(windspeed_bins):
#         if low <= windspeed < high:
#             grouped_distributions[idx].append(size_dist)
#             mean_windspeeds[idx].append(windspeed)
#             break

# # Print summary
# for idx, group in grouped_distributions.items():
#     print(f"Windspeed bin {idx} ({windspeed_bins[idx]} m/s): {len(group)} legs")

# print(f"Total legs with missing windspeed data: {missing_windspeed_count}")

# # Step 3: Plot binned size distributions
# plt.figure(figsize=(10, 8))

# for idx, (low, high) in enumerate(windspeed_bins):
#     if grouped_distributions[idx]:
#         avg_distribution = np.mean(grouped_distributions[idx], axis=0)
#         avg_windspeed = np.mean(mean_windspeeds[idx])
#         num_legs = len(grouped_distributions[idx])

#         plt.plot(common_bins, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linewidth=3)

# plt.yscale('log')
# plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=16, fontweight="bold")
# plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=16, fontweight="bold")
# plt.title('Dry Size Distributions Binned by Average Wind Speed', fontweight='bold', fontsize=17)
# plt.legend(title=r"Average wind speed m s$^{-1}$")
# plt.tight_layout()
# plt.ylim(1e-4, 10**0)
# plt.xticks(fontsize=14, fontweight='bold')
# plt.yticks(fontsize=14, fontweight='bold')
# plt.show()

# total_legs = sum(len(group) for group in grouped_distributions.values())
# print(f"Total number of legs plotted: {total_legs}")
#%%
# # Select the windspeed bin 5-7 m/s (index 1)
# idx = 1  # Index for the 5-7 m/s bin

# # Check if the bin contains data
# if grouped_distributions[idx]:
#     avg_distribution = np.mean(grouped_distributions[idx], axis=0)
#     avg_windspeed = np.mean(mean_windspeeds[idx])
#     num_legs = len(grouped_distributions[idx])

#     # Plot only the 5-7 m/s bin
#     plt.figure(figsize=(10, 6))
#     plt.plot(common_bins, avg_distribution, color='orange', label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")

#     # Set log scales
#     plt.yscale('log')
#     plt.xscale('log')
#     plt.xticks(fontsize=16, fontweight='bold')
#     plt.yticks(fontsize=16, fontweight='bold')
#     plt.ylim(10**-4, 10**2)
#     # Labels and title
#     plt.ylabel(' concentration (/cm³)', fontweight='bold')
#     plt.xlabel('Bin diameter (µm)', fontweight='bold')
#     plt.title('Below Cloud Base CAS January-June 2022 wind speed 5-7 m/s', fontweight='bold')
#     plt.legend()

#     plt.tight_layout()
#     plt.show()
# else:
#     print("No data available for the 5-7 m/s windspeed bin.")































































































































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
#%%
#adding error bars

import random

# Define wind speed bins (same as CAS)
windspeed_bins = [(0, 3), (3.001, 6.5), (6.501, 8.5), (8.501, np.inf)]
cas_bin_counts = [78, 174, 62, 54]  # Exact CAS leg counts per bin

# Store binned distributions for CDP
grouped_distributions_CDP = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds_CDP = {i: [] for i in range(len(windspeed_bins))}

missing_windspeed_count_CDP = 0

# Define common bins (25 bin centers between 2 and 10 µm)
common_bins = np.linspace(2, 10, 25)

# Step 1: Collect CDP legs into bins
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
            break

# Step 2: Randomly select legs to match CAS counts
final_grouped_CDP = {i: [] for i in range(len(windspeed_bins))}
final_mean_windspeeds_CDP = {i: [] for i in range(len(windspeed_bins))}

for idx in range(len(windspeed_bins)):
    if len(grouped_distributions_CDP[idx]) >= cas_bin_counts[idx]:
        selected_indices = random.sample(range(len(grouped_distributions_CDP[idx])), cas_bin_counts[idx])
    else:
        selected_indices = list(range(len(grouped_distributions_CDP[idx])))  # Use all if not enough

    final_grouped_CDP[idx] = [grouped_distributions_CDP[idx][i] for i in selected_indices]
    final_mean_windspeeds_CDP[idx] = [mean_windspeeds_CDP[idx][i] for i in selected_indices]

# Step 3: Compute error bars (Standard Error) and plot with legend entries
plt.figure(figsize=(10, 8))
windspeed_colors = ['blue', 'orange', 'green', 'red']  # Order must match bins
legend_entries = []

for idx, (low, high) in enumerate(windspeed_bins):
    if final_grouped_CDP[idx]:
        avg_distribution_CDP = np.mean(final_grouped_CDP[idx], axis=0)  # Average size distribution
        std_dev_CDP = np.std(final_grouped_CDP[idx], axis=0)  # Standard deviation
        std_error_CDP = std_dev_CDP / np.sqrt(len(final_grouped_CDP[idx]))  # Standard error

        avg_windspeed_CDP = np.mean(final_mean_windspeeds_CDP[idx])
        num_legs_CDP = len(final_grouped_CDP[idx])
        avg_se = np.mean(std_error_CDP)  # Compute average SE for legend

        # Plot the average distribution
        plt.plot(common_bins, avg_distribution_CDP, 
                 label=f"{avg_windspeed_CDP:.1f} m/s, SE={avg_se:.3f}", 
                 linewidth=2.5, color=windspeed_colors[idx])

        # Add error bars
        plt.errorbar(common_bins, avg_distribution_CDP, yerr=std_error_CDP, fmt='o', color=windspeed_colors[idx],
                     capsize=3, capthick=1.5, markersize=5, label=None)

# Customize plot labels and settings
plt.yscale('log')
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=21, fontweight="bold")
plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=22, fontweight="bold")
plt.title('CDP Dry Size Distributions \nBinned by Average Wind Speed', fontweight='bold', fontsize=21)

# Update legend format
plt.legend(title=r"Wind Speed & Errors (m s$^{-1}$)", title_fontsize=19, fontsize=19, frameon=True)
plt.xticks(fontweight="bold", fontsize=21)
plt.yticks(fontweight="bold", fontsize=21)
plt.ylim(1e-4, 10**0)
plt.tight_layout()

# Show the plot
plt.show()



# %%
# #adding uncertainty

# plt.figure(figsize=(10, 8))

# # Choose whether to use Standard Error (SE) or Interquartile Range (IQR) for uncertainty
# use_standard_error = True  # Set to False to use IQR instead

# for idx, (low, high) in enumerate(windspeed_bins):
#     if final_grouped_CDP[idx]:  # Ensure the bin is not empty
#         avg_distribution_CDP = np.mean(final_grouped_CDP[idx], axis=0)  # Mean size distribution
#         num_legs_CDP = len(final_grouped_CDP[idx])
#         avg_windspeed_CDP = np.mean(final_mean_windspeeds_CDP[idx])

#         # Standard Error (SE) Option
#         if use_standard_error:
#             std_error_CDP = np.std(final_grouped_CDP[idx], axis=0, ddof=1) / np.sqrt(num_legs_CDP)
#             lower_bound_CDP = avg_distribution_CDP - std_error_CDP
#             upper_bound_CDP = avg_distribution_CDP + std_error_CDP

#         # Interquartile Range (IQR) Option
#         else:
#             lower_bound_CDP = np.percentile(final_grouped_CDP[idx], 25, axis=0)  # 25th percentile
#             upper_bound_CDP = np.percentile(final_grouped_CDP[idx], 75, axis=0)  # 75th percentile

#         # Plot the mean size distribution for CDP
#         plt.plot(common_bins, avg_distribution_CDP, label=f"{avg_windspeed_CDP:.1f} m/s, n={num_legs_CDP} legs", linewidth=2.5)

#         # Plot uncertainty as shaded region
#         plt.fill_between(common_bins, lower_bound_CDP, upper_bound_CDP, alpha=0.3)

# # Set Y-axis scaling
# plt.yscale('log')  # Change to 'linear' if needed for checking uncertainty

# # Labels and formatting
# plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=16, fontweight="bold")
# plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=16, fontweight="bold")
# plt.title('CDP Dry Size Distributions Binned by Average Wind Speed CDP', fontweight='bold', fontsize=17)
# plt.legend(title=r"Average wind speed m s$^{-1}$")
# plt.tight_layout()
# plt.ylim(1e-4, 10**0)
# plt.xticks(fontsize=14, fontweight='bold')
# plt.yticks(fontsize=14, fontweight='bold')

# # Show the plot
# plt.show()

# # Print sample sizes per bin for verification
# for idx, group in final_grouped_CDP.items():
#     print(f"Windspeed bin {idx} ({windspeed_bins[idx]} m/s): {len(group)} legs")
#%%
# # Define the exponential function
# def fit_function(x, n0, D):
#     return n0 * np.exp(-x / D)

# # Store fitted results for each windspeed bin
# fit_results_CDP = {}

# plt.figure(figsize=(10, 8))

# # Colors must match the bins
# windspeed_colors = ['blue', 'orange', 'green', 'red']

# for idx, (low, high) in enumerate(windspeed_bins):
#     if final_grouped_CDP[idx]:  # Use only the legs selected to match CAS
#         concentrations_array = np.array(final_grouped_CDP[idx])
        
#         # Compute average concentration for each bin
#         avg_concentration_CDP = np.mean(concentrations_array, axis=0)
        
#         # Avoid fitting negative or zero values
#         avg_concentration_CDP = np.where(avg_concentration_CDP <= 0, 1e-10, avg_concentration_CDP)

#         avg_windspeed_CDP = np.mean(final_mean_windspeeds_CDP[idx])
#         num_legs_CDP = len(final_grouped_CDP[idx])

#         # Perform exponential fit only after selection
#         try:
#             popt, _ = curve_fit(fit_function, common_bins, avg_concentration_CDP, p0=(1, 5), maxfev=5000)
#             n0_fit, D_fit = popt
#             fit_results_CDP[idx] = {'n0': n0_fit, 'D': D_fit, 'avg_windspeed': avg_windspeed_CDP, 'num_legs': num_legs_CDP}

#             # Generate fitted curve
#             x_fit = np.linspace(min(common_bins), max(common_bins), 10)
#             y_fit = fit_function(x_fit, *popt)

#             # Plot fitted lines
#             plt.plot(x_fit, y_fit, color=windspeed_colors[idx], linewidth=3, linestyle='-',
#                      label=f"{avg_windspeed_CDP:.1f} m/s, n={num_legs_CDP} legs")

#         except RuntimeError:
#             print(f"Exponential fit failed for windspeed bin {avg_windspeed_CDP:.1f} m/s")

# # Plot Formatting
# plt.ylabel(r'Number concentration (cm$^{-3}$ µm$^{-1}$)', fontweight='bold', fontsize=18)
# plt.xlabel('Bin diameter (µm)', fontweight='bold', fontsize=18)
# plt.yscale('log')
# # plt.ylim(10**-4, 10**0)
# # plt.title('CDP Below Cloud Base January - June 2022\nFitted Dry Size Distributions Binned by Average Windspeed', fontweight='bold', fontsize=16)
# # plt.legend(title=r"Wind speed bins (m s$^{-1}$)", title_fontsize=15, fontsize=13, frameon=True, prop={'weight': 'bold'})
# # plt.xticks(fontsize=16, fontweight='bold')
# # plt.yticks(fontsize=16, fontweight='bold')
# # plt.tight_layout()
# # plt.show()
 #%%
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit

# # Define the fit function (exponential)
# def fit_function(x, n0, D):
#     return n0 * np.exp(-x / D)

# # Create figure and primary y-axis
# fig, ax1 = plt.subplots(figsize=(10, 8))

# # Colors for wind speed bins
# windspeed_colors = ['blue', 'orange', 'green', 'red']

# # Store fit results for each windspeed bin
# fit_results_CDP = {}

# # Create secondary y-axis for residuals
# ax2 = ax1.twinx()

# residual_handles = []  # Store residual legend handles
# residual_labels = []  # Store unique residual labels

# for idx, (low, high) in enumerate(windspeed_bins):
#     if final_grouped_CDP[idx]:  # Use only the legs selected to match CAS
#         concentrations_array = np.array(final_grouped_CDP[idx])
        
#         # Compute average concentration for each bin
#         avg_concentration_CDP = np.mean(concentrations_array, axis=0)
        
#         # Avoid fitting negative or zero values
#         avg_concentration_CDP = np.where(avg_concentration_CDP <= 0, 1e-10, avg_concentration_CDP)

#         avg_windspeed_CDP = np.mean(final_mean_windspeeds_CDP[idx])
#         num_legs_CDP = len(final_grouped_CDP[idx])

#         try:
#             # Perform exponential fit
#             popt, _ = curve_fit(fit_function, common_bins, avg_concentration_CDP, p0=(1, 5), maxfev=5000)
#             n0_fit, D_fit = popt
#             fit_results_CDP[idx] = {'n0': n0_fit, 'D': D_fit, 'avg_windspeed': avg_windspeed_CDP, 'num_legs': num_legs_CDP}

#             # Generate fitted curve
#             x_fit = np.linspace(min(common_bins), max(common_bins), 10)
#             y_fit = fit_function(x_fit, *popt)

#             # Plot fitted lines
#             ax1.plot(x_fit, y_fit, color=windspeed_colors[idx], linewidth=3, linestyle='-',
#                      label=f"{avg_windspeed_CDP:.1f} m/s, n={num_legs_CDP} legs")

#             # Compute residuals
#             residuals = avg_concentration_CDP - fit_function(common_bins, *popt)

#             # Plot residuals on secondary y-axis
#             scatter = ax2.scatter(common_bins, residuals, color=windspeed_colors[idx], marker='s', edgecolor='black',
#                                   label=f"Residuals {low}-{high} m/s")

#             # Store legend labels
#             residual_handles.append(scatter)
#             residual_labels.append(f"Residuals {low}-{high} m/s")

#         except RuntimeError:
#             print(f"Exponential fit failed for windspeed bin {avg_windspeed_CDP:.1f} m/s")

# # **Set left y-axis to log scale**
# ax1.set_yscale("log")

# # Set axis labels
# ax1.set_xlabel("Bin diameter (µm)", fontsize=14, fontweight="bold")
# ax1.set_ylabel("Number concentration (cm$^{-3}$ µm$^{-1}$)", fontsize=14, fontweight="bold")
# ax2.set_ylabel("Residuals (Observed - Fitted)", fontsize=14, fontweight="bold", color="red")

# # Ensure residuals remain in linear scale
# ax2.axhline(0, color='black', linestyle='--', linewidth=1)  # Reference line for residuals

# # Update legends
# ax1.legend(loc="upper right", fontsize=12, frameon=True)
# ax2.legend(residual_handles, residual_labels, loc="lower right", fontsize=12, frameon=True)

# plt.title("CDP Fitted Dry Size Distributions with Residuals", fontsize=16, fontweight="bold")

# # Show plot
# plt.show()

# #%%
# #adding residuals to each 

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit

# # Define the fit function (exponential)
# def fit_function(x, n0, D):
#     return n0 * np.exp(-x / D)

# # Store residuals for each windspeed bin
# residuals_CDP = {}

# plt.figure(figsize=(10, 8))

# # Colors must match bins
# windspeed_colors = ['blue', 'orange', 'green', 'red']

# for idx, (low, high) in enumerate(windspeed_bins):
#     if final_grouped_CDP[idx]:  # Use only the legs selected to match CAS
#         concentrations_array = np.array(final_grouped_CDP[idx])

#         # Compute average concentration per bin
#         avg_concentration_CDP = np.mean(concentrations_array, axis=0)
        
#         # Avoid negative or zero values for fitting
#         avg_concentration_CDP = np.where(avg_concentration_CDP <= 0, 1e-10, avg_concentration_CDP)

#         avg_windspeed_CDP = np.mean(final_mean_windspeeds_CDP[idx])
#         num_legs_CDP = len(final_grouped_CDP[idx])

#         try:
#             # Fit exponential model
#             popt, _ = curve_fit(fit_function, common_bins, avg_concentration_CDP, p0=(1, 5), maxfev=5000)
#             n0_fit, D_fit = popt

#             # Compute fitted values
#             fitted_values = fit_function(common_bins, *popt)

#             # Compute residuals (Observed - Fitted)
#             residual_values = concentrations_array - fitted_values  # Compute residuals per bin

#             # Compute mean residuals per bin
#             mean_residuals = np.mean(residual_values, axis=0)

#             # Compute standard deviation of residuals per bin
#             std_residuals = np.std(residual_values, axis=0)

#             # Store residuals
#             residuals_CDP[idx] = {'mean_residuals': mean_residuals, 'std_residuals': std_residuals}

#             # Plot residuals
#             plt.errorbar(common_bins, mean_residuals, yerr=std_residuals, fmt='o', color=windspeed_colors[idx],
#                          capsize=3, capthick=1.5, markersize=5, label=f"{avg_windspeed_CDP:.1f} m/s, n={num_legs_CDP}")

#         except RuntimeError:
#             print(f"Exponential fit failed for windspeed bin {avg_windspeed_CDP:.1f} m/s")

# # Plot Formatting
# plt.axhline(y=0, color='black', linestyle='--', linewidth=2)  # Add zero residual reference line
# plt.ylabel("Residuals (Observed - Fitted)", fontweight='bold', fontsize=16)
# plt.xlabel("Bin Diameter (µm)", fontweight='bold', fontsize=16)
# plt.title("Residuals of Fitted Dry Size Distributions\nBinned by Average Wind Speed (CDP)", fontweight='bold', fontsize=16)
# plt.legend(title="Windspeed Bins (m/s)", title_fontsize=14, fontsize=13, frameon=True)
# plt.xticks(fontsize=14, fontweight='bold')
# plt.yticks(fontsize=14, fontweight='bold')
# plt.tight_layout()

# # Show the plot
# plt.show()

# # Print Residual Summary
# for idx, res in residuals_CDP.items():
#     print(f"Windspeed Bin {idx}: Mean Residuals = {np.mean(res['mean_residuals']):.3f}, Std Dev = {np.mean(res['std_residuals']):.3f}")
# #%%




#%%
# #adding error bars 

# # Define the fit function (exponential)
# def fit_function(x, n0, D):
#     return n0 * np.exp(-x / D)

# # Store fitted results for each windspeed bin
# fit_results_CDP = {}

# plt.figure(figsize=(10, 8))

# # Colors must match the bins
# windspeed_colors = ['blue', 'orange', 'green', 'red']
# errorbar_handles = []

# for idx, (low, high) in enumerate(windspeed_bins):
#     if final_grouped_CDP[idx]:  # Use only the legs selected to match CAS
#         concentrations_array = np.array(final_grouped_CDP[idx])
        
#         # Compute average concentration for each bin
#         avg_concentration_CDP = np.mean(concentrations_array, axis=0)
#         std_dev_CDP = np.std(concentrations_array, axis=0)  # Standard deviation
#         std_error_CDP = std_dev_CDP / np.sqrt(len(final_grouped_CDP[idx]))  # Standard error

#         # Avoid fitting negative or zero values
#         avg_concentration_CDP = np.where(avg_concentration_CDP <= 0, 1e-10, avg_concentration_CDP)

#         avg_windspeed_CDP = np.mean(final_mean_windspeeds_CDP[idx])
#         num_legs_CDP = len(final_grouped_CDP[idx])
#         avg_se = np.mean(std_error_CDP)  # Compute average SE for legend

#         # Perform exponential fit only after selection
#         try:
#             popt, _ = curve_fit(fit_function, common_bins, avg_concentration_CDP, p0=(1, 5), maxfev=5000)
#             n0_fit, D_fit = popt
#             fit_results_CDP[idx] = {'n0': n0_fit, 'D': D_fit, 'avg_windspeed': avg_windspeed_CDP, 'num_legs': num_legs_CDP}

#             # Generate fitted curve
#             x_fit = np.linspace(min(common_bins), max(common_bins), 10)
#             y_fit = fit_function(x_fit, *popt)

#             # Plot fitted lines
#             plt.plot(x_fit, y_fit, color=windspeed_colors[idx], linewidth=3, linestyle='-',
#                      label=f"{avg_windspeed_CDP:.1f} m/s, SE={avg_se:.3f}, n={num_legs_CDP} legs")

#             # Add error bars to the fitted data
#             errbar = plt.errorbar(common_bins, avg_concentration_CDP, yerr=std_error_CDP, fmt='o', color=windspeed_colors[idx],
#                                   capsize=3, capthick=1.5, markersize=5, label=None)
#             errorbar_handles.append(errbar)

#         except RuntimeError:
#             print(f"Exponential fit failed for windspeed bin {avg_windspeed_CDP:.1f} m/s")

# # Plot Formatting
# plt.ylabel(r'Number concentration (cm$^{-3}$ µm$^{-1}$)', fontweight='bold', fontsize=18)
# plt.xlabel('Bin diameter (µm)', fontweight='bold', fontsize=18)
# plt.yscale('log')
# plt.ylim(10**-4, 10**0)
# plt.title('Below Cloud Base January - June 2022\nFitted Dry Size Distributions Binned by Average Windspeed (CDP)', 
#           fontweight='bold', fontsize=16)

# # Update legend format
# plt.legend(title=r"Wind Speed & Errors (m s$^{-1}$)", title_fontsize=15, fontsize=13, frameon=True)
# plt.xticks(fontsize=16, fontweight='bold')
# plt.yticks(fontsize=16, fontweight='bold')
# plt.tight_layout()

# # Show the plot
# plt.show()


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
                color="green", s=100, edgecolor='black', zorder=3)

# Fit line
x_fit_CDP = np.linspace(min(windspeed_values_CDP), max(windspeed_values_CDP), 100)
y_fit_CDP = linear_model(x_fit_CDP, *popt_CDP)
plt.plot(x_fit_CDP, y_fit_CDP, 'r-', label=f'Fit: y = {m_fit_CDP:.3f}x + {b_fit_CDP:.3f}, R² = {r_squared_CDP:.2f}')

# Labels and formatting
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=16, fontweight='bold')
plt.ylabel("Total Wind Speed Bin Concentration (cm$^{-3}$)", fontsize=16, fontweight='bold')
plt.title("CDP Wind Speed and Total Concentration Correlation", fontsize=16, fontweight='bold')

# Legend
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

# Print regression results
print(f"Slope (m): {m_fit_CDP:.3f}")
print(f"Intercept (b): {b_fit_CDP:.3f}")
print(f"R² value: {r_squared_CDP:.2f}")
#%%
#addingR value

# ✅ Define linear regression function
def linear_model(x, m, b):
    return m * x + b

# ✅ Extract average wind speeds and total concentrations from binned distributions
avg_windspeeds_CDP = []
total_concentrations_CDP = []

# ✅ Ensure bin edges match the common_bins used for integration
bin_edges_CDP = np.linspace(2, 10, len(common_bins) + 1)  # +1 ensures correct edges
bin_widths_CDP = np.diff(bin_edges_CDP)  # Compute widths between bin edges

# ✅ Convert size distributions from cm⁻³ μm⁻¹ to total concentration (cm⁻³)
for idx, (low, high) in enumerate(windspeed_bins):
    if final_grouped_CDP[idx]:  # Use only selected CDP legs
        avg_windspeed_CDP = np.mean(final_mean_windspeeds_CDP[idx])  # Average windspeed for this bin

        # Convert using correct bin widths (ensuring dimensions match)
        avg_concentration_per_leg_CDP = [np.sum(dist[:len(bin_widths_CDP)] * bin_widths_CDP) for dist in final_grouped_CDP[idx]]
        avg_concentration_CDP = np.mean(avg_concentration_per_leg_CDP)  # Average over all legs in this bin

        avg_windspeeds_CDP.append(avg_windspeed_CDP)
        total_concentrations_CDP.append(avg_concentration_CDP)

# ✅ Convert to numpy arrays for fitting
windspeed_values_CDP = np.array(avg_windspeeds_CDP)
total_concentrations_CDP = np.array(total_concentrations_CDP)

# ✅ Perform linear regression
popt_CDP, _ = curve_fit(linear_model, windspeed_values_CDP, total_concentrations_CDP)
m_fit_CDP, b_fit_CDP = popt_CDP

# ✅ Compute R² value
residuals_CDP = total_concentrations_CDP - linear_model(windspeed_values_CDP, *popt_CDP)
ss_res_CDP = np.sum(residuals_CDP**2)
ss_tot_CDP = np.sum((total_concentrations_CDP - np.mean(total_concentrations_CDP))**2)
r_squared_CDP = 1 - (ss_res_CDP / ss_tot_CDP)

# ✅ Compute Pearson correlation coefficient (R)
pearson_corr_CDP, _ = pearsonr(windspeed_values_CDP, total_concentrations_CDP)

# ✅ Plot Wind Speed vs. Total Droplet Concentration for CDP
plt.figure(figsize=(8, 6))

# ✅ Plot data points (Green for CDP)
plt.scatter(windspeed_values_CDP, total_concentrations_CDP, 
            color="green", s=100, edgecolor='black', zorder=3, label="CDP")

# ✅ Fit line
x_fit_CDP = np.linspace(min(windspeed_values_CDP), max(windspeed_values_CDP), 100)
y_fit_CDP = linear_model(x_fit_CDP, *popt_CDP)
plt.plot(x_fit_CDP, y_fit_CDP, 'black', linewidth=2, 
         label=f'Fit: y = {m_fit_CDP:.3f}x + {b_fit_CDP:.3f}\nR² = {r_squared_CDP:.2f}, R = {pearson_corr_CDP:.2f}')

# ✅ Labels and formatting
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=16, fontweight='bold')
plt.ylabel("Total Wind Speed Bin Concentration (cm$^{-3}$)", fontsize=16, fontweight='bold')
plt.title("CDP Wind Speed and Total Concentration Correlation", fontsize=16, fontweight='bold')

# ✅ Legend: Only show fit equation, R², and R
plt.legend(fontsize=14, title_fontsize=14, loc='best', frameon=True)

# ✅ Final Formatting
plt.tight_layout()
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.ylim(0.1, 0.7)
plt.xlim(0, 10)
plt.show()

# ✅ Print regression results
print(f"Slope (m): {m_fit_CDP:.3f}")
print(f"Intercept (b): {b_fit_CDP:.3f}")
print(f"R² value: {r_squared_CDP:.2f}")
print(f"R value (Pearson correlation): {pearson_corr_CDP:.3f}")

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.optimize import curve_fit

# ✅ Define linear regression function
def linear_model(x, m, b):
    return m * x + b

# ✅ Extract average wind speeds and total concentrations from binned distributions
avg_windspeeds_CDP = []
total_concentrations_CDP = []
standard_errors_CDP = []

# ✅ Ensure bin edges match the common_bins used for integration
bin_edges_CDP = np.linspace(2, 10, len(common_bins) + 1)  # +1 ensures correct edges
bin_widths_CDP = np.diff(bin_edges_CDP)  # Compute widths between bin edges

# ✅ Convert size distributions from cm⁻³ μm⁻¹ to total concentration (cm⁻³)
for idx, (low, high) in enumerate(windspeed_bins):
    if final_grouped_CDP[idx]:  # Use only selected CDP legs
        avg_windspeed_CDP = np.mean(final_mean_windspeeds_CDP[idx])  # Average windspeed for this bin

        # Convert using correct bin widths (ensuring dimensions match)
        avg_concentration_per_leg_CDP = [np.sum(dist[:len(bin_widths_CDP)] * bin_widths_CDP) for dist in final_grouped_CDP[idx]]
        avg_concentration_CDP = np.mean(avg_concentration_per_leg_CDP)  # Average over all legs in this bin
        std_concentration_CDP = np.std(avg_concentration_per_leg_CDP, ddof=1)  # Standard deviation
        N_CDP = len(avg_concentration_per_leg_CDP)  # Number of legs in this bin
        SE_concentration_CDP = std_concentration_CDP / np.sqrt(N_CDP)  # Standard Error

        avg_windspeeds_CDP.append(avg_windspeed_CDP)
        total_concentrations_CDP.append(avg_concentration_CDP)
        standard_errors_CDP.append(SE_concentration_CDP)

# ✅ Convert to numpy arrays for fitting
windspeed_values_CDP = np.array(avg_windspeeds_CDP)
total_concentrations_CDP = np.array(total_concentrations_CDP)
standard_errors_CDP = np.array(standard_errors_CDP)

# ✅ Perform linear regression
popt_CDP, _ = curve_fit(linear_model, windspeed_values_CDP, total_concentrations_CDP)
m_fit_CDP, b_fit_CDP = popt_CDP

# ✅ Compute R² value
residuals_CDP = total_concentrations_CDP - linear_model(windspeed_values_CDP, *popt_CDP)
ss_res_CDP = np.sum(residuals_CDP**2)
ss_tot_CDP = np.sum((total_concentrations_CDP - np.mean(total_concentrations_CDP))**2)
r_squared_CDP = 1 - (ss_res_CDP / ss_tot_CDP)

# ✅ Compute Pearson correlation coefficient (R)
pearson_corr_CDP, _ = pearsonr(windspeed_values_CDP, total_concentrations_CDP)

# ✅ Plot Wind Speed vs. Total Droplet Concentration for CDP
plt.figure(figsize=(8, 6))

# ✅ Plot data points with error bars (Green for CDP)
plt.errorbar(windspeed_values_CDP, total_concentrations_CDP, 
             yerr=standard_errors_CDP, fmt='o', color='green', 
             markersize=10, capsize=5, capthick=2, label="CDP", 
             ecolor='black', elinewidth=1.5, zorder=3)

# ✅ Fit line
x_fit_CDP = np.linspace(min(windspeed_values_CDP), max(windspeed_values_CDP), 100)
y_fit_CDP = linear_model(x_fit_CDP, *popt_CDP)
plt.plot(x_fit_CDP, y_fit_CDP, 'black', linewidth=2, 
         label=f'Fit: y = {m_fit_CDP:.3f}x + {b_fit_CDP:.3f}\nR² = {r_squared_CDP:.2f}, R = {pearson_corr_CDP:.2f}')

# ✅ Labels and formatting
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=16, fontweight='bold')
plt.ylabel("Total Wind Speed Bin Concentration (cm$^{-3}$)", fontsize=16, fontweight='bold')
plt.title("CDP Wind Speed and Total Concentration Correlation", fontsize=16, fontweight='bold')

# ✅ Legend: Only show fit equation, R², and R
plt.legend(fontsize=14, title_fontsize=14, loc='best', frameon=True)

# ✅ Final Formatting
plt.tight_layout()
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.ylim(0.1, 0.7)
plt.xlim(0, 10)
plt.show()

# ✅ Print regression results
print(f"Slope (m): {m_fit_CDP:.3f}")
print(f"Intercept (b): {b_fit_CDP:.3f}")
print(f"R² value: {r_squared_CDP:.2f}")
print(f"R value (Pearson correlation): {pearson_corr_CDP:.3f}")


# %%
#overlaying CDP and CAS concentration correlation


# ✅ Define linear regression function
def linear_model(x, m, b):
    return m * x + b

# ====== CAS Data Processing ======
avg_windspeeds_CAS = []
total_concentrations_CAS = []
standard_errors_CAS = []

# Convert size distributions from cm⁻³ μm⁻¹ to total concentration (cm⁻³)
for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:  # Ensure bin has data
        avg_windspeed_CAS = np.mean(mean_windspeeds[idx])  # Average windspeed for this bin

        # Convert using correct bin widths (integrate dN/dD)
        avg_concentration_per_leg_CAS = [np.sum(dist * bin_widths) for dist in grouped_distributions[idx]]
        avg_concentration_CAS = np.mean(avg_concentration_per_leg_CAS)  # Mean concentration
        std_concentration_CAS = np.std(avg_concentration_per_leg_CAS, ddof=1)  # Standard deviation
        N_CAS = len(avg_concentration_per_leg_CAS)  # Number of legs in this bin
        SE_concentration_CAS = std_concentration_CAS / np.sqrt(N_CAS)  # Standard Error

        avg_windspeeds_CAS.append(avg_windspeed_CAS)
        total_concentrations_CAS.append(avg_concentration_CAS)
        standard_errors_CAS.append(SE_concentration_CAS)

# Convert to numpy arrays for CAS fitting
windspeed_values_CAS = np.array(avg_windspeeds_CAS)
total_concentrations_CAS = np.array(total_concentrations_CAS)
standard_errors_CAS = np.array(standard_errors_CAS)

# Perform linear regression for CAS
popt_CAS, _ = curve_fit(linear_model, windspeed_values_CAS, total_concentrations_CAS)
m_fit_CAS, b_fit_CAS = popt_CAS

# Compute R² and R for CAS
residuals_CAS = total_concentrations_CAS - linear_model(windspeed_values_CAS, *popt_CAS)
ss_res_CAS = np.sum(residuals_CAS**2)
ss_tot_CAS = np.sum((total_concentrations_CAS - np.mean(total_concentrations_CAS))**2)
r_squared_CAS = 1 - (ss_res_CAS / ss_tot_CAS)
pearson_corr_CAS, _ = pearsonr(windspeed_values_CAS, total_concentrations_CAS)

# ====== CDP Data Processing ======
avg_windspeeds_CDP = []
total_concentrations_CDP = []
standard_errors_CDP = []

# Convert size distributions from cm⁻³ μm⁻¹ to total concentration (cm⁻³)
for idx, (low, high) in enumerate(windspeed_bins):
    if final_grouped_CDP[idx]:  # Use only selected CDP legs
        avg_windspeed_CDP = np.mean(final_mean_windspeeds_CDP[idx])  # Average windspeed for this bin

        # Convert using correct bin widths (ensuring dimensions match)
        avg_concentration_per_leg_CDP = [np.sum(dist[:len(bin_widths_CDP)] * bin_widths_CDP) for dist in final_grouped_CDP[idx]]
        avg_concentration_CDP = np.mean(avg_concentration_per_leg_CDP)  # Mean concentration
        std_concentration_CDP = np.std(avg_concentration_per_leg_CDP, ddof=1)  # Standard deviation
        N_CDP = len(avg_concentration_per_leg_CDP)  # Number of legs in this bin
        SE_concentration_CDP = std_concentration_CDP / np.sqrt(N_CDP)  # Standard Error

        avg_windspeeds_CDP.append(avg_windspeed_CDP)
        total_concentrations_CDP.append(avg_concentration_CDP)
        standard_errors_CDP.append(SE_concentration_CDP)

# Convert to numpy arrays for CDP fitting
windspeed_values_CDP = np.array(avg_windspeeds_CDP)
total_concentrations_CDP = np.array(total_concentrations_CDP)
standard_errors_CDP = np.array(standard_errors_CDP)

# Perform linear regression for CDP
popt_CDP, _ = curve_fit(linear_model, windspeed_values_CDP, total_concentrations_CDP)
m_fit_CDP, b_fit_CDP = popt_CDP

# Compute R² and R for CDP
residuals_CDP = total_concentrations_CDP - linear_model(windspeed_values_CDP, *popt_CDP)
ss_res_CDP = np.sum(residuals_CDP**2)
ss_tot_CDP = np.sum((total_concentrations_CDP - np.mean(total_concentrations_CDP))**2)
r_squared_CDP = 1 - (ss_res_CDP / ss_tot_CDP)
pearson_corr_CDP, _ = pearsonr(windspeed_values_CDP, total_concentrations_CDP)

# ====== PLOTTING CAS & CDP TOGETHER ======
plt.figure(figsize=(8, 6))

# ✅ Plot CAS data (Blue) with error bars
plt.errorbar(windspeed_values_CAS, total_concentrations_CAS, 
             yerr=standard_errors_CAS, fmt='o', color='blue', 
             markersize=10, capsize=5, capthick=2, label="CAS", 
             ecolor='black', elinewidth=1.5, zorder=3)

# ✅ Fit line for CAS
x_fit_CAS = np.linspace(min(windspeed_values_CAS), max(windspeed_values_CAS), 100)
y_fit_CAS = linear_model(x_fit_CAS, *popt_CAS)
plt.plot(x_fit_CAS, y_fit_CAS, 'blue', linewidth=2, 
         label=f'CAS Fit: y = {m_fit_CAS:.3f}x + {b_fit_CAS:.3f}\nR² = {r_squared_CAS:.2f}, R = {pearson_corr_CAS:.2f}')

# ✅ Plot CDP data (Green) with error bars
plt.errorbar(windspeed_values_CDP, total_concentrations_CDP, 
             yerr=standard_errors_CDP, fmt='o', color='green', 
             markersize=10, capsize=5, capthick=2, label="CDP", 
             ecolor='black', elinewidth=1.5, zorder=3)

# ✅ Fit line for CDP
x_fit_CDP = np.linspace(min(windspeed_values_CDP), max(windspeed_values_CDP), 100)
y_fit_CDP = linear_model(x_fit_CDP, *popt_CDP)
plt.plot(x_fit_CDP, y_fit_CDP, 'green', linewidth=2, 
         label=f'CDP Fit: y = {m_fit_CDP:.3f}x + {b_fit_CDP:.3f}\nR² = {r_squared_CDP:.2f}, R = {pearson_corr_CDP:.2f}')

# ✅ Labels & Titles
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=19, fontweight='bold')
plt.ylabel("Total Wind Speed Bin Concentration\n (cm$^{-3}$)", fontsize=19, fontweight='bold')
plt.title("Wind Speed and Total Concentration Correlation", fontsize=18, fontweight='bold')

# ✅ Legend: Fit equations only
plt.legend(fontsize=12, title_fontsize=14, loc='best', frameon=True)

# ✅ Final Plot Settings
plt.tight_layout()
plt.xlim(0, 10)
plt.ylim(0.1, 0.7)
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.show()

# ✅ Print Results
print(f"CAS Slope: {m_fit_CAS:.3f}, Intercept: {b_fit_CAS:.3f}, R² = {r_squared_CAS:.2f}, R = {pearson_corr_CAS:.2f}")
print(f"CDP Slope: {m_fit_CDP:.3f}, Intercept: {b_fit_CDP:.3f}, R² = {r_squared_CDP:.2f}, R = {pearson_corr_CDP:.2f}")






















#Lewis and Schwartz
#%%
# # ✅ Wind speed bins (to match your new binning choice)
# windspeed_bins = [(0, 5), (5.001, 7), (7.001, 9), (9.001, np.inf)]

# # ✅ Store binned distributions
# grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
# mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# missing_windspeed_count = 0

# # ✅ Convert bin centers from diameter to radius
# bin_radius = np.array(bin_center_CDP, dtype=float) / 2  # Ensure array and convert to radius

# # ✅ Use already fitted exponential size distributions for ambient
# for entry in CDP_fits_10:
#     date = entry['Date']
#     BCB_start = entry['BCB_start']
#     BCB_stop = entry['BCB_stop']
    
#     n0 = entry['Intercept_n0']
#     D = entry['E_folding_D']

#     # Match windspeed from df_combined
#     windspeed_entry = df_combined[
#         (df_combined['Date'] == date) & 
#         (df_combined['BCB_start'] == BCB_start) & 
#         (df_combined['BCB_stop'] == BCB_stop)
#     ]

#     if windspeed_entry.empty or np.isnan(n0) or np.isnan(D):
#         missing_windspeed_count += 1
#         continue

#     windspeed = windspeed_entry['Windspeed'].values[0]

#     # ✅ Use the already fitted exponential size distribution
#     size_dist = n0 * np.exp(-bin_radius / D)  # Use radius instead of diameter

#     # ✅ Bin the distribution by wind speed
#     for idx, (low, high) in enumerate(windspeed_bins):
#         if low <= windspeed < high:
#             grouped_distributions[idx].append(size_dist)
#             mean_windspeeds[idx].append(windspeed)
#             break

# # ✅ Print summary
# for idx, group in grouped_distributions.items():
#     print(f"Windspeed bin {idx} ({windspeed_bins[idx]} m/s): {len(group)} legs")

# print(f"Total legs with missing windspeed data: {missing_windspeed_count}")

# # ✅ Step 3: Plot binned size distributions
# plt.figure(figsize=(10, 8))

# for idx, (low, high) in enumerate(windspeed_bins):
#     if grouped_distributions[idx]:
#         avg_distribution = np.nanmean(grouped_distributions[idx], axis=0)  # ✅ Average size distribution
#         avg_windspeed = np.nanmean(mean_windspeeds[idx])
#         num_legs = len(grouped_distributions[idx])

#         plt.plot(bin_radius, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linewidth=2.5)

# plt.xscale('log')  # ✅ Match Lewis & Schwartz log scale
# plt.yscale('log')
# plt.ylabel(r"Number Concentration (cm$^{-3}$)", fontsize=16, fontweight="bold")  
# plt.xlabel("Bin Center Radius (µm)", fontsize=16, fontweight="bold")
# plt.title('Ambient Size Distributions Binned by Average Wind Speed', fontweight='bold', fontsize=17)
# plt.legend(title=r"Average wind speed (m s$^{-1}$)")
# plt.tight_layout()
# plt.ylim(1e-6, 10**2)  # ✅ Match y-axis with Lewis & Schwartz

# plt.xticks(fontsize=14, fontweight='bold')
# plt.yticks(fontsize=14, fontweight='bold')
# plt.show()

# total_legs = sum(len(group) for group in grouped_distributions.values())
# print(f"Total number of legs plotted: {total_legs}")

# #%%
# # ✅ Define Wind Speed Bins (matching your chosen bins)
# windspeed_bins = [(0, 5), (5.001, 7), (7.001, 9), (9.001, np.inf)]

# # ✅ Storage for Binned Distributions
# grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
# mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# missing_windspeed_count = 0

# # ✅ Convert bin centers from diameter to radius
# bin_radius = np.array(bin_center_CDP, dtype=float) / 2  # Convert to radius

# # ✅ Loop through each leg in Y_BCB_calc
# for entry in Y_CDP_calc:
#     date = entry['Date']
#     BCB_start = entry['BCB_start']
#     BCB_stop = entry['BCB_stop']

#     # ✅ Get the observed size distribution for this leg
#     size_dist = np.array([entry.get(f'Bin{i:02d}_Y_mean', np.nan) for i in range(30)], dtype=float)

#     # ✅ Skip if all values are NaN
#     if np.isnan(size_dist).all():
#         continue

#     # ✅ Find Matching Wind Speed Data
#     windspeed_entry = df_combined[
#         (df_combined['Date'] == date) & 
#         (df_combined['BCB_start'] == BCB_start) & 
#         (df_combined['BCB_stop'] == BCB_stop)
#     ]

#     if windspeed_entry.empty:
#         missing_windspeed_count += 1
#         continue

#     windspeed = windspeed_entry['Windspeed'].values[0]

#     # ✅ Bin the distribution by wind speed
#     for idx, (low, high) in enumerate(windspeed_bins):
#         if low <= windspeed < high:
#             grouped_distributions[idx].append(size_dist)
#             mean_windspeeds[idx].append(windspeed)
#             break

# # ✅ Print Summary
# for idx, group in grouped_distributions.items():
#     print(f"Windspeed bin {idx} ({windspeed_bins[idx]} m/s): {len(group)} legs")

# print(f"Total legs with missing windspeed data: {missing_windspeed_count}")

# # ✅ Step 3: Plot Binned Size Distributions
# plt.figure(figsize=(10, 8))

# for idx, (low, high) in enumerate(windspeed_bins):
#     if grouped_distributions[idx]:
#         avg_distribution = np.nanmean(grouped_distributions[idx], axis=0)  # ✅ Average over legs
#         avg_windspeed = np.nanmean(mean_windspeeds[idx])
#         num_legs = len(grouped_distributions[idx])

#         plt.plot(bin_radius, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linewidth=2.5)

# plt.xscale('log')  # ✅ Match Lewis & Schwartz log scale
# plt.yscale('log')
# plt.ylabel(r"Number Concentration (cm$^{-3}$)", fontsize=16, fontweight="bold")  
# plt.xlabel("Bin Center Radius (µm)", fontsize=16, fontweight="bold")
# plt.title('Ambient Size Distributions Binned by Average Wind Speed', fontweight='bold', fontsize=17)
# plt.legend(title=r"Average wind speed (m s$^{-1}$)")
# plt.tight_layout()
# plt.ylim(1e-6, 10**2)  # ✅ Match y-axis with Lewis & Schwartz

# plt.xticks(fontsize=14, fontweight='bold')
# plt.yticks(fontsize=14, fontweight='bold')
# plt.show()

# total_legs = sum(len(group) for group in grouped_distributions.values())
# print(f"Total number of legs plotted: {total_legs}")

# # %%
# # ✅ Select the windspeed bin 5-7 m/s (index 1)
# idx = 1  # Index for the 5-7 m/s bin

# # ✅ Check if the bin contains data
# if grouped_distributions[idx]:
#     avg_distribution = np.nanmean(grouped_distributions[idx], axis=0)  # Average over all legs in the bin
#     avg_windspeed = np.nanmean(mean_windspeeds[idx])  # Compute average windspeed
#     num_legs = len(grouped_distributions[idx])  # Count number of legs in this bin

#     # ✅ Convert bin centers from diameter to radius
#     bin_radius = np.array(bin_center_CDP, dtype=float) / 2  # Ensure array and convert to radius


#     # ✅ Plot only the 5-7 m/s bin
#     plt.figure(figsize=(10, 6))
#     plt.plot(bin_radius, avg_distribution, color='red', label=f"5-7 m/s", linewidth=3)

#     # ✅ Set log-log scale (matching Lewis & Schwartz)
#     plt.yscale('log')
#     plt.xscale('log')  # ✅ Set x-axis to log scale
#     plt.xlim(0.1, 100)  # ✅ Match Lewis & Schwartz x-axis range

#     # plt.xscale('log')
#     plt.xticks(fontsize=16, fontweight='bold')
#     plt.yticks(fontsize=16, fontweight='bold')
#     plt.ylim(10**-6, 10**2)

#     # ✅ Labels and title
#     plt.ylabel('Concentration (/cm³)', fontweight='bold', fontsize=16)
#     plt.xlabel('Bin Center Radius (µm)', fontweight='bold', fontsize=16)
#     plt.title('Below Cloud Base CAS January-June 2022\n Wind Speed 5-7 m/s', fontweight='bold', fontsize=18)
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

# else:
#     print("No data available for the 5-7 m/s windspeed bin.")
# #%%
# # ✅ Select the windspeed bin 5-7 m/s (index 1)
# idx = 1  # Index for the 5-7 m/s bin

# # ✅ Check if the bin contains data
# if grouped_distributions[idx]:
#     # avg_distribution = np.nanmean(grouped_distributions[idx], axis=0)
#     avg_distribution = np.nanmean(grouped_distributions[idx], axis=0) / 2  # Adjust scaling factor if needed
#   # Average over all legs in the bin
#     avg_windspeed = np.nanmean(mean_windspeeds[idx])  # Compute average windspeed
#     num_legs = len(grouped_distributions[idx])  # Count number of legs in this bin

#     # ✅ Convert bin centers from diameter to radius & apply a slight shift if needed
#     bin_radius = np.array(bin_center_CDP, dtype=float) / 2 * 1.05  # Ensure proper alignment

#     # ✅ Plot only the 5-7 m/s bin
#     plt.figure(figsize=(10, 6))
#     plt.plot(bin_radius, avg_distribution, color='blue', label=f"5-7 m/s", linewidth=3)

#     # ✅ Set log-log scale (matching Lewis & Schwartz)
#     plt.yscale('log')
#     plt.xscale('log')  # ✅ Set x-axis to log scale
#     plt.xlim(0.1, 100)  # ✅ Match Lewis & Schwartz x-axis range

#     # ✅ Adjust x-axis ticks to match Lewis & Schwartz
#     import matplotlib.ticker as ticker
#     plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
#     plt.xticks([0.1, 1, 10, 100], labels=["0.1", "1", "10", "100"], fontsize=16, fontweight="bold")

#     plt.yticks(fontsize=16, fontweight='bold')
#     plt.ylim(10**-6, 10**2)

#     # ✅ Labels and title
#     plt.ylabel('Concentration (/cm³)', fontweight='bold', fontsize=16)
#     plt.xlabel('Bin Center Radius (µm)', fontweight='bold', fontsize=16)
#     plt.title('CDP Below Cloud Base January-June 2022\n Wind Speed 5-7 m/s', fontweight='bold', fontsize=18)
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

# else:
#     print("No data available for the 5-7 m/s windspeed bin.")

# %%
# # ✅ Convert bin centers from diameter to radius
# bin_radius = bin_center_CDP / 2  # Convert diameter to radius

# # ✅ Define log-spaced bin centers (matching Lewis & Schwartz)
# bin_radius_log = np.logspace(np.log10(0.1), np.log10(100), len(bin_radius))

# # ✅ Select the windspeed bin 5-7 m/s (index 1)
# idx = 1  # Index for the 5-7 m/s bin

# # ✅ Check if the bin contains data
# if grouped_distributions[idx]:
#     avg_distribution = np.nanmean(grouped_distributions[idx], axis=0)
#     avg_windspeed = np.nanmean(mean_windspeeds[idx])
#     num_legs = len(grouped_distributions[idx])

#     # ✅ Plot only the 5-7 m/s bin
#     plt.figure(figsize=(10, 6))
#     plt.plot(bin_radius_log, avg_distribution, color='blue', label=f"{avg_windspeed:.1f} m/s")

#     # ✅ Set log-log scale (matching Lewis & Schwartz)
#     plt.yscale('log')
#     plt.xscale('log')
#     plt.xticks(fontsize=16, fontweight='bold')
#     plt.yticks(fontsize=16, fontweight='bold')
#     plt.xlim(0.1, 100)  # Match Lewis & Schwartz range
#     plt.ylim(10**-6, 10**2)

#     # ✅ Labels and title
#     plt.ylabel('Concentration (/cm³)', fontweight='bold', fontsize=16)
#     plt.xlabel('Bin Center Radius (µm)', fontweight='bold', fontsize=16)
#     plt.title('CDP Below Cloud Base January-June 2022\n Wind Speed 5-7 m/s', fontweight='bold', fontsize=18)
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

# else:
#     print("No data available for the 5-7 m/s windspeed bin.")


# %%
#computing mass against wind speed regression
grouped_mass_values = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds_mass = {i: [] for i in range(len(windspeed_bins))}

# Loop through filtered mass data and match it to wind speed bins
for mass_entry in filtered_dry_mass_inf_CDP:
    date = mass_entry['Date']
    BCB_start = mass_entry['BCB_start']
    BCB_stop = mass_entry['BCB_stop']
    
    # Find the corresponding windspeed entry
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) & 
        (df_combined['BCB_start'] == BCB_start) & 
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    if windspeed_entry.empty:
        continue  # Skip if no matching windspeed data

    windspeed = windspeed_entry['Windspeed'].values[0]
    mass_value = mass_entry['Dry Mass (µg/m³)']

    # Bin the mass value based on windspeed
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
        avg_windspeed = np.mean(mean_windspeeds_mass[idx])  # Avg windspeed
        avg_mass = np.mean(mass_list)  # Avg dry mass

        avg_windspeeds_mass.append(avg_windspeed)
        total_mass_values.append(avg_mass)

# Convert to numpy arrays
windspeed_values_mass = np.array(avg_windspeeds_mass)
total_mass_values = np.array(total_mass_values)

# Perform linear regression
popt_mass, _ = curve_fit(linear_model, windspeed_values_mass, total_mass_values)
m_fit_mass, b_fit_mass = popt_mass

# Compute R² value
residuals_mass = total_mass_values - linear_model(windspeed_values_mass, *popt_mass)
ss_res_mass = np.sum(residuals_mass**2)
ss_tot_mass = np.sum((total_mass_values - np.mean(total_mass_values))**2)
r_squared_mass = 1 - (ss_res_mass / ss_tot_mass)

# %%
plt.figure(figsize=(8, 6))
for idx in range(len(windspeed_bins)):
    plt.scatter(windspeed_values_mass[idx], total_mass_values[idx], 
                color=colors[idx], s=100, edgecolor='black', zorder=3)

# Fit line
x_fit_mass = np.linspace(min(windspeed_values_mass), max(windspeed_values_mass), 100)
y_fit_mass = linear_model(x_fit_mass, *popt_mass)
plt.plot(x_fit_mass, y_fit_mass, 'r-', label=f'Fit: y = {m_fit_mass:.3f}x + {b_fit_mass:.3f}, R² = {r_squared_mass:.2f}')

# Labels & Titles
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=16, fontweight='bold')
plt.ylabel("Total Dry Mass (µg/m³)", fontsize=16, fontweight='bold')
plt.title("CDP Below Cloud Base January - June 2022", fontsize=16, fontweight='bold')

# Legend
legend_labels_mass = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[idx], markersize=10, 
               label=f"{windspeed_values_mass[idx]:.1f} m/s") for idx in range(len(windspeed_bins))
]
plt.legend(handles=legend_labels_mass + [plt.Line2D([0], [0], color='red', 
               label=f'Fit: y = {m_fit_mass:.3f}x + {b_fit_mass:.3f}, R² = {r_squared_mass:.2f}')], 
           title="Wind Speed Bins", title_fontsize=14, fontsize=13)

# Final Plot Settings
plt.tight_layout()
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.show()

# Print Results
print(f"Slope (m): {m_fit_mass:.3f}")
print(f"Intercept (b): {b_fit_mass:.3f}")
print(f"R² value: {r_squared_mass:.2f}")
#%%
#mass against wind speed 


# Define linear regression function
def linear_model(x, m, b):
    return m * x + b

# Colors for windspeed bins
colors = ['navy', 'orange', 'red', 'green']

# Initialize storage for binned mass data
grouped_mass_values_CDP = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds_mass_CDP = {i: [] for i in range(len(windspeed_bins))}

# Match mass data with windspeed bins
for mass_entry in filtered_dry_mass_inf_CDP:
    date = mass_entry['Date']
    BCB_start = mass_entry['BCB_start']
    BCB_stop = mass_entry['BCB_stop']
    
    # Find corresponding windspeed entry
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) & 
        (df_combined['BCB_start'] == BCB_start) & 
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    if windspeed_entry.empty:
        continue  # Skip if no matching windspeed data

    windspeed = windspeed_entry['Windspeed'].values[0]
    mass_value = mass_entry['Dry Mass (µg/m³)']  # Extract mass value

    # Bin the mass value based on windspeed
    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:
            grouped_mass_values_CDP[idx].append(mass_value)
            mean_windspeeds_mass_CDP[idx].append(windspeed)
            break

# Compute average wind speed and mass per bin
avg_windspeeds_mass_CDP = []
total_mass_values_CDP = []

for idx, mass_list in grouped_mass_values_CDP.items():
    if mass_list:
        avg_windspeed = np.mean(mean_windspeeds_mass_CDP[idx])  # Average windspeed
        avg_mass = np.mean(mass_list)  # Average mass

        avg_windspeeds_mass_CDP.append(avg_windspeed)
        total_mass_values_CDP.append(avg_mass)

# Convert to numpy arrays
windspeed_values_mass_CDP = np.array(avg_windspeeds_mass_CDP)
total_mass_values_CDP = np.array(total_mass_values_CDP)

# Perform linear regression
popt_mass_CDP, _ = curve_fit(linear_model, windspeed_values_mass_CDP, total_mass_values_CDP)
m_fit_mass_CDP, b_fit_mass_CDP = popt_mass_CDP

# Compute R² value
residuals_mass_CDP = total_mass_values_CDP - linear_model(windspeed_values_mass_CDP, *popt_mass_CDP)
ss_res_mass_CDP = np.sum(residuals_mass_CDP**2)
ss_tot_mass_CDP = np.sum((total_mass_values_CDP - np.mean(total_mass_values_CDP))**2)
r_squared_mass_CDP = 1 - (ss_res_mass_CDP / ss_tot_mass_CDP)

# Plot Wind Speed vs. Total Dry Mass (CDP)
plt.figure(figsize=(8, 6))
for idx in range(len(windspeed_bins)):
    plt.scatter(windspeed_values_mass_CDP[idx], total_mass_values_CDP[idx], 
                color="green", s=100, edgecolor='black', zorder=3)

# Fit line
x_fit_mass_CDP = np.linspace(min(windspeed_values_mass_CDP), max(windspeed_values_mass_CDP), 100)
y_fit_mass_CDP = linear_model(x_fit_mass_CDP, *popt_mass_CDP)
plt.plot(x_fit_mass_CDP, y_fit_mass_CDP, 'r-', label=f'Fit: y = {m_fit_mass_CDP:.3f}x + {b_fit_mass_CDP:.3f}, R² = {r_squared_mass_CDP:.2f}')

# Labels & Titles
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=16, fontweight='bold')
plt.ylabel("Total Dry Mass (µg/m³)", fontsize=16, fontweight='bold')
plt.title("Below Cloud Base January - June 2022", fontsize=16, fontweight='bold')

# Legend
legend_labels_mass_CDP = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[idx], markersize=10, 
               label=f"{windspeed_values_mass_CDP[idx]:.1f} m/s") for idx in range(len(windspeed_bins))
]
plt.legend(handles=legend_labels_mass_CDP + [plt.Line2D([0], [0], color='black', 
               label=f'Fit: y = {m_fit_mass_CDP:.3f}x + {b_fit_mass_CDP:.3f}, R² = {r_squared_mass_CDP:.2f}')], 
           title="Wind Speed Bins", title_fontsize=14, fontsize=13)

# Final Plot Settings
plt.tight_layout()
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.ylim(0, 35)
plt.xlim(0, 10)
plt.show()

# Print regression results
print(f"Slope (m): {m_fit_mass_CDP:.3f}")
print(f"Intercept (b): {b_fit_mass_CDP:.3f}")
print(f"R² value: {r_squared_mass_CDP:.2f}")

#%%
#adding the R value and error bars 


# ✅ Define linear regression function
def linear_model(x, m, b):
    return m * x + b

# Colors for windspeed bins
colors = ['navy', 'orange', 'red', 'green']

# Initialize storage for binned mass data
grouped_mass_values_CDP = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds_mass_CDP = {i: [] for i in range(len(windspeed_bins))}

# Match mass data with windspeed bins
for mass_entry in filtered_dry_mass_inf_CDP:
    date = mass_entry['Date']
    BCB_start = mass_entry['BCB_start']
    BCB_stop = mass_entry['BCB_stop']
    
    # Find corresponding windspeed entry
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) & 
        (df_combined['BCB_start'] == BCB_start) & 
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    if windspeed_entry.empty:
        continue  # Skip if no matching windspeed data

    windspeed = windspeed_entry['Windspeed'].values[0]
    mass_value = mass_entry['Dry Mass (µg/m³)']  # Extract mass value

    # Bin the mass value based on windspeed
    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:
            grouped_mass_values_CDP[idx].append(mass_value)
            mean_windspeeds_mass_CDP[idx].append(windspeed)
            break

# Compute average wind speed, mass, and error bars per bin
avg_windspeeds_mass_CDP = []
total_mass_values_CDP = []
standard_errors_mass_CDP = []

for idx, mass_list in grouped_mass_values_CDP.items():
    if mass_list:
        avg_windspeed = np.mean(mean_windspeeds_mass_CDP[idx])  # Average windspeed
        avg_mass = np.mean(mass_list)  # Average mass
        std_mass = np.std(mass_list, ddof=1)  # Standard deviation
        N_mass = len(mass_list)  # Number of legs in this bin
        SE_mass = std_mass / np.sqrt(N_mass)  # Standard Error

        avg_windspeeds_mass_CDP.append(avg_windspeed)
        total_mass_values_CDP.append(avg_mass)
        standard_errors_mass_CDP.append(SE_mass)

# Convert to numpy arrays
windspeed_values_mass_CDP = np.array(avg_windspeeds_mass_CDP)
total_mass_values_CDP = np.array(total_mass_values_CDP)
standard_errors_mass_CDP = np.array(standard_errors_mass_CDP)

# ✅ Perform linear regression
popt_mass_CDP, _ = curve_fit(linear_model, windspeed_values_mass_CDP, total_mass_values_CDP)
m_fit_mass_CDP, b_fit_mass_CDP = popt_mass_CDP

# ✅ Compute R² value
residuals_mass_CDP = total_mass_values_CDP - linear_model(windspeed_values_mass_CDP, *popt_mass_CDP)
ss_res_mass_CDP = np.sum(residuals_mass_CDP**2)
ss_tot_mass_CDP = np.sum((total_mass_values_CDP - np.mean(total_mass_values_CDP))**2)
r_squared_mass_CDP = 1 - (ss_res_mass_CDP / ss_tot_mass_CDP)

# ✅ Compute Pearson correlation coefficient (R)
pearson_corr_mass_CDP, _ = pearsonr(windspeed_values_mass_CDP, total_mass_values_CDP)

# ✅ Plot Wind Speed vs. Total Dry Mass (CDP)
plt.figure(figsize=(8, 6))

# ✅ Plot data points with error bars (Green for CDP)
plt.errorbar(windspeed_values_mass_CDP, total_mass_values_CDP, 
             yerr=standard_errors_mass_CDP, fmt='o', color='green', 
             markersize=10, capsize=5, capthick=2, label="CDP", 
             ecolor='black', elinewidth=1.5, zorder=3)

# ✅ Fit line
x_fit_mass_CDP = np.linspace(min(windspeed_values_mass_CDP), max(windspeed_values_mass_CDP), 100)
y_fit_mass_CDP = linear_model(x_fit_mass_CDP, *popt_mass_CDP)
plt.plot(x_fit_mass_CDP, y_fit_mass_CDP, 'black', linewidth=2, 
         label=f'Fit: y = {m_fit_mass_CDP:.3f}x + {b_fit_mass_CDP:.3f}\nR² = {r_squared_mass_CDP:.2f}, R = {pearson_corr_mass_CDP:.2f}')

# ✅ Labels & Titles
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=16, fontweight='bold')
plt.ylabel("Total Dry Mass (µg/m³)", fontsize=16, fontweight='bold')
plt.title("CDP Wind Speed vs. Total Dry Mass Correlation", fontsize=16, fontweight='bold')

# ✅ Legend: Only R², R, and the regression equation
plt.legend(fontsize=13, title_fontsize=14, loc='best', frameon=True)

# ✅ Final Plot Settings
plt.tight_layout()
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.ylim(0, 35)
plt.xlim(0, 10)
plt.show()

# ✅ Print Results
print(f"Slope (m): {m_fit_mass_CDP:.3f}")
print(f"Intercept (b): {b_fit_mass_CDP:.3f}")
print(f"R² value: {r_squared_mass_CDP:.2f}")
print(f"R value (Pearson correlation): {pearson_corr_mass_CDP:.3f}")


# %%
#combining mass CDP and CAS

# ✅ Define linear regression function
def linear_model(x, m, b):
    return m * x + b

# ✅ Prepare CAS Data (Blue)
windspeed_values_mass = np.array(avg_windspeeds_mass)
total_mass_values = np.array(total_mass_values)
standard_errors_mass = np.array(standard_errors_mass)

popt_mass, _ = curve_fit(linear_model, windspeed_values_mass, total_mass_values)
m_fit_mass, b_fit_mass = popt_mass

# Compute R² and R for CAS
residuals_mass = total_mass_values - linear_model(windspeed_values_mass, *popt_mass)
ss_res_mass = np.sum(residuals_mass**2)
ss_tot_mass = np.sum((total_mass_values - np.mean(total_mass_values))**2)
r_squared_mass = 1 - (ss_res_mass / ss_tot_mass)
r_value_mass, _ = pearsonr(windspeed_values_mass, total_mass_values)

# ✅ Prepare CDP Data (Green)
windspeed_values_mass_CDP = np.array(avg_windspeeds_mass_CDP)
total_mass_values_CDP = np.array(total_mass_values_CDP)
standard_errors_mass_CDP = np.array(standard_errors_mass_CDP)

popt_mass_CDP, _ = curve_fit(linear_model, windspeed_values_mass_CDP, total_mass_values_CDP)
m_fit_mass_CDP, b_fit_mass_CDP = popt_mass_CDP

# Compute R² and R for CDP
residuals_mass_CDP = total_mass_values_CDP - linear_model(windspeed_values_mass_CDP, *popt_mass_CDP)
ss_res_mass_CDP = np.sum(residuals_mass_CDP**2)
ss_tot_mass_CDP = np.sum((total_mass_values_CDP - np.mean(total_mass_values_CDP))**2)
r_squared_mass_CDP = 1 - (ss_res_mass_CDP / ss_tot_mass_CDP)
r_value_mass_CDP, _ = pearsonr(windspeed_values_mass_CDP, total_mass_values_CDP)

# ✅ Plot CAS & CDP Dry Mass vs. Wind Speed
plt.figure(figsize=(8, 6))

# ✅ Plot CAS data with error bars (Blue)
plt.errorbar(windspeed_values_mass, total_mass_values, 
             yerr=standard_errors_mass, fmt='o', color='blue', 
             markersize=10, capsize=5, capthick=2, label="CAS", 
             ecolor='black', elinewidth=1.5, zorder=3)

# ✅ Plot CDP data with error bars (Green)
plt.errorbar(windspeed_values_mass_CDP, total_mass_values_CDP, 
             yerr=standard_errors_mass_CDP, fmt='o', color='green', 
             markersize=10, capsize=5, capthick=2, label="CDP", 
             ecolor='black', elinewidth=1.5, zorder=3)

# ✅ Fit lines
x_fit = np.linspace(0, 10, 100)  # Range of wind speeds
y_fit_mass = linear_model(x_fit, *popt_mass)
y_fit_mass_CDP = linear_model(x_fit, *popt_mass_CDP)

plt.plot(x_fit, y_fit_mass, 'r-', linewidth=2, 
         label=f'CAS: y = {m_fit_mass:.3f}x + {b_fit_mass:.3f}\nR² = {r_squared_mass:.2f}, R = {r_value_mass:.2f}')
plt.plot(x_fit, y_fit_mass_CDP, 'black', linewidth=2, 
         label=f'CDP: y = {m_fit_mass_CDP:.3f}x + {b_fit_mass_CDP:.3f}\nR² = {r_squared_mass_CDP:.2f}, R = {r_value_mass_CDP:.2f}')

# ✅ Labels & Titles
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=19, fontweight='bold')
plt.ylabel("Total Dry Mass (µg/m³)", fontsize=19, fontweight='bold')
plt.title("Wind Speed vs. Dry Mass Correlation", fontsize=19, fontweight='bold')

# ✅ Legend: Only CAS/CDP Labels & Regression Equations
plt.legend(fontsize=13, title_fontsize=14, loc='best', frameon=True)

# ✅ Final Plot Settings
plt.tight_layout()
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.ylim(0, 35)
plt.xlim(0, 10)
plt.show()

# ✅ Print Regression Results
print("=== CAS Regression Results ===")
print(f"Slope (m): {m_fit_mass:.3f}")
print(f"Intercept (b): {b_fit_mass:.3f}")
print(f"R² value: {r_squared_mass:.2f}")
print(f"R value (Pearson correlation): {r_value_mass:.3f}")

print("\n=== CDP Regression Results ===")
print(f"Slope (m): {m_fit_mass_CDP:.3f}")
print(f"Intercept (b): {b_fit_mass_CDP:.3f}")
print(f"R² value: {r_squared_mass_CDP:.2f}")
print(f"R value (Pearson correlation): {r_value_mass_CDP:.3f}")

# %%
