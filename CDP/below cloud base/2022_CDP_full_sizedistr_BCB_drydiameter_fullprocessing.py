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
from scipy.stats import gaussian_kde
from scipy.integrate import quad

#%%

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
#Summary Data and meteorlogical data import
col_name = ['Time_mid', 'Latitude', 'Longitude', 'GPS_altitude', 'Pressure_Altitude',
             'Pitch', 'Roll', 'True_Heading', 'True_Air_Speed', 
             'Static_Air_Temp', 'IR_Surf_Temp', 'Static_Pressure',
             'Wind_Speed']
summary=[]
dates_sum = ['2022-01-11', '2022-01-12','2022-01-15', '2022-01-18', 
             '2022-01-19', '2022-01-24', '2022-01-26', '2022-01-27',
             '2022-02-01', '2022-02-02', '2022-02-03', '2022-02-05', 
             '2022-02-15', '2022-02-16', '2022-02-19', '2022-02-22',
             '2022-02-26', #'2022-03-02',
               '2022-03-03', '2022-03-04', #'2022-03-07', 
             '2022-03-13', '2022-03-14', '2022-03-18', '2022-03-22',
             '2022-03-26', '2022-03-28', '2022-03-29', #'2022-05-03'
             '2022-05-05', '2022-05-10', '2022-05-16','2022-05-17', '2022-05-18',
             '2022-05-20','2022-05-21', '2022-05-31', '2022-06-02', 
             '2022-06-03', '2022-06-05', '2022-06-07', '2022-06-08', 
             '2022-06-10','2022-06-11', '2022-06-13', '2022-06-14',
             '2022-06-17', '2022-06-18']

for date in dates_sum:
    datestr = date.replace('-', '')
    fname_sum = sorted(glob.glob(f'/home/disk/eos4/kathem24/activate/data/MET/2022/Summary/csv/ACTIVATE-SUMMARY_HU25_{datestr}_R*.csv'), reverse=True)
    # print(date)
    # print(fname_sum)

    run = 1
    for file_path in fname_sum: 
        num_file_paths = len(fname_sum)

        #num = index = dates_sum.index(date)
        
        # try:
        if date > '2022-01-12':
            df_sum = pd.read_csv(file_path, skiprows=47, quoting=csv.QUOTE_NONE)
            # elif date > '2022-05-03':
            #     df_sum = pd.read_csv(file_path, skiprows=47, quoting=csv.QUOTE_NONE)
            # elif date ==  '2022-05-03':
            #     df_sum = pd.read_csv(file_path, skiprows=51, quoting=csv.QUOTE_NONE)
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
# %%
#Import the flight leg time stamps and leg lengths
leg_data = []
leg_name=['Time_Start', '  Time_Stop', '  Julian_Day', 
          '  Date', '  LegIndex']

dates_legs= ['2022-01-11', '2022-01-12','2022-01-15', '2022-01-18', 
             '2022-01-19', '2022-01-24', '2022-01-26', '2022-01-27',
             '2022-02-01', '2022-02-02', '2022-02-03', '2022-02-05', 
             '2022-02-15', '2022-02-16', '2022-02-19',  '2022-02-22',
             '2022-02-26', #'2022-03-02', 
             '2022-03-03', '2022-03-04', #'2022-03-07', 
             '2022-03-13', '2022-03-14', '2022-03-18', '2022-03-22',
             '2022-03-26', '2022-03-28', '2022-03-29', #'2022-05-03',
             '2022-05-05', '2022-05-10','2022-05-16', '2022-05-17', '2022-05-18',
             '2022-05-20', '2022-05-21', '2022-05-31', '2022-06-02', 
             '2022-06-03', '2022-06-05', '2022-06-07', '2022-06-08', 
             '2022-06-10','2022-06-11','2022-06-13', '2022-06-14',
             '2022-06-17', '2022-06-18']

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

# %%
#2D-S Data Import for total checking number concentration to remove cloudy data from our
#clear sky analysis#
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
        reverse=False  # Ensure L1 comes before L2
    )

    print(f"Processing {date}... Found files: {file_paths}")

    run = 1
    dfs_for_date = []

    for file_path in file_paths:
        # Detect the correct header row dynamically
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
dates_h20 = ['2022-01-11', '2022-01-12','2022-01-15', '2022-01-18', 
             '2022-01-19', '2022-01-24', '2022-01-26', '2022-01-27',
             '2022-02-01', '2022-02-02', '2022-02-03', '2022-02-05', 
             '2022-02-15', '2022-02-16', '2022-02-19', '2022-02-22',
             '2022-02-26', #'2022-03-02', 
             '2022-03-03', '2022-03-04', #'2022-03-07', 
             '2022-03-13', '2022-03-14', '2022-03-18', '2022-03-22',
             '2022-03-26', '2022-03-28', '2022-03-29', #'2022-05-03'
             '2022-05-05', '2022-05-10', '2022-05-16','2022-05-17', '2022-05-18',
             '2022-05-20','2022-05-21', '2022-05-31', '2022-06-02', 
             '2022-06-03', '2022-06-05', '2022-06-07', '2022-06-08', 
             '2022-06-10','2022-06-11', '2022-06-13', '2022-06-14',
             '2022-06-17', '2022-06-18']
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
#Import the instrument data for the cloud droplet probe 

#Make sure to work with bins 0-30 for the coarse mode aerosol
bin_name = ['CDP_Bin00', 'CDP_Bin01', 'CDP_Bin02', 'CDP_Bin03', 
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
#Using align to align the timestamps of the two datasets
# master_CDP_BCB = []
# leg_info = []

# for i in range(len(dates_legs)):
#     date = dates_legs[i]
#     leg_dict = leg_data[i]

#     flight_date = leg_dict['Date']
#     BCB_start = leg_dict['LegIndex_02']['StartTimes']
#     BCB_stop = leg_dict['LegIndex_02']['StopTimes']

#     CDP_flight = CDP_1Hz[i]
#     twoDS_flight = twoDS[i]

#     times = np.array(CDP_flight.Time_Start.values, dtype=float)
#     times1 = np.array(twoDS_flight.Time_Start.values, dtype=float)
#     lwc = CDP_flight.LWC_CDP.values
#     N_total = twoDS_flight['N-total_2DS'].values

#     # Efficiently align timestamps
#     times1_mapping = {time: idx for idx, time in enumerate(times1)}  # Create a dictionary for fast lookup
#     CDP_indices = [idx for idx, time in enumerate(times) if time in times1_mapping]
#     twods_indices = [times1_mapping[time] for time in times if time in times1_mapping]

#     aligned_lwc = lwc[CDP_indices]
#     aligned_N_total = N_total[twods_indices]
#     aligned_times = times[CDP_indices]

#     total_BCB_means = []

#     for k in range(len(BCB_start)):
#         start20 = int(BCB_start[k])
#         end20 = int(BCB_stop[k])

#         data_labels = []
#         BCB_means = []  # List to store the means for this BCB interval

#         # Find start and end indices in aligned times
#         interval_indices = [
#             idx for idx, aligned_time in enumerate(aligned_times)
#             if start20 <= int(aligned_time) <= end20
#         ]

#         if interval_indices:
#             for idx in interval_indices:
#                 lwc_val = aligned_lwc[idx]
#                 N_val = aligned_N_total[idx]
#                 lwc_label = 'Y' if 0 <= lwc_val <= 0.0025 else 'N'
#                 N_label = 'Y' if 0 <= N_val <= 100 else 'N'
#                 label = 'Y' if lwc_label == 'Y' and N_label == 'Y' else 'N'
#                 data_labels.append(label)

#                 # Collect bin values
#                 bin_values = [
#                     CDP_flight[f'CDP_Bin{bin_label:02d}'].values[CDP_indices[idx]]
#                     for bin_label in range(30)
#                 ]
#                 BCB_means.append(bin_values)

#             if BCB_means:
#                 total_BCB_means.append(BCB_means)

#             leg_info.append({
#                 'Date': date,
#                 'BCB_start': start20,
#                 'BCB_stop': end20,
#                 'Data_Labels': data_labels,
#             })

#     master_CDP_BCB.append(total_BCB_means)

# # Print leg_info or use it as needed
# for leg in leg_info:
#     print(f"Date: {leg['Date']}, Start: {leg['BCB_start']}, Stop: {leg['BCB_stop']}, Data Labels: {leg['Data_Labels']}")
#%%
##Checking where or loc instead of align
master_CDP_BCB = []
leg_info = []

for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict = leg_data[i]

    flight_date = leg_dict['Date']
    BCB_start = leg_dict['LegIndex_02']['StartTimes']
    BCB_stop = leg_dict['LegIndex_02']['StopTimes']

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

    total_BCB_means = []

    for k in range(len(BCB_start)):
        start20 = BCB_start[k]
        end20 = BCB_stop[k]

        # Find indices in the BCB range for CDP and TwoDS
        CDP_indices_in_range = np.where((CDP_times >= start20) & (CDP_times <= end20))[0]
        TwoDS_indices_in_range = np.where((TwoDS_times >= start20) & (TwoDS_times <= end20))[0]

        if CDP_indices_in_range.size > 0 and TwoDS_indices_in_range.size > 0:
            data_labels = []
            BCB_means = []

            for CDP_idx, TwoDS_idx in zip(CDP_indices_in_range, TwoDS_indices_in_range):
                lwc_val = CDP_lwc[CDP_idx]
                N_val = TwoDS_N_total[TwoDS_idx]

                # Assign labels
                lwc_label = 'Y' if 0 <= lwc_val <= 0.0025 else 'N'
                N_label = 'Y' if 0 <= N_val <= 100 else 'N'
                label = 'Y' if lwc_label == 'Y' and N_label == 'Y' else 'N'
                data_labels.append(label)

                # Collect bin values
                bin_values = [CDP_bins[f'CDP_Bin{bin_label:02d}'][CDP_idx] for bin_label in range(30)]
                BCB_means.append(bin_values)

            if BCB_means:
                total_BCB_means.append(BCB_means)

            leg_info.append({
                'Date': date,
                'BCB_start': start20,
                'BCB_stop': end20,
                'Data_Labels': data_labels,
            })

    master_CDP_BCB.append(total_BCB_means)

# Print leg_info or use it as needed
for leg in leg_info:
    print(f"Date: {leg['Date']}, Start: {leg['BCB_start']}, Stop: {leg['BCB_stop']}, Data Labels: {leg['Data_Labels']}")


#%%
#Double check the number of legs associated with each date to compare with the CAS legs 


from collections import Counter

# Count the number of legs for each date
leg_count = Counter([leg['Date'] for leg in leg_info])

# Print the count for each date
print("Number of legs associated with each date:")
for date, count in sorted(leg_count.items()):
    print(f"Date: {date}, Number of Legs: {count}")

#%%
master_CDP_BCB = []
leg_info = []

for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict = leg_data[i]

    flight_date = leg_dict['Date']
    BCB_start = leg_dict['LegIndex_02']['StartTimes']
    BCB_stop = leg_dict['LegIndex_02']['StopTimes']

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

    total_BCB_means = []

    for k in range(len(BCB_start)):
        start20 = BCB_start[k]
        end20 = BCB_stop[k]

        bin_means = {f'Bin{bin_label:02d}_Y_mean': [] for bin_label in range(30)}
        bin_means.update({f'Bin{bin_label:02d}_N_mean': [] for bin_label in range(30)})
        bin_means.update({'Date': date, 'BCB_start': start20, 'BCB_stop': end20})

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
                    bin_means[bin_key].append(bins[f'CDP_Bin{bin_label:02d}'][cdp_idx])

        # Calculate means
        for bin_label in range(30):
            for label in ['Y', 'N']:
                bin_key = f'Bin{bin_label:02d}_{label}_mean'
                if bin_means[bin_key]:
                    bin_means[bin_key] = np.nanmean(bin_means[bin_key])

        total_BCB_means.append(bin_means)

    master_CDP_BCB.append(total_BCB_means)

# Print or use master_CDP_BCB as needed
for item in master_CDP_BCB:
    for bin_means in item:
        print(f"Date: {bin_means['Date']}, Start: {bin_means['BCB_start']}, Stop: {bin_means['BCB_stop']}")
        for bin_label in range(30):
            for label in ['Y', 'N']:
                bin_key = f'Bin{bin_label:02d}_{label}_mean'
                print(f"   {bin_key}: {bin_means[bin_key]}")

#%%
#uses align to align the timestamps of the two datasets
# master_CDP_BCB = []
# leg_info = []

# for i in range(len(dates_legs)):
#     date = dates_legs[i]
#     leg_dict = leg_data[i]

#     flight_date = leg_dict['Date']
#     BCB_start = np.array(leg_dict['LegIndex_02']['StartTimes'], dtype=float)
#     BCB_stop = np.array(leg_dict['LegIndex_02']['StopTimes'], dtype=float)

#     CDP_flight = CDP_1Hz[i]
#     twoDS_flight = twoDS[i]

#     # Convert times to float and round for alignment
#     times = np.round(np.array(CDP_flight.Time_Start.values, dtype=float), 3)
#     times1 = np.round(np.array(twoDS_flight.Time_Start.values, dtype=float), 3)

#     lwc = np.array(CDP_flight.LWC_CDP.values, dtype=float)
#     N_total = np.array(twoDS_flight['N-total_2DS'].values, dtype=float)

#     # Create a mapping for TwoDS times with indices for fast lookup
#     times_dict = {time: idx for idx, time in enumerate(times1)}
#     aligned_indices = [
#         (idx, times_dict[time]) for idx, time in enumerate(times) if time in times_dict
#     ]

#     # Extract aligned indices and validate
#     if aligned_indices:
#         CDP_indices, twods_indices = zip(*aligned_indices)
#         CDP_indices = np.array(CDP_indices, dtype=int)
#         twods_indices = np.array(twods_indices, dtype=int)
#     else:
#         CDP_indices = np.array([], dtype=int)
#         twods_indices = np.array([], dtype=int)

#     aligned_lwc = lwc[CDP_indices] if CDP_indices.size > 0 else np.array([])
#     aligned_N_total = N_total[twods_indices] if twods_indices.size > 0 else np.array([])
#     aligned_times = times[CDP_indices] if CDP_indices.size > 0 else np.array([])

#     # Pre-fetch all bin values
#     bins = {f'CDP_Bin{bin_label:02d}': np.array(CDP_flight[f'CDP_Bin{bin_label:02d}'].values, dtype=float) for bin_label in range(30)}
#     bins = {key: values[CDP_indices] for key, values in bins.items()} if CDP_indices.size > 0 else {key: np.array([]) for key in bins}

#     total_BCB_means = []

#     for k in range(len(BCB_start)):
#         start20 = BCB_start[k]
#         end20 = BCB_stop[k]

#         bin_means = {f'Bin{bin_label:02d}_Y_mean': [] for bin_label in range(30)}
#         bin_means.update({f'Bin{bin_label:02d}_N_mean': [] for bin_label in range(30)})
#         bin_means.update({'Date': date, 'BCB_start': start20, 'BCB_stop': end20})

#         # Find indices within the current interval
#         indices_in_range = np.where((aligned_times >= start20) & (aligned_times <= end20))[0]

#         # Ensure indices are within valid bounds
#         max_index = len(aligned_N_total)
#         indices_in_range = indices_in_range[indices_in_range < max_index]

#         if indices_in_range.size > 0:
#             for j in indices_in_range:
#                 lwc_val = aligned_lwc[j]
#                 N_val = aligned_N_total[j]
#                 lwc_label = 'Y' if 0 <= lwc_val <= 0.0025 else 'N'
#                 N_label = 'Y' if 0 <= N_val <= 100 else 'N'
#                 label = 'Y' if lwc_label == 'Y' and N_label == 'Y' else 'N'

#                 for bin_label in range(30):
#                     bin_key = f'Bin{bin_label:02d}_{label}_mean'
#                     bin_means[bin_key].append(bins[f'CDP_Bin{bin_label:02d}'][j])

#             # Calculate means
#             for bin_label in range(30):
#                 for label in ['Y', 'N']:
#                     bin_key = f'Bin{bin_label:02d}_{label}_mean'
#                     if bin_means[bin_key]:
#                         bin_means[bin_key] = np.nanmean(bin_means[bin_key])
#                     else:
#                         bin_means[bin_key] = np.nan

#         total_BCB_means.append(bin_means)

#     master_CDP_BCB.append(total_BCB_means)

# # Print or use master_CDP_BCB as needed
# for item in master_CDP_BCB:
#     for bin_means in item:
#         print(f"Date: {bin_means['Date']}, Start: {bin_means['BCB_start']}, Stop: {bin_means['BCB_stop']}")
#         for bin_label in range(30):
#             for label in ['Y', 'N']:
#                 bin_key = f'Bin{bin_label:02d}_{label}_mean'
#                 print(f"   {bin_key}: {bin_means[bin_key]}")

#%%# 

# bin_name = [
#     'dNdlogD_total_003_2DS', 'dNdlogD_total_004_2DS', 
#     'dNdlogD_total_005_2DS', 'dNdlogD_total_006_2DS'
# ]

# twoDS = []
# dates_twoDS = [
#     '2022-01-11', '2022-01-12', '2022-01-15', '2022-01-18', 
#     '2022-01-19', '2022-01-24', '2022-01-26', '2022-01-27',
#     '2022-02-01', '2022-02-02', '2022-02-03', '2022-02-05', 
#     '2022-02-15', '2022-02-16', '2022-02-19', '2022-02-22',
#     '2022-02-26', '2022-03-03', '2022-03-04', '2022-03-13', 
#     '2022-03-14', '2022-03-18', '2022-03-22', '2022-03-26',
#     '2022-03-28', '2022-03-29', '2022-05-05', '2022-05-10',
#     '2022-05-16', '2022-05-17', '2022-05-18', '2022-05-20', 
#     '2022-05-21', '2022-05-31', '2022-06-02', '2022-06-03', 
#     '2022-06-05', '2022-06-07', '2022-06-08', '2022-06-10',
#     '2022-06-11', '2022-06-13', '2022-06-14', '2022-06-17', 
#     '2022-06-18'
# ]

# for date in dates_twoDS:
#     datestr = date.replace('-', '')
#     file_paths = sorted(
#         glob.glob(f'/home/disk/eos4/kathem24/activate/data/twoDspectrometer/horizontal/csv/ACTIVATE-2DS-H-Arm_HU25_{datestr}_R*.csv'), 
#         reverse=False  # Ensure L1 comes before L2
#     )

#     print(f"Processing {date}... Found files: {file_paths}")

#     run = 1
#     dfs_for_date = []

#     for file_path in file_paths:
#         # Detect the correct header row dynamically
#         header_row = None
#         with open(file_path, 'r') as f:
#             for i, line in enumerate(f):
#                 if 'Time_Start' in line and 'LWC_2DS' in line:
#                     header_row = i
#                     print(f"Detected header row for {file_path}: Line {header_row}")
#                     print(f"Header content: {line.strip()}")
#                     break

#         if header_row is None:
#             print(f"Error: Could not find header row in file {file_path}")
#             continue

#         try:
#             # Read the file using the identified header row
#             df_2DS = pd.read_csv(
#                 file_path, 
#                 skiprows=header_row, 
#                 quoting=csv.QUOTE_NONE,
#                 engine='python'
#             )

#             # Clean up column names
#             df_2DS.columns = df_2DS.columns.str.strip('"')
#             print(f"Columns for {file_path}: {df_2DS.columns[:10]}")

#             # Replace -9999 values with 0
#             df_2DS.replace([-9999, -9999.0], 0, inplace=True)
#             for col in df_2DS.select_dtypes(include=['object']).columns:
#                 df_2DS[col] = df_2DS[col].str.strip('"')

#             # Append the cleaned DataFrame to the list
#             dfs_for_date.append(df_2DS)

#         except Exception as e:
#             print(f"Error processing file {file_path}: {e}")


#     # Combine DataFrames for this date, if there are multiple files
#     if len(dfs_for_date) == 2:
#         df4, df5 = dfs_for_date[0], dfs_for_date[1]
#         combined_df = pd.concat([df4, df5], ignore_index=True)
#         twoDS.append(combined_df)
#         print(f"Combined DataFrame for {date} (first 5 rows):")
#         print(combined_df.head())
#     elif len(dfs_for_date) == 1:
#         twoDS.append(dfs_for_date[0])
#         print(f"Single file DataFrame for {date} (first 5 rows):")
#         print(dfs_for_date[0].head())
#     else:
#         print(f"No valid data for {date}")

# # Final check
# print(f"Total dates processed: {len(twoDS)}")

#%%
# %%
Y_CDP_calc = []
N_CDP_calc = []

for flight_data in master_CDP_BCB:
    for bin_means in flight_data:
        Y_calc = {'Date': bin_means['Date'], 'BCB_start': bin_means['BCB_start'], 'BCB_stop': bin_means['BCB_stop']}
        N_calc = {'Date': bin_means['Date'], 'BCB_start': bin_means['BCB_start'], 'BCB_stop': bin_means['BCB_stop']}
        
        for bin_label in range(0, 30):
            bin_key_Y = f'Bin{bin_label:02d}_Y_mean'
            bin_key_N = f'Bin{bin_label:02d}_N_mean'

            Y_calc[bin_key_Y] = np.nanmean(bin_means[bin_key_Y]) * Logg[bin_label]
            N_calc[bin_key_N] = np.nanmean(bin_means[bin_key_N]) * Logg[bin_label]

        Y_CDP_calc.append(Y_calc)
        N_CDP_calc.append(N_calc)
#%%
# Generate all_bin_means
all_bin_means = []

for entry in Y_CDP_calc:
    
    bin_means = []
    for i in range(0, 30): 
        key = f'Bin{i:02d}_Y_mean'  
        if key in entry: 
            bin_means.append(entry[key]) 
        else:
            bin_means.append(np.nan) 

    
    all_bin_means.append({
        'Date': entry['Date'],
        'BCB_start': entry.get('BCB_start', np.nan),
        'BCB_stop': entry.get('BCB_stop', np.nan),
        'Bin_means': bin_means
    })

# Create a list of unique dates
unique_dates = sorted(set(entry['Date'] for entry in all_bin_means))
colors = cm.viridis(np.linspace(0, 1, len(unique_dates)))
date_color_map = {date: color for date, color in zip(unique_dates, colors)}

# Plot the data
plt.figure(figsize=(10, 6))
added_dates = set() 

for entry in all_bin_means:
    bin_means = np.array(entry['Bin_means'])
    valid_indices = ~np.isnan(bin_means) 
    date = entry['Date']
    color = date_color_map[date]
    
    if date not in added_dates:
        plt.scatter(np.array(bin_center)[valid_indices], bin_means[valid_indices], color=color, marker='o', label=date)
        added_dates.add(date)
    else:
        plt.scatter(np.array(bin_center)[valid_indices], bin_means[valid_indices], color=color, marker='o')

plt.xlabel('Bin centers diameter (um)', fontsize=12, fontweight='bold')
plt.ylabel('Clear mean droplet concentration \n (/cm^3/um)', fontsize=12, fontweight='bold')
plt.title('Below cloud base \n January-June 2022', fontsize=12, fontweight='bold')
plt.xticks(np.arange(0, 50, 2))
num_cols = 7
# plt.legend(title='Date', ncol=num_cols, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True)
plt.tight_layout()
plt.show()
#%%
#Fit an exponential to the entire distribution
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

all_bin_means = []

for entry in Y_CDP_calc:
    bin_means = []
    for i in range(0, 30): 
        key = f'Bin{i:02d}_Y_mean'
        if key in entry:  
            bin_means.append(entry[key]) 
        else:
            bin_means.append(np.nan) 

    all_bin_means.append({
        'Date': entry['Date'],
        'BCB_start': entry.get('BCB_start', np.nan),
        'BCB_stop': entry.get('BCB_stop', np.nan),
        'Bin_means': bin_means
    })

unique_dates = sorted(set(entry['Date'] for entry in all_bin_means))
colors = cm.viridis(np.linspace(0, 1, len(unique_dates)))
date_color_map = {date: color for date, color in zip(unique_dates, colors)}

plt.figure(figsize=(10, 6))
added_dates = set()

for entry in all_bin_means:
    bin_means = np.array(entry['Bin_means'])
    valid_indices = ~np.isnan(bin_means)
    date = entry['Date']
    color = date_color_map[date]
    
    if date not in added_dates:
        plt.scatter(np.array(bin_center)[valid_indices], bin_means[valid_indices], color=color, marker='o', label=date)
        added_dates.add(date)
    else:
        plt.scatter(np.array(bin_center)[valid_indices], bin_means[valid_indices], color=color, marker='o')

flat_bin_centers = np.tile(bin_center, len(all_bin_means))
flat_bin_means = np.concatenate([np.array(entry['Bin_means']) for entry in all_bin_means])
flat_bin_centers = np.array(flat_bin_centers, dtype=float)
flat_bin_means = np.array(flat_bin_means, dtype=float)
valid_indices = ~np.isnan(flat_bin_means)
flat_bin_centers = flat_bin_centers[valid_indices]
flat_bin_means = flat_bin_means[valid_indices]
popt, pcov = curve_fit(exponential, flat_bin_centers, flat_bin_means, p0=(1, 1))

n0 = popt[0]
D = popt[1]

print(f"Intercept (n0): {n0:.2e}")
print(f"E-folding diameter (D): {D:.2f} um")

x_fit = np.linspace(min(bin_center), max(bin_center), 100)
y_fit = exponential(x_fit, *popt)
plt.plot(x_fit, y_fit, color='red', label=f'Exponential fit: y = {n0:.2e} * exp(-x / {D:.2f})')
plt.xlabel('Bin centers diameter (um)', fontsize=12, fontweight='bold')
plt.ylabel('Clear mean droplet concentration \n (/cm^3/um)', fontsize=12, fontweight='bold')
plt.title('Below cloud base  \n January-June 2022', fontsize=12, fontweight='bold')
# plt.yscale('log')
plt.xticks(np.arange(0, 50, 5))
num_cols = 7
#plt.legend(title='Date', ncol=num_cols, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True)
plt.tight_layout()
# plt.ylim(10**-5, 10**1)
plt.show()

#%%
#Fit an exponential to each leg's distribution

def exponential(x, n0, D):
    return n0 * np.exp(-x / D)
all_bin_means = []
dates = []
for entry in Y_CDP_calc:
   
    bin_means = []
    for i in range(0, 30): 
        key = f'Bin{i:02d}_Y_mean'  
        if key in entry:  
            bin_means.append(entry[key]) 
        else:
            bin_means.append(np.nan)
    all_bin_means.append({
        'Date': entry['Date'],
        'BCB_start': entry.get('BCB_start', np.nan),
        'BCB_stop': entry.get('BCB_stop', np.nan),
        'Bin_means': bin_means
    })
    dates.append(entry['Date']) 

unique_dates = sorted(set(dates))
colors = cm.viridis(np.linspace(0, 1, len(unique_dates)))
date_color_map = {date: color for date, color in zip(unique_dates, colors)}

fit_color = 'purple'

plt.figure(figsize=(10, 6))
added_dates = set() 
for entry in all_bin_means:
    bin_means = np.array(entry['Bin_means'])
    valid_indices = ~np.isnan(bin_means)
    date = entry['Date']
    color = date_color_map[date]

    if not valid_indices.any():
        print(f"No valid data for date {date}")
        continue

    bin_centers = np.array(bin_center)[valid_indices]
    bin_means = bin_means[valid_indices]
    
    try:
        popt, _ = curve_fit(exponential, bin_centers, bin_means, p0=(1, 1))
        n0, D = popt
        
        print(f"Date: {date}")
        print(f"  Intercept (n0): {n0:.2e}")
        print(f"  E-folding diameter (D): {D:.2f} um")
        
        x_fit = np.linspace(min(bin_centers), max(bin_centers), 100)
        y_fit = exponential(x_fit, *popt)
        plt.plot(x_fit, y_fit, color=fit_color, label=f'{date} fit: y = {n0:.2e} * exp(-x / {D:.2f})')

    except RuntimeError:
        print(f"Fit could not be performed for date {date}")
    
    if date not in added_dates:
        plt.scatter(bin_centers, bin_means, color=color, marker='o', label=date)
        added_dates.add(date)
    else:
        plt.scatter(bin_centers, bin_means, color=color, marker='o')

plt.xlabel('Bin centers diameter (um)', fontsize=12, fontweight='bold')
plt.ylabel('Clear mean droplet concentration \n (/cm^3/um)', fontsize=12, fontweight='bold')
plt.title('Below cloud base  \n January-June 2022', fontsize=12, fontweight='bold')
plt.yscale('log')
plt.xticks(np.arange(0, 50, 5))
num_cols = 7
#plt.legend(title='Date', ncol=num_cols, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True)
plt.tight_layout()
plt.show()
#%%
#Reshape the distrubtions into typical size distributions

def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

master_BCB_exponential = {}

unique_dates = sorted(set(entry['Date'] for entry in all_bin_means))
colors = cm.viridis(np.linspace(0, 1, len(unique_dates)))
date_color_map = {date: color for date, color in zip(unique_dates, colors)}

plt.figure(figsize=(10, 6))
added_dates = set()

for entry in all_bin_means:
    bin_means = np.array(entry['Bin_means'], dtype=float)
    valid_indices = ~np.isnan(bin_means)
    bin_centers = np.array(bin_center)[valid_indices]
    bin_means = bin_means[valid_indices]
    
    
    if not valid_indices.any():
        print(f"No valid data for date {entry['Date']}")
        continue
    
    
    try:
        popt, _ = curve_fit(exponential, bin_centers, bin_means, p0=(1, 1))
        n0, D = popt
        
    
        if entry['Date'] not in master_BCB_exponential:
            master_BCB_exponential[entry['Date']] = []
        
        master_BCB_exponential[entry['Date']].append({
            'Date': entry['Date'],
            'BCB_start': entry.get('BCB_start', np.nan),
            'BCB_stop': entry.get('BCB_stop', np.nan),
            'n0': n0,
            'D': D
        })
        
        
        print(f"Date: {entry['Date']}")
        print(f"  Intercept (n0): {n0:.2e}")
        print(f"  E-folding diameter (D): {D:.2f} um")
        
    
        x_fit = np.linspace(min(bin_centers), max(bin_centers), 100)
        y_fit = exponential(x_fit, *popt)
        plt.plot(x_fit, y_fit, color='purple', label=f'{entry["Date"]} fit: y = {n0:.2e} * exp(-x / {D:.2f})')

    except RuntimeError:
        print(f"Fit could not be performed for date {entry['Date']}")
    
  
    date = entry['Date']
    if date not in added_dates:
        plt.scatter(bin_centers, bin_means, color=date_color_map[date], marker='o', label=date)
        added_dates.add(date)
    else:
        plt.scatter(bin_centers, bin_means, color=date_color_map[date], marker='o')


plt.xlabel('Bin centers diameter (um)', fontsize=12, fontweight='bold')
plt.ylabel('Clear mean droplet concentration \n (/cm^3/um)', fontsize=12, fontweight='bold')
plt.title('Below cloud base\n January-June 2022', fontsize=12, fontweight='bold')
plt.xticks(np.arange(0, 50, 5))
plt.tight_layout()
plt.show()

#%%
# Print the number of inner dictionaries for each date in master_min_exponential
for date, entries in master_BCB_exponential.items():
    print(f"Date: {date} has {len(entries)} entries")
total_entries = sum(len(entries) for entries in master_BCB_exponential.values())
print(f"Total number of inner dictionaries: {total_entries}")

for date, entries in master_BCB_exponential.items():
    print(f"Date: {date}")
    for entry in entries:
        start_time = entry.get('BCB_start', 'N/A')  
        stop_time = entry.get('BCB_stop', 'N/A')   
        print(f"  Start Time: {start_time}, Stop Time: {stop_time}")
total_legs = sum(len(entries) for entries in master_BCB_exponential.values())
print(f"Total number of legs: {total_legs}")
#%%
master_BCB_RH_CDP = []

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
            rh9_mean = np.nanmean(rh9)

        rh_times['Rh_mean'].append(rh9_mean)
        all_BCB.append(rh_times) 

    master_BCB_RH_CDP.append(all_BCB)

for flight in master_BCB_RH_CDP:
    for leg in flight:
        rh_mean_list = leg['Rh_mean']
        leg['Rh_mean'] = [np.nan if value <=0 else value for value in rh_mean_list]
#%%
#for only the legs present after LWC filtration and master_BCB_exponential 

date_leg_set = set()
for entries in master_BCB_exponential.values():
    for entry in entries:
        date = entry['Date']
        BCB_start = entry.get('BCB_start', np.nan)
        BCB_stop = entry.get('BCB_stop', np.nan)
        date_leg_set.add((date, BCB_start, BCB_stop))

filtered_master_BCB_RH_CDP = []

for flight in master_BCB_RH_CDP:
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
# Flatten the list of lists and count the total number of entries
total_entries_filtered_master_BCB_RH_CDP = sum(len(legs) for legs in filtered_master_BCB_RH_CDP)
print(f"Total entries in filtered_master_BCB_RH: {total_entries_filtered_master_BCB_RH_CDP}")

#%%
## Convert to dry diameter
## Begin converting to dry size distribution

##Equations 1, 2, and 3 on Rob's sheets 

##ambient salt distributions n(d) measured as a function of wet diameter
## dw

##dry salt diameter ddry

##the total concentration of all the drops with ambient diameter larger than 
##dmin is Nt = integral from dmin to infinity of n(d) * dd

##where n(d) is represented by Noe(-d/D*) where we obtained No and D* from 
##code above 

##sub this equation into Nt to now get Nt = NoD*e(-dmin/D*)

##our threshold minimum dry diameter is 2um

##first we need g(RH) which is about = [1.7 / (1-RH)]^0.31
#%%

##Obtaining g(RH) = [1.7 / (1-RH)]^0.31 for all mean RH values 
## ie for every leg 

master_BCB_gRH_CDP = []
for flight in master_BCB_RH_CDP:
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
total_entries_filtered_master_BCB_gRH_CDP = sum(len(legs) for legs in filtered_master_BCB_gRH_CDP)
print(f"Total entries in filtered_master_BCB_gRH_CDP: {total_entries_filtered_master_BCB_gRH_CDP}")
#%%
##but before we do this we need to solve for average N+ intercept for every flight leg
##We know that N+ = No(gRH) where No is the intercept of our ambient distribution

# master_BCB_interceptdry_dict_CDP = {}
# if isinstance(master_BCB_gRH_CDP[0], list):
#     master_BCB_gRH = [item for sublist in master_BCB_gRH_CDP for item in sublist]

# for entry in master_BCB_gRH_CDP:
#     date = entry['Date']
#     BCB_start = entry['BCB_start']
#     BCB_stop = entry['BCB_stop']
#     gRh_mean = entry['gRh_mean'][0] 
#     Rh_mean = entry['Rh_mean'][0] 

#     if Rh_mean < 0:
#         continue

    
#     key = (date, BCB_start, BCB_stop)


#     if date in master_BCB_exponential:
#         exp_params_list = master_BCB_exponential[date]
#         for exp_params in exp_params_list:
#             n0 = exp_params['n0']
#             D = exp_params['D']
            
        
#             dryintercept = n0 * (gRh_mean)
#             master_BCB_interceptdry_dict_CDP[key] = {
#                 'Date': date,
#                 'BCB_start': BCB_start,
#                 'BCB_stop': BCB_stop,
#                 'Rh_mean': entry['Rh_mean'],
#                 'gRh_mean': entry['gRh_mean'],
#                 'dry intercept': dryintercept
#             }

# master_BCB_dryintercept_CDP = list(master_BCB_interceptdry_dict_CDP.values())
#%%
#filtered dry intercept 

filtered_master_BCB_interceptdry_dict_CDP = {}
if isinstance(filtered_master_BCB_gRH_CDP[0], list):
    filtered_master_BCB_gRH_CDP = [item for sublist in filtered_master_BCB_gRH_CDP for item in sublist]

unique_keys = set()

flattened_exponential = []
for exp_list in master_BCB_exponential.values():
    flattened_exponential.extend(exp_list)

exponential_dict = {(exp['Date'], exp['BCB_start'], exp['BCB_stop']): exp for exp in flattened_exponential}
for entry in filtered_master_BCB_gRH_CDP:
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    gRh_mean = entry['gRh_mean'][0] 
    Rh_mean = entry['Rh_mean'][0]

    if Rh_mean < 0:
        continue

    key = (date, BCB_start, BCB_stop)

    if date in master_BCB_exponential:
        exp_params_list = master_BCB_exponential[date]
        for exp_params in exp_params_list:
            n0 = exp_params['n0']
            D = exp_params['D']
            
            dryintercept = n0 * (gRh_mean)
            filtered_master_BCB_interceptdry_dict_CDP[key] = {
                'Date': date,
                'BCB_start': BCB_start,
                'BCB_stop': BCB_stop,
                'Rh_mean': entry['Rh_mean'],
                'gRh_mean': entry['gRh_mean'],
                'dry intercept': dryintercept
            }

filtered_master_BCB_dryintercept_CDP = list(filtered_master_BCB_interceptdry_dict_CDP.values())
print(f"Length of filtered_master_BCB_dryintercept_CDP: {len(filtered_master_BCB_dryintercept_CDP)}")
#%%
# master_BCB_ntd_dict_CDP = {}
# if isinstance(master_BCB_gRH_CDP[0], list):
#     master_BCB_gRH_CDP = [item for sublist in master_BCB_gRH_CDP for item in sublist]

# # Define a constant for ddrymin
# ddrymin = 2 
# for entry in master_BCB_gRH_CDP:
#     date = entry['Date']
#     BCB_start = entry['BCB_start']
#     BCB_stop = entry['BCB_stop']
#     gRh_mean = entry['gRh_mean'][0] 
#     Rh_mean = entry['Rh_mean'][0]

    
#     if Rh_mean < 0:
#         continue

    
#     key = (date, BCB_start, BCB_stop)

   
#     if date in master_BCB_exponential:
#         exp_params_list = master_BCB_exponential[date]
#         for exp_params in exp_params_list:
#             D = exp_params['D']
#             n0 = exp_params['n0']
#             # dryintercept = n0 * gRh_mean
#             # Calculate Ntd
#             Ntd = n0 * D * np.exp(-(gRh_mean * ddrymin / D))
#             master_BCB_ntd_dict_CDP[key] = {
#                 'Date': date,
#                 'BCB_start': BCB_start,
#                 'BCB_stop': BCB_stop,
#                 'Rh_mean': entry['Rh_mean'],
#                 'gRh_mean': entry['gRh_mean'],
#                 'Ntd': Ntd
#             }

# master_BCB_ntd_CDP = list(master_BCB_ntd_dict_CDP.values())
#%%
#filtered ntd 
filtered_master_BCB_ntd_dict_CDP = {}
if isinstance(filtered_master_BCB_gRH_CDP[0], list):
    filtered_master_BCB_gRH_CDP = [item for sublist in filtered_master_BCB_gRH_CDP for item in sublist]


unique_keys = set()
flattened_exponential = []
for exp_list in master_BCB_exponential.values():
    flattened_exponential.extend(exp_list)
exponential_dict = {(exp['Date'], exp['BCB_start'], exp['BCB_stop']): exp for exp in flattened_exponential}
# Define a constant for ddrymin
ddrymin = 2 
for entry in filtered_master_BCB_gRH_CDP:
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    gRh_mean = entry['gRh_mean'][0]  
    Rh_mean = entry['Rh_mean'][0]  

   
    if Rh_mean < 0:
        continue

   
    key = (date, BCB_start, BCB_stop)

    
    if date in master_BCB_exponential:
        exp_params_list = master_BCB_exponential[date]
        for exp_params in exp_params_list:
            D = exp_params['D']
            n0 = exp_params['n0']
            # dryintercept = n0 * gRh_mean
            # Calculate Ntd
            Ntd = n0 * D * np.exp(-(gRh_mean * ddrymin / D))
            
            
            filtered_master_BCB_ntd_dict_CDP[key] = {
                'Date': date,
                'BCB_start': BCB_start,
                'BCB_stop': BCB_stop,
                'Rh_mean': entry['Rh_mean'],
                'gRh_mean': entry['gRh_mean'],
                'Ntd': Ntd
            }

filtered_master_BCB_ntd_CDP = list(filtered_master_BCB_ntd_dict_CDP.values())
print(f"Length of filtered_master_BCB_ntd_CDP: {len(filtered_master_BCB_ntd_CDP)}")

#%%
# %%
## Now we can ask how ntd is related to nt (ambient)

# ## ntd / nt = e(-grh* drymin + dmin / D*) where dmin is 2.5 for the CAS data
# ##this is equation 8 on Rob's sheet

# master_BCB_NtdNt_dict_CDP = {}
# if isinstance(master_BCB_gRH_CDP[0], list):
#     master_BCB_gRH_CDP = [item for sublist in master_BCB_gRH_CDP for item in sublist]

# ddrymin = 2
# dmin = 2.5  

# for entry in master_BCB_gRH_CDP:
#     date = entry['Date']
#     BCB_start = entry['BCB_start']
#     BCB_stop = entry['BCB_stop']
#     gRh_mean = entry['gRh_mean'][0]  
#     Rh_mean = entry['Rh_mean'][0] 
    
#     if Rh_mean < 0:
#         continue

    
#     key = (date, BCB_start, BCB_stop)

    
#     if date in master_BCB_exponential:
#         exp_params_list = master_BCB_exponential[date]
#         for exp_params in exp_params_list:
#             D = exp_params['D']
            
            
#             NtdNt = np.exp((-gRh_mean * ddrymin + dmin) / D)
            
#             master_BCB_NtdNt_dict_CDP[key] = {
#                 'Date': date,
#                 'BCB_start': BCB_start,
#                 'BCB_stop': BCB_stop,
#                 'Rh_mean': entry['Rh_mean'],
#                 'gRh_mean': entry['gRh_mean'],
#                 'NtdNt': NtdNt
#             }

# master_BCB_NtdNt_CDP = list(master_BCB_NtdNt_dict_CDP.values())
#%%
#filtered master_BCB_NtdNt

filtered_master_BCB_NtdNt_dict_CDP = {}
if isinstance(filtered_master_BCB_gRH_CDP[0], list):
    filtered_master_BCB_gRH_CDP = [item for sublist in filtered_master_BCB_gRH_CDP for item in sublist]

ddrymin = 2
dmin = 2.5  

unique_keys = set()

flattened_exponential = []
for exp_list in master_BCB_exponential.values():
    flattened_exponential.extend(exp_list)

exponential_dict = {(exp['Date'], exp['BCB_start'], exp['BCB_stop']): exp for exp in flattened_exponential}

for entry in filtered_master_BCB_gRH_CDP:
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    gRh_mean = entry['gRh_mean'][0] 
    Rh_mean = entry['Rh_mean'][0] 

    
    if Rh_mean < 0:
        continue

    
    key = (date, BCB_start, BCB_stop)

    
    if key in exponential_dict:
        exp_params = exponential_dict[key]
        D = exp_params['D']
        
        
        NtdNt = np.exp((-gRh_mean * ddrymin + dmin) / D)
        
        
        filtered_master_BCB_NtdNt_dict_CDP[key] = {
            'Date': date,
            'BCB_start': BCB_start,
            'BCB_stop': BCB_stop,
            'Rh_mean': entry['Rh_mean'],
            'gRh_mean': entry['gRh_mean'],
            'NtdNt': NtdNt
        }

filtered_master_BCB_NtdNt_CDP = list(filtered_master_BCB_NtdNt_dict_CDP.values())
print(f"Length of filtered_master_BCB_NtdNt_CDP: {len(filtered_master_BCB_NtdNt_CDP)}")

# %%

#Time to gather wind speed and altitude data for each leg
master_BCB_CDP = []
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

        all_BCB_means.append(wind_alt)
        
    master_BCB_CDP.append(all_BCB_means)
#%%
Z0 = 0.02
Z10 = 10 


corrected_calc_bcb = {'Date': [], 'Corrected_bcb_windspeed': []}

for flight in master_BCB_CDP:
    for wind_alt in flight:
        date = wind_alt['Date']
        windspeed = wind_alt['Winds_mean']
        altitude = wind_alt['Alts_mean']

        for wind_mean, alt_mean in zip(windspeed, altitude):
            
            new_windspeed = wind_mean * (np.log(Z10/Z0) / np.log(alt_mean / Z0))

            
            corrected_calc_bcb['Date'].append(date)
            corrected_calc_bcb['Corrected_bcb_windspeed'].append(new_windspeed)
            
for date, wind_mean in zip(corrected_calc_bcb['Date'], corrected_calc_bcb['Corrected_bcb_windspeed']):
    print(f"Date: {date}, Corrected_bcb_windspeed: {wind_mean}")
# %%
Z0 = 0.02 
Z10 = 10 

def correct_windspeed(windspeed, altitude):
    return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))

combined_data = {
    'Date': [],
    'Altitude': [],
    'RH': [],
    'Windspeed': []
}

for i, flight in enumerate(master_BCB_CDP):
    for j, wind_alt in enumerate(flight):
        try:
            date = wind_alt['Date']
            alt_mean = wind_alt['Alts_mean'][0]
            rh_mean = filtered_master_BCB_RH_CDP[i][j]['Rh_mean'][0]
            
            
            wind_mean = wind_alt['Winds_mean'][0]
            if not np.isnan(alt_mean) and alt_mean > 0:
                corrected_windspeed = correct_windspeed(wind_mean, alt_mean)
            else:
                corrected_windspeed = np.nan

            
            if rh_mean < 0 or rh_mean > 100:
                print(f"Invalid RH value {rh_mean} at index {i}, skipping...")
                continue

            combined_data['Date'].append(date)
            combined_data['Altitude'].append(alt_mean)
            combined_data['RH'].append(rh_mean)
            combined_data['Windspeed'].append(corrected_windspeed)

        except IndexError as e:
            print(f"Index error at i={i}, j={j}: {e}")
            continue

df_combined = pd.DataFrame(combined_data)

print(df_combined.describe())

df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]

plt.figure(figsize=(10, 8))

sc = plt.scatter(df_with_windspeed['RH'], df_with_windspeed['Altitude'], 
                 c=df_with_windspeed['Windspeed'], cmap='viridis', s=50, label='Windspeed Present')

plt.scatter(df_nan_windspeed['RH'], df_nan_windspeed['Altitude'], 
            color='black', s=50, label='Windspeed NaN')

plt.colorbar(sc, label='Mean leg wind speed (m/s)')
plt.xlabel('Mean leg RH (%)', fontsize=14, fontweight='bold')
plt.ylabel('Mean leg altitude (m)', fontsize=14, fontweight='bold')
plt.title('Below cloud base January-June 2022', fontsize=14, fontweight='bold')
plt.show()
#%%
#filtered 
Z0 = 0.02 
Z10 = 10 

def correct_windspeed(windspeed, altitude):
    return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))

combined_data = {
    'Date': [],
    'Altitude': [],
    'RH': [],
    'Windspeed': []
}

for i, flight in enumerate(master_BCB_CDP):
    for j, wind_alt in enumerate(flight):
        try:
            date = wind_alt['Date']
            alt_mean = wind_alt['Alts_mean'][0]
            rh_mean = filtered_master_BCB_RH_CDP[i][j]['Rh_mean'][0]
            
            
            wind_mean = wind_alt['Winds_mean'][0]
            if not np.isnan(alt_mean) and alt_mean > 0:
                corrected_windspeed = correct_windspeed(wind_mean, alt_mean)
            else:
                corrected_windspeed = np.nan

            
            if rh_mean < 0 or rh_mean > 100:
                print(f"Invalid RH value {rh_mean} at index {i}, skipping...")
                continue

            combined_data['Date'].append(date)
            combined_data['Altitude'].append(alt_mean)
            combined_data['RH'].append(rh_mean)
            combined_data['Windspeed'].append(corrected_windspeed)

        except IndexError as e:
            print(f"Index error at i={i}, j={j}: {e}")
            continue

df_combined = pd.DataFrame(combined_data)

print(df_combined.describe())

df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]
plt.figure(figsize=(10, 8))


sc = plt.scatter(df_with_windspeed['RH'], df_with_windspeed['Altitude'], 
                 c=df_with_windspeed['Windspeed'], cmap='viridis', s=50, label='Windspeed Present')

plt.scatter(df_nan_windspeed['RH'], df_nan_windspeed['Altitude'], 
            color='black', s=50, label='Windspeed NaN')

plt.colorbar(sc, label='Mean leg wind speed (m/s)')
plt.xlabel('Mean leg RH (%)', fontsize=14, fontweight='bold')
plt.ylabel('Mean leg altitude (m)', fontsize=14, fontweight='bold')
plt.title('Below cloud base January-June 2022', fontsize=14, fontweight='bold')
plt.show()
# %%
# Z0 = 0.02 
# Z10 = 10  

# def correct_windspeed(windspeed, altitude):
#     return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))
# combined_data = {
#     'Date': [],
#     'D': [],
#     'dryintercept': [],
#     'Windspeed': []
# }

# for i, flight in enumerate(master_BCB_CDP):
#     for j, wind_alt in enumerate(flight):
#         try:
#             date = wind_alt['Date']
#             wind_mean = wind_alt['Winds_mean'][0]
#             alt_mean = wind_alt['Alts_mean'][0]
            
            
#             if not np.isnan(alt_mean) and alt_mean > 0:
#                 corrected_windspeed = correct_windspeed(wind_mean, alt_mean)
#             else:
#                 corrected_windspeed = np.nan
            
#             if date in master_BCB_exponential:
#                 exp_params_list = master_BCB_exponential[date]
#                 gRh_mean = master_BCB_gRH_CDP[i][j]['gRh_mean'][0]  
#                 for exp_params in exp_params_list:
#                     D = exp_params['D']
#                     n0 = exp_params['n0']
#                     dryintercept = n0 * gRh_mean
                    
                
#                     combined_data['Date'].append(date)
#                     combined_data['D'].append(D)
#                     combined_data['dryintercept'].append(dryintercept)
#                     combined_data['Windspeed'].append(corrected_windspeed)

#         except IndexError as e:
#             print(f"Index error at i={i}, j={j}: {e}")
#             continue

# df_combined = pd.DataFrame(combined_data)
# print(df_combined.describe())
# df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
# df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]
# correlation_matrix = df_combined[['dryintercept', 'D', 'Windspeed']].corr()

# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Matrix')
# plt.show()

# from mpl_toolkits.mplot3d import Axes3D

# sns.pairplot(df_combined[['dryintercept', 'D', 'Windspeed']], diag_kind='kde', palette='viridis')
# plt.suptitle('Pair Plot of Dry Intercept, D, and Windspeed', y=1.02)
# plt.show()

# plt.figure(figsize=(10, 8))
# sc = plt.scatter(df_with_windspeed['D'], df_with_windspeed['dryintercept'], 
#                  c=df_with_windspeed['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

# plt.scatter(df_nan_windspeed['D'], df_nan_windspeed['dryintercept'], 
#             color='black', s=100, label='Windspeed NaN')

# plt.colorbar(sc, label='Corrected Windspeed (m/s)')
# plt.xlabel('Slope', fontsize=14, fontweight='bold')
# plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
# plt.yscale('log')
# plt.xscale('log')
# plt.title('BCB January-June 2022', fontsize=14, fontweight='bold')
# plt.show()
#%%

#Filtered 

Z0 = 0.02 
Z10 = 10 
def correct_windspeed(windspeed, altitude):
    return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))
combined_data = {
    'Date': [],
    'D': [],
    'dryintercept': [],
    'Windspeed': []
}

for i, flight in enumerate(master_BCB_CDP):
    for j, wind_alt in enumerate(flight):
        try:
            date = wind_alt['Date']
            wind_mean = wind_alt['Winds_mean'][0]
            alt_mean = wind_alt['Alts_mean'][0]
            
            
            if not np.isnan(alt_mean) and alt_mean > 0:
                corrected_windspeed = correct_windspeed(wind_mean, alt_mean)
            else:
                corrected_windspeed = np.nan
            
            
            if date in master_BCB_exponential:
                exp_params_list = master_BCB_exponential[date]
                gRh_mean = filtered_master_BCB_gRH_CDP[i][j]['gRh_mean'][0]  # Assume matching index

                for exp_params in exp_params_list:
                    D = exp_params['D']
                    n0 = exp_params['n0']
                    dryintercept = n0 * gRh_mean
                    
                    
                    combined_data['Date'].append(date)
                    combined_data['D'].append(D)
                    combined_data['dryintercept'].append(dryintercept)
                    combined_data['Windspeed'].append(corrected_windspeed)

        except IndexError as e:
            print(f"Index error at i={i}, j={j}: {e}")
            continue
df_combined = pd.DataFrame(combined_data)
df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]


# Calculate the correlation matrix
correlation_matrix = df_combined[['dryintercept', 'D', 'Windspeed']].corr()

# Visualize the correlation matrix using a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation matrix')
plt.show()
sns.pairplot(df_combined[['dryintercept', 'D', 'Windspeed']], diag_kind='kde', palette='viridis')

plt.suptitle('Dry intercept, slope, and wind speed', y=1.02)
plt.yscale('log')
plt.xscale('log')
plt.show()
sc = plt.scatter(df_with_windspeed['D'], df_with_windspeed['dryintercept'], 
                 c=df_with_windspeed['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

plt.scatter(df_nan_windspeed['D'], df_nan_windspeed['dryintercept'], 
            color='grey', s=100, label='Windspeed NaN')
plt.colorbar(sc, label='10 m wind speed (m/s)')
plt.xlabel('Slope', fontsize=14, fontweight='bold')
plt.ylabel('Dry intercept', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.xlim(10**-0.5, 10**1)
plt.title('Slope vs. dry intercept', fontsize=14, fontweight='bold')
plt.xscale('log')
plt.show()
#%%
##Phase space for D and No

#%%

Z0 = 0.02  # meters (typical value for open terrain)
Z10 = 10  # target height m

# Function to apply windspeed correction
def correct_windspeed(windspeed, altitude):
    return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))

combined_data = {
    'Date': [],
    'D': [],
    'dryintercept': [],
    'Windspeed': []
}

# Validate and combine the data
for i, flight in enumerate(master_BCB_CDP):
    for j, wind_alt in enumerate(flight):
        try:
            date = wind_alt['Date']
            wind_mean = wind_alt['Winds_mean'][0]
            alt_mean = wind_alt['Alts_mean'][0]
            
            # Correct windspeed using the provided formula
            if not np.isnan(alt_mean) and alt_mean > 0:
                corrected_windspeed = correct_windspeed(wind_mean, alt_mean)
            else:
                corrected_windspeed = np.nan
            
            # Find the corresponding exponential parameters and dry intercept
            if date in master_BCB_exponential:
                exp_params_list = master_BCB_exponential[date]
                gRh_mean = filtered_master_BCB_gRH_CDP[i][j]['gRh_mean'][0]  # Assume matching index

                for exp_params in exp_params_list:
                    D = exp_params['D']
                    n0 = exp_params['n0']
                    dryintercept = n0 * gRh_mean
                    
                    # Append data to the combined dictionary
                    combined_data['Date'].append(date)
                    combined_data['D'].append(D)
                    combined_data['dryintercept'].append(dryintercept)
                    combined_data['Windspeed'].append(corrected_windspeed)

        except IndexError as e:
            print(f"Index error at i={i}, j={j}: {e}")
            continue

# Convert to DataFrame
df_combined = pd.DataFrame(combined_data)

# Check the data for any anomalies
print(df_combined.describe())

# Separate data based on NaN Windspeed
df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]

# Check for NaN or Inf values
if df_combined[['D', 'dryintercept']].isnull().any().any():
    print("Data contains NaN values.")
    # Filter out NaN values
    df_combined = df_combined.dropna(subset=['D', 'dryintercept'])

if np.any(np.isinf(df_combined[['D', 'dryintercept']].values)):
    print("Data contains Inf values.")
    # Optionally filter out inf values
    df_combined = df_combined[np.isfinite(df_combined[['D', 'dryintercept']].values).all(axis=1)]

# Filter out NaN and Inf values for plotting
filtered_combined = df_combined[df_combined['Windspeed'].notna() & df_combined['D'].notna() & df_combined['dryintercept'].notna()]
filtered_combined = filtered_combined[np.isfinite(filtered_combined[['D', 'dryintercept']].values).all(axis=1)]

# Plot data points with valid Windspeed values
sc = plt.scatter(filtered_combined['D'], filtered_combined['dryintercept'], 
                 c=filtered_combined['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

# Plot data points with NaN Windspeed values in black
plt.scatter(df_nan_windspeed['D'], df_nan_windspeed['dryintercept'], 
            color='grey', s=100, label='Windspeed NaN')

# Calculate density for contours
kde = gaussian_kde(np.vstack([filtered_combined['D'], filtered_combined['dryintercept']]))
xgrid = np.linspace(min(filtered_combined['D']), max(filtered_combined['D']), 100)
ygrid = np.linspace(min(filtered_combined['dryintercept']), max(filtered_combined['dryintercept']), 100)
X, Y = np.meshgrid(xgrid, ygrid)
Z = kde(np.vstack([X.ravel(), Y.ravel()]))
Z = Z.reshape(X.shape)

# Plot contours for total concentration
contour_levels = np.linspace(0, Z.max(), num=10)  # Adjust number of levels as needed
plt.contour(X, Y, Z, levels=contour_levels, colors='blue', alpha=0.5)

# Add color bar
plt.colorbar(sc, label='Corrected Windspeed (m/s)')

# Add labels and title
plt.xlabel('D', fontsize=14, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.title('Density Contours' , fontweight='bold')
plt.xscale('log')
plt.xlim(10**-2, 10**2)
plt.ylim(10**-3, 10**3)
plt.show()
#%%


# Full integrated mass calculation: M = N0 * D^4
def calculate_mass(N0, D):
    return N0 * D**4

# Filter out rows where D or dryintercept is NaN or <= 0
filtered_combined_clean = filtered_combined[(filtered_combined['D'] > 0) & 
                                            (filtered_combined['dryintercept'] > 0)].copy()

# Recalculate mass for each point using the full integrated mass equation
filtered_combined_clean['Mass'] = calculate_mass(filtered_combined_clean['dryintercept'], 
                                                 filtered_combined_clean['D'])

# Debugging: Check minimum and maximum mass values
print(f"Min mass: {filtered_combined_clean['Mass'].min()}, Max mass: {filtered_combined_clean['Mass'].max()}")

# Create data-based grids to better match the distribution of the points
xgrid = np.logspace(np.log10(filtered_combined_clean['D'].min()), 
                    np.log10(filtered_combined_clean['D'].max()), 100)
ygrid = np.logspace(np.log10(filtered_combined_clean['dryintercept'].min()), 
                    np.log10(filtered_combined_clean['dryintercept'].max()), 100)
D_grid, dryintercept_grid = np.meshgrid(xgrid, ygrid)

# Calculate mass for each point on the grid
mass_grid = calculate_mass(dryintercept_grid, D_grid)

# Debugging: Check if mass_grid contains meaningful values
print(f"Mass grid: Min = {np.nanmin(mass_grid)}, Max = {np.nanmax(mass_grid)}")

# Define mass contour levels (adjust to fit data range)
mass_levels = np.logspace(np.log10(filtered_combined_clean['Mass'].min()), 
                          np.log10(filtered_combined_clean['Mass'].max()), 20)
print(f"Mass levels: {mass_levels}")

# Plot the scatter plot with Windspeed color map
sc = plt.scatter(filtered_combined_clean['D'], filtered_combined_clean['dryintercept'], 
                 c=filtered_combined_clean['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

# Plot NaN Windspeed points in black (if any)
if not df_nan_windspeed.empty:
    plt.scatter(df_nan_windspeed['D'], df_nan_windspeed['dryintercept'], 
                color='grey', s=100, label='Windspeed NaN')

# Plot the mass contours using the full integrated mass formula
contour_plot = plt.contour(D_grid, dryintercept_grid, mass_grid, levels=mass_levels, colors='blue', alpha=0.75)

# Check if contours were created
if len(contour_plot.allsegs[0]) == 0:
    print("No contours were created. Check your data range or mass grid calculation.")

# Add color bar for windspeed
plt.colorbar(sc, label='Corrected Windspeed (m/s)')

# Add labels and formatting
plt.xlabel('D', fontsize=14, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.title(' Mass Contours', fontsize=14, fontweight='bold')
plt.xlim(10**-2, 10**2)
plt.ylim(10**-2, 10**2.5)

# Display the plot
plt.show()
#%%
#extending mass contours

# Define the mass calculation function
def calculate_mass(N0, D):
    integrand = lambda d: np.exp(-d / D) * d**3
    mass_integral, _ = quad(integrand, 0, np.inf)  # Integrate from 0 to infinity
    return N0 * mass_integral

# Define the full x and y axis limits
x_min, x_max = 10**-0.2, 10**1.05  # Full x-axis range
y_min, y_max = 10**-1.7, 10**1.9  # Full y-axis range

# Create extended grids to cover the full plot range
xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200)  # Denser grid
ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)
D_grid_extended, dryintercept_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)

# Recalculate mass grid over the extended grid
mass_grid_extended = np.zeros_like(D_grid_extended)
for i in range(D_grid_extended.shape[0]):
    for j in range(D_grid_extended.shape[1]):  # Fix: Added closing parenthesis
        mass_grid_extended[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j])

# Define mass contour levels with finer spacing
mass_levels = np.logspace(-2, 5, 50)  # Adjust the range and number of levels as needed

# Plot the scatter plot with Windspeed color map
plt.figure(figsize=(10, 8))
sc = plt.scatter(filtered_combined_clean['D'], filtered_combined_clean['dryintercept'], 
                 c=filtered_combined_clean['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

# Add colorbar and labels
cbar = plt.colorbar(sc)  # Create the colorbar
cbar.set_label('Corrected Windspeed (m/s)', fontsize=14, fontweight='bold')  # Set label with fontsize and fontweight
cbar.ax.tick_params(labelsize=12)  # Adjust colorbar tick size

# Add mass contours
contour_plot = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_extended, 
                           levels=mass_levels, colors='red', alpha=0.75)

# Add labels to the mass contours
plt.clabel(contour_plot, inline=True, fontsize=10, fmt='%1.1e', use_clabeltext=True, colors='red')

# Add axis labels and titles
plt.xlabel('Slope', fontsize=19, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=19, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.title('Below Cloud Base January - June 2022', fontsize=19, fontweight='bold')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Adjust tick mark sizes
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()




#%%
#adjusting integral to 2 to infinity and changing contour number and appearance 


# Function to calculate mass with lower limit of integration at 2
def calculate_mass(N0, D):
    integrand = lambda d: np.exp(-d / D) * d**3
    mass_integral, _ = quad(integrand, 2, np.inf)
    return N0 * mass_integral

# Filter out rows where D or dryintercept is NaN or <= 0
filtered_combined_clean = filtered_combined[(filtered_combined['D'] > 0) & 
                                            (filtered_combined['dryintercept'] > 0)].copy()

# Recalculate mass for each point using the integrated mass equation (apply row-wise)
filtered_combined_clean['Mass'] = filtered_combined_clean.apply(
    lambda row: calculate_mass(row['dryintercept'], row['D']), axis=1)

# Debugging: Check minimum and maximum mass values
print(f"Min mass: {filtered_combined_clean['Mass'].min()}, Max mass: {filtered_combined_clean['Mass'].max()}")

# Create data-based grids to better match the distribution of the points
xgrid = np.logspace(np.log10(filtered_combined_clean['D'].min()), 
                    np.log10(filtered_combined_clean['D'].max()), 100)
ygrid = np.logspace(np.log10(filtered_combined_clean['dryintercept'].min()), 
                    np.log10(filtered_combined_clean['dryintercept'].max()), 100)
D_grid, dryintercept_grid = np.meshgrid(xgrid, ygrid)

# Calculate mass for each point on the grid (use np.vectorize to apply calculate_mass element-wise)
vectorized_mass = np.vectorize(calculate_mass)
mass_grid = vectorized_mass(dryintercept_grid, D_grid)

# Debugging: Check if mass_grid contains meaningful values
print(f"Mass grid: Min = {np.nanmin(mass_grid)}, Max = {np.nanmax(mass_grid)}")

# Define fewer mass contour levels (about half as many as before)
mass_levels = np.logspace(np.log10(filtered_combined_clean['Mass'].min()), 
                          np.log10(filtered_combined_clean['Mass'].max()), 10)
print(f"Mass levels: {mass_levels}")

# Plot the scatter plot with Windspeed color map
sc = plt.scatter(filtered_combined_clean['D'], filtered_combined_clean['dryintercept'], 
                 c=filtered_combined_clean['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

# Plot NaN Windspeed points in black (if any)
if not df_nan_windspeed.empty:
    plt.scatter(df_nan_windspeed['D'], df_nan_windspeed['dryintercept'], 
                color='grey', s=100, label='Windspeed NaN')

# Plot the mass contours using the full integrated mass formula with reduced number of contours
contour_plot = plt.contour(D_grid, dryintercept_grid, mass_grid, levels=mass_levels, 
                           colors='red', linewidths=1, alpha=0.75)

# Check if contours were created
if len(contour_plot.allsegs[0]) == 0:
    print("No contours were created. Check your data range or mass grid calculation.")

# Add color bar for windspeed
plt.colorbar(sc, label='Corrected Windspeed (m/s)')

# Add labels and formatting
plt.xlabel('D', fontsize=14, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.title(' Mass Contours', fontsize=14, fontweight='bold')
plt.xlim(10**-2, 10**2)
plt.ylim(10**-2, 10**2.5)

# Display the plot
plt.show()
#%%
#2 to infinity with extended mass contours 
# Define the mass calculation function
def calculate_mass(N0, D):
    integrand = lambda d: np.exp(-d / D) * d**3
    mass_integral, _ = quad(integrand, 2, np.inf)  # Integrate from 0 to infinity
    return N0 * mass_integral

# Define the full x and y axis limits
x_min, x_max = 10**-0.2, 10**1.05  # Full x-axis range
y_min, y_max = 10**-1.7, 10**1.9  # Full y-axis range

# Create extended grids to cover the full plot range
xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200)  # Denser grid
ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)
D_grid_extended, dryintercept_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)

# Recalculate mass grid over the extended grid
mass_grid_extended = np.zeros_like(D_grid_extended)
for i in range(D_grid_extended.shape[0]):
    for j in range(D_grid_extended.shape[1]):  # Fix: Added closing parenthesis
        mass_grid_extended[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j])

# Define mass contour levels with finer spacing
mass_levels = np.logspace(-2, 5, 50)  # Adjust the range and number of levels as needed

# Plot the scatter plot with Windspeed color map
plt.figure(figsize=(10, 8))
sc = plt.scatter(filtered_combined_clean['D'], filtered_combined_clean['dryintercept'], 
                 c=filtered_combined_clean['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

# Add colorbar and labels
cbar = plt.colorbar(sc)  # Create the colorbar
cbar.set_label('Corrected Windspeed (m/s)', fontsize=14, fontweight='bold')  # Set label with fontsize and fontweight
cbar.ax.tick_params(labelsize=12)  # Adjust colorbar tick size

# Add mass contours
contour_plot = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_extended, 
                           levels=mass_levels, colors='red', alpha=0.75)

# Add labels to the mass contours
plt.clabel(contour_plot, inline=True, fontsize=10, fmt='%1.1e', use_clabeltext=True, colors='red')

# Add axis labels and titles
plt.xlabel('Slope', fontsize=19, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=19, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.title('Below Cloud Base January - June 2022', fontsize=19, fontweight='bold')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Adjust tick mark sizes
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()
#%%
#reduces contours and extends them
# Define the mass calculation function
def calculate_mass(N0, D):
    integrand = lambda d: np.exp(-d / D) * d**3
    mass_integral, _ = quad(integrand, 2, np.inf)  # Integrate from 0 to infinity
    return N0 * mass_integral

# Define the full x and y axis limits
x_min, x_max = 10**-0.2, 10**1.05  # Full x-axis range
y_min, y_max = 10**-1.7, 10**1.9  # Full y-axis range

# Create extended grids to cover the full plot range
xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200)  # Denser grid
ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)
D_grid_extended, dryintercept_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)

# Recalculate mass grid over the extended grid
mass_grid_extended = np.zeros_like(D_grid_extended)
for i in range(D_grid_extended.shape[0]):
    for j in range(D_grid_extended.shape[1]):  # Fix: Added closing parenthesis
        mass_grid_extended[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j])

# Define mass contour levels with finer spacing
mass_levels = np.logspace(-2, 5, 20)  # Adjust the range and number of levels as needed

# Plot the scatter plot with Windspeed color map
plt.figure(figsize=(10, 8))
sc = plt.scatter(filtered_combined_clean['D'], filtered_combined_clean['dryintercept'], 
                 c=filtered_combined_clean['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

# Add colorbar and labels
cbar = plt.colorbar(sc)  # Create the colorbar
cbar.set_label('Corrected Windspeed (m/s)', fontsize=14, fontweight='bold')  # Set label with fontsize and fontweight
cbar.ax.tick_params(labelsize=12)  # Adjust colorbar tick size

# Add mass contours
contour_plot = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_extended, 
                           levels=mass_levels, colors='red', alpha=0.75)

# Add labels to the mass contours
plt.clabel(contour_plot, inline=True, fontsize=10, fmt='%1.1e', use_clabeltext=True, colors='red')

# Add axis labels and titles
plt.xlabel('Slope', fontsize=19, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=19, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.title('Below Cloud Base January - June 2022', fontsize=19, fontweight='bold')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Adjust tick mark sizes
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()


#%%
# Function to calculate mass with lower limit of integration at 2
def calculate_mass(N0, D):
    integrand = lambda d: np.exp(-d / D) * d**3
    mass_integral, _ = quad(integrand, 2, np.inf)
    return N0 * mass_integral

# Filter out rows where D or dryintercept is NaN or <= 0
filtered_combined_clean = filtered_combined[(filtered_combined['D'] > 0) & 
                                            (filtered_combined['dryintercept'] > 0)].copy()

# Recalculate mass for each point using the integrated mass equation (apply row-wise)
filtered_combined_clean['Mass'] = filtered_combined_clean.apply(
    lambda row: calculate_mass(row['dryintercept'], row['D']), axis=1)

# Debugging: Check minimum and maximum mass values
print(f"Min mass: {filtered_combined_clean['Mass'].min()}, Max mass: {filtered_combined_clean['Mass'].max()}")

# Create data-based grids to better match the distribution of the points
xgrid = np.logspace(np.log10(filtered_combined_clean['D'].min()), 
                    np.log10(filtered_combined_clean['D'].max()), 100)
ygrid = np.logspace(np.log10(filtered_combined_clean['dryintercept'].min()), 
                    np.log10(filtered_combined_clean['dryintercept'].max()), 100)
D_grid, dryintercept_grid = np.meshgrid(xgrid, ygrid)

# Calculate mass for each point on the grid (use np.vectorize to apply calculate_mass element-wise)
vectorized_mass = np.vectorize(calculate_mass)
mass_grid = vectorized_mass(dryintercept_grid, D_grid)

# Debugging: Check if mass_grid contains meaningful values
print(f"Mass grid: Min = {np.nanmin(mass_grid)}, Max = {np.nanmax(mass_grid)}")

low_levels = np.logspace(np.log10(filtered_combined_clean['Mass'].min()), np.log10(1), 6, endpoint=False)  # 3 contours in low range (up to 10^0)
high_levels = np.logspace(np.log10(1), np.log10(filtered_combined_clean['Mass'].max()), 5)  # More contours in higher range (10^0 to max)

# Ensure contours are strictly increasing by combining low and high levels
mass_levels = np.concatenate([low_levels, high_levels])

print(f"Mass levels: {mass_levels}")

# Plot the scatter plot with Windspeed color map
sc = plt.scatter(filtered_combined_clean['D'], filtered_combined_clean['dryintercept'], 
                 c=filtered_combined_clean['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

# Plot the mass contours using the full integrated mass formula with reduced number of contours
contour_plot = plt.contour(D_grid, dryintercept_grid, mass_grid, levels=mass_levels, 
                           colors='red', linewidths=0.5, alpha=0.75)

# Check if contours were created
if len(contour_plot.allsegs[0]) == 0:
    print("No contours were created. Check your data range or mass grid calculation.")

# Add color bar for windspeed
plt.colorbar(sc, label='Corrected Windspeed (m/s)')

# Add labels and formatting
plt.xlabel('D', fontsize=14, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.title('Mass Contours', fontsize=14, fontweight='bold')
plt.xlim(10**-2, 10**2)
plt.ylim(10**-2, 10**2.5)

# Display the plot
plt.show()
#%%
##overlaying dry concentration contours onto mass contour plot


# Function to calculate mass with lower limit of integration at 2
def calculate_mass(N0, D):
    integrand = lambda d: np.exp(-d / D) * d**3
    mass_integral, _ = quad(integrand, 2, np.inf)
    return N0 * mass_integral

# Filter out rows where D or dryintercept is NaN or <= 0
filtered_combined_clean = filtered_combined[(filtered_combined['D'] > 0) & 
                                            (filtered_combined['dryintercept'] > 0)].copy()

# Recalculate mass for each point using the integrated mass equation (apply row-wise)
filtered_combined_clean['Mass'] = filtered_combined_clean.apply(
    lambda row: calculate_mass(row['dryintercept'], row['D']), axis=1)

# Debugging: Check minimum and maximum mass values
print(f"Min mass: {filtered_combined_clean['Mass'].min()}, Max mass: {filtered_combined_clean['Mass'].max()}")

# Create data-based grids to better match the distribution of the points
xgrid = np.logspace(np.log10(filtered_combined_clean['D'].min()), 
                    np.log10(filtered_combined_clean['D'].max()), 100)
ygrid = np.logspace(np.log10(filtered_combined_clean['dryintercept'].min()), 
                    np.log10(filtered_combined_clean['dryintercept'].max()), 100)
D_grid, dryintercept_grid = np.meshgrid(xgrid, ygrid)

# Calculate mass for each point on the grid (use np.vectorize to apply calculate_mass element-wise)
vectorized_mass = np.vectorize(calculate_mass)
mass_grid = vectorized_mass(dryintercept_grid, D_grid)

# Debugging: Check if mass_grid contains meaningful values
print(f"Mass grid: Min = {np.nanmin(mass_grid)}, Max = {np.nanmax(mass_grid)}")

low_levels = np.logspace(np.log10(filtered_combined_clean['Mass'].min()), np.log10(1), 6, endpoint=False)  # 3 contours in low range (up to 10^0)
high_levels = np.logspace(np.log10(1), np.log10(filtered_combined_clean['Mass'].max()), 5)  # More contours in higher range (10^0 to max)

# Ensure contours are strictly increasing by combining low and high levels
mass_levels = np.concatenate([low_levels, high_levels])

print(f"Mass levels: {mass_levels}")

# Plot the scatter plot with Windspeed color map
sc = plt.scatter(filtered_combined_clean['D'], filtered_combined_clean['dryintercept'], 
                 c=filtered_combined_clean['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

# Plot the mass contours using the full integrated mass formula with reduced number of contours
contour_plot = plt.contour(D_grid, dryintercept_grid, mass_grid, levels=mass_levels, 
                           colors='red', linewidths=0.5, alpha=0.75)

# Check if contours were created
if len(contour_plot.allsegs[0]) == 0:
    print("No contours were created. Check your data range or mass grid calculation.")

# Overlay dry concentration contours
# Use actual data range for the grid
N0_values = filtered_combined_clean['dryintercept'].values  # Use dry intercept from the cleaned dataset
D_values = filtered_combined_clean['D'].values  # Use D values from the cleaned dataset

# Create grids that align more closely with the data
N0_grid = np.linspace(min(N0_values), max(N0_values), 200)  # Match grid to actual N0 values
D_grid_conc = np.linspace(min(D_values), max(D_values), 200)  # Match grid to actual D values

# Create empty array for concentration grid
concentration_grid = np.zeros((len(N0_grid), len(D_grid_conc)))

# Calculate concentration for each combination of N0 and D on the grid
for i, N0 in enumerate(N0_grid):
    for j, D in enumerate(D_grid_conc):
        concentration_grid[i, j] = N0 * D  # Calculate C_d

# Overlay the dry concentration contours using this refined grid
concentration_contour = plt.contour(D_grid_conc, N0_grid, concentration_grid, levels=20, colors='blue', linewidths=0.75, alpha=0.6)

# Add color bar for windspeed
plt.colorbar(sc, label='Corrected Windspeed (m/s)')

# Add labels and formatting
plt.xlabel('D', fontsize=14, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.title('Mass and Dry Concentration Contours', fontsize=14, fontweight='bold')
plt.xlim(10**-2, 10**2)
plt.ylim(10**-2, 10**2.5)

# Display the plot
plt.show()
#%%

# Function to calculate mass with lower limit of integration at 2
def calculate_mass(N0, D):
    integrand = lambda d: np.exp(-d / D) * d**3
    mass_integral, _ = quad(integrand, 2, np.inf)
    return N0 * mass_integral

# Filter out rows where D or dryintercept is NaN or <= 0
filtered_combined_clean = filtered_combined[(filtered_combined['D'] > 0) & 
                                            (filtered_combined['dryintercept'] > 0)].copy()

# Recalculate mass for each point using the integrated mass equation (apply row-wise)
filtered_combined_clean['Mass'] = filtered_combined_clean.apply(
    lambda row: calculate_mass(row['dryintercept'], row['D']), axis=1)

# Debugging: Check minimum and maximum mass values
print(f"Min mass: {filtered_combined_clean['Mass'].min()}, Max mass: {filtered_combined_clean['Mass'].max()}")

# Create data-based grids to better match the distribution of the points
xgrid = np.logspace(np.log10(filtered_combined_clean['D'].min()), 
                    np.log10(filtered_combined_clean['D'].max()), 100)
ygrid = np.logspace(np.log10(filtered_combined_clean['dryintercept'].min()), 
                    np.log10(filtered_combined_clean['dryintercept'].max()), 100)
D_grid, dryintercept_grid = np.meshgrid(xgrid, ygrid)

# Calculate mass for each point on the grid (use np.vectorize to apply calculate_mass element-wise)
vectorized_mass = np.vectorize(calculate_mass)
mass_grid = vectorized_mass(dryintercept_grid, D_grid)

# Debugging: Check if mass_grid contains meaningful values
print(f"Mass grid: Min = {np.nanmin(mass_grid)}, Max = {np.nanmax(mass_grid)}")

low_levels = np.logspace(np.log10(filtered_combined_clean['Mass'].min()), np.log10(1), 6, endpoint=False)  # 3 contours in low range (up to 10^0)
high_levels = np.logspace(np.log10(1), np.log10(filtered_combined_clean['Mass'].max()), 5)  # More contours in higher range (10^0 to max)

# Ensure contours are strictly increasing by combining low and high levels
mass_levels = np.concatenate([low_levels, high_levels])

print(f"Mass levels: {mass_levels}")

plt.figure(figsize=(15, 10))
# Plot the scatter plot with Windspeed color map
sc = plt.scatter(filtered_combined_clean['D'], filtered_combined_clean['dryintercept'], 
                 c=filtered_combined_clean['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

# Plot the mass contours using the full integrated mass formula with reduced number of contours
mass_contour_plot = plt.contour(D_grid, dryintercept_grid, mass_grid, levels=mass_levels, 
                           colors='red', linewidths=0.5, alpha=0.75)

# Overlay dry concentration contours (dashed lines)
concentration_contour = plt.contour(D_grid_conc, N0_grid, concentration_grid, 
                                     levels=20, colors='blue', linewidths=0.75, 
                                     alpha=0.6, linestyles='dashed')

# Create legend handles
mass_contour_label = plt.Line2D([0], [0], color='red', lw=2, label='Mass Contours')
concentration_contour_label = plt.Line2D([0], [0], color='blue', linestyle='dashed', lw=2, label='Dry Concentration Contours')

# Add legends
plt.legend(handles=[mass_contour_label, concentration_contour_label], loc='upper left', 
           bbox_to_anchor=(1.2, 1.1), frameon=False)  # Adjusted bbox_to_anchor to move right

# Add color bar for windspeed
cbar = plt.colorbar(sc, label='Corrected Windspeed (m/s)')

# Add labels and formatting
plt.xlabel('D', fontsize=14, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.title('Mass and Dry Concentration Contours', fontsize=14, fontweight='bold')
plt.xlim(10**-2, 10**2)
plt.ylim(10**-2, 10**2.5)

# Display the plot
plt.show()

#%%
# # Extract dry intercept (N0) and D values from the dictionaries
# N0_values = [entry['dry intercept'] for entry in filtered_master_BCB_dryintercept]
# D_values = [entry['D'] for entry in filtered_master_BCB_ddry]

# # Define the mass integrand function
# def mass_integrand(d, D):
#     return np.exp(-d / D) * d**3

# # Define a function to calculate the total mass M
# def calculate_mass(N0, D):
#     # Perform the integration from 0 to infinity
#     mass, error = quad(mass_integrand, 0, np.inf, args=(D,))
#     return N0 * mass

# # Create empty lists to store mass values
# mass_values = []
# for N0, D in zip(N0_values, D_values):
#     mass = calculate_mass(N0, D)
#     mass_values.append(mass)

# # Convert mass_values to a NumPy array for easier handling
# mass_values = np.array(mass_values)

# # Set a threshold to replace non-positive values with NaN (only if they exist)
# mass_values[mass_values <= 0] = np.nan  # Replace non-positive values with NaN

# # Plotting the mass values against D and N0
# plt.figure(figsize=(10, 8))

# # Scatter plot to visualize mass values
# plt.scatter(D_values, N0_values, c=mass_values, cmap='viridis', edgecolor='k', s=100)
# plt.colorbar(label='Mass (M)')
# plt.xlabel('D (e-folding diameter)')
# plt.ylabel('Dry Intercept (N0)')
# plt.title('Scatter Plot of Mass M vs D and N0')
# plt.xscale('log')
# plt.yscale('log')
# plt.show()
#%%
N0_grid = np.linspace(min(N0_values), max(N0_values), 200)  # Create 100 points for N0
D_grid = np.linspace(min(D_values), max(D_values), 200)  # Create 100 points for D

# Create empty array for mass grid
mass_grid = np.zeros((len(N0_grid), len(D_grid)))

# Calculate mass for each combination of N0 and D on the grid
for i, N0 in enumerate(N0_grid):
    for j, D in enumerate(D_grid):
        mass_grid[i, j] = calculate_mass(N0, D)

# Create contour plot with contour lines
plt.figure(figsize=(10, 8))
contour = plt.contour(D_grid, N0_grid, mass_grid, levels=20, cmap='viridis')  # Use plt.contour() for lines
plt.clabel(contour, inline=True, fontsize=8)  # Add labels to the contour lines
plt.colorbar(contour, label='Mass (M)')  # Add a colorbar for reference
plt.xlabel('D (e-folding diameter)')
plt.ylabel('Dry Intercept (N0)')
plt.title('Contour Plot of Mass M vs D and N0')
# plt.xscale('log')
plt.xlim(0, 13)
plt.ylim(0,280)
# plt.ylim(10**0.6, 10**2.4)
# plt.yscale('log')
plt.show()
#%%
##trying thr 0th moment for dry constant concentration contours

# Extract dry intercept (N0) and D values from the dictionaries
N0_values = [entry['dry intercept'] for entry in filtered_master_BCB_dryintercept_CDP]
D_values = [entry['D'] for entry in filtered_master_BCB_ddry_CDP]

# Create grids for contour plotting
N0_grid = np.linspace(min(N0_values), max(N0_values), 200)  # Create 200 points for N0
D_grid = np.linspace(min(D_values), max(D_values), 200)  # Create 200 points for D

# Create empty array for concentration grid
concentration_grid = np.zeros((len(N0_grid), len(D_grid)))

# Calculate concentration for each combination of N0 and D on the grid
for i, N0 in enumerate(N0_grid):
    for j, D in enumerate(D_grid):
        concentration_grid[i, j] = N0 * D  # Calculate C_d

# Create contour plot with contour lines
plt.figure(figsize=(10, 8))
contour = plt.contour(D_grid, N0_grid, concentration_grid, levels=20, cmap='viridis')  # Use plt.contour() for lines
plt.clabel(contour, inline=True, fontsize=8)  # Add labels to the contour lines
plt.colorbar(contour, label='Dry Concentration')  # Add a colorbar for reference
plt.xlabel('D')
plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
plt.title(' Dry Concentration', fontsize=14, fontweight='bold')
plt.xscale('log')
plt.yscale('log')
# plt.xlim(10**0, 10**3)
# plt.ylim(10**0, 10**3)
plt.show()
#%%
#extending the dry concentration contours
import numpy as np
import matplotlib.pyplot as plt

# Define the full x and y axis limits for extended plotting
x_min, x_max = 10**0.2, 10**1.05  # Slope range
y_min, y_max = 10**0.7, 10**1.3  # Dry Intercept range

# Create extended grids for plotting
xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200)  # Denser grid
ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)
D_grid_extended, N0_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)

# Calculate dry concentration for each combination of N0 and D on the extended grid
concentration_grid_extended = N0_grid_extended * D_grid_extended

# Define dry concentration contour levels with finer spacing
concentration_levels = np.linspace(concentration_grid_extended.min(), concentration_grid_extended.max(), 20)

# Create contour plot with scatter points
plt.figure(figsize=(10, 8))
contour_plot = plt.contour(D_grid_extended, N0_grid_extended, concentration_grid_extended, 
                           levels=concentration_levels, cmap='viridis')

# Add labels to the contour lines
plt.clabel(contour_plot, inline=True, fontsize=10, fmt='%1.1e')

# Add a colorbar for reference
cbar = plt.colorbar(contour_plot, label='Dry Concentration (particles/cm³·µm)')
cbar.ax.tick_params(labelsize=16)  # Adjust colorbar tick size

# Set bold font for colorbar ticks
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')

# Add axis labels and title
plt.xlabel('Slope', fontsize=19, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=19, fontweight='bold')
plt.title('Dry Concentration Contours', fontsize=19, fontweight='bold')

# Set axis scales and limits
plt.xscale('log')
plt.yscale('log')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Adjust major and minor ticks for both axes
plt.tick_params(axis='both', which='major', labelsize=16, width=2, length=10, direction='in')
plt.tick_params(axis='both', which='minor', labelsize=16, width=1.5, length=6, direction='in')

# Set font properties for bold tick labels
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontweight('bold')

# Tight layout for cleaner visualization
plt.tight_layout()
plt.show()
#%%
#combining extended mass contours with dry concentration contours
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import quad

# Define the mass calculation function
def calculate_mass(N0, D):
    integrand = lambda d: np.exp(-d / D) * d**3
    mass_integral, _ = quad(integrand, 2, np.inf)  # Integrate from 0 to infinity
    return N0 * mass_integral

# Define the full x and y axis limits for CDP
x_min, x_max = 10**-0.2, 10**1.05  # Adjusted range for CDP slope
y_min, y_max = 10**-1.7, 10**1.8  # Adjusted range for CDP dry intercept

# Create extended grids for the full plot range
xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200)  # Denser grid
ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)
D_grid_extended, dryintercept_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)

# Recalculate mass grid over the extended grid
mass_grid_extended = np.zeros_like(D_grid_extended)
for i in range(D_grid_extended.shape[0]):
    for j in range(D_grid_extended.shape[1]):
        mass_grid_extended[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j])

# Define mass contour levels
mass_levels = np.logspace(-2, 5, 12)  # Adjust the range and number of levels for CDP

# Recalculate dry concentration grid over the extended grid
concentration_grid_extended = np.zeros_like(D_grid_extended)
for i in range(D_grid_extended.shape[0]):
    for j in range(D_grid_extended.shape[1]):
        concentration_grid_extended[i, j] = dryintercept_grid_extended[i, j] * D_grid_extended[i, j]  # C_d = N0 * D

# Define dry concentration contour levels
concentration_levels = np.logspace(-2, 3, 20)  # Adjust the range and number of levels for CDP

# Plot the scatter plot with Windspeed color map
plt.figure(figsize=(17, 10))
sc = plt.scatter(filtered_combined_clean['D'], filtered_combined_clean['dryintercept'], 
                 c=filtered_combined_clean['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

# Add colorbar and labels
cbar = plt.colorbar(sc)  # Create the colorbar
cbar.set_label('Corrected Windspeed (m/s)', fontsize=14, fontweight='bold')  # Set label with fontsize and fontweight
cbar.ax.tick_params(labelsize=12)  # Adjust colorbar tick size

# Add mass contours
mass_contour = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_extended, 
                           levels=mass_levels, colors='red', alpha=0.75)

# Add labels to the mass contours in the same direction
plt.clabel(mass_contour, inline=True, fontsize=11, fmt='%1.1e', use_clabeltext=True, colors='red')

# Add dry concentration contours
dry_concentration_contour = plt.contour(D_grid_extended, dryintercept_grid_extended, concentration_grid_extended, 
                                        levels=concentration_levels, colors='blue', alpha=0.75)

# Add labels to the dry concentration contours in the same direction
plt.clabel(dry_concentration_contour, inline=True, fontsize=11, fmt='%1.1e', colors='blue')

# Add legend for mass and dry concentration contours
legend_elements = [
    Line2D([0], [0], color='red', linewidth=3, label='Mass'),
    Line2D([0], [0], color='blue', linewidth=3, label='Dry Concentration')
]
plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.45, 1), fontsize=19)

# Add axis labels and titles
plt.xlabel('Slope', fontsize=19, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=19, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.title('Below Cloud Base January - June 2022', fontsize=19, fontweight='bold')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Adjust tick mark sizes
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()

#%%
##Attempting constant dry concentration contours

# # Step 1: Function to calculate total concentration (zeroth moment integral)
# def total_concentration(D, dryintercept):
#     # Assuming a simple relationship for demonstration; modify as needed
#     # For example, mass M = N0 * integral { exp(-d/D) d^3 dd }
#     # This is a placeholder; replace with your actual total concentration logic
#     return (dryintercept / D)  # Placeholder calculation

# # Step 2: Create a grid of D and dryintercept values
# d_values = np.linspace(df_combined['D'].min(), df_combined['D'].max(), 100)
# dryintercept_values = np.linspace(df_combined['dryintercept'].min(), df_combined['dryintercept'].max(), 100)
# D_grid, dryintercept_grid = np.meshgrid(d_values, dryintercept_values)

# # Step 3: Calculate total concentration on the grid
# total_concentration_grid = total_concentration(D_grid, dryintercept_grid)

# # Step 4: Plotting
# plt.figure(figsize=(10, 8))

# # Plot existing contours of density
# sns.kdeplot(x=df_combined['D'], y=df_combined['dryintercept'], cmap='viridis', fill=True, thresh=0)

# # Plot contours of total concentration
# contour_levels = np.linspace(total_concentration_grid.min(), total_concentration_grid.max(), 10)  # Adjust the number of levels
# contour = plt.contour(D_grid, dryintercept_grid, total_concentration_grid, levels=contour_levels, colors='red', linewidths=1, linestyles='solid')
# plt.clabel(contour, inline=True, fontsize=10)

# # Labels and title
# plt.xlabel('D (Diameter)')
# plt.ylabel('Dry Intercept')
# plt.title('Contour Plot of Total Concentration and Density Estimate')
# plt.colorbar(label='Density Estimate')

# # Show the plot
# plt.show()
#%%
##D50 mass calculation 

# Assuming you have these lists from your previous calculations
N0_values = [entry['dry intercept'] for entry in filtered_master_BCB_dryintercept_CDP]
D_values = [entry['D'] for entry in filtered_master_BCB_ddry_CDP]

# Define the mass integrand function as before
def mass_integrand(d, D):
    return np.exp(-d / D) * d**3

# Calculate total mass for different D
total_mass = []
for D in D_values:
    mass, _ = quad(mass_integrand, 0, np.inf, args=(D,))
    total_mass.append(mass)

# Calculate cumulative mass
cumulative_mass = np.cumsum(total_mass)

# Calculate total mass to find the median
total_mass_value = np.sum(total_mass)
median_mass_threshold = total_mass_value / 2

# Find diameter corresponding to 50% cumulative mass
median_diameter = None
for i, cm in enumerate(cumulative_mass):
    if cm >= median_mass_threshold:
        median_diameter = D_values[i]
        break

# Create contour plot with contour lines
plt.figure(figsize=(10, 8))
contour = plt.contour(D_grid, N0_grid, concentration_grid, levels=20, cmap='viridis')  # Use plt.contour() for lines
plt.clabel(contour, inline=True, fontsize=8)  # Add labels to the contour lines

# Add a colorbar for reference
cbar = plt.colorbar(contour)
cbar.set_label('Dry Concentration', fontsize=14, fontweight='bold')  # Set the colorbar label
cbar.ax.tick_params(labelsize=12)  # Set the font size for colorbar ticks

plt.xlabel('D', fontsize=14, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
plt.title('Dry Concentration with Median Diameter', fontsize=14, fontweight='bold')
plt.xscale('log')
plt.yscale('log')
plt.xlim(10**-1, 10**1.5)
plt.ylim(10**1, 10**2.1)

# Add a vertical line for the median diameter
if median_diameter is not None:
    plt.axvline(x=median_diameter, color='red', linestyle='--', label=f'Median Diameter: {median_diameter:.2f}')
    plt.legend()

plt.show()
#%%
##mass contours with median diameter
# # Extract dry intercept (N0) and D values from the dictionaries
# N0_values = [entry['dry intercept'] for entry in filtered_master_BCB_dryintercept]
# D_values = [entry['D'] for entry in filtered_master_BCB_ddry]

# # Create grids for contour plotting
# N0_grid = np.linspace(min(N0_values), max(N0_values), 200)  # Create 200 points for N0
# D_grid = np.linspace(min(D_values), max(D_values), 200)  # Create 200 points for D

# # Create empty array for mass grid
# mass_grid = np.zeros((len(N0_grid), len(D_grid)))

# # Define the mass integrand function as before
# def mass_integrand(d, D):
#     return np.exp(-d / D) * d**3

# # Calculate total mass for each combination of N0 and D on the grid
# for i, N0 in enumerate(N0_grid):
#     for j, D in enumerate(D_grid):
#         mass, _ = quad(mass_integrand, 0, np.inf, args=(D,))
#         mass_grid[i, j] = N0 * mass  # Calculate the total mass based on N0

# # Create cumulative mass for finding median diameter
# total_mass = np.sum(mass_grid, axis=0)  # Total mass for each D
# cumulative_mass = np.cumsum(total_mass)  # Cumulative mass
# total_mass_value = np.sum(total_mass)  # Total mass value
# median_mass_threshold = total_mass_value / 2  # Half the total mass

# # Find diameter corresponding to 50% cumulative mass
# median_diameter = None
# for i, cm in enumerate(cumulative_mass):
#     if cm >= median_mass_threshold:
#         median_diameter = D_grid[i]
#         break

# # Create contour plot for mass
# plt.figure(figsize=(10, 8))
# contour = plt.contour(D_grid, N0_grid, mass_grid, levels=20, cmap='viridis')  # Use plt.contour() for lines
# plt.clabel(contour, inline=True, fontsize=8)  # Add labels to the contour lines

# # Add a colorbar for reference
# plt.colorbar(label='Mass')  # Set the colorbar label

# # Add median diameter line
# if median_diameter is not None:
#     plt.axvline(x=median_diameter, color='red', linestyle='--', label=f'Median Diameter: {median_diameter:.2f}')
#     plt.legend()

# plt.xlabel('D)', fontsize=14, fontweight='bold')
# plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
# plt.title('Mass Contours with Median Diameter (um)', fontsize=14, fontweight='bold')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(10**0, 10**1.2)
# plt.ylim(10**1, 10**2.1)
# plt.show()

# %%
##We still have to calculate d dry
##There are x amount of bins ranging from 2.5 to 50 aka 30 bins
##ddry=d/grh
##I need to do every bin by grh for every leg 

# master_BCB_ddry_CDP = []
# for i, (date, legs_exponential) in enumerate(master_BCB_exponential.items()):
#     if i >= len(master_BCB_gRH_CDP):
#         print(f"No corresponding gRh data for date {date}, skipping...")
#         continue
    
#     legs_grh = master_BCB_gRH_CDP[i]
    
#     for j, (leg_exponential, leg_grh) in enumerate(zip(legs_exponential, legs_grh)):
#         n0 = leg_exponential['n0']
#         D = leg_exponential['D']
        
#         gRh_mean = leg_grh['gRh_mean'][0] 
        
       
#         if gRh_mean is not np.nan and gRh_mean != 0:
#             ddry_values = [center / gRh_mean for center in bin_center]
#         else:
#             ddry_values = [np.nan] * len(bin_center)
#             print(f"Skipping division for leg {j} on date {date} due to invalid gRh_mean.")
        
        
#         master_BCB_ddry_CDP.append({
#             'Date': date,
#             'Leg_index': j,
#             'ddry': ddry_values,
#             'n0': n0,
#             'D': D,
#             'gRh_mean': gRh_mean
#         })

# for entry in master_BCB_ddry_CDP:
#     print(f"Date: {entry['Date']}, Leg index: {entry['Leg_index']}, ddry: {entry['ddry']}")

#%%
filtered_master_BCB_ddry_CDP = []
for i, (date, legs_exponential) in enumerate(master_BCB_exponential.items()):
    
    if i >= len(filtered_master_BCB_gRH_CDP):
        continue
    
    filtered_legs_grh_CDP = filtered_master_BCB_gRH_CDP[i]
    
    for j, (leg_exponential, leg_grh) in enumerate(zip(legs_exponential, filtered_legs_grh_CDP)):
        n0 = leg_exponential['n0']
        D = leg_exponential['D']
        gRh_mean = leg_grh['gRh_mean'][0] 
        BCB_start = leg_grh['BCB_start']
        BCB_stop = leg_grh['BCB_stop'] 
        
        if gRh_mean is not np.nan and gRh_mean != 0:
            filtered_ddry_values_CDP = [center / gRh_mean for center in bin_center]
        else:
            filtered_ddry_values_CDP = [np.nan] * len(bin_center)
            print(f"Skipping division for leg {j} on date {date} due to invalid gRh_mean.")
        
        
        filtered_master_BCB_ddry_CDP.append({
            'Date': date,
            'BCB_start': BCB_start,
            'BCB_stop': BCB_stop,
            'ddry': filtered_ddry_values_CDP,
            'n0': n0,
            'D': D,
            'gRh_mean': gRh_mean
        })

print(f"Length of filtered_master_BCB_ddry_CDP: {len(filtered_master_BCB_ddry_CDP)}")

# %%
# master_BCB_ddry_CDP = []
# for i, (date, legs_exponential) in enumerate(master_BCB_exponential.items()):
    
#     if i >= len(master_BCB_gRH_CDP):
#         print(f"No corresponding gRh data for date {date}, skipping...")
#         continue
    
#     legs_grh = master_BCB_gRH_CDP[i]
    
#     for j, (leg_exponential, leg_grh) in enumerate(zip(legs_exponential, legs_grh)):
#         n0 = leg_exponential['n0']
#         D = leg_exponential['D']
        
#         gRh_mean = leg_grh['gRh_mean'][0] 
        
        
#         print(f"Date: {date}, Leg index: {j}, gRh_mean: {gRh_mean}")
        
        
#         if gRh_mean is not np.nan and gRh_mean != 0:
#             ddry_values_CDP = [center / gRh_mean for center in bin_center]
            
            
#             print(f"Sample bin centers: {bin_center[:5]}")
#             print(f"Sample ddry_values: {ddry_values[:5]}")
            
#         else:
#             ddry_values_CDP = [np.nan] * len(bin_center)
#             print(f"Skipping division for leg {j} on date {date} due to invalid gRh_mean.")
        
        
#         master_BCB_ddry_CDP.append({
#             'Date': date,
#             'Leg_index': j,
#             'ddry': ddry_values_CDP,
#             'n0': n0,
#             'D': D,
#             'gRh_mean': gRh_mean
#         })
# for entry in master_BCB_ddry_CDP:
#     print(f"Date: {entry['Date']}, Leg index: {entry['Leg_index']}, ddry: {entry['ddry'][:5]}")
#%%
filtered_master_BCB_ddry_CDP = []
for i, (date, legs_exponential) in enumerate(master_BCB_exponential.items()):
    if i >= len(filtered_master_BCB_gRH_CDP):
        print(f"No corresponding gRh data for date {date}, skipping...")
        continue
    
    filtered_legs_grh_CDP = filtered_master_BCB_gRH_CDP[i]
    
    for j, (leg_exponential, leg_grh) in enumerate(zip(legs_exponential, filtered_legs_grh_CDP)):
        n0 = leg_exponential['n0']
        D = leg_exponential['D']
        
        gRh_mean = leg_grh['gRh_mean'][0] 
        
        
        BCB_start = leg_grh['BCB_start']
        BCB_stop = leg_grh['BCB_stop']
        
        
        print(f"Date: {date}, gRh_mean: {gRh_mean}")
        
        
        if gRh_mean is not np.nan and gRh_mean != 0:
            filtered_ddry_values_CDP = [center / gRh_mean for center in bin_center]
            
           
            print(f"Sample bin centers: {bin_center[:5]}")
            print(f"Sample filtered_ddry_values_CDP: {filtered_ddry_values_CDP[:5]}")
            
        else:
            filtered_ddry_values_CDP = [np.nan] * len(bin_center)
            print(f"Skipping division for leg {j} on date {date} due to invalid gRh_mean.")
        
        
        filtered_master_BCB_ddry_CDP.append({
            'Date': date,
            'BCB_start': BCB_start,
            'BCB_stop': BCB_stop,
            'filtered ddry': filtered_ddry_values_CDP,
            'n0': n0,
            'D': D,
            'gRh_mean': gRh_mean
        })
print(f"Length of filtered_master_BCB_ddry_CDP: {len(filtered_master_BCB_ddry_CDP)}")
# %%
# master_BCB_ddry_CDP = []
# bin_center = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 
#             10.5, 11.5, 12.5, 13.5, 15, 17, 19, 
#             21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49]

# print(f"Full length of bin_center: {len(bin_center)}")
# print(f"Full bin_center values: {bin_center}")
# for i, (date, legs_exponential) in enumerate(master_BCB_exponential.items()):
    
#     if i >= len(master_BCB_gRH_CDP):
#         print(f"No corresponding gRh data for date {date}, skipping...")
#         continue
    
#     legs_grh = master_BCB_gRH_CDP[i]
    
#     for j, (leg_exponential, leg_grh) in enumerate(zip(legs_exponential, legs_grh)):
#         n0 = leg_exponential['n0']
#         D = leg_exponential['D']
        
#         gRh_mean = leg_grh['gRh_mean'][0] 
        
        
#         print(f"Date: {date}, Leg index: {j}, gRh_mean: {gRh_mean}")
        
        
#         print(f"Processing leg {j} for date {date}")
#         print(f"Length of bin_center: {len(bin_center)}")
#         print(f"Bin centers for date {date}, leg {j}: {bin_center}")
        
        
#         if gRh_mean is not np.nan and gRh_mean != 0:
#             ddry_values = [center / gRh_mean for center in bin_center]
            
            
#             print(f"Sample bin centers: {bin_center[:5]}")
#             print(f"Calculated ddry_values: {ddry_values[:5]}")
#             print(f"Full ddry_values: {ddry_values}")
            
#         else:
#             ddry_values = [np.nan] * len(bin_center)
#             print(f"Skipping division for leg {j} on date {date} due to invalid gRh_mean.")
        
#         master_BCB_ddry_CDP.append({
#             'Date': date,
#             'Leg_index': j,
#             'ddry': ddry_values,
#             'n0': n0,
#             'D': D,
#             'gRh_mean': gRh_mean
#         })

# for entry in master_BCB_ddry_CDP:
#     print(f"Date: {entry['Date']}, Leg index: {entry['Leg_index']}, ddry length: {len(entry['ddry'])}, first 5 ddry values: {entry['ddry'][:5]}")
#%%
filtered_master_BCB_ddry_CDP = []
bin_center = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 
            10.5, 11.5, 12.5, 13.5, 15, 17, 19, 
            21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49]
print(f"Full length of bin_center: {len(bin_center)}")
print(f"Full bin_center values: {bin_center}")

for i, (date, legs_exponential) in enumerate(master_BCB_exponential.items()):
    
    if i >= len(filtered_master_BCB_gRH_CDP):
        print(f"No corresponding gRh data for date {date}, skipping...")
        continue
    
    filtered_legs_grh_CDP = filtered_master_BCB_gRH_CDP[i]
    
    for j, (leg_exponential, filtered_leg_grh) in enumerate(zip(legs_exponential, filtered_legs_grh_CDP)):
        n0 = leg_exponential['n0']
        D = leg_exponential['D']
        
        gRh_mean = filtered_leg_grh['gRh_mean'][0] 
        
        BCB_start = filtered_leg_grh['BCB_start']
        BCB_stop = filtered_leg_grh['BCB_stop']
        
        
        print(f"Date: {date}, Leg index: {j}, gRh_mean: {gRh_mean}")
        
        
        print(f"Processing leg {j} for date {date}")
        print(f"Length of bin_center: {len(bin_center)}")
        print(f"Bin centers for date {date}, leg {j}: {bin_center}")
        
        
        if not np.isnan(gRh_mean) and gRh_mean != 0:
            filtered_ddry_values_CDP = [center / gRh_mean for center in bin_center]
            
            
            print(f"Sample bin centers: {bin_center[:5]}")
            print(f"Calculated filtered_ddry_values_CDP: {filtered_ddry_values_CDP[:5]}")
            print(f"Full filtered_ddry_values_CDP: {filtered_ddry_values_CDP}")
            
        else:
            filtered_ddry_values = [np.nan] * len(bin_center)
            print(f"Skipping division for leg {j} on date {date} due to invalid gRh_mean.")
        
       
        filtered_master_BCB_ddry_CDP.append({
            'Date': date,
            'Leg_index': j,
            'BCB_start': BCB_start,
            'BCB_stop': BCB_stop,
            'filtered_ddry': filtered_ddry_values_CDP,
            'n0': n0,
            'D': D,
            'gRh_mean': gRh_mean
        })

print(f"Length of filtered_master_BCB_ddry_CDP: {len(filtered_master_BCB_ddry_CDP)}")
# %%
# Loop through each entry in master_BCB_ddry_CDP
# for entry in master_BCB_ddry_CDP:
#     date = entry['Date']
#     leg_index = entry['Leg_index']
#     ddry_values = entry['ddry']

    
#     original_bin_centers = np.array(bin_center)
#     plt.plot(original_bin_centers, [0] * len(original_bin_centers), 'o-', color='blue', label='Original Bin Centers')

    
#     plt.plot(ddry_values, [0] * len(ddry_values), 'o-', color='purple', label='ddry Values')
#     plt.xlabel('Bin centers diameter (um)')
#     plt.ylabel('Droplet concentration (/cm^3/um)')
#     plt.title(f'Size Distribution Comparison\nDate: {date}, Leg index: {leg_index}')
#     plt.legend()
#     plt.show()
#     print(f"Date: {date}, Leg index: {leg_index}")
#     print(f"Original bin centers: {original_bin_centers}")
#     print(f"ddry values: {ddry_values}\n")
#%%
#Filtered 
# Loop through each entry in master_BCB_ddry_CDP
# for entry in filtered_master_BCB_ddry_CDP:
#     date = entry['Date']
#     leg_index = entry['Leg_index']
#     filtered_ddry_values_CDP = entry['filtered_ddry']
#     original_bin_centers = np.array(bin_center)
    
#     plt.plot(original_bin_centers, [0] * len(original_bin_centers), 'o-', color='blue', label='Original Bin Centers')

    
#     plt.plot(filtered_ddry_values_CDP, [0] * len(filtered_ddry_values_CDP), 'o-', color='purple', label='filtered_ddry values')
#     plt.xlabel('Bin centers diameter (um)')
#     plt.ylabel('Droplet concentration (/cm^3/um)')
#     plt.title(f'Size Distribution Comparison\nDate: {date}, Leg index: {leg_index}')
#     plt.legend()
#     plt.show()
#     print(f"Date: {date}, Leg index: {leg_index}")
#     print(f"Original bin centers: {original_bin_centers}")
#     print(f"filtered_ddry values: {filtered_ddry_values_CDP}\n")
#%%
##Begin by comparing Ntd with ambient concentrations

#We first need to get our ambient concentrations which is Nt=No*D*e^(-dmin/D)

master_BCB_NT_dict_CDP = {}
if isinstance(master_BCB_gRH_CDP[0], list):
    master_BCB_gRH_CDP = [item for sublist in master_BCB_gRH_CDP for item in sublist]

dmin = 2.5  
for entry in master_BCB_gRH_CDP:
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    gRh_mean = entry['gRh_mean'][0] 
    Rh_mean = entry['Rh_mean'][0]  
    if Rh_mean < 0:
        continue
    key = (date, BCB_start, BCB_stop)
    if date in master_BCB_exponential:
        exp_params_list = master_BCB_exponential[date]
        for exp_params in exp_params_list:
            D = exp_params['D']
            n0 = exp_params['n0']
            NT = n0* D * np.exp(-dmin / D)
            
            master_BCB_NT_dict_CDP[key] = {
                'Date': date,
                'BCB_start': BCB_start,
                'BCB_stop': BCB_stop,
                'Rh_mean': entry['Rh_mean'],
                'gRh_mean': entry['gRh_mean'],
                'NT': NT
            }

master_BCB_NT_CDP = list(master_BCB_NT_dict_CDP.values())
#%%
#Filtered 

filtered_master_BCB_NT_dict_CDP = {}

if isinstance(filtered_master_BCB_gRH_CDP[0], list):
    filtered_master_BCB_gRH_CDP = [item for sublist in filtered_master_BCB_gRH_CDP for item in sublist]

unique_keys = set()
flattened_exponential = []
for exp_list in master_BCB_exponential.values():
    flattened_exponential.extend(exp_list)

exponential_dict = {(exp['Date'], exp['BCB_start'], exp['BCB_stop']): exp for exp in flattened_exponential}
dmin = 2.5  

for entry in filtered_master_BCB_gRH_CDP:
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    gRh_mean = entry['gRh_mean'][0] 
    Rh_mean = entry['Rh_mean'][0]  
    if Rh_mean < 0:
        continue
    key = (date, BCB_start, BCB_stop)
    if date in master_BCB_exponential:
        exp_params_list = master_BCB_exponential[date]
        for exp_params in exp_params_list:
            D = exp_params['D']
            n0 = exp_params['n0']
            NT = n0* D * np.exp(-dmin / D)
            
            
            filtered_master_BCB_NT_dict_CDP[key] = {
                'Date': date,
                'BCB_start': BCB_start,
                'BCB_stop': BCB_stop,
                'Rh_mean': entry['Rh_mean'],
                'gRh_mean': entry['gRh_mean'],
                'filtered_NT': NT
            }

filtered_master_BCB_NT_CDP = list(filtered_master_BCB_NT_dict_CDP.values())
print(f"Length of filtered_master_BCB_NT_CDP: {len(filtered_master_BCB_NT_CDP)}")
#%%
NT_values = [entry['NT'] for entry in master_BCB_NT_CDP]  
Ntd_values = [entry['Ntd'] for entry in master_BCB_ntd_CDP]
if len(NT_values) != len(Ntd_values):
    raise ValueError(f"The lengths of NT_values ({len(NT_values)}) and Ntd_values ({len(Ntd_values)}) do not match!")
plt.figure(figsize=(10, 6))
plt.scatter(NT_values, Ntd_values, color='blue', label='NT vs. Ntd')
plt.xlabel('Ambient Concentration', fontsize=12, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.ylabel('Total droplet concentration \n dry droplets larger than \n 2um diam', fontsize=12, fontweight='bold')
plt.title('Below cloud base January-June 2022 ', fontsize=14, fontweight='bold')
plt.show()
#%%
filtered_NT_values_CDP = [entry['filtered_NT'] for entry in filtered_master_BCB_NT_CDP]  
filtered_Ntd_values_CDP = [entry['Ntd'] for entry in filtered_master_BCB_ntd_CDP]
if len(filtered_NT_values_CDP) != len(filtered_Ntd_values_CDP):
    raise ValueError(f"The lengths of filtered_NT_values ({len(filtered_NT_values_CDP)}) and Ntd_values ({len(filtered_Ntd_values_CDP)}) do not match!")
plt.figure(figsize=(10, 6))
plt.scatter(filtered_NT_values_CDP, filtered_Ntd_values_CDP, color='blue')
plt.xlabel('Ambient concentration (/cm^3/um)', fontsize=12, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.xlim(10**-2.3, 10**0.5)
plt.ylim(10**-4.5, 10**0)
plt.ylabel('Total droplet concentration \n dry droplets larger than \n 2um diam', fontsize=12, fontweight='bold')
plt.title('Below cloud base January-June 2022', fontsize=14, fontweight='bold')
plt.show()
#%%
#Comparing Ntdnt with ambient concentrations

# NT_values = [entry['NT'] for entry in master_BCB_NT]  
# NtdNt_values = [entry['NtdNt'] for entry in master_BCB_NtdNt]
# if len(NT_values) != len(NtdNt_values):
#     raise ValueError(f"The lengths of NT_values ({len(NT_values)}) and NtdNt_values ({len(NtdNt_values)}) do not match!")
# plt.figure(figsize=(10, 6))
# plt.scatter(NT_values, NtdNt_values, color='blue', label='NT vs. NtdNt')
# plt.xlabel('Ambient Concentration', fontsize=12, fontweight='bold')
# plt.yscale('log')
# plt.xscale('log')
# plt.ylabel('Ratio of total droplet concentration \n dry droplets larger than \n 2um diameter \n to ambient concentration', fontsize=12, fontweight='bold')
# plt.title('Below Cloud Base January-June 2022 ', fontsize=14, fontweight='bold')
# plt.show()


# Ntd_values = [entry['Ntd'] for entry in master_BCB_ntd]

# plt.figure(figsize=(10, 6))
# plt.scatter(range(len(Ntd_values)), Ntd_values, color='blue', label='Ntd Distribution')
# plt.xlabel('Chronological legs ', fontsize=12, fontweight='bold')
# plt.ylabel('Distribution of total droplet concentration \nof dry droplets with diameter>2um', fontsize=12, fontweight='bold')
# plt.title('Below Cloud Base January-June 2022', fontsize=14, fontweight='bold')
# plt.yscale('log')  # Optional: use log scale for y-axis if needed
# plt.show()

# NtdNt_values = [entry['NtdNt'] for entry in master_BCB_NtdNt]

# plt.figure(figsize=(10, 6))
# plt.scatter(range(len(NtdNt_values)), NtdNt_values, color='blue', label='NtdNt Distribution')
# plt.xlabel('Chronological legs', fontsize=12, fontweight='bold')
# plt.ylabel('Distribution of ratio of total droplet concentration of dry droplets \n with diameter larger than 2um \n to ambient concentration', fontsize=12, fontweight='bold')
# plt.title('Below Cloud Base January-June 2022', fontsize=14, fontweight='bold')
# plt.yscale('log')  # Optional: use log scale for y-axis if needed
# plt.show()
#%%
#Filtered 
filtered_NT_values_CDP = [entry['filtered_NT'] for entry in filtered_master_BCB_NT_CDP]  
filtered_NtdNt_values_CDP = [entry['NtdNt'] for entry in filtered_master_BCB_NtdNt_CDP]
if len(filtered_NT_values_CDP) != len(filtered_NtdNt_values_CDP):
    raise ValueError(f"The lengths of filtered_NT_values ({len(filtered_NT_values_CDP)}) and NtdNt_values ({len(filtered_NtdNt_values_CDP)}) do not match!")
plt.figure(figsize=(10, 6))
plt.scatter(filtered_NT_values_CDP, filtered_NtdNt_values_CDP, color='blue', label='NT vs. NtdNt')
plt.xlabel('Ambient concentration (/cm^3/um)', fontsize=12, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.ylim(10**-2.5, 10**0.5)
plt.xlim(10**-2.5, 10**0.5)
plt.ylabel('Ratio of total droplet concentration \n dry droplets larger than \n 2um diameter \n to ambient concentration', fontsize=12, fontweight='bold')
plt.title('Below cloud base January-June 2022', fontsize=14, fontweight='bold')
plt.show()


filtered_Ntd_values_CDP = [entry['Ntd'] for entry in filtered_master_BCB_ntd_CDP]

plt.figure(figsize=(10, 6))
plt.scatter(range(len(filtered_Ntd_values_CDP)), filtered_Ntd_values_CDP, color='blue', label='Ntd Distribution')
plt.xlabel('Chronological legs January 11 - June 18', fontsize=12, fontweight='bold')
plt.ylabel('Distribution of total droplet concentration \nof dry droplets 2 um d', fontsize=12, fontweight='bold')
plt.title('Below cloud base January-June 2022', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.ylim(10**-3.5, 10**0.5)
plt.show()
#%%
filtered_NtdNt_values_CDP = [entry['NtdNt'] for entry in filtered_master_BCB_NtdNt_CDP]

plt.figure(figsize=(10, 6))
plt.scatter(range(len(filtered_NtdNt_values_CDP)), filtered_NtdNt_values_CDP, color='blue', label='NtdNt Distribution')
plt.xlabel('Chronological legs', fontsize=12, fontweight='bold')
plt.ylabel('Distribution of ratio of total droplet concentration of dry droplets \n with diameter larger than 2um \n to ambient concentration', fontsize=10, fontweight='bold')
plt.title('Below cloud base January-June 2022', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.ylim(10**-2, 10**0) 
plt.show()
#%%
##plotting gRH to see distribution
# grh_values = [entry['gRh_mean'] for entry in master_BCB_gRH]  
# plt.figure(figsize=(10, 6))
# plt.plot(grh_values, color='blue')
# plt.ylabel('Humidity growth factor', fontsize=12, fontweight='bold')
# plt.title('Below Cloud Base January-June 2022 ', fontsize=14, fontweight='bold')
# plt.show()
#%%
#filtered
filtered_grh_values_CDP = [entry['gRh_mean'] for entry in filtered_master_BCB_gRH_CDP]  
plt.figure(figsize=(10, 6))
plt.plot(filtered_grh_values_CDP, color='blue')
plt.ylabel('Humidity growth factor', fontsize=12, fontweight='bold')
plt.title('Below cloud base January-June 2022', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.show()
# %%
#See how Nt and Ntd vary with windspeed
# Z0 = 0.02
# Z10 = 10 

# def correct_windspeed(windspeed, altitude):
#     return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))

# combined_data = {
#     'Date': [],
#     'BCB_start': [],
#     'BCB_end': [],
#     'Ntd': [],
#     'NT': [],
#     'Windspeed': []
# }

# for i, flight in enumerate(master_BCB_CDP):
#     for j, wind_alt in enumerate(flight):
#         try:
#             date = wind_alt['Date']
#             wind_mean = wind_alt['Winds_mean'][0]
#             alt_mean = wind_alt['Alts_mean'][0]
#             BCB_start = wind_alt['BCB_start']
#             BCB_end = wind_alt['BCB_end']
            
            
#             if not np.isnan(alt_mean) and alt_mean > 0:
#                 corrected_windspeed = correct_windspeed(wind_mean, alt_mean)
#             else:
#                 corrected_windspeed = np.nan

            
#             Ntd = None
#             NT = None

            
#             for exp_params in master_BCB_ntd_CDP:
#                 if exp_params['Date'] == date and exp_params['BCB_start'] == BCB_start and exp_params['BCB_end'] == BCB_end:
#                     Ntd = exp_params['Ntd']
#                     break

#             for exp_params in master_BCB_NT_CDP:
#                 if exp_params['Date'] == date and exp_params['BCB_start'] == BCB_start and exp_params['BCB_end'] == BCB_end:
#                     NT = exp_params['NT']
#                     break
            
#             if Ntd is not None and NT is not None:
#                 print(f"Appending data for Date: {date}, BCB_start: {BCB_start}, BCB_end: {BCB_end}")
#                 combined_data['Date'].append(date)
#                 combined_data['BCB_start'].append(BCB_start)
#                 combined_data['BCB_end'].append(BCB_end)
#                 combined_data['Ntd'].append(Ntd)
#                 combined_data['NT'].append(NT)
#                 combined_data['Windspeed'].append(corrected_windspeed)

#         except IndexError as e:
#             print(f"Index error at i={i}, j={j}: {e}")
#             continue

# df_combined = pd.DataFrame(combined_data)
# print(df_combined.describe())
# df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
# df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]

# plt.figure(figsize=(10, 8))
# sc = plt.scatter(df_with_windspeed['NT'], df_with_windspeed['Ntd'], 
#                  c=df_with_windspeed['Windspeed'], cmap='viridis', s=100, label='Windspeed')
# plt.scatter(df_nan_windspeed['NT'], df_nan_windspeed['Ntd'], 
#             color='black', s=100, label='Windspeed NaN')
# plt.colorbar(sc, label='Corrected Windspeed (m/s)')
# plt.xlabel('Ambient Concentration', fontsize=14, fontweight='bold')
# plt.ylabel('Total droplet concentration \n dry droplets larger than \n 2um diameter', fontsize=14, fontweight='bold')
# plt.title('Below Cloud Base January-June 2022', fontsize=14, fontweight='bold')
# plt.yscale('log')
# plt.xscale('log')
# plt.show()
#%%
#Filtered 
#See how Nt and Ntd vary with windspeed
Z0 = 0.02 
Z10 = 10 

def correct_windspeed(windspeed, altitude):
    return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))


combined_data = {
    'Date': [],
    'BCB_start': [],
    'BCB_stop': [],
    'Ntd': [],
    'filtered_NT': [],
    'Windspeed': []
}

for i, flight in enumerate(master_BCB_CDP):
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

            
            Ntd = None
            NT = None

        
            for exp_params in filtered_master_BCB_ntd_CDP:
                if exp_params['Date'] == date and exp_params['BCB_start'] == BCB_start and exp_params['BCB_stop'] == BCB_stop:
                    Ntd = exp_params['Ntd']
                    break

            for exp_params in filtered_master_BCB_NT_CDP:
                if exp_params['Date'] == date and exp_params['BCB_start'] == BCB_start and exp_params['BCB_stop'] == BCB_stop:
                    NT = exp_params['filtered_NT']
                    break
            
            
            if Ntd is not None and NT is not None:
                print(f"Appending data for Date: {date}, BCB_start: {BCB_start}, BCB_stop: {BCB_stop}")
                combined_data['Date'].append(date)
                combined_data['BCB_start'].append(BCB_start)
                combined_data['BCB_stop'].append(BCB_stop)
                combined_data['Ntd'].append(Ntd)
                combined_data['filtered_NT'].append(NT)
                combined_data['Windspeed'].append(corrected_windspeed)

        except IndexError as e:
            print(f"Index error at i={i}, j={j}: {e}")
            continue

df_combined = pd.DataFrame(combined_data)
print(df_combined.describe())
df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]

plt.figure(figsize=(10, 8))
sc = plt.scatter(df_with_windspeed['filtered_NT'], df_with_windspeed['Ntd'], 
                 c=df_with_windspeed['Windspeed'], cmap='viridis', s=100, label='Windspeed')

plt.scatter(df_nan_windspeed['filtered_NT'], df_nan_windspeed['Ntd'], 
            color='black', s=100, label='Windspeed NaN')
plt.colorbar(sc, label='Corrected Windspeed (m/s)')

plt.xlabel('Ambient concentration (/cm^3/um)', fontsize=14, fontweight='bold')
plt.ylabel('Total droplet concentration \n dry droplets larger than \n 2um diameter', fontsize=14, fontweight='bold')
plt.title('Below cloud base January-June 2022', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.ylim(10**-3, 10**0)
plt.xlim(10**-2.5, 10**0.5)
plt.xscale('log')
plt.show()

#%%
#See how NT and Ntndt vary with windspeed 
# Z0 = 0.02
# Z10 = 10 

# def correct_windspeed(windspeed, altitude):
#     return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))

# combined_data = {
#     'Date': [],
#     'BCB_start': [],
#     'BCB_end': [],
#     'NtdNt': [],
#     'NT': [],
#     'Windspeed': []
# }

# for i, flight in enumerate(master_BCB_CDP):
#     for j, wind_alt in enumerate(flight):
#         try:
#             date = wind_alt['Date']
#             wind_mean = wind_alt['Winds_mean'][0]
#             alt_mean = wind_alt['Alts_mean'][0]
#             BCB_start = wind_alt['BCB_start']
#             BCB_end = wind_alt['BCB_end']
            
            
#             if not np.isnan(alt_mean) and alt_mean > 0:
#                 corrected_windspeed = correct_windspeed(wind_mean, alt_mean)
#             else:
#                 corrected_windspeed = np.nan

            
#             NtdNt = None
#             NT = None

            
#             for exp_params in master_BCB_NtdNt_CDP:
#                 if exp_params['Date'] == date and exp_params['BCB_start'] == BCB_start and exp_params['BCB_end'] == BCB_end:
#                     NtdNt = exp_params['NtdNt']
#                     break

#             for exp_params in master_BCB_NT_CDP:
#                 if exp_params['Date'] == date and exp_params['BCB_start'] == BCB_start and exp_params['BCB_end'] == BCB_end:
#                     NT = exp_params['NT']
#                     break
            
            
#             if NtdNt is not None and NT is not None:
#                 print(f"Appending data for Date: {date}, BCB_start: {BCB_start}, BCB_end: {BCB_end}")
#                 combined_data['Date'].append(date)
#                 combined_data['BCB_start'].append(BCB_start)
#                 combined_data['BCB_end'].append(BCB_end)
#                 combined_data['NtdNt'].append(NtdNt)
#                 combined_data['NT'].append(NT)
#                 combined_data['Windspeed'].append(corrected_windspeed)

#         except IndexError as e:
#             print(f"Index error at i={i}, j={j}: {e}")
#             continue

# df_combined = pd.DataFrame(combined_data)
# print(df_combined.describe())

# df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
# df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]

# plt.figure(figsize=(10, 8))

# sc = plt.scatter(df_with_windspeed['NT'], df_with_windspeed['NtdNt'], 
#                  c=df_with_windspeed['Windspeed'], cmap='viridis', s=100, label='Windspeed')


# plt.scatter(df_nan_windspeed['NT'], df_nan_windspeed['NtdNt'], 
#             color='black', s=100, label='Windspeed NaN')
# plt.colorbar(sc, label='Corrected Windspeed (m/s)')
# plt.xlabel('Ambient concentration (/cm^3/um)', fontsize=14, fontweight='bold')
# plt.ylabel('Ratio of total droplet concentration \n dry droplets larger than \n 2um diameter to ambient concentration', fontsize=14, fontweight='bold')
# plt.title('Below Cloud Base January-June 2022', fontsize=14, fontweight='bold')
# plt.yscale('log')
# # plt.ylim(10**-1, 10**2)
# # plt.xlim(10**-2, 10**1)
# plt.xscale('log')
# plt.show()
#%%
#Filtered 
#See how NT and Ntndt vary with windspeed 
Z0 = 0.02
Z10 = 10 

def correct_windspeed(windspeed, altitude):
    return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))

combined_data = {
    'Date': [],
    'BCB_start': [],
    'BCB_stop': [],
    'NtdNt': [],
    'filtered_NT': [],
    'Windspeed': []
}

for i, flight in enumerate(master_BCB_CDP):
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

            
            NtdNt = None
            NT = None

            for exp_params in filtered_master_BCB_NtdNt_CDP:
                if exp_params['Date'] == date and exp_params['BCB_start'] == BCB_start and exp_params['BCB_stop'] == BCB_stop:
                    NtdNt = exp_params['NtdNt']
                    break

            for exp_params in filtered_master_BCB_NT_CDP:
                if exp_params['Date'] == date and exp_params['BCB_start'] == BCB_start and exp_params['BCB_stop'] == BCB_stop:
                    NT = exp_params['filtered_NT']
                    break
            
            
            if NtdNt is not None and NT is not None:
                print(f"Appending data for Date: {date}, BCB_start: {BCB_start}, BCB_stop: {BCB_stop}")
                combined_data['Date'].append(date)
                combined_data['BCB_start'].append(BCB_start)
                combined_data['BCB_stop'].append(BCB_stop)
                combined_data['NtdNt'].append(NtdNt)
                combined_data['filtered_NT'].append(NT)
                combined_data['Windspeed'].append(corrected_windspeed)

        except IndexError as e:
            print(f"Index error at i={i}, j={j}: {e}")
            continue

df_combined = pd.DataFrame(combined_data)
print(df_combined.describe())
df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]

plt.figure(figsize=(10, 8))
sc = plt.scatter(df_with_windspeed['filtered_NT'], df_with_windspeed['NtdNt'], 
                 c=df_with_windspeed['Windspeed'], cmap='viridis', s=100, label='Windspeed')

plt.scatter(df_nan_windspeed['filtered_NT'], df_nan_windspeed['NtdNt'], 
            color='black', s=100, label='Windspeed NaN')
plt.colorbar(sc, label='10m wind speed (m/s)')
plt.xlabel('Ambient concentration (/cm^3/um)', fontsize=14, fontweight='bold')
plt.ylabel('Ratio of total droplet concentration \n dry droplets larger than \n 2um diameter to ambient concentration', fontsize=14, fontweight='bold')
plt.title('Below cloud base January-June 2022', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.ylim(10**-3, 10**0.2)
plt.xlim(10**-2.5, 10**0.4)
plt.xscale('log')
plt.show()
#%%
##Ntd distribution varies with windspeed

# Z0 = 0.02
# Z10 = 10

# def correct_windspeed(windspeed, altitude):
#     return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))

# combined_data = {
#     'Date': [],
#     'BCB_start': [],
#     'BCB_end': [],
#     'Ntd': [],
#     'Windspeed': []
# }

# # Validate and combine the data
# for i, flight in enumerate(master_BCB_CDP):
#     for j, wind_alt in enumerate(flight):
#         try:
#             date = wind_alt['Date']
#             wind_mean = wind_alt['Winds_mean'][0]
#             alt_mean = wind_alt['Alts_mean'][0]
#             BCB_start = wind_alt['BCB_start']
#             BCB_end = wind_alt['BCB_end']
            
            
#             if not np.isnan(alt_mean) and alt_mean > 0:
#                 corrected_windspeed = correct_windspeed(wind_mean, alt_mean)
#             else:
#                 corrected_windspeed = np.nan

            
#             Ntd = None
#             for exp_params in master_BCB_ntd_CDP:
#                 if exp_params['Date'] == date and exp_params['BCB_start'] == BCB_start and exp_params['BCB_end'] == BCB_end:
#                     Ntd = exp_params['Ntd']
#                     break
            
        
#             if Ntd is not None:
#                 print(f"Appending data for Date: {date}, BCB_start: {BCB_start}, BCB_end: {BCB_end}")
#                 combined_data['Date'].append(date)
#                 combined_data['BCB_start'].append(BCB_start)
#                 combined_data['BCB_end'].append(BCB_end)
#                 combined_data['Ntd'].append(Ntd)
#                 combined_data['Windspeed'].append(corrected_windspeed)

#         except IndexError as e:
#             print(f"Index error at i={i}, j={j}: {e}")
#             continue


# df_combined = pd.DataFrame(combined_data)

# print(df_combined.describe())
# df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
# df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]

# plt.figure(figsize=(10, 8))
# sc = plt.scatter(df_with_windspeed['Windspeed'], df_with_windspeed['Ntd'], 
#                  c=df_with_windspeed['Windspeed'], cmap='viridis', s=100, label='Windspeed')

# plt.scatter(df_nan_windspeed['Windspeed'], df_nan_windspeed['Ntd'], 
#             color='black', s=100, label='Windspeed NaN')

# plt.colorbar(sc, label='10 m wind speed (m/s)')
# plt.xlabel('Wind speed (m/s)', fontsize=14, fontweight='bold')
# plt.ylabel('Total droplet concentration of dry droplets \n with diameter larger than 2um', fontsize=14, fontweight='bold')
# plt.title('Below cloud base January - June 2022', fontsize=14, fontweight='bold')
# plt.yscale('log') 
# plt.show()
#%%
##Ntd distribution varies with windspeed

#Filtered 

Z0 = 0.02 
Z10 = 10 
def correct_windspeed(windspeed, altitude):
    return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))
combined_data = {
    'Date': [],
    'BCB_start': [],
    'BCB_stop': [],
    'Ntd': [],
    'Windspeed': []
}

for i, flight in enumerate(master_BCB_CDP):
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

            
            Ntd = None
            for exp_params in filtered_master_BCB_ntd_CDP:
                if exp_params['Date'] == date and exp_params['BCB_start'] == BCB_start and exp_params['BCB_stop'] == BCB_stop:
                    Ntd = exp_params['Ntd']
                    break
            
            
            if Ntd is not None:
                print(f"Appending data for Date: {date}, BCB_start: {BCB_start}, BCB_stop: {BCB_stop}")
                combined_data['Date'].append(date)
                combined_data['BCB_start'].append(BCB_start)
                combined_data['BCB_stop'].append(BCB_stop)
                combined_data['Ntd'].append(Ntd)
                combined_data['Windspeed'].append(corrected_windspeed)

        except IndexError as e:
            print(f"Index error at i={i}, j={j}: {e}")
            continue


df_combined = pd.DataFrame(combined_data)
print(df_combined.describe())
df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]

plt.figure(figsize=(10, 8))
sc = plt.scatter(df_with_windspeed['Windspeed'], df_with_windspeed['Ntd'], 
                 c=df_with_windspeed['Windspeed'], cmap='viridis', s=100, label='Windspeed')

plt.scatter(df_nan_windspeed['Windspeed'], df_nan_windspeed['Ntd'], 
            color='black', s=100, label='Windspeed NaN')
plt.colorbar(sc, label='10 m wind speed (m/s)')
plt.xlabel('Wind speed (m/s)', fontsize=14, fontweight='bold')
plt.ylabel('Total droplet concentration of droplets \n with ddry larger than 2um d (Ntd)', fontsize=14, fontweight='bold')
plt.title('Below cloud base January - June 2022', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.ylim(10**-3.2, 10**0.2)
plt.show()
# %%
## Comparing ambient concentration to our concentration
##over 2um diameter to see how variable 

##So we need to compare ambient concentration to Ntd and ambient
##concentration to Ntdnt

##finally compare Ntdnt to Ntd
#%%

##Begin by comparing Ntd with ambient concentrations

##We first need to get our ambient concentrations which is Nt=No*D*e^(-dmin/D)

# master_BCB_NT_dict_CDP = {}
# if isinstance(master_BCB_gRH_CDP[0], list):
#     master_BCB_gRH_CDP = [item for sublist in master_BCB_gRH_CDP for item in sublist]

# dmin = 2.5  
# for entry in master_BCB_gRH_CDP:
#     date = entry['Date']
#     BCB_start = entry['BCB_start']
#     BCB_end = entry['BCB_end']
#     gRh_mean = entry['gRh_mean'][0] 
#     Rh_mean = entry['Rh_mean'][0]
#     if Rh_mean < 0:
#         continue
#     key = (date, BCB_start, BCB_end)
#     if date in master_BCB_exponential:
#         exp_params_list = master_BCB_exponential[date]
#         for exp_params in exp_params_list:
#             D = exp_params['D']
#             n0 = exp_params['n0']
#             NT = n0* D * np.exp(-dmin / D)
            
            
#             master_BCB_NT_dict_CDP[key] = {
#                 'Date': date,
#                 'Min_start': BCB_start,
#                 'Min_end': BCB_end,
#                 'Rh_mean': entry['Rh_mean'],
#                 'gRh_mean': entry['gRh_mean'],
#                 'NT': NT
#             }

# master_min_NT_CDP = list(master_BCB_NT_dict_CDP.values())

#%%
##Variation of NtdNt with windspeed

# Z0 = 0.02
# Z10 = 10
# def correct_windspeed(windspeed, altitude):
#     return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))

# combined_data = {
#     'Date': [],
#     'BCB_start': [],
#     'BCB_stop': [],
#     'NtdNt': [],
#     'Windspeed': []
# }
# for i, flight in enumerate(master_BCB_CDP):
#     for j, wind_alt in enumerate(flight):
#         try:
#             date = wind_alt['Date']
#             wind_mean = wind_alt['Winds_mean'][0]
#             alt_mean = wind_alt['Alts_mean'][0]
#             BCB_start = wind_alt['BCB_start']
#             BCB_end = wind_alt['BCB_end']
            
           
#             if not np.isnan(alt_mean) and alt_mean > 0:
#                 corrected_windspeed = correct_windspeed(wind_mean, alt_mean)
#             else:
#                 corrected_windspeed = np.nan

            
#             NtdNt = None
#             for exp_params in master_BCB_NtdNt_CDP:
#                 if exp_params['Date'] == date and exp_params['BCB_start'] == BCB_start and exp_params['BCB_end'] == BCB_end:
#                     NtdNt = exp_params['NtdNt']
#                     break
            
#             # Append data only if Ntd was found
#             if NtdNt is not None:
#                 print(f"Appending data for Date: {date}, BCB_start: {BCB_start}, BCB_end: {BCB_end}")
#                 combined_data['Date'].append(date)
#                 combined_data['BCB_start'].append(BCB_start)
#                 combined_data['BCB_end'].append(BCB_end)
#                 combined_data['NtdNt'].append(NtdNt)
#                 combined_data['Windspeed'].append(corrected_windspeed)

#         except IndexError as e:
#             print(f"Index error at i={i}, j={j}: {e}")
#             continue

# # Convert to DataFrame
# df_combined = pd.DataFrame(combined_data)

# # Check the data for any anomalies
# print(df_combined.describe())

# # Separate data based on NaN Windspeed
# df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
# df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]

# plt.figure(figsize=(10, 8))

# # Plot data with valid Windspeed values
# sc = plt.scatter(df_with_windspeed['Windspeed'], df_with_windspeed['NtdNt'], 
#                  c=df_with_windspeed['Windspeed'], cmap='viridis', s=100, label='Windspeed')

# # Plot data with NaN Windspeed values in black
# plt.scatter(df_nan_windspeed['Windspeed'], df_nan_windspeed['NtdNt'], 
#             color='black', s=100, label='Windspeed NaN')

# # Add color bar
# plt.colorbar(sc, label='Corrected Windspeed (m/s)')

# # Add labels and title
# plt.xlabel(' Windspeed (m/s)', fontsize=14, fontweight='bold')
# plt.ylabel('Ratio of Total Droplet Concentration of dry droplets \n with diameter larger than 2um \n to ambient concentration', fontsize=14, fontweight='bold')
# plt.title('Below Cloud Base January - June 2022', fontsize=14, fontweight='bold')
# plt.yscale('log')  # Optional: use log scale for y-axis if needed
# plt.show()
#%%
#Filtered 
##Variation of NtdNt with windspeed

Z0 = 0.02
Z10 = 10  
def correct_windspeed(windspeed, altitude):
    return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))

combined_data = {
    'Date': [],
    'BCB_start': [],
    'BCB_stop': [],
    'NtdNt': [],
    'Windspeed': []
}

for i, flight in enumerate(master_BCB_CDP):
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

            
            NtdNt = None
            for exp_params in filtered_master_BCB_NtdNt_CDP:
                if exp_params['Date'] == date and exp_params['BCB_start'] == BCB_start and exp_params['BCB_stop'] == BCB_stop:
                    NtdNt = exp_params['NtdNt']
                    break
            
            
            if NtdNt is not None:
                print(f"Appending data for Date: {date}, BCB_start: {BCB_start}, BCB_stop: {BCB_stop}")
                combined_data['Date'].append(date)
                combined_data['BCB_start'].append(BCB_start)
                combined_data['BCB_stop'].append(BCB_stop)
                combined_data['NtdNt'].append(NtdNt)
                combined_data['Windspeed'].append(corrected_windspeed)

        except IndexError as e:
            print(f"Index error at i={i}, j={j}: {e}")
            continue

df_combined = pd.DataFrame(combined_data)
print(df_combined.describe())
df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]

plt.figure(figsize=(10, 8))
sc = plt.scatter(df_with_windspeed['Windspeed'], df_with_windspeed['NtdNt'], 
                 c=df_with_windspeed['Windspeed'], cmap='viridis', s=100, label='Windspeed')
plt.scatter(df_nan_windspeed['Windspeed'], df_nan_windspeed['NtdNt'], 
            color='black', s=100, label='Windspeed NaN')
plt.colorbar(sc, label='10 m wind speed (m/s)')
plt.xlabel('Wind speed (m/s)', fontsize=14, fontweight='bold')
plt.ylabel('Ratio of total droplet concentration of dry droplets \n with diameter larger than 2um \n to ambient concentration', fontsize=14, fontweight='bold')
plt.title('Below cloud base January - June 2022', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.ylim(10**-3.2, 10**0.2)
plt.show()
# %%
# ##Looking at slope versus NtdNT
# combined_data = {
#     'Date': [],
#     'D': [],
#     'NtdNt': []
# }

# flat_ntdNt = [item for item in master_BCB_NtdNt_CDP]

# ntdNt_index = 0 
# for date, exp_params_list in master_BCB_exponential.items():
#     for exp_params in exp_params_list:
#         if ntdNt_index >= len(flat_ntdNt):
#             print(f"Exhausted flat_ntdNt at date={date}")
#             break

#         try:
            
#             ntdNt_data = flat_ntdNt[ntdNt_index]
#             ntdNt_index += 1
            
           
#             D = exp_params['D']
#             if isinstance(D, str):
#                 D = float(D)
            
            
#             print(f"Date: {date}, D value: {D}")
            
#             NtdNt = ntdNt_data['NtdNt']
            
            
#             combined_data['Date'].append(date)
#             combined_data['D'].append(D)
#             combined_data['NtdNt'].append(NtdNt)

#         except ValueError as e:
#             print(f"Value error at date={date} for D value: {exp_params['D']} - {e}")
#         except TypeError as e:
#             print(f"Type error at date={date} for D value: {exp_params['D']} - {e}")
#         except IndexError as e:
#             print(f"Index error at date={date}: {e}")
#         except Exception as e:
#             print(f"Unexpected error at date={date}: {e}")
# df_combined = pd.DataFrame(combined_data)
# plt.figure(figsize=(10, 6))
# plt.scatter(df_combined['D'], df_combined['NtdNt'])
# plt.xlabel('Slope', fontsize=14, fontweight='bold')
# plt.ylabel('Ratio of total droplet concentration of dry droplets \n with diameter larger than 2um \n to ambient concentration', fontsize=14, fontweight='bold')
# plt.title('Below Cloud Base January - June 2022', fontsize=14, fontweight='bold')
# plt.grid(True)
# plt.yscale('log')
# plt.xscale('log')
# plt.show()
#%%
#Filtered 
##Looking at slope versus NtdNT

combined_data = {
    'Date': [],
    'D': [],
    'NtdNt': []
}

flat_ntdNt = [item for item in filtered_master_BCB_NtdNt_CDP]

ntdNt_index = 0
for date, exp_params_list in master_BCB_exponential.items():
    for exp_params in exp_params_list:
        if ntdNt_index >= len(flat_ntdNt):
            print(f"Exhausted flat_ntdNt at date={date}")
            break

        try:
            
            ntdNt_data = flat_ntdNt[ntdNt_index]
            ntdNt_index += 1
            
            
            D = exp_params['D']
            if isinstance(D, str):
                D = float(D)
            
            
            print(f"Date: {date}, D value: {D}")
            
            NtdNt = ntdNt_data['NtdNt']
            
            
            combined_data['Date'].append(date)
            combined_data['D'].append(D)
            combined_data['NtdNt'].append(NtdNt)

        except ValueError as e:
            print(f"Value error at date={date} for D value: {exp_params['D']} - {e}")
        except TypeError as e:
            print(f"Type error at date={date} for D value: {exp_params['D']} - {e}")
        except IndexError as e:
            print(f"Index error at date={date}: {e}")
        except Exception as e:
            print(f"Unexpected error at date={date}: {e}")

df_combined = pd.DataFrame(combined_data)

plt.figure(figsize=(10, 6))
plt.scatter(df_combined['D'], df_combined['NtdNt'])
plt.xlabel('Slope', fontsize=14, fontweight='bold')
plt.ylabel('Ratio of total droplet concentration of dry droplets \n with diameter larger than 2um \n to ambient concentration', fontsize=14, fontweight='bold')
plt.title('Below cloud base January - June 2022', fontsize=14, fontweight='bold')
plt.grid(True)
plt.yscale('log')
plt.ylim(10**-2, 10**0.5)
plt.xlim(10**-0.2, 10**0.9)
plt.xscale('log')
plt.show()

# %%
## Now we need to try and take all of our legs with our new 18 bins of dry diameter sizes and average them 
##onto a consistent x-axis. 

##To do this we need to first create a basic random array with our bins sizes that will act as our 
##constant y-axis in which we will correct all our dry size distrbutions to. We dont care about the number of bins. We
##will plot our new dry size distributions on top of it. 

##To do this, we need D and dryintercept, which we already have from our dry size distributions

#%%
##Eventually we need to bin by windspeed. Where each windspeed bin will contain a certain number of size distributions that
##fit in that windspeed bins and these create straight lines 
#%%
from scipy.interpolate import interp1d
#%%
#25 bins: A common x-axis is created to average the dry size distributions onto
# We will use this common x-axis to interpolate the dry size distributions. Note that the 
# number of bins is arbitrary and can be adjusted as needed and the spacing between the bins is equal.  
def size_distribution(x, dryint, D):
    dryint = dryint 
    return dryint * np.exp(-x / D)
#We are randomly assigning bin lengths here, but we have 25 bins that end at 10 um d or any set diameter you want 
common_bins = np.linspace(0, 10, 25)

interpolated_values = []

# Iterate through entries in filtered_master_BCB_ddry
for i in range(len(filtered_master_BCB_ddry_CDP)):
    entry_ddry = filtered_master_BCB_ddry_CDP[i]
    entry_dryintercept = filtered_master_BCB_dryintercept_CDP[i] 
    
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']
    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry']) 
    dryint = entry_dryintercept['dry intercept']
    
    # Interpolate the distribution
    interp_func = interp1d(ddry_values, size_distribution(ddry_values, dryint, D), kind='linear', fill_value='extrapolate')
    interpolated_leg_values = interp_func(common_bins)
    
    # Store the interpolated values with metadata
    interpolated_values.append({
        'Date': date,
        'Leg_index': leg_index,
        'BCB_start': BCB_start,
        'BCB_stop': BCB_stop,
        'interpolated_values': interpolated_leg_values.tolist()
    })

for entry in interpolated_values:
    print(f"Date: {entry['Date']}, Leg_index: {entry['Leg_index']}, BCB_start: {entry['BCB_start']}, BCB_stop: {entry['BCB_stop']}, Interpolated Values: {entry['interpolated_values']}")


common_bins = np.linspace(0, 10, 25)
plt.figure(figsize=(12, 8))
for entry in interpolated_values:
    date = entry['Date']
    leg_index = entry['Leg_index']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    leg_values = entry['interpolated_values'] 
    plt.plot(common_bins, leg_values, label=f"Date: {date}, Leg: {leg_index} ({BCB_start} - {BCB_stop})")

plt.ylabel('Clear mean droplet concentration (/cm^3/um)', fontweight='bold')
plt.xlabel('Bin diameter (um)', fontweight='bold')
plt.title('Below cloud base dry size distribution January-June 2022', fontweight='bold')
plt.tight_layout()

line_count = len(plt.gca().get_lines())
print(f"Total number of lines plotted: {line_count}")
plt.show()
#%%
#Determine the problematic legs and remove them
problematic_legs = []

for entry in interpolated_values:
    leg_values = np.array(entry['interpolated_values'])
    

    if np.any(leg_values < 0):
        problematic_legs.append({
            'Date': entry['Date'],
            'Leg_index': entry['Leg_index'],
            'BCB_start': entry['BCB_start'],
            'BCB_stop': entry['BCB_stop'],
            'Min_value': np.min(leg_values)
        })

if problematic_legs:
    print("Problematic Legs (Values Below 0):")
    for leg in problematic_legs:
        print(f"Date: {leg['Date']}, Leg_index: {leg['Leg_index']}, "
              f"BCB_start: {leg['BCB_start']}, BCB_stop: {leg['BCB_stop']}, "
              f"Min_value: {leg['Min_value']}")
else:
    print("No problematic legs found (no values below 0).")
#%%
problematic_legs = [
    ('2022-02-15', 0),
    ('2022-02-15', 1),
    ('2022-02-15', 2),
    ('2022-02-15', 3),
    ('2022-02-15', 4),
    ('2022-02-15', 5),
    ('2022-02-15', 6),
    ('2022-02-15', 7),
    ('2022-02-15', 8),
    ('2022-02-15', 9),
    ('2022-02-15', 10),
    ('2022-02-15', 11),
    ('2022-02-15', 12),
    ('2022-02-15', 13),
   
]
problematic_set = set(problematic_legs)
#%%
def size_distribution(x, dryint, D):
    return dryint * np.exp(-x / D)

# Define common bins
common_bins = np.linspace(0, 10, 25)

interpolated_values = []

for i in range(len(filtered_master_BCB_ddry_CDP)):
    entry_ddry = filtered_master_BCB_ddry_CDP[i]
    entry_dryintercept = filtered_master_BCB_dryintercept_CDP[i] 
    
    
    date = entry_ddry['Date']
    leg_index = entry_ddry['Leg_index']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    
    
    if (date, leg_index) in problematic_set:
        continue
    
    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])
    dryint = entry_dryintercept['dry intercept']
    
   
    interp_func = interp1d(ddry_values, size_distribution(ddry_values, dryint, D), 
                           kind='linear', fill_value='extrapolate')
    interpolated_leg_values = interp_func(common_bins)
    
    
    interpolated_values.append({
        'Date': date,
        'Leg_index': leg_index,
        'BCB_start': BCB_start,
        'BCB_stop': BCB_stop,
        'interpolated_values': interpolated_leg_values.tolist()
    })

plt.figure(figsize=(12, 8))
for entry in interpolated_values:
    date = entry['Date']
    leg_index = entry['Leg_index']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    leg_values = entry['interpolated_values'] 
    plt.plot(common_bins, leg_values, label=f"Date: {date}, Leg: {leg_index} ({BCB_start} - {BCB_stop})")

plt.ylabel('Clear mean droplet concentration (/cm^3/um)', fontweight='bold')
plt.xlabel('Bin diameter (um)', fontweight='bold')
plt.title('Below cloud base dry size distribution January-June 2022', fontweight='bold')
plt.tight_layout()
line_count = len(plt.gca().get_lines())
print(f"Total number of lines plotted: {line_count}")
plt.show()

#%%
# Using 4 random windspeed bins 

# # Function to compute size distribution
# def size_distribution(x, dryint, D):
#     return dryint * np.exp(-x / D)

# # Define common bins
# common_bins = np.linspace(0, 10, 25)

# # Manually define 4 windspeed bins
# windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]

# # Initialize grouped_distributions and mean_windspeeds for the bins
# grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
# mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# # Debug variables
# missing_windspeed_count = 0
# interpolation_failures = 0

# # Total legs
# print(f"Total input legs: {len(filtered_master_BCB_ddry_CDP)}")

# # Iterate through the legs
# for i in range(len(filtered_master_BCB_ddry_CDP)):
#     entry_ddry = filtered_master_BCB_ddry_CDP[i]
#     entry_dryintercept = filtered_master_BCB_dryintercept_CDP[i]
#     date = entry_ddry['Date']
#     BCB_start = entry_ddry['BCB_start']
#     BCB_stop = entry_ddry['BCB_stop']
#     leg_index = entry_ddry['Leg_index']

#     # Skip problematic legs
#     if (date, leg_index) in problematic_set:
#         continue

#     D = entry_ddry['D']
#     ddry_values = np.array(entry_ddry['filtered_ddry'])
#     dryint = entry_dryintercept['dry intercept']

#     # Find windspeed
#     windspeed_entry = df_combined[
#         (df_combined['Date'] == date) & 
#         (df_combined['BCB_start'] == BCB_start) & 
#         (df_combined['BCB_stop'] == BCB_stop)
#     ]

#     if windspeed_entry.empty:
#         missing_windspeed_count += 1
#         continue

#     windspeed = windspeed_entry['Windspeed'].values[0]

#     # Interpolation
#     try:
#         interp_func = interp1d(ddry_values, size_distribution(ddry_values, dryint, D), kind='linear', fill_value='extrapolate')
#         interpolated_leg_values = interp_func(common_bins)
#     except Exception as e:
#         interpolation_failures += 1
#         continue

#     # Bin by manually defined windspeed bins
#     for idx, (low, high) in enumerate(windspeed_bins):
#         if low <= windspeed < high:  # Ensure no overlaps
#             grouped_distributions[idx].append(interpolated_leg_values)
#             mean_windspeeds[idx].append(windspeed)
#             break

# # Debug: Total problematic legs excluded
# print(f"Total problematic legs excluded: {len(problematic_set)}")

# # Debug: Total legs with missing windspeed data
# print(f"Total legs with missing windspeed data: {missing_windspeed_count}")

# # Debug: Total interpolation failures
# print(f"Total interpolation failures: {interpolation_failures}")

# # Debug: Legs in each windspeed bin
# for idx, group in grouped_distributions.items():
#     print(f"Windspeed bin {idx} ({windspeed_bins[idx]} m/s): {len(group)} legs")

# # Plot
# plt.figure(figsize=(12, 8))
# for idx, ranges in enumerate(windspeed_bins):
#     if grouped_distributions[idx]:
#         avg_distribution = np.mean(grouped_distributions[idx], axis=0)
#         avg_windspeed = np.mean(mean_windspeeds[idx])
#         num_legs = len(grouped_distributions[idx])
#         plt.plot(common_bins, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")

# # Total legs plotted
# total_legs = sum(len(group) for group in grouped_distributions.values())
# print(f"Total number of legs plotted: {total_legs}")

# plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
# plt.xlabel('Bin diameter (µm)', fontweight='bold')
# plt.title('Size distribution by wind speed', fontweight='bold')
# plt.legend(title="Average wind speed (m/s)")
# plt.tight_layout()
# plt.show()

#%%
def size_distribution(x, dryint, D):
    return dryint * np.exp(-x / D)

common_bins = np.linspace(0, 10, 25)
windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]

grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}
missing_windspeed_count = 0
interpolation_failures = 0
print(f"Total input legs: {len(filtered_master_BCB_ddry_CDP)}")

for i in range(len(filtered_master_BCB_ddry_CDP)):
    entry_ddry = filtered_master_BCB_ddry_CDP[i]
    entry_dryintercept = filtered_master_BCB_dryintercept_CDP[i]
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']

    
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])
    dryint = entry_dryintercept['dry intercept']

   
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) & 
        (df_combined['BCB_start'] == BCB_start) & 
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    if windspeed_entry.empty:
        missing_windspeed_count += 1
        continue

    windspeed = windspeed_entry['Windspeed'].values[0]

    
    try:
        interpolated_leg_values = interp1d(
            ddry_values, 
            size_distribution(ddry_values, dryint, D), 
            kind='linear', 
            fill_value='extrapolate'
        )(common_bins)
        
        
        interpolated_leg_values = np.nan_to_num(interpolated_leg_values, nan=0.0)
    except Exception as e:
        interpolation_failures += 1
        continue

    
    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:  
            grouped_distributions[idx].append(interpolated_leg_values)
            mean_windspeeds[idx].append(windspeed)
            break


print(f"Total problematic legs excluded: {len(problematic_set)}")

print(f"Total legs with missing windspeed data: {missing_windspeed_count}")

print(f"Total interpolation failures: {interpolation_failures}")

for idx, group in grouped_distributions.items():
    print(f"Windspeed bin {idx} ({windspeed_bins[idx]} m/s): {len(group)} legs")

plt.figure(figsize=(12, 8))
for idx, ranges in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        
        avg_distribution = np.nanmean(grouped_distributions[idx], axis=0)
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])
        plt.plot(common_bins, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")
    else:
        print(f" - Bin {idx} is empty")

total_legs = sum(len(group) for group in grouped_distributions.values())
print(f"Total number of legs plotted: {total_legs}")

plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Size distribution by wind speed', fontweight='bold')
plt.legend(title="Average wind speed (m/s)")
plt.yscale('log')
plt.tight_layout()
plt.show()

#%%
# Dynamic binning with random number of wind speed bins 

def size_distribution(x, dryint, D):
    return dryint * np.exp(-x / D)

common_bins = np.linspace(0, 10, 25)

min_windspeed = df_combined['Windspeed'].min()
max_windspeed = df_combined['Windspeed'].max()

bin_width = 2  
windspeed_bins = [(low, low + bin_width) for low in np.arange(0, max_windspeed, bin_width)]
windspeed_bins[-1] = (windspeed_bins[-1][0], np.inf)

print(f"Generated windspeed bins: {windspeed_bins}")

grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

missing_windspeed_count = 0
interpolation_failures = 0

print(f"Total input legs: {len(filtered_master_BCB_ddry_CDP)}")

for i in range(len(filtered_master_BCB_ddry_CDP)):
    entry_ddry = filtered_master_BCB_ddry_CDP[i]
    entry_dryintercept = filtered_master_BCB_dryintercept_CDP[i]
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']

    
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])
    dryint = entry_dryintercept['dry intercept']

    
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    if windspeed_entry.empty:
        missing_windspeed_count += 1
        continue

    windspeed = windspeed_entry['Windspeed'].values[0]

    
    try:
        interp_func = interp1d(ddry_values, size_distribution(ddry_values, dryint, D),
                               kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)
        if np.isnan(interpolated_leg_values).any():  
            raise ValueError("Interpolation produced NaNs")
    except Exception as e:
        interpolation_failures += 1
        continue

    
    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:
            grouped_distributions[idx].append(interpolated_leg_values)
            mean_windspeeds[idx].append(windspeed)
            break

print(f"Total problematic legs excluded: {len(problematic_set)}")
print(f"Total legs with missing windspeed data: {missing_windspeed_count}")
print(f"Total interpolation failures: {interpolation_failures}")

for idx, group in grouped_distributions.items():
    print(f"Windspeed bin {idx} ({windspeed_bins[idx]} m/s): {len(group)} legs")

plt.figure(figsize=(12, 8))
for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]: 
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])
        plt.plot(common_bins, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")

total_legs = sum(len(group) for group in grouped_distributions.values())
print(f"Total number of legs plotted: {total_legs}")

plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Size distribution by wind speed', fontweight='bold')
plt.legend(title="Average wind speed (m/s)")
plt.tight_layout()
plt.yscale('log')
plt.show()
#%%
# Create a histogram of all the windspeed values
windspeeds = df_combined['Windspeed'].dropna()  
plt.figure(figsize=(10, 6))
sns.histplot(windspeeds, bins=15, kde=True) 
plt.xlabel('10 m wind speed (m/s)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Distribution of wind speed values', fontsize=16)
plt.show()
#%%
#trying windspeed bins based on the percentile plot 

def log_size_distribution(x, dryint, D):
    return np.log(dryint * np.exp(-x / D))
common_bins = np.linspace(0, 10, 25)

windspeeds = df_combined['Windspeed'].dropna()
percentiles = np.percentile(windspeeds, [0, 25, 50, 75, 100])

windspeed_bins = [(percentiles[i], percentiles[i + 1]) for i in range(len(percentiles) - 1)]
print(f"Generated windspeed bins (percentile-based): {windspeed_bins}")

grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

missing_windspeed_count = 0
interpolation_failures = 0

print(f"Total input legs: {len(filtered_master_BCB_ddry_CDP)}")

for i in range(len(filtered_master_BCB_ddry_CDP)):
    entry_ddry = filtered_master_BCB_ddry_CDP[i]
    entry_dryintercept = filtered_master_BCB_dryintercept_CDP[i]
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']


    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])
    dryint = entry_dryintercept['dry intercept']

    
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    if windspeed_entry.empty or np.isnan(windspeed_entry['Windspeed'].values[0]):
        missing_windspeed_count += 1
        continue

    windspeed = windspeed_entry['Windspeed'].values[0]

   
    try:
        interp_func = interp1d(ddry_values, log_size_distribution(ddry_values, dryint, D),
                               kind='linear', fill_value='extrapolate')
        interpolated_leg_values = np.exp(interp_func(common_bins))  
        if np.isnan(interpolated_leg_values).any():
            raise ValueError("Interpolation produced NaNs")
    except Exception as e:
        interpolation_failures += 1
        continue

    
    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed <= high:
            grouped_distributions[idx].append(interpolated_leg_values)
            mean_windspeeds[idx].append(windspeed)
            break

print(f"Total problematic legs excluded: {len(problematic_set)}")
print(f"Total legs with missing windspeed data: {missing_windspeed_count}")
print(f"Total interpolation failures: {interpolation_failures}")

for idx, group in grouped_distributions.items():
    print(f"Windspeed bin {idx} ({windspeed_bins[idx]} m/s): {len(group)} legs")

plt.figure(figsize=(12, 8))
for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]: 
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])
        plt.plot(common_bins, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")

total_legs = sum(len(group) for group in grouped_distributions.values())
print(f"Total number of legs plotted: {total_legs}")

plt.yscale('log') 
plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold', fontsize=14)
plt.xlabel('Bin diameter (µm)', fontweight='bold', fontsize=14)
plt.title('Size distribution by wind speed', fontweight='bold', fontsize=14)
plt.legend(title="Wind speed (m/s)")
plt.tight_layout()
plt.show()

#%%
#Percentile binning with error bars 

def log_size_distribution(x, dryint, D):
    return np.log(dryint * np.exp(-x / D))

common_bins = np.linspace(0, 10, 25)
windspeeds = df_combined['Windspeed'].dropna()
percentiles = np.percentile(windspeeds, [0, 25, 50, 75, 100])

windspeed_bins = [(percentiles[i], percentiles[i + 1]) for i in range(len(percentiles) - 1)]
print(f"Generated windspeed bins: {windspeed_bins}")

grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}


colors = sns.color_palette("husl", len(windspeed_bins))


missing_windspeed_count = 0
interpolation_failures = 0


for i in range(len(filtered_master_BCB_ddry_CDP)):
    entry_ddry = filtered_master_BCB_ddry_CDP[i]
    entry_dryintercept = filtered_master_BCB_dryintercept_CDP[i]

    
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']

    
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])
    dryint = entry_dryintercept['dry intercept']

    
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    if windspeed_entry.empty or np.isnan(windspeed_entry['Windspeed'].values[0]):
        missing_windspeed_count += 1
        continue

    windspeed = windspeed_entry['Windspeed'].values[0]

    
    try:
        interp_func = interp1d(ddry_values, log_size_distribution(ddry_values, dryint, D),
                               kind='linear', fill_value='extrapolate')
        interpolated_leg_values = np.exp(interp_func(common_bins)) 
        if np.isnan(interpolated_leg_values).any():
            raise ValueError("NaN values found in interpolated results.")
    except Exception as e:
        interpolation_failures += 1
        continue

    
    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:
            grouped_distributions[idx].append(interpolated_leg_values)
            mean_windspeeds[idx].append(windspeed)
            break

print(f"Total problematic legs skipped: {len(problematic_set)}")
print(f"Total legs with missing windspeed data: {missing_windspeed_count}")
print(f"Total interpolation failures: {interpolation_failures}")


plt.figure(figsize=(12, 8))
for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)
        lower_percentile = np.percentile(grouped_distributions[idx], 25, axis=0)
        upper_percentile = np.percentile(grouped_distributions[idx], 75, axis=0)
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])

        
        plt.plot(common_bins, avg_distribution, color=colors[idx],
                 label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linewidth=2)

       
        plt.fill_between(common_bins, lower_percentile, upper_percentile,
                         color=colors[idx], alpha=0.4)

plt.yscale('log')
plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold', fontsize=14)
plt.xlabel('Bin diameter (µm)', fontweight='bold', fontsize=16)
plt.title('Size distribution by wind speed', fontweight='bold', fontsize=16)
plt.legend(title="Wind speed (m/s)", fontsize=16)
plt.tight_layout()
plt.show()

#%%
#trying a third order polynomial 

def polynomial_fit(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

common_bins = np.linspace(0, 10, 25)


windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]

grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

missing_windspeed_count = 0
interpolation_failures = 0

for i in range(len(filtered_master_BCB_ddry_CDP)):
    entry_ddry = filtered_master_BCB_ddry_CDP[i]
    entry_dryintercept = filtered_master_BCB_dryintercept_CDP[i]

    
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']

    
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])
    dryint = entry_dryintercept['dry intercept']

    
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    if windspeed_entry.empty or np.isnan(windspeed_entry['Windspeed'].values[0]):
        missing_windspeed_count += 1
        continue

    windspeed = windspeed_entry['Windspeed'].values[0]

    
    try:
        interp_func = interp1d(ddry_values, dryint * np.exp(-ddry_values / D), kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        if np.isnan(interpolated_leg_values).any():
            raise ValueError("NaN values found in interpolated results.")
    except Exception as e:
        interpolation_failures += 1
        continue

    
    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:
            grouped_distributions[idx].append(interpolated_leg_values)
            mean_windspeeds[idx].append(windspeed)
            break

print(f"Total problematic legs skipped: {len(problematic_set)}")
print(f"Total legs with missing windspeed data: {missing_windspeed_count}")
print(f"Total interpolation failures: {interpolation_failures}")

plt.figure(figsize=(12, 8))
for idx, ranges in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])

        
        try:
            popt, _ = curve_fit(polynomial_fit, common_bins, avg_distribution)
            poly_curve = polynomial_fit(common_bins, *popt)
        except Exception as e:
            print(f"Curve fit failed for bin {idx} ({ranges} m/s): {e}")
            continue

        
        lower_percentile = np.percentile(grouped_distributions[idx], 25, axis=0)
        upper_percentile = np.percentile(grouped_distributions[idx], 75, axis=0)

        
        plt.plot(common_bins, avg_distribution, 'o', label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", markersize=5)

        
        plt.plot(common_bins, poly_curve, label=f"Poly fit {avg_windspeed:.1f} m/s", linewidth=2)

        
        plt.fill_between(common_bins, lower_percentile, upper_percentile, alpha=0.2)

plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Size distribution by wind speed', fontweight='bold')
plt.legend(title="Wind speed (m/s)")
plt.yscale('log')
plt.tight_layout()
plt.show()


#%%
#fitting a 4th order polynomial 
def polynomial_fit_4th_order(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e

common_bins = np.linspace(0, 10, 25)


windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]

grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

missing_windspeed_count = 0
interpolation_failures = 0

for i in range(len(filtered_master_BCB_ddry_CDP)):
    entry_ddry = filtered_master_BCB_ddry_CDP[i]
    entry_dryintercept = filtered_master_BCB_dryintercept_CDP[i]

    
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']

    
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])
    dryint = entry_dryintercept['dry intercept']

    
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    if windspeed_entry.empty or np.isnan(windspeed_entry['Windspeed'].values[0]):
        missing_windspeed_count += 1
        continue

    windspeed = windspeed_entry['Windspeed'].values[0]

    
    try:
        interp_func = interp1d(ddry_values, dryint * np.exp(-ddry_values / D), kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        if np.isnan(interpolated_leg_values).any():
            raise ValueError("NaN values found in interpolated results.")
    except Exception as e:
        interpolation_failures += 1
        continue

    
    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:
            grouped_distributions[idx].append(interpolated_leg_values)
            mean_windspeeds[idx].append(windspeed)
            break

print(f"Total problematic legs skipped: {len(problematic_set)}")
print(f"Total legs with missing windspeed data: {missing_windspeed_count}")
print(f"Total interpolation failures: {interpolation_failures}")

plt.figure(figsize=(12, 8))
for idx, ranges in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])

        
        try:
            popt, _ = curve_fit(polynomial_fit, common_bins, avg_distribution)
            poly_curve = polynomial_fit(common_bins, *popt)
        except Exception as e:
            print(f"Curve fit failed for bin {idx} ({ranges} m/s): {e}")
            continue

        
        lower_percentile = np.percentile(grouped_distributions[idx], 25, axis=0)
        upper_percentile = np.percentile(grouped_distributions[idx], 75, axis=0)

        
        plt.plot(common_bins, avg_distribution, 'o', label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", markersize=5)

        
        plt.plot(common_bins, poly_curve, label=f"Poly fit {avg_windspeed:.1f} m/s", linewidth=2)

        
        plt.fill_between(common_bins, lower_percentile, upper_percentile, alpha=0.2)

plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Size distribution by wind speed', fontweight='bold')
plt.legend(title="Wind speed (m/s)")
# plt.yscale('log')
plt.tight_layout()
plt.show()
#%%
##fitting a fith order polynomial 

def polynomial_fit(x, a, b, c, d, e, f):
    return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f

common_bins = np.linspace(0, 10, 25)


windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]

grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

missing_windspeed_count = 0
interpolation_failures = 0

for i in range(len(filtered_master_BCB_ddry_CDP)):
    entry_ddry = filtered_master_BCB_ddry_CDP[i]
    entry_dryintercept = filtered_master_BCB_dryintercept_CDP[i]

    
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']

    
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])
    dryint = entry_dryintercept['dry intercept']

    
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    if windspeed_entry.empty or np.isnan(windspeed_entry['Windspeed'].values[0]):
        missing_windspeed_count += 1
        continue

    windspeed = windspeed_entry['Windspeed'].values[0]

    
    try:
        interp_func = interp1d(ddry_values, dryint * np.exp(-ddry_values / D), kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        if np.isnan(interpolated_leg_values).any():
            raise ValueError("NaN values found in interpolated results.")
    except Exception as e:
        interpolation_failures += 1
        continue

    
    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:
            grouped_distributions[idx].append(interpolated_leg_values)
            mean_windspeeds[idx].append(windspeed)
            break

print(f"Total problematic legs skipped: {len(problematic_set)}")
print(f"Total legs with missing windspeed data: {missing_windspeed_count}")
print(f"Total interpolation failures: {interpolation_failures}")

plt.figure(figsize=(12, 8))
for idx, ranges in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])

        
        try:
            popt, _ = curve_fit(polynomial_fit, common_bins, avg_distribution)
            poly_curve = polynomial_fit(common_bins, *popt)
        except Exception as e:
            print(f"Curve fit failed for bin {idx} ({ranges} m/s): {e}")
            continue

        
        lower_percentile = np.percentile(grouped_distributions[idx], 25, axis=0)
        upper_percentile = np.percentile(grouped_distributions[idx], 75, axis=0)

        
        plt.plot(common_bins, avg_distribution, 'o', label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", markersize=5)

        
        plt.plot(common_bins, poly_curve, label=f"Poly fit {avg_windspeed:.1f} m/s", linewidth=2)

        
        plt.fill_between(common_bins, lower_percentile, upper_percentile, alpha=0.2)

plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Size distribution by wind speed', fontweight='bold')
plt.legend(title="Wind speed (m/s)")
plt.yscale('log')
plt.tight_layout()
plt.show()

#%%
#How windspeed, slope, and Ntd correlate with windspeed 

Z0 = 0.02 
Z10 = 10  
def correct_windspeed(windspeed, altitude):
    return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))
combined_data = {
    'Date': [],
    'BCB_start': [],
    'BCB_stop': [],
    'Ntd': [],
    'D': [],
    'Windspeed': []
}


for i, flight in enumerate(master_BCB_CDP):
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


            Ntd = None
            D = None

            
            if date in master_BCB_exponential:
                for exp_params in master_BCB_exponential[date]:
                    
                    if exp_params['BCB_start'] == BCB_start and exp_params['BCB_stop'] == BCB_stop:
                        D = exp_params['D']
                        print(f"Found D for Date: {date}, BCB_start: {BCB_start}, BCB_stop: {BCB_stop}, D: {D}")
                        break

            
            for ntd_data in filtered_master_BCB_ntd_CDP:
                if ntd_data['Date'] == date and ntd_data['BCB_start'] == BCB_start and ntd_data['BCB_stop'] == BCB_stop:
                    Ntd = ntd_data['Ntd']
                    break

            
            if Ntd is not None and D is not None:
                combined_data['Date'].append(date)
                combined_data['BCB_start'].append(BCB_start)
                combined_data['BCB_stop'].append(BCB_stop)
                combined_data['Ntd'].append(Ntd)
                combined_data['D'].append(D)
                combined_data['Windspeed'].append(corrected_windspeed)

        except IndexError as e:
            print(f"Index error at i={i}, j={j}: {e}")
            continue


df_combined = pd.DataFrame(combined_data)
print(df_combined.describe())

df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]

plt.figure(figsize=(10, 8))
sc = plt.scatter(df_with_windspeed['D'], df_with_windspeed['Ntd'], 
                 c=df_with_windspeed['Windspeed'], cmap='viridis', s=100, label='Windspeed')
plt.scatter(df_nan_windspeed['D'], df_nan_windspeed['Ntd'], 
            color='black', s=100, label='Windspeed NaN')
plt.colorbar(sc, label='Corrected Windspeed (m/s)')
plt.xlabel('Slope (D)', fontsize=14, fontweight='bold')
plt.ylabel('Total droplet concentration \n dry droplets larger than \n 2um diameter', fontsize=14, fontweight='bold')
plt.title('Below cloud base January - June 2022', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.ylim(10**-3.5, 10**0)
plt.xlim(10**-2, 10**1)
plt.xscale('log')
plt.show()
#%%
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.hist(df_with_windspeed['Windspeed'], bins=30, color='skyblue', edgecolor='black')
plt.title('Wind speed distribution', fontweight='bold')
plt.xlabel('10m wind speed (m/s)', fontweight='bold')
plt.ylabel('Frequency', fontweight='bold')

plt.subplot(3, 1, 2)
plt.hist(df_with_windspeed['Ntd'], bins=30, color='lightgreen', edgecolor='black')
plt.title('Total concentration above 2um diameter Distribution', fontweight='bold')
plt.xlabel('Total concentration above 2um diameter', fontweight='bold')
plt.ylabel('Frequency')
plt.yscale('log')

plt.subplot(3, 1, 3)
plt.hist(df_with_windspeed['D'], bins=30, color='lightcoral', edgecolor='black')
plt.title('Slope distribution', fontweight='bold')
plt.xlabel('Slope', fontweight='bold')
plt.ylabel('Frequency', fontweight='bold')
plt.tight_layout()
plt.show()
#%%

#Let's try to determine the correct number of bins using clustering.
#%%
#Let's first begin with the Elbow Method to determine the optimal number of clusters for K-means clustering for wind speed.
from sklearn.cluster import KMeans
windspeed_data = df_with_windspeed['Windspeed'].values.reshape(-1, 1)

sse = []
cluster_range = range(1, 8)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42) 
    kmeans.fit(windspeed_data)
    sse.append(kmeans.inertia_) 

plt.figure(figsize=(8, 6))
plt.plot(cluster_range, sse, marker='o', linestyle='--')
plt.xlabel('Number of Clusters', fontsize=14, fontweight='bold')
plt.ylabel('Sum of Squared Errors (SSE)', fontsize=14, fontweight='bold')
plt.title('Elbow Method for Optimal Clusters', fontsize=16, fontweight='bold')
plt.xticks(cluster_range)
plt.grid()
plt.tight_layout()
plt.show()
#%%
#Elbow method for Slope
D_data = df_with_windspeed['D'].dropna().values.reshape(-1, 1)
sse_D = []
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(D_data)
    sse_D.append(kmeans.inertia_)
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, sse_D, marker='o', linestyle='--', label='D (Slope)')
plt.xlabel('Number of Clusters', fontsize=14, fontweight='bold')
plt.ylabel('Sum of Squared Errors (SSE)', fontsize=14, fontweight='bold')
plt.title('Elbow Method for optimal clusters for slope', fontsize=16, fontweight='bold')
plt.xticks(cluster_range)
plt.grid()
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
#%%
#Elbow method for total droplet concentration with droplet diameter greater than 2 um

Ntd_data = df_with_windspeed['Ntd'].dropna().values.reshape(-1, 1)  

sse_Ntd = []

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(Ntd_data)
    sse_Ntd.append(kmeans.inertia_)
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, sse_Ntd, marker='o', linestyle='--', label='Ntd (Concentration)')
plt.xlabel('Number of Clusters', fontsize=14, fontweight='bold')
plt.ylabel('Sum of Squared Errors (SSE)', fontsize=14, fontweight='bold')
plt.title('Elbow Method for optimal clusters \nTotal droplet concentration for droplet diameters greater than 2 um', fontsize=16, fontweight='bold')
plt.xticks(cluster_range)
plt.grid()
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

#%%
#Perform k-means clustering on wind speed alone 

windspeed_data = df_with_windspeed[['Windspeed']].dropna()

kmeans_windspeed = KMeans(n_clusters=2, random_state=42)
df_with_windspeed['Windspeed_cluster'] = kmeans_windspeed.fit_predict(windspeed_data)
plt.figure(figsize=(10, 6))
plt.scatter(range(len(windspeed_data)), windspeed_data, 
            c=df_with_windspeed['Windspeed_cluster'], cmap='viridis', s=100)
plt.colorbar(label='Wind speed Cluster')
plt.xlabel('Index', fontsize=14, fontweight='bold')
plt.ylabel('Wind speed (m/s)', fontsize=14, fontweight='bold')
plt.title('Clustering based on wind speed', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

#%%
#Perform k-means clustering on slope alone 

slope_data = df_with_windspeed[['D']].dropna()

kmeans_slope = KMeans(n_clusters=2, random_state=42)
df_with_windspeed['Slope_cluster'] = kmeans_slope.fit_predict(slope_data)
plt.figure(figsize=(10, 6))
plt.scatter(range(len(slope_data)), slope_data, 
            c=df_with_windspeed['Slope_cluster'], cmap='viridis', s=100)
plt.colorbar(label='Slope cluster')
plt.xlabel('Index', fontsize=14, fontweight='bold')
plt.ylabel('Slope', fontsize=14, fontweight='bold')
plt.title('Clustering based on slope', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
#Perform k-means clustering on Ntd alone 

conc_data = df_with_windspeed[['Ntd']].dropna()
kmeans_ntd = KMeans(n_clusters=2, random_state=42)
df_with_windspeed.loc[conc_data.index, 'Ntd_cluster'] = kmeans_ntd.fit_predict(conc_data)
plt.figure(figsize=(10, 6))
plt.scatter(conc_data.index, conc_data['Ntd'], 
            c=kmeans_ntd.labels_, cmap='viridis', s=100)
plt.colorbar(label='Total droplet concentration cluster')
plt.xlabel('Index', fontsize=14, fontweight='bold')
plt.ylabel('Total droplet concentration (Ntd)', fontsize=14, fontweight='bold')
plt.title('Clustering based on total droplet concentration', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
# Cluster with Ntd and Windspeed

ntd_windspeed_data = df_with_windspeed[['Ntd', 'Windspeed']].dropna()

ntd_windspeed_data = ntd_windspeed_data.reset_index()

kmeans_ntd_windspeed = KMeans(n_clusters=2, random_state=42)
ntd_windspeed_data['Ntd_Windspeed_cluster'] = kmeans_ntd_windspeed.fit_predict(ntd_windspeed_data[['Ntd', 'Windspeed']])
df_with_windspeed = df_with_windspeed.merge(ntd_windspeed_data[['index', 'Ntd_Windspeed_cluster']], 
                                            left_index=True, right_on='index', how='left')
plt.figure(figsize=(10, 6))
plt.scatter(ntd_windspeed_data['Windspeed'], ntd_windspeed_data['Ntd'], 
            c=ntd_windspeed_data['Ntd_Windspeed_cluster'], cmap='viridis', s=100)
plt.colorbar(label='Ntd & Windspeed Cluster')
plt.xlabel('Windspeed (m/s)', fontsize=14, fontweight='bold')
plt.ylabel('Total Droplet Concentration greater than 2 um d', fontsize=14, fontweight='bold')
plt.title('Clustering based on concentration and Windspeed', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
#Cluster with Ntd and slope

ntd_D_data = df_with_windspeed[['Ntd', 'D']].dropna()

ntd_D_data = ntd_D_data.reset_index()

kmeans_ntd_D = KMeans(n_clusters=2, random_state=42)
ntd_D_data['Ntd_D_cluster'] = kmeans_ntd_D.fit_predict(ntd_D_data[['Ntd', 'D']])
df_with_D = df_with_windspeed.merge(ntd_D_data[['index', 'Ntd_D_cluster']], 
                                            left_index=True, right_on='index', how='left')
plt.figure(figsize=(10, 6))
plt.scatter(ntd_D_data['D'], ntd_D_data['Ntd'], 
            c=ntd_D_data['Ntd_D_cluster'], cmap='viridis', s=100)
plt.colorbar(label='Ntd & D Cluster')
plt.xlabel('Slope', fontsize=14, fontweight='bold')
plt.ylabel('Total Droplet Concentration greater than 2 um d', fontsize=14, fontweight='bold')
plt.title('Clustering based on concentration and slope', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
#Cluster with Windspeed and slope
wind_D_data = df_with_windspeed[['Windspeed', 'D']].dropna()

wind_D_data = wind_D_data.reset_index()

kmeans_wind_D = KMeans(n_clusters=2, random_state=42)
wind_D_data['Wind_D_cluster'] = kmeans_wind_D.fit_predict(wind_D_data[['Windspeed', 'D']])
df_with_D = df_with_windspeed.merge(wind_D_data[['index', 'Wind_D_cluster']], 
                                            left_index=True, right_on='index', how='left')
plt.figure(figsize=(10, 6))
plt.scatter(wind_D_data['D'], wind_D_data['Windspeed'], 
            c=wind_D_data['Wind_D_cluster'], cmap='viridis', s=100)
plt.colorbar(label='Wind speed & slope cluster')
plt.xlabel('Slope', fontsize=14, fontweight='bold')
plt.ylabel('Wind speed (m/s)', fontsize=14, fontweight='bold')
plt.title('Clustering based on wind speed and slope', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
# Cluster with all three variables

from mpl_toolkits.mplot3d import Axes3D

all_three_data = df_with_windspeed[['Ntd', 'D', 'Windspeed']].dropna().reset_index()

kmeans_all_three = KMeans(n_clusters=2, random_state=42)
all_three_data['All_Three_cluster'] = kmeans_all_three.fit_predict(all_three_data[['Ntd', 'D', 'Windspeed']])

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(all_three_data['Windspeed'], all_three_data['D'], all_three_data['Ntd'], 
                c=all_three_data['All_Three_cluster'], cmap='viridis', s=100)
plt.colorbar(sc, label='All Three Variables Cluster')
ax.set_xlabel('Wind speed (m/s)', fontsize=14, fontweight='bold')
ax.set_ylabel('Slope', fontsize=14, fontweight='bold')
ax.set_zlabel('Total Droplet Concentration greater than 2 um d', fontsize=14, fontweight='bold')
ax.set_title('Clustering based on All Three Variables', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
#%%

#Use the k-means clustering to recreate our dry size distribution

def size_distribution(x, dryint, D):
    return dryint * np.exp(-x / D)

common_bins = np.linspace(0, 10, 25)

unique_clusters = df_with_windspeed['Windspeed_cluster'].dropna().unique()
grouped_distributions_by_cluster = {cluster: [] for cluster in unique_clusters}

print(f"Identified clusters: {unique_clusters}")

for i in range(len(filtered_master_BCB_ddry_CDP)):
    entry_ddry = filtered_master_BCB_ddry_CDP[i]
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']

   
    if (date, leg_index) in problematic_set:
        print(f"Skipping problematic leg: Date {date}, Leg {leg_index}")
        continue

    
    windspeed_entry = df_with_windspeed[
        (df_with_windspeed['Date'] == date) &
        (df_with_windspeed['BCB_start'] == BCB_start) &
        (df_with_windspeed['BCB_stop'] == BCB_stop)
    ]

    if windspeed_entry.empty:
        print(f"No windspeed entry found for Date {date}, Leg {leg_index}. Skipping.")
        continue

    windspeed = windspeed_entry['Windspeed'].values[0]
    cluster = windspeed_entry['Windspeed_cluster'].values[0]

    if np.isnan(cluster):
        print(f"Missing cluster for Date {date}, Leg {leg_index}. Skipping.")
        continue

    
    ddry_values = np.array(entry_ddry['filtered_ddry'])
    dryint = filtered_master_BCB_dryintercept_CDP[i]['dry intercept']
    D = entry_ddry['D']

    
    try:
        interp_func = interp1d(ddry_values, size_distribution(ddry_values, dryint, D), 
                               kind='linear', fill_value='extrapolate')
        interpolated_distribution = interp_func(common_bins)

        
        if np.isnan(interpolated_distribution).any():
            print(f"NaN values found after interpolation for Date {date}, Leg {leg_index}. Skipping.")
            continue
    except Exception as e:
        print(f"Interpolation failed for Date {date}, Leg {leg_index}. Error: {e}")
        continue

   
    grouped_distributions_by_cluster[cluster].append(interpolated_distribution)

for cluster, distributions in grouped_distributions_by_cluster.items():
    print(f"Cluster {cluster}: {len(distributions)} legs")

plt.figure(figsize=(12, 8))
for cluster, distributions in grouped_distributions_by_cluster.items():
    if distributions:
        avg_distribution = np.mean(distributions, axis=0)
        lower_percentile = np.percentile(distributions, 25, axis=0)
        upper_percentile = np.percentile(distributions, 75, axis=0)

        plt.plot(common_bins, avg_distribution, label=f"Cluster {cluster} ({len(distributions)} legs)", linewidth=2)
        plt.fill_between(common_bins, lower_percentile, upper_percentile, alpha=0.2)

plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Size distributions by wind speed clusters', fontweight='bold')
plt.legend(title="Wind speed Cluster")
plt.tight_layout()
plt.show()

#%%
#old code from before new 2DS import 

num_bins = 4  # Adjust this number based on your data distribution

# Create windspeed bins based on quantiles
df_with_windspeed['Windspeed_bin'] = pd.qcut(df_with_windspeed['Windspeed'], num_bins, labels=False)

# Display the bins and their corresponding ranges
windspeed_bin_ranges = pd.qcut(df_with_windspeed['Windspeed'], num_bins, retbins=True)[1]
print("Windspeed bin ranges:", windspeed_bin_ranges)

# Group by windspeed bins
grouped_data = df_with_windspeed.groupby('Windspeed_bin').agg({'D': 'mean', 'Ntd': 'mean'}).reset_index()
print(grouped_data)

# Visualize the relationship between windspeed bins, Ntd, and D
plt.figure(figsize=(10, 6))
plt.scatter(grouped_data['D'], grouped_data['Ntd'], 
            c=grouped_data['Windspeed_bin'], cmap='viridis', s=100, label='Windspeed Bin')
plt.colorbar(label='Windspeed Bin')
plt.xlabel('Average Slope (D)', fontsize=14, fontweight='bold')
plt.ylabel('Average Ntd', fontsize=14, fontweight='bold')
plt.title('Average Slope and Ntd for Each Windspeed Bin', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.show()

from sklearn.cluster import KMeans

## can calculate performance of clustering, SSE error, elbow method, we want
##small error, every number of clusters has a different about of error, i want 
##smallest error for smallest nuber of clusters 


windspeed_data = df_with_windspeed['Windspeed'].values.reshape(-1, 1)

# Apply K-means clustering
num_clusters = 2  # Adjust number of clusters based on data
kmeans = KMeans(n_clusters=num_clusters)
df_with_windspeed['Windspeed_cluster'] = kmeans.fit_predict(windspeed_data)

# Visualize cluster assignments
plt.figure(figsize=(10, 6))
plt.scatter(df_with_windspeed['Windspeed'], df_with_windspeed['Ntd'], 
            c=df_with_windspeed['Windspeed_cluster'], cmap='viridis', s=100)
plt.colorbar(label='Windspeed Cluster')
plt.xlabel('Windspeed (m/s)', fontsize=14, fontweight='bold')
plt.ylabel('Ntd', fontsize=14, fontweight='bold')
plt.title('Clustered Windspeed vs. Ntd', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.show()


# Scatter plot with linear fit for each windspeed bin
sns.lmplot(x='D', y='Ntd', hue='Windspeed_bin', data=df_with_windspeed, 
           palette='viridis', ci=None, height=6, aspect=1.5)
plt.xscale('log')
plt.yscale('log')
plt.title('Linear Fit for Each Windspeed Bin', fontsize=14, fontweight='bold')
plt.xlabel('Slope (D)', fontsize=14, fontweight='bold')
plt.ylabel('Total droplet concentration (Ntd)', fontsize=14, fontweight='bold')
plt.show()
#%%
#old code from before new 2DS import
# problematic_legs = [
#     ('2022-01-11', 1), ('2022-01-11', 2), ('2022-01-11', 3), ('2022-01-11', 4),
#     ('2022-01-11', 8), ('2022-01-11', 9), ('2022-01-11', 14), ('2022-01-11', 16),
#     ('2022-01-11', 17),('2022-01-26', 16), ('2022-01-26', 14), ('2022-01-26', 12), 
#     ('2022-01-26', 15),
#     ('2022-01-26', 13), ('2022-01-26', 11), ('2022-03-29', 0), ('2022-05-05', 3),
#     ('2022-05-05', 7), ('2022-05-05', 3), ('2022-06-11', 4), ('2022-03-29', 9), ('2022-03-29', 9), 
#     ('2022-03-29', 0), ('2022-03-29', 6), ('2022-03-28', 4), ('2022-03-28', 0), ('2022-03-26', 15),
#     ('2022-03-26', 6), ('2022-03-26', 0), ('2022-03-13', 8)]
# problematic_set = set(problematic_legs)

# # Define the function to compute log-transformed size distribution
# def log_size_distribution(x, dryint, D):
#     return np.log(dryint * np.exp(-x / D))

# # Define your common bins
# common_bins = np.linspace(0, 25, 20)

# # Define your windspeed ranges
# windspeed_bins = [(0, 4), (4.1, 7), (7.1, 11), (11.1, np.inf)]

# grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
# mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# # Iterate through the entries in filtered_master_min_ddry and match dry intercept
# for i in range(len(filtered_master_BCB_ddry)):
#     entry_ddry = filtered_master_BCB_ddry[i]
#     entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Match by index

#     # Extract necessary values
#     date = entry_ddry['Date']
#     BCB_start = entry_ddry['BCB_start']
#     BCB_stop = entry_ddry['BCB_stop']
#     leg_index = entry_ddry['Leg_index']  # Extract Leg_index

#     # Skip problematic legs
#     if (date, leg_index) in problematic_set:
#         continue

#     D = entry_ddry['D']  # Extract D value
#     ddry_values = np.array(entry_ddry['filtered_ddry'])  # x-axis values for this leg
#     dryint = entry_dryintercept['dry intercept']  # Extract dry intercept

#     # Find the corresponding entry in df_combined based on Date, Min_start, and Min_end
#     windspeed_entry = df_combined[
#         (df_combined['Date'] == date) &
#         (df_combined['BCB_start'] == BCB_start) &
#         (df_combined['BCB_stop'] == BCB_stop)
#     ]

#     # If there's a matching windspeed entry
#     if not windspeed_entry.empty:
#         windspeed = windspeed_entry['Windspeed'].values[0]

#         # Interpolate the log-transformed size distribution for this leg
#         interp_func = interp1d(ddry_values, log_size_distribution(ddry_values, dryint, D), kind='linear', fill_value='extrapolate')
#         interpolated_leg_values = interp_func(common_bins)

#         # Exponentiate the interpolated values to get back to the original scale for plotting
#         interpolated_leg_values = np.exp(interpolated_leg_values)

#         # Categorize the leg based on windspeed range
#         for idx, (low, high) in enumerate(windspeed_bins):
#             if low <= windspeed <= high:
#                 grouped_distributions[idx].append(interpolated_leg_values)
#                 mean_windspeeds[idx].append(windspeed)
#                 break

# # Average and plot the results
# plt.figure(figsize=(12, 8))
# for idx, ranges in enumerate(windspeed_bins):
#     if grouped_distributions[idx]:
#         avg_distribution = np.mean(grouped_distributions[idx], axis=0)
#         avg_windspeed = np.mean(mean_windspeeds[idx])
#         num_legs = len(grouped_distributions[idx])  # Count the number of legs
#         plt.plot(common_bins, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")

# plt.yscale('log')  # Set y-axis to log scale
# plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
# plt.xlabel('Bin diameter (µm)', fontweight='bold')
# plt.title('Size Distribution by Windspeed (Log Scale, Excluding Problematic Legs)', fontweight='bold')
# plt.legend(title="Windspeed")
# plt.tight_layout()
# plt.show()
#%%

#%%
#How windspeed slope and NT correlate with windspeed
Z0 = 0.02  
Z10 = 10  

def correct_windspeed(windspeed, altitude):
    return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))

combined_data = {
    'Date': [],
    'BCB_start': [],
    'BCB_stop': [],
    'filtered_NT': [],
    'D': [],
    'Windspeed': []
}

for i, flight in enumerate(master_BCB_CDP):
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

            
            filtered_NT = None
            D = None

            
            if date in master_BCB_exponential:
                for exp_params in master_BCB_exponential[date]:
       
                    if exp_params['BCB_start'] == BCB_start and exp_params['BCB_stop'] == BCB_stop:
                        D = exp_params['D']
                        print(f"Found D for Date: {date}, BCB_start: {BCB_start}, BCB_stop: {BCB_stop}, D: {D}")
                        break

           
            for filtered_NT_data in filtered_master_BCB_NT_CDP:
                if filtered_NT_data['Date'] == date and filtered_NT_data['BCB_start'] == BCB_start and filtered_NT_data['BCB_stop'] == BCB_stop:
                    filtered_NT = filtered_NT_data['filtered_NT']
                    break

           
            if filtered_NT is not None and D is not None:
                combined_data['Date'].append(date)
                combined_data['BCB_start'].append(BCB_start)
                combined_data['BCB_stop'].append(BCB_stop)
                combined_data['filtered_NT'].append(filtered_NT)
                combined_data['D'].append(D)
                combined_data['Windspeed'].append(corrected_windspeed)

        except IndexError as e:
            print(f"Index error at i={i}, j={j}: {e}")
            continue

df_combined = pd.DataFrame(combined_data)


print(df_combined.describe())

df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]


plt.figure(figsize=(10, 8))


sc = plt.scatter(df_with_windspeed['D'], df_with_windspeed['filtered_NT'], 
                 c=df_with_windspeed['Windspeed'], cmap='viridis', s=100, label='Windspeed')

plt.scatter(df_nan_windspeed['D'], df_nan_windspeed['filtered_NT'], 
            color='black', s=100, label='Windspeed NaN')
plt.colorbar(sc, label='10 wind speed (m/s)')
plt.xlabel('Slope', fontsize=14, fontweight='bold')
plt.ylabel('Ambient concentration', fontsize=14, fontweight='bold')
plt.title('Below cloud base January - June 2022', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.ylim(10**-3.5, 10**0.3)

plt.show()
#%%
#Ambient distribution

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.hist(df_with_windspeed['Windspeed'], bins=30, color='skyblue', edgecolor='black')
plt.title('wind speed distribution', fontweight='bold')
plt.xlabel('wind speed (m/s)', fontweight='bold')
plt.ylabel('Frequency', fontweight='bold')

plt.subplot(3, 1, 2)
plt.hist(df_with_windspeed['filtered_NT'], bins=30, color='lightgreen', edgecolor='black')
plt.title('Ambient distribution (/cm^3/um)', fontweight='bold')
plt.xlabel('Ambient concentration (/cm^3/um)', fontweight='bold')
plt.ylabel('Frequency', fontweight='bold')
plt.yscale('log')

plt.subplot(3, 1, 3)
plt.hist(df_with_windspeed['D'], bins=30, color='lightcoral', edgecolor='black')
plt.title('Slope distribution', fontweight='bold')
plt.xlabel('Slope', fontweight='bold')
plt.ylabel('Frequency', fontweight='bold')
plt.tight_layout()
plt.show()
# %%

#Old code from before 2DS import
# # post k-means clustering 
# problematic_legs = [
#        ('2022-01-11', 1),
#     ('2022-01-11', 2),
#     ('2022-01-11', 3),
#     ('2022-01-11', 4),
#     ('2022-01-11', 8),
#     ('2022-01-11', 9),
#     ('2022-01-11', 14),
#     ('2022-01-11', 16),
#     ('2022-01-11', 17),
#     ('2022-01-26', 14),
#     ('2022-01-26', 12),
#     ('2022-01-26', 15),
#     ('2022-01-26', 13),
#     ('2022-01-26', 11),
#     ('2022-03-29', 0),
#     ('2022-05-05', 3),
#     ('2022-05-05', 7),
# ]
# problematic_set = set(problematic_legs)

# # Define the function to compute log-transformed size distribution
# def log_size_distribution(x, dryint, D):
#     return np.log(dryint * np.exp(-x / D))

# # Define your common bins
# common_bins = np.linspace(0, 25, 20)

# # Define your windspeed ranges
# windspeed_bins = [(0, 4), (4.1, 7), (7.1, 11), (11.1, np.inf)]

# grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
# mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# # Iterate through the entries in filtered_master_min_ddry and match dry intercept
# for i in range(len(filtered_master_BCB_ddry)):
#     entry_ddry = filtered_master_BCB_ddry[i]
#     entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Match by index

#     # Extract necessary values
#     date = entry_ddry['Date']
#     BCB_start = entry_ddry['BCB_start']
#     BCB_stop = entry_ddry['BCB_stop']
#     leg_index = entry_ddry['Leg_index']  # Extract Leg_index

#     # Skip problematic legs
#     if (date, leg_index) in problematic_set:
#         continue

#     D = entry_ddry['D']  # Extract D value
#     ddry_values = np.array(entry_ddry['filtered_ddry'])  # x-axis values for this leg
#     dryint = entry_dryintercept['dry intercept']  # Extract dry intercept

#     # Find the corresponding entry in df_combined based on Date, Min_start, and Min_end
#     windspeed_entry = df_combined[
#         (df_combined['Date'] == date) &
#         (df_combined['BCB_start'] == BCB_start) &
#         (df_combined['BCB_stop'] == BCB_stop)
#     ]

#     # If there's a matching windspeed entry
#     if not windspeed_entry.empty:
#         windspeed = windspeed_entry['Windspeed'].values[0]

#         # Interpolate the log-transformed size distribution for this leg
#         interp_func = interp1d(ddry_values, log_size_distribution(ddry_values, dryint, D), kind='linear', fill_value='extrapolate')
#         interpolated_leg_values = interp_func(common_bins)

#         # Exponentiate the interpolated values to get back to the original scale for plotting
#         interpolated_leg_values = np.exp(interpolated_leg_values)

#         # Categorize the leg based on windspeed range
#         for idx, (low, high) in enumerate(windspeed_bins):
#             if low <= windspeed <= high:
#                 grouped_distributions[idx].append(interpolated_leg_values)
#                 mean_windspeeds[idx].append(windspeed)
#                 break

# # Average and plot the results
# plt.figure(figsize=(12, 8))
# for idx, ranges in enumerate(windspeed_bins):
#     if grouped_distributions[idx]:
#         avg_distribution = np.mean(grouped_distributions[idx], axis=0)
#         avg_windspeed = np.mean(mean_windspeeds[idx])
#         num_legs = len(grouped_distributions[idx])  # Count the number of legs
        
#         # Calculate percentiles for shading
#         lower_percentile = np.percentile(grouped_distributions[idx], 25, axis=0)
#         upper_percentile = np.percentile(grouped_distributions[idx], 75, axis=0)

#         # Plot the average distribution
#         plt.plot(common_bins, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")

#         # Plot the shaded area for the interquartile range
#         plt.fill_between(common_bins, lower_percentile, upper_percentile, alpha=0.4)

#         # Optionally plot the percentile lines
#         plt.plot(common_bins, lower_percentile, linestyle='--', linewidth=1, color='gray')
#         plt.plot(common_bins, upper_percentile, linestyle='--', linewidth=1, color='gray')

# plt.yscale('log')  # Set y-axis to log scale
# plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
# plt.xlabel('Bin diameter (µm)', fontweight='bold')
# plt.title('Size Distribution by Windspeed (Log Scale, Excluding Problematic Legs)', fontweight='bold')
# plt.legend(title="Windspeed")
# plt.tight_layout()
# plt.show()
# %%

#%%
#Average across all bins and then fit a distrubtion using the average 
common_bins = np.linspace(0, 10, 25)
# common_bins = np.linspace(0, 20, 25)  old bins from before 2DS import

#old bins from before 2DS import
# windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]

windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]
grouped_concentrations = {i: [] for i in range(len(windspeed_bins))}  # To accumulate droplet concentrations
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Iterate through the entries in filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry_CDP)):
    entry_ddry = filtered_master_BCB_ddry_CDP[i]
    entry_dryintercept = filtered_master_BCB_dryintercept_CDP[i]  # Match by index

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']  # Extract Leg_index

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']  # Extract D value
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # x-axis values for this leg
    dryint = entry_dryintercept['dry intercept']  # Extract dry intercept

    # Find the corresponding entry in df_combined based on Date, Min_start, and Min_end
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    
    # If there's a matching windspeed entry
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]
        
        # Interpolate the size distribution for this leg
        size_dist = size_distribution(ddry_values, dryint, D)
        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)
        
        # Categorize the leg based on windspeed range
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_concentrations[idx].append(interpolated_leg_values)  # Add size distribution to this bin
                mean_windspeeds[idx].append(windspeed)
                break

# Step 2: Average each particle size bin, compute standard deviation for error bars, then fit
plt.figure(figsize=(12, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_concentrations[idx]:
        
        concentrations_array = np.array(grouped_concentrations[idx])
        

        avg_concentration = np.mean(concentrations_array, axis=0)
        
        
        std_dev = np.std(concentrations_array, axis=0)
        
        
        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)  # Replace zeros with small value
        
        
        def fit_function(x, dryint, D):
            return dryint * np.exp(-x / D)
        
        popt, _ = curve_fit(fit_function, common_bins, avg_concentration, p0=[1, 1])  # Initial guess for dryint, D
        
        
        fitted_curve = fit_function(common_bins, *popt)
        
        
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_concentrations[idx])
        
       
        plt.plot(common_bins, fitted_curve, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")
        
        
        plt.errorbar(common_bins, avg_concentration, yerr=std_dev, fmt='o', capsize=5, label=f"{low}-{high} m/s")

plt.yscale('log')
plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Fitted size distribution by wind speed', fontweight='bold')
plt.legend(title="wind speed (m/s)")
plt.tight_layout()
plt.show()
#%%

def size_distribution(x, dryint, D):
    return dryint * np.exp(-x / D)

common_bins = np.linspace(0, 10, 25)
windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]
grouped_concentrations = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

missing_windspeed_count = 0
interpolation_failures = 0
problematic_legs_skipped = 0

print(f"Total input legs: {len(filtered_master_BCB_ddry_CDP)}")


for i in range(len(filtered_master_BCB_ddry_CDP)):
    entry_ddry = filtered_master_BCB_ddry_CDP[i]
    entry_dryintercept = filtered_master_BCB_dryintercept_CDP[i]
    
    
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']

    
    if (date, leg_index) in problematic_set:
        problematic_legs_skipped += 1
        continue

    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])
    dryint = entry_dryintercept['dry intercept']

    
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    
    if windspeed_entry.empty:
        missing_windspeed_count += 1
        continue
    
    windspeed = windspeed_entry['Windspeed'].values[0]

    
    try:
        size_dist = size_distribution(ddry_values, dryint, D)
        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_values = interp_func(common_bins)
        interpolated_values = np.nan_to_num(interpolated_values, nan=0.0, posinf=0.0, neginf=0.0)  # Clean NaNs/infs
    except Exception as e:
        interpolation_failures += 1
        continue

   
    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:
            grouped_concentrations[idx].append(interpolated_values)
            mean_windspeeds[idx].append(windspeed)
            break


print(f"Total problematic legs skipped: {problematic_legs_skipped}")
print(f"Total legs with missing windspeed data: {missing_windspeed_count}")
print(f"Total interpolation failures: {interpolation_failures}")
plt.figure(figsize=(12, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_concentrations[idx]:
        concentrations_array = np.array(grouped_concentrations[idx])

    
        concentrations_array = concentrations_array[~np.isnan(concentrations_array).any(axis=1)]
        if len(concentrations_array) == 0:
            print(f" - Bin {idx} has no valid data after cleaning. Skipping.")
            continue

        avg_concentration = np.mean(concentrations_array, axis=0)
        std_dev = np.std(concentrations_array, axis=0)

        
        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)

        
        def fit_function(x, dryint, D):
            return dryint * np.exp(-x / D)
        
        try:
            popt, _ = curve_fit(fit_function, common_bins, avg_concentration, p0=[1, 1])
            fitted_curve = fit_function(common_bins, *popt)
        except Exception as e:
            print(f"Curve fitting failed for bin {idx}. Error: {e}")
            continue

        
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_concentrations[idx])

        plt.plot(common_bins, fitted_curve, label=f"Fit: {avg_windspeed:.1f} m/s, n={num_legs} legs", linewidth=2)
        plt.errorbar(common_bins, avg_concentration, yerr=std_dev, fmt='o', capsize=5, label=f"Data: {low}-{high} m/s")

plt.yscale('log')
plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Fitted size distribution by wind speed', fontweight='bold')
plt.legend(title="Wind speed (m/s)")
plt.tight_layout()
plt.show()

print(f"Total legs plotted: {sum(len(group) for group in grouped_concentrations.values())}")

#%%

def size_distribution(x, dryint, D):
    return dryint * np.exp(-x / D)

common_bins = np.linspace(0, 10, 10)
windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]
grouped_concentrations = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

missing_windspeed_count = 0
interpolation_failures = 0
problematic_legs_skipped = 0

print(f"Total input legs: {len(filtered_master_BCB_ddry_CDP)}")


for i in range(len(filtered_master_BCB_ddry_CDP)):
    entry_ddry = filtered_master_BCB_ddry_CDP[i]
    entry_dryintercept = filtered_master_BCB_dryintercept_CDP[i]
    
    
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']

    
    if (date, leg_index) in problematic_set:
        problematic_legs_skipped += 1
        continue

    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])
    dryint = entry_dryintercept['dry intercept']

    
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    
    if windspeed_entry.empty:
        missing_windspeed_count += 1
        continue
    
    windspeed = windspeed_entry['Windspeed'].values[0]

    
    try:
        size_dist = size_distribution(ddry_values, dryint, D)
        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_values = interp_func(common_bins)
        interpolated_values = np.nan_to_num(interpolated_values, nan=0.0, posinf=0.0, neginf=0.0)  # Clean NaNs/infs
    except Exception as e:
        interpolation_failures += 1
        continue

   
    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:
            grouped_concentrations[idx].append(interpolated_values)
            mean_windspeeds[idx].append(windspeed)
            break


print(f"Total problematic legs skipped: {problematic_legs_skipped}")
print(f"Total legs with missing windspeed data: {missing_windspeed_count}")
print(f"Total interpolation failures: {interpolation_failures}")
plt.figure(figsize=(12, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_concentrations[idx]:
        concentrations_array = np.array(grouped_concentrations[idx])

    
        concentrations_array = concentrations_array[~np.isnan(concentrations_array).any(axis=1)]
        if len(concentrations_array) == 0:
            print(f" - Bin {idx} has no valid data after cleaning. Skipping.")
            continue

        avg_concentration = np.mean(concentrations_array, axis=0)
        std_dev = np.std(concentrations_array, axis=0)

        
        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)

        
        def fit_function(x, dryint, D):
            return dryint * np.exp(-x / D)
        
        try:
            popt, _ = curve_fit(fit_function, common_bins, avg_concentration, p0=[1, 1])
            fitted_curve = fit_function(common_bins, *popt)
        except Exception as e:
            print(f"Curve fitting failed for bin {idx}. Error: {e}")
            continue

        
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_concentrations[idx])

        plt.plot(common_bins, fitted_curve, label=f"Fit: {avg_windspeed:.1f} m/s, n={num_legs} legs", linewidth=2)
        plt.errorbar(common_bins, avg_concentration, yerr=std_dev, fmt='o', capsize=5, label=f"Data: {low}-{high} m/s")

plt.yscale('log')
plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Fitted size distribution by wind speed', fontweight='bold')
plt.legend(title="Wind speed (m/s)")
plt.tight_layout()
plt.show()

print(f"Total legs plotted: {sum(len(group) for group in grouped_concentrations.values())}")

#%%
##trying to fix size distribution and add ntd 

# def size_distribution(x, dryint, D):
#     return dryint * np.exp(-x / D)
# common_bins = np.linspace(0, 10, 25)

# interpolated_values = []

# for i in range(len(filtered_master_BCB_ddry)):
#     entry_ddry = filtered_master_BCB_ddry[i]
#     entry_dryintercept = filtered_master_BCB_dryintercept[i]  
    
#     date = entry_ddry['Date']
#     BCB_start = entry_ddry['BCB_start']
#     BCB_stop = entry_ddry['BCB_stop']
#     leg_index = entry_ddry['Leg_index']  
#     D = entry_ddry['D'] 
#     ddry_values = np.array(entry_ddry['filtered_ddry'])  
#     dryint = entry_dryintercept['dry intercept']  
   
#     interp_func = interp1d(ddry_values, size_distribution(ddry_values, dryint, D), kind='linear', fill_value='extrapolate')
#     interpolated_leg_values = interp_func(common_bins)

    
#     key = (date, BCB_start, BCB_stop)


#     if key in filtered_master_BCB_ntd_dict:
#         Ntd_value = filtered_master_BCB_ntd_dict[key]['Ntd']
#     else:
#         Ntd_value = np.nan  # Use NaN or some other placeholder if no match is found

    
#     interpolated_values.append({
#         'Date': date,
#         'Leg_index': leg_index,
#         'BCB_start': BCB_start,
#         'BCB_stop': BCB_stop,
#         'interpolated_values': interpolated_leg_values.tolist(),
#         'Ntd': Ntd_value  
#     })

# plt.figure(figsize=(12, 8))

# for entry in interpolated_values:
#     date = entry['Date']
#     leg_index = entry['Leg_index']
#     BCB_start = entry['BCB_start']
#     BCB_stop = entry['BCB_stop']
#     interpolated_leg_values = entry['interpolated_values']
#     Ntd_value = entry['Ntd']
    
    
#     if not np.isnan(Ntd_value):  # Only plot if Ntd is available
#         plt.plot(common_bins, [Ntd_value] * len(common_bins), label=f"Date: {date}, Leg: {leg_index} ({BCB_start} - {BCB_stop})")

# plt.ylabel('Total Number Concentration (Ntd) (/cm^3)', fontweight='bold')
# plt.xlabel('Bin diameter (um)', fontweight='bold')
# plt.title('Below cloud base total concentration January-June 2022', fontweight='bold')
# plt.tight_layout()
# plt.yscale('log')
# plt.show()
#%%
##trying to fix size distribution and add ntd 

def size_distribution(d, dryint, D):
    return dryint * np.exp(-d / D)

dmin = 2  
dmax = np.inf 

integrated_concentrations = []
for i in range(len(filtered_master_BCB_ddry_CDP)):
    entry_ddry = filtered_master_BCB_ddry_CDP[i]
    entry_dryintercept = filtered_master_BCB_dryintercept_CDP[i]  
    
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index'] 
    D = entry_ddry['D']  
    dryint = entry_dryintercept['dry intercept']  
    

    total_concentration, _ = quad(size_distribution, dmin, dmax, args=(dryint, D))
    
    
    integrated_concentrations.append({
        'Date': date,
        'Leg_index': leg_index,
        'BCB_start': BCB_start,
        'BCB_stop': BCB_stop,
        'Total Concentration': total_concentration
    })


for total in integrated_concentrations:
    print(f"Date: {total['Date']}, Leg_index: {total['Leg_index']}, "
          f"Total Concentration: {total['Total Concentration']:.3f} /cm^3/um")

dates = [entry['Date'] for entry in integrated_concentrations]
legs = [entry['Leg_index'] for entry in integrated_concentrations]
total_concs = [entry['Total Concentration'] for entry in integrated_concentrations]

plt.figure(figsize=(12, 8))
plt.bar(range(len(total_concs)), total_concs, tick_label=[f"{d}-{l}" for d, l in zip(dates, legs)])
plt.ylabel('Total Concentration (/cm^3/um)', fontweight='bold')
plt.xlabel('Leg (Date-Leg)', fontweight='bold')
plt.title('Total below cloud base concentration (/cm^3/um)', fontweight='bold')
plt.tight_layout()
plt.show()
#%%
#Size distributiona using d instead of ddry
def size_distribution(d, dryint, D):
    return dryint * np.exp(-d / D)

dmin = 2  
dmax = np.inf  


# common_bins = np.linspace(0, 25, 20) old bins from before 2DS import
common_bins = np.linspace(0, 10, 25)

integrated_concentrations = []
size_distributions = []

for i in range(len(filtered_master_BCB_ddry_CDP)):
    entry_ddry = filtered_master_BCB_ddry_CDP[i]
    entry_dryintercept = filtered_master_BCB_dryintercept_CDP[i] 
    
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index'] 
    D = entry_ddry['D'] 
    dryint = entry_dryintercept['dry intercept']  
    

    total_concentration, _ = quad(size_distribution, dmin, dmax, args=(dryint, D))
    
    distribution_values = size_distribution(common_bins, dryint, D)

    integrated_concentrations.append({
        'Date': date,
        'Leg_index': leg_index,
        'BCB_start': BCB_start,
        'BCB_stop': BCB_stop,
        'Total Concentration': total_concentration
    })
    
    size_distributions.append({
        'Date': date,
        'Leg_index': leg_index,
        'BCB_start': BCB_start,
        'BCB_stop': BCB_stop,
        'Distribution Values': distribution_values
    })


for total in integrated_concentrations:
    print(f"Date: {total['Date']}, Leg_index: {total['Leg_index']}, "
          f"Total Concentration: {total['Total Concentration']:.3f} /cm^3")

plt.figure(figsize=(12, 8))

for dist in size_distributions:
    date = dist['Date']
    leg_index = dist['Leg_index']
    BCB_start = dist['BCB_start']
    BCB_stop = dist['BCB_stop']
    dist_values = dist['Distribution Values']
    
    plt.plot(common_bins, dist_values, label=f"Date: {date}, Leg: {leg_index} ({BCB_start} - {BCB_stop})")

plt.ylabel('Total droplet concentration (/cm^3/um)', fontweight='bold')
plt.xlabel('Bin diameter (um)', fontweight='bold')
plt.title('Below cloud base January - June 2022', fontweight='bold')
plt.tight_layout()
plt.show()
#%%

#size distribution using ddry instead of d
def size_distribution(ddry, N0, D):
    return N0 * np.exp(-ddry / D)

# common_bins = np.linspace(0, 25, 20)
common_bins = np.linspace(0, 16, 25)
plt.figure(figsize=(12, 8))

for i in range(len(filtered_master_BCB_ddry_CDP)):
    entry_ddry = filtered_master_BCB_ddry_CDP[i]
    entry_ntd = filtered_master_BCB_ntd_CDP[i]  

    date = entry_ddry['Date']
    leg_index = entry_ddry['Leg_index']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    D = entry_ddry['D']  
    Ntd = entry_ntd['Ntd'] 
    ddry_values = np.array(entry_ddry['filtered_ddry']) 
    N0 = entry_ddry['n0']  
    
    size_dist_values = size_distribution(ddry_values, dryintercept, D)
    
    
    interp_func = interp1d(ddry_values, size_dist_values, kind='linear', fill_value='extrapolate')
    interpolated_leg_values = interp_func(common_bins)

    plt.plot(common_bins, interpolated_leg_values, label=f"Date: {date}, Leg: {leg_index} ({BCB_start} - {BCB_stop})")

plt.ylabel('Concentration (/cm^3/μm)', fontweight='bold')
plt.xlabel('Dry diameter (μm)', fontweight='bold')
plt.title('Below cloud base January-June 2022', fontweight='bold')
plt.tight_layout()
plt.ylim(0, 1)
plt.show()
#%%
#skipping problem legs
# Size distribution using ddry instead of d
def size_distribution(ddry, N0, D):
    return N0 * np.exp(-ddry / D)

common_bins = np.linspace(0, 16, 25)
plt.figure(figsize=(12, 8))

for i in range(len(filtered_master_BCB_ddry_CDP)):
    entry_ddry = filtered_master_BCB_ddry_CDP[i]
    entry_ntd = filtered_master_BCB_ntd_CDP[i]

    date = entry_ddry['Date']
    leg_index = entry_ddry['Leg_index']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    D = entry_ddry['D']
    Ntd = entry_ntd['Ntd']
    ddry_values = np.array(entry_ddry['filtered_ddry'])
    N0 = entry_ddry['n0']

    # Compute size distribution and interpolate
    size_dist_values = size_distribution(ddry_values, N0, D)
    interp_func = interp1d(ddry_values, size_dist_values, kind='linear', fill_value='extrapolate')
    interpolated_leg_values = interp_func(common_bins)

    # Check if any interpolated values are below 0
    if np.any(interpolated_leg_values < 0):
        print(f"Skipping leg {date}, {leg_index} due to negative values")
        continue  # Skip this leg if any values are negative

    plt.plot(common_bins, interpolated_leg_values, label=f"Date: {date}, Leg: {leg_index} ({BCB_start} - {BCB_stop})")

plt.ylabel('Concentration (/cm³/μm)', fontweight='bold')
plt.xlabel('Dry diameter (μm)', fontweight='bold')
plt.title('Below cloud base January-June 2022', fontweight='bold')
plt.ylim(0, 10)  # Keep the y-limit as specified
plt.tight_layout()
plt.show()
#%%
def size_distribution(ddry, N0, D):
    return N0 * np.exp(-ddry / D)

common_bins = np.linspace(0, 10, 10)
windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]
grouped_concentrations = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

skipped_nan = 0
skipped_no_bin = 0

for i in range(len(filtered_master_BCB_ddry_CDP)):
    entry_ddry = filtered_master_BCB_ddry_CDP[i]
    entry_dryintercept = filtered_master_BCB_dryintercept_CDP[i]

    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']
    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])
    N0 = entry_ddry['n0']

    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]

        size_dist_values = size_distribution(ddry_values, N0, D)

        if np.any(np.isnan(size_dist_values)):
            skipped_nan += 1
            continue

        interp_func = interp1d(ddry_values, size_dist_values, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        # Check for NaN values in interpolated results
        if np.any(np.isnan(interpolated_leg_values)):
            skipped_nan += 1
            continue

        # Assign to the appropriate windspeed bin
        assigned = False
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_concentrations[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                assigned = True
                break
        
        if not assigned:
            skipped_no_bin += 1

# Debugging outputs
print(f"Skipped legs due to NaN values: {skipped_nan}")
print(f"Skipped legs due to no windspeed bin match: {skipped_no_bin}")

# Plot averaged size distributions
plt.figure(figsize=(12, 8))
for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_concentrations[idx]:
        concentrations_array = np.array(grouped_concentrations[idx])
        avg_concentration = np.mean(concentrations_array, axis=0)
        std_dev = np.std(concentrations_array, axis=0)
        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)  # Avoid zeros
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_concentrations[idx])

        plt.errorbar(common_bins, avg_concentration, yerr=std_dev, fmt='o', capsize=5, 
                     label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linestyle='-', alpha=0.7)

plt.yscale('log')
plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Average size distribution by wind speed', fontweight='bold')
plt.legend(title="Average wind speed (m/s)")
plt.tight_layout()
plt.show()

#%%
 #Dry size distribution function CORRECT

def size_distribution(ddry, N0, D):
    return N0 * np.exp(-ddry / D)

common_bins = np.linspace(0, 10, 10)

# windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)] old bins from before 2DS import
windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]
grouped_concentrations = {i: [] for i in range(len(windspeed_bins))}  
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}


for i in range(len(filtered_master_BCB_ddry_CDP)):
    entry_ddry = filtered_master_BCB_ddry_CDP[i]
    entry_dryintercept = filtered_master_BCB_dryintercept_CDP[i]

    
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']
    D = entry_ddry['D']  
    ddry_values = np.array(entry_ddry['filtered_ddry']) 
    N0 = entry_ddry['n0']  

    
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]

        size_dist_values = size_distribution(ddry_values, N0, D)
        
       
        if np.any(np.isnan(size_dist_values)):
            continue  
        
        
        interp_func = interp1d(ddry_values, size_dist_values, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)  

        
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_concentrations[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break

# Debugging outputs
print(f"Skipped legs due to NaN values: {skipped_nan}")
print(f"Skipped legs due to no windspeed bin match: {skipped_no_bin}")
plt.figure(figsize=(12, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_concentrations[idx]:
       
        concentrations_array = np.array(grouped_concentrations[idx])
        avg_concentration = np.mean(concentrations_array, axis=0)
        std_dev = np.std(concentrations_array, axis=0)
        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)  
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_concentrations[idx])

        plt.errorbar(common_bins, avg_concentration, yerr=std_dev, fmt='o', capsize=5, 
                     label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linestyle='-', alpha=0.7)


plt.yscale('log')
plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Average size distribution by wind speed ', fontweight='bold')
plt.legend(title="Average wind speed (m/s)")
plt.tight_layout()
plt.show()
#%%
def fit_function(x, D):
    return dryint * np.exp(-x / D)

common_bins = np.linspace(0, 10, 10)  
# windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]
windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]
grouped_concentrations = {i: [] for i in range(len(windspeed_bins))}  # To accumulate droplet concentrations
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

for i in range(len(filtered_master_BCB_ddry_CDP)):
    entry_ddry = filtered_master_BCB_ddry_CDP[i]
    entry_dryintercept = filtered_master_BCB_dryintercept_CDP[i]

  
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']

  
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])
    dryint = entry_dryintercept['dry intercept']

   
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]

        
        size_dist = size_distribution(ddry_values, dryint, D)  
        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)  

        
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_concentrations[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break


plt.figure(figsize=(12, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_concentrations[idx]:
        
        concentrations_array = np.array(grouped_concentrations[idx])
        
        
        avg_concentration = np.mean(concentrations_array, axis=0)
        
        
        std_dev = np.std(concentrations_array, axis=0)
        
       
        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)  # Replace zeros with small value
        
        
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_concentrations[idx])
        
        
        plt.errorbar(common_bins, avg_concentration, yerr=std_dev, fmt='o', capsize=5, 
                     label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linestyle='-', alpha=0.7)

        try:
    # Fit the exponential model to the average concentration
            popt, pcov = curve_fit(fit_function, common_bins, avg_concentration, p0=[1, 1])
            dryint_fit, D_fit = popt

    # Generate fitted values for plotting
            fitted_values = fit_function(common_bins, dryint_fit, D_fit)
            plt.plot(common_bins, fitted_values, linestyle='--', 
                    label=f"Fit {avg_windspeed:.1f} m/s: y = {dryint_fit:.2f} * exp(-x / {D_fit:.2f})")
    
            # Print the fitted equation
            print(f"Windspeed {avg_windspeed:.1f} m/s: y = {dryint_fit:.2f} * exp(-x / {D_fit:.2f})")

        except RuntimeError:
            print(f"Could not fit the exponential model for windspeed {avg_windspeed:.1f} m/s")
        # try:
        #     popt, pcov = curve_fit(fit_function, common_bins, avg_concentration, p0=[1, 1])
        #     dryint_fit, D_fit = popt

            
        #     fitted_values = fit_function(common_bins, dryint_fit, D_fit)
        #     plt.plot(common_bins, fitted_values, linestyle='--', 
        #              label=f"Fit {avg_windspeed:.1f} m/s: y = {dryint_fit:.2f} * exp(-x / {D_fit:.2f})")
            
           
        #     print(f"Windspeed {avg_windspeed:.1f} m/s: y = {dryint_fit:.2f} * exp(-x / {D_fit:.2f})")

        # except RuntimeError:
        #     print(f"Could not fit the exponential model for windspeed {avg_windspeed:.1f} m/s")
        
plt.yscale('log')
plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Average size distribution by wind speed', fontweight='bold')
plt.legend(title="Average wind speed (m/s)")
plt.tight_layout()
plt.show()
#%%

def size_distribution(ddry, N0, D):
    return N0 * np.exp(-ddry / D)

def fit_function(x, N0, D):
    return N0 * np.exp(-x / D)

common_bins = np.linspace(0, 10, 10)

# windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]
windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]  
grouped_concentrations = {i: [] for i in range(len(windspeed_bins))}  # To accumulate droplet concentrations
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}


for i in range(len(filtered_master_BCB_ddry_CDP)):
    entry_ddry = filtered_master_BCB_ddry_CDP[i]
    entry_dryintercept = filtered_master_BCB_dryintercept_CDP[i]

    
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']
    N0 = entry_ddry['n0']
    D = entry_ddry['D']  
    ddry_values = np.array(entry_ddry['filtered_ddry'])  
    dryint = entry_dryintercept['dry intercept']  

    
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]

       
        size_dist_values = size_distribution(ddry_values, dryint, D)  

        
        if np.any(np.isnan(size_dist_values)):
            continue  

        
        interp_func = interp1d(ddry_values, size_dist_values, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins) 

        
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_concentrations[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break

plt.figure(figsize=(12, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_concentrations[idx]:
        
        concentrations_array = np.array(grouped_concentrations[idx])
        
        
        avg_concentration = np.mean(concentrations_array, axis=0)
        
        
        std_dev = np.std(concentrations_array, axis=0)
        
        
        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)  # Replace zeros with small value
        
        
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_concentrations[idx])
        
        
        plt.errorbar(common_bins, avg_concentration, yerr=std_dev, fmt='o', capsize=5, 
                     label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linestyle='-', alpha=0.7)

        
        try:
            popt, pcov = curve_fit(fit_function, common_bins, avg_concentration, p0=[1, 1])
            dryint_fit, D_fit = popt

            
            fitted_values = fit_function(common_bins, dryint_fit, D_fit)
            plt.plot(common_bins, fitted_values, linestyle='--', 
                     label=f"Fit {avg_windspeed:.1f} m/s: y = {dryint_fit:.2f} * exp(-x / {D_fit:.2f})")
            
           
            print(f"Windspeed {avg_windspeed:.1f} m/s: y = {dryint_fit:.2f} * exp(-x / {D_fit:.2f})")

        except RuntimeError:
            print(f"Could not fit the exponential model for windspeed {avg_windspeed:.1f} m/s")


plt.yscale('log')
plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Average size distribution by wind speed', fontweight='bold')
plt.legend(title="Average wind speed (m/s)")
plt.tight_layout()
plt.show()
#%%


def size_distribution(ddry, N0, D):
    return N0 * np.exp(-ddry / D)
def fit_function(x, N0, D):
    return N0 * np.exp(-x / D)

common_bins = np.linspace(0, 10, 10)


# windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]
windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]
grouped_concentrations = {i: [] for i in range(len(windspeed_bins))}  # To accumulate droplet concentrations
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}


for i in range(len(filtered_master_BCB_ddry_CDP)):
    entry_ddry = filtered_master_BCB_ddry_CDP[i]


    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']
    N0 = entry_ddry['n0']  
    D = entry_ddry['D'] 
    ddry_values = np.array(entry_ddry['filtered_ddry'])  

    
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]

        
        size_dist_values = size_distribution(ddry_values, N0, D)  

        
        if np.any(np.isnan(size_dist_values)):
            continue  

        
        interp_func = interp1d(ddry_values, size_dist_values, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins) 

        
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_concentrations[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break

plt.figure(figsize=(12, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_concentrations[idx]:
        
        concentrations_array = np.array(grouped_concentrations[idx])
        
       
        avg_concentration = np.mean(concentrations_array, axis=0)
        
       
        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)  # Replace zeros with small value
        
       
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_concentrations[idx])
        
        
        plt.plot(common_bins, avg_concentration, 'o-', label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", alpha=0.7)

    
        try:
            popt, pcov = curve_fit(fit_function, common_bins, avg_concentration, p0=[1, 1])
            N0_fit, D_fit = popt

           
            fitted_values = fit_function(common_bins, N0_fit, D_fit)
            plt.plot(common_bins, fitted_values, linestyle='--', 
                     label=f"Fit {avg_windspeed:.1f} m/s: y = {N0_fit:.2f} * exp(-x / {D_fit:.2f})")
            
            
            print(f"Windspeed {avg_windspeed:.1f} m/s: y = {N0_fit:.2f} * exp(-x / {D_fit:.2f})")

        except RuntimeError:
            print(f"Could not fit the exponential model for windspeed {avg_windspeed:.1f} m/s")

plt.yscale('log')

# for idx, (low, high) in enumerate(windspeed_bins):
#     if grouped_concentrations[idx]:
#         # Convert the list of arrays to a numpy array for easier calculations
#         concentrations_array = np.array(grouped_concentrations[idx])
        
#         # Average each size bin (column) across all size distributions
#         avg_concentration = np.mean(concentrations_array, axis=0)
        
#         # Calculate standard deviation for error bars
#         std_dev = np.std(concentrations_array, axis=0)
        
#         # Handle zeros in avg_concentration for log scale plotting
#         avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)  # Replace zeros with small value
        
#         # Calculate average windspeed and number of legs in this bin
#         avg_windspeed = np.mean(mean_windspeeds[idx])
#         num_legs = len(grouped_concentrations[idx])
        
#         # Plot the average concentration with error bars and connecting lines
#         plt.errorbar(common_bins, avg_concentration, yerr=std_dev, fmt='o', capsize=5, 
#                      label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linestyle='-', alpha=0.7)

#         # Step 5: Fit your custom exponential model to the average concentrations
#         try:
#             popt, pcov = curve_fit(fit_function, common_bins, avg_concentration, p0=[1, 1])
#             N0_fit, D_fit = popt

#             # Plot the fitted exponential line
#             fitted_values = fit_function(common_bins, N0_fit, D_fit)
#             plt.plot(common_bins, fitted_values, linestyle='--', 
#                      label=f"Fit {avg_windspeed:.1f} m/s: y = {N0_fit:.2f} * exp(-x / {D_fit:.2f})")
            
#             # Print the equation for each line
#             print(f"Windspeed {avg_windspeed:.1f} m/s: y = {N0_fit:.2f} * exp(-x / {D_fit:.2f})")

#         except RuntimeError:
#             print(f"Could not fit the exponential model for windspeed {avg_windspeed:.1f} m/s")


plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin center diameter (µm)', fontweight='bold')
plt.title('Below cloud base January - June 2022', fontweight='bold')
plt.legend(title="Average wind speed binning ")
plt.tight_layout()
plt.show()
#%%
##trying to calculate area under the curve
##calculating area under the curve for total concentration

# def fit_function(x, D):
#     return N0 * np.exp(-x / D) 
# common_bins = np.linspace(2.5, 10, 10)
# # windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]
# windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]
# grouped_concentrations = {i: [] for i in range(len(windspeed_bins))}  # Accumulate droplet concentrations
# mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}
# for i in range(len(filtered_master_BCB_ddry)):
#     entry_ddry = filtered_master_BCB_ddry[i]
#     N0 = entry_ddry['n0']  


#     date = entry_ddry['Date']
#     leg_index = entry_ddry['Leg_index']
#     D = entry_ddry['D'] 
#     ddry_values = np.array(entry_ddry['filtered_ddry'])

   
#     if (date, leg_index) in problematic_set:
#         continue

    
#     windspeed_entry = df_combined[
#         (df_combined['Date'] == date) & 
#         (df_combined['Leg_index'] == leg_index) 
#     ]

#     if not windspeed_entry.empty:
#         windspeed = windspeed_entry['Windspeed'].values[0]

       
#         size_dist = fit_function(ddry_values, n0, D)  
#         interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
#         interpolated_leg_values = interp_func(common_bins)  

        
#         for idx, (low, high) in enumerate(windspeed_bins):
#             if low <= windspeed <= high:
#                 grouped_concentrations[idx].append(interpolated_leg_values)
#                 mean_windspeeds[idx].append(windspeed)
#                 break
# plt.figure(figsize=(12, 8))

# for idx, (low, high) in enumerate(windspeed_bins):
#     if grouped_concentrations[idx]:
#         concentrations_array = np.array(grouped_concentrations[idx])  # Convert to numpy array
        
        
#         avg_concentration = np.mean(concentrations_array, axis=0)
        
        
#         std_dev = np.std(concentrations_array, axis=0)
#         avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)  # Replace zeros
        
        
#         avg_windspeed = np.mean(mean_windspeeds[idx])
#         num_legs = len(grouped_concentrations[idx])

        
#         plt.errorbar(common_bins, avg_concentration, yerr=std_dev, fmt='o', capsize=5, 
#                      label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linestyle='-', alpha=0.7)

        
#         try:
#             popt, pcov = curve_fit(lambda x, D: fit_function(x, n0, D), common_bins, avg_concentration, p0=[1])
#             D_fit = popt[0]

           
#             fitted_values = fit_function(common_bins, n0, D_fit)
#             plt.plot(common_bins, fitted_values, linestyle='--', 
#                      label=f"Fit {avg_windspeed:.1f} m/s: y = {n0:.2f} * exp(-x / {D_fit:.2f})")
            
            
#             total_concentration = np.trapz(fitted_values, common_bins)
#             print(f"Total concentration for windspeed {avg_windspeed:.1f} m/s: {total_concentration:.2f} / cm³")
        
#         except RuntimeError:
#             print(f"Could not fit the exponential model for windspeed {avg_windspeed:.1f} m/s")
# plt.yscale('log')
# plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
# plt.xlabel('Bin diameter (µm)', fontweight='bold')
# plt.title('Average size distribution by wind speed)', fontweight='bold')
# plt.legend(title="Average wind speed (m/s)")
# plt.tight_layout()
# plt.show()

# Function to compute size distribution
def fit_function(x, n0, D):
    return n0 * np.exp(-x / D)

common_bins = np.linspace(2.5, 10, 10)
windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]
grouped_concentrations = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

for i in range(len(filtered_master_BCB_ddry_CDP)):
    entry_ddry = filtered_master_BCB_ddry_CDP[i]
    N0 = entry_ddry['n0']  
    date = entry_ddry['Date']
    leg_index = entry_ddry['Leg_index']
    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])

    
    if (date, leg_index) in problematic_set:
        continue

    
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) & 
        (df_combined['BCB_start'] == entry_ddry['BCB_start']) & 
        (df_combined['BCB_stop'] == entry_ddry['BCB_stop'])
    ]

    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]

       
        size_dist = fit_function(ddry_values, N0, D)
        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_concentrations[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break

plt.figure(figsize=(12, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_concentrations[idx]:
        concentrations_array = np.array(grouped_concentrations[idx])
        avg_concentration = np.mean(concentrations_array, axis=0)
        std_dev = np.std(concentrations_array, axis=0)

        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_concentrations[idx])

        plt.errorbar(common_bins, avg_concentration, yerr=std_dev, fmt='o', capsize=5,
                     label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linestyle='-', alpha=0.7)

        try:
            popt, pcov = curve_fit(lambda x, D: fit_function(x, N0, D), common_bins, avg_concentration, p0=[1])
            D_fit = popt[0]

            fitted_values = fit_function(common_bins, N0, D_fit)
            plt.plot(common_bins, fitted_values, linestyle='--',
                     label=f"Fit {avg_windspeed:.1f} m/s: y = {N0:.2f} * exp(-x / {D_fit:.2f})")

        
            total_concentration = np.trapz(fitted_values, common_bins)
            print(f"Total concentration for windspeed {avg_windspeed:.1f} m/s: {total_concentration:.2f} / cm³")

        except RuntimeError:
            print(f"Could not fit the exponential model for windspeed {avg_windspeed:.1f} m/s")

plt.yscale('log')
plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Average size distribution by wind speed)', fontweight='bold')
plt.legend(title="Average wind speed (m/s)")
plt.tight_layout()
plt.show()
#%%
# Fitting an exponential to averaged curves using a basic exponential

def size_distribution(ddry, N0, D):
    return N0 * np.exp(-ddry / D)

def fit_function(x, dryint, D):
    return dryint * np.exp(-D * x)

common_bins = np.linspace(0, 10, 10)
windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]
grouped_concentrations = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

for i in range(len(filtered_master_BCB_ddry_CDP)):
    entry_ddry = filtered_master_BCB_ddry_CDP[i]
    entry_dryintercept = filtered_master_BCB_dryintercept_CDP[i]

    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']

    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])
    dryint = entry_dryintercept['dry intercept']

    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]

        # Interpolate size distribution
        size_dist = size_distribution(ddry_values, dryint, D)
        if np.any(np.isnan(size_dist)):  # Check for invalid values in size_dist
            continue
        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        if np.any(np.isnan(interpolated_leg_values)) or np.any(np.isinf(interpolated_leg_values)):
            continue  # Skip this leg if interpolation failed

        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_concentrations[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break

# Plotting and fitting
plt.figure(figsize=(12, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_concentrations[idx]:
        concentrations_array = np.array(grouped_concentrations[idx])

        # Average concentration and standard deviation
        avg_concentration = np.mean(concentrations_array, axis=0)
        std_dev = np.std(concentrations_array, axis=0)

        # Replace invalid values
        avg_concentration = np.where(np.isnan(avg_concentration) | np.isinf(avg_concentration), 1e-10, avg_concentration)

        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_concentrations[idx])

        # Plot averaged curve with error bars
        plt.errorbar(common_bins, avg_concentration, yerr=std_dev, fmt='o', capsize=5,
                     label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linestyle='-', alpha=0.7)

        try:
            # Fit exponential model
            popt, pcov = curve_fit(fit_function, common_bins, avg_concentration, p0=[1, 1])
            dryint_fit, D_fit = popt

            # Generate fitted curve
            fitted_values = fit_function(common_bins, dryint_fit, D_fit)
            plt.plot(common_bins, fitted_values, linestyle='--',
                     label=f"Fit {avg_windspeed:.1f} m/s: y = {dryint_fit:.2f} * exp(-x / {D_fit:.2f})")

            print(f"Windspeed {avg_windspeed:.1f} m/s: y = {dryint_fit:.2f} * exp(-x / {D_fit:.2f})")

        except RuntimeError:
            print(f"Could not fit the exponential model for windspeed {avg_windspeed:.1f} m/s")

plt.yscale('log')
plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Average size distribution by wind speed', fontweight='bold')
plt.legend(title="Average wind speed (m/s)")
plt.tight_layout()
plt.show()


#%%
#fitting an exponential to averaged curves using a basic exponential 

# def fit_function(x, dryint, D):
#     return dryint * np.exp(-D*x)

# common_bins = np.linspace(0, 10, 10)
# # windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]
# windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]  
# grouped_concentrations = {i: [] for i in range(len(windspeed_bins))}  # To accumulate droplet concentrations
# mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# for i in range(len(filtered_master_BCB_ddry_CDP)):
#     entry_ddry = filtered_master_BCB_ddry_CDP[i]
#     entry_dryintercept = filtered_master_BCB_dryintercept_CDP[i]

    
#     date = entry_ddry['Date']
#     BCB_start = entry_ddry['BCB_start']
#     BCB_stop = entry_ddry['BCB_stop']
#     leg_index = entry_ddry['Leg_index']

    
#     if (date, leg_index) in problematic_set:
#         continue

#     D = entry_ddry['D']
#     ddry_values = np.array(entry_ddry['filtered_ddry'])
#     dryint = entry_dryintercept['dry intercept']

    
#     windspeed_entry = df_combined[
#         (df_combined['Date'] == date) &
#         (df_combined['BCB_start'] == BCB_start) &
#         (df_combined['BCB_stop'] == BCB_stop)
#     ]
    
#     if not windspeed_entry.empty:
#         windspeed = windspeed_entry['Windspeed'].values[0]

        
#         size_dist = size_distribution(ddry_values, dryint, D)  
#         interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
#         interpolated_leg_values = interp_func(common_bins)  

        
#         for idx, (low, high) in enumerate(windspeed_bins):
#             if low <= windspeed <= high:
#                 grouped_concentrations[idx].append(interpolated_leg_values)
#                 mean_windspeeds[idx].append(windspeed)
#                 break

# plt.figure(figsize=(12, 8))

# for idx, (low, high) in enumerate(windspeed_bins):
#     if grouped_concentrations[idx]:
        
#         concentrations_array = np.array(grouped_concentrations[idx])
        
        
#         avg_concentration = np.mean(concentrations_array, axis=0)
        
        
#         std_dev = np.std(concentrations_array, axis=0)
        
        
#         avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)  # Replace zeros with small value
        
        
#         avg_windspeed = np.mean(mean_windspeeds[idx])
#         num_legs = len(grouped_concentrations[idx])
        
        
#         plt.errorbar(common_bins, avg_concentration, yerr=std_dev, fmt='o', capsize=5, 
#                      label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linestyle='-', alpha=0.7)

        
#         try:
#             popt, pcov = curve_fit(fit_function, common_bins, avg_concentration, p0=[1, 1])
#             dryint_fit, D_fit = popt

            
#             fitted_values = fit_function(common_bins, dryint_fit, D_fit)
#             plt.plot(common_bins, fitted_values, linestyle='--', 
#                      label=f"Fit {avg_windspeed:.1f} m/s: y = {dryint_fit:.2f} * exp(-x / {D_fit:.2f})")
            
            
#             print(f"Windspeed {avg_windspeed:.1f} m/s: y = {dryint_fit:.2f} * exp(-x / {D_fit:.2f})")

#         except RuntimeError:
#             print(f"Could not fit the exponential model for windspeed {avg_windspeed:.1f} m/s")

# plt.yscale('log')
# plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
# plt.xlabel('Bin diameter (µm)', fontweight='bold')
# plt.title('Average size distribution by wind speed', fontweight='bold')
# plt.legend(title="Average wind speed (m/s)")
# plt.tight_layout()
# plt.show()
#%%
##calculating area under the curve for total concentration
def fit_function(x, dryint, D):
    return dryint * np.exp(-x / D)

common_bins = np.linspace(2.5, 10, 10)
windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]
grouped_concentrations = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Loop through legs
for i in range(len(filtered_master_BCB_ddry_CDP)):
    entry_ddry = filtered_master_BCB_ddry_CDP[i]
    entry_dryintercept = filtered_master_BCB_dryintercept_CDP[i]

    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']

    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])
    dryint = entry_dryintercept['dry intercept']

    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]

        # Interpolate size distribution
        size_dist = size_distribution(ddry_values, dryint, D)
        if np.any(np.isnan(size_dist)) or np.any(np.isinf(size_dist)):
            continue  # Skip invalid size distributions

        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        if np.any(np.isnan(interpolated_leg_values)) or np.any(np.isinf(interpolated_leg_values)):
            continue  # Skip invalid interpolated values

        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_concentrations[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break

# Plot and fit
plt.figure(figsize=(12, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_concentrations[idx]:
        concentrations_array = np.array(grouped_concentrations[idx])

        # Average concentration and standard deviation
        avg_concentration = np.mean(concentrations_array, axis=0)
        std_dev = np.std(concentrations_array, axis=0)

        # Clean invalid values
        avg_concentration = np.where(np.isnan(avg_concentration) | np.isinf(avg_concentration), 1e-10, avg_concentration)
        avg_concentration = np.clip(avg_concentration, 1e-10, None)  # Ensure values are small positive numbers

        if np.any(avg_concentration <= 0):  # Verify no invalid values remain
            print(f"Skipping bin {idx} due to invalid concentrations.")
            continue

        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_concentrations[idx])

        # Plot averaged data
        plt.errorbar(common_bins, avg_concentration, yerr=std_dev, fmt='o', capsize=5,
                     label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linestyle='-', alpha=0.7)

        try:
            # Curve fitting
            popt, pcov = curve_fit(fit_function, common_bins, avg_concentration, p0=[np.max(avg_concentration), 1])
            dryint_fit, D_fit = popt

            # Plot fitted curve
            fitted_values = fit_function(common_bins, dryint_fit, D_fit)
            plt.plot(common_bins, fitted_values, linestyle='--',
                     label=f"Fit {avg_windspeed:.1f} m/s: y = {dryint_fit:.2f} * exp(-x / {D_fit:.2f})")

            # Calculate total concentration
            total_concentration = np.trapz(fitted_values, common_bins)
            print(f"Total concentration for {avg_windspeed:.1f} m/s: {total_concentration:.2f} / cm³")

        except RuntimeError:
            print(f"Curve fitting failed for windspeed {avg_windspeed:.1f} m/s")

plt.yscale('log')
plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Average size distribution by wind speed', fontweight='bold')
plt.legend(title="Average wind speed (m/s)")
plt.tight_layout()
plt.show()
#%%

# Function to compute size distribution
def size_distribution(ddry, dryint, D):
    return dryint * np.exp(-ddry / D)

# Fit function
def fit_function(x, dryint, D):
    return dryint * np.exp(-x / D)

common_bins = np.linspace(0, 10, 25)
windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]
grouped_concentrations = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Process each entry
for i in range(len(filtered_master_BCB_ddry_CDP)):
    entry_ddry = filtered_master_BCB_ddry_CDP[i]
    entry_dryintercept = filtered_master_BCB_dryintercept_CDP[i]

    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])
    dryint = entry_dryintercept['dry intercept']

    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]

        # Interpolate size distribution
        size_dist = size_distribution(ddry_values, dryint, D)
        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        if np.any(np.isnan(interpolated_leg_values)) or np.any(np.isinf(interpolated_leg_values)):
            continue  # Skip legs with invalid interpolated values

        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_concentrations[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break

# Plot and fit
plt.figure(figsize=(12, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_concentrations[idx]:
        grouped_conc_array = np.array(grouped_concentrations[idx])

        # Compute average and percentiles
        avg_concentration = np.mean(grouped_conc_array, axis=0)
        perc_25 = np.percentile(grouped_conc_array, 25, axis=0)
        perc_75 = np.percentile(grouped_conc_array, 75, axis=0)

        # Remove invalid values from avg_concentration
        avg_concentration = np.where(
            (np.isnan(avg_concentration)) | (np.isinf(avg_concentration)) | (avg_concentration <= 0),
            1e-10, avg_concentration
        )

        # Check validity before fitting
        if np.any(avg_concentration <= 0) or np.any(np.isnan(avg_concentration)):
            print(f"Skipping bin {idx} due to invalid average concentrations.")
            continue

        # Fit exponential curve
        try:
            popt, _ = curve_fit(fit_function, common_bins, avg_concentration, p0=[1, 1])
            fitted_curve = fit_function(common_bins, *popt)
        except RuntimeError:
            print(f"Fit failed for bin {idx}")
            continue

        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_concentrations[idx])

        # Plot the fitted curve
        plt.plot(common_bins, fitted_curve, label=f"Fit {avg_windspeed:.1f} m/s, n={num_legs} legs")

        # Plot the shaded percentile area
        plt.fill_between(common_bins, perc_25, perc_75, alpha=0.2)

plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.yscale('log')
plt.title('Fitted size distribution by wind speed', fontweight='bold')
plt.legend(title="Wind speed (m/s)")
plt.tight_layout()
plt.show()

# %%


# common_bins = np.linspace(0, 10, 25)
# # windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]
# windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]
# fitted_params = {i: [] for i in range(len(windspeed_bins))}  
# mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# for i in range(len(filtered_master_BCB_ddry)):
#     entry_ddry = filtered_master_BCB_ddry[i]
#     entry_dryintercept = filtered_master_BCB_dryintercept[i]  

   
#     date = entry_ddry['Date']
#     BCB_start = entry_ddry['BCB_start']
#     BCB_stop = entry_ddry['BCB_stop']
#     leg_index = entry_ddry['Leg_index'] 

    
#     if (date, leg_index) in problematic_set:
#         continue

#     D = entry_ddry['D']  
#     ddry_values = np.array(entry_ddry['filtered_ddry'])  
#     dryint = entry_dryintercept['dry intercept']  


#     windspeed_entry = df_combined[
#         (df_combined['Date'] == date) &
#         (df_combined['BCB_start'] == BCB_start) &
#         (df_combined['BCB_stop'] == BCB_stop)
#     ]
    
    
#     if not windspeed_entry.empty:
#         windspeed = windspeed_entry['Windspeed'].values[0]
        
        
#         size_dist = size_distribution(ddry_values, dryint, D)
#         interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
#         interpolated_leg_values = interp_func(common_bins)

        
#         def fit_function(x, dryint, D):
#             return dryint * np.exp(-x / D)

#         try:
#             popt, _ = curve_fit(fit_function, common_bins, interpolated_leg_values, p0=[1, 1])
#             fitted_params_idx = -1

            
#             for idx, (low, high) in enumerate(windspeed_bins):
#                 if low <= windspeed <= high:
#                     fitted_params[idx].append(popt)  
#                     mean_windspeeds[idx].append(windspeed)
#                     fitted_params_idx = idx
#                     break
            
#         except RuntimeError:
#             print(f"Fitting failed for leg {date}, Leg {leg_index}")


# averaged_params = []
# for idx in range(len(windspeed_bins)):
#     if fitted_params[idx]:
#         avg_dryint = np.mean([param[0] for param in fitted_params[idx]])
#         avg_D = np.mean([param[1] for param in fitted_params[idx]])
#         averaged_params.append((avg_dryint, avg_D))


# plt.figure(figsize=(12, 8))


# for idx, (low, high) in enumerate(windspeed_bins):
#     if fitted_params[idx]:
#         avg_dryint, avg_D = averaged_params[idx]
#         fitted_curve = fit_function(common_bins, avg_dryint, avg_D)
        
        
#         avg_windspeed = np.mean(mean_windspeeds[idx])
#         num_legs = len(fitted_params[idx])
        
        
#         plt.plot(common_bins, fitted_curve, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")

# plt.yscale('log') 
# plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
# plt.xlabel('Bin diameter (µm)', fontweight='bold')
# plt.title('Fitted size distribution by wind speed', fontweight='bold')
# plt.legend(title="Wind speed (m/s)")
# plt.tight_layout()
# plt.show()

# %%
##trying third order polynomial to the fitted distributions

def size_distribution(ddry, N0, D):
    return N0 * np.exp(-ddry / D)

def poly_fit(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

# Common bins and windspeed bins
common_bins = np.linspace(0, 10, 25)
windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]
fitted_params = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Process data
for i in range(len(filtered_master_BCB_ddry_CDP)):
    entry_ddry = filtered_master_BCB_ddry_CDP[i]
    entry_dryintercept = filtered_master_BCB_dryintercept_CDP[i]

    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']

    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])
    dryint = entry_dryintercept['dry intercept']

    # Fetch windspeed
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    if windspeed_entry.empty:
        continue
    windspeed = windspeed_entry['Windspeed'].values[0]

    # Interpolate size distribution
    size_dist = size_distribution(ddry_values, dryint, D)
    interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
    interpolated_leg_values = interp_func(common_bins)

    # Remove invalid values (NaN, Inf, or <= 0)
    valid_indices = np.isfinite(interpolated_leg_values) & (interpolated_leg_values > 0)
    common_bins_valid = common_bins[valid_indices]
    interpolated_leg_values_valid = interpolated_leg_values[valid_indices]

    if len(common_bins_valid) < 4:  # Ensure enough valid points for fitting
        print(f"Skipping leg {date}, {leg_index} due to insufficient valid points.")
        continue

    # Log-transform and fit polynomial
    try:
        log_values = np.log(interpolated_leg_values_valid)
        popt, _ = curve_fit(poly_fit, common_bins_valid, log_values)

        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                fitted_params[idx].append(popt)
                mean_windspeeds[idx].append(windspeed)
                break
    except RuntimeError:
        print(f"Fitting failed for leg {date}, Leg {leg_index}")

# Average and plot results
plt.figure(figsize=(12, 8))
for idx, (low, high) in enumerate(windspeed_bins):
    if fitted_params[idx]:
        avg_params = np.mean(fitted_params[idx], axis=0)
        fitted_curve_log = poly_fit(common_bins, *avg_params)
        fitted_curve = np.exp(fitted_curve_log)

        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(fitted_params[idx])
        plt.plot(common_bins, fitted_curve, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")

plt.yscale('log')
plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Fitted size distribution by wind speed', fontweight='bold')
plt.legend(title="Wind speed (m/s)")
plt.tight_layout()
plt.show()

#%%
# Exclude dynamically detected problematic legs
all_problematic_legs = problematic_set.union(detected_problematic_legs)

common_bins = np.linspace(0, 10, 25)
windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]
fitted_params = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

def polynomial_fit(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def calculate_r_squared(y_actual, y_pred):
    residual_sum_of_squares = np.sum((y_actual - y_pred) ** 2)
    total_sum_of_squares = np.sum((y_actual - np.mean(y_actual)) ** 2)
    r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)
    return r_squared

for i in range(len(filtered_master_BCB_ddry_CDP)):
    entry_ddry = filtered_master_BCB_ddry_CDP[i]
    entry_dryintercept = filtered_master_BCB_dryintercept_CDP[i]  

    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']  

    # Skip legs in the problematic set
    if (date, leg_index) in all_problematic_legs:
        continue

    D = entry_ddry['D']  
    ddry_values = np.array(entry_ddry['filtered_ddry']) 
    dryint = entry_dryintercept['dry intercept'] 

    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]
        
        size_dist = size_distribution(ddry_values, dryint, D)
        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        # Check for NaN or Inf in interpolated values
        if np.any(np.isnan(interpolated_leg_values)) or np.any(np.isinf(interpolated_leg_values)):
            print(f"Skipping leg {date}, Leg {leg_index} due to NaN or Inf in interpolated values")
            continue

        # Fit a cubic polynomial to the log of the interpolated values
        try:
            log_values = np.log(interpolated_leg_values + 1e-10)  
            popt, _ = curve_fit(polynomial_fit, common_bins, log_values)

            for idx, (low, high) in enumerate(windspeed_bins):
                if low <= windspeed <= high:
                    fitted_params[idx].append(popt) 
                    mean_windspeeds[idx].append(windspeed)
                    break
            
        except RuntimeError:
            print(f"Fitting failed for leg {date}, Leg {leg_index}")

averaged_params = []
for idx in range(len(windspeed_bins)):
    if fitted_params[idx]:
        avg_params = np.mean(fitted_params[idx], axis=0) 
        averaged_params.append(avg_params)

plt.figure(figsize=(12, 8))

# Loop through each windspeed bin and plot the fitted average distributions
for idx, (low, high) in enumerate(windspeed_bins):
    if fitted_params[idx]:
        avg_params = averaged_params[idx]
        fitted_curve_log = polynomial_fit(common_bins, *avg_params)  # Get the fitted values in log space
        fitted_curve = np.exp(fitted_curve_log)  # Convert back to original space
        
        # Calculate average windspeed and number of legs in this bin
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(fitted_params[idx])
        
        # Plot the fitted curve in log scale
        plt.plot(common_bins, fitted_curve, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")

plt.yscale('log')
plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Fitted size distribution by wind speed', fontweight='bold')
plt.legend(title="wind speed (m/s)")
plt.tight_layout()
plt.show()
#%%
#third order with 0.90 specification

# Size distribution function
def size_distribution(ddry, N0, D):
    return N0 * np.exp(-ddry / D)

# Polynomial fit
def polynomial_fit(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

# Bin configuration
common_bins = np.linspace(0, 10, 25)
windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]
averaged_distributions = {i: [] for i in range(len(windspeed_bins))}

# Iterate through all legs
for i in range(len(filtered_master_BCB_ddry_CDP)):
    entry_ddry = filtered_master_BCB_ddry_CDP[i]
    entry_dryintercept = filtered_master_BCB_dryintercept_CDP[i]

    # Extract data
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])
    dryint = entry_dryintercept['dry intercept']

    # Match windspeed
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    if windspeed_entry.empty:
        continue
    windspeed = windspeed_entry['Windspeed'].values[0]

    # Interpolate the size distribution
    try:
        size_dist = size_distribution(ddry_values, dryint, D)
        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)
    except Exception as e:
        print(f"Interpolation failed for leg {date}, {leg_index}: {e}")
        continue

    # Filter out invalid interpolated values
    if np.any(~np.isfinite(interpolated_leg_values)) or np.any(interpolated_leg_values <= 0):
        print(f"Invalid values found in interpolated data for leg {date}, {leg_index}")
        continue

    # Add to appropriate windspeed bin
    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed <= high:
            averaged_distributions[idx].append(interpolated_leg_values)
            break

# Average distributions and fit
plt.figure(figsize=(12, 8))
for idx, (low, high) in enumerate(windspeed_bins):
    if averaged_distributions[idx]:
        avg_distribution = np.mean(averaged_distributions[idx], axis=0)

        # Further validate averaged data
        if np.any(~np.isfinite(avg_distribution)) or np.any(avg_distribution <= 0):
            print(f"Invalid averaged values in windspeed bin {low}-{high} m/s")
            continue

        try:
            popt, _ = curve_fit(polynomial_fit, common_bins, avg_distribution, p0=[1, 1, 1, 1])
            fitted_curve = polynomial_fit(common_bins, *popt)

            plt.plot(common_bins, fitted_curve, label=f"Windspeed {low}-{high} m/s")
        except RuntimeError as e:
            print(f"Curve fitting failed for windspeed bin {low}-{high} m/s: {e}")

plt.yscale('log')
plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Fitted size distribution by wind speed', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()

#%%
#%%
from numpy.polynomial import Polynomial

windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)
]

fitted_params = [[] for _ in range(len(windspeed_bins))]
mean_windspeeds = [[] for _ in range(len(windspeed_bins))]


for i in range(len(filtered_master_BCB_ddry_CDP)):
    entry_ddry = filtered_master_BCB_ddry_CDP[i]
    entry_dryintercept = filtered_master_BCB_dryintercept_CDP[i] 


    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']  

   
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']  
    ddry_values = np.array(entry_ddry['filtered_ddry']) 
    dryint = entry_dryintercept['dry intercept']  

    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    
    
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]
        
        
        size_dist = size_distribution(ddry_values, dryint, D)
        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        
        order = 4  
        poly_coeffs = Polynomial.fit(common_bins, interpolated_leg_values, order)

       
        fitted_curve = poly_coeffs(common_bins)

       
        residuals = interpolated_leg_values - fitted_curve
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((interpolated_leg_values - np.mean(interpolated_leg_values))**2)
        r_squared = 1 - (ss_res / ss_tot)

        
        if r_squared > 0.90:
            fitted_params_idx = -1

    
            for idx, (low, high) in enumerate(windspeed_bins):
                if low <= windspeed <= high:
                    fitted_params[idx].append((poly_coeffs.coef[0], poly_coeffs.coef[1]))  # Store coefficients
                    mean_windspeeds[idx].append(windspeed)
                    fitted_params_idx = idx
                    break

        else:
            print(f"R² value for windspeed bin is below threshold: {r_squared:.2f}")


averaged_params = []
for idx in range(len(windspeed_bins)):
    if fitted_params[idx]:
        avg_dryint = np.mean([param[0] for param in fitted_params[idx]])
        averaged_params.append(avg_dryint)


plt.figure(figsize=(12, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if fitted_params[idx]:
        avg_dryint = averaged_params[idx]
        
        
        fitted_curve = Polynomial([avg_dryint, 0])(common_bins)  

        
        plt.plot(common_bins, fitted_curve, label=f"Windspeed {low}-{high} m/s")

plt.yscale('log') 
plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Fitted size distribution by wind speed', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()
#%%
from numpy.polynomial import Polynomial
windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]


fitted_params = [[] for _ in range(len(windspeed_bins))]
mean_windspeeds = [[] for _ in range(len(windspeed_bins))]

for i in range(len(filtered_master_BCB_ddry_CDP)):
    entry_ddry = filtered_master_BCB_ddry_CDP[i]
    entry_dryintercept = filtered_master_BCB_dryintercept_CDP[i]

    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']

   
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])
    dryint = entry_dryintercept['dry intercept']

    
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]

        
        size_dist = size_distribution(ddry_values, dryint, D)
        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

       
        if np.any(np.isnan(interpolated_leg_values)) or np.any(np.isinf(interpolated_leg_values)):
            print(f"Skipping leg {date}, Leg {leg_index} due to NaN or Inf in interpolated values")
            continue

        
        try:
            order = 4
            poly_coeffs = Polynomial.fit(common_bins, interpolated_leg_values, order)
            fitted_curve = poly_coeffs(common_bins)

            
            residuals = interpolated_leg_values - fitted_curve
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((interpolated_leg_values - np.mean(interpolated_leg_values))**2)
            r_squared = 1 - (ss_res / ss_tot)

            if r_squared > 0.90:
                for idx, (low, high) in enumerate(windspeed_bins):
                    if low <= windspeed <= high:
                        fitted_params[idx].append((poly_coeffs.coef[0], poly_coeffs.coef[1]))  # Store coefficients
                        mean_windspeeds[idx].append(windspeed)
                        break
            else:
                print(f"R² value below threshold for leg {date}, Leg {leg_index}: {r_squared:.2f}")

        except Exception as e:
            print(f"Polynomial fitting failed for leg {date}, Leg {leg_index}: {e}")

averaged_params = []
for idx in range(len(windspeed_bins)):
    if fitted_params[idx]:
        avg_dryint = np.mean([param[0] for param in fitted_params[idx]])
        averaged_params.append(avg_dryint)

plt.figure(figsize=(12, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if fitted_params[idx]:
        avg_dryint = averaged_params[idx]
        fitted_curve = Polynomial([avg_dryint, 0])(common_bins)  # Generate fitted curve
        plt.plot(common_bins, fitted_curve, label=f"Windspeed {low}-{high} m/s")

plt.yscale('log')
plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Fitted size distribution by wind speed', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()

#%%

def exponential_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

for idx in range(len(windspeed_bins)):
    if averaged_distributions[idx]:  
        avg_distribution = np.mean(averaged_distributions[idx], axis=0)
        
    
        try:
            popt, _ = curve_fit(exponential_decay, common_bins, avg_distribution, p0=[1, 1, 1])
            fitted_curve = exponential_decay(common_bins, *popt)
            
            plt.plot(common_bins, fitted_curve, label=f"Exponential Fit for Windspeed {windspeed_bins[idx]} m/s")
        
        except RuntimeError:
            print(f"Fitting failed for windspeed bin {idx}")

plt.yscale('log')
plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Fitted size distribution by wind speed', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()

#%%

common_bins = np.linspace(0.1, 10, 25)


def polynomial_fit(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e


# windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]
windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]  
fitted_params = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}


for i in range(len(filtered_master_BCB_ddry_CDP)):
    entry_ddry = filtered_master_BCB_ddry_CDP[i]
    entry_dryintercept = filtered_master_BCB_dryintercept_CDP[i]

    
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']

   
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])
    dryint = entry_dryintercept['dry intercept']

    
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    
    
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]
        
       
        size_dist = size_distribution(ddry_values, dryint, D)
        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        
        if np.any(np.isnan(interpolated_leg_values)) or np.any(np.isinf(interpolated_leg_values)):
            print(f"Interpolated values contain NaNs or Infs for leg {date}, Leg {leg_index}")
            continue

    
        interpolated_leg_values[interpolated_leg_values <= 0] = np.nan

        
        if np.isnan(interpolated_leg_values).all():
            print(f"All interpolated values became NaN for leg {date}, Leg {leg_index}")
            continue

        
        try:
            
            log_values = np.log(np.nan_to_num(interpolated_leg_values, nan=1e-10))  # Replace NaNs with small values
            popt, _ = curve_fit(polynomial_fit, common_bins, log_values, p0=[1, 1, 1, 1, 1])
            
            
            for idx, (low, high) in enumerate(windspeed_bins):
                if low <= windspeed <= high:
                    fitted_params[idx].append(popt)
                    mean_windspeeds[idx].append(windspeed)
                    break

        except RuntimeError:
            print(f"Fitting failed for leg {date}, Leg {leg_index}")


averaged_params = []
for idx in range(len(windspeed_bins)):
    if fitted_params[idx]:
        avg_params = np.mean(fitted_params[idx], axis=0)
        averaged_params.append(avg_params)

plt.figure(figsize=(12, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if fitted_params[idx]:
        avg_params = averaged_params[idx]
        fitted_curve = polynomial_fit(common_bins, *avg_params)

        
        plt.plot(common_bins, np.exp(fitted_curve), label=f"Windspeed {low}-{high} m/s")


plt.yscale('log')  
plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Fitted size distribution by wind speed', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()


# %%
#trying a fifth order polynomial 

def polynomial_fit(x, a, b, c, d, e, f):
    return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f

windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]
# windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]
fitted_params = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

for i in range(len(filtered_master_BCB_ddry_CDP)):
    entry_ddry = filtered_master_BCB_ddry_CDP[i]
    entry_dryintercept = filtered_master_BCB_dryintercept_CDP[i]

   
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']

    
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])
    dryint = entry_dryintercept['dry intercept']

   
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    
    
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]
        
        
        size_dist = size_distribution(ddry_values, dryint, D)
        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        
        if np.any(np.isnan(interpolated_leg_values)) or np.any(np.isinf(interpolated_leg_values)):
            print(f"Interpolated values contain NaNs or Infs for leg {date}, Leg {leg_index}")
            continue

       
        try:
            
            if np.any(interpolated_leg_values <= 0):
                print(f"Non-positive interpolated values found for leg {date}, Leg {leg_index}: {interpolated_leg_values}")
                continue

            
            log_values = np.log(interpolated_leg_values + 1e-10)  
            
            
            if np.any(np.isnan(log_values)) or np.any(np.isinf(log_values)):
                print(f"Log-transformed values contain NaNs or Infs for leg {date}, Leg {leg_index}")
                continue

            
            popt, _ = curve_fit(polynomial_fit, common_bins, log_values, p0=[1, 1, 1, 1, 1, 1])

            fitted_params_idx = -1

            
            for idx, (low, high) in enumerate(windspeed_bins):
                if low <= windspeed <= high:
                    fitted_params[idx].append(popt)
                    mean_windspeeds[idx].append(windspeed)
                    fitted_params_idx = idx
                    break

        except RuntimeError:
            print(f"Fitting failed for leg {date}, Leg {leg_index}")


averaged_params = []
for idx in range(len(windspeed_bins)):
    if fitted_params[idx]:
        avg_params = np.mean(fitted_params[idx], axis=0)
        averaged_params.append(avg_params)

plt.figure(figsize=(12, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if fitted_params[idx]:
        avg_params = averaged_params[idx]
        fitted_curve = polynomial_fit(common_bins, *avg_params)

        
        plt.plot(common_bins, np.exp(fitted_curve), label=f"Windspeed {low}-{high} m/s")

plt.yscale('log')  
plt.ylim(10**-3, 10**0.3)
plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Fitted size distribution by wind speed', fontweight='bold')
plt.legend()
plt.show()


# %%
##Trying to look at mass so I can better understand the phase space. 

#I am first doing to try and look at mass, before fitting an exponential to 
#the distribution, and before binning by windspeed. 

# Key Concepts
# Mass Calculation: Mass for each diameter bin is calculated as:

#m(d)=N(d)⋅ρ⋅6π​d^3

#Cumulative Mass:Compute the cumulative sum of mass across bins:
#Mcumulative (d)= d′≤d∑m(d′) 
#Normalize by the total mass 
#𝑀total:𝑀normalized(𝑑)=𝑀cumulative(𝑑)/Mtotal(d)

#Finding the Diameter at 50% Mass:
#Use linear interpolation on the normalized cumulative mass curve to find the diameter where 
#Mnormalized=0.5. This diameter is the D50 value.

#%%
# Function to calculate mass with lower limit of integration at 2 and plot mass contours with windspeed
def calculate_mass(N0, D):
    integrand = lambda d: np.exp(-d / D) * d**3
    mass_integral, _ = quad(integrand, 2, np.inf)
    return N0 * mass_integral


filtered_combined_clean = filtered_combined[(filtered_combined['D'] > 0) & 
                                            (filtered_combined['dryintercept'] > 0)].copy()

filtered_combined_clean['Mass'] = filtered_combined_clean.apply(
    lambda row: calculate_mass(row['dryintercept'], row['D']), axis=1)

print(f"Min mass: {filtered_combined_clean['Mass'].min()}, Max mass: {filtered_combined_clean['Mass'].max()}")

xgrid = np.logspace(np.log10(filtered_combined_clean['D'].min()), 
                    np.log10(filtered_combined_clean['D'].max()), 100)
ygrid = np.logspace(np.log10(filtered_combined_clean['dryintercept'].min()), 
                    np.log10(filtered_combined_clean['dryintercept'].max()), 100)
D_grid, dryintercept_grid = np.meshgrid(xgrid, ygrid)

vectorized_mass = np.vectorize(calculate_mass)
mass_grid = vectorized_mass(dryintercept_grid, D_grid)

print(f"Mass grid: Min = {np.nanmin(mass_grid)}, Max = {np.nanmax(mass_grid)}")

low_levels = np.logspace(np.log10(filtered_combined_clean['Mass'].min()), np.log10(1), 6, endpoint=False)  # 3 contours in low range (up to 10^0)
high_levels = np.logspace(np.log10(1), np.log10(filtered_combined_clean['Mass'].max()), 5)  # More contours in higher range (10^0 to max)

mass_levels = np.concatenate([low_levels, high_levels])

print(f"Mass levels: {mass_levels}")


sc = plt.scatter(filtered_combined_clean['D'], filtered_combined_clean['dryintercept'], 
                 c=filtered_combined_clean['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

contour_plot = plt.contour(D_grid, dryintercept_grid, mass_grid, levels=mass_levels, 
                           colors='red', linewidths=0.5, alpha=0.75)

if len(contour_plot.allsegs[0]) == 0:
    print("No contours were created. Check your data range or mass grid calculation.")


plt.colorbar(sc, label='10m Wind speed (m/s)')


plt.xlabel('D', fontsize=14, fontweight='bold')
plt.ylabel('Dry intercept', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.title('Mass contours', fontsize=14, fontweight='bold')
plt.xlim(10**-2, 10**2)
plt.ylim(10**-2, 10**2.5)
plt.show()
#%%
# Make Y_BCB_calc which is 476 in length match with filtered_master_BCB_grH
#which has 460 entries. 

grh_leg_set = set(
    (entry["Date"], entry["BCB_start"], entry["BCB_stop"])
    for sublist in filtered_master_BCB_gRH for entry in sublist
)
filtered_Y_BCB_calc = [
    entry for entry in Y_BCB_calc
    if (entry["Date"], entry["BCB_start"], entry["BCB_stop"]) in grh_leg_set
]


print(f"Total entries in Y_BCB_calc: {len(Y_BCB_calc)}")
print(f"Total entries in filtered_master_BCB_gRH: {sum(len(sublist) for sublist in filtered_master_BCB_gRH)}")  # Adjusted for nested lists
print(f"Total entries in filtered_Y_BCB_calc: {len(filtered_Y_BCB_calc)}")


assert len(filtered_Y_BCB_calc) == sum(len(sublist) for sublist in filtered_master_BCB_gRH), "Filtered list length does not match grh list!"

#%%
Y_BCB_df = pd.DataFrame(filtered_Y_BCB_calc)
Y_BCB_df["Leg_index"] = Y_BCB_df.groupby("Date").cumcount()
print(f"Sample data with Leg_index:\n{Y_BCB_df[['Date', 'Leg_index', 'BCB_start', 'BCB_stop']].head(20)}")
#%%

Y_BCB_clean = Y_BCB_df[
    ~Y_BCB_df[["Date", "Leg_index"]].apply(tuple, axis=1).isin(problematic_set)
].copy()

Y_BCB_clean.fillna(0, inplace=True)
#%%

common_bins = np.array(filtered_master_BCB_ddry[0]["filtered_ddry"])  # Raw bin diameters
def calculate_mass(diameter, concentration):
    return concentration * (np.pi / 6) * (diameter**3)
def calculate_cumulative_mass(diameters, masses):
    cumulative_mass = np.cumsum(masses)
    total_mass = np.sum(masses)
    normalized_cumulative_mass = cumulative_mass / total_mass

    interpolation_func = interp1d(normalized_cumulative_mass, diameters, kind="linear", fill_value="extrapolate")
    diameter_at_50_percent = interpolation_func(0.5)
    return normalized_cumulative_mass, diameter_at_50_percent

plt.figure(figsize=(12, 8))
diameters_at_50_mass = [] 


for i, row in Y_BCB_clean.iterrows():
   
    concentrations = np.array([row.get(f"Bin{j}_Y_mean", 0) for j in range(12, 30)])  # Replace NaN with 0

    
    masses = calculate_mass(common_bins, concentrations)

    
    normalized_cumulative_mass, diameter_at_50 = calculate_cumulative_mass(common_bins, masses)
    diameters_at_50_mass.append(diameter_at_50)

    
    plt.plot(common_bins, normalized_cumulative_mass, label=f"Leg {row['Leg_index']}, Date {row['Date']}")

plt.xlabel("Diameter (µm)", fontsize=14, fontweight="bold")
plt.ylabel("Cumulative mass (Normalized)", fontsize=14, fontweight="bold")
plt.title("Cumulative mass distribution (Raw Data)", fontsize=14, fontweight="bold")
plt.grid()
plt.tight_layout()
plt.show()

print("Diameters at 50% Mass for Each Leg:")
for i, diameter in enumerate(diameters_at_50_mass, 1):
    print(f"Leg {i}: {diameter:.2f} µm")
#%%
#average with just one line
all_masses = []
for _, row in Y_BCB_clean.iterrows():
    
    concentrations = np.array([row.get(f"Bin{j}_Y_mean", 0) for j in range(12, 30)])  # Replace NaN with 0

    
    masses = calculate_mass(common_bins, concentrations)
    all_masses.append(masses)
all_masses = np.array(all_masses)

average_masses = np.mean(all_masses, axis=0)
normalized_cumulative_mass, diameter_at_50 = calculate_cumulative_mass(common_bins, average_masses)
plt.figure(figsize=(12, 8))
plt.plot(common_bins, normalized_cumulative_mass, label=f"Average Distribution (50% Mass at {diameter_at_50:.2f} µm)", color='blue')
plt.axvline(diameter_at_50, color='red', linestyle='--', alpha=0.7, label=f"50% Mass at {diameter_at_50:.2f} µm")
plt.xlabel("Diameter (µm)", fontsize=14, fontweight="bold")
plt.ylabel("Cumulative mass (Normalized)", fontsize=14, fontweight="bold")
plt.title("Average cumulative mass distribution (Raw Data)", fontsize=14, fontweight="bold")
plt.grid()
plt.tight_layout()
plt.legend(fontsize=10, loc="upper left")
plt.show()
print(f"Diameter at 50% Mass (Average Distribution): {diameter_at_50:.2f} µm")


# %%

##Function to calculate the 50% mass diameter for fitted distribution using 
#slope and dry-intercept values for every leg pre-windspeed binning. 

def cumulative_mass(N0, D, d_max):
   
    integrand = lambda d: N0 * np.exp(-d / D) * d**3
    cumulative_mass, _ = quad(integrand, 2, d_max) 
    return cumulative_mass

def total_mass(N0, D):
    integrand = lambda d: N0 * np.exp(-d / D) * d**3
    total_mass, _ = quad(integrand, 2, np.inf)
    return total_mass

diameters = np.linspace(2, 50, 100) 

diameters_at_50_mass = []
plt.figure(figsize=(12, 8))

for i, row in filtered_combined_clean.iterrows():
    N0 = row['dryintercept']
    D = row['D']

    
    M_total = total_mass(N0, D)
    cumulative_masses = [cumulative_mass(N0, D, d) for d in diameters]

    
    normalized_cumulative_mass = np.array(cumulative_masses) / M_total


    interpolation_func = interp1d(normalized_cumulative_mass, diameters, kind='linear', fill_value='extrapolate')
    diameter_at_50 = interpolation_func(0.5)  
    diameters_at_50_mass.append(diameter_at_50)
    plt.plot(diameters, normalized_cumulative_mass, label=f"Leg {i}")
    plt.axvline(diameter_at_50, color='red', linestyle='--', alpha=0.5, 
                label=f"50% Mass at {diameter_at_50:.2f} µm (Leg {i})")

plt.xlabel('Diameter (µm)', fontsize=14, fontweight='bold')
plt.ylabel('Cumulative mass (Normalized)', fontsize=14, fontweight='bold')
plt.title('Cumulative mass distribution (50% Mass)', fontsize=14, fontweight='bold')
plt.grid()
plt.tight_layout()
plt.show()

print("Diameters at 50% Mass for Each Leg:")
for i, diameter in enumerate(diameters_at_50_mass, 1):
    print(f"Leg {i}: {diameter:.2f} µm")

# %%

##Function to calculate the 50% mass diameter for fitted distribution using 
#slope and dry-intercept values for average or cumulative pre-windspeed binning.

def cumulative_mass(N0, D, d_max):

    integrand = lambda d: N0 * np.exp(-d / D) * d**3
    cumulative_mass, _ = quad(integrand, 2, d_max)  # Integrate from 2 µm to d_max
    return cumulative_mass

def total_mass(N0, D):
    integrand = lambda d: N0 * np.exp(-d / D) * d**3
    total_mass, _ = quad(integrand, 2, np.inf)  
    return total_mass


diameters = np.linspace(2, 50, 100) 

cumulative_mass_all_legs = np.zeros_like(diameters)

for i, row in filtered_combined_clean.iterrows():
    N0 = row['dryintercept']
    D = row['D']

    
    M_total = total_mass(N0, D)

    
    cumulative_masses = np.array([cumulative_mass(N0, D, d) for d in diameters]) / M_total

   
    cumulative_mass_all_legs += cumulative_masses


average_cumulative_mass = cumulative_mass_all_legs / len(filtered_combined_clean)


interpolation_func = interp1d(average_cumulative_mass, diameters, kind='linear', fill_value='extrapolate')
diameter_at_50 = interpolation_func(0.5)  


plt.figure(figsize=(12, 8))
plt.plot(diameters, average_cumulative_mass, label='Average Cumulative Mass')
plt.axvline(diameter_at_50, color='red', linestyle='--', alpha=0.7, 
            label=f"50% Mass at {diameter_at_50:.2f} µm")
plt.xlabel('Diameter (µm)', fontsize=14, fontweight='bold')
plt.ylabel('Cumulative Mass (Normalized)', fontsize=14, fontweight='bold')
plt.title('Average Cumulative Mass Distribution (50% Mass)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid()
plt.tight_layout()
plt.show()
print(f"Diameter at 50% Average Cumulative Mass: {diameter_at_50:.2f} µm")

# %%
##Function to calculate the 50% mass diameter for fitted distribution using 
#slope and dry-intercept values for every leg post-windspeed binning both AND average. 



def cumulative_mass(N0, D, d_max):
    integrand = lambda d: N0 * np.exp(-d / D) * d**3
    cumulative_mass, _ = quad(integrand, 2, d_max)  
    return cumulative_mass


def total_mass(N0, D):
    integrand = lambda d: N0 * np.exp(-d / D) * d**3
    total_mass, _ = quad(integrand, 2, np.inf) 
    return total_mass


diameters = np.linspace(2, 50, 100)  


windspeed_bins = {  
    '0-3 m/s': [],
    '3.001-6 m/s': [],
    '6.001-8 m/s': [],
    '8.001+ m/s': []
}


for i, row in filtered_combined_clean.iterrows():
    N0 = row['dryintercept']
    D = row['D']
    windspeed = row['Windspeed']
    
   
    if 0 <= windspeed <= 3.0:
        windspeed_bins['0-3 m/s'].append((N0, D))
    elif 3.001 <= windspeed <= 6.0:
        windspeed_bins['3.001-6 m/s'].append((N0, D))
    elif 6.001 <= windspeed <= 8.0:
        windspeed_bins['6.001-8 m/s'].append((N0, D))
    elif windspeed > 8.001:
        windspeed_bins['8.001+ m/s'].append((N0, D))


plt.figure(figsize=(12, 8))
diameters_at_50_mass = []  
for bin_label, legs in windspeed_bins.items():
    if not legs:
        continue
    
   
    bin_cumulative_mass = []
    for N0, D in legs:
       
        cumulative_masses = [cumulative_mass(N0, D, d) for d in diameters]
        M_total = total_mass(N0, D)  
        normalized_cumulative_mass = np.array(cumulative_masses) / M_total

       
        plt.plot(diameters, normalized_cumulative_mass, label=f"{bin_label}")

        
        interpolation_func = interp1d(normalized_cumulative_mass, diameters, kind='linear', fill_value='extrapolate')
        diameter_at_50 = interpolation_func(0.5)
        diameters_at_50_mass.append(diameter_at_50)
        plt.axvline(diameter_at_50, color='red', linestyle='--', alpha=0.5, label=f"50% Mass at {diameter_at_50:.2f} µm ({bin_label})")


plt.xlabel('Diameter (µm)', fontsize=14, fontweight='bold')
plt.ylabel('Cumulative mass (Normalized)', fontsize=14, fontweight='bold')
plt.title('Cumulative mass distribution by wind speed bin', fontsize=14, fontweight='bold')

plt.grid()
plt.tight_layout()
plt.show()


average_cumulative_mass = np.zeros_like(diameters)
total_legs = sum(len(legs) for legs in windspeed_bins.values())
for bin_label, legs in windspeed_bins.items():
    for N0, D in legs:
        
        cumulative_masses = [cumulative_mass(N0, D, d) for d in diameters]
        average_cumulative_mass += np.array(cumulative_masses)
average_cumulative_mass /= total_legs
M_total_average = total_mass(np.mean([N0 for legs in windspeed_bins.values() for N0, D in legs]),
                             np.mean([D for legs in windspeed_bins.values() for N0, D in legs]))
average_cumulative_mass /= M_total_average

interpolation_func_avg = interp1d(average_cumulative_mass, diameters, kind='linear', fill_value='extrapolate')
diameter_at_50_avg = interpolation_func_avg(0.5)
plt.figure(figsize=(12, 8))
plt.plot(diameters, average_cumulative_mass, label='Average Cumulative Mass')
plt.axvline(diameter_at_50_avg, color='red', linestyle='--', alpha=0.5, label=f"50% Mass at {diameter_at_50_avg:.2f} µm")

plt.xlabel('Diameter (µm)', fontsize=14, fontweight='bold')
plt.ylabel('Cumulative mass (Normalized)', fontsize=14, fontweight='bold')
plt.title('Average cumulative mass distribution', fontsize=14, fontweight='bold')
plt.grid()
plt.tight_layout()
plt.show()

# %%


# Function to calculate cumulative mass
def cumulative_mass(N0, D, d_max):
    integrand = lambda d: N0 * np.exp(-d / D) * d**3
    cumulative_mass, _ = quad(integrand, 2, d_max)  # Integration limit set to 2 to d_max
    return cumulative_mass

# Function to calculate total mass
def total_mass(N0, D):
    integrand = lambda d: N0 * np.exp(-d / D) * d**3
    total_mass, _ = quad(integrand, 2, 500)  # Replace np.inf with a large value like 500 for stability
    return total_mass

# Range of diameters
diameters = np.linspace(2, 50, 100)

# Windspeed bins
windspeed_bins = {
    '0-3 m/s': [],
    '3.001-6 m/s': [],
    '6.001-8 m/s': [],
    '8.001+ m/s': []
}

# Organize legs into windspeed bins
for i, row in filtered_combined_clean.iterrows():
    N0 = row['dryintercept']
    D = row['D']
    windspeed = row['Windspeed']

    if 0 <= windspeed <= 3.0:
        windspeed_bins['0-3 m/s'].append((N0, D))
    elif 3.001 <= windspeed <= 6.0:
        windspeed_bins['3.001-6 m/s'].append((N0, D))
    elif 6.001 <= windspeed <= 8.0:
        windspeed_bins['6.001-8 m/s'].append((N0, D))
    elif windspeed > 8.001:
        windspeed_bins['8.001+ m/s'].append((N0, D))

# Plot cumulative mass distributions for each windspeed bin
plt.figure(figsize=(12, 8))
diameters_at_50_mass = []

for bin_label, legs in windspeed_bins.items():
    if not legs:
        continue

    for N0, D in legs:
        # Calculate cumulative masses and normalize
        cumulative_masses = [cumulative_mass(N0, D, d) for d in diameters]
        M_total = total_mass(N0, D)
        normalized_cumulative_mass = np.array(cumulative_masses) / M_total
        normalized_cumulative_mass = np.clip(normalized_cumulative_mass, 0, 1)  # Ensure values are within [0, 1]

        # Plot cumulative mass curve
        plt.plot(diameters, normalized_cumulative_mass, label=f"{bin_label}")

        # Find 50% mass diameter
        interpolation_func = interp1d(normalized_cumulative_mass, diameters, kind='linear', fill_value='extrapolate')
        diameter_at_50 = interpolation_func(0.5)
        diameters_at_50_mass.append(diameter_at_50)
        plt.axvline(diameter_at_50, color='red', linestyle='--', alpha=0.5, 
                    label=f"50% Mass at {diameter_at_50:.2f} µm ({bin_label})")

plt.xlabel('Diameter (µm)', fontsize=14, fontweight='bold')
plt.ylabel('Cumulative mass (Normalized)', fontsize=14, fontweight='bold')
plt.title('Cumulative mass distribution by wind speed bin', fontsize=14, fontweight='bold')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Calculate and plot average cumulative mass distribution
average_cumulative_mass = np.zeros_like(diameters)
total_legs = sum(len(legs) for legs in windspeed_bins.values())

for bin_label, legs in windspeed_bins.items():
    for N0, D in legs:
        cumulative_masses = [cumulative_mass(N0, D, d) for d in diameters]
        average_cumulative_mass += np.array(cumulative_masses)

average_cumulative_mass /= total_legs
M_total_average = total_mass(
    np.mean([N0 for legs in windspeed_bins.values() for N0, D in legs]),
    np.mean([D for legs in windspeed_bins.values() for N0, D in legs])
)
average_cumulative_mass /= M_total_average
average_cumulative_mass = np.clip(average_cumulative_mass, 0, 1)  # Ensure values are within [0, 1]

# Interpolation for 50% mass diameter in average distribution
interpolation_func_avg = interp1d(average_cumulative_mass, diameters, kind='linear', fill_value='extrapolate')
diameter_at_50_avg = interpolation_func_avg(0.5)

# Plot average cumulative mass distribution
plt.figure(figsize=(12, 8))
plt.plot(diameters, average_cumulative_mass, label='Average Cumulative Mass')
plt.axvline(diameter_at_50_avg, color='red', linestyle='--', alpha=0.5, label=f"50% Mass at {diameter_at_50_avg:.2f} µm")
plt.xlabel('Diameter (µm)', fontsize=14, fontweight='bold')
plt.ylabel('Cumulative mass (Normalized)', fontsize=14, fontweight='bold')
plt.title('Average cumulative mass distribution', fontsize=14, fontweight='bold')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# %%
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Function to calculate cumulative mass
def cumulative_mass(N0, D, d_max):
    integrand = lambda d: N0 * np.exp(-d / D) * d**3
    cumulative_mass, _ = quad(integrand, 2, d_max)
    return cumulative_mass

# Function to calculate total mass
def total_mass(N0, D):
    integrand = lambda d: N0 * np.exp(-d / D) * d**3
    total_mass, _ = quad(integrand, 2, 500)  # Large finite bound
    return total_mass

# Range of diameters
diameters = np.linspace(2, 50, 500)  # Adjusted to a manageable size

# Windspeed bins
windspeed_bins = {
    '0-3 m/s': [],
    '3.001-6 m/s': [],
    '6.001-8 m/s': [],
    '8.001+ m/s': []
}

# Organize legs into windspeed bins
for i, row in filtered_combined_clean.iterrows():
    N0 = row['dryintercept']
    D = row['D']
    windspeed = row['Windspeed']

    if 0 <= windspeed <= 3.0:
        windspeed_bins['0-3 m/s'].append((N0, D))
    elif 3.001 <= windspeed <= 6.0:
        windspeed_bins['3.001-6 m/s'].append((N0, D))
    elif 6.001 <= windspeed <= 8.0:
        windspeed_bins['6.001-8 m/s'].append((N0, D))
    elif windspeed > 8.001:
        windspeed_bins['8.001+ m/s'].append((N0, D))

# Plot cumulative mass distributions for each windspeed bin
plt.figure(figsize=(12, 8))
diameters_at_50_mass = []
step = 10  # Subsample for plotting

for bin_label, legs in windspeed_bins.items():
    if not legs:
        continue

    for N0, D in legs:
        # Calculate cumulative masses and normalize
        cumulative_masses = [cumulative_mass(N0, D, d) for d in diameters]
        M_total = total_mass(N0, D)
        normalized_cumulative_mass = np.array(cumulative_masses) / M_total
        normalized_cumulative_mass = np.clip(normalized_cumulative_mass, 0, 1)

        # Plot cumulative mass curve
        plt.plot(diameters[::step], normalized_cumulative_mass[::step], label=f"{bin_label}")

        # Find 50% mass diameter
        interpolation_func = interp1d(normalized_cumulative_mass, diameters, kind='linear', fill_value='extrapolate')
        diameter_at_50 = interpolation_func(0.5)
        diameters_at_50_mass.append(diameter_at_50)
        plt.axvline(diameter_at_50, color='red', linestyle='--', alpha=0.5, 
                    label=f"50% Mass at {diameter_at_50:.2f} µm ({bin_label})")

plt.xlabel('Diameter (µm)', fontsize=14, fontweight='bold')
plt.ylabel('Cumulative mass (Normalized)', fontsize=14, fontweight='bold')
plt.title('Cumulative mass distribution by wind speed bin', fontsize=14, fontweight='bold')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Calculate and plot average cumulative mass distribution
average_cumulative_mass = np.zeros_like(diameters)
total_legs = sum(len(legs) for legs in windspeed_bins.values())

for bin_label, legs in windspeed_bins.items():
    for N0, D in legs:
        cumulative_masses = [cumulative_mass(N0, D, d) for d in diameters]
        average_cumulative_mass += np.array(cumulative_masses)

average_cumulative_mass /= total_legs
M_total_average = total_mass(
    np.mean([N0 for legs in windspeed_bins.values() for N0, D in legs]),
    np.mean([D for legs in windspeed_bins.values() for N0, D in legs])
)
average_cumulative_mass /= M_total_average
average_cumulative_mass = np.clip(average_cumulative_mass, 0, 1)

# Interpolation for 50% mass diameter in average distribution
interpolation_func_avg = interp1d(average_cumulative_mass, diameters, kind='linear', fill_value='extrapolate')
diameter_at_50_avg = interpolation_func_avg(0.5)

# Plot average cumulative mass distribution
plt.figure(figsize=(12, 8))
plt.plot(diameters[::step], average_cumulative_mass[::step], label='Average Cumulative Mass')
plt.axvline(diameter_at_50_avg, color='red', linestyle='--', alpha=0.5, label=f"50% Mass at {diameter_at_50_avg:.2f} µm")
plt.xlabel('Diameter (µm)', fontsize=14, fontweight='bold')
plt.ylabel('Cumulative mass (Normalized)', fontsize=14, fontweight='bold')
plt.title('Average cumulative mass distribution', fontsize=14, fontweight='bold')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# %%
#Averaging the size distribution for all legs and plotting the average size distribution.

# Reset the index of the dataframe to avoid indexing issues
filtered_combined_clean = filtered_combined_clean.reset_index(drop=True)

# Diameters for the size distribution bins (adjusted for CDP data range)
diameters = np.linspace(2.5, 10, 10)  # Diameters in µm (adjust as needed for CDP data)

# Initialize an array to store the size distributions for all legs
size_distributions_all_legs = np.zeros((len(filtered_combined_clean), len(diameters)))

# Loop through each leg to calculate the size distribution
for i, row in filtered_combined_clean.iterrows():
    N0 = row['dryintercept']  # Dry intercept for the leg
    D = row['D']  # Slope for the leg
    
    # Calculate the size distribution (exponential distribution)
    size_distribution = N0 * np.exp(-diameters / D)
    
    # Normalize the size distribution for this leg
    size_distribution /= np.sum(size_distribution)
    
    # Store the normalized size distribution for this leg
    size_distributions_all_legs[i, :] = size_distribution

# Calculate the average size distribution across all legs
average_size_distribution = np.mean(size_distributions_all_legs, axis=0)

# Plot the average size distribution
plt.figure(figsize=(12, 8))
plt.plot(diameters, average_size_distribution, label='Average Size Distribution', color='blue', linewidth=2.5)

# Plot formatting
plt.xlabel('Dry diameter (µm)', fontsize=16, fontweight='bold')
plt.ylabel('Average droplet concentration (/cm³/µm)', fontsize=16, fontweight='bold')
plt.title('Average Size Distribution January-June 2022', fontsize=17, fontweight='bold')
plt.yscale('log')  # Logarithmic y-axis for concentration
plt.tick_params(axis='both', which='major', labelsize=15, width=3)
plt.tight_layout()
plt.legend(fontsize=18)
plt.show()

# %%
