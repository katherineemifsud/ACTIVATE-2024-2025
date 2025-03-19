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
from scipy.stats import mannwhitneyu
import matplotlib.colors as mcolors
#%%
#This is how we will correct our droplet concentration units from 
#dN/dlogD to dN/dD
#We will use the bin width to convert the units
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
#We are only focused on bins 12-30 because we want coarse mode aerosol at 2.5-50 um diameter 
bin_center=[ 2.25, 2.75, 3.25, 3.75, 4.5, 5.75, 6.85, 7.55, 
            9.05, 11.4, 13.8, 17.5, 22.5, 27.5, 32.5, 
            37.5, 42.5, 47.5]
#%%
#Summary Data and meteological data import
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
# %%

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

# %%
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
#Import the instrument data for the cloud-aerosol spectrometer

#Make sure to only work with bins 12-30 for the coarse mode aerosol
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

    dataset = {'Date': date, 'Clear Means': [], 'Cloud Means': []}  # Initialize a dataset dictionary
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
            # print("YESSSSSSSSSSSSSSSSSSSSSSSSSSSSs")
            CAS.append(df_CAS)

        run = run+1 
#%%
#We need to search for our flight legs across the CAS, the 2DS, and the leg files. 
#Then we need to move through every second of every flight leg and filter as cloudy or clear sky. 

# #Attempting to use np.where instead of align to search for leg times across multiple files. 
# master_CAS_BCB = []
# leg_info = []

# for i in range(len(dates_legs)):
#     date = dates_legs[i]
#     leg_dict = leg_data[i]

#     flight_date = leg_dict['Date']
#     BCB_start = leg_dict['LegIndex_02']['StartTimes']
#     BCB_stop = leg_dict['LegIndex_02']['StopTimes']

#     CAS_flight = CAS[i]
#     twoDS_flight = twoDS[i]

#     # Convert Time_Start to numeric
#     CAS_flight['Time_mid'] = pd.to_numeric(CAS_flight['Time_mid'], errors='coerce')
#     twoDS_flight['Time_Start'] = pd.to_numeric(twoDS_flight['Time_Start'], errors='coerce')

#     # Extract required columns
#     CAS_times = CAS_flight['Time_mid'].values
#     CAS_lwc = CAS_flight['LWC_CAS'].values
#     CAS_bins = {f'CAS_Bin{bin_label:02d}': CAS_flight[f'CAS_Bin{bin_label:02d}'].values for bin_label in range(12, 30)}

#     TwoDS_times = twoDS_flight['Time_Start'].values
#     TwoDS_N_total = twoDS_flight['N-total_2DS'].values

#     total_BCB_means = []

#     for k in range(len(BCB_start)):
#         start20 = BCB_start[k]
#         end20 = BCB_stop[k]

#         # Find indices in the BCB range for CAS and TwoDS
#         CAS_indices_in_range = np.where((CAS_times >= start20) & (CAS_times <= end20))[0]
#         TwoDS_indices_in_range = np.where((TwoDS_times >= start20) & (TwoDS_times <= end20))[0]

#         if CAS_indices_in_range.size > 0 and TwoDS_indices_in_range.size > 0:
#             data_labels = []
#             BCB_means = []

#             for CAS_idx, TwoDS_idx in zip(CAS_indices_in_range, TwoDS_indices_in_range):
#                 lwc_val = CAS_lwc[CAS_idx]
#                 N_val = TwoDS_N_total[TwoDS_idx]

#                 # Assign labels
#                 lwc_label = 'Y' if 0 <= lwc_val <= 0.0025 else 'N'
#                 N_label = 'Y' if 0 <= N_val <= 100 else 'N'
#                 label = 'Y' if lwc_label == 'Y' and N_label == 'Y' else 'N'
#                 data_labels.append(label)

#                 # Collect bin values
#                 bin_values = [CAS_bins[f'CAS_Bin{bin_label:02d}'][CAS_idx] for bin_label in range(12, 30)]
#                 BCB_means.append(bin_values)

#             if BCB_means:
#                 total_BCB_means.append(BCB_means)

#             leg_info.append({
#                 'Date': date,
#                 'BCB_start': start20,
#                 'BCB_stop': end20,
#                 'Data_Labels': data_labels,
#             })

#     master_CAS_BCB.append(total_BCB_means)

# # Print leg_info or use it as needed
# for leg in leg_info:
#     print(f"Date: {leg['Date']}, Start: {leg['BCB_start']}, Stop: {leg['BCB_stop']}, Data Labels: {leg['Data_Labels']}")
# # %%
# #Double check the number of legs associated with each date to compare across multiple instruments.  
# leg_count = Counter([leg['Date'] for leg in leg_info])
# print("Number of legs associated with each date:")
# total_legs = 0
# for date, count in sorted(leg_count.items()):
#     print(f"Date: {date}, Number of Legs: {count}")
#     total_legs += count
# print(f"\nTotal number of legs: {total_legs}")
# # %%
# #Now we need to pull the droplet concentration from each bin for each flight leg and calculate the bin
# #mean concentration for each leg. You should end up with 18 mean values, 1 for each bin, for each leg. 

# #This method uses np.where to move across each file.
# master_CAS_BCB = []
# leg_info = []

# for i in range(len(dates_legs)):
#     date = dates_legs[i]
#     leg_dict = leg_data[i]

#     flight_date = leg_dict['Date']
#     BCB_start = np.array(leg_dict['LegIndex_02']['StartTimes'], dtype=float)
#     BCB_stop = np.array(leg_dict['LegIndex_02']['StopTimes'], dtype=float)

#     CAS_flight = CAS[i]
#     twoDS_flight = twoDS[i]

#     # Convert times and data to numeric arrays for filtering
#     CAS_times = np.array(CAS_flight['Time_mid'], dtype=float)
#     TwoDS_times = np.array(twoDS_flight['Time_Start'], dtype=float)

#     lwc = np.array(CAS_flight['LWC_CAS'], dtype=float)
#     N_total = np.array(twoDS_flight['N-total_2DS'], dtype=float)

#     # Pre-fetch bin values for range 12 to 29
#     bins = {
#         f'CAS_Bin{bin_label:02d}': np.array(CAS_flight[f'CAS_Bin{bin_label:02d}'], dtype=float)
#         for bin_label in range(12, 30)
#     }

#     total_BCB_means = []

#     for k in range(len(BCB_start)):
#         start20 = BCB_start[k]
#         end20 = BCB_stop[k]

#         bin_means = {f'Bin{bin_label:02d}_Y_mean': [] for bin_label in range(12, 30)}
#         bin_means.update({f'Bin{bin_label:02d}_N_mean': [] for bin_label in range(12, 30)})
#         bin_means.update({'Date': date, 'BCB_start': start20, 'BCB_stop': end20})

#         # Use np.where to find indices within the BCB interval
#         CAS_indices_in_range = np.where((CAS_times >= start20) & (CAS_times <= end20))[0]
#         TwoDS_indices_in_range = np.where((TwoDS_times >= start20) & (TwoDS_times <= end20))[0]

#         if CAS_indices_in_range.size > 0 and TwoDS_indices_in_range.size > 0:
#             for cas_idx, twods_idx in zip(CAS_indices_in_range, TwoDS_indices_in_range):
#                 lwc_val = lwc[cas_idx]
#                 N_val = N_total[twods_idx]
#                 lwc_label = 'Y' if 0 <= lwc_val <= 0.0025 else 'N'
#                 N_label = 'Y' if 0 <= N_val <= 100 else 'N'
#                 label = 'Y' if lwc_label == 'Y' and N_label == 'Y' else 'N'

#                 for bin_label in range(12, 30):
#                     bin_key = f'Bin{bin_label:02d}_{label}_mean'
#                     bin_means[bin_key].append(bins[f'CAS_Bin{bin_label:02d}'][cas_idx])

#         # Calculate means
#         for bin_label in range(12, 30):
#             for label in ['Y', 'N']:
#                 bin_key = f'Bin{bin_label:02d}_{label}_mean'
#                 if bin_means[bin_key]:
#                     bin_means[bin_key] = np.nanmean(bin_means[bin_key])
#                 else:
#                     bin_means[bin_key] = np.nan

#         total_BCB_means.append(bin_means)

#     master_CAS_BCB.append(total_BCB_means)

# # Print or use master_CAS_BCB as needed
# for item in master_CAS_BCB:
#     for bin_means in item:
#         print(f"Date: {bin_means['Date']}, Start: {bin_means['BCB_start']}, Stop: {bin_means['BCB_stop']}")
#         for bin_label in range(12, 30):
#             for label in ['Y', 'N']:
#                 bin_key = f'Bin{bin_label:02d}_{label}_mean'
#                 print(f"   {bin_key}: {bin_means[bin_key]}")
# # %%
# # Count the total number of legs from master_CAS_BCB
# total_legs = sum(len(item) for item in master_CAS_BCB)
# print(f"Total number of legs: {total_legs}")
# #%%
# #Now we need to apply our conversion from dNdlog10D to dNdD for each bin and calculate the mean concentration
# Y_BCB_calc = []
# N_BCB_calc = []

# for flight_data in master_CAS_BCB:
#     for bin_means in flight_data:
#         Y_calc = {'Date': bin_means['Date'], 'BCB_start': bin_means['BCB_start'], 'BCB_stop': bin_means['BCB_stop']}
#         N_calc = {'Date': bin_means['Date'], 'BCB_start': bin_means['BCB_start'], 'BCB_stop': bin_means['BCB_stop']}
        
#         for bin_label in range(12, 30):
#             bin_key_Y = f'Bin{bin_label}_Y_mean'
#             bin_key_N = f'Bin{bin_label}_N_mean'

#             Y_calc[bin_key_Y] = np.nanmean(bin_means[bin_key_Y]) * Logg[bin_label - 12]
#             N_calc[bin_key_N] = np.nanmean(bin_means[bin_key_N]) * Logg[bin_label - 12]

#         Y_BCB_calc.append(Y_calc)
#         N_BCB_calc.append(N_calc)
# %%
##We need to step away from leg-averaging. If we are working with rain, 
# rain is continual. So we need every second of our concentrations rather than one
#droplet concentration per bin per leg. 
# Y_BCB_calc_full = []  # Clear sky (time-resolved)
# N_BCB_calc_full = []  # In-cloud (time-resolved)

# for i in range(len(dates_legs)):
#     date = dates_legs[i]
#     leg_dict = leg_data[i]

#     flight_date = leg_dict['Date']
#     BCB_start = leg_dict['LegIndex_02']['StartTimes']
#     BCB_stop = leg_dict['LegIndex_02']['StopTimes']

#     CAS_flight = CAS[i]
#     twoDS_flight = twoDS[i]

#     # Convert times and columns to numeric
#     CAS_flight['Time_mid'] = pd.to_numeric(CAS_flight['Time_mid'], errors='coerce')
#     twoDS_flight['Time_Start'] = pd.to_numeric(twoDS_flight['Time_Start'], errors='coerce')

#     CAS_times = CAS_flight['Time_mid'].values
#     CAS_lwc = CAS_flight['LWC_CAS'].values
#     CAS_bins = {f'CAS_Bin{bin_label:02d}': CAS_flight[f'CAS_Bin{bin_label:02d}'].values for bin_label in range(12, 30)}

#     TwoDS_times = twoDS_flight['Time_Start'].values
#     TwoDS_N_total = twoDS_flight['N-total_2DS'].values

#     for k in range(len(BCB_start)):
#         start20 = BCB_start[k]
#         end20 = BCB_stop[k]

#         # Find indices within the BCB range
#         CAS_indices_in_range = np.where((CAS_times >= start20) & (CAS_times <= end20))[0]
#         TwoDS_indices_in_range = np.where((TwoDS_times >= start20) & (TwoDS_times <= end20))[0]

#         for CAS_idx, TwoDS_idx in zip(CAS_indices_in_range, TwoDS_indices_in_range):
#             lwc_val = CAS_lwc[CAS_idx]
#             N_val = TwoDS_N_total[TwoDS_idx]

#             # Apply both CAS and 2DS filters
#             lwc_label = 'Y' if 0 <= lwc_val <= 0.0025 else 'N'  # CAS filter
#             N_label = 'Y' if 0 <= N_val <= 100 else 'N'  # 2DS filter
#             label = 'Y' if lwc_label == 'Y' and N_label == 'Y' else 'N'

#             # Prepare dictionaries for time-resolved data
#             calc_entry = {
#                 'Date': date,
#                 'Time': CAS_times[CAS_idx],  # Use CAS time for consistency
#                 'BCB_start': start20,
#                 'BCB_stop': end20,
#                 'LWC': lwc_val
#             }

#             for bin_label in range(12, 30):
#                 bin_key = f'Bin{bin_label}_mean'
#                 calc_entry[bin_key] = np.nanmean(CAS_bins[f'CAS_Bin{bin_label:02d}'][CAS_idx]) * Logg[bin_label - 12]

#             # Append to appropriate dataset
#             if label == 'Y':
#                 Y_BCB_calc_full.append(calc_entry)  # Clear sky (meets both CAS and 2DS filters)
#             else:
#                 N_BCB_calc_full.append(calc_entry)  # In-cloud or outside filter criteria
# #this is for eveyr bin not total concentration across all bins 
#%%
# #for total concentration across all bins
# Y_BCB_calc_full = []  # Clear sky (time-resolved)
# N_BCB_calc_full = []  # In-cloud (time-resolved)

# for i in range(len(dates_legs)):
#     date = dates_legs[i]
#     leg_dict = leg_data[i]

#     flight_date = leg_dict['Date']
#     BCB_start = leg_dict['LegIndex_02']['StartTimes']
#     BCB_stop = leg_dict['LegIndex_02']['StopTimes']

#     CAS_flight = CAS[i]
#     twoDS_flight = twoDS[i]

#     # Convert times and columns to numeric
#     CAS_flight['Time_mid'] = pd.to_numeric(CAS_flight['Time_mid'], errors='coerce')
#     twoDS_flight['Time_Start'] = pd.to_numeric(twoDS_flight['Time_Start'], errors='coerce')

#     CAS_times = CAS_flight['Time_mid'].values
#     CAS_lwc = CAS_flight['LWC_CAS'].values
#     CAS_bins = {f'CAS_Bin{bin_label:02d}': CAS_flight[f'CAS_Bin{bin_label:02d}'].values for bin_label in range(12, 30)}

#     TwoDS_times = twoDS_flight['Time_Start'].values
#     TwoDS_N_total = twoDS_flight['N-total_2DS'].values

#     for k in range(len(BCB_start)):
#         start20 = BCB_start[k]
#         end20 = BCB_stop[k]

#         # Find indices within the BCB range
#         CAS_indices_in_range = np.where((CAS_times >= start20) & (CAS_times <= end20))[0]
#         TwoDS_indices_in_range = np.where((TwoDS_times >= start20) & (TwoDS_times <= end20))[0]

#         for CAS_idx, TwoDS_idx in zip(CAS_indices_in_range, TwoDS_indices_in_range):
#             lwc_val = CAS_lwc[CAS_idx]
#             N_val = TwoDS_N_total[TwoDS_idx]

#             # Apply both CAS and 2DS filters
#             lwc_label = 'Y' if 0 <= lwc_val <= 0.0025 else 'N'  # CAS filter
#             N_label = 'Y' if 0 <= N_val <= 100 else 'N'  # 2DS filter
#             label = 'Y' if lwc_label == 'Y' and N_label == 'Y' else 'N'

#             # Calculate total concentration across all bins
#             total_concentration = sum(
#                 np.nan_to_num(CAS_bins[f'CAS_Bin{bin_label:02d}'][CAS_idx]) * Logg[bin_label - 12]
#                 for bin_label in range(12, 30)
#             )

#             # Prepare dictionaries for time-resolved data
#             calc_entry = {
#                 'Date': date,
#                 'Time': CAS_times[CAS_idx],  # Use CAS time for consistency
#                 'BCB_start': start20,
#                 'BCB_stop': end20,
#                 'LWC': lwc_val,
#                 'Total_Concentration': total_concentration
#             }

#             # Append to appropriate dataset
#             if label == 'Y':
#                 Y_BCB_calc_full.append(calc_entry)  # Clear sky (meets both CAS and 2DS filters)
#             else:
#                 N_BCB_calc_full.append(calc_entry)  # In-cloud or outside filter criteria

#%%
#in-cloud droplet concentrations, we need to adjust lwc threshold. 
#continue using the CAS for lwc and pull CAS concentrations but filter 2DS N also to get drizzle drops out 

# New list for in-cloud droplet concentrations
in_cloud_concentrations = []

for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict = leg_data[i]

    ACB_start = leg_dict['LegIndex_03']['StartTimes']
    ACB_stop = leg_dict['LegIndex_03']['StopTimes']
    BCT_start =leg_dict['LegIndex_04']['StartTimes']
    BCT_stop = leg_dict['LegIndex_04']['StopTimes']
    CAS_flight = CAS[i]
    twoDS_flight = twoDS[i]

    # Convert times to numeric
    CAS_flight['Time_mid'] = pd.to_numeric(CAS_flight['Time_mid'], errors='coerce')
    twoDS_flight['Time_Start'] = pd.to_numeric(twoDS_flight['Time_Start'], errors='coerce')

    CAS_times = CAS_flight['Time_mid'].values
    CAS_lwc = CAS_flight['LWC_CAS'].values
    CAS_bins = {f'CAS_Bin{bin_label:02d}': CAS_flight[f'CAS_Bin{bin_label:02d}'].values for bin_label in range(12, 30)}

    TwoDS_times = twoDS_flight['Time_Start'].values
    # TwoDS_N_total = twoDS_flight['N-total_2DS'].values

    for k in range(len(ACB_start)):
        start_time = ACB_start[k]
        end_time = ACB_stop[k]

        # Find indices for the CAS and 2DS within the BCB range
        CAS_indices_in_range = np.where((CAS_times >= start_time) & (CAS_times <= end_time))[0]
        # TwoDS_indices_in_range = np.where((TwoDS_times >= start_time) & (TwoDS_times <= end_time))[0]

        for CAS_idx in zip(CAS_indices_in_range):
            lwc_val = CAS_lwc[CAS_idx]
            # N_val = TwoDS_N_total[TwoDS_idx]

            # Apply in-cloud filters
            if lwc_val >= 0.01:
                calc_entry = {
                    'Date': date,
                    'Time': CAS_times[CAS_idx],
                    'BCB_start': start_time,
                    'BCB_stop': end_time,
                    'CWC': lwc_val,
                    # 'N_total_2DS': N_val  # Optional: Store the 2DS number concentration
                }

                # Include droplet concentrations from CAS bins
                for bin_label in range(12, 30):
                    bin_key = f'Bin{bin_label}_concentration'
                    calc_entry[bin_key] = CAS_bins[f'CAS_Bin{bin_label:02d}'][CAS_idx]

                # Append the in-cloud entry
                in_cloud_concentrations.append(calc_entry)

#this reports in bins not total 
#%%
#adding BCT and ACB legs together in a combined dictionary 
# New list for in-cloud droplet concentrations
in_cloud_concentrations = []

for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict = leg_data[i]

    ACB_start = leg_dict['LegIndex_03']['StartTimes']
    ACB_stop = leg_dict['LegIndex_03']['StopTimes']
    BCT_start = leg_dict['LegIndex_04']['StartTimes']
    BCT_stop = leg_dict['LegIndex_04']['StopTimes']

    CAS_flight = CAS[i]
    twoDS_flight = twoDS[i]

    # Convert times to numeric
    CAS_flight['Time_mid'] = pd.to_numeric(CAS_flight['Time_mid'], errors='coerce')

    CAS_times = CAS_flight['Time_mid'].values
    CAS_lwc = CAS_flight['LWC_CAS'].values
    CAS_bins = {f'CAS_Bin{bin_label:02d}': CAS_flight[f'CAS_Bin{bin_label:02d}'].values for bin_label in range(12, 30)}

    # Combine ACB and BCT legs into one loop
    combined_legs = [
        (ACB_start, ACB_stop),
        (BCT_start, BCT_stop)
    ]

    for leg_start, leg_stop in combined_legs:
        for k in range(len(leg_start)):
            start_time = leg_start[k]
            end_time = leg_stop[k]

            # Find indices for the CAS within the leg range
            CAS_indices_in_range = np.where((CAS_times >= start_time) & (CAS_times <= end_time))[0]

            for CAS_idx in CAS_indices_in_range:
                lwc_val = CAS_lwc[CAS_idx]

                # Apply in-cloud filters
                if lwc_val >= 0.01:  # Adjust LWC threshold as needed
                    calc_entry = {
                        'Date': date,
                        'Time': CAS_times[CAS_idx],
                        'Leg_start': start_time,
                        'Leg_stop': end_time,
                        'CWC': lwc_val  # Cloud water content
                    }

                    # Include droplet concentrations from CAS bins
                    for bin_label in range(12, 30):
                        bin_key = f'Bin{bin_label}_concentration'
                        calc_entry[bin_key] = CAS_bins[f'CAS_Bin{bin_label:02d}'][CAS_idx]

                    # Append the in-cloud entry
                    in_cloud_concentrations.append(calc_entry)

# Debugging: Check a few entries of in-cloud concentrations
print(f"Number of in-cloud entries: {len(in_cloud_concentrations)}")
print(f"First 5 entries: {in_cloud_concentrations[:5]}")

#%%
#this is total concentration
# in_cloud_concentrations = []

# for i in range(len(dates_legs)):
#     date = dates_legs[i]
#     leg_dict = leg_data[i]

#     BCB_start = leg_dict['LegIndex_02']['StartTimes']
#     BCB_stop = leg_dict['LegIndex_02']['StopTimes']

#     CAS_flight = CAS[i]
#     twoDS_flight = twoDS[i]

#     # Convert times to numeric
#     CAS_flight['Time_mid'] = pd.to_numeric(CAS_flight['Time_mid'], errors='coerce')
#     twoDS_flight['Time_Start'] = pd.to_numeric(twoDS_flight['Time_Start'], errors='coerce')

#     CAS_times = CAS_flight['Time_mid'].values
#     CAS_lwc = CAS_flight['LWC_CAS'].values
#     CAS_bins = {f'CAS_Bin{bin_label:02d}': CAS_flight[f'CAS_Bin{bin_label:02d}'].values for bin_label in range(12, 30)}

#     TwoDS_times = twoDS_flight['Time_Start'].values
#     TwoDS_N_total = twoDS_flight['N-total_2DS'].values

#     for k in range(len(BCB_start)):
#         start_time = BCB_start[k]
#         end_time = BCB_stop[k]

#         # Find indices for the CAS and 2DS within the BCB range
#         CAS_indices_in_range = np.where((CAS_times >= start_time) & (CAS_times <= end_time))[0]
#         TwoDS_indices_in_range = np.where((TwoDS_times >= start_time) & (TwoDS_times <= end_time))[0]

#         for CAS_idx, TwoDS_idx in zip(CAS_indices_in_range, TwoDS_indices_in_range):
#             lwc_val = CAS_lwc[CAS_idx]
#             N_val = TwoDS_N_total[TwoDS_idx]

#             # Apply in-cloud filters
#             if lwc_val >= 0.01 and 0 <= N_val <= 100:
#                 # Calculate the total concentration across all bins
#                 total_concentration = sum(
#                     np.nan_to_num(CAS_bins[f'CAS_Bin{bin_label:02d}'][CAS_idx]) * Logg[bin_label - 12]
#                     for bin_label in range(12, 30)
#                 )

#                 # Prepare the entry for this second
#                 calc_entry = {
#                     'Date': date,
#                     'Time': CAS_times[CAS_idx],
#                     'BCB_start': start_time,
#                     'BCB_stop': end_time,
#                     'LWC': lwc_val,
#                     'N_total_2DS': N_val,  # Optional: Store the 2DS number concentration
#                     'Total_Concentration': total_concentration
#                 }

#                 # Append the in-cloud entry
#                 in_cloud_concentrations.append(calc_entry)

#%%
# This code calculates total concentration in cm³
in_cloud_concentrations = []

for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict = leg_data[i]

    # Get start and stop times for ACB (03) and BCT (04) legs
    ACB_start = leg_dict['LegIndex_03']['StartTimes']
    ACB_stop = leg_dict['LegIndex_03']['StopTimes']
    BCT_start = leg_dict['LegIndex_04']['StartTimes']
    BCT_stop = leg_dict['LegIndex_04']['StopTimes']

    CAS_flight = CAS[i]

    # Convert times to numeric
    CAS_flight['Time_mid'] = pd.to_numeric(CAS_flight['Time_mid'], errors='coerce')

    CAS_times = CAS_flight['Time_mid'].values
    CAS_lwc = CAS_flight['LWC_CAS'].values
    CAS_bins = {f'CAS_Bin{bin_label:02d}': CAS_flight[f'CAS_Bin{bin_label:02d}'].values for bin_label in range(12, 30)}

    # Define bin widths for each CAS bin (in µm)
    bin_widths = [bin_log[bin_label - 12] for bin_label in range(12, 30)]

    # Combine ACB and BCT legs
    all_legs_start = ACB_start + BCT_start
    all_legs_stop = ACB_stop + BCT_stop

    # Loop through all combined legs
    for k in range(len(all_legs_start)):
        start_time = all_legs_start[k]
        end_time = all_legs_stop[k]

        # Find indices for the CAS within the leg range
        CAS_indices_in_range = np.where((CAS_times >= start_time) & (CAS_times <= end_time))[0]

        for CAS_idx in CAS_indices_in_range:
            lwc_val = CAS_lwc[CAS_idx]

            # Apply in-cloud filter for LWC
            if lwc_val >= 0.01:
                # Calculate the total concentration across all bins
                total_concentration = sum(
                    np.nan_to_num(CAS_bins[f'CAS_Bin{bin_label:02d}'][CAS_idx]) * bin_width
                    for bin_label, bin_width in zip(range(12, 30), bin_widths)
                )

                # Prepare the entry for this second
                calc_entry = {
                    'Date': date,
                    'Time': CAS_times[CAS_idx],
                    'Leg_start': start_time,
                    'Leg_stop': end_time,
                    'CWC': lwc_val,
                    'Total_Concentration': total_concentration  # Units: cm³
                }

                # Append the in-cloud entry
                in_cloud_concentrations.append(calc_entry)

# Debugging: Print the number of entries and a sample
print(f"Number of in-cloud entries: {len(in_cloud_concentrations)}")
print(f"Sample entries: {in_cloud_concentrations[:5]}")

# %%
# # Find dates not included in the in-cloud dataset
# all_dates = set(dates_legs)
# included_dates = set(entry['Date'] for entry in in_cloud_concentrations)
# excluded_dates = sorted(all_dates - included_dates)

# print(f"Excluded Dates ({len(excluded_dates)}): {excluded_dates}")

# # Check LWC and N distributions for excluded dates
# for date in excluded_dates:
#     date_index = dates_legs.index(date)
#     CAS_flight = CAS[date_index]
#     twoDS_flight = twoDS[date_index]

#     lwc_values = CAS_flight['LWC_CAS']
#     n_values = twoDS_flight['N-total_2DS']

#     print(f"\nDate: {date}")
#     print(f"  LWC Values (CAS): Min={np.nanmin(lwc_values):.3f}, Max={np.nanmax(lwc_values):.3f}")
#     print(f"  2DS N Values: Min={np.nanmin(n_values):.3f}, Max={np.nanmax(n_values):.3f}")
#%%
# # Find dates not included in the in-cloud dataset
# all_dates = set(dates_legs)
# included_dates = set(entry['Date'] for entry in in_cloud_concentrations)
# excluded_dates = sorted(all_dates - included_dates)

# print(f"Excluded Dates ({len(excluded_dates)}): {excluded_dates}")
# for date in excluded_dates:
#     date_index = dates_legs.index(date)
#     CAS_flight = CAS[date_index]
#     lwc_values = CAS_flight['LWC_CAS']
#     print(f"\nDate: {date}")
#     print(f"  LWC Values (CAS): Min={np.nanmin(lwc_values):.3f}, Max={np.nanmax(lwc_values):.3f}")
# #%%
# #debugging 5/20
# # Debugging for 2022-05-20
# target_date = '2022-05-20'

# # Locate the index for the target date
# if target_date in dates_legs:
#     date_index = dates_legs.index(target_date)
#     date = dates_legs[date_index]
#     leg_dict = leg_data[date_index]

#     ACB_start = leg_dict['LegIndex_03']['StartTimes']
#     ACB_stop = leg_dict['LegIndex_03']['StopTimes']
#     BCT_start = leg_dict['LegIndex_04']['StartTimes']
#     BCT_stop = leg_dict['LegIndex_04']['StopTimes']

#     CAS_flight = CAS[date_index]
#     CAS_flight['Time_mid'] = pd.to_numeric(CAS_flight['Time_mid'], errors='coerce')
#     CAS_times = CAS_flight['Time_mid'].values
#     CAS_lwc = CAS_flight['LWC_CAS'].values

#     print(f"Debugging for {target_date}...")
#     print(f"Number of ACB Legs: {len(ACB_start)}")
#     print(f"Number of BCT Legs: {len(BCT_start)}")

#     # Check ACB legs
#     for k in range(len(ACB_start)):
#         start_time = ACB_start[k]
#         end_time = ACB_stop[k]
#         CAS_indices_in_range = np.where((CAS_times >= start_time) & (CAS_times <= end_time))[0]

#         print(f"\nACB Leg {k + 1}: Start={start_time}, Stop={end_time}, Indices={CAS_indices_in_range}")
#         if CAS_indices_in_range.size > 0:
#             lwc_vals_in_leg = CAS_lwc[CAS_indices_in_range]
#             print(f"  LWC Values in Leg: Min={np.nanmin(lwc_vals_in_leg):.3f}, Max={np.nanmax(lwc_vals_in_leg):.3f}")
#             print(f"  Filtered Indices: {np.where(lwc_vals_in_leg >= 0.01)[0]}")
#         else:
#             print("  No CAS data in this range.")

#     # Check BCT legs
#     for k in range(len(BCT_start)):
#         start_time = BCT_start[k]
#         end_time = BCT_stop[k]
#         CAS_indices_in_range = np.where((CAS_times >= start_time) & (CAS_times <= end_time))[0]

#         print(f"\nBCT Leg {k + 1}: Start={start_time}, Stop={end_time}, Indices={CAS_indices_in_range}")
#         if CAS_indices_in_range.size > 0:
#             lwc_vals_in_leg = CAS_lwc[CAS_indices_in_range]
#             print(f"  LWC Values in Leg: Min={np.nanmin(lwc_vals_in_leg):.3f}, Max={np.nanmax(lwc_vals_in_leg):.3f}")
#             print(f"  Filtered Indices: {np.where(lwc_vals_in_leg >= 0.01)[0]}")
#         else:
#             print("  No CAS data in this range.")
# else:
#     print(f"Date {target_date} is not in the dataset.")
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

    # # Extract ACB start and stop times
    # ACB_start = leg_dict['LegIndex_03']['StartTimes']
    # ACB_stop = leg_dict['LegIndex_03']['StopTimes']

    # Combine ACB and BCT legs like the rain code
    ACB_start = leg_dict['LegIndex_03']['StartTimes']
    ACB_stop = leg_dict['LegIndex_03']['StopTimes']
    BCT_start = leg_dict['LegIndex_04']['StartTimes']
    BCT_stop = leg_dict['LegIndex_04']['StopTimes']
    all_legs_start = ACB_start + BCT_start
    all_legs_stop = ACB_stop + BCT_stop

    # Extract relevant data from twoDS
    twoDS_flight = twoDS[i]
    twoDS_flight['Time_Start'] = pd.to_numeric(twoDS_flight['Time_Start'], errors='coerce')

    twoDS_times = twoDS_flight['Time_Start'].values
    twoDS_lwc = twoDS_flight['LWC_2DS'].values
    twoDS_bins = {f'dNdlogD_liquid_{bin_label:03d}_2DS': twoDS_flight[f'dNdlogD_liquid_{bin_label:03d}_2DS'].values
                  for bin_label in range(6, 129)}

    # Loop through ACB legs only
    for k in range(len(all_legs_start)):
        start_time = all_legs_start[k]
        end_time = all_legs_stop[k]

        # Find indices for the 2DS within the leg range
        twoDS_indices_in_range = np.where((twoDS_times >= start_time) & (twoDS_times <= end_time))[0]

        for twoDS_idx in twoDS_indices_in_range:
            lwc_val = twoDS_lwc[twoDS_idx]

            # Apply rain-specific filters
            if lwc_val >= 0.00001:  # LWC threshold (0.01 g/m³ = 1e-5 kg/m³)
                # Compute the total concentration using precomputed log widths
                # total_concentration = sum(
                #     np.nan_to_num(twoDS_bins[f'dNdlogD_liquid_{bin_label:03d}_2DS'][twoDS_idx]) * log_width
                #     for bin_label, log_width in zip(range(6, 129), twoDS_logg)
                # )

                # Compute the total concentration using logarithmic bin widths
                total_concentration = sum(
                    np.nan_to_num(twoDS_bins[f'dNdlogD_liquid_{bin_label:03d}_2DS'][twoDS_idx]) * log_width
                    for bin_label, log_width in zip(range(6, 129), twoDS_logg)
                )

# Convert Total Concentration from m³ to cm³
                total_concentration /= 1e6  # /m³ to /cm³


                # Prepare the entry for this time step
                rain_entry = {
                    'Date': date,
                    'Time': twoDS_times[twoDS_idx],
                    'Leg_start': start_time,
                    'Leg_stop': end_time,
                    'LWC': lwc_val,  # Liquid water content (in kg/m³)
                    'Total_Concentration': total_concentration  # Converted to /m³
                }

                # Append the rain droplet concentration entry
                rain_concentrations.append(rain_entry)

# Debugging: Check a few entries of rain concentrations
print(f"Number of rain entries: {len(rain_concentrations)}")
print(f"First 5 entries: {rain_concentrations[:5]}")
#%%
# rain_concentrations = []

# for i in range(len(dates_legs)):
#     date = dates_legs[i]
#     leg_dict = leg_data[i]

#     # Combine ACB and BCT legs
#     ACB_start = leg_dict['LegIndex_03']['StartTimes']
#     ACB_stop = leg_dict['LegIndex_03']['StopTimes']
#     BCT_start = leg_dict['LegIndex_04']['StartTimes']
#     BCT_stop = leg_dict['LegIndex_04']['StopTimes']
#     all_legs_start = ACB_start + BCT_start
#     all_legs_stop = ACB_stop + BCT_stop

#     # Extract relevant data from twoDS
#     twoDS_flight = twoDS[i]
#     twoDS_flight['Time_Start'] = pd.to_numeric(twoDS_flight['Time_Start'], errors='coerce')

#     twoDS_times = twoDS_flight['Time_Start'].values
#     twoDS_lwc = twoDS_flight['LWC_2DS'].values
#     twoDS_bins = {f'dNdlogD_liquid_{bin_label:03d}_2DS': twoDS_flight[f'dNdlogD_liquid_{bin_label:03d}_2DS'].values
#                   for bin_label in range(6, 129)}

#     # Loop through combined ACB and BCT legs
#     for k in range(len(all_legs_start)):
#         start_time = all_legs_start[k]
#         end_time = all_legs_stop[k]

#         # Find indices within the leg range
#         twoDS_indices_in_range = np.where((twoDS_times >= start_time) & (twoDS_times <= end_time))[0]

#         for twoDS_idx in twoDS_indices_in_range:
#             lwc_val = twoDS_lwc[twoDS_idx]
#             N_liquid_total = 0  # Initialize total concentration for this time step

#             # Apply rain-specific filters
#             if lwc_val >= 0.00001:  # LWC threshold (0.01 g/m³ = 1e-5 kg/m³)
#                 # Compute total concentration using bins
#                 for bin_label in range(6, 129):
#                     bin_column = f'dNdlogD_liquid_{bin_label:03d}_2DS'
#                     if bin_column in twoDS_flight.columns:
#                         N_bin = twoDS_flight[bin_column].iloc[twoDS_idx]  # Raw bin value in /m³

#                         # Convert dNdlogD to dN
#                         N_dD = (N_bin * twoDS_logg[bin_label - 6])

#                         # Accumulate total concentration
#                         N_liquid_total += N_dD

#                 # Convert Total Concentration from m³ to cm³
#                 N_liquid_total /= 1e6  # /m³ to /cm³

#                 # Prepare the entry for this time step
#                 rain_entry = {
#                     'Date': date,
#                     'Time': twoDS_times[twoDS_idx],
#                     'Leg_start': start_time,
#                     'Leg_stop': end_time,
#                     'LWC': lwc_val,  # Liquid water content (in g/m³)
#                     'Total_Concentration': N_liquid_total  # Total concentration (/cm³)
#                 }

#                 # Append the rain droplet concentration entry
#                 rain_concentrations.append(rain_entry)

# # Debugging: Check the total entries and sample data
# print(f"Number of rain concentration entries: {len(rain_concentrations)}")
# print(f"First 5 entries: {rain_concentrations[:5]}")


# %%
# Convert LWC to g/m³ and N_liquid to /cm³
for entry in rain_concentrations:
    entry['LWC'] = entry['LWC'] * 1e3  # kg/m³ to g/m³
    # entry['Total_Concentration'] = entry['Total_Concentration'] / 1e6  # /m³ to /cm³

# Verify the converted data
print("Sample entries after unit conversion:")
for sample in rain_concentrations[:5]:
    print(sample)
#%%
# #BCT legs
# rain_concentrations_BCT = []

# for i in range(len(dates_legs)):
#     date = dates_legs[i]
#     leg_dict = leg_data[i]

#     # Extract BCT start and stop times
#     BCT_start = leg_dict['LegIndex_04']['StartTimes']
#     BCT_stop = leg_dict['LegIndex_04']['StopTimes']

#     # Extract relevant data from twoDS
#     twoDS_flight = twoDS[i]
#     twoDS_flight['Time_Start'] = pd.to_numeric(twoDS_flight['Time_Start'], errors='coerce')

#     twoDS_times = twoDS_flight['Time_Start'].values
#     twoDS_lwc = twoDS_flight['LWC_2DS'].values
#     twoDS_bins = {f'dNdlogD_liquid_{bin_label:03d}_2DS': twoDS_flight[f'dNdlogD_liquid_{bin_label:03d}_2DS'].values
#                   for bin_label in range(6, 129)}

#     # Loop through BCT legs only
#     for k in range(len(BCT_start)):
#         start_time = BCT_start[k]
#         end_time = BCT_stop[k]

#         # Find indices for the 2DS within the leg range
#         twoDS_indices_in_range = np.where((twoDS_times >= start_time) & (twoDS_times <= end_time))[0]

#         for twoDS_idx in twoDS_indices_in_range:
#             lwc_val = twoDS_lwc[twoDS_idx]

#             # Apply rain-specific filters
#             if lwc_val >= 0.00001:
#                 # Compute the total concentration using precomputed log widths
#                 total_concentration = sum(
#                     np.nan_to_num(twoDS_bins[f'dNdlogD_liquid_{bin_label:03d}_2DS'][twoDS_idx]) * log_width
#                     for bin_label, log_width in zip(range(6, 129), twoDS_logg)
#                 )

#                 # Prepare the entry for this time step
#                 rain_entry_BCT = {
#                     'Date': date,
#                     'Time': twoDS_times[twoDS_idx],
#                     'Leg_start': start_time,
#                     'Leg_stop': end_time,
#                     'LWC': lwc_val,  # Liquid water content (in kg/m³)
#                     'Total_Concentration': total_concentration  # Converted to /m³
#                 }

#                 # Append the rain droplet concentration entry
#                 rain_concentrations_BCT.append(rain_entry_BCT)

# # Debugging: Check a few entries of rain concentrations
# print(f"Number of rain entries: {len(rain_concentrations_BCT)}")
# print(f"First 5 entries: {rain_concentrations_BCT[:5]}")
# #%%
# for entry in rain_concentrations_BCT:
#     entry['LWC'] = entry['LWC'] * 1e3  # kg/m³ to g/m³
#     entry['Total_Concentration'] = entry['Total_Concentration'] / 1e6  # /m³ to /cm³

# # Verify the converted data
# print("Sample entries after unit conversion:")
# for sample_BCT in rain_concentrations_BCT[:5]:
#     print(sample_BCT)
#%%
#first calculate the bin centers 
# %%
# Convert Bin_Lower and Bin_Upper from µm to meters (once, since they are constant)
Bin_Lower_m = [lower / 1e6 for lower in Bin_Lower]  # Convert µm to m
Bin_Upper_m = [upper / 1e6 for upper in Bin_Upper]  # Convert µm to m

# Calculate bin centers
Bin_Centers_m = [(lower + upper) / 2 for lower, upper in zip(Bin_Lower_m, Bin_Upper_m)]  # Bin centers in meters
Bin_Centers_Cubed = [center**3 for center in Bin_Centers_m]  # Cube each center

# Print the cubed values
print("Cubed Bin Centers (in m³):")
for i, (center, cubed) in enumerate(zip(Bin_Centers_m, Bin_Centers_Cubed), start=1):
    print(f"Bin {i}: Center = {center:.6e} m, Center³ = {cubed:.6e} m³")



# %%
#calculating rain water content 
rho_water = 1e3 # Density of water in g/m³
pi_over_6 = np.pi / 6

# Initialize RWC results
rain_water_content = []

for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict = leg_data[i]

    # Combine ACB and BCT legs like the rain code
    ACB_start = leg_dict['LegIndex_03']['StartTimes']
    ACB_stop = leg_dict['LegIndex_03']['StopTimes']
    BCT_start = leg_dict['LegIndex_04']['StartTimes']
    BCT_stop = leg_dict['LegIndex_04']['StopTimes']
    all_legs_start = ACB_start + BCT_start
    all_legs_stop = ACB_stop + BCT_stop

    # Extract relevant data from twoDS
    twoDS_flight = twoDS[i]
    twoDS_flight['Time_Start'] = pd.to_numeric(twoDS_flight['Time_Start'], errors='coerce')

    twoDS_times = twoDS_flight['Time_Start'].values
    twoDS_bins = {f'dNdlogD_liquid_{bin_label:03d}_2DS': twoDS_flight[f'dNdlogD_liquid_{bin_label:03d}_2DS'].values
                  for bin_label in range(6, 129)}

    # Loop through combined ACB and BCT legs
    for k in range(len(all_legs_start)):
        start_time = all_legs_start[k]
        end_time = all_legs_stop[k]

        # Find indices within the leg range
        twoDS_indices_in_range = np.where((twoDS_times >= start_time) & (twoDS_times <= end_time))[0]

        for twoDS_idx in twoDS_indices_in_range:
            lwc_val = twoDS_flight['LWC_2DS'].iloc[twoDS_idx]
            N_liquid_total = 0

            if lwc_val >= 0.00001:  # LWC threshold (0.01 g/m³ = 1e-5 kg/m³)
            # Calculate RWC using bins
                for bin_label in (range(6, 129)):
                    bin_column = f'dNdlogD_liquid_{bin_label:03d}_2DS'
                    if bin_column in twoDS_flight.columns:
                        N_bin = twoDS_flight[bin_column].iloc[twoDS_idx]  # Raw bin value in /m³
                        
                        N_dD = (N_bin * twoDS_logg[bin_label - 6])
                        
                        N_liquid_total += N_dD * Bin_Centers_Cubed[bin_label - 6]


                # Compute RWC for this time step
                RWC = pi_over_6 * rho_water * N_liquid_total # kg/m³

                # Append results
                rain_water_content.append({
                    'Date': date,
                    'Time': twoDS_times[twoDS_idx],
                    'Leg_start': start_time,
                    'Leg_stop': end_time,
                    'LWC': lwc_val,
                    'RWC': RWC

                })

# Debugging: Check a few entries of rain water content
print(f"Number of RWC entries: {len(rain_water_content)}")
print(f"First 5 entries: {rain_water_content[:5]}")
#%%
# convert RWC to g/m³ 
for entry in rain_water_content:
    entry['RWC'] = entry['RWC'] * 1e3  # kg/m³ to g/m³
    entry['LWC'] = entry['LWC'] * 1e3  # kg/m³ to g/m³
#%%
#add RWC and CWC for total LWC
# Initialize total liquid water results
total_liquid_water = []

# Ensure CWC and RWC datasets are aligned by time
for rwc_entry in rain_water_content:  # Loop through RWC entries
    matching_time = rwc_entry['Time']
    matching_date = rwc_entry['Date']

    # Find the corresponding CWC entry by matching time and date
    matching_cwc = next((entry for entry in in_cloud_concentrations if entry['Time'] == matching_time and entry['Date'] == matching_date), None)

    if matching_cwc:
        # Sum CWC and RWC to calculate total liquid water
        cwc_val = matching_cwc['CWC']  # Cloud water content (in kg/m³)
        rwc_val = rwc_entry['RWC']  # Rain water content (in kg/m³)
        total_liquid = cwc_val + rwc_val

        # Append the total liquid water result
        total_liquid_water.append({
            'Date': matching_date,
            'Time': matching_time,
            'Leg_start': rwc_entry['Leg_start'],
            'Leg_stop': rwc_entry['Leg_stop'],
            'CWC': cwc_val,
            'RWC': rwc_val,
            'Total_Liquid_Water': total_liquid  # Sum of CWC and RWC
        })

# Debugging: Check a few entries of total liquid water
print(f"Number of total liquid water entries: {len(total_liquid_water)}")
print(f"First 5 entries: {total_liquid_water[:5]}")

#%%
#add the Nc + Nr for total concentration
total_combined_concentration = []

# Ensure in-cloud and rain concentrations are aligned by time
for in_cloud_entry in in_cloud_concentrations:  # Loop through in-cloud concentration entries
    matching_time = in_cloud_entry['Time']
    matching_date = in_cloud_entry['Date']

    matching_rain = next((entry for entry in rain_concentrations if entry['Time'] == matching_time and entry['Date'] == matching_date), None)
    
    
    if matching_rain:
        rain_val = matching_rain['Total_Concentration']
        inc_val = in_cloud_entry['Total_Concentration']  # In-cloud concentration (/cm³)
        combined_conc = inc_val + rain_val

        # Append the total combined concentration result
        total_combined_concentration.append({
            'Date': matching_date,
            'Time': matching_time,
            'Leg_start': matching_rain['Leg_start'],
            'Leg_stop': matching_rain['Leg_stop'],
            'In_Cloud_Concentration': inc_val,
            'Rain_Concentration': rain_val,
            'Total_Combined_Concentration': combined_conc  # Sum of in-cloud and rain concentrations
        })

# Debugging: Check a few entries of total combined concentration
print(f"Number of total combined concentration entries: {len(total_combined_concentration)}")
print(f"First 5 entries: {total_combined_concentration[:5]}")

#%% 

# Extract values as lists without converting to NumPy arrays
concentration = [entry['Total_Combined_Concentration'] for entry in total_combined_concentration]
total_liquid_water_values = [entry['Total_Liquid_Water'] for entry in total_liquid_water]  
rain_water_content_values = [entry['RWC'] for entry in total_liquid_water]  

# Compute RWC percentage safely (avoid division by zero)
rwc_percentage = []
for rwc, total in zip(rain_water_content_values, total_liquid_water_values):
    if total > 0:
        rwc_percentage.append((rwc / total) * 100)  # Convert to percentage
    else:
        rwc_percentage.append(0)  # Avoid division by zero

# Define 2D histogram bins
bins = 100  

# Create the 2D histogram shading by RWC percentage
plt.figure(figsize=(8, 6))
hist, xedges, yedges, img = plt.hist2d(concentration, total_liquid_water_values, bins=bins, 
                                       weights=rwc_percentage, cmap='RdBu_r', cmin=1)

# Log scales for axes
plt.xscale('log')
plt.yscale('log')

# Labels and title
plt.xlabel('Nr+Nc /cm³ (log scale)', fontsize=16, fontweight='bold')
plt.ylabel('LWC g/m³ (log scale)', fontsize=16, fontweight='bold')
plt.title('RWC Percentage of Total Liquid Water', fontsize=18, fontweight='bold')

# Colorbar settings (linear scale from 0 to 100%)
cbar = plt.colorbar(img)
cbar.set_label("Rainwater % of Total Liquid Water", fontsize=14)  

# Grid for readability
plt.grid(which="both", linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()

# Show plot
plt.show()

# %%
# Create histogram bins
bins = 100

# Compute histogram counts
counts, xedges, yedges = np.histogram2d(concentration, total_liquid_water_values, bins=bins)

# Compute weighted sum of RWC%
sum_rwc, _, _ = np.histogram2d(concentration, total_liquid_water_values, bins=bins, weights=rwc_percentage)

# Compute mean RWC% per bin (avoid division by zero)
mean_rwc = np.divide(sum_rwc, counts, out=np.zeros_like(sum_rwc), where=counts > 0)

# Plot 2D color plot using the corrected mean values
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(xedges, yedges, mean_rwc.T, cmap='RdBu_r', vmin=1, vmax=100)

# Add colorbar
cbar = plt.colorbar(img)
cbar.set_label("RWC/LWC (%)", fontsize=14)
plt.ylim(10**-2, 10**0.2)  # Adjust if needed

plt.xscale('log')
plt.yscale('log')
plt.tick_params(axis='both', which='major', labelsize=16, width=3, length=8)  # Major ticks
plt.tick_params(axis='both', which='minor', labelsize=14, width=2, length=5)
plt.xlabel('Nr+Nc /cm³', fontsize=16, fontweight='bold')
plt.ylabel('LWC g/m³', fontsize=16, fontweight='bold')
plt.title('CAS in-cloud January - June 2022', fontsize=18, fontweight='bold')
plt.grid(which="both", linestyle='--', linewidth=0.5, alpha=0.7)

plt.show()

# %%
plt.figure(figsize=(8, 6))
plt.scatter(concentration, total_liquid_water_values, c=rwc_percentage, 
            cmap='RdBu_r', alpha=0.3, edgecolors='none', s=10)

plt.xscale('log')
plt.yscale('log')

cbar = plt.colorbar()
cbar.set_label("RWC/LWC (%)")

plt.tick_params(axis='both', which='major', labelsize=16, width=3, length=8)  # Major ticks
plt.tick_params(axis='both', which='minor', labelsize=14, width=2, length=5)
plt.xlabel('Nr+Nc /cm³', fontsize=16, fontweight='bold')
plt.ylabel('LWC g/m³', fontsize=16, fontweight='bold')
plt.title('CAS in-cloud January - June 2022', fontsize=18, fontweight='bold')
plt.grid(which="both", linestyle='--', linewidth=0.5, alpha=0.7)

plt.show()

# %%
##Calculating effective radius
# Loop through total_combined_concentration and find matching Reff_CAS
for idx, entry in enumerate(total_combined_concentration):  
    matching_time = entry['Time']
    matching_date = entry['Date']

    # Find which date this entry corresponds to in dates_legs
    cas_flight = None
    for i in range(len(dates_legs)):
        if dates_legs[i] == matching_date:
            cas_flight = CAS[i]  # Get the correct CAS flight dataset
            break
    
    if cas_flight is not None:
        # Convert CAS times to numeric if not already
        cas_flight['Time_mid'] = pd.to_numeric(cas_flight['Time_mid'], errors='coerce')
        
        # Find the matching row in CAS_flight
        matching_cas_row = cas_flight.loc[cas_flight['Time_mid'] == matching_time]
        
        if not matching_cas_row.empty:
            reff_value = matching_cas_row['Reff_CAS'].values[0]  # Extract Reff_CAS
        else:
            reff_value = np.nan  # No match found, store NaN
    else:
        reff_value = np.nan  # If no matching CAS flight is found

    # **Store Reff_CAS in total_combined_concentration**
    entry['Reff_CAS'] = reff_value  

# Debugging: Check if values are stored
print(f"Extracted {len(total_combined_concentration)} effective radii.")
print(f"First 5 entries with radii: {[entry['Reff_CAS'] for entry in total_combined_concentration[:5]]}")


# %%
# Compute Deff_CAS from Reff_CAS
for entry in total_combined_concentration:
    if 'Reff_CAS' in entry:  # Ensure the radius exists before multiplying
        entry['Deff_CAS'] = entry['Reff_CAS'] * 2  # Convert radius to diameter
#%%
# Extract Deff_CAS from total_combined_concentration
deff_values = [entry['Deff_CAS'] for entry in total_combined_concentration if 'Deff_CAS' in entry]

# Debugging: Check extracted Deff_CAS values
print(f"Extracted {len(deff_values)} effective diameter values.")
print(f"First 5 values: {deff_values[:5]}")

# %%
# Filter all data arrays together
filtered_data = [
    (conc, lwc, deff, rwc) 
    for conc, lwc, deff, rwc in zip(concentration, total_liquid_water_values, deff_values, rwc_percentage)
    if np.isfinite(conc) and np.isfinite(lwc) and np.isfinite(deff) and np.isfinite(rwc)
]

# Unpack filtered data
filtered_concentration, filtered_liquid_water_values, filtered_deff_values, filtered_rwc_percentage = zip(*filtered_data)

# %%
#Scatterplot with diameter contours 
fig, ax = plt.subplots(figsize=(8, 6))

sc = ax.scatter(filtered_concentration, filtered_liquid_water_values, c=filtered_rwc_percentage, 
                cmap='RdBu_r', alpha=0.6, edgecolors='none', s=15)  # Slightly bigger points

contour_levels = np.linspace(np.nanmin(filtered_deff_values), 
                             np.nanmax(filtered_deff_values), 10)  # 10 contour levels from min to max


contour = ax.tricontourf(filtered_concentration, filtered_liquid_water_values, filtered_deff_values, 
                         levels=contour_levels, cmap="Greys", alpha=0.4)  # Gray shading for larger diameters
cbar_deff = plt.colorbar(contour, ax=ax, pad=0.08)
cbar_deff.set_label("Effective Diameter (μm)", fontsize=12, labelpad=10)
ax.set_xscale('log')
ax.set_yscale('log')
cbar_rwc = plt.colorbar(sc, ax=ax, pad=0.02)
cbar_rwc.set_label("RWC/LWC (%)", fontsize=12, labelpad=10)
ax.set_xlabel('Nr+Nc /cm³', fontsize=16, fontweight='bold')
ax.set_ylabel('LWC g/m³', fontsize=16, fontweight='bold')
ax.set_title('CAS In-Cloud January-June 2022', fontsize=18, fontweight='bold')
ax.grid(which="both", linestyle='--', linewidth=0.5, alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=16, width=3, length=8)  # Major ticks
plt.tick_params(axis='both', which='minor', labelsize=14, width=2, length=5)
plt.show()

# %%
# Boxplots
df_CAS_rainwater = pd.DataFrame({
    'RWC': [entry['RWC'] for entry in total_liquid_water],
    'Deff_CAS': [entry['Deff_CAS'] for entry in total_combined_concentration]  # Make sure 'Deff_CDP' exists
})

# Ensure no NaN values before binning
df_CAS_rainwater = df_CAS_rainwater.dropna(subset=['RWC', 'Deff_CAS'])

# Define RWC bins
bins = [0, 0.001, 0.01, 0.1, 1]  # Light drizzle to heavy rain
labels = ['<0.001 g/m³', '0.001-0.01 g/m³', '0.01-0.1 g/m³', '>0.1 g/m³']

# Assign bin labels
df_CAS_rainwater['RWC_Category'] = pd.cut(df_CAS_rainwater['RWC'], bins=bins, labels=labels, include_lowest=True)

# Verify if RWC_Category was successfully created
print(df_CAS_rainwater[['RWC', 'RWC_Category']].head())

# Remove NaN values again to ensure no issues
df_CAS_rainwater = df_CAS_rainwater.dropna()

# Check if there are any valid rows left after filtering
if df_CAS_rainwater.empty:
    print("⚠️ No valid data after filtering. Check RWC values!")
else:
    # Create the Boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='RWC_Category', y='Deff_CAS', data=df_CAS_rainwater, palette='Blues')

    # Customize plot
    plt.xlabel('RWC g/m^3', fontsize=14, fontweight='bold')
    plt.ylabel('Effective Diameter (μm)', fontsize=14, fontweight='bold')
    plt.title('CAS in-cloud January-June 2022', fontsize=16, fontweight='bold')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=12, width=3, length=8)
    plt.tick_params(axis='both', which='minor', labelsize=12, width=2, length=5)

    # Show plot
    plt.show()


# %%
#Violin plots 
df_CAS_rainwater = pd.DataFrame({
    'RWC': [entry['RWC'] for entry in total_liquid_water],
    'Deff_CAS': [entry['Deff_CAS'] for entry in total_combined_concentration]  
})

df_CAS_rainwater = df_CAS_rainwater.dropna(subset=['RWC', 'Deff_CAS'])
bins = [0, 0.001, 0.01, 0.1, 1]  
labels = [r'RWC < 0.001 g m$^{-3}$', 
          r'0.001 - 0.01 g m$^{-3}$', 
          r'0.01 - 0.1 g m$^{-3}$', 
          r'RWC > 0.1 g m$^{-3}$']
# Assign bin labels
df_CAS_rainwater['RWC_Category'] = pd.cut(df_CAS_rainwater['RWC'], bins=bins, labels=labels, include_lowest=True)

# Drop remaining NaNs
df_CAS_rainwater = df_CAS_rainwater.dropna()

# Check if there are valid rows left
if df_CAS_rainwater.empty:
    print(" No valid data after filtering. Check RWC values!")
else:
    # ---- PLOT: Violin Plot ----
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='RWC_Category', y='Deff_CAS', data=df_CAS_rainwater, palette='Blues', inner='box', scale='width')

    # Customize plot
    plt.xlabel(r'RWC (g m$^{-3}$)', fontsize=14, fontweight='bold')
    plt.ylabel('Effective Diameter (μm)', fontsize=14, fontweight='bold')
    plt.title('CAS in-cloud January - June 2022', fontsize=16, fontweight='bold')

    # Formatting
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=12, width=3, length=8)
    plt.tick_params(axis='both', which='minor', labelsize=12, width=2, length=5)

    # Show plot
    plt.show()

#%%
import matplotlib.colors as mcolors
# Extract relevant values
concentration = np.array([entry['Total_Combined_Concentration'] for entry in total_combined_concentration])
total_liquid_water_values = np.array([entry['Total_Liquid_Water'] for entry in total_liquid_water])
rain_water_content_values = np.array([entry['RWC'] for entry in total_liquid_water])

# Compute RWC percentage safely
rwc_percentage = np.divide(rain_water_content_values, total_liquid_water_values, 
                           out=np.full_like(rain_water_content_values, np.nan), where=total_liquid_water_values > 0) * 100  

# Define log-spaced bins
num_bins = 5
x_bins = np.logspace(np.log10(min(concentration)), np.log10(max(concentration)), num_bins)
y_bins = np.logspace(np.log10(min(total_liquid_water_values)), np.log10(max(total_liquid_water_values)), num_bins)

# Compute histogram bins
counts, xedges, yedges = np.histogram2d(concentration, total_liquid_water_values, bins=[x_bins, y_bins])

# Compute **average RWC% per bin**
sum_rwc, _, _ = np.histogram2d(concentration, total_liquid_water_values, bins=[x_bins, y_bins], weights=rwc_percentage)
mean_rwc = np.divide(sum_rwc, counts, out=np.full_like(sum_rwc, np.nan), where=counts > 0)  

# ---- Create a Mask for No-Data Regions ----
masked_rwc = np.ma.masked_where(np.isnan(mean_rwc), mean_rwc)

# ---- Define Custom Colormap ----
cmap = plt.get_cmap('RdBu_r')
cmap.set_bad(color='gray')  # No-data areas will now explicitly appear gray

# ---- Plot the Main Heatmap ----
plt.figure(figsize=(8, 6))
norm = mcolors.Normalize(vmin=1, vmax=100)
img = plt.pcolormesh(xedges, yedges, masked_rwc.T, cmap=cmap, norm=norm, shading='auto')

# Colorbar formatting
cbar = plt.colorbar(img)
cbar.set_label("RWC/LWC %", fontsize=20, fontweight='bold')
cbar.ax.tick_params(labelsize=15, width=2, length=5) 
for t in cbar.ax.get_yticklabels():  
    t.set_fontweight('bold')

# Axis formatting
plt.xscale('log')
plt.yscale('log')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=20, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=20, fontweight='bold')
plt.title('CAS (in cloud) \nJanuary-June 2022', fontsize=20, fontweight='bold')

plt.tight_layout()
plt.show()
#%%
#Just our selected region of N and LWC without averaging 


# # Extract relevant values
# concentration = np.array([entry['Total_Combined_Concentration'] for entry in total_combined_concentration])
# total_liquid_water_values = np.array([entry['Total_Liquid_Water'] for entry in total_liquid_water])
# rain_water_content_values = np.array([entry['RWC'] for entry in total_liquid_water])

# # Compute RWC percentage safely (avoiding division by zero)
# rwc_percentage = np.divide(rain_water_content_values, total_liquid_water_values, 
#                            out=np.full_like(rain_water_content_values, np.nan), where=total_liquid_water_values > 0) * 100  

# # Define the region of interest (LWC: 0.1 to 0.3, Nr+Nc: 50 to 200)
# box_x_min, box_x_max = 50, 200  # Nr+Nc range
# box_y_min, box_y_max = 0.1, 0.3  # LWC range

# # Filter data to only include points in this range
# mask = (concentration >= box_x_min) & (concentration <= box_x_max) & \
#        (total_liquid_water_values >= box_y_min) & (total_liquid_water_values <= box_y_max)

# filtered_concentration = concentration[mask]
# filtered_lwc = total_liquid_water_values[mask]
# filtered_rwc = rwc_percentage[mask]

# # Define bins for the new heatmap (same log scaling)
# num_bins = 11
# x_bins = np.logspace(np.log10(box_x_min), np.log10(box_x_max), num_bins)
# y_bins = np.logspace(np.log10(box_y_min), np.log10(box_y_max), num_bins)

# # Compute histogram for **raw data density**, not averages
# counts, xedges, yedges = np.histogram2d(filtered_concentration, filtered_lwc, bins=[x_bins, y_bins])

# # Mask bins with no data
# masked_counts = np.ma.masked_where(counts == 0, counts)

# # ---- Define Custom Colormap ----
# cmap = plt.get_cmap('RdBu_r')
# cmap.set_bad(color='gray')  # No-data areas explicitly appear gray

# # ---- Plot the New Heatmap ----
# plt.figure(figsize=(8, 6))
# norm = mcolors.Normalize(vmin=1, vmax=np.nanmax(masked_counts))  # Normalize based on valid counts
# img = plt.pcolormesh(xedges, yedges, masked_counts.T, cmap=cmap, norm=norm, shading='auto')

# # Colorbar formatting
# cbar = plt.colorbar(img)
# cbar.set_label("Data Density", fontsize=14, fontweight='bold')
# cbar.ax.tick_params(labelsize=12, width=2, length=5) 
# for t in cbar.ax.get_yticklabels():  
#     t.set_fontweight('bold')

# # Axis formatting
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=16, fontweight='bold')
# plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=16, fontweight='bold')
# plt.title('CAS in-cloud January-June 2022 (Zoomed Region)', fontsize=18, fontweight='bold')

# plt.tight_layout()
# plt.show()


# %%
#50 bins, average of all the RWC/LWC in each bin 


# # Extract values
# concentration = np.array([entry['Total_Combined_Concentration'] for entry in total_combined_concentration])
# total_liquid_water_values = np.array([entry['Total_Liquid_Water'] for entry in total_liquid_water])
# rain_water_content_values = np.array([entry['RWC'] for entry in total_liquid_water])

# # Compute RWC percentage safely
# rwc_percentage = np.divide(rain_water_content_values, total_liquid_water_values, 
#                            out=np.zeros_like(rain_water_content_values), where=total_liquid_water_values > 0) * 100

# # Define log-spaced bins
# num_bins = 50  
# x_bins = np.logspace(np.log10(min(concentration)), np.log10(max(concentration)), num_bins)
# y_bins = np.logspace(np.log10(min(total_liquid_water_values)), np.log10(max(total_liquid_water_values)), num_bins)

# # Compute histogram bins
# counts, xedges, yedges = np.histogram2d(concentration, total_liquid_water_values, bins=[x_bins, y_bins])

# # Compute mean RWC% per bin (avoid division by zero)
# sum_rwc, _, _ = np.histogram2d(concentration, total_liquid_water_values, bins=[x_bins, y_bins], weights=rwc_percentage)
# mean_rwc = np.divide(sum_rwc, counts, out=np.full_like(sum_rwc, -1, dtype=float), where=counts > 0)  # -1 for empty bins
# cmap = plt.get_cmap("RdBu_r") 
# cmap.set_under("silver")
# plt.figure(figsize=(8, 6))
# img = plt.pcolormesh(xedges, yedges, mean_rwc.T, cmap=cmap, norm=mcolors.Normalize(vmin=1, vmax=100), shading='auto')
# cbar = plt.colorbar(img)
# cbar.set_label("RWC / LWC (%)", fontsize=14)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=16, fontweight='bold')
# plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=16, fontweight='bold')
# plt.title('CAS in-cloud January-June 2022', fontsize=18, fontweight='bold')
# plt.tight_layout()
# plt.show()

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
plt.title('CAS (in cloud)\n January-June 2022\n RWC as a function of number concentration', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
#reducing speckling

concentration = np.array([entry['Total_Combined_Concentration'] for entry in total_combined_concentration])
total_liquid_water_values = np.array([entry['Total_Liquid_Water'] for entry in total_liquid_water])
rain_water_content_values = np.array([entry['RWC'] for entry in total_liquid_water])

num_bins = 5  # Adjusted to give approximately 4x4 bins
x_bins = np.logspace(np.log10(1), np.log10(max(concentration)), num_bins)
y_bins = np.logspace(np.log10(min(total_liquid_water_values)), np.log10(max(total_liquid_water_values)), num_bins)

sum_rwc, xedges, yedges = np.histogram2d(concentration, total_liquid_water_values, bins=[x_bins, y_bins], weights=rain_water_content_values)
sum_lwc, _, _ = np.histogram2d(concentration, total_liquid_water_values, bins=[x_bins, y_bins], weights=total_liquid_water_values)
counts, _, _ = np.histogram2d(concentration, total_liquid_water_values, bins=[x_bins, y_bins])

avg_rwc = np.divide(sum_rwc, counts, out=np.full_like(sum_rwc, np.nan), where=counts > 0)
avg_lwc = np.divide(sum_lwc, counts, out=np.full_like(sum_lwc, np.nan), where=counts > 0)
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
plt.title('CAS (in cloud)\n January-June 2022\n RWC as a function of number concentration', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()

#%%
#trying to now perform averaging in the selected region as opposed to averaging and then picking the region

# from scipy.stats import ttest_ind

# # Extract relevant values
# concentration = np.array([entry['Total_Combined_Concentration'] for entry in total_combined_concentration])
# total_liquid_water_values = np.array([entry['Total_Liquid_Water'] for entry in total_liquid_water])
# rain_water_content_values = np.array([entry['RWC'] for entry in total_liquid_water])

# # Define the region of interest (LWC: 0.1 to 0.3, Nr+Nc: 50 to 200)
# box_x_min, box_x_max = 50, 200  
# box_y_min, box_y_max = 0.1, 0.3  

# # Filter data to only include points in this range
# mask = (concentration >= box_x_min) & (concentration <= box_x_max) & \
#        (total_liquid_water_values >= box_y_min) & (total_liquid_water_values <= box_y_max)

# filtered_concentration = concentration[mask]
# filtered_lwc = total_liquid_water_values[mask]
# filtered_rwc = rain_water_content_values[mask]

# # Define bins for this region
# num_bins = 8  
# x_bins = np.logspace(np.log10(box_x_min), np.log10(box_x_max), num_bins)
# y_bins = np.logspace(np.log10(box_y_min), np.log10(box_y_max), num_bins)

# # Compute sum of RWC and LWC per bin
# sum_rwc, xedges, yedges = np.histogram2d(filtered_concentration, filtered_lwc, bins=[x_bins, y_bins], weights=filtered_rwc)
# sum_lwc, _, _ = np.histogram2d(filtered_concentration, filtered_lwc, bins=[x_bins, y_bins], weights=filtered_lwc)
# counts, _, _ = np.histogram2d(filtered_concentration, filtered_lwc, bins=[x_bins, y_bins])

# # Compute **means** and **standard deviations**
# avg_rwc = np.divide(sum_rwc, counts, out=np.full_like(sum_rwc, np.nan), where=counts > 0)  
# avg_lwc = np.divide(sum_lwc, counts, out=np.full_like(sum_lwc, np.nan), where=counts > 0)  
# rwc_lwc_ratio = np.divide(avg_rwc, avg_lwc, out=np.full_like(avg_rwc, np.nan), where=avg_lwc > 0) * 100  

# # Compute **standard deviation of RWC/LWC ratio** in each bin
# std_rwc, _, _ = np.histogram2d(filtered_concentration, filtered_lwc, bins=[x_bins, y_bins], weights=(filtered_rwc - np.nanmean(filtered_rwc))**2)
# std_lwc, _, _ = np.histogram2d(filtered_concentration, filtered_lwc, bins=[x_bins, y_bins], weights=(filtered_lwc - np.nanmean(filtered_lwc))**2)
# std_rwc_lwc_ratio = np.sqrt(std_rwc / (counts - 1))  # Standard deviation per bin

# # Mask bins with no data
# masked_rwc_lwc_ratio = np.ma.masked_where(np.isnan(rwc_lwc_ratio), rwc_lwc_ratio)

# # ---- Plot the New Heatmap ----
# plt.figure(figsize=(8, 6))
# norm = mcolors.Normalize(vmin=1, vmax=100)
# img = plt.pcolormesh(xedges, yedges, masked_rwc_lwc_ratio.T, cmap="RdBu_r", norm=norm, shading='auto')

# # Gray out empty bins
# gray_mask = np.isnan(rwc_lwc_ratio)  
# gray_values = np.full_like(rwc_lwc_ratio, np.nan)
# gray_values[gray_mask] = 1  
# plt.pcolormesh(xedges, yedges, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)

# # Colorbar formatting
# cbar = plt.colorbar(img)
# cbar.set_label("RWC/LWC %", fontsize=14, fontweight='bold')  
# cbar.ax.tick_params(labelsize=12, width=2, length=5) 
# for t in cbar.ax.get_yticklabels():  
#     t.set_fontweight('bold')

# # Axis formatting
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Nr+Nc /cm³', fontsize=16, fontweight='bold')
# plt.ylabel('LWC g/m³', fontsize=16, fontweight='bold')
# plt.title('CAS in-cloud (Zoomed Region: 50-200 cm⁻³, 0.1-0.3 g/m³)', fontsize=18, fontweight='bold')
# plt.tight_layout()
# plt.show()

#%%
#Density of observations for entire region 

concentration = np.array([entry['Total_Combined_Concentration'] for entry in total_combined_concentration])
total_liquid_water_values = np.array([entry['Total_Liquid_Water'] for entry in total_liquid_water])

# Define bin edges
num_bins = 5
x_bins = np.logspace(np.log10(1), np.log10(max(concentration)), num_bins)
y_bins = np.logspace(np.log10(min(total_liquid_water_values)), np.log10(max(total_liquid_water_values)), num_bins)

# Compute density of observations
density_counts, xedges, yedges = np.histogram2d(concentration, total_liquid_water_values, bins=[x_bins, y_bins])

# Plot the density heatmap
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(xedges, yedges, density_counts.T, cmap="plasma", shading='auto', 
                      norm=mcolors.LogNorm(vmax=np.max(density_counts) * 1.1))
cbar = plt.colorbar(img)
cbar.set_label("Density of Observations", fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=18, width=2, length=5)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight('bold')

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=18, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=18, fontweight='bold')
plt.title('CAS in-cloud January-June 2022', fontsize=18, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=18, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=18, width=2, length=5)
plt.tight_layout()
plt.show()
#%%
#adding a black box 
import matplotlib.colors as mcolors

# Extract data
concentration = np.array([entry['Total_Combined_Concentration'] for entry in total_combined_concentration])
total_liquid_water_values = np.array([entry['Total_Liquid_Water'] for entry in total_liquid_water])

# Define bin edges
num_bins = 5
x_bins = np.logspace(np.log10(1), np.log10(max(concentration)), num_bins)
y_bins = np.logspace(np.log10(min(total_liquid_water_values)), np.log10(max(total_liquid_water_values)), num_bins)

# Compute density of observations
density_counts, xedges, yedges = np.histogram2d(concentration, total_liquid_water_values, bins=[x_bins, y_bins])

# Plot the density heatmap
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(xedges, yedges, density_counts.T, cmap="plasma", shading='auto', 
                      norm=mcolors.LogNorm(vmax=np.max(density_counts) * 1.1))
cbar = plt.colorbar(img)
cbar.set_label("Density of Observations", fontsize=17, fontweight='bold')
cbar.ax.tick_params(labelsize=19, width=2, length=5)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight('bold')

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.title('CAS (in cloud)\n Density of Observations\n January-June 2022', fontsize=18, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)

# ✅ Add black rectangle (box) for 0.1-0.3 LWC and 50-200 concentration
x_min, x_max = 50, 200  # X-axis (Nr+Nc cm⁻³)
y_min, y_max = 0.1, 0.3  # Y-axis (LWC g/m³)
plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                  edgecolor='black', facecolor='none', linewidth=2.5))

plt.tight_layout()
plt.show()
#%%

# Extract data
concentration = np.array([entry['Total_Combined_Concentration'] for entry in total_combined_concentration])
total_liquid_water_values = np.array([entry['Total_Liquid_Water'] for entry in total_liquid_water])

# Define bin edges
num_bins = 5
x_bins = np.logspace(np.log10(1), np.log10(max(concentration)), num_bins)
y_bins = np.logspace(np.log10(min(total_liquid_water_values)), np.log10(max(total_liquid_water_values)), num_bins)

density_counts, xedges, yedges = np.histogram2d(concentration, total_liquid_water_values, bins=[x_bins, y_bins])

gray_mask = np.zeros_like(density_counts, dtype=float)
gray_mask[density_counts == 0] = 1  # Mark bins with no data for masking

plt.figure(figsize=(8, 6))
img = plt.pcolormesh(xedges, yedges, density_counts.T, cmap="plasma", shading='auto', 
                      norm=mcolors.LogNorm(vmin=1, vmax=np.max(density_counts) * 1.1))

mask = np.ma.masked_where(gray_mask == 0, gray_mask)  # Mask valid data, leaving only missing bins
plt.pcolormesh(xedges, yedges, mask.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)

cbar = plt.colorbar(img)
cbar.set_label("Density of Observations", fontsize=17, fontweight='bold')
cbar.ax.tick_params(labelsize=19, width=2, length=5)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight('bold')

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.title('CAS (in cloud)\n Density of Observations\n January-June 2022', fontsize=18, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)

x_min, x_max = 50, 200  # X-axis (Nr+Nc cm⁻³)
y_min, y_max = 0.1, 0.3  # Y-axis (LWC g/m³)
plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                  edgecolor='black', facecolor='none', linewidth=2.5))

plt.tight_layout()
plt.show()
#%%

#%%
#basic stats summary of whole dataset

concentration = np.array([entry['Total_Combined_Concentration'] for entry in total_combined_concentration])
total_liquid_water_values = np.array([entry['Total_Liquid_Water'] for entry in total_liquid_water])
rain_water_content_values = np.array([entry['RWC'] for entry in total_liquid_water])
mean_concentration = np.nanmean(concentration)  # Average Nr+Nc over all data
mean_lwc = np.nanmean(total_liquid_water_values)  # Average LWC over all data
mean_rwc = np.nanmean(rain_water_content_values)  # Average RWC over all data
rwc_lwc_ratio_global = (mean_rwc / mean_lwc) * 100 if mean_lwc > 0 else np.nan
print(f"Average Nr+Nc (Total Droplet Concentration): {mean_concentration:.2f} cm⁻³")
print(f"Average LWC (Liquid Water Content): {mean_lwc:.4f} g/m³")
print(f"Average RWC (Rain Water Content): {mean_rwc:.4f} g/m³")
print(f"Global RWC/LWC Ratio: {rwc_lwc_ratio_global:.2f} %")
#%%
#basic stats summary of bins

concentration = np.array([entry['Total_Combined_Concentration'] for entry in total_combined_concentration])
total_liquid_water_values = np.array([entry['Total_Liquid_Water'] for entry in total_liquid_water])
rain_water_content_values = np.array([entry['RWC'] for entry in total_liquid_water])
num_bins = 11
x_bins = np.logspace(np.log10(1), np.log10(max(concentration)), num_bins)
y_bins = np.logspace(np.log10(min(total_liquid_water_values)), np.log10(max(total_liquid_water_values)), num_bins)
sum_rwc, xedges, yedges = np.histogram2d(concentration, total_liquid_water_values, bins=[x_bins, y_bins], weights=rain_water_content_values)
sum_lwc, _, _ = np.histogram2d(concentration, total_liquid_water_values, bins=[x_bins, y_bins], weights=total_liquid_water_values)
counts, _, _ = np.histogram2d(concentration, total_liquid_water_values, bins=[x_bins, y_bins])

# Compute bin-wise averages
avg_concentration = np.divide(sum_rwc, counts, out=np.full_like(sum_rwc, np.nan), where=counts > 0)  # Avg N in each bin
avg_lwc = np.divide(sum_lwc, counts, out=np.full_like(sum_lwc, np.nan), where=counts > 0)  # Avg LWC in each bin
avg_rwc = np.divide(sum_rwc, counts, out=np.full_like(sum_rwc, np.nan), where=counts > 0)  # Avg RWC in each bin
rwc_lwc_ratio_bins = np.divide(avg_rwc, avg_lwc, out=np.full_like(avg_rwc, np.nan), where=avg_lwc > 0) * 100  # Bin-wise RWC/LWC%

# Print summary statistics for bins
print(" **Bin-wise Summary:**")
for i in range(len(xedges) - 1):
    for j in range(len(yedges) - 1):
        print(f"Bin {i+1},{j+1}:")
        print(f"   Avg Nr+Nc: {avg_concentration[i, j]:.2f} cm⁻³")
        print(f"   Avg LWC: {avg_lwc[i, j]:.4f} g/m³")
        print(f"   Avg RWC: {avg_rwc[i, j]:.4f} g/m³")
        print(f"   RWC/LWC Ratio: {rwc_lwc_ratio_bins[i, j]:.2f}%")
        print("-" * 40)
#%%
#make a table of stats
num_bins_x = len(xedges) - 1
num_bins_y = len(yedges) - 1
bin_labels = [f"Bin {i+1}" for i in range(num_bins_x)]
df_rwc_lwc = pd.DataFrame(rwc_lwc_ratio_bins, index=bin_labels, columns=bin_labels)
df_rwc_lwc = df_rwc_lwc.applymap(lambda x: f"{x:.2f}%" if not np.isnan(x) else "")

fig, ax = plt.subplots(figsize=(10, 6)) 
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=df_rwc_lwc.values, 
                 colLabels=df_rwc_lwc.columns, 
                 rowLabels=df_rwc_lwc.index, 
                 cellLoc='center', 
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2) 
plt.title("Bin-wise RWC/LWC Ratio (%)", fontsize=14, fontweight='bold')
plt.show()

#%%
#density of observations from 0.1 to 0.3 LWC and 50 to 200 /cm3 

num_bins = 5
x_bins = np.logspace(np.log10(1), np.log10(max(concentration)), num_bins)
y_bins = np.logspace(np.log10(min(total_liquid_water_values)), np.log10(max(total_liquid_water_values)), num_bins)
x_min, x_max = 50, 200  # Nr+Nc range
y_min, y_max = 0.1, 0.3  # LWC range
mask = (concentration >= x_min) & (concentration <= x_max) & \
       (total_liquid_water_values >= y_min) & (total_liquid_water_values <= y_max)

filtered_concentration = concentration[mask]

filtered_lwc = total_liquid_water_values[mask]
density_counts, xedges, yedges = np.histogram2d(filtered_concentration, filtered_lwc, bins=[x_bins, y_bins])

# ---- PLOT 1: Density of Observations Heatmap ----
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(xedges, yedges, density_counts.T, cmap="plasma", shading='auto', 
                      norm=mcolors.LogNorm(vmax=np.max(density_counts) * 1.1))
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

# ---- PLOT 2: Histogram of Observations in Log-Space ----
plt.figure(figsize=(8, 6))
plt.hist(filtered_concentration, bins=x_bins, color="darkred", alpha=0.7, log=True)
plt.xscale('log')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=16, fontweight='bold')
plt.ylabel('Frequency', fontsize=16, fontweight='bold')
plt.title('CAS in-cloud January-June 2022', fontsize=18, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=12, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=12, width=2, length=5)
plt.xlim(10**1.5, 10**2.5)
plt.tight_layout()
plt.show()
#%%
#Basic stats about this region

#%%
#adding the black box to selected region

# Data extraction
concentration = np.array([entry['Total_Combined_Concentration'] for entry in total_combined_concentration])
total_liquid_water_values = np.array([entry['Total_Liquid_Water'] for entry in total_liquid_water])
rain_water_content_values = np.array([entry['RWC'] for entry in total_liquid_water])

# Define bins
num_bins = 5
x_bins = np.logspace(np.log10(1), np.log10(max(concentration)), num_bins)
y_bins = np.logspace(np.log10(min(total_liquid_water_values)), np.log10(max(total_liquid_water_values)), num_bins)

# Compute histograms
sum_rwc, xedges, yedges = np.histogram2d(concentration, total_liquid_water_values, bins=[x_bins, y_bins], weights=rain_water_content_values)
sum_lwc, _, _ = np.histogram2d(concentration, total_liquid_water_values, bins=[x_bins, y_bins], weights=total_liquid_water_values)
counts, _, _ = np.histogram2d(concentration, total_liquid_water_values, bins=[x_bins, y_bins])

# Compute averages
avg_rwc = np.divide(sum_rwc, counts, out=np.full_like(sum_rwc, np.nan), where=counts > 0)  
avg_lwc = np.divide(sum_lwc, counts, out=np.full_like(sum_rwc, np.nan), where=counts > 0)  
rwc_lwc_ratio = np.divide(avg_rwc, avg_lwc, out=np.full_like(avg_rwc, np.nan), where=avg_lwc > 0) * 100  

# Mask NaN values
masked_rwc_lwc_ratio = np.ma.masked_where(np.isnan(rwc_lwc_ratio), rwc_lwc_ratio)

# Plot heatmap
plt.figure(figsize=(8, 6))
norm = mcolors.Normalize(vmin=1, vmax=100)
img = plt.pcolormesh(xedges, yedges, masked_rwc_lwc_ratio.T, cmap="RdBu_r", norm=norm, shading='auto')

# Gray out NaN areas
gray_mask = np.isnan(rwc_lwc_ratio)  
gray_values = np.full_like(rwc_lwc_ratio, np.nan)
gray_values[gray_mask] = 1  
plt.pcolormesh(xedges, yedges, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)
cbar = plt.colorbar(img)
cbar.set_label("RWC / LWC (%)", fontsize=17, fontweight='bold') 
cbar.ax.tick_params(labelsize=19, width=2, length=5)  # Adjust tick size
for t in cbar.ax.get_yticklabels():  
    t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.title('CAS (in cloud)\n January-June 2022\n RWC as a function of number concentration', fontsize=18, fontweight='bold')
box_x_min, box_x_max = 50, 200 
box_y_min, box_y_max = 0.1, 0.3 
plt.plot([box_x_min, box_x_max, box_x_max, box_x_min, box_x_min],
         [box_y_min, box_y_min, box_y_max, box_y_max, box_y_min], 
         color='black', linewidth=3) 

plt.tight_layout()
plt.show()
# %%
#We need to figure out how to incporate GCCN into this RWC vs concentration relationship
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

    # Convert Time_Start to numeric
    CAS_flight['Time_mid'] = pd.to_numeric(CAS_flight['Time_mid'], errors='coerce')
    twoDS_flight['Time_Start'] = pd.to_numeric(twoDS_flight['Time_Start'], errors='coerce')

    # Extract required columns
    CAS_times = CAS_flight['Time_mid'].values
    CAS_lwc = CAS_flight['LWC_CAS'].values
    CAS_bins = {f'CAS_Bin{bin_label:02d}': CAS_flight[f'CAS_Bin{bin_label:02d}'].values for bin_label in range(12, 30)}

    TwoDS_times = twoDS_flight['Time_Start'].values
    TwoDS_N_total = twoDS_flight['N-total_2DS'].values

    total_BCB_means = []

    for k in range(len(BCB_start)):
        start20 = BCB_start[k]
        end20 = BCB_stop[k]

        # Find indices in the BCB range for CAS and TwoDS
        CAS_indices_in_range = np.where((CAS_times >= start20) & (CAS_times <= end20))[0]
        TwoDS_indices_in_range = np.where((TwoDS_times >= start20) & (TwoDS_times <= end20))[0]

        if CAS_indices_in_range.size > 0 and TwoDS_indices_in_range.size > 0:
            data_labels = []
            BCB_means = []

            for CAS_idx, TwoDS_idx in zip(CAS_indices_in_range, TwoDS_indices_in_range):
                lwc_val = CAS_lwc[CAS_idx]
                N_val = TwoDS_N_total[TwoDS_idx]

                # Assign labels
                lwc_label = 'Y' if 0 <= lwc_val <= 0.0025 else 'N'
                N_label = 'Y' if 0 <= N_val <= 100 else 'N'
                label = 'Y' if lwc_label == 'Y' and N_label == 'Y' else 'N'
                data_labels.append(label)

                # Collect bin values
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

# Print leg_info or use it as needed
for leg in leg_info:
    print(f"Date: {leg['Date']}, Start: {leg['BCB_start']}, Stop: {leg['BCB_stop']}, Data Labels: {leg['Data_Labels']}")

#%%
leg_count = Counter([leg['Date'] for leg in leg_info])
print("Number of legs associated with each date:")
total_legs = 0
for date, count in sorted(leg_count.items()):
    print(f"Date: {date}, Number of Legs: {count}")
    total_legs += count
print(f"\nTotal number of legs: {total_legs}")
#%%
#Now we need to pull the droplet concentration from each bin for each flight leg and calculate the bin
#mean concentration for each leg. You should end up with 18 mean values, 1 for each bin, for each leg. 


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

    
    CAS_times = np.array(CAS_flight['Time_mid'], dtype=float)
    TwoDS_times = np.array(twoDS_flight['Time_Start'], dtype=float)

    lwc = np.array(CAS_flight['LWC_CAS'], dtype=float)
    N_total = np.array(twoDS_flight['N-total_2DS'], dtype=float)

    
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

        if CAS_indices_in_range.size > 0 and TwoDS_indices_in_range.size > 0:
            for cas_idx, twods_idx in zip(CAS_indices_in_range, TwoDS_indices_in_range):
                lwc_val = lwc[cas_idx]
                N_val = N_total[twods_idx]
                lwc_label = 'Y' if 0 <= lwc_val <= 0.0025 else 'N'
                N_label = 'Y' if 0 <= N_val <= 100 else 'N'
                label = 'Y' if lwc_label == 'Y' and N_label == 'Y' else 'N'

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
total_legs = sum(len(item) for item in master_CAS_BCB)
print(f"Total number of legs: {total_legs}")
#%%
#Now we need to apply our conversion from dNdlog10D to dNdD for each bin and calculate the mean concentration
Y_BCB_calc = []
N_BCB_calc = []

for flight_data in master_CAS_BCB:
    for bin_means in flight_data:
        Y_calc = {'Date': bin_means['Date'], 'BCB_start': bin_means['BCB_start'], 'BCB_stop': bin_means['BCB_stop']}
        N_calc = {'Date': bin_means['Date'], 'BCB_start': bin_means['BCB_start'], 'BCB_stop': bin_means['BCB_stop']}
        
        for bin_label in range(12, 30):
            bin_key_Y = f'Bin{bin_label}_Y_mean'
            bin_key_N = f'Bin{bin_label}_N_mean'

            Y_calc[bin_key_Y] = np.nanmean(bin_means[bin_key_Y]) * bin_log[bin_label - 12]
            N_calc[bin_key_N] = np.nanmean(bin_means[bin_key_N]) * bin_log[bin_label - 12]

        Y_BCB_calc.append(Y_calc)
        N_BCB_calc.append(N_calc)
# %%
#Sum across all bins for each leg in dN for a total concentration per leg of GCCN
for entry in Y_BCB_calc:
    bin_keys = [f'Bin{bin_label}_Y_mean' for bin_label in range(12, 30)]
    entry['Total_GCCN_Concentration'] = np.nansum([entry[key] for key in bin_keys if key in entry])

# %%
from collections import defaultdict

GCCN_flight_totals = defaultdict(lambda: {'Legs': [], 'Total_GCCN_Concentration': 0, 'Leg_Count': 0})

for entry in Y_BCB_calc:
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
        average_gccn_per_flight[date] = np.nan  # In case there are no legs

print("Average GCCN per Flight Dictionary:")
for date, avg_gccn in average_gccn_per_flight.items():
    print(f"Date: {date}, Average GCCN: {avg_gccn:.4f}")

#%%
#splitting flights based on high and low average GCCN
gccn_values = np.array(list(average_gccn_per_flight.values()))

# Set threshold at the 80th percentile (so that top 20% are "High GCCN")
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

#distribution of high and low GCCN flights concentration
df_gccn = pd.DataFrame({
    "GCCN Concentration (cm⁻³)": np.concatenate([high_gccn_values, low_gccn_values]),
    "Flight Type": ["High GCCN"] * len(high_gccn_values) + ["Low GCCN"] * len(low_gccn_values)
})

plt.figure(figsize=(8, 6))
sns.violinplot(x="Flight Type", y="GCCN Concentration (cm⁻³)", data=df_gccn, inner="box", palette=["lavender", "lightblue"], scale="width")
plt.yscale('log')
plt.ylabel("GCCN Concentration (cm⁻³)", fontsize=16, fontweight="bold")
plt.xlabel("GCCN Flight Category", fontsize=16, fontweight="bold")
plt.title("Comparison of High & Low GCCN Flight Concentrations", fontsize=16, fontweight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tick_params(axis="both", which="major", labelsize=14, width=3, length=8)
plt.tick_params(axis="both", which="minor", labelsize=14, width=2, length=5)
plt.show()
#%%
#Averagge concentration stats
avg_high_gccn = np.mean(high_gccn_values)
avg_low_gccn = np.mean(low_gccn_values)
num_high_flights = len(high_gccn_values)
num_low_flights = len(low_gccn_values)
print(f"Average High GCCN Flight Concentration: {avg_high_gccn:.4f} cm⁻³")
print(f"Number of High GCCN Flights: {num_high_flights}")

print(f"Average Low GCCN Flight Concentration: {avg_low_gccn:.4f} cm⁻³")
print(f"Number of Low GCCN Flights: {num_low_flights}")
#%%
#%%
from collections import defaultdict


# Constants
rho_salt = 2200  # kg/m³

def calculate_mass(N0, D):
    N0_m4 = N0 * 10**6  # Convert cm⁻³µm⁻¹ to m⁻⁴
    integrand = lambda d: np.exp(-d / D) * (d * 1e-6)**3  # Convert µm³ → m³
    mass_integral, _ = quad(integrand, 2, np.inf)  # Integrate from 2µm to ∞
    return (np.pi / 6) * rho_salt * N0_m4 * mass_integral  

# Aggregating mass per flight
dry_mass_flight_totals = defaultdict(lambda: {'Legs': [], 'Total_GCCN_Mass': 0, 'Leg_Count': 0})

for entry in dry_exponential_fits:
    date = entry['Date']
    start_time = entry['BCB_start']
    stop_time = entry['BCB_stop']
    dry_intercept = entry['Dry_Intercept_n0']
    dry_slope = entry['Dry_E_folding_D']
    
    if dry_slope > 0 and dry_intercept > 0:
        mass_value = calculate_mass(dry_intercept, dry_slope) * 1e9  # Convert kg/m³ to µg/m³
        
        dry_mass_flight_totals[date]['Legs'].append({
            'Leg_start': start_time,
            'Leg_stop': stop_time,
            'Leg_GCCN_Mass': mass_value
        })
        
        dry_mass_flight_totals[date]['Total_GCCN_Mass'] += mass_value
        dry_mass_flight_totals[date]['Leg_Count'] += 1  

dry_mass_flight_totals = dict(dry_mass_flight_totals)

# Compute average GCCN mass per flight
average_mass_per_flight = {}
for date, flight_data in dry_mass_flight_totals.items():
    if flight_data['Leg_Count'] > 0:
        average_mass_per_flight[date] = flight_data['Total_GCCN_Mass'] / flight_data['Leg_Count']
    else:
        average_mass_per_flight[date] = np.nan  # In case there are no legs

# Splitting flights based on high and low average mass
mass_values = np.array(list(average_mass_per_flight.values()))
threshold = np.percentile(mass_values, 50)  # 80th percentile threshold

high_GCCN_mass = {}
low_GCCN_mass = {}
for date, avg_mass in average_mass_per_flight.items():
    if avg_mass >= threshold:
        high_GCCN_mass[date] = avg_mass  
    else:
        low_GCCN_mass[date] = avg_mass 

# Distribution of high and low GCCN mass flights
df_mass = pd.DataFrame({
    "GCCN Mass (µg/m³)": np.concatenate([list(high_GCCN_mass.values()), list(low_GCCN_mass.values())]),
    "Flight Type": ["High GCCN Mass"] * len(high_GCCN_mass) + ["Low GCCN Mass"] * len(low_GCCN_mass)
})

plt.figure(figsize=(8, 6))
sns.violinplot(x="Flight Type", y="GCCN Mass (µg/m³)", data=df_mass, inner="box", palette=["lavender", "lightblue"], scale="width")
plt.yscale('log')
plt.ylabel("GCCN Mass (µg/m³)", fontsize=19, fontweight="bold")
plt.xlabel("GCCN Flight Category", fontsize=19, fontweight="bold")
plt.title("Comparison of High & Low GCCN Flight Mass", fontsize=19, fontweight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tick_params(axis="both", which="major", labelsize=19, width=3, length=8)
plt.tick_params(axis="both", which="minor", labelsize=19, width=2, length=5)
plt.show()

# Average mass stats
avg_high_mass = np.mean(list(high_GCCN_mass.values()))
avg_low_mass = np.mean(list(low_GCCN_mass.values()))
num_high_mass_flights = len(high_GCCN_mass)
num_low_mass_flights = len(low_GCCN_mass)

print(f"Average High GCCN Flight Mass: {avg_high_mass:.4f} µg/m³")
print(f"Number of High GCCN Mass Flights: {num_high_mass_flights}")
print(f"Average Low GCCN Flight Mass: {avg_low_mass:.4f} µg/m³")
print(f"Number of Low GCCN Mass Flights: {num_low_mass_flights}")
#%%
#Now we need to plot the RWC vs LWC for the high and low GCCN mass flights
#We need to split the data based on the high and low GCCN mass flights
# Categorize data based on high and low mass flights
high_mass_data = [entry for entry in total_combined_concentration if entry['Date'] in high_GCCN_mass]
low_mass_data = [entry for entry in total_combined_concentration if entry['Date'] in low_GCCN_mass]

# Extract concentration, LWC, and RWC for high-mass flights
high_concentration = np.array([entry['Total_Combined_Concentration'] for entry in high_mass_data])
high_lwc = np.array([entry['Total_Liquid_Water'] for entry in total_liquid_water if entry['Date'] in high_GCCN_mass])
high_rwc = np.array([entry['RWC'] for entry in total_liquid_water if entry['Date'] in high_GCCN_mass])

# Extract concentration, LWC, and RWC for low-mass flights
low_concentration = np.array([entry['Total_Combined_Concentration'] for entry in low_mass_data])
low_lwc = np.array([entry['Total_Liquid_Water'] for entry in total_liquid_water if entry['Date'] in low_GCCN_mass])
low_rwc = np.array([entry['RWC'] for entry in total_liquid_water if entry['Date'] in low_GCCN_mass])

# Define bins for histograms
num_bins = 3
x_bins = np.logspace(np.log10(1), np.log10(max(high_concentration.tolist() + low_concentration.tolist())), num_bins)
y_bins = np.logspace(np.log10(min(high_lwc.tolist() + low_lwc.tolist())), np.log10(max(high_lwc.tolist() + low_lwc.tolist())), num_bins)

# Compute histograms for high-mass flights
sum_rwc_high, xedges, yedges = np.histogram2d(high_concentration, high_lwc, bins=[x_bins, y_bins], weights=high_rwc)
sum_lwc_high, _, _ = np.histogram2d(high_concentration, high_lwc, bins=[x_bins, y_bins], weights=high_lwc)
counts_high, _, _ = np.histogram2d(high_concentration, high_lwc, bins=[x_bins, y_bins])

# Compute averages
avg_rwc_high = np.divide(sum_rwc_high, counts_high, out=np.full_like(sum_rwc_high, np.nan), where=counts_high > 0)
avg_lwc_high = np.divide(sum_lwc_high, counts_high, out=np.full_like(sum_lwc_high, np.nan), where=counts_high > 0)
rwc_lwc_ratio_high = np.divide(avg_rwc_high, avg_lwc_high, out=np.full_like(avg_rwc_high, np.nan), where=avg_lwc_high > 0) * 100
masked_rwc_high = np.ma.masked_where(np.isnan(rwc_lwc_ratio_high), rwc_lwc_ratio_high)

# Compute histograms for low-mass flights
sum_rwc_low, _, _ = np.histogram2d(low_concentration, low_lwc, bins=[x_bins, y_bins], weights=low_rwc)
sum_lwc_low, _, _ = np.histogram2d(low_concentration, low_lwc, bins=[x_bins, y_bins], weights=low_lwc)
counts_low, _, _ = np.histogram2d(low_concentration, low_lwc, bins=[x_bins, y_bins])

# Compute averages
avg_rwc_low = np.divide(sum_rwc_low, counts_low, out=np.full_like(sum_rwc_low, np.nan), where=counts_low > 0)
avg_lwc_low = np.divide(sum_lwc_low, counts_low, out=np.full_like(sum_lwc_low, np.nan), where=counts_low > 0)
rwc_lwc_ratio_low = np.divide(avg_rwc_low, avg_lwc_low, out=np.full_like(avg_rwc_low, np.nan), where=avg_lwc_low > 0) * 100
masked_rwc_low = np.ma.masked_where(np.isnan(rwc_lwc_ratio_low), rwc_lwc_ratio_low)

# Plot high-mass flights heatmap
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
plt.title('High GCCN Mass Flights January-June 2022', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()

# Plot low-mass flights heatmap
plt.figure(figsize=(8, 6))
plt.pcolormesh(xedges, yedges, masked_rwc_low.T, cmap="RdBu_r", norm=norm, shading='auto')
plt.colorbar(label="RWC / LWC (%)")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc /cm³', fontsize=19, fontweight='bold')
plt.ylabel('LWC g/m³', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Low GCCN Mass Flights January-June 2022', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()


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
num_bins = 3
x_bins = np.logspace(np.log10(1), np.log10(max(high_concentration.tolist() + low_concentration.tolist())), num_bins)
y_bins = np.logspace(np.log10(min(high_lwc.tolist() + low_lwc.tolist())), np.log10(max(high_lwc.tolist() + low_lwc.tolist())), num_bins)
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
plt.title('High GCCN Flights January-June 2022', fontsize=19, fontweight='bold')
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
plt.title('Low GCCN Flights January-June 2022', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
#number of points in each bin 

#%%

# Compute NaN masks for high and low GCCN flights
gray_mask_high = np.isnan(rwc_lwc_ratio_high)
gray_values_high = np.full_like(rwc_lwc_ratio_high, np.nan)
gray_values_high[gray_mask_high] = 1  # Assign 1 to masked regions for gray overlay

gray_mask_low = np.isnan(rwc_lwc_ratio_low)
gray_values_low = np.full_like(rwc_lwc_ratio_low, np.nan)
gray_values_low[gray_mask_low] = 1  # Assign 1 to masked regions for gray overlay

# Plot High GCCN Flights
plt.figure(figsize=(8, 6))
norm = mcolors.Normalize(vmin=1, vmax=100)

# Plot main heatmap
img = plt.pcolormesh(xedges, yedges, masked_rwc_high.T, cmap="RdBu_r", norm=norm, shading='auto')

# Overlay gray mask for NaN regions
plt.pcolormesh(xedges, yedges, gray_values_high.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)

# Colorbar and plot settings
cbar = plt.colorbar(img)
cbar.set_label("RWC / LWC (%)", fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc /cm³', fontsize=16, fontweight='bold')
plt.ylabel('LWC g/m³', fontsize=16, fontweight='bold')
plt.title('High GCCN Flights January-June 2022', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()

# Plot Low GCCN Flights
plt.figure(figsize=(8, 6))

# Plot main heatmap
img = plt.pcolormesh(xedges, yedges, masked_rwc_low.T, cmap="RdBu_r", norm=norm, shading='auto')

# Overlay gray mask for NaN regions
plt.pcolormesh(xedges, yedges, gray_values_low.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)

# Colorbar and plot settings
cbar = plt.colorbar(img)
cbar.set_label("RWC / LWC (%)", fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc /cm³', fontsize=16, fontweight='bold')
plt.ylabel('LWC g/m³', fontsize=16, fontweight='bold')
plt.title('Low GCCN Flights January-June 2022', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
#trying to filter based on fixed LWC and N 

x_min, x_max = 50, 200  # Nr+Nc range
y_min, y_max = 0.1, 0.3  # LWC range
high_mask = (high_concentration >= x_min) & (high_concentration <= x_max) & \
            (high_lwc >= y_min) & (high_lwc <= y_max)

filtered_high_concentration = high_concentration[high_mask]
filtered_high_lwc = high_lwc[high_mask]
filtered_high_rwc = high_rwc[high_mask]
low_mask = (low_concentration >= x_min) & (low_concentration <= x_max) & \
           (low_lwc >= y_min) & (low_lwc <= y_max)

filtered_low_concentration = low_concentration[low_mask]
filtered_low_lwc = low_lwc[low_mask]
filtered_low_rwc = low_rwc[low_mask]
num_bins = 3
x_bins = np.logspace(np.log10(x_min), np.log10(x_max), num_bins)
y_bins = np.logspace(np.log10(y_min), np.log10(y_max), num_bins)
sum_rwc_high, xedges, yedges = np.histogram2d(filtered_high_concentration, filtered_high_lwc, bins=[x_bins, y_bins], weights=filtered_high_rwc)
sum_lwc_high, _, _ = np.histogram2d(filtered_high_concentration, filtered_high_lwc, bins=[x_bins, y_bins], weights=filtered_high_lwc)
counts_high, _, _ = np.histogram2d(filtered_high_concentration, filtered_high_lwc, bins=[x_bins, y_bins])

avg_rwc_high = np.divide(sum_rwc_high, counts_high, out=np.full_like(sum_rwc_high, np.nan), where=counts_high > 0)
avg_lwc_high = np.divide(sum_lwc_high, counts_high, out=np.full_like(sum_lwc_high, np.nan), where=counts_high > 0)
rwc_lwc_ratio_high = np.divide(avg_rwc_high, avg_lwc_high, out=np.full_like(avg_rwc_high, np.nan), where=avg_lwc_high > 0) * 100
masked_rwc_high = np.ma.masked_where(np.isnan(rwc_lwc_ratio_high), rwc_lwc_ratio_high)
sum_rwc_low, _, _ = np.histogram2d(filtered_low_concentration, filtered_low_lwc, bins=[x_bins, y_bins], weights=filtered_low_rwc)
sum_lwc_low, _, _ = np.histogram2d(filtered_low_concentration, filtered_low_lwc, bins=[x_bins, y_bins], weights=filtered_low_lwc)
counts_low, _, _ = np.histogram2d(filtered_low_concentration, filtered_low_lwc, bins=[x_bins, y_bins])

avg_rwc_low = np.divide(sum_rwc_low, counts_low, out=np.full_like(sum_rwc_low, np.nan), where=counts_low > 0)
avg_lwc_low = np.divide(sum_lwc_low, counts_low, out=np.full_like(sum_lwc_low, np.nan), where=counts_low > 0)
rwc_lwc_ratio_low = np.divide(avg_rwc_low, avg_lwc_low, out=np.full_like(avg_rwc_low, np.nan), where=avg_lwc_low > 0) * 100
masked_rwc_low = np.ma.masked_where(np.isnan(rwc_lwc_ratio_low), rwc_lwc_ratio_low)
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
cbar.set_label("RWC / LWC (%)", fontsize=18, fontweight='bold') 
cbar.ax.tick_params(labelsize=19, width=2, length=5) 
for t in cbar.ax.get_yticklabels():  
    t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.title('High GCCN Flights', fontsize=19, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(xedges, yedges, masked_rwc_low.T, cmap="RdBu_r", norm=norm, shading='auto')
plt.pcolormesh(xedges, yedges, gray_values_low.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)
cbar = plt.colorbar(img)
cbar.set_label("RWC / LWC (%)", fontsize=18, fontweight='bold') 
cbar.ax.tick_params(labelsize=19, width=2, length=5) 
for t in cbar.ax.get_yticklabels():  
    t.set_fontweight('bold')
plt.xscale('log')

plt.yscale('log')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.title('Low GCCN Flights', fontsize=19, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.tight_layout()
plt.show()
#%%
#mass in fixed region
# Define the fixed region for filtering
x_min, x_max = 50, 200  # Nr+Nc range
y_min, y_max = 0.1, 0.3  # LWC range

# Apply the filtering to high mass flights
high_mask = (high_concentration >= x_min) & (high_concentration <= x_max) & \
            (high_lwc >= y_min) & (high_lwc <= y_max)

filtered_high_concentration = high_concentration[high_mask]
filtered_high_lwc = high_lwc[high_mask]
filtered_high_rwc = high_rwc[high_mask]

# Apply the filtering to low mass flights
low_mask = (low_concentration >= x_min) & (low_concentration <= x_max) & \
           (low_lwc >= y_min) & (low_lwc <= y_max)

filtered_low_concentration = low_concentration[low_mask]
filtered_low_lwc = low_lwc[low_mask]
filtered_low_rwc = low_rwc[low_mask]

# Define binning for the filtered region
num_bins = 3
x_bins = np.logspace(np.log10(x_min), np.log10(x_max), num_bins)
y_bins = np.logspace(np.log10(y_min), np.log10(y_max), num_bins)

# Compute histograms for high mass flights
sum_rwc_high, xedges, yedges = np.histogram2d(filtered_high_concentration, filtered_high_lwc, bins=[x_bins, y_bins], weights=filtered_high_rwc)
sum_lwc_high, _, _ = np.histogram2d(filtered_high_concentration, filtered_high_lwc, bins=[x_bins, y_bins], weights=filtered_high_lwc)
counts_high, _, _ = np.histogram2d(filtered_high_concentration, filtered_high_lwc, bins=[x_bins, y_bins])

# Compute average values for high mass flights
avg_rwc_high = np.divide(sum_rwc_high, counts_high, out=np.full_like(sum_rwc_high, np.nan), where=counts_high > 0)
avg_lwc_high = np.divide(sum_lwc_high, counts_high, out=np.full_like(sum_lwc_high, np.nan), where=counts_high > 0)
rwc_lwc_ratio_high = np.divide(avg_rwc_high, avg_lwc_high, out=np.full_like(avg_rwc_high, np.nan), where=avg_lwc_high > 0) * 100

# Mask NaN values for visualization
masked_rwc_high = np.ma.masked_where(np.isnan(rwc_lwc_ratio_high), rwc_lwc_ratio_high)

# Compute histograms for low mass flights
sum_rwc_low, _, _ = np.histogram2d(filtered_low_concentration, filtered_low_lwc, bins=[x_bins, y_bins], weights=filtered_low_rwc)
sum_lwc_low, _, _ = np.histogram2d(filtered_low_concentration, filtered_low_lwc, bins=[x_bins, y_bins], weights=filtered_low_lwc)
counts_low, _, _ = np.histogram2d(filtered_low_concentration, filtered_low_lwc, bins=[x_bins, y_bins])

# Compute average values for low mass flights
avg_rwc_low = np.divide(sum_rwc_low, counts_low, out=np.full_like(sum_rwc_low, np.nan), where=counts_low > 0)
avg_lwc_low = np.divide(sum_lwc_low, counts_low, out=np.full_like(sum_lwc_low, np.nan), where=counts_low > 0)
rwc_lwc_ratio_low = np.divide(avg_rwc_low, avg_lwc_low, out=np.full_like(avg_rwc_low, np.nan), where=avg_lwc_low > 0) * 100

# Mask NaN values for visualization
masked_rwc_low = np.ma.masked_where(np.isnan(rwc_lwc_ratio_low), rwc_lwc_ratio_low)

# Create the high GCCN mass heatmap
plt.figure(figsize=(8, 6))
norm = mcolors.Normalize(vmin=1, vmax=100)
img = plt.pcolormesh(xedges, yedges, masked_rwc_high.T, cmap="RdBu_r", norm=norm, shading='auto')

# Gray out missing data
gray_mask_high = np.isnan(rwc_lwc_ratio_high)
gray_values_high = np.full_like(rwc_lwc_ratio_high, np.nan)
gray_values_high[gray_mask_high] = 1
plt.pcolormesh(xedges, yedges, gray_values_high.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)

# Colorbar
cbar = plt.colorbar(img)
cbar.set_label("RWC / LWC (%)", fontsize=18, fontweight='bold') 
cbar.ax.tick_params(labelsize=19, width=2, length=5) 
for t in cbar.ax.get_yticklabels():  
    t.set_fontweight('bold')

# Plot formatting
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.title('High GCCN Mass Flights', fontsize=19, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.tight_layout()
plt.show()

# Create the low GCCN mass heatmap
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(xedges, yedges, masked_rwc_low.T, cmap="RdBu_r", norm=norm, shading='auto')

# Gray out missing data
gray_mask_low = np.isnan(rwc_lwc_ratio_low)
gray_values_low = np.full_like(rwc_lwc_ratio_low, np.nan)
gray_values_low[gray_mask_low] = 1
plt.pcolormesh(xedges, yedges, gray_values_low.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)

# Colorbar
cbar = plt.colorbar(img)
cbar.set_label("RWC / LWC (%)", fontsize=18, fontweight='bold') 
cbar.ax.tick_params(labelsize=19, width=2, length=5) 
for t in cbar.ax.get_yticklabels():  
    t.set_fontweight('bold')

# Plot formatting
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.title('Low GCCN Mass Flights', fontsize=19, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.tight_layout()
plt.show()

#%%
#violin plots 


# Extract RWC values for High and Low GCCN flights (removing NaNs)
high_rwc_values = filtered_high_rwc[~np.isnan(filtered_high_rwc)]
low_rwc_values = filtered_low_rwc[~np.isnan(filtered_low_rwc)]

# Create a DataFrame for seaborn violin plot
df_violin = pd.DataFrame({
    "RWC (g/m³)": np.concatenate([high_rwc_values, low_rwc_values]),
    "GCCN Category": ["High GCCN"] * len(high_rwc_values) + ["Low GCCN"] * len(low_rwc_values)
})

# Plot the violin plot
plt.figure(figsize=(8, 6))
sns.violinplot(x="GCCN Category", y="RWC (g/m³)", data=df_violin, inner="box",
               palette=["plum", "lightblue"], scale="width")

# Customize plot
plt.yscale('log')  # Log scale for better visibility
plt.ylabel("RWC (g/m³)", fontsize=19, fontweight="bold")
plt.xlabel("GCCN Flight Category", fontsize=19, fontweight="bold")
plt.title("Comparison of RWC for High & Low GCCN Flights", fontsize=17, fontweight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tick_params(axis="both", which="major", labelsize=19, width=3, length=8)
plt.tick_params(axis="both", which="minor", labelsize=19, width=2, length=5)

plt.show()

#%%
#mass violin plot 
# Flatten the RWC/LWC ratio arrays for plotting
high_rwc_lwc_values = masked_rwc_high.compressed()  # Remove NaN values
low_rwc_lwc_values = masked_rwc_low.compressed()  # Remove NaN values

# Create a DataFrame for seaborn plotting
df_violin = pd.DataFrame({
    "RWC/LWC (%)": np.concatenate([high_rwc_lwc_values, low_rwc_lwc_values]),
    "GCCN Mass Category": ["High GCCN Mass"] * len(high_rwc_lwc_values) + ["Low GCCN Mass"] * len(low_rwc_lwc_values)
})

# Plot the violin plot
plt.figure(figsize=(8, 6))
sns.violinplot(x="GCCN Mass Category", y="RWC/LWC (%)", data=df_violin, inner="box", palette=["lavender", "lightblue"], scale="width")

# Customize plot
plt.yscale('log')  # Log scale for better visibility
plt.ylabel("RWC/LWC (%)", fontsize=19, fontweight="bold")
plt.xlabel("GCCN Flight Category", fontsize=19, fontweight="bold")
plt.title("RWC High & Low GCCN Mass Flights", fontsize=17, fontweight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tick_params(axis="both", which="major", labelsize=19, width=3, length=8)
plt.tick_params(axis="both", which="minor", labelsize=19, width=2, length=5)

plt.show()
#%% mass and concentration biolin plot together 

# Extract RWC values for High & Low GCCN Concentration flights (removing NaNs)
high_rwc_values = filtered_high_rwc[~np.isnan(filtered_high_rwc)]
low_rwc_values = filtered_low_rwc[~np.isnan(filtered_low_rwc)]

# Extract RWC/LWC values for High & Low GCCN Mass flights (removing NaNs)
high_rwc_lwc_values = masked_rwc_high.compressed()  # Remove NaN values
low_rwc_lwc_values = masked_rwc_low.compressed()  # Remove NaN values

# Create a DataFrame for seaborn violin plot
df_violin = pd.DataFrame({
    "RWC (g/m³)": np.concatenate([high_rwc_values, low_rwc_values, high_rwc_lwc_values, low_rwc_lwc_values]),
    "GCCN Category": (["High Concentration"] * len(high_rwc_values) +
                      ["Low Concentration"] * len(low_rwc_values) +
                      ["High Mass"] * len(high_rwc_lwc_values) +
                      ["Low Mass"] * len(low_rwc_lwc_values))
})

# Plot the violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x="GCCN Category", y="RWC (g/m³)", data=df_violin, inner="box",
               palette=["orchid", "lightblue", "lavender", "plum"], scale="width")

# Customize plot
plt.yscale('log')  # Log scale for better visibility
plt.ylabel("RWC (g/m³)", fontsize=19, fontweight="bold")
plt.xlabel("GCCN Flight Category", fontsize=19, fontweight="bold")
plt.title("RWC for Mass & Concentration", fontsize=17, fontweight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tick_params(axis="both", which="major", labelsize=14, width=3, length=8)
plt.tick_params(axis="both", which="minor", labelsize=19, width=2, length=5)

plt.show()


#%%
#how many points in each bin

# Define the region of interest
x_min, x_max = 50, 200  # Nr+Nc range
y_min, y_max = 0.1, 0.3  # LWC range

# Apply filtering masks
high_mask = (high_concentration >= x_min) & (high_concentration <= x_max) & \
            (high_lwc >= y_min) & (high_lwc <= y_max)
filtered_high_concentration = high_concentration[high_mask]
filtered_high_lwc = high_lwc[high_mask]

low_mask = (low_concentration >= x_min) & (low_concentration <= x_max) & \
           (low_lwc >= y_min) & (low_lwc <= y_max)
filtered_low_concentration = low_concentration[low_mask]
filtered_low_lwc = low_lwc[low_mask]

# Define bin edges
num_bins = 3  # Adjust if needed
x_bins = np.logspace(np.log10(x_min), np.log10(x_max), num_bins)
y_bins = np.logspace(np.log10(y_min), np.log10(y_max), num_bins)

# Compute the number of data points in each bin
counts_high, xedges, yedges = np.histogram2d(filtered_high_concentration, filtered_high_lwc, bins=[x_bins, y_bins])
counts_low, _, _ = np.histogram2d(filtered_low_concentration, filtered_low_lwc, bins=[x_bins, y_bins])

# Convert bin edges to bin center values for better readability
x_bin_centers = (xedges[:-1] + xedges[1:]) / 2
y_bin_centers = (yedges[:-1] + yedges[1:]) / 2

# Create tables for High GCCN bin counts
df_high = pd.DataFrame(counts_high, index=y_bin_centers, columns=x_bin_centers)
df_high.index.name = 'LWC (g/m³)'
df_high.columns.name = 'Nr+Nc (cm³)'

# Create tables for Low GCCN bin counts
df_low = pd.DataFrame(counts_low, index=y_bin_centers, columns=x_bin_centers)
df_low.index.name = 'LWC (g/m³)'
df_low.columns.name = 'Nr+Nc (cm³)'

# Print the tables
print("\nHigh GCCN Bin Counts:")
print(df_high)

print("\nLow GCCN Bin Counts:")
print(df_low)

# Optionally, save the tables as CSV files
df_high.to_csv("High_GCCN_Bin_Counts.csv")
df_low.to_csv("Low_GCCN_Bin_Counts.csv")

#%%
# Define bins (ensuring they match the previous analysis)
num_bins = 3
x_bins = np.logspace(np.log10(x_min), np.log10(x_max), num_bins)
y_bins = np.logspace(np.log10(y_min), np.log10(y_max), num_bins)

# Compute statistics for High GCCN
sum_rwc_high, xedges, yedges = np.histogram2d(filtered_high_concentration, filtered_high_lwc, bins=[x_bins, y_bins], weights=filtered_high_rwc)
sum_lwc_high, _, _ = np.histogram2d(filtered_high_concentration, filtered_high_lwc, bins=[x_bins, y_bins], weights=filtered_high_lwc)
counts_high, _, _ = np.histogram2d(filtered_high_concentration, filtered_high_lwc, bins=[x_bins, y_bins])

# Compute standard deviation for RWC/LWC in each bin for High GCCN
std_rwc_high, _, _ = np.histogram2d(filtered_high_concentration, filtered_high_lwc, bins=[x_bins, y_bins], weights=(filtered_high_rwc / filtered_high_lwc) ** 2)
std_rwc_high = np.sqrt(std_rwc_high / counts_high - (sum_rwc_high / sum_lwc_high) ** 2)  # Standard deviation formula

# Compute statistics for Low GCCN
sum_rwc_low, _, _ = np.histogram2d(filtered_low_concentration, filtered_low_lwc, bins=[x_bins, y_bins], weights=filtered_low_rwc)
sum_lwc_low, _, _ = np.histogram2d(filtered_low_concentration, filtered_low_lwc, bins=[x_bins, y_bins], weights=filtered_low_lwc)
counts_low, _, _ = np.histogram2d(filtered_low_concentration, filtered_low_lwc, bins=[x_bins, y_bins])

# Compute standard deviation for RWC/LWC in each bin for Low GCCN
std_rwc_low, _, _ = np.histogram2d(filtered_low_concentration, filtered_low_lwc, bins=[x_bins, y_bins], weights=(filtered_low_rwc / filtered_low_lwc) ** 2)
std_rwc_low = np.sqrt(std_rwc_low / counts_low - (sum_rwc_low / sum_lwc_low) ** 2)

# Compute mean RWC/LWC per bin
avg_rwc_high = np.divide(sum_rwc_high, counts_high, out=np.full_like(sum_rwc_high, np.nan), where=counts_high > 0)
avg_lwc_high = np.divide(sum_lwc_high, counts_high, out=np.full_like(sum_lwc_high, np.nan), where=counts_high > 0)
rwc_lwc_ratio_high = np.divide(avg_rwc_high, avg_lwc_high, out=np.full_like(avg_rwc_high, np.nan), where=avg_lwc_high > 0) * 100

avg_rwc_low = np.divide(sum_rwc_low, counts_low, out=np.full_like(sum_rwc_low, np.nan), where=counts_low > 0)
avg_lwc_low = np.divide(sum_lwc_low, counts_low, out=np.full_like(sum_rwc_low, np.nan), where=counts_low > 0)
rwc_lwc_ratio_low = np.divide(avg_rwc_low, avg_lwc_low, out=np.full_like(avg_rwc_low, np.nan), where=avg_lwc_low > 0) * 100
#%%





































#%%
#Differenc in each bin

diff_rwc_lwc = avg_rwc_high - avg_rwc_low

masked_diff = np.ma.masked_where(np.isnan(diff_rwc_lwc), diff_rwc_lwc)

plt.figure(figsize=(8, 6))
cmap = "RdBu_r"  # Red = Increase in High GCCN, Blue = Decrease
img = plt.pcolormesh(x_bins, y_bins, masked_diff.T, cmap=cmap, shading='auto')

gray_mask = np.isnan(diff_rwc_lwc)
gray_values = np.full_like(diff_rwc_lwc, np.nan)
gray_values[gray_mask] = 1  
plt.pcolormesh(x_bins, y_bins, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)

cbar = plt.colorbar(img)
cbar.set_label("RWC/LWC Difference", fontsize=14, fontweight='bold')
cbar.ax.tick_params(labelsize=12, width=2, length=5)

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=16, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=16, fontweight='bold')
plt.title('Difference in RWC/LWC  between high and low GCCN', fontsize=14, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=11, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=11, width=2, length=5)
plt.tight_layout()
plt.show()
#%%

# Compute the difference in each bin
diff_rwc_lwc = avg_rwc_high - avg_rwc_low

# Mask NaN values for plotting
masked_diff = np.ma.masked_where(np.isnan(diff_rwc_lwc), diff_rwc_lwc)

# Ensure vmin and vmax are properly centered around zero
abs_max = max(abs(np.nanmin(diff_rwc_lwc)), abs(np.nanmax(diff_rwc_lwc)))
vmin, vmax = -abs_max, abs_max  # Ensure a balanced color scale

# Define the diverging colormap centered at zero
divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
cmap = plt.cm.RdBu_r  # Red-Blue reversed colormap

# Create the plot
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(x_bins, y_bins, masked_diff.T, cmap=cmap, norm=divnorm, shading='auto')

# Add gray overlay for NaN values
gray_mask = np.isnan(diff_rwc_lwc)
gray_values = np.full_like(diff_rwc_lwc, np.nan)
gray_values[gray_mask] = 1  
plt.pcolormesh(x_bins, y_bins, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)

# Add colorbar
cbar = plt.colorbar(img)
cbar.set_label("RWC/LWC Difference", fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=19, width=2, length=5)

# Set log scales
plt.xscale('log')
plt.yscale('log')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=16, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=16, width=2, length=5)

# Labels and title
plt.title('Difference in RWC/LWC \nbetween High and Low GCCN', fontsize=17, fontweight='bold')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')

plt.show()
#%%
#mass difference 
# Compute the difference in each bin (RWC/LWC difference between high and low mass flights)
diff_rwc_lwc = avg_rwc_high - avg_rwc_low

# Mask NaN values for visualization
masked_diff = np.ma.masked_where(np.isnan(diff_rwc_lwc), diff_rwc_lwc)

# Ensure the color scale is centered around zero
abs_max = max(abs(np.nanmin(diff_rwc_lwc)), abs(np.nanmax(diff_rwc_lwc)))
vmin, vmax = -abs_max, abs_max  # Ensure a balanced color scale

# Define the diverging colormap centered at zero
divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
cmap = plt.cm.RdBu_r  # Red-Blue reversed colormap

# Create the plot
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(x_bins, y_bins, masked_diff.T, cmap=cmap, norm=divnorm, shading='auto')

# Add gray overlay for NaN values
gray_mask = np.isnan(diff_rwc_lwc)
gray_values = np.full_like(diff_rwc_lwc, np.nan)
gray_values[gray_mask] = 1  
plt.pcolormesh(x_bins, y_bins, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)

# Add colorbar
cbar = plt.colorbar(img)
cbar.set_label("RWC/LWC Difference", fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=19, width=2, length=5)

# Set log scales
plt.xscale('log')
plt.yscale('log')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=16, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=16, width=2, length=5)

# Labels and title
plt.title('Difference in RWC/LWC \nbetween High and Low GCCN Mass Flights', fontsize=17, fontweight='bold')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')

plt.show()


























































































































#%%
#still trying significance

from scipy.stats import ttest_ind_from_stats

std_rwc_high = np.sqrt(sum_rwc_high / np.maximum(counts_high, 1))  
std_rwc_low = np.sqrt(sum_rwc_low / np.maximum(counts_low, 1))

# Perform t-test for each bin
t_values = np.zeros_like(avg_rwc_high)
p_values = np.ones_like(avg_rwc_high)

for i in range(avg_rwc_high.shape[0]):
    for j in range(avg_rwc_high.shape[1]):
        if counts_high[i, j] > 1 and counts_low[i, j] > 1:  # Ensure enough samples for valid statistics
            t_stat, p_val = ttest_ind_from_stats(
                mean1=avg_rwc_high[i, j], std1=std_rwc_high[i, j], nobs1=counts_high[i, j],
                mean2=avg_rwc_low[i, j], std2=std_rwc_low[i, j], nobs2=counts_low[i, j],
                equal_var=False  # Welch’s t-test
            )
            t_values[i, j] = t_stat
            p_values[i, j] = p_val

# Apply a significance threshold (p < 0.05)
significant_mask = p_values < 0.05  

# Mask NaN values for plotting
masked_diff = np.ma.masked_where(np.isnan(diff_rwc_lwc), diff_rwc_lwc)

plt.figure(figsize=(8, 6))

# Define diverging colormap centered at zero
divnorm = mcolors.TwoSlopeNorm(vmin=np.nanmin(diff_rwc_lwc), vcenter=0, vmax=np.nanmax(diff_rwc_lwc))
cmap = plt.cm.RdBu_r  

# Plot the difference in RWC/LWC
img = plt.pcolormesh(x_bins, y_bins, masked_diff.T, cmap=cmap, norm=divnorm, shading='auto')

# Overlay hatching for significant bins
plt.contourf(x_bins[:-1], y_bins[:-1], significant_mask.T, hatches=['///'], colors='none', alpha=0, linewidths=0.5)

# Add colorbar
cbar = plt.colorbar(img)
cbar.set_label("RWC/LWC Difference", fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=19, width=2, length=5)

# Set log scales
plt.xscale('log')
plt.yscale('log')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=16, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=16, width=2, length=5)
plt.title('Difference in RWC/LWC \nbetween High and Low GCCN (Significance Test)', fontsize=17, fontweight='bold')

# Labels
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')

plt.tight_layout()
plt.show()





#%%
#Compute the mean RWC% for high and low GCCN flights in this region.
mean_rwc_high = np.nanmean(rwc_lwc_ratio_high)
mean_rwc_low = np.nanmean(rwc_lwc_ratio_low)
print(f"Mean RWC% for High GCCN Flights: {mean_rwc_high:.2f}%")
print(f"Mean RWC% for Low GCCN Flights: {mean_rwc_low:.2f}%")
#%%
#See if the difference is statistically significant Mann-Whitney in this region

# Compute the mean RWC% for high and low GCCN flights **in the specified region**
mean_rwc_high = np.nanmean(rwc_lwc_ratio_high)
mean_rwc_low = np.nanmean(rwc_lwc_ratio_low)

print(f"Mean RWC% for High GCCN Flights: {mean_rwc_high:.2f}%")
print(f"Mean RWC% for Low GCCN Flights: {mean_rwc_low:.2f}%")

# Extract non-NaN values **only within the defined region**
rwc_high_values = rwc_lwc_ratio_high[~np.isnan(rwc_lwc_ratio_high)].flatten()
rwc_low_values = rwc_lwc_ratio_low[~np.isnan(rwc_lwc_ratio_low)].flatten()

### **Mann-Whitney U test (non-parametric)**
stat, p_value = mannwhitneyu(rwc_high_values, rwc_low_values, alternative='two-sided')
significance = "Significant" if p_value < 0.05 else "Not Significant"
print(f"Mann-Whitney U Statistic: {stat:.2f}")
print(f"P-value: {p_value:.5f}")
print(f"Result: The difference is {significance} at the 95% confidence level.")

### **Kolmogorov-Smirnov (KS) Test (distribution comparison)**
ks_stat, ks_p_value = ks_2samp(rwc_high_values, rwc_low_values)
ks_significance = "Significant" if ks_p_value < 0.05 else "Not Significant"
print(f"Kolmogorov-Smirnov Test Statistic: {ks_stat:.5f}")
print(f"P-value: {ks_p_value:.5f}")
print(f"Result: The difference is {ks_significance} at the 95% confidence level.")

### **Welch’s T-test (parametric, assumes normality)**
t_stat, t_p_value = ttest_ind(rwc_high_values, rwc_low_values, equal_var=False, nan_policy='omit')
t_significance = "Significant" if t_p_value < 0.05 else "Not Significant"
print(f"Welch’s T-Test Statistic: {t_stat:.5f}")
print(f"P-value: {t_p_value:.5f}")
print(f"Result: The difference is {t_significance} at the 95% confidence level.")
#%%
# make a table 

data = {
    "Test": ["Mann-Whitney U Test", "Kolmogorov-Smirnov Test", "Welch’s T-Test (for mean comparison)"],
    "Statistic": ["U = 7366.00", "KS = 0.53222", "T = 7.89925"],
    "P-value": ["0.00000", "0.00000", "0.00000"],
    "Result": ["Statistically Significant", "Statistically Significant", "Statistically Significant"],
    "Interpretation": [
        "This non-parametric test (which doesn’t assume normality) shows that the distributions of RWC% for high and low GCCN flights are statistically different. "
        "A p-value of 0.00000 means the probability of this difference being random is extremely low.",
        
        "The KS test checks if the two datasets come from the same distribution. A high KS statistic (0.53222) suggests a large difference between the two distributions. "
        "The p-value means that the shapes of the distributions are significantly different.",
        
        "Welch’s T-test compares the means of the two datasets, assuming unequal variance. "
        "A large t-statistic means that the average RWC% is significantly higher or lower in one of the groups. "
        "The p-value confirms this difference is highly unlikely due to chance."
    ]
}

df_stats = pd.DataFrame(data)
fig, ax = plt.subplots(figsize=(12, 6))  
ax.axis('tight')
ax.axis('off')


table = ax.table(cellText=df_stats.values, 
                 colLabels=df_stats.columns, 
                 cellLoc='center', 
                 loc='center',
                 colWidths=[0.2, 0.15, 0.1, 0.2, 0.6],  # Adjusted column widths for better spacing
                 colColours=["lightgray"] * len(df_stats.columns))

table.auto_set_font_size(False)
table.set_fontsize(12)  
table.scale(1.2, 1.5) 
for i in range(len(df_stats) + 1):  
    table[i, -1].set_height(0.2)
plt.title("Statistical Test Results for High vs. Low GCCN Flights", fontsize=14, fontweight='bold')
plt.show()

#%%
rwc_high_values = rwc_lwc_ratio_high[~np.isnan(rwc_lwc_ratio_high)].flatten()
rwc_low_values = rwc_lwc_ratio_low[~np.isnan(rwc_lwc_ratio_low)].flatten()
stat, p_value = mannwhitneyu(rwc_high_values, rwc_low_values, alternative='two-sided')
significance = "Significant" if p_value < 0.05 else "Not Significant"
print(f"Mann-Whitney U Statistic: {stat:.2f}")
print(f"P-value: {p_value:.5f}")
print(f"Result: The difference is {significance} at the 95% confidence level.")

from scipy.stats import ks_2samp

# Perform Kolmogorov-Smirnov (KS) test
ks_stat, ks_p_value = ks_2samp(rwc_high_values, rwc_low_values)
ks_significance = "Significant" if ks_p_value < 0.05 else "Not Significant"
print(f"Kolmogorov-Smirnov Test Statistic: {ks_stat:.5f}")
print(f"P-value: {ks_p_value:.5f}")
print(f"Result: The difference is {ks_significance} at the 95% confidence level.")

from scipy.stats import ttest_ind

# Perform Welch's T-test (assumes normality, good for large samples)
t_stat, t_p_value = ttest_ind(rwc_high_values, rwc_low_values, equal_var=False, nan_policy='omit')
t_significance = "Significant" if t_p_value < 0.05 else "Not Significant"
print(f"Welch’s T-Test Statistic: {t_stat:.5f}")
print(f"P-value: {t_p_value:.5f}")
print(f"Result: The difference is {t_significance} at the 95% confidence level.")
#%%
# Create a boxplot comparing RWC% for high and low GCCN flights
rwc_high_values = rwc_lwc_ratio_high[~np.isnan(rwc_lwc_ratio_high)].flatten()
rwc_low_values = rwc_lwc_ratio_low[~np.isnan(rwc_lwc_ratio_low)].flatten()
plt.figure(figsize=(8, 6))
plt.boxplot([rwc_high_values, rwc_low_values], labels=['High GCCN', 'Low GCCN'], patch_artist=True,
            boxprops=dict(facecolor='orange', color='orange'),
            medianprops=dict(color='black', linewidth=2))
plt.ylabel("RWC / LWC (%)", fontsize=14, fontweight='bold')
plt.title("CAS January-June 2022", fontsize=16, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#%%
#create dictionaries to store all relevant variables for each classification

high_full_data= {}
low_full_data = {}

for entry in total_combined_concentration:
    date = entry['Date']

    if date in high_GCCN_concentrations:
        if date not in high_full_data:
            high_full_data[date] = {
                'Average_GCCN_Concentration': high_GCCN_concentrations[date],  # Store total GCCN for this date
                'Total_Combined_Concentration': [],
                'LWC': [],
                'RWC': []
            }
        high_full_data[date]['Total_Combined_Concentration'].append(entry['Total_Combined_Concentration'])

for entry in total_liquid_water:
    date = entry['Date']

    if date in high_GCCN_concentrations:
       high_full_data[date]['LWC'].append(entry['Total_Liquid_Water'])
       high_full_data[date]['RWC'].append(entry['RWC'])

for entry in total_combined_concentration:
    date = entry['Date']

    if date in low_GCCN_concentrations:
        if date not in low_full_data :
            low_full_data [date] = {
                'Average_GCCN_Concentration': low_GCCN_concentrations[date],  # Store total GCCN for this date
                'Total_Combined_Concentration': [],
                'LWC': [],
                'RWC': []
            }
        low_full_data [date]['Total_Combined_Concentration'].append(entry['Total_Combined_Concentration'])

for entry in total_liquid_water:
    date = entry['Date']

    if date in low_GCCN_concentrations:
       low_full_data[date]['LWC'].append(entry['Total_Liquid_Water'])
       low_full_data[date]['RWC'].append(entry['RWC'])

#%%
#Box and whisker for RWC/LWC
from scipy.stats import mannwhitneyu
high_rwc_lwc_ratio = [
    rwc / lwc for date in high_full_data
    for rwc, lwc in zip(high_full_data[date]['RWC'], high_full_data[date]['LWC']) if lwc > 0
]

low_rwc_lwc_ratio = [
    rwc / lwc for date in low_full_data
    for rwc, lwc in zip(low_full_data[date]['RWC'], low_full_data[date]['LWC']) if lwc > 0
]

plt.figure(figsize=(8, 6))
sns.boxplot(data=[high_rwc_lwc_ratio, low_rwc_lwc_ratio], palette=['lavender', 'lightblue'])

plt.yscale('log') 
plt.xticks([0, 1], ['High GCCN flights', 'Low GCCN flights'], fontsize=14, fontweight='bold')
plt.ylabel('RWC/LWC', fontsize=14, fontweight='bold')
plt.title('CAS in-cloud January - June 2022', fontsize=16, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=12, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=12, width=2, length=5)
#Mann-Whitney U Test (Non-parametric test for difference in medians)
stat, p_value = mannwhitneyu(high_rwc_lwc_ratio, low_rwc_lwc_ratio, alternative='greater')
plt.figtext(0.15, 0.85, f'Mann-Whitney U p-value: {p_value:.2e}', fontsize=12, fontweight='bold', color='black')
plt.show()

#%%

#Checking average flight leg altitude for above cloud base 
master_ACB = []


for i in range(len(dates_legs)):
    date = dates_legs[i]

    leg_dict = leg_data[i]

    flight_date = leg_dict['Date'] 
    ACB_start = leg_dict['LegIndex_03']['StartTimes']
    ACB_stop = leg_dict['LegIndex_03']['StopTimes']
    sum_flight = summary[i]

    times = sum_flight.Time_mid.values
    winds = sum_flight.Wind_Speed.values
    alts = sum_flight.GPS_altitude.values
    
    all_ACB_means = []


    for i in range(len(ACB_start)):
        index1_start=None
        index1_end=None  
        start = int(ACB_start[i])
        end = ACB_stop[i]

        wind_alt = {
            'Date': date,
            'ACB_start': start,
            'ACB_end': end,
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

        all_ACB_means.append(wind_alt) #List that contains all the BCB wind/alt mean dictionaries for 1 flight
        
    master_ACB.append(all_ACB_means) #List that contains all BCB flights  
#%%
Z0 = 0.02  # meters (typical value for open ocean)
Z10 = 10  # target height m

corrected_calc_Acb = {'Date': [], 'Corrected_acb_windspeed': []}

for flight in master_ACB:
    for wind_alt in flight:
        date = wind_alt['Date']
        windspeed = wind_alt['Winds_mean']
        altitude = wind_alt['Alts_mean']

        for wind_mean, alt_mean in zip(windspeed, altitude):
            # Apply the formula
            new_windspeed = wind_mean * (np.log(Z10/Z0) / np.log(alt_mean / Z0))

            corrected_calc_Acb['Date'].append(date)
            corrected_calc_Acb['Corrected_acb_windspeed'].append(new_windspeed)
for date, wind_mean in zip(corrected_calc_Acb['Date'], corrected_calc_Acb['Corrected_acb_windspeed']):
    print(f"Date: {date}, Corrected_acb_windspeed: {wind_mean}")
# %%
Z0 = 0.02  # meters (typical value for open ocean)
Z10 = 10  # target height m

corrected_calc_Acb = {'Date': [], 'Corrected_acb_windspeed': [], 'Min_alt': [], 'Max_alt': []}

for flight in master_ACB:
    for wind_alt in flight:
        date = wind_alt['Date']
        windspeed = wind_alt['Winds_mean']
        altitude = wind_alt['Alts_mean']

        if len(altitude) > 0:
            min_alt = np.nanmin(altitude)
            max_alt = np.nanmax(altitude)
        else:
            min_alt = np.nan
            max_alt = np.nan

        for wind_mean, alt_mean in zip(windspeed, altitude):
            # Apply the formula
            new_windspeed = wind_mean * (np.log(Z10/Z0) / np.log(alt_mean / Z0))

            corrected_calc_Acb['Date'].append(date)
            corrected_calc_Acb['Corrected_acb_windspeed'].append(new_windspeed)
            corrected_calc_Acb['Min_alt'].append(min_alt)
            corrected_calc_Acb['Max_alt'].append(max_alt)

for date, wind_mean, min_alt, max_alt in zip(corrected_calc_Acb['Date'], corrected_calc_Acb['Corrected_acb_windspeed'], corrected_calc_Acb['Min_alt'], corrected_calc_Acb['Max_alt']):
    print(f"Date: {date}, Corrected_acb_windspeed: {wind_mean}, Min_alt: {min_alt}, Max_alt: {max_alt}")
#%%
# Collect all altitude values from the dataset
# Collect all altitude values from the dataset
all_altitudes = []

for flight in master_ACB:
    for wind_alt in flight:
        altitude = wind_alt['Alts_mean']
        all_altitudes.extend(altitude)  # Flatten all altitude values into a single list

# Convert to a NumPy array for computation
all_altitudes = np.array(all_altitudes)

# Filter out negative values and NaNs
valid_altitudes = all_altitudes[(all_altitudes >= 0) & (~np.isnan(all_altitudes))]

# Compute min, max, and mean altitude
min_alt = np.nanmin(valid_altitudes)
max_alt = np.nanmax(valid_altitudes)
mean_alt = np.nanmean(valid_altitudes)

# Print the results
print(f"Minimum Altitude (excluding negatives): {min_alt}")
print(f"Maximum Altitude: {max_alt}")
print(f"Average Altitude: {mean_alt}")



# %%
#Checking average flight leg altitude for below cloud top 
master_BCT = []


for i in range(len(dates_legs)):
    date = dates_legs[i]

    leg_dict = leg_data[i]

    flight_date = leg_dict['Date'] 
    BCT_start = leg_dict['LegIndex_04']['StartTimes']
    BCT_stop = leg_dict['LegIndex_04']['StopTimes']
    sum_flight = summary[i]

    times = sum_flight.Time_mid.values
    winds = sum_flight.Wind_Speed.values
    alts = sum_flight.GPS_altitude.values
    
    all_BCT_means = []


    for i in range(len(BCT_start)):
        index1_start=None
        index1_end=None  
        start = int(BCT_start[i])
        end = BCT_stop[i]

        wind_alt = {
            'Date': date,
            'BCT_start': start,
            'BCT_end': end,
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

        all_BCT_means.append(wind_alt) #List that contains all the BCB wind/alt mean dictionaries for 1 flight
        
    master_BCT.append(all_BCT_means) #List that contains all BCB flights  
#%%
Z0 = 0.02  # meters (typical value for open ocean)
Z10 = 10  # target height m

corrected_calc_bct = {'Date': [], 'Corrected_bct_windspeed': []}

for flight in master_BCT:
    for wind_alt in flight:
        date = wind_alt['Date']
        windspeed = wind_alt['Winds_mean']
        altitude = wind_alt['Alts_mean']

        for wind_mean, alt_mean in zip(windspeed, altitude):
            # Apply the formula
            new_windspeed = wind_mean * (np.log(Z10/Z0) / np.log(alt_mean / Z0))

            corrected_calc_bct['Date'].append(date)
            corrected_calc_bct['Corrected_bct_windspeed'].append(new_windspeed)
for date, wind_mean in zip(corrected_calc_bct['Date'], corrected_calc_bct['Corrected_bct_windspeed']):
    print(f"Date: {date}, Corrected_bct_windspeed: {wind_mean}")
# %%
Z0 = 0.02  # meters (typical value for open ocean)
Z10 = 10  # target height m

corrected_calc_bct = {'Date': [], 'Corrected_bct_windspeed': [], 'Min_alt': [], 'Max_alt': []}

for flight in master_BCT:
    for wind_alt in flight:
        date = wind_alt['Date']
        windspeed = wind_alt['Winds_mean']
        altitude = wind_alt['Alts_mean']

        if len(altitude) > 0:
            min_alt = np.nanmin(altitude)
            max_alt = np.nanmax(altitude)
        else:
            min_alt = np.nan
            max_alt = np.nan

        for wind_mean, alt_mean in zip(windspeed, altitude):
            # Apply the formula
            new_windspeed = wind_mean * (np.log(Z10/Z0) / np.log(alt_mean / Z0))

            corrected_calc_bct['Date'].append(date)
            corrected_calc_bct['Corrected_bct_windspeed'].append(new_windspeed)
            corrected_calc_bct['Min_alt'].append(min_alt)
            corrected_calc_bct['Max_alt'].append(max_alt)

for date, wind_mean, min_alt, max_alt in zip(corrected_calc_bct['Date'], corrected_calc_bct['Corrected_bct_windspeed'], corrected_calc_bct['Min_alt'], corrected_calc_bct['Max_alt']):
    print(f"Date: {date}, Corrected_bct_windspeed: {wind_mean}, Min_alt: {min_alt}, Max_alt: {max_alt}")
#%%
# Collect all altitude values from the dataset
# Collect all altitude values from the dataset
all_altitudes_BCT = []

for flight in master_BCT:
    for wind_alt in flight:
        altitude = wind_alt['Alts_mean']
        all_altitudes_BCT.extend(altitude)  # Flatten all altitude values into a single list

# Convert to a NumPy array for computation
all_altitudes_BCT = np.array(all_altitudes_BCT)

# Filter out negative values and NaNs
valid_altitudes_BCT = all_altitudes_BCT[(all_altitudes_BCT >= 0) & (~np.isnan(all_altitudes_BCT))]

# Compute min, max, and mean altitude
min_alt = np.nanmin(valid_altitudes_BCT)
max_alt = np.nanmax(valid_altitudes_BCT)
mean_alt = np.nanmean(valid_altitudes_BCT)

# Print the results
print(f"Minimum Altitude (excluding negatives): {min_alt}")
print(f"Maximum Altitude: {max_alt}")
print(f"Average Altitude: {mean_alt}")

# %%
#Checking average flight leg altitude for below cloud base
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
# %%
Z0 = 0.02  # meters (typical value for open ocean)
Z10 = 10  # target height m

corrected_calc_bcb = {'Date': [], 'Corrected_bcb_windspeed': [], 'Min_alt': [], 'Max_alt': []}

for flight in master_BCB:
    for wind_alt in flight:
        date = wind_alt['Date']
        windspeed = wind_alt['Winds_mean']
        altitude = wind_alt['Alts_mean']

        if len(altitude) > 0:
            min_alt = np.nanmin(altitude)
            max_alt = np.nanmax(altitude)
        else:
            min_alt = np.nan
            max_alt = np.nan

        for wind_mean, alt_mean in zip(windspeed, altitude):
            # Apply the formula
            new_windspeed = wind_mean * (np.log(Z10/Z0) / np.log(alt_mean / Z0))

            corrected_calc_bcb['Date'].append(date)
            corrected_calc_bcb['Corrected_bcb_windspeed'].append(new_windspeed)
            corrected_calc_bcb['Min_alt'].append(min_alt)
            corrected_calc_bcb['Max_alt'].append(max_alt)

for date, wind_mean, min_alt, max_alt in zip(corrected_calc_bcb['Date'], corrected_calc_bcb['Corrected_bcb_windspeed'], corrected_calc_bcb['Min_alt'], corrected_calc_bcb['Max_alt']):
    print(f"Date: {date}, Corrected_bcb_windspeed: {wind_mean}, Min_alt: {min_alt}, Max_alt: {max_alt}")
#%%
# Collect all altitude values from the dataset
# Collect all altitude values from the dataset
all_altitudes_BCB = []

for flight in master_BCB:
    for wind_alt in flight:
        altitude = wind_alt['Alts_mean']
        all_altitudes_BCB.extend(altitude)  # Flatten all altitude values into a single list

# Convert to a NumPy array for computation
all_altitudes_BCB = np.array(all_altitudes_BCB)

# Filter out negative values and NaNs
valid_altitudes_BCB = all_altitudes_BCB[(all_altitudes_BCB >= 0) & (~np.isnan(all_altitudes_BCB))]

# Compute min, max, and mean altitude
min_alt = np.nanmin(valid_altitudes_BCB)
max_alt = np.nanmax(valid_altitudes_BCB)
mean_alt = np.nanmean(valid_altitudes_BCB)

# Print the results
print(f"Minimum Altitude (excluding negatives): {min_alt}")
print(f"Maximum Altitude: {max_alt}")
print(f"Average Altitude: {mean_alt}")
# %%
#Checking average flight leg altitude for min alt
master_Min = []


for i in range(len(dates_legs)):
    date = dates_legs[i]

    leg_dict = leg_data[i]

    flight_date = leg_dict['Date'] 
    Min_start = leg_dict['LegIndex_06']['StartTimes']
    Min_stop = leg_dict['LegIndex_06']['StopTimes']
    sum_flight = summary[i]

    times = sum_flight.Time_mid.values
    winds = sum_flight.Wind_Speed.values
    alts = sum_flight.GPS_altitude.values
    
    all_Min_means = []


    for i in range(len(Min_start)):
        index1_start=None
        index1_end=None  
        start = int(Min_start[i])
        end = Min_stop[i]

        wind_alt = {
            'Date': date,
            'Min_start': start,
            'Min_end': end,
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

        all_Min_means.append(wind_alt) #List that contains all the BCB wind/alt mean dictionaries for 1 flight
        
    master_Min.append(all_Min_means) #List that contains all BCB flights  
#%%
Z0 = 0.02  # meters (typical value for open ocean)
Z10 = 10  # target height m

corrected_calc_min = {'Date': [], 'Corrected_min_windspeed': []}

for flight in master_Min:
    for wind_alt in flight:
        date = wind_alt['Date']
        windspeed = wind_alt['Winds_mean']
        altitude = wind_alt['Alts_mean']

        for wind_mean, alt_mean in zip(windspeed, altitude):
            # Apply the formula
            new_windspeed = wind_mean * (np.log(Z10/Z0) / np.log(alt_mean / Z0))

            corrected_calc_min['Date'].append(date)
            corrected_calc_min['Corrected_min_windspeed'].append(new_windspeed)
for date, wind_mean in zip(corrected_calc_min['Date'], corrected_calc_min['Corrected_min_windspeed']):
    print(f"Date: {date}, Corrected_min_windspeed: {wind_mean}")
# %%
Z0 = 0.02  # meters (typical value for open ocean)
Z10 = 10  # target height m

corrected_calc_min = {'Date': [], 'Corrected_min_windspeed': [], 'Min_alt': [], 'Max_alt': []}

for flight in master_Min:
    for wind_alt in flight:
        date = wind_alt['Date']
        windspeed = wind_alt['Winds_mean']
        altitude = wind_alt['Alts_mean']

        if len(altitude) > 0:
            min_alt = np.nanmin(altitude)
            max_alt = np.nanmax(altitude)
        else:
            min_alt = np.nan
            max_alt = np.nan

        for wind_mean, alt_mean in zip(windspeed, altitude):
            # Apply the formula
            new_windspeed = wind_mean * (np.log(Z10/Z0) / np.log(alt_mean / Z0))

            corrected_calc_min['Date'].append(date)
            corrected_calc_min['Corrected_min_windspeed'].append(new_windspeed)
            corrected_calc_min['Min_alt'].append(min_alt)
            corrected_calc_min['Max_alt'].append(max_alt)

for date, wind_mean, min_alt, max_alt in zip(corrected_calc_min['Date'], corrected_calc_min['Corrected_min_windspeed'], corrected_calc_min['Min_alt'], corrected_calc_min['Max_alt']):
    print(f"Date: {date}, Corrected_min_windspeed: {wind_mean}, Min_alt: {min_alt}, Max_alt: {max_alt}")
#%%
# Collect all altitude values from the dataset
# Collect all altitude values from the dataset
all_altitudes_Min = []

for flight in master_Min:
    for wind_alt in flight:
        altitude = wind_alt['Alts_mean']
        all_altitudes_Min.extend(altitude)  # Flatten all altitude values into a single list

# Convert to a NumPy array for computation
all_altitudes_Min = np.array(all_altitudes_Min)

# Filter out negative values and NaNs
valid_altitudes_Min = all_altitudes_Min[(all_altitudes_Min >= 0) & (~np.isnan(all_altitudes_Min))]

# Compute min, max, and mean altitude
min_alt = np.nanmin(valid_altitudes_Min)
max_alt = np.nanmax(valid_altitudes_Min)
mean_alt = np.nanmean(valid_altitudes_Min)

# Print the results
print(f"Minimum Altitude (excluding negatives): {min_alt}")
print(f"Maximum Altitude: {max_alt}")
print(f"Average Altitude: {mean_alt}")
# %%
#trying to predict GCCN flight label based on features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.DataFrame({
    'Nr_Nc': np.concatenate([filtered_high_concentration, filtered_low_concentration]),  # X1
    'LWC': np.concatenate([filtered_high_lwc, filtered_low_lwc]),  # X2
    'RWC_LWC': np.concatenate([filtered_high_rwc / filtered_high_lwc, filtered_low_rwc / filtered_low_lwc]),  # X3
    'GCCN_Label': np.concatenate([np.ones(len(filtered_high_concentration)), np.zeros(len(filtered_low_concentration))])  # Y (1=High, 0=Low)
})

df = df.dropna()
X = df[['Nr_Nc', 'LWC', 'RWC_LWC']]
Y = df['GCCN_Label']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)
Y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Model Accuracy: {accuracy:.2%}")
print("Classification Report:\n", classification_report(Y_test, Y_pred))
feature_importances = rf_model.feature_importances_
features = X.columns
plt.figure(figsize=(6, 4))
sns.barplot(x=feature_importances, y=features, palette="Blues")
plt.xlabel("Feature Importance", fontsize=14, fontweight='bold')
plt.ylabel("Feature", fontsize=14, fontweight='bold')
plt.title("Feature Importance for GCCN Classification", fontsize=16, fontweight='bold')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()
#%%
