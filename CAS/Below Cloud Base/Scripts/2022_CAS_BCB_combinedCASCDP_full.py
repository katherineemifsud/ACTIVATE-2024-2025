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
#beginning with CAS code 

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
# %%
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

#Attempting to use np.where instead of align to search for leg times across multiple files. 
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
#Double check the number of legs associated with each date to compare across multiple instruments.  
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

#This method uses np.where to move across each file.
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

    # Convert times and data to numeric arrays for filtering
    CAS_times = np.array(CAS_flight['Time_mid'], dtype=float)
    TwoDS_times = np.array(twoDS_flight['Time_Start'], dtype=float)

    lwc = np.array(CAS_flight['LWC_CAS'], dtype=float)
    N_total = np.array(twoDS_flight['N-total_2DS'], dtype=float)

    # Pre-fetch bin values for range 12 to 29
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

        # Use np.where to find indices within the BCB interval
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

        # Calculate means
        for bin_label in range(12, 30):
            for label in ['Y', 'N']:
                bin_key = f'Bin{bin_label:02d}_{label}_mean'
                if bin_means[bin_key]:
                    bin_means[bin_key] = np.nanmean(bin_means[bin_key])
                else:
                    bin_means[bin_key] = np.nan

        total_BCB_means.append(bin_means)

    master_CAS_BCB.append(total_BCB_means)

# Print or use master_CAS_BCB as needed
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

            Y_calc[bin_key_Y] = np.nanmean(bin_means[bin_key_Y]) * Logg[bin_label - 12]
            N_calc[bin_key_N] = np.nanmean(bin_means[bin_key_N]) * Logg[bin_label - 12]

        Y_BCB_calc.append(Y_calc)
        N_BCB_calc.append(N_calc)
#%%
#Create ambient size distributions for each leg.  
all_bin_means = []

for entry in Y_BCB_calc:
    bin_means = []
    for i in range(12, 30):  # Bins from 12 to 29
        key = f'Bin{i}_Y_mean'
        bin_means.append(entry.get(key, np.nan))

    # Append the bin means along with the date and leg times
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
added_dates = set()  # To track which dates have been added to the legend

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
plt.ylabel('Clear mean droplet concentration \n (/cm^3)', fontsize=12, fontweight='bold')
plt.title('Below Cloud Base clear sky distribution \n January-June 2022', fontsize=12, fontweight='bold')
plt.xticks(np.arange(0, 50, 2))
num_cols = 7
plt.legend(title='Date', ncol=num_cols, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True)
plt.tight_layout()
plt.show()

#%%
#Fit an exponential function to the whole data set.  

def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

all_bin_means = []

for entry in Y_BCB_calc:
    bin_means = []
    for i in range(12, 30):  # Bins from 12 to 29
        key = f'Bin{i}_Y_mean'
        bin_means.append(entry.get(key, np.nan))

    # Append the bin means along with the date and leg times
    all_bin_means.append({
        'Date': entry['Date'],
        'BCB_start': entry.get('BCB_start', np.nan),
        'BCB_stop': entry.get('BCB_stop', np.nan),
        'Bin_means': bin_means
    })

# Get unique dates and colors for plotting
unique_dates = sorted(set(entry['Date'] for entry in all_bin_means))
colors = cm.viridis(np.linspace(0, 1, len(unique_dates)))
date_color_map = {date: color for date, color in zip(unique_dates, colors)}

# Scatter plot
plt.figure(figsize=(10, 6))
added_dates = set()  # To track which dates have been added to the legend

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

# Flatten the bin_center and all_bin_means for fitting
flat_bin_centers = np.tile(bin_center, len(all_bin_means))
flat_bin_means = np.concatenate([np.array(entry['Bin_means']) for entry in all_bin_means])

# Convert to NumPy array with a float type
flat_bin_centers = np.array(flat_bin_centers, dtype=float)
flat_bin_means = np.array(flat_bin_means, dtype=float)

# Remove NaN values
valid_indices = ~np.isnan(flat_bin_means)
flat_bin_centers = flat_bin_centers[valid_indices]
flat_bin_means = flat_bin_means[valid_indices]

# Fit the exponential function to the data
popt, pcov = curve_fit(exponential, flat_bin_centers, flat_bin_means, p0=(1, 1))

# Extract the intercept (n0) and e-folding diameter (D)
n0 = popt[0]
D = popt[1]

# Print the intercept and e-folding diameter
print(f"Intercept (n0): {n0:.2e}")
print(f"E-folding diameter (D): {D:.2f} um")

# Plot the fitted curve
x_fit = np.linspace(min(bin_center), max(bin_center), 100)
y_fit = exponential(x_fit, *popt)
plt.plot(x_fit, y_fit, color='red', label=f'Exponential fit: y = {n0:.2e} * exp(-x / {D:.2f})')

# Add labels, title, and legend
plt.xlabel('Bin centers diameter (um)', fontsize=12, fontweight='bold')
plt.ylabel('Clear mean droplet concentration \n (/cm^3)', fontsize=12, fontweight='bold')
plt.title('Below Cloud Base clear sky distribution \n January-June 2022', fontsize=12, fontweight='bold')
# plt.yscale('log')
plt.xticks(np.arange(0, 50, 5))
num_cols = 7
plt.legend(title='Date', ncol=num_cols, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True)
plt.tight_layout()
# plt.ylim(10**-5, 10**1)
plt.show()

#%%
#Fit an exponential function to each leg.

# Define the exponential function
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

# Process the data
all_bin_means = []
dates = []
for entry in Y_BCB_calc:
    bin_means = []
    for i in range(12, 30):  # Bins from 12 to 29
        key = f'Bin{i}_Y_mean'
        bin_means.append(entry.get(key, np.nan))
    all_bin_means.append({
        'Date': entry['Date'],
        'BCB_start': entry.get('BCB_start', np.nan),
        'BCB_stop': entry.get('BCB_stop', np.nan),
        'Bin_means': bin_means
    })
    dates.append(entry['Date'])

# Get unique dates and colors for plotting
unique_dates = sorted(set(dates))
colors = cm.viridis(np.linspace(0, 1, len(unique_dates)))
date_color_map = {date: color for date, color in zip(unique_dates, colors)}

# Define the color for the fits
fit_color = 'purple'

# Scatter plot
plt.figure(figsize=(10, 6))
added_dates = set()  # To track which dates have been added to the legend

# Loop through each leg's data
for entry in all_bin_means:
    bin_means = np.array(entry['Bin_means'])
    valid_indices = ~np.isnan(bin_means)
    date = entry['Date']
    color = date_color_map[date]
    
    # Skip if bin_means is empty
    if not valid_indices.any():
        print(f"No valid data for date {date}")
        continue

    bin_centers = np.array(bin_center)[valid_indices]
    bin_means = bin_means[valid_indices]
    
    # Fit the exponential function to the data of each leg
    try:
        popt, _ = curve_fit(exponential, bin_centers, bin_means, p0=(1, 1))
        n0, D = popt
        
        # Print the intercept and e-folding diameter for each leg
        print(f"Date: {date}")
        print(f"  Intercept (n0): {n0:.2e}")
        print(f"  E-folding diameter (D): {D:.2f} um")
        
        # Plot the fitted curve for each leg
        x_fit = np.linspace(min(bin_centers), max(bin_centers), 100)
        y_fit = exponential(x_fit, *popt)
        plt.plot(x_fit, y_fit, color=fit_color, label=f'{date} fit: y = {n0:.2e} * exp(-x / {D:.2f})')

    except RuntimeError:
        print(f"Fit could not be performed for date {date}")
    
    # Plot the data points for each leg
    if date not in added_dates:
        plt.scatter(bin_centers, bin_means, color=color, marker='o', label=date)
        added_dates.add(date)
    else:
        plt.scatter(bin_centers, bin_means, color=color, marker='o')

# Add labels, title, and legend
plt.xlabel('Bin centers diameter (um)', fontsize=12, fontweight='bold')
plt.ylabel('Clear mean droplet concentration \n (/cm^3/um)', fontsize=12, fontweight='bold')
plt.title('Below Cloud Base clear sky distribution \n January-June 2022', fontsize=12, fontweight='bold')
plt.yscale('log')
plt.xticks(np.arange(0, 50, 5))
num_cols = 7
plt.legend(title='Date', ncol=num_cols, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True)
plt.tight_layout()
plt.show()
#%%
#Plot the exponential fits for each size distribution and extract the efolding diameter and intercept for each leg. 
# Define the exponential function
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

# Initialize the dictionary
master_BCB_exponential = {}

# Process the data
for entry in all_bin_means:
    bin_means = np.array(entry['Bin_means'], dtype=float)  # Ensure bin_means is a float array
    valid_indices = ~np.isnan(bin_means)
    bin_centers = np.array(bin_center)[valid_indices]
    bin_means = bin_means[valid_indices]
    
    # Skip if bin_means is empty
    if not valid_indices.any():
        print(f"No valid data for date {entry['Date']}")
        continue
    
    # Fit the exponential function to the data
    try:
        popt, _ = curve_fit(exponential, bin_centers, bin_means, p0=(1, 1))
        n0, D = popt
        
        # Store the n0 and D values in the master_min_exponential dictionary
        if entry['Date'] not in master_BCB_exponential:
            master_BCB_exponential[entry['Date']] = []
        
        master_BCB_exponential[entry['Date']].append({
            'Date': entry['Date'],
            'BCB_start': entry.get('BCB_start', np.nan),
            'BCB_stop': entry.get('BCB_stop', np.nan),
            'n0': n0,
            'D': D
        })
        
        # Print the intercept and e-folding diameter for each leg
        print(f"Date: {entry['Date']}")
        print(f"  Intercept (n0): {n0:.2e}")
        print(f"  E-folding diameter (D): {D:.2f} um")
        
        # Plot the fitted curve for each leg
        x_fit = np.linspace(min(bin_centers), max(bin_centers), 100)
        y_fit = exponential(x_fit, *popt)
        plt.plot(x_fit, y_fit, color='purple', label=f'{entry["Date"]} fit: y = {n0:.2e} * exp(-x / {D:.2f})')

    except RuntimeError:
        print(f"Fit could not be performed for date {entry['Date']}")
    
    # Plot the data points for each leg
    if entry['Date'] not in added_dates:
        plt.scatter(bin_centers, bin_means, color='purple', marker='o', label=entry['Date'])
        added_dates.add(entry['Date'])
    else:
        plt.scatter(bin_centers, bin_means, color='purple', marker='o')

# Add labels, title, and legend
plt.xlabel('Bin centers diameter (um)', fontsize=12, fontweight='bold')
plt.ylabel('Clear mean droplet concentration \n (/cm^3/um)', fontsize=12, fontweight='bold')
plt.title('Below Cloud Base clear sky distribution \n January-June 2022', fontsize=12, fontweight='bold')
plt.xticks(np.arange(0, 50, 5))
plt.tight_layout()
plt.show()
#%%
# Print the number of inner dictionaries for each date in master_BCB_exponential
for date, entries in master_BCB_exponential.items():
    print(f"Date: {date} has {len(entries)} entries")
total_entries = sum(len(entries) for entries in master_BCB_exponential.values())
print(f"Total number of inner dictionaries: {total_entries}")
#%%
# Loop through each date and its entries in master_BCB_exponential
for date, entries in master_BCB_exponential.items():
    print(f"Date: {date}")
    for entry in entries:
        start_time = entry.get('BCB_start', 'N/A')  
        stop_time = entry.get('BCB_stop', 'N/A')   
        print(f"  Start Time: {start_time}, Stop Time: {stop_time}")

# Optionally, count the total number of leg times for verification
total_legs = sum(len(entries) for entries in master_BCB_exponential.values())
print(f"Total number of legs: {total_legs}")
#%%


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
date_leg_set = set()
for entries in master_BCB_exponential.values():
    for entry in entries:
        date = entry['Date']
        BCB_start = entry.get('BCB_start', np.nan)
        BCB_stop = entry.get('BCB_stop', np.nan)
        date_leg_set.add((date, BCB_start, BCB_stop))

# Filter master_BCB_RH
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
# Flatten the list of lists and count the total number of entries
total_entries_filtered_master_BCB_RH = sum(len(legs) for legs in filtered_master_BCB_RH)
print(f"Total entries in filtered_master_BCB_RH: {total_entries_filtered_master_BCB_RH}")
#%%
##Obtaining g(RH) = [1.7 / (1-RH)]^0.31 for all mean RH values 
## ie for every leg 

master_BCB_gRH = []

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
    
    master_BCB_gRH.append(flight_gRH)
#%%
#only the grh from filtered_master_BCB_RH
filtered_master_BCB_gRH = []

# Iterate over each flight in master_BCB_RH
for flight in filtered_master_BCB_RH:
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
    
    filtered_master_BCB_gRH.append(flight_gRH)
#%%
total_entries_filtered_master_BCB_gRH = sum(len(legs) for legs in filtered_master_BCB_gRH)
print(f"Total entries in filtered_master_BCB_gRH: {total_entries_filtered_master_BCB_gRH}")
#%%
#filtered dry intercept calculation

filtered_master_BCB_interceptdry_dict = {}

# Flatten master_min_gRH if it's a list of lists
if isinstance(filtered_master_BCB_gRH[0], list):
    filtered_master_BCB_gRH = [item for sublist in filtered_master_BCB_gRH for item in sublist]


unique_keys = set()

# Flatten master_BCB_exponential
flattened_exponential = []
for exp_list in master_BCB_exponential.values():
    flattened_exponential.extend(exp_list)

# Create a dictionary for quick lookup
exponential_dict = {(exp['Date'], exp['BCB_start'], exp['BCB_stop']): exp for exp in flattened_exponential}

# Loop through each entry in filtered_master_BCB_gRH
for entry in filtered_master_BCB_gRH:
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

    # Find corresponding exponential parameters
    if date in master_BCB_exponential:
        exp_params_list = master_BCB_exponential[date]
        for exp_params in exp_params_list:
            n0 = exp_params['n0']
            D = exp_params['D']
            
            
            dryintercept = n0 * (gRh_mean)
            
            # Store the result in the dictionary
            filtered_master_BCB_interceptdry_dict[key] = {
                'Date': date,
                'BCB_start': BCB_start,
                'BCB_stop': BCB_stop,
                'Rh_mean': entry['Rh_mean'],
                'gRh_mean': entry['gRh_mean'],
                'dry intercept': dryintercept
            }

filtered_master_BCB_dryintercept = list(filtered_master_BCB_interceptdry_dict.values())
print(f"Length of filtered_master_BCB_dryintercept: {len(filtered_master_BCB_dryintercept)}")
#%%
#filtered total concentration of droplets with dry diameter larger than ddrymin
filtered_master_BCB_ntd_dict = {}

# Flatten master_BCB_gRH if it's a list of lists
if isinstance(filtered_master_BCB_gRH[0], list):
    filtered_master_BCB_gRH = [item for sublist in filtered_master_BCB_gRH for item in sublist]


unique_keys = set()

# Flatten master_BCB_exponential
flattened_exponential = []
for exp_list in master_BCB_exponential.values():
    flattened_exponential.extend(exp_list)

# Create a dictionary for quick lookup
exponential_dict = {(exp['Date'], exp['BCB_start'], exp['BCB_stop']): exp for exp in flattened_exponential}

ddrymin = 2  # Update this as needed

# Loop through each entry in master_BCB_gRH
for entry in filtered_master_BCB_gRH:
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

    # Find corresponding exponential parameters
    if date in master_BCB_exponential:
        exp_params_list = master_BCB_exponential[date]
        for exp_params in exp_params_list:
            D = exp_params['D']
            n0 = exp_params['n0']
            # dryintercept = n0 * gRh_mean
            # Calculate Ntd
            Ntd = n0 * D * np.exp(-(gRh_mean * ddrymin / D))
            
            # Store the result in the dictionary
            filtered_master_BCB_ntd_dict[key] = {
                'Date': date,
                'BCB_start': BCB_start,
                'BCB_stop': BCB_stop,
                'Rh_mean': entry['Rh_mean'],
                'gRh_mean': entry['gRh_mean'],
                'Ntd': Ntd
            }

filtered_master_BCB_ntd = list(filtered_master_BCB_ntd_dict.values())
print(f"Length of filtered_master_BCB_ntd: {len(filtered_master_BCB_ntd)}")
#%%
#filtered total concentration of droplets with dry diameter larger than ddrymin (ntd)
# related to the ambient concentration (NT)

filtered_master_BCB_NtdNt_dict = {}
# Flatten master_BCB_gRH if it's a list of lists
if isinstance(filtered_master_BCB_gRH[0], list):
    filtered_master_BCB_gRH = [item for sublist in filtered_master_BCB_gRH for item in sublist]

ddrymin = 2
dmin = 2.5  

unique_keys = set()
flattened_exponential = []
for exp_list in master_BCB_exponential.values():
    flattened_exponential.extend(exp_list)

exponential_dict = {(exp['Date'], exp['BCB_start'], exp['BCB_stop']): exp for exp in flattened_exponential}

# Loop through each entry in filtered_master_BCB_gRH
for entry in filtered_master_BCB_gRH:
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
        
       
        filtered_master_BCB_NtdNt_dict[key] = {
            'Date': date,
            'BCB_start': BCB_start,
            'BCB_stop': BCB_stop,
            'Rh_mean': entry['Rh_mean'],
            'gRh_mean': entry['gRh_mean'],
            'NtdNt': NtdNt
        }

filtered_master_BCB_NtdNt = list(filtered_master_BCB_NtdNt_dict.values())
print(f"Length of filtered_master_BCB_NtdNt: {len(filtered_master_BCB_NtdNt)}")
#%%
#Now we need to pull our windspeed values from the summary data and calculate the corrected windspeeds down to 10m
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
#PLotting density contours for dry intercept and slope with corrected windspeeds
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

# for i, flight in enumerate(master_BCB):
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
#                 gRh_mean = filtered_master_BCB_gRH[i][j]['gRh_mean'][0] 

#                 for exp_params in exp_params_list:
#                     D = exp_params['D']
#                     n0 = exp_params['n0']
#                     dryintercept = n0 * gRh_mean
                    
#                     # Append data to the combined dictionary
#                     combined_data['Date'].append(date)
#                     combined_data['D'].append(D)
#                     combined_data['dryintercept'].append(dryintercept)
#                     combined_data['Windspeed'].append(corrected_windspeed)

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

# # Check for NaN or Inf values
# if df_combined[['D', 'dryintercept']].isnull().any().any():
#     print("Data contains NaN values.")
#     # Filter out NaN values
#     df_combined = df_combined.dropna(subset=['D', 'dryintercept'])

# if np.any(np.isinf(df_combined[['D', 'dryintercept']].values)):
#     print("Data contains Inf values.")
#     # Optionally filter out inf values
#     df_combined = df_combined[np.isfinite(df_combined[['D', 'dryintercept']].values).all(axis=1)]

# # Filter out NaN and Inf values for plotting
# filtered_combined = df_combined[df_combined['Windspeed'].notna() & df_combined['D'].notna() & df_combined['dryintercept'].notna()]
# filtered_combined = filtered_combined[np.isfinite(filtered_combined[['D', 'dryintercept']].values).all(axis=1)]

# # Plot data points with valid Windspeed values
# sc = plt.scatter(filtered_combined['D'], filtered_combined['dryintercept'], 
#                  c=filtered_combined['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

# # Plot data points with NaN Windspeed values in black
# plt.scatter(df_nan_windspeed['D'], df_nan_windspeed['dryintercept'], 
#             color='grey', s=100, label='Windspeed NaN')

# # Calculate density for contours
# kde = gaussian_kde(np.vstack([filtered_combined['D'], filtered_combined['dryintercept']]))
# xgrid = np.linspace(min(filtered_combined['D']), max(filtered_combined['D']), 100)
# ygrid = np.linspace(min(filtered_combined['dryintercept']), max(filtered_combined['dryintercept']), 100)
# X, Y = np.meshgrid(xgrid, ygrid)
# Z = kde(np.vstack([X.ravel(), Y.ravel()]))
# Z = Z.reshape(X.shape)

# # Plot contours for total concentration
# contour_levels = np.linspace(0, Z.max(), num=10)  # Adjust number of levels as needed
# plt.contour(X, Y, Z, levels=contour_levels, colors='red', alpha=0.75)

# # Add color bar
# plt.colorbar(sc, label='Corrected Windspeed (m/s)')

# # Add labels and title
# plt.xlabel('Slope', fontsize=14, fontweight='bold')
# plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
# plt.yscale('log')
# plt.title('Below Cloud Base January - June 2022 Density Contours' , fontweight='bold')
# plt.xscale('log')
# plt.xlim(10**-0.5, 10**1.2)
# plt.ylim(10**-1.7, 10**1.8)
# plt.show()
#%%
combined_data = {
    'Date': [],
    'D': [],
    'NtdNt': []
}

flat_ntdNt = [item for item in filtered_master_BCB_NtdNt]

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
plt.xlim(10**-1, 10**1)
plt.xscale('log')
plt.show()
#%%
#density contours for dry intercept and slope


# Prepare data for plotting
slope_data = combined_data['D']  # Slope values
dry_intercept_data = [entry['dry intercept'] for entry in filtered_master_BCB_dryintercept]  # Dry intercept values

# Ensure lengths match for consistency
if len(slope_data) != len(dry_intercept_data):
    print(f"Mismatch in lengths: Slope={len(slope_data)}, Dry Intercept={len(dry_intercept_data)}")
slope_data = slope_data[:len(dry_intercept_data)]  # Truncate to match lengths if necessary

# Convert data to DataFrame for filtering and consistency
df_combined = pd.DataFrame({
    'Slope': slope_data,
    'Dry Intercept': dry_intercept_data,
})

# Filter out NaN and Inf values
df_combined = df_combined.dropna(subset=['Slope', 'Dry Intercept'])
df_combined = df_combined[np.isfinite(df_combined[['Slope', 'Dry Intercept']].values).all(axis=1)]

# Extract filtered values for plotting
filtered_slope = df_combined['Slope']
filtered_dry_intercept = df_combined['Dry Intercept']

# Plot data points
plt.figure(figsize=(10, 8))
plt.scatter(filtered_slope, filtered_dry_intercept, c='blue', s=80, alpha=0.7)

# Calculate density for contours
kde = gaussian_kde(np.vstack([filtered_slope, filtered_dry_intercept]))
xgrid = np.logspace(np.log10(filtered_slope.min()), np.log10(filtered_slope.max()), 100)
ygrid = np.logspace(np.log10(filtered_dry_intercept.min()), np.log10(filtered_dry_intercept.max()), 100)
X, Y = np.meshgrid(xgrid, ygrid)
Z = kde(np.vstack([X.ravel(), Y.ravel()]))
Z = Z.reshape(X.shape)

# Plot density contours
contour_levels = np.linspace(0, Z.max(), num=10)  # Adjust number of levels as needed
plt.contour(X, Y, Z, levels=contour_levels, colors='red', alpha=0.75)

# Add labels and title
plt.xlabel('Slope', fontsize=19, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=19, fontweight='bold')
plt.title('Below Cloud Base January - June 2022', fontsize=19, fontweight='bold')

# Set logarithmic scales
plt.xscale('log')
plt.yscale('log')

# Set axis limits
plt.xlim(10**-0.5, 10**1.2)
plt.ylim(10**-1, 10**1.7)

# Adjust tick sizes
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')

# Finalize and show the plot
plt.tight_layout()
plt.legend(fontsize=14)
plt.show()

#%%
#Fixing combined_clean points
# # Constants for windspeed correction
# Z0 = 0.02
# Z10 = 10  

# # Windspeed correction function
# def correct_windspeed(windspeed, altitude):
#     return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))

# # Initialize combined_data
# combined_data = {
#     'Date': [],
#     'D': [],
#     'dryintercept': [],
#     'Windspeed': []
# }

# # Process the data
# for i, flight in enumerate(master_BCB):
#     for j, wind_alt in enumerate(flight):
#         try:
#             # Extract flight information
#             date = wind_alt['Date']
#             wind_mean = wind_alt['Winds_mean'][0]
#             alt_mean = wind_alt['Alts_mean'][0]

#             # Correct windspeed
#             if not np.isnan(alt_mean) and alt_mean > 0:
#                 corrected_windspeed = correct_windspeed(wind_mean, alt_mean)
#             else:
#                 corrected_windspeed = np.nan

#             # Check if date exists in master_BCB_exponential
#             if date in master_BCB_exponential:
#                 exp_params_list = master_BCB_exponential[date]
#                 gRh_mean = filtered_master_BCB_gRH[i][j]['gRh_mean'][0]

#                 # Process each entry in exp_params_list only once per leg
#                 if len(exp_params_list) == 1:  # Expecting one entry per leg
#                     exp_params = exp_params_list[0]
#                     D = exp_params['D']
#                     n0 = exp_params['n0']
#                     dryintercept = n0 * gRh_mean

#                     # Append data to combined dictionary
#                     combined_data['Date'].append(date)
#                     combined_data['D'].append(D)
#                     combined_data['dryintercept'].append(dryintercept)
#                     combined_data['Windspeed'].append(corrected_windspeed)

#                 elif len(exp_params_list) > 1:
#                     print(f"Warning: Multiple entries in exp_params_list for date {date}, taking first entry.")
#                     exp_params = exp_params_list[0]
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

# # Convert combined_data to DataFrame
# df_combined = pd.DataFrame(combined_data)

# # Check for duplicates and unique dates
# unique_dates = df_combined['Date'].nunique()
# print(f"Number of unique dates: {unique_dates}")
# print(f"Total entries in combined_data: {len(df_combined)}")

# # Optional: Drop duplicates if necessary
# df_combined = df_combined.drop_duplicates(subset=['Date', 'D', 'dryintercept'])

# # Final DataFrame
# print(df_combined.describe())

# # Filter data with valid windspeed
# filtered_combined = df_combined.dropna(subset=['Windspeed', 'D', 'dryintercept'])
# print(f"Filtered combined_data with valid windspeed: {len(filtered_combined)} entries")

#%%

# Full integrated mass calculation: M = N0 * D^4
# #Plotting mass contours for dry intercept and slope with corrected windspeeds
# def calculate_mass(N0, D):
#     return N0 * D**4

# # Filter out rows where D or dryintercept is NaN or <= 0
# filtered_combined_clean = filtered_combined[(filtered_combined['D'] > 0) & 
#                                             (filtered_combined['dryintercept'] > 0)].copy()

# # Recalculate mass for each point using the full integrated mass equation
# filtered_combined_clean['Mass'] = calculate_mass(filtered_combined_clean['dryintercept'], 
#                                                  filtered_combined_clean['D'])

# # Debugging: Check minimum and maximum mass values
# print(f"Min mass: {filtered_combined_clean['Mass'].min()}, Max mass: {filtered_combined_clean['Mass'].max()}")

# # Create data-based grids to better match the distribution of the points
# xgrid = np.logspace(np.log10(filtered_combined_clean['D'].min()), 
#                     np.log10(filtered_combined_clean['D'].max()), 100)
# ygrid = np.logspace(np.log10(filtered_combined_clean['dryintercept'].min()), 
#                     np.log10(filtered_combined_clean['dryintercept'].max()), 100)
# D_grid, dryintercept_grid = np.meshgrid(xgrid, ygrid)

# # Calculate mass for each point on the grid
# mass_grid = calculate_mass(dryintercept_grid, D_grid)

# # Debugging: Check if mass_grid contains meaningful values
# print(f"Mass grid: Min = {np.nanmin(mass_grid)}, Max = {np.nanmax(mass_grid)}")

# # Define mass contour levels (adjust to fit data range)
# mass_levels = np.logspace(np.log10(filtered_combined_clean['Mass'].min()), 
#                           np.log10(filtered_combined_clean['Mass'].max()), 20)
# print(f"Mass levels: {mass_levels}")

# # Plot the scatter plot with Windspeed color map
# sc = plt.scatter(filtered_combined_clean['D'], filtered_combined_clean['dryintercept'], 
#                  c=filtered_combined_clean['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

# # Plot NaN Windspeed points in black (if any)
# if not df_nan_windspeed.empty:
#     plt.scatter(df_nan_windspeed['D'], df_nan_windspeed['dryintercept'], 
#                 color='grey', s=100, label='Windspeed NaN')

# # Plot the mass contours using the full integrated mass formula
# contour_plot = plt.contour(D_grid, dryintercept_grid, mass_grid, levels=mass_levels, colors='red', alpha=0.75)
# if len(contour_plot.allsegs[0]) == 0:
#     print("No contours were created. Check your data range or mass grid calculation.")

# plt.colorbar(sc, label='Corrected Windspeed (m/s)')
# plt.xlabel('Slope', fontsize=14, fontweight='bold')
# plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
# plt.yscale('log')
# plt.xscale('log')
# plt.title('Below Cloud Base January - June 2022', fontsize=14, fontweight='bold')
# plt.xlim(10**-0.5, 10**1.3)
# plt.ylim(10**-1.7, 10**1.8)
# plt.show()
#%%
#Extending the contours outside designated grid 

# def calculate_mass(N0, D):
#     integrand = lambda d: np.exp(-d / D) * d**3
#     mass_integral, _ = quad(integrand, 0, np.inf)  # Integrate from 0 to infinity
#     return N0 * mass_integral

# # Define the full x and y axis limits
# x_min, x_max = 10**-0.1, 10**1.05  # Full x-axis range
# y_min, y_max = 10**-1.7, 10**1.75  # Full y-axis range

# # Create extended grids to cover the full plot range
# xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200)  # Denser grid
# ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)
# D_grid_extended, dryintercept_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)

# # Recalculate mass grid over the extended grid
# mass_grid_extended = np.zeros_like(D_grid_extended)
# for i in range(D_grid_extended.shape[0]):
#     for j in range(D_grid_extended.shape[1]):
#         mass_grid_extended[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j])

# # Define mass contour levels with finer spacing
# mass_levels = np.logspace(-2, 5, 50)  # Adjust the range and number of levels as needed

# # Plot the scatter plot with Windspeed color map
# plt.figure(figsize=(10, 8))
# sc = plt.scatter(filtered_combined_clean['D'], filtered_combined_clean['dryintercept'], 
#                  c=filtered_combined_clean['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

# # Add colorbar and labels
# cbar = plt.colorbar(sc)  # Create the colorbar
# cbar.set_label('Corrected Windspeed (m/s)', fontsize=14, fontweight='bold')  # Set label with fontsize and fontweight
# cbar.ax.tick_params(labelsize=12)  # Adjust colorbar tick size

# # Add mass contours
# contour_plot = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_extended, 
#                            levels=mass_levels, colors='red', alpha=0.75)

# # Add labels to the mass contours
# plt.clabel(contour_plot, inline=True, fontsize=10, fmt='%1.1e', use_clabeltext=True, colors='red')

# # Add axis labels and titles
# plt.xlabel('Slope', fontsize=19, fontweight='bold')
# plt.ylabel('Dry Intercept', fontsize=19, fontweight='bold')
# plt.yscale('log')
# plt.xscale('log')
# plt.title('Below Cloud Base January - June 2022', fontsize=19, fontweight='bold')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)

# # Adjust tick mark sizes
# plt.xticks(fontsize=16, fontweight='bold')
# plt.yticks(fontsize=16, fontweight='bold')

# plt.tight_layout()
# plt.show()
#%%
#plotting just slope vs dry intercept 
slope_data = combined_data['D']

# Extract dry intercept from filtered_master_BCB_dryintercept
dry_intercept_data = [entry['dry intercept'] for entry in filtered_master_BCB_dryintercept]

# Ensure the lengths of slope and dry intercept match
if len(slope_data) != len(dry_intercept_data):
    print(f"Mismatch in data lengths: Slope={len(slope_data)}, Dry Intercept={len(dry_intercept_data)}")

# Create a DataFrame for plotting
df_debug = pd.DataFrame({
    'Slope': slope_data[:len(dry_intercept_data)],  # Truncate to match lengths if needed
    'Dry Intercept': dry_intercept_data
})

# Scatter plot: Slope vs Dry Intercept
plt.figure(figsize=(10, 8))
plt.scatter(df_debug['Slope'], df_debug['Dry Intercept'], alpha=0.7, s=80, color='blue')

# Add axis labels and title
plt.xlabel('Slope', fontsize=19, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=19, fontweight='bold')
plt.title('CAS Below Cloud Base January - June 2022', fontsize=19, fontweight='bold')

# Set logarithmic scales for both axes
plt.xscale('log')
plt.yscale('log')

# Adjust axis limits if needed
plt.xlim(10**-0.1, 10**1.05)
plt.ylim(10**-1.7, 10**1.75)

# Adjust tick sizes
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')

# Show the plot
plt.tight_layout()
plt.show()
#%%
# #Extending mass contours for 2 to inf integration and adding labels 
# def calculate_mass(N0, D):
#     integrand = lambda d: np.exp(-d / D) * d**3
#     mass_integral, _ = quad(integrand, 2, np.inf)  # Integrate from 0 to infinity
#     return N0 * mass_integral

# # Define the full x and y axis limits
# x_min, x_max = 10**-0.1, 10**1.05  # Full x-axis range
# y_min, y_max = 10**-1.7, 10**1.75  # Full y-axis range

# # Create extended grids to cover the full plot range
# xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200)  # Denser grid
# ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)
# D_grid_extended, dryintercept_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)

# # Recalculate mass grid over the extended grid
# mass_grid_extended = np.zeros_like(D_grid_extended)
# for i in range(D_grid_extended.shape[0]):
#     for j in range(D_grid_extended.shape[1]):
#         mass_grid_extended[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j])

# # Define mass contour levels with finer spacing
# mass_levels = np.logspace(-2, 5, 50)  # Adjust the range and number of levels as needed

# # Plot the scatter plot with Windspeed color map
# plt.figure(figsize=(10, 8))
# sc = plt.scatter(filtered_combined_clean['D'], filtered_combined_clean['dryintercept'], 
#                  c=filtered_combined_clean['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

# # Add colorbar and labels
# cbar = plt.colorbar(sc)  # Create the colorbar
# cbar.set_label('Corrected Windspeed (m/s)', fontsize=14, fontweight='bold')  # Set label with fontsize and fontweight
# cbar.ax.tick_params(labelsize=12)  # Adjust colorbar tick size

# # Add mass contours
# contour_plot = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_extended, 
#                            levels=mass_levels, colors='red', alpha=0.75)

# # Add labels to the mass contours
# plt.clabel(contour_plot, inline=True, fontsize=10, fmt='%1.1e', use_clabeltext=True, colors='red')

# # Add axis labels and titles
# plt.xlabel('Slope', fontsize=19, fontweight='bold')
# plt.ylabel('Dry Intercept', fontsize=19, fontweight='bold')
# plt.yscale('log')
# plt.xscale('log')
# plt.title('Below Cloud Base January - June 2022', fontsize=19, fontweight='bold')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)

# # Adjust tick mark sizes
# plt.xticks(fontsize=16, fontweight='bold')
# plt.yticks(fontsize=16, fontweight='bold')

# plt.tight_layout()
# plt.show()
#%%
#fixing mass contours from 2 to inf 

# Function to calculate mass
def calculate_mass(N0, D):
    # Integrate mass over d^3 with an exponential decay
    integrand = lambda d: np.exp(-d / D) * d**3
    mass_integral, _ = quad(integrand, 2, np.inf)  # Integrate from 2 (not 0) to infinity
    return N0 * mass_integral

# Define the full x and y axis limits
x_min, x_max = 10**-0.1, 10**1.05  # Full x-axis range for slope
y_min, y_max = 10**-1.2, 10**1.6  # Full y-axis range for dry intercept

# Create extended grids to cover the full plot range
xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200)  # Denser grid for slope
ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)  # Denser grid for dry intercept
D_grid_extended, dryintercept_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)

# Recalculate mass grid over the extended grid
mass_grid_extended = np.zeros_like(D_grid_extended)
for i in range(D_grid_extended.shape[0]):
    for j in range(D_grid_extended.shape[1]):
        mass_grid_extended[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j])

# Define mass contour levels with finer spacing
mass_levels = np.logspace(-2, 5, 50)  # Adjust the range and number of levels as needed

# Extract slope (D) and dry intercept from combined data
slope_data = combined_data['D']  # Slope values
dry_intercept_data = [entry['dry intercept'] for entry in filtered_master_BCB_dryintercept]  # Dry intercept values

# Ensure lengths match for plotting
if len(slope_data) != len(dry_intercept_data):
    print(f"Mismatch in lengths: Slope={len(slope_data)}, Dry Intercept={len(dry_intercept_data)}")
slope_data = slope_data[:len(dry_intercept_data)]  # Adjust lengths if needed

# Plot the scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(slope_data, dry_intercept_data, c='blue', s=80, alpha=0.7)

# Add mass contours
contour_plot = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_extended, 
                           levels=mass_levels, colors='red', alpha=0.75)
plt.clabel(contour_plot, inline=True, fontsize=10, fmt='%1.1e', use_clabeltext=True, colors='red')

# Add labels and color bar
plt.xlabel('Slope', fontsize=19, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=19, fontweight='bold')
plt.title('Below Cloud Base January - June 2022', fontsize=19, fontweight='bold')

# Set logarithmic scales
plt.xscale('log')
plt.yscale('log')

# Set axis limits
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Adjust tick sizes
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')

# Add legend and finalize plot
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()

#%%
#removing some contours 
# def calculate_mass(N0, D):
#     integrand = lambda d: np.exp(-d / D) * d**3
#     mass_integral, _ = quad(integrand, 2, np.inf)  # Integrate from 0 to infinity
#     return N0 * mass_integral

# # Define the full x and y axis limits
# x_min, x_max = 10**-0.1, 10**1.05  # Full x-axis range
# y_min, y_max = 10**-1.7, 10**1.75  # Full y-axis range

# # Create extended grids to cover the full plot range
# xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200)  # Denser grid
# ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)
# D_grid_extended, dryintercept_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)

# # Recalculate mass grid over the extended grid
# mass_grid_extended = np.zeros_like(D_grid_extended)
# for i in range(D_grid_extended.shape[0]):
#     for j in range(D_grid_extended.shape[1]):
#         mass_grid_extended[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j])

# # Define mass contour levels with finer spacing
# mass_levels = np.logspace(-2, 5, 12)  # Adjust the range and number of levels as needed

# # Plot the scatter plot with Windspeed color map
# plt.figure(figsize=(10, 8))
# sc = plt.scatter(filtered_combined_clean['D'], filtered_combined_clean['dryintercept'], 
#                  c=filtered_combined_clean['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

# # Add colorbar and labels
# cbar = plt.colorbar(sc)  # Create the colorbar
# cbar.set_label('Corrected Windspeed (m/s)', fontsize=14, fontweight='bold')  # Set label with fontsize and fontweight
# cbar.ax.tick_params(labelsize=12)  # Adjust colorbar tick size

# # Add mass contours
# contour_plot = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_extended, 
#                            levels=mass_levels, colors='red', alpha=0.75)

# # Add labels to the mass contours
# plt.clabel(contour_plot, inline=True, fontsize=10, fmt='%1.1e', use_clabeltext=True, colors='red')

# # Add axis labels and titles
# plt.xlabel('Slope', fontsize=19, fontweight='bold')
# plt.ylabel('Dry Intercept', fontsize=19, fontweight='bold')
# plt.yscale('log')
# plt.xscale('log')
# plt.title('Below Cloud Base January - June 2022', fontsize=19, fontweight='bold')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)

# # Adjust tick mark sizes
# plt.xticks(fontsize=16, fontweight='bold')
# plt.yticks(fontsize=16, fontweight='bold')

# plt.tight_layout()
# plt.show()
#%%
#removing some contours 

# Function to calculate mass
def calculate_mass(N0, D):
    # Integrate mass over d^3 with an exponential decay
    integrand = lambda d: np.exp(-d / D) * d**3
    mass_integral, _ = quad(integrand, 2, np.inf)  # Integrate from 2 (not 0) to infinity
    return N0 * mass_integral

# Define the full x and y axis limits
x_min, x_max = 10**-0.1, 10**1.05  # Full x-axis range for slope
y_min, y_max = 10**-1.2, 10**1.6  # Full y-axis range for dry intercept

# Create extended grids to cover the full plot range
xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200)  # Denser grid for slope
ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)  # Denser grid for dry intercept
D_grid_extended, dryintercept_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)

# Recalculate mass grid over the extended grid
mass_grid_extended = np.zeros_like(D_grid_extended)
for i in range(D_grid_extended.shape[0]):
    for j in range(D_grid_extended.shape[1]):
        mass_grid_extended[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j])

# Define the number of mass contour levels
number_of_contours = 20  # Change this value to adjust the number of contours
mass_levels = np.logspace(-2, 5, number_of_contours)

# Extract slope (D) and dry intercept from combined data
slope_data = combined_data['D']  # Slope values
dry_intercept_data = [entry['dry intercept'] for entry in filtered_master_BCB_dryintercept]  # Dry intercept values

# Ensure lengths match for plotting
if len(slope_data) != len(dry_intercept_data):
    print(f"Mismatch in lengths: Slope={len(slope_data)}, Dry Intercept={len(dry_intercept_data)}")
slope_data = slope_data[:len(dry_intercept_data)]  # Adjust lengths if needed

# Plot the scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(slope_data, dry_intercept_data, c='blue', s=80, alpha=0.7)

# Add mass contours
contour_plot = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_extended, 
                           levels=mass_levels, colors='red', alpha=0.75)
plt.clabel(contour_plot, inline=True, fontsize=10, fmt='%1.1e', use_clabeltext=True, colors='red')

# Add labels and title
plt.xlabel('Slope', fontsize=19, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=19, fontweight='bold')
plt.title('Below Cloud Base January - June 2022', fontsize=19, fontweight='bold')

# Set logarithmic scales
plt.xscale('log')
plt.yscale('log')

# Set axis limits
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Adjust tick sizes
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')

# Finalize plot
plt.tight_layout()
plt.show()
#%%
#saving the masses
# Constants for windspeed correction
# Z0 = 0.02
# Z10 = 10

# # Windspeed correction function
# def correct_windspeed(windspeed, altitude):
#     return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))

# # Ensure combined_data is correct
# print(f"Length of combined_data: {len(combined_data['Date'])} entries")

# # Convert combined_data to DataFrame
# df_combined = pd.DataFrame(combined_data)

# # Filter valid entries
# filtered_combined_clean = df_combined.dropna(subset=['Windspeed', 'D', 'dryintercept'])
# filtered_combined_clean = filtered_combined_clean[np.isfinite(filtered_combined_clean[['D', 'dryintercept']].values).all(axis=1)]

# # Verify the filtered data
# print(f"Number of entries in filtered_combined_clean: {len(filtered_combined_clean)}")

# # Initialize mass_data for valid entries
# mass_data = []

# # Calculate mass for each valid entry in filtered_combined_clean
# for index, row in filtered_combined_clean.iterrows():
#     N0 = row['dryintercept']  # Dry intercept
#     D = row['D']             # Slope
#     mass = calculate_mass(N0, D)  # Calculate mass
#     mass_data.append(mass)

# # Verify the size of mass_data
# print(f"Number of masses in mass_data: {len(mass_data)}")
# print(f"First 10 masses: {mass_data[:10]}")

# # Create extended grids for contour plotting
# x_min, x_max = 10**-0.1, 10**0.8  # Full x-axis range
# y_min, y_max = 10**-1, 10**1.6  # Full y-axis range

# xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200)
# ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)
# D_grid_extended, dryintercept_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)

# # Recalculate mass grid over the extended grid
# mass_grid_extended = np.zeros_like(D_grid_extended)
# for i in range(D_grid_extended.shape[0]):
#     for j in range(D_grid_extended.shape[1]):
#         mass_grid_extended[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j])

# # Define mass contour levels
# mass_levels = np.logspace(-2, 5, 50)

# # Plot the scatter plot with Windspeed color map
# plt.figure(figsize=(10, 8))
# sc = plt.scatter(filtered_combined_clean['D'], filtered_combined_clean['dryintercept'], 
#                  c=filtered_combined_clean['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

# # Add colorbar and labels
# cbar = plt.colorbar(sc)
# cbar.set_label('Corrected Windspeed (m/s)', fontsize=14, fontweight='bold')
# cbar.ax.tick_params(labelsize=12)

# # Add mass contours
# contour_plot = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_extended, 
#                            levels=mass_levels, colors='red', alpha=0.75)
# plt.clabel(contour_plot, inline=True, fontsize=10, fmt='%1.1e', use_clabeltext=True, colors='red')

# # Add axis labels and titles
# plt.xlabel('Slope', fontsize=19, fontweight='bold')
# plt.ylabel('Dry Intercept', fontsize=19, fontweight='bold')
# plt.yscale('log')
# plt.xscale('log')
# plt.title('Below Cloud Base January - June 2022', fontsize=19, fontweight='bold')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)

# # Adjust tick mark sizes
# plt.xticks(fontsize=16, fontweight='bold')
# plt.yticks(fontsize=16, fontweight='bold')

# plt.tight_layout()
# plt.show()
#%%
#saving the masses
# Ensure combined_data is correct
print(f"Length of combined_data: {len(combined_data['Date'])} entries")

# Convert combined_data to DataFrame
df_combined = pd.DataFrame({
    'Date': combined_data['Date'],
    'D': combined_data['D'],  # Slope
    'dryintercept': [entry['dry intercept'] for entry in filtered_master_BCB_dryintercept]  # Dry intercept
})

# Filter valid entries
filtered_combined_clean = df_combined.dropna(subset=['D', 'dryintercept'])
filtered_combined_clean = filtered_combined_clean[np.isfinite(filtered_combined_clean[['D', 'dryintercept']].values).all(axis=1)]

# Verify the filtered data
print(f"Number of entries in filtered_combined_clean: {len(filtered_combined_clean)}")

# Initialize mass_data for valid entries
mass_data = []

# Calculate mass for each valid entry in filtered_combined_clean
for index, row in filtered_combined_clean.iterrows():
    N0 = row['dryintercept']  # Dry intercept
    D = row['D']             # Slope
    mass = calculate_mass(N0, D)  # Calculate mass
    mass_data.append(mass)

# Verify the size of mass_data
print(f"Number of masses in mass_data: {len(mass_data)}")
print(f"First 10 masses: {mass_data[:10]}")

# Create extended grids for contour plotting
x_min, x_max = 10**-0.1, 10**0.8  # Full x-axis range
y_min, y_max = 10**-1, 10**1.6  # Full y-axis range

xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200)
ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)
D_grid_extended, dryintercept_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)

# Recalculate mass grid over the extended grid
mass_grid_extended = np.zeros_like(D_grid_extended)
for i in range(D_grid_extended.shape[0]):
    for j in range(D_grid_extended.shape[1]):
        mass_grid_extended[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j])

# Define mass contour levels
mass_levels = np.logspace(-2, 5, 50)

# Plot the scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(filtered_combined_clean['D'], filtered_combined_clean['dryintercept'], 
            color='blue', s=100, alpha=0.7, label='Data Points')

# Add mass contours
contour_plot = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_extended, 
                           levels=mass_levels, colors='red', alpha=0.75)
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
#Showing a PDF of leg-average mass

# plt.figure(figsize=(8, 6))
# sns.kdeplot(mass_data, fill=True, bw_adjust=0.5, color='blue', label='PDF of Masses')
# plt.title("CAS PDF Below Cloud Base January-June 2022", fontsize=16, fontweight='bold')
# plt.xlabel("Mass", fontsize=14, fontweight='bold')
# plt.ylabel("Probability Density", fontsize=14, fontweight='bold')
# plt.grid(True)
# plt.legend(fontsize=12)
# plt.tight_layout()
# plt.show()
plt.figure(figsize=(8, 6))

# Generate KDE with bounds
kde_cas = gaussian_kde(mass_data, bw_method=0.5)  # Adjust bandwidth for smoother KDE
x_vals_cas = np.linspace(0, max(mass_data), 1000)  # Start from 0 to avoid negative values
y_vals_cas = kde_cas(x_vals_cas)

# Plot the KDE
plt.fill_between(x_vals_cas, y_vals_cas, alpha=0.5, color='blue', label='PDF of CAS Masses')
plt.plot(x_vals_cas, y_vals_cas, color='blue')

# Add labels, title, and legend
plt.title("CAS PDF Below Cloud Base January-June 2022", fontsize=16, fontweight='bold')
plt.xlabel("Mass (ug/m)", fontsize=14, fontweight='bold')
plt.ylabel("Probability Density", fontsize=14, fontweight='bold')
plt.xscale('log')
plt.tight_layout()
plt.show()
#%%
#histogram 

plt.figure(figsize=(10, 6))
sns.histplot(mass_data, bins=50, kde=True, color='blue', edgecolor='black')

# Add axis labels and title
plt.xlabel('Mass', fontsize=16, fontweight='bold')
plt.ylabel('Density', fontsize=16, fontweight='bold')
plt.title('Histogram and KDE of Masses', fontsize=18, fontweight='bold')

# Adjust tick sizes and x-axis limit
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.xlim(0, 800)  # Limiting x-axis

plt.tight_layout()
plt.show()
#%%
# Define custom bin edges
bin_1_edges = np.linspace(0.74, 4758.67, 25)  # Split Bin 1 into 6 smaller bins
remaining_bins = np.linspace(4758.67, 47580.05, 9)  # Keep the rest the same
custom_bins = np.concatenate([bin_1_edges, remaining_bins[1:]])  # Merge bins

# Compute histogram with custom bins
counts, bin_edges = np.histogram(mass_data, bins=custom_bins)

# Print bin counts
for i in range(len(counts)):
    print(f"Bin {i + 1}: Range ({bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}) → {counts[i]} points")

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(mass_data, bins=custom_bins, color='blue', alpha=0.7, edgecolor='black')

# Add axis labels and title
plt.xlabel('Mass', fontsize=16, fontweight='bold')
plt.ylabel('Frequency', fontsize=16, fontweight='bold')
plt.title('Histogram of Calculated Masses (Refined Binning)', fontsize=18, fontweight='bold')

# Adjust tick sizes
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.xlim(0,3000)
plt.tight_layout()
plt.xscale('log')
plt.show()

#%%
filtered_master_BCB_ddry = []

# Iterate over each date in master_BCB_exponential and corresponding legs in master_BCB_gRH
for i, (date, legs_exponential) in enumerate(master_BCB_exponential.items()):
    # Ensure that we have a corresponding entry in master_BCB_gRH
    if i >= len(filtered_master_BCB_gRH):
        # print(f"No corresponding gRh data for date {date}, skipping...")
        continue
    
    filtered_legs_grh = filtered_master_BCB_gRH[i]
    
    for j, (leg_exponential, leg_grh) in enumerate(zip(legs_exponential, filtered_legs_grh)):
        n0 = leg_exponential['n0']
        D = leg_exponential['D']
        gRh_mean = leg_grh['gRh_mean'][0]  # Assuming gRh_mean is a list with one value
        BCB_start = leg_grh['BCB_start']  # Extract BCB_start for this leg
        BCB_stop = leg_grh['BCB_stop']      # Extract BCB_stop for this leg
        
        # Calculate ddry by dividing bin centers by gRh_mean
        if gRh_mean is not np.nan and gRh_mean != 0:
            filtered_ddry_values = [center / gRh_mean for center in bin_center]
        else:
            filtered_ddry_values = [np.nan] * len(bin_center)
            print(f"Skipping division for leg {j} on date {date} due to invalid gRh_mean.")
        
        # Store the results in filtered_master_min_ddry
        filtered_master_BCB_ddry.append({
            'Date': date,
            'BCB_start': BCB_start,
            'BCB_stop': BCB_stop,
            'ddry': filtered_ddry_values,
            'n0': n0,
            'D': D,
            'gRh_mean': gRh_mean
        })

print(f"Length of filtered_master_BCB_ddry: {len(filtered_master_BCB_ddry)}")
#%%
filtered_master_BCB_ddry = []

# Iterate over each date in master_BCB_exponential and corresponding legs in master_BCB_gRH
for i, (date, legs_exponential) in enumerate(master_BCB_exponential.items()):
    # Ensure that we have a corresponding entry in master_BCB_gRH
    if i >= len(filtered_master_BCB_gRH):
        print(f"No corresponding gRh data for date {date}, skipping...")
        continue
    
    filtered_legs_grh = filtered_master_BCB_gRH[i]
    
    for j, (leg_exponential, leg_grh) in enumerate(zip(legs_exponential, filtered_legs_grh)):
        n0 = leg_exponential['n0']
        D = leg_exponential['D']
        
        gRh_mean = leg_grh['gRh_mean'][0]  # Assuming gRh_mean is a list with one value
        
        # Extract BCB_start and BCB_end for this leg
        BCB_start = leg_grh['BCB_start']
        BCB_stop = leg_grh['BCB_stop']
        
        # Print the gRh_mean value for verification
        print(f"Date: {date}, gRh_mean: {gRh_mean}")
        
        # Calculate ddry by dividing bin centers by gRh_mean
        if gRh_mean is not np.nan and gRh_mean != 0:
            filtered_ddry_values = [center / gRh_mean for center in bin_center]
            
            # Print some sample calculations to verify correctness
            print(f"Sample bin centers: {bin_center[:5]}")
            print(f"Sample filtered_ddry_values: {filtered_ddry_values[:5]}")
            
        else:
            filtered_ddry_values = [np.nan] * len(bin_center)
            print(f"Skipping division for leg {j} on date {date} due to invalid gRh_mean.")
        
        # Store the results in filtered_master_BCB_ddry
        filtered_master_BCB_ddry.append({
            'Date': date,
            'BCB_start': BCB_start,
            'BCB_stop': BCB_stop,
            'filtered ddry': filtered_ddry_values,
            'n0': n0,
            'D': D,
            'gRh_mean': gRh_mean
        })

print(f"Length of filtered_master_BCB_ddry: {len(filtered_master_BCB_ddry)}")
#%%
filtered_master_BCB_ddry = []

# Example initialization of bin_center (ensure this matches the original size distribution)
bin_center = [2.25, 2.75, 3.25, 3.75, 4.5, 5.75, 6.85, 7.55, 9.05, 11.4, 13.8, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5]

# Print bin_center length and content for verification
print(f"Full length of bin_center: {len(bin_center)}")
print(f"Full bin_center values: {bin_center}")

# Iterate over each date in master_BCB_exponential and corresponding legs in master_BCB_gRH
for i, (date, legs_exponential) in enumerate(master_BCB_exponential.items()):
    # Ensure that we have a corresponding entry in master_BCB_gRH
    if i >= len(filtered_master_BCB_gRH):
        print(f"No corresponding gRh data for date {date}, skipping...")
        continue
    
    filtered_legs_grh = filtered_master_BCB_gRH[i]
    
    for j, (leg_exponential, filtered_leg_grh) in enumerate(zip(legs_exponential, filtered_legs_grh)):
        n0 = leg_exponential['n0']
        D = leg_exponential['D']
        
        gRh_mean = filtered_leg_grh['gRh_mean'][0] 
        
        # Extract BCB_start and BCB_stop for this leg
        BCB_start = filtered_leg_grh['BCB_start']
        BCB_stop = filtered_leg_grh['BCB_stop']
        
        # Print the gRh_mean value for verification
        print(f"Date: {date}, Leg index: {j}, gRh_mean: {gRh_mean}")
        
        # Print the current leg's data
        print(f"Processing leg {j} for date {date}")
        print(f"Length of bin_center: {len(bin_center)}")
        print(f"Bin centers for date {date}, leg {j}: {bin_center}")
        
        # Calculate ddry by dividing bin centers by gRh_mean
        if not np.isnan(gRh_mean) and gRh_mean != 0:
            filtered_ddry_values = [center / gRh_mean for center in bin_center]
            
            # Print some sample calculations to verify correctness
            print(f"Sample bin centers: {bin_center[:5]}")
            print(f"Calculated filtered_ddry_values: {filtered_ddry_values[:5]}")
            print(f"Full filtered_ddry_values: {filtered_ddry_values}")
            
        else:
            filtered_ddry_values = [np.nan] * len(bin_center)
            print(f"Skipping division for leg {j} on date {date} due to invalid gRh_mean.")
        
        # Store the results in filtered_master_BCB_ddry
        filtered_master_BCB_ddry.append({
            'Date': date,
            'Leg_index': j,
            'BCB_start': BCB_start,
            'BCB_stop': BCB_stop,
            'filtered_ddry': filtered_ddry_values,
            'n0': n0,
            'D': D,
            'gRh_mean': gRh_mean
        })

print(f"Length of filtered_master_BCB_ddry: {len(filtered_master_BCB_ddry)}")
#%%
##trying thr 0th moment for dry constant concentration contours

# Extract dry intercept (N0) and D values from the dictionaries
N0_values = [entry['dry intercept'] for entry in filtered_master_BCB_dryintercept]
D_values = [entry['D'] for entry in filtered_master_BCB_ddry]

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

# Calculate midpoints along the contours for label placement
label_positions = []
for collection in contour.collections:
    for path in collection.get_paths():
        # Extract vertices of the contour line
        vertices = path.vertices
        if len(vertices) > 0:
            midpoint = len(vertices) // 2  # Get the middle point
            label_positions.append(vertices[midpoint])  # Append the midpoint coordinates

# Add labels to the contour lines at calculated positions
plt.clabel(contour, inline=True, fontsize=10, manual=label_positions, fmt='%1.1e')  

# Add a colorbar for reference
cbar = plt.colorbar(contour, label='Dry Concentration (particles/cm³·µm)')
cbar.ax.tick_params(labelsize=16)  # Adjust colorbar tick size

# Set bold font for colorbar ticks
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')

# Add axis labels and title
plt.xlabel('Slope', fontsize=19, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=19, fontweight='bold')
plt.title(' Dry Concentration Contours', fontsize=19, fontweight='bold')

# Set axis scales and limits
plt.xscale('log')
plt.yscale('log')
plt.xlim(10**0.2, 10**1.05)
plt.ylim(10**0.7, 10**1.3)

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
# Define the mass calculation function
# def calculate_mass(N0, D):
#     integrand = lambda d: np.exp(-d / D) * d**3
#     mass_integral, _ = quad(integrand, 2, np.inf)  # Integrate from 2 to infinity
#     return N0 * mass_integral

# # Define the full x and y axis limits
# x_min, x_max = 10**-0.1, 10**1.05  # Full x-axis range
# y_min, y_max = 10**-1.7, 10**1.75  # Full y-axis range

# # Create extended grids to cover the full plot range
# xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200)  # Denser grid
# ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)
# D_grid_extended, dryintercept_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)

# # Recalculate mass grid over the extended grid
# mass_grid_extended = np.zeros_like(D_grid_extended)
# for i in range(D_grid_extended.shape[0]):
#     for j in range(D_grid_extended.shape[1]):
#         mass_grid_extended[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j])

# # Define mass contour levels
# mass_levels = np.logspace(-2, 5, 12)  # Adjust the range and number of levels as needed

# # Recalculate dry concentration grid over the extended grid
# concentration_grid_extended = np.zeros_like(D_grid_extended)
# for i in range(D_grid_extended.shape[0]):
#     for j in range(D_grid_extended.shape[1]):
#         concentration_grid_extended[i, j] = dryintercept_grid_extended[i, j] * D_grid_extended[i, j]  # C_d = N0 * D

# # Define dry concentration contour levels
# concentration_levels = np.logspace(-2, 3, 20)  # Adjust the range and number of levels

# # Plot the scatter plot with Windspeed color map
# plt.figure(figsize=(17, 10))
# sc = plt.scatter(filtered_combined_clean['D'], filtered_combined_clean['dryintercept'], 
#                  c=filtered_combined_clean['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

# # Add colorbar and labels
# cbar = plt.colorbar(sc)  # Create the colorbar
# cbar.set_label('Corrected Windspeed (m/s)', fontsize=14, fontweight='bold')  # Set label with fontsize and fontweight
# cbar.ax.tick_params(labelsize=12)  # Adjust colorbar tick size

# # Add mass contours
# mass_contour = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_extended, 
#                            levels=mass_levels, colors='red', alpha=0.75)

# # Add labels to the mass contours in the same direction
# plt.clabel(mass_contour, inline=True, fontsize=11, fmt='%1.1e', use_clabeltext=True, colors='red')

# # Add dry concentration contours
# dry_concentration_contour = plt.contour(D_grid_extended, dryintercept_grid_extended, concentration_grid_extended, 
#                                         levels=concentration_levels, colors='blue', alpha=0.75)

# # Add labels to the dry concentration contours in the same direction
# plt.clabel(dry_concentration_contour, inline=True, fontsize=11, fmt='%1.1e', colors='blue')

# # Add legend for mass and dry concentration contours
# legend_elements = [
#     Line2D([0], [0], color='red', linewidth=3, label='Mass'),
#     Line2D([0], [0], color='blue', linewidth=3, label='Dry Concentration')
# ]
# plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.45, 1), fontsize=19)

# # Add axis labels and titles
# plt.xlabel('Slope', fontsize=19, fontweight='bold')
# plt.ylabel('Dry Intercept', fontsize=19, fontweight='bold')
# plt.yscale('log')
# plt.xscale('log')
# plt.title('Below Cloud Base January - June 2022', fontsize=19, fontweight='bold')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)

# # Adjust tick mark sizes
# plt.xticks(fontsize=16, fontweight='bold')
# plt.yticks(fontsize=16, fontweight='bold')

# plt.tight_layout()
# plt.show()
#%%
#Combining mass and dry concentration contours 
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Function to calculate mass
def calculate_mass(N0, D):
    integrand = lambda d: np.exp(-d / D) * d**3
    mass_integral, _ = quad(integrand, 2, np.inf)  # Integrate from 2 to infinity
    return N0 * mass_integral

# Define the full x and y axis limits
x_min, x_max = 10**-0.1, 10**1.05  # Full x-axis range
y_min, y_max = 10**-1, 10**1.7  # Full y-axis range

# Create extended grids to cover the full plot range
xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200)  # Denser grid for slope
ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)  # Denser grid for dry intercept
D_grid_extended, dryintercept_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)

# Recalculate mass grid over the extended grid
mass_grid_extended = np.zeros_like(D_grid_extended)
for i in range(D_grid_extended.shape[0]):
    for j in range(D_grid_extended.shape[1]):
        mass_grid_extended[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j])

# Define mass contour levels
mass_levels = np.logspace(-2, 5, 12)  # Adjust the range and number of levels as needed

# Recalculate dry concentration grid over the extended grid
concentration_grid_extended = np.zeros_like(D_grid_extended)
for i in range(D_grid_extended.shape[0]):
    for j in range(D_grid_extended.shape[1]):
        concentration_grid_extended[i, j] = dryintercept_grid_extended[i, j] * D_grid_extended[i, j]  # C_d = N0 * D

# Define dry concentration contour levels
concentration_levels = np.logspace(-2, 3, 20)  # Adjust the range and number of levels

# Extract slope (D) and dry intercept from combined data
slope_data = combined_data['D']  # Slope values
dry_intercept_data = [entry['dry intercept'] for entry in filtered_master_BCB_dryintercept]  # Dry intercept values

# Ensure matching lengths
if len(slope_data) != len(dry_intercept_data):
    print(f"Mismatch in lengths: Slope={len(slope_data)}, Dry Intercept={len(dry_intercept_data)}")
slope_data = slope_data[:len(dry_intercept_data)]

# Filter valid data
valid_indices = np.isfinite(slope_data) & np.isfinite(dry_intercept_data)
slope_data = np.array(slope_data)[valid_indices]
dry_intercept_data = np.array(dry_intercept_data)[valid_indices]

# Plot scatter plot
plt.figure(figsize=(17, 10))
plt.scatter(slope_data, dry_intercept_data, color='blue', s=100, alpha=0.7, label='Data Points')

# Add mass contours
mass_contour = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_extended, 
                           levels=mass_levels, colors='red', alpha=0.75)
plt.clabel(mass_contour, inline=True, fontsize=11, fmt='%1.1e', use_clabeltext=True, colors='red')

# Add dry concentration contours
dry_concentration_contour = plt.contour(D_grid_extended, dryintercept_grid_extended, concentration_grid_extended, 
                                        levels=concentration_levels, colors='blue', alpha=0.75)
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
#Filtered Variation of NtdNt with windspeed
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

            NtdNt = None
            for exp_params in filtered_master_BCB_NtdNt:
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
plt.colorbar(sc, label='Corrected Windspeed (m/s)')
plt.xlabel(' Windspeed (m/s)', fontsize=14, fontweight='bold')
plt.ylabel('Ratio of Total Droplet Concentration of dry droplets \n with diameter larger than 2um d\n to ambient concentration', fontsize=14, fontweight='bold')
plt.title('Below cloud base January - June 2022', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.show()
#%%


combined_data = {
    'Date': [],
    'D': [],
    'NtdNt': []
}

flat_ntdNt = [item for item in filtered_master_BCB_NtdNt]

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
plt.xlim(10**-1, 10**1)
plt.xscale('log')
plt.show()
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
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i] 
    
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
    leg_values = entry['interpolated_values']  # Renamed to avoid conflict
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
problematic_legs = [
    ('2022-01-26', 12),
    ('2022-01-11', 4),
    ('2022-01-11', 8), 
    ('2022-01-11', 14)
   
]
problematic_set = set(problematic_legs)

# Function to compute size distribution
def size_distribution(x, dryint, D):
    return dryint * np.exp(-x / D)

# Define common bins
common_bins = np.linspace(2, 10, 50)

# Manually define 4 windspeed bins
windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]

# Initialize grouped_distributions and mean_windspeeds for the bins
grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Debug variables
missing_windspeed_count = 0
interpolation_failures = 0

# Total legs
print(f"Total input legs: {len(filtered_master_BCB_ddry)}")

# Iterate through the legs
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]
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

    # Find windspeed
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) & 
        (df_combined['BCB_start'] == BCB_start) & 
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    if windspeed_entry.empty:
        missing_windspeed_count += 1
        continue

    windspeed = windspeed_entry['Windspeed'].values[0]

    # Interpolation
    try:
        interp_func = interp1d(ddry_values, size_distribution(ddry_values, dryint, D), kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)
    except Exception as e:
        interpolation_failures += 1
        continue

    # Bin by manually defined windspeed bins
    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:  # Ensure no overlaps
            grouped_distributions[idx].append(interpolated_leg_values)
            mean_windspeeds[idx].append(windspeed)
            break

# Debug: Total problematic legs excluded
print(f"Total problematic legs excluded: {len(problematic_set)}")

# Debug: Total legs with missing windspeed data
print(f"Total legs with missing windspeed data: {missing_windspeed_count}")

# Debug: Total interpolation failures
print(f"Total interpolation failures: {interpolation_failures}")

# Debug: Legs in each windspeed bin
for idx, group in grouped_distributions.items():
    print(f"Windspeed bin {idx} ({windspeed_bins[idx]} m/s): {len(group)} legs")

# Plot
plt.figure(figsize=(12, 8))
for idx, ranges in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])
        plt.plot(common_bins, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")

# Total legs plotted
total_legs = sum(len(group) for group in grouped_distributions.values())
print(f"Total number of legs plotted: {total_legs}")
plt.yscale('log')
plt.xscale('log')
plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Size distribution by wind speed', fontweight='bold')
plt.legend(title="Average wind speed (m/s)")
plt.tight_layout()
plt.show()
#%%
#trying to recreate fig 22 in lewis and schwartz
# Using 4 random windspeed bins 
problematic_legs = [
    ('2022-01-26', 12),
    ('2022-01-11', 4),
    ('2022-01-11', 8), 
    ('2022-01-11', 14)
   
]
problematic_set = set(problematic_legs)

# Function to compute size distribution
def size_distribution(x, dryint, D):
    return dryint * np.exp(-x / D)

# Define common bins
common_bins = np.linspace(2, 10, 50)

# Manually define 4 windspeed bins
windspeed_bins = [(0, 5), (5.001, 7), (7.001, 9), (9.001, np.inf)]

# Initialize grouped_distributions and mean_windspeeds for the bins
grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Debug variables
missing_windspeed_count = 0
interpolation_failures = 0

# Total legs
print(f"Total input legs: {len(filtered_master_BCB_ddry)}")

# Iterate through the legs
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]
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

    # Find windspeed
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) & 
        (df_combined['BCB_start'] == BCB_start) & 
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    if windspeed_entry.empty:
        missing_windspeed_count += 1
        continue

    windspeed = windspeed_entry['Windspeed'].values[0]

    # Interpolation
    try:
        interp_func = interp1d(ddry_values, size_distribution(ddry_values, dryint, D), kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)
    except Exception as e:
        interpolation_failures += 1
        continue

    # Bin by manually defined windspeed bins
    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:  # Ensure no overlaps
            grouped_distributions[idx].append(interpolated_leg_values)
            mean_windspeeds[idx].append(windspeed)
            break

# Debug: Total problematic legs excluded
print(f"Total problematic legs excluded: {len(problematic_set)}")

# Debug: Total legs with missing windspeed data
print(f"Total legs with missing windspeed data: {missing_windspeed_count}")

# Debug: Total interpolation failures
print(f"Total interpolation failures: {interpolation_failures}")

# Debug: Legs in each windspeed bin
for idx, group in grouped_distributions.items():
    print(f"Windspeed bin {idx} ({windspeed_bins[idx]} m/s): {len(group)} legs")

# Plot
plt.figure(figsize=(12, 8))
for idx, ranges in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])
        plt.plot(common_bins, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")

# Total legs plotted
total_legs = sum(len(group) for group in grouped_distributions.values())
print(f"Total number of legs plotted: {total_legs}")
plt.yscale('log')
plt.xscale('log')
plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Size distribution by wind speed', fontweight='bold')
plt.legend(title="Average wind speed (m/s)")
plt.tight_layout()
plt.show()
#%%
import matplotlib.pyplot as plt
import numpy as np

# Select the windspeed bin 5-7 m/s (index 1)
idx = 1  # Index for the 5-7 m/s bin

# Check if the bin contains data
if grouped_distributions[idx]:
    avg_distribution = np.mean(grouped_distributions[idx], axis=0)
    avg_windspeed = np.mean(mean_windspeeds[idx])
    num_legs = len(grouped_distributions[idx])

    # Plot only the 5-7 m/s bin
    plt.figure(figsize=(10, 6))
    plt.plot(common_bins, avg_distribution, color='orange', label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")

    # Set log scales
    plt.yscale('log')
    plt.xscale('log')

    # Labels and title
    plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
    plt.xlabel('Bin diameter (µm)', fontweight='bold')
    plt.title('Size distribution for wind speeds 5-7 m/s', fontweight='bold')
    plt.legend()

    plt.tight_layout()
    plt.show()
else:
    print("No data available for the 5-7 m/s windspeed bin.")

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


for i, flight in enumerate(master_BCB):
    for j, wind_alt in enumerate(flight):
        try:
            
            date = wind_alt['Date']
            wind_mean = wind_alt['Winds_mean'][0]
            alt_mean = wind_alt['Alts_mean'][0]
            BCB_start = wind_alt['BCB_start']
            BCB_stop = wind_alt['BCB_end']
            
            # Correct windspeed using the provided formula
            if not np.isnan(alt_mean) and alt_mean > 0:
                corrected_windspeed = correct_windspeed(wind_mean, alt_mean)
            else:
                corrected_windspeed = np.nan


            Ntd = None
            D = None

            # Find corresponding exponential parameters (D) from master_min_exponential
            if date in master_BCB_exponential:
                for exp_params in master_BCB_exponential[date]:
                    # Check if the current dictionary matches Min_start and Min_end
                    if exp_params['BCB_start'] == BCB_start and exp_params['BCB_stop'] == BCB_stop:
                        D = exp_params['D']
                        print(f"Found D for Date: {date}, BCB_start: {BCB_start}, BCB_stop: {BCB_stop}, D: {D}")
                        break

            # Find corresponding Ntd from filtered_master_min_ntd
            for ntd_data in filtered_master_BCB_ntd:
                if ntd_data['Date'] == date and ntd_data['BCB_start'] == BCB_start and ntd_data['BCB_stop'] == BCB_stop:
                    Ntd = ntd_data['Ntd']
                    break

            # Append data if both Ntd and D are found
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

# Convert the combined data to a DataFrame
df_combined = pd.DataFrame(combined_data)

# Check for anomalies or missing values
print(df_combined.describe())

# Separate data based on NaN Windspeed
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
plt.xscale('log')
plt.show()
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
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Match by index

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
#changing the number of bins being average over to 10

common_bins = np.linspace(0, 10, 10)  


windspeed_bins =[(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]
#[(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]  old bins from before 2DS import
grouped_concentrations = {i: [] for i in range(len(windspeed_bins))}  # To accumulate droplet concentrations
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]

    
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

        
        size_dist = size_distribution(ddry_values, dryint, D)  # Compute size distribution
        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)  # Interpolate to common bins

        
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

plt.yscale('log')

plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Fitted average size distribution by wind speed', fontweight='bold')
plt.legend(title="Average wind speed (m/s)")
plt.tight_layout()
plt.show()

#%%
# ##trying to fix size distribution and add ntd 

# def size_distribution(d, dryint, D):
#     return dryint * np.exp(-d / D)

# dmin = 2  
# dmax = np.inf 

# integrated_concentrations = []
# for i in range(len(filtered_master_BCB_ddry)):
#     entry_ddry = filtered_master_BCB_ddry[i]
#     entry_dryintercept = filtered_master_BCB_dryintercept[i]  
    
#     date = entry_ddry['Date']
#     BCB_start = entry_ddry['BCB_start']
#     BCB_stop = entry_ddry['BCB_stop']
#     leg_index = entry_ddry['Leg_index'] 
#     D = entry_ddry['D']  
#     dryint = entry_dryintercept['dry intercept']  
    

#     total_concentration, _ = quad(size_distribution, dmin, dmax, args=(dryint, D))
    
    
#     integrated_concentrations.append({
#         'Date': date,
#         'Leg_index': leg_index,
#         'BCB_start': BCB_start,
#         'BCB_stop': BCB_stop,
#         'Total Concentration': total_concentration
#     })


# for total in integrated_concentrations:
#     print(f"Date: {total['Date']}, Leg_index: {total['Leg_index']}, "
#           f"Total Concentration: {total['Total Concentration']:.3f} /cm^3/um")

# dates = [entry['Date'] for entry in integrated_concentrations]
# legs = [entry['Leg_index'] for entry in integrated_concentrations]
# total_concs = [entry['Total Concentration'] for entry in integrated_concentrations]

# plt.figure(figsize=(12, 8))
# plt.bar(range(len(total_concs)), total_concs, tick_label=[f"{d}-{l}" for d, l in zip(dates, legs)])
# plt.ylabel('Total Concentration (/cm^3/um)', fontweight='bold')
# plt.xlabel('Leg (Date-Leg)', fontweight='bold')
# plt.title('Total below cloud base concentration (/cm^3/um)', fontweight='bold')
# plt.tight_layout()
# plt.show()
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

for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i] 
    
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

for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_ntd = filtered_master_BCB_ntd[i]  

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


for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]

    
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

# Average each particle size bin for plotting before fitting
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
def size_distribution(ddry, N0, D):
    return N0 * np.exp(-ddry / D)

def fit_function(x, N0, D):
    return N0 * np.exp(-x / D)

common_bins = np.linspace(2, 10, 10)

# windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]
windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]  
grouped_concentrations = {i: [] for i in range(len(windspeed_bins))}  # To accumulate droplet concentrations
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}


for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]

    
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

common_bins = np.linspace(2.5, 10, 10)


# windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]
windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]
grouped_concentrations = {i: [] for i in range(len(windspeed_bins))}  # To accumulate droplet concentrations
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}


for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]


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
plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin center diameter (µm)', fontweight='bold')
plt.title('Below cloud base January - June 2022', fontweight='bold')
plt.legend(title="Average wind speed binning ")
plt.tight_layout()
plt.show()
#%%
# Function to compute size distribution
def fit_function(x, n0, D):
    return n0 * np.exp(-x / D)

common_bins = np.linspace(2.5, 10, 10)
windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]
grouped_concentrations = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    N0 = entry_ddry['n0']  
    date = entry_ddry['Date']
    leg_index = entry_ddry['Leg_index']
    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    # Match windspeed using date, start, and stop times
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) & 
        (df_combined['BCB_start'] == entry_ddry['BCB_start']) & 
        (df_combined['BCB_stop'] == entry_ddry['BCB_stop'])
    ]

    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]

        # Calculate size distribution
        size_dist = fit_function(ddry_values, N0, D)
        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        # Bin by windspeed
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_concentrations[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break

# Plot average size distributions
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

            # Calculate total concentration (area under the curve)
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

#%%
#fitting an exponential to averaged curves using a basic exponential 

def fit_function(x, dryint, D):
    return dryint * np.exp(-D*x)

common_bins = np.linspace(2.5, 10, 10)
# windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]
windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]  
grouped_concentrations = {i: [] for i in range(len(windspeed_bins))}  # To accumulate droplet concentrations
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]

    
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
common_bins = np.linspace(2.5, 10, 25)
# windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]
windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]
grouped_concentrations = {i: [] for i in range(len(windspeed_bins))} 
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i] 

    
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
                grouped_concentrations[idx].append(interpolated_leg_values)  # Add size distribution to this bin
                mean_windspeeds[idx].append(windspeed)
                break

plt.figure(figsize=(12, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_concentrations[idx]:
        
        grouped_conc_array = np.array(grouped_concentrations[idx])

        
        avg_concentration = np.mean(grouped_conc_array, axis=0)
        

        perc_25 = np.percentile(grouped_conc_array, 25, axis=0)
        perc_75 = np.percentile(grouped_conc_array, 75, axis=0)
        
        
        def fit_function(x, dryint, D):
            return dryint * np.exp(-x / D)
        
        popt, _ = curve_fit(fit_function, common_bins, avg_concentration, p0=[1, 1])  # Initial guess for dryint, D
        
        
        fitted_curve = fit_function(common_bins, *popt)
        
        
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_concentrations[idx])
        
       
        plt.plot(common_bins, fitted_curve, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")
        
        
        plt.fill_between(common_bins, perc_25, perc_75, alpha=0.2, label=f"25th-75th percentile (avg {avg_windspeed:.1f} m/s)")


plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.yscale('log')
plt.title('Fitted size distribution by wind speed', fontweight='bold')
plt.legend(title="Wind speed (m/s)")
plt.tight_layout()
plt.show()
#%%
#mass contours with median diameter
# Extract dry intercept (N0) and D values from the dictionaries
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
#%%
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

diameters = np.linspace(2.5, 50, 100) 

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


diameters = np.linspace(2.5, 50, 100) 

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
plt.xlabel('Dry diameter (µm)', fontsize=17, fontweight='bold')
plt.ylabel('Cumulative Mass', fontsize=17, fontweight='bold')
plt.title('CAS Average Cumulative Mass Distribution', fontsize=14, fontweight='bold')
plt.legend(fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15, width=3)
plt.tight_layout()
plt.show()
print(f"Diameter at 50% Average Cumulative Mass: {diameter_at_50:.2f} µm")
#%%
# Reset the index of the dataframe to avoid indexing issues
filtered_combined_clean = filtered_combined_clean.reset_index(drop=True)

# Diameters for the size distribution bins (same as cumulative mass plot)
diameters = np.linspace(2, 10, 10)  # Diameters in µm

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
plt.ylabel('Average droplet concentration (/cm3/um)', fontsize=16, fontweight='bold')
plt.title('CAS Average Size Distribution January-June 2022', fontsize=17, fontweight='bold')
plt.yscale('log')  # Logarithmic y-axis for concentration
plt.tick_params(axis='both', which='major', labelsize=15, width=3)
plt.tight_layout()
plt.legend(fontsize=18)
plt.show()

# %%
#%%
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
#%%
from matplotlib import cm

# Generate all_bin_means
all_bin_means_CDP = []

for entry in Y_CDP_calc:
    
    bin_means_CDP = []
    for i in range(0, 30): 
        key = f'Bin{i:02d}_Y_mean'  
        if key in entry: 
            bin_means_CDP.append(entry[key]) 
        else:
            bin_means_CDP.append(np.nan) 

    
    all_bin_means_CDP.append({
        'Date': entry['Date'],
        'BCB_start': entry.get('BCB_start', np.nan),
        'BCB_stop': entry.get('BCB_stop', np.nan),
        'Bin_means': bin_means_CDP
    })

# Create a list of unique dates
unique_dates = sorted(set(entry['Date'] for entry in all_bin_means_CDP))
colors = cm.viridis(np.linspace(0, 1, len(unique_dates)))
date_color_map = {date: color for date, color in zip(unique_dates, colors)}

# Plot the data
plt.figure(figsize=(10, 6))
added_dates = set() 

for entry in all_bin_means_CDP:
    bin_means_CDP = np.array(entry['Bin_means'])
    valid_indices = ~np.isnan(bin_means_CDP) 
    date = entry['Date']
    color = date_color_map[date]
    
    if date not in added_dates:
        plt.scatter(np.array(bin_center_CDP)[valid_indices], bin_means_CDP[valid_indices], color=color, marker='o', label=date)
        added_dates.add(date)
    else:
        plt.scatter(np.array(bin_center_CDP)[valid_indices], bin_means_CDP[valid_indices], color=color, marker='o')

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

all_bin_means_CDP = []

for entry in Y_CDP_calc:
    bin_means_CDP = []
    for i in range(0, 30): 
        key = f'Bin{i:02d}_Y_mean'
        if key in entry:  
            bin_means_CDP.append(entry[key]) 
        else:
            bin_means_CDP.append(np.nan) 

    all_bin_means_CDP.append({
        'Date': entry['Date'],
        'BCB_start': entry.get('BCB_start', np.nan),
        'BCB_stop': entry.get('BCB_stop', np.nan),
        'Bin_means': bin_means_CDP
    })

unique_dates = sorted(set(entry['Date'] for entry in all_bin_means_CDP))
colors = cm.viridis(np.linspace(0, 1, len(unique_dates)))
date_color_map = {date: color for date, color in zip(unique_dates, colors)}

plt.figure(figsize=(10, 6))
added_dates = set()

for entry in all_bin_means_CDP:
    bin_means_CDP = np.array(entry['Bin_means'])
    valid_indices = ~np.isnan(bin_means_CDP)
    date = entry['Date']
    color = date_color_map[date]
    
    if date not in added_dates:
        plt.scatter(np.array(bin_center_CDP)[valid_indices], bin_means_CDP[valid_indices], color=color, marker='o', label=date)
        added_dates.add(date)
    else:
        plt.scatter(np.array(bin_center_CDP)[valid_indices], bin_means_CDP[valid_indices], color=color, marker='o')

flat_bin_centers_CDP = np.tile(bin_center_CDP, len(all_bin_means_CDP))
flat_bin_means_CDP = np.concatenate([np.array(entry['Bin_means']) for entry in all_bin_means_CDP])
flat_bin_centers_CDP = np.array(flat_bin_centers_CDP, dtype=float)
flat_bin_means_CDP = np.array(flat_bin_means_CDP, dtype=float)
valid_indices_CDP = ~np.isnan(flat_bin_means_CDP)
flat_bin_centers_CDP = flat_bin_centers_CDP[valid_indices_CDP]
flat_bin_means_CDP = flat_bin_means_CDP[valid_indices_CDP]
popt, pcov = curve_fit(exponential, flat_bin_centers_CDP, flat_bin_means_CDP, p0=(1, 1))

n0 = popt[0]
D = popt[1]

print(f"Intercept (n0): {n0:.2e}")
print(f"E-folding diameter (D): {D:.2f} um")

x_fit = np.linspace(min(bin_center_CDP), max(bin_center_CDP), 100)
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
all_bin_means_CDP = []
dates_CDP = []
for entry in Y_CDP_calc:
   
    bin_means_CDP = []
    for i in range(0, 30): 
        key = f'Bin{i:02d}_Y_mean'  
        if key in entry:  
            bin_means_CDP.append(entry[key]) 
        else:
            bin_means_CDP.append(np.nan)
    all_bin_means_CDP.append({
        'Date': entry['Date'],
        'BCB_start': entry.get('BCB_start', np.nan),
        'BCB_stop': entry.get('BCB_stop', np.nan),
        'Bin_means': bin_means_CDP
    })
    dates_CDP.append(entry['Date']) 

unique_dates = sorted(set(dates_CDP))
colors = cm.viridis(np.linspace(0, 1, len(unique_dates)))
date_color_map = {date: color for date, color in zip(unique_dates, colors)}

fit_color = 'purple'

plt.figure(figsize=(10, 6))
added_dates = set() 
for entry in all_bin_means_CDP:
    bin_means_CDP = np.array(entry['Bin_means'])
    valid_indices_CDP = ~np.isnan(bin_means_CDP)
    date = entry['Date']
    color = date_color_map[date]

    if not valid_indices_CDP.any():
        print(f"No valid data for date {date}")
        continue

    bin_centers_CDP = np.array(bin_center_CDP)[valid_indices_CDP]
    bin_means_CDP = bin_means_CDP[valid_indices_CDP]
    
    try:
        popt, _ = curve_fit(exponential, bin_centers_CDP, bin_means_CDP, p0=(1, 1))
        n0, D = popt
        
        print(f"Date: {date}")
        print(f"  Intercept (n0): {n0:.2e}")
        print(f"  E-folding diameter (D): {D:.2f} um")
        
        x_fit = np.linspace(min(bin_centers_CDP), max(bin_centers_CDP), 100)
        y_fit = exponential(x_fit, *popt)
        plt.plot(x_fit, y_fit, color=fit_color, label=f'{date} fit: y = {n0:.2e} * exp(-x / {D:.2f})')

    except RuntimeError:
        print(f"Fit could not be performed for date {date}")
    
    if date not in added_dates:
        plt.scatter(bin_centers_CDP, bin_means_CDP, color=color, marker='o', label=date)
        added_dates.add(date)
    else:
        plt.scatter(bin_centers_CDP, bin_means_CDP, color=color, marker='o')

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

master_BCB_exponential_CDP = {}

unique_dates = sorted(set(entry['Date'] for entry in all_bin_means_CDP))
colors = cm.viridis(np.linspace(0, 1, len(unique_dates)))
date_color_map = {date: color for date, color in zip(unique_dates, colors)}

plt.figure(figsize=(10, 6))
added_dates = set()

for entry in all_bin_means_CDP:
    bin_means_CDP = np.array(entry['Bin_means'], dtype=float)
    valid_indices_CDP = ~np.isnan(bin_means_CDP)
    bin_centers_CDP = np.array(bin_center_CDP)[valid_indices_CDP]
    bin_means_CDP = bin_means_CDP[valid_indices_CDP]
    
    
    if not valid_indices_CDP.any():
        print(f"No valid data for date {entry['Date']}")
        continue
    
    
    try:
        popt, _ = curve_fit(exponential, bin_centers_CDP, bin_means_CDP, p0=(1, 1))
        n0, D = popt
        
    
        if entry['Date'] not in master_BCB_exponential_CDP:
            master_BCB_exponential_CDP[entry['Date']] = []
        
        master_BCB_exponential_CDP[entry['Date']].append({
            'Date': entry['Date'],
            'BCB_start': entry.get('BCB_start', np.nan),
            'BCB_stop': entry.get('BCB_stop', np.nan),
            'n0': n0,
            'D': D
        })
        
        
        print(f"Date: {entry['Date']}")
        print(f"  Intercept (n0): {n0:.2e}")
        print(f"  E-folding diameter (D): {D:.2f} um")
        
    
        x_fit = np.linspace(min(bin_centers_CDP), max(bin_centers_CDP), 100)
        y_fit = exponential(x_fit, *popt)
        plt.plot(x_fit, y_fit, color='purple', label=f'{entry["Date"]} fit: y = {n0:.2e} * exp(-x / {D:.2f})')

    except RuntimeError:
        print(f"Fit could not be performed for date {entry['Date']}")
    
  
    date = entry['Date']
    if date not in added_dates:
        plt.scatter(bin_centers_CDP, bin_means_CDP, color=date_color_map[date], marker='o', label=date)
        added_dates.add(date)
    else:
        plt.scatter(bin_centers_CDP, bin_means_CDP, color=date_color_map[date], marker='o')


plt.xlabel('Bin centers diameter (um)', fontsize=12, fontweight='bold')
plt.ylabel('Clear mean droplet concentration \n (/cm^3/um)', fontsize=12, fontweight='bold')
plt.title('Below cloud base\n January-June 2022', fontsize=12, fontweight='bold')
plt.xticks(np.arange(0, 50, 5))
plt.tight_layout()
plt.show()
#%%
# Print the number of inner dictionaries for each date in master_min_exponential
for date, entries in master_BCB_exponential_CDP.items():
    print(f"Date: {date} has {len(entries)} entries")
total_entries_CDP = sum(len(entries) for entries in master_BCB_exponential_CDP.values())
print(f"Total number of inner dictionaries: {total_entries_CDP}")

for date, entries in master_BCB_exponential_CDP.items():
    print(f"Date: {date}")
    for entry in entries:
        start_time = entry.get('BCB_start', 'N/A')  
        stop_time = entry.get('BCB_stop', 'N/A')   
        print(f"  Start Time: {start_time}, Stop Time: {stop_time}")
total_legs_CDP = sum(len(entries) for entries in master_BCB_exponential_CDP.values())
print(f"Total number of legs: {total_legs_CDP}")
#%%
master_BCB_RH_CDP = []

for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict_CDP = leg_data[i]

    flight_date = leg_dict_CDP['Date'] 
    BCB_start = leg_dict_CDP['LegIndex_02']['StartTimes']
    BCB_stop = leg_dict_CDP['LegIndex_02']['StopTimes']

    rh_flight = h20[i]
    times_rh = rh_flight.Time_Start.values
    rh_values = rh_flight.RHw_DLH.values

    all_BCB_CDP = []

    for j in range(len(BCB_start)):
        start = int(BCB_start[j])
        end = int(BCB_stop[j])

        rh_times_CDP = {
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

        rh_times_CDP['Rh_mean'].append(rh9_mean)
        all_BCB_CDP.append(rh_times_CDP) 

    master_BCB_RH_CDP.append(all_BCB_CDP)

for flight in master_BCB_RH_CDP:
    for leg in flight:
        rh_mean_list = leg['Rh_mean']
        leg['Rh_mean'] = [np.nan if value <=0 else value for value in rh_mean_list]
#%%
#for only the legs present after LWC filtration and master_BCB_exponential 

date_leg_set = set()
for entries in master_BCB_exponential_CDP.values():
    for entry in entries:
        date = entry['Date']
        BCB_start = entry.get('BCB_start', np.nan)
        BCB_stop = entry.get('BCB_stop', np.nan)
        date_leg_set.add((date, BCB_start, BCB_stop))

filtered_master_BCB_RH_CDP = []

for flight in master_BCB_RH_CDP:
    filtered_legs_CDP = []
    for leg in flight:
        date = leg['Date']
        BCB_start = leg['BCB_start']
        BCB_stop = leg['BCB_stop']
        if (date, BCB_start, BCB_stop) in date_leg_set:
            filtered_legs_CDP.append(leg)
    if filtered_legs_CDP:
        filtered_master_BCB_RH_CDP.append(filtered_legs_CDP)
#%%
# Flatten the list of lists and count the total number of entries
total_entries_filtered_master_BCB_RH_CDP = sum(len(legs) for legs in filtered_master_BCB_RH_CDP)
print(f"Total entries in filtered_master_BCB_RH: {total_entries_filtered_master_BCB_RH_CDP}")
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
#filtered dry intercept 

filtered_master_BCB_interceptdry_dict_CDP = {}
if isinstance(filtered_master_BCB_gRH_CDP[0], list):
    filtered_master_BCB_gRH_CDP = [item for sublist in filtered_master_BCB_gRH_CDP for item in sublist]

unique_keys = set()

flattened_exponential_CDP = []
for exp_list in master_BCB_exponential_CDP.values():
    flattened_exponential_CDP.extend(exp_list)

exponential_dict = {(exp['Date'], exp['BCB_start'], exp['BCB_stop']): exp for exp in flattened_exponential_CDP}
for entry in filtered_master_BCB_gRH_CDP:
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    gRh_mean = entry['gRh_mean'][0] 
    Rh_mean = entry['Rh_mean'][0]

    if Rh_mean < 0:
        continue

    key = (date, BCB_start, BCB_stop)

    if date in master_BCB_exponential_CDP:
        exp_params_list = master_BCB_exponential_CDP[date]
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
#filtered ntd 
filtered_master_BCB_ntd_dict_CDP = {}
if isinstance(filtered_master_BCB_gRH_CDP[0], list):
    filtered_master_BCB_gRH_CDP = [item for sublist in filtered_master_BCB_gRH_CDP for item in sublist]


unique_keys = set()
flattened_exponential_CDP = []
for exp_list in master_BCB_exponential_CDP.values():
    flattened_exponential_CDP.extend(exp_list)
exponential_dict = {(exp['Date'], exp['BCB_start'], exp['BCB_stop']): exp for exp in flattened_exponential_CDP}
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

    
    if date in master_BCB_exponential_CDP:
        exp_params_list = master_BCB_exponential_CDP[date]
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
#filtered master_BCB_NtdNt

filtered_master_BCB_NtdNt_dict_CDP = {}
if isinstance(filtered_master_BCB_gRH_CDP[0], list):
    filtered_master_BCB_gRH_CDP = [item for sublist in filtered_master_BCB_gRH_CDP for item in sublist]

ddrymin = 2
dmin = 2.5  

unique_keys = set()

flattened_exponential_CDP = []
for exp_list in master_BCB_exponential_CDP.values():
    flattened_exponential_CDP.extend(exp_list)

exponential_dict = {(exp['Date'], exp['BCB_start'], exp['BCB_stop']): exp for exp in flattened_exponential_CDP}

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
#%%
#Time to gather wind speed and altitude data for each leg
master_BCB_CDP = []
for i in range(len(dates_legs)):
    date = dates_legs[i]

    leg_dict_CDP = leg_data[i]

    flight_date = leg_dict_CDP['Date']
    BCB_start = leg_dict_CDP['LegIndex_02']['StartTimes']
    BCB_stop = leg_dict_CDP['LegIndex_02']['StopTimes']
    sum_flight = summary[i]
    times = sum_flight.Time_mid.values
    winds = sum_flight.Wind_Speed.values
    alts = sum_flight.GPS_altitude.values
    
    all_BCB_means_CDP = []


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

        all_BCB_means_CDP.append(wind_alt)
        
    master_BCB_CDP.append(all_BCB_means_CDP)
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
#%%
Z0 = 0.02 
Z10 = 10 

def correct_windspeed(windspeed, altitude):
    return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))

combined_data_CDP = {
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

            combined_data_CDP['Date'].append(date)
            combined_data_CDP['Altitude'].append(alt_mean)
            combined_data_CDP['RH'].append(rh_mean)
            combined_data_CDP['Windspeed'].append(corrected_windspeed)

        except IndexError as e:
            print(f"Index error at i={i}, j={j}: {e}")
            continue

df_combined_CDP = pd.DataFrame(combined_data_CDP)

print(df_combined_CDP.describe())

df_with_windspeed = df_combined_CDP[df_combined_CDP['Windspeed'].notna()]
df_nan_windspeed = df_combined_CDP[df_combined_CDP['Windspeed'].isna()]
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
#%%
# Z0 = 0.02 
# Z10 = 10 

# def correct_windspeed(windspeed, altitude):
#     return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))

# combined_data_CDP = {
#     'Date': [],
#     'D': [],
#     'dryintercept': [],
#     'Windspeed': []
# }

# for i, flight in enumerate(master_BCB_CDP):
#     for j, wind_alt in enumerate(flight):
#         try:
#             # Extract basic parameters
#             date = wind_alt['Date']
#             wind_mean = wind_alt['Winds_mean'][0]
#             alt_mean = wind_alt['Alts_mean'][0]

#             # Apply windspeed correction
#             corrected_windspeed = (
#                 correct_windspeed(wind_mean, alt_mean) 
#                 if not np.isnan(alt_mean) and alt_mean > 0 
#                 else np.nan
#             )

#             # Ensure alignment with filtered_master_BCB_gRH_CDP
#             if i < len(filtered_master_BCB_gRH_CDP) and j < len(filtered_master_BCB_gRH_CDP[i]):
#                 gRh_mean = filtered_master_BCB_gRH_CDP[i][j]['gRh_mean'][0]
#             else:
#                 print(f"Index mismatch for gRh_mean at i={i}, j={j}")
#                 continue

#             # Check if date exists in master_BCB_exponential_CDP
#             if date in master_BCB_exponential_CDP:
#                 # Append only one entry per leg
#                 exp_params = master_BCB_exponential_CDP[date][0]  # Assuming 1:1 mapping
#                 D = exp_params['D']
#                 n0 = exp_params['n0']
#                 dryintercept = n0 * gRh_mean

#                 # Append the processed data
#                 combined_data_CDP['Date'].append(date)
#                 combined_data_CDP['D'].append(D)
#                 combined_data_CDP['dryintercept'].append(dryintercept)
#                 combined_data_CDP['Windspeed'].append(corrected_windspeed)

#         except IndexError as e:
#             print(f"Index error at i={i}, j={j}: {e}")
#             continue

# # Convert to DataFrame
# df_combined_CDP = pd.DataFrame(combined_data_CDP)
# print(f"Number of rows in df_combined_CDP: {len(df_combined_CDP)}")

# # Visualizations
# df_with_windspeed = df_combined_CDP[df_combined_CDP['Windspeed'].notna()]
# df_nan_windspeed = df_combined_CDP[df_combined_CDP['Windspeed'].isna()]

# plt.figure(figsize=(10, 8))
# sc = plt.scatter(df_with_windspeed['D'], df_with_windspeed['dryintercept'], 
#                  c=df_with_windspeed['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

# plt.scatter(df_nan_windspeed['D'], df_nan_windspeed['dryintercept'], 
#             color='grey', s=100, label='Windspeed NaN')
# plt.colorbar(sc, label='10 m wind speed (m/s)')
# plt.xlabel('Slope', fontsize=14, fontweight='bold')
# plt.ylabel('Dry intercept', fontsize=14, fontweight='bold')
# plt.yscale('log')
# plt.xscale('log')
# plt.title('Slope vs. dry intercept', fontsize=14, fontweight='bold')
# plt.show()
#%%
# just slope vs dry int
# Ensure combined_data_CDP is correct
print(f"Length of combined_data_CDP: {len(combined_data_CDP['Date'])} entries")

# Convert combined_data_CDP to DataFrame
df_combined_CDP = pd.DataFrame({
    'Date': combined_data_CDP['Date'],
    'D': combined_data_CDP['D'],  # Slope
    'dryintercept': [entry['dry intercept'] for entry in filtered_master_BCB_dryintercept_CDP]  # Dry intercept
})

# Filter valid entries (removing NaN or infinite values)
filtered_combined_clean_CDP = df_combined_CDP.dropna(subset=['D', 'dryintercept'])
filtered_combined_clean_CDP = filtered_combined_clean_CDP[
    np.isfinite(filtered_combined_clean_CDP[['D', 'dryintercept']].values).all(axis=1)
]

# Verify the filtered data
print(f"Number of entries in filtered_combined_clean_CDP: {len(filtered_combined_clean_CDP)}")

# Plot scatter plot: Slope vs Dry Intercept
plt.figure(figsize=(10, 8))
plt.scatter(filtered_combined_clean_CDP['D'], filtered_combined_clean_CDP['dryintercept'], 
            color='green', s=100, alpha=0.7, label='Data Points')

# Add axis labels and title
plt.xlabel('Slope', fontsize=19, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=19, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.title('CDP Below Cloud Base January - June 2022', fontsize=19, fontweight='bold')

# Set axis limits
x_min, x_max = 10**-0.2, 10**0.9  # Adjust based on expected range
y_min, y_max = 10**-2.35, 10**1  # Adjust based on expected range
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Adjust tick sizes
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()
#%%
# Z0 = 0.02  # meters (typical value for open terrain)
# Z10 = 10  # target height m

# # Function to apply windspeed correction
# def correct_windspeed(windspeed, altitude):
#     return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))

# combined_data_CDP = {
#     'Date': [],
#     'D': [],
#     'dryintercept': [],
#     'Windspeed': []
# }

# # Validate and combine the data
# for i, flight in enumerate(master_BCB_CDP):
#     for j, wind_alt in enumerate(flight):
#         try:
#             date = wind_alt['Date']
#             wind_mean = wind_alt['Winds_mean'][0]
#             alt_mean = wind_alt['Alts_mean'][0]
            
#             # Correct windspeed using the provided formula
#             if not np.isnan(alt_mean) and alt_mean > 0:
#                 corrected_windspeed = correct_windspeed(wind_mean, alt_mean)
#             else:
#                 corrected_windspeed = np.nan
            
#             # Find the corresponding exponential parameters and dry intercept
#             if date in master_BCB_exponential_CDP:
#                 exp_params_list = master_BCB_exponential_CDP[date]
#                 gRh_mean = filtered_master_BCB_gRH_CDP[i][j]['gRh_mean'][0]  # Assume matching index

#                 for exp_params in exp_params_list:
#                     D = exp_params['D']
#                     n0 = exp_params['n0']
#                     dryintercept = n0 * gRh_mean
                    
#                     # Append data to the combined dictionary
#                     combined_data_CDP['Date'].append(date)
#                     combined_data_CDP['D'].append(D)
#                     combined_data_CDP['dryintercept'].append(dryintercept)
#                     combined_data_CDP['Windspeed'].append(corrected_windspeed)

#         except IndexError as e:
#             print(f"Index error at i={i}, j={j}: {e}")
#             continue

# # Convert to DataFrame
# df_combined_CDP = pd.DataFrame(combined_data_CDP)

# # Check the data for any anomalies
# print(df_combined_CDP.describe())

# # Separate data based on NaN Windspeed
# df_with_windspeed = df_combined_CDP[df_combined_CDP['Windspeed'].notna()]
# df_nan_windspeed = df_combined_CDP[df_combined_CDP['Windspeed'].isna()]

# # Check for NaN or Inf values
# if df_combined_CDP[['D', 'dryintercept']].isnull().any().any():
#     print("Data contains NaN values.")
#     # Filter out NaN values
#     df_combined_CDP = df_combined_CDP.dropna(subset=['D', 'dryintercept'])

# if np.any(np.isinf(df_combined_CDP[['D', 'dryintercept']].values)):
#     print("Data contains Inf values.")
#     # Optionally filter out inf values
#     df_combined_CDP = df_combined_CDP[np.isfinite(df_combined_CDP[['D', 'dryintercept']].values).all(axis=1)]

# # Filter out NaN and Inf values for plotting
# filtered_combined_CDP = df_combined_CDP[df_combined_CDP['Windspeed'].notna() & df_combined_CDP['D'].notna() & df_combined_CDP['dryintercept'].notna()]
# filtered_combined_CDP = filtered_combined_CDP[np.isfinite(filtered_combined_CDP[['D', 'dryintercept']].values).all(axis=1)]

# # Plot data points with valid Windspeed values
# sc = plt.scatter(filtered_combined_CDP['D'], filtered_combined_CDP['dryintercept'], 
#                  c=filtered_combined_CDP['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

# # Plot data points with NaN Windspeed values in black
# plt.scatter(df_nan_windspeed['D'], df_nan_windspeed['dryintercept'], 
#             color='grey', s=100, label='Windspeed NaN')

# # Calculate density for contours
# kde = gaussian_kde(np.vstack([filtered_combined_CDP['D'], filtered_combined_CDP['dryintercept']]))
# xgrid = np.linspace(min(filtered_combined_CDP['D']), max(filtered_combined_CDP['D']), 100)
# ygrid = np.linspace(min(filtered_combined_CDP['dryintercept']), max(filtered_combined_CDP['dryintercept']), 100)
# X, Y = np.meshgrid(xgrid, ygrid)
# Z = kde(np.vstack([X.ravel(), Y.ravel()]))
# Z = Z.reshape(X.shape)

# # Plot contours for total concentration
# contour_levels = np.linspace(0, Z.max(), num=10)  # Adjust number of levels as needed
# plt.contour(X, Y, Z, levels=contour_levels, colors='blue', alpha=0.5)

# # Add color bar
# plt.colorbar(sc, label='Corrected Windspeed (m/s)')

# # Add labels and title
# plt.xlabel('D', fontsize=14, fontweight='bold')
# plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
# plt.yscale('log')
# plt.title('Density Contours' , fontweight='bold')
# plt.xscale('log')
# plt.xlim(10**0, 10**0.8)
# plt.ylim(10**-2.7, 10**2)
# plt.show()
#%%

# # Constants for windspeed correction
# Z0 = 0.02  # meters (typical value for open terrain)
# Z10 = 10  # target height m

# # Function to apply windspeed correction
# def correct_windspeed(windspeed, altitude):
#     return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))

# # Initialize combined data dictionary
# combined_data_CDP = {
#     'Date': [],
#     'D': [],
#     'dryintercept': [],
#     'Windspeed': []
# }

# # Validate and combine the data
# for i, flight in enumerate(master_BCB_CDP):
#     for j, wind_alt in enumerate(flight):
#         try:
#             # Extract necessary values
#             date = wind_alt['Date']
#             wind_mean = wind_alt['Winds_mean'][0]
#             alt_mean = wind_alt['Alts_mean'][0]

#             # Correct windspeed using the provided formula
#             if not np.isnan(alt_mean) and alt_mean > 0:
#                 corrected_windspeed = correct_windspeed(wind_mean, alt_mean)
#             else:
#                 corrected_windspeed = np.nan

#             # Find the corresponding exponential parameters and dry intercept
#             if date in master_BCB_exponential_CDP:
#                 exp_params_list = master_BCB_exponential_CDP[date]
#                 gRh_mean = filtered_master_BCB_gRH_CDP[i][j]['gRh_mean'][0]  # Assume matching index

#                 # Limit to the first set of exponential parameters for this date
#                 exp_params = exp_params_list[0]  # Avoid duplicates
#                 D = exp_params['D']
#                 n0 = exp_params['n0']
#                 dryintercept = n0 * gRh_mean

#                 # Append data to the combined dictionary
#                 combined_data_CDP['Date'].append(date)
#                 combined_data_CDP['D'].append(D)
#                 combined_data_CDP['dryintercept'].append(dryintercept)
#                 combined_data_CDP['Windspeed'].append(corrected_windspeed)

#         except IndexError as e:
#             print(f"Index error at i={i}, j={j}: {e}")
#             continue

# # Convert to DataFrame
# df_combined_CDP = pd.DataFrame(combined_data_CDP)
# print(f"Number of rows in df_combined_CDP before filtering: {len(df_combined_CDP)}")

# # Handle NaN and Inf values
# df_combined_CDP = df_combined_CDP.dropna(subset=['D', 'dryintercept', 'Windspeed'])
# df_combined_CDP = df_combined_CDP[np.isfinite(df_combined_CDP[['D', 'dryintercept', 'Windspeed']].values).all(axis=1)]

# # Remove duplicate entries
# df_combined_CDP = df_combined_CDP.drop_duplicates(subset=['Date', 'D', 'dryintercept', 'Windspeed'])

# # Final filtered data
# filtered_combined_CDP = df_combined_CDP[df_combined_CDP['Windspeed'].notna()]
# print(f"Number of valid rows in filtered_combined_CDP: {len(filtered_combined_CDP)}")

# # Plot data points with valid Windspeed values
# plt.figure(figsize=(10, 8))
# sc = plt.scatter(filtered_combined_CDP['D'], filtered_combined_CDP['dryintercept'], 
#                  c=filtered_combined_CDP['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

# # Calculate density for contours
# kde = gaussian_kde(np.vstack([filtered_combined_CDP['D'], filtered_combined_CDP['dryintercept']]))
# xgrid = np.linspace(min(filtered_combined_CDP['D']), max(filtered_combined_CDP['D']), 100)
# ygrid = np.linspace(min(filtered_combined_CDP['dryintercept']), max(filtered_combined_CDP['dryintercept']), 100)
# X, Y = np.meshgrid(xgrid, ygrid)
# Z = kde(np.vstack([X.ravel(), Y.ravel()]))
# Z = Z.reshape(X.shape)

# # Plot density contours
# contour_levels = np.linspace(0, Z.max(), num=10)  # Adjust number of levels as needed
# plt.contour(X, Y, Z, levels=contour_levels, colors='blue', alpha=0.5)

# # Add color bar
# plt.colorbar(sc, label='Corrected Windspeed (m/s)')

# # Add labels and title
# plt.xlabel('Slope', fontsize=14, fontweight='bold')
# plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
# plt.yscale('log')
# plt.xscale('log')
# plt.title('Density Contours', fontsize=16, fontweight='bold')
# plt.xlim(10**0.1, 10**0.6)
# plt.ylim(10**-2.3, 10**1.2)
# plt.tight_layout()
# plt.show()
#%%
#Density contours 
# Convert combined_data_CDP to DataFrame
df_combined_CDP = pd.DataFrame({
    'Date': combined_data_CDP['Date'],
    'D': combined_data_CDP['D'],  # Slope
    'dryintercept': [entry['dry intercept'] for entry in filtered_master_BCB_dryintercept_CDP]  # Dry intercept
})

# Filter valid entries (removing NaN or infinite values)
filtered_combined_clean_CDP = df_combined_CDP.dropna(subset=['D', 'dryintercept'])
filtered_combined_clean_CDP = filtered_combined_clean_CDP[
    np.isfinite(filtered_combined_clean_CDP[['D', 'dryintercept']].values).all(axis=1)
]

# Verify the filtered data
print(f"Number of valid rows in filtered_combined_clean_CDP: {len(filtered_combined_clean_CDP)}")

# Calculate density for contours
kde = gaussian_kde(np.vstack([filtered_combined_clean_CDP['D'], filtered_combined_clean_CDP['dryintercept']]))
xgrid = np.logspace(np.log10(10**-0.2), np.log10(10**0.9), 100)  # Adjusted x-axis grid for slope
ygrid = np.logspace(np.log10(10**-2.35), np.log10(10**1), 100)  # Adjusted y-axis grid for dry intercept
X, Y = np.meshgrid(xgrid, ygrid)
Z = kde(np.vstack([X.ravel(), Y.ravel()]))
Z = Z.reshape(X.shape)

# Plot scatter plot with density contours
plt.figure(figsize=(10, 8))
plt.scatter(filtered_combined_clean_CDP['D'], filtered_combined_clean_CDP['dryintercept'], 
            color='green', s=100, alpha=0.7, label='Data Points')

# Plot density contours
contour_levels = np.linspace(0, Z.max(), num=10)  # Adjust the number of levels as needed
plt.contour(X, Y, Z, levels=contour_levels, colors='blue', alpha=0.5)

# Add axis labels and title
plt.xlabel('Slope', fontsize=19, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=19, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.title('CDP Below Cloud Base January - June 2022', fontsize=19, fontweight='bold')

# Set axis limits
plt.xlim(10**-0.2, 10**0.9)  # Adjusted range for slope
plt.ylim(10**-2.35, 10**1)  # Adjusted range for dry intercept

# Adjust tick sizes
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()

#%%
# # Full integrated mass calculation: M = N0 * D^4
# def calculate_mass(N0, D):
#     return N0 * D**4

# # Filter out rows where D or dryintercept is NaN or <= 0
# filtered_combined_clean_CDP = filtered_combined_CDP[(filtered_combined_CDP['D'] > 0) & 
#                                             (filtered_combined_CDP['dryintercept'] > 0)].copy()

# # Recalculate mass for each point using the full integrated mass equation
# filtered_combined_clean_CDP['Mass'] = calculate_mass(filtered_combined_clean_CDP['dryintercept'], 
#                                                  filtered_combined_clean_CDP['D'])

# # Debugging: Check minimum and maximum mass values
# print(f"Min mass: {filtered_combined_clean_CDP['Mass'].min()}, Max mass: {filtered_combined_clean_CDP['Mass'].max()}")

# # Create data-based grids to better match the distribution of the points
# xgrid = np.logspace(np.log10(filtered_combined_clean_CDP['D'].min()), 
#                     np.log10(filtered_combined_clean_CDP['D'].max()), 100)
# ygrid = np.logspace(np.log10(filtered_combined_clean_CDP['dryintercept'].min()), 
#                     np.log10(filtered_combined_clean_CDP['dryintercept'].max()), 100)
# D_grid, dryintercept_grid = np.meshgrid(xgrid, ygrid)

# # Calculate mass for each point on the grid
# mass_grid = calculate_mass(dryintercept_grid, D_grid)

# # Debugging: Check if mass_grid contains meaningful values
# print(f"Mass grid: Min = {np.nanmin(mass_grid)}, Max = {np.nanmax(mass_grid)}")

# # Define mass contour levels (adjust to fit data range)
# mass_levels = np.logspace(np.log10(filtered_combined_clean_CDP['Mass'].min()), 
#                           np.log10(filtered_combined_clean_CDP['Mass'].max()), 20)
# print(f"Mass levels: {mass_levels}")
# sc = plt.scatter(filtered_combined_clean_CDP['D'], filtered_combined_clean_CDP['dryintercept'], 
#                  c=filtered_combined_clean_CDP['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

# if not df_nan_windspeed.empty:
#     plt.scatter(df_nan_windspeed['D'], df_nan_windspeed['dryintercept'], 
#                 color='grey', s=100, label='Windspeed NaN')

# # Plot the mass contours using the full integrated mass formula
# contour_plot = plt.contour(D_grid, dryintercept_grid, mass_grid, levels=mass_levels, colors='blue', alpha=0.75)

# if len(contour_plot.allsegs[0]) == 0:
#     print("No contours were created. Check your data range or mass grid calculation.")
# plt.colorbar(sc, label='Corrected Windspeed (m/s)')
# plt.xlabel('D', fontsize=14, fontweight='bold')
# plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
# plt.yscale('log')
# plt.xscale('log')
# plt.title(' Mass Contours', fontsize=14, fontweight='bold')
# plt.xlim(10**-2, 10**2)
# plt.ylim(10**-2, 10**2.5)
# plt.show()
#%%
#extending mass contours

# def calculate_mass(N0, D):
#     integrand = lambda d: np.exp(-d / D) * d**3
#     mass_integral, _ = quad(integrand, 0, np.inf)  # Integrate from 0 to infinity
#     return N0 * mass_integral

# x_min, x_max = 10**0, 10**0.7
# y_min, y_max = 10**-1.77, 10**1.1 
# xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200) 
# ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)
# D_grid_extended, dryintercept_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)
# mass_grid_extended = np.zeros_like(D_grid_extended)
# for i in range(D_grid_extended.shape[0]):
#     for j in range(D_grid_extended.shape[1]):
#         mass_grid_extended[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j])
# mass_levels = np.logspace(-2, 5, 50)
# plt.figure(figsize=(10, 8))
# sc = plt.scatter(filtered_combined_clean_CDP['D'], filtered_combined_clean_CDP['dryintercept'], 
#                  c=filtered_combined_clean_CDP['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')
# cbar = plt.colorbar(sc)
# cbar.set_label('Corrected Windspeed (m/s)', fontsize=14, fontweight='bold')
# cbar.ax.tick_params(labelsize=12) 

# contour_plot = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_extended, 
#                            levels=mass_levels, colors='red', alpha=0.75)

# plt.clabel(contour_plot, inline=True, fontsize=10, fmt='%1.1e', use_clabeltext=True, colors='red')
# plt.xlabel('Slope', fontsize=19, fontweight='bold')
# plt.ylabel('Dry Intercept', fontsize=19, fontweight='bold')
# plt.yscale('log')
# plt.xscale('log')
# plt.title('Below Cloud Base January - June 2022', fontsize=19, fontweight='bold')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(fontsize=16, fontweight='bold')
# plt.yticks(fontsize=16, fontweight='bold')
# plt.tight_layout()
# plt.show()
#%%
# #2 to infinity with extended mass contours 

# def calculate_mass(N0, D):
#     integrand = lambda d: np.exp(-d / D) * d**3
#     mass_integral, _ = quad(integrand, 2, np.inf)  # Integrate from 0 to infinity
#     return N0 * mass_integral
# x_min, x_max = 10**0, 10**0.7
# y_min, y_max = 10**-1.77, 10**1.1  
# xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200)  # Denser grid
# ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)
# D_grid_extended, dryintercept_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)
# mass_grid_extended = np.zeros_like(D_grid_extended)
# for i in range(D_grid_extended.shape[0]):
#     for j in range(D_grid_extended.shape[1]):
#         mass_grid_extended[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j])
# mass_levels = np.logspace(-2, 5, 50)
# plt.figure(figsize=(10, 8))
# sc = plt.scatter(filtered_combined_clean_CDP['D'], filtered_combined_clean_CDP['dryintercept'], 
#                  c=filtered_combined_clean_CDP['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')
# cbar = plt.colorbar(sc) 
# cbar.set_label('Corrected Windspeed (m/s)', fontsize=14, fontweight='bold')  # Set label with fontsize and fontweight
# cbar.ax.tick_params(labelsize=12)
# contour_plot = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_extended, 
#                            levels=mass_levels, colors='red', alpha=0.75)

# plt.clabel(contour_plot, inline=True, fontsize=10, fmt='%1.1e', use_clabeltext=True, colors='red')
# plt.xlabel('Slope', fontsize=19, fontweight='bold')
# plt.ylabel('Dry Intercept', fontsize=19, fontweight='bold')
# plt.yscale('log')
# plt.xscale('log')
# plt.title('Below Cloud Base January - June 2022', fontsize=19, fontweight='bold')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(fontsize=16, fontweight='bold')
# plt.yticks(fontsize=16, fontweight='bold')
# plt.tight_layout()
# plt.show()
#%%

# Function to calculate mass
def calculate_mass(N0, D):
    integrand = lambda d: np.exp(-d / D) * d**3
    mass_integral, _ = quad(integrand, 2, np.inf)  # Integrate from 2 to infinity
    return N0 * mass_integral

# Set grid ranges
x_min, x_max = 10**0, 10**0.7  # Slope range
y_min, y_max = 10**-1.77, 10**1.1  # Dry intercept range
xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200)  # Denser grid for slope
ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)  # Denser grid for dry intercept
D_grid_extended, dryintercept_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)

# Initialize arrays
mass_grid_extended = np.zeros_like(D_grid_extended)
mass_data_CDP = []  # List to store leg-averaged masses

# Calculate mass grid and leg-averaged masses
for i, row in filtered_combined_clean_CDP.iterrows():
    N0 = row['dryintercept']
    D = row['D']
    # Calculate mass for each leg
    mass = calculate_mass(N0, D)
    mass_data_CDP.append(mass)

# Recalculate the mass grid for contours
for i in range(D_grid_extended.shape[0]):
    for j in range(D_grid_extended.shape[1]):
        mass_grid_extended[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j])

# Define mass contour levels
mass_levels = np.logspace(-2, 5, 50)

# Scatter plot with windspeed colormap
plt.figure(figsize=(10, 8))
sc = plt.scatter(filtered_combined_clean_CDP['D'], filtered_combined_clean_CDP['dryintercept'], 
                 c=filtered_combined_clean_CDP['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

# Add colorbar
cbar = plt.colorbar(sc)
cbar.set_label('Corrected Windspeed (m/s)', fontsize=14, fontweight='bold')
cbar.ax.tick_params(labelsize=12)

# Add mass contours
contour_plot = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_extended, 
                           levels=mass_levels, colors='red', alpha=0.75)

# Label mass contours
plt.clabel(contour_plot, inline=True, fontsize=10, fmt='%1.1e', use_clabeltext=True, colors='red')

# Plot formatting
plt.xlabel('Slope', fontsize=19, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=19, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.title('Below Cloud Base January - June 2022', fontsize=19, fontweight='bold')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Print the total number of masses calculated
print(f"Total number of masses calculated: {len(mass_data_CDP)}")
#%%
# Function to calculate mass
def calculate_mass(N0, D):
    integrand = lambda d: np.exp(-d / D) * d**3
    mass_integral, _ = quad(integrand, 2, np.inf)  # Integrate from 2 to infinity
    return N0 * mass_integral

# Set grid ranges for slope (x-axis) and dry intercept (y-axis)
x_min, x_max = 10**0, 10**0.7  # Slope range
y_min, y_max = 10**-2, 10**1  # Dry intercept range
xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200)  # Denser grid for slope
ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)  # Denser grid for dry intercept
D_grid_extended, dryintercept_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)

# Initialize arrays
mass_grid_extended = np.zeros_like(D_grid_extended)
mass_data_CDP = []  # List to store leg-averaged masses

# Calculate mass for each valid entry in the dataset
for i, row in filtered_combined_clean_CDP.iterrows():
    N0 = row['dryintercept']  # Dry intercept
    D = row['D']             # Slope
    mass = calculate_mass(N0, D)  # Calculate mass for each leg
    mass_data_CDP.append(mass)

# Recalculate the mass grid for contour plotting
for i in range(D_grid_extended.shape[0]):
    for j in range(D_grid_extended.shape[1]):
        mass_grid_extended[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j])

# Define mass contour levels
mass_levels = np.logspace(-2, 5, 50)

# Plot scatter plot of slope vs. dry intercept
plt.figure(figsize=(10, 8))
plt.scatter(filtered_combined_clean_CDP['D'], filtered_combined_clean_CDP['dryintercept'], 
            color='green', s=100, alpha=0.7, label='Data Points')

# Add mass contours
contour_plot = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_extended, 
                           levels=mass_levels, colors='red', alpha=0.75)

# Label the mass contours
plt.clabel(contour_plot, inline=True, fontsize=10, fmt='%1.1e', use_clabeltext=True, colors='red')

# Add axis labels and title
plt.xlabel('Slope', fontsize=19, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=19, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.title('CDP Below Cloud Base January - June 2022', fontsize=19, fontweight='bold')

# Set axis limits
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Adjust tick sizes
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()

# Print the total number of masses calculated
print(f"Total number of masses calculated: {len(mass_data_CDP)}")
#%%
#Removing less than 0 
# Filter out entries where D or dryintercept is <= 0
filtered_combined_clean_CDP = filtered_combined_clean_CDP[
    (filtered_combined_clean_CDP['D'] > 0) & (filtered_combined_clean_CDP['dryintercept'] > 0)
]

# Verify the filtered data
print(f"Number of valid rows after removing values <= 0: {len(filtered_combined_clean_CDP)}")

# Function to calculate mass
def calculate_mass(N0, D):
    integrand = lambda d: np.exp(-d / D) * d**3
    mass_integral, _ = quad(integrand, 2, np.inf)  # Integrate from 2 to infinity
    return N0 * mass_integral

# Set grid ranges for slope (x-axis) and dry intercept (y-axis)
x_min, x_max = 10**0, 10**0.7  # Slope range
y_min, y_max = 10**-2, 10**1.1  # Dry intercept range
xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200)  # Denser grid for slope
ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)  # Denser grid for dry intercept
D_grid_extended, dryintercept_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)

# Initialize arrays
mass_grid_extended = np.zeros_like(D_grid_extended)
mass_data_CDP = []  # List to store leg-averaged masses

# Calculate mass for each valid entry in the dataset
for i, row in filtered_combined_clean_CDP.iterrows():
    N0 = row['dryintercept']  # Dry intercept
    D = row['D']             # Slope
    mass = calculate_mass(N0, D)  # Calculate mass for each leg
    mass_data_CDP.append(mass)

# Recalculate the mass grid for contour plotting
for i in range(D_grid_extended.shape[0]):
    for j in range(D_grid_extended.shape[1]):
        mass_grid_extended[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j])

# Define mass contour levels
mass_levels = np.logspace(-2, 5, 30)

# Plot scatter plot of slope vs. dry intercept
plt.figure(figsize=(10, 8))
plt.scatter(filtered_combined_clean_CDP['D'], filtered_combined_clean_CDP['dryintercept'], 
            color='green', s=100, alpha=0.7, label='Data Points')

# Add mass contours
contour_plot = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_extended, 
                           levels=mass_levels, colors='red', alpha=0.75)

# Label the mass contours
plt.clabel(contour_plot, inline=True, fontsize=10, fmt='%1.1e', use_clabeltext=True, colors='red')

# Add axis labels and title
plt.xlabel('Slope', fontsize=19, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=19, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.title('CDP Below Cloud Base January - June 2022', fontsize=19, fontweight='bold')

# Set axis limits
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Adjust tick sizes
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()

# Print the total number of masses calculated
print(f"Total number of masses calculated: {len(mass_data_CDP)}")

#%%
#PDF of CDP masses

plt.figure(figsize=(8, 6))
sns.kdeplot(mass_data_CDP, fill=True, bw_adjust=0.5, color='blue', label='PDF of Masses')
plt.title("Probability Density Function of Leg-Average Masses", fontsize=16, fontweight='bold')
plt.xlabel("Mass)", fontsize=14, fontweight='bold')
plt.ylabel("Probability Density", fontsize=14, fontweight='bold')
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))

# Generate KDE with bounds
kde = gaussian_kde(mass_data_CDP, bw_method=0.5)  # Bandwidth adjustment for smoothing
x_vals = np.linspace(0, max(mass_data_CDP), 1000)  # Start from 0 to avoid negative values
y_vals = kde(x_vals)

# Plot the KDE
plt.fill_between(x_vals, y_vals, alpha=0.5, color='green')
plt.plot(x_vals, y_vals, color='green')

# Add labels, title, and legend
plt.title('CDP Below Cloud Base January-June 2022' ,fontsize=16, fontweight='bold')
plt.xlabel("Mass", fontsize=14, fontweight='bold')
plt.ylabel("Probability Density", fontsize=14, fontweight='bold')
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()



#%%
#reduces contours and extends them

def calculate_mass(N0, D):
    integrand = lambda d: np.exp(-d / D) * d**3
    mass_integral, _ = quad(integrand, 2, np.inf)  # Integrate from 0 to infinity
    return N0 * mass_integral

x_min, x_max = 10**0, 10**0.7
y_min, y_max = 10**-1.77, 10**1.1  
xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200) 
ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)
D_grid_extended, dryintercept_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)

mass_grid_extended = np.zeros_like(D_grid_extended)
for i in range(D_grid_extended.shape[0]):
    for j in range(D_grid_extended.shape[1]):
        mass_grid_extended[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j])
mass_levels = np.logspace(-2, 5, 20)
plt.figure(figsize=(10, 8))
sc = plt.scatter(filtered_combined_clean_CDP['D'], filtered_combined_clean_CDP['dryintercept'], 
                 c=filtered_combined_clean_CDP['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

cbar = plt.colorbar(sc) 
cbar.set_label('Corrected Windspeed (m/s)', fontsize=14, fontweight='bold')
cbar.ax.tick_params(labelsize=12)
contour_plot = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_extended, 
                           levels=mass_levels, colors='red', alpha=0.75)
plt.clabel(contour_plot, inline=True, fontsize=10, fmt='%1.1e', use_clabeltext=True, colors='red')
plt.xlabel('Slope', fontsize=19, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=19, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.title('Below Cloud Base January - June 2022', fontsize=19, fontweight='bold')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
# ##trying thr 0th moment for dry constant concentration contours

# N0_values = [entry['dry intercept'] for entry in filtered_master_BCB_dryintercept_CDP]
# D_values = [entry['D'] for entry in filtered_master_BCB_ddry_CDP]
# N0_grid = np.linspace(min(N0_values), max(N0_values), 200)  
# D_grid = np.linspace(min(D_values), max(D_values), 200)  
# concentration_grid = np.zeros((len(N0_grid), len(D_grid)))
# for i, N0 in enumerate(N0_grid):
#     for j, D in enumerate(D_grid):
#         concentration_grid[i, j] = N0 * D 
# plt.figure(figsize=(10, 8))
# contour = plt.contour(D_grid, N0_grid, concentration_grid, levels=20, cmap='viridis')  # Use plt.contour() for lines
# plt.clabel(contour, inline=True, fontsize=8)  
# plt.colorbar(contour, label='Dry Concentration')  
# plt.xlabel('D')
# plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
# plt.title(' Dry Concentration', fontsize=14, fontweight='bold')
# plt.xscale('log')
# plt.yscale('log')
# plt.show()
#%%
filtered_master_BCB_ddry_CDP = []
for i, (date, legs_exponential) in enumerate(master_BCB_exponential_CDP.items()):
    
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
            filtered_ddry_values_CDP = [center / gRh_mean for center in bin_center_CDP]
        else:
            filtered_ddry_values_CDP = [np.nan] * len(bin_center_CDP)
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
#%%
filtered_master_BCB_ddry_CDP = []
for i, (date, legs_exponential) in enumerate(master_BCB_exponential_CDP.items()):
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
            filtered_ddry_values_CDP = [center / gRh_mean for center in bin_center_CDP]
            
           
            print(f"Sample bin centers: {bin_center_CDP[:5]}")
            print(f"Sample filtered_ddry_values_CDP: {filtered_ddry_values_CDP[:5]}")
            
        else:
            filtered_ddry_values_CDP = [np.nan] * len(bin_center_CDP)
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
#%%
filtered_master_BCB_ddry_CDP = []
bin_center_CDP = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 
            10.5, 11.5, 12.5, 13.5, 15, 17, 19, 
            21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49]
print(f"Full length of bin_center: {len(bin_center_CDP)}")
print(f"Full bin_center values: {bin_center_CDP}")

for i, (date, legs_exponential) in enumerate(master_BCB_exponential_CDP.items()):
    
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
        print(f"Length of bin_center: {len(bin_center_CDP)}")
        print(f"Bin centers for date {date}, leg {j}: {bin_center_CDP}")
        
        
        if not np.isnan(gRh_mean) and gRh_mean != 0:
            filtered_ddry_values_CDP = [center / gRh_mean for center in bin_center_CDP]
            
            
            print(f"Sample bin centers: {bin_center_CDP[:5]}")
            print(f"Calculated filtered_ddry_values_CDP: {filtered_ddry_values_CDP[:5]}")
            print(f"Full filtered_ddry_values_CDP: {filtered_ddry_values_CDP}")
            
        else:
            filtered_ddry_values_CDP = [np.nan] * len(bin_center_CDP)
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
#%%
##trying thr 0th moment for dry constant concentration contours
N0_values = [entry['dry intercept'] for entry in filtered_master_BCB_dryintercept_CDP]
D_values = [entry['D'] for entry in filtered_master_BCB_ddry_CDP]
N0_grid = np.linspace(min(N0_values), max(N0_values), 200)
D_grid = np.linspace(min(D_values), max(D_values), 200)
concentration_grid = np.zeros((len(N0_grid), len(D_grid)))
for i, N0 in enumerate(N0_grid):
    for j, D in enumerate(D_grid):
        concentration_grid[i, j] = N0 * D 
plt.figure(figsize=(10, 8))
contour = plt.contour(D_grid, N0_grid, concentration_grid, levels=20, cmap='viridis')
plt.clabel(contour, inline=True, fontsize=8)
plt.colorbar(contour, label='Dry Concentration') 
plt.xlabel('D')
plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
plt.title(' Dry Concentration', fontsize=14, fontweight='bold')
plt.xscale('log')
plt.yscale('log')
plt.show()
#%%
#extending the dry concentration contours

x_min, x_max = 10**0, 10**0.7  # Slope range
y_min, y_max = 10**-1.77, 10**1.1  # Dry intercept range 
xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200)
ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)
D_grid_extended, N0_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)
concentration_grid_extended = N0_grid_extended * D_grid_extended
concentration_levels = np.linspace(concentration_grid_extended.min(), concentration_grid_extended.max(), 20)
plt.figure(figsize=(10, 8))
contour_plot = plt.contour(D_grid_extended, N0_grid_extended, concentration_grid_extended, 
                           levels=concentration_levels, cmap='viridis')

plt.clabel(contour_plot, inline=True, fontsize=10, fmt='%1.1e')
cbar = plt.colorbar(contour_plot, label='Dry Concentration (/cm³/µm)')
cbar.ax.tick_params(labelsize=16)
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')
plt.xlabel('Slope', fontsize=19, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=19, fontweight='bold')
plt.title('Dry Concentration', fontsize=19, fontweight='bold')
plt.xscale('log')
plt.yscale('log')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.tick_params(axis='both', which='major', labelsize=16, width=2, length=10, direction='in')
plt.tick_params(axis='both', which='minor', labelsize=16, width=1.5, length=6, direction='in')
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontweight('bold')
plt.tight_layout()
plt.show()
#%%
#combining extended mass contours with dry concentration contours

# def calculate_mass(N0, D):
#     integrand = lambda d: np.exp(-d / D) * d**3
#     mass_integral, _ = quad(integrand, 2, np.inf)
#     return N0 * mass_integral
# x_min, x_max = 10**0, 10**0.7
# y_min, y_max = 10**-1.77, 10**1.1 

# xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200) 
# ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)
# D_grid_extended, dryintercept_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)

# mass_grid_extended = np.zeros_like(D_grid_extended)
# for i in range(D_grid_extended.shape[0]):
#     for j in range(D_grid_extended.shape[1]):
#         mass_grid_extended[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j])
# mass_levels = np.logspace(-2, 5, 12)
# concentration_grid_extended = np.zeros_like(D_grid_extended)
# for i in range(D_grid_extended.shape[0]):
#     for j in range(D_grid_extended.shape[1]):
#         concentration_grid_extended[i, j] = dryintercept_grid_extended[i, j] * D_grid_extended[i, j]
# concentration_levels = np.logspace(-2, 3, 20)
# plt.figure(figsize=(17, 10))
# sc = plt.scatter(filtered_combined_clean_CDP['D'], filtered_combined_clean_CDP['dryintercept'], 
#                  c=filtered_combined_clean_CDP['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

# cbar = plt.colorbar(sc) 
# cbar.set_label('Corrected Windspeed (m/s)', fontsize=14, fontweight='bold')
# cbar.ax.tick_params(labelsize=12)
# mass_contour = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_extended, 
#                            levels=mass_levels, colors='red', alpha=0.75)

# plt.clabel(mass_contour, inline=True, fontsize=11, fmt='%1.1e', use_clabeltext=True, colors='red')
# dry_concentration_contour = plt.contour(D_grid_extended, dryintercept_grid_extended, concentration_grid_extended, 
#                                         levels=concentration_levels, colors='blue', alpha=0.75)

# plt.clabel(dry_concentration_contour, inline=True, fontsize=11, fmt='%1.1e', colors='blue')

# legend_elements = [
#     Line2D([0], [0], color='red', linewidth=3, label='Mass'),
#     Line2D([0], [0], color='blue', linewidth=3, label='Dry Concentration')
# ]
# plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.45, 1), fontsize=19)
# plt.xlabel('Slope', fontsize=19, fontweight='bold')
# plt.ylabel('Dry Intercept', fontsize=19, fontweight='bold')
# plt.yscale('log')
# plt.xscale('log')
# plt.title('Below Cloud Base January - June 2022', fontsize=19, fontweight='bold')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(fontsize=16, fontweight='bold')
# plt.yticks(fontsize=16, fontweight='bold')
# plt.tight_layout()
# plt.show()
#%%
#combing dry concentration and mass 
# Function to calculate mass
def calculate_mass(N0, D):
    integrand = lambda d: np.exp(-d / D) * d**3
    mass_integral, _ = quad(integrand, 2, np.inf)  # Integrate from 2 to infinity
    return N0 * mass_integral

# Set grid ranges for slope and dry intercept
x_min, x_max = 10**0, 10**0.7  # Slope range
y_min, y_max = 10**-2, 10**1.1  # Dry intercept range

xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200)  # Denser grid for slope
ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)  # Denser grid for dry intercept
D_grid_extended, dryintercept_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)

# Calculate mass grid
mass_grid_extended = np.zeros_like(D_grid_extended)
for i in range(D_grid_extended.shape[0]):
    for j in range(D_grid_extended.shape[1]):
        mass_grid_extended[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j])

# Define mass contour levels
mass_levels = np.logspace(-2, 5, 30)  # Adjusted to 30 contour levels

# Calculate dry concentration grid
concentration_grid_extended = np.zeros_like(D_grid_extended)
for i in range(D_grid_extended.shape[0]):
    for j in range(D_grid_extended.shape[1]):
        concentration_grid_extended[i, j] = dryintercept_grid_extended[i, j] * D_grid_extended[i, j]  # C_d = N0 * D

# Define dry concentration contour levels
concentration_levels = np.logspace(-2, 3, 20)

# Scatter plot of slope vs dry intercept
plt.figure(figsize=(17, 10))
plt.scatter(
    filtered_combined_clean_CDP['D'],  # Slope values
    filtered_combined_clean_CDP['dryintercept'],  # Dry intercept values
    color='green', s=100, alpha=0.7
)

# Add mass contours
mass_contour = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_extended, 
                           levels=mass_levels, colors='red', alpha=0.75)
plt.clabel(mass_contour, inline=True, fontsize=11, fmt='%1.1e', use_clabeltext=True, colors='red')

# Add dry concentration contours
dry_concentration_contour = plt.contour(D_grid_extended, dryintercept_grid_extended, concentration_grid_extended, 
                                        levels=concentration_levels, colors='blue', alpha=0.75)
plt.clabel(dry_concentration_contour, inline=True, fontsize=11, fmt='%1.1e', colors='blue')

# Add legend for mass and dry concentration contours
legend_elements = [
    Line2D([0], [0], color='red', linewidth=3, label='Mass'),
    Line2D([0], [0], color='blue', linewidth=3, label='Dry Concentration')
]
plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.45, 1), fontsize=19)

# Add axis labels and title
plt.xlabel('Slope', fontsize=19, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=19, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.title('CDP Below Cloud Base January - June 2022', fontsize=19, fontweight='bold')

# Set axis limits
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Adjust tick sizes
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()


#%%
#combing the scatter plot with density contours
def calculate_mass(N0, D):
    integrand = lambda d: np.exp(-d / D) * d**3
    mass_integral, _ = quad(integrand, 2, np.inf)
    return N0 * mass_integral

# Define grid limits
x_min, x_max = 10**0, 10**0.7
y_min, y_max = 10**-2.2, 10**1.3

# Generate extended grids
xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200) 
ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)
D_grid_extended, dryintercept_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)

# Calculate mass grid
mass_grid_extended = np.zeros_like(D_grid_extended)
for i in range(D_grid_extended.shape[0]):
    for j in range(D_grid_extended.shape[1]):
        mass_grid_extended[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j])
mass_levels = np.logspace(-2, 5, 12)

# Calculate dry concentration grid
concentration_grid_extended = np.zeros_like(D_grid_extended)
for i in range(D_grid_extended.shape[0]):
    for j in range(D_grid_extended.shape[1]):
        concentration_grid_extended[i, j] = dryintercept_grid_extended[i, j] * D_grid_extended[i, j]
concentration_levels = np.logspace(-2, 3, 20)

# **Verify Unique Points**
unique_points = filtered_combined_clean_CDP[['D', 'dryintercept']].drop_duplicates()
print(f"Unique points: {len(unique_points)} / Total points: {len(filtered_combined_clean_CDP)}")

# Add jitter to overlapping points if necessary
filtered_combined_clean_CDP['D_jitter'] = filtered_combined_clean_CDP['D'] + np.random.normal(0, 0.01, len(filtered_combined_clean_CDP))
filtered_combined_clean_CDP['dryintercept_jitter'] = filtered_combined_clean_CDP['dryintercept'] + np.random.normal(0, 0.01, len(filtered_combined_clean_CDP))

# **Check Points Outside Plot Limits**
x_outliers = filtered_combined_clean_CDP[(filtered_combined_clean_CDP['D'] < x_min) | (filtered_combined_clean_CDP['D'] > x_max)]
y_outliers = filtered_combined_clean_CDP[(filtered_combined_clean_CDP['dryintercept'] < y_min) | (filtered_combined_clean_CDP['dryintercept'] > y_max)]
print(f"Points outside x-axis range: {len(x_outliers)}")
print(f"Points outside y-axis range: {len(y_outliers)}")

# Plot with KDE for density visualization
plt.figure(figsize=(12, 10))
sns.kdeplot(data=filtered_combined_clean_CDP, x='D', y='dryintercept', cmap='coolwarm', fill=True, alpha=0.5)
plt.title('Point Density', fontsize=19, fontweight='bold')
plt.xlabel('D', fontsize=19, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=19, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.show()

# Main scatter plot with contours
plt.figure(figsize=(15, 10))
sc = plt.scatter(
    filtered_combined_clean_CDP['D_jitter'], 
    filtered_combined_clean_CDP['dryintercept_jitter'], 
    c=filtered_combined_clean_CDP['Windspeed'], cmap='viridis', s=30, alpha=0.7, label='Windspeed Present'
)

# Add color bar
cbar = plt.colorbar(sc) 
cbar.set_label('Corrected Windspeed (m/s)', fontsize=14, fontweight='bold')
cbar.ax.tick_params(labelsize=12)

# Add mass contours
mass_contour = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_extended, 
                           levels=mass_levels, colors='red', alpha=0.75)
plt.clabel(mass_contour, inline=True, fontsize=11, fmt='%1.1e', use_clabeltext=True, colors='red')

# Add dry concentration contours
dry_concentration_contour = plt.contour(D_grid_extended, dryintercept_grid_extended, concentration_grid_extended, 
                                        levels=concentration_levels, colors='blue', alpha=0.75)
plt.clabel(dry_concentration_contour, inline=True, fontsize=11, fmt='%1.1e', colors='blue')

# Legend
legend_elements = [
    Line2D([0], [0], color='red', linewidth=3, label='Mass'),
    Line2D([0], [0], color='blue', linewidth=3, label='Dry Concentration')
]
plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.45, 1), fontsize=19)

# Axis formatting
plt.xlabel('Slope', fontsize=19, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=19, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.title('Below Cloud Base January - June 2022', fontsize=19, fontweight='bold')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

#%%
# ##D50 mass calculation 
# N0_values = [entry['dry intercept'] for entry in filtered_master_BCB_dryintercept_CDP]
# D_values = [entry['D'] for entry in filtered_master_BCB_ddry_CDP]
# def mass_integrand(d, D):
#     return np.exp(-d / D) * d**3
# total_mass_CDP = []
# for D in D_values:
#     mass, _ = quad(mass_integrand, 2, np.inf, args=(D,))
#     total_mass_CDP.append(mass)

# cumulative_mass_CDP = np.cumsum(total_mass_CDP)
# total_mass_value_CDP = np.sum(total_mass_CDP)
# median_mass_threshold = total_mass_value_CDP / 2

# median_diameter = None
# for i, cm in enumerate(cumulative_mass_CDP):
#     if cm >= median_mass_threshold:
#         median_diameter = D_values[i]
#         break
# plt.figure(figsize=(10, 8))
# contour = plt.contour(D_grid, N0_grid, concentration_grid, levels=20, cmap='viridis')
# plt.clabel(contour, inline=True, fontsize=8)
# cbar = plt.colorbar(contour)
# cbar.set_label('Dry Concentration', fontsize=14, fontweight='bold')
# cbar.ax.tick_params(labelsize=12)

# plt.xlabel('D', fontsize=14, fontweight='bold')
# plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
# plt.title('Dry Concentration with Median Diameter', fontsize=14, fontweight='bold')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(10**-1, 10**1.5)
# plt.ylim(10**1, 10**2.1)

# if median_diameter is not None:
#     plt.axvline(x=median_diameter, color='red', linestyle='--', label=f'Median Diameter: {median_diameter:.2f}')
#     plt.legend()
# plt.show()
#%%
##Ntd distribution varies with windspeed

Z0 = 0.02 
Z10 = 10 
def correct_windspeed(windspeed, altitude):
    return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))
combined_data_CDP = {
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
                combined_data_CDP['Date'].append(date)
                combined_data_CDP['BCB_start'].append(BCB_start)
                combined_data_CDP['BCB_stop'].append(BCB_stop)
                combined_data_CDP['Ntd'].append(Ntd)
                combined_data_CDP['Windspeed'].append(corrected_windspeed)

        except IndexError as e:
            print(f"Index error at i={i}, j={j}: {e}")
            continue


df_combined_CDP = pd.DataFrame(combined_data_CDP)
print(df_combined_CDP.describe())
df_with_windspeed = df_combined_CDP[df_combined_CDP['Windspeed'].notna()]
df_nan_windspeed = df_combined_CDP[df_combined_CDP['Windspeed'].isna()]
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
#%%
# ##Looking at slope versus NtdNT

combined_data_CDP = {
    'Date': [],
    'D': [],
    'NtdNt': []
}

flat_ntdNt = [item for item in filtered_master_BCB_NtdNt_CDP]

ntdNt_index = 0
for date, exp_params_list in master_BCB_exponential_CDP.items():
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
            
            
            combined_data_CDP['Date'].append(date)
            combined_data_CDP['D'].append(D)
            combined_data_CDP['NtdNt'].append(NtdNt)

        except ValueError as e:
            print(f"Value error at date={date} for D value: {exp_params['D']} - {e}")
        except TypeError as e:
            print(f"Type error at date={date} for D value: {exp_params['D']} - {e}")
        except IndexError as e:
            print(f"Index error at date={date}: {e}")
        except Exception as e:
            print(f"Unexpected error at date={date}: {e}")

df_combined_CDP = pd.DataFrame(combined_data_CDP)

plt.figure(figsize=(10, 6))
plt.scatter(df_combined_CDP['D'], df_combined_CDP['NtdNt'])
plt.xlabel('Slope', fontsize=14, fontweight='bold')
plt.ylabel('Ratio of total droplet concentration of dry droplets \n with diameter larger than 2um \n to ambient concentration', fontsize=14, fontweight='bold')
plt.title('Below cloud base January - June 2022', fontsize=14, fontweight='bold')
plt.grid(True)
plt.yscale('log')
plt.ylim(10**-2, 10**0.5)
plt.xlim(10**-0.2, 10**0.9)
plt.xscale('log')
plt.show()
#%%
#25 bins: A common x-axis is created to average the dry size distributions onto
# We will use this common x-axis to interpolate the dry size distributions. Note that the 
# number of bins is arbitrary and can be adjusted as needed and the spacing between the bins is equal.  
def size_distribution(x, dryint, D):
    dryint = dryint 
    return dryint * np.exp(-x / D)
#We are randomly assigning bin lengths here, but we have 25 bins that end at 10 um d or any set diameter you want 
common_bins = np.linspace(2.5, 10, 25)

interpolated_values_CDP = []
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
    
    
    interp_func = interp1d(ddry_values, size_distribution(ddry_values, dryint, D), kind='linear', fill_value='extrapolate')
    interpolated_leg_values = interp_func(common_bins)
    interpolated_values_CDP.append({
        'Date': date,
        'Leg_index': leg_index,
        'BCB_start': BCB_start,
        'BCB_stop': BCB_stop,
        'interpolated_values': interpolated_leg_values.tolist()
    })

for entry in interpolated_values_CDP:
    print(f"Date: {entry['Date']}, Leg_index: {entry['Leg_index']}, BCB_start: {entry['BCB_start']}, BCB_stop: {entry['BCB_stop']}, Interpolated Values: {entry['interpolated_values']}")


common_bins = np.linspace(0, 10, 25)
plt.figure(figsize=(12, 8))
for entry in interpolated_values_CDP:
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
problematic_legs_CDP = []

for entry in interpolated_values_CDP:
    leg_values = np.array(entry['interpolated_values'])
    

    if np.any(leg_values < 0):
        problematic_legs_CDP.append({
            'Date': entry['Date'],
            'Leg_index': entry['Leg_index'],
            'BCB_start': entry['BCB_start'],
            'BCB_stop': entry['BCB_stop'],
            'Min_value': np.min(leg_values)
        })

if problematic_legs_CDP:
    print("Problematic Legs (Values Below 0):")
    for leg in problematic_legs_CDP:
        print(f"Date: {leg['Date']}, Leg_index: {leg['Leg_index']}, "
              f"BCB_start: {leg['BCB_start']}, BCB_stop: {leg['BCB_stop']}, "
              f"Min_value: {leg['Min_value']}")
else:
    print("No problematic legs found (no values below 0).")
#%%
problematic_legs_CDP = [
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
problematic_set_CDP = set(problematic_legs_CDP)
#%%
def size_distribution(x, dryint, D):
    return dryint * np.exp(-x / D)
common_bins = np.linspace(2.5, 10, 25)
interpolated_values_CDP = []
for i in range(len(filtered_master_BCB_ddry_CDP)):
    entry_ddry = filtered_master_BCB_ddry_CDP[i]
    entry_dryintercept = filtered_master_BCB_dryintercept_CDP[i] 
    
    
    date = entry_ddry['Date']
    leg_index = entry_ddry['Leg_index']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    
    
    if (date, leg_index) in problematic_set_CDP:
        continue
    
    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])
    dryint = entry_dryintercept['dry intercept']
    
   
    interp_func = interp1d(ddry_values, size_distribution(ddry_values, dryint, D), 
                           kind='linear', fill_value='extrapolate')
    interpolated_leg_values = interp_func(common_bins)
    
    
    interpolated_values_CDP.append({
        'Date': date,
        'Leg_index': leg_index,
        'BCB_start': BCB_start,
        'BCB_stop': BCB_stop,
        'interpolated_values': interpolated_leg_values.tolist()
    })

plt.figure(figsize=(12, 8))
for entry in interpolated_values_CDP:
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
def size_distribution(x, dryint, D):
    return dryint * np.exp(-x / D)

common_bins = np.linspace(2.5, 10, 25)
windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]

grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}
missing_windspeed_count = 0
interpolation_failures = 0

unique_leg_ids = set()  # Set to track unique legs

print(f"Total input legs: {len(filtered_master_BCB_ddry_CDP)}")

for i in range(len(filtered_master_BCB_ddry_CDP)):
    entry_ddry = filtered_master_BCB_ddry_CDP[i]
    entry_dryintercept = filtered_master_BCB_dryintercept_CDP[i]
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']

    if (date, leg_index) in problematic_set_CDP:
        continue

    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])
    dryint = entry_dryintercept['dry intercept']

    windspeed_entry = df_combined_CDP[
        (df_combined_CDP['Date'] == date) & 
        (df_combined_CDP['BCB_start'] == BCB_start) & 
        (df_combined_CDP['BCB_stop'] == BCB_stop)
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

    # Assign to the appropriate windspeed bin
    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:
            grouped_distributions[idx].append(interpolated_leg_values)
            mean_windspeeds[idx].append(windspeed)
            unique_leg_ids.add((date, leg_index))  # Add to unique leg set
            break

print(f"Total problematic legs excluded: {len(problematic_set_CDP)}")
print(f"Total legs with missing windspeed data: {missing_windspeed_count}")
print(f"Total interpolation failures: {interpolation_failures}")

for idx, group in grouped_distributions.items():
    print(f"Windspeed bin {idx} ({windspeed_bins[idx]} m/s): {len(group)} legs")

# Plot the data
plt.figure(figsize=(12, 8))
for idx, ranges in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        avg_distribution = np.nanmean(grouped_distributions[idx], axis=0)
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])
        plt.plot(common_bins, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")
    else:
        print(f" - Bin {idx} is empty")

total_legs_CDP = len(unique_leg_ids)  # Unique count of legs
print(f"Total number of unique legs plotted: {total_legs_CDP}")

plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Size distribution by wind speed', fontweight='bold')
plt.legend(title="Average wind speed (m/s)")
plt.yscale('log')
plt.tight_layout()
plt.show()

#%%
#How windspeed, slope, and Ntd correlate with windspeed 

Z0 = 0.02 
Z10 = 10  
def correct_windspeed(windspeed, altitude):
    return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))
combined_data_CDP = {
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

            
            if date in master_BCB_exponential_CDP:
                for exp_params in master_BCB_exponential_CDP[date]:
                    
                    if exp_params['BCB_start'] == BCB_start and exp_params['BCB_stop'] == BCB_stop:
                        D = exp_params['D']
                        print(f"Found D for Date: {date}, BCB_start: {BCB_start}, BCB_stop: {BCB_stop}, D: {D}")
                        break

            
            for ntd_data in filtered_master_BCB_ntd_CDP:
                if ntd_data['Date'] == date and ntd_data['BCB_start'] == BCB_start and ntd_data['BCB_stop'] == BCB_stop:
                    Ntd = ntd_data['Ntd']
                    break

            
            if Ntd is not None and D is not None:
                combined_data_CDP['Date'].append(date)
                combined_data_CDP['BCB_start'].append(BCB_start)
                combined_data_CDP['BCB_stop'].append(BCB_stop)
                combined_data_CDP['Ntd'].append(Ntd)
                combined_data_CDP['D'].append(D)
                combined_data_CDP['Windspeed'].append(corrected_windspeed)

        except IndexError as e:
            print(f"Index error at i={i}, j={j}: {e}")
            continue


df_combined_CDP = pd.DataFrame(combined_data_CDP)
print(df_combined_CDP.describe())

df_with_windspeed = df_combined_CDP[df_combined_CDP['Windspeed'].notna()]
df_nan_windspeed = df_combined_CDP[df_combined_CDP['Windspeed'].isna()]

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
def size_distribution(x, dryint, D):
    return dryint * np.exp(-x / D)

common_bins = np.linspace(2.5, 10, 25)
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

    
    if (date, leg_index) in problematic_set_CDP:
        problematic_legs_skipped += 1
        continue

    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])
    dryint = entry_dryintercept['dry intercept']

    
    windspeed_entry = df_combined_CDP[
        (df_combined_CDP['Date'] == date) &
        (df_combined_CDP['BCB_start'] == BCB_start) &
        (df_combined_CDP['BCB_stop'] == BCB_stop)
    ]
    
    if windspeed_entry.empty:
        missing_windspeed_count += 1
        continue
    
    windspeed = windspeed_entry['Windspeed'].values[0]

    
    try:
        size_dist = size_distribution(ddry_values, dryint, D)
        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_values_CDP = interp_func(common_bins)
        interpolated_values_CDP = np.nan_to_num(interpolated_values_CDP, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        interpolation_failures += 1
        continue

   
    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:
            grouped_concentrations[idx].append(interpolated_values_CDP)
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

common_bins = np.linspace(2.5, 10, 10)
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

    
    if (date, leg_index) in problematic_set_CDP:
        problematic_legs_skipped += 1
        continue

    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])
    dryint = entry_dryintercept['dry intercept']

    
    windspeed_entry = df_combined_CDP[
        (df_combined_CDP['Date'] == date) &
        (df_combined_CDP['BCB_start'] == BCB_start) &
        (df_combined_CDP['BCB_stop'] == BCB_stop)
    ]
    
    if windspeed_entry.empty:
        missing_windspeed_count += 1
        continue
    
    windspeed = windspeed_entry['Windspeed'].values[0]

    
    try:
        size_dist = size_distribution(ddry_values, dryint, D)
        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_values_CDP = interp_func(common_bins)
        interpolated_values_CDP = np.nan_to_num(interpolated_values_CDP, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        interpolation_failures += 1
        continue

   
    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:
            grouped_concentrations[idx].append(interpolated_values_CDP)
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
# ##trying to fix size distribution and add ntd 

# def size_distribution(d, dryint, D):
#     return dryint * np.exp(-d / D)

# dmin = 2  
# dmax = np.inf 

# integrated_concentrations = []
# for i in range(len(filtered_master_BCB_ddry_CDP)):
#     entry_ddry = filtered_master_BCB_ddry_CDP[i]
#     entry_dryintercept = filtered_master_BCB_dryintercept_CDP[i]  
    
#     date = entry_ddry['Date']
#     BCB_start = entry_ddry['BCB_start']
#     BCB_stop = entry_ddry['BCB_stop']
#     leg_index = entry_ddry['Leg_index'] 
#     D = entry_ddry['D']  
#     dryint = entry_dryintercept['dry intercept']  
    

#     total_concentration, _ = quad(size_distribution, dmin, dmax, args=(dryint, D))
    
    
#     integrated_concentrations.append({
#         'Date': date,
#         'Leg_index': leg_index,
#         'BCB_start': BCB_start,
#         'BCB_stop': BCB_stop,
#         'Total Concentration': total_concentration
#     })


# for total in integrated_concentrations:
#     print(f"Date: {total['Date']}, Leg_index: {total['Leg_index']}, "
#           f"Total Concentration: {total['Total Concentration']:.3f} /cm^3/um")

# dates = [entry['Date'] for entry in integrated_concentrations]
# legs = [entry['Leg_index'] for entry in integrated_concentrations]
# total_concs = [entry['Total Concentration'] for entry in integrated_concentrations]
# plt.figure(figsize=(12, 8))
# plt.bar(range(len(total_concs)), total_concs, tick_label=[f"{d}-{l}" for d, l in zip(dates, legs)])
# plt.ylabel('Total Concentration (/cm^3/um)', fontweight='bold')
# plt.xlabel('Leg (Date-Leg)', fontweight='bold')
# plt.title('Total below cloud base concentration (/cm^3/um)', fontweight='bold')
# plt.tight_layout()
# plt.show()
#%%
#Size distributiona using d instead of ddry
def size_distribution(d, dryint, D):
    return dryint * np.exp(-d / D)

dmin = 2  
dmax = np.inf  
common_bins = np.linspace(2.5, 10, 25)

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
common_bins = np.linspace(2.5, 16, 25)
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

common_bins = np.linspace(2.5, 16, 25)
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


    size_dist_values = size_distribution(ddry_values, N0, D)
    interp_func = interp1d(ddry_values, size_dist_values, kind='linear', fill_value='extrapolate')
    interpolated_leg_values = interp_func(common_bins)

    
    if np.any(interpolated_leg_values < 0):
        print(f"Skipping leg {date}, {leg_index} due to negative values")
        continue 

    plt.plot(common_bins, interpolated_leg_values, label=f"Date: {date}, Leg: {leg_index} ({BCB_start} - {BCB_stop})")

plt.ylabel('Concentration (/cm³/μm)', fontweight='bold')
plt.xlabel('Dry diameter (μm)', fontweight='bold')
plt.title('Below cloud base January-June 2022', fontweight='bold')
plt.ylim(0, 10)
plt.tight_layout()
plt.show()
#%%
def size_distribution(ddry, N0, D):
    return N0 * np.exp(-ddry / D)

common_bins = np.linspace(2.5, 10, 10)
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

    windspeed_entry = df_combined_CDP[
        (df_combined_CDP['Date'] == date) &
        (df_combined_CDP['BCB_start'] == BCB_start) &
        (df_combined_CDP['BCB_stop'] == BCB_stop)
    ]

    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]

        size_dist_values = size_distribution(ddry_values, N0, D)

        if np.any(np.isnan(size_dist_values)):
            skipped_nan += 1
            continue

        interp_func = interp1d(ddry_values, size_dist_values, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        
        if np.any(np.isnan(interpolated_leg_values)):
            skipped_nan += 1
            continue

        
        assigned = False
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_concentrations[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                assigned = True
                break
        
        if not assigned:
            skipped_no_bin += 1

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
plt.title('Average size distribution by wind speed', fontweight='bold')
plt.legend(title="Average wind speed (m/s)")
plt.tight_layout()
plt.show()
#%%
#Dry size distribution function CORRECT

def size_distribution(ddry, N0, D):
    return N0 * np.exp(-ddry / D)

common_bins = np.linspace(2.5, 10, 10)
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

    
    windspeed_entry = df_combined_CDP[
        (df_combined_CDP['Date'] == date) &
        (df_combined_CDP['BCB_start'] == BCB_start) &
        (df_combined_CDP['BCB_stop'] == BCB_stop)
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
# def fit_function(x, D):
#     return dryint * np.exp(-x / D)

# common_bins = np.linspace(0, 10, 10)  
# windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]
# grouped_concentrations = {i: [] for i in range(len(windspeed_bins))}
# mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# for i in range(len(filtered_master_BCB_ddry_CDP)):
#     entry_ddry = filtered_master_BCB_ddry_CDP[i]
#     entry_dryintercept = filtered_master_BCB_dryintercept_CDP[i]

  
#     date = entry_ddry['Date']
#     BCB_start = entry_ddry['BCB_start']
#     BCB_stop = entry_ddry['BCB_stop']
#     leg_index = entry_ddry['Leg_index']

  
#     if (date, leg_index) in problematic_set_CDP:
#         continue

#     D = entry_ddry['D']
#     ddry_values = np.array(entry_ddry['filtered_ddry'])
#     dryint = entry_dryintercept['dry intercept']

   
#     windspeed_entry = df_combined_CDP[
#         (df_combined_CDP['Date'] == date) &
#         (df_combined_CDP['BCB_start'] == BCB_start) &
#         (df_combined_CDP['BCB_stop'] == BCB_stop)
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
#                     label=f"Fit {avg_windspeed:.1f} m/s: y = {dryint_fit:.2f} * exp(-x / {D_fit:.2f})")
    
            
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
def size_distribution(ddry, N0, D):
    return N0 * np.exp(-ddry / D)

def fit_function(x, N0, D):
    return N0 * np.exp(-x / D)

common_bins = np.linspace(2.5, 10, 10)
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
    N0 = entry_ddry['n0']
    D = entry_ddry['D']  
    ddry_values = np.array(entry_ddry['filtered_ddry'])  
    dryint = entry_dryintercept['dry intercept']  

    
    windspeed_entry = df_combined_CDP[
        (df_combined_CDP['Date'] == date) &
        (df_combined_CDP['BCB_start'] == BCB_start) &
        (df_combined_CDP['BCB_stop'] == BCB_stop)
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

common_bins = np.linspace(2.5, 10, 10)
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

    
    windspeed_entry = df_combined_CDP[
        (df_combined_CDP['Date'] == date) &
        (df_combined_CDP['BCB_start'] == BCB_start) &
        (df_combined_CDP['BCB_stop'] == BCB_stop)
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
plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin center diameter (µm)', fontweight='bold')
plt.title('Below cloud base January - June 2022', fontweight='bold')
plt.legend(title="Average wind speed binning ")
plt.tight_layout()
plt.show()
#%%
# Fitting an exponential to averaged curves using a basic exponential

def size_distribution(ddry, N0, D):
    return N0 * np.exp(-ddry / D)

def fit_function(x, dryint, D):
    return dryint * np.exp(-D * x)

common_bins = np.linspace(2.5, 10, 10)
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

    if (date, leg_index) in problematic_set_CDP:
        continue

    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])
    dryint = entry_dryintercept['dry intercept']

    windspeed_entry = df_combined_CDP[
        (df_combined_CDP['Date'] == date) &
        (df_combined_CDP['BCB_start'] == BCB_start) &
        (df_combined_CDP['BCB_stop'] == BCB_stop)
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
# Make Y_BCB_calc which is 476 in length match with filtered_master_BCB_grH
#which has 460 entries. 

grh_leg_set = set(
    (entry["Date"], entry["BCB_start"], entry["BCB_stop"])
    for sublist in filtered_master_BCB_gRH_CDP for entry in sublist
)
filtered_Y_CDP_calc = [
    entry for entry in Y_CDP_calc
    if (entry["Date"], entry["BCB_start"], entry["BCB_stop"]) in grh_leg_set
]


print(f"Total entries in Y_CDP_calc: {len(Y_CDP_calc)}")
print(f"Total entries in filtered_master_BCB_gRH_CDP: {sum(len(sublist) for sublist in filtered_master_BCB_gRH_CDP)}")  # Adjusted for nested lists
print(f"Total entries in filtered_Y_CDP_calc: {len(filtered_Y_CDP_calc)}")


assert len(filtered_Y_CDP_calc) == sum(len(sublist) for sublist in filtered_master_BCB_gRH_CDP), "Filtered list length does not match grh list!"

#%%
Y_CDP_df = pd.DataFrame(filtered_Y_CDP_calc)
Y_CDP_df["Leg_index"] = Y_CDP_df.groupby("Date").cumcount()
print(f"Sample data with Leg_index:\n{Y_CDP_df[['Date', 'Leg_index', 'BCB_start', 'BCB_stop']].head(20)}")
#%%

Y_CDP_clean = Y_CDP_df[
    ~Y_CDP_df[["Date", "Leg_index"]].apply(tuple, axis=1).isin(problematic_set_CDP)
].copy()

Y_CDP_clean.fillna(0, inplace=True)
#%%
common_bins = np.array(filtered_master_BCB_ddry_CDP[0]["filtered_ddry"])  # Raw bin diameters
def calculate_mass(diameter, concentration):
    return concentration * (np.pi / 6) * (diameter**3)
def calculate_cumulative_mass(diameters, masses):
    cumulative_mass_CDP = np.cumsum(masses)
    total_mass_CDP = np.sum(masses)
    normalized_cumulative_mass = cumulative_mass_CDP / total_mass_CDP

    interpolation_func = interp1d(normalized_cumulative_mass, diameters, kind="linear", fill_value="extrapolate")
    diameter_at_50_percent = interpolation_func(0.5)
    return normalized_cumulative_mass, diameter_at_50_percent

plt.figure(figsize=(12, 8))
diameters_at_50_mass = [] 


for i, row in Y_CDP_clean.iterrows():
   
    concentrations = np.array([row.get(f"Bin{j}_Y_mean", 0) for j in range(0, 30)])

    
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

#%%
#average with just one line
all_masses_CDP = []
for _, row in Y_CDP_clean.iterrows():
    
    concentrations = np.array([row.get(f"Bin{j}_Y_mean", 0) for j in range(0, 30)])

    
    masses = calculate_mass(common_bins, concentrations)
    all_masses_CDP.append(masses)
all_masses_CDP = np.array(all_masses_CDP)

average_masses_CDP = np.mean(all_masses_CDP, axis=0)
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
#%%
all_masses_CDP = []
for _, row in Y_CDP_clean.iterrows():
    # Get concentrations for all bins
    concentrations = np.array([row.get(f"Bin{j}_Y_mean", 0) for j in range(0, 30)])
    
    # Calculate masses for the current leg
    masses = calculate_mass(common_bins, concentrations)
    all_masses_CDP.append(masses)

# Convert list of masses to a NumPy array for easier averaging
all_masses_CDP = np.array(all_masses_CDP)

# Calculate average mass distribution across all legs
average_masses_CDP = np.mean(all_masses_CDP, axis=0)

# Calculate cumulative mass and the diameter at 50% cumulative mass for the average distribution
normalized_cumulative_mass, diameter_at_50 = calculate_cumulative_mass(common_bins, average_masses_CDP)

# Plot the average cumulative mass distribution
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

#%%
##Function to calculate the 50% mass diameter for fitted distribution using 
#slope and dry-intercept values for every leg pre-windspeed binning. 

def cumulative_mass_CDP(N0, D, d_max):
   
    integrand = lambda d: N0 * np.exp(-d / D) * d**3
    cumulative_mass_CDP, _ = quad(integrand, 2, d_max) 
    return cumulative_mass_CDP

def total_mass_CDP(N0, D):
    integrand = lambda d: N0 * np.exp(-d / D) * d**3
    total_mass_CDP, _ = quad(integrand, 2, np.inf)
    return total_mass_CDP

diameters = np.linspace(2, 50, 100) 

diameters_at_50_mass = []
plt.figure(figsize=(12, 8))

for i, row in filtered_combined_clean_CDP.iterrows():
    N0 = row['dryintercept']
    D = row['D']

    
    M_total = total_mass_CDP(N0, D)
    cumulative_masses_CDP = [cumulative_mass_CDP(N0, D, d) for d in diameters]

    
    normalized_cumulative_mass = np.array(cumulative_masses_CDP) / M_total


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
#%%



#%%
##Function to calculate the 50% mass diameter for fitted distribution using 
#slope and dry-intercept values for average or cumulative pre-windspeed binning.

def cumulative_mass_CDP(N0, D, d_max):

    integrand = lambda d: N0 * np.exp(-d / D) * d**3
    cumulative_mass_CDP, _ = quad(integrand, 2, d_max)  # Integrate from 2 µm to d_max
    return cumulative_mass_CDP

def total_mass_CDP(N0, D):
    integrand = lambda d: N0 * np.exp(-d / D) * d**3
    total_mass_CDP, _ = quad(integrand, 2, np.inf)  
    return total_mass_CDP


diameters = np.linspace(2, 50, 100) 

cumulative_mass_all_legs = np.zeros_like(diameters)

for i, row in filtered_combined_clean_CDP.iterrows():
    N0 = row['dryintercept']
    D = row['D']

    
    M_total = total_mass_CDP(N0, D)

    
    cumulative_masses_CDP = np.array([cumulative_mass_CDP(N0, D, d) for d in diameters]) / M_total

   
    cumulative_mass_all_legs += cumulative_masses_CDP


average_cumulative_mass = cumulative_mass_all_legs / len(filtered_combined_clean_CDP)


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
#%%
#Averaging the size distribution for all legs and plotting the average size distribution.

filtered_combined_clean_CDP = filtered_combined_clean_CDP.reset_index(drop=True)
diameters = np.linspace(2.5, 10, 10)
size_distributions_all_legs = np.zeros((len(filtered_combined_clean_CDP), len(diameters)))
for i, row in filtered_combined_clean_CDP.iterrows():
    N0 = row['dryintercept'] 
    D = row['D']  
    

    size_distribution = N0 * np.exp(-diameters / D)
    
    
    size_distribution /= np.sum(size_distribution)
    
    
    size_distributions_all_legs[i, :] = size_distribution
average_size_distribution = np.mean(size_distributions_all_legs, axis=0)
plt.figure(figsize=(12, 8))
plt.plot(diameters, average_size_distribution, label='Average Size Distribution', color='blue', linewidth=2.5)
plt.xlabel('Dry diameter (µm)', fontsize=16, fontweight='bold')
plt.ylabel('Average droplet concentration (/cm³/µm)', fontsize=16, fontweight='bold')
plt.title('Average CDP Size Distribution January-June 2022', fontsize=17, fontweight='bold')
plt.yscale('log')
plt.tick_params(axis='both', which='major', labelsize=15, width=3)
plt.tight_layout()
plt.legend(fontsize=18)
plt.show()
# %%
#Plotting CAS and CDP average size distributions together 
# Reset the indices of both dataframes to avoid indexing issues
filtered_combined_clean_CDP = filtered_combined_clean_CDP.reset_index(drop=True)
filtered_combined_clean = filtered_combined_clean.reset_index(drop=True)

# Diameters for the size distribution bins (adjust as needed for consistency)
diameters_CDP = np.linspace(2, 50, 10)  # Diameters for CDP (µm)
diameters_CAS = np.linspace(2, 50, 10)   # Diameters for CAS (µm)

# Initialize arrays to store size distributions
size_distributions_CDP = np.zeros((len(filtered_combined_clean_CDP), len(diameters_CDP)))
size_distributions_CAS = np.zeros((len(filtered_combined_clean), len(diameters_CAS)))

# Calculate size distributions for CDP
for i, row in filtered_combined_clean_CDP.iterrows():
    N0 = row['dryintercept']  # Dry intercept for CDP leg
    D = row['D']              # Slope for CDP leg
    size_distribution = N0 * np.exp(-diameters_CDP / D)
    size_distribution /= np.sum(size_distribution)  # Normalize
    size_distributions_CDP[i, :] = size_distribution

# Calculate size distributions for CAS
for i, row in filtered_combined_clean.iterrows():
    N0 = row['dryintercept']  # Dry intercept for CAS leg
    D = row['D']              # Slope for CAS leg
    size_distribution = N0 * np.exp(-diameters_CAS / D)
    size_distribution /= np.sum(size_distribution)  # Normalize
    size_distributions_CAS[i, :] = size_distribution

# Calculate the average size distributions
average_size_distribution_CDP = np.mean(size_distributions_CDP, axis=0)
average_size_distribution_CAS = np.mean(size_distributions_CAS, axis=0)

# Plot the average size distributions
plt.figure(figsize=(12, 8))
plt.plot(diameters_CDP, average_size_distribution_CDP, label='CDP', color='green', linewidth=2.5)
plt.plot(diameters_CAS, average_size_distribution_CAS, label='CAS', color='blue', linewidth=2.5)

# Plot formatting
plt.xlabel('Dry diameter (µm)', fontsize=16, fontweight='bold')
plt.ylabel('Average droplet concentration (/cm³/µm)', fontsize=16, fontweight='bold')
plt.title('Below cloud base average size distributions January-June 2022', fontsize=17, fontweight='bold')


# %%
#Plotting average cumulative median diameter for CDP and CAS together 
# Function to calculate cumulative and total mass Directly computes the cumulative mass distribution for each leg and averages those cumulative mass distributions across all legs.
#The plotted lines represent the averaged cumulative mass distributions across legs.
def cumulative_mass_CDP(N0, D, d_max):
    integrand = lambda d: N0 * np.exp(-d / D) * d**3
    cumulative_mass_CDP, _ = quad(integrand, 2, d_max)  # Integrate from 2 µm to d_max
    return cumulative_mass_CDP

def total_mass_CDP(N0, D):
    integrand = lambda d: N0 * np.exp(-d / D) * d**3
    total_mass_CDP, _ = quad(integrand, 2, np.inf)  
    return total_mass_CDP

def cumulative_mass_CAS(N0, D, d_max):
    integrand = lambda d: N0 * np.exp(-d / D) * d**3
    cumulative_mass_CAS, _ = quad(integrand, 2, d_max)  # Integrate from 2 µm to d_max
    return cumulative_mass_CAS

def total_mass_CAS(N0, D):
    integrand = lambda d: N0 * np.exp(-d / D) * d**3
    total_mass_CAS, _ = quad(integrand, 2, np.inf)  
    return total_mass_CAS

# Define diameter range
diameters = np.linspace(2, 50, 100)

# Initialize cumulative mass arrays
cumulative_mass_CDP_all_legs = np.zeros_like(diameters)
cumulative_mass_CAS_all_legs = np.zeros_like(diameters)

# Calculate cumulative mass for CDP
for i, row in filtered_combined_clean_CDP.iterrows():
    N0 = row['dryintercept']
    D = row['D']
    M_total_CDP = total_mass_CDP(N0, D)
    cumulative_masses_CDP = np.array([cumulative_mass_CDP(N0, D, d) for d in diameters]) / M_total_CDP
    cumulative_mass_CDP_all_legs += cumulative_masses_CDP

# Average cumulative mass for CDP
average_cumulative_mass_CDP = cumulative_mass_CDP_all_legs / len(filtered_combined_clean_CDP)

# Calculate diameter at 50% cumulative mass for CDP
interpolation_func_CDP = interp1d(average_cumulative_mass_CDP, diameters, kind='linear', fill_value='extrapolate')
diameter_at_50_CDP = interpolation_func_CDP(0.5)

# Calculate cumulative mass for CAS
for i, row in filtered_combined_clean.iterrows():
    N0 = row['dryintercept']
    D = row['D']
    M_total_CAS = total_mass_CAS(N0, D)
    cumulative_masses_CAS = np.array([cumulative_mass_CAS(N0, D, d) for d in diameters]) / M_total_CAS
    cumulative_mass_CAS_all_legs += cumulative_masses_CAS

# Average cumulative mass for CAS
average_cumulative_mass_CAS = cumulative_mass_CAS_all_legs / len(filtered_combined_clean)

# Calculate diameter at 50% cumulative mass for CAS
interpolation_func_CAS = interp1d(average_cumulative_mass_CAS, diameters, kind='linear', fill_value='extrapolate')
diameter_at_50_CAS = interpolation_func_CAS(0.5)

# Plotting
plt.figure(figsize=(12, 8))
plt.plot(diameters, average_cumulative_mass_CDP, label=f'CDP Average Cumulative Mass (50%: {diameter_at_50_CDP:.2f} µm)', color='green', linewidth=2.5)
plt.plot(diameters, average_cumulative_mass_CAS, label=f'CAS Average Cumulative Mass (50%: {diameter_at_50_CAS:.2f} µm)', color='blue', linewidth=2.5)
plt.axvline(diameter_at_50_CDP, color='green', linestyle='--', alpha=0.7)
plt.axvline(diameter_at_50_CAS, color='blue', linestyle='--', alpha=0.7)
plt.xlabel('Dry Diameter (µm)', fontsize=16, fontweight='bold')
plt.ylabel('Cumulative Mass', fontsize=16, fontweight='bold')
plt.title('Below cloud base cumulative mass distributions January-June 2022', fontsize=17, fontweight='bold')
plt.legend(fontsize=14)
plt.grid()
plt.tight_layout()
plt.show()

# Print diameters at 50% mass
print(f"CDP Diameter at 50% Average Cumulative Mass: {diameter_at_50_CDP:.2f} µm")
print(f"CAS Diameter at 50% Average Cumulative Mass: {diameter_at_50_CAS:.2f} µm")

# %%
# Define function to calculate mass
#Computes the average mass distribution for CDP and CAS by averaging the masses across all legs.
#Then, it computes the cumulative mass distribution from the averaged mass distribution.
#The plotted lines represent the cumulative mass calculated from the averaged mass distribution.
def calculate_mass(diameter, concentration):
    return concentration * (np.pi / 6) * (diameter**3)

# Define function to calculate cumulative mass and diameter at 50% mass
def calculate_cumulative_mass(diameters, masses):
    cumulative_mass = np.cumsum(masses)
    total_mass = np.sum(masses)
    normalized_cumulative_mass = cumulative_mass / total_mass

    interpolation_func = interp1d(normalized_cumulative_mass, diameters, kind="linear", fill_value="extrapolate")
    diameter_at_50_percent = interpolation_func(0.5)
    return normalized_cumulative_mass, diameter_at_50_percent

# Define bins for CDP and CAS
common_bins_CDP = np.linspace(2, 25, 30)
common_bins_CAS = np.linspace(2, 25, 18) 

# Process CDP data
all_masses_CDP = []
for _, row in Y_CDP_clean.iterrows():
    concentrations = np.array([row.get(f"Bin{j}_Y_mean", 0) for j in range(0, 30)])
    masses = calculate_mass(common_bins_CDP, concentrations)
    all_masses_CDP.append(masses)
all_masses_CDP = np.array(all_masses_CDP)

average_masses_CDP = np.mean(all_masses_CDP, axis=0)
normalized_cumulative_mass_CDP, diameter_at_50_CDP = calculate_cumulative_mass(common_bins_CDP, average_masses_CDP)

# Process CAS data
all_masses_CAS = []
for _, row in Y_BCB_clean.iterrows():
    concentrations = np.array([row.get(f"Bin{j}_Y_mean", 0) for j in range(12, 30)])
    masses = calculate_mass(common_bins_CAS, concentrations)
    all_masses_CAS.append(masses)
all_masses_CAS = np.array(all_masses_CAS)

average_masses_CAS = np.mean(all_masses_CAS, axis=0)
normalized_cumulative_mass_CAS, diameter_at_50_CAS = calculate_cumulative_mass(common_bins_CAS, average_masses_CAS)

# Plotting the combined distributions
plt.figure(figsize=(12, 8))
plt.plot(common_bins_CDP, normalized_cumulative_mass_CDP, label=f'CDP Average Distribution (50% Mass at {diameter_at_50_CDP:.2f} µm)', color='blue')
plt.plot(common_bins_CAS, normalized_cumulative_mass_CAS, label=f'CAS Average Distribution (50% Mass at {diameter_at_50_CAS:.2f} µm)', color='green')
plt.axvline(diameter_at_50_CDP, color='blue', linestyle='--', alpha=0.7)
plt.axvline(diameter_at_50_CAS, color='green', linestyle='--', alpha=0.7)

# Formatting
plt.xlabel("Diameter (µm)", fontsize=14, fontweight="bold")
plt.ylabel("Cumulative mass", fontsize=14, fontweight="bold")
plt.title("Below cloud base average mass distributions ", fontsize=14, fontweight="bold")
plt.grid()
plt.tight_layout()
plt.legend(fontsize=10, loc="upper left")
plt.show()

# Print the diameters at 50% mass
print(f"CDP Diameter at 50% Mass: {diameter_at_50_CDP:.2f} µm")
print(f"CAS Diameter at 50% Mass: {diameter_at_50_CAS:.2f} µm")
#%%
#When you change the range or resolution of the common_bins 
# (e.g., from 10 to 25 to 50), the average diameter at 50% mass 
# increases because the cumulative mass distribution is sensitive to 
# how the mass is distributed across the bins. Here's why:
#Wider common_bins Range Includes More Mass:
#By increasing the upper limit of common_bins (e.g., from 10 µm to 50 µm), you're considering larger droplets, which typically contribute disproportionately to the mass because of their cubic dependence on diameter in the mass calculation:
#This shifts the cumulative mass distribution toward larger diameters.

#The cumulative mass is calculated as a normalized sum: Including larger bins (e.g., up to 50 µm) shifts the cumulative sum curve toward larger diameters. As a result, the interpolated diameter at 50% cumulative mass (diameter_at_50) increases.
#Mass Concentration in Larger Bins:
#Larger bins may have low concentrations but significant mass due 
# to the cubic dependence on diameter. Even a small concentration of 
# large droplets can disproportionately affect the cumulative mass and 
# shift the 50% point.
#Narrow Range (e.g., 2–10 µm):
#Excludes larger droplets, so the cumulative mass curve primarily reflects smaller bins, resulting in a smaller diameter_at_50.
#Wider Range (e.g., 2–50 µm):
#Includes larger droplets, which dominate the total mass, 
# shifting the cumulative mass curve and the 50% point to larger diameters.
# %%
# CAS Data
#Combined CDP and CAS slope vs dry int plots 
slope_data_CAS = combined_data['D']
dry_intercept_data_CAS = [entry['dry intercept'] for entry in filtered_master_BCB_dryintercept]

# Truncate CAS data if lengths don't match
min_len_CAS = min(len(slope_data_CAS), len(dry_intercept_data_CAS))
slope_data_CAS = slope_data_CAS[:min_len_CAS]
dry_intercept_data_CAS = dry_intercept_data_CAS[:min_len_CAS]

# CDP Data
slope_data_CDP = combined_data_CDP['D']
dry_intercept_data_CDP = [entry['dry intercept'] for entry in filtered_master_BCB_dryintercept_CDP]

# Truncate CDP data if lengths don't match
min_len_CDP = min(len(slope_data_CDP), len(dry_intercept_data_CDP))
slope_data_CDP = slope_data_CDP[:min_len_CDP]
dry_intercept_data_CDP = dry_intercept_data_CDP[:min_len_CDP]

# Create the scatter plot
plt.figure(figsize=(10, 8))

# Plot CAS
plt.scatter(slope_data_CAS, dry_intercept_data_CAS, color='blue', s=80, alpha=0.7, label='CAS')

# Plot CDP
plt.scatter(slope_data_CDP, dry_intercept_data_CDP, color='green', s=80, alpha=0.7, label='CDP')

# Add axis labels and title
plt.xlabel('Slope', fontsize=19, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=19, fontweight='bold')
plt.title('Below Cloud Base January - June 2022', fontsize=19, fontweight='bold')

# Set logarithmic scales for both axes
plt.xscale('log')
plt.yscale('log')

# Set axis limits for the combined plot
plt.xlim(10**-0.2, 10**1.05)
plt.ylim(10**-2.35, 10**1.75)

# Add legend
plt.legend(fontsize=16)

# Finalize the plot
plt.tight_layout()
plt.show()

#%%
# Define function to calculate mass for both instruments
def calculate_mass(N0, D):
    integrand = lambda d: np.exp(-d / D) * d**3
    mass_integral, _ = quad(integrand, 2, np.inf)  # Integrate from 2 µm to infinity
    return N0 * mass_integral

# Extended grids for mass contours
x_min, x_max = 10**-0.2, 10**0.9
y_min, y_max = 10**-1.9, 10**1.7
xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200)
ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)
D_grid_extended, dryintercept_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)

# CDP mass grid
mass_grid_CDP = np.zeros_like(D_grid_extended)
for i in range(D_grid_extended.shape[0]):
    for j in range(D_grid_extended.shape[1]):
        mass_grid_CDP[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j])

# CAS mass grid
mass_grid_CAS = np.zeros_like(D_grid_extended)
for i in range(D_grid_extended.shape[0]):
    for j in range(D_grid_extended.shape[1]):
        mass_grid_CAS[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j])

# Define mass contour levels
mass_levels = np.logspace(-2, 5, 50)

# Plotting points with mass contours
plt.figure(figsize=(10, 8))

# CDP points and mass contours
plt.scatter(filtered_combined_clean_CDP['D'], filtered_combined_clean_CDP['dryintercept'], 
            color='green', s=100, label='CDP Points')
contour_CDP = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_CDP, 
                          levels=mass_levels, colors='blue', alpha=0.75)
plt.clabel(contour_CDP, inline=True, fontsize=10, fmt='%1.1e', use_clabeltext=True, colors='blue')

# CAS points and mass contours
plt.scatter(filtered_combined_clean['D'], filtered_combined_clean['dryintercept'], 
            color='blue', s=100, label='CAS Points')
contour_CAS = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_CAS, 
                          levels=mass_levels, colors='green', alpha=0.75)
plt.clabel(contour_CAS, inline=True, fontsize=10, fmt='%1.1e', use_clabeltext=True, colors='green')

# Axis labels, title, and legend
plt.xlabel('Slope', fontsize=19, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=19, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.title('Below Cloud Base January - June 2022 (Points and Contours)', fontsize=19, fontweight='bold')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()
#%%
# Define function to calculate mass
def calculate_mass(N0, D):
    integrand = lambda d: np.exp(-d / D) * d**3
    mass_integral, _ = quad(integrand, 2, np.inf)  # Integrate from 2 µm to infinity
    return N0 * mass_integral

# Define extended grids for mass contours
x_min, x_max = 10**-0.2, 10**1.05
y_min, y_max = 10**-1.77, 10**1.9
xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200)
ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)
D_grid_extended, dryintercept_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)

# Calculate CDP mass grid
mass_grid_CDP = np.zeros_like(D_grid_extended)
for i in range(D_grid_extended.shape[0]):
    for j in range(D_grid_extended.shape[1]):
        mass_grid_CDP[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j])

# Calculate CAS mass grid
mass_grid_CAS = np.zeros_like(D_grid_extended)
for i in range(D_grid_extended.shape[0]):
    for j in range(D_grid_extended.shape[1]):
        mass_grid_CAS[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j])

# Define mass contour levels
mass_levels = np.logspace(-2, 5, 50)

# Plotting points with mass contours
plt.figure(figsize=(10, 8))

# CAS points and mass contours
plt.scatter(filtered_combined_clean['D'], filtered_combined_clean['dryintercept'], 
            color='blue', s=100, label='CAS Points', alpha=0.9)
contour_CAS = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_CAS, 
                          levels=mass_levels, colors='green', alpha=0.75)
plt.clabel(contour_CAS, inline=True, fontsize=10, fmt='%1.1e', use_clabeltext=True, colors='green')

# CDP points and mass contours
plt.scatter(filtered_combined_clean_CDP['D'], filtered_combined_clean_CDP['dryintercept'], 
            color='green', s=100, label='CDP Points', alpha=0.9)
contour_CDP = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_CDP, 
                          levels=mass_levels, colors='blue', alpha=0.75)
plt.clabel(contour_CDP, inline=True, fontsize=10, fmt='%1.1e', use_clabeltext=True, colors='blue')

# Axis labels, title, and legend
plt.xlabel('Slope', fontsize=19, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=19, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.title('Below Cloud Base January - June 2022', fontsize=19, fontweight='bold')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()
#%%
# Define function to calculate mass for both instruments
# def calculate_mass(N0, D):
#     integrand = lambda d: np.exp(-d / D) * d**3
#     mass_integral, _ = quad(integrand, 2, np.inf)  # Integrate from 2 µm to infinity
#     return N0 * mass_integral

# # Extended grids for mass contours
# x_min, x_max = 10**-0.2, 10**0.9
# y_min, y_max = 10**-1.9, 10**1.7
# xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200)
# ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)
# D_grid_extended, dryintercept_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)

# # CDP mass grid
# mass_grid_CDP = np.zeros_like(D_grid_extended)
# for i in range(D_grid_extended.shape[0]):
#     for j in range(D_grid_extended.shape[1]):
#         mass_grid_CDP[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j])

# # CAS mass grid
# mass_grid_CAS = np.zeros_like(D_grid_extended)
# for i in range(D_grid_extended.shape[0]):
#     for j in range(D_grid_extended.shape[1]):
#         mass_grid_CAS[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j])

# # Define mass contour levels
# mass_levels = np.logspace(-2, 5, 50)

# # Add jitter to CDP and CAS points
# filtered_combined_clean_CDP['D_jitter'] = filtered_combined_clean_CDP['D'] + np.random.normal(0, 0.01, len(filtered_combined_clean_CDP))
# filtered_combined_clean_CDP['dryintercept_jitter'] = filtered_combined_clean_CDP['dryintercept'] + np.random.normal(0, 0.01, len(filtered_combined_clean_CDP))

# filtered_combined_clean['D_jitter'] = filtered_combined_clean['D'] + np.random.normal(0, 0.01, len(filtered_combined_clean))
# filtered_combined_clean['dryintercept_jitter'] = filtered_combined_clean['dryintercept'] + np.random.normal(0, 0.01, len(filtered_combined_clean))

# # Plotting points with mass contours
# plt.figure(figsize=(10, 8))

# # CDP points and mass contours
# plt.scatter(filtered_combined_clean_CDP['D_jitter'], filtered_combined_clean_CDP['dryintercept_jitter'], 
#             color='green', s=100, label='CDP Points')
# contour_CDP = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_CDP, 
#                           levels=mass_levels, colors='blue', alpha=0.75)
# plt.clabel(contour_CDP, inline=True, fontsize=10, fmt='%1.1e', use_clabeltext=True, colors='blue')

# # CAS points and mass contours
# plt.scatter(filtered_combined_clean['D_jitter'], filtered_combined_clean['dryintercept_jitter'], 
#             color='blue', s=100, label='CAS Points')
# contour_CAS = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_CAS, 
#                           levels=mass_levels, colors='green', alpha=0.75)
# plt.clabel(contour_CAS, inline=True, fontsize=10, fmt='%1.1e', use_clabeltext=True, colors='green')

# # Axis labels, title, and legend
# plt.xlabel('Slope', fontsize=19, fontweight='bold')
# plt.ylabel('Dry Intercept', fontsize=19, fontweight='bold')
# plt.yscale('log')
# plt.xscale('log')
# plt.title('Below Cloud Base January - June 2022 (Points and Contours)', fontsize=19, fontweight='bold')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(fontsize=16, fontweight='bold')
# plt.yticks(fontsize=16, fontweight='bold')
# plt.legend(fontsize=14)
# plt.tight_layout()
# plt.show()
#%%
#Showing mass for both instruments 

# Function to calculate mass
def calculate_mass(N0, D):
    integrand = lambda d: np.exp(-d / D) * d**3
    mass_integral, _ = quad(integrand, 2, np.inf)  # Integrate from 2 µm to infinity
    return N0 * mass_integral

# Extended grids for mass contours
x_min, x_max = 10**-0.2, 10**0.9
y_min, y_max = 10**-2.3, 10**1.5
xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200)
ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)
D_grid_extended, dryintercept_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)

# CDP mass grid
mass_grid_CDP = np.zeros_like(D_grid_extended)
for i in range(D_grid_extended.shape[0]):
    for j in range(D_grid_extended.shape[1]):
        mass_grid_CDP[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j])

# CAS mass grid
mass_grid_CAS = np.zeros_like(D_grid_extended)
for i in range(D_grid_extended.shape[0]):
    for j in range(D_grid_extended.shape[1]):
        mass_grid_CAS[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j])

# Define mass contour levels
mass_levels = np.logspace(-2, 5, 30)  # Reduced to 30 levels for clarity

# Extract slope and dry intercept data for CAS
slope_data_CAS = combined_data['D']
dry_intercept_data_CAS = [entry['dry intercept'] for entry in filtered_master_BCB_dryintercept]

# Ensure the lengths of slope and dry intercept match for CAS
min_len_CAS = min(len(slope_data_CAS), len(dry_intercept_data_CAS))
slope_data_CAS = slope_data_CAS[:min_len_CAS]
dry_intercept_data_CAS = dry_intercept_data_CAS[:min_len_CAS]

# Extract slope and dry intercept data for CDP
slope_data_CDP = combined_data_CDP['D']
dry_intercept_data_CDP = [entry['dry intercept'] for entry in filtered_master_BCB_dryintercept_CDP]

# Ensure the lengths of slope and dry intercept match for CDP
min_len_CDP = min(len(slope_data_CDP), len(dry_intercept_data_CDP))
slope_data_CDP = slope_data_CDP[:min_len_CDP]
dry_intercept_data_CDP = dry_intercept_data_CDP[:min_len_CDP]

# Plotting points with mass contours
plt.figure(figsize=(12, 10))

# CDP points and mass contours
plt.scatter(slope_data_CDP, dry_intercept_data_CDP, 
            color='green', s=100, label='CDP Points')
contour_CDP = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_CDP, 
                          levels=mass_levels, colors='blue', alpha=0.75)
plt.clabel(contour_CDP, inline=True, fontsize=10, fmt='%1.1e', use_clabeltext=True, colors='blue')

# CAS points and mass contours
plt.scatter(slope_data_CAS, dry_intercept_data_CAS, 
            color='blue', s=100, label='CAS Points')
contour_CAS = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_CAS, 
                          levels=mass_levels, colors='green', alpha=0.75)
plt.clabel(contour_CAS, inline=True, fontsize=10, fmt='%1.1e', use_clabeltext=True, colors='green')

# Axis labels, title, and legend
plt.xlabel('Slope', fontsize=19, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=19, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.title('Below Cloud Base January - June 2022', fontsize=19, fontweight='bold')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()


#%%

# Define the size distribution function
def size_distribution(ddry, N0, D):
    return N0 * np.exp(-ddry / D)

# Define common bins for the distributions
common_bins = np.linspace(2.5, 10, 10)

# Windspeed bins (same for both instruments)
windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]

# Initialize grouped data for CDP
grouped_concentrations_CDP = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds_CDP = {i: [] for i in range(len(windspeed_bins))}

# Initialize grouped data for CAS
grouped_concentrations_CAS = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds_CAS = {i: [] for i in range(len(windspeed_bins))}

# **Process CDP data**
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

    windspeed_entry = df_combined_CDP[
        (df_combined_CDP['Date'] == date) &
        (df_combined_CDP['BCB_start'] == BCB_start) &
        (df_combined_CDP['BCB_stop'] == BCB_stop)
    ]

    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]

        # Compute size distribution
        size_dist = size_distribution(ddry_values, dryint, D)
        interp_func = np.interp(common_bins, ddry_values, size_dist)
        interpolated_leg_values = interp_func

        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_concentrations_CDP[idx].append(interpolated_leg_values)
                mean_windspeeds_CDP[idx].append(windspeed)
                break

# **Process CAS data**
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]

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

        # Compute size distribution
        size_dist = size_distribution(ddry_values, N0, D)
        interp_func = np.interp(common_bins, ddry_values, size_dist)
        interpolated_leg_values = interp_func

        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_concentrations_CAS[idx].append(interpolated_leg_values)
                mean_windspeeds_CAS[idx].append(windspeed)
                break

# **Plot the combined data**
plt.figure(figsize=(12, 8))

# Plot CDP lines (Blue)
for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_concentrations_CDP[idx]:  # Check if CDP data exists for this bin
        concentrations_array = np.array(grouped_concentrations_CDP[idx])
        avg_concentration = np.mean(concentrations_array, axis=0)
        std_dev = np.std(concentrations_array, axis=0)

        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)  # Avoid log-scale errors
        avg_windspeed = np.mean(mean_windspeeds_CDP[idx])
        num_legs = len(grouped_concentrations_CDP[idx])

        # Plot CDP line
        plt.errorbar(common_bins, avg_concentration, yerr=std_dev, fmt='o', capsize=5,
                     label=f"CDP: {avg_windspeed:.1f} m/s, n={num_legs} legs", linestyle='-', color='blue', alpha=0.9)

# Plot CAS lines (Green)
for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_concentrations_CAS[idx]:  # Check if CAS data exists for this bin
        concentrations_array = np.array(grouped_concentrations_CAS[idx])
        avg_concentration = np.mean(concentrations_array, axis=0)
        std_dev = np.std(concentrations_array, axis=0)

        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)  # Avoid log-scale errors
        avg_windspeed = np.mean(mean_windspeeds_CAS[idx])
        num_legs = len(grouped_concentrations_CAS[idx])

        # Plot CAS line
        plt.errorbar(common_bins, avg_concentration, yerr=std_dev, fmt='o', capsize=5,
                     label=f"CAS: {avg_windspeed:.1f} m/s, n={num_legs} legs", linestyle='--', color='green', alpha=0.9)

# Formatting the Plot
plt.yscale('log')
plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Below Cloud Base January-June 2022', fontweight='bold')
plt.legend(title="Instrument and Average Windspeed (m/s)", fontsize=10)
plt.tight_layout()
plt.show()
# %%

print("CDP Average Concentrations and Standard Deviations by Bin:")
for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_concentrations_CDP[idx]:
        concentrations_array = np.array(grouped_concentrations_CDP[idx])
        if np.isnan(concentrations_array).all():
            print(f"CDP Bin {idx} is empty or all NaN.")
            continue
        avg_concentration = np.nanmean(concentrations_array, axis=0)
        std_dev = np.nanstd(concentrations_array, axis=0)
        print(f"  CDP Bin {idx}: Avg Concentration = {avg_concentration}, Std Dev = {std_dev}")
    else:
        print(f"  CDP Bin {idx} has no data.")

print("\nCAS Average Concentrations and Standard Deviations by Bin:")
for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_concentrations_CAS[idx]:
        concentrations_array = np.array(grouped_concentrations_CAS[idx])
        if np.isnan(concentrations_array).all():
            print(f"CAS Bin {idx} is empty or all NaN.")
            continue
        avg_concentration = np.nanmean(concentrations_array, axis=0)
        std_dev = np.nanstd(concentrations_array, axis=0)
        print(f"  CAS Bin {idx}: Avg Concentration = {avg_concentration}, Std Dev = {std_dev}")
    else:
        print(f"  CAS Bin {idx} has no data.")

# Plot Droplet Concentrations with Offsets
offset = 0.1  # Small horizontal offset for CAS

plt.figure(figsize=(12, 8))

# CDP Plot (Blue)
for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_concentrations_CDP[idx]:
        concentrations_array = np.array(grouped_concentrations_CDP[idx])
        avg_concentration = np.nanmean(concentrations_array, axis=0)
        std_dev = np.nanstd(concentrations_array, axis=0)
        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)
        avg_windspeed = np.mean(mean_windspeeds_CDP[idx])
        num_legs = len(grouped_concentrations_CDP[idx])
        plt.errorbar(common_bins, avg_concentration, yerr=std_dev, fmt='o', capsize=5,
                     label=f"CDP: {avg_windspeed:.1f} m/s, n={num_legs} legs", linestyle='-', color='blue', alpha=0.9)

# CAS Plot (Green, with Offset)
for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_concentrations_CAS[idx]:
        concentrations_array = np.array(grouped_concentrations_CAS[idx])
        avg_concentration = np.nanmean(concentrations_array, axis=0)
        std_dev = np.nanstd(concentrations_array, axis=0)
        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)
        avg_windspeed = np.mean(mean_windspeeds_CAS[idx])
        num_legs = len(grouped_concentrations_CAS[idx])
        plt.errorbar(common_bins + offset, avg_concentration, yerr=std_dev, fmt='o', capsize=5,
                     label=f"CAS: {avg_windspeed:.1f} m/s, n={num_legs} legs", linestyle='--', color='green', alpha=0.9)

plt.yscale('log')
plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold', fontsize=19)
plt.xlabel('Bin diameter (µm)', fontweight='bold', fontsize=19)
plt.title('Below Cloud Base January-June 2022', fontweight='bold', fontsize=19)
plt.legend(title="Instrument and Average Windspeed (m/s)", fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=16, width=3, length=8)  # Major ticks
plt.tick_params(axis='both', which='minor', labelsize=14, width=2, length=5)  # Minor ticks

plt.tight_layout()
plt.show()
# %%
print("CDP Average Concentrations and Standard Deviations by Bin:")
for idx, (low, high) in enumerate(windspeed_bins):
    if len(grouped_concentrations_CDP[idx]) > 0:  # Explicitly check if the list has elements
        concentrations_array = np.array(grouped_concentrations_CDP[idx])
        if np.isnan(concentrations_array).all():
            print(f"CDP Bin {idx} is empty or all NaN.")
            continue
        avg_concentration = np.nanmean(concentrations_array, axis=0)
        std_dev = np.nanstd(concentrations_array, axis=0)
        print(f"  CDP Bin {idx}: Avg Concentration = {avg_concentration}, Std Dev = {std_dev}")
    else:
        print(f"  CDP Bin {idx} has no data.")

print("\nCAS Average Concentrations and Standard Deviations by Bin:")
for idx, (low, high) in enumerate(windspeed_bins):
    if len(grouped_concentrations_CAS[idx]) > 0:  # Explicitly check if the list has elements
        concentrations_array = np.array(grouped_concentrations_CAS[idx])
        if np.isnan(concentrations_array).all():
            print(f"CAS Bin {idx} is empty or all NaN.")
            continue
        avg_concentration = np.nanmean(concentrations_array, axis=0)
        std_dev = np.nanstd(concentrations_array, axis=0)
        print(f"  CAS Bin {idx}: Avg Concentration = {avg_concentration}, Std Dev = {std_dev}")
    else:
        print(f"  CAS Bin {idx} has no data.")

# Plot Droplet Concentrations with Offsets
offset = 0.1  # Small horizontal offset for CAS

plt.figure(figsize=(12, 8))

# CDP Plot (Blue)
for idx, (low, high) in enumerate(windspeed_bins):
    if len(grouped_concentrations_CDP[idx]) > 0:  # Explicitly check if the list has elements
        concentrations_array = np.array(grouped_concentrations_CDP[idx])
        avg_concentration = np.nanmean(concentrations_array, axis=0)
        std_dev = np.nanstd(concentrations_array, axis=0)
        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)
        avg_windspeed = np.mean(mean_windspeeds_CDP[idx])
        num_legs = len(grouped_concentrations_CDP[idx])
        plt.errorbar(common_bins, avg_concentration, yerr=std_dev, fmt='o', capsize=5,
                     label=f"CDP: {avg_windspeed:.1f} m/s, n={num_legs} legs", linestyle='-', color='blue', alpha=0.9)

# CAS Plot (Green, with Offset)
for idx, (low, high) in enumerate(windspeed_bins):
    if len(grouped_concentrations_CAS[idx]) > 0:  # Explicitly check if the list has elements
        concentrations_array = np.array(grouped_concentrations_CAS[idx])
        avg_concentration = np.nanmean(concentrations_array, axis=0)
        std_dev = np.nanstd(concentrations_array, axis=0)
        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)
        avg_windspeed = np.mean(mean_windspeeds_CAS[idx])
        num_legs = len(grouped_concentrations_CAS[idx])
        plt.errorbar(common_bins + offset, avg_concentration, yerr=std_dev, fmt='o', capsize=5,
                     label=f"CAS: {avg_windspeed:.1f} m/s, n={num_legs} legs", linestyle='--', color='green', alpha=0.9)

plt.yscale('log')
plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold', fontsize=19)
plt.xlabel('Bin diameter (µm)', fontweight='bold', fontsize=19)
plt.title('Below Cloud Base January-June 2022', fontweight='bold', fontsize=19)
plt.legend(title="Instrument and Average Windspeed (m/s)", fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=16, width=3, length=8)  # Major ticks
plt.tick_params(axis='both', which='minor', labelsize=14, width=2, length=5)  # Minor ticks

plt.tight_layout()
plt.show()
#%%
# Define unique colors for each windspeed bin
bin_colors = ['red', 'blue', 'green', 'purple']

plt.figure(figsize=(12, 8))

# CDP Plot (Solid Lines)
for idx, (low, high) in enumerate(windspeed_bins):
    if len(grouped_concentrations_CDP[idx]) > 0:  # Check if the list has elements
        concentrations_array = np.array(grouped_concentrations_CDP[idx])
        avg_concentration = np.nanmean(concentrations_array, axis=0)
        std_dev = np.nanstd(concentrations_array, axis=0)
        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)
        avg_windspeed = np.mean(mean_windspeeds_CDP[idx])
        num_legs = len(grouped_concentrations_CDP[idx])
        plt.errorbar(
            common_bins, avg_concentration, yerr=std_dev, capsize=5,
            label=f"CDP: {avg_windspeed:.1f} m/s, n={num_legs} legs",
            color=bin_colors[idx], linestyle='-', alpha=0.9
        )

# CAS Plot (Dashed Lines)
offset = 0.1  # Small horizontal offset for CAS
for idx, (low, high) in enumerate(windspeed_bins):
    if len(grouped_concentrations_CAS[idx]) > 0:  # Check if the list has elements
        concentrations_array = np.array(grouped_concentrations_CAS[idx])
        avg_concentration = np.nanmean(concentrations_array, axis=0)
        std_dev = np.nanstd(concentrations_array, axis=0)
        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)
        avg_windspeed = np.mean(mean_windspeeds_CAS[idx])
        num_legs = len(grouped_concentrations_CAS[idx])
        plt.errorbar(
            common_bins + offset, avg_concentration, yerr=std_dev, capsize=5,
            label=f"CAS: {avg_windspeed:.1f} m/s, n={num_legs} legs",
            color=bin_colors[idx], linestyle='--', alpha=0.9
        )

plt.yscale('log')
plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold', fontsize=19)
plt.xlabel('Bin diameter (µm)', fontweight='bold', fontsize=19)
plt.title('Below Cloud Base January-June 2022', fontweight='bold', fontsize=19)
plt.legend(title="Instrument and Average Windspeed (m/s)", fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=16, width=3, length=8)  # Major ticks
plt.tick_params(axis='both', which='minor', labelsize=14, width=2, length=5)  # Minor ticks

plt.tight_layout()
plt.show()

#%%
# Define unique colors for each windspeed bin
bin_colors = ['red', 'blue', 'green', 'purple']

plt.figure(figsize=(12, 8))

# CDP Plot (Solid Lines)
for idx, (low, high) in enumerate(windspeed_bins):
    if len(grouped_concentrations_CDP[idx]) > 0:  # Check if the list has elements
        concentrations_array = np.array(grouped_concentrations_CDP[idx])
        avg_concentration = np.nanmean(concentrations_array, axis=0)
        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)
        avg_windspeed = np.mean(mean_windspeeds_CDP[idx])
        num_legs = len(grouped_concentrations_CDP[idx])
        plt.plot(
            common_bins, avg_concentration,
            label=f"CDP: {avg_windspeed:.1f} m/s, n={num_legs} legs",
            color=bin_colors[idx], linestyle='-', alpha=0.9
        )

# CAS Plot (Dashed Lines)
offset = 0.1  # Small horizontal offset for CAS
for idx, (low, high) in enumerate(windspeed_bins):
    if len(grouped_concentrations_CAS[idx]) > 0:  # Check if the list has elements
        concentrations_array = np.array(grouped_concentrations_CAS[idx])
        avg_concentration = np.nanmean(concentrations_array, axis=0)
        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)
        avg_windspeed = np.mean(mean_windspeeds_CAS[idx])
        num_legs = len(grouped_concentrations_CAS[idx])
        plt.plot(
            common_bins + offset, avg_concentration,
            label=f"CAS: {avg_windspeed:.1f} m/s, n={num_legs} legs",
            color=bin_colors[idx], linestyle='--', alpha=0.9
        )

plt.yscale('log')
plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold', fontsize=19)
plt.xlabel('Bin diameter (µm)', fontweight='bold', fontsize=19)
plt.title('Below Cloud Base January-June 2022', fontweight='bold', fontsize=19)
plt.legend(title="Instrument and Average Windspeed (m/s)", fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=16, width=3, length=8)  # Major ticks
plt.tick_params(axis='both', which='minor', labelsize=14, width=2, length=5)  # Minor ticks

plt.tight_layout()
plt.show()
#%%
# Define unique colors for each windspeed bin
bin_colors = ['red', 'blue', 'green', 'purple']

plt.figure(figsize=(12, 8))

# Ensure matching legs between CDP and CAS
for idx in range(len(windspeed_bins)):
    num_legs_CDP = len(grouped_concentrations_CDP[idx])
    num_legs_CAS = len(grouped_concentrations_CAS[idx])
    min_legs = min(num_legs_CDP, num_legs_CAS)  # Match the number of legs between the two instruments

    # Truncate to match legs
    grouped_concentrations_CDP[idx] = grouped_concentrations_CDP[idx][:min_legs]
    mean_windspeeds_CDP[idx] = mean_windspeeds_CDP[idx][:min_legs]
    grouped_concentrations_CAS[idx] = grouped_concentrations_CAS[idx][:min_legs]
    mean_windspeeds_CAS[idx] = mean_windspeeds_CAS[idx][:min_legs]

# CDP Plot (Solid Lines)
for idx, (low, high) in enumerate(windspeed_bins):
    if len(grouped_concentrations_CDP[idx]) > 0:  # Check if the list has elements
        concentrations_array = np.array(grouped_concentrations_CDP[idx])
        avg_concentration = np.nanmean(concentrations_array, axis=0)
        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)
        avg_windspeed = np.mean(mean_windspeeds_CDP[idx])
        num_legs = len(grouped_concentrations_CDP[idx])
        plt.plot(
            common_bins, avg_concentration,
            label=f"CDP: {avg_windspeed:.1f} m/s, n={num_legs} legs",
            color=bin_colors[idx], linestyle='-', alpha=0.9
        )

# CAS Plot (Dashed Lines)
offset = 0.1  # Small horizontal offset for CAS
for idx, (low, high) in enumerate(windspeed_bins):
    if len(grouped_concentrations_CAS[idx]) > 0:  # Check if the list has elements
        concentrations_array = np.array(grouped_concentrations_CAS[idx])
        avg_concentration = np.nanmean(concentrations_array, axis=0)
        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)
        avg_windspeed = np.mean(mean_windspeeds_CAS[idx])
        num_legs = len(grouped_concentrations_CAS[idx])
        plt.plot(
            common_bins + offset, avg_concentration,
            label=f"CAS: {avg_windspeed:.1f} m/s, n={num_legs} legs",
            color=bin_colors[idx], linestyle='--', alpha=0.9
        )

plt.yscale('log')
plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold', fontsize=19)
plt.xlabel('Bin diameter (µm)', fontweight='bold', fontsize=19)
plt.title('Below Cloud Base January-June 2022', fontweight='bold', fontsize=19)
plt.legend(title="Instrument and Average Windspeed (m/s)", fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=16, width=3, length=8)  # Major ticks
plt.tick_params(axis='both', which='minor', labelsize=14, width=2, length=5)  # Minor ticks

plt.tight_layout()
plt.show()

#%%

# %%
#Combined PDF for CDP and CAS
plt.figure(figsize=(10, 8))
sns.kdeplot(mass_data, fill=True, bw_adjust=0.5, color='blue', label='CAS', alpha=0.5)
sns.kdeplot(mass_data_CDP, fill=True, bw_adjust=0.5, color='green', label='CDP', alpha=0.5)
plt.title("Below cloud base January-June 2022", fontsize=16, fontweight='bold')
plt.xlabel("Mass", fontsize=14, fontweight='bold')
plt.ylabel("Probability Density", fontsize=14, fontweight='bold')
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(10, 8))

# CAS KDE
sns.kdeplot(
    mass_data, 
    fill=True, 
    bw_adjust=0.5, 
    clip=(0, None),  # Clip to avoid values below 0
    color='blue', 
    label='CAS', 
    alpha=0.5
)

# CDP KDE
sns.kdeplot(
    mass_data_CDP, 
    fill=True, 
    bw_adjust=0.5, 
    clip=(0, None),  # Clip to avoid values below 0
    color='green', 
    label='CDP', 
    alpha=0.5
)

# Add title, labels, and grid
plt.title("Below Cloud Base January-June 2022", fontsize=16, fontweight='bold')
plt.xlabel("Mass", fontsize=14, fontweight='bold')
plt.ylabel("Probability Density", fontsize=14, fontweight='bold')
plt.grid(True)

# Add legend
plt.legend(fontsize=12)

# Tight layout
plt.tight_layout()

# Show plot
plt.show()



# %%
plt.figure(figsize=(10, 8))

# Plot CAS KDE
sns.kdeplot(
    mass_data, 
    fill=True, 
    bw_adjust=0.5, 
    color='blue', 
    label='CAS', 
    linewidth=2,  # Thicker line for clarity
    alpha=0.5
)

# Plot CDP KDE
sns.kdeplot(
    mass_data_CDP, 
    fill=True, 
    bw_adjust=0.5, 
    color='green', 
    label='CDP', 
    linewidth=2,  # Thicker line for clarity
    alpha=0.5
)

# Focus on the main range for better visualization
plt.xlim(0, 500)  # Adjust to focus on the main region of density

# Add title, labels, and legend
plt.title("Below Cloud Base January-June 2022", fontsize=16, fontweight='bold')
plt.xlabel("Mass", fontsize=14, fontweight='bold')
plt.ylabel("Probability Density", fontsize=14, fontweight='bold')
plt.grid(True)
plt.legend(fontsize=12)

# Adjust layout and show
plt.tight_layout()
plt.show()


# %%
