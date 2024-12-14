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
#%%
#%%
#Trying other version of 2DS import to compare 
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

from collections import Counter
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

# %%
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


# %%

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
##but before we do this we need to solve for average N+ intercept for every flight leg
##We know that N+ = No(gRH) where No is the intercept of our ambient distribution

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
            
            # Calculate Ntd
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

#Obtain the total concentration of droplets with dry diameter larger than ddrymin

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
# %%
## Now we can ask how total concentration of droplets with dry diameter larger than ddrymin (ntd)
#  is related to the ambient concentration (NT)

## ntd / nt = e(-grh* drymin + dmin / D*) where dmin is 2.5 for the CAS data
##this is equation 8 on Rob's sheet

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
# %%

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

#Filtered visualization of relationships between D, Dry Intercept, and Windspeed

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

for i, flight in enumerate(master_BCB):
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
                gRh_mean = filtered_master_BCB_gRH[i][j]['gRh_mean'][0]  # Assume matching index

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

correlation_matrix = df_combined[['dryintercept', 'D', 'Windspeed']].corr()


sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Below Cloud Base w/clear sky restrictions January - June 2022')
plt.show()

sns.pairplot(df_combined[['dryintercept', 'D', 'Windspeed']], diag_kind='kde', palette='viridis')

plt.suptitle('Below Cloud Base w/clear sky restrictions January - June 2022', y=1.02)
plt.yscale('log')
plt.xscale('log')
plt.show()


sc = plt.scatter(df_with_windspeed['D'], df_with_windspeed['dryintercept'], 
                 c=df_with_windspeed['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

plt.scatter(df_nan_windspeed['D'], df_nan_windspeed['dryintercept'], 
            color='grey', s=100, label='Windspeed NaN')

plt.colorbar(sc, label='Corrected Windspeed (m/s)')

plt.xlabel('D', fontsize=14, fontweight='bold')
plt.ylabel('dryintercept', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.title('Below cloud base January - June 2022', fontsize=14, fontweight='bold')
plt.xscale('log')
plt.show()
#%%
## Another way to look at GCCN relationships is to visualize their phase space for slope (D) and intercept(No)


#%%
#PLotting density contours for dry intercept and slope with corrected windspeeds
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

for i, flight in enumerate(master_BCB):
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
                gRh_mean = filtered_master_BCB_gRH[i][j]['gRh_mean'][0] 

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
plt.contour(X, Y, Z, levels=contour_levels, colors='red', alpha=0.75)

# Add color bar
plt.colorbar(sc, label='Corrected Windspeed (m/s)')

# Add labels and title
plt.xlabel('D', fontsize=14, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.title('Below Cloud Base January - June 2022 Density Contours' , fontweight='bold')
plt.xscale('log')
plt.xlim(10**-2, 10**2)
plt.ylim(10**-2, 10**2.5)
plt.show()
#%%

# Full integrated mass calculation: M = N0 * D^4
#Plotting mass contours for dry intercept and slope with corrected windspeeds
def calculate_mass(N0, D):
    return N0 * D**4
filtered_combined_clean = filtered_combined[(filtered_combined['D'] > 0) & 
                                            (filtered_combined['dryintercept'] > 0)].copy()


filtered_combined_clean['Mass'] = calculate_mass(filtered_combined_clean['dryintercept'], 
                                                 filtered_combined_clean['D'])


print(f"Min mass: {filtered_combined_clean['Mass'].min()}, Max mass: {filtered_combined_clean['Mass'].max()}")


xgrid = np.logspace(np.log10(filtered_combined_clean['D'].min()), 
                    np.log10(filtered_combined_clean['D'].max()), 100)
ygrid = np.logspace(np.log10(filtered_combined_clean['dryintercept'].min()), 
                    np.log10(filtered_combined_clean['dryintercept'].max()), 100)
D_grid, dryintercept_grid = np.meshgrid(xgrid, ygrid)

mass_grid = calculate_mass(dryintercept_grid, D_grid)

print(f"Mass grid: Min = {np.nanmin(mass_grid)}, Max = {np.nanmax(mass_grid)}")


mass_levels = np.logspace(np.log10(filtered_combined_clean['Mass'].min()), 
                          np.log10(filtered_combined_clean['Mass'].max()), 20)
print(f"Mass levels: {mass_levels}")


sc = plt.scatter(filtered_combined_clean['D'], filtered_combined_clean['dryintercept'], 
                 c=filtered_combined_clean['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')


if not df_nan_windspeed.empty:
    plt.scatter(df_nan_windspeed['D'], df_nan_windspeed['dryintercept'], 
                color='grey', s=100, label='Windspeed NaN')


contour_plot = plt.contour(D_grid, dryintercept_grid, mass_grid, levels=mass_levels, colors='red', alpha=0.75)
if len(contour_plot.allsegs[0]) == 0:
    print("No contours were created. Check your data range or mass grid calculation.")

plt.colorbar(sc, label='Corrected Windspeed (m/s)')
plt.xlabel('D', fontsize=14, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.title('Below Cloud Base January - June 2022', fontsize=14, fontweight='bold')
plt.xlim(10**-2, 10**2)
plt.ylim(10**-2, 10**2.5)
plt.show()
#%%
#adjusting integral to 2 to infinity and changing mass contour number and appearance 


# Function to calculate mass with lower limit of integration at 2
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

mass_levels = np.logspace(np.log10(filtered_combined_clean['Mass'].min()), 
                          np.log10(filtered_combined_clean['Mass'].max()), 10)
print(f"Mass levels: {mass_levels}")

sc = plt.scatter(filtered_combined_clean['D'], filtered_combined_clean['dryintercept'], 
                 c=filtered_combined_clean['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

if not df_nan_windspeed.empty:
    plt.scatter(df_nan_windspeed['D'], df_nan_windspeed['dryintercept'], 
                color='grey', s=100, label='Windspeed NaN')


contour_plot = plt.contour(D_grid, dryintercept_grid, mass_grid, levels=mass_levels, 
                           colors='red', linewidths=1, alpha=0.75)
if len(contour_plot.allsegs[0]) == 0:
    print("No contours were created. Check your data range or mass grid calculation.")

plt.colorbar(sc, label='Corrected Windspeed (m/s)')
plt.xlabel('D', fontsize=14, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.title('Below Cloud Base January - June 2022', fontsize=14, fontweight='bold')
plt.xlim(10**-2, 10**2)
plt.ylim(10**-2, 10**2.5)
plt.show()
#%%
# Function to calculate mass with lower limit of integration at 2
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

plt.colorbar(sc, label='Corrected Windspeed (m/s)')
plt.xlabel('D', fontsize=14, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.title('Below Cloud Base January - June 2022', fontsize=14, fontweight='bold')
plt.xlim(10**-2, 10**2)
plt.ylim(10**-2, 10**2.5)
plt.show()
#%%
##overlaying dry concentration contours onto mass contour plot

def calculate_mass(N0, D):
    integrand = lambda d: np.exp(-d / D) * d**3
    mass_integral, _ = quad(integrand, 2, np.inf)
    return N0 * mass_integral

filtered_combined_clean = filtered_combined[(filtered_combined['D'] > 0) & 
                                            (filtered_combined['dryintercept'] > 0)].copy()

# Recalculate mass for each point using the integrated mass equation (apply row-wise)
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
plt.figure(figsize=(15, 10))

sc = plt.scatter(filtered_combined_clean['D'], filtered_combined_clean['dryintercept'], 
                 c=filtered_combined_clean['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')


contour_plot = plt.contour(D_grid, dryintercept_grid, mass_grid, levels=mass_levels, 
                           colors='red', linewidths=0.7, alpha=0.75)

if len(contour_plot.allsegs[0]) == 0:
    print("No contours were created. Check your data range or mass grid calculation.")

N0_values = filtered_combined_clean['dryintercept'].values  
D_values = filtered_combined_clean['D'].values 

N0_grid = np.linspace(min(N0_values), max(N0_values), 200)  
D_grid_conc = np.linspace(min(D_values), max(D_values), 200) 

concentration_grid = np.zeros((len(N0_grid), len(D_grid_conc)))

# Calculate concentration for each combination of N0 and D on the grid
for i, N0 in enumerate(N0_grid):
    for j, D in enumerate(D_grid_conc):
        concentration_grid[i, j] = N0 * D 

# Overlay the dry concentration contours using this refined grid
concentration_contour = plt.contour(D_grid_conc, N0_grid, concentration_grid, levels=20, colors='blue', linewidths=0.75, alpha=0.6)

plt.colorbar(sc, label='Corrected Windspeed (m/s)')
plt.xlabel('D', fontsize=20, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=20, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.title('Below Cloud Base January - June 2022', fontsize=20, fontweight='bold')
plt.xlim(10**-2, 10**2)
plt.ylim(10**-2, 10**2.5)
plt.show()
#%%
##We still have to calculate d dry
##There are x amount of bins ranging from 2.5 to 50 aka 12 bins
##ddry=d/grh
##I need to do every bin by grh for every leg 
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
# Extract dry intercept (N0) and D values from the dictionaries
N0_values = [entry['dry intercept'] for entry in filtered_master_BCB_dryintercept]
D_values = [entry['D'] for entry in filtered_master_BCB_ddry]

def mass_integrand(d, D):
    return np.exp(-d / D) * d**3

def calculate_mass(N0, D):
    # Perform the integration from 0 to infinity
    mass, error = quad(mass_integrand, 0, np.inf, args=(D,))
    return N0 * mass

mass_values = []
for N0, D in zip(N0_values, D_values):
    mass = calculate_mass(N0, D)
    mass_values.append(mass)

mass_values = np.array(mass_values)

# Set a threshold to replace non-positive values with NaN (only if they exist)
mass_values[mass_values <= 0] = np.nan  # Replace non-positive values with NaN

# Plotting the mass values against D and N0
plt.figure(figsize=(10, 8))

# Scatter plot to visualize mass values
plt.scatter(D_values, N0_values, c=mass_values, cmap='viridis', edgecolor='k', s=100)
plt.colorbar(label='Mass (M)')
plt.xlabel('D (e-folding diameter)')
plt.ylabel('Dry Intercept (N0)')
plt.title('Scatter Plot of Mass M vs D and N0')
plt.xscale('log')
plt.yscale('log')
plt.show()
#%%
N0_grid = np.linspace(min(N0_values), max(N0_values), 200)  
D_grid = np.linspace(min(D_values), max(D_values), 200)  

mass_grid = np.zeros((len(N0_grid), len(D_grid)))

for i, N0 in enumerate(N0_grid):
    for j, D in enumerate(D_grid):
        mass_grid[i, j] = calculate_mass(N0, D)

plt.figure(figsize=(10, 8))
contour = plt.contour(D_grid, N0_grid, mass_grid, levels=20, cmap='viridis')  # Use plt.contour() for lines
plt.clabel(contour, inline=True, fontsize=8)  # Add labels to the contour lines
plt.colorbar(contour, label='Mass (M)') 
plt.xlabel('D (e-folding diameter)')
plt.ylabel('Dry Intercept (N0)')
plt.title('Contour Plot of Mass M vs D and N0')
# plt.xscale('log')
plt.xlim(0, 13)
plt.ylim(0,40)
# plt.ylim(10**0.6, 10**2.4)
# plt.yscale('log')
plt.show()
#%%
##trying thr 0th moment for dry constant concentration contours

N0_values = [entry['dry intercept'] for entry in filtered_master_BCB_dryintercept]
D_values = [entry['D'] for entry in filtered_master_BCB_ddry]
N0_grid = np.linspace(min(N0_values), max(N0_values), 200) 
D_grid = np.linspace(min(D_values), max(D_values), 200)  

concentration_grid = np.zeros((len(N0_grid), len(D_grid)))
for i, N0 in enumerate(N0_grid):
    for j, D in enumerate(D_grid):
        concentration_grid[i, j] = N0 * D 

plt.figure(figsize=(10, 8))
contour = plt.contour(D_grid, N0_grid, concentration_grid, levels=20, cmap='viridis')  # Use plt.contour() for lines
plt.clabel(contour, inline=True, fontsize=8) 
plt.colorbar(contour, label='Dry Concentration')  
plt.xlabel('D')
plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
plt.title(' Dry Concentration', fontsize=14, fontweight='bold')
plt.xscale('log')
plt.yscale('log')
# plt.xlim(10**0, 10**3)
# plt.ylim(10**0, 10**3)
plt.show()

#%%
##Attempting constant dry concentration contours

# Step 1: Function to calculate total concentration (zeroth moment integral)
def total_concentration(D, dryintercept):
    # Assuming a simple relationship for demonstration; modify as needed
    # For example, mass M = N0 * integral { exp(-d/D) d^3 dd }
    # This is a placeholder; replace with your actual total concentration logic
    return (dryintercept / D) 


d_values = np.linspace(df_combined['D'].min(), df_combined['D'].max(), 100)
dryintercept_values = np.linspace(df_combined['dryintercept'].min(), df_combined['dryintercept'].max(), 100)
D_grid, dryintercept_grid = np.meshgrid(d_values, dryintercept_values)


total_concentration_grid = total_concentration(D_grid, dryintercept_grid)
plt.figure(figsize=(10, 8))

sns.kdeplot(x=df_combined['D'], y=df_combined['dryintercept'], cmap='viridis', fill=True, thresh=0)

contour_levels = np.linspace(total_concentration_grid.min(), total_concentration_grid.max(), 10)  # Adjust the number of levels
contour = plt.contour(D_grid, dryintercept_grid, total_concentration_grid, levels=contour_levels, colors='red', linewidths=1, linestyles='solid')
plt.clabel(contour, inline=True, fontsize=10)

plt.xlabel('D (Diameter)')
plt.ylabel('Dry Intercept')
plt.title('Contour Plot of Total Concentration and Density Estimate')
plt.colorbar(label='Density Estimate')
plt.show()
#%%
##D50 mass calculation 

# Assuming you have these lists from your previous calculations
N0_values = [entry['dry intercept'] for entry in filtered_master_BCB_dryintercept]
D_values = [entry['D'] for entry in filtered_master_BCB_ddry]

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


plt.figure(figsize=(10, 8))
contour = plt.contour(D_grid, N0_grid, concentration_grid, levels=20, cmap='viridis')  # Use plt.contour() for lines
plt.clabel(contour, inline=True, fontsize=8)  
cbar = plt.colorbar(contour)
cbar.set_label('Dry Concentration', fontsize=14, fontweight='bold')  
cbar.ax.tick_params(labelsize=12) 

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
#mass contours with median diameter
# Extract dry intercept (N0) and D values from the dictionaries
N0_values = [entry['dry intercept'] for entry in filtered_master_BCB_dryintercept]
D_values = [entry['D'] for entry in filtered_master_BCB_ddry]

# Create grids for contour plotting
N0_grid = np.linspace(min(N0_values), max(N0_values), 200)  # Create 200 points for N0
D_grid = np.linspace(min(D_values), max(D_values), 200)  # Create 200 points for D

# Create empty array for mass grid
mass_grid = np.zeros((len(N0_grid), len(D_grid)))

# Define the mass integrand function as before
def mass_integrand(d, D):
    return np.exp(-d / D) * d**3

# Calculate total mass for each combination of N0 and D on the grid
for i, N0 in enumerate(N0_grid):
    for j, D in enumerate(D_grid):
        mass, _ = quad(mass_integrand, 0, np.inf, args=(D,))
        mass_grid[i, j] = N0 * mass  # Calculate the total mass based on N0

# Create cumulative mass for finding median diameter
total_mass = np.sum(mass_grid, axis=0)  # Total mass for each D
cumulative_mass = np.cumsum(total_mass)  # Cumulative mass
total_mass_value = np.sum(total_mass)  # Total mass value
median_mass_threshold = total_mass_value / 2  # Half the total mass

# Find diameter corresponding to 50% cumulative mass
median_diameter = None
for i, cm in enumerate(cumulative_mass):
    if cm >= median_mass_threshold:
        median_diameter = D_grid[i]
        break

# Create contour plot for mass
plt.figure(figsize=(10, 8))
contour = plt.contour(D_grid, N0_grid, mass_grid, levels=20, cmap='viridis')  # Use plt.contour() for lines
plt.clabel(contour, inline=True, fontsize=8)  # Add labels to the contour lines

# Add a colorbar for reference
plt.colorbar(label='Mass')  # Set the colorbar label

# Add median diameter line
if median_diameter is not None:
    plt.axvline(x=median_diameter, color='red', linestyle='--', label=f'Median Diameter: {median_diameter:.2f}')
    plt.legend()

plt.xlabel('D)', fontsize=14, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
plt.title('Mass Contours with Median Diameter (um)', fontsize=14, fontweight='bold')
plt.xscale('log')
plt.yscale('log')
plt.xlim(10**0, 10**1.2)
plt.ylim(10**1, 10**2.1)
plt.show()
#%%
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

#%%
# Create a set of unique legs from Y_BCB_clean
valid_legs_set = set(
    zip(Y_BCB_clean["Date"], Y_BCB_clean["Leg_index"])
)

# Filter filtered_combined_clean
filtered_combined_clean = filtered_combined_clean[
    filtered_combined_clean[["Date", "Leg_index"]].apply(tuple, axis=1).isin(valid_legs_set)
]

print(f"Filtered combined_clean shape after alignment: {filtered_combined_clean.shape}")
#%%
# Set of legs in Y_BCB_clean
legs_in_Y_BCB_clean = set(zip(Y_BCB_clean["Date"], Y_BCB_clean["Leg_index"]))

# Set of legs in filtered_combined_clean
legs_in_combined_clean = set(zip(filtered_combined_clean["Date"], filtered_combined_clean["Leg_index"]))

# Find the missing legs
missing_legs = legs_in_Y_BCB_clean - legs_in_combined_clean
print(f"Missing legs: {missing_legs}")
print(f"Number of missing legs: {len(missing_legs)}")

# %%
missing_rows = Y_BCB_clean[
    (Y_BCB_clean["Date"] == "2022-06-18") &
    (Y_BCB_clean["Leg_index"].isin({0, 1, 2, 3}))
]
filtered_combined_clean = pd.concat([filtered_combined_clean, missing_rows], ignore_index=True)
#%%
def cumulative_mass(N0, D, d_max):
    integrand = lambda d: N0 * np.exp(-d / D) * d**3
    cumulative_mass, _ = quad(integrand, 2, d_max)
    return cumulative_mass

def total_mass(N0, D):
    integrand = lambda d: N0 * np.exp(-d / D) * d**3
    total_mass, _ = quad(integrand, 2, np.inf)
    return total_mass

# Define common diameters for integration
diameters = np.linspace(2, 50, 100)

# Initialize arrays to store results
diameters_at_50_mass = []
slopes = []

# Process each leg
plt.figure(figsize=(12, 8))

for _, row in filtered_combined_clean.iterrows():
    N0 = row['dryintercept']
    D = row['D']
    
    # Skip legs with missing or invalid N0 or D
    if np.isnan(N0) or np.isnan(D) or N0 <= 0 or D <= 0:
        diameters_at_50_mass.append(np.nan)
        slopes.append(np.nan)
        continue

    # Calculate total mass and cumulative mass
    try:
        M_total = total_mass(N0, D)
        cumulative_masses = [cumulative_mass(N0, D, d) for d in diameters]
        normalized_cumulative_mass = np.array(cumulative_masses) / M_total

        # Interpolate to find the diameter at 50% mass
        interpolation_func = interp1d(normalized_cumulative_mass, diameters, kind='linear', fill_value='extrapolate')
        diameter_at_50 = interpolation_func(0.5)
        diameters_at_50_mass.append(diameter_at_50)
        slopes.append(D)

        # Plot cumulative mass distribution for each leg
        plt.plot(diameters, normalized_cumulative_mass, alpha=0.5, label=f"Leg {row['Leg_index']}, Date {row['Date']}")
        plt.axvline(diameter_at_50, color='red', linestyle='--', alpha=0.5, label=f"50% Mass at {diameter_at_50:.2f} µm (Leg {row['Leg_index']})")

    except Exception as e:
        print(f"Error processing leg {row['Leg_index']} on {row['Date']}: {e}")
        diameters_at_50_mass.append(np.nan)
        slopes.append(np.nan)

plt.xlabel('Diameter (µm)', fontsize=14, fontweight='bold')
plt.ylabel('Cumulative Mass (Normalized)', fontsize=14, fontweight='bold')
plt.title('Cumulative Mass Distribution (50% Mass)', fontsize=14, fontweight='bold')
plt.grid()
plt.tight_layout()
plt.show()

print("Diameters at 50% Mass and Slopes for Each Leg:")
for i, (diameter, slope) in enumerate(zip(diameters_at_50_mass, slopes), 1):
    print(f"Leg {i}: Diameter at 50% mass = {diameter:.2f} µm, Slope (D) = {slope:.2f}")

# Filter out NaN values for correlation calculation and plotting
valid_indices = ~np.isnan(diameters_at_50_mass) & ~np.isnan(slopes)
valid_diameters = np.array(diameters_at_50_mass)[valid_indices]
valid_slopes = np.array(slopes)[valid_indices]

# Scatter plot of Diameter at 50% Mass vs Slope (D)
plt.figure(figsize=(10, 6))
plt.scatter(valid_slopes, valid_diameters, c='blue', alpha=0.7)
plt.xlabel('Slope (D)', fontsize=14, fontweight='bold')
plt.ylabel('Diameter at 50% Mass (µm)', fontsize=14, fontweight='bold')
plt.title('Diameter at 50% Mass vs Slope (D)', fontsize=14, fontweight='bold')
plt.grid()
plt.tight_layout()
plt.show()

# Calculate correlation coefficient
if len(valid_diameters) > 1 and len(valid_slopes) > 1:
    correlation_matrix = np.corrcoef(valid_slopes, valid_diameters)
    correlation_coefficient = correlation_matrix[0, 1]
    print(f"Correlation coefficient between Slope (D) and Diameter at 50% Mass: {correlation_coefficient:.2f}")
else:
    print("Insufficient valid data points to calculate correlation.")
#%%
ratios = np.array(diameters_at_50_mass) / filtered_combined_clean['D']

# Plot the ratios
plt.figure(figsize=(12, 8))
plt.plot(ratios, marker='o', linestyle='', color='purple', alpha=0.7)
plt.xlabel('Leg Index', fontsize=14, fontweight='bold')
plt.ylabel('Diameter at 50% Mass / Slope (D)', fontsize=14, fontweight='bold')
plt.title('Ratio of Diameter at 50% Mass to Slope (D)', fontsize=14, fontweight='bold')
plt.grid()
plt.tight_layout()
plt.show()

# Check if the ratios are nearly constant
print(f"Mean Ratio: {np.mean(ratios):.2f}, Std Dev of Ratio: {np.std(ratios):.2f}")
#%%
# Function to calculate the mass for a given diameter and concentration
def calculate_mass(diameter, concentration):
    return concentration * (np.pi / 6) * (diameter**3)

# Function to calculate the cumulative mass and find the 50% mass diameter
def calculate_cumulative_mass(diameters, masses):
    cumulative_mass = np.cumsum(masses)
    total_mass = np.sum(masses)
    normalized_cumulative_mass = cumulative_mass / total_mass

    # Interpolate to find the diameter where 50% of the mass occurs
    interpolation_func = interp1d(normalized_cumulative_mass, diameters, kind="linear", fill_value="extrapolate")
    diameter_at_50_percent = interpolation_func(0.5)
    return diameter_at_50_percent

# Initialize lists to store results
diameters_at_50_mass = []
slope_values = []

# Define the common bin diameters (size distribution bins)
common_bins = np.array(filtered_master_BCB_ddry[0]["filtered_ddry"])

# Iterate through each leg in filtered_combined_clean
for i, row in filtered_combined_clean.iterrows():
    # Extract slope (D) and dry intercept (concentration at dry diameter)
    slope = row["D"]
    dryintercept = row["dryintercept"]

    # Ensure valid slope and dry intercept
    if np.isnan(slope) or np.isnan(dryintercept) or slope <= 0 or dryintercept <= 0:
        continue

    # Generate size distribution based on slope and dry intercept
    concentrations = dryintercept * np.exp(-common_bins / slope)  # Exponential size distribution
    masses = calculate_mass(common_bins, concentrations)  # Calculate masses for each bin

    # Calculate the diameter at which 50% of the mass falls
    try:
        diameter_at_50 = calculate_cumulative_mass(common_bins, masses)
        diameters_at_50_mass.append(diameter_at_50)
        slope_values.append(slope)
    except Exception as e:
        print(f"Error calculating 50% mass for leg {i}: {e}")

# Convert to numpy arrays for analysis
diameters_at_50_mass = np.array(diameters_at_50_mass)
slope_values = np.array(slope_values)

# Scatter plot: Diameter at 50% Mass vs. Slope (D)
plt.figure(figsize=(12, 8))
plt.scatter(slope_values, diameters_at_50_mass, alpha=0.7, color="blue")
plt.xlabel("Slope (D)", fontsize=14, fontweight="bold")
plt.ylabel("Diameter at 50% Mass (µm)", fontsize=14, fontweight="bold")
plt.title("Diameter at 50% Mass vs. Slope (D)", fontsize=14, fontweight="bold")
plt.grid()
plt.tight_layout()
plt.show()

# Calculate correlation coefficient
correlation = np.corrcoef(slope_values, diameters_at_50_mass)[0, 1]
print(f"Correlation coefficient between Slope (D) and Diameter at 50% Mass: {correlation:.2f}")
#%%

# Function to calculate mass from diameter and concentration
def calculate_mass(diameter, concentration):
    return concentration * (np.pi / 6) * (diameter**3)

# Function to calculate the cumulative mass and find the 50% mass diameter
def calculate_cumulative_mass(diameters, masses):
    cumulative_mass = np.cumsum(masses)
    total_mass = np.sum(masses)
    normalized_cumulative_mass = cumulative_mass / total_mass

    # Interpolate to find the diameter where 50% of the mass occurs
    interpolation_func = interp1d(normalized_cumulative_mass, diameters, kind="linear", fill_value="extrapolate")
    diameter_at_50_percent = interpolation_func(0.5)
    return diameter_at_50_percent

# Initialize lists to store results
diameters_at_50_mass = []
slopes_at_50_mass = []

# Define the common bin diameters (size distribution bins)
common_bins = np.array(filtered_master_BCB_ddry[0]["filtered_ddry"])

# Iterate through each leg in filtered_combined_clean
for i, row in filtered_combined_clean.iterrows():
    # Extract slope (D) and dry intercept (concentration at dry diameter)
    slope = row["D"]
    dryintercept = row["dryintercept"]

    # Ensure valid slope and dry intercept
    if np.isnan(slope) or np.isnan(dryintercept) or slope <= 0 or dryintercept <= 0:
        continue

    # Generate size distribution based on slope and dry intercept
    concentrations = dryintercept * np.exp(-common_bins / slope)  # Exponential size distribution
    masses = calculate_mass(common_bins, concentrations)  # Calculate masses for each bin

    # Calculate the diameter at which 50% of the mass falls
    try:
        diameter_at_50 = calculate_cumulative_mass(common_bins, masses)

        # Calculate the slope (D50) at the 50% mass diameter
        slope_at_50 = diameter_at_50 / slope  # Ratio of D50 to slope for normalization

        # Append results to the lists
        diameters_at_50_mass.append(diameter_at_50)
        slopes_at_50_mass.append(slope_at_50)
    except Exception as e:
        print(f"Error calculating 50% mass for leg {i}: {e}")

# Convert to numpy arrays for analysis
diameters_at_50_mass = np.array(diameters_at_50_mass)
slopes_at_50_mass = np.array(slopes_at_50_mass)

# Scatter plot: Diameter at 50% Mass (D50) vs. Slope at 50% (normalized)
plt.figure(figsize=(12, 8))
plt.scatter(slopes_at_50_mass, diameters_at_50_mass, alpha=0.7, color="green")
plt.xlabel("Normalized Slope at 50% Mass (D50 / D)", fontsize=14, fontweight="bold")
plt.ylabel("Diameter at 50% Mass (µm)", fontsize=14, fontweight="bold")
plt.title("Diameter at 50% Mass vs. Normalized Slope at 50% Mass", fontsize=14, fontweight="bold")
plt.grid()
plt.tight_layout()
plt.show()

# Calculate correlation coefficient
correlation = np.corrcoef(slopes_at_50_mass, diameters_at_50_mass)[0, 1]
print(f"Correlation coefficient between Normalized Slope at 50% Mass and Diameter at 50% Mass: {correlation:.2f}")
#%%
# Function to calculate mass for each diameter bin
def calculate_mass(diameter, concentration):
    return concentration * (np.pi / 6) * (diameter**3)

# Function to calculate total mass for each leg
def calculate_total_mass(diameters, masses):
    return np.sum(masses)  # Full integral (sum over all bins)

# Initialize lists to store results
total_masses = []
diameters_at_50_mass = []

# Define the common bin diameters (size distribution bins)
common_bins = np.array(filtered_master_BCB_ddry[0]["filtered_ddry"])

# Iterate through each leg in filtered_combined_clean
for i, row in filtered_combined_clean.iterrows():
    # Extract slope (D) and dry intercept (concentration at dry diameter)
    slope = row["D"]
    dryintercept = row["dryintercept"]

    # Ensure valid slope and dry intercept
    if np.isnan(slope) or np.isnan(dryintercept) or slope <= 0 or dryintercept <= 0:
        continue

    # Generate size distribution based on slope and dry intercept
    concentrations = dryintercept * np.exp(-common_bins / slope)  # Exponential size distribution
    masses = calculate_mass(common_bins, concentrations)  # Calculate masses for each bin

    # Calculate the total mass
    total_mass = calculate_total_mass(common_bins, masses)
    total_masses.append(total_mass)

    # Calculate the diameter at which 50% of the mass falls
    try:
        cumulative_mass = np.cumsum(masses)
        normalized_cumulative_mass = cumulative_mass / np.sum(masses)
        interpolation_func = interp1d(normalized_cumulative_mass, common_bins, kind="linear", fill_value="extrapolate")
        diameter_at_50 = interpolation_func(0.5)
        diameters_at_50_mass.append(diameter_at_50)
    except Exception as e:
        print(f"Error calculating 50% mass for leg {i}: {e}")

# Convert to numpy arrays for analysis
total_masses = np.array(total_masses)
diameters_at_50_mass = np.array(diameters_at_50_mass)

# Scatter plot: Total Mass vs. Diameter at 50% Mass (D50)
plt.figure(figsize=(12, 8))
plt.scatter(diameters_at_50_mass, total_masses, alpha=0.7, color="blue")
plt.xlabel("Diameter at 50% Mass (D50, um)", fontsize=14, fontweight="bold")
plt.ylabel("Total Mass (µm³)", fontsize=14, fontweight="bold")
plt.title("Total Mass vs. Diameter at 50% Mass (D50)", fontsize=14, fontweight="bold")
plt.grid()
plt.tight_layout()
plt.show()

# Calculate correlation coefficient
correlation = np.corrcoef(diameters_at_50_mass, total_masses)[0, 1]
print(f"Correlation coefficient between Total Mass and Diameter at 50% Mass (D50): {correlation:.2f}")
#%%
# Function to calculate mass for each diameter bin
def calculate_mass(diameter, concentration):
    return concentration * (np.pi / 6) * (diameter**3)

# Function to calculate total mass for each leg
def calculate_total_mass(diameters, masses):
    return np.sum(masses)  # Full integral (sum over all bins)

# Initialize lists to store results
total_masses = []
slope_values = []

# Define the common bin diameters (size distribution bins)
common_bins = np.array(filtered_master_BCB_ddry[0]["filtered_ddry"])

# Iterate through each leg in filtered_combined_clean
for i, row in filtered_combined_clean.iterrows():
    # Extract slope (D) and dry intercept (concentration at dry diameter)
    slope = row["D"]
    dryintercept = row["dryintercept"]

    # Ensure valid slope and dry intercept
    if np.isnan(slope) or np.isnan(dryintercept) or slope <= 0 or dryintercept <= 0:
        continue

    # Generate size distribution based on slope and dry intercept
    concentrations = dryintercept * np.exp(-common_bins / slope)  # Exponential size distribution
    masses = calculate_mass(common_bins, concentrations)  # Calculate masses for each bin

    # Calculate the total mass
    total_mass = calculate_total_mass(common_bins, masses)
    total_masses.append(total_mass)
    slope_values.append(slope)

# Convert to numpy arrays for analysis
total_masses = np.array(total_masses)
slope_values = np.array(slope_values)

# Scatter plot: Total Mass vs. Slope (D)
plt.figure(figsize=(12, 8))
plt.scatter(slope_values, total_masses, alpha=0.7, color="green")
plt.xlabel("Slope (D)", fontsize=14, fontweight="bold")
plt.ylabel("Total Mass (µm³)", fontsize=14, fontweight="bold")
plt.title("Total Mass vs. Slope (D)", fontsize=14, fontweight="bold")
plt.grid()
plt.tight_layout()
plt.show()

# Calculate correlation coefficient
correlation = np.corrcoef(slope_values, total_masses)[0, 1]
print(f"Correlation coefficient between Total Mass and Slope (D): {correlation:.2f}")
#%%
# Function to calculate mass for each diameter bin
def calculate_mass(diameter, concentration):
    return concentration * (np.pi / 6) * (diameter**3)

# Function to calculate cumulative mass and find D50
def calculate_cumulative_mass(diameters, masses):
    cumulative_mass = np.cumsum(masses)
    total_mass = np.sum(masses)
    normalized_cumulative_mass = cumulative_mass / total_mass

    # Interpolate to find the diameter where 50% of the mass occurs
    interpolation_func = interp1d(normalized_cumulative_mass, diameters, kind="linear", fill_value="extrapolate")
    diameter_at_50_percent = interpolation_func(0.5)
    return diameter_at_50_percent

# Initialize lists to store results
total_masses = []
slope50_values = []

# Define the common bin diameters (size distribution bins)
common_bins = np.array(filtered_master_BCB_ddry[0]["filtered_ddry"])

# Iterate through each leg in filtered_combined_clean
for i, row in filtered_combined_clean.iterrows():
    # Extract slope (D) and dry intercept (concentration at dry diameter)
    slope = row["D"]
    dryintercept = row["dryintercept"]

    # Ensure valid slope and dry intercept
    if np.isnan(slope) or np.isnan(dryintercept) or slope <= 0 or dryintercept <= 0:
        continue

    # Generate size distribution based on slope and dry intercept
    concentrations = dryintercept * np.exp(-common_bins / slope)  # Exponential size distribution
    masses = calculate_mass(common_bins, concentrations)  # Calculate masses for each bin

    # Calculate the diameter where 50% of the mass falls (D50)
    try:
        diameter_at_50 = calculate_cumulative_mass(common_bins, masses)
    except Exception as e:
        print(f"Error calculating D50 for leg {i}: {e}")
        continue

    # Define Slope50 as the slope (D) at the D50 diameter
    slope50 = slope * np.exp(-diameter_at_50 / slope)

    # Calculate total mass for the leg
    total_mass = np.sum(masses)

    # Append results
    total_masses.append(total_mass)
    slope50_values.append(slope50)

# Convert to numpy arrays for analysis
total_masses = np.array(total_masses)
slope50_values = np.array(slope50_values)

# Scatter plot: Total Mass vs. Slope50
plt.figure(figsize=(12, 8))
plt.scatter(slope50_values, total_masses, alpha=0.7, color="orange")
plt.xlabel("Slope50 (Slope at D50)", fontsize=14, fontweight="bold")
plt.ylabel("Total Mass (µm³)", fontsize=14, fontweight="bold")
plt.title("Total Mass vs. Slope at 50% Mass (Slope50)", fontsize=14, fontweight="bold")
plt.grid()
plt.tight_layout()
plt.show()

# Calculate correlation coefficient
correlation = np.corrcoef(slope50_values, total_masses)[0, 1]
print(f"Correlation coefficient between Total Mass and Slope50: {correlation:.2f}")

#%%

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
#%%
#The goal now is to investigate the relationship between 
# these 50% values and the slope (D). If the slope wider (represents a broader distribution), 
# we expect that the diameter at 50% mass will also be larger, as the mass is distributed across larger particle sizes. 
# To analyze this relationship, we can plot the 50% mass diameters against the slope values for all legs in your dataset.
# %%
#This code is based off the cumulative mass code above (not for each individual leg)

# Arrays to store 50% mass diameters and corresponding slopes (D)
fifty_percent_diameters = []
slopes = []

# Loop through all legs to calculate the 50% diameter and store slope (D)
for i, row in filtered_combined_clean.iterrows():
    N0 = row['dryintercept']
    D = row['D']

    # Calculate the total mass for this leg
    M_total = total_mass(N0, D)

    # Calculate the cumulative mass as a function of diameter
    cumulative_masses = np.array([cumulative_mass(N0, D, d) for d in diameters]) / M_total

    # Interpolate to find the diameter at 50% cumulative mass
    interpolation_func = interp1d(cumulative_masses, diameters, kind='linear', fill_value='extrapolate')
    diameter_at_50 = interpolation_func(0.5)

    # Store results
    fifty_percent_diameters.append(diameter_at_50)
    slopes.append(D)

# Convert results to numpy arrays for plotting
fifty_percent_diameters = np.array(fifty_percent_diameters)
slopes = np.array(slopes)

# Scatter plot to visualize the relationship
plt.figure(figsize=(10, 6))
plt.scatter(slopes, fifty_percent_diameters, color='blue', alpha=0.7, edgecolor='k')
plt.xlabel('Slope (D)', fontsize=14, fontweight='bold')
plt.ylabel('Diameter at 50% Cumulative Mass (µm)', fontsize=14, fontweight='bold')
plt.title('Relationship Between Slope (D) and 50% Mass Diameter', fontsize=14, fontweight='bold')
plt.grid()
plt.tight_layout()
plt.show()

# Print correlation coefficient
correlation = np.corrcoef(slopes, fifty_percent_diameters)[0, 1]
print(f"Correlation between Slope (D) and Diameter at 50% Mass: {correlation:.2f}")
#%%
#This code is based off the mass code for every individual leg. 


# Functions to calculate cumulative mass and total mass
def cumulative_mass(N0, D, d_max):
    integrand = lambda d: N0 * np.exp(-d / D) * d**3
    cumulative_mass, _ = quad(integrand, 2, d_max)  # Integrate from 2 µm to d_max
    return cumulative_mass

def total_mass(N0, D):
    integrand = lambda d: N0 * np.exp(-d / D) * d**3
    total_mass, _ = quad(integrand, 2, np.inf)  # Total mass over all diameters
    return total_mass

# Diameters to evaluate
diameters = np.linspace(2, 50, 100)

# Store results for each leg
diameters_at_50_mass = []
slopes_D = []

# Loop over individual legs
for i, row in filtered_combined_clean.iterrows():
    N0 = row['dryintercept']
    D = row['D']

    # Skip invalid rows
    if np.isnan(N0) or np.isnan(D) or D <= 0:
        print(f"Skipping leg {i}: Invalid N0 or D")
        continue

    # Calculate total mass and cumulative mass
    M_total = total_mass(N0, D)
    cumulative_masses = [cumulative_mass(N0, D, d) for d in diameters]
    normalized_cumulative_mass = np.array(cumulative_masses) / M_total

    # Interpolate to find the diameter at 50% mass
    interpolation_func = interp1d(normalized_cumulative_mass, diameters, kind='linear', fill_value='extrapolate')
    diameter_at_50 = interpolation_func(0.5)  # Diameter where 50% of mass is reached
    diameters_at_50_mass.append(diameter_at_50)
    slopes_D.append(D)

    print(f"Leg {i}: Diameter at 50% mass = {diameter_at_50:.2f} µm, Slope (D) = {D:.2f}")

# Plot diameter at 50% mass vs. slope (D)
plt.figure(figsize=(10, 6))
plt.scatter(slopes_D, diameters_at_50_mass, color='blue', alpha=0.7)
plt.xlabel('Slope (D)', fontsize=14, fontweight='bold')
plt.ylabel('Diameter at 50% Mass (µm)', fontsize=14, fontweight='bold')
plt.title('Diameter at 50% Mass vs. Slope (D)', fontsize=14, fontweight='bold')
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate correlation coefficient
correlation_coefficient = np.corrcoef(slopes_D, diameters_at_50_mass)[0, 1]
print(f"Correlation Coefficient between Diameter at 50% Mass and Slope (D): {correlation_coefficient:.2f}")
#%%
# Flattening the nested lists
flattened_gRH = [entry for sublist in filtered_master_BCB_gRH for entry in sublist]

# Debug: Check lengths and samples
print("Number of entries in flattened_gRH:", len(flattened_gRH))
print("Sample from flattened_gRH:", flattened_gRH[:5])
#%%
# Convert flattened_gRH to DataFrame
gRH_df = pd.DataFrame(flattened_gRH)

gRH_df["Rh_mean"] = gRH_df["Rh_mean"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else np.nan)
gRH_df["gRh_mean"] = gRH_df["gRh_mean"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else np.nan)

# Drop rows with NaN in Rh_mean or gRh_mean
gRH_df_cleaned = gRH_df.dropna(subset=["Rh_mean", "gRh_mean"])

# Debug: Check cleaned DataFrames
print("Cleaned gRH_df shape:", gRH_df_cleaned.shape)
print("Sample cleaned gRH_df:\n", gRH_df_cleaned.head())
#%%
# Convert flattened_gRH to DataFrame
gRH_df = pd.DataFrame(flattened_gRH)

# Extract scalar values from lists in Rh_mean and gRh_mean
gRH_df["Rh_mean"] = gRH_df["Rh_mean"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else np.nan)
gRH_df["gRh_mean"] = gRH_df["gRh_mean"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else np.nan)



#%%
##Attempting machine learning using Random Forest 

#First lets create a dataset that includes 
# Initialize new columns for RH and gRH in filtered_combined_clean
filtered_combined_clean['Rh_mean'] = np.nan
filtered_combined_clean['gRh_mean'] = np.nan

# Iterate through rows in filtered_combined_clean
for i, row in filtered_combined_clean.iterrows():
    # Find matching rows in filtered_master_BCB_gRH based on BCB_start and BCB_stop
    match = [
        entry for entry in flattened_gRH  # Assuming you flattened filtered_master_BCB_gRH earlier
        if entry['BCB_start'] == row['BCB_start'] and entry['BCB_stop'] == row['BCB_stop']
    ]
    
    # If a match is found, assign Rh_mean and gRh_mean
    if match:
        filtered_combined_clean.at[i, 'Rh_mean'] = match[0]['Rh_mean']
        filtered_combined_clean.at[i, 'gRh_mean'] = match[0]['gRh_mean']

# Debug: Check the updated filtered_combined_clean
print("Updated filtered_combined_clean shape:", filtered_combined_clean.shape)
print("Sample rows with Rh_mean and gRh_mean:\n", filtered_combined_clean.head())

#%%
# # Save the cleaned dataset
# merged_cleaned.to_csv("merged_cleaned_dataset.csv", index=False)
#%%



















































































































































































































































#%%
#Filtered ambient concentrations

filtered_master_BCB_NT_dict = {}

if isinstance(filtered_master_BCB_gRH[0], list):
    filtered_master_BCB_gRH = [item for sublist in filtered_master_BCB_gRH for item in sublist]

unique_keys = set()

# Flatten master_min_exponential
flattened_exponential = []
for exp_list in master_BCB_exponential.values():
    flattened_exponential.extend(exp_list)

# Create a dictionary for quick lookup
exponential_dict = {(exp['Date'], exp['BCB_start'], exp['BCB_stop']): exp for exp in flattened_exponential}
dmin = 2.5  


for entry in filtered_master_BCB_gRH:
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    gRh_mean = entry['gRh_mean'][0]  # Assuming gRh_mean is a list with one value
    Rh_mean = entry['Rh_mean'][0]  # Assuming Rh_mean is a list with one value
    if Rh_mean < 0:
        continue
    key = (date, BCB_start, BCB_stop)
    if date in master_BCB_exponential:
        exp_params_list = master_BCB_exponential[date]
        for exp_params in exp_params_list:
            D = exp_params['D']
            n0 = exp_params['n0']
            NT = n0* D * np.exp(-dmin / D)
            
            # Store the result in the dictionary
            filtered_master_BCB_NT_dict[key] = {
                'Date': date,
                'BCB_start': BCB_start,
                'BCB_stop': BCB_stop,
                'Rh_mean': entry['Rh_mean'],
                'gRh_mean': entry['gRh_mean'],
                'filtered_NT': NT
            }

filtered_master_BCB_NT = list(filtered_master_BCB_NT_dict.values())
print(f"Length of filtered_master_BCB_NT: {len(filtered_master_BCB_NT)}")

# %%

#%%
#Filtered Nt and Ntd vary with windspeed
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

# Validate and combine the data
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

            # Initialize variables to store found data
            Ntd = None
            NT = None

            # Find the corresponding exponential parameters and dry intercept
            for exp_params in filtered_master_BCB_ntd:
                if exp_params['Date'] == date and exp_params['BCB_start'] == BCB_start and exp_params['BCB_stop'] == BCB_stop:
                    Ntd = exp_params['Ntd']
                    break

            for exp_params in filtered_master_BCB_NT:
                if exp_params['Date'] == date and exp_params['BCB_start'] == BCB_start and exp_params['BCB_stop'] == BCB_stop:
                    NT = exp_params['filtered_NT']
                    break
            
            # Append data only if both Ntd and NT were found
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

# Convert to DataFrame
df_combined = pd.DataFrame(combined_data)

# Check the data for any anomalies
print(df_combined.describe())

# Separate data based on NaN Windspeed
df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]

plt.figure(figsize=(10, 8))

# Plot data with valid Windspeed values
sc = plt.scatter(df_with_windspeed['filtered_NT'], df_with_windspeed['Ntd'], 
                 c=df_with_windspeed['Windspeed'], cmap='viridis', s=100, label='Windspeed')

# Plot data with NaN Windspeed values in black
plt.scatter(df_nan_windspeed['filtered_NT'], df_nan_windspeed['Ntd'], 
            color='black', s=100, label='Windspeed NaN')

# Add color bar
plt.colorbar(sc, label='Corrected Windspeed (m/s)')

# Add labels and title
plt.xlabel('Ambient Concentration (/cm3)', fontsize=14, fontweight='bold')
plt.ylabel('Total droplet concentration \n dry droplets larger than \n 2um diameter (/cm3)', fontsize=14, fontweight='bold')
plt.title('Below cloud base January-June 2022', fontsize=14, fontweight='bold')
plt.yscale('log')
# plt.ylim(10**-1, 10**2)
plt.xlim(10**-2.6, 10**1)

plt.xscale('log')
plt.show()

#%%
#Filtered NT and Ntndt vary with windspeed 

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
            NT = None

            
            for exp_params in filtered_master_BCB_NtdNt:
                if exp_params['Date'] == date and exp_params['BCB_start'] == BCB_start and exp_params['BCB_stop'] == BCB_stop:
                    NtdNt = exp_params['NtdNt']
                    break

            for exp_params in filtered_master_BCB_NT:
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

# Convert to DataFrame
df_combined = pd.DataFrame(combined_data)

print(df_combined.describe())

df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]

plt.figure(figsize=(10, 8))

sc = plt.scatter(df_with_windspeed['filtered_NT'], df_with_windspeed['NtdNt'], 
                 c=df_with_windspeed['Windspeed'], cmap='viridis', s=100, label='Windspeed')

plt.scatter(df_nan_windspeed['filtered_NT'], df_nan_windspeed['NtdNt'], 
            color='black', s=100, label='Windspeed NaN')

plt.colorbar(sc, label='Corrected Windspeed (m/s)')

plt.xlabel('Ambient Concentration (/cm3)', fontsize=14, fontweight='bold')
plt.ylabel('Ratio of total droplet concentration \n dry droplets larger than \n 2um diameter to ambient concentration', fontsize=14, fontweight='bold')
plt.title('Below cloud base January-June 2022', fontsize=14, fontweight='bold')
plt.yscale('log')
# plt.ylim(10**-1, 10**2)
plt.xlim(10**-2.7, 10**1)
plt.xscale('log')
plt.show()

#%%
#Filtered Ntd distribution varies with windspeed

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

        
            Ntd = None
            for exp_params in filtered_master_BCB_ntd:
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
plt.colorbar(sc, label='Corrected Windspeed (m/s)')
plt.xlabel('Windspeed (m/s)', fontsize=14, fontweight='bold')
plt.ylabel('Total Droplet Concentration of droplets \n with ddry larger than 2um d (/cm3)', fontsize=14, fontweight='bold')
plt.title('Below cloud base January - June 2022 ', fontsize=14, fontweight='bold')
plt.yscale('log')  
plt.show()
# %%
## Comparing ambient concentration to our concentration
##over 2um diameter to see how variable 

##So we need to compare ambient concentration to Ntd and ambient
##concentration to Ntdnt

##finally compare Ntdnt to Ntd

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
#Filtered looking at slope versus NtdNT

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

# %%
## Now we need to try and take all of our legs with our new 18 bins of dry diameter sizes and average them 
##onto a consistent x-axis. 

##To do this we need to first create a basic random array with our bins sizes that will act as our 
##constant y-axis in which we will correct all our dry size distrbutions to. We dont care about the number of bins. We
##will plot our new dry size distributions on top of it. 

##To do this, we need D and dryintercept, which we already have from our dry size distributions

#possible array we can use 

##Eventually we need to bin by windspeed. Where each windspeed bin will contain a certain number of size distributions that
##fit in that windspeed bins and these create straight lines 

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
common_bins = np.linspace(0, 10, 25)

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

plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Size distribution by wind speed', fontweight='bold')
plt.legend(title="Average wind speed (m/s)")
plt.tight_layout()
plt.show()

#%%
# Dynamic binning with random number of wind speed bins 
problematic_legs = [
    ('2022-01-26', 12), ]
problematic_set = set(problematic_legs)

# Function to compute size distribution
def size_distribution(x, dryint, D):
    return dryint * np.exp(-x / D)

# Define common bins
common_bins = np.linspace(0, 10, 25)

# Inspect windspeed range in the dataset
min_windspeed = df_combined['Windspeed'].min()
max_windspeed = df_combined['Windspeed'].max()

# Define fine-grained bins dynamically
bin_width = 2  # Adjust this for finer bins
windspeed_bins = [(low, low + bin_width) for low in np.arange(0, max_windspeed, bin_width)]
windspeed_bins[-1] = (windspeed_bins[-1][0], np.inf)  # Ensure the last bin catches all remaining values

# Debug: Print windspeed bins
print(f"Generated windspeed bins: {windspeed_bins}")

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

    # Bin by windspeed dynamically
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

plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Size distribution by wind speed', fontweight='bold')
plt.legend(title="Average wind speed (m/s)")
plt.tight_layout()
plt.show()

#%%
#trying windspeed bins based on the percentile plot 

# Assuming df_combined is already defined and contains 'Windspeed'
windspeeds = df_combined['Windspeed'].dropna()  # Drop NaN values

# Calculate percentiles for windspeed bins
percentiles = np.percentile(windspeeds, [0, 25, 50, 75, 100])  # Change percentiles as needed

# Define your windspeed ranges based on percentiles
windspeed_bins = [(percentiles[i], percentiles[i + 1]) for i in range(len(percentiles) - 1)]

# Define the function to compute log-transformed size distribution
def log_size_distribution(x, dryint, D):
    return np.log(dryint * np.exp(-x / D))

# Define your common bins
common_bins = np.linspace(0, 10, 25)

grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
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

        # Interpolate the log-transformed size distribution for this leg
        interp_func = interp1d(ddry_values, log_size_distribution(ddry_values, dryint, D), kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        # Exponentiate the interpolated values to get back to the original scale for plotting
        interpolated_leg_values = np.exp(interpolated_leg_values)

        # Categorize the leg based on windspeed range
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_distributions[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break

# Average and plot the results
plt.figure(figsize=(12, 8))
for idx, ranges in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])  # Count the number of legs
        plt.plot(common_bins, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")

plt.yscale('log')  # Set y-axis to log scale
plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Size distribution by wind speed', fontweight='bold')
plt.legend(title="Wind speed (m/s)")
plt.tight_layout()
plt.show()
#%%
#Percentile binning with error bars 

# Assuming df_combined is already defined and contains 'Windspeed'
windspeeds = df_combined['Windspeed'].dropna()  # Drop NaN values

# Calculate percentiles for windspeed bins
percentiles = np.percentile(windspeeds, [0, 25, 50, 75, 100])

# Define your windspeed ranges based on percentiles
windspeed_bins = [(percentiles[i], percentiles[i + 1]) for i in range(len(percentiles) - 1)]


# Define the function to compute log-transformed size distribution
def log_size_distribution(x, dryint, D):
    return np.log(dryint * np.exp(-x / D))

# Define your common bins
common_bins = np.linspace(0, 10, 25)

# Set a color palette
colors = sns.color_palette("husl", len(windspeed_bins))  # Choose a palette with distinct colors

grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
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

        # Interpolate the log-transformed size distribution for this leg
        interp_func = interp1d(ddry_values, log_size_distribution(ddry_values, dryint, D), kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        # Exponentiate the interpolated values to get back to the original scale for plotting
        interpolated_leg_values = np.exp(interpolated_leg_values)

        # Categorize the leg based on windspeed range
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_distributions[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break

# Average and plot the results
plt.figure(figsize=(12, 8))
for idx, ranges in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)

        # Calculate percentiles for the shading
        lower_percentile = np.percentile(grouped_distributions[idx], 25, axis=0)
        upper_percentile = np.percentile(grouped_distributions[idx], 75, axis=0)

        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])  # Count the number of legs
        
        # Plot the average distribution with different colors
        plt.plot(common_bins, avg_distribution, color=colors[idx], label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linewidth=2)

        # Plot the shaded area for the interquartile range with transparency
        plt.fill_between(common_bins, lower_percentile, upper_percentile, color=colors[idx], alpha=0.4, linewidth=0)

        # Optionally plot the percentile lines
        plt.plot(common_bins, lower_percentile, color=colors[idx], linestyle='--', linewidth=1, label=f'{avg_windspeed:.1f} m/s 25th Percentile')
        plt.plot(common_bins, upper_percentile, color=colors[idx], linestyle='--', linewidth=1, label=f'{avg_windspeed:.1f} m/s 75th Percentile')

plt.yscale('log')  # Set y-axis to log scale
plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Size distribution by wind speed', fontweight='bold')
plt.legend(title="Wind speed (m/s)")
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

# Iterate through the entries in filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Match by index

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # x-axis values for this leg
    dryint = entry_dryintercept['dry intercept']

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
        interp_func = interp1d(ddry_values, dryint * np.exp(-ddry_values / D), kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        # Categorize the leg based on windspeed range
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_distributions[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break

# Step 3: Average the results and fit a third-order polynomial using curve_fit
plt.figure(figsize=(12, 8))
for idx, ranges in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        # Average the size distributions for the windspeed bin
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])

        # Fit the third-order polynomial using curve_fit
        popt, pcov = curve_fit(polynomial_fit, common_bins, avg_distribution)

        # Generate the polynomial curve
        poly_curve = polynomial_fit(common_bins, *popt)

        # Calculate percentiles for shading
        lower_percentile = np.percentile(grouped_distributions[idx], 25, axis=0)
        upper_percentile = np.percentile(grouped_distributions[idx], 75, axis=0)

        # Plot the average distribution
        plt.plot(common_bins, avg_distribution, 'o', label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", markersize=5)

        # Plot the fitted third-order polynomial curve
        plt.plot(common_bins, poly_curve, label=f"Poly fit {avg_windspeed:.1f} m/s")

        # Plot the shaded area for the interquartile range
        plt.fill_between(common_bins, lower_percentile, upper_percentile, alpha=0.2)

# plt.yscale('log')
plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Size distribution by wind speed with 3rd order polynomial fit', fontweight='bold')
plt.legend(title="Wind speed (m/s)")
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

# Iterate through the entries in filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Match by index

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # x-axis values for this leg
    dryint = entry_dryintercept['dry intercept']

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
        interp_func = interp1d(ddry_values, dryint * np.exp(-ddry_values / D), kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        # Categorize the leg based on windspeed range
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_distributions[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break

# Step 3: Average the results and fit a fourth-order polynomial using curve_fit
plt.figure(figsize=(12, 8))
for idx, ranges in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        # Average the size distributions for the windspeed bin
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])

        # Fit the fourth-order polynomial using curve_fit
        popt, pcov = curve_fit(polynomial_fit, common_bins, avg_distribution)

        # Generate the polynomial curve
        poly_curve = polynomial_fit(common_bins, *popt)

        # Calculate percentiles for shading
        lower_percentile = np.percentile(grouped_distributions[idx], 25, axis=0)
        upper_percentile = np.percentile(grouped_distributions[idx], 75, axis=0)

        # Plot the average distribution
        plt.plot(common_bins, avg_distribution, 'o', label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", markersize=5)

        # Plot the fitted fourth-order polynomial curve
        plt.plot(common_bins, poly_curve, label=f"4th-order Poly fit {avg_windspeed:.1f} m/s")

        # Plot the shaded area for the interquartile range
        plt.fill_between(common_bins, lower_percentile, upper_percentile, alpha=0.2)

plt.yscale('log') 
plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Size distribution by wind speed with 4th Order Polynomial Fit', fontweight='bold')
plt.legend(title="Wind speed (m/s)")
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

# Iterate through the entries in filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Match by index

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # x-axis values for this leg
    dryint = entry_dryintercept['dry intercept']

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
        interp_func = interp1d(ddry_values, dryint * np.exp(-ddry_values / D), kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        # Categorize the leg based on windspeed range
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_distributions[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break

# Step 3: Average the results and fit a fifth-order polynomial using curve_fit
plt.figure(figsize=(12, 8))
for idx, ranges in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        # Average the size distributions for the windspeed bin
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])

        # Fit the fifth-order polynomial using curve_fit
        popt, pcov = curve_fit(polynomial_fit, common_bins, avg_distribution)

        # Generate the polynomial curve
        poly_curve = polynomial_fit(common_bins, *popt)

        # Calculate percentiles for shading
        lower_percentile = np.percentile(grouped_distributions[idx], 25, axis=0)
        upper_percentile = np.percentile(grouped_distributions[idx], 75, axis=0)

        # Plot the average distribution
        plt.plot(common_bins, avg_distribution, 'o', label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", markersize=5)

        # Plot the fitted fifth-order polynomial curve
        plt.plot(common_bins, poly_curve, label=f"5th-order Poly fit {avg_windspeed:.1f} m/s")

        # Plot the shaded area for the interquartile range
        plt.fill_between(common_bins, lower_percentile, upper_percentile, alpha=0.2)
# plt.yscale('log')  
plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Size Distribution by wind speed with 5th Order Polynomial Fit', fontweight='bold')
plt.legend(title="Wind speed (m/s)")
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

#Let's try to determine the correct number of bins using clustering.
#%%
#Let's first begin with the Elbow Method to determine the optimal number of clusters for K-means clustering for wind speed.
from sklearn.cluster import KMeans

# Extract the windspeed data as a NumPy array (assuming 'df_with_windspeed' is your DataFrame)
windspeed_data = df_with_windspeed['Windspeed'].values.reshape(-1, 1)

# List to store the sum of squared errors for each number of clusters
sse = []

# Define the range of clusters to test (e.g., from 1 to 10 clusters)
cluster_range = range(1, 8)

# Apply K-means clustering for each number of clusters
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)  # Use a fixed random_state for reproducibility
    kmeans.fit(windspeed_data)
    sse.append(kmeans.inertia_)  # SSE (inertia) for each clustering

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
D_data = df_with_windspeed['D'].dropna().values.reshape(-1, 1)  # Drop missing values if any

# List to store the sum of squared errors for each number of clusters
sse_D = []

# Apply K-means clustering for each number of clusters (1 to 8, for example)
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(D_data)
    sse_D.append(kmeans.inertia_)

# Plot the elbow curve
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

Ntd_data = df_with_windspeed['Ntd'].dropna().values.reshape(-1, 1)  # Drop missing values if any

# List to store the sum of squared errors for each number of clusters
sse_Ntd = []

# Apply K-means clustering for each number of clusters (1 to 8, for example)
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(Ntd_data)
    sse_Ntd.append(kmeans.inertia_)

# Plot the elbow curve
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

# Apply K-means clustering
kmeans_windspeed = KMeans(n_clusters=2, random_state=42)
df_with_windspeed['Windspeed_cluster'] = kmeans_windspeed.fit_predict(windspeed_data)

# Visualize the windspeed clusters
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

grouped_distributions_by_cluster = {cluster: [] for cluster in range(2)}  # 2 clusters as determined by elbow method
common_bins = np.linspace(0, 10, 25)
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']

    
    if (date, leg_index) in problematic_set:
        print(f"Skipping problematic leg {date}, index {leg_index}")
        continue

    # Find matching windspeed entry
    windspeed_entry = df_with_windspeed[
        (df_with_windspeed['Date'] == date) &
        (df_with_windspeed['BCB_start'] == BCB_start) &
        (df_with_windspeed['BCB_stop'] == BCB_stop)
    ]

    # Skip if no matching windspeed entry is found
    if windspeed_entry.empty:
        print(f"No windspeed entry found for leg {i}. Skipping.")
        continue

    # Extract windspeed and cluster
    windspeed = windspeed_entry['Windspeed'].values[0]
    cluster = windspeed_entry['Windspeed_cluster'].values[0]

    # Extract and interpolate size distribution
    ddry_values = np.array(entry_ddry['filtered_ddry'])
    dryint = filtered_master_BCB_dryintercept[i]['dry intercept']
    D = entry_ddry['D']

    try:
        interp_func = interp1d(ddry_values, size_distribution(ddry_values, dryint, D), kind='linear', fill_value='extrapolate')
        interpolated_distribution = interp_func(common_bins)
    except Exception as e:
        print(f"Interpolation failed for leg {i}. Error: {e}")
        continue

    # Append to the correct cluster
    grouped_distributions_by_cluster[cluster].append(interpolated_distribution)

# Debug: Print cluster sizes
for cluster, distributions in grouped_distributions_by_cluster.items():
    print(f"Cluster {cluster}: {len(distributions)} legs")

# Plot average size distributions for each cluster
plt.figure(figsize=(12, 8))
for cluster, distributions in grouped_distributions_by_cluster.items():
    if distributions:
        avg_distribution = np.mean(distributions, axis=0)
        plt.plot(common_bins, avg_distribution, label=f"Cluster {cluster} ({len(distributions)} legs)")

plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Size distributions by wind speed clusters', fontweight='bold')
plt.legend(title="Wind speed Cluster")
plt.tight_layout()
plt.show()

#%%
#Quantile based binning for 4 quantiles 
problematic_set = set(problematic_legs)

# Define your common bins
common_bins = np.linspace(0, 10, 25)

# Define the number of quantile bins
num_bins = 4  # Change this to adjust the number of bins

# Create windspeed bins using quantiles
df_with_windspeed['Windspeed_bin'] = pd.qcut(df_with_windspeed['Windspeed'], q=num_bins, labels=False)

# Optional: Get bin edges for reference
bin_edges = df_with_windspeed['Windspeed'].quantile(np.linspace(0, 1, num_bins + 1)).values
print("Bin edges:", bin_edges)

# Group by windspeed bin and calculate means
grouped_data = df_with_windspeed.groupby('Windspeed_bin').agg({'Ntd': 'mean', 'D': 'mean'}).reset_index()

# Plot averaged results by windspeed bin
plt.figure(figsize=(10, 8))
plt.scatter(grouped_data['D'], grouped_data['Ntd'], c='blue', s=100)
plt.xscale('log')
plt.yscale('log')

# Add labels and title
plt.xlabel('Average Slope (D)', fontsize=14, fontweight='bold')
plt.ylabel('Average Total droplet concentration (Ntd)', fontsize=14, fontweight='bold')
plt.title('Average Ntd vs Average D by Windspeed Quantile Bins', fontsize=14, fontweight='bold')
plt.show()

# Initialize storage for distributions and mean wind speeds
grouped_distributions = {i: [] for i in range(num_bins)}
mean_windspeeds = {i: [] for i in range(num_bins)}
std_distributions = {i: [] for i in range(num_bins)}  # To store standard deviations

# Iterate through the entries in filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Match by index

    
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']  # Extract Leg_index

   
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

    # Check if windspeed_entry is not empty
    if windspeed_entry.empty:
        print(f"No matching windspeed entry found for Date: {date}, BCB_start: {BCB_start}, BCB_end: {BCB_stop}")
        continue  # Skip to the next iteration if no match found

    windspeed = windspeed_entry['Windspeed'].values[0]

    # Interpolate the log-transformed size distribution for this leg
    interp_func = interp1d(ddry_values, log_size_distribution(ddry_values, dryint, D), 
                           kind='linear', fill_value='extrapolate')
    interpolated_leg_values = interp_func(common_bins)

    # Exponentiate the interpolated values to get back to the original scale for plotting
    interpolated_leg_values = np.exp(interpolated_leg_values)

   
    windspeed_bin = pd.qcut(df_with_windspeed['Windspeed'], q=num_bins, labels=False, duplicates='drop')
    windspeed_bin_index = windspeed_bin[windspeed_bin.index[df_with_windspeed['Windspeed'] == windspeed]].values

    if len(windspeed_bin_index) == 0:
        print(f"No windspeed bin found for windspeed: {windspeed}")
        continue  

    windspeed_bin = windspeed_bin_index[0]

    if windspeed_bin in grouped_distributions:
        grouped_distributions[windspeed_bin].append(interpolated_leg_values)
        mean_windspeeds[windspeed_bin].append(windspeed)
        std_distributions[windspeed_bin].append(np.std(interpolated_leg_values))  # Store std deviation

# Average and plot the results
plt.figure(figsize=(12, 8))
for idx in range(num_bins):
    if grouped_distributions[idx]:
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)
        avg_windspeed = np.mean(mean_windspeeds[idx]) if mean_windspeeds[idx] else 0
        num_legs = len(grouped_distributions[idx])  # Count the number of legs

        # Calculate standard deviation across the distributions for uncertainty shading
        std_dev = np.std(grouped_distributions[idx], axis=0)

        
        plt.plot(common_bins, avg_distribution, label=f"Bin {idx + 1}: {avg_windspeed:.1f} m/s, n={num_legs} legs")

        # Add shaded area for uncertainty (mean ± std dev)
        plt.fill_between(common_bins, 
                         avg_distribution - std_dev, 
                         avg_distribution + std_dev, 
                         alpha=0.2)  # Adjust alpha for transparency

plt.yscale('log')  
plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Size distribution by wind speed Quantile Bins', fontweight='bold')
plt.legend(title="wind speed Bins")
plt.tight_layout()
plt.show()
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

            
            filtered_NT = None
            D = None

            
            if date in master_BCB_exponential:
                for exp_params in master_BCB_exponential[date]:
       
                    if exp_params['BCB_start'] == BCB_start and exp_params['BCB_stop'] == BCB_stop:
                        D = exp_params['D']
                        print(f"Found D for Date: {date}, BCB_start: {BCB_start}, BCB_stop: {BCB_stop}, D: {D}")
                        break

           
            for filtered_NT_data in filtered_master_BCB_NT:
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
##trying to fix size distribution and add ntd 

def size_distribution(d, dryint, D):
    return dryint * np.exp(-d / D)

dmin = 2  
dmax = np.inf 

integrated_concentrations = []
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
    
#     plt.plot(common_bins, dist_values, label=f"Date: {date}, Leg: {leg_index} ({BCB_start} - {BCB_stop})")

# plt.ylabel('Total droplet concentration (/cm^3/um)', fontweight='bold')
# plt.xlabel('Bin diameter (um)', fontweight='bold')
# plt.title('Below cloud base January - June 2022', fontweight='bold')
# plt.tight_layout()
# plt.show()
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
def fit_function(x, D):
    return dryint * np.exp(-x / D)

common_bins = np.linspace(0, 10, 10)  
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

#Average each particle size bin for plotting and fit the custom exponential model
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

common_bins = np.linspace(0, 10, 10)


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

common_bins = np.linspace(0, 10, 10)
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

##calculating area under the curve for total concentration

def fit_function(x, dryint, D):
    return dryint * np.exp(-x / D)


common_bins = np.linspace(2.5, 10, 10)  # Adjust common bins to start from 2.5 um
# windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]
winspeed_bins=[(0,3),(3.001,6),(6.001,8),(8.001,np.inf)]
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

            # Calculate the total concentration (integral of N(d) * dd)
            total_concentration = np.trapz(fitted_values, common_bins)
            print(f"Total concentration for windspeed {avg_windspeed:.1f} m/s: {total_concentration:.2f} / cm³")
            
            # Alternatively, use quad for continuous integration if necessary
            # total_concentration_continuous, _ = quad(fit_function, 2.5, np.max(common_bins), args=(dryint_fit, D_fit))
            # print(f"Continuous integration result: {total_concentration_continuous:.2f} / cm³")

        except RuntimeError:
            print(f"Could not fit the exponential model for windspeed {avg_windspeed:.1f} m/s")
        
plt.yscale('log')

plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Average size distribution by wind speed', fontweight='bold')
plt.legend(title="Average wind speed (m/s)")
plt.tight_layout()
plt.show()

# %%


common_bins = np.linspace(0, 10, 25)
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

# %%
##trying third order polynomial to the fitted distributions

common_bins = np.linspace(0, 10, 25)


# windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]
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

        # Fit a cubic polynomial to the log of the interpolated values
        def poly_fit(x, a, b, c, d):
            return a * x**3 + b * x**2 + c * x + d

        try:
            log_values = np.log(interpolated_leg_values + 1e-10)  
            popt, _ = curve_fit(poly_fit, common_bins, log_values)

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

# Loop through each windspeed bin and plot the fitted average distributions
for idx, (low, high) in enumerate(windspeed_bins):
    if fitted_params[idx]:
        avg_params = averaged_params[idx]
        fitted_curve_log = poly_fit(common_bins, *avg_params)  # Get the fitted values in log space
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

for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  

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
##third order with 0.90 specification  

common_bins = np.linspace(0, 10, 25)

# windspeed_bins = [(0, 2.1), (2.2, 5.0), (5.1, 5.6), (5.7, 8.5), (8.6, np.inf)]
windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]

averaged_distributions = {i: [] for i in range(len(windspeed_bins))}


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

    # Find the corresponding entry in df_combined based on Date, BCB_start, and BCB_stop
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

        # Determine the windspeed bin for this leg and store the interpolated values
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                averaged_distributions[idx].append(interpolated_leg_values)  # Append the interpolated values
                break

# Step 2: Average the raw distributions for each windspeed bin
averaged_params = []
plt.figure(figsize=(12, 8))

for idx in range(len(windspeed_bins)):
    if averaged_distributions[idx]:  # Only proceed if there are distributions
        avg_distribution = np.mean(averaged_distributions[idx], axis=0)

        # Fit a third-order polynomial to the averaged distribution
        def polynomial_fit(x, a, b, c, d):
            return a * x**3 + b * x**2 + c * x + d

        try:
            popt, _ = curve_fit(polynomial_fit, common_bins, avg_distribution, p0=[1, 1, 1, 1])
            fitted_curve = polynomial_fit(common_bins, *popt)

            # Calculate R²
            residuals = avg_distribution - fitted_curve
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((avg_distribution - np.mean(avg_distribution))**2)
            r_squared = 1 - (ss_res / ss_tot)

            # Ensure R² is greater than 0.95
            if r_squared > 0.90:
                plt.plot(common_bins, fitted_curve, label=f"Windspeed {windspeed_bins[idx]} m/s (R²={r_squared:.2f})")
            else:
                print(f"R² value for windspeed bin {idx} is below threshold: {r_squared:.2f}")

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
from numpy.polynomial import Polynomial

windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)
]

fitted_params = [[] for _ in range(len(windspeed_bins))]
mean_windspeeds = [[] for _ in range(len(windspeed_bins))]


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
