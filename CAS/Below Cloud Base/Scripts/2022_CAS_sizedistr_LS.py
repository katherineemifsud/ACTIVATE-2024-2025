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
# %%
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
#%%
#We are only focused on bins 12-30 because we want coarse mode aerosol at 2.5-50 um diameter 
bin_center=[ 
2.25, 2.75, 3.25, 3.75, 4.5, 5.75, 6.85, 7.55, 
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
#%%
# #Import the instrument data for the cloud-aerosol spectrometer

# #Make sure to only work with bins 12-30 for the coarse mode aerosol
# bin_name = ['CAS_Bin00', 'CAS_Bin01', 'CAS_Bin02', 'CAS_Bin03',
#             'CAS_Bin04', 'CAS_Bin05', 'CAS_Bin06', 'CAS_Bin07', 
#             'CAS_Bin08', 'CAS_Bin09', 'CAS_Bin10', 'CAS_Bin11',
#             'CAS_Bin12' ,'CAS_Bin13', 'CAS_Bin14', 'CAS_Bin15', 
#              'CAS_Bin16', 'CAS_Bin17', 
#             'CAS_Bin18', 'CAS_Bin19', 'CAS_Bin20', 'CAS_Bin21', 'CAS_Bin22', 
#              'CAS_Bin23', 'CAS_Bin24', 'CAS_Bin25', 'CAS_Bin26',
#              'CAS_Bin27', 'CAS_Bin28', 'CAS_Bin29']

# CAS = []

# dates_CAS = [
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

# for date in dates_CAS:

#     dataset = {'Date': date, 'Clear Means': [], 'Cloud Means': []}  # Initialize a dataset dictionary
#     datestr = date.replace('-', '')
#     fname_CAS = sorted(glob.glob(f'/home/disk/eos4/kathem24/activate/data/cloudaerospect/2022csv/ACTIVATE-LARGE-CAS_HU25_{datestr}_R*.csv'), reverse=True)
    
#     run = 1
#     for file_path in fname_CAS:
#         nums_file_paths = len(fname_CAS)

#         if date <= ('2022-03-29'):
#             df_CAS = pd.read_csv(file_path, skiprows= 71, quoting=csv.QUOTE_NONE)
#         elif date >= ('2022-05-05'):
#             df_CAS = pd.read_csv(file_path, skiprows= 72, quoting=csv.QUOTE_NONE)
        
        
#         for bin_ in bin_name:
#             if bin_ in df_CAS.columns:
#                 df_CAS.columns = df_CAS.columns.str.strip('"')
#                 df_CAS[bin_] = pd.to_numeric(df_CAS[bin_], errors='coerce')
#                 df_CAS.replace([-9999, -9999.00], np.NaN, inplace=True)
#         for col in ['Time_mid', 'LWC_CAS', 'CAS_Bin00', 'CAS_Bin01', 'CAS_Bin02', 
#                     'CAS_Bin03',
#             'CAS_Bin04', 'CAS_Bin05', 'CAS_Bin06', 'CAS_Bin07', 
#             'CAS_Bin08', 'CAS_Bin09', 'CAS_Bin10', 'CAS_Bin11',
#             'CAS_Bin12', 'CAS_Bin13', 'CAS_Bin14', 
#                     'CAS_Bin15', 'CAS_Bin16', 'CAS_Bin17', 
#                     'CAS_Bin18', 'CAS_Bin19', 'CAS_Bin20', 
#                     'CAS_Bin21', 'CAS_Bin22', 'CAS_Bin23', 
#                     'CAS_Bin24', 'CAS_Bin25', 'CAS_Bin26',
#                     'CAS_Bin27', 'CAS_Bin28', 'CAS_Bin29']:
#             if df_CAS[col].dtype == 'O':  
#                 df_CAS[col] = df_CAS[col].str.strip('"')
        
#         df_CAS['Time_mid']= pd.to_numeric(df_CAS['Time_mid'], errors='coerce')
#         df_CAS['CAS_Bin12']= pd.to_numeric(df_CAS['CAS_Bin12'], errors='coerce')
#         df_CAS['CAS_Bin13']= pd.to_numeric(df_CAS['CAS_Bin13'], errors='coerce')
#         df_CAS['CAS_Bin14']= pd.to_numeric(df_CAS['CAS_Bin14'], errors='coerce')
#         df_CAS['CAS_Bin15']= pd.to_numeric(df_CAS['CAS_Bin15'], errors='coerce')
#         df_CAS['CAS_Bin16']= pd.to_numeric(df_CAS['CAS_Bin16'], errors='coerce')
#         df_CAS['CAS_Bin17']= pd.to_numeric(df_CAS['CAS_Bin17'], errors='coerce')
#         df_CAS['CAS_Bin18']= pd.to_numeric(df_CAS['CAS_Bin18'], errors='coerce')
#         df_CAS['CAS_Bin19']= pd.to_numeric(df_CAS['CAS_Bin19'], errors='coerce')
#         df_CAS['CAS_Bin20']= pd.to_numeric(df_CAS['CAS_Bin20'], errors='coerce')
#         df_CAS['CAS_Bin21']= pd.to_numeric(df_CAS['CAS_Bin21'], errors='coerce')
#         df_CAS['CAS_Bin22']= pd.to_numeric(df_CAS['CAS_Bin22'], errors='coerce')
#         df_CAS['CAS_Bin23']= pd.to_numeric(df_CAS['CAS_Bin23'], errors='coerce')
#         df_CAS['CAS_Bin24']= pd.to_numeric(df_CAS['CAS_Bin24'], errors='coerce')
#         df_CAS['CAS_Bin25']= pd.to_numeric(df_CAS['CAS_Bin25'], errors='coerce')
#         df_CAS['CAS_Bin26']= pd.to_numeric(df_CAS['CAS_Bin26'], errors='coerce')
#         df_CAS['CAS_Bin27']= pd.to_numeric(df_CAS['CAS_Bin27'], errors='coerce')
#         df_CAS['CAS_Bin28']= pd.to_numeric(df_CAS['CAS_Bin28'], errors='coerce')
#         df_CAS['CAS_Bin29']= pd.to_numeric(df_CAS['CAS_Bin29'], errors='coerce')
#         df_CAS['CAS_Bin00']=pd.to_numeric(df_CAS['CAS_Bin00'], errors='coerce')
#         df_CAS['CAS_Bin01']= pd.to_numeric(df_CAS['CAS_Bin01'], errors='coerce')
#         df_CAS['CAS_Bin02']= pd.to_numeric(df_CAS['CAS_Bin02'], errors='coerce')
#         df_CAS['CAS_Bin03']= pd.to_numeric(df_CAS['CAS_Bin03'], errors='coerce')
#         df_CAS['CAS_Bin04']= pd.to_numeric(df_CAS['CAS_Bin04'], errors='coerce')
#         df_CAS['CAS_Bin05']= pd.to_numeric(df_CAS['CAS_Bin05'], errors='coerce')
#         df_CAS['CAS_Bin06']= pd.to_numeric(df_CAS['CAS_Bin06'], errors='coerce')
#         df_CAS['CAS_Bin07']= pd.to_numeric(df_CAS['CAS_Bin07'], errors='coerce')
#         df_CAS['CAS_Bin08']= pd.to_numeric(df_CAS['CAS_Bin08'], errors='coerce')
#         df_CAS['CAS_Bin09']= pd.to_numeric(df_CAS['CAS_Bin09'], errors='coerce')
#         df_CAS['CAS_Bin10']= pd.to_numeric(df_CAS['CAS_Bin10'], errors='coerce')
#         df_CAS['CAS_Bin11']= pd.to_numeric(df_CAS['CAS_Bin11'], errors='coerce')
      
        

#         if nums_file_paths==2:
#             if run==1:
#                 df4 = df_CAS 
#             elif run==2:
#                 df5 = df_CAS 
#                 frames = [df5,df4]
#                 df_CAS = pd.concat(frames)
#                 CAS.append(df_CAS)
#                 break

#         if nums_file_paths ==1:
#             # print("YESSSSSSSSSSSSSSSSSSSSSSSSSSSSs")
#             CAS.append(df_CAS)

#         run = run+1 
#%%
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

    
    CAS_flight['Time_mid'] = pd.to_numeric(CAS_flight['Time_mid'], errors='coerce')
    twoDS_flight['Time_Start'] = pd.to_numeric(twoDS_flight['Time_Start'], errors='coerce')

    
    CAS_times = CAS_flight['Time_mid'].values
    CAS_lwc = CAS_flight['LWC_CAS'].values
    CAS_bins = {f'CAS_Bin{bin_label:02d}': CAS_flight[f'CAS_Bin{bin_label:02d}'].values for bin_label in range(12, 30)}

    TwoDS_times = twoDS_flight['Time_Start'].values
    TwoDS_N_total = twoDS_flight['N-total_2DS'].values

    total_BCB_means = []

    for k in range(len(BCB_start)):
        start20 = BCB_start[k]
        end20 = BCB_stop[k]

        CAS_indices_in_range = np.where((CAS_times >= start20) & (CAS_times <= end20))[0]
        TwoDS_indices_in_range = np.where((TwoDS_times >= start20) & (TwoDS_times <= end20))[0]

        if CAS_indices_in_range.size > 0 and TwoDS_indices_in_range.size > 0:
            data_labels = []
            BCB_means = []

            for CAS_idx, TwoDS_idx in zip(CAS_indices_in_range, TwoDS_indices_in_range):
                lwc_val = CAS_lwc[CAS_idx]
                N_val = TwoDS_N_total[TwoDS_idx]

                
                lwc_label = 'Y' if 0 <= lwc_val <= 0.0025 else 'N'
                N_label = 'Y' if 0 <= N_val <= 100 else 'N'
                label = 'Y' if lwc_label == 'Y' and N_label == 'Y' else 'N'
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
leg_count = Counter([leg['Date'] for leg in leg_info])
print("Number of legs associated with each date:")
total_legs = 0
for date, count in sorted(leg_count.items()):
    print(f"Date: {date}, Number of Legs: {count}")
    total_legs += count
print(f"\nTotal number of legs: {total_legs}")
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
#average ambient 
sum_bin_means_CAS = np.zeros(len(bin_center))
count_bin_means_CAS= np.zeros(len(bin_center))
for entry in Y_BCB_calc:
    bin_means = np.array([entry.get(f'Bin{i}_Y_mean', np.nan) for i in range(12, 30)], dtype=float)
    valid_indices = (bin_means > 0) & ~np.isnan(bin_means)

    sum_bin_means_CAS[valid_indices] += bin_means[valid_indices]
    count_bin_means_CAS[valid_indices] += 1

average_bin_means_CAS = np.divide(sum_bin_means_CAS, count_bin_means_CAS, where=count_bin_means_CAS > 0)

plt.figure(figsize=(8, 6))
plt.plot(bin_center, average_bin_means_CAS, color='red', linewidth=2, label='Average CAS Size Distribution')

plt.xlabel("Deliquesced Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"CAS Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xlim(0, 40)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("CAS Average Ambient Below Cloud Base Size Distribution\n January - June 2022", fontsize=14, fontweight="bold")
plt.show()
#%%

#%%
#Calculating total number concentration 
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

#%%

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

for entry in Y_BCB_calc:  # Iterate over the list directly
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
# %%
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
#Histogram of gRH values
gRH_values = [
    leg['gRh_mean'][0] for flight in filtered_master_BCB_gRH for leg in flight if not np.isnan(leg['gRh_mean'][0])
]

# Create histogram
plt.figure(figsize=(8, 6))
plt.hist(gRH_values, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel('Growth factor (gRH)', fontsize=15, fontweight='bold')
plt.ylabel('Frequency of flight legs', fontsize=15, fontweight='bold')
plt.title('Applying the growth factor equation to RH mean values', fontweight='bold', fontsize=15)
plt.xticks(fontweight='bold', fontsize=14)
plt.yticks(fontweight='bold', fontsize=14)

plt.show()

# %%
#filtered dry intercept calculation

# Create a dictionary for quick lookup of ambient intercepts
ambient_fits_dict = {(fit['Date'], fit['BCB_start'], fit['BCB_stop']): fit for fit in ambient_fits}

# Dictionary to store filtered dry intercept results
filtered_master_BCB_interceptdry_dict = {}

# Flatten master_min_gRH if it's a list of lists
if isinstance(filtered_master_BCB_gRH[0], list):
    filtered_master_BCB_gRH = [item for sublist in filtered_master_BCB_gRH for item in sublist]

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

    # Find corresponding ambient exponential parameters
    if key in ambient_fits_dict:
        n0 = ambient_fits_dict[key]['Intercept_n0']  # Extract n0 from ambient_fits
        
        # Calculate dry intercept
        dryintercept = n0 / gRh_mean if gRh_mean > 0 else np.nan

        # Store the result in the dictionary
        filtered_master_BCB_interceptdry_dict[key] = {
            'Date': date,
            'BCB_start': BCB_start,
            'BCB_stop': BCB_stop,
            'Rh_mean': entry['Rh_mean'],
            'gRh_mean': entry['gRh_mean'],
            'dry intercept': dryintercept
        }

# Convert dictionary to a list
filtered_master_BCB_dryintercept = list(filtered_master_BCB_interceptdry_dict.values())

print(f"Length of filtered_master_BCB_dryintercept: {len(filtered_master_BCB_dryintercept)}")

# Histogram of dry intercept values
dryintercept_values = [
    leg['dry intercept'] for leg in filtered_master_BCB_dryintercept if not np.isnan(leg['dry intercept'])
]

plt.figure(figsize=(8, 6))
plt.hist(dryintercept_values, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel(r"$\mathbf{Dry\ intercept\ (cm^{-3}\ \mu m^{-1})}$", fontsize=15)
plt.ylabel('Frequency', fontsize=15, fontweight='bold')
plt.title('Dry intercept (gRH / N0)', fontweight='bold', fontsize=16)
plt.grid(True)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.show()
#%%

# %%
filtered_master_BCB_ddry = []

# Iterate over each entry in filtered_master_BCB_gRH
for entry in filtered_master_BCB_gRH:
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    gRh_mean = entry['gRh_mean'][0]  # Assuming it's stored as a list

    # Compute dry bin centers
    if gRh_mean > 0:
        ddry_values = np.array([D_amb / gRh_mean for D_amb in bin_center])
    else:
        ddry_values = np.full(len(bin_center), np.nan)
        print(f"Skipping division for {date}, {BCB_start}-{BCB_stop} due to invalid gRh_mean.")

    # Compute bin widths for dry size distribution (∆Ddry = Ddry[i+1] - Ddry[i])
    ddry_bin_widths = np.diff(ddry_values, append=np.nan)  # Append NaN to keep array size consistent

    # Find the corresponding ambient size distribution
    raw_concentrations = next(
        (leg for leg in Y_BCB_calc if leg['Date'] == date and leg['BCB_start'] == BCB_start and leg['BCB_stop'] == BCB_stop),
        None
    )

    if raw_concentrations:
        # Extract raw bin concentrations (dN/dDambient)
        dN_dD_ambient = np.array([raw_concentrations.get(f'Bin{i}_Y_mean', np.nan) for i in range(12, 30)], dtype=float)

        # Apply the transformation using the correct bin widths
        dN_dD_dry = np.where(
        (~np.isnan(dN_dD_ambient)) & (~np.isnan(ddry_bin_widths)) & (gRh_mean > 0),
        dN_dD_ambient * (np.array(bin_center) / ddry_values) * (np.diff(bin_center, append=np.nan) / ddry_bin_widths),
        np.nan
    )

    else:
        dN_dD_dry = np.full(len(bin_center), np.nan)
        print(f"Missing raw size distribution for {date}, {BCB_start}-{BCB_stop}")

    # Store the raw dry size distribution
    filtered_master_BCB_ddry.append({
        'Date': date,
        'BCB_start': BCB_start,
        'BCB_stop': BCB_stop,
        'ddry': ddry_values.tolist(),
        'dN/dDdry': dN_dD_dry.tolist(),
        'ddry_bin_widths': ddry_bin_widths.tolist(),  # Store bin widths separately
        'gRh_mean': gRh_mean
    })

print(f"Length of filtered_master_BCB_ddry: {len(filtered_master_BCB_ddry)}")
#%%

from scipy.interpolate import interp1d

# Define common bin centers for interpolation
common_bins = np.linspace(2, 25, 25)  # Adjust bin range and count as needed

plt.figure(figsize=(8, 6))

# Loop through each dry size distribution
for entry in filtered_master_BCB_ddry:
    ddry_values = np.array(entry['ddry'])  # The unique dry bin centers for this leg
    dN_dD_dry = np.array(entry['dN/dDdry'])  # The corresponding concentration values

    # Remove NaN values before interpolation
    valid_indices = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry)
    if np.sum(valid_indices) < 2:
        continue  # Skip if not enough valid points for interpolation

    # Interpolate onto the common bins
    interp_func = interp1d(ddry_values[valid_indices], dN_dD_dry[valid_indices], 
                           kind='linear', bounds_error=False, fill_value=np.nan)
    interpolated_dN_dD_dry = interp_func(common_bins)

    # Plot the interpolated dry size distribution
    plt.plot(common_bins, interpolated_dN_dD_dry, color='black', alpha=0.2)

# Formatting and labels
plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base January - June 2022\n Raw Dry Size Distributions", fontsize=14, fontweight="bold")
plt.show()
#%%%
#Removing the 0s

# Define common bin centers for interpolation
common_bins = np.linspace(2, 25, 25)  # Adjust bin range and count as needed

plt.figure(figsize=(8, 6))

# Loop through each dry size distribution
for entry in filtered_master_BCB_ddry:
    ddry_values = np.array(entry['ddry'])  # Unique dry bin centers for this leg
    dN_dD_dry = np.array(entry['dN/dDdry'])  # Corresponding concentration values

    # Remove NaN values before interpolation
    valid_indices = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry)
    if np.sum(valid_indices) < 2:
        continue  # Skip if not enough valid points for interpolation

    # Interpolate onto the common bins
    interp_func = interp1d(ddry_values[valid_indices], dN_dD_dry[valid_indices], 
                           kind='linear', bounds_error=False, fill_value=np.nan)
    interpolated_dN_dD_dry = interp_func(common_bins)

    # Mask: Remove NaNs and zero values
    valid_interpolated_indices = (interpolated_dN_dD_dry > 0) & ~np.isnan(interpolated_dN_dD_dry)
    filtered_bins = common_bins[valid_interpolated_indices]
    filtered_dN_dD_dry = interpolated_dN_dD_dry[valid_interpolated_indices]

    if len(filtered_bins) > 0:  # Only plot if valid data exists
        plt.plot(filtered_bins, filtered_dN_dD_dry, color='black', alpha=0.2)

# Formatting and labels
plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base January - June 2022\n Raw Dry Size Distributions", fontsize=14, fontweight="bold")
plt.show()

#%%
# Check transformation step
for entry in filtered_master_BCB_ddry[:5]:  # Checking first 5 entries
    print(f"Date: {entry['Date']}, Start: {entry['BCB_start']}, Stop: {entry['BCB_stop']}")
    print("  gRh_mean:", entry['gRh_mean'])
    print("  dN/dDdry first 5 bins:", entry['dN/dDdry'][:5])
    print("  ddry_bin_widths first 5 bins:", entry['ddry_bin_widths'][:5])
    print("  Original bin widths:", np.diff(bin_center, append=np.nan)[:5])
    print("  -----")
#%%
for entry in filtered_master_BCB_ddry[:5]:  # Check first 5 entries
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    gRh_mean = entry['gRh_mean']

    raw_concentrations = next(
        (leg for leg in Y_BCB_calc if leg['Date'] == date and leg['BCB_start'] == BCB_start and leg['BCB_stop'] == BCB_stop),
        None
    )

    if raw_concentrations:
        dN_dD_ambient = np.array([raw_concentrations.get(f'Bin{i}_Y_mean', np.nan) for i in range(12, 30)])

        print(f"Date: {date}, Start: {BCB_start}, Stop: {BCB_stop}")
        print(f"  gRh_mean: {gRh_mean}")
        print(f"  dN/dDambient first 5 bins: {dN_dD_ambient[:5]}")
        print(f"  dN/dDdry first 5 bins: {entry['dN/dDdry'][:5]}")
        print(f"  Ratio (dN/dDdry / dN/dDambient): {np.array(entry['dN/dDdry'][:5]) / dN_dD_ambient[:5]}")
        print("  -----")
#%%
#Fitting exponential to the dry distributions

def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

dry_exponential_fits = []

plt.figure(figsize=(8, 6))

for entry in filtered_master_BCB_ddry:
    ddry_values = np.array(entry['ddry'])
    dN_dD_dry = np.array(entry['dN/dDdry'])

    valid_indices = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry)
    if np.sum(valid_indices) < 5:  
        continue

    try:
        popt, _ = curve_fit(exponential, ddry_values[valid_indices], dN_dD_dry[valid_indices], p0=(1, 5), maxfev=5000)
        n0, D = popt

        dry_exponential_fits.append({
            'Date': entry['Date'],
            'BCB_start': entry['BCB_start'],
            'BCB_stop': entry['BCB_stop'],
            'Dry_Intercept_n0': n0,
            'Dry_E_folding_D': D
        })

        x_fit = np.linspace(min(ddry_values[valid_indices]), max(ddry_values[valid_indices]), 100)
        y_fit = exponential(x_fit, *popt)

        plt.plot(x_fit, y_fit, color='black', alpha=0.2)

    except RuntimeError:
        print(f"Fit could not be performed for date {entry['Date']}")

plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(1e-33, 1e3)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base January - June 2022\n Fitted Dry Size Distributions", fontsize=14, fontweight="bold")
plt.show()
print(f"Total successful dry exponential fits: {len(dry_exponential_fits)}")
#%%
#removing those two weird lines 

def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

dry_exponential_fits = []

plt.figure(figsize=(8, 6))

for entry in filtered_master_BCB_ddry:
    ddry_values = np.array(entry['ddry'])
    dN_dD_dry = np.array(entry['dN/dDdry'])

    # Ensure valid data points
    valid_indices = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry) & (dN_dD_dry > 0)
    
    if np.sum(valid_indices) < 2:  
        continue

    try:
        popt, _ = curve_fit(exponential, ddry_values[valid_indices], dN_dD_dry[valid_indices], 
                            p0=(1, 5), maxfev=5000)
        n0, D = popt

        dry_exponential_fits.append({
            'Date': entry['Date'],
            'BCB_start': entry['BCB_start'],
            'BCB_stop': entry['BCB_stop'],
            'Dry_Intercept_n0': n0,
            'Dry_E_folding_D': D
        })

        x_fit = np.linspace(min(ddry_values[valid_indices]), max(ddry_values[valid_indices]), 100)
        y_fit = exponential(x_fit, *popt)

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
plt.title("Below Cloud Base January - June 2022\n Exponential Fitted Dry Size Distributions", fontsize=14, fontweight="bold")

plt.show()

print(f"Total successful dry exponential fits: {len(dry_exponential_fits)}")
#%%
#fitting an exponential to dry distributions, removing those two weird lines, and removing extreme slopes after 10 um slope


# Define exponential function
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

dry_exponential_fits = []

plt.figure(figsize=(8, 6))

for entry in filtered_master_BCB_ddry:
    ddry_values = np.array(entry['ddry'])
    dN_dD_dry = np.array(entry['dN/dDdry'])

    # Ensure valid data points
    valid_indices = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry) & (dN_dD_dry > 0)
    
    if np.sum(valid_indices) < 2:  
        continue

    try:
        # Fit exponential function
        popt, _ = curve_fit(exponential, ddry_values[valid_indices], dN_dD_dry[valid_indices], 
                            p0=(1, 5), maxfev=5000)
        n0, D = popt

        # **Filter out extreme slopes where D > 20 µm**
        if D > 20:
            continue  # Skip this fit

        dry_exponential_fits.append({
            'Date': entry['Date'],
            'BCB_start': entry['BCB_start'],
            'BCB_stop': entry['BCB_stop'],
            'Dry_Intercept_n0': n0,
            'Dry_E_folding_D': D
        })

        # Generate fitted curve
        x_fit = np.linspace(min(ddry_values[valid_indices]), max(ddry_values[valid_indices]), 100)
        y_fit = exponential(x_fit, *popt)

        # Plot only valid fits
        if np.all(y_fit > 1e-33):
            plt.plot(x_fit, y_fit, color='black', alpha=0.2)

    except RuntimeError:
        print(f"Fit could not be performed for date {entry['Date']}")

# Formatting
plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-33, 10**1)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base January - June 2022\n Fitted Dry Size Distributions", fontsize=14, fontweight="bold")

plt.show()

print(f"Total successful dry exponential fits (D ≤ 20 µm): {len(dry_exponential_fits)}")

#%%
#only to 10um 

def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

dry_exponential_fits_10 = []

plt.figure(figsize=(8, 6))

for entry in filtered_master_BCB_ddry:
    ddry_values = np.array(entry['ddry'])
    dN_dD_dry = np.array(entry['dN/dDdry'])

    # Filter data to only include bins ≤ 10 µm
    valid_indices = (ddry_values <= 10) & ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry)

    # If there are no valid points within ≤ 10 µm, store NaNs but do NOT skip
    if np.sum(valid_indices) == 0:
        dry_exponential_fits_10.append({
            'Date': entry['Date'],
            'BCB_start': entry['BCB_start'],
            'BCB_stop': entry['BCB_stop'],
            'Dry_Intercept_n0': np.nan,
            'Dry_E_folding_D': np.nan
        })
        continue  # Move to next entry, but store NaNs instead of skipping

    try:
        # Fit the exponential only using data up to 10 µm
        popt, _ = curve_fit(exponential, ddry_values[valid_indices], dN_dD_dry[valid_indices], p0=(1, 5), maxfev=5000)
        n0, D = popt

    except RuntimeError:
        print(f"Fit could not be performed for date {entry['Date']}")
        n0, D = np.nan, np.nan  # Store NaN if fitting fails

    # Store fitted parameters (including NaNs for failed fits)
    dry_exponential_fits_10.append({
        'Date': entry['Date'],
        'BCB_start': entry['BCB_start'],
        'BCB_stop': entry['BCB_stop'],
        'Dry_Intercept_n0': n0,
        'Dry_E_folding_D': D
    })

    # Generate fitted curve only up to 10 µm if the fit was successful
    if not np.isnan(n0) and not np.isnan(D):
        x_fit = np.linspace(min(ddry_values[valid_indices]), 10, 100)
        y_fit = exponential(x_fit, n0, D)
        plt.plot(x_fit, y_fit, color='black', alpha=0.2)

# Formatting
plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=12, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=12, fontweight="bold")
plt.yscale("log")
plt.ylim(1e-33, 1e3)
plt.xticks(fontweight="bold", fontsize=10)
plt.yticks(fontweight="bold", fontsize=10)
plt.title("Below Cloud Base January - June 2022\n Fitted Dry Size Distributions (≤10 µm)", fontsize=14, fontweight="bold")

plt.show()
print(f"Total successful dry exponential fits: {len([fit for fit in dry_exponential_fits if not np.isnan(fit['Dry_Intercept_n0'])])}")
#%%
#removing those 2 lines

# Define the exponential function
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

dry_exponential_fits_10 = []

plt.figure(figsize=(8, 6))

for entry in filtered_master_BCB_ddry:
    ddry_values = np.array(entry['ddry'])
    dN_dD_dry = np.array(entry['dN/dDdry'])

    # Filter data to only include bins ≤ 10 µm
    valid_indices = (ddry_values <= 10) & ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry)

    # If no valid points within ≤ 10 µm, store NaNs but do NOT skip
    if np.sum(valid_indices) == 0:
        dry_exponential_fits_10.append({
            'Date': entry['Date'],
            'BCB_start': entry['BCB_start'],
            'BCB_stop': entry['BCB_stop'],
            'Dry_Intercept_n0': np.nan,
            'Dry_E_folding_D': np.nan
        })
        continue  # Move to next entry but store NaNs instead of skipping

    try:
        # Fit the exponential only using data up to 10 µm
        popt, _ = curve_fit(exponential, ddry_values[valid_indices], dN_dD_dry[valid_indices], p0=(1, 5), maxfev=5000)
        n0, D = popt

        # **Sanity check: Only accept reasonable D values**
        if D < 0.5 or D > 20:  # Arbitrary threshold, can adjust
            raise RuntimeError("D value out of range")

    except RuntimeError:
        print(f"Fit failed for {entry['Date']} (D={D:.2f})")
        n0, D = np.nan, np.nan  # Store NaN if fitting fails

    # Store fitted parameters (including NaNs for failed fits)
    dry_exponential_fits_10.append({
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
plt.title("Below Cloud Base January - June 2022\n Fitted Dry Size Distributions (≤10 µm)", fontsize=15, fontweight="bold")

plt.show()
print(f"Total successful dry exponential fits: {len([fit for fit in dry_exponential_fits if not np.isnan(fit['Dry_Intercept_n0'])])}")
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
#trying to select our windspeed bins based off of our ambient distributions 

windspeed_bins = [(0, 5), (5.001, 7), (7.001, 9), (9.001, np.inf)]
grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}
missing_windspeed_count = 0
bin_indices = range(12, 30)
bin_center = np.array(bin_center)
for entry in Y_BCB_calc:
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']

    bin_means = np.array([entry.get(f'Bin{i}_Y_mean', np.nan) for i in bin_indices], dtype=float)

    if np.isnan(bin_means).all():
        continue

    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    if windspeed_entry.empty:
        missing_windspeed_count += 1
        continue

    windspeed = windspeed_entry['Windspeed'].values[0]

    
    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:
            grouped_distributions[idx].append(bin_means)
            mean_windspeeds[idx].append(windspeed)
            break
plt.figure(figsize=(10, 8))
for idx, (low, high) in enumerate(windspeed_bins):
    distributions = grouped_distributions[idx]
    if distributions:
        dist_array = np.array(distributions)
        avg_distribution = np.nanmean(dist_array, axis=0)
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(distributions)

        plt.plot(bin_center, avg_distribution, linewidth=2.5,
                 label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")
plt.yscale('log')
plt.ylabel(r"CAS Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=16, fontweight="bold")
plt.xlabel("Deliquesced Diameter (μm)", fontsize=16, fontweight="bold")
plt.title('Hydrated Size Distributions Binned by Average Wind Speed', fontweight='bold', fontsize=17)
plt.legend(title=r"Average wind speed (m s$^{-1}$)", fontsize=13)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.ylim(1e-7, 10)
plt.tight_layout()
plt.show()
for idx, group in grouped_distributions.items():
    print(f"Windspeed bin {idx} ({windspeed_bins[idx]} m/s): {len(group)} legs")
print(f"Total legs with missing windspeed data: {missing_windspeed_count}")
#%%
#coverting units from cm⁻³ to µm⁻¹ cm⁻³
windspeed_bins = [(0, 5), (5.001, 7), (7.001, 9), (9.001, np.inf)]
grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}
missing_windspeed_count = 0
bin_indices = range(12, 30)
bin_center = np.array(bin_center)  # already defined elsewhere with 18 values

bin_edges = np.concatenate([
    [bin_center[0] - (bin_center[1] - bin_center[0]) / 2],
    (bin_center[:-1] + bin_center[1:]) / 2,
    [bin_center[-1] + (bin_center[-1] - bin_center[-2]) / 2]
])
bin_widths = np.diff(bin_edges)  # shape (18,)
for entry in Y_BCB_calc:
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']

    bin_means = np.array([entry.get(f'Bin{i}_Y_mean', np.nan) for i in bin_indices], dtype=float)

    if np.isnan(bin_means).all():
        continue

    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    if windspeed_entry.empty:
        missing_windspeed_count += 1
        continue

    windspeed = windspeed_entry['Windspeed'].values[0]

    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:
            grouped_distributions[idx].append(bin_means)
            mean_windspeeds[idx].append(windspeed)
            break
plt.figure(figsize=(10, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    distributions = grouped_distributions[idx]
    if distributions:
        dist_array = np.array(distributions)  # shape: (n_legs, 18)
        avg_distribution = np.nanmean(dist_array, axis=0)
        avg_distribution_total = avg_distribution * bin_widths

        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(distributions)

        plt.plot(bin_center, avg_distribution_total, linewidth=2.5,
                 label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")
plt.yscale('log')
plt.xlabel("Deliquesced Diameter (μm)", fontsize=16, fontweight="bold")
plt.ylabel(r"CAS Number Concentration (cm$^{-3}$)", fontsize=16, fontweight="bold")
plt.title('Hydrated Size Distributions Binned by Average Wind Speed', fontweight='bold', fontsize=17)
plt.legend(title=r"Average wind speed (m s$^{-1}$)", fontsize=13)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.ylim(1e-7, 10)
plt.tight_layout()
plt.show()
for idx, group in grouped_distributions.items():
    print(f"Windspeed bin {idx} ({windspeed_bins[idx]} m/s): {len(group)} legs")
print(f"Total legs with missing windspeed data: {missing_windspeed_count}")

#%%
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
# for entry in dry_exponential_fits_10:
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
common_bins = np.linspace(2, 100, 100)  # Creates 10 bin centers between 2 and 10 µm

# Define the exponential function
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)


windspeed_bins = [(0, 5), (5.001, 7), (7.001, 9), (9.001, np.inf)]

# Store binned distributions
grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

missing_windspeed_count = 0

# Use dry_exponential_fits_10 instead of dry_exponential_fits
for entry in dry_exponential_fits_10:
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
        missing_windspeed_count += 1
        continue

    windspeed = windspeed_entry['Windspeed'].values[0]

    # Generate size distribution from exponential fit
    ddry_values = np.array(common_bins)
    size_dist = n0 * np.exp(-ddry_values / D)

    # Bin the distribution by windspeed
    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:
            grouped_distributions[idx].append(size_dist)
            mean_windspeeds[idx].append(windspeed)
            break

# Print summary
for idx, group in grouped_distributions.items():
    print(f"Windspeed bin {idx} ({windspeed_bins[idx]} m/s): {len(group)} legs")

print(f"Total legs with missing windspeed data: {missing_windspeed_count}")

# Step 3: Plot binned size distributions
plt.figure(figsize=(10, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])

        plt.plot(common_bins, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linewidth=3)

plt.yscale('log')
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=16, fontweight="bold")
plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=16, fontweight="bold")
plt.title('Dry Size Distributions Binned by Average Wind Speed', fontweight='bold', fontsize=17)
plt.legend(title=r"Average wind speed m s$^{-1}$")
plt.tight_layout()
plt.ylim(1e-4, 10**0)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.show()

total_legs = sum(len(group) for group in grouped_distributions.values())
print(f"Total number of legs plotted: {total_legs}")
#%%
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
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    plt.ylim(10**-4, 10**2)
    # Labels and title
    plt.ylabel(' concentration (/cm³)', fontweight='bold')
    plt.xlabel('Bin diameter (µm)', fontweight='bold')
    plt.title('Below Cloud Base CAS January-June 2022 wind speed 5-7 m/s', fontweight='bold')
    plt.legend()

    plt.tight_layout()
    plt.show()
else:
    print("No data available for the 5-7 m/s windspeed bin.")
#%%
common_bins = np.linspace(2, 10, 10)  # Creates 10 bin centers between 2 and 10 µm

# Define the exponential function
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)


windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]

# Store binned distributions
grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

missing_windspeed_count = 0

# Use dry_exponential_fits_10 instead of dry_exponential_fits
for entry in dry_exponential_fits_10:
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
        missing_windspeed_count += 1
        continue

    windspeed = windspeed_entry['Windspeed'].values[0]

    # Generate size distribution from exponential fit
    ddry_values = np.array(common_bins)
    size_dist = n0 * np.exp(-ddry_values / D)

    # Bin the distribution by windspeed
    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:
            grouped_distributions[idx].append(size_dist)
            mean_windspeeds[idx].append(windspeed)
            break

# Print summary
for idx, group in grouped_distributions.items():
    print(f"Windspeed bin {idx} ({windspeed_bins[idx]} m/s): {len(group)} legs")

print(f"Total legs with missing windspeed data: {missing_windspeed_count}")

# Step 3: Plot binned size distributions
plt.figure(figsize=(10, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])

        plt.plot(common_bins, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linewidth=3)

plt.yscale('log')
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=16, fontweight="bold")
plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=16, fontweight="bold")
plt.title('Dry Size Distributions Binned by Average Wind Speed', fontweight='bold', fontsize=17)
plt.legend(title=r"Average wind speed m s$^{-1}$")
plt.tight_layout()
plt.ylim(1e-4, 10**0)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.show()

total_legs = sum(len(group) for group in grouped_distributions.values())
print(f"Total number of legs plotted: {total_legs}")
#%%



# Define wind speed bins
windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]

# Store binned distributions
grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

missing_windspeed_count = 0

# Use raw size distributions from `dry_exponential_fits_10`
for entry in dry_exponential_fits_10:
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    
    ddry_values = np.array(common_bins)  # Actual bin centers
    size_dist = np.array(entry['dN/dDdry'])  # Observed size distribution

    # Match windspeed
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) & 
        (df_combined['BCB_start'] == BCB_start) & 
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    if windspeed_entry.empty or np.any(np.isnan(size_dist)):
        missing_windspeed_count += 1
        continue

    windspeed = windspeed_entry['Windspeed'].values[0]

    # Bin the distribution by windspeed
    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:
            grouped_distributions[idx].append(size_dist)
            mean_windspeeds[idx].append(windspeed)
            break

# Print summary
for idx, group in grouped_distributions.items():
    print(f"Windspeed bin {idx} ({windspeed_bins[idx]} m/s): {len(group)} legs")

print(f"Total legs with missing windspeed data: {missing_windspeed_count}")

# Step 3: Plot binned size distributions
plt.figure(figsize=(10, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])

        plt.plot(ddry_values, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linewidth=3)

plt.yscale('log')
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=16, fontweight="bold")
plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=16, fontweight="bold")
plt.title('Dry Size Distributions Binned by Average Wind Speed', fontweight='bold', fontsize=17)
plt.legend(title=r"Average wind speed m s$^{-1}$")
plt.tight_layout()
plt.ylim(1e-4, 10**0)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.show()

total_legs = sum(len(group) for group in grouped_distributions.values())
print(f"Total number of legs plotted: {total_legs}")

#%%

#Fitting an exponential to wind speed bins 
def fit_function(x, n0, D):
    return n0 * np.exp(-x / D)

windspeed_colors = ['blue', 'orange', 'green', 'red']  # Order must match bins

fit_results = {}

plt.figure(figsize=(10, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        
        concentrations_array = np.array(grouped_distributions[idx])
        
        avg_concentration = np.mean(concentrations_array, axis=0)
        
        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)
        
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])

        try:
            popt, _ = curve_fit(fit_function, common_bins, avg_concentration, p0=(1, 5), maxfev=5000)
            n0_fit, D_fit = popt
            fit_results[idx] = {'n0': n0_fit, 'D': D_fit, 'avg_windspeed': avg_windspeed, 'num_legs': num_legs}

            x_fit = np.linspace(min(common_bins), max(common_bins), 10)
            y_fit = fit_function(x_fit, *popt)

            plt.plot(x_fit, y_fit, color=windspeed_colors[idx], linewidth=3, linestyle='-',
                     label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")

        except RuntimeError:
            print(f"Exponential fit failed for windspeed bin {avg_windspeed:.1f} m/s")

plt.ylabel(r'Number concentration (cm$^{-3}$ µm$^{-1}$)', fontweight='bold', fontsize=18)
plt.xlabel('Bin diameter (µm)', fontweight='bold', fontsize=18)
plt.yscale('log')
plt.ylim(10**-4, 10**0)
plt.title('Below Cloud Base January - June 2022\nFitted Dry Size Distributions Binned by Average Windspeed', fontweight='bold', fontsize=18)
plt.legend(title=r"Wind speed bins (m s$^{-1}$)", title_fontsize=15, fontsize=13, frameon=True, prop={'weight': 'bold'})
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


# %%


# %%
#Trying ambient wind speed relationship 

windspeed_bins = [(0, 5), (5.001, 7), (7.001, 9), (9.001, np.inf)]

# Store binned distributions
grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

missing_windspeed_count = 0

# Use bin_center (ambient size distributions) instead of common_bins
bin_radius = bin_center / 2 

# Use already fitted exponential size distributions for ambient
for entry in ambient_fits:
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    
    n0 = entry['Intercept_n0']
    D = entry['E_folding_D']

    # Match windspeed from df_combined
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) & 
        (df_combined['BCB_start'] == BCB_start) & 
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    if windspeed_entry.empty or np.isnan(n0) or np.isnan(D):
        missing_windspeed_count += 1
        continue

    windspeed = windspeed_entry['Windspeed'].values[0]

    # ✅ Bin the distribution by windspeed
    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:
            grouped_distributions[idx].append(size_dist_valid)
            mean_windspeeds[idx].append(windspeed)
            break

# Print summary
for idx, group in grouped_distributions.items():
    print(f"Windspeed bin {idx} ({windspeed_bins[idx]} m/s): {len(group)} legs")

print(f"Total legs with missing windspeed data: {missing_windspeed_count}")

# Step 3: Plot binned size distributions
plt.figure(figsize=(10, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        avg_distribution = np.nanmean(grouped_distributions[idx], axis=0)  # ✅ Use mean of raw distributions
        avg_windspeed = np.nanmean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])

        plt.plot(bin_radius_valid, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linewidth=2.5)

plt.xscale('log')  # ✅ Match Lewis & Schwartz log scale
plt.yscale('log')
plt.ylabel(r"Number Concentration (cm$^{-3}$)", fontsize=16, fontweight="bold")  
plt.xlabel("Bin Center Radius (µm)", fontsize=16, fontweight="bold")
plt.title('Ambient Size Distributions Binned by Average Wind Speed', fontweight='bold', fontsize=17)
plt.legend(title=r"Average wind speed (m s$^{-1}$)")
plt.tight_layout()
plt.ylim(1e-4, 10**2)  # ✅ Match y-axis with Lewis & Schwartz

plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.show()

total_legs = sum(len(group) for group in grouped_distributions.values())
print(f"Total number of legs plotted: {total_legs}")

# %%


# ✅ Wind speed bins (to match your new binning choice)
windspeed_bins = [(0, 5), (5.001, 7), (7.001, 9), (9.001, np.inf)]

# ✅ Store binned distributions
grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

missing_windspeed_count = 0

# ✅ Convert bin centers from diameter to radius
bin_radius = np.array(bin_center, dtype=float) / 2  # Ensure array and convert to radius

# ✅ Use already fitted exponential size distributions for ambient
for entry in ambient_fits_10:
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    
    n0 = entry['Intercept_n0']
    D = entry['E_folding_D']

    # Match windspeed from df_combined
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) & 
        (df_combined['BCB_start'] == BCB_start) & 
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    if windspeed_entry.empty or np.isnan(n0) or np.isnan(D):
        missing_windspeed_count += 1
        continue

    windspeed = windspeed_entry['Windspeed'].values[0]

    # ✅ Use the already fitted exponential size distribution
    size_dist = n0 * np.exp(-bin_radius / D)  # Use radius instead of diameter

    # ✅ Bin the distribution by wind speed
    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:
            grouped_distributions[idx].append(size_dist)
            mean_windspeeds[idx].append(windspeed)
            break

# ✅ Print summary
for idx, group in grouped_distributions.items():
    print(f"Windspeed bin {idx} ({windspeed_bins[idx]} m/s): {len(group)} legs")

print(f"Total legs with missing windspeed data: {missing_windspeed_count}")

# ✅ Step 3: Plot binned size distributions
plt.figure(figsize=(10, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        avg_distribution = np.nanmean(grouped_distributions[idx], axis=0)  # ✅ Average size distribution
        avg_windspeed = np.nanmean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])

        plt.plot(bin_radius, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linewidth=2.5)

plt.xscale('log')  # ✅ Match Lewis & Schwartz log scale
plt.yscale('log')
plt.ylabel(r"Number Concentration (cm$^{-3}$)", fontsize=16, fontweight="bold")  
plt.xlabel("Bin Center Radius (µm)", fontsize=16, fontweight="bold")
plt.title('Ambient Size Distributions Binned by Average Wind Speed', fontweight='bold', fontsize=17)
plt.legend(title=r"Average wind speed (m s$^{-1}$)")
plt.tight_layout()
plt.ylim(1e-6, 10**2)  # ✅ Match y-axis with Lewis & Schwartz

plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.show()

total_legs = sum(len(group) for group in grouped_distributions.values())
print(f"Total number of legs plotted: {total_legs}")

#%%
# ✅ Define Wind Speed Bins (matching your chosen bins)
windspeed_bins = [(0, 5), (5.001, 7), (7.001, 9), (9.001, np.inf)]

# ✅ Storage for Binned Distributions
grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

missing_windspeed_count = 0

# ✅ Convert bin centers from diameter to radius
bin_radius = np.array(bin_center, dtype=float) / 2  # Convert to radius

# ✅ Loop through each leg in Y_BCB_calc
for entry in Y_BCB_calc:
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']

    # ✅ Get the observed size distribution for this leg
    size_dist = np.array([entry.get(f'Bin{i:02d}_Y_mean', np.nan) for i in range(30)], dtype=float)

    # ✅ Skip if all values are NaN
    if np.isnan(size_dist).all():
        continue

    # ✅ Find Matching Wind Speed Data
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) & 
        (df_combined['BCB_start'] == BCB_start) & 
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    if windspeed_entry.empty:
        missing_windspeed_count += 1
        continue

    windspeed = windspeed_entry['Windspeed'].values[0]

    # ✅ Bin the distribution by wind speed
    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:
            grouped_distributions[idx].append(size_dist)
            mean_windspeeds[idx].append(windspeed)
            break

# ✅ Print Summary
for idx, group in grouped_distributions.items():
    print(f"Windspeed bin {idx} ({windspeed_bins[idx]} m/s): {len(group)} legs")

print(f"Total legs with missing windspeed data: {missing_windspeed_count}")

# ✅ Step 3: Plot Binned Size Distributions
plt.figure(figsize=(10, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        avg_distribution = np.nanmean(grouped_distributions[idx], axis=0)  # ✅ Average over legs
        avg_windspeed = np.nanmean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])

        plt.plot(bin_radius, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linewidth=2.5)

plt.xscale('log')  # ✅ Match Lewis & Schwartz log scale
plt.yscale('log')
plt.ylabel(r"Number Concentration (cm$^{-3}$)", fontsize=16, fontweight="bold")  
plt.xlabel("Bin Center Radius (µm)", fontsize=16, fontweight="bold")
plt.title('Ambient Size Distributions Binned by Average Wind Speed', fontweight='bold', fontsize=17)
plt.legend(title=r"Average wind speed (m s$^{-1}$)")
plt.tight_layout()
plt.ylim(1e-6, 10**2)  # ✅ Match y-axis with Lewis & Schwartz

plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.show()

total_legs = sum(len(group) for group in grouped_distributions.values())
print(f"Total number of legs plotted: {total_legs}")

# %%
# ✅ Select the windspeed bin 5-7 m/s (index 1)
idx = 1  # Index for the 5-7 m/s bin

# ✅ Check if the bin contains data
if grouped_distributions[idx]:
    avg_distribution = np.nanmean(grouped_distributions[idx], axis=0)  # Average over all legs in the bin
    avg_windspeed = np.nanmean(mean_windspeeds[idx])  # Compute average windspeed
    num_legs = len(grouped_distributions[idx])  # Count number of legs in this bin

    # ✅ Convert bin centers from diameter to radius
    bin_radius = np.array(bin_center, dtype=float) / 2  # Ensure array and convert to radius


    # ✅ Plot only the 5-7 m/s bin
    plt.figure(figsize=(10, 6))
    plt.plot(bin_radius, avg_distribution, color='red', label=f"5-7 m/s", linewidth=3)

    # ✅ Set log-log scale (matching Lewis & Schwartz)
    plt.yscale('log')
    plt.xscale('log')  # ✅ Set x-axis to log scale
    plt.xlim(0.1, 100)  # ✅ Match Lewis & Schwartz x-axis range

    # plt.xscale('log')
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    plt.ylim(10**-4, 10**2)

    # ✅ Labels and title
    plt.ylabel('Concentration (/cm³)', fontweight='bold', fontsize=16)
    plt.xlabel('Bin Center Radius (µm)', fontweight='bold', fontsize=16)
    plt.title('Below Cloud Base CAS January-June 2022\n Wind Speed 5-7 m/s', fontweight='bold', fontsize=18)
    plt.legend()

    plt.tight_layout()
    plt.show()

else:
    print("No data available for the 5-7 m/s windspeed bin.")
#%%
# ✅ Select the windspeed bin 5-7 m/s (index 1)
idx = 1  # Index for the 5-7 m/s bin

# ✅ Check if the bin contains data
if grouped_distributions[idx]:
    # avg_distribution = np.nanmean(grouped_distributions[idx], axis=0)
    avg_distribution = np.nanmean(grouped_distributions[idx], axis=0) / 2  # Adjust scaling factor if needed
  # Average over all legs in the bin
    avg_windspeed = np.nanmean(mean_windspeeds[idx])  # Compute average windspeed
    num_legs = len(grouped_distributions[idx])  # Count number of legs in this bin

    # ✅ Convert bin centers from diameter to radius & apply a slight shift if needed
    bin_radius = np.array(bin_center, dtype=float) / 2 * 1.05  # Ensure proper alignment

    # ✅ Plot only the 5-7 m/s bin
    plt.figure(figsize=(10, 6))
    plt.plot(bin_radius, avg_distribution, color='red', label=f"5-7 m/s", linewidth=3)

    # ✅ Set log-log scale (matching Lewis & Schwartz)
    plt.yscale('log')
    plt.xscale('log')  # ✅ Set x-axis to log scale
    plt.xlim(0.1, 100)  # ✅ Match Lewis & Schwartz x-axis range

    # ✅ Adjust x-axis ticks to match Lewis & Schwartz
    import matplotlib.ticker as ticker
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    plt.xticks([0.1, 1, 10, 100], labels=["0.1", "1", "10", "100"], fontsize=16, fontweight="bold")

    plt.yticks(fontsize=16, fontweight='bold')
    plt.ylim(10**-4, 10**2)

    # ✅ Labels and title
    plt.ylabel('Concentration (/cm³)', fontweight='bold', fontsize=16)
    plt.xlabel('Bin Center Radius (µm)', fontweight='bold', fontsize=16)
    plt.title('Below Cloud Base CAS January-June 2022\n Wind Speed 5-7 m/s', fontweight='bold', fontsize=18)
    plt.legend()

    plt.tight_layout()
    plt.show()

else:
    print("No data available for the 5-7 m/s windspeed bin.")

# %%
# ✅ Convert bin centers from diameter to radius
bin_radius = bin_center / 2  # Convert diameter to radius

# ✅ Define log-spaced bin centers (matching Lewis & Schwartz)
bin_radius_log = np.logspace(np.log10(0.1), np.log10(100), len(bin_radius))

# ✅ Select the windspeed bin 5-7 m/s (index 1)
idx = 1  # Index for the 5-7 m/s bin

# ✅ Check if the bin contains data
if grouped_distributions[idx]:
    avg_distribution = np.nanmean(grouped_distributions[idx], axis=0)
    avg_windspeed = np.nanmean(mean_windspeeds[idx])
    num_legs = len(grouped_distributions[idx])

    # ✅ Plot only the 5-7 m/s bin
    plt.figure(figsize=(10, 6))
    plt.plot(bin_radius_log, avg_distribution, color='orange', label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")

    # ✅ Set log-log scale (matching Lewis & Schwartz)
    plt.yscale('log')
    plt.xscale('log')
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    plt.xlim(0.1, 100)  # Match Lewis & Schwartz range
    plt.ylim(10**-4, 10**2)

    # ✅ Labels and title
    plt.ylabel('Concentration (/cm³)', fontweight='bold', fontsize=16)
    plt.xlabel('Bin Center Radius (µm)', fontweight='bold', fontsize=16)
    plt.title('Below Cloud Base CAS January-June 2022\n Wind Speed 5-7 m/s', fontweight='bold', fontsize=18)
    plt.legend()

    plt.tight_layout()
    plt.show()

else:
    print("No data available for the 5-7 m/s windspeed bin.")

# %%
#Plotting Lewis and Schwartz cannonical distribution 

r80 = np.logspace(np.log10(0.1), np.log10(20), 300)
center = 6 * np.exp(-((np.log10(r80) - np.log10(0.6))**2) / (2 * 0.35**2))
log_r = np.log10([0.1, 0.6, 20])
log_upper = np.log10([3.16, 6, 1e-3])
log_lower = np.log10([0.316, 0.8, 1e-4]) 
interp_upper = interp1d(log_r, log_upper, kind='quadratic')
interp_lower = interp1d(log_r, log_lower, kind='quadratic')
log_r80 = np.log10(r80)
upper_bound = 10**interp_upper(log_r80)
lower_bound = 10**interp_lower(log_r80)
plt.figure(figsize=(8, 6))
plt.fill_between(r80, lower_bound, upper_bound, color='gray', alpha=0.5, label='Canonical range (Lewis & Schwartz 2004)')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$r_{80}$ [$\mu$m]', fontsize=19, fontweight='bold')
plt.ylabel(r'$n(r_{80})$ [cm$^{-3}$]', fontsize=19, fontweight='bold')
plt.ylim(1e-5, 20)
plt.xlim(0.08, 30)
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.grid()
plt.title("Comparison of Below Cloud Aerosol Size Distributions (ACTIVATE)\nand Canonical Marine Aerosol Profiles\nWind Speed Range: 5–7 m s$^{-1}$", 
          fontsize=19, fontweight='bold')
plt.legend(fontsize=15, loc='lower left') 
plt.show()
