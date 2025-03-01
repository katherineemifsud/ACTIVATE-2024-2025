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
#We are only focused on bins 12-30 because we want coarse mode aerosol at 2.5-50 um diameter 
bin_center=[0.555, 0.645, 0.715, 0.785, 0.855, 0.925, 0.995,
            1.07, 1.14, 1.21, 1.38, 1.75, 
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

# #%%
# #Double check the number of legs associated with each date to compare across multiple instruments.  
# leg_count = Counter([leg['Date'] for leg in leg_info])
# print("Number of legs associated with each date:")
# total_legs = 0
# for date, count in sorted(leg_count.items()):
#     print(f"Date: {date}, Number of Legs: {count}")
#     total_legs += count
# print(f"\nTotal number of legs: {total_legs}")
# #%%
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

#             Y_calc[bin_key_Y] = np.nanmean(bin_means[bin_key_Y]) * bin_log[bin_label - 12]
#             N_calc[bin_key_N] = np.nanmean(bin_means[bin_key_N]) * bin_log[bin_label - 12]

#         Y_BCB_calc.append(Y_calc)
#         N_BCB_calc.append(N_calc)
# #%%
# #Create ambient size distributions for each leg.  
# all_bin_means = []

# for entry in Y_BCB_calc:
#     bin_means = []
#     for i in range(12, 30):  # Bins from 12 to 29
#         key = f'Bin{i}_Y_mean'
#         bin_means.append(entry.get(key, np.nan))

#     # Append the bin means along with the date and leg times
#     all_bin_means.append({
#         'Date': entry['Date'],
#         'BCB_start': entry.get('BCB_start', np.nan),
#         'BCB_stop': entry.get('BCB_stop', np.nan),
#         'Bin_means': bin_means
#     })

# unique_dates = sorted(set(entry['Date'] for entry in all_bin_means))
# colors = cm.viridis(np.linspace(0, 1, len(unique_dates)))
# date_color_map = {date: color for date, color in zip(unique_dates, colors)}

# plt.figure(figsize=(10, 6))
# added_dates = set()  # To track which dates have been added to the legend

# for entry in all_bin_means:
#     bin_means = np.array(entry['Bin_means'])
#     valid_indices = ~np.isnan(bin_means)
#     date = entry['Date']
#     color = date_color_map[date]
    
#     if date not in added_dates:
#         plt.scatter(np.array(bin_center)[valid_indices], bin_means[valid_indices], color=color, marker='o', label=date)
#         added_dates.add(date)
#     else:
#         plt.scatter(np.array(bin_center)[valid_indices], bin_means[valid_indices], color=color, marker='o')

# plt.xlabel('Bin centers diameter (um)', fontsize=12, fontweight='bold')
# plt.ylabel('Clear mean droplet concentration \n (/cm^3)', fontsize=12, fontweight='bold')
# plt.title('Below Cloud Base clear sky distribution \n January-June 2022', fontsize=12, fontweight='bold')
# plt.xticks(np.arange(0, 50, 2))
# num_cols = 7
# plt.legend(title='Date', ncol=num_cols, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True)
# plt.tight_layout()
# plt.show()

# #%%
# #Fit an exponential function to the whole data set.  

# def exponential(x, n0, D):
#     return n0 * np.exp(-x / D)

# all_bin_means = []

# for entry in Y_BCB_calc:
#     bin_means = []
#     for i in range(12, 30):  # Bins from 12 to 29
#         key = f'Bin{i}_Y_mean'
#         bin_means.append(entry.get(key, np.nan))

#     # Append the bin means along with the date and leg times
#     all_bin_means.append({
#         'Date': entry['Date'],
#         'BCB_start': entry.get('BCB_start', np.nan),
#         'BCB_stop': entry.get('BCB_stop', np.nan),
#         'Bin_means': bin_means
#     })

# # Get unique dates and colors for plotting
# unique_dates = sorted(set(entry['Date'] for entry in all_bin_means))
# colors = cm.viridis(np.linspace(0, 1, len(unique_dates)))
# date_color_map = {date: color for date, color in zip(unique_dates, colors)}

# # Scatter plot
# plt.figure(figsize=(10, 6))
# added_dates = set()  # To track which dates have been added to the legend

# for entry in all_bin_means:
#     bin_means = np.array(entry['Bin_means'])
#     valid_indices = ~np.isnan(bin_means)
#     date = entry['Date']
#     color = date_color_map[date]
    
#     if date not in added_dates:
#         plt.scatter(np.array(bin_center)[valid_indices], bin_means[valid_indices], color=color, marker='o', label=date)
#         added_dates.add(date)
#     else:
#         plt.scatter(np.array(bin_center)[valid_indices], bin_means[valid_indices], color=color, marker='o')

# # Flatten the bin_center and all_bin_means for fitting
# flat_bin_centers = np.tile(bin_center, len(all_bin_means))
# flat_bin_means = np.concatenate([np.array(entry['Bin_means']) for entry in all_bin_means])

# # Convert to NumPy array with a float type
# flat_bin_centers = np.array(flat_bin_centers, dtype=float)
# flat_bin_means = np.array(flat_bin_means, dtype=float)

# # Remove NaN values
# valid_indices = ~np.isnan(flat_bin_means)
# flat_bin_centers = flat_bin_centers[valid_indices]
# flat_bin_means = flat_bin_means[valid_indices]

# # Fit the exponential function to the data
# popt, pcov = curve_fit(exponential, flat_bin_centers, flat_bin_means, p0=(1, 1))

# # Extract the intercept (n0) and e-folding diameter (D)
# n0 = popt[0]
# D = popt[1]

# # Print the intercept and e-folding diameter
# print(f"Intercept (n0): {n0:.2e}")
# print(f"E-folding diameter (D): {D:.2f} um")

# # Plot the fitted curve
# x_fit = np.linspace(min(bin_center), max(bin_center), 100)
# y_fit = exponential(x_fit, *popt)
# plt.plot(x_fit, y_fit, color='red', label=f'Exponential fit: y = {n0:.2e} * exp(-x / {D:.2f})')

# # Add labels, title, and legend
# plt.xlabel('Bin centers diameter (um)', fontsize=12, fontweight='bold')
# plt.ylabel('Clear mean droplet concentration \n (/cm^3)', fontsize=12, fontweight='bold')
# plt.title('Below Cloud Base clear sky distribution \n January-June 2022', fontsize=12, fontweight='bold')
# # plt.yscale('log')
# plt.xticks(np.arange(0, 50, 5))
# num_cols = 7
# plt.legend(title='Date', ncol=num_cols, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True)
# plt.tight_layout()
# # plt.ylim(10**-5, 10**1)
# plt.show()

# #%%
# #Fit an exponential function to each leg.

# # Define the exponential function
# def exponential(x, n0, D):
#     return n0 * np.exp(-x / D)

# # Process the data
# all_bin_means = []
# dates = []
# for entry in Y_BCB_calc:
#     bin_means = []
#     for i in range(12, 30):  # Bins from 12 to 29
#         key = f'Bin{i}_Y_mean'
#         bin_means.append(entry.get(key, np.nan))
#     all_bin_means.append({
#         'Date': entry['Date'],
#         'BCB_start': entry.get('BCB_start', np.nan),
#         'BCB_stop': entry.get('BCB_stop', np.nan),
#         'Bin_means': bin_means
#     })
#     dates.append(entry['Date'])

# # Get unique dates and colors for plotting
# unique_dates = sorted(set(dates))
# colors = cm.viridis(np.linspace(0, 1, len(unique_dates)))
# date_color_map = {date: color for date, color in zip(unique_dates, colors)}

# # Define the color for the fits
# fit_color = 'purple'

# # Scatter plot
# plt.figure(figsize=(10, 6))
# added_dates = set()  # To track which dates have been added to the legend

# # Loop through each leg's data
# for entry in all_bin_means:
#     bin_means = np.array(entry['Bin_means'])
#     valid_indices = ~np.isnan(bin_means)
#     date = entry['Date']
#     color = date_color_map[date]
    
#     # Skip if bin_means is empty
#     if not valid_indices.any():
#         print(f"No valid data for date {date}")
#         continue

#     bin_centers = np.array(bin_center)[valid_indices]
#     bin_means = bin_means[valid_indices]
    
#     # Fit the exponential function to the data of each leg
#     try:
#         popt, _ = curve_fit(exponential, bin_centers, bin_means, p0=(1, 1))
#         n0, D = popt
        
#         # Print the intercept and e-folding diameter for each leg
#         print(f"Date: {date}")
#         print(f"  Intercept (n0): {n0:.2e}")
#         print(f"  E-folding diameter (D): {D:.2f} um")
        
#         # Plot the fitted curve for each leg
#         x_fit = np.linspace(min(bin_centers), max(bin_centers), 100)
#         y_fit = exponential(x_fit, *popt)
#         plt.plot(x_fit, y_fit, color=fit_color, label=f'{date} fit: y = {n0:.2e} * exp(-x / {D:.2f})')

#     except RuntimeError:
#         print(f"Fit could not be performed for date {date}")
    
#     # Plot the data points for each leg
#     if date not in added_dates:
#         plt.scatter(bin_centers, bin_means, color=color, marker='o', label=date)
#         added_dates.add(date)
#     else:
#         plt.scatter(bin_centers, bin_means, color=color, marker='o')

# # Add labels, title, and legend
# plt.xlabel('Bin centers diameter (um)', fontsize=12, fontweight='bold')
# plt.ylabel('Clear mean droplet concentration \n (/cm^3/um)', fontsize=12, fontweight='bold')
# plt.title('Below Cloud Base clear sky distribution \n January-June 2022', fontsize=12, fontweight='bold')
# plt.yscale('log')
# plt.xticks(np.arange(0, 50, 5))
# num_cols = 7
# plt.legend(title='Date', ncol=num_cols, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True)
# plt.tight_layout()
# plt.show()
# #%%
# #Plot the exponential fits for each size distribution and extract the efolding diameter and intercept for each leg. 
# # Define the exponential function
# def exponential(x, n0, D):
#     return n0 * np.exp(-x / D)

# # Initialize the dictionary
# master_BCB_exponential = {}

# # Process the data
# for entry in all_bin_means:
#     bin_means = np.array(entry['Bin_means'], dtype=float)  # Ensure bin_means is a float array
#     valid_indices = ~np.isnan(bin_means)
#     bin_centers = np.array(bin_center)[valid_indices]
#     bin_means = bin_means[valid_indices]
    
#     # Skip if bin_means is empty
#     if not valid_indices.any():
#         print(f"No valid data for date {entry['Date']}")
#         continue
    
#     # Fit the exponential function to the data
#     try:
#         popt, _ = curve_fit(exponential, bin_centers, bin_means, p0=(1, 1))
#         n0, D = popt
        
#         # Store the n0 and D values in the master_min_exponential dictionary
#         if entry['Date'] not in master_BCB_exponential:
#             master_BCB_exponential[entry['Date']] = []
        
#         master_BCB_exponential[entry['Date']].append({
#             'Date': entry['Date'],
#             'BCB_start': entry.get('BCB_start', np.nan),
#             'BCB_stop': entry.get('BCB_stop', np.nan),
#             'n0': n0,
#             'D': D
#         })
        
#         # Print the intercept and e-folding diameter for each leg
#         print(f"Date: {entry['Date']}")
#         print(f"  Intercept (n0): {n0:.2e}")
#         print(f"  E-folding diameter (D): {D:.2f} um")
        
#         # Plot the fitted curve for each leg
#         x_fit = np.linspace(min(bin_centers), max(bin_centers), 100)
#         y_fit = exponential(x_fit, *popt)
#         plt.plot(x_fit, y_fit, color='purple', label=f'{entry["Date"]} fit: y = {n0:.2e} * exp(-x / {D:.2f})')

#     except RuntimeError:
#         print(f"Fit could not be performed for date {entry['Date']}")
    
#     # Plot the data points for each leg
#     if entry['Date'] not in added_dates:
#         plt.scatter(bin_centers, bin_means, color='purple', marker='o', label=entry['Date'])
#         added_dates.add(entry['Date'])
#     else:
#         plt.scatter(bin_centers, bin_means, color='purple', marker='o')

# # Add labels, title, and legend
# plt.xlabel('Bin centers diameter (um)', fontsize=12, fontweight='bold')
# plt.ylabel('Clear mean droplet concentration \n (/cm^3/um)', fontsize=12, fontweight='bold')
# plt.title('Below Cloud Base clear sky distribution \n January-June 2022', fontsize=12, fontweight='bold')
# plt.xticks(np.arange(0, 50, 5))
# plt.tight_layout()
# plt.show()
# #%%
# # Print the number of inner dictionaries for each date in master_BCB_exponential
# for date, entries in master_BCB_exponential.items():
#     print(f"Date: {date} has {len(entries)} entries")
# total_entries = sum(len(entries) for entries in master_BCB_exponential.values())
# print(f"Total number of inner dictionaries: {total_entries}")
# #%%
# # Loop through each date and its entries in master_BCB_exponential
# for date, entries in master_BCB_exponential.items():
#     print(f"Date: {date}")
#     for entry in entries:
#         start_time = entry.get('BCB_start', 'N/A')  
#         stop_time = entry.get('BCB_stop', 'N/A')   
#         print(f"  Start Time: {start_time}, Stop Time: {stop_time}")

# # Optionally, count the total number of leg times for verification
# total_legs = sum(len(entries) for entries in master_BCB_exponential.values())
# print(f"Total number of legs: {total_legs}")
# #%%


# #%%
# #Calculate the relative humidity of each leg.

# master_BCB_RH = []

# for i in range(len(dates_legs)):
#     date = dates_legs[i]
#     leg_dict = leg_data[i]

#     flight_date = leg_dict['Date']  # Get date of flight from dictionary 
#     BCB_start = leg_dict['LegIndex_02']['StartTimes']
#     BCB_stop = leg_dict['LegIndex_02']['StopTimes']

#     rh_flight = h20[i]
#     times_rh = rh_flight.Time_Start.values
#     rh_values = rh_flight.RHw_DLH.values

#     all_BCB = []

#     for j in range(len(BCB_start)):
#         start = int(BCB_start[j])
#         end = int(BCB_stop[j])

#         rh_times = {
#             'Date': date,
#             'BCB_start': start,
#             'BCB_stop': end,
#             'Rh_mean': [],
#         }

        
#         index1_start = None
#         for k in range(len(times_rh)):
#             if times_rh[k] == start:
#                 index1_start = k
#                 break

        
#         index1_end = None
#         for k in range(len(times_rh)):
#             if times_rh[k] == end:
#                 index1_end = k
#                 break

#         if index1_start is None or index1_end is None:
#             rh9_mean = np.nan
#         else:
#             rh9 = rh_values[index1_start:index1_end + 1]
#             rh9_mean = np.nanmean(rh9)

#         rh_times['Rh_mean'].append(rh9_mean)
#         all_BCB.append(rh_times)  # List that contains all the BCB wind/alt mean dictionaries for 1 flight

#     master_BCB_RH.append(all_BCB)

# # Step 2: Replace -999 with NaN in master_BCB_RH
# for flight in master_BCB_RH:
#     for leg in flight:
#         rh_mean_list = leg['Rh_mean']
#         leg['Rh_mean'] = [np.nan if value <=0 else value for value in rh_mean_list]

# #%%
# #for only the legs present after LWC filtration and master_BCB_exponential 
# #Filtering master_BCB_RH because some legs may not contain humidity data. So we need to remove those legs. 
# #Do not linearly interpolate the data. We are working with observations and linearly interpolating the data would no longer represent observation. 
# # Extract the (date, BCB_start, BCB_stop) from master_BCB_exponential
# date_leg_set = set()
# for entries in master_BCB_exponential.values():
#     for entry in entries:
#         date = entry['Date']
#         BCB_start = entry.get('BCB_start', np.nan)
#         BCB_stop = entry.get('BCB_stop', np.nan)
#         date_leg_set.add((date, BCB_start, BCB_stop))

# # Filter master_BCB_RH
# filtered_master_BCB_RH = []

# for flight in master_BCB_RH:
#     filtered_legs = []
#     for leg in flight:
#         date = leg['Date']
#         BCB_start = leg['BCB_start']
#         BCB_stop = leg['BCB_stop']
#         if (date, BCB_start, BCB_stop) in date_leg_set:
#             filtered_legs.append(leg)
#     if filtered_legs:
#         filtered_master_BCB_RH.append(filtered_legs)
# #%%
# #Make sure the leg counts 
# # Flatten the list of lists and count the total number of entries
# total_entries_filtered_master_BCB_RH = sum(len(legs) for legs in filtered_master_BCB_RH)
# print(f"Total entries in filtered_master_BCB_RH: {total_entries_filtered_master_BCB_RH}")
# #%%
# ##Obtaining g(RH) = [1.7 / (1-RH)]^0.31 for all mean RH values 
# ## ie for every leg 

# master_BCB_gRH = []

# # Iterate over each flight in master_BCB_RH
# for flight in master_BCB_RH:
#     flight_gRH = []  # To store the modified data for each flight
    
#     for leg in flight:
#         new_leg = leg.copy()  # Copy the dictionary to preserve the structure
        
#         # Access the single Rh_mean value (it's in a list)
#         rh_mean = new_leg['Rh_mean'][0] / 100.0  # Convert percentage to a decimal
        
#         # Apply the equation to Rh_mean and store the result
#         if np.isnan(rh_mean) or rh_mean >= 1:
#             # If Rh_mean is NaN or greater than or equal to 1, set gRH_value to NaN
#             gRH_value = np.nan
#             print(f"Skipping calculation for Rh_mean = {new_leg['Rh_mean'][0]} as it results in division by zero or invalid value.")
#         else:
#             gRH_value = (1.7 / (1 - rh_mean)) ** 0.31
           

#         new_leg['gRh_mean'] = [gRH_value]
        
#         flight_gRH.append(new_leg)
    
#     master_BCB_gRH.append(flight_gRH)
# #%%
# #only the grh from filtered_master_BCB_RH
# filtered_master_BCB_gRH = []

# # Iterate over each flight in master_BCB_RH
# for flight in filtered_master_BCB_RH:
#     flight_gRH = []  # To store the modified data for each flight
    
#     for leg in flight:
#         new_leg = leg.copy()  # Copy the dictionary to preserve the structure
        
#         # Access the single Rh_mean value (it's in a list)
#         rh_mean = new_leg['Rh_mean'][0] / 100.0  # Convert percentage to a decimal
        
#         # Apply the equation to Rh_mean and store the result
#         if np.isnan(rh_mean) or rh_mean >= 1:
#             # If Rh_mean is NaN or greater than or equal to 1, set gRH_value to NaN
#             gRH_value = np.nan
#             print(f"Skipping calculation for Rh_mean = {new_leg['Rh_mean'][0]} as it results in division by zero or invalid value.")
#         else:
#             gRH_value = (1.7 / (1 - rh_mean)) ** 0.31
           

#         new_leg['gRh_mean'] = [gRH_value]
        
#         flight_gRH.append(new_leg)
    
#     filtered_master_BCB_gRH.append(flight_gRH)
# #%%
# total_entries_filtered_master_BCB_gRH = sum(len(legs) for legs in filtered_master_BCB_gRH)
# print(f"Total entries in filtered_master_BCB_gRH: {total_entries_filtered_master_BCB_gRH}")
# #%%
# #filtered dry intercept calculation

# filtered_master_BCB_interceptdry_dict = {}

# # Flatten master_min_gRH if it's a list of lists
# if isinstance(filtered_master_BCB_gRH[0], list):
#     filtered_master_BCB_gRH = [item for sublist in filtered_master_BCB_gRH for item in sublist]


# unique_keys = set()

# # Flatten master_BCB_exponential
# flattened_exponential = []
# for exp_list in master_BCB_exponential.values():
#     flattened_exponential.extend(exp_list)

# # Create a dictionary for quick lookup
# exponential_dict = {(exp['Date'], exp['BCB_start'], exp['BCB_stop']): exp for exp in flattened_exponential}

# # Loop through each entry in filtered_master_BCB_gRH
# for entry in filtered_master_BCB_gRH:
#     date = entry['Date']
#     BCB_start = entry['BCB_start']
#     BCB_stop = entry['BCB_stop']
#     gRh_mean = entry['gRh_mean'][0]  # Assuming gRh_mean is a list with one value
#     Rh_mean = entry['Rh_mean'][0]  # Assuming Rh_mean is a list with one value


#     # Skip entries with invalid Rh_mean values
#     if Rh_mean < 0:
#         continue

#     # Create a unique key for this entry
#     key = (date, BCB_start, BCB_stop)

#     # Find corresponding exponential parameters
#     if date in master_BCB_exponential:
#         exp_params_list = master_BCB_exponential[date]
#         for exp_params in exp_params_list:
#             n0 = exp_params['n0']
#             D = exp_params['D']
            
            
#             dryintercept = n0 * (gRh_mean)
            
#             # Store the result in the dictionary
#             filtered_master_BCB_interceptdry_dict[key] = {
#                 'Date': date,
#                 'BCB_start': BCB_start,
#                 'BCB_stop': BCB_stop,
#                 'Rh_mean': entry['Rh_mean'],
#                 'gRh_mean': entry['gRh_mean'],
#                 'dry intercept': dryintercept
#             }

# filtered_master_BCB_dryintercept = list(filtered_master_BCB_interceptdry_dict.values())
# print(f"Length of filtered_master_BCB_dryintercept: {len(filtered_master_BCB_dryintercept)}")
# #%%
# #filtered total concentration of droplets with dry diameter larger than ddrymin
# filtered_master_BCB_ntd_dict = {}

# # Flatten master_BCB_gRH if it's a list of lists
# if isinstance(filtered_master_BCB_gRH[0], list):
#     filtered_master_BCB_gRH = [item for sublist in filtered_master_BCB_gRH for item in sublist]


# unique_keys = set()

# # Flatten master_BCB_exponential
# flattened_exponential = []
# for exp_list in master_BCB_exponential.values():
#     flattened_exponential.extend(exp_list)

# # Create a dictionary for quick lookup
# exponential_dict = {(exp['Date'], exp['BCB_start'], exp['BCB_stop']): exp for exp in flattened_exponential}

# ddrymin = 2  # Update this as needed

# # Loop through each entry in master_BCB_gRH
# for entry in filtered_master_BCB_gRH:
#     date = entry['Date']
#     BCB_start = entry['BCB_start']
#     BCB_stop = entry['BCB_stop']
#     gRh_mean = entry['gRh_mean'][0]  # Assuming gRh_mean is a list with one value
#     Rh_mean = entry['Rh_mean'][0]  # Assuming Rh_mean is a list with one value

#     # Skip entries with invalid Rh_mean values
#     if Rh_mean < 0:
#         continue

#     # Create a unique key for this entry
#     key = (date, BCB_start, BCB_stop)

#     # Find corresponding exponential parameters
#     if date in master_BCB_exponential:
#         exp_params_list = master_BCB_exponential[date]
#         for exp_params in exp_params_list:
#             D = exp_params['D']
#             n0 = exp_params['n0']
#             # dryintercept = n0 * gRh_mean
#             # Calculate Ntd
#             Ntd = n0 * D * np.exp(-(gRh_mean * ddrymin / D))
            
#             # Store the result in the dictionary
#             filtered_master_BCB_ntd_dict[key] = {
#                 'Date': date,
#                 'BCB_start': BCB_start,
#                 'BCB_stop': BCB_stop,
#                 'Rh_mean': entry['Rh_mean'],
#                 'gRh_mean': entry['gRh_mean'],
#                 'Ntd': Ntd
#             }

# filtered_master_BCB_ntd = list(filtered_master_BCB_ntd_dict.values())
# print(f"Length of filtered_master_BCB_ntd: {len(filtered_master_BCB_ntd)}")
# #%%
# #filtered total concentration of droplets with dry diameter larger than ddrymin (ntd)
# # related to the ambient concentration (NT)

# filtered_master_BCB_NtdNt_dict = {}
# # Flatten master_BCB_gRH if it's a list of lists
# if isinstance(filtered_master_BCB_gRH[0], list):
#     filtered_master_BCB_gRH = [item for sublist in filtered_master_BCB_gRH for item in sublist]

# ddrymin = 2
# dmin = 2.5  

# unique_keys = set()
# flattened_exponential = []
# for exp_list in master_BCB_exponential.values():
#     flattened_exponential.extend(exp_list)

# exponential_dict = {(exp['Date'], exp['BCB_start'], exp['BCB_stop']): exp for exp in flattened_exponential}

# # Loop through each entry in filtered_master_BCB_gRH
# for entry in filtered_master_BCB_gRH:
#     date = entry['Date']
#     BCB_start = entry['BCB_start']
#     BCB_stop = entry['BCB_stop']
#     gRh_mean = entry['gRh_mean'][0]  
#     Rh_mean = entry['Rh_mean'][0]  

#     if Rh_mean < 0:
#         continue

#     key = (date, BCB_start, BCB_stop)
#     if key in exponential_dict:
#         exp_params = exponential_dict[key]
#         D = exp_params['D']
        
      
#         NtdNt = np.exp((-gRh_mean * ddrymin + dmin) / D)
        
       
#         filtered_master_BCB_NtdNt_dict[key] = {
#             'Date': date,
#             'BCB_start': BCB_start,
#             'BCB_stop': BCB_stop,
#             'Rh_mean': entry['Rh_mean'],
#             'gRh_mean': entry['gRh_mean'],
#             'NtdNt': NtdNt
#         }

# filtered_master_BCB_NtdNt = list(filtered_master_BCB_NtdNt_dict.values())
# print(f"Length of filtered_master_BCB_NtdNt: {len(filtered_master_BCB_NtdNt)}")
# #%%
# #Now we need to pull our windspeed values from the summary data and calculate the corrected windspeeds down to 10m
# master_BCB = []


# for i in range(len(dates_legs)):
#     date = dates_legs[i]

#     leg_dict = leg_data[i]

#     flight_date = leg_dict['Date'] 
#     BCB_start = leg_dict['LegIndex_02']['StartTimes']
#     BCB_stop = leg_dict['LegIndex_02']['StopTimes']
#     sum_flight = summary[i]

#     times = sum_flight.Time_mid.values
#     winds = sum_flight.Wind_Speed.values
#     alts = sum_flight.GPS_altitude.values
    
#     all_BCB_means = []


#     for i in range(len(BCB_start)):
#         index1_start=None
#         index1_end=None  
#         start = int(BCB_start[i])
#         end = BCB_stop[i]

#         wind_alt = {
#             'Date': date,
#             'BCB_start': start,
#             'BCB_end': end,
#             'Alts_mean': [],
#             'Winds_mean': []
#         }

#         for i in range(len(times)):
#             start9 = int(times[i][0:5])
#                 #print(times[i][0:5])
#             if start9 == start:
#                 index1_start = i
#                 break
        
    
#         for i in range(len(times)):
#             end9 = int(times[i][0:5])
#             if end9 == end:
#                 index1_end = i
#                 break
        
#         if index1_start == None:
#                 # print(date)
#                 # print('Did not find start time in Summary')
#             winds9_mean = np.nan
#             alts9_mean = np.nan
#         if index1_end == None:
#                 # print(date)
#             winds9_mean = np.nan
#             alts9_mean = np.nan
#             break
#         else:
#             winds9 = winds[index1_start:index1_end]
                
#             winds9_mean = np.nanmean(winds9)

#             alts9 = alts[index1_start:index1_end]
#             alts9_mean = np.nanmean(alts9)

#         wind_alt['Winds_mean'].append(winds9_mean)
#         wind_alt['Alts_mean'].append(alts9_mean)

#         all_BCB_means.append(wind_alt) #List that contains all the BCB wind/alt mean dictionaries for 1 flight
        
#     master_BCB.append(all_BCB_means) #List that contains all BCB flights  
# #%%
# Z0 = 0.02  # meters (typical value for open ocean)
# Z10 = 10  # target height m

# corrected_calc_bcb = {'Date': [], 'Corrected_bcb_windspeed': []}

# for flight in master_BCB:
#     for wind_alt in flight:
#         date = wind_alt['Date']
#         windspeed = wind_alt['Winds_mean']
#         altitude = wind_alt['Alts_mean']

#         for wind_mean, alt_mean in zip(windspeed, altitude):
#             # Apply the formula
#             new_windspeed = wind_mean * (np.log(Z10/Z0) / np.log(alt_mean / Z0))

#             corrected_calc_bcb['Date'].append(date)
#             corrected_calc_bcb['Corrected_bcb_windspeed'].append(new_windspeed)
# for date, wind_mean in zip(corrected_calc_bcb['Date'], corrected_calc_bcb['Corrected_bcb_windspeed']):
#     print(f"Date: {date}, Corrected_bcb_windspeed: {wind_mean}")
# # %%
# combined_data = {
#     'Date': [],
#     'D': [],
#     'NtdNt': []
# }

# flat_ntdNt = [item for item in filtered_master_BCB_NtdNt]

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
# plt.title('Below cloud base January - June 2022', fontsize=14, fontweight='bold')
# plt.grid(True)
# plt.yscale('log')
# plt.ylim(10**-2, 10**0.5)
# plt.xlim(10**-1, 10**1)
# plt.xscale('log')
# plt.show()
# # %%
# filtered_master_BCB_ddry = []

# # Iterate over each date in master_BCB_exponential and corresponding legs in master_BCB_gRH
# for i, (date, legs_exponential) in enumerate(master_BCB_exponential.items()):
#     # Ensure that we have a corresponding entry in master_BCB_gRH
#     if i >= len(filtered_master_BCB_gRH):
#         # print(f"No corresponding gRh data for date {date}, skipping...")
#         continue
    
#     filtered_legs_grh = filtered_master_BCB_gRH[i]
    
#     for j, (leg_exponential, leg_grh) in enumerate(zip(legs_exponential, filtered_legs_grh)):
#         n0 = leg_exponential['n0']
#         D = leg_exponential['D']
#         gRh_mean = leg_grh['gRh_mean'][0]  # Assuming gRh_mean is a list with one value
#         BCB_start = leg_grh['BCB_start']  # Extract BCB_start for this leg
#         BCB_stop = leg_grh['BCB_stop']      # Extract BCB_stop for this leg
        
#         # Calculate ddry by dividing bin centers by gRh_mean
#         if gRh_mean is not np.nan and gRh_mean != 0:
#             filtered_ddry_values = [center / gRh_mean for center in bin_center]
#         else:
#             filtered_ddry_values = [np.nan] * len(bin_center)
#             print(f"Skipping division for leg {j} on date {date} due to invalid gRh_mean.")
        
#         # Store the results in filtered_master_min_ddry
#         filtered_master_BCB_ddry.append({
#             'Date': date,
#             'BCB_start': BCB_start,
#             'BCB_stop': BCB_stop,
#             'ddry': filtered_ddry_values,
#             'n0': n0,
#             'D': D,
#             'gRh_mean': gRh_mean
#         })

# print(f"Length of filtered_master_BCB_ddry: {len(filtered_master_BCB_ddry)}")
# #%%
# filtered_master_BCB_ddry = []

# # Iterate over each date in master_BCB_exponential and corresponding legs in master_BCB_gRH
# for i, (date, legs_exponential) in enumerate(master_BCB_exponential.items()):
#     # Ensure that we have a corresponding entry in master_BCB_gRH
#     if i >= len(filtered_master_BCB_gRH):
#         print(f"No corresponding gRh data for date {date}, skipping...")
#         continue
    
#     filtered_legs_grh = filtered_master_BCB_gRH[i]
    
#     for j, (leg_exponential, leg_grh) in enumerate(zip(legs_exponential, filtered_legs_grh)):
#         n0 = leg_exponential['n0']
#         D = leg_exponential['D']
        
#         gRh_mean = leg_grh['gRh_mean'][0]  # Assuming gRh_mean is a list with one value
        
#         # Extract BCB_start and BCB_end for this leg
#         BCB_start = leg_grh['BCB_start']
#         BCB_stop = leg_grh['BCB_stop']
        
#         # Print the gRh_mean value for verification
#         print(f"Date: {date}, gRh_mean: {gRh_mean}")
        
#         # Calculate ddry by dividing bin centers by gRh_mean
#         if gRh_mean is not np.nan and gRh_mean != 0:
#             filtered_ddry_values = [center / gRh_mean for center in bin_center]
            
#             # Print some sample calculations to verify correctness
#             print(f"Sample bin centers: {bin_center[:5]}")
#             print(f"Sample filtered_ddry_values: {filtered_ddry_values[:5]}")
            
#         else:
#             filtered_ddry_values = [np.nan] * len(bin_center)
#             print(f"Skipping division for leg {j} on date {date} due to invalid gRh_mean.")
        
#         # Store the results in filtered_master_BCB_ddry
#         filtered_master_BCB_ddry.append({
#             'Date': date,
#             'BCB_start': BCB_start,
#             'BCB_stop': BCB_stop,
#             'filtered ddry': filtered_ddry_values,
#             'n0': n0,
#             'D': D,
#             'gRh_mean': gRh_mean
#         })

# print(f"Length of filtered_master_BCB_ddry: {len(filtered_master_BCB_ddry)}")
# #%%
# filtered_master_BCB_ddry = []

# # Example initialization of bin_center (ensure this matches the original size distribution)
# bin_center = [2.25, 2.75, 3.25, 3.75, 4.5, 5.75, 6.85, 7.55, 9.05, 11.4, 13.8, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5]

# # Print bin_center length and content for verification
# print(f"Full length of bin_center: {len(bin_center)}")
# print(f"Full bin_center values: {bin_center}")

# # Iterate over each date in master_BCB_exponential and corresponding legs in master_BCB_gRH
# for i, (date, legs_exponential) in enumerate(master_BCB_exponential.items()):
#     # Ensure that we have a corresponding entry in master_BCB_gRH
#     if i >= len(filtered_master_BCB_gRH):
#         print(f"No corresponding gRh data for date {date}, skipping...")
#         continue
    
#     filtered_legs_grh = filtered_master_BCB_gRH[i]
    
#     for j, (leg_exponential, filtered_leg_grh) in enumerate(zip(legs_exponential, filtered_legs_grh)):
#         n0 = leg_exponential['n0']
#         D = leg_exponential['D']
        
#         gRh_mean = filtered_leg_grh['gRh_mean'][0] 
        
#         # Extract BCB_start and BCB_stop for this leg
#         BCB_start = filtered_leg_grh['BCB_start']
#         BCB_stop = filtered_leg_grh['BCB_stop']
        
#         # Print the gRh_mean value for verification
#         print(f"Date: {date}, Leg index: {j}, gRh_mean: {gRh_mean}")
        
#         # Print the current leg's data
#         print(f"Processing leg {j} for date {date}")
#         print(f"Length of bin_center: {len(bin_center)}")
#         print(f"Bin centers for date {date}, leg {j}: {bin_center}")
        
#         # Calculate ddry by dividing bin centers by gRh_mean
#         if not np.isnan(gRh_mean) and gRh_mean != 0:
#             filtered_ddry_values = [center / gRh_mean for center in bin_center]
            
#             # Print some sample calculations to verify correctness
#             print(f"Sample bin centers: {bin_center[:5]}")
#             print(f"Calculated filtered_ddry_values: {filtered_ddry_values[:5]}")
#             print(f"Full filtered_ddry_values: {filtered_ddry_values}")
            
#         else:
#             filtered_ddry_values = [np.nan] * len(bin_center)
#             print(f"Skipping division for leg {j} on date {date} due to invalid gRh_mean.")
        
#         # Store the results in filtered_master_BCB_ddry
#         filtered_master_BCB_ddry.append({
#             'Date': date,
#             'Leg_index': j,
#             'BCB_start': BCB_start,
#             'BCB_stop': BCB_stop,
#             'filtered_ddry': filtered_ddry_values,
#             'n0': n0,
#             'D': D,
#             'gRh_mean': gRh_mean
#         })

# print(f"Length of filtered_master_BCB_ddry: {len(filtered_master_BCB_ddry)}")
# #%%
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

# for i, flight in enumerate(master_BCB):
#     for j, wind_alt in enumerate(flight):
#         try:
#             date = wind_alt['Date']
#             wind_mean = wind_alt['Winds_mean'][0]
#             alt_mean = wind_alt['Alts_mean'][0]
#             BCB_start = wind_alt['BCB_start']
#             BCB_stop = wind_alt['BCB_end']
            
            
#             if not np.isnan(alt_mean) and alt_mean > 0:
#                 corrected_windspeed = correct_windspeed(wind_mean, alt_mean)
#             else:
#                 corrected_windspeed = np.nan

#             NtdNt = None
#             for exp_params in filtered_master_BCB_NtdNt:
#                 if exp_params['Date'] == date and exp_params['BCB_start'] == BCB_start and exp_params['BCB_stop'] == BCB_stop:
#                     NtdNt = exp_params['NtdNt']
#                     break
            
            
#             if NtdNt is not None:
#                 print(f"Appending data for Date: {date}, BCB_start: {BCB_start}, BCB_stop: {BCB_stop}")
#                 combined_data['Date'].append(date)
#                 combined_data['BCB_start'].append(BCB_start)
#                 combined_data['BCB_stop'].append(BCB_stop)
#                 combined_data['NtdNt'].append(NtdNt)
#                 combined_data['Windspeed'].append(corrected_windspeed)

#         except IndexError as e:
#             print(f"Index error at i={i}, j={j}: {e}")
#             continue


# df_combined = pd.DataFrame(combined_data)
# print(df_combined.describe())
# df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
# df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]
# plt.figure(figsize=(10, 8))
# sc = plt.scatter(df_with_windspeed['Windspeed'], df_with_windspeed['NtdNt'], 
#                  c=df_with_windspeed['Windspeed'], cmap='viridis', s=100, label='Windspeed')
# plt.scatter(df_nan_windspeed['Windspeed'], df_nan_windspeed['NtdNt'], 
#             color='black', s=100, label='Windspeed NaN')
# plt.colorbar(sc, label='Corrected Windspeed (m/s)')
# plt.xlabel(' Windspeed (m/s)', fontsize=14, fontweight='bold')
# plt.ylabel('Ratio of Total Droplet Concentration of dry droplets \n with diameter larger than 2um d\n to ambient concentration', fontsize=14, fontweight='bold')
# plt.title('Below cloud base January - June 2022', fontsize=14, fontweight='bold')
# plt.yscale('log')
# plt.show()

# # %%
# #25 bins: A common x-axis is created to average the dry size distributions onto
# # We will use this common x-axis to interpolate the dry size distributions. Note that the 
# # number of bins is arbitrary and can be adjusted as needed and the spacing between the bins is equal.  
# def size_distribution(x, dryint, D):
#     dryint = dryint 
#     return dryint * np.exp(-x / D)
# #We are randomly assigning bin lengths here, but we have 25 bins that end at 10 um d or any set diameter you want 
# common_bins = np.linspace(0, 10, 25)

# interpolated_values = []

# # Iterate through entries in filtered_master_BCB_ddry
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
    
#     # Interpolate the distribution
#     interp_func = interp1d(ddry_values, size_distribution(ddry_values, dryint, D), kind='linear', fill_value='extrapolate')
#     interpolated_leg_values = interp_func(common_bins)
    
#     # Store the interpolated values with metadata
#     interpolated_values.append({
#         'Date': date,
#         'Leg_index': leg_index,
#         'BCB_start': BCB_start,
#         'BCB_stop': BCB_stop,
#         'interpolated_values': interpolated_leg_values.tolist()
#     })

# for entry in interpolated_values:
#     print(f"Date: {entry['Date']}, Leg_index: {entry['Leg_index']}, BCB_start: {entry['BCB_start']}, BCB_stop: {entry['BCB_stop']}, Interpolated Values: {entry['interpolated_values']}")


# common_bins = np.linspace(0, 10, 25)
# plt.figure(figsize=(12, 8))
# for entry in interpolated_values:
#     date = entry['Date']
#     leg_index = entry['Leg_index']
#     BCB_start = entry['BCB_start']
#     BCB_stop = entry['BCB_stop']
#     leg_values = entry['interpolated_values']  # Renamed to avoid conflict
#     plt.plot(common_bins, leg_values, label=f"Date: {date}, Leg: {leg_index} ({BCB_start} - {BCB_stop})")

# plt.ylabel('Clear mean droplet concentration (/cm^3/um)', fontweight='bold')
# plt.xlabel('Bin diameter (um)', fontweight='bold')
# plt.title('Below cloud base dry size distribution January-June 2022', fontweight='bold')
# plt.tight_layout()
# line_count = len(plt.gca().get_lines())
# print(f"Total number of lines plotted: {line_count}")
# plt.show()

# # %%
# # Using 4 random windspeed bins 
# problematic_legs = [
#     ('2022-01-26', 12),
#     ('2022-01-11', 4),
#     ('2022-01-11', 8), 
#     ('2022-01-11', 14)
   
# ]
# problematic_set = set(problematic_legs)

# # Function to compute size distribution
# def size_distribution(x, dryint, D):
#     return dryint * np.exp(-x / D)

# # Define common bins
# common_bins = np.linspace(2, 10, 50)

# # Manually define 4 windspeed bins
# windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]

# # Initialize grouped_distributions and mean_windspeeds for the bins
# grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
# mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# # Debug variables
# missing_windspeed_count = 0
# interpolation_failures = 0

# # Total legs
# print(f"Total input legs: {len(filtered_master_BCB_ddry)}")

# # Iterate through the legs
# for i in range(len(filtered_master_BCB_ddry)):
#     entry_ddry = filtered_master_BCB_ddry[i]
#     entry_dryintercept = filtered_master_BCB_dryintercept[i]
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
# plt.yscale('log')
# plt.xscale('log')
# plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
# plt.xlabel('Bin diameter (µm)', fontweight='bold')
# plt.title('Size distribution by wind speed', fontweight='bold')
# plt.legend(title="Average wind speed (m/s)")
# plt.tight_layout()
# plt.show()
# #%%
# #trying to recreate fig 22 in lewis and schwartz
# # Using 4 random windspeed bins 
# problematic_legs = [
#     ('2022-01-26', 12),
#     ('2022-01-11', 4),
#     ('2022-01-11', 8), 
#     ('2022-01-11', 14)
   
# ]
# problematic_set = set(problematic_legs)

# # Function to compute size distribution
# def size_distribution(x, dryint, D):
#     return dryint * np.exp(-x / D)

# # Define common bins
# common_bins = np.linspace(2, 10, 50)

# # Manually define 4 windspeed bins
# windspeed_bins = [(0, 5), (5.001, 7), (7.001, 9), (9.001, np.inf)]

# # Initialize grouped_distributions and mean_windspeeds for the bins
# grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
# mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# # Debug variables
# missing_windspeed_count = 0
# interpolation_failures = 0

# # Total legs
# print(f"Total input legs: {len(filtered_master_BCB_ddry)}")

# # Iterate through the legs
# for i in range(len(filtered_master_BCB_ddry)):
#     entry_ddry = filtered_master_BCB_ddry[i]
#     entry_dryintercept = filtered_master_BCB_dryintercept[i]
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
# plt.yscale('log')
# plt.xscale('log')
# plt.ylabel('Clear mean droplet concentration (/cm³)', fontweight='bold')
# plt.xlabel('Bin diameter (µm)', fontweight='bold')
# plt.title('Size distribution by wind speed', fontweight='bold')
# plt.legend(title="Average wind speed (m/s)")
# plt.tight_layout()
# plt.show()
# #%%

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
#     # Labels and title
#     plt.ylabel('Clear mean droplet concentration (/cm³)', fontweight='bold')
#     plt.xlabel('Bin diameter (µm)', fontweight='bold')
#     plt.title('Below Cloud Base CAS January-June 2022 wind speed 5-7 m/s', fontweight='bold')
#     plt.legend()

#     plt.tight_layout()
#     plt.show()
# else:
#     print("No data available for the 5-7 m/s windspeed bin.")

# #%%
# #How windspeed, slope, and Ntd correlate with windspeed 

# Z0 = 0.02 
# Z10 = 10  
# def correct_windspeed(windspeed, altitude):
#     return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))
# combined_data = {
#     'Date': [],
#     'BCB_start': [],
#     'BCB_stop': [],
#     'Ntd': [],
#     'D': [],
#     'Windspeed': []
# }


# for i, flight in enumerate(master_BCB):
#     for j, wind_alt in enumerate(flight):
#         try:
            
#             date = wind_alt['Date']
#             wind_mean = wind_alt['Winds_mean'][0]
#             alt_mean = wind_alt['Alts_mean'][0]
#             BCB_start = wind_alt['BCB_start']
#             BCB_stop = wind_alt['BCB_end']
            
#             # Correct windspeed using the provided formula
#             if not np.isnan(alt_mean) and alt_mean > 0:
#                 corrected_windspeed = correct_windspeed(wind_mean, alt_mean)
#             else:
#                 corrected_windspeed = np.nan


#             Ntd = None
#             D = None

#             # Find corresponding exponential parameters (D) from master_min_exponential
#             if date in master_BCB_exponential:
#                 for exp_params in master_BCB_exponential[date]:
#                     # Check if the current dictionary matches Min_start and Min_end
#                     if exp_params['BCB_start'] == BCB_start and exp_params['BCB_stop'] == BCB_stop:
#                         D = exp_params['D']
#                         print(f"Found D for Date: {date}, BCB_start: {BCB_start}, BCB_stop: {BCB_stop}, D: {D}")
#                         break

#             # Find corresponding Ntd from filtered_master_min_ntd
#             for ntd_data in filtered_master_BCB_ntd:
#                 if ntd_data['Date'] == date and ntd_data['BCB_start'] == BCB_start and ntd_data['BCB_stop'] == BCB_stop:
#                     Ntd = ntd_data['Ntd']
#                     break

#             # Append data if both Ntd and D are found
#             if Ntd is not None and D is not None:
#                 combined_data['Date'].append(date)
#                 combined_data['BCB_start'].append(BCB_start)
#                 combined_data['BCB_stop'].append(BCB_stop)
#                 combined_data['Ntd'].append(Ntd)
#                 combined_data['D'].append(D)
#                 combined_data['Windspeed'].append(corrected_windspeed)

#         except IndexError as e:
#             print(f"Index error at i={i}, j={j}: {e}")
#             continue

# # Convert the combined data to a DataFrame
# df_combined = pd.DataFrame(combined_data)

# # Check for anomalies or missing values
# print(df_combined.describe())

# # Separate data based on NaN Windspeed
# df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
# df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]

# plt.figure(figsize=(10, 8))
# sc = plt.scatter(df_with_windspeed['D'], df_with_windspeed['Ntd'], 
#                  c=df_with_windspeed['Windspeed'], cmap='viridis', s=100, label='Windspeed')
# plt.scatter(df_nan_windspeed['D'], df_nan_windspeed['Ntd'], 
#             color='black', s=100, label='Windspeed NaN')
# plt.colorbar(sc, label='Corrected Windspeed (m/s)')
# plt.xlabel('Slope (D)', fontsize=14, fontweight='bold')
# plt.ylabel('Total droplet concentration \n dry droplets larger than \n 2um diameter', fontsize=14, fontweight='bold')
# plt.title('Below cloud base January - June 2022', fontsize=14, fontweight='bold')
# plt.yscale('log')
# plt.xscale('log')
# plt.show()
# #%%
# #Average across all bins and then fit a distrubtion using the average 
# common_bins = np.linspace(0, 10, 25)
# # common_bins = np.linspace(0, 20, 25)  old bins from before 2DS import

# #old bins from before 2DS import
# # windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]

# windspeed_bins = [(0, 5), (5.001, 7), (7.001, 9), (9.001, np.inf)]
# grouped_concentrations = {i: [] for i in range(len(windspeed_bins))}  # To accumulate droplet concentrations
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
        
#         # Interpolate the size distribution for this leg
#         size_dist = size_distribution(ddry_values, dryint, D)
#         interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
#         interpolated_leg_values = interp_func(common_bins)
        
#         # Categorize the leg based on windspeed range
#         for idx, (low, high) in enumerate(windspeed_bins):
#             if low <= windspeed <= high:
#                 grouped_concentrations[idx].append(interpolated_leg_values)  # Add size distribution to this bin
#                 mean_windspeeds[idx].append(windspeed)
#                 break

# # Step 2: Average each particle size bin, compute standard deviation for error bars, then fit
# plt.figure(figsize=(12, 8))

# for idx, (low, high) in enumerate(windspeed_bins):
#     if grouped_concentrations[idx]:
        
#         concentrations_array = np.array(grouped_concentrations[idx])
        

#         avg_concentration = np.mean(concentrations_array, axis=0)
        
        
#         std_dev = np.std(concentrations_array, axis=0)
        
        
#         avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)  # Replace zeros with small value
        
        
#         def fit_function(x, dryint, D):
#             return dryint * np.exp(-x / D)
        
#         popt, _ = curve_fit(fit_function, common_bins, avg_concentration, p0=[1, 1])  # Initial guess for dryint, D
        
        
#         fitted_curve = fit_function(common_bins, *popt)
        
        
#         avg_windspeed = np.mean(mean_windspeeds[idx])
#         num_legs = len(grouped_concentrations[idx])
        
       
#         plt.plot(common_bins, fitted_curve, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")
        
        
#         plt.errorbar(common_bins, avg_concentration, yerr=std_dev, fmt='o', capsize=5, label=f"{low}-{high} m/s")

# plt.yscale('log')
# plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
# plt.xlabel('Bin diameter (µm)', fontweight='bold')
# plt.title('Fitted size distribution by wind speed', fontweight='bold')
# plt.legend(title="wind speed (m/s)")
# plt.tight_layout()
# plt.show()

# #changing the number of bins being average over to 10

# common_bins = np.linspace(0, 10, 10)  


# windspeed_bins =[(0, 5), (5.001, 7), (7.001, 9), (9.001, np.inf)]
# #[(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]  old bins from before 2DS import
# grouped_concentrations = {i: [] for i in range(len(windspeed_bins))}  # To accumulate droplet concentrations
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

        
#         size_dist = size_distribution(ddry_values, dryint, D)  # Compute size distribution
#         interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
#         interpolated_leg_values = interp_func(common_bins)  # Interpolate to common bins

        
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

# plt.yscale('log')

# plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
# plt.xlabel('Bin diameter (µm)', fontweight='bold')
# plt.title('Fitted average size distribution by wind speed', fontweight='bold')
# plt.legend(title="Average wind speed (m/s)")
# plt.tight_layout()
# plt.show()

# #%%
# # ##trying to fix size distribution and add ntd 

# # def size_distribution(d, dryint, D):
# #     return dryint * np.exp(-d / D)

# # dmin = 2  
# # dmax = np.inf 

# # integrated_concentrations = []
# # for i in range(len(filtered_master_BCB_ddry)):
# #     entry_ddry = filtered_master_BCB_ddry[i]
# #     entry_dryintercept = filtered_master_BCB_dryintercept[i]  
    
# #     date = entry_ddry['Date']
# #     BCB_start = entry_ddry['BCB_start']
# #     BCB_stop = entry_ddry['BCB_stop']
# #     leg_index = entry_ddry['Leg_index'] 
# #     D = entry_ddry['D']  
# #     dryint = entry_dryintercept['dry intercept']  
    

# #     total_concentration, _ = quad(size_distribution, dmin, dmax, args=(dryint, D))
    
    
# #     integrated_concentrations.append({
# #         'Date': date,
# #         'Leg_index': leg_index,
# #         'BCB_start': BCB_start,
# #         'BCB_stop': BCB_stop,
# #         'Total Concentration': total_concentration
# #     })


# # for total in integrated_concentrations:
# #     print(f"Date: {total['Date']}, Leg_index: {total['Leg_index']}, "
# #           f"Total Concentration: {total['Total Concentration']:.3f} /cm^3/um")

# # dates = [entry['Date'] for entry in integrated_concentrations]
# # legs = [entry['Leg_index'] for entry in integrated_concentrations]
# # total_concs = [entry['Total Concentration'] for entry in integrated_concentrations]

# # plt.figure(figsize=(12, 8))
# # plt.bar(range(len(total_concs)), total_concs, tick_label=[f"{d}-{l}" for d, l in zip(dates, legs)])
# # plt.ylabel('Total Concentration (/cm^3/um)', fontweight='bold')
# # plt.xlabel('Leg (Date-Leg)', fontweight='bold')
# # plt.title('Total below cloud base concentration (/cm^3/um)', fontweight='bold')
# # plt.tight_layout()
# # plt.show()
# #%%
# #Size distributiona using d instead of ddry
# def size_distribution(d, dryint, D):
#     return dryint * np.exp(-d / D)

# dmin = 2  
# dmax = np.inf  


# # common_bins = np.linspace(0, 25, 20) old bins from before 2DS import
# common_bins = np.linspace(2, 10, 25)

# integrated_concentrations = []
# size_distributions = []

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
    
#     distribution_values = size_distribution(common_bins, dryint, D)

#     integrated_concentrations.append({
#         'Date': date,
#         'Leg_index': leg_index,
#         'BCB_start': BCB_start,
#         'BCB_stop': BCB_stop,
#         'Total Concentration': total_concentration
#     })
    
#     size_distributions.append({
#         'Date': date,
#         'Leg_index': leg_index,
#         'BCB_start': BCB_start,
#         'BCB_stop': BCB_stop,
#         'Distribution Values': distribution_values
#     })


# for total in integrated_concentrations:
#     print(f"Date: {total['Date']}, Leg_index: {total['Leg_index']}, "
#           f"Total Concentration: {total['Total Concentration']:.3f} /cm^3")

# plt.figure(figsize=(12, 8))

# for dist in size_distributions:
#     date = dist['Date']
#     leg_index = dist['Leg_index']
#     BCB_start = dist['BCB_start']
#     BCB_stop = dist['BCB_stop']
#     dist_values = dist['Distribution Values']
    
#     plt.plot(common_bins, dist_values, label=f"Date: {date}, Leg: {leg_index} ({BCB_start} - {BCB_stop})")

# plt.ylabel('Total droplet concentration (/cm^3/um)', fontweight='bold')
# plt.xlabel('Bin diameter (um)', fontweight='bold')
# plt.title('Below cloud base January - June 2022', fontweight='bold')
# plt.tight_layout()
# plt.show()
# #%%
# #size distribution using ddry instead of d
# def size_distribution(ddry, N0, D):
#     return N0 * np.exp(-ddry / D)

# # common_bins = np.linspace(0, 25, 20)
# common_bins = np.linspace(2, 16, 25)
# plt.figure(figsize=(12, 8))

# for i in range(len(filtered_master_BCB_ddry)):
#     entry_ddry = filtered_master_BCB_ddry[i]
#     entry_ntd = filtered_master_BCB_ntd[i]  

#     date = entry_ddry['Date']
#     leg_index = entry_ddry['Leg_index']
#     BCB_start = entry_ddry['BCB_start']
#     BCB_stop = entry_ddry['BCB_stop']
#     D = entry_ddry['D']  
#     Ntd = entry_ntd['Ntd'] 
#     ddry_values = np.array(entry_ddry['filtered_ddry']) 
#     N0 = entry_ddry['n0']  
    
#     size_dist_values = size_distribution(ddry_values, dryintercept, D)
    
    
#     interp_func = interp1d(ddry_values, size_dist_values, kind='linear', fill_value='extrapolate')
#     interpolated_leg_values = interp_func(common_bins)

#     plt.plot(common_bins, interpolated_leg_values, label=f"Date: {date}, Leg: {leg_index} ({BCB_start} - {BCB_stop})")

# plt.ylabel('Concentration (/cm^3/μm)', fontweight='bold')
# plt.xlabel('Dry diameter (μm)', fontweight='bold')
# plt.title('Below cloud base January-June 2022', fontweight='bold')
# plt.tight_layout()
# plt.show()
# #%%
# #Dry size distribution function CORRECT

# def size_distribution(ddry, N0, D):
#     return N0 * np.exp(-ddry / D)

# common_bins = np.linspace(2, 10, 10)

# # windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)] old bins from before 2DS import
# windspeed_bins = [(0, 5), (5.001, 7), (7.001, 9), (9.001, np.inf)]
# grouped_concentrations = {i: [] for i in range(len(windspeed_bins))}  
# mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}


# for i in range(len(filtered_master_BCB_ddry)):
#     entry_ddry = filtered_master_BCB_ddry[i]
#     entry_dryintercept = filtered_master_BCB_dryintercept[i]

    
#     date = entry_ddry['Date']
#     BCB_start = entry_ddry['BCB_start']
#     BCB_stop = entry_ddry['BCB_stop']
#     leg_index = entry_ddry['Leg_index']
#     D = entry_ddry['D']  
#     ddry_values = np.array(entry_ddry['filtered_ddry']) 
#     N0 = entry_ddry['n0']  

    
#     windspeed_entry = df_combined[
#         (df_combined['Date'] == date) &
#         (df_combined['BCB_start'] == BCB_start) &
#         (df_combined['BCB_stop'] == BCB_stop)
#     ]
    
#     if not windspeed_entry.empty:
#         windspeed = windspeed_entry['Windspeed'].values[0]

#         size_dist_values = size_distribution(ddry_values, N0, D)
        
       
#         if np.any(np.isnan(size_dist_values)):
#             continue  
        
        
#         interp_func = interp1d(ddry_values, size_dist_values, kind='linear', fill_value='extrapolate')
#         interpolated_leg_values = interp_func(common_bins)  

        
#         for idx, (low, high) in enumerate(windspeed_bins):
#             if low <= windspeed <= high:
#                 grouped_concentrations[idx].append(interpolated_leg_values)
#                 mean_windspeeds[idx].append(windspeed)
#                 break

# # Average each particle size bin for plotting before fitting
# plt.figure(figsize=(12, 8))

# for idx, (low, high) in enumerate(windspeed_bins):
#     if grouped_concentrations[idx]:
       
#         concentrations_array = np.array(grouped_concentrations[idx])
#         avg_concentration = np.mean(concentrations_array, axis=0)
#         std_dev = np.std(concentrations_array, axis=0)
#         avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)  
#         avg_windspeed = np.mean(mean_windspeeds[idx])
#         num_legs = len(grouped_concentrations[idx])

#         plt.errorbar(common_bins, avg_concentration, yerr=std_dev, fmt='o', capsize=5, 
#                      label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linestyle='-', alpha=0.7)


# plt.yscale('log')
# plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
# plt.xlabel('Bin diameter (µm)', fontweight='bold')
# plt.title('Average size distribution by wind speed ', fontweight='bold')
# plt.legend(title="Average wind speed (m/s)")
# plt.tight_layout()
# plt.show()

# #%%
# def size_distribution(ddry, N0, D):
#     return N0 * np.exp(-ddry / D)

# def fit_function(x, N0, D):
#     return N0 * np.exp(-x / D)
# common_bins = np.linspace(2, 10, 10)

# # windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]
# windspeed_bins = [(0, 5), (5.001, 7), (7.001, 9), (9.001, np.inf)]  
# grouped_concentrations = {i: [] for i in range(len(windspeed_bins))}  # To accumulate droplet concentrations
# mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}


# for i in range(len(filtered_master_BCB_ddry)):
#     entry_ddry = filtered_master_BCB_ddry[i]
#     entry_dryintercept = filtered_master_BCB_dryintercept[i]

    
#     date = entry_ddry['Date']
#     BCB_start = entry_ddry['BCB_start']
#     BCB_stop = entry_ddry['BCB_stop']
#     leg_index = entry_ddry['Leg_index']
#     N0 = entry_ddry['n0']
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

       
#         size_dist_values = size_distribution(ddry_values, dryint, D)  

        
#         if np.any(np.isnan(size_dist_values)):
#             continue  

        
#         interp_func = interp1d(ddry_values, size_dist_values, kind='linear', fill_value='extrapolate')
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
# # %%
#%%

           
#%%
#Import the instrument data for the cloud-aerosol spectrometer

#Make sure to only work with bins 12-30 for the coarse mode aerosol
bin_name = ['CAS_Bin00', 'CAS_Bin01', 'CAS_Bin02', 'CAS_Bin03',
            'CAS_Bin04', 'CAS_Bin05', 'CAS_Bin06', 'CAS_Bin07', 
            'CAS_Bin08', 'CAS_Bin09', 'CAS_Bin10', 'CAS_Bin11',
            'CAS_Bin12' ,'CAS_Bin13', 'CAS_Bin14', 'CAS_Bin15', 
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
        for col in ['Time_mid', 'LWC_CAS', 'CAS_Bin00', 'CAS_Bin01', 'CAS_Bin02', 
                    'CAS_Bin03',
            'CAS_Bin04', 'CAS_Bin05', 'CAS_Bin06', 'CAS_Bin07', 
            'CAS_Bin08', 'CAS_Bin09', 'CAS_Bin10', 'CAS_Bin11',
            'CAS_Bin12', 'CAS_Bin13', 'CAS_Bin14', 
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
        df_CAS['CAS_Bin00']=pd.to_numeric(df_CAS['CAS_Bin00'], errors='coerce')
        df_CAS['CAS_Bin01']= pd.to_numeric(df_CAS['CAS_Bin01'], errors='coerce')
        df_CAS['CAS_Bin02']= pd.to_numeric(df_CAS['CAS_Bin02'], errors='coerce')
        df_CAS['CAS_Bin03']= pd.to_numeric(df_CAS['CAS_Bin03'], errors='coerce')
        df_CAS['CAS_Bin04']= pd.to_numeric(df_CAS['CAS_Bin04'], errors='coerce')
        df_CAS['CAS_Bin05']= pd.to_numeric(df_CAS['CAS_Bin05'], errors='coerce')
        df_CAS['CAS_Bin06']= pd.to_numeric(df_CAS['CAS_Bin06'], errors='coerce')
        df_CAS['CAS_Bin07']= pd.to_numeric(df_CAS['CAS_Bin07'], errors='coerce')
        df_CAS['CAS_Bin08']= pd.to_numeric(df_CAS['CAS_Bin08'], errors='coerce')
        df_CAS['CAS_Bin09']= pd.to_numeric(df_CAS['CAS_Bin09'], errors='coerce')
        df_CAS['CAS_Bin10']= pd.to_numeric(df_CAS['CAS_Bin10'], errors='coerce')
        df_CAS['CAS_Bin11']= pd.to_numeric(df_CAS['CAS_Bin11'], errors='coerce')
      
        

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

    
    CAS_times = CAS_flight['Time_mid'].values
    CAS_lwc = CAS_flight['LWC_CAS'].values
    CAS_bins = {f'CAS_Bin{bin_label:02d}': CAS_flight[f'CAS_Bin{bin_label:02d}'].values for bin_label in range(0, 30)}

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

                
                lwc_label = 'Y' if 0 <= lwc_val <= 0.0025 else 'N'
                N_label = 'Y' if 0 <= N_val <= 100 else 'N'
                label = 'Y' if lwc_label == 'Y' and N_label == 'Y' else 'N'
                data_labels.append(label)

                
                bin_values = [CAS_bins[f'CAS_Bin{bin_label:02d}'][CAS_idx] for bin_label in range(0, 30)]
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
# %%
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

    # Pre-fetch bin values for range 0 to 29
    bins = {
        f'CAS_Bin{bin_label:02d}': np.array(CAS_flight[f'CAS_Bin{bin_label:02d}'], dtype=float)
        for bin_label in range(0, 30)
    }

    total_BCB_means = []

    for k in range(len(BCB_start)):
        start20 = BCB_start[k]
        end20 = BCB_stop[k]

        # ✅ Initialize bin_means with empty lists to store values
        bin_means = {f'Bin{bin_label:02d}_{label}_mean': [] for bin_label in range(0, 30) for label in ['Y', 'N']}
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

                for bin_label in range(0, 30):
                    bin_key = f'Bin{bin_label:02d}_{label}_mean'
                    bin_means[bin_key].append(bins[f'CAS_Bin{bin_label:02d}'][cas_idx])  # ✅ Append correctly

        # ✅ Convert lists to means
        for bin_label in range(0, 30):
            for label in ['Y', 'N']:
                bin_key = f'Bin{bin_label:02d}_{label}_mean'
                if bin_means[bin_key]:  # Only calculate mean if there are values
                    bin_means[bin_key] = np.nanmean(bin_means[bin_key])
                else:
                    bin_means[bin_key] = np.nan  # If no values, keep it as NaN

        total_BCB_means.append(bin_means)

    master_CAS_BCB.append(total_BCB_means)

# ✅ Print output to check results
for item in master_CAS_BCB:
    for bin_means in item:
        print(f"Date: {bin_means['Date']}, Start: {bin_means['BCB_start']}, Stop: {bin_means['BCB_stop']}")
        for bin_label in range(0, 30):
            for label in ['Y', 'N']:
                bin_key = f'Bin{bin_label:02d}_{label}_mean'
                print(f"   {bin_key}: {bin_means[bin_key]}")  # ✅ Print final means correctly

#%%
# Count the total number of legs from master_CAS_BCB
total_legs = sum(len(item) for item in master_CAS_BCB)
print(f"Total number of legs: {total_legs}")
#%%
bin_centers_array = np.array(bin_center, dtype=float)
#%%
#Now we need to apply our conversion from dNdlog10D to dNdD for each bin and calculate the mean concentration
Y_BCB_calc = []
N_BCB_calc = []

for flight_data in master_CAS_BCB:
    for bin_means in flight_data:
        Y_calc = {'Date': bin_means['Date'], 'BCB_start': bin_means['BCB_start'], 'BCB_stop': bin_means['BCB_stop']}
        N_calc = {'Date': bin_means['Date'], 'BCB_start': bin_means['BCB_start'], 'BCB_stop': bin_means['BCB_stop']}
        
        for bin_label in range(0, 30):
            bin_key_Y = f'Bin{bin_label:02d}_Y_mean'  # Ensure consistent key formatting
            bin_key_N = f'Bin{bin_label:02d}_N_mean'

            # ✅ Ensure the keys exist before computing
            Y_value = bin_means.get(bin_key_Y, np.nan)
            N_value = bin_means.get(bin_key_N, np.nan)

            Y_calc[bin_key_Y] = np.nanmean(Y_value) if not np.isnan(Y_value).all() else np.nan
            N_calc[bin_key_N] = np.nanmean(N_value) if not np.isnan(N_value).all() else np.nan

        Y_BCB_calc.append(Y_calc)
        N_BCB_calc.append(N_calc)

# ✅ Print output to verify correctness
print("✅ Successfully processed Y_BCB_calc and N_BCB_calc!")
print(f"Total Y entries: {len(Y_BCB_calc)}, Total N entries: {len(N_BCB_calc)}")

# %%
plt.figure(figsize=(8, 6))
for entry in Y_BCB_calc:
    bin_means = np.array([entry.get(f'Bin{i}_Y_mean', np.nan) for i in range(0, 30)], dtype=float)  
    valid_indices = ~np.isnan(bin_means)  
    bin_centers_valid = np.array(bin_center)[valid_indices]
    bin_means_valid = bin_means[valid_indices]
    print(f"Valid indices found: {np.sum(valid_indices)}")

    plt.plot(bin_centers_valid, bin_means_valid, color='black', alpha=0.5)

# Labels and formatting
plt.xlabel("Deliquesed Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r" CAS Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-7, 10**1)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base January - June 2022\n Raw Ambient Size Distributions", fontsize=14, fontweight="bold")

plt.show()
#%%
plt.figure(figsize=(8, 6))

for entry in Y_BCB_calc:
    bin_means = np.array([entry.get(f'Bin{i:02d}_Y_mean', np.nan) for i in range(29)], dtype=float)  # ✅ Exclude Bin 29  
    valid_indices = ~np.isnan(bin_means)  

    # ✅ Skip entries where too many bins are missing
    if np.sum(valid_indices) < 10:  
        continue  

    bin_centers_valid = np.array(bin_center[:29])[valid_indices]  # ✅ Exclude Bin 29
    bin_means_valid = bin_means[valid_indices]

    plt.plot(bin_centers_valid, bin_means_valid, color='black', alpha=0.5)

# Labels and formatting
plt.xlabel("Deliquesced Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"CAS Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
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

    # Mask: Remove NaNs and zero values
    valid_indices = (bin_means > 0) & ~np.isnan(bin_means)  
    bin_centers_valid = bin_centers[valid_indices]
    bin_means_valid = bin_means[valid_indices]

    if len(bin_centers_valid) > 0:  # Only plot if valid data exists
        plt.plot(bin_centers_valid, bin_means_valid, color='black', alpha=0.5)

# Labels and formatting
plt.xlabel("Deliquesced Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"CAS Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-7, 10**1)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base January - June 2022\n Raw Ambient Size Distributions", fontsize=14, fontweight="bold")

plt.show()

#%%

#Fitting an exponential to each size distribution


# Define the exponential function
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

# Dictionary to store fitted parameters
ambient_fits = []

# Create figure for plotting
plt.figure(figsize=(8, 6))

# Process and fit each leg's data
for entry in Y_BCB_calc:
    bin_means = np.array([entry.get(f'Bin{i}_Y_mean', np.nan) for i in range(0, 30)], dtype=float)
    valid_indices = ~np.isnan(bin_means)  # Mask for valid (non-NaN) values
    bin_centers_valid = np.array(bin_center)[valid_indices]
    bin_means_valid = bin_means[valid_indices]

    # Skip if no valid data
    if not valid_indices.any():
        print(f"No valid data for date {entry['Date']}")
        continue

    # Fit exponential function
    try:
        popt, _ = curve_fit(exponential, bin_centers_valid, bin_means_valid, p0=(1, 1), maxfev=5000)
        n0, D = popt

        # Store fitted parameters
        ambient_fits.append({
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
plt.xlabel("Deliquesed Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"CAS Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base January - June 2022\n Exponential Fit to Ambient Size Distributions", fontsize=14, fontweight="bold")
plt.show()
#%%
#weighting all distributions
# Define the exponential function
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

# Dictionary to store fitted parameters for all legs
ambient_fits_poisson_1 = []

# Create figure for plotting
plt.figure(figsize=(8, 6))

# Process and fit each leg's data
for entry in Y_BCB_calc:
    bin_means = np.array([entry.get(f'Bin{i}_Y_mean', np.nan) for i in range(12, 30)], dtype=float)
    valid_indices = ~np.isnan(bin_means)  # Mask for valid (non-NaN) values
    bin_centers_valid = np.array(bin_center)[valid_indices]
    bin_means_valid = bin_means[valid_indices]

    # Skip if no valid data
    if not valid_indices.any():
        print(f"No valid data for date {entry['Date']}")
        continue

    # Compute Poisson standard deviation (sigma = sqrt(dn/dd))
    # Compute bin widths (approximate by taking midpoints)
    bin_widths = np.diff(bin_center, prepend=bin_center[0])

# Poisson sigma (accounting for bin width variation)
    sigma = np.sqrt(bin_means_valid) / bin_widths[valid_indices]


    # Fit exponential function using Poisson-weighted curve fitting
    try:
        popt, _ = curve_fit(exponential, bin_centers_valid, bin_means_valid, 
                            p0=(max(bin_means_valid), 3), 
                            sigma=sigma, absolute_sigma=True, 
                            maxfev=10000)
        n0, D = popt

        # Store fitted parameters
        ambient_fits_poisson_1.append({
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
plt.xlabel("Deliquesed Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"CAS Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-33, 10**1)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base January - June 2022\n Weighted Fit to Ambient Size Distributions", fontsize=14, fontweight="bold")

plt.show()
#%%
# Define the exponential function
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

# Dictionary to store fitted parameters
ambient_fits_poisson = []
ambient_fits_regular = []

# Create figure for plotting
plt.figure(figsize=(8, 6))

# Process and fit each leg's data
for entry in Y_BCB_calc:
    bin_means = np.array([entry.get(f'Bin{i}_Y_mean', np.nan) for i in range(12, 30)], dtype=float)
    valid_indices = ~np.isnan(bin_means)  # Mask for valid (non-NaN) values
    bin_centers_valid = np.array(bin_center)[valid_indices]
    bin_means_valid = bin_means[valid_indices]

    # Skip if no valid data
    if not valid_indices.any():
        print(f"No valid data for date {entry['Date']}")
        continue

    # **Regular Fit (No Poisson Weighting)**
    try:
        popt_reg, _ = curve_fit(exponential, bin_centers_valid, bin_means_valid, p0=(1, 1), maxfev=5000)
        n0_reg, D_reg = popt_reg

        # Store regular fit parameters
        ambient_fits_regular.append({
            'Date': entry['Date'],
            'BCB_start': entry.get('BCB_start', np.nan),
            'BCB_stop': entry.get('BCB_stop', np.nan),
            'Intercept_n0': n0_reg,
            'E_folding_D': D_reg
        })

        # Generate fitted curve
        x_fit = np.linspace(min(bin_centers_valid), max(bin_centers_valid), 100)
        y_fit = exponential(x_fit, *popt_reg)

        # Plot the regular fit in **blue**
        plt.plot(x_fit, y_fit, color='black', alpha=0.1)

    except RuntimeError:
        print(f"Regular fit could not be performed for date {entry['Date']}")

    # **Poisson-Weighted Fit**
    try:
        # Compute bin widths (approximate by taking midpoints)
        bin_widths = np.diff(bin_center, prepend=bin_center[0])

# Poisson sigma (accounting for bin width variation)
        sigma = np.sqrt(bin_means_valid) / bin_widths[valid_indices]


        popt_pois, _ = curve_fit(exponential, bin_centers_valid, bin_means_valid, 
                                 p0=(max(bin_means_valid), 3), 
                                 sigma=sigma, absolute_sigma=True, 
                                 maxfev=10000)
        n0_pois, D_pois = popt_pois

        # Store Poisson-weighted fit parameters
        ambient_fits_poisson.append({
            'Date': entry['Date'],
            'BCB_start': entry.get('BCB_start', np.nan),
            'BCB_stop': entry.get('BCB_stop', np.nan),
            'Intercept_n0': n0_pois,
            'E_folding_D': D_pois
        })

        # Generate fitted curve
        y_fit_pois = exponential(x_fit, *popt_pois)

        # Plot the Poisson-weighted fit in **green**
        plt.plot(x_fit, y_fit_pois, color='red', alpha=0.1)

    except RuntimeError:
        print(f"Poisson-weighted fit could not be performed for date {entry['Date']}")

# Formatting and labels
plt.xlabel("Deliquesed Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-33, 10**1)
regular_fit_patch = mpatches.Patch(color='black', label='Regular Fit')  # Darker black
poisson_fit_patch = mpatches.Patch(color='red', label='Poisson-Weighted Fit')  # Defined red

# Add the custom legend with bold text
plt.legend(handles=[regular_fit_patch, poisson_fit_patch], loc='lower left', fontsize=12, frameon=True)

plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base January - June 2022\n Comparison of Ambient Fits", 
          fontsize=14, fontweight="bold")

plt.show()
#%%
#Poisson histogram normal
#histogram of slopes
# List to store slope values
poisson_slope_normal = []
for fit in ambient_fits_poisson_1:
    if 'E_folding_D' in fit and not np.isnan(fit['E_folding_D']):
        poisson_slope_normal.append(fit['E_folding_D'])
# Plot histogram of slopes
plt.figure(figsize=(8, 6))
plt.hist(poisson_slope_normal, bins=21, color='blue', alpha=0.7)
plt.xlabel('Slope (um)', fontsize=14, fontweight="bold")
plt.ylabel('Frequency of flight legs', fontsize=14, fontweight="bold")
plt.title('Poisson-weighted Ambient Size Distributions', fontsize=14, fontweight="bold")
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.show()
#%%
ambient_slope_normal_p = []
for fit in ambient_fits_regular:
    if 'E_folding_D' in fit and not np.isnan(fit['E_folding_D']):
        ambient_slope_normal_p.append(fit['E_folding_D'])
# Plot histogram of slopes
plt.figure(figsize=(8, 6))
plt.hist(ambient_slope_normal_p, bins=21, color='blue', alpha=0.7)
plt.xlabel('Slope (um)', fontsize=14, fontweight="bold")
plt.ylabel('Frequency of flight legs', fontsize=14, fontweight="bold")
plt.title('Fitted Ambient Size Distributions', fontsize=14, fontweight="bold")
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.show()
#%%
#poisson and ambient together 
plt.figure(figsize=(8, 6))

bins = np.arange(0, 21, 1)  # Ensure bins only go from 0 to 20

plt.hist(ambient_slope_normal_p, bins=bins, color='blue', alpha=0.3, label='Full Range')
plt.hist(poisson_slope_normal, bins=bins, color='red', alpha=0.5, label='Poisson-Weighted')
plt.xlabel('Slope (µm)', fontsize=14, fontweight="bold")
plt.ylabel('Frequency of flight legs', fontsize=14, fontweight="bold")
plt.title('Ambient Size Distributions', fontsize=14, fontweight="bold")
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.legend()

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
    valid_indices = (bin_center <= 10) & ~np.isnan(bin_means)  # **Only keep data ≤ 10 µm**
    bin_centers_valid = np.array(bin_center)[valid_indices]
    bin_means_valid = bin_means[valid_indices]

    if valid_indices.any():
        try:
            popt, _ = curve_fit(exponential, bin_centers_valid, bin_means_valid, 
                                p0=(1, 1), maxfev=5000, 
                                bounds=([0, 0.1], [np.inf, 20]))  # Constrain D
            n0, D = popt

            # Detect extreme slopes (even within bounds)
            if D > 15:  # Change threshold as needed
                print(f"⚠️ High slope detected! Date: {entry['Date']}, D: {D:.2f}")

        except RuntimeError:
            print(f"❌ Fit failed for date {entry['Date']}")
            n0, D = np.nan, np.nan  # Assign NaN if fitting fails
    else:
        n0, D = np.nan, np.nan  # No valid data, store NaN values


    # Store fitted parameters
    ambient_fits_10.append({
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
plt.xlabel("Deliquesed Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.ylim(10**-33, 10**1)
plt.title("Below Cloud Base January - June 2022\n Exponential Fit Ambient Size Distributions (≤10 µm)", fontsize=14, fontweight="bold")
plt.show()

print(f"Total successful ambient exponential fits: {np.sum(~np.isnan([fit['E_folding_D'] for fit in ambient_fits]))}")
#%%
#histogram for slope for 10um
# List to store slope values
ambient_slope_10 = []
for fit in ambient_fits_10:
    if 'E_folding_D' in fit and not np.isnan(fit['E_folding_D']):
        ambient_slope_10.append(fit['E_folding_D'])
# Plot histogram of slopes
plt.figure(figsize=(8, 6))
plt.hist(ambient_slope_10, bins=20, color='blue', alpha=0.7)
plt.xlabel('Slope (um)', fontsize=14, fontweight="bold")
plt.ylabel('Frequency of flight legs', fontsize=14, fontweight="bold")
plt.title('Fitted Ambient Size Distributions (≤10 µm)', fontsize=14, fontweight="bold")
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.show()
#%%
#histogram of ambient_slope_10 and ambient_slope_normal

plt.figure(figsize=(8, 6))
ambient_slope_normal = []
for fit in ambient_fits:
    if 'E_folding_D' in fit and not np.isnan(fit['E_folding_D']):
        ambient_slope_normal.append(fit['E_folding_D'])
bins = np.arange(0, 21, 1)  # Ensure bins only go from 0 to 20

plt.hist(ambient_slope_normal, bins=bins, color='blue', alpha=0.3, label='Full Range')
plt.hist(ambient_slope_10, bins=bins, color='red', alpha=0.5, label='≤10 µm')

plt.xlabel('Slope (µm)', fontsize=14, fontweight="bold")
plt.ylabel('Frequency of flight legs', fontsize=14, fontweight="bold")
plt.title('Fitted Ambient Size Distributions', fontsize=14, fontweight="bold")
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.legend()

plt.show()
#%%
# List to store slope values
min_value = min(min(ambient_slope_10), min(poisson_slope_normal))
max_value = max(max(ambient_slope_10), max(poisson_slope_normal))

bins = np.linspace(min_value, max_value, 21) 
# Plot histogram of slopes
plt.figure(figsize=(8, 6))
plt.hist(poisson_slope_normal, bins=bins, color='red', alpha=0.7, label='Poisson-Weighted', linewidth=1.2, hatch='.')
plt.hist(ambient_slope_normal_p, bins=bins, color='blue', alpha=0.5, label='Full Range')
plt.hist(ambient_slope_10, bins=bins, color='green', alpha=0.3, label='≤10 µm', linewidth=1.2, hatch='x')
plt.xlabel('Slope (um)', fontsize=14, fontweight="bold")
plt.ylabel('Frequency of flight legs', fontsize=14, fontweight="bold")
plt.title('Fitted Ambient Size Distributions', fontsize=14, fontweight="bold")
plt.legend()
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.show()
#%%

#%%
#Overlaying the fits

# # Define the exponential function
# def exponential(x, n0, D):
#     return n0 * np.exp(-x / D)

# # Create figure for plotting
# plt.figure(figsize=(8, 6))

# # Dictionary to store fitted parameters
# ambient_fits_full = []
# ambient_fits_filtered = []

# for entry in Y_BCB_calc:
#     # Extract bin means
#     bin_means = np.array([entry.get(f'Bin{i}_Y_mean', np.nan) for i in range(12, 30)], dtype=float)
#     bin_centers = np.array(bin_center)

#     valid_indices_full = ~np.isnan(bin_means)  # Original valid indices
#     valid_indices_filtered = (bin_centers <= 10) & ~np.isnan(bin_means)  # Only keep data ≤ 10µm

#     bin_centers_full = bin_centers[valid_indices_full]
#     bin_means_full = bin_means[valid_indices_full]

#     bin_centers_filtered = bin_centers[valid_indices_filtered]
#     bin_means_filtered = bin_means[valid_indices_filtered]

#     # Skip if no valid data
#     if len(bin_centers_full) < 5 or len(bin_centers_filtered) < 3:
#         print(f"Skipping {entry['Date']} due to insufficient valid data.")
#         continue

#     try:
#         # Fit for full range
#         popt_full, _ = curve_fit(exponential, bin_centers_full, bin_means_full, p0=(1, 1), maxfev=5000)
#         n0_full, D_full = popt_full
#         ambient_fits_full.append({'Date': entry['Date'], 'BCB_start': entry['BCB_start'], 'BCB_stop': entry['BCB_stop'],
#                                   'Intercept_n0': n0_full, 'E_folding_D': D_full})

#         # Fit for filtered range (≤10µm)
#         popt_filtered, _ = curve_fit(exponential, bin_centers_filtered, bin_means_filtered, p0=(1, 1), maxfev=5000)
#         n0_filtered, D_filtered = popt_filtered
#         ambient_fits_filtered.append({'Date': entry['Date'], 'BCB_start': entry['BCB_start'], 'BCB_stop': entry['BCB_stop'],
#                                       'Intercept_n0': n0_filtered, 'E_folding_D': D_filtered})

#         # Generate fitted curves
#         x_fit_full = np.linspace(min(bin_centers_full), max(bin_centers_full), 100)
#         y_fit_full = exponential(x_fit_full, *popt_full)

#         x_fit_filtered = np.linspace(min(bin_centers_filtered), max(bin_centers_filtered), 100)
#         y_fit_filtered = exponential(x_fit_filtered, *popt_filtered)

#         # Plot full fit in gray (background)
#         plt.plot(x_fit_full, y_fit_full, color='gray', alpha=0.3, linewidth=1.5, label="Full Fit" if entry == Y_BCB_calc[0] else "")

#         # Plot filtered fit in blue (highlighted)
#         plt.plot(x_fit_filtered, y_fit_filtered, color='blue', alpha=0.8, linewidth=2, linestyle='-', label="Filtered Fit (≤10 µm)" if entry == Y_BCB_calc[0] else "")

#     except RuntimeError:
#         print(f"Fit could not be performed for date {entry['Date']}")

# # Formatting
# plt.xlabel("Deliquesced Diameter (μm)", fontsize=12, fontweight="bold")
# plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=12, fontweight="bold")
# plt.yscale("log")
# plt.xticks(fontweight="bold", fontsize=10)
# plt.yticks(fontweight="bold", fontsize=10)
# plt.title("Below Cloud Base January - June 2022\nOverlay of Full and Filtered Exponential Fits", fontsize=14, fontweight="bold")

# # Add legend
# plt.legend()
# plt.show()


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
# Dictionary to store total concentrations for each leg
total_concentration_cm3 = []

# Process total concentration for each leg
for entry in Y_BCB_calc_cm3:
    total_Y_concentration = np.nansum([entry[f'Bin{i}_Y_mean'] for i in range(12, 30)])  # Sum all valid bin concentrations

    total_concentration_cm3.append({
        'Date': entry['Date'],
        'BCB_start': entry['BCB_start'],
        'BCB_stop': entry['BCB_stop'],
        'Total_Y_Concentration_cm3': total_Y_concentration
    })
#%% 
#Recreating C-R 1a
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ✅ Create a dictionary for fast lookup of corrected wind speed based on (Date, BCB_start, BCB_stop)
wind_speed_dict = {
    (row['Date'], row['BCB_start'], row['BCB_stop']): row['Windspeed']
    for _, row in df_combined.iterrows()
}

# ✅ Lists to store matched wind speeds and total concentrations
matched_wind_speeds = []
matched_total_concentrations = []

# ✅ Match each total concentration with its corrected wind speed
for entry in total_concentration_cm3:
    key = (entry['Date'], entry['BCB_start'], entry['BCB_stop'])

    if key in wind_speed_dict:
        matched_total_concentrations.append(entry['Total_Y_Concentration_cm3'])
        matched_wind_speeds.append(wind_speed_dict[key])

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
plt.title("Total Concentration vs. Corrected Wind Speed", fontsize=14, fontweight='bold')

# ✅ Add reference lines if needed (adjust values based on expectations)
plt.axhline(0.05, color='red', linestyle='--', label="Reference Min (0.05 cm⁻³)")
plt.axhline(0.3, color='blue', linestyle='--', label="Reference Max (0.3 cm⁻³)")
plt.legend()

plt.tight_layout()
plt.show()
#%%
#Calculating correlation and fitting a line
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
plt.title("Ambient Concentration vs. 10m Wind Speed", fontsize=16, fontweight='bold')

# ✅ Optional: Add reference lines if applicable
plt.axhline(0.05, color='red', linestyle='--', label="Reference Min (0.05 cm⁻³)")
plt.axhline(0.3, color='blue', linestyle='--', label="Reference Max (0.3 cm⁻³)")
plt.legend()  # Display legend with equation
plt.xticks(fontweight='bold', fontsize=14)
plt.yticks(fontweight='bold', fontsize=14)  
plt.tight_layout()
plt.show()

# %%
#One June ambient leg 

# Define the specific leg to plot
selected_date = "2022-06-10"
selected_start = 51245.0
selected_stop = 51433.0

# Find the corresponding ambient size distribution entry in Y_BCB_calc
selected_leg = next(
    (entry for entry in Y_BCB_calc 
     if entry['Date'] == selected_date and entry['BCB_start'] == selected_start and entry['BCB_stop'] == selected_stop),
    None
)

# Find the fitted parameters for this leg in ambient_fits
selected_fit = next(
    (fit for fit in ambient_fits 
     if fit['Date'] == selected_date and fit['BCB_start'] == selected_start and fit['BCB_stop'] == selected_stop),
    None
)

# Ensure both the raw data and the fit parameters exist
if selected_leg is not None and selected_fit is not None:
    # Extract fitted parameters
    n0, D = selected_fit['Intercept_n0'], selected_fit['E_folding_D']

    # Extract bin means and remove NaNs
    bin_means = np.array([selected_leg.get(f'Bin{i}_Y_mean', np.nan) for i in range(12, 30)], dtype=float)
    valid_indices = ~np.isnan(bin_means)
    bin_centers_valid = np.array(bin_center)[valid_indices]
    bin_means_valid = bin_means[valid_indices]

    # Generate exponential fit line
    x_fit = np.linspace(min(bin_centers_valid), max(bin_centers_valid), 100)
    y_fit = exponential(x_fit, n0, D)

    # Plot the ambient size distribution (raw data)
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers_valid, bin_means_valid, color='blue', marker='o', linestyle='-', label="Observed Data")
    plt.plot(x_fit, y_fit, color='red', linestyle='--', label=f"Fit: y = {n0:.2e} * exp(-x / {D:.2f})")

    # Labels and title
    plt.xlabel("Deliquesced Diameter (μm)", fontsize=14, fontweight="bold")
    plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
    plt.title(f"Ambient Size Distribution - {selected_date}\nStart: {selected_start} | Stop: {selected_stop}", 
              fontsize=12, fontweight="bold")
    plt.legend()
    plt.yscale('log')  
    plt.xticks(fontweight="bold", fontsize=12)
    plt.yticks(fontweight="bold", fontsize=12)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.ylim(10**-33, 10**1)
    plt.show()

   
    print(f"Selected Date: {selected_date}")
    print(f"Start Time: {selected_start}, Stop Time: {selected_stop}")
    print(f"Intercept (n0): {n0:.2e}, E-folding Diameter (D): {D:.2f} μm")

else:
    print(f"Leg not found or missing fitted parameters for {selected_date} with start {selected_start} and stop {selected_stop}.")
#%%
#one june leg 
# Define the specific leg to plot
selected_date = "2022-06-10"
selected_start = 51245.0
selected_stop = 51433.0

# Find the corresponding ambient size distribution entry in Y_BCB_calc
selected_leg = next(
    (entry for entry in Y_BCB_calc 
     if entry['Date'] == selected_date and entry['BCB_start'] == selected_start and entry['BCB_stop'] == selected_stop),
    None
)

# Find the fitted parameters for this leg in ambient_fits
selected_fit = next(
    (fit for fit in ambient_fits 
     if fit['Date'] == selected_date and fit['BCB_start'] == selected_start and fit['BCB_stop'] == selected_stop),
    None
)

# Ensure both the raw data and the fit parameters exist
if selected_leg is not None and selected_fit is not None:
    # Extract fitted parameters
    n0, D = selected_fit['Intercept_n0'], selected_fit['E_folding_D']

    # Extract bin means and remove NaNs
    bin_means = np.array([selected_leg.get(f'Bin{i}_Y_mean', np.nan) for i in range(12, 30)], dtype=float)
    valid_indices = ~np.isnan(bin_means)
    bin_centers_valid = np.array(bin_center)[valid_indices]
    bin_means_valid = bin_means[valid_indices]

    # **Filter data to only include points where bin center ≤ 10 µm**
    mask_10um = bin_centers_valid <= 10
    bin_centers_10um = bin_centers_valid[mask_10um]
    bin_means_10um = bin_means_valid[mask_10um]

    # Fit exponential function only on this subset
    try:
        popt, _ = curve_fit(exponential, bin_centers_10um, bin_means_10um, p0=(1, 1), maxfev=5000)
        n0_10um, D_10um = popt

        # Generate exponential fit only up to 10 µm
        x_fit = np.linspace(min(bin_centers_10um), 10, 100)
        y_fit = exponential(x_fit, n0_10um, D_10um)

        # Plot observed data (entire range)
        plt.figure(figsize=(8, 6))
        plt.plot(bin_centers_valid, bin_means_valid, color='blue', marker='o', linestyle='-', alpha=0.4, label="Observed Data")

        # Plot observed data ≤ 10 µm in bold
        plt.plot(bin_centers_10um, bin_means_10um, color='blue', marker='o', linestyle='-', label="Observed Data ≤ 10 µm")

        # Plot the exponential fit only up to 10 µm
        plt.plot(x_fit, y_fit, color='red', linestyle='--', label=f"Fit (≤10µm): y = {n0_10um:.2e} * exp(-x / {D_10um:.2f})")

        # Labels and title
        plt.xlabel("Deliquesced Diameter (μm)", fontsize=14, fontweight="bold")
        plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
        plt.title(f"Ambient Size Distribution - {selected_date}\nStart: {selected_start} | Stop: {selected_stop}", 
                  fontsize=12, fontweight="bold")
        plt.legend()
        plt.yscale('log')  
        plt.xticks(fontweight="bold", fontsize=14)
        plt.yticks(fontweight="bold", fontsize=14)  
        plt.ylim(10**-33, 10**1)
        # Show the plot
        plt.show()

        # Print selected leg details
        print(f"Selected Date: {selected_date}")
        print(f"Start Time: {selected_start}, Stop Time: {selected_stop}")
        print(f"Intercept (n0, ≤10µm): {n0_10um:.2e}, E-folding Diameter (D, ≤10µm): {D_10um:.2f} μm")

    except RuntimeError:
        print(f"Fit could not be performed for {selected_date} (≤10 µm).")

else:
    print(f"Leg not found or missing fitted parameters for {selected_date} with start {selected_start} and stop {selected_stop}.")
#%%
# Define the specific leg to plot
selected_date = "2022-06-10"
selected_start = 51245.0
selected_stop = 51433.0

# Find the corresponding ambient size distribution entry in Y_BCB_calc
selected_leg = next(
    (entry for entry in Y_BCB_calc 
     if entry['Date'] == selected_date and entry['BCB_start'] == selected_start and entry['BCB_stop'] == selected_stop),
    None
)

# Find the fitted parameters for this leg in ambient_fits
selected_fit = next(
    (fit for fit in ambient_fits 
     if fit['Date'] == selected_date and fit['BCB_start'] == selected_start and fit['BCB_stop'] == selected_stop),
    None
)

# Ensure both the raw data and the fit parameters exist
if selected_leg is not None and selected_fit is not None:
    # Extract fitted parameters
    n0_full, D_full = selected_fit['Intercept_n0'], selected_fit['E_folding_D']

    # Extract bin means and remove NaNs
    bin_means = np.array([selected_leg.get(f'Bin{i}_Y_mean', np.nan) for i in range(12, 30)], dtype=float)
    valid_indices = ~np.isnan(bin_means)
    bin_centers_valid = np.array(bin_center)[valid_indices]
    bin_means_valid = bin_means[valid_indices]

    # **Filter data to only include points where bin center ≤ 10 µm**
    mask_10um = bin_centers_valid <= 10
    bin_centers_10um = bin_centers_valid[mask_10um]
    bin_means_10um = bin_means_valid[mask_10um]

    # Fit exponential function only on this subset
    try:
        popt, _ = curve_fit(exponential, bin_centers_10um, bin_means_10um, p0=(1, 1), maxfev=5000)
        n0_10um, D_10um = popt

        # Generate exponential fit lines
        x_fit_full = np.linspace(min(bin_centers_valid), max(bin_centers_valid), 100)
        y_fit_full = exponential(x_fit_full, n0_full, D_full)

        x_fit_10um = np.linspace(min(bin_centers_10um), 10, 100)
        y_fit_10um = exponential(x_fit_10um, n0_10um, D_10um)

        # Create figure
        plt.figure(figsize=(8, 6))

        # Plot observed data (full range)
        plt.plot(bin_centers_valid, bin_means_valid, color='blue', marker='o', linestyle='-', alpha=0.5, label="Observed Data (Full Range)")

        # Plot the original full-range exponential fit
        plt.plot(x_fit_full, y_fit_full, color='red', linestyle='--', label=f"Full Fit: y = {n0_full:.2e} * exp(-x / {D_full:.2f})")

        # Plot the observed data ≤ 10 µm in bold
        plt.plot(bin_centers_10um, bin_means_10um, color='blue', marker='o', linestyle='-', label="Observed Data (≤ 10 µm)")

        # Plot the new exponential fit only up to 10 µm
        plt.plot(x_fit_10um, y_fit_10um, color='green', linestyle='--', label=f"Fit (≤10 µm): y = {n0_10um:.2e} * exp(-x / {D_10um:.2f})")

        # Labels and title
        plt.xlabel("Deliquesced Diameter (μm)", fontsize=14, fontweight="bold")
        plt.ylabel("Number Concentration (/cm³/μm)", fontsize=14, fontweight="bold")
        plt.title(f"Ambient Size Distribution - {selected_date}\nStart: {selected_start} | Stop: {selected_stop}", 
                  fontsize=14, fontweight="bold")

        # Formatting
        plt.legend()
        plt.yscale('log')  
        plt.ylim(10**-33, 10**1)
        plt.xticks(fontweight="bold", fontsize=14)
        plt.yticks(fontweight="bold", fontsize=14)  
        # Show the plot
        plt.show()

        # Print selected leg details
        print(f"Selected Date: {selected_date}")
        print(f"Start Time: {selected_start}, Stop Time: {selected_stop}")
        print(f"Full Fit - Intercept (n0): {n0_full:.2e}, E-folding Diameter (D): {D_full:.2f} μm")
        print(f"≤10 µm Fit - Intercept (n0): {n0_10um:.2e}, E-folding Diameter (D): {D_10um:.2f} μm")

    except RuntimeError:
        print(f"Fit could not be performed for {selected_date} (≤10 µm).")

else:
    print(f"Leg not found or missing fitted parameters for {selected_date} with start {selected_start} and stop {selected_stop}.")
#%%
# **Filter data to only include points where bin center ≤ 10 µm**
# **Filter data to only include points where bin center ≤ 10 µm**
mask_10um = bin_centers_valid <= 10
bin_centers_10um = bin_centers_valid[mask_10um]
bin_means_10um = bin_means_valid[mask_10um]

# Compute Poisson standard deviation (sigma = sqrt(dn/dd))
sigma_10um = np.sqrt(bin_means_10um)

# Handling zero-count bins correctly (very small nonzero weight)
sigma_10um[sigma_10um == 0] = np.min(sigma_10um[sigma_10um > 0]) * 0.1  # Assign a small weight

# Ensure there are enough points to fit
if len(bin_centers_10um) > 2:
    try:
        # Fit with correct Poisson weighting (sigma = sqrt(dn/dd))
        popt_10um, _ = curve_fit(exponential, bin_centers_10um, bin_means_10um, 
                                 p0=(max(bin_means_10um), 3), 
                                 sigma=sigma_10um, absolute_sigma=True, 
                                 maxfev=10000)

        # Extract the new fit parameters
        n0_10um, D_10um = popt_10um

        # Generate the new fit line **only below 10 µm**
        x_fit_10um = np.linspace(min(bin_centers_10um), 10, 50)
        y_fit_10um = exponential(x_fit_10um, n0_10um, D_10um)

        # Print values to compare
        print(f"Weighted Fit (≤10 µm) - Intercept (n0): {n0_10um:.2e}, E-folding Diameter (D): {D_10um:.2f} μm")

    except RuntimeError:
        print("Curve fitting failed for ≤10 µm data.")
else:
    print("Not enough data points to fit ≤10 µm.")

#%%
# Generate the original full-range exponential fit curve
# Generate the truncated full-range exponential fit curve up to 10 µm
x_fit_full = np.linspace(min(bin_centers_valid), 10, 50)  # NEW: Stops at 10 µm
y_fit_full = exponential(x_fit_full, n0_full, D_full)

# Generate the new weighted fit curve **only up to 10 µm**
x_fit_10um = np.linspace(min(bin_centers_10um), 10, 50)
y_fit_10um = exponential(x_fit_10um, n0_10um, D_10um)

# Create the plot
plt.figure(figsize=(8, 6))

# Plot observed data (full range)
plt.plot(bin_centers_valid, bin_means_valid, color='blue', marker='o', linestyle='-', alpha=0.5, label="Observed Data (Full Range)")

# Plot observed data ≤ 10 µm in bold
plt.plot(bin_centers_10um, bin_means_10um, color='blue', marker='o', linestyle='-', label="Observed Data (≤ 10 µm)")

# Plot the original full-range exponential fit
plt.plot(x_fit_full, y_fit_full, color='red', linestyle='--', label=f"Fit (≤10 µm): y = {n0_full:.2e} * exp(-x / {D_full:.2f})")

# Plot the new **weighted** fit only up to 10 µm
plt.plot(x_fit_10um, y_fit_10um, color='green', linestyle='--', label=f"Weighted Fit (≤10 µm): y = {n0_10um:.2e} * exp(-x / {D_10um:.2f})")

# Labels and title
plt.xlabel("Deliquesced Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.title(f"Ambient Size Distribution - {selected_date}\nStart: {selected_start} | Stop: {selected_stop}", 
          fontsize=14, fontweight="bold")

# Formatting
plt.legend()
plt.yscale('log')  
plt.ylim(10**-33, 10**1)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)  
# Show the plot
plt.show()



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
#Ambient histogram 
ambient_intercept_values = [
    fit['Intercept_n0'] for fit in ambient_fits if not np.isnan(fit['Intercept_n0'])
]

# Plot histogram of ambient intercepts
plt.figure(figsize=(8, 6))
plt.hist(ambient_intercept_values, bins=20, edgecolor='black', alpha=0.7)

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
# Extract data for scatter plot
ambient_n0_values = [fit['Intercept_n0'] for fit in ambient_fits if not np.isnan(fit['Intercept_n0'])]
dry_intercept_values = [leg['dry intercept'] for leg in filtered_master_BCB_dryintercept if not np.isnan(leg['dry intercept'])]

# Ensure matching lengths before plotting scatter
if len(ambient_n0_values) == len(dry_intercept_values):
    plt.figure(figsize=(8, 6))
    plt.scatter(ambient_n0_values, dry_intercept_values, alpha=0.5, edgecolor='black')
    plt.xlabel(r'Ambient Intercept $N_0$ (cm$^{-3} \mu$m$^{-1}$)', fontsize=14)
    plt.ylabel(r'Dry Intercept (cm$^{-3} \mu$m$^{-1}$)', fontsize=14)
    plt.title('Scatter Plot: Ambient vs. Dry Intercept', fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
else:
    print("Mismatch in data lengths between ambient intercepts and dry intercepts!")

# Overlaid Histogram of Ambient and Dry Intercepts
plt.figure(figsize=(8, 6))
plt.hist(ambient_n0_values, bins=20, alpha=0.5, label="Ambient Intercept (N0)", edgecolor="black")
plt.hist(dry_intercept_values, bins=20, alpha=0.5, label="Dry Intercept (N0/gRH)", edgecolor="black")

# Labels and Formatting
plt.xlabel(r"Intercept Value (cm$^{-3} \mu$m$^{-1}$)", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.title("Comparison of Ambient vs. Dry Intercept Distributions", fontsize=16, fontweight='bold')
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
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
common_bins = np.linspace(2, 40, 35)

plt.figure(figsize=(8, 6))

# Plot both ambient and dry size distributions for comparison
for entry_ambient, entry_dry in zip(Y_BCB_calc, filtered_master_BCB_ddry):
    # Extract ambient data
    ambient_dd = np.array(bin_center)
    ambient_dN_dD = np.array([entry_ambient.get(f'Bin{i}_Y_mean', np.nan) for i in range(12, 30)])

    # Extract dry data
    dry_dd = np.array(entry_dry['ddry'])
    dry_dN_dD = np.array(entry_dry['dN/dDdry'])

    # Remove NaN values before interpolation
    valid_ambient = ~np.isnan(ambient_dd) & ~np.isnan(ambient_dN_dD)
    valid_dry = ~np.isnan(dry_dd) & ~np.isnan(dry_dN_dD)

    if np.sum(valid_ambient) < 2 or np.sum(valid_dry) < 2:
        continue  # Skip if not enough valid points

    # Interpolate onto common bins
    interp_ambient = interp1d(ambient_dd[valid_ambient], ambient_dN_dD[valid_ambient], 
                              kind='linear', bounds_error=False, fill_value=np.nan)
    interp_dry = interp1d(dry_dd[valid_dry], dry_dN_dD[valid_dry], 
                          kind='linear', bounds_error=False, fill_value=np.nan)

    interpolated_ambient = interp_ambient(common_bins)
    interpolated_dry = interp_dry(common_bins)

    # Plot ambient in blue, dry in red
    plt.plot(common_bins, interpolated_ambient, color='blue', alpha=0.3, label="Ambient" if 'Ambient' not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.plot(common_bins, interpolated_dry, color='red', alpha=0.3, label="Dry" if 'Dry' not in plt.gca().get_legend_handles_labels()[1] else "")

# Formatting
plt.xlabel("Bin Centers Diameter (μm)", fontsize=12, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=12, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-7, 10**1)
plt.xlim(0,20)
plt.xticks(fontweight="bold", fontsize=10)
plt.yticks(fontweight="bold", fontsize=10)
plt.title("Below Cloud Base January - June 2022\n Raw Size Distributions", fontsize=14, fontweight="bold")
plt.legend()
plt.show()
#%%
#Both filtered 0


# Define common bins for interpolation
common_bins = np.linspace(2, 40, 35)

plt.figure(figsize=(8, 6))

# Plot both ambient and dry size distributions for comparison
for entry_ambient, entry_dry in zip(Y_BCB_calc, filtered_master_BCB_ddry):
    # Extract ambient data
    ambient_dd = np.array(bin_center)
    ambient_dN_dD = np.array([entry_ambient.get(f'Bin{i}_Y_mean', np.nan) for i in range(12, 30)])

    # Extract dry data
    dry_dd = np.array(entry_dry['ddry'])
    dry_dN_dD = np.array(entry_dry['dN/dDdry'])

    # Remove NaN values before interpolation
    valid_ambient = ~np.isnan(ambient_dd) & ~np.isnan(ambient_dN_dD)
    valid_dry = ~np.isnan(dry_dd) & ~np.isnan(dry_dN_dD)

    if np.sum(valid_ambient) < 2 or np.sum(valid_dry) < 2:
        continue  # Skip if not enough valid points

    # Interpolate onto common bins
    interp_ambient = interp1d(ambient_dd[valid_ambient], ambient_dN_dD[valid_ambient], 
                              kind='linear', bounds_error=False, fill_value=np.nan)
    interp_dry = interp1d(dry_dd[valid_dry], dry_dN_dD[valid_dry], 
                          kind='linear', bounds_error=False, fill_value=np.nan)

    interpolated_ambient = interp_ambient(common_bins)
    interpolated_dry = interp_dry(common_bins)

    # Filter out zero and NaN values
    valid_ambient_bins = (interpolated_ambient > 0) & ~np.isnan(interpolated_ambient)
    valid_dry_bins = (interpolated_dry > 0) & ~np.isnan(interpolated_dry)

    # Apply filters to remove zero values before plotting
    filtered_ambient_bins = common_bins[valid_ambient_bins]
    filtered_ambient_values = interpolated_ambient[valid_ambient_bins]

    filtered_dry_bins = common_bins[valid_dry_bins]
    filtered_dry_values = interpolated_dry[valid_dry_bins]

    # Plot ambient in blue, dry in red
    if len(filtered_ambient_bins) > 0:
        plt.plot(filtered_ambient_bins, filtered_ambient_values, color='blue', alpha=0.3, label="Ambient" if 'Ambient' not in plt.gca().get_legend_handles_labels()[1] else "")

    if len(filtered_dry_bins) > 0:
        plt.plot(filtered_dry_bins, filtered_dry_values, color='red', alpha=0.3, label="Dry" if 'Dry' not in plt.gca().get_legend_handles_labels()[1] else "")

# Formatting
plt.xlabel("Bin Centers Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-7, 10**1)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base January - June 2022\n Raw Size Distributions", fontsize=14, fontweight="bold")
import matplotlib.lines as mlines

# Create custom legend handles with thicker lines
ambient_legend = mlines.Line2D([], [], color='blue', linewidth=6, label="Ambient")  # Thicker blue line
dry_legend = mlines.Line2D([], [], color='red', linewidth=6, label="Dry")  # Thicker red line

# Apply the custom legend with thicker lines
plt.legend(handles=[ambient_legend, dry_legend], fontsize=12, frameon=True)

plt.legend()
plt.show()
#%%
common_bins = np.linspace(2, 25, 25)
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
#histogram of slopes for regular dry exponential fits 
dry_normal_exp=[fit['Dry_E_folding_D'] for fit in dry_exponential_fits if not np.isnan(fit['Dry_E_folding_D'])] 
plt.figure(figsize=(8, 6))
plt.hist(dry_normal_exp, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel(r"Slope", fontsize=14, fontweight="bold")
plt.ylabel('Frequency', fontsize=14, fontweight="bold")
plt.title('Dry Exponential fit', fontsize=16, fontweight="bold")
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
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
#%%

#%%
#histogram comapring less than 10um and regular fit exponential 
# Extracting slopes for both fits
dry_slopes_10 = [fit['Dry_E_folding_D'] for fit in dry_exponential_fits_10 if not np.isnan(fit['Dry_E_folding_D'])] 
# Plotting histograms
# bins = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 
#                  16384, 32768, 65536, 131072])
plt.figure(figsize=(10, 6))
plt.hist(dry_slopes_10, alpha=0.5,  bins=20, label='≤ 10 µm', color='blue', edgecolor='black')
plt.hist(dry_normal_exp, alpha=0.5, bins=20, label='Regular Fit', color='red', edgecolor='black')
plt.xlabel('Slope (D)', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')
plt.title('Histogram of Dry E-folding D Values', fontsize=16, fontweight='bold')
plt.legend()
# plt.xscale('log')
plt.grid(True)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.show()
#%%
#Histogram together 
bins = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 
                 16384, 32768, 65536, 131072]) 
plt.figure(figsize=(10, 6))
plt.hist(dry_slopes_10, alpha=0.5,  bins=bins, label='≤ 10 µm', color='blue', edgecolor='black')
plt.hist(dry_normal_exp, alpha=0.5, bins=bins, label='Regular Fit', color='red', edgecolor='black')
plt.xlabel('Slope (D)', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')
plt.title('Dry', fontsize=16, fontweight='bold')
plt.legend()
plt.xscale('log')
plt.grid(True)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.show()
#%%
#weighted poisson fits for dry 


# Define exponential function
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

# Lists to store fit results
dry_fits_regular = []
dry_fits_poisson = []

# Create figure for plotting
plt.figure(figsize=(8, 6))

# Process each entry in filtered_master_BCB_ddry
for entry in filtered_master_BCB_ddry:
    ddry_values = np.array(entry['ddry'])
    dN_dD_dry = np.array(entry['dN/dDdry'])

    # Ensure valid data points
    valid_indices = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry) & (dN_dD_dry > 0)
    
    if np.sum(valid_indices) < 2:  
        continue

    # **Regular Exponential Fit (No Poisson Weighting)**
    try:
        popt_reg, _ = curve_fit(exponential, ddry_values[valid_indices], dN_dD_dry[valid_indices], 
                                p0=(1, 5), maxfev=100)
        n0_reg, D_reg = popt_reg

        # **Filter out extreme slopes where D > 20 µm**
        if D_reg > 20:
            continue  

        # Store regular fit parameters
        dry_fits_regular.append({
            'Date': entry['Date'],
            'BCB_start': entry['BCB_start'],
            'BCB_stop': entry['BCB_stop'],
            'Intercept_n0': n0_reg,
            'E_folding_D': D_reg
        })

        # Generate fitted curve
        x_fit = np.linspace(min(ddry_values[valid_indices]), max(ddry_values[valid_indices]), 100)
        y_fit_reg = exponential(x_fit, *popt_reg)
        y_fit[y_fit < 1e-8] = np.nan 
        # Plot the regular fit in **blue**
        plt.plot(x_fit, y_fit_reg, color='black', alpha=0.1)

    except RuntimeError:
        print(f"Regular fit could not be performed for date {entry['Date']}")

    # **Poisson-Weighted Exponential Fit**
    try:
        # Compute bin widths for Poisson weighting
        bin_widths = np.diff(ddry_values, prepend=ddry_values[0])  # Approximate bin widths
        sigma = np.sqrt(dN_dD_dry[valid_indices]) / bin_widths[valid_indices]  # Poisson error

        popt_pois, _ = curve_fit(exponential, ddry_values[valid_indices], dN_dD_dry[valid_indices], 
                                 p0=(1, 5), sigma=sigma, absolute_sigma=True, maxfev=5000)
        n0_pois, D_pois = popt_pois

        # **Filter out extreme slopes where D > 20 µm**
        if D_pois > 20:
            continue  

        # Store Poisson-weighted fit parameters
        dry_fits_poisson.append({
            'Date': entry['Date'],
            'BCB_start': entry['BCB_start'],
            'BCB_stop': entry['BCB_stop'],
            'Intercept_n0': n0_pois,
            'E_folding_D': D_pois
        })

        # Generate fitted curve
        y_fit_pois = exponential(x_fit, *popt_pois)
        y_fit_pois[y_fit_pois < 1e-8] = np.nan 

        # Plot the Poisson-weighted fit in **green**
        plt.plot(x_fit, y_fit_pois, color='red', alpha=0.1)

    except RuntimeError:
        print(f"Poisson-weighted fit could not be performed for date {entry['Date']}")

# Formatting and labels
plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-33, 10**1)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base January - June 2022\nExponential vs Weighted Fits", 
          fontsize=14, fontweight="bold")
dry_fit_patch = mpatches.Patch(color='black', label='Regular Fit')  # Darker black
dry_poisson_fit_patch = mpatches.Patch(color='red', label='Poisson-Weighted Fit')  # Defined red

# Add the custom legend with bold text
plt.legend(handles=[dry_fit_patch, dry_poisson_fit_patch], loc='lower left', fontsize=12, frameon=True)
plt.show()

# Print the number of successful fits
print(f"Total successful regular fits (D ≤ 20 µm): {len(dry_fits_regular)}")
print(f"Total successful Poisson-weighted fits (D ≤ 20 µm): {len(dry_fits_poisson)}")
#%%
#histogram of poisson and regular exponential 
# Extracting slopes for both fits
bins=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
                  16384, 32768, 65536, 131072]
dry_slopes_poisson = [fit['E_folding_D'] for fit in dry_fits_poisson if not np.isnan(fit['E_folding_D'])]
# Plotting histograms
plt.figure(figsize=(10, 6))
plt.hist(dry_slopes_poisson, alpha=0.5, bins=bins, label='Poisson-Weighted Fit', color='green', edgecolor='black')
plt.hist(dry_normal_exp, alpha=0.5, bins=bins, label='Regular Fit', color='red', edgecolor='black')
plt.hist(dry_slopes_10, alpha=0.5, bins=bins, label='≤ 10 µm Fit', color='blue', edgecolor='black') 
plt.xlabel('Slope (D)', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')
plt.xscale('log')
plt.title('Dry Fits', fontsize=16, fontweight='bold')
plt.legend()
#%%
# Define common bin edges
min_value = min(min(dry_slopes_poisson), min(dry_normal_exp))
max_value = max(max(dry_slopes_poisson), max(dry_normal_exp))

bins = np.linspace(min_value, max_value, 21)  # 21 edges for 20 bins

# Plot histograms with the same bins
plt.figure(figsize=(10, 6))
plt.hist(dry_slopes_poisson, bins=bins, alpha=0.3, label='Poisson-Weighted Fit', color='green', edgecolor='black')
plt.hist(dry_normal_exp, bins=bins, alpha=0.2, label='Regular Fit', color='red', edgecolor='black')
plt.hist(dry_slopes_10, bins=bins, alpha=0.4, label='≤ 10 µm Fit', color='orange', edgecolor='black')
# Formatting
plt.xlabel('Slope (D)', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')
plt.title('Dry Size Distributions', fontsize=16, fontweight='bold')
plt.legend()
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.show()

#%%


# Define common bin edges
min_value = min(min(dry_slopes_poisson), min(dry_normal_exp), min(dry_slopes_10))
max_value = max(max(dry_slopes_poisson), max(dry_normal_exp), max(dry_slopes_10))
bins = np.linspace(min_value, max_value, 21)  # 21 edges for 20 bins

# Plot histograms with better visibility
plt.figure(figsize=(10, 6))

plt.hist(dry_slopes_poisson, bins=bins, alpha=0.6, label='Poisson-Weighted Fit', 
         color='blue', edgecolor='black', linewidth=1.2, hatch='.')

plt.hist(dry_normal_exp, bins=bins, alpha=0.5, label='Regular Fit', 
         color='red', edgecolor='black', linewidth=1.2)

plt.hist(dry_slopes_10, bins=bins, alpha=0.4, label='≤ 10 µm Fit', 
         color='gold', edgecolor='black', linewidth=1.2, hatch='x')

# Formatting
plt.xlabel('Slope (D)', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')
plt.title('Dry Size Distributions', fontsize=16, fontweight='bold')
plt.legend()
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')

plt.show()
#%%

# ✅ Define common bin centers for interpolation
common_bins = np.linspace(2, 40, 35)

# ✅ Define the specific case to plot
selected_date = "2022-06-10"
selected_start = 51245.0
selected_stop = 51433.0

# ✅ Find the corresponding dry size distribution
selected_dry_leg = next(
    (entry for entry in filtered_master_BCB_ddry if entry['Date'] == selected_date and 
     entry['BCB_start'] == selected_start and entry['BCB_stop'] == selected_stop), 
    None
)

# ✅ Find the corresponding ambient size distribution
selected_ambient_leg = next(
    (entry for entry in Y_BCB_calc 
     if entry['Date'] == selected_date and entry['BCB_start'] == selected_start and entry['BCB_stop'] == selected_stop),
    None
)

# ✅ Ensure both dry and ambient data exist
if selected_dry_leg and selected_ambient_leg:
    plt.figure(figsize=(8, 6))

    # ✅ Extract ambient data
    ambient_dd = np.array(bin_center)
    ambient_dN_dD = np.array([selected_ambient_leg.get(f'Bin{i}_Y_mean', np.nan) for i in range(12, 30)])

    # ✅ Extract dry data
    dry_dd = np.array(selected_dry_leg['ddry'])
    dry_dN_dD = np.array(selected_dry_leg['dN/dDdry'])

    # ✅ Remove NaN values before interpolation
    valid_ambient = ~np.isnan(ambient_dd) & ~np.isnan(ambient_dN_dD)
    valid_dry = ~np.isnan(dry_dd) & ~np.isnan(dry_dN_dD)

    if np.sum(valid_ambient) >= 2 and np.sum(valid_dry) >= 2:
        # ✅ Interpolate onto common bins
        interp_ambient = interp1d(ambient_dd[valid_ambient], ambient_dN_dD[valid_ambient], 
                                  kind='linear', bounds_error=False, fill_value=np.nan)
        interp_dry = interp1d(dry_dd[valid_dry], dry_dN_dD[valid_dry], 
                              kind='linear', bounds_error=False, fill_value=np.nan)

        interpolated_ambient = interp_ambient(common_bins)
        interpolated_dry = interp_dry(common_bins)

        # ✅ Plot ambient in **darker blue**, dry in **darker red**
        plt.plot(common_bins, interpolated_ambient, color='darkblue', alpha=0.9, linewidth=2, label="Ambient")
        plt.plot(common_bins, interpolated_dry, color='darkred', alpha=0.9, linewidth=2, label="Dry")

    # ✅ Step 3: Formatting and Labels
    plt.xlabel("Bin Centers Diameter (μm)", fontsize=15, fontweight="bold")
    plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=15, fontweight="bold")
    plt.yscale("log")
    plt.ylim(10**-7, 10**1)
    plt.xlim(0, 20)
    plt.xticks(fontweight="bold", fontsize=14)
    plt.yticks(fontweight="bold", fontsize=14)
    plt.title(f"Dry vs Ambient Size Distributions - {selected_date}\nStart: {selected_start} | Stop: {selected_stop}", fontsize=14, fontweight="bold")

    # ✅ Step 4: Make **legend lines thicker and darker**
    ambient_legend = mlines.Line2D([], [], color='darkblue', linewidth=4, label="Ambient (dN/dD)")  # Thick dark blue
    dry_legend = mlines.Line2D([], [], color='darkred', linewidth=4, label="Dry (dN/dDdry)")  # Thick dark red
    plt.legend(handles=[ambient_legend, dry_legend], fontsize=12, frameon=True)
    plt.show()

    print(f"Selected Date: {selected_date}")
    print(f"Start Time: {selected_start}, Stop Time: {selected_stop}")

else:
    print(f"Leg not found or missing data for {selected_date}")

#%%
#finding a good leg for this 
# ✅ Store valid legs
valid_legs = []

# ✅ Loop through each dry size distribution
for entry in filtered_master_BCB_ddry:
    ddry_values = np.array(entry['ddry'])  # Dry bin centers
    dN_dD_dry = np.array(entry['dN/dDdry'])  # Concentration values

    # ✅ Remove NaN values
    valid_indices = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry) & (dN_dD_dry > 0)

    if np.sum(valid_indices) > 0:  # Ensure there's valid data
        max_ddry = np.max(ddry_values[valid_indices])  # Find the largest valid bin center
        
        # ✅ If the max bin center is at least 15 µm, store the leg
        if max_ddry >= 15:
            valid_legs.append({
                'Date': entry['Date'],
                'BCB_start': entry['BCB_start'],
                'BCB_stop': entry['BCB_stop'],
                'Max_ddry': max_ddry
            })

# ✅ Sort by `Max_ddry` (descending) to find the best option
valid_legs = sorted(valid_legs, key=lambda x: x['Max_ddry'], reverse=True)

# ✅ Print the best options
if valid_legs:
    print(f"Found {len(valid_legs)} dry legs with data extending beyond 15 µm:\n")
    for leg in valid_legs[:5]:  # Show top 5 options
        print(f"Date: {leg['Date']}, Start: {leg['BCB_start']}, Stop: {leg['BCB_stop']}, Max_ddry: {leg['Max_ddry']:.2f} µm")
else:
    print("No suitable dry size distributions found that extend to at least 15 µm.")

#%%
#Plotting Just dry leg 


# ✅ Define common bin centers for interpolation
common_bins = np.linspace(2, 30, 25)

# ✅ Define the specific case to plot
selected_date = "2022-03-13"
selected_start = 50135.0
selected_stop = 50496.0

# ✅ Find the corresponding dry size distribution
selected_dry_leg = next(
    (entry for entry in filtered_master_BCB_ddry if entry['Date'] == selected_date and 
     entry['BCB_start'] == selected_start and entry['BCB_stop'] == selected_stop), 
    None
)

# ✅ Ensure dry data exists
if selected_dry_leg:
    plt.figure(figsize=(8, 6))

    # ✅ Extract dry data
    dry_dd = np.array(selected_dry_leg['ddry'])
    dry_dN_dD = np.array(selected_dry_leg['dN/dDdry'])

    # ✅ Remove NaN values before interpolation
    valid_dry = ~np.isnan(dry_dd) & ~np.isnan(dry_dN_dD)

    if np.sum(valid_dry) >= 2:
        # ✅ Interpolate onto common bins
        interp_dry = interp1d(dry_dd[valid_dry], dry_dN_dD[valid_dry], 
                              kind='linear', bounds_error=False, fill_value=np.nan)

        interpolated_dry = interp_dry(common_bins)

        # ✅ Plot the Dry Size Distribution (Dark Red Line)
        plt.plot(common_bins, interpolated_dry, color='black', alpha=0.9, linewidth=2, label="Dry Observed Data")

    # ✅ Step 3: Formatting and Labels
    plt.xlabel("Bin Centers Diameter (μm)", fontsize=15, fontweight="bold")
    plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=15, fontweight="bold")
    plt.yscale("log")
    plt.ylim(10**-7, 10**1)
    plt.xlim(0, 30)
    plt.xticks(fontweight="bold", fontsize=14)
    plt.yticks(fontweight="bold", fontsize=14)
    plt.title(f"Dry Size Distribution - {selected_date}\nStart: {selected_start} | Stop: {selected_stop}", fontsize=14, fontweight="bold")

    # ✅ Step 4: Add a Legend
    
    plt.show()

    print(f"Selected Date: {selected_date}")
    print(f"Start Time: {selected_start}, Stop Time: {selected_stop}")

else:
    print(f"Dry leg not found for {selected_date}")



#%%
#Fit the exponential to it 


# ✅ Define the exponential function
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

# ✅ Define the specific case to fit
selected_date = "2022-03-13"
selected_start = 50135.0
selected_stop = 50496.0

# ✅ Find the corresponding dry size distribution
selected_dry_leg = next(
    (entry for entry in filtered_master_BCB_ddry if entry['Date'] == selected_date and 
     entry['BCB_start'] == selected_start and entry['BCB_stop'] == selected_stop), 
    None
)

# ✅ Ensure dry data exists
if selected_dry_leg:
    plt.figure(figsize=(8, 6))

    # ✅ Extract dry data
    ddry_values = np.array(selected_dry_leg['ddry'])
    dN_dD_dry = np.array(selected_dry_leg['dN/dDdry'])

    # ✅ Remove NaN values
    valid_indices = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry)
    
    if np.sum(valid_indices) >= 5:  # Ensure sufficient data points for fitting
        try:
            # ✅ Fit the exponential function
            popt, _ = curve_fit(exponential, ddry_values[valid_indices], dN_dD_dry[valid_indices], 
                                p0=(1, 5), maxfev=100)
            n0_fit, D_fit = popt

            # ✅ Generate fit curve
            x_fit = np.linspace(min(ddry_values[valid_indices]), max(ddry_values[valid_indices]), 100)
            y_fit = exponential(x_fit, *popt)

            # ✅ Plot the Dry Observed Data (Red)
            plt.plot(ddry_values[valid_indices], dN_dD_dry[valid_indices], 
                     color='darkred', marker='o', linestyle='-', label="Dry distribution")

            # ✅ Plot the Fitted Curve (Black Dashed Line)
            plt.plot(x_fit, y_fit, color='black', linestyle='--', linewidth=2, label=f"Exponential Fit")

            # ✅ Step 3: Formatting and Labels
            plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=14, fontweight="bold")
            plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
            plt.yscale("log")
            plt.ylim(1e-7, 1e1)
            plt.xlim(0, 30)
            plt.xticks(fontweight="bold", fontsize=14)
            plt.yticks(fontweight="bold", fontsize=14)
            plt.title(f"Fitted Dry Size Distribution - {selected_date}\nStart: {selected_start} | Stop: {selected_stop}", fontsize=14, fontweight="bold")
            plt.legend()
            # ✅ Step 4: Add a Legend
            plt.show()

            # ✅ Print Fit Parameters
            print(f"Selected Date: {selected_date}")
            print(f"Start Time: {selected_start}, Stop Time: {selected_stop}")
            print(f"Dry Intercept (n0): {n0_fit:.2e}")
            print(f"Dry E-folding Diameter (D): {D_fit:.2f} μm")

        except RuntimeError:
            print(f"Fit could not be performed for {selected_date}")

else:
    print(f"Dry leg not found for {selected_date}")
#%%


# ✅ Define the exponential function
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

# ✅ Define the specific case to fit
selected_date = "2022-03-13"
selected_start = 50135.0
selected_stop = 50496.0

# ✅ Find the corresponding dry size distribution
selected_dry_leg = next(
    (entry for entry in filtered_master_BCB_ddry if entry['Date'] == selected_date and 
     entry['BCB_start'] == selected_start and entry['BCB_stop'] == selected_stop), 
    None
)

# ✅ Ensure dry data exists
if selected_dry_leg:
    plt.figure(figsize=(8, 6))

    # ✅ Extract dry data
    ddry_values = np.array(selected_dry_leg['ddry'])
    dN_dD_dry = np.array(selected_dry_leg['dN/dDdry'])

    # ✅ Remove NaN values
    valid_indices = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry) & (dN_dD_dry > 0)
    
    if np.sum(valid_indices) >= 5:  # Ensure sufficient data points for fitting
        try:
            ### ✅ FULL FIT (ONLY USING BINS > 10 µm)
            mask_full = (ddry_values > 10) & valid_indices  # Only bins greater than 10 µm
            ddry_full = ddry_values[mask_full]
            dN_dD_dry_full = dN_dD_dry[mask_full]

            if len(ddry_full) >= 5:
                popt_full, _ = curve_fit(exponential, ddry_full, dN_dD_dry_full, 
                                         p0=(max(dN_dD_dry_full), 10), maxfev=5000)  # Larger initial guess for `D`
                n0_full, D_full = popt_full

                # ✅ Generate full-range fit curve
                x_fit_full = np.linspace(min(ddry_full), max(ddry_full), 100)
                y_fit_full = exponential(x_fit_full, *popt_full)

            ### ✅ ≤10 µm FIT (ONLY USING BINS ≤ 10 µm)
            mask_10um = (ddry_values <= 10) & valid_indices  # Only bins ≤ 10 µm
            ddry_10um = ddry_values[mask_10um]
            dN_dD_dry_10um = dN_dD_dry[mask_10um]

            if len(ddry_10um) >= 5:
                popt_10um, _ = curve_fit(exponential, ddry_10um, dN_dD_dry_10um, 
                                         p0=(max(dN_dD_dry_10um), 3), maxfev=5000)  # Smaller initial guess for `D`
                n0_10um, D_10um = popt_10um

                # ✅ Generate fit curve for ≤10 µm **(independent fit!)**
                x_fit_10um = np.linspace(min(ddry_10um), max(ddry_10um), 50)
                y_fit_10um = exponential(x_fit_10um, *popt_10um)

                # ✅ Plot the Dry Observed Data (Red)
                plt.plot(ddry_values[valid_indices], dN_dD_dry[valid_indices], 
                         color='darkred', marker='o', linestyle='-', label="Dry Distribution")

                # ✅ Plot the Full-Range Fit (Black Dashed Line) **ensuring larger `D`**
                if len(ddry_full) >= 5:
                    plt.plot(x_fit_full, y_fit_full, color='black', linestyle='--', linewidth=2, 
                             label=f"Full Fit (>10 µm): y = {n0_full:.2e} * exp(-x / {D_full:.2f})")

                # ✅ Plot the **Independent** ≤10 µm Fit (Blue Dotted Line)
                plt.plot(x_fit_10um, y_fit_10um, color='blue', linestyle=':', linewidth=2, 
                         label=f"≤10 µm Fit: y = {n0_10um:.2e} * exp(-x / {D_10um:.2f})")

                # ✅ Formatting and Labels
                plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=14, fontweight="bold")
                plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
                plt.yscale("log")
                plt.ylim(1e-7, 1e1)
                plt.xlim(0, 30)
                plt.xticks(fontweight="bold", fontsize=14)
                plt.yticks(fontweight="bold", fontsize=14)
                plt.title(f"Dry Size Distribution - {selected_date}\nStart: {selected_start} | Stop: {selected_stop}", fontsize=14, fontweight="bold")

                # ✅ Add a Legend
                plt.legend()
                plt.grid(True, which="both", linestyle="--", linewidth=0.5)
                plt.show()

                # ✅ Print Fit Parameters
                print(f"Selected Date: {selected_date}")
                print(f"Start Time: {selected_start}, Stop Time: {selected_stop}")
                
                if len(ddry_full) >= 5:
                    print(f"\n🔹 Full-Range Fit Parameters (Using Only Bins > 10 µm):")
                    print(f"Dry Intercept (n0): {n0_full:.2e}")
                    print(f"Dry E-folding Diameter (D): {D_full:.2f} μm")
                
                print(f"\n🔹 ≤10 µm Independent Fit Parameters:")
                print(f"Dry Intercept (n0): {n0_10um:.2e}")
                print(f"Dry E-folding Diameter (D): {D_10um:.2f} μm")

            else:
                print("Not enough valid points to fit ≤10 µm data.")

        except RuntimeError:
            print(f"Fit could not be performed for {selected_date}")

else:
    print(f"Dry leg not found for {selected_date}")
#%%
common_bins=(2, 30, 30)
# ✅ Define the exponential function
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

# ✅ Define the specific case to fit
selected_date = "2022-03-13"
selected_start = 50135.0
selected_stop = 50496.0

# ✅ Find the corresponding dry size distribution
selected_dry_leg = next(
    (entry for entry in filtered_master_BCB_ddry if entry['Date'] == selected_date and 
     entry['BCB_start'] == selected_start and entry['BCB_stop'] == selected_stop), 
    None
)

# ✅ Ensure dry data exists
if selected_dry_leg:
    plt.figure(figsize=(8, 6))

    # ✅ Extract dry data
    ddry_values = np.array(selected_dry_leg['ddry'])
    dN_dD_dry = np.array(selected_dry_leg['dN/dDdry'])

    # ✅ Remove NaN values
    valid_indices = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry) & (dN_dD_dry > 0)
    
    if np.sum(valid_indices) >= 3:  # Ensure sufficient data points for fitting
        try:
            ### ✅ FULL FIT (ONLY USING BINS > 10 µm)
            mask_full = (ddry_values > 4) & valid_indices  # Only bins greater than 10 µm
            ddry_full = ddry_values[mask_full]
            dN_dD_dry_full = dN_dD_dry[mask_full]

            if len(ddry_full) >= 4:
                popt_full, _ = curve_fit(exponential, ddry_full, dN_dD_dry_full, 
                                         p0=(max(dN_dD_dry_full), 4), maxfev=5000)  # Larger initial guess for `D`
                n0_full, D_full = popt_full

                # ✅ Extend full fit across the entire x-range (not just >10 µm)
                x_fit_full = np.linspace(min(ddry_values[valid_indices]), max(ddry_values[valid_indices]), 100)
                y_fit_full = exponential(x_fit_full, *popt_full)

            ### ✅ ≤10 µm FIT (ONLY USING BINS ≤ 10 µm)
            mask_10um = (ddry_values <= 10) & valid_indices  # Only bins ≤ 10 µm
            ddry_10um = ddry_values[mask_10um]
            dN_dD_dry_10um = dN_dD_dry[mask_10um]

            if len(ddry_10um) >= 5:
                popt_10um, _ = curve_fit(exponential, ddry_10um, dN_dD_dry_10um, 
                                         p0=(max(dN_dD_dry_10um), 5), maxfev=5000)  # Smaller initial guess for `D`
                n0_10um, D_10um = popt_10um

                # ✅ Generate fit curve for ≤10 µm **(independent fit!)**
                x_fit_10um = np.linspace(min(ddry_10um), max(ddry_10um), 50)
                y_fit_10um = exponential(x_fit_10um, *popt_10um)

                # ✅ Plot the Dry Observed Data (Red)
                plt.plot(ddry_values[valid_indices], dN_dD_dry[valid_indices], 
                         color='black', marker='o', linestyle='-', label="Dry Distribution")

                # ✅ Plot the Full-Range Fit (Black Dashed Line) **across entire x-range**
                if len(ddry_full) >= 5:
                    plt.plot(x_fit_full, y_fit_full, color='red', linestyle='--', linewidth=3, 
                             label=f"Full Fit: y = {n0_full:.2e} * exp(-x / {D_full:.2f})")

                # ✅ Plot the **Independent** ≤10 µm Fit (Blue Dotted Line)
                plt.plot(x_fit_10um, y_fit_10um, color='blue', linestyle=':', linewidth=3, 
                         label=f"≤10 µm Fit: y = {n0_10um:.2e} * exp(-x / {D_10um:.2f})")

                plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=14, fontweight="bold")
                plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
                plt.yscale("log")
                plt.ylim(1e-7, 1e1)
                plt.xlim(0, 30)
                plt.xticks(fontweight="bold", fontsize=14)
                plt.yticks(fontweight="bold", fontsize=14)
                plt.title(f"Dry Size Distribution - {selected_date}\nStart: {selected_start} | Stop: {selected_stop}", fontsize=14, fontweight="bold")

                
                plt.legend()
                plt.show()

#                 print(f"Selected Date: {selected_date}")
#                 print(f"Start Time: {selected_start}, Stop Time: {selected_stop}")
                
#                 if len(ddry_full) >= 5:
#                     print(f"\n🔹 Full-Range Fit Parameters (Using Only Bins > 10 µm):")
#                     print(f"Dry Intercept (n0): {n0_full:.2e}")
#                     print(f"Dry E-folding Diameter (D): {D_full:.2f} μm")
                
#                 print(f"\n🔹 ≤10 µm Independent Fit Parameters:")
#                 print(f"Dry Intercept (n0): {n0_10um:.2e}")
#                 print(f"Dry E-folding Diameter (D): {D_10um:.2f} μm")

#             else:
#                 print("Not enough valid points to fit ≤10 µm data.")

#         except RuntimeError:
#             print(f"Fit could not be performed for {selected_date}")

# else:
#     print(f"Dry leg not found for {selected_date}")
#%%
#Just exponential 
common_bins=(2, 30, 30)
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

selected_date = "2022-03-13"
selected_start = 50135.0
selected_stop = 50496.0

selected_dry_leg = next(
    (entry for entry in filtered_master_BCB_ddry if entry['Date'] == selected_date and 
     entry['BCB_start'] == selected_start and entry['BCB_stop'] == selected_stop), 
    None
)

if selected_dry_leg:
    plt.figure(figsize=(8, 6))

    ddry_values = np.array(selected_dry_leg['ddry'])
    dN_dD_dry = np.array(selected_dry_leg['dN/dDdry'])

    valid_indices = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry) & (dN_dD_dry > 0)
    
    if np.sum(valid_indices) >= 3:  # Ensure sufficient data points for fitting
        try:
            ### ✅ FULL FIT (ONLY USING BINS > 10 µm)
            mask_full = (ddry_values > 4) & valid_indices  # Only bins greater than 10 µm
            ddry_full = ddry_values[mask_full]
            dN_dD_dry_full = dN_dD_dry[mask_full]

            if len(ddry_full) >= 4:
                popt_full, _ = curve_fit(exponential, ddry_full, dN_dD_dry_full, 
                                         p0=(max(dN_dD_dry_full), 4), maxfev=5000)  # Larger initial guess for `D`
                n0_full, D_full = popt_full

                # ✅ Extend full fit across the entire x-range (not just >10 µm)
                x_fit_full = np.linspace(min(ddry_values[valid_indices]), max(ddry_values[valid_indices]), 100)
                y_fit_full = exponential(x_fit_full, *popt_full)

                # ✅ Plot the Dry Observed Data (Red)
                plt.plot(ddry_values[valid_indices], dN_dD_dry[valid_indices], 
                         color='black', marker='o', linestyle='-', label="Dry Distribution")

                # ✅ Plot the Full-Range Fit (Black Dashed Line) **across entire x-range**
                if len(ddry_full) >= 5:
                    plt.plot(x_fit_full, y_fit_full, color='red', linestyle='--', linewidth=3, 
                             label=f"Full Fit: y = {n0_full:.2e} * exp(-x / {D_full:.2f})")

    

                plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=14, fontweight="bold")
                plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
                plt.yscale("log")
                plt.ylim(1e-7, 1e1)
                plt.xlim(0, 30)
                plt.xticks(fontweight="bold", fontsize=14)
                plt.yticks(fontweight="bold", fontsize=14)
                plt.title(f"Dry Size Distribution - {selected_date}\nStart: {selected_start} | Stop: {selected_stop}", fontsize=14, fontweight="bold")

                
                plt.legend()
                plt.show()

#                 print(f"Selected Date: {selected_date}")
#                 print(f"Start Time: {selected_start}, Stop Time: {selected_stop}")
                
#                 if len(ddry_full) >= 5:
#                     print(f"\n🔹 Full-Range Fit Parameters (Using Only Bins > 10 µm):")
#                     print(f"Dry Intercept (n0): {n0_full:.2e}")
#                     print(f"Dry E-folding Diameter (D): {D_full:.2f} μm")
                
#                 print(f"\n🔹 ≤10 µm Independent Fit Parameters:")
#                 print(f"Dry Intercept (n0): {n0_10um:.2e}")
#                 print(f"Dry E-folding Diameter (D): {D_10um:.2f} μm")

            else:
                print("Not enough valid points to fit ≤10 µm data.")

        except RuntimeError:
            print(f"Fit could not be performed for {selected_date}")

else:
    print(f"Dry leg not found for {selected_date}")
































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
common_bins=np.linspace(2, 10, 10)
#%%
# Define wind speed bins
windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]

# Store binned distributions
grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

missing_windspeed_count = 0

# Define common bins (10 bin centers between 2 and 10 µm)
common_bins = np.linspace(2, 10, 10)

# Use already fitted exponential size distributions
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

    # Use the already fitted size distribution (DO NOT FIT AGAIN)
    size_dist = n0 * np.exp(-common_bins / D)  # Use existing (n0, D)

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
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)  # Average size distribution
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])

        plt.plot(common_bins, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linewidth=2.5)

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
#Lewis and Schwartz
windspeed_bins = [(0, 5), (5.001, 7), (7.001, 9), (9.001, np.inf)]
common_bins=np.linspace(2, 100, 100)

# # Define wind speed bins
# windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]

# Store binned distributions
grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

missing_windspeed_count = 0

# Define common bins (10 bin centers between 2 and 10 µm)
common_bins=np.linspace(2, 100, 100)

# Use already fitted exponential size distributions
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

    # Use the already fitted size distribution (DO NOT FIT AGAIN)
    size_dist = n0 * np.exp(-common_bins / D)  # Use existing (n0, D)

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
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)  # Average size distribution
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])

        plt.plot(common_bins, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linewidth=2.5)

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
#trying to fix 
import numpy as np
import matplotlib.pyplot as plt

# Define wind speed bins based on Lewis & Schwartz Fig. 22
windspeed_bins = [(0, 5), (5.001, 7), (7.001, 9), (9.001, np.inf)]

# Define common bins (log-spaced to match Lewis & Schwartz)
common_bins = np.logspace(np.log10(0.1), np.log10(100), 50)  # 50 log-spaced bins
bin_widths = np.diff(common_bins)  # Compute bin widths

# Store binned distributions
grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

missing_windspeed_count = 0

# Use already fitted exponential size distributions
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

    # Use the already fitted size distribution (DO NOT FIT AGAIN)
    size_dist = n0 * np.exp(-common_bins / D)  # Use existing (n0, D)

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
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)  # Average size distribution
        avg_distribution_total = avg_distribution[:-1] * bin_widths  # Convert to cm⁻³

        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])

        plt.plot(common_bins[:-1], avg_distribution_total, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linewidth=2.5)

# Set log scale for x-axis (diameter) and y-axis (concentration)
plt.xscale('log')  # Match Lewis & Schwartz log scale
plt.yscale('log')
plt.ylabel(r"Number Concentration (cm$^{-3}$)", fontsize=16, fontweight="bold")  
plt.xlabel("Dry Bin Centers Diameter (µm)", fontsize=16, fontweight="bold")
plt.title('Dry Size Distributions Binned by Average Wind Speed', fontweight='bold', fontsize=17)
plt.legend(title=r"Average wind speed (m s$^{-1}$)")
plt.tight_layout()
plt.ylim(1e-4, 10**2)  # Adjust y-axis to match Lewis & Schwartz

plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.show()

total_legs = sum(len(group) for group in grouped_distributions.values())
print(f"Total number of legs plotted: {total_legs}")

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



import matplotlib.image as mpimg

# Load Lewis & Schwartz image
img = mpimg.imread('/home/disk/eos4/kathem24/activate/data/CAS/LS 57.png')  # Load the figure
fig, ax = plt.subplots(figsize=(8, 6))

# ✅ Set the background image with correct extent
ax.imshow(img, extent=[0.1, 100, 1e-6, 1e2], aspect='auto', alpha=0.3)

# ✅ Convert bin centers to radius (µm)
bin_radius = np.array(bin_center, dtype=float) / 2  

# ✅ Plot the 5-7 m/s bin data
ax.plot(bin_radius, avg_distribution, color='red', linewidth=2.5, label="5-7 m/s, n=99")

# ✅ Set log scales
ax.set_xscale('log')
ax.set_yscale('log')

# ✅ Labels and formatting
ax.set_xlabel("Bin Center Radius (µm)", fontsize=14, fontweight="bold")
ax.set_ylabel("Concentration (/cm³)", fontsize=14, fontweight="bold")
ax.set_title("Overlay on Lewis & Schwartz (5-7 m/s)", fontsize=16, fontweight="bold")
ax.set_xlim(0.1, 100)
ax.set_ylim(1e-6, 1e2)

# ✅ Add legend
ax.legend()

# ✅ Show plot
plt.show()

# %%
