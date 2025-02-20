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
from scipy.spatial import distance
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
   
    for file_path in fname_h20:
        
        df_h20 = pd.read_csv(file_path, skiprows=36, quoting=csv.QUOTE_NONE)

        
        
        df_h20.columns = df_h20.columns.str.strip().str.replace('"', '')

        
        

        # Ensure each column is treated as a string, then strip quotes and convert to numeric
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
            
            
            dryintercept = n0 / (gRh_mean)
            
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
plt.xlim(10**-0.5, 10**1.1)
plt.ylim(10**-1.5, 10**0.7)

# Adjust tick sizes
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')

# Finalize and show the plot
plt.tight_layout()
plt.legend(fontsize=14)
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
plt.ylim(10**-1.9, 10**1)

# Adjust tick sizes
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')

# Show the plot
plt.tight_layout()
plt.show()

#%%
#mass contours in kg/m3
# Define sea salt density (kg/m³)
rho_salt = 2200  

# Function to calculate mass
def calculate_mass(N0, D):
    # Convert N0 from cm⁻³µm⁻¹ to m⁻⁴
    N0_m4 = N0 * 10**6  # Convert cm⁻³ to m⁻³

    # Integrate mass over d^3 with an exponential decay
    integrand = lambda d: np.exp(-d / D) * (d * 1e-6)**3  # Convert µm³ → m³
    mass_integral, _ = quad(integrand, 2, np.inf)  # Integrate from 2 µm to inf

    return rho_salt * N0_m4 * mass_integral  # Multiply by density

# Define the full x and y axis limits
x_min, x_max = 10**-0.1, 10**0.5  # Full x-axis range for slope
y_min, y_max = 10**-1.7, 10**1.5  # Full y-axis range for dry intercept

# Create extended grids to cover the full plot range
xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200)  # Denser grid for slope
ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)  # Denser grid for dry intercept
D_grid_extended, dryintercept_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)

# Recalculate mass grid over the extended grid
mass_grid_extended = np.zeros_like(D_grid_extended)

for i in range(D_grid_extended.shape[0]):
    for j in range(D_grid_extended.shape[1]):
        mass_grid_extended[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j])

# ** Debugging: Check Mass Grid Values**
print("Mass Grid Min:", np.min(mass_grid_extended))
print("Mass Grid Max:", np.max(mass_grid_extended))
print("Mass Grid Sample:", mass_grid_extended[::50, ::50])  # Print some values

# Define the number of mass contour levels
number_of_contours = 20  # Change this value to adjust the number of contours
mass_levels = np.logspace(np.log10(np.min(mass_grid_extended)), np.log10(np.max(mass_grid_extended)), number_of_contours)

# Extract slope (D) and dry intercept from combined data
slope_data = np.array(combined_data['D'])  # Slope values
dry_intercept_data = np.array([entry['dry intercept'] for entry in filtered_master_BCB_dryintercept])  # Dry intercept values

# Ensure lengths match for plotting
if len(slope_data) != len(dry_intercept_data):
    print(f"Mismatch in lengths: Slope={len(slope_data)}, Dry Intercept={len(dry_intercept_data)}")
slope_data = slope_data[:len(dry_intercept_data)]  # Adjust lengths if needed

# ** Debugging: Check min/max slope and intercept**
print("Slope Data Min:", np.min(slope_data), "Max:", np.max(slope_data))
print("Dry Intercept Data Min:", np.min(dry_intercept_data), "Max:", np.max(dry_intercept_data))

# * Plot the scatter plot**
plt.figure(figsize=(10, 8))
plt.scatter(slope_data, dry_intercept_data, c='blue', s=80, alpha=0.7, label="Data Points")

# ** Add mass contours**
contour_plot = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_extended, 
                           levels=mass_levels, colors='red', alpha=0.75)

plt.clabel(contour_plot, inline=True, fontsize=10, fmt='%1.1e', colors='red')

# Add labels and title
plt.xlabel('Slope (µm)', fontsize=19, fontweight='bold')
plt.ylabel('Dry Intercept (cm⁻³ µm⁻¹)', fontsize=19, fontweight='bold')
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
plt.tight_layout()
plt.show()
# %%
#my contour levels in ug/m3
# Define sea salt density (kg/m³)
rho_salt = 2200  

# Function to calculate mass
def calculate_mass(N0, D):
    # Convert N0 from cm⁻³µm⁻¹ to m⁻⁴
    N0_m4 = N0 * 10**6  # Convert cm⁻³ to m⁻³

    # Integrate mass over d^3 with an exponential decay
    integrand = lambda d: np.exp(-d / D) * (d * 1e-6)**3  # Convert µm³ → m³
    mass_integral, _ = quad(integrand, 2, np.inf)  # Integrate from 2 µm to inf

    return rho_salt * N0_m4 * mass_integral  # Multiply by density

# Define the full x and y axis limits
x_min, x_max = 10**-0.1, 10**1.05  # Full x-axis range for slope
y_min, y_max = 10**-1.7, 10**0.8  # Full y-axis range for dry intercept

# Create extended grids to cover the full plot range
xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200)  # Denser grid for slope
ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)  # Denser grid for dry intercept
D_grid_extended, dryintercept_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)

# Recalculate mass grid over the extended grid
mass_grid_extended = np.zeros_like(D_grid_extended)

for i in range(D_grid_extended.shape[0]):
    for j in range(D_grid_extended.shape[1]):
        mass_grid_extended[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j]) * 1e9  # Convert kg/m³ to µg/m³

# ** Debugging: Check Mass Grid Values**
print("Mass Grid Min:", np.min(mass_grid_extended))
print("Mass Grid Max:", np.max(mass_grid_extended))
print("Mass Grid Sample:", mass_grid_extended[::50, ::50])  # Print some values

# Define the number of mass contour levels
number_of_contours = 18  # Change this value to adjust the number of contours
mass_levels = np.logspace(np.log10(np.min(mass_grid_extended[mass_grid_extended > 0])),
                          np.log10(np.max(mass_grid_extended)), number_of_contours)

# Extract slope (D) and dry intercept from combined data
slope_data = np.array(combined_data['D'])  # Slope values
dry_intercept_data = np.array([entry['dry intercept'] for entry in filtered_master_BCB_dryintercept if not np.isnan(entry['dry intercept'])])

# Ensure lengths match for plotting
if len(slope_data) != len(dry_intercept_data):
    print(f"Mismatch in lengths: Slope={len(slope_data)}, Dry Intercept={len(dry_intercept_data)}")
slope_data = slope_data[:len(dry_intercept_data)]  # Adjust lengths if needed

# ** Debugging: Check min/max slope and intercept**
print("Slope Data Min:", np.min(slope_data), "Max:", np.max(slope_data))
print("Dry Intercept Data Min:", np.min(dry_intercept_data), "Max:", np.max(dry_intercept_data))

# * Plot the scatter plot**
plt.figure(figsize=(10, 8))
plt.scatter(slope_data, dry_intercept_data, c='blue', s=80, alpha=0.7, label="Data Points")

# Apply manually defined contour levels
contour_plot = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_extended, 
                           levels=mass_levels, colors='red', alpha=0.75)

# Improve label readability by using integer formatting
plt.clabel(contour_plot, inline=True, fontsize=11, fmt='%d µg/m³', colors='black', inline_spacing=5)
for txt in contour_plot.labelTexts:
    txt.set_fontweight('bold')
    txt.set_rotation(15)    # Make labels bold

plt.xlabel('Slope (µm)', fontsize=19, fontweight='bold')
plt.ylabel('Dry Intercept (cm⁻³ µm⁻¹)', fontsize=19, fontweight='bold')
plt.title('Below Cloud Base January - June 2022', fontsize=19, fontweight='bold')

plt.xscale('log')
plt.yscale('log')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
# %%
#setting Rob's contour levels 

# Define sea salt density (kg/m³)
rho_salt = 2200  

# Function to calculate mass
def calculate_mass(N0, D):
    # Convert N0 from cm⁻³µm⁻¹ to m⁻⁴
    N0_m4 = N0 * 10**6  # Convert cm⁻³ to m⁻³

    # Integrate mass over d^3 with an exponential decay
    integrand = lambda d: np.exp(-d / D) * (d * 1e-6)**3  # Convert µm³ → m³
    mass_integral, _ = quad(integrand, 2, np.inf)  # Integrate from 2 µm to inf

    return rho_salt * N0_m4 * mass_integral  # Multiply by density

# Define the full x and y axis limits
x_min, x_max = 10**-0.1, 10**1.05  # Full x-axis range for slope
y_min, y_max = 10**-1.7, 10**0.8  # Full y-axis range for dry intercept

# Create extended grids to cover the full plot range
xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200)  # Denser grid for slope
ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)  # Denser grid for dry intercept
D_grid_extended, dryintercept_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)

# Recalculate mass grid over the extended grid
mass_grid_extended = np.zeros_like(D_grid_extended)

for i in range(D_grid_extended.shape[0]):
    for j in range(D_grid_extended.shape[1]):
        mass_grid_extended[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j]) * 1e9  # Convert kg/m³ to µg/m³

# ** Debugging: Check Mass Grid Values**
print("Mass Grid Min:", np.min(mass_grid_extended))
print("Mass Grid Max:", np.max(mass_grid_extended))

# Define contour levels based on the dry mass grid (rounded, log-spaced as per advisor's request)
mass_levels = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]

# Extract slope (D) and dry intercept from combined data
slope_data = np.array(combined_data['D'])  # Slope values
dry_intercept_data = np.array([entry['dry intercept'] for entry in filtered_master_BCB_dryintercept if not np.isnan(entry['dry intercept'])])

# Ensure lengths match for plotting
if len(slope_data) != len(dry_intercept_data):
    print(f"Mismatch in lengths: Slope={len(slope_data)}, Dry Intercept={len(dry_intercept_data)}")
slope_data = slope_data[:len(dry_intercept_data)]  # Adjust lengths if needed

# ** Debugging: Check min/max slope and intercept**
print("Slope Data Min:", np.min(slope_data), "Max:", np.max(slope_data))
print("Dry Intercept Data Min:", np.min(dry_intercept_data), "Max:", np.max(dry_intercept_data))

# ** Plot the scatter plot**
plt.figure(figsize=(10, 8))
plt.scatter(slope_data, dry_intercept_data, c='blue', s=80, alpha=0.7, label="Data Points")

# Apply manually defined contour levels with improved labels
contour_plot = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_extended, 
                           levels=mass_levels, colors='red', alpha=0.75, linewidths=1.5)

# Improve label readability with larger, bold, round numbers
fmt = {level: f'{int(level)} µg/m³' for level in mass_levels}  # Convert to integer labels
plt.clabel(contour_plot, inline=True, fontsize=13, fmt=fmt, colors='black', inline_spacing=5)
for txt in contour_plot.labelTexts:
    txt.set_fontweight('bold')
    txt.set_rotation(15)  # Rotate labels to follow contour lines

# Set labels, title, and formatting
plt.xlabel('Slope (µm)', fontsize=19, fontweight='bold')
plt.ylabel('Dry Intercept (cm⁻³ µm⁻¹)', fontsize=19, fontweight='bold')
plt.title('Below Cloud Base January - June 2022 Dry Mass', fontsize=19, fontweight='bold')

# Set logarithmic scales
plt.xscale('log')
plt.yscale('log')

# Set axis limits
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Adjust tick sizes
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# %%
#Saving the dry masses to their own dictionary
# Define sea salt density (kg/m³)
dry_mass = []

# Extract date and D values correctly
date_list = combined_data['Date']
D_list = combined_data['D']

# Compute dry mass for each entry in `filtered_master_BCB_dryintercept`
for entry in filtered_master_BCB_dryintercept:
    date = entry['Date']
    dry_intercept = entry['dry intercept']

    # Find matching D value
    if date in date_list:
        index = date_list.index(date)  # Find index of the date
        D = D_list[index]  # Get corresponding D value
    else:
        D = np.nan  # Assign NaN if no match found

    # Compute dry mass if valid D value is found
    if not np.isnan(D):
        mass_value = calculate_mass(dry_intercept, D) * 1e9  # Convert kg/m³ to µg/m³

        # Store in dictionary
        dry_mass.append({
            'Date': date,
            'Slope (D)': D,
            'Dry Intercept': dry_intercept,
            'Dry Mass (µg/m³)': mass_value
        })

# # ** Save to a file **
# import json
# with open("dry_mass_data.json", "w") as f:
#     json.dump(dry_mass, f, indent=4)
# print("Dry mass data saved to dry_mass_data.json")

print("Mass Grid Min:", np.min(mass_grid_extended))
print("Mass Grid Max:", np.max(mass_grid_extended))

mass_levels = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]

slope_data = np.array(combined_data['D'])
dry_intercept_data = np.array([entry['dry intercept'] for entry in filtered_master_BCB_dryintercept if not np.isnan(entry['dry intercept'])])
if len(slope_data) != len(dry_intercept_data):
    print(f"Mismatch in lengths: Slope={len(slope_data)}, Dry Intercept={len(dry_intercept_data)}")
slope_data = slope_data[:len(dry_intercept_data)] 

print("Slope Data Min:", np.min(slope_data), "Max:", np.max(slope_data))
print("Dry Intercept Data Min:", np.min(dry_intercept_data), "Max:", np.max(dry_intercept_data))

plt.figure(figsize=(10, 8))
plt.scatter(slope_data, dry_intercept_data, c='blue', s=80, alpha=0.7, label="Data Points")

contour_plot = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_extended, 
                           levels=mass_levels, colors='red', alpha=0.75, linewidths=1.5)

fmt = {level: f'{int(level)} µg/m³' for level in mass_levels} 
plt.clabel(contour_plot, inline=True, fontsize=13, fmt=fmt, colors='black', inline_spacing=5)
for txt in contour_plot.labelTexts:
    txt.set_fontweight('bold')
    txt.set_rotation(15) 

plt.xlabel('Slope (µm)', fontsize=19, fontweight='bold')
plt.ylabel('Dry Intercept (cm⁻³ µm⁻¹)', fontsize=19, fontweight='bold')
plt.title('Below Cloud Base January - June 2022 Dry Mass', fontsize=19, fontweight='bold')
plt.xscale('log')
plt.yscale('log')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# %%
# Extract all mass values from dry_mass
mass_values_ug = [entry['Dry Mass (µg/m³)'] for entry in dry_mass]
min_mass_ug = min(mass_values_ug)
max_mass_ug = max(mass_values_ug)
print(f"Min Mass (µg/m³): {min_mass_ug}")
print(f"Max Mass (µg/m³): {max_mass_ug}")
# %%
#Fix mismatch in slope and int length 

filtered_dry_mass = [entry for entry in dry_mass if not np.isnan(entry['Slope (D)']) and not np.isnan(entry['Dry Intercept'])]

print(f"Filtered Dry Mass Entries: {len(filtered_dry_mass)}")  # Should be equal for slope & intercept now

# Extract slope and intercept as NumPy arrays
slope_array = np.array([entry['Slope (D)'] for entry in filtered_dry_mass]).reshape(-1, 1)
intercept_array = np.array([entry['Dry Intercept'] for entry in filtered_dry_mass]).reshape(-1, 1)
data_points = np.column_stack((slope_array, intercept_array))


# %%

# Define the three specific (Slope, Dry Intercept) cases
slope_col = "Slope (D)"  # Corrected slope column
intercept_col = "Dry Intercept"  # Corrected dry intercept column
mass_col = "Dry Mass (µg/m³)"  # Corrected mass column
slope_array = np.array([entry[slope_col] for entry in filtered_dry_mass]).reshape(-1, 1)
intercept_array = np.array([entry[intercept_col] for entry in filtered_dry_mass]).reshape(-1, 1)
data_points = np.column_stack((slope_array, intercept_array))
def find_closest_match(target_slope, target_intercept):
    distances = np.linalg.norm(data_points - np.array([target_slope, target_intercept]), axis=1)
    closest_index = distances.argmin()
    return filtered_dry_mass[closest_index]  

# Define the three specific (Slope, Dry Intercept) cases
target_cases = [
    (1.5, 0.2),
    (1.5, 1),
    (1.2, 3)
]

closest_matches = []
for D, N0 in target_cases:
    closest_match = find_closest_match(D, N0)
    closest_matches.append(closest_match)
    print(f"Target (Slope={D}, Intercept={N0}) → Closest Match: "
          f"Slope={closest_match[slope_col]}, Intercept={closest_match[intercept_col]}, "
          f"Mass (µg/m³)={closest_match[mass_col]}")
df_closest_matches = pd.DataFrame(closest_matches)
print("\nClosest Matches for Target Cases:")
print(df_closest_matches)

# %%

#bins of power of 2 for histogram 
mass_values_ug = [entry['Dry Mass (µg/m³)'] for entry in filtered_dry_mass]
bins = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 
                 16384, 32768, 65536, 131072]) 

plt.figure(figsize=(10, 6))
plt.hist(mass_values_ug, bins=bins, color='blue', alpha=0.7, edgecolor='black', density=False)
plt.xscale('log')
plt.yscale('log') 
plt.xlabel('Mass (µg/m³)', fontsize=16, fontweight='bold')
plt.ylabel('Frequency of flight legs', fontsize=16, fontweight='bold')
plt.title('Below Cloud Base January-June 2022', fontsize=18, fontweight='bold')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()

# %%
#calculating hydrated mass ug/m3
df_combined = pd.DataFrame({
    'Date': combined_data['Date'],
    'D': combined_data['D'],  # Slope (D)
    'dryintercept': [entry['dry intercept'] for entry in filtered_master_BCB_dryintercept]  # Dry intercept (N0)
})

filtered_combined_clean = df_combined.dropna(subset=['D', 'dryintercept'])
filtered_combined_clean = filtered_combined_clean[np.isfinite(filtered_combined_clean[['D', 'dryintercept']].values).all(axis=1)]

print(f"Number of valid entries in filtered_combined_clean: {len(filtered_combined_clean)}")
# %%
#saving the hydrated masses 
rho_salt = 2200  
def calculate_mass(N0, D):
    # Convert N0 from cm⁻³µm⁻¹ to m⁻⁴
    N0_m4 = N0 * 10**6  # Convert cm⁻³ to m⁻³

    
    integrand = lambda d: np.exp(-d / D) * (d * 1e-6)**3  # Convert µm³ → m³
    mass_integral, _ = quad(integrand, 2, np.inf)  # Integrate from 2 µm to ∞

    return rho_salt * N0_m4 * mass_integral  


ambient_mass_dict = {}
for entry in master_BCB_exponential:
    for leg in master_BCB_exponential[entry]:  
        date = leg['Date']
        N0 = leg['n0'] 
        D = filtered_combined_clean.loc[filtered_combined_clean['Date'] == date, 'D'].values  # Get matching D value

        if len(D) == 0:
            continue  

        D = D[0] 
        mass = calculate_mass(N0, D) * 1e9  # Convert kg/m³ to µg/m³

        
        if date not in ambient_mass_dict:
            ambient_mass_dict[date] = [] 

        ambient_mass_dict[date].append({
            'Date': date,
            'Slope (D)': D,
            'Hydrated Intercept (N0)': N0,
            'Mass (µg/m³)': mass
        })

ambient_mass_ug = [entry['Mass (µg/m³)'] for sublist in ambient_mass_dict.values() for entry in sublist]

x_min, x_max = 10**-0.1, 10**1.05 
y_min, y_max = 10**-1.6, 10**0.95  

xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200)  
ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)  
D_grid_extended, hydratedintercept_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)
mass_grid_extended = np.zeros_like(D_grid_extended)

for i in range(D_grid_extended.shape[0]):
    for j in range(D_grid_extended.shape[1]):
        mass_grid_extended[i, j] = calculate_mass(hydratedintercept_grid_extended[i, j], D_grid_extended[i, j]) * 1e9  # Convert kg/m³ to µg/m³
mass_levels = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
slope_data = np.array(filtered_combined_clean['D'])

hydrated_intercept_data = np.array([entry['n0'] for sublist in master_BCB_exponential.values() for entry in sublist])
min_length = min(len(slope_data), len(hydrated_intercept_data))
slope_data = slope_data[:min_length]
hydrated_intercept_data = hydrated_intercept_data[:min_length]
plt.figure(figsize=(10, 8))
plt.scatter(slope_data, hydrated_intercept_data, c='blue', s=80, alpha=0.7, label="Hydrated Data Points")
contour_plot = plt.contour(D_grid_extended, hydratedintercept_grid_extended, mass_grid_extended, 
                           levels=mass_levels, colors='blue', linestyles='solid', alpha=0.75)

plt.clabel(contour_plot, inline=True, fontsize=12, fmt='%d µg/m³', colors='black', inline_spacing=5)
for txt in contour_plot.labelTexts:
    txt.set_fontweight('bold')
    txt.set_rotation(30)   

plt.xlabel('Slope (µm)', fontsize=19, fontweight='bold')
plt.ylabel('Hydrated Intercept (cm⁻³ µm⁻¹)', fontsize=19, fontweight='bold')
plt.title('Below Cloud Base Hydrated Mass January - June 2022', fontsize=19, fontweight='bold')
plt.xscale('log')
plt.yscale('log')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# %%
#ambient mass histogram 
hydrated_mass_values_ug = [entry['Mass (µg/m³)'] for entries in ambient_mass_dict.values() for entry in entries]
bins = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 
                 16384, 32768, 65536, 131072]) 

plt.figure(figsize=(10, 6))
plt.hist(hydrated_mass_values_ug, bins=bins, color='red', alpha=0.7, edgecolor='black', density=False)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Mass (µg/m³)', fontsize=16, fontweight='bold')
plt.ylabel('Frequency of flight legs', fontsize=16, fontweight='bold')
plt.title('Histogram of Hydrated Mass', fontsize=18, fontweight='bold')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()
# %%
#Dry mass and ambient mass together 
mass_values_ug = [entry['Dry Mass (µg/m³)'] for entry in filtered_dry_mass]
hydrated_mass_values_ug = [entry['Mass (µg/m³)'] for entries in ambient_mass_dict.values() for entry in entries]
bins = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 
                 16384, 32768, 65536, 131072]) 
plt.figure(figsize=(10, 6))
plt.hist(mass_values_ug, bins=bins, color='red', alpha=0.6, edgecolor='black', label="Dry Mass", density=False)
plt.hist(hydrated_mass_values_ug, bins=bins, color='blue', alpha=0.5, edgecolor='black', label="Hydrated Mass", density=False)
plt.xscale('log')  
plt.yscale('log')
plt.xlabel('Mass (µg/m³)', fontsize=16, fontweight='bold')
plt.ylabel('Frequency of flight legs', fontsize=16, fontweight='bold')
plt.title('Comparison of Dry vs Hydrated Mass', fontsize=18, fontweight='bold')
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()

mean_dry_mass = np.mean(mass_values_ug)
mean_hydrated_mass = np.mean(hydrated_mass_values_ug)

median_dry_mass = np.median(mass_values_ug)
median_hydrated_mass = np.median(hydrated_mass_values_ug)

# Print results
print(f"Mean Dry Mass: {mean_dry_mass:.2f} µg/m³, Mean Hydrated Mass: {mean_hydrated_mass:.2f} µg/m³")
print(f"Median Dry Mass: {median_dry_mass:.2f} µg/m³, Median Hydrated Mass: {median_hydrated_mass:.2f} µg/m³")
# %%

print(f"Min Dry Mass: {min(mass_values_ug):.2f}, Max Dry Mass: {max(mass_values_ug):.2f}")
print(f"Min Hydrated Mass: {min(hydrated_mass_values_ug):.2f}, Max Hydrated Mass: {max(hydrated_mass_values_ug):.2f}")

# %%

plt.figure(figsize=(8, 6))
sns.boxplot(data=[mass_values_ug, hydrated_mass_values_ug], palette=['red', 'blue'])
plt.xticks([0, 1], ['Dry Mass', 'Hydrated Mass'], fontsize=14)
plt.ylabel('Mass (µg/m³)', fontsize=14, fontweight='bold')
plt.title('Boxplot of Dry vs Hydrated Mass', fontsize=16, fontweight='bold')
plt.yscale('log')
plt.show()

# %%
def remove_outliers(data):
    Q1, Q3 = np.percentile(data, [25, 75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return [x for x in data if lower_bound <= x <= upper_bound]
filtered_dry_mass_values = remove_outliers(mass_values_ug)
filtered_hydrated_mass_values = remove_outliers(hydrated_mass_values_ug)
mean_filtered_dry_mass = np.mean(filtered_dry_mass_values)
mean_filtered_hydrated_mass = np.mean(filtered_hydrated_mass_values)
median_filtered_dry_mass = np.median(filtered_dry_mass_values)
median_filtered_hydrated_mass = np.median(filtered_hydrated_mass_values)
print(f"Filtered Mean Dry Mass: {mean_filtered_dry_mass:.2f} µg/m³, Filtered Mean Hydrated Mass: {mean_filtered_hydrated_mass:.2f} µg/m³")
print(f"Filtered Median Dry Mass: {median_filtered_dry_mass:.2f} µg/m³, Filtered Median Hydrated Mass: {median_filtered_hydrated_mass:.2f} µg/m³")

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
# %%
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
# %%
#25 bins: A common x-axis is created to average the dry size distributions onto
# We will use this common x-axis to interpolate the dry size distributions. Note that the 
# number of bins is arbitrary and can be adjusted as needed and the spacing between the bins is equal.  
def size_distribution(x, dryint, D):
    dryint = dryint 
    return dryint * np.exp(-x / D)
#We are randomly assigning bin lengths here, but we have 25 bins that end at 10 um d or any set diameter you want 
common_bins = np.linspace(2.5, 10, 25)

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

plt.ylabel('Clear mean droplet concentration (/cm^3/um)', fontweight='bold', fontsize=14)
plt.xlabel('Bin diameter (um)', fontweight='bold', fontsize=14)
plt.title('Below cloud base dry size distribution January-June 2022', fontweight='bold', fontsize=16)
plt.yscale('log')
plt.xscale('linear')
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.tight_layout()
line_count = len(plt.gca().get_lines())
print(f"Total number of lines plotted: {line_count}")
plt.show()
# %%
#Just plotting the three mass size distributions

# Define the three specific cases
target_cases = [
    {"Date": "2022-01-24", "Slope": 1.546385 , "Dry Intercept": 0.240671},
    {"Date": "2022-06-08", "Slope": 1.532267, "Dry Intercept":  0.997450 },
    {"Date": "2022-06-14", "Slope": 1.075513, "Dry Intercept": 2.968380}
]

# Define common bins
common_bins = np.linspace(2.5, 10, 25)

# Interpolated values for the three cases
interpolated_values_cases = []
matched_cases = {}  # Dictionary to store the best match per case

# Iterate through filtered_master_BCB_ddry to find matching cases
for entry_ddry in filtered_master_BCB_ddry:
    date = entry_ddry['Date']
    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])

    # Find matching dry intercept entry
    matching_intercept = next(
        (entry for entry in filtered_master_BCB_dryintercept if entry['Date'] == date), None
    )
    
    if matching_intercept is None:
        continue  # Skip if no match found

    dryint = matching_intercept['dry intercept']

    # Check if this entry matches one of the target cases
    for case in target_cases:
        if date == case["Date"]:
            # Compute distance to target slope & intercept
            dist = np.sqrt((D - case["Slope"])**2 + (dryint - case["Dry Intercept"])**2)

            # Store the best match (smallest distance)
            if date not in matched_cases or dist < matched_cases[date]['distance']:
                matched_cases[date] = {
                    'Slope': D,
                    'Dry Intercept': dryint,
                    'Dry Sizes': ddry_values,
                    'distance': dist  # Keep track of how close this match is
                }

# Now that we have the best match per case, interpolate and store them
for date, match in matched_cases.items():
    interp_func = interp1d(match['Dry Sizes'], match['Dry Intercept'] * np.exp(-match['Dry Sizes'] / match['Slope']), kind='linear', fill_value='extrapolate')
    interpolated_leg_values = interp_func(common_bins)

    interpolated_values_cases.append({
        'Date': date,
        'Slope': match['Slope'],
        'Dry Intercept': match['Dry Intercept'],
        'interpolated_values': interpolated_leg_values.tolist()
    })

# Check how many cases were found
print(f"\n Final Matched Cases: {len(interpolated_values_cases)} (should be 3)")

# Plot only the three selected cases
plt.figure(figsize=(12, 8))

colors = ['blue', 'orange', 'green']
for idx, entry in enumerate(interpolated_values_cases):
    date = entry['Date']
    slope = entry['Slope']
    dry_intercept = entry['Dry Intercept']
    leg_values = entry['interpolated_values']

    plt.plot(common_bins, leg_values, color=colors[idx], linewidth=2.5, label=f"{date}, Slope={slope:.2f}, Int={dry_intercept:.2f}")

plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Size Distributions for Selected GCCN Cases (One Match per Date)', fontweight='bold')
plt.yscale('log')
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

# Plot all size distributions in the background
plt.figure(figsize=(12, 8))

# Plot all size distributions in light gray for reference
for entry in interpolated_values:
    leg_values = entry['interpolated_values']
    plt.plot(common_bins, leg_values, color='lightgray', alpha=0.5, linewidth=0.5)

# Plot the three selected cases in bold colors
colors = ['blue', 'orange', 'green']
for idx, entry in enumerate(interpolated_values_cases):
    date = entry['Date']
    slope = entry['Slope']
    dry_intercept = entry['Dry Intercept']
    leg_values = entry['interpolated_values']

    plt.plot(common_bins, leg_values, color=colors[idx], linewidth=2, label=f"{date}, Slope={slope:.2f}, Int={dry_intercept:.2f}")

# Formatting
plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold', fontsize=14)
plt.xlabel('Bin diameter (µm)', fontweight='bold', fontsize=14)
plt.title('Comparison of Selected GCCN Cases with All Size Distributions', fontweight='bold', fontsize=16)
plt.yscale('log')
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()
# %%
size_distribution_dict = {}

# Iterate through entries in filtered_master_BCB_ddry
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i] 
    
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']
    D = entry_ddry['D']
    dryint = entry_dryintercept['dry intercept']
    ddry_values = np.array(entry_ddry['filtered_ddry'])

    # Interpolate the size distribution
    interp_func = interp1d(ddry_values, size_distribution(ddry_values, dryint, D), kind='linear', fill_value='extrapolate')
    interpolated_leg_values = interp_func(common_bins)

    # Store in dictionary
    size_distribution_dict[(date, leg_index)] = {
        'Date': date,
        'Leg_index': leg_index,
        'Slope': D,
        'Dry Intercept': dryint,
        'Interpolated Values': interpolated_leg_values
    }

# Print the first few entries to verify
for key, value in list(size_distribution_dict.items())[:5]:  # Print first 5 entries
    print(f"{key}: Slope={value['Slope']}, Dry Int={value['Dry Intercept']}")
# %%
target_cases = [
    {"Date": "2022-01-24", "Slope": 1.546385 , "Dry Intercept": 0.240671},
    {"Date": "2022-06-08", "Slope": 1.532267, "Dry Intercept":  0.997450 },
    {"Date": "2022-06-14", "Slope": 1.075513, "Dry Intercept": 2.968380}
]


# Find closest matches in the dictionary
selected_distributions = []

for case in target_cases:
    best_match = None
    min_diff = float("inf")

    for key, value in size_distribution_dict.items():
        if value["Date"] == case["Date"]:
            diff = abs(value["Slope"] - case["Slope"]) + abs(value["Dry Intercept"] - case["Dry Intercept"])
            if diff < min_diff:
                min_diff = diff
                best_match = value

    if best_match:
        selected_distributions.append(best_match)

# Plot only the three selected cases
plt.figure(figsize=(12, 8))
colors = ['blue', 'orange', 'green']

for idx, entry in enumerate(selected_distributions):
    date = entry['Date']
    slope = entry['Slope']
    dry_intercept = entry['Dry Intercept']
    leg_values = entry['Interpolated Values']

    plt.plot(common_bins, leg_values, color=colors[idx], linewidth=2, 
             label=f"{date}")

# Formatting
plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Size Distributions for Selected GCCN Cases', fontweight='bold')
plt.yscale('log')
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()
# %%
# Define the function for size distribution
def size_distribution(x, dryint, D):
    return dryint * np.exp(-x / D)

# Define the common bins
common_bins = np.linspace(2.5, 10, 25)

# Create dictionary to store size distributions
size_distribution_dict = {}

# Iterate through entries in filtered_master_BCB_ddry
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i] 

    date = entry_ddry['Date']
    leg_index = entry_ddry['Leg_index']
    D = entry_ddry['D']
    dryint = entry_dryintercept['dry intercept']
    ddry_values = np.array(entry_ddry['filtered_ddry'])

    # Interpolate the size distribution
    interp_func = interp1d(ddry_values, size_distribution(ddry_values, dryint, D), kind='linear', fill_value='extrapolate')
    interpolated_leg_values = interp_func(common_bins)

    # Store in dictionary
    size_distribution_dict[(date, leg_index)] = {
        'Date': date,
        'Leg_index': leg_index,
        'Slope': D,
        'Dry Intercept': dryint,
        'Interpolated Values': interpolated_leg_values
    }

# Print first few entries to verify
for key, value in list(size_distribution_dict.items())[:5]:
    print(f"{key}: Slope={value['Slope']}, Dry Int={value['Dry Intercept']}")
# %%

#%%

from scipy.optimize import curve_fit


# Extract mass values for matched cases
dry_mass_dict = {entry["Date"]: entry["Dry Mass (µg/m³)"] for entry in dry_mass}

# Plot all size distributions in gray
plt.figure(figsize=(12, 8))
for entry in size_distribution_dict.values():
    plt.plot(common_bins, entry["Interpolated Values"], color="gray", alpha=0.2)

# Plot the three selected cases in distinct colors
colors = ["blue", "orange", "green"]
for entry, color in zip(selected_distributions, colors):
    date = entry["Date"]
    slope = entry["Slope"]
    dry_intercept = entry["Dry Intercept"]
    mass_value = dry_mass_dict.get(date, "N/A")  # Fetch the mass from the corrected dictionary

    plt.plot(common_bins, entry["Interpolated Values"], color=color, linewidth=2.5,
             label=f"{date}, Slope={slope:.2f}, Int={dry_intercept:.2f}, Mass={mass_value:.6f} µg/m³")

# Axis labels and title
plt.ylabel("Clear mean droplet concentration (/cm³/µm)", fontweight="bold")
plt.xlabel("Bin diameter (µm)", fontweight="bold")
plt.title(" Below Cloud Base January - June 2022", fontweight="bold")
plt.yscale("log")

# Adjust layout and legend
plt.legend()
plt.tight_layout()
plt.show()
# %%
#trying to fit an exponential 


# Define bin centers
common_bins = np.linspace(2.5, 10, 25)  # Define bin centers from 0 to 10 µm with 10 bins

# Define the exponential function
def size_distribution(ddry, N0, D):
    return N0 * np.exp(-ddry / D)

# Dictionary to store fitted parameters
fitted_params = {}

# Print the number of distributions being plotted
print(f"Total size distributions being plotted: {len(size_distribution_dict)}")

# Plot all raw size distributions in gray
plt.figure(figsize=(12, 8))

for date, entry in size_distribution_dict.items():
    x_data = np.array(common_bins)  # Bin centers (fixed)
    y_data = np.array(entry["Interpolated Values"])  # Observed droplet concentrations

    # Ensure x_data and y_data are the same length
    min_length = min(len(x_data), len(y_data))
    x_data = x_data[:min_length]
    y_data = y_data[:min_length]

    # Remove NaN and non-positive values from y_data while keeping corresponding x_data
    valid_indices = ~np.isnan(y_data) & (y_data > 0)
    x_data_filtered = x_data[valid_indices]
    y_data_filtered = y_data[valid_indices]

    # Skip if not enough valid data points
    if len(x_data_filtered) < 3:
        print(f"Skipping {date}: Not enough valid data points for fitting.")
        continue

    # Define initial parameter guesses
    initial_guess = [max(y_data_filtered), 2.0]  # (Initial N0, Initial D)

    try:
        # Fit the function to the data
        popt, _ = curve_fit(size_distribution, x_data_filtered, y_data_filtered, p0=initial_guess, maxfev=5000)

        # Store the fitted parameters
        fitted_params[date] = {"N0": popt[0], "D": popt[1]}

        # Plot the raw size distribution in gray
        # plt.plot(x_data, y_data, color="gray", alpha=0.2)

        # Generate fitted curve
        fitted_curve = size_distribution(x_data_filtered, *popt)

        # Plot fitted exponential function
        plt.plot(x_data_filtered, fitted_curve, linestyle='-', linewidth=2, alpha=0.8, label=f"{date}")

    except RuntimeError:
        print(f"Fit did not converge for {date}")

plt.yscale("log")
plt.xlabel("Bin diameter (µm)", fontsize=14, fontweight="bold")
plt.ylabel("Droplet concentration (/cm³/µm)", fontsize=14, fontweight="bold")
plt.title("Below Cloud Base January - June 2022", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()
# %%
# Fitting an exponential model for GCCN size distributions
# Define bin centers
common_bins = np.linspace(2.5, 10, 25)  # Define bin centers from 0 to 10 µm

# Define the exponential function
def size_distribution(ddry, N0, D):
    return N0 * np.exp(-ddry / D)

# Dictionary to store fitted parameters
fitted_params = {}

# Selected cases with known leg indices
selected_cases = [
    {"Date": "2022-01-24", "Leg_index": 8 },
    {"Date": "2022-06-08", "Leg_index":17 },
    {"Date": "2022-06-14", "Leg_index": 3}
]

# Print the number of distributions being plotted
print(f"Total size distributions being plotted: {len(size_distribution_dict)}")

# Track which selected cases are found
found_cases = []

# Initialize plot
plt.figure(figsize=(12, 8))

# Iterate through all distributions
for entry in size_distribution_dict.values():
    date = entry["Date"]
    leg_index = entry["Leg_index"]
    x_data = np.array(common_bins)  # Bin centers (fixed)
    y_data = np.array(entry["Interpolated Values"])  # Observed droplet concentrations

    # Ensure x_data and y_data are the same length
    min_length = min(len(x_data), len(y_data))
    x_data = x_data[:min_length]
    y_data = y_data[:min_length]

    # Remove NaN and non-positive values from y_data while keeping corresponding x_data
    valid_indices = ~np.isnan(y_data) & (y_data > 0)
    x_data_filtered = x_data[valid_indices]
    y_data_filtered = y_data[valid_indices]

    # Skip if not enough valid data points
    if len(x_data_filtered) < 3:
        print(f"Skipping {date}, Leg {leg_index}: Not enough valid data points for fitting.")
        continue

    # Define initial parameter guesses
    initial_guess = [max(y_data_filtered), 2.0]  # (Initial N0, Initial D)

    try:
        # Fit the function to the data
        popt, _ = curve_fit(size_distribution, x_data_filtered, y_data_filtered, p0=initial_guess, maxfev=5000)

        # Store the fitted parameters
        fitted_params[f"{date}_Leg{leg_index}"] = {"N0": popt[0], "D": popt[1]}

        # Generate fitted curve
        fitted_curve = size_distribution(x_data_filtered, *popt)

        # Check if this distribution is a selected case
        is_selected = any(
            case["Date"] == date and case["Leg_index"] == leg_index
            for case in selected_cases
        )

        # Assign color: gray for all, specific colors for selected cases
        if is_selected:
            found_cases.append((date, leg_index))  # Track found cases
            case_index = next(
                (i for i, case in enumerate(selected_cases)
                 if case["Date"] == date and case["Leg_index"] == leg_index),
                None
            )
            if case_index is not None:
                color = ["blue", "orange", "green"][case_index]
                label = f"{date}"
                plt.plot(x_data_filtered, fitted_curve, color=color, linewidth=2.5, label=label)
        else:
            plt.plot(x_data_filtered, fitted_curve, linestyle='-', color="gray", alpha=0.2)

    except RuntimeError:
        print(f"Fit did not converge for {date}, Leg {leg_index}")

print("\nFound Cases:")
for case in found_cases:
    print(f"  {case[0]}, Leg {case[1]}")
missing_cases = [case for case in selected_cases if (case["Date"], case["Leg_index"]) not in found_cases]
if missing_cases:
    print("\nMissing Cases:")
    for case in missing_cases:
        print(f"  {case['Date']}, Leg {case['Leg_index']} (not found in fitted data)")

plt.yscale("log")
plt.xlabel("Bin diameter (µm)", fontsize=14, fontweight="bold")
plt.ylabel("Droplet concentration (/cm³/µm)", fontsize=14, fontweight="bold")
plt.title("Below Cloud Base January - June 2022", fontsize=16, fontweight="bold")
plt.legend()
plt.tight_layout()
plt.show()
# %%
# Define the exponential function
def size_distribution(ddry, N0, D):
    return N0 * np.exp(-ddry / D)

# Dictionary to store fitted parameters
fitted_params = {}

# Selected cases with known leg indices
selected_cases = [
    {"Date": "2022-01-24", "Leg_index": 8, "Mass": 17.47},
    {"Date": "2022-06-08", "Leg_index": 17, "Mass": 65.76},
    {"Date": "2022-06-14", "Leg_index": 3, "Mass": 57.82}
]

# Print the number of distributions being plotted
print(f"Total size distributions being plotted: {len(size_distribution_dict)}")

# Track which selected cases are found
found_cases = []

# Initialize plot
plt.figure(figsize=(12, 8))

# Iterate through all distributions
for entry in size_distribution_dict.values():
    date = entry["Date"]
    leg_index = entry["Leg_index"]
    x_data = np.array(common_bins)  # Bin centers (fixed)
    y_data = np.array(entry["Interpolated Values"])  # Observed droplet concentrations

    # Ensure x_data and y_data are the same length
    min_length = min(len(x_data), len(y_data))
    x_data = x_data[:min_length]
    y_data = y_data[:min_length]

    # Remove NaN and non-positive values from y_data while keeping corresponding x_data
    valid_indices = ~np.isnan(y_data) & (y_data > 0)
    x_data_filtered = x_data[valid_indices]
    y_data_filtered = y_data[valid_indices]

    # Skip if not enough valid data points
    if len(x_data_filtered) < 3:
        print(f"Skipping {date}, Leg {leg_index}: Not enough valid data points for fitting.")
        continue

    # Define initial parameter guesses
    initial_guess = [max(y_data_filtered), 2.0]  # (Initial N0, Initial D)

    try:
        # Fit the function to the data
        popt, _ = curve_fit(size_distribution, x_data_filtered, y_data_filtered, p0=initial_guess, maxfev=5000)

        # Store the fitted parameters
        fitted_params[f"{date}_Leg{leg_index}"] = {"N0": popt[0], "D": popt[1]}

        # Generate fitted curve
        fitted_curve = size_distribution(x_data_filtered, *popt)

        # Check if this distribution is a selected case
        is_selected = any(
            case["Date"] == date and case["Leg_index"] == leg_index
            for case in selected_cases
        )

        # Assign color: gray for all, specific colors for selected cases
        if is_selected:
            found_cases.append((date, leg_index))  # Track found cases
            case_index = next(
                (i for i, case in enumerate(selected_cases)
                 if case["Date"] == date and case["Leg_index"] == leg_index),
                None
            )
            if case_index is not None:
                color = ["blue", "orange", "green"][case_index]
                mass_value = selected_cases[case_index]["Mass"]  # Get mass value
                label = f"{date}, Mass={mass_value:.2f} µg/m³"
                plt.plot(x_data_filtered, fitted_curve, color=color, linewidth=2.5, label=label)
        else:
            plt.plot(x_data_filtered, fitted_curve, linestyle='-', color="gray", alpha=0.2)

    except RuntimeError:
        print(f"Fit did not converge for {date}, Leg {leg_index}")

# Check which selected cases were actually found
print("\nFound Cases:")
for case in found_cases:
    print(f"  {case[0]}, Leg {case[1]}")

# Check if any cases were missing
missing_cases = [case for case in selected_cases if (case["Date"], case["Leg_index"]) not in found_cases]
if missing_cases:
    print("\nMissing Cases:")
    for case in missing_cases:
        print(f"  {case['Date']}, Leg {case['Leg_index']} (not found in fitted data)")

# Formatting
plt.yscale("log")
plt.xlabel("Bin diameter (µm)", fontsize=14, fontweight="bold")
plt.ylabel("Droplet concentration (/cm³/µm)", fontsize=14, fontweight="bold")
plt.title("Below Cloud Base January - June 2022", fontsize=16, fontweight="bold")
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
# Show only the legend for selected cases
plt.legend()
plt.tight_layout()
plt.show()

# %%
import pickle

Jun8_hi_case = "/home/disk/eos4/kathem24/activate/data/CAS/Jason's Model/1_15_hi.pickle"
Jun8_lo_case = "/home/disk/eos4/kathem24/activate/data/CAS/Jason's Model/1_15_lo.pickle"
Jun14_hi_case = "/home/disk/eos4/kathem24/activate/data/CAS/Jason's Model/3_12_hi.pickle"
Jun14_lo_case = "/home/disk/eos4/kathem24/activate/data/CAS/Jason's Model/3_12_lo.pickle"
Jan__hi_case = "/home/disk/eos4/kathem24/activate/data/CAS/Jason's Model/02_15_hi.pickle"
Jan__lo_case = "/home/disk/eos4/kathem24/activate/data/CAS/Jason's Model/02_15_lo.pickle"
No_GCCN_hi = "/home/disk/eos4/kathem24/activate/data/CAS/Jason's Model/no_gccn.pickle"
No_GCCN_lo = "/home/disk/eos4/kathem24/activate/data/CAS/Jason's Model/no_gccn_lo.pickle"

# Load the pickle files
with open(Jun8_hi_case, "rb") as f:
    Jun8_hi_case_data = pickle.load(f)

with open(Jun8_lo_case, "rb") as f:
    Jun8_lo_case_data = pickle.load(f)

with open(Jun14_hi_case, "rb") as f:
    Jun14_hi_case_data = pickle.load(f)

with open(Jun14_lo_case, "rb") as f:
    Jun14_lo_case_data = pickle.load(f)

with open(Jan__hi_case, "rb") as f:
    Jan__hi_case_data = pickle.load(f)
with open(Jan__lo_case, "rb") as f:
    Jan__lo_case_data = pickle.load(f)
with open(No_GCCN_lo, "rb") as f:
    No_GCCN_lo_data = pickle.load(f)    
with open(No_GCCN_hi, "rb") as f:
    No_GCCN_hi_data = pickle.load(f)


print("Jun8_hi_case Data Type:", type(Jun8_hi_case_data))
print("Jun8_lo_case Data Type:", type(Jun8_lo_case_data))
print("Jun14_hi_case Data Type:", type(Jun14_hi_case_data))
print("No_GCCN_hi Data Type:", type(No_GCCN_hi_data))
print("Jun14_lo_case Data Type:", type(Jun14_lo_case_data))
print("Jan__hi_case Data Type:", type(Jan__hi_case_data))
print("Jan__lo_case Data Type:", type(Jan__lo_case_data))
print("No_GCCN_lo Data Type:", type(No_GCCN_lo_data))

# If the data is a dictionary, print the keys
if isinstance(Jun8_hi_case_data, dict):
    print("Jun8_hi_case_data Data Keys:", Jun8_hi_case_data.keys())

if isinstance(Jun8_lo_case_data, dict):
    print("Jun8_lo_case_data Data Keys:", Jun8_lo_case_data.keys())

if isinstance(Jun14_hi_case_data, dict):
    print("Jun14_hi_case_data Data Keys:", Jun14_hi_case_data.keys())
if isinstance(No_GCCN_hi_data, dict):
    print("No_GCCN_hi_data Data Keys:", No_GCCN_hi_data.keys())
if isinstance(Jun14_lo_case_data, dict):
    print("Jun14_lo_case_data Data Keys:", Jun14_lo_case_data.keys())
if isinstance(Jan__hi_case_data, dict):
    print("Jan__hi_case_data Data Keys:", Jan__hi_case_data.keys())
if isinstance(Jan__lo_case_data, dict):
    print("Jan__lo_case_data Data Keys:", Jan__lo_case_data.keys())
if isinstance(No_GCCN_lo_data, dict):
    print("No_GCCN_lo_data Data Keys:", No_GCCN_lo_data.keys())
# %%
# Check the structure of the second dimension
june8_hi_rain = Jun8_hi_case_data[0]['surface precipitation']

# Print the type and shape
print("june8_hi_rain Type:", type(june8_hi_rain))
print("june8_hi_rain Shape:", june8_hi_rain.shape)

# If it's an array with another dictionary structure inside:
if isinstance(june8_hi_rain, dict):
    print("Keys inside 'surface precipitation':", june8_hi_rain.keys())

# If it's an array, let's check its dimensions
elif isinstance(june8_hi_rain, np.ndarray):
    print("Rain Data Dimensions:", len(june8_hi_rain.shape))
#%%

# Choose one file to inspect
file_to_check = Jun8_hi_case  # Change this to inspect others
with open(file_to_check, "rb") as f:
    data = pickle.load(f)

# Print the top-level keys
if isinstance(data, dict):
    print(f"Top-level keys in {file_to_check}: {data.keys()}")

    # Now, check if "wet" or "dry" size distributions exist
    for key in data.keys():
        if "dry" in key.lower() or "wet" in key.lower():
            print(f"Potential match found: {key} -> Type: {type(data[key])}")

            # If it's an array or list, print its shape or length
            if isinstance(data[key], np.ndarray):
                print(f"Shape of {key}: {data[key].shape}")
            elif isinstance(data[key], list):
                print(f"Length of {key}: {len(data[key])}")

# If it's a list, check the structure of the first entry
elif isinstance(data, list):
    print(f"First entry type in {file_to_check}: {type(data[0])}")

    if isinstance(data[0], dict):
        print(f"Keys in first dictionary entry: {data[0].keys()}")

# %%
# Extract rain rate time series for all 30 simulations in each dataset
june8_hi_rain_rates = [ Jun8_hi_case_data[i]['surface precipitation'] for i in range(len( Jun8_hi_case_data))]
june8_lo_rain_rates = [ Jun8_lo_case_data[i]['surface precipitation'] for i in range(len( Jun8_lo_case_data))]  
june14_hi_rain_rates = [ Jun14_hi_case_data[i]['surface precipitation'] for i in range(len( Jun14_hi_case_data))]   
june14_lo_rain_rates = [ Jun14_lo_case_data[i]['surface precipitation'] for i in range(len( Jun14_lo_case_data))]   
jan_hi_rain_rates = [ Jan__hi_case_data[i]['surface precipitation'] for i in range(len( Jan__hi_case_data))]    
jan_lo_rain_rates = [ Jan__lo_case_data[i]['surface precipitation'] for i in range(len( Jan__lo_case_data))]    
no_gccn_hi_rain_rates = [ No_GCCN_hi_data[i]['surface precipitation'] for i in range(len( No_GCCN_hi_data))]    
no_gccn_lo_rain_rates = [ No_GCCN_lo_data[i]['surface precipitation'] for i in range(len( No_GCCN_lo_data))]
# %%
# # Compute the mean rain rate for each of the 30 simulations
june8_hi_mean_rain = [np.mean(sim) for sim in june8_hi_rain_rates]
june8_lo_mean_rain = [np.mean(sim) for sim in june8_lo_rain_rates]  
june14_hi_mean_rain = [np.mean(sim) for sim in june14_hi_rain_rates]    
june14_lo_mean_rain = [np.mean(sim) for sim in june14_lo_rain_rates]
jan_hi_mean_rain = [np.mean(sim) for sim in jan_hi_rain_rates]
jan_lo_mean_rain = [np.mean(sim) for sim in jan_lo_rain_rates]
no_gccn_hi_mean_rain = [np.mean(sim) for sim in no_gccn_hi_rain_rates]
no_gccn_lo_mean_rain = [np.mean(sim) for sim in no_gccn_lo_rain_rates]  
#%% 
time_hours = np.array(Jun8_hi_case_data[0]['t']) / 3600  # Convert time steps to hours

plt.figure(figsize=(10,6))

# Plot all 30 simulations for March
for i in range(len(june8_hi_rain_rates)):
    plt.plot(time_hours, june8_hi_rain_rates[i], color='orange', alpha=0.3)
plt.plot(time_hours, np.mean(june8_hi_rain_rates, axis=0), color='orange', label='June 8 High LWP Mean Rain Rate', linewidth=2)

for i in range(len(june8_lo_rain_rates)):
    plt.plot(time_hours, june8_lo_rain_rates[i], color='blue', alpha=0.3)
plt.plot(time_hours, np.mean(june8_lo_rain_rates, axis=0), color='blue', label='June 8 Low LWP Mean Rain Rate', linewidth=2)    

for i in range(len(june14_hi_rain_rates)):
    plt.plot(time_hours, june14_hi_rain_rates[i], color='green', alpha=0.3)
plt.plot(time_hours, np.mean(june14_hi_rain_rates, axis=0), color='green', label='June 14 High LWP Mean Rain Rate', linewidth=2)

for i in range(len(june14_lo_rain_rates)):
    plt.plot(time_hours, june14_lo_rain_rates[i], color='red', alpha=0.3)
plt.plot(time_hours, np.mean(june14_lo_rain_rates, axis=0), color='red', label='June 14 Low LWP Mean Rain Rate', linewidth=2)

for i in range(len(jan_hi_rain_rates)):
    plt.plot(time_hours, jan_hi_rain_rates[i], color='purple', alpha=0.3)
plt.plot(time_hours, np.mean(jan_hi_rain_rates, axis=0), color='purple', label='Jan High LWP Mean Rain Rate', linewidth=2)

for i in range(len(jan_lo_rain_rates)):
    plt.plot(time_hours, jan_lo_rain_rates[i], color='black', alpha=0.3)
plt.plot(time_hours, np.mean(jan_lo_rain_rates, axis=0), color='black', label='Jan Low LWP Mean Rain Rate', linewidth=2)

for i in range(len(no_gccn_hi_rain_rates)):
    plt.plot(time_hours, no_gccn_hi_rain_rates[i], color='brown', alpha=0.3)
plt.plot(time_hours, np.mean(no_gccn_hi_rain_rates, axis=0), color='brown', label='No GCCN High LWP Mean Rain Rate', linewidth=2)   

for i in range(len(no_gccn_lo_rain_rates)):
    plt.plot(time_hours, no_gccn_lo_rain_rates[i], color='pink', alpha=0.3)
plt.plot(time_hours, np.mean(no_gccn_lo_rain_rates, axis=0), color='pink', label='No GCCN Low LWP Mean Rain Rate', linewidth=2) 
# Update x-axis to reflect hours instead of raw time steps
plt.xlabel("Time (Hours)", fontsize=14)
plt.ylabel("Rain Rate (mm/hr)", fontsize=14)
plt.title("Time Series of Rain Rate for Each Case", fontsize=16)
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.5)

plt.show()
# %%
dt = 10 
# Compute total accumulated rain for each simulation
jun8_hi_totals = [np.sum(sim) *dt for sim in june8_hi_rain_rates]
jun8_lo_totals = [np.sum(sim) *dt for sim in june8_lo_rain_rates]
jun14_hi_totals = [np.sum(sim) *dt for sim in june14_hi_rain_rates]
jun14_lo_totals = [np.sum(sim) *dt for sim in june14_lo_rain_rates]
jan_hi_totals = [np.sum(sim) *dt for sim in jan_hi_rain_rates]
jan_lo_totals = [np.sum(sim) *dt for sim in jan_lo_rain_rates]
no_gccn_hi_totals = [np.sum(sim) *dt for sim in no_gccn_hi_rain_rates]
no_gccn_lo_totals = [np.sum(sim) *dt for sim in no_gccn_lo_rain_rates]
# Boxplot using total rain per simulation
plt.figure(figsize=(8,6))


# Separate data for Low LWP and High LWP
rain_rate_data_low = [no_gccn_lo_totals, jan_lo_totals, jun8_lo_totals, jun14_lo_totals]
rain_rate_data_high = [no_gccn_hi_totals, jan_hi_totals, jun8_hi_totals, jun14_hi_totals]

labels_low = ["No GCCN Low LWP", "January Low LWP", "June 8 Low LWP", "June 14 Low LWP"]
labels_high = ["No GCCN High LWP", "January High LWP", "June 8 High LWP", "June 14 High LWP"]

# ** Plot for Low LWP **
plt.figure(figsize=(8, 6))
plt.boxplot(rain_rate_data_low, labels=labels_low, patch_artist=True,
            boxprops=dict(facecolor="orange"), medianprops=dict(color="black", linewidth=2))

# Formatting
plt.ylabel("Total Rainfall (mm/hr)", fontsize=14, fontweight='bold')
plt.title("Distribution of Total Rainfall - Low LWP Cases", fontsize=16, fontweight='bold')
plt.yscale("log")  
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.xticks(fontsize=12, fontweight='bold', rotation=15)
plt.yticks(fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# ** Plot for High LWP **
plt.figure(figsize=(8, 6))
plt.boxplot(rain_rate_data_high, labels=labels_high, patch_artist=True,
            boxprops=dict(facecolor="blue"), medianprops=dict(color="black", linewidth=2))

# Formatting
plt.ylabel("Total Rainfall (mm/hr)", fontsize=14, fontweight='bold')
plt.title("Total Accumulated Rainfall per Simulation - High LWP Cases", fontsize=16, fontweight='bold')
plt.yscale("log")  
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.xticks(fontsize=12, fontweight='bold', rotation=15)
plt.yticks(fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# %%
# Compute total accumulated rain for each simulation

jun8_hi_totals = [np.sum(sim) *dt for sim in june8_hi_rain_rates]
jun8_lo_totals = [np.sum(sim) *dt for sim in june8_lo_rain_rates]
jun14_hi_totals = [np.sum(sim) *dt for sim in june14_hi_rain_rates]
jun14_lo_totals = [np.sum(sim) *dt for sim in june14_lo_rain_rates]
jan_hi_totals = [np.sum(sim) *dt for sim in jan_hi_rain_rates]
jan_lo_totals = [np.sum(sim) *dt for sim in jan_lo_rain_rates]
no_gccn_hi_totals = [np.sum(sim) *dt for sim in no_gccn_hi_rain_rates]
no_gccn_lo_totals = [np.sum(sim) *dt for sim in no_gccn_lo_rain_rates]
  

# Boxplot using total rain per simulation
plt.figure(figsize=(8,6))

rain_rate_data_low = [no_gccn_lo_totals, jan_lo_totals, jun8_lo_totals, jun14_lo_totals]
rain_rate_data_high = [no_gccn_hi_totals, jan_hi_totals, jun8_hi_totals, jun14_hi_totals]

# Corresponding mass values (now including No GCCN case explicitly)
low_mass_values = [0, 17.47, 65.76, 57.82]  # No GCCN, January, June8, June14
high_mass_values = [0, 17.47, 65.76, 57.82]  # No GCCN, January, June8, June14
# Convert mass values to string for x-axis labels
low_mass_labels = [f"{m:.0e}" for m in low_mass_values] 
high_mass_labels = [f"{m:.0e}" for m in high_mass_values] 

# Create boxplot with total rain per simulation
plt.boxplot(rain_rate_data_low, patch_artist=True, 
            boxprops=dict(facecolor="orange"), medianprops=dict(color="black", linewidth=2))

# Set x-axis labels to mass values
plt.xticks(ticks=[1,2,3,4], labels=low_mass_labels)  

# Formatting
plt.ylabel("Total Rainfall (mm/hr)", fontsize=14, fontweight='bold')
plt.title("Total Accumulated Rainfall per Simulation - Low LWP Cases", fontsize=16, fontweight='bold')
plt.yscale("log")  
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.xticks(fontsize=12, fontweight='bold', rotation=15)
plt.yticks(fontsize=12, fontweight='bold')
plt.xlabel("Mass (µg/m³)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.boxplot(rain_rate_data_high, labels=labels_high, patch_artist=True,
            boxprops=dict(facecolor="blue"), medianprops=dict(color="black", linewidth=2))
plt.xticks(ticks=[1,2,3,4], labels=high_mass_labels) 
# Formatting
plt.ylabel("Total Rainfall (mm/hr)", fontsize=14, fontweight='bold')
plt.title("Total Accumulated Rainfall per Simulation - High LWP Cases", fontsize=16, fontweight='bold')
plt.yscale("log")  
plt.xlabel("Mass (µg/m³)", fontsize=14, fontweight='bold')
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.xticks(fontsize=12, fontweight='bold', rotation=15)
plt.yticks(fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
# %%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

dt = 10  # Time step for integration

# Compute total rainfall for each case
jun8_hi_totals = [np.sum(sim) * dt for sim in june8_hi_rain_rates]
jun8_lo_totals = [np.sum(sim) * dt for sim in june8_lo_rain_rates]
jun14_hi_totals = [np.sum(sim) * dt for sim in june14_hi_rain_rates]
jun14_lo_totals = [np.sum(sim) * dt for sim in june14_lo_rain_rates]
jan_hi_totals = [np.sum(sim) * dt for sim in jan_hi_rain_rates]
jan_lo_totals = [np.sum(sim) * dt for sim in jan_lo_rain_rates]
no_gccn_hi_totals = [np.sum(sim) * dt for sim in no_gccn_hi_rain_rates]
no_gccn_lo_totals = [np.sum(sim) * dt for sim in no_gccn_lo_rain_rates]

# Organize data
rain_rate_data_low = [no_gccn_lo_totals, jan_lo_totals, jun8_lo_totals, jun14_lo_totals]
rain_rate_data_high = [no_gccn_hi_totals, jan_hi_totals, jun8_hi_totals, jun14_hi_totals]

# Mass values corresponding to each case
low_mass_values = [0, 17.47, 65.76, 57.82]  # No GCCN, January, June 8, June 14
high_mass_values = [0, 17.47, 65.76, 57.82]

# Flatten data for seaborn violin plot format
low_rainfall_data = [(mass, value) for mass, values in zip(low_mass_values, rain_rate_data_low) for value in values]
high_rainfall_data = [(mass, value) for mass, values in zip(high_mass_values, rain_rate_data_high) for value in values]

# Convert to DataFrame
import pandas as pd

df_low = pd.DataFrame(low_rainfall_data, columns=["Mass (µg/m³)", "Total Rainfall (mm/hr)"])
df_high = pd.DataFrame(high_rainfall_data, columns=["Mass (µg/m³)", "Total Rainfall (mm/hr)"])

# **Plot for Low LWP Cases**
plt.figure(figsize=(8, 6))
sns.violinplot(x="Mass (µg/m³)", y="Total Rainfall (mm/hr)", data=df_low, scale="width", inner="quartile", palette="Oranges")

# Formatting
plt.xlabel("Mass (µg/m³)", fontsize=14, fontweight='bold')
plt.ylabel("Total Rainfall (mm/hr)", fontsize=14, fontweight='bold')
plt.title("Distribution of Total Rainfall - Low LWP Cases", fontsize=16, fontweight='bold')
plt.yscale("log")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.xticks(fontsize=12, fontweight='bold', rotation=15)
plt.yticks(fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# **Plot for High LWP Cases**
plt.figure(figsize=(8, 6))
sns.violinplot(x="Mass (µg/m³)", y="Total Rainfall (mm/hr)", data=df_high, scale="width", inner="quartile", palette="Blues")

# Formatting
plt.xlabel("Mass (µg/m³)", fontsize=14, fontweight='bold')
plt.ylabel("Total Rainfall (mm/hr)", fontsize=14, fontweight='bold')
plt.title("Total Accumulated Rainfall per Simulation - High LWP Cases", fontsize=16, fontweight='bold')
plt.yscale("log")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.xticks(fontsize=12, fontweight='bold', rotation=15)
plt.yticks(fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# %%


import numpy as np
import matplotlib.pyplot as plt

# Access first dictionary inside the list
first_entry = Jun8_hi_case_data[0]  

# Extract dry and wet size distributions
dry_spectrum = first_entry.get("dry spectrum", None)
wet_spectrum = first_entry.get("wet spectrum", None)

# Ensure the third dimension is not empty
if dry_spectrum is not None and dry_spectrum.shape[-1] > 0:
    dry_spectrum = np.sum(dry_spectrum, axis=-1)  # Sum over the last dimension

if wet_spectrum is not None and wet_spectrum.shape[-1] > 0:
    wet_spectrum = np.sum(wet_spectrum, axis=-1)  # Sum over the last dimension

# Check final shapes
print("Processed Dry Spectrum Shape:", np.shape(dry_spectrum))
print("Processed Wet Spectrum Shape:", np.shape(wet_spectrum))

# Plot if valid
plt.figure(figsize=(8, 6))

if dry_spectrum is not None and dry_spectrum.shape[-1] > 0:
    plt.plot(dry_spectrum, label="Dry Spectrum", color="red")

if wet_spectrum is not None and wet_spectrum.shape[-1] > 0:
    plt.plot(wet_spectrum, label="Wet Spectrum", color="blue")

plt.xlabel("Size Bin")
plt.ylabel("Concentration")
plt.title("Wet vs Dry Size Distributions Before Updraft")
plt.legend()
plt.yscale("log")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.show()


# %%
# Extract first two dimensions only
dry_spectrum_sample = np.array(Jun8_hi_case_data[0]["dry spectrum"])  # (26, 100)
wet_spectrum_sample = np.array(Jun8_hi_case_data[0]["wet spectrum"])  # (26, 100)

# Check if spectra contain data
if dry_spectrum_sample.size > 0 and not np.all(np.isnan(dry_spectrum_sample)):
    print("Dry Spectrum Min:", np.nanmin(dry_spectrum_sample), "Max:", np.nanmax(dry_spectrum_sample))
else:
    print("Dry Spectrum is completely empty or NaN.")

if wet_spectrum_sample.size > 0 and not np.all(np.isnan(wet_spectrum_sample)):
    print("Wet Spectrum Min:", np.nanmin(wet_spectrum_sample), "Max:", np.nanmax(wet_spectrum_sample))
else:
    print("Wet Spectrum is completely empty or NaN.")

# Confirm new shape
print("Processed Dry Spectrum Shape:", dry_spectrum_sample.shape)
print("Processed Wet Spectrum Shape:", wet_spectrum_sample.shape)

# %%
