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
# %%
plt.figure(figsize=(8, 6))
for entry in Y_BCB_calc:
    bin_means = np.array([entry.get(f'Bin{i}_Y_mean', np.nan) for i in range(12, 30)], dtype=float)  
    valid_indices = ~np.isnan(bin_means)  
    bin_centers_valid = np.array(bin_center)[valid_indices]
    bin_means_valid = bin_means[valid_indices]

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
#average ambient 
sum_bin_means_CAS = np.zeros(len(bin_center))
count_bin_means_CAS= np.zeros(len(bin_center))
# Loop through each entry and accumulate valid bin means
for entry in Y_BCB_calc:
    bin_means = np.array([entry.get(f'Bin{i}_Y_mean', np.nan) for i in range(12, 30)], dtype=float)
    valid_indices = (bin_means > 0) & ~np.isnan(bin_means)

    # Accumulate sums and counts
    sum_bin_means_CAS[valid_indices] += bin_means[valid_indices]
    count_bin_means_CAS[valid_indices] += 1

average_bin_means_CAS = np.divide(sum_bin_means_CAS, count_bin_means_CAS, where=count_bin_means_CAS > 0)

plt.figure(figsize=(8, 6))
plt.plot(bin_center, average_bin_means_CAS, color='red', linewidth=2, label='Average CAS Size Distribution')

plt.xlabel("Deliquesced Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"CAS Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("CAS Average Ambient Below Cloud Base Size Distribution\n January - June 2022", fontsize=14, fontweight="bold")
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
#average distribution


# Initialize arrays for averaging
sum_bin_means = np.zeros(len(bin_center))
count_bin_means = np.zeros(len(bin_center))

# Iterate through all size distributions
for entry in Y_BCB_calc:
    bin_means = np.array([entry.get(f'Bin{i}_Y_mean', np.nan) for i in range(12, 30)], dtype=float)

    # Mask: Remove NaNs and zero values
    valid_indices = (bin_means > 0) & ~np.isnan(bin_means)

    # Accumulate sum and count for averaging
    sum_bin_means[valid_indices] += bin_means[valid_indices]
    count_bin_means[valid_indices] += 1

# Compute average size distribution (avoid division by zero)
average_bin_means = np.divide(sum_bin_means, count_bin_means, where=count_bin_means > 0)

# Plot the averaged size distribution
plt.figure(figsize=(8, 6))
plt.plot(bin_center, average_bin_means, color='red', linewidth=2, label='Average Size Distribution')

# Labels and formatting
plt.xlabel("Deliquesced Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"CAS Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xlim(0, 45)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("CAS Average Ambient Below Cloud Base Size Distribution\n January - June 2022", fontsize=14, fontweight="bold")

# Show plot
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
    bin_means = np.array([entry.get(f'Bin{i}_Y_mean', np.nan) for i in range(12, 30)], dtype=float)
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
plt.ylabel(r"CAS Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
# plt.ylim(10**-33, 10**1)
plt.ylim(10**-7, 10**1)
plt.xlim(0, 10)
plt.title(" CAS Below Cloud Base January - June 2022\n Exponential Fit Ambient Size Distributions (≤10 µm)", fontsize=14, fontweight="bold")
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

# Extract total number concentrations
total_Y_concentrations = [entry['Total_Y_Concentration_cm3'] for entry in total_concentration_cm3]

# Remove NaN values if any
total_Y_concentrations = [conc for conc in total_Y_concentrations if not np.isnan(conc)]

# Calculate mean total concentration
mean_total_concentration = np.mean(total_Y_concentrations)

# Print result
print(f"Mean Total Number Concentration: {mean_total_concentration:.2f} cm⁻³")

#%% 
#Recreating C-R 1a


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
plt.title("CAS Below Cloud Base January - June 2022", fontsize=16, fontweight='bold')

# ✅ Optional: Add reference lines if applicable
# plt.axhline(0.05, color='red', linestyle='--', label="Reference Min (0.05 cm⁻³)")
# plt.axhline(0.3, color='blue', linestyle='--', label="Reference Max (0.3 cm⁻³)")
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
ambient_fits_dict_10 = {(fit['Date'], fit['BCB_start'], fit['BCB_stop']): fit for fit in ambient_fits_10}

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
    if key in ambient_fits_dict_10:
        n0 = ambient_fits_dict_10[key]['Intercept_n0']  # Extract n0 from ambient_fits
        
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
ambient_n0_values = [fit['Intercept_n0'] for fit in ambient_fits_10 if not np.isnan(fit['Intercept_n0'])]
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
common_bins = np.linspace(2, 25, 35)  # Adjust bin range and count as needed

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
common_bins = np.linspace(2, 25, 35)  # Adjust bin range and count as needed

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
        plt.plot(filtered_bins, filtered_dN_dD_dry, color='red', alpha=0.2)

# Formatting and labels
plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.ylim(10**-7, 10**1)
plt.xlim(0, 40)
plt.title("Below Cloud Base January - June 2022\n Raw Dry Size Distributions", fontsize=14, fontweight="bold")
plt.show()
#%%
#average dry distribution 


# Define common bin centers for interpolation
common_bins = np.linspace(2, 25, 35)  # Adjust bin range and count as needed

# Initialize sum and count arrays for averaging
sum_interpolated_dN_dD_dry = np.zeros_like(common_bins, dtype=float)
count_interpolated_dN_dD_dry = np.zeros_like(common_bins, dtype=int)

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
    valid_interpolated_indices = ~np.isnan(interpolated_dN_dD_dry)

    # Accumulate sum and count for averaging
    sum_interpolated_dN_dD_dry[valid_interpolated_indices] += interpolated_dN_dD_dry[valid_interpolated_indices]
    count_interpolated_dN_dD_dry[valid_interpolated_indices] += 1

# Compute average dry size distribution (avoid division by zero)
average_dN_dD_dry = np.divide(sum_interpolated_dN_dD_dry, count_interpolated_dN_dD_dry, where=count_interpolated_dN_dD_dry > 0)

# Plot the averaged dry size distribution
plt.figure(figsize=(8, 6))
plt.plot(common_bins, average_dN_dD_dry, color='red', linewidth=2, label='Average Dry Size Distribution')

# Formatting and labels
plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"CAS Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xlim(0, 45)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("CAS Average Below Cloud Base Dry Size Distribution\n January - June 2022", fontsize=14, fontweight="bold")
plt.legend()

# Show plot
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
plt.xlim(0,40)
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
common_bins = np.linspace(2, 25, 35)
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
#counting errors 


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Define constants
sample_area_cm2 = 0.0025  # CAS sample area in cm²
plane_speed_cm_s = 1.2e4  # Plane speed in cm/s (120 m/s)
sampling_time_s = 198  # Each leg is 3.3 minutes = 198 seconds

# Compute total sample volume per leg
sample_volume = sample_area_cm2 * plane_speed_cm_s * sampling_time_s  # cm³

# Define common bin centers for interpolation
common_bins = np.linspace(2, 25, 35)  # Adjust bin range as needed

plt.figure(figsize=(8, 6))

# Loop through each dry size distribution
for entry in filtered_master_BCB_ddry:
    ddry_values = np.array(entry['ddry'])  # Bin centers
    dN_dD_dry = np.array(entry['dN/dDdry'])  # Concentration (cm⁻³ µm⁻¹)

    # Remove NaN values before interpolation
    valid_indices = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry)
    if np.sum(valid_indices) < 2:
        continue  # Skip if not enough valid points for interpolation

    # Interpolate onto common bins
    interp_func = interp1d(ddry_values[valid_indices], dN_dD_dry[valid_indices], 
                           kind='linear', bounds_error=False, fill_value=np.nan)
    interpolated_dN_dD_dry = interp_func(common_bins)

    # Mask: Remove NaNs and zero values
    valid_interpolated_indices = (interpolated_dN_dD_dry > 0) & ~np.isnan(interpolated_dN_dD_dry)
    filtered_bins = common_bins[valid_interpolated_indices]
    filtered_dN_dD_dry = interpolated_dN_dD_dry[valid_interpolated_indices]

    # Compute expected counts per bin
    N_counts = filtered_dN_dD_dry * sample_volume  # N = concentration * sample volume

    # Avoid divide-by-zero errors
    N_counts[N_counts <= 0] = np.nan  

    # Compute relative counting error using the advisor's equation
    rel_error = (np.sqrt(N_counts))/ N_counts  # Correct equation

    # Plot relative error vs. bin size
    plt.plot(filtered_bins, rel_error, marker="o", linestyle="--", alpha=0.3)

# Formatting
plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel("Relative Counting Error", fontsize=14, fontweight="bold")
plt.yscale("log")  # Log scale to highlight trends
plt.xlim(2, 25)
plt.title("CAS Counting Error", fontsize=14, fontweight="bold")

plt.show()

#%%


# Define constants
sample_area_cm2 = 0.0025  # CAS sample area in cm²
plane_speed_cm_s = 1.2e4  # Plane speed in cm/s (120 m/s)
sampling_time_s = 198  # Each leg is 3.3 minutes = 198 seconds

# Compute total sample volume per leg
sample_volume = sample_area_cm2 * plane_speed_cm_s * sampling_time_s  # cm³

# Define common bin centers
bin_centers = np.linspace(2, 25, 35)  # Adjust bin range as needed
all_relative_errors = []  # Store all errors for statistics

for entry in filtered_master_BCB_ddry:
    ddry_values = np.array(entry['ddry'])  # Bin centers
    dN_dD_dry = np.array(entry['dN/dDdry'])  # Concentration (cm⁻³ µm⁻¹)

    # Remove NaN values before interpolation
    valid_indices = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry)
    if np.sum(valid_indices) < 2:
        continue  # Skip if not enough valid points for interpolation

    # Interpolate onto common bins
    interp_func = interp1d(ddry_values[valid_indices], dN_dD_dry[valid_indices], 
                           kind='linear', bounds_error=False, fill_value=np.nan)
    interpolated_dN_dD_dry = interp_func(bin_centers)

    # Mask: Remove NaNs and zero values
    valid_interpolated_indices = (interpolated_dN_dD_dry > 0) & ~np.isnan(interpolated_dN_dD_dry)
    filtered_bins = bin_centers[valid_interpolated_indices]
    filtered_dN_dD_dry = interpolated_dN_dD_dry[valid_interpolated_indices]

    # Compute expected counts per bin
    N_counts = filtered_dN_dD_dry * sample_volume  # N = concentration * sample volume

    # Avoid divide-by-zero errors
    N_counts[N_counts <= 0] = np.nan  

    # Compute relative counting error
    rel_error = 1 / np.sqrt(N_counts)

    # Store errors for statistical analysis
    all_relative_errors.append(rel_error)

# Convert to DataFrame for easier statistics
relative_errors_df = pd.DataFrame(all_relative_errors, columns=bin_centers)

# Compute statistics across all legs
stats_summary = pd.DataFrame({
    'Bin Center (µm)': bin_centers,
    'Mean Relative Error': np.nanmean(relative_errors_df, axis=0),
    'Median Relative Error': np.nanmedian(relative_errors_df, axis=0),
    'Min Relative Error': np.nanmin(relative_errors_df, axis=0),
    'Max Relative Error': np.nanmax(relative_errors_df, axis=0),
})

# Display statistics
print(stats_summary)

# Optional: Plot mean relative error vs. bin diameter
plt.figure(figsize=(8, 6))
plt.plot(stats_summary['Bin Center (µm)'], stats_summary['Mean Relative Error'], marker='o', linestyle='-', label="Mean Error")
plt.plot(stats_summary['Bin Center (µm)'], stats_summary['Max Relative Error'], marker='s', linestyle='--', label="Max Error", alpha=0.5)
plt.plot(stats_summary['Bin Center (µm)'], stats_summary['Min Relative Error'], marker='^', linestyle='--', label="Min Error", alpha=0.5)

plt.xlabel("Dry Bin Centers Diameter (µm)", fontsize=14, fontweight="bold")
plt.ylabel("Relative Counting Error", fontsize=14, fontweight="bold")
plt.yscale("log")  # Log scale to highlight trends
plt.legend()
plt.title("Mean, Min, and Max Counting Errors Across Dry Size Bins", fontsize=14, fontweight="bold")

plt.show()
#%%


# Define constants
sample_area_cm2 = 0.0025  # CAS sample area in cm²
plane_speed_cm_s = 1.2e4  # Plane speed in cm/s (120 m/s)
sampling_time_s = 198  # Each leg is 3.3 minutes = 198 seconds

# Compute total sample volume per leg
sample_volume = sample_area_cm2 * plane_speed_cm_s * sampling_time_s  # cm³

# Define common bin centers
bin_centers = np.linspace(2, 25, 35)  # Ensure 35 bins
all_relative_errors = []  # Store all errors for statistics

for entry in filtered_master_BCB_ddry:
    ddry_values = np.array(entry['ddry'])  # Bin centers
    dN_dD_dry = np.array(entry['dN/dDdry'])  # Concentration (cm⁻³ µm⁻¹)

    # Remove NaN values before interpolation
    valid_indices = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry)
    if np.sum(valid_indices) < 2:
        continue  # Skip if not enough valid points for interpolation

    # Interpolate onto common bins
    interp_func = interp1d(ddry_values[valid_indices], dN_dD_dry[valid_indices], 
                           kind='linear', bounds_error=False, fill_value=np.nan)
    interpolated_dN_dD_dry = interp_func(bin_centers)

    # Compute expected counts per bin
    N_counts = interpolated_dN_dD_dry * sample_volume  # N = concentration * sample volume

    # Avoid divide-by-zero errors
    N_counts[N_counts <= 0] = np.nan  

    # Compute relative counting error
    rel_error = 1 / np.sqrt(N_counts)

    # Ensure that rel_error always has 35 values (same as bin_centers)
    if len(rel_error) != len(bin_centers):
        rel_error = np.pad(rel_error, (0, len(bin_centers) - len(rel_error)), constant_values=np.nan)

    # Store errors for statistical analysis
    all_relative_errors.append(rel_error)

# Convert to DataFrame for easier statistics
relative_errors_df = pd.DataFrame(all_relative_errors, columns=bin_centers)

# Compute statistics across all legs
stats_summary = pd.DataFrame({
    'Bin Center (µm)': bin_centers,
    'Mean Relative Error': np.nanmean(relative_errors_df, axis=0),
    'Median Relative Error': np.nanmedian(relative_errors_df, axis=0),
    'Min Relative Error': np.nanmin(relative_errors_df, axis=0),
    'Max Relative Error': np.nanmax(relative_errors_df, axis=0),
})

# Save statistics as a CSV file for further analysis
# stats_summary.to_csv("relative_counting_error_statistics.csv", index=False)

# Display statistics
print(stats_summary)

# Optional: Plot mean relative error vs. bin diameter
plt.figure(figsize=(8, 6))
plt.plot(stats_summary['Bin Center (µm)'], stats_summary['Mean Relative Error'], marker='o', linestyle='-', label="Mean Error")
plt.plot(stats_summary['Bin Center (µm)'], stats_summary['Max Relative Error'], marker='s', linestyle='--', label="Max Error", alpha=0.5)
plt.plot(stats_summary['Bin Center (µm)'], stats_summary['Min Relative Error'], marker='^', linestyle='--', label="Min Error", alpha=0.5)

plt.xlabel("Dry Bin Centers Diameter (µm)", fontsize=14, fontweight="bold")
plt.ylabel("Relative Counting Error", fontsize=14, fontweight="bold")
plt.yscale("log")  # Log scale to highlight trends
plt.legend()
plt.title("CAS Counting Errors Across Dry Size Bins", fontsize=14, fontweight="bold")

plt.show()
#%%

# Define constants
sample_area_cm2 = 0.0025  # CAS sample area in cm²
plane_speed_cm_s = 1.2e4  # Plane speed in cm/s (120 m/s)
sampling_time_s = 198  # Each leg is 3.3 minutes = 198 seconds

# Compute total sample volume per leg
sample_volume = sample_area_cm2 * plane_speed_cm_s * sampling_time_s  # cm³

# Define common bin centers
bin_centers = np.linspace(2, 25, 35)  # Ensure 35 bins
all_relative_errors = []  # Store all errors for statistics

for entry in filtered_master_BCB_ddry:
    ddry_values = np.array(entry['ddry'])  # Bin centers
    dN_dD_dry = np.array(entry['dN/dDdry'])  # Concentration (cm⁻³ µm⁻¹)

    # Remove NaN values before interpolation
    valid_indices = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry)
    if np.sum(valid_indices) < 2:
        continue  # Skip if not enough valid points for interpolation

    # Interpolate onto common bins
    interp_func = interp1d(ddry_values[valid_indices], dN_dD_dry[valid_indices], 
                           kind='linear', bounds_error=False, fill_value=np.nan)
    interpolated_dN_dD_dry = interp_func(bin_centers)

    # Compute expected counts per bin
    N_counts = interpolated_dN_dD_dry * sample_volume  # N = concentration * sample volume

    # Avoid divide-by-zero errors
    N_counts[N_counts <= 0] = np.nan  

    # Corrected relative error calculation based on advisor's formula
    rel_error = np.sqrt(N_counts) / N_counts  # Instead of 1/sqrt(N_counts)

    # Ensure that rel_error always has 35 values (same as bin_centers)
    if len(rel_error) != len(bin_centers):
        rel_error = np.pad(rel_error, (0, len(bin_centers) - len(rel_error)), constant_values=np.nan)

    # Store errors for statistical analysis
    all_relative_errors.append(rel_error)

# Convert to DataFrame for easier statistics
relative_errors_df = pd.DataFrame(all_relative_errors, columns=bin_centers)

# Compute statistics across all legs
stats_summary = pd.DataFrame({
    'Bin Center (µm)': bin_centers,
    'Mean Relative Error': np.nanmean(relative_errors_df, axis=0),
    'Median Relative Error': np.nanmedian(relative_errors_df, axis=0),
    'Min Relative Error': np.nanmin(relative_errors_df, axis=0),
    'Max Relative Error': np.nanmax(relative_errors_df, axis=0),
})

# Save statistics as a CSV file for further analysis
stats_summary.to_csv("corrected_relative_counting_error_statistics.csv", index=False)

# Display statistics
print(stats_summary)

# Optional: Plot mean relative error vs. bin diameter
plt.figure(figsize=(8, 6))
plt.plot(stats_summary['Bin Center (µm)'], stats_summary['Mean Relative Error'], marker='o', linestyle='-', label="Mean Error")
plt.plot(stats_summary['Bin Center (µm)'], stats_summary['Max Relative Error'], marker='s', linestyle='--', label="Max Error", alpha=0.5)
plt.plot(stats_summary['Bin Center (µm)'], stats_summary['Min Relative Error'], marker='^', linestyle='--', label="Min Error", alpha=0.5)

plt.xlabel("Dry Bin Centers Diameter (µm)", fontsize=14, fontweight="bold")
plt.ylabel("Corrected Relative Counting Error", fontsize=14, fontweight="bold")
plt.yscale("log")  # Log scale to highlight trends
plt.legend()
plt.title("Corrected Mean, Min, and Max Counting Errors Across Dry Size Bins", fontsize=14, fontweight="bold")

plt.show()

#%%
import pandas as pd

# Define cutoff index around 10 µm
cutoff_bin = 25  # Adjust if needed based on the actual bin index

# Extract relevant statistics to justify the cutoff
summary_table = stats_summary.iloc[:cutoff_bin+5][['Bin Center (µm)', 'Mean Relative Error', 'Max Relative Error', 'Min Relative Error']]

# Save the summary table as a CSV (optional)
summary_table.to_csv("key_counting_error_statistics.csv", index=False)

# Print summary table for review
print(summary_table)


#%%
#average fitted distribution



# # Define the exponential function
# def exponential(x, n0, D):
#     return n0 * np.exp(-x / D)

# Define common bin centers for interpolation
common_bins = np.linspace(2, 25, 35)  # Adjust bin range and count as needed

# # Initialize sum and count arrays for averaging
# sum_interpolated_dN_dD_dry = (common_bins, dtype=float)
# count_interpolated_dN_dD_dry = (common_bins, dtype=int)

# Loop through each dry size distribution and accumulate values
for entry in filtered_master_BCB_ddry:
    ddry_values = np.array(entry['ddry'])
    dN_dD_dry = np.array(entry['dN/dDdry'])

    # Ensure valid data points
    valid_indices = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry) & (dN_dD_dry > 0)
    if np.sum(valid_indices) < 2:  
        continue

    # Interpolation onto common bins
    interp_func = interp1d(ddry_values[valid_indices], dN_dD_dry[valid_indices], 
                           kind='linear', bounds_error=False, fill_value=np.nan)
    interpolated_dN_dD_dry = interp_func(common_bins)

    # Mask: Remove NaNs and zero values
    valid_interpolated_indices = (interpolated_dN_dD_dry > 0) & ~np.isnan(interpolated_dN_dD_dry)

    # Accumulate sum and count for averaging
    sum_interpolated_dN_dD_dry[valid_interpolated_indices] += interpolated_dN_dD_dry[valid_interpolated_indices]
    count_interpolated_dN_dD_dry[valid_interpolated_indices] += 1

# Compute the average dry size distribution
average_dN_dD_dry = np.divide(sum_interpolated_dN_dD_dry, count_interpolated_dN_dD_dry, where=count_interpolated_dN_dD_dry > 0)

# Fit an exponential function to the averaged size distribution
valid_fit_indices = ~np.isnan(average_dN_dD_dry) & (average_dN_dD_dry > 0)
fit_bins = common_bins[valid_fit_indices]
fit_values = average_dN_dD_dry[valid_fit_indices]

# try:
#     popt, _ = curve_fit(exponential, fit_bins, fit_values, p0=(1, 5), maxfev=5000)
#     n0_fit, D_fit = popt

#     # Generate exponential fit curve
#     x_fit = np.linspace(min(fit_bins), max(fit_bins), 100)
#     y_fit = exponential(x_fit, *popt)

# except RuntimeError:
#     print("Exponential fit could not be performed.")
#     n0_fit, D_fit = np.nan, np.nan
#     x_fit, y_fit = [], []

# Plot the averaged dry size distribution (Raw Data)
plt.figure(figsize=(8, 6))
plt.plot(common_bins, average_dN_dD_dry, color='red', linewidth=2, label='Average Dry Size Distribution')

# Plot the Exponential Fit
# plt.plot(x_fit, y_fit, 'b--', linewidth=2, label=f'≤10 µm Fit: n0={n0_fit:.2e}, D={D_fit:.2f} µm')

# Formatting
plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"CAS Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-7, 10**0)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("CAS Average Below Cloud Base Exponential Fitted Dry Size Distribution\n January - June 2022", fontsize=14, fontweight="bold")
plt.legend()

# Show plot
plt.show()

# print(f"Final Exponential Fit Parameters: n0 = {n0_fit:.2e}, D = {D_fit:.2f} µm")
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the exponential function
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

# Filter bins for fitting (only include x ≤ 10 μm)
valid_fit_indices = common_bins <= 10
x_fit = common_bins[valid_fit_indices]
y_fit = average_dN_dD_dry[valid_fit_indices]

# Remove NaNs or zeros to avoid fitting issues
valid_data_indices = ~np.isnan(y_fit) & (y_fit > 0)
x_fit = x_fit[valid_data_indices]
y_fit = y_fit[valid_data_indices]

# Perform curve fitting
try:
    popt, pcov = curve_fit(exponential, x_fit, y_fit, p0=(1e-2, 2))  # Initial guess: (n0=0.01, D=2 μm)
    n0_fit, D_fit = popt  # Extract fitted parameters
except RuntimeError:
    print("Exponential fit failed.")
    n0_fit, D_fit = None, None

# Plot the averaged dry size distribution
plt.figure(figsize=(8, 6))
plt.plot(common_bins, average_dN_dD_dry, color='black', linewidth=2, label='Average Dry Size Distribution')

# Overlay the exponential fit if successful
if n0_fit is not None and D_fit is not None:
    plt.plot(x_fit, exponential(x_fit, *popt), 'r--', linewidth=2, label=f'Exponential Fit: $N_0$={n0_fit:.2e}, $D$={D_fit:.2f} μm')

# Formatting and labels
plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"CAS Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("CAS Average Below Cloud Base Dry Size Distribution\n January - June 2022", fontsize=14, fontweight="bold")
plt.legend()

# Show plot
plt.show()

# Print the fit parameters
if n0_fit is not None and D_fit is not None:
    print(f"Fitted Parameters: N_0 = {n0_fit:.3e}, D = {D_fit:.3f} μm")



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
plt.ylabel(r"CAS Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-7, 10**1)
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
plt.ylabel(r"CAS Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=15, fontweight="bold")
plt.yscale("log")
plt.xlim()
plt.xlim(0,10)

plt.ylim(1e-7, 1e1)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("CAS Below Cloud Base January - June 2022\n Fitted Dry Size Distributions (≤10 µm)", fontsize=15, fontweight="bold")

plt.show()
print(f"Total successful dry exponential fits: {len([fit for fit in dry_exponential_fits if not np.isnan(fit['Dry_Intercept_n0'])])}")
#%%

#%%
#histogram comapring less than 10um and regular fit exponential 
# Extracting slopes for both fits
dry_slopes_10 = [fit['Dry_E_folding_D'] for fit in dry_exponential_fits_10 if not np.isnan(fit['Dry_E_folding_D'])] 
# Plotting histograms
#%%
dry_intercepts_10=[fit['Dry_Intercept_n0'] for fit in dry_exponential_fits_10 if not np.isnan(fit['Dry_Intercept_n0'])]
#%%
dry_slopes_10 = [fit['Dry_E_folding_D'] for fit in dry_exponential_fits_10 if not np.isnan(fit['Dry_E_folding_D'])] 

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
import matplotlib.lines as mlines
#%%

# ✅ Define common bin centers for interpolation
common_bins = np.linspace(2, 40, 35)

# ✅ Define the specific case to plot
# selected_date = "2022-06-10"
# selected_start = 51245.0
# selected_stop = 51433.0

selected_date = "2022-03-13"
selected_start = 50135.0
selected_stop = 50496.0
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
        # plt.plot(common_bins, interpolated_dry, color='darkred', alpha=0.9, linewidth=2, label="Dry")

    # ✅ Step 3: Formatting and Labels
    plt.xlabel("Bin Centers Diameter (μm)", fontsize=15, fontweight="bold")
    plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=15, fontweight="bold")
    plt.yscale("log")
    plt.ylim(10**-7, 10**1)
    plt.xlim(0, 41)
    plt.xticks(fontweight="bold", fontsize=14)
    plt.yticks(fontweight="bold", fontsize=14)
    plt.title(f"Ambient Size Distributions - {selected_date}\nStart: {selected_start} | Stop: {selected_stop}", fontsize=14, fontweight="bold")

    # ✅ Step 4: Make **legend lines thicker and darker**
    ambient_legend = mlines.Line2D([], [], color='darkblue', linewidth=4, label="Ambient (dN/dD)")  # Thick dark blue
    # dry_legend = mlines.Line2D([], [], color='darkred', linewidth=4, label="Dry (dN/dDdry)")  # Thick dark red
    # plt.legend(handles=[ambient_legend], fontsize=12, frameon=True)
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
        plt.plot(common_bins, interpolated_dry, color='red', alpha=0.9, linewidth=2, label="Dry Observed Data")

    # ✅ Step 3: Formatting and Labels
    plt.xlabel("Bin Centers Diameter (μm)", fontsize=15, fontweight="bold")
    plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=15, fontweight="bold")
    plt.yscale("log")
    plt.ylim(10**-7, 10**1)
    plt.xlim(0, 40)
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
#Scatterplot of ambient slope versus ambient intercept

ambient_slopes = []
ambient_intercepts = []

for key, entry in ambient_fits_dict_10.items():
    n0 = entry['Intercept_n0']  # Ambient intercept
    D = entry['E_folding_D']    # Ambient slope (e-folding diameter)

    # Store values
    ambient_intercepts.append(n0)
    ambient_slopes.append(D)

df_ambient = pd.DataFrame({
    'Ambient_Intercept_N0': ambient_intercepts,
    'Ambient_Slope_D': ambient_slopes
})

plt.figure(figsize=(8, 6))
plt.scatter(df_ambient['Ambient_Slope_D'], df_ambient['Ambient_Intercept_N0'], alpha=0.6, color='blue')
plt.xlabel('Slope (um)', fontsize=19, fontweight='bold')
plt.ylabel(r'Ambient Intercept (cm$^{-3}$ $\mu$m$^{-1}$)', fontsize=19, fontweight='bold')
plt.title('Ambient Below Cloud Base January - June 2022', fontsize=19, fontweight='bold')
plt.xscale('log')
plt.yscale('log')
plt.xlim(10**-0.3, 10**1.1)
plt.ylim(10**-1.5, 10**1.1)
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# %%
#Ambient density contours 
ambient_slopes = []
ambient_intercepts = []

for key, entry in ambient_fits_dict_10.items():
    n0 = entry['Intercept_n0']  
    D = entry['E_folding_D']    

    ambient_intercepts.append(n0)
    ambient_slopes.append(D)

df_ambient = pd.DataFrame({
    'Ambient_Intercept_N0': ambient_intercepts,
    'Ambient_Slope_D': ambient_slopes
})

df_ambient = df_ambient.dropna()
df_ambient = df_ambient[np.isfinite(df_ambient[['Ambient_Slope_D', 'Ambient_Intercept_N0']].values).all(axis=1)]

filtered_slope = df_ambient['Ambient_Slope_D']
filtered_ambient_intercept = df_ambient['Ambient_Intercept_N0']

plt.figure(figsize=(10, 8))
plt.scatter(filtered_slope, filtered_ambient_intercept, c='blue', s=80, alpha=0.7, label="Data Points")

kde = gaussian_kde(np.vstack([filtered_slope, filtered_ambient_intercept]))

xgrid = np.logspace(np.log10(filtered_slope.min()), np.log10(filtered_slope.max()), 100)
ygrid = np.logspace(np.log10(filtered_ambient_intercept.min()), np.log10(filtered_ambient_intercept.max()), 100)
X, Y = np.meshgrid(xgrid, ygrid)
Z = kde(np.vstack([X.ravel(), Y.ravel()]))
Z = Z.reshape(X.shape)

contour_levels = np.linspace(Z.min(), Z.max(), num=10)
plt.contour(X, Y, Z, levels=contour_levels, colors='red', alpha=0.75)

plt.xlabel('Slope (E-folding Diameter D)', fontsize=19, fontweight='bold')
plt.ylabel(r'Ambient Intercept (cm$^{-3}$ $\mu$m$^{-1}$)', fontsize=19, fontweight='bold')
plt.title('Ambient Below Cloud Base January - June 2022\nDensity Contours', fontsize=19, fontweight='bold')

plt.xscale('log')
plt.yscale('log')

plt.xlim(10**-0.3, 10**1.1)
plt.ylim(10**-1.5, 10**1.1)
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# %%
#Calculating hydrated mass

rho = 1000  # kg/m³

def calculate_mass(N0, D):
    N0_m4 = N0 * 10**6  # Convert cm⁻³ to m⁻³

    integrand = lambda d: np.exp(-d / D) * (d * 1e-6)**3  # Convert µm³ → m³
    mass_integral, _ = quad(integrand, 2, 10)  # Integrate from 2 µm to ∞

    return (np.pi / 6) * rho * N0_m4 * mass_integral  

ambient_mass_dict = {}

for key, entry in ambient_fits_dict.items():
    date, BCB_start, BCB_stop = key  # Extract date and time identifiers
    D_ambient = entry['E_folding_D']  # Ambient slope
    N0_ambient = entry['Intercept_n0']  # Ambient intercept

    try:
        mass = calculate_mass(N0_ambient, D_ambient) * 1e9  # Convert kg/m³ to µg/m³

        if date not in ambient_mass_dict:
            ambient_mass_dict[date] = []  
        
        ambient_mass_dict[date].append({
            'Date': date,
            'Slope (D)': D_ambient,
            'Hydrated Intercept (N0)': N0_ambient,
            'Mass (µg/m³)': mass
        })

    except (ValueError, TypeError) as e:
        print(f"Error at {date} | Start: {BCB_start}, Stop: {BCB_stop} - {e}")

ambient_mass_ug = [entry['Mass (µg/m³)'] for sublist in ambient_mass_dict.values() for entry in sublist]

x_min, x_max = 10**-0.1, 10**1.05
y_min, y_max = 10**-1.6, 10**0.95

xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200)
ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)
D_grid_extended, hydratedintercept_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)
mass_grid_extended = np.zeros_like(D_grid_extended)

for i in range(D_grid_extended.shape[0]):
    for j in range(D_grid_extended.shape[1]):
        mass_grid_extended[i, j] = calculate_mass(hydratedintercept_grid_extended[i, j], D_grid_extended[i, j]) * 1e9

mass_levels = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]

slope_data = np.array([entry['Slope (D)'] for sublist in ambient_mass_dict.values() for entry in sublist])
hydrated_intercept_data = np.array([entry['Hydrated Intercept (N0)'] for sublist in ambient_mass_dict.values() for entry in sublist])

min_length = min(len(slope_data), len(hydrated_intercept_data))
slope_data = slope_data[:min_length]
hydrated_intercept_data = hydrated_intercept_data[:min_length]

plt.figure(figsize=(10, 8))
plt.scatter(slope_data, hydrated_intercept_data, c='darkgreen', s=80, alpha=0.7, label="Hydrated Data Points")

contour_plot = plt.contour(D_grid_extended, hydratedintercept_grid_extended, mass_grid_extended, 
                           levels=mass_levels, colors='red', linestyles='solid', alpha=0.75)

plt.clabel(contour_plot, inline=True, fontsize=12, fmt='%d µg/m³', colors='black', inline_spacing=5)
for txt in contour_plot.labelTexts:
    txt.set_fontweight('bold')
    txt.set_rotation(30)

plt.xlabel(r'Ambient Slope ($\mu$m)', fontsize=19, fontweight='bold')
plt.ylabel(r'Ambient Intercept (cm$^{-3}$ $\mu$m$^{-1}$)', fontsize=19, fontweight='bold')
plt.title('CAS Below Cloud Base January - June 2022\nContours of Hydrated Mass', fontsize=19, fontweight='bold')
plt.xscale('log')
plt.yscale('log')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
#%%%


hydrated_mass_values_ug = np.array([entry['Mass (µg/m³)'] for entries in ambient_mass_dict.values() for entry in entries])

# Count NaN values
num_nans = np.sum(np.isnan(hydrated_mass_values_ug))
print(f"Number of NaN values in hydrated mass data: {num_nans}")

# %%
#Histogram of hydrated mass
hydrated_mass_values_ug = [entry['Mass (µg/m³)'] for entries in ambient_mass_dict.values() for entry in entries]

bins = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 
                 16384, 32768, 65536, 131072])  

plt.figure(figsize=(8, 6))
plt.hist(hydrated_mass_values_ug, bins=bins, color='purple', alpha=0.7, edgecolor='black', density=False)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Hydrated Mass ($\mu$g m$^{-3}$)', fontsize=16, fontweight='bold')
plt.ylabel('Frequency of Flight Legs', fontsize=16, fontweight='bold')
plt.title('CAS Below Cloud Base January - June 2022\n Hydrated Mass', fontsize=18, fontweight='bold')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()
#%%
mean_hydrated_mass = np.nanmean(hydrated_mass_values_ug)
median_hydrated_mass = np.nanmedian(hydrated_mass_values_ug)

print(f"Mean Hydrated Mass: {mean_hydrated_mass:.2f} µg/m³")
print(f"Median Hydrated Mass: {median_hydrated_mass:.2f} µg/m³")

# %%
dry_slopes = []
dry_intercepts = []

for entry in dry_exponential_fits_10:
    n0 = entry['Dry_Intercept_n0']  # Dry intercept
    D = entry['Dry_E_folding_D']    # Dry slope (e-folding diameter)

    dry_intercepts.append(n0)
    dry_slopes.append(D)

df_dry = pd.DataFrame({
    'Dry_Intercept_N0': dry_intercepts,
    'Dry_Slope_D': dry_slopes
})

plt.figure(figsize=(10, 6))
plt.scatter(df_dry['Dry_Slope_D'], df_dry['Dry_Intercept_N0'], alpha=0.6, color='black')

plt.xlabel('Dry Slope (um)', fontsize=14, fontweight='bold')
plt.ylabel(r'Dry Intercept (cm$^{-3}$ $\mu$m$^{-1}$)', fontsize=14, fontweight='bold')
plt.title('Below Cloud Base January - June 2022\n Dry', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.xlim(10**-0.7, 10**1.1) 
plt.ylim(10**-1.3, 10**2)
plt.show()
# %%
plt.figure(figsize=(10, 6))

plt.scatter(df_ambient['Ambient_Slope_D'], df_ambient['Ambient_Intercept_N0'], 
            alpha=0.6, color='blue', label='Ambient')

# Plot dry data in red
plt.scatter(df_dry['Dry_Slope_D'], df_dry['Dry_Intercept_N0'], 
            alpha=0.6, color='red', label='Dry')

plt.xlabel('Slope (um)', fontsize=14, fontweight='bold')
plt.ylabel(r'Intercept (cm$^{-3}$ $\mu$m$^{-1}$)', fontsize=14, fontweight='bold')
plt.title('Comparison of Ambient and Dry', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.xlim(10**-0.6, 10**1.1)  
plt.ylim(10**-1.5, 10**2.1)
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.legend()
plt.show()

# %%
#Dry mass 10um 

rho_salt = 2200

# Function to calculate dry mass using exponential fit parameters
def calculate_mass(N0, D):
    """Compute dry mass using the exponential fit (N0, D)."""
    N0_m4 = N0 * 10**6  # Convert cm⁻³µm⁻¹ to m⁻⁴

    # Mass integral over size distribution
    integrand = lambda d: np.exp(-d / D) * (d * 1e-6)**3  # Convert µm³ → m³
    # mass_integral, _ = quad(integrand, 2, np.inf)  # Integrate from 2µm to ∞
    mass_integral, _ = quad(integrand, 2, 10)  # Integrate from 2µm to 10µm
    # Compute mass with the full equation
    return (np.pi / 6) * rho_salt * N0_m4 * mass_integral

# Define log-spaced grid for slope (D) and dry intercept (N₀)
# x_min, x_max = 10**-0.1, 10**1.05  
# y_min, y_max = 10**-1.7, 10**0.8  
x_min, x_max = 10**-0.4, 10**1  
y_min, y_max = 10**-1.6, 10**1.3 

xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200)  # Dry slope grid
ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)  # Dry intercept grid

D_grid_extended, dryintercept_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)

# Compute dry mass grid
mass_grid_extended = np.zeros_like(D_grid_extended)
for i in range(D_grid_extended.shape[0]):
    for j in range(D_grid_extended.shape[1]):
        mass_grid_extended[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j]) * 1e9  # Convert kg/m³ to µg/m³

# Extract dry slope and dry intercept values from `dry_exponential_fits`
dry_slopes = []
dry_intercepts = []

for entry in dry_exponential_fits_10:
    n0 = entry['Dry_Intercept_n0']  # Dry intercept
    D = entry['Dry_E_folding_D']    # Dry slope (e-folding diameter)

    dry_intercepts.append(n0)
    dry_slopes.append(D)

# Convert to NumPy arrays for plotting
dry_slopes = np.array(dry_slopes)
dry_intercepts = np.array(dry_intercepts)

# Define mass contour levels
mass_levels = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]

# Create scatter plot with mass contours
plt.figure(figsize=(10, 8))
plt.scatter(dry_slopes, dry_intercepts, c='blue', s=80, alpha=0.7, label="Dry Data Points")

# Overlay mass contours
contour_plot = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_extended, 
                           levels=mass_levels, colors='red', alpha=0.75, linewidths=1.5)

# Add contour labels
fmt = {level: f'{int(level)} µg/m³' for level in mass_levels}
plt.clabel(contour_plot, inline=True, fontsize=13, fmt=fmt, colors='black', inline_spacing=5)
for txt in contour_plot.labelTexts:
    txt.set_fontweight('bold')
    txt.set_rotation(45)  

# Formatting and labels
plt.xlabel(r'Dry Slope ($\mu$m)', fontsize=19, fontweight='bold')
plt.ylabel(r'Dry Intercept (cm$^{-3}$ $\mu$m$^{-1}$)', fontsize=19, fontweight='bold')
plt.title('CAS Below Cloud Base January - June 2022\nContours of Dry Mass', fontsize=19, fontweight='bold')
plt.xscale('log')
plt.yscale('log')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
#Dry mass inf

rho_salt = 2200

# Function to calculate dry mass using exponential fit parameters
def calculate_mass(N0, D):
    """Compute dry mass using the exponential fit (N0, D)."""
    N0_m4 = N0 * 10**6  # Convert cm⁻³µm⁻¹ to m⁻⁴

    # Mass integral over size distribution
    integrand = lambda d: np.exp(-d / D) * (d * 1e-6)**3  # Convert µm³ → m³
    mass_integral, _ = quad(integrand, 2, np.inf)  # Integrate from 2µm to ∞
    # mass_integral, _ = quad(integrand, 2, 10)  # Integrate from 2µm to 10µm
    # Compute mass with the full equation
    return (np.pi / 6) * rho_salt * N0_m4 * mass_integral

# Define log-spaced grid for slope (D) and dry intercept (N₀)
# x_min, x_max = 10**-0.1, 10**1.05  
# y_min, y_max = 10**-1.7, 10**0.8  
x_min, x_max = 10**-0.4, 10**1  
y_min, y_max = 10**-1.6, 10**1.3 

xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200)  # Dry slope grid
ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)  # Dry intercept grid

D_grid_extended, dryintercept_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)

# Compute dry mass grid
mass_grid_extended = np.zeros_like(D_grid_extended)
for i in range(D_grid_extended.shape[0]):
    for j in range(D_grid_extended.shape[1]):
        mass_grid_extended[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j]) * 1e9  # Convert kg/m³ to µg/m³

# Extract dry slope and dry intercept values from `dry_exponential_fits`
dry_slopes = []
dry_intercepts = []

for entry in dry_exponential_fits_10:
    n0 = entry['Dry_Intercept_n0']  # Dry intercept
    D = entry['Dry_E_folding_D']    # Dry slope (e-folding diameter)

    dry_intercepts.append(n0)
    dry_slopes.append(D)

# Convert to NumPy arrays for plotting
dry_slopes = np.array(dry_slopes)
dry_intercepts = np.array(dry_intercepts)

# Define mass contour levels
mass_levels = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]

# Create scatter plot with mass contours
plt.figure(figsize=(10, 8))
plt.scatter(dry_slopes, dry_intercepts, c='blue', s=80, alpha=0.7, label="Dry Data Points")

# Overlay mass contours
contour_plot = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_extended, 
                           levels=mass_levels, colors='red', alpha=0.75, linewidths=1.5)

# Add contour labels
fmt = {level: f'{int(level)} µg/m³' for level in mass_levels}
plt.clabel(contour_plot, inline=True, fontsize=13, fmt=fmt, colors='black', inline_spacing=5)
for txt in contour_plot.labelTexts:
    txt.set_fontweight('bold')
    txt.set_rotation(45)  

# Formatting and labels
plt.xlabel(r'Dry Slope ($\mu$m)', fontsize=19, fontweight='bold')
plt.ylabel(r'Dry Intercept (cm$^{-3}$ $\mu$m$^{-1}$)', fontsize=19, fontweight='bold')
plt.title('CAS Below Cloud Base January - June 2022\nContours of Dry Mass', fontsize=19, fontweight='bold')
plt.xscale('log')
plt.yscale('log')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
# %%
#Fixing contours 
min_slope_threshold = np.percentile(dry_slopes, 1)  # Remove the lowest 1% of slope values

filtered_slopes = [D for D in dry_slopes if D >= min_slope_threshold]
filtered_intercepts = [N0 for D, N0 in zip(dry_slopes, dry_intercepts) if D >= min_slope_threshold]

x_min = np.percentile(filtered_slopes, 5)  # 5th percentile
x_max = np.percentile(filtered_slopes, 95)  # 95th percentile
y_min = np.percentile(filtered_intercepts, 5)  # 5th percentile
y_max = np.percentile(filtered_intercepts, 95)  # 95th percentile

xgrid_adjusted = np.logspace(np.log10(x_min), np.log10(x_max), 200)
ygrid_adjusted = np.logspace(np.log10(y_min), np.log10(y_max), 200)
D_grid_adjusted, dryintercept_grid_adjusted = np.meshgrid(xgrid_adjusted, ygrid_adjusted)

mass_grid_adjusted = np.zeros_like(D_grid_adjusted)
for i in range(D_grid_adjusted.shape[0]):
    for j in range(D_grid_adjusted.shape[1]):
        mass_grid_adjusted[i, j] = calculate_mass(dryintercept_grid_adjusted[i, j], D_grid_adjusted[i, j]) * 1e9

plt.figure(figsize=(10, 8))
plt.scatter(filtered_slopes, filtered_intercepts, c='blue', s=80, alpha=0.7, label="Dry Data Points")

contour_plot = plt.contour(D_grid_adjusted, dryintercept_grid_adjusted, mass_grid_adjusted, 
                           levels=mass_levels, colors='red', alpha=0.75, linewidths=1.5)

plt.clabel(contour_plot, inline=True, fontsize=13, fmt=fmt, colors='black', inline_spacing=5)
for txt in contour_plot.labelTexts:
    txt.set_fontweight('bold')
    txt.set_rotation(15)

plt.xlabel(r'Dry Slope ($\mu$m)', fontsize=19, fontweight='bold')
plt.ylabel(r'Dry Intercept (cm$^{-3}$ $\mu$m$^{-1}$)', fontsize=19, fontweight='bold')
plt.title('CAS Below Cloud Base January - June 2022\nContours of Dry Mass', fontsize=19, fontweight='bold')
plt.xscale('log')
plt.yscale('log')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# %%

rho_salt = 2200 

def calculate_mass(N0, D):
    """Compute dry mass using exponential fit parameters."""
    N0_m4 = N0 * 10**6  # Convert cm⁻³µm⁻¹ to m⁻⁴
    integrand = lambda d: np.exp(-d / D) * (d * 1e-6)**3  # Convert µm³ → m³
    # mass_integral, _ = quad(integrand, 2, np.inf)
    mass_integral, _ = quad(integrand, 2, 10)  # Integrate from 2µm to ∞
    return (np.pi / 6) * rho_salt * N0_m4 * mass_integral  

# **Store results in dictionary**
dry_mass_data = []

# **Compute dry mass for each entry**
for entry in dry_exponential_fits_10:
    date = entry['Date']
    dry_intercept = entry['Dry_Intercept_n0']
    dry_slope = entry['Dry_E_folding_D']

    if dry_slope > 0 and dry_intercept > 0:
        mass_value = calculate_mass(dry_intercept, dry_slope) * 1e9  # Convert kg/m³ to µg/m³
        dry_mass_data.append({
            'Date': date,
            'Dry Slope (D)': dry_slope,
            'Dry Intercept (N0)': dry_intercept,
            'Dry Mass (µg/m³)': mass_value
        })

# **Convert to NumPy arrays for processing**
dry_slopes = np.array([entry['Dry Slope (D)'] for entry in dry_mass_data])
dry_intercepts = np.array([entry['Dry Intercept (N0)'] for entry in dry_mass_data])
dry_masses = np.array([entry['Dry Mass (µg/m³)'] for entry in dry_mass_data])

min_slope_threshold = np.percentile(dry_slopes, 1)  # Remove the lowest 1% of slope values

filtered_slopes = [D for D in dry_slopes if D >= min_slope_threshold]
filtered_intercepts = [N0 for D, N0 in zip(dry_slopes, dry_intercepts) if D >= min_slope_threshold]

x_min = np.percentile(filtered_slopes, 5)  # 5th percentile
x_max = np.percentile(filtered_slopes, 95)  # 95th percentile
y_min = np.percentile(filtered_intercepts, 5)  # 5th percentile
y_max = np.percentile(filtered_intercepts, 95)  # 95th percentile

xgrid_adjusted = np.logspace(np.log10(x_min), np.log10(x_max), 200)
ygrid_adjusted = np.logspace(np.log10(y_min), np.log10(y_max), 200)
D_grid_adjusted, dryintercept_grid_adjusted = np.meshgrid(xgrid_adjusted, ygrid_adjusted)

# **Compute mass grid for contours**
mass_grid_adjusted = np.zeros_like(D_grid_adjusted)
for i in range(D_grid_adjusted.shape[0]):
    for j in range(D_grid_adjusted.shape[1]):
        mass_grid_adjusted[i, j] = calculate_mass(dryintercept_grid_adjusted[i, j], D_grid_adjusted[i, j]) * 1e9

# **Define contour levels**
mass_levels = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]

mass_grid_adjusted = np.zeros_like(D_grid_adjusted)
for i in range(D_grid_adjusted.shape[0]):
    for j in range(D_grid_adjusted.shape[1]):
        mass_grid_adjusted[i, j] = calculate_mass(dryintercept_grid_adjusted[i, j], D_grid_adjusted[i, j]) * 1e9

plt.figure(figsize=(10, 8))
plt.scatter(filtered_slopes, filtered_intercepts, c='blue', s=80, alpha=0.7, label="Dry Data Points")

contour_plot = plt.contour(D_grid_adjusted, dryintercept_grid_adjusted, mass_grid_adjusted, 
                           levels=mass_levels, colors='red', alpha=0.75, linewidths=1.5)

plt.clabel(contour_plot, inline=True, fontsize=13, fmt=fmt, colors='black', inline_spacing=5)
for txt in contour_plot.labelTexts:
    txt.set_fontweight('bold')
    txt.set_rotation(45)

plt.xlabel(r'Dry Slope ($\mu$m)', fontsize=19, fontweight='bold')
plt.ylabel(r'Dry Intercept (cm$^{-3}$ $\mu$m$^{-1}$)', fontsize=19, fontweight='bold')
plt.title('CAS Below Cloud Base January - June 2022\nContours of Dry Mass', fontsize=19, fontweight='bold')
plt.xscale('log')
plt.yscale('log')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
rho_salt = 2200 

def calculate_mass(N0, D):
    """Compute dry mass using exponential fit parameters."""
    N0_m4 = N0 * 10**6  # Convert cm⁻³µm⁻¹ to m⁻⁴
    integrand = lambda d: np.exp(-d / D) * (d * 1e-6)**3  # Convert µm³ → m³
    # mass_integral, _ = quad(integrand, 2, np.inf)
    mass_integral, _ = quad(integrand, 2, 10)  # Integrate from 2µm to ∞
    return (np.pi / 6) * rho_salt * N0_m4 * mass_integral  

# **Store results in dictionary**
dry_mass_data_10 = []

# **Compute dry mass for each entry**
for entry in dry_exponential_fits_10:
    date = entry['Date']
    dry_intercept = entry['Dry_Intercept_n0']
    dry_slope = entry['Dry_E_folding_D']

    if dry_slope > 0 and dry_intercept > 0:
        mass_value = calculate_mass(dry_intercept, dry_slope) * 1e9  # Convert kg/m³ to µg/m³
        dry_mass_data_10.append({
            'Date': date,
            'Dry Slope (D)': dry_slope,
            'Dry Intercept (N0)': dry_intercept,
            'Dry Mass (µg/m³)': mass_value
        })

# **Convert to NumPy arrays for processing**
dry_slopes = np.array([entry['Dry Slope (D)'] for entry in dry_mass_data_10])
dry_intercepts = np.array([entry['Dry Intercept (N0)'] for entry in dry_mass_data_10])
dry_masses = np.array([entry['Dry Mass (µg/m³)'] for entry in dry_mass_data_10])

min_slope_threshold = np.percentile(dry_slopes, 1)  # Remove the lowest 1% of slope values

filtered_slopes = [D for D in dry_slopes if D >= min_slope_threshold]
filtered_intercepts = [N0 for D, N0 in zip(dry_slopes, dry_intercepts) if D >= min_slope_threshold]

x_min = np.percentile(filtered_slopes, 5)  # 5th percentile
x_max = np.percentile(filtered_slopes, 95)  # 95th percentile
y_min = np.percentile(filtered_intercepts, 5)  # 5th percentile
y_max = np.percentile(filtered_intercepts, 95)  # 95th percentile

xgrid_adjusted = np.logspace(np.log10(x_min), np.log10(x_max), 200)
ygrid_adjusted = np.logspace(np.log10(y_min), np.log10(y_max), 200)
D_grid_adjusted, dryintercept_grid_adjusted = np.meshgrid(xgrid_adjusted, ygrid_adjusted)

# **Compute mass grid for contours**
mass_grid_adjusted = np.zeros_like(D_grid_adjusted)
for i in range(D_grid_adjusted.shape[0]):
    for j in range(D_grid_adjusted.shape[1]):
        mass_grid_adjusted[i, j] = calculate_mass(dryintercept_grid_adjusted[i, j], D_grid_adjusted[i, j]) * 1e9

# **Define contour levels**
mass_levels = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]

mass_grid_adjusted = np.zeros_like(D_grid_adjusted)
for i in range(D_grid_adjusted.shape[0]):
    for j in range(D_grid_adjusted.shape[1]):
        mass_grid_adjusted[i, j] = calculate_mass(dryintercept_grid_adjusted[i, j], D_grid_adjusted[i, j]) * 1e9

plt.figure(figsize=(10, 8))
plt.scatter(filtered_slopes, filtered_intercepts, c='blue', s=80, alpha=0.7, label="Dry Data Points")

contour_plot = plt.contour(D_grid_adjusted, dryintercept_grid_adjusted, mass_grid_adjusted, 
                           levels=mass_levels, colors='red', alpha=0.75, linewidths=1.5)

plt.clabel(contour_plot, inline=True, fontsize=13, fmt=fmt, colors='black', inline_spacing=5)
for txt in contour_plot.labelTexts:
    txt.set_fontweight('bold')
    txt.set_rotation(45)

plt.xlabel(r'Dry Slope ($\mu$m)', fontsize=19, fontweight='bold')
plt.ylabel(r'Dry Intercept (cm$^{-3}$ $\mu$m$^{-1}$)', fontsize=19, fontweight='bold')
plt.title('CAS Below Cloud Base January - June 2022\nContours of Dry Mass', fontsize=19, fontweight='bold')
plt.xscale('log')
plt.yscale('log')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
#mass to inf
rho_salt = 2200 

def calculate_mass(N0, D):
    """Compute dry mass using exponential fit parameters."""
    N0_m4 = N0 * 10**6  # Convert cm⁻³µm⁻¹ to m⁻⁴
    integrand = lambda d: np.exp(-d / D) * (d * 1e-6)**3  # Convert µm³ → m³
    mass_integral, _ = quad(integrand, 2, np.inf)
    # mass_integral, _ = quad(integrand, 2, 10)  # Integrate from 2µm to ∞
    return (np.pi / 6) * rho_salt * N0_m4 * mass_integral  

# **Store results in dictionary**
dry_mass_data_inf = []

# **Compute dry mass for each entry**
for entry in dry_exponential_fits_10:
    date = entry['Date']
    dry_intercept = entry['Dry_Intercept_n0']
    dry_slope = entry['Dry_E_folding_D']

    if dry_slope > 0 and dry_intercept > 0:
        mass_value = calculate_mass(dry_intercept, dry_slope) * 1e9  # Convert kg/m³ to µg/m³
        dry_mass_data_inf.append({
            'Date': date,
            'Dry Slope (D)': dry_slope,
            'Dry Intercept (N0)': dry_intercept,
            'Dry Mass (µg/m³)': mass_value
        })

# **Convert to NumPy arrays for processing**
dry_slopes = np.array([entry['Dry Slope (D)'] for entry in dry_mass_data_inf])
dry_intercepts = np.array([entry['Dry Intercept (N0)'] for entry in dry_mass_data_inf])
dry_masses = np.array([entry['Dry Mass (µg/m³)'] for entry in dry_mass_data_inf])

min_slope_threshold = np.percentile(dry_slopes, 1)  # Remove the lowest 1% of slope values

filtered_slopes = [D for D in dry_slopes if D >= min_slope_threshold]
filtered_intercepts = [N0 for D, N0 in zip(dry_slopes, dry_intercepts) if D >= min_slope_threshold]

x_min = np.percentile(filtered_slopes, 5)  # 5th percentile
x_max = np.percentile(filtered_slopes, 95)  # 95th percentile
y_min = np.percentile(filtered_intercepts, 5)  # 5th percentile
y_max = np.percentile(filtered_intercepts, 95)  # 95th percentile

xgrid_adjusted = np.logspace(np.log10(x_min), np.log10(x_max), 200)
ygrid_adjusted = np.logspace(np.log10(y_min), np.log10(y_max), 200)
D_grid_adjusted, dryintercept_grid_adjusted = np.meshgrid(xgrid_adjusted, ygrid_adjusted)

# **Compute mass grid for contours**
mass_grid_adjusted = np.zeros_like(D_grid_adjusted)
for i in range(D_grid_adjusted.shape[0]):
    for j in range(D_grid_adjusted.shape[1]):
        mass_grid_adjusted[i, j] = calculate_mass(dryintercept_grid_adjusted[i, j], D_grid_adjusted[i, j]) * 1e9

# **Define contour levels**
mass_levels = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]

mass_grid_adjusted = np.zeros_like(D_grid_adjusted)
for i in range(D_grid_adjusted.shape[0]):
    for j in range(D_grid_adjusted.shape[1]):
        mass_grid_adjusted[i, j] = calculate_mass(dryintercept_grid_adjusted[i, j], D_grid_adjusted[i, j]) * 1e9

plt.figure(figsize=(10, 8))
plt.scatter(filtered_slopes, filtered_intercepts, c='blue', s=80, alpha=0.7, label="Dry Data Points")

contour_plot = plt.contour(D_grid_adjusted, dryintercept_grid_adjusted, mass_grid_adjusted, 
                           levels=mass_levels, colors='red', alpha=0.75, linewidths=1.5)

plt.clabel(contour_plot, inline=True, fontsize=13, fmt=fmt, colors='black', inline_spacing=5)
for txt in contour_plot.labelTexts:
    txt.set_fontweight('bold')
    txt.set_rotation(45)

plt.xlabel(r'Dry Slope ($\mu$m)', fontsize=19, fontweight='bold')
plt.ylabel(r'Dry Intercept (cm$^{-3}$ $\mu$m$^{-1}$)', fontsize=19, fontweight='bold')
plt.title('CAS Below Cloud Base January - June 2022\nContours of Dry Mass', fontsize=19, fontweight='bold')
plt.xscale('log')
plt.yscale('log')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


# %%
#Saving dry mass
#mass to 10
rho_salt = 2200  # kg/m³

def calculate_mass(N0, D):
    N0_m4 = N0 * 10**6  # Convert cm⁻³µm⁻¹ to m⁻⁴
    integrand = lambda d: np.exp(-d / D) * (d * 1e-6)**3  # Convert µm³ → m³
    mass_integral, _ = quad(integrand, 2, 10)  # Integrate from 2µm to ∞
    return (np.pi / 6) * rho_salt * N0_m4 * mass_integral  

dry_mass_data_10 = []

for entry in dry_exponential_fits:
    date = entry['Date']
    dry_intercept = entry['Dry_Intercept_n0']
    dry_slope = entry['Dry_E_folding_D']

    if dry_slope > 0 and dry_intercept > 0:
        mass_value = calculate_mass(dry_intercept, dry_slope) * 1e9  # Convert kg/m³ to µg/m³
        dry_mass_data_10.append({
            'Date': date,
            'Dry Slope (D)': dry_slope,
            'Dry Intercept (N0)': dry_intercept,
            'Dry Mass (µg/m³)': mass_value
        })

dry_slopes = np.array([entry['Dry Slope (D)'] for entry in dry_mass_data_10])
dry_intercepts = np.array([entry['Dry Intercept (N0)'] for entry in dry_mass_data_10])

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
plt.title('CAS Below Cloud Base January - June 2022\nContours of Dry Mass', fontsize=19, fontweight='bold')
plt.xscale('log')
plt.yscale('log')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
#Saving dry mass
#mass to inf
rho_salt = 2200  # kg/m³

def calculate_mass(N0, D):
    N0_m4 = N0 * 10**6  # Convert cm⁻³µm⁻¹ to m⁻⁴
    integrand = lambda d: np.exp(-d / D) * (d * 1e-6)**3  # Convert µm³ → m³
    mass_integral, _ = quad(integrand, 2, np.inf)  # Integrate from 2µm to ∞
    return (np.pi / 6) * rho_salt * N0_m4 * mass_integral  

dry_mass_data_inf = []

for entry in dry_exponential_fits:
    date = entry['Date']
    dry_intercept = entry['Dry_Intercept_n0']
    dry_slope = entry['Dry_E_folding_D']

    if dry_slope > 0 and dry_intercept > 0:
        mass_value = calculate_mass(dry_intercept, dry_slope) * 1e9  # Convert kg/m³ to µg/m³
        dry_mass_data_inf.append({
        'Date': date,
        'BCB_start': entry['BCB_start'],  # Add start time
        'BCB_stop': entry['BCB_stop'],  # Add stop time
        'Dry Slope (D)': dry_slope,
        'Dry Intercept (N0)': dry_intercept,
        'Dry Mass (µg/m³)': mass_value
    })


dry_slopes = np.array([entry['Dry Slope (D)'] for entry in dry_mass_data_inf])
dry_intercepts = np.array([entry['Dry Intercept (N0)'] for entry in dry_mass_data_inf])

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
plt.title('CAS Below Cloud Base January - June 2022\nContours of Dry Mass', fontsize=19, fontweight='bold')
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
mass_values_ug_10 = [entry['Dry Mass (µg/m³)'] for entry in dry_mass_data_10]
min_mass_ug_10 = min(mass_values_ug_10)
max_mass_ug_10 = max(mass_values_ug_10)
print(f"Min Mass (µg/m³): {min_mass_ug_10}")
print(f"Max Mass (µg/m³): {max_mass_ug_10}")

mass_values_ug_inf = [entry['Dry Mass (µg/m³)'] for entry in dry_mass_data_inf]
min_mass_ug_inf = min(mass_values_ug_inf)
max_mass_ug_inf = max(mass_values_ug_inf)
print(f"Min Mass (µg/m³): {min_mass_ug_inf}")
print(f"Max Mass (µg/m³): {max_mass_ug_inf}")
# %%
# filtered_dry_mass = [entry for entry in dry_mass_data if not np.isnan(entry['Dry Slope (D)']) and not np.isnan(entry['Dry Intercept (N0)'])]

# print(f"Filtered Dry Mass Entries: {len(filtered_dry_mass)}")  # Should be equal for slope & intercept now

# # Extract slope and intercept as NumPy arrays

# slope_array = np.array([entry['Dry Slope (D)'] for entry in filtered_dry_mass]).reshape(-1, 1)
# intercept_array = np.array([entry['Dry Intercept (N0)'] for entry in filtered_dry_mass]).reshape(-1, 1)
# data_points = np.column_stack((slope_array, intercept_array))
#%%
# Set the mass threshold
mass_threshold = 90  # µg/m³

# Filter out outliers with mass greater than 50 µg/m³
filtered_dry_mass_10 = [entry for entry in dry_mass_data_10 if (
    not np.isnan(entry['Dry Slope (D)']) and 
    not np.isnan(entry['Dry Intercept (N0)']) and 
    entry['Dry Mass (µg/m³)'] <= mass_threshold
)]

print(f"Filtered Dry Mass Entries: {len(filtered_dry_mass_10)} (after removing masses > {mass_threshold} µg/m³)")

# Extract slope and intercept as NumPy arrays after filtering
slope_array = np.array([entry['Dry Slope (D)'] for entry in filtered_dry_mass_10]).reshape(-1, 1)
intercept_array = np.array([entry['Dry Intercept (N0)'] for entry in filtered_dry_mass_10]).reshape(-1, 1)
data_points = np.column_stack((slope_array, intercept_array))
#%%
filtered_mass_values_ug_10 = [entry['Dry Mass (µg/m³)'] for entry in filtered_dry_mass_10]

mean_mass_filtered_10 = np.mean(filtered_mass_values_ug_10)
median_mass_filtered_10 = np.median(filtered_mass_values_ug_10)

print(f"Filtered Mean Mass: {mean_mass_filtered_10:.2f} µg/m³")
print(f"Filtered Median Mass: {median_mass_filtered_10:.2f} µg/m³")
#%%
# Set the mass threshold
mass_threshold = 300  # µg/m³


filtered_dry_mass_inf = [entry for entry in dry_mass_data_inf if (
    not np.isnan(entry['Dry Slope (D)']) and 
    not np.isnan(entry['Dry Intercept (N0)']) and 
    entry['Dry Mass (µg/m³)'] <= mass_threshold
)]


print(f"Filtered Dry Mass Entries: {len(filtered_dry_mass_inf)} (after removing masses > {mass_threshold} µg/m³)")

# Extract slope and intercept as NumPy arrays after filtering
slope_array = np.array([entry['Dry Slope (D)'] for entry in filtered_dry_mass_inf]).reshape(-1, 1)
intercept_array = np.array([entry['Dry Intercept (N0)'] for entry in filtered_dry_mass_inf]).reshape(-1, 1)
data_points = np.column_stack((slope_array, intercept_array))
#%%
filtered_mass_values_ug_inf = [entry['Dry Mass (µg/m³)'] for entry in filtered_dry_mass_inf]

mean_mass_filtered_inf = np.mean(filtered_mass_values_ug_inf)
median_mass_filtered_inf = np.median(filtered_mass_values_ug_inf)

print(f"Filtered Mean Mass: {mean_mass_filtered_inf:.2f} µg/m³")
print(f"Filtered Median Mass: {median_mass_filtered_inf:.2f} µg/m³")
# %%
#ambient and dry histogram 

# dry_mass_values_ug = [entry['Dry Mass (µg/m³)'] for entry in filtered_mass_values_ug]
# hydrated_mass_values_ug = [entry['Mass (µg/m³)'] for entries in ambient_mass_dict.values() for entry in entries]

bins = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 
                 16384, 32768, 65536, 131072])  

plt.figure(figsize=(10, 6))
plt.hist(filtered_mass_values_ug_10, bins=bins, color='blue', alpha=0.6, edgecolor='black', label="Dry Mass 10um", density=False)
# plt.hist(hydrated_mass_values_ug, bins=bins, color='red', alpha=0.5, edgecolor='black', label="Hydrated Mass", density=False)
plt.hist(filtered_mass_values_ug_inf, bins=bins, color='red', alpha=0.6, edgecolor='black', label="Dry Mass full", density=False)

plt.xscale('log')  
# plt.yscale('log')
plt.xlabel('Dry Mass (µg/m³)', fontsize=16, fontweight='bold')
plt.ylabel('Frequency', fontsize=16, fontweight='bold')
plt.title('CAS dry mass', fontsize=18, fontweight='bold')
plt.legend(fontsize=14)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.tight_layout()
plt.legend()
plt.show()
# mean_dry_mass = np.mean(filtered_mass_values_ug)
# # mean_hydrated_mass = np.mean(hydrated_mass_values_ug)
# median_dry_mass = np.median(filtered_mass_values_ug)
# median_hydrated_mass = np.median(hydrated_mass_values_ug)
# print(f"Mean Dry Mass: {mean_dry_mass:.2f} µg/m³")
# # print(f"Mean Hydrated Mass: {mean_hydrated_mass:.2f} µg/m³")
# print(f"Median Dry Mass: {median_dry_mass:.2f} µg/m³")
# print(f"Median Hydrated Mass: {median_hydrated_mass:.2f} µg/m³")

# %%
#Choosing 3 new cases 

rho_salt = 2200  # kg/m³

def calculate_mass(N0, D):
    """Compute dry mass using exponential fit parameters."""
    N0_m4 = N0 * 10**6  # Convert cm⁻³µm⁻¹ to m⁻⁴
    integrand = lambda d: np.exp(-d / D) * (d * 1e-6)**3  # Convert µm³ → m³
    mass_integral, _ = quad(integrand, 2, np.inf)  # Integrate from 2µm to ∞
    return (np.pi / 6) * rho_salt * N0_m4 * mass_integral  

dry_mass_data_inf = []

for entry in dry_exponential_fits:
    date = entry['Date']
    dry_intercept = entry['Dry_Intercept_n0']
    dry_slope = entry['Dry_E_folding_D']

    if dry_slope > 0 and dry_intercept > 0:
        mass_value = calculate_mass(dry_intercept, dry_slope) * 1e9  # Convert kg/m³ to µg/m³
        dry_mass_data_inf.append({
            'Date': date,
            'Dry Slope (D)': dry_slope,
            'Dry Intercept (N0)': dry_intercept,
            'Dry Mass (µg/m³)': mass_value
        })

dry_slopes = np.array([entry['Dry Slope (D)'] for entry in dry_mass_data_inf])
dry_intercepts = np.array([entry['Dry Intercept (N0)'] for entry in dry_mass_data_inf])

min_slope_threshold = np.percentile(dry_slopes, 1)  
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
plt.scatter(filtered_slopes, filtered_intercepts, c='blue', s=80, alpha=0.7)

contour_plot = plt.contour(D_grid_adjusted, dryintercept_grid_adjusted, mass_grid_adjusted, 
                           levels=mass_levels, colors='red', alpha=0.75, linewidths=1.5)

plt.clabel(contour_plot, inline=True, fontsize=13, fmt=lambda x: f"{int(x)} µg/m³", colors='black', inline_spacing=5)
for txt in contour_plot.labelTexts:
    txt.set_fontweight('bold')
    txt.set_rotation(15)

highlight_points = [
    (0.7, 10),   # Same slope, high intercept
    (0.7, 2),  # Same slope, lower intercept
    (1, 3)     # Different slope, different intercept
]

for x, y in highlight_points:
    plt.scatter(x, y, s=250, color='lime', marker='*', edgecolors='black', linewidth=1.5)

plt.xlabel(r'Dry Slope ($\mu$m)', fontsize=19, fontweight='bold')
plt.ylabel(r'Dry Intercept (cm$^{-3}$ $\mu$m$^{-1}$)', fontsize=19, fontweight='bold')
plt.title('CAS Below Cloud Base January - June 2022\nContours of Dry Mass', fontsize=19, fontweight='bold')
plt.xscale('log')
plt.yscale('log')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

# %%
#Finding our new cases 

slope_col = "Dry Slope (D)"
intercept_col = "Dry Intercept (N0)"
mass_col = "Dry Mass (µg/m³)"

slope_array = np.array([entry[slope_col] for entry in filtered_dry_mass_inf]).reshape(-1, 1)
intercept_array = np.array([entry[intercept_col] for entry in filtered_dry_mass_inf]).reshape(-1, 1)
data_points = np.column_stack((slope_array, intercept_array))

def find_closest_match(target_slope, target_intercept):
    """Find the closest real data point to a target (Slope, Intercept)."""
    distances = np.linalg.norm(data_points - np.array([target_slope, target_intercept]), axis=1)
    closest_index = distances.argmin()
    return filtered_dry_mass_inf[closest_index]

target_cases = [
    (0.7, 10),  # Same slope, high intercept
    (0.7, 2),   # Same slope, lower intercept
    (1.2, 3)      # Different slope, different intercept
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

#%%
#Pull the concentrations 
# Define target cases with matched Date, Start, Stop
target_cases = [
    {'Date': '2022-01-26', 'BCB_start': 51422, 'BCB_stop': 51621, 'Slope (D)': 0.6969, 'Intercept (N0)': 9.9214},
    {'Date': '2022-01-24', 'BCB_start': 69488, 'BCB_stop': 69692, 'Slope (D)': 0.7387, 'Intercept (N0)': 2.0512},
    {'Date': '2022-02-03', 'BCB_start': 72639, 'BCB_stop': 72800, 'Slope (D)': 1.1759, 'Intercept (N0)': 3.0752}
]

# Function to find and extract concentration from Y_BCB_calc_cm3
def find_concentration(date, start, stop):
    """Retrieve concentration from Y_BCB_calc_cm3 matching Date, Start, and Stop."""
    for entry in total_concentration_cm3:
        if (entry['Date'] == date and entry['BCB_start'] == start and entry['BCB_stop'] == stop):
            return entry['Total_Y_Concentration_cm3']  # Assuming 'Concentration' stores the values
    return np.nan  # Return NaN if not found

# Add concentration to target cases
for case in target_cases:
    concentration = find_concentration(case['Date'], case['BCB_start'], case['BCB_stop'])
    case['Total_Y_Concentration_cm3'] = concentration  # Add concentration to dictionary

# Convert to DataFrame for display
df_target_cases = pd.DataFrame(target_cases)

# Print the updated cases
print("\nUpdated Target Cases with Concentrations:")
print(df_target_cases)


# %%
#  {'Date': '2022-01-26',
#   'BCB_start': 51422,
#   'BCB_stop': 51621,
#   'Dry_Intercept_n0': 9.921370441446506,
#   'Dry_E_folding_D': 0.6968534961618702},

# # {'Date': '2022-01-24',
#   'BCB_start': 69488,
#   'BCB_stop': 69692,
#   'Dry_Intercept_n0': 2.0512199943091174,
#   'Dry_E_folding_D': 0.7387033586599581},

#  {'Date': '2022-02-03',
#   'BCB_start': 72639,
#   'BCB_stop': 72800,
#   'Dry_Intercept_n0': 3.0752158674359507,
#   'Dry_E_folding_D': 1.1759378877732407},
#%%
#Size distributions for these 3 cases

def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

target_cases = {
    ('2022-01-26', 51422, 51621): {"Dry Mass (µg/m³)": 10.936, "Total Concentration (cm⁻³)": 0.9967, "Color": "purple", "LineStyle": "solid"},
    ('2022-01-24', 69488, 69692): {"Dry Mass (µg/m³)": 3.007, "Total Concentration (cm⁻³)": 0.2284, "Color": "navy", "LineStyle": "solid"},
    ('2022-02-03', 72639, 72800): {"Dry Mass (µg/m³)": 36.850, "Total Concentration (cm⁻³)": 0.8043, "Color": "orange", "LineStyle": "solid"},
}

dry_exponential_fits = []

plt.figure(figsize=(8, 6))

legend_labels = []  

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

        case_key = (entry['Date'], entry['BCB_start'], entry['BCB_stop'])
        if case_key in target_cases:
            case_info = target_cases[case_key]
            dry_mass = case_info["Dry Mass (µg/m³)"]
            total_concentration = case_info["Total Concentration (cm⁻³)"]
            color = case_info["Color"]
            linestyle = case_info["LineStyle"]

            legend_label = f"{entry['Date']} | Mass: {dry_mass:.2f} µg/m³ | Conc: {total_concentration:.4f} cm⁻³"
            legend_labels.append((legend_label, color, linestyle))

            plt.plot(x_fit, y_fit, color=color, linewidth=3.5, linestyle=linestyle)

        else:
            plt.plot(x_fit, y_fit, color='gray', alpha=0.05, linewidth=1)  # Background curves more transparent

    except RuntimeError:
        print(f"Fit could not be performed for date {entry['Date']}")

plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(1e-7, 1e1)
plt.xlim(0, 25) 

plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base January - June 2022\nFitted Dry Size Distributions", fontsize=14, fontweight="bold")
handles = [plt.Line2D([0], [0], color=color, linewidth=3.5, linestyle=linestyle, label=label)
           for label, color, linestyle in legend_labels]
plt.legend(handles=handles, loc='upper right', fontsize=10)

plt.show()

print(f"Total successful dry exponential fits: {len(dry_exponential_fits)}")
#%%
#only to 10 um d 

common_bins=(2, 10, 10)
# Define the exponential function
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

# Target cases for highlighting
target_cases = {
    ('2022-01-26', 51422, 51621): {"Dry Mass (µg/m³)": 10.936, "Total Concentration (cm⁻³)": 0.9967, "Color": "purple", "LineStyle": "solid"},
    ('2022-01-24', 69488, 69692): {"Dry Mass (µg/m³)": 3.007, "Total Concentration (cm⁻³)": 0.2284, "Color": "navy", "LineStyle": "solid"},
    ('2022-02-03', 72639, 72800): {"Dry Mass (µg/m³)": 36.850, "Total Concentration (cm⁻³)": 0.8043, "Color": "orange", "LineStyle": "solid"},
}

dry_exponential_fits_ = []

plt.figure(figsize=(8, 6))
legend_labels = []

for entry in filtered_master_BCB_ddry:
    ddry_values = np.array(entry['ddry'])
    dN_dD_dry = np.array(entry['dN/dDdry'])

    # **Filter to only include values ≤ 10 µm before fitting**
    valid_indices = (ddry_values <= 10) & ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry)
    ddry_values_10um = ddry_values[valid_indices]
    dN_dD_dry_10um = dN_dD_dry[valid_indices]

    # Ensure there are enough valid points for fitting (min 5 points)
    if len(ddry_values_10um) < 5:
        continue

    try:
        # Fit the exponential **only using data ≤ 10 µm**
        popt, _ = curve_fit(exponential, ddry_values_10um, dN_dD_dry_10um, 
                            p0=(max(dN_dD_dry_10um), 5), maxfev=5000)
        n0, D = popt

        dry_exponential_fits.append({
            'Date': entry['Date'],
            'BCB_start': entry['BCB_start'],
            'BCB_stop': entry['BCB_stop'],
            'Dry_Intercept_n0': n0,
            'Dry_E_folding_D': D
        })

        # Generate fitted curve **only up to 10 µm**
        x_fit = np.linspace(min(ddry_values_10um), 10, 100)
        y_fit = exponential(x_fit, *popt)

        # Check if this entry is in the highlighted cases
        case_key = (entry['Date'], entry['BCB_start'], entry['BCB_stop'])
        if case_key in target_cases:
            case_info = target_cases[case_key]
            dry_mass = case_info["Dry Mass (µg/m³)"]
            total_concentration = case_info["Total Concentration (cm⁻³)"]
            color = case_info["Color"]
            linestyle = case_info["LineStyle"]

            legend_label = f"{entry['Date']} | Mass: {dry_mass:.2f} µg/m³ | Conc: {total_concentration:.4f} cm⁻³"
            legend_labels.append((legend_label, color, linestyle))

            plt.plot(x_fit, y_fit, color=color, linewidth=3.5, linestyle=linestyle)

        else:
            plt.plot(x_fit, y_fit, color='gray', alpha=0.05, linewidth=1)  # Background curves more transparent

    except RuntimeError:
        print(f"Fit could not be performed for date {entry['Date']}")

# Formatting
plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(1e-7, 1e1)
plt.xlim(0, 10)  # Ensure the plot is limited to ≤ 10 µm

plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base January - June 2022\nFitted Dry Size Distributions (≤10 µm)", fontsize=14, fontweight="bold")

# Add legend for highlighted cases
handles = [plt.Line2D([0], [0], color=color, linewidth=3.5, linestyle=linestyle, label=label)
           for label, color, linestyle in legend_labels]
plt.legend(handles=handles, loc='upper right', fontsize=10)

plt.show()

print(f"Total successful dry exponential fits: {len(dry_exponential_fits)}")


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
#histogram of altitudes


all_altitudes = []

for flight in master_BCB:
    for wind_alt in flight:
        all_altitudes.extend(wind_alt['Alts_mean'])

all_altitudes = [alt for alt in all_altitudes if not np.isnan(alt)]

plt.figure(figsize=(10, 8))
plt.hist(all_altitudes, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel('Mean altitude (m)', fontsize=16, fontweight='bold')
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.ylabel('Frequency of flight legs', fontsize=16, fontweight='bold')
plt.title('460 below cloud base legs\n January - June 2022', fontsize=18, fontweight='bold')
plt.show()
#%%
#mean windspeed

# Extract corrected wind speed values
corrected_windspeeds = corrected_calc_bcb['Corrected_bcb_windspeed']

# Remove NaN values if any
corrected_windspeeds = [ws for ws in corrected_windspeeds if not np.isnan(ws)]

# Calculate mean wind speed
mean_corrected_windspeed = np.mean(corrected_windspeeds)

# Print result
print(f"Mean Corrected Wind Speed: {mean_corrected_windspeed:.2f} m/s")

#%%
#standard deviation of windspeed
mean_corrected_windspeed = sum(corrected_windspeeds) / len(corrected_windspeeds)

# Calculate variance (sum of squared differences from the mean)
variance = sum((ws - mean_corrected_windspeed) ** 2 for ws in corrected_windspeeds) / (len(corrected_windspeeds) - 1)

# Calculate standard deviation (square root of variance)
std_corrected_windspeed = variance ** 0.5

# Print result
print(f"Standard Deviation of Corrected Wind Speed: {std_corrected_windspeed:.2f} m/s")

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
common_bins=np.linspace(2, 10, 25)
#%%
# Define wind speed bins
# windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]
windspeed_bins = [(0, 3), (3.001, 6.5), (6.501, 8.5), (8.501, np.inf)]

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
#adding error bars 
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Compute Mean and Standard Error for Each Windspeed Bin
bin_means = {}
bin_stds = {}
bin_stderrs = {}

for idx, data in grouped_distributions.items():
    if data:
        data_array = np.array(data)  # Convert list to array
        bin_means[idx] = np.mean(data_array, axis=0)  # Mean of distributions
        bin_stds[idx] = np.std(data_array, axis=0)  # Standard deviation
        bin_stderrs[idx] = bin_stds[idx] / np.sqrt(len(data))  # Standard error

# Step 2: Plot with Error Bars
plt.figure(figsize=(10, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if idx in bin_means:
        avg_distribution = bin_means[idx]
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])
        error_bars = bin_stderrs[idx]  # Standard error bars

        plt.errorbar(
            common_bins, avg_distribution, yerr=error_bars, 
            label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", 
            linewidth=2.5, capsize=3, capthick=1.5, fmt='-o', markersize=4
        )

plt.yscale('log')
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=16, fontweight="bold")
plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=16, fontweight="bold")
plt.title('Dry Size Distributions Binned by Average Wind Speed', fontweight='bold', fontsize=17)
plt.legend(title=r"Average wind speed m s$^{-1}$")
plt.tight_layout()
plt.ylim(1e-4, 10**0)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')

# Show plot
plt.show()
#%%
# Prepare the legend text with statistics
legend_texts = []
for idx, (low, high) in enumerate(windspeed_bins):
    if idx in bin_means:
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])
        avg_std_error = np.mean(bin_stderrs[idx])  # Average standard error across all bins

        legend_texts.append(f"{avg_windspeed:.1f} m/s, n={num_legs} legs\nAvg SE: {avg_std_error:.3f}")

# Create the plot with error bars and legend containing error stats
plt.figure(figsize=(10, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if idx in bin_means:
        avg_distribution = bin_means[idx]
        error_bars = bin_stderrs[idx]  # Standard error bars

        plt.errorbar(
            common_bins, avg_distribution, yerr=error_bars,
            label=legend_texts[idx], linewidth=2.5, capsize=3, capthick=1.5, fmt='-o', markersize=4
        )

plt.yscale('log')
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=16, fontweight="bold")
plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=16, fontweight="bold")
plt.title('CAS Dry Size Distributions Binned by Average Wind Speed', fontweight='bold', fontsize=17)
plt.legend(title="Windspeed & Error Stats", loc="upper right")
plt.tight_layout()
plt.ylim(1e-4, 10**0)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')

# Show the plot with enhanced legend
plt.show()

#%%
#adding uncertainty
plt.figure(figsize=(10, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)  # Mean
        std_distribution = np.std(grouped_distributions[idx], axis=0, ddof=1)  # Standard deviation
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])

        # Plot mean line
        plt.plot(common_bins, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linewidth=2.5)

        # Add uncertainty as shaded region (1 standard deviation)
        plt.fill_between(common_bins, avg_distribution - std_distribution, avg_distribution + std_distribution, alpha=0.3)

plt.yscale('log')
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=16, fontweight="bold")
plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=16, fontweight="bold")
plt.title('CAS Dry Size Distributions Binned by Average Wind Speed', fontweight='bold', fontsize=17)
plt.legend(title=r"Average wind speed m s$^{-1}$")
plt.tight_layout()
plt.ylim(1e-4, 10**0)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.show()
#%%

plt.figure(figsize=(10, 8))

# Choose whether to use Standard Error (SE) or Interquartile Range (IQR) for uncertainty
use_standard_error = True  # Set to False to use IQR instead

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)  # Mean
        num_legs = len(grouped_distributions[idx])
        avg_windspeed = np.mean(mean_windspeeds[idx])

        # Standard Error (SE) Option
        if use_standard_error:
            std_error = np.std(grouped_distributions[idx], axis=0, ddof=1) / np.sqrt(num_legs)
            lower_bound = avg_distribution - std_error
            upper_bound = avg_distribution + std_error

        # Interquartile Range (IQR) Option
        else:
            lower_bound = np.percentile(grouped_distributions[idx], 25, axis=0)  # 25th percentile
            upper_bound = np.percentile(grouped_distributions[idx], 75, axis=0)  # 75th percentile

        # Plot the mean size distribution
        plt.plot(common_bins, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linewidth=2.5)

        # Plot uncertainty as shaded region
        plt.fill_between(common_bins, lower_bound, upper_bound, alpha=0.3)

# Set Y-axis scaling
plt.yscale('log')  # Change to 'linear' temporarily if needed for checking uncertainty

# Labels and formatting
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=16, fontweight="bold")
plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=16, fontweight="bold")
plt.title('Dry Size Distributions Binned by Average Wind Speed', fontweight='bold', fontsize=17)
plt.legend(title=r"Average wind speed m s$^{-1}$")
plt.tight_layout()
plt.ylim(1e-4, 10**0)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')

# Show the plot
plt.show()

# Print sample sizes per bin
for idx, group in grouped_distributions.items():
    print(f"Windspeed bin {idx} ({windspeed_bins[idx]} m/s): {len(group)} legs")
#%%
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.histplot(corrected_windspeeds, bins=30, kde=True, stat="density", color="blue", edgecolor="black", alpha=0.7)

# Add vertical lines for wind speed bin edges
for low, high in windspeed_bins:
    plt.axvline(low, color='red', linestyle='--', alpha=0.7)
    plt.axvline(high, color='red', linestyle='--', alpha=0.7)

plt.xlabel("Corrected Wind Speed (m/s)", fontsize=14, fontweight="bold")
plt.ylabel("Probability Density", fontsize=14, fontweight="bold")
plt.title("10 m Wind Speeds Below Cloud Base\nJanuary-June 2022", fontsize=16, fontweight="bold")
plt.xticks(fontsize=12, fontweight="bold")
plt.yticks(fontsize=12, fontweight="bold")
plt.show()













































































































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
#%%
#adding error bars to fitted bins

# Define the fit function (exponential)
def fit_function(x, n0, D):
    return n0 * np.exp(-x / D)

# Wind speed bin colors (must match the bins order)
windspeed_colors = ['blue', 'orange', 'green', 'red']  # Order must match bins

# Store fit results
fit_results = {}
errorbar_handles = []


plt.figure(figsize=(10, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:  # Only process non-empty bins

        # Convert list to numpy array
        concentrations_array = np.array(grouped_distributions[idx])
        
        # Compute mean and standard deviation for each bin
        avg_concentration = np.mean(concentrations_array, axis=0)
        std_concentration = np.std(concentrations_array, axis=0)
        std_error = std_concentration / np.sqrt(len(grouped_distributions[idx]))  # Standard error

        # Avoid log issues by replacing zero or negative values
        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)

        # Compute average wind speed in this bin
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])

        try:
            # Fit exponential function to the size distribution
            popt, _ = curve_fit(fit_function, common_bins, avg_concentration, p0=(1, 5), maxfev=5000)
            n0_fit, D_fit = popt
            fit_results[idx] = {'n0': n0_fit, 'D': D_fit, 'avg_windspeed': avg_windspeed, 'num_legs': num_legs}

            # Generate fitted line using the optimal parameters
            x_fit = np.linspace(min(common_bins), max(common_bins), 10)
            y_fit = fit_function(x_fit, *popt)

            # Plot the fitted exponential curve
            plt.plot(x_fit, y_fit, color=windspeed_colors[idx], linewidth=3, linestyle='-',
                     label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")

            # Add error bars to the original data points
            # plt.errorbar(common_bins, avg_concentration, yerr=std_error, fmt='o', color=windspeed_colors[idx],
            #              capsize=3, capthick=1.5, markersize=5, label=None)  # Error bars

        except RuntimeError:
            print(f"Exponential fit failed for windspeed bin {avg_windspeed:.1f} m/s")
# errbar = plt.errorbar(common_bins, avg_concentration, yerr=std_error, fmt='o', color=windspeed_colors[idx],
#                       capsize=3, capthick=1.5, markersize=5, label=None)  # Error bars
# errorbar_handles.append((errbar, f"Error bars ({avg_windspeed:.1f} m/s)"))

# Customize plot labels and settings
plt.ylabel(r'Number concentration (cm$^{-3}$ µm$^{-1}$)', fontweight='bold', fontsize=18)
plt.xlabel('Bin diameter (µm)', fontweight='bold', fontsize=18)
plt.yscale('log')
plt.ylim(10**-4, 10**0)
plt.title('CAS Below Cloud Base January - June 2022\nFitted Dry Size Distributions Binned by Average Windspeed', fontweight='bold', fontsize=18)
plt.legend(title=r"Wind speed bins (m s$^{-1}$)", title_fontsize=15, fontsize=13, frameon=True, prop={'weight': 'bold'})
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.tight_layout()
# handles, labels = plt.gca().get_legend_handles_labels()
# for err_handle, err_label in errorbar_handles:
#     handles.append(err_handle[0])  # Add error bar handle
#     labels.append(err_label)  # Add corresponding label

plt.legend(handles, labels, title=r"Wind speed bins", title_fontsize=15, fontsize=13, frameon=True, prop={'weight': 'bold'})

# Show the plot with error bars
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the fit function (exponential)
def fit_function(x, n0, D):
    return n0 * np.exp(-x / D)

# Wind speed bin colors (must match the bins order)
windspeed_colors = ['blue', 'orange', 'green', 'red']  # Order must match bins

# Store fit results
fit_results = {}
errorbar_handles = []
legend_texts = []  # Store text for legend entries

plt.figure(figsize=(10, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:  # Only process non-empty bins

        # Convert list to numpy array
        concentrations_array = np.array(grouped_distributions[idx])
        
        # Compute mean and standard deviation for each bin
        avg_concentration = np.mean(concentrations_array, axis=0)
        std_concentration = np.std(concentrations_array, axis=0)
        std_error = std_concentration / np.sqrt(len(grouped_distributions[idx]))  # Standard error

        # Avoid log issues by replacing zero or negative values
        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)

        # Compute average wind speed in this bin
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])

        # Compute average standard error for legend
        avg_se = np.mean(std_error)

        try:
            # Fit exponential function to the size distribution
            popt, _ = curve_fit(fit_function, common_bins, avg_concentration, p0=(1, 5), maxfev=5000)
            n0_fit, D_fit = popt
            fit_results[idx] = {'n0': n0_fit, 'D': D_fit, 'avg_windspeed': avg_windspeed, 'num_legs': num_legs}

            # Generate fitted line using the optimal parameters
            x_fit = np.linspace(min(common_bins), max(common_bins), 10)
            y_fit = fit_function(x_fit, *popt)

            # Plot the fitted exponential curve
            plt.plot(x_fit, y_fit, color=windspeed_colors[idx], linewidth=3, linestyle='-',
                     label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")

            # Add error bars to the original data points
            errbar = plt.errorbar(common_bins, avg_concentration, yerr=std_error, fmt='o', color=windspeed_colors[idx],
                                  capsize=3, capthick=1.5, markersize=5, label=None)  # Error bars
            
            # Store error bar handle and text for the legend
            errorbar_handles.append(errbar)
            # legend_texts.append(f"Errors ({avg_windspeed:.1f} m/s, Avg SE: {avg_se:.3f})")

        except RuntimeError:
            print(f"Exponential fit failed for windspeed bin {avg_windspeed:.1f} m/s")

# Customize plot labels and settings
plt.ylabel(r'Number concentration (cm$^{-3}$ µm$^{-1}$)', fontweight='bold', fontsize=18)
plt.xlabel('Bin diameter (µm)', fontweight='bold', fontsize=18)
plt.yscale('log')
plt.ylim(10**-4, 10**0)
plt.title('CAS Below Cloud Base January - June 2022\nFitted Dry Size Distributions Binned by Average Windspeed', fontweight='bold', fontsize=18)

# Create legend with both fitted curves and error bars
handles, labels = plt.gca().get_legend_handles_labels()
for err_handle, err_label in zip(errorbar_handles, legend_texts):
    handles.append(err_handle[0])  # Add error bar handle
    labels.append(err_label)  # Add corresponding label with average SE

plt.legend(handles, labels, title=r"Wind speed bins & Errors (m s$^{-1}$)", title_fontsize=15, fontsize=13, frameon=True, prop={'weight': 'bold'})
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.tight_layout()

# Show the plot with error bars and their legend
plt.show()
#%%
residuals = avg_concentration - fit_function(common_bins, *popt)

plt.figure(figsize=(10, 6))
plt.axhline(0, color='black', linestyle='--', linewidth=1)  # Reference line at 0
plt.errorbar(common_bins, residuals, yerr=std_error, fmt='o', color=windspeed_colors[idx], capsize=3)
plt.xlabel("Bin diameter (µm)", fontsize=14, fontweight="bold")
plt.ylabel("Residuals (Observed - Fitted)", fontsize=14, fontweight="bold")
plt.title("Residuals of Exponential Fit", fontsize=16, fontweight="bold")
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the fit function (exponential)
def fit_function(x, n0, D):
    return n0 * np.exp(-x / D)

# Create figure and axis
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot fitted distributions
for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:  # Only process non-empty bins

        # Convert list to numpy array
        concentrations_array = np.array(grouped_distributions[idx])

        # Compute mean and standard deviation for each bin
        avg_concentration = np.mean(concentrations_array, axis=0)
        std_concentration = np.std(concentrations_array, axis=0)
        std_error = std_concentration / np.sqrt(len(grouped_distributions[idx]))  # Standard error

        # Compute average wind speed in this bin
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])

        try:
            # Fit exponential function to the size distribution
            popt, _ = curve_fit(fit_function, common_bins, avg_concentration, p0=(1, 5), maxfev=5000)
            n0_fit, D_fit = popt

            # Generate fitted line
            x_fit = np.linspace(min(common_bins), max(common_bins), 10)
            y_fit = fit_function(x_fit, *popt)

            # Plot the fitted curve
            ax1.plot(x_fit, y_fit, color=windspeed_colors[idx], linewidth=3, linestyle='-',
                     label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")

            # Plot the original data points with error bars
            ax1.errorbar(common_bins, avg_concentration, yerr=std_error, fmt='o', color=windspeed_colors[idx], capsize=3)

            # Compute residuals
            residuals = avg_concentration - fit_function(common_bins, *popt)

        except RuntimeError:
            print(f"Exponential fit failed for windspeed bin {avg_windspeed:.1f} m/s")

# Create secondary y-axis for residuals
# Create secondary y-axis for residuals
ax2 = ax1.twinx()
residual_handles = []  # Store handles only once
residual_labels = []  # Store unique labels

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:  # Only process non-empty bins
        popt, _ = curve_fit(fit_function, common_bins, np.mean(np.array(grouped_distributions[idx]), axis=0), p0=(1, 5), maxfev=5000)
        residuals = np.mean(np.array(grouped_distributions[idx]), axis=0) - fit_function(common_bins, *popt)

        # Plot residuals on the secondary y-axis
        scatter = ax2.scatter(common_bins, residuals, color=windspeed_colors[idx], marker='s', edgecolor='black', label=f"Residuals {low}-{high} m/s")

        # Only add one legend entry per wind speed bin
        residual_handles.append(scatter)
        residual_labels.append(f"Residuals {low}-{high} m/s")

# Update the legend
ax2.legend(residual_handles, residual_labels, loc="lower right", fontsize=12, frameon=True)

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the fit function (exponential)
def fit_function(x, n0, D):
    return n0 * np.exp(-x / D)

# Create figure and axis
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot fitted distributions
for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:  # Only process non-empty bins

        # Convert list to numpy array
        concentrations_array = np.array(grouped_distributions[idx])

        # Compute mean and standard deviation for each bin
        avg_concentration = np.mean(concentrations_array, axis=0)
        std_concentration = np.std(concentrations_array, axis=0)
        std_error = std_concentration / np.sqrt(len(grouped_distributions[idx]))  # Standard error

        # Compute average wind speed in this bin
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])

        try:
            # Fit exponential function to the size distribution
            popt, _ = curve_fit(fit_function, common_bins, avg_concentration, p0=(1, 5), maxfev=5000)
            n0_fit, D_fit = popt

            # Generate fitted line
            x_fit = np.linspace(min(common_bins), max(common_bins), 10)
            y_fit = fit_function(x_fit, *popt)

            # Plot the fitted curve
            ax1.plot(x_fit, y_fit, color=windspeed_colors[idx], linewidth=3, linestyle='-',
                     label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")

            # Plot the original data points with error bars
            ax1.errorbar(common_bins, avg_concentration, yerr=std_error, fmt='o', color=windspeed_colors[idx], capsize=3)

            # Compute residuals
            residuals = avg_concentration - fit_function(common_bins, *popt)

        except RuntimeError:
            print(f"Exponential fit failed for windspeed bin {avg_windspeed:.1f} m/s")

# **Set left y-axis (fitted distributions) to log scale**
ax1.set_yscale("log")

# Create secondary y-axis for residuals
ax2 = ax1.twinx()
residual_handles = []  # Store handles only once
residual_labels = []  # Store unique labels

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:  # Only process non-empty bins
        popt, _ = curve_fit(fit_function, common_bins, np.mean(np.array(grouped_distributions[idx]), axis=0), p0=(1, 5), maxfev=5000)
        residuals = np.mean(np.array(grouped_distributions[idx]), axis=0) - fit_function(common_bins, *popt)

        # Plot residuals on the secondary y-axis
        scatter = ax2.scatter(common_bins, residuals, color=windspeed_colors[idx], marker='s', edgecolor='black', label=f"Residuals {low}-{high} m/s")

        # Only add one legend entry per wind speed bin
        residual_handles.append(scatter)
        residual_labels.append(f"Residuals {low}-{high} m/s")

# **Set labels and log scale**
ax1.set_xlabel("Bin diameter (µm)", fontsize=14, fontweight="bold")
ax1.set_ylabel("Number concentration (cm$^{-3}$ µm$^{-1}$)", fontsize=14, fontweight="bold")
ax2.set_ylabel("Residuals (Observed - Fitted)", fontsize=14, fontweight="bold", color="red")

# **Ensure residuals remain in linear scale**
ax2.axhline(0, color='black', linestyle='--', linewidth=1)  # Reference line for residuals

# **Update legends**
ax1.legend(loc="upper right", fontsize=12, frameon=True)
ax2.legend(residual_handles, residual_labels, loc="lower right", fontsize=12, frameon=True)

plt.title("CAS Fitted Dry Size Distributions with Residuals", fontsize=16, fontweight="bold")

# Show plot
plt.show()


#%%
#Creating a table 
# Define headers for the table
headers = ["Wind Speed Bin (m/s)", "Avg. Wind Speed (m/s)", "n₀ (cm⁻³ µm⁻¹)", "D (µm)", "Number of Legs"]

# Extract values for each bin
table_data = []
for idx, result in fit_results.items():
    table_data.append([
        f"{windspeed_bins[idx][0]} - {windspeed_bins[idx][1]}",
        f"{result['avg_windspeed']:.1f}",
        f"{result['n0']:.3e}",
        f"{result['D']:.3f}",
        f"{result['num_legs']}"
    ])

# Determine column widths dynamically
col_widths = [max(len(str(item)) for item in col) for col in zip(*([headers] + table_data))]

# Print the table with formatting
separator = "+".join(["-" * (width + 2) for width in col_widths])
print(separator)
print("| " + " | ".join(header.ljust(width) for header, width in zip(headers, col_widths)) + " |")
print(separator)
for row in table_data:
    print("| " + " | ".join(str(item).ljust(width) for item, width in zip(row, col_widths)) + " |")
print(separator)
#%%


windspeed_bins = [(0, 3), (3, 6.5), (6.5, 8.5), (8.5, np.inf)]

# Convert fit_results dictionary to a pandas DataFrame
table_df = pd.DataFrame.from_dict(fit_results, orient='index')

# Add wind speed bin ranges and reformat column names
table_df.insert(0, "Wind Speed Bin (m/s)", [f"{windspeed_bins[idx][0]} - {windspeed_bins[idx][1]}" for idx in table_df.index])
table_df.rename(columns={
    "avg_windspeed": "Avg. Wind Speed (m/s)",
    "n0": "n₀ (cm⁻³ µm⁻¹)",
    "D": "D (µm)",
    "num_legs": "Number of Legs"
}, inplace=True)

# Convert scientific notation for n₀
table_df["n₀ (cm⁻³ µm⁻¹)"] = table_df["n₀ (cm⁻³ µm⁻¹)"].apply(lambda x: f"{x:.3e}")
table_df["D (µm)"] = table_df["D (µm)"].apply(lambda x: f"{x:.3f}")

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('tight')
ax.axis('off')

# Create table in the plot
table = ax.table(cellText=table_df.values, colLabels=table_df.columns, cellLoc='center', loc='center')

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width([0, 1, 2, 3, 4])

# Add title
plt.title("Below Cloud Base Wind Speed Dry Size Distributions", fontsize=12, fontweight="bold")

# Save the table as an image

# Show the table
plt.show()


#%%
#trying a third order polynomial

# Define wind speed bins
windspeed_colors = ['blue', 'orange', 'green', 'red']  # Order must match bins
fit_results = {}

plt.figure(figsize=(10, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        
        concentrations_array = np.array(grouped_distributions[idx])
        
        avg_concentration = np.mean(concentrations_array, axis=0)
        
        # Ensure no zero or negative values (replace with small positive number)
        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)
        
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])

        try:
            # Fit a third-order polynomial (degree=3)
            poly_coeffs = np.polyfit(common_bins, np.log10(avg_concentration), 3)  # Fit in log space
            poly_fit_func = np.poly1d(poly_coeffs)

            # Generate polynomial fit curve
            x_fit = np.linspace(min(common_bins), max(common_bins), 100)
            y_fit = 10**poly_fit_func(x_fit)  # Convert back from log scale

            # Store fit results
            fit_results[idx] = {'poly_coeffs': poly_coeffs, 'avg_windspeed': avg_windspeed, 'num_legs': num_legs}

            # Plot polynomial fit
            plt.plot(x_fit, y_fit, color=windspeed_colors[idx], linewidth=3, linestyle='-',
                     label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")

        except np.linalg.LinAlgError:
            print(f"Polynomial fit failed for windspeed bin {avg_windspeed:.1f} m/s")

# Formatting
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

#%%
# Ensure correct bin edges (10 bins from 2 to 10)
bin_edges = np.linspace(2, 10, 11)  # 11 edges to create 10 bins
bin_widths = np.diff(bin_edges)  # Compute bin widths (should be 10 values)
for idx in range(len(windspeed_bins)):
    if grouped_distributions[idx]:
        print(f"Windspeed Bin {idx}:")
        print(f"   - Shape of size distributions: {np.array(grouped_distributions[idx]).shape}")  # Should be (N, 10)
        print(f"   - Shape of bin_widths: {bin_widths.shape}")  # Should be (10,)
for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        avg_windspeed = np.mean(mean_windspeeds[idx])  # Average windspeed for this bin

        # Ensure the shape of dist matches bin_widths
        avg_concentration_per_leg = [np.sum(dist[:len(bin_widths)] * bin_widths) for dist in grouped_distributions[idx]]

        avg_concentration = np.mean(avg_concentration_per_leg)  # Average over all legs in this bin

        avg_windspeeds.append(avg_windspeed)
        total_concentrations.append(avg_concentration)

#%%
#Computing regression
colors = ['navy', 'orange', 'purple', 'darkgreen']
# Define linear regression function
def linear_model(x, m, b):
    return m * x + b

# Extract average wind speeds and total concentrations from binned distributions
avg_windspeeds = []
total_concentrations = []

bin_widths = np.diff(common_bins)  # Compute bin widths once

# Compute correct bin widths
bin_edges = np.linspace(2, 10, 10)  # Ensures correct bin edges from 2 to 10 μm
bin_widths = np.diff(bin_edges)  # Compute widths between bin edges

# Convert size distributions from cm⁻³ μm⁻¹ to total concentration (cm⁻³)
for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        avg_windspeed = np.mean(mean_windspeeds[idx])  # Average windspeed for this bin

        # Convert using correct bin widths
        avg_concentration_per_leg = [np.sum(dist * bin_widths) for dist in grouped_distributions[idx]]
        avg_concentration = np.mean(avg_concentration_per_leg)  # Average over all legs in this bin

        avg_windspeeds.append(avg_windspeed)
        total_concentrations.append(avg_concentration)


# Convert to numpy arrays for fitting
windspeed_values = np.array(avg_windspeeds)
total_concentrations = np.array(total_concentrations)

# Perform linear regression
popt, _ = curve_fit(linear_model, windspeed_values, total_concentrations)
m_fit, b_fit = popt

# Compute R² value
residuals = total_concentrations - linear_model(windspeed_values, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((total_concentrations - np.mean(total_concentrations))**2)
r_squared = 1 - (ss_res / ss_tot)

# Plot Wind Speed vs. Total Droplet Concentration
plt.figure(figsize=(8, 6))
plt.figure(figsize=(8, 6))
for idx in range(len(windspeed_bins)):
    plt.scatter(windspeed_values[idx], total_concentrations[idx], 
                color=colors[idx], s=100, edgecolor='black', zorder=3)
x_fit = np.linspace(min(windspeed_values), max(windspeed_values), 100)
y_fit = linear_model(x_fit, *popt)
plt.plot(x_fit, y_fit, 'r-', label=f'Fit: y = {m_fit:.3f}x + {b_fit:.3f}, R² = {r_squared:.2f}')
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=16, fontweight='bold')
plt.ylabel("Total Wind Speed Bin Concentration (cm$^{-3}$)", fontsize=16, fontweight='bold')
plt.title("Wind Speed and Total Concentration Correlation", fontsize=16, fontweight='bold')
legend_labels = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[idx], markersize=10, 
                             label=f"{windspeed_values[idx]:.1f} m/s") for idx in range(len(windspeed_bins))]

plt.legend(handles=legend_labels + [plt.Line2D([0], [0], color='red', label=f'Fit: y = {m_fit:.3f}x + {b_fit:.3f}, R² = {r_squared:.2f}')], 
           title="Wind Speed Bins", title_fontsize=14, fontsize=14)
plt.tight_layout()
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.show()
print(f"Slope (m): {m_fit:.3f}")
print(f"Intercept (b): {b_fit:.3f}")
print(f"R² value: {r_squared:.2f}")

# %%

# Define wind speed bins
# windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]
windspeed_bins = [(0, 3), (3.001, 6.5), (6.501, 8.5), (8.501, np.inf)]

colors = ['blue', 'orange', 'green', 'red']  # Colors must match bins

# Ensure correct bin edges (10 bins from 2 to 10 µm)
bin_edges = np.linspace(2, 10, 11)  # 11 edges to create 10 bins
bin_widths = np.diff(bin_edges)  # Compute bin widths (should be 10 values)

# Store results
avg_windspeeds = []
total_concentrations = []

# Convert size distributions from cm⁻³ μm⁻¹ to total concentration (cm⁻³)
for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:  # Ensure bin has data
        avg_windspeed = np.mean(mean_windspeeds[idx])  # Average windspeed for this bin

        # Convert using correct bin widths (integrate dN/dD)
        avg_concentration_per_leg = [np.sum(dist * bin_widths) for dist in grouped_distributions[idx]]
        avg_concentration = np.mean(avg_concentration_per_leg)  # Average over all legs in this bin

        avg_windspeeds.append(avg_windspeed)
        total_concentrations.append(avg_concentration)

# Convert to numpy arrays for fitting
windspeed_values = np.array(avg_windspeeds)
total_concentrations = np.array(total_concentrations)

# Perform linear regression
def linear_model(x, m, b):
    return m * x + b

popt, _ = curve_fit(linear_model, windspeed_values, total_concentrations)
m_fit, b_fit = popt

# Compute R² value
residuals = total_concentrations - linear_model(windspeed_values, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((total_concentrations - np.mean(total_concentrations))**2)
r_squared = 1 - (ss_res / ss_tot)

# Plot Wind Speed vs. Total Droplet Concentration
plt.figure(figsize=(8, 6))
for idx in range(len(windspeed_bins)):
    plt.scatter(windspeed_values[idx], total_concentrations[idx], 
                color=colors[idx], s=100, edgecolor='black', zorder=3)

# Fit line
x_fit = np.linspace(min(windspeed_values), max(windspeed_values), 100)
y_fit = linear_model(x_fit, *popt)
plt.plot(x_fit, y_fit, 'r-', label=f'Fit: y = {m_fit:.3f}x + {b_fit:.3f}, R² = {r_squared:.2f}')

# Labels & Titles
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=16, fontweight='bold')
plt.ylabel("Total Wind Speed Bin Concentration (cm$^{-3}$)", fontsize=16, fontweight='bold')
plt.title("Wind Speed and Total Concentration Correlation", fontsize=16, fontweight='bold')

# Legend
legend_labels = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[idx], markersize=10, 
               label=f"{windspeed_values[idx]:.1f} m/s") for idx in range(len(windspeed_bins))
]
plt.legend(handles=legend_labels + [plt.Line2D([0], [0], color='red', label=f'Fit: y = {m_fit:.3f}x + {b_fit:.3f}, R² = {r_squared:.2f}')], 
           title="Wind Speed Bins", title_fontsize=14, fontsize=13)

# Final Plot Settings
plt.tight_layout()
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.show()

# Print Results
print(f"Slope (m): {m_fit:.3f}")
print(f"Intercept (b): {b_fit:.3f}")
print(f"R² value: {r_squared:.2f}")

# %%
#fitting a second order polynomial

# Define wind speed bins and colors
windspeed_colors = ['blue', 'orange', 'green', 'red']
fit_results = {}

plt.figure(figsize=(10, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        
        concentrations_array = np.array(grouped_distributions[idx])
        avg_concentration = np.mean(concentrations_array, axis=0)

        # Avoid negative or zero values
        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)

        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])

        try:
            # Fit a second-order polynomial (degree=2) in log space
            poly_coeffs = np.polyfit(common_bins, np.log10(avg_concentration), 2)
            poly_fit_func = np.poly1d(poly_coeffs)

            # Generate polynomial fit curve
            x_fit = np.linspace(min(common_bins), max(common_bins), 100)
            y_fit = 10**poly_fit_func(x_fit)  # Convert back from log space

            # Store fit results
            fit_results[idx] = {'poly_coeffs': poly_coeffs, 'avg_windspeed': avg_windspeed, 'num_legs': num_legs}

            # Plot polynomial fit
            plt.plot(x_fit, y_fit, color=windspeed_colors[idx], linewidth=3, linestyle='-',
                     label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")

        except np.linalg.LinAlgError:
            print(f"Polynomial fit failed for windspeed bin {avg_windspeed:.1f} m/s")

# Formatting
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
#correlation for 2nd order polynomial

windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]
colors = ['blue', 'orange', 'green', 'red']  # Colors must match bins

# Ensure correct bin edges (10 bins from 2 to 10 µm)
bin_edges = np.linspace(2, 10, 11)  # 11 edges to create 10 bins
bin_widths = np.diff(bin_edges)  # Compute bin widths (should be 10 values)

# Store results
avg_windspeeds = []
total_concentrations = []

# Compute total concentration (cm⁻³) by integrating over size bins
for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:  # Ensure bin has data
        avg_windspeed = np.mean(mean_windspeeds[idx])  # Average windspeed for this bin

        # Convert using correct bin widths (integrate dN/dD)
        avg_concentration_per_leg = [np.sum(dist * bin_widths) for dist in grouped_distributions[idx]]
        avg_concentration = np.mean(avg_concentration_per_leg)  # Average over all legs in this bin

        avg_windspeeds.append(avg_windspeed)
        total_concentrations.append(avg_concentration)

# Convert to numpy arrays for fitting
windspeed_values = np.array(avg_windspeeds)
total_concentrations = np.array(total_concentrations)

# Fit a second-order polynomial (quadratic)
poly_coeffs = np.polyfit(windspeed_values, total_concentrations, 2)
poly_fit_func = np.poly1d(poly_coeffs)  # Creates polynomial function

# Compute R² value
y_fit_values = poly_fit_func(windspeed_values)  # Predicted values from the polynomial
residuals = total_concentrations - y_fit_values
ss_res = np.sum(residuals**2)
ss_tot = np.sum((total_concentrations - np.mean(total_concentrations))**2)
r_squared = 1 - (ss_res / ss_tot)

# Plot Wind Speed vs. Total Droplet Concentration
plt.figure(figsize=(8, 6))
for idx in range(len(windspeed_bins)):
    plt.scatter(windspeed_values[idx], total_concentrations[idx], 
                color=colors[idx], s=100, edgecolor='black', zorder=3)

# Generate smooth curve for the polynomial fit
x_fit = np.linspace(min(windspeed_values), max(windspeed_values), 100)
y_fit = poly_fit_func(x_fit)
plt.plot(x_fit, y_fit, 'r-', label=f'Fit: y = {poly_coeffs[0]:.3f}x² + {poly_coeffs[1]:.3f}x + {poly_coeffs[2]:.3f}, R² = {r_squared:.2f}')

# Labels & Titles
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=16, fontweight='bold')
plt.ylabel("Total Wind Speed Bin Concentration (cm$^{-3}$)", fontsize=16, fontweight='bold')
plt.title("Wind Speed and Total Concentration Correlation", fontsize=16, fontweight='bold')

# Legend
legend_labels = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[idx], markersize=10, 
               label=f"{windspeed_values[idx]:.1f} m/s") for idx in range(len(windspeed_bins))
]
plt.legend(handles=legend_labels + [plt.Line2D([0], [0], color='red', label=f'Fit: y = {poly_coeffs[0]:.3f}x² + {poly_coeffs[1]:.3f}x + {poly_coeffs[2]:.3f}, R² = {r_squared:.2f}')], 
           title="Wind Speed Bins", title_fontsize=14, fontsize=12)

# Final Plot Settings
plt.tight_layout()
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.show()

# Print Results
print(f"Quadratic Fit Coefficients: a={poly_coeffs[0]:.3f}, b={poly_coeffs[1]:.3f}, c={poly_coeffs[2]:.3f}")
print(f"R² value: {r_squared:.2f}")

# %%
#Trying ambient wind speed relationship 


# Define wind speed bins
windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]

# Store binned distributions
grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

missing_windspeed_count = 0

# Use bin_center (ambient size distributions) instead of common_bins
bin_center = np.array(bin_center)

# Use already fitted exponential size distributions for ambient
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

    # Use the already fitted size distribution (DO NOT FIT AGAIN)
    size_dist = n0 * np.exp(-bin_center / D)  # Use existing (n0, D)

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

        plt.plot(bin_center, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linewidth=2.5)

plt.yscale('log')
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=16, fontweight="bold")
plt.xlabel("Ambient Bin Centers Diameter (μm)", fontsize=16, fontweight="bold")
plt.title('Ambient Size Distributions Binned by Average Wind Speed', fontweight='bold', fontsize=17)
plt.legend(title=r"Average wind speed m s$^{-1}$")
plt.tight_layout()
plt.ylim(1e-4, 10**0)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.show()

total_legs = sum(len(group) for group in grouped_distributions.values())
print(f"Total number of legs plotted: {total_legs}")

# %%
#fitting an exponential to ambient wind speed bins


# Define exponential function
def fit_function(x, n0, D):
    return n0 * np.exp(-x / D)

# Define windspeed colors (ensure order matches bins)
windspeed_colors = ['blue', 'orange', 'green', 'red']

# Store fit results
fit_results = {}

plt.figure(figsize=(10, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        
        concentrations_array = np.array(grouped_distributions[idx])
        
        avg_concentration = np.mean(concentrations_array, axis=0)
        
        # Ensure no zero or negative values (set small positive threshold)
        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)
        
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])

        try:
            # Fit an exponential function to the binned ambient size distributions
            popt, _ = curve_fit(fit_function, bin_center, avg_concentration, p0=(1, 5), maxfev=5000)
            n0_fit, D_fit = popt

            # Store results
            fit_results[idx] = {'n0': n0_fit, 'D': D_fit, 'avg_windspeed': avg_windspeed, 'num_legs': num_legs}

            # Generate the fitted curve
            x_fit = np.linspace(min(bin_center), max(bin_center), 100)
            y_fit = fit_function(x_fit, *popt)

            # Plot the fitted exponential curve
            plt.plot(x_fit, y_fit, color=windspeed_colors[idx], linewidth=3, linestyle='-',
                     label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")

        except RuntimeError:
            print(f"Exponential fit failed for windspeed bin {avg_windspeed:.1f} m/s")

# Formatting
plt.ylabel(r'Number concentration (cm$^{-3}$ µm$^{-1}$)', fontweight='bold', fontsize=18)
plt.xlabel('Ambient Bin Centers Diameter (µm)', fontweight='bold', fontsize=18)
plt.yscale('log')
plt.ylim(10**-4, 10**0)
plt.title('Below Cloud Base January - June 2022\nFitted Ambient Size Distributions Binned by Average Windspeed', 
          fontweight='bold', fontsize=17)
plt.legend(title=r"Wind speed bins (m s$^{-1}$)", title_fontsize=15, fontsize=13, frameon=True, prop={'weight': 'bold'})
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# %%
#ambient regression 


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define windspeed bins and corresponding colors
windspeed_bins = [(0, 3), (3.001, 6), (6.001, 8), (8.001, np.inf)]
colors = ['blue', 'orange', 'green', 'red']  # Colors must match bins

# Ensure correct bin widths using bin_log (ambient)
bin_widths = np.array(bin_log)  # Convert from cm⁻³ µm⁻¹ to cm⁻³

# Store results
avg_windspeeds = []
total_concentrations = []

# Compute total concentration (cm⁻³) by integrating over size bins
for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:  # Ensure bin has data
        avg_windspeed = np.mean(mean_windspeeds[idx])  # Average windspeed for this bin

        # Convert using correct bin widths (integrate dN/dD)
        avg_concentration_per_leg = [np.sum(dist * bin_widths) for dist in grouped_distributions[idx]]
        avg_concentration = np.mean(avg_concentration_per_leg)  # Average over all legs in this bin

        avg_windspeeds.append(avg_windspeed)
        total_concentrations.append(avg_concentration)

# Convert to numpy arrays for fitting
windspeed_values = np.array(avg_windspeeds)
total_concentrations = np.array(total_concentrations)

# ✅ Define linear regression function
def linear_model(x, m, b):
    return m * x + b

# ✅ Perform linear regression
popt, _ = curve_fit(linear_model, windspeed_values, total_concentrations)
m_fit, b_fit = popt

# ✅ Compute R² value
residuals = total_concentrations - linear_model(windspeed_values, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((total_concentrations - np.mean(total_concentrations))**2)
r_squared = 1 - (ss_res / ss_tot)

# ✅ Plot Wind Speed vs. Total Droplet Concentration
plt.figure(figsize=(8, 6))
for idx in range(len(windspeed_bins)):
    plt.scatter(windspeed_values[idx], total_concentrations[idx], 
                color=colors[idx], s=100, edgecolor='black', zorder=3)

# Generate linear fit
x_fit = np.linspace(min(windspeed_values), max(windspeed_values), 100)
y_fit = linear_model(x_fit, *popt)
plt.plot(x_fit, y_fit, 'r-', label=f'Fit: y = {m_fit:.3f}x + {b_fit:.3f}, R² = {r_squared:.2f}')

# Labels & Titles
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=16, fontweight='bold')
plt.ylabel("Total Wind Speed Bin Concentration (cm$^{-3}$)", fontsize=16, fontweight='bold')
plt.title("Wind Speed and Total Concentration Correlation (Ambient)", fontsize=16, fontweight='bold')

# Legend
legend_labels = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[idx], markersize=10, 
               label=f"{windspeed_values[idx]:.1f} m/s") for idx in range(len(windspeed_bins))
]
plt.legend(handles=legend_labels + [plt.Line2D([0], [0], color='red', label=f'Fit: y = {m_fit:.3f}x + {b_fit:.3f}, R² = {r_squared:.2f}')], 
           title="Wind Speed Bins", title_fontsize=14, fontsize=12)

# ✅ Final Plot Settings
plt.tight_layout()
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.show()

# ✅ Print Results
print(f"Slope (m): {m_fit:.3f}")
print(f"Intercept (b): {b_fit:.3f}")
print(f"R² value: {r_squared:.2f}")

# %%
#total mass against wind speed 

grouped_mass_values = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds_mass = {i: [] for i in range(len(windspeed_bins))}

# Loop through filtered mass data and match it to wind speed bins
for mass_entry in filtered_dry_mass_inf:
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
plt.title("CAS Below Cloud Base January - June 2022", fontsize=16, fontweight='bold')

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

# %%
