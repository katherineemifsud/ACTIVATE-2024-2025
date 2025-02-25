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
plt.xlabel("Deliquesed Diameter (μm)", fontsize=12, fontweight="bold")
plt.ylabel(r" CAS Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=12, fontweight="bold")
plt.yscale("log")
plt.xticks(fontweight="bold", fontsize=10)
plt.yticks(fontweight="bold", fontsize=10)
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
plt.xlabel("Deliquesed Diameter (μm)", fontsize=12, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=12, fontweight="bold")
plt.yscale("log")
plt.xticks(fontweight="bold", fontsize=10)
plt.yticks(fontweight="bold", fontsize=10)
plt.title("Below Cloud Base January - June 2022\n Exponential Fit to Ambient Size Distributions", fontsize=14, fontweight="bold")
plt.show()

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
    plt.xlabel("Deliquesced Diameter (μm)", fontsize=12, fontweight="bold")
    plt.ylabel("Clear Mean Droplet Concentration (/cm³/μm)", fontsize=12, fontweight="bold")
    plt.title(f"Ambient Size Distribution - {selected_date}\nStart: {selected_start} | Stop: {selected_stop}", 
              fontsize=12, fontweight="bold")
    plt.legend()
    plt.yscale('log')  
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Show the plot
    plt.show()

    # Print selected leg details
    print(f"Selected Date: {selected_date}")
    print(f"Start Time: {selected_start}, Stop Time: {selected_stop}")
    print(f"Intercept (n0): {n0:.2e}, E-folding Diameter (D): {D:.2f} μm")

else:
    print(f"Leg not found or missing fitted parameters for {selected_date} with start {selected_start} and stop {selected_stop}.")

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
plt.ylabel('Frequency', fontsize=15, fontweight='bold')
plt.title('Leg average RH January - June 2022', fontweight='bold', fontsize=16)
plt.grid(True)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
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
plt.ylabel('Frequency', fontsize=15, fontweight='bold')
plt.title('Applying the growth factor equation to RH mean values', fontweight='bold', fontsize=16)
plt.grid(True)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
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
common_bins = np.linspace(2, 50, 25)  # Adjust bin range and count as needed

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
plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=12, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=12, fontweight="bold")
plt.yscale("log")
plt.xticks(fontweight="bold", fontsize=10)
plt.yticks(fontweight="bold", fontsize=10)
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
common_bins = np.linspace(2, 50, 25)

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
plt.xticks(fontweight="bold", fontsize=10)
plt.yticks(fontweight="bold", fontsize=10)
plt.title("Below Cloud Base January - June 2022\n Raw Size Distributions", fontsize=14, fontweight="bold")
plt.legend()
plt.show()
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

plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=12, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=12, fontweight="bold")
plt.yscale("log")
plt.ylim(1e-15, 1e3)
plt.xticks(fontweight="bold", fontsize=10)
plt.yticks(fontweight="bold", fontsize=10)
plt.title("Below Cloud Base January - June 2022\n Fitted Dry Size Distributions", fontsize=14, fontweight="bold")
plt.show()
print(f"Total successful dry exponential fits: {len(dry_exponential_fits)}")

# %%
#Scatterplot of ambient slope versus ambient intercept

ambient_slopes = []
ambient_intercepts = []

for key, entry in ambient_fits_dict.items():
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

for key, entry in ambient_fits_dict.items():
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

rho_salt = 1000  # kg/m³

def calculate_mass(N0, D):
    N0_m4 = N0 * 10**6  # Convert cm⁻³ to m⁻³

    integrand = lambda d: np.exp(-d / D) * (d * 1e-6)**3  # Convert µm³ → m³
    mass_integral, _ = quad(integrand, 2, np.inf)  # Integrate from 2 µm to ∞

    return (np.pi / 6) * rho_salt * N0_m4 * mass_integral  

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

# %%
dry_slopes = []
dry_intercepts = []

for entry in dry_exponential_fits:
    n0 = entry['Dry_Intercept_n0']  # Dry intercept
    D = entry['Dry_E_folding_D']    # Dry slope (e-folding diameter)

    # Store values
    dry_intercepts.append(n0)
    dry_slopes.append(D)

# Convert to Pandas DataFrame for easy manipulation
df_dry = pd.DataFrame({
    'Dry_Intercept_N0': dry_intercepts,
    'Dry_Slope_D': dry_slopes
})

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df_dry['Dry_Slope_D'], df_dry['Dry_Intercept_N0'], alpha=0.6, color='red')

# Labels and title
plt.xlabel('Dry Slope (E-folding Diameter D)', fontsize=14, fontweight='bold')
plt.ylabel('Dry Intercept (N₀)', fontsize=14, fontweight='bold')
plt.title('Below Cloud Base January - June 2022\n Dry Slope vs. Dry Intercept', fontsize=14, fontweight='bold')

plt.yscale('log')
plt.xscale('log')
plt.xlim(10**-0.3, 10**1.1)  # Set min/max for x-axis
plt.ylim(10**-1.5, 10**2)
plt.show()
# %%
