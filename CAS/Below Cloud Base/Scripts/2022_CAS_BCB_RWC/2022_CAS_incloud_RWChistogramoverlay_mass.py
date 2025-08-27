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
import matplotlib.colors as mcolors
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
import matplotlib.patheffects as path_effects
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from scipy.stats import stats

#%%
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
            if df_sum[col].dtype == 'O': 
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
            if df_legs[col].dtype == 'O': 
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
        reverse=False 
    )

    print(f"Processing {date}... Found files: {file_paths}")

    run = 1
    dfs_for_date = []

    for file_path in file_paths:
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
            df_2DS = pd.read_csv(
                file_path, 
                skiprows=header_row, 
                quoting=csv.QUOTE_NONE,
                engine='python'
            )

            df_2DS.columns = df_2DS.columns.str.strip('"')
            print(f"Columns for {file_path}: {df_2DS.columns[:10]}")

            df_2DS.replace([-9999, -9999.0], 0, inplace=True)
            for col in df_2DS.select_dtypes(include=['object']).columns:
                df_2DS[col] = df_2DS[col].str.strip('"')

            dfs_for_date.append(df_2DS)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")


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

print(f"Total dates processed: {len(twoDS)}")
# %%
#Import humidity data. 
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
            CAS.append(df_CAS)

        run = run+1 

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

   
    CAS_flight['Time_mid'] = pd.to_numeric(CAS_flight['Time_mid'], errors='coerce')
    twoDS_flight['Time_Start'] = pd.to_numeric(twoDS_flight['Time_Start'], errors='coerce')

    CAS_times = CAS_flight['Time_mid'].values
    CAS_lwc = CAS_flight['LWC_CAS'].values
    CAS_bins = {f'CAS_Bin{bin_label:02d}': CAS_flight[f'CAS_Bin{bin_label:02d}'].values for bin_label in range(12, 30)}

    TwoDS_times = twoDS_flight['Time_Start'].values

    for k in range(len(ACB_start)):
        start_time = ACB_start[k]
        end_time = ACB_stop[k]

        CAS_indices_in_range = np.where((CAS_times >= start_time) & (CAS_times <= end_time))[0]

        for CAS_idx in zip(CAS_indices_in_range):
            lwc_val = CAS_lwc[CAS_idx]

            if lwc_val >= 0.01:
                calc_entry = {
                    'Date': date,
                    'Time': CAS_times[CAS_idx],
                    'BCB_start': start_time,
                    'BCB_stop': end_time,
                    'CWC': lwc_val,
                }

                for bin_label in range(12, 30):
                    bin_key = f'Bin{bin_label}_concentration'
                    calc_entry[bin_key] = CAS_bins[f'CAS_Bin{bin_label:02d}'][CAS_idx]

                in_cloud_concentrations.append(calc_entry)

#%%
#adding BCT and ACB legs together in a combined dictionary 
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

    CAS_flight['Time_mid'] = pd.to_numeric(CAS_flight['Time_mid'], errors='coerce')

    CAS_times = CAS_flight['Time_mid'].values
    CAS_lwc = CAS_flight['LWC_CAS'].values
    CAS_bins = {f'CAS_Bin{bin_label:02d}': CAS_flight[f'CAS_Bin{bin_label:02d}'].values for bin_label in range(12, 30)}

    combined_legs = [
        (ACB_start, ACB_stop),
        (BCT_start, BCT_stop)
    ]

    for leg_start, leg_stop in combined_legs:
        for k in range(len(leg_start)):
            start_time = leg_start[k]
            end_time = leg_stop[k]

            CAS_indices_in_range = np.where((CAS_times >= start_time) & (CAS_times <= end_time))[0]

            for CAS_idx in CAS_indices_in_range:
                lwc_val = CAS_lwc[CAS_idx]

                if lwc_val >= 0.01:  # Adjust LWC threshold as needed
                    calc_entry = {
                        'Date': date,
                        'Time': CAS_times[CAS_idx],
                        'Leg_start': start_time,
                        'Leg_stop': end_time,
                        'CWC': lwc_val  # Cloud water content
                    }

                    for bin_label in range(12, 30):
                        bin_key = f'Bin{bin_label}_concentration'
                        calc_entry[bin_key] = CAS_bins[f'CAS_Bin{bin_label:02d}'][CAS_idx]

                    
                    in_cloud_concentrations.append(calc_entry)

print(f"Number of in-cloud entries: {len(in_cloud_concentrations)}")
print(f"First 5 entries: {in_cloud_concentrations[:5]}")


#%%
# This code calculates total concentration in cm³
in_cloud_concentrations = []

for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict = leg_data[i]

    ACB_start = leg_dict['LegIndex_03']['StartTimes']
    ACB_stop = leg_dict['LegIndex_03']['StopTimes']
    BCT_start = leg_dict['LegIndex_04']['StartTimes']
    BCT_stop = leg_dict['LegIndex_04']['StopTimes']

    CAS_flight = CAS[i]

    
    CAS_flight['Time_mid'] = pd.to_numeric(CAS_flight['Time_mid'], errors='coerce')

    CAS_times = CAS_flight['Time_mid'].values
    CAS_lwc = CAS_flight['LWC_CAS'].values
    CAS_bins = {f'CAS_Bin{bin_label:02d}': CAS_flight[f'CAS_Bin{bin_label:02d}'].values for bin_label in range(12, 30)}

    bin_widths = [bin_log[bin_label - 12] for bin_label in range(12, 30)]

    all_legs_start = ACB_start + BCT_start
    all_legs_stop = ACB_stop + BCT_stop

    for k in range(len(all_legs_start)):
        start_time = all_legs_start[k]
        end_time = all_legs_stop[k]

        CAS_indices_in_range = np.where((CAS_times >= start_time) & (CAS_times <= end_time))[0]

        for CAS_idx in CAS_indices_in_range:
            lwc_val = CAS_lwc[CAS_idx]

          
            if lwc_val >= 0.01:
                total_concentration = sum(
                    np.nan_to_num(CAS_bins[f'CAS_Bin{bin_label:02d}'][CAS_idx]) * bin_width
                    for bin_label, bin_width in zip(range(12, 30), bin_widths)
                )

                
                calc_entry = {
                    'Date': date,
                    'Time': CAS_times[CAS_idx],
                    'Leg_start': start_time,
                    'Leg_stop': end_time,
                    'CWC': lwc_val,
                    'Total_Concentration': total_concentration  # Units: cm³
                }

                
                in_cloud_concentrations.append(calc_entry)

print(f"Number of in-cloud entries: {len(in_cloud_concentrations)}")
print(f"Sample entries: {in_cloud_concentrations[:5]}")
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

    ACB_start = leg_dict['LegIndex_03']['StartTimes']
    ACB_stop = leg_dict['LegIndex_03']['StopTimes']
    BCT_start = leg_dict['LegIndex_04']['StartTimes']
    BCT_stop = leg_dict['LegIndex_04']['StopTimes']
    all_legs_start = ACB_start + BCT_start
    all_legs_stop = ACB_stop + BCT_stop

    twoDS_flight = twoDS[i]
    twoDS_flight['Time_Start'] = pd.to_numeric(twoDS_flight['Time_Start'], errors='coerce')

    twoDS_times = twoDS_flight['Time_Start'].values
    twoDS_lwc = twoDS_flight['LWC_2DS'].values
    twoDS_bins = {f'dNdlogD_liquid_{bin_label:03d}_2DS': twoDS_flight[f'dNdlogD_liquid_{bin_label:03d}_2DS'].values
                  for bin_label in range(6, 129)}

    for k in range(len(all_legs_start)):
        start_time = all_legs_start[k]
        end_time = all_legs_stop[k]

        twoDS_indices_in_range = np.where((twoDS_times >= start_time) & (twoDS_times <= end_time))[0]

        for twoDS_idx in twoDS_indices_in_range:
            lwc_val = twoDS_lwc[twoDS_idx]

            if lwc_val >= 0.00001:  # LWC threshold (0.01 g/m³ = 1e-5 kg/m³)
                
                total_concentration = sum(
                    np.nan_to_num(twoDS_bins[f'dNdlogD_liquid_{bin_label:03d}_2DS'][twoDS_idx]) * log_width
                    for bin_label, log_width in zip(range(6, 129), twoDS_logg)
                )


                total_concentration /= 1e6  # /m³ to /cm³


                rain_entry = {
                    'Date': date,
                    'Time': twoDS_times[twoDS_idx],
                    'Leg_start': start_time,
                    'Leg_stop': end_time,
                    'LWC': lwc_val,  
                    'Total_Concentration': total_concentration 
                }

                rain_concentrations.append(rain_entry)

print(f"Number of rain entries: {len(rain_concentrations)}")
print(f"First 5 entries: {rain_concentrations[:5]}")


# %%
# Convert LWC to g/m³ and N_liquid to /cm³
for entry in rain_concentrations:
    entry['LWC'] = entry['LWC'] * 1e3  # kg/m³ to g/m³

print("Sample entries after unit conversion:")
for sample in rain_concentrations[:5]:
    print(sample)

# %%
# Convert Bin_Lower and Bin_Upper from µm to meters (once, since they are constant)
Bin_Lower_m = [lower / 1e6 for lower in Bin_Lower]  # Convert µm to m
Bin_Upper_m = [upper / 1e6 for upper in Bin_Upper]  # Convert µm to m
Bin_Centers_m = [(lower + upper) / 2 for lower, upper in zip(Bin_Lower_m, Bin_Upper_m)]  # Bin centers in meters
Bin_Centers_Cubed = [center**3 for center in Bin_Centers_m] 
print("Cubed Bin Centers (in m³):")
for i, (center, cubed) in enumerate(zip(Bin_Centers_m, Bin_Centers_Cubed), start=1):
    print(f"Bin {i}: Center = {center:.6e} m, Center³ = {cubed:.6e} m³")



# %%
#calculating rain water content 
rho_water = 1e3 # Density of water in g/m³
pi_over_6 = np.pi / 6
rain_water_content = []

for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict = leg_data[i]

    ACB_start = leg_dict['LegIndex_03']['StartTimes']
    ACB_stop = leg_dict['LegIndex_03']['StopTimes']
    BCT_start = leg_dict['LegIndex_04']['StartTimes']
    BCT_stop = leg_dict['LegIndex_04']['StopTimes']
    all_legs_start = ACB_start + BCT_start
    all_legs_stop = ACB_stop + BCT_stop

    twoDS_flight = twoDS[i]
    twoDS_flight['Time_Start'] = pd.to_numeric(twoDS_flight['Time_Start'], errors='coerce')

    twoDS_times = twoDS_flight['Time_Start'].values
    twoDS_bins = {f'dNdlogD_liquid_{bin_label:03d}_2DS': twoDS_flight[f'dNdlogD_liquid_{bin_label:03d}_2DS'].values
                  for bin_label in range(6, 129)}

    for k in range(len(all_legs_start)):
        start_time = all_legs_start[k]
        end_time = all_legs_stop[k]

        twoDS_indices_in_range = np.where((twoDS_times >= start_time) & (twoDS_times <= end_time))[0]

        for twoDS_idx in twoDS_indices_in_range:
            lwc_val = twoDS_flight['LWC_2DS'].iloc[twoDS_idx]
            N_liquid_total = 0

            if lwc_val >= 0.00001:  # LWC threshold (0.01 g/m³ = 1e-5 kg/m³)
            
                for bin_label in (range(6, 129)):
                    bin_column = f'dNdlogD_liquid_{bin_label:03d}_2DS'
                    if bin_column in twoDS_flight.columns:
                        N_bin = twoDS_flight[bin_column].iloc[twoDS_idx]  # Raw bin value in /m³
                        
                        N_dD = (N_bin * twoDS_logg[bin_label - 6])
                        
                        N_liquid_total += N_dD * Bin_Centers_Cubed[bin_label - 6]


                RWC = pi_over_6 * rho_water * N_liquid_total # kg/m³

             

                rain_water_content.append({
                    'Date': date,
                    'Time': twoDS_times[twoDS_idx],
                    'Leg_start': start_time,
                    'Leg_stop': end_time,
                    'LWC': lwc_val,
                    'RWC': RWC

                })

print(f"Number of RWC entries: {len(rain_water_content)}")
print(f"First 5 entries: {rain_water_content[:5]}")
#%%
# convert RWC to g/m³ 
for entry in rain_water_content:
    entry['RWC'] = entry['RWC'] * 1e3  # kg/m³ to g/m³
    entry['LWC'] = entry['LWC'] * 1e3  # kg/m³ to g/m³
#%%
#add RWC and CWC for total LWC
total_liquid_water = []

for rwc_entry in rain_water_content:  
    matching_time = rwc_entry['Time']
    matching_date = rwc_entry['Date']

    matching_cwc = next((entry for entry in in_cloud_concentrations if entry['Time'] == matching_time and entry['Date'] == matching_date), None)

    if matching_cwc:
        cwc_val = matching_cwc['CWC'] 
        rwc_val = rwc_entry['RWC'] 
        total_liquid = cwc_val + rwc_val

        total_liquid_water.append({
            'Date': matching_date,
            'Time': matching_time,
            'Leg_start': rwc_entry['Leg_start'],
            'Leg_stop': rwc_entry['Leg_stop'],
            'CWC': cwc_val,
            'RWC': rwc_val,
            'Total_Liquid_Water': total_liquid  
        })

print(f"Number of total liquid water entries: {len(total_liquid_water)}")
print(f"First 5 entries: {total_liquid_water[:5]}")

#%%
#add the Nc + Nr for total concentration
total_combined_concentration = []

for in_cloud_entry in in_cloud_concentrations: 
    matching_time = in_cloud_entry['Time']
    matching_date = in_cloud_entry['Date']

    matching_rain = next((entry for entry in rain_concentrations if entry['Time'] == matching_time and entry['Date'] == matching_date), None)
    
    
    if matching_rain:
        rain_val = matching_rain['Total_Concentration']
        inc_val = in_cloud_entry['Total_Concentration'] 
        combined_conc = inc_val + rain_val

        total_combined_concentration.append({
            'Date': matching_date,
            'Time': matching_time,
            'Leg_start': matching_rain['Leg_start'],
            'Leg_stop': matching_rain['Leg_stop'],
            'In_Cloud_Concentration': inc_val,
            'Rain_Concentration': rain_val,
            'Total_Combined_Concentration': combined_conc 
        })

print(f"Number of total combined concentration entries: {len(total_combined_concentration)}")
print(f"First 5 entries: {total_combined_concentration[:5]}")

#%% 
concentration = [entry['Total_Combined_Concentration'] for entry in total_combined_concentration]
total_liquid_water_values = [entry['Total_Liquid_Water'] for entry in total_liquid_water]  
rain_water_content_values = [entry['RWC'] for entry in total_liquid_water]  

rwc_percentage = []
for rwc, total in zip(rain_water_content_values, total_liquid_water_values):
    if total > 0:
        rwc_percentage.append((rwc / total) * 100)
    else:
        rwc_percentage.append(0) 
bins = 100  
plt.figure(figsize=(8, 6))
hist, xedges, yedges, img = plt.hist2d(concentration, total_liquid_water_values, bins=bins, 
                                       weights=rwc_percentage, cmap='RdBu_r', cmin=1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc /cm³ (log scale)', fontsize=16, fontweight='bold')
plt.ylabel('LWC g/m³ (log scale)', fontsize=16, fontweight='bold')
plt.title('RWC Percentage of Total Liquid Water', fontsize=18, fontweight='bold')
cbar = plt.colorbar(img)
cbar.set_label("Rainwater % of Total Liquid Water", fontsize=14)  
plt.grid(which="both", linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.show()

# %%
# Create histogram bins
bins = 100
counts, xedges, yedges = np.histogram2d(concentration, total_liquid_water_values, bins=bins)
sum_rwc, _, _ = np.histogram2d(concentration, total_liquid_water_values, bins=bins, weights=rwc_percentage)
mean_rwc = np.divide(sum_rwc, counts, out=np.zeros_like(sum_rwc), where=counts > 0)

plt.figure(figsize=(8, 6))
img = plt.pcolormesh(xedges, yedges, mean_rwc.T, cmap='RdBu_r', vmin=1, vmax=100)
cbar = plt.colorbar(img)
cbar.set_label("RWC/LWC (%)", fontsize=14)
plt.ylim(10**-2, 10**0.2) 
plt.xscale('log')
plt.yscale('log')
plt.tick_params(axis='both', which='major', labelsize=16, width=3, length=8)  # Major ticks
plt.tick_params(axis='both', which='minor', labelsize=14, width=2, length=5)
plt.xlabel('Nr+Nc /cm³', fontsize=16, fontweight='bold')
plt.ylabel('LWC g/m³', fontsize=16, fontweight='bold')
plt.title('CAS in-cloud January - June 2022', fontsize=18, fontweight='bold')
plt.grid(which="both", linestyle='--', linewidth=0.5, alpha=0.7)
plt.show()

#%%
concentration = np.array([entry['Total_Combined_Concentration'] for entry in total_combined_concentration])
total_liquid_water_values = np.array([entry['Total_Liquid_Water'] for entry in total_liquid_water])
rain_water_content_values = np.array([entry['RWC'] for entry in total_liquid_water])
rwc_percentage = np.divide(rain_water_content_values, total_liquid_water_values, 
                           out=np.full_like(rain_water_content_values, np.nan), where=total_liquid_water_values > 0) * 100  
num_bins = 5
x_bins = np.logspace(np.log10(min(concentration)), np.log10(max(concentration)), num_bins)
y_bins = np.logspace(np.log10(min(total_liquid_water_values)), np.log10(max(total_liquid_water_values)), num_bins)
counts, xedges, yedges = np.histogram2d(concentration, total_liquid_water_values, bins=[x_bins, y_bins])
sum_rwc, _, _ = np.histogram2d(concentration, total_liquid_water_values, bins=[x_bins, y_bins], weights=rwc_percentage)
mean_rwc = np.divide(sum_rwc, counts, out=np.full_like(sum_rwc, np.nan), where=counts > 0)  
masked_rwc = np.ma.masked_where(np.isnan(mean_rwc), mean_rwc)
cmap = plt.get_cmap('RdBu_r')
cmap.set_bad(color='gray') 
plt.figure(figsize=(8, 6))
norm = mcolors.Normalize(vmin=1, vmax=100)
img = plt.pcolormesh(xedges, yedges, masked_rwc.T, cmap=cmap, norm=norm, shading='auto')
cbar = plt.colorbar(img)
cbar.set_label("RWC/LWC %", fontsize=20, fontweight='bold')
cbar.ax.tick_params(labelsize=15, width=2, length=5) 
for t in cbar.ax.get_yticklabels():  
    t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=20, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=20, fontweight='bold')
plt.title('CAS (in cloud) \nJanuary-June 2022', fontsize=20, fontweight='bold')
plt.tight_layout()
plt.show()
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
#trying to seperate LWC and RWC 
masked_avg_rwc = np.ma.masked_where(np.isnan(avg_rwc), avg_rwc)

plt.figure(figsize=(8, 6))
img = plt.pcolormesh(
    xedges,
    yedges,
    masked_avg_rwc.T,
    cmap="viridis",
    shading='auto',
    vmin=0, vmax=1
)
gray_mask = np.isnan(avg_rwc)
gray_values = np.full_like(avg_rwc, np.nan)
gray_values[gray_mask] = 1
plt.pcolormesh(
    xedges,
    yedges,
    gray_values.T,
    cmap=mcolors.ListedColormap(["gray"]),
    shading='auto',
    alpha=0.6
)
cbar = plt.colorbar(img)
cbar.set_label("Mean RWC (g m$^{-3}$)", fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=18, width=2, length=5)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight('bold')

plt.xscale('log')
plt.yscale('log')
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.xlabel('Nr+Nc /cm³', fontsize=19, fontweight='bold')
plt.ylabel('LWC g/m³', fontsize=19, fontweight='bold')
plt.title('CAS (in cloud)\nMean RWC\nJanuary–June 2022', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()

#%%
masked_avg_lwc = np.ma.masked_where(np.isnan(avg_lwc), avg_lwc)

plt.figure(figsize=(8, 6))
img = plt.pcolormesh(xedges, yedges, masked_avg_lwc.T, cmap="plasma", shading='auto')

gray_mask = np.isnan(avg_lwc)
gray_values = np.full_like(avg_lwc, np.nan)
gray_values[gray_mask] = 1
plt.pcolormesh(xedges, yedges, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)

cbar = plt.colorbar(img)
cbar.set_label("Mean LWC (g m$^{-3}$)", fontsize=18, fontweight='bold') 
cbar.ax.tick_params(labelsize=18, width=2, length=5)
for t in cbar.ax.get_yticklabels():  
    t.set_fontweight('bold')

plt.xscale('log')
plt.yscale('log')
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.xlabel('Nr+Nc /cm³', fontsize=19, fontweight='bold')
plt.ylabel('LWC g/m³', fontsize=19, fontweight='bold')
plt.title('Mean LWC\nJanuary–June 2022 (CAS in cloud)', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
#%%
#density of observations
num_bins = 5
x_bins = np.logspace(np.log10(1), np.log10(max(concentration)), num_bins)
y_bins = np.logspace(np.log10(min(total_liquid_water_values)), np.log10(max(total_liquid_water_values)), num_bins)
mask = (concentration >= x_min) & (concentration <= x_max) & \
       (total_liquid_water_values >= y_min) & (total_liquid_water_values <= y_max)

filtered_concentration = concentration[mask]

filtered_lwc = total_liquid_water_values[mask]
density_counts, xedges, yedges = np.histogram2d(filtered_concentration, filtered_lwc, bins=[x_bins, y_bins])
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
num_bins = 5
x_bins = np.logspace(np.log10(1), np.log10(np.max(concentration)), num_bins)
y_bins = np.logspace(np.log10(np.min(total_liquid_water_values)), np.log10(np.max(total_liquid_water_values)), num_bins)
density_counts, xedges, yedges = np.histogram2d(
    concentration, 
    total_liquid_water_values, 
    bins=[x_bins, y_bins]
)
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(
    xedges, yedges, density_counts.T,
    cmap="plasma", 
    shading='auto',
    norm=mcolors.LogNorm(vmax=np.max(density_counts) * 1.1)
)
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
plt.figure(figsize=(8, 6))
plt.hist(concentration, bins=x_bins, color="darkred", alpha=0.7, log=True)
plt.xscale('log')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=16, fontweight='bold')
plt.ylabel('Frequency', fontsize=16, fontweight='bold')
plt.title('CAS in-cloud January-June 2022', fontsize=18, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=12, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=12, width=2, length=5)
plt.xlim(10**1, 10**np.ceil(np.log10(np.max(concentration))))
plt.tight_layout()
plt.show()
#%%
cmap = plt.cm.plasma.copy()
cmap.set_bad(color="gray")
masked_counts = np.ma.masked_where(density_counts == 0, density_counts)
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(
    xedges,
    yedges,
    masked_counts.T,
    cmap=cmap,
    shading="auto",
    norm=mcolors.LogNorm(vmax=np.max(density_counts) * 1.1)
)
cbar = plt.colorbar(img)
cbar.set_label("Density of Observations", fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=18, width=2, length=5)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.title('CAS (in-cloud)\n January-June 2022)', fontsize=19, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=12, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=12, width=2, length=5)
plt.tight_layout()
plt.show()
#%%
#printing obs in each box
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(
    xedges,
    yedges,
    masked_counts.T,
    cmap=cmap,
    shading="auto",
    norm=mcolors.LogNorm(vmax=np.max(density_counts) * 1.1)
)
cbar = plt.colorbar(img)
cbar.set_label("Density of Observations", fontsize=20, fontweight='bold')
cbar.ax.tick_params(labelsize=20, width=2, length=5)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.xticks(fontsize=20, fontweight='bold')
plt.yticks(fontsize=20, fontweight='bold')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=20, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=20, fontweight='bold')
plt.title('CAS (in-cloud)\nJanuary-June 2022', fontsize=19, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=16, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=16, width=2, length=5)
x_centers = 0.5 * (xedges[:-1] + xedges[1:])
y_centers = 0.5 * (yedges[:-1] + yedges[1:])
for i, xc in enumerate(x_centers):
    for j, yc in enumerate(y_centers):
        val = density_counts[i, j]
        if val > 0:  
            plt.text(
                xc, yc, int(val),
                ha='center', va='center',
                color='black', fontsize=16, fontweight='bold'
            )

plt.tight_layout()
plt.show()
#%%
#adding the black box to selected region
concentration = np.array([entry['Total_Combined_Concentration'] for entry in total_combined_concentration])
total_liquid_water_values = np.array([entry['Total_Liquid_Water'] for entry in total_liquid_water])
rain_water_content_values = np.array([entry['RWC'] for entry in total_liquid_water])
num_bins = 5
x_bins = np.logspace(np.log10(1), np.log10(max(concentration)), num_bins)
y_bins = np.logspace(np.log10(min(total_liquid_water_values)), np.log10(max(total_liquid_water_values)), num_bins)
sum_rwc, xedges, yedges = np.histogram2d(concentration, total_liquid_water_values, bins=[x_bins, y_bins], weights=rain_water_content_values)
sum_lwc, _, _ = np.histogram2d(concentration, total_liquid_water_values, bins=[x_bins, y_bins], weights=total_liquid_water_values)
counts, _, _ = np.histogram2d(concentration, total_liquid_water_values, bins=[x_bins, y_bins])
avg_rwc = np.divide(sum_rwc, counts, out=np.full_like(sum_rwc, np.nan), where=counts > 0)  
avg_lwc = np.divide(sum_lwc, counts, out=np.full_like(sum_rwc, np.nan), where=counts > 0)  
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
cbar.set_label("RWC / LWC (%)", fontsize=17, fontweight='bold') 
cbar.ax.tick_params(labelsize=19, width=2, length=5) 
for t in cbar.ax.get_yticklabels():  
    t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.title('CAS (in cloud)\n January-June 2022\n RWC as a function of number concentration', fontsize=18, fontweight='bold')
box_x_min, box_x_max = 25.182, 634.143  # Nr+Nc range
box_y_min, box_y_max = 0.041, 0.580 
plt.plot([box_x_min, box_x_max, box_x_max, box_x_min, box_x_min],
         [box_y_min, box_y_min, box_y_max, box_y_max, box_y_min], 
         color='black', linewidth=3) 
plt.tight_layout()
plt.show()
#%%
num_bins = 5
x_bins = np.logspace(np.log10(1), np.log10(max(concentration)), num_bins)
y_bins = np.logspace(np.log10(min(total_liquid_water_values)), np.log10(max(total_liquid_water_values)), num_bins)
sum_rwc, xedges, yedges = np.histogram2d(
    concentration,
    total_liquid_water_values,
    bins=[x_bins, y_bins],
    weights=rain_water_content_values
)
sum_lwc, _, _ = np.histogram2d(
    concentration,
    total_liquid_water_values,
    bins=[x_bins, y_bins],
    weights=total_liquid_water_values
)
counts, _, _ = np.histogram2d(
    concentration,
    total_liquid_water_values,
    bins=[x_bins, y_bins]
)
avg_rwc = np.divide(sum_rwc, counts, out=np.full_like(sum_rwc, np.nan), where=counts > 0)
avg_lwc = np.divide(sum_lwc, counts, out=np.full_like(sum_lwc, np.nan), where=counts > 0)
rwc_lwc_ratio = np.divide(avg_rwc, avg_lwc, out=np.full_like(avg_rwc, np.nan), where=avg_lwc > 0) * 100
masked_rwc_lwc_ratio = np.ma.masked_invalid(rwc_lwc_ratio)
cmap = plt.cm.plasma.copy()
cmap.set_bad(color="gray")
plt.figure(figsize=(8, 6))
norm = mcolors.Normalize(vmin=1, vmax=100)
img = plt.pcolormesh(
    xedges,
    yedges,
    masked_rwc_lwc_ratio.T,
    cmap=cmap,
    norm=norm,
    shading='auto'
)
cbar = plt.colorbar(img)
cbar.set_label("RWC / LWC (%)", fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=19, width=2, length=5)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.title('CAS (in-cloud)\n January-June 2022\n RWC as a function of number concentration', fontsize=18, fontweight='bold')
box_x_min, box_x_max = 25.182, 634.143  # Nr+Nc range
box_y_min, box_y_max = 0.041, 0.580
plt.plot(
    [box_x_min, box_x_max, box_x_max, box_x_min, box_x_min],
    [box_y_min, box_y_min, box_y_max, box_y_max, box_y_min],
    color='black',
    linewidth=3
)
plt.tight_layout()
plt.show()
#%%
#creating my own colorbar
from matplotlib.cm import get_cmap
num_bins = 5
x_bins = np.logspace(np.log10(1), np.log10(max(concentration)), num_bins)
y_bins = np.logspace(np.log10(min(total_liquid_water_values)), np.log10(max(total_liquid_water_values)), num_bins)
sum_rwc, xedges, yedges = np.histogram2d(
    concentration,
    total_liquid_water_values,
    bins=[x_bins, y_bins],
    weights=rain_water_content_values
)
sum_lwc, _, _ = np.histogram2d(
    concentration,
    total_liquid_water_values,
    bins=[x_bins, y_bins],
    weights=total_liquid_water_values
)
counts, _, _ = np.histogram2d(
    concentration,
    total_liquid_water_values,
    bins=[x_bins, y_bins]
)
avg_rwc = np.divide(sum_rwc, counts, out=np.full_like(sum_rwc, np.nan), where=counts > 0)
avg_lwc = np.divide(sum_lwc, counts, out=np.full_like(sum_lwc, np.nan), where=counts > 0)
rwc_lwc_ratio = np.divide(avg_rwc, avg_lwc, out=np.full_like(avg_rwc, np.nan), where=avg_lwc > 0) * 100
masked_rwc_lwc_ratio = np.ma.masked_invalid(rwc_lwc_ratio)
valid_data = rwc_lwc_ratio[~np.isnan(rwc_lwc_ratio)].flatten()
bounds = [0, 2, 5, 10, 20, 40, 60, 80, 100] 
cmap = plt.cm.plasma.copy()
cmap.set_bad(color="gray")
norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N, extend='neither')
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(
    xedges,
    yedges,
    masked_rwc_lwc_ratio.T,
    cmap=cmap,
    norm=norm,
    shading='auto'
)
cbar = plt.colorbar(img, ticks=bounds)
cbar.set_label("RWC / LWC (%)", fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=19, width=2, length=5)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.title('CAS (in-cloud)\nJanuary–June 2022\nRWC as a function of number concentration',
          fontsize=18, fontweight='bold')
x_start_idx = 2
x_end_idx = 4
y_start_idx = 1
y_end_idx = 3
box_x_min = xedges[x_start_idx]
box_x_max = xedges[x_end_idx]
box_y_min = yedges[y_start_idx]
box_y_max = yedges[y_end_idx]
plt.plot(
    [box_x_min, box_x_max, box_x_max, box_x_min, box_x_min],
    [box_y_min, box_y_min, box_y_max, box_y_max, box_y_min],
    color='black',
    linewidth=3
)
plt.tight_layout()
plt.show()
print(f"Black box x-range: {box_x_min:.3f} to {box_x_max:.3f}")
print(f"Black box y-range: {box_y_min:.3f} to {box_y_max:.3f}")
# %%
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

#%%
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
        average_gccn_per_flight[date] = np.nan 

print("Average GCCN per Flight Dictionary:")
for date, avg_gccn in average_gccn_per_flight.items():
    print(f"Date: {date}, Average GCCN: {avg_gccn:.4f}")

#%%
#splitting flights based on high and low average GCCN
gccn_values = np.array(list(average_gccn_per_flight.values()))
#%%
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
df_gccn = pd.DataFrame({
    "GCCN Concentration (cm⁻³)": np.concatenate([high_gccn_values, low_gccn_values]),
    "Flight Type": ["High GCCN"] * len(high_gccn_values) + ["Low GCCN"] * len(low_gccn_values)
})

plt.figure(figsize=(8, 6))
sns.violinplot(x="Flight Type", y="GCCN Concentration (cm⁻³)", data=df_gccn, inner="box", palette=["mediumpurple", "rebeccapurple"], scale="width")
plt.yscale('log')
plt.ylabel("GCCN Concentration (cm⁻³)", fontsize=20, fontweight="bold")
plt.xlabel("GCCN Flight Category", fontsize=20, fontweight="bold")
plt.title("Comparison of High & Low GCCN Flight Categories", fontsize=16, fontweight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tick_params(axis="both", which="major", labelsize=18, width=3, length=8)
plt.tick_params(axis="both", which="minor", labelsize=18, width=2, length=5)
plt.show()
#%%
#Average concentration stats
avg_high_gccn = np.mean(high_gccn_values)
avg_low_gccn = np.mean(low_gccn_values)
num_high_flights = len(high_gccn_values)
num_low_flights = len(low_gccn_values)
print(f"Average High GCCN Flight Concentration: {avg_high_gccn:.4f} cm⁻³")
print(f"Number of High GCCN Flights: {num_high_flights}")

print(f"Average Low GCCN Flight Concentration: {avg_low_gccn:.4f} cm⁻³")
print(f"Number of Low GCCN Flights: {num_low_flights}")

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
num_bins = 5
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
cbar.set_label("RWC / LWC (%)", fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc /cm³', fontsize=16, fontweight='bold')
plt.ylabel('LWC g/m³', fontsize=16, fontweight='bold')
plt.title('High GCCN Flights January-June 2022', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(xedges, yedges, masked_rwc_low.T, cmap="RdBu_r", norm=norm, shading='auto')
plt.pcolormesh(xedges, yedges, gray_values_low.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)
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
#separating rwc and lwc 
masked_avg_rwc_high = np.ma.masked_where(np.isnan(avg_rwc_high), avg_rwc_high)
masked_avg_rwc_low = np.ma.masked_where(np.isnan(avg_rwc_low), avg_rwc_low)
vmin = 0
vmax = 1
plt.figure(figsize=(8, 6))
plt.pcolormesh(xedges, yedges, masked_avg_rwc_high.T, cmap="viridis", shading='auto', vmin=vmin, vmax=vmax)
plt.colorbar(label="Mean RWC (g m$^{-3}$)")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Mean RWC — High GCCN Flights (Jan–Jun 2022)', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
plt.pcolormesh(xedges, yedges, masked_avg_rwc_low.T, cmap="viridis", shading='auto', vmin=vmin, vmax=vmax)
plt.colorbar(label="Mean RWC (g m$^{-3}$)")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Mean RWC — Low GCCN Flights (Jan–Jun 2022)', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
#fixing gray NANs
masked_avg_rwc_high = np.ma.masked_where(np.isnan(avg_rwc_high), avg_rwc_high)
masked_avg_rwc_low = np.ma.masked_where(np.isnan(avg_rwc_low), avg_rwc_low)
vmin = 0
vmax = 1
cmap = plt.cm.viridis.copy()
cmap.set_bad(color='gray')  
plt.figure(figsize=(8, 6))
img_high = plt.pcolormesh(
    xedges, yedges, masked_avg_rwc_high.T,
    cmap=cmap, shading='auto', vmin=vmin, vmax=vmax
)
cbar_high = plt.colorbar(img_high)
cbar_high.set_label("Mean RWC (g m$^{-3}$)", fontsize=20, fontweight='bold')
cbar_high.ax.tick_params(labelsize=20, width=2, length=5)
for t in cbar_high.ax.get_yticklabels():
    t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Mean RWC\nHigh GCCN Flights\n(January–June 2022)', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
img_low = plt.pcolormesh(
    xedges, yedges, masked_avg_rwc_low.T,
    cmap=cmap, shading='auto', vmin=vmin, vmax=vmax
)
cbar_low = plt.colorbar(img_low)
cbar_low.set_label("Mean RWC (g m$^{-3}$)", fontsize=20, fontweight='bold')
cbar_low.ax.tick_params(labelsize=20, width=2, length=5)
for t in cbar_low.ax.get_yticklabels():
    t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Mean RWC\nLow GCCN Flights\n(January–June 2022)', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
masked_avg_lwc_high = np.ma.masked_where(np.isnan(avg_lwc_high), avg_lwc_high)
plt.figure(figsize=(8, 6))
plt.pcolormesh(xedges, yedges, masked_avg_lwc_high.T, cmap="plasma", shading='auto')
plt.colorbar(label="Mean LWC (g m$^{-3}$)")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Mean LWC — High GCCN Flights (Jan–Jun 2022)', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
masked_avg_lwc_low = np.ma.masked_where(np.isnan(avg_lwc_low), avg_lwc_low)
plt.figure(figsize=(8, 6))
plt.pcolormesh(xedges, yedges, masked_avg_lwc_low.T, cmap="plasma", shading='auto')
plt.colorbar(label="Mean LWC (g m$^{-3}$)")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Mean LWC — Low GCCN Flights (Jan–Jun 2022)', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
#creating our own color bar for RWC
valid_data_high = avg_rwc_high[~np.isnan(avg_rwc_high)].flatten()
valid_data_low = avg_rwc_low[~np.isnan(avg_rwc_low)].flatten()
all_valid_rwc = np.concatenate([valid_data_high, valid_data_low])
bounds = [0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8]
#for 2x2 binning scheme
# bounds = [0, 0.005, 0.01, 0.01, 0.02, 0.028, 0.03, 0.038, 0.05]
cmap = plt.cm.viridis.copy()
cmap.set_bad(color='gray') 
norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N, extend='neither')
masked_avg_rwc_high = np.ma.masked_where(np.isnan(avg_rwc_high), avg_rwc_high)
masked_avg_rwc_low = np.ma.masked_where(np.isnan(avg_rwc_low), avg_rwc_low)
plt.figure(figsize=(8, 6))
img_high = plt.pcolormesh(
    xedges, yedges, masked_avg_rwc_high.T,
    cmap=cmap, norm=norm, shading='auto'
)
cbar_high = plt.colorbar(img_high, ticks=bounds)
cbar_high.set_label("Mean RWC (g m$^{-3}$)", fontsize=20, fontweight='bold')
cbar_high.ax.tick_params(labelsize=20, width=2, length=5)
for t in cbar_high.ax.get_yticklabels():
    t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Mean RWC\nHigh GCCN Flights\n(January–June 2022)', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
img_low = plt.pcolormesh(
    xedges, yedges, masked_avg_rwc_low.T,
    cmap=cmap, norm=norm, shading='auto'
)
cbar_low = plt.colorbar(img_low, ticks=bounds)
cbar_low.set_label("Mean RWC (g m$^{-3}$)", fontsize=20, fontweight='bold')
cbar_low.ax.tick_params(labelsize=20, width=2, length=5)
for t in cbar_low.ax.get_yticklabels():
    t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Mean RWC\nLow GCCN Flights\n(January–June 2022)', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
#creating our own colorbar but for LWC 
masked_avg_lwc_high = np.ma.masked_where(np.isnan(avg_lwc_high), avg_lwc_high)
masked_avg_lwc_low = np.ma.masked_where(np.isnan(avg_lwc_low), avg_lwc_low)
valid_data_high = avg_lwc_high[~np.isnan(avg_lwc_high)].flatten()
valid_data_low = avg_lwc_low[~np.isnan(avg_lwc_low)].flatten()
all_valid_lwc = np.concatenate([valid_data_high, valid_data_low])
bounds = [0, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0] 
#bounds for 2x2 binning scheme 
# bounds = [0, 0.04, 0.06, 0.07, 0.09, 0.1, 0.15, 0.18, 0.2, 0.25, 0.3, 0.35, 0.38, 0.4] 
cmap = plt.cm.plasma.copy()
cmap.set_bad(color='gray')
norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N, extend='neither')
plt.figure(figsize=(8, 6))
img_high = plt.pcolormesh(
    xedges, yedges, masked_avg_lwc_high.T,
    cmap=cmap, norm=norm, shading='auto'
)
cbar_high = plt.colorbar(img_high, ticks=bounds)
cbar_high.set_label("Mean LWC (g m$^{-3}$)", fontsize=20, fontweight='bold')
cbar_high.ax.tick_params(labelsize=20, width=2, length=5)
for t in cbar_high.ax.get_yticklabels():
    t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Mean LWC\nHigh GCCN Flights\n(January–June 2022)', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
img_low = plt.pcolormesh(
    xedges, yedges, masked_avg_lwc_low.T,
    cmap=cmap, norm=norm, shading='auto'
)
cbar_low = plt.colorbar(img_low, ticks=bounds)
cbar_low.set_label("Mean LWC (g m$^{-3}$)", fontsize=20, fontweight='bold')
cbar_low.ax.tick_params(labelsize=20, width=2, length=5)
for t in cbar_low.ax.get_yticklabels():
    t.set_fontweight('bold')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Mean LWC\nLow GCCN Flights\n(January–June 2022)', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
#using ratio of high lwc/low lwc and high rwc/low rwc 
ratio_rwc = np.divide(
    avg_rwc_high,
    avg_rwc_low,
    out=np.full_like(avg_rwc_high, np.nan),
    where=avg_rwc_low > 0
)
ratio_lwc = np.divide(
    avg_lwc_high,
    avg_lwc_low,
    out=np.full_like(avg_lwc_high, np.nan),
    where=avg_lwc_low > 0
)
masked_ratio_rwc = np.ma.masked_where(np.isnan(ratio_rwc), ratio_rwc)
masked_ratio_lwc = np.ma.masked_where(np.isnan(ratio_lwc), ratio_lwc)
norm = mcolors.Normalize(vmin=0, vmax=2)
plt.figure(figsize=(8,6))
img = plt.pcolormesh(
    xedges, yedges,
    masked_ratio_rwc.T,
    cmap="RdBu_r",
    norm=norm,
    shading="auto"
)
cbar = plt.colorbar(img)
cbar.set_label("RWC Ratio (High / Low)", fontsize=18, fontweight="bold")
cbar.ax.tick_params(labelsize=16)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"Nr+Nc (cm$^{-3}$)", fontsize=19, fontweight="bold")
plt.ylabel(r"LWC (g m$^{-3}$)", fontsize=19, fontweight="bold")
plt.title("RWC Ratio — High / Low GCCN Flights", fontsize=19, fontweight="bold")
plt.tight_layout()
plt.show()
plt.figure(figsize=(8,6))
img = plt.pcolormesh(
    xedges, yedges,
    masked_ratio_lwc.T,
    cmap="RdBu_r",
    norm=norm,
    shading="auto"
)
cbar = plt.colorbar(img)
cbar.set_label("LWC Ratio (High / Low)", fontsize=18, fontweight="bold")
cbar.ax.tick_params(labelsize=16)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"Nr+Nc (cm$^{-3}$)", fontsize=19, fontweight="bold")
plt.ylabel(r"LWC (g m$^{-3}$)", fontsize=19, fontweight="bold")
plt.title("LWC Ratio — High / Low GCCN Flights", fontsize=19, fontweight="bold")
plt.tight_layout()
plt.show()
#%%
masked_avg_rwc_low = np.ma.masked_where(np.isnan(avg_rwc_low), avg_rwc_low)
masked_avg_lwc_low = np.ma.masked_where(np.isnan(avg_lwc_low), avg_lwc_low)
norm = mcolors.Normalize(vmin=0, vmax=2)
cmap_rwc = plt.cm.viridis.copy()
cmap_rwc.set_bad(color='gray')
cmap_lwc = plt.cm.plasma.copy()
cmap_lwc.set_bad(color='gray')
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(
    xedges, yedges,
    masked_avg_rwc_low.T,
    cmap=cmap_rwc,
    shading='auto',
    norm=norm 
)
cbar = plt.colorbar(img)
cbar.set_label("Mean RWC (g m$^{-3}$)", fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=19, width=2, length=5)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.title('Low GCCN Flights — Mean RWC', fontsize=19, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(
    xedges, yedges,
    masked_avg_lwc_low.T,
    cmap=cmap_lwc,
    shading='auto',
    norm=norm 
)
cbar = plt.colorbar(img)
cbar.set_label("Mean LWC (g m$^{-3}$)", fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=19, width=2, length=5)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.title('Low GCCN Flights — Mean LWC', fontsize=19, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.tight_layout()
plt.show()
#%%
#trying to fix color scale 
ratio_rwc = np.divide(
    avg_rwc_high,
    avg_rwc_low,
    out=np.full_like(avg_rwc_high, np.nan),
    where=avg_rwc_low > 0
)
masked_ratio_rwc = np.ma.masked_where(np.isnan(ratio_rwc), ratio_rwc)
# custom_bounds = [1.2, 1.4, 1.6, 1.7, 2.0, 2.2, 2.4, 3.5, 3.8, 4.0, 4.3, 7.0]
custom_bounds = [1.1, 1.2, 2.0, 2.2, 2.4, 2.7, 3.0, 3.5, 3.8, 7.0]
cmap = plt.cm.viridis.copy()
cmap.set_bad(color='gray')
norm = BoundaryNorm(boundaries=custom_bounds, ncolors=cmap.N)
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(
    xedges, yedges,
    masked_ratio_rwc.T,
    cmap=cmap,
    norm=norm,
    shading="auto"
)
cbar = plt.colorbar(img, ticks=custom_bounds)
cbar.set_label("RWC Ratio (High / Low)", fontsize=19, fontweight="bold")
cbar.ax.tick_params(labelsize=19)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
plt.xscale("log")
plt.yscale("log")
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.xlabel(r"Nr+Nc (cm$^{-3}$)", fontsize=19, fontweight="bold")
plt.ylabel(r"LWC (g m$^{-3}$)", fontsize=19, fontweight="bold")
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.title("CAS (in cloud)\nRWC Ratio High / Low GCCN Flights\nJanuary–June 2022", fontsize=19, fontweight="bold")
plt.tight_layout()
plt.show()
#%%
min_rwc_threshold = 0.003
valid_mask = (avg_rwc_low > min_rwc_threshold) & ~np.isnan(avg_rwc_low) & ~np.isnan(avg_rwc_high)
safe_ratio = np.full_like(avg_rwc_high, np.nan)
safe_ratio[valid_mask] = avg_rwc_high[valid_mask] / avg_rwc_low[valid_mask]
masked_ratio_rwc = np.ma.masked_where(np.isnan(safe_ratio), safe_ratio)
masked_ratio_rwc = np.ma.masked_where(masked_ratio_rwc > 5, masked_ratio_rwc)  # cap extremes
custom_bounds = [0, 0.2, 0.5, 1.0, 1.1, 1.2, 1.3, 1.4, 2.0, 2.2, 2.4, 2.7, 3.0, 3.5]
cmap = plt.cm.viridis.copy()
cmap.set_bad(color='gray')
norm = BoundaryNorm(boundaries=custom_bounds, ncolors=cmap.N)
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(
    xedges, yedges,
    masked_ratio_rwc.T,
    cmap=cmap,
    norm=norm,
    shading="auto"
)
cbar = plt.colorbar(img, ticks=custom_bounds)
cbar.set_label("RWC Ratio (High / Low)", fontsize=19, fontweight="bold")
cbar.ax.tick_params(labelsize=19)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
plt.xscale("log")
plt.yscale("log")
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.xlabel(r"Nr+Nc (cm$^{-3}$)", fontsize=19, fontweight="bold")
plt.ylabel(r"LWC (g m$^{-3}$)", fontsize=19, fontweight="bold")
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.title("CAS (in cloud)\nRWC Ratio High / Low GCCN Flights\nJanuary–June 2022", fontsize=19, fontweight="bold")
plt.tight_layout()
plt.show()
#%%




































































#Calculating mass 
from collections import defaultdict
GCCN_mass_flight_totals = defaultdict(lambda: {'Legs': [], 'Total_GCCN_Mass': 0, 'Leg_Count': 0})

for entry in filtered_dry_mass_inf:
    date = entry['Date']
    start_time = entry['BCB_start']
    stop_time = entry['BCB_stop']
    mass_val = entry['Dry Mass (µg/m³)']

    # Store leg info
    GCCN_mass_flight_totals[date]['Legs'].append({
        'Leg_start': start_time,
        'Leg_stop': stop_time,
        'Leg_GCCN_Mass': mass_val
    })

    # Update flight sums
    GCCN_mass_flight_totals[date]['Total_GCCN_Mass'] += mass_val
    GCCN_mass_flight_totals[date]['Leg_Count'] += 1

GCCN_mass_flight_totals = dict(GCCN_mass_flight_totals)

# --- Step 2: Flight-level averages ---
average_mass_per_flight = {}
for date, flight_data in GCCN_mass_flight_totals.items():
    if flight_data['Leg_Count'] > 0:
        average_mass_per_flight[date] = flight_data['Total_GCCN_Mass'] / flight_data['Leg_Count']
    else:
        average_mass_per_flight[date] = np.nan


#%%
mass_values = np.array(list(average_mass_per_flight.values()))
threshold_mass = np.nanpercentile(mass_values, 50)  # median split

high_GCCN_mass_flights = {d: v for d, v in average_mass_per_flight.items() if v >= threshold_mass}
low_GCCN_mass_flights  = {d: v for d, v in average_mass_per_flight.items() if v < threshold_mass}

print(f"Mass Threshold: Low < {threshold_mass:.2f} µg/m³, High ≥ {threshold_mass:.2f} µg/m³")
print(f"High Mass Flights: {len(high_GCCN_mass_flights)}")
print(f"Low Mass Flights: {len(low_GCCN_mass_flights)}")

#%%
# Add a "Mass_Category" to each second of liquid water data
for lw_entry in total_liquid_water:
    date = lw_entry['Date']
    if date in high_GCCN_mass_flights:
        lw_entry['Mass_Category'] = 'High'
    elif date in low_GCCN_mass_flights:
        lw_entry['Mass_Category'] = 'Low'
    else:
        lw_entry['Mass_Category'] = 'Uncategorized'

#%%
high_rwc = [e['RWC'] for e in total_liquid_water if e['Mass_Category'] == 'High']
low_rwc  = [e['RWC'] for e in total_liquid_water if e['Mass_Category'] == 'Low']
#%%
print("\n=== High GCCN Mass Flights ===")
for date, mass_val in sorted(high_GCCN_mass_flights.items(), key=lambda x: x[1], reverse=True):
    print(f"{date}: {mass_val:.2f} µg/m³")

print("\n=== Low GCCN Mass Flights ===")
for date, mass_val in sorted(low_GCCN_mass_flights.items(), key=lambda x: x[1]):
    print(f"{date}: {mass_val:.2f} µg/m³")

print(f"\nTotal flights: {len(high_GCCN_mass_flights) + len(low_GCCN_mass_flights)}")
print(f"High mass flights: {len(high_GCCN_mass_flights)}")
print(f"Low mass flights: {len(low_GCCN_mass_flights)}")

#%%

high_mass_vals = np.array(list(high_GCCN_mass_flights.values()))
low_mass_vals = np.array(list(low_GCCN_mass_flights.values()))

df_mass = pd.DataFrame({
    "GCCN Mass (µg/m³)": np.concatenate([high_mass_vals, low_mass_vals]),
    "Flight Type": ["High GCCN Mass"] * len(high_mass_vals) + ["Low GCCN Mass"] * len(low_mass_vals)
})

plt.figure(figsize=(8, 6))
sns.violinplot(
    x="Flight Type",
    y="GCCN Mass (µg/m³)",
    data=df_mass,
    inner="box",
    palette=["mediumseagreen", "seagreen"],
    scale="width"
)
plt.yscale('log')
plt.ylabel("GCCN Mass (µg/m³)", fontsize=20, fontweight="bold")
plt.xlabel("GCCN Flight Category", fontsize=20, fontweight="bold")
plt.title("Comparison of High & Low GCCN Mass Flights", fontsize=16, fontweight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tick_params(axis="both", which="major", labelsize=18, width=3, length=8)
plt.tick_params(axis="both", which="minor", labelsize=18, width=2, length=5)
plt.show()
#%%
# Build total_combined_mass just like total_combined_concentration
total_combined_mass = []
for mass_entry in filtered_dry_mass_inf:
    date = mass_entry['Date']
    start = mass_entry['BCB_start']
    stop = mass_entry['BCB_stop']
    mass_val = mass_entry['Dry Mass (µg/m³)']
    matching_liquid = next(
        (entry for entry in total_liquid_water if entry['Date'] == date and 
         entry['Leg_start'] == start and entry['Leg_stop'] == stop),
        None
    )

    if matching_liquid is not None:
        total_combined_mass.append({
            'Date': date,
            'BCB_start': start,
            'BCB_stop': stop,
            'Total_Combined_Mass': mass_val,
            'Total_Liquid_Water': matching_liquid['Total_Liquid_Water'],
            'RWC': matching_liquid['RWC']
        })
#%%
df_rwc = pd.DataFrame({
    "RWC (g m$^{-3}$)": high_rwc + low_rwc,
    "Flight Category": ["High GCCN Mass"] * len(high_rwc) + ["Low GCCN Mass"] * len(low_rwc)
})
plt.figure(figsize=(8,6))
sns.violinplot(
    x="Flight Category", 
    y="RWC (g m$^{-3}$)", 
    data=df_rwc, 
    inner="box", 
    palette=["mediumseagreen", "seagreen"],
    scale="width"
)
plt.yscale("log")
plt.ylabel("Rainwater Content (g m$^{-3}$)", fontsize=18, fontweight="bold")
plt.xlabel("GCCN Mass Category", fontsize=18, fontweight="bold")
plt.title("Per-second RWC Split by High vs Low GCCN Mass Flights", fontsize=16, fontweight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tick_params(axis="both", which="major", labelsize=16, width=3, length=8)
plt.tick_params(axis="both", which="minor", labelsize=16, width=2, length=5)
plt.show()

#%%
# === Split the RWC plots based on flights categorized as high or low GCCN mass ===
high_conc = np.array([e['Total_Combined_Concentration'] for e in total_combined_concentration if e['Date'] in high_GCCN_mass_flights])
high_lwc  = np.array([e['Total_Liquid_Water'] for e in total_liquid_water if e['Date'] in high_GCCN_mass_flights])
high_rwc  = np.array([e['RWC'] for e in total_liquid_water if e['Date'] in high_GCCN_mass_flights])

low_conc = np.array([e['Total_Combined_Concentration'] for e in total_combined_concentration if e['Date'] in low_GCCN_mass_flights])
low_lwc  = np.array([e['Total_Liquid_Water'] for e in total_liquid_water if e['Date'] in low_GCCN_mass_flights])
low_rwc  = np.array([e['RWC'] for e in total_liquid_water if e['Date'] in low_GCCN_mass_flights])
num_bins = 5
x_bins = np.logspace(np.log10(1), np.log10(max(high_conc.max(), low_conc.max())), num_bins)
y_bins = np.logspace(np.log10(min(high_lwc.min(), low_lwc.min())), np.log10(max(high_lwc.max(), low_lwc.max())), num_bins)
sum_rwc_high, xedges, yedges = np.histogram2d(high_conc, high_lwc, bins=[x_bins, y_bins], weights=high_rwc)
sum_lwc_high, _, _           = np.histogram2d(high_conc, high_lwc, bins=[x_bins, y_bins], weights=high_lwc)
counts_high, _, _            = np.histogram2d(high_conc, high_lwc, bins=[x_bins, y_bins])

avg_rwc_high = np.divide(sum_rwc_high, counts_high, out=np.full_like(sum_rwc_high, np.nan), where=counts_high > 0)
avg_lwc_high = np.divide(sum_lwc_high, counts_high, out=np.full_like(sum_lwc_high, np.nan), where=counts_high > 0)
rwc_lwc_ratio_high = np.divide(avg_rwc_high, avg_lwc_high, out=np.full_like(avg_rwc_high, np.nan), where=avg_lwc_high > 0) * 100
masked_rwc_high = np.ma.masked_where(np.isnan(rwc_lwc_ratio_high), rwc_lwc_ratio_high)
sum_rwc_low, _, _ = np.histogram2d(low_conc, low_lwc, bins=[x_bins, y_bins], weights=low_rwc)
sum_lwc_low, _, _ = np.histogram2d(low_conc, low_lwc, bins=[x_bins, y_bins], weights=low_lwc)
counts_low, _, _  = np.histogram2d(low_conc, low_lwc, bins=[x_bins, y_bins])

avg_rwc_low = np.divide(sum_rwc_low, counts_low, out=np.full_like(sum_rwc_low, np.nan), where=counts_low > 0)
avg_lwc_low = np.divide(sum_lwc_low, counts_low, out=np.full_like(sum_lwc_low, np.nan), where=counts_low > 0)
rwc_lwc_ratio_low = np.divide(avg_rwc_low, avg_lwc_low, out=np.full_like(avg_rwc_low, np.nan), where=avg_lwc_low > 0) * 100
masked_rwc_low = np.ma.masked_where(np.isnan(rwc_lwc_ratio_low), rwc_lwc_ratio_low)
plt.figure(figsize=(8, 6))
norm = mcolors.Normalize(vmin=1, vmax=100)
plt.pcolormesh(xedges, yedges, masked_rwc_high.T, cmap="RdBu_r", norm=norm, shading='auto')
plt.colorbar(label="RWC / LWC (%)")
plt.xscale('log'); plt.yscale('log')
plt.xlabel('Nd (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.title('High GCCN Mass Flights January–June 2022', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
plt.pcolormesh(xedges, yedges, masked_rwc_low.T, cmap="RdBu_r", norm=norm, shading='auto')
plt.colorbar(label="RWC / LWC (%)")
plt.xscale('log'); plt.yscale('log')
plt.xlabel('Nd (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.title('Low GCCN Mass Flights January–June 2022', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
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
cbar.set_label("RWC / LWC (%)", fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc /cm³', fontsize=16, fontweight='bold')
plt.ylabel('LWC g/m³', fontsize=16, fontweight='bold')
plt.title('High GCCN Flights January-June 2022', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(xedges, yedges, masked_rwc_low.T, cmap="RdBu_r", norm=norm, shading='auto')
plt.pcolormesh(xedges, yedges, gray_values_low.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)
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
masked_avg_rwc_high = np.ma.masked_where(np.isnan(avg_rwc_high), avg_rwc_high)
masked_avg_rwc_low = np.ma.masked_where(np.isnan(avg_rwc_low), avg_rwc_low)
vmin = 0
vmax = 1
plt.figure(figsize=(8, 6))
plt.pcolormesh(xedges, yedges, masked_avg_rwc_high.T, cmap="viridis", shading='auto', vmin=vmin, vmax=vmax)
plt.colorbar(label="Mean RWC (g m$^{-3}$)")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Mean RWC — High GCCN Flights (Jan–Jun 2022)', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
plt.pcolormesh(xedges, yedges, masked_avg_rwc_low.T, cmap="viridis", shading='auto', vmin=vmin, vmax=vmax)
plt.colorbar(label="Mean RWC (g m$^{-3}$)")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Mean RWC — Low GCCN Flights (Jan–Jun 2022)', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
#fixing gray NANs
masked_avg_rwc_high = np.ma.masked_where(np.isnan(avg_rwc_high), avg_rwc_high)
masked_avg_rwc_low = np.ma.masked_where(np.isnan(avg_rwc_low), avg_rwc_low)
vmin = 0
vmax = 1
cmap = plt.cm.viridis.copy()
cmap.set_bad(color='gray')  
plt.figure(figsize=(8, 6))
img_high = plt.pcolormesh(
    xedges, yedges, masked_avg_rwc_high.T,
    cmap=cmap, shading='auto', vmin=vmin, vmax=vmax
)
cbar_high = plt.colorbar(img_high)
cbar_high.set_label("Mean RWC (g m$^{-3}$)", fontsize=20, fontweight='bold')
cbar_high.ax.tick_params(labelsize=20, width=2, length=5)
for t in cbar_high.ax.get_yticklabels():
    t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Mean RWC\nHigh GCCN Mass Flights\n(January–June 2022)', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
img_low = plt.pcolormesh(
    xedges, yedges, masked_avg_rwc_low.T,
    cmap=cmap, shading='auto', vmin=vmin, vmax=vmax
)
cbar_low = plt.colorbar(img_low)
cbar_low.set_label("Mean RWC (g m$^{-3}$)", fontsize=20, fontweight='bold')
cbar_low.ax.tick_params(labelsize=20, width=2, length=5)
for t in cbar_low.ax.get_yticklabels():
    t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Mean RWC\nLow GCCN Mass Flights\n(January–June 2022)', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
#creating our own color bar for RWC
valid_data_high = avg_rwc_high[~np.isnan(avg_rwc_high)].flatten()
valid_data_low = avg_rwc_low[~np.isnan(avg_rwc_low)].flatten()
all_valid_rwc = np.concatenate([valid_data_high, valid_data_low])
bounds = [0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8]
#for 2x2 binning scheme
# bounds = [0, 0.005, 0.01, 0.01, 0.02, 0.028, 0.03, 0.038, 0.05]
cmap = plt.cm.viridis.copy()
cmap.set_bad(color='gray') 
norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N, extend='neither')
masked_avg_rwc_high = np.ma.masked_where(np.isnan(avg_rwc_high), avg_rwc_high)
masked_avg_rwc_low = np.ma.masked_where(np.isnan(avg_rwc_low), avg_rwc_low)
plt.figure(figsize=(8, 6))
img_high = plt.pcolormesh(
    xedges, yedges, masked_avg_rwc_high.T,
    cmap=cmap, norm=norm, shading='auto'
)
cbar_high = plt.colorbar(img_high, ticks=bounds)
cbar_high.set_label("Mean RWC (g m$^{-3}$)", fontsize=20, fontweight='bold')
cbar_high.ax.tick_params(labelsize=20, width=2, length=5)
for t in cbar_high.ax.get_yticklabels():
    t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Mean RWC\nHigh GCCN Mass Flights\n(January–June 2022)', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
img_low = plt.pcolormesh(
    xedges, yedges, masked_avg_rwc_low.T,
    cmap=cmap, norm=norm, shading='auto'
)
cbar_low = plt.colorbar(img_low, ticks=bounds)
cbar_low.set_label("Mean RWC (g m$^{-3}$)", fontsize=20, fontweight='bold')
cbar_low.ax.tick_params(labelsize=20, width=2, length=5)
for t in cbar_low.ax.get_yticklabels():
    t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Mean RWC\nLow GCCN Mass Flights\n(January–June 2022)', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
#creating our own colorbar but for LWC 
masked_avg_lwc_high = np.ma.masked_where(np.isnan(avg_lwc_high), avg_lwc_high)
masked_avg_lwc_low = np.ma.masked_where(np.isnan(avg_lwc_low), avg_lwc_low)
valid_data_high = avg_lwc_high[~np.isnan(avg_lwc_high)].flatten()
valid_data_low = avg_lwc_low[~np.isnan(avg_lwc_low)].flatten()
all_valid_lwc = np.concatenate([valid_data_high, valid_data_low])
bounds = [0, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0] 
#bounds for 2x2 binning scheme 
# bounds = [0, 0.04, 0.06, 0.07, 0.09, 0.1, 0.15, 0.18, 0.2, 0.25, 0.3, 0.35, 0.38, 0.4] 
cmap = plt.cm.plasma.copy()
cmap.set_bad(color='gray')
norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N, extend='neither')
plt.figure(figsize=(8, 6))
img_high = plt.pcolormesh(
    xedges, yedges, masked_avg_lwc_high.T,
    cmap=cmap, norm=norm, shading='auto'
)
cbar_high = plt.colorbar(img_high, ticks=bounds)
cbar_high.set_label("Mean LWC (g m$^{-3}$)", fontsize=20, fontweight='bold')
cbar_high.ax.tick_params(labelsize=20, width=2, length=5)
for t in cbar_high.ax.get_yticklabels():
    t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Mean LWC\nHigh GCCN Mass Flights\n(January–June 2022)', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
img_low = plt.pcolormesh(
    xedges, yedges, masked_avg_lwc_low.T,
    cmap=cmap, norm=norm, shading='auto'
)
cbar_low = plt.colorbar(img_low, ticks=bounds)
cbar_low.set_label("Mean LWC (g m$^{-3}$)", fontsize=20, fontweight='bold')
cbar_low.ax.tick_params(labelsize=20, width=2, length=5)
for t in cbar_low.ax.get_yticklabels():
    t.set_fontweight('bold')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Mean LWC\nLow GCCN Flights\n(January–June 2022)', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
#using ratio of high lwc/low lwc and high rwc/low rwc 
ratio_rwc = np.divide(
    avg_rwc_high,
    avg_rwc_low,
    out=np.full_like(avg_rwc_high, np.nan),
    where=avg_rwc_low > 0
)
ratio_lwc = np.divide(
    avg_lwc_high,
    avg_lwc_low,
    out=np.full_like(avg_lwc_high, np.nan),
    where=avg_lwc_low > 0
)
masked_ratio_rwc = np.ma.masked_where(np.isnan(ratio_rwc), ratio_rwc)
masked_ratio_lwc = np.ma.masked_where(np.isnan(ratio_lwc), ratio_lwc)
norm = mcolors.Normalize(vmin=0, vmax=2)
plt.figure(figsize=(8,6))
img = plt.pcolormesh(
    xedges, yedges,
    masked_ratio_rwc.T,
    cmap="RdBu_r",
    norm=norm,
    shading="auto"
)
cbar = plt.colorbar(img)
cbar.set_label("RWC Ratio (High / Low)", fontsize=18, fontweight="bold")
cbar.ax.tick_params(labelsize=16)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"Nr+Nc (cm$^{-3}$)", fontsize=19, fontweight="bold")
plt.ylabel(r"LWC (g m$^{-3}$)", fontsize=19, fontweight="bold")
plt.title("RWC Ratio — High / Low GCCN Flights", fontsize=19, fontweight="bold")
plt.tight_layout()
plt.show()
plt.figure(figsize=(8,6))
img = plt.pcolormesh(
    xedges, yedges,
    masked_ratio_lwc.T,
    cmap="RdBu_r",
    norm=norm,
    shading="auto"
)
cbar = plt.colorbar(img)
cbar.set_label("LWC Ratio (High / Low)", fontsize=18, fontweight="bold")
cbar.ax.tick_params(labelsize=16)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"Nr+Nc (cm$^{-3}$)", fontsize=19, fontweight="bold")
plt.ylabel(r"LWC (g m$^{-3}$)", fontsize=19, fontweight="bold")
plt.title("LWC Ratio — High / Low GCCN Flights", fontsize=19, fontweight="bold")
plt.tight_layout()
plt.show()
#%%
masked_avg_rwc_low = np.ma.masked_where(np.isnan(avg_rwc_low), avg_rwc_low)
masked_avg_lwc_low = np.ma.masked_where(np.isnan(avg_lwc_low), avg_lwc_low)
norm = mcolors.Normalize(vmin=0, vmax=2)
cmap_rwc = plt.cm.viridis.copy()
cmap_rwc.set_bad(color='gray')
cmap_lwc = plt.cm.plasma.copy()
cmap_lwc.set_bad(color='gray')
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(
    xedges, yedges,
    masked_avg_rwc_low.T,
    cmap=cmap_rwc,
    shading='auto',
    norm=norm 
)
cbar = plt.colorbar(img)
cbar.set_label("Mean RWC (g m$^{-3}$)", fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=19, width=2, length=5)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.title('Low GCCN Flights — Mean RWC', fontsize=19, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(
    xedges, yedges,
    masked_avg_lwc_low.T,
    cmap=cmap_lwc,
    shading='auto',
    norm=norm 
)
cbar = plt.colorbar(img)
cbar.set_label("Mean LWC (g m$^{-3}$)", fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=19, width=2, length=5)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.title('Low GCCN Flights — Mean LWC', fontsize=19, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.tight_layout()
plt.show()
#%%
#trying to fix color scale 
ratio_rwc = np.divide(
    avg_rwc_high,
    avg_rwc_low,
    out=np.full_like(avg_rwc_high, np.nan),
    where=avg_rwc_low > 0
)
masked_ratio_rwc = np.ma.masked_where(np.isnan(ratio_rwc), ratio_rwc)
# custom_bounds = [1.2, 1.4, 1.6, 1.7, 2.0, 2.2, 2.4, 3.5, 3.8, 4.0, 4.3, 7.0]
custom_bounds = [1.1, 1.2, 2.0, 2.2, 2.4, 2.7, 3.0, 3.5, 3.8, 7.0]
cmap = plt.cm.viridis.copy()
cmap.set_bad(color='gray')
norm = BoundaryNorm(boundaries=custom_bounds, ncolors=cmap.N)
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(
    xedges, yedges,
    masked_ratio_rwc.T,
    cmap=cmap,
    norm=norm,
    shading="auto"
)
cbar = plt.colorbar(img, ticks=custom_bounds)
cbar.set_label("RWC Ratio (High / Low)", fontsize=19, fontweight="bold")
cbar.ax.tick_params(labelsize=19)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
plt.xscale("log")
plt.yscale("log")
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.xlabel(r"Nr+Nc (cm$^{-3}$)", fontsize=19, fontweight="bold")
plt.ylabel(r"LWC (g m$^{-3}$)", fontsize=19, fontweight="bold")
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.title("CAS (in cloud)\nRWC Ratio High / Low GCCN Flights\nJanuary–June 2022", fontsize=19, fontweight="bold")
plt.tight_layout()
plt.show()
#%%
min_rwc_threshold = 0.003
valid_mask = (avg_rwc_low > min_rwc_threshold) & ~np.isnan(avg_rwc_low) & ~np.isnan(avg_rwc_high)
safe_ratio = np.full_like(avg_rwc_high, np.nan)
safe_ratio[valid_mask] = avg_rwc_high[valid_mask] / avg_rwc_low[valid_mask]
masked_ratio_rwc = np.ma.masked_where(np.isnan(safe_ratio), safe_ratio)
masked_ratio_rwc = np.ma.masked_where(masked_ratio_rwc > 5, masked_ratio_rwc)  # cap extremes
custom_bounds = [0, 0.2, 0.5, 1.0, 1.1, 1.2, 1.3, 1.4, 2.0, 2.2, 2.4, 2.7, 3.0, 3.2]
cmap = plt.cm.viridis.copy()
cmap.set_bad(color='gray')
norm = BoundaryNorm(boundaries=custom_bounds, ncolors=cmap.N)
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(
    xedges, yedges,
    masked_ratio_rwc.T,
    cmap=cmap,
    norm=norm,
    shading="auto"
)
cbar = plt.colorbar(img, ticks=custom_bounds)
cbar.set_label("RWC Ratio (High / Low)", fontsize=19, fontweight="bold")
cbar.ax.tick_params(labelsize=19)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
plt.xscale("log")
plt.yscale("log")
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.xlabel(r"Nr+Nc (cm$^{-3}$)", fontsize=19, fontweight="bold")
plt.ylabel(r"LWC (g m$^{-3}$)", fontsize=19, fontweight="bold")
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.title("CAS (in cloud)\nRWC Ratio High / Low GCCN Flights\nJanuary–June 2022", fontsize=19, fontweight="bold")
plt.tight_layout()
plt.show()

























































#%%
#create dictionaries to store all relevant variables for each classification

# === Build HIGH/LOW full mass dictionaries ===
high_full_mass, low_full_mass = {}, {}

for entry in total_liquid_water:   # second-by-second RWC/LWC
    d = entry['Date']
    if d in high_GCCN_mass_flights:
        if d not in high_full_mass:
            high_full_mass[d] = {
                'Average_GCCN_Mass': high_GCCN_mass_flights[d],
                'LWC': [],
                'RWC': [],
                'Total_Combined_Mass': []
            }
        high_full_mass[d]['LWC'].append(entry['Total_Liquid_Water'])
        high_full_mass[d]['RWC'].append(entry['RWC'])
        high_full_mass[d]['Total_Combined_Mass'].append(high_GCCN_mass_flights[d])  # repeat avg mass for alignment

    elif d in low_GCCN_mass_flights:
        if d not in low_full_mass:
            low_full_mass[d] = {
                'Average_GCCN_Mass': low_GCCN_mass_flights[d],
                'LWC': [],
                'RWC': [],
                'Total_Combined_Mass': []
            }
        low_full_mass[d]['LWC'].append(entry['Total_Liquid_Water'])
        low_full_mass[d]['RWC'].append(entry['RWC'])
        low_full_mass[d]['Total_Combined_Mass'].append(low_GCCN_mass_flights[d])  # repeat avg mass

#%%
# #for mass 
# # Build per-date containers for HIGH / LOW GCCN MASS flights
# high_full_mass = {}
# low_full_mass  = {}

# def _ensure_mass_bucket(bucket, date, avg_mass):
#     if date not in bucket:
#         bucket[date] = {
#             'Average_GCCN_Mass': avg_mass,   # flight-level avg mass for this date
#             'Total_Combined_Mass': [],       # leg-level (or segment) mass values
#             'LWC': [],                       # per-second LWC pulled by date
#             'RWC': []                        # per-second RWC pulled by date
#         }
# for entry in total_combined_mass:
#     d = entry['Date']
#     if d in high_GCCN_mass_flights:
#         _ensure_mass_bucket(high_full_mass, d, high_GCCN_mass_flights[d])
#         high_full_mass[d]['Total_Combined_Mass'].append(entry['Total_Combined_Mass'])
#     elif d in low_GCCN_mass_flights:
#         _ensure_mass_bucket(low_full_mass, d, low_GCCN_mass_flights[d])
#         low_full_mass[d]['Total_Combined_Mass'].append(entry['Total_Combined_Mass'])
# for entry in total_liquid_water:
#     d = entry['Date']
#     if d in high_GCCN_mass_flights:
#         _ensure_mass_bucket(high_full_mass, d, high_GCCN_mass_flights[d])
#         high_full_mass[d]['LWC'].append(entry['Total_Liquid_Water'])
#         high_full_mass[d]['RWC'].append(entry['RWC'])
#     elif d in low_GCCN_mass_flights:
#         _ensure_mass_bucket(low_full_mass, d, low_GCCN_mass_flights[d])
#         low_full_mass[d]['LWC'].append(entry['Total_Liquid_Water'])
#         low_full_mass[d]['RWC'].append(entry['RWC'])

# %%
#trying to make histograms in each bin of all the points without bootsrapping HIGH concentration
# rwc_lwc_high = avg_rwc_high / avg_lwc_high * 100  # unit: %
# num_bins = 5
# x_bins = np.logspace(np.log10(x_min), np.log10(x_max), num_bins)
# y_bins = np.logspace(np.log10(y_min), np.log10(y_max), num_bins)
# raw_bin_data_high = {}

# for i in range(len(x_bins)-1):
#     for j in range(len(y_bins)-1):
#         bin_mask = (
#             (high_full_mass >= x_bins[i]) & (high_full_mass < x_bins[i+1]) &
#             (avg_lwc_high >= y_bins[j]) & (avg_lwc_high < y_bins[j+1])
#         )
#         bin_vals = rwc_lwc_high[bin_mask]
#         if len(bin_vals) > 0:
#             raw_bin_data_high[(i, j)] = bin_vals
# fig, axes = plt.subplots(nrows=len(x_bins)-1, ncols=len(y_bins)-1, figsize=(20, 16), sharex=True, sharey=True)
# for (i, j), values in raw_bin_data_high.items():
#     ax = axes[i, j] if isinstance(axes[0], np.ndarray) else axes[i * (len(y_bins)-1) + j]
#     sns.histplot(values, ax=ax, bins=30, color='blue', kde=False)
#     x_range = f"{x_bins[i]:.0f}-{x_bins[i+1]:.0f} cm⁻³"
#     y_range = f"{y_bins[j]:.2f}-{y_bins[j+1]:.2f} g/m³"
#     ax.set_title(f"Bin ({i},{j}):\nNr+Nc = {x_range}, \nLWC = {y_range}", fontsize=16, fontweight='bold')
#     ax.axvline(0, color='red', linestyle='--', linewidth=1)
#     ax.set_xlabel("RWC / LWC (%)", fontsize=18, fontweight='bold')
#     ax.set_ylabel("Number of Legs", fontsize=18, fontweight='bold')
#     ax.tick_params(axis='x', labelbottom=True)
#     ax.tick_params(axis='both', labelsize=16)
#%%

# === Flatten high/low dictionaries into arrays ===
high_mass = np.concatenate([vals['Total_Combined_Mass'] for vals in high_full_mass.values()])
high_lwc  = np.concatenate([vals['LWC'] for vals in high_full_mass.values()])
high_rwc  = np.concatenate([vals['RWC'] for vals in high_full_mass.values()])

low_mass  = np.concatenate([vals['Total_Combined_Mass'] for vals in low_full_mass.values()])
low_lwc   = np.concatenate([vals['LWC'] for vals in low_full_mass.values()])
low_rwc   = np.concatenate([vals['RWC'] for vals in low_full_mass.values()])

# === Bin setup ===
num_bins = 5
x_bins = np.logspace(np.log10(1), np.log10(max(np.concatenate([high_mass, low_mass]))), num_bins)
y_bins = np.logspace(np.log10(min(np.concatenate([high_lwc, low_lwc]))),
                     np.log10(max(np.concatenate([high_lwc, low_lwc]))), num_bins)

# === High mass binning ===
sum_rwc_high, xedges, yedges = np.histogram2d(
    high_mass, high_lwc, bins=[x_bins, y_bins], weights=high_rwc
)
sum_lwc_high, _, _ = np.histogram2d(
    high_mass, high_lwc, bins=[x_bins, y_bins], weights=high_lwc
)
counts_high, _, _ = np.histogram2d(high_mass, high_lwc, bins=[x_bins, y_bins])

avg_rwc_high = np.divide(sum_rwc_high, counts_high,
                         out=np.full_like(sum_rwc_high, np.nan),
                         where=counts_high > 0)
avg_lwc_high = np.divide(sum_lwc_high, counts_high,
                         out=np.full_like(sum_lwc_high, np.nan),
                         where=counts_high > 0)
rwc_lwc_ratio_high = np.divide(avg_rwc_high, avg_lwc_high,
                               out=np.full_like(avg_rwc_high, np.nan),
                               where=avg_lwc_high > 0) * 100
masked_rwc_high = np.ma.masked_where(np.isnan(rwc_lwc_ratio_high), rwc_lwc_ratio_high)

# === Low mass binning ===
sum_rwc_low, _, _ = np.histogram2d(
    low_mass, low_lwc, bins=[x_bins, y_bins], weights=low_rwc
)
sum_lwc_low, _, _ = np.histogram2d(
    low_mass, low_lwc, bins=[x_bins, y_bins], weights=low_lwc
)
counts_low, _, _ = np.histogram2d(low_mass, low_lwc, bins=[x_bins, y_bins])

avg_rwc_low = np.divide(sum_rwc_low, counts_low,
                        out=np.full_like(sum_rwc_low, np.nan),
                        where=counts_low > 0)
avg_lwc_low = np.divide(sum_lwc_low, counts_low,
                        out=np.full_like(sum_lwc_low, np.nan),
                        where=counts_low > 0)
rwc_lwc_ratio_low = np.divide(avg_rwc_low, avg_lwc_low,
                              out=np.full_like(avg_rwc_low, np.nan),
                              where=avg_lwc_low > 0) * 100
masked_rwc_low = np.ma.masked_where(np.isnan(rwc_lwc_ratio_low), rwc_lwc_ratio_low)

# === Plot High Mass Flights ===
plt.figure(figsize=(8, 6))
norm = mcolors.Normalize(vmin=1, vmax=100)
img = plt.pcolormesh(xedges, yedges, masked_rwc_high.T,
                     cmap="RdBu_r", norm=norm, shading='auto')
plt.colorbar(img, label="RWC / LWC (%)")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Total GCCN Mass (µg/m³)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('High GCCN Mass Flights (Jan–Jun 2022)', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()

# === Plot Low Mass Flights ===
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(xedges, yedges, masked_rwc_low.T,
                     cmap="RdBu_r", norm=norm, shading='auto')
plt.colorbar(img, label="RWC / LWC (%)")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Total GCCN Mass (µg/m³)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Low GCCN Mass Flights (Jan–Jun 2022)', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()


# %%
#trying to make histograms in each bin of all the points without bootsrapping LOW concentration

# Compute ratio (RWC/LWC * 100) for low-mass flights
rwc_lwc_low = np.divide(low_rwc, low_lwc, out=np.full_like(low_rwc, np.nan), where=low_lwc > 0) * 100

num_bins = 5
x_bins = np.logspace(np.log10(low_mass.min()), np.log10(low_mass.max()), num_bins)
y_bins = np.logspace(np.log10(low_lwc.min()), np.log10(low_lwc.max()), num_bins)

raw_bin_data_low = {}
for i in range(len(x_bins)-1):
    for j in range(len(y_bins)-1):
        bin_mask = (
            (low_mass >= x_bins[i]) & (low_mass < x_bins[i+1]) &
            (low_lwc  >= y_bins[j]) & (low_lwc  < y_bins[j+1])
        )
        bin_vals = rwc_lwc_low[bin_mask]
        if len(bin_vals) > 0:
            raw_bin_data_low[(i, j)] = bin_vals

fig, axes = plt.subplots(nrows=len(x_bins)-1, ncols=len(y_bins)-1, 
                         figsize=(20, 16), sharex=True, sharey=True)

for (i, j), values in raw_bin_data_low.items():
    ax = axes[i, j] if isinstance(axes[0], np.ndarray) else axes[i * (len(y_bins)-1) + j]
    sns.histplot(values, ax=ax, bins=30, color='blue', kde=False)

    x_range = f"{x_bins[i]:.2f}-{x_bins[i+1]:.2f} µg/m³"  # update units for mass
    y_range = f"{y_bins[j]:.3f}-{y_bins[j+1]:.3f} g/m³"
    ax.set_title(f"Bin ({i},{j}):\nMass = {x_range}, \nLWC = {y_range}",
                 fontsize=16, fontweight='bold')

    ax.axvline(0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel("RWC / LWC (%)", fontsize=18, fontweight='bold')
    ax.set_ylabel("Count", fontsize=18, fontweight='bold')
    ax.tick_params(axis='x', labelbottom=True)
    ax.tick_params(axis='both', labelsize=16)

plt.tight_layout()
plt.show()
#%%
# Build aligned per-second mass dataset
high_mass_data = []
low_mass_data  = []

for entry in total_liquid_water:   # per-second RWC/LWC
    d = entry['Date']
    if d in high_GCCN_mass_flights:  # high-mass flights
        high_mass_data.append({
            'Date': d,
            'Total_Combined_Mass': high_GCCN_mass_flights[d],  # assign flight avg mass
            'Total_Liquid_Water': entry['Total_Liquid_Water'],
            'Rain_Concentration': entry['RWC']
        })
    elif d in low_GCCN_mass_flights:  # low-mass flights
        low_mass_data.append({
            'Date': d,
            'Total_Combined_Mass': low_GCCN_mass_flights[d],
            'Total_Liquid_Water': entry['Total_Liquid_Water'],
            'Rain_Concentration': entry['RWC']
        })
#%%
# Build total_combined_mass aligned by Date
total_combined_mass = []

for mass_entry in filtered_dry_mass_inf:
    date = mass_entry['Date']
    mass_val = mass_entry['Dry Mass (µg/m³)']
    matching_liquids = [entry for entry in total_liquid_water if entry['Date'] == date]

    for liquid in matching_liquids:
        total_combined_mass.append({
            'Date': date,
            'Total_Combined_Mass': mass_val,              # flight-average mass
            'Total_Liquid_Water': liquid['Total_Liquid_Water'],
            'Rain_Concentration': liquid['RWC']
        })

#%%
num_bins = 5
x_bins = np.logspace(np.log10(x_min), np.log10(x_max), num_bins)
y_bins = np.logspace(np.log10(y_min), np.log10(y_max), num_bins)
n_bootstrap = 10000
confidence_level = 0.90
lower_percentile = (1 - confidence_level) / 2 * 100
upper_percentile = (1 + confidence_level) / 2 * 100
def group_by_flight(data):
    flights = defaultdict(list)
    for entry in data:
        flights[entry['Date']].append(entry)
    return flights
def compute_flight_bin_means(flight_data):
    bin_means = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    for flight in flight_data.values():
        mass = np.array([e['Total_Combined_Mass'] for e in flight])
        lwc = np.array([e['Total_Liquid_Water'] for e in flight])
        rwc = np.array([e['Rain_Concentration'] for e in flight])
        ratio = np.where(lwc > 0, rwc / lwc * 100, np.nan)

       
        print(f"Flight range: Mass {mass.min():.2f}–{mass.max():.2f}, LWC {lwc.min():.3f}–{lwc.max():.3f}")

        for i in range(len(x_bins)-1):
            for j in range(len(y_bins)-1):
                mask = (mass >= x_bins[i]) & (mass < x_bins[i+1]) & \
                       (lwc >= y_bins[j]) & (lwc < y_bins[j+1])
                if np.any(mask):
                    mean_val = np.nanmean(ratio[mask])
                    if not np.isnan(mean_val):
                        bin_means[i][j].append(mean_val)
    return bin_means
def bootstrap_bin_differences(bin_high, bin_low):
    boot_dists = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            high_vals = bin_high[i][j]
            low_vals = bin_low[i][j]
            print(f"Bin ({i},{j}): High Flights = {len(high_vals)}, Low Flights = {len(low_vals)}")  # Debug
            if len(high_vals) >= 2 and len(low_vals) >= 2:
                boot_diff = [
                    np.mean(np.random.choice(high_vals, len(high_vals), replace=True)) -
                    np.mean(np.random.choice(low_vals, len(low_vals), replace=True))
                    for _ in range(n_bootstrap)
                ]
                boot_dists[i][j] = np.array(boot_diff)
    return boot_dists
def plot_histograms(boot_dists):
    fig, axes = plt.subplots(nrows=len(x_bins)-1, ncols=len(y_bins)-1, figsize=(14, 10), sharex=True, sharey=True)
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            ax = axes[i][j] if isinstance(axes[0], np.ndarray) else axes[i*(len(y_bins)-1)+j]
            data = boot_dists[i][j]
            if len(data) > 0:
                sns.histplot(data, bins=30, kde=False, ax=ax, color='skyblue')
                ax.axvline(0, color='red', linestyle='--')
                lower = np.percentile(data, lower_percentile)
                upper = np.percentile(data, upper_percentile)
                ax.axvline(lower, color='black', linestyle=':', linewidth=1)
                ax.axvline(upper, color='black', linestyle=':', linewidth=1)
                ax.set_title(f"Bin ({i},{j})\n90% CI: {lower:.1f}, {upper:.1f}%")
                ax.set_xlabel("RWC/LWC Difference (%)")
                ax.set_ylabel("Count")
            else:
                ax.set_visible(False)
    plt.suptitle("Bootstrapped RWC/LWC Differences Per Bin (Flights as Legs)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
gccn_values = np.array(list(average_mass_per_flight.values()))
threshold = np.percentile(gccn_values, 50)

high_dates = {date for date, val in average_mass_per_flight.items() if val >= threshold}
low_dates = {date for date, val in average_mass_per_flight.items() if val < threshold}

high_gccn_data = [entry for entry in total_combined_mass if entry['Date'] in high_dates]
low_gccn_data = [entry for entry in total_combined_mass if entry['Date'] in low_dates]

grouped_high = group_by_flight(high_gccn_data)
grouped_low  = group_by_flight(low_gccn_data)


bin_means_high = compute_flight_bin_means(grouped_high)
bin_means_low  = compute_flight_bin_means(grouped_low)

boot_distributions = bootstrap_bin_differences(bin_means_high, bin_means_low)
plot_histograms(boot_distributions)
for date, flights in grouped_high.items():
    print(f"\n{date} - {len(flights)} entries")
    print(f"Keys: {list(flights[0].keys())}")
    break

bin_means_high = compute_flight_bin_means(grouped_high)
bin_means_low = compute_flight_bin_means(grouped_low)

boot_distributions = bootstrap_bin_differences(bin_means_high, bin_means_low)

plot_histograms(boot_distributions)

def plot_histograms_with_flight_means_corrected(boot_dists, bin_means_high, bin_means_low, grouped_high, grouped_low, highlight_date="2022-05-10"):
    fig, axes = plt.subplots(nrows=len(x_bins)-1, ncols=len(y_bins)-1, figsize=(14, 10), sharex=True, sharey=True)

    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            ax = axes[i][j] if isinstance(axes[0], np.ndarray) else axes[i*(len(y_bins)-1)+j]
            dist = boot_dists[i][j]

            if len(dist) > 0:
                sns.histplot(dist, bins=30, kde=False, ax=ax, color='lightblue')
                ax.axvline(0, color='red', linestyle='--')
                lower = np.percentile(dist, lower_percentile)
                upper = np.percentile(dist, upper_percentile)
                ax.axvline(lower, color='black', linestyle=':', linewidth=1)
                ax.axvline(upper, color='black', linestyle=':', linewidth=1)
                percent_above_zero = np.sum(dist > 0) / len(dist) * 100

                ax.text(
                    0.98, 0.95,
                    f"{percent_above_zero:.1f}% > 0",
                    transform=ax.transAxes,
                    ha='right',
                    va='top',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.7)
                )

                ax.set_title(f"Bin ({i},{j})\n90% CI: {lower:.1f}, {upper:.1f}%", fontsize=10)


                for val in bin_means_high[i][j]:
                    ax.scatter(val, ax.get_ylim()[1] * 0.02, color='blue', s=20, alpha=0.6, label='High GCCN' if (i==0 and j==0) else "")
                for val in bin_means_low[i][j]:
                    ax.scatter(val, ax.get_ylim()[1] * 0.04, color='orange', s=20, alpha=0.6, label='Low GCCN' if (i==0 and j==0) else "")

                flight = None
                if highlight_date in grouped_high:
                    flight = grouped_high[highlight_date]
                elif highlight_date in grouped_low:
                    flight = grouped_low[highlight_date]

                if flight is not None:
                    mass = np.array([e['Total_Combined_Mass'] for e in flight])
                    lwc = np.array([e['Total_Liquid_Water'] for e in flight])
                    rwc = np.array([e['Rain_Concentration'] for e in flight])
                    ratio = np.where(lwc > 0, rwc / lwc * 100, np.nan)
                    mask = (mass >= x_bins[i]) & (mass < x_bins[i+1]) & (lwc >= y_bins[j]) & (lwc < y_bins[j+1])
                    if np.any(mask):
                        val = np.nanmean(ratio[mask])
                        ax.scatter(val, ax.get_ylim()[1] * 0.1, color='black', s=40, marker='X', label='May 10' if (i==0 and j==0) else "")
            else:
                ax.set_visible(False)
#%%

#%%
#for mass
num_bins = 5
x_bins = np.logspace(np.log10(x_min), np.log10(x_max), num_bins)  # Nr+Nc bins
y_bins = np.logspace(np.log10(y_min), np.log10(y_max), num_bins)  # LWC bins

n_bootstrap = 10000
confidence_level = 0.90
lower_percentile = (1 - confidence_level) / 2 * 100
upper_percentile = (1 + confidence_level) / 2 * 100

from collections import defaultdict

def group_by_flight(data):
    flights = defaultdict(list)
    for entry in data:
        flights[entry['Date']].append(entry)
    return flights

def compute_flight_bin_means(flight_data):
    bin_means = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    for flight in flight_data.values():
        conc = np.array([e['Total_Combined_Concentration'] for e in flight])  # Nr+Nc
        lwc  = np.array([e['Total_Liquid_Water']       for e in flight])
        rwc  = np.array([e['Rain_Concentration']       for e in flight])      # per-second RWC
        ratio = np.where(lwc > 0, rwc / lwc * 100, np.nan)

        print(f"Flight range: Conc {np.nanmin(conc):.2f}–{np.nanmax(conc):.2f}, "
              f"LWC {np.nanmin(lwc):.3f}–{np.nanmax(lwc):.3f}")

        for i in range(len(x_bins)-1):
            for j in range(len(y_bins)-1):
                mask = (conc >= x_bins[i]) & (conc < x_bins[i+1]) & \
                       (lwc  >= y_bins[j]) & (lwc  < y_bins[j+1])
                if np.any(mask):
                    mean_val = np.nanmean(ratio[mask])
                    if not np.isnan(mean_val):
                        bin_means[i][j].append(mean_val)
    return bin_means

def bootstrap_bin_differences(bin_high, bin_low):
    boot_dists = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            high_vals = bin_high[i][j]
            low_vals  = bin_low[i][j]
            print(f"Bin ({i},{j}): High Flights = {len(high_vals)}, Low Flights = {len(low_vals)}")
            if len(high_vals) >= 2 and len(low_vals) >= 2:
                boot_diff = [
                    np.mean(np.random.choice(high_vals, len(high_vals), replace=True)) -
                    np.mean(np.random.choice(low_vals,  len(low_vals),  replace=True))
                    for _ in range(n_bootstrap)
                ]
                boot_dists[i][j] = np.array(boot_diff)
    return boot_dists

def plot_histograms(boot_dists):
    fig, axes = plt.subplots(nrows=len(x_bins)-1, ncols=len(y_bins)-1,
                             figsize=(14, 10), sharex=True, sharey=True)
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            ax = axes[i][j] if isinstance(axes[0], np.ndarray) else axes[i*(len(y_bins)-1)+j]
            data = boot_dists[i][j]
            if len(data) > 0:
                sns.histplot(data, bins=30, kde=False, ax=ax, color='skyblue')
                ax.axvline(0, color='red', linestyle='--')
                lower = np.percentile(data, lower_percentile)
                upper = np.percentile(data, upper_percentile)
                ax.axvline(lower, color='black', linestyle=':', linewidth=1)
                ax.axvline(upper, color='black', linestyle=':', linewidth=1)
                ax.set_title(f"Bin ({i},{j})\n90% CI: {lower:.1f}, {upper:.1f}%")
                ax.set_xlabel("RWC/LWC Difference (%)")
                ax.set_ylabel("Count")
            else:
                ax.set_visible(False)
    plt.suptitle("Bootstrapped RWC/LWC Differences Per Bin (High − Low GCCN Mass)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_histograms_with_flight_means_corrected(boot_dists, bin_means_high, bin_means_low,
                                                grouped_high, grouped_low, highlight_date="2022-05-10"):
    fig, axes = plt.subplots(nrows=len(x_bins)-1, ncols=len(y_bins)-1,
                             figsize=(14, 10), sharex=True, sharey=True)
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            ax = axes[i][j] if isinstance(axes[0], np.ndarray) else axes[i*(len(y_bins)-1)+j]
            dist = boot_dists[i][j]
            if len(dist) > 0:
                sns.histplot(dist, bins=30, kde=False, ax=ax, color='lightblue')
                ax.axvline(0, color='red', linestyle='--')
                lower = np.percentile(dist, lower_percentile)
                upper = np.percentile(dist, upper_percentile)
                ax.axvline(lower, color='black', linestyle=':', linewidth=1)
                ax.axvline(upper, color='black', linestyle=':', linewidth=1)
                pct_pos = np.sum(dist > 0) / len(dist) * 100
                ax.text(0.98, 0.95, f"{pct_pos:.1f}% > 0", transform=ax.transAxes,
                        ha='right', va='top', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.7))
                ax.set_title(f"Bin ({i},{j})\n90% CI: {lower:.1f}, {upper:.1f}%", fontsize=10)

                for val in bin_means_high[i][j]:
                    ax.scatter(val, ax.get_ylim()[1] * 0.02, color='blue',   s=20, alpha=0.6,
                               label='High GCCN Mass' if (i==0 and j==0) else "")
                for val in bin_means_low[i][j]:
                    ax.scatter(val, ax.get_ylim()[1] * 0.04, color='orange', s=20, alpha=0.6,
                               label='Low GCCN Mass' if (i==0 and j==0) else "")

                flight = grouped_high.get(highlight_date) or grouped_low.get(highlight_date)
                if flight is not None:
                    conc = np.array([e['Total_Combined_Concentration'] for e in flight])
                    lwc  = np.array([e['Total_Liquid_Water']       for e in flight])
                    rwc  = np.array([e['Rain_Concentration']       for e in flight])
                    ratio = np.where(lwc > 0, rwc / lwc * 100, np.nan)
                    mask = (conc >= x_bins[i]) & (conc < x_bins[i+1]) & \
                           (lwc  >= y_bins[j]) & (lwc  < y_bins[j+1])
                    if np.any(mask):
                        val = np.nanmean(ratio[mask])
                        ax.scatter(val, ax.get_ylim()[1] * 0.1, color='black', s=40, marker='X',
                                   label=highlight_date if (i==0 and j==0) else "")
            else:
                ax.set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper right')
    plt.suptitle("Bootstrapped RWC/LWC Differences Per Bin (High − Low GCCN Mass)\nwith Flight Means",
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
mass_vals   = np.array(list(average_mass_per_flight.values()))
mass_thresh = np.nanpercentile(mass_vals, 50)

high_dates = {date for date, val in average_mass_per_flight.items() if val >= mass_thresh}
low_dates  = {date for date, val in average_mass_per_flight.items() if val <  mass_thresh}
high_mass_data = [e for e in total_combined_concentration if e['Date'] in high_dates]
low_mass_data  = [e for e in total_combined_concentration if e['Date'] in low_dates]

grouped_high = group_by_flight(high_mass_data)
grouped_low  = group_by_flight(low_mass_data)
for date, flight_entries in grouped_high.items():
    print(f"\n{date} - {len(flight_entries)} entries")
    print(f"Keys: {list(flight_entries[0].keys())}")
    break

bin_means_high = compute_flight_bin_means(grouped_high)
bin_means_low  = compute_flight_bin_means(grouped_low)

boot_distributions = bootstrap_bin_differences(bin_means_high, bin_means_low)

plot_histograms(boot_distributions)
plot_histograms_with_flight_means_corrected(
    boot_distributions, bin_means_high, bin_means_low, grouped_high, grouped_low,
    highlight_date="2022-05-10"   
)

#%%
#printing the >0% values for each bin
num_bins = 5
x_bins = np.logspace(np.log10(x_min), np.log10(x_max), num_bins)
y_bins = np.logspace(np.log10(y_min), np.log10(y_max), num_bins)
n_bootstrap = 10000
confidence_level = 0.90
lower_percentile = (1 - confidence_level) / 2 * 100
upper_percentile = (1 + confidence_level) / 2 * 100
def group_by_flight(data):
    flights = defaultdict(list)
    for entry in data:
        flights[entry['Date']].append(entry)
    return flights
def compute_flight_bin_means(flight_data):
    bin_means = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    for flight in flight_data.values():
        conc = np.array([e['Total_Combined_Concentration'] for e in flight])
        lwc = np.array([e['Total_Liquid_Water'] for e in flight])
        rwc = np.array([e['Rain_Concentration'] for e in flight])
        ratio = np.where(lwc > 0, rwc / lwc * 100, np.nan)

        for i in range(len(x_bins)-1):
            for j in range(len(y_bins)-1):
                mask = (conc >= x_bins[i]) & (conc < x_bins[i+1]) & \
                       (lwc >= y_bins[j]) & (lwc < y_bins[j+1])
                if np.any(mask):
                    mean_val = np.nanmean(ratio[mask])
                    if not np.isnan(mean_val):
                        bin_means[i][j].append(mean_val)
    return bin_means
def bootstrap_bin_differences(bin_high, bin_low):
    boot_dists = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            high_vals = bin_high[i][j]
            low_vals = bin_low[i][j]
            if len(high_vals) >= 2 and len(low_vals) >= 2:
                boot_diff = [
                    np.mean(np.random.choice(high_vals, len(high_vals), replace=True)) -
                    np.mean(np.random.choice(low_vals, len(low_vals), replace=True))
                    for _ in range(n_bootstrap)
                ]
                boot_dists[i][j] = np.array(boot_diff)
    return boot_dists
def plot_histograms_with_percentage(boot_dists):
    fig, axes = plt.subplots(nrows=len(x_bins)-1, ncols=len(y_bins)-1, figsize=(14, 10), sharex=True, sharey=True)

    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            ax = axes[i][j] if isinstance(axes[0], np.ndarray) else axes[i*(len(y_bins)-1)+j]
            dist = boot_dists[i][j]

            if len(dist) > 0:
                sns.histplot(dist, bins=30, kde=False, ax=ax, color='skyblue')
                ax.axvline(0, color='red', linestyle='--')
                lower = np.percentile(dist, lower_percentile)
                upper = np.percentile(dist, upper_percentile)
                ax.axvline(lower, color='black', linestyle=':', linewidth=1)
                ax.axvline(upper, color='black', linestyle=':', linewidth=1)

            
                percent_above_zero = np.sum(dist > 0) / len(dist) * 100
                mean_val = np.mean(dist)
                std_val = np.std(dist)

                
                annotation = f"{percent_above_zero:.1f}% > 0\nμ = {mean_val:.2f}, σ = {std_val:.2f}"
                ax.text(
                    0.98, 0.95,
                    annotation,
                    transform=ax.transAxes,
                    ha='right',
                    va='top',
                    fontsize=14,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
                )


                ax.set_xlabel("RWC/LWC Difference (%)", fontsize=18, fontweight='bold')
                ax.set_ylabel("Count", fontsize=18, fontweight='bold')
                ax.tick_params(axis='x', labelbottom=True)
                ax.tick_params(axis='both', labelsize=16)
                ax.set_ylabel("Count")
            else:
                ax.set_visible(False)

    plt.suptitle("Full Bootstrapping Below Cloud Base January-June 2022", fontsize=19, fontweight='bold')
    plt.tight_layout()
    plt.show()
gccn_values = np.array(list(average_gccn_per_flight.values()))
threshold = np.percentile(gccn_values, 50)

high_dates = {date for date, val in average_gccn_per_flight.items() if val >= threshold}
low_dates = {date for date, val in average_gccn_per_flight.items() if val < threshold}

high_gccn_data = [entry for entry in total_combined_concentration if entry['Date'] in high_dates]
low_gccn_data = [entry for entry in total_combined_concentration if entry['Date'] in low_dates]

grouped_high = group_by_flight(high_gccn_data)
grouped_low = group_by_flight(low_gccn_data)

bin_means_high = compute_flight_bin_means(grouped_high)
bin_means_low = compute_flight_bin_means(grouped_low)

boot_distributions = bootstrap_bin_differences(bin_means_high, bin_means_low)

plot_histograms_with_percentage(boot_distributions)

from scipy import stats

print("\n=== One-sided T-test Results Per Bin (H₀: mean ≤ 0, H₁: mean > 0) ===")
for i in range(len(x_bins) - 1):
    for j in range(len(y_bins) - 1):
        dist = boot_distributions[i][j]
        if len(dist) > 1:
            t_stat, p_two_sided = stats.ttest_1samp(dist, popmean=0)
            p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - (p_two_sided / 2)
            print(
                f"Bin ({i},{j}): mean = {np.mean(dist):.2f}, std = {np.std(dist):.2f}, "
                f"t = {t_stat:.2f}, p = {p_one_sided:.2e}"
            )
#%%
gccn_values = np.array(list(average_gccn_per_flight.values()))
threshold = np.percentile(gccn_values, 50)
high_dates = {date for date, val in average_gccn_per_flight.items() if val >= threshold}
low_dates = {date for date, val in average_gccn_per_flight.items() if val < threshold}
high_gccn_data = [e for e in total_combined_concentration if e['Date'] in high_dates]
low_gccn_data = [e for e in total_combined_concentration if e['Date'] in low_dates]

grouped_high = group_by_flight(high_gccn_data)
grouped_low = group_by_flight(low_gccn_data)
bin_means_high, src_high = compute_flight_bin_means(grouped_high)
bin_means_low, src_low = compute_flight_bin_means(grouped_low)

boot_dists, boot_may10_flags = bootstrap_bin_differences_random12_with_may10(bin_means_high, bin_means_low, src_high, src_low)
plot_histograms_may10_stacked(boot_dists, boot_may10_flags)

# %%
#  Revised Code: Treat Each Flight as a Leg, Bootstrap Across Flights per Bin
num_bins = 5
x_bins = np.logspace(np.log10(x_min), np.log10(x_max), num_bins)
y_bins = np.logspace(np.log10(y_min), np.log10(y_max), num_bins)
n_bootstrap = 10000
confidence_level = 0.90
lower_percentile = (1 - confidence_level) / 2 * 100
upper_percentile = (1 + confidence_level) / 2 * 100
def group_by_flight(data):
    flights = defaultdict(list)
    for entry in data:
        key = (entry['Date'], entry['Time'])
        flights[entry['Date']].append(entry)
    return flights
def compute_flight_bin_means(flight_data):
    bin_means = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    for flight in flight_data.values():
        conc = np.array([e['Total_Combined_Concentration'] for e in flight])
        lwc = np.array([e['Total_Liquid_Water'] for e in flight])
        rwc = np.array([e['RWC'] for e in flight])
        ratio = np.where(lwc > 0, rwc / lwc * 100, np.nan)
        for i in range(len(x_bins)-1):
            for j in range(len(y_bins)-1):
                mask = (conc >= x_bins[i]) & (conc < x_bins[i+1]) & \
                       (lwc >= y_bins[j]) & (lwc < y_bins[j+1])
                if np.any(mask):
                    mean_val = np.nanmean(ratio[mask])
                    if not np.isnan(mean_val):
                        bin_means[i][j].append(mean_val)
    return bin_means
def bootstrap_bin_differences(bin_high, bin_low):
    boot_dists = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            high_vals = bin_high[i][j]
            low_vals = bin_low[i][j]
            if len(high_vals) >= 2 and len(low_vals) >= 2:
                boot_diff = [
                    np.mean(np.random.choice(high_vals, len(high_vals), replace=True)) -
                    np.mean(np.random.choice(low_vals, len(low_vals), replace=True))
                    for _ in range(n_bootstrap)
                ]
                boot_dists[i][j] = np.array(boot_diff)
    return boot_dists
def plot_histograms(boot_dists):
    fig, axes = plt.subplots(nrows=len(x_bins)-1, ncols=len(y_bins)-1, figsize=(14, 10), sharex=True, sharey=True)
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            ax = axes[i][j] if isinstance(axes[0], np.ndarray) else axes[i*(len(y_bins)-1)+j]
            data = boot_dists[i][j]
            if len(data) > 0:
                sns.histplot(data, bins=30, kde=False, ax=ax, color='skyblue')
                ax.axvline(0, color='red', linestyle='--')
                lower = np.percentile(data, lower_percentile)
                upper = np.percentile(data, upper_percentile)
                ax.axvline(lower, color='black', linestyle=':', linewidth=1)
                ax.axvline(upper, color='black', linestyle=':', linewidth=1)
                ax.set_title(f"Bin ({i},{j})\n90% CI: {lower:.1f}, {upper:.1f}%")
                ax.set_xlabel("RWC/LWC Difference (%)")
                ax.set_ylabel("Count")
            else:
                ax.set_visible(False)
    plt.suptitle("Bootstrapped RWC/LWC Differences Per Bin", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
gccn_values = np.array(list(average_gccn_per_flight.values()))
threshold = np.percentile(gccn_values, 50)

high_dates = {date for date, val in average_gccn_per_flight.items() if val >= threshold}
low_dates = {date for date, val in average_gccn_per_flight.items() if val < threshold}
water_dict = {(entry['Date'], entry['Time']): entry for entry in total_liquid_water}

for entry in total_combined_concentration:
    key = (entry['Date'], entry['Time'])
    if key in water_dict:
        entry['Total_Liquid_Water'] = water_dict[key]['Total_Liquid_Water']
        entry['RWC'] = water_dict[key]['RWC']

high_data = [e for e in total_combined_concentration if e['Date'] in high_dates]
low_data = [e for e in total_combined_concentration if e['Date'] in low_dates]

grouped_high = group_by_flight(high_data)
grouped_low = group_by_flight(low_data)

bin_means_high = compute_flight_bin_means(grouped_high)
bin_means_low = compute_flight_bin_means(grouped_low)

boot_distributions = bootstrap_bin_differences(bin_means_high, bin_means_low)
plot_histograms(boot_distributions)
def plot_histograms_with_flight_means_corrected(boot_dists, bin_means_high, bin_means_low, highlight_date="2022-05-10"):
    fig, axes = plt.subplots(nrows=len(x_bins)-1, ncols=len(y_bins)-1, figsize=(14, 10), sharex=True, sharey=True)

    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            ax = axes[i][j] if isinstance(axes[0], np.ndarray) else axes[i*(len(y_bins)-1)+j]
            dist = boot_dists[i][j]

            if len(dist) > 0:
                sns.histplot(dist, bins=30, kde=False, ax=ax, color='lightblue')
                ax.axvline(0, color='red', linestyle='--')
                lower = np.percentile(dist, lower_percentile)
                upper = np.percentile(dist, upper_percentile)
                ax.axvline(lower, color='black', linestyle=':', linewidth=1)
                ax.axvline(upper, color='black', linestyle=':', linewidth=1)

                for val in bin_means_high[i][j]:
                    ax.scatter(val, 0.5, color='blue', s=20, alpha=0.5, label='High GCCN' if (i == 0 and j == 0) else "")
                for val in bin_means_low[i][j]:
                    ax.scatter(val, 0.5, color='orange', s=20, alpha=0.5, label='Low GCCN' if (i == 0 and j == 0) else "")

                for dataset, color in zip([grouped_high, grouped_low], ['blue', 'orange']):
                    if highlight_date in dataset:
                        flight = dataset[highlight_date]
                        conc = np.array([e['Total_Combined_Concentration'] for e in flight])
                        lwc = np.array([e['Total_Liquid_Water'] for e in flight])
                        rwc = np.array([e['RWC'] for e in flight])
                        ratio = np.where(lwc > 0, rwc / lwc * 100, np.nan)
                        mask = (conc >= x_bins[i]) & (conc < x_bins[i+1]) & (lwc >= y_bins[j]) & (lwc < y_bins[j+1])
                        if np.any(mask):
                            val = np.nanmean(ratio[mask])
                            ax.scatter(val, 1, color='black', s=40, marker='X', label='May 10' if (i == 0 and j == 0) else "")

                ax.set_title(f"Bin ({i},{j})", fontsize=11)
                ax.set_xlabel("RWC/LWC Difference (%)")
                ax.set_ylabel("Boot Count")
            else:
                ax.set_visible(False)

    plt.suptitle("Flight-level RWC/LWC Diffs & Bootstraps — Check for May 10", fontsize=16, fontweight='bold')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    plt.tight_layout()
    plt.show()

print("\n=== Percentage of Bootstrapped Values > 0 Per Bin ===")
for i in range(len(x_bins)-1):
    for j in range(len(y_bins)-1):
        dist = boot_distributions[i][j]
        if len(dist) > 0:
            percent_above_zero = np.sum(dist > 0) / len(dist) * 100
            print(f"Bin ({i},{j}): {percent_above_zero:.1f}% of values > 0")
#%%
#mass
# ==========================================================
# Bootstrapping code for MASS (not concentration)
# Each "flight" == all seconds for a given Date
# ==========================================================
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 0) Make sure we have a mass dataset ----------
# Expectation for each row in total_combined_mass:
# {'Date': ..., 'Total_Combined_Mass': <float>, 'Total_Liquid_Water': <float>, 'RWC': <float>}
#
# If total_combined_mass is empty or missing keys, rebuild a simple version
# by assigning each second the flight-average mass for that Date.
def ensure_mass_dataset(total_combined_mass, total_liquid_water, average_mass_per_flight):
    need_rebuild = (
        (not total_combined_mass) or
        any(k not in total_combined_mass[0] for k in ['Date','Total_Combined_Mass','Total_Liquid_Water','RWC'])
    )
    if not need_rebuild:
        return total_combined_mass

    rebuilt = []
    for lw in total_liquid_water:
        d = lw['Date']
        if d in average_mass_per_flight:
            rebuilt.append({
                'Date': d,
                'Total_Combined_Mass': float(average_mass_per_flight[d]),  # flight-avg mass for that date
                'Total_Liquid_Water' : float(lw['Total_Liquid_Water']),
                'RWC'                : float(lw['RWC']),
            })
    return rebuilt

# Call the guard to ensure the dataset is usable
total_combined_mass = ensure_mass_dataset(total_combined_mass, total_liquid_water, average_mass_per_flight)

# ---------- 1) Bin setup from the data ----------
mass_vals = np.array([e['Total_Combined_Mass'] for e in total_combined_mass], dtype=float)
lwc_vals  = np.array([e['Total_Liquid_Water']  for e in total_combined_mass], dtype=float)

# Guard against non-positive values for log bins
mass_pos = mass_vals[mass_vals > 0]
lwc_pos  = lwc_vals[lwc_vals > 0]
if len(mass_pos) == 0 or len(lwc_pos) == 0:
    raise ValueError("No positive mass or LWC values available to build log-spaced bins.")

num_bins = 5
x_min, x_max = np.nanmin(mass_pos), np.nanmax(mass_pos)
y_min, y_max = np.nanmin(lwc_pos), np.nanmax(lwc_pos)

x_bins = np.logspace(np.log10(x_min), np.log10(x_max), num_bins)   # mass bins
y_bins = np.logspace(np.log10(y_min), np.log10(y_max), num_bins)   # LWC bins

# ---------- 2) Parameters ----------
n_bootstrap = 10_000
confidence_level = 0.90
lower_percentile = (1 - confidence_level) / 2 * 100
upper_percentile = (1 + confidence_level) / 2 * 100

# ---------- 3) Split flights: high vs low mass ----------
# Use the median of average_mass_per_flight to decide high/low by date
gccn_values = np.array(list(average_mass_per_flight.values()), dtype=float)
threshold = np.nanpercentile(gccn_values, 50)

high_dates = {d for d, v in average_mass_per_flight.items() if v >= threshold}
low_dates  = {d for d, v in average_mass_per_flight.items() if v <  threshold}

high_data = [e for e in total_combined_mass if e['Date'] in high_dates]
low_data  = [e for e in total_combined_mass if e['Date'] in low_dates]

# ---------- 4) Flight grouping ----------
def group_by_flight(data):
    flights = defaultdict(list)
    for entry in data:
        flights[entry['Date']].append(entry)  # "flight" == one Date
    return flights

grouped_high = group_by_flight(high_data)
grouped_low  = group_by_flight(low_data)

# ---------- 5) Compute flight-level bin means ----------
def compute_flight_bin_means(flight_data):
    # bin_means[i][j] will hold one value per flight that has data in bin (i,j)
    bin_means = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]

    for flight in flight_data.values():
        mass = np.array([e['Total_Combined_Mass'] for e in flight], dtype=float)
        lwc  = np.array([e['Total_Liquid_Water']  for e in flight], dtype=float)
        rwc  = np.array([e['RWC']                 for e in flight], dtype=float)
        ratio = np.where(lwc > 0, rwc / lwc * 100.0, np.nan)  # RWC/LWC (%)

        # Optional: debug ranges to confirm data fall into bins
        # print(f"Flight range: Mass {mass.min():.2f}-{mass.max():.2f}, LWC {lwc.min():.3f}-{lwc.max():.3f}")

        for i in range(len(x_bins)-1):
            for j in range(len(y_bins)-1):
                mask = (mass >= x_bins[i]) & (mass < x_bins[i+1]) & \
                       (lwc  >= y_bins[j]) & (lwc  < y_bins[j+1])
                if np.any(mask):
                    mean_val = np.nanmean(ratio[mask])
                    if not np.isnan(mean_val):
                        bin_means[i][j].append(float(mean_val))
    return bin_means

bin_means_high = compute_flight_bin_means(grouped_high)
bin_means_low  = compute_flight_bin_means(grouped_low)

# ---------- 6) Bootstrap differences per bin ----------
def bootstrap_bin_differences(bin_high, bin_low):
    boot_dists = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            high_vals = bin_high[i][j]
            low_vals  = bin_low[i][j]
            # Need at least 2 flights in each group for a sensible bootstrap
            if len(high_vals) >= 2 and len(low_vals) >= 2:
                # Resample at the "flight mean" level
                boot_diff = np.empty(n_bootstrap, dtype=float)
                for b in range(n_bootstrap):
                    m_high = np.mean(np.random.choice(high_vals, len(high_vals), replace=True))
                    m_low  = np.mean(np.random.choice(low_vals,  len(low_vals),  replace=True))
                    boot_diff[b] = m_high - m_low
                boot_dists[i][j] = boot_diff
            # else: remains empty
    return boot_dists

boot_distributions = bootstrap_bin_differences(bin_means_high, bin_means_low)

# ---------- 7) Plot per-bin bootstrap histograms ----------
def plot_histograms(boot_dists):
    fig, axes = plt.subplots(nrows=len(x_bins)-1, ncols=len(y_bins)-1, figsize=(14, 10), sharex=True, sharey=True)

    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            ax = axes[i][j] if isinstance(axes[0], np.ndarray) else axes[i*(len(y_bins)-1)+j]
            data = boot_dists[i][j]
            if isinstance(data, np.ndarray) and data.size > 0:
                sns.histplot(data, bins=30, kde=False, ax=ax)
                ax.axvline(0, color='red', linestyle='--', linewidth=1)

                lower = np.percentile(data, lower_percentile)
                upper = np.percentile(data, upper_percentile)
                ax.axvline(lower, color='black', linestyle=':', linewidth=1)
                ax.axvline(upper, color='black', linestyle=':', linewidth=1)

                pct_above_0 = (np.sum(data > 0) / data.size) * 100.0
                ax.set_title(f"Bin ({i},{j})\n90% CI: {lower:.1f}, {upper:.1f}%\n{pct_above_0:.1f}% > 0", fontsize=10)
                ax.set_xlabel("Δ(RWC/LWC %) High - Low")
                ax.set_ylabel("Boot count")
            else:
                ax.set_visible(False)

    plt.suptitle("Bootstrapped RWC/LWC Differences Per Bin (Mass-based)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

plot_histograms(boot_distributions)

# ---------- 8) Quick text summary ----------
print("\n=== Flights per bin (counts of flight means) ===")
for i in range(len(x_bins)-1):
    for j in range(len(y_bins)-1):
        print(f"Bin ({i},{j}): High={len(bin_means_high[i][j])}, Low={len(bin_means_low[i][j])}")


# %%
#all flghts 
num_bins = 5
x_bins = np.logspace(np.log10(x_min), np.log10(x_max), num_bins)
y_bins = np.logspace(np.log10(y_min), np.log10(y_max), num_bins)
n_bootstrap = 10000
confidence_level = 0.90
lower_percentile = (1 - confidence_level) / 2 * 100
upper_percentile = (1 + confidence_level) / 2 * 100


def group_by_flight(data):
    flights = defaultdict(list)
    for entry in data:
        flights[entry['Date']].append(entry)
    return flights


def compute_flight_bin_means(flight_data):
    bin_means = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    for flight in flight_data.values():
        conc = np.array([e['Total_Combined_Concentration'] for e in flight])
        lwc = np.array([e['Total_Liquid_Water'] for e in flight])
        rwc = np.array([e['Rain_Concentration'] for e in flight])
        ratio = np.where(lwc > 0, rwc / lwc * 100, np.nan)

        for i in range(len(x_bins) - 1):
            for j in range(len(y_bins) - 1):
                mask = (conc >= x_bins[i]) & (conc < x_bins[i+1]) & \
                       (lwc >= y_bins[j]) & (lwc < y_bins[j+1])
                if np.any(mask):
                    mean_val = np.nanmean(ratio[mask])
                    if not np.isnan(mean_val):
                        bin_means[i][j].append(mean_val)
    return bin_means

def bootstrap_bin_differences_random12(bin_high, bin_low):
    boot_dists = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            high_vals = bin_high[i][j]
            low_vals = bin_low[i][j]
            if len(high_vals) >= 2 and len(low_vals) >= 2:
                boot_diff = []
                for _ in range(n_bootstrap):
                    sampled_high = np.random.choice(high_vals, (len(high_vals)), replace=True)
                    sampled_low = np.random.choice(low_vals, (len(low_vals)), replace=True)
                    boot_diff.append(np.mean(sampled_high) - np.mean(sampled_low))
                boot_dists[i][j] = np.array(boot_diff)
    return boot_dists

def plot_histograms_with_percentage(boot_dists):
    fig, axes = plt.subplots(nrows=len(x_bins)-1, ncols=len(y_bins)-1, figsize=(14, 10), sharex=True, sharey=True)

    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            ax = axes[i][j] if isinstance(axes[0], np.ndarray) else axes[i*(len(y_bins)-1)+j]
            dist = boot_dists[i][j]

            if len(dist) > 0:
                sns.histplot(dist, bins=30, kde=False, ax=ax, color='skyblue')
                ax.axvline(0, color='red', linestyle='--')
                lower = np.percentile(dist, lower_percentile)
                upper = np.percentile(dist, upper_percentile)
                ax.axvline(lower, color='black', linestyle=':', linewidth=1)
                ax.axvline(upper, color='black', linestyle=':', linewidth=1)

                percent_above_zero = np.sum(dist > 0) / len(dist) * 100
                mean_val = np.mean(dist)
                std_val = np.std(dist)

                annotation = f"{percent_above_zero:.1f}% > 0\nμ = {mean_val:.2f}, σ = {std_val:.2f}"
                ax.text(
                    0.98, 0.95,
                    annotation,
                    transform=ax.transAxes,
                    ha='right',
                    va='top',
                    fontsize=14,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
                )

                ax.set_xlabel("RWC/LWC Difference (%)", fontsize=18, fontweight='bold')
                ax.set_ylabel("Count", fontsize=18, fontweight='bold')
                ax.tick_params(axis='both', labelsize=16)
                ax.tick_params(axis='x', labelbottom=True)


                ax.set_xlabel("RWC/LWC Difference (%)", fontsize=18, fontweight='bold')
                ax.set_ylabel("Count", fontsize=18, fontweight='bold')
                ax.tick_params(axis='both', labelsize=16)
            else:
                ax.set_visible(False)

    plt.suptitle("Bootstrapping Using All Flights Per Iteration\nBelow Cloud Base January-June 2022", fontsize=19, fontweight='bold')
    plt.tight_layout()
    plt.show()
gccn_values = np.array(list(average_gccn_per_flight.values()))
threshold = np.percentile(gccn_values, 50)
high_dates = {date for date, val in average_gccn_per_flight.items() if val >= threshold}
low_dates = {date for date, val in average_gccn_per_flight.items() if val < threshold}
high_gccn_data = [entry for entry in total_combined_concentration if entry['Date'] in high_dates]
low_gccn_data = [entry for entry in total_combined_concentration if entry['Date'] in low_dates]
grouped_high = group_by_flight(high_gccn_data)
grouped_low = group_by_flight(low_gccn_data)
bin_means_high = compute_flight_bin_means(grouped_high)
bin_means_low = compute_flight_bin_means(grouped_low)
boot_distributions = bootstrap_bin_differences_random12(bin_means_high, bin_means_low)
plot_histograms_with_percentage(boot_distributions)
print("\n=== One-sided T-test Results Per Bin (H₀: mean ≤ 0, H₁: mean > 0) ===")
for i in range(len(x_bins) - 1):
    for j in range(len(y_bins) - 1):
        dist = boot_distributions[i][j]
        if len(dist) > 1:
            t_stat, p_two_sided = stats.ttest_1samp(dist, popmean=0)
            p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - (p_two_sided / 2)
            print(
                f"Bin ({i},{j}): mean = {np.mean(dist):.2f}, std = {np.std(dist):.2f}, "
                f"t = {t_stat:.2f}, p = {p_one_sided:.2e}"
            )
#%%
#mass

# %%
#separately for rwc and lwc 
num_bins = 5
x_bins = np.logspace(np.log10(x_min), np.log10(x_max), num_bins)
y_bins = np.logspace(np.log10(y_min), np.log10(y_max), num_bins)
n_bootstrap = 10000
confidence_level = 0.90
lower_percentile = (1 - confidence_level) / 2 * 100
upper_percentile = (1 + confidence_level) / 2 * 100
def group_by_flight(data):
    flights = defaultdict(list)
    for entry in data:
        flights[entry['Date']].append(entry)
    return flights
def compute_flight_bin_means_RWC(flight_data):
    bin_means = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    for flight in flight_data.values():
        conc = np.array([e['Total_Combined_Concentration'] for e in flight])
        lwc = np.array([e['Total_Liquid_Water'] for e in flight])
        rwc = np.array([e['Rain_Concentration'] for e in flight])
        for i in range(len(x_bins) - 1):
            for j in range(len(y_bins) - 1):
                mask = (conc >= x_bins[i]) & (conc < x_bins[i+1]) & \
                       (lwc >= y_bins[j]) & (lwc < y_bins[j+1])
                if np.any(mask):
                    mean_val = np.nanmean(rwc[mask])
                    if not np.isnan(mean_val):
                        bin_means[i][j].append(mean_val)
    return bin_means
def bootstrap_bin_differences_random12(bin_high, bin_low):
    boot_dists = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            high_vals = bin_high[i][j]
            low_vals = bin_low[i][j]
            if len(high_vals) >= 2 and len(low_vals) >= 2:
                boot_diff = []
                for _ in range(n_bootstrap):
                    sampled_high = np.random.choice(high_vals, len(high_vals), replace=True)
                    sampled_low = np.random.choice(low_vals, len(low_vals), replace=True)
                    boot_diff.append(np.mean(sampled_high) - np.mean(sampled_low))
                boot_dists[i][j] = np.array(boot_diff)
    return boot_dists
def plot_histograms_with_percentage(boot_dists):
    fig, axes = plt.subplots(nrows=len(x_bins)-1, ncols=len(y_bins)-1, figsize=(14, 10), sharex=True, sharey=True)

    bin_edges = np.linspace(-0.040, 0.040, 15)
    x_axis_min, x_axis_max = -0.040, 0.040
    y_axis_max = 8000

    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            ax = axes[i][j]
            dist = boot_dists[i][j]
            if len(dist) > 0:
                sns.histplot(dist, bins=bin_edges, kde=False, ax=ax, color='skyblue')
                ax.axvline(0, color='red', linestyle='--')
                lower = np.percentile(dist, lower_percentile)
                upper = np.percentile(dist, upper_percentile)
                ax.axvline(lower, color='black', linestyle=':', linewidth=1)
                ax.axvline(upper, color='black', linestyle=':', linewidth=1)
                percent_above_zero = np.sum(dist > 0) / len(dist) * 100
                mean_val = np.mean(dist)
                std_val = np.std(dist)
                annotation = f"{percent_above_zero:.1f}% > 0\nμ = {mean_val:.2f}, σ = {std_val:.2f}"
                ax.text(0.98, 0.95, annotation, transform=ax.transAxes,
                        ha='right', va='top', fontsize=14,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                ax.set_xlabel("RWC Difference (g/m³)", fontsize=18, fontweight='bold')
                ax.set_ylabel("Count", fontsize=18, fontweight='bold')
                ax.tick_params(axis='both', labelsize=16)
                ax.set_xlim(x_axis_min, x_axis_max)
                ax.set_ylim(0, y_axis_max)
            else:
                ax.set_visible(False)

    plt.suptitle("Bootstrapped RWC Differences (All Flights/Iter)", fontsize=19, fontweight='bold')
    plt.tight_layout()
    plt.show()
gccn_values = np.array(list(average_gccn_per_flight.values()))
threshold = np.percentile(gccn_values, 50)
high_dates = {date for date, val in average_gccn_per_flight.items() if val >= threshold}
low_dates = {date for date, val in average_gccn_per_flight.items() if val < threshold}
high_data = [entry for entry in total_combined_concentration if entry['Date'] in high_dates]
low_data = [entry for entry in total_combined_concentration if entry['Date'] in low_dates]

grouped_high = group_by_flight(high_data)
grouped_low = group_by_flight(low_data)
bin_means_high = compute_flight_bin_means_RWC(grouped_high)
bin_means_low = compute_flight_bin_means_RWC(grouped_low)
boot_distributions = bootstrap_bin_differences_random12(bin_means_high, bin_means_low)
plot_histograms_with_percentage(boot_distributions)
print("\n=== One-sided T-test Results Per Bin (H₀: mean ≤ 0, H₁: mean > 0) ===")
for i in range(len(x_bins) - 1):
    for j in range(len(y_bins) - 1):
        dist = boot_distributions[i][j]
        if len(dist) > 1:
            t_stat, p_two_sided = stats.ttest_1samp(dist, popmean=0)
            p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - (p_two_sided / 2)
            print(f"Bin ({i},{j}): mean = {np.mean(dist):.2f}, std = {np.std(dist):.2f}, "
                  f"t = {t_stat:.2f}, p = {p_one_sided:.2e}")

# %%
num_bins = 5
x_bins = np.logspace(np.log10(x_min), np.log10(x_max), num_bins)
y_bins = np.logspace(np.log10(y_min), np.log10(y_max), num_bins)
n_bootstrap = 10000
confidence_level = 0.90
lower_percentile = (1 - confidence_level) / 2 * 100
upper_percentile = (1 + confidence_level) / 2 * 100
def group_by_flight(data):
    flights = defaultdict(list)
    for entry in data:
        flights[entry['Date']].append(entry)
    return flights
def compute_flight_bin_means_LWC(flight_data):
    bin_means = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    for flight in flight_data.values():
        conc = np.array([e['Total_Combined_Concentration'] for e in flight])
        lwc = np.array([e['Total_Liquid_Water'] for e in flight])
        for i in range(len(x_bins) - 1):
            for j in range(len(y_bins) - 1):
                mask = (conc >= x_bins[i]) & (conc < x_bins[i+1]) & \
                       (lwc >= y_bins[j]) & (lwc < y_bins[j+1])
                if np.any(mask):
                    mean_val = np.nanmean(lwc[mask])
                    if not np.isnan(mean_val):
                        bin_means[i][j].append(mean_val)
    return bin_means
def bootstrap_bin_differences_random12(bin_high, bin_low):
    boot_dists = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            high_vals = bin_high[i][j]
            low_vals = bin_low[i][j]
            if len(high_vals) >= 2 and len(low_vals) >= 2:
                boot_diff = []
                for _ in range(n_bootstrap):
                    sampled_high = np.random.choice(high_vals, len(high_vals), replace=True)
                    sampled_low = np.random.choice(low_vals, len(low_vals), replace=True)
                    boot_diff.append(np.mean(sampled_high) - np.mean(sampled_low))
                boot_dists[i][j] = np.array(boot_diff)
    return boot_dists
def plot_histograms_with_percentage(boot_dists):
    fig, axes = plt.subplots(nrows=len(x_bins)-1, ncols=len(y_bins)-1, figsize=(14, 10), sharex=True, sharey=True)
    x_axis_min, x_axis_max = -0.050, 0.050
    y_axis_max = 6000


    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            ax = axes[i][j]
            dist = boot_dists[i][j]
            if len(dist) > 0:
                bin_edges = np.linspace(-0.050, 0.050, 15)  # 30 bins across full fixed range
                sns.histplot(dist, bins=bin_edges, kde=False, ax=ax, color='skyblue')
                ax.axvline(0, color='red', linestyle='--')
                lower = np.percentile(dist, lower_percentile)
                upper = np.percentile(dist, upper_percentile)
                ax.axvline(lower, color='black', linestyle=':', linewidth=1)
                ax.axvline(upper, color='black', linestyle=':', linewidth=1)
                percent_above_zero = np.sum(dist > 0) / len(dist) * 100
                mean_val = np.mean(dist)
                std_val = np.std(dist)
                annotation = f"{percent_above_zero:.1f}% > 0\nμ = {mean_val:.2f}, σ = {std_val:.2f}"
                ax.text(0.98, 0.95, annotation, transform=ax.transAxes,
                        ha='right', va='top', fontsize=14,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                ax.set_xlabel("LWC Difference (g/m³)", fontsize=18, fontweight='bold')
                ax.set_ylabel("Count", fontsize=18, fontweight='bold')
                ax.tick_params(axis='both', labelsize=16)
                ax.set_xlim(x_axis_min, x_axis_max)
                ax.set_ylim(0, y_axis_max)
            else:
                ax.set_visible(False)

    plt.suptitle("Bootstrapped LWC Differences (All Flights/Iter)", fontsize=19, fontweight='bold')
    plt.tight_layout()
    plt.show()
gccn_values = np.array(list(average_gccn_per_flight.values()))
threshold = np.percentile(gccn_values, 50)
high_dates = {date for date, val in average_gccn_per_flight.items() if val >= threshold}
low_dates = {date for date, val in average_gccn_per_flight.items() if val < threshold}
high_data = [entry for entry in total_combined_concentration if entry['Date'] in high_dates]
low_data = [entry for entry in total_combined_concentration if entry['Date'] in low_dates]
grouped_high = group_by_flight(high_data)
grouped_low = group_by_flight(low_data)
bin_means_high = compute_flight_bin_means_LWC(grouped_high)
bin_means_low = compute_flight_bin_means_LWC(grouped_low)
boot_distributions = bootstrap_bin_differences_random12(bin_means_high, bin_means_low)
plot_histograms_with_percentage(boot_distributions)
print("\n=== One-sided T-test Results Per Bin (H₀: mean ≤ 0, H₁: mean > 0) ===")
for i in range(len(x_bins) - 1):
    for j in range(len(y_bins) - 1):
        dist = boot_distributions[i][j]
        if len(dist) > 1:
            t_stat, p_two_sided = stats.ttest_1samp(dist, popmean=0)
            p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - (p_two_sided / 2)
            print(f"Bin ({i},{j}): mean = {np.mean(dist):.2f}, std = {np.std(dist):.2f}, "
                  f"t = {t_stat:.2f}, p = {p_one_sided:.2e}")
#%%
#trying histograms for ratio of high/low rwc 
num_bins = 5
x_bins = np.logspace(np.log10(x_min), np.log10(x_max), num_bins)
y_bins = np.logspace(np.log10(y_min), np.log10(y_max), num_bins)
n_bootstrap = 10000
confidence_level = 0.90
lower_percentile = (1 - confidence_level) / 2 * 100
upper_percentile = (1 + confidence_level) / 2 * 100
def group_by_flight(data):
    flights = defaultdict(list)
    for entry in data:
        flights[entry['Date']].append(entry)
    return flights
def compute_flight_bin_means_RWC(flight_data):
    bin_means = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    for flight in flight_data.values():
        conc = np.array([e['Total_Combined_Concentration'] for e in flight])
        lwc = np.array([e['Total_Liquid_Water'] for e in flight])
        rwc = np.array([e['Rain_Concentration'] for e in flight])
        for i in range(len(x_bins) - 1):
            for j in range(len(y_bins) - 1):
                mask = (conc >= x_bins[i]) & (conc < x_bins[i + 1]) & \
                       (lwc >= y_bins[j]) & (lwc < y_bins[j + 1])
                if np.any(mask):
                    mean_val = np.nanmean(rwc[mask])
                    if not np.isnan(mean_val):
                        bin_means[i][j].append(mean_val)
    return bin_means
def bootstrap_ratio_distributions(bin_high, bin_low):
    boot_ratios = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            high_vals = bin_high[i][j]
            low_vals = bin_low[i][j]
            if len(high_vals) >= 2 and len(low_vals) >= 2:
                boot_sample_ratios = []
                for _ in range(n_bootstrap):
                    sampled_high = np.random.choice(high_vals, len(high_vals), replace=True)
                    sampled_low = np.random.choice(low_vals, len(low_vals), replace=True)
                    sampled_low = np.where(sampled_low == 0, np.nan, sampled_low)
                    ratio = np.nanmean(sampled_high) / np.nanmean(sampled_low)
                    boot_sample_ratios.append(ratio)
                boot_ratios[i][j] = np.array(boot_sample_ratios)
    return boot_ratios
def plot_histograms_with_percentage_ratio(boot_dists):
    fig, axes = plt.subplots(
        nrows=len(x_bins) - 1,
        ncols=len(y_bins) - 1,
        figsize=(14, 10),
        sharex=True,
        sharey=True
    )
    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            ax = axes[i][j]
            dist = boot_dists[i][j]
            if len(dist) > 0:
                clipped_ratio = np.clip(dist, -10, 10)
                bin_edges = np.arange(-10, 10, 1)

                ax.hist(clipped_ratio, bins=bin_edges, color='skyblue', edgecolor='black')

                ax.axvline(1, color='red', linestyle='--')
                lower = np.percentile(dist, lower_percentile)
                upper = np.percentile(dist, upper_percentile)
                ax.axvline(lower, color='black', linestyle=':', linewidth=1)
                ax.axvline(upper, color='black', linestyle=':', linewidth=1)

                percent_above_one = np.sum(dist > 1) / len(dist) * 100
                mean_val = np.nanmean(dist)
                std_val = np.nanstd(dist)
                annotation = f"{percent_above_one:.1f}% > 1\nμ = {mean_val:.2f}, σ = {std_val:.2f}"
                ax.text(0.98, 0.95, annotation, transform=ax.transAxes,
                        ha='right', va='top', fontsize=14,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

                ax.set_xlim(-10, 10)
                ax.tick_params(axis='both', labelsize=18)
            else:
                ax.set_visible(False)

   
    fig.suptitle("CAS (in cloud)\nBootstrapped RWC Ratio (High/Low GCCN)\n January–June 2022", fontsize=20, fontweight='bold')
    fig.supxlabel("RWC Ratio (High / Low)", fontsize=20, fontweight='bold')
    fig.supylabel("Count", fontsize=20, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    plt.show()
gccn_values = np.array(list(average_gccn_per_flight.values()))
threshold = np.percentile(gccn_values, 50)
high_dates = {date for date, val in average_gccn_per_flight.items() if val >= threshold}
low_dates = {date for date, val in average_gccn_per_flight.items() if val < threshold}
high_data = [entry for entry in total_combined_concentration if entry['Date'] in high_dates]
low_data = [entry for entry in total_combined_concentration if entry['Date'] in low_dates]
grouped_high = group_by_flight(high_data)
grouped_low = group_by_flight(low_data)
bin_means_high = compute_flight_bin_means_RWC(grouped_high)
bin_means_low = compute_flight_bin_means_RWC(grouped_low)
boot_ratio_distributions = bootstrap_ratio_distributions(bin_means_high, bin_means_low)
plot_histograms_with_percentage_ratio(boot_ratio_distributions)
#%%
# ==========================================================
# Bootstrapped RWC Ratio (High / Low) per (Mass, LWC) bin
# ==========================================================
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 1) Choose bin edges (logspace) ----------
# If you already have x_min/x_max/y_min/y_max, keep them.
all_mass = np.array([e['Total_Combined_Mass'] for e in total_combined_mass if e['Total_Combined_Mass'] > 0])
all_lwc  = np.array([e['Total_Liquid_Water']  for e in total_combined_mass if e['Total_Liquid_Water']  > 0])

num_bins = 5
x_min = float(np.nanmax([np.nanmin(all_mass), 1e-3]))   # avoid log(0)
x_max = float(np.nanmax(all_mass))
y_min = float(np.nanmax([np.nanmin(all_lwc),  1e-3]))
y_max = float(np.nanmax(all_lwc))

x_bins = np.logspace(np.log10(x_min), np.log10(x_max), num_bins)
y_bins = np.logspace(np.log10(y_min), np.log10(y_max), num_bins)

# Bootstrap settings
n_bootstrap = 10000
confidence_level = 0.90
lower_percentile = (1 - confidence_level) / 2 * 100    # 5
upper_percentile = (1 + confidence_level) / 2 * 100    # 95

# ---------- 2) High/Low MASS split ----------
gccn_values = np.array(list(average_mass_per_flight.values()))
threshold   = np.percentile(gccn_values, 50)  # median split

high_dates = {d for d, v in average_mass_per_flight.items() if v >= threshold}
low_dates  = {d for d, v in average_mass_per_flight.items() if v <  threshold}

high_data = [e for e in total_combined_mass if e['Date'] in high_dates]
low_data  = [e for e in total_combined_mass if e['Date'] in low_dates]

# ---------- 3) Helpers ----------
def group_by_flight(data):
    flights = defaultdict(list)
    for entry in data:
        flights[entry['Date']].append(entry)   # one group per flight/day
    return flights

def compute_flight_bin_means_RWC_mass(flight_data):
    """
    For each flight, compute the mean RWC within each (mass, LWC) bin.
    Return a [nx][ny] list of lists where each cell contains
    the per-flight means for that bin across flights.
    """
    bin_means = [[[] for _ in range(len(y_bins)-1)] for _ in range(len(x_bins)-1)]
    for flight in flight_data.values():
        mass = np.asarray([e['Total_Combined_Mass'] for e in flight], dtype=float)
        lwc  = np.asarray([e['Total_Liquid_Water']  for e in flight], dtype=float)
        rwc  = np.asarray([e['RWC']                 for e in flight], dtype=float)

        # valid points for ratio-free mean of RWC (we're averaging RWC itself here)
        valid = np.isfinite(mass) & np.isfinite(lwc) & np.isfinite(rwc) & (mass > 0) & (lwc > 0)
        if not np.any(valid):
            continue

        mass = mass[valid]; lwc = lwc[valid]; rwc = rwc[valid]

        for i in range(len(x_bins)-1):
            for j in range(len(y_bins)-1):
                mask = (mass >= x_bins[i]) & (mass < x_bins[i+1]) & \
                       (lwc  >= y_bins[j]) & (lwc  < y_bins[j+1])
                if np.any(mask):
                    mean_rwc = np.nanmean(rwc[mask])
                    if np.isfinite(mean_rwc):
                        bin_means[i][j].append(mean_rwc)
    return bin_means

def summarize_bin_coverage(bin_high, bin_low, label):
    print(f"\n=== {label} (flights contributing per bin) ===")
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            nh, nl = len(bin_high[i][j]), len(bin_low[i][j])
            if nh + nl > 0:
                print(f"Bin ({i},{j}): High={nh}, Low={nl}")

def bootstrap_ratio_distributions_mass(bin_high, bin_low, n_bootstrap=10000, eps=1e-12):
    """
    Bootstrap ratio = mean(High) / mean(Low) per bin.
    Works with as few as 1 flight in each group (resamples with replacement).
    Skips bins where the bootstrapped Low mean hits zero/NaN too often.
    """
    boot_ratios = [[[] for _ in range(len(y_bins)-1)] for _ in range(len(x_bins)-1)]
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            H = np.asarray(bin_high[i][j], dtype=float)
            L = np.asarray(bin_low[i][j],  dtype=float)
            if len(H) >= 1 and len(L) >= 1:
                # sample at least size 2 to introduce bootstrap variability
                hsize = max(2, len(H))
                lsize = max(2, len(L))
                ratios = np.empty(n_bootstrap, dtype=float)
                for b in range(n_bootstrap):
                    mH = np.mean(np.random.choice(H, size=hsize, replace=True))
                    mL = np.mean(np.random.choice(L, size=lsize, replace=True))
                    # Guard against ~0 in denominator
                    if not np.isfinite(mL) or abs(mL) < eps:
                        ratios[b] = np.nan
                    else:
                        ratios[b] = mH / mL
                ratios = ratios[np.isfinite(ratios)]
                if ratios.size > 0:
                    boot_ratios[i][j] = ratios
    return boot_ratios

def plot_histograms_with_percentage_ratio_mass(boot_dists, title):
    fig, axes = plt.subplots(len(x_bins)-1, len(y_bins)-1, figsize=(14,10), sharex=True, sharey=True)
    axes = np.array(axes, ndmin=2)

    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            ax = axes[i, j]
            dist = boot_dists[i][j]
            if len(dist) > 0:
                # ratios are positive; clip for display if there are long tails
                clipped = np.clip(dist, 0, 5)
                ax.hist(clipped, bins=30, edgecolor='black')
                ax.axvline(1.0, color='red', linestyle='--', linewidth=1)

                lower = np.percentile(dist, lower_percentile)
                upper = np.percentile(dist, upper_percentile)
                ax.axvline(lower, color='black', linestyle=':', linewidth=1)
                ax.axvline(upper, color='black', linestyle=':', linewidth=1)

                pct_gt1 = 100.0 * np.mean(dist > 1.0)
                mu = np.nanmean(dist); sd = np.nanstd(dist)
                ax.set_title(f"Bin ({i},{j})\n90% CI: {lower:.2f}, {upper:.2f}\n{pct_gt1:.1f}% > 1",
                             fontsize=10)
                ax.text(0.98, 0.92, f"μ={mu:.2f}\nσ={sd:.2f}",
                        transform=ax.transAxes, ha='right', va='top',
                        fontsize=9, bbox=dict(boxstyle='round,pad=0.25',
                                              facecolor='white', alpha=0.7))
                ax.set_xlim(0, 5)
            else:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"Bin ({i},{j})", fontsize=10)
            ax.set_xlabel("High / Low mean RWC")
            ax.set_ylabel("Count")
            ax.tick_params(axis='both', labelsize=10)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ---------- 4) Compute per-flight bin means ----------
grouped_high = group_by_flight(high_data)
grouped_low  = group_by_flight(low_data)

bin_means_high = compute_flight_bin_means_RWC_mass(grouped_high)
bin_means_low  = compute_flight_bin_means_RWC_mass(grouped_low)

# (optional) see coverage
summarize_bin_coverage(bin_means_high, bin_means_low, "RWC means per (Mass,LWC) bin")

# ---------- 5) Bootstrap ratios & plot ----------
boot_ratio_distributions = bootstrap_ratio_distributions_mass(bin_means_high, bin_means_low,
                                                              n_bootstrap=n_bootstrap)
plot_histograms_with_percentage_ratio_mass(
    boot_ratio_distributions,
    title="CAS (in cloud)\nBootstrapped RWC Ratio (High/Low GCCN mass)\nJan–Jun 2022"
)

#%%
#fixing ranges 
conc_all = np.array([e['Total_Combined_Concentration'] for e in total_combined_concentration])
lwc_all  = np.array([e['Total_Liquid_Water'] for e in total_combined_concentration])
x_min = np.nanmin(conc_all[conc_all > 0])
x_max = np.nanmax(conc_all)
y_min = np.nanmin(lwc_all[lwc_all > 0])
y_max = np.nanmax(lwc_all)
print("Concentration range (Nr+Nc):", x_min, x_max)
print("LWC range:", y_min, y_max)
num_bins = 5
x_bins = np.logspace(np.log10(x_min), np.log10(x_max), num_bins)
y_bins = np.logspace(np.log10(y_min), np.log10(y_max), num_bins)
n_bootstrap = 10000
confidence_level = 0.90
lower_percentile = (1 - confidence_level) / 2 * 100
upper_percentile = (1 + confidence_level) / 2 * 100
def group_by_flight(data):
    flights = defaultdict(list)
    for entry in data:
        flights[entry['Date']].append(entry)
    return flights
def compute_flight_bin_means_RWC(flight_data):
    bin_means = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    for flight in flight_data.values():
        conc = np.array([e['Total_Combined_Concentration'] for e in flight])
        lwc = np.array([e['Total_Liquid_Water'] for e in flight])
        rwc = np.array([e['Rain_Concentration'] for e in flight])
        for i in range(len(x_bins) - 1):
            for j in range(len(y_bins) - 1):
                mask = (conc >= x_bins[i]) & (conc < x_bins[i + 1]) & \
                       (lwc >= y_bins[j]) & (lwc < y_bins[j + 1])
                if np.any(mask):
                    mean_val = np.nanmean(rwc[mask])
                    if not np.isnan(mean_val):
                        bin_means[i][j].append(mean_val)
    return bin_means
def bootstrap_ratio_distributions(bin_high, bin_low):
    boot_ratios = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            high_vals = bin_high[i][j]
            low_vals = bin_low[i][j]
            if len(high_vals) >= 2 and len(low_vals) >= 2:
                boot_sample_ratios = []
                for _ in range(n_bootstrap):
                    sampled_high = np.random.choice(high_vals, len(high_vals), replace=True)
                    sampled_low = np.random.choice(low_vals, len(low_vals), replace=True)
                    sampled_low = np.where(sampled_low == 0, np.nan, sampled_low)
                    ratio = np.nanmean(sampled_high) / np.nanmean(sampled_low)
                    boot_sample_ratios.append(ratio)
                boot_ratios[i][j] = np.array(boot_sample_ratios)
    return boot_ratios
def plot_histograms_with_percentage_ratio(boot_dists):
    fig, axes = plt.subplots(
        nrows=len(x_bins) - 1,
        ncols=len(y_bins) - 1,
        figsize=(14, 10),
        sharex=True,
        sharey=True
    )
    ratio_min, ratio_max = np.inf, -np.inf
    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            if len(boot_dists[i][j]) > 0:
                ratio_min = min(ratio_min, np.nanmin(boot_dists[i][j]))
                ratio_max = max(ratio_max, np.nanmax(boot_dists[i][j]))
    ratio_min = max(0, ratio_min)
    ratio_max = min(10, ratio_max)
    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            ax = axes[i][j]
            dist = boot_dists[i][j]
            if len(dist) > 0:
                ax.hist(dist, bins=30, color='skyblue', edgecolor='black')

                ax.axvline(1, color='red', linestyle='--')
                lower = np.percentile(dist, lower_percentile)
                upper = np.percentile(dist, upper_percentile)
                ax.axvline(lower, color='black', linestyle=':', linewidth=1)
                ax.axvline(upper, color='black', linestyle=':', linewidth=1)

                percent_above_one = np.sum(dist > 1) / len(dist) * 100
                mean_val = np.nanmean(dist)
                std_val = np.nanstd(dist)
                annotation = f"{percent_above_one:.1f}% > 1\nμ = {mean_val:.2f}, σ = {std_val:.2f}"
                ax.text(0.98, 0.95, annotation, transform=ax.transAxes,
                        ha='right', va='top', fontsize=12,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

                ax.set_xlim(ratio_min, ratio_max)
                ax.tick_params(axis='both', labelsize=10)
            else:
                ax.set_visible(False)

    fig.suptitle("CAS (in cloud)\nBootstrapped RWC Ratio (High/Low GCCN)\nJanuary–June 2022",
                 fontsize=18, fontweight='bold')
    fig.supxlabel("RWC Ratio (High / Low)", fontsize=16, fontweight='bold')
    fig.supylabel("Count", fontsize=16, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    plt.show()
gccn_values = np.array(list(average_gccn_per_flight.values()))
threshold = np.percentile(gccn_values, 50)
high_dates = {date for date, val in average_gccn_per_flight.items() if val >= threshold}
low_dates = {date for date, val in average_gccn_per_flight.items() if val < threshold}
high_data = [entry for entry in total_combined_concentration if entry['Date'] in high_dates]
low_data  = [entry for entry in total_combined_concentration if entry['Date'] in low_dates]
grouped_high = group_by_flight(high_data)
grouped_low  = group_by_flight(low_data)
bin_means_high = compute_flight_bin_means_RWC(grouped_high)
bin_means_low  = compute_flight_bin_means_RWC(grouped_low)
boot_ratio_distributions = bootstrap_ratio_distributions(bin_means_high, bin_means_low)
plot_histograms_with_percentage_ratio(boot_ratio_distributions)

# %%
#boostrapped heatmap
heatmap_data = np.full((len(x_bins) - 1, len(y_bins) - 1), np.nan)
min_rwc_threshold = 0.005 
ratio_cap = 5.0            
for i in range(len(x_bins) - 1): 
    for j in range(len(y_bins) - 1): 
        dist = boot_ratio_distributions[i][j]
        if len(dist) > 0:
            dist = dist[np.isfinite(dist)]
            dist = dist[dist <= ratio_cap]
            if len(dist) > 0:
                heatmap_data[i][j] = np.nanmean(dist)  
masked_ratio_rwc = np.ma.masked_where(np.isnan(heatmap_data), heatmap_data)
custom_bounds = [0.5, 1.0, 1.1, 1.4, 1.6, 1.8, 2.0, 2.5]
cmap = plt.cm.viridis.copy()
cmap.set_bad(color='gray')
norm = BoundaryNorm(boundaries=custom_bounds, ncolors=cmap.N)
fig, ax = plt.subplots(figsize=(8, 6))
img = ax.pcolormesh(x_bins, y_bins, masked_ratio_rwc.T,
                    cmap=cmap, norm=norm, shading="auto")
cbar = plt.colorbar(img, ticks=custom_bounds)
cbar.set_label("Bootstrapped RWC Ratio (High / Low)", fontsize=19, fontweight="bold")
cbar.ax.tick_params(labelsize=19)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Nr+Nc (cm$^{-3}$)", fontsize=19, fontweight="bold")
ax.set_ylabel(r"LWC (g m$^{-3}$)", fontsize=19, fontweight="bold")
ax.set_title("CAS (in cloud)\nRWC Ratio High / Low GCCN Flights\nJanuary–June 2022",
             fontsize=19, fontweight="bold")
ax.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
ax.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
for i in range(len(y_bins) - 1):
    for j in range(len(x_bins) - 1):
        dist = boot_ratio_distributions[j][i]
        if len(dist) > 0:
            dist = dist[np.isfinite(dist)]
            dist = dist[dist <= ratio_cap]
            if len(dist) == 0:
                continue

            percent_above = np.sum(dist > 1) / len(dist) * 100
            mean_val = np.nanmean(dist)
            std_val = np.nanstd(dist)

            label = f"{percent_above:.1f}% > 1\nμ={mean_val:.2f}, σ={std_val:.2f}"
            x_center = 10 ** ((np.log10(x_bins[j]) + np.log10(x_bins[j + 1])) / 2)
            y_center = 10 ** ((np.log10(y_bins[i]) + np.log10(y_bins[i + 1])) / 2)

            txt = ax.text(x_center, y_center, label,
                          ha='center', va='center',
                          fontsize=10, fontweight='bold', linespacing=1.1)
            txt.set_path_effects([path_effects.withStroke(linewidth=1.3, foreground='white')])
plt.tight_layout()
plt.show()
# %%
