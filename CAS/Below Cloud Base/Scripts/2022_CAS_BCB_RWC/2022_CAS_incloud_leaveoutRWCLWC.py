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
                    'LWC': lwc_val,  # Liquid water content (in kg/m³)
                    'Total_Concentration': total_concentration  # Converted to /m³
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
img = plt.pcolormesh(xedges, yedges, masked_avg_rwc.T, cmap="viridis", shading='auto')

gray_mask = np.isnan(avg_rwc)
gray_values = np.full_like(avg_rwc, np.nan)
gray_values[gray_mask] = 1
plt.pcolormesh(xedges, yedges, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)

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
plt.title('Mean RWC\nJanuary–June 2022 (CAS in cloud)', fontsize=18, fontweight='bold')
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
from matplotlib.colors import PowerNorm, ListedColormap

vmin = min(np.nanmin(avg_rwc), np.nanmin(avg_lwc))
vmax = max(np.nanmax(avg_rwc), np.nanmax(avg_lwc))

norm = PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)  # You can adjust gamma for more/less stretch
masked_avg_rwc = np.ma.masked_where(np.isnan(avg_rwc), avg_rwc)

plt.figure(figsize=(8, 6))
img = plt.pcolormesh(xedges, yedges, masked_avg_rwc.T, cmap="viridis", shading='auto', norm=norm)

gray_mask = np.isnan(avg_rwc)
gray_values = np.full_like(avg_rwc, np.nan)
gray_values[gray_mask] = 1
plt.pcolormesh(xedges, yedges, gray_values.T, cmap=ListedColormap(["gray"]), shading='auto', alpha=0.6)

cbar = plt.colorbar(img)
cbar.set_label("Mean RWC (g m$^{-3}$)", fontsize=18, fontweight='bold') 
cbar.ax.tick_params(labelsize=18, width=2, length=5)
for t in cbar.ax.get_yticklabels():  
    t.set_fontweight('bold')

plt.xscale('log')
plt.yscale('log')
plt.ylim(0.1, 0.3)
plt.xlim(50, 200)
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.xlabel('Nr+Nc /cm³', fontsize=19, fontweight='bold')
plt.ylabel('LWC g/m³', fontsize=19, fontweight='bold')
plt.title('CAS Mean RWC\nCloudy conditions January–June 2022', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()

# === Plot LWC ===
masked_avg_lwc = np.ma.masked_where(np.isnan(avg_lwc), avg_lwc)

plt.figure(figsize=(8, 6))
img = plt.pcolormesh(xedges, yedges, masked_avg_lwc.T, cmap="viridis", shading='auto', norm=norm)

gray_mask = np.isnan(avg_lwc)
gray_values = np.full_like(avg_lwc, np.nan)
gray_values[gray_mask] = 1
plt.pcolormesh(xedges, yedges, gray_values.T, cmap=ListedColormap(["gray"]), shading='auto', alpha=0.6)

cbar = plt.colorbar(img)
cbar.set_label("Mean LWC (g m$^{-3}$)", fontsize=18, fontweight='bold') 
cbar.ax.tick_params(labelsize=18, width=2, length=5)
for t in cbar.ax.get_yticklabels():  
    t.set_fontweight('bold')

plt.xscale('log')
plt.yscale('log')
plt.ylim(0.1, 0.3)
plt.xlim(50, 200)
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.xlabel('Nr+Nc /cm³', fontsize=19, fontweight='bold')
plt.ylabel('LWC g/m³', fontsize=19, fontweight='bold')
plt.title('CAS Mean LWC\nCloudy conditions January–June 2022', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
#fixed LWC and N region
from matplotlib.colors import PowerNorm, ListedColormap
x_bin_centers = (xedges[:-1] + xedges[1:]) / 2
y_bin_centers = (yedges[:-1] + yedges[1:]) / 2
X, Y = np.meshgrid(x_bin_centers, y_bin_centers, indexing='ij')  # shape matches avg_rwc
box_mask = (X >= 50) & (X <= 200) & (Y >= 0.1) & (Y <= 0.3)
avg_rwc_masked = np.where(box_mask, avg_rwc, np.nan)
avg_lwc_masked = np.where(box_mask, avg_lwc, np.nan)
vmin_rwc = np.nanmin(avg_rwc_masked)
vmax_rwc = np.nanmax(avg_rwc_masked)
vmin_lwc = np.nanmin(avg_lwc_masked)
vmax_lwc = np.nanmax(avg_lwc_masked)

norm_rwc = PowerNorm(gamma=0.5, vmin=vmin_rwc, vmax=vmax_rwc)
norm_lwc = PowerNorm(gamma=0.5, vmin=vmin_lwc, vmax=vmax_lwc)
masked_avg_rwc = np.ma.masked_invalid(avg_rwc_masked)
gray_mask = np.isnan(avg_rwc_masked)
gray_values = np.full_like(avg_rwc_masked, np.nan)
gray_values[gray_mask] = 1

plt.figure(figsize=(8, 6))
img = plt.pcolormesh(xedges, yedges, masked_avg_rwc.T, cmap="viridis", shading='auto', norm=norm_rwc)
plt.pcolormesh(xedges, yedges, gray_values.T, cmap=ListedColormap(["gray"]), shading='auto', alpha=0.6)
cbar = plt.colorbar(img)
cbar.set_label("Mean RWC (g m$^{-3}$)", fontsize=18, fontweight='bold') 
cbar.ax.tick_params(labelsize=18, width=2, length=5)
for t in cbar.ax.get_yticklabels(): t.set_fontweight('bold')

plt.xscale('log')
plt.yscale('log')
plt.xlim(50, 200)
plt.ylim(0.1, 0.3)
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.xlabel('Nr+Nc /cm³', fontsize=19, fontweight='bold')
plt.ylabel('LWC g/m³', fontsize=19, fontweight='bold')
plt.title('CAS Mean RWC\nCloudy conditions January–June 2022\nFixed conditions', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()
masked_avg_lwc = np.ma.masked_invalid(avg_lwc_masked)
gray_mask = np.isnan(avg_lwc_masked)
gray_values = np.full_like(avg_lwc_masked, np.nan)
gray_values[gray_mask] = 1

plt.figure(figsize=(8, 6))
img = plt.pcolormesh(xedges, yedges, masked_avg_lwc.T, cmap="viridis", shading='auto', norm=norm_lwc)
plt.pcolormesh(xedges, yedges, gray_values.T, cmap=ListedColormap(["gray"]), shading='auto', alpha=0.6)
cbar = plt.colorbar(img)
cbar.set_label("Mean LWC (g m$^{-3}$)", fontsize=18, fontweight='bold') 
cbar.ax.tick_params(labelsize=18, width=2, length=5)
for t in cbar.ax.get_yticklabels(): t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.xlim(50, 200)
plt.ylim(0.1, 0.3)
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.xlabel('Nr+Nc /cm³', fontsize=19, fontweight='bold')
plt.ylabel('LWC g/m³', fontsize=19, fontweight='bold')
plt.title('CAS Mean LWC\nCloudy conditions January–June 2022\nFixed conditions', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
from matplotlib.colors import PowerNorm, ListedColormap
x_min, x_max = 50, 200       # in cm^-3
y_min, y_max = 0.1, 0.3      # in g/m^3
x_centers = (xedges[:-1] + xedges[1:]) / 2
y_centers = (yedges[:-1] + yedges[1:]) / 2
x_mask = (x_centers >= x_min) & (x_centers <= x_max)
y_mask = (y_centers >= y_min) & (y_centers <= y_max)
region_mask = np.outer(x_mask, y_mask).T 
masked_avg_rwc = np.where(region_mask, avg_rwc, np.nan)
masked_avg_rwc = np.ma.masked_where(np.isnan(masked_avg_rwc), masked_avg_rwc)
vmin = np.nanmin(masked_avg_rwc)
vmax = np.nanmax(masked_avg_rwc)
norm = PowerNorm(gamma=0.5, vmin=0, vmax=0.45) 
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(xedges, yedges, masked_avg_rwc.T, cmap="viridis", shading='auto', norm=norm)
gray_mask = np.isnan(masked_avg_rwc)
gray_values = np.full_like(masked_avg_rwc, np.nan)
gray_values[gray_mask] = 1
plt.pcolormesh(xedges, yedges, gray_values.T, cmap=ListedColormap(["gray"]), shading='auto', alpha=0.6)
cbar = plt.colorbar(img)
cbar.set_label("Mean RWC (g m$^{-3}$)", fontsize=18, fontweight='bold') 
cbar.ax.tick_params(labelsize=18, width=2, length=5)
for t in cbar.ax.get_yticklabels():  
    t.set_fontweight('bold')
selected_values = masked_avg_rwc.compressed()
print("Mean RWC in selected box:", np.nanmean(selected_values))
print("Max RWC in selected box:", np.nanmax(selected_values))
print("Min RWC in selected box:", np.nanmin(selected_values))
print("Number of bins in selection:", len(selected_values))
plt.xscale('log')
plt.yscale('log')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.xlabel('Nr+Nc /cm³', fontsize=19, fontweight='bold')
plt.ylabel('LWC g/m³', fontsize=19, fontweight='bold')
plt.title('CAS Mean RWC\nCloudy conditions January–June 2022\nFixed conditions', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
# #reducing speckling

# concentration = np.array([entry['Total_Combined_Concentration'] for entry in total_combined_concentration])
# total_liquid_water_values = np.array([entry['Total_Liquid_Water'] for entry in total_liquid_water])
# rain_water_content_values = np.array([entry['RWC'] for entry in total_liquid_water])

# num_bins = 5  # Adjusted to give approximately 4x4 bins
# x_bins = np.logspace(np.log10(1), np.log10(max(concentration)), num_bins)
# y_bins = np.logspace(np.log10(min(total_liquid_water_values)), np.log10(max(total_liquid_water_values)), num_bins)

# sum_rwc, xedges, yedges = np.histogram2d(concentration, total_liquid_water_values, bins=[x_bins, y_bins], weights=rain_water_content_values)
# sum_lwc, _, _ = np.histogram2d(concentration, total_liquid_water_values, bins=[x_bins, y_bins], weights=total_liquid_water_values)
# counts, _, _ = np.histogram2d(concentration, total_liquid_water_values, bins=[x_bins, y_bins])

# avg_rwc = np.divide(sum_rwc, counts, out=np.full_like(sum_rwc, np.nan), where=counts > 0)
# avg_lwc = np.divide(sum_lwc, counts, out=np.full_like(sum_lwc, np.nan), where=counts > 0)
# rwc_lwc_ratio = np.divide(avg_rwc, avg_lwc, out=np.full_like(avg_rwc, np.nan), where=avg_lwc > 0) * 100
# masked_rwc_lwc_ratio = np.ma.masked_where(np.isnan(rwc_lwc_ratio), rwc_lwc_ratio)

# plt.figure(figsize=(8, 6))
# norm = mcolors.Normalize(vmin=1, vmax=100)
# img = plt.pcolormesh(xedges, yedges, masked_rwc_lwc_ratio.T, cmap="RdBu_r", norm=norm, shading='auto')

# gray_mask = np.isnan(rwc_lwc_ratio)
# gray_values = np.full_like(rwc_lwc_ratio, np.nan)
# gray_values[gray_mask] = 1
# plt.pcolormesh(xedges, yedges, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)

# cbar = plt.colorbar(img)
# cbar.set_label("RWC/LWC (%)", fontsize=18, fontweight='bold') 
# cbar.ax.tick_params(labelsize=18, width=2, length=5)
# for t in cbar.ax.get_yticklabels():
#     t.set_fontweight('bold')

# plt.xscale('log')
# plt.yscale('log')
# plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
# plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
# plt.xlabel('Nr+Nc /cm³', fontsize=19, fontweight='bold')
# plt.ylabel('LWC g/m³', fontsize=19, fontweight='bold')
# plt.title('CAS (in cloud)\n January-June 2022\n RWC as a function of number concentration', fontsize=18, fontweight='bold')
# plt.tight_layout()
# plt.show()
#%%
# #trying a color blind friendly gradient 
# from matplotlib.colors import LinearSegmentedColormap
# concentration = np.array([entry['Total_Combined_Concentration'] for entry in total_combined_concentration])
# total_liquid_water_values = np.array([entry['Total_Liquid_Water'] for entry in total_liquid_water])
# rain_water_content_values = np.array([entry['RWC'] for entry in total_liquid_water])

# num_bins = 5
# x_bins = np.logspace(np.log10(1), np.log10(max(concentration)), num_bins)
# y_bins = np.logspace(np.log10(min(total_liquid_water_values)), np.log10(max(total_liquid_water_values)), num_bins)

# sum_rwc, xedges, yedges = np.histogram2d(concentration, total_liquid_water_values, bins=[x_bins, y_bins], weights=rain_water_content_values)
# sum_lwc, _, _ = np.histogram2d(concentration, total_liquid_water_values, bins=[x_bins, y_bins], weights=total_liquid_water_values)
# counts, _, _ = np.histogram2d(concentration, total_liquid_water_values, bins=[x_bins, y_bins])

# avg_rwc = np.divide(sum_rwc, counts, out=np.full_like(sum_rwc, np.nan), where=counts > 0)
# avg_lwc = np.divide(sum_lwc, counts, out=np.full_like(sum_lwc, np.nan), where=counts > 0)
# rwc_lwc_ratio = np.divide(avg_rwc, avg_lwc, out=np.full_like(avg_rwc, np.nan), where=avg_lwc > 0) * 100
# masked_rwc_lwc_ratio = np.ma.masked_where(np.isnan(rwc_lwc_ratio), rwc_lwc_ratio)
# custom_cmap = LinearSegmentedColormap.from_list(
#     "green_purple", ["#1b9e77", "white", "#984ea3"], N=256
# )
# plt.figure(figsize=(8, 6))
# norm = mcolors.Normalize(vmin=1, vmax=100)
# img = plt.pcolormesh(xedges, yedges, masked_rwc_lwc_ratio.T, cmap=custom_cmap, norm=norm, shading='auto')
# gray_mask = np.isnan(rwc_lwc_ratio)
# gray_values = np.full_like(rwc_lwc_ratio, np.nan)
# gray_values[gray_mask] = 1
# plt.pcolormesh(xedges, yedges, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)
# cbar = plt.colorbar(img)
# cbar.set_label("RWC/LWC (%)", fontsize=20, fontweight='bold') 
# cbar.ax.tick_params(labelsize=18, width=2, length=5)
# for t in cbar.ax.get_yticklabels():
#     t.set_fontweight('bold')
# plt.xscale('log')
# plt.yscale('log')
# plt.tick_params(axis='both', which='major', labelsize=20, width=3, length=8)
# plt.tick_params(axis='both', which='minor', labelsize=20, width=2, length=5)
# plt.xlabel('Nr+Nc /cm³', fontsize=20, fontweight='bold')
# plt.ylabel('LWC g/m³', fontsize=20, fontweight='bold')
# plt.title('CAS (in cloud)\n January-June 2022\n RWC as a function of number concentration and LWC', fontsize=16, fontweight='bold')
# plt.tight_layout()
# plt.show()

#%%
#density of observations from 0.1 to 0.3 LWC and 50 to 200 /cm3 

num_bins = 15
x_bins = np.logspace(np.log10(1), np.log10(max(concentration)), num_bins)
y_bins = np.logspace(np.log10(min(total_liquid_water_values)), np.log10(max(total_liquid_water_values)), num_bins)
x_min, x_max = 50, 200 
y_min, y_max = 0.1, 0.3 
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
box_x_min, box_x_max = 50, 200 
box_y_min, box_y_max = 0.1, 0.3 
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
box_x_min, box_x_max = 50, 200
box_y_min, box_y_max = 0.1, 0.3
plt.plot(
    [box_x_min, box_x_max, box_x_max, box_x_min, box_x_min],
    [box_y_min, box_y_min, box_y_max, box_y_max, box_y_min],
    color='black',
    linewidth=3
)

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
vmin = min(masked_avg_rwc_high.min(), masked_avg_rwc_low.min())
vmax = max(masked_avg_rwc_high.max(), masked_avg_rwc_low.max())
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
num_bins = 5
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
img = plt.pcolormesh(xedges, yedges, masked_rwc_high.T, cmap="cividis", norm=norm, shading='auto')
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
img = plt.pcolormesh(xedges, yedges, masked_rwc_low.T, cmap="cividis", norm=norm, shading='auto')
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
#trying to separate RWC and LWC

vmin = np.nanmin([masked_avg_rwc_high.min(), masked_avg_lwc_high.min()])
vmax = np.nanmax([masked_avg_rwc_high.max(), masked_avg_lwc_high.max()])
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap_rwc = plt.cm.viridis.copy()
cmap_rwc.set_bad(color='gray')

cmap_lwc = plt.cm.plasma.copy()
cmap_lwc.set_bad(color='gray')
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(
    xedges, yedges,
    masked_avg_rwc_high.T,
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
plt.title('CAS (in cloud)\nHigh GCCN Flights\nJanuary-June 2022', fontsize=19, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(
    xedges, yedges,
    masked_avg_lwc_high.T,
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
plt.title('High GCCN Flights — Mean LWC', fontsize=19, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.tight_layout()
plt.show()

#%%
#low gccn rwc and lwc 
vmin = np.nanmin([masked_avg_rwc_low.min(), masked_avg_lwc_low.min()])
vmax = np.nanmax([masked_avg_rwc_low.max(), masked_avg_lwc_low.max()])

norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(
    xedges, yedges,
    masked_avg_rwc_low.T,
    cmap="viridis",
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
plt.title('CAS (in cloud)\nLow GCCN Flights\nJanuary-June 2022', fontsize=19, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(
    xedges, yedges,
    masked_avg_lwc_low.T,
    cmap="plasma",
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

cmap_rwc = plt.cm.viridis.copy()
cmap_rwc.set_bad(color='gray')

plt.figure(figsize=(8, 6))
img = plt.pcolormesh(
    xedges,
    yedges,
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
plt.title('CAS (in cloud)\nLow GCCN Flights\nJanuary-June 2022', fontsize=19, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
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
#trying to fix color scale 
global_min = np.nanmin([rwc_ratio.min(), lwc_ratio.min()])
global_max = np.nanmax([rwc_ratio.max(), lwc_ratio.max()])
print(f"Shared ratio min: {global_min:.3f}, max: {global_max:.3f}")
shared_norm = mcolors.Normalize(vmin=global_min, vmax=global_max)
plt.figure(figsize=(8,6))
img = plt.pcolormesh(
    xedges,
    yedges,
    rwc_ratio.T,
    cmap="RdBu_r",
    norm=shared_norm,
    shading="auto"
)
cbar = plt.colorbar(img)
cbar.set_label("RWC Ratio (High/Low)", fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=19)
plt.xscale("log")
plt.yscale("log")
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.title("RWC Ratio High/Low GCCN Flights", fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
plt.figure(figsize=(8,6))
img = plt.pcolormesh(
    xedges,
    yedges,
    lwc_ratio.T,
    cmap="RdBu_r",
    norm=shared_norm,
    shading="auto"
)
cbar = plt.colorbar(img)
cbar.set_label("LWC Ratio (High/Low)", fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=19)
plt.xscale("log")
plt.yscale("log")
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.title("LWC Ratio High/Low GCCN Flights", fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()



#%%
masked_avg_rwc_low = np.ma.masked_where(np.isnan(avg_rwc_low), avg_rwc_low)
masked_avg_lwc_low = np.ma.masked_where(np.isnan(avg_lwc_low), avg_lwc_low)
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(xedges, yedges, masked_avg_rwc_low.T, cmap="viridis", shading='auto')
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
img = plt.pcolormesh(xedges, yedges, masked_avg_lwc_low.T, cmap="plasma", shading='auto')
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
#bootstrapping

filtered_high_ratio = filtered_high_rwc / filtered_high_lwc
filtered_low_ratio = filtered_low_rwc / filtered_low_lwc
n_bootstrap = 10000
confidence_level = 0.90
lower_percentile = (1 - confidence_level) / 2 * 100
upper_percentile = (1 + confidence_level) / 2 * 100

significance_mask = np.full((len(x_bins)-1, len(y_bins)-1), False)
for i in range(len(x_bins)-1):
    for j in range(len(y_bins)-1):
        bin_high = (filtered_high_concentration >= x_bins[i]) & (filtered_high_concentration < x_bins[i+1]) & \
                   (filtered_high_lwc >= y_bins[j]) & (filtered_high_lwc < y_bins[j+1])
        bin_low = (filtered_low_concentration >= x_bins[i]) & (filtered_low_concentration < x_bins[i+1]) & \
                  (filtered_low_lwc >= y_bins[j]) & (filtered_low_lwc < y_bins[j+1])

        high_ratios_in_bin = filtered_high_ratio[bin_high]
        low_ratios_in_bin = filtered_low_ratio[bin_low]

        if len(high_ratios_in_bin) >= 3 and len(low_ratios_in_bin) >= 3:
            boot_diff = []
            for _ in range(n_bootstrap):
                sample_high = np.random.choice(high_ratios_in_bin, size=len(high_ratios_in_bin), replace=True)
                sample_low = np.random.choice(low_ratios_in_bin, size=len(low_ratios_in_bin), replace=True)
                boot_diff.append(np.mean(sample_high) - np.mean(sample_low))
            
            boot_diff = np.array(boot_diff)
            lower = np.percentile(boot_diff, lower_percentile)
            upper = np.percentile(boot_diff, upper_percentile)

            if lower > 0 or upper < 0: 
                significance_mask[i, j] = True

diff_rwc_lwc = rwc_lwc_ratio_high - rwc_lwc_ratio_low
masked_diff = np.ma.masked_where(np.isnan(diff_rwc_lwc), diff_rwc_lwc)
abs_max = max(abs(np.nanmin(diff_rwc_lwc)), abs(np.nanmax(diff_rwc_lwc)))
vmin, vmax = -abs_max, abs_max
divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

plt.figure(figsize=(8, 6))
img = plt.pcolormesh(xedges, yedges, masked_diff.T, cmap="viridis", norm=divnorm, shading='auto')

gray_mask = np.isnan(diff_rwc_lwc)
gray_values = np.full_like(diff_rwc_lwc, np.nan)
gray_values[gray_mask] = 1  
plt.pcolormesh(xedges, yedges, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)

plt.pcolormesh(xedges, yedges, np.ma.masked_where(~significance_mask, significance_mask).T,
               shading='auto', hatch='///', alpha=0.0, edgecolors='black', linewidth=0.0)

cbar = plt.colorbar(img)
cbar.set_label("RWC / LWC Difference (%)", fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=19, width=2, length=5)

plt.xscale('log')
plt.yscale('log')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Difference in RWC/LWC \nbetween High and Low GCCN\nBootstrapping', fontsize=17, fontweight='bold')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
i = 0
j = 1

bin_high = (filtered_high_concentration >= x_bins[i]) & (filtered_high_concentration < x_bins[i+1]) & \
           (filtered_high_lwc >= y_bins[j]) & (filtered_high_lwc < y_bins[j+1])
bin_low = (filtered_low_concentration >= x_bins[i]) & (filtered_low_concentration < x_bins[i+1]) & \
          (filtered_low_lwc >= y_bins[j]) & (filtered_low_lwc < y_bins[j+1])

data_high = filtered_high_rwc[bin_high] / filtered_high_lwc[bin_high]
data_low = filtered_low_rwc[bin_low] / filtered_low_lwc[bin_low]

boot_diff = []
for _ in range(10000):
    sample_high = np.random.choice(data_high, size=len(data_high), replace=True)
    sample_low = np.random.choice(data_low, size=len(data_low), replace=True)
    boot_diff.append(np.mean(sample_high) - np.mean(sample_low))

plt.hist(boot_diff, bins=50, density=True)
plt.axvline(0, color='k', linestyle='--')
plt.title(f'Bootstrap Distribution of (High - Low) RWC/LWC\nfor bin ({i},{j})')
plt.xlabel('Mean Difference')
plt.ylabel('Probability Density')
plt.show()
#%%
# import os
# os.makedirs("bootstrap_histograms", exist_ok=True)

# filtered_high_ratio = filtered_high_rwc / filtered_high_lwc
# filtered_low_ratio = filtered_low_rwc / filtered_low_lwc


# n_bootstrap = 10000
# confidence_level = 0.90
# lower_percentile = (1 - confidence_level) / 2 * 100
# upper_percentile = (1 + confidence_level) / 2 * 100

# bin_i_list = []
# bin_j_list = []
# mean_diff_list = []
# lower_ci_list = []
# upper_ci_list = []
# significance_list = []

# for i in range(len(x_bins) - 1):
#     for j in range(len(y_bins) - 1):
#         # Select points inside bin
#         bin_high = (filtered_high_concentration >= x_bins[i]) & (filtered_high_concentration < x_bins[i+1]) & \
#                    (filtered_high_lwc >= y_bins[j]) & (filtered_high_lwc < y_bins[j+1])
#         bin_low = (filtered_low_concentration >= x_bins[i]) & (filtered_low_concentration < x_bins[i+1]) & \
#                   (filtered_low_lwc >= y_bins[j]) & (filtered_low_lwc < y_bins[j+1])

#         high_ratios_in_bin = filtered_high_ratio[bin_high]
#         low_ratios_in_bin = filtered_low_ratio[bin_low]

#         if len(high_ratios_in_bin) >= 3 and len(low_ratios_in_bin) >= 3:
#             # Bootstrap resampling
#             boot_diff = []
#             for _ in range(n_bootstrap):
#                 sample_high = np.random.choice(high_ratios_in_bin, size=len(high_ratios_in_bin), replace=True)
#                 sample_low = np.random.choice(low_ratios_in_bin, size=len(low_ratios_in_bin), replace=True)
#                 boot_diff.append(np.mean(sample_high) - np.mean(sample_low))
            
#             boot_diff = np.array(boot_diff)

#             # Calculate stats
#             mean_diff = np.mean(boot_diff)
#             lower = np.percentile(boot_diff, lower_percentile)
#             upper = np.percentile(boot_diff, upper_percentile)
#             significant = (lower > 0) or (upper < 0)

#             bin_i_list.append(i)
#             bin_j_list.append(j)
#             mean_diff_list.append(mean_diff)
#             lower_ci_list.append(lower)
#             upper_ci_list.append(upper)
#             significance_list.append(significant)

            
#             plt.figure(figsize=(6,5))
#             plt.hist(boot_diff, bins=50, density=True)
#             plt.axvline(0, color='k', linestyle='--', label='Zero')
#             plt.title(f'Bootstrap Dist. (High - Low) for bin ({i},{j})')
#             plt.xlabel('Mean Difference (RWC/LWC)')
#             plt.ylabel('Probability Density')
#             if significant:
#                 plt.text(0.05, 0.9, 'Significant!', transform=plt.gca().transAxes, fontsize=12, color='red')
#             plt.legend()
#             plt.tight_layout()
#             plt.savefig(f"bootstrap_histograms/bootstrap_bin_{i}_{j}.png")
#             plt.close()

#         else:
#             bin_i_list.append(i)
#             bin_j_list.append(j)
#             mean_diff_list.append(np.nan)
#             lower_ci_list.append(np.nan)
#             upper_ci_list.append(np.nan)
#             significance_list.append(False)


# summary_table = pd.DataFrame({
#     'Bin_i': bin_i_list,
#     'Bin_j': bin_j_list,
#     'Mean_Difference': mean_diff_list,
#     '5th_Percentile': lower_ci_list,
#     '95th_Percentile': upper_ci_list,
#     'Significant': significance_list
# })

#%%
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(xedges, yedges, masked_diff.T, cmap="viridis", norm=divnorm, shading='auto')

gray_mask = np.isnan(diff_rwc_lwc)
gray_values = np.full_like(diff_rwc_lwc, np.nan)
gray_values[gray_mask] = 1
plt.pcolormesh(xedges, yedges, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)

plt.pcolormesh(
    xedges, yedges, np.ma.masked_where(~significance_mask, significance_mask).T,
    shading='auto',
    hatch='///',
    facecolors='none',  
    edgecolors='black', linewidth=0.0
)
cbar = plt.colorbar(img)
cbar.set_label("RWC/LWC Difference (%)", fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=19, width=2, length=5)

plt.xscale('log')
plt.yscale('log')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=16, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=16, width=2, length=5)

plt.title('Difference in RWC/LWC\nHigh vs Low GCCN\nBootstrapping', fontsize=17, fontweight='bold')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')

plt.tight_layout()
plt.show()
#%%
# plt.figure(figsize=(8, 6))
# img = plt.pcolormesh(xedges, yedges, masked_diff.T, cmap="viridis", norm=divnorm, shading='auto')
# gray_mask = np.isnan(diff_rwc_lwc)
# gray_values = np.full_like(diff_rwc_lwc, np.nan)
# gray_values[gray_mask] = 1  
# plt.pcolormesh(xedges, yedges, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)

# hatch_layer = plt.pcolormesh(
#     xedges, yedges,
#     np.ma.masked_where(~significance_mask, significance_mask).T,
#     shading='flat',
#     hatch='///',
#     facecolors='none',  # <-- important to make hatch-only
#     edgecolors='black',  # <-- important to outline hatch
#     linewidth=0.0
# )

# hatch_layer.set_zorder(10)
# cbar = plt.colorbar(img)
# cbar.set_label("RWC/LWC Difference (%)", fontsize=18, fontweight='bold')
# cbar.ax.tick_params(labelsize=19, width=2, length=5)
# plt.xscale('log')
# plt.yscale('log')
# plt.xticks(fontsize=19, fontweight='bold')
# plt.yticks(fontsize=19, fontweight='bold')
# plt.tick_params(axis='both', which='major', labelsize=16, width=3, length=8)
# plt.tick_params(axis='both', which='minor', labelsize=16, width=2, length=5)
# plt.title('Difference in RWC/LWC\nHigh vs Low GCCN\nBelow Cloud Base January-June 2022', fontsize=17, fontweight='bold')
# plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
# plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=16, fontweight='bold')

# plt.tight_layout()
# plt.show()
#%%

# bin_i_list = []
# bin_j_list = []
# high_count_list = []
# low_count_list = []
# for i in range(len(x_bins) - 1):
#     for j in range(len(y_bins) - 1):
#         bin_high = (filtered_high_concentration >= x_bins[i]) & (filtered_high_concentration < x_bins[i+1]) & \
#                    (filtered_high_lwc >= y_bins[j]) & (filtered_high_lwc < y_bins[j+1])
#         bin_low = (filtered_low_concentration >= x_bins[i]) & (filtered_low_concentration < x_bins[i+1]) & \
#                   (filtered_low_lwc >= y_bins[j]) & (filtered_low_lwc < y_bins[j+1])

#         high_count = np.sum(bin_high)
#         low_count = np.sum(bin_low)

       
#         bin_i_list.append(i)
#         bin_j_list.append(j)
#         high_count_list.append(high_count)
#         low_count_list.append(low_count)
# bootstrap_counts = pd.DataFrame({
#     'Bin_i': bin_i_list,
#     'Bin_j': bin_j_list,
#     'High_GCCN_Count': high_count_list,
#     'Low_GCCN_Count': low_count_list
# })
# print(bootstrap_counts)
#%%

# sem_diff_matrix = np.full((len(x_bins) - 1, len(y_bins) - 1), np.nan)

# for i in range(len(x_bins) - 1):
#     for j in range(len(y_bins) - 1):
#         bin_high = (filtered_high_concentration >= x_bins[i]) & (filtered_high_concentration < x_bins[i+1]) & \
#                    (filtered_high_lwc >= y_bins[j]) & (filtered_high_lwc < y_bins[j+1])
#         bin_low = (filtered_low_concentration >= x_bins[i]) & (filtered_low_concentration < x_bins[i+1]) & \
#                   (filtered_low_lwc >= y_bins[j]) & (filtered_low_lwc < y_bins[j+1])

#         high_vals = filtered_high_rwc[bin_high] / filtered_high_lwc[bin_high]
#         low_vals = filtered_low_rwc[bin_low] / filtered_low_lwc[bin_low]

#         if len(high_vals) >= 3 and len(low_vals) >= 3:
#             sem_high = np.std(high_vals, ddof=1) / np.sqrt(len(high_vals))
#             sem_low = np.std(low_vals, ddof=1) / np.sqrt(len(low_vals))
#             sem_diff = np.sqrt(sem_high**2 + sem_low**2)
#             sem_diff_matrix[i, j] = sem_diff

# plt.figure(figsize=(8, 6))
# img = plt.pcolormesh(xedges, yedges, masked_diff.T, cmap="RdBu_r", norm=divnorm, shading='auto')
# gray_mask = np.isnan(diff_rwc_lwc)
# gray_values = np.full_like(diff_rwc_lwc, np.nan)
# gray_values[gray_mask] = 1
# plt.pcolormesh(xedges, yedges, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)
# hatch_layer = plt.pcolormesh(
#     xedges, yedges,
#     np.ma.masked_where(~significance_mask, significance_mask).T,
#     shading='flat',
#     hatch='///',
#     facecolors='none',
#     edgecolors='black', linewidth=0.0
# )
# hatch_layer.set_zorder(10)
# for i in range(len(x_bins) - 1):
#     for j in range(len(y_bins) - 1):
#         sem = sem_diff_matrix[i, j]
#         if not np.isnan(sem):
#             x_center = (xedges[i] + xedges[i+1]) / 2
#             y_center = (yedges[j] + yedges[j+1]) / 2
#             plt.text(x_center, y_center, f"±{sem:.2f}%", ha='center', va='center',
#                      fontsize=11, color='black')

# cbar = plt.colorbar(img)
# cbar.set_label("RWC/LWC Difference (%)", fontsize=18, fontweight='bold')
# cbar.ax.tick_params(labelsize=19, width=2, length=5)

# plt.xscale('log')
# plt.yscale('log')
# plt.xticks(fontsize=19, fontweight='bold')
# plt.yticks(fontsize=19, fontweight='bold')
# plt.tick_params(axis='both', which='major', labelsize=16, width=3, length=8)
# plt.tick_params(axis='both', which='minor', labelsize=16, width=2, length=5)

# plt.title('Difference in RWC/LWC\nHigh vs Low GCCN\n(Bootstrap Test with SEM)', fontsize=17, fontweight='bold')
# plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
# plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')

# plt.tight_layout()
# plt.show()
#%%
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(xedges, yedges, masked_diff.T, cmap="RdBu_r", norm=divnorm, shading='auto')
gray_mask = np.isnan(diff_rwc_lwc)
gray_values = np.full_like(diff_rwc_lwc, np.nan)
gray_values[gray_mask] = 1
plt.pcolormesh(xedges, yedges, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)
plt.pcolormesh(
    xedges, yedges,
    np.ma.masked_where(~significance_mask, significance_mask).T,
    shading='flat',
    hatch='///',
    facecolors='none',
    edgecolors='black', linewidth=0.0
).set_zorder(10)

for i in range(len(x_bins) - 1):
    for j in range(len(y_bins) - 1):
        diff = diff_rwc_lwc[i, j]
        sem = sem_diff_matrix[i, j]
        if not np.isnan(diff) and not np.isnan(sem):
            x_center = (xedges[i] + xedges[i+1]) / 2
            y_center = (yedges[j] + yedges[j+1]) / 2
            plt.text(
                x_center, y_center,
                f"{diff:+.2f}%\n±{sem:.2f}%",
                ha='center', va='center',
                fontsize=11, color='black', fontweight='bold'
            )

cbar = plt.colorbar(img)
cbar.set_label("RWC/LWC Difference (%)", fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=19, width=2, length=5)

plt.xscale('log')
plt.yscale('log')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=16, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=16, width=2, length=5)

plt.title('Difference in RWC/LWC\nBelow Cloud Base January-June 2022', fontsize=16, fontweight='bold')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')

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
x_min, x_max = 50, 200  # Nr+Nc range
y_min, y_max = 0.1, 0.3  # LWC range
high_mask = (high_concentration >= x_min) & (high_concentration <= x_max) & \
            (high_lwc >= y_min) & (high_lwc <= y_max)
filtered_high_concentration = high_concentration[high_mask]
filtered_high_lwc = high_lwc[high_mask]

low_mask = (low_concentration >= x_min) & (low_concentration <= x_max) & \
           (low_lwc >= y_min) & (low_lwc <= y_max)
filtered_low_concentration = low_concentration[low_mask]
filtered_low_lwc = low_lwc[low_mask]
num_bins = 3  # Adjust if needed
x_bins = np.logspace(np.log10(x_min), np.log10(x_max), num_bins)
y_bins = np.logspace(np.log10(y_min), np.log10(y_max), num_bins)
counts_high, xedges, yedges = np.histogram2d(filtered_high_concentration, filtered_high_lwc, bins=[x_bins, y_bins])
counts_low, _, _ = np.histogram2d(filtered_low_concentration, filtered_low_lwc, bins=[x_bins, y_bins])
x_bin_centers = (xedges[:-1] + xedges[1:]) / 2
y_bin_centers = (yedges[:-1] + yedges[1:]) / 2
df_high = pd.DataFrame(counts_high, index=y_bin_centers, columns=x_bin_centers)
df_high.index.name = 'LWC (g/m³)'
df_high.columns.name = 'Nr+Nc (cm³)'
df_low = pd.DataFrame(counts_low, index=y_bin_centers, columns=x_bin_centers)
df_low.index.name = 'LWC (g/m³)'
df_low.columns.name = 'Nr+Nc (cm³)'
print("\nHigh GCCN Bin Counts:")
print(df_high)
print("\nLow GCCN Bin Counts:")
print(df_low)

#%%
# Define bins (ensuring they match the previous analysis)
num_bins = 3
x_bins = np.logspace(np.log10(x_min), np.log10(x_max), num_bins)
y_bins = np.logspace(np.log10(y_min), np.log10(y_max), num_bins)
sum_rwc_high, xedges, yedges = np.histogram2d(filtered_high_concentration, filtered_high_lwc, bins=[x_bins, y_bins], weights=filtered_high_rwc)
sum_lwc_high, _, _ = np.histogram2d(filtered_high_concentration, filtered_high_lwc, bins=[x_bins, y_bins], weights=filtered_high_lwc)
counts_high, _, _ = np.histogram2d(filtered_high_concentration, filtered_high_lwc, bins=[x_bins, y_bins])
std_rwc_high, _, _ = np.histogram2d(filtered_high_concentration, filtered_high_lwc, bins=[x_bins, y_bins], weights=(filtered_high_rwc / filtered_high_lwc) ** 2)
std_rwc_high = np.sqrt(std_rwc_high / counts_high - (sum_rwc_high / sum_lwc_high) ** 2)  # Standard deviation formula
sum_rwc_low, _, _ = np.histogram2d(filtered_low_concentration, filtered_low_lwc, bins=[x_bins, y_bins], weights=filtered_low_rwc)
sum_lwc_low, _, _ = np.histogram2d(filtered_low_concentration, filtered_low_lwc, bins=[x_bins, y_bins], weights=filtered_low_lwc)
counts_low, _, _ = np.histogram2d(filtered_low_concentration, filtered_low_lwc, bins=[x_bins, y_bins])
std_rwc_low, _, _ = np.histogram2d(filtered_low_concentration, filtered_low_lwc, bins=[x_bins, y_bins], weights=(filtered_low_rwc / filtered_low_lwc) ** 2)
std_rwc_low = np.sqrt(std_rwc_low / counts_low - (sum_rwc_low / sum_lwc_low) ** 2)
avg_rwc_high = np.divide(sum_rwc_high, counts_high, out=np.full_like(sum_rwc_high, np.nan), where=counts_high > 0)
avg_lwc_high = np.divide(sum_lwc_high, counts_high, out=np.full_like(sum_lwc_high, np.nan), where=counts_high > 0)
rwc_lwc_ratio_high = np.divide(avg_rwc_high, avg_lwc_high, out=np.full_like(avg_rwc_high, np.nan), where=avg_lwc_high > 0) * 100

avg_rwc_low = np.divide(sum_rwc_low, counts_low, out=np.full_like(sum_rwc_low, np.nan), where=counts_low > 0)
avg_lwc_low = np.divide(sum_lwc_low, counts_low, out=np.full_like(sum_rwc_low, np.nan), where=counts_low > 0)
rwc_lwc_ratio_low = np.divide(avg_rwc_low, avg_lwc_low, out=np.full_like(avg_rwc_low, np.nan), where=avg_lwc_low > 0) * 100
#%%
#Difference in each bin

diff_rwc_lwc = avg_rwc_high - avg_rwc_low

masked_diff = np.ma.masked_where(np.isnan(diff_rwc_lwc), diff_rwc_lwc)

plt.figure(figsize=(8, 6))
cmap = "viridis"  # Red = Increase in High GCCN, Blue = Decrease
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

# Compute the difference in each bin with new color bar
diff_rwc_lwc = avg_rwc_high - avg_rwc_low
masked_diff = np.ma.masked_where(np.isnan(diff_rwc_lwc), diff_rwc_lwc)
abs_max = max(abs(np.nanmin(diff_rwc_lwc)), abs(np.nanmax(diff_rwc_lwc)))
vmin, vmax = -abs_max, abs_max  # Ensure a balanced color scale
divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
cmap = plt.cm.RdBu_r  # Red-Blue reversed colormap
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(x_bins, y_bins, masked_diff.T, cmap=cmap, norm=divnorm, shading='auto')
gray_mask = np.isnan(diff_rwc_lwc)
gray_values = np.full_like(diff_rwc_lwc, np.nan)
gray_values[gray_mask] = 1  
plt.pcolormesh(x_bins, y_bins, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)
cbar = plt.colorbar(img)
cbar.set_label("RWC/LWC Difference", fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=19, width=2, length=5)
plt.xscale('log')
plt.yscale('log')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=16, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=16, width=2, length=5)
plt.title('Difference in RWC/LWC \nbetween High and Low GCCN', fontsize=17, fontweight='bold')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.show()
#%%
#trying a new color scale 
diff_rwc_lwc = avg_rwc_high - avg_rwc_low
masked_diff = np.ma.masked_where(np.isnan(diff_rwc_lwc), diff_rwc_lwc)
abs_max = max(abs(np.nanmin(diff_rwc_lwc)), abs(np.nanmax(diff_rwc_lwc)))
vmin, vmax = -abs_max, abs_max
divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
cmap = "PRGn"
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(x_bins, y_bins, masked_diff.T, cmap=cmap, norm=divnorm, shading='auto')
gray_mask = np.isnan(diff_rwc_lwc)
gray_values = np.full_like(diff_rwc_lwc, np.nan)
gray_values[gray_mask] = 1  
plt.pcolormesh(x_bins, y_bins, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)
cbar = plt.colorbar(img)
cbar.set_label("RWC/LWC Difference", fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=19, width=2, length=5)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight('bold')
plt.xscale('log')
plt.yscale('log')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.title('Difference in RWC/LWC \nbetween High and Low GCCN', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
#separating rwc and lwc 
# Compute RWC difference
diff_rwc = avg_rwc_high - avg_rwc_low
masked_diff_rwc = np.ma.masked_where(np.isnan(diff_rwc), diff_rwc)
abs_max_rwc = max(abs(np.nanmin(diff_rwc)), abs(np.nanmax(diff_rwc)))
divnorm_rwc = mcolors.TwoSlopeNorm(vmin=-abs_max_rwc, vcenter=0, vmax=abs_max_rwc)
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(xedges, yedges, masked_diff_rwc.T, cmap='RdBu_r', norm=divnorm_rwc, shading='auto')
gray_mask = np.isnan(diff_rwc)
gray_values = np.full_like(diff_rwc, np.nan)
gray_values[gray_mask] = 1
plt.pcolormesh(xedges, yedges, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)
cbar = plt.colorbar(img)
cbar.set_label("RWC Difference (g m$^{-3}$)", fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=19, width=2, length=5)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.title('Difference in RWC\nHigh – Low GCCN Flights', fontsize=17, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=16, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=16, width=2, length=5)
plt.tight_layout()
plt.show()
#%%
# Compute LWC difference
diff_lwc = avg_lwc_high - avg_lwc_low
masked_diff_lwc = np.ma.masked_where(np.isnan(diff_lwc), diff_lwc)
abs_max_lwc = max(abs(np.nanmin(diff_lwc)), abs(np.nanmax(diff_lwc)))
divnorm_lwc = mcolors.TwoSlopeNorm(vmin=-abs_max_lwc, vcenter=0, vmax=abs_max_lwc)
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(xedges, yedges, masked_diff_lwc.T, cmap='RdBu_r', norm=divnorm_lwc, shading='auto')
gray_mask = np.isnan(diff_lwc)
gray_values = np.full_like(diff_lwc, np.nan)
gray_values[gray_mask] = 1
plt.pcolormesh(xedges, yedges, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)
cbar = plt.colorbar(img)
cbar.set_label("LWC Difference (g m$^{-3}$)", fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=19, width=2, length=5)

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.title('Difference in LWC\nHigh – Low GCCN Flights', fontsize=17, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=16, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=16, width=2, length=5)
plt.tight_layout()
plt.show()


#%%
#mass difference 
diff_rwc_lwc = avg_rwc_high - avg_rwc_low
masked_diff = np.ma.masked_where(np.isnan(diff_rwc_lwc), diff_rwc_lwc)
abs_max = max(abs(np.nanmin(diff_rwc_lwc)), abs(np.nanmax(diff_rwc_lwc)))
vmin, vmax = -abs_max, abs_max  # Ensure a balanced color scale
divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
cmap = plt.cm.RdBu_r  # Red-Blue reversed colormap
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(x_bins, y_bins, masked_diff.T, cmap=cmap, norm=divnorm, shading='auto')
gray_mask = np.isnan(diff_rwc_lwc)
gray_values = np.full_like(diff_rwc_lwc, np.nan)
gray_values[gray_mask] = 1  
plt.pcolormesh(x_bins, y_bins, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)
cbar = plt.colorbar(img)
cbar.set_label("RWC/LWC Difference", fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=19, width=2, length=5)
plt.xscale('log')
plt.yscale('log')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=16, width=3, length=8)
plt.tick_params(axis='both', which='minor', labelsize=16, width=2, length=5)
plt.title('Difference in RWC/LWC \nbetween High and Low GCCN Mass Flights', fontsize=17, fontweight='bold')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.show()
#%%
#mass bootstrapping 


# Compute RWC/LWC ratios
filtered_high_ratio = filtered_high_rwc / filtered_high_lwc
filtered_low_ratio = filtered_low_rwc / filtered_low_lwc

# Set bootstrapping parameters
n_bootstrap = 10000
confidence_level = 0.90
lower_percentile = (1 - confidence_level) / 2 * 100
upper_percentile = (1 + confidence_level) / 2 * 100

# Prepare output lists
bin_i_list = []
bin_j_list = []
mean_diff_list = []
lower_ci_list = []
upper_ci_list = []
std_err_list = []
significance_list = []

# Loop through each bin
for i in range(len(x_bins) - 1):
    for j in range(len(y_bins) - 1):
        # Filter data for this bin
        bin_high = (filtered_high_concentration >= x_bins[i]) & (filtered_high_concentration < x_bins[i+1]) & \
                   (filtered_high_lwc >= y_bins[j]) & (filtered_high_lwc < y_bins[j+1])
        bin_low = (filtered_low_concentration >= x_bins[i]) & (filtered_low_concentration < x_bins[i+1]) & \
                  (filtered_low_lwc >= y_bins[j]) & (filtered_low_lwc < y_bins[j+1])

        high_vals = filtered_high_ratio[bin_high]
        low_vals = filtered_low_ratio[bin_low]

        if len(high_vals) >= 3 and len(low_vals) >= 3:
            # Bootstrap resampling
            boot_diff = []
            for _ in range(n_bootstrap):
                sample_high = np.random.choice(high_vals, size=len(high_vals), replace=True)
                sample_low = np.random.choice(low_vals, size=len(low_vals), replace=True)
                boot_diff.append(np.mean(sample_high) - np.mean(sample_low))

            boot_diff = np.array(boot_diff)
            mean_diff = np.mean(boot_diff) * 100  # convert to %
            lower_ci = np.percentile(boot_diff, lower_percentile) * 100
            upper_ci = np.percentile(boot_diff, upper_percentile) * 100
            std_err = np.std(boot_diff, ddof=1) * 100
            significant = (lower_ci > 0) or (upper_ci < 0)
        else:
            mean_diff = np.nan
            lower_ci = np.nan
            upper_ci = np.nan
            std_err = np.nan
            significant = False

        # Store results
        bin_i_list.append(i)
        bin_j_list.append(j)
        mean_diff_list.append(mean_diff)
        lower_ci_list.append(lower_ci)
        upper_ci_list.append(upper_ci)
        std_err_list.append(std_err)
        significance_list.append(significant)

# Create summary DataFrame
bootstrap_mass_summary = pd.DataFrame({
    'Bin_i': bin_i_list,
    'Bin_j': bin_j_list,
    'Mean_Diff (%)': mean_diff_list,
    'Lower_90CI (%)': lower_ci_list,
    'Upper_90CI (%)': upper_ci_list,
    'Std_Error (%)': std_err_list,
    'Significant': significance_list
})

print(bootstrap_mass_summary)
#%%
# Rebuild 2D arrays for plotting
diff_rwc_lwc = np.full((len(x_bins) - 1, len(y_bins) - 1), np.nan)
sem_diff_matrix = np.full_like(diff_rwc_lwc, np.nan)

# Fill arrays from bootstrap summary
for row in bootstrap_mass_summary.itertuples():
    i, j = row.Bin_i, row.Bin_j
    diff_rwc_lwc[i, j] = row._3  # Mean_Diff (%)
    sem_diff_matrix[i, j] = row._6  # Std_Error (%)

# Mask NaNs for display
masked_diff = np.ma.masked_where(np.isnan(diff_rwc_lwc), diff_rwc_lwc)

# Set up color scale
abs_max = max(abs(np.nanmin(diff_rwc_lwc)), abs(np.nanmax(diff_rwc_lwc)))
vmin, vmax = -abs_max, abs_max
divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

# Plotting
plt.figure(figsize=(8, 6))
img = plt.pcolormesh(xedges, yedges, masked_diff.T, cmap="RdBu_r", norm=divnorm, shading='auto')

# Gray out NaNs
gray_mask = np.isnan(diff_rwc_lwc)
gray_values = np.full_like(diff_rwc_lwc, np.nan)
gray_values[gray_mask] = 1
plt.pcolormesh(xedges, yedges, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)

# Annotate each bin with Diff ± SEM
for i in range(len(x_bins) - 1):
    for j in range(len(y_bins) - 1):
        diff = diff_rwc_lwc[i, j]
        sem = sem_diff_matrix[i, j]
        if not np.isnan(diff) and not np.isnan(sem):
            x_center = (xedges[i] + xedges[i+1]) / 2
            y_center = (yedges[j] + yedges[j+1]) / 2
            plt.text(
                x_center, y_center,
                f"{diff:+.2f}%\n±{sem:.2f}%",
                ha='center', va='center',
                fontsize=11, color='black', fontweight='bold'
            )

# Colorbar
cbar = plt.colorbar(img)
cbar.set_label("RWC/LWC Difference (%)", fontsize=18, fontweight='bold')
cbar.ax.tick_params(labelsize=19, width=2, length=5)

# Axes formatting
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.title('Difference in RWC/LWC\nHigh vs Low GCCN Mass Flights', fontsize=17, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.tick_params(axis='both', which='major', width=3, length=8)
plt.tick_params(axis='both', which='minor', width=2, length=5)

plt.tight_layout()
plt.show()

#%%
#Compute the mean RWC% for high and low GCCN flights in this region.
mean_rwc_high = np.nanmean(rwc_lwc_ratio_high)
mean_rwc_low = np.nanmean(rwc_lwc_ratio_low)
print(f"Mean RWC% for High GCCN Flights: {mean_rwc_high:.2f}%")
print(f"Mean RWC% for Low GCCN Flights: {mean_rwc_low:.2f}%")
#%%
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
#Working with only 1/2 the flights 
import random
random.seed(42)
all_flight_dates = list(average_gccn_per_flight.keys())
total_flights = len(all_flight_dates)
half_count = total_flights // 2
selected_dates = random.sample(all_flight_dates, half_count)
average_gccn_half = {date: average_gccn_per_flight[date] for date in selected_dates}
gccn_values_half = np.array(list(average_gccn_half.values()))
threshold_half = np.percentile(gccn_values_half, 50)

high_GCCN_half = {}

low_GCCN_half = {}

for date, avg_gccn in average_gccn_half.items():
    if avg_gccn >= threshold_half:
        high_GCCN_half[date] = avg_gccn
    else:
        low_GCCN_half[date] = avg_gccn
print(f"Total flights used: {len(average_gccn_half)}")
print(f"Threshold from subsample: {threshold_half:.4f} cm⁻³")
print(f"High GCCN flights: {len(high_GCCN_half)}")
print(f"Low GCCN flights: {len(low_GCCN_half)}")

# %%
# Step 1: Filter the data using half of the flights
high_gccn_data_half = [entry for entry in total_combined_concentration if entry['Date'] in high_GCCN_half]
low_gccn_data_half = [entry for entry in total_combined_concentration if entry['Date'] in low_GCCN_half]
high_lwc_half = np.array([entry['Total_Liquid_Water'] for entry in total_liquid_water if entry['Date'] in high_GCCN_half])
high_rwc_half = np.array([entry['RWC'] for entry in total_liquid_water if entry['Date'] in high_GCCN_half])
low_lwc_half = np.array([entry['Total_Liquid_Water'] for entry in total_liquid_water if entry['Date'] in low_GCCN_half])
low_rwc_half = np.array([entry['RWC'] for entry in total_liquid_water if entry['Date'] in low_GCCN_half])
high_concentration_half = np.array([entry['Total_Combined_Concentration'] for entry in high_gccn_data_half])
low_concentration_half = np.array([entry['Total_Combined_Concentration'] for entry in low_gccn_data_half])
num_bins = 3
x_bins = np.logspace(np.log10(1), np.log10(max(high_concentration_half.tolist() + low_concentration_half.tolist())), num_bins)
y_bins = np.logspace(np.log10(min(high_lwc_half.tolist() + low_lwc_half.tolist())), np.log10(max(high_lwc_half.tolist() + low_lwc_half.tolist())), num_bins)
sum_rwc_high, xedges, yedges = np.histogram2d(high_concentration_half, high_lwc_half, bins=[x_bins, y_bins], weights=high_rwc_half)
sum_lwc_high, _, _ = np.histogram2d(high_concentration_half, high_lwc_half, bins=[x_bins, y_bins], weights=high_lwc_half)
counts_high, _, _ = np.histogram2d(high_concentration_half, high_lwc_half, bins=[x_bins, y_bins])

avg_rwc_high = np.divide(sum_rwc_high, counts_high, out=np.full_like(sum_rwc_high, np.nan), where=counts_high > 0)
avg_lwc_high = np.divide(sum_lwc_high, counts_high, out=np.full_like(sum_lwc_high, np.nan), where=counts_high > 0)
rwc_lwc_ratio_high = np.divide(avg_rwc_high, avg_lwc_high, out=np.full_like(avg_rwc_high, np.nan), where=avg_lwc_high > 0) * 100
masked_rwc_high = np.ma.masked_where(np.isnan(rwc_lwc_ratio_high), rwc_lwc_ratio_high)
sum_rwc_low, _, _ = np.histogram2d(low_concentration_half, low_lwc_half, bins=[x_bins, y_bins], weights=low_rwc_half)
sum_lwc_low, _, _ = np.histogram2d(low_concentration_half, low_lwc_half, bins=[x_bins, y_bins], weights=low_lwc_half)
counts_low, _, _ = np.histogram2d(low_concentration_half, low_lwc_half, bins=[x_bins, y_bins])

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
plt.title('High GCCN Flights (Half Sample)', fontsize=19, fontweight='bold')
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
plt.title('Low GCCN Flights (Half Sample)', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()

# %%
# # Filter to fixed LWC and concentration region for 1/2 flights
# x_min, x_max = 50, 200  # Nr+Nc range
# y_min, y_max = 0.1, 0.3  # LWC range
# high_mask = (high_concentration_half >= x_min) & (high_concentration_half <= x_max) & \
#             (high_lwc_half >= y_min) & (high_lwc_half <= y_max)
# filtered_high_concentration = high_concentration_half[high_mask]
# filtered_high_lwc = high_lwc_half[high_mask]
# filtered_high_rwc = high_rwc_half[high_mask]
# low_mask = (low_concentration_half >= x_min) & (low_concentration_half <= x_max) & \
#            (low_lwc_half >= y_min) & (low_lwc_half <= y_max)
# filtered_low_concentration = low_concentration_half[low_mask]
# filtered_low_lwc = low_lwc_half[low_mask]
# filtered_low_rwc = low_rwc_half[low_mask]
# num_bins = 3
# x_bins = np.logspace(np.log10(x_min), np.log10(x_max), num_bins)
# y_bins = np.logspace(np.log10(y_min), np.log10(y_max), num_bins)
# sum_rwc_high, xedges, yedges = np.histogram2d(filtered_high_concentration, filtered_high_lwc, bins=[x_bins, y_bins], weights=filtered_high_rwc)
# sum_lwc_high, _, _ = np.histogram2d(filtered_high_concentration, filtered_high_lwc, bins=[x_bins, y_bins], weights=filtered_high_lwc)
# counts_high, _, _ = np.histogram2d(filtered_high_concentration, filtered_high_lwc, bins=[x_bins, y_bins])

# avg_rwc_high = np.divide(sum_rwc_high, counts_high, out=np.full_like(sum_rwc_high, np.nan), where=counts_high > 0)
# avg_lwc_high = np.divide(sum_lwc_high, counts_high, out=np.full_like(sum_lwc_high, np.nan), where=counts_high > 0)
# rwc_lwc_ratio_high = np.divide(avg_rwc_high, avg_lwc_high, out=np.full_like(avg_rwc_high, np.nan), where=avg_lwc_high > 0) * 100
# masked_rwc_high = np.ma.masked_where(np.isnan(rwc_lwc_ratio_high), rwc_lwc_ratio_high)

# sum_rwc_low, _, _ = np.histogram2d(filtered_low_concentration, filtered_low_lwc, bins=[x_bins, y_bins], weights=filtered_low_rwc)
# sum_lwc_low, _, _ = np.histogram2d(filtered_low_concentration, filtered_low_lwc, bins=[x_bins, y_bins], weights=filtered_low_lwc)
# counts_low, _, _ = np.histogram2d(filtered_low_concentration, filtered_low_lwc, bins=[x_bins, y_bins])

# avg_rwc_low = np.divide(sum_rwc_low, counts_low, out=np.full_like(sum_rwc_low, np.nan), where=counts_low > 0)
# avg_lwc_low = np.divide(sum_lwc_low, counts_low, out=np.full_like(sum_lwc_low, np.nan), where=counts_low > 0)
# rwc_lwc_ratio_low = np.divide(avg_rwc_low, avg_lwc_low, out=np.full_like(avg_rwc_low, np.nan), where=avg_lwc_low > 0) * 100
# masked_rwc_low = np.ma.masked_where(np.isnan(rwc_lwc_ratio_low), rwc_lwc_ratio_low)
# gray_values_high = np.full_like(rwc_lwc_ratio_high, np.nan)
# gray_values_high[np.isnan(rwc_lwc_ratio_high)] = 1
# gray_values_low = np.full_like(rwc_lwc_ratio_low, np.nan)
# gray_values_low[np.isnan(rwc_lwc_ratio_low)] = 1
# plt.figure(figsize=(8, 6))
# norm = mcolors.Normalize(vmin=1, vmax=100)
# img = plt.pcolormesh(xedges, yedges, masked_rwc_high.T, cmap="RdBu_r", norm=norm, shading='auto')
# plt.pcolormesh(xedges, yedges, gray_values_high.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)
# cbar = plt.colorbar(img)
# cbar.set_label("RWC / LWC (%)", fontsize=18, fontweight='bold')
# cbar.ax.tick_params(labelsize=19, width=2, length=5)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
# plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
# plt.title('High GCCN Flights (Half Sample, Fixed Region)', fontsize=19, fontweight='bold')
# plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
# plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
# plt.tight_layout()
# plt.show()

# # Plot LOW
# plt.figure(figsize=(8, 6))
# img = plt.pcolormesh(xedges, yedges, masked_rwc_low.T, cmap="RdBu_r", norm=norm, shading='auto')
# plt.pcolormesh(xedges, yedges, gray_values_low.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)
# cbar = plt.colorbar(img)
# cbar.set_label("RWC / LWC (%)", fontsize=18, fontweight='bold')
# cbar.ax.tick_params(labelsize=19, width=2, length=5)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
# plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
# plt.title('Low GCCN Flights (Half Sample, Fixed Region)', fontsize=19, fontweight='bold')
# plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
# plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
# plt.tight_layout()
# plt.show()

# # %%
# # Compute the difference in each bin (using half sample data)
# diff_rwc_lwc = avg_rwc_high - avg_rwc_low
# masked_diff = np.ma.masked_where(np.isnan(diff_rwc_lwc), diff_rwc_lwc)
# abs_max = max(abs(np.nanmin(diff_rwc_lwc)), abs(np.nanmax(diff_rwc_lwc)))
# vmin, vmax = -abs_max, abs_max
# divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
# cmap = plt.cm.RdBu_r  
# gray_mask = np.isnan(diff_rwc_lwc)
# gray_values = np.full_like(diff_rwc_lwc, np.nan)
# gray_values[gray_mask] = 1  
# plt.figure(figsize=(8, 6))
# img = plt.pcolormesh(xedges, yedges, masked_diff.T, cmap=cmap, norm=divnorm, shading='auto')
# plt.pcolormesh(xedges, yedges, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)
# cbar = plt.colorbar(img)
# cbar.set_label("RWC/LWC Difference (%)", fontsize=18, fontweight='bold')
# cbar.ax.tick_params(labelsize=19, width=2, length=5)
# plt.xscale('log')
# plt.yscale('log')
# plt.xticks(fontsize=19, fontweight='bold')
# plt.yticks(fontsize=19, fontweight='bold')
# plt.tick_params(axis='both', which='major', labelsize=16, width=3, length=8)
# plt.tick_params(axis='both', which='minor', labelsize=16, width=2, length=5)
# plt.title('Difference in RWC/LWC\nBelow Cloud Base January-June 2022\n Half the total flights', fontsize=17, fontweight='bold')
# plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
# plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
# plt.tight_layout()
# plt.show()

# # %%
# # Calculate RWC/LWC ratios before binning
# filtered_high_ratio = filtered_high_rwc / filtered_high_lwc
# filtered_low_ratio = filtered_low_rwc / filtered_low_lwc

# n_bootstrap = 10000
# confidence_level = 0.90
# lower_percentile = (1 - confidence_level) / 2 * 100
# upper_percentile = (1 + confidence_level) / 2 * 100

# significance_mask = np.full((len(x_bins) - 1, len(y_bins) - 1), False)
# for i in range(len(x_bins) - 1):
#     for j in range(len(y_bins) - 1):
#         bin_high = (filtered_high_concentration >= x_bins[i]) & (filtered_high_concentration < x_bins[i+1]) & \
#                    (filtered_high_lwc >= y_bins[j]) & (filtered_high_lwc < y_bins[j+1])
#         bin_low = (filtered_low_concentration >= x_bins[i]) & (filtered_low_concentration < x_bins[i+1]) & \
#                   (filtered_low_lwc >= y_bins[j]) & (filtered_low_lwc < y_bins[j+1])

#         high_ratios_in_bin = filtered_high_ratio[bin_high]
#         low_ratios_in_bin = filtered_low_ratio[bin_low]

#         if len(high_ratios_in_bin) >= 3 and len(low_ratios_in_bin) >= 3:
#             boot_diff = []
#             for _ in range(n_bootstrap):
#                 sample_high = np.random.choice(high_ratios_in_bin, size=len(high_ratios_in_bin), replace=True)
#                 sample_low = np.random.choice(low_ratios_in_bin, size=len(low_ratios_in_bin), replace=True)
#                 boot_diff.append(np.mean(sample_high) - np.mean(sample_low))
            
#             lower = np.percentile(boot_diff, lower_percentile)
#             upper = np.percentile(boot_diff, upper_percentile)

           
#             if lower > 0 or upper < 0:
#                 significance_mask[i, j] = True
# diff_rwc_lwc = rwc_lwc_ratio_high - rwc_lwc_ratio_low
# masked_diff = np.ma.masked_where(np.isnan(diff_rwc_lwc), diff_rwc_lwc)
# abs_max = max(abs(np.nanmin(diff_rwc_lwc)), abs(np.nanmax(diff_rwc_lwc)))
# divnorm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
# plt.figure(figsize=(8, 6))
# img = plt.pcolormesh(xedges, yedges, masked_diff.T, cmap="RdBu_r", norm=divnorm, shading='auto')
# gray_mask = np.isnan(diff_rwc_lwc)
# gray_values = np.full_like(diff_rwc_lwc, np.nan)
# gray_values[gray_mask] = 1
# plt.pcolormesh(xedges, yedges, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)
# plt.pcolormesh(xedges, yedges, np.ma.masked_where(~significance_mask, significance_mask).T,
#                shading='auto', hatch='///', alpha=0.0, edgecolors='black', linewidth=0.0)
# cbar = plt.colorbar(img)
# cbar.set_label("RWC / LWC Difference (%)", fontsize=18, fontweight='bold')
# cbar.ax.tick_params(labelsize=19, width=2, length=5)
# plt.xscale('log')
# plt.yscale('log')
# plt.xticks(fontsize=19, fontweight='bold')
# plt.yticks(fontsize=19, fontweight='bold')
# plt.title('Difference in RWC/LWC\nHigh vs Low GCCN (Half Sample, Bootstrap Test)', fontsize=17, fontweight='bold')
# plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
# plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
# plt.tight_layout()
# plt.show()

# # %%
# # Example bootstrap histogram for a specific bin (i, j)
# i, j = 0, 1 
# bin_high = (filtered_high_concentration >= x_bins[i]) & (filtered_high_concentration < x_bins[i+1]) & \
#            (filtered_high_lwc >= y_bins[j]) & (filtered_high_lwc < y_bins[j+1])
# bin_low = (filtered_low_concentration >= x_bins[i]) & (filtered_low_concentration < x_bins[i+1]) & \
#           (filtered_low_lwc >= y_bins[j]) & (filtered_low_lwc < y_bins[j+1])

# data_high = filtered_high_rwc[bin_high] / filtered_high_lwc[bin_high]
# data_low = filtered_low_rwc[bin_low] / filtered_low_lwc[bin_low]
# boot_diff = []
# for _ in range(n_bootstrap):
#     sample_high = np.random.choice(data_high, size=len(data_high), replace=True)
#     sample_low = np.random.choice(data_low, size=len(data_low), replace=True)
#     boot_diff.append(np.mean(sample_high) - np.mean(sample_low))
# plt.hist(boot_diff, bins=50, density=True)
# plt.axvline(0, color='k', linestyle='--')
# plt.title(f'Bootstrap Distribution of (High - Low) RWC/LWC\nfor bin ({i},{j})')
# plt.xlabel('Mean Difference')
# plt.ylabel('Probability Density')
# plt.tight_layout()
# plt.show()

# # %%

# os.makedirs("bootstrap_histograms", exist_ok=True)

# filtered_high_ratio = filtered_high_rwc / filtered_high_lwc
# filtered_low_ratio = filtered_low_rwc / filtered_low_lwc

# n_bootstrap = 10000
# confidence_level = 0.90
# lower_percentile = (1 - confidence_level) / 2 * 100
# upper_percentile = (1 + confidence_level) / 2 * 100

# bin_i_list = []
# bin_j_list = []
# mean_diff_list = []
# lower_ci_list = []
# upper_ci_list = []
# significance_list = []
# significance_mask = np.full((len(x_bins) - 1, len(y_bins) - 1), False)

# for i in range(len(x_bins) - 1):
#     for j in range(len(y_bins) - 1):
#         bin_high = (filtered_high_concentration >= x_bins[i]) & (filtered_high_concentration < x_bins[i+1]) & \
#                    (filtered_high_lwc >= y_bins[j]) & (filtered_high_lwc < y_bins[j+1])
#         bin_low = (filtered_low_concentration >= x_bins[i]) & (filtered_low_concentration < x_bins[i+1]) & \
#                   (filtered_low_lwc >= y_bins[j]) & (filtered_low_lwc < y_bins[j+1])

#         high_ratios_in_bin = filtered_high_ratio[bin_high]
#         low_ratios_in_bin = filtered_low_ratio[bin_low]

#         if len(high_ratios_in_bin) >= 3 and len(low_ratios_in_bin) >= 3:
#             boot_diff = []
#             for _ in range(n_bootstrap):
#                 sample_high = np.random.choice(high_ratios_in_bin, size=len(high_ratios_in_bin), replace=True)
#                 sample_low = np.random.choice(low_ratios_in_bin, size=len(low_ratios_in_bin), replace=True)
#                 boot_diff.append(np.mean(sample_high) - np.mean(sample_low))

#             boot_diff = np.array(boot_diff)
#             mean_diff = np.mean(boot_diff)
#             lower = np.percentile(boot_diff, lower_percentile)
#             upper = np.percentile(boot_diff, upper_percentile)
#             significant = (lower > 0) or (upper < 0)
#             if significant:
#                 significance_mask[i, j] = True

#             bin_i_list.append(i)
#             bin_j_list.append(j)
#             mean_diff_list.append(mean_diff)
#             lower_ci_list.append(lower)
#             upper_ci_list.append(upper)
#             significance_list.append(significant)

           
#             plt.figure(figsize=(6, 5))
#             plt.hist(boot_diff, bins=50, density=True)
#             plt.axvline(0, color='k', linestyle='--', label='Zero')
#             plt.title(f'Bootstrap Dist. (High - Low) for bin ({i},{j})')
#             plt.xlabel('Mean Difference (RWC/LWC)')
#             plt.ylabel('Probability Density')
#             if significant:
#                 plt.text(0.05, 0.9, 'Significant!', transform=plt.gca().transAxes, fontsize=12, color='red')
#             plt.legend()
#             plt.tight_layout()
#             plt.savefig(f"bootstrap_histograms/bootstrap_bin_{i}_{j}.png")
#             plt.close()

#         else:
#             bin_i_list.append(i)
#             bin_j_list.append(j)
#             mean_diff_list.append(np.nan)
#             lower_ci_list.append(np.nan)
#             upper_ci_list.append(np.nan)
#             significance_list.append(False)

# summary_table = pd.DataFrame({
#     'Bin_i': bin_i_list,
#     'Bin_j': bin_j_list,
#     'Mean_Difference': mean_diff_list,
#     '5th_Percentile': lower_ci_list,
#     '95th_Percentile': upper_ci_list,
#     'Significant': significance_list
# })

# # %%
# # Plot difference map again with significance hatching
# plt.figure(figsize=(8, 6))
# img = plt.pcolormesh(xedges, yedges, masked_diff.T, cmap="RdBu_r", norm=divnorm, shading='auto')
# gray_mask = np.isnan(diff_rwc_lwc)
# gray_values = np.full_like(diff_rwc_lwc, np.nan)
# gray_values[gray_mask] = 1
# plt.pcolormesh(xedges, yedges, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)
# plt.pcolormesh(
#     xedges, yedges,
#     np.ma.masked_where(~significance_mask, significance_mask).T,
#     shading='auto',
#     hatch='///',
#     facecolors='none',
#     edgecolors='black', linewidth=0.0
# )

# cbar = plt.colorbar(img)
# cbar.set_label("RWC/LWC Difference (%)", fontsize=18, fontweight='bold')
# cbar.ax.tick_params(labelsize=19, width=2, length=5)
# plt.xscale('log')
# plt.yscale('log')
# plt.xticks(fontsize=19, fontweight='bold')
# plt.yticks(fontsize=19, fontweight='bold')
# plt.tick_params(axis='both', which='major', labelsize=16, width=3, length=8)
# plt.tick_params(axis='both', which='minor', labelsize=16, width=2, length=5)
# plt.title('Difference in RWC/LWC\nBelow Cloud Base January-June 2022\n Half the total flights', fontsize=17, fontweight='bold')
# plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
# plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
# plt.tight_layout()
# plt.show()

# # %%
# #count data points in each bin
# bin_i_list = []
# bin_j_list = []
# high_count_list = []
# low_count_list = []

# for i in range(len(x_bins) - 1):
#     for j in range(len(y_bins) - 1):
#         bin_high = (filtered_high_concentration >= x_bins[i]) & (filtered_high_concentration < x_bins[i+1]) & \
#                    (filtered_high_lwc >= y_bins[j]) & (filtered_high_lwc < y_bins[j+1])
#         bin_low = (filtered_low_concentration >= x_bins[i]) & (filtered_low_concentration < x_bins[i+1]) & \
#                   (filtered_low_lwc >= y_bins[j]) & (filtered_low_lwc < y_bins[j+1])

#         high_count = np.sum(bin_high)
#         low_count = np.sum(bin_low)

#         bin_i_list.append(i)
#         bin_j_list.append(j)
#         high_count_list.append(high_count)
#         low_count_list.append(low_count)

# bootstrap_counts = pd.DataFrame({
#     'Bin_i': bin_i_list,
#     'Bin_j': bin_j_list,
#     'High_GCCN_Count': high_count_list,
#     'Low_GCCN_Count': low_count_list
# })
# print(bootstrap_counts)

# # %%
# # --- Compute Standard Error Matrix ---
# sem_diff_matrix = np.full((len(x_bins) - 1, len(y_bins) - 1), np.nan)

# for i in range(len(x_bins) - 1):
#     for j in range(len(y_bins) - 1):
#         bin_high = (filtered_high_concentration >= x_bins[i]) & (filtered_high_concentration < x_bins[i+1]) & \
#                    (filtered_high_lwc >= y_bins[j]) & (filtered_high_lwc < y_bins[j+1])
#         bin_low = (filtered_low_concentration >= x_bins[i]) & (filtered_low_concentration < x_bins[i+1]) & \
#                   (filtered_low_lwc >= y_bins[j]) & (filtered_low_lwc < y_bins[j+1])

#         high_vals = filtered_high_rwc[bin_high] / filtered_high_lwc[bin_high]
#         low_vals = filtered_low_rwc[bin_low] / filtered_low_lwc[bin_low]

#         if len(high_vals) >= 3 and len(low_vals) >= 3:
#             sem_high = np.std(high_vals, ddof=1) / np.sqrt(len(high_vals))
#             sem_low = np.std(low_vals, ddof=1) / np.sqrt(len(low_vals))
#             sem_diff = np.sqrt(sem_high**2 + sem_low**2)
#             sem_diff_matrix[i, j] = sem_diff

# # %%
# plt.figure(figsize=(8, 6))
# img = plt.pcolormesh(xedges, yedges, masked_diff.T, cmap="RdBu_r", norm=divnorm, shading='auto')
# plt.pcolormesh(xedges, yedges, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)
# plt.pcolormesh(xedges, yedges, np.ma.masked_where(~significance_mask, significance_mask).T,
#                shading='flat', hatch='///', facecolors='none', edgecolors='black', linewidth=0.0).set_zorder(10)
# for i in range(len(x_bins) - 1):
#     for j in range(len(y_bins) - 1):
#         diff = diff_rwc_lwc[i, j]
#         sem = sem_diff_matrix[i, j]
#         if not np.isnan(diff) and not np.isnan(sem):
#             x_center = (xedges[i] + xedges[i+1]) / 2
#             y_center = (yedges[j] + yedges[j+1]) / 2
#             plt.text(
#                 x_center, y_center,
#                 f"{diff:+.2f}%\n±{sem:.2f}%",
#                 ha='center', va='center',
#                 fontsize=11, color='black', fontweight='bold'
#             )

# cbar = plt.colorbar(img)
# cbar.set_label("RWC/LWC Difference (%)", fontsize=18, fontweight='bold')
# cbar.ax.tick_params(labelsize=19, width=2, length=5)
# plt.xscale('log')
# plt.yscale('log')
# plt.xticks(fontsize=19, fontweight='bold')
# plt.yticks(fontsize=19, fontweight='bold')
# plt.tick_params(axis='both', which='major', labelsize=16, width=3, length=8)
# plt.tick_params(axis='both', which='minor', labelsize=16, width=2, length=5)
# plt.title('Difference in RWC/LWC\nBelow Cloud Base January-June 2022\n Half the total flights', fontsize=17, fontweight='bold')
# plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
# plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
# plt.tight_layout()
# plt.show()
# #%%
# # #Working with the OTHER 1/2 flights 
# # random.seed(42)
# # all_flight_dates = list(average_gccn_per_flight.keys())
# # total_flights = len(all_flight_dates)
# # half_count = total_flights // 2
# # selected_dates = random.sample(all_flight_dates, half_count)
# # other_half_dates = [date for date in all_flight_dates if date not in selected_dates]
# average_gccn_other_half = {date: average_gccn_per_flight[date] for date in other_half_dates}
# gccn_values_other_half = np.array(list(average_gccn_other_half.values()))
# threshold_other_half = np.percentile(gccn_values_other_half, 50)
# high_GCCN_other = {}
# low_GCCN_other = {}

# for date, avg_gccn in average_gccn_other_half.items():
#     if avg_gccn >= threshold_other_half:
#         high_GCCN_other[date] = avg_gccn
#     else:
#         low_GCCN_other[date] = avg_gccn

# print(f"Total flights used (other half): {len(average_gccn_other_half)}")
# print(f"Threshold from other half: {threshold_other_half:.4f} cm⁻³")
# print(f"High GCCN flights (other half): {len(high_GCCN_other)}")
# print(f"Low GCCN flights (other half): {len(low_GCCN_other)}")
# #%%
# # Step 1: Filter the data using the OTHER half of the flights
# high_gccn_data_other = [entry for entry in total_combined_concentration if entry['Date'] in high_GCCN_other]
# low_gccn_data_other = [entry for entry in total_combined_concentration if entry['Date'] in low_GCCN_other]
# high_lwc_other = np.array([entry['Total_Liquid_Water'] for entry in total_liquid_water if entry['Date'] in high_GCCN_other])
# high_rwc_other = np.array([entry['RWC'] for entry in total_liquid_water if entry['Date'] in high_GCCN_other])
# low_lwc_other = np.array([entry['Total_Liquid_Water'] for entry in total_liquid_water if entry['Date'] in low_GCCN_other])
# low_rwc_other = np.array([entry['RWC'] for entry in total_liquid_water if entry['Date'] in low_GCCN_other])
# high_concentration_other = np.array([entry['Total_Combined_Concentration'] for entry in high_gccn_data_other])
# low_concentration_other = np.array([entry['Total_Combined_Concentration'] for entry in low_gccn_data_other])
# num_bins = 3
# x_bins = np.logspace(np.log10(1), np.log10(max(high_concentration_other.tolist() + low_concentration_other.tolist())), num_bins)
# y_bins = np.logspace(np.log10(min(high_lwc_other.tolist() + low_lwc_other.tolist())), np.log10(max(high_lwc_other.tolist() + low_lwc_other.tolist())), num_bins)
# sum_rwc_high, xedges, yedges = np.histogram2d(high_concentration_other, high_lwc_other, bins=[x_bins, y_bins], weights=high_rwc_other)
# sum_lwc_high, _, _ = np.histogram2d(high_concentration_other, high_lwc_other, bins=[x_bins, y_bins], weights=high_lwc_other)
# counts_high, _, _ = np.histogram2d(high_concentration_other, high_lwc_other, bins=[x_bins, y_bins])

# avg_rwc_high = np.divide(sum_rwc_high, counts_high, out=np.full_like(sum_rwc_high, np.nan), where=counts_high > 0)
# avg_lwc_high = np.divide(sum_lwc_high, counts_high, out=np.full_like(sum_lwc_high, np.nan), where=counts_high > 0)
# rwc_lwc_ratio_high = np.divide(avg_rwc_high, avg_lwc_high, out=np.full_like(avg_rwc_high, np.nan), where=avg_lwc_high > 0) * 100
# masked_rwc_high = np.ma.masked_where(np.isnan(rwc_lwc_ratio_high), rwc_lwc_ratio_high)
# sum_rwc_low, _, _ = np.histogram2d(low_concentration_other, low_lwc_other, bins=[x_bins, y_bins], weights=low_rwc_other)
# sum_lwc_low, _, _ = np.histogram2d(low_concentration_other, low_lwc_other, bins=[x_bins, y_bins], weights=low_lwc_other)
# counts_low, _, _ = np.histogram2d(low_concentration_other, low_lwc_other, bins=[x_bins, y_bins])

# avg_rwc_low = np.divide(sum_rwc_low, counts_low, out=np.full_like(sum_rwc_low, np.nan), where=counts_low > 0)
# avg_lwc_low = np.divide(sum_lwc_low, counts_low, out=np.full_like(sum_lwc_low, np.nan), where=counts_low > 0)
# rwc_lwc_ratio_low = np.divide(avg_rwc_low, avg_lwc_low, out=np.full_like(avg_rwc_low, np.nan), where=avg_lwc_low > 0) * 100
# masked_rwc_low = np.ma.masked_where(np.isnan(rwc_lwc_ratio_low), rwc_lwc_ratio_low)
# plt.figure(figsize=(8, 6))
# norm = mcolors.Normalize(vmin=1, vmax=100)
# plt.pcolormesh(xedges, yedges, masked_rwc_high.T, cmap="RdBu_r", norm=norm, shading='auto')
# plt.colorbar(label="RWC / LWC (%)")
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Nr+Nc /cm³', fontsize=19, fontweight='bold')
# plt.ylabel('LWC g/m³', fontsize=19, fontweight='bold')
# plt.xticks(fontsize=19, fontweight='bold')
# plt.yticks(fontsize=19, fontweight='bold')
# plt.title('High GCCN Flights (Other Half)', fontsize=19, fontweight='bold')
# plt.tight_layout()
# plt.show()
# plt.figure(figsize=(8, 6))
# plt.pcolormesh(xedges, yedges, masked_rwc_low.T, cmap="RdBu_r", norm=norm, shading='auto')
# plt.colorbar(label="RWC / LWC (%)")
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Nr+Nc /cm³', fontsize=19, fontweight='bold')
# plt.ylabel('LWC g/m³', fontsize=19, fontweight='bold')
# plt.xticks(fontsize=19, fontweight='bold')
# plt.yticks(fontsize=19, fontweight='bold')
# plt.title('Low GCCN Flights (Other Half)', fontsize=19, fontweight='bold')
# plt.tight_layout()
# plt.show()
# #%%
# # Filter to fixed LWC and concentration region for the OTHER half
# x_min, x_max = 50, 200  # Nr+Nc range
# y_min, y_max = 0.1, 0.3  # LWC range
# high_mask = (high_concentration_other >= x_min) & (high_concentration_other <= x_max) & \
#             (high_lwc_other >= y_min) & (high_lwc_other <= y_max)
# filtered_high_concentration = high_concentration_other[high_mask]
# filtered_high_lwc = high_lwc_other[high_mask]
# filtered_high_rwc = high_rwc_other[high_mask]
# low_mask = (low_concentration_other >= x_min) & (low_concentration_other <= x_max) & \
#            (low_lwc_other >= y_min) & (low_lwc_other <= y_max)
# filtered_low_concentration = low_concentration_other[low_mask]
# filtered_low_lwc = low_lwc_other[low_mask]
# filtered_low_rwc = low_rwc_other[low_mask]
# num_bins = 3
# x_bins = np.logspace(np.log10(x_min), np.log10(x_max), num_bins)
# y_bins = np.logspace(np.log10(y_min), np.log10(y_max), num_bins)
# sum_rwc_high, xedges, yedges = np.histogram2d(filtered_high_concentration, filtered_high_lwc, bins=[x_bins, y_bins], weights=filtered_high_rwc)
# sum_lwc_high, _, _ = np.histogram2d(filtered_high_concentration, filtered_high_lwc, bins=[x_bins, y_bins], weights=filtered_high_lwc)
# counts_high, _, _ = np.histogram2d(filtered_high_concentration, filtered_high_lwc, bins=[x_bins, y_bins])

# avg_rwc_high = np.divide(sum_rwc_high, counts_high, out=np.full_like(sum_rwc_high, np.nan), where=counts_high > 0)
# avg_lwc_high = np.divide(sum_lwc_high, counts_high, out=np.full_like(sum_lwc_high, np.nan), where=counts_high > 0)
# rwc_lwc_ratio_high = np.divide(avg_rwc_high, avg_lwc_high, out=np.full_like(avg_rwc_high, np.nan), where=avg_lwc_high > 0) * 100
# masked_rwc_high = np.ma.masked_where(np.isnan(rwc_lwc_ratio_high), rwc_lwc_ratio_high)
# sum_rwc_low, _, _ = np.histogram2d(filtered_low_concentration, filtered_low_lwc, bins=[x_bins, y_bins], weights=filtered_low_rwc)
# sum_lwc_low, _, _ = np.histogram2d(filtered_low_concentration, filtered_low_lwc, bins=[x_bins, y_bins], weights=filtered_low_lwc)
# counts_low, _, _ = np.histogram2d(filtered_low_concentration, filtered_low_lwc, bins=[x_bins, y_bins])

# avg_rwc_low = np.divide(sum_rwc_low, counts_low, out=np.full_like(sum_rwc_low, np.nan), where=counts_low > 0)
# avg_lwc_low = np.divide(sum_lwc_low, counts_low, out=np.full_like(sum_lwc_low, np.nan), where=counts_low > 0)
# rwc_lwc_ratio_low = np.divide(avg_rwc_low, avg_lwc_low, out=np.full_like(avg_rwc_low, np.nan), where=avg_lwc_low > 0) * 100
# masked_rwc_low = np.ma.masked_where(np.isnan(rwc_lwc_ratio_low), rwc_lwc_ratio_low)

# # Gray out missing bins
# gray_values_high = np.full_like(rwc_lwc_ratio_high, np.nan)
# gray_values_high[np.isnan(rwc_lwc_ratio_high)] = 1
# gray_values_low = np.full_like(rwc_lwc_ratio_low, np.nan)
# gray_values_low[np.isnan(rwc_lwc_ratio_low)] = 1
# plt.figure(figsize=(8, 6))
# norm = mcolors.Normalize(vmin=1, vmax=100)
# img = plt.pcolormesh(xedges, yedges, masked_rwc_high.T, cmap="RdBu_r", norm=norm, shading='auto')
# plt.pcolormesh(xedges, yedges, gray_values_high.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)
# cbar = plt.colorbar(img)
# cbar.set_label("RWC / LWC (%)", fontsize=18, fontweight='bold')
# cbar.ax.tick_params(labelsize=19, width=2, length=5)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
# plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
# plt.title('High GCCN Flights (Other Half, Fixed Region)', fontsize=19, fontweight='bold')
# plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
# plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
# plt.tight_layout()
# plt.show()
# plt.figure(figsize=(8, 6))
# img = plt.pcolormesh(xedges, yedges, masked_rwc_low.T, cmap="RdBu_r", norm=norm, shading='auto')
# plt.pcolormesh(xedges, yedges, gray_values_low.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)
# cbar = plt.colorbar(img)
# cbar.set_label("RWC / LWC (%)", fontsize=18, fontweight='bold')
# cbar.ax.tick_params(labelsize=19, width=2, length=5)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
# plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
# plt.title('Low GCCN Flights (Other Half, Fixed Region)', fontsize=19, fontweight='bold')
# plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
# plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
# plt.tight_layout()
# plt.show()
# #%%
# # Compute the difference in each bin (using the OTHER half sample data)
# diff_rwc_lwc = avg_rwc_high - avg_rwc_low  # These are already from the OTHER half context
# masked_diff = np.ma.masked_where(np.isnan(diff_rwc_lwc), diff_rwc_lwc)
# abs_max = max(abs(np.nanmin(diff_rwc_lwc)), abs(np.nanmax(diff_rwc_lwc)))
# vmin, vmax = -abs_max, abs_max
# divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
# cmap = plt.cm.RdBu_r  
# gray_mask = np.isnan(diff_rwc_lwc)
# gray_values = np.full_like(diff_rwc_lwc, np.nan)
# gray_values[gray_mask] = 1  
# plt.figure(figsize=(8, 6))
# img = plt.pcolormesh(xedges, yedges, masked_diff.T, cmap=cmap, norm=divnorm, shading='auto')
# plt.pcolormesh(xedges, yedges, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)
# cbar = plt.colorbar(img)
# cbar.set_label("RWC/LWC Difference (%)", fontsize=18, fontweight='bold')
# cbar.ax.tick_params(labelsize=19, width=2, length=5)

# plt.xscale('log')
# plt.yscale('log')
# plt.xticks(fontsize=19, fontweight='bold')
# plt.yticks(fontsize=19, fontweight='bold')
# plt.tick_params(axis='both', which='major', labelsize=16, width=3, length=8)
# plt.tick_params(axis='both', which='minor', labelsize=16, width=2, length=5)

# plt.title('Difference in RWC/LWC\nOther Half of Flights, Jan–Jun 2022', fontsize=17, fontweight='bold')
# plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
# plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
# plt.tight_layout()
# plt.show()
#%%
# # Calculate RWC/LWC ratios before binning (OTHER HALF)
# filtered_high_ratio = filtered_high_rwc / filtered_high_lwc
# filtered_low_ratio = filtered_low_rwc / filtered_low_lwc
# n_bootstrap = 10000
# confidence_level = 0.90
# lower_percentile = (1 - confidence_level) / 2 * 100
# upper_percentile = (1 + confidence_level) / 2 * 100
# significance_mask = np.full((len(x_bins) - 1, len(y_bins) - 1), False)
# for i in range(len(x_bins) - 1):
#     for j in range(len(y_bins) - 1):
#         bin_high = (filtered_high_concentration >= x_bins[i]) & (filtered_high_concentration < x_bins[i+1]) & \
#                    (filtered_high_lwc >= y_bins[j]) & (filtered_high_lwc < y_bins[j+1])
#         bin_low = (filtered_low_concentration >= x_bins[i]) & (filtered_low_concentration < x_bins[i+1]) & \
#                   (filtered_low_lwc >= y_bins[j]) & (filtered_low_lwc < y_bins[j+1])

#         high_ratios_in_bin = filtered_high_ratio[bin_high]
#         low_ratios_in_bin = filtered_low_ratio[bin_low]

#         if len(high_ratios_in_bin) >= 3 and len(low_ratios_in_bin) >= 3:
#             boot_diff = []
#             for _ in range(n_bootstrap):
#                 sample_high = np.random.choice(high_ratios_in_bin, size=len(high_ratios_in_bin), replace=True)
#                 sample_low = np.random.choice(low_ratios_in_bin, size=len(low_ratios_in_bin), replace=True)
#                 boot_diff.append(np.mean(sample_high) - np.mean(sample_low))
            
#             lower = np.percentile(boot_diff, lower_percentile)
#             upper = np.percentile(boot_diff, upper_percentile)

#             if lower > 0 or upper < 0:
#                 significance_mask[i, j] = True
# masked_diff = np.ma.masked_where(np.isnan(diff_rwc_lwc), diff_rwc_lwc)
# abs_max = max(abs(np.nanmin(diff_rwc_lwc)), abs(np.nanmax(diff_rwc_lwc)))
# divnorm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

# plt.figure(figsize=(8, 6))
# img = plt.pcolormesh(xedges, yedges, masked_diff.T, cmap="RdBu_r", norm=divnorm, shading='auto')

# gray_values = np.full_like(diff_rwc_lwc, np.nan)
# gray_values[np.isnan(diff_rwc_lwc)] = 1
# plt.pcolormesh(xedges, yedges, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)
# plt.pcolormesh(
#     xedges, yedges,
#     np.ma.masked_where(~significance_mask, significance_mask).T,
#     shading='auto',
#     hatch='///',
#     alpha=0.0,
#     edgecolors='black',
#     linewidth=0.0
# )

# cbar = plt.colorbar(img)
# cbar.set_label("RWC / LWC Difference (%)", fontsize=18, fontweight='bold')
# cbar.ax.tick_params(labelsize=19, width=2, length=5)

# plt.xscale('log')
# plt.yscale('log')
# plt.xticks(fontsize=19, fontweight='bold')
# plt.yticks(fontsize=19, fontweight='bold')
# plt.title('Difference in RWC/LWC\nOther Half of Flights (Bootstrap Test)', fontsize=17, fontweight='bold')
# plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
# plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
# plt.tight_layout()
# plt.show()
# #%%
# # %% Create output directory
# os.makedirs("bootstrap_histograms_other_half", exist_ok=True)

# # Compute ratios again just to be safe
# filtered_high_ratio = filtered_high_rwc / filtered_high_lwc
# filtered_low_ratio = filtered_low_rwc / filtered_low_lwc
# bin_i_list = []
# bin_j_list = []
# mean_diff_list = []
# lower_ci_list = []
# upper_ci_list = []
# significance_list = []
# significance_mask = np.full((len(x_bins) - 1, len(y_bins) - 1), False)

# for i in range(len(x_bins) - 1):
#     for j in range(len(y_bins) - 1):
#         bin_high = (filtered_high_concentration >= x_bins[i]) & (filtered_high_concentration < x_bins[i+1]) & \
#                    (filtered_high_lwc >= y_bins[j]) & (filtered_high_lwc < y_bins[j+1])
#         bin_low = (filtered_low_concentration >= x_bins[i]) & (filtered_low_concentration < x_bins[i+1]) & \
#                   (filtered_low_lwc >= y_bins[j]) & (filtered_low_lwc < y_bins[j+1])

#         high_ratios_in_bin = filtered_high_ratio[bin_high]
#         low_ratios_in_bin = filtered_low_ratio[bin_low]

#         if len(high_ratios_in_bin) >= 3 and len(low_ratios_in_bin) >= 3:
#             boot_diff = []
#             for _ in range(n_bootstrap):
#                 sample_high = np.random.choice(high_ratios_in_bin, size=len(high_ratios_in_bin), replace=True)
#                 sample_low = np.random.choice(low_ratios_in_bin, size=len(low_ratios_in_bin), replace=True)
#                 boot_diff.append(np.mean(sample_high) - np.mean(sample_low))

#             boot_diff = np.array(boot_diff)
#             mean_diff = np.mean(boot_diff)
#             lower = np.percentile(boot_diff, lower_percentile)
#             upper = np.percentile(boot_diff, upper_percentile)
#             significant = (lower > 0) or (upper < 0)
#             if significant:
#                 significance_mask[i, j] = True

#             bin_i_list.append(i)
#             bin_j_list.append(j)
#             mean_diff_list.append(mean_diff)
#             lower_ci_list.append(lower)
#             upper_ci_list.append(upper)
#             significance_list.append(significant)

#             plt.figure(figsize=(6, 5))
#             plt.hist(boot_diff, bins=50, density=True)
#             plt.axvline(0, color='k', linestyle='--', label='Zero')
#             plt.title(f'Bootstrap Dist. (High - Low) for bin ({i},{j})')
#             plt.xlabel('Mean Difference (RWC/LWC)')
#             plt.ylabel('Probability Density')
#             if significant:
#                 plt.text(0.05, 0.9, 'Significant!', transform=plt.gca().transAxes, fontsize=12, color='red')
#             plt.legend()
#             plt.tight_layout()
#             plt.savefig(f"bootstrap_histograms_other_half/bootstrap_bin_{i}_{j}.png")
#             plt.close()
#         else:
#             bin_i_list.append(i)
#             bin_j_list.append(j)
#             mean_diff_list.append(np.nan)
#             lower_ci_list.append(np.nan)
#             upper_ci_list.append(np.nan)
#             significance_list.append(False)
# summary_table_other_half = pd.DataFrame({
#     'Bin_i': bin_i_list,
#     'Bin_j': bin_j_list,
#     'Mean_Difference': mean_diff_list,
#     '5th_Percentile': lower_ci_list,
#     '95th_Percentile': upper_ci_list,
#     'Significant': significance_list
# })
# #%%
# # Plot difference with significance hatching
# plt.figure(figsize=(8, 6))
# img = plt.pcolormesh(xedges, yedges, masked_diff.T, cmap="RdBu_r", norm=divnorm, shading='auto')

# gray_mask = np.isnan(diff_rwc_lwc)
# gray_values = np.full_like(diff_rwc_lwc, np.nan)
# gray_values[gray_mask] = 1
# plt.pcolormesh(xedges, yedges, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)

# plt.pcolormesh(
#     xedges, yedges,
#     np.ma.masked_where(~significance_mask, significance_mask).T,
#     shading='auto',
#     hatch='///',
#     facecolors='none',
#     edgecolors='black', linewidth=0.0
# )

# cbar = plt.colorbar(img)
# cbar.set_label("RWC/LWC Difference (%)", fontsize=18, fontweight='bold')
# cbar.ax.tick_params(labelsize=19, width=2, length=5)

# plt.xscale('log')
# plt.yscale('log')
# plt.xticks(fontsize=19, fontweight='bold')
# plt.yticks(fontsize=19, fontweight='bold')
# plt.tick_params(axis='both', which='major', labelsize=16, width=3, length=8)
# plt.tick_params(axis='both', which='minor', labelsize=16, width=2, length=5)

# plt.title('Difference in RWC/LWC\nOther Half of Flights, Jan–Jun 2022', fontsize=17, fontweight='bold')
# plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
# plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
# plt.tight_layout()
# plt.show()
# #%%
# # Count number of data points per bin
# bin_i_list = []
# bin_j_list = []
# high_count_list = []
# low_count_list = []

# for i in range(len(x_bins) - 1):
#     for j in range(len(y_bins) - 1):
#         bin_high = (filtered_high_concentration >= x_bins[i]) & (filtered_high_concentration < x_bins[i+1]) & \
#                    (filtered_high_lwc >= y_bins[j]) & (filtered_high_lwc < y_bins[j+1])
#         bin_low = (filtered_low_concentration >= x_bins[i]) & (filtered_low_concentration < x_bins[i+1]) & \
#                   (filtered_low_lwc >= y_bins[j]) & (filtered_low_lwc < y_bins[j+1])

#         high_count = np.sum(bin_high)
#         low_count = np.sum(bin_low)

#         bin_i_list.append(i)
#         bin_j_list.append(j)
#         high_count_list.append(high_count)
#         low_count_list.append(low_count)

# bootstrap_counts_other_half = pd.DataFrame({
#     'Bin_i': bin_i_list,
#     'Bin_j': bin_j_list,
#     'High_GCCN_Count': high_count_list,
#     'Low_GCCN_Count': low_count_list
# })
# print(bootstrap_counts_other_half)
# #%%
# # Compute Standard Error Matrix for difference
# sem_diff_matrix = np.full((len(x_bins) - 1, len(y_bins) - 1), np.nan)

# for i in range(len(x_bins) - 1):
#     for j in range(len(y_bins) - 1):
#         bin_high = (filtered_high_concentration >= x_bins[i]) & (filtered_high_concentration < x_bins[i+1]) & \
#                    (filtered_high_lwc >= y_bins[j]) & (filtered_high_lwc < y_bins[j+1])
#         bin_low = (filtered_low_concentration >= x_bins[i]) & (filtered_low_concentration < x_bins[i+1]) & \
#                   (filtered_low_lwc >= y_bins[j]) & (filtered_low_lwc < y_bins[j+1])

#         high_vals = filtered_high_rwc[bin_high] / filtered_high_lwc[bin_high]
#         low_vals = filtered_low_rwc[bin_low] / filtered_low_lwc[bin_low]

#         if len(high_vals) >= 3 and len(low_vals) >= 3:
#             sem_high = np.std(high_vals, ddof=1) / np.sqrt(len(high_vals))
#             sem_low = np.std(low_vals, ddof=1) / np.sqrt(len(low_vals))
#             sem_diff = np.sqrt(sem_high**2 + sem_low**2)
#             sem_diff_matrix[i, j] = sem_diff

# #%%
# plt.figure(figsize=(8, 6))
# img = plt.pcolormesh(xedges, yedges, masked_diff.T, cmap="RdBu_r", norm=divnorm, shading='auto')
# plt.pcolormesh(xedges, yedges, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)
# plt.pcolormesh(
#     xedges, yedges,
#     np.ma.masked_where(~significance_mask, significance_mask).T,
#     shading='flat',
#     hatch='///',
#     facecolors='none',
#     edgecolors='black',
#     linewidth=0.0
# ).set_zorder(10)
# for i in range(len(x_bins) - 1):
#     for j in range(len(y_bins) - 1):
#         diff = diff_rwc_lwc[i, j]
#         sem = sem_diff_matrix[i, j]
#         if not np.isnan(diff) and not np.isnan(sem):
#             x_center = (xedges[i] + xedges[i+1]) / 2
#             y_center = (yedges[j] + yedges[j+1]) / 2
#             plt.text(
#                 x_center, y_center,
#                 f"{diff:+.2f}%\n±{sem:.2f}%",
#                 ha='center', va='center',
#                 fontsize=11, color='black', fontweight='bold'
#             )

# cbar = plt.colorbar(img)
# cbar.set_label("RWC/LWC Difference (%)", fontsize=18, fontweight='bold')
# cbar.ax.tick_params(labelsize=19, width=2, length=5)

# plt.xscale('log')
# plt.yscale('log')
# plt.xticks(fontsize=19, fontweight='bold')
# plt.yticks(fontsize=19, fontweight='bold')
# plt.tick_params(axis='both', which='major', labelsize=16, width=3, length=8)
# plt.tick_params(axis='both', which='minor', labelsize=16, width=2, length=5)

# plt.title('Difference in RWC/LWC\nOther Half of Flights, Jan–Jun 2022', fontsize=17, fontweight='bold')
# plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
# plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
# plt.tight_layout()
# plt.show()
#%%
# # Define the outlier date to exclude
# excluded_date = '2022-05-10'
# filtered_high_concentration = np.array([val for i, val in enumerate(high_concentration) if high_gccn_data[i]['Date'] != excluded_date])
# filtered_high_lwc = np.array([val for i, val in enumerate(high_lwc) if high_gccn_data[i]['Date'] != excluded_date])
# filtered_high_rwc = np.array([val for i, val in enumerate(high_rwc) if high_gccn_data[i]['Date'] != excluded_date])

# filtered_low_concentration = np.array([val for i, val in enumerate(low_concentration) if low_gccn_data[i]['Date'] != excluded_date])
# filtered_low_lwc = np.array([val for i, val in enumerate(low_lwc) if low_gccn_data[i]['Date'] != excluded_date])
# filtered_low_rwc = np.array([val for i, val in enumerate(low_rwc) if low_gccn_data[i]['Date'] != excluded_date])
# x_min, x_max = 50, 200
# y_min, y_max = 0.1, 0.3
# high_mask = (filtered_high_concentration >= x_min) & (filtered_high_concentration <= x_max) & \
#             (filtered_high_lwc >= y_min) & (filtered_high_lwc <= y_max)
# low_mask = (filtered_low_concentration >= x_min) & (filtered_low_concentration <= x_max) & \
#            (filtered_low_lwc >= y_min) & (filtered_low_lwc <= y_max)

# filtered_high_concentration = filtered_high_concentration[high_mask]
# filtered_high_lwc = filtered_high_lwc[high_mask]
# filtered_high_rwc = filtered_high_rwc[high_mask]

# filtered_low_concentration = filtered_low_concentration[low_mask]
# filtered_low_lwc = filtered_low_lwc[low_mask]
# filtered_low_rwc = filtered_low_rwc[low_mask]
# num_bins = 3
# x_bins = np.logspace(np.log10(x_min), np.log10(x_max), num_bins)
# y_bins = np.logspace(np.log10(y_min), np.log10(y_max), num_bins)
# sum_rwc_high, xedges, yedges = np.histogram2d(filtered_high_concentration, filtered_high_lwc, bins=[x_bins, y_bins], weights=filtered_high_rwc)
# sum_lwc_high, _, _ = np.histogram2d(filtered_high_concentration, filtered_high_lwc, bins=[x_bins, y_bins], weights=filtered_high_lwc)
# counts_high, _, _ = np.histogram2d(filtered_high_concentration, filtered_high_lwc, bins=[x_bins, y_bins])

# avg_rwc_high = np.divide(sum_rwc_high, counts_high, out=np.full_like(sum_rwc_high, np.nan), where=counts_high > 0)
# avg_lwc_high = np.divide(sum_lwc_high, counts_high, out=np.full_like(sum_lwc_high, np.nan), where=counts_high > 0)
# rwc_lwc_ratio_high = np.divide(avg_rwc_high, avg_lwc_high, out=np.full_like(avg_rwc_high, np.nan), where=avg_lwc_high > 0) * 100
# masked_rwc_high = np.ma.masked_where(np.isnan(rwc_lwc_ratio_high), rwc_lwc_ratio_high)
# sum_rwc_low, _, _ = np.histogram2d(filtered_low_concentration, filtered_low_lwc, bins=[x_bins, y_bins], weights=filtered_low_rwc)
# sum_lwc_low, _, _ = np.histogram2d(filtered_low_concentration, filtered_low_lwc, bins=[x_bins, y_bins], weights=filtered_low_lwc)
# counts_low, _, _ = np.histogram2d(filtered_low_concentration, filtered_low_lwc, bins=[x_bins, y_bins])

# avg_rwc_low = np.divide(sum_rwc_low, counts_low, out=np.full_like(sum_rwc_low, np.nan), where=counts_low > 0)
# avg_lwc_low = np.divide(sum_lwc_low, counts_low, out=np.full_like(sum_lwc_low, np.nan), where=counts_low > 0)
# rwc_lwc_ratio_low = np.divide(avg_rwc_low, avg_lwc_low, out=np.full_like(avg_rwc_low, np.nan), where=avg_lwc_low > 0) * 100
# masked_rwc_low = np.ma.masked_where(np.isnan(rwc_lwc_ratio_low), rwc_lwc_ratio_low)
# gray_mask_high = np.isnan(rwc_lwc_ratio_high)
# gray_values_high = np.full_like(rwc_lwc_ratio_high, np.nan)
# gray_values_high[gray_mask_high] = 1

# gray_mask_low = np.isnan(rwc_lwc_ratio_low)
# gray_values_low = np.full_like(rwc_lwc_ratio_low, np.nan)
# gray_values_low[gray_mask_low] = 1
# plt.figure(figsize=(8, 6))
# norm = mcolors.Normalize(vmin=1, vmax=100)
# img = plt.pcolormesh(xedges, yedges, masked_rwc_high.T, cmap="RdBu_r", norm=norm, shading='auto')
# plt.pcolormesh(xedges, yedges, gray_values_high.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)
# cbar = plt.colorbar(img)
# cbar.set_label("RWC / LWC (%)", fontsize=18, fontweight='bold')
# cbar.ax.tick_params(labelsize=19, width=2, length=5)
# for t in cbar.ax.get_yticklabels():
#     t.set_fontweight('bold')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
# plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
# plt.title('High GCCN Flights (Outlier Removed)', fontsize=19, fontweight='bold')
# plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
# plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
# plt.tight_layout()
# plt.show()
# plt.figure(figsize=(8, 6))
# img = plt.pcolormesh(xedges, yedges, masked_rwc_low.T, cmap="RdBu_r", norm=norm, shading='auto')
# plt.pcolormesh(xedges, yedges, gray_values_low.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)
# cbar = plt.colorbar(img)
# cbar.set_label("RWC / LWC (%)", fontsize=18, fontweight='bold')
# cbar.ax.tick_params(labelsize=19, width=2, length=5)
# for t in cbar.ax.get_yticklabels():
#     t.set_fontweight('bold')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
# plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
# plt.title('Low GCCN Flights (Outlier Removed)', fontsize=19, fontweight='bold')
# plt.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
# plt.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
# plt.tight_layout()
# plt.show()

# # %%
# # Difference in each bin (outlier already removed in upstream processing)
# diff_rwc_lwc = avg_rwc_high - avg_rwc_low
# masked_diff = np.ma.masked_where(np.isnan(diff_rwc_lwc), diff_rwc_lwc)
# plt.figure(figsize=(8, 6))
# cmap = "RdBu_r"  # Red = Increase in High GCCN, Blue = Decrease
# img = plt.pcolormesh(x_bins, y_bins, masked_diff.T, cmap=cmap, shading='auto')
# gray_mask = np.isnan(diff_rwc_lwc)
# gray_values = np.full_like(diff_rwc_lwc, np.nan)
# gray_values[gray_mask] = 1
# plt.pcolormesh(x_bins, y_bins, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)

# cbar = plt.colorbar(img)
# cbar.set_label("RWC/LWC Difference", fontsize=14, fontweight='bold')
# cbar.ax.tick_params(labelsize=12, width=2, length=5)

# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=16, fontweight='bold')
# plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=16, fontweight='bold')
# plt.title('Difference in RWC/LWC between High and Low GCCN\n(Outlier Removed)', fontsize=14, fontweight='bold')
# plt.tick_params(axis='both', which='major', labelsize=11, width=3, length=8)
# plt.tick_params(axis='both', which='minor', labelsize=11, width=2, length=5)
# plt.tight_layout()
# plt.show()

# #%% Enhanced color scaling centered at 0

# # Recalculate for diverging scale
# diff_rwc_lwc = avg_rwc_high - avg_rwc_low
# masked_diff = np.ma.masked_where(np.isnan(diff_rwc_lwc), diff_rwc_lwc)
# abs_max = max(abs(np.nanmin(diff_rwc_lwc)), abs(np.nanmax(diff_rwc_lwc)))
# vmin, vmax = -abs_max, abs_max
# divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

# plt.figure(figsize=(8, 6))
# img = plt.pcolormesh(x_bins, y_bins, masked_diff.T, cmap=plt.cm.RdBu_r, norm=divnorm, shading='auto')

# gray_mask = np.isnan(diff_rwc_lwc)
# gray_values = np.full_like(diff_rwc_lwc, np.nan)
# gray_values[gray_mask] = 1
# plt.pcolormesh(x_bins, y_bins, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)

# cbar = plt.colorbar(img)
# cbar.set_label("RWC/LWC Difference", fontsize=18, fontweight='bold')
# cbar.ax.tick_params(labelsize=19, width=2, length=5)

# plt.xscale('log')
# plt.yscale('log')
# plt.xticks(fontsize=19, fontweight='bold')
# plt.yticks(fontsize=19, fontweight='bold')
# plt.tick_params(axis='both', which='major', labelsize=16, width=3, length=8)
# plt.tick_params(axis='both', which='minor', labelsize=16, width=2, length=5)

# plt.title('Difference in RWC/LWC\nbetween High and Low GCCN (Outlier Removed)', fontsize=17, fontweight='bold')
# plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
# plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')

# plt.tight_layout()
# plt.show()

# # %%
# os.makedirs("bootstrap_histograms", exist_ok=True)

# filtered_high_ratio = filtered_high_rwc / filtered_high_lwc
# filtered_low_ratio = filtered_low_rwc / filtered_low_lwc

# n_bootstrap = 10000
# confidence_level = 0.90
# lower_percentile = (1 - confidence_level) / 2 * 100
# upper_percentile = (1 + confidence_level) / 2 * 100

# bin_i_list = []
# bin_j_list = []
# mean_diff_list = []
# lower_ci_list = []
# upper_ci_list = []
# significance_list = []
# significance_mask = np.full((len(x_bins) - 1, len(y_bins) - 1), False)

# for i in range(len(x_bins) - 1):
#     for j in range(len(y_bins) - 1):
#         bin_high = (filtered_high_concentration >= x_bins[i]) & (filtered_high_concentration < x_bins[i+1]) & \
#                    (filtered_high_lwc >= y_bins[j]) & (filtered_high_lwc < y_bins[j+1])
#         bin_low = (filtered_low_concentration >= x_bins[i]) & (filtered_low_concentration < x_bins[i+1]) & \
#                   (filtered_low_lwc >= y_bins[j]) & (filtered_low_lwc < y_bins[j+1])

#         high_ratios_in_bin = filtered_high_ratio[bin_high]
#         low_ratios_in_bin = filtered_low_ratio[bin_low]

#         if len(high_ratios_in_bin) >= 3 and len(low_ratios_in_bin) >= 3:
#             boot_diff = []
#             for _ in range(n_bootstrap):
#                 sample_high = np.random.choice(high_ratios_in_bin, size=len(high_ratios_in_bin), replace=True)
#                 sample_low = np.random.choice(low_ratios_in_bin, size=len(low_ratios_in_bin), replace=True)
#                 boot_diff.append(np.mean(sample_high) - np.mean(sample_low))

#             boot_diff = np.array(boot_diff)
#             mean_diff = np.mean(boot_diff)
#             lower = np.percentile(boot_diff, lower_percentile)
#             upper = np.percentile(boot_diff, upper_percentile)
#             significant = (lower > 0) or (upper < 0)

#             bin_i_list.append(i)
#             bin_j_list.append(j)
#             mean_diff_list.append(mean_diff)
#             lower_ci_list.append(lower)
#             upper_ci_list.append(upper)
#             significance_list.append(significant)

#             if significant:
#                 significance_mask[i, j] = True

#             plt.figure(figsize=(6,5))
#             plt.hist(boot_diff, bins=50, density=True)
#             plt.axvline(0, color='k', linestyle='--', label='Zero')
#             plt.title(f'Bootstrap Dist. (High - Low) for bin ({i},{j})')
#             plt.xlabel('Mean Difference (RWC/LWC)')
#             plt.ylabel('Probability Density')
#             if significant:
#                 plt.text(0.05, 0.9, 'Significant!', transform=plt.gca().transAxes, fontsize=12, color='red')
#             plt.legend()
#             plt.tight_layout()
#             plt.savefig(f"bootstrap_histograms/bootstrap_bin_{i}_{j}.png")
#             plt.close()

#         else:
#             bin_i_list.append(i)
#             bin_j_list.append(j)
#             mean_diff_list.append(np.nan)
#             lower_ci_list.append(np.nan)
#             upper_ci_list.append(np.nan)
#             significance_list.append(False)

# summary_table = pd.DataFrame({
#     'Bin_i': bin_i_list,
#     'Bin_j': bin_j_list,
#     'Mean_Difference': mean_diff_list,
#     '5th_Percentile': lower_ci_list,
#     '95th_Percentile': upper_ci_list,
#     'Significant': significance_list
# })

# # %%
# plt.figure(figsize=(8, 6))
# img = plt.pcolormesh(xedges, yedges, masked_diff.T, cmap="RdBu_r", norm=divnorm, shading='auto')
# gray_mask = np.isnan(diff_rwc_lwc)
# gray_values = np.full_like(diff_rwc_lwc, np.nan)
# gray_values[gray_mask] = 1
# plt.pcolormesh(xedges, yedges, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)
# plt.pcolormesh(
#     xedges, yedges,
#     np.ma.masked_where(~significance_mask, significance_mask).T,
#     shading='auto',
#     hatch='///',
#     facecolors='none',
#     edgecolors='black', linewidth=0.0
# )

# cbar = plt.colorbar(img)
# cbar.set_label("RWC/LWC Difference (%)", fontsize=18, fontweight='bold')
# cbar.ax.tick_params(labelsize=19, width=2, length=5)
# plt.xscale('log')
# plt.yscale('log')
# plt.xticks(fontsize=19, fontweight='bold')
# plt.yticks(fontsize=19, fontweight='bold')
# plt.tick_params(axis='both', which='major', labelsize=16, width=3, length=8)
# plt.tick_params(axis='both', which='minor', labelsize=16, width=2, length=5)

# plt.title('Difference in RWC/LWC\nHigh vs Low GCCN\n(Bootstrap Test)', fontsize=17, fontweight='bold')
# plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
# plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
# plt.tight_layout()
# plt.show()

# # %%
# bin_i_list = []
# bin_j_list = []
# high_count_list = []
# low_count_list = []

# for i in range(len(x_bins) - 1):
#     for j in range(len(y_bins) - 1):
#         bin_high = (filtered_high_concentration >= x_bins[i]) & (filtered_high_concentration < x_bins[i+1]) & \
#                    (filtered_high_lwc >= y_bins[j]) & (filtered_high_lwc < y_bins[j+1])
#         bin_low = (filtered_low_concentration >= x_bins[i]) & (filtered_low_concentration < x_bins[i+1]) & \
#                   (filtered_low_lwc >= y_bins[j]) & (filtered_low_lwc < y_bins[j+1])

#         high_count = np.sum(bin_high)
#         low_count = np.sum(bin_low)

#         bin_i_list.append(i)
#         bin_j_list.append(j)
#         high_count_list.append(high_count)
#         low_count_list.append(low_count)

# bootstrap_counts = pd.DataFrame({
#     'Bin_i': bin_i_list,
#     'Bin_j': bin_j_list,
#     'High_GCCN_Count': high_count_list,
#     'Low_GCCN_Count': low_count_list
# })
# print(bootstrap_counts)

# # %%
# sem_diff_matrix = np.full((len(x_bins) - 1, len(y_bins) - 1), np.nan)

# for i in range(len(x_bins) - 1):
#     for j in range(len(y_bins) - 1):
#         bin_high = (filtered_high_concentration >= x_bins[i]) & (filtered_high_concentration < x_bins[i+1]) & \
#                    (filtered_high_lwc >= y_bins[j]) & (filtered_high_lwc < y_bins[j+1])
#         bin_low = (filtered_low_concentration >= x_bins[i]) & (filtered_low_concentration < x_bins[i+1]) & \
#                   (filtered_low_lwc >= y_bins[j]) & (filtered_low_lwc < y_bins[j+1])

#         high_vals = filtered_high_rwc[bin_high] / filtered_high_lwc[bin_high]
#         low_vals = filtered_low_rwc[bin_low] / filtered_low_lwc[bin_low]

#         if len(high_vals) >= 3 and len(low_vals) >= 3:
#             sem_high = np.std(high_vals, ddof=1) / np.sqrt(len(high_vals))
#             sem_low = np.std(low_vals, ddof=1) / np.sqrt(len(low_vals))
#             sem_diff = np.sqrt(sem_high**2 + sem_low**2)
#             sem_diff_matrix[i, j] = sem_diff

# # %%
# plt.figure(figsize=(8, 6))
# img = plt.pcolormesh(xedges, yedges, masked_diff.T, cmap="RdBu_r", norm=divnorm, shading='auto')
# gray_mask = np.isnan(diff_rwc_lwc)
# gray_values = np.full_like(diff_rwc_lwc, np.nan)
# gray_values[gray_mask] = 1
# plt.pcolormesh(xedges, yedges, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)
# hatch_layer = plt.pcolormesh(
#     xedges, yedges,
#     np.ma.masked_where(~significance_mask, significance_mask).T,
#     shading='flat',
#     hatch='///',
#     facecolors='none',
#     edgecolors='black', linewidth=0.0
# )
# hatch_layer.set_zorder(10)

# for i in range(len(x_bins) - 1):
#     for j in range(len(y_bins) - 1):
#         sem = sem_diff_matrix[i, j]
#         if not np.isnan(sem):
#             x_center = (xedges[i] + xedges[i+1]) / 2
#             y_center = (yedges[j] + yedges[j+1]) / 2
#             plt.text(x_center, y_center, f"±{sem:.2f}%", ha='center', va='center',
#                      fontsize=11, color='black')

# cbar = plt.colorbar(img)
# cbar.set_label("RWC/LWC Difference (%)", fontsize=18, fontweight='bold')
# cbar.ax.tick_params(labelsize=19, width=2, length=5)

# plt.xscale('log')
# plt.yscale('log')
# plt.xticks(fontsize=19, fontweight='bold')
# plt.yticks(fontsize=19, fontweight='bold')
# plt.tick_params(axis='both', which='major', labelsize=16, width=3, length=8)
# plt.tick_params(axis='both', which='minor', labelsize=16, width=2, length=5)

# plt.title('Difference in RWC/LWC\nHigh vs Low GCCN\n(Bootstrap Test with SEM)', fontsize=17, fontweight='bold')
# plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
# plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
# plt.tight_layout()
# plt.show()

# # %%
# plt.figure(figsize=(8, 6))
# img = plt.pcolormesh(xedges, yedges, masked_diff.T, cmap="RdBu_r", norm=divnorm, shading='auto')
# gray_mask = np.isnan(diff_rwc_lwc)
# gray_values = np.full_like(diff_rwc_lwc, np.nan)
# gray_values[gray_mask] = 1
# plt.pcolormesh(xedges, yedges, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)

# plt.pcolormesh(
#     xedges, yedges,
#     np.ma.masked_where(~significance_mask, significance_mask).T,
#     shading='flat',
#     hatch='///',
#     facecolors='none',
#     edgecolors='black', linewidth=0.0
# ).set_zorder(10)

# for i in range(len(x_bins) - 1):
#     for j in range(len(y_bins) - 1):
#         diff = diff_rwc_lwc[i, j]
#         sem = sem_diff_matrix[i, j]
#         if not np.isnan(diff) and not np.isnan(sem):
#             x_center = (xedges[i] + xedges[i+1]) / 2
#             y_center = (yedges[j] + yedges[j+1]) / 2
#             plt.text(
#                 x_center, y_center,
#                 f"{diff:+.2f}%\n±{sem:.2f}%",
#                 ha='center', va='center',s
#                 fontsize=11, color='black', fontweight='bold'
#             )

# cbar = plt.colorbar(img)
# cbar.set_label("RWC/LWC Difference (%)", fontsize=18, fontweight='bold')
# cbar.ax.tick_params(labelsize=19, width=2, length=5)

# plt.xscale('log')
# plt.yscale('log')
# plt.xticks(fontsize=19, fontweight='bold')
# plt.yticks(fontsize=19, fontweight='bold')
# plt.tick_params(axis='both', which='major', labelsize=16, width=3, length=8)
# plt.tick_params(axis='both', which='minor', labelsize=16, width=2, length=5)

# plt.title('Difference in RWC/LWC\nBelow Cloud Base January-June 2022', fontsize=16, fontweight='bold')
# plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
# plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')

# plt.tight_layout()
# plt.show()

#%%
#Leave out method where I leave a new flight out each run 
unique_dates = list(average_gccn_per_flight.keys())

for excluded_date in unique_dates:
    print(f"\n=== Excluding flight: {excluded_date} ===")

    temp_avg_gccn = {
        date: avg for date, avg in average_gccn_per_flight.items() if date != excluded_date
    }

    gccn_vals_temp = np.array(list(temp_avg_gccn.values()))
    threshold = np.percentile(gccn_vals_temp, 50)

    high_dates = {date for date, val in temp_avg_gccn.items() if val >= threshold}
    low_dates = {date for date, val in temp_avg_gccn.items() if val < threshold}

    
    high_gccn_data = [entry for entry in total_combined_concentration if entry['Date'] in high_dates]
    low_gccn_data = [entry for entry in total_combined_concentration if entry['Date'] in low_dates]

    high_concentration = np.array([entry['Total_Combined_Concentration'] for entry in high_gccn_data])
    high_lwc = np.array([entry['Total_Liquid_Water'] for entry in total_liquid_water if entry['Date'] in high_dates])
    high_rwc = np.array([entry['RWC'] for entry in total_liquid_water if entry['Date'] in high_dates])

    low_concentration = np.array([entry['Total_Combined_Concentration'] for entry in low_gccn_data])
    low_lwc = np.array([entry['Total_Liquid_Water'] for entry in total_liquid_water if entry['Date'] in low_dates])
    low_rwc = np.array([entry['RWC'] for entry in total_liquid_water if entry['Date'] in low_dates])

    x_min, x_max = 50, 200
    y_min, y_max = 0.1, 0.3
    num_bins = 3

    x_bins = np.logspace(np.log10(x_min), np.log10(x_max), num_bins)
    y_bins = np.logspace(np.log10(y_min), np.log10(y_max), num_bins)

    def compute_rwc_lwc_ratio(concentration, lwc, rwc):
        mask = (concentration >= x_min) & (concentration <= x_max) & \
               (lwc >= y_min) & (lwc <= y_max)
        f_conc, f_lwc, f_rwc = concentration[mask], lwc[mask], rwc[mask]
        sum_rwc, xedges, yedges = np.histogram2d(f_conc, f_lwc, bins=[x_bins, y_bins], weights=f_rwc)
        sum_lwc, _, _ = np.histogram2d(f_conc, f_lwc, bins=[x_bins, y_bins], weights=f_lwc)
        counts, _, _ = np.histogram2d(f_conc, f_lwc, bins=[x_bins, y_bins])
        avg_rwc = np.divide(sum_rwc, counts, out=np.full_like(sum_rwc, np.nan), where=counts > 0)
        avg_lwc = np.divide(sum_lwc, counts, out=np.full_like(sum_lwc, np.nan), where=counts > 0)
        ratio = np.divide(avg_rwc, avg_lwc, out=np.full_like(avg_rwc, np.nan), where=avg_lwc > 0) * 100
        return np.ma.masked_where(np.isnan(ratio), ratio), xedges, yedges

    masked_rwc_high, xedges, yedges = compute_rwc_lwc_ratio(high_concentration, high_lwc, high_rwc)
    masked_rwc_low, _, _ = compute_rwc_lwc_ratio(low_concentration, low_lwc, low_rwc)

   
    norm = mcolors.Normalize(vmin=1, vmax=100)

    for i in range(diff_rwc_lwc.shape[0]):
        for j in range(diff_rwc_lwc.shape[1]):
            mean_val = diff_rwc_lwc[i, j]
            
            sem_high = np.nanstd(avg_rwc_high[i, j]) / np.sqrt(counts_high[i, j]) if counts_high[i, j] > 0 else np.nan
            sem_low = np.nanstd(avg_rwc_low[i, j]) / np.sqrt(counts_low[i, j]) if counts_low[i, j] > 0 else np.nan
            sem_diff = np.sqrt(sem_high**2 + sem_low**2) if not np.isnan(sem_high) and not np.isnan(sem_low) else np.nan

            if not np.isnan(mean_val):
                x_center = (xedges[i] + xedges[i + 1]) / 2
                y_center = (yedges[j] + yedges[j + 1]) / 2
                plt.text(
                    x_center, y_center,
                    f"{mean_val:+.2f}%\n±{sem_diff:.2f}%",
                    color='black',
                    ha='center', va='center',
                    fontsize=12, fontweight='bold'
                )


    plt.figure(figsize=(8, 6))
    plt.pcolormesh(xedges, yedges, masked_rwc_high.T, cmap="RdBu_r", norm=norm, shading='auto')
    plt.colorbar(label="RWC / LWC (%)")
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"High GCCN Flights (Excluded: {excluded_date})", fontsize=14, fontweight='bold')
    plt.xlabel('Nr+Nc (cm⁻³)', fontsize=12, fontweight='bold')
    plt.ylabel('LWC (g m⁻³)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(xedges, yedges, masked_rwc_low.T, cmap="RdBu_r", norm=norm, shading='auto')
    plt.colorbar(label="RWC / LWC (%)")
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"Low GCCN Flights (Excluded: {excluded_date})", fontsize=14, fontweight='bold')
    plt.xlabel('Nr+Nc (cm⁻³)', fontsize=12, fontweight='bold')
    plt.ylabel('LWC (g m⁻³)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

    diff_rwc_lwc = avg_rwc_high - avg_rwc_low
    masked_diff = np.ma.masked_where(np.isnan(diff_rwc_lwc), diff_rwc_lwc)

    abs_max = max(abs(np.nanmin(diff_rwc_lwc)), abs(np.nanmax(diff_rwc_lwc)))
    vmin, vmax = -abs_max, abs_max
    divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    gray_mask = np.isnan(diff_rwc_lwc)
    gray_values = np.full_like(diff_rwc_lwc, np.nan)
    gray_values[gray_mask] = 1

    plt.figure(figsize=(8, 6))
    img = plt.pcolormesh(xedges, yedges, masked_diff.T, cmap=plt.cm.RdBu_r, norm=divnorm, shading='auto')
    plt.pcolormesh(xedges, yedges, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)

    cbar = plt.colorbar(img)
    cbar.set_label("RWC / LWC Difference", fontsize=18, fontweight='bold')
    cbar.ax.tick_params(labelsize=16, width=2, length=5)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight('bold')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
    plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
    plt.title(f'Difference in RWC/LWC\n(Excluded: {excluded_date})', fontsize=17, fontweight='bold')
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    plt.tick_params(axis='both', which='major', width=3, length=8)
    plt.tick_params(axis='both', which='minor', width=2, length=5)
    plt.tight_layout()
    plt.show()
#%%
#bootstrapping 


def compute_rwc_lwc_ratio(concentration, lwc, rwc, x_bins, y_bins, x_min, x_max, y_min, y_max):
    mask = (concentration >= x_min) & (concentration <= x_max) & \
           (lwc >= y_min) & (lwc <= y_max)
    f_conc, f_lwc, f_rwc = concentration[mask], lwc[mask], rwc[mask]
    sum_rwc, xedges, yedges = np.histogram2d(f_conc, f_lwc, bins=[x_bins, y_bins], weights=f_rwc)
    sum_lwc, _, _ = np.histogram2d(f_conc, f_lwc, bins=[x_bins, y_bins], weights=f_lwc)
    counts, _, _ = np.histogram2d(f_conc, f_lwc, bins=[x_bins, y_bins])
    avg_rwc = np.divide(sum_rwc, counts, out=np.full_like(sum_rwc, np.nan), where=counts > 0)
    avg_lwc = np.divide(sum_lwc, counts, out=np.full_like(sum_lwc, np.nan), where=counts > 0)
    ratio = np.divide(avg_rwc, avg_lwc, out=np.full_like(avg_rwc, np.nan), where=avg_lwc > 0) * 100
    return np.ma.masked_where(np.isnan(ratio), ratio), xedges, yedges

def bootstrap_difference_mask(
    filtered_high_concentration, filtered_high_lwc, filtered_high_rwc,
    filtered_low_concentration, filtered_low_lwc, filtered_low_rwc,
    rwc_lwc_ratio_high, rwc_lwc_ratio_low,
    x_bins, y_bins,
    n_bootstrap=10000,
    confidence_level=0.90
):
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = (1 + confidence_level) / 2 * 100

    significance_mask = np.full((len(x_bins) - 1, len(y_bins) - 1), False)

    filtered_high_ratio = filtered_high_rwc / filtered_high_lwc
    filtered_low_ratio = filtered_low_rwc / filtered_low_lwc

    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            bin_high = (filtered_high_concentration >= x_bins[i]) & (filtered_high_concentration < x_bins[i+1]) & \
                       (filtered_high_lwc >= y_bins[j]) & (filtered_high_lwc < y_bins[j+1])
            bin_low = (filtered_low_concentration >= x_bins[i]) & (filtered_low_concentration < x_bins[i+1]) & \
                      (filtered_low_lwc >= y_bins[j]) & (filtered_low_lwc < y_bins[j+1])

            high_ratios = filtered_high_ratio[bin_high]
            low_ratios = filtered_low_ratio[bin_low]

            if len(high_ratios) >= 3 and len(low_ratios) >= 3:
                boot_diff = []
                for _ in range(n_bootstrap):
                    sample_high = np.random.choice(high_ratios, size=len(high_ratios), replace=True)
                    sample_low = np.random.choice(low_ratios, size=len(low_ratios), replace=True)
                    boot_diff.append(np.mean(sample_high) - np.mean(sample_low))
                boot_diff = np.array(boot_diff)
                lower = np.percentile(boot_diff, lower_percentile)
                upper = np.percentile(boot_diff, upper_percentile)
                if lower > 0 or upper < 0:
                    significance_mask[i, j] = True

    diff_rwc_lwc = rwc_lwc_ratio_high - rwc_lwc_ratio_low
    masked_diff = np.ma.masked_where(np.isnan(diff_rwc_lwc), diff_rwc_lwc)

    return significance_mask, masked_diff

# Main leave-one-out loop
unique_dates = list(average_gccn_per_flight.keys())

for excluded_date in unique_dates:
    print(f"\n=== Excluding flight: {excluded_date} ===")

    temp_avg_gccn = {
        date: avg for date, avg in average_gccn_per_flight.items() if date != excluded_date
    }

    gccn_vals_temp = np.array(list(temp_avg_gccn.values()))
    threshold = np.percentile(gccn_vals_temp, 50)

    high_dates = {date for date, val in temp_avg_gccn.items() if val >= threshold}
    low_dates = {date for date, val in temp_avg_gccn.items() if val < threshold}

    high_gccn_data = [entry for entry in total_combined_concentration if entry['Date'] in high_dates]
    low_gccn_data = [entry for entry in total_combined_concentration if entry['Date'] in low_dates]

    high_concentration = np.array([entry['Total_Combined_Concentration'] for entry in high_gccn_data])
    high_lwc = np.array([entry['Total_Liquid_Water'] for entry in total_liquid_water if entry['Date'] in high_dates])
    high_rwc = np.array([entry['RWC'] for entry in total_liquid_water if entry['Date'] in high_dates])

    low_concentration = np.array([entry['Total_Combined_Concentration'] for entry in low_gccn_data])
    low_lwc = np.array([entry['Total_Liquid_Water'] for entry in total_liquid_water if entry['Date'] in low_dates])
    low_rwc = np.array([entry['RWC'] for entry in total_liquid_water if entry['Date'] in low_dates])

    x_min, x_max = 50, 200
    y_min, y_max = 0.1, 0.3
    num_bins = 3

    x_bins = np.logspace(np.log10(x_min), np.log10(x_max), num_bins)
    y_bins = np.logspace(np.log10(y_min), np.log10(y_max), num_bins)

    masked_rwc_high, xedges, yedges = compute_rwc_lwc_ratio(high_concentration, high_lwc, high_rwc, x_bins, y_bins, x_min, x_max, y_min, y_max)
    masked_rwc_low, _, _ = compute_rwc_lwc_ratio(low_concentration, low_lwc, low_rwc, x_bins, y_bins, x_min, x_max, y_min, y_max)

    # Bootstrap significance test + difference
    significance_mask, masked_diff = bootstrap_difference_mask(
        high_concentration, high_lwc, high_rwc,
        low_concentration, low_lwc, low_rwc,
        masked_rwc_high, masked_rwc_low,
        x_bins, y_bins
    )

    # === PLOT ===
    abs_max = max(abs(np.nanmin(masked_diff)), abs(np.nanmax(masked_diff)))
    divnorm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    gray_mask = np.isnan(masked_diff)
    gray_values = np.full_like(masked_diff, np.nan)
    gray_values[gray_mask] = 1

    plt.figure(figsize=(8, 6))
    img = plt.pcolormesh(xedges, yedges, masked_diff.T, cmap=plt.cm.RdBu_r, norm=divnorm, shading='auto')
    plt.pcolormesh(xedges, yedges, gray_values.T, cmap=mcolors.ListedColormap(["gray"]), shading='auto', alpha=0.6)

    # Add hatching for significance
    plt.pcolormesh(
        xedges, yedges,
        np.ma.masked_where(~significance_mask, significance_mask).T,
        shading='flat',
        hatch='///',
        facecolors='none',
        edgecolors='black',
        linewidth=0.0
    ).set_zorder(10)

    cbar = plt.colorbar(img)
    cbar.set_label("RWC / LWC Difference (%)", fontsize=18, fontweight='bold')
    cbar.ax.tick_params(labelsize=16, width=2, length=5)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight('bold')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
    plt.ylabel(r'LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
    plt.title(f'Difference in RWC/LWC\n(Excluded: {excluded_date})', fontsize=17, fontweight='bold')
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    plt.tick_params(axis='both', which='major', width=3, length=8)
    plt.tick_params(axis='both', which='minor', width=2, length=5)
    plt.tight_layout()
    plt.show()

# %%
#trying to make histograms in each bin of all the points without bootsrapping HIGH

rwc_lwc_high = filtered_high_rwc / filtered_high_lwc * 100  # unit: %
num_bins = 3
x_bins = np.logspace(np.log10(x_min), np.log10(x_max), num_bins)
y_bins = np.logspace(np.log10(y_min), np.log10(y_max), num_bins)
raw_bin_data_high = {}

for i in range(len(x_bins)-1):
    for j in range(len(y_bins)-1):
        bin_mask = (
            (filtered_high_concentration >= x_bins[i]) & (filtered_high_concentration < x_bins[i+1]) &
            (filtered_high_lwc >= y_bins[j]) & (filtered_high_lwc < y_bins[j+1])
        )
        bin_vals = rwc_lwc_high[bin_mask]
        if len(bin_vals) > 0:
            raw_bin_data_high[(i, j)] = bin_vals
fig, axes = plt.subplots(nrows=len(x_bins)-1, ncols=len(y_bins)-1, figsize=(20, 16), sharex=True, sharey=True)
for (i, j), values in raw_bin_data_high.items():
    ax = axes[i, j] if isinstance(axes[0], np.ndarray) else axes[i * (len(y_bins)-1) + j]
    sns.histplot(values, ax=ax, bins=30, color='blue', kde=False)
    x_range = f"{x_bins[i]:.0f}-{x_bins[i+1]:.0f} cm⁻³"
    y_range = f"{y_bins[j]:.2f}-{y_bins[j+1]:.2f} g/m³"
    ax.set_title(f"Bin ({i},{j}):\nNr+Nc = {x_range}, \nLWC = {y_range}", fontsize=16, fontweight='bold')
    ax.axvline(0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel("RWC / LWC (%)", fontsize=18, fontweight='bold')
    ax.set_ylabel("Number of Legs", fontsize=18, fontweight='bold')
    ax.tick_params(axis='x', labelbottom=True)
    ax.tick_params(axis='both', labelsize=16)

# %%
#trying to make histograms in each bin of all the points without bootsrapping LOW

rwc_lwc_low = filtered_low_rwc / filtered_low_lwc * 100  # unit: %
num_bins = 3
x_bins = np.logspace(np.log10(x_min), np.log10(x_max), num_bins)
y_bins = np.logspace(np.log10(y_min), np.log10(y_max), num_bins)
raw_bin_data_low = {}
for i in range(len(x_bins)-1):
    for j in range(len(y_bins)-1):
        bin_mask = (
            (filtered_low_concentration >= x_bins[i]) & (filtered_low_concentration < x_bins[i+1]) &
            (filtered_low_lwc >= y_bins[j]) & (filtered_low_lwc < y_bins[j+1])
        )
        bin_vals = rwc_lwc_low[bin_mask]
        if len(bin_vals) > 0:
            raw_bin_data_low[(i, j)] = bin_vals

fig, axes = plt.subplots(nrows=len(x_bins)-1, ncols=len(y_bins)-1, figsize=(20, 16), sharex=True, sharey=True)
for (i, j), values in raw_bin_data_low.items():
    ax = axes[i, j] if isinstance(axes[0], np.ndarray) else axes[i * (len(y_bins)-1) + j]
    sns.histplot(values, ax=ax, bins=30, color='blue', kde=False)

    x_range = f"{x_bins[i]:.0f}-{x_bins[i+1]:.0f} cm⁻³"
    y_range = f"{y_bins[j]:.2f}-{y_bins[j+1]:.2f} g/m³"
    ax.set_title(f"Bin ({i},{j}):\nNr+Nc = {x_range}, \nLWC = {y_range}", fontsize=16, fontweight='bold')

    ax.axvline(0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel("RWC / LWC (%)", fontsize=18, fontweight='bold')
    ax.set_ylabel("Number of Legs", fontsize=18, fontweight='bold')

    
    ax.tick_params(axis='x', labelbottom=True)
    ax.tick_params(axis='both', labelsize=16)
#%%
x_min, x_max = 50, 200
y_min, y_max = 0.1, 0.3
num_bins = 3
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

       
        print(f"Flight range: Conc {conc.min():.2f}–{conc.max():.2f}, LWC {lwc.min():.3f}–{lwc.max():.3f}")

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
gccn_values = np.array(list(average_gccn_per_flight.values()))
threshold = np.percentile(gccn_values, 50)

high_dates = {date for date, val in average_gccn_per_flight.items() if val >= threshold}
low_dates = {date for date, val in average_gccn_per_flight.items() if val < threshold}

high_gccn_data = [entry for entry in total_combined_concentration if entry['Date'] in high_dates]
low_gccn_data = [entry for entry in total_combined_concentration if entry['Date'] in low_dates]

grouped_high = group_by_flight(high_gccn_data)
grouped_low = group_by_flight(low_gccn_data)
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
                    conc = np.array([e['Total_Combined_Concentration'] for e in flight])
                    lwc = np.array([e['Total_Liquid_Water'] for e in flight])
                    rwc = np.array([e['Rain_Concentration'] for e in flight])
                    ratio = np.where(lwc > 0, rwc / lwc * 100, np.nan)
                    mask = (conc >= x_bins[i]) & (conc < x_bins[i+1]) & (lwc >= y_bins[j]) & (lwc < y_bins[j+1])
                    if np.any(mask):
                        val = np.nanmean(ratio[mask])
                        ax.scatter(val, ax.get_ylim()[1] * 0.1, color='black', s=40, marker='X', label='May 10' if (i==0 and j==0) else "")
            else:
                ax.set_visible(False)

#%%
#printing the >0% values for each bin

x_min, x_max = 50, 200
y_min, y_max = 0.1, 0.3
num_bins = 3
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
#seperating RWC and LWC 
x_min, x_max = 50, 200
y_min, y_max = 0.1, 0.3
num_bins = 3
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
def compute_flight_bin_means(flight_data, var_key):
    bin_means = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    for flight in flight_data.values():
        conc = np.array([e['Total_Combined_Concentration'] for e in flight])
        lwc = np.array([e['Total_Liquid_Water'] for e in flight])
        var = np.array([e[var_key] for e in flight])

        for i in range(len(x_bins)-1):
            for j in range(len(y_bins)-1):
                mask = (conc >= x_bins[i]) & (conc < x_bins[i+1]) & \
                       (lwc >= y_bins[j]) & (lwc < y_bins[j+1])
                if np.any(mask):
                    mean_val = np.nanmean(var[mask])
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
def plot_histograms_with_percentage(boot_dists, label):
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

                annotation = f"{percent_above_zero:.1f}% > 0\nμ = {mean_val:.4f}, σ = {std_val:.4f}"

                ax.text(
                    0.98, 0.95,
                    annotation,
                    transform=ax.transAxes,
                    ha='right', va='top', fontsize=14,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
                )
                ax.set_xlabel(f"{label} Difference", fontsize=18, fontweight='bold')
                ax.set_ylabel("Count", fontsize=18, fontweight='bold')
                ax.tick_params(axis='x', labelbottom=True)
                ax.tick_params(axis='both', labelsize=16)
            else:
                ax.set_visible(False)

    plt.suptitle(f"{label} Bootstrapping Below Cloud Base January-June 2022", fontsize=19, fontweight='bold')
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
bin_high_rwc = compute_flight_bin_means(grouped_high, 'Rain_Concentration')
bin_low_rwc = compute_flight_bin_means(grouped_low, 'Rain_Concentration')
boot_rwc = bootstrap_bin_differences(bin_high_rwc, bin_low_rwc)
plot_histograms_with_percentage(boot_rwc, label="RWC")
bin_high_lwc = compute_flight_bin_means(grouped_high, 'Total_Liquid_Water')
bin_low_lwc = compute_flight_bin_means(grouped_low, 'Total_Liquid_Water')
boot_lwc = bootstrap_bin_differences(bin_high_lwc, bin_low_lwc)
plot_histograms_with_percentage(boot_lwc, label="LWC")
def print_bin_stats(boot_dists, label):
    print(f"\n=== One-sided T-test Results Per Bin for {label} (H₀: mean ≤ 0, H₁: mean > 0) ===")
    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            dist = boot_dists[i][j]
            if len(dist) > 1:
                t_stat, p_two_sided = stats.ttest_1samp(dist, popmean=0)
                p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - (p_two_sided / 2)
                percent_above_zero = np.sum(dist > 0) / len(dist) * 100
                print(
                    f"{label} Bin ({i},{j}): mean = {np.mean(dist):.2f}, std = {np.std(dist):.2f}, "
                    f"% > 0 = {percent_above_zero:.1f}%, t = {t_stat:.2f}, p = {p_one_sided:.2e}"
                )

print_bin_stats(boot_rwc, "RWC")
print_bin_stats(boot_lwc, "LWC")


#%%
#showing contribution of may 10
x_min, x_max = 50, 200
y_min, y_max = 0.1, 0.3
num_bins = 3
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
    bin_sources = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    for date, flight in flight_data.items():
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
                        bin_sources[i][j].append(date)
    return bin_means, bin_sources
def bootstrap_bin_differences_with_may10(bin_high, bin_low, src_high, src_low):
    boot_dists = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    boot_may10_flags = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]

    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            high_vals = bin_high[i][j]
            low_vals = bin_low[i][j]
            high_src = src_high[i][j]
            low_src = src_low[i][j]

            if len(high_vals) >= 2 and len(low_vals) >= 2:
                dist = []
                includes_may10 = []
                for _ in range(n_bootstrap):
                    high_idx = np.random.choice(len(high_vals), len(high_vals), replace=True)
                    low_idx = np.random.choice(len(low_vals), len(low_vals), replace=True)
                    high_sample = [high_vals[idx] for idx in high_idx]
                    low_sample = [low_vals[idx] for idx in low_idx]
                    dist.append(np.mean(high_sample) - np.mean(low_sample))
                    high_dates = [high_src[idx] for idx in high_idx]
                    low_dates = [low_src[idx] for idx in low_idx]
                    includes_may10.append('2022-05-10' in high_dates or '2022-05-10' in low_dates)

                boot_dists[i][j] = np.array(dist)
                boot_may10_flags[i][j] = np.array(includes_may10)
    return boot_dists, boot_may10_flags
def plot_histograms_may10_stacked(boot_dists, boot_includes_may10):
    fig, axes = plt.subplots(nrows=len(x_bins)-1, ncols=len(y_bins)-1, figsize=(14, 10), sharex=True, sharey=True)

    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            ax = axes[i][j]
            dist = boot_dists[i][j]
            included = boot_includes_may10[i][j]

            if len(dist) == 0:
                ax.set_visible(False)
                continue

            may10_vals = dist[included]
            other_vals = dist[~included]

            bins = np.linspace(min(dist), max(dist), 30)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            hist_other, _ = np.histogram(other_vals, bins=bins)
            hist_may10, _ = np.histogram(may10_vals, bins=bins)

            ax.bar(bin_centers, hist_other, width=np.diff(bins), color='skyblue', edgecolor='black')
            ax.bar(bin_centers, hist_may10, bottom=hist_other, width=np.diff(bins), color='orange', edgecolor='black')

            ax.axvline(0, color='red', linestyle='--')
            lower = np.percentile(dist, lower_percentile)
            upper = np.percentile(dist, upper_percentile)
            ax.axvline(lower, color='black', linestyle=':', linewidth=1)
            ax.axvline(upper, color='black', linestyle=':', linewidth=1)

            percent_above_zero = np.sum(dist > 0) / len(dist) * 100
            mean_val = np.mean(dist)
            std_val = np.std(dist)
            annotation = f"{percent_above_zero:.1f}% > 0\nμ = {mean_val:.2f}, σ = {std_val:.2f}"
            ax.text(0.98, 0.95, annotation, transform=ax.transAxes, ha='right', va='top', fontsize=12,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

            ax.set_xlabel("RWC/LWC Difference (%)", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.tick_params(axis='both', labelsize=10)
            ax.tick_params(axis='x', labelbottom=True)


    fig.suptitle("Full Bootstrapping — May 10 Highlighted \n January-June Below Cloud Base", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
gccn_values = np.array(list(average_gccn_per_flight.values()))
threshold = np.percentile(gccn_values, 50)

high_dates = {date for date, val in average_gccn_per_flight.items() if val >= threshold}
low_dates = {date for date, val in average_gccn_per_flight.items() if val < threshold}

high_gccn_data = [entry for entry in total_combined_concentration if entry['Date'] in high_dates]
low_gccn_data = [entry for entry in total_combined_concentration if entry['Date'] in low_dates]

grouped_high = group_by_flight(high_gccn_data)
grouped_low = group_by_flight(low_gccn_data)

bin_means_high, src_high = compute_flight_bin_means(grouped_high)
bin_means_low, src_low = compute_flight_bin_means(grouped_low)

boot_distributions, boot_includes_may10 = bootstrap_bin_differences_with_may10(bin_means_high, bin_means_low, src_high, src_low)

plot_histograms_may10_stacked(boot_distributions, boot_includes_may10)


#%%
#only 12 flights 
x_min, x_max = 50, 200
y_min, y_max = 0.1, 0.3
num_bins = 3
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
                    sampled_high = np.random.choice(high_vals, min(12, len(high_vals)), replace=True)
                    sampled_low = np.random.choice(low_vals, min(12, len(low_vals)), replace=True)
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

    plt.suptitle("Bootstrapping Using 12 Flights Per Iteration (Random Sample)\nBelow Cloud Base January-June 2022", fontsize=19, fontweight='bold')
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
#separating rwc and lwc 
x_min, x_max = 50, 200
y_min, y_max = 0.1, 0.3
num_bins = 3
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
def compute_flight_bin_means_quantity(flight_data, varname):
    bin_means = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    for flight in flight_data.values():
        conc = np.array([e['Total_Combined_Concentration'] for e in flight])
        lwc = np.array([e['Total_Liquid_Water'] for e in flight])
        variable = np.array([e[varname] for e in flight])

        for i in range(len(x_bins) - 1):
            for j in range(len(y_bins) - 1):
                mask = (conc >= x_bins[i]) & (conc < x_bins[i+1]) & \
                       (lwc >= y_bins[j]) & (lwc < y_bins[j+1])
                if np.any(mask):
                    mean_val = np.nanmean(variable[mask])
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
                    sampled_high = np.random.choice(high_vals, min(12, len(high_vals)), replace=True)
                    sampled_low = np.random.choice(low_vals, min(12, len(low_vals)), replace=True)
                    boot_diff.append(np.mean(sampled_high) - np.mean(sampled_low))
                boot_dists[i][j] = np.array(boot_diff)
    return boot_dists
def plot_histograms_with_percentage(boot_dists, title):
    fig, axes = plt.subplots(nrows=len(x_bins)-1, ncols=len(y_bins)-1, figsize=(14, 10), sharex=True, sharey=True)

    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            ax = axes[i][j]
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
                annotation = f"{percent_above_zero:.1f}% > 0\nμ = {mean_val:.4f}, σ = {std_val:.4f}"
                ax.text(0.98, 0.95, annotation, transform=ax.transAxes, ha='right', va='top', fontsize=14,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

                ax.set_xlabel("Difference", fontsize=18, fontweight='bold')
                ax.set_ylabel("Count", fontsize=18, fontweight='bold')
                ax.tick_params(axis='both', labelsize=16)
            else:
                ax.set_visible(False)

    plt.suptitle(title, fontsize=19, fontweight='bold')
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
bin_means_high_rwc = compute_flight_bin_means_quantity(grouped_high, 'Rain_Concentration')
bin_means_low_rwc = compute_flight_bin_means_quantity(grouped_low, 'Rain_Concentration')
boot_dist_rwc = bootstrap_bin_differences_random12(bin_means_high_rwc, bin_means_low_rwc)
plot_histograms_with_percentage(boot_dist_rwc, "Bootstrapped RWC Difference Using 12 Flights")
bin_means_high_lwc = compute_flight_bin_means_quantity(grouped_high, 'Total_Liquid_Water')
bin_means_low_lwc = compute_flight_bin_means_quantity(grouped_low, 'Total_Liquid_Water')
boot_dist_lwc = bootstrap_bin_differences_random12(bin_means_high_lwc, bin_means_low_lwc)
plot_histograms_with_percentage(boot_dist_lwc, "Bootstrapped LWC Difference Using 12 Flights")
print("\n=== One-sided T-test Results Per Bin (H₀: mean ≤ 0, H₁: mean > 0) ===")
for i in range(len(x_bins) - 1):
    for j in range(len(y_bins) - 1):
        dist = boot_dist_rwc[i][j]
        if len(dist) > 1:
            t_stat, p_two_sided = stats.ttest_1samp(dist, popmean=0)
            p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - (p_two_sided / 2)
            print(f"[RWC] Bin ({i},{j}): mean = {np.mean(dist):.2f}, std = {np.std(dist):.2f}, "
                  f"t = {t_stat:.2f}, p = {p_one_sided:.2e}")

for i in range(len(x_bins) - 1):
    for j in range(len(y_bins) - 1):
        dist = boot_dist_lwc[i][j]
        if len(dist) > 1:
            t_stat, p_two_sided = stats.ttest_1samp(dist, popmean=0)
            p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - (p_two_sided / 2)
            print(f"[LWC] Bin ({i},{j}): mean = {np.mean(dist):.2f}, std = {np.std(dist):.2f}, "
                  f"t = {t_stat:.2f}, p = {p_one_sided:.2e}")


#%%
x_min, x_max = 50, 200
y_min, y_max = 0.1, 0.3
num_bins = 3
x_bins = np.logspace(np.log10(x_min), np.log10(x_max), num_bins)
y_bins = np.logspace(np.log10(y_min), np.log10(y_max), num_bins)
n_bootstrap = 10000
confidence_level = 0.90
lower_percentile = (1 - confidence_level) / 2 * 100
upper_percentile = (1 + confidence_level) / 2 * 100
def group_by_flight(data):
    from collections import defaultdict
    flights = defaultdict(list)
    for entry in data:
        flights[entry['Date']].append(entry)
    return flights
def compute_flight_bin_means(flight_data):
    bin_means = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    bin_sources = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    for date, flight in flight_data.items():
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
                        bin_sources[i][j].append(date)
    return bin_means, bin_sources
def bootstrap_bin_differences_random12_with_may10(bin_high, bin_low, src_high, src_low):
    boot_dists = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    boot_may10_flags = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            high_vals = bin_high[i][j]
            low_vals = bin_low[i][j]
            high_src = src_high[i][j]
            low_src = src_low[i][j]
            if len(high_vals) >= 2 and len(low_vals) >= 2:
                dist = []
                includes_may10 = []
                for _ in range(n_bootstrap):
                    high_idx = np.random.choice(len(high_vals), min(12, len(high_vals)), replace=True)
                    low_idx = np.random.choice(len(low_vals), min(12, len(low_vals)), replace=True)
                    high_sample = [high_vals[idx] for idx in high_idx]
                    low_sample = [low_vals[idx] for idx in low_idx]
                    dist.append(np.mean(high_sample) - np.mean(low_sample))
                    high_dates = [high_src[idx] for idx in high_idx]
                    low_dates = [low_src[idx] for idx in low_idx]
                    includes_may10.append('2022-05-10' in high_dates or '2022-05-10' in low_dates)
                boot_dists[i][j] = np.array(dist)
                boot_may10_flags[i][j] = np.array(includes_may10)
    return boot_dists, boot_may10_flags

def plot_histograms_may10_stacked(boot_dists, boot_includes_may10):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows=len(x_bins)-1, ncols=len(y_bins)-1, figsize=(14, 10), sharex=True, sharey=True)

    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            ax = axes[i][j]
            dist = boot_dists[i][j]
            included = boot_includes_may10[i][j]

            if len(dist) == 0:
                ax.set_visible(False)
                continue

            may10_vals = dist[included]
            other_vals = dist[~included]

            bins = np.linspace(min(dist), max(dist), 30)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            hist_other, _ = np.histogram(other_vals, bins=bins)
            hist_may10, _ = np.histogram(may10_vals, bins=bins)

            ax.bar(bin_centers, hist_other, width=np.diff(bins), color='skyblue', edgecolor='black')
            ax.bar(bin_centers, hist_may10, bottom=hist_other, width=np.diff(bins), color='orange', edgecolor='black')

            ax.axvline(0, color='red', linestyle='--')
            lower = np.percentile(dist, lower_percentile)
            upper = np.percentile(dist, upper_percentile)
            ax.axvline(lower, color='black', linestyle=':', linewidth=1)
            ax.axvline(upper, color='black', linestyle=':', linewidth=1)

            percent_above_zero = np.sum(dist > 0) / len(dist) * 100
            mean_val = np.mean(dist)
            std_val = np.std(dist)
            annotation = f"{percent_above_zero:.1f}% > 0\nμ = {mean_val:.2f}, σ = {std_val:.2f}"
            ax.text(0.98, 0.95, annotation, transform=ax.transAxes, ha='right', va='top', fontsize=12,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

            ax.tick_params(axis='x', labelbottom=True) 
            ax.set_xlabel("RWC/LWC Difference (%)", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.tick_params(axis='both', labelsize=10)

    fig.suptitle("Bootstrapping Using 12 Flights Per Iteration\nMay 10 Highlighted", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

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

x_min, x_max = 50, 200
y_min, y_max = 0.1, 0.3
num_bins = 3
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
#separating RWC and LWC




# %%
# #only 2 flights 

# x_min, x_max = 50, 200
# y_min, y_max = 0.1, 0.3
# num_bins = 3
# x_bins = np.logspace(np.log10(x_min), np.log10(x_max), num_bins)
# y_bins = np.logspace(np.log10(y_min), np.log10(y_max), num_bins)
# n_bootstrap = 10000
# confidence_level = 0.90
# lower_percentile = (1 - confidence_level) / 2 * 100
# upper_percentile = (1 + confidence_level) / 2 * 100


# def group_by_flight(data):
#     flights = defaultdict(list)
#     for entry in data:
#         flights[entry['Date']].append(entry)
#     return flights


# def compute_flight_bin_means(flight_data):
#     bin_means = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
#     for flight in flight_data.values():
#         conc = np.array([e['Total_Combined_Concentration'] for e in flight])
#         lwc = np.array([e['Total_Liquid_Water'] for e in flight])
#         rwc = np.array([e['Rain_Concentration'] for e in flight])
#         ratio = np.where(lwc > 0, rwc / lwc * 100, np.nan)

#         for i in range(len(x_bins) - 1):
#             for j in range(len(y_bins) - 1):
#                 mask = (conc >= x_bins[i]) & (conc < x_bins[i+1]) & \
#                        (lwc >= y_bins[j]) & (lwc < y_bins[j+1])
#                 if np.any(mask):
#                     mean_val = np.nanmean(ratio[mask])
#                     if not np.isnan(mean_val):
#                         bin_means[i][j].append(mean_val)
#     return bin_means

# def bootstrap_bin_differences_random12(bin_high, bin_low):
#     boot_dists = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
#     for i in range(len(x_bins)-1):
#         for j in range(len(y_bins)-1):
#             high_vals = bin_high[i][j]
#             low_vals = bin_low[i][j]
#             if len(high_vals) >= 2 and len(low_vals) >= 2:
#                 boot_diff = []
#                 for _ in range(n_bootstrap):
#                     sampled_high = np.random.choice(high_vals, min(2, len(high_vals)), replace=True)
#                     sampled_low = np.random.choice(low_vals, min(2, len(low_vals)), replace=True)
#                     boot_diff.append(np.mean(sampled_high) - np.mean(sampled_low))
#                 boot_dists[i][j] = np.array(boot_diff)
#     return boot_dists

# def plot_histograms_with_percentage(boot_dists):
#     fig, axes = plt.subplots(nrows=len(x_bins)-1, ncols=len(y_bins)-1, figsize=(14, 10), sharex=True, sharey=True)

#     for i in range(len(x_bins)-1):
#         for j in range(len(y_bins)-1):
#             ax = axes[i][j] if isinstance(axes[0], np.ndarray) else axes[i*(len(y_bins)-1)+j]
#             dist = boot_dists[i][j]

#             # Inside the for-loop over i and j:
#             if len(dist) > 0:
#                 sns.histplot(dist, bins=30, kde=False, ax=ax, color='skyblue')
#                 ax.axvline(0, color='red', linestyle='--')
#                 lower = np.percentile(dist, lower_percentile)
#                 upper = np.percentile(dist, upper_percentile)
#                 ax.axvline(lower, color='black', linestyle=':', linewidth=1)
#                 ax.axvline(upper, color='black', linestyle=':', linewidth=1)

#                 percent_above_zero = np.sum(dist > 0) / len(dist) * 100
#                 mean_val = np.mean(dist)
#                 std_val = np.std(dist)

#                 # Annotate on plot
#                 annotation = f"{percent_above_zero:.1f}% > 0\nμ = {mean_val:.2f}, σ = {std_val:.2f}"
#                 ax.text(
#                     0.98, 0.95,
#                     annotation,
#                     transform=ax.transAxes,
#                     ha='right',
#                     va='top',
#                     fontsize=14,
#                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
#                 )

#                 ax.set_xlabel("RWC/LWC Difference (%)", fontsize=18, fontweight='bold')
#                 ax.set_ylabel("Count", fontsize=18, fontweight='bold')
#                 ax.tick_params(axis='both', labelsize=16)
#                 ax.tick_params(axis='x', labelbottom=True)
#   # Enable x-ticks on top


#                 ax.set_xlabel("RWC/LWC Difference (%)", fontsize=18, fontweight='bold')
#                 ax.set_ylabel("Count", fontsize=18, fontweight='bold')
#                 ax.tick_params(axis='both', labelsize=16)
#             else:
#                 ax.set_visible(False)

#     plt.suptitle("Bootstrapping Using 2 Flights Per Iteration (Random Sample)\nBelow Cloud Base January-June 2022", fontsize=19, fontweight='bold')
#     plt.tight_layout()
#     plt.show()


# gccn_values = np.array(list(average_gccn_per_flight.values()))
# threshold = np.percentile(gccn_values, 50)

# high_dates = {date for date, val in average_gccn_per_flight.items() if val >= threshold}
# low_dates = {date for date, val in average_gccn_per_flight.items() if val < threshold}

# high_gccn_data = [entry for entry in total_combined_concentration if entry['Date'] in high_dates]
# low_gccn_data = [entry for entry in total_combined_concentration if entry['Date'] in low_dates]

# grouped_high = group_by_flight(high_gccn_data)
# grouped_low = group_by_flight(low_gccn_data)

# bin_means_high = compute_flight_bin_means(grouped_high)
# bin_means_low = compute_flight_bin_means(grouped_low)

# boot_distributions = bootstrap_bin_differences_random12(bin_means_high, bin_means_low)

# plot_histograms_with_percentage(boot_distributions)

# print("\n=== One-sided T-test Results Per Bin (H₀: mean ≤ 0, H₁: mean > 0) ===")
# for i in range(len(x_bins) - 1):
#     for j in range(len(y_bins) - 1):
#         dist = boot_distributions[i][j]
#         if len(dist) > 1:
#             t_stat, p_two_sided = stats.ttest_1samp(dist, popmean=0)
#             p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - (p_two_sided / 2)
#             print(
#                 f"Bin ({i},{j}): mean = {np.mean(dist):.2f}, std = {np.std(dist):.2f}, "
#                 f"t = {t_stat:.2f}, p = {p_one_sided:.2e}"
#             )
# %%
#all flghts 

x_min, x_max = 50, 200
y_min, y_max = 0.1, 0.3
num_bins = 3
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

    plt.suptitle("Bootstrapping Using All Flights Per Iteration (Random Sample)\nBelow Cloud Base January-June 2022", fontsize=19, fontweight='bold')
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
# %%
#separately for rwc and lwc 

x_min, x_max = 50, 200
y_min, y_max = 0.1, 0.3
num_bins = 3
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
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            ax = axes[i][j]
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
                ax.text(0.98, 0.95, annotation, transform=ax.transAxes,
                        ha='right', va='top', fontsize=14,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                ax.set_xlabel("RWC Difference (g/m³)", fontsize=18, fontweight='bold')
                ax.set_ylabel("Count", fontsize=18, fontweight='bold')
                ax.tick_params(axis='both', labelsize=16)
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
x_min, x_max = 50, 200
y_min, y_max = 0.1, 0.3
num_bins = 3
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
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            ax = axes[i][j]
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
                ax.text(0.98, 0.95, annotation, transform=ax.transAxes,
                        ha='right', va='top', fontsize=14,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                ax.set_xlabel("RWC Difference (g/m³)", fontsize=18, fontweight='bold')
                ax.set_ylabel("Count", fontsize=18, fontweight='bold')
                ax.tick_params(axis='both', labelsize=16)
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

# %%
#only 12 of the flights 

x_min, x_max = 50, 200
y_min, y_max = 0.1, 0.3
num_bins = 3
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
                mask = (
                    (conc >= x_bins[i]) & (conc < x_bins[i + 1]) &
                    (lwc >= y_bins[j]) & (lwc < y_bins[j + 1])
                )
                if np.any(mask):
                    mean_val = np.nanmean(ratio[mask])
                    if not np.isnan(mean_val):
                        bin_means[i][j].append(mean_val)
    return bin_means

def bootstrap_bin_differences_random12(bin_high, bin_low):
    boot_dists = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            high_vals = bin_high[i][j]
            low_vals = bin_low[i][j]
            if len(high_vals) >= 2 and len(low_vals) >= 2:
                boot_diff = []
                for _ in range(n_bootstrap):
                    sampled_high = np.random.choice(high_vals, size=12, replace=True)
                    sampled_low = np.random.choice(low_vals, size=12, replace=True)
                    diff = np.mean(sampled_high) - np.mean(sampled_low)
                    boot_diff.append(diff)
                boot_dists[i][j] = np.array(boot_diff)
    return boot_dists

def plot_histograms_with_percentage(boot_dists):
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
                sns.histplot(dist, bins=30, kde=False, ax=ax, color='skyblue')
                ax.axvline(0, color='red', linestyle='--')
                lower = np.percentile(dist, lower_percentile)
                upper = np.percentile(dist, upper_percentile)
                ax.axvline(lower, color='black', linestyle=':', linewidth=1)
                ax.axvline(upper, color='black', linestyle=':', linewidth=1)

                percent_above_zero = np.sum(dist > 0) / len(dist) * 100
                mean_val = np.mean(dist)
                std_val = np.std(dist)

                annotation = (
                    f"{percent_above_zero:.1f}% > 0\n"
                    f"μ = {mean_val:.2f}, σ = {std_val:.2f}"
                )
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
            else:
                ax.set_visible(False)

    plt.suptitle(
        "Bootstrapping Using 12 Flights Per Iteration (Random Sample)\nBelow Cloud Base January–June 2022",
        fontsize=19,
        fontweight='bold'
    )
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

x_min, x_max = 50, 200
y_min, y_max = 0.1, 0.3
num_bins = 3
x_bins = np.logspace(np.log10(x_min), np.log10(x_max), num_bins)
y_bins = np.logspace(np.log10(y_min), np.log10(y_max), num_bins)
n_bootstrap = 10000
confidence_level = 0.90
lower_percentile = (1 - confidence_level) / 2 * 100
upper_percentile = (1 + confidence_level) / 2 * 100
def group_by_flight(data):
    flights = defaultdict(list)
    for entry in data:
        flights[entry["Date"]].append(entry)
    return flights
def compute_flight_bin_means_RWC(flight_data):
    flight_bin_means = {}
    for date, flight in flight_data.items():
        bin_means = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
        conc = np.array([e["Total_Combined_Concentration"] for e in flight])
        lwc = np.array([e["Total_Liquid_Water"] for e in flight])
        rwc = np.array([e["Rain_Concentration"] for e in flight])
        for i in range(len(x_bins) - 1):
            for j in range(len(y_bins) - 1):
                mask = (
                    (conc >= x_bins[i]) & (conc < x_bins[i + 1]) &
                    (lwc >= y_bins[j]) & (lwc < y_bins[j + 1])
                )
                if np.any(mask):
                    mean_val = np.nanmean(rwc[mask])
                    if not np.isnan(mean_val):
                        bin_means[i][j].append(mean_val)
        flight_bin_means[date] = bin_means
    return flight_bin_means
def bootstrap_bin_differences_random12(flight_bin_means_high, flight_bin_means_low):
    boot_dists = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    high_dates = list(flight_bin_means_high.keys())
    low_dates = list(flight_bin_means_low.keys())
    for _ in range(n_bootstrap):
        sampled_high_dates = np.random.choice(high_dates, size=12, replace=True)
        sampled_low_dates = np.random.choice(low_dates, size=12, replace=True)
        sampled_high_means = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
        sampled_low_means = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
        for date in sampled_high_dates:
            bins = flight_bin_means_high[date]
            for i in range(len(x_bins) - 1):
                for j in range(len(y_bins) - 1):
                    sampled_high_means[i][j].extend(bins[i][j])
        for date in sampled_low_dates:
            bins = flight_bin_means_low[date]
            for i in range(len(x_bins) - 1):
                for j in range(len(y_bins) - 1):
                    sampled_low_means[i][j].extend(bins[i][j])
        for i in range(len(x_bins) - 1):
            for j in range(len(y_bins) - 1):
                high_vals = sampled_high_means[i][j]
                low_vals = sampled_low_means[i][j]
                if len(high_vals) >= 2 and len(low_vals) >= 2:
                    mean_high = np.mean(high_vals)
                    mean_low = np.mean(low_vals)
                    boot_dists[i][j].append(mean_high - mean_low)

   
    boot_dists = [[np.array(b) if len(b) > 0 else [] for b in row] for row in boot_dists]
    return boot_dists
def plot_histograms_with_percentage(boot_dists):
    fig, axes = plt.subplots(nrows=len(x_bins)-1, ncols=len(y_bins)-1,
                             figsize=(14, 10), sharex=True, sharey=True)
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            ax = axes[i][j]
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
                ax.text(0.98, 0.95, annotation, transform=ax.transAxes,
                        ha='right', va='top', fontsize=14,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                ax.set_xlabel("RWC Difference (g/m³)", fontsize=18, fontweight='bold')
                ax.set_ylabel("Count", fontsize=18, fontweight='bold')
                ax.tick_params(axis='both', labelsize=16)
            else:
                ax.set_visible(False)
    plt.suptitle("Bootstrapped RWC Differences (12 Flights/Iter)", fontsize=19, fontweight='bold')
    plt.tight_layout()
    plt.show()
gccn_values = np.array(list(average_gccn_per_flight.values()))
threshold = np.percentile(gccn_values, 50)
high_dates = {date for date, val in average_gccn_per_flight.items() if val >= threshold}
low_dates = {date for date, val in average_gccn_per_flight.items() if val < threshold}
high_data = [entry for entry in total_combined_concentration if entry["Date"] in high_dates]
low_data = [entry for entry in total_combined_concentration if entry["Date"] in low_dates]

grouped_high = group_by_flight(high_data)
grouped_low = group_by_flight(low_data)
flight_bin_means_high = compute_flight_bin_means_RWC(grouped_high)
flight_bin_means_low = compute_flight_bin_means_RWC(grouped_low)
boot_distributions = bootstrap_bin_differences_random12(flight_bin_means_high, flight_bin_means_low)
plot_histograms_with_percentage(boot_distributions)
print("\n=== One-sided T-test Results Per Bin (H₀: mean ≤ 0, H₁: mean > 0) ===")
for i in range(len(x_bins)-1):
    for j in range(len(y_bins)-1):
        dist = boot_distributions[i][j]
        if len(dist) > 1:
            t_stat, p_two_sided = stats.ttest_1samp(dist, popmean=0)
            p_one_sided = p_two_sided/2 if t_stat > 0 else 1 - (p_two_sided/2)
            print(f"Bin ({i},{j}): mean = {np.mean(dist):.2f}, std = {np.std(dist):.2f}, "
                  f"t = {t_stat:.2f}, p = {p_one_sided:.2e}")

# %%

x_min, x_max = 50, 200
y_min, y_max = 0.1, 0.3
num_bins = 3
x_bins = np.logspace(np.log10(x_min), np.log10(x_max), num_bins)
y_bins = np.logspace(np.log10(y_min), np.log10(y_max), num_bins)
n_bootstrap = 10000
confidence_level = 0.90
lower_percentile = (1 - confidence_level) / 2 * 100
upper_percentile = (1 + confidence_level) / 2 * 100
def group_by_flight(data):
    flights = defaultdict(list)
    for entry in data:
        flights[entry["Date"]].append(entry)
    return flights
def compute_flight_bin_means_LWC(flight_data):
    flight_bin_means = {}
    for date, flight in flight_data.items():
        bin_means = [[[] for _ in range(len(y_bins)-1)] for _ in range(len(x_bins)-1)]
        conc = np.array([e["Total_Combined_Concentration"] for e in flight])
        lwc = np.array([e["Total_Liquid_Water"] for e in flight])
        for i in range(len(x_bins)-1):
            for j in range(len(y_bins)-1):
                mask = (
                    (conc >= x_bins[i]) & (conc < x_bins[i+1]) &
                    (lwc >= y_bins[j]) & (lwc < y_bins[j+1])
                )
                if np.any(mask):
                    mean_val = np.nanmean(lwc[mask])
                    if not np.isnan(mean_val):
                        bin_means[i][j].append(mean_val)
        flight_bin_means[date] = bin_means
    return flight_bin_means
def bootstrap_bin_differences_random12(flight_bin_means_high, flight_bin_means_low):
    boot_dists = [[[] for _ in range(len(y_bins)-1)] for _ in range(len(x_bins)-1)]
    high_dates = list(flight_bin_means_high.keys())
    low_dates = list(flight_bin_means_low.keys())
    for _ in range(n_bootstrap):
        sampled_high_dates = np.random.choice(high_dates, size=12, replace=True)
        sampled_low_dates = np.random.choice(low_dates, size=12, replace=True)
        sampled_high_means = [[[] for _ in range(len(y_bins)-1)] for _ in range(len(x_bins)-1)]
        sampled_low_means = [[[] for _ in range(len(y_bins)-1)] for _ in range(len(x_bins)-1)]
        for date in sampled_high_dates:
            bins = flight_bin_means_high[date]
            for i in range(len(x_bins)-1):
                for j in range(len(y_bins)-1):
                    sampled_high_means[i][j].extend(bins[i][j])
        for date in sampled_low_dates:
            bins = flight_bin_means_low[date]
            for i in range(len(x_bins)-1):
                for j in range(len(y_bins)-1):
                    sampled_low_means[i][j].extend(bins[i][j])

        for i in range(len(x_bins)-1):
            for j in range(len(y_bins)-1):
                high_vals = sampled_high_means[i][j]
                low_vals = sampled_low_means[i][j]
                if len(high_vals) >= 2 and len(low_vals) >= 2:
                    mean_high = np.mean(high_vals)
                    mean_low = np.mean(low_vals)
                    boot_dists[i][j].append(mean_high - mean_low)

    boot_dists = [[np.array(b) if len(b)>0 else [] for b in row] for row in boot_dists]
    return boot_dists

def plot_histograms_with_percentage(boot_dists):
    fig, axes = plt.subplots(nrows=len(x_bins)-1, ncols=len(y_bins)-1,
                             figsize=(14, 10), sharex=True, sharey=True)
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            ax = axes[i][j]
            dist = boot_dists[i][j]
            if len(dist) > 0:
                sns.histplot(dist, bins=30, kde=False, ax=ax, color="skyblue")
                ax.axvline(0, color="red", linestyle="--")
                lower = np.percentile(dist, lower_percentile)
                upper = np.percentile(dist, upper_percentile)
                ax.axvline(lower, color="black", linestyle=":", linewidth=1)
                ax.axvline(upper, color="black", linestyle=":", linewidth=1)
                percent_above_zero = np.sum(dist > 0) / len(dist) * 100
                mean_val = np.mean(dist)
                std_val = np.std(dist)
                annotation = f"{percent_above_zero:.1f}% > 0\nμ = {mean_val:.2f}, σ = {std_val:.2f}"
                ax.text(0.98, 0.95, annotation, transform=ax.transAxes,
                        ha="right", va="top", fontsize=14,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
                ax.set_xlabel("LWC Difference (g/m³)", fontsize=18, fontweight="bold")
                ax.set_ylabel("Count", fontsize=18, fontweight="bold")
                ax.tick_params(axis="both", labelsize=16)
            else:
                ax.set_visible(False)
    plt.suptitle("Bootstrapped LWC Differences (12 Flights/Iter)", fontsize=19, fontweight="bold")
    plt.tight_layout()
    plt.show()
gccn_values = np.array(list(average_gccn_per_flight.values()))
threshold = np.percentile(gccn_values, 50)
high_dates = {date for date, val in average_gccn_per_flight.items() if val >= threshold}
low_dates = {date for date, val in average_gccn_per_flight.items() if val < threshold}
high_data = [entry for entry in total_combined_concentration if entry["Date"] in high_dates]
low_data = [entry for entry in total_combined_concentration if entry["Date"] in low_dates]

grouped_high = group_by_flight(high_data)
grouped_low = group_by_flight(low_data)
flight_bin_means_high = compute_flight_bin_means_LWC(grouped_high)
flight_bin_means_low = compute_flight_bin_means_LWC(grouped_low)
boot_distributions = bootstrap_bin_differences_random12(flight_bin_means_high, flight_bin_means_low)
plot_histograms_with_percentage(boot_distributions)
print("\n=== One-sided T-test Results Per Bin (H₀: mean ≤ 0, H₁: mean > 0) ===")
for i in range(len(x_bins)-1):
    for j in range(len(y_bins)-1):
        dist = boot_distributions[i][j]
        if len(dist) > 1:
            t_stat, p_two_sided = stats.ttest_1samp(dist, popmean=0)
            p_one_sided = p_two_sided/2 if t_stat > 0 else 1 - (p_two_sided/2)
            print(f"Bin ({i},{j}): mean = {np.mean(dist):.2f}, std = {np.std(dist):.2f}, "
                  f"t = {t_stat:.2f}, p = {p_one_sided:.2e}")

# %%
