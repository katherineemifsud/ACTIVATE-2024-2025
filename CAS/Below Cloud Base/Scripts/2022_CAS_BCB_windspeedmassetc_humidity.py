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

    dataset = {'Date': date, 'Clear Means': [], 'Cloud Means': []} 
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
#We need to search for our flight legs across the CAS, the 2DS, and the leg files. 
#Then we need to move through every second of every flight leg and filter as cloudy or clear sky. 

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
    rh_flight = h20[i]

    
    CAS_flight['Time_mid'] = pd.to_numeric(CAS_flight['Time_mid'], errors='coerce')
    twoDS_flight['Time_Start'] = pd.to_numeric(twoDS_flight['Time_Start'], errors='coerce')
    rh_flight['Time_Start'] = pd.to_numeric(rh_flight['Time_Start'], errors='coerce')
    
    CAS_times = CAS_flight['Time_mid'].values
    CAS_lwc = CAS_flight['LWC_CAS'].values
    CAS_bins = {f'CAS_Bin{bin_label:02d}': CAS_flight[f'CAS_Bin{bin_label:02d}'].values for bin_label in range(12, 30)}

    TwoDS_times = twoDS_flight['Time_Start'].values
    TwoDS_N_total = twoDS_flight['N-total_2DS'].values

    rh_times = rh_flight['Time_Start'].values
    rh_values = rh_flight.RHw_DLH.values

    total_BCB_means = []

    for k in range(len(BCB_start)):
        start20 = BCB_start[k]
        end20 = BCB_stop[k]

        CAS_indices_in_range = np.where((CAS_times >= start20) & (CAS_times <= end20))[0]
        TwoDS_indices_in_range = np.where((TwoDS_times >= start20) & (TwoDS_times <= end20))[0]
        rh_indices_in_range = np.where((rh_times >= start20) & (rh_times <= end20))[0]
    
        if CAS_indices_in_range.size > 0 and TwoDS_indices_in_range.size > 0 and rh_indices_in_range.size > 0:
            data_labels = []
            BCB_means = []

            for CAS_idx, TwoDS_idx, rh_idx in zip(CAS_indices_in_range, TwoDS_indices_in_range, rh_indices_in_range):
                lwc_val = CAS_lwc[CAS_idx]
                N_val = TwoDS_N_total[TwoDS_idx]
                rh_val = rh_values[rh_idx]

                
                lwc_label = 'Y' if 0 <= lwc_val <= 0.0025 else 'N'
                N_label = 'Y' if 0 <= N_val <= 100 else 'N'
                rh_label = 'Y' if 0 <= rh_val <= 95 else 'N'
                label = 'Y' if lwc_label == 'Y' and N_label == 'Y' and rh_label == 'Y' else 'N'
                if label == 'Y' and rh_val > 95:
                    print(f"❗ RH violation: {rh_val:.2f} passed at time {CAS_times[CAS_idx]}")

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
#Double check the number of legs associated with each date to compare across multiple instruments.  
leg_count = Counter([leg['Date'] for leg in leg_info])
print("Number of legs associated with each date:")
total_legs = 0
for date, count in sorted(leg_count.items()):
    print(f"Date: {date}, Number of Legs: {count}")
    total_legs += count
print(f"\nTotal number of legs: {total_legs}")
#%%
rh_Y_values = []
for leg in leg_info:
    date = leg['Date']
    start = leg['BCB_start']
    stop = leg['BCB_stop']
    
    flight_index = dates_legs.index(date)
    rh_flight = h20[flight_index]
    rh_times = rh_flight['Time_Start'].values
    rh_vals = rh_flight['RHw_DLH'].values

    rh_leg_indices = np.where((rh_times >= start) & (rh_times <= stop))[0]
    rh_leg_vals = rh_vals[rh_leg_indices]
    labels = leg['Data_Labels']
    for idx, label in enumerate(labels):
        if label == 'Y' and idx < len(rh_leg_vals):
            rh_Y_values.append(rh_leg_vals[idx])
plt.hist(rh_Y_values, bins=30, color='teal', edgecolor='black')
plt.axvline(95, color='red', linestyle='--', label='RH = 95% threshold')
plt.xlabel("RH (%)")
plt.ylabel("Count")
plt.title("RH Values for Seconds Labeled 'Y'")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

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
    rh_flight = h20[i]

    CAS_times = np.array(CAS_flight['Time_mid'], dtype=float)
    TwoDS_times = np.array(twoDS_flight['Time_Start'], dtype=float)
    rh_times = np.array(rh_flight['Time_Start'], dtype=float)

    lwc = np.array(CAS_flight['LWC_CAS'], dtype=float)
    N_total = np.array(twoDS_flight['N-total_2DS'], dtype=float)
    rh_total = np.array(rh_flight['RHw_DLH'], dtype=float)

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
        rh_indices_in_range = np.where((rh_times >= start20) & (rh_times <= end20))[0]

        if CAS_indices_in_range.size > 0 and TwoDS_indices_in_range.size > 0 and rh_indices_in_range.size >0:
            for cas_idx, twods_idx, rh_idx in zip(CAS_indices_in_range, TwoDS_indices_in_range, rh_indices_in_range):
                lwc_val = lwc[cas_idx]
                N_val = N_total[twods_idx]
                rh_val = rh_total[rh_idx]
                lwc_label = 'Y' if 0 <= lwc_val <= 0.0025 else 'N'
                N_label = 'Y' if 0 <= N_val <= 100 else 'N'
                rh_label = 'Y' if 0 <= rh_val <= 95 else 'N'
                label = 'Y' if lwc_label == 'Y' and N_label == 'Y' and rh_label == 'Y' else 'N'

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
#average distribution
sum_bin_means = np.zeros(len(bin_center))
count_bin_means = np.zeros(len(bin_center))
for entry in Y_BCB_calc:
    bin_means = np.array([entry.get(f'Bin{i}_Y_mean', np.nan) for i in range(12, 30)], dtype=float)

    valid_indices = (bin_means > 0) & ~np.isnan(bin_means)

    sum_bin_means[valid_indices] += bin_means[valid_indices]
    count_bin_means[valid_indices] += 1

average_bin_means = np.divide(sum_bin_means, count_bin_means, where=count_bin_means > 0)
plt.figure(figsize=(8, 6))
plt.plot(bin_center, average_bin_means, color='red', linewidth=2, label='Average Size Distribution')
plt.xlabel("Deliquesced Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"CAS Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=19, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xlim(0, 45)
plt.xticks(fontweight="bold", fontsize=18)
plt.yticks(fontweight="bold", fontsize=18)
plt.title("CAS Average Ambient \nBelow Cloud Base Size Distribution\n January - June 2022", fontsize=19, fontweight="bold")
plt.show()
#%%
#Fitting an exponential to each size distribution
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)
ambient_fits = []
plt.figure(figsize=(8, 6))
for entry in Y_BCB_calc:
    bin_means = np.array([entry.get(f'Bin{i}_Y_mean', np.nan) for i in range(12, 30)], dtype=float)
    valid_indices = ~np.isnan(bin_means)
    bin_centers_valid = np.array(bin_center)[valid_indices]
    bin_means_valid = bin_means[valid_indices]

  
    if not valid_indices.any():
        print(f"No valid data for date {entry['Date']}")
        continue

    try:
        popt, _ = curve_fit(exponential, bin_centers_valid, bin_means_valid, p0=(1, 1), maxfev=5000)
        n0, D = popt

        ambient_fits.append({
            'Date': entry['Date'],
            'BCB_start': entry.get('BCB_start', np.nan),
            'BCB_stop': entry.get('BCB_stop', np.nan),
            'Intercept_n0': n0,
            'E_folding_D': D
        })

        x_fit = np.linspace(min(bin_centers_valid), max(bin_centers_valid), 100)
        y_fit = exponential(x_fit, *popt)

        plt.plot(x_fit, y_fit, color='black', alpha=0.2)

    except RuntimeError:
        print(f"Fit could not be performed for date {entry['Date']}")
plt.xlabel("Deliquesed Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"CAS Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base January - June 2022\n Exponential Fit to Ambient Size Distributions", fontsize=14, fontweight="bold")
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
    valid_indices = (bin_center <= 10) & ~np.isnan(bin_means) 
    bin_centers_valid = np.array(bin_center)[valid_indices]
    bin_means_valid = bin_means[valid_indices]

    if valid_indices.any():
        try:
            popt, _ = curve_fit(exponential, bin_centers_valid, bin_means_valid, 
                                p0=(1, 1), maxfev=5000, 
                                bounds=([0, 0.1], [np.inf, 20])) 
            n0, D = popt

            if D > 15:
                print(f"⚠️ High slope detected! Date: {entry['Date']}, D: {D:.2f}")

        except RuntimeError:
            print(f"❌ Fit failed for date {entry['Date']}")
            n0, D = np.nan, np.nan 
    else:
        n0, D = np.nan, np.nan 


    ambient_fits_10.append({
        'Date': entry['Date'],
        'BCB_start': entry.get('BCB_start', np.nan),
        'BCB_stop': entry.get('BCB_stop', np.nan),
        'Intercept_n0': n0,
        'E_folding_D': D
    })

    if not np.isnan(n0) and not np.isnan(D):
        x_fit = np.linspace(min(bin_centers_valid), 10, 100)
        y_fit = exponential(x_fit, *popt)
        plt.plot(x_fit, y_fit, color='black', alpha=0.2)
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

ambient_slope_10 = []
for fit in ambient_fits_10:
    if 'E_folding_D' in fit and not np.isnan(fit['E_folding_D']):
        ambient_slope_10.append(fit['E_folding_D'])
plt.figure(figsize=(8, 6))
plt.hist(ambient_slope_10, bins=20, color='blue', alpha=0.7)
plt.xlabel('Slope (um)', fontsize=14, fontweight="bold")
plt.ylabel('Frequency of flight legs', fontsize=14, fontweight="bold")
plt.title('Fitted Ambient Size Distributions (≤10 µm)', fontsize=14, fontweight="bold")
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.show()
#%%
# #building an ambient fit csv for Jason 

# target_date = "2022-02-15"
# rows = []
# for i, entry in enumerate([e for e in Y_BCB_calc if e["Date"] == target_date]):
#     start = entry["BCB_start"]
#     stop = entry["BCB_stop"]
#     row_dist = {
#         "Date": target_date,
#         "Leg_start (min)": start,
#         "Leg_stop (min)": stop,
#         "Type": "Ambient Distribution",
#         "Intercept_n0 (cm^-3 µm^-1)": np.nan,
#         "E_folding_D (µm)": np.nan,
#     }
#     for j, D in enumerate(bin_center[12:30]):
#         conc = entry.get(f"Bin{j+12}_Y_mean", np.nan)
#         row_dist[f"Conc_at_{D:.2f}µm (cm^-3 µm^-1)"] = conc
#     rows.append(row_dist)
#     if i < len(ambient_fits):
#         fit_full = ambient_fits[i]
#         row_fit_full = {
#             "Date": target_date,
#             "Leg_start (min)": start,
#             "Leg_stop (min)": stop,
#             "Type": "Full Exponential Fit",
#             "Intercept_n0 (cm^-3 µm^-1)": fit_full.get("Intercept_n0", np.nan),
#             "E_folding_D (µm)": fit_full.get("E_folding_D", np.nan),
#         }
#         rows.append(row_fit_full)
#     if i < len(ambient_fits_10):
#         fit_10 = ambient_fits_10[i]
#         row_fit_10 = {
#             "Date": target_date,
#             "Leg_start (min)": start,
#             "Leg_stop (min)": stop,
#             "Type": "≤10 µm Exponential Fit",
#             "Intercept_n0 (cm^-3 µm^-1)": fit_10.get("Intercept_n0", np.nan),
#             "E_folding_D (µm)": fit_10.get("E_folding_D", np.nan),
#         }
#         rows.append(row_fit_10)
# df = pd.DataFrame(rows)

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
#save total concentration to csv
# save_dir = "/home/disk/eos4/kathem24/activate/data/CAS"
# os.makedirs(save_dir, exist_ok=True)   # ensures directory exists
# save_path = os.path.join(save_dir, "total_Y_concentration_cm3.csv")
# total_concentration_df = pd.DataFrame(total_concentration_cm3)
# total_concentration_df.to_csv(save_path, index=False)

# print(f"Saved to: {save_path}")


#%%
#making a PDF of the total number concentrations for the legs labeled 'Y' across all flights. This will help us understand the distribution of total number concentrations below cloud base during the study period.
total_Y_concentrations = [entry['Total_Y_Concentration_cm3'] for entry in total_concentration_cm3 if not np.isnan(entry['Total_Y_Concentration_cm3']    
)]
plt.figure(figsize=(8, 6))
sns.histplot(total_Y_concentrations, bins=20, kde=True, color='purple', edgecolor='black', alpha=0.7)
plt.xlabel('Total Number Concentration (cm⁻³)', fontsize=14, fontweight="bold")
plt.ylabel('Frequency of Flight Legs', fontsize=14, fontweight="bold")
plt.title('Distribution of Total Number Concentrations Below Cloud Base\n for Legs Labeled "Y" (January - June 2022)', fontsize=14, fontweight="bold")
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.xlim(0, max(total_Y_concentrations) * 1.1)
plt.grid(True)
plt.tight_layout()
plt.show()
print(f"Total number of legs with valid total Y concentration: {len(total_Y_concentrations)}")
print(f"Mean Total Number Concentration: {mean_total_concentration:.2f} cm⁻³")
print(f"Median Total Number Concentration: {np.median(total_Y_concentrations):.2f} cm⁻³")
print(f"Standard Deviation of Total Number Concentration: {np.std(total_Y_concentrations):.2f} cm⁻³")
print(f"Minimum Total Number Concentration: {np.min(total_Y_concentrations):.2f} cm⁻³")
print(f"Maximum Total Number Concentration: {np.max(total_Y_concentrations):.2f} cm⁻³")
print(f"25th Percentile of Total Number Concentration: {np.percentile(total_Y_concentrations, 25):.2f} cm⁻³")
print(f"75th Percentile of Total Number Concentration: {np.percentile(total_Y_concentrations, 75):.2f} cm⁻³")
print(f"Interquartile Range of Total Number Concentration: {np.percentile(total_Y_concentrations, 75) - np.percentile(total_Y_concentrations, 25):.2f} cm⁻³")   
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
            rh9 = rh9[(rh9 <= 95) & (rh9 > 0)]  # filter for RH ≤ 95 and ignore missing/bad (-999 or 0)
            rh9_mean = np.nanmean(rh9)


        rh_times['Rh_mean'].append(rh9_mean)
        all_BCB.append(rh_times)  # List that contains all the BCB wind/alt mean dictionaries for 1 flight

    master_BCB_RH.append(all_BCB)
for flight in master_BCB_RH:
    for leg in flight:
        rh_mean_list = leg['Rh_mean']
        leg['Rh_mean'] = [np.nan if value <=0 else value for value in rh_mean_list]

#%%
date_leg_set = set()

for entry in Y_BCB_calc: 
    date = entry['Date']
    BCB_start = entry.get('BCB_start', np.nan)
    BCB_stop = entry.get('BCB_stop', np.nan)
    date_leg_set.add((date, BCB_start, BCB_stop))

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
for flight in master_BCB_RH:
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
for flight in filtered_master_BCB_RH:
    flight_gRH = []
    
    for leg in flight:
        new_leg = leg.copy() 
        
        rh_mean = new_leg['Rh_mean'][0] / 100.0  # Convert percentage to a decimal
        
        if np.isnan(rh_mean) or rh_mean >= 1:
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
ambient_fits_dict_10 = {(fit['Date'], fit['BCB_start'], fit['BCB_stop']): fit for fit in ambient_fits_10}
filtered_master_BCB_interceptdry_dict = {}
if isinstance(filtered_master_BCB_gRH[0], list):
    filtered_master_BCB_gRH = [item for sublist in filtered_master_BCB_gRH for item in sublist]
for entry in filtered_master_BCB_gRH:
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    gRh_mean = entry['gRh_mean'][0]  
    Rh_mean = entry['Rh_mean'][0] 
    if Rh_mean < 0:
        continue

    key = (date, BCB_start, BCB_stop)
    if key in ambient_fits_dict_10:
        n0 = ambient_fits_dict_10[key]['Intercept_n0'] 
        
        dryintercept = n0 / gRh_mean if gRh_mean > 0 else np.nan

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
plt.figure(figsize=(8, 6))
plt.hist(ambient_intercept_values, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel(r"$\mathbf{Ambient\ Intercept\ (cm^{-3}\ \mu m^{-1})}$", fontsize=15)
plt.ylabel('Frequency', fontsize=15, fontweight='bold')
plt.title('Histogram of Ambient Intercepts (N0)', fontsize=16, fontweight='bold')
plt.grid(True)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.show()
# %%
filtered_master_BCB_ddry = []
for entry in filtered_master_BCB_gRH:
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    gRh_mean = entry['gRh_mean'][0] 

   
    if gRh_mean > 0:
        ddry_values = np.array([D_amb / gRh_mean for D_amb in bin_center])
    else:
        ddry_values = np.full(len(bin_center), np.nan)
        print(f"Skipping division for {date}, {BCB_start}-{BCB_stop} due to invalid gRh_mean.")

    ddry_bin_widths = np.diff(ddry_values, append=np.nan) 

    raw_concentrations = next(
        (leg for leg in Y_BCB_calc if leg['Date'] == date and leg['BCB_start'] == BCB_start and leg['BCB_stop'] == BCB_stop),
        None
    )

    if raw_concentrations:
        dN_dD_ambient = np.array([raw_concentrations.get(f'Bin{i}_Y_mean', np.nan) for i in range(12, 30)], dtype=float)

        dN_dD_dry = np.where(
        (~np.isnan(dN_dD_ambient)) & (~np.isnan(ddry_bin_widths)) & (gRh_mean > 0),
        dN_dD_ambient * (np.array(bin_center) / ddry_values) * (np.diff(bin_center, append=np.nan) / ddry_bin_widths),
        np.nan
    )

    else:
        dN_dD_dry = np.full(len(bin_center), np.nan)
        print(f"Missing raw size distribution for {date}, {BCB_start}-{BCB_stop}")


    filtered_master_BCB_ddry.append({
        'Date': date,
        'BCB_start': BCB_start,
        'BCB_stop': BCB_stop,
        'ddry': ddry_values.tolist(),
        'dN/dDdry': dN_dD_dry.tolist(),
        'ddry_bin_widths': ddry_bin_widths.tolist(), 
        'gRh_mean': gRh_mean
    })

print(f"Length of filtered_master_BCB_ddry: {len(filtered_master_BCB_ddry)}")
#%%

from scipy.interpolate import interp1d
common_bins = np.linspace(2, 25, 35)
plt.figure(figsize=(8, 6))
for entry in filtered_master_BCB_ddry:
    ddry_values = np.array(entry['ddry']) 
    dN_dD_dry = np.array(entry['dN/dDdry'])
    valid_indices = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry)
    if np.sum(valid_indices) < 2:
        continue  

    interp_func = interp1d(ddry_values[valid_indices], dN_dD_dry[valid_indices], 
                           kind='linear', bounds_error=False, fill_value=np.nan)
    interpolated_dN_dD_dry = interp_func(common_bins)
    plt.plot(common_bins, interpolated_dN_dD_dry, color='black', alpha=0.2)

plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base January - June 2022\n Raw Dry Size Distributions", fontsize=14, fontweight="bold")
plt.show()
#%%%
#Removing the 0s
common_bins = np.linspace(2, 25, 35) 
plt.figure(figsize=(8, 6))
for entry in filtered_master_BCB_ddry:
    ddry_values = np.array(entry['ddry'])  
    dN_dD_dry = np.array(entry['dN/dDdry']) 
   
    valid_indices = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry)
    if np.sum(valid_indices) < 2:
        continue 

   
    interp_func = interp1d(ddry_values[valid_indices], dN_dD_dry[valid_indices], 
                           kind='linear', bounds_error=False, fill_value=np.nan)
    interpolated_dN_dD_dry = interp_func(common_bins)

    valid_interpolated_indices = (interpolated_dN_dD_dry > 0) & ~np.isnan(interpolated_dN_dD_dry)
    filtered_bins = common_bins[valid_interpolated_indices]
    filtered_dN_dD_dry = interpolated_dN_dD_dry[valid_interpolated_indices]

    if len(filtered_bins) > 0: 
        plt.plot(filtered_bins, filtered_dN_dD_dry, color='purple', alpha=0.2)

plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=20, fontweight="bold")
plt.ylabel("CAS Number Concentration\n(cm$^{-3}$ μm$^{-1}$)", fontsize=20, fontweight="bold")
plt.yscale("log")
plt.xticks(fontweight="bold", fontsize=20)
plt.yticks(fontweight="bold", fontsize=20)
plt.ylim(10**-7, 10**1.5)
plt.xlim(0.5, 40)
plt.title("CAS Below Cloud Base\n January-June 2022\n Raw Dry Size Distributions", fontsize=20, fontweight="bold")
plt.show()
#%%
#Distributions as a heatmap

from matplotlib.colors import Normalize
common_bins = np.linspace(2, 25, 200)
all_interp_distributions = []

for entry in filtered_master_BCB_ddry:
    ddry = np.array(entry['ddry'])
    dist = np.array(entry['dN/dDdry'])

    valid = ~np.isnan(ddry) & ~np.isnan(dist)
    if np.sum(valid) < 2:
        continue

    interp = interp1d(ddry[valid], dist[valid], kind='linear', bounds_error=False, fill_value=np.nan)
    interpolated = interp(common_bins)
    interpolated[(interpolated <= 0) | np.isnan(interpolated)] = np.nan
    all_interp_distributions.append(interpolated)

y_matrix = np.array(all_interp_distributions)
y_matrix[np.isnan(y_matrix)] = 0

y_bins_log = np.logspace(-7, 1.5, 150)

H, xedges, yedges = np.histogram2d(
    np.repeat(common_bins, y_matrix.shape[0]),
    y_matrix.T.flatten(),
    bins=[common_bins, y_bins_log]
)
H = H / y_matrix.shape[0]
H[H == 0] = np.nan
plt.figure(figsize=(9, 6))
plt.pcolormesh(xedges, yedges, H.T, shading='auto', cmap='viridis')

plt.yscale("log")
plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=20, fontweight="bold")
plt.ylabel("CAS Number Concentration\n(cm$^{-3}$ μm$^{-1}$)", fontsize=20, fontweight="bold")
plt.xticks(fontsize=20, fontweight="bold")
plt.yticks(fontsize=20, fontweight="bold")
plt.ylim(1e-7, 10**1.5)
plt.xlim(0, 25)
plt.title("CAS Below Cloud Base\nJanuary–June 2022\nRaw Dry Size Distributions", fontsize=20, fontweight="bold")

cbar = plt.colorbar()
cbar.set_label("Fraction of Legs", fontsize=20)
cbar.ax.tick_params(labelsize=18)
cbar.set_ticks([1e-2, 1e-1, 1e0])
cbar.set_ticklabels(['$10^{-2}$', '$10^{-1}$', '$10^{0}$'])

plt.tight_layout()
plt.show()
#%%
#fixing the color scale to fading 
from matplotlib.colors import LinearSegmentedColormap
base_cmap = plt.cm.viridis
colors = base_cmap(np.linspace(0, 1, 256))
colors[:50] = np.linspace([1, 1, 1, 1], colors[50], 50) 
fading_viridis = LinearSegmentedColormap.from_list("fading_viridis", colors)
common_bins = np.linspace(2, 25, 200)
all_interp_distributions = []

for entry in filtered_master_BCB_ddry:
    ddry = np.array(entry['ddry'])
    dist = np.array(entry['dN/dDdry'])

    valid = ~np.isnan(ddry) & ~np.isnan(dist)
    if np.sum(valid) < 2:
        continue

    interp = interp1d(ddry[valid], dist[valid], kind='linear', bounds_error=False, fill_value=np.nan)
    interpolated = interp(common_bins)
    interpolated[(interpolated <= 0) | np.isnan(interpolated)] = np.nan
    all_interp_distributions.append(interpolated)

y_matrix = np.array(all_interp_distributions)
y_matrix[np.isnan(y_matrix)] = 0

y_bins_log = np.logspace(-7, 1.5, 150)

H, xedges, yedges = np.histogram2d(
    np.repeat(common_bins, y_matrix.shape[0]),
    y_matrix.T.flatten(),
    bins=[common_bins, y_bins_log]
)
H = H / y_matrix.shape[0]
H[H == 0] = np.nan
plt.figure(figsize=(9, 6))
plt.pcolormesh(xedges, yedges, H.T, shading='auto', cmap=fading_viridis)
plt.contour(xedges[:-1], yedges[:-1], H.T, levels=5, colors='black', linewidths=0.5)
plt.yscale("log")
plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=20, fontweight="bold")
plt.ylabel("CAS Number Concentration\n(cm$^{-3}$ μm$^{-1}$)", fontsize=20, fontweight="bold")
plt.xticks(fontsize=20, fontweight="bold")
plt.yticks(fontsize=20, fontweight="bold")
plt.ylim(1e-7, 10**1.5)
plt.xlim(0, 25)
plt.title("CAS Below Cloud Base\nJanuary–June 2022\nRaw Dry Size Distributions", fontsize=20, fontweight="bold")
cbar = plt.colorbar()
cbar.set_label("Fraction of Legs", fontsize=20)
cbar.ax.tick_params(labelsize=18)
cbar.set_ticks([1e-2, 1e-1, 1e0])
cbar.set_ticklabels(['$10^{-2}$', '$10^{-1}$', '$10^{0}$'])
plt.tight_layout()
plt.show()
#%%
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy.ma as ma
#%%
base_cmap = plt.cm.viridis
colors = base_cmap(np.linspace(0, 1, 256))
colors[:80] = np.linspace([1, 1, 1, 1], colors[80], 80)
fading_viridis = LinearSegmentedColormap.from_list("fading_viridis", colors)
common_bins = np.linspace(2, 25, 200)
all_interp_distributions = []

for entry in filtered_master_BCB_ddry:
    ddry = np.array(entry['ddry'])
    dist = np.array(entry['dN/dDdry'])

    valid = ~np.isnan(ddry) & ~np.isnan(dist)
    if np.sum(valid) < 2:
        continue

    interp = interp1d(ddry[valid], dist[valid], kind='linear', bounds_error=False, fill_value=np.nan)
    interpolated = interp(common_bins)
    interpolated[(interpolated <= 0) | np.isnan(interpolated)] = np.nan
    all_interp_distributions.append(interpolated)
y_matrix = np.array(all_interp_distributions)
y_matrix[np.isnan(y_matrix)] = 0

y_bins_log = np.logspace(-7, 1.5, 150)

H, xedges, yedges = np.histogram2d(
    np.repeat(common_bins, y_matrix.shape[0]),
    y_matrix.T.flatten(),
    bins=[common_bins, y_bins_log]
)
H = H / y_matrix.shape[0]
H_masked = ma.masked_where(H == 0, H) 
plt.figure(figsize=(9, 6))
norm = LogNorm(vmin=1e-4, vmax=1)
img = plt.pcolormesh(xedges, yedges, H_masked.T, shading='auto', cmap=fading_viridis, norm=norm)
plt.yscale("log")
plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=20, fontweight="bold")
plt.ylabel("CAS Number Concentration\n(cm$^{-3}$ μm$^{-1}$)", fontsize=20, fontweight="bold")
plt.xticks(fontsize=20, fontweight="bold")
plt.yticks(fontsize=20, fontweight="bold")
plt.ylim(1e-7, 10**1.5)
plt.xlim(0, 25)
plt.title("CAS Below Cloud Base\nJanuary–June 2022\nRaw Dry Size Distributions", fontsize=20, fontweight="bold")
cbar = plt.colorbar(img)
cbar.set_label("Fraction of Legs", fontsize=20)
cbar.ax.tick_params(labelsize=18)
cbar.set_ticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
cbar.set_ticklabels([r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$'])

plt.tight_layout()
plt.show()
#%%
#average dry distribution ********
common_bins = np.linspace(2, 25, 35)  
sum_interpolated_dN_dD_dry = np.zeros_like(common_bins, dtype=float)
count_interpolated_dN_dD_dry = np.zeros_like(common_bins, dtype=int)

for entry in filtered_master_BCB_ddry:
    ddry_values = np.array(entry['ddry'])  
    dN_dD_dry = np.array(entry['dN/dDdry'])

    valid_indices = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry)
    if np.sum(valid_indices) < 2:
        continue 

    interp_func = interp1d(ddry_values[valid_indices], dN_dD_dry[valid_indices], 
                           kind='linear', bounds_error=False, fill_value=np.nan)
    interpolated_dN_dD_dry = interp_func(common_bins)

    valid_interpolated_indices = ~np.isnan(interpolated_dN_dD_dry)

    sum_interpolated_dN_dD_dry[valid_interpolated_indices] += interpolated_dN_dD_dry[valid_interpolated_indices]
    count_interpolated_dN_dD_dry[valid_interpolated_indices] += 1

average_dN_dD_dry = np.divide(sum_interpolated_dN_dD_dry, count_interpolated_dN_dD_dry, where=count_interpolated_dN_dD_dry > 0)

plt.figure(figsize=(8, 6))
plt.plot(common_bins, average_dN_dD_dry, color='red', linewidth=2, label='Average Dry Size Distribution')

plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=20, fontweight="bold")
plt.ylabel(r"CAS Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=20, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xlim(0, 45)
plt.xticks(fontweight="bold", fontsize=20)
plt.yticks(fontweight="bold", fontsize=20)
plt.title("CAS Average Below Cloud Base \nDry Size Distribution\n January - June 2022", fontsize=20, fontweight="bold")
plt.legend()
plt.show()

#%%
# Check transformation step
for entry in filtered_master_BCB_ddry[:5]:  
    print(f"Date: {entry['Date']}, Start: {entry['BCB_start']}, Stop: {entry['BCB_stop']}")
    print("  gRh_mean:", entry['gRh_mean'])
    print("  dN/dDdry first 5 bins:", entry['dN/dDdry'][:5])
    print("  ddry_bin_widths first 5 bins:", entry['ddry_bin_widths'][:5])
    print("  Original bin widths:", np.diff(bin_center, append=np.nan)[:5])
    print("  -----")
#%%
for entry in filtered_master_BCB_ddry[:5]: 
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

for entry_ambient, entry_dry in zip(Y_BCB_calc, filtered_master_BCB_ddry):
    ambient_dd = np.array(bin_center)
    ambient_dN_dD = np.array([entry_ambient.get(f'Bin{i}_Y_mean', np.nan) for i in range(12, 30)])

    dry_dd = np.array(entry_dry['ddry'])
    dry_dN_dD = np.array(entry_dry['dN/dDdry'])

    valid_ambient = ~np.isnan(ambient_dd) & ~np.isnan(ambient_dN_dD)
    valid_dry = ~np.isnan(dry_dd) & ~np.isnan(dry_dN_dD)

    if np.sum(valid_ambient) < 2 or np.sum(valid_dry) < 2:
        continue  

    interp_ambient = interp1d(ambient_dd[valid_ambient], ambient_dN_dD[valid_ambient], 
                              kind='linear', bounds_error=False, fill_value=np.nan)
    interp_dry = interp1d(dry_dd[valid_dry], dry_dN_dD[valid_dry], 
                          kind='linear', bounds_error=False, fill_value=np.nan)

    interpolated_ambient = interp_ambient(common_bins)
    interpolated_dry = interp_dry(common_bins)

    plt.plot(common_bins, interpolated_ambient, color='blue', alpha=0.3, label="Ambient" if 'Ambient' not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.plot(common_bins, interpolated_dry, color='red', alpha=0.3, label="Dry" if 'Dry' not in plt.gca().get_legend_handles_labels()[1] else "")


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
# #creating dry files for Jason

# def exponential(x, n0, D):
#     return n0 * np.exp(-x / D)

# target_date = "2022-02-15"
# dry_rows = []
# common_bins = np.linspace(2, 25, 35)
# dry_exponential_fits = []
# dry_exponential_fits_10 = []
# for entry in [e for e in filtered_master_BCB_ddry if e["Date"] == target_date]:
#     ddry = np.array(entry['ddry'])
#     dist = np.array(entry['dN/dDdry'])
#     valid = ~np.isnan(ddry) & ~np.isnan(dist) & (dist > 0)

#     if np.sum(valid) < 2:
#         continue

#     ddry_valid = ddry[valid]
#     dist_valid = dist[valid]

#     try:
#         popt, _ = curve_fit(exponential, ddry_valid, dist_valid, p0=(1, 5), maxfev=5000)
#         n0, D = popt
#     except RuntimeError:
#         n0, D = np.nan, np.nan

#     dry_exponential_fits.append({
#         'Date': entry['Date'],
#         'BCB_start': entry['BCB_start'],
#         'BCB_stop': entry['BCB_stop'],
#         'Dry_Intercept_n0': n0,
#         'Dry_E_folding_D': D
#     })

    
#     mask_10 = ddry_valid <= 10
#     if np.sum(mask_10) >= 2:
#         try:
#             popt10, _ = curve_fit(exponential, ddry_valid[mask_10], dist_valid[mask_10],
#                                   p0=(1, 5), maxfev=5000,
#                                   bounds=([0, 0.1], [np.inf, 20]))
#             n0_10, D_10 = popt10
#         except RuntimeError:
#             n0_10, D_10 = np.nan, np.nan
#     else:
#         n0_10, D_10 = np.nan, np.nan

#     dry_exponential_fits_10.append({
#         'Date': entry['Date'],
#         'BCB_start': entry['BCB_start'],
#         'BCB_stop': entry['BCB_stop'],
#         'Dry_Intercept_n0': n0_10,
#         'Dry_E_folding_D': D_10
#     })

# for i, entry in enumerate([e for e in filtered_master_BCB_ddry if e["Date"] == target_date]):
#     date = entry['Date']
#     start = entry['BCB_start']
#     stop = entry['BCB_stop']
#     row_dist = {
#         "Date": date,
#         "Leg_start (min)": start,
#         "Leg_stop (min)": stop,
#         "Type": "Dry Distribution",
#         "Intercept_n0 (cm^-3 µm^-1)": np.nan,
#         "E_folding_D (µm)": np.nan,
#     }

#     ddry = np.array(entry['ddry'])
#     dist = np.array(entry['dN/dDdry'])
#     valid = ~np.isnan(ddry) & ~np.isnan(dist) & (dist > 0)

#     if np.sum(valid) >= 2:
#         interp = interp1d(ddry[valid], dist[valid], kind='linear',
#                           bounds_error=False, fill_value=np.nan)
#         interp_vals = interp(common_bins)
#         for j, D in enumerate(common_bins):
#             row_dist[f"Conc_at_{D:.2f}µm (cm^-3 µm^-1)"] = interp_vals[j]

#     dry_rows.append(row_dist)
#     fit_full = dry_exponential_fits[i]
#     dry_rows.append({
#         "Date": date,
#         "Leg_start (min)": start,
#         "Leg_stop (min)": stop,
#         "Type": "Full Dry Exponential Fit",
#         "Intercept_n0 (cm^-3 µm^-1)": fit_full["Dry_Intercept_n0"],
#         "E_folding_D (µm)": fit_full["Dry_E_folding_D"],
#     })

#     fit_10 = dry_exponential_fits_10[i]
#     dry_rows.append({
#         "Date": date,
#         "Leg_start (min)": start,
#         "Leg_stop (min)": stop,
#         "Type": "≤10 µm Dry Exponential Fit",
#         "Intercept_n0 (cm^-3 µm^-1)": fit_10["Dry_Intercept_n0"],
#         "E_folding_D (µm)": fit_10["Dry_E_folding_D"],
#     })
# df_dry = pd.DataFrame(dry_rows)
# save_path = "/home/disk/eos4/kathem24/activate/data/CAS/Jason's Model/Feb15_CAS_BCB_Dry_Distributions_and_Fits.csv"
# df_dry.to_csv(save_path, index=False)
# print(" CSV created at:", save_path)
#%%
# #all dry files for Jason 
# def exponential(x, n0, D):
#     return n0 * np.exp(-x / D)

# common_bins = np.linspace(2, 25, 35)  # consistent with your dry plots

# dry_rows = []
# dry_exponential_fits = []
# dry_exponential_fits_10 = []
# all_dates = sorted(set([e["Date"] for e in filtered_master_BCB_ddry]))

# for target_date in all_dates:
#     daily_entries = [e for e in filtered_master_BCB_ddry if e["Date"] == target_date]

#     for entry in daily_entries:

#         ddry = np.array(entry['ddry'])
#         dist = np.array(entry['dN/dDdry'])
#         valid = ~np.isnan(ddry) & ~np.isnan(dist) & (dist > 0)

#         if np.sum(valid) < 2:
#             dry_exponential_fits.append({
#                 'Date': entry['Date'],
#                 'BCB_start': entry['BCB_start'],
#                 'BCB_stop': entry['BCB_stop'],
#                 'Dry_Intercept_n0': np.nan,
#                 'Dry_E_folding_D': np.nan
#             })
#             dry_exponential_fits_10.append({
#                 'Date': entry['Date'],
#                 'BCB_start': entry['BCB_start'],
#                 'BCB_stop': entry['BCB_stop'],
#                 'Dry_Intercept_n0': np.nan,
#                 'Dry_E_folding_D': np.nan
#             })
#             continue

#         ddry_valid = ddry[valid]
#         dist_valid = dist[valid]
#         try:
#             popt, _ = curve_fit(exponential, ddry_valid, dist_valid,
#                                 p0=(1, 5), maxfev=5000)
#             n0, D = popt
#         except RuntimeError:
#             n0, D = np.nan, np.nan

#         dry_exponential_fits.append({
#             'Date': entry['Date'],
#             'BCB_start': entry['BCB_start'],
#             'BCB_stop': entry['BCB_stop'],
#             'Dry_Intercept_n0': n0,
#             'Dry_E_folding_D': D
#         })
#         mask_10 = ddry_valid <= 10
#         if np.sum(mask_10) >= 2:
#             try:
#                 popt10, _ = curve_fit(exponential,
#                                       ddry_valid[mask_10], dist_valid[mask_10],
#                                       p0=(1, 5), maxfev=5000,
#                                       bounds=([0, 0.1], [np.inf, 20]))
#                 n0_10, D_10 = popt10
#             except RuntimeError:
#                 n0_10, D_10 = np.nan, np.nan
#         else:
#             n0_10, D_10 = np.nan, np.nan

#         dry_exponential_fits_10.append({
#             'Date': entry['Date'],
#             'BCB_start': entry['BCB_start'],
#             'BCB_stop': entry['BCB_stop'],
#             'Dry_Intercept_n0': n0_10,
#             'Dry_E_folding_D': D_10
#         })

# fit_idx = 0

# for target_date in all_dates:

#     daily_entries = [e for e in filtered_master_BCB_ddry if e["Date"] == target_date]

#     for entry in daily_entries:
#         date = entry['Date']
#         start = entry['BCB_start']
#         stop = entry['BCB_stop']
#         row_dist = {
#             "Date": date,
#             "Leg_start (min)": start,
#             "Leg_stop (min)": stop,
#             "Type": "Dry Distribution",
#             "Intercept_n0 (cm^-3 µm^-1)": np.nan,
#             "E_folding_D (µm)": np.nan,
#         }

#         ddry = np.array(entry['ddry'])
#         dist = np.array(entry['dN/dDdry'])
#         valid = ~np.isnan(ddry) & ~np.isnan(dist) & (dist > 0)

#         if np.sum(valid) >= 2:
#             interp = interp1d(ddry[valid], dist[valid], kind='linear',
#                               bounds_error=False, fill_value=np.nan)
#             interp_vals = interp(common_bins)
#             for j, D in enumerate(common_bins):
#                 row_dist[f"Conc_at_{D:.2f}µm (cm^-3 µm^-1)"] = interp_vals[j]

#         dry_rows.append(row_dist)
#         fit_full = dry_exponential_fits[fit_idx]
#         dry_rows.append({
#             "Date": date,
#             "Leg_start (min)": start,
#             "Leg_stop (min)": stop,
#             "Type": "Full Dry Exponential Fit",
#             "Intercept_n0 (cm^-3 µm^-1)": fit_full["Dry_Intercept_n0"],
#             "E_folding_D (µm)": fit_full["Dry_E_folding_D"],
#         })
#         fit_10 = dry_exponential_fits_10[fit_idx]
#         dry_rows.append({
#             "Date": date,
#             "Leg_start (min)": start,
#             "Leg_stop (min)": stop,
#             "Type": "≤10 µm Dry Exponential Fit",
#             "Intercept_n0 (cm^-3 µm^-1)": fit_10["Dry_Intercept_n0"],
#             "E_folding_D (µm)": fit_10["Dry_E_folding_D"],
#         })

#         fit_idx += 1
# df_dry = pd.DataFrame(dry_rows)
# save_path = "/home/disk/eos4/kathem24/activate/data/CAS/Jason's Model/ALLDATES_CAS_BCB_Dry_Distributions_and_Fits.csv"
# df_dry.to_csv(save_path, index=False)
# print(" CSV created at:", save_path)
#%%
#checking 
# old_path = "/home/disk/eos4/kathem24/activate/data/CAS/Jason's Model/Feb15_CAS_BCB_Dry_Distributions_and_Fits.csv"
# df_old = pd.read_csv(old_path)
# new_path = "/home/disk/eos4/kathem24/activate/data/CAS/Jason's Model/ALLDATES_CAS_BCB_Dry_Distributions_and_Fits.csv"
# df_new = pd.read_csv(new_path)

# df_new_15 = df_new[df_new["Date"] == "2022-02-15"].reset_index(drop=True)
# df_old_15 = df_old.reset_index(drop=True)
# print("Old Feb15 shape:", df_old_15.shape)
# print("New Feb15 shape:", df_new_15.shape)
# print("Column difference (old - new):", set(df_old_15.columns) - set(df_new_15.columns))
# print("Column difference (new - old):", set(df_new_15.columns) - set(df_old_15.columns))
# set()
# set()
# comparison = df_old_15.equals(df_new_15)
# print("Are the dataframes EXACTLY identical?", comparison)
#%%
# #randomly selecting half the flights 
# df_all = pd.read_csv("/home/disk/eos4/kathem24/activate/data/CAS/Jason's Model/ALLDATES_CAS_BCB_Dry_Distributions_and_Fits.csv")

# df_all["Date"] = pd.to_datetime(df_all["Date"])
# legs_per_date = df_all.groupby("Date").size().reset_index(name="num_legs")
# legs_per_date["Date"] = pd.to_datetime(legs_per_date["Date"])
# legs_per_date["Month"] = legs_per_date["Date"].dt.month
# monthly_legs = legs_per_date.groupby("Month")["num_legs"].sum()
# print(monthly_legs)
# np.random.seed(42)  # for reproducibility
# TARGET = 200
# proportional_target = (monthly_legs / monthly_legs.sum()) * TARGET
# proportional_target = proportional_target.round().astype(int)
# print(proportional_target)
# monthly_targets = {
#     1: 49,
#     2: 45,
#     3: 44,
#     5: 21,
#     6: 42
# }

# np.random.seed(42)

# selected_dates = []

# for month, target in monthly_targets.items():
#     month_df = legs_per_date[legs_per_date["Month"] == month].copy()
#     month_df = month_df.sample(frac=1, random_state=42)  # shuffle dates

#     running = 0
#     chosen = []

#     for _, row in month_df.iterrows():
#         if running >= target:
#             break
#         chosen.append(row["Date"])
#         running += row["num_legs"]

#     print(f"Month {month}: selected {len(chosen)} flights, {running} legs")
#     selected_dates.extend(chosen)
# final_df = df_all[df_all["Date"].isin(selected_dates)]
# print("Total legs selected:", len(final_df))
# print("Dates selected:")
# print(sorted([str(d.date()) for d in selected_dates]))
# output_path = "/home/disk/eos4/kathem24/activate/data/CAS/Jason's Model/Selected_FlightSubset_309Legs.csv"
# final_df.to_csv(output_path, index=False)
# print("Saved to:", output_path)



#%%
#counting errors 
# sample_area_cm2 = 0.0025  # CAS sample area in cm²
# plane_speed_cm_s = 1.2e4  # Plane speed in cm/s (120 m/s)
# sampling_time_s = 198  # Each leg is 3.3 minutes = 198 seconds
# sample_volume = sample_area_cm2 * plane_speed_cm_s * sampling_time_s 
# common_bins = np.linspace(2, 25, 35)
# from itertools import cycle

# colorblind_friendly_colors = [
#     "#000000", "#E69F00", "#56B4E9", "#009E73",
#     "#F0E442", "#0072B2", "#D55E00", "#CC79A7"
# ]
# color_cycle = cycle(colorblind_friendly_colors) 


# plt.figure(figsize=(8, 6))

# for entry in filtered_master_BCB_ddry:
#     ddry_values = np.array(entry['ddry'])
#     dN_dD_dry = np.array(entry['dN/dDdry'])

#     valid_indices = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry)
#     if np.sum(valid_indices) < 2:
#         continue

#     interp_func = interp1d(ddry_values[valid_indices], dN_dD_dry[valid_indices],
#                            kind='linear', bounds_error=False, fill_value=np.nan)
#     interpolated_dN_dD_dry = interp_func(common_bins)

#     valid_interpolated_indices = (interpolated_dN_dD_dry > 0) & ~np.isnan(interpolated_dN_dD_dry)
#     filtered_bins = common_bins[valid_interpolated_indices]
#     filtered_dN_dD_dry = interpolated_dN_dD_dry[valid_interpolated_indices]

#     N_counts = filtered_dN_dD_dry * sample_volume
#     N_counts[N_counts <= 0] = np.nan

#     rel_error = (np.sqrt(N_counts)) / N_counts

#     color = next(color_cycle, "#999999") 
#     plt.plot(filtered_bins, rel_error, marker="o", linestyle="--", alpha=0.4, color=color)

# # Formatting
# plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=20, fontweight="bold")
# plt.ylabel("Relative Counting Error", fontsize=20, fontweight="bold")
# plt.yscale("log") 
# plt.xlim(0, 25)
# plt.xticks(fontsize=20, fontweight="bold")
# plt.yticks(fontsize=20, fontweight="bold")
# plt.title("CAS Instrument Counting Error", fontsize=20, fontweight="bold")
# plt.show()
#%%
# sample_area_cm2 = 0.0025  # CAS sample area in cm²
# plane_speed_cm_s = 1.2e4  # Plane speed in cm/s (120 m/s)
# sampling_time_s = 198  # Each leg is 3.3 minutes = 198 seconds

# sample_volume = sample_area_cm2 * plane_speed_cm_s * sampling_time_s 
# bin_centers = np.linspace(2, 25, 35) 
# all_relative_errors = []  

# for entry in filtered_master_BCB_ddry:
#     ddry_values = np.array(entry['ddry']) 
#     dN_dD_dry = np.array(entry['dN/dDdry']) 

#     valid_indices = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry)
#     if np.sum(valid_indices) < 2:
#         continue 

#     interp_func = interp1d(ddry_values[valid_indices], dN_dD_dry[valid_indices], 
#                            kind='linear', bounds_error=False, fill_value=np.nan)
#     interpolated_dN_dD_dry = interp_func(bin_centers)

#     N_counts = interpolated_dN_dD_dry * sample_volume 
#     N_counts[N_counts <= 0] = np.nan  

#     rel_error = 1 / np.sqrt(N_counts)

#     if len(rel_error) != len(bin_centers):
#         rel_error = np.pad(rel_error, (0, len(bin_centers) - len(rel_error)), constant_values=np.nan)

#     all_relative_errors.append(rel_error)

# relative_errors_df = pd.DataFrame(all_relative_errors, columns=bin_centers)

# stats_summary = pd.DataFrame({
#     'Bin Center (µm)': bin_centers,
#     'Mean Relative Error': np.nanmean(relative_errors_df, axis=0),
#     'Median Relative Error': np.nanmedian(relative_errors_df, axis=0),
#     'Min Relative Error': np.nanmin(relative_errors_df, axis=0),
#     'Max Relative Error': np.nanmax(relative_errors_df, axis=0),
# })

# print(stats_summary)

# plt.figure(figsize=(8, 6))
# plt.plot(stats_summary['Bin Center (µm)'], stats_summary['Mean Relative Error'], marker='o', linestyle='-', label="Mean Error")
# plt.plot(stats_summary['Bin Center (µm)'], stats_summary['Max Relative Error'], marker='s', linestyle='--', label="Max Error", alpha=0.5)
# plt.plot(stats_summary['Bin Center (µm)'], stats_summary['Min Relative Error'], marker='^', linestyle='--', label="Min Error", alpha=0.5)

# plt.xlabel("Dry Bin Centers Diameter (µm)", fontsize=19, fontweight="bold")
# plt.ylabel("Relative Counting Error", fontsize=19, fontweight="bold")
# plt.yscale("log")
# plt.legend()
# plt.xticks(fontweight="bold", fontsize=17)
# plt.yticks(fontweight="bold", fontsize=17)
# plt.title("CAS Counting Errors Across Dry Size Bins", fontsize=19, fontweight="bold")

# plt.show()
# #%%
# sample_area_cm2 = 0.0025  # CAS sample area in cm²
# plane_speed_cm_s = 1.2e4  # Plane speed in cm/s (120 m/s)
# sampling_time_s = 198  # Each leg is 3.3 minutes = 198 seconds
# sample_volume = sample_area_cm2 * plane_speed_cm_s * sampling_time_s 

# bin_centers = np.linspace(2, 25, 35) 
# all_relative_errors = []

# for entry in filtered_master_BCB_ddry:
#     ddry_values = np.array(entry['ddry'])  
#     dN_dD_dry = np.array(entry['dN/dDdry']) 

#     valid_indices = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry)
#     if np.sum(valid_indices) < 2:
#         continue  

#     interp_func = interp1d(ddry_values[valid_indices], dN_dD_dry[valid_indices], 
#                            kind='linear', bounds_error=False, fill_value=np.nan)
#     interpolated_dN_dD_dry = interp_func(bin_centers)

#     N_counts = interpolated_dN_dD_dry * sample_volume

#     N_counts[N_counts <= 0] = np.nan  

#     rel_error = np.sqrt(N_counts) / N_counts 

    
#     if len(rel_error) != len(bin_centers):
#         rel_error = np.pad(rel_error, (0, len(bin_centers) - len(rel_error)), constant_values=np.nan)

#     all_relative_errors.append(rel_error)

# relative_errors_df = pd.DataFrame(all_relative_errors, columns=bin_centers)

# stats_summary = pd.DataFrame({
#     'Bin Center (µm)': bin_centers,
#     'Mean Relative Error': np.nanmean(relative_errors_df, axis=0),
#     'Median Relative Error': np.nanmedian(relative_errors_df, axis=0),
#     'Min Relative Error': np.nanmin(relative_errors_df, axis=0),
#     'Max Relative Error': np.nanmax(relative_errors_df, axis=0),
# })

# print(stats_summary)
# plt.figure(figsize=(8, 6))
# plt.plot(stats_summary['Bin Center (µm)'], stats_summary['Mean Relative Error'], 
#          marker='o', linestyle='-', color='#4daf4a', label="Mean Error")  # green
# plt.plot(stats_summary['Bin Center (µm)'], stats_summary['Max Relative Error'], 
#          marker='s', linestyle='--', color='#984ea3', alpha=0.7, label="Max Error")  # purple

# plt.plot(stats_summary['Bin Center (µm)'], stats_summary['Min Relative Error'], 
#          marker='^', linestyle='--', color='#ff7f00', alpha=0.7, label="Min Error")  # orange

# plt.xlabel("Dry Bin Centers Diameter (µm)", fontsize=20, fontweight="bold")
# plt.ylabel("Relative Counting Error", fontsize=20, fontweight="bold")
# plt.yscale("log")
# plt.xticks(fontweight="bold", fontsize=20)
# plt.yticks(fontweight="bold", fontsize=20)  
# plt.legend(loc='lower right', frameon=False, fontsize=18)
# plt.title("CAS Instrument Counting Error", fontsize=20, fontweight="bold")
# plt.tight_layout()
# plt.show()

#%%
cutoff_bin = 25  
summary_table = stats_summary.iloc[:cutoff_bin+5][['Bin Center (µm)', 'Mean Relative Error', 'Max Relative Error', 'Min Relative Error']]
#%%
#average fitted distribution

common_bins = np.linspace(2, 25, 35)
for entry in filtered_master_BCB_ddry:
    ddry_values = np.array(entry['ddry'])
    dN_dD_dry = np.array(entry['dN/dDdry'])
    valid_indices = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry) & (dN_dD_dry > 0)
    if np.sum(valid_indices) < 2:  
        continue

    interp_func = interp1d(ddry_values[valid_indices], dN_dD_dry[valid_indices], 
                           kind='linear', bounds_error=False, fill_value=np.nan)
    interpolated_dN_dD_dry = interp_func(common_bins)

    valid_interpolated_indices = (interpolated_dN_dD_dry > 0) & ~np.isnan(interpolated_dN_dD_dry)

    sum_interpolated_dN_dD_dry[valid_interpolated_indices] += interpolated_dN_dD_dry[valid_interpolated_indices]
    count_interpolated_dN_dD_dry[valid_interpolated_indices] += 1

average_dN_dD_dry = np.divide(sum_interpolated_dN_dD_dry, count_interpolated_dN_dD_dry, where=count_interpolated_dN_dD_dry > 0)

valid_fit_indices = ~np.isnan(average_dN_dD_dry) & (average_dN_dD_dry > 0)
fit_bins = common_bins[valid_fit_indices]
fit_values = average_dN_dD_dry[valid_fit_indices]
plt.figure(figsize=(8, 6))
plt.plot(common_bins, average_dN_dD_dry, color='red', linewidth=2, label='Average Dry Size Distribution')

plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"CAS Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-7, 10**0)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("CAS Average Below Cloud Base Exponential Fitted Dry Size Distribution\n January - June 2022", fontsize=14, fontweight="bold")
plt.legend()
plt.show()

#%%
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

valid_fit_indices = common_bins <= 10
x_fit = common_bins[valid_fit_indices]
y_fit = average_dN_dD_dry[valid_fit_indices]

valid_data_indices = ~np.isnan(y_fit) & (y_fit > 0)
x_fit = x_fit[valid_data_indices]
y_fit = y_fit[valid_data_indices]

try:
    popt, pcov = curve_fit(exponential, x_fit, y_fit, p0=(1e-2, 2)) 
    n0_fit, D_fit = popt  
except RuntimeError:
    print("Exponential fit failed.")
    n0_fit, D_fit = None, None

plt.figure(figsize=(8, 6))
plt.plot(common_bins, average_dN_dD_dry, color='black', linewidth=2, label='Average Dry Size Distribution')

if n0_fit is not None and D_fit is not None:
    plt.plot(x_fit, exponential(x_fit, *popt), 'r--', linewidth=2, label=f'Exponential Fit: $N_0$={n0_fit:.2e}, $D$={D_fit:.2f} μm')

plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"CAS Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("CAS Average Below Cloud Base Dry Size Distribution\n January - June 2022", fontsize=14, fontweight="bold")
plt.legend()

plt.show()


if n0_fit is not None and D_fit is not None:
    print(f"Fitted Parameters: N_0 = {n0_fit:.3e}, D = {D_fit:.3f} μm")



#%%
#fitting an exponential to dry distributions, removing those two weird lines, and removing extreme slopes after 10 um slope

def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

dry_exponential_fits = []

plt.figure(figsize=(8, 6))

for entry in filtered_master_BCB_ddry:
    ddry_values = np.array(entry['ddry'])
    dN_dD_dry = np.array(entry['dN/dDdry'])

    
    valid_indices = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry) & (dN_dD_dry > 0)
    
    if np.sum(valid_indices) < 2:  
        continue

    try:
       
        popt, _ = curve_fit(exponential, ddry_values[valid_indices], dN_dD_dry[valid_indices], 
                            p0=(1, 5), maxfev=5000)
        n0, D = popt

       
        if D > 20:
            continue  

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

plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=19, fontweight="bold")
plt.ylabel(r"CAS Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=19, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-7, 10**1.5)
plt.xticks(fontweight="bold", fontsize=19)
plt.yticks(fontweight="bold", fontsize=19)
plt.title("CAS Below Cloud Base\n January-June 2022\nFitted Dry Size Distributions", fontsize=20, fontweight="bold")
plt.show()
#%%
x_common = np.linspace(2, 25, 200) 
y_matrix = []
for entry in dry_exponential_fits:
    n0 = entry['Dry_Intercept_n0']
    D = entry['Dry_E_folding_D']
    y_fit = exponential(x_common, n0, D)
    y_matrix.append(y_fit)

y_matrix = np.array(y_matrix)
y_matrix[np.isnan(y_matrix)] = 0
y_matrix[y_matrix <= 0] = 1e-10

y_bins_log = np.logspace(-7, 1.5, 150)

H, xedges, yedges = np.histogram2d(
    np.repeat(x_common, y_matrix.shape[0]),         
    y_matrix.T.flatten(),                            
    bins=[x_common, y_bins_log]
)
H = H / y_matrix.shape[0] 
plt.figure(figsize=(9, 6))
plt.pcolormesh(xedges, yedges, H.T, shading='auto', cmap='viridis', norm=LogNorm())
plt.yscale("log")
plt.ylim(10**-7, 10**1.5) 
plt.xlim(0.5, 25)
plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=20, fontweight="bold")
plt.ylabel("CAS Number Concentration\n(cm$^{-3}$ μm$^{-1}$)", fontsize=19, fontweight="bold")
plt.xticks(fontsize=20, fontweight="bold")
plt.yticks(fontsize=20, fontweight="bold")
plt.title("CAS Below Cloud Base\nJanuary–June 2022\n Fitted Dry Size Distributions", fontsize=20, fontweight="bold")
import matplotlib.ticker as ticker
cbar = plt.colorbar()
cbar.set_label("Fraction of Legs", fontsize=20)
cbar.ax.tick_params(labelsize=18) 
cbar.set_ticks([1e-2, 1e-1, 1e0]) 
cbar.set_ticklabels(['$10^{-2}$', '$10^{-1}$', '$10^{0}$']) 
plt.tight_layout()
plt.show()
#%%
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)
base_cmap = plt.cm.viridis
colors = base_cmap(np.linspace(0, 1, 256))
colors[:80] = np.linspace([1, 1, 1, 1], colors[80], 80)
fading_viridis = LinearSegmentedColormap.from_list("fading_viridis", colors)
x_common = np.linspace(2, 25, 200)
y_matrix = []
for entry in dry_exponential_fits:
    n0 = entry['Dry_Intercept_n0']
    D = entry['Dry_E_folding_D']
    y_fit = exponential(x_common, n0, D)
    y_matrix.append(y_fit)
y_matrix = np.array(y_matrix)
y_matrix[np.isnan(y_matrix)] = 0
y_matrix[y_matrix <= 0] = 1e-10 
y_bins_log = np.logspace(-7, 1.5, 150)
H, xedges, yedges = np.histogram2d(
    np.repeat(x_common, y_matrix.shape[0]),
    y_matrix.T.flatten(),
    bins=[x_common, y_bins_log]
)
H = H / y_matrix.shape[0]  
H_masked = ma.masked_where(H == 0, H) 
plt.figure(figsize=(9, 6))
norm = LogNorm(vmin=1e-4, vmax=1)
img = plt.pcolormesh(xedges, yedges, H_masked.T, shading='auto', cmap=fading_viridis, norm=norm)
plt.yscale("log")
plt.ylim(1e-7, 10**1.5)
plt.xlim(0.5, 25)
plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=20, fontweight="bold")
plt.ylabel("CAS Number Concentration\n(cm$^{-3}$ μm$^{-1}$)", fontsize=19, fontweight="bold")
plt.xticks(fontsize=20, fontweight="bold")
plt.yticks(fontsize=20, fontweight="bold")
plt.title("CAS Below Cloud Base\nJanuary–June 2022\nFitted Dry Size Distributions", fontsize=20, fontweight="bold")
cbar = plt.colorbar(img)
cbar.set_label("Fraction of Legs", fontsize=20)
cbar.ax.tick_params(labelsize=18)
cbar.set_ticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
cbar.set_ticklabels([r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$'])
plt.tight_layout()
plt.show()
#%%
#only to 10um 

def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

dry_exponential_fits_10 = []

plt.figure(figsize=(8, 6))

for entry in filtered_master_BCB_ddry:
    ddry_values = np.array(entry['ddry'])
    dN_dD_dry = np.array(entry['dN/dDdry'])
    valid_indices = (ddry_values <= 10) & ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry)
    if np.sum(valid_indices) == 0:
        dry_exponential_fits_10.append({
            'Date': entry['Date'],
            'BCB_start': entry['BCB_start'],
            'BCB_stop': entry['BCB_stop'],
            'Dry_Intercept_n0': np.nan,
            'Dry_E_folding_D': np.nan
        })
        continue 

    try:
        popt, _ = curve_fit(exponential, ddry_values[valid_indices], dN_dD_dry[valid_indices], p0=(1, 5), maxfev=5000)
        n0, D = popt

    except RuntimeError:
        print(f"Fit could not be performed for date {entry['Date']}")
        n0, D = np.nan, np.nan  
    dry_exponential_fits_10.append({
        'Date': entry['Date'],
        'BCB_start': entry['BCB_start'],
        'BCB_stop': entry['BCB_stop'],
        'Dry_Intercept_n0': n0,
        'Dry_E_folding_D': D
    })
    if not np.isnan(n0) and not np.isnan(D):
        x_fit = np.linspace(min(ddry_values[valid_indices]), 10, 100)
        y_fit = exponential(x_fit, n0, D)
        plt.plot(x_fit, y_fit, color='black', alpha=0.2)
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
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

dry_exponential_fits_10 = []
plt.figure(figsize=(8, 6))
for entry in filtered_master_BCB_ddry:
    ddry_values = np.array(entry['ddry'])
    dN_dD_dry = np.array(entry['dN/dDdry'])

    valid_indices = (ddry_values <= 10) & ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry)
    if np.sum(valid_indices) == 0:
        dry_exponential_fits_10.append({
            'Date': entry['Date'],
            'BCB_start': entry['BCB_start'],
            'BCB_stop': entry['BCB_stop'],
            'Dry_Intercept_n0': np.nan,
            'Dry_E_folding_D': np.nan
        })
        continue  

    try:
        popt, _ = curve_fit(exponential, ddry_values[valid_indices], dN_dD_dry[valid_indices], p0=(1, 5), maxfev=5000)
        n0, D = popt

        if D < 0.5 or D > 20: 
            raise RuntimeError("D value out of range")

    except RuntimeError:
        print(f"Fit failed for {entry['Date']} (D={D:.2f})")
        n0, D = np.nan, np.nan 

    dry_exponential_fits_10.append({
        'Date': entry['Date'],
        'BCB_start': entry['BCB_start'],
        'BCB_stop': entry['BCB_stop'],
        'Dry_Intercept_n0': n0,
        'Dry_E_folding_D': D
    })

    if not np.isnan(n0) and not np.isnan(D):
        x_fit = np.linspace(2, 10, 100)  
        y_fit = exponential(x_fit, n0, D)

        y_fit[y_fit < 1e-15] = np.nan

        plt.plot(x_fit, y_fit, color='black', alpha=0.2)
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
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)
base_cmap = plt.cm.viridis
colors = base_cmap(np.linspace(0, 1, 256))
colors[:80] = np.linspace([1, 1, 1, 1], colors[80], 80)
fading_viridis = LinearSegmentedColormap.from_list("fading_viridis", colors)
dry_exponential_fits_10 = []
fitted_curves = []

x_fit = np.linspace(2, 10, 200)

for entry in filtered_master_BCB_ddry:
    ddry_values = np.array(entry['ddry'])
    dN_dD_dry = np.array(entry['dN/dDdry'])

    valid = (ddry_values <= 10) & ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry)

    if np.sum(valid) == 0:
        dry_exponential_fits_10.append({
            'Date': entry['Date'],
            'BCB_start': entry['BCB_start'],
            'BCB_stop': entry['BCB_stop'],
            'Dry_Intercept_n0': np.nan,
            'Dry_E_folding_D': np.nan
        })
        continue

    try:
        popt, _ = curve_fit(exponential, ddry_values[valid], dN_dD_dry[valid], p0=(1, 5), maxfev=5000)
        n0, D = popt

        if D < 0.5 or D > 20:
            raise RuntimeError("D out of range")
    except RuntimeError:
        n0, D = np.nan, np.nan

    dry_exponential_fits_10.append({
        'Date': entry['Date'],
        'BCB_start': entry['BCB_start'],
        'BCB_stop': entry['BCB_stop'],
        'Dry_Intercept_n0': n0,
        'Dry_E_folding_D': D
    })

    if not np.isnan(n0) and not np.isnan(D):
        y_fit = exponential(x_fit, n0, D)
        y_fit[y_fit <= 0] = np.nan
        fitted_curves.append(y_fit)
y_matrix = np.array(fitted_curves)
y_matrix[np.isnan(y_matrix)] = 0
y_bins_log = np.logspace(-7, 1.5, 150)
H, xedges, yedges = np.histogram2d(
    np.repeat(x_fit, y_matrix.shape[0]),
    y_matrix.T.flatten(),
    bins=[x_fit, y_bins_log]
)
H = H / y_matrix.shape[0]
H_masked = ma.masked_where(H == 0, H)
plt.figure(figsize=(9, 6))
norm = LogNorm(vmin=1e-4, vmax=1)
img = plt.pcolormesh(xedges, yedges, H_masked.T, shading='auto', cmap=fading_viridis, norm=norm)
plt.yscale("log")
plt.ylim(1e-7, 1e1)
plt.xlim(0, 10)
plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=19, fontweight="bold")
plt.ylabel("CAS Number Concentration\n(cm$^{-3}$ μm$^{-1}$)", fontsize=19, fontweight="bold")
plt.xticks(fontsize=19, fontweight="bold")
plt.yticks(fontsize=19, fontweight="bold")
plt.title("CAS Below Cloud Base January – June 2022\nFitted Dry Size Distributions (≤10 μm)", fontsize=19, fontweight="bold")
cbar = plt.colorbar(img)
cbar.set_label("Fraction of Legs", fontsize=15)
cbar.ax.tick_params(labelsize=13)
cbar.set_ticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
cbar.set_ticklabels([r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$'])
plt.tight_layout()
plt.show()
success_count = sum(not np.isnan(f['Dry_Intercept_n0']) for f in dry_exponential_fits_10)
print(f"Total successful dry exponential fits: {success_count}")

#%%
#histogram comapring less than 10um and regular fit exponential 
dry_slopes_10 = [fit['Dry_E_folding_D'] for fit in dry_exponential_fits_10 if not np.isnan(fit['Dry_E_folding_D'])] 
#%%
dry_intercepts_10=[fit['Dry_Intercept_n0'] for fit in dry_exponential_fits_10 if not np.isnan(fit['Dry_Intercept_n0'])]
# %%
#Scatterplot of ambient slope versus ambient intercept

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
    n0 = entry['Dry_Intercept_n0'] 
    D = entry['Dry_E_folding_D']  

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
#%%
#save dry_exponential_fits_10 to csv
# save_dir = "/home/disk/eos4/kathem24/activate/data/CAS"
# os.makedirs(save_dir, exist_ok=True)   # ensures directory exists
# save_path = os.path.join(save_dir, "dry_exponential_fits_10.csv")
# dry_exponential_fits_10_df = pd.DataFrame(dry_exponential_fits_10)
# dry_exponential_fits_10_df.to_csv(save_path, index=False)
# print(f"Saved to: {save_path}")
# %%
plt.figure(figsize=(10, 6))
plt.scatter(df_ambient['Ambient_Slope_D'], df_ambient['Ambient_Intercept_N0'], 
            alpha=0.6, color='blue', label='Ambient')
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
def calculate_mass(N0, D):
    N0_m4 = N0 * 10**6  # Convert cm⁻³µm⁻¹ to m⁻⁴
    integrand = lambda d: np.exp(-d / D) * (d * 1e-6)**3  # Convert µm³ → m³
    # mass_integral, _ = quad(integrand, 2, np.inf)  # Integrate from 2µm to ∞
    mass_integral, _ = quad(integrand, 2, 10)  # Integrate from 2µm to 10µm
    return (np.pi / 6) * rho_salt * N0_m4 * mass_integral  
x_min, x_max = 10**-0.4, 10**1  
y_min, y_max = 10**-1.6, 10**1.3 

xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200)
ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200)  
D_grid_extended, dryintercept_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)
mass_grid_extended = np.zeros_like(D_grid_extended)
for i in range(D_grid_extended.shape[0]):
    for j in range(D_grid_extended.shape[1]):
        mass_grid_extended[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j]) * 1e9  # Convert kg/m³ to µg/m³
dry_slopes = []
dry_intercepts = []

for entry in dry_exponential_fits_10:
    n0 = entry['Dry_Intercept_n0']
    D = entry['Dry_E_folding_D'] 

    dry_intercepts.append(n0)
    dry_slopes.append(D)
dry_slopes = np.array(dry_slopes)
dry_intercepts = np.array(dry_intercepts)
mass_levels = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
plt.figure(figsize=(10, 8))
plt.scatter(dry_slopes, dry_intercepts, c='blue', s=80, alpha=0.7, label="Dry Data Points")
contour_plot = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_extended, 
                           levels=mass_levels, colors='red', alpha=0.75, linewidths=1.5)
fmt = {level: f'{int(level)} µg/m³' for level in mass_levels}
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
#Dry mass inf

rho_salt = 2200
def calculate_mass(N0, D):
    """Compute dry mass using the exponential fit (N0, D)."""
    N0_m4 = N0 * 10**6 
    integrand = lambda d: np.exp(-d / D) * (d * 1e-6)**3  # Convert µm³ → m³
    mass_integral, _ = quad(integrand, 2, np.inf)  # Integrate from 2µm to ∞
    # mass_integral, _ = quad(integrand, 2, 10)  # Integrate from 2µm to 10µm
    return (np.pi / 6) * rho_salt * N0_m4 * mass_integral 
x_min, x_max = 10**-0.4, 10**1  
y_min, y_max = 10**-1.6, 10**1.3 

xgrid_extended = np.logspace(np.log10(x_min), np.log10(x_max), 200)
ygrid_extended = np.logspace(np.log10(y_min), np.log10(y_max), 200) 

D_grid_extended, dryintercept_grid_extended = np.meshgrid(xgrid_extended, ygrid_extended)
mass_grid_extended = np.zeros_like(D_grid_extended)
for i in range(D_grid_extended.shape[0]):
    for j in range(D_grid_extended.shape[1]):
        mass_grid_extended[i, j] = calculate_mass(dryintercept_grid_extended[i, j], D_grid_extended[i, j]) * 1e9  # Convert kg/m³ to µg/m³
dry_slopes = []
dry_intercepts = []

for entry in dry_exponential_fits_10:
    n0 = entry['Dry_Intercept_n0']  
    D = entry['Dry_E_folding_D']  

    dry_intercepts.append(n0)
    dry_slopes.append(D)
dry_slopes = np.array(dry_slopes)
dry_intercepts = np.array(dry_intercepts)
mass_levels = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
plt.figure(figsize=(10, 8))
plt.scatter(dry_slopes, dry_intercepts, c='blue', s=80, alpha=0.7, label="Dry Data Points")
contour_plot = plt.contour(D_grid_extended, dryintercept_grid_extended, mass_grid_extended, 
                           levels=mass_levels, colors='red', alpha=0.75, linewidths=1.5)
fmt = {level: f'{int(level)} µg/m³' for level in mass_levels}
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
#Fixing contours 
min_slope_threshold = np.percentile(dry_slopes, 1)
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
    N0_m4 = N0 * 10**6  # Convert cm⁻³µm⁻¹ to m⁻⁴
    integrand = lambda d: np.exp(-d / D) * (d * 1e-6)**3  # Convert µm³ → m³
    mass_integral, _ = quad(integrand, 2, 10)  # Integrate from 2µm to ∞
    return (np.pi / 6) * rho_salt * N0_m4 * mass_integral  
dry_mass_data = []
for entry in dry_exponential_fits_10:
    date = entry['Date']
    dry_intercept = entry['Dry_Intercept_n0']
    dry_slope = entry['Dry_E_folding_D']

    if dry_slope > 0 and dry_intercept > 0:
        mass_value = calculate_mass(dry_intercept, dry_slope) * 1e9 
        dry_mass_data.append({
            'Date': date,
            'Dry Slope (D)': dry_slope,
            'Dry Intercept (N0)': dry_intercept,
            'Dry Mass (µg/m³)': mass_value
        })
dry_slopes = np.array([entry['Dry Slope (D)'] for entry in dry_mass_data])
dry_intercepts = np.array([entry['Dry Intercept (N0)'] for entry in dry_mass_data])
dry_masses = np.array([entry['Dry Mass (µg/m³)'] for entry in dry_mass_data])
min_slope_threshold = np.percentile(dry_slopes, 1)
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
    N0_m4 = N0 * 10**6  # Convert cm⁻³µm⁻¹ to m⁻⁴
    integrand = lambda d: np.exp(-d / D) * (d * 1e-6)**3  # Convert µm³ → m³
    mass_integral, _ = quad(integrand, 2, 10)  # Integrate from 2µm to ∞
    return (np.pi / 6) * rho_salt * N0_m4 * mass_integral  
dry_mass_data_10 = []
for entry in dry_exponential_fits_10:
    date = entry['Date']
    dry_intercept = entry['Dry_Intercept_n0']
    dry_slope = entry['Dry_E_folding_D']

    if dry_slope > 0 and dry_intercept > 0:
        mass_value = calculate_mass(dry_intercept, dry_slope) * 1e9 
        dry_mass_data_10.append({
            'Date': date,
            'Dry Slope (D)': dry_slope,
            'Dry Intercept (N0)': dry_intercept,
            'Dry Mass (µg/m³)': mass_value
        })
dry_slopes = np.array([entry['Dry Slope (D)'] for entry in dry_mass_data_10])
dry_intercepts = np.array([entry['Dry Intercept (N0)'] for entry in dry_mass_data_10])
dry_masses = np.array([entry['Dry Mass (µg/m³)'] for entry in dry_mass_data_10])

min_slope_threshold = np.percentile(dry_slopes, 1) 

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
    N0_m4 = N0 * 10**6  # Convert cm⁻³µm⁻¹ to m⁻⁴
    integrand = lambda d: np.exp(-d / D) * (d * 1e-6)**3  # Convert µm³ → m³
    mass_integral, _ = quad(integrand, 2, np.inf)
    # mass_integral, _ = quad(integrand, 2, 10)  # Integrate from 2µm to ∞
    return (np.pi / 6) * rho_salt * N0_m4 * mass_integral  
dry_mass_data_inf = []
for entry in dry_exponential_fits_10:
    date = entry['Date']
    dry_intercept = entry['Dry_Intercept_n0']
    dry_slope = entry['Dry_E_folding_D']

    if dry_slope > 0 and dry_intercept > 0:
        mass_value = calculate_mass(dry_intercept, dry_slope) * 1e9 
        dry_mass_data_inf.append({
            'Date': date,
            'Dry Slope (D)': dry_slope,
            'Dry Intercept (N0)': dry_intercept,
            'Dry Mass (µg/m³)': mass_value
        })
dry_slopes = np.array([entry['Dry Slope (D)'] for entry in dry_mass_data_inf])
dry_intercepts = np.array([entry['Dry Intercept (N0)'] for entry in dry_mass_data_inf])
dry_masses = np.array([entry['Dry Mass (µg/m³)'] for entry in dry_mass_data_inf])

min_slope_threshold = np.percentile(dry_slopes, 1) 

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
        mass_value = calculate_mass(dry_intercept, dry_slope) * 1e9
        dry_mass_data_10.append({
            'Date': date,
            'Dry Slope (D)': dry_slope,
            'Dry Intercept (N0)': dry_intercept,
            'Dry Mass (µg/m³)': mass_value
        })

dry_slopes = np.array([entry['Dry Slope (D)'] for entry in dry_mass_data_10])
dry_intercepts = np.array([entry['Dry Intercept (N0)'] for entry in dry_mass_data_10])

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
        'BCB_start': entry['BCB_start'], 
        'BCB_stop': entry['BCB_stop'],  
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
mass_threshold = 100  # µg/m³
# Filter out outliers with mass greater than 50 µg/m³
filtered_dry_mass_10 = [entry for entry in dry_mass_data_10 if (
    not np.isnan(entry['Dry Slope (D)']) and 
    not np.isnan(entry['Dry Intercept (N0)']) and 
    entry['Dry Mass (µg/m³)'] <= mass_threshold
)]

print(f"Filtered Dry Mass Entries: {len(filtered_dry_mass_10)} (after removing masses > {mass_threshold} µg/m³)")
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
mass_threshold = 100  # µg/m³


filtered_dry_mass_inf = [entry for entry in dry_mass_data_inf if (
    not np.isnan(entry['Dry Slope (D)']) and 
    not np.isnan(entry['Dry Intercept (N0)']) and 
    entry['Dry Mass (µg/m³)'] <= mass_threshold
)]
print(f"Filtered Dry Mass Entries: {len(filtered_dry_mass_inf)} (after removing masses > {mass_threshold} µg/m³)")
slope_array = np.array([entry['Dry Slope (D)'] for entry in filtered_dry_mass_inf]).reshape(-1, 1)
intercept_array = np.array([entry['Dry Intercept (N0)'] for entry in filtered_dry_mass_inf]).reshape(-1, 1)
data_points = np.column_stack((slope_array, intercept_array))
#%%
mass_values = np.array([entry['Dry Mass (µg/m³)'] for entry in filtered_dry_mass_inf])
mean_mass = np.mean(mass_values)
print(f"Mean mass after filtering: {mean_mass:.3f} µg/m³")
median_mass = np.median(mass_values)
print(f"Median mass after filtering: {median_mass:.3f} µg/m³")
#%%
filtered_mass_values_ug_inf = [entry['Dry Mass (µg/m³)'] for entry in filtered_dry_mass_inf]
mean_mass_filtered_inf = np.mean(filtered_mass_values_ug_inf)
median_mass_filtered_inf = np.median(filtered_mass_values_ug_inf)
print(f"Filtered Mean Mass: {mean_mass_filtered_inf:.2f} µg/m³")
print(f"Filtered Median Mass: {median_mass_filtered_inf:.2f} µg/m³")
#%%
# saving mass to a csv for Jason (FILTERED ≤ 100)
# df_dry_mass_inf = pd.DataFrame(filtered_dry_mass_inf)
# output_path = "Dry_mass_BCB2022_lessthan100massREAL.csv"
# df_dry_mass_inf.to_csv(output_path, index=False)
# print(f"Saved filtered dry mass data (≤100) to {output_path}")
# mass_col = "Dry Mass (µg/m³)"
# print("Max in saved df:", df_dry_mass_inf[mass_col].max())
# print("Mean in saved df:", df_dry_mass_inf[mass_col].mean())
#%%
slope_values = np.array([entry['Dry Slope (D)'] for entry in filtered_dry_mass_inf])
mean_slope = np.mean(slope_values)
median_slope = np.median(slope_values)
print(f"Mean slope (D): {mean_slope:.3f}")
print(f"Median slope (D): {median_slope:.3f}")
#%%
#removing corresponding concentration legs based on the mass threshold 
mass_threshold = 100.0  # µg/m^3
def make_leg_key(entry):
    return (entry["Date"], int(entry["BCB_start"]), int(entry["BCB_stop"]))
bad_leg_keys = {
    make_leg_key(e)
    for e in dry_mass_data_inf
    if (not np.isnan(e["Dry Mass (µg/m³)"])) and (e["Dry Mass (µg/m³)"] > mass_threshold)
}
print("Legs with mass > threshold:", len(bad_leg_keys))
#%%
def make_leg_key_ddry(entry):
    return (entry["Date"], int(entry["BCB_start"]), int(entry["BCB_stop"]))
filtered_master_BCB_ddry_mass100gone = [
    e for e in filtered_master_BCB_ddry
    if make_leg_key_ddry(e) not in bad_leg_keys
]
print("Original ddry legs:", len(filtered_master_BCB_ddry))
print("After removing high-mass legs:", len(filtered_master_BCB_ddry_mass100gone))
#%%
import pickle
with open("CAS_ddry_massLE100.pkl", "wb") as f:
    pickle.dump(filtered_master_BCB_ddry_mass100gone, f)
print("Saved CAS filtered legs.")
#%%
common_bins = np.linspace(2, 25, 35)
sum_interpolated = np.zeros_like(common_bins, dtype=float)
count_interpolated = np.zeros_like(common_bins, dtype=int)

for entry in filtered_master_BCB_ddry_mass100gone:
    ddry_values = np.array(entry["ddry"])
    dN_dD_dry = np.array(entry["dN/dDdry"])

    valid = np.isfinite(ddry_values) & np.isfinite(dN_dD_dry)
    if valid.sum() < 2:
        continue

    interp_func = interp1d(ddry_values[valid], dN_dD_dry[valid],
                           kind="linear", bounds_error=False, fill_value=np.nan)

    y = interp_func(common_bins)
    good = np.isfinite(y)

    sum_interpolated[good] += y[good]
    count_interpolated[good] += 1

average_dN_dD_dry = np.divide(sum_interpolated, count_interpolated,
                              where=count_interpolated > 0)
print("Number of legs plotted:", len(filtered_master_BCB_ddry_mass100gone))
plt.figure(figsize=(8, 6))
plt.plot(common_bins, average_dN_dD_dry, color="red", linewidth=2,
         label="Average Dry Size Distribution")
plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=20, fontweight="bold")
plt.ylabel(r"CAS Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=20, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xlim(0, 45)
plt.xticks(fontweight="bold", fontsize=20)
plt.yticks(fontweight="bold", fontsize=20)
plt.title("CAS Average Below Cloud Base \nDry Size Distribution\n January - June 2022", fontsize=20, fontweight="bold")
plt.legend()
plt.show()
#%%
#saving the average dry size distribution 
np.savez("CAS_average_drysize_nomass100.npz",
         bins=common_bins,
         average=average_dN_dD_dry)
print("Saved CAS averaged distribution.")
print("Saved to:", os.path.abspath("CAS_ddry_massLE100.pkl"))
print("Exists?", os.path.exists("CAS_ddry_massLE100.pkl"))
#%%
#new concentration stats
mass_threshold = 100.0  # µg/m^3

def make_leg_key(entry):
    return (entry["Date"], int(entry["BCB_start"]), int(entry["BCB_stop"]))

# legs to remove (mass > 100)
bad_leg_keys = {
    make_leg_key(e)
    for e in dry_mass_data_inf
    if (not np.isnan(e["Dry Mass (µg/m³)"])) and (e["Dry Mass (µg/m³)"] > mass_threshold)
}
print("Legs with mass > threshold:", len(bad_leg_keys))

# ---- KEEP YOUR ORIGINAL METHOD EXACTLY ----
# (assumes you already built total_concentration_cm3 using your bin_log method)

# remove those legs from the concentration list
total_concentration_cm3_mass100gone = [
    e for e in total_concentration_cm3
    if make_leg_key(e) not in bad_leg_keys
]

# compute mean/median exactly like before
total_Y_concentrations = [
    e["Total_Y_Concentration_cm3"] for e in total_concentration_cm3_mass100gone
]
total_Y_concentrations = [c for c in total_Y_concentrations if not np.isnan(c)]

print("N legs (before):", len([e["Total_Y_Concentration_cm3"] for e in total_concentration_cm3 if not np.isnan(e["Total_Y_Concentration_cm3"])]))
print("N legs (after) :", len(total_Y_concentrations))

print(f"Mean Total Number Concentration (mass≤100): {np.mean(total_Y_concentrations):.2f} cm⁻³")
print(f"Median Total Number Concentration (mass≤100): {np.median(total_Y_concentrations):.2f} cm⁻³")
# %%
#ambient and dry histogram 

# dry_mass_values_ug = [entry['Dry Mass (µg/m³)'] for entry in filtered_mass_values_ug]
# hydrated_mass_values_ug = [entry['Mass (µg/m³)'] for entries in ambient_mass_dict.values() for entry in entries]

bins = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 
                 16384, 32768, 65536, 131072])  

plt.figure(figsize=(10, 6))
plt.hist(filtered_mass_values_ug_10, bins=bins, color='blue', alpha=0.6, edgecolor='black', label="Dry Mass 10um", density=False)
plt.hist(filtered_mass_values_ug_inf, bins=bins, color='red', alpha=0.6, edgecolor='black', label="Dry Mass full", density=False)
plt.xscale('log')  
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
#%%
#just the dry histogram 
bins = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256]) 
plt.figure(figsize=(10, 6))
plt.hist(filtered_mass_values_ug_inf, bins=bins, color='red', alpha=0.6, edgecolor='black', density=False)
plt.xscale('log')  
plt.xlabel('Dry Mass (µg/m³)', fontsize=19, fontweight='bold')
plt.ylabel('Frequency', fontsize=19, fontweight='bold')
plt.title('CAS dry mass', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.tight_layout()
plt.legend()
plt.show()
#%%
# #1 Hz mass
# def exponential(d, n0, D):
#     return n0 * np.exp(-d / D)

# rho_salt = 2200  # kg/m³

# def calculate_mass(N0, D):
#     N0_m4 = N0 * 1e6  # cm⁻³µm⁻¹ → m⁻⁴
#     integrand = lambda d: np.exp(-d / D) * (d * 1e-6)**3
#     mass_integral, _ = quad(integrand, 2, np.inf)
#     return (np.pi / 6) * rho_salt * N0_m4 * mass_integral  # returns kg/m³
# mass_val = calculate_mass(n0, D) * 1e9  # convert kg/m³ → µg/m³

# per_second_mass = []

# for i in range(len(dates_legs)):
#     date = dates_legs[i]
#     CAS_flight = CAS[i]

#     CAS_times = np.array(CAS_flight['Time_mid'], dtype=float)
#     bins = {f'CAS_Bin{b:02d}': np.array(CAS_flight[f'CAS_Bin{b:02d}'], dtype=float) 
#             for b in range(12, 30)}
#     bin_centers = np.array(bin_center)  # your CAS bin centers in µm

#     for t_idx, t in enumerate(CAS_times):
#         # grab spectrum at this second
#         dN_dD = np.array([bins[f'CAS_Bin{b:02d}'][t_idx] for b in range(12, 30)])
#         valid = (~np.isnan(dN_dD)) & (dN_dD > 0)

#         if np.sum(valid) < 3:
#             continue  # skip seconds with no signal

#         try:
#             popt, _ = curve_fit(exponential, bin_centers[valid], dN_dD[valid], 
#                                 p0=(1, 5), maxfev=2000)
#             n0, D = popt
#             if D < 0.5 or D > 20:  # sanity check
#                 continue

#             mass_val = calculate_mass(n0, D)

#             per_second_mass.append({
#                 'Date': date,
#                 'Time': t,
#                 'Dry_Intercept_n0': n0,
#                 'Dry_E_folding_D': D,
#                 'Dry_Mass_ugm3': mass_val
#             })
#         except RuntimeError:
#             continue

# # %%
# #Choosing 3 new cases 

# rho_salt = 2200  # kg/m³

# def calculate_mass(N0, D):
#     """Compute dry mass using exponential fit parameters."""
#     N0_m4 = N0 * 10**6  # Convert cm⁻³µm⁻¹ to m⁻⁴
#     integrand = lambda d: np.exp(-d / D) * (d * 1e-6)**3  # Convert µm³ → m³
#     mass_integral, _ = quad(integrand, 2, np.inf)  # Integrate from 2µm to ∞
#     return (np.pi / 6) * rho_salt * N0_m4 * mass_integral  

# dry_mass_data_inf = []

# for entry in dry_exponential_fits:
#     date = entry['Date']
#     dry_intercept = entry['Dry_Intercept_n0']
#     dry_slope = entry['Dry_E_folding_D']

#     if dry_slope > 0 and dry_intercept > 0:
#         mass_value = calculate_mass(dry_intercept, dry_slope) * 1e9  # Convert kg/m³ to µg/m³
#         dry_mass_data_inf.append({
#             'Date': date,
#             'Dry Slope (D)': dry_slope,
#             'Dry Intercept (N0)': dry_intercept,
#             'Dry Mass (µg/m³)': mass_value
#         })

# dry_slopes = np.array([entry['Dry Slope (D)'] for entry in dry_mass_data_inf])
# dry_intercepts = np.array([entry['Dry Intercept (N0)'] for entry in dry_mass_data_inf])

# min_slope_threshold = np.percentile(dry_slopes, 1)  
# filtered_slopes = dry_slopes[dry_slopes >= min_slope_threshold]
# filtered_intercepts = dry_intercepts[dry_slopes >= min_slope_threshold]

# x_min, x_max = np.percentile(filtered_slopes, [5, 95])
# y_min, y_max = np.percentile(filtered_intercepts, [5, 95])

# xgrid_adjusted = np.logspace(np.log10(x_min), np.log10(x_max), 200)
# ygrid_adjusted = np.logspace(np.log10(y_min), np.log10(y_max), 200)
# D_grid_adjusted, dryintercept_grid_adjusted = np.meshgrid(xgrid_adjusted, ygrid_adjusted)

# mass_grid_adjusted = np.zeros_like(D_grid_adjusted)
# for i in range(D_grid_adjusted.shape[0]):
#     for j in range(D_grid_adjusted.shape[1]):
#         mass_grid_adjusted[i, j] = calculate_mass(dryintercept_grid_adjusted[i, j], D_grid_adjusted[i, j]) * 1e9

# mass_levels = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]

# plt.figure(figsize=(10, 8))
# plt.scatter(filtered_slopes, filtered_intercepts, c='blue', s=80, alpha=0.7)

# contour_plot = plt.contour(D_grid_adjusted, dryintercept_grid_adjusted, mass_grid_adjusted, 
#                            levels=mass_levels, colors='red', alpha=0.75, linewidths=1.5)

# plt.clabel(contour_plot, inline=True, fontsize=13, fmt=lambda x: f"{int(x)} µg/m³", colors='black', inline_spacing=5)
# for txt in contour_plot.labelTexts:
#     txt.set_fontweight('bold')
#     txt.set_rotation(15)

# highlight_points = [
#     (0.7, 10),   # Same slope, high intercept
#     (0.7, 2),  # Same slope, lower intercept
#     (1, 3)     # Different slope, different intercept
# ]

# for x, y in highlight_points:
#     plt.scatter(x, y, s=250, color='lime', marker='*', edgecolors='black', linewidth=1.5)

# plt.xlabel(r'Dry Slope ($\mu$m)', fontsize=19, fontweight='bold')
# plt.ylabel(r'Dry Intercept (cm$^{-3}$ $\mu$m$^{-1}$)', fontsize=19, fontweight='bold')
# plt.title('CAS Below Cloud Base January - June 2022\nContours of Dry Mass', fontsize=19, fontweight='bold')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)

# plt.xticks(fontsize=16, fontweight='bold')
# plt.yticks(fontsize=16, fontweight='bold')
# plt.legend()
# plt.tight_layout()
# plt.show()

# %%
#Finding our new cases 

# slope_col = "Dry Slope (D)"
# intercept_col = "Dry Intercept (N0)"
# mass_col = "Dry Mass (µg/m³)"

# slope_array = np.array([entry[slope_col] for entry in filtered_dry_mass_inf]).reshape(-1, 1)
# intercept_array = np.array([entry[intercept_col] for entry in filtered_dry_mass_inf]).reshape(-1, 1)
# data_points = np.column_stack((slope_array, intercept_array))

# def find_closest_match(target_slope, target_intercept):
#     """Find the closest real data point to a target (Slope, Intercept)."""
#     distances = np.linalg.norm(data_points - np.array([target_slope, target_intercept]), axis=1)
#     closest_index = distances.argmin()
#     return filtered_dry_mass_inf[closest_index]

# target_cases = [
#     (0.7, 10),  # Same slope, high intercept
#     (0.7, 2),   # Same slope, lower intercept
#     (1.2, 3)      # Different slope, different intercept
# ]

# closest_matches = []
# for D, N0 in target_cases:
#     closest_match = find_closest_match(D, N0)
#     closest_matches.append(closest_match)
#     print(f"Target (Slope={D}, Intercept={N0}) → Closest Match: "
#           f"Slope={closest_match[slope_col]}, Intercept={closest_match[intercept_col]}, "
#           f"Mass (µg/m³)={closest_match[mass_col]}")

# df_closest_matches = pd.DataFrame(closest_matches)
# print("\nClosest Matches for Target Cases:")
# print(df_closest_matches)

# #%%
# #Pull the concentrations 
# # Define target cases with matched Date, Start, Stop
# target_cases = [
#     {'Date': '2022-01-26', 'BCB_start': 51422, 'BCB_stop': 51621, 'Slope (D)': 0.6969, 'Intercept (N0)': 9.9214},
#     {'Date': '2022-01-24', 'BCB_start': 69488, 'BCB_stop': 69692, 'Slope (D)': 0.7387, 'Intercept (N0)': 2.0512},
#     {'Date': '2022-02-03', 'BCB_start': 72639, 'BCB_stop': 72800, 'Slope (D)': 1.1759, 'Intercept (N0)': 3.0752}
# ]

# # # Function to find and extract concentration from Y_BCB_calc_cm3
# # def find_concentration(date, start, stop):
# #     """Retrieve concentration from Y_BCB_calc_cm3 matching Date, Start, and Stop."""
# #     for entry in total_concentration_cm3:
# #         if (entry['Date'] == date and entry['BCB_start'] == start and entry['BCB_stop'] == stop):
# #             return entry['Total_Y_Concentration_cm3']  # Assuming 'Concentration' stores the values
# #     return np.nan  # Return NaN if not found

# # # Add concentration to target cases
# for case in target_cases:
#     concentration = find_concentration(case['Date'], case['BCB_start'], case['BCB_stop'])
#     case['Total_Y_Concentration_cm3'] = concentration  # Add concentration to dictionary

# # Convert to DataFrame for display
# df_target_cases = pd.DataFrame(target_cases)

# # Print the updated cases
# print("\nUpdated Target Cases with Concentrations:")
# print(df_target_cases)


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
# #Size distributions for these 3 cases

# def exponential(x, n0, D):
#     return n0 * np.exp(-x / D)

# target_cases = {
#     ('2022-01-26', 51422, 51621): {"Dry Mass (µg/m³)": 10.936, "Total Concentration (cm⁻³)": 0.9967, "Color": "purple", "LineStyle": "solid"},
#     ('2022-01-24', 69488, 69692): {"Dry Mass (µg/m³)": 3.007, "Total Concentration (cm⁻³)": 0.2284, "Color": "navy", "LineStyle": "solid"},
#     ('2022-02-03', 72639, 72800): {"Dry Mass (µg/m³)": 36.850, "Total Concentration (cm⁻³)": 0.8043, "Color": "orange", "LineStyle": "solid"},
# }

# dry_exponential_fits = []

# plt.figure(figsize=(8, 6))

# legend_labels = []  

# for entry in filtered_master_BCB_ddry:
#     ddry_values = np.array(entry['ddry'])
#     dN_dD_dry = np.array(entry['dN/dDdry'])

#     valid_indices = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry)
#     if np.sum(valid_indices) < 5:
#         continue

#     try:
#         popt, _ = curve_fit(exponential, ddry_values[valid_indices], dN_dD_dry[valid_indices], p0=(1, 5), maxfev=5000)
#         n0, D = popt

#         dry_exponential_fits.append({
#             'Date': entry['Date'],
#             'BCB_start': entry['BCB_start'],
#             'BCB_stop': entry['BCB_stop'],
#             'Dry_Intercept_n0': n0,
#             'Dry_E_folding_D': D
#         })

#         x_fit = np.linspace(min(ddry_values[valid_indices]), max(ddry_values[valid_indices]), 100)
#         y_fit = exponential(x_fit, *popt)

#         case_key = (entry['Date'], entry['BCB_start'], entry['BCB_stop'])
#         if case_key in target_cases:
#             case_info = target_cases[case_key]
#             dry_mass = case_info["Dry Mass (µg/m³)"]
#             total_concentration = case_info["Total Concentration (cm⁻³)"]
#             color = case_info["Color"]
#             linestyle = case_info["LineStyle"]

#             legend_label = f"{entry['Date']} | Mass: {dry_mass:.2f} µg/m³ | Conc: {total_concentration:.4f} cm⁻³"
#             legend_labels.append((legend_label, color, linestyle))

#             plt.plot(x_fit, y_fit, color=color, linewidth=3.5, linestyle=linestyle)

#         else:
#             plt.plot(x_fit, y_fit, color='gray', alpha=0.05, linewidth=1)  # Background curves more transparent

#     except RuntimeError:
#         print(f"Fit could not be performed for date {entry['Date']}")

# plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=14, fontweight="bold")
# plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
# plt.yscale("log")
# plt.ylim(1e-7, 1e1)
# plt.xlim(0, 25) 

# plt.xticks(fontweight="bold", fontsize=14)
# plt.yticks(fontweight="bold", fontsize=14)
# plt.title("Below Cloud Base January - June 2022\nFitted Dry Size Distributions", fontsize=14, fontweight="bold")
# handles = [plt.Line2D([0], [0], color=color, linewidth=3.5, linestyle=linestyle, label=label)
#            for label, color, linestyle in legend_labels]
# plt.legend(handles=handles, loc='upper right', fontsize=10)

# plt.show()

# print(f"Total successful dry exponential fits: {len(dry_exponential_fits)}")
#%%
# #only to 10 um d 

# common_bins=(2, 10, 10)
# # Define the exponential function
# def exponential(x, n0, D):
#     return n0 * np.exp(-x / D)

# # Target cases for highlighting
# target_cases = {
#     ('2022-01-26', 51422, 51621): {"Dry Mass (µg/m³)": 10.936, "Total Concentration (cm⁻³)": 0.9967, "Color": "purple", "LineStyle": "solid"},
#     ('2022-01-24', 69488, 69692): {"Dry Mass (µg/m³)": 3.007, "Total Concentration (cm⁻³)": 0.2284, "Color": "navy", "LineStyle": "solid"},
#     ('2022-02-03', 72639, 72800): {"Dry Mass (µg/m³)": 36.850, "Total Concentration (cm⁻³)": 0.8043, "Color": "orange", "LineStyle": "solid"},
# }

# dry_exponential_fits_ = []

# plt.figure(figsize=(8, 6))
# legend_labels = []

# for entry in filtered_master_BCB_ddry:
#     ddry_values = np.array(entry['ddry'])
#     dN_dD_dry = np.array(entry['dN/dDdry'])

#     # **Filter to only include values ≤ 10 µm before fitting**
#     valid_indices = (ddry_values <= 10) & ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry)
#     ddry_values_10um = ddry_values[valid_indices]
#     dN_dD_dry_10um = dN_dD_dry[valid_indices]

#     # Ensure there are enough valid points for fitting (min 5 points)
#     if len(ddry_values_10um) < 5:
#         continue

#     try:
#         # Fit the exponential **only using data ≤ 10 µm**
#         popt, _ = curve_fit(exponential, ddry_values_10um, dN_dD_dry_10um, 
#                             p0=(max(dN_dD_dry_10um), 5), maxfev=5000)
#         n0, D = popt

#         dry_exponential_fits.append({
#             'Date': entry['Date'],
#             'BCB_start': entry['BCB_start'],
#             'BCB_stop': entry['BCB_stop'],
#             'Dry_Intercept_n0': n0,
#             'Dry_E_folding_D': D
#         })

#         # Generate fitted curve **only up to 10 µm**
#         x_fit = np.linspace(min(ddry_values_10um), 10, 100)
#         y_fit = exponential(x_fit, *popt)

#         # Check if this entry is in the highlighted cases
#         case_key = (entry['Date'], entry['BCB_start'], entry['BCB_stop'])
#         if case_key in target_cases:
#             case_info = target_cases[case_key]
#             dry_mass = case_info["Dry Mass (µg/m³)"]
#             total_concentration = case_info["Total Concentration (cm⁻³)"]
#             color = case_info["Color"]
#             linestyle = case_info["LineStyle"]

#             legend_label = f"{entry['Date']} | Mass: {dry_mass:.2f} µg/m³ | Conc: {total_concentration:.4f} cm⁻³"
#             legend_labels.append((legend_label, color, linestyle))

#             plt.plot(x_fit, y_fit, color=color, linewidth=3.5, linestyle=linestyle)

#         else:
#             plt.plot(x_fit, y_fit, color='gray', alpha=0.05, linewidth=1)  # Background curves more transparent

#     except RuntimeError:
#         print(f"Fit could not be performed for date {entry['Date']}")

# # Formatting
# plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=14, fontweight="bold")
# plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
# plt.yscale("log")
# plt.ylim(1e-7, 1e1)
# plt.xlim(0, 10)  # Ensure the plot is limited to ≤ 10 µm

# plt.xticks(fontweight="bold", fontsize=14)
# plt.yticks(fontweight="bold", fontsize=14)
# plt.title("Below Cloud Base January - June 2022\nFitted Dry Size Distributions (≤10 µm)", fontsize=14, fontweight="bold")
# handles = [plt.Line2D([0], [0], color=color, linewidth=3.5, linestyle=linestyle, label=label)
#            for label, color, linestyle in legend_labels]
# plt.legend(handles=handles, loc='upper right', fontsize=10)

# plt.show()

# print(f"Total successful dry exponential fits: {len(dry_exponential_fits)}")


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
            winds9_mean = np.nan
            alts9_mean = np.nan
        if index1_end == None:
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
Z0 = 0.02
Z10 = 10 

corrected_calc_bcb = {'Date': [], 'Corrected_bcb_windspeed': []}

for flight in master_BCB:
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
corrected_windspeeds = corrected_calc_bcb['Corrected_bcb_windspeed']
corrected_windspeeds = [ws for ws in corrected_windspeeds if not np.isnan(ws)]
mean_corrected_windspeed = np.mean(corrected_windspeeds)
print(f"Mean Corrected Wind Speed: {mean_corrected_windspeed:.2f} m/s")

#%%
#standard deviation of windspeed
mean_corrected_windspeed = sum(corrected_windspeeds) / len(corrected_windspeeds)
variance = sum((ws - mean_corrected_windspeed) ** 2 for ws in corrected_windspeeds) / (len(corrected_windspeeds) - 1)
std_corrected_windspeed = variance ** 0.5
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
#monthly trend of corrected windspeed
df_wind = pd.DataFrame(combined_data).copy()
df_wind = df_wind[df_wind["Date"].astype(str).str.startswith("2022-")].copy()
df_wind["Month"] = df_wind["Date"].astype(str).str[5:7].astype(int)
df_wind = df_wind[df_wind["Month"].between(1, 6)].copy()
df_wind_sorted = df_wind.sort_values(["Date", "BCB_start"], kind="mergesort").reset_index(drop=True)
wind = df_wind_sorted["Windspeed"].astype(float).values
x = np.arange(len(df_wind_sorted))
plt.figure(figsize=(12, 4.8))
plt.plot(x, wind, '-')
plt.grid(alpha=0.3)
plt.xlabel("Leg index (sorted by Date, then BCB_start)", fontsize=13, fontweight="bold")
plt.ylabel("Corrected Wind Speed (m/s)", fontsize=13, fontweight="bold")
plt.title("Corrected Wind Speed Timeline (Jan–Jun 2022)\nLegs ordered by Date then BCB_start",
          fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
#%%
#color coded by month with
month_name = {1:"January", 2:"February", 3:"March", 5:"May", 6:"June"}
plt.figure(figsize=(12, 4.8))
for m in sorted(df_wind_sorted["Month"].unique()):
    if m not in month_name:
        continue
    m_mask = (df_wind_sorted["Month"].values == m)
    vals = wind[m_mask]
    good = np.isfinite(vals)
    mean_val = np.mean(vals[good]) if np.any(good) else np.nan
    plt.plot(
        x[m_mask],
        wind[m_mask],
        '-',
        label=f"{month_name[m]} (mean: {mean_val:.2f} m/s)"
    )
plt.grid(alpha=0.3)
plt.ylabel("Wind Speed (m/s)", fontsize=16, fontweight="bold")
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.title("BCB Wind Speed\nJanuary–June 2022 Monthly Means",
          fontsize=18, fontweight="bold")
plt.legend(ncol=2, fontsize=10)
plt.yticks(fontsize=14, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
#%%
dfp = pd.DataFrame(combined_data).copy()
dfp = dfp[dfp["Date"].astype(str).str.startswith("2022-")].copy()
dfp["Month"] = dfp["Date"].astype(str).str[5:7].astype(int)
dfp = dfp[dfp["Month"].between(1, 6)].copy()
dfp["Date_dt"] = pd.to_datetime(dfp["Date"])
sort_cols = ["Date_dt"]
if "BCB_start" in dfp.columns:
    sort_cols.append("BCB_start")
elif "Min_start" in dfp.columns:
    sort_cols.append("Min_start")
dfp = dfp.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
x = np.arange(len(dfp))
wind_arr = pd.to_numeric(dfp["Windspeed"], errors="coerce").to_numpy()
date_first = dfp.groupby(dfp["Date_dt"].dt.date, sort=False).head(1)
tick_pos = date_first.index.to_numpy()
tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
month_name = {1:"January", 2:"February", 3:"March", 4:"April", 5:"May", 6:"June"}
fig_w = max(22, 0.55 * len(tick_pos))
fig, ax = plt.subplots(figsize=(fig_w, 6.2))

for m in sorted(dfp["Month"].unique()):
    if m not in month_name:
        continue
    m_mask = (dfp["Month"].values == m)
    mean_w = np.nanmean(wind_arr[m_mask])
    ax.plot(
        x[m_mask], wind_arr[m_mask],
        '-', linewidth=1.5,
        label=f"{month_name[m]} (mean: {mean_w:.2f} m/s)"
    )
for p in tick_pos:
    ax.axvline(p, color="k", alpha=0.06, linewidth=1)

ax.grid(alpha=0.3)
ax.set_ylabel("Corrected Wind Speed (m/s)", fontsize=16, fontweight="bold")
ax.set_xlabel("Flight Date", fontsize=16, fontweight="bold")
ax.set_title("BCB Corrected Wind Speed January–June 2022\nMonthly Trend",
             fontsize=18, fontweight="bold")
ax.legend(ncol=2, fontsize=10, loc="upper right")
ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_lab, rotation=60, ha="right", fontsize=7, fontweight="bold")
labels = ax.get_xticklabels()
for i, lab in enumerate(labels):
    lab.set_text("\n" * (i % 4) + lab.get_text())
ax.set_xticklabels([lab.get_text() for lab in labels])
fig.subplots_adjust(bottom=0.40)
fig.tight_layout()
plt.show()

#%%
#%%
month_colors = {
    1: "tab:blue",    
    2: "tab:orange",  
    3: "tab:green",   
    5: "tab:red",     
    6: "tab:purple"   
}

dfp = df_wind.copy()
dfp["Date_dt"] = pd.to_datetime(dfp["Date"])
if "Month" not in dfp.columns:
    dfp["Month"] = dfp["Date"].astype(str).str[5:7].astype(int)
sort_cols = ["Date_dt"]
if "BCB_start" in dfp.columns:
    sort_cols.append("BCB_start")
elif "Min_start" in dfp.columns:
    sort_cols.append("Min_start")
dfp = dfp.sort_values(sort_cols).reset_index(drop=True)
x = np.arange(len(dfp))
wind_arr = pd.to_numeric(dfp["Windspeed"], errors="coerce").to_numpy()
date_first = dfp.groupby(dfp["Date_dt"].dt.date).head(1)
tick_pos = date_first.index.to_numpy()
tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
fig_w = max(22, 0.55 * len(tick_pos))
fig, ax = plt.subplots(figsize=(fig_w, 6.2))
legend_handles = []
month_name = {1:"January", 2:"February", 3:"March", 4:"April", 5:"May", 6:"June"}
for m in sorted(dfp["Month"].unique()):
    if m not in month_name:
        continue

    m_mask = (dfp["Month"].values == m)
    if not np.any(m_mask):
        continue
    c = month_colors.get(m, "k")

    mean_w   = np.nanmean(wind_arr[m_mask])
    median_w = np.nanmedian(wind_arr[m_mask])
    ax.plot(
        x[m_mask], wind_arr[m_mask],
        '-', linewidth=1.5, color=c
    )

    month_x = x[m_mask]
    mid_x = month_x[len(month_x) // 2]
    ax.plot(
        mid_x, mean_w,
        marker="^", linestyle="None",
        markersize=13,
        markerfacecolor=c,
        markeredgecolor=c,   
        markeredgewidth=2.2,
        zorder=6
    )
    ax.plot(
        mid_x + 5, median_w,
        marker="o", linestyle="None",
        markersize=12,
        markerfacecolor=c,
        markeredgecolor=c,  
        markeredgewidth=2.2,
        zorder=6
    )
    legend_handles.extend([
    Line2D([0], [0], color=c, lw=2, label=month_name[m]),
    Line2D([0], [0], marker="^", lw=0, markersize=10,
           markerfacecolor=c, markeredgecolor=c,
           label=f"{month_name[m]} mean = {mean_w:.2f} m/s"),
    Line2D([0], [0], marker="o", lw=0, markersize=10,
           markerfacecolor=c, markeredgecolor=c,
           label=f"{month_name[m]} median = {median_w:.2f} m/s"),
])

for p in tick_pos:
    ax.axvline(p, color="k", alpha=0.06, linewidth=1)
ax.grid(alpha=0.3)
plt.yticks(fontsize=16, fontweight="bold")
ax.set_ylabel("Wind Speed (m s$^{-1}$)", fontsize=20, fontweight="bold")
ax.set_xlabel("Flight Date", fontsize=20, fontweight="bold")
ax.set_title("Wind Speed Janury–June 2022\nMonthly Trend",
             fontsize=20, fontweight="bold")

ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_lab, rotation=60, ha="right", fontsize=10, fontweight="bold")
labels = ax.get_xticklabels()
for i, lab in enumerate(labels):
    base = lab.get_text()
    lab.set_text("\n" * (i % 4) + base)
ax.set_xticklabels([lab.get_text() for lab in labels])
target_dates = {"2022-06-05": 0, "2022-06-07": 3}
labels = ax.get_xticklabels()
for lab in labels:
    txt = lab.get_text().replace("\n", "")
    if txt in target_dates:
        lab.set_text("\n" * target_dates[txt] + txt)
ax.set_xticklabels([lab.get_text() for lab in labels])
ax.legend(
    handles=legend_handles,
    ncol=1,
    fontsize=9,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=True
)
fig.subplots_adjust(bottom=0.40)
fig.tight_layout()
plt.show()
#save to pdf
fig.savefig("Wind_Speed_Monthly_Trend.pdf", bbox_inches="tight")
#%%
common_bins=np.linspace(2, 10, 25)
#%%
#6 windspeed bins
windspeed_bins = [
    (0, 2.5),
    (2.501, 3.5),
    (3.501, 5),
    (5.001, 7),
    (7.001, 9),
    (9.001, np.inf)
]
grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}
missing_windspeed_count = 0
common_bins = np.linspace(2, 10, 10)
for entry in dry_exponential_fits_10:
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    n0 = entry['Dry_Intercept_n0']
    D = entry['Dry_E_folding_D']
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) & 
        (df_combined['BCB_start'] == BCB_start) & 
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    if windspeed_entry.empty or np.isnan(n0) or np.isnan(D):
        missing_windspeed_count += 1
        continue

    windspeed = windspeed_entry['Windspeed'].values[0]

    size_dist = n0 * np.exp(-common_bins / D)

    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:
            grouped_distributions[idx].append(size_dist)
            mean_windspeeds[idx].append(windspeed)
            break
for idx, group in grouped_distributions.items():
    print(f"Windspeed bin {idx} ({windspeed_bins[idx]} m/s): {len(group)} legs")

print(f"Total legs with missing windspeed data: {missing_windspeed_count}")

plt.figure(figsize=(10, 8))
for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)
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
bin_means = {}
bin_stds = {}
bin_stderrs = {}

for idx, data in grouped_distributions.items():
    if data:
        data_array = np.array(data)
        bin_means[idx] = np.mean(data_array, axis=0) 
        bin_stds[idx] = np.std(data_array, axis=0) 
        bin_stderrs[idx] = bin_stds[idx] / np.sqrt(len(data)) 
plt.figure(figsize=(10, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if idx in bin_means:
        avg_distribution = bin_means[idx]
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])
        error_bars = bin_stderrs[idx]
        plt.errorbar(
            common_bins, avg_distribution, yerr=error_bars, 
            label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", 
            linewidth=2.5, capsize=3, capthick=1.5, fmt='-o', markersize=4
        )
plt.yscale('log')
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=22, fontweight="bold")
plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=22, fontweight="bold")
plt.title('Dry Size Distributions Binned by Average Wind Speed', fontweight='bold', fontsize=19)
plt.legend(title=r"Average wind speed m s$^{-1}$")
plt.tight_layout()
plt.ylim(1e-4, 10**0)
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.show()
#%%
legend_texts = []
for idx, (low, high) in enumerate(windspeed_bins):
    if idx in bin_means:
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])
        avg_std_error = np.mean(bin_stderrs[idx]) 
        legend_texts.append(f"{avg_windspeed:.1f} m/s, n={num_legs} legs\nAvg SE: {avg_std_error:.3f}")
plt.figure(figsize=(10, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if idx in bin_means:
        avg_distribution = bin_means[idx]
        error_bars = bin_stderrs[idx] 

        plt.errorbar(
            common_bins, avg_distribution, yerr=error_bars,
            label=legend_texts[idx], linewidth=2.5, capsize=3, capthick=1.5, fmt='-o', markersize=4
        )
plt.yscale('log')
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=23, fontweight="bold")
plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=23, fontweight="bold")
plt.title('CAS Dry Size Distributions \nBinned by Average Wind Speed', fontweight='bold', fontsize=23)
plt.legend(
    title="Windspeed & Error Stats", title_fontsize=20, fontsize=20,
    prop={'weight': 'bold'}, loc="upper right"
)

plt.tight_layout()
plt.ylim(1e-4, 10**0)
plt.xlim(0,10)
plt.xticks(fontsize=21, fontweight='bold')
plt.yticks(fontsize=21, fontweight='bold')
plt.show()
#%%
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
avg_wind_speeds = []
se_wind_speeds = []
bin_leg_counts = []
for idx in range(len(windspeed_bins)):
    windspeeds = mean_windspeeds.get(idx, [])
    if windspeeds:
        windspeeds = np.array(windspeeds)
        avg_ws = np.mean(windspeeds)
        se_ws = np.std(windspeeds, ddof=1) / np.sqrt(len(windspeeds))

        avg_wind_speeds.append(avg_ws)
        se_wind_speeds.append(se_ws)
        bin_leg_counts.append(len(windspeeds))
plt.figure(figsize=(8, 6))

for idx, (x, y) in enumerate(zip(avg_wind_speeds, se_wind_speeds)):
    plt.errorbar(x, y, yerr=0.0, fmt='o', color=colors[idx], markersize=10,
                 ecolor='black', capsize=5, label=f'{x:.1f} m/s')
    plt.text(x, y + 0.02, f'n={bin_leg_counts[idx]}', fontsize=12,
             color=colors[idx], ha='center', fontweight='bold')

plt.xlabel("Average Wind Speed (m s$^{-1}$)", fontsize=20, fontweight="bold")
plt.ylabel("Standard Error (m s$^{-1}$)", fontsize=20, fontweight="bold")
plt.title("Standard Error of Wind Speed", fontsize=20, fontweight="bold")
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=20, fontweight='bold')
plt.yticks(fontsize=20, fontweight='bold')
plt.ylim(0, max(se_wind_speeds) + 0.05)
plt.xlim(0, 12)
plt.tight_layout()
plt.show()
# %%
windspeed_bins = [
    (0, 2.5),
    (2.501, 3.5),
    (3.501, 5),
    (5.001, 7),
    (7.001, 9),
    (9.001, np.inf)
]
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown'] 
bin_edges = np.linspace(2, 10, 11)  
bin_widths = np.diff(bin_edges) 
avg_windspeeds = []
total_concentrations = []

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:  
        avg_windspeed = np.mean(mean_windspeeds[idx]) 

        avg_concentration_per_leg = [np.sum(dist * bin_widths) for dist in grouped_distributions[idx]]
        avg_concentration = np.mean(avg_concentration_per_leg) 
        avg_windspeeds.append(avg_windspeed)
        total_concentrations.append(avg_concentration)

windspeed_values = np.array(avg_windspeeds)
total_concentrations = np.array(total_concentrations)
def linear_model(x, m, b):
    return m * x + b
popt, _ = curve_fit(linear_model, windspeed_values, total_concentrations)
m_fit, b_fit = popt
residuals = total_concentrations - linear_model(windspeed_values, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((total_concentrations - np.mean(total_concentrations))**2)
r_squared = 1 - (ss_res / ss_tot)
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


legend_labels = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[idx], markersize=10, 
               label=f"{windspeed_values[idx]:.1f} m/s") for idx in range(len(windspeed_bins))
]
plt.legend(handles=legend_labels + [plt.Line2D([0], [0], color='red', label=f'Fit: y = {m_fit:.3f}x + {b_fit:.3f}, R² = {r_squared:.2f}')], 
           title="Wind Speed Bins", title_fontsize=14, fontsize=13)
plt.tight_layout()
plt.xlim(0,12)
plt.ylim(0.1, 0.7)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.show()
print(f"Slope (m): {m_fit:.3f}")
print(f"Intercept (b): {b_fit:.3f}")
print(f"R² value: {r_squared:.2f}")
#%%
#adding error bars to total concentration vs wind speed 
windspeed_bins = [
    (0, 2.5),
    (2.501, 3.5),
    (3.501, 5),
    (5.001, 7),
    (7.001, 9),
    (9.001, np.inf)
]
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown'] 

bin_edges = np.linspace(2, 10, 11) 
bin_widths = np.diff(bin_edges)  


avg_windspeeds = []
total_concentrations = []
standard_errors = []

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:  
        avg_windspeed = np.mean(mean_windspeeds[idx]) 

        avg_concentration_per_leg = [np.sum(dist * bin_widths) for dist in grouped_distributions[idx]]
        avg_concentration = np.mean(avg_concentration_per_leg)  
        std_concentration = np.std(avg_concentration_per_leg, ddof=1)  
        N_legs = len(avg_concentration_per_leg) 
        SE_concentration = std_concentration / np.sqrt(N_legs) 

        avg_windspeeds.append(avg_windspeed)
        total_concentrations.append(avg_concentration)
        standard_errors.append(SE_concentration)

windspeed_values = np.array(avg_windspeeds)
total_concentrations = np.array(total_concentrations)
standard_errors = np.array(standard_errors)
def linear_model(x, m, b):
    return m * x + b
popt, _ = curve_fit(linear_model, windspeed_values, total_concentrations)
m_fit, b_fit = popt
residuals = total_concentrations - linear_model(windspeed_values, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((total_concentrations - np.mean(total_concentrations))**2)
r_squared = 1 - (ss_res / ss_tot)
plt.figure(figsize=(8, 6))
for idx in range(len(windspeed_bins)):
    plt.errorbar(windspeed_values[idx], total_concentrations[idx], 
                 yerr=standard_errors[idx], fmt='o', color=colors[idx], 
                 markersize=10, capsize=5, capthick=2, label=f"{windspeed_values[idx]:.1f} m/s", 
                 ecolor='black', elinewidth=1.5, zorder=3)
x_fit = np.linspace(min(windspeed_values), max(windspeed_values), 100)
y_fit = linear_model(x_fit, *popt)
plt.plot(x_fit, y_fit, 'r-', label=f'Fit: y = {m_fit:.3f}x + {b_fit:.3f}, R² = {r_squared:.2f}')
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=16, fontweight='bold')
plt.ylabel("Total Wind Speed Bin Concentration (cm$^{-3}$)", fontsize=16, fontweight='bold')
plt.title("CAS Wind Speed and Total Concentration Correlation", fontsize=16, fontweight='bold')
plt.legend(title="Wind Speed Bins", title_fontsize=14, fontsize=13, loc='best', frameon=True)
plt.tight_layout()
plt.xlim(0, 12)
plt.ylim(0.1, 0.7)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.show()
print(f"Slope (m): {m_fit:.3f}")
print(f"Intercept (b): {b_fit:.3f}")
print(f"R² value: {r_squared:.2f}")
#%%
#adding R value 
windspeed_bins = [
    (0, 2.5),
    (2.501, 3.5),
    (3.501, 5),
    (5.001, 7),
    (7.001, 9),
    (9.001, np.inf)
]
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
bin_edges = np.linspace(2, 10, 11)
bin_widths = np.diff(bin_edges) 
avg_windspeeds = []
total_concentrations = []
standard_errors = []

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        avg_windspeed = np.mean(mean_windspeeds[idx])

        avg_concentration_per_leg = [np.sum(dist * bin_widths) for dist in grouped_distributions[idx]]
        avg_concentration = np.mean(avg_concentration_per_leg)  
        std_concentration = np.std(avg_concentration_per_leg, ddof=1) 
        N_legs = len(avg_concentration_per_leg) 
        SE_concentration = std_concentration / np.sqrt(N_legs)

        avg_windspeeds.append(avg_windspeed)
        total_concentrations.append(avg_concentration)
        standard_errors.append(SE_concentration)

windspeed_values = np.array(avg_windspeeds)
total_concentrations = np.array(total_concentrations)
standard_errors = np.array(standard_errors)
def linear_model(x, m, b):
    return m * x + b
popt, _ = curve_fit(linear_model, windspeed_values, total_concentrations)
m_fit, b_fit = popt
residuals = total_concentrations - linear_model(windspeed_values, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((total_concentrations - np.mean(total_concentrations))**2)
r_squared = 1 - (ss_res / ss_tot)
r_value = np.sign(m_fit) * np.sqrt(r_squared)
plt.figure(figsize=(8, 6))
plt.errorbar(windspeed_values, total_concentrations, 
             yerr=standard_errors, fmt='o', color='blue', 
             markersize=10, capsize=5, capthick=2, label="CAS", 
             ecolor='black', elinewidth=1.5, zorder=3)
x_fit = np.linspace(min(windspeed_values), max(windspeed_values), 100)
y_fit = linear_model(x_fit, *popt)
plt.plot(x_fit, y_fit, 'r-', label=f'Fit: y = {m_fit:.3f}x + {b_fit:.3f}\nR² = {r_squared:.2f}, R = {r_value:.2f}')
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=16, fontweight='bold')
plt.ylabel("Total Wind Speed Bin Concentration (cm$^{-3}$)", fontsize=16, fontweight='bold')
plt.title("CAS Wind Speed and Total Concentration Correlation", fontsize=16, fontweight='bold')
plt.legend(fontsize=13, title_fontsize=14, loc='best', frameon=True)
plt.tight_layout()
plt.xlim(0, 12)
plt.ylim(0.1, 0.7)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.show()
print(f"Slope (m): {m_fit:.3f}")
print(f"Intercept (b): {b_fit:.3f}")
print(f"R² value: {r_squared:.2f}")
print(f"R value (Pearson correlation): {r_value:.3f}")
# %%
#2sigma
plt.figure(figsize=(8, 6))

plt.errorbar(windspeed_values, total_concentrations,
             yerr=2 * standard_errors, fmt='none',
             ecolor='lightgray', elinewidth=6, capsize=0, label='95% (±2σ)', zorder=1)

plt.errorbar(windspeed_values, total_concentrations, 
             yerr=standard_errors, fmt='o', color='blue', 
             markersize=10, capsize=5, capthick=2, label="CAS", 
             ecolor='black', elinewidth=1.5, zorder=3)
x_fit = np.linspace(min(windspeed_values), max(windspeed_values), 100)
y_fit = linear_model(x_fit, *popt)
plt.plot(x_fit, y_fit, 'r-', label=f'Fit: y = {m_fit:.3f}x + {b_fit:.3f}\nR² = {r_squared:.2f}, R = {r_value:.2f}')
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=20, fontweight='bold')
plt.ylabel("Total Wind Speed Bin \nConcentration (cm$^{-3}$)", fontsize=20, fontweight='bold')
plt.title("CAS Wind Speed and Total Concentration", fontsize=20, fontweight='bold')
plt.legend(fontsize=13, title_fontsize=14, loc='best', frameon=True)
plt.tight_layout()
plt.xlim(0, 12)
plt.ylim(0.1, 0.7)
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')
plt.show()
#%%
#adding slope uncertainty
windspeed_bins = [
    (0, 2.5),
    (2.501, 3.5),
    (3.501, 5),
    (5.001, 7),
    (7.001, 9),
    (9.001, np.inf)
]
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

bin_edges = np.linspace(2, 10, 11)
bin_widths = np.diff(bin_edges) 
avg_windspeeds = []
total_concentrations = []
standard_errors = []

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:  
        avg_windspeed = np.mean(mean_windspeeds[idx])
        avg_concentration_per_leg = [np.sum(dist * bin_widths) for dist in grouped_distributions[idx]]
        avg_concentration = np.mean(avg_concentration_per_leg)
        std_concentration = np.std(avg_concentration_per_leg, ddof=1)
        N_legs = len(avg_concentration_per_leg)
        SE_concentration = std_concentration / np.sqrt(N_legs)
        avg_windspeeds.append(avg_windspeed)
        total_concentrations.append(avg_concentration)
        standard_errors.append(SE_concentration)
windspeed_values = np.array(avg_windspeeds)
total_concentrations = np.array(total_concentrations)
standard_errors = np.array(standard_errors)
def linear_model(x, m, b):
    return m * x + b
popt, pcov = curve_fit(linear_model, windspeed_values, total_concentrations)
m_fit, b_fit = popt
perr = np.sqrt(np.diag(pcov))
m_err, b_err = perr
residuals = total_concentrations - linear_model(windspeed_values, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((total_concentrations - np.mean(total_concentrations))**2)
r_squared = 1 - (ss_res / ss_tot)
r_value = np.sign(m_fit) * np.sqrt(r_squared)
plt.figure(figsize=(8, 6))
plt.errorbar(windspeed_values, total_concentrations, 
             yerr=standard_errors, fmt='o', color='blue', 
             markersize=10, capsize=5, capthick=2, label="CAS", 
             ecolor='black', elinewidth=1.5, zorder=3)

x_fit = np.linspace(min(windspeed_values), max(windspeed_values), 100)
y_fit = linear_model(x_fit, *popt)
plt.plot(x_fit, y_fit, 'r-', 
         label=f'Fit: y = ({m_fit:.3f}±{m_err:.3f})x + {b_fit:.3f}\nR² = {r_squared:.2f}, R = {r_value:.2f}')

plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=16, fontweight='bold')
plt.ylabel("Total Wind Speed Bin Concentration (cm$^{-3}$)", fontsize=16, fontweight='bold')
plt.title("CAS Wind Speed and Total Concentration Correlation", fontsize=16, fontweight='bold')
plt.legend(fontsize=13, title_fontsize=14, loc='best', frameon=True)
plt.tight_layout()
plt.xlim(0, 12)
plt.ylim(0.1, 0.7)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.show()
print(f"Slope (m): {m_fit:.3f} ± {m_err:.3f}")
print(f"Intercept (b): {b_fit:.3f}")
print(f"R² value: {r_squared:.2f}")
print(f"R value (Pearson correlation): {r_value:.3f}")
plt.figure(figsize=(8, 6))
plt.errorbar(windspeed_values, total_concentrations,
             yerr=2 * standard_errors, fmt='none',
             ecolor='lightgray', elinewidth=6, capsize=0, label='95% (±2σ)', zorder=1)
plt.errorbar(windspeed_values, total_concentrations, 
             yerr=standard_errors, fmt='o', color='#4daf4a', 
             markersize=10, capsize=5, capthick=2, label="CAS", 
             ecolor='black', elinewidth=1.5, zorder=3)
plt.plot(x_fit, y_fit, '-', color='#984ea3', linewidth=2.5,
         label=f'Fit: y = ({m_fit:.3f}±{m_err:.3f})x + {b_fit:.3f}\nR² = {r_squared:.2f}, R = {r_value:.2f}')

plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=20, fontweight='bold')
plt.ylabel("Total Wind Speed Bin \nConcentration (cm$^{-3}$)", fontsize=20, fontweight='bold')
plt.title("CAS Below Cloud Base \n January-June 2022", fontsize=20, fontweight='bold')
plt.legend(
    fontsize=16,
    title_fontsize=14,
    loc='upper left',     
    frameon=False        
)
plt.tight_layout()
plt.xlim(0, 12)
plt.ylim(0, 1)
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')
plt.show()
#%%
#PDF of wind speed with 6 bins idicated 
# plt.figure(figsize=(8, 6))
# plt.hist(df_combined['Windspeed'], bins=30, density=True, alpha=0.5, color='blue', edgecolor='black', linewidth=1.2)
# plt.xlabel("10 m Wind Speed (m s$^{-1}$)", fontsize=18, fontweight='bold')
# plt.ylabel("Probability Density", fontsize=18, fontweight='bold')
# plt.title("10 m Wind Speeds\n Below Cloud Base\n January-June 2022", fontsize=18, fontweight='bold')
# for low, high in windspeed_bins:
#     plt.axvline(x=low, color='red', linestyle='--', linewidth=2)
# plt.axvline(x=windspeed_bins[-1][1], color='red', linestyle='--', linewidth=2)
# plt.tight_layout()
# plt.xlim(0, 16)
# plt.ylim(0, 0.2)
# plt.xticks(fontsize=18, fontweight='bold')
# plt.yticks(fontsize=18, fontweight='bold')
# plt.show()
#%%
#calculating counting error bars
bin_edges = np.linspace(2, 10, 11)       # μm
common_bins = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_widths_um = np.diff(bin_edges)
sample_area_cm2 = 0.0025          # CAS sample area in cm²
plane_speed_cm_s = 1.2e4          # 120 m/s = 12000 cm/s
sampling_time_s = 198             # 3.3 minutes
T = sampling_time_s
V = sample_area_cm2 * plane_speed_cm_s  # cm³/s
windspeed_bins = [
    (0, 2.5), (2.501, 3.5), (3.501, 5),
    (5.001, 7), (7.001, 9), (9.001, np.inf)
]

grouped_absolute_errors = {i: [] for i in range(len(windspeed_bins))}

for entry in dry_exponential_fits_10:
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    n0 = entry['Dry_Intercept_n0']
    D = entry['Dry_E_folding_D']

    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    if windspeed_entry.empty or np.isnan(n0) or np.isnan(D):
        continue

    windspeed = windspeed_entry['Windspeed'].values[0]
    n_i = n0 * np.exp(-common_bins / D)   # cm⁻³ μm⁻¹

    summation = np.nansum(n_i * (bin_widths_um**2))  # cm⁻³ · μm²
    concentration_error = np.sqrt(summation / (T * V))  # cm⁻³

    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:
            grouped_absolute_errors[idx].append(concentration_error)
            break
median_absolute_errors_per_bin = []
for idx, errors in grouped_absolute_errors.items():
    if errors:
        median_abs_err = np.nanmedian(errors)
    else:
        median_abs_err = np.nan
    median_absolute_errors_per_bin.append(median_abs_err)
    print(f"Bin {idx} ({windspeed_bins[idx]}): median abs error = {median_abs_err:.4f} cm⁻³")
avg_windspeeds = []
total_concentrations = []
standard_errors = []

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:  
        avg_windspeed = np.mean(mean_windspeeds[idx])
        avg_concentration_per_leg = [np.sum(dist * bin_widths_um) for dist in grouped_distributions[idx]]
        avg_concentration = np.mean(avg_concentration_per_leg)
        std_concentration = np.std(avg_concentration_per_leg, ddof=1)
        N_legs = len(avg_concentration_per_leg)
        SE_concentration = std_concentration / np.sqrt(N_legs)

        avg_windspeeds.append(avg_windspeed)
        total_concentrations.append(avg_concentration)
        standard_errors.append(SE_concentration)
windspeed_values = np.array(avg_windspeeds)
total_concentrations = np.array(total_concentrations)
standard_errors = 2 * np.array(standard_errors)           # now ±2 SE
counting_errors_CAS = 2 * np.array(median_absolute_errors_per_bin)  # now ±2σ counting error
def linear_model(x, m, b):
    return m * x + b

popt, pcov = curve_fit(linear_model, windspeed_values, total_concentrations)
m_fit, b_fit = popt
perr = np.sqrt(np.diag(pcov))
m_err, b_err = perr
residuals = total_concentrations - linear_model(windspeed_values, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((total_concentrations - np.mean(total_concentrations))**2)
r_squared = 1 - (ss_res / ss_tot)
r_value = np.sign(m_fit) * np.sqrt(r_squared)

x_fit = np.linspace(min(windspeed_values), max(windspeed_values), 100)
y_fit = linear_model(x_fit, *popt)
plt.figure(figsize=(8, 6))
plt.errorbar(windspeed_values, total_concentrations,
             yerr=standard_errors, fmt='o', color='black',
             ecolor='black', elinewidth=1.5, capsize=5, capthick=2,
             label="±2 Standard Errors", zorder=3)
plt.errorbar(windspeed_values - 0.2, total_concentrations,
             yerr=counting_errors_CAS,
             fmt='s', 
             markersize=4,
             markerfacecolor='#8c510a',
             markeredgecolor='black',
             ecolor='#8c510a',
             elinewidth=4,
             capsize=8,
             capthick=3,
             label="±2σ CAS Instrument Error",
             zorder=2)


plt.plot(x_fit, y_fit, '-', color='black', linewidth=2.5,
         label=f'Fit: y = ({m_fit:.3f}±{m_err:.3f})x + {b_fit:.3f}\nR² = {r_squared:.2f}, R = {r_value:.2f}')

plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=20, fontweight='bold')
plt.ylabel("Total Wind Speed Bin \nConcentration (cm$^{-3}$)", fontsize=20, fontweight='bold')
plt.title("CAS Below Cloud Base \n January–June 2022", fontsize=20, fontweight='bold')
plt.legend(fontsize=16, title_fontsize=14, loc='upper left', frameon=False)
plt.tight_layout()
plt.xlim(0, 12)
plt.ylim(0, 1)
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')
plt.show()
#%%
print("CAS concentration per-bin points:")
for i, (x, y, se2, ce2) in enumerate(zip(windspeed_values,
                                         total_concentrations,
                                         standard_errors,
                                         counting_errors_CAS)):
    print(f"  Bin{i}: WS={x:.2f} m/s, Conc={y:.3f} cm⁻³, ±2SE={se2:.3f}, ±2σCountErr≈{ce2:.3f}")
def linear_model(x, m, b):
    return m * x + b
mask = np.isfinite(windspeed_values) & np.isfinite(total_concentrations)
x = windspeed_values[mask]
y = total_concentrations[mask]
yerr = standard_errors[mask]  
popt, pcov = curve_fit(linear_model, x, y)
m_fit, b_fit = popt
perr = np.sqrt(np.diag(pcov))
m_err, b_err = perr
residuals = y - linear_model(x, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y - np.mean(y))**2)
r_squared = 1 - (ss_res / ss_tot)
r_value = np.sign(m_fit) * np.sqrt(r_squared)
x_fit = np.linspace(min(x), max(x), 100)
y_fit = linear_model(x_fit, *popt)
print("\nFit results:")
print(f"Slope = {m_fit:.4f} ± {m_err:.4f}")
print(f"Intercept = {b_fit:.4f} ± {b_err:.4f}")
print(f"R² = {r_squared:.3f}")
print(f"R  = {r_value:.3f}")
# %%
#total mass against wind speed 

grouped_mass_values = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds_mass = {i: [] for i in range(len(windspeed_bins))}
for mass_entry in filtered_dry_mass_inf:
    date = mass_entry['Date']
    BCB_start = mass_entry['BCB_start']
    BCB_stop = mass_entry['BCB_stop']
    
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) & 
        (df_combined['BCB_start'] == BCB_start) & 
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    if windspeed_entry.empty:
        continue  

    windspeed = windspeed_entry['Windspeed'].values[0]
    mass_value = mass_entry['Dry Mass (µg/m³)']

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
        avg_windspeed = np.mean(mean_windspeeds_mass[idx])  
        avg_mass = np.mean(mass_list)

        avg_windspeeds_mass.append(avg_windspeed)
        total_mass_values.append(avg_mass)
windspeed_values_mass = np.array(avg_windspeeds_mass)
total_mass_values = np.array(total_mass_values)
popt_mass, _ = curve_fit(linear_model, windspeed_values_mass, total_mass_values)
m_fit_mass, b_fit_mass = popt_mass
residuals_mass = total_mass_values - linear_model(windspeed_values_mass, *popt_mass)
ss_res_mass = np.sum(residuals_mass**2)
ss_tot_mass = np.sum((total_mass_values - np.mean(total_mass_values))**2)
r_squared_mass = 1 - (ss_res_mass / ss_tot_mass)

# %%
plt.figure(figsize=(8, 6))
for idx in range(len(windspeed_bins)):
    plt.scatter(windspeed_values_mass[idx], total_mass_values[idx], 
                color="blue", s=100, edgecolor='black', zorder=3)
x_fit_mass = np.linspace(min(windspeed_values_mass), max(windspeed_values_mass), 100)
y_fit_mass = linear_model(x_fit_mass, *popt_mass)
plt.plot(x_fit_mass, y_fit_mass, 'r-', label=f'Fit: y = {m_fit_mass:.3f}x + {b_fit_mass:.3f}, R² = {r_squared_mass:.2f}')
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=16, fontweight='bold')
plt.ylabel("Total Dry Mass (µg/m³)", fontsize=16, fontweight='bold')
plt.title("CAS Below Cloud Base January - June 2022", fontsize=16, fontweight='bold')
legend_labels_mass = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[idx], markersize=10, 
               label=f"{windspeed_values_mass[idx]:.1f} m/s") for idx in range(len(windspeed_bins))
]
plt.legend(handles=legend_labels_mass + [plt.Line2D([0], [0], color='red', 
               label=f'Fit: y = {m_fit_mass:.3f}x + {b_fit_mass:.3f}, R² = {r_squared_mass:.2f}')], 
           title="Wind Speed Bins", title_fontsize=14, fontsize=13)
plt.tight_layout()
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.show()
print(f"Slope (m): {m_fit_mass:.3f}")
print(f"Intercept (b): {b_fit_mass:.3f}")
print(f"R² value: {r_squared_mass:.2f}")
#%%
windspeed_bins = [
    (0, 2.5),
    (2.501, 3.5),
    (3.501, 5),
    (5.001, 7),
    (7.001, 9),
    (9.001, np.inf)
]
avg_windspeeds_mass = []
total_mass_values = []
standard_errors_mass = []
for idx, mass_list in grouped_mass_values.items():
    if mass_list:
        avg_windspeed = np.mean(mean_windspeeds_mass[idx]) 
        avg_mass = np.mean(mass_list) 
        std_mass = np.std(mass_list, ddof=1) 
        N_mass = len(mass_list)  
        SE_mass = std_mass / np.sqrt(N_mass) 

        avg_windspeeds_mass.append(avg_windspeed)
        total_mass_values.append(avg_mass)
        standard_errors_mass.append(SE_mass)
windspeed_values_mass = np.array(avg_windspeeds_mass)
total_mass_values = np.array(total_mass_values)
standard_errors_mass = np.array(standard_errors_mass)
def linear_model(x, m, b):
    return m * x + b

popt_mass, _ = curve_fit(linear_model, windspeed_values_mass, total_mass_values)
m_fit_mass, b_fit_mass = popt_mass
residuals_mass = total_mass_values - linear_model(windspeed_values_mass, *popt_mass)
ss_res_mass = np.sum(residuals_mass**2)
ss_tot_mass = np.sum((total_mass_values - np.mean(total_mass_values))**2)
r_squared_mass = 1 - (ss_res_mass / ss_tot_mass)
r_value_mass = np.sign(m_fit_mass) * np.sqrt(r_squared_mass)
plt.figure(figsize=(8, 6))
plt.errorbar(windspeed_values_mass, total_mass_values, 
             yerr=standard_errors_mass, fmt='o', color='blue', 
             markersize=10, capsize=5, capthick=2, label="CAS", 
             ecolor='black', elinewidth=1.5, zorder=3)
x_fit_mass = np.linspace(min(windspeed_values_mass), max(windspeed_values_mass), 100)
y_fit_mass = linear_model(x_fit_mass, *popt_mass)
plt.plot(x_fit_mass, y_fit_mass, 'r-', label=f'Fit: y = {m_fit_mass:.3f}x + {b_fit_mass:.3f}\nR² = {r_squared_mass:.2f}, R = {r_value_mass:.2f}')
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=16, fontweight='bold')
plt.ylabel("Total Dry Mass (µg/m³)", fontsize=16, fontweight='bold')
plt.title("CAS Below Cloud Base January - June 2022", fontsize=16, fontweight='bold')
plt.legend(fontsize=13, title_fontsize=14, loc='best', frameon=True)
plt.tight_layout()
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.ylim(0,35)
plt.xlim()
plt.show()
print(f"Slope (m): {m_fit_mass:.3f}")
print(f"Intercept (b): {b_fit_mass:.3f}")
print(f"R² value: {r_squared_mass:.2f}")
print(f"R value (Pearson correlation): {r_value_mass:.3f}")
#%%
#2(sigma) uncertainty
plt.figure(figsize=(8, 6))
plt.errorbar(windspeed_values_mass, total_mass_values,
             yerr=2 * standard_errors_mass, fmt='none',
             ecolor='lightgray', elinewidth=6, capsize=0, label='95% (±2σ)', zorder=1)

plt.errorbar(windspeed_values_mass, total_mass_values, 
             yerr=standard_errors_mass, fmt='o', color='blue', 
             markersize=10, capsize=5, capthick=2, label="CAS", 
             ecolor='black', elinewidth=1.5, zorder=3)
x_fit_mass = np.linspace(min(windspeed_values_mass), max(windspeed_values_mass), 100)
y_fit_mass = linear_model(x_fit_mass, *popt_mass)
plt.plot(x_fit_mass, y_fit_mass, 'r-', label=f'Fit: y = {m_fit_mass:.3f}x + {b_fit_mass:.3f}\nR² = {r_squared_mass:.2f}, R = {r_value_mass:.2f}')
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=20, fontweight='bold')
plt.ylabel("Total Dry Mass (µg/m³)", fontsize=20, fontweight='bold')
plt.title("CAS Below Cloud Base January - June 2022", fontsize=20, fontweight='bold')
plt.legend(fontsize=13, title_fontsize=14, loc='best', frameon=True)
plt.tight_layout()
plt.xticks(fontsize=20, fontweight='bold')
plt.yticks(fontsize=20, fontweight='bold')
plt.ylim(0, 35)
plt.xlim(0, 12)
plt.show()
#%%
windspeed_bins = [
    (0, 2.5),
    (2.501, 3.5),
    (3.501, 5),
    (5.001, 7),
    (7.001, 9),
    (9.001, np.inf)
]
avg_windspeeds_mass = []
total_mass_values = []
standard_errors_mass = []
for idx, mass_list in grouped_mass_values.items():
    if mass_list:
        avg_windspeed = np.mean(mean_windspeeds_mass[idx])  
        avg_mass = np.mean(mass_list) 
        std_mass = np.std(mass_list, ddof=1) 
        N_mass = len(mass_list) 
        SE_mass = std_mass / np.sqrt(N_mass)
        avg_windspeeds_mass.append(avg_windspeed)
        total_mass_values.append(avg_mass)
        standard_errors_mass.append(SE_mass)
windspeed_values_mass = np.array(avg_windspeeds_mass)
total_mass_values = np.array(total_mass_values)
standard_errors_mass = np.array(standard_errors_mass)
def linear_model(x, m, b):
    return m * x + b

popt_mass, pcov_mass = curve_fit(linear_model, windspeed_values_mass, total_mass_values)
m_fit_mass, b_fit_mass = popt_mass
perr_mass = np.sqrt(np.diag(pcov_mass))
m_err_mass, b_err_mass = perr_mass
residuals_mass = total_mass_values - linear_model(windspeed_values_mass, *popt_mass)
ss_res_mass = np.sum(residuals_mass**2)
ss_tot_mass = np.sum((total_mass_values - np.mean(total_mass_values))**2)
r_squared_mass = 1 - (ss_res_mass / ss_tot_mass)
r_value_mass = np.sign(m_fit_mass) * np.sqrt(r_squared_mass)
plt.figure(figsize=(8, 6))

plt.errorbar(windspeed_values_mass, total_mass_values, 
             yerr=standard_errors_mass, fmt='o', color='#4daf4a',  # green
             markersize=10, capsize=5, capthick=2, label="CAS", 
             ecolor='black', elinewidth=1.5, zorder=3)

x_fit_mass = np.linspace(min(windspeed_values_mass), max(windspeed_values_mass), 100)
y_fit_mass = linear_model(x_fit_mass, *popt_mass)

plt.plot(x_fit_mass, y_fit_mass, '-', color='#984ea3', linewidth=2.5,  # purple
         label=f'Fit: y = ({m_fit_mass:.3f}±{m_err_mass:.3f})x + {b_fit_mass:.3f}\nR² = {r_squared_mass:.2f}, R = {r_value_mass:.2f}')

plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=16, fontweight='bold')
plt.ylabel("Total Dry Mass (µg/m³)", fontsize=16, fontweight='bold')
plt.title("CAS Below Cloud Base January - June 2022", fontsize=16, fontweight='bold')

plt.legend(fontsize=13, loc='upper left', frameon=False)
plt.tight_layout()
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.ylim(0, 35)
plt.xlim(0, 12)
plt.show()
print(f"Slope (m): {m_fit_mass:.3f} ± {m_err_mass:.3f}")
print(f"Intercept (b): {b_fit_mass:.3f}")
print(f"R² value: {r_squared_mass:.2f}")
print(f"R value (Pearson correlation): {r_value_mass:.3f}")
plt.figure(figsize=(8, 6))
plt.errorbar(windspeed_values_mass, total_mass_values,
             yerr=2 * standard_errors_mass, fmt='none',
             ecolor='lightgray', elinewidth=6, capsize=0, label='95% (±2σ)', zorder=1)
plt.errorbar(windspeed_values_mass, total_mass_values, 
             yerr=standard_errors_mass, fmt='o', color='#4daf4a',
             markersize=10, capsize=5, capthick=2, label="CAS", 
             ecolor='black', elinewidth=1.5, zorder=3)
plt.plot(x_fit_mass, y_fit_mass, '-', color='#984ea3', linewidth=2.5,
         label=f'Fit: y = ({m_fit_mass:.3f}±{m_err_mass:.3f})x + {b_fit_mass:.3f}\nR² = {r_squared_mass:.2f}, R = {r_value_mass:.2f}')

plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=20, fontweight='bold')
plt.ylabel("Total WindDry Mass (µg/m³)", fontsize=20, fontweight='bold')
plt.title("CAS Below Cloud Base \nJanuary-June 2022", fontsize=20, fontweight='bold')
plt.legend(fontsize=16, loc='upper left', frameon=False)
plt.tight_layout()
plt.xticks(fontsize=20, fontweight='bold')
plt.yticks(fontsize=20, fontweight='bold')
plt.ylim(0, 25)
plt.xlim(0, 12)
plt.show()
#%%
windspeed_bins = [
    (0, 2.5),
    (2.501, 3.5),
    (3.501, 5),
    (5.001, 7),
    (7.001, 9),
    (9.001, np.inf)
]
bin_centers_um = np.array([
    2.25, 2.75, 3.25, 3.75, 4.5, 5.75,
    6.85, 7.55, 9.05, 11.4, 13.8, 17.5,
    22.5, 27.5, 32.5, 37.5, 42.5, 47.5
])
bin_widths_um = np.diff(np.concatenate(([2], bin_centers_um + np.diff(bin_centers_um, prepend=0)/2)))
radii_um = bin_centers_um / 2
windspeed_values_mass = []
total_mass_values = []
standard_errors_mass = []

for entry in filtered_dry_mass_inf:  
    date = entry['Date']
    start = entry['BCB_start']
    stop = entry['BCB_stop']
    total_mass = entry['Dry Mass (µg/m³)']
    ws_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == start) &
        (df_combined['BCB_stop'] == stop)
    ]
    if ws_entry.empty or np.isnan(total_mass):
        continue

    windspeed = ws_entry['Windspeed'].values[0]
    windspeed_values_mass.append(windspeed)
    total_mass_values.append(total_mass)
    if 'Standard_Error' in entry:
        standard_errors_mass.append(entry['Standard_Error'])
    else:
        standard_errors_mass.append(0) 
windspeed_values_mass = np.array(windspeed_values_mass)
total_mass_values = np.array(total_mass_values)
standard_errors_mass = np.array(standard_errors_mass)
#%%
sample_area_cm2 = 0.0025
plane_speed_cm_s = 1.2e4
sampling_time_s = 198  # seconds
T = sampling_time_s
V = sample_area_cm2 * plane_speed_cm_s  # cm³/s
bin_centers_um = np.array([2.25, 2.75, 3.25, 3.75, 4.5, 5.75,
                           6.85, 7.55, 9.05, 11.4, 13.8, 17.5,
                           22.5, 27.5, 32.5, 37.5, 42.5, 47.5])
bin_widths_um = np.diff(np.concatenate(([2], bin_centers_um + np.diff(bin_centers_um, prepend=0)/2)))
radii_um = bin_centers_um / 2
rho_salt_ug_cm3 = 2.2e6  # µg/cm³
eta = (4/3) * np.pi * rho_salt_ug_cm3  # µg/cm³
grouped_mass_errors = {i: [] for i in range(len(windspeed_bins))}

for entry in dry_exponential_fits_10:
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    n0 = entry['Dry_Intercept_n0']
    D = entry['Dry_E_folding_D']
    ws_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    if ws_entry.empty or np.isnan(n0) or np.isnan(D):
        continue
    windspeed = ws_entry['Windspeed'].values[0]
    n_i = n0 * np.exp(-bin_centers_um / D)  # cm⁻³/μm
    radii_cm = radii_um * 1e-4
    bin_widths_cm = bin_widths_um * 1e-4
    n_i = n0 * np.exp(-bin_centers_um / D)
    print(f"\nDate: {date}, BCB_start: {BCB_start}, Windspeed: {windspeed:.2f}")
    print(f"  n0: {n0:.4e}, D: {D:.4f}")
    print(f"  n_i (first 5): {n_i[:5]}")
    print(f"  bin_widths_cm (first 5): {bin_widths_cm[:5]}")
    print(f"  radii_cm (first 5): {radii_cm[:5]}")

    term = n_i * bin_widths_cm * (radii_cm ** 6)
    print(f"  term (first 5): {term[:5]}")
    print(f"  sum(term): {np.nansum(term):.4e}")

    if np.nansum(term) == 0:
        print("⚠️ WARNING: Term sum is zero — likely cause of mass_error = 0")
    summation = np.nansum(term)
    mass_error = eta * np.sqrt(summation / (T * V))  # µg/m³


    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:
            grouped_mass_errors[idx].append(mass_error)
            break
median_mass_errors_per_bin = []
for idx, errs in grouped_mass_errors.items():
    if errs:
        median_mass_errors_per_bin.append(np.nanmedian(errs))
    else:
        median_mass_errors_per_bin.append(np.nan)
print("\nMass Counting Errors per Wind Speed Bin:")
for idx, (bounds, err) in enumerate(zip(windspeed_bins, counting_errors_mass)):
    label = f"{bounds[0]}–{bounds[1] if bounds[1] != np.inf else '∞'} m/s"
    if not np.isnan(err):
        print(f"  Bin {idx} ({label}): {err:.4f} µg/m³")
    else:
        print(f"  Bin {idx} ({label}): NaN (no data)")


# %%
counting_errors_mass = np.array(median_mass_errors_per_bin)

# %%
def linear_model(x, m, b):
    return m * x + b

popt_mass, pcov_mass = curve_fit(linear_model, windspeed_values_mass, total_mass_values)
m_fit_mass, b_fit_mass = popt_mass
m_err_mass, b_err_mass = np.sqrt(np.diag(pcov_mass))

residuals = total_mass_values - linear_model(windspeed_values_mass, *popt_mass)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((total_mass_values - np.mean(total_mass_values))**2)
r_squared_mass = 1 - (ss_res / ss_tot)
r_value_mass = np.sign(m_fit_mass) * np.sqrt(r_squared_mass)

x_fit_mass = np.linspace(min(windspeed_values_mass), max(windspeed_values_mass), 100)
y_fit_mass = linear_model(x_fit_mass, *popt_mass)

plt.figure(figsize=(8, 6))
plt.errorbar(windspeed_values_mass, total_mass_values,
             yerr=standard_errors_mass,
             fmt='o', color='#4daf4a',  # green markers
             markersize=10, capsize=5, capthick=2,
             ecolor='black', elinewidth=1.5,
             label="CAS Standard Error", zorder=3)

plt.errorbar(windspeed_values_mass - 0.15,
             total_mass_values,
             yerr=counting_errors_mass,
             fmt='s', color='brown',  
             markersize=6,
             capsize=6, capthick=2, elinewidth=3,
             label="CAS Instrument Counting Error", zorder=2)
plt.plot(x_fit_mass, y_fit_mass, '-', color='black', linewidth=2.5,
         label=f'Fit: y = ({m_fit_mass:.3f}±{m_err_mass:.3f})x + {b_fit_mass:.3f}\n'
               f'R² = {r_squared_mass:.2f}, R = {r_value_mass:.2f}')
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=20, fontweight='bold')
plt.ylabel("Total Dry Mass (µg/m³)", fontsize=20, fontweight='bold')
plt.title("CAS Below Cloud Base\nJanuary–June 2022", fontsize=20, fontweight='bold')
plt.legend(fontsize=14, loc='upper left', frameon=False)
plt.tight_layout()
plt.xlim(0, 12)
plt.ylim(0, 25)
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')
plt.show()
print(f"Slope (m): {m_fit_mass:.3f} ± {m_err_mass:.3f}")
print(f"Intercept (b): {b_fit_mass:.3f}")
print(f"R² value: {r_squared_mass:.2f}")
print(f"R value (Pearson correlation): {r_value_mass:.3f}")
#%%
windspeed_bins = [
    (0, 2.5), (2.501, 3.5), (3.501, 5),
    (5.001, 7), (7.001, 9), (9.001, np.inf)
]

bin_centers_um = np.array([2.25, 2.75, 3.25, 3.75, 4.5, 5.75,
                           6.85, 7.55, 9.05, 11.4, 13.8, 17.5,
                           22.5, 27.5, 32.5, 37.5, 42.5, 47.5])
edges_um = np.empty(len(bin_centers_um) + 1)
edges_um[0] = 2.0 
edges_um[1:-1] = 0.5 * (bin_centers_um[:-1] + bin_centers_um[1:])
edges_um[-1] = bin_centers_um[-1] + 0.5 * (bin_centers_um[-1] - bin_centers_um[-2])
bin_widths_um = np.diff(edges_um)                  # strictly positive (µm)
radii_cm = (bin_centers_um / 2.0) * 1e-4          # µm -> cm
sample_area_cm2 = 0.0025
plane_speed_cm_s = 1.2e4
T = 198                                           # s
V = sample_area_cm2 * plane_speed_cm_s            # cm^3/s
rho_salt_ug_cm3 = 2.2e6                           # µg/cm^3
eta_cm = (4.0/3.0) * np.pi * rho_salt_ug_cm3      # µg/cm^3
eta_m = eta_cm * 1e6                               # convert to µg/m^3 for final output
grouped_mass_errors = {i: [] for i in range(len(windspeed_bins))}
for entry in dry_mass_data_inf: 
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    n0 = entry.get('Dry Intercept (N0)')
    D  = entry.get('Dry Slope (D)')               

    ws_row = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    if ws_row.empty or np.isnan(n0) or np.isnan(D):
        continue
    windspeed = ws_row['Windspeed'].values[0]
    n_i = n0 * np.exp(-bin_centers_um / D)       

    term_cm3 = (n_i * bin_widths_um) * (radii_cm ** 6)  # cm^3
    summation_cm3 = np.nansum(term_cm3)

    if summation_cm3 <= 0 or not np.isfinite(summation_cm3):
        mass_err_ug_m3 = 0.0
    else:
        factor = np.sqrt(summation_cm3 / (T * V))
        # result in µg/m^3
        mass_err_ug_m3 = eta_m * factor

    for idx, (lo, hi) in enumerate(windspeed_bins):
        if lo <= windspeed < hi:
            grouped_mass_errors[idx].append(mass_err_ug_m3)
            break
median_mass_errors_per_bin = [np.nanmedian(errs) if errs else np.nan
                              for errs in grouped_mass_errors.values()]

print("\n✅ Mass Counting Errors per Wind Speed Bin (unfiltered):")
for idx, (bounds, err) in enumerate(zip(windspeed_bins, median_mass_errors_per_bin)):
    label = f"{bounds[0]}–{bounds[1] if bounds[1] != np.inf else '∞'} m/s"
    if not np.isnan(err):
        print(f"  Bin {idx} ({label}): {err:.4f} µg/m³")
    else:
        print(f"  Bin {idx} ({label}): NaN (no data)")
#%%
counting_errors_mass = np.array(median_mass_errors_per_bin)
print("Counting errors used for plotting:", counting_errors_mass)

# %%
def linear_model(x, m, b):
    return m * x + b

popt_mass, pcov_mass = curve_fit(linear_model, windspeed_values_mass, total_mass_values)
m_fit_mass, b_fit_mass = popt_mass
m_err_mass, b_err_mass = np.sqrt(np.diag(pcov_mass))

residuals = total_mass_values - linear_model(windspeed_values_mass, *popt_mass)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((total_mass_values - np.mean(total_mass_values))**2)
r_squared_mass = 1 - (ss_res / ss_tot)
r_value_mass = np.sign(m_fit_mass) * np.sqrt(r_squared_mass)

x_fit_mass = np.linspace(min(windspeed_values_mass), max(windspeed_values_mass), 100)
y_fit_mass = linear_model(x_fit_mass, *popt_mass)
plt.figure(figsize=(8, 6))
offset = 0.4

plt.errorbar(windspeed_values_mass, total_mass_values,
             yerr=standard_errors_mass,
             fmt='o', color='black',
             markersize=10, capsize=5, capthick=2,
             ecolor='black', elinewidth=1.5,
             label="CAS Standard Error", zorder=3)
plt.errorbar(windspeed_values_mass + offset, total_mass_values,
             yerr=counting_errors_mass,
             fmt='o', color='brown',
             markersize=10, capsize=5, capthick=2,
             ecolor='black', elinewidth=1.5,
             label="CAS Error in Total Mass", zorder=2)
plt.plot(x_fit_mass, y_fit_mass, '-', color='black', linewidth=2.5,
         label=f'Fit: y = ({m_fit_mass:.3f}±{m_err_mass:.3f})x + {b_fit_mass:.3f}\n'
               f'R² = {r_squared_mass:.2f}, R = {r_value_mass:.2f}')
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=20, fontweight='bold')
plt.ylabel("Total Dry Mass (µg/m³)", fontsize=20, fontweight='bold')
plt.title("CAS Below Cloud Base\nJanuary–June 2022", fontsize=20, fontweight='bold')
plt.legend(fontsize=12, loc='upper left', frameon=False)
plt.tight_layout()
plt.xlim(0, 12)
plt.ylim(0, 25)
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')
plt.show()
print(f"Slope (m): {m_fit_mass:.3f} ± {m_err_mass:.3f}")
print(f"Intercept (b): {b_fit_mass:.3f}")
print(f"R² value: {r_squared_mass:.2f}")
print(f"R value (Pearson correlation): {r_value_mass:.3f}")

# %%

windspeed_bins = [(0,2.5),(2.501,3.5),(3.501,5),(5.001,7),(7.001,9),(9.001,np.inf)]
bin_centers_um = np.array([2.25, 2.75, 3.25, 3.75, 4.5, 5.75,
                           6.85, 7.55, 9.05, 11.4, 13.8, 17.5,
                           22.5, 27.5, 32.5, 37.5, 42.5, 47.5])
edges_um = np.empty(len(bin_centers_um) + 1)
edges_um[0] = 2.0
edges_um[1:-1] = 0.5*(bin_centers_um[:-1] + bin_centers_um[1:])
edges_um[-1] = bin_centers_um[-1] + 0.5*(bin_centers_um[-1] - bin_centers_um[-2])
bin_widths_um = np.diff(edges_um)
radii_cm = (bin_centers_um/2.0)*1e-4  # µm -> cm
sample_area_cm2 = 0.0025
plane_speed_cm_s = 1.2e4
T = 198
V = sample_area_cm2 * plane_speed_cm_s
rho_salt_ug_cm3 = 2.2e6
eta_cm = (4/3)*np.pi*rho_salt_ug_cm3
eta_m = eta_cm*1e6  # -> µg/m^3
try:
    source_mass_entries = filtered_dry_mass_inf
except NameError:
    source_mass_entries = dry_mass_data_inf
grouped_mass_errors = {i: [] for i in range(len(windspeed_bins))}

for entry in dry_mass_data_inf:  
    n0 = entry.get('Dry Intercept (N0)')
    D  = entry.get('Dry Slope (D)')
    if n0 is None or D is None or np.isnan(n0) or np.isnan(D):
        continue
    ws_row = df_combined[
        (df_combined['Date']==entry['Date']) &
        (df_combined['BCB_start']==entry['BCB_start']) &
        (df_combined['BCB_stop']==entry['BCB_stop'])
    ]
    if ws_row.empty: 
        continue
    ws = float(ws_row['Windspeed'].values[0])
    n_i = n0*np.exp(-bin_centers_um/D)                     # cm^-3 per µm
    term_cm3 = (n_i*bin_widths_um) * (radii_cm**6)         # cm^3
    S = np.nansum(term_cm3)
    if S<=0 or not np.isfinite(S):
        mass_err = 0.0
    else:
        mass_err = eta_m * np.sqrt(S/(T*V))                # µg/m^3

    for k,(lo,hi) in enumerate(windspeed_bins):
        if lo <= ws < hi:
            grouped_mass_errors[k].append(mass_err)
            break

counting_errors_mass = np.array([
    np.nanmedian(vals) if len(vals)>0 else np.nan
    for vals in grouped_mass_errors.values()
])
grouped_mass_vals   = {i: [] for i in range(len(windspeed_bins))}
grouped_windspeeds  = {i: [] for i in range(len(windspeed_bins))}

for entry in source_mass_entries:
    mass_val = entry.get('Dry Mass (µg/m³)')
    if mass_val is None or not np.isfinite(mass_val):
        continue

    ws_row = df_combined[
        (df_combined['Date']==entry['Date']) &
        (df_combined['BCB_start']==entry['BCB_start']) &
        (df_combined['BCB_stop']==entry['BCB_stop'])
    ]
    if ws_row.empty: 
        continue
    ws = float(ws_row['Windspeed'].values[0])

    for k,(lo,hi) in enumerate(windspeed_bins):
        if lo <= ws < hi:
            grouped_mass_vals[k].append(mass_val)
            grouped_windspeeds[k].append(ws)
            break

avg_ws, avg_mass, se_mass = [], [], []
for k in range(len(windspeed_bins)):
    vals = np.array(grouped_mass_vals[k], dtype=float)
    wss  = np.array(grouped_windspeeds[k], dtype=float)
    if vals.size == 0:
        continue
    avg_ws.append(np.nanmean(wss))
    avg_mass.append(np.nanmean(vals))
    if vals.size >= 2:
        se_mass.append(np.nanstd(vals, ddof=1)/np.sqrt(vals.size))
    else:
        se_mass.append(np.nan)

avg_ws  = np.array(avg_ws, dtype=float)
avg_mass= np.array(avg_mass, dtype=float)
se_mass = np.array(se_mass, dtype=float)
print("CAS mass per-bin points:")
for x,y,se,ce in zip(avg_ws, avg_mass, se_mass, counting_errors_mass[np.isfinite(counting_errors_mass)]):
    print(f"  WS={x:.2f} m/s, Mass={y:.2f} µg/m³, SE={se:.2f}, CountErr≈{ce:.2f}")
def linear_model(x,m,b): return m*x + b
mask = np.isfinite(avg_ws) & np.isfinite(avg_mass)
x = avg_ws[mask]; y = avg_mass[mask]; yerr = 2 * se_mass[mask] 

popt, pcov = curve_fit(linear_model, x, y)
m_fit, b_fit = popt
m_err, b_err = np.sqrt(np.diag(pcov))
res = y - linear_model(x, *popt)
r2 = 1 - (np.sum(res**2) / np.sum((y - y.mean())**2))
R  = np.sign(m_fit)*np.sqrt(max(r2,0))
x_fit = np.linspace(x.min(), x.max(), 100)
y_fit = linear_model(x_fit, *popt)
plt.figure(figsize=(8,6))
plt.errorbar(x, y, yerr=yerr, fmt='o', color='black',
             ecolor='black', elinewidth=1.5, capsize=5, capthick=2,
             label="±2 Standard Errors"
, zorder=3)
offset = 0.35
count_err_for_points = 2 * counting_errors_mass[~np.isnan(counting_errors_mass)][:len(x)]  # ±2σ mass error
plt.errorbar(x+offset, y, yerr=count_err_for_points, fmt='o', color='#8c510a',
             ecolor='black', elinewidth=1.5, capsize=5, capthick=2,
             label="±2σ CAS Counting Error"
, zorder=2)

plt.plot(x_fit, y_fit, '-', color='black', linewidth=2.5,
         label=f"Fit: y = ({m_fit:.3f}±{m_err:.3f})x + {b_fit:.3f}\nR² = {r2:.2f}, R = {R:.2f}")

plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=20, fontweight='bold')
plt.ylabel("Total Dry Mass (µg/m³)", fontsize=20, fontweight='bold')
plt.title("CAS Below Cloud Base\nJanuary–June 2022", fontsize=20, fontweight='bold')
plt.legend(fontsize=14, frameon=False, loc='upper left')
plt.xlim(0, 12); plt.ylim(0, 30)
plt.xticks(fontsize=18, fontweight='bold'); plt.yticks(fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n=== CAS Mass Fit ===")
print(f"Slope (m): {m_fit:.3f} ± {m_err:.3f}")
print(f"Intercept (b): {b_fit:.3f}")
print(f"R² value: {r2:.2f}, R = {R:.2f}")

# %%
