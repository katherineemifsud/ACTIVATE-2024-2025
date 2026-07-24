#%%
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import datetime
import pathlib
import statistics
import glob
import os
import re
import math
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
import pickle
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
#LWC histogram
lwc_values = [
    entry['LWC_mean']
    for entry in Y_BCB_calc
    if np.isfinite(entry['LWC_mean'])
]
plt.figure(figsize=(8, 6))
plt.hist(
    lwc_values,
    bins=20,
    edgecolor='black',
    alpha=0.7
)
plt.xlabel('Mean LWC (g m$^{-3}$)', fontsize=15, fontweight='bold')
plt.ylabel('Frequency of flight legs', fontsize=15, fontweight='bold')
plt.title('Leg-average CAS LWC', fontweight='bold', fontsize=16)
plt.xticks(fontweight='bold', fontsize=14)
plt.yticks(fontweight='bold', fontsize=14)
plt.savefig("LWC_CAS.pdf", dpi=300, bbox_inches='tight')
plt.show()
print(f"Number of LWC legs plotted: {len(lwc_values)}")
print(f"Mean leg-average LWC: {np.nanmean(lwc_values):.6f} g m^-3")
print(f"Median leg-average LWC: {np.nanmedian(lwc_values):.6f} g m^-3")

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
filtered_master_BCB_ddry = []

for flight in filtered_master_BCB_gRH:
    for entry in flight:

        date = entry['Date']
        BCB_start = entry['BCB_start']
        BCB_stop = entry['BCB_stop']
        gRh_mean = entry['gRh_mean'][0]

        if gRh_mean > 0:
            ddry_values = np.array([
                D_amb / gRh_mean for D_amb in bin_center
            ])
        else:
            ddry_values = np.full(len(bin_center), np.nan)
            print(
                f"Skipping division for {date}, "
                f"{BCB_start}-{BCB_stop} due to invalid gRh_mean."
            )

        ddry_bin_widths = np.diff(
            ddry_values,
            append=np.nan
        )

        raw_concentrations = next(
            (
                leg for leg in Y_BCB_calc
                if leg['Date'] == date
                and leg['BCB_start'] == BCB_start
                and leg['BCB_stop'] == BCB_stop
            ),
            None
        )

        if raw_concentrations is not None:

            dN_dD_ambient = np.array([
                raw_concentrations.get(
                    f'Bin{i}_Y_mean',
                    np.nan
                )
                for i in range(12, 30)
            ], dtype=float)

            dN_dD_dry = np.where(
                (~np.isnan(dN_dD_ambient))
                & (~np.isnan(ddry_bin_widths))
                & (gRh_mean > 0),

                dN_dD_ambient
                * (np.array(bin_center) / ddry_values)
                * (
                    np.diff(bin_center, append=np.nan)
                    / ddry_bin_widths
                ),

                np.nan
            )

        else:
            dN_dD_dry = np.full(
                len(bin_center),
                np.nan
            )

            print(
                f"Missing raw size distribution for "
                f"{date}, {BCB_start}-{BCB_stop}"
            )

        filtered_master_BCB_ddry.append({
            'Date': date,
            'BCB_start': BCB_start,
            'BCB_stop': BCB_stop,
            'ddry': ddry_values.tolist(),
            'dN/dDdry': dN_dD_dry.tolist(),
            'ddry_bin_widths': ddry_bin_widths.tolist(),
            'gRh_mean': gRh_mean
        })

print(
    f"Length of filtered_master_BCB_ddry: "
    f"{len(filtered_master_BCB_ddry)}"
)
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
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy.ma as ma
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
#save the average distribution
average_dry_distribution = pd.DataFrame({
    'Dry_Diameter_um': common_bins,
    'Average_dN_dD_dry': average_dN_dD_dry,
    'N_profiles': count_interpolated_dN_dD_dry
})
save_dir = "/home/disk/eos4/kathem24/activate/data/2021/CAS"
save_path = os.path.join(save_dir, "Average_Dry_Size_Distribution_beforemass2022CAS.csv")
average_dry_distribution.to_csv(save_path, index=False)
print(f"Saved to: {save_path}")
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
#fitting an exponential to dry distributions, and removing extreme slopes after 10 um slope

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
# %%
#trying other roughness lengths
#%%
Z0 = 0.001
Z10 = 10 

corrected_calc_bcb1 = {'Date': [], 'Corrected_bcb_windspeed1': []}

for flight in master_BCB:
    for wind_alt in flight:
        date = wind_alt['Date']
        windspeed = wind_alt['Winds_mean']
        altitude = wind_alt['Alts_mean']

        for wind_mean, alt_mean in zip(windspeed, altitude):
            new_windspeed = wind_mean * (np.log(Z10/Z0) / np.log(alt_mean / Z0))

            corrected_calc_bcb1['Date'].append(date)
            corrected_calc_bcb1['Corrected_bcb_windspeed1'].append(new_windspeed)
for date, wind_mean in zip(corrected_calc_bcb1['Date'], corrected_calc_bcb1['Corrected_bcb_windspeed1']):
    print(f"Date: {date}, Corrected_bcb_windspeed1: {wind_mean}")

# %%
#mean windspeed
corrected_windspeeds1 = corrected_calc_bcb1['Corrected_bcb_windspeed1']
corrected_windspeeds1 = [ws for ws in corrected_windspeeds1 if not np.isnan(ws)]
mean_corrected_windspeed1 = np.mean(corrected_windspeeds1)
print(f"Mean Corrected Wind Speed: {mean_corrected_windspeed1:.2f} m/s")

#%%
#standard deviation of windspeed
mean_corrected_windspeed1 = sum(corrected_windspeeds1) / len(corrected_windspeeds1)
variance = sum((ws - mean_corrected_windspeed1) ** 2 for ws in corrected_windspeeds1) / (len(corrected_windspeeds1) - 1)
std_corrected_windspeed1 = variance ** 0.5
print(f"Standard Deviation of Corrected Wind Speed: {std_corrected_windspeed1:.2f} m/s")
# %%
Z0 = 0.0001
Z10 = 10 

corrected_calc_bcb2 = {'Date': [], 'Corrected_bcb_windspeed2': []}

for flight in master_BCB:
    for wind_alt in flight:
        date = wind_alt['Date']
        windspeed = wind_alt['Winds_mean']
        altitude = wind_alt['Alts_mean']

        for wind_mean, alt_mean in zip(windspeed, altitude):
            new_windspeed = wind_mean * (np.log(Z10/Z0) / np.log(alt_mean / Z0))

            corrected_calc_bcb2['Date'].append(date)
            corrected_calc_bcb2['Corrected_bcb_windspeed2'].append(new_windspeed)
for date, wind_mean in zip(corrected_calc_bcb2['Date'], corrected_calc_bcb2['Corrected_bcb_windspeed2']):
    print(f"Date: {date}, Corrected_bcb_windspeed2: {wind_mean}")

# %%
#mean windspeed
corrected_windspeeds2 = corrected_calc_bcb2['Corrected_bcb_windspeed2']
corrected_windspeeds2 = [ws for ws in corrected_windspeeds2 if not np.isnan(ws)]
mean_corrected_windspeed2 = np.mean(corrected_windspeeds2)
print(f"Mean Corrected Wind Speed: {mean_corrected_windspeed2:.2f} m/s")

#%%
#standard deviation of windspeed
mean_corrected_windspeed2 = sum(corrected_windspeeds2) / len(corrected_windspeeds2)
variance = sum((ws - mean_corrected_windspeed2) ** 2 for ws in corrected_windspeeds2) / (len(corrected_windspeeds2) - 1)
std_corrected_windspeed2 = variance ** 0.5
print(f"Standard Deviation of Corrected Wind Speed: {std_corrected_windspeed2:.2f} m/s")
# %%
wind_Z002 = np.asarray(
    corrected_calc_bcb["Corrected_bcb_windspeed"],
    dtype=float)

wind_Z0001 = np.asarray(
    corrected_calc_bcb1["Corrected_bcb_windspeed1"],
    dtype=float)

wind_Z00001 = np.asarray(
    corrected_calc_bcb2["Corrected_bcb_windspeed2"],
    dtype=float)

dates_wind = np.asarray(
    corrected_calc_bcb["Date"])
print("z0 = 0.02 m legs:", len(wind_Z002))
print("z0 = 0.001 m legs:", len(wind_Z0001))
print("z0 = 0.0001 m legs:", len(wind_Z00001))

if not (
    len(wind_Z002) ==
    len(wind_Z0001) ==
    len(wind_Z00001)
):
    raise ValueError(
        "No."
    )
valid = (
    np.isfinite(wind_Z002) &
    np.isfinite(wind_Z0001) &
    np.isfinite(wind_Z00001))

wind_comparison_df = pd.DataFrame({
    "Date": dates_wind[valid],
    "Wind_Z0_0.02": wind_Z002[valid],
    "Wind_Z0_0.001": wind_Z0001[valid],
    "Wind_Z0_0.0001": wind_Z00001[valid]})

print(
    "Number of commonly valid BCB legs:",
    len(wind_comparison_df))
print(wind_comparison_df.head())
# %%
baseline = wind_comparison_df["Wind_Z0_0.02"].to_numpy()

wind_cases = {
    "Original: z0 = 0.02 m":
        wind_comparison_df["Wind_Z0_0.02"].to_numpy(),

    "Sensitivity: z0 = 0.001 m":
        wind_comparison_df["Wind_Z0_0.001"].to_numpy(),

    "Sensitivity: z0 = 0.0001 m":
        wind_comparison_df["Wind_Z0_0.0001"].to_numpy()
}

baseline_mean = np.mean(baseline)

wind_stats_records = []

for case_name, wind_values in wind_cases.items():

    differences = wind_values - baseline

    # Avoid dividing by zero for any zero-wind legs
    valid_percent = (
        np.isfinite(baseline) &
        np.isfinite(wind_values) &
        (baseline != 0)
    )

    per_leg_percent_difference = (
        100 *
        differences[valid_percent] /
        baseline[valid_percent]
    )

    correlation = np.corrcoef(
        baseline,
        wind_values
    )[0, 1]

    wind_stats_records.append({
        "Case": case_name,
        "Number_of_legs": len(wind_values),
        "Mean_wind": np.mean(wind_values),
        "Median_wind": np.median(wind_values),
        "Std_wind": np.std(wind_values, ddof=1),
        "Minimum_wind": np.min(wind_values),
        "Maximum_wind": np.max(wind_values),

        # Difference relative to z0 = 0.02 m
        "Mean_leg_difference": np.mean(differences),
        "Std_leg_difference": np.std(differences, ddof=1),
        "Mean_absolute_difference": np.mean(
            np.abs(differences)
        ),

        # Difference between the overall means
        "Percent_change_in_mean": (
            100 *
            (np.mean(wind_values) - baseline_mean) /
            baseline_mean
        ),

        # Typical percentage difference leg by leg
        "Median_leg_percent_difference": (
            np.median(per_leg_percent_difference)
        ),

        "Maximum_absolute_difference": np.max(
            np.abs(differences)
        ),

        "Correlation_with_original": correlation
    })

wind_sensitivity_stats = pd.DataFrame(
    wind_stats_records
)

print(
    wind_sensitivity_stats.to_string(
        index=False,
        formatters={
            "Mean_wind": lambda x: f"{x:.3f}",
            "Median_wind": lambda x: f"{x:.3f}",
            "Std_wind": lambda x: f"{x:.3f}",
            "Minimum_wind": lambda x: f"{x:.3f}",
            "Maximum_wind": lambda x: f"{x:.3f}",
            "Mean_leg_difference":
                lambda x: f"{x:+.3f}",
            "Std_leg_difference":
                lambda x: f"{x:.3f}",
            "Mean_absolute_difference":
                lambda x: f"{x:.3f}",
            "Percent_change_in_mean":
                lambda x: f"{x:+.1f}%",
            "Median_leg_percent_difference":
                lambda x: f"{x:+.1f}%",
            "Maximum_absolute_difference":
                lambda x: f"{x:.3f}",
            "Correlation_with_original":
                lambda x: f"{x:.5f}"
        }
    )
)
# %%
pretty_table = wind_sensitivity_stats[
    [
        "Case",
        "Mean_wind",
        "Std_wind",
        "Mean_leg_difference",
        "Percent_change_in_mean",
        "Mean_absolute_difference",
        "Maximum_absolute_difference",
        "Correlation_with_original"
    ]
].copy()
case_name_map = {
    "Original: z0 = 0.02 m":
        r"Original: $z_0=0.02$ m",

    "Sensitivity: z0 = 0.001 m":
        r"Sensitivity: $z_0=0.001$ m",

    "Sensitivity: z0 = 0.0001 m":
        r"Sensitivity: $z_0=0.0001$ m"
}

pretty_table["Case"] = (
    pretty_table["Case"]
    .replace(case_name_map)
)
case_order = [
    r"Original: $z_0=0.02$ m",
    r"Sensitivity: $z_0=0.001$ m",
    r"Sensitivity: $z_0=0.0001$ m"
]

pretty_table["Case"] = pd.Categorical(
    pretty_table["Case"],
    categories=case_order,
    ordered=True
)

pretty_table = (
    pretty_table
    .sort_values("Case")
    .reset_index(drop=True)
)
display_table = pretty_table.copy()

display_table["Mean_wind"] = (
    display_table["Mean_wind"]
    .map(lambda x: f"{x:.2f}")
)

display_table["Std_wind"] = (
    display_table["Std_wind"]
    .map(lambda x: f"{x:.2f}")
)

display_table["Mean_leg_difference"] = (
    display_table["Mean_leg_difference"]
    .map(lambda x: f"{x:+.2f}")
)

display_table["Percent_change_in_mean"] = (
    display_table["Percent_change_in_mean"]
    .map(lambda x: f"{x:+.1f}%")
)

display_table["Mean_absolute_difference"] = (
    display_table["Mean_absolute_difference"]
    .map(lambda x: f"{x:.2f}")
)

display_table["Maximum_absolute_difference"] = (
    display_table["Maximum_absolute_difference"]
    .map(lambda x: f"{x:.2f}")
)

display_table["Correlation_with_original"] = (
    display_table["Correlation_with_original"]
    .map(lambda x: f"{x:.4f}")
)
display_table.columns = [
    "Roughness-length",
    "Mean $U_{10}$\n(m s$^{-1}$)",
    "SD\n(m s$^{-1}$)",
    "Mean difference\n(m s$^{-1}$)",
    "Mean change\n(%)",
    "Mean absolute\ndifference\n(m s$^{-1}$)",
    "Maximum absolute\ndifference\n(m s$^{-1}$)",
    "Correlation\nwith original"
]
fig, ax = plt.subplots(figsize=(18, 5.6))
ax.axis("off")
fig.text(
    0.5,
    0.945,
    "Sensitivity of Estimated 10m Wind Speed to Surface Roughness Length",
    ha="center",
    va="center",
    fontsize=20,
    fontweight="bold"
)
table = ax.table(
    cellText=display_table.values,
    colLabels=display_table.columns,
    cellLoc="center",
    colLoc="center",
    loc="center",
    bbox=[0.01, 0.17, 1.1, 0.62]
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.1)
column_widths = [
    0.20,  # case
    0.10,  # mean
    0.08,  # SD
    0.11,  # mean difference
    0.09,  # percentage
    0.13,  # mean absolute difference
    0.14,  # maximum difference
    0.10   # correlation
]
background_color = "white"
text_color = "black"
grid_color = "black"

number_of_columns = len(display_table.columns)
number_of_rows = len(display_table)
for column_number in range(number_of_columns):

    header_cell = table[(0, column_number)]

    header_cell.set_facecolor(background_color)
    header_cell.set_edgecolor(grid_color)
    header_cell.set_linewidth(1.5)

    header_cell.get_text().set_color(text_color)
    header_cell.get_text().set_fontweight("bold")
    header_cell.get_text().set_fontsize(10)
for row_number in range(1, number_of_rows + 1):

    for column_number in range(number_of_columns):

        cell = table[(row_number, column_number)]

        cell.set_facecolor(background_color)
        cell.set_edgecolor(grid_color)
        cell.set_linewidth(1.5)

        cell.get_text().set_color(text_color)
        cell.get_text().set_fontsize(10)
        if column_number == 0:
            cell.get_text().set_ha("left")
            cell.PAD = 0.04
plt.savefig(
    "wind_roughness_length_sensitivity_table.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.savefig(
    "wind_roughness_length_sensitivity_table.pdf",
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
# %%
wind_bin_edges = [
    0.0,
    2.5,
    3.5,
    5.0,
    7.0,
    9.0,
    np.inf
]
wind_bin_labels = [
    "0–2.5",
    "2.5–3.5",
    "3.5–5",
    "5–7",
    "7–9",
    ">9"
]

wind_comparison_df["Bin_Z0_0.02"] = pd.cut(
    wind_comparison_df["Wind_Z0_0.02"],
    bins=wind_bin_edges,
    labels=wind_bin_labels,
    include_lowest=True,
    right=True
)

wind_comparison_df["Bin_Z0_0.001"] = pd.cut(
    wind_comparison_df["Wind_Z0_0.001"],
    bins=wind_bin_edges,
    labels=wind_bin_labels,
    include_lowest=True,
    right=True
)

wind_comparison_df["Bin_Z0_0.0001"] = pd.cut(
    wind_comparison_df["Wind_Z0_0.0001"],
    bins=wind_bin_edges,
    labels=wind_bin_labels,
    include_lowest=True,
    right=True
)

changed_Z0001 = (
    wind_comparison_df["Bin_Z0_0.001"] !=
    wind_comparison_df["Bin_Z0_0.02"]
)

changed_Z00001 = (
    wind_comparison_df["Bin_Z0_0.0001"] !=
    wind_comparison_df["Bin_Z0_0.02"]
)

number_changed_Z0001 = changed_Z0001.sum()
number_changed_Z00001 = changed_Z00001.sum()

percent_changed_Z0001 = (
    100 *
    number_changed_Z0001 /
    len(wind_comparison_df)
)

percent_changed_Z00001 = (
    100 *
    number_changed_Z00001 /
    len(wind_comparison_df)
)

print("Compared with z0 = 0.02 m:")
print(
    f"z0 = 0.001 m: "
    f"{number_changed_Z0001} legs changed bins "
    f"({percent_changed_Z0001:.1f}%)"
)

print(
    f"z0 = 0.0001 m: "
    f"{number_changed_Z00001} legs changed bins "
    f"({percent_changed_Z00001:.1f}%)"
)
# %%
