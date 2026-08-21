#%%
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import datetime
import pathlib
import statistics
import pickle
import mputil
import shutil
import glob
import os
import re
import math
from matplotlib.colors import BoundaryNorm
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
from collections import defaultdict
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy.ma as ma
import matplotlib.patheffects as path_effects
from scipy.interpolate import interp1d
import matplotlib.colors as mcolors
#%%
#Now CDP data
#This is how we will correct our droplet concentration units from 
#dN/dlogD to dN/dD
#We will use the bin width to convert the units
L_00=math.log10(3)-math.log10(2)
L_01=math.log10(4)-math.log10(3)
L_02=math.log10(5)-math.log10(4)
L_03=math.log10(6)-math.log10(5)
L_04=math.log10(7)-math.log10(6)
L_05=math.log10(8)-math.log10(7)
L_06=math.log10(9)-math.log10(8)
L_07=math.log10(10)-math.log10(9)
L_08=math.log10(11)-math.log10(10)
L_09=math.log10(12)-math.log10(11)
L_10=math.log10(13)-math.log10(12)
L_11=math.log10(14)-math.log10(13)
L_12=math.log10(16)-math.log10(14)
L_13=math.log10(18)-math.log10(16)
L_14=math.log10(20)-math.log10(18)
L_15=math.log10(22)-math.log10(20)
L_16=math.log10(24)-math.log10(22)
L_17=math.log10(26)-math.log10(24)
L_18=math.log10(28)-math.log10(26)
L_19=math.log10(30)-math.log10(28)
L_20=math.log10(32)-math.log10(30)
L_21=math.log10(34)-math.log10(32)
L_22=math.log10(36)-math.log10(34)
L_23=math.log10(38)-math.log10(36)
L_24=math.log10(40)-math.log10(38)
L_25=math.log10(42)-math.log10(40)
L_26=math.log10(44)-math.log10(42)
L_27=math.log10(46)-math.log10(44)
L_28=math.log10(48)-math.log10(46)
L_29=math.log10(50)-math.log10(48)


bin_log_CDP=[L_00, L_01, L_02, L_03, L_04, L_05, L_06, L_07, L_08,
          L_09, L_10, L_11,
          L_12, L_13, L_14, L_15, L_16, 
        L_17, L_18, L_19, L_20, L_21, L_22, L_23, 
        L_24, L_25, L_26, L_27, L_28, L_29]


P00=(3-2)
P01=(4-3)
P02=(5-4)
P03=(6-5)
P04=(7-6)
P05=(8-7)
P06=(9-8)
P07=(10-9)
P08=(11-10)
P09=(12-11)
P10=(13-12)
P11=(14-13)
P12 = (16-14)
P13 = (18-16)
P14 = (20-18)
P15 = (22-20)
P16 = (24-22)
P17 = (26-24)
P18 = (28-26)
P19 = (30-28)
P20 = (32-30)
P21 = (34-32)
P22 = (36-34)
P23 = (38-36)
P24 = (40-38)
P25 = (42-40)
P26 = (44-42)
P27 = (46-44)
P28 = (48-46)
P29 = (50-48)


J00=(L_00 / P00)
J01=(L_01 / P01)
J02=(L_02 / P02)
J03=(L_03 / P03)
J04=(L_04 / P04)
J05=(L_05 / P05)
J06=(L_06 / P06)
J07=(L_07 / P07)
J08=(L_08 / P08)
J09=(L_09 / P09)
J10=(L_10 / P10)
J11=(L_11 / P11)
J12 = (L_12 / P12)
J13 = (L_13 / P13)
J14 = (L_14 / P14)
J15 = (L_15 / P15)
J16 = (L_16 / P16)
J17 = (L_17 / P17)
J18 = (L_18 / P18)
J19 = (L_19 / P19)
J20 = (L_20 / P20)
J21 = (L_21 / P21)
J22 = (L_22 / P22)
J23 = (L_23 / P23)
J24 = (L_24 / P24)
J25 = (L_25 / P25)
J26 = (L_26 / P26)
J27 = (L_27 / P27)
J28 = (L_28 / P28)
J29 = (L_29 / P29)


Logg_CDP = [J00, J01, J02, J03, J04, J05, J06, J07, J08, J09, J10, 
            J11, J12, J13, J14, J15, J16, J17, J18, J19, J20, J21,
            J22, J23, J24, J25, J26, J27, J28, J29]

Logg_CDP = np.array(Logg_CDP)
bin_center_CDP=[2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 
            10.5, 11.5, 12.5, 13.5, 15, 17, 19, 
            21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49]
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
#Import the instrument data for the cloud droplet probe 
#Make sure to work with bins 0-30 for the coarse mode aerosol
bin_name_CDP = ['CDP_Bin00', 'CDP_Bin01', 'CDP_Bin02', 'CDP_Bin03', 
            'CDP_Bin04', 'CDP_Bin05', 'CDP_Bin06', 'CDP_Bin07', 
            'CDP_Bin08', 'CDP_Bin09', 'CDP_Bin11', 'CDP_Bin12',
            'CDP_Bin13', 'CDP_Bin14', 'CDP_Bin15', 'CDP_Bin16', 
            'CDP_Bin17', 'CDP_Bin18', 'CDP_Bin19', 'CDP_Bin20', 
            'CDP_Bin21', 'CDP_Bin22', 'CDP_Bin23', 'CDP_Bin24', 
            'CDP_Bin25', 'CDP_Bin26', 'CDP_Bin27',
            'CDP_Bin28', 'CDP_Bin29']

CDP_1Hz = []

dates_CDP = ['2022-01-11', '2022-01-12','2022-01-15', '2022-01-18', 
             '2022-01-19', '2022-01-24', '2022-01-26', '2022-01-27',
             '2022-02-01', '2022-02-02', '2022-02-03', '2022-02-05', 
             '2022-02-15', '2022-02-16', '2022-02-19', '2022-02-22',
             '2022-02-26', #'2022-03-02',
             '2022-03-03', '2022-03-04', 
             '2022-03-13', '2022-03-14', '2022-03-18', '2022-03-22',
             '2022-03-26', '2022-03-28', '2022-03-29', #'2022-05-03',
             '2022-05-05', '2022-05-10','2022-05-16', '2022-05-17',
             '2022-05-18',
             '2022-05-20','2022-05-21', '2022-05-31', '2022-06-02', 
             '2022-06-03', '2022-06-05','2022-06-07', '2022-06-08', 
             '2022-06-10','2022-06-11','2022-06-13', '2022-06-14',
             '2022-06-17', '2022-06-18']

for date in dates_CDP:
    dataset = {'Date': date, 'Clear Means': [], 'Cloud Means': []}

    file_path = f'/home/disk/eos4/kathem24/activate/data/CDP/2022/csv/CDP_1Hz_files/CDP_1Hz_{date}.csv'
    if not os.path.exists(file_path):
        print(f"File not found for date {date}: {file_path}")
        continue
    df_CDP = pd.read_csv(file_path)

    print(f"First rows for date {date}:")
    print(df_CDP.head())
#%%
for date in dates_CDP:
    file_path = f"/home/disk/eos4/kathem24/activate/data/CDP/2022/csv/CDP_1Hz_files/CDP_1Hz_{date}.csv"

    if not os.path.exists(file_path):
        print(f"File not found for date {date}: {file_path}")
        continue

    df_CDP = pd.read_csv(file_path)
    print(f"Loaded file for {date}:")
    print(df_CDP.head()) 
    CDP_1Hz.append(df_CDP)

#%%
print(f"Number of entries in CDP_1Hz: {len(CDP_1Hz)}")
if CDP_1Hz:
    print("Sample entry in CDP_1Hz:")
    print(CDP_1Hz[0].head())
#%%
for i, df in enumerate(CDP_1Hz):
    print(f"Date {dates_CDP[i]} matches file content with Date column:")
    print(df['Date'].unique())
#%%
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
#%%
#CDP
in_cloud_concentrations_CDP = []
for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict = leg_data[i]
    ACB_start = leg_dict['LegIndex_03']['StartTimes']
    ACB_stop = leg_dict['LegIndex_03']['StopTimes']
    BCT_start =leg_dict['LegIndex_04']['StartTimes']
    BCT_stop = leg_dict['LegIndex_04']['StopTimes']
    CDP_flight = CDP_1Hz[i]
    twoDS_flight = twoDS[i]

    CDP_flight['Time_Start'] = pd.to_numeric(CDP_flight['Time_Start'], errors='coerce')
    twoDS_flight['Time_Start'] = pd.to_numeric(twoDS_flight['Time_Start'], errors='coerce')

    CDP_times = CDP_flight['Time_Start'].values
    CDP_lwc = CDP_flight['LWC_CDP'].values
    CDP_bins = {f'CDP_Bin{bin_label:02d}': CDP_flight[f'CDP_Bin{bin_label:02d}'].values for bin_label in range(0, 30)}
    TwoDS_times = twoDS_flight['Time_Start'].values

    for k in range(len(ACB_start)):
        start_time = ACB_start[k]
        end_time = ACB_stop[k]

        CDP_indices_in_range = np.where((CDP_times >= start_time) & (CDP_times <= end_time))[0]

        for CDP_idx in zip(CDP_indices_in_range):
            lwc_val = CDP_lwc[CDP_idx]

            if lwc_val >= 0.01:
                calc_entry = {
                    'Date': date,
                    'Time': CDP_times[CDP_idx],
                    'BCB_start': start_time,
                    'BCB_stop': end_time,
                    'CWC': lwc_val,
                }

                for bin_label in range(0, 30):
                    bin_key = f'Bin{bin_label}_concentration'
                    calc_entry[bin_key] = CDP_bins[f'CDP_Bin{bin_label:02d}'][CDP_idx]

                in_cloud_concentrations_CDP.append(calc_entry)
#%%
#adding BCT and ACB legs together in a combined dictionary 
in_cloud_concentrations_CDP = []
for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict = leg_data[i]
    ACB_start = leg_dict['LegIndex_03']['StartTimes']
    ACB_stop = leg_dict['LegIndex_03']['StopTimes']
    BCT_start = leg_dict['LegIndex_04']['StartTimes']
    BCT_stop = leg_dict['LegIndex_04']['StopTimes']

    CDP_flight = CDP_1Hz[i]
    twoDS_flight = twoDS[i]

    CDP_flight['Time_Start'] = pd.to_numeric(CDP_flight['Time_Start'], errors='coerce')

    CDP_times = CDP_flight['Time_Start'].values
    CDP_lwc = CDP_flight['LWC_CDP'].values
    CDP_bins = {f'CDP_Bin{bin_label:02d}': CDP_flight[f'CDP_Bin{bin_label:02d}'].values for bin_label in range(0, 30)}

    combined_legs = [
        (ACB_start, ACB_stop),
        (BCT_start, BCT_stop)
    ]

    for leg_start, leg_stop in combined_legs:
        for k in range(len(leg_start)):
            start_time = leg_start[k]
            end_time = leg_stop[k]

            CDP_indices_in_range = np.where((CDP_times >= start_time) & (CDP_times <= end_time))[0]

            for CDP_idx in CDP_indices_in_range:
                lwc_val = CDP_lwc[CDP_idx]

                if lwc_val >= 0.01:  # Adjust LWC threshold as needed
                    calc_entry = {
                        'Date': date,
                        'Time': CDP_times[CDP_idx],
                        'Leg_start': start_time,
                        'Leg_stop': end_time,
                        'CWC': lwc_val  # Cloud water content
                    }

                    for bin_label in range(0, 30):
                        bin_key = f'Bin{bin_label}_concentration'
                        calc_entry[bin_key] = CDP_bins[f'CDP_Bin{bin_label:02d}'][CDP_idx]
                    in_cloud_concentrations_CDP.append(calc_entry)
print(f"Number of in-cloud entries: {len(in_cloud_concentrations_CDP)}")
print(f"First 5 entries: {in_cloud_concentrations_CDP[:5]}")
#%%
# This code calculates total concentration in cm³
in_cloud_concentrations_CDP = []
for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict = leg_data[i]

    ACB_start = leg_dict['LegIndex_03']['StartTimes']
    ACB_stop = leg_dict['LegIndex_03']['StopTimes']
    BCT_start = leg_dict['LegIndex_04']['StartTimes']
    BCT_stop = leg_dict['LegIndex_04']['StopTimes']

    CDP_flight = CDP_1Hz[i]

    CDP_flight['Time_Start'] = pd.to_numeric(CDP_flight['Time_Start'], errors='coerce')

    CDP_times = CDP_flight['Time_Start'].values
    CDP_lwc = CDP_flight['LWC_CDP'].values
    CDP_bins = {f'CDP_Bin{bin_label:02d}': CDP_flight[f'CDP_Bin{bin_label:02d}'].values for bin_label in range(0, 30)}

    bin_widths = [bin_log_CDP[bin_label] for bin_label in range(0, 30)]

    all_legs_start = ACB_start + BCT_start
    all_legs_stop = ACB_stop + BCT_stop

    for k in range(len(all_legs_start)):
        start_time = all_legs_start[k]
        end_time = all_legs_stop[k]

        CDP_indices_in_range = np.where((CDP_times >= start_time) & (CDP_times <= end_time))[0]

        for CDP_idx in CDP_indices_in_range:
            lwc_val = CDP_lwc[CDP_idx]

          
            if lwc_val >= 0.01:
                total_concentration = sum(
                    np.nan_to_num(CDP_bins[f'CDP_Bin{bin_label:02d}'][CDP_idx]) * bin_width
                    for bin_label, bin_width in zip(range(0, 30), bin_widths)
                )

                
                calc_entry = {
                    'Date': date,
                    'Time': CDP_times[CDP_idx],
                    'Leg_start': start_time,
                    'Leg_stop': end_time,
                    'CWC': lwc_val,
                    'Total_Concentration': total_concentration  # Units: cm³
                }
                in_cloud_concentrations_CDP.append(calc_entry)
print(f"Number of in-cloud entries: {len(in_cloud_concentrations_CDP)}")
print(f"Sample entries: {in_cloud_concentrations_CDP[:5]}")
#%%
#CAS now 
in_cloud_concentrations_CAS = []

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

                in_cloud_concentrations_CAS.append(calc_entry)

#%%
#adding BCT and ACB legs together in a combined dictionary 
in_cloud_concentrations_CAS = []

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

                    in_cloud_concentrations_CAS.append(calc_entry)
                    in_cloud_concentrations_CAS.append(calc_entry)

print(f"Number of in-cloud entries: {len(in_cloud_concentrations_CAS)}")
print(f"First 5 entries: {in_cloud_concentrations_CAS[:5]}")
#%%
# This code calculates total concentration in cm³
in_cloud_concentrations_CAS = []

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

                
                in_cloud_concentrations_CAS.append(calc_entry)
print(f"Number of in-cloud entries: {len(in_cloud_concentrations_CAS)}")
print(f"Sample entries: {in_cloud_concentrations_CAS[:5]}")
#%%
#combined 
#combined 
cas_cloud_df = pd.DataFrame(
    in_cloud_concentrations_CAS)
cdp_cloud_df = pd.DataFrame(
    in_cloud_concentrations_CDP)
print("CAS in-cloud entries:",
    len(cas_cloud_df))
print("CDP in-cloud entries:",
    len(cdp_cloud_df))
cas_cloud_df["Time"] = pd.to_numeric(
    cas_cloud_df["Time"],
    errors="coerce")
cdp_cloud_df["Time"] = pd.to_numeric(
    cdp_cloud_df["Time"],
    errors="coerce")
# %%
cas_cloud_df["Time_match"] = np.floor(
    cas_cloud_df["Time"])
cdp_cloud_df["Time_match"] = np.floor(
    cdp_cloud_df["Time"])
# %%
cas_1hz = (
    cas_cloud_df
    .groupby(
        ["Date", "Time_match"],
        as_index=False )
    .agg({
        "CWC": "mean",
        "Total_Concentration": "mean"}))
cdp_1hz = (
    cdp_cloud_df
    .groupby(
        ["Date", "Time_match"],
        as_index=False)
    .agg({
        "CWC": "mean",
        "Total_Concentration": "mean"}))
cas_1hz = cas_1hz.rename(
    columns={
        "CWC": "CAS_CWC",
        "Total_Concentration":
            "CAS_Total_Concentration"})
cdp_1hz = cdp_1hz.rename(
    columns={
        "CWC": "CDP_CWC",
        "Total_Concentration":
            "CDP_Total_Concentration"})

# %%
combined_cloud_df = pd.merge(
    cas_1hz,
    cdp_1hz,
    on=["Date", "Time_match"],
    how="inner")
print(
    "Matched CAS + CDP seconds:",
    len(combined_cloud_df))
# %%
# Average CAS and CDP LWC
combined_cloud_df["CWC"] = (
    combined_cloud_df[
        ["CAS_CWC", "CDP_CWC"]]
    .mean(axis=1))
combined_cloud_df[
    "Total_Concentration"] = (combined_cloud_df[
        ["CAS_Total_Concentration",
            "CDP_Total_Concentration"  ]]
    .mean(axis=1))
combined_cloud_df["Time"] = (
    combined_cloud_df["Time_match"])
# %%
combined_cloud_df = combined_cloud_df[
    np.isfinite(combined_cloud_df["CWC"]) &
    np.isfinite(
        combined_cloud_df[
            "Total_Concentration"])
].copy()
# %%
in_cloud_concentrations_combined = (
    combined_cloud_df[
        ["Date",
            "Time",
            "CWC",
            "Total_Concentration" ]]
    .to_dict("records"))
in_cloud_concentrations = (
    in_cloud_concentrations_combined)
print("Final combined CAS + CDP "
    "in-cloud entries:",
    len(in_cloud_concentrations))
print("First 5 combined entries:")
print(in_cloud_concentrations[:5])
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
# %%
# Add RWC and CWC for total LWC

# Make a fast lookup table using Date + Time
cwc_lookup = {
    (entry['Date'], entry['Time']): entry
    for entry in in_cloud_concentrations
}

total_liquid_water = []

for rwc_entry in rain_water_content:

    matching_time = rwc_entry['Time']
    matching_date = rwc_entry['Date']

    matching_cwc = cwc_lookup.get(
        (matching_date, matching_time)
    )

    if matching_cwc is not None:

        cwc_val = matching_cwc['CWC']
        rwc_val = rwc_entry['RWC']

        total_liquid = (
            cwc_val +
            rwc_val
        )

        total_liquid_water.append({
            'Date': matching_date,
            'Time': matching_time,
            'Leg_start': rwc_entry['Leg_start'],
            'Leg_stop': rwc_entry['Leg_stop'],
            'CWC': cwc_val,
            'RWC': rwc_val,
            'Total_Liquid_Water': total_liquid
        })


print(
    "Number of total liquid water entries:",
    len(total_liquid_water)
)

print(
    "First 5 entries:",
    total_liquid_water[:5]
)
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
# %%
# Add Nc + Nr for total concentration

# Make fast lookup table for 2D-S rain concentration
rain_lookup = {
    (entry['Date'], entry['Time']): entry
    for entry in rain_concentrations
}

total_combined_concentration = []

for in_cloud_entry in in_cloud_concentrations:

    matching_time = in_cloud_entry['Time']
    matching_date = in_cloud_entry['Date']

    matching_rain = rain_lookup.get(
        (matching_date, matching_time)
    )

    if matching_rain is not None:

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

print(
    "Number of total combined concentration entries:",
    len(total_combined_concentration)
)

print(
    "First 5 entries:",
    total_combined_concentration[:5]
)

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
plt.title('CDP in-cloud January - June 2022', fontsize=18, fontweight='bold')
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
plt.title('CDP (in cloud) \nJanuary-June 2022', fontsize=20, fontweight='bold')
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
plt.title('CDP (in cloud)\n January-June 2022\n RWC as a function of number concentration', fontsize=18, fontweight='bold')
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
plt.title('CDP (in cloud)\nMean RWC\nJanuary–June 2022', fontsize=18, fontweight='bold')
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
plt.title('Mean LWC\nJanuary–June 2022 (CDP in cloud)', fontsize=18, fontweight='bold')
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
plt.title('CDP in-cloud January-June 2022', fontsize=18, fontweight='bold')
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
plt.title('CDP (in-cloud)\n January-June 2022)', fontsize=19, fontweight='bold')
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
plt.title('CDP(in-cloud)\nJanuary-June 2022', fontsize=19, fontweight='bold')
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
plt.title('CDP (in cloud)\n January-June 2022\n RWC as a function of number concentration', fontsize=18, fontweight='bold')
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
plt.title('CDP (in-cloud)\n January-June 2022\n RWC as a function of number concentration', fontsize=18, fontweight='bold')
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
    weights=rain_water_content_values)
sum_lwc, _, _ = np.histogram2d(
    concentration,
    total_liquid_water_values,
    bins=[x_bins, y_bins],
    weights=total_liquid_water_values)
counts, _, _ = np.histogram2d(
    concentration,
    total_liquid_water_values,
    bins=[x_bins, y_bins])

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
    shading='auto')
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
plt.title('CDP (in-cloud)\nJanuary–June 2022\nRWC as a function of number concentration',
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
#%%
# Import combined CAS + CDP flight-mean GCCN mass
mass_file = (
    "/home/disk/p/kathem24/activate/ACTIVATE-2024-2025/"
    "CAS/Below Cloud Base/Scripts/"
    "CAS_CDP_GCCN_mass_flight_mean_2022.pkl")
with open(mass_file, "rb") as f:
    average_mass_per_flight = pickle.load(f)
print("Loaded combined CAS + CDP " 
      "flight-mean GCCN mass.")
print("Number of flights:",len(average_mass_per_flight))
#%%
mass_values = np.array(list(average_mass_per_flight.values()))
mass_threshold = np.percentile(mass_values, 50)
high_mass_flights = {}
low_mass_flights = {}
for date, avg_mass in average_mass_per_flight.items():
    if avg_mass >= mass_threshold:
        high_mass_flights[date] = avg_mass
    else:
        low_mass_flights[date] = avg_mass
print(f"\nMass Threshold: Low < {mass_threshold:.2f} µg/m³, High ≥ {mass_threshold:.2f} µg/m³")
#%%
mass_values = np.array(list(average_mass_per_flight.values()))
high_mass_values = np.array(list(high_mass_flights.values()))
low_mass_values = np.array(list(low_mass_flights.values()))
df_mass = pd.DataFrame({
    "GCCN Mass (µg/m³)": np.concatenate([high_mass_values, low_mass_values]),
    "Flight Type": ["High GCCN Mass"] * len(high_mass_values) + ["Low GCCN Mass"] * len(low_mass_values)
})
plt.figure(figsize=(8, 6))
sns.violinplot(
    x="Flight Type",
    y="GCCN Mass (µg/m³)",
    data=df_mass,
    inner="box",
    palette=["darkorange", "goldenrod"],
    scale="width"
)
plt.yscale('log')
plt.ylabel("GCCN Mass (µg/m³)", fontsize=20, fontweight="bold")
plt.xlabel("GCCN Flight Category", fontsize=20, fontweight="bold")
plt.title("Comparison of High & Low GCCN Mass Flight Categories", fontsize=16, fontweight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tick_params(axis="both", which="major", labelsize=18, width=3, length=8)
plt.tick_params(axis="both", which="minor", labelsize=18, width=2, length=5)
plt.tight_layout()
plt.show()

#%%
avg_high_mass = np.mean(high_mass_values)
avg_low_mass = np.mean(low_mass_values)
num_high_mass_flights = len(high_mass_values)
num_low_mass_flights = len(low_mass_values)
print(f"Average High GCCN Mass Flight: {avg_high_mass:.2f} µg/m³")
print(f"Number of High GCCN Mass Flights: {num_high_mass_flights}")
print(f"Average Low GCCN Mass Flight: {avg_low_mass:.2f} µg/m³")
print(f"Number of Low GCCN Mass Flights: {num_low_mass_flights}")
#%%
#save as a pickle 
combined_mass_flight_data = {
    "average_mass_per_flight": average_mass_per_flight,
    "mass_threshold": mass_threshold,
    "high_mass_flights": high_mass_flights,
    "low_mass_flights": low_mass_flights
}

with open(
    "CAS_CDP_GCCN_mass_flight_split_2022.pkl",
    "wb"
) as f:
    pickle.dump(
        combined_mass_flight_data,
        f
    )

print("Saved combined CAS + CDP GCCN mass flight data.")
print("Total flights:", len(average_mass_per_flight))
print("High mass flights:", len(high_mass_flights))
print("Low mass flights:", len(low_mass_flights))
print(f"Mass threshold: {mass_threshold:.2f} µg/m³")
#%%
# import pickle
# #mass uncertainty
# BASE_DIR = (
#     "/home/disk/p/kathem24/activate/"
#     "ACTIVATE-2024-2025/CDP/below cloud base"
# )
# with open(
#     f"{BASE_DIR}/CDP_mass_uncertainty_massLE1002022.pkl",
#     "rb"
# ) as f:
#     filtered_dry_mass_inf_CDP = pickle.load(f)
# mass_flight_totals = defaultdict(
#     lambda: {
#         'Legs': [],
#         'Total_GCCN_Mass': 0,
#         'Leg_Count': 0
#     }
# )
# for entry in filtered_dry_mass_inf_CDP:
#     date = entry['Date']
#     start_time = entry['BCB_start']
#     stop_time = entry['BCB_stop']
#     total_mass = entry['Dry Mass (µg/m³)']
#     mass_flight_totals[date]['Legs'].append({
#         'Leg_start': start_time,
#         'Leg_stop': stop_time,
#         'Leg_GCCN_Mass': total_mass
#     })
#     mass_flight_totals[date]['Total_GCCN_Mass'] += total_mass
#     mass_flight_totals[date]['Leg_Count'] += 1
# mass_flight_totals = dict(mass_flight_totals)
# #%%
# #%%
# # Rebuild the CDP flight totals with mass uncertainty included
# mass_flight_totals = defaultdict(
#     lambda: {
#         'Legs': [],
#         'Total_GCCN_Mass': 0,
#         'Total_Mass_Uncertainty_Squared': 0,
#         'Leg_Count': 0
#     })
# for entry in filtered_dry_mass_inf_CDP:
#     date = entry['Date']
#     start_time = entry['BCB_start']
#     stop_time = entry['BCB_stop']
#     total_mass = entry[
#         'Dry Mass (µg/m³)'    ]
#     mass_uncertainty = entry[
#         'Dry Mass Uncertainty 1sigma (µg/m³)'    ]
#     fractional_mass_uncertainty = entry[
#         'Dry Mass Fractional Uncertainty 1sigma'    ]
#     mass_flight_totals[date]['Legs'].append({
#         'Leg_start':
#             start_time,
#         'Leg_stop':
#             stop_time,
#         'Leg_GCCN_Mass':
#             total_mass,
#         'Leg_GCCN_Mass_Uncertainty_1sigma':
#             mass_uncertainty,
#         'Leg_GCCN_Mass_Fractional_Uncertainty_1sigma':
#             fractional_mass_uncertainty
#     })
#     mass_flight_totals[date][
#         'Total_GCCN_Mass'
#     ] += total_mass
#     mass_flight_totals[date][
#         'Total_Mass_Uncertainty_Squared'
#     ] += mass_uncertainty**2
#     mass_flight_totals[date][
#         'Leg_Count'
#     ] += 1
# mass_flight_totals = dict(
#     mass_flight_totals)
# #%%
# # Calculate CDP flight-mean mass uncertainty
# mass_uncertainty_per_flight = {}
# for date, flight_data in mass_flight_totals.items():
#     leg_count = flight_data[
#         "Leg_Count"    ]
#     if leg_count > 0:
#         mean_mass = (
#             flight_data[
#                 "Total_GCCN_Mass"
#             ] /
#             leg_count        )
#         mean_mass_uncertainty_1sigma = (
#             np.sqrt(
#                 flight_data[
#                     "Total_Mass_Uncertainty_Squared"
#                 ]
#             ) /
#             leg_count        )
#         if mean_mass > 0:
#             fractional_mass_uncertainty_1sigma = (
#                 mean_mass_uncertainty_1sigma /
#                 mean_mass
#             )
#         else:

#             fractional_mass_uncertainty_1sigma = np.nan
#         mass_uncertainty_per_flight[date] = {

#             "Mean_GCCN_Mass":
#                 mean_mass,

#             "Mean_GCCN_Mass_Uncertainty_1sigma":
#                 mean_mass_uncertainty_1sigma,

#             "Mean_GCCN_Mass_Fractional_Uncertainty_1sigma":
#                 fractional_mass_uncertainty_1sigma,

#             "Mean_GCCN_Mass_Fractional_Uncertainty_1sigma (%)":
#                 100 *
#                 fractional_mass_uncertainty_1sigma
#         }
# print(
#     "\nCDP flight-mean GCCN mass uncertainties:"
# )
# for date, values in mass_uncertainty_per_flight.items():
#     print(
#         f"{date}: "
#         f"{values['Mean_GCCN_Mass']:.2f} ± "
#         f"{values['Mean_GCCN_Mass_Uncertainty_1sigma']:.2f} ± "
#         f"{values['Mean_GCCN_Mass_Uncertainty_1sigma']:.2f} µg/m³ "
#         f"({values['Mean_GCCN_Mass_Fractional_Uncertainty_1sigma (%)']:.2f}%)"
#     )
# #%%
# #mass variability within each flight 
# mass_variability_per_flight = {}
# for date, flight_data in mass_flight_totals.items():
#     leg_masses = np.asarray([
#         leg["Leg_GCCN_Mass"]
#         for leg in flight_data["Legs"]
#     ], dtype=float)
#     leg_masses = leg_masses[
#         np.isfinite(leg_masses)]
#     number_of_legs = len(leg_masses)
#     if number_of_legs >= 2:
#         mean_mass = np.mean(
#             leg_masses)
#         within_flight_sd = np.std(
#             leg_masses,
#             ddof=1)
#         within_flight_sem = (
#             within_flight_sd /
#             np.sqrt(number_of_legs))
#         if mean_mass > 0:
#             coefficient_of_variation_percent = (
#                 100 *
#                 within_flight_sd /
#                 mean_mass)
#             relative_sem_percent = (
#                 100 *
#                 within_flight_sem /
#                 mean_mass)
#         else:
#             coefficient_of_variation_percent = np.nan
#             relative_sem_percent = np.nan
#         mass_variability_per_flight[date] = {
#             "Mean_GCCN_Mass":
#                 mean_mass,
#             "Number_of_Legs":
#                 number_of_legs,
#             "Within_Flight_SD":
#                 within_flight_sd,
#             "Within_Flight_SEM":
#                 within_flight_sem,
#             "Coefficient_of_Variation (%)":
#                 coefficient_of_variation_percent,
#             "Relative_SEM (%)":
#                 relative_sem_percent}
# print(
#     "\nCDP within-flight mass variability:")
# for date, values in mass_variability_per_flight.items():
#     print(
#         f"{date}: "
#         f"Mean = {values['Mean_GCCN_Mass']:.2f} µg/m³, "
#         f"SD = {values['Within_Flight_SD']:.2f} µg/m³, "
#         f"SEM = {values['Within_Flight_SEM']:.2f} µg/m³, "
#         f"Relative SEM = {values['Relative_SEM (%)']:.2f}%, "
#         f"N legs = {values['Number_of_Legs']}")
# #%%
# # CDP within-flight mass variability
# relative_sem_mass_values_CDP = np.asarray([
#     values["Relative_SEM (%)"]
#     for values in mass_variability_per_flight.values()
# ], dtype=float)
# cv_mass_values_CDP = np.asarray([
#     values["Coefficient_of_Variation (%)"]
#     for values in mass_variability_per_flight.values()
# ], dtype=float)
# relative_sem_mass_values_CDP = (
#     relative_sem_mass_values_CDP[
#         np.isfinite(
#             relative_sem_mass_values_CDP)])
# cv_mass_values_CDP = (
#     cv_mass_values_CDP[
#         np.isfinite(
#             cv_mass_values_CDP)])
# print(
#     "\nNumber of CDP flights with at least two mass legs:",
#     len(mass_variability_per_flight))
# print(
#     "Mean relative mass SEM:",
#     f"{np.mean(relative_sem_mass_values_CDP):.2f}%")
# print(
#     "Median relative mass SEM:",
#     f"{np.median(relative_sem_mass_values_CDP):.2f}%")
# print(
#     "Relative mass SEM 25th–75th percentile:",
#     f"{np.percentile(relative_sem_mass_values_CDP, 25):.2f}% to "
#     f"{np.percentile(relative_sem_mass_values_CDP, 75):.2f}%")
# print(
#     "Median within-flight mass coefficient of variation:",
#     f"{np.median(cv_mass_values_CDP):.2f}%")
# #%%
# # Plot relative SEM against mean CDP mass
# mean_masses_CDP = np.asarray([
#     values["Mean_GCCN_Mass"]
#     for values in mass_variability_per_flight.values()
# ], dtype=float)
# relative_sem_mass_percent_CDP = np.asarray([
#     values["Relative_SEM (%)"]
#     for values in mass_variability_per_flight.values()
# ], dtype=float)
# number_of_mass_legs_CDP = np.asarray([
#     values["Number_of_Legs"]
#     for values in mass_variability_per_flight.values()
# ], dtype=float)
# valid_mass_CDP = (
#     np.isfinite(mean_masses_CDP) &
#     np.isfinite(relative_sem_mass_percent_CDP) &
#     (mean_masses_CDP > 0))
# mean_masses_CDP = mean_masses_CDP[
#     valid_mass_CDP]
# relative_sem_mass_percent_CDP = (
#     relative_sem_mass_percent_CDP[
#         valid_mass_CDP])
# number_of_mass_legs_CDP = number_of_mass_legs_CDP[
#     valid_mass_CDP]
# median_relative_mass_sem_CDP = np.median(
#     relative_sem_mass_percent_CDP)
# fig, ax = plt.subplots(figsize=(8, 6))
# scatter = ax.scatter(
#     mean_masses_CDP,
#     relative_sem_mass_percent_CDP,
#     s=40 + 20 * number_of_mass_legs_CDP,
#     alpha=0.8,
#     edgecolor="black")
# ax.axhline(
#     median_relative_mass_sem_CDP,
#     linestyle="--",
#     linewidth=2,
#     label=(
#         f"Median relative SEM = "
#         f"{median_relative_mass_sem_CDP:.2f}%"))
# ax.set_xscale("log")
# ax.set_xlabel(
#     r"Mean GCCN Mass ($\mu$g m$^{-3}$)",
#     fontsize=16,
#     fontweight="bold")
# ax.set_ylabel(
#     r"Relative SEM, "
#     r"$100(\mathrm{SEM}/\overline{M})$ (%)",
#     fontsize=16,
#     fontweight="bold")
# ax.set_title(
#     "CDP Relative Uncertainty in Flight-Mean GCCN Mass",
#     fontsize=16,
#     fontweight="bold")
# ax.grid(
#     linestyle="--",
#     alpha=0.5)
# ax.legend(
#     fontsize=12)
# ax.text(
#     0.02,
#     0.90,
#     "Marker size represents number of BCB legs",
#     transform=ax.transAxes,
#     ha="left",
#     va="top",
#     fontsize=11,
#     fontweight="bold")
# ax.tick_params(
#     axis="both",
#     which="major",
#     labelsize=14,
#     width=2,
#     length=6)
# for tick_label in (
#     ax.get_xticklabels() + ax.get_yticklabels()):
#     tick_label.set_fontweight("bold")
# plt.tight_layout()
# plt.show()
#%%
high_mass_data = [entry for entry in total_combined_concentration if entry['Date'] in high_mass_flights]
low_mass_data = [entry for entry in total_combined_concentration if entry['Date'] in low_mass_flights]

high_concentration = np.array([entry['Total_Combined_Concentration'] for entry in high_mass_data])
high_lwc = np.array([entry['Total_Liquid_Water'] for entry in total_liquid_water if entry['Date'] in high_mass_flights])
high_rwc = np.array([entry['RWC'] for entry in total_liquid_water if entry['Date'] in high_mass_flights])

low_concentration = np.array([entry['Total_Combined_Concentration'] for entry in low_mass_data])
low_lwc = np.array([entry['Total_Liquid_Water'] for entry in total_liquid_water if entry['Date'] in low_mass_flights])
low_rwc = np.array([entry['RWC'] for entry in total_liquid_water if entry['Date'] in low_mass_flights])
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
counts_cdp_high = counts_high.copy()
counts_cdp_low  = counts_low.copy()
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
plt.title('High GCCN Mass Flights (CDP Jan–Jun 2022)', fontsize=19, fontweight='bold')
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
plt.title('Low GCCN Mass Flights (CDP Jan–Jun 2022)', fontsize=19, fontweight='bold')
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
plt.title('High GCCN Mass CDP Flights\n January-June 2022', fontsize=18, fontweight='bold')
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
plt.title('Low GCCN Mass CDP Flights \nJanuary-June 2022', fontsize=18, fontweight='bold')
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
plt.title('Mean RWC — High GCCN Mass CDP Flights \nJanuary–June 2022)', fontsize=19, fontweight='bold')
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
plt.title('Mean RWC — Low GCCN Mass CDP Flights January–June 2022', fontsize=19, fontweight='bold')
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
plt.title('Mean RWC\nHigh CDP GCCN Mass Flights\nJanuary–June 2022', fontsize=19, fontweight='bold')
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
plt.title('Mean RWC\nLow GCCN Mass CDP Flights\nJanuary–June 2022', fontsize=19, fontweight='bold')
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
plt.title('Mean LWC — High GCCN Mass CDP Flights January–June 2022', fontsize=19, fontweight='bold')
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
plt.title('Mean LWC — Low GCCN Mass CDP Flights January–June 2022', fontsize=19, fontweight='bold')
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
plt.title('Mean RWC\nHigh CDP GCCN Mass Flights\nJanuary–June 2022', fontsize=19, fontweight='bold')
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
plt.title('Mean RWC\nLow CDP GCCN Mass Flights\nJanuary–June 2022', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
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

# === Overlay bin counts ===
x_centers = 0.5 * (xedges[:-1] + xedges[1:])
y_centers = 0.5 * (yedges[:-1] + yedges[1:])
for i in range(len(x_centers)):
    for j in range(len(y_centers)):
        count = counts_high[i, j]
        if count > 0:
            plt.text(x_centers[i], y_centers[j], str(int(count)),
                     ha='center', va='center', fontsize=13,
                     fontweight='bold', color='black')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Mean RWC\nHigh CDP GCCN Mass Flights\nJanuary–June 2022', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
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

# === Overlay bin counts ===
x_centers = 0.5 * (xedges[:-1] + xedges[1:])
y_centers = 0.5 * (yedges[:-1] + yedges[1:])
for i in range(len(x_centers)):
    for j in range(len(y_centers)):
        count = counts_low[i, j]
        if count > 0:
            plt.text(x_centers[i], y_centers[j], str(int(count)),
                     ha='center', va='center', fontsize=13,
                     fontweight='bold', color='black')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nr+Nc (cm$^{-3}$)', fontsize=19, fontweight='bold')
plt.ylabel('LWC (g m$^{-3}$)', fontsize=19, fontweight='bold')
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.title('Mean RWC\nLow CDP GCCN Mass Flights\nJanuary–June 2022', fontsize=19, fontweight='bold')
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
plt.title('Mean LWC\nHigh CDP GCCN Mass Flights\nJanuary–June 2022', fontsize=19, fontweight='bold')
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
plt.title('Mean LWC\nLow GCCN CDP Mass Flights\nJanuary–June 2022', fontsize=19, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
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
plt.title("CDP RWC Ratio — High / Low GCCN Flights", fontsize=19, fontweight="bold")
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
cbar.set_label("CDP LWC Ratio (High / Low)", fontsize=18, fontweight="bold")
cbar.ax.tick_params(labelsize=16)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"Nr+Nc (cm$^{-3}$)", fontsize=19, fontweight="bold")
plt.ylabel(r"LWC (g m$^{-3}$)", fontsize=19, fontweight="bold")
plt.title("CDP Ratio — High / Low GCCN Flights", fontsize=19, fontweight="bold")
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
plt.title('CDP Low GCCN Flights — Mean RWC', fontsize=19, fontweight='bold')
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
plt.title('CDP Low GCCN Flights — Mean LWC', fontsize=19, fontweight='bold')
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
custom_bounds = [0, 0.2, 0.5, 1.0, 1.1, 1.2, 1.3, 1.4, 2.0, 2.2, 2.4, 2.7, 3.0, 3.5, 3.6, 7, 8.5]
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
plt.title("CDP (in cloud)\nRWC Ratio High / Low GCCN Mass Flights\nJanuary–June 2022", fontsize=19, fontweight="bold")
plt.tight_layout()
plt.show()
#%%
#masking bins with less than 100 samples
num_bins = 5
x_bins = np.logspace(np.log10(min(np.concatenate([high_concentration, low_concentration]))),
                     np.log10(max(np.concatenate([high_concentration, low_concentration]))),
                     num_bins)
y_bins = np.logspace(np.log10(min(np.concatenate([high_lwc, low_lwc]))),
                     np.log10(max(np.concatenate([high_lwc, low_lwc]))),
                     num_bins)
xedges, yedges = x_bins, y_bins
def make_empty_bins():
    return [[[] for _ in range(len(y_bins)-1)] for _ in range(len(x_bins)-1)]

rwc_bins_high = make_empty_bins()
rwc_bins_low  = make_empty_bins()
lwc_bins_high = make_empty_bins()
lwc_bins_low  = make_empty_bins()
for conc, lwc, rwc in zip(high_concentration, high_lwc, high_rwc):
    i = np.searchsorted(x_bins, conc, side='right') - 1
    j = np.searchsorted(y_bins, lwc,  side='right') - 1
    if 0 <= i < len(x_bins)-1 and 0 <= j < len(y_bins)-1 and not np.isnan(rwc) and not np.isnan(lwc):
        rwc_bins_high[i][j].append(rwc)
        lwc_bins_high[i][j].append(lwc)

for conc, lwc, rwc in zip(low_concentration, low_lwc, low_rwc):
    i = np.searchsorted(x_bins, conc, side='right') - 1
    j = np.searchsorted(y_bins, lwc,  side='right') - 1
    if 0 <= i < len(x_bins)-1 and 0 <= j < len(y_bins)-1 and not np.isnan(rwc) and not np.isnan(lwc):
        rwc_bins_low[i][j].append(rwc)
        lwc_bins_low[i][j].append(lwc)
avg_rwc_high = np.full((len(x_bins)-1, len(y_bins)-1), np.nan)
avg_rwc_low  = np.full((len(x_bins)-1, len(y_bins)-1), np.nan)
avg_lwc_high = np.full((len(x_bins)-1, len(y_bins)-1), np.nan)
avg_lwc_low  = np.full((len(x_bins)-1, len(y_bins)-1), np.nan)

for i in range(len(x_bins)-1):
    for j in range(len(y_bins)-1):
        if rwc_bins_high[i][j]:
            avg_rwc_high[i, j] = np.mean(rwc_bins_high[i][j])
        if rwc_bins_low[i][j]:
            avg_rwc_low[i, j]  = np.mean(rwc_bins_low[i][j])
        if lwc_bins_high[i][j]:
            avg_lwc_high[i, j] = np.mean(lwc_bins_high[i][j])
        if lwc_bins_low[i][j]:
            avg_lwc_low[i, j]  = np.mean(lwc_bins_low[i][j])
min_samples = 100
valid_bins = np.full((len(x_bins)-1, len(y_bins)-1), False)
for i in range(len(x_bins)-1):
    for j in range(len(y_bins)-1):
        total = len(rwc_bins_high[i][j]) + len(rwc_bins_low[i][j])
        if total >= min_samples:
            valid_bins[i, j] = True

avg_rwc_high[~valid_bins] = np.nan
avg_rwc_low[~valid_bins]  = np.nan
avg_lwc_high[~valid_bins] = np.nan
avg_lwc_low[~valid_bins]  = np.nan
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
plt.title("CDP RWC Ratio — High / Low GCCN Flights", fontsize=19, fontweight="bold")
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
cbar.set_label("CDP LWC Ratio (High / Low)", fontsize=18, fontweight="bold")
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
custom_bounds = [0, 0.2, 0.5, 1.0, 1.1, 1.2, 1.3, 1.4, 2.0, 2.2, 2.4, 2.7, 3.0, 3.5, 3.6, 7, 8.5]
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
plt.title("CDP (in cloud)\nRWC Ratio High / Low Mass GCCN Flights\nJanuary–June 2022", fontsize=19, fontweight="bold")
plt.tight_layout()
plt.show()

#%%
# Merge mass and liquid water lists into a unified dataset
combined_dataset = []

for conc_entry in total_combined_concentration:
    date = conc_entry['Date']
    time = conc_entry['Time']
    matching_lwc = next((e for e in total_liquid_water 
                         if e['Date'] == date and e['Time'] == time), None)

    if matching_lwc:
        combined_dataset.append({
            'Date': date,
            'Time': time,
            'Leg_start': conc_entry['Leg_start'],
            'Leg_stop': conc_entry['Leg_stop'],
            'Total_Combined_Concentration': conc_entry['Total_Combined_Concentration'],
            'Rain_Concentration': conc_entry['Rain_Concentration'],
            'Total_Liquid_Water': matching_lwc['Total_Liquid_Water']
        })
#%%
#trying histograms for ratio of high/low rwc 
x_min = np.nanmin([entry['Total_Combined_Concentration'] for entry in combined_dataset])
x_max = np.nanmax([entry['Total_Combined_Concentration'] for entry in combined_dataset])
y_min = np.nanmin([entry['Total_Liquid_Water'] for entry in combined_dataset])
y_max = np.nanmax([entry['Total_Liquid_Water'] for entry in combined_dataset])
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
min_samples = 100  # require at least this many raw points per bin before bootstrapping

def compute_flight_bin_means_RWC(flight_data):
    bin_means = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    bin_counts = np.zeros((len(x_bins) - 1, len(y_bins) - 1), dtype=int)

    for flight in flight_data.values():
        conc = np.array([e['Total_Combined_Concentration'] for e in flight])
        lwc = np.array([e['Total_Liquid_Water'] for e in flight])
        rwc = np.array([e['Rain_Concentration'] for e in flight])

        for i in range(len(x_bins) - 1):
            for j in range(len(y_bins) - 1):
                mask = (conc >= x_bins[i]) & (conc < x_bins[i + 1]) & \
                       (lwc >= y_bins[j]) & (lwc < y_bins[j + 1])
                if np.any(mask):
                    vals = rwc[mask]
                    vals = vals[~np.isnan(vals)]
                    if len(vals) > 0:
                        bin_means[i][j].extend(vals.tolist())
                        bin_counts[i, j] += len(vals)

    return bin_means, bin_counts


def bootstrap_ratio_distributions(bin_high, bin_low, counts_high, counts_low, min_samples=100):
    boot_ratios = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            if (counts_high[i, j] + counts_low[i, j]) >= min_samples:
                high_vals = bin_high[i][j]
                low_vals = bin_low[i][j]
                if len(high_vals) > 1 and len(low_vals) > 1:
                    boot_sample_ratios = []
                    for _ in range(n_bootstrap):
                        sampled_high = np.random.choice(high_vals, len(high_vals), replace=True)
                        sampled_low = np.random.choice(low_vals, len(low_vals), replace=True)
                        sampled_low = np.where(sampled_low == 0, np.nan, sampled_low)
                        mean_low = np.nanmean(sampled_low)
                        if mean_low > 0 and not np.isnan(mean_low):
                            ratio = np.nanmean(sampled_high) / mean_low
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
                clipped_ratio = np.clip(dist, 0, 10)
                bin_edges = np.arange(0, 10, 1)

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

                ax.set_xlim(0, 10)
                ax.tick_params(axis='both', labelsize=18)
            else:
                ax.set_visible(False)

   
    fig.suptitle("CDP (in cloud)\nMass Bootstrapped RWC Ratio (High/Low GCCN)\n January–June 2022", fontsize=20, fontweight='bold')
    fig.supxlabel("RWC Ratio (High / Low)", fontsize=20, fontweight='bold')
    fig.supylabel("Count", fontsize=20, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    plt.show()
gccn_values = np.array(list(average_mass_per_flight.values()))
threshold = np.percentile(gccn_values, 50)
high_dates = {date for date, val in average_mass_per_flight.items() if val >= threshold}
low_dates = {date for date, val in average_mass_per_flight.items() if val < threshold}
high_data = [entry for entry in combined_dataset if entry['Date'] in high_dates]
low_data = [entry for entry in combined_dataset if entry['Date'] in low_dates]
grouped_high = group_by_flight(high_data)
grouped_low = group_by_flight(low_data)
bin_means_high, counts_high = compute_flight_bin_means_RWC(grouped_high)
bin_means_low, counts_low   = compute_flight_bin_means_RWC(grouped_low)

boot_ratio_distributions = bootstrap_ratio_distributions(
    bin_means_high, bin_means_low, counts_high, counts_low, min_samples=100
)

plot_histograms_with_percentage_ratio(boot_ratio_distributions)
# %%
#fixing ranges 
conc_all = np.array([e['Total_Combined_Concentration'] for e in combined_dataset])
lwc_all  = np.array([e['Total_Liquid_Water'] for e in combined_dataset])
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
min_samples = 100  # threshold before bootstrapping

def group_by_flight(data):
    flights = defaultdict(list)
    for entry in data:
        flights[entry['Date']].append(entry)
    return flights

def compute_flight_bin_means_RWC(flight_data):
    bin_means = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    bin_counts = np.zeros((len(x_bins) - 1, len(y_bins) - 1), dtype=int)

    for flight in flight_data.values():
        conc = np.array([e['Total_Combined_Concentration'] for e in flight])
        lwc = np.array([e['Total_Liquid_Water'] for e in flight])
        rwc = np.array([e['Rain_Concentration'] for e in flight])

        for i in range(len(x_bins) - 1):
            for j in range(len(y_bins) - 1):
                mask = (conc >= x_bins[i]) & (conc < x_bins[i + 1]) & \
                       (lwc >= y_bins[j]) & (lwc < y_bins[j + 1])
                if np.any(mask):
                    vals = rwc[mask]
                    vals = vals[~np.isnan(vals)]
                    if len(vals) > 0:
                        bin_means[i][j].extend(vals.tolist())
                        bin_counts[i, j] += len(vals)

    return bin_means, bin_counts

def bootstrap_ratio_distributions(bin_high, bin_low, counts_high, counts_low, min_samples=100):
    boot_ratios = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            if (counts_high[i, j] + counts_low[i, j]) >= min_samples:
                high_vals = bin_high[i][j]
                low_vals = bin_low[i][j]
                if len(high_vals) > 1 and len(low_vals) > 1:
                    boot_sample_ratios = []
                    for _ in range(n_bootstrap):
                        sampled_high = np.random.choice(high_vals, len(high_vals), replace=True)
                        sampled_low = np.random.choice(low_vals, len(low_vals), replace=True)

                        mean_high = np.nanmean(sampled_high)
                        mean_low = np.nanmean(sampled_low)

                        if np.isfinite(mean_high) and np.isfinite(mean_low) and mean_low > 0:
                            ratio = mean_high / mean_low
                            boot_sample_ratios.append(ratio)
                        else:
                            boot_sample_ratios.append(np.nan)  # optional: could skip instead

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
gccn_values = np.array(list(average_mass_per_flight.values()))
threshold = np.percentile(gccn_values, 50)
high_dates = {date for date, val in average_mass_per_flight.items() if val >= threshold}
low_dates  = {date for date, val in average_mass_per_flight.items() if val < threshold}

high_data = [entry for entry in combined_dataset if entry['Date'] in high_dates]
low_data  = [entry for entry in combined_dataset if entry['Date'] in low_dates]

grouped_high = group_by_flight(high_data)
grouped_low  = group_by_flight(low_data)

bin_means_high, counts_high = compute_flight_bin_means_RWC(grouped_high)
bin_means_low,  counts_low  = compute_flight_bin_means_RWC(grouped_low)

boot_ratio_distributions = bootstrap_ratio_distributions(
    bin_means_high, bin_means_low, counts_high, counts_low, min_samples=100
)

plot_histograms_with_percentage_ratio(boot_ratio_distributions)

# %%
ratio_cap = 50.0
heatmap_data = np.full((len(x_bins) - 1, len(y_bins) - 1), np.nan)
min_rwc_threshold = 0.005          
for i in range(len(x_bins) - 1): 
    for j in range(len(y_bins) - 1): 
        dist = boot_ratio_distributions[j][i]
        if len(dist) > 0:
            dist = dist[np.isfinite(dist)]
            dist = dist[dist <= ratio_cap]
            if len(dist) > 0:
                heatmap_data[j][i] = np.nanmean(dist)  
masked_ratio_rwc = np.ma.masked_where(np.isnan(heatmap_data), heatmap_data)
custom_bounds = [0.1, 0.2, 0.4, 0.7, 1.0, 1.1, 1.4, 1.6, 1.8, 2.0, 2.5]
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
ax.set_title("CDP (in cloud)\nRWC Ratio High / Low GCCN Flights\nJanuary–June 2022",
             fontsize=19, fontweight="bold")
ax.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
ax.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
for i in range(len(y_bins) - 1):
    for j in range(len(x_bins) - 1):
        dist = boot_ratio_distributions[i][j]
        
        if (counts_high[j][i] + counts_low[j][i]) >= min_samples:
            
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

                ax.text(
                    x_center, y_center, label,
                    ha='center', va='center',
                    fontsize=9, fontweight='bold', linespacing=1.1
                )

plt.tight_layout()
plt.show()

# %%
heatmap_data = np.full((len(x_bins) - 1, len(y_bins) - 1), np.nan)
min_samples = 100
for i in range(len(x_bins) - 1): 
    for j in range(len(y_bins) - 1): 
        dist = boot_ratio_distributions[j][i]
        if len(dist) > 0:
            dist = dist[np.isfinite(dist)]
            dist = dist[dist <= ratio_cap]
            if len(dist) > 0:
                heatmap_data[j][i] = np.nanmean(dist)  

masked_ratio_rwc = np.ma.masked_where(np.isnan(heatmap_data), heatmap_data)
custom_bounds = [0.1, 0.2, 0.4, 0.7, 1.0, 1.1, 1.4, 1.6, 1.8, 2.0, 2.5]
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
ax.set_title("CDP (in cloud)\nRWC Ratio High / Low GCCN Flights\nJanuary–June 2022",
             fontsize=19, fontweight="bold")
ax.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
ax.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
for i in range(len(x_bins) - 1):
    for j in range(len(y_bins) - 1):
        dist = boot_ratio_distributions[j][i]
        if (counts_high[j][i] + counts_low[j][i]) >= min_samples:
            dist = dist[np.isfinite(dist)]
            dist = dist[dist <= ratio_cap]

            if len(dist) == 0:
                continue

            percent_above = np.sum(dist > 1) / len(dist) * 100
            mean_val = np.nanmean(dist)
            std_val = np.nanstd(dist)
            if mean_val < 0.75 or mean_val > 2.5:
                print(f"⚠️  Suspicious bin at Nr+Nc bin {i}, LWC bin {j} → μ = {mean_val:.2f}, σ = {std_val:.2f}, {percent_above:.1f}% > 1")

            label = f"{percent_above:.1f}% > 1\nμ={mean_val:.2f}, σ={std_val:.2f}"
            x_center = 10 ** ((np.log10(x_bins[i]) + np.log10(x_bins[i + 1])) / 2)
            y_center = 10 ** ((np.log10(y_bins[j]) + np.log10(y_bins[j + 1])) / 2)

            ax.text(
                x_center, y_center, label,
                ha='center', va='center',
                fontsize=9, fontweight='bold', linespacing=1.1
            )

plt.tight_layout()
plt.show()

# %%
#making sure we only use bins greater than 100 samples and adding uncertainty
heatmap_data = np.full((len(x_bins) - 1, len(y_bins) - 1), np.nan)
min_rwc_threshold = 0.005 
ratio_cap = 50    
min_samples = 100            
for i in range(len(x_bins) - 1): 
    for j in range(len(y_bins) - 1): 
        dist = boot_ratio_distributions[i][j]
        if len(dist) >= min_samples:
            dist = dist[np.isfinite(dist)]
            dist = dist[dist <= ratio_cap]
            if len(dist) > 0:
                heatmap_data[i][j] = np.nanmean(dist)  
masked_ratio_rwc = np.ma.masked_where(np.isnan(heatmap_data), heatmap_data)
custom_bounds = [0, 0.2, 0.5, 1.0, 1.1, 1.2, 1.3, 1.4, 2.0, 2.2, 2.4, 2.7, 3.0, 3.5, 3.6, 7, 9]
cmap = plt.cm.viridis.copy()
cmap.set_bad(color='gray')
norm = BoundaryNorm(boundaries=custom_bounds, ncolors=cmap.N)
fig, ax = plt.subplots(figsize=(8, 6))
img = ax.pcolormesh(
    x_bins, y_bins, masked_ratio_rwc.T,
    cmap=cmap, norm=norm, shading="auto"
)
cbar = plt.colorbar(img, ticks=custom_bounds)
cbar.set_label("Bootstrapped RWC Ratio (High / Low)", fontsize=19, fontweight="bold")
cbar.ax.tick_params(labelsize=19)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Nr+Nc (cm$^{-3}$)", fontsize=19, fontweight="bold")
ax.set_ylabel(r"LWC (g m$^{-3}$)", fontsize=19, fontweight="bold")
ax.set_title("CDP (in cloud)\nRWC Ratio High / Low GCCN Flights\nJanuary–June 2022",
             fontsize=19, fontweight="bold")
ax.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
ax.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
for i in range(len(y_bins) - 1):
    for j in range(len(x_bins) - 1):
        dist = boot_ratio_distributions[j][i]
        if len(dist) >= min_samples:
            dist = dist[np.isfinite(dist)]
            dist = dist[dist <= ratio_cap]
            if len(dist) == 0:
                continue
            percent_above = np.sum(dist > 1) / len(dist) * 100
            mean_val = np.nanmean(dist)
            std_val = np.nanstd(dist)
            sem_val = std_val / np.sqrt(len(dist))  # Standard error of the mean
            ci_lower = np.percentile(dist, lower_percentile)
            ci_upper = np.percentile(dist, upper_percentile)

            label = (f"{percent_above:.1f}% > 1\n"
                     f"μ={mean_val:.2f} ± {sem_val:.2f} (SEM)\n"
                     f"90% CI [{ci_lower:.2f}, {ci_upper:.2f}]")
            x_center = 10 ** ((np.log10(x_bins[j]) + np.log10(x_bins[j + 1])) / 2)
            y_center = 10 ** ((np.log10(y_bins[i]) + np.log10(y_bins[i + 1])) / 2)

            ax.text(
                x_center, y_center, label,
                ha='center', va='center',
                fontsize=7, fontweight='bold', linespacing=1.2
            )

plt.tight_layout()
plt.show()
# %%
heatmap_data = np.full((len(x_bins) - 1, len(y_bins) - 1), np.nan)
min_samples = 100
for i in range(len(x_bins) - 1): 
    for j in range(len(y_bins) - 1): 
        dist = boot_ratio_distributions[i][j]
        if len(dist) >= min_samples:
            dist = dist[np.isfinite(dist)]
            dist = dist[dist <= ratio_cap]
            if len(dist) > 0:
                heatmap_data[i][j] = np.nanmean(dist)

masked_ratio_rwc = np.ma.masked_where(np.isnan(heatmap_data), heatmap_data)
custom_bounds = [0, 0.2, 0.5, 1.0, 1.1, 1.2, 1.3, 1.4, 2.0, 2.2, 2.4, 2.7, 3.0, 3.5, 3.6, 7, 9]
cmap = plt.cm.viridis.copy()
cmap.set_bad(color='gray')
norm = BoundaryNorm(boundaries=custom_bounds, ncolors=cmap.N)
fig, ax = plt.subplots(figsize=(8, 6))
img = ax.pcolormesh(
    x_bins, y_bins, masked_ratio_rwc.T,
    cmap=cmap, norm=norm, shading="auto"
)
cbar = plt.colorbar(img, ticks=custom_bounds)
cbar.set_label("Bootstrapped RWC Ratio (High / Low)", fontsize=19, fontweight="bold")
cbar.ax.tick_params(labelsize=19)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Nr+Nc (cm$^{-3}$)", fontsize=19, fontweight="bold")
ax.set_ylabel(r"LWC (g m$^{-3}$)", fontsize=19, fontweight="bold")
ax.set_title("CDP (in cloud)\nRWC Ratio High / Low GCCN Flights\nJanuary–June 2022",
             fontsize=19, fontweight="bold")
ax.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
ax.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
for i in range(len(x_bins) - 1):
    for j in range(len(y_bins) - 1):
        dist = boot_ratio_distributions[i][j]
        if len(dist) >= min_samples:
            dist = dist[np.isfinite(dist)]
            dist = dist[dist <= ratio_cap]
            if len(dist) == 0:
                continue

            percent_above = np.sum(dist > 1) / len(dist) * 100
            mean_val = np.nanmean(dist)
            std_val = np.nanstd(dist)
            sem_val = std_val / np.sqrt(len(dist))
            ci_lower = np.percentile(dist, lower_percentile)
            ci_upper = np.percentile(dist, upper_percentile)

            label = (f"{percent_above:.1f}% > 1\n"
                     f"μ={mean_val:.2f} ± {sem_val:.2f} (SEM)\n"
                     f"90% CI [{ci_lower:.2f}, {ci_upper:.2f}]")

            x_center = 10 ** ((np.log10(x_bins[i]) + np.log10(x_bins[i + 1])) / 2)
            y_center = 10 ** ((np.log10(y_bins[j]) + np.log10(y_bins[j + 1])) / 2)

            ax.text(
                x_center, y_center, label,
                ha='center', va='center',
                fontsize=7, fontweight='bold', linespacing=1.2
            )

plt.tight_layout()
plt.show()
#%%
n_bootstrap = 10000
confidence_level = 0.90
lower_percentile = (1 - confidence_level) / 2 * 100
upper_percentile = (1 + confidence_level) / 2 * 100
valid_bins = np.full((len(x_bins)-1, len(y_bins)-1), False)
for i in range(len(x_bins)-1):
    for j in range(len(y_bins)-1):
        total = len(rwc_bins_high[i][j]) + len(rwc_bins_low[i][j])
        if total >= 100:
            valid_bins[i, j] = True
boot_ratio_distributions = [[[] for _ in range(len(y_bins) - 1)] for _ in range(len(x_bins) - 1)]
heatmap_data = np.full((len(x_bins) - 1, len(y_bins) - 1), np.nan)

for i in range(len(x_bins) - 1):
    for j in range(len(y_bins) - 1):
        if valid_bins[i, j]:
            high_vals = np.array(rwc_bins_high[i][j])
            low_vals  = np.array(rwc_bins_low[i][j])
            
            if len(high_vals) > 0 and len(low_vals) > 0:
                ratios = []
                for _ in range(n_bootstrap):
                    sample_high = np.random.choice(high_vals, size=len(high_vals), replace=True)
                    sample_low  = np.random.choice(low_vals,  size=len(low_vals),  replace=True)

                    mean_high = np.mean(sample_high)
                    mean_low  = np.mean(sample_low)

                    if mean_low > 0:
                        ratios.append(mean_high / mean_low)

                boot_ratio_distributions[i][j] = ratios
                heatmap_data[i][j] = np.nanmean(ratios)

masked_ratio_rwc = np.ma.masked_where(np.isnan(heatmap_data), heatmap_data)
custom_bounds = [0, 0.2, 0.5, 1.0, 1.1, 1.2, 1.3, 1.4, 2.0, 2.2, 2.4, 2.7, 3.0, 3.5, 3.6, 7, 9]
cmap = plt.cm.viridis.copy()
cmap.set_bad(color='gray')
norm = BoundaryNorm(boundaries=custom_bounds, ncolors=cmap.N)

fig, ax = plt.subplots(figsize=(8, 6))
img = ax.pcolormesh(
    x_bins, y_bins, masked_ratio_rwc.T,
    cmap=cmap, norm=norm, shading="auto"
)

cbar = plt.colorbar(img, ticks=custom_bounds)
cbar.set_label("Bootstrapped RWC Ratio (High / Low)", fontsize=19, fontweight="bold")
cbar.ax.tick_params(labelsize=19)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Nr+Nc (cm$^{-3}$)", fontsize=19, fontweight="bold")
ax.set_ylabel(r"LWC (g m$^{-3}$)", fontsize=19, fontweight="bold")
ax.set_title("Combined CAS and CDP (in cloud)\nRWC Ratio High / Low Mass \nJanuary–June 2022",
             fontsize=19, fontweight="bold")
ax.tick_params(axis='both', which='major', labelsize=19, width=3, length=8)
ax.tick_params(axis='both', which='minor', labelsize=19, width=2, length=5)
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
for i in range(len(x_bins) - 1):
    for j in range(len(y_bins) - 1):
        dist = boot_ratio_distributions[i][j]
        if valid_bins[i][j] and len(dist) > 0:
            percent_above = np.sum(np.array(dist) > 1) / len(dist) * 100
            mean_val = np.nanmean(dist)
            std_val = np.nanstd(dist)
            sem_val = std_val / np.sqrt(len(dist))
            ci_lower = np.percentile(dist, lower_percentile)
            ci_upper = np.percentile(dist, upper_percentile)

            label = (f"{percent_above:.1f}% > 1\n"
                     f"μ={mean_val:.2f} ± {sem_val:.2f} (SEM)\n"
                     f"90% CI [{ci_lower:.2f}, {ci_upper:.2f}]")

            x_center = 10 ** ((np.log10(x_bins[i]) + np.log10(x_bins[i + 1])) / 2)
            y_center = 10 ** ((np.log10(y_bins[j]) + np.log10(y_bins[j + 1])) / 2)

            ax.text(
                x_center, y_center, label,
                ha='center', va='center',
                fontsize=7, fontweight='bold', linespacing=1.2
            )

plt.tight_layout()
plt.show()
# %%
