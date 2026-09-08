
#%%
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pathlib
import statistics
import glob
import pickle
from collections import defaultdict
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
# %%
# Add RWC and CWC for total LWC
cwc_lookup = {
    (entry['Date'], entry['Time']): entry
    for entry in in_cloud_concentrations}
total_liquid_water = []
for rwc_entry in rain_water_content:
    matching_time = rwc_entry['Time']
    matching_date = rwc_entry['Date']
    matching_cwc = cwc_lookup.get(
        (matching_date, matching_time)    )
    if matching_cwc is not None:
        cwc_val = matching_cwc['CWC']
        rwc_val = rwc_entry['RWC']
        total_liquid = (
            cwc_val +
            rwc_val        )
        total_liquid_water.append({
            'Date': matching_date,
            'Time': matching_time,
            'Leg_start': rwc_entry['Leg_start'],
            'Leg_stop': rwc_entry['Leg_stop'],
            'CWC': cwc_val,
            'RWC': rwc_val,
            'Total_Liquid_Water': total_liquid        })
print(
    "Number of total liquid water entries:",
    len(total_liquid_water))
print(
    "First 5 entries:",
    total_liquid_water[:5])
#%%
# %%
# Add Nc + Nr for total concentration
rain_lookup = {
    (entry['Date'], entry['Time']): entry
    for entry in rain_concentrations}
total_combined_concentration = []
for in_cloud_entry in in_cloud_concentrations:
    matching_time = in_cloud_entry['Time']
    matching_date = in_cloud_entry['Date']
    matching_rain = rain_lookup.get(
        (matching_date, matching_time)    )
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
            'Total_Combined_Concentration': combined_conc })
print("Number of total combined concentration entries:",
    len(total_combined_concentration))
print("First 5 entries:", total_combined_concentration[:5])
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
plt.title('CDP and CAS combined (in cloud)\n January-June 2022\n RWC as a function of number concentration', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
#change to a log scale color bar
# trying to separate LWC and RWC

masked_avg_rwc = np.ma.masked_where(
    (~np.isfinite(avg_rwc))
    | (avg_rwc <= 0),
    avg_rwc
)

positive_rwc = avg_rwc[
    np.isfinite(avg_rwc)
    & (avg_rwc > 0)
]

rwc_min = np.min(positive_rwc)

print(
    "Minimum positive mean RWC:",
    rwc_min
)

plt.figure(figsize=(8, 6))

norm = mcolors.LogNorm(
    vmin=rwc_min,
    vmax=0.3
)

img = plt.pcolormesh(
    xedges,
    yedges,
    masked_avg_rwc.T,
    cmap="viridis",
    norm=norm,
    shading="auto"
)

gray_mask = (
    np.isnan(avg_rwc)
    | (avg_rwc <= 0)
)

gray_values = np.full_like(
    avg_rwc,
    np.nan
)

gray_values[gray_mask] = 1

plt.pcolormesh(
    xedges,
    yedges,
    gray_values.T,
    cmap=mcolors.ListedColormap(["gray"]),
    shading="auto",
    alpha=0.6
)

cbar = plt.colorbar(img)

cbar.set_label(
    "Mean RWC (g m$^{-3}$)",
    fontsize=18,
    fontweight="bold"
)

cbar.ax.tick_params(
    labelsize=18,
    width=2,
    length=5
)

for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")

plt.xscale("log")
plt.yscale("log")

plt.tick_params(
    axis="both",
    which="major",
    labelsize=19,
    width=3,
    length=8
)

plt.tick_params(
    axis="both",
    which="minor",
    labelsize=19,
    width=2,
    length=5
)

plt.xlabel(
    r'Nr+Nc (cm$^{-3}$)',
    fontsize=19,
    fontweight="bold"
)

plt.ylabel(
    r'LWC (g m$^{-3}$)',
    fontsize=19,
    fontweight="bold"
)

plt.title(
    "CDP and CAS combined (in cloud)\n"
    "Mean RWC\n"
    "January–June 2022",
    fontsize=18,
    fontweight="bold"
)

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
    vmin=0, vmax=0.3
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
plt.title('CDP and CAS combined (in cloud)\nMean RWC\nJanuary–June 2022', fontsize=18, fontweight='bold')
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
#change to logscale 
from matplotlib.colors import LogNorm

masked_avg_lwc = np.ma.masked_where(
    np.isnan(avg_lwc),
    avg_lwc
)

plt.figure(figsize=(8, 6))

norm = LogNorm(
    vmin=0.01,
    vmax=1
)

img = plt.pcolormesh(
    xedges,
    yedges,
    masked_avg_lwc.T,
    cmap="plasma",
    norm=norm,
    shading="auto"
)

gray_mask = np.isnan(avg_lwc)

gray_values = np.full_like(
    avg_lwc,
    np.nan
)

gray_values[gray_mask] = 1

plt.pcolormesh(
    xedges,
    yedges,
    gray_values.T,
    cmap=mcolors.ListedColormap(["gray"]),
    shading="auto",
    alpha=0.6
)

cbar = plt.colorbar(
    img,
    ticks=[0.01, 0.1, 1]
)

cbar.set_label(
    "Mean LWC (g m$^{-3}$)",
    fontsize=18,
    fontweight="bold"
)

cbar.ax.set_yticklabels([
    "0.01",
    "0.1",
    "1"
])

cbar.ax.tick_params(
    labelsize=18,
    width=2,
    length=5
)

for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")

plt.xscale("log")
plt.yscale("log")

plt.xlabel(
    r'Nr+Nc (cm$^{-3}$)',
    fontsize=19,
    fontweight="bold"
)

plt.ylabel(
    r'LWC (g m$^{-3}$)',
    fontsize=19,
    fontweight="bold"
)

plt.title(
    "Mean LWC\nJanuary–June 2022 (CDP in cloud)",
    fontsize=18,
    fontweight="bold"
)

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
# %%
# Import combined CAS + CDP flight-mean GCCN conc
gccn_file = (
    "/home/disk/p/kathem24/activate/ACTIVATE-2024-2025/"
    "CAS/Below Cloud Base/Scripts/"
    "CAS_CDP_GCCN_concentration_flight_mean_2022.pkl"
)
with open(gccn_file, "rb") as f:
    average_gccn_per_flight = pickle.load(f)
print(
    "Loaded combined CAS + CDP "
    "flight-mean GCCN concentration."
)

print(
    "Number of flights:",
    len(average_gccn_per_flight)
)
#%%
# Match Nr+Nc, LWC, and RWC by Date + Time FIRST
liquid_water_lookup = {
    (entry['Date'], entry['Time']): entry
    for entry in total_liquid_water}
matched_rwc_data = []
for conc_entry in total_combined_concentration:
    key = (
        conc_entry['Date'],
        conc_entry['Time']    )
    liquid_entry = liquid_water_lookup.get(key)
    if liquid_entry is not None:
        matched_rwc_data.append({
            'Date': conc_entry['Date'],
            'Time': conc_entry['Time'],
            'Total_Combined_Concentration':
                conc_entry['Total_Combined_Concentration'],
            'Total_Liquid_Water':
                liquid_entry['Total_Liquid_Water'],
            'RWC':
                liquid_entry['RWC']})
print(
    "Matched concentration/LWC/RWC points:",
    len(matched_rwc_data))
#%%# %%
box_x_min = 25.182
box_x_max = 634.143
box_y_min = 0.041
box_y_max = 0.580
dense_region_data = [
    entry for entry in matched_rwc_data
    if (
        box_x_min
        <= entry["Total_Combined_Concentration"]
        < box_x_max
        and
        box_y_min
        <= entry["Total_Liquid_Water"]
        < box_y_max    )]
print(
    "Total matched observations:",
    len(matched_rwc_data))
print("Observations inside dense region:",
    len(dense_region_data))
# %%
from collections import defaultdict
dense_region_by_flight = defaultdict(list)
for entry in dense_region_data:
    dense_region_by_flight[
        entry["Date"]
    ].append(entry)
print(
    "Number of flights represented:",
    len(dense_region_by_flight))
print("\nObservations per flight:")
for date in sorted(dense_region_by_flight):
    print(
        date,
        len(dense_region_by_flight[date])    )
# %%
minimum_points_per_flight = 0
RWC_LWC_ratio_per_flight = {}
for date, entries in dense_region_by_flight.items():
    rwc_values = np.array([
        entry["RWC"]
        for entry in entries
    ], dtype=float)
    lwc_values = np.array([
        entry["Total_Liquid_Water"]
        for entry in entries
    ], dtype=float)
    valid = (
        np.isfinite(rwc_values)
        & np.isfinite(lwc_values)
        & (lwc_values > 0)    )
    rwc_values = rwc_values[valid]
    lwc_values = lwc_values[valid]
    n_points = len(rwc_values)
    if n_points < minimum_points_per_flight:
        continue
    flight_rwc_lwc_ratio = (
        np.mean(rwc_values)
        / np.mean(lwc_values)
    ) * 100
    RWC_LWC_ratio_per_flight[date] = {
        "RWC_LWC": flight_rwc_lwc_ratio,
        "n_points": n_points,
        "mean_RWC": np.mean(rwc_values),
        "mean_LWC": np.mean(lwc_values)    }
print("Flights retained:",
    len(RWC_LWC_ratio_per_flight))
print("\nFlight-average RWC/LWC:")
for date in sorted(RWC_LWC_ratio_per_flight):
    result = RWC_LWC_ratio_per_flight[date]
    print( date,
        f"RWC/LWC = {result['RWC_LWC']:.2f}%",
        f"| n = {result['n_points']}"    )
# %%
common_N_dates = sorted(
    set(average_gccn_per_flight.keys())
    & set(RWC_LWC_ratio_per_flight.keys()))
print("Flights with both number concentration and RWC/LWC:",
    len(common_N_dates))
print("\nMatched flights:")
for date in common_N_dates:
    print(date,
        f"Number concentration = {average_gccn_per_flight[date]:.2f} /cm³",
        f"| RWC/LWC = "
        f"{RWC_LWC_ratio_per_flight[date]['RWC_LWC']:.2f}%")
# %%
# Create matched flight-level arrays
flight_gccn = np.array([
    average_gccn_per_flight[date]
    for date in common_N_dates
], dtype=float)
flight_rwc_lwc = np.array([
    RWC_LWC_ratio_per_flight[date]["RWC_LWC"]
    for date in common_N_dates
], dtype=float)
flight_n_points = np.array([
    RWC_LWC_ratio_per_flight[date]["n_points"]
    for date in common_N_dates
], dtype=int)
valid = (
    np.isfinite(flight_gccn)
    & np.isfinite(flight_rwc_lwc)
    & (flight_gccn > 0))
flight_gccn = flight_gccn[valid]
flight_rwc_lwc = flight_rwc_lwc[valid]
flight_n_points = flight_n_points[valid]
valid_dates = np.array(common_N_dates)[valid]
print("Final flights used for number concentration correlation:",
    len(flight_gccn))
print("Number concentration range:",
    np.min(flight_gccn),
    "to",
    np.max(flight_gccn),
    "/cm³")
print("RWC/LWC range:",
    np.min(flight_rwc_lwc),
    "to",
    np.max(flight_rwc_lwc),
    "%")
#%%
from scipy.stats import linregress
regression = linregress(
    flight_gccn,
    flight_rwc_lwc)
slope = regression.slope
intercept = regression.intercept
r_value = regression.rvalue
r_squared = r_value ** 2
p_value = regression.pvalue
print(f"Slope: {slope:.3f}")
print(f"Intercept: {intercept:.3f}")
print(f"r: {r_value:.3f}")
print(
    f"R²: {r_squared:.3f}")
print(f"p-value: {p_value:.4f}")
# %%
#scatterplot of flight-average GCCN mass vs. flight-average RWC/LWC
plt.figure(figsize=(8, 6))
plt.scatter(
    flight_gccn,
    flight_rwc_lwc,
    s=80,
    edgecolor="black",
    alpha=0.8)
x_vals = np.linspace(
    np.min(flight_gccn),
    np.max(flight_gccn),
    200)
y_vals = (
    intercept
    + slope * x_vals)
plt.plot(
    x_vals,
    y_vals,
    color="black",
    linewidth=2,
    label=(
        f"R² = {r_squared:.3f}, "
        f"Slope: {slope:.3f}, "
        f"r: {r_value:.3f}"))
plt.xlabel("Flight-average Number Concentration (cm⁻³)",
    fontsize=17,
    fontweight="bold")
plt.ylabel("Flight-average RWC/LWC (%)",
    fontsize=17,
    fontweight="bold")
plt.title("Combined CAS and CDP\n"
    "January–June 2022",
    fontsize=18,
    fontweight="bold")
plt.tick_params(
    axis="both",
    which="major",
    labelsize=18,
    width=2,
    length=6)
plt.tick_params(
    axis="both",
    which="minor",
    labelsize=18,
    width=1,
    length=4)
plt.legend(
    fontsize=15,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5))
plt.show()
# %%
#plotting LWC per flight per instrument
#do not take flight average
#plot every 1 Hz second 
#%%
#CAS
CDP_LWC_df = pd.DataFrame(in_cloud_concentrations_CDP)
CAS_LWC_df = pd.DataFrame(in_cloud_concentrations_CAS)
CDP_LWC_df["CWC"] = pd.to_numeric(
    CDP_LWC_df["CWC"],
    errors="coerce"
)

CAS_LWC_df["CWC"] = pd.to_numeric(
    CAS_LWC_df["CWC"],
    errors="coerce"
)
CDP_LWC_df = CDP_LWC_df[
    np.isfinite(CDP_LWC_df["CWC"]) &
    (CDP_LWC_df["CWC"] >= 0.01)
].copy()

CAS_LWC_df = CAS_LWC_df[
    np.isfinite(CAS_LWC_df["CWC"]) &
    (CAS_LWC_df["CWC"] >= 0.01)
].copy()
print("CDP 1-Hz LWC points:", len(CDP_LWC_df))
print("CAS 1-Hz LWC points:", len(CAS_LWC_df))

print(
    "CDP flights:",
    CDP_LWC_df["Date"].nunique()
)

print(
    "CAS flights:",
    CAS_LWC_df["Date"].nunique()
)
# %%
#%%
def calculate_leg_lwc_stats(
    df,
    instrument_name):
    leg_stats = []
    grouped = df.groupby(
        [
            "Date",
            "Leg_start",
            "Leg_stop"        ]    )
    for (
        date,
        leg_start,
        leg_stop
    ), group in grouped:
        lwc = group["CWC"].values.astype(float)
        lwc = lwc[
            np.isfinite(lwc)        ]
        if len(lwc) == 0:
            continue
        mean_lwc = np.mean(lwc)
        median_lwc = np.median(lwc)
        if len(lwc) >= 3:
            skewness = skew(
                lwc,
                bias=False            )
        else:
            skewness = np.nan

        if median_lwc > 0:
            mean_median_ratio = (
                mean_lwc /
                median_lwc            )
        else:
            mean_median_ratio = np.nan

        leg_stats.append({
            "Instrument": instrument_name,
            "Date": date,
            "Leg_start": leg_start,
            "Leg_stop": leg_stop,
            "N_1Hz": len(lwc),
            "Mean_LWC": mean_lwc,
            "Median_LWC": median_lwc,
            "Std_LWC": np.std(
                lwc,
                ddof=1
            ) if len(lwc) > 1 else np.nan,
            "Skewness": skewness,
            "P95_LWC": np.percentile(
                lwc,
                95
            ),
            "P99_LWC": np.percentile(
                lwc,
                99
            ),
            "Max_LWC": np.max(lwc),
            "Mean_Median_Ratio":
                mean_median_ratio        })
    return pd.DataFrame(leg_stats)
CDP_leg_stats = calculate_leg_lwc_stats(
    CDP_LWC_df,
    "CDP")
CAS_leg_stats = calculate_leg_lwc_stats(
    CAS_LWC_df,
    "CAS")
print("\nCDP leg statistics:")
print(CDP_leg_stats)
print("\nCAS leg statistics:")
print(CAS_leg_stats)
#%%
def plot_leg_lwc_pdfs(
    df,
    instrument_name,
    number_bins=30):
    unique_dates = sorted(
        df["Date"].unique()    )
    for date in unique_dates:
        flight_data = df[
            df["Date"] == date
        ].copy()
        grouped_legs = list(
            flight_data.groupby(
                [
                    "Leg_start",
                    "Leg_stop"
                ]            )        )
        number_legs = len(grouped_legs)
        if number_legs == 0:
            continue
        number_columns = 3
        number_rows = int(
            np.ceil(
                number_legs /
                number_columns            )        )
        fig, axes = plt.subplots(
            number_rows,
            number_columns,
            figsize=(
                15,
                4 * number_rows            )        )
        axes = np.atleast_1d(
            axes
        ).flatten()
        all_lwc = flight_data[
            "CWC"
        ].values.astype(float)
        all_lwc = all_lwc[
            np.isfinite(all_lwc)        ]
        min_lwc = 0.01
        max_lwc = np.max(
            all_lwc        )
        bins = np.linspace(
            min_lwc,
            max_lwc,
            number_bins + 1        )
        for leg_number, (
            (
                leg_start,
                leg_stop
            ),
            leg_data
        ) in enumerate(grouped_legs):
            ax = axes[
                leg_number            ]
            lwc = leg_data[
                "CWC"
            ].values.astype(float)
            lwc = lwc[
                np.isfinite(lwc)            ]
            ax.hist(
                lwc,
                bins=bins,
                density=True,
                alpha=0.7,
                edgecolor="black"            )
            mean_lwc = np.mean(
                lwc            )
            ax.axvline(
                mean_lwc,
                linestyle="--",
                linewidth=2,
                label=(
                    f"Mean = "
                    f"{mean_lwc:.3f}"                )            )
            median_lwc = np.median(
                lwc            )
            ax.axvline(
                median_lwc,
                linestyle=":",
                linewidth=2,
                label=(
                    f"Median = "
                    f"{median_lwc:.3f}"                )            )
            ax.set_title(
                f"Leg {leg_number + 1}\n"
                f"{leg_start:.0f}–"
                f"{leg_stop:.0f} s\n"
                f"n = {len(lwc)}",
                fontsize=11,
                fontweight="bold"            )
            ax.set_xlabel(
                "LWC (g m$^{-3}$)",
                fontsize=11,
                fontweight="bold"            )
            ax.set_ylabel(
                "Probability Density",
                fontsize=11,
                fontweight="bold"            )
            ax.tick_params(
                labelsize=10,
                width=1.5            )
            ax.legend(
                fontsize=9            )
        for empty_axis in range(
            number_legs,
            len(axes)
        ):
            fig.delaxes(
                axes[empty_axis]            )
        fig.suptitle(
            f"{instrument_name} "
            f"1-Hz LWC PDFs — {date}",
            fontsize=18,
            fontweight="bold",
            y=1.02        )
        plt.tight_layout()
        plt.show()
# %%
#%%
CAS_dates = set(CAS_LWC_df["Date"].unique())
CDP_dates = set(CDP_LWC_df["Date"].unique())
extra_CAS_dates = sorted(
    CAS_dates - CDP_dates)
extra_CDP_dates = sorted(
    CDP_dates - CAS_dates)
print("CAS-only flights:", extra_CAS_dates)
print("CDP-only flights:", extra_CDP_dates)
print("CAS flights:", len(CAS_dates))
print("CDP flights:", len(CDP_dates))
#%%
common_dates = sorted(
    CAS_dates & CDP_dates)
print("Common flights:", len(common_dates))
CAS_LWC_common = CAS_LWC_df[
    CAS_LWC_df["Date"].isin(common_dates)].copy()
CDP_LWC_common = CDP_LWC_df[
    CDP_LWC_df["Date"].isin(common_dates)].copy()
print(
    "CAS common flights:",
    CAS_LWC_common["Date"].nunique())
print(
    "CDP common flights:",
    CDP_LWC_common["Date"].nunique())
#%%
def plot_flight_lwc_pdfs(
    df,
    instrument_name,
    number_bins=30):
    unique_dates = sorted(
        df["Date"].unique()    )
    number_flights = len(unique_dates)
    print(
        f"{instrument_name} flights:",
        number_flights    )
    number_columns = 5
    number_rows = int(
        np.ceil(
            number_flights /
            number_columns        )    )
    fig, axes = plt.subplots(
        number_rows,
        number_columns,
        figsize=(
            16,
            3.5 * number_rows        )    )
    axes = np.atleast_1d(
        axes
    ).flatten()
    all_lwc = df[
        "CWC"
    ].values.astype(float)
    all_lwc = all_lwc[
        np.isfinite(all_lwc)    ]
    min_lwc = 0.01
    max_lwc = np.max(all_lwc)
    bins = np.linspace(
        min_lwc,
        max_lwc,
        number_bins + 1    )
    for flight_number, date in enumerate(
        unique_dates    ):
        ax = axes[
            flight_number        ]
        flight_data = df[
            df["Date"] == date        ]
        lwc = flight_data[
            "CWC"
        ].values.astype(float)
        lwc = lwc[
            np.isfinite(lwc)        ]
        ax.hist(
            lwc,
            bins=bins,
            density=False,
            alpha=0.7,
            edgecolor="black"        )
        ax.set_xlim(
            0.01,
            1        )
        ax.set_title(
            f"{date}\n"
            f"n = {len(lwc)} seconds",
            fontsize=11,
            fontweight="bold"        )
        ax.set_xlabel(
            "LWC (g m$^{-3}$)",
            fontsize=11,
            fontweight="bold"        )
        ax.set_ylabel(
            "Number of 1 Hz Obs",
            fontsize=11,
            fontweight="bold"        )
        ax.tick_params(
            labelsize=9,
            width=1.5        )
    for empty_axis in range(
        number_flights,
        len(axes)    ):
        fig.delaxes(
            axes[empty_axis]        )
    fig.suptitle(
        f"{instrument_name} 1Hz LWC by Flight",
        fontsize=20,
        fontweight="bold",
        y=1.01    )
    plt.tight_layout()
    plt.show()
#%%
plot_flight_lwc_pdfs(
    CAS_LWC_common,
    "CAS", number_bins=30)
#%%
plot_flight_lwc_pdfs(
    CDP_LWC_common,
    "CDP", number_bins=30)
#%%
CAS_CDP_flight_LWC_stats = pd.merge(
    CAS_flight_LWC_stats,
    CDP_flight_LWC_stats,
    on="Date",
    how="inner"
)

print(
    "Number of matched flights:",
    len(CAS_CDP_flight_LWC_stats)
)

print(
    CAS_CDP_flight_LWC_stats
)
# %%
#comparing the two instruments 
def plot_cas_cdp_flight_lwc_overlay(
    CAS_df,
    CDP_df,
    number_bins=30,
    x_max=1.0):
    common_dates = sorted(
        set(CAS_df["Date"].unique()) &
        set(CDP_df["Date"].unique())    )
    number_flights = len(common_dates)
    print("Number of common flights:", number_flights)
    number_columns = 4
    number_rows = int(
        np.ceil(number_flights / number_columns)    )

    fig, axes = plt.subplots(
        number_rows,
        number_columns,
        figsize=(18, 3.8 * number_rows)    )
    axes = np.atleast_1d(axes).flatten()
    min_lwc = 0.01
    bins = np.linspace(
        min_lwc,
        x_max,
        number_bins + 1    )
    for flight_number, date in enumerate(common_dates):
        ax = axes[flight_number]
        CAS_flight = CAS_df[
            CAS_df["Date"] == date
        ].copy()
        CDP_flight = CDP_df[
            CDP_df["Date"] == date
        ].copy()
        cas_lwc = CAS_flight["CWC"].values.astype(float)
        cdp_lwc = CDP_flight["CWC"].values.astype(float)
        cas_lwc = cas_lwc[np.isfinite(cas_lwc)]
        cdp_lwc = cdp_lwc[np.isfinite(cdp_lwc)]
        ax.hist(
            cas_lwc,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            label=f"CAS (n={len(cas_lwc)})"        )
        ax.hist(
            cdp_lwc,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            label=f"CDP (n={len(cdp_lwc)})"        )
        ax.set_xlim(0.01, x_max)
        ax.set_title(
            f"{date}",
            fontsize=11,
            fontweight="bold"        )
        ax.set_xlabel(
            "LWC (g m$^{-3}$)",
            fontsize=10,
            fontweight="bold"        )
        ax.set_ylabel(
            "Probability Density",
            fontsize=10,
            fontweight="bold"        )
        ax.tick_params(
            labelsize=9,
            width=1.2        )
        ax.legend(
            fontsize=8        )
    for empty_axis in range(number_flights, len(axes)):
        fig.delaxes(axes[empty_axis])
    plt.tight_layout()
    plt.show()
#%%
plot_cas_cdp_flight_lwc_overlay(
    CAS_LWC_common,
    CDP_LWC_common,
    number_bins=30,
    x_max=1.0)
# %%
def calculate_flight_lwc_stats(
    df,
    instrument_name
):

    flight_stats = []

    for date, flight_data in df.groupby("Date"):

        lwc = flight_data[
            "CWC"
        ].values.astype(float)

        lwc = lwc[
            np.isfinite(lwc)
        ]

        if len(lwc) == 0:
            continue

        flight_stats.append({

            "Date": date,

            f"{instrument_name}_n":
                len(lwc),

            f"{instrument_name}_mean (g m$^{-3}$)":
                np.mean(lwc),

            f"{instrument_name}_median (g m$^{-3}$)":
                np.median(lwc),

            f"{instrument_name}_Q25":
                np.percentile(
                    lwc,
                    25
                ),

            f"{instrument_name}_Q75":
                np.percentile(
                    lwc,
                    75
                )
        })

    return pd.DataFrame(
        flight_stats
    )
#%%
#%%
CAS_flight_LWC_stats = (
    calculate_flight_lwc_stats(
        CAS_LWC_common,
        "CAS"
    )
)

CDP_flight_LWC_stats = (
    calculate_flight_lwc_stats(
        CDP_LWC_common,
        "CDP"
    )
)
#%%
#%%
print(
    CAS_CDP_flight_LWC_stats[
        [
            "Date",
            "CAS_mean (g m$^{-3}$)",
            "CDP_mean (g m$^{-3}$)",
            "CAS_median (g m$^{-3}$)",
            "CDP_median (g m$^{-3}$)",
            "CAS_n",
            "CDP_n"
        ]
    ]
)
#%%
table_data = CAS_CDP_flight_LWC_stats[
    [        "Date",
        "CAS_mean (g m$^{-3}$)",
        "CDP_mean (g m$^{-3}$)",
        "CAS_median (g m$^{-3}$)",
        "CDP_median (g m$^{-3}$)",
        "CAS_n",
        "CDP_n"    ]].copy()
table_data.columns = [
    "Date",
    "CAS Mean (g m$^{-3}$)",
    "CDP Mean (g m$^{-3}$)",
    "CAS Median (g m$^{-3}$)",
    "CDP Median (g m$^{-3}$)",
    "CAS n",
    "CDP n"]
columns_to_round = [
    "CAS Mean (g m$^{-3}$)",
    "CDP Mean (g m$^{-3}$)",
    "CAS Median (g m$^{-3}$)",
    "CDP Median (g m$^{-3}$)",]
table_data[columns_to_round] = (
    table_data[columns_to_round]
    .round(2))
table_data["CAS n"] = (
    table_data["CAS n"].astype(int))
table_data["CDP n"] = (
    table_data["CDP n"].astype(int))
for column in columns_to_round:
    table_data[column] = table_data[column].map(
        lambda x: f"{x:.2f}"    )
fig, ax = plt.subplots(
    figsize=(15, 18))
ax.axis("off")
table = ax.table(
    cellText=table_data.values,
    colLabels=table_data.columns,
    cellLoc="center",
    colLoc="center",
    loc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(
    1.0,
    1.5)
for column_number in range(
    len(table_data.columns)):
    header_cell = table[
        (0, column_number)    ]
    header_cell.get_text().set_fontweight(
        "bold"    )
    header_cell.set_linewidth(
        1.5    )
for row_number in range(
    1,
    len(table_data) + 1):
    for column_number in range(
        len(table_data.columns)    ):
        cell = table[
            (row_number, column_number)        ]
        cell.set_linewidth(
            0.9       )
plt.title(
    "CAS and CDP Flight-Level LWC Statistics",
    fontsize=18,
    fontweight="bold",
    pad=20)
plt.tight_layout()
plt.show()
# %%
