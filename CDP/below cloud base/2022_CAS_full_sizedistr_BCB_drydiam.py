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
# bin_center=[0.555, 0.645, 0.715, 0.785, 0.855, 0.925, 
#             0.995, 1.07, 1.14, 1.21, 1.38, 1.75, 2.25, 
bin_center=[ 2.25, 2.75, 3.25, 3.75, 4.5, 5.75, 6.85, 7.55, 
            9.05, 11.4, 13.8, 17.5, 22.5, 27.5, 32.5, 
            37.5, 42.5, 47.5]
#%%
#Summary Data
col_name = ['Time_mid', 'Latitude', 'Longitude', 'GPS_altitude', 'Pressure_Altitude',
             'Pitch', 'Roll', 'True_Heading', 'True_Air_Speed', 
             'Static_Air_Temp', 'IR_Surf_Temp', 'Static_Pressure',
             'Wind_Speed']
summary=[]
dates_sum = ['2022-01-11', '2022-01-12','2022-01-15', '2022-01-18', 
             '2022-01-19', '2022-01-24', '2022-01-26', '2022-01-27',
             '2022-02-01', '2022-02-02', '2022-02-03', '2022-02-05', 
             '2022-02-15', '2022-02-16', '2022-02-19', '2022-02-22',
             '2022-02-26', '2022-03-02', '2022-03-03', '2022-03-04', #'2022-03-07', 
             '2022-03-13', '2022-03-14', '2022-03-18', '2022-03-22',
             '2022-03-26', '2022-03-28', '2022-03-29', #'2022-05-03'
             '2022-05-05', '2022-05-10', '2022-05-16','2022-05-17', '2022-05-18',
             '2022-05-20','2022-05-21', '2022-05-31', '2022-06-02', 
             '2022-06-03', '2022-06-05', '2022-06-07', '2022-06-08', 
             '2022-06-10','2022-06-11', '2022-06-13', '2022-06-14',
             '2022-06-17', '2022-06-18']

for date in dates_sum:
    datestr = date.replace('-', '')
    fname_sum = sorted(glob.glob(f'/home/disk/eos4/kathem24/activate/data/MET/2022/Summary/csv/ACTIVATE-SUMMARY_HU25_{datestr}_R*.csv'), reverse=True)
    # print(date)
    # print(fname_sum)

    run = 1
    for file_path in fname_sum: 
        num_file_paths = len(fname_sum)

        #num = index = dates_sum.index(date)
        
        # try:
        if date > '2022-01-12':
            df_sum = pd.read_csv(file_path, skiprows=47, quoting=csv.QUOTE_NONE)
            # elif date > '2022-05-03':
            #     df_sum = pd.read_csv(file_path, skiprows=47, quoting=csv.QUOTE_NONE)
            # elif date ==  '2022-05-03':
            #     df_sum = pd.read_csv(file_path, skiprows=51, quoting=csv.QUOTE_NONE)
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
leg_data = []
leg_name=['Time_Start', '  Time_Stop', '  Julian_Day', 
          '  Date', '  LegIndex']

dates_legs= ['2022-01-11', '2022-01-12','2022-01-15', '2022-01-18', 
             '2022-01-19', '2022-01-24', '2022-01-26', '2022-01-27',
             '2022-02-01', '2022-02-02', '2022-02-03', '2022-02-05', 
             '2022-02-15', '2022-02-16', '2022-02-19',  '2022-02-22',
             '2022-02-26', '2022-03-02', '2022-03-03', '2022-03-04', #'2022-03-07', 
             '2022-03-13', '2022-03-14', '2022-03-18', '2022-03-22',
             '2022-03-26', '2022-03-28', '2022-03-29', #'2022-05-03',
             '2022-05-05', '2022-05-10','2022-05-16', '2022-05-17', '2022-05-18',
             '2022-05-20', '2022-05-21', '2022-05-31', '2022-06-02', 
             '2022-06-03', '2022-06-05', '2022-06-07', '2022-06-08', 
             '2022-06-10','2022-06-11','2022-06-13', '2022-06-14',
             '2022-06-17', '2022-06-18']

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
## 2D-S Data Import

bin_name = ['dNdlogD_total_003_2DS', 'dNdlogD_total_004_2DS' ,'dNdlogD_total_005_2DS', 
            'dNdlogD_total_006_2DS']

twoDS=[]

dates_twoDS = ['2022-01-11', '2022-01-12','2022-01-15', '2022-01-18', 
             '2022-01-19', '2022-01-24', '2022-01-26', '2022-01-27',
             '2022-02-01', '2022-02-02', '2022-02-03', '2022-02-05', 
             '2022-02-15', '2022-02-16', '2022-02-19',  '2022-02-22',
             '2022-02-26', '2022-03-02', '2022-03-03', '2022-03-04', #'2022-03-07', 
             '2022-03-13', '2022-03-14', '2022-03-18', '2022-03-22',
             '2022-03-26', '2022-03-28', '2022-03-29', #'2022-05-03',
              '2022-05-05', '2022-05-10','2022-05-16', '2022-05-17', '2022-05-18',
             '2022-05-20', '2022-05-21', '2022-05-31', '2022-06-02', 
             '2022-06-03', '2022-06-05', '2022-06-07', '2022-06-08', 
             '2022-06-10','2022-06-11','2022-06-13', '2022-06-14',
             '2022-06-17', '2022-06-18']

for date in dates_twoDS:
    datestr = date.replace('-', '')
    fname_twoDS = sorted(glob.glob(f'/home/disk/eos4/kathem24/activate/data/twoDspectrometer/horizontal/csv/ACTIVATE-2DS-H-Arm_HU25_{datestr}_R*.csv'), reverse=True)
    print(date)
    print(fname_twoDS)

    run = 1
    for file_path in fname_twoDS:
        nums_file_paths = len(fname_twoDS)
        if date <= ('2022-03-29'):
            df_2DS = pd.read_csv(file_path, skiprows= 423, quoting=csv.QUOTE_NONE)
        elif date >= '2022-05-05':
            df_legs = pd.read_csv(file_path, skiprows=424, quoting=csv.QUOTE_NONE)
        for bin_ in bin_name:
            if bin_ in df_2DS.columns:
                df_2DS.columns = df_2DS.columns.str.strip('"')
                df_2DS[bin_] = pd.to_numeric(df_2DS[bin_], errors='coerce')
                df_2DS.replace([-9999, -9999.00], np.NaN, inplace=True)
        for col in ['dNdlogD_total_003_2DS', 
                    'dNdlogD_total_004_2DS' ,
                    'dNdlogD_total_005_2DS', 
                     'dNdlogD_total_006_2DS',
                      'Time_Start', 'LWC_2DS', 'N-total_2DS', 
                      'ED-total_2DS', 'N-liquid_2DS']:
            if df_2DS[col].dtype == 'O':  # 'O' stands for Object (usually string columns)
                df_2DS[col] = df_2DS[col].str.strip('"')

        df_2DS['Time_Start']= pd.to_numeric(df_2DS['Time_Start'], errors='coerce')
        df_2DS['dNdlogD_total_003_2DS']= pd.to_numeric(df_2DS['dNdlogD_total_003_2DS'], errors='coerce')
        df_2DS['dNdlogD_total_004_2DS']= pd.to_numeric(df_2DS['dNdlogD_total_004_2DS'], errors='coerce')
        df_2DS['dNdlogD_total_005_2DS']= pd.to_numeric(df_2DS['dNdlogD_total_004_2DS'], errors='coerce')
        df_2DS['dNdlogD_total_006_2DS']= pd.to_numeric(df_2DS['dNdlogD_total_006_2DS'], errors='coerce')
        df_2DS['LWC_2DS']= pd.to_numeric(df_2DS['LWC_2DS'], errors='coerce')
        df_2DS['N-total_2DS']= pd.to_numeric(df_2DS['N-total_2DS'], errors='coerce')
        df_2DS['ED-total_2DS']= pd.to_numeric(df_2DS['ED-total_2DS'], errors='coerce')
        df_2DS['N-liquid_2DS']=pd.to_numeric(df_2DS['N-liquid_2DS'], errors='coerce')
        if nums_file_paths==2:
            if run==1:
                df4 = df_2DS 
            elif run==2:
                df5 = df_2DS 
                frames = [df5,df4]
                df_2DS = pd.concat(frames)
                twoDS.append(df_2DS)
                break

        if nums_file_paths ==1:
            print("YESSSSSSSSSSSSSSSSSSSSSSSSSSSSs")
            twoDS.append(df_2DS)

        run = run+1

# %%
col_name_h20 = ['Time_Start', 'H2O_DLH', 'RHi_DLH', 'RHw_DLH']
h20=[]
dates_h20 = ['2022-01-11', '2022-01-12','2022-01-15', '2022-01-18', 
             '2022-01-19', '2022-01-24', '2022-01-26', '2022-01-27',
             '2022-02-01', '2022-02-02', '2022-02-03', '2022-02-05', 
             '2022-02-15', '2022-02-16', '2022-02-19', '2022-02-22',
             '2022-02-26', '2022-03-02', '2022-03-03', '2022-03-04', #'2022-03-07', 
             '2022-03-13', '2022-03-14', '2022-03-18', '2022-03-22',
             '2022-03-26', '2022-03-28', '2022-03-29', #'2022-05-03'
             '2022-05-05', '2022-05-10', '2022-05-16','2022-05-17', '2022-05-18',
             '2022-05-20','2022-05-21', '2022-05-31', '2022-06-02', 
             '2022-06-03', '2022-06-05', '2022-06-07', '2022-06-08', 
             '2022-06-10','2022-06-11', '2022-06-13', '2022-06-14',
             '2022-06-17', '2022-06-18']
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
#CAS data

bin_name = ['CAS_Bin12' ,'CAS_Bin13', 'CAS_Bin14', 'CAS_Bin15', 
             'CAS_Bin16', 'CAS_Bin17', 
            'CAS_Bin18', 'CAS_Bin19', 'CAS_Bin20', 'CAS_Bin21', 'CAS_Bin22', 
             'CAS_Bin23', 'CAS_Bin24', 'CAS_Bin25', 'CAS_Bin26',
             'CAS_Bin27', 'CAS_Bin28', 'CAS_Bin29']

CAS = []

dates_CAS = ['2022-01-11', '2022-01-12','2022-01-15', '2022-01-18', 
             '2022-01-19', '2022-01-24', '2022-01-26', '2022-01-27',
             '2022-02-01', '2022-02-02', '2022-02-03', '2022-02-05', 
             '2022-02-15', '2022-02-16', '2022-02-19', '2022-02-22',
             '2022-02-26', '2022-03-02', '2022-03-03', '2022-03-04', 
             '2022-03-13', '2022-03-14', '2022-03-18', '2022-03-22',
             '2022-03-26', '2022-03-28', '2022-03-29', '2022-05-03',
             '2022-05-05', '2022-05-10','2022-05-16','2022-05-18',
             '2022-05-20','2022-05-21', '2022-05-31', '2022-06-02', 
             '2022-06-03', '2022-06-05','2022-06-07', '2022-06-08', 
             '2022-06-10','2022-06-11','2022-06-13', '2022-06-14',
             '2022-06-17', '2022-06-18']

# clear_means = [] 
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

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

    times = CAS_flight.Time_mid.values
    times1 = twoDS_flight.Time_Start.values
    lwc = CAS_flight.LWC_CAS.values
    N_total = twoDS_flight['N-total_2DS'].values

    # Aligning timestamps between CAS and twoDS data
    cas_indices = []
    twods_indices = []

    for j, time in enumerate(times):
        if time in times1:
            cas_indices.append(j)
            twods_indices.append(np.where(times1 == time)[0][0])

    aligned_lwc = lwc[cas_indices]
    aligned_N_total = N_total[twods_indices]
    aligned_times = times[cas_indices]

    total_BCB_means = []

    for k in range(len(BCB_start)):
        start20 = int(BCB_start[k])
        end20 = BCB_stop[k]

        data_labels = []
        BCB_means = []  # List to store the means for this BCB interval

        # Find the start and end indices in the aligned times array
        index7_start = None
        index7_end = None

        for j, aligned_time in enumerate(aligned_times):
            if int(aligned_time) == start20 and index7_start is None:
                index7_start = j
            if int(aligned_time) == end20:
                index7_end = j
                break

        if index7_start is not None and index7_end is not None:
            for j in range(index7_start, index7_end + 1):
                lwc_val = aligned_lwc[j]
                N_val = aligned_N_total[j]
                lwc_label = 'Y' if 0 <= lwc_val <= 0.0025 else 'N'
                N_label = 'Y' if 0 <= N_val <= 100 else 'N'
                label = 'Y' if lwc_label == 'Y' and N_label == 'Y' else 'N'
                data_labels.append(label)
                
                # Collect bin values for calculating means
                bin_values = [
                    CAS_flight.CAS_Bin12.values[cas_indices[j]], CAS_flight.CAS_Bin13.values[cas_indices[j]],
                    CAS_flight.CAS_Bin14.values[cas_indices[j]], CAS_flight.CAS_Bin15.values[cas_indices[j]],
                    CAS_flight.CAS_Bin16.values[cas_indices[j]], CAS_flight.CAS_Bin17.values[cas_indices[j]],
                    CAS_flight.CAS_Bin18.values[cas_indices[j]], CAS_flight.CAS_Bin19.values[cas_indices[j]],
                    CAS_flight.CAS_Bin20.values[cas_indices[j]], CAS_flight.CAS_Bin21.values[cas_indices[j]],
                    CAS_flight.CAS_Bin22.values[cas_indices[j]], CAS_flight.CAS_Bin23.values[cas_indices[j]],
                    CAS_flight.CAS_Bin24.values[cas_indices[j]], CAS_flight.CAS_Bin25.values[cas_indices[j]],
                    CAS_flight.CAS_Bin26.values[cas_indices[j]], CAS_flight.CAS_Bin27.values[cas_indices[j]],
                    CAS_flight.CAS_Bin28.values[cas_indices[j]], CAS_flight.CAS_Bin29.values[cas_indices[j]]
                ]
                BCB_means.append(bin_values)

            if BCB_means:
                BCB_means = np.mean(BCB_means, axis=0)  # Calculate means for this interval
                total_BCB_means.append(BCB_means)  # Add to the total BCB means list

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
    times = CAS_flight.Time_mid.values
    times1 = twoDS_flight.Time_Start.values
    lwc = CAS_flight.LWC_CAS.values
    N_total = twoDS_flight['N-total_2DS'].values

    # Aligning timestamps between CAS and twoDS data
    cas_indices = []
    twods_indices = []

    for j, time in enumerate(times):
        if time in times1:
            cas_indices.append(j)
            twods_indices.append(np.where(times1 == time)[0][0])

    aligned_lwc = lwc[cas_indices]
    aligned_N_total = N_total[twods_indices]
    aligned_times = times[cas_indices]

    # Extracting bin values
    Bin12 = CAS_flight.CAS_Bin12.values[cas_indices]
    Bin13 = CAS_flight.CAS_Bin13.values[cas_indices]
    Bin14 = CAS_flight.CAS_Bin14.values[cas_indices]
    Bin15 = CAS_flight.CAS_Bin15.values[cas_indices]
    Bin16 = CAS_flight.CAS_Bin16.values[cas_indices]
    Bin17 = CAS_flight.CAS_Bin17.values[cas_indices]
    Bin18 = CAS_flight.CAS_Bin18.values[cas_indices]
    Bin19 = CAS_flight.CAS_Bin19.values[cas_indices]
    Bin20 = CAS_flight.CAS_Bin20.values[cas_indices]
    Bin21 = CAS_flight.CAS_Bin21.values[cas_indices]
    Bin22 = CAS_flight.CAS_Bin22.values[cas_indices]
    Bin23 = CAS_flight.CAS_Bin23.values[cas_indices]
    Bin24 = CAS_flight.CAS_Bin24.values[cas_indices]
    Bin25 = CAS_flight.CAS_Bin25.values[cas_indices]
    Bin26 = CAS_flight.CAS_Bin26.values[cas_indices]
    Bin27 = CAS_flight.CAS_Bin27.values[cas_indices]
    Bin28 = CAS_flight.CAS_Bin28.values[cas_indices]
    Bin29 = CAS_flight.CAS_Bin29.values[cas_indices]

    total_BCB_means = []

    for k in range(len(BCB_start)):
        start20 = int(BCB_start[k])
        end20 = BCB_stop[k]

        bin_means = {
            'Date': date,
            'BCB_start': start20,
            'BCB_stop': end20,
            'Bin12_Y_mean': [],
            'Bin13_Y_mean': [],
            'Bin14_Y_mean': [],
            'Bin15_Y_mean': [],
            'Bin16_Y_mean': [],
            'Bin17_Y_mean': [],
            'Bin18_Y_mean': [],
            'Bin19_Y_mean': [],
            'Bin20_Y_mean': [],
            'Bin21_Y_mean': [],
            'Bin22_Y_mean': [],
            'Bin23_Y_mean': [],
            'Bin24_Y_mean': [],
            'Bin25_Y_mean': [],
            'Bin26_Y_mean': [],
            'Bin27_Y_mean': [],
            'Bin28_Y_mean': [],
            'Bin29_Y_mean': [],
            'Bin12_N_mean': [],
            'Bin13_N_mean': [],
            'Bin14_N_mean': [],
            'Bin15_N_mean': [],
            'Bin16_N_mean': [],
            'Bin17_N_mean': [],
            'Bin18_N_mean': [],
            'Bin19_N_mean': [],
            'Bin20_N_mean': [],
            'Bin21_N_mean': [],
            'Bin22_N_mean': [],
            'Bin23_N_mean': [],
            'Bin24_N_mean': [],
            'Bin25_N_mean': [],
            'Bin26_N_mean': [],
            'Bin27_N_mean': [],
            'Bin28_N_mean': [],
            'Bin29_N_mean': [],
        }

        index7_start = None
        index7_end = None

        for j, aligned_time in enumerate(aligned_times):
            if int(aligned_time) == start20 and index7_start is None:
                index7_start = j
            if int(aligned_time) == end20:
                index7_end = j
                break

        if index7_start is not None and index7_end is not None:
            for j in range(index7_start, index7_end + 1):
                lwc_val = aligned_lwc[j]
                N_val = aligned_N_total[j]
                lwc_label = 'Y' if 0 <= lwc_val <= 0.0025 else 'N'
                N_label = 'Y' if 0 <= N_val <= 100 else 'N'
                label = 'Y' if lwc_label == 'Y' and N_label == 'Y' else 'N'

                for bin_label in range(12, 30):
                    bin_key = f'Bin{bin_label}_{label}_mean'
                    bin_value = CAS_flight[f'CAS_Bin{bin_label}'].values[cas_indices[j]]
                    bin_means[bin_key].append(bin_value)

            total_BCB_means.append(bin_means)

    master_CAS_BCB.append(total_BCB_means)

# Print or use master_CAS_BCB as needed
for item in master_CAS_BCB:
    for bin_means in item:
        print(f"Date: {bin_means['Date']}, Start: {bin_means['BCB_start']}, Stop: {bin_means['BCB_stop']}")
        for bin_label in range(12, 30):
            for label in ['Y', 'N']:
                bin_key = f'Bin{bin_label}_{label}_mean'
                if bin_means[bin_key]:
                    mean_value = np.nanmean(bin_means[bin_key])
                    print(f"   Bin{bin_label}_{label} Mean: {mean_value}")
                else:
                    print(f"   Bin{bin_label}_{label} Mean: No valid data")
#%%
# %%
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

# # Check the structure of all_bin_means
# for entry in all_bin_means:
#     print(entry)

# Create a list of unique dates
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
# Print the number of inner dictionaries for each date in master_min_exponential
for date, entries in master_BCB_exponential.items():
    print(f"Date: {date} has {len(entries)} entries")
total_entries = sum(len(entries) for entries in master_BCB_exponential.values())
print(f"Total number of inner dictionaries: {total_entries}")
#%%
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
        all_BCB.append(rh_times)  # List that contains all the Min wind/alt mean dictionaries for 1 flight

    master_BCB_RH.append(all_BCB)

# Step 2: Replace -999 with NaN in master_Min_RH
for flight in master_BCB_RH:
    for leg in flight:
        rh_mean_list = leg['Rh_mean']
        leg['Rh_mean'] = [np.nan if value <=0 else value for value in rh_mean_list]

#%%
#for only the legs present after LWC filtration and master_min_exponential 

# Extract the (date, Min_start, Min_stop) tuples from master_min_exponential
date_leg_set = set()
for entries in master_BCB_exponential.values():
    for entry in entries:
        date = entry['Date']
        BCB_start = entry.get('BCB_start', np.nan)
        BCB_stop = entry.get('BCB_stop', np.nan)
        date_leg_set.add((date, BCB_start, BCB_stop))

# Filter master_Min_RH
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
# Flatten the list of lists and count the total number of entries
total_entries_filtered_master_BCB_RH = sum(len(legs) for legs in filtered_master_BCB_RH)
print(f"Total entries in filtered_master_BCB_RH: {total_entries_filtered_master_BCB_RH}")
#%%
#%%

#%%
## Convert to dry diameter
## Begin converting to dry size distribution

##Equations 1, 2, and 3 on Rob's sheets 

##ambient salt distributions n(d) measured as a function of wet diameter
## dw

##dry salt diameter ddry

##the total concentration of all the drops with ambient diameter larger than 
##dmin is Nt = integral from dmin to infinity of n(d) * dd

##where n(d) is represented by Noe(-d/D*) where we obtained No and D* from 
##code above 

##sub this equation into Nt to now get Nt = NoD*e(-dmin/D*)

##our threshold minimum dry diameter is 2um

##first we need g(RH) which is about = [1.7 / (1-RH)]^0.31
#%%

##Obtaining g(RH) = [1.7 / (1-RH)]^0.31 for all mean RH values 
## ie for every leg 

master_BCB_gRH = []

# Iterate over each flight in master_Min_RH
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
#only the grh from filtered_master_min_RH
filtered_master_BCB_gRH = []

# Iterate over each flight in master_Min_RH
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
##but before we do this we need to solve for average N+ intercept for every flight leg
##We know that N+ = No(gRH) where No is the intercept of our ambient distribution

master_BCB_interceptdry_dict = {}

# Flatten master_min_gRH if it's a list of lists
if isinstance(master_BCB_gRH[0], list):
    master_BCB_gRH = [item for sublist in master_BCB_gRH for item in sublist]

# Loop through each entry in master_min_gRH
for entry in master_BCB_gRH:
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    gRh_mean = entry['gRh_mean'][0]  # Assuming gRh_mean is a list with one value
    Rh_mean = entry['Rh_mean'][0]  # Assuming Rh_mean is a list with one value

    # Skip entries with invalid Rh_mean values
    if Rh_mean < 0:
        continue

    # Create a unique key for this entry
    key = (date, BCB_start, BCB_end)

    # Find corresponding exponential parameters
    if date in master_BCB_exponential:
        exp_params_list = master_BCB_exponential[date]
        for exp_params in exp_params_list:
            n0 = exp_params['n0']
            D = exp_params['D']
            
            # Calculate Ntd
            dryintercept = n0 * (gRh_mean)
            
            # Store the result in the dictionary
            master_BCB_interceptdry_dict[key] = {
                'Date': date,
                'BCB_start': BCB_start,
                'BCB_stop': BCB_end,
                'Rh_mean': entry['Rh_mean'],
                'gRh_mean': entry['gRh_mean'],
                'dry intercept': dryintercept
            }

master_BCB_dryintercept = list(master_BCB_interceptdry_dict.values())
#%%
#filtered dry intercept 

filtered_master_BCB_interceptdry_dict = {}

# Flatten master_min_gRH if it's a list of lists
if isinstance(filtered_master_BCB_gRH[0], list):
    filtered_master_BCB_gRH = [item for sublist in filtered_master_BCB_gRH for item in sublist]


unique_keys = set()

# Flatten master_min_exponential
flattened_exponential = []
for exp_list in master_BCB_exponential.values():
    flattened_exponential.extend(exp_list)

# Create a dictionary for quick lookup
exponential_dict = {(exp['Date'], exp['BCB_start'], exp['BCB_stop']): exp for exp in flattened_exponential}

# Loop through each entry in filtered_master_min_gRH
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
            
            # Calculate Ntd
            dryintercept = n0 * (gRh_mean)
            
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
master_BCB_ntd_dict = {}

# Flatten master_min_gRH if it's a list of lists
if isinstance(master_BCB_gRH[0], list):
    master_BCB_gRH = [item for sublist in master_BCB_gRH for item in sublist]

# Define a constant for ddrymin
ddrymin = 2  # Update this as needed

# Loop through each entry in master_min_gRH
for entry in master_BCB_gRH:
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
            master_BCB_ntd_dict[key] = {
                'Date': date,
                'BCB_start': BCB_start,
                'BCB_end': BCB_stop,
                'Rh_mean': entry['Rh_mean'],
                'gRh_mean': entry['gRh_mean'],
                'Ntd': Ntd
            }

master_BCB_ntd = list(master_BCB_ntd_dict.values())
#%%
#filtered ntd 
filtered_master_BCB_ntd_dict = {}

# Flatten master_min_gRH if it's a list of lists
if isinstance(filtered_master_BCB_gRH[0], list):
    filtered_master_BCB_gRH = [item for sublist in filtered_master_BCB_gRH for item in sublist]


unique_keys = set()

# Flatten master_min_exponential
flattened_exponential = []
for exp_list in master_BCB_exponential.values():
    flattened_exponential.extend(exp_list)

# Create a dictionary for quick lookup
exponential_dict = {(exp['Date'], exp['BCB_start'], exp['BCB_stop']): exp for exp in flattened_exponential}
# Define a constant for ddrymin
ddrymin = 2  # Update this as needed

# Loop through each entry in master_min_gRH
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
# %%
## Now we can ask how ntd is related to nt (ambient)

## ntd / nt = e(-grh* drymin + dmin / D*) where dmin is 2.5 for the CAS data
##this is equation 8 on Rob's sheet

master_BCB_NtdNt_dict = {}

# Flatten master_min_gRH if it's a list of lists
if isinstance(master_BCB_gRH[0], list):
    master_BCB_gRH = [item for sublist in master_BCB_gRH for item in sublist]

# Constants for calculations
ddrymin = 2
dmin = 2.5  

# Loop through each entry in master_min_gRH
for entry in master_BCB_gRH:
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_end = entry['BCB_end']
    gRh_mean = entry['gRh_mean'][0]  # Assuming gRh_mean is a list with one value
    Rh_mean = entry['Rh_mean'][0]  # Assuming Rh_mean is a list with one value

    # Skip entries with invalid Rh_mean values
    if Rh_mean < 0:
        continue

    # Create a unique key for this entry
    key = (date, BCB_start, BCB_end)

    # Find corresponding exponential parameters
    if date in master_BCB_exponential:
        exp_params_list = master_BCB_exponential[date]
        for exp_params in exp_params_list:
            D = exp_params['D']
            
            # Calculate NtdNt using the given formula
            NtdNt = np.exp((-gRh_mean * ddrymin + dmin) / D)
            
            # Store the result in the dictionary
            master_BCB_NtdNt_dict[key] = {
                'Date': date,
                'BCB_start': BCB_start,
                'BCB_end': BCB_end,
                'Rh_mean': entry['Rh_mean'],
                'gRh_mean': entry['gRh_mean'],
                'NtdNt': NtdNt
            }

# Convert the dictionary to a list
master_BCB_NtdNt = list(master_BCB_NtdNt_dict.values())
#%%
#filtered master_BCB_NtdNt

filtered_master_BCB_NtdNt_dict = {}
# Flatten master_min_gRH if it's a list of lists
if isinstance(filtered_master_BCB_gRH[0], list):
    filtered_master_BCB_gRH = [item for sublist in filtered_master_BCB_gRH for item in sublist]

# Constants for calculations
ddrymin = 2
dmin = 2.5  

# Create a set of unique keys to avoid duplicates
unique_keys = set()

# Flatten master_min_exponential
flattened_exponential = []
for exp_list in master_BCB_exponential.values():
    flattened_exponential.extend(exp_list)

# Create a dictionary for quick lookup
exponential_dict = {(exp['Date'], exp['BCB_start'], exp['BCB_stop']): exp for exp in flattened_exponential}

# Loop through each entry in filtered_master_min_gRH
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

    # Check if the key exists in the exponential dictionary
    if key in exponential_dict:
        exp_params = exponential_dict[key]
        D = exp_params['D']
        
        # Calculate NtdNt using the given formula
        NtdNt = np.exp((-gRh_mean * ddrymin + dmin) / D)
        
        # Store the result in the dictionary
        filtered_master_BCB_NtdNt_dict[key] = {
            'Date': date,
            'BCB_start': BCB_start,
            'BCB_stop': BCB_stop,
            'Rh_mean': entry['Rh_mean'],
            'gRh_mean': entry['gRh_mean'],
            'NtdNt': NtdNt
        }

# Convert the dictionary to a list
filtered_master_BCB_NtdNt = list(filtered_master_BCB_NtdNt_dict.values())

print(f"Length of filtered_master_BCB_NtdNt: {len(filtered_master_BCB_NtdNt)}")

# %%
master_BCB = []


for i in range(len(dates_legs)):
    date = dates_legs[i]

    leg_dict = leg_data[i]

    flight_date = leg_dict['Date'] #Get date of flight from dictionary 
    BCB_start = leg_dict['LegIndex_02']['StartTimes']
    BCB_stop = leg_dict['LegIndex_02']['StopTimes']
    sum_flight = summary[i]
    # if date == '2022-02-02':
    #     print(date)
    #     print(Min_Start)
    #     print(Min_Stop)

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

        all_BCB_means.append(wind_alt) #List that contains all the Min wind/alt mean dictionaries for 1 flight
        
    master_BCB.append(all_BCB_means) #List that contains all Min flights  
#%%
Z0 = 0.03  # meters (typical value for open terrain)
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

            # Append to the new dictionary
            corrected_calc_bcb['Date'].append(date)
            corrected_calc_bcb['Corrected_bcb_windspeed'].append(new_windspeed)
            
for date, wind_mean in zip(corrected_calc_bcb['Date'], corrected_calc_bcb['Corrected_bcb_windspeed']):
    print(f"Date: {date}, Corrected_bcb_windspeed: {wind_mean}")
# %%
Z0 = 0.03  # meters (typical value for open terrain)
Z10 = 10  # target height m

# Function to apply windspeed correction
def correct_windspeed(windspeed, altitude):
    return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))

# Initialize combined data dictionary
combined_data = {
    'Date': [],
    'Altitude': [],
    'RH': [],
    'Windspeed': []
}

# Validate and combine the data
for i, flight in enumerate(master_BCB):
    for j, wind_alt in enumerate(flight):
        try:
            date = wind_alt['Date']
            alt_mean = wind_alt['Alts_mean'][0]
            rh_mean = master_BCB_RH[i][j]['Rh_mean'][0]
            
            # Correct windspeed using the provided formula
            wind_mean = wind_alt['Winds_mean'][0]
            if not np.isnan(alt_mean) and alt_mean > 0:
                corrected_windspeed = correct_windspeed(wind_mean, alt_mean)
            else:
                corrected_windspeed = np.nan

            # Check if RH is valid
            if rh_mean < 0 or rh_mean > 100:
                print(f"Invalid RH value {rh_mean} at index {i}, skipping...")
                continue

            combined_data['Date'].append(date)
            combined_data['Altitude'].append(alt_mean)
            combined_data['RH'].append(rh_mean)
            combined_data['Windspeed'].append(corrected_windspeed)

        except IndexError as e:
            print(f"Index error at i={i}, j={j}: {e}")
            continue

df_combined = pd.DataFrame(combined_data)

# Check the data for any anomalies
print(df_combined.describe())

# Separate data based on NaN Windspeed
df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]

# Plot data
plt.figure(figsize=(10, 8))

# Plot data with valid Windspeed values
sc = plt.scatter(df_with_windspeed['RH'], df_with_windspeed['Altitude'], 
                 c=df_with_windspeed['Windspeed'], cmap='viridis', s=50, label='Windspeed Present')

# Plot data with NaN Windspeed values in black
plt.scatter(df_nan_windspeed['RH'], df_nan_windspeed['Altitude'], 
            color='black', s=50, label='Windspeed NaN')

plt.colorbar(sc, label='Mean Leg Windspeed (m/s)')
plt.xlabel('Mean Leg RH (%)', fontsize=14, fontweight='bold')
plt.ylabel('Mean Leg Altitude (m)', fontsize=14, fontweight='bold')
plt.title('BCB January-June 2022', fontsize=14, fontweight='bold')
# plt.legend()
plt.show()
#%%
#filtered 
Z0 = 0.03  # meters (typical value for open terrain)
Z10 = 10  # target height m

# Function to apply windspeed correction
def correct_windspeed(windspeed, altitude):
    return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))

# Initialize combined data dictionary
combined_data = {
    'Date': [],
    'Altitude': [],
    'RH': [],
    'Windspeed': []
}

# Validate and combine the data
for i, flight in enumerate(master_BCB):
    for j, wind_alt in enumerate(flight):
        try:
            date = wind_alt['Date']
            alt_mean = wind_alt['Alts_mean'][0]
            rh_mean = filtered_master_BCB_RH[i][j]['Rh_mean'][0]
            
            # Correct windspeed using the provided formula
            wind_mean = wind_alt['Winds_mean'][0]
            if not np.isnan(alt_mean) and alt_mean > 0:
                corrected_windspeed = correct_windspeed(wind_mean, alt_mean)
            else:
                corrected_windspeed = np.nan

            # Check if RH is valid
            if rh_mean < 0 or rh_mean > 100:
                print(f"Invalid RH value {rh_mean} at index {i}, skipping...")
                continue

            combined_data['Date'].append(date)
            combined_data['Altitude'].append(alt_mean)
            combined_data['RH'].append(rh_mean)
            combined_data['Windspeed'].append(corrected_windspeed)

        except IndexError as e:
            print(f"Index error at i={i}, j={j}: {e}")
            continue

df_combined = pd.DataFrame(combined_data)

# Check the data for any anomalies
print(df_combined.describe())

# Separate data based on NaN Windspeed
df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]

# Plot data
plt.figure(figsize=(10, 8))

# Plot data with valid Windspeed values
sc = plt.scatter(df_with_windspeed['RH'], df_with_windspeed['Altitude'], 
                 c=df_with_windspeed['Windspeed'], cmap='viridis', s=50, label='Windspeed Present')

# Plot data with NaN Windspeed values in black
plt.scatter(df_nan_windspeed['RH'], df_nan_windspeed['Altitude'], 
            color='black', s=50, label='Windspeed NaN')

plt.colorbar(sc, label='Mean Leg Windspeed (m/s)')
plt.xlabel('Mean Leg RH (%)', fontsize=14, fontweight='bold')
plt.ylabel('Mean Leg Altitude (m)', fontsize=14, fontweight='bold')
plt.title('Below Cloud Base January-June 2022 w/ clear sky restrictions', fontsize=14, fontweight='bold')
# plt.legend()
plt.show()
# %%
Z0 = 0.03  # meters (typical value for open terrain)
Z10 = 10  # target height m

# Function to apply windspeed correction
def correct_windspeed(windspeed, altitude):
    return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))
combined_data = {
    'Date': [],
    'D': [],
    'dryintercept': [],
    'Windspeed': []
}
# Validate and combine the data
for i, flight in enumerate(master_BCB):
    for j, wind_alt in enumerate(flight):
        try:
            date = wind_alt['Date']
            wind_mean = wind_alt['Winds_mean'][0]
            alt_mean = wind_alt['Alts_mean'][0]
            
            # Correct windspeed using the provided formula
            if not np.isnan(alt_mean) and alt_mean > 0:
                corrected_windspeed = correct_windspeed(wind_mean, alt_mean)
            else:
                corrected_windspeed = np.nan
            
            # Find the corresponding exponential parameters and dry intercept
            if date in master_BCB_exponential:
                exp_params_list = master_BCB_exponential[date]
                gRh_mean = master_BCB_gRH[i][j]['gRh_mean'][0]  # Assume matching index

                for exp_params in exp_params_list:
                    D = exp_params['D']
                    n0 = exp_params['n0']
                    dryintercept = n0 * gRh_mean
                    
                    # Append data to the combined dictionary
                    combined_data['Date'].append(date)
                    combined_data['D'].append(D)
                    combined_data['dryintercept'].append(dryintercept)
                    combined_data['Windspeed'].append(corrected_windspeed)

        except IndexError as e:
            print(f"Index error at i={i}, j={j}: {e}")
            continue
# Convert to DataFrame
df_combined = pd.DataFrame(combined_data)

# Check the data for any anomalies
print(df_combined.describe())

# Separate data based on NaN Windspeed
df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]


# Calculate the correlation matrix
correlation_matrix = df_combined[['dryintercept', 'D', 'Windspeed']].corr()

# Visualize the correlation matrix using a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

from mpl_toolkits.mplot3d import Axes3D

# # 3D scatter plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# sc = ax.scatter(df_combined['D'], df_combined['dryintercept'], df_combined['Windspeed'], 
#                 c=df_combined['Windspeed'], cmap='viridis', s=100)
# ax.set_xlabel('D')
# ax.set_ylabel('Dry Intercept')
# ax.set_zlabel('Windspeed')

# # Add color bar
# fig.colorbar(sc, label='Windspeed (m/s)')
# plt.title('3D Scatter Plot of D, Dry Intercept, and Windspeed')
# plt.show()
sns.pairplot(df_combined[['dryintercept', 'D', 'Windspeed']], diag_kind='kde', palette='viridis')
plt.suptitle('Pair Plot of Dry Intercept, D, and Windspeed', y=1.02)
plt.show()
# sns.lmplot(x='D', y='dryintercept', data=df_combined, hue='Windspeed', palette='viridis', height=7, aspect=1.5)
# plt.title('Linear Regression of Dry Intercept vs D with Windspeed Hue')
# plt.show()


# # Create a hexbin plot
# plt.figure(figsize=(10, 8))
# hb = plt.hexbin(df_combined['D'], df_combined['dryintercept'], gridsize=30, cmap='viridis', mincnt=1)
# plt.colorbar(hb, label='Counts')
# plt.xlabel('D')
# plt.ylabel('Dry Intercept')
# plt.title('Hexbin Plot of Dry Intercept vs D')
# plt.show()

plt.figure(figsize=(10, 8))

# Plot data with valid Windspeed values
sc = plt.scatter(df_with_windspeed['D'], df_with_windspeed['dryintercept'], 
                 c=df_with_windspeed['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

# Plot data with NaN Windspeed values in black
plt.scatter(df_nan_windspeed['D'], df_nan_windspeed['dryintercept'], 
            color='black', s=100, label='Windspeed NaN')

# Add color bar
plt.colorbar(sc, label='Corrected Windspeed (m/s)')

# Add labels and title
plt.xlabel('Slope', fontsize=14, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.title('BCB January-June 2022', fontsize=14, fontweight='bold')

# Show the plot
plt.show()
#%%

#Filtered 

Z0 = 0.03  # meters (typical value for open terrain)
Z10 = 10  # target height m

# Function to apply windspeed correction
def correct_windspeed(windspeed, altitude):
    return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))
combined_data = {
    'Date': [],
    'D': [],
    'dryintercept': [],
    'Windspeed': []
}
# Validate and combine the data
for i, flight in enumerate(master_BCB):
    for j, wind_alt in enumerate(flight):
        try:
            date = wind_alt['Date']
            wind_mean = wind_alt['Winds_mean'][0]
            alt_mean = wind_alt['Alts_mean'][0]
            
            # Correct windspeed using the provided formula
            if not np.isnan(alt_mean) and alt_mean > 0:
                corrected_windspeed = correct_windspeed(wind_mean, alt_mean)
            else:
                corrected_windspeed = np.nan
            
            # Find the corresponding exponential parameters and dry intercept
            if date in master_BCB_exponential:
                exp_params_list = master_BCB_exponential[date]
                gRh_mean = filtered_master_BCB_gRH[i][j]['gRh_mean'][0]  # Assume matching index

                for exp_params in exp_params_list:
                    D = exp_params['D']
                    n0 = exp_params['n0']
                    dryintercept = n0 * gRh_mean
                    
                    # Append data to the combined dictionary
                    combined_data['Date'].append(date)
                    combined_data['D'].append(D)
                    combined_data['dryintercept'].append(dryintercept)
                    combined_data['Windspeed'].append(corrected_windspeed)

        except IndexError as e:
            print(f"Index error at i={i}, j={j}: {e}")
            continue
# Convert to DataFrame
df_combined = pd.DataFrame(combined_data)

# Check the data for any anomalies
# print(df_combined.describe())

# Separate data based on NaN Windspeed
df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]


# # Calculate the correlation matrix
# correlation_matrix = df_combined[['dryintercept', 'D', 'Windspeed']].corr()

# Visualize the correlation matrix using a heatmap
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Matrix w/ clear sky restrictions')
# plt.show()


# from mpl_toolkits.mplot3d import Axes3D

# # 3D scatter plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# sc = ax.scatter(df_combined['D'], df_combined['dryintercept'], df_combined['Windspeed'], 
#                 c=df_combined['Windspeed'], cmap='viridis', s=100)

# # ax.set_xscale('log')
# # # ax.set_yscale('log')
# # # ax.set_zscale('log')
# ax.set_xlabel('D')
# ax.set_ylabel('Dry Intercept')
# ax.set_zlabel('Windspeed')

# # Add color bar
# fig.colorbar(sc, label='Windspeed (m/s)')
# plt.title('3D Scatter Plot of D, Dry Intercept, and Windspeed w/ clear sky restrictions')
# plt.show()
# sns.pairplot(df_combined[['dryintercept', 'D', 'Windspeed']], diag_kind='kde', palette='viridis')

# plt.suptitle('Pair Plot of Dry Intercept, D, and Windspeed', y=1.02)
# plt.yscale('log')
# plt.xscale('log')
# plt.show()
# # sns.lmplot(x='D', y='dryintercept', data=df_combined, hue='Windspeed', palette='viridis', height=7, aspect=1.5)
# # plt.title('Linear Regression of Dry Intercept vs D with Windspeed Hue w/ clear sky restrictions')
# # plt.show()


# # Create a hexbin plot
# plt.figure(figsize=(10, 8))
# hb = plt.hexbin(df_combined['D'], df_combined['dryintercept'], gridsize=30, cmap='viridis', mincnt=1)
# plt.colorbar(hb, label='Counts')
# plt.xlabel('D')
# plt.ylabel('Dry Intercept')
# plt.title('Hexbin Plot of Dry Intercept vs D w/ clear sky restrictions')
# plt.show()

# plt.figure(figsize=(10, 8))

# Plot data with valid Windspeed values
sc = plt.scatter(df_with_windspeed['D'], df_with_windspeed['dryintercept'], 
                 c=df_with_windspeed['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

# Plot data with NaN Windspeed values in black
plt.scatter(df_nan_windspeed['D'], df_nan_windspeed['dryintercept'], 
            color='grey', s=100, label='Windspeed NaN')

# Add color bar
plt.colorbar(sc, label='Corrected Windspeed (m/s)')

# Add labels and title
plt.xlabel('D', fontsize=14, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.title('Scatterplot of D vs. Dry Intercept w/ clear sky restrictions', fontsize=14, fontweight='bold')
# Show the plot
plt.xscale('log')
plt.show()
#%%
##Phase space for D and No
from scipy.stats import gaussian_kde
from scipy.integrate import quad

#%%

Z0 = 0.03  # meters (typical value for open terrain)
Z10 = 10  # target height m

# Function to apply windspeed correction
def correct_windspeed(windspeed, altitude):
    return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))

combined_data = {
    'Date': [],
    'D': [],
    'dryintercept': [],
    'Windspeed': []
}

# Validate and combine the data
for i, flight in enumerate(master_BCB):
    for j, wind_alt in enumerate(flight):
        try:
            date = wind_alt['Date']
            wind_mean = wind_alt['Winds_mean'][0]
            alt_mean = wind_alt['Alts_mean'][0]
            
            # Correct windspeed using the provided formula
            if not np.isnan(alt_mean) and alt_mean > 0:
                corrected_windspeed = correct_windspeed(wind_mean, alt_mean)
            else:
                corrected_windspeed = np.nan
            
            # Find the corresponding exponential parameters and dry intercept
            if date in master_BCB_exponential:
                exp_params_list = master_BCB_exponential[date]
                gRh_mean = filtered_master_BCB_gRH[i][j]['gRh_mean'][0]  # Assume matching index

                for exp_params in exp_params_list:
                    D = exp_params['D']
                    n0 = exp_params['n0']
                    dryintercept = n0 * gRh_mean
                    
                    # Append data to the combined dictionary
                    combined_data['Date'].append(date)
                    combined_data['D'].append(D)
                    combined_data['dryintercept'].append(dryintercept)
                    combined_data['Windspeed'].append(corrected_windspeed)

        except IndexError as e:
            print(f"Index error at i={i}, j={j}: {e}")
            continue

# Convert to DataFrame
df_combined = pd.DataFrame(combined_data)

# Check the data for any anomalies
print(df_combined.describe())

# Separate data based on NaN Windspeed
df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]

# Check for NaN or Inf values
if df_combined[['D', 'dryintercept']].isnull().any().any():
    print("Data contains NaN values.")
    # Filter out NaN values
    df_combined = df_combined.dropna(subset=['D', 'dryintercept'])

if np.any(np.isinf(df_combined[['D', 'dryintercept']].values)):
    print("Data contains Inf values.")
    # Optionally filter out inf values
    df_combined = df_combined[np.isfinite(df_combined[['D', 'dryintercept']].values).all(axis=1)]

# Filter out NaN and Inf values for plotting
filtered_combined = df_combined[df_combined['Windspeed'].notna() & df_combined['D'].notna() & df_combined['dryintercept'].notna()]
filtered_combined = filtered_combined[np.isfinite(filtered_combined[['D', 'dryintercept']].values).all(axis=1)]

# Plot data points with valid Windspeed values
sc = plt.scatter(filtered_combined['D'], filtered_combined['dryintercept'], 
                 c=filtered_combined['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

# Plot data points with NaN Windspeed values in black
plt.scatter(df_nan_windspeed['D'], df_nan_windspeed['dryintercept'], 
            color='grey', s=100, label='Windspeed NaN')

# Calculate density for contours
kde = gaussian_kde(np.vstack([filtered_combined['D'], filtered_combined['dryintercept']]))
xgrid = np.linspace(min(filtered_combined['D']), max(filtered_combined['D']), 100)
ygrid = np.linspace(min(filtered_combined['dryintercept']), max(filtered_combined['dryintercept']), 100)
X, Y = np.meshgrid(xgrid, ygrid)
Z = kde(np.vstack([X.ravel(), Y.ravel()]))
Z = Z.reshape(X.shape)

# Plot contours for total concentration
contour_levels = np.linspace(0, Z.max(), num=10)  # Adjust number of levels as needed
plt.contour(X, Y, Z, levels=contour_levels, colors='blue', alpha=0.5)

# Add color bar
plt.colorbar(sc, label='Corrected Windspeed (m/s)')

# Add labels and title
plt.xlabel('D', fontsize=14, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.title('Density Contours' , fontweight='bold')
plt.xscale('log')
plt.xlim(10**-2, 10**2)
plt.ylim(10**-2, 10**2.5)
plt.show()
#%%


# Full integrated mass calculation: M = N0 * D^4
def calculate_mass(N0, D):
    return N0 * D**4

# Filter out rows where D or dryintercept is NaN or <= 0
filtered_combined_clean = filtered_combined[(filtered_combined['D'] > 0) & 
                                            (filtered_combined['dryintercept'] > 0)].copy()

# Recalculate mass for each point using the full integrated mass equation
filtered_combined_clean['Mass'] = calculate_mass(filtered_combined_clean['dryintercept'], 
                                                 filtered_combined_clean['D'])

# Debugging: Check minimum and maximum mass values
print(f"Min mass: {filtered_combined_clean['Mass'].min()}, Max mass: {filtered_combined_clean['Mass'].max()}")

# Create data-based grids to better match the distribution of the points
xgrid = np.logspace(np.log10(filtered_combined_clean['D'].min()), 
                    np.log10(filtered_combined_clean['D'].max()), 100)
ygrid = np.logspace(np.log10(filtered_combined_clean['dryintercept'].min()), 
                    np.log10(filtered_combined_clean['dryintercept'].max()), 100)
D_grid, dryintercept_grid = np.meshgrid(xgrid, ygrid)

# Calculate mass for each point on the grid
mass_grid = calculate_mass(dryintercept_grid, D_grid)

# Debugging: Check if mass_grid contains meaningful values
print(f"Mass grid: Min = {np.nanmin(mass_grid)}, Max = {np.nanmax(mass_grid)}")

# Define mass contour levels (adjust to fit data range)
mass_levels = np.logspace(np.log10(filtered_combined_clean['Mass'].min()), 
                          np.log10(filtered_combined_clean['Mass'].max()), 20)
print(f"Mass levels: {mass_levels}")

# Plot the scatter plot with Windspeed color map
sc = plt.scatter(filtered_combined_clean['D'], filtered_combined_clean['dryintercept'], 
                 c=filtered_combined_clean['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

# Plot NaN Windspeed points in black (if any)
if not df_nan_windspeed.empty:
    plt.scatter(df_nan_windspeed['D'], df_nan_windspeed['dryintercept'], 
                color='grey', s=100, label='Windspeed NaN')

# Plot the mass contours using the full integrated mass formula
contour_plot = plt.contour(D_grid, dryintercept_grid, mass_grid, levels=mass_levels, colors='blue', alpha=0.75)

# Check if contours were created
if len(contour_plot.allsegs[0]) == 0:
    print("No contours were created. Check your data range or mass grid calculation.")

# Add color bar for windspeed
plt.colorbar(sc, label='Corrected Windspeed (m/s)')

# Add labels and formatting
plt.xlabel('D', fontsize=14, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.title(' Mass Contours', fontsize=14, fontweight='bold')
plt.xlim(10**-2, 10**2)
plt.ylim(10**-2, 10**2.5)

# Display the plot
plt.show()
#%%
#adjusting integral to 2 to infinity and changing contour number and appearance 


# Function to calculate mass with lower limit of integration at 2
def calculate_mass(N0, D):
    integrand = lambda d: np.exp(-d / D) * d**3
    mass_integral, _ = quad(integrand, 2, np.inf)
    return N0 * mass_integral

# Filter out rows where D or dryintercept is NaN or <= 0
filtered_combined_clean = filtered_combined[(filtered_combined['D'] > 0) & 
                                            (filtered_combined['dryintercept'] > 0)].copy()

# Recalculate mass for each point using the integrated mass equation (apply row-wise)
filtered_combined_clean['Mass'] = filtered_combined_clean.apply(
    lambda row: calculate_mass(row['dryintercept'], row['D']), axis=1)

# Debugging: Check minimum and maximum mass values
print(f"Min mass: {filtered_combined_clean['Mass'].min()}, Max mass: {filtered_combined_clean['Mass'].max()}")

# Create data-based grids to better match the distribution of the points
xgrid = np.logspace(np.log10(filtered_combined_clean['D'].min()), 
                    np.log10(filtered_combined_clean['D'].max()), 100)
ygrid = np.logspace(np.log10(filtered_combined_clean['dryintercept'].min()), 
                    np.log10(filtered_combined_clean['dryintercept'].max()), 100)
D_grid, dryintercept_grid = np.meshgrid(xgrid, ygrid)

# Calculate mass for each point on the grid (use np.vectorize to apply calculate_mass element-wise)
vectorized_mass = np.vectorize(calculate_mass)
mass_grid = vectorized_mass(dryintercept_grid, D_grid)

# Debugging: Check if mass_grid contains meaningful values
print(f"Mass grid: Min = {np.nanmin(mass_grid)}, Max = {np.nanmax(mass_grid)}")

# Define fewer mass contour levels (about half as many as before)
mass_levels = np.logspace(np.log10(filtered_combined_clean['Mass'].min()), 
                          np.log10(filtered_combined_clean['Mass'].max()), 10)
print(f"Mass levels: {mass_levels}")

# Plot the scatter plot with Windspeed color map
sc = plt.scatter(filtered_combined_clean['D'], filtered_combined_clean['dryintercept'], 
                 c=filtered_combined_clean['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

# Plot NaN Windspeed points in black (if any)
if not df_nan_windspeed.empty:
    plt.scatter(df_nan_windspeed['D'], df_nan_windspeed['dryintercept'], 
                color='grey', s=100, label='Windspeed NaN')

# Plot the mass contours using the full integrated mass formula with reduced number of contours
contour_plot = plt.contour(D_grid, dryintercept_grid, mass_grid, levels=mass_levels, 
                           colors='red', linewidths=1, alpha=0.75)

# Check if contours were created
if len(contour_plot.allsegs[0]) == 0:
    print("No contours were created. Check your data range or mass grid calculation.")

# Add color bar for windspeed
plt.colorbar(sc, label='Corrected Windspeed (m/s)')

# Add labels and formatting
plt.xlabel('D', fontsize=14, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.title(' Mass Contours', fontsize=14, fontweight='bold')
plt.xlim(10**-2, 10**2)
plt.ylim(10**-2, 10**2.5)

# Display the plot
plt.show()
#%%

# Function to calculate mass with lower limit of integration at 2
def calculate_mass(N0, D):
    integrand = lambda d: np.exp(-d / D) * d**3
    mass_integral, _ = quad(integrand, 2, np.inf)
    return N0 * mass_integral

# Filter out rows where D or dryintercept is NaN or <= 0
filtered_combined_clean = filtered_combined[(filtered_combined['D'] > 0) & 
                                            (filtered_combined['dryintercept'] > 0)].copy()

# Recalculate mass for each point using the integrated mass equation (apply row-wise)
filtered_combined_clean['Mass'] = filtered_combined_clean.apply(
    lambda row: calculate_mass(row['dryintercept'], row['D']), axis=1)

# Debugging: Check minimum and maximum mass values
print(f"Min mass: {filtered_combined_clean['Mass'].min()}, Max mass: {filtered_combined_clean['Mass'].max()}")

# Create data-based grids to better match the distribution of the points
xgrid = np.logspace(np.log10(filtered_combined_clean['D'].min()), 
                    np.log10(filtered_combined_clean['D'].max()), 100)
ygrid = np.logspace(np.log10(filtered_combined_clean['dryintercept'].min()), 
                    np.log10(filtered_combined_clean['dryintercept'].max()), 100)
D_grid, dryintercept_grid = np.meshgrid(xgrid, ygrid)

# Calculate mass for each point on the grid (use np.vectorize to apply calculate_mass element-wise)
vectorized_mass = np.vectorize(calculate_mass)
mass_grid = vectorized_mass(dryintercept_grid, D_grid)

# Debugging: Check if mass_grid contains meaningful values
print(f"Mass grid: Min = {np.nanmin(mass_grid)}, Max = {np.nanmax(mass_grid)}")

low_levels = np.logspace(np.log10(filtered_combined_clean['Mass'].min()), np.log10(1), 6, endpoint=False)  # 3 contours in low range (up to 10^0)
high_levels = np.logspace(np.log10(1), np.log10(filtered_combined_clean['Mass'].max()), 5)  # More contours in higher range (10^0 to max)

# Ensure contours are strictly increasing by combining low and high levels
mass_levels = np.concatenate([low_levels, high_levels])

print(f"Mass levels: {mass_levels}")

# Plot the scatter plot with Windspeed color map
sc = plt.scatter(filtered_combined_clean['D'], filtered_combined_clean['dryintercept'], 
                 c=filtered_combined_clean['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

# Plot the mass contours using the full integrated mass formula with reduced number of contours
contour_plot = plt.contour(D_grid, dryintercept_grid, mass_grid, levels=mass_levels, 
                           colors='red', linewidths=0.5, alpha=0.75)

# Check if contours were created
if len(contour_plot.allsegs[0]) == 0:
    print("No contours were created. Check your data range or mass grid calculation.")

# Add color bar for windspeed
plt.colorbar(sc, label='Corrected Windspeed (m/s)')

# Add labels and formatting
plt.xlabel('D', fontsize=14, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.title('Mass Contours', fontsize=14, fontweight='bold')
plt.xlim(10**-2, 10**2)
plt.ylim(10**-2, 10**2.5)

# Display the plot
plt.show()
#%%
##overlaying dry concentration contours onto mass contour plot


# Function to calculate mass with lower limit of integration at 2
def calculate_mass(N0, D):
    integrand = lambda d: np.exp(-d / D) * d**3
    mass_integral, _ = quad(integrand, 2, np.inf)
    return N0 * mass_integral

# Filter out rows where D or dryintercept is NaN or <= 0
filtered_combined_clean = filtered_combined[(filtered_combined['D'] > 0) & 
                                            (filtered_combined['dryintercept'] > 0)].copy()

# Recalculate mass for each point using the integrated mass equation (apply row-wise)
filtered_combined_clean['Mass'] = filtered_combined_clean.apply(
    lambda row: calculate_mass(row['dryintercept'], row['D']), axis=1)

# Debugging: Check minimum and maximum mass values
print(f"Min mass: {filtered_combined_clean['Mass'].min()}, Max mass: {filtered_combined_clean['Mass'].max()}")

# Create data-based grids to better match the distribution of the points
xgrid = np.logspace(np.log10(filtered_combined_clean['D'].min()), 
                    np.log10(filtered_combined_clean['D'].max()), 100)
ygrid = np.logspace(np.log10(filtered_combined_clean['dryintercept'].min()), 
                    np.log10(filtered_combined_clean['dryintercept'].max()), 100)
D_grid, dryintercept_grid = np.meshgrid(xgrid, ygrid)

# Calculate mass for each point on the grid (use np.vectorize to apply calculate_mass element-wise)
vectorized_mass = np.vectorize(calculate_mass)
mass_grid = vectorized_mass(dryintercept_grid, D_grid)

# Debugging: Check if mass_grid contains meaningful values
print(f"Mass grid: Min = {np.nanmin(mass_grid)}, Max = {np.nanmax(mass_grid)}")

low_levels = np.logspace(np.log10(filtered_combined_clean['Mass'].min()), np.log10(1), 6, endpoint=False)  # 3 contours in low range (up to 10^0)
high_levels = np.logspace(np.log10(1), np.log10(filtered_combined_clean['Mass'].max()), 5)  # More contours in higher range (10^0 to max)

# Ensure contours are strictly increasing by combining low and high levels
mass_levels = np.concatenate([low_levels, high_levels])

print(f"Mass levels: {mass_levels}")

# Plot the scatter plot with Windspeed color map
sc = plt.scatter(filtered_combined_clean['D'], filtered_combined_clean['dryintercept'], 
                 c=filtered_combined_clean['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

# Plot the mass contours using the full integrated mass formula with reduced number of contours
contour_plot = plt.contour(D_grid, dryintercept_grid, mass_grid, levels=mass_levels, 
                           colors='red', linewidths=0.5, alpha=0.75)

# Check if contours were created
if len(contour_plot.allsegs[0]) == 0:
    print("No contours were created. Check your data range or mass grid calculation.")

# Overlay dry concentration contours
# Use actual data range for the grid
N0_values = filtered_combined_clean['dryintercept'].values  # Use dry intercept from the cleaned dataset
D_values = filtered_combined_clean['D'].values  # Use D values from the cleaned dataset

# Create grids that align more closely with the data
N0_grid = np.linspace(min(N0_values), max(N0_values), 200)  # Match grid to actual N0 values
D_grid_conc = np.linspace(min(D_values), max(D_values), 200)  # Match grid to actual D values

# Create empty array for concentration grid
concentration_grid = np.zeros((len(N0_grid), len(D_grid_conc)))

# Calculate concentration for each combination of N0 and D on the grid
for i, N0 in enumerate(N0_grid):
    for j, D in enumerate(D_grid_conc):
        concentration_grid[i, j] = N0 * D  # Calculate C_d

# Overlay the dry concentration contours using this refined grid
concentration_contour = plt.contour(D_grid_conc, N0_grid, concentration_grid, levels=20, colors='blue', linewidths=0.75, alpha=0.6)

# Add color bar for windspeed
plt.colorbar(sc, label='Corrected Windspeed (m/s)')

# Add labels and formatting
plt.xlabel('D', fontsize=14, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.title('Mass and Dry Concentration Contours', fontsize=14, fontweight='bold')
plt.xlim(10**-2, 10**2)
plt.ylim(10**-2, 10**2.5)

# Display the plot
plt.show()
#%%

# Function to calculate mass with lower limit of integration at 2
def calculate_mass(N0, D):
    integrand = lambda d: np.exp(-d / D) * d**3
    mass_integral, _ = quad(integrand, 2, np.inf)
    return N0 * mass_integral

# Filter out rows where D or dryintercept is NaN or <= 0
filtered_combined_clean = filtered_combined[(filtered_combined['D'] > 0) & 
                                            (filtered_combined['dryintercept'] > 0)].copy()

# Recalculate mass for each point using the integrated mass equation (apply row-wise)
filtered_combined_clean['Mass'] = filtered_combined_clean.apply(
    lambda row: calculate_mass(row['dryintercept'], row['D']), axis=1)

# Debugging: Check minimum and maximum mass values
print(f"Min mass: {filtered_combined_clean['Mass'].min()}, Max mass: {filtered_combined_clean['Mass'].max()}")

# Create data-based grids to better match the distribution of the points
xgrid = np.logspace(np.log10(filtered_combined_clean['D'].min()), 
                    np.log10(filtered_combined_clean['D'].max()), 100)
ygrid = np.logspace(np.log10(filtered_combined_clean['dryintercept'].min()), 
                    np.log10(filtered_combined_clean['dryintercept'].max()), 100)
D_grid, dryintercept_grid = np.meshgrid(xgrid, ygrid)

# Calculate mass for each point on the grid (use np.vectorize to apply calculate_mass element-wise)
vectorized_mass = np.vectorize(calculate_mass)
mass_grid = vectorized_mass(dryintercept_grid, D_grid)

# Debugging: Check if mass_grid contains meaningful values
print(f"Mass grid: Min = {np.nanmin(mass_grid)}, Max = {np.nanmax(mass_grid)}")

low_levels = np.logspace(np.log10(filtered_combined_clean['Mass'].min()), np.log10(1), 6, endpoint=False)  # 3 contours in low range (up to 10^0)
high_levels = np.logspace(np.log10(1), np.log10(filtered_combined_clean['Mass'].max()), 5)  # More contours in higher range (10^0 to max)

# Ensure contours are strictly increasing by combining low and high levels
mass_levels = np.concatenate([low_levels, high_levels])

print(f"Mass levels: {mass_levels}")

plt.figure(figsize=(15, 10))
# Plot the scatter plot with Windspeed color map
sc = plt.scatter(filtered_combined_clean['D'], filtered_combined_clean['dryintercept'], 
                 c=filtered_combined_clean['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

# Plot the mass contours using the full integrated mass formula with reduced number of contours
mass_contour_plot = plt.contour(D_grid, dryintercept_grid, mass_grid, levels=mass_levels, 
                           colors='red', linewidths=0.5, alpha=0.75)

# Overlay dry concentration contours (dashed lines)
concentration_contour = plt.contour(D_grid_conc, N0_grid, concentration_grid, 
                                     levels=20, colors='blue', linewidths=0.75, 
                                     alpha=0.6, linestyles='dashed')

# Create legend handles
mass_contour_label = plt.Line2D([0], [0], color='red', lw=2, label='Mass Contours')
concentration_contour_label = plt.Line2D([0], [0], color='blue', linestyle='dashed', lw=2, label='Dry Concentration Contours')

# Add legends
plt.legend(handles=[mass_contour_label, concentration_contour_label], loc='upper left', 
           bbox_to_anchor=(1.2, 1.1), frameon=False)  # Adjusted bbox_to_anchor to move right

# Add color bar for windspeed
cbar = plt.colorbar(sc, label='Corrected Windspeed (m/s)')

# Add labels and formatting
plt.xlabel('D', fontsize=14, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.title('Mass and Dry Concentration Contours', fontsize=14, fontweight='bold')
plt.xlim(10**-2, 10**2)
plt.ylim(10**-2, 10**2.5)

# Display the plot
plt.show()

#%%
# # Extract dry intercept (N0) and D values from the dictionaries
# N0_values = [entry['dry intercept'] for entry in filtered_master_BCB_dryintercept]
# D_values = [entry['D'] for entry in filtered_master_BCB_ddry]

# # Define the mass integrand function
# def mass_integrand(d, D):
#     return np.exp(-d / D) * d**3

# # Define a function to calculate the total mass M
# def calculate_mass(N0, D):
#     # Perform the integration from 0 to infinity
#     mass, error = quad(mass_integrand, 0, np.inf, args=(D,))
#     return N0 * mass

# # Create empty lists to store mass values
# mass_values = []
# for N0, D in zip(N0_values, D_values):
#     mass = calculate_mass(N0, D)
#     mass_values.append(mass)

# # Convert mass_values to a NumPy array for easier handling
# mass_values = np.array(mass_values)

# # Set a threshold to replace non-positive values with NaN (only if they exist)
# mass_values[mass_values <= 0] = np.nan  # Replace non-positive values with NaN

# # Plotting the mass values against D and N0
# plt.figure(figsize=(10, 8))

# # Scatter plot to visualize mass values
# plt.scatter(D_values, N0_values, c=mass_values, cmap='viridis', edgecolor='k', s=100)
# plt.colorbar(label='Mass (M)')
# plt.xlabel('D (e-folding diameter)')
# plt.ylabel('Dry Intercept (N0)')
# plt.title('Scatter Plot of Mass M vs D and N0')
# plt.xscale('log')
# plt.yscale('log')
# plt.show()
#%%
N0_grid = np.linspace(min(N0_values), max(N0_values), 200)  # Create 100 points for N0
D_grid = np.linspace(min(D_values), max(D_values), 200)  # Create 100 points for D

# Create empty array for mass grid
mass_grid = np.zeros((len(N0_grid), len(D_grid)))

# Calculate mass for each combination of N0 and D on the grid
for i, N0 in enumerate(N0_grid):
    for j, D in enumerate(D_grid):
        mass_grid[i, j] = calculate_mass(N0, D)

# Create contour plot with contour lines
plt.figure(figsize=(10, 8))
contour = plt.contour(D_grid, N0_grid, mass_grid, levels=20, cmap='viridis')  # Use plt.contour() for lines
plt.clabel(contour, inline=True, fontsize=8)  # Add labels to the contour lines
plt.colorbar(contour, label='Mass (M)')  # Add a colorbar for reference
plt.xlabel('D (e-folding diameter)')
plt.ylabel('Dry Intercept (N0)')
plt.title('Contour Plot of Mass M vs D and N0')
# plt.xscale('log')
plt.xlim(0, 13)
plt.ylim(0,127)
# plt.ylim(10**0.6, 10**2.4)
# plt.yscale('log')
plt.show()
#%%
##trying thr 0th moment for dry constant concentration contours

# Extract dry intercept (N0) and D values from the dictionaries
N0_values = [entry['dry intercept'] for entry in filtered_master_BCB_dryintercept]
D_values = [entry['D'] for entry in filtered_master_BCB_ddry]

# Create grids for contour plotting
N0_grid = np.linspace(min(N0_values), max(N0_values), 200)  # Create 200 points for N0
D_grid = np.linspace(min(D_values), max(D_values), 200)  # Create 200 points for D

# Create empty array for concentration grid
concentration_grid = np.zeros((len(N0_grid), len(D_grid)))

# Calculate concentration for each combination of N0 and D on the grid
for i, N0 in enumerate(N0_grid):
    for j, D in enumerate(D_grid):
        concentration_grid[i, j] = N0 * D  # Calculate C_d

# Create contour plot with contour lines
plt.figure(figsize=(10, 8))
contour = plt.contour(D_grid, N0_grid, concentration_grid, levels=20, cmap='viridis')  # Use plt.contour() for lines
plt.clabel(contour, inline=True, fontsize=8)  # Add labels to the contour lines
plt.colorbar(contour, label='Dry Concentration')  # Add a colorbar for reference
plt.xlabel('D')
plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
plt.title(' Dry Concentration', fontsize=14, fontweight='bold')
plt.xscale('log')
plt.yscale('log')
# plt.xlim(10**0, 10**3)
# plt.ylim(10**0, 10**3)
plt.show()

#%%
##Attempting constant dry concentration contours

# # Step 1: Function to calculate total concentration (zeroth moment integral)
# def total_concentration(D, dryintercept):
#     # Assuming a simple relationship for demonstration; modify as needed
#     # For example, mass M = N0 * integral { exp(-d/D) d^3 dd }
#     # This is a placeholder; replace with your actual total concentration logic
#     return (dryintercept / D)  # Placeholder calculation

# # Step 2: Create a grid of D and dryintercept values
# d_values = np.linspace(df_combined['D'].min(), df_combined['D'].max(), 100)
# dryintercept_values = np.linspace(df_combined['dryintercept'].min(), df_combined['dryintercept'].max(), 100)
# D_grid, dryintercept_grid = np.meshgrid(d_values, dryintercept_values)

# # Step 3: Calculate total concentration on the grid
# total_concentration_grid = total_concentration(D_grid, dryintercept_grid)

# # Step 4: Plotting
# plt.figure(figsize=(10, 8))

# # Plot existing contours of density
# sns.kdeplot(x=df_combined['D'], y=df_combined['dryintercept'], cmap='viridis', fill=True, thresh=0)

# # Plot contours of total concentration
# contour_levels = np.linspace(total_concentration_grid.min(), total_concentration_grid.max(), 10)  # Adjust the number of levels
# contour = plt.contour(D_grid, dryintercept_grid, total_concentration_grid, levels=contour_levels, colors='red', linewidths=1, linestyles='solid')
# plt.clabel(contour, inline=True, fontsize=10)

# # Labels and title
# plt.xlabel('D (Diameter)')
# plt.ylabel('Dry Intercept')
# plt.title('Contour Plot of Total Concentration and Density Estimate')
# plt.colorbar(label='Density Estimate')

# # Show the plot
# plt.show()
#%%
##D50 mass calculation 

# Assuming you have these lists from your previous calculations
N0_values = [entry['dry intercept'] for entry in filtered_master_BCB_dryintercept]
D_values = [entry['D'] for entry in filtered_master_BCB_ddry]

# Define the mass integrand function as before
def mass_integrand(d, D):
    return np.exp(-d / D) * d**3

# Calculate total mass for different D
total_mass = []
for D in D_values:
    mass, _ = quad(mass_integrand, 0, np.inf, args=(D,))
    total_mass.append(mass)

# Calculate cumulative mass
cumulative_mass = np.cumsum(total_mass)

# Calculate total mass to find the median
total_mass_value = np.sum(total_mass)
median_mass_threshold = total_mass_value / 2

# Find diameter corresponding to 50% cumulative mass
median_diameter = None
for i, cm in enumerate(cumulative_mass):
    if cm >= median_mass_threshold:
        median_diameter = D_values[i]
        break

# Create contour plot with contour lines
plt.figure(figsize=(10, 8))
contour = plt.contour(D_grid, N0_grid, concentration_grid, levels=20, cmap='viridis')  # Use plt.contour() for lines
plt.clabel(contour, inline=True, fontsize=8)  # Add labels to the contour lines

# Add a colorbar for reference
cbar = plt.colorbar(contour)
cbar.set_label('Dry Concentration', fontsize=14, fontweight='bold')  # Set the colorbar label
cbar.ax.tick_params(labelsize=12)  # Set the font size for colorbar ticks

plt.xlabel('D', fontsize=14, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
plt.title('Dry Concentration with Median Diameter', fontsize=14, fontweight='bold')
plt.xscale('log')
plt.yscale('log')
plt.xlim(10**-1, 10**1.5)
plt.ylim(10**1, 10**2.1)

# Add a vertical line for the median diameter
if median_diameter is not None:
    plt.axvline(x=median_diameter, color='red', linestyle='--', label=f'Median Diameter: {median_diameter:.2f}')
    plt.legend()

plt.show()
#%%
##mass contours with median diameter
# # Extract dry intercept (N0) and D values from the dictionaries
# N0_values = [entry['dry intercept'] for entry in filtered_master_BCB_dryintercept]
# D_values = [entry['D'] for entry in filtered_master_BCB_ddry]

# # Create grids for contour plotting
# N0_grid = np.linspace(min(N0_values), max(N0_values), 200)  # Create 200 points for N0
# D_grid = np.linspace(min(D_values), max(D_values), 200)  # Create 200 points for D

# # Create empty array for mass grid
# mass_grid = np.zeros((len(N0_grid), len(D_grid)))

# # Define the mass integrand function as before
# def mass_integrand(d, D):
#     return np.exp(-d / D) * d**3

# # Calculate total mass for each combination of N0 and D on the grid
# for i, N0 in enumerate(N0_grid):
#     for j, D in enumerate(D_grid):
#         mass, _ = quad(mass_integrand, 0, np.inf, args=(D,))
#         mass_grid[i, j] = N0 * mass  # Calculate the total mass based on N0

# # Create cumulative mass for finding median diameter
# total_mass = np.sum(mass_grid, axis=0)  # Total mass for each D
# cumulative_mass = np.cumsum(total_mass)  # Cumulative mass
# total_mass_value = np.sum(total_mass)  # Total mass value
# median_mass_threshold = total_mass_value / 2  # Half the total mass

# # Find diameter corresponding to 50% cumulative mass
# median_diameter = None
# for i, cm in enumerate(cumulative_mass):
#     if cm >= median_mass_threshold:
#         median_diameter = D_grid[i]
#         break

# # Create contour plot for mass
# plt.figure(figsize=(10, 8))
# contour = plt.contour(D_grid, N0_grid, mass_grid, levels=20, cmap='viridis')  # Use plt.contour() for lines
# plt.clabel(contour, inline=True, fontsize=8)  # Add labels to the contour lines

# # Add a colorbar for reference
# plt.colorbar(label='Mass')  # Set the colorbar label

# # Add median diameter line
# if median_diameter is not None:
#     plt.axvline(x=median_diameter, color='red', linestyle='--', label=f'Median Diameter: {median_diameter:.2f}')
#     plt.legend()

# plt.xlabel('D)', fontsize=14, fontweight='bold')
# plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
# plt.title('Mass Contours with Median Diameter (um)', fontsize=14, fontweight='bold')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(10**0, 10**1.2)
# plt.ylim(10**1, 10**2.1)
# plt.show()





























# %%
##We still have to calculate d dry
##There are x amount of bins ranging from 2.5 to 50 aka 12 bins
##ddry=d/grh
##I need to do every bin by grh for every leg 

# Initialize master_min_ddry to store the results
master_BCB_ddry = []

# Iterate over each date in master_min_exponential and corresponding legs in master_min_gRH
for i, (date, legs_exponential) in enumerate(master_BCB_exponential.items()):
    # Ensure that we have a corresponding entry in master_min_gRH
    if i >= len(master_BCB_gRH):
        print(f"No corresponding gRh data for date {date}, skipping...")
        continue
    
    legs_grh = master_BCB_gRH[i]
    
    for j, (leg_exponential, leg_grh) in enumerate(zip(legs_exponential, legs_grh)):
        n0 = leg_exponential['n0']
        D = leg_exponential['D']
        
        gRh_mean = leg_grh['gRh_mean'][0]  # Assuming gRh_mean is a list with one value
        
        # Calculate ddry by dividing bin centers by gRh_mean
        if gRh_mean is not np.nan and gRh_mean != 0:
            ddry_values = [center / gRh_mean for center in bin_center]
        else:
            ddry_values = [np.nan] * len(bin_center)
            print(f"Skipping division for leg {j} on date {date} due to invalid gRh_mean.")
        
        # Store the results in master_min_ddry
        master_BCB_ddry.append({
            'Date': date,
            'Leg_index': j,
            'ddry': ddry_values,
            'n0': n0,
            'D': D,
            'gRh_mean': gRh_mean
        })

# Example of how to access the stored values in master_min_ddry
for entry in master_BCB_ddry:
    print(f"Date: {entry['Date']}, Leg index: {entry['Leg_index']}, ddry: {entry['ddry']}")

#%%
filtered_master_BCB_ddry = []

# Iterate over each date in master_BCB_exponential and corresponding legs in master_BCB_gRH
for i, (date, legs_exponential) in enumerate(master_BCB_exponential.items()):
    # Ensure that we have a corresponding entry in master_min_gRH
    if i >= len(filtered_master_BCB_gRH):
        # print(f"No corresponding gRh data for date {date}, skipping...")
        continue
    
    filtered_legs_grh = filtered_master_BCB_gRH[i]
    
    for j, (leg_exponential, leg_grh) in enumerate(zip(legs_exponential, filtered_legs_grh)):
        n0 = leg_exponential['n0']
        D = leg_exponential['D']
        gRh_mean = leg_grh['gRh_mean'][0]  # Assuming gRh_mean is a list with one value
        BCB_start = leg_grh['BCB_start']  # Extract Min_start for this leg
        BCB_stop = leg_grh['BCB_stop']      # Extract Min_end for this leg
        
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

# %%
# Initialize master_min_ddry to store the results
master_BCB_ddry = []

# Iterate over each date in master_min_exponential and corresponding legs in master_min_gRH
for i, (date, legs_exponential) in enumerate(master_BCB_exponential.items()):
    # Ensure that we have a corresponding entry in master_min_gRH
    if i >= len(master_BCB_gRH):
        print(f"No corresponding gRh data for date {date}, skipping...")
        continue
    
    legs_grh = master_BCB_gRH[i]
    
    for j, (leg_exponential, leg_grh) in enumerate(zip(legs_exponential, legs_grh)):
        n0 = leg_exponential['n0']
        D = leg_exponential['D']
        
        gRh_mean = leg_grh['gRh_mean'][0]  # Assuming gRh_mean is a list with one value
        
        # Print the gRh_mean value for verification
        print(f"Date: {date}, Leg index: {j}, gRh_mean: {gRh_mean}")
        
        # Calculate ddry by dividing bin centers by gRh_mean
        if gRh_mean is not np.nan and gRh_mean != 0:
            ddry_values = [center / gRh_mean for center in bin_center]
            
            # Print some sample calculations to verify correctness
            print(f"Sample bin centers: {bin_center[:5]}")
            print(f"Sample ddry_values: {ddry_values[:5]}")
            
        else:
            ddry_values = [np.nan] * len(bin_center)
            print(f"Skipping division for leg {j} on date {date} due to invalid gRh_mean.")
        
        # Store the results in master_min_ddry
        master_BCB_ddry.append({
            'Date': date,
            'Leg_index': j,
            'ddry': ddry_values,
            'n0': n0,
            'D': D,
            'gRh_mean': gRh_mean
        })

# Example of how to access the stored values in master_min_ddry
for entry in master_BCB_ddry:
    print(f"Date: {entry['Date']}, Leg index: {entry['Leg_index']}, ddry: {entry['ddry'][:5]}")  # Print first 5 ddry values for brevity
#%%
filtered_master_BCB_ddry = []

# Iterate over each date in master_min_exponential and corresponding legs in master_min_gRH
for i, (date, legs_exponential) in enumerate(master_BCB_exponential.items()):
    # Ensure that we have a corresponding entry in master_min_gRH
    if i >= len(filtered_master_BCB_gRH):
        print(f"No corresponding gRh data for date {date}, skipping...")
        continue
    
    filtered_legs_grh = filtered_master_BCB_gRH[i]
    
    for j, (leg_exponential, leg_grh) in enumerate(zip(legs_exponential, filtered_legs_grh)):
        n0 = leg_exponential['n0']
        D = leg_exponential['D']
        
        gRh_mean = leg_grh['gRh_mean'][0]  # Assuming gRh_mean is a list with one value
        
        # Extract Min_start and Min_end for this leg
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
        
        # Store the results in filtered_master_min_ddry
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
# %%
# Initialize master_min_ddry to store the results
master_BCB_ddry = []

# Example initialization of bin_center (ensure this matches the original size distribution)
bin_center = [2.25, 2.75, 3.25, 3.75, 4.5, 5.75, 6.85, 7.55, 9.05, 11.4, 13.8, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5]

# Print bin_center length and content for verification
print(f"Full length of bin_center: {len(bin_center)}")
print(f"Full bin_center values: {bin_center}")

# Iterate over each date in master_min_exponential and corresponding legs in master_min_gRH
for i, (date, legs_exponential) in enumerate(master_BCB_exponential.items()):
    # Ensure that we have a corresponding entry in master_min_gRH
    if i >= len(master_BCB_gRH):
        print(f"No corresponding gRh data for date {date}, skipping...")
        continue
    
    legs_grh = master_BCB_gRH[i]
    
    for j, (leg_exponential, leg_grh) in enumerate(zip(legs_exponential, legs_grh)):
        n0 = leg_exponential['n0']
        D = leg_exponential['D']
        
        gRh_mean = leg_grh['gRh_mean'][0]  # Assuming gRh_mean is a list with one value
        
        # Print the gRh_mean value for verification
        print(f"Date: {date}, Leg index: {j}, gRh_mean: {gRh_mean}")
        
        # Print the current leg's data
        print(f"Processing leg {j} for date {date}")
        print(f"Length of bin_center: {len(bin_center)}")
        print(f"Bin centers for date {date}, leg {j}: {bin_center}")
        
        # Calculate ddry by dividing bin centers by gRh_mean
        if gRh_mean is not np.nan and gRh_mean != 0:
            ddry_values = [center / gRh_mean for center in bin_center]
            
            # Print some sample calculations to verify correctness
            print(f"Sample bin centers: {bin_center[:5]}")
            print(f"Calculated ddry_values: {ddry_values[:5]}")
            print(f"Full ddry_values: {ddry_values}")
            
        else:
            ddry_values = [np.nan] * len(bin_center)
            print(f"Skipping division for leg {j} on date {date} due to invalid gRh_mean.")
        
        # Store the results in master_min_ddry
        master_BCB_ddry.append({
            'Date': date,
            'Leg_index': j,
            'ddry': ddry_values,
            'n0': n0,
            'D': D,
            'gRh_mean': gRh_mean
        })

# Example of how to access the stored values in master_min_ddry
for entry in master_BCB_ddry:
    print(f"Date: {entry['Date']}, Leg index: {entry['Leg_index']}, ddry length: {len(entry['ddry'])}, first 5 ddry values: {entry['ddry'][:5]}")
#%%
filtered_master_BCB_ddry = []

# Example initialization of bin_center (ensure this matches the original size distribution)
bin_center = [2.25, 2.75, 3.25, 3.75, 4.5, 5.75, 6.85, 7.55, 9.05, 11.4, 13.8, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5]

# Print bin_center length and content for verification
print(f"Full length of bin_center: {len(bin_center)}")
print(f"Full bin_center values: {bin_center}")

# Iterate over each date in master_min_exponential and corresponding legs in master_min_gRH
for i, (date, legs_exponential) in enumerate(master_BCB_exponential.items()):
    # Ensure that we have a corresponding entry in master_min_gRH
    if i >= len(filtered_master_BCB_gRH):
        print(f"No corresponding gRh data for date {date}, skipping...")
        continue
    
    filtered_legs_grh = filtered_master_BCB_gRH[i]
    
    for j, (leg_exponential, filtered_leg_grh) in enumerate(zip(legs_exponential, filtered_legs_grh)):
        n0 = leg_exponential['n0']
        D = leg_exponential['D']
        
        gRh_mean = filtered_leg_grh['gRh_mean'][0]  # Assuming gRh_mean is a list with one value
        
        # Extract Min_start and Min_end for this leg
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
# Loop through each entry in master_min_ddry
for entry in master_BCB_ddry:
    date = entry['Date']
    leg_index = entry['Leg_index']
    ddry_values = entry['ddry']

    # Original bin centers
    original_bin_centers = np.array(bin_center)
    
    # Plot original size distribution
    plt.plot(original_bin_centers, [0] * len(original_bin_centers), 'o-', color='blue', label='Original Bin Centers')

    # Plot new size distribution using ddry values
    plt.plot(ddry_values, [0] * len(ddry_values), 'o-', color='purple', label='ddry Values')

    # Add labels, title, and legend for each leg
    plt.xlabel('Bin centers diameter (um)')
    plt.ylabel('Droplet concentration (/cm^3/um)')
    plt.title(f'Size Distribution Comparison\nDate: {date}, Leg index: {leg_index}')
    plt.legend()
    
    # Show the plot
    plt.show()

    # Print comparison of original and ddry values
    print(f"Date: {date}, Leg index: {leg_index}")
    print(f"Original bin centers: {original_bin_centers}")
    print(f"ddry values: {ddry_values}\n")
#%%
#Filtered 
# Loop through each entry in master_min_ddry
for entry in filtered_master_BCB_ddry:
    date = entry['Date']
    leg_index = entry['Leg_index']
    filtered_ddry_values = entry['filtered_ddry']

    # Original bin centers
    original_bin_centers = np.array(bin_center)
    
    # Plot original size distribution
    plt.plot(original_bin_centers, [0] * len(original_bin_centers), 'o-', color='blue', label='Original Bin Centers')

    # Plot new size distribution using ddry values
    plt.plot(filtered_ddry_values, [0] * len(filtered_ddry_values), 'o-', color='purple', label='filtered_ddry values')

    # Add labels, title, and legend for each leg
    plt.xlabel('Bin centers diameter (um)')
    plt.ylabel('Droplet concentration (/cm^3/um)')
    plt.title(f'Size Distribution Comparison\nDate: {date}, Leg index: {leg_index}')
    plt.legend()
    
    # Show the plot
    plt.show()

    # Print comparison of original and ddry values
    print(f"Date: {date}, Leg index: {leg_index}")
    print(f"Original bin centers: {original_bin_centers}")
    print(f"filtered_ddry values: {filtered_ddry_values}\n")
#%%
##Begin by comparing Ntd with ambient concentrations

##We first need to get our ambient concentrations which is Nt=No*D*e^(-dmin/D)

master_BCB_NT_dict = {}
if isinstance(master_BCB_gRH[0], list):
    master_BCB_gRH = [item for sublist in master_BCB_gRH for item in sublist]

dmin = 2.5  
for entry in master_BCB_gRH:
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_end = entry['BCB_end']
    gRh_mean = entry['gRh_mean'][0]  # Assuming gRh_mean is a list with one value
    Rh_mean = entry['Rh_mean'][0]  # Assuming Rh_mean is a list with one value
    if Rh_mean < 0:
        continue
    key = (date, BCB_start, BCB_end)
    if date in master_BCB_exponential:
        exp_params_list = master_BCB_exponential[date]
        for exp_params in exp_params_list:
            D = exp_params['D']
            n0 = exp_params['n0']
            NT = n0* D * np.exp(-dmin / D)
            
            # Store the result in the dictionary
            master_BCB_NT_dict[key] = {
                'Date': date,
                'BCB_start': BCB_start,
                'BCB_end': BCB_end,
                'Rh_mean': entry['Rh_mean'],
                'gRh_mean': entry['gRh_mean'],
                'NT': NT
            }

master_BCB_NT = list(master_BCB_NT_dict.values())
#%%
#Filtered 

filtered_master_BCB_NT_dict = {}

if isinstance(filtered_master_BCB_gRH[0], list):
    filtered_master_BCB_gRH = [item for sublist in filtered_master_BCB_gRH for item in sublist]

unique_keys = set()

# Flatten master_min_exponential
flattened_exponential = []
for exp_list in master_BCB_exponential.values():
    flattened_exponential.extend(exp_list)

# Create a dictionary for quick lookup
exponential_dict = {(exp['Date'], exp['BCB_start'], exp['BCB_stop']): exp for exp in flattened_exponential}
dmin = 2.5  


for entry in filtered_master_BCB_gRH:
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    gRh_mean = entry['gRh_mean'][0]  # Assuming gRh_mean is a list with one value
    Rh_mean = entry['Rh_mean'][0]  # Assuming Rh_mean is a list with one value
    if Rh_mean < 0:
        continue
    key = (date, BCB_start, BCB_stop)
    if date in master_BCB_exponential:
        exp_params_list = master_BCB_exponential[date]
        for exp_params in exp_params_list:
            D = exp_params['D']
            n0 = exp_params['n0']
            NT = n0* D * np.exp(-dmin / D)
            
            # Store the result in the dictionary
            filtered_master_BCB_NT_dict[key] = {
                'Date': date,
                'BCB_start': BCB_start,
                'BCB_stop': BCB_stop,
                'Rh_mean': entry['Rh_mean'],
                'gRh_mean': entry['gRh_mean'],
                'filtered_NT': NT
            }

filtered_master_BCB_NT = list(filtered_master_BCB_NT_dict.values())
print(f"Length of filtered_master_BCB_NT: {len(filtered_master_BCB_NT)}")
#%%
NT_values = [entry['NT'] for entry in master_BCB_NT]  
Ntd_values = [entry['Ntd'] for entry in master_BCB_ntd]
if len(NT_values) != len(Ntd_values):
    raise ValueError(f"The lengths of NT_values ({len(NT_values)}) and Ntd_values ({len(Ntd_values)}) do not match!")
plt.figure(figsize=(10, 6))
plt.scatter(NT_values, Ntd_values, color='blue', label='NT vs. Ntd')
plt.xlabel('Ambient Concentration', fontsize=12, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.ylabel('Total droplet concentration \n dry droplets larger than \n 2um diam', fontsize=12, fontweight='bold')
plt.title('Below Cloud Base January-June 2022 ', fontsize=14, fontweight='bold')
plt.show()
#%%
filtered_NT_values = [entry['filtered_NT'] for entry in filtered_master_BCB_NT]  
filtered_Ntd_values = [entry['Ntd'] for entry in filtered_master_BCB_ntd]
if len(filtered_NT_values) != len(filtered_Ntd_values):
    raise ValueError(f"The lengths of filtered_NT_values ({len(filtered_NT_values)}) and Ntd_values ({len(filtered_Ntd_values)}) do not match!")
plt.figure(figsize=(10, 6))
plt.scatter(filtered_NT_values, filtered_Ntd_values, color='blue')
plt.xlabel('Ambient Concentration', fontsize=12, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.ylabel('Total droplet concentration \n dry droplets larger than \n 2um diam', fontsize=12, fontweight='bold')
plt.title('Below Cloud Base January-June 2022 w/ clear sky restrictions ', fontsize=14, fontweight='bold')
plt.show()
#%%
#Comparing Ntdnt with ambient concentrations

NT_values = [entry['NT'] for entry in master_BCB_NT]  
NtdNt_values = [entry['NtdNt'] for entry in master_BCB_NtdNt]
if len(NT_values) != len(NtdNt_values):
    raise ValueError(f"The lengths of NT_values ({len(NT_values)}) and NtdNt_values ({len(NtdNt_values)}) do not match!")
plt.figure(figsize=(10, 6))
plt.scatter(NT_values, NtdNt_values, color='blue', label='NT vs. NtdNt')
plt.xlabel('Ambient Concentration', fontsize=12, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.ylabel('Ratio of total droplet concentration \n dry droplets larger than \n 2um diameter \n to ambient concentration', fontsize=12, fontweight='bold')
plt.title('Below Cloud Base January-June 2022 ', fontsize=14, fontweight='bold')
plt.show()


Ntd_values = [entry['Ntd'] for entry in master_BCB_ntd]

plt.figure(figsize=(10, 6))
plt.scatter(range(len(Ntd_values)), Ntd_values, color='blue', label='Ntd Distribution')
plt.xlabel('Chronological legs ', fontsize=12, fontweight='bold')
plt.ylabel('Distribution of total droplet concentration \nof dry droplets with diameter>2um', fontsize=12, fontweight='bold')
plt.title('Below Cloud Base January-June 2022', fontsize=14, fontweight='bold')
plt.yscale('log')  # Optional: use log scale for y-axis if needed
plt.show()

NtdNt_values = [entry['NtdNt'] for entry in master_BCB_NtdNt]

plt.figure(figsize=(10, 6))
plt.scatter(range(len(NtdNt_values)), NtdNt_values, color='blue', label='NtdNt Distribution')
plt.xlabel('Chronological legs', fontsize=12, fontweight='bold')
plt.ylabel('Distribution of ratio of total droplet concentration of dry droplets \n with diameter larger than 2um \n to ambient concentration', fontsize=12, fontweight='bold')
plt.title('Below Cloud Base January-June 2022', fontsize=14, fontweight='bold')
plt.yscale('log')  # Optional: use log scale for y-axis if needed
plt.show()
#%%
#Filtered 
filtered_NT_values = [entry['filtered_NT'] for entry in filtered_master_BCB_NT]  
filtered_NtdNt_values = [entry['NtdNt'] for entry in filtered_master_BCB_NtdNt]
if len(filtered_NT_values) != len(filtered_NtdNt_values):
    raise ValueError(f"The lengths of filtered_NT_values ({len(filtered_NT_values)}) and NtdNt_values ({len(filtered_NtdNt_values)}) do not match!")
plt.figure(figsize=(10, 6))
plt.scatter(filtered_NT_values, filtered_NtdNt_values, color='blue', label='NT vs. NtdNt')
plt.xlabel('Ambient Concentration', fontsize=12, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.ylabel('Ratio of total droplet concentration \n dry droplets larger than \n 2um diameter \n to ambient concentration', fontsize=12, fontweight='bold')
plt.title('Below cloud base January-June 2022 w/clear sky restrictions', fontsize=14, fontweight='bold')
plt.show()


filtered_Ntd_values = [entry['Ntd'] for entry in filtered_master_BCB_ntd]

plt.figure(figsize=(10, 6))
plt.scatter(range(len(filtered_Ntd_values)), filtered_Ntd_values, color='blue', label='Ntd Distribution')
plt.xlabel('Chronological legs ', fontsize=12, fontweight='bold')
plt.ylabel('Distribution of total droplet concentration \nof dry droplets with diameter>2um', fontsize=12, fontweight='bold')
plt.title('Below cloud base January-June 2022 w/clear sky restrictions', fontsize=14, fontweight='bold')
plt.yscale('log')  # Optional: use log scale for y-axis if needed
plt.show()
#%%
filtered_NtdNt_values = [entry['NtdNt'] for entry in filtered_master_BCB_NtdNt]

plt.figure(figsize=(10, 6))
plt.scatter(range(len(filtered_NtdNt_values)), filtered_NtdNt_values, color='blue', label='NtdNt Distribution')
plt.xlabel('Chronological legs', fontsize=12, fontweight='bold')
plt.ylabel('Distribution of ratio of total droplet concentration of dry droplets \n with diameter larger than 2um \n to ambient concentration', fontsize=12, fontweight='bold')
plt.title('Below cloud base January-June 2022 w/clear sky restrictions', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.ylim(10**-2, 10**0)  # Optional: use log scale for y-axis if needed
plt.show()
#%%
##plotting gRH to see distribution
grh_values = [entry['gRh_mean'] for entry in master_BCB_gRH]  
plt.figure(figsize=(10, 6))
plt.plot(grh_values, color='blue')
plt.ylabel('Humidity growth factor', fontsize=12, fontweight='bold')
# plt.xlabel('Ratio of total droplet concentration \n dry droplets larger than \n 2.5um diameter \n to ambient concentration', fontsize=12, fontweight='bold')
plt.title('Below Cloud Base January-June 2022 ', fontsize=14, fontweight='bold')
# plt.yscale('log')
plt.show()
#%%
#filtered
filtered_grh_values = [entry['gRh_mean'] for entry in filtered_master_BCB_gRH]  
plt.figure(figsize=(10, 6))
plt.plot(filtered_grh_values, color='blue')
plt.ylabel('Humidity growth factor', fontsize=12, fontweight='bold')
# plt.xlabel('Ratio of total droplet concentration \n dry droplets larger than \n 2.5um diameter \n to ambient concentration', fontsize=12, fontweight='bold')
plt.title('Below cloud base  January-June 2022 w/clear sky restrictions', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.show()
# %%
#See how Nt and Ntd vary with windspeed
Z0 = 0.03  # meters (typical value for open terrain)
Z10 = 10  # target height m

# Function to apply windspeed correction
def correct_windspeed(windspeed, altitude):
    return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))

# Adjusted approach since master_min_ntd and master_min_NT are lists
combined_data = {
    'Date': [],
    'BCB_start': [],
    'BCB_end': [],
    'Ntd': [],
    'NT': [],
    'Windspeed': []
}

# Validate and combine the data
for i, flight in enumerate(master_BCB):
    for j, wind_alt in enumerate(flight):
        try:
            date = wind_alt['Date']
            wind_mean = wind_alt['Winds_mean'][0]
            alt_mean = wind_alt['Alts_mean'][0]
            BCB_start = wind_alt['BCB_start']
            BCB_end = wind_alt['BCB_end']
            
            # Correct windspeed using the provided formula
            if not np.isnan(alt_mean) and alt_mean > 0:
                corrected_windspeed = correct_windspeed(wind_mean, alt_mean)
            else:
                corrected_windspeed = np.nan

            # Initialize variables to store found data
            Ntd = None
            NT = None

            # Find the corresponding exponential parameters and dry intercept
            for exp_params in master_BCB_ntd:
                if exp_params['Date'] == date and exp_params['BCB_start'] == BCB_start and exp_params['BCB_end'] == BCB_end:
                    Ntd = exp_params['Ntd']
                    break

            for exp_params in master_BCB_NT:
                if exp_params['Date'] == date and exp_params['BCB_start'] == BCB_start and exp_params['BCB_end'] == BCB_end:
                    NT = exp_params['NT']
                    break
            
            # Append data only if both Ntd and NT were found
            if Ntd is not None and NT is not None:
                print(f"Appending data for Date: {date}, BCB_start: {BCB_start}, BCB_end: {BCB_end}")
                combined_data['Date'].append(date)
                combined_data['BCB_start'].append(BCB_start)
                combined_data['BCB_end'].append(BCB_end)
                combined_data['Ntd'].append(Ntd)
                combined_data['NT'].append(NT)
                combined_data['Windspeed'].append(corrected_windspeed)

        except IndexError as e:
            print(f"Index error at i={i}, j={j}: {e}")
            continue

# Convert to DataFrame
df_combined = pd.DataFrame(combined_data)

# Check the data for any anomalies
print(df_combined.describe())

# Separate data based on NaN Windspeed
df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]

plt.figure(figsize=(10, 8))

# Plot data with valid Windspeed values
sc = plt.scatter(df_with_windspeed['NT'], df_with_windspeed['Ntd'], 
                 c=df_with_windspeed['Windspeed'], cmap='viridis', s=100, label='Windspeed')

# Plot data with NaN Windspeed values in black
plt.scatter(df_nan_windspeed['NT'], df_nan_windspeed['Ntd'], 
            color='black', s=100, label='Windspeed NaN')

# Add color bar
plt.colorbar(sc, label='Corrected Windspeed (m/s)')

# Add labels and title
plt.xlabel('Ambient Concentration', fontsize=14, fontweight='bold')
plt.ylabel('Total droplet concentration \n dry droplets larger than \n 2um diameter', fontsize=14, fontweight='bold')
plt.title('Below Cloud Base January-June 2022', fontsize=14, fontweight='bold')
plt.yscale('log')
# plt.ylim(10**-1, 10**2)
# plt.xlim(10**-2, 10**1)
plt.xscale('log')
plt.show()
#%%
#Filtered 
#See how Nt and Ntd vary with windspeed
Z0 = 0.03  # meters (typical value for open terrain)
Z10 = 10  # target height m

# Function to apply windspeed correction
def correct_windspeed(windspeed, altitude):
    return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))

# Adjusted approach since master_min_ntd and master_min_NT are lists
combined_data = {
    'Date': [],
    'BCB_start': [],
    'BCB_stop': [],
    'Ntd': [],
    'filtered_NT': [],
    'Windspeed': []
}

# Validate and combine the data
for i, flight in enumerate(master_BCB):
    for j, wind_alt in enumerate(flight):
        try:
            date = wind_alt['Date']
            wind_mean = wind_alt['Winds_mean'][0]
            alt_mean = wind_alt['Alts_mean'][0]
            BCB_start = wind_alt['BCB_start']
            BCB_stop = wind_alt['BCB_end']
            
            # Correct windspeed using the provided formula
            if not np.isnan(alt_mean) and alt_mean > 0:
                corrected_windspeed = correct_windspeed(wind_mean, alt_mean)
            else:
                corrected_windspeed = np.nan

            # Initialize variables to store found data
            Ntd = None
            NT = None

            # Find the corresponding exponential parameters and dry intercept
            for exp_params in filtered_master_BCB_ntd:
                if exp_params['Date'] == date and exp_params['BCB_start'] == BCB_start and exp_params['BCB_stop'] == BCB_stop:
                    Ntd = exp_params['Ntd']
                    break

            for exp_params in filtered_master_BCB_NT:
                if exp_params['Date'] == date and exp_params['BCB_start'] == BCB_start and exp_params['BCB_stop'] == BCB_stop:
                    NT = exp_params['filtered_NT']
                    break
            
            # Append data only if both Ntd and NT were found
            if Ntd is not None and NT is not None:
                print(f"Appending data for Date: {date}, BCB_start: {BCB_start}, BCB_stop: {BCB_stop}")
                combined_data['Date'].append(date)
                combined_data['BCB_start'].append(BCB_start)
                combined_data['BCB_stop'].append(BCB_stop)
                combined_data['Ntd'].append(Ntd)
                combined_data['filtered_NT'].append(NT)
                combined_data['Windspeed'].append(corrected_windspeed)

        except IndexError as e:
            print(f"Index error at i={i}, j={j}: {e}")
            continue

# Convert to DataFrame
df_combined = pd.DataFrame(combined_data)

# Check the data for any anomalies
print(df_combined.describe())

# Separate data based on NaN Windspeed
df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]

plt.figure(figsize=(10, 8))

# Plot data with valid Windspeed values
sc = plt.scatter(df_with_windspeed['filtered_NT'], df_with_windspeed['Ntd'], 
                 c=df_with_windspeed['Windspeed'], cmap='viridis', s=100, label='Windspeed')

# Plot data with NaN Windspeed values in black
plt.scatter(df_nan_windspeed['filtered_NT'], df_nan_windspeed['Ntd'], 
            color='black', s=100, label='Windspeed NaN')

# Add color bar
plt.colorbar(sc, label='Corrected Windspeed (m/s)')

# Add labels and title
plt.xlabel('Ambient Concentration', fontsize=14, fontweight='bold')
plt.ylabel('Total droplet concentration \n dry droplets larger than \n 2um diameter', fontsize=14, fontweight='bold')
plt.title('Below cloud base January-June 2022 w/ clear sky restrictions', fontsize=14, fontweight='bold')
plt.yscale('log')
# plt.ylim(10**-1, 10**2)
# plt.xlim(10**-2, 10**1)
plt.xscale('log')
plt.show()

#%%
#See how NT and Ntndt vary with windspeed 

#See how Nt and Ntd vary with windspeed
Z0 = 0.03  # meters (typical value for open terrain)
Z10 = 10  # target height m

# Function to apply windspeed correction
def correct_windspeed(windspeed, altitude):
    return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))

# Adjusted approach since master_min_ntd and master_min_NT are lists
combined_data = {
    'Date': [],
    'BCB_start': [],
    'BCB_end': [],
    'NtdNt': [],
    'NT': [],
    'Windspeed': []
}

# Validate and combine the data
for i, flight in enumerate(master_BCB):
    for j, wind_alt in enumerate(flight):
        try:
            date = wind_alt['Date']
            wind_mean = wind_alt['Winds_mean'][0]
            alt_mean = wind_alt['Alts_mean'][0]
            BCB_start = wind_alt['BCB_start']
            BCB_end = wind_alt['BCB_end']
            
            # Correct windspeed using the provided formula
            if not np.isnan(alt_mean) and alt_mean > 0:
                corrected_windspeed = correct_windspeed(wind_mean, alt_mean)
            else:
                corrected_windspeed = np.nan

            # Initialize variables to store found data
            NtdNt = None
            NT = None

            # Find the corresponding exponential parameters and dry intercept
            for exp_params in master_BCB_NtdNt:
                if exp_params['Date'] == date and exp_params['BCB_start'] == BCB_start and exp_params['BCB_end'] == BCB_end:
                    NtdNt = exp_params['NtdNt']
                    break

            for exp_params in master_BCB_NT:
                if exp_params['Date'] == date and exp_params['BCB_start'] == BCB_start and exp_params['BCB_end'] == BCB_end:
                    NT = exp_params['NT']
                    break
            
            # Append data only if both Ntd and NT were found
            if NtdNt is not None and NT is not None:
                print(f"Appending data for Date: {date}, BCB_start: {BCB_start}, BCB_end: {BCB_end}")
                combined_data['Date'].append(date)
                combined_data['BCB_start'].append(BCB_start)
                combined_data['BCB_end'].append(BCB_end)
                combined_data['NtdNt'].append(NtdNt)
                combined_data['NT'].append(NT)
                combined_data['Windspeed'].append(corrected_windspeed)

        except IndexError as e:
            print(f"Index error at i={i}, j={j}: {e}")
            continue

# Convert to DataFrame
df_combined = pd.DataFrame(combined_data)

# Check the data for any anomalies
print(df_combined.describe())

# Separate data based on NaN Windspeed
df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]

plt.figure(figsize=(10, 8))

# Plot data with valid Windspeed values
sc = plt.scatter(df_with_windspeed['NT'], df_with_windspeed['NtdNt'], 
                 c=df_with_windspeed['Windspeed'], cmap='viridis', s=100, label='Windspeed')

# Plot data with NaN Windspeed values in black
plt.scatter(df_nan_windspeed['NT'], df_nan_windspeed['NtdNt'], 
            color='black', s=100, label='Windspeed NaN')

# Add color bar
plt.colorbar(sc, label='Corrected Windspeed (m/s)')

# Add labels and title
plt.xlabel('Ambient Concentration', fontsize=14, fontweight='bold')
plt.ylabel('Ratio of total droplet concentration \n dry droplets larger than \n 2um diameter to ambient concentration', fontsize=14, fontweight='bold')
plt.title('Below Cloud Base January-June 2022', fontsize=14, fontweight='bold')
plt.yscale('log')
# plt.ylim(10**-1, 10**2)
# plt.xlim(10**-2, 10**1)
plt.xscale('log')
plt.show()
#%%
#Filtered 
#See how NT and Ntndt vary with windspeed 

#See how Nt and Ntd vary with windspeed
Z0 = 0.03  # meters (typical value for open terrain)
Z10 = 10  # target height m

# Function to apply windspeed correction
def correct_windspeed(windspeed, altitude):
    return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))

# Adjusted approach since master_min_ntd and master_min_NT are lists
combined_data = {
    'Date': [],
    'BCB_start': [],
    'BCB_stop': [],
    'NtdNt': [],
    'filtered_NT': [],
    'Windspeed': []
}

# Validate and combine the data
for i, flight in enumerate(master_BCB):
    for j, wind_alt in enumerate(flight):
        try:
            date = wind_alt['Date']
            wind_mean = wind_alt['Winds_mean'][0]
            alt_mean = wind_alt['Alts_mean'][0]
            BCB_start = wind_alt['BCB_start']
            BCB_stop = wind_alt['BCB_end']
            
            # Correct windspeed using the provided formula
            if not np.isnan(alt_mean) and alt_mean > 0:
                corrected_windspeed = correct_windspeed(wind_mean, alt_mean)
            else:
                corrected_windspeed = np.nan

            # Initialize variables to store found data
            NtdNt = None
            NT = None

            # Find the corresponding exponential parameters and dry intercept
            for exp_params in filtered_master_BCB_NtdNt:
                if exp_params['Date'] == date and exp_params['BCB_start'] == BCB_start and exp_params['BCB_stop'] == BCB_stop:
                    NtdNt = exp_params['NtdNt']
                    break

            for exp_params in filtered_master_BCB_NT:
                if exp_params['Date'] == date and exp_params['BCB_start'] == BCB_start and exp_params['BCB_stop'] == BCB_stop:
                    NT = exp_params['filtered_NT']
                    break
            
            # Append data only if both Ntd and NT were found
            if NtdNt is not None and NT is not None:
                print(f"Appending data for Date: {date}, BCB_start: {BCB_start}, BCB_stop: {BCB_stop}")
                combined_data['Date'].append(date)
                combined_data['BCB_start'].append(BCB_start)
                combined_data['BCB_stop'].append(BCB_stop)
                combined_data['NtdNt'].append(NtdNt)
                combined_data['filtered_NT'].append(NT)
                combined_data['Windspeed'].append(corrected_windspeed)

        except IndexError as e:
            print(f"Index error at i={i}, j={j}: {e}")
            continue

# Convert to DataFrame
df_combined = pd.DataFrame(combined_data)

# Check the data for any anomalies
print(df_combined.describe())

# Separate data based on NaN Windspeed
df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]

plt.figure(figsize=(10, 8))

# Plot data with valid Windspeed values
sc = plt.scatter(df_with_windspeed['filtered_NT'], df_with_windspeed['NtdNt'], 
                 c=df_with_windspeed['Windspeed'], cmap='viridis', s=100, label='Windspeed')

# Plot data with NaN Windspeed values in black
plt.scatter(df_nan_windspeed['filtered_NT'], df_nan_windspeed['NtdNt'], 
            color='black', s=100, label='Windspeed NaN')

# Add color bar
plt.colorbar(sc, label='Corrected Windspeed (m/s)')

# Add labels and title
plt.xlabel('Ambient Concentration', fontsize=14, fontweight='bold')
plt.ylabel('Ratio of total droplet concentration \n dry droplets larger than \n 2um diameter to ambient concentration', fontsize=14, fontweight='bold')
plt.title('Below cloud base January-June 2022 w/ clear sky restrictions', fontsize=14, fontweight='bold')
plt.yscale('log')
# plt.ylim(10**-1, 10**2)
# plt.xlim(10**-2, 10**1)
plt.xscale('log')
plt.show()
#%%
##Ntd distribution varies with windspeed


# Constants
Z0 = 0.03  # meters (typical value for open terrain)
Z10 = 10  # target height in meters

# Function to apply windspeed correction
def correct_windspeed(windspeed, altitude):
    return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))

# Adjusted approach since master_min_ntd is a list
combined_data = {
    'Date': [],
    'BCB_start': [],
    'BCB_end': [],
    'Ntd': [],
    'Windspeed': []
}

# Validate and combine the data
for i, flight in enumerate(master_BCB):
    for j, wind_alt in enumerate(flight):
        try:
            date = wind_alt['Date']
            wind_mean = wind_alt['Winds_mean'][0]
            alt_mean = wind_alt['Alts_mean'][0]
            BCB_start = wind_alt['BCB_start']
            BCB_end = wind_alt['BCB_end']
            
            # Correct windspeed using the provided formula
            if not np.isnan(alt_mean) and alt_mean > 0:
                corrected_windspeed = correct_windspeed(wind_mean, alt_mean)
            else:
                corrected_windspeed = np.nan

            # Find the corresponding Ntd value
            Ntd = None
            for exp_params in master_BCB_ntd:
                if exp_params['Date'] == date and exp_params['BCB_start'] == BCB_start and exp_params['BCB_end'] == BCB_end:
                    Ntd = exp_params['Ntd']
                    break
            
            # Append data only if Ntd was found
            if Ntd is not None:
                print(f"Appending data for Date: {date}, BCB_start: {BCB_start}, BCB_end: {BCB_end}")
                combined_data['Date'].append(date)
                combined_data['BCB_start'].append(BCB_start)
                combined_data['BCB_end'].append(BCB_end)
                combined_data['Ntd'].append(Ntd)
                combined_data['Windspeed'].append(corrected_windspeed)

        except IndexError as e:
            print(f"Index error at i={i}, j={j}: {e}")
            continue

# Convert to DataFrame
df_combined = pd.DataFrame(combined_data)

# Check the data for any anomalies
print(df_combined.describe())

# Separate data based on NaN Windspeed
df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]

plt.figure(figsize=(10, 8))

# Plot data with valid Windspeed values
sc = plt.scatter(df_with_windspeed['Windspeed'], df_with_windspeed['Ntd'], 
                 c=df_with_windspeed['Windspeed'], cmap='viridis', s=100, label='Windspeed')

# Plot data with NaN Windspeed values in black
plt.scatter(df_nan_windspeed['Windspeed'], df_nan_windspeed['Ntd'], 
            color='black', s=100, label='Windspeed NaN')

# Add color bar
plt.colorbar(sc, label='Corrected Windspeed (m/s)')

# Add labels and title
plt.xlabel('Windspeed (m/s)', fontsize=14, fontweight='bold')
plt.ylabel('Total Droplet Concentration of dry droplets \n with diameter larger than 2um', fontsize=14, fontweight='bold')
plt.title('Below Cloud Base January - June 2022', fontsize=14, fontweight='bold')
plt.yscale('log')  # Optional: use log scale for y-axis if needed
plt.show()
#%%
##Ntd distribution varies with windspeed

#Filtered 
# Constants
Z0 = 0.03  # meters (typical value for open terrain)
Z10 = 10  # target height in meters

# Function to apply windspeed correction
def correct_windspeed(windspeed, altitude):
    return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))

# Adjusted approach since master_min_ntd is a list
combined_data = {
    'Date': [],
    'BCB_start': [],
    'BCB_stop': [],
    'Ntd': [],
    'Windspeed': []
}

# Validate and combine the data
for i, flight in enumerate(master_BCB):
    for j, wind_alt in enumerate(flight):
        try:
            date = wind_alt['Date']
            wind_mean = wind_alt['Winds_mean'][0]
            alt_mean = wind_alt['Alts_mean'][0]
            BCB_start = wind_alt['BCB_start']
            BCB_stop = wind_alt['BCB_end']
            
            # Correct windspeed using the provided formula
            if not np.isnan(alt_mean) and alt_mean > 0:
                corrected_windspeed = correct_windspeed(wind_mean, alt_mean)
            else:
                corrected_windspeed = np.nan

            # Find the corresponding Ntd value
            Ntd = None
            for exp_params in filtered_master_BCB_ntd:
                if exp_params['Date'] == date and exp_params['BCB_start'] == BCB_start and exp_params['BCB_stop'] == BCB_stop:
                    Ntd = exp_params['Ntd']
                    break
            
            # Append data only if Ntd was found
            if Ntd is not None:
                print(f"Appending data for Date: {date}, BCB_start: {BCB_start}, BCB_stop: {BCB_stop}")
                combined_data['Date'].append(date)
                combined_data['BCB_start'].append(BCB_start)
                combined_data['BCB_stop'].append(BCB_stop)
                combined_data['Ntd'].append(Ntd)
                combined_data['Windspeed'].append(corrected_windspeed)

        except IndexError as e:
            print(f"Index error at i={i}, j={j}: {e}")
            continue

# Convert to DataFrame
df_combined = pd.DataFrame(combined_data)

# Check the data for any anomalies
print(df_combined.describe())

# Separate data based on NaN Windspeed
df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]

plt.figure(figsize=(10, 8))

# Plot data with valid Windspeed values
sc = plt.scatter(df_with_windspeed['Windspeed'], df_with_windspeed['Ntd'], 
                 c=df_with_windspeed['Windspeed'], cmap='viridis', s=100, label='Windspeed')

# Plot data with NaN Windspeed values in black
plt.scatter(df_nan_windspeed['Windspeed'], df_nan_windspeed['Ntd'], 
            color='black', s=100, label='Windspeed NaN')

# Add color bar
plt.colorbar(sc, label='Corrected Windspeed (m/s)')

# Add labels and title
plt.xlabel('Windspeed (m/s)', fontsize=14, fontweight='bold')
plt.ylabel('Total Droplet Concentration of droplets \n with ddry larger than 2um d (Ntd)', fontsize=14, fontweight='bold')
plt.title('Below cloud base January - June 2022 w/ clear sky restrictions', fontsize=14, fontweight='bold')
plt.yscale('log')  # Optional: use log scale for y-axis if needed
plt.show()
# %%
## Comparing ambient concentration to our concentration
##over 2um diameter to see how variable 

##So we need to compare ambient concentration to Ntd and ambient
##concentration to Ntdnt

##finally compare Ntdnt to Ntd
#%%

##Begin by comparing Ntd with ambient concentrations

##We first need to get our ambient concentrations which is Nt=No*D*e^(-dmin/D)

master_BCB_NT_dict = {}
if isinstance(master_BCB_gRH[0], list):
    master_BCB_gRH = [item for sublist in master_BCB_gRH for item in sublist]

dmin = 2.5  
for entry in master_BCB_gRH:
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_end = entry['BCB_end']
    gRh_mean = entry['gRh_mean'][0]  # Assuming gRh_mean is a list with one value
    Rh_mean = entry['Rh_mean'][0]  # Assuming Rh_mean is a list with one value
    if Rh_mean < 0:
        continue
    key = (date, BCB_start, BCB_end)
    if date in master_BCB_exponential:
        exp_params_list = master_BCB_exponential[date]
        for exp_params in exp_params_list:
            D = exp_params['D']
            n0 = exp_params['n0']
            NT = n0* D * np.exp(-dmin / D)
            
            # Store the result in the dictionary
            master_BCB_NT_dict[key] = {
                'Date': date,
                'Min_start': BCB_start,
                'Min_end': BCB_end,
                'Rh_mean': entry['Rh_mean'],
                'gRh_mean': entry['gRh_mean'],
                'NT': NT
            }

master_min_NT = list(master_BCB_NT_dict.values())
#%%

NT_values = [entry['NT'] for entry in master_BCB_NT]  
Ntd_values = [entry['Ntd'] for entry in master_BCB_ntd]
if len(NT_values) != len(Ntd_values):
    raise ValueError(f"The lengths of NT_values ({len(NT_values)}) and Ntd_values ({len(Ntd_values)}) do not match!")
plt.figure(figsize=(10, 6))
plt.scatter(NT_values, Ntd_values, color='blue', label='NT vs. Ntd')
plt.xlabel('Ambient Concentration ()', fontsize=12, fontweight='bold')
plt.ylabel('Total droplet concentration \n dry droplets larger than \n 2.5um diameter', fontsize=12, fontweight='bold')
plt.title('Below Cloud Base January-June 2022 ', fontsize=14, fontweight='bold')
plt.show()

#%%
#Comparing Ntdnt with ambient concentrations

NT_values = [entry['NT'] for entry in master_BCB_NT]  
Ntd_values = [entry['NtdNt'] for entry in master_BCB_NtdNt]
if len(NT_values) != len(Ntd_values):
    raise ValueError(f"The lengths of NT_values ({len(NT_values)}) and Ntd_values ({len(Ntd_values)}) do not match!")
plt.figure(figsize=(10, 6))
plt.scatter(NT_values, Ntd_values, color='blue', label='NT vs. Ntd')
plt.xlabel('Ambient Concentration ()', fontsize=12, fontweight='bold')
plt.ylabel('Ratio of total droplet concentration \n dry droplets larger than \n 2.5um diameter \n to ambient concentration', fontsize=12, fontweight='bold')
plt.title('Below Cloud Base January-June 2022 ', fontsize=14, fontweight='bold')
plt.show()
#%%
##plotting gRH to see distribution
grh_values = [entry['gRh_mean'] for entry in master_BCB_gRH]  
plt.figure(figsize=(10, 6))
plt.plot(grh_values, color='blue')
plt.ylabel('Humidity growth factor ()', fontsize=12, fontweight='bold')
# plt.xlabel('Ratio of total droplet concentration \n dry droplets larger than \n 2.5um diameter \n to ambient concentration', fontsize=12, fontweight='bold')
plt.title('Below Cloud Base January-June 2022 ', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.show()

# %%
#See how Nt and Ntd vary with windspeed
Z0 = 0.03  # meters (typical value for open terrain)
Z10 = 10  # target height m

# Function to apply windspeed correction
def correct_windspeed(windspeed, altitude):
    return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))

# Adjusted approach since master_min_ntd and master_min_NT are lists
combined_data = {
    'Date': [],
    'BCB_start': [],
    'BCB_end': [],
    'Ntd': [],
    'NT': [],
    'Windspeed': []
}

# Validate and combine the data
for i, flight in enumerate(master_BCB):
    for j, wind_alt in enumerate(flight):
        try:
            date = wind_alt['Date']
            wind_mean = wind_alt['Winds_mean'][0]
            alt_mean = wind_alt['Alts_mean'][0]
            BCB_start = wind_alt['Min_start']
            BCB_end = wind_alt['Min_end']
            
            # Correct windspeed using the provided formula
            if not np.isnan(alt_mean) and alt_mean > 0:
                corrected_windspeed = correct_windspeed(wind_mean, alt_mean)
            else:
                corrected_windspeed = np.nan

            # Initialize variables to store found data
            Ntd = None
            NT = None

            # Find the corresponding exponential parameters and dry intercept
            for exp_params in master_BCB:
                if exp_params['Date'] == date and exp_params['BCB_start'] == BCB_start and exp_params['BCB_end'] == BCB_end:
                    Ntd = exp_params['Ntd']
                    break

            for exp_params in master_BCB_NT:
                if exp_params['Date'] == date and exp_params['BCB_start'] == BCB_start and exp_params['BCB_end'] == BCB_end:
                    NT = exp_params['NT']
                    break
            
            # Append data only if both Ntd and NT were found
            if Ntd is not None and NT is not None:
                print(f"Appending data for Date: {date}, BCB_start: {BCB_start}, BCB_end: {BCB_end}")
                combined_data['Date'].append(date)
                combined_data['BCB_start'].append(BCB_start)
                combined_data['BCB_end'].append(BCB_end)
                combined_data['Ntd'].append(Ntd)
                combined_data['NT'].append(NT)
                combined_data['Windspeed'].append(corrected_windspeed)

        except IndexError as e:
            print(f"Index error at i={i}, j={j}: {e}")
            continue

# Convert to DataFrame
df_combined = pd.DataFrame(combined_data)

# Check the data for any anomalies
print(df_combined.describe())

# Separate data based on NaN Windspeed
df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]

plt.figure(figsize=(10, 8))

# Plot data with valid Windspeed values
sc = plt.scatter(df_with_windspeed['NT'], df_with_windspeed['Ntd'], 
                 c=df_with_windspeed['Windspeed'], cmap='viridis', s=100, label='Windspeed')

# Plot data with NaN Windspeed values in black
plt.scatter(df_nan_windspeed['NT'], df_nan_windspeed['Ntd'], 
            color='black', s=100, label='Windspeed NaN')

# Add color bar
plt.colorbar(sc, label='Corrected Windspeed (m/s)')

# Add labels and title
plt.xlabel('Ambient Concentration', fontsize=14, fontweight='bold')
plt.ylabel('Total droplet concentration \n dry droplets larger than \n 2.5um diameter', fontsize=14, fontweight='bold')
plt.title('Below Cloud Base January-June 2022', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.show()
#%%
##Ntd distribution varies with windspeed

#Filtered 
# Constants
Z0 = 0.03  # meters (typical value for open terrain)
Z10 = 10  # target height in meters

# Function to apply windspeed correction
def correct_windspeed(windspeed, altitude):
    return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))

# Adjusted approach since master_min_ntd is a list
combined_data = {
    'Date': [],
    'BCB_start': [],
    'BCB_stop': [],
    'Ntd': [],
    'Windspeed': []
}

# Validate and combine the data
for i, flight in enumerate(master_BCB):
    for j, wind_alt in enumerate(flight):
        try:
            date = wind_alt['Date']
            wind_mean = wind_alt['Winds_mean'][0]
            alt_mean = wind_alt['Alts_mean'][0]
            BCB_start = wind_alt['BCB_start']
            BCB_stop = wind_alt['BCB_end']
            
            # Correct windspeed using the provided formula
            if not np.isnan(alt_mean) and alt_mean > 0:
                corrected_windspeed = correct_windspeed(wind_mean, alt_mean)
            else:
                corrected_windspeed = np.nan

            # Find the corresponding Ntd value
            Ntd = None
            for exp_params in filtered_master_BCB_ntd:
                if exp_params['Date'] == date and exp_params['BCB_start'] == BCB_start and exp_params['BCB_stop'] == BCB_stop:
                    Ntd = exp_params['Ntd']
                    break
            
            # Append data only if Ntd was found
            if Ntd is not None:
                print(f"Appending data for Date: {date}, BCB_start: {BCB_start}, BCB_stop: {BCB_stop}")
                combined_data['Date'].append(date)
                combined_data['BCB_start'].append(BCB_start)
                combined_data['BCB_stop'].append(BCB_stop)
                combined_data['Ntd'].append(Ntd)
                combined_data['Windspeed'].append(corrected_windspeed)

        except IndexError as e:
            print(f"Index error at i={i}, j={j}: {e}")
            continue

# Convert to DataFrame
df_combined = pd.DataFrame(combined_data)

# Check the data for any anomalies
print(df_combined.describe())

# Separate data based on NaN Windspeed
df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]

plt.figure(figsize=(10, 8))

# Plot data with valid Windspeed values
sc = plt.scatter(df_with_windspeed['Windspeed'], df_with_windspeed['Ntd'], 
                 c=df_with_windspeed['Windspeed'], cmap='viridis', s=100, label='Windspeed')

# Plot data with NaN Windspeed values in black
plt.scatter(df_nan_windspeed['Windspeed'], df_nan_windspeed['Ntd'], 
            color='black', s=100, label='Windspeed NaN')

# Add color bar
plt.colorbar(sc, label='Corrected Windspeed (m/s)')

# Add labels and title
plt.xlabel('Windspeed (m/s)', fontsize=14, fontweight='bold')
plt.ylabel('Total Droplet Concentration of droplets \n with ddry larger than 2um d (Ntd)', fontsize=14, fontweight='bold')
plt.title('Below cloud base January - June 2022 w/ clear sky restrictions', fontsize=14, fontweight='bold')
plt.yscale('log')  # Optional: use log scale for y-axis if needed
plt.show()
#%%
##Variation of NtdNt with windspeed


# Constants
Z0 = 0.03  # meters (typical value for open terrain)
Z10 = 10  # target height in meters

# Function to apply windspeed correction
def correct_windspeed(windspeed, altitude):
    return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))

# Adjusted approach since master_min_ntd is a list
combined_data = {
    'Date': [],
    'BCB_start': [],
    'BCB_stop': [],
    'NtdNt': [],
    'Windspeed': []
}

# Validate and combine the data
for i, flight in enumerate(master_BCB):
    for j, wind_alt in enumerate(flight):
        try:
            date = wind_alt['Date']
            wind_mean = wind_alt['Winds_mean'][0]
            alt_mean = wind_alt['Alts_mean'][0]
            BCB_start = wind_alt['BCB_start']
            BCB_end = wind_alt['BCB_end']
            
            # Correct windspeed using the provided formula
            if not np.isnan(alt_mean) and alt_mean > 0:
                corrected_windspeed = correct_windspeed(wind_mean, alt_mean)
            else:
                corrected_windspeed = np.nan

            # Find the corresponding Ntd value
            NtdNt = None
            for exp_params in master_BCB_NtdNt:
                if exp_params['Date'] == date and exp_params['BCB_start'] == BCB_start and exp_params['BCB_end'] == BCB_end:
                    NtdNt = exp_params['NtdNt']
                    break
            
            # Append data only if Ntd was found
            if NtdNt is not None:
                print(f"Appending data for Date: {date}, BCB_start: {BCB_start}, BCB_end: {BCB_end}")
                combined_data['Date'].append(date)
                combined_data['BCB_start'].append(BCB_start)
                combined_data['BCB_end'].append(BCB_end)
                combined_data['NtdNt'].append(NtdNt)
                combined_data['Windspeed'].append(corrected_windspeed)

        except IndexError as e:
            print(f"Index error at i={i}, j={j}: {e}")
            continue

# Convert to DataFrame
df_combined = pd.DataFrame(combined_data)

# Check the data for any anomalies
print(df_combined.describe())

# Separate data based on NaN Windspeed
df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]

plt.figure(figsize=(10, 8))

# Plot data with valid Windspeed values
sc = plt.scatter(df_with_windspeed['Windspeed'], df_with_windspeed['NtdNt'], 
                 c=df_with_windspeed['Windspeed'], cmap='viridis', s=100, label='Windspeed')

# Plot data with NaN Windspeed values in black
plt.scatter(df_nan_windspeed['Windspeed'], df_nan_windspeed['NtdNt'], 
            color='black', s=100, label='Windspeed NaN')

# Add color bar
plt.colorbar(sc, label='Corrected Windspeed (m/s)')

# Add labels and title
plt.xlabel(' Windspeed (m/s)', fontsize=14, fontweight='bold')
plt.ylabel('Ratio of Total Droplet Concentration of dry droplets \n with diameter larger than 2um \n to ambient concentration', fontsize=14, fontweight='bold')
plt.title('Below Cloud Base January - June 2022', fontsize=14, fontweight='bold')
plt.yscale('log')  # Optional: use log scale for y-axis if needed
plt.show()
#%%
#Filtered 
##Variation of NtdNt with windspeed

# Constants
Z0 = 0.03  # meters (typical value for open terrain)
Z10 = 10  # target height in meters

# Function to apply windspeed correction
def correct_windspeed(windspeed, altitude):
    return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))

# Adjusted approach since master_min_ntd is a list
combined_data = {
    'Date': [],
    'BCB_start': [],
    'BCB_stop': [],
    'NtdNt': [],
    'Windspeed': []
}

# Validate and combine the data
for i, flight in enumerate(master_BCB):
    for j, wind_alt in enumerate(flight):
        try:
            date = wind_alt['Date']
            wind_mean = wind_alt['Winds_mean'][0]
            alt_mean = wind_alt['Alts_mean'][0]
            BCB_start = wind_alt['BCB_start']
            BCB_stop = wind_alt['BCB_end']
            
            # Correct windspeed using the provided formula
            if not np.isnan(alt_mean) and alt_mean > 0:
                corrected_windspeed = correct_windspeed(wind_mean, alt_mean)
            else:
                corrected_windspeed = np.nan

            # Find the corresponding Ntd value
            NtdNt = None
            for exp_params in filtered_master_BCB_NtdNt:
                if exp_params['Date'] == date and exp_params['BCB_start'] == BCB_start and exp_params['BCB_stop'] == BCB_stop:
                    NtdNt = exp_params['NtdNt']
                    break
            
            # Append data only if Ntd was found
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

# Convert to DataFrame
df_combined = pd.DataFrame(combined_data)

# Check the data for any anomalies
print(df_combined.describe())

# Separate data based on NaN Windspeed
df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]

plt.figure(figsize=(10, 8))

# Plot data with valid Windspeed values
sc = plt.scatter(df_with_windspeed['Windspeed'], df_with_windspeed['NtdNt'], 
                 c=df_with_windspeed['Windspeed'], cmap='viridis', s=100, label='Windspeed')

# Plot data with NaN Windspeed values in black
plt.scatter(df_nan_windspeed['Windspeed'], df_nan_windspeed['NtdNt'], 
            color='black', s=100, label='Windspeed NaN')

# Add color bar
plt.colorbar(sc, label='Corrected Windspeed (m/s)')

# Add labels and title
plt.xlabel(' Windspeed (m/s)', fontsize=14, fontweight='bold')
plt.ylabel('Ratio of Total Droplet Concentration of dry droplets \n with diameter larger than 2um \n to ambient concentration', fontsize=14, fontweight='bold')
plt.title('Below cloud base January - June 2022 w/ clear sky restrictions', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.show()
# %%
##Looking at slope versus NtdNT
# Initialize the combined data dictionary
combined_data = {
    'Date': [],
    'D': [],
    'NtdNt': []
}

# Flatten master_min_ntdNt into a list for easier sequential access
flat_ntdNt = [item for item in master_BCB_NtdNt]

ntdNt_index = 0  # Index for flat_ntdNt

# Loop through master_min_exponential and match with flat_ntdNt
for date, exp_params_list in master_BCB_exponential.items():
    for exp_params in exp_params_list:
        if ntdNt_index >= len(flat_ntdNt):
            print(f"Exhausted flat_ntdNt at date={date}")
            break

        try:
            # Access the NtdNt entry
            ntdNt_data = flat_ntdNt[ntdNt_index]
            ntdNt_index += 1
            
            # Extract data
            D = exp_params['D']
            if isinstance(D, str):
                D = float(D)  # Convert D to a float if it's a string
            
            # Debug: Print out D to verify it's being read correctly
            print(f"Date: {date}, D value: {D}")
            
            NtdNt = ntdNt_data['NtdNt']
            
            # Append data to the combined dictionary
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

# Convert to DataFrame
df_combined = pd.DataFrame(combined_data)

# # Check for anomalies
# print(df_combined.describe())

# Plot: NtdNt vs D
plt.figure(figsize=(10, 6))
plt.scatter(df_combined['D'], df_combined['NtdNt'])
plt.xlabel('Slope', fontsize=14, fontweight='bold')
plt.ylabel('Ratio of total droplet concentration of dry droplets \n with diameter larger than 2um \n to ambient concentration', fontsize=14, fontweight='bold')
plt.title('Below Cloud Base January - June 2022', fontsize=14, fontweight='bold')
plt.grid(True)
plt.yscale('log')
plt.xscale('log')
plt.show()
#%%
#Filtered 
##Looking at slope versus NtdNT
# Initialize the combined data dictionary
combined_data = {
    'Date': [],
    'D': [],
    'NtdNt': []
}

# Flatten master_min_ntdNt into a list for easier sequential access
flat_ntdNt = [item for item in filtered_master_BCB_NtdNt]

ntdNt_index = 0  # Index for flat_ntdNt

# Loop through master_min_exponential and match with flat_ntdNt
for date, exp_params_list in master_BCB_exponential.items():
    for exp_params in exp_params_list:
        if ntdNt_index >= len(flat_ntdNt):
            print(f"Exhausted flat_ntdNt at date={date}")
            break

        try:
            # Access the NtdNt entry
            ntdNt_data = flat_ntdNt[ntdNt_index]
            ntdNt_index += 1
            
            # Extract data
            D = exp_params['D']
            if isinstance(D, str):
                D = float(D)  # Convert D to a float if it's a string
            
            # Debug: Print out D to verify it's being read correctly
            print(f"Date: {date}, D value: {D}")
            
            NtdNt = ntdNt_data['NtdNt']
            
            # Append data to the combined dictionary
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

# Convert to DataFrame
df_combined = pd.DataFrame(combined_data)

# # Check for anomalies
# print(df_combined.describe())

# Plot: NtdNt vs D
plt.figure(figsize=(10, 6))
plt.scatter(df_combined['D'], df_combined['NtdNt'])
plt.xlabel('Slope', fontsize=14, fontweight='bold')
plt.ylabel('Ratio of total droplet concentration of dry droplets \n with diameter larger than 2um \n to ambient concentration', fontsize=14, fontweight='bold')
plt.title('Below cloud base January - June 2022 w/ clear sky restrictions', fontsize=14, fontweight='bold')
plt.grid(True)
plt.yscale('log')
plt.ylim(10**-5, 10**0.5)
plt.xscale('log')
plt.show()

# %%
## Now we need to try and take all of our legs with our new 18 bins of dry diameter sizes and average them 
##onto a consistent x-axis. 

##To do this we need to first create a basic random array with our bins sizes that will act as our 
##constant y-axis in which we will correct all our dry size distrbutions to. We dont care about the number of bins. We
##will plot our new dry size distributions on top of it. 

##To do this, we need D and dryintercept, which we already have from our dry size distributions

#possible array we can use 

#%%

##Eventually we need to bin by windspeed. Where each windspeed bin will contain a certain number of size distributions that
##fit in that windspeed bins and these create straight lines 
#%%
from scipy.interpolate import interp1d
#%%
#20 bins 
def size_distribution(x, dryint, D):
    dryint = dryint 
    return dryint * np.exp(-x / D)

# Prepare your common bins
common_bins = np.linspace(0, 25, 20)

interpolated_values = []

# Iterate through entries in filtered_master_min_ddry
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Directly match by index
    
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']  # Extract Leg_index
    D = entry_ddry['D']  # Extract D value
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # x-axis values for this leg
    dryint = entry_dryintercept['dry intercept']  # Use the matched N0 value from the same index
    
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

# Convert to a structured format for easy manipulation
# Each entry now contains Date, Leg_index, Min_start, Min_end, and the interpolated values
for entry in interpolated_values:
    print(f"Date: {entry['Date']}, Leg_index: {entry['Leg_index']}, BCB_start: {entry['BCB_start']}, BCB_stop: {entry['BCB_stop']}, Interpolated Values: {entry['interpolated_values']}")
# %%
# Prepare your common bins
common_bins = np.linspace(0, 25, 20)

# Create a new figure
plt.figure(figsize=(12, 8))

# Iterate through entries in interpolated_values and plot each distribution
for entry in interpolated_values:
    date = entry['Date']
    leg_index = entry['Leg_index']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    interpolated_values = entry['interpolated_values']
    
    # Plot each distribution with a label for the date and leg index
    plt.plot(common_bins, interpolated_values, label=f"Date: {date}, Leg: {leg_index} ({BCB_start} - {BCB_stop})")

# Add labels and legend
plt.ylabel('Clear mean droplet concentration (/cm^3/um)', fontweight='bold')
plt.xlabel('Bin diameter (um)', fontweight='bold')
plt.title('Below cloud base dry size distribution January-June 2022', fontweight='bold')
plt.tight_layout()
plt.show()

# %%

























































































#%%
#20 bins 
def size_distribution(x, dryint, D):
    dryint = dryint 
    return dryint * np.exp(-x / D)

# Prepare your common bins
common_bins = np.linspace(0, 25, 20)

interpolated_values = []

# Iterate through entries in filtered_master_min_ddry
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Directly match by index
    
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']  # Extract Leg_index
    D = entry_ddry['D']  # Extract D value
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # x-axis values for this leg
    dryint = entry_dryintercept['dry intercept']  # Use the matched N0 value from the same index
    
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

# Convert to a structured format for easy manipulation
# Each entry now contains Date, Leg_index, Min_start, Min_end, and the interpolated value

# Iterate through entries in filtered_master_min_ddry
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

    interp_func = interp1d(ddry_values, size_distribution(ddry_values, dryint, D), kind='linear', fill_value='extrapolate')
    interpolated_leg_values = interp_func(common_bins)

    # Debug print
    print(f"Appending: Date: {date}, Leg_index: {leg_index}, Interpolated Values: {interpolated_leg_values.tolist()}")

    # Store the interpolated values with metadata
    interpolated_values.append({
        'Date': date,
        'Leg_index': leg_index,
        'BCB_start': BCB_start,
        'BCB_stop': BCB_stop,
        'interpolated_values': interpolated_leg_values.tolist()
    })

#%%
# Step 1: Define the problematic legs
# problematic_legs = [
#     ('2022-01-11', 1),
#     ('2022-01-11', 2),
#     ('2022-01-11', 3),
#     ('2022-01-11', 4),
#     ('2022-01-11', 8),
#     ('2022-01-11', 9),
#     ('2022-01-11', 14),
#     ('2022-01-11', 16),
#     ('2022-01-11', 17),
#     ('2022-01-26', 14),
#     ('2022-01-26', 12),
#     ('2022-01-26', 15),
#     ('2022-01-26', 13),
#     ('2022-01-26', 11),
#     ('2022-03-29', 0),
#     ('2022-05-05', 3),
#     ('2022-05-05', 7)]
problematic_legs = [
    ('2022-01-11', 1), ('2022-01-11', 2), ('2022-01-11', 3), ('2022-01-11', 4),
    ('2022-01-11', 8), ('2022-01-11', 9), ('2022-01-11', 14), ('2022-01-11', 16),
    ('2022-01-11', 17),('2022-01-26', 16), ('2022-01-26', 14), ('2022-01-26', 12), 
    ('2022-01-26', 15),
    ('2022-01-26', 13), ('2022-01-26', 11), ('2022-03-29', 0), ('2022-05-05', 3),
    ('2022-05-05', 7), ('2022-05-05', 3), ('2022-06-11', 4), ('2022-03-29', 9), ('2022-03-29', 9), 
    ('2022-03-29', 0), ('2022-03-29', 6), ('2022-03-28', 4), ('2022-03-28', 0), ('2022-03-26', 15),
    ('2022-03-26', 6), ('2022-03-26', 0), ('2022-03-13', 8)]
problematic_set = set(problematic_legs)
# Convert the list of problematic legs into a set for faster lookups
problematic_set = set(problematic_legs)

# Step 2: Filter out problematic legs from interpolated_values
filtered_interpolated_values = [
    entry for entry in interpolated_values
    if (entry['Date'], entry['Leg_index']) not in problematic_set
]

# Step 3: Plot the filtered size distributions
plt.figure(figsize=(12, 8))

# Prepare common bins again if needed
common_bins = np.linspace(0, 50, 20)

for entry in filtered_interpolated_values:
    date = entry['Date']
    leg_index = entry['Leg_index']
    interpolated_values = entry['interpolated_values']

    # Plot each distribution
    plt.plot(common_bins, interpolated_values, label=f"Date: {date}, Leg: {leg_index}")

# Add labels and legend
plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Below cloud base Dry Size Distribution (Excluding Problematic Legs)', fontweight='bold')
# plt.legend()
plt.tight_layout()
plt.show()
#%%
# Extract windspeed values from your dataset
windspeeds = df_combined['Windspeed'].dropna()  # Drop NaN values

# Plot the histogram and density plot
plt.figure(figsize=(10, 6))
sns.histplot(windspeeds, bins=15, kde=True)  # kde=True adds the kernel density estimate

# Add labels and title
plt.xlabel('Windspeed (m/s)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Distribution of Windspeed Values', fontsize=16)
plt.show()
#%%

#MAYBE 
# Define the function to compute size distribution
def size_distribution(x, dryint, D):
    return dryint * np.exp(-x / D)

# Define your common bins
common_bins = np.linspace(0, 25, 20)

# Dictionary to store results for each windspeed range
windspeed_bins = [(0, 4), (4.1, 6), (6.1, 8), (8.1, 10), (10.1, 12), (12.1, np.inf)]
grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Iterate through the entries in filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Match by index
    
    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']  # Extract Leg_index
    D = entry_ddry['D']  # Extract D value
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # x-axis values for this leg
    dryint = entry_dryintercept['dry intercept']  # Extract dry intercept

    # Find the corresponding entry in df_combined based on Date, Min_start, and Min_end
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    
    # If there's a matching windspeed entry
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]
        
        # Interpolate the size distribution for this leg
        interp_func = interp1d(ddry_values, size_distribution(ddry_values, dryint, D), kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)
        
        # Categorize the leg based on windspeed range
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_distributions[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break

# Average and plot the results
plt.figure(figsize=(12, 8))
for idx, ranges in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])  # Count the number of legs
        plt.plot(common_bins, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")

plt.ylabel('Clear mean droplet concentration (/cm^3/um)')
plt.xlabel('Bin diameter (um)')
plt.title('Size Distribution by Windspeed')
plt.legend(title="Windspeed")
plt.show()
#%%
# Step 1: Define the problematic legs (if not already defined)
problematic_legs = [
       ('2022-01-11', 1),
    ('2022-01-11', 2),
    ('2022-01-11', 3),
    ('2022-01-11', 4),
    ('2022-01-11', 8),
    ('2022-01-11', 9),
    ('2022-01-11', 14),
    ('2022-01-11', 16),
    ('2022-01-11', 17),
    ('2022-01-26', 14),
    ('2022-01-26', 12),
    ('2022-01-26', 15),
    ('2022-01-26', 13),
    ('2022-01-26', 11),
    ('2022-03-29', 0),
    ('2022-05-05', 3),
    ('2022-05-05', 7),
]
problematic_set = set(problematic_legs)

# Define the function to compute size distribution
def size_distribution(x, dryint, D):
    return dryint * np.exp(-x / D)

# Define your common bins
common_bins = np.linspace(0, 25, 20)

# Dictionary to store results for each windspeed range
windspeed_bins = [(0, 4), (4.1, 6), (6.1, 8), (8.1, 10), (10.1, 12), (12.1, np.inf)]
grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Iterate through the entries in filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Match by index
    
    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']  # Extract Leg_index

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']  # Extract D value
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # x-axis values for this leg
    dryint = entry_dryintercept['dry intercept']  # Extract dry intercept

    # Find the corresponding entry in df_combined based on Date, Min_start, and Min_end
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    
    # If there's a matching windspeed entry
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]
        
        # Interpolate the size distribution for this leg
        interp_func = interp1d(ddry_values, size_distribution(ddry_values, dryint, D), kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)
        
        # Categorize the leg based on windspeed range
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_distributions[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break

# Average and plot the results
plt.figure(figsize=(12, 8))
for idx, ranges in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])  # Count the number of legs
        plt.plot(common_bins, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")

plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Size Distribution by Windspeed (Excluding Problematic Legs)', fontweight='bold')
plt.legend(title="Windspeed")
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.show()
#%%
#Trying 4 legs 


# Step 1: Define the problematic legs (if not already defined)
problematic_legs = [
      ('2022-01-11', 1),
    ('2022-01-11', 2),
    ('2022-01-11', 3),
    ('2022-01-11', 4),
    ('2022-01-11', 8),
    ('2022-01-11', 9),
    ('2022-01-11', 14),
    ('2022-01-11', 16),
    ('2022-01-11', 17),
    ('2022-01-26', 14),
    ('2022-01-26', 12),
    ('2022-01-26', 15),
    ('2022-01-26', 13),
    ('2022-01-26', 11),
    ('2022-03-29', 0),
    ('2022-05-05', 3),
    ('2022-05-05', 7),
]

# problematic_legs = [
#     ('2022-01-11', 1), ('2022-01-11', 2), ('2022-01-11', 3), ('2022-01-11', 4),
#     ('2022-01-11', 8), ('2022-01-11', 9), ('2022-01-11', 14), ('2022-01-11', 16),
#     ('2022-01-11', 17),('2022-01-26', 16), ('2022-01-26', 14), ('2022-01-26', 12), 
#     ('2022-01-26', 15),
#     ('2022-01-26', 13), ('2022-01-26', 11), ('2022-03-29', 0), ('2022-05-05', 3),
#     ('2022-05-05', 7), ('2022-05-05', 3), ('2022-06-11', 4), ('2022-03-29', 9), ('2022-03-29', 9), 
#     ('2022-03-29', 0), ('2022-03-29', 6), ('2022-03-28', 4), ('2022-03-28', 0), ('2022-03-26', 15),
#     ('2022-03-26', 6), ('2022-03-26', 0), ('2022-03-13', 8)]
# problematic_set = set(problematic_legs)

problematic_set = set(problematic_legs)

# Define the function to compute size distribution
def size_distribution(x, dryint, D):
    return dryint * np.exp(-x / D)

# Define your common bins
common_bins = np.linspace(0, 25, 20)

# Dictionary to store results for each windspeed range
windspeed_bins = [(0, 4), (4.1, 6), (6.1, 8), (8.1, np.inf)]
grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Iterate through the entries in filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Match by index
    
    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']  # Extract Leg_index

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']  # Extract D value
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # x-axis values for this leg
    dryint = entry_dryintercept['dry intercept']  # Extract dry intercept

    # Find the corresponding entry in df_combined based on Date, Min_start, and Min_end
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    
    # If there's a matching windspeed entry
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]
        
        # Interpolate the size distribution for this leg
        interp_func = interp1d(ddry_values, size_distribution(ddry_values, dryint, D), kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)
        
        # Categorize the leg based on windspeed range
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_distributions[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break

# Average and plot the results
plt.figure(figsize=(12, 8))
for idx, ranges in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])  # Count the number of legs
        plt.plot(common_bins, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")

plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Size Distribution by Windspeed (Excluding Problematic Legs)', fontweight='bold')
plt.legend(title="Windspeed")
plt.yscale('log')
# plt.xscale('log')
plt.tight_layout()
plt.show()
#%%
#trying windspeed bins based on the percentile plot 

# Assuming df_combined is already defined and contains 'Windspeed'
windspeeds = df_combined['Windspeed'].dropna()  # Drop NaN values

# Calculate percentiles for windspeed bins
percentiles = np.percentile(windspeeds, [0, 25, 50, 75, 100])  # Change percentiles as needed

# Define your windspeed ranges based on percentiles
windspeed_bins = [(percentiles[i], percentiles[i + 1]) for i in range(len(percentiles) - 1)]

# Define the problematic legs
problematic_legs = [
    ('2022-01-11', 1),
    ('2022-01-11', 2),
    ('2022-01-11', 3),
    ('2022-01-11', 4),
    ('2022-01-11', 8),
    ('2022-01-11', 9),
    ('2022-01-11', 14),
    ('2022-01-11', 16),
    ('2022-01-11', 17),
    ('2022-01-26', 14),
    ('2022-01-26', 12),
    ('2022-01-26', 15),
    ('2022-01-26', 13),
    ('2022-01-26', 11),
    ('2022-03-29', 0),
    ('2022-05-05', 3),
    ('2022-05-05', 7),
]
problematic_set = set(problematic_legs)

# Define the function to compute log-transformed size distribution
def log_size_distribution(x, dryint, D):
    return np.log(dryint * np.exp(-x / D))

# Define your common bins
common_bins = np.linspace(0, 25, 20)

grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Iterate through the entries in filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Match by index

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']  # Extract Leg_index

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']  # Extract D value
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # x-axis values for this leg
    dryint = entry_dryintercept['dry intercept']  # Extract dry intercept

    # Find the corresponding entry in df_combined based on Date, Min_start, and Min_end
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    # If there's a matching windspeed entry
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]

        # Interpolate the log-transformed size distribution for this leg
        interp_func = interp1d(ddry_values, log_size_distribution(ddry_values, dryint, D), kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        # Exponentiate the interpolated values to get back to the original scale for plotting
        interpolated_leg_values = np.exp(interpolated_leg_values)

        # Categorize the leg based on windspeed range
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_distributions[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break

# Average and plot the results
plt.figure(figsize=(12, 8))
for idx, ranges in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])  # Count the number of legs
        plt.plot(common_bins, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")

plt.yscale('log')  # Set y-axis to log scale
plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Size Distribution by Windspeed (Log Scale, Excluding Problematic Legs)', fontweight='bold')
plt.legend(title="Windspeed")
plt.tight_layout()
plt.show()
#%%
#Percentile binning with error bars 

# Assuming df_combined is already defined and contains 'Windspeed'
windspeeds = df_combined['Windspeed'].dropna()  # Drop NaN values

# Calculate percentiles for windspeed bins
percentiles = np.percentile(windspeeds, [0, 25, 50, 75, 100])

# Define your windspeed ranges based on percentiles
windspeed_bins = [(percentiles[i], percentiles[i + 1]) for i in range(len(percentiles) - 1)]

# Define the problematic legs
problematic_legs = [
       ('2022-01-11', 1),
    ('2022-01-11', 2),
    ('2022-01-11', 3),
    ('2022-01-11', 4),
    ('2022-01-11', 8),
    ('2022-01-11', 9),
    ('2022-01-11', 14),
    ('2022-01-11', 16),
    ('2022-01-11', 17),
    ('2022-01-26', 14),
    ('2022-01-26', 12),
    ('2022-01-26', 15),
    ('2022-01-26', 13),
    ('2022-01-26', 11),
    ('2022-03-29', 0),
    ('2022-05-05', 3),
    ('2022-05-05', 7),
]
problematic_set = set(problematic_legs)

# Define the function to compute log-transformed size distribution
def log_size_distribution(x, dryint, D):
    return np.log(dryint * np.exp(-x / D))

# Define your common bins
common_bins = np.linspace(0, 25, 20)

# Set a color palette
colors = sns.color_palette("husl", len(windspeed_bins))  # Choose a palette with distinct colors

grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Iterate through the entries in filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Match by index

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']  # Extract Leg_index

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']  # Extract D value
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # x-axis values for this leg
    dryint = entry_dryintercept['dry intercept']  # Extract dry intercept

    # Find the corresponding entry in df_combined based on Date, Min_start, and Min_end
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    # If there's a matching windspeed entry
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]

        # Interpolate the log-transformed size distribution for this leg
        interp_func = interp1d(ddry_values, log_size_distribution(ddry_values, dryint, D), kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        # Exponentiate the interpolated values to get back to the original scale for plotting
        interpolated_leg_values = np.exp(interpolated_leg_values)

        # Categorize the leg based on windspeed range
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_distributions[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break

# Average and plot the results
plt.figure(figsize=(12, 8))
for idx, ranges in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)

        # Calculate percentiles for the shading
        lower_percentile = np.percentile(grouped_distributions[idx], 25, axis=0)
        upper_percentile = np.percentile(grouped_distributions[idx], 75, axis=0)

        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])  # Count the number of legs
        
        # Plot the average distribution with different colors
        plt.plot(common_bins, avg_distribution, color=colors[idx], label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linewidth=2)

        # Plot the shaded area for the interquartile range with transparency
        plt.fill_between(common_bins, lower_percentile, upper_percentile, color=colors[idx], alpha=0.4, linewidth=0)

        # Optionally plot the percentile lines
        plt.plot(common_bins, lower_percentile, color=colors[idx], linestyle='--', linewidth=1, label=f'{avg_windspeed:.1f} m/s 25th Percentile')
        plt.plot(common_bins, upper_percentile, color=colors[idx], linestyle='--', linewidth=1, label=f'{avg_windspeed:.1f} m/s 75th Percentile')

plt.yscale('log')  # Set y-axis to log scale
plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Size Distribution by Windspeed (Log Scale, Excluding Problematic Legs)', fontweight='bold')
plt.legend(title="Windspeed")
plt.tight_layout()
plt.show()
#%%

problematic_legs = [
       ('2022-01-11', 1),
    ('2022-01-11', 2),
    ('2022-01-11', 3),
    ('2022-01-11', 4),
    ('2022-01-11', 8),
    ('2022-01-11', 9),
    ('2022-01-11', 14),
    ('2022-01-11', 16),
    ('2022-01-11', 17),
    ('2022-01-26', 14),
    ('2022-01-26', 12),
    ('2022-01-26', 15),
    ('2022-01-26', 13),
    ('2022-01-26', 11),
    ('2022-03-29', 0),
    ('2022-05-05', 3),
    ('2022-05-05', 7),
]
problematic_set = set(problematic_legs)

# Define the function to compute log-transformed size distribution
def log_size_distribution(x, dryint, D):
    return np.log(dryint * np.exp(-x / D))

# Define your common bins
common_bins = np.linspace(0, 25, 20)

# Define your windspeed ranges
windspeed_bins = [
    (0, 4), (4.1, 6), (6.1, 8), (8.1, 10), (10.1, 12), (12.1, np.inf)
]
grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Iterate through the entries in filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Match by index

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']  # Extract Leg_index

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']  # Extract D value
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # x-axis values for this leg
    dryint = entry_dryintercept['dry intercept']  # Extract dry intercept

    # Find the corresponding entry in df_combined based on Date, Min_start, and Min_end
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    # If there's a matching windspeed entry
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]

        # Interpolate the log-transformed size distribution for this leg
        interp_func = interp1d(ddry_values, log_size_distribution(ddry_values, dryint, D), kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        # Exponentiate the interpolated values to get back to the original scale for plotting
        interpolated_leg_values = np.exp(interpolated_leg_values)

        # Categorize the leg based on windspeed range
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_distributions[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break

# Average and plot the results
plt.figure(figsize=(12, 8))
for idx, ranges in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)
        std_distribution = np.std(grouped_distributions[idx], axis=0)  # Standard deviation for uncertainty
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])  # Count the number of legs
        
        # Plot the mean distribution
        plt.plot(common_bins, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linestyle='-', marker='o')

        # Plot the uncertainty (percentiles)
        lower_percentile = np.percentile(grouped_distributions[idx], 25, axis=0)  # 25th percentile
        upper_percentile = np.percentile(grouped_distributions[idx], 75, axis=0)  # 75th percentile
        plt.fill_between(common_bins, lower_percentile, upper_percentile, alpha=0.2)

plt.yscale('log')  # Set y-axis to log scale
plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Size Distribution by Windspeed (Log Scale, Excluding Problematic Legs)', fontweight='bold')
plt.legend(title="Windspeed")
plt.tight_layout()
plt.show()
#%%

problematic_legs = [
       ('2022-01-11', 1),
    ('2022-01-11', 2),
    ('2022-01-11', 3),
    ('2022-01-11', 4),
    ('2022-01-11', 8),
    ('2022-01-11', 9),
    ('2022-01-11', 14),
    ('2022-01-11', 16),
    ('2022-01-11', 17),
    ('2022-01-26', 14),
    ('2022-01-26', 12),
    ('2022-01-26', 15),
    ('2022-01-26', 13),
    ('2022-01-26', 11),
    ('2022-03-29', 0),
    ('2022-05-05', 3),
    ('2022-05-05', 7),
]
problematic_set = set(problematic_legs)

# Define the function to compute log-transformed size distribution
def log_size_distribution(x, dryint, D):
    return np.log(dryint * np.exp(-x / D))

# Define your common bins
common_bins = np.linspace(0, 25, 20)

# Define your windspeed ranges
windspeed_bins = [(0, 1),(1.1, 1.6), (1.7,2), (2.1, 3.9), (4, 5.5), (5.6, 6.4), (6.5, 7.4), (7.4, 8),(8.1, 9.5), (9.6, 11), (11.1, 12.0), (12.1, np.inf)]

grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Iterate through the entries in filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Match by index

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']  # Extract Leg_index

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']  # Extract D value
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # x-axis values for this leg
    dryint = entry_dryintercept['dry intercept']  # Extract dry intercept

    # Find the corresponding entry in df_combined based on Date, Min_start, and Min_end
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    # If there's a matching windspeed entry
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]

        # Interpolate the log-transformed size distribution for this leg
        interp_func = interp1d(ddry_values, log_size_distribution(ddry_values, dryint, D), kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        # Exponentiate the interpolated values to get back to the original scale for plotting
        interpolated_leg_values = np.exp(interpolated_leg_values)

        # Categorize the leg based on windspeed range
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_distributions[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break

# Average and plot the results
plt.figure(figsize=(12, 8))
for idx, ranges in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])  # Count the number of legs
        
        # Calculate percentiles for shading
        lower_percentile = np.percentile(grouped_distributions[idx], 25, axis=0)
        upper_percentile = np.percentile(grouped_distributions[idx], 75, axis=0)

        # Plot the average distribution
        plt.plot(common_bins, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")

        # Plot the shaded area for the interquartile range
        plt.fill_between(common_bins, lower_percentile, upper_percentile, alpha=0.4)

        # Optionally plot the percentile lines
        plt.plot(common_bins, lower_percentile, linestyle='--', linewidth=1, color='gray')
        plt.plot(common_bins, upper_percentile, linestyle='--', linewidth=1, color='gray')

plt.yscale('log')  # Set y-axis to log scale
plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Size Distribution by Windspeed (Log Scale, Excluding Problematic Legs)', fontweight='bold')
plt.legend(title="Windspeed")
plt.tight_layout()
plt.show()
#%%

problematic_legs = [
       ('2022-01-11', 1),
    ('2022-01-11', 2),
    ('2022-01-11', 3),
    ('2022-01-11', 4),
    ('2022-01-11', 8),
    ('2022-01-11', 9),
    ('2022-01-11', 14),
    ('2022-01-11', 16),
    ('2022-01-11', 17),
    ('2022-01-26', 14),
    ('2022-01-26', 12),
    ('2022-01-26', 15),
    ('2022-01-26', 13),
    ('2022-01-26', 11),
    ('2022-03-29', 0),
    ('2022-05-05', 3),
    ('2022-05-05', 7),
]
problematic_set = set(problematic_legs)

# Define the function to compute log-transformed size distribution
def log_size_distribution(x, dryint, D):
    return np.log(dryint * np.exp(-x / D))

# Define your common bins
common_bins = np.linspace(0, 25, 20)

# Define your windspeed ranges
windspeed_bins = [(0, 4), (4.1, 6), (6.1, 8), (8.1, np.inf)]

grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Iterate through the entries in filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Match by index

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']  # Extract Leg_index

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']  # Extract D value
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # x-axis values for this leg
    dryint = entry_dryintercept['dry intercept']  # Extract dry intercept

    # Find the corresponding entry in df_combined based on Date, Min_start, and Min_end
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    # If there's a matching windspeed entry
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]

        # Interpolate the log-transformed size distribution for this leg
        interp_func = interp1d(ddry_values, log_size_distribution(ddry_values, dryint, D), kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        # Exponentiate the interpolated values to get back to the original scale for plotting
        interpolated_leg_values = np.exp(interpolated_leg_values)

        # Categorize the leg based on windspeed range
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_distributions[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break

# Average and plot the results
plt.figure(figsize=(12, 8))
for idx, ranges in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])  # Count the number of legs
        
        # Calculate percentiles for shading
        lower_percentile = np.percentile(grouped_distributions[idx], 25, axis=0)
        upper_percentile = np.percentile(grouped_distributions[idx], 75, axis=0)

        # Plot the average distribution
        plt.plot(common_bins, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")

        # Plot the shaded area for the interquartile range
        plt.fill_between(common_bins, lower_percentile, upper_percentile, alpha=0.4)

        # Optionally plot the percentile lines
        plt.plot(common_bins, lower_percentile, linestyle='--', linewidth=1, color='gray')
        plt.plot(common_bins, upper_percentile, linestyle='--', linewidth=1, color='gray')

plt.yscale('log')  # Set y-axis to log scale
plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Size Distribution by Windspeed (Log Scale, Excluding Problematic Legs)', fontweight='bold')
plt.legend(title="Windspeed")
plt.tight_layout()
plt.show()
#%%
##trying a third order polynomial 
# Define the third-order polynomial function
def polynomial_fit(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

# Define problematic legs (if not already defined)
problematic_legs = [
    ('2022-01-11', 1), ('2022-01-11', 2), ('2022-01-11', 3), ('2022-01-11', 4),
    ('2022-01-11', 8), ('2022-01-11', 9), ('2022-01-11', 14), ('2022-01-11', 16),
    ('2022-01-11', 17), ('2022-01-26', 14), ('2022-01-26', 12), ('2022-01-26', 15),
    ('2022-01-26', 13), ('2022-01-26', 11), ('2022-03-29', 0), ('2022-05-05', 3),
    ('2022-05-05', 7)
]
problematic_set = set(problematic_legs)

# Define your common bins and windspeed bins
common_bins = np.linspace(0, 25, 20)
windspeed_bins = [(0, 4), (4.1, 6), (6.1, 8), (8.1, np.inf)]

grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Iterate through the entries in filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Match by index

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # x-axis values for this leg
    dryint = entry_dryintercept['dry intercept']

    # Find the corresponding entry in df_combined based on Date, Min_start, and Min_end
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    # If there's a matching windspeed entry
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]

        # Interpolate the size distribution for this leg
        interp_func = interp1d(ddry_values, dryint * np.exp(-ddry_values / D), kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        # Categorize the leg based on windspeed range
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_distributions[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break

# Step 3: Average the results and fit a third-order polynomial using curve_fit
plt.figure(figsize=(12, 8))
for idx, ranges in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        # Average the size distributions for the windspeed bin
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])

        # Fit the third-order polynomial using curve_fit
        popt, pcov = curve_fit(polynomial_fit, common_bins, avg_distribution)

        # Generate the polynomial curve
        poly_curve = polynomial_fit(common_bins, *popt)

        # Calculate percentiles for shading
        lower_percentile = np.percentile(grouped_distributions[idx], 25, axis=0)
        upper_percentile = np.percentile(grouped_distributions[idx], 75, axis=0)

        # Plot the average distribution
        plt.plot(common_bins, avg_distribution, 'o', label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", markersize=5)

        # Plot the fitted third-order polynomial curve
        plt.plot(common_bins, poly_curve, label=f"Poly fit {avg_windspeed:.1f} m/s")

        # Plot the shaded area for the interquartile range
        plt.fill_between(common_bins, lower_percentile, upper_percentile, alpha=0.2)

# Step 4: Set y-axis to log scale and plot labels
plt.yscale('log')  # Set y-axis to log scale
plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Size Distribution by Windspeed with 3rd Order Polynomial Fit', fontweight='bold')
plt.legend(title="Windspeed")
plt.tight_layout()
plt.show()


#%%
##fitting a 4th order polynomial 


# Define the fourth-order polynomial function
def polynomial_fit(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e

# Define problematic legs (if not already defined)
problematic_legs = [
    ('2022-01-11', 1), ('2022-01-11', 2), ('2022-01-11', 3), ('2022-01-11', 4),
    ('2022-01-11', 8), ('2022-01-11', 9), ('2022-01-11', 14), ('2022-01-11', 16),
    ('2022-01-11', 17), ('2022-01-26', 14), ('2022-01-26', 12), ('2022-01-26', 15),
    ('2022-01-26', 13), ('2022-01-26', 11), ('2022-03-29', 0), ('2022-05-05', 3),
    ('2022-05-05', 7)
]
problematic_set = set(problematic_legs)

# Define your common bins and windspeed bins
common_bins = np.linspace(0, 25, 20)
windspeed_bins = [(0, 4), (4.1, 6), (6.1, 8), (8.1, np.inf)]

grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Iterate through the entries in filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Match by index

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # x-axis values for this leg
    dryint = entry_dryintercept['dry intercept']

    # Find the corresponding entry in df_combined based on Date, Min_start, and Min_end
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    # If there's a matching windspeed entry
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]

        # Interpolate the size distribution for this leg
        interp_func = interp1d(ddry_values, dryint * np.exp(-ddry_values / D), kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        # Categorize the leg based on windspeed range
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_distributions[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break

# Step 3: Average the results and fit a fourth-order polynomial using curve_fit
plt.figure(figsize=(12, 8))
for idx, ranges in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        # Average the size distributions for the windspeed bin
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])

        # Fit the fourth-order polynomial using curve_fit
        popt, pcov = curve_fit(polynomial_fit, common_bins, avg_distribution)

        # Generate the polynomial curve
        poly_curve = polynomial_fit(common_bins, *popt)

        # Calculate percentiles for shading
        lower_percentile = np.percentile(grouped_distributions[idx], 25, axis=0)
        upper_percentile = np.percentile(grouped_distributions[idx], 75, axis=0)

        # Plot the average distribution
        plt.plot(common_bins, avg_distribution, 'o', label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", markersize=5)

        # Plot the fitted fourth-order polynomial curve
        plt.plot(common_bins, poly_curve, label=f"4th-order Poly fit {avg_windspeed:.1f} m/s")

        # Plot the shaded area for the interquartile range
        plt.fill_between(common_bins, lower_percentile, upper_percentile, alpha=0.2)

# Step 4: Set y-axis to log scale and plot labels
plt.yscale('log')  # Set y-axis to log scale
plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Size Distribution by Windspeed with 4th Order Polynomial Fit', fontweight='bold')
plt.legend(title="Windspeed")
plt.tight_layout()
plt.show()
#%%
##fitting a fith order polynomial 

# Define the fifth-order polynomial function
def polynomial_fit(x, a, b, c, d, e, f):
    return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f

# Define problematic legs (if not already defined)
problematic_legs = [
    ('2022-01-11', 1), ('2022-01-11', 2), ('2022-01-11', 3), ('2022-01-11', 4),
    ('2022-01-11', 8), ('2022-01-11', 9), ('2022-01-11', 14), ('2022-01-11', 16),
    ('2022-01-11', 17), ('2022-01-26', 14), ('2022-01-26', 12), ('2022-01-26', 15),
    ('2022-01-26', 13), ('2022-01-26', 11), ('2022-03-29', 0), ('2022-05-05', 3),
    ('2022-05-05', 7)
]
problematic_set = set(problematic_legs)

# Define your common bins and windspeed bins
common_bins = np.linspace(0, 25, 20)
windspeed_bins = [(0, 4), (4.1, 6), (6.1, 8), (8.1, np.inf)]

grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Iterate through the entries in filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Match by index

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # x-axis values for this leg
    dryint = entry_dryintercept['dry intercept']

    # Find the corresponding entry in df_combined based on Date, Min_start, and Min_end
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    # If there's a matching windspeed entry
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]

        # Interpolate the size distribution for this leg
        interp_func = interp1d(ddry_values, dryint * np.exp(-ddry_values / D), kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        # Categorize the leg based on windspeed range
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_distributions[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break

# Step 3: Average the results and fit a fifth-order polynomial using curve_fit
plt.figure(figsize=(12, 8))
for idx, ranges in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        # Average the size distributions for the windspeed bin
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])

        # Fit the fifth-order polynomial using curve_fit
        popt, pcov = curve_fit(polynomial_fit, common_bins, avg_distribution)

        # Generate the polynomial curve
        poly_curve = polynomial_fit(common_bins, *popt)

        # Calculate percentiles for shading
        lower_percentile = np.percentile(grouped_distributions[idx], 25, axis=0)
        upper_percentile = np.percentile(grouped_distributions[idx], 75, axis=0)

        # Plot the average distribution
        plt.plot(common_bins, avg_distribution, 'o', label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", markersize=5)

        # Plot the fitted fifth-order polynomial curve
        plt.plot(common_bins, poly_curve, label=f"5th-order Poly fit {avg_windspeed:.1f} m/s")

        # Plot the shaded area for the interquartile range
        plt.fill_between(common_bins, lower_percentile, upper_percentile, alpha=0.2)

# Step 4: Set y-axis to log scale and plot labels
plt.yscale('log')  # Set y-axis to log scale
plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Size Distribution by Windspeed with 5th Order Polynomial Fit', fontweight='bold')
plt.legend(title="Windspeed")
plt.tight_layout()
plt.show()












































































































#%%
problematic_set = set(problematic_legs)

# Define the function to compute log-transformed size distribution
def log_size_distribution(x, dryint, D):
    return np.log(dryint * np.exp(-x / D))

# Define your common bins
common_bins = np.linspace(0, 25, 20)

# Define your windspeed ranges
windspeed_bins = [(0, 2.5), (2.6, 4), (4.1, 5.3),(5.4, 6.4), (6.5, 7.3), (7.4, 8), (8.1, 9.7), (9.8, np.inf)]

grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Iterate through the entries in filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Match by index

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']  # Extract Leg_index

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']  # Extract D value
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # x-axis values for this leg
    dryint = entry_dryintercept['dry intercept']  # Extract dry intercept

    # Find the corresponding entry in df_combined based on Date, Min_start, and Min_end
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    # If there's a matching windspeed entry
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]

        # Interpolate the log-transformed size distribution for this leg
        interp_func = interp1d(ddry_values, log_size_distribution(ddry_values, dryint, D), kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        # Exponentiate the interpolated values to get back to the original scale for plotting
        interpolated_leg_values = np.exp(interpolated_leg_values)

        # Categorize the leg based on windspeed range
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_distributions[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break

# Average and plot the results
plt.figure(figsize=(12, 8))
for idx, ranges in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])  # Count the number of legs
        plt.plot(common_bins, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")

plt.yscale('log')  # Set y-axis to log scale
plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Size Distribution by Windspeed (Log Scale, Excluding Problematic Legs)', fontweight='bold')
plt.legend(title="Windspeed")
plt.tight_layout()
plt.show()
#%%
#How windspeed, slope, and Ntd correlate with windspeed 

Z0 = 0.03  # meters (typical value for open terrain)
Z10 = 10  # target height m

# Function to apply windspeed correction
def correct_windspeed(windspeed, altitude):
    return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))

# Initialize dictionary to hold combined data
combined_data = {
    'Date': [],
    'BCB_start': [],
    'BCB_stop': [],
    'Ntd': [],
    'D': [],
    'Windspeed': []
}

# Loop through master_Min to retrieve windspeed and altitude data
for i, flight in enumerate(master_BCB):
    for j, wind_alt in enumerate(flight):
        try:
            # Extract Date, windspeed, and altitude from master_Min
            date = wind_alt['Date']
            wind_mean = wind_alt['Winds_mean'][0]
            alt_mean = wind_alt['Alts_mean'][0]
            BCB_start = wind_alt['BCB_start']
            BCB_stop = wind_alt['BCB_end']
            
            # Correct windspeed using the provided formula
            if not np.isnan(alt_mean) and alt_mean > 0:
                corrected_windspeed = correct_windspeed(wind_mean, alt_mean)
            else:
                corrected_windspeed = np.nan

            # Initialize variables to store found data
            Ntd = None
            D = None

            # Find corresponding exponential parameters (D) from master_min_exponential
            if date in master_BCB_exponential:
                for exp_params in master_BCB_exponential[date]:
                    # Check if the current dictionary matches Min_start and Min_end
                    if exp_params['BCB_start'] == BCB_start and exp_params['BCB_stop'] == BCB_stop:
                        D = exp_params['D']
                        print(f"Found D for Date: {date}, BCB_start: {BCB_start}, BCB_stop: {BCB_stop}, D: {D}")
                        break

            # Find corresponding Ntd from filtered_master_min_ntd
            for ntd_data in filtered_master_BCB_ntd:
                if ntd_data['Date'] == date and ntd_data['BCB_start'] == BCB_start and ntd_data['BCB_stop'] == BCB_stop:
                    Ntd = ntd_data['Ntd']
                    break

            # Append data if both Ntd and D are found
            if Ntd is not None and D is not None:
                combined_data['Date'].append(date)
                combined_data['BCB_start'].append(BCB_start)
                combined_data['BCB_stop'].append(BCB_stop)
                combined_data['Ntd'].append(Ntd)
                combined_data['D'].append(D)
                combined_data['Windspeed'].append(corrected_windspeed)

        except IndexError as e:
            print(f"Index error at i={i}, j={j}: {e}")
            continue

# Convert the combined data to a DataFrame
df_combined = pd.DataFrame(combined_data)

# Check for anomalies or missing values
print(df_combined.describe())

# Separate data based on NaN Windspeed
df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]

# Plot: Ntd vs D, colored by Windspeed
plt.figure(figsize=(10, 8))

# Plot data with valid Windspeed values
sc = plt.scatter(df_with_windspeed['D'], df_with_windspeed['Ntd'], 
                 c=df_with_windspeed['Windspeed'], cmap='viridis', s=100, label='Windspeed')

# Plot data with NaN Windspeed values in black
plt.scatter(df_nan_windspeed['D'], df_nan_windspeed['Ntd'], 
            color='black', s=100, label='Windspeed NaN')

# Add color bar
plt.colorbar(sc, label='Corrected Windspeed (m/s)')

# Add labels and title
plt.xlabel('Slope (D)', fontsize=14, fontweight='bold')
plt.ylabel('Total droplet concentration \n dry droplets larger than \n 2um diameter (Ntd)', fontsize=14, fontweight='bold')
plt.title('Below cloud base January - June 2022 w/ clear sky restrictions', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.xscale('log')

# Show the plot
plt.show()
#%%
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.hist(df_with_windspeed['Windspeed'], bins=30, color='skyblue', edgecolor='black')
plt.title('Windspeed Distribution')
plt.xlabel('Corrected Windspeed (m/s)')
plt.ylabel('Frequency')

plt.subplot(3, 1, 2)
plt.hist(df_with_windspeed['Ntd'], bins=30, color='lightgreen', edgecolor='black')
plt.title('Total concentration above 2um diameter Distribution')
plt.xlabel('Total concentration above 2um diameter')
plt.ylabel('Frequency')
plt.yscale('log')

plt.subplot(3, 1, 3)
plt.hist(df_with_windspeed['D'], bins=30, color='lightcoral', edgecolor='black')
plt.title('Slope Distribution')
plt.xlabel('Slope')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
#%%
# Define number of bins
num_bins = 4  # Adjust this number based on your data distribution

# Create windspeed bins based on quantiles
df_with_windspeed['Windspeed_bin'] = pd.qcut(df_with_windspeed['Windspeed'], num_bins, labels=False)

# Display the bins and their corresponding ranges
windspeed_bin_ranges = pd.qcut(df_with_windspeed['Windspeed'], num_bins, retbins=True)[1]
print("Windspeed bin ranges:", windspeed_bin_ranges)

# Group by windspeed bins
grouped_data = df_with_windspeed.groupby('Windspeed_bin').agg({'D': 'mean', 'Ntd': 'mean'}).reset_index()
print(grouped_data)

# Visualize the relationship between windspeed bins, Ntd, and D
plt.figure(figsize=(10, 6))
plt.scatter(grouped_data['D'], grouped_data['Ntd'], 
            c=grouped_data['Windspeed_bin'], cmap='viridis', s=100, label='Windspeed Bin')
plt.colorbar(label='Windspeed Bin')
plt.xlabel('Average Slope (D)', fontsize=14, fontweight='bold')
plt.ylabel('Average Ntd', fontsize=14, fontweight='bold')
plt.title('Average Slope and Ntd for Each Windspeed Bin', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.show()

from sklearn.cluster import KMeans

## can calculate performance of clustering, SSE error, elbow method, we want
##small error, every number of clusters has a different about of error, i want 
##smallest error for smallest nuber of clusters 


windspeed_data = df_with_windspeed['Windspeed'].values.reshape(-1, 1)

# Apply K-means clustering
num_clusters = 5  # Adjust number of clusters based on data
kmeans = KMeans(n_clusters=num_clusters)
df_with_windspeed['Windspeed_cluster'] = kmeans.fit_predict(windspeed_data)

# Visualize cluster assignments
plt.figure(figsize=(10, 6))
plt.scatter(df_with_windspeed['Windspeed'], df_with_windspeed['Ntd'], 
            c=df_with_windspeed['Windspeed_cluster'], cmap='viridis', s=100)
plt.colorbar(label='Windspeed Cluster')
plt.xlabel('Windspeed (m/s)', fontsize=14, fontweight='bold')
plt.ylabel('Ntd', fontsize=14, fontweight='bold')
plt.title('Clustered Windspeed vs. Ntd', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.show()


# Scatter plot with linear fit for each windspeed bin
sns.lmplot(x='D', y='Ntd', hue='Windspeed_bin', data=df_with_windspeed, 
           palette='viridis', ci=None, height=6, aspect=1.5)
plt.xscale('log')
plt.yscale('log')
plt.title('Linear Fit for Each Windspeed Bin', fontsize=14, fontweight='bold')
plt.xlabel('Slope (D)', fontsize=14, fontweight='bold')
plt.ylabel('Total droplet concentration (Ntd)', fontsize=14, fontweight='bold')
plt.show()
#%%

# Define range of clusters to test
cluster_range = range(1, 11)  # Try from 1 to 10 clusters

# List to store SSE values for each number of clusters
sse = []

# Reshape the data for K-means
windspeed_data = df_with_windspeed['Windspeed'].values.reshape(-1, 1)

# Loop over cluster range to compute SSE
for num_clusters in cluster_range:
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(windspeed_data)
    sse.append(kmeans.inertia_)  # Inertia is the sum of squared distances to closest cluster center

# Plot SSE vs. number of clusters
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, sse, marker='o')
plt.xlabel('Number of Clusters', fontsize=14, fontweight='bold')
plt.ylabel('SSE (Sum of Squared Errors)', fontsize=14, fontweight='bold')
plt.title('Elbow Method to Determine Optimal Number of Clusters', fontsize=14, fontweight='bold')
plt.show()
#%%
# Set the optimal number of clusters from the elbow plot
optimal_clusters = 4  # Replace this with the number you identify from the elbow method

# Apply K-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=0)
df_with_windspeed['Windspeed_cluster'] = kmeans.fit_predict(windspeed_data)

# Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(df_with_windspeed['Windspeed'], df_with_windspeed['Ntd'], 
            c=df_with_windspeed['Windspeed_cluster'], cmap='viridis', s=100)
plt.colorbar(label='Windspeed Cluster')
plt.xlabel('Windspeed (m/s)', fontsize=14, fontweight='bold')
plt.ylabel('Ntd', fontsize=14, fontweight='bold')
plt.title('Clustered Windspeed vs. Ntd', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.show()
#%%
# post k-means clustering 
# problematic_legs = [
#     ('2022-01-11', 8),
#     ('2022-01-11', 9),
#     ('2022-01-11', 10),
#     ('2022-02-26', 5),
#     ('2022-02-26', 6),
#     ('2022-05-05', 1),
#     ('2022-05-05', 3),
#     ('2022-05-05', 7),
#     ('2022-05-10', 0),
#     ('2022-06-10', 1)
# ]

problematic_legs = [
    ('2022-01-11', 1), ('2022-01-11', 2), ('2022-01-11', 3), ('2022-01-11', 4),
    ('2022-01-11', 8), ('2022-01-11', 9), ('2022-01-11', 14), ('2022-01-11', 16),
    ('2022-01-11', 17),('2022-01-26', 16), ('2022-01-26', 14), ('2022-01-26', 12), 
    ('2022-01-26', 15),
    ('2022-01-26', 13), ('2022-01-26', 11), ('2022-03-29', 0), ('2022-05-05', 3),
    ('2022-05-05', 7), ('2022-05-05', 3), ('2022-06-11', 4), ('2022-03-29', 9), ('2022-03-29', 9), 
    ('2022-03-29', 0), ('2022-03-29', 6), ('2022-03-28', 4), ('2022-03-28', 0), ('2022-03-26', 15),
    ('2022-03-26', 6), ('2022-03-26', 0), ('2022-03-13', 8)]
problematic_set = set(problematic_legs)

# Define the function to compute log-transformed size distribution
def log_size_distribution(x, dryint, D):
    return np.log(dryint * np.exp(-x / D))

# Define your common bins
common_bins = np.linspace(0, 25, 20)

# Define your windspeed ranges
windspeed_bins = [(0, 4), (4.1, 7), (7.1, 11), (11.1, np.inf)]

grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Iterate through the entries in filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Match by index

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']  # Extract Leg_index

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']  # Extract D value
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # x-axis values for this leg
    dryint = entry_dryintercept['dry intercept']  # Extract dry intercept

    # Find the corresponding entry in df_combined based on Date, Min_start, and Min_end
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    # If there's a matching windspeed entry
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]

        # Interpolate the log-transformed size distribution for this leg
        interp_func = interp1d(ddry_values, log_size_distribution(ddry_values, dryint, D), kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        # Exponentiate the interpolated values to get back to the original scale for plotting
        interpolated_leg_values = np.exp(interpolated_leg_values)

        # Categorize the leg based on windspeed range
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_distributions[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break

# Average and plot the results
plt.figure(figsize=(12, 8))
for idx, ranges in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])  # Count the number of legs
        plt.plot(common_bins, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")

plt.yscale('log')  # Set y-axis to log scale
plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Size Distribution by Windspeed (Log Scale, Excluding Problematic Legs)', fontweight='bold')
plt.legend(title="Windspeed")
plt.tight_layout()
plt.show()















































































#%%
problematic_set = set(problematic_legs)

# Define your common bins
common_bins = np.linspace(0, 25, 20)

# Define the number of quantile bins
num_bins = 3  # Change this to adjust the number of bins

# Create windspeed bins using quantiles
df_with_windspeed['Windspeed_bin'] = pd.qcut(df_with_windspeed['Windspeed'], q=num_bins, labels=False)

# Optional: Get bin edges for reference
bin_edges = df_with_windspeed['Windspeed'].quantile(np.linspace(0, 1, num_bins + 1)).values
print("Bin edges:", bin_edges)

# Group by windspeed bin and calculate means
grouped_data = df_with_windspeed.groupby('Windspeed_bin').agg({'Ntd': 'mean', 'D': 'mean'}).reset_index()

# Plot averaged results by windspeed bin
plt.figure(figsize=(10, 8))
plt.scatter(grouped_data['D'], grouped_data['Ntd'], c='blue', s=100)
plt.xscale('log')
plt.yscale('log')

# Add labels and title
plt.xlabel('Average Slope (D)', fontsize=14, fontweight='bold')
plt.ylabel('Average Total droplet concentration (Ntd)', fontsize=14, fontweight='bold')
plt.title('Average Ntd vs Average D by Windspeed Quantile Bins', fontsize=14, fontweight='bold')
plt.show()

# Initialize storage for distributions and mean wind speeds
grouped_distributions = {i: [] for i in range(num_bins)}
mean_windspeeds = {i: [] for i in range(num_bins)}
std_distributions = {i: [] for i in range(num_bins)}  # To store standard deviations

# Iterate through the entries in filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Match by index

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']  # Extract Leg_index

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']  # Extract D value
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # x-axis values for this leg
    dryint = entry_dryintercept['dry intercept']  # Extract dry intercept

    # Find the corresponding entry in df_combined based on Date, Min_start, and Min_end
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) & 
        (df_combined['BCB_start'] == BCB_start) & 
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    # Check if windspeed_entry is not empty
    if windspeed_entry.empty:
        print(f"No matching windspeed entry found for Date: {date}, BCB_start: {BCB_start}, BCB_end: {BCB_stop}")
        continue  # Skip to the next iteration if no match found

    windspeed = windspeed_entry['Windspeed'].values[0]

    # Interpolate the log-transformed size distribution for this leg
    interp_func = interp1d(ddry_values, log_size_distribution(ddry_values, dryint, D), 
                           kind='linear', fill_value='extrapolate')
    interpolated_leg_values = interp_func(common_bins)

    # Exponentiate the interpolated values to get back to the original scale for plotting
    interpolated_leg_values = np.exp(interpolated_leg_values)

    # Categorize the leg based on windspeed bin
    # Safely find the windspeed bin
    windspeed_bin = pd.qcut(df_with_windspeed['Windspeed'], q=num_bins, labels=False, duplicates='drop')
    windspeed_bin_index = windspeed_bin[windspeed_bin.index[df_with_windspeed['Windspeed'] == windspeed]].values

    if len(windspeed_bin_index) == 0:
        print(f"No windspeed bin found for windspeed: {windspeed}")
        continue  # Skip if no windspeed bin is found

    windspeed_bin = windspeed_bin_index[0]

    if windspeed_bin in grouped_distributions:
        grouped_distributions[windspeed_bin].append(interpolated_leg_values)
        mean_windspeeds[windspeed_bin].append(windspeed)
        std_distributions[windspeed_bin].append(np.std(interpolated_leg_values))  # Store std deviation

# Average and plot the results
plt.figure(figsize=(12, 8))
for idx in range(num_bins):
    if grouped_distributions[idx]:
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)
        avg_windspeed = np.mean(mean_windspeeds[idx]) if mean_windspeeds[idx] else 0
        num_legs = len(grouped_distributions[idx])  # Count the number of legs

        # Calculate standard deviation across the distributions for uncertainty shading
        std_dev = np.std(grouped_distributions[idx], axis=0)

        # Plot the average distribution
        plt.plot(common_bins, avg_distribution, label=f"Bin {idx + 1}: {avg_windspeed:.1f} m/s, n={num_legs} legs")

        # Add shaded area for uncertainty (mean ± std dev)
        plt.fill_between(common_bins, 
                         avg_distribution - std_dev, 
                         avg_distribution + std_dev, 
                         alpha=0.2)  # Adjust alpha for transparency

plt.yscale('log')  # Set y-axis to log scale
plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Size Distribution by Windspeed Quantile Bins (Log Scale, Excluding Problematic Legs)', fontweight='bold')
plt.legend(title="Windspeed Bins")
plt.tight_layout()
plt.show()
#%%
#How windspeed slope and NT correlate with windspeed
Z0 = 0.03  # meters (typical value for open terrain)
Z10 = 10  # target height m

# Function to apply windspeed correction
def correct_windspeed(windspeed, altitude):
    return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))

# Initialize dictionary to hold combined data
combined_data = {
    'Date': [],
    'BCB_start': [],
    'BCB_stop': [],
    'filtered_NT': [],
    'D': [],
    'Windspeed': []
}

# Loop through master_Min to retrieve windspeed and altitude data
for i, flight in enumerate(master_BCB):
    for j, wind_alt in enumerate(flight):
        try:
            # Extract Date, windspeed, and altitude from master_Min
            date = wind_alt['Date']
            wind_mean = wind_alt['Winds_mean'][0]
            alt_mean = wind_alt['Alts_mean'][0]
            BCB_start = wind_alt['BCB_start']
            BCB_stop = wind_alt['BCB_end']
            
            # Correct windspeed using the provided formula
            if not np.isnan(alt_mean) and alt_mean > 0:
                corrected_windspeed = correct_windspeed(wind_mean, alt_mean)
            else:
                corrected_windspeed = np.nan

            # Initialize variables to store found data
            filtered_NT = None
            D = None

            # Find corresponding exponential parameters (D) from master_min_exponential
            if date in master_BCB_exponential:
                for exp_params in master_BCB_exponential[date]:
       
                    if exp_params['BCB_start'] == BCB_start and exp_params['BCB_stop'] == BCB_stop:
                        D = exp_params['D']
                        print(f"Found D for Date: {date}, BCB_start: {BCB_start}, BCB_stop: {BCB_stop}, D: {D}")
                        break

            # Find corresponding Ntd from filtered_master_min_ntd
            for filtered_NT_data in filtered_master_BCB_NT:
                if filtered_NT_data['Date'] == date and filtered_NT_data['BCB_start'] == BCB_start and filtered_NT_data['BCB_stop'] == BCB_stop:
                    filtered_NT = filtered_NT_data['filtered_NT']
                    break

            # Append data if both Ntd and D are found
            if filtered_NT is not None and D is not None:
                combined_data['Date'].append(date)
                combined_data['BCB_start'].append(BCB_start)
                combined_data['BCB_stop'].append(BCB_stop)
                combined_data['filtered_NT'].append(filtered_NT)
                combined_data['D'].append(D)
                combined_data['Windspeed'].append(corrected_windspeed)

        except IndexError as e:
            print(f"Index error at i={i}, j={j}: {e}")
            continue

# Convert the combined data to a DataFrame
df_combined = pd.DataFrame(combined_data)

# Check for anomalies or missing values
print(df_combined.describe())

# Separate data based on NaN Windspeed
df_with_windspeed = df_combined[df_combined['Windspeed'].notna()]
df_nan_windspeed = df_combined[df_combined['Windspeed'].isna()]

# Plot: Ntd vs D, colored by Windspeed
plt.figure(figsize=(10, 8))

# Plot data with valid Windspeed values
sc = plt.scatter(df_with_windspeed['D'], df_with_windspeed['filtered_NT'], 
                 c=df_with_windspeed['Windspeed'], cmap='viridis', s=100, label='Windspeed')

# Plot data with NaN Windspeed values in black
plt.scatter(df_nan_windspeed['D'], df_nan_windspeed['filtered_NT'], 
            color='black', s=100, label='Windspeed NaN')

# Add color bar
plt.colorbar(sc, label='Corrected Windspeed (m/s)')

# Add labels and title
plt.xlabel('Slope (D)', fontsize=14, fontweight='bold')
plt.ylabel('Ambient Concentration', fontsize=14, fontweight='bold')
plt.title('Below cloud base January - June 2022 w/ clear sky restrictions', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.xscale('log')

# Show the plot
plt.show()
#%%
#Ambient distribution

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.hist(df_with_windspeed['Windspeed'], bins=30, color='skyblue', edgecolor='black')
plt.title('Windspeed Distribution')
plt.xlabel('Windspeed (m/s)')
plt.ylabel('Frequency')

plt.subplot(3, 1, 2)
plt.hist(df_with_windspeed['filtered_NT'], bins=30, color='lightgreen', edgecolor='black')
plt.title('Ambient Distribution')
plt.xlabel('Ambient')
plt.ylabel('Frequency')
plt.yscale('log')

plt.subplot(3, 1, 3)
plt.hist(df_with_windspeed['D'], bins=30, color='lightcoral', edgecolor='black')
plt.title('Slope Distribution')
plt.xlabel('Slope')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
# %%
# post k-means clustering 
problematic_legs = [
       ('2022-01-11', 1),
    ('2022-01-11', 2),
    ('2022-01-11', 3),
    ('2022-01-11', 4),
    ('2022-01-11', 8),
    ('2022-01-11', 9),
    ('2022-01-11', 14),
    ('2022-01-11', 16),
    ('2022-01-11', 17),
    ('2022-01-26', 14),
    ('2022-01-26', 12),
    ('2022-01-26', 15),
    ('2022-01-26', 13),
    ('2022-01-26', 11),
    ('2022-03-29', 0),
    ('2022-05-05', 3),
    ('2022-05-05', 7),
]
problematic_set = set(problematic_legs)

# Define the function to compute log-transformed size distribution
def log_size_distribution(x, dryint, D):
    return np.log(dryint * np.exp(-x / D))

# Define your common bins
common_bins = np.linspace(0, 25, 20)

# Define your windspeed ranges
windspeed_bins = [(0, 4), (4.1, 7), (7.1, 11), (11.1, np.inf)]

grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Iterate through the entries in filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Match by index

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']  # Extract Leg_index

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']  # Extract D value
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # x-axis values for this leg
    dryint = entry_dryintercept['dry intercept']  # Extract dry intercept

    # Find the corresponding entry in df_combined based on Date, Min_start, and Min_end
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    # If there's a matching windspeed entry
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]

        # Interpolate the log-transformed size distribution for this leg
        interp_func = interp1d(ddry_values, log_size_distribution(ddry_values, dryint, D), kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        # Exponentiate the interpolated values to get back to the original scale for plotting
        interpolated_leg_values = np.exp(interpolated_leg_values)

        # Categorize the leg based on windspeed range
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_distributions[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break

# Average and plot the results
plt.figure(figsize=(12, 8))
for idx, ranges in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])  # Count the number of legs
        
        # Calculate percentiles for shading
        lower_percentile = np.percentile(grouped_distributions[idx], 25, axis=0)
        upper_percentile = np.percentile(grouped_distributions[idx], 75, axis=0)

        # Plot the average distribution
        plt.plot(common_bins, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")

        # Plot the shaded area for the interquartile range
        plt.fill_between(common_bins, lower_percentile, upper_percentile, alpha=0.4)

        # Optionally plot the percentile lines
        plt.plot(common_bins, lower_percentile, linestyle='--', linewidth=1, color='gray')
        plt.plot(common_bins, upper_percentile, linestyle='--', linewidth=1, color='gray')

plt.yscale('log')  # Set y-axis to log scale
plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Size Distribution by Windspeed (Log Scale, Excluding Problematic Legs)', fontweight='bold')
plt.legend(title="Windspeed")
plt.tight_layout()
plt.show()
# %%

























#%%
###Trying Katie's method where we average across all bins and then fit
##a distrubtion using the average 

# Step 1: Define the problematic legs (if not already defined)
problematic_legs = [
    ('2022-01-11', 1), ('2022-01-11', 2), ('2022-01-11', 3), ('2022-01-11', 4),
    ('2022-01-11', 8), ('2022-01-11', 9), ('2022-01-11', 14), ('2022-01-11', 16),
    ('2022-01-11', 17), ('2022-01-26', 14), ('2022-01-26', 12), ('2022-01-26', 15),
    ('2022-01-26', 13), ('2022-01-26', 11), ('2022-03-29', 0), ('2022-05-05', 3),
    ('2022-05-05', 7),
]
problematic_set = set(problematic_legs)

# Define your common bins
common_bins = np.linspace(0, 25, 20)

# Dictionary to store droplet concentrations for each windspeed range
windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]
grouped_concentrations = {i: [] for i in range(len(windspeed_bins))}  # To accumulate droplet concentrations
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Iterate through the entries in filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Match by index

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']  # Extract Leg_index

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']  # Extract D value
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # x-axis values for this leg
    dryint = entry_dryintercept['dry intercept']  # Extract dry intercept

    # Find the corresponding entry in df_combined based on Date, Min_start, and Min_end
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    
    # If there's a matching windspeed entry
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]
        
        # Interpolate the size distribution for this leg
        size_dist = size_distribution(ddry_values, dryint, D)
        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)
        
        # Categorize the leg based on windspeed range
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_concentrations[idx].append(interpolated_leg_values)  # Add size distribution to this bin
                mean_windspeeds[idx].append(windspeed)
                break

# Step 2: Average each particle size bin, compute standard deviation for error bars, then fit
plt.figure(figsize=(12, 8))

# Loop through each windspeed bin and compute averages
for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_concentrations[idx]:
        # Convert the list of arrays to a numpy array for easier calculations
        concentrations_array = np.array(grouped_concentrations[idx])
        
        # Average each size bin (column) across all size distributions
        avg_concentration = np.mean(concentrations_array, axis=0)
        
        # Calculate the standard deviation for error bars
        std_dev = np.std(concentrations_array, axis=0)
        
        # Handle zeros in avg_concentration for log scale plotting
        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)  # Replace zeros with small value
        
        # Fit the average size distribution using curve_fit
        def fit_function(x, dryint, D):
            return dryint * np.exp(-x / D)
        
        popt, _ = curve_fit(fit_function, common_bins, avg_concentration, p0=[1, 1])  # Initial guess for dryint, D
        
        # Generate fitted curve for plotting
        fitted_curve = fit_function(common_bins, *popt)
        
        # Calculate average windspeed and number of legs in this bin
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_concentrations[idx])
        
        # Plot the fitted curve
        plt.plot(common_bins, fitted_curve, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")
        
        # Add error bars (standard deviation) to the average concentration plot
        plt.errorbar(common_bins, avg_concentration, yerr=std_dev, fmt='o', capsize=5, label=f"{low}-{high} m/s")

# Set y-axis to log scale
plt.yscale('log')

plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Fitted Size Distribution by Windspeed (Excluding Problematic Legs)', fontweight='bold')
plt.legend(title="Windspeed")
plt.tight_layout()
plt.show()
#%%

#%%
#changing the number of bins being average over to 10
## Define common bins from 0 to 10 µm 

common_bins = np.linspace(0, 10, 10)  # Common x-axis values

# Step 2: Initialize dictionaries to store droplet concentrations for each windspeed range
windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]
grouped_concentrations = {i: [] for i in range(len(windspeed_bins))}  # To accumulate droplet concentrations
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Step 3: Iterate through filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])
    dryint = entry_dryintercept['dry intercept']

    # Find the corresponding windspeed entry in df_combined
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]

        # Interpolate the size distribution for this leg
        size_dist = size_distribution(ddry_values, dryint, D)  # Compute size distribution
        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)  # Interpolate to common bins

        # Categorize the leg based on windspeed range
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_concentrations[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break

# Step 4: Average each particle size bin for plotting before fitting
plt.figure(figsize=(12, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_concentrations[idx]:
        # Convert the list of arrays to a numpy array for easier calculations
        concentrations_array = np.array(grouped_concentrations[idx])
        
        # Average each size bin (column) across all size distributions
        avg_concentration = np.mean(concentrations_array, axis=0)
        
        # Calculate standard deviation for error bars
        std_dev = np.std(concentrations_array, axis=0)
        
        # Handle zeros in avg_concentration for log scale plotting
        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)  # Replace zeros with small value
        
        # Calculate average windspeed and number of legs in this bin
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_concentrations[idx])
        
        # Plot the average concentration with error bars and connecting lines
        plt.errorbar(common_bins, avg_concentration, yerr=std_dev, fmt='o', capsize=5, 
                     label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linestyle='-', alpha=0.7)

# Set y-axis to log scale
plt.yscale('log')

plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Average Size Distribution by Windspeed (Excluding Problematic Legs)', fontweight='bold')
plt.legend(title="Average Windspeed")
plt.tight_layout()
plt.show()

# for idx, (low, high) in enumerate(windspeed_bins):
#     if grouped_concentrations[idx]:
#         # Convert the list of arrays to a numpy array for easier calculations
#         concentrations_array = np.array(grouped_concentrations[idx])
        
#         # Average each size bin (column) across all size distributions
#         avg_concentration = np.mean(concentrations_array, axis=0)
        
#         # Calculate standard deviation for error bars
#         std_dev = np.std(concentrations_array, axis=0)
        
#         # Handle zeros in avg_concentration for log scale plotting
#         avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)  # Replace zeros with small value
        
#         # Calculate average windspeed and number of legs in this bin
#         avg_windspeed = np.mean(mean_windspeeds[idx])
#         num_legs = len(grouped_concentrations[idx])
        
#         # Plot the average concentration with error bars
#         plt.errorbar(common_bins, avg_concentration, yerr=std_dev, fmt='o', capsize=5, 
#                      label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", alpha=0.7)

# # Set y-axis to log scale
# plt.yscale('log')

# plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
# plt.xlabel('Bin diameter (µm)', fontweight='bold')
# plt.title('Average Size Distribution by Windspeed (Excluding Problematic Legs)', fontweight='bold')
# plt.legend(title="Average Windspeed")
# plt.tight_layout()
# plt.show()
#%%
##trying to fix size distribution and add ntd 

# Function for calculating the size distribution
def size_distribution(x, dryint, D):
    return dryint * np.exp(-x / D)

# Prepare your common bins
common_bins = np.linspace(0, 25, 20)

# This will hold the interpolated values along with their metadata
interpolated_values = []

# Iterate through entries in filtered_master_BCB_ddry
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Directly match by index
    
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']  # Extract Leg_index
    D = entry_ddry['D']  # Extract D value
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # x-axis values for this leg
    dryint = entry_dryintercept['dry intercept']  # Use the matched N0 value from the same index

    # Interpolate the distribution
    interp_func = interp1d(ddry_values, size_distribution(ddry_values, dryint, D), kind='linear', fill_value='extrapolate')
    interpolated_leg_values = interp_func(common_bins)

    # Create a unique key for matching with Ntd
    key = (date, BCB_start, BCB_stop)

    # Find the corresponding Ntd value from filtered_master_BCB_ntd_dict
    if key in filtered_master_BCB_ntd_dict:
        Ntd_value = filtered_master_BCB_ntd_dict[key]['Ntd']
    else:
        Ntd_value = np.nan  # Use NaN or some other placeholder if no match is found

    # Store the interpolated values with metadata
    interpolated_values.append({
        'Date': date,
        'Leg_index': leg_index,
        'BCB_start': BCB_start,
        'BCB_stop': BCB_stop,
        'interpolated_values': interpolated_leg_values.tolist(),
        'Ntd': Ntd_value  # Store the Ntd value
    })

# Plot the results
plt.figure(figsize=(12, 8))

# Iterate through entries in interpolated_values and plot each distribution
for entry in interpolated_values:
    date = entry['Date']
    leg_index = entry['Leg_index']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    interpolated_leg_values = entry['interpolated_values']
    Ntd_value = entry['Ntd']
    
    # Use Ntd_value for y-axis instead of interpolated_values if needed
    if not np.isnan(Ntd_value):  # Only plot if Ntd is available
        plt.plot(common_bins, [Ntd_value] * len(common_bins), label=f"Date: {date}, Leg: {leg_index} ({BCB_start} - {BCB_stop})")

# Add labels and legend
plt.ylabel('Total Number Concentration (Ntd) (/cm^3)', fontweight='bold')
plt.xlabel('Bin diameter (um)', fontweight='bold')
plt.title('Below cloud base total concentration January-June 2022', fontweight='bold')
plt.tight_layout()
plt.show()
#%%

# Define the size distribution function (exponential decay)
def size_distribution(d, dryint, D):
    return dryint * np.exp(-d / D)

# Integration limits
dmin = 2  # Lower limit of the integration (in micrometers)
dmax = np.inf  # Upper limit (infinity)

# Prepare an array to store total concentrations
integrated_concentrations = []

# Iterate through entries in filtered_master_BCB_ddry
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Directly match by index
    
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']  # Extract Leg_index
    D = entry_ddry['D']  # Extract D value
    dryint = entry_dryintercept['dry intercept']  # Use the matched N0 value from the same index
    
    # Perform the integration from dmin to infinity
    total_concentration, _ = quad(size_distribution, dmin, dmax, args=(dryint, D))
    
    # Store the integrated total concentration with metadata
    integrated_concentrations.append({
        'Date': date,
        'Leg_index': leg_index,
        'BCB_start': BCB_start,
        'BCB_stop': BCB_stop,
        'Total Concentration': total_concentration
    })

# Print the integrated total concentrations for each leg
for total in integrated_concentrations:
    print(f"Date: {total['Date']}, Leg_index: {total['Leg_index']}, "
          f"Total Concentration: {total['Total Concentration']:.3f} /cm^3")

# Plotting total concentrations as a bar plot (or any preferred plot type)
dates = [entry['Date'] for entry in integrated_concentrations]
legs = [entry['Leg_index'] for entry in integrated_concentrations]
total_concs = [entry['Total Concentration'] for entry in integrated_concentrations]

plt.figure(figsize=(12, 8))
plt.bar(range(len(total_concs)), total_concs, tick_label=[f"{d}-{l}" for d, l in zip(dates, legs)])
plt.ylabel('Total Concentration (/cm^3)', fontweight='bold')
plt.xlabel('Leg (Date-Leg)', fontweight='bold')
plt.title('Total Below-Cloud-Base Concentration (BCB)', fontweight='bold')
plt.tight_layout()
plt.show()
#%%

# Define the size distribution function (exponential decay)
def size_distribution(d, dryint, D):
    return dryint * np.exp(-d / D)

# Integration limits
dmin = 2  # Lower limit of the integration (in micrometers)
dmax = np.inf  # Upper limit (infinity)

# Prepare common bins for plotting the size distribution
common_bins = np.linspace(0, 25, 20)

# Prepare an array to store total concentrations and size distributions
integrated_concentrations = []
size_distributions = []

# Iterate through entries in filtered_master_BCB_ddry
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Directly match by index
    
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']  # Extract Leg_index
    D = entry_ddry['D']  # Extract D value
    dryint = entry_dryintercept['dry intercept']  # Use the matched N0 value from the same index
    
    # Perform the integration from dmin to infinity for total concentration
    total_concentration, _ = quad(size_distribution, dmin, dmax, args=(dryint, D))
    
    # Calculate the size distribution at common_bins
    distribution_values = size_distribution(common_bins, dryint, D)
    
    # Store the results
    integrated_concentrations.append({
        'Date': date,
        'Leg_index': leg_index,
        'BCB_start': BCB_start,
        'BCB_stop': BCB_stop,
        'Total Concentration': total_concentration
    })
    
    size_distributions.append({
        'Date': date,
        'Leg_index': leg_index,
        'BCB_start': BCB_start,
        'BCB_stop': BCB_stop,
        'Distribution Values': distribution_values
    })

# Print the integrated total concentrations for each leg
for total in integrated_concentrations:
    print(f"Date: {total['Date']}, Leg_index: {total['Leg_index']}, "
          f"Total Concentration: {total['Total Concentration']:.3f} /cm^3")

# Plotting size distributions for each leg
plt.figure(figsize=(12, 8))

for dist in size_distributions:
    date = dist['Date']
    leg_index = dist['Leg_index']
    BCB_start = dist['BCB_start']
    BCB_stop = dist['BCB_stop']
    dist_values = dist['Distribution Values']
    
    plt.plot(common_bins, dist_values, label=f"Date: {date}, Leg: {leg_index} ({BCB_start} - {BCB_stop})")

# Add labels and legend for the size distribution plot
plt.ylabel('Concentration (/cm^3/um)', fontweight='bold')
plt.xlabel('Bin diameter (um)', fontweight='bold')
plt.title('Size Distribution Below Cloud Base (January - June 2022)', fontweight='bold')
plt.tight_layout()
plt.show()
#%%

# Define the size distribution function
def size_distribution(ddry, N0, D):
    return N0 * np.exp(-ddry / D)

# Prepare the common bins for ddry (x-axis)
common_bins = np.linspace(0, 25, 20)

# Create a new figure for plotting
plt.figure(figsize=(12, 8))

# Loop through each entry in the filtered_master_BCB_ddry dataset
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_ntd = filtered_master_BCB_ntd[i]  # Match Ntd by index
    
    # Extract data for the current leg
    date = entry_ddry['Date']
    leg_index = entry_ddry['Leg_index']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    D = entry_ddry['D']  # E-folding diameter
    Ntd = entry_ntd['Ntd']  # Total concentration (Ntd)
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # ddry values for this leg
    N0 = entry_ddry['n0']  # Use the matched dry intercept value for N0
    
    # Calculate the size distribution for each ddry value using dryintercept as N0
    size_dist_values = size_distribution(ddry_values, dryintercept, D)
    
    # Interpolate the size distribution to match the common bins
    interp_func = interp1d(ddry_values, size_dist_values, kind='linear', fill_value='extrapolate')
    interpolated_leg_values = interp_func(common_bins)
    
    # Plot the interpolated size distribution
    plt.plot(common_bins, interpolated_leg_values, label=f"Date: {date}, Leg: {leg_index} ({BCB_start} - {BCB_stop})")

# Add labels and legend to the plot
plt.ylabel('Concentration N(d) (/cm^3/μm)', fontweight='bold')
plt.xlabel('Dry diameter (μm)', fontweight='bold')
plt.title('Below Cloud Base Dry Size Distribution (January-June 2022)', fontweight='bold')
plt.tight_layout()
plt.show()
#%%

# Define the size distribution function CORRECT
def size_distribution(ddry, N0, D):
    return N0 * np.exp(-ddry / D)

# Prepare the common bins for ddry (x-axis)
common_bins = np.linspace(0, 10, 10)

# Initialize dictionaries to store droplet concentrations for each windspeed range
windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]
grouped_concentrations = {i: [] for i in range(len(windspeed_bins))}  # To accumulate droplet concentrations
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Loop through each entry in the filtered_master_BCB_ddry dataset
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']
    D = entry_ddry['D']  # E-folding diameter
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # ddry values for this leg
    N0 = entry_ddry['n0']  # Use the matched dry intercept value for N0

    # Find the corresponding windspeed entry in df_combined
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]

        # Calculate the size distribution for each ddry value using dryintercept as N0
        size_dist_values = size_distribution(ddry_values, N0, D)
        
        # Check for NaN values in the size distribution
        if np.any(np.isnan(size_dist_values)):
            continue  # Skip this leg if any NaN values are present
        
        # Interpolate the size distribution to match the common bins
        interp_func = interp1d(ddry_values, size_dist_values, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)  # Interpolate to common bins

        # Categorize the leg based on windspeed range
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_concentrations[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break

# Step 4: Average each particle size bin for plotting before fitting
plt.figure(figsize=(12, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_concentrations[idx]:
        # Convert the list of arrays to a numpy array for easier calculations
        concentrations_array = np.array(grouped_concentrations[idx])
        
        # Average each size bin (column) across all size distributions
        avg_concentration = np.mean(concentrations_array, axis=0)
        
        # Calculate standard deviation for error bars
        std_dev = np.std(concentrations_array, axis=0)
        
        # Handle zeros in avg_concentration for log scale plotting
        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)  # Replace zeros with small value
        
        # Calculate average windspeed and number of legs in this bin
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_concentrations[idx])
        
        # Plot the average concentration with error bars and connecting lines
        plt.errorbar(common_bins, avg_concentration, yerr=std_dev, fmt='o', capsize=5, 
                     label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linestyle='-', alpha=0.7)

# Set y-axis to log scale
plt.yscale('log')

plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Average Size Distribution by Windspeed (Excluding Problematic Legs)', fontweight='bold')
plt.legend(title="Average Windspeed")
plt.tight_layout()
plt.show()
#%%
def fit_function(x, D):
    return dryint * np.exp(-x / D)

# Step 2: Initialize dictionaries to store droplet concentrations for each windspeed range
common_bins = np.linspace(0, 10, 10)  # Common x-axis values
windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]
grouped_concentrations = {i: [] for i in range(len(windspeed_bins))}  # To accumulate droplet concentrations
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Step 3: Iterate through filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])
    dryint = entry_dryintercept['dry intercept']

    # Find the corresponding windspeed entry in df_combined
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]

        # Interpolate the size distribution for this leg
        size_dist = size_distribution(ddry_values, dryint, D)  # Compute size distribution
        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)  # Interpolate to common bins

        # Categorize the leg based on windspeed range
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_concentrations[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break

# Step 4: Average each particle size bin for plotting and fit the custom exponential model
plt.figure(figsize=(12, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_concentrations[idx]:
        # Convert the list of arrays to a numpy array for easier calculations
        concentrations_array = np.array(grouped_concentrations[idx])
        
        # Average each size bin (column) across all size distributions
        avg_concentration = np.mean(concentrations_array, axis=0)
        
        # Calculate standard deviation for error bars
        std_dev = np.std(concentrations_array, axis=0)
        
        # Handle zeros in avg_concentration for log scale plotting
        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)  # Replace zeros with small value
        
        # Calculate average windspeed and number of legs in this bin
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_concentrations[idx])
        
        # Plot the average concentration with error bars and connecting lines
        plt.errorbar(common_bins, avg_concentration, yerr=std_dev, fmt='o', capsize=5, 
                     label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linestyle='-', alpha=0.7)

        # Step 5: Fit your custom exponential model to the average concentrations
        try:
            popt, pcov = curve_fit(fit_function, common_bins, avg_concentration, p0=[1, 1])
            dryint_fit, D_fit = popt

            # Plot the fitted exponential line
            fitted_values = fit_function(common_bins, dryint_fit, D_fit)
            plt.plot(common_bins, fitted_values, linestyle='--', 
                     label=f"Fit {avg_windspeed:.1f} m/s: y = {dryint_fit:.2f} * exp(-x / {D_fit:.2f})")
            
            # Print the equation for each line
            print(f"Windspeed {avg_windspeed:.1f} m/s: y = {dryint_fit:.2f} * exp(-x / {D_fit:.2f})")

        except RuntimeError:
            print(f"Could not fit the exponential model for windspeed {avg_windspeed:.1f} m/s")
        
# Set y-axis to log scale
plt.yscale('log')

plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Average Size Distribution by Windspeed (Excluding Problematic Legs)', fontweight='bold')
plt.legend(title="Average Windspeed")
plt.tight_layout()
plt.show()
#%%

# Define the size distribution function
def size_distribution(ddry, N0, D):
    return N0 * np.exp(-ddry / D)

# Define the fitting function
def fit_function(x, N0, D):
    return N0 * np.exp(-x / D)

# Prepare the common bins for ddry (x-axis)
common_bins = np.linspace(0, 10, 10)

# Initialize dictionaries to store droplet concentrations for each windspeed range
windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]
grouped_concentrations = {i: [] for i in range(len(windspeed_bins))}  # To accumulate droplet concentrations
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Step 3: Iterate through filtered_master_BCB_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']
    N0 = entry_ddry['n0']
    D = entry_ddry['D']  # E-folding diameter
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # ddry values for this leg
    dryint = entry_dryintercept['dry intercept']  # Use the matched dry intercept value for N0

    # Find the corresponding windspeed entry in df_combined
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]

        # Calculate the size distribution for each ddry value using dryintercept as N0
        size_dist_values = size_distribution(ddry_values, dryint, D)  # Compute size distribution

        # Check for NaN values in the size distribution
        if np.any(np.isnan(size_dist_values)):
            continue  # Skip this leg if any NaN values are present

        # Interpolate the size distribution for this leg
        interp_func = interp1d(ddry_values, size_dist_values, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)  # Interpolate to common bins

        # Categorize the leg based on windspeed range
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_concentrations[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break

# Step 4: Average each particle size bin for plotting and fit the custom exponential model
plt.figure(figsize=(12, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_concentrations[idx]:
        # Convert the list of arrays to a numpy array for easier calculations
        concentrations_array = np.array(grouped_concentrations[idx])
        
        # Average each size bin (column) across all size distributions
        avg_concentration = np.mean(concentrations_array, axis=0)
        
        # Calculate standard deviation for error bars
        std_dev = np.std(concentrations_array, axis=0)
        
        # Handle zeros in avg_concentration for log scale plotting
        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)  # Replace zeros with small value
        
        # Calculate average windspeed and number of legs in this bin
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_concentrations[idx])
        
        # Plot the average concentration with error bars and connecting lines
        plt.errorbar(common_bins, avg_concentration, yerr=std_dev, fmt='o', capsize=5, 
                     label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linestyle='-', alpha=0.7)

        # Step 5: Fit your custom exponential model to the average concentrations
        try:
            popt, pcov = curve_fit(fit_function, common_bins, avg_concentration, p0=[1, 1])
            dryint_fit, D_fit = popt

            # Plot the fitted exponential line
            fitted_values = fit_function(common_bins, dryint_fit, D_fit)
            plt.plot(common_bins, fitted_values, linestyle='--', 
                     label=f"Fit {avg_windspeed:.1f} m/s: y = {dryint_fit:.2f} * exp(-x / {D_fit:.2f})")
            
            # Print the equation for each line
            print(f"Windspeed {avg_windspeed:.1f} m/s: y = {dryint_fit:.2f} * exp(-x / {D_fit:.2f})")

        except RuntimeError:
            print(f"Could not fit the exponential model for windspeed {avg_windspeed:.1f} m/s")

# Set y-axis to log scale
plt.yscale('log')

plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Average Size Distribution by Windspeed (Excluding Problematic Legs)', fontweight='bold')
plt.legend(title="Average Windspeed")
plt.tight_layout()
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

# Define the size distribution function
def size_distribution(ddry, N0, D):
    return N0 * np.exp(-ddry / D)

# Define the fitting function
def fit_function(x, N0, D):
    return N0 * np.exp(-x / D)

# Prepare the common bins for ddry (x-axis)
common_bins = np.linspace(0, 10, 10)

# Initialize dictionaries to store droplet concentrations for each windspeed range
windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]
grouped_concentrations = {i: [] for i in range(len(windspeed_bins))}  # To accumulate droplet concentrations
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Step 3: Iterate through filtered_master_BCB_ddry and use N0 directly
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']
    N0 = entry_ddry['n0']  # Use N0 directly from the entry
    D = entry_ddry['D']  # E-folding diameter
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # ddry values for this leg

    # Find the corresponding windspeed entry in df_combined
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]

        # Calculate the size distribution for each ddry value using N0
        size_dist_values = size_distribution(ddry_values, N0, D)  # Compute size distribution

        # Check for NaN values in the size distribution
        if np.any(np.isnan(size_dist_values)):
            continue  # Skip this leg if any NaN values are present

        # Interpolate the size distribution for this leg
        interp_func = interp1d(ddry_values, size_dist_values, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)  # Interpolate to common bins

        # Categorize the leg based on windspeed range
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_concentrations[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break

# Step 4: Average each particle size bin for plotting and fit the custom exponential model
# plt.figure(figsize=(12, 8))
plt.figure(figsize=(12, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_concentrations[idx]:
        # Convert the list of arrays to a numpy array for easier calculations
        concentrations_array = np.array(grouped_concentrations[idx])
        
        # Average each size bin (column) across all size distributions
        avg_concentration = np.mean(concentrations_array, axis=0)
        
        # Handle zeros in avg_concentration for log scale plotting
        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)  # Replace zeros with small value
        
        # Calculate average windspeed and number of legs in this bin
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_concentrations[idx])
        
        # Plot the average concentration without error bars
        plt.plot(common_bins, avg_concentration, 'o-', label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", alpha=0.7)

        # Step 5: Fit your custom exponential model to the average concentrations
        try:
            popt, pcov = curve_fit(fit_function, common_bins, avg_concentration, p0=[1, 1])
            N0_fit, D_fit = popt

            # Plot the fitted exponential line
            fitted_values = fit_function(common_bins, N0_fit, D_fit)
            plt.plot(common_bins, fitted_values, linestyle='--', 
                     label=f"Fit {avg_windspeed:.1f} m/s: y = {N0_fit:.2f} * exp(-x / {D_fit:.2f})")
            
            # Print the equation for each line
            print(f"Windspeed {avg_windspeed:.1f} m/s: y = {N0_fit:.2f} * exp(-x / {D_fit:.2f})")

        except RuntimeError:
            print(f"Could not fit the exponential model for windspeed {avg_windspeed:.1f} m/s")

# Set y-axis to log scale
plt.yscale('log')

# for idx, (low, high) in enumerate(windspeed_bins):
#     if grouped_concentrations[idx]:
#         # Convert the list of arrays to a numpy array for easier calculations
#         concentrations_array = np.array(grouped_concentrations[idx])
        
#         # Average each size bin (column) across all size distributions
#         avg_concentration = np.mean(concentrations_array, axis=0)
        
#         # Calculate standard deviation for error bars
#         std_dev = np.std(concentrations_array, axis=0)
        
#         # Handle zeros in avg_concentration for log scale plotting
#         avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)  # Replace zeros with small value
        
#         # Calculate average windspeed and number of legs in this bin
#         avg_windspeed = np.mean(mean_windspeeds[idx])
#         num_legs = len(grouped_concentrations[idx])
        
#         # Plot the average concentration with error bars and connecting lines
#         plt.errorbar(common_bins, avg_concentration, yerr=std_dev, fmt='o', capsize=5, 
#                      label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linestyle='-', alpha=0.7)

#         # Step 5: Fit your custom exponential model to the average concentrations
#         try:
#             popt, pcov = curve_fit(fit_function, common_bins, avg_concentration, p0=[1, 1])
#             N0_fit, D_fit = popt

#             # Plot the fitted exponential line
#             fitted_values = fit_function(common_bins, N0_fit, D_fit)
#             plt.plot(common_bins, fitted_values, linestyle='--', 
#                      label=f"Fit {avg_windspeed:.1f} m/s: y = {N0_fit:.2f} * exp(-x / {D_fit:.2f})")
            
#             # Print the equation for each line
#             print(f"Windspeed {avg_windspeed:.1f} m/s: y = {N0_fit:.2f} * exp(-x / {D_fit:.2f})")

#         except RuntimeError:
#             print(f"Could not fit the exponential model for windspeed {avg_windspeed:.1f} m/s")

# # Set y-axis to log scale
# plt.yscale('log')

plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin  center diameter (µm)', fontweight='bold')
plt.title('Below cloud base January - June 2022', fontweight='bold')
plt.legend(title="Average wind speed binning ")
plt.tight_layout()
plt.show()
#%%
##trying to calculate area under the curve
##calculating area under the curve for total concentration
from scipy.integrate import quad  # Import for continuous integration
def fit_function(x, D):
    return N0 * np.exp(-x / D) 
common_bins = np.linspace(2.5, 10, 10)  # Adjust common bins to start from 2.5 um
windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]
grouped_concentrations = {i: [] for i in range(len(windspeed_bins))}  # Accumulate droplet concentrations
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    N0 = entry_ddry['n0']  # Get dry intercept (n0)

    # Extract necessary values
    date = entry_ddry['Date']
    leg_index = entry_ddry['Leg_index']
    D = entry_ddry['D']  # Get D from the entry
    ddry_values = np.array(entry_ddry['filtered_ddry'])

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    # Find corresponding windspeed entry in df_combined
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) & 
        (df_combined['Leg_index'] == leg_index)  # Assuming you can match by leg index
    ]

    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]

        # Interpolate the size distribution for this leg
        size_dist = fit_function(ddry_values, n0, D)  # Use n0 directly
        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)  # Interpolate to common bins

        # Categorize based on windspeed range
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_concentrations[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break
plt.figure(figsize=(12, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_concentrations[idx]:
        concentrations_array = np.array(grouped_concentrations[idx])  # Convert to numpy array
        
        # Average each size bin (column) across all size distributions
        avg_concentration = np.mean(concentrations_array, axis=0)
        
        # Standard deviation for error bars
        std_dev = np.std(concentrations_array, axis=0)
        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)  # Replace zeros
        
        # Average windspeed
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_concentrations[idx])

        # Plot the average concentration with error bars
        plt.errorbar(common_bins, avg_concentration, yerr=std_dev, fmt='o', capsize=5, 
                     label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linestyle='-', alpha=0.7)

        # Fit the custom exponential model
        try:
            popt, pcov = curve_fit(lambda x, D: fit_function(x, n0, D), common_bins, avg_concentration, p0=[1])
            D_fit = popt[0]

            # Plot the fitted exponential line
            fitted_values = fit_function(common_bins, n0, D_fit)
            plt.plot(common_bins, fitted_values, linestyle='--', 
                     label=f"Fit {avg_windspeed:.1f} m/s: y = {n0:.2f} * exp(-x / {D_fit:.2f})")
            
            # Calculate total concentration
            total_concentration = np.trapz(fitted_values, common_bins)
            print(f"Total concentration for windspeed {avg_windspeed:.1f} m/s: {total_concentration:.2f} / cm³")
        
        except RuntimeError:
            print(f"Could not fit the exponential model for windspeed {avg_windspeed:.1f} m/s")
# Set y-axis to log scale
plt.yscale('log')
plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Average Size Distribution by Windspeed (Excluding Problematic Legs)', fontweight='bold')
plt.legend(title="Average Windspeed")
plt.tight_layout()
plt.show()




















































































































































#%%
##fitting an exponential to Katie's curves using Rob's exponential equation

def fit_function(x, dryint, D):
    return dryint * np.exp(-x / D)

# Step 2: Initialize dictionaries to store droplet concentrations for each windspeed range
common_bins = np.linspace(0, 10, 10)  # Common x-axis values
windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]
grouped_concentrations = {i: [] for i in range(len(windspeed_bins))}  # To accumulate droplet concentrations
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Step 3: Iterate through filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])
    dryint = entry_dryintercept['dry intercept']

    # Find the corresponding windspeed entry in df_combined
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]

        # Interpolate the size distribution for this leg
        size_dist = size_distribution(ddry_values, dryint, D)  # Compute size distribution
        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)  # Interpolate to common bins

        # Categorize the leg based on windspeed range
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_concentrations[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break

# Step 4: Average each particle size bin for plotting and fit the custom exponential model
plt.figure(figsize=(12, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_concentrations[idx]:
        # Convert the list of arrays to a numpy array for easier calculations
        concentrations_array = np.array(grouped_concentrations[idx])
        
        # Average each size bin (column) across all size distributions
        avg_concentration = np.mean(concentrations_array, axis=0)
        
        # Calculate standard deviation for error bars
        std_dev = np.std(concentrations_array, axis=0)
        
        # Handle zeros in avg_concentration for log scale plotting
        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)  # Replace zeros with small value
        
        # Calculate average windspeed and number of legs in this bin
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_concentrations[idx])
        
        # Plot the average concentration with error bars and connecting lines
        plt.errorbar(common_bins, avg_concentration, yerr=std_dev, fmt='o', capsize=5, 
                     label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linestyle='-', alpha=0.7)

        # Step 5: Fit your custom exponential model to the average concentrations
        try:
            popt, pcov = curve_fit(fit_function, common_bins, avg_concentration, p0=[1, 1])
            dryint_fit, D_fit = popt

            # Plot the fitted exponential line
            fitted_values = fit_function(common_bins, dryint_fit, D_fit)
            plt.plot(common_bins, fitted_values, linestyle='--', 
                     label=f"Fit {avg_windspeed:.1f} m/s: y = {dryint_fit:.2f} * exp(-x / {D_fit:.2f})")
            
            # Print the equation for each line
            print(f"Windspeed {avg_windspeed:.1f} m/s: y = {dryint_fit:.2f} * exp(-x / {D_fit:.2f})")

        except RuntimeError:
            print(f"Could not fit the exponential model for windspeed {avg_windspeed:.1f} m/s")
        
# Set y-axis to log scale
plt.yscale('log')

plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Average Size Distribution by Windspeed (Excluding Problematic Legs)', fontweight='bold')
plt.legend(title="Average Windspeed")
plt.tight_layout()
plt.show()




#%%
#fitting an exponential to Katie's curves using a basic exponential 


# Step 1: Define your custom exponential model for fitting
def fit_function(x, dryint, D):
    return dryint * np.exp(-D*x)

# Step 2: Initialize dictionaries to store droplet concentrations for each windspeed range
common_bins = np.linspace(0, 10, 10)  # Common x-axis values
windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]
grouped_concentrations = {i: [] for i in range(len(windspeed_bins))}  # To accumulate droplet concentrations
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Step 3: Iterate through filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])
    dryint = entry_dryintercept['dry intercept']

    # Find the corresponding windspeed entry in df_combined
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]

        # Interpolate the size distribution for this leg
        size_dist = size_distribution(ddry_values, dryint, D)  # Compute size distribution
        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)  # Interpolate to common bins

        # Categorize the leg based on windspeed range
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_concentrations[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break

# Step 4: Average each particle size bin for plotting and fit the custom exponential model
plt.figure(figsize=(12, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_concentrations[idx]:
        # Convert the list of arrays to a numpy array for easier calculations
        concentrations_array = np.array(grouped_concentrations[idx])
        
        # Average each size bin (column) across all size distributions
        avg_concentration = np.mean(concentrations_array, axis=0)
        
        # Calculate standard deviation for error bars
        std_dev = np.std(concentrations_array, axis=0)
        
        # Handle zeros in avg_concentration for log scale plotting
        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)  # Replace zeros with small value
        
        # Calculate average windspeed and number of legs in this bin
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_concentrations[idx])
        
        # Plot the average concentration with error bars and connecting lines
        plt.errorbar(common_bins, avg_concentration, yerr=std_dev, fmt='o', capsize=5, 
                     label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linestyle='-', alpha=0.7)

        # Step 5: Fit your custom exponential model to the average concentrations
        try:
            popt, pcov = curve_fit(fit_function, common_bins, avg_concentration, p0=[1, 1])
            dryint_fit, D_fit = popt

            # Plot the fitted exponential line
            fitted_values = fit_function(common_bins, dryint_fit, D_fit)
            plt.plot(common_bins, fitted_values, linestyle='--', 
                     label=f"Fit {avg_windspeed:.1f} m/s: y = {dryint_fit:.2f} * exp(-x / {D_fit:.2f})")
            
            # Print the equation for each line
            print(f"Windspeed {avg_windspeed:.1f} m/s: y = {dryint_fit:.2f} * exp(-x / {D_fit:.2f})")

        except RuntimeError:
            print(f"Could not fit the exponential model for windspeed {avg_windspeed:.1f} m/s")
        
# Set y-axis to log scale
plt.yscale('log')

plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Average Size Distribution by Windspeed (Excluding Problematic Legs)', fontweight='bold')
plt.legend(title="Average Windspeed")
plt.tight_layout()
plt.show()
#%%

##calculating area under the curve for total concentration
from scipy.integrate import quad  # Import for continuous integration

# Step 1: Define your custom exponential model for fitting
def fit_function(x, dryint, D):
    return dryint * np.exp(-x / D)

# Step 2: Initialize dictionaries to store droplet concentrations for each windspeed range
common_bins = np.linspace(2.5, 10, 10)  # Adjust common bins to start from 2.5 um
windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]
grouped_concentrations = {i: [] for i in range(len(windspeed_bins))}  # To accumulate droplet concentrations
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Step 3: Iterate through filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])
    dryint = entry_dryintercept['dry intercept']

    # Find the corresponding windspeed entry in df_combined
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]

        # Interpolate the size distribution for this leg
        size_dist = size_distribution(ddry_values, dryint, D)  # Compute size distribution
        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)  # Interpolate to common bins

        # Categorize the leg based on windspeed range
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_concentrations[idx].append(interpolated_leg_values)
                mean_windspeeds[idx].append(windspeed)
                break

# Step 4: Average each particle size bin for plotting and fit the custom exponential model
plt.figure(figsize=(12, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_concentrations[idx]:
        # Convert the list of arrays to a numpy array for easier calculations
        concentrations_array = np.array(grouped_concentrations[idx])
        
        # Average each size bin (column) across all size distributions
        avg_concentration = np.mean(concentrations_array, axis=0)
        
        # Calculate standard deviation for error bars
        std_dev = np.std(concentrations_array, axis=0)
        
        # Handle zeros in avg_concentration for log scale plotting
        avg_concentration = np.where(avg_concentration <= 0, 1e-10, avg_concentration)  # Replace zeros with small value
        
        # Calculate average windspeed and number of legs in this bin
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_concentrations[idx])
        
        # Plot the average concentration with error bars and connecting lines
        plt.errorbar(common_bins, avg_concentration, yerr=std_dev, fmt='o', capsize=5, 
                     label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linestyle='-', alpha=0.7)

        # Step 5: Fit your custom exponential model to the average concentrations
        try:
            popt, pcov = curve_fit(fit_function, common_bins, avg_concentration, p0=[1, 1])
            dryint_fit, D_fit = popt

            # Plot the fitted exponential line
            fitted_values = fit_function(common_bins, dryint_fit, D_fit)
            plt.plot(common_bins, fitted_values, linestyle='--', 
                     label=f"Fit {avg_windspeed:.1f} m/s: y = {dryint_fit:.2f} * exp(-x / {D_fit:.2f})")
            
            # Print the equation for each line
            print(f"Windspeed {avg_windspeed:.1f} m/s: y = {dryint_fit:.2f} * exp(-x / {D_fit:.2f})")

            # Step 6: Calculate the total concentration (integral of N(d) * dd)
            total_concentration = np.trapz(fitted_values, common_bins)
            print(f"Total concentration for windspeed {avg_windspeed:.1f} m/s: {total_concentration:.2f} / cm³")
            
            # Alternatively, use quad for continuous integration if necessary
            # total_concentration_continuous, _ = quad(fit_function, 2.5, np.max(common_bins), args=(dryint_fit, D_fit))
            # print(f"Continuous integration result: {total_concentration_continuous:.2f} / cm³")

        except RuntimeError:
            print(f"Could not fit the exponential model for windspeed {avg_windspeed:.1f} m/s")
        
# Set y-axis to log scale
plt.yscale('log')

plt.ylabel('Mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.title('Average Size Distribution by Windspeed (Excluding Problematic Legs)', fontweight='bold')
plt.legend(title="Average Windspeed")
plt.tight_layout()
plt.show()
 













































































































































# %%

# Step 1: Define the problematic legs (if not already defined)
problematic_legs = [
    ('2022-01-11', 1), ('2022-01-11', 2), ('2022-01-11', 3), ('2022-01-11', 4),
    ('2022-01-11', 8), ('2022-01-11', 9), ('2022-01-11', 14), ('2022-01-11', 16),
    ('2022-01-11', 17), ('2022-01-26', 14), ('2022-01-26', 12), ('2022-01-26', 15),
    ('2022-01-26', 13), ('2022-01-26', 11), ('2022-03-29', 0), ('2022-05-05', 3),
    ('2022-05-05', 7),
]
problematic_set = set(problematic_legs)

# Define your common bins
common_bins = np.linspace(0, 25, 20)

# Dictionary to store droplet concentrations for each windspeed range
windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]
grouped_concentrations = {i: [] for i in range(len(windspeed_bins))}  # To accumulate droplet concentrations
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Iterate through the entries in filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Match by index

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']  # Extract Leg_index

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']  # Extract D value
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # x-axis values for this leg
    dryint = entry_dryintercept['dry intercept']  # Extract dry intercept

    # Find the corresponding entry in df_combined based on Date, Min_start, and Min_end
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    
    # If there's a matching windspeed entry
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]
        
        # Interpolate the size distribution for this leg
        size_dist = size_distribution(ddry_values, dryint, D)
        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)
        
        # Categorize the leg based on windspeed range
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                grouped_concentrations[idx].append(interpolated_leg_values)  # Add size distribution to this bin
                mean_windspeeds[idx].append(windspeed)
                break

# Step 2: Average each particle size bin first across all size distributions, then fit
plt.figure(figsize=(12, 8))

# Loop through each windspeed bin and compute averages and percentiles
for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_concentrations[idx]:
        # Convert grouped concentrations into a NumPy array
        grouped_conc_array = np.array(grouped_concentrations[idx])

        # Average each size bin (column) across all size distributions
        avg_concentration = np.mean(grouped_conc_array, axis=0)
        
        # Calculate the 25th and 75th percentiles across the legs for each bin
        perc_25 = np.percentile(grouped_conc_array, 25, axis=0)
        perc_75 = np.percentile(grouped_conc_array, 75, axis=0)
        
        # Fit the average size distribution using curve_fit
        def fit_function(x, dryint, D):
            return dryint * np.exp(-x / D)
        
        popt, _ = curve_fit(fit_function, common_bins, avg_concentration, p0=[1, 1])  # Initial guess for dryint, D
        
        # Generate fitted curve for plotting
        fitted_curve = fit_function(common_bins, *popt)
        
        # Calculate average windspeed and number of legs in this bin
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_concentrations[idx])
        
        # Plot the fitted curve
        plt.plot(common_bins, fitted_curve, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")
        
        # Fill the area between the 25th and 75th percentiles
        plt.fill_between(common_bins, perc_25, perc_75, alpha=0.2, label=f"25th-75th percentile (avg {avg_windspeed:.1f} m/s)")

# Add labels and legend
plt.ylabel('Clear mean droplet concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin diameter (µm)', fontweight='bold')
plt.yscale('log')
plt.title('Fitted Size Distribution by Windspeed (Excluding Problematic Legs)', fontweight='bold')
plt.legend(title="Windspeed")
plt.tight_layout()
plt.show()

# %%

# Assuming you have all the previous setup and imports
# Step 1: Define the problematic legs (if not already defined)
problematic_legs = [
    ('2022-01-11', 1), ('2022-01-11', 2), ('2022-01-11', 3), ('2022-01-11', 4),
    ('2022-01-11', 8), ('2022-01-11', 9), ('2022-01-11', 14), ('2022-01-11', 16),
    ('2022-01-11', 17), ('2022-01-26', 14), ('2022-01-26', 12), ('2022-01-26', 15),
    ('2022-01-26', 13), ('2022-01-26', 11), ('2022-03-29', 0), ('2022-05-05', 3),
    ('2022-05-05', 7),
]
problematic_set = set(problematic_legs)

# Define your common bins
common_bins = np.linspace(0, 25, 20)

# Dictionary to store fitted parameters for each windspeed range
windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]
fitted_params = {i: [] for i in range(len(windspeed_bins))}  # Store (dryint, D) tuples
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Iterate through the entries in filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Match by index

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']  # Extract Leg_index

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']  # Extract D value
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # x-axis values for this leg
    dryint = entry_dryintercept['dry intercept']  # Extract dry intercept

    # Find the corresponding entry in df_combined based on Date, Min_start, and Min_end
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    
    # If there's a matching windspeed entry
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]
        
        # Interpolate the size distribution for this leg
        size_dist = size_distribution(ddry_values, dryint, D)
        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        # Fit the individual distribution to get dryint and D
        def fit_function(x, dryint, D):
            return dryint * np.exp(-x / D)

        try:
            popt, _ = curve_fit(fit_function, common_bins, interpolated_leg_values, p0=[1, 1])
            fitted_params_idx = -1

            # Determine the windspeed bin for this leg
            for idx, (low, high) in enumerate(windspeed_bins):
                if low <= windspeed <= high:
                    fitted_params[idx].append(popt)  # Append the fitted parameters
                    mean_windspeeds[idx].append(windspeed)
                    fitted_params_idx = idx
                    break
            
        except RuntimeError:
            print(f"Fitting failed for leg {date}, Leg {leg_index}")

# Step 2: Average the fitted parameters across windspeed bins
averaged_params = []
for idx in range(len(windspeed_bins)):
    if fitted_params[idx]:
        avg_dryint = np.mean([param[0] for param in fitted_params[idx]])
        avg_D = np.mean([param[1] for param in fitted_params[idx]])
        averaged_params.append((avg_dryint, avg_D))

# Step 3: Plot the average fitted distributions
plt.figure(figsize=(12, 8))

# Loop through each windspeed bin and plot the fitted average distributions
for idx, (low, high) in enumerate(windspeed_bins):
    if fitted_params[idx]:
        avg_dryint, avg_D = averaged_params[idx]
        fitted_curve = fit_function(common_bins, avg_dryint, avg_D)
        
        # Calculate average windspeed and number of legs in this bin
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(fitted_params[idx])
        
        # Plot the fitted curve in log scale
        plt.plot(common_bins, fitted_curve, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")

plt.yscale('log')  # Set y-axis to log scale
plt.ylabel('Log of Clear Mean Droplet Concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin Diameter (µm)', fontweight='bold')
plt.title('Fitted Size Distribution by Windspeed (Excluding Problematic Legs)', fontweight='bold')
plt.legend(title="Windspeed")
plt.tight_layout()
plt.show()

# %%
##trying third order polynomial 


# Step 1: Define the problematic legs (if not already defined)
problematic_legs = [
    ('2022-01-11', 1), ('2022-01-11', 2), ('2022-01-11', 3), ('2022-01-11', 4),
    ('2022-01-11', 8), ('2022-01-11', 9), ('2022-01-11', 14), ('2022-01-11', 16),
    ('2022-01-11', 17), ('2022-01-26', 14), ('2022-01-26', 12), ('2022-01-26', 15),
    ('2022-01-26', 13), ('2022-01-26', 11), ('2022-03-29', 0), ('2022-05-05', 3),
    ('2022-05-05', 7),
]
problematic_set = set(problematic_legs)

# Define your common bins
common_bins = np.linspace(0, 25, 20)

# Dictionary to store fitted parameters for each windspeed range
windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]
fitted_params = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

def polynomial_fit(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

# Step 4: Function to calculate R^2
def calculate_r_squared(y_actual, y_pred):
    residual_sum_of_squares = np.sum((y_actual - y_pred) ** 2)
    total_sum_of_squares = np.sum((y_actual - np.mean(y_actual)) ** 2)
    r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)
    return r_squared


# Iterate through the entries in filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Match by index

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']  # Extract Leg_index

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']  # Extract D value
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # x-axis values for this leg
    dryint = entry_dryintercept['dry intercept']  # Extract dry intercept

    # Find the corresponding entry in df_combined based on Date, BCB_start, and BCB_stop
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    
    # If there's a matching windspeed entry
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]
        
        # Interpolate the size distribution for this leg
        size_dist = size_distribution(ddry_values, dryint, D)
        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        # Fit a cubic polynomial to the log of the interpolated values
        def poly_fit(x, a, b, c, d):
            return a * x**3 + b * x**2 + c * x + d

        try:
            log_values = np.log(interpolated_leg_values + 1e-10)  # Adding a small value to avoid log(0)
            popt, _ = curve_fit(poly_fit, common_bins, log_values)

            fitted_params_idx = -1
            # Determine the windspeed bin for this leg
            for idx, (low, high) in enumerate(windspeed_bins):
                if low <= windspeed <= high:
                    fitted_params[idx].append(popt)  # Append the fitted parameters
                    mean_windspeeds[idx].append(windspeed)
                    fitted_params_idx = idx
                    break
            
        except RuntimeError:
            print(f"Fitting failed for leg {date}, Leg {leg_index}")

# Step 2: Average the fitted parameters across windspeed bins
averaged_params = []
for idx in range(len(windspeed_bins)):
    if fitted_params[idx]:
        avg_params = np.mean(fitted_params[idx], axis=0)  # Average polynomial coefficients
        averaged_params.append(avg_params)

# Step 3: Plot the average fitted distributions
plt.figure(figsize=(12, 8))

# Loop through each windspeed bin and plot the fitted average distributions
for idx, (low, high) in enumerate(windspeed_bins):
    if fitted_params[idx]:
        avg_params = averaged_params[idx]
        fitted_curve_log = poly_fit(common_bins, *avg_params)  # Get the fitted values in log space
        fitted_curve = np.exp(fitted_curve_log)  # Convert back to original space
        
        # Calculate average windspeed and number of legs in this bin
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(fitted_params[idx])
        
        # Plot the fitted curve in log scale
        plt.plot(common_bins, fitted_curve, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs")

plt.yscale('log')  # Set y-axis to log scale
plt.ylabel('Log of Clear Mean Droplet Concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin Diameter (µm)', fontweight='bold')
plt.title('Fitted Size Distribution by Windspeed (Excluding Problematic Legs)', fontweight='bold')
plt.legend(title="Windspeed")
plt.tight_layout()
plt.show()
#%%

# Define your common bins
common_bins = np.linspace(0, 25, 20)

# Define the problematic legs
problematic_legs = [
    ('2022-01-11', 1), ('2022-01-11', 2), ('2022-01-11', 3), ('2022-01-11', 4),
    ('2022-01-11', 8), ('2022-01-11', 9), ('2022-01-11', 14), ('2022-01-11', 16),
    ('2022-01-11', 17), ('2022-01-26', 14), ('2022-01-26', 12), ('2022-01-26', 15),
    ('2022-01-26', 13), ('2022-01-26', 11), ('2022-03-29', 0), ('2022-05-05', 3),
    ('2022-05-05', 7),
]
problematic_set = set(problematic_legs)

# Wind speed bins
windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]
fitted_params = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}
r_squared_values = {i: [] for i in range(len(windspeed_bins))}  # Store R² values

# Polynomial fit function
def polynomial_fit(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

# R² calculation
def calculate_r_squared(y_actual, y_pred):
    residual_sum_of_squares = np.sum((y_actual - y_pred) ** 2)
    total_sum_of_squares = np.sum((y_actual - np.mean(y_actual)) ** 2)
    r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)
    return r_squared

# Iterate through the entries in filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Match by index

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']  # Extract Leg_index

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']  # Extract D value
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # x-axis values for this leg
    dryint = entry_dryintercept['dry intercept']  # Extract dry intercept

    # Find the corresponding entry in df_combined based on Date, BCB_start, and BCB_stop
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    
    # If there's a matching windspeed entry
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]
        
        # Interpolate the size distribution for this leg
        size_dist = size_distribution(ddry_values, dryint, D)
        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        # Check for NaNs or Infs in interpolated values
        if np.any(np.isnan(interpolated_leg_values)) or np.any(np.isinf(interpolated_leg_values)):
            print(f"Interpolated values contain NaNs or Infs for leg {date}, Leg {leg_index}")
            continue

        # Fit a cubic polynomial to the log of the interpolated values
        try:
            log_values = np.log(interpolated_leg_values + 1e-10)  # Adding a small value to avoid log(0)
            popt, _ = curve_fit(poly_fit, common_bins, log_values)
            
            # Further processing...
        except RuntimeError as e:
            print(f"Fitting failed for leg {date}, Leg {leg_index}: {e}")


# Average the fitted parameters and R² across windspeed bins
averaged_params = []
average_r_squared = []
for idx in range(len(windspeed_bins)):
    if fitted_params[idx]:
        avg_params = np.mean(fitted_params[idx], axis=0)  # Average polynomial coefficients
        avg_r_squared = np.mean(r_squared_values[idx])  # Average R² values
        averaged_params.append(avg_params)
        average_r_squared.append(avg_r_squared)

# Plot the average fitted distributions
plt.figure(figsize=(12, 8))

# Loop through each windspeed bin and plot the fitted average distributions
for idx, (low, high) in enumerate(windspeed_bins):
    if fitted_params[idx]:
        avg_params = averaged_params[idx]
        fitted_curve_log = polynomial_fit(common_bins, *avg_params)  # Get the fitted values in log space
        fitted_curve = np.exp(fitted_curve_log)  # Convert back to original space
        
        # Calculate average windspeed and number of legs in this bin
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(fitted_params[idx])
        
        # Plot the fitted curve in log scale
        plt.plot(common_bins, fitted_curve, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs, R²={average_r_squared[idx]:.2f}")

plt.yscale('log')  # Set y-axis to log scale
plt.ylabel('Log of Clear Mean Droplet Concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin Diameter (µm)', fontweight='bold')
plt.title('Fitted Size Distribution by Windspeed (Excluding Problematic Legs)', fontweight='bold')
plt.legend(title="Windspeed")
plt.tight_layout()
plt.show()
#%%
# %%

##supposedly a third order polynomial but spits out 4 linear lines
problematic_legs = [
    ('2022-01-11', 1), ('2022-01-11', 2), ('2022-01-11', 3), ('2022-01-11', 4),
    ('2022-01-11', 8), ('2022-01-11', 9), ('2022-01-11', 14), ('2022-01-11', 16),
    ('2022-01-11', 17), ('2022-01-26', 14), ('2022-01-26', 12), ('2022-01-26', 15),
    ('2022-01-26', 13), ('2022-01-26', 11), ('2022-03-29', 0), ('2022-05-05', 3),
    ('2022-05-05', 7),
]
problematic_set = set(problematic_legs)

# Define your common bins
common_bins = np.linspace(0, 25, 20)

# Dictionary to store fitted parameters for each windspeed range
windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]
fitted_params = {i: [] for i in range(len(windspeed_bins))}  # Store (dryint, D) tuples
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Iterate through the entries in filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Match by index

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']  # Extract Leg_index

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']  # Extract D value
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # x-axis values for this leg
    dryint = entry_dryintercept['dry intercept']  # Extract dry intercept

    # Find the corresponding entry in df_combined based on Date, Min_start, and Min_end
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    
    # If there's a matching windspeed entry
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]
        
        # Interpolate the size distribution for this leg
        size_dist = size_distribution(ddry_values, dryint, D)
        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        # Fit the individual distribution to get dryint and D
        def fit_function(x, dryint, D):
            return dryint * np.exp(-x / D)

        try:
            popt, _ = curve_fit(fit_function, common_bins, interpolated_leg_values, p0=[1, 1])
            fitted_params_idx = -1

            # Determine the windspeed bin for this leg
            for idx, (low, high) in enumerate(windspeed_bins):
                if low <= windspeed <= high:
                    fitted_params[idx].append(popt)  # Append the fitted parameters
                    mean_windspeeds[idx].append(windspeed)
                    fitted_params_idx = idx
                    break
            
        except RuntimeError:
            print(f"Fitting failed for leg {date}, Leg {leg_index}")

# Step 2: Average the fitted parameters across windspeed bins
averaged_params = []
for idx in range(len(windspeed_bins)):
    if fitted_params[idx]:
        avg_dryint = np.mean([param[0] for param in fitted_params[idx]])
        avg_D = np.mean([param[1] for param in fitted_params[idx]])
        averaged_params.append((avg_dryint, avg_D))

# Step 3: Plot the average fitted distributions
plt.figure(figsize=(12, 8))

# Loop through each windspeed bin and plot the fitted average distributions
for idx, (low, high) in enumerate(windspeed_bins):
    if fitted_params[idx]:
        avg_dryint, avg_D = averaged_params[idx]
        
        # Calculate fitted curve
        fitted_curve = fit_function(common_bins, avg_dryint, avg_D)
        
        # Calculate R²
        residuals = interpolated_leg_values - fitted_curve
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((interpolated_leg_values - np.mean(interpolated_leg_values))**2)
        r_squared = 1 - (ss_res / ss_tot)

        # Plot the fitted curve with a label
        plt.plot(common_bins, fitted_curve, label=f"Windspeed {low}-{high} m/s (R²={r_squared:.2f})")

# Set the axis scales and labels
# If needed
plt.yscale('log')  # Set y-axis to log scale
plt.ylabel('Log of Clear Mean Droplet Concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin Diameter (µm)', fontweight='bold')
plt.title('Fitted Size Distribution by Windspeed (Excluding Problematic Legs)', fontweight='bold')

# Add legend
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()
#%%
##weird mountain curves

# Your problematic legs set
problematic_legs = [
    ('2022-01-11', 1), ('2022-01-11', 2), ('2022-01-11', 3), ('2022-01-11', 4),
    ('2022-01-11', 8), ('2022-01-11', 9), ('2022-01-11', 14), ('2022-01-11', 16),
    ('2022-01-11', 17), ('2022-01-26', 14), ('2022-01-26', 12), ('2022-01-26', 15),
    ('2022-01-26', 13), ('2022-01-26', 11), ('2022-03-29', 0), ('2022-05-05', 3),
    ('2022-05-05', 7),
]
problematic_set = set(problematic_legs)

# Define your common bins
common_bins = np.linspace(0, 25, 20)

# Dictionary to store fitted parameters for each windspeed range
windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]
fitted_params = {i: [] for i in range(len(windspeed_bins))}  # Store (dryint, D) tuples
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Iterate through the entries in filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Match by index

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']  # Extract Leg_index

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']  # Extract D value
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # x-axis values for this leg
    dryint = entry_dryintercept['dry intercept']  # Extract dry intercept

    # Find the corresponding entry in df_combined based on Date, BCB_start, and BCB_stop
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    
    # If there's a matching windspeed entry
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]
        
        # Interpolate the size distribution for this leg
        size_dist = size_distribution(ddry_values, dryint, D)
        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        # Adjusted fit function to a cubic polynomial
        def fit_function(x, a, b, c, d):
            return a * x**3 + b * x**2 + c * x + d

        # Fit the individual distribution to get dryint and D
        try:
            popt, _ = curve_fit(fit_function, common_bins, interpolated_leg_values, p0=[1, 1, 1, 1])
            fitted_params_idx = -1

            # Determine the windspeed bin for this leg
            for idx, (low, high) in enumerate(windspeed_bins):
                if low <= windspeed <= high:
                    fitted_params[idx].append(popt)  # Append the fitted parameters
                    mean_windspeeds[idx].append(windspeed)
                    fitted_params_idx = idx
                    break
            
        except RuntimeError:
            print(f"Fitting failed for leg {date}, Leg {leg_index}")

# Step 2: Average the fitted parameters across windspeed bins
averaged_params = []
for idx in range(len(windspeed_bins)):
    if fitted_params[idx]:
        avg_dryint = np.mean([param[0] for param in fitted_params[idx]])
        avg_D = np.mean([param[1] for param in fitted_params[idx]])
        averaged_params.append((avg_dryint, avg_D))

# Step 3: Plot the average fitted distributions
plt.figure(figsize=(12, 8))

# Loop through each windspeed bin and plot the fitted average distributions
for idx, (low, high) in enumerate(windspeed_bins):
    if fitted_params[idx]:
        avg_dryint, avg_D = averaged_params[idx]
        
        # Calculate fitted curve
        fitted_curve = fit_function(common_bins, avg_dryint, 0, avg_D, 0)  # Using average parameters
        
        # Calculate R²
        residuals = interpolated_leg_values - fitted_curve
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((interpolated_leg_values - np.mean(interpolated_leg_values))**2)
        r_squared = 1 - (ss_res / ss_tot)

        # Plot the fitted curve with a label
        plt.plot(common_bins, fitted_curve, label=f"Windspeed {low}-{high} m/s (R²={r_squared:.2f})")

# Set the axis scales and labels
plt.yscale('log')  # Set y-axis to log scale
plt.ylabel('Log of Clear Mean Droplet Concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin Diameter (µm)', fontweight='bold')
plt.title('Fitted Size Distribution by Windspeed (Excluding Problematic Legs)', fontweight='bold')

# Add legend
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()
#%%
##third order with 0.90 specification  

# Problematic legs (example data)
problematic_legs = [
    ('2022-01-11', 1), ('2022-01-11', 2), ('2022-01-11', 3), ('2022-01-11', 4),
    ('2022-01-11', 8), ('2022-01-11', 9), ('2022-01-11', 14), ('2022-01-11', 16),
    ('2022-01-11', 17),('2022-01-26', 16), ('2022-01-26', 14), ('2022-01-26', 12), 
    ('2022-01-26', 15),
    ('2022-01-26', 13), ('2022-01-26', 11), ('2022-03-29', 0), ('2022-05-05', 3),
    ('2022-05-05', 7), ('2022-05-05', 3), ('2022-06-11', 4), ('2022-03-29', 9), ('2022-03-29', 9), 
    ('2022-03-29', 0), ('2022-03-29', 6), ('2022-03-28', 4), ('2022-03-28', 0), ('2022-03-26', 15),
    ('2022-03-26', 6), ('2022-03-26', 0), ('2022-03-13', 8)]
problematic_set = set(problematic_legs)

# Define your common bins
common_bins = np.linspace(0, 25, 20)

# Define wind speed bins
windspeed_bins = [(0, 2.1), (2.2, 5.0), (5.1, 5.6), (5.7, 8.5), (8.6, np.inf)]

# Prepare storage for distributions per windspeed bin
averaged_distributions = {i: [] for i in range(len(windspeed_bins))}

# Iterate through the entries in filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Match by index

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']  # Extract Leg_index

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']  # Extract D value
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # x-axis values for this leg
    dryint = entry_dryintercept['dry intercept']  # Extract dry intercept

    # Find the corresponding entry in df_combined based on Date, BCB_start, and BCB_stop
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]

    # If there's a matching windspeed entry
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]
        
        # Interpolate the size distribution for this leg
        size_dist = size_distribution(ddry_values, dryint, D)
        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        # Determine the windspeed bin for this leg and store the interpolated values
        for idx, (low, high) in enumerate(windspeed_bins):
            if low <= windspeed <= high:
                averaged_distributions[idx].append(interpolated_leg_values)  # Append the interpolated values
                break

# Step 2: Average the raw distributions for each windspeed bin
averaged_params = []
plt.figure(figsize=(12, 8))

for idx in range(len(windspeed_bins)):
    if averaged_distributions[idx]:  # Only proceed if there are distributions
        avg_distribution = np.mean(averaged_distributions[idx], axis=0)

        # Fit a third-order polynomial to the averaged distribution
        def polynomial_fit(x, a, b, c, d):
            return a * x**3 + b * x**2 + c * x + d

        try:
            popt, _ = curve_fit(polynomial_fit, common_bins, avg_distribution, p0=[1, 1, 1, 1])
            fitted_curve = polynomial_fit(common_bins, *popt)

            # Calculate R²
            residuals = avg_distribution - fitted_curve
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((avg_distribution - np.mean(avg_distribution))**2)
            r_squared = 1 - (ss_res / ss_tot)

            # Ensure R² is greater than 0.95
            if r_squared > 0.90:
                plt.plot(common_bins, fitted_curve, label=f"Windspeed {windspeed_bins[idx]} m/s (R²={r_squared:.2f})")
            else:
                print(f"R² value for windspeed bin {idx} is below threshold: {r_squared:.2f}")

        except RuntimeError:
            print(f"Fitting failed for windspeed bin {idx}")

# Set the axis scales and labels
plt.yscale('log')  # Set y-axis to log scale
plt.ylabel('Log of Clear Mean Droplet Concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin Diameter (µm)', fontweight='bold')
plt.title('Fitted Size Distribution by Windspeed (Excluding Problematic Legs)', fontweight='bold')

# Add legend
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()
#%%
from numpy.polynomial import Polynomial

# Assuming these functions and datasets are defined earlier
# size_distribution function needs to be defined

# Example: Define windspeed bins
windspeed_bins = [
    (0, 2),      # Bin 1: 0 to 2 m/s
    (2.1, 4),    # Bin 2: 2.1 to 4 m/s
    (4.1, 6),    # Bin 3: 4.1 to 6 m/s
    (6.1, 8),    # Bin 4: 6.1 to 8 m/s
    (8.1, 10),   # Bin 5: 8.1 to 10 m/s
    (10.1, np.inf)  # Bin 6: Greater than 10 m/s
]

# Initialize storage for fitted parameters and mean wind speeds
fitted_params = [[] for _ in range(len(windspeed_bins))]
mean_windspeeds = [[] for _ in range(len(windspeed_bins))]

# Iterate through entries in filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Match by index

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']  # Extract Leg_index

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']  # Extract D value
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # x-axis values for this leg
    dryint = entry_dryintercept['dry intercept']  # Extract dry intercept

    # Find the corresponding entry in df_combined based on Date, BCB_start, and BCB_stop
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    
    # If there's a matching windspeed entry
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]
        
        # Interpolate the size distribution for this leg
        size_dist = size_distribution(ddry_values, dryint, D)
        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        # Fit a polynomial of a specified order
        order = 4  # Change this based on your data
        poly_coeffs = Polynomial.fit(common_bins, interpolated_leg_values, order)

        # Create fitted curve
        fitted_curve = poly_coeffs(common_bins)

        # Calculate R²
        residuals = interpolated_leg_values - fitted_curve
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((interpolated_leg_values - np.mean(interpolated_leg_values))**2)
        r_squared = 1 - (ss_res / ss_tot)

        # Check R² threshold
        if r_squared > 0.90:
            fitted_params_idx = -1

            # Determine the windspeed bin for this leg
            for idx, (low, high) in enumerate(windspeed_bins):
                if low <= windspeed <= high:
                    fitted_params[idx].append((poly_coeffs.coef[0], poly_coeffs.coef[1]))  # Store coefficients
                    mean_windspeeds[idx].append(windspeed)
                    fitted_params_idx = idx
                    break

        else:
            print(f"R² value for windspeed bin is below threshold: {r_squared:.2f}")

# Additional processing or visualization of fitted_params and mean_windspeeds...

# Step 2: Average the fitted parameters across windspeed bins
averaged_params = []
for idx in range(len(windspeed_bins)):
    if fitted_params[idx]:
        avg_dryint = np.mean([param[0] for param in fitted_params[idx]])
        averaged_params.append(avg_dryint)

# Step 3: Plot the average fitted distributions
plt.figure(figsize=(12, 8))

# Loop through each windspeed bin and plot the fitted average distributions
for idx, (low, high) in enumerate(windspeed_bins):
    if fitted_params[idx]:
        avg_dryint = averaged_params[idx]
        
        # Calculate fitted curve using the polynomial coefficients
        fitted_curve = Polynomial([avg_dryint, 0])(common_bins)  # Use coefficients here

        # Plot the fitted curve with a label
        plt.plot(common_bins, fitted_curve, label=f"Windspeed {low}-{high} m/s")

# Set the axis scales and labels
plt.yscale('log')  # Set y-axis to log scale
plt.ylabel('Log of Clear Mean Droplet Concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin Diameter (µm)', fontweight='bold')
plt.title('Fitted Size Distribution by Windspeed (Excluding Problematic Legs)', fontweight='bold')

# Add legend
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()
#%%


# Define your exponential decay function
def exponential_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

# Fit your data
for idx in range(len(windspeed_bins)):
    if averaged_distributions[idx]:  # Only proceed if there are distributions
        avg_distribution = np.mean(averaged_distributions[idx], axis=0)
        
        # Perform curve fitting
        try:
            popt, _ = curve_fit(exponential_decay, common_bins, avg_distribution, p0=[1, 1, 1])
            fitted_curve = exponential_decay(common_bins, *popt)
            
            plt.plot(common_bins, fitted_curve, label=f"Exponential Fit for Windspeed {windspeed_bins[idx]} m/s")
        
        except RuntimeError:
            print(f"Fitting failed for windspeed bin {idx}")

# Set the axis scales and labels
plt.yscale('log')  # Set y-axis to log scale
plt.ylabel('Log of Clear Mean Droplet Concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin Diameter (µm)', fontweight='bold')
plt.title('Fitted Size Distribution by Windspeed', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()

#%%































































































#%%
##extremely weird fit makes no sense 

# Define your common bins
common_bins = np.linspace(0.1, 25, 20)  # Adjusting to avoid log(0)

# Polynomial fitting function (degree can be adjusted)
def polynomial_fit(x, *coeffs):
    # Coefficients for the polynomial can be of any degree; here, we use degree 4.
    return sum(c * x**i for i, c in enumerate(coeffs))

# Dictionary to store fitted parameters for each windspeed range
windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]
fitted_params = {i: [] for i in range(len(windspeed_bins))}  # Store coefficients tuples
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Iterate through the entries in filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]  # Match by index

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']  # Extract Leg_index

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']  # Extract D value
    ddry_values = np.array(entry_ddry['filtered_ddry'])  # x-axis values for this leg
    dryint = entry_dryintercept['dry intercept']  # Extract dry intercept

    # Find the corresponding entry in df_combined based on Date, Min_start, and Min_end
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    
    # If there's a matching windspeed entry
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]
        
        # Interpolate the size distribution for this leg
        size_dist = size_distribution(ddry_values, dryint, D)
        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        # Fit the polynomial distribution to get coefficients
        try:
            popt, _ = curve_fit(polynomial_fit, common_bins, interpolated_leg_values, p0=[1, 1, 1, 1, 1])  # Adjusted for a 4th degree fit
            fitted_params_idx = -1

            # Determine the windspeed bin for this leg
            for idx, (low, high) in enumerate(windspeed_bins):
                if low <= windspeed <= high:
                    fitted_params[idx].append(popt)  # Append the fitted parameters
                    mean_windspeeds[idx].append(windspeed)
                    fitted_params_idx = idx
                    break
            
        except RuntimeError:
            print(f"Fitting failed for leg {date}, Leg {leg_index}")

# Step 2: Average the fitted parameters across windspeed bins
averaged_params = []
for idx in range(len(windspeed_bins)):
    if fitted_params[idx]:
        avg_params = np.mean(fitted_params[idx], axis=0)  # Average the parameters
        averaged_params.append(avg_params)

# Step 3: Plot the average fitted distributions
plt.figure(figsize=(12, 8))

# Loop through each windspeed bin and plot the fitted average distributions
for idx, (low, high) in enumerate(windspeed_bins):
    if fitted_params[idx]:
        avg_params = averaged_params[idx]
        
        # Calculate fitted curve
        fitted_curve = polynomial_fit(common_bins, *avg_params)
        
        # Calculate R²
        residuals = interpolated_leg_values - fitted_curve
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((interpolated_leg_values - np.mean(interpolated_leg_values))**2)
        r_squared = 1 - (ss_res / ss_tot)

        # Plot the fitted curve with a label
        plt.plot(common_bins, fitted_curve, label=f"Windspeed {low}-{high} m/s (R²={r_squared:.2f})")

# Set the y-axis to log scale while keeping the x-axis linear
plt.yscale('log')  # Set y-axis to log scale
plt.ylabel('Log of Clear Mean Droplet Concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin Diameter (µm)', fontweight='bold')
plt.title('Fitted Size Distribution by Windspeed (Excluding Problematic Legs)', fontweight='bold')

# Add legend
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()
#%%

# Define common_bins with a minimum value to avoid log(0)
common_bins = np.linspace(0.1, 25, 20)

# Define the problematic legs
problematic_legs = [
    ('2022-01-11', 1), ('2022-01-11', 2), ('2022-01-11', 3), ('2022-01-11', 4),
    ('2022-01-11', 8), ('2022-01-11', 9), ('2022-01-11', 14), ('2022-01-11', 16),
    ('2022-01-11', 17), ('2022-01-26', 14), ('2022-01-26', 12), ('2022-01-26', 15),
    ('2022-01-26', 13), ('2022-01-26', 11), ('2022-03-29', 0), ('2022-05-05', 3),
    ('2022-05-05', 7),
]
problematic_set = set(problematic_legs)

# Polynomial fitting function (4th degree)
def polynomial_fit(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e

# Wind speed bins
windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]
fitted_params = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Iterate through the entries in filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])
    dryint = entry_dryintercept['dry intercept']

    # Find the corresponding windspeed entry
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    
    # If there's a matching windspeed entry
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]
        
        # Interpolate the size distribution for this leg
        size_dist = size_distribution(ddry_values, dryint, D)
        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        # Check for NaNs or Infs in interpolated values
        if np.any(np.isnan(interpolated_leg_values)) or np.any(np.isinf(interpolated_leg_values)):
            print(f"Interpolated values contain NaNs or Infs for leg {date}, Leg {leg_index}")
            continue

        # Remove zeros and negative values from interpolated_leg_values
        interpolated_leg_values[interpolated_leg_values <= 0] = np.nan

        # Check if there are still NaNs after filtering
        if np.isnan(interpolated_leg_values).all():
            print(f"All interpolated values became NaN for leg {date}, Leg {leg_index}")
            continue

        # Fit the polynomial distribution
        try:
            # Add a small value to avoid log(0) issues
            log_values = np.log(np.nan_to_num(interpolated_leg_values, nan=1e-10))  # Replace NaNs with small values
            popt, _ = curve_fit(polynomial_fit, common_bins, log_values, p0=[1, 1, 1, 1, 1])
            
            # Append the fitted parameters to the corresponding windspeed bin
            for idx, (low, high) in enumerate(windspeed_bins):
                if low <= windspeed <= high:
                    fitted_params[idx].append(popt)
                    mean_windspeeds[idx].append(windspeed)
                    break

        except RuntimeError:
            print(f"Fitting failed for leg {date}, Leg {leg_index}")

# Step 2: Average the fitted parameters across windspeed bins
averaged_params = []
for idx in range(len(windspeed_bins)):
    if fitted_params[idx]:
        avg_params = np.mean(fitted_params[idx], axis=0)
        averaged_params.append(avg_params)

# Step 3: Plot the average fitted distributions
# Plot the average fitted distributions
plt.figure(figsize=(12, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if fitted_params[idx]:
        avg_params = averaged_params[idx]
        fitted_curve = polynomial_fit(common_bins, *avg_params)

        # Plot the fitted curve with a label showing only the windspeed bin range
        plt.plot(common_bins, np.exp(fitted_curve), label=f"Windspeed {low}-{high} m/s")

# Set the y-axis to log scale
plt.yscale('log')  
plt.ylabel('Log of Clear Mean Droplet Concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin Diameter (µm)', fontweight='bold')
plt.title('Fitted Size Distribution by Windspeed (Excluding Problematic Legs)', fontweight='bold')

# Add legend and show plot
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()


# %%
#trying a fifth order polynomial 
# Polynomial fitting function (5th degree)
# Polynomial fitting function (5th degree)
def polynomial_fit(x, a, b, c, d, e, f):
    return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f

# Wind speed bins
windspeed_bins = [(0, 3.6), (3.7, 5.4), (5.5, 7.5), (7.6, np.inf)]
fitted_params = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}

# Iterate through the entries in filtered_master_min_ddry and match dry intercept
for i in range(len(filtered_master_BCB_ddry)):
    entry_ddry = filtered_master_BCB_ddry[i]
    entry_dryintercept = filtered_master_BCB_dryintercept[i]

    # Extract necessary values
    date = entry_ddry['Date']
    BCB_start = entry_ddry['BCB_start']
    BCB_stop = entry_ddry['BCB_stop']
    leg_index = entry_ddry['Leg_index']

    # Skip problematic legs
    if (date, leg_index) in problematic_set:
        continue

    D = entry_ddry['D']
    ddry_values = np.array(entry_ddry['filtered_ddry'])
    dryint = entry_dryintercept['dry intercept']

    # Find the corresponding windspeed entry
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    
    # If there's a matching windspeed entry
    if not windspeed_entry.empty:
        windspeed = windspeed_entry['Windspeed'].values[0]
        
        # Interpolate the size distribution for this leg
        size_dist = size_distribution(ddry_values, dryint, D)
        interp_func = interp1d(ddry_values, size_dist, kind='linear', fill_value='extrapolate')
        interpolated_leg_values = interp_func(common_bins)

        # Check for NaNs or Infs in interpolated values
        if np.any(np.isnan(interpolated_leg_values)) or np.any(np.isinf(interpolated_leg_values)):
            print(f"Interpolated values contain NaNs or Infs for leg {date}, Leg {leg_index}")
            continue

        # Log-transform the interpolated values
        try:
            # Check for non-positive values that would cause issues with log
            if np.any(interpolated_leg_values <= 0):
                print(f"Non-positive interpolated values found for leg {date}, Leg {leg_index}: {interpolated_leg_values}")
                continue

            # Remove potential negative values and zeroes to avoid log(0) issues
            log_values = np.log(interpolated_leg_values + 1e-10)  # Adding a small value to avoid log(0)
            
            # Check for NaNs or Infs after log transformation
            if np.any(np.isnan(log_values)) or np.any(np.isinf(log_values)):
                print(f"Log-transformed values contain NaNs or Infs for leg {date}, Leg {leg_index}")
                continue

            # Fit the polynomial distribution (5th degree)
            popt, _ = curve_fit(polynomial_fit, common_bins, log_values, p0=[1, 1, 1, 1, 1, 1])

            fitted_params_idx = -1

            # Determine the windspeed bin for this leg
            for idx, (low, high) in enumerate(windspeed_bins):
                if low <= windspeed <= high:
                    fitted_params[idx].append(popt)
                    mean_windspeeds[idx].append(windspeed)
                    fitted_params_idx = idx
                    break

        except RuntimeError:
            print(f"Fitting failed for leg {date}, Leg {leg_index}")

# Step 2: Average the fitted parameters across windspeed bins
averaged_params = []
for idx in range(len(windspeed_bins)):
    if fitted_params[idx]:
        avg_params = np.mean(fitted_params[idx], axis=0)
        averaged_params.append(avg_params)

# Step 3: Plot the average fitted distributions
plt.figure(figsize=(12, 8))

for idx, (low, high) in enumerate(windspeed_bins):
    if fitted_params[idx]:
        avg_params = averaged_params[idx]
        fitted_curve = polynomial_fit(common_bins, *avg_params)

        # Plot the fitted curve with a label showing only the windspeed bin range
        plt.plot(common_bins, np.exp(fitted_curve), label=f"Windspeed {low}-{high} m/s")

# Set the y-axis to log scale
plt.yscale('log')  
plt.ylim(10**-2, 10**0)
plt.ylabel('Log of Clear Mean Droplet Concentration (/cm³/µm)', fontweight='bold')
plt.xlabel('Bin Diameter (µm)', fontweight='bold')
plt.title('Fitted Size Distribution by Windspeed (Excluding Problematic Legs)', fontweight='bold')

# Add legend and show plot
plt.legend()
plt.show()


# %%
##Trying to look at mass so I can better understand the phase space. 

#I am first doing to try and look at mass, before fitting an exponential to 
#the distribution, and before binning by windspeed. 

# Key Concepts
# Mass Calculation: Mass for each diameter bin is calculated as:

#m(d)=N(d)⋅ρ⋅6π​d^3

#Cumulative Mass:Compute the cumulative sum of mass across bins:
#Mcumulative (d)= d′≤d∑m(d′) 
#Normalize by the total mass 
#𝑀total:𝑀normalized(𝑑)=𝑀cumulative(𝑑)/Mtotal(d)

#Finding the Diameter at 50% Mass:
#Use linear interpolation on the normalized cumulative mass curve to find the diameter where 
#Mnormalized=0.5. This diameter is the D50 value.

#%%
# Function to calculate mass with lower limit of integration at 2 and plot mass contours with windspeed
def calculate_mass(N0, D):
    integrand = lambda d: np.exp(-d / D) * d**3
    mass_integral, _ = quad(integrand, 2, np.inf)
    return N0 * mass_integral

# Filter out rows where D or dryintercept is NaN or <= 0
filtered_combined_clean = filtered_combined[(filtered_combined['D'] > 0) & 
                                            (filtered_combined['dryintercept'] > 0)].copy()

# Recalculate mass for each point using the integrated mass equation (apply row-wise)
filtered_combined_clean['Mass'] = filtered_combined_clean.apply(
    lambda row: calculate_mass(row['dryintercept'], row['D']), axis=1)

# Debugging: Check minimum and maximum mass values
print(f"Min mass: {filtered_combined_clean['Mass'].min()}, Max mass: {filtered_combined_clean['Mass'].max()}")

# Create data-based grids to better match the distribution of the points
xgrid = np.logspace(np.log10(filtered_combined_clean['D'].min()), 
                    np.log10(filtered_combined_clean['D'].max()), 100)
ygrid = np.logspace(np.log10(filtered_combined_clean['dryintercept'].min()), 
                    np.log10(filtered_combined_clean['dryintercept'].max()), 100)
D_grid, dryintercept_grid = np.meshgrid(xgrid, ygrid)

# Calculate mass for each point on the grid (use np.vectorize to apply calculate_mass element-wise)
vectorized_mass = np.vectorize(calculate_mass)
mass_grid = vectorized_mass(dryintercept_grid, D_grid)

# Debugging: Check if mass_grid contains meaningful values
print(f"Mass grid: Min = {np.nanmin(mass_grid)}, Max = {np.nanmax(mass_grid)}")

low_levels = np.logspace(np.log10(filtered_combined_clean['Mass'].min()), np.log10(1), 6, endpoint=False)  # 3 contours in low range (up to 10^0)
high_levels = np.logspace(np.log10(1), np.log10(filtered_combined_clean['Mass'].max()), 5)  # More contours in higher range (10^0 to max)

# Ensure contours are strictly increasing by combining low and high levels
mass_levels = np.concatenate([low_levels, high_levels])

print(f"Mass levels: {mass_levels}")

# Plot the scatter plot with Windspeed color map
sc = plt.scatter(filtered_combined_clean['D'], filtered_combined_clean['dryintercept'], 
                 c=filtered_combined_clean['Windspeed'], cmap='viridis', s=100, label='Windspeed Present')

# Plot the mass contours using the full integrated mass formula with reduced number of contours
contour_plot = plt.contour(D_grid, dryintercept_grid, mass_grid, levels=mass_levels, 
                           colors='red', linewidths=0.5, alpha=0.75)

# Check if contours were created
if len(contour_plot.allsegs[0]) == 0:
    print("No contours were created. Check your data range or mass grid calculation.")

# Add color bar for windspeed
plt.colorbar(sc, label='Corrected Windspeed (m/s)')

# Add labels and formatting
plt.xlabel('D', fontsize=14, fontweight='bold')
plt.ylabel('Dry Intercept', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.title('Mass Contours', fontsize=14, fontweight='bold')
plt.xlim(10**-2, 10**2)
plt.ylim(10**-2, 10**2.5)

# Display the plot
plt.show()
#%%

# Function to calculate mass per droplet size bin
def calculate_mass(diameter, concentration):
    # Mass per bin: m(d) = N(d) * (π/6) * d³
    return concentration * (np.pi / 6) * (diameter**3)

# Function to calculate cumulative mass and find 50% diameter
def calculate_cumulative_mass(diameters, masses):
    # Compute cumulative mass
    cumulative_mass = np.cumsum(masses)
    total_mass = np.sum(masses)
    
    # Normalize cumulative mass
    normalized_cumulative_mass = cumulative_mass / total_mass

    # Interpolate to find diameter at 50% of total mass
    interpolation_func = interp1d(normalized_cumulative_mass, diameters, kind='linear', fill_value='extrapolate')
    diameter_at_50_percent = interpolation_func(0.5)  # Diameter where 50% of mass is reached
    
    return normalized_cumulative_mass, diameter_at_50_percent

# Prepare bins (common_bins) for droplet diameters (ensuring bins start from 2 µm)
common_bins = np.linspace(2, 50, 18)  # Match the original size distribution

# Initialize a list to store the diameter at 50% mass for each leg
diameters_at_50_mass = []

plt.figure(figsize=(12, 8))

# Filter out problematic legs
filtered_data = [
    entry for entry in filtered_master_BCB_ddry
    if (entry['Date'], entry['Leg_index']) not in problematic_set
]

# Loop through each leg in the filtered data
for i, entry in enumerate(filtered_data):
    # Extract the raw concentrations for this leg
    raw_concentrations = np.array(entry['filtered_ddry_concentrations'])  # Replace with your actual raw concentration data
    diameters = np.array(entry['filtered_ddry'])  # Replace with your actual bin diameters

    # Check for mismatched bins or NaNs
    if len(diameters) != len(raw_concentrations) or np.any(np.isnan(raw_concentrations)) or np.any(np.isnan(diameters)):
        print(f"Skipping leg {i} due to mismatched bin sizes or NaNs.")
        continue

    # Calculate mass for each bin
    masses = calculate_mass(diameters, raw_concentrations)

    # Calculate cumulative mass and determine the diameter at 50% of the total mass
    normalized_cumulative_mass, diameter_at_50 = calculate_cumulative_mass(diameters, masses)
    diameters_at_50_mass.append(diameter_at_50)

    # Plot the cumulative mass distribution for this leg
    plt.plot(diameters, normalized_cumulative_mass, label=f"Leg {i}")
    plt.axvline(diameter_at_50, color='red', linestyle='--', alpha=0.5, 
                label=f"50% Mass at {diameter_at_50:.2f} µm (Leg {i})")

# Add labels, title, and legend
plt.xlabel('Diameter (µm)', fontsize=14, fontweight='bold')
plt.ylabel('Cumulative Mass (Normalized)', fontsize=14, fontweight='bold')
plt.title('Cumulative Mass Distribution for Each Leg (Raw Data)', fontsize=14, fontweight='bold')
plt.legend(fontsize=8, loc='best', ncol=2)
plt.grid()
plt.tight_layout()
plt.show()

# Print the diameters at 50% mass for all legs
print("Diameters at 50% Mass for Each Leg:")
for i, diameter in enumerate(diameters_at_50_mass, 1):
    print(f"Leg {i}: {diameter:.2f} µm")



# %%

# Function to calculate mass per bin
def calculate_mass(diameter, concentration):
    # Mass per bin: m(d) = N(d) * (pi/6) * d^3
    return concentration * (np.pi / 6) * (diameter**3)

# Function to calculate cumulative mass and find the diameter at 50% of the total mass
def calculate_cumulative_mass(diameters, masses):
    # Compute cumulative mass
    cumulative_mass = np.cumsum(masses)
    total_mass = np.sum(masses)
    
    # Normalize cumulative mass
    normalized_cumulative_mass = cumulative_mass / total_mass

    # Interpolate to find diameter at 50% of total mass
    interpolation_func = interp1d(normalized_cumulative_mass, diameters, kind='linear', fill_value='extrapolate')
    diameter_at_50_percent = interpolation_func(0.5)  # Diameter where 50% of mass is reached
    
    return cumulative_mass, diameter_at_50_percent

# Ensure problematic legs are excluded
problematic_legs = [
    ('2022-01-11', 1), ('2022-01-11', 2), ('2022-01-11', 3), ('2022-01-11', 4),
    ('2022-01-11', 8), ('2022-01-11', 9), ('2022-01-11', 14), ('2022-01-11', 16),
    ('2022-01-11', 17), ('2022-01-26', 16), ('2022-01-26', 14), ('2022-01-26', 12),
    ('2022-01-26', 15), ('2022-01-26', 13), ('2022-01-26', 11), ('2022-03-29', 0),
    ('2022-05-05', 3), ('2022-05-05', 7), ('2022-05-05', 3), ('2022-06-11', 4),
    ('2022-03-29', 9), ('2022-03-29', 9), ('2022-03-29', 0), ('2022-03-29', 6),
    ('2022-03-28', 4), ('2022-03-28', 0), ('2022-03-26', 15), ('2022-03-26', 6),
    ('2022-03-26', 0), ('2022-03-13', 8)
]
problematic_set = set(problematic_legs)

# Filter the dataset to exclude problematic legs
filtered_interpolated_values = [
    entry for entry in interpolated_values  # Replace `interpolated_values` with your dataset
    if (entry['Date'], entry['Leg_index']) not in problematic_set
]

# Prepare plot
plt.figure(figsize=(12, 8))

# List to store 50% mass diameters for all legs
diameters_at_50_mass = []

# Process each leg's size distribution
for entry in filtered_interpolated_values:
    diameters = np.array(entry['filtered_ddry'])  # Bin diameters
    concentrations = np.array(entry['filtered_ddry_concentrations'])  # Concentrations
    
    # Calculate mass per bin
    masses = calculate_mass(diameters, concentrations)

    # Calculate cumulative mass and find the diameter at 50% of the total mass
    cumulative_mass, diameter_at_50 = calculate_cumulative_mass(diameters, masses)
    diameters_at_50_mass.append(diameter_at_50)

    # Plot the mass distribution
    plt.plot(diameters, masses, label=f"Leg {entry['Leg_index']} ({entry['Date']})")
    plt.axvline(diameter_at_50, color='red', linestyle='--', alpha=0.5, 
                label=f"50% Mass at {diameter_at_50:.2f} µm (Leg {entry['Leg_index']})")

# Add labels, title, and legend
plt.xlabel('Diameter (µm)', fontsize=14, fontweight='bold')
plt.ylabel('Mass (arbitrary units)', fontsize=14, fontweight='bold')
plt.title('Mass Phase Space (Before Windspeed Binning)', fontsize=14, fontweight='bold')
plt.legend(fontsize=8, loc='upper right')
plt.grid()
plt.tight_layout()
plt.show()

# Print the diameters at 50% mass for all legs
print("Diameters at 50% Mass for Each Leg:")
for i, diameter in enumerate(diameters_at_50_mass, 1):
    print(f"Leg {i}: {diameter:.2f} µm")

#%%
# Make Y_BCB_calc which is 441 in length match with filtered_master_BCB_grH
#which has 306 entries. 
# Step 1: Create a set of (Date, BCB_start, BCB_stop) tuples from filtered_master_BCB_gRH
grh_leg_set = set(
    (entry["Date"], entry["BCB_start"], entry["BCB_stop"])
    for sublist in filtered_master_BCB_gRH for entry in sublist
)

# Step 2: Filter Y_BCB_calc based on (Date, BCB_start, BCB_stop) tuples
filtered_Y_BCB_calc = [
    entry for entry in Y_BCB_calc
    if (entry["Date"], entry["BCB_start"], entry["BCB_stop"]) in grh_leg_set
]

# Step 3: Print the results for verification
print(f"Total entries in Y_BCB_calc: {len(Y_BCB_calc)}")
print(f"Total entries in filtered_master_BCB_gRH: {sum(len(sublist) for sublist in filtered_master_BCB_gRH)}")  # Adjusted for nested lists
print(f"Total entries in filtered_Y_BCB_calc: {len(filtered_Y_BCB_calc)}")

# Ensure the filtered list matches the expected length
assert len(filtered_Y_BCB_calc) == sum(len(sublist) for sublist in filtered_master_BCB_gRH), "Filtered list length does not match grh list!"

#%%
problematic_legs = [
    ('2022-01-11', 1), ('2022-01-11', 2), ('2022-01-11', 3), ('2022-01-11', 4),
    ('2022-01-11', 8), ('2022-01-11', 9), ('2022-01-11', 14), ('2022-01-11', 16),
    ('2022-01-11', 17), ('2022-01-26', 16), ('2022-01-26', 14), ('2022-01-26', 12),
    ('2022-01-26', 15), ('2022-01-26', 13), ('2022-01-26', 11), ('2022-03-29', 0),
    ('2022-05-05', 7), ('2022-05-05', 3), ('2022-06-11', 4),
    ('2022-03-29', 9), ('2022-03-29', 9), ('2022-03-29', 0), ('2022-03-29', 6),
    ('2022-03-28', 4), ('2022-03-28', 0), ('2022-03-26', 15), ('2022-03-26', 6),
    ('2022-03-26', 0), ('2022-03-13', 8)
]
problematic_set = set(problematic_legs)
#%%
# Convert Y_BCB_calc to a DataFrame
Y_BCB_df = pd.DataFrame(filtered_Y_BCB_calc)

# Group by Date and assign Leg_index within each group
Y_BCB_df["Leg_index"] = Y_BCB_df.groupby("Date").cumcount()

# Verify the numbering
print(f"Sample data with Leg_index:\n{Y_BCB_df[['Date', 'Leg_index', 'BCB_start', 'BCB_stop']].head(20)}")
#%%
# Remove problematic legs
Y_BCB_clean = Y_BCB_df[
    ~Y_BCB_df[["Date", "Leg_index"]].apply(tuple, axis=1).isin(problematic_set)
].copy()

# Replace NaN values in concentrations with 0
Y_BCB_clean.fillna(0, inplace=True)

#%%


# Example bin diameters from filtered_master_BCB_ddry
common_bins = np.array(filtered_master_BCB_ddry[0]["filtered_ddry"])  # Raw bin diameters

# Function to calculate mass for each bin
def calculate_mass(diameter, concentration):
    return concentration * (np.pi / 6) * (diameter**3)

# Function to calculate cumulative mass and find 50% diameter
def calculate_cumulative_mass(diameters, masses):
    cumulative_mass = np.cumsum(masses)
    total_mass = np.sum(masses)
    normalized_cumulative_mass = cumulative_mass / total_mass
    # Interpolate to find diameter at 50% mass
    interpolation_func = interp1d(normalized_cumulative_mass, diameters, kind="linear", fill_value="extrapolate")
    diameter_at_50_percent = interpolation_func(0.5)
    return normalized_cumulative_mass, diameter_at_50_percent

# Prepare for plotting
plt.figure(figsize=(12, 8))
diameters_at_50_mass = []  # To store diameters at 50% mass for each leg

# Iterate through each leg in Y_BCB_clean
for i, row in Y_BCB_clean.iterrows():
    # Extract concentrations for each bin
    concentrations = np.array([row.get(f"Bin{j}_Y_mean", 0) for j in range(12, 30)])  # Replace NaN with 0

    # Calculate mass for each bin
    masses = calculate_mass(common_bins, concentrations)

    # Calculate cumulative mass and determine 50% mass diameter
    normalized_cumulative_mass, diameter_at_50 = calculate_cumulative_mass(common_bins, masses)
    diameters_at_50_mass.append(diameter_at_50)

    # Plot the cumulative mass curve for this leg
    plt.plot(common_bins, normalized_cumulative_mass, label=f"Leg {row['Leg_index']}, Date {row['Date']}")

# Add labels and legend
plt.xlabel("Diameter (µm)", fontsize=14, fontweight="bold")
plt.ylabel("Cumulative Mass (Normalized)", fontsize=14, fontweight="bold")
plt.title("Cumulative Mass Distribution (Raw Data)", fontsize=14, fontweight="bold")
plt.grid()
plt.tight_layout()
# plt.legend(fontsize=8, loc="upper left", ncol=2)
plt.show()

# Print 50% diameters for all legs
print("Diameters at 50% Mass for Each Leg:")
for i, diameter in enumerate(diameters_at_50_mass, 1):
    print(f"Leg {i}: {diameter:.2f} µm")
#%%
#average with just one line
# Initialize arrays to store cumulative masses and masses for averaging
all_masses = []

# Iterate through each leg in Y_BCB_clean
for _, row in Y_BCB_clean.iterrows():
    # Extract concentrations for each bin
    concentrations = np.array([row.get(f"Bin{j}_Y_mean", 0) for j in range(12, 30)])  # Replace NaN with 0

    # Calculate mass for each bin
    masses = calculate_mass(common_bins, concentrations)
    all_masses.append(masses)

# Convert all masses into a numpy array for easy averaging
all_masses = np.array(all_masses)

# Calculate the average mass for each bin
average_masses = np.mean(all_masses, axis=0)

# Calculate cumulative mass and determine the diameter at 50% mass
normalized_cumulative_mass, diameter_at_50 = calculate_cumulative_mass(common_bins, average_masses)

# Plot the average cumulative mass distribution
plt.figure(figsize=(12, 8))
plt.plot(common_bins, normalized_cumulative_mass, label=f"Average Distribution (50% Mass at {diameter_at_50:.2f} µm)", color='blue')

# Add labels and legend
plt.axvline(diameter_at_50, color='red', linestyle='--', alpha=0.7, label=f"50% Mass at {diameter_at_50:.2f} µm")
plt.xlabel("Diameter (µm)", fontsize=14, fontweight="bold")
plt.ylabel("Cumulative Mass (Normalized)", fontsize=14, fontweight="bold")
plt.title("Average Cumulative Mass Distribution (Raw Data)", fontsize=14, fontweight="bold")
plt.grid()
plt.tight_layout()
plt.legend(fontsize=10, loc="upper left")
plt.show()

# Print the 50% mass diameter
print(f"Diameter at 50% Mass (Average Distribution): {diameter_at_50:.2f} µm")







































































# %%

##Function to calculate the 50% mass diameter for fitted distribution using 
#slope and dry-intercept values for every leg pre-windspeed binning. 

def cumulative_mass(N0, D, d_max):
    # Integrand for mass calculation
    integrand = lambda d: N0 * np.exp(-d / D) * d**3
    cumulative_mass, _ = quad(integrand, 2, d_max)  # Integrate from 2 µm to d_max
    return cumulative_mass

# Function to calculate total mass (for normalization)
def total_mass(N0, D):
    integrand = lambda d: N0 * np.exp(-d / D) * d**3
    total_mass, _ = quad(integrand, 2, np.inf)  # Integrate from 2 µm to infinity
    return total_mass

# Prepare diameters for cumulative mass calculation
diameters = np.linspace(2, 50, 100)  # 2 µm to 50 µm with 100 points

# Initialize a list to store the diameter at 50% mass for each leg
diameters_at_50_mass = []

# Initialize a figure for plotting
plt.figure(figsize=(12, 8))

# Loop through each leg in the dataset
for i, row in filtered_combined_clean.iterrows():
    N0 = row['dryintercept']
    D = row['D']

    # Calculate total mass for normalization
    M_total = total_mass(N0, D)

    # Calculate cumulative mass at each diameter
    cumulative_masses = [cumulative_mass(N0, D, d) for d in diameters]

    # Normalize cumulative mass
    normalized_cumulative_mass = np.array(cumulative_masses) / M_total

    # Interpolate to find the diameter at 50% of the total mass
    interpolation_func = interp1d(normalized_cumulative_mass, diameters, kind='linear', fill_value='extrapolate')
    diameter_at_50 = interpolation_func(0.5)  # Diameter where 50% mass is reached
    diameters_at_50_mass.append(diameter_at_50)

    # Plot the cumulative mass distribution for this leg
    plt.plot(diameters, normalized_cumulative_mass, label=f"Leg {i}")
    plt.axvline(diameter_at_50, color='red', linestyle='--', alpha=0.5, 
                label=f"50% Mass at {diameter_at_50:.2f} µm (Leg {i})")

# Add labels, title, and legend
plt.xlabel('Diameter (µm)', fontsize=14, fontweight='bold')
plt.ylabel('Cumulative Mass (Normalized)', fontsize=14, fontweight='bold')
plt.title('Cumulative Mass Distribution (50% Mass)', fontsize=14, fontweight='bold')
# plt.legend(fontsize=8, loc='upper left', ncol=2)
plt.grid()
plt.tight_layout()
plt.show()

# Print the diameters at 50% mass for all legs
print("Diameters at 50% Mass for Each Leg:")
for i, diameter in enumerate(diameters_at_50_mass, 1):
    print(f"Leg {i}: {diameter:.2f} µm")

# %%

##Function to calculate the 50% mass diameter for fitted distribution using 
#slope and dry-intercept values for average or cumulative pre-windspeed binning.


# Function to calculate the cumulative mass up to a given diameter
def cumulative_mass(N0, D, d_max):
    # Integrand for mass calculation
    integrand = lambda d: N0 * np.exp(-d / D) * d**3
    cumulative_mass, _ = quad(integrand, 2, d_max)  # Integrate from 2 µm to d_max
    return cumulative_mass

# Function to calculate total mass (for normalization)
def total_mass(N0, D):
    integrand = lambda d: N0 * np.exp(-d / D) * d**3
    total_mass, _ = quad(integrand, 2, np.inf)  # Integrate from 2 µm to infinity
    return total_mass

# Prepare diameters for cumulative mass calculation
diameters = np.linspace(2, 50, 100)  # 2 µm to 50 µm with 100 points

# Initialize arrays to store cumulative masses
cumulative_mass_all_legs = np.zeros_like(diameters)

# Loop through each leg in the dataset to calculate cumulative masses
for i, row in filtered_combined_clean.iterrows():
    N0 = row['dryintercept']
    D = row['D']

    # Calculate total mass for normalization
    M_total = total_mass(N0, D)

    # Calculate cumulative mass at each diameter for this leg
    cumulative_masses = np.array([cumulative_mass(N0, D, d) for d in diameters]) / M_total

    # Add to the cumulative mass array
    cumulative_mass_all_legs += cumulative_masses

# Average cumulative mass across all legs
average_cumulative_mass = cumulative_mass_all_legs / len(filtered_combined_clean)

# Interpolate to find the diameter at 50% of the total mass
interpolation_func = interp1d(average_cumulative_mass, diameters, kind='linear', fill_value='extrapolate')
diameter_at_50 = interpolation_func(0.5)  # Diameter where 50% mass is reached

# Plot the average cumulative mass curve
plt.figure(figsize=(12, 8))
plt.plot(diameters, average_cumulative_mass, label='Average Cumulative Mass')
plt.axvline(diameter_at_50, color='red', linestyle='--', alpha=0.7, 
            label=f"50% Mass at {diameter_at_50:.2f} µm")

# Add labels, title, and legend
plt.xlabel('Diameter (µm)', fontsize=14, fontweight='bold')
plt.ylabel('Cumulative Mass (Normalized)', fontsize=14, fontweight='bold')
plt.title('Average Cumulative Mass Distribution (50% Mass)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid()
plt.tight_layout()
plt.show()

# Print the diameter at 50% mass
print(f"Diameter at 50% Average Cumulative Mass: {diameter_at_50:.2f} µm")

# %%
##Function to calculate the 50% mass diameter for fitted distribution using 
#slope and dry-intercept values for every leg post-windspeed binning both AND average. 


# Function to calculate cumulative mass up to a given diameter
def cumulative_mass(N0, D, d_max):
    integrand = lambda d: N0 * np.exp(-d / D) * d**3
    cumulative_mass, _ = quad(integrand, 2, d_max)  # Integrate from 2 µm to d_max
    return cumulative_mass

# Function to calculate total mass (for normalization)
def total_mass(N0, D):
    integrand = lambda d: N0 * np.exp(-d / D) * d**3
    total_mass, _ = quad(integrand, 2, np.inf)  # Integrate from 2 µm to infinity
    return total_mass

# Prepare diameters for cumulative mass calculation
diameters = np.linspace(2, 50, 100)  # 2 µm to 50 µm with 100 points

# Placeholder dictionaries for windspeed binning
windspeed_bins = {  # Example bins
    '0-3.6 m/s': [],
    '3.7-5.4 m/s': [],
    '5.5-7.5 m/s': [],
    '7.6+ m/s': []
}

# Step 1: Group the post windspeed binned data by windspeed bins
for i, row in filtered_combined_clean.iterrows():
    N0 = row['dryintercept']
    D = row['D']
    windspeed = row['Windspeed']
    
    # Assign each leg to the appropriate windspeed bin
    if 0 <= windspeed <= 3.6:
        windspeed_bins['0-3.6 m/s'].append((N0, D))
    elif 3.7 <= windspeed <= 5.4:
        windspeed_bins['3.7-5.4 m/s'].append((N0, D))
    elif 5.5 <= windspeed <= 7.5:
        windspeed_bins['5.5-7.5 m/s'].append((N0, D))
    elif windspeed > 7.5:
        windspeed_bins['7.6+ m/s'].append((N0, D))

# Step 2: Plot cumulative mass for each windspeed bin (all distributions)
plt.figure(figsize=(12, 8))
diameters_at_50_mass = []  # Store 50% mass diameters for each bin
for bin_label, legs in windspeed_bins.items():
    if not legs:
        continue
    
    # Average N0 and D for this bin
    bin_cumulative_mass = []
    for N0, D in legs:
        # Calculate cumulative mass for this leg
        cumulative_masses = [cumulative_mass(N0, D, d) for d in diameters]
        M_total = total_mass(N0, D)  # Total mass for normalization
        normalized_cumulative_mass = np.array(cumulative_masses) / M_total

        # Plot individual cumulative mass curves for this bin
        plt.plot(diameters, normalized_cumulative_mass, label=f"{bin_label}")

        # Interpolate to find the diameter at 50% mass for each leg
        interpolation_func = interp1d(normalized_cumulative_mass, diameters, kind='linear', fill_value='extrapolate')
        diameter_at_50 = interpolation_func(0.5)
        diameters_at_50_mass.append(diameter_at_50)
        plt.axvline(diameter_at_50, color='red', linestyle='--', alpha=0.5, label=f"50% Mass at {diameter_at_50:.2f} µm ({bin_label})")

# Add labels, title, and legend
plt.xlabel('Diameter (µm)', fontsize=14, fontweight='bold')
plt.ylabel('Cumulative Mass (Normalized)', fontsize=14, fontweight='bold')
plt.title('Cumulative Mass Distribution by Windspeed Bin', fontsize=14, fontweight='bold')
# plt.legend(fontsize=8, loc='upper left', ncol=2)
plt.grid()
plt.tight_layout()
plt.show()

# Step 3: Calculate average cumulative mass across all bins
average_cumulative_mass = np.zeros_like(diameters)
total_legs = sum(len(legs) for legs in windspeed_bins.values())
for bin_label, legs in windspeed_bins.items():
    for N0, D in legs:
        # Calculate cumulative mass for each leg and add to the total
        cumulative_masses = [cumulative_mass(N0, D, d) for d in diameters]
        average_cumulative_mass += np.array(cumulative_masses)

# Normalize the average cumulative mass
average_cumulative_mass /= total_legs
M_total_average = total_mass(np.mean([N0 for legs in windspeed_bins.values() for N0, D in legs]),
                             np.mean([D for legs in windspeed_bins.values() for N0, D in legs]))
average_cumulative_mass /= M_total_average

# Find 50% mass for the average curve
interpolation_func_avg = interp1d(average_cumulative_mass, diameters, kind='linear', fill_value='extrapolate')
diameter_at_50_avg = interpolation_func_avg(0.5)

# Plot the average cumulative mass
plt.figure(figsize=(12, 8))
plt.plot(diameters, average_cumulative_mass, label='Average Cumulative Mass')
plt.axvline(diameter_at_50_avg, color='red', linestyle='--', alpha=0.5, label=f"50% Mass at {diameter_at_50_avg:.2f} µm")

# Add labels, title, and legend
plt.xlabel('Diameter (µm)', fontsize=14, fontweight='bold')
plt.ylabel('Cumulative Mass (Normalized)', fontsize=14, fontweight='bold')
plt.title('Average Cumulative Mass Distribution (Post Windspeed Binning)', fontsize=14, fontweight='bold')
# plt.legend(fontsize=12)
plt.grid()
plt.tight_layout()
plt.show()

# %%
