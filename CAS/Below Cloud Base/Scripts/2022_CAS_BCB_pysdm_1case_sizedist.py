#%%
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pathlib
import statistics
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
import pickle
import glob
import sys
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
sys.modules['numpy._core'] = np
sys.modules['numpy._core.multiarray'] = np.core.multiarray
trial_case = "/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions/0.pickle"
with open(trial_case, "rb") as f:
    trial_data = pickle.load(f)
print("Trial Data Type:", type(trial_data))
if isinstance(trial_data, dict):
    print("Trial Data Keys:", trial_data.keys())
elif isinstance(trial_data, list):
    print(f"Length of list: {len(trial_data)}")

# %%
trial_rain = trial_data['surface precipitation']
print("Trial Rain Type:", type(trial_rain))
if isinstance(trial_rain, np.ndarray):
    print("Trial Rain Shape:", trial_rain.shape)
elif isinstance(trial_rain, dict):
    print("Keys inside 'surface precipitation':", trial_rain.keys())
dry_spec = trial_data['dry spectrum']
print(type(dry_spec))
print(getattr(dry_spec, 'shape', None))
dry_spec = trial_data['dry spectrum'].squeeze() 
print(dry_spec.shape)
mean_spectrum = dry_spec.mean(axis=0)  
#%%
wet_spec = trial_data['wet spectrum']
print(type(wet_spec))
print(getattr(wet_spec, 'shape', None))
wet_spec = trial_data['wet spectrum'].squeeze() 
print(wet_spec.shape)
mean_spectrum_wet = wet_spec.mean(axis=0)  
#%%
print(type(trial_data['t']))
print(np.shape(trial_data['t']))
print(trial_data['t'][:10]) 
#%%
print(type(trial_data['surface precipitation']))
print(np.shape(trial_data['surface precipitation']))
print(trial_data['surface precipitation'][:361])
# %%

#using Jason's spectrum of np.logspace (-7, -5, 101) and bin edges in radius. his units are 
#/m4

r_edges_m = np.logspace(-7, -5, 101)                    
r_centers_m = np.sqrt(r_edges_m[:-1] * r_edges_m[1:])  
dr_m = np.diff(r_edges_m)                              
dry_spec = np.asarray(trial_data['dry spectrum']).squeeze()  # [#/m⁴]
mean_dry = np.nanmean(dry_spec, axis=0)  # still #/m⁴
N_per_m3_per_m = mean_dry 
plt.figure(figsize=(7,4))
plt.semilogx(r_centers_m, N_per_m3_per_m, lw=2)
plt.xlabel('Dry Radius (m)', fontsize=17, fontweight='bold')
plt.ylabel('Number Concentration \n(m$^{-3}$ m$^{-1}$)',
           fontsize=17, fontweight='bold')
plt.yscale('log')
plt.grid(True, which='both', ls='--', alpha=0.4)
plt.tight_layout()
plt.show()
#%%
r_edges = np.logspace(-7, -5, 101) 
print("First 5 edges:", r_edges[:5])
print("Last 5 edges:",  r_edges[-5:])
print("Min edge:", r_edges.min(), " Max edge:", r_edges.max())

# %%
#printing all the bin centers in m 
r_edges_m = np.logspace(-7, -5, 101)
r_centers_m = np.sqrt(r_edges_m[:-1] * r_edges_m[1:])
for i, r in enumerate(r_centers_m, 1):
    print(f"Bin {i:3d}: {r:.3e} m")
r_centers_m=np.array(r_centers_m)


# %%
#converting from radius in m to diameter in um
d_centers_um = r_centers_m * 2 * 1e6
for i, d in enumerate(d_centers_um, 1):
    print(f"Bin {i:3d}: {d:.3f} µm")
d_centers_um = np.array(d_centers_um)
#%%
#converting from #/m4 to #/cm3/um
r_edges_m  = np.logspace(-7, -5, 101)
r_centers_m = np.sqrt(r_edges_m[:-1] * r_edges_m[1:])
d_centers_um = r_centers_m * 2 * 1e6
conversion_factor = 1e-12 
N_per_cm3_per_um = N_per_m3_per_m * conversion_factor
plt.figure(figsize=(7,4))
plt.semilogx(d_centers_um, N_per_cm3_per_um, lw=2)
plt.axvline(x=2.0, color='red', linestyle='--', linewidth=2,
            label='2 µm')
plt.xlabel('Dry Diameter (µm)', fontsize=17, fontweight='bold')
plt.ylabel('Number Concentration\n(cm$^{-3}$ µm$^{-1}$)',
           fontsize=17, fontweight='bold')
plt.yscale('log')
plt.title('CAS BCB February 15, 2022\n Leg 1\n Dry Size Distribution', fontsize=17, fontweight='bold')
plt.grid(True, which='both', ls='--', alpha=0.4)
plt.tight_layout()
plt.show()
#%%
#checking bin centers 

print("First 10 bin centers (radius in meters):")
for i, r in enumerate(r_centers_m[:10], 1):
    print(f"  Bin {i:2d}: {r:.3e} m")
print("\nFirst 10 bin centers (diameter in micrometers):")
for i, d in enumerate(d_centers_um[:100], 1):
    print(f"  Bin {i:2d}: {d:.3f} µm")

print(f"\nTotal bins: {len(d_centers_um)}")

#%%
# Bin edges in diameter (µm) from your radius edges
d_edges_um = r_edges_m * 2 * 1e6
d_widths_um = np.diff(d_edges_um)
total_conc = np.nansum(N_per_cm3_per_um * d_widths_um)
mask_2um = d_centers_um >= 2.0
conc_above_2um = np.nansum(N_per_cm3_per_um[mask_2um] *
                           d_widths_um[mask_2um])

print(f"Total concentration = {total_conc:.2f} cm^-3")
print(f"Concentration ≥2 µm = {conc_above_2um:.2f} cm^-3")
plt.figure(figsize=(7,4))
plt.semilogx(d_centers_um, N_per_cm3_per_um, lw=2)
plt.axvline(x=2.0, color='red', linestyle='--', linewidth=2,
            label='2 µm')
plt.xlabel('Dry Diameter (µm)', fontsize=17, fontweight='bold')
plt.ylabel('Number Concentration\n(cm$^{-3}$ µm$^{-1}$)',
           fontsize=17, fontweight='bold')
plt.yscale('log')
plt.title('CAS BCB January 11, 2022\nLeg 1\nDry Size Distribution',
          fontsize=17, fontweight='bold')
plt.grid(True, which='both', ls='--', alpha=0.4)
txt = (f"Total Concentration: {total_conc:.2f} cm$^{{-3}}$\n"
       f"Concentration >2 µm diameter: {conc_above_2um:.2f} cm$^{{-3}}$")
plt.text(0.58, 0.95, txt, transform=plt.gca().transAxes,
         fontsize=7, fontweight='bold', va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

plt.tight_layout()
plt.show()
#%%
target_date = "2022-02-15"
target_leg = 1 
entry = next((e for e in filtered_master_BCB_ddry if e['Date'] == target_date), None)

if entry is None:
    print(f"No entry found for {target_date}")
else:
    ddry_values = np.array(entry['ddry'])
    dN_dD_dry = np.array(entry['dN/dDdry'])
    valid = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry) & (dN_dD_dry > 0)

    plt.figure(figsize=(7, 4))
    plt.semilogx(ddry_values[valid], dN_dD_dry[valid], lw=2, color='tab:blue')
    plt.axvline(x=2.0, color='red', linestyle='--', lw=2, label='2 µm')
    plt.xlabel("Dry Diameter (µm)", fontsize=17, fontweight="bold")
    plt.ylabel("Number Concentration\n(cm$^{-3}$ µm$^{-1}$)",
               fontsize=17, fontweight="bold")
    plt.yscale("log")
    plt.title("CAS BCB Feb 15, 2022 – Leg 1\nRaw Dry Size Distribution",
              fontsize=17, fontweight="bold")
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

#%%
model_path = "/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions/0.pickle"
with open(model_path, "rb") as f:
    trial_data = pickle.load(f)
r_edges_m = np.logspace(-7, -5, 101)
r_centers_m = np.sqrt(r_edges_m[:-1] * r_edges_m[1:])
d_centers_um_model = r_centers_m * 2 * 1e6  # convert to µm diameter
dry_spec = np.asarray(trial_data["dry spectrum"]).squeeze()  # [#/m⁴]
mean_dry = np.nanmean(dry_spec, axis=0)
N_model = mean_dry * 1e-12  # → /cm³/µm
entry = next(e for e in filtered_master_BCB_ddry if e['Date'] == "2022-02-15")
d_centers_um_cas = np.array(entry['ddry'])
N_obs = np.array(entry['dN/dDdry'])
mask = ~np.isnan(d_centers_um_cas) & ~np.isnan(N_obs) & (N_obs > 0)
d_centers_um_cas, N_obs = d_centers_um_cas[mask], N_obs[mask]
interp_model = interp1d(d_centers_um_model, N_model,
                        bounds_error=False, fill_value=np.nan)
N_model_on_CAS_bins = interp_model(d_centers_um_cas)
plt.figure(figsize=(7,4))
plt.semilogx(d_centers_um_cas, N_model_on_CAS_bins, color='tab:blue', lw=2, label="Model (interpolated to CAS bins)")
plt.semilogx(d_centers_um_cas, N_obs, 'o-', color='red', lw=2, label="Observed (CAS)")
plt.axvline(2.0, color='k', linestyle='--', lw=1)
plt.xlabel("Dry Diameter (µm)", fontsize=17, fontweight='bold')
plt.ylabel("Number Concentration\n(cm$^{-3}$ µm$^{-1}$)", fontsize=17, fontweight='bold')
plt.yscale("log")
plt.title("Feb 15 2022 – Leg 1\nDry Size Distribution: Model vs CAS", fontsize=17, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, which='both', ls='--', alpha=0.4)
plt.tight_layout()
plt.show()
#%%
entry = next(e for e in filtered_master_BCB_ddry if e["Date"] == "2022-02-15")
d_centers_um_cas = np.array(entry["ddry"])
N_obs = np.array(entry["dN/dDdry"])
mask = ~np.isnan(d_centers_um_cas) & ~np.isnan(N_obs) & (N_obs > 0)
d_centers_um_cas, N_obs = d_centers_um_cas[mask], N_obs[mask]
cas_bin_edges = np.array([2.25, 2.75, 3.25, 3.75, 4.5, 5.75, 6.85, 7.55,
                          9.05, 11.4, 13.8, 17.5, 22.5, 27.5, 32.5, 37.5,
                          42.5, 47.5])
cas_bin_edges = np.append(cas_bin_edges, cas_bin_edges[-1] * 1.2)  # extend last edge slightly
binned_model = []
for i in range(len(cas_bin_edges) - 1):
    mask = (d_centers_um_model >= cas_bin_edges[i]) & (d_centers_um_model < cas_bin_edges[i + 1])
    if np.any(mask):
        binned_model.append(np.nanmean(N_model[mask]))
    else:
        binned_model.append(np.nan)
binned_model = np.array(binned_model)
plt.figure(figsize=(7,4))
plt.semilogx(d_centers_um_cas, binned_model[:len(d_centers_um_cas)],
             color="tab:blue", lw=2, label="Model (binned to CAS, ≤10 µm)")
plt.semilogx(d_centers_um_cas, N_obs, "o-", color="red", lw=2, label="Observed (CAS, ≤10 µm)")
plt.axvline(2.0, color="k", linestyle="--", lw=1)
plt.xscale("log")
plt.xlim(1, 10) 
plt.yscale("log")
plt.xlabel("Dry Diameter (µm)", fontsize=17, fontweight="bold")
plt.ylabel("Number Concentration\n(cm$^{-3}$ µm$^{-1}$)", fontsize=17, fontweight="bold")
plt.title("Feb 15 2022 – Leg 1\nDry Size Distribution (≤10 µm)", fontsize=17, fontweight="bold")
plt.legend(fontsize=10)
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.tight_layout()
plt.show()

#%%

r_edges_m = np.logspace(-7, -5, 101)
r_centers_m = np.sqrt(r_edges_m[:-1] * r_edges_m[1:])
d_centers_um_model = r_centers_m * 2 * 1e6  
dry_spec = np.asarray(trial_data["dry spectrum"]).squeeze() 
mean_dry = np.nanmean(dry_spec, axis=0)
N_model = mean_dry * 1e-12 
print("=== Jason's Model Dry Spectrum ===")
print(f"Min diameter: {d_centers_um_model.min():.3f} µm")
print(f"Max diameter: {d_centers_um_model.max():.3f} µm")
print(f"Nonzero bins: {(N_model > 0).sum()} / {len(N_model)}")
print("\nDiameter (µm)    N_model (#/cm³/µm)")
for d, n in zip(d_centers_um_model[::5], N_model[::5]):  
    print(f"{d:10.3f}        {n:10.3e}")
#%%
precip_rate = np.asarray(trial_data["surface precipitation"]).squeeze()  # [kg/m²/s ≡ mm/s]
time_s = np.asarray(trial_data["t"]).squeeze()                           # [s]
dt = np.median(np.diff(time_s))  # should be 10 s
print(f"Time step: {dt} s, Length: {len(time_s)} points, Duration: {time_s[-1]/60:.1f} min")
precip_rate_mmhr = precip_rate * 3600.0  # 1 s → 3600 s/hr
accum_precip_mm = np.cumsum(precip_rate * dt)  # result in mm
fig, ax1 = plt.subplots(figsize=(8,4))
color1 = "tab:blue"
ax1.plot(time_s/60, precip_rate_mmhr, color=color1, lw=2)
ax1.set_xlabel("Time (minutes)", fontsize=14, fontweight="bold")
ax1.set_ylabel("Precipitation Rate (mm hr$^{-1}$)", color=color1,
               fontsize=14, fontweight="bold")
ax1.tick_params(axis="y", labelcolor=color1)
ax1.grid(True, which="both", ls="--", alpha=0.4)
ax2 = ax1.twinx()
color2 = "tab:red"
ax2.plot(time_s/60, accum_precip_mm, color=color2, lw=2)
ax2.set_ylabel("Accumulated Precipitation (mm)", color=color2,
               fontsize=14, fontweight="bold")
ax2.tick_params(axis="y", labelcolor=color2)
plt.title("Model Surface Precipitation Time Series", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.show()
print(f"Total accumulated precipitation after 1 hour: {accum_precip_mm[-1]:.3f} mm")
#%%
surface_precip = np.asarray(trial_data["surface precipitation"]).squeeze()
print("Surface precip shape:", surface_precip.shape)
print("Min:", np.nanmin(surface_precip), "Max:", np.nanmax(surface_precip))
print("Mean:", np.nanmean(surface_precip))
print("Number of nonzero points:", np.count_nonzero(surface_precip))
nonzero_idx = np.where(surface_precip > 0)[0]
if len(nonzero_idx) == 0:
    print("⚠️ No nonzero surface precipitation values found.")
else:
    print("\nFirst few nonzero entries (index, value):")
    for i in nonzero_idx[:10]:
        print(f"  t={i:4d}  value={surface_precip[i]:.3e}")
#%%

import os
base_path = "/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions"
num_legs = 14
all_accum = []  
time_series = [] 

for i in range(num_legs):
    file_path = os.path.join(base_path, f"{i}.pickle")
    if not os.path.exists(file_path):
        print(f"⚠️ Missing file: {file_path}")
        continue

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    precip_rate = np.asarray(data["surface precipitation"]).squeeze()  # kg/m²/s = mm/s
    time_s = np.asarray(data["t"]).squeeze()

    if len(precip_rate) == 0:
        print(f"⚠️ Empty precip in {i}.pickle, skipping.")
        continue

    dt = np.median(np.diff(time_s))
    accum_precip_mm = np.cumsum(precip_rate * dt)
    total_precip_mm = accum_precip_mm[-1]

    all_accum.append(total_precip_mm)
    time_series.append((time_s, precip_rate, i))

    print(f"Leg {i:02d}: Total accumulation = {total_precip_mm:.3e} mm")
plt.figure(figsize=(9,5))

for time_s, precip_rate, i in time_series:
    plt.plot(time_s/60, precip_rate*3600, lw=1.8, label=f"Leg {i} ({all_accum[i]:.1e} mm total)")

plt.xlabel("Time (minutes)", fontsize=15, fontweight="bold")
plt.ylabel("Precipitation Rate (mm hr$^{-1}$)", fontsize=15, fontweight="bold")
plt.title("Surface Precipitation \n February 15\n 14 Legs", fontsize=16, fontweight="bold")
plt.yscale("log")
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend(fontsize=9, ncol=2, loc="upper right")
plt.tight_layout()
plt.show()
print("\n=== Summary of total accumulated precipitation ===")
for i, total in enumerate(all_accum):
    print(f"Leg {i:02d}: {total:.3e} mm")
#%%
r_edges_m = np.logspace(-7, -5, 101)
r_centers_m = np.sqrt(r_edges_m[:-1] * r_edges_m[1:])
d_centers_um = r_centers_m * 2 * 1e6
plt.figure(figsize=(8,5))

for i in range(num_legs):
    file_path = os.path.join(base_path, f"{i}.pickle")
    if not os.path.exists(file_path):
        print(f"⚠️ Missing file: {file_path}")
        continue

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    dry_spec = np.asarray(data["dry spectrum"]).squeeze()  # [#/m⁴]
    if dry_spec.size == 0:
        print(f"⚠️ Empty dry spectrum in {i}.pickle, skipping.")
        continue
    mean_dry = np.nanmean(dry_spec, axis=0)
    N_model = mean_dry * 1e-12 

    plt.semilogx(d_centers_um, N_model, lw=1.5, label=f"Leg {i}")

plt.xlabel("Dry Diameter (µm)", fontsize=16, fontweight="bold")
plt.ylabel("Number Concentration (cm$^{-3}$ µm$^{-1}$)", fontsize=16, fontweight="bold")
plt.title("Model Dry Spectra for All Legs", fontsize=16, fontweight="bold")
plt.yscale("log")
plt.xlim(0.2, 20)
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend(fontsize=9, ncol=2)
plt.tight_layout()
plt.show()
#%%
n0_10 = 0.812
D_10  = 0.828

def exponential(x, n0, D):
    return n0 * np.exp(-x / D)
x_fit = np.linspace(2, 10, 200)
y_fit = exponential(x_fit, n0_10, D_10)
plt.figure(figsize=(6, 5))
plt.semilogy(x_fit, y_fit, color="black", lw=1.2, alpha=0.8)
plt.xlabel("Dry Bin Centers Diameter (µm)", fontsize=13, fontweight="bold")
plt.ylabel("CAS Number Concentration (cm$^{-3}$ µm$^{-1}$)", fontsize=13, fontweight="bold")
plt.title("CAS Below Cloud Base – Feb 15 2022 (Leg 1)\nFitted Dry Size Distribution (≤10 µm)",
          fontsize=13, fontweight="bold")
plt.xlim(0, 10)
plt.ylim(1e-7, 1e1)
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.tight_layout()
plt.show()
#%%
n0_10 = 0.812
D_10  = 0.828
def exponential(D, n0, D_e):
    return n0 * np.exp(-D / D_e)
x_fit = np.linspace(2, 10, 200)
y_fit = exponential(x_fit, n0_10, D_10)
r_edges_m = np.logspace(-7, -5, 101)
r_centers_m = np.sqrt(r_edges_m[:-1] * r_edges_m[1:])
d_centers_um_model = r_centers_m * 2 * 1e6
dry_spec = np.asarray(trial_data["dry spectrum"]).squeeze()
mean_dry = np.nanmean(dry_spec, axis=0)
N_model = mean_dry * 1e-12
cas_bins = np.array([
    2.25, 2.75, 3.25, 3.75, 4.5, 5.75, 6.85, 7.55, 9.05,
    11.4, 13.8, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5
])
cas_bins_10 = cas_bins[cas_bins <= 10]
model_interp = np.interp(cas_bins_10, d_centers_um_model, N_model, left=np.nan, right=np.nan)
plt.figure(figsize=(6,5))
plt.semilogy(x_fit, y_fit, color="black", lw=1.2, alpha=0.8, label="Observational Exponential Fit (≤10 µm)")
plt.semilogy(cas_bins_10, model_interp, "o-", color="tab:blue", lw=1.5, label="Model Dry Distribution (interp. to CAS bins)")

plt.xlabel("Dry Bin Centers Diameter (µm)", fontsize=13, fontweight="bold")
plt.ylabel("CAS Number Concentration (cm$^{-3}$ µm$^{-1}$)", fontsize=13, fontweight="bold")
plt.title("CAS Below Cloud Base – Feb 15 2022 (Leg 1)\nFitted vs Model Dry Size Distribution (≤10 µm)",
          fontsize=13, fontweight="bold")
plt.xlim(0, 10)
plt.ylim(1e-7, 1e1)
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend(fontsize=9)
plt.tight_layout()
plt.show()
#%%
fits_feb15 = [
    {"n0": 0.812, "D": 0.828},
    {"n0": 1.073, "D": 0.757},
    {"n0": 5.232, "D": 0.574},
    {"n0": 1.819, "D": 0.798},
    {"n0": 2.263, "D": 0.744},
    {"n0": 1.146, "D": 0.944},
    {"n0": 0.672, "D": 0.944},
    {"n0": 1.919, "D": 0.546},
    {"n0": 1.704, "D": 0.790},
    {"n0": 2.018, "D": 0.934},
    {"n0": 2.269, "D": 0.846},
    {"n0": 1.554, "D": 0.958},
    {"n0": 2.728, "D": 0.570},
    {"n0": 1.058, "D": 0.709},
]

def exponential(D, n0, D_e):
    return n0 * np.exp(-D / D_e)
for leg_idx, fit in enumerate(fits_feb15):
    model_path = f"/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions/{leg_idx}.pickle"

    with open(model_path, "rb") as f:
        trial_data = pickle.load(f)
    dry_spec = np.asarray(trial_data["dry spectrum"]).squeeze()
    mean_dry = np.nanmean(dry_spec, axis=0)
    r_edges_m = np.logspace(-7, -5, 101)
    r_centers_m = np.sqrt(r_edges_m[:-1] * r_edges_m[1:])
    d_centers_um = r_centers_m * 2 * 1e6
    N_model = mean_dry * 1e-12  # [#/cm³/µm]
    mask = d_centers_um <= 10
    D_model = d_centers_um[mask]
    N_model = N_model[mask]
    n0, D_e = fit["n0"], fit["D"]
    D_fit = np.linspace(2, 10, 200)
    N_fit = exponential(D_fit, n0, D_e)
    plt.figure(figsize=(6, 5))
    plt.semilogy(D_model, N_model, color="tab:blue", lw=2, label=f"Model Spectrum (Leg {leg_idx+1})")
    plt.semilogy(D_fit, N_fit, "r--", lw=2, label=f"Exponential Fit\n$n_0$={n0:.3f}, D={D_e:.3f} µm")
    plt.axvline(2.0, color="k", ls="--", lw=1)
    plt.xlim(2, 10)
    plt.ylim(1e-7, 1e1)
    plt.xlabel("Dry Diameter (µm)", fontsize=13, fontweight="bold")
    plt.ylabel("Number Concentration (cm$^{-3}$ µm$^{-1}$)", fontsize=13, fontweight="bold")
    plt.title(f"Feb 15 2022 – Leg {leg_idx+1}\nModel vs Exponential Fit (≤10 µm)", fontsize=13, fontweight="bold")
    plt.legend(fontsize=9)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()
#%%
fits_feb15 = [
    {"n0": 0.812, "D": 0.828},
    {"n0": 1.073, "D": 0.757},
    {"n0": 5.232, "D": 0.574},
    {"n0": 1.819, "D": 0.798},
    {"n0": 2.263, "D": 0.744},
    {"n0": 1.146, "D": 0.944},
    {"n0": 0.672, "D": 0.944},
    {"n0": 1.919, "D": 0.546},
    {"n0": 1.704, "D": 0.790},
    {"n0": 2.018, "D": 0.934},
    {"n0": 2.269, "D": 0.846},
    {"n0": 1.554, "D": 0.958},
    {"n0": 2.728, "D": 0.570},
    {"n0": 1.058, "D": 0.709},
]

def exponential(D, n0, D_e):
    return n0 * np.exp(-D / D_e)
cas_bins = np.array([
    2.25, 2.75, 3.25, 3.75, 4.5, 5.75, 6.85, 7.55, 9.05,
    11.4, 13.8, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5
])
cas_bins_10 = cas_bins[cas_bins <= 10] 
for leg_idx, fit in enumerate(fits_feb15):
    model_path = f"/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions/{leg_idx}.pickle"
    with open(model_path, "rb") as f:
        trial_data = pickle.load(f)
    r_edges_m = np.logspace(-7, -5, 101)
    r_centers_m = np.sqrt(r_edges_m[:-1] * r_edges_m[1:])
    d_centers_um_model = r_centers_m * 2 * 1e6
    dry_spec = np.asarray(trial_data["dry spectrum"]).squeeze()
    mean_dry = np.nanmean(dry_spec, axis=0)
    N_model = mean_dry * 1e-12  # [#/cm³/µm]
    N_model_interp = np.interp(cas_bins_10, d_centers_um_model, N_model, left=np.nan, right=np.nan)
    n0, D_e = fit["n0"], fit["D"]
    N_fit_interp = exponential(cas_bins_10, n0, D_e)
    plt.figure(figsize=(6, 5))
    plt.semilogy(cas_bins_10, N_model_interp, "o-", color="tab:blue", lw=2,
                 label=f"Model (interpolated to CAS bins)")
    plt.semilogy(cas_bins_10, N_fit_interp, "r--", lw=2,
                 label=f"Exponential Fit\n$n_0$={n0:.3f}, D={D_e:.3f} µm")

    plt.axvline(2.0, color="k", ls="--", lw=1)
    plt.xlim(2, 10)
    plt.ylim(1e-7, 1e1)
    plt.xlabel("Dry Diameter (µm)", fontsize=13, fontweight="bold")
    plt.ylabel("Number Concentration (cm$^{-3}$ µm$^{-1}$)", fontsize=13, fontweight="bold")
    plt.title(f"Feb 15 2022 – Leg {leg_idx+1}\nModel vs Exponential Fit (≤10 µm)",
              fontsize=13, fontweight="bold")
    plt.legend(fontsize=9)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()
#%%

# Exponential parameters (your fits)
fits_feb15 = [
    {"n0": 0.812, "D": 0.828},
    {"n0": 1.073, "D": 0.757},
    {"n0": 5.232, "D": 0.574},
    {"n0": 1.819, "D": 0.798},
    {"n0": 2.263, "D": 0.744},
    {"n0": 1.146, "D": 0.944},
    {"n0": 0.672, "D": 0.944},
    {"n0": 1.919, "D": 0.546},
    {"n0": 1.704, "D": 0.790},
    {"n0": 2.018, "D": 0.934},
    {"n0": 2.269, "D": 0.846},
    {"n0": 1.554, "D": 0.958},
    {"n0": 2.728, "D": 0.570},
    {"n0": 1.058, "D": 0.709},
]

def exponential(D, n0, D_e):
    """Exponential function: N(D) = n0 * exp(-D / D_e)"""
    return n0 * np.exp(-D / D_e)

# Set up bin structure for Jason’s model (100 bins, diameter in µm)
r_edges_m = np.logspace(-7, -5, 101)
r_centers_m = np.sqrt(r_edges_m[:-1] * r_edges_m[1:])
d_centers_um = r_centers_m * 2 * 1e6

# Directory where all pickle files are stored
base_path = "/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions"
num_legs = len(fits_feb15)

for leg_idx, fit in enumerate(fits_feb15):
    file_path = os.path.join(base_path, f"{leg_idx}.pickle")
    if not os.path.exists(file_path):
        print(f"⚠️ Missing file: {file_path}")
        continue

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    dry_spec = np.asarray(data["dry spectrum"]).squeeze()  # [#/m⁴]
    if dry_spec.size == 0:
        print(f"⚠️ Empty dry spectrum in {leg_idx}.pickle, skipping.")
        continue

    # Compute mean spectrum and convert to [#/cm³/µm]
    mean_dry = np.nanmean(dry_spec, axis=0)
    N_model = mean_dry * 1e-12  # convert from #/m⁴ to #/cm³/µm

    # Compute exponential fit on same D grid (no interpolation)
    n0, D_e = fit["n0"], fit["D"]
    N_fit = exponential(d_centers_um, n0, D_e)

    # --- Plot ---
    plt.figure(figsize=(7, 5))
    plt.semilogy(d_centers_um, N_model, color="tab:blue", lw=2, label="Model Dry Spectrum")
    plt.semilogy(d_centers_um, N_fit, "r--", lw=2,
                 label=f"Exponential Fit\n$n_0$={n0:.3f}, D={D_e:.3f} µm")

    plt.axvline(2.0, color="k", ls="--", lw=1)
    plt.xlim(0.2, 20)
    plt.ylim(1e-7, 1e1)
    plt.xlabel("Dry Diameter (µm)", fontsize=14, fontweight="bold")
    plt.ylabel("Number Concentration (cm$^{-3}$ µm$^{-1}$)", fontsize=14, fontweight="bold")
    plt.title(f"Feb 15 2022 – Leg {leg_idx+1}\nModel vs Exponential Fit", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()
#%%


fits_feb15 = [
    {"n0": 0.812, "D": 0.828},
    {"n0": 1.073, "D": 0.757},
    {"n0": 5.232, "D": 0.574},
    {"n0": 1.819, "D": 0.798},
    {"n0": 2.263, "D": 0.744},
    {"n0": 1.146, "D": 0.944},
    {"n0": 0.672, "D": 0.944},
    {"n0": 1.919, "D": 0.546},
    {"n0": 1.704, "D": 0.790},
    {"n0": 2.018, "D": 0.934},
    {"n0": 2.269, "D": 0.846},
    {"n0": 1.554, "D": 0.958},
    {"n0": 2.728, "D": 0.570},
    {"n0": 1.058, "D": 0.709},
]

def exponential(D_um, n0, D_e):
    return n0 * np.exp(-D_um / D_e)

# --- model bins ---
r_edges_m = np.logspace(-7, -5, 101)
r_centers_m = np.sqrt(r_edges_m[:-1] * r_edges_m[1:])
d_centers_m = r_centers_m * 2
d_centers_um = d_centers_m * 1e6

base_path = "/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions"

for leg_idx, fit in enumerate(fits_feb15):
    file_path = os.path.join(base_path, f"{leg_idx}.pickle")
    if not os.path.exists(file_path):
        print(f"⚠️ Missing file: {file_path}")
        continue

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    dry_spec = np.asarray(data["dry spectrum"]).squeeze()  # [#/m⁴]
    if dry_spec.size == 0:
        continue

    mean_dry = np.nanmean(dry_spec, axis=0)  # [#/m⁴]
    N_model_m4 = mean_dry

    # --- cumulative from RIGHT→LEFT in meters ---
    D_m = d_centers_m
    N_rev = np.flip(N_model_m4)
    D_rev = np.flip(D_m)
    N_cum_rev = np.zeros_like(N_rev)
    N_cum_rev[1:] = np.cumsum(
        0.5 * (N_rev[1:] + N_rev[:-1]) * np.diff(D_rev)
    )  # integrates over diameter [m], yields [#/m³]
    N_model_cum_m3 = np.flip(N_cum_rev)

    # convert to #/cm³
    N_model_cum_cm3 = N_model_cum_m3 * 1e-6

    # restrict to ≤10 µm
    mask_10 = d_centers_um <= 10
    D_plot = d_centers_um[mask_10]
    N_model_cum_cm3 = N_model_cum_cm3[mask_10]

    print(f"Leg {leg_idx+1}: cumulative max = {np.nanmax(N_model_cum_cm3):.3e}")

    # exponential fit (same units as before)
    n0, D_e = fit["n0"], fit["D"]
    D_fit = np.linspace(2, 10, 300)
    N_fit = exponential(D_fit, n0, D_e)

    # --- Plot ---
    plt.figure(figsize=(7, 5))
    plt.semilogy(D_plot, N_model_cum_cm3, color="tab:blue", lw=2,
                 label="Cumulative Model (Right→Left)")
    plt.semilogy(D_fit, N_fit, "r--", lw=2,
                 label=f"Exponential Fit\n$n_0$={n0:.3f}, D={D_e:.3f} µm")

    plt.axvline(2.0, color="k", ls="--", lw=1)
    plt.xlim(2, 10)
    plt.ylim(1e-3, 1e3)
    plt.xlabel("Dry Diameter (µm)", fontsize=14, fontweight="bold")
    plt.ylabel("Cumulative Number (cm$^{-3}$)", fontsize=14, fontweight="bold")
    plt.title(f"Feb 15 2022 – Leg {leg_idx+1}\nCumulative Model (Right→Left) vs Exponential Fit (≤10 µm)",
              fontsize=13, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


# %%
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

# --- Leg 0 exponential fit parameters (your Feb15 list, first entry) ---
n0, D_e = 0.812, 0.828  # from fits_feb15[0]

def exponential(D_um, n0, D_e):
    return n0 * np.exp(-D_um / D_e)

# --- Jason's model bin structure ---
r_edges_m = np.logspace(-7, -5, 101)                     # 100 bins
r_centers_m = np.sqrt(r_edges_m[:-1] * r_edges_m[1:])    # radius centers
d_centers_m = r_centers_m * 2
d_centers_um = d_centers_m * 1e6                         # µm centers

# --- Load leg 0 file ---
base_path = "/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions"
file_path = os.path.join(base_path, "0.pickle")

with open(file_path, "rb") as f:
    data = pickle.load(f)

dry_spec = np.asarray(data["dry spectrum"]).squeeze()  # [#/m⁴]
mean_dry = np.nanmean(dry_spec, axis=0)

# --- Convert to #/cm³/µm for plotting ---
N_model = mean_dry * 1e-12  # (#/m⁴) * (1e-6 m³/cm³) * (1e6 µm/m) = 1e-12

# --- Compute cumulative (Right→Left) directly on Jason's bins ---
D = d_centers_um
N_rev = np.flip(N_model)
D_rev = np.flip(D)
# integrate using trapezoidal rule in µm space
N_cum_rev = np.zeros_like(N_rev)
for i in range(1, len(D_rev)):
    N_cum_rev[i] = N_cum_rev[i-1] + 0.5 * (N_rev[i] + N_rev[i-1]) * abs(D_rev[i] - D_rev[i-1])

N_model_cum = np.flip(N_cum_rev)

# --- Print diagnostics ---
print(f"Leg 0: raw N_model range = {np.nanmin(N_model):.3e} – {np.nanmax(N_model):.3e}")
print(f"Leg 0: cumulative N range = {np.nanmin(N_model_cum):.3e} – {np.nanmax(N_model_cum):.3e}")

# --- Exponential fit for comparison ---
D_fit = np.linspace(2, 10, 300)
N_fit = exponential(D_fit, n0, D_e)

# --- Plot ---
plt.figure(figsize=(7, 5))
plt.semilogy(D, N_model_cum, color="tab:blue", lw=2, label="Cumulative Model (Right→Left)")
plt.semilogy(D_fit, N_fit, "r--", lw=2, label=f"Exponential Fit\n$n_0$={n0:.3f}, D={D_e:.3f} µm")

plt.axvline(2.0, color="k", ls="--", lw=1)
plt.xlim(2, 10)
plt.ylim(1e-6, 1e3)
plt.xlabel("Dry Diameter (µm)", fontsize=14, fontweight="bold")
plt.ylabel("Cumulative Number (cm$^{-3}$)", fontsize=14, fontweight="bold")
plt.title("Feb 15 2022 – Leg 0\nCumulative Model (Right→Left) vs Exponential Fit (≤ 10 µm)",
          fontsize=13, fontweight="bold")
plt.legend(fontsize=10)
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.tight_layout()
plt.show()

# %%
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

# --- Exponential fit parameters for each leg ---
fits_feb15 = [
    {"n0": 0.812, "D": 0.828},
    {"n0": 1.073, "D": 0.757},
    {"n0": 5.232, "D": 0.574},
    {"n0": 1.819, "D": 0.798},
    {"n0": 2.263, "D": 0.744},
    {"n0": 1.146, "D": 0.944},
    {"n0": 0.672, "D": 0.944},
    {"n0": 1.919, "D": 0.546},
    {"n0": 1.704, "D": 0.790},
    {"n0": 2.018, "D": 0.934},
    {"n0": 2.269, "D": 0.846},
    {"n0": 1.554, "D": 0.958},
    {"n0": 2.728, "D": 0.570},
    {"n0": 1.058, "D": 0.709},
]

def exponential(D_um, n0, D_e):
    return n0 * np.exp(-D_um / D_e)

# --- Jason’s original 100-bin grid (diameters in µm) ---
r_edges_m = np.logspace(-7, -5, 101)
r_centers_m = np.sqrt(r_edges_m[:-1] * r_edges_m[1:])
d_centers_m = r_centers_m * 2
d_centers_um = d_centers_m * 1e6

base_path = "/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions"

# --- Loop through each leg ---
for leg_idx, fit in enumerate(fits_feb15):
    file_path = os.path.join(base_path, f"{leg_idx}.pickle")
    if not os.path.exists(file_path):
        print(f"⚠️ Missing file: {file_path}")
        continue

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    dry_spec = np.asarray(data["dry spectrum"]).squeeze()  # [#/m⁴]
    if dry_spec.size == 0:
        print(f"⚠️ Empty dry spectrum in leg {leg_idx}")
        continue

    # --- Convert to #/cm³ µm⁻¹ for plotting ---
    mean_dry = np.nanmean(dry_spec, axis=0)
    N_model = mean_dry * 1e-12

    # --- Compute cumulative (Right→Left) on 100 bins ---
    D = d_centers_um
    N_rev = np.flip(N_model)
    D_rev = np.flip(D)
    N_cum_rev = np.zeros_like(N_rev)
    N_cum_rev[1:] = np.cumsum(
        0.5 * (N_rev[1:] + N_rev[:-1]) * np.abs(np.diff(D_rev))
    )
    N_model_cum = np.flip(N_cum_rev)

    print(f"Leg {leg_idx+1}: cumulative max = {np.nanmax(N_model_cum):.3e}")

    # --- Exponential fit for comparison ---
    n0, D_e = fit["n0"], fit["D"]
    D_fit = np.linspace(2, 10, 300)
    N_fit = exponential(D_fit, n0, D_e)

    # --- Plot each leg separately ---
    plt.figure(figsize=(7, 5))
    plt.semilogy(D, N_model_cum, color="tab:blue", lw=2,
                 label="Cumulative Model (Right→Left)")
    plt.semilogy(D_fit, N_fit, "r--", lw=2,
                 label=f"Exponential Fit\n$n_0$={n0:.3f}, D={D_e:.3f} µm")

    plt.axvline(2.0, color="k", ls="--", lw=1)
    plt.xlim(2, 10)
    plt.ylim(1e-6, 1e3)
    plt.xlabel("Dry Diameter (µm)", fontsize=14, fontweight="bold")
    plt.ylabel("Cumulative Number (cm$^{-3}$)", fontsize=14, fontweight="bold")
    plt.title(f"Feb 15 2022 – Leg {leg_idx+1}\nCumulative Model (Right→Left) vs Exponential Fit (≤ 10 µm)",
              fontsize=13, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

# %%
fits_feb15 = [
    {"n0": 0.812, "D": 0.828},
    {"n0": 1.073, "D": 0.757},
    {"n0": 5.232, "D": 0.574},
    {"n0": 1.819, "D": 0.798},
    {"n0": 2.263, "D": 0.744},
    {"n0": 1.146, "D": 0.944},
    {"n0": 0.672, "D": 0.944},
    {"n0": 1.919, "D": 0.546},
    {"n0": 1.704, "D": 0.790},
    {"n0": 2.018, "D": 0.934},
    {"n0": 2.269, "D": 0.846},
    {"n0": 1.554, "D": 0.958},
    {"n0": 2.728, "D": 0.570},
    {"n0": 1.058, "D": 0.709},
]

def N_exp(D_um, n0, De):
    return n0 * np.exp(-D_um / De)
def cumulative_right_to_left(D_um, N_diff):
    D_rev = np.flip(D_um)
    N_rev = np.flip(N_diff)
    cum_rev = np.zeros_like(N_rev)
    cum_rev[1:] = np.cumsum(0.5 * (N_rev[1:] + N_rev[:-1]) * np.abs(np.diff(D_rev)))
    return np.flip(cum_rev)
r_edges_m = np.logspace(-7, -5, 101)
r_centers_m = np.sqrt(r_edges_m[:-1] * r_edges_m[1:])
D_um = (r_centers_m * 2) * 1e6 
base_path = "/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions"
for leg_idx, fit in enumerate(fits_feb15):
    fp = os.path.join(base_path, f"{leg_idx}.pickle")
    if not os.path.exists(fp):
        print(f"⚠️ Missing file: {fp}")
        continue

    with open(fp, "rb") as f:
        trial = pickle.load(f)

    dry_spec = np.asarray(trial["dry spectrum"]).squeeze() 
    if dry_spec.size == 0:
        print(f"⚠️ Empty dry spectrum in leg {leg_idx}")
        continue
    N_model = (np.nanmean(dry_spec, axis=0) / 2) * 1e-12
    N_model_cum = cumulative_right_to_left(D_um, N_model)
    n0, De = fit["n0"], fit["D"]
    N_fit_diff = N_exp(D_um, n0, De)
    N_fit_cum  = cumulative_right_to_left(D_um, N_fit_diff)
    m10 = D_um <= 10
    Dx   = D_um[m10]
    Mcum = N_model_cum[m10]
    Fcum = N_fit_cum[m10]
    ymin = max(1e-6, np.nanmin([Mcum[Mcum>0].min() if np.any(Mcum>0) else 1e-6,
                                Fcum[Fcum>0].min() if np.any(Fcum>0) else 1e-6]))
    ymax = np.nanmax([Mcum.max(), Fcum.max(), 1e0]) * 1.2

    plt.figure(figsize=(7,5))
    plt.semilogy(Dx, Mcum, lw=2, label="Cumulative Model (Right→Left)")
    plt.semilogy(Dx, Fcum, "r--", lw=2,
                 label=f"Cumulative Exponential Fit\n$n_0$={n0:.3f}, D={De:.3f} µm")
    plt.axvline(2.0, color="k", ls="--", lw=1)
    plt.xlim(2, 10)
    plt.ylim(ymin, ymax)
    plt.xlabel("Dry Diameter (µm)", fontsize=14, fontweight="bold")
    plt.ylabel("Cumulative Number (cm$^{-3}$)", fontsize=14, fontweight="bold")
    plt.title(f"Feb 15 2022 – Leg {leg_idx+1}\nCumulative Model vs Cumulative Fit (≤ 10 µm)",
              fontsize=13, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()
# %%

n0 = 0.812
De = 0.828
D = np.linspace(0, 10, 400)
N_diff = n0 * np.exp(-D / De)
N_cum = n0 * De * np.exp(-D / De)
plt.figure(figsize=(6,4))
plt.semilogy(D, N_diff, 'r--', lw=2, label='Differential $N(D)$')
plt.semilogy(D, N_cum, 'b-',  lw=2, label='Cumulative $N_{cum}(D)$')
plt.xlabel("Dry Diameter (µm)", fontsize=13, fontweight='bold')
plt.ylabel("Number (cm$^{-3}$ µm$^{-1}$ or cm$^{-3}$)", fontsize=13, fontweight='bold')
plt.title("Exponential Fit – Log Scale", fontsize=13, fontweight='bold')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()
plt.figure(figsize=(6,4))
plt.plot(D, N_diff, 'r--', lw=2, label='Differential $N(D)$')
plt.plot(D, N_cum, 'b-',  lw=2, label='Cumulative $N_{cum}(D)$')
plt.xlabel("Dry Diameter (µm)", fontsize=13, fontweight='bold')
plt.ylabel("Number", fontsize=13, fontweight='bold')
plt.title("Exponential Fit – Linear Scale", fontsize=13, fontweight='bold')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()
# %%
