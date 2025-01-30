#%%
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import datetime
import pathlib
import statistics
import mpu
import shutil
import glob
import os
import re
import math
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from scipy.optimize import curve_fit

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
    
        #     if date == '2022-01-15':
        #         print(f"\nFirst row for date {'2022-01-15'}:")
        #         print(df_sum.head())
        # except Exception as e:
        #     print(f"Error processing file: {file_path}, Date: {date}")
        #     print(f"Error message: {str(e)}")
         
        # print(date)
        # time = df_sum['Time_mid']
        # print("yes")  
        
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
        # timestop=df_legs['  Time_Stop']
        # timestart=df_legs['Time_Start']
        # legindex=df_legs['  LegIndex']

        # print(f"Before filtering:\n{df_legs[['  LegIndex', 'Time_Start']]}")
        leg_index_02 = df_legs[df_legs['  LegIndex'] % 100 == 2]
        leg_index_06 = df_legs[df_legs['  LegIndex'] % 100 == 6]
        leg_dictionary['LegIndex_02']['StartTimes'].extend(leg_index_02['Time_Start'].tolist())
        leg_dictionary['LegIndex_02']['StopTimes'].extend(leg_index_02['  Time_Stop'].tolist())
        leg_dictionary['LegIndex_06']['StartTimes'].extend(leg_index_06['Time_Start'].tolist())
        leg_dictionary['LegIndex_06']['StopTimes'].extend(leg_index_06['  Time_Stop'].tolist())
    #print(leg_data)
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

# for i, twoDS_flight in enumerate(twoDS):
#     times1 = twoDS_flight.Time_Start.values
#     N_total = twoDS_flight['N-total_2DS'].values

        # if date >= '2022-01-11':
        #         print(f"\nFirst row for date {'2022-01-11'}:")
        
        # print(date)
        # time = df_2DS['Time_Start']
        # print("yes")
# %%
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
master_CAS_Min = []
leg_info = []

for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict = leg_data[i]

    flight_date = leg_dict['Date']
    Min_start = leg_dict['LegIndex_06']['StartTimes']
    Min_stop = leg_dict['LegIndex_06']['StopTimes']

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

    total_Min_means = []

    for k in range(len(Min_start)):
        start20 = int(Min_start[k])
        end20 = Min_stop[k]

        data_labels = []
        Min_means = []  # List to store the means for this Min interval

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
                Min_means.append(bin_values)

            if Min_means:
                Min_means = np.mean(Min_means, axis=0)  # Calculate means for this interval
                total_Min_means.append(Min_means)  # Add to the total Min means list

            leg_info.append({
                'Date': date,
                'Min_start': start20,
                'Min_stop': end20,
                'Data_Labels': data_labels,
            })
    
    master_CAS_Min.append(total_Min_means)

# Print leg_info or use it as needed
for leg in leg_info:
    print(f"Date: {leg['Date']}, Start: {leg['Min_start']}, Stop: {leg['Min_stop']}, Data Labels: {leg['Data_Labels']}")

# %%
master_CAS_Min = []
leg_info = []

for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict = leg_data[i]

    flight_date = leg_dict['Date']
    Min_start = leg_dict['LegIndex_06']['StartTimes']
    Min_stop = leg_dict['LegIndex_06']['StopTimes']

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

    total_Min_means = []

    for k in range(len(Min_start)):
        start20 = int(Min_start[k])
        end20 = Min_stop[k]

        bin_means = {
            'Date': date,
            'Min_start': start20,
            'Min_stop': end20,
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

            total_Min_means.append(bin_means)

    master_CAS_Min.append(total_Min_means)

# Print or use master_CAS_Min as needed
for item in master_CAS_Min:
    for bin_means in item:
        print(f"Date: {bin_means['Date']}, Start: {bin_means['Min_start']}, Stop: {bin_means['Min_stop']}")
        for bin_label in range(12, 30):
            for label in ['Y', 'N']:
                bin_key = f'Bin{bin_label}_{label}_mean'
                if bin_means[bin_key]:
                    mean_value = np.nanmean(bin_means[bin_key])
                    print(f"   Bin{bin_label}_{label} Mean: {mean_value}")
                else:
                    print(f"   Bin{bin_label}_{label} Mean: No valid data")
# %%
Y_Min_calc = []
N_Min_calc = []

for flight_data in master_CAS_Min:
    for bin_means in flight_data:
        Y_calc = {'Date': bin_means['Date'], 'Min_start': bin_means['Min_start'], 'Min_stop': bin_means['Min_stop']}
        N_calc = {'Date': bin_means['Date'], 'Min_start': bin_means['Min_start'], 'Min_stop': bin_means['Min_stop']}
        
        for bin_label in range(12, 30):
            bin_key_Y = f'Bin{bin_label}_Y_mean'
            bin_key_N = f'Bin{bin_label}_N_mean'

            Y_calc[bin_key_Y] = np.nanmean(bin_means[bin_key_Y]) * Logg[bin_label - 12]
            N_calc[bin_key_N] = np.nanmean(bin_means[bin_key_N]) * Logg[bin_label - 12]

        Y_Min_calc.append(Y_calc)
        N_Min_calc.append(N_calc)
# %%
all_bin_means = []
dates=[]
for entry in Y_Min_calc:
    bin_means = []
    for i in range(12, 30):  # Bins from 12 to 29
        key = f'Bin{i}_Y_mean'
        bin_means.append(entry.get(key, np.nan))
    all_bin_means.append(bin_means)
    dates.append(entry['Date'])
# colors = cm.viridis(np.linspace(0, 1, len(dates)))

# # Convert to numpy arrays for plotting
# bin_center = np.array(bin_center)
# all_bin_means = np.array(all_bin_means)

# # Flatten the list of bin means for scatter plotting
# flattened_bin_means = all_bin_means.flatten()

# # Scatter plot
# plt.figure(figsize=(10, 6))
# for bin_means in all_bin_means:
#     plt.scatter(bin_center, bin_means, color='b', marker='o')

unique_dates = sorted(set(dates))
colors = cm.viridis(np.linspace(0, 1, len(unique_dates)))

# Create a dictionary to map each date to a color
date_color_map = {date: color for date, color in zip(unique_dates, colors)}

# Scatter plot
plt.figure(figsize=(10, 6))
added_dates = set()  # To track which dates have been added to the legend

for bin_means, date in zip(all_bin_means, dates):
    color = date_color_map[date]
    bin_means = np.array(bin_means)
    valid_indices = ~np.isnan(bin_means)
    
    if date not in added_dates:
        plt.scatter(np.array(bin_center)[valid_indices], bin_means[valid_indices], color=color, marker='o', label=date)
        added_dates.add(date)
    else:
        plt.scatter(np.array(bin_center)[valid_indices], bin_means[valid_indices], color=color, marker='o')


plt.xlabel('Bin centers diameter (um)', fontsize=12, fontweight='bold')
plt.ylabel('Clear mean droplet concentration \n (/cm^3)', fontsize=12, fontweight='bold')
plt.title('Minimum altitude clear sky distribution \n January-June 2022', fontsize=12, fontweight='bold')
plt.yscale('log')
plt.xticks(np.arange(0, 50, 2))
num_cols = 7
plt.legend(title='Date', ncol=num_cols, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True)
plt.tight_layout()
plt.ylim(10**-5, 10**1)
plt.show()

# %%
## Begin converting to dry size distribution

#%%

##Start by importing humidity data 

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
    print(fname_h20)
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
# for date in dates_h20:
#     datestr = date.replace('-', '')
#     fname_h20 = sorted(glob.glob(f'/home/disk/eos4/kathem24/activate/data/DLH_H20/csv/ACTIVATE-DLH-H2O_HU25_{datestr}_R*.csv'), reverse=True)

#     run = 1
#     for file_path in fname_h20: 
#         num_file_paths = len(fname_h20)

#         if date >= '2022-01-11':
#             df_h20 = pd.read_csv(file_path, skiprows=36, quoting=csv.QUOTE_NONE)
#         for col_ in col_name_h20:
#             if col_ in df_h20.columns:
#                 df_h20.columns = df_h20.columns.str.strip('"')
#                 df_h20[col_] = pd.to_numeric(df_h20[col_], errors='coerce')
#                 df_h20.replace([-9999, -9999.00], np.NaN, inplace=True)
#         for col in ['Time_Start', 'H2O_DLH', 'RHi_DLH', 'RHw_DLH']:
#             if df_h20[col].dtype == 'O': 
#                 df_h20[col] = df_h20[col].str.strip('"')
    if len(frames) > 1:
        df_h20_combined = pd.concat(frames, ignore_index=True)
        print(f"Combined {len(frames)} files for date {date}")

    else:
        df_h20_combined = frames[0]
        print(f"Only one file found for date {date}")
    h20.append(df_h20_combined)
    print(df_h20_combined.head(10))  
    print(df_h20_combined.tail(10))
        # if num_file_paths==2:
        #     if run==1:
        #         df1 = df_h20 
        #     elif run==2:
        #         df2 = df_h20 
        #         frames = [df2,df1]
        #         df_h20 = pd.concat(frames)
        #         h20.append(df_h20)
        #         break

        # if num_file_paths ==1:
        #     print("YESSSSSSSSSSSSSSSSSSSSSSSSSSSSs")
        #     h20.append(df_h20)

        # run = run+1  
        
# %%
##Fitting an exponential to the size distribution to give No and D* in our wet/
#ambient size distribution 

#No is the intercept 
#D* is the efolding diameter

#%%

def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

# Your existing data processing code
all_bin_means = []
dates = []
for entry in Y_Min_calc:
    bin_means = []
    for i in range(12, 30):  # Bins from 12 to 29
        key = f'Bin{i}_Y_mean'
        bin_means.append(entry.get(key, np.nan))
    all_bin_means.append(bin_means)
    dates.append(entry['Date'])

# Get unique dates and colors for plotting
unique_dates = sorted(set(dates))
colors = cm.viridis(np.linspace(0, 1, len(unique_dates)))

# Create a dictionary to map each date to a color
date_color_map = {date: color for date, color in zip(unique_dates, colors)}

# Scatter plot
plt.figure(figsize=(10, 6))
added_dates = set()  # To track which dates have been added to the legend

for bin_means, date in zip(all_bin_means, dates):
    color = date_color_map[date]
    bin_means = np.array(bin_means)
    valid_indices = ~np.isnan(bin_means)
    
    if date not in added_dates:
        plt.scatter(np.array(bin_center)[valid_indices], bin_means[valid_indices], color=color, marker='o', label=date)
        added_dates.add(date)
    else:
        plt.scatter(np.array(bin_center)[valid_indices], bin_means[valid_indices], color=color, marker='o')

# Flatten the bin_center and all_bin_means for fitting
flat_bin_centers = np.tile(bin_center, len(all_bin_means))
flat_bin_means = np.array(all_bin_means).flatten()

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
plt.title('Minimum altitude clear sky distribution \n January-June 2022', fontsize=12, fontweight='bold')
# plt.yscale('log')
plt.xticks(np.arange(0, 50, 5))
num_cols = 7
plt.legend(title='Date', ncol=num_cols, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True)
plt.tight_layout()
# plt.ylim(10**-5, 10**1)
plt.show()

# %%

# %%
## fitting an exponential to each leg, rather than a size distribution
##of the average of each leg 


# Define the exponential function based on your form
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)

# Your existing data processing code
all_bin_means = []
dates = []
for entry in Y_Min_calc:
    bin_means = []
    for i in range(12, 30):  # Bins from 12 to 29
        key = f'Bin{i}_Y_mean'
        bin_means.append(entry.get(key, np.nan))
    all_bin_means.append(bin_means)
    dates.append(entry['Date'])

# Define the color purple
purple_color = 'purple'

# Scatter plot
plt.figure(figsize=(10, 6))
added_dates = set()  # To track which dates have been added to the legend

# Loop through each leg's data
for bin_means, date in zip(all_bin_means, dates):
    bin_means = np.array(bin_means)
    valid_indices = ~np.isnan(bin_means)
    bin_centers = np.array(bin_center)[valid_indices]
    bin_means = bin_means[valid_indices]
    
    # Skip if bin_means is empty
    if bin_means.size == 0:
        print(f"No valid data for date {date}")
        continue
    
    # Fit the exponential function to the data of each leg
    try:
        popt, pcov = curve_fit(exponential, bin_centers, bin_means, p0=(1, 1))
        
        # Extract the intercept (n0) and e-folding diameter (D)
        n0 = popt[0]
        D = popt[1]
        
        # Print the intercept and e-folding diameter for each leg
        print(f"Date: {date}")
        print(f"  Intercept (n0): {n0:.2e}")
        print(f"  E-folding diameter (D): {D:.2f} um")
        
        # Plot the fitted curve for each leg
        x_fit = np.linspace(min(bin_centers), max(bin_centers), 100)
        y_fit = exponential(x_fit, *popt)
        plt.plot(x_fit, y_fit, color=purple_color, label=f'{date} fit: y = {n0:.2e} * exp(-x / {D:.2f})')

    except RuntimeError:
        print(f"Fit could not be performed for date {date}")
    
    # Plot the data points for each leg
    if date not in added_dates:
        plt.scatter(bin_centers, bin_means, color=purple_color, marker='o', label=date)
        added_dates.add(date)
    else:
        plt.scatter(bin_centers, bin_means, color=purple_color, marker='o')

# Add labels, title, and legend
plt.xlabel('Bin centers diameter (um)', fontsize=12, fontweight='bold')
plt.ylabel('Clear mean droplet concentration \n (/cm^3/um)', fontsize=12, fontweight='bold')
plt.title('Min Alt clear sky distribution \n January-June 2022', fontsize=12, fontweight='bold')
# plt.yscale('log')
plt.xticks(np.arange(0, 50, 5))
# num_cols = 7
# plt.legend(title='Date', ncol=num_cols, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True)
plt.tight_layout()
# plt.ylim(10**-5, 10**1)
plt.show()
# %%
master_Min = []


for i in range(len(dates_legs)):
    date = dates_legs[i]

    leg_dict = leg_data[i]

    flight_date = leg_dict['Date'] #Get date of flight from dictionary 
    BCB_start = leg_dict['LegIndex_02']['StartTimes']
    BCB_stop = leg_dict['LegIndex_02']['StopTimes']
    Min_Start = leg_dict['LegIndex_06']['StartTimes']
    Min_Stop = leg_dict['LegIndex_06']['StopTimes']

    sum_flight = summary[i]
    # if date == '2022-02-02':
    #     print(date)
    #     print(Min_Start)
    #     print(Min_Stop)

    times = sum_flight.Time_mid.values
    winds = sum_flight.Wind_Speed.values
    alts = sum_flight.GPS_altitude.values
    
    all_Min_means = []


    for i in range(len(Min_Start)):
        index1_start=None
        index1_end=None  
        start = int(Min_Start[i])
        end = Min_Stop[i]

        wind_alt = {
            'Date': date,
            'Min_start': start,
            'Min_end': end,
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

        all_Min_means.append(wind_alt) #List that contains all the Min wind/alt mean dictionaries for 1 flight
        
    master_Min.append(all_Min_means) #List that contains all Min flights  
#%%
Z0 = 0.03  # meters (typical value for open terrain)
Z10 = 10  # target height m


corrected_calc_min = {'Date': [], 'Corrected_min_windspeed': []}

for flight in master_Min:
    for wind_alt in flight:
        date = wind_alt['Date']
        windspeed = wind_alt['Winds_mean']
        altitude = wind_alt['Alts_mean']

        for wind_mean, alt_mean in zip(windspeed, altitude):
            # Apply the formula
            new_windspeed = wind_mean * (np.log(Z10/Z0) / np.log(alt_mean / Z0))

            # Append to the new dictionary
            corrected_calc_min['Date'].append(date)
            corrected_calc_min['Corrected_min_windspeed'].append(new_windspeed)
            
for date, wind_mean in zip(corrected_calc_min['Date'], corrected_calc_min['Corrected_min_windspeed']):
    print(f"Date: {date}, Corrected_min_windspeed: {wind_mean}")
# %%

# %%
master_Min_RH = []

for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict = leg_data[i]

    flight_date = leg_dict['Date']  # Get date of flight from dictionary 
    Min_Start = leg_dict['LegIndex_06']['StartTimes']
    Min_Stop = leg_dict['LegIndex_06']['StopTimes']

    rh_flight = h20[i]
    times_rh = rh_flight.Time_Start.values
    rh_values = rh_flight.RHw_DLH.values

    all_Min_means = []

    for j in range(len(Min_Start)):
        start = int(Min_Start[j])
        end = int(Min_Stop[j])

        rh_times = {
            'Date': date,
            'Min_start': start,
            'Min_end': end,
            'Rh_mean': [],
        }

        # Find the start index
        index1_start = None
        for k in range(len(times_rh)):
            if times_rh[k] == start:
                index1_start = k
                break

        # Find the end index
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
        all_Min_means.append(rh_times)  # List that contains all the Min wind/alt mean dictionaries for 1 flight

    master_Min_RH.append(all_Min_means)  # List that contains all Min flights
##this is only pulling from the second file for the date, not the first file as well
##thats why we are seeing NANs for the values 
# %%
import seaborn as sns
#%%

combined_data = {
    'Date': [],
    'Altitude': [],
    'RH': [],
    'Windspeed': []
}

# Validate and combine the data
for i, flight in enumerate(master_Min):
    for j, wind_alt in enumerate(flight):
        try:
            date = wind_alt['Date']
            alt_mean = wind_alt['Alts_mean'][0]
            wind_mean = wind_alt['Winds_mean'][0]
            rh_mean = master_Min_RH[i][j]['Rh_mean'][0]

            # Check if RH is valid
            if rh_mean < 0 or rh_mean > 100:
                print(f"Invalid RH value {rh_mean} at index {i}, skipping...")
                continue

            combined_data['Date'].append(date)
            combined_data['Altitude'].append(alt_mean)
            combined_data['RH'].append(rh_mean)
            combined_data['Windspeed'].append(wind_mean)

        except IndexError as e:
            print(f"Index error at i={i}, j={j}: {e}")
            continue

df_combined = pd.DataFrame(combined_data)

# Check the data for any anomalies
print(df_combined.describe())

# Step 2: Round Windspeed to nearest whole number
df_combined['Windspeed_Rounded'] = df_combined['Windspeed'].round()

# Step 3: Create the plot
plt.figure(figsize=(10, 8))
sc = plt.scatter(df_combined['RH'], df_combined['Altitude'], c=df_combined['Windspeed_Rounded'], cmap='viridis', s=50)
plt.colorbar(sc, label='Windspeed (Rounded)')
plt.xlabel('RH (%)')
plt.ylabel('Altitude (m)')
plt.title('Altitude vs RH with Windspeed Color Coding')
plt.grid(True)
plt.show()

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
for i, flight in enumerate(master_Min):
    for j, wind_alt in enumerate(flight):
        try:
            date = wind_alt['Date']
            alt_mean = wind_alt['Alts_mean'][0]
            rh_mean = master_Min_RH[i][j]['Rh_mean'][0]
            
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
plt.title('Minimum Altitude January-June 2022', fontsize=14, fontweight='bold')
# plt.legend()
plt.show()
# %%
# Initialize a list to store details of entries with altitudes above 200 meters
entries_above_200 = []

# Loop through each flight in master_Min
for flight in master_Min:
    for wind_alt in flight:
        # Extract the altitude mean values
        alt_means = wind_alt['Alts_mean']
        date = wind_alt['Date']
        start_time = wind_alt.get('Min_start', 'Unknown')  # Replace 'Unknown' if not present
        end_time = wind_alt.get('Min_end', 'Unknown')      # Replace 'Unknown' if not present
        
        # Check for altitudes above 200 meters and store details
        for alt in alt_means:
            if alt > 200:
                entries_above_200.append({
                    'Date': date,
                    'Start Time': start_time,
                    'End Time': end_time,
                    'Altitude': alt
                })

# Print the entries with altitudes above 200 meters
if entries_above_200:
    print("Entries with altitudes above 200 meters:")
    for entry in entries_above_200:
        print(f"Date: {entry['Date']}, Start Time: {entry['Start Time']}, End Time: {entry['End Time']}, Altitude: {entry['Altitude']}")
else:
    print("No entries with altitudes above 200 meters.")

# %%
