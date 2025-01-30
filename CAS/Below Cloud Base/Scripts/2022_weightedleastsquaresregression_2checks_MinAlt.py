#%%
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import datetime
import pathlib
import statistics
import shutil
import glob
import os
import re
import math
import matplotlib.patches as mpatches
import statsmodels.api as sm
# %%
# %%
# C_0=math.log(0.61)-math.log(0.5)
# C_01=math.log(0.68)-math.log(0.61)
# C_02=math.log(0.75)-math.log(0.68)
# C_03=math.log(0.82)-math.log(0.75)
# C_04=math.log(0.89)-math.log(0.82)
# C_05=math.log(0.96)-math.log(0.89)
# C_06=math.log(1.03)-math.log(0.96)
# C_07=math.log(1.1)-math.log(1.03)
# C_08=math.log(1.17)-math.log(1.1)
# C_09=math.log(1.25)-math.log(1.17)
# C_10=math.log(1.5)-math.log(1.25)
# C_11=math.log(2)-math.log(1.5)
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
#[C_0, C_01, C_02, C_03, C_04, C_05, C_06, C_07, C_08, C_09, C_10, C_11, C_12,
# bin_center=[0.555, 0.645, 0.715, 0.785, 0.855, 0.925, 
#             0.995, 1.07, 1.14, 1.21, 1.38, 1.75, 2.25, 
#             2.75, 3.25, 3.75, 4.5, 5.75, 6.85, 7.55, 
#             9.05, 11.4, 13.8, 17.5, 22.5, 27.5, 32.5, 
#             37.5, 42.5, 47.5]
# bin_center = np.array(bin_center)

# D0 = (0.61-0.5)
# D01 = (0.68-0.61)
# D02 = (0.75-0.68)
# D03 = (0.82-0.75)
# D04 = (0.89-0.82)
# D05 = (0.96-0.86)
# D06 = (1.03-0.96)
# D07 = (1.1-1.03)
# D08 = (1.17-1.1)
# D09 = (1.25-1.17)
# D10 = (1.5-1.25)
# D11 = (2-1.5)
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

# F0 = (C_0 / D0)
# F01 = (C_01 / D01)
# F02 = (C_02 / D02)
# F03 = (C_03 / D03)
# F04 = (C_04 / D04)
# F05 = (C_05 / D05)
# F06 = (C_06/ D06)
# F07 = (C_07 / D07)
# F08 = (C_08 / D08)
# F09 = (C_09 / D09)
# F10 = (C_10 / D10)
# F11 = (C_11 / D11)
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
#[F0, F01, F02, F03, F04, F05, F06, F07, F08, F09, F10, F11, 
Logg = np.array(Logg)

# %%
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

        if date >= '2022-01-11':
                print(f"\nFirst row for date {'2022-01-11'}:")
        
        print(date)
        time = df_2DS['Time_Start']
        print("yes")
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
#%%
#CAS data

bin_name = ['CAS_Bin12', 'CAS_Bin13', 'CAS_Bin14', 'CAS_Bin15', 'CAS_Bin16', 'CAS_Bin17', 
            'CAS_Bin18', 'CAS_Bin19', 'CAS_Bin20', 'CAS_Bin21', 'CAS_Bin22', 'CAS_Bin23', 'CAS_Bin24', 'CAS_Bin25', 'CAS_Bin26',
            'CAS_Bin27', 'CAS_Bin28', 'CAS_Bin29']

CAS = []

dates_CAS = ['2022-01-11', '2022-01-12','2022-01-15', '2022-01-18', 
              '2022-01-19', '2022-01-24', '2022-01-26', '2022-01-27',
              '2022-02-01', '2022-02-02', '2022-02-03', '2022-02-05', 
             '2022-02-15', '2022-02-16', '2022-02-19', '2022-02-22',
              '2022-02-26', '2022-03-02', '2022-03-03', '2022-03-04', #'2022-03-07',
              '2022-03-13', '2022-03-14', '2022-03-18', '2022-03-22',
             '2022-03-26', '2022-03-28', '2022-03-29', #'2022-05-03',
             '2022-05-05', '2022-05-10','2022-05-16', '2022-05-17',
             '2022-05-18',
             '2022-05-20','2022-05-21', '2022-05-31', '2022-06-02', 
              '2022-06-03', '2022-06-05','2022-06-07', '2022-06-08', 
               '2022-06-10','2022-06-11','2022-06-13', '2022-06-14',
             '2022-06-17', '2022-06-18']

for date in dates_CAS:
    datestr = date.replace('-', '')
    fname_CAS = sorted(glob.glob(f'/home/disk/eos4/kathem24/activate/data/cloudaerospect/2022csv/ACTIVATE-LARGE-CAS_HU25_{datestr}_R*.csv'), reverse=True)
    # print(date)
    # print(fname_CAS)

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
            if df_CAS[col].dtype == 'O':  # 'O' stands for Object (usually string columns)
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
    Min_Start = leg_dict['LegIndex_06']['StartTimes']
    Min_Stop = leg_dict['LegIndex_06']['StopTimes']

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

    for k in range(len(Min_Start)):
        start20 = int(Min_Start[k])
        end20 = Min_Stop[k]

        data_labels = []
        Min_means = []  # List to store the means for this BCB interval

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
                total_Min_means.append(Min_means)  # Add to the total BCB means list

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
#%%

master_CAS_Min = []
leg_info = []

for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict = leg_data[i]

    flight_date = leg_dict['Date']
    Min_Start = leg_dict['LegIndex_06']['StartTimes']
    Min_Stop = leg_dict['LegIndex_06']['StopTimes']

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

    for k in range(len(Min_Start)):
        start20 = int(Min_Start[k])
        end20 = Min_Stop[k]

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

# Print or use master_CAS_BCB as needed
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
#%%
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
#%%
def summary_bin_means(data):
    summary_list = []

    for flight_dict in data:
        bin_means = []
        for bin_label in range(12, 30):
            bin_key_Y = f'Bin{bin_label}_Y_mean'
            bin_mean = flight_dict.get(bin_key_Y, np.nan)
            if not np.isnan(bin_mean):
                bin_means.append(bin_mean)
        
        total_summary = sum(bin_means)
        summary_dict = {
            'Date': flight_dict.get('Date', ''),
            'Min_start': flight_dict.get('Min_start', ''),
            'Min_stop': flight_dict.get('Min_stop', ''),
            'Sum': total_summary
        }
        summary_list.append(summary_dict)

    return summary_list

# Apply the function to Y_BCB_calc and N_BCB_calc
Y_sum_list = summary_bin_means(Y_Min_calc)
#%%
def sum_bin_means(data):
    sum_list = []

    for flight_dict in data:
        bin_means = []
        for bin_label in range(12, 30):
            bin_key_N = f'Bin{bin_label}_N_mean'
            bin_mean = flight_dict.get(bin_key_N, np.nan)
            if not np.isnan(bin_mean):
                bin_means.append(bin_mean)
        
        total_sum = sum(bin_means)
        sum_dict = {
            'Date': flight_dict.get('Date', ''),
            'Min_start': flight_dict.get('Min_start', ''),
            'Min_stop': flight_dict.get('Min_stop', ''),
            'Sum': total_sum
        }
        sum_list.append(sum_dict)

    return sum_list

# Apply the function to N_BCB_calc
N_sum_list = sum_bin_means(N_Min_calc)


#%%
sum_values = [entry['Sum'] for entry in Y_sum_list]
wind_speeds_min = corrected_calc_min['Corrected_min_windspeed']

valid_indices = [i for i in range(len(wind_speeds_min)) if not np.isnan(wind_speeds_min[i]) and not np.isnan(Y_sum_list[i]['Sum'])]
wind_speeds_min_filtered = [wind_speeds_min[i] for i in valid_indices]
sum_values_filtered = [Y_sum_list[i]['Sum'] for i in valid_indices]

# Create the scatterplot
plt.scatter(wind_speeds_min_filtered, sum_values_filtered, color='green')
plt.title('Minimum Altitude January - June 2022', fontsize=12, fontweight='bold')
plt.xlabel('Mean windspeed (m/s)', fontsize=12, fontweight='bold')
plt.ylabel('Total droplet concentration dN (per cm^3) \nfor clear skies', fontsize=12, fontweight='bold')
# plt.yscale('log')
# plt.ylim(10**-2.2, 10**2)
plt.xlim(0, 17)
plt.ylim(-1, 25)
plt.show()

#%%
#Binning by mean windspeed 0.5 m/s


from collections import defaultdict
sum_values = [entry['Sum'] for entry in Y_sum_list]
wind_speeds_min = corrected_calc_min['Corrected_min_windspeed']

# Filter out NaN values
valid_indices = [i for i in range(len(wind_speeds_min)) if not np.isnan(wind_speeds_min[i]) and not np.isnan(sum_values[i])]
wind_speeds_min_filtered = [wind_speeds_min[i] for i in valid_indices]
sum_values_filtered = [sum_values[i] for i in valid_indices]

# Create the initial scatterplot
plt.figure(figsize=(10, 6))
plt.scatter(wind_speeds_min_filtered, sum_values_filtered, color='green')
plt.title('Minimum Altitude January - June 2022', fontsize=12, fontweight='bold')
plt.xlabel('Mean windspeed (m/s)', fontsize=12, fontweight='bold')
plt.ylabel('Total droplet concentration dN (per cm^3) \nfor clear skies', fontsize=12, fontweight='bold')
# plt.yscale('log')
# plt.ylim(10**-2, 10**2)
# plt.xlim(0, 17)
plt.ylim(-1, 3)
plt.show()

# Step 2: Binning process to calculate mean droplet concentration
bin_width = 0.5
bins = np.arange(0, max(wind_speeds_min_filtered) + bin_width, bin_width)
binned_wind_speeds = []
binned_mean_concentrations = []

print("Bin range (m/s): Number of droplet concentration values")

for i in range(len(bins) - 1):
    bin_min = bins[i]
    bin_max = bins[i + 1]
    bin_indices = [j for j in range(len(wind_speeds_min_filtered)) if bin_min <= wind_speeds_min_filtered[j] < bin_max]
    bin_concentrations = [sum_values_filtered[j] for j in bin_indices]
    if bin_concentrations:
        mean_concentration = np.mean(bin_concentrations)
        binned_wind_speeds.append((bin_min + bin_max) / 2)  # Use the midpoint of the bin for plotting
        binned_mean_concentrations.append(mean_concentration)
        print(f"{bin_min}-{bin_max}: {len(bin_concentrations)}")

# Create the new scatterplot of binned means
plt.figure(figsize=(10, 6))
plt.scatter(binned_wind_speeds, binned_mean_concentrations, color='green', label='mean droplet conc.')
plt.title('Minimum Altitude January - June 2022 \n(Binned at 0.5 m/s)', fontsize=14, fontweight='bold')
plt.xlabel('Mean windpseed (m/s)', fontsize=12, fontweight='bold')
plt.ylabel('Mean droplet concentration dN (per cm^3) \nfor clear skies', fontsize=12, fontweight='bold')
# plt.yscale('log')
# plt.ylim(10**-2, 10**2)
plt.xlim(0, 17)
plt.ylim(-1, 3)
plt.legend()
plt.show()


#%%
 #Weighted LSR based on 0.5 m/s bins 

# Step 1: Extract data for the initial scatterplot
sum_values = [entry['Sum'] for entry in Y_sum_list]
wind_speeds_bcb = corrected_calc_min['Corrected_min_windspeed']

# Filter out NaN values
valid_indices = [i for i in range(len(wind_speeds_min)) if not np.isnan(wind_speeds_min[i]) and not np.isnan(sum_values[i])]
wind_speeds_min_filtered = [wind_speeds_min[i] for i in valid_indices]
sum_values_filtered = [sum_values[i] for i in valid_indices]

# Create the initial scatterplot
plt.figure(figsize=(10, 6))
plt.scatter(wind_speeds_min_filtered, sum_values_filtered, color='green')
plt.title('Minimum Altitude January - June 2022', fontsize=12, fontweight='bold')
plt.xlabel('Mean windspeed (m/s)', fontsize=12, fontweight='bold')
plt.ylabel('Total droplet concentration dN (per cm^3) \nfor clear skies', fontsize=12, fontweight='bold')
plt.xlim(0, 17)
# plt.ylim(-1, 10)
plt.ylim(10**-1.5, 10**2)
plt.yscale('log')
plt.show()

# Step 2: Binning process to calculate mean droplet concentration
bin_width = 0.5
bins = np.arange(0, max(wind_speeds_min_filtered) + bin_width, bin_width)
binned_wind_speeds = []
binned_mean_concentrations = []
weights = []

print("Bin range (m/s): Number of droplet concentration values")

for i in range(len(bins) - 1):
    bin_min = bins[i]
    bin_max = bins[i + 1]
    bin_indices = [j for j in range(len(wind_speeds_min_filtered)) if bin_min <= wind_speeds_min_filtered[j] < bin_max]
    bin_concentrations = [sum_values_filtered[j] for j in bin_indices]
    if bin_concentrations:
        mean_concentration = np.mean(bin_concentrations)
        binned_wind_speeds.append((bin_min + bin_max) / 2)  # Use the midpoint of the bin for plotting
        binned_mean_concentrations.append(mean_concentration)
        weights.append(len(bin_concentrations))  # The weight is the number of values in the bin
        # Print the bin range and the number of values used to calculate the mean for that bin
        print(f"{bin_min}-{bin_max}: {len(bin_concentrations)}")

# Create the new scatterplot of binned means
plt.figure(figsize=(10, 6))
plt.scatter(binned_wind_speeds, binned_mean_concentrations, color='green', label='mean droplet conc.')
plt.title('Minimum Altitude January - June 2022 \n(Binned 0.5 m/s)', fontsize=12, fontweight='bold')
plt.xlabel('Mean windspeed (m/s)', fontsize=12, fontweight='bold')
plt.ylabel('Mean droplet concentration dN (per cm^3) \nfor clear skies', fontsize=12, fontweight='bold')
plt.legend()
plt.ylim(10**-2, 10**1)
plt.yscale('log')
# plt.ylim(-0.25, 2)
plt.xlim(0,16)
plt.show()

# Step 3: Perform weighted least squares regression
# Convert lists to numpy arrays
binned_wind_speeds_np = np.array(binned_wind_speeds)
binned_mean_concentrations_np = np.array(binned_mean_concentrations)
weights_np = np.array(weights)

# Add a constant to the predictor variable (for the intercept term)
X = sm.add_constant(binned_wind_speeds_np)

# Fit the weighted least squares model
model = sm.WLS(binned_mean_concentrations_np, X, weights=weights_np)
results = model.fit()

# Print the summary of the regression results
print(results.summary())

# Step 4: Plot the regression line
plt.figure(figsize=(10, 6))
plt.scatter(binned_wind_speeds, binned_mean_concentrations, color='green', label='mean droplet conc.')
plt.plot(binned_wind_speeds, results.predict(X), color='purple', label='WLS Regression Line')
plt.title('Minimum Altitude January - June 2022 \n(Binned 0.5 m/s)', fontsize=12, fontweight='bold')
plt.xlabel('Mean windspeed (m/s)', fontsize=12, fontweight='bold')
plt.ylabel('Mean droplet concentration dN (per cm^3) \nfor clear skies', fontsize=12, fontweight='bold')
plt.legend()
plt.ylim(10**-2, 10**1)
plt.yscale('log')
plt.xlim(0,17)
# plt.ylim(-0.25, 2)
plt.show()


# %%

sum_values = [entry['Sum'] for entry in Y_sum_list]
wind_speeds_bcb = corrected_calc_min['Corrected_min_windspeed']

# Filter out NaN values
valid_indices = [i for i in range(len(wind_speeds_min)) if not np.isnan(wind_speeds_min[i]) and not np.isnan(sum_values[i])]
wind_speeds_min_filtered = [wind_speeds_min[i] for i in valid_indices]
sum_values_filtered = [sum_values[i] for i in valid_indices]

# Create the initial scatterplot
plt.figure(figsize=(10, 6))
plt.scatter(wind_speeds_min_filtered, sum_values_filtered, color='green')
plt.title('Minimum Altitude January - June 2022', fontsize=12, fontweight='bold')
plt.xlabel('Mean windspeed (m/s)', fontsize=12, fontweight='bold')
plt.ylabel('Total droplet concentration dN (per cm^3) \nfor clear skies', fontsize=12, fontweight='bold')
# # plt.xlim(0, 17)
# plt.ylim(-1, 3)
plt.ylim(10**-1.5, 10**2)
plt.yscale('log')
plt.xlim(0,16)
plt.show()

# Step 2: Binning process to calculate mean droplet concentration and variance
bin_width = 0.5
bins = np.arange(0, max(wind_speeds_min_filtered) + bin_width, bin_width)
binned_wind_speeds = []
binned_mean_concentrations = []
weights = []

print("Bin range (m/s): Number of droplet concentration values")

for i in range(len(bins) - 1):
    bin_min = bins[i]
    bin_max = bins[i + 1]
    bin_indices = [j for j in range(len(wind_speeds_min_filtered)) if bin_min <= wind_speeds_min_filtered[j] < bin_max]
    bin_concentrations = [sum_values_filtered[j] for j in bin_indices]
    if bin_concentrations:
        mean_concentration = np.mean(bin_concentrations)
        variance = np.var(bin_concentrations) if len(bin_concentrations) > 1 else 1  # Avoid zero variance for single data points
        weight = 1 / variance if variance > 0 else 1  # Avoid division by zero
        
        binned_wind_speeds.append((bin_min + bin_max) / 2)  # Use the midpoint of the bin for plotting
        binned_mean_concentrations.append(mean_concentration)
        weights.append(weight)  # The weight is the inverse of the variance
        
        # Print the bin range, the number of values used to calculate the mean, and the variance
        print(f"{bin_min}-{bin_max}: {len(bin_concentrations)} values, variance={variance:.4f}")

# Create the new scatterplot of binned means
plt.figure(figsize=(10, 6))
plt.scatter(binned_wind_speeds, binned_mean_concentrations, color='green', label='mean droplet conc.')
plt.title('Minimum Altitude January - June 2022 \n(Binned 0.5 m/s)', fontsize=12, fontweight='bold')
plt.xlabel('Mean windspeed (m/s)', fontsize=12, fontweight='bold')
plt.ylabel('Mean droplet concentration dN (per cm^3) \nfor clear skies', fontsize=12, fontweight='bold')
plt.legend()
plt.ylim(10**-2, 10**2)
plt.yscale('log')
plt.xlim(0,16)
# plt.ylim(-0.25, 2)
plt.show()

# Step 3: Perform weighted least squares regression based on variance
# Convert lists to numpy arrays
binned_wind_speeds_np = np.array(binned_wind_speeds)
binned_mean_concentrations_np = np.array(binned_mean_concentrations)
weights_np = np.array(weights)

# Add a constant to the predictor variable (for the intercept term)
X = sm.add_constant(binned_wind_speeds_np)

# Fit the weighted least squares model
model = sm.WLS(binned_mean_concentrations_np, X, weights=weights_np)
results = model.fit()

# Print the summary of the regression results
print(results.summary())

# Step 4: Plot the regression line
plt.figure(figsize=(10, 6))
plt.scatter(binned_wind_speeds, binned_mean_concentrations, color='green', label='mean droplet conc.')
plt.plot(binned_wind_speeds, results.predict(X), color='purple', label='WLS Regression Line')
plt.title('Minimum Altitude January - June 2022 \n(Binned 0.5 m/s)', fontsize=12, fontweight='bold')
plt.xlabel('Mean windspeed (m/s)', fontsize=12, fontweight='bold')
plt.ylabel('Mean droplet concentration dN (per cm^3) \nfor clear skies', fontsize=12, fontweight='bold')
plt.legend()
plt.ylim(10**-8, 10**1)
plt.yscale('log')
plt.xlim(0,16)
# plt.ylim(-0.25, 2)
plt.show()
# %%
# WLS but to fit the log

#transform the dependent variable by taking the log of mean droplet concentration
#before I fit the model 

# Assuming binned_wind_speeds and binned_mean_concentrations are already defined

# Convert lists to numpy arrays for regression
x = np.array(binned_wind_speeds)
y = np.array(binned_mean_concentrations)

# Add a small offset to y to avoid taking the log of zero or negative values
epsilon = 1e-8  # Small positive offset
y = np.maximum(y, epsilon)

# Take natural logarithm of y
y_log = np.log(y)

# Define weights (example: inverse of x values)
weights = 1 / x

# Perform WLS regression on log-transformed data
coefficients = np.polyfit(x, y_log, deg=1, w=weights)

# Extract slope and intercept from coefficients
slope_log, intercept_log = coefficients

# Calculate fitted values in log space
y_log_fit = slope_log * x + intercept_log

# Convert fitted values back to original scale (exponential)
y_fit = np.exp(y_log_fit)

# Plot the data and fitted exponential curve
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Data')
plt.plot(x, y_fit, color='red', linestyle='--', label='Exponential Fit (Log WLS)')
plt.title('Exponential Fit (Log WLS): Mean Droplet Concentration vs. Mean Wind Speed', fontsize=14, fontweight='bold')
plt.xlabel('Mean wind speed (m/s)', fontsize=12, fontweight='bold')
plt.ylabel('Mean droplet concentration dN (per cm^3) for clear skies', fontsize=12, fontweight='bold')
# plt.ylim(0, 2)  # Adjust y-axis limits as needed
plt.yscale('log')  # Set y-axis to logarithmic scale for data points
plt.legend()

plt.show()

# Print regression results
print(f"Slope (log scale): {slope_log:.2f}")
print(f"Intercept (log scale): {intercept_log:.2f}")

#%%

# %%
#WLS based off the spread (variance) in droplet concentration points
#you can use the inverse of the variance as weights. This approach assigns more 
# weight to bins where the droplet concentration values are tightly 
# clustered (lower variance) and less weight to bins where the values 
# are more spread out (higher variance)

#How did I do this? 
#Calculate Variance: For each bin, calculate the variance of droplet 
# concentrations. Variance is used to measure the spread of 
# data points within each bin.

#Handle Zero Variance: Ensure that there are no bins with zero variance 
# to avoid division by zero errors by replacing zero variance with the 
# smallest positive float number.

#Calculate Weights: Use the inverse of the variance as weights. 
# This means bins with lower variance (tightly clustered points) will 
# have higher weights.
#%%

# Assuming corrected_calc_bcb and Y_sum_list are already defined

# Step 1: Extract data for the initial scatterplot
sum_values = [entry['Sum'] for entry in Y_sum_list]
wind_speeds_min = corrected_calc_min['Corrected_min_windspeed']

# Filter out NaN values
valid_indices = [i for i in range(len(wind_speeds_min)) if not np.isnan(wind_speeds_min[i]) and not np.isnan(sum_values[i])]
wind_speeds_min_filtered = [wind_speeds_min[i] for i in valid_indices]
sum_values_filtered = [sum_values[i] for i in valid_indices]

# Create the initial scatterplot
plt.figure(figsize=(10, 6))
plt.scatter(wind_speeds_min_filtered, sum_values_filtered)
plt.title('Minimum Altitude January - June 2022', fontsize=12, fontweight='bold')
plt.xlabel('Mean windspeed (m/s)', fontsize=12, fontweight='bold')
plt.ylabel('Total droplet concentration dN (per cm^3) \nfor clear skies', fontsize=12, fontweight='bold')
# # plt.xlim(0, 17)
# # plt.ylim(-1, 10)
# plt.ylim(10**-1.5, 10**2)
# plt.yscale('log')
plt.xlim(0,16)
plt.show()

# Step 2: Binning process to calculate mean droplet concentration and variance
bin_width = 0.5
bins = np.arange(0, max(wind_speeds_min_filtered) + bin_width, bin_width)
binned_wind_speeds = []
binned_mean_concentrations = []
weights = []

print("Bin range (m/s): Number of droplet concentration values")

for i in range(len(bins) - 1):
    bin_min = bins[i]
    bin_max = bins[i + 1]
    bin_indices = [j for j in range(len(wind_speeds_min_filtered)) if bin_min <= wind_speeds_min_filtered[j] < bin_max]
    bin_concentrations = [sum_values_filtered[j] for j in bin_indices]
    if bin_concentrations:
        mean_concentration = np.mean(bin_concentrations)
        variance = np.var(bin_concentrations) if len(bin_concentrations) > 1 else 1  # Avoid zero variance for single data points
        weight = 1 / variance if variance > 0 else 1  # Avoid division by zero
        
        binned_wind_speeds.append((bin_min + bin_max) / 2)  # Use the midpoint of the bin for plotting
        binned_mean_concentrations.append(mean_concentration)
        weights.append(weight)  # The weight is the inverse of the variance
        
        # Print the bin range, the number of values used to calculate the mean, and the variance
        print(f"{bin_min}-{bin_max}: {len(bin_concentrations)} values, variance={variance:.4f}")

# Create the new scatterplot of binned means
plt.figure(figsize=(10, 6))
plt.scatter(binned_wind_speeds, binned_mean_concentrations, color='red', label='mean droplet conc.')
plt.title('Minimum Altitude January - June 2022 \n(Binned 0.5 m/s)', fontsize=12, fontweight='bold')
plt.xlabel('Corrected Min Alt Wind Speed (m/s)', fontsize=12, fontweight='bold')
plt.ylabel('Mean droplet concentration dN (per cm^3) \nfor clear skies', fontsize=12, fontweight='bold')
plt.legend()
# plt.ylim(10**-1.5, 10**2)
# plt.yscale('log')
# plt.xlim(0,16)
plt.show()

# Step 3: Perform weighted least squares regression based on variance
# Convert lists to numpy arrays
binned_wind_speeds_np = np.array(binned_wind_speeds)
binned_mean_concentrations_np = np.array(binned_mean_concentrations)
weights_np = np.array(weights)

# Add a constant to the predictor variable (for the intercept term)
X = sm.add_constant(binned_wind_speeds_np)

# Fit the weighted least squares model
model = sm.WLS(binned_mean_concentrations_np, X, weights=weights_np)
results = model.fit()

# Print the summary of the regression results
print(results.summary())

# Step 4: Plot the regression line
plt.figure(figsize=(10, 6))
plt.scatter(binned_wind_speeds, binned_mean_concentrations, color='blue', label='mean droplet conc.')
plt.plot(binned_wind_speeds, results.predict(X), color='red', label='WLS Regression Line')
plt.title('Minimum Altitude January - June 2022 \n(Binned 0.5 m/s)', fontsize=12, fontweight='bold')
plt.xlabel('Mean windspeed (m/s)', fontsize=12, fontweight='bold')
plt.ylabel('Mean droplet concentration dN (per cm^3) \nfor clear skies', fontsize=12, fontweight='bold')
plt.legend()
# plt.ylim(10**-1.5, 10**2)
# plt.yscale('log')
# plt.xlim(0,16)
plt.show()

# %%

#Percentile binning 

#10 bins

import numpy as np
import matplotlib.pyplot as plt

# Assuming corrected_calc_bcb and Y_sum_list are already defined

# Step 1: Extract data for the initial scatterplot
sum_values = [entry['Sum'] for entry in Y_sum_list]
wind_speeds_min = corrected_calc_min['Corrected_min_windspeed']

# Filter out NaN values
valid_indices = [i for i in range(len(wind_speeds_min)) if not np.isnan(wind_speeds_min[i]) and not np.isnan(sum_values[i])]
wind_speeds_min_filtered = [wind_speeds_min[i] for i in valid_indices]
sum_values_filtered = [sum_values[i] for i in valid_indices]

# Sort data by wind speed
sorted_indices = np.argsort(wind_speeds_min_filtered)
wind_speeds_min_sorted = np.array(wind_speeds_min_filtered)[sorted_indices]
sum_values_sorted = np.array(sum_values_filtered)[sorted_indices]

# Calculate deciles
num_points = len(wind_speeds_min_sorted)
percentiles = np.linspace(0, 100, 11)  # Generate 11 percentiles (0, 10, 20, ..., 100)
decile_boundaries = np.percentile(wind_speeds_min_sorted, percentiles)

# Initialize lists for binned data
binned_wind_speeds = []
binned_mean_concentrations = []

print("Bin range (m/s): Number of droplet concentration values")

# Iterate through decile bins
for i in range(len(percentiles) - 1):
    bin_min = decile_boundaries[i]
    bin_max = decile_boundaries[i + 1]
    bin_indices = np.where((wind_speeds_min_sorted >= bin_min) & (wind_speeds_min_sorted < bin_max))[0]
    bin_concentrations = sum_values_sorted[bin_indices]
    
    if len(bin_concentrations) > 0:
        mean_concentration = np.mean(bin_concentrations)
        binned_wind_speeds.append((bin_min + bin_max) / 2)
        binned_mean_concentrations.append(mean_concentration)
        print(f"{bin_min:.2f}-{bin_max:.2f}: {len(bin_concentrations)}")

# Create the scatterplot of binned means
plt.figure(figsize=(10, 6))
plt.scatter(binned_wind_speeds, binned_mean_concentrations, color='blue', label='mean droplet conc.')
plt.title('Minimum Altitude January - June 2022 \n(Decile Binning)', fontsize=14, fontweight='bold')
plt.xlabel('Mean wind speed (m/s)', fontsize=12, fontweight='bold')
plt.ylabel('Mean droplet concentration dN (per cm^3) \nfor clear skies', fontsize=12, fontweight='bold')
# plt.yscale('log')
# plt.ylim(10**-2, 10**2)
plt.legend()
plt.show()
#%%

#Percentile binning 

#9 bins


# Assuming corrected_calc_bcb and Y_sum_list are already defined

# Step 1: Extract data for the initial scatterplot
sum_values = [entry['Sum'] for entry in Y_sum_list]
wind_speeds_min = corrected_calc_min['Corrected_min_windspeed']

# Filter out NaN values
valid_indices = [i for i in range(len(wind_speeds_min)) if not np.isnan(wind_speeds_min[i]) and not np.isnan(sum_values[i])]
wind_speeds_min_filtered = [wind_speeds_min[i] for i in valid_indices]
sum_values_filtered = [sum_values[i] for i in valid_indices]

# Sort data by wind speed
sorted_indices = np.argsort(wind_speeds_min_filtered)
wind_speeds_min_sorted = np.array(wind_speeds_min_filtered)[sorted_indices]
sum_values_sorted = np.array(sum_values_filtered)[sorted_indices]

# Calculate deciles
num_points = len(wind_speeds_min_sorted)
percentiles = np.linspace(0, 100, 10)  # Generate 11 percentiles (0, 10, 20, ..., 100)
decile_boundaries = np.percentile(wind_speeds_min_sorted, percentiles)

# Initialize lists for binned data
binned_wind_speeds = []
binned_mean_concentrations = []

print("Bin range (m/s): Number of droplet concentration values")

# Iterate through decile bins
for i in range(len(percentiles) - 1):
    bin_min = decile_boundaries[i]
    bin_max = decile_boundaries[i + 1]
    bin_indices = np.where((wind_speeds_min_sorted >= bin_min) & (wind_speeds_min_sorted < bin_max))[0]
    bin_concentrations = sum_values_sorted[bin_indices]
    
    if len(bin_concentrations) > 0:
        mean_concentration = np.mean(bin_concentrations)
        binned_wind_speeds.append((bin_min + bin_max) / 2)
        binned_mean_concentrations.append(mean_concentration)
        print(f"{bin_min:.2f}-{bin_max:.2f}: {len(bin_concentrations)}")

# Create the scatterplot of binned means
plt.figure(figsize=(10, 6))
plt.scatter(binned_wind_speeds, binned_mean_concentrations, color='blue', label='mean droplet conc.')
plt.title('Minimum Altitude January - June 2022 \n(Decile Binning)', fontsize=14, fontweight='bold')
plt.xlabel('Mean wind speed (m/s)', fontsize=12, fontweight='bold')
plt.ylabel('Mean droplet concentration dN (per cm^3) \nfor clear skies', fontsize=12, fontweight='bold')
plt.yscale('log')
plt.ylim(10**-2, 10**2)
plt.legend()
plt.show()
#%%
#Percentile binning 

#15 bins


# Assuming corrected_calc_bcb and Y_sum_list are already defined

# Step 1: Extract data for the initial scatterplot
sum_values = [entry['Sum'] for entry in Y_sum_list]
wind_speeds_min = corrected_calc_min['Corrected_min_windspeed']

# Filter out NaN values
valid_indices = [i for i in range(len(wind_speeds_min)) if not np.isnan(wind_speeds_min[i]) and not np.isnan(sum_values[i])]
wind_speeds_min_filtered = [wind_speeds_min[i] for i in valid_indices]
sum_values_filtered = [sum_values[i] for i in valid_indices]

# Sort data by wind speed
sorted_indices = np.argsort(wind_speeds_min_filtered)
wind_speeds_min_sorted = np.array(wind_speeds_min_filtered)[sorted_indices]
sum_values_sorted = np.array(sum_values_filtered)[sorted_indices]

# Calculate deciles
num_points = len(wind_speeds_min_sorted)
percentiles = np.linspace(0, 100, 16)  # Generate 11 percentiles (0, 10, 20, ..., 100)
decile_boundaries = np.percentile(wind_speeds_min_sorted, percentiles)

# Initialize lists for binned data
binned_wind_speeds = []
binned_mean_concentrations = []

print("Bin range (m/s): Number of droplet concentration values")

# Iterate through decile bins
for i in range(len(percentiles) - 1):
    bin_min = decile_boundaries[i]
    bin_max = decile_boundaries[i + 1]
    bin_indices = np.where((wind_speeds_min_sorted >= bin_min) & (wind_speeds_min_sorted < bin_max))[0]
    bin_concentrations = sum_values_sorted[bin_indices]
    
    if len(bin_concentrations) > 0:
        mean_concentration = np.mean(bin_concentrations)
        binned_wind_speeds.append((bin_min + bin_max) / 2)
        binned_mean_concentrations.append(mean_concentration)
        print(f"{bin_min:.2f}-{bin_max:.2f}: {len(bin_concentrations)}")

# Create the scatterplot of binned means
plt.figure(figsize=(10, 6))
plt.scatter(binned_wind_speeds, binned_mean_concentrations, color='blue', label='mean droplet conc.')
plt.title('Minimum Altitude January - June 2022 \n(Decile Binning)', fontsize=14, fontweight='bold')
plt.xlabel('Mean wind speed (m/s)', fontsize=12, fontweight='bold')
plt.ylabel('Mean droplet concentration dN (per cm^3) \nfor clear skies', fontsize=12, fontweight='bold')
plt.yscale('log')
plt.ylim(10**-2, 10**2)
plt.legend()
plt.show()
#%%
#Percentile binning 

#20 bins


# Assuming corrected_calc_bcb and Y_sum_list are already defined

# Step 1: Extract data for the initial scatterplot
sum_values = [entry['Sum'] for entry in Y_sum_list]
wind_speeds_min = corrected_calc_min['Corrected_min_windspeed']

# Filter out NaN values
valid_indices = [i for i in range(len(wind_speeds_min)) if not np.isnan(wind_speeds_min[i]) and not np.isnan(sum_values[i])]
wind_speeds_min_filtered = [wind_speeds_min[i] for i in valid_indices]
sum_values_filtered = [sum_values[i] for i in valid_indices]

# Sort data by wind speed
sorted_indices = np.argsort(wind_speeds_min_filtered)
wind_speeds_min_sorted = np.array(wind_speeds_min_filtered)[sorted_indices]
sum_values_sorted = np.array(sum_values_filtered)[sorted_indices]

# Calculate deciles
num_points = len(wind_speeds_min_sorted)
percentiles = np.linspace(0, 100, 16)  # Generate 11 percentiles (0, 10, 20, ..., 100)
decile_boundaries = np.percentile(wind_speeds_min_sorted, percentiles)

# Initialize lists for binned data
binned_wind_speeds = []
binned_mean_concentrations = []

print("Bin range (m/s): Number of droplet concentration values")

# Iterate through decile bins
for i in range(len(percentiles) - 1):
    bin_min = decile_boundaries[i]
    bin_max = decile_boundaries[i + 1]
    bin_indices = np.where((wind_speeds_min_sorted >= bin_min) & (wind_speeds_min_sorted < bin_max))[0]
    bin_concentrations = sum_values_sorted[bin_indices]
    
    if len(bin_concentrations) > 0:
        mean_concentration = np.mean(bin_concentrations)
        binned_wind_speeds.append((bin_min + bin_max) / 2)
        binned_mean_concentrations.append(mean_concentration)
        print(f"{bin_min:.2f}-{bin_max:.2f}: {len(bin_concentrations)}")

# Create the scatterplot of binned means
plt.figure(figsize=(10, 6))
plt.scatter(binned_wind_speeds, binned_mean_concentrations, color='blue', label='mean droplet conc.')
plt.title('Minimum Altitude January - June 2022 \n(Decile Binning)', fontsize=14, fontweight='bold')
plt.xlabel('Mean wind speed (m/s)', fontsize=12, fontweight='bold')
plt.ylabel('Mean droplet concentration dN (per cm^3) \nfor clear skies', fontsize=12, fontweight='bold')
plt.yscale('log')
plt.ylim(10**-2, 10**2)
plt.legend()
plt.show()
#%%
from statsmodels.regression.linear_model import WLS
# %%


#WLS Regression for 10 bins

##WLS regression with 10 bins
sum_values = [entry['Sum'] for entry in Y_sum_list]
wind_speeds_bcb = corrected_calc_min['Corrected_min_windspeed']

# Filter out NaN values
valid_indices = [i for i in range(len(wind_speeds_min)) if not np.isnan(wind_speeds_min[i]) and not np.isnan(sum_values[i])]
wind_speeds_min_filtered = [wind_speeds_min[i] for i in valid_indices]
sum_values_filtered = [sum_values[i] for i in valid_indices]

# Sort data by wind speed
sorted_indices = np.argsort(wind_speeds_min_filtered)
wind_speeds_min_sorted = np.array(wind_speeds_min_filtered)[sorted_indices]
sum_values_sorted = np.array(sum_values_filtered)[sorted_indices]

# Calculate percentiles for binning
num_points = len(wind_speeds_min_sorted)
percentiles = np.linspace(0, 100, 11)  # Generate 21 percentiles (0, 5, 10, ..., 100)
decile_boundaries = np.percentile(wind_speeds_min_sorted, percentiles)

# Initialize lists for binned data
binned_wind_speeds = []
binned_mean_concentrations = []
binned_weights = []

print("Bin range (m/s): Number of droplet concentration values")

# Iterate through percentile bins
for i in range(len(percentiles) - 1):
    bin_min = decile_boundaries[i]
    bin_max = decile_boundaries[i + 1]
    bin_indices = np.where((wind_speeds_min_sorted >= bin_min) & (wind_speeds_min_sorted < bin_max))[0]
    bin_concentrations = sum_values_sorted[bin_indices]
    
    if len(bin_concentrations) > 0:
        mean_concentration = np.mean(bin_concentrations)
        binned_wind_speeds.append((bin_min + bin_max) / 2)
        binned_mean_concentrations.append(mean_concentration)
        binned_weights.append(len(bin_concentrations))  # Using the number of points in each bin as weights
        print(f"{bin_min:.2f}-{bin_max:.2f}: {len(bin_concentrations)}")

# Perform WLS regression
binned_wind_speeds = np.array(binned_wind_speeds)
binned_mean_concentrations = np.array(binned_mean_concentrations)
binned_weights = np.array(binned_weights)

# Add a constant (intercept) to the model
X = sm.add_constant(binned_wind_speeds)
model = WLS(binned_mean_concentrations, X, weights=binned_weights)
results = model.fit()

# Print the results
print(results.summary())

# Extract the fitted parameters and their uncertainties
slope = results.params[1]
intercept = results.params[0]
slope_std_err = results.bse[1]
intercept_std_err = results.bse[0]

print(f"Slope: {slope:.4f} ± {slope_std_err:.4f}")
print(f"Intercept: {intercept:.4f} ± {intercept_std_err:.4f}")

# Create the scatterplot of binned means with the regression line
plt.figure(figsize=(10, 6))
plt.scatter(binned_wind_speeds, binned_mean_concentrations, color='blue', label='Mean droplet conc.')
plt.plot(binned_wind_speeds, intercept + slope * binned_wind_speeds, color='red', label='WLS fit')
plt.title('Minimum Altitude January - June 2022 \n(Percentile Binning)', fontsize=14, fontweight='bold')
plt.xlabel('Mean wind speed (m/s)', fontsize=12, fontweight='bold')
plt.ylabel('Mean droplet concentration dN (per cm^3) \nfor clear skies', fontsize=12, fontweight='bold')
# plt.yscale('log')
# plt.ylim(10**-2, 10**2)
plt.legend()
plt.show()


##Slope: 0.0325 ± 0.0213
##Intercept: 0.3566 ± 0.1418

#%%
## WLS regression for decile binning exponential 


##WLS regression with 10 bins
sum_values = [entry['Sum'] for entry in Y_sum_list]
wind_speeds_bcb = corrected_calc_min['Corrected_min_windspeed']

# Filter out NaN values
valid_indices = [i for i in range(len(wind_speeds_min)) if not np.isnan(wind_speeds_min[i]) and not np.isnan(sum_values[i])]
wind_speeds_min_filtered = [wind_speeds_min[i] for i in valid_indices]
sum_values_filtered = [sum_values[i] for i in valid_indices]

# Sort data by wind speed
sorted_indices = np.argsort(wind_speeds_min_filtered)
wind_speeds_min_sorted = np.array(wind_speeds_min_filtered)[sorted_indices]
sum_values_sorted = np.array(sum_values_filtered)[sorted_indices]

# Calculate percentiles for binning
num_points = len(wind_speeds_min_sorted)
percentiles = np.linspace(0, 100, 11)  # Generate 21 percentiles (0, 5, 10, ..., 100)
decile_boundaries = np.percentile(wind_speeds_min_sorted, percentiles)

# Initialize lists for binned data
binned_wind_speeds = []
binned_mean_concentrations = []
binned_weights = []

print("Bin range (m/s): Number of droplet concentration values")

# Iterate through percentile bins
for i in range(len(percentiles) - 1):
    bin_min = decile_boundaries[i]
    bin_max = decile_boundaries[i + 1]
    bin_indices = np.where((wind_speeds_min_sorted >= bin_min) & (wind_speeds_min_sorted < bin_max))[0]
    bin_concentrations = sum_values_sorted[bin_indices]
    
    if len(bin_concentrations) > 0:
        mean_concentration = np.mean(bin_concentrations)
        binned_wind_speeds.append((bin_min + bin_max) / 2)
        binned_mean_concentrations.append(mean_concentration)
        binned_weights.append(len(bin_concentrations))  # Using the number of points in each bin as weights
        print(f"{bin_min:.2f}-{bin_max:.2f}: {len(bin_concentrations)}")

# Perform WLS regression
binned_wind_speeds = np.array(binned_wind_speeds)
binned_mean_concentrations = np.array(binned_mean_concentrations)
binned_weights = np.array(binned_weights)

binned_mean_concentrations_positive = [y if y > 0 else 1e-10 for y in binned_mean_concentrations]
log_y = np.log(binned_mean_concentrations_positive)

X = sm.add_constant(binned_wind_speeds)
y = log_y

# Perform WLS regression
wls_model = WLS(y, X, weights=binned_weights)
wls_results = wls_model.fit()

# Print the results
print(wls_results.summary())

# Extract the slope and intercept
slope = wls_results.params[1]
intercept = wls_results.params[0]
slope_std_err = wls_results.bse[1]
intercept_std_err = wls_results.bse[0]

print(f"Slope: {slope:.4f} ± {slope_std_err:.4f}")
print(f"Intercept: {intercept:.4f} ± {intercept_std_err:.4f}")

# Create the scatterplot of binned means with the regression line
plt.figure(figsize=(10, 6))
plt.scatter(binned_wind_speeds, binned_mean_concentrations, color='blue', label='Mean droplet conc.')
plt.plot(binned_wind_speeds, np.exp(wls_results.predict(X)), color='red', label='WLS fit exponential ')
plt.title('Minimum Altitude January - June 2022 \n(Decile Binning)', fontsize=14, fontweight='bold')
plt.xlabel('Mean wind speed (m/s)', fontsize=12, fontweight='bold')
plt.ylabel('Mean droplet concentration dN (per cm^3) \nfor clear skies', fontsize=12, fontweight='bold')
plt.legend()
plt.show()

#%%
#WLS Regression for 9 bins


sum_values = [entry['Sum'] for entry in Y_sum_list]
wind_speeds_bcb = corrected_calc_min['Corrected_min_windspeed']

# Filter out NaN values
valid_indices = [i for i in range(len(wind_speeds_min)) if not np.isnan(wind_speeds_min[i]) and not np.isnan(sum_values[i])]
wind_speeds_min_filtered = [wind_speeds_min[i] for i in valid_indices]
sum_values_filtered = [sum_values[i] for i in valid_indices]

# Sort data by wind speed
sorted_indices = np.argsort(wind_speeds_min_filtered)
wind_speeds_min_sorted = np.array(wind_speeds_min_filtered)[sorted_indices]
sum_values_sorted = np.array(sum_values_filtered)[sorted_indices]

# Calculate percentiles for binning
num_points = len(wind_speeds_min_sorted)
percentiles = np.linspace(0, 100, 10)  # Generate 21 percentiles (0, 5, 10, ..., 100)
decile_boundaries = np.percentile(wind_speeds_min_sorted, percentiles)

# Initialize lists for binned data
binned_wind_speeds = []
binned_mean_concentrations = []
binned_weights = []

print("Bin range (m/s): Number of droplet concentration values")

# Iterate through percentile bins
for i in range(len(percentiles) - 1):
    bin_min = decile_boundaries[i]
    bin_max = decile_boundaries[i + 1]
    bin_indices = np.where((wind_speeds_min_sorted >= bin_min) & (wind_speeds_min_sorted < bin_max))[0]
    bin_concentrations = sum_values_sorted[bin_indices]
    
    if len(bin_concentrations) > 0:
        mean_concentration = np.mean(bin_concentrations)
        binned_wind_speeds.append((bin_min + bin_max) / 2)
        binned_mean_concentrations.append(mean_concentration)
        binned_weights.append(len(bin_concentrations))  # Using the number of points in each bin as weights
        print(f"{bin_min:.2f}-{bin_max:.2f}: {len(bin_concentrations)}")

# Perform WLS regression
binned_wind_speeds = np.array(binned_wind_speeds)
binned_mean_concentrations = np.array(binned_mean_concentrations)
binned_weights = np.array(binned_weights)

# Add a constant (intercept) to the model
X = sm.add_constant(binned_wind_speeds)
model = WLS(binned_mean_concentrations, X, weights=binned_weights)
results = model.fit()

# Print the results
print(results.summary())

# Extract the fitted parameters and their uncertainties
slope = results.params[1]
intercept = results.params[0]
slope_std_err = results.bse[1]
intercept_std_err = results.bse[0]

print(f"Slope: {slope:.4f} ± {slope_std_err:.4f}")
print(f"Intercept: {intercept:.4f} ± {intercept_std_err:.4f}")

# Create the scatterplot of binned means with the regression line
plt.figure(figsize=(10, 6))
plt.scatter(binned_wind_speeds, binned_mean_concentrations, color='blue', label='Mean droplet conc.')
plt.plot(binned_wind_speeds, intercept + slope * binned_wind_speeds, color='red', label='WLS fit')
plt.title('Minimum Altitude January - June 2022 \n(Percentile Binning)', fontsize=14, fontweight='bold')
plt.xlabel('Mean wind speed (m/s)', fontsize=12, fontweight='bold')
plt.ylabel('Mean droplet concentration dN (per cm^3) \nfor clear skies', fontsize=12, fontweight='bold')
plt.yscale('log')
plt.ylim(10**-2, 10**2)
plt.legend()
plt.show()


##Slope: 0.0315 ± 0.0193
##Intercept: 0.3629 ± 0.1287

#%%

#WLS Regression 15 bins


sum_values = [entry['Sum'] for entry in Y_sum_list]
wind_speeds_bcb = corrected_calc_min['Corrected_min_windspeed']

# Filter out NaN values
valid_indices = [i for i in range(len(wind_speeds_min)) if not np.isnan(wind_speeds_min[i]) and not np.isnan(sum_values[i])]
wind_speeds_min_filtered = [wind_speeds_min[i] for i in valid_indices]
sum_values_filtered = [sum_values[i] for i in valid_indices]

# Sort data by wind speed
sorted_indices = np.argsort(wind_speeds_min_filtered)
wind_speeds_min_sorted = np.array(wind_speeds_min_filtered)[sorted_indices]
sum_values_sorted = np.array(sum_values_filtered)[sorted_indices]

# Calculate percentiles for binning
num_points = len(wind_speeds_min_sorted)
percentiles = np.linspace(0, 100, 16)  # Generate 21 percentiles (0, 5, 10, ..., 100)
decile_boundaries = np.percentile(wind_speeds_min_sorted, percentiles)

# Initialize lists for binned data
binned_wind_speeds = []
binned_mean_concentrations = []
binned_weights = []

print("Bin range (m/s): Number of droplet concentration values")

# Iterate through percentile bins
for i in range(len(percentiles) - 1):
    bin_min = decile_boundaries[i]
    bin_max = decile_boundaries[i + 1]
    bin_indices = np.where((wind_speeds_min_sorted >= bin_min) & (wind_speeds_min_sorted < bin_max))[0]
    bin_concentrations = sum_values_sorted[bin_indices]
    
    if len(bin_concentrations) > 0:
        mean_concentration = np.mean(bin_concentrations)
        binned_wind_speeds.append((bin_min + bin_max) / 2)
        binned_mean_concentrations.append(mean_concentration)
        binned_weights.append(len(bin_concentrations))  # Using the number of points in each bin as weights
        print(f"{bin_min:.2f}-{bin_max:.2f}: {len(bin_concentrations)}")

# Perform WLS regression
binned_wind_speeds = np.array(binned_wind_speeds)
binned_mean_concentrations = np.array(binned_mean_concentrations)
binned_weights = np.array(binned_weights)

# Add a constant (intercept) to the model
X = sm.add_constant(binned_wind_speeds)
model = WLS(binned_mean_concentrations, X, weights=binned_weights)
results = model.fit()

# Print the results
print(results.summary())

# Extract the fitted parameters and their uncertainties
slope = results.params[1]
intercept = results.params[0]
slope_std_err = results.bse[1]
intercept_std_err = results.bse[0]

print(f"Slope: {slope:.4f} ± {slope_std_err:.4f}")
print(f"Intercept: {intercept:.4f} ± {intercept_std_err:.4f}")

# Create the scatterplot of binned means with the regression line
plt.figure(figsize=(10, 6))
plt.scatter(binned_wind_speeds, binned_mean_concentrations, color='blue', label='Mean droplet conc.')
plt.plot(binned_wind_speeds, intercept + slope * binned_wind_speeds, color='red', label='WLS fit')
plt.title('Minimum Altitude January - June 2022 \n(Percentile Binning)', fontsize=14, fontweight='bold')
plt.xlabel('Mean wind speed (m/s)', fontsize=12, fontweight='bold')
plt.ylabel('Mean droplet concentration dN (per cm^3) \nfor clear skies', fontsize=12, fontweight='bold')
plt.yscale('log')
plt.ylim(10**-2, 10**2)
plt.legend()
plt.show()


##Slope:  0.0276 ± 0.0221
##Intercept: 0.3869 ± 0.1461

#%%

#WLS Regression 20 bins

sum_values = [entry['Sum'] for entry in Y_sum_list]
wind_speeds_bcb = corrected_calc_min['Corrected_min_windspeed']

# Filter out NaN values
valid_indices = [i for i in range(len(wind_speeds_min)) if not np.isnan(wind_speeds_min[i]) and not np.isnan(sum_values[i])]
wind_speeds_min_filtered = [wind_speeds_min[i] for i in valid_indices]
sum_values_filtered = [sum_values[i] for i in valid_indices]

# Sort data by wind speed
sorted_indices = np.argsort(wind_speeds_min_filtered)
wind_speeds_min_sorted = np.array(wind_speeds_min_filtered)[sorted_indices]
sum_values_sorted = np.array(sum_values_filtered)[sorted_indices]

# Calculate percentiles for binning
num_points = len(wind_speeds_min_sorted)
percentiles = np.linspace(0, 100, 21)  # Generate 21 percentiles (0, 5, 10, ..., 100)
decile_boundaries = np.percentile(wind_speeds_min_sorted, percentiles)

# Initialize lists for binned data
binned_wind_speeds = []
binned_mean_concentrations = []
binned_weights = []

print("Bin range (m/s): Number of droplet concentration values")

# Iterate through percentile bins
for i in range(len(percentiles) - 1):
    bin_min = decile_boundaries[i]
    bin_max = decile_boundaries[i + 1]
    bin_indices = np.where((wind_speeds_min_sorted >= bin_min) & (wind_speeds_min_sorted < bin_max))[0]
    bin_concentrations = sum_values_sorted[bin_indices]
    
    if len(bin_concentrations) > 0:
        mean_concentration = np.mean(bin_concentrations)
        binned_wind_speeds.append((bin_min + bin_max) / 2)
        binned_mean_concentrations.append(mean_concentration)
        binned_weights.append(len(bin_concentrations))  # Using the number of points in each bin as weights
        print(f"{bin_min:.2f}-{bin_max:.2f}: {len(bin_concentrations)}")

# Perform WLS regression
binned_wind_speeds = np.array(binned_wind_speeds)
binned_mean_concentrations = np.array(binned_mean_concentrations)
binned_weights = np.array(binned_weights)

# Add a constant (intercept) to the model
X = sm.add_constant(binned_wind_speeds)
model = WLS(binned_mean_concentrations, X, weights=binned_weights)
results = model.fit()

# Print the results
print(results.summary())

# Extract the fitted parameters and their uncertainties
slope = results.params[1]
intercept = results.params[0]
slope_std_err = results.bse[1]
intercept_std_err = results.bse[0]

print(f"Slope: {slope:.4f} ± {slope_std_err:.4f}")
print(f"Intercept: {intercept:.4f} ± {intercept_std_err:.4f}")

# Create the scatterplot of binned means with the regression line
plt.figure(figsize=(10, 6))
plt.scatter(binned_wind_speeds, binned_mean_concentrations, color='blue', label='Mean droplet conc.')
plt.plot(binned_wind_speeds, intercept + slope * binned_wind_speeds, color='red', label='WLS fit')
plt.title('Minimum Altitude January - June 2022 \n(Percentile Binning)', fontsize=14, fontweight='bold')
plt.xlabel('Mean wind speed (m/s)', fontsize=12, fontweight='bold')
plt.ylabel('Mean droplet concentration dN (per cm^3) \nfor clear skies', fontsize=12, fontweight='bold')
plt.yscale('log')
plt.ylim(10**-2, 10**2)
plt.legend()
plt.show()


##Slope:  0.0301 ± 0.0237
##Intercept:  0.3727 ± 0.1562
# %%

# Assuming binned_wind_speeds and binned_mean_concentrations are already defined

# Convert lists to numpy arrays for regression
x = np.array(binned_wind_speeds)
y = np.array(binned_mean_concentrations)

# Take natural logarithm of y
y_log = np.log(y)

# Perform linear regression on log-transformed data
slope, intercept, r_value, p_value, std_err = linregress(x, y_log)

# Calculate regression line in log space
regression_line_log = slope * x + intercept

# Convert regression line back to original scale
regression_line = np.exp(regression_line_log)

# Plot the data and regression line
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Data')
plt.plot(x, regression_line, color='red', linestyle='--', label='Linear Regression on \nlog transformed data')
plt.title('Minimum Altitude January - June 2022', fontsize=14, fontweight='bold')
plt.xlabel('Mean wind speed (m/s)', fontsize=12, fontweight='bold')
plt.ylabel('Mean droplet concentration dN (per cm^3)', fontsize=12, fontweight='bold')
plt.yscale('log')  # Set y-axis to log scale
# plt.ylim(10**-2, 10**2)
plt.legend()
plt.show()

# Print regression results
print(f"Slope: {slope:.2f}")
print(f"Intercept: {intercept:.2f}")
print(f"R-squared: {r_value**2:.2f}")
print(f"P-value: {p_value:.4f}")



# %%
# Convert lists to numpy arrays for regression
x = np.array(binned_wind_speeds)
y = np.array(binned_mean_concentrations)

# Take natural logarithm of y
y_log = np.log(y)

# Perform linear regression on log-transformed data
slope, intercept, r_value, p_value, std_err = linregress(x, y_log)

# Calculate fitted values in log space
y_log_fit = slope * x + intercept

# Convert fitted values back to original scale (exponential)
y_fit = np.exp(y_log_fit)

# Plot the data and fitted exponential curve
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Mean droplet concentration')
plt.plot(x, y_fit, color='red', linestyle='--', label='Exponential Fit')
plt.title('Minimum Altitude  January-June 2022', fontsize=14, fontweight='bold')
plt.xlabel('Mean wind speed (m/s)', fontsize=12, fontweight='bold')
plt.ylabel('Mean droplet concentration dN (per cm^3)', fontsize=12, fontweight='bold')
plt.yscale('log')  # Set y-axis to log scale
# plt.ylim(10**-2, 10**2)
plt.legend()

plt.show()

# Print regression results
print(f"Slope: {slope:.2f}")
print(f"Intercept: {intercept:.2f}")
print(f"R-squared: {r_value**2:.2f}")
print(f"P-value: {p_value:.4f}")




# %%

# Assuming binned_wind_speeds and binned_mean_concentrations are already defined

# Convert lists to numpy arrays for regression
x = np.array(binned_wind_speeds)
y = np.array(binned_mean_concentrations)

# Define weights (example: inverse of x values)
weights = 1 / x

# Take natural logarithm of y
y_log = np.log(y)

# Perform WLS regression on log-transformed data
coefficients = np.polyfit(x, y_log, deg=1, w=weights)

# Extract slope and intercept from coefficients
slope, intercept = coefficients

# Calculate fitted values in log space
y_log_fit = slope * x + intercept

# Convert fitted values back to original scale (exponential)
y_fit = np.exp(y_log_fit)

# Plot the data and fitted exponential curve
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Mean droplet concentration')
plt.plot(x, y_fit, color='red', linestyle='--', label='Exponential Fit (WLS)')
plt.title('Minimum Altitude January-June 2022', fontsize=14, fontweight='bold')
plt.xlabel('Mean wind speed (m/s)', fontsize=12, fontweight='bold')
plt.ylabel('Mean droplet concentration dN (per cm^3)', fontsize=12, fontweight='bold')
plt.yscale('log')  # Set y-axis to log scale
# plt.ylim(10**-2, 10**2)
plt.legend()
# plt.grid(True)
plt.show()

# Print regression results
print(f"Slope: {slope:.2f}")
print(f"Intercept: {intercept:.2f}")



# %%
#Non log space WLS regression


# Assuming binned_wind_speeds and binned_mean_concentrations are already defined

# Convert lists to numpy arrays for regression
x = np.array(binned_wind_speeds)
y = np.array(binned_mean_concentrations)

# Define weights (example: inverse of x values)
weights = 1 / x

# Perform WLS regression
coefficients = np.polyfit(x, y, deg=1, w=weights)

# Extract slope and intercept from coefficients
slope, intercept = coefficients

# Calculate fitted values
y_fit = slope * x + intercept

# Plot the data and fitted line
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Mean Droplet Concentration')
plt.plot(x, y_fit, color='red', linestyle='--', label='WLS Fit')
plt.title('Minimum Altitude January - June 2022', fontsize=14, fontweight='bold')
plt.xlabel('Mean wind speed (m/s)', fontsize=12, fontweight='bold')
plt.ylabel('Mean droplet concentration dN (per cm^3) for clear skies', fontsize=12, fontweight='bold')
# plt.ylim(0, 2)  # Adjust y-axis limits as needed
plt.legend()
plt.yscale('log')
plt.show()

# Print regression results
print(f"Slope: {slope:.2f}")
print(f"Intercept: {intercept:.2f}")

# %%
## WLS regression 0.5 m/s bins with stats output 

# Step 1: Extract data for the initial scatterplot
sum_values = [entry['Sum'] for entry in Y_sum_list]
wind_speeds_min = corrected_calc_min['Corrected_min_windspeed']

# Filter out NaN values
valid_indices = [i for i in range(len(wind_speeds_min)) if not np.isnan(wind_speeds_min[i]) and not np.isnan(sum_values[i])]
wind_speeds_min_filtered = [wind_speeds_min[i] for i in valid_indices]
sum_values_filtered = [sum_values[i] for i in valid_indices]


# Step 2: Binning process to calculate mean droplet concentration
bin_width = 0.5
bins = np.arange(0, max(wind_speeds_min_filtered) + bin_width, bin_width)
binned_wind_speeds = []
binned_mean_concentrations = []
weights = []

print("Bin range (m/s): Number of droplet concentration values")

for i in range(len(bins) - 1):
    bin_min = bins[i]
    bin_max = bins[i + 1]
    bin_indices = [j for j in range(len(wind_speeds_min_filtered)) if bin_min <= wind_speeds_min_filtered[j] < bin_max]
    bin_concentrations = [sum_values_filtered[j] for j in bin_indices]
    if bin_concentrations:
        mean_concentration = np.mean(bin_concentrations)
        binned_wind_speeds.append((bin_min + bin_max) / 2)  # Use the midpoint of the bin for plotting
        binned_mean_concentrations.append(mean_concentration)
        weights.append(len(bin_concentrations))  # The weight is the number of values in the bin
        # Print the bin range and the number of values used to calculate the mean for that bin
        print(f"{bin_min:.2f}-{bin_max:.2f}: {len(bin_concentrations)}")

# Create the new scatterplot of binned means
plt.figure(figsize=(10, 6))
plt.scatter(binned_wind_speeds, binned_mean_concentrations, color='green', label='mean droplet conc.')
plt.title('Minimum Altitude January - June 2022 \n(Binned 0.5 m/s)', fontsize=12, fontweight='bold')
plt.xlabel('Mean windspeed (m/s)', fontsize=12, fontweight='bold')
plt.ylabel('Mean droplet concentration dN (per cm^3) \nfor clear skies', fontsize=12, fontweight='bold')
plt.legend()
# plt.ylim(10**-2, 10**1)
# plt.yscale('log')
# plt.xlim(0, 16)
plt.show()

# Step 3: Perform weighted least squares regression
# Convert lists to numpy arrays
binned_wind_speeds_np = np.array(binned_wind_speeds)
binned_mean_concentrations_np = np.array(binned_mean_concentrations)
weights_np = np.array(weights)

# Add a constant to the predictor variable (for the intercept term)
X = sm.add_constant(binned_wind_speeds_np)

# Fit the weighted least squares model
model = sm.WLS(binned_mean_concentrations_np, X, weights=weights_np)
results = model.fit()

# Print the summary of the regression results
print(results.summary())

# Extract the slope and intercept
slope = results.params[1]
intercept = results.params[0]

print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")

# Step 4: Plot the regression line
plt.figure(figsize=(10, 6))
plt.scatter(binned_wind_speeds, binned_mean_concentrations, color='green', label='mean droplet conc.')
plt.plot(binned_wind_speeds, results.predict(X), color='purple', label='WLS Regression Line')
plt.title('Minimum Altitude January - June 2022 \n(Binned 0.5 m/s)', fontsize=12, fontweight='bold')
plt.xlabel('Mean windspeed (m/s)', fontsize=12, fontweight='bold')
plt.ylabel('Mean droplet concentration dN (per cm^3) \nfor clear skies', fontsize=12, fontweight='bold')
plt.legend()
# plt.ylim(10**-2, 10**1)
# plt.yscale('log')
plt.xlim(0, 17)
plt.show()

# %%
## WLS regression 0.5 m/s bins exponential 


# Step 1: Extract data for the initial scatterplot
sum_values = [entry['Sum'] for entry in Y_sum_list]
wind_speeds_min = corrected_calc_min['Corrected_min_windspeed']

# Filter out NaN values
valid_indices = [i for i in range(len(wind_speeds_min)) if not np.isnan(wind_speeds_min[i]) and not np.isnan(sum_values[i])]
wind_speeds_min_filtered = [wind_speeds_min[i] for i in valid_indices]
sum_values_filtered = [sum_values[i] for i in valid_indices]


# Step 2: Binning process to calculate mean droplet concentration
bin_width = 0.5
bins = np.arange(0, max(wind_speeds_min_filtered) + bin_width, bin_width)
binned_wind_speeds = []
binned_mean_concentrations = []
weights = []

print("Bin range (m/s): Number of droplet concentration values")

for i in range(len(bins) - 1):
    bin_min = bins[i]
    bin_max = bins[i + 1]
    bin_indices = [j for j in range(len(wind_speeds_min_filtered)) if bin_min <= wind_speeds_min_filtered[j] < bin_max]
    bin_concentrations = [sum_values_filtered[j] for j in bin_indices]
    if bin_concentrations:
        mean_concentration = np.mean(bin_concentrations)
        binned_wind_speeds.append((bin_min + bin_max) / 2)  # Use the midpoint of the bin for plotting
        binned_mean_concentrations.append(mean_concentration)
        weights.append(len(bin_concentrations))  # The weight is the number of values in the bin
        # Print the bin range and the number of values used to calculate the mean for that bin
        print(f"{bin_min:.2f}-{bin_max:.2f}: {len(bin_concentrations)}")

# Create the new scatterplot of binned means
plt.figure(figsize=(10, 6))
plt.scatter(binned_wind_speeds, binned_mean_concentrations, color='green', label='mean droplet conc.')
plt.title('Minimum Altitude January - June 2022 \n(Binned 0.5 m/s)', fontsize=12, fontweight='bold')
plt.xlabel('Mean windspeed (m/s)', fontsize=12, fontweight='bold')
plt.ylabel('Mean droplet concentration dN (per cm^3) \nfor clear skies', fontsize=12, fontweight='bold')
plt.legend()
# plt.ylim(10**-2, 10**1)
# plt.yscale('log')
# plt.xlim(0, 16)
plt.show()

# Step 3: Perform weighted least squares regression
# Convert lists to numpy arrays
binned_wind_speeds_np = np.array(binned_wind_speeds)
binned_mean_concentrations_np = np.array(binned_mean_concentrations)
weights_np = np.array(weights)

binned_mean_concentrations_positive = [y if y > 0 else 1e-10 for y in binned_mean_concentrations]
log_y = np.log(binned_mean_concentrations_positive)

# Prepare data for WLS regression
X = sm.add_constant(binned_wind_speeds)  # Add a constant term for the intercept
y = log_y

# Perform WLS regression
wls_model = sm.WLS(y, X, weights=weights)
wls_results = wls_model.fit()

# Print regression results
print(wls_results.summary())

# Extract the slope and intercept
intercept, slope = wls_results.params
intercept_std_err, slope_std_err = wls_results.bse

# Print the slope and intercept with their standard errors
print(f"Slope: {slope:.4f} ± {slope_std_err:.4f}")
print(f"Intercept: {intercept:.4f} ± {intercept_std_err:.4f}")


# Step 4: Plot the regression line
plt.figure(figsize=(10, 6))
plt.scatter(binned_wind_speeds, binned_mean_concentrations, color='green', label='mean droplet conc.')
plt.plot(binned_wind_speeds,  np.exp(wls_results.predict(X)), color='brown', label='WLS Regression Line (Exponential)')
plt.title('Minimum Altitude January - June 2022 \n(Binned 0.5 m/s)', fontsize=12, fontweight='bold')
plt.xlabel('Mean windspeed (m/s)', fontsize=12, fontweight='bold')
plt.ylabel('Mean droplet concentration dN (per cm^3) \nfor clear skies', fontsize=12, fontweight='bold')
plt.legend()
# plt.ylim(10**-2, 10**1)
# plt.yscale('log')
plt.xlim(0, 17)
plt.show()
# %%
## WLS regression based on spread/variance exponential 

# Extract sum values
sum_values = [entry['Sum'] for entry in Y_sum_list]

# Extract wind speeds for below cloud base and minimum altitude
windspeeds_min = corrected_calc_min['Corrected_min_windspeed']


# Filter out NaN values
valid_indices = [i for i in range(len(wind_speeds_min)) if not np.isnan(wind_speeds_min[i]) and not np.isnan(sum_values[i])]
wind_speeds_min_filtered = [wind_speeds_min[i] for i in valid_indices]
sum_values_filtered = [sum_values[i] for i in valid_indices]

# Calculate bin boundaries based on variance
bin_width = np.std(wind_speeds_min_filtered)  # Adjust this based on your preference
bins = np.arange(min(wind_speeds_min_filtered), max(wind_speeds_min_filtered) + bin_width, bin_width)

# Initialize lists for binned data
binned_wind_speeds = []
binned_mean_concentrations = []
binned_weights = []

print("Bin range (m/s): Number of droplet concentration values")

# Iterate through bins
for i in range(len(bins) - 1):
    bin_min = bins[i]
    bin_max = bins[i + 1]
    bin_indices = np.where((wind_speeds_min_filtered >= bin_min) & (wind_speeds_min_filtered < bin_max))[0]
    bin_concentrations = sum_values_sorted[bin_indices]
    
    if len(bin_concentrations) > 0:
        mean_concentration = np.mean(bin_concentrations)
        binned_wind_speeds.append((bin_min + bin_max) / 2)
        binned_mean_concentrations.append(mean_concentration)
        binned_weights.append(len(bin_concentrations))
        print(f"{bin_min:.2f}-{bin_max:.2f}: {len(bin_concentrations)}")

# Convert lists to arrays
binned_wind_speeds = np.array(binned_wind_speeds)
binned_mean_concentrations = np.array(binned_mean_concentrations)
binned_weights = np.array(binned_weights)

# Transform the dependent variable to log scale
binned_mean_concentrations_positive = [y if y > 0 else 1e-10 for y in binned_mean_concentrations]
log_y = np.log(binned_mean_concentrations_positive)

# Add a constant (intercept) to the model
X = sm.add_constant(binned_wind_speeds)
y = log_y

# Perform WLS regression
wls_model = WLS(y, X, weights=binned_weights)
wls_results = wls_model.fit()

# Print the results
print(wls_results.summary())

# Extract the slope and intercept
slope = wls_results.params[1]
intercept = wls_results.params[0]
slope_std_err = wls_results.bse[1]
intercept_std_err = wls_results.bse[0]

print(f"Slope: {slope:.4f} ± {slope_std_err:.4f}")
print(f"Intercept: {intercept:.4f} ± {intercept_std_err:.4f}")

# Create the scatterplot of binned means with the exponential fit
plt.figure(figsize=(10, 6))
plt.scatter(binned_wind_speeds, binned_mean_concentrations, color='blue', label='Mean Droplet Conc.')
plt.plot(binned_wind_speeds, np.exp(wls_results.predict(X)), color='red', label='WLS Regression Line (Exponential)')
plt.title('Minimum Altitude January - June 2022 \n(Binned by Variance)', fontsize=14, fontweight='bold')
plt.xlabel('Mean wind speed (m/s)', fontsize=12, fontweight='bold')
plt.ylabel('Mean droplet concentration dN (per cm^3) \nfor clear skies', fontsize=12, fontweight='bold')
plt.legend()
plt.show()

# %%


## WLS based on spread/variance
# Extract sum values

# Extract sum values
sum_values = [entry['Sum'] for entry in Y_sum_list]
wind_speeds_min = corrected_calc_min['Corrected_min_windspeed']

# Filter out NaN values
valid_indices = [i for i in range(len(wind_speeds_min)) if not np.isnan(wind_speeds_min[i]) and not np.isnan(sum_values[i])]
wind_speeds_min_filtered = np.array([wind_speeds_min[i] for i in valid_indices])
sum_values_filtered = np.array([sum_values[i] for i in valid_indices])

# Calculate bin boundaries based on variance
bin_width = np.std(wind_speeds_min_filtered)  # Adjust this based on your preference
bins = np.arange(min(wind_speeds_min_filtered), max(wind_speeds_min_filtered) + bin_width, bin_width)

# Initialize lists for binned data
binned_wind_speeds = []
binned_mean_concentrations = []
binned_weights = []

print("Bin range (m/s): Number of droplet concentration values")

# Iterate through bins
for i in range(len(bins) - 1):
    bin_min = bins[i]
    bin_max = bins[i + 1]
    bin_indices = np.where((wind_speeds_min_filtered >= bin_min) & (wind_speeds_min_filtered < bin_max))[0]
    bin_concentrations = sum_values_filtered[bin_indices]
    
    if len(bin_concentrations) > 0:
        mean_concentration = np.mean(bin_concentrations)
        binned_wind_speeds.append((bin_min + bin_max) / 2)
        binned_mean_concentrations.append(mean_concentration)
        binned_weights.append(len(bin_concentrations))
        print(f"{bin_min:.2f}-{bin_max:.2f}: {len(bin_concentrations)}")

# Convert lists to arrays
binned_wind_speeds = np.array(binned_wind_speeds)
binned_mean_concentrations = np.array(binned_mean_concentrations)
binned_weights = np.array(binned_weights)

# Add a constant (intercept) to the model
X = sm.add_constant(binned_wind_speeds)
y = binned_mean_concentrations

# Perform WLS regression
wls_model = WLS(y, X, weights=binned_weights)
wls_results = wls_model.fit()

# Print the results
print(wls_results.summary())

# Extract the slope and intercept
slope = wls_results.params[1]
intercept = wls_results.params[0]
slope_std_err = wls_results.bse[1]
intercept_std_err = wls_results.bse[0]

print(f"Slope: {slope:.4f} ± {slope_std_err:.4f}")
print(f"Intercept: {intercept:.4f} ± {intercept_std_err:.4f}")

# Create the scatterplot of binned means with the linear WLS regression line
plt.figure(figsize=(10, 6))
plt.scatter(binned_wind_speeds, binned_mean_concentrations, color='blue', label='Mean Droplet Conc.')
plt.plot(binned_wind_speeds, wls_results.predict(X), color='red', label='WLS Regression Line (Linear)')
plt.title('Minimum Altitude January - June 2022 \n(Binned by Variance)', fontsize=14, fontweight='bold')
plt.xlabel('Mean wind speed (m/s)', fontsize=12, fontweight='bold')
plt.ylabel('Mean droplet concentration dN (per cm^3) \nfor clear skies', fontsize=12, fontweight='bold')
plt.legend()
plt.show()


# %%
