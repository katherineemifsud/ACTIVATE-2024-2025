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
#%%
C_0=math.log(0.61)-math.log(0.5)
C_01=math.log(0.68)-math.log(0.61)
C_02=math.log(0.75)-math.log(0.68)
C_03=math.log(0.82)-math.log(0.75)
C_04=math.log(0.89)-math.log(0.82)
C_05=math.log(0.96)-math.log(0.89)
C_06=math.log(1.03)-math.log(0.96)
C_07=math.log(1.1)-math.log(1.03)
C_08=math.log(1.17)-math.log(1.1)
C_09=math.log(1.25)-math.log(1.17)
C_10=math.log(1.5)-math.log(1.25)
C_11=math.log(2)-math.log(1.5)
C_12=math.log(2.5)-math.log(2)
C_13=math.log(3)-math.log(2.5)
C_14=math.log(3.5)-math.log(3)
C_15=math.log(4)-math.log(3.5)
C_16=math.log(5)-math.log(4)
C_17=math.log(6.5)-math.log(5)
C_18=math.log(7.2)-math.log(6.5)
C_19=math.log(7.9)-math.log(7.2)
C_20=math.log(10.2)-math.log(7.9)
C_21=math.log(12.5)-math.log(10.2)
C_22=math.log(15)-math.log(12.5)
C_23=math.log(20)-math.log(15)
C_24=math.log(25)-math.log(20)
C_25=math.log(30)-math.log(25)
C_26=math.log(35)-math.log(30)
C_27=math.log(40)-math.log(35)
C_28=math.log(45)-math.log(40)
C_29=math.log(50)-math.log(45)


bin_log=[C_0, C_01, C_02, C_03, C_04, C_05, C_06, C_07, C_08, C_09, C_10, C_11, C_12,C_12, C_13, C_14, C_15, C_16, 
        C_17, C_18, C_19, C_20, C_21, C_22, C_23, C_24, C_25, C_26, C_27, C_28, C_29]
bin_center=[0.555, 0.645, 0.715, 0.785, 0.855, 0.925, 
            0.995, 1.07, 1.14, 1.21, 1.38, 1.75, 2.25, 
            2.75, 3.25, 3.75, 4.5, 5.75, 6.85, 7.55, 
            9.05, 11.4, 13.8, 17.5, 22.5, 27.5, 32.5, 
            37.5, 42.5, 47.5]
bin_center = np.array(bin_center)

D0 = (0.61-0.5)
D01 = (0.68-0.61)
D02 = (0.75-0.68)
D03 = (0.82-0.75)
D04 = (0.89-0.82)
D05 = (0.96-0.86)
D06 = (1.03-0.96)
D07 = (1.1-1.03)
D08 = (1.17-1.1)
D09 = (1.25-1.17)
D10 = (1.5-1.25)
D11 = (2-1.5)
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

F0 = (C_0 / D0)
F01 = (C_01 / D01)
F02 = (C_02 / D02)
F03 = (C_03 / D03)
F04 = (C_04 / D04)
F05 = (C_05 / D05)
F06 = (C_06/ D06)
F07 = (C_07 / D07)
F08 = (C_08 / D08)
F09 = (C_09 / D09)
F10 = (C_10 / D10)
F11 = (C_11 / D11)
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


Logg = [F0, F01, F02, F03, F04, F05, F06, F07, F08, F09, F10, F11, F12,
        F13, F14, F15, F16, F17, F18, F19, F20, F21, F22, F23, F24, F25,
        F26, F27, F28, F29] 
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
    
            # if date == '2022-01-15':
            #     print(f"\nFirst row for date {'2022-01-15'}:")
            #     print(df_sum.head())
        # except Exception as e:
        #     print(f"Error processing file: {file_path}, Date: {date}")
        #     print(f"Error message: {str(e)}")
        #  
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
    
    all_Min_means=[]
    
    for i in range(len(Min_Start)):
        index_start = None
        index_end = None  
        start1 = int(Min_Start[i])
        end1 = Min_Stop[i]

        windspeed_altitude = {
            'Date': date,
            'Min_Start': start1,
            'Min_Stop': end1,
            'Altitude_mean': [],
            'Windspeed_mean': []
        }
      
        for i in range(len(times)):
            start3 = int(times[i][0:5])
            #print(times[i][0:5])
            if start3 == start1:
                index_start = i
                break
            
        for i in range(len(times)):
            end3 = int(times[i][0:5])
            if end3 == end1:
                index_end = i
                break
         
        if index_start == None:
            #print(date)
            #print('Did not find start time in Summary')
            winds3_mean = np.nan
            alts3_mean = np.nan
        if index_end == None:
            # print(date)
            winds3_mean = np.nan
            alts3_mean = np.nan
            break
        else:
            winds3 = winds[index_start:index_end]
            
            winds3_mean = np.nanmean(winds3)

            alts3 = alts[index_start:index_end]
            alts3_mean = np.nanmean(alts3)

        windspeed_altitude['Windspeed_mean'].append(winds3_mean)
        windspeed_altitude['Altitude_mean'].append(alts3_mean)

        all_Min_means.append(windspeed_altitude) #List that contains all the Min wind/alt mean dictionaries for 1 flight
    
    master_Min.append(all_Min_means) #List that contains all Min flights
#%%
master_BCB = []


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

# %%
Z0 = 0.03  # meters (typical value for open terrain)
Z10 = 10  # target height m


corrected_calc_min = {'Date': [], 'Corrected_mean_windspeed': []}

for flight in master_Min:
    for windspeed_altitude in flight:
        date = windspeed_altitude['Date']
        windspeed_means = windspeed_altitude['Windspeed_mean']
        altitude_means = windspeed_altitude['Altitude_mean']

        for windspeed_mean, altitude_mean in zip(windspeed_means, altitude_means):
            # Apply the formula
            new_windspeed_mean = windspeed_mean * (np.log(Z10/Z0) / np.log(altitude_mean / Z0))

            # Append to the new dictionary
            corrected_calc_min['Date'].append(date)
            corrected_calc_min['Corrected_mean_windspeed'].append(new_windspeed_mean)
            
for date, windspeed_mean in zip(corrected_calc_min['Date'], corrected_calc_min['Corrected_mean_windspeed']):
    print(f"Date: {date}, Corrected_mean_windspeed: {windspeed_mean}")
# %%
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
#CAS data

bin_name = ['CAS_Bin00', 'CAS_Bin01', 'CAS_Bin02', 'CAS_Bin03', 'CAS_Bin04',
            'CAS_Bin05', 'CAS_Bin06', 'CAS_Bin07', 'CAS_Bin08',
            'CAS_Bin09', 'CAS_Bin10', 'CAS_Bin11', 'CAS_Bin12', 'CAS_Bin13', 'CAS_Bin14', 'CAS_Bin15', 'CAS_Bin16', 'CAS_Bin17', 
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
        for col in ['Time_mid', 'LWC_CAS', 'CAS_Bin00', 
                    'CAS_Bin01', 'CAS_Bin02', 'CAS_Bin03', 'CAS_Bin04',
                    'CAS_Bin05', 'CAS_Bin06', 'CAS_Bin07', 'CAS_Bin08',
                    'CAS_Bin09', 'CAS_Bin10', 'CAS_Bin11',
                    'CAS_Bin12', 'CAS_Bin13', 'CAS_Bin14', 
                    'CAS_Bin15', 'CAS_Bin16', 'CAS_Bin17', 
                    'CAS_Bin18', 'CAS_Bin19', 'CAS_Bin20', 
                    'CAS_Bin21', 'CAS_Bin22', 'CAS_Bin23', 
                    'CAS_Bin24', 'CAS_Bin25', 'CAS_Bin26',
                    'CAS_Bin27', 'CAS_Bin28', 'CAS_Bin29']:
            if df_CAS[col].dtype == 'O':  # 'O' stands for Object (usually string columns)
                df_CAS[col] = df_CAS[col].str.strip('"')
        
        df_CAS['Time_mid']= pd.to_numeric(df_CAS['Time_mid'], errors='coerce')
        df_CAS['CAS_Bin00']= pd.to_numeric(df_CAS['CAS_Bin00'], errors='coerce')
        df_CAS['CAS_Bin01']= pd.to_numeric(df_CAS['CAS_Bin01'], errors='coerce')
        df_CAS['CAS_Bin02']= pd.to_numeric(df_CAS['CAS_Bin02'], errors='coerce')
        df_CAS['CAS_Bin03']= pd.to_numeric(df_CAS['CAS_Bin03'], errors='coerce')
        df_CAS['CAS_Bin04']= pd.to_numeric(df_CAS['CAS_Bin04'], errors='coerce')
        df_CAS['CAS_Bin05']= pd.to_numeric(df_CAS['CAS_Bin05'], errors='coerce')
        df_CAS['CAS_Bin06']= pd.to_numeric(df_CAS['CAS_Bin06'], errors='coerce')
        df_CAS['CAS_Bin07']= pd.to_numeric(df_CAS['CAS_Bin07'], errors='coerce')
        df_CAS['CAS_Bin08']= pd.to_numeric(df_CAS['CAS_Bin08'], errors='coerce')
        df_CAS['CAS_Bin09']= pd.to_numeric(df_CAS['CAS_Bin09'], errors='coerce')
        df_CAS['CAS_Bin10']= pd.to_numeric(df_CAS['CAS_Bin10'], errors='coerce')
        df_CAS['CAS_Bin11']= pd.to_numeric(df_CAS['CAS_Bin11'], errors='coerce')
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

# %%
# %%
master_CAS_BCB = []
leg_info = []

for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict = leg_data[i]

    flight_date = leg_dict['Date']
    BCB_index = leg_dict['LegIndex_02']
    Min_index = leg_dict['LegIndex_06']
    BCB_start = leg_dict['LegIndex_02']['StartTimes']
    BCB_stop = leg_dict['LegIndex_02']['StopTimes']
    Min_Start = leg_dict['LegIndex_06']['StartTimes']
    Min_Stop = leg_dict['LegIndex_06']['StopTimes']

    CAS_flight = CAS[i]
    times = CAS_flight.Time_mid.values
    Bin00 = CAS_flight.CAS_Bin00.values
    Bin01 = CAS_flight.CAS_Bin01.values
    Bin02 = CAS_flight.CAS_Bin02.values
    Bin03 = CAS_flight.CAS_Bin03.values
    Bin04 = CAS_flight.CAS_Bin04.values
    Bin05 = CAS_flight.CAS_Bin05.values
    Bin06 = CAS_flight.CAS_Bin06.values
    Bin07 = CAS_flight.CAS_Bin07.values
    Bin08 = CAS_flight.CAS_Bin08.values
    Bin09 = CAS_flight.CAS_Bin09.values
    Bin10 = CAS_flight.CAS_Bin10.values
    Bin11 = CAS_flight.CAS_Bin11.values
    Bin12 = CAS_flight.CAS_Bin12.values
    Bin13 = CAS_flight.CAS_Bin13.values
    Bin14 = CAS_flight.CAS_Bin14.values
    Bin15 = CAS_flight.CAS_Bin15.values
    Bin16 = CAS_flight.CAS_Bin16.values
    Bin17 = CAS_flight.CAS_Bin17.values
    Bin18 = CAS_flight.CAS_Bin18.values
    Bin19 = CAS_flight.CAS_Bin19.values
    Bin20 = CAS_flight.CAS_Bin20.values
    Bin21 = CAS_flight.CAS_Bin21.values
    Bin22 = CAS_flight.CAS_Bin22.values
    Bin23 = CAS_flight.CAS_Bin23.values
    Bin24 = CAS_flight.CAS_Bin24.values
    Bin25 = CAS_flight.CAS_Bin25.values
    Bin26 = CAS_flight.CAS_Bin26.values
    Bin27 = CAS_flight.CAS_Bin27.values
    Bin28 = CAS_flight.CAS_Bin28.values
    Bin29 = CAS_flight.CAS_Bin29.values
    lwc = CAS_flight.LWC_CAS.values

    total_BCB_means = []

    for i in range(len(BCB_start)):
        index7_start = None
        index7_end = None
        start20 = int(BCB_start[i])
        end20 = BCB_stop[i]

        lwc_labels = []

        for i in range(len(times)):
            start21 = int(times[i])
            if start21 == start20:
                index7_start = i
                break

        for i in range(len(times)):
            end21 = int(times[i])
            if end21 == end20:
                index7_end = i
                break

        if index7_start is not None and index7_end is not None:
            for i in range(index7_start, index7_end + 1):
                lwc_val = lwc[i]
                label = 'Y' if 0 <= lwc_val <= 0.01 else 'N'
                lwc_labels.append(label)

            leg_info.append({
                'Date': date,
                'BCB_start': start20,
                'BCB_stop': end20,
                'LWC_Labels': lwc_labels,
            })

    master_CAS_BCB.append(total_BCB_means)

# Print leg_info or use it as needed
for leg in leg_info:
    print(f"Date: {leg['Date']}, Start: {leg['BCB_start']}, Stop: {leg['BCB_stop']}, LWC Labels: {leg['LWC_Labels']}")
# %%
master_CAS_BCB = []
leg_info = []

for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict = leg_data[i]

    flight_date = leg_dict['Date']
    BCB_index = leg_dict['LegIndex_02']
    Min_index = leg_dict['LegIndex_06']
    BCB_start = leg_dict['LegIndex_02']['StartTimes']
    BCB_stop = leg_dict['LegIndex_02']['StopTimes']
    Min_Start = leg_dict['LegIndex_06']['StartTimes']
    Min_Stop = leg_dict['LegIndex_06']['StopTimes']

    CAS_flight = CAS[i]
    times = CAS_flight.Time_mid.values
    Bin00 = CAS_flight.CAS_Bin00.values
    Bin01 = CAS_flight.CAS_Bin01.values
    Bin02 = CAS_flight.CAS_Bin02.values
    Bin03 = CAS_flight.CAS_Bin03.values
    Bin04 = CAS_flight.CAS_Bin04.values
    Bin05 = CAS_flight.CAS_Bin05.values
    Bin06 = CAS_flight.CAS_Bin06.values
    Bin07 = CAS_flight.CAS_Bin07.values
    Bin08 = CAS_flight.CAS_Bin08.values
    Bin09 = CAS_flight.CAS_Bin09.values
    Bin10 = CAS_flight.CAS_Bin10.values
    Bin11 = CAS_flight.CAS_Bin11.values
    Bin12 = CAS_flight.CAS_Bin12.values
    Bin13 = CAS_flight.CAS_Bin13.values
    Bin14 = CAS_flight.CAS_Bin14.values
    Bin15 = CAS_flight.CAS_Bin15.values
    Bin16 = CAS_flight.CAS_Bin16.values
    Bin17 = CAS_flight.CAS_Bin17.values
    Bin18 = CAS_flight.CAS_Bin18.values
    Bin19 = CAS_flight.CAS_Bin19.values
    Bin20 = CAS_flight.CAS_Bin20.values
    Bin21 = CAS_flight.CAS_Bin21.values
    Bin22 = CAS_flight.CAS_Bin22.values
    Bin23 = CAS_flight.CAS_Bin23.values
    Bin24 = CAS_flight.CAS_Bin24.values
    Bin25 = CAS_flight.CAS_Bin25.values
    Bin26 = CAS_flight.CAS_Bin26.values
    Bin27 = CAS_flight.CAS_Bin27.values
    Bin28 = CAS_flight.CAS_Bin28.values
    Bin29 = CAS_flight.CAS_Bin29.values
    lwc = CAS_flight.LWC_CAS.values

    total_BCB_means = []

    for i in range(len(BCB_start)):
        index7_start = None
        index7_end = None
        start20 = int(BCB_start[i])
        end20 = BCB_stop[i]

        bin_means = {
            'Date': date,
            'BCB_start': start20,
            'BCB_stop': end20,
            'Bin00_Y_mean': [],
            'Bin01_Y_mean': [],
            'Bin02_Y_mean': [],
            'Bin03_Y_mean': [],
            'Bin04_Y_mean': [],
            'Bin05_Y_mean': [],
            'Bin06_Y_mean': [],
            'Bin07_Y_mean': [],
            'Bin08_Y_mean': [],
            'Bin09_Y_mean': [],
            'Bin10_Y_mean': [],
            'Bin11_Y_mean': [],
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
            'Bin00_N_mean': [],
            'Bin01_N_mean': [],
            'Bin02_N_mean': [],
            'Bin03_N_mean': [],
            'Bin04_N_mean': [],
            'Bin05_N_mean': [],
            'Bin06_N_mean': [],
            'Bin07_N_mean': [],
            'Bin08_N_mean': [],
            'Bin09_N_mean': [],
            'Bin10_N_mean': [],
            'Bin11_N_mean': [],
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

        for i in range(len(times)):
            start21 = int(times[i])
            if start21 == start20:
                index7_start = i
                break

        for i in range(len(times)):
            end21 = int(times[i])
            if end21 == end20:
                index7_end = i
                break

        if index7_start is not None and index7_end is not None:
            for i in range(index7_start, index7_end + 1):
                lwc_val = lwc[i]
                label = 'Y' if 0 <= lwc_val <= 0.01 else 'N'
                for bin_label in range(0, 30):
                    bin_key = f'Bin{bin_label}_{label}_mean'
                    bin_value = CAS_flight[f'CAS_Bin{bin_label}'].values[i]
                    bin_means[bin_key] = pd.to_numeric(bin_means[bin_key], errors='coerce')  # Convert to numeric, and handle errors by converting to NaN
                    bin_means[bin_key] = np.nan_to_num(bin_means[bin_key], nan=0)
                    bin_means[bin_key] = list(bin_means[bin_key]) + [bin_value]
                    bin_means[bin_key].append(bin_value)
            total_BCB_means.append(bin_means)

    master_CAS_BCB.append(total_BCB_means)

# Print or use master_CAS_BCB as needed
for item in master_CAS_BCB:
    for bin_means in item:
        print(f"Date: {bin_means['Date']}, Start: {bin_means['BCB_start']}, Stop: {bin_means['BCB_stop']}")
        for bin_label in range(0, 30):
            for label in ['Y', 'N']:
                bin_key = f'Bin{bin_label}_{label}_mean'
                mean_value = np.nanmean(bin_means[bin_key])
                print(f"   Bin{bin_label}_{label} Mean: {mean_value}")
# %%
