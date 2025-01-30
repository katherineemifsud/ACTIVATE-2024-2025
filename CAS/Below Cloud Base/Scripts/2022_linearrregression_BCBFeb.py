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


#%% 

#%%
#Summary Data
col_name = ['Time_mid', 'Latitude', 'Longitude', 'GPS_altitude', 'Pressure_Altitude',
             'Pitch', 'Roll', 'True_Heading', 'True_Air_Speed', 
             'Static_Air_Temp', 'IR_Surf_Temp', 'Static_Pressure',
             'Wind_Speed']
summary=[]
dates_sum = ['2022-02-01', '2022-02-02', '2022-02-03', '2022-02-05', 
             '2022-02-15', '2022-02-16', '2022-02-19', '2022-02-22',
             '2022-02-26']

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
        if date >= '2022-02-01':
            df_sum = pd.read_csv(file_path, skiprows=47, quoting=csv.QUOTE_NONE)
                 

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
 
# %%
#Leg data
leg_data = []
leg_name=['Time_Start', '  Time_Stop', '  Julian_Day', 
          '  Date', '  LegIndex']

dates_legs= ['2022-02-01', '2022-02-02', '2022-02-03', '2022-02-05', 
             '2022-02-15', '2022-02-16', '2022-02-19',  '2022-02-22',
             '2022-02-26']

for date in dates_legs:
    datestr = date.replace('-', '')
    fname_legs = sorted(glob.glob(f'/home/disk/eos4/kathem24/activate/data/MET/2022/LegFLags/csv/ACTIVATE-LegFlags_HU25_{datestr}_R*.csv'), reverse=True)

    leg_dictionary = {
        'Date': date,
        'LegIndex_02': {'StartTimes': [], 'StopTimes': []},
        'LegIndex_06': {'StartTimes': [], 'StopTimes': []}
    }

    for file_path in fname_legs:
        if date == '2022-02-05':
            df_legs = pd.read_csv(file_path, skiprows=44, quoting=csv.QUOTE_NONE)
        elif date < '2022-02-02':
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
#
## 2D-S Data Import

bin_name = ['dNdlogD_total_003_2DS', 'dNdlogD_total_004_2DS' ,'dNdlogD_total_005_2DS', 
            'dNdlogD_total_006_2DS']

twoDS=[]

dates_twoDS = ['2022-02-01', '2022-02-02', '2022-02-03', '2022-02-05', 
             '2022-02-15', '2022-02-16', '2022-02-19', '2022-02-22',
              '2022-02-26']

for date in dates_twoDS:
    datestr = date.replace('-', '')
    fname_twoDS = sorted(glob.glob(f'/home/disk/eos4/kathem24/activate/data/twoDspectrometer/horizontal/csv/ACTIVATE-2DS-H-Arm_HU25_{datestr}_R*.csv'), reverse=True)
    print(date)
    print(fname_twoDS)

    run = 1
    for file_path in fname_twoDS:
        nums_file_paths = len(fname_twoDS)
        if date <= ('2022-02-26'):
            df_2DS = pd.read_csv(file_path, skiprows= 423, quoting=csv.QUOTE_NONE)

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

        if date == '2022-02-26':
                print(f"\nFirst row for date {'2022-02-26'}:")
        
        print(date)
        time = df_2DS['Time_Start']
        print("yes")
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

#%%

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

#CAS data

bin_name = ['CAS_Bin00', 'CAS_Bin01', 'CAS_Bin02', 'CAS_Bin03', 'CAS_Bin04', 'CAS_Bin05', 'CAS_Bin06',
            'CAS_Bin07', 'CAS_Bin08', 'CAS_Bin09', 'CAS_Bin10', 'CAS_Bin11',
            'CAS_Bin12', 'CAS_Bin13', 'CAS_Bin14', 'CAS_Bin15', 'CAS_Bin16', 'CAS_Bin17', 
            'CAS_Bin18', 'CAS_Bin19', 'CAS_Bin20', 'CAS_Bin21', 'CAS_Bin22', 'CAS_Bin23', 'CAS_Bin24', 'CAS_Bin25', 'CAS_Bin26',
            'CAS_Bin27', 'CAS_Bin28', 'CAS_Bin29']

CAS = []

dates_CAS = ['2022-02-01', '2022-02-02', '2022-02-03', '2022-02-05', 
             '2022-02-15', '2022-02-16', '2022-02-19', '2022-02-22',
              '2022-02-26']

for date in dates_CAS:
    datestr = date.replace('-', '')
    fname_CAS = sorted(glob.glob(f'/home/disk/eos4/kathem24/activate/data/cloudaerospect/2022csv/ACTIVATE-LARGE-CAS_HU25_{datestr}_R*.csv'), reverse=True)
    # print(date)
    # print(fname_CAS)

    run = 1
    for file_path in fname_CAS:
        nums_file_paths = len(fname_CAS)

        if date <= ('2022-02-26'):
            df_CAS = pd.read_csv(file_path, skiprows= 71, quoting=csv.QUOTE_NONE)
             
        
        for bin_ in bin_name:
            if bin_ in df_CAS.columns:
                df_CAS.columns = df_CAS.columns.str.strip('"')
                df_CAS[bin_] = pd.to_numeric(df_CAS[bin_], errors='coerce')
                df_CAS.replace([-9999, -9999.00], np.NaN, inplace=True)
        for col in ['Time_mid', 'LWC_CAS', 'CAS_Bin00', 'CAS_Bin01', 'CAS_Bin02', 'CAS_Bin03', 'CAS_Bin04', 'CAS_Bin05', 'CAS_Bin06',
            'CAS_Bin07', 'CAS_Bin08', 'CAS_Bin09', 'CAS_Bin10', 'CAS_Bin11','CAS_Bin12', 'CAS_Bin13', 'CAS_Bin14', 
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

#%%

# master_CAS_BCB = []
# leg_info = []

# for i in range(len(dates_legs)):
#     date = dates_legs[i]
#     leg_dict = leg_data[i]

#     flight_date = leg_dict['Date']
#     BCB_index = leg_dict['LegIndex_02']
#     Min_index = leg_dict['LegIndex_06']
#     BCB_start = leg_dict['LegIndex_02']['StartTimes']
#     BCB_stop = leg_dict['LegIndex_02']['StopTimes']
#     Min_Start = leg_dict['LegIndex_06']['StartTimes']
#     Min_Stop = leg_dict['LegIndex_06']['StopTimes']

#     CAS_flight = CAS[i]
#     twoDS_flight = twoDS[i]
#     times = CAS_flight.Time_mid.values
#     times1 = twoDS_flight.Time_Start.values
#     Bin12 = CAS_flight.CAS_Bin12.values
#     Bin13 = CAS_flight.CAS_Bin13.values
#     Bin14 = CAS_flight.CAS_Bin14.values
#     Bin15 = CAS_flight.CAS_Bin15.values
#     Bin16 = CAS_flight.CAS_Bin16.values
#     Bin17 = CAS_flight.CAS_Bin17.values
#     Bin18 = CAS_flight.CAS_Bin18.values
#     Bin19 = CAS_flight.CAS_Bin19.values
#     Bin20 = CAS_flight.CAS_Bin20.values
#     Bin21 = CAS_flight.CAS_Bin21.values
#     Bin22 = CAS_flight.CAS_Bin22.values
#     Bin23 = CAS_flight.CAS_Bin23.values
#     Bin24 = CAS_flight.CAS_Bin24.values
#     Bin25 = CAS_flight.CAS_Bin25.values
#     Bin26 = CAS_flight.CAS_Bin26.values
#     Bin27 = CAS_flight.CAS_Bin27.values
#     Bin28 = CAS_flight.CAS_Bin28.values
#     Bin29 = CAS_flight.CAS_Bin29.values
#     lwc = CAS_flight.LWC_CAS.values
#     N_total = twoDS_flight['N-total_2DS'].values

#     cas_indices = []
#     twods_indices = []

#     for j, time in enumerate(times):
#         if time in times1:
#             cas_indices.append(j)
#             twods_indices.append(np.where(times1 == time)[0][0])

#     aligned_lwc = lwc[cas_indices]
#     aligned_N_total = N_total[twods_indices]

#     total_BCB_means = []

#     for k in range(len(BCB_start)):
#         index7_start = None
#         index7_end = None
#         start20 = int(BCB_start[k])
#         end20 = BCB_stop[k]

#         data_labels = []

#         for j in range(len(times)):
#             start21 = int(times[j])
#             if start21 == start20:
#                 index7_start = j
#                 break

#         for j in range(len(times)):
#             end21 = int(times[j])
#             if end21 == end20:
#                 index7_end = j
#                 break

#         if index7_start is not None and index7_end is not None:
#             BCB_means = []  # List to store the means for this BCB interval
#             for j in range(index7_start, index7_end + 1):
#                 lwc_val = aligned_lwc[j]
#                 N_val = aligned_N_total[j]
#                 lwc_label = 'Y' if 0 <= lwc_val <= 0.0025 else 'N'
#                 N_label = 'Y' if 0 <= N_val <= 100 else 'N'
#                 if lwc_label == 'Y' and N_label == 'Y':
#                     label = 'Y'
#                 else:
#                     label = 'N'
#                 data_labels.append(label)
#                 # Collect bin values for calculating means
#                 bin_values = [
#                     CAS_flight.CAS_Bin12.values[j], CAS_flight.CAS_Bin13.values[j],
#                     CAS_flight.CAS_Bin14.values[j], CAS_flight.CAS_Bin15.values[j],
#                     CAS_flight.CAS_Bin16.values[j], CAS_flight.CAS_Bin17.values[j],
#                     CAS_flight.CAS_Bin18.values[j], CAS_flight.CAS_Bin19.values[j],
#                     CAS_flight.CAS_Bin20.values[j], CAS_flight.CAS_Bin21.values[j],
#                     CAS_flight.CAS_Bin22.values[j], CAS_flight.CAS_Bin23.values[j],
#                     CAS_flight.CAS_Bin24.values[j], CAS_flight.CAS_Bin25.values[j],
#                     CAS_flight.CAS_Bin26.values[j], CAS_flight.CAS_Bin27.values[j],
#                     CAS_flight.CAS_Bin28.values[j], CAS_flight.CAS_Bin29.values[j]
#                 ]
#                 BCB_means.append(bin_values)

#             if BCB_means:
#                 BCB_means = np.mean(BCB_means, axis=0)  # Calculate means for this interval
#                 total_BCB_means.append(BCB_means)  # Add to the total BCB means list

#             leg_info.append({
#                 'Date': date,
#                 'BCB_start': start20,
#                 'BCB_stop': end20,
#                 'Data_Labels': data_labels,
#             })
    
#     master_CAS_BCB.append(total_BCB_means)

# # Print leg_info or use it as needed
# for leg in leg_info:
#     print(f"Date: {leg['Date']}, Start: {leg['BCB_start']}, Stop: {leg['BCB_stop']}, Data Labels: {leg['Data_Labels']}")

#     # for i in range(len(BCB_start)):
    #     index7_start = None
    #     index7_end = None
    #     start20 = int(BCB_start[i])
    #     end20 = BCB_stop[i]

    #     data_labels = []

    #     for i in range(len(times)):
    #         start21 = int(times[i])
    #         if start21 == start20:
    #             index7_start = i
    #             break

    #     for i in range(len(times)):
    #         end21 = int(times[i])
    #         if end21 == end20:
    #             index7_end = i
    #             break

    #     if index7_start is not None and index7_end is not None:
    #         for i in range(index7_start, index7_end + 1):
    #             lwc_val = lwc[i]
    #             label = 'Y' if 0 <= lwc_val <= 0.01 else 'N'
    #             data_labels.append(label)

    #         leg_info.append({
    #             'Date': date,
    #             'BCB_start': start20,
    #             'BCB_stop': end20,
    #             'LWC_Labels': data_labels,
    #         })

    # master_CAS_BCB.append(total_BCB_means)





#         if index7_start is not None and index7_end is not None:
#             for i in range(index7_start, index7_end + 1):
#                 lwc_val = lwc[i]
#                 N_val = N_total[i]
#                 lwc_label = 'Y' if 0 <= lwc_val <= 0.0025 else 'N'
#                 N_label = 'Y' if 0 <= N_val <= 100 else 'N'
#                 if lwc_label == 'Y' and N_label == 'Y':
#                     label = 'Y'
#                 else:
#                     label = 'N'
#                 data_labels.append(label)

#             leg_info.append({
#                 'Date': date,
#                 'BCB_start': start20,
#                 'BCB_stop': end20,
#                 'Data_Labels': data_labels,
#             })
#     master_CAS_BCB.append(total_BCB_means)

# # Print leg_info or use it as needed
# for leg in leg_info:
#     print(f"Date: {leg['Date']}, Start: {leg['BCB_start']}, Stop: {leg['BCB_stop']}, Data Labels: {leg['Data_Labels']}")

#%%
import numpy as np

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

#%%
# master_CAS_BCB = []
# leg_info = []

# for i in range(len(dates_legs)):
#     date = dates_legs[i]
#     leg_dict = leg_data[i]

#     flight_date = leg_dict['Date']
#     BCB_index = leg_dict['LegIndex_02']
#     Min_index = leg_dict['LegIndex_06']
#     BCB_start = leg_dict['LegIndex_02']['StartTimes']
#     BCB_stop = leg_dict['LegIndex_02']['StopTimes']
#     Min_Start = leg_dict['LegIndex_06']['StartTimes']
#     Min_Stop = leg_dict['LegIndex_06']['StopTimes']

#     CAS_flight = CAS[i]
#     times = CAS_flight.Time_mid.values
#     Bin12 = CAS_flight.CAS_Bin12.values
#     Bin13 = CAS_flight.CAS_Bin13.values
#     Bin14 = CAS_flight.CAS_Bin14.values
#     Bin15 = CAS_flight.CAS_Bin15.values
#     Bin16 = CAS_flight.CAS_Bin16.values
#     Bin17 = CAS_flight.CAS_Bin17.values
#     Bin18 = CAS_flight.CAS_Bin18.values
#     Bin19 = CAS_flight.CAS_Bin19.values
#     Bin20 = CAS_flight.CAS_Bin20.values
#     Bin21 = CAS_flight.CAS_Bin21.values
#     Bin22 = CAS_flight.CAS_Bin22.values
#     Bin23 = CAS_flight.CAS_Bin23.values
#     Bin24 = CAS_flight.CAS_Bin24.values
#     Bin25 = CAS_flight.CAS_Bin25.values
#     Bin26 = CAS_flight.CAS_Bin26.values
#     Bin27 = CAS_flight.CAS_Bin27.values
#     Bin28 = CAS_flight.CAS_Bin28.values
#     Bin29 = CAS_flight.CAS_Bin29.values
#     lwc = CAS_flight.LWC_CAS.values

#     total_BCB_means = []

#     for i in range(len(BCB_start)):
#         index7_start = None
#         index7_end = None
#         start20 = int(BCB_start[i])
#         end20 = BCB_stop[i]

#         bin_means = {
#             'Date': date,
#             'BCB_start': start20,
#             'BCB_stop': end20,
#             'Bin12_Y_mean': [],
#             'Bin13_Y_mean': [],
#             'Bin14_Y_mean': [],
#             'Bin15_Y_mean': [],
#             'Bin16_Y_mean': [],
#             'Bin17_Y_mean': [],
#             'Bin18_Y_mean': [],
#             'Bin19_Y_mean': [],
#             'Bin20_Y_mean': [],
#             'Bin21_Y_mean': [],
#             'Bin22_Y_mean': [],
#             'Bin23_Y_mean': [],
#             'Bin24_Y_mean': [],
#             'Bin25_Y_mean': [],
#             'Bin26_Y_mean': [],
#             'Bin27_Y_mean': [],
#             'Bin28_Y_mean': [],
#             'Bin29_Y_mean': [],
#             'Bin12_N_mean': [],
#             'Bin13_N_mean': [],
#             'Bin14_N_mean': [],
#             'Bin15_N_mean': [],
#             'Bin16_N_mean': [],
#             'Bin17_N_mean': [],
#             'Bin18_N_mean': [],
#             'Bin19_N_mean': [],
#             'Bin20_N_mean': [],
#             'Bin21_N_mean': [],
#             'Bin22_N_mean': [],
#             'Bin23_N_mean': [],
#             'Bin24_N_mean': [],
#             'Bin25_N_mean': [],
#             'Bin26_N_mean': [],
#             'Bin27_N_mean': [],
#             'Bin28_N_mean': [],
#             'Bin29_N_mean': [],
#         }

#         for i in range(len(times)):
#             start21 = int(times[i])
#             if start21 == start20:
#                 index7_start = i
#                 break

#         for i in range(len(times)):
#             end21 = int(times[i])
#             if end21 == end20:
#                 index7_end = i
#                 break

#         if index7_start is not None and index7_end is not None:
#             for i in range(index7_start, index7_end + 1):
#                 lwc_val = lwc[i]
#                 label = 'Y' if 0 <= lwc_val <= 0.01 else 'N'

#                 for bin_label in range(12, 30):
#                     bin_key = f'Bin{bin_label}_{label}_mean'
#                     bin_value = CAS_flight[f'CAS_Bin{bin_label}'].values[i]
#                     bin_means[bin_key] = pd.to_numeric(bin_means[bin_key], errors='coerce')  # Convert to numeric, and handle errors by converting to NaN
#                     bin_means[bin_key] = np.nan_to_num(bin_means[bin_key], nan=0)
#                     bin_means[bin_key] = list(bin_means[bin_key]) + [bin_value]
#                     bin_means[bin_key].append(bin_value)
#             total_BCB_means.append(bin_means)

#     master_CAS_BCB.append(total_BCB_means)

# # Print or use master_CAS_BCB as needed
# for item in master_CAS_BCB:
#     for bin_means in item:
#         print(f"Date: {bin_means['Date']}, Start: {bin_means['BCB_start']}, Stop: {bin_means['BCB_stop']}")
#         for bin_label in range(12, 30):
#             for label in ['Y', 'N']:
#                 bin_key = f'Bin{bin_label}_{label}_mean'
#                 mean_value = np.nanmean(bin_means[bin_key])
#                 print(f"   Bin{bin_label}_{label} Mean: {mean_value}")
    
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


#%%

# def summary_bin_means(data):
#     summary_list = []

#     for flight_dict in data:
#         total_summary = sum(flight_dict.get(f'Bin{bin_label}_Y_mean', 0) for bin_label in range(12, 30))
#         summary_dict = {
#             'Date': flight_dict.get('Date', ''),
#             'BCB_start': flight_dict.get('BCB_start', ''),
#             'BCB_stop': flight_dict.get('BCB_stop', ''),
#             'Sum': total_summary
#         }
#         summary_list.append(summary_dict)

#     return summary_list

# # Apply the function to Y_BCB_calc and N_BCB_calc
# Y_sum_list = summary_bin_means(Y_BCB_calc)
# N_sum_list = sum_bin_means(N_BCB_calc)
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
            'BCB_start': flight_dict.get('BCB_start', ''),
            'BCB_stop': flight_dict.get('BCB_stop', ''),
            'Sum': total_summary
        }
        summary_list.append(summary_dict)

    return summary_list

# Apply the function to Y_BCB_calc and N_BCB_calc
Y_sum_list = summary_bin_means(Y_BCB_calc)

#%%
# def sum_bin_means(data):
#     sum_list = []

#     for flight_dict in data:
#         total_sum = sum(flight_dict.get(f'Bin{bin_label}_N_mean', 0) for bin_label in range(12, 30))
#         sum_dict = {
#             'Date': flight_dict.get('Date', ''),
#             'BCB_start': flight_dict.get('BCB_start', ''),
#             'BCB_stop': flight_dict.get('BCB_stop', ''),
#             'Sum': total_sum
#         }
#         sum_list.append(sum_dict)

#     return sum_list

# # Apply the function to Y_BCB_calc and N_BCB_calc
# N_sum_list = sum_bin_means(N_BCB_calc)
# # N_sum_list = sum_bin_means(N_BCB_calc)
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
            'BCB_start': flight_dict.get('BCB_start', ''),
            'BCB_stop': flight_dict.get('BCB_stop', ''),
            'Sum': total_sum
        }
        sum_list.append(sum_dict)

    return sum_list

# Apply the function to N_BCB_calc
N_sum_list = sum_bin_means(N_BCB_calc)


#%%

#%%
wind_mapping = {date: wind_speed for date, wind_speed in zip(corrected_calc_bcb['Date'], corrected_calc_bcb['Corrected_bcb_windspeed'])}

# Extract data for the scatterplot
dates = [entry['Date'] for entry in Y_sum_list]
sum_values = [entry['Sum'] for entry in Y_sum_list]
corrected_windspeeds = [wind_mapping.get(date, np.nan) for date in dates]

# Create the scatterplot
plt.scatter(sum_values, corrected_windspeeds)
plt.title('Below Cloud Base February 2022', fontsize=14, fontweight='bold')
plt.xlabel('Total droplet concentration dN (per cm^3) for clear skies ', fontsize=14, fontweight='bold')
plt.ylabel('Mean windspeed (m/s)', fontsize=14, fontweight='bold')
plt.xscale('log')
plt.show()
#%%
wind_mapping = {date: wind_speed for date, wind_speed in zip(corrected_calc_bcb['Date'], corrected_calc_bcb['Corrected_bcb_windspeed'])}

# Extract data for the scatterplot
dates = [entry['Date'] for entry in N_sum_list]
sum_values = [entry['Sum'] for entry in N_sum_list]
corrected_windspeeds = [wind_mapping.get(date, np.nan) for date in dates]

# Create the scatterplot
plt.scatter(sum_values, corrected_windspeeds, color='red')
plt.title('Below Cloud Base February 2022', fontsize=14, fontweight='bold')
plt.xlabel('Total droplet concentration dN (per cm^3) for cloudy skies ', fontsize=14, fontweight='bold')
plt.ylabel('Mean windspeed (m/s)', fontsize=14, fontweight='bold')
plt.xscale('log')
plt.show()
#%%
wind_mapping = {date: wind_speed for date, wind_speed in zip(corrected_calc_bcb['Date'], corrected_calc_bcb['Corrected_bcb_windspeed'])}

# Extract data for the scatterplot
dates = [entry['Date'] for entry in Y_sum_list]
sum_values = [entry['Sum'] for entry in Y_sum_list]
corrected_windspeeds = [wind_mapping.get(date, np.nan) for date in dates]

# Create the scatterplot
plt.scatter(sum_values, corrected_windspeeds)
plt.title('Below Cloud Base February 2022', fontsize=14, fontweight='bold')
plt.xlabel('Total droplet concentration dN (per cm^3) for clear skies ', fontsize=14, fontweight='bold')
plt.ylabel('Mean windspeed (m/s)', fontsize=14, fontweight='bold')
plt.show()
#%%
wind_mapping = {date: wind_speed for date, wind_speed in zip(corrected_calc_bcb['Date'], corrected_calc_bcb['Corrected_bcb_windspeed'])}

# Extract data for the scatterplot
dates = [entry['Date'] for entry in N_sum_list]
sum_values = [entry['Sum'] for entry in N_sum_list]
corrected_windspeeds = [wind_mapping.get(date, np.nan) for date in dates]

# Create the scatterplot
plt.scatter(sum_values, corrected_windspeeds, color='red')
plt.title('Below Cloud Base February 2022 ', fontsize=14, fontweight='bold')
plt.xlabel('Total droplet concentration dN (per cm^3) for cloudy skies ', fontsize=14, fontweight='bold')
plt.ylabel('Mean windspeed (m/s)', fontsize=14, fontweight='bold')
plt.show()

#%%
# %%














#%%
#This is the code to run a linear regression analysis

# %%
# Extract data specifically from Y_sum_list
dates_Y = [entry['Date'] for entry in Y_sum_list]
sum_values_Y = [entry['Sum'] for entry in Y_sum_list]
corrected_windspeeds_Y = [wind_mapping.get(date, np.nan) for date in dates_Y]

# Remove missing values from the data
valid_indices_Y = [i for i in range(len(corrected_windspeeds_Y)) if not np.isnan(corrected_windspeeds_Y[i]) and not np.isnan(sum_values_Y[i])]
X_valid_Y = [corrected_windspeeds_Y[i] for i in valid_indices_Y]
y_valid_Y = [sum_values_Y[i] for i in valid_indices_Y]

# Define the independent variable (mean wind speed) and the dependent variable (droplet concentration) for Y_sum_list only
X_Y = X_valid_Y  # Independent variable for Y_sum_list
y_Y = y_valid_Y  # Dependent variable for Y_sum_list

# Add a constant to the independent variable (required for statsmodels)
X_Y = sm.add_constant(X_Y)

# Fit the linear regression model for Y_sum_list only
model_Y = sm.OLS(y_Y, X_Y).fit()

# Print the summary of the regression model for Y_sum_list only
print(model_Y.summary())

# %%













#%%

#Time series for one leg of 2/26 where the total droplet concentration for clear skies is 
# is 11.18 /cm^3 the start time is 48862 and the stop time is 49061. The cloudy total 
#droplet concentration is 393.06318 /cm^3 for the same start and stop time. 

#This time series runs from 48862 to 49061 and is for a Below Cloud Base leg. I have 
#broken it up into cloudy vs clear and each line represents one bin. There should be 30
#lines. 

#%%
time_CAS_BCB = []
legs_info = []

# Filter data for '2022-02-26'
filtered_indices = [i for i, date in enumerate(dates_legs) if date == '2022-02-26']

# print("Processing data for date '2022-02-26'")

for i in filtered_indices:
    date = dates_legs[i]
    leg_dict = leg_data[i]

    BCB_start_list = (leg_dict['LegIndex_02']['StartTimes'])
    BCB_stop_list = (leg_dict['LegIndex_02']['StopTimes'])
    # print(f"Leg at date {date} has start time: {BCB_start}, stop time: {BCB_stop}")
    if BCB_start_list and BCB_stop_list:
        # Get the first start and stop times
        BCB_start = int(BCB_start_list[0])
        BCB_stop = int(BCB_stop_list[0])
        print(f"Leg at date {date} has start time: {BCB_start}, stop time: {BCB_stop}")

        CAS_flight = CAS[i]
        times = CAS_flight.Time_mid.values
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

        total_BCB_time = []

        # for i in range(len(BCB_start)):
        index99_start = None
        index99_end = None
            # start80 = int(BCB_start)
            # end80 = BCB_stop

        bins = {
            'Date': date,
            'BCB_start': BCB_start,
            'BCB_stop': BCB_stop,
            'Bin12_Y': [],
            'Bin13_Y': [],
            'Bin14_Y': [],
            'Bin15_Y': [],
            'Bin16_Y': [],
            'Bin17_Y': [],
            'Bin18_Y': [],
            'Bin19_Y': [],
            'Bin20_Y': [],
            'Bin21_Y': [],
            'Bin22_Y': [],
            'Bin23_Y': [],
            'Bin24_Y': [],
            'Bin25_Y': [],
            'Bin26_Y': [],
            'Bin27_Y': [],
            'Bin28_Y': [],
            'Bin29_Y': [],
            'Bin12_N': [],
            'Bin13_N': [],
            'Bin14_N': [],
            'Bin15_N': [],
            'Bin16_N': [],
            'Bin17_N': [],
            'Bin18_N': [],
            'Bin19_N': [],
            'Bin20_N': [],
            'Bin21_N': [],
            'Bin22_N': [],
            'Bin23_N': [],
            'Bin24_N': [],
            'Bin25_N': [],
            'Bin26_N': [],
            'Bin27_N': [],
            'Bin28_N': [],
            'Bin29_N': [],
        }
        for idx, t in enumerate(times):
            if BCB_start <= int(t) <= BCB_stop:
                lwc_val = lwc[idx]
                for bin_label in range(12, 30):
                    bin_key = f'Bin{bin_label}_{"Y" if 0 <= lwc_val <= 0.01 else "N"}'
                    bin_value = CAS_flight[f'CAS_Bin{bin_label}'].values[idx]
                    bins[bin_key].append(bin_value)
    
        total_BCB_time.append(bins)
        break
            

time_CAS_BCB.append(total_BCB_time)
# %%

# Create separate lists to store Y and N values for each bin
Y_values = [[] for _ in range(12, 30)]
N_values = [[] for _ in range(12, 30)]

# Iterate over the data in time_CAS_BCB
for bins_data in time_CAS_BCB:
    for bins in bins_data:
        # Extract data for each bin and append to respective lists
        for bin_label in range(12, 30):
            bin_key_Y = f'Bin{bin_label}_Y'
            bin_key_N = f'Bin{bin_label}_N'
            Y_values[bin_label - 12].extend(bins[bin_key_Y])
            N_values[bin_label - 12].extend(bins[bin_key_N])
    break  # Exit after processing the first entry

# Find the maximum length among Y and N lists
max_length = max(max(len(lst) for lst in Y_values), max(len(lst) for lst in N_values))

# Pad lists with NaN values to make them equal length
for lst in Y_values + N_values:
    while len(lst) < max_length:
        lst.append(np.nan)

# Create x-axis values from BCB_start to BCB_stop
x_values = list(range(BCB_start, BCB_start + max_length))

# Plotting Y values
plt.figure(figsize=(10, 6))
for bin_label, Y_data in enumerate(Y_values, start=12):
    plt.plot(x_values, Y_data, label=f'Bin{bin_label}_Y')

plt.xlabel('Seconds after midnight',  fontsize=16, fontweight='bold')
plt.ylabel('Droplet concentration (dN/dlogd)',  fontsize=16, fontweight='bold')
plt.legend(fontsize=4.5)
plt.grid(True)
plt.yscale('log')
plt.xlim(BCB_start, BCB_stop)  # Set the limits of the x-axis

plt.title('2022-02-26\n 48862 - 49061 seconds after midnight\n Clear sky', fontsize=16, fontweight='bold')
plt.show()

# # Plotting N values
# plt.figure(figsize=(10, 6))
# for bin_label, N_data in enumerate(N_values, start=12):
#     plt.plot(x_values, N_data, label=f'Bin{bin_label}_N')
# plt.xlim(BCB_start, BCB_stop)  # Set the limits of the x-axis

# plt.xlabel('Seconds after midnight',  fontsize=16, fontweight='bold')
# plt.ylabel('Droplet concentration (dN/dlogd)',  fontsize=16, fontweight='bold')
# plt.title('2022-02-26\n 48862 - 49061 seconds after midnight\n Cloudy sky', fontsize=16, fontweight='bold')
# plt.legend(fontsize=4)
# plt.grid(True)
# plt.show()


# %%

























#Time series for one leg of 2/26 where the liquid water content for clear skies is 
# is plotted against the start time 48862 and the stop time 49061.  
#This time series runs from 48862 to 49061 and is for a Below Cloud Base leg. 
#%%
lwc_CAS_BCB = []


# Filter data for '2022-02-26'
filtered_indices = [i for i, date in enumerate(dates_legs) if date == '2022-02-26']

# print("Processing data for date '2022-02-26'")

for i in filtered_indices:
    date = dates_legs[i]
    leg_dict = leg_data[i]

    BCB_start_list = (leg_dict['LegIndex_02']['StartTimes'])
    BCB_stop_list = (leg_dict['LegIndex_02']['StopTimes'])
    # print(f"Leg at date {date} has start time: {BCB_start}, stop time: {BCB_stop}")
    if BCB_start_list and BCB_stop_list:
        # Get the first start and stop times
        BCB_start = int(BCB_start_list[0])
        BCB_stop = int(BCB_stop_list[0])
        print(f"Leg at date {date} has start time: {BCB_start}, stop time: {BCB_stop}")

        CAS_flight = CAS[i]
        times = CAS_flight.Time_mid.values
        lwc = CAS_flight.LWC_CAS.values

        LWC_Y = []
        LWC_N=[]
        # index99_start = None
        # index99_end = None
           

        # lwc = {
        #     'Date': date,
        #     'BCB_start': BCB_start,
        #     'BCB_stop': BCB_stop,
        #     'LWC_Y': [],
        #     'LWC_N': [],
        # }
        for idx, t in enumerate(times):
            if BCB_start <= int(t) <= BCB_stop:
                lwc_val = lwc[idx]
                if 0 <= lwc_val <= 0.01:
                    LWC_Y.append(lwc_val)
                else:
                    LWC_N.append(lwc_val)
    
        lwc_CAS_BCB.append({'Date': date, 'LWC_Y':LWC_Y, 'LWC_N': LWC_N})
        break
            


# %%

lwc_Y_values = []


for entry in lwc_CAS_BCB:
    lwc_Y_values.extend(entry['LWC_Y'])

# Find the maximum length among lwc_Y list
max_length = len(lwc_Y_values)

# Create x-axis values from BCB_start to BCB_stop
x_values = list(range(BCB_start, BCB_start + max_length))

# Plotting lwc_Y values
plt.figure(figsize=(10, 6))
plt.plot(x_values, lwc_Y_values, label='CAS_LWC g/m^3', color='blue')

plt.xlabel('Seconds after midnight', fontsize=16, fontweight='bold')
plt.ylabel('Clear LWC g/m^3', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True)
# plt.yscale('log')
plt.xlim(BCB_start, BCB_stop)  # Set the limits of the x-axis

plt.title('2022-02-26\n 48862 - 49061 seconds after midnight\n Clear sky', fontsize=16, fontweight='bold')
plt.show()

# %%
























#Time series for one leg of 2/26 where the liquid water content for clear skies is 
# is plotted against the start time 48862 and the stop time 49061.  
#This time series runs from 48862 to 49061 and is for a Below Cloud Base leg.I plotted both the 
#given lwc values found in the CAS in blue with my own hand-calculated LWC values in red.  

#%%

C_0=math.log10(0.61)-math.log10(0.5)
C_01=math.log10(0.68)-math.log10(0.61)
C_02=math.log10(0.75)-math.log10(0.68)
C_03=math.log10(0.82)-math.log10(0.75)
C_04=math.log10(0.89)-math.log10(0.82)
C_05=math.log10(0.96)-math.log10(0.89)
C_06=math.log10(1.03)-math.log10(0.96)
C_07=math.log10(1.1)-math.log10(1.03)
C_08=math.log10(1.17)-math.log10(1.1)
C_09=math.log10(1.25)-math.log10(1.17)
C_10=math.log10(1.5)-math.log10(1.25)
C_11=math.log10(2)-math.log10(1.5)
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
#%%
    
#%%
# bin_center=[0.555, 0.645, 0.715, 0.785, 0.855, 0.925, 
#             0.995, 1.07, 1.14, 1.21, 1.38, 1.75, 2.25, 
#             2.75, 3.25, 3.75, 4.5, 5.75, 6.85, 7.55, 
#             9.05, 11.4, 13.8, 17.5, 22.5, 27.5, 32.5, 
#             37.5, 42.5, 47.5]
bin_center_radius=[0.2775, 0.3225, 0.3575, 0.3925,
                    0.4275, 0.4625, 0.4975, 0.535, 0.57, 0.605,
                      0.69, 0.875, 1.125, 1.375, 1.625, 1.875, 
                      2.25, 2.875, 3.425, 3.775, 4.525, 5.7, 
                      6.9, 8.75, 11.25, 13.75, 16.25, 18.75, 
                      21.25, 23.75]

#%%

# #%%
# B_00=((0.2775)**3) / (10000)
# B_01= ((0.3225)**3) / (10000)
# B_02=((0.3575)**3)/ (10000)
# B_03=((0.3925)**3)/  (10000)
# B_04=((0.4275)**3) / (10000)
# B_05=((0.4625,)**3)/ (10000)
# B_06=((0.4975)**3) / (10000)
# #%%
# B_07=((0.535)**3)/(10000)
# B_08=((0.57)**3)/(10000)
# B_09=((0.605)**3)/(10000)
# B_10=((0.69)**3)/(10000)
# B_11=((0.875)**3)/(10000)
# B_12=((1.125)**3)/(10000)
# B_13=((1.375)**3)/(10000)
# B_14=((1.625)**3)/(10000)
# B_15=((1.875)**3)/(10000)
# #%%
# B_16=((2.25)**3)/(10000)
# B_17=((2.875)**3)/(10000)
# B_18=((3.425)**3)/(10000)
# B_19=((3.775)**3)/(10000)
# B_20=((4.525)**3)/(10000)
# B_21=((5.7)**3)/(10000)
# B_22=((6.9)**3)/(10000)
# B_23=((8.75)**3)/(10000)
# B_24=((11.25)**3)/(10000)
# #%%
# B_25=((13.75)**3) /(10000)
# B_26=((16.25) **3)/ (10000)
# B_27=( (18.75)**3)/ (10000)
# B_28=((21.25) **3)/ (10000)
# B_29=(( 23.75)**3)/ (10000)

# radius=[B_00, B_01, B_02, B_03, B_04, B_05, B_06, B_07, B_08,
#         B_09, B_10, B_11, B_12, B_13, B_14, B_15, B_16, B_17, B_18,
#         B_19,B_20, B_21,B_22, B_23, B_24, B_25, B_26, B_27, B_28,
#         B_29]
#%%
C_values=[C_0, C_01, C_02, C_03, C_04, C_05, C_06, C_07, C_08, C_09, 
         C_10, C_11, C_12, C_13, C_14, C_15, C_16, C_17, C_18, 
         C_19, C_20, C_21, C_22, C_23, C_24, C_25, C_26, C_27, C_28, C_29]
# %%
# N_00=123.6
# N_01=444.6
# N_02=182.8
# N_03=72.92
# N_04=37.38
# N_05=11.12
# N_06=13.05
# N_07=10.48
# N_08=8.686
# N_09=12.73
# N_10=7.977
# N_11=1.863
# N_12=1.715
# N_13=0.8397
# N_14=1.49
# N_15=1.72
# N_16=2.058
# N_17=0.8753
# N_18=0
# N_19=0
# N_20=0.2996
# N_21=0
# N_22=0
# N_23=0
# N_24=0
# N_25=0
# N_26=0
# N_27=0
# N_28=0
# N_29=0
# %%

# A_0 =(N_00*C_0) /(100)
# A_01 =(N_01*C_01) /(100)
# A_02 =(N_02*C_02) /(100)
# A_03 =(N_03*C_03) /(100)
# A_04 =(N_04*C_04) /(100)
# A_05 =(N_05*C_05) /(100)
# A_06 =(N_06*C_06)/(100)
# A_07 =(N_07*C_07 )/(100)
# A_08 =(N_08*C_08)/(100)
# A_09 =(N_09*C_09)/(100)
# A_10 =(N_10*C_10) /(100)
# A_11 =(N_11*C_11) /(100)
# A_12 =(N_12*C_12) /(100)
# A_13 =(N_13*C_13) /(100)
# A_14 =(N_14*C_14) /(100)
# A_15 =(N_15*C_15) /(100)
# A_16 =(N_16*C_16) /(100)
# A_17 =(N_17*C_17) /(100)
# A_18 =(N_18*C_18) /(100)
# A_19 =(N_19*C_19) /(100)
# A_20 =(N_20*C_20) /(100)
# A_21 =(N_21*C_21) /(100)
# A_22 =(N_22*C_22) /(100)
# A_22 =(N_22*C_22) /(100)
# A_23 =(N_23*C_23) /(100)
# A_24 =(N_24*C_24) /(100)
# A_25 =(N_25*C_25) /(100)
# A_26 =(N_26*C_26) /(100)
# A_27 =(N_27*C_27) /(100)
# A_28 =(N_28*C_28) /(100)
# A_29 =(N_29*C_29) /(100)



#%%
# %%
# F_0 = (((4/3) * 3.14) * (B_00) * (A_0) ) 
# F_01 = ((A_01) * ((4/3) * 3.14) * (B_01))
# F_02 = ((A_02) * ((4/3) * 3.14) * (B_02)) 
# F_03 = ((A_03) * ((4/3) * 3.14)* (B_03)) 
# F_04 = ((A_04) * ((4/3) * 3.14) * (B_04)) 
# F_05 = ((A_05) * ((4/3) * 3.14) * (B_05)) 
# F_06 = ((A_06) * ((4/3) * 3.14) * (B_06)) 
# F_07 = ((A_07) * ((4/3) * 3.14) * (B_07)) 
# F_08 = ((A_08) * ((4/3) * 3.14) * (B_08)) 
# F_09 = ((A_09) * ((4/3) * 3.14)* (B_09)) 
# F_10 = ((A_10) * ((4/3) * 3.14)* (B_10)) 
# F_11 = ((A_11) * ((4/3) * 3.14)* (B_11)) 
# F_12 = ((A_12) * ((4/3) * 3.14) * (B_12)) 
# F_13 = ((A_13) * ((4/3) * 3.14) * (B_13)) 
# F_14 = ((A_14) * ((4/3) * 3.14) * (B_14)) 
# F_15 = ((A_15) * ((4/3) * 3.14) * (B_15))
# F_16 = ((A_16)* ((4/3) * 3.14) * (B_16)) 
# F_17 = ((A_17)* ((4/3) * 3.14) * (B_17)) 
# F_18 = ((A_18) * ((4/3) * 3.14) * (B_18)) 
# F_19 = ((A_19) * ((4/3) * 3.14) * (B_19)) 
# F_20 = ((A_20) * ((4/3) * 3.14) * (B_20)) 
# F_21 = ((A_21) * ((4/3) * 3.14) * (B_21))
# F_22 = ((A_22) * ((4/3) * 3.14) * (B_22))
# F_23 = ((A_23) * ((4/3) * 3.14) * (B_23))
# F_24 = ((A_24) * ((4/3) * 3.14) * (B_24))
# F_25 = ((A_25) * ((4/3) * 3.14) * (B_25))
# F_26 = ((A_26) * ((4/3) * 3.14) * (B_26))
# F_27 = ((A_27) * ((4/3) * 3.14) * (B_27))
# F_28 = ((A_28) * ((4/3) * 3.14) * (B_28))
# F_29 = ((A_29) * ((4/3) * 3.14) * (B_29))
# # %%
# LWC=(F_0+F_01+F_02+F_03+F_04+F_05+F_06+F_07+F_08+F_09+F_10+F_11+F_12+F_13+
#      F_14+F_15+F_16+F_17+F_18+F_19+F_20+F_21+F_22+F_23+F_24+F_25+F_26+
#      F_27+F_28+F_29)
# print(LWC)

# %%
#Convert radius to cm3 to cancel out with 1 g/cm3 water density and convert
#concentration to m3 to get lwc in g/m3. 

filtered_indices = [i for i, date in enumerate(dates_legs) if date == '2022-02-26']

# print("Processing data for date '2022-02-26'")

for i in filtered_indices:
    date = dates_legs[i]
    leg_dict = leg_data[i]

    BCB_start_list = (leg_dict['LegIndex_02']['StartTimes'])
    BCB_stop_list = (leg_dict['LegIndex_02']['StopTimes'])
    # print(f"Leg at date {date} has start time: {BCB_start}, stop time: {BCB_stop}")
    if BCB_start_list and BCB_stop_list:
        # Get the first start and stop times
        BCB_start = int(BCB_start_list[0])
        BCB_stop = int(BCB_stop_list[0])
        print(f"Leg at date {date} has start time: {BCB_start}, stop time: {BCB_stop}")
#%%
time_CAS_BCB_full = []
legs_info_full = []

# Filter data for '2022-02-26'
filtered_indices = [i for i, date in enumerate(dates_legs) if date == '2022-02-26']

# print("Processing data for date '2022-02-26'")

for i in filtered_indices:
    date = dates_legs[i]
    leg_dict = leg_data[i]

    BCB_start_list = (leg_dict['LegIndex_02']['StartTimes'])
    BCB_stop_list = (leg_dict['LegIndex_02']['StopTimes'])
    # print(f"Leg at date {date} has start time: {BCB_start}, stop time: {BCB_stop}")
    if BCB_start_list and BCB_stop_list:
        # Get the first start and stop times
        BCB_start = int(BCB_start_list[0])
        BCB_stop = int(BCB_stop_list[0])
        print(f"Leg at date {date} has start time: {BCB_start}, stop time: {BCB_stop}")

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

        total_BCB_time_full = []

        # for i in range(len(BCB_start)):
        index99_start = None
        index99_end = None
            # start80 = int(BCB_start)
            # end80 = BCB_stop

        bins_full = {
            'Date': date,
            'BCB_start': BCB_start,
            'BCB_stop': BCB_stop,
            'Bin00_Y': [],
            'Bin01_Y': [],
            'Bin02_Y': [],
            'Bin03_Y': [],
            'Bin04_Y': [],
            'Bin05_Y': [],
            'Bin06_Y': [],
            'Bin07_Y': [],
            'Bin08_Y': [],
            'Bin09_Y': [],
            'Bin10_Y': [],
            'Bin11_Y': [],
            'Bin12_Y': [],
            'Bin13_Y': [],
            'Bin14_Y': [],
            'Bin15_Y': [],
            'Bin16_Y': [],
            'Bin17_Y': [],
            'Bin18_Y': [],
            'Bin19_Y': [],
            'Bin20_Y': [],
            'Bin21_Y': [],
            'Bin22_Y': [],
            'Bin23_Y': [],
            'Bin24_Y': [],
            'Bin25_Y': [],
            'Bin26_Y': [],
            'Bin27_Y': [],
            'Bin28_Y': [],
            'Bin29_Y': [],
            'Bin00_N': [],
            'Bin01_N': [],
            'Bin02_N': [],
            'Bin03_N': [],
            'Bin04_N': [],
            'Bin05_N': [],
            'Bin06_N': [],
            'Bin07_N': [],
            'Bin08_N': [],
            'Bin09_N': [],
            'Bin10_N': [],
            'Bin11_N': [],
            'Bin12_N': [],
            'Bin13_N': [],
            'Bin14_N': [],
            'Bin15_N': [],
            'Bin16_N': [],
            'Bin17_N': [],
            'Bin18_N': [],
            'Bin19_N': [],
            'Bin20_N': [],
            'Bin21_N': [],
            'Bin22_N': [],
            'Bin23_N': [],
            'Bin24_N': [],
            'Bin25_N': [],
            'Bin26_N': [],
            'Bin27_N': [],
            'Bin28_N': [],
            'Bin29_N': [],
        }
        for idx, t in enumerate(times):
            if BCB_start <= int(t) <= BCB_stop:
                lwc_val = lwc[idx]
                for bin_label in range(30):
                    bin_key = f'Bin{bin_label:02d}_{"Y" if 0 <= lwc_val <= 0.01 else "N"}'
                    bin_value = CAS_flight[f'CAS_Bin{bin_label:02d}'].values[idx]
                    bins_full[bin_key].append(bin_value)

        total_BCB_time_full.append(bins_full)
        break

time_CAS_BCB_full.append(total_BCB_time_full)

# %%

time_CAS_BCB_full_modified = []

# Iterate over each dictionary (which is actually a list) in time_CAS_BCB_full
for data_list in time_CAS_BCB_full:
    modified_dict = {}
    # Iterate over each dictionary in the list
    for dictionary in data_list:
        # Iterate over the keys in the current dictionary
        for key, value in dictionary.items():
            # Check if the key ends with '_Y'
            if key.endswith('_Y'):
                # If it does, add it to the modified dictionary
                modified_dict[key] = value
    # Append the modified dictionary to time_CAS_BCB_full_modified
    time_CAS_BCB_full_modified.append(modified_dict)

# Now time_CAS_BCB_full_modified contains dictionaries with keys ending in '_Y'

#%%
# Loop through each dictionary in total_BCB_time_full
for index_data in time_CAS_BCB_full_modified:
    # Loop through each key-value pair in the dictionary
    for key, value in index_data.items():
        # Check if the value is nan
        if isinstance(value, float) and math.isnan(value):
            # Replace nan with 0
            index_data[key] = 0

#%%


#%%


#%%
# LWC_list = []

# # Loop through each index in total_BCB_time_full
# for index_data in time_CAS_BCB_full_modified:
#     LWC_row = []  # Initialize list to store F values for the current row
#     for idx in range(len(index_data['Bin00_Y'])):  # Assuming 'Bin00_Y' has the same length as other bins
#         F_sum = 0  # Initialize sum of F values for the current row
#         for bin_label in range(30):
#             # Get the N value and corresponding bin center value
#             N_value = index_data[f'Bin{bin_label:02d}_Y'][idx]
#             bin_center_value = bin_center_radius[bin_label]
            
#             # Calculate A and F for each bin
#             A = (N_value * C_values[bin_label]) / 1e6
#             B = (bin_center_value ** 3) / 1e12
#             F = A * ((4 / 3) * 3.14) * B
            
#             # Add F to the sum for the current row
#             F_sum += F
            
#             # Print individual F values for each bin
#         #     print(f"F for Bin{bin_label:02d}: {F}")
        
#         # # Print the sum of F values for the current row
#         # print(f"Sum of F values for row {idx}: {F_sum}")
        
#         # Append the sum of F values to the list for the current row
#         LWC_row.append(F_sum)
    
#     # Append the list of F values for the current row to the LWC list
#     LWC_list.append(LWC_row)

# # print(LWC_list)
# #%%
# import numpy as np  # Import numpy for NaN handling

# LWC_list = []

# # Loop through each index in total_BCB_time_full
# for index_data in time_CAS_BCB_full_modified:
#     LWC_row = []  # Initialize list to store F values for the current row
#     for idx in range(len(index_data['Bin00_Y'])):  # Assuming 'Bin00_Y' has the same length as other bins
#         F_sum = 0  # Initialize sum of F values for the current row
#         for bin_label in range(30):
#             # Get the N value and corresponding bin center value
#             N_value = index_data[f'Bin{bin_label:02d}_Y'][idx]
#             bin_center_value = bin_center_radius[bin_label]
            
#             # Calculate A and F for each bin
#             A = (N_value * C_values[bin_label]) / 1e6
#             B = (bin_center_value ** 3) / 1e12
#             F = A * ((4 / 3) * 3.14) * B
            
#             # Check if F is NaN and replace it with 0
#             if np.isnan(F):
#                 F = 0
            
#             # Add F to the sum for the current row
#             F_sum += F
            
#             # Print individual F values for each bin
#         #     print(f"F for Bin{bin_label:02d}: {F}")
        
#         # # Print the sum of F values for the current row
#         # print(f"Sum of F values for row {idx}: {F_sum}")
        
#         # Append the sum of F values to the list for the current row
#         LWC_row.append(F_sum)
    
#     # Append the list of F values for the current row to the LWC list
#     LWC_list.append(LWC_row)

# print(LWC_list)
#%%



LWC_list = []
sums_list = []  # Initialize list to store sum values for each row

# Loop through each index in total_BCB_time_full
for index_data in time_CAS_BCB_full_modified:
    LWC_row = []  # Initialize list to store F values for the current row
    for idx in range(len(index_data['Bin00_Y'])):  # Assuming 'Bin00_Y' has the same length as other bins
        F_sum = 0  # Initialize sum of F values for the current row
        for bin_label in range(30):
            # Get the N value and corresponding bin center value
            N_value = index_data[f'Bin{bin_label:02d}_Y'][idx]
            bin_center_value = bin_center_radius[bin_label]
            
            # Calculate A and F for each bin
            A = (N_value * C_values[bin_label])/1e6
            B = (bin_center_value ** 3) / 1e12
            F = A * ((4 / 3) * 3.14) * B * 1
            
            # Check if F is NaN and replace it with 0
            if np.isnan(F):
                F = 0
            
            # Add F to the sum for the current row
            F_sum += F
        
        # Append the sum of F values to the list for the current row
        LWC_row.append(F_sum)
        
        # Append the sum to the sums_list
        sums_list.append(F_sum)
    
    # Append the list of F values for the current row to the LWC list
    LWC_list.append(LWC_row)

print(LWC_list)
print(sums_list)

# %%

# Your existing code for plotting lwc_Y values
plt.figure(figsize=(10, 6))
plt.plot(sums_list, lwc_Y_values, label='LWC g/m^3', color='blue')
plt.xlabel('Calculated lwc g/m^3', fontsize=16, fontweight='bold')
plt.ylabel('CAS_LWC', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
# plt.yscale('log')
# plt.xscale('log')
plt.grid(True)
# plt.xlim(BCB_start, BCB_stop)
plt.title('2022-02-26\n 48862 - 49061 seconds after midnight\n Clear sky', fontsize=16, fontweight='bold')


# %%
x_values = list(range(BCB_start, BCB_start + max_length))
plt.figure(figsize=(10, 6))
plt.plot(x_values, sums_list, label='Calculated_LWC g/m^3', color='red')
plt.xlabel('Seconds after midnight', fontsize=16, fontweight='bold')
plt.ylabel('Calculated LWC g/m^3', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True)
plt.xlim(BCB_start, BCB_stop)
plt.title('2022-02-26\n 48862 - 49061 seconds after midnight\n Clear sky: Below Cloud Base', fontsize=16, fontweight='bold')




#%%
plt.figure(figsize=(10, 6))
plt.plot(x_values, lwc_Y_values, label='CAS_LWC', color='blue')
plt.xlabel('Seconds after midnight', fontsize=16, fontweight='bold')
plt.ylabel('LWC g/m^3', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True)
plt.xlim(BCB_start, BCB_stop)
plt.title('2022-02-26\n 48862 - 49061 seconds after midnight\n Clear sky: Below Cloud Base', fontsize=16, fontweight='bold')

# Plotting sums_list on the same graph
plt.plot(x_values, sums_list, label='Calculated LWC', color='red')  # Add this line
plt.legend(fontsize=12)  # Add this line to display legend for the new plot

plt.show()

# %%
