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
#%%
C_12=math.log10(4.5)-math.log10(3)
C_13=math.log10(6.0)-math.log10(4.5)
C_14=math.log10(8)-math.log10(6)
C_15=math.log10(10)-math.log10(8)
C_16=math.log10(12)-math.log10(10)
C_17=math.log10(14)-math.log10(12)
C_18=math.log10(16)-math.log10(14)
C_19=math.log10(18)-math.log10(16)
C_20=math.log10(21)-math.log10(18)
C_21=math.log10(24)-math.log10(21)
C_22=math.log10(27)-math.log10(24)
C_23=math.log10(30)-math.log10(27)
C_24=math.log10(33)-math.log10(30)
C_25=math.log10(36)-math.log10(33)
C_26=math.log10(39)-math.log10(36)
C_27=math.log10(42)-math.log10(39)
C_28=math.log10(46)-math.log10(42)
C_29=math.log10(50)-math.log10(46)
# %%
bin_log=[C_12, C_13, C_14, C_15, C_16, 
        C_17, C_18, C_19, C_20, C_21, C_22, C_23, C_24, C_25, C_26, C_27, C_28, C_29]
# %%
bin_center_diameter=[3.75, 5.25, 7, 9, 11, 13, 15, 17, 19.5, 
            22.5, 25.5, 28.5, 31.5, 34.5, 37.5, 40.5 ,44,48]
# %%
D12 = (4.5-3)
D13 = (6-4.5)
D14 = (8-6)
D15 = (10-8)
D16 = (12-10)
D17 = (14-12)
D18 = (16-14)
D19 = (18-16)
D20 = (21-18)
D21 = (24-21)
D22 = (27-24)
D23 = (30-27)
D24 = (33-30)
D25 = (36-33)
D26 = (39-36)
D27 = (42-39)
D28 = (46-42)
D29 = (50-46)


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
# %%
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
#FCDP data

bin_name = ['dNdlogD_003_FCDP', 'dNdlogD_004_FCDP' ,'dNdlogD_005_FCDP', 'dNdlogD_006_FCDP',
            'dNdlogD_007_FCDP', 'dNdlogD_008_FCDP', 'dNdlogD_009_FCDP', 'dNdlogD_010_FCDP' ,
            'dNdlogD_011_FCDP', 'dNdlogD_012_FCDP' , 'dNdlogD_013_FCDP',
            'dNdlogD_014_FCDP','dNdlogD_015_FCDP','dNdlogD_016_FCDP',
            'dNdlogD_017_FCDP', 'dNdlogD_018_FCDP','dNdlogD_019_FCDP',
            'dNdlogD_020_FCDP']


FCDP = []

dates_FCDP = ['2022-02-01', '2022-02-02', '2022-02-03', '2022-02-05', 
             '2022-02-15', '2022-02-16', '2022-02-19', '2022-02-22',
              '2022-02-26']

for date in dates_FCDP:
    datestr = date.replace('-', '')
    fname_FCDP = sorted(glob.glob(f'/home/disk/eos4/kathem24/activate/data/FCDP/2022/csv/ACTIVATE-FCDP_HU25_{datestr}_R*.csv'), reverse=True)
    print(date)
    print(fname_FCDP)

    run = 1
    for file_path in fname_FCDP:
        nums_file_paths = len(fname_FCDP)

        if date <= ('2022-02-26'):
            df_FCDP = pd.read_csv(file_path, skiprows= 57, quoting=csv.QUOTE_NONE)

        for bin_ in bin_name:
            if bin_ in df_FCDP.columns:
                df_FCDP.columns = df_FCDP.columns.str.strip('"')
                df_FCDP[bin_] = pd.to_numeric(df_FCDP[bin_], errors='coerce')
                df_FCDP.replace([-9999, -9999.00], np.NaN, inplace=True)
        for col in ['dNdlogD_003_FCDP', 'dNdlogD_004_FCDP' ,'dNdlogD_005_FCDP', 'dNdlogD_006_FCDP',
            'dNdlogD_007_FCDP', 'dNdlogD_008_FCDP', 'dNdlogD_009_FCDP', 'dNdlogD_010_FCDP' ,
            'dNdlogD_011_FCDP', 'dNdlogD_012_FCDP' , 'dNdlogD_013_FCDP',
            'dNdlogD_014_FCDP','dNdlogD_015_FCDP','dNdlogD_016_FCDP',
            'dNdlogD_017_FCDP', 'dNdlogD_018_FCDP','dNdlogD_019_FCDP',
            'dNdlogD_020_FCDP', 'Time_Start',
            'N_FCDP', 'LWC_FCDP','ED_FCDP','MVD_FCDP']:
            if df_FCDP[col].dtype == 'O':  # 'O' stands for Object (usually string columns)
                df_FCDP[col] = df_FCDP[col].str.strip('"')
        
        df_FCDP['Time_Start']= pd.to_numeric(df_FCDP['Time_Start'], errors='coerce')
        df_FCDP['dNdlogD_003_FCDP']= pd.to_numeric(df_FCDP['dNdlogD_003_FCDP'], errors='coerce')
        df_FCDP['dNdlogD_004_FCDP']= pd.to_numeric(df_FCDP['dNdlogD_004_FCDP'], errors='coerce')
        df_FCDP['dNdlogD_005_FCDP']= pd.to_numeric(df_FCDP['dNdlogD_005_FCDP'], errors='coerce')
        df_FCDP['dNdlogD_006_FCDP']= pd.to_numeric(df_FCDP['dNdlogD_006_FCDP'], errors='coerce')
        df_FCDP['dNdlogD_007_FCDP']= pd.to_numeric(df_FCDP['dNdlogD_007_FCDP'], errors='coerce')
        df_FCDP['dNdlogD_008_FCDP']= pd.to_numeric(df_FCDP['dNdlogD_008_FCDP'], errors='coerce')
        df_FCDP['dNdlogD_009_FCDP']= pd.to_numeric(df_FCDP['dNdlogD_009_FCDP'], errors='coerce')
        df_FCDP['dNdlogD_010_FCDP']= pd.to_numeric(df_FCDP['dNdlogD_010_FCDP'], errors='coerce')
        df_FCDP['dNdlogD_011_FCDP']= pd.to_numeric(df_FCDP['dNdlogD_011_FCDP'], errors='coerce')
        df_FCDP['dNdlogD_012_FCDP']= pd.to_numeric(df_FCDP['dNdlogD_012_FCDP'], errors='coerce')
        df_FCDP['dNdlogD_013_FCDP']= pd.to_numeric(df_FCDP['dNdlogD_013_FCDP'], errors='coerce')
        df_FCDP['dNdlogD_014_FCDP']= pd.to_numeric(df_FCDP['dNdlogD_014_FCDP'], errors='coerce')
        df_FCDP['dNdlogD_015_FCDP']= pd.to_numeric(df_FCDP['dNdlogD_015_FCDP'], errors='coerce')
        df_FCDP['dNdlogD_016_FCDP']= pd.to_numeric(df_FCDP['dNdlogD_016_FCDP'], errors='coerce')
        df_FCDP['dNdlogD_017_FCDP']= pd.to_numeric(df_FCDP['dNdlogD_017_FCDP'], errors='coerce')
        df_FCDP['dNdlogD_018_FCDP']= pd.to_numeric(df_FCDP['dNdlogD_018_FCDP'], errors='coerce')
        df_FCDP['dNdlogD_019_FCDP']= pd.to_numeric(df_FCDP['dNdlogD_019_FCDP'], errors='coerce')
        df_FCDP['dNdlogD_020_FCDP']= pd.to_numeric(df_FCDP['dNdlogD_020_FCDP'], errors='coerce')
        df_FCDP['LWC_FCDP']=pd.to_numeric(df_FCDP['LWC_FCDP'], errors='coerce')
        

        if nums_file_paths==2:
            if run==1:
                df4 = df_FCDP 
            elif run==2:
                df5 = df_FCDP 
                frames = [df5,df4]
                df_FCDP = pd.concat(frames)
                FCDP.append(df_FCDP)
                break

        if nums_file_paths ==1:
            print("YESSSSSSSSSSSSSSSSSSSSSSSSSSSSs")
            FCDP.append(df_FCDP)

        run = run+1

        if date == '2022-02-26':
                print(f"\nFirst row for date {'2022-02-26'}:")
        
        print(date)
        time = df_FCDP['Time_Start']
        print("yes")
# %%
master_FCDP_BCB = []
leg_info_FCDP = []

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

    FCDP_flight = FCDP[i]
    times = FCDP_flight.Time_Start.values
    Bin3 = FCDP_flight.dNdlogD_003_FCDP.values
    Bin4 = FCDP_flight.dNdlogD_004_FCDP.values
    Bin5 = FCDP_flight.dNdlogD_005_FCDP.values
    Bin6 = FCDP_flight.dNdlogD_006_FCDP.values
    Bin7 = FCDP_flight.dNdlogD_007_FCDP.values
    Bin8 = FCDP_flight.dNdlogD_008_FCDP.values
    Bin9 = FCDP_flight.dNdlogD_009_FCDP.values
    Bin10= FCDP_flight.dNdlogD_010_FCDP.values
    Bin11 = FCDP_flight.dNdlogD_011_FCDP.values
    Bin12 =FCDP_flight.dNdlogD_012_FCDP.values
    Bin13 = FCDP_flight.dNdlogD_013_FCDP.values
    Bin14 = FCDP_flight.dNdlogD_014_FCDP.values
    Bin15 = FCDP_flight.dNdlogD_015_FCDP.values
    Bin16 = FCDP_flight.dNdlogD_016_FCDP.values
    Bin17 =FCDP_flight.dNdlogD_017_FCDP.values
    Bin18 = FCDP_flight.dNdlogD_018_FCDP.values
    Bin19 = FCDP_flight.dNdlogD_019_FCDP.values
    Bin20 = FCDP_flight.dNdlogD_020_FCDP.values
    lwc = FCDP_flight.LWC_FCDP.values

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
                label = 'Y' if 0 <= lwc_val <= 0.000001 else 'N'
                lwc_labels.append(label)

            leg_info_FCDP.append({
                'Date': date,
                'BCB_start': start20,
                'BCB_stop': end20,
                'LWC_Labels': lwc_labels,
            })

    master_FCDP_BCB.append(total_BCB_means)

# Print leg_info or use it as needed
for leg in leg_info_FCDP:
    print(f"Date: {leg['Date']}, Start: {leg['BCB_start']}, Stop: {leg['BCB_stop']}, LWC Labels: {leg['LWC_Labels']}")

# %%
master_FCDP_BCB = []
leg_info_FCDP = []

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

    FCDP_flight = FCDP[i]
    times = FCDP_flight.Time_Start.values
    Bin3 = FCDP_flight.dNdlogD_003_FCDP.values
    Bin4 = FCDP_flight.dNdlogD_004_FCDP.values
    Bin5 = FCDP_flight.dNdlogD_005_FCDP.values
    Bin6 = FCDP_flight.dNdlogD_006_FCDP.values
    Bin7 = FCDP_flight.dNdlogD_007_FCDP.values
    Bin8 = FCDP_flight.dNdlogD_008_FCDP.values
    Bin9 = FCDP_flight.dNdlogD_009_FCDP.values
    Bin10= FCDP_flight.dNdlogD_010_FCDP.values
    Bin11 = FCDP_flight.dNdlogD_011_FCDP.values
    Bin12 =FCDP_flight.dNdlogD_012_FCDP.values
    Bin13 = FCDP_flight.dNdlogD_013_FCDP.values
    Bin14 = FCDP_flight.dNdlogD_014_FCDP.values
    Bin15 = FCDP_flight.dNdlogD_015_FCDP.values
    Bin16 = FCDP_flight.dNdlogD_016_FCDP.values
    Bin17 =FCDP_flight.dNdlogD_017_FCDP.values
    Bin18 = FCDP_flight.dNdlogD_018_FCDP.values
    Bin19 = FCDP_flight.dNdlogD_019_FCDP.values
    Bin20 = FCDP_flight.dNdlogD_020_FCDP.values
    lwc = FCDP_flight.LWC_FCDP.values

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
            'Bin003_Y_mean': [],
            'Bin004_Y_mean': [],
            'Bin005_Y_mean': [],
            'Bin006_Y_mean': [],
            'Bin007_Y_mean': [],
            'Bin008_Y_mean': [],
            'Bin009_Y_mean': [],
            'Bin010_Y_mean': [],
            'Bin011_Y_mean': [],
            'Bin012_Y_mean': [],
            'Bin013_Y_mean': [],
            'Bin014_Y_mean': [],
            'Bin015_Y_mean': [],
            'Bin016_Y_mean': [],
            'Bin017_Y_mean': [],
            'Bin018_Y_mean': [],
            'Bin019_Y_mean': [],
            'Bin020_Y_mean': [],
            'Bin003_N_mean': [],
            'Bin004_N_mean': [],
            'Bin005_N_mean': [],
            'Bin006_N_mean': [],
            'Bin007_N_mean': [],
            'Bin008_N_mean': [],
            'Bin009_N_mean': [],
            'Bin010_N_mean': [],
            'Bin011_N_mean': [],
            'Bin012_N_mean': [],
            'Bin013_N_mean': [],
            'Bin014_N_mean': [],
            'Bin015_N_mean': [],
            'Bin016_N_mean': [],
            'Bin017_N_mean': [],
            'Bin018_N_mean': [],
            'Bin019_N_mean': [],
            'Bin020_N_mean': [],
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

        # Loop through each bin label
        for bin_label in range(3, 21):
            # Initialize lists to store values for 'Y' and 'N' labels
            values_y = []
            values_n = []

            # Loop through each row of data
            if index7_start is not None and index7_end is not None:
                for i in range(index7_start, index7_end + 1):
                    lwc_val = lwc[i]
                    label = 'Y' if 0 <= lwc_val <= 0.00001 else 'N'
            
                    column_name = f'dNdlogD_{bin_label:03d}_FCDP'
                    bin_value = FCDP_flight[column_name].values[i]
                    if np.isnan(bin_value):
                        bin_value = 0
                    # Append the bin value to the corresponding list based on the label
                    if label == 'Y':
                        values_y.append(bin_value)
                    else:
                        values_n.append(bin_value)
                    # print(f"Date: {date}, Start: {start20}, Stop: {end20}, LWC: {lwc_val}, Label: {label}")
            # Calculate mean for 'Y' label
            bin_means[f'Bin{bin_label:03d}_Y_mean'] = np.nanmean(values_y)

        # Calculate mean for 'N' label
            bin_means[f'Bin{bin_label:03d}_N_mean'] = np.nanmean(values_n)
        total_BCB_means.append(bin_means)
    master_FCDP_BCB.append(total_BCB_means)
# Now bin_means_y and bin_means_n contain means for 'Y' and 'N' labels for each bin


        # for bin_label in range(3, 21):
        #     values_y = []
        #     values_n = []
        #     print("index7_start:", index7_start)
        #     print("index7_end:", index7_end)



        # if index7_start is not None and index7_end is not None:
        #     for i in range(index7_start, index7_end + 1):
        #         lwc_val = lwc[i]
        #         label = 'Y' if 0 <= lwc_val <= 0.00001 else 'N'
        #         # print(f"Start: {start20}, Stop: {end20}, LWC: {lwc_val}, Label: {label}")

        #         column_name = f'dNdlogD_{bin_label:03d}_FCDP'
        #         bin_value = FCDP_flight[column_name].values[i]
        #         if np.isnan(bin_value):
        #             bin_value = 0
        #         if label == 'Y':
        #             values_y.append(bin_value)
        #         else:
        #             values_n.append(bin_value)
        #     print("Values Y:", values_y)
        #     print("Values N:", values_n)


        #     bin_key_y = f'Bin{bin_label:03d}_Y_mean'
        #     bin_means[bin_key_y] = np.nanmean(values_y)
        #     bin_key_n = f'Bin{bin_label:03d}_N_mean'
        #     bin_means[bin_key_n] = np.nanmean(values_n)
    #     total_BCB_means.append(bin_means)
    # master_FCDP_BCB.append(total_BCB_means)
        # if index7_start is not None and index7_end is not None:
        #     for i in range(index7_start, index7_end + 1):
        #         lwc_val = lwc[i]
        #         label = 'Y' if 0 <= lwc_val <= 0.00001 else 'N'
        #         # print(f"Start: {start20}, Stop: {end20}, LWC: {lwc_val}, Label: {label}")

        #     for bin_label in range(3,21):
        #         bin_key = f'Bin{bin_label:03d}_{label}_mean'
        #         column_name = f'dNdlogD_{bin_label:03d}_FCDP'  
        #         bin_value = FCDP_flight[column_name].values[i]  
        #         bin_means[bin_key] = pd.to_numeric(bin_means[bin_key], errors='coerce') 
        #         bin_means[bin_key] = np.nan_to_num(bin_means[bin_key], nan=0)
        #         bin_means[bin_key] = list(bin_means[bin_key]) + [bin_value]
        #         # bin_means[bin_key].append(bin_value)
        #     total_BCB_means.append(bin_means)
        # master_FCDP_BCB.append(total_BCB_means)

#%%
Y_BCB_calc = []
N_BCB_calc = []


for flight_data in master_FCDP_BCB:
    for bin_means in flight_data:
        Y_calc = {'Date': bin_means['Date'], 'BCB_start': bin_means['BCB_start'], 'BCB_stop': bin_means['BCB_stop']}
        N_calc = {'Date': bin_means['Date'], 'BCB_start': bin_means['BCB_start'], 'BCB_stop': bin_means['BCB_stop']}
        
        for bin_label in range(3, 21):
            bin_key_Y = f'Bin{bin_label:03d}_Y_mean'
            bin_key_N = f'Bin{bin_label:03d}_N_mean'

            Y_calc[bin_key_Y] = np.nanmean(bin_means[bin_key_Y]) * Logg[bin_label - 3]
            N_calc[bin_key_N] = np.nanmean(bin_means[bin_key_N]) * Logg[bin_label- 3]

        Y_BCB_calc.append(Y_calc)
        N_BCB_calc.append(N_calc)
#need to multiply by a million
# %%

# %%
Y_BCB_calc_cm3 = []
N_BCB_calc_cm3 = []

# Convert mean values in Y_BCB_calc from per m^3 to per cm^3 and append to new list
for entry in Y_BCB_calc:
    new_entry = {}
    for key, value in entry.items():
        if key not in ['Date', 'BCB_start', 'BCB_stop']:
            new_entry[key] = value / 1e6  # Convert from per m^3 to per cm^3
        else:
            new_entry[key] = value  # Keep 'Date', 'BCB_start', and 'BCB_stop' unchanged
    Y_BCB_calc_cm3.append(new_entry)

# Convert mean values in N_BCB_calc from per m^3 to per cm^3 and append to new list
for entry in N_BCB_calc:
    new_entry = {}
    for key, value in entry.items():
        if key not in ['Date', 'BCB_start', 'BCB_stop']:
            new_entry[key] = value / 1e6  # Convert from per m^3 to per cm^3
        else:
            new_entry[key] = value  # Keep 'Date', 'BCB_start', and 'BCB_stop' unchanged
    N_BCB_calc_cm3.append(new_entry)

# %%
def summary_bin_means(data):
    summary_list = []

    for flight_dict in data:
        total_summary = sum(flight_dict[f'Bin{bin_label:03d}_Y_mean'] for bin_label in range(3,21))
        summary_dict = {
            'Date': flight_dict.get('Date', ''),
            'BCB_start': flight_dict.get('BCB_start', ''),
            'BCB_stop': flight_dict.get('BCB_stop', ''),
            'Sum': total_summary
        }
        summary_list.append(summary_dict)

    return summary_list
print("Y_BCB_calc_cm3 before summary_bin_means:", Y_BCB_calc_cm3)  # Add this line to check Y_ACB_calc before processing
Y_sum_list = summary_bin_means(Y_BCB_calc_cm3)
print("Y_sum_list after summary_bin_means:", Y_sum_list)  # Add this line to check Y_sum_list after processing
print("\n")

# %%
def sum_bin_means(data):
    sum_list = []

    for flight_dict in data:
        total_sum = sum(flight_dict[f'Bin{bin_label:03d}_N_mean'] for bin_label in range(3, 21))
        sum_dict = {
            'Date': flight_dict.get('Date', ''),
            'BCB_start': flight_dict.get('BCB_start', ''),
            'BCB_stop': flight_dict.get('BCB_stop', ''),
            'Sum': total_sum
        }
        sum_list.append(sum_dict)

    return sum_list

# Apply the function to Y_BCB_calc and N_BCB_calc
# N_sum_list = sum_bin_means(N_ACB_calc)
# N_sum_list = sum_bin_means(N_BCB_calc)
print("N_sum_list after summary_bin_means:", N_BCB_calc_cm3)  # Add this line to check N_ACB_calc before processing
N_sum_list = sum_bin_means(N_BCB_calc_cm3)
print("N_sum_list after sum_bin_means:", N_sum_list)
# %%
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
# %%
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
# %%
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
# %%
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
# %%
#Time series for one leg of 2/26 where 
# the start time is 69536, and the stop time is 69710. 
#%%
time_FCDP_BCB = []
legs_info_FCDP = []

# Filter data for '2022-02-26'
filtered_indices = [i for i, date in enumerate(dates_legs) if date == '2022-02-26']

# print("Processing data for date '2022-02-26'")

for i in filtered_indices:
    date = dates_legs[i]
    leg_dict = leg_data[i]

    BCB_start_list = (leg_dict['LegIndex_02']['StartTimes'])
    BCB_stop_list = (leg_dict['LegIndex_02']['StopTimes'])
    # print(f"Leg at date {date} has start time: {ACB_start}, stop time: {ACB_stop}")
    if BCB_start_list and BCB_stop_list:
        # Get the first start and stop times
        BCB_start = int(BCB_start_list[0])
        BCB_stop = int(BCB_stop_list[0])
        print(f"Leg at date {date} has start time: {BCB_start}, stop time: {BCB_stop}")

        FCDP_flight = FCDP[i]
        times = FCDP_flight.Time_Start.values
        Bin3 = FCDP_flight.dNdlogD_003_FCDP.values
        Bin4 = FCDP_flight.dNdlogD_004_FCDP.values
        Bin5 = FCDP_flight.dNdlogD_005_FCDP.values
        Bin6 = FCDP_flight.dNdlogD_006_FCDP.values
        Bin7 = FCDP_flight.dNdlogD_007_FCDP.values
        Bin8 = FCDP_flight.dNdlogD_008_FCDP.values
        Bin9 = FCDP_flight.dNdlogD_009_FCDP.values
        Bin10= FCDP_flight.dNdlogD_010_FCDP.values
        Bin11 = FCDP_flight.dNdlogD_011_FCDP.values
        Bin12 =FCDP_flight.dNdlogD_012_FCDP.values
        Bin13 = FCDP_flight.dNdlogD_013_FCDP.values
        Bin14 = FCDP_flight.dNdlogD_014_FCDP.values
        Bin15 = FCDP_flight.dNdlogD_015_FCDP.values
        Bin16 = FCDP_flight.dNdlogD_016_FCDP.values
        Bin17 =FCDP_flight.dNdlogD_017_FCDP.values
        Bin18 = FCDP_flight.dNdlogD_018_FCDP.values
        Bin19 = FCDP_flight.dNdlogD_019_FCDP.values
        Bin20 = FCDP_flight.dNdlogD_020_FCDP.values
        lwc = FCDP_flight.LWC_FCDP.values

        total_BCB_time = []
        found_N_values = False

       
        index99_start = None
        index99_end = None

        bins = {
            'Date': date,
            'BCB_start': BCB_start,
            'BCB_stop': BCB_stop,
            'Bin003_Y': [],
            'Bin004_Y': [],
            'Bin005_Y': [],
            'Bin006_Y': [],
            'Bin007_Y': [],
            'Bin008_Y': [],
            'Bin009_Y': [],
            'Bin010_Y': [],
            'Bin011_Y': [],
            'Bin012_Y': [],
            'Bin013_Y': [],
            'Bin014_Y': [],
            'Bin015_Y': [],
            'Bin016_Y': [],
            'Bin017_Y': [],
            'Bin018_Y': [],
            'Bin019_Y': [],
            'Bin020_Y': [],
            'Bin003_N': [],
            'Bin004_N': [],
            'Bin005_N': [],
            'Bin006_N': [],
            'Bin007_N': [],
            'Bin008_N': [],
            'Bin009_N': [],
            'Bin010_N': [],
            'Bin011_N': [],
            'Bin012_N': [],
            'Bin013_N': [],
            'Bin014_N': [],
            'Bin015_N': [],
            'Bin016_N': [],
            'Bin017_N': [],
            'Bin018_N': [],
            'Bin019_N': [],
            'Bin020_N': [],
        }
        # for idx, t in enumerate(times):
        #     if BCB_start <= int(t) <= BCB_stop:
        #         lwc_val = lwc[idx]
        #         for bin_label in range(3, 21):
        #             column_name = f'dNdlogD_{bin_label:03d}_FCDP'
        #             bin_key = f'Bin{bin_label:03d}_{"Y" if 0 <= lwc_val <= 0.01 else "N"}'
        #             bin_value = FCDP_flight[column_name].values[idx]
        #             bins[bin_key].append(bin_value)
        for idx, t in enumerate(times):
            if BCB_start <= int(t) <= BCB_stop:
                lwc_val = lwc[idx]
                print(f"LWC value at index {idx}: {lwc_val}")
                category = 'Y' if 0 <= lwc_val <= 0.01 else 'N'
                if category == 'N':
                    found_N_values = True
                for bin_label in range(3, 21):
                    column_name = f'dNdlogD_{bin_label:03d}_FCDP'
                    bin_key = f'Bin{bin_label:03d}_{category}'
                    bin_value = FCDP_flight[column_name].values[idx]
                    bins[bin_key].append(bin_value)
    
        total_BCB_time.append(bins)
        if not found_N_values:
            print("No 'N' values found in LWC between BCB start and stop times.")
        break
            

time_FCDP_BCB.append(total_BCB_time)
# %%
#This code is for the clear BCB values 





# Create separate lists to store Y and N values for each bin
Y_values = [[] for _ in range(3, 21)]
N_values = [[] for _ in range(3, 21)]

# Iterate over the data in time_CAS_BCB
for bins_data in time_FCDP_BCB:
    for bins in bins_data:
        # Extract data for each bin and append to respective lists
        for bin_label in range(3, 21):
            bin_key_Y = f'Bin{bin_label:03d}_Y'
            bin_key_N = f'Bin{bin_label:03d}_N'
            Y_values[bin_label - 3].extend(bins[bin_key_Y])
            N_values[bin_label - 3].extend(bins[bin_key_N])
    break  # Exit after processing the first entry

# Find the maximum length among Y and N lists
max_length = max(max(len(lst) for lst in Y_values), max(len(lst) for lst in N_values))

# Pad lists with NaN values to make them equal length
for lst in Y_values + N_values:
    while len(lst) < max_length:
        lst.append(np.nan)

# Create x-axis values from ACB_start to ACB_stop
x_values = list(range(BCB_start, BCB_start + max_length))

plt.figure(figsize=(10, 6))
for bin_label, Y_data in enumerate(Y_values, start=3):
    plt.plot(x_values, Y_data, label=f'Bin{bin_label}_Y')

plt.xlabel('Seconds after midnight',  fontsize=16, fontweight='bold')
plt.ylabel('Droplet concentration (dN/dlogd)',  fontsize=16, fontweight='bold')
plt.legend(fontsize=3.2, loc='upper right')
plt.grid(True)
plt.yscale('log')
plt.xlim(BCB_start, BCB_stop)  # Set the limits of the x-axis

plt.title('2022-02-26\n 48862 - 49061 seconds after midnight\n Clear sky', fontsize=16, fontweight='bold')
plt.show()

# %%
#This code is for the cloudy BCB values 





# Create separate lists to store Y and N values for each bin
Y_values = [[] for _ in range(3, 21)]
N_values = [[] for _ in range(3, 21)]

# Iterate over the data in time_CAS_BCB
for bins_data in time_FCDP_BCB:
    for bins in bins_data:
        # Extract data for each bin and append to respective lists
        for bin_label in range(3, 21):
            bin_key_Y = f'Bin{bin_label:03d}_Y'
            bin_key_N = f'Bin{bin_label:03d}_N'
            Y_values[bin_label - 3].extend(bins[bin_key_Y])
            N_values[bin_label - 3].extend(bins[bin_key_N])
    break  # Exit after processing the first entry

# Find the maximum length among Y and N lists
max_length = max(max(len(lst) for lst in Y_values), max(len(lst) for lst in N_values))

# Pad lists with NaN values to make them equal length
for lst in Y_values + N_values:
    while len(lst) < max_length:
        lst.append(np.nan)

# Create x-axis values from ACB_start to ACB_stop
x_values = list(range(BCB_start, BCB_start + max_length))

# Plotting Y values
plt.figure(figsize=(10, 6))
for bin_label, N_data in enumerate(N_values, start=3):
    plt.plot(x_values, N_data, label=f'Bin{bin_label}_N')

plt.xlabel('Seconds after midnight',  fontsize=16, fontweight='bold')
plt.ylabel('Droplet concentration (dN/dlogd)',  fontsize=16, fontweight='bold')
plt.legend(fontsize=3.2, loc='upper right')
plt.grid(True)
plt.yscale('log')
plt.xlim(BCB_start, BCB_stop)  # Set the limits of the x-axis

plt.title('2022-02-26\n 48862 - 49061 seconds after midnight\n Cloudy sky', fontsize=16, fontweight='bold')
plt.show()

# %%
#Time series for one leg of 2/26 where the liquid water content for clear skies is 
# is plotted against the start time 48862 and the stop time 49061.  
#This time series runs from 48862 to 49061 and is for a Below Cloud Base leg. 
#%%
lwc_FCDP_BCB = []


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

        FCDP_flight =FCDP[i]
        times = FCDP_flight.Time_Start.values
        lwc = FCDP_flight.LWC_FCDP.values

        LWC_Y = []
        LWC_N=[]
    
        for idx, t in enumerate(times):
            if BCB_start <= int(t) <= BCB_stop:
                lwc_val = lwc[idx]
                if 0 <= lwc_val <= 0.01:
                    LWC_Y.append(lwc_val)
                else:
                    LWC_N.append(lwc_val)
    
        lwc_FCDP_BCB.append({'Date': date, 'LWC_Y':LWC_Y, 'LWC_N': LWC_N})
        break
            

# %%
#This code is for clear sky BCB

lwc_Y_values = []

for entry in lwc_FCDP_BCB:
    lwc_Y_values.extend(entry['LWC_Y'])

# Find the maximum length among lwc_Y list
max_length = len(lwc_Y_values)

# Create x-axis values from BCB_start to BCB_stop
x_values = list(range(BCB_start, BCB_start + max_length))

# plt.figure(figsize=(10, 6))
plt.plot(x_values, lwc_Y_values, label='FCDP_LWC', color='blue')

plt.xlabel('Seconds after midnight', fontsize=16, fontweight='bold')
plt.ylabel('Clear LWC', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.xlim(BCB_start, BCB_stop)  # Set the limits of the x-axis
plt.title('2022-02-26\n 48862 - 49061 seconds after midnight\n Clear sky', fontsize=16, fontweight='bold')
plt.show()
# %%
#Time series for one leg of 2/26 where the liquid water content for clear skies is 
# is plotted against the start time 48862 and the stop time 49061.  
#This time series runs from 48862 to 49061 and is for a Below Cloud Base leg.I plotted both the 
#given lwc values found in the CAS in blue with my own hand-calculated LWC values in red.  
# %%
C_12=math.log(4.5)-math.log(3)
C_13=math.log(6.0)-math.log(4.5)
C_14=math.log(8)-math.log(6)
C_15=math.log(10)-math.log(8)
C_16=math.log(12)-math.log(10)
C_17=math.log(14)-math.log(12)
C_18=math.log(16)-math.log(14)
C_19=math.log(18)-math.log(16)
C_20=math.log(21)-math.log(18)
C_21=math.log(24)-math.log(21)
C_22=math.log(27)-math.log(24)
C_23=math.log(30)-math.log(27)
C_24=math.log(33)-math.log(30)
C_25=math.log(36)-math.log(33)
C_26=math.log(39)-math.log(36)
C_27=math.log(42)-math.log(39)
C_28=math.log(46)-math.log(42)
C_29=math.log(50)-math.log(46)
# %%
bin_center=[3.75, 5.25, 7, 9, 11, 13, 15, 17, 19.5, 
            22.5, 25.5, 28.5, 31.5, 34.5, 37.5, 40.5, 44, 48]
# %%
B_00=((3.75)**3)
B_01= ((5.25)**3)
B_02=((7)**3)
B_03=((9)**3)
B_04=((11)**3)
B_05=((13)**3)
B_06=((15)**3) 
#%%
B_07=((17)**3)
B_08=((19.5)**3)
B_09=((22.5)**3)
B_10=((25.5)**3)
B_11=((28.5)**3)
B_12=((31.5)**3)
B_13=((34.5)**3)
B_14=((37.5)**3)
B_15=((40.5)**3)
#%%
B_16=((44)**3)
B_17=((48)**3)

#%%
radius=[B_00, B_01, B_02, B_03, B_04, B_05, B_06, B_07, B_08,
        B_09, B_10, B_11, B_12, B_13, B_14, B_15, B_16, B_17]
#%%
C_values=[ C_12, C_13, C_14, C_15, C_16, C_17, C_18, 
         C_19, C_20, C_21, C_22, C_23, C_24, C_25, C_26, C_27, C_28, C_29]
# %%
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
# %%
time_FCDP_BCB_full = []
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

        FCDP_flight = FCDP[i]
        times = FCDP_flight.Time_Start.values
        Bin3 = FCDP_flight.dNdlogD_003_FCDP.values
        Bin4 = FCDP_flight.dNdlogD_004_FCDP.values
        Bin5 = FCDP_flight.dNdlogD_005_FCDP.values
        Bin6 = FCDP_flight.dNdlogD_006_FCDP.values
        Bin7 = FCDP_flight.dNdlogD_007_FCDP.values
        Bin8 = FCDP_flight.dNdlogD_008_FCDP.values
        Bin9 = FCDP_flight.dNdlogD_009_FCDP.values
        Bin10= FCDP_flight.dNdlogD_010_FCDP.values
        Bin11 = FCDP_flight.dNdlogD_011_FCDP.values
        Bin12 =FCDP_flight.dNdlogD_012_FCDP.values
        Bin13 = FCDP_flight.dNdlogD_013_FCDP.values
        Bin14 = FCDP_flight.dNdlogD_014_FCDP.values
        Bin15 = FCDP_flight.dNdlogD_015_FCDP.values
        Bin16 = FCDP_flight.dNdlogD_016_FCDP.values
        Bin17 =FCDP_flight.dNdlogD_017_FCDP.values
        Bin18 = FCDP_flight.dNdlogD_018_FCDP.values
        Bin19 = FCDP_flight.dNdlogD_019_FCDP.values
        Bin20 = FCDP_flight.dNdlogD_020_FCDP.values
        lwc = FCDP_flight.LWC_FCDP.values

        total_BCB_time_full = []


        
        index99_start = None
        index99_end = None
            

        bins_full = {
            'Date': date,
            'BCB_start': BCB_start,
            'BCB_stop': BCB_stop,
            'Bin003_Y': [],
            'Bin004_Y': [],
            'Bin005_Y': [],
            'Bin006_Y': [],
            'Bin007_Y': [],
            'Bin008_Y': [],
            'Bin009_Y': [],
            'Bin010_Y': [],
            'Bin011_Y': [],
            'Bin012_Y': [],
            'Bin013_Y': [],
            'Bin014_Y': [],
            'Bin015_Y': [],
            'Bin016_Y': [],
            'Bin017_Y': [],
            'Bin018_Y': [],
            'Bin019_Y': [],
            'Bin020_Y': [],
            'Bin003_N': [],
            'Bin004_N': [],
            'Bin005_N': [],
            'Bin006_N': [],
            'Bin007_N': [],
            'Bin008_N': [],
            'Bin009_N': [],
            'Bin010_N': [],
            'Bin011_N': [],
            'Bin012_N': [],
            'Bin013_N': [],
            'Bin014_N': [],
            'Bin015_N': [],
            'Bin016_N': [],
            'Bin017_N': [],
            'Bin018_N': [],
            'Bin019_N': [],
            'Bin020_N': [],
        }
        

        for idx, t in enumerate(times):
            if BCB_start <= int(t) <= BCB_stop:
                lwc_val = lwc[idx]
                print(f"LWC value at index {idx}: {lwc_val}")
                category = 'Y' if 0 <= lwc_val <= 0.01 else 'N'
                if category == 'N':
                    found_N_values = True
                for bin_label in range(3, 21):
                    column_name = f'dNdlogD_{bin_label:03d}_FCDP'
                    bin_key = f'Bin{bin_label:03d}_{category}'
                    bin_value = FCDP_flight[column_name].values[idx]
                    bins_full[bin_key].append(bin_value)
    
        total_BCB_time_full.append(bins_full)
        if not found_N_values:
            print("No 'N' values found in LWC between BCB start and stop times.")
        break
            

time_FCDP_BCB_full.append(total_BCB_time_full)
# %%
time_FCDP_BCB_full_modified = []

# Iterate over each dictionary (which is actually a list) in time_CAS_BCB_full
for data_list in time_FCDP_BCB_full:
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
    time_FCDP_BCB_full_modified.append(modified_dict)

# Now time_CAS_BCB_full_modified contains dictionaries with keys ending in '_Y'
# %%
time_FCDP_BCB_full_modified_N = []

# Iterate over each dictionary (which is actually a list) in time_CAS_BCB_full
for data_list_N in time_FCDP_BCB_full:
    modified_dict_N = {}
    # Iterate over each dictionary in the list
    for dictionary in data_list_N:
        # Iterate over the keys in the current dictionary
        for key, value in dictionary.items():
            # Check if the key ends with '_Y'
            if key.endswith('_N'):
                # If it does, add it to the modified dictionary
                modified_dict_N[key] = value
    # Append the modified dictionary to time_CAS_BCB_full_modified
    time_FCDP_BCB_full_modified_N.append(modified_dict_N)
# %%
# Loop through each dictionary in total_BCB_time_full
for index_data in time_FCDP_BCB_full_modified:
    # Loop through each key-value pair in the dictionary
    for key, value in index_data.items():
        # Check if the value is nan
        if isinstance(value, float) and math.isnan(value):
            # Replace nan with 0
            index_data[key] = 0

#%%
# Loop through each dictionary in total_BCB_time_full
for index_data_N in time_FCDP_BCB_full_modified_N:
    # Loop through each key-value pair in the dictionary
    for key, value in index_data_N.items():
        # Check if the value is nan
        if isinstance(value, float) and math.isnan(value):
            # Replace nan with 0
            index_data_N[key] = 0

# %%
#This is for clear sky BCB

LWC_list = []
sums_list = []  # Initialize list to store sum values for each row

# Loop through each index in total_BCB_time_full
for index_data in time_FCDP_BCB_full_modified:
    LWC_row = []  # Initialize list to store F values for the current row
    for idx in range(len(index_data['Bin003_Y'])):  # Assuming 'Bin00_Y' has the same length as other bins
        F_sum = 0  # Initialize sum of F values for the current row
        for bin_label in range(18):
            # Get the N value and corresponding bin center value
            bin_key = f'Bin{bin_label+3:03d}_Y'  # Construct the key
            N_value = index_data[bin_key][idx]
            # N_value = index_data[f'Bin{bin_label:03d}_Y'][idx]
            bin_center_value = bin_center[bin_label]
            
            # Calculate A and F for each bin
            A = (N_value * C_values[bin_label])
            B = (bin_center_value ** 3) /1000000000000000000
            F = (A * ((4 / 3) * 3.14) * B ) 
            
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

# %%


# Your existing code for plotting lwc_Y values
plt.figure(figsize=(10, 6))
plt.plot(sums_list, lwc_Y_values, label='FCDP_LWC', color='blue')
plt.xlabel('Calculated lwc', fontsize=16, fontweight='bold')
plt.ylabel('FCDP LWC', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.yscale('log')
plt.xscale('log')
plt.grid(True)
# plt.xlim(BCB_start, BCB_stop)
plt.title('2022-02-26\n 48862 - 49061 seconds after midnight\n Clear sky: Below Cloud Base', fontsize=16, fontweight='bold')

# %%
#This is for clear

plt.figure(figsize=(10, 6))
plt.plot(x_values, sums_list, label='Calculated_LWC', color='red')
plt.xlabel('Seconds after midnight', fontsize=16, fontweight='bold')
plt.ylabel('Clear LWC', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True)
plt.xlim(BCB_start, BCB_stop)
plt.title('2022-02-26\n 48862 - 49061 seconds after midnight\n Clear sky: Above Cloud Base', fontsize=16, fontweight='bold')

# %%
plt.figure(figsize=(10, 6))
plt.plot(x_values, lwc_Y_values, label='CAS_LWC', color='blue')
plt.xlabel('Seconds after midnight', fontsize=16, fontweight='bold')
plt.ylabel('Caclulated LWC', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True)
plt.xlim(BCB_start, BCB_stop)
plt.title('2022-02-26\n 48862 - 49061 seconds after midnight\n Clear sky: Below Cloud Base', fontsize=16, fontweight='bold')

# Plotting sums_list on the same graph
plt.plot(x_values, sums_list, label='Calculated LWC', color='red')  # Add this line
plt.legend(fontsize=12)  # Add this line to display legend for the new plot

plt.show()
# %%
