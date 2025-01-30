#%%
#This code is for the linear regression run for May at the minimum altitude level with dust. 



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


bin_log=[C_12, C_13, C_14, C_15, C_16, 
        C_17, C_18, C_19, C_20, C_21, C_22, C_23, C_24, C_25, C_26, C_27, C_28, C_29]
#[C_0, C_01, C_02, C_03, C_04, C_05, C_06, C_07, C_08, C_09, C_10, C_11, C_12,
bin_center=[0.555, 0.645, 0.715, 0.785, 0.855, 0.925, 
            0.995, 1.07, 1.14, 1.21, 1.38, 1.75, 2.25, 
            2.75, 3.25, 3.75, 4.5, 5.75, 6.85, 7.55, 
            9.05, 11.4, 13.8, 17.5, 22.5, 27.5, 32.5, 
            37.5, 42.5, 47.5]
bin_center = np.array(bin_center)

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
dates_sum = ['2022-05-05', '2022-05-10', '2022-05-16', '2022-05-17', '2022-05-18', 
             '2022-05-20','2022-05-21', '2022-05-31']
#'2022-05-17',2022-05-18'
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
        if date >= '2022-05-05':
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
    
        if date >= '2022-05-05':
        #         print(f"\nFirst row for date {'2022-01-15'}:")
        #         print(df_sum.head())
        # except Exception as e:
        #     print(f"Error processing file: {file_path}, Date: {date}")
        #     print(f"Error message: {str(e)}")
         
            print(date)
            time = df_sum['Time_mid']
            print("yes")  
            
# %%
 
# %%
leg_data = []
leg_name=['Time_Start', '  Time_Stop', '  Julian_Day', 
          '  Date', '  LegIndex']

dates_legs= ['2022-05-05', '2022-05-10','2022-05-16', '2022-05-17','2022-05-18',
             '2022-05-20', '2022-05-21', '2022-05-31']
#'2022-05-17','2022-05-18',
for date in dates_legs:
    datestr = date.replace('-', '')
    fname_legs = sorted(glob.glob(f'/home/disk/eos4/kathem24/activate/data/MET/2022/LegFLags/csv/ACTIVATE-LegFlags_HU25_{datestr}_R*.csv'), reverse=True)

    leg_dictionary = {
        'Date': date,
        'LegIndex_02': {'StartTimes': [], 'StopTimes': []},
        'LegIndex_06': {'StartTimes': [], 'StopTimes': []}
    }

    for file_path in fname_legs:
        if date >= '2022-05-05':
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

# %%

Z0 = 0.002  # meters (typical value for open terrain)
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

#%%
#%%

# %%



#CAS data

bin_name = ['CAS_Bin12', 'CAS_Bin13', 'CAS_Bin14', 'CAS_Bin15', 'CAS_Bin16', 'CAS_Bin17', 
            'CAS_Bin18', 'CAS_Bin19', 'CAS_Bin20', 'CAS_Bin21', 'CAS_Bin22', 'CAS_Bin23', 'CAS_Bin24', 'CAS_Bin25', 'CAS_Bin26',
            'CAS_Bin27', 'CAS_Bin28', 'CAS_Bin29']

CAS = []

dates_CAS = ['2022-05-05', '2022-05-10','2022-05-16', '2022-05-17','2022-05-18',
             '2022-05-18',
             '2022-05-20','2022-05-21', '2022-05-31']
#'2022-05-17','2022-05-18',
for date in dates_CAS:
    datestr = date.replace('-', '')
    fname_CAS = sorted(glob.glob(f'/home/disk/eos4/kathem24/activate/data/cloudaerospect/2022csv/ACTIVATE-LARGE-CAS_HU25_{datestr}_R*.csv'), reverse=True)
    # print(date)
    # print(fname_CAS)

    run = 1
    for file_path in fname_CAS:
        nums_file_paths = len(fname_CAS)

        if date >= ('2022-05-05'):
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

# %%

#%%


# %%
master_CAS_Min = []
leggs_info = []

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

    total_Min_means = []

    for i in range(len(Min_Start)):
        index11_start=None
        index11_end=None  
        start40 = int(Min_Start[i])
        end40 = Min_Stop[i]

        lwc_criteria = []

        for i in range(len(times)):
            start41 = int(times[i])
            if start41 == start40:
                index11_start = i
                break

        for i in range(len(times)):
            end41 = int(times[i])
            if end41 == end40:
                index11_end = i
                break

        if index11_start is not None and index11_end is not None:
            for i in range(index11_start, index11_end + 1):
                lwc_vals = lwc[i]
                labelss = 'Y' if 0 <= lwc_vals <= 0.01 else 'N'
                lwc_criteria.append(labelss)

            leggs_info.append({
                'Date': date,
                'Min_start': start40,
                'Min_stop': end40,
                'LWC_criteria': lwc_criteria
            })

    master_CAS_Min.append(total_Min_means)

# Print leg_info or use it as needed
for leg in leggs_info:
    print(f"Date: {leg['Date']}, Start: {leg['Min_start']}, Stop: {leg['Min_stop']}, LWC criteria: {leg['LWC_criteria']}")

#%%
master_CAS_Min = []
leggs_info = []

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

    total_Min_means = []

    for i in range(len(Min_Start)):
        index11_start = None
        index11_end = None
        start40 = int(Min_Start[i])
        end40 = Min_Stop[i]

        bin_meanss = {
            'Date': date,
            'Min_start': start40,
            'Min_stop': end40,
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

        for i in range(len(times)):
            start41 = int(times[i])
            if start41 == start40:
                index11_start = i
                break

        for i in range(len(times)):
            end41 = int(times[i])
            if end41 == end40:
                index11_end = i
                break

        if index11_start is not None and index11_end is not None:
            for i in range(index11_start, index11_end + 1):
                lwc_vals = lwc[i]
                labelss = 'Y' if 0 <= lwc_vals <= 0.01 else 'N'

                for bin_criteria in range(12, 30):
                    bin_keys = f'Bin{bin_criteria}_{labelss}_mean'
                    bin_values = CAS_flight[f'CAS_Bin{bin_criteria}'].values[i]
                    bin_meanss[bin_keys] = pd.to_numeric(bin_meanss[bin_keys], errors='coerce')  # Convert to numeric, and handle errors by converting to NaN
                    bin_meanss[bin_keys] = np.nan_to_num(bin_meanss[bin_keys], nan=0)
                    bin_meanss[bin_keys] = list(bin_meanss[bin_keys]) + [bin_values]
                    bin_meanss[bin_keys].append(bin_values)
            total_Min_means.append(bin_meanss)

    master_CAS_Min.append(total_Min_means)

# Print or use master_CAS_BCB as needed
for item in master_CAS_Min:
    for bin_meanss in item:
        print(f"Date: {bin_meanss['Date']}, Start: {bin_meanss['Min_start']}, Stop: {bin_meanss['Min_stop']}")
        for bin_criteria in range(12, 30):
            for labelss in ['Y', 'N']:
                bin_keys = f'Bin{bin_criteria}_{labelss}_mean'
                mean_values = np.nanmean(bin_meanss[bin_keys])
                print(f"   Bin{bin_criteria}_{labelss} Mean: {mean_values}")
#%%
Y_Min_calc = []
N_Min_calc = []

for flight_data in master_CAS_Min:
    for bin_meanss in flight_data:
        Y_calcs = {'Date': bin_meanss['Date'], 'Min_start': bin_meanss['Min_start'], 'Min_stop': bin_meanss['Min_stop']}
        N_calcs = {'Date': bin_meanss['Date'], 'Min_start': bin_meanss['Min_start'], 'Min_stop': bin_meanss['Min_stop']}
        
        for bin_criteria in range(12, 30):
            bin_keys_Y = f'Bin{bin_criteria}_Y_mean'
            bin_keys_N = f'Bin{bin_criteria}_N_mean'

            Y_calcs[bin_keys_Y] = np.nanmean(bin_meanss[bin_keys_Y]) * Logg[bin_criteria - 12]
            N_calcs[bin_keys_N] = np.nanmean(bin_meanss[bin_keys_N]) * Logg[bin_criteria - 12]

        Y_Min_calc.append(Y_calcs)
        N_Min_calc.append(N_calcs)
#%%
#%%
def min_sum_bin_means(data):
    min_sum_list = []

    for flight_dict in data:
        total_sum_min = sum(flight_dict.get(f'Bin{bin_criteria}_Y_mean', 0) for bin_criteria in range(12, 30))
        sum_min_dict = {
            'Date': flight_dict.get('Date', ''),
            'Min_start': flight_dict.get('Min_start', ''),
            'Min_stop': flight_dict.get('Min_stop', ''),
            'Sum': total_sum_min
        }
        min_sum_list.append(sum_min_dict)

    return min_sum_list

# Apply the function to Y_BCB_calc and N_BCB_calc
Y_min_list = min_sum_bin_means(Y_Min_calc)
# N_sum_list = sum_bin_means(N_BCB_calc)
#%%
def min_sum_bin_means2(data):
    min_sum_list2 = []

    for flight_dict in data:
        total_sum_min2 = sum(flight_dict.get(f'Bin{bin_criteria}_N_mean', 0) for bin_criteria in range(12, 30))
        sum_min_dict2 = {
            'Date': flight_dict.get('Date', ''),
            'Min_start': flight_dict.get('Min_start', ''),
            'Min_stop': flight_dict.get('Min_stop', ''),
            'Sum': total_sum_min2
        }
        min_sum_list2.append(sum_min_dict2)

    return min_sum_list2

# Apply the function to Y_BCB_calc and N_BCB_calc
N_min_list = min_sum_bin_means2(N_Min_calc)
# N_sum_list = sum_bin_means(N_BCB_calc)
#%%
wind_mappingg = {date: wind_speed for date, wind_speed in zip(corrected_calc_min['Date'], corrected_calc_min['Corrected_mean_windspeed'])}

# Extract data for the scatterplot
dates = [entry['Date'] for entry in Y_min_list]
sum_valuess = [entry['Sum'] for entry in Y_min_list]
corrected_windspeedss = [wind_mappingg.get(date, np.nan) for date in dates]

# Create the scatterplot
plt.scatter(sum_valuess, corrected_windspeedss)
plt.title('Minimum altitude with dust May 2022', fontsize=14, fontweight='bold')
plt.xlabel('Total droplet concentration dN (per cm^3) for clear skies ', fontsize=14, fontweight='bold')
plt.ylabel('Mean windspeed (m/s)', fontsize=14, fontweight='bold')
plt.xscale('log')
plt.show()
#%%
wind_mapping2 = {date: wind_speed for date, wind_speed in zip(corrected_calc_min['Date'], corrected_calc_min['Corrected_mean_windspeed'])}

# Extract data for the scatterplot
dates = [entry['Date'] for entry in N_min_list]
sum_values2 = [entry['Sum'] for entry in N_min_list]
corrected_windspeeds2 = [wind_mapping2.get(date, np.nan) for date in dates]

# Create the scatterplot
plt.scatter(sum_values2, corrected_windspeeds2, color='red')
plt.title('Minimum altitude May 2022', fontsize=14, fontweight='bold')
plt.xlabel('Total droplet concentration dN (per cm^3) for cloudy skies ', fontsize=14, fontweight='bold')
plt.ylabel('Mean windspeed (m/s)', fontsize=14, fontweight='bold')
plt.xscale('log')
plt.show()
#%%
wind_mappingg = {date: wind_speed for date, wind_speed in zip(corrected_calc_min['Date'], corrected_calc_min['Corrected_mean_windspeed'])}

# Extract data for the scatterplot
dates = [entry['Date'] for entry in Y_min_list]
sum_valuess = [entry['Sum'] for entry in Y_min_list]
corrected_windspeedss = [wind_mappingg.get(date, np.nan) for date in dates]

# Create the scatterplot
plt.scatter(sum_valuess, corrected_windspeedss)
plt.title('Minimum altitude May 2022', fontsize=14, fontweight='bold')
plt.xlabel('Total droplet concentration dN (per cm^3) for clear skies ', fontsize=14, fontweight='bold')
plt.ylabel('Mean windspeed (m/s)', fontsize=14, fontweight='bold')
plt.show()
#%%
wind_mapping2 = {date: wind_speed for date, wind_speed in zip(corrected_calc_min['Date'], corrected_calc_min['Corrected_mean_windspeed'])}

# Extract data for the scatterplot
dates = [entry['Date'] for entry in N_min_list]
sum_values2 = [entry['Sum'] for entry in N_min_list]
corrected_windspeeds2 = [wind_mapping2.get(date, np.nan) for date in dates]

# Create the scatterplot
plt.scatter(sum_values2, corrected_windspeeds2, color='red')
plt.title('Minimum altitude May 2022', fontsize=14, fontweight='bold')
plt.xlabel('Total droplet concentration dN (per cm^3) for cloudy skies ', fontsize=14, fontweight='bold')
plt.ylabel('Mean windspeed (m/s)', fontsize=14, fontweight='bold')
plt.show()

#%%
import statsmodels.api as sm

















#%%

#linear regression analysis 

dates_Y = [entry['Date'] for entry in Y_min_list]
sum_valuess_Y = [entry['Sum'] for entry in Y_min_list]
corrected_windspeedss_Y = [wind_mappingg.get(date, np.nan) for date in dates_Y]

# Remove missing values from the data
valid_indices_Y = [i for i in range(len(corrected_windspeedss)) if not np.isnan(corrected_windspeedss_Y[i]) and not np.isnan(sum_valuess_Y[i])]
X_valid_Y = [corrected_windspeedss_Y[i] for i in valid_indices_Y]
y_valid_Y = [sum_valuess_Y[i] for i in valid_indices_Y]

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

#Plotting the regression line and confidence interval 


# Your existing code for creating the scatterplot
plt.scatter(sum_valuess, corrected_windspeedss)
plt.title('Minimum altitude May with dust 2022', fontsize=14, fontweight='bold')
plt.xlabel('Total droplet concentration dN (per cm^3) for clear skies ', fontsize=14, fontweight='bold')
plt.ylabel('Mean windspeed (m/s)', fontsize=14, fontweight='bold')
plt.xscale('log')

# Plotting the regression line
# Extract the coefficients from the model
intercept_Y, slope_Y = model_Y.params

# Define the range of x values for the regression line
x_range = np.linspace(min(sum_valuess), max(sum_valuess), 100)

# Calculate the corresponding y values using the regression equation
y_range = intercept_Y + slope_Y * x_range

# Plot the regression line
plt.plot(x_range, y_range, color='red', label='Regression Line')

# Plotting the regression line with confidence intervals
conf_int = model_Y.conf_int(alpha=0.05)  # 95% confidence intervals
plt.fill_between(x_range, conf_int[0][1] + conf_int[1][1] * x_range, conf_int[0][0] + conf_int[1][0] * x_range, color='blue', alpha=0.2, label='95% Confidence Interval')

plt.legend()
# %%
