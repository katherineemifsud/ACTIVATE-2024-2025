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
bin_center=[2.25, 
            2.75, 3.25, 3.75, 4.5, 5.75, 6.85, 7.55, 
            9.05, 11.4, 13.8, 17.5, 22.5, 27.5, 32.5, 
            37.5, 42.5, 47.5]
bin_center = np.array(bin_center)


#%% 

#%%
#Summary Data
col_name = ['Time_mid', 'Latitude', 'Longitude', 'GPS_altitude', 'Pressure_Altitude',
             'Pitch', 'Roll', 'True_Heading', 'True_Air_Speed', 
             'Static_Air_Temp', 'IR_Surf_Temp', 'Static_Pressure',
             'Wind_Speed']
summary=[]
dates_sum = ['2022-06-02', 
             '2022-06-03', '2022-06-05', '2022-06-07', '2022-06-08', 
             '2022-06-10', '2022-06-13',
             '2022-06-17', '2022-06-18']
#'2022-06-11', #'2022-06-14',
for date in dates_sum:
    datestr = date.replace('-', '')
    fname_sum = sorted(glob.glob(f'/home/disk/eos4/kathem24/activate/data/MET/2022/SummaryREAL/ict/ACTIVATE-SUMMARY_HU25_{datestr}_R1*.csv'), reverse=True)
    print(date)
    print(fname_sum)

    run = 1
    for file_path in fname_sum: 
        num_file_paths = len(fname_sum)

        #num = index = dates_sum.index(date)
        
        # try:
        if date >= '2022-06-02':
            df_sum = pd.read_csv(file_path, skiprows=48, quoting=csv.QUOTE_NONE)
        # elif date == '2022-06-14':
        #     df_sum = pd.read_csv(file_path, skiprows= 50, quoting=csv.QUOTE_NONE)


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
            print("YESSSSSSSSSSSSSSSSSSSSSSSSSSSSs")
            summary.append(df_sum)

        run = run+1

    #     if date == '2022-06-18':
    #             print(f"\nFirst row for date {'2022-06-18'}:")
    #             print(df_sum.head())
                    
    # except Exception as e:
    #         print(f"Error processing file: {file_path}, Date: {date}")
    #         print(f"Error message: {str(e)}")
        
    # #     except Exception as e:
    # #         print(f"Error processing file: {file_path}, Date: {date}")
    # #          print(f"Error message: {str(e)}")
        
    #     print(date)
    # time = df_sum['Time_mid']
    # print("yes")
    
        if date >= '2022-06-02':
            # print(f"\nFirst row for date {'2022-06-02'}:")
            # print(df_sum.head())
        # except Exception as e:
        #         print(f"Error processing file: {file_path}, Date: {date}")
        #         print(f"Error message: {str(e)}")
            
            print(date)
            time = df_sum['Time_mid']
            print("yes")  
        
# %%
 
# %%
leg_data = []
leg_name=['Time_Start', '  Time_Stop', '  Julian_Day', 
          '  Date', '  LegIndex']

dates_legs= ['2022-06-02', 
             '2022-06-03', '2022-06-05', '2022-06-07', '2022-06-08', 
             '2022-06-10','2022-06-13',
             '2022-06-17', '2022-06-18']
#'2022-06-11', #'2022-06-14',
for date in dates_legs:
    datestr = date.replace('-', '')
    fname_legs = sorted(glob.glob(f'/home/disk/eos4/kathem24/activate/data/MET/2022/LegFLags/csv/ACTIVATE-LegFlags_HU25_{datestr}_R*.csv'), reverse=True)

    leg_dictionary = {
        'Date': date,
        'LegIndex_02': {'StartTimes': [], 'StopTimes': []},
        'LegIndex_06': {'StartTimes': [], 'StopTimes': []}
    }

    for file_path in fname_legs:
        if date >= '2022-06-02': 
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


    times = sum_flight.Time_mid.values
    winds = sum_flight.Wind_Speed.values
    alts = sum_flight.GPS_altitude.values
    
    all_BCB_means = []

    print(f"Date: {date}, Number of BCB starts: {len(BCB_start)}")

    for i in range(len(BCB_start)):
        # print(f"Processing BCB leg {i+1} out of {len(BCB_start)}, Start time: {BCB_start[i]}, End time: {BCB_stop[i]}")
        index1_start=None
        index1_end=None  
        start = int(BCB_start[i])
        end = BCB_stop[i]
        # print(f"Processing BCB leg {i+1} out of {len(BCB_start)}, Start time: {start}, End time: {end}")
        
        
        wind_alt = {
            'Date': date,
            'BCB_start': start,
            'BCB_end': end,
            'Alts_mean': [],
            'Winds_mean': []
        }
        # print(f"Summary data for date {date}:")
        # print(f"Times: {times}")
        # print(f"Winds: {winds}")
        # print(f"Alts: {alts}")

        for i in range(len(times)):
            start9 = int(times[i][0:5])
                
            if start9 == start:
                index1_start = i
                break
        
    
        for i in range(len(times)):
            end9 = int(times[i][0:5])
            if end9 == end:
                index1_end = i
                # print(f"Found end index {index1_end}")
                break
        # print(f"Index for start time: {index1_start}, Index for end time: {index1_end}")

        
        if index1_start == None:
                
            winds9_mean = np.nan
            alts9_mean = np.nan
        if index1_end == None:
            # print(f"Wind speeds: {winds9}")
            # print(f"Altitudes: {alts9}")
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

        all_BCB_means.append(wind_alt) 
        
    master_BCB.append(all_BCB_means) 

#%%

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

dates_CAS = ['2022-06-02', 
              '2022-06-03', '2022-06-05','2022-06-07', '2022-06-08', 
               '2022-06-10','2022-06-13',
             '2022-06-17', '2022-06-18']
#'2022-06-11','2022-06-14',
for date in dates_CAS:
    datestr = date.replace('-', '')
    fname_CAS = sorted(glob.glob(f'/home/disk/eos4/kathem24/activate/data/cloudaerospect/2022csv/ACTIVATE-LARGE-CAS_HU25_{datestr}_R*.csv'), reverse=True)
    # print(date)
    # print(fname_CAS)

    run = 1
    for file_path in fname_CAS:
        nums_file_paths = len(fname_CAS)

        if date  >= ('2022-06-02'):
            df_CAS = pd.read_csv(file_path, skiprows= 72, quoting=csv.QUOTE_NONE)
       
        
        
        for bin_ in bin_name:
            if bin_ in df_CAS.columns:
                df_CAS.columns = df_CAS.columns.str.strip('"')
                df_CAS[bin_] = pd.to_numeric(df_CAS[bin_], errors='coerce')
                df_CAS.replace([-9999, -9999.00], np.NaN, inplace=True)
        for col in ['Time_mid', 'LWC_CAS','CAS_Bin00', 'CAS_Bin01', 'CAS_Bin02', 'CAS_Bin03', 'CAS_Bin04', 'CAS_Bin05', 'CAS_Bin06',
            'CAS_Bin07', 'CAS_Bin08', 'CAS_Bin09', 'CAS_Bin10', 'CAS_Bin11',
            'CAS_Bin12', 'CAS_Bin13', 'CAS_Bin14', 'CAS_Bin15', 'CAS_Bin16', 'CAS_Bin17', 
            'CAS_Bin18', 'CAS_Bin19', 'CAS_Bin20', 'CAS_Bin21', 'CAS_Bin22', 'CAS_Bin23', 'CAS_Bin24', 'CAS_Bin25', 'CAS_Bin26',
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
# leg_info_filtered = []
# for leg in leg_info:
#     if not (leg['Date'] == '2022-06-13' and leg['BCB_stop'] == 68749):
#         leg_info_filtered.append(leg)

# # Print leg_info or use it as needed
# for leg in leg_info:
#     print(f"Date: {leg['Date']}, Start: {leg['BCB_start']}, Stop: {leg['BCB_stop']}, LWC Labels: {leg['LWC_Labels']}")

# master_CAS_BCB_filtered = []
# for flight_data in master_CAS_BCB:
#     filtered_flight_data = []
#     for bin_means in flight_data:
#         if not (bin_means['Date'] == '2022-06-13' and bin_means['BCB_stop'] == 68749):
#             filtered_flight_data.append(bin_means)
#     master_CAS_BCB_filtered.append(filtered_flight_data)


#%%

#%%
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

                for bin_label in range(12, 30):
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
        for bin_label in range(12, 30):
            for label in ['Y', 'N']:
                bin_key = f'Bin{bin_label}_{label}_mean'
                mean_value = np.nanmean(bin_means[bin_key])
                print(f"   Bin{bin_label}_{label} Mean: {mean_value}")
    

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

            Y_calc[bin_key_Y] = np.nanmean(bin_means[bin_key_Y]) * bin_log[bin_label - 12]
            N_calc[bin_key_N] = np.nanmean(bin_means[bin_key_N]) * bin_log[bin_label - 12]

        Y_BCB_calc.append(Y_calc)
        N_BCB_calc.append(N_calc)
#Y_BCB_calc and N_BCB_calc are 91 in length without 6/11 and 6/13 but before removing the 7
#5 dusty individual legs. 
#%%

#%%
def summary_bin_means(data):
    summary_list = []

    for flight_dict in data:
        total_summary = sum(flight_dict.get(f'Bin{bin_label}_Y_mean', 0) for bin_label in range(12, 30))
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
# N_sum_list = sum_bin_means(N_BCB_calc)
#%%
def sum_bin_means(data):
    sum_list = []

    for flight_dict in data:
        total_sum = sum(flight_dict.get(f'Bin{bin_label}_N_mean', 0) for bin_label in range(12, 30))
        sum_dict = {
            'Date': flight_dict.get('Date', ''),
            'BCB_start': flight_dict.get('BCB_start', ''),
            'BCB_stop': flight_dict.get('BCB_stop', ''),
            'Sum': total_sum
        }
        sum_list.append(sum_dict)

    return sum_list

# Apply the function to Y_BCB_calc and N_BCB_calc
N_sum_list = sum_bin_means(N_BCB_calc)
# N_sum_list = sum_bin_means(N_BCB_calc)

#Y_sum_list and N_sum_list are 91 in length without 6/11 and 6/13 but before removing the 7
#5 dusty individual legs.

#%%

#%%
wind_mapping = {date: wind_speed for date, wind_speed in zip(corrected_calc_bcb['Date'], corrected_calc_bcb['Corrected_bcb_windspeed'])}

# Extract data for the scatterplot
dates = [entry['Date'] for entry in Y_sum_list]
sum_values = [entry['Sum'] for entry in Y_sum_list]
corrected_windspeeds = [wind_mapping.get(date, np.nan) for date in dates]

# Create the scatterplot
plt.scatter(sum_values, corrected_windspeeds)
plt.title('Below Cloud Base June 2022', fontsize=14, fontweight='bold')
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
plt.title('Below Cloud Base June 2022', fontsize=14, fontweight='bold')
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
plt.title('Below Cloud Base June 2022', fontsize=14, fontweight='bold')
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
plt.title('Below Cloud Base June 2022', fontsize=14, fontweight='bold')
plt.xlabel('Total droplet concentration dN (per cm^3) for cloudy skies ', fontsize=14, fontweight='bold')
plt.ylabel('Mean windspeed (m/s)', fontsize=14, fontweight='bold')
plt.show()

#%%

import numpy as np
import statsmodels.api as sm
from scipy.stats import linregress
#%%
Y_sum_values = [entry['Sum'] for entry in Y_sum_list]

X_filtered = np.array(corrected_windspeeds)
Y_filtered = np.array(Y_sum_values)

# Add a constant to the independent variable (windspeeds)
X_filtered = sm.add_constant(X_filtered)

# Fit the regression model
model = sm.OLS(Y_filtered, X_filtered).fit()

# Print the summary of the regression model
print(model.summary())
#%%
#%%
from scipy.stats import linregress

# Convert data to numpy arrays
X = np.array(corrected_windspeeds)
Y = np.array(Y_sum_values)

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(X, Y)

# Print the results
print("Slope:", slope)
print("Intercept:", intercept)
print("R-squared:", r_value**2)
print("P-value:", p_value)
print("Standard error:", std_err)



#%%

# Create a dictionary to map dates to corrected wind speeds
wind_mapping = dict(zip(corrected_calc_bcb['Date'], corrected_calc_bcb['Corrected_bcb_windspeed']))

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


#%%
# THESE GIVE THE SAME OUTPUT#



# %%

# %%
# # Find the index of the entry corresponding to the date '6/10'
# index_6_10 = next((i for i, entry in enumerate(Y_sum_list) if entry['Date'] == '2022-06-10'), None)

# # Initialize Y_sum_list_modified as a copy of Y_sum_list
# Y_sum_list_modified = Y_sum_list.copy()

# # Exclude legs 5, 6, 7, and 8 of 6/10
# if index_6_10 is not None:
#     excluded_indices_6_10 = [index_6_10 - 4, index_6_10 - 5, index_6_10 - 6, index_6_10 - 7]
#     excluded_indices_6_10 = [i for i in excluded_indices_6_10 if i >= 0]  # Ensure indices are not negative
#     print("Excluded indices for 2022-06-10:", excluded_indices_6_10)
#     Y_sum_list_modified = [entry for i, entry in enumerate(Y_sum_list_modified) if i not in excluded_indices_6_10]


# # Find the index of the entry corresponding to the date '6/13'
# index_6_13 = next((i for i, entry in enumerate(Y_sum_list_modified) if entry['Date'] == '2022-06-13'), None)

# if index_6_13 is not None:
#     excluded_indices_6_13 = [index_6_13 - 1]
#     excluded_indices_6_13 = [i for i in excluded_indices_6_13 if i >= 0]  # Ensure indices are not negative
#     print("Excluded indices for 2022-06-13:", excluded_indices_6_13)
#     Y_sum_list_modified = [entry for i, entry in enumerate(Y_sum_list_modified) if i not in excluded_indices_6_13]

# # Extract data specifically from Y_sum_list_modified
# dates_Y = [entry['Date'] for entry in Y_sum_list_modified]
# sum_values_Y = [entry['Sum'] for entry in Y_sum_list_modified]
# corrected_windspeeds_Y = [wind_mapping.get(date, np.nan) for date in dates_Y]

# # Define the independent variable (mean wind speed) and the dependent variable (droplet concentration) for Y_sum_list only
# X_Y = corrected_windspeeds_Y  # Independent variable for Y_sum_list
# y_Y = sum_values_Y  # Dependent variable for Y_sum_list

# # Add a constant to the independent variable (required for statsmodels)
# X_Y = sm.add_constant(X_Y)

# # Fit the linear regression model for Y_sum_list only
# model_Y = sm.OLS(y_Y, X_Y).fit()

# # Print the summary of the regression model for Y_sum_list only
# print(model_Y.summary())



# %%
# # Find the index of the entry corresponding to the date '6/10'
# index_6_10 = next((i for i, entry in enumerate(Y_sum_list) if entry['Date'] == '2022-06-10'), None)

# # Initialize Y_sum_list_modified as a copy of Y_sum_list
# Y_sum_list_modified = Y_sum_list.copy()

# # Exclude legs 5, 6, 7, and 8 of 6/10
# if index_6_10 is not None:
#     excluded_indices_6_10 = [index_6_10 - 4, index_6_10 - 5, index_6_10 - 6, index_6_10 - 7]
#     excluded_indices_6_10 = [i for i in excluded_indices_6_10 if i >= 0]  # Ensure indices are not negative
#     print("Excluded indices for 2022-06-10:", excluded_indices_6_10)
#     Y_sum_list_modified = [entry for i, entry in enumerate(Y_sum_list_modified) if i not in excluded_indices_6_10]

#     # Exclude corresponding entries from corrected_windspeeds_Y
#     corrected_windspeeds_Y = [entry for i, entry in enumerate(corrected_windspeeds_Y) if i not in excluded_indices_6_10]

# # Find the index of the entry corresponding to the date '6/13'
# index_6_13 = next((i for i, entry in enumerate(Y_sum_list_modified) if entry['Date'] == '2022-06-13'), None)

# if index_6_13 is not None:
#     excluded_indices_6_13 = [index_6_13 - 1]
#     excluded_indices_6_13 = [i for i in excluded_indices_6_13 if i >= 0]  # Ensure indices are not negative
#     print("Excluded indices for 2022-06-13:", excluded_indices_6_13)
#     Y_sum_list_modified = [entry for i, entry in enumerate(Y_sum_list_modified) if i not in excluded_indices_6_13]

#     # Exclude corresponding entries from corrected_windspeeds_Y
#     corrected_windspeeds_Y = [entry for i, entry in enumerate(corrected_windspeeds_Y) if i not in excluded_indices_6_13]

# dates_Y = [entry['Date'] for entry in Y_sum_list_modified]
# sum_values_Y = [entry['Sum'] for entry in Y_sum_list_modified]
# corrected_windspeeds_Y = [wind_mapping.get(date, np.nan) for date in dates_Y]

# # Define the independent variable (mean wind speed) and the dependent variable (droplet concentration) for Y_sum_list only
# X_Y = corrected_windspeeds_Y  # Independent variable for Y_sum_list
# y_Y = sum_values_Y  # Dependent variable for Y_sum_list

# # Add a constant to the independent variable (required for statsmodels)
# X_Y = sm.add_constant(X_Y)

# # Fit the linear regression model for Y_sum_list only
# model_Y = sm.OLS(y_Y, X_Y).fit()

# # Print the summary of the regression model for Y_sum_list only
# print(model_Y.summary())

# %%
# # Find the index of the entry corresponding to the date '6/10'
# index_6_10 = next((i for i, entry in enumerate(Y_sum_list) if entry['Date'] == '2022-06-10'), None)

# # Initialize Y_sum_list_modified as a copy of Y_sum_list
# Y_sum_list_modified = Y_sum_list.copy()

# # Exclude legs 5, 6, 7, and 8 of 6/10
# if index_6_10 is not None:
#     excluded_indices_6_10 = [index_6_10 - 4, index_6_10 - 5, index_6_10 - 6, index_6_10 - 7]
#     excluded_indices_6_10 = [i for i in excluded_indices_6_10 if i >= 0]  # Ensure indices are not negative
#     print("Excluded indices for 2022-06-10:", excluded_indices_6_10)
#     Y_sum_list_modified = [entry for i, entry in enumerate(Y_sum_list_modified) if i not in excluded_indices_6_10]

#     # Exclude corresponding entries from corrected_windspeeds_Y
#     corrected_windspeeds_Y = [entry for i, entry in enumerate(corrected_windspeeds_Y) if i not in excluded_indices_6_10]

# # Find the index of the entry corresponding to the date '6/13'
# index_6_13 = next((i for i, entry in enumerate(Y_sum_list_modified) if entry['Date'] == '2022-06-13'), None)

# if index_6_13 is not None:
#     excluded_indices_6_13 = [index_6_13 - 1]
#     excluded_indices_6_13 = [i for i in excluded_indices_6_13 if i >= 0]  # Ensure indices are not negative
#     print("Excluded indices for 2022-06-13:", excluded_indices_6_13)
#     Y_sum_list_modified = [entry for i, entry in enumerate(Y_sum_list_modified) if i not in excluded_indices_6_13]

#     # Create filtered wind speeds for '6/13'
#     dates_Y_6_13 = [entry['Date'] for entry in Y_sum_list_modified]
#     sum_values_Y_6_13 = [entry['Sum'] for entry in Y_sum_list_modified]
#     corrected_windspeeds_Y_6_13 = [wind_mapping.get(date, np.nan) for date in dates_Y_6_13]

#     # Define the independent variable (mean wind speed) and the dependent variable (droplet concentration) for '6/13' only
#     X_Y_6_13 = corrected_windspeeds_Y_6_13  # Independent variable for '6/13'
#     y_Y_6_13 = sum_values_Y_6_13  # Dependent variable for '6/13'

#     # Add a constant to the independent variable (required for statsmodels)
#     X_Y_6_13 = sm.add_constant(X_Y_6_13)

#     # Fit the linear regression model for '6/13' only
#     model_Y_6_13 = sm.OLS(y_Y_6_13, X_Y_6_13).fit()

#     # Print the summary of the regression model for '6/13' only
#     print(model_Y_6_13.summary())


# %%
# Specify the indices to exclude
indices_to_exclude = [60, 61, 62, 63, 77]

# Exclude the specified indices from Y_sum_list and corrected_windspeeds
Y_sum_list_modified = [entry for i, entry in enumerate(Y_sum_list) if i not in indices_to_exclude]
corrected_windspeeds_Y_modified = [entry for i, entry in enumerate(corrected_windspeeds) if i not in indices_to_exclude]

# Define the independent variable (mean wind speed) and the dependent variable (droplet concentration)
X_Y_modified = corrected_windspeeds_Y_modified
y_Y_modified = [entry['Sum'] for entry in Y_sum_list_modified]

# Add a constant to the independent variable (required for statsmodels)
X_Y_modified = sm.add_constant(X_Y_modified)

# Fit the linear regression model
model_Y_modified = sm.OLS(y_Y_modified, X_Y_modified).fit()

# Print the summary of the regression model
print(model_Y_modified.summary())

# %%

# %%



























#%%
# {'Date': '2022-06-18',
#   'BCB_start': 48588,
#   'BCB_stop': 48818,
#   'Sum': 8.120861458216073}]
#%%
time_CAS_BCB = []
legs_info = []

# Filter data for '2022-02-26'
filtered_indices = [i for i, date in enumerate(dates_legs) if date == '2022-06-18']

# print("Processing data for date '2022-02-26'")

for i in filtered_indices:
    date = dates_legs[i]
    leg_dict = leg_data[i]

    BCB_start_list = (leg_dict['LegIndex_02']['StartTimes'])
    BCB_stop_list = (leg_dict['LegIndex_02']['StopTimes'])
    # print(f"Leg at date {date} has start time: {BCB_start}, stop time: {BCB_stop}")
    if BCB_start_list and BCB_stop_list:
        # Get the first start and stop times
        BCB_start = int(BCB_start_list[3])
        BCB_stop = int(BCB_stop_list[3])
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

plt.title('2022-06-18\n 48588 - 48818 seconds after midnight\n Clear sky', fontsize=16, fontweight='bold')
plt.show()
# %%
lwc_CAS_BCB = []


# Filter data for '2022-02-26'
filtered_indices = [i for i, date in enumerate(dates_legs) if date == '2022-06-18']

# print("Processing data for date '2022-02-26'")

for i in filtered_indices:
    date = dates_legs[i]
    leg_dict = leg_data[i]

    BCB_start_list = (leg_dict['LegIndex_02']['StartTimes'])
    BCB_stop_list = (leg_dict['LegIndex_02']['StopTimes'])
    # print(f"Leg at date {date} has start time: {BCB_start}, stop time: {BCB_stop}")
    if BCB_start_list and BCB_stop_list:
        # Get the first start and stop times
        BCB_start = int(BCB_start_list[3])
        BCB_stop = int(BCB_stop_list[3])
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
plt.plot(x_values, lwc_Y_values, label='CAS_LWC', color='blue')

plt.xlabel('Seconds after midnight', fontsize=16, fontweight='bold')
plt.ylabel('Clear LWC', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True)
# plt.yscale('log')
plt.xlim(BCB_start, BCB_stop)  # Set the limits of the x-axis

plt.title('2022-06-18\n 48588 - 48818 seconds after midnight\n Clear sky', fontsize=16, fontweight='bold')
plt.show()

# %%
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
#%%
    
#%%
bin_center=[0.555, 0.645, 0.715, 0.785, 0.855, 0.925, 
            0.995, 1.07, 1.14, 1.21, 1.38, 1.75, 2.25, 
            2.75, 3.25, 3.75, 4.5, 5.75, 6.85, 7.55, 
            9.05, 11.4, 13.8, 17.5, 22.5, 27.5, 32.5, 
            37.5, 42.5, 47.5]
#%%

#%%
B_00=((0.555)**3) / (10000)
B_01= ((0.645)**3) / (10000)
B_02=((0.715)**3)/ (10000)
B_03=((0.785)**3)/  (10000)
B_04=((0.855)**3) / (10000)
B_05=((0.925)**3)/ (10000)
B_06=((0.995)**3) / (10000)
#%%
B_07=((1.07)**3)/(10000)
B_08=((1.14)**3)/(10000)
B_09=((1.21)**3)/(10000)
B_10=((1.38)**3)/(10000)
B_11=((1.75)**3)/(10000)
B_12=((2.25)**3)/(10000)
B_13=((2.75)**3)/(10000)
B_14=((3.25)**3)/(10000)
B_15=((3.75)**3)/(10000)
#%%
B_16=((4.5)**3)/(10000)
B_17=((5.75)**3)/(10000)
B_18=((6.85)**3)/(10000)
B_19=((7.55)**3)/(10000)
B_20=((9.05)**3)/(10000)
B_21=((11.4)**3)/(10000)
B_22=((13.8)**3)/(10000)
B_23=((17.5)**3)/(10000)
B_24=((22.5)**3)/(10000)
#%%
B_25=((27.5)**3) /(10000)
B_26=((32.5) **3)/ (10000)
B_27=( (37.5)**3)/ (10000)
B_28=((42.5) **3)/ (10000)
B_29=((47.5)**3)/ (10000)

radius=[B_00, B_01, B_02, B_03, B_04, B_05, B_06, B_07, B_08,
        B_09, B_10, B_11, B_12, B_13, B_14, B_15, B_16, B_17, B_18,
        B_19,B_20, B_21,B_22, B_23, B_24, B_25, B_26, B_27, B_28,
        B_29]
#%%
C_values=[C_0, C_01, C_02, C_03, C_04, C_05, C_06, C_07, C_08, C_09, 
         C_10, C_11, C_12, C_13, C_14, C_15, C_16, C_17, C_18, 
         C_19, C_20, C_21, C_22, C_23, C_24, C_25, C_26, C_27, C_28, C_29]
#%%
time_CAS_BCB_full = []
legs_info_full = []

# Filter data for '2022-02-26'
filtered_indices = [i for i, date in enumerate(dates_legs) if date == '2022-06-18']

# print("Processing data for date '2022-02-26'")

for i in filtered_indices:
    date = dates_legs[i]
    leg_dict = leg_data[i]

    BCB_start_list = (leg_dict['LegIndex_02']['StartTimes'])
    BCB_stop_list = (leg_dict['LegIndex_02']['StopTimes'])
    # print(f"Leg at date {date} has start time: {BCB_start}, stop time: {BCB_stop}")
    if BCB_start_list and BCB_stop_list:
        # Get the first start and stop times
        BCB_start = int(BCB_start_list[3])
        BCB_stop = int(BCB_stop_list[3])
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
            bin_center_value = bin_center[bin_label]
            
            # Calculate A and F for each bin
            A = (N_value * C_values[bin_label]) / 100
            B = (bin_center_value ** 3) / 10000
            F = A * ((4 / 3) * 3.14) * B
            
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
#%%
plt.figure(figsize=(10, 6))
plt.plot(x_values, lwc_Y_values, label='CAS_LWC', color='blue')
plt.xlabel('Seconds after midnight', fontsize=16, fontweight='bold')
plt.ylabel('Clear LWC', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True)
plt.xlim(BCB_start, BCB_stop)
plt.title('2022-06-18\n 48588 - 48818 seconds after midnight\n Clear sky', fontsize=16, fontweight='bold')

# Plotting sums_list on the same graph
plt.plot(x_values, sums_list, label='Calculated LWC', color='red')  # Add this line
plt.legend(fontsize=12)  # Add this line to display legend for the new plot

plt.show()














# %%
