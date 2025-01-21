#%%

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
#%%

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


bin_log= [C_12, C_13, C_14, C_15, C_16, C_17, C_18, C_19, 
         C_20, C_21, C_22, C_23, C_24, C_25,C_26, C_27, C_28, C_29]
         
         
        # C_13, C_14, C_15, C_16, 
        # C_17, C_18, C_19, C_20, C_21, C_22, C_23, C_24, C_25, C_26, C_27, C_28, C_29]
#C_0, C_01, C_02, C_03, C_04, C_05, C_06, C_07, C_08, C_09, C_10, C_11,

#C_0, C_01, C_02, C_03, C_04, C_05, C_06, C_07, C_08, C_09, C_10, C_11,
bin_center=[ 2.25, 2.75, 3.25, 3.75, 4.5, 5.75, 6.85, 7.55, 9.05, 
             11.4, 13.8, 17.5, 22.5, 27.5, 32.5, 
            37.5, 42.5, 47.5]
#2.25, 2.75, 3.25, 3.75, 4.5, 5.75, 6.85, 7.55, 9.05
# 3.25, 3.75, 4.5, 5.75, 6.85, 7.55, 
#             9.05, 11.4, 13.8, 17.5, 22.5, 27.5, 32.5, 
#             37.5, 42.5, 47.5]
#0.555, 0.645, 0.715, 0.785, 0.855, 0.925, 0.995, 1.07, 1.14, 1.21, 1.38, 1.75,
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


Logg = [F12, F13, F14, F15, F16, F17, F18, F19, F20, F21, F22,
        F23, F24, F25, F26, F27, F28, F29]
Logg = np.array(Logg)


# %%
#CAS data

bin_name = ['CAS_Bin12', 'CAS_Bin13', 'CAS_Bin14', 'CAS_Bin15', 
            'CAS_Bin16', 'CAS_Bin17', 
            'CAS_Bin18', 'CAS_Bin19', 'CAS_Bin20', 'CAS_Bin21', 'CAS_Bin22', 
            'CAS_Bin23', 'CAS_Bin24', 'CAS_Bin25', 'CAS_Bin26',
            'CAS_Bin27', 'CAS_Bin28', 'CAS_Bin29']

leg_data = []
leg_name=['Time_Start', '  Time_Stop', '  Julian_Day', 
          '  Date', '  LegIndex']

dates_legs= ['2022-05-05']

for date in dates_legs:
    datestr = date.replace('-', '')
    fname_legs = sorted(glob.glob(f'/home/disk/eos4/kathem24/activate/data/MET/2022/LegFLags/csv/ACTIVATE-LegFlags_HU25_{datestr}_R*.csv'), reverse=True)

    leg_dictionary = {
        'Date': date,
        'LegIndex_06': {'StartTimes': [], 'StopTimes': []},
    }

    for file_path in fname_legs:
        if date == '2022-05-05':
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
# print(f"Before filtering:\n{df_legs[['  LegIndex', 'Time_Start']]}")
        leg_index_06 = df_legs[df_legs['  LegIndex'] % 100 == 6]
        
        leg_dictionary['LegIndex_06']['StartTimes'].extend(leg_index_06['Time_Start'].tolist())
        leg_dictionary['LegIndex_06']['StopTimes'].extend(leg_index_06['  Time_Stop'].tolist())
    #print(leg_data)
    leg_data.append(leg_dictionary)


#%%

datasets = []
dates_CAS = ['2022-05-05']
clear_means = [] 

for date in dates_CAS:
    print(date)
    

    dataset = {'Date': date, 'Clear Means': [], 'Cloud Means': []}  # Initialize a dataset dictionary
    datestr = date.replace('-', '')
    fname_CAS = sorted(glob.glob(f'/home/disk/eos4/kathem24/activate/data/cloudaerospect/2022csv/ACTIVATE-LARGE-CAS_HU25_{datestr}_R*.csv'), reverse=True)
    
    # print(f"Date: {date}")
    # print(f"Files: {fname_CAS}")
    run = 1
    for file_path in fname_CAS:
        nums_file_paths = len(fname_CAS)

    # num = index = dates_CAS.index(date)

        if date == ('2022-05-05'):
            df_CAS = pd.read_csv(file_path, skiprows= 72, quoting=csv.QUOTE_NONE)
    # else:
    #     df_CAS = pd.read_csv(file_path, skiprows= 71, quoting=csv.QUOTE_NONE)

    # clear_bin_means = []  # Initialize clear bin means for this dataset
    # cloud_bin_means = []  # Initialize cloud bin means for this dataset

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
        # print(date)
        # print("yes")
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
                datasets.append(df_CAS)
                break

        if nums_file_paths ==1:
            datasets.append(df_CAS)

        run = run+1


    # Change left end of dataset to "Time_mid"
    df_CAS = df_CAS.reset_index().set_index('Time_mid')

    # Creating a list that will replace the "index" column 
    new_col = [] 
    for i in range(len(df_CAS)):
        new_col.append(i)
    # Change list to array
    new_col = np.array(new_col)

    # Replace the "index" column with the correct indicies (0 -> 23017)
    n = df_CAS.columns[0]
    df_CAS[n] = new_col

    # Change left end of dataset back to "index"
    df_CAS = df_CAS.reset_index().set_index('index')

     
    #Accessing the stuff in your dictionary 
    flight_date = leg_dictionary['Date'] #Get date of flight from dictionary 
    ACB_Start = leg_dictionary['LegIndex_06']['StartTimes']
    ACB_Stop = leg_dictionary['LegIndex_06']['StopTimes']

    final_cloud_conc=[]
    final_clear_conc=[]
    for i in range(len(ACB_Start)):
        start_leg = ACB_Start[i]
        end_leg = ACB_Stop[i]

        start_ind = np.where(df_CAS['Time_mid']==start_leg)[0][0]
        end_ind = np.where(df_CAS['Time_mid']==end_leg)[0][0]

        leg = df_CAS[start_ind:end_ind]
        print(f"Leg {i+1}:")
        print(f"Total LWC contents: {len(leg['LWC_CAS'])}")

    # Count clear and cloudy LWC contents
        clear_count = sum(1 for lwc_val in leg['LWC_CAS'] if 0 <= lwc_val <= 0.01)
        cloudy_count = sum(1 for lwc_val in leg['LWC_CAS'] if lwc_val > 0.01)

        print(f"Clear LWC contents: {clear_count} ({(clear_count / len(leg['LWC_CAS'])) * 100:.2f}% of total)")
        print(f"Cloudy LWC contents: {cloudy_count} ({(cloudy_count / len(leg['LWC_CAS'])) * 100:.2f}% of total)")
        # print(f"Length of LWC contents in leg {i+1}: {len(leg['LWC_CAS'])}")
    #     print(f"Leg {i+1}:")
    #     print(f"Total LWC contents: {len(leg['LWC_CAS'])}")

    # # Count clear and cloudy LWC contents
    #     clear_count = sum(1 for lwc_val in leg['LWC_CAS'] if 0 <= lwc_val <= 0.01)
    #     cloudy_count = sum(1 for lwc_val in leg['LWC_CAS'] if lwc_val > 0.01)

    #     print(f"Clear LWC contents: {clear_count}")
    #     print(f"Cloudy LWC contents: {cloudy_count}")
        lwc_clear = []
        lwc_cloud = []

        for i in range(start_ind, end_ind): 
            lwc_val = leg['LWC_CAS'][i]
            if 0 <= lwc_val <= 0.01: # check if LWC_val clear
                lwc_clear.append(lwc_val)
            clear_bins = np.zeros((len(lwc_clear),len(bin_name))) # create clear array with 30 columns for each bin
            if 0.01<lwc_val: # check if LWC_val cloud
                lwc_cloud.append(lwc_val)
            cloud_bins = np.zeros((len(lwc_cloud),len(bin_name))) # create cloud array
        index7 = 0
        index8 = 0
        for i in range(start_ind, end_ind):
            lwc_val = leg['LWC_CAS'][i]
            if 0 <= lwc_val <= 0.01: # check if LWC_val clear
                for j in range(len(bin_name)):
                    conc = leg[(bin_name[j])][i] # pull num conc. from jth bin
                    clear_bins[index7,j] = conc
                index7 = index7+1 
            if 0.01<lwc_val: # check if LWC_val cloud
                for j in range(len(bin_name)):
                    conc = leg[(bin_name[j])][i] # pull num conc. from jth bin
                    cloud_bins[index8,j] = conc
                index8 = index8+1

        clear_means = []
        cloud_means = []

        for j in range(len(bin_name)):
            mean = np.nanmean(clear_bins[:,j])
            clear_means.append(mean)

        for j in range(len(bin_name)):
            mean = np.nanmean(cloud_bins[:,j])
            cloud_means.append(mean)

        clear_means = np.array(clear_means)
        cloud_means = np.array(cloud_means)

        log_clear_means =[] 
        log_cloud_means=[]

        for i in range(len(clear_means)):
            # Multiply each value in clear_means and cloud_means by the corresponding value in Logg
                log_clear_means.append((Logg[i] * clear_means[i]))
                log_cloud_means.append((Logg[i]* cloud_means[i]))

        log_cloud_means=np.array(log_cloud_means)
        log_clear_means=np.array(log_clear_means)

        final_cloud_conc.append(log_cloud_means)
        final_clear_conc.append(log_clear_means)




#%%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(bin_center,  final_clear_conc[0], label='Leg 1', color='blue')
ax2.scatter(bin_center, final_cloud_conc[0], label='Leg 1', color='blue')
ax1.scatter(bin_center,  final_clear_conc[1], label='Leg 2', color='purple')
ax2.scatter(bin_center, final_cloud_conc[1], label='Leg 2', color='purple')
ax1.scatter(bin_center,  final_clear_conc[2], label='Leg 3', color='green')
ax2.scatter(bin_center, final_cloud_conc[2], label='Leg 3', color='green')
ax1.scatter(bin_center,  final_clear_conc[3], label='Leg 4', color='pink')
ax2.scatter(bin_center, final_cloud_conc[3], label='leg 4', color='pink')
ax1.scatter(bin_center,  final_clear_conc[4], label='Leg 5', color='brown')
ax2.scatter(bin_center, final_cloud_conc[4], label='Leg 5', color='brown')
ax1.scatter(bin_center,  final_clear_conc[5], label='Leg 6', color='black')
ax2.scatter(bin_center, final_cloud_conc[5], label='Leg 6', color='black')
ax1.scatter(bin_center,  final_clear_conc[6], label='Leg 7', color='darkred')
ax2.scatter(bin_center, final_cloud_conc[6], label='Leg 7', color='darkred')
ax1.scatter(bin_center,  final_clear_conc[7], label='Leg 8', color='darkgreen')
ax2.scatter(bin_center, final_cloud_conc[7], label='Leg 8', color='darkgreen')
ax1.scatter(bin_center,  final_clear_conc[8], label='Leg 9', color='darkred')
ax2.scatter(bin_center, final_cloud_conc[8], label='Leg 9', color='darkred')
ax1.scatter(bin_center,  final_clear_conc[9], label='Leg 10', color='darkgreen')
ax2.scatter(bin_center, final_cloud_conc[9], label='Leg 10', color='darkgreen')
ax1.scatter(bin_center,  final_clear_conc[10], label='Leg 11', color='grey')
ax2.scatter(bin_center, final_cloud_conc[10], label='Leg 11', color='grey')



ax1.set_xlim([10**0.1, 10**2])
ax1.set_ylim([10**-5.6, 10**2])
ax2.set_xlim([10**0.1, 10**2])
ax2.set_ylim([10**-5, 10**3])
# Set plot properties for the first subplot
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_title(f'{date} Min Altitude Clear Mean Droplet\n Concentration vs Bin Size', fontsize=16, fontweight='bold')
ax1.set_ylabel('Clear Mean Droplet Concentration\n (cm^3/um)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Bin Centers Diameter (um)', fontsize=12, fontweight='bold')

# Set plot properties for the second subplot
ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.set_title(f'{date} Min Altitude Cloud Mean Droplet\n Concentration vs Bin Size for 2022', fontsize=16, fontweight='bold')
ax2.set_ylabel('Cloud Mean Droplet Concentration\n (cm^3/um)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Bin Centers Diameter (um)', fontsize=12, fontweight='bold')

ax1.legend(prop={'size': 8})
ax2.legend(prop={'size': 8})
plt.tight_layout()
plt.show()
#%%








