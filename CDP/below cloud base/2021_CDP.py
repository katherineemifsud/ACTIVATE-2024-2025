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
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy.ma as ma
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
from scipy.stats import gaussian_kde
import pickle
from scipy.integrate import quad
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D
from collections import Counter
from scipy.spatial import distance

#%%
#Now CDP data
#This is how we will correct our droplet concentration units from 
#dN/dlogD to dN/dD
#We will use the bin width to convert the units
L_00=math.log10(3)-math.log10(2)
L_01=math.log10(4)-math.log10(3)
L_02=math.log10(5)-math.log10(4)
L_03=math.log10(6)-math.log10(5)
L_04=math.log10(7)-math.log10(6)
L_05=math.log10(8)-math.log10(7)
L_06=math.log10(9)-math.log10(8)
L_07=math.log10(10)-math.log10(9)
L_08=math.log10(11)-math.log10(10)
L_09=math.log10(12)-math.log10(11)
L_10=math.log10(13)-math.log10(12)
L_11=math.log10(14)-math.log10(13)
L_12=math.log10(16)-math.log10(14)
L_13=math.log10(18)-math.log10(16)
L_14=math.log10(20)-math.log10(18)
L_15=math.log10(22)-math.log10(20)
L_16=math.log10(24)-math.log10(22)
L_17=math.log10(26)-math.log10(24)
L_18=math.log10(28)-math.log10(26)
L_19=math.log10(30)-math.log10(28)
L_20=math.log10(32)-math.log10(30)
L_21=math.log10(34)-math.log10(32)
L_22=math.log10(36)-math.log10(34)
L_23=math.log10(38)-math.log10(36)
L_24=math.log10(40)-math.log10(38)
L_25=math.log10(42)-math.log10(40)
L_26=math.log10(44)-math.log10(42)
L_27=math.log10(46)-math.log10(44)
L_28=math.log10(48)-math.log10(46)
L_29=math.log10(50)-math.log10(48)


bin_log_CDP=[L_00, L_01, L_02, L_03, L_04, L_05, L_06, L_07, L_08,
          L_09, L_10, L_11,
          L_12, L_13, L_14, L_15, L_16, 
        L_17, L_18, L_19, L_20, L_21, L_22, L_23, 
        L_24, L_25, L_26, L_27, L_28, L_29]


P00=(3-2)
P01=(4-3)
P02=(5-4)
P03=(6-5)
P04=(7-6)
P05=(8-7)
P06=(9-8)
P07=(10-9)
P08=(11-10)
P09=(12-11)
P10=(13-12)
P11=(14-13)
P12 = (16-14)
P13 = (18-16)
P14 = (20-18)
P15 = (22-20)
P16 = (24-22)
P17 = (26-24)
P18 = (28-26)
P19 = (30-28)
P20 = (32-30)
P21 = (34-32)
P22 = (36-34)
P23 = (38-36)
P24 = (40-38)
P25 = (42-40)
P26 = (44-42)
P27 = (46-44)
P28 = (48-46)
P29 = (50-48)


J00=(L_00 / P00)
J01=(L_01 / P01)
J02=(L_02 / P02)
J03=(L_03 / P03)
J04=(L_04 / P04)
J05=(L_05 / P05)
J06=(L_06 / P06)
J07=(L_07 / P07)
J08=(L_08 / P08)
J09=(L_09 / P09)
J10=(L_10 / P10)
J11=(L_11 / P11)
J12 = (L_12 / P12)
J13 = (L_13 / P13)
J14 = (L_14 / P14)
J15 = (L_15 / P15)
J16 = (L_16 / P16)
J17 = (L_17 / P17)
J18 = (L_18 / P18)
J19 = (L_19 / P19)
J20 = (L_20 / P20)
J21 = (L_21 / P21)
J22 = (L_22 / P22)
J23 = (L_23 / P23)
J24 = (L_24 / P24)
J25 = (L_25 / P25)
J26 = (L_26 / P26)
J27 = (L_27 / P27)
J28 = (L_28 / P28)
J29 = (L_29 / P29)


Logg_CDP = [J00, J01, J02, J03, J04, J05, J06, J07, J08, J09, J10, 
            J11, J12, J13, J14, J15, J16, J17, J18, J19, J20, J21,
            J22, J23, J24, J25, J26, J27, J28, J29]

Logg_CDP = np.array(Logg_CDP)
 
bin_center_CDP=[2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 
            10.5, 11.5, 12.5, 13.5, 15, 17, 19, 
            21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49]
# %%
#Summary Data and meteorological data import
col_name = [
    "Time_mid",
    "Latitude",
    "Longitude",
    "GPS_altitude",
    "Pressure_Altitude",
    "Pitch",
    "Roll",
    "True_Heading",
    "True_Air_Speed",
    "Static_Air_Temp",
    "IR_Surf_Temp",
    "Static_Pressure",
    "Wind_Speed",
    "Wind_Direction"
]
summary = []
dates_sum = [
    "2021-01-27", "2021-01-29",
    "2021-02-03", "2021-03-04", "2021-03-05", "2021-03-08",
    "2021-03-09", "2021-03-12", "2021-03-20", "2021-03-23",
    "2021-03-29", "2021-03-30", "2021-04-02", "2021-05-13",
    "2021-05-14", "2021-05-15", "2021-05-18", "2021-05-19",
    "2021-05-20", "2021-05-21", "2021-05-25", "2021-06-01",
    "2021-06-02", "2021-06-05", "2021-06-07", "2021-06-08",
    "2021-06-15", "2021-06-16", "2021-06-17", "2021-06-22",
    "2021-06-24", "2021-06-26", "2021-06-28", "2021-06-29",
    "2021-06-30"
]

def get_ict_skiprows(file_path):
    
    with open(file_path, "r") as f:
        first_line = f.readline().strip()

    try:
        header_line_count = int(first_line.split(",")[0])
    except (ValueError, IndexError):
        raise ValueError(
            f"Could not determine header length for:\n{file_path}\n"
            f"First line: {first_line}"
        )
    return header_line_count - 1
for date in dates_sum:
    datestr = date.replace("-", "")
    fname_sum = sorted(
        glob.glob(
            f"/home/disk/eos4/kathem24/activate/data/2021/"
            f"Summary/ACTIVATE-SUMMARY_HU25_{datestr}_R*.ict"
        )
    )
    if len(fname_sum) == 0:
        print(f"No Summary file found for {date}")
        continue
    flight_parts = []
    for file_path in fname_sum:
        skiprows = get_ict_skiprows(file_path)

        print(
            date,
            file_path.split("/")[-1],
            "skiprows =",
            skiprows
        )

        df_sum = pd.read_csv(
            file_path,
            skiprows=skiprows,
            quoting=csv.QUOTE_NONE
        )

        df_sum.columns = (
            df_sum.columns
            .astype(str)
            .str.replace('"', "", regex=False)
            .str.strip()
        )
        if "Time_mid" not in df_sum.columns:
            print(f"Incorrect header for {file_path}")
            print("Columns found:", df_sum.columns.tolist())
            continue
        df_sum.replace(
            [-9999, -9999.00, "-9999", "-9999.00"],
            np.nan,
            inplace=True
        )
        for col_ in col_name:
            if col_ in df_sum.columns:
                df_sum[col_] = pd.to_numeric(
                    df_sum[col_],
                    errors="coerce"
                )
        flight_parts.append(df_sum)
    if len(flight_parts) == 0:
        print(f"No valid Summary parts imported for {date}")
        continue
    df_flight = pd.concat(
        flight_parts,
        axis=0,
        ignore_index=True
    )
    df_flight = (
        df_flight
        .sort_values("Time_mid")
        .reset_index(drop=True)
    )
    summary.append(df_flight)
    print(
        f"{date}: imported {len(flight_parts)} file part(s), "
        f"{len(df_flight)} total rows, "
        f"time {df_flight['Time_mid'].min()}–"
        f"{df_flight['Time_mid'].max()}"
    )
print("\nNumber of Summary flight dates imported:", len(summary))
#%%
#Import the flight leg time stamps and leg lengths 
leg_data = []
dates_legs = [
    '2021-01-27', '2021-01-29',
    '2021-02-03', '2021-03-04', '2021-03-05', '2021-03-08', '2021-03-09',
    '2021-03-12', '2021-03-20', '2021-03-23', '2021-03-29', '2021-03-30',
    '2021-04-02', '2021-05-13', '2021-05-14', '2021-05-15', 
    '2021-05-18', '2021-05-19', '2021-05-20', '2021-05-21',
    '2021-05-25', '2021-06-01', '2021-06-02', '2021-06-05',
    '2021-06-07', '2021-06-08', '2021-06-15', '2021-06-16',
    '2021-06-17', '2021-06-22', '2021-06-24', '2021-06-26', '2021-06-28', 
    '2021-06-29', '2021-06-30'
]
def find_leg_header_row(file_path):
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if line.strip() == 'Time_Start,Time_Stop,Julian_Day,Date,LegIndex':
                return i
    raise ValueError(f"Could not find real data header in {file_path}")
for date in dates_legs:
    datestr = date.replace('-', '')
    fname_legs = sorted(glob.glob(
        f'/home/disk/eos4/kathem24/activate/data/2021/Leg Flags/ACTIVATE-LegFlags_HU25_{datestr}_R*.ict'
    ), reverse=True)
    leg_dictionary = {
        'Date': date,
        'LegIndex_02': {'StartTimes': [], 'StopTimes': []},
    }
    if len(fname_legs) == 0:
        print(f"No Leg Flags file found for {date}")
        leg_data.append(leg_dictionary)
        continue
    for file_path in fname_legs:
        skiprows = find_leg_header_row(file_path)
        df_legs = pd.read_csv(
            file_path,
            skiprows=skiprows,
            quoting=csv.QUOTE_NONE,
            skipinitialspace=True
        )
        df_legs.columns = (
            df_legs.columns
            .str.replace('"', '', regex=False)
            .str.strip()
        )
        df_legs.replace([-9999, -9999.00], np.nan, inplace=True)
        df_legs['Time_Start'] = pd.to_numeric(df_legs['Time_Start'], errors='coerce')
        df_legs['Time_Stop'] = pd.to_numeric(df_legs['Time_Stop'], errors='coerce')
        df_legs['LegIndex'] = pd.to_numeric(df_legs['LegIndex'], errors='coerce')
        df_legs.dropna(
            subset=['Time_Start', 'Time_Stop', 'LegIndex'],
            inplace=True
        )
        leg_index_02 = df_legs[df_legs['LegIndex'] % 100 == 2]
        leg_dictionary['LegIndex_02']['StartTimes'].extend(
            leg_index_02['Time_Start'].tolist()
        )
        leg_dictionary['LegIndex_02']['StopTimes'].extend(
            leg_index_02['Time_Stop'].tolist()
        )
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
    '2021-01-27', '2021-01-29',
    '2021-02-03', '2021-03-04', '2021-03-05', '2021-03-08', '2021-03-09',
    '2021-03-12', '2021-03-20', '2021-03-23', '2021-03-29', '2021-03-30',
    '2021-04-02', '2021-05-13', '2021-05-14', '2021-05-15', 
    '2021-05-18', '2021-05-19', '2021-05-20', '2021-05-21',
    '2021-05-25', '2021-06-01', '2021-06-02', '2021-06-05',
    '2021-06-07', '2021-06-08', '2021-06-15', '2021-06-16',
    '2021-06-17', '2021-06-22', '2021-06-24', '2021-06-26', '2021-06-28', 
    '2021-06-29', '2021-06-30'
]
for date in dates_twoDS:
    datestr = date.replace('-', '')
    file_paths = sorted(
        glob.glob(f'/home/disk/eos4/kathem24/activate/data/2021/2DS/ACTIVATE-2DS-H-Arm_HU25_{datestr}_R*.ict'), 
        reverse=False 
    )
    print(f"\nProcessing {date}... Found files: {file_paths}")
    dfs_for_date = []
    for file_path in file_paths:
        header_row = None
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if ('Time_Start' in line) and ('LWC_2DS' in line):
                    header_row = i
                    break
        if header_row is None:
            print(f"Could not find 2DS header row in {file_path}")
            continue
        df_2DS = pd.read_csv(
            file_path,
            skiprows=header_row,
            quoting=csv.QUOTE_NONE,
            skipinitialspace=True,
            engine='python'
        )
        df_2DS.columns = (
            df_2DS.columns
            .str.replace('"', '', regex=False)
            .str.strip()
        )
        df_2DS.replace([-9999, -9999.0], 0, inplace=True)
        for col in df_2DS.select_dtypes(include=['object']).columns:
            df_2DS[col] = (
                df_2DS[col]
                .str.replace('"', '', regex=False)
                .str.strip()
            )
        dfs_for_date.append(df_2DS)
    if len(dfs_for_date) == 2:
        df_2DS = pd.concat(dfs_for_date, ignore_index=True)
        twoDS.append(df_2DS)
    elif len(dfs_for_date) == 1:
        twoDS.append(dfs_for_date[0])
    else:
        print(f"No valid 2DS data for {date}")
print("Total dates processed:", len(twoDS))
# %%
#Import humidity data. 
col_name_h20 = ['Time_Start', 'H2O_DLH', 'RHi_DLH', 'RHw_DLH']
h20 = []
dates_h20 = [
    '2021-01-27', '2021-01-29',
    '2021-02-03', '2021-03-04', '2021-03-05', '2021-03-08', '2021-03-09',
    '2021-03-12', '2021-03-20', '2021-03-23', '2021-03-29', '2021-03-30',
    '2021-04-02', '2021-05-13', '2021-05-14', '2021-05-15', 
    '2021-05-18', '2021-05-19', '2021-05-20', '2021-05-21',
    '2021-05-25', '2021-06-01', '2021-06-02', '2021-06-05',
    '2021-06-07', '2021-06-08', '2021-06-15', '2021-06-16',
    '2021-06-17', '2021-06-22', '2021-06-24', '2021-06-26', '2021-06-28', 
    '2021-06-29', '2021-06-30'
]
def find_dlh_header_row(file_path):
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if ('Time_Start' in line) and ('H2O_DLH' in line) and ('RHw_DLH' in line):
                return i
    raise ValueError(f"Could not find DLH header in {file_path}")
for date in dates_h20:
    datestr = date.replace('-', '')
    fname_h20 = sorted(glob.glob(
        f'/home/disk/eos4/kathem24/activate/data/2021/DLH20/ACTIVATE-DLH-H2O_HU25_{datestr}_R*.ict'
    ))
    print(f"\nProcessing {date}... Found files: {fname_h20}")
    frames = []
    for file_path in fname_h20:
        skiprows = find_dlh_header_row(file_path)
        df_h20 = pd.read_csv(
            file_path,
            skiprows=skiprows,
            quoting=csv.QUOTE_NONE,
            skipinitialspace=True
        )
        df_h20.columns = (
            df_h20.columns
            .str.replace('"', '', regex=False)
            .str.strip()
        )
        df_h20.replace([-9999, -9999.00, -999.9], np.nan, inplace=True)
        for col_ in col_name_h20:
            if col_ in df_h20.columns:
                df_h20[col_] = pd.to_numeric(df_h20[col_], errors='coerce')
        frames.append(df_h20)
    if len(frames) > 1:
        df_h20_combined = pd.concat(frames, ignore_index=True)
        h20.append(df_h20_combined)
    elif len(frames) == 1:
        h20.append(frames[0])
    else:
        print(f"No DLH file found for {date}")
#%%
#%%
#Import the instrument data for the cloud droplet probe 
bin_name_CDP = ['CDP_Bin00', 'CDP_Bin01', 'CDP_Bin02', 'CDP_Bin03', 
                'CDP_Bin04', 'CDP_Bin05', 'CDP_Bin06', 'CDP_Bin07', 
                'CDP_Bin08', 'CDP_Bin09', 'CDP_Bin10', 'CDP_Bin11', 'CDP_Bin12',
                'CDP_Bin13', 'CDP_Bin14', 'CDP_Bin15', 'CDP_Bin16', 
                'CDP_Bin17', 'CDP_Bin18', 'CDP_Bin19', 'CDP_Bin20', 
                'CDP_Bin21', 'CDP_Bin22', 'CDP_Bin23', 'CDP_Bin24', 
                'CDP_Bin25', 'CDP_Bin26', 'CDP_Bin27',
                'CDP_Bin28', 'CDP_Bin29']
dates_CDP = [
    '2021-01-27', '2021-01-29',
    '2021-02-03', '2021-03-04', '2021-03-05', '2021-03-08', '2021-03-09',
    '2021-03-12', '2021-03-20', '2021-03-23', '2021-03-29', '2021-03-30',
    '2021-04-02', '2021-05-13', '2021-05-14', '2021-05-15', 
    '2021-05-18', '2021-05-19', '2021-05-20', '2021-05-21',
    '2021-05-25', '2021-06-01', '2021-06-02', '2021-06-05',
    '2021-06-07', '2021-06-08', '2021-06-15', '2021-06-16',
    '2021-06-17', '2021-06-22', '2021-06-24', '2021-06-26', '2021-06-28', 
    '2021-06-29', '2021-06-30'
]
input_dir = (
    "/home/disk/eos4/kathem24/activate/data/2021/CDP/CDP_1Hz"
)

CDP_1Hz = []
loaded_dates_CDP = []
loaded_files_CDP = []
def file_part_order(file_path):
    file_name = os.path.basename(file_path)

    if "_L1" in file_name:
        return 1
    elif "_L2" in file_name:
        return 2
    else:
        return 0


for date in dates_CDP:

    file_paths = glob.glob(
        os.path.join(
            input_dir,
            f"*{date.replace('-', '')}*_1Hz.ict"
        )
    )

    file_paths = sorted(file_paths, key=file_part_order)

    if len(file_paths) == 0:
        print(f"File not found for {date}")
        continue

    daily_frames = []

    print(f"\nDate: {date}")

    for file_path in file_paths:

        df_CDP = pd.read_csv(file_path)

        df_CDP.replace(
            [-9999, -9999.00],
            np.nan,
            inplace=True
        )

        df_CDP["Time_Start"] = pd.to_numeric(
            df_CDP["Time_Start"],
            errors="coerce"
        )

        df_CDP = df_CDP.dropna(subset=["Time_Start"])
        df_CDP["Date"] = date

        daily_frames.append(df_CDP)
        loaded_files_CDP.append(os.path.basename(file_path))

        print(
            f"Loaded {os.path.basename(file_path)}: "
            f"{df_CDP['Time_Start'].iloc[0]} to "
            f"{df_CDP['Time_Start'].iloc[-1]}"
        )
    df_CDP_date = pd.concat(
        daily_frames,
        ignore_index=True
    )
    time_diff = df_CDP_date["Time_Start"].diff()

    print(
        "Time is monotonic:",
        df_CDP_date["Time_Start"].is_monotonic_increasing
    )
    print(
        "Backward time jumps:",
        int((time_diff < 0).sum())
    )
    print(
        "Duplicate seconds:",
        int(df_CDP_date["Time_Start"].duplicated().sum())
    )

    CDP_1Hz.append(df_CDP_date)
    loaded_dates_CDP.append(date)

    print(
        f"Combined {len(file_paths)} file(s) for {date}: "
        f"{len(df_CDP_date)} rows"
    )
# %%
master_CDP_BCB = []
leg_info_CDP = []
for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict_CDP = leg_data[i]
    flight_date = leg_dict_CDP['Date']
    BCB_start = leg_dict_CDP['LegIndex_02']['StartTimes']
    BCB_stop = leg_dict_CDP['LegIndex_02']['StopTimes']

    CDP_flight = CDP_1Hz[i]
    twoDS_flight = twoDS[i]
    rh_flight = h20[i]
    CDP_flight['Time_Start'] = pd.to_numeric(CDP_flight['Time_Start'], errors='coerce')
    twoDS_flight['Time_Start'] = pd.to_numeric(twoDS_flight['Time_Start'], errors='coerce')
    rh_flight['Time_Start'] = pd.to_numeric(rh_flight['Time_Start'], errors='coerce')
    CDP_times = CDP_flight['Time_Start'].values
    CDP_lwc = CDP_flight['LWC_CDP'].values
    CDP_bins = {f'CDP_Bin{bin_label:02d}': CDP_flight[f'CDP_Bin{bin_label:02d}'].values for bin_label in range(30)}
    TwoDS_times = twoDS_flight['Time_Start'].values
    TwoDS_N_total = twoDS_flight['N-total_2DS'].values
    rh_times = rh_flight['Time_Start'].values
    rh_values = rh_flight.RHw_DLH.values
    total_BCB_means = []
    for k in range(len(BCB_start)):
        start20 = BCB_start[k]
        end20 = BCB_stop[k]
        CDP_indices_in_range = np.where((CDP_times >= start20) & (CDP_times <= end20))[0]
        TwoDS_indices_in_range = np.where((TwoDS_times >= start20) & (TwoDS_times <= end20))[0]
        rh_indices_in_range = np.where((rh_times >= start20) & (rh_times <= end20))[0]
        if CDP_indices_in_range.size > 0 and TwoDS_indices_in_range.size > 0 and rh_indices_in_range.size > 0:
            data_labels = []
            BCB_means = []
            for CDP_idx, TwoDS_idx, rh_idx in zip(CDP_indices_in_range, TwoDS_indices_in_range, rh_indices_in_range):
                lwc_val = CDP_lwc[CDP_idx]
                N_val = TwoDS_N_total[TwoDS_idx]
                rh_val = rh_values[rh_idx]
                lwc_label = 'Y' if 0 <= lwc_val <= 0.0025 else 'N'
                N_label = 'Y' if 0 <= N_val <= 100 else 'N'
                rh_label = 'Y' if 0 <= rh_val <= 95 else 'N'
                label = 'Y' if lwc_label == 'Y' and N_label == 'Y' and rh_label == 'Y' else 'N'
                if label == 'Y' and rh_val > 95:
                    print(f"RH violation: {rh_val:.2f} passed at time {CDP_times[CDP_idx]}")
                data_labels.append(label)
                bin_values = [CDP_bins[f'CDP_Bin{bin_label:02d}'][CDP_idx] for bin_label in range(30)]
                BCB_means.append(bin_values)
            if BCB_means:
                total_BCB_means.append(BCB_means)
            leg_info_CDP.append({
                'Date': date,
                'BCB_start': start20,
                'BCB_stop': end20,
                'Data_Labels': data_labels,
            })
    master_CDP_BCB.append(total_BCB_means)
for leg in leg_info_CDP:
    print(f"Date: {leg['Date']}, Start: {leg['BCB_start']}, Stop: {leg['BCB_stop']}, Data Labels: {leg['Data_Labels']}")
#%%
#Double check the number of legs associated with each date to compare across multiple instruments.  
leg_count = Counter([leg['Date'] for leg in leg_info_CDP])
print("Number of legs associated with each date:")
total_legs = 0
for date, count in sorted(leg_count.items()):
    print(f"Date: {date}, Number of Legs: {count}")
    total_legs += count
print(f"\nTotal number of legs: {total_legs}")
#%%
CDP_bin_widths = np.array([
    P00, P01, P02, P03, P04, P05, P06, P07, P08, P09,
    P10, P11, P12, P13, P14, P15, P16, P17, P18, P19,
    P20, P21, P22, P23, P24, P25, P26, P27, P28, P29
], dtype=float)
CDP_bin_centers = np.array(
    bin_center_CDP,
    dtype=float)
CDP_bin_names = [
    f"CDP_Bin{i:02d}"
    for i in range(30)
]
print("Number of CDP bins:", len(CDP_bin_names))
print("Number of Logg_CDP values:", len(Logg_CDP))
print("Number of linear bin widths:", len(CDP_bin_widths))
print("Number of bin centers:", len(CDP_bin_centers))
assert len(CDP_bin_names) == 30
assert len(Logg_CDP) == 30
assert len(CDP_bin_widths) == 30
assert len(CDP_bin_centers) == 30
#%%
# Flatten master_CDP_BCB so that its legs align with leg_info_CDP
flat_CDP_legs = []
for flight_legs in master_CDP_BCB:
    for leg_values in flight_legs:
        flat_CDP_legs.append(leg_values)
print("Legs stored in master_CDP_BCB:", len(flat_CDP_legs))
print("Legs stored in leg_info_CDP:", len(leg_info_CDP))
if len(flat_CDP_legs) != len(leg_info_CDP):
    print(
        "WARNING: master_CDP_BCB and leg_info_CDP "
        "do not contain the same number of legs.")
#%%
# Match CDP dataframes explicitly by date rather than only by list index
CDP_by_date = {
    date: dataframe
    for date, dataframe in zip(
        loaded_dates_CDP,
        CDP_1Hz)}
#%%
CDP_raw_diagnostics = []
number_of_matching_legs = min(
    len(flat_CDP_legs),
    len(leg_info_CDP))
for global_leg_index in range(number_of_matching_legs):
    leg_values = np.asarray(
        flat_CDP_legs[global_leg_index],
        dtype=float)
    leg_information = leg_info_CDP[
        global_leg_index]
    date = leg_information["Date"]
    start_time = leg_information["BCB_start"]
    stop_time = leg_information["BCB_stop"]
    data_labels = np.asarray(
        leg_information["Data_Labels"],
        dtype=object)
    if leg_values.ndim != 2:
        print(
            f"Unexpected shape for {date}, "
            f"{start_time}-{stop_time}: "
            f"{leg_values.shape}"
        )
        continue
    if leg_values.shape[1] != 30:
        print(
            f"Wrong number of bins for {date}, "
            f"{start_time}-{stop_time}: "
            f"{leg_values.shape[1]}"
        )
        continue
    if len(data_labels) != len(leg_values):
        print(
            f"LABEL LENGTH MISMATCH for {date}, "
            f"{start_time}-{stop_time}: "
            f"{len(data_labels)} labels versus "
            f"{len(leg_values)} CDP rows"
        )
    number_of_rows = min(
        len(leg_values),
        len(data_labels)
    )

    leg_values = leg_values[:number_of_rows]
    data_labels = data_labels[:number_of_rows]
    if date in CDP_by_date:
        CDP_flight = CDP_by_date[date]
        CDP_time_values = pd.to_numeric(
            CDP_flight["Time_Start"],
            errors="coerce"
        ).to_numpy()
        CDP_indices = np.where(
            (CDP_time_values >= start_time)
            & (CDP_time_values <= stop_time)
        )[0]
        CDP_indices = CDP_indices[:number_of_rows]
        timestamps = CDP_time_values[
            CDP_indices]

    else:
        timestamps = np.full(
            number_of_rows,
            np.nan)
    raw_dNdlogD_sum = np.nansum(
        leg_values,
        axis=1
    )
    dNdD_values = (
        leg_values
        * Logg_CDP[np.newaxis, :])
    concentration_by_bin = (
        dNdD_values
        * CDP_bin_widths[np.newaxis, :]
    )

    total_concentration = np.nansum(
        concentration_by_bin,
        axis=1
    )
    all_bins_nan = np.isnan(
        leg_values
    ).all(axis=1)

    raw_dNdlogD_sum[
        all_bins_nan
    ] = np.nan

    total_concentration[
        all_bins_nan
    ] = np.nan

    # Largest raw and integrated bin in every second
    safe_raw_values = np.where(
        np.isnan(leg_values),
        -np.inf,
        leg_values
    )

    safe_concentration_values = np.where(
        np.isnan(concentration_by_bin),
        -np.inf,
        concentration_by_bin
    )

    largest_raw_bin_index = np.argmax(
        safe_raw_values,
        axis=1
    )

    largest_integrated_bin_index = np.argmax(
        safe_concentration_values,
        axis=1
    )

    for row_index in range(number_of_rows):

        raw_bin_index = int(
            largest_raw_bin_index[row_index]
        )

        concentration_bin_index = int(
            largest_integrated_bin_index[row_index]
        )

        timestamp = (
            timestamps[row_index]
            if row_index < len(timestamps)
            else np.nan
        )

        diagnostic_entry = {
            "Global_Leg_Index": global_leg_index,
            "Date": date,
            "BCB_start": start_time,
            "BCB_stop": stop_time,
            "Row_in_Leg": row_index,
            "Time_Start": timestamp,
            "Label": data_labels[row_index],

            "Raw_dNdlogD_Sum": (
                raw_dNdlogD_sum[row_index]
            ),

            "Calculated_Total_Concentration": (
                total_concentration[row_index]
            ),

            "Largest_Raw_Bin": (
                CDP_bin_names[raw_bin_index]
            ),

            "Largest_Raw_Bin_Center": (
                CDP_bin_centers[raw_bin_index]
            ),

            "Largest_Raw_Bin_Value": (
                leg_values[
                    row_index,
                    raw_bin_index
                ]
            ),

            "Largest_Integrated_Bin": (
                CDP_bin_names[
                    concentration_bin_index
                ]
            ),

            "Largest_Integrated_Bin_Center": (
                CDP_bin_centers[
                    concentration_bin_index
                ]
            ),

            "Largest_Integrated_Bin_Value": (
                concentration_by_bin[
                    row_index,
                    concentration_bin_index
                ]
            ),

            "Number_of_NaN_Bins": int(
                np.isnan(
                    leg_values[row_index]
                ).sum()
            )
        }

        # Store all 30 raw dN/dlogD values
        for bin_index, bin_name in enumerate(
            CDP_bin_names
        ):
            diagnostic_entry[
                bin_name
            ] = leg_values[
                row_index,
                bin_index
            ]

        CDP_raw_diagnostics.append(
            diagnostic_entry
        )


CDP_raw_diagnostics_df = pd.DataFrame(
    CDP_raw_diagnostics
)


print(
    "Total CDP seconds examined:",
    len(CDP_raw_diagnostics_df)
)
#%%
CDP_Y_diagnostics = (
    CDP_raw_diagnostics_df[
        CDP_raw_diagnostics_df["Label"] == "Y"
    ]
    .copy())
print("Total Y seconds:", len(CDP_Y_diagnostics))
print(
    "\nCalculated concentration statistics "
    "for Y records:")
print(
    CDP_Y_diagnostics[
        "Calculated_Total_Concentration"
    ].describe(
        percentiles=[
            0.50,
            0.90,
            0.95,
            0.99,
            0.999
        ]))
#%%
largest_CDP_seconds = (
    CDP_Y_diagnostics
    .sort_values(
        "Calculated_Total_Concentration",
        ascending=False
    )
    .head(50))
columns_to_show = [
    "Date",
    "BCB_start",
    "BCB_stop",
    "Row_in_Leg",
    "Time_Start",
    "Label",
    "Calculated_Total_Concentration",
    "Raw_dNdlogD_Sum",
    "Largest_Raw_Bin",
    "Largest_Raw_Bin_Center",
    "Largest_Raw_Bin_Value",
    "Largest_Integrated_Bin",
    "Largest_Integrated_Bin_Value",
    "Number_of_NaN_Bins"
]
print(
    largest_CDP_seconds[
        columns_to_show
    ].to_string(index=False))
# %%
print(    check_cases[
        [
            "Calculated_Total_Concentration",
            "Reported_N_CDP",
            "Calculated_minus_Reported",
            "Calculated_divided_by_Reported"
        ]
    ].describe())
# %%
check_cases = largest_CDP_seconds[
    [
        "Date",
        "BCB_start",
        "BCB_stop",
        "Time_Start",
        "Calculated_Total_Concentration"
    ]
].copy()


reported_values = []
for _, row in check_cases.iterrows():

    date = row["Date"]
    target_time = row["Time_Start"]

    CDP_flight = CDP_by_date[date].copy()

    CDP_flight["Time_Start"] = pd.to_numeric(
        CDP_flight["Time_Start"],
        errors="coerce"
    )

    CDP_flight["N_CDP"] = pd.to_numeric(
        CDP_flight["N_CDP"],
        errors="coerce"
    )

    CDP_flight["LWC_CDP"] = pd.to_numeric(
        CDP_flight["LWC_CDP"],
        errors="coerce"
    )
    match = CDP_flight[
        np.floor(CDP_flight["Time_Start"]).astype("Int64")
        == int(target_time)
    ]

    if len(match) == 0:

        reported_values.append({
            "Reported_Time_Start": np.nan,
            "Reported_N_CDP": np.nan,
            "Reported_LWC_CDP": np.nan
        })

    else:

        selected = match.iloc[0]

        reported_values.append({
            "Reported_Time_Start": selected["Time_Start"],
            "Reported_N_CDP": selected["N_CDP"],
            "Reported_LWC_CDP": selected["LWC_CDP"]
        })


reported_values_df = pd.DataFrame(
    reported_values,
    index=check_cases.index
)


check_cases = pd.concat(
    [
        check_cases,
        reported_values_df
    ],
    axis=1
)


check_cases["Calculated_minus_Reported"] = (
    check_cases["Calculated_Total_Concentration"]
    - check_cases["Reported_N_CDP"]
)


check_cases["Calculated_divided_by_Reported"] = (
    check_cases["Calculated_Total_Concentration"]
    / check_cases["Reported_N_CDP"]
)


print(
    check_cases.head(50).to_string(index=False)
)
#%%
rh_Y_values_CDP = []
for leg in leg_info_CDP:
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
            rh_Y_values_CDP.append(rh_leg_vals[idx])
plt.hist(rh_Y_values_CDP, bins=30, color='teal', edgecolor='black')
plt.axvline(95, color='red', linestyle='--', label='RH = 95% threshold')
plt.xlabel("RH (%)")
plt.ylabel("Count")
plt.title("RH Values for Seconds Labeled 'Y'")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
master_CDP_BCB = []
leg_info_CDP = []
for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict = leg_data[i]
    flight_date = leg_dict['Date']
    BCB_start = np.array(leg_dict['LegIndex_02']['StartTimes'], dtype=float)
    BCB_stop = np.array(leg_dict['LegIndex_02']['StopTimes'], dtype=float)
    CDP_flight = CDP_1Hz[i]
    twoDS_flight = twoDS[i]
    rh_flight = h20[i]
    CDP_times = np.array(CDP_flight['Time_Start'], dtype=float)
    TwoDS_times = np.array(twoDS_flight['Time_Start'], dtype=float)
    rh_times = np.array(rh_flight['Time_Start'], dtype=float)
    lwc = np.array(CDP_flight['LWC_CDP'], dtype=float)
    N_total = np.array(twoDS_flight['N-total_2DS'], dtype=float)
    rh_total = np.array(rh_flight['RHw_DLH'], dtype=float)

    bins = {
        f'CDP_Bin{bin_label:02d}': np.array(CDP_flight[f'CDP_Bin{bin_label:02d}'], dtype=float)
        for bin_label in range(30)
    }
    total_BCB_means = []
    for k in range(len(BCB_start)):
        start20 = BCB_start[k]
        end20 = BCB_stop[k]

        bin_means = {f'Bin{bin_label:02d}_Y_mean': [] for bin_label in range(30)}
        bin_means.update({f'Bin{bin_label:02d}_N_mean': [] for bin_label in range(30)})
        bin_means.update({'Date': date, 'BCB_start': start20, 'BCB_stop': end20})

        CDP_indices_in_range = np.where((CDP_times >= start20) & (CDP_times <= end20))[0]
        TwoDS_indices_in_range = np.where((TwoDS_times >= start20) & (TwoDS_times <= end20))[0]
        rh_indices_in_range = np.where((rh_times >= start20) & (rh_times <= end20))[0]

        if CDP_indices_in_range.size > 0 and TwoDS_indices_in_range.size > 0 and rh_indices_in_range.size >0:
            for cdp_idx, twods_idx, rh_idx in zip(CDP_indices_in_range, TwoDS_indices_in_range, rh_indices_in_range):
                lwc_val = lwc[cdp_idx]
                N_val = N_total[twods_idx]
                rh_val = rh_total[rh_idx]
                lwc_label = 'Y' if 0 <= lwc_val <= 0.0025 else 'N'
                N_label = 'Y' if 0 <= N_val <= 100 else 'N'
                rh_label = 'Y' if 0 <= rh_val <= 95 else 'N'
                label = 'Y' if lwc_label == 'Y' and N_label == 'Y' and rh_label == 'Y' else 'N'

                for bin_label in range(30):
                    bin_key = f'Bin{bin_label:02d}_{label}_mean'
                    bin_means[bin_key].append(bins[f'CDP_Bin{bin_label:02d}'][cdp_idx])
        for bin_label in range(30):
            for label in ['Y', 'N']:
                bin_key = f'Bin{bin_label:02d}_{label}_mean'
                if bin_means[bin_key]:
                    bin_means[bin_key] = np.nanmean(bin_means[bin_key])
                else:
                    bin_means[bin_key] = np.nan
        total_BCB_means.append(bin_means)
    master_CDP_BCB.append(total_BCB_means)
for item in master_CDP_BCB:
    for bin_means in item:
        print(f"Date: {bin_means['Date']}, Start: {bin_means['BCB_start']}, Stop: {bin_means['BCB_stop']}")
        for bin_label in range(30):
            for label in ['Y', 'N']:
                bin_key = f'Bin{bin_label:02d}_{label}_mean'
                print(f"   {bin_key}: {bin_means[bin_key]}")

#%%
# Count the total number of legs from master_CDP_BCB
total_legs_CDP = sum(len(item) for item in master_CDP_BCB)
print(f"Total number of legs: {total_legs_CDP}")
#%%
Y_BCB_calc_CDP = []
N_BCB_calc_CDP = []
for flight_data in master_CDP_BCB:
    for bin_means_CDP in flight_data:
        Y_calc = {'Date': bin_means_CDP['Date'], 'BCB_start': bin_means_CDP['BCB_start'], 'BCB_stop': bin_means_CDP['BCB_stop']}
        N_calc = {'Date': bin_means_CDP['Date'], 'BCB_start': bin_means_CDP['BCB_start'], 'BCB_stop': bin_means_CDP['BCB_stop']}
        for bin_label in range(0, 30):
            bin_key_Y = f'Bin{bin_label:02d}_Y_mean'
            bin_key_N = f'Bin{bin_label:02d}_N_mean'
            Y_calc[bin_key_Y] = np.nanmean(bin_means_CDP[bin_key_Y]) * Logg_CDP[bin_label]
            N_calc[bin_key_N] = np.nanmean(bin_means_CDP[bin_key_N]) * Logg_CDP[bin_label]
        Y_BCB_calc_CDP.append(Y_calc)
        N_BCB_calc_CDP.append(N_calc)
# %%
plt.figure(figsize=(8, 6))
for entry in Y_BCB_calc_CDP:
    bin_means = np.array([entry.get(f'Bin{i:02d}_Y_mean', np.nan) for i in range(30)], dtype=float)  
    valid_indices = ~np.isnan(bin_means)  
    bin_centers_valid = np.array(bin_center_CDP)[valid_indices]
    bin_means_valid = bin_means[valid_indices]

    plt.plot(bin_centers_valid, bin_means_valid, color='black', alpha=0.5)
plt.xlabel("Deliquesed Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"CDP Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-7, 10**1)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base February-June 2021\n Raw Ambient Size Distributions", fontsize=14, fontweight="bold")
plt.show()
#%%
#Filtering the 0s
plt.figure(figsize=(8, 6))
for entry in Y_BCB_calc_CDP:
    bin_means = np.array([entry.get(f'Bin{i:02d}_Y_mean', np.nan) for i in range(30)], dtype=float)  
    bin_centers = np.array(bin_center_CDP)
    valid_indices = (bin_means > 0) & ~np.isnan(bin_means)  
    bin_centers_valid = bin_centers[valid_indices]
    bin_means_valid = bin_means[valid_indices]
    if len(bin_centers_valid) > 0:  
        plt.plot(bin_centers_valid, bin_means_valid, color='black', alpha=0.5)
plt.xlabel("Deliquesced Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"CDP Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-7, 10**1)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base February-June 2021\n Raw Ambient Size Distributions", fontsize=14, fontweight="bold")
plt.show()
#%%
#checking for negatives 
negative_Y_values = []
for entry in Y_BCB_calc_CDP:
    for bin_label in range(30):
        bin_key = f"Bin{bin_label:02d}_Y_mean"
        value = entry.get(bin_key, np.nan)
        if np.isfinite(value) and value < 0:
            negative_Y_values.append({
                "Date": entry["Date"],
                "BCB_start": entry["BCB_start"],
                "BCB_stop": entry["BCB_stop"],
                "Bin": bin_key,
                "Bin_center_um": bin_center_CDP[bin_label],
                "Value": value
            })
if len(negative_Y_values) == 0:
    print("No negative values found in Y_BCB_calc_CDP.")
else:
    negative_Y_df = pd.DataFrame(negative_Y_values)
    print(
        f"Found {len(negative_Y_df)} negative values "
        "in Y_BCB_calc_CDP:"
    )
    print(
        negative_Y_df.to_string(index=False)
    )
#%%
# Check N_BCB_calc_CDP for negative values
negative_N_values = []
for entry in N_BCB_calc_CDP:
    for bin_label in range(30):
        bin_key = f"Bin{bin_label:02d}_N_mean"
        value = entry.get(bin_key, np.nan)
        if np.isfinite(value) and value < 0:
            negative_N_values.append({
                "Date": entry["Date"],
                "BCB_start": entry["BCB_start"],
                "BCB_stop": entry["BCB_stop"],
                "Bin": bin_key,
                "Bin_center_um": bin_center_CDP[bin_label],
                "Value": value
            })
if len(negative_N_values) == 0:
    print("No negative values found in N_BCB_calc_CDP.")
else:
    negative_N_df = pd.DataFrame(negative_N_values)
    print(
        f"Found {len(negative_N_df)} negative values "
        "in N_BCB_calc_CDP:"
    )
    print(
        negative_N_df.to_string(index=False)
    )
#%%
#average distribution
sum_bin_means = np.zeros(len(bin_center_CDP))
count_bin_means = np.zeros(len(bin_center_CDP))
for entry in Y_BCB_calc_CDP:
    bin_means = np.array([entry.get(f'Bin{i}_Y_mean', np.nan) for i in range(30)], dtype=float)
    valid_indices = (bin_means > 0) & ~np.isnan(bin_means)
    sum_bin_means[valid_indices] += bin_means[valid_indices]
    count_bin_means[valid_indices] += 1
average_bin_means = np.divide(sum_bin_means, count_bin_means, where=count_bin_means > 0)
plt.figure(figsize=(8, 6))
plt.plot(bin_center_CDP, average_bin_means, color='red', linewidth=2, label='Average Size Distribution')
plt.xlabel("Deliquesced Diameter (μm)", fontsize=14, fontweight="bold")
plt.ylabel(r"CDP Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=19, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xlim(0, 45)
plt.xticks(fontweight="bold", fontsize=18)
plt.yticks(fontweight="bold", fontsize=18)
plt.title("CDP Average Ambient \nBelow Cloud Base Size Distribution\n February-June 2021", fontsize=19, fontweight="bold")
plt.show()
#%%
#Fitting an exponential to each size distribution
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)
CDP_fits = []
plt.figure(figsize=(8, 6))
for entry in Y_BCB_calc_CDP:
    bin_means_CDP = np.array([entry.get(f'Bin{i:02d}_Y_mean', np.nan) for i in range(0, 30)], dtype=float)
    valid_indices = ~np.isnan(bin_means_CDP)
    bin_centers_valid = np.array(bin_center_CDP)[valid_indices]
    bin_means_valid = bin_means_CDP[valid_indices]
    if not valid_indices.any():
        print(f"No valid data for date {entry['Date']}")
        continue
    try:
        popt, _ = curve_fit(exponential, bin_centers_valid, bin_means_valid, p0=(1, 1), maxfev=5000)
        n0, D = popt

        CDP_fits.append({
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
plt.ylabel(r"CDP Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("Below Cloud Base February-June 2021\n Exponential Fit to Ambient Size Distributions", fontsize=14, fontweight="bold")
plt.show()

#%%
#Trying to fit and stop at 10um 
bin_center_CDPs=np.array(bin_center_CDP)
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)
CDP_fits_10 = []
plt.figure(figsize=(8, 6))
for entry in Y_BCB_calc_CDP:
    bin_means_CDP = np.array([entry.get(f'Bin{i:02d}_Y_mean', np.nan) for i in range(0, 30)], dtype=float)
    valid_indices = (bin_center_CDPs <= 10) & ~np.isnan(bin_means_CDP)
    bin_centers_valid = np.array(bin_center_CDP)[valid_indices]
    bin_means_valid = bin_means_CDP[valid_indices]
    if valid_indices.any():
        try:
            popt, _ = curve_fit(exponential, bin_centers_valid, bin_means_valid, 
                                p0=(1, 1), maxfev=5000, 
                                bounds=([0, 0.1], [np.inf, 20])) 
            n0, D = popt

            if D > 15:
                print(f"High slope detected! Date: {entry['Date']}, D: {D:.2f}")

        except RuntimeError:
            print(f"Fit failed for date {entry['Date']}")
            n0, D = np.nan, np.nan 
    else:
        n0, D = np.nan, np.nan 
    CDP_fits_10.append({
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
plt.ylabel(r"CDP Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
# plt.ylim(10**-33, 10**1)
plt.ylim(10**-7, 10**1)
plt.xlim(0, 10)
plt.title("CDP Below Cloud Base February-June 2021\n Exponential Fit Ambient Size Distributions (≤10 µm)", fontsize=14, fontweight="bold")
plt.show()
print(f"Total successful ambient exponential fits: {np.sum(~np.isnan([fit['E_folding_D'] for fit in CDP_fits_10]))}")
#%%
#histogram for slope for 10um
CDP_slope_10 = []
for fit in CDP_fits_10:
    if 'E_folding_D' in fit and not np.isnan(fit['E_folding_D']):
        CDP_slope_10.append(fit['E_folding_D'])
plt.figure(figsize=(8, 6))
plt.hist(CDP_slope_10, bins=20, color='blue', alpha=0.7)
plt.xlabel('Slope (um)', fontsize=14, fontweight="bold")
plt.ylabel('Frequency of flight legs', fontsize=14, fontweight="bold")
plt.title('Fitted Ambient Size Distributions (≤10 µm)', fontsize=14, fontweight="bold")
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.show()
#%%
#Calculating total number concentration 
Y_BCB_calc_cm3_CDP = []
N_BCB_calc_cm3_CDP = []

for flight_data in master_CDP_BCB:
    for bin_means_CDP in flight_data:
        Y_calc_CDP = {'Date': bin_means_CDP['Date'], 'BCB_start': bin_means_CDP['BCB_start'], 'BCB_stop': bin_means_CDP['BCB_stop']}
        N_calc_CDP = {'Date': bin_means_CDP['Date'], 'BCB_start': bin_means_CDP['BCB_start'], 'BCB_stop': bin_means_CDP['BCB_stop']}
        for bin_label in range(30):
            bin_key_Y = f'Bin{bin_label:02d}_Y_mean'
            bin_key_N = f'Bin{bin_label:02d}_N_mean'
            Y_calc_CDP[bin_key_Y] = np.nanmean(bin_means_CDP[bin_key_Y]) * bin_log_CDP[bin_label]
            N_calc_CDP[bin_key_N] = np.nanmean(bin_means_CDP[bin_key_N]) * bin_log_CDP[bin_label]
        Y_BCB_calc_cm3_CDP.append(Y_calc_CDP)
        N_BCB_calc_cm3_CDP.append(N_calc_CDP)
# %%
#Calculating total number concentration 
total_concentration_cm3 = []
for entry in Y_BCB_calc_cm3_CDP:
    total_Y_concentration_CDP = np.nansum([entry[f'Bin{i:02d}_Y_mean'] for i in range(30)])  # Sum all valid bin concentrations
    total_concentration_cm3.append({
        'Date': entry['Date'],
        'BCB_start': entry['BCB_start'],
        'BCB_stop': entry['BCB_stop'],
        'Total_Y_Concentration_cm3': total_Y_concentration_CDP
    })
#%%
total_Y_concentrations_CDP = [entry['Total_Y_Concentration_cm3'] for entry in total_concentration_cm3]
total_Y_concentrations_CDP = [conc for conc in total_Y_concentrations_CDP if not np.isnan(conc)]
mean_total_concentration = np.mean(total_Y_concentrations_CDP)
print(f"Mean Total Number Concentration: {mean_total_concentration:.2f} cm⁻³")
#%%
#save total concentration to csv
# save_dir = "/home/disk/eos4/kathem24/activate/data/2021/CDP/1Hz"
# os.makedirs(save_dir, exist_ok=True)   # ensures directory exists
# save_path = os.path.join(save_dir, "total_Y_concentration_cm3_beforemass_CDP.csv")
# total_concentration_df = pd.DataFrame(total_concentration_cm3)
# total_concentration_df.to_csv(save_path, index=False)
# print(f"Saved to: {save_path}")
#%%
#making a PDF of the total number concentrations for the legs labeled 'Y' across all flights. This will help us understand the distribution of total number concentrations below cloud base during the study period.
total_Y_concentrations_CDP = [entry['Total_Y_Concentration_cm3'] for entry in total_concentration_cm3 if not np.isnan(entry['Total_Y_Concentration_cm3']    
)]
plt.figure(figsize=(8, 6))
sns.histplot(total_Y_concentrations_CDP, bins=20, kde=True, color='purple', edgecolor='black', alpha=0.7)
plt.xlabel('Total Number Concentration (cm⁻³)', fontsize=14, fontweight="bold")
plt.ylabel('Frequency of Flight Legs', fontsize=14, fontweight="bold")
plt.title('Distribution of Total Number Concentrations Below Cloud Base\n for Legs Labeled "Y" (February-June 2021)', fontsize=14, fontweight="bold")
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.xlim(0, max(total_Y_concentrations_CDP) * 1.1)
plt.grid(True)
plt.tight_layout()
plt.show()
print(f"Total number of legs with valid total Y concentration: {len(total_Y_concentrations_CDP)}")
print(f"Mean Total Number Concentration: {mean_total_concentration:.2f} cm⁻³")
print(f"Median Total Number Concentration: {np.median(total_Y_concentrations_CDP):.2f} cm⁻³")
print(f"Standard Deviation of Total Number Concentration: {np.std(total_Y_concentrations_CDP):.2f} cm⁻³")
print(f"Minimum Total Number Concentration: {np.min(total_Y_concentrations_CDP):.2f} cm⁻³")
print(f"Maximum Total Number Concentration: {np.max(total_Y_concentrations_CDP):.2f} cm⁻³")
print(f"25th Percentile of Total Number Concentration: {np.percentile(total_Y_concentrations_CDP, 25):.2f} cm⁻³")
print(f"75th Percentile of Total Number Concentration: {np.percentile(total_Y_concentrations_CDP, 75):.2f} cm⁻³")
print(f"Interquartile Range of Total Number Concentration: {np.percentile(total_Y_concentrations_CDP, 75) - np.percentile(total_Y_concentrations_CDP, 25):.2f} cm⁻³")   
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
for entry in Y_BCB_calc_CDP: 
    date = entry['Date']
    BCB_start = entry.get('BCB_start', np.nan)
    BCB_stop = entry.get('BCB_stop', np.nan)
    date_leg_set.add((date, BCB_start, BCB_stop))

filtered_master_BCB_RH_CDP = []
for flight in master_BCB_RH:
    filtered_legs = []
    for leg in flight:
        date = leg['Date']
        BCB_start = leg['BCB_start']
        BCB_stop = leg['BCB_stop']
        if (date, BCB_start, BCB_stop) in date_leg_set:
            filtered_legs.append(leg)
    if filtered_legs:
        filtered_master_BCB_RH_CDP.append(filtered_legs)
#%%
total_entries_filtered_master_BCB_RH_CDP = sum(len(legs) for legs in filtered_master_BCB_RH_CDP)
print(f"Total entries in filtered_master_BCB_RH_CDP: {total_entries_filtered_master_BCB_RH_CDP}")
# %%
##Obtaining g(RH) = [1.7 / (1-RH)]^0.31 for all mean RH values 
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
#LWC histogram
master_BCB_LWC_CDP_Y = []
for i in range(len(dates_legs)):
    date = dates_legs[i]
    leg_dict_CDP = leg_data[i]
    BCB_start = leg_dict_CDP[
        "LegIndex_02"
    ]["StartTimes"]
    BCB_stop = leg_dict_CDP[
        "LegIndex_02"
    ]["StopTimes"]
    CDP_flight = CDP_1Hz[i]
    twoDS_flight = twoDS[i]
    rh_flight = h20[i]
    CDP_flight["Time_Start"] = pd.to_numeric(
        CDP_flight["Time_Start"],
        errors="coerce"
    )
    CDP_flight["LWC_CDP"] = pd.to_numeric(
        CDP_flight["LWC_CDP"],
        errors="coerce"
    )
    twoDS_flight["Time_Start"] = pd.to_numeric(
        twoDS_flight["Time_Start"],
        errors="coerce"
    )

    twoDS_flight["N-total_2DS"] = pd.to_numeric(
        twoDS_flight["N-total_2DS"],
        errors="coerce"
    )

    rh_flight["Time_Start"] = pd.to_numeric(
        rh_flight["Time_Start"],
        errors="coerce"
    )

    rh_flight["RHw_DLH"] = pd.to_numeric(
        rh_flight["RHw_DLH"],
        errors="coerce"
    )
    CDP_times = CDP_flight["Time_Start"].to_numpy()
    CDP_lwc = CDP_flight["LWC_CDP"].to_numpy()
    TwoDS_times = twoDS_flight["Time_Start"].to_numpy()
    TwoDS_N = twoDS_flight["N-total_2DS"].to_numpy()
    rh_times = rh_flight["Time_Start"].to_numpy()
    rh_values = rh_flight["RHw_DLH"].to_numpy()
    for j in range(len(BCB_start)):

        start = BCB_start[j]
        end = BCB_stop[j]

        CDP_idx = np.where(
            (CDP_times >= start)
            & (CDP_times <= end)
        )[0]

        TwoDS_idx = np.where(
            (TwoDS_times >= start)
            & (TwoDS_times <= end)
        )[0]

        rh_idx = np.where(
            (rh_times >= start)
            & (rh_times <= end)
        )[0]

        if (
            len(CDP_idx) == 0
            or len(TwoDS_idx) == 0
            or len(rh_idx) == 0
        ):
            continue
        accepted_LWC = []
        for cdp_i, twods_i, rh_i in zip(
            CDP_idx,
            TwoDS_idx,
            rh_idx
        ):
            lwc_val = CDP_lwc[cdp_i]
            N_val = TwoDS_N[twods_i]
            rh_val = rh_values[rh_i]
            label_Y = (
                np.isfinite(lwc_val)
                and np.isfinite(N_val)
                and np.isfinite(rh_val)
                and 0 <= lwc_val <= 0.0025
                and 0 <= N_val <= 100
                and 0 <= rh_val <= 95
            )
            if label_Y:
                accepted_LWC.append(lwc_val)
        if len(accepted_LWC) > 0:
            master_BCB_LWC_CDP_Y.append({
                "Date": date,
                "BCB_start": start,
                "BCB_stop": end,
                "LWC_mean": np.nanmean(accepted_LWC),
                "Number_of_Y_seconds": len(accepted_LWC),
                "Total_seconds": min(
                    len(CDP_idx),
                    len(TwoDS_idx),
                    len(rh_idx)
                )
            })
lwc_values_Y = np.array(
    [
        entry["LWC_mean"]
        for entry in master_BCB_LWC_CDP_Y
        if np.isfinite(entry["LWC_mean"])
    ],
    dtype=float
)

print(
    "Number of accepted LWC legs:",
    len(lwc_values_Y)
)

print(
    "Maximum accepted leg-average LWC:",
    np.max(lwc_values_Y)
)

print(
    "Number above 0.0025:",
    np.sum(lwc_values_Y > 0.0025)
)


plt.figure(figsize=(8, 6))

plt.hist(
    lwc_values_Y,
    bins=20,
    edgecolor="black",
    alpha=0.7
)

plt.axvline(
    0.0025,
    linestyle="--",
    linewidth=2,
    label="LWC threshold"
)
plt.xlabel(
    "Mean accepted LWC (g m$^{-3}$)",
    fontsize=15,
    fontweight="bold"
)
plt.ylabel(
    "Frequency of flight legs",
    fontsize=15,
    fontweight="bold"
)
plt.title(
    "Leg-average CDP LWC\n"
    "Using Accepted Y Measurements Only",
    fontweight="bold",
    fontsize=16
)
plt.xticks(
    fontweight="bold",
    fontsize=14
)
plt.yticks(
    fontweight="bold",
    fontsize=14
)
plt.legend()
plt.show()
print(
    f"Mean Y-only leg-average LWC: "
    f"{np.mean(lwc_values_Y):.6f} g m^-3"
)

print(
    f"Median Y-only leg-average LWC: "
    f"{np.median(lwc_values_Y):.6f} g m^-3"
)
#%%
#save lwc to .csv
# save_dir = "/home/disk/eos4/kathem24/activate/data/2021/CDP/1Hz"
# os.makedirs(save_dir, exist_ok=True)
# save_path = os.path.join(save_dir, "BCB_LWC_means_beforemass_CDP.csv")
# master_BCB_LWC_df = pd.DataFrame(master_BCB_LWC_CDP)
# master_BCB_LWC_df.to_csv(save_path, index=False)
# print(f"Saved to: {save_path}")
#%%
#Histogram of RH values
rh_values = [
    leg['Rh_mean'][0] for flight in filtered_master_BCB_RH_CDP for leg in flight if not np.isnan(leg['Rh_mean'][0])
]
plt.figure(figsize=(8, 6))
plt.hist(rh_values, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel('Relative Humidity (%)', fontsize=15, fontweight='bold')
plt.ylabel('Frequency of flight legs', fontsize=15, fontweight='bold')
plt.title('Leg average RH January-June 2021', fontweight='bold', fontsize=16)
plt.xticks(fontweight='bold', fontsize=14)
plt.yticks(fontweight='bold', fontsize=14)
plt.show()
# %%
#only the grh from filtered_master_BCB_RH
filtered_master_BCB_gRH_CDP = []
removed_nan_RH_legs = []
for flight in filtered_master_BCB_RH_CDP:
    flight_gRH = []
    for leg in flight:
        rh_percent = leg["Rh_mean"][0]
        if not np.isfinite(rh_percent):
            removed_nan_RH_legs.append({
                "Date": leg.get("Date"),
                "BCB_start": leg.get("BCB_start"),
                "BCB_stop": leg.get("BCB_stop"),
                "Rh_mean": rh_percent})

            continue
        rh_fraction = rh_percent / 100.0
        if rh_fraction < 0 or rh_fraction >= 1:
            removed_nan_RH_legs.append({
                "Date": leg.get("Date"),
                "BCB_start": leg.get("BCB_start"),
                "BCB_stop": leg.get("BCB_stop"),
                "Rh_mean": rh_percent})
            continue
        gRH_value = (
            1.7 / (1 - rh_fraction)
        ) ** 0.31
        new_leg = leg.copy()
        new_leg["gRh_mean"] = [gRH_value]
        flight_gRH.append(new_leg)
    filtered_master_BCB_gRH_CDP.append(flight_gRH)
#%%
original_leg_count = sum(
    len(flight)
    for flight in filtered_master_BCB_RH_CDP)
remaining_leg_count = sum(
    len(flight)
    for flight in filtered_master_BCB_gRH_CDP)
print("Original RH legs:", original_leg_count)
print("Removed invalid RH legs:", len(removed_nan_RH_legs))
print("Remaining valid gRH legs:", remaining_leg_count)
print(
    "Count check:",
    original_leg_count
    == remaining_leg_count + len(removed_nan_RH_legs))
removed_nan_RH_df = pd.DataFrame(
    removed_nan_RH_legs)
print(removed_nan_RH_df.to_string(index=False))
#%%
remaining_gRH_values = np.array(
    [
        leg["gRh_mean"][0]
        for flight in filtered_master_BCB_gRH_CDP
        for leg in flight
    ],dtype=float)
print("Number of NaN gRH values remaining:",np.isnan(remaining_gRH_values).sum())
print("Number of infinite gRH values remaining:",np.isinf(remaining_gRH_values).sum())
#%%
total_entries_filtered_master_BCB_gRH_CDP = sum(len(legs) for legs in filtered_master_BCB_gRH_CDP)
print(f"Total entries in filtered_master_BCB_gRH_CDP: {total_entries_filtered_master_BCB_gRH_CDP}")
#%%
#Histogram of gRH values
gRH_values = [
    leg['gRh_mean'][0] for flight in filtered_master_BCB_gRH_CDP for leg in flight if not np.isnan(leg['gRh_mean'][0])
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
flat_filtered_master_BCB_gRH_CDP = [
    leg
    for flight in filtered_master_BCB_gRH_CDP
    for leg in flight
]
#%%
filtered_master_BCB_ddry_CDP = []
for entry in flat_filtered_master_BCB_gRH_CDP:
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    gRh_mean = entry['gRh_mean'][0] 
    if gRh_mean > 0:
        ddry_values = np.array([D_amb / gRh_mean for D_amb in bin_center_CDP])
    else:
        ddry_values = np.full(len(bin_center_CDP), np.nan)
        print(f"Skipping division for {date}, {BCB_start}-{BCB_stop} due to invalid gRh_mean.")
    ddry_bin_widths = np.diff(ddry_values, append=np.nan) 
    raw_concentrations = next(
        (leg for leg in Y_BCB_calc_CDP if leg['Date'] == date and leg['BCB_start'] == BCB_start and leg['BCB_stop'] == BCB_stop),
        None
    )
    if raw_concentrations:
        dN_dD_ambient = np.array([raw_concentrations.get(f'Bin{i:02d}_Y_mean', np.nan) for i in range(30)], dtype=float)

        dN_dD_dry = np.where(
        (~np.isnan(dN_dD_ambient)) & (~np.isnan(ddry_bin_widths)) & (gRh_mean > 0),
        dN_dD_ambient * (np.array(bin_center_CDP) / ddry_values) * (np.diff(bin_center_CDP, append=np.nan) / ddry_bin_widths),
        np.nan
    )
    else:
        dN_dD_dry = np.full(len(bin_center_CDP), np.nan)
        print(f"Missing raw size distribution for {date}, {BCB_start}-{BCB_stop}")
    filtered_master_BCB_ddry_CDP.append({
        'Date': date,
        'BCB_start': BCB_start,
        'BCB_stop': BCB_stop,
        'ddry': ddry_values.tolist(),
        'dN/dDdry': dN_dD_dry.tolist(),
        'ddry_bin_widths': ddry_bin_widths.tolist(), 
        'gRh_mean': gRh_mean
    })
print(f"Length of filtered_master_BCB_ddry_CDP: {len(filtered_master_BCB_ddry_CDP)}")
#%%
from scipy.interpolate import interp1d
common_bins = np.linspace(2, 25, 35)
plt.figure(figsize=(8, 6))
for entry in filtered_master_BCB_ddry_CDP:
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
plt.title("Below Cloud Base February-June 2021\n Raw Dry Size Distributions", fontsize=14, fontweight="bold")
plt.show()
#%%%
#Removing the 0s
common_bins = np.linspace(2, 25, 35) 
plt.figure(figsize=(8, 6))
for entry in filtered_master_BCB_ddry_CDP:
    ddry_values = np.array(entry['ddry'])  
    dN_dD_dry = np.array(entry['dN/dDdry']) 
   
    valid_indices = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry)
    if np.sum(valid_indices) < 2:
        continue 
    interp_func = interp1d(ddry_values[valid_indices], dN_dD_dry[valid_indices], 
                           kind='linear', bounds_error=False, fill_value=np.nan)
    interpolated_dN_dD_dry = interp_func(common_bins)
    valid_interpolated_indices = (interpolated_dN_dD_dry > 0) & ~np.isnan(interpolated_dN_dD_dry)
    filtered_bins = common_bins[valid_interpolated_indices]
    filtered_dN_dD_dry = interpolated_dN_dD_dry[valid_interpolated_indices]
    if len(filtered_bins) > 0: 
        plt.plot(filtered_bins, filtered_dN_dD_dry, color='purple', alpha=0.2)
plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=20, fontweight="bold")
plt.ylabel("CDP Number Concentration\n(cm$^{-3}$ μm$^{-1}$)", fontsize=20, fontweight="bold")
plt.yscale("log")
plt.xticks(fontweight="bold", fontsize=20)
plt.yticks(fontweight="bold", fontsize=20)
plt.ylim(10**-7, 10**1.5)
plt.xlim(0.5, 40)
plt.title("CDP Below Cloud Base\n January-June 2021\n Raw Dry Size Distributions", fontsize=20, fontweight="bold")
plt.show()
#%%
#average dry distribution ********
common_bins = np.linspace(2, 25, 35)  
sum_interpolated_dN_dD_dry = np.zeros_like(common_bins, dtype=float)
count_interpolated_dN_dD_dry = np.zeros_like(common_bins, dtype=int)
for entry in filtered_master_BCB_ddry_CDP:
    ddry_values = np.array(entry['ddry'])  
    dN_dD_dry = np.array(entry['dN/dDdry'])
    valid_indices = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry)
    if np.sum(valid_indices) < 2:
        continue 
    interp_func = interp1d(ddry_values[valid_indices], dN_dD_dry[valid_indices], 
                           kind='linear', bounds_error=False, fill_value=np.nan)
    interpolated_dN_dD_dry = interp_func(common_bins)
    valid_interpolated_indices = ~np.isnan(interpolated_dN_dD_dry)
    sum_interpolated_dN_dD_dry[valid_interpolated_indices] += interpolated_dN_dD_dry[valid_interpolated_indices]
    count_interpolated_dN_dD_dry[valid_interpolated_indices] += 1
average_dN_dD_dry = np.divide(sum_interpolated_dN_dD_dry, count_interpolated_dN_dD_dry, where=count_interpolated_dN_dD_dry > 0)
plt.figure(figsize=(8, 6))
plt.plot(common_bins, average_dN_dD_dry, color='red', linewidth=2, label='Average Dry Size Distribution')
plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=20, fontweight="bold")
plt.ylabel(r"CDP Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=20, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xlim(0, 45)
plt.xticks(fontweight="bold", fontsize=20)
plt.yticks(fontweight="bold", fontsize=20)
plt.title("CDP Average Below Cloud Base \nDry Size Distribution\n January-June 2021", fontsize=20, fontweight="bold")
plt.legend()
plt.show()
#%%
#save the average distribution
# average_dry_distribution = pd.DataFrame({
#     'Dry_Diameter_um': common_bins,
#     'Average_dN_dD_dry': average_dN_dD_dry,
#     'N_profiles': count_interpolated_dN_dD_dry
# })
# save_dir = "/home/disk/eos4/kathem24/activate/data/2021/CDP/1Hz"
# save_path = os.path.join(save_dir, "Average_Dry_Size_Distribution_beforemass_CDP.csv")
# average_dry_distribution.to_csv(save_path, index=False)
# print(f"Saved to: {save_path}")
#%%
# Check transformation step
for entry in filtered_master_BCB_ddry_CDP[:5]:  
    print(f"Date: {entry['Date']}, Start: {entry['BCB_start']}, Stop: {entry['BCB_stop']}")
    print("  gRh_mean:", entry['gRh_mean'])
    print("  dN/dDdry first 5 bins:", entry['dN/dDdry'][:5])
    print("  ddry_bin_widths first 5 bins:", entry['ddry_bin_widths'][:5])
    print("  Original bin widths:", np.diff(bin_center_CDP, append=np.nan)[:5])
    print("  -----")
#%%
for entry in filtered_master_BCB_ddry_CDP[:5]: 
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    gRh_mean = entry['gRh_mean']
    raw_concentrations = next(
        (leg for leg in Y_BCB_calc_CDP if leg['Date'] == date and leg['BCB_start'] == BCB_start and leg['BCB_stop'] == BCB_stop),
        None
    )
    if raw_concentrations:
        dN_dD_ambient = np.array([raw_concentrations.get(f'Bin{i:02d}_Y_mean', np.nan) for i in range(30)])
        print(f"Date: {date}, Start: {BCB_start}, Stop: {BCB_stop}")
        print(f"  gRh_mean: {gRh_mean}")
        print(f"  dN/dDambient first 5 bins: {dN_dD_ambient[:5]}")
        print(f"  dN/dDdry first 5 bins: {entry['dN/dDdry'][:5]}")
        print(f"  Ratio (dN/dDdry / dN/dDambient): {np.array(entry['dN/dDdry'][:5]) / dN_dD_ambient[:5]}")
        print("  -----")
#%%
common_bins = np.linspace(2, 40, 35)
plt.figure(figsize=(8, 6))
for entry_ambient, entry_dry in zip(Y_BCB_calc_CDP, filtered_master_BCB_ddry_CDP):
    ambient_dd = np.array(bin_center_CDP)
    ambient_dN_dD = np.array([entry_ambient.get(f'Bin{i:02d}_Y_mean', np.nan) for i in range(30)])
    dry_dd = np.array(entry_dry['ddry'])
    dry_dN_dD = np.array(entry_dry['dN/dDdry'])
    valid_ambient = ~np.isnan(ambient_dd) & ~np.isnan(ambient_dN_dD)
    valid_dry = ~np.isnan(dry_dd) & ~np.isnan(dry_dN_dD)
    if np.sum(valid_ambient) < 2 or np.sum(valid_dry) < 2:
        continue  
    interp_ambient = interp1d(ambient_dd[valid_ambient], ambient_dN_dD[valid_ambient], 
                              kind='linear', bounds_error=False, fill_value=np.nan)
    interp_dry = interp1d(dry_dd[valid_dry], dry_dN_dD[valid_dry], 
                          kind='linear', bounds_error=False, fill_value=np.nan)
    interpolated_ambient = interp_ambient(common_bins)
    interpolated_dry = interp_dry(common_bins)
    plt.plot(common_bins, interpolated_ambient, color='blue', alpha=0.3, label="Ambient" if 'Ambient' not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.plot(common_bins, interpolated_dry, color='red', alpha=0.3, label="Dry" if 'Dry' not in plt.gca().get_legend_handles_labels()[1] else "")
plt.xlabel("Bin Centers Diameter (μm)", fontsize=12, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=12, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-7, 10**1)
plt.xlim(0,40)
plt.xticks(fontweight="bold", fontsize=10)
plt.yticks(fontweight="bold", fontsize=10)
plt.title("Below Cloud Base February-June 2021\n Raw Size Distributions", fontsize=14, fontweight="bold")
plt.legend()
plt.show()
#%%
#Fitting exponential to the dry distributions
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)
dry_exponential_fits = []
plt.figure(figsize=(8, 6))
for entry in filtered_master_BCB_ddry_CDP:
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
plt.title("Below Cloud Base February-June 2021\n Fitted Dry Size Distributions", fontsize=14, fontweight="bold")
plt.show()
print(f"Total successful dry exponential fits: {len(dry_exponential_fits)}")
#%%
positive_counts = []
for entry in filtered_master_BCB_ddry_CDP:
    y = np.array(entry['dN/dDdry'], dtype=float)
    positive_counts.append(np.sum(y > 0))
unique, counts = np.unique(positive_counts, return_counts=True)
for u, c in zip(unique, counts):
    print(f"{u:2d} positive bins : {c}")
#%%
#average fitted distribution
common_bins = np.linspace(2, 25, 35)
for entry in filtered_master_BCB_ddry_CDP:
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
plt.ylabel(r"CDP Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-7, 10**0)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("CDP Average Below Cloud Base Exponential Fitted Dry Size Distribution\n January-June 2021", fontsize=14, fontweight="bold")
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
plt.ylabel(r"CDP Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=14, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.title("CDP Average Below Cloud Base Dry Size Distribution\n January-June 2021", fontsize=14, fontweight="bold")
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
for entry in filtered_master_BCB_ddry_CDP:
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
plt.ylabel(r"CDP Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=19, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-7, 10**1.5)
plt.xticks(fontweight="bold", fontsize=19)
plt.yticks(fontweight="bold", fontsize=19)
plt.title("CDP Below Cloud Base\n January-June 2021\nFitted Dry Size Distributions", fontsize=20, fontweight="bold")
plt.show()
#%%
# Fit exponential only to dry diameters <= 10 µm
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)
dry_exponential_fits_10 = []
plt.figure(figsize=(8, 6))
for entry in filtered_master_BCB_ddry_CDP:
    ddry_values = np.asarray(entry["ddry"], dtype=float)
    dN_dD_dry = np.asarray(entry["dN/dDdry"], dtype=float)
    valid_indices = (
        (ddry_values <= 10)
        & np.isfinite(ddry_values)
        & np.isfinite(dN_dD_dry)
        & (dN_dD_dry > 0))
    n_valid = np.sum(valid_indices)
    if n_valid < 2:
        dry_exponential_fits_10.append({
            "Date": entry["Date"],
            "BCB_start": entry["BCB_start"],
            "BCB_stop": entry["BCB_stop"],
            "Dry_Intercept_n0": np.nan,
            "Dry_E_folding_D": np.nan
        })
        continue
    x_data = ddry_values[valid_indices]
    y_data = dN_dD_dry[valid_indices]
    try:
        popt, _ = curve_fit(
            exponential,
            x_data,
            y_data,
            p0=(1, 5),
            bounds=([0, 0], [np.inf, np.inf]),
            maxfev=5000)
        n0, D = popt
        if not np.isfinite(D) or D <= 0 or D > 20:
            n0, D = np.nan, np.nan

    except (RuntimeError, ValueError, TypeError):
        print(
            f"Fit could not be performed for date {entry['Date']}, "
            f"BCB start {entry['BCB_start']}")
        n0, D = np.nan, np.nan
    dry_exponential_fits_10.append({
        "Date": entry["Date"],
        "BCB_start": entry["BCB_start"],
        "BCB_stop": entry["BCB_stop"],
        "Dry_Intercept_n0": n0,
        "Dry_E_folding_D": D})
    if np.isfinite(n0) and np.isfinite(D):
        x_fit = np.linspace(np.min(x_data), np.max(x_data), 100)
        y_fit = exponential(x_fit, n0, D)
        if np.all(np.isfinite(y_fit)) and np.all(y_fit > 1e-33):
            plt.plot(x_fit, y_fit, color="black", alpha=0.2)
plt.xlabel("Dry Bin Centers Diameter (μm)",
    fontsize=12,
    fontweight="bold")
plt.ylabel(r"CDP Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)",
    fontsize=12,
    fontweight="bold")
plt.yscale("log")
plt.ylim(1e-7, 1e1)
plt.xticks(fontweight="bold", fontsize=10)
plt.yticks(fontweight="bold", fontsize=10)
plt.title("CDP Below Cloud Base\n"
    "January-June 2021\n"
    "Fitted Dry Size Distributions (≤10 µm)",
    fontsize=14,
    fontweight="bold")
plt.show()
successful_fits_10 = [
    fit for fit in dry_exponential_fits_10
    if np.isfinite(fit["Dry_Intercept_n0"])
    and np.isfinite(fit["Dry_E_folding_D"])]
print(f"Total entries: {len(dry_exponential_fits_10)}")
print(f"Total successful dry exponential fits ≤10 µm: {len(successful_fits_10)}")
#%%
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)
diagnostics = []
for entry in filtered_master_BCB_ddry_CDP:
    ddry = np.array(entry['ddry'], dtype=float)
    y = np.array(entry['dN/dDdry'], dtype=float)
    valid = (
        (ddry <= 10) &
        ~np.isnan(ddry) &
        ~np.isnan(y) &
        (y > 0))
    n_valid = np.sum(valid)
    result = {
        'Date': entry['Date'],
        'BCB_start': entry['BCB_start'],
        'BCB_stop': entry['BCB_stop'],
        'n_positive_bins_le10': n_valid,
        'min_y': np.nan,
        'max_y': np.nan,
        'dynamic_range': np.nan,
        'n0': np.nan,
        'D': np.nan,
        'fit_status': ''
    }
    if n_valid < 3:
        result['fit_status'] = 'too few positive bins'
        diagnostics.append(result)
        continue
    xfit = ddry[valid]
    yfit = y[valid]
    result['min_y'] = np.min(yfit)
    result['max_y'] = np.max(yfit)
    result['dynamic_range'] = np.max(yfit) / np.min(yfit)
    try:
        popt, _ = curve_fit(
            exponential,
            xfit,
            yfit,
            p0=(np.max(yfit), 3),
            bounds=([0, 0.1], [np.inf, 100]),
            maxfev=10000)
        n0, D = popt
        result['n0'] = n0
        result['D'] = D
        if D > 20:
            result['fit_status'] = 'D > 20, nearly flat/unstable'
        elif D > 4:
            result['fit_status'] = 'D > 4'
        else:
            result['fit_status'] = 'ok'

    except Exception as e:
        result['fit_status'] = f'fit failed: {e}'
    diagnostics.append(result)
df_diag = pd.DataFrame(diagnostics)
print(df_diag['fit_status'].value_counts())
print(df_diag.sort_values('D', ascending=False).head(30))
#%%
suspicious = df_diag[df_diag['D'] > 4].sort_values('D', ascending=False)
for _, row in suspicious.iterrows():
    for entry in filtered_master_BCB_ddry_CDP:
        if (
            entry['Date'] == row['Date'] and
            entry['BCB_start'] == row['BCB_start'] and
            entry['BCB_stop'] == row['BCB_stop']):
            ddry = np.array(entry['ddry'], dtype=float)
            y = np.array(entry['dN/dDdry'], dtype=float)
            valid = (
                (ddry <= 10) &
                ~np.isnan(ddry) &
                ~np.isnan(y) &
                (y > 0))
            plt.figure(figsize=(7,5))
            plt.plot(ddry[valid], y[valid], 'o-', color='black')
            if not np.isnan(row['D']):
                xfit = np.linspace(np.min(ddry[valid]), 10, 100)
                yfit = exponential(xfit, row['n0'], row['D'])
                plt.plot(xfit, yfit, 'r--', label=f"D = {row['D']:.2f}")
            plt.yscale('log')
            plt.ylim(1e-7, 1e1)
            plt.xlim(0, 10)
            plt.xlabel("Dry Diameter (μm)")
            plt.ylabel(r"CDP Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)")
            plt.title(f"{row['Date']} {row['BCB_start']}-{row['BCB_stop']}\npositive bins={row['n_positive_bins_le10']}")
            plt.legend()
            plt.show()
            break
#%%
#histogram comapring less than 10um and regular fit exponential 
dry_slopes_10 = [fit['Dry_E_folding_D'] for fit in dry_exponential_fits_10 if not np.isnan(fit['Dry_E_folding_D'])] 
#%%
dry_intercepts_10=[fit['Dry_Intercept_n0'] for fit in dry_exponential_fits_10 if not np.isnan(fit['Dry_Intercept_n0'])]
# %%
dry_slopes = []
dry_intercepts = []
for entry in dry_exponential_fits_10:
    n0 = entry['Dry_Intercept_n0'] 
    D = entry['Dry_E_folding_D']  
    dry_intercepts.append(n0)
    dry_slopes.append(D)
df_dry = pd.DataFrame({
    'Dry_Intercept_N0': dry_intercepts,
    'Dry_Slope_D': dry_slopes})
plt.figure(figsize=(10, 6))
plt.scatter(df_dry['Dry_Slope_D'], df_dry['Dry_Intercept_N0'], alpha=0.6, color='black')
plt.xlabel('Dry Slope (um)', fontsize=14, fontweight='bold')
plt.ylabel(r'Dry Intercept (cm$^{-3}$ $\mu$m$^{-1}$)', fontsize=14, fontweight='bold')
plt.title('Below Cloud Base February-June 2021\n Dry', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.xscale('log')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.xlim(10**-0.7, 10**1.1) 
plt.ylim(10**-1.3, 10**2)
plt.show()
#%%
#save dry_exponential_fits_10 to csv
# save_dir = "/home/disk/eos4/kathem24/activate/data/2021/CDP/1Hz"
# os.makedirs(save_dir, exist_ok=True)   # ensures directory exists
# save_path = os.path.join(save_dir, "dry_exponential_fits_10_CDP.csv")
# dry_exponential_fits_10_df = pd.DataFrame(dry_exponential_fits_10)
# dry_exponential_fits_10_df.to_csv(save_path, index=False)
# print(f"Saved to: {save_path}")
#%%
#Saving dry mass
#mass to inf
rho_salt = 2200  # kg/m³
def calculate_mass(N0, D):
    N0_m4 = N0 * 10**6  # Convert cm⁻³µm⁻¹ to m⁻⁴
    integrand = lambda d: np.exp(-d / D) * (d * 1e-6)**3  # Convert µm³ → m³
    mass_integral, _ = quad(integrand, 2, np.inf)  # Integrate from 2µm to ∞
    return (np.pi / 6) * rho_salt * N0_m4 * mass_integral  
dry_mass_data_inf = []
for entry in dry_exponential_fits:
    date = entry['Date']
    dry_intercept = entry['Dry_Intercept_n0']
    dry_slope = entry['Dry_E_folding_D']
    if dry_slope > 0 and dry_intercept > 0:
        mass_value = calculate_mass(dry_intercept, dry_slope) * 1e9  # Convert kg/m³ to µg/m³
        dry_mass_data_inf.append({
        'Date': date,
        'BCB_start': entry['BCB_start'], 
        'BCB_stop': entry['BCB_stop'],  
        'Dry Slope (D)': dry_slope,
        'Dry Intercept (N0)': dry_intercept,
        'Dry Mass (µg/m³)': mass_value
    })
dry_slopes = np.array([entry['Dry Slope (D)'] for entry in dry_mass_data_inf])
dry_intercepts = np.array([entry['Dry Intercept (N0)'] for entry in dry_mass_data_inf])
min_slope_threshold = np.percentile(dry_slopes, 1) 
filtered_slopes = dry_slopes[dry_slopes >= min_slope_threshold]
filtered_intercepts = dry_intercepts[dry_slopes >= min_slope_threshold]
x_min, x_max = np.percentile(filtered_slopes, [5, 95])
y_min, y_max = np.percentile(filtered_intercepts, [5, 95])
xgrid_adjusted = np.logspace(np.log10(x_min), np.log10(x_max), 200)
ygrid_adjusted = np.logspace(np.log10(y_min), np.log10(y_max), 200)
D_grid_adjusted, dryintercept_grid_adjusted = np.meshgrid(xgrid_adjusted, ygrid_adjusted)
mass_grid_adjusted = np.zeros_like(D_grid_adjusted)
for i in range(D_grid_adjusted.shape[0]):
    for j in range(D_grid_adjusted.shape[1]):
        mass_grid_adjusted[i, j] = calculate_mass(dryintercept_grid_adjusted[i, j], D_grid_adjusted[i, j]) * 1e9

mass_levels = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]

plt.figure(figsize=(10, 8))
plt.scatter(filtered_slopes, filtered_intercepts, c='blue', s=80, alpha=0.7, label="Dry Data Points")
contour_plot = plt.contour(D_grid_adjusted, dryintercept_grid_adjusted, mass_grid_adjusted, 
                           levels=mass_levels, colors='red', alpha=0.75, linewidths=1.5)

plt.clabel(contour_plot, inline=True, fontsize=13, fmt=lambda x: f"{int(x)} µg/m³", colors='black', inline_spacing=5)
for txt in contour_plot.labelTexts:
    txt.set_fontweight('bold')
    txt.set_rotation(15)
plt.xlabel(r'Dry Slope ($\mu$m)', fontsize=19, fontweight='bold')
plt.ylabel(r'Dry Intercept (cm$^{-3}$ $\mu$m$^{-1}$)', fontsize=19, fontweight='bold')
plt.title('CDP Below Cloud Base January - June 2021\nContours of Dry Mass', fontsize=19, fontweight='bold')
plt.xscale('log')
plt.yscale('log')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
# %%
# Extract all mass values from dry_mass
mass_values_ug_inf = [entry['Dry Mass (µg/m³)'] for entry in dry_mass_data_inf]
min_mass_ug_inf = min(mass_values_ug_inf)
max_mass_ug_inf = max(mass_values_ug_inf)
print(f"Min Mass (µg/m³): {min_mass_ug_inf}")
print(f"Max Mass (µg/m³): {max_mass_ug_inf}")
#%%
# Set the mass threshold
mass_threshold = 100  # µg/m³
filtered_dry_mass_inf = [entry for entry in dry_mass_data_inf if (
    not np.isnan(entry['Dry Slope (D)']) and 
    not np.isnan(entry['Dry Intercept (N0)']) and 
    entry['Dry Mass (µg/m³)'] <= mass_threshold
)]
print(f"Filtered Dry Mass Entries: {len(filtered_dry_mass_inf)} (after removing masses > {mass_threshold} µg/m³)")
slope_array = np.array([entry['Dry Slope (D)'] for entry in filtered_dry_mass_inf]).reshape(-1, 1)
intercept_array = np.array([entry['Dry Intercept (N0)'] for entry in filtered_dry_mass_inf]).reshape(-1, 1)
data_points = np.column_stack((slope_array, intercept_array))
#%%
mass_values = np.array([entry['Dry Mass (µg/m³)'] for entry in filtered_dry_mass_inf])
mean_mass = np.mean(mass_values)
print(f"Mean mass after filtering: {mean_mass:.3f} µg/m³")
median_mass = np.median(mass_values)
print(f"Median mass after filtering: {median_mass:.3f} µg/m³")
#%%
#saving mass to a csv (FILTERED ≤ 100)
df_dry_mass_inf = pd.DataFrame(filtered_dry_mass_inf)
output_path = "Dry_mass_BCB2021_lessthan100massREAL_CDP.csv"
df_dry_mass_inf.to_csv(output_path, index=False)
print(f"Saved filtered dry mass data (≤100) to {output_path}")
mass_col = "Dry Mass (µg/m³)"
print("Max in saved df:", df_dry_mass_inf[mass_col].max())
print("Mean in saved df:", df_dry_mass_inf[mass_col].mean())
#%%
slope_values = np.array([entry['Dry Slope (D)'] for entry in filtered_dry_mass_inf])
mean_slope = np.mean(slope_values)
median_slope = np.median(slope_values)
print(f"Mean slope (D): {mean_slope:.3f}")
print(f"Median slope (D): {median_slope:.3f}")
#%%
#removing corresponding slope legs based on the mass threshold 
mass_threshold = 100.0  # µg/m^3
def make_leg_key(entry):
    return (entry["Date"], int(entry["BCB_start"]), int(entry["BCB_stop"]))
bad_leg_keys = {
    make_leg_key(e)
    for e in dry_mass_data_inf
    if (not np.isnan(e["Dry Mass (µg/m³)"])) and (e["Dry Mass (µg/m³)"] > mass_threshold)
}
print("Legs with mass > threshold:", len(bad_leg_keys))
filtered_slope_mass100gone = [
    e for e in dry_mass_data_inf
    if (
        make_leg_key(e) not in bad_leg_keys
        and not np.isnan(e["Dry Slope (D)"])
        and not np.isnan(e["Dry Intercept (N0)"])
        and not np.isnan(e["Dry Mass (µg/m³)"]))
]
print("Original slope/mass entries:", len(dry_mass_data_inf))
print("After removing high-mass legs:", len(filtered_slope_mass100gone))
slope_values_mass100gone = np.array([e["Dry Slope (D)"] for e in filtered_slope_mass100gone])
print(f"Mean slope after mass filter: {np.mean(slope_values_mass100gone):.3f}")
print(f"Median slope after mass filter: {np.median(slope_values_mass100gone):.3f}")
# with open("CDP_slope_massLE1002021.pkl", "wb") as f:
#     pickle.dump(filtered_slope_mass100gone, f)
# print("Saved CDP slope-filtered dataset.")
# print("Saved to:", os.path.abspath("CDP_slope_massLE1002021.pkl"))
# print("Exists?", os.path.exists("CDP_slope_massLE1002021.pkl"))
#%%
#removing corresponding concentration legs based on the mass threshold 
def make_leg_key(entry):
    return (entry["Date"], int(entry["BCB_start"]), int(entry["BCB_stop"]))

good_leg_keys = {
    make_leg_key(e)
    for e in filtered_slope_mass100gone}
total_concentration_cm3_mass100gone = [e for e in total_concentration_cm3
    if make_leg_key(e) in good_leg_keys]
print("Final CDP slope/mass legs:", len(filtered_slope_mass100gone))
print("Original CDP concentration legs:", len(total_concentration_cm3))
print("Matched CDP concentration legs:", len(total_concentration_cm3_mass100gone))
# with open("CDP_concentration_massLE1002021.pkl", "wb") as f:
#     pickle.dump(total_concentration_cm3_mass100gone, f)
# print("Saved CDP concentration filtered dataset.")
#%%
# # CDP ddry dataset
filtered_master_BCB_ddry_mass100gone = [
    e for e in filtered_master_BCB_ddry_CDP
    if make_leg_key(e) in good_leg_keys
]
print("Original CDP ddry legs:", len(filtered_master_BCB_ddry_CDP))
print("After matching CDP slope/mass legs:", len(filtered_master_BCB_ddry_mass100gone))
with open("CDP_ddry_massLE1002021.pkl", "wb") as f:
    pickle.dump(filtered_master_BCB_ddry_mass100gone, f)
print("Saved CDP filtered ddry dataset.")
#%%
common_bins = np.linspace(2, 25, 35)
sum_interpolated = np.zeros_like(common_bins, dtype=float)
count_interpolated = np.zeros_like(common_bins, dtype=int)
for entry in filtered_master_BCB_ddry_mass100gone:
    ddry_values = np.array(entry["ddry"])
    dN_dD_dry = np.array(entry["dN/dDdry"])
    valid = np.isfinite(ddry_values) & np.isfinite(dN_dD_dry)
    if valid.sum() < 2:
        continue
    interp_func = interp1d(ddry_values[valid], dN_dD_dry[valid],
                           kind="linear", bounds_error=False, fill_value=np.nan)
    y = interp_func(common_bins)
    good = np.isfinite(y)

    sum_interpolated[good] += y[good]
    count_interpolated[good] += 1

average_dN_dD_dry = np.divide(sum_interpolated, count_interpolated,
                              where=count_interpolated > 0)
print("Number of legs plotted:", len(filtered_master_BCB_ddry_mass100gone))
plt.figure(figsize=(8, 6))
plt.plot(common_bins, average_dN_dD_dry, color="red", linewidth=2,
         label="Average Dry Size Distribution")
plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=20, fontweight="bold")
plt.ylabel(r"CDP Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=20, fontweight="bold")
plt.yscale("log")
plt.ylim(10**-4, 10**0)
plt.xlim(0, 45)
plt.xticks(fontweight="bold", fontsize=20)
plt.yticks(fontweight="bold", fontsize=20)
plt.title("CDP Average Below Cloud Base \nDry Size Distribution\n January - June 2021", fontsize=20, fontweight="bold")
plt.legend()
plt.show()
#%%
#saving the average dry size distribution 
# np.savez("CDP_average_drysize_nomass1002021.npz",
#          bins=common_bins,
#          average=average_dN_dD_dry)
# print("Saved CDP averaged distribution.")
# print("Saved to:", os.path.abspath("CDP_ddry_massLE1002021.pkl"))
# print("Exists?", os.path.exists("CDP_ddry_massLE1002021.pkl"))
# %%
#calculating spherical surface area second moment 
def calculate_mass(N0, D):
    N0_m4 = N0 * 10**6  # Convert cm⁻³µm⁻¹ to m⁻⁴
    integrand = lambda d: np.exp(-d / D) * (d * 1e-6)**2 
    mass_integral, _ = quad(integrand, 2, np.inf)
    return np.pi * N0_m4 * mass_integral  
dry_mass_data_inf = []
for entry in dry_exponential_fits:
    date = entry['Date']
    dry_intercept = entry['Dry_Intercept_n0']
    dry_slope = entry['Dry_E_folding_D']
    if dry_slope > 0 and dry_intercept > 0:
        mass_value = calculate_mass(dry_intercept, dry_slope) * 1e9  # Convert kg/m³ to µg/m³
        dry_mass_data_inf.append({
        'Date': date,
        'BCB_start': entry['BCB_start'], 
        'BCB_stop': entry['BCB_stop'],  
        'Dry Slope (D)': dry_slope,
        'Dry Intercept (N0)': dry_intercept,
        'Dry Mass (µg/m³)': mass_value
    })
dry_slopes = np.array([entry['Dry Slope (D)'] for entry in dry_mass_data_inf])
dry_intercepts = np.array([entry['Dry Intercept (N0)'] for entry in dry_mass_data_inf])
min_slope_threshold = np.percentile(dry_slopes, 1) 
filtered_slopes = dry_slopes[dry_slopes >= min_slope_threshold]
filtered_intercepts = dry_intercepts[dry_slopes >= min_slope_threshold]
x_min, x_max = np.percentile(filtered_slopes, [5, 95])
y_min, y_max = np.percentile(filtered_intercepts, [5, 95])
xgrid_adjusted = np.logspace(np.log10(x_min), np.log10(x_max), 200)
ygrid_adjusted = np.logspace(np.log10(y_min), np.log10(y_max), 200)
D_grid_adjusted, dryintercept_grid_adjusted = np.meshgrid(xgrid_adjusted, ygrid_adjusted)
mass_grid_adjusted = np.zeros_like(D_grid_adjusted)
for i in range(D_grid_adjusted.shape[0]):
    for j in range(D_grid_adjusted.shape[1]):
        mass_grid_adjusted[i, j] = calculate_mass(dryintercept_grid_adjusted[i, j], D_grid_adjusted[i, j]) * 1e9

mass_levels = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]

plt.figure(figsize=(10, 8))
plt.scatter(filtered_slopes, filtered_intercepts, c='blue', s=80, alpha=0.7, label="Dry Data Points")
contour_plot = plt.contour(D_grid_adjusted, dryintercept_grid_adjusted, mass_grid_adjusted, 
                           levels=mass_levels, colors='red', alpha=0.75, linewidths=1.5)

plt.clabel(contour_plot, inline=True, fontsize=13, fmt=lambda x: f"{int(x)} µg/m³", colors='black', inline_spacing=5)
for txt in contour_plot.labelTexts:
    txt.set_fontweight('bold')
    txt.set_rotation(15)
plt.xlabel(r'Dry Slope ($\mu$m)', fontsize=19, fontweight='bold')
plt.ylabel(r'Dry Intercept (cm$^{-3}$ $\mu$m$^{-1}$)', fontsize=19, fontweight='bold')
plt.title('CDP Below Cloud Base January-June 2021\nContours of Spherical Surface Area', fontsize=19, fontweight='bold')
plt.xscale('log')
plt.yscale('log')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
common_bins = np.linspace(2, 25, 35)

plt.figure(figsize=(8, 6))
for entry in filtered_master_BCB_ddry_CDP:
    ddry_values = np.array(entry['ddry'])
    dN_dD_dry = np.array(entry['dN/dDdry'])
    valid_indices = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry)
    if np.sum(valid_indices) < 2:
        continue
    interp_func = interp1d(
        ddry_values[valid_indices],
        dN_dD_dry[valid_indices],
        kind='linear',
        bounds_error=False,
        fill_value=np.nan    )

    interpolated_dN_dD_dry = interp_func(common_bins)
    surface_area_distribution = (
        np.pi * common_bins**2 * interpolated_dN_dD_dry    )

    valid_interpolated_indices = (
        (surface_area_distribution > 0) &
        ~np.isnan(surface_area_distribution)    )

    filtered_bins = common_bins[valid_interpolated_indices]
    filtered_surface_area = surface_area_distribution[
        valid_interpolated_indices    ]

    if len(filtered_bins) > 0:
        plt.plot(
            filtered_bins,
            filtered_surface_area,
            color='purple',
            alpha=0.2        )
plt.xlabel(
    "Dry Bin Center Diameter (μm)",
    fontsize=20,
    fontweight="bold")
plt.ylabel(
    "CDP Aerosol Surface-Area Distribution\n"
    r"($\mu$m$^2$ cm$^{-3}$ $\mu$m$^{-1}$)",
    fontsize=20,
    fontweight="bold")
plt.yscale("log")
plt.xticks(fontweight="bold", fontsize=20)
plt.yticks(fontweight="bold", fontsize=20)
plt.xlim(0.5, 40)
plt.title(
    "CDP Average Below Cloud Base\nJanuary-June 2021\n",
    fontsize=20,
    fontweight="bold")
plt.show()
#%%
#average surface area distribution
common_bins = np.linspace(2, 25, 35)
sum_surface_area_distribution = np.zeros_like(common_bins, dtype=float)
count_surface_area_distribution = np.zeros_like(common_bins, dtype=int)

for entry in filtered_master_BCB_ddry_CDP:
    ddry_values = np.array(entry['ddry'])
    dN_dD_dry = np.array(entry['dN/dDdry'])

    valid_indices = ~np.isnan(ddry_values) & ~np.isnan(dN_dD_dry)

    if np.sum(valid_indices) < 2:
        continue

    interp_func = interp1d(
        ddry_values[valid_indices],
        dN_dD_dry[valid_indices],
        kind='linear',
        bounds_error=False,
        fill_value=np.nan    )

    interpolated_dN_dD_dry = interp_func(common_bins)
    surface_area_distribution = (
        np.pi * common_bins**2 * interpolated_dN_dD_dry    )

    valid_interpolated_indices = (
        ~np.isnan(surface_area_distribution) &
        (surface_area_distribution > 0)    )
    sum_surface_area_distribution[valid_interpolated_indices] += (
        surface_area_distribution[valid_interpolated_indices]    )
    count_surface_area_distribution[valid_interpolated_indices] += 1
average_surface_area_distribution = np.full_like(
    common_bins,
    np.nan,
    dtype=float)
np.divide(
    sum_surface_area_distribution,
    count_surface_area_distribution,
    out=average_surface_area_distribution,
    where=count_surface_area_distribution > 0)
plt.figure(figsize=(8, 6))
plt.plot(
    common_bins,
    average_surface_area_distribution,
    color='red',
    linewidth=2,
    label='Average Surface-Area Distribution')
plt.xlabel(
    "Dry Bin Center Diameter (μm)",
    fontsize=20,
    fontweight="bold")
plt.ylabel(
    "CDP Aerosol Surface-Area Distribution\n"
    r"($\mu$m$^2$ cm$^{-3}$ $\mu$m$^{-1}$)",
    fontsize=18,
    fontweight="bold")
plt.yscale("log")
plt.xlim(0, 45)
plt.xticks(fontweight="bold", fontsize=20)
plt.yticks(fontweight="bold", fontsize=20)
plt.title(
    "CDP Average Below Cloud Base\n"
    "Dry Aerosol Surface-Area Distribution\n"
    "January-June 2021",
    fontsize=18,
    fontweight="bold")
plt.legend()
plt.tight_layout()
plt.show()
#%%
#Now we need to pull our windspeed values from the summary data and calculate the corrected windspeeds down to 10m
master_BCB = []
for j in range(len(dates_legs)):
    date = dates_legs[j]
    leg_dict = leg_data[j]
    BCB_start = np.array(leg_dict['LegIndex_02']['StartTimes'], dtype=float)
    BCB_stop = np.array(leg_dict['LegIndex_02']['StopTimes'], dtype=float)
    sum_flight = summary[j]
    times = pd.to_numeric(sum_flight.Time_mid, errors='coerce').values
    winds = pd.to_numeric(sum_flight.Wind_Speed, errors='coerce').values
    alts = pd.to_numeric(sum_flight.GPS_altitude, errors='coerce').values
    all_BCB_means = []
    for k in range(len(BCB_start)):
        start = BCB_start[k]
        end = BCB_stop[k]
        in_leg = (times >= start) & (times <= end)
        if np.sum(in_leg) > 0:
            winds_mean = np.nanmean(winds[in_leg])
            alts_mean = np.nanmean(alts[in_leg])
        else:
            winds_mean = np.nan
            alts_mean = np.nan
        wind_alt = {
            'Date': date,
            'BCB_start': start,
            'BCB_end': end,
            'Alts_mean': [alts_mean],
            'Winds_mean': [winds_mean]
        }
        all_BCB_means.append(wind_alt)
    master_BCB.append(all_BCB_means)
print("Done")
print("Total BCB legs:", sum(len(x) for x in master_BCB))
#%%
Z0 = 0.02
Z10 = 10 
corrected_calc_bcb = {'Date': [], 'Corrected_bcb_windspeed': []}
for flight in master_BCB:
    for wind_alt in flight:
        date = wind_alt['Date']
        windspeed = wind_alt['Winds_mean']
        altitude = wind_alt['Alts_mean']
        for wind_mean, alt_mean in zip(windspeed, altitude):
            new_windspeed = wind_mean * (np.log(Z10/Z0) / np.log(alt_mean / Z0))

            corrected_calc_bcb['Date'].append(date)
            corrected_calc_bcb['Corrected_bcb_windspeed'].append(new_windspeed)
for date, wind_mean in zip(corrected_calc_bcb['Date'], corrected_calc_bcb['Corrected_bcb_windspeed']):
    print(f"Date: {date}, Corrected_bcb_windspeed: {wind_mean}")
#%%
#histogram of altitudes
all_altitudes = []
for flight in master_BCB:
    for wind_alt in flight:
        all_altitudes.extend(wind_alt['Alts_mean'])
all_altitudes = [alt for alt in all_altitudes if not np.isnan(alt)]
plt.figure(figsize=(10, 8))
plt.hist(all_altitudes, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel('Mean altitude (m)', fontsize=16, fontweight='bold')
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.ylabel('Frequency of flight legs', fontsize=16, fontweight='bold')
plt.title('221 below cloud base legs\n FMAS 2020', fontsize=18, fontweight='bold')
plt.show()
#%%
#mean windspeed
corrected_windspeeds = corrected_calc_bcb['Corrected_bcb_windspeed']
corrected_windspeeds = [ws for ws in corrected_windspeeds if not np.isnan(ws)]
mean_corrected_windspeed = np.mean(corrected_windspeeds)
print(f"Mean Corrected Wind Speed: {mean_corrected_windspeed:.2f} m/s")

#%%
#standard deviation of windspeed
mean_corrected_windspeed = sum(corrected_windspeeds) / len(corrected_windspeeds)
variance = sum((ws - mean_corrected_windspeed) ** 2 for ws in corrected_windspeeds) / (len(corrected_windspeeds) - 1)
std_corrected_windspeed = variance ** 0.5
print(f"Standard Deviation of Corrected Wind Speed: {std_corrected_windspeed:.2f} m/s")

#%%
#Use a dictionary of windspeeds 
def correct_windspeed(windspeed, altitude):
    return windspeed * (np.log(Z10 / Z0) / np.log(altitude / Z0))
combined_data = []
for i, flight in enumerate(master_BCB):
    for j, wind_alt in enumerate(flight):
        try:
            date = wind_alt['Date']
            wind_mean = wind_alt['Winds_mean'][0]
            alt_mean = wind_alt['Alts_mean'][0]
            BCB_start = wind_alt['BCB_start']
            BCB_stop = wind_alt['BCB_end']
            
            if not np.isnan(alt_mean) and alt_mean > 0:
                corrected_windspeed = correct_windspeed(wind_mean, alt_mean)
            else:
                corrected_windspeed = np.nan
            combined_data.append({
                'Date': date,
                'BCB_start': BCB_start,
                'BCB_stop': BCB_stop,
                'Windspeed': corrected_windspeed
            })
        except IndexError as e:
            print(f"Index error at i={i}, j={j}: {e}")
            continue
df_combined = pd.DataFrame(combined_data)
#%%
#monthly trend of corrected windspeed
df_wind = pd.DataFrame(combined_data).copy()
df_wind = df_wind[df_wind["Date"].astype(str).str.startswith("2021-")].copy()
df_wind["Month"] = df_wind["Date"].astype(str).str[5:7].astype(int)
df_wind = df_wind[df_wind["Month"].isin([2, 3, 4, 5, 6])].copy()
df_wind_sorted = df_wind.sort_values(
    ["Date", "BCB_start"],
    kind="mergesort"
).reset_index(drop=True)

wind = df_wind_sorted["Windspeed"].astype(float).values
x = np.arange(len(df_wind_sorted))
plt.figure(figsize=(12, 4.8))
plt.plot(x, wind, '-')
plt.grid(alpha=0.3)
plt.xlabel("Leg index (sorted by Date, then BCB_start)", fontsize=13, fontweight="bold")
plt.ylabel("Corrected Wind Speed (m/s)", fontsize=13, fontweight="bold")
plt.title("Corrected Wind Speed Timeline February-June 2021\nLegs ordered by Date then BCB_start",
          fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
#%%
#color coded by month with
month_name = {2:"February", 3:"March", 4:"April", 5:"May", 6:"June"}
plt.figure(figsize=(12, 4.8))
for m in sorted(df_wind_sorted["Month"].unique()):
    if m not in month_name:
        continue
    m_mask = (df_wind_sorted["Month"].values == m)
    vals = wind[m_mask]
    good = np.isfinite(vals)
    mean_val = np.mean(vals[good]) if np.any(good) else np.nan
    plt.plot(
        x[m_mask],
        wind[m_mask],
        '-',
        label=f"{month_name[m]} (mean: {mean_val:.2f} m/s)"
    )
plt.grid(alpha=0.3)
plt.ylabel("Wind Speed (m/s)", fontsize=16, fontweight="bold")
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.title("BCB Wind Speed\nFebruary - June 2021 Monthly Means",
          fontsize=18, fontweight="bold")
plt.legend(ncol=2, fontsize=10)
plt.yticks(fontsize=14, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
#%%
dfp = pd.DataFrame(combined_data).copy()
dfp = dfp[dfp["Date"].astype(str).str.startswith("2021-")].copy()
dfp["Month"] = dfp["Date"].astype(str).str[5:7].astype(int)
dfp = dfp[dfp["Month"].isin([2, 3, 4, 5, 6])].copy()
dfp["Date_dt"] = pd.to_datetime(dfp["Date"])
sort_cols = ["Date_dt"]
if "BCB_start" in dfp.columns:
    sort_cols.append("BCB_start")
elif "Min_start" in dfp.columns:
    sort_cols.append("Min_start")
dfp = dfp.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
x = np.arange(len(dfp))
wind_arr = pd.to_numeric(dfp["Windspeed"], errors="coerce").to_numpy()
date_first = dfp.groupby(dfp["Date_dt"].dt.date, sort=False).head(1)
tick_pos = date_first.index.to_numpy()
tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
month_name = {2:"February", 3:"March", 4:"April", 5:"May", 6:"June"}
fig_w = max(22, 0.55 * len(tick_pos))
fig, ax = plt.subplots(figsize=(fig_w, 6.2))
for m in sorted(dfp["Month"].unique()):
    if m not in month_name:
        continue
    m_mask = (dfp["Month"].values == m)
    mean_w = np.nanmean(wind_arr[m_mask])
    ax.plot(
        x[m_mask], wind_arr[m_mask],
        '-', linewidth=1.5,
        label=f"{month_name[m]} (mean: {mean_w:.2f} m/s)"
    )
for p in tick_pos:
    ax.axvline(p, color="k", alpha=0.06, linewidth=1)

ax.grid(alpha=0.3)
ax.set_ylabel("Corrected Wind Speed (m/s)", fontsize=16, fontweight="bold")
ax.set_xlabel("Flight Date", fontsize=16, fontweight="bold")
ax.set_title("BCB Corrected Wind Speed February - June 2021\nMonthly Trend",
             fontsize=18, fontweight="bold")
ax.legend(ncol=2, fontsize=10, loc="upper right")
ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_lab, rotation=60, ha="right", fontsize=7, fontweight="bold")
labels = ax.get_xticklabels()
for i, lab in enumerate(labels):
    lab.set_text("\n" * (i % 4) + lab.get_text())
ax.set_xticklabels([lab.get_text() for lab in labels])
fig.subplots_adjust(bottom=0.40)
fig.tight_layout()
plt.show()
#%%
month_colors = {
    2: "tab:blue",    
    3: "tab:orange",  
    4: "tab:green",   
    5: "tab:red",     
    6: "tab:purple"   
}
dfp = df_wind.copy()
dfp["Date_dt"] = pd.to_datetime(dfp["Date"])
if "Month" not in dfp.columns:
    dfp["Month"] = dfp["Date"].astype(str).str[5:7].astype(int)
sort_cols = ["Date_dt"]
if "BCB_start" in dfp.columns:
    sort_cols.append("BCB_start")
elif "Min_start" in dfp.columns:
    sort_cols.append("Min_start")
dfp = dfp.sort_values(sort_cols).reset_index(drop=True)
x = np.arange(len(dfp))
wind_arr = pd.to_numeric(dfp["Windspeed"], errors="coerce").to_numpy()
date_first = dfp.groupby(dfp["Date_dt"].dt.date).head(1)
tick_pos = date_first.index.to_numpy()
tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
fig_w = max(22, 0.55 * len(tick_pos))
fig, ax = plt.subplots(figsize=(fig_w, 6.2))
legend_handles = []
month_name = {2:"February", 3:"March", 4:"April", 5:"May", 6:"June"}
for m in sorted(dfp["Month"].unique()):
    if m not in month_name:
        continue
    m_mask = (dfp["Month"].values == m)
    if not np.any(m_mask):
        continue
    c = month_colors.get(m, "k")
    mean_w   = np.nanmean(wind_arr[m_mask])
    median_w = np.nanmedian(wind_arr[m_mask])
    ax.plot(
        x[m_mask], wind_arr[m_mask],
        '-', linewidth=1.5, color=c
    )

    month_x = x[m_mask]
    mid_x = month_x[len(month_x) // 2]
    ax.plot(
        mid_x, mean_w,
        marker="^", linestyle="None",
        markersize=13,
        markerfacecolor=c,
        markeredgecolor=c,   
        markeredgewidth=2.2,
        zorder=6
    )
    ax.plot(
        mid_x + 5, median_w,
        marker="o", linestyle="None",
        markersize=12,
        markerfacecolor=c,
        markeredgecolor=c,  
        markeredgewidth=2.2,
        zorder=6
    )
    legend_handles.extend([
    Line2D([0], [0], color=c, lw=2, label=month_name[m]),
    Line2D([0], [0], marker="^", lw=0, markersize=10,
           markerfacecolor=c, markeredgecolor=c,
           label=f"{month_name[m]} mean = {mean_w:.2f} m/s"),
    Line2D([0], [0], marker="o", lw=0, markersize=10,
           markerfacecolor=c, markeredgecolor=c,
           label=f"{month_name[m]} median = {median_w:.2f} m/s"),
])

for p in tick_pos:
    ax.axvline(p, color="k", alpha=0.06, linewidth=1)
ax.grid(alpha=0.3)
plt.yticks(fontsize=16, fontweight="bold")
ax.set_ylabel("Wind Speed (m s$^{-1}$)", fontsize=20, fontweight="bold")
ax.set_xlabel("Flight Date", fontsize=20, fontweight="bold")
ax.set_title("Wind Speed January - June 2021\nMonthly Trend",
             fontsize=20, fontweight="bold")
ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_lab, rotation=60, ha="right", fontsize=10, fontweight="bold")
labels = ax.get_xticklabels()
for i, lab in enumerate(labels):
    base = lab.get_text()
    lab.set_text("\n" * (i % 4) + base)
ax.set_xticklabels([lab.get_text() for lab in labels])
target_dates = {"2021-06-05": 0, "2021-06-07": 3}
labels = ax.get_xticklabels()
for lab in labels:
    txt = lab.get_text().replace("\n", "")
    if txt in target_dates:
        lab.set_text("\n" * target_dates[txt] + txt)
ax.set_xticklabels([lab.get_text() for lab in labels])
ax.legend(
    handles=legend_handles,
    ncol=1,
    fontsize=9,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=True
)
fig.subplots_adjust(bottom=0.40)
fig.tight_layout()
plt.show()
# #save to pdf
# fig.savefig("Wind_Speed_Monthly_Trend2021.pdf", bbox_inches="tight")
#%%
common_bins=np.linspace(2, 10, 25)
#%%
#6 windspeed bins
windspeed_bins = [
    (0, 2.5),
    (2.501, 3.5),
    (3.501, 5),
    (5.001, 7),
    (7.001, 9),
    (9.001, np.inf)
]
grouped_distributions = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds = {i: [] for i in range(len(windspeed_bins))}
missing_windspeed_count = 0
common_bins = np.linspace(2, 10, 10)
for entry in dry_exponential_fits_10:
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    n0 = entry['Dry_Intercept_n0']
    D = entry['Dry_E_folding_D']
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) & 
        (df_combined['BCB_start'] == BCB_start) & 
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    if windspeed_entry.empty or np.isnan(n0) or np.isnan(D):
        missing_windspeed_count += 1
        continue
    windspeed = windspeed_entry['Windspeed'].values[0]
    size_dist = n0 * np.exp(-common_bins / D)
    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:
            grouped_distributions[idx].append(size_dist)
            mean_windspeeds[idx].append(windspeed)
            break
for idx, group in grouped_distributions.items():
    print(f"Windspeed bin {idx} ({windspeed_bins[idx]} m/s): {len(group)} legs")
print(f"Total legs with missing windspeed data: {missing_windspeed_count}")
plt.figure(figsize=(10, 8))
for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        avg_distribution = np.mean(grouped_distributions[idx], axis=0)
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])

        plt.plot(common_bins, avg_distribution, label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", linewidth=2.5)
plt.yscale('log')
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=16, fontweight="bold")
plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=16, fontweight="bold")
plt.title('Dry Size Distributions Binned by Average Wind Speed', fontweight='bold', fontsize=17)
plt.legend(title=r"Average wind speed m s$^{-1}$")
plt.tight_layout()
plt.ylim(1e-4, 10**0)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.show()
total_legs = sum(len(group) for group in grouped_distributions.values())
print(f"Total number of legs plotted: {total_legs}")
#%%
#adding error bars 
bin_means = {}
bin_stds = {}
bin_stderrs = {}
for idx, data in grouped_distributions.items():
    if data:
        data_array = np.array(data)
        bin_means[idx] = np.mean(data_array, axis=0) 
        bin_stds[idx] = np.std(data_array, axis=0) 
        bin_stderrs[idx] = bin_stds[idx] / np.sqrt(len(data)) 
plt.figure(figsize=(10, 8))
for idx, (low, high) in enumerate(windspeed_bins):
    if idx in bin_means:
        avg_distribution = bin_means[idx]
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])
        error_bars = bin_stderrs[idx]
        plt.errorbar(
            common_bins, avg_distribution, yerr=error_bars, 
            label=f"{avg_windspeed:.1f} m/s, n={num_legs} legs", 
            linewidth=2.5, capsize=3, capthick=1.5, fmt='-o', markersize=4
        )
plt.yscale('log')
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=22, fontweight="bold")
plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=22, fontweight="bold")
plt.title('Dry Size Distributions Binned by Average Wind Speed', fontweight='bold', fontsize=19)
plt.legend(title=r"Average wind speed m s$^{-1}$")
plt.tight_layout()
plt.ylim(1e-4, 10**0)
plt.xticks(fontsize=19, fontweight='bold')
plt.yticks(fontsize=19, fontweight='bold')
plt.show()
#%%
legend_texts = []
for idx, (low, high) in enumerate(windspeed_bins):
    if idx in bin_means:
        avg_windspeed = np.mean(mean_windspeeds[idx])
        num_legs = len(grouped_distributions[idx])
        avg_std_error = np.mean(bin_stderrs[idx]) 
        legend_texts.append(f"{avg_windspeed:.1f} m/s, n={num_legs} legs\nAvg SE: {avg_std_error:.3f}")
plt.figure(figsize=(10, 8))
for idx, (low, high) in enumerate(windspeed_bins):
    if idx in bin_means:
        avg_distribution = bin_means[idx]
        error_bars = bin_stderrs[idx] 
        plt.errorbar(
            common_bins, avg_distribution, yerr=error_bars,
            label=legend_texts[idx], linewidth=2.5, capsize=3, capthick=1.5, fmt='-o', markersize=4
        )
plt.yscale('log')
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=23, fontweight="bold")
plt.xlabel("Dry Bin Centers Diameter (μm)", fontsize=23, fontweight="bold")
plt.title('CDP Dry Size Distributions \nBinned by Average Wind Speed', fontweight='bold', fontsize=23)
plt.legend(
    title="Windspeed & Error Stats", title_fontsize=20, fontsize=20,
    prop={'weight': 'bold'}, loc="upper right"
)
plt.tight_layout()
plt.ylim(1e-4, 10**0)
plt.xlim(0,10)
plt.xticks(fontsize=21, fontweight='bold')
plt.yticks(fontsize=21, fontweight='bold')
plt.show()
#%%
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
avg_wind_speeds = []
se_wind_speeds = []
bin_leg_counts = []
for idx in range(len(windspeed_bins)):
    windspeeds = mean_windspeeds.get(idx, [])
    if windspeeds:
        windspeeds = np.array(windspeeds)
        avg_ws = np.mean(windspeeds)
        se_ws = np.std(windspeeds, ddof=1) / np.sqrt(len(windspeeds))
        avg_wind_speeds.append(avg_ws)
        se_wind_speeds.append(se_ws)
        bin_leg_counts.append(len(windspeeds))
plt.figure(figsize=(8, 6))

for idx, (x, y) in enumerate(zip(avg_wind_speeds, se_wind_speeds)):
    plt.errorbar(x, y, yerr=0.0, fmt='o', color=colors[idx], markersize=10,
                 ecolor='black', capsize=5, label=f'{x:.1f} m/s')
    plt.text(x, y + 0.02, f'n={bin_leg_counts[idx]}', fontsize=12,
             color=colors[idx], ha='center', fontweight='bold')
plt.xlabel("Average Wind Speed (m s$^{-1}$)", fontsize=20, fontweight="bold")
plt.ylabel("Standard Error (m s$^{-1}$)", fontsize=20, fontweight="bold")
plt.title("Standard Error of Wind Speed", fontsize=20, fontweight="bold")
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=20, fontweight='bold')
plt.yticks(fontsize=20, fontweight='bold')
plt.ylim(0, max(se_wind_speeds) + 0.05)
plt.xlim(0, 12)
plt.tight_layout()
plt.show()
# %%
windspeed_bins = [
    (0, 2.5),
    (2.501, 3.5),
    (3.501, 5),
    (5.001, 7),
    (7.001, 9),
    (9.001, np.inf)
]
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown'] 
bin_edges = np.linspace(2, 10, 11)  
bin_widths = np.diff(bin_edges) 
avg_windspeeds = []
total_concentrations = []
for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:  
        avg_windspeed = np.mean(mean_windspeeds[idx]) 
        avg_concentration_per_leg = [np.sum(dist * bin_widths) for dist in grouped_distributions[idx]]
        avg_concentration = np.mean(avg_concentration_per_leg) 
        avg_windspeeds.append(avg_windspeed)
        total_concentrations.append(avg_concentration)
windspeed_values = np.array(avg_windspeeds)
total_concentrations = np.array(total_concentrations)
def linear_model(x, m, b):
    return m * x + b
popt, _ = curve_fit(linear_model, windspeed_values, total_concentrations)
m_fit, b_fit = popt
residuals = total_concentrations - linear_model(windspeed_values, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((total_concentrations - np.mean(total_concentrations))**2)
r_squared = 1 - (ss_res / ss_tot)
plt.figure(figsize=(8, 6))
for idx in range(len(windspeed_bins)):
    plt.scatter(windspeed_values[idx], total_concentrations[idx], 
                color=colors[idx], s=100, edgecolor='black', zorder=3)
x_fit = np.linspace(min(windspeed_values), max(windspeed_values), 100)
y_fit = linear_model(x_fit, *popt)
plt.plot(x_fit, y_fit, 'r-', label=f'Fit: y = {m_fit:.3f}x + {b_fit:.3f}, R² = {r_squared:.2f}')
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=16, fontweight='bold')
plt.ylabel("Total Wind Speed Bin Concentration (cm$^{-3}$)", fontsize=16, fontweight='bold')
plt.title("Wind Speed and Total Concentration Correlation", fontsize=16, fontweight='bold')
legend_labels = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[idx], markersize=10, 
               label=f"{windspeed_values[idx]:.1f} m/s") for idx in range(len(windspeed_bins))
]
plt.legend(handles=legend_labels + [plt.Line2D([0], [0], color='red', label=f'Fit: y = {m_fit:.3f}x + {b_fit:.3f}, R² = {r_squared:.2f}')], 
           title="Wind Speed Bins", title_fontsize=14, fontsize=13)
plt.tight_layout()
plt.xlim(0,12)
plt.ylim(0.1, 0.7)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.show()
print(f"Slope (m): {m_fit:.3f}")
print(f"Intercept (b): {b_fit:.3f}")
print(f"R² value: {r_squared:.2f}")
#%%
#adding error bars to total concentration vs wind speed 
windspeed_bins = [
    (0, 2.5),
    (2.501, 3.5),
    (3.501, 5),
    (5.001, 7),
    (7.001, 9),
    (9.001, np.inf)
]
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown'] 
bin_edges = np.linspace(2, 10, 11) 
bin_widths = np.diff(bin_edges)  
avg_windspeeds = []
total_concentrations = []
standard_errors = []
for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:  
        avg_windspeed = np.mean(mean_windspeeds[idx]) 
        avg_concentration_per_leg = [np.sum(dist * bin_widths) for dist in grouped_distributions[idx]]
        avg_concentration = np.mean(avg_concentration_per_leg)  
        std_concentration = np.std(avg_concentration_per_leg, ddof=1)  
        N_legs = len(avg_concentration_per_leg) 
        SE_concentration = std_concentration / np.sqrt(N_legs) 

        avg_windspeeds.append(avg_windspeed)
        total_concentrations.append(avg_concentration)
        standard_errors.append(SE_concentration)
windspeed_values = np.array(avg_windspeeds)
total_concentrations = np.array(total_concentrations)
standard_errors = np.array(standard_errors)
def linear_model(x, m, b):
    return m * x + b
popt, _ = curve_fit(linear_model, windspeed_values, total_concentrations)
m_fit, b_fit = popt
residuals = total_concentrations - linear_model(windspeed_values, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((total_concentrations - np.mean(total_concentrations))**2)
r_squared = 1 - (ss_res / ss_tot)
plt.figure(figsize=(8, 6))
for idx in range(len(windspeed_bins)):
    plt.errorbar(windspeed_values[idx], total_concentrations[idx], 
                 yerr=standard_errors[idx], fmt='o', color=colors[idx], 
                 markersize=10, capsize=5, capthick=2, label=f"{windspeed_values[idx]:.1f} m/s", 
                 ecolor='black', elinewidth=1.5, zorder=3)
x_fit = np.linspace(min(windspeed_values), max(windspeed_values), 100)
y_fit = linear_model(x_fit, *popt)
plt.plot(x_fit, y_fit, 'r-', label=f'Fit: y = {m_fit:.3f}x + {b_fit:.3f}, R² = {r_squared:.2f}')
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=16, fontweight='bold')
plt.ylabel("Total Wind Speed Bin Concentration (cm$^{-3}$)", fontsize=16, fontweight='bold')
plt.title("CDP Wind Speed and Total Concentration Correlation", fontsize=16, fontweight='bold')
plt.legend(title="Wind Speed Bins", title_fontsize=14, fontsize=13, loc='best', frameon=True)
plt.tight_layout()
plt.xlim(0, 12)
plt.ylim(0.0, 0.3)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.show()
print(f"Slope (m): {m_fit:.3f}")
print(f"Intercept (b): {b_fit:.3f}")
print(f"R² value: {r_squared:.2f}")
#%%
#adding R value 
windspeed_bins = [
    (0, 2.5),
    (2.501, 3.5),
    (3.501, 5),
    (5.001, 7),
    (7.001, 9),
    (9.001, np.inf)
]
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
bin_edges = np.linspace(2, 10, 11)
bin_widths = np.diff(bin_edges) 
avg_windspeeds = []
total_concentrations = []
standard_errors = []
for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:
        avg_windspeed = np.mean(mean_windspeeds[idx])
        avg_concentration_per_leg = [np.sum(dist * bin_widths) for dist in grouped_distributions[idx]]
        avg_concentration = np.mean(avg_concentration_per_leg)  
        std_concentration = np.std(avg_concentration_per_leg, ddof=1) 
        N_legs = len(avg_concentration_per_leg) 
        SE_concentration = std_concentration / np.sqrt(N_legs)
        avg_windspeeds.append(avg_windspeed)
        total_concentrations.append(avg_concentration)
        standard_errors.append(SE_concentration)
windspeed_values = np.array(avg_windspeeds)
total_concentrations = np.array(total_concentrations)
standard_errors = np.array(standard_errors)
def linear_model(x, m, b):
    return m * x + b
popt, _ = curve_fit(linear_model, windspeed_values, total_concentrations)
m_fit, b_fit = popt
residuals = total_concentrations - linear_model(windspeed_values, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((total_concentrations - np.mean(total_concentrations))**2)
r_squared = 1 - (ss_res / ss_tot)
r_value = np.sign(m_fit) * np.sqrt(r_squared)
plt.figure(figsize=(8, 6))
plt.errorbar(windspeed_values, total_concentrations, 
             yerr=standard_errors, fmt='o', color='blue', 
             markersize=10, capsize=5, capthick=2, label="CAS", 
             ecolor='black', elinewidth=1.5, zorder=3)
x_fit = np.linspace(min(windspeed_values), max(windspeed_values), 100)
y_fit = linear_model(x_fit, *popt)
plt.plot(x_fit, y_fit, 'r-', label=f'Fit: y = {m_fit:.3f}x + {b_fit:.3f}\nR² = {r_squared:.2f}, R = {r_value:.2f}')
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=16, fontweight='bold')
plt.ylabel("Total Wind Speed Bin Concentration (cm$^{-3}$)", fontsize=16, fontweight='bold')
plt.title("CDP Wind Speed and Total Concentration Correlation", fontsize=16, fontweight='bold')
plt.legend(fontsize=13, title_fontsize=14, loc='best', frameon=True)
plt.tight_layout()
plt.xlim(0, 12)
plt.ylim(0.0, 0.3)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.show()
print(f"Slope (m): {m_fit:.3f}")
print(f"Intercept (b): {b_fit:.3f}")
print(f"R² value: {r_squared:.2f}")
print(f"R value (Pearson correlation): {r_value:.3f}")
# %%
#2sigma
plt.figure(figsize=(8, 6))
plt.errorbar(windspeed_values, total_concentrations,
             yerr=2 * standard_errors, fmt='none',
             ecolor='lightgray', elinewidth=6, capsize=0, label='95% (±2σ)', zorder=1)
plt.errorbar(windspeed_values, total_concentrations, 
             yerr=standard_errors, fmt='o', color='blue', 
             markersize=10, capsize=5, capthick=2, label="CAS", 
             ecolor='black', elinewidth=1.5, zorder=3)
x_fit = np.linspace(min(windspeed_values), max(windspeed_values), 100)
y_fit = linear_model(x_fit, *popt)
plt.plot(x_fit, y_fit, 'r-', label=f'Fit: y = {m_fit:.3f}x + {b_fit:.3f}\nR² = {r_squared:.2f}, R = {r_value:.2f}')
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=20, fontweight='bold')
plt.ylabel("Total Wind Speed Bin \nConcentration (cm$^{-3}$)", fontsize=20, fontweight='bold')
plt.title("CDP Wind Speed and Total Concentration", fontsize=20, fontweight='bold')
plt.legend(fontsize=13, title_fontsize=14, loc='best', frameon=True)
plt.tight_layout()
plt.xlim(0, 12)
plt.ylim(0.0, 0.3)
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')
plt.show()
#%%
#adding slope uncertainty
windspeed_bins = [
    (0, 2.5),
    (2.501, 3.5),
    (3.501, 5),
    (5.001, 7),
    (7.001, 9),
    (9.001, np.inf)
]
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
bin_edges = np.linspace(2, 10, 11)
bin_widths = np.diff(bin_edges) 
avg_windspeeds = []
total_concentrations = []
standard_errors = []
for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:  
        avg_windspeed = np.mean(mean_windspeeds[idx])
        avg_concentration_per_leg = [np.sum(dist * bin_widths) for dist in grouped_distributions[idx]]
        avg_concentration = np.mean(avg_concentration_per_leg)
        std_concentration = np.std(avg_concentration_per_leg, ddof=1)
        N_legs = len(avg_concentration_per_leg)
        SE_concentration = std_concentration / np.sqrt(N_legs)
        avg_windspeeds.append(avg_windspeed)
        total_concentrations.append(avg_concentration)
        standard_errors.append(SE_concentration)
windspeed_values = np.array(avg_windspeeds)
total_concentrations = np.array(total_concentrations)
standard_errors = np.array(standard_errors)
def linear_model(x, m, b):
    return m * x + b
popt, pcov = curve_fit(linear_model, windspeed_values, total_concentrations)
m_fit, b_fit = popt
perr = np.sqrt(np.diag(pcov))
m_err, b_err = perr
residuals = total_concentrations - linear_model(windspeed_values, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((total_concentrations - np.mean(total_concentrations))**2)
r_squared = 1 - (ss_res / ss_tot)
r_value = np.sign(m_fit) * np.sqrt(r_squared)
plt.figure(figsize=(8, 6))
plt.errorbar(windspeed_values, total_concentrations, 
             yerr=standard_errors, fmt='o', color='blue', 
             markersize=10, capsize=5, capthick=2, label="CAS", 
             ecolor='black', elinewidth=1.5, zorder=3)

x_fit = np.linspace(min(windspeed_values), max(windspeed_values), 100)
y_fit = linear_model(x_fit, *popt)
plt.plot(x_fit, y_fit, 'r-', 
         label=f'Fit: y = ({m_fit:.3f}±{m_err:.3f})x + {b_fit:.3f}\nR² = {r_squared:.2f}, R = {r_value:.2f}')
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=16, fontweight='bold')
plt.ylabel("Total Wind Speed Bin Concentration (cm$^{-3}$)", fontsize=16, fontweight='bold')
plt.title("CDP Wind Speed and Total Concentration Correlation", fontsize=16, fontweight='bold')
plt.legend(fontsize=13, title_fontsize=14, loc='best', frameon=True)
plt.tight_layout()
plt.xlim(0, 12)
plt.ylim(0.0, 0.3)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.show()
print(f"Slope (m): {m_fit:.3f} ± {m_err:.3f}")
print(f"Intercept (b): {b_fit:.3f}")
print(f"R² value: {r_squared:.2f}")
print(f"R value (Pearson correlation): {r_value:.3f}")
plt.figure(figsize=(8, 6))
plt.errorbar(windspeed_values, total_concentrations,
             yerr=2 * standard_errors, fmt='none',
             ecolor='lightgray', elinewidth=6, capsize=0, label='95% (±2σ)', zorder=1)
plt.errorbar(windspeed_values, total_concentrations, 
             yerr=standard_errors, fmt='o', color='#4daf4a', 
             markersize=10, capsize=5, capthick=2, label="CAS", 
             ecolor='black', elinewidth=1.5, zorder=3)
plt.plot(x_fit, y_fit, '-', color='#984ea3', linewidth=2.5,
         label=f'Fit: y = ({m_fit:.3f}±{m_err:.3f})x + {b_fit:.3f}\nR² = {r_squared:.2f}, R = {r_value:.2f}')
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=20, fontweight='bold')
plt.ylabel("Total Wind Speed Bin \nConcentration (cm$^{-3}$)", fontsize=20, fontweight='bold')
plt.title("CDP Below Cloud Base \n January-June 2022", fontsize=20, fontweight='bold')
plt.legend(
    fontsize=16,
    title_fontsize=14,
    loc='upper left',     
    frameon=False        
)
plt.tight_layout()
plt.xlim(0, 12)
plt.ylim(0, 1)
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')
plt.show()
#%%
#calculating counting error bars
bin_edges = np.linspace(2, 10, 11)       # μm
common_bins = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_widths_um = np.diff(bin_edges)
sample_area_cm2 = 0.00323          # CDP sample area in cm²
plane_speed_cm_s = 1.2e4          # 120 m/s = 12000 cm/s
sampling_time_s = 198             # 3.3 minutes
T = sampling_time_s
V = sample_area_cm2 * plane_speed_cm_s  # cm³/s
windspeed_bins = [
    (0, 2.5), (2.501, 3.5), (3.501, 5),
    (5.001, 7), (7.001, 9), (9.001, np.inf)]
grouped_absolute_errors = {i: [] for i in range(len(windspeed_bins))}
for entry in dry_exponential_fits_10:
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    n0 = entry['Dry_Intercept_n0']
    D = entry['Dry_E_folding_D']
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    if windspeed_entry.empty or np.isnan(n0) or np.isnan(D):
        continue
    windspeed = windspeed_entry['Windspeed'].values[0]
    n_i = n0 * np.exp(-common_bins / D)   # cm⁻³ μm⁻¹
    summation = np.nansum(n_i * (bin_widths_um**2))  # cm⁻³ · μm²
    concentration_error = np.sqrt(summation / (T * V))  # cm⁻³
    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:
            grouped_absolute_errors[idx].append(concentration_error)
            break
median_absolute_errors_per_bin = []
for idx, errors in grouped_absolute_errors.items():
    if errors:
        median_abs_err = np.nanmedian(errors)
    else:
        median_abs_err = np.nan
    median_absolute_errors_per_bin.append(median_abs_err)
    print(f"Bin {idx} ({windspeed_bins[idx]}): median abs error = {median_abs_err:.4f} cm⁻³")
avg_windspeeds = []
total_concentrations = []
standard_errors = []
for idx, (low, high) in enumerate(windspeed_bins):
    if grouped_distributions[idx]:  
        avg_windspeed = np.mean(mean_windspeeds[idx])
        avg_concentration_per_leg = [np.sum(dist * bin_widths_um) for dist in grouped_distributions[idx]]
        avg_concentration = np.mean(avg_concentration_per_leg)
        std_concentration = np.std(avg_concentration_per_leg, ddof=1)
        N_legs = len(avg_concentration_per_leg)
        SE_concentration = std_concentration / np.sqrt(N_legs)
        avg_windspeeds.append(avg_windspeed)
        total_concentrations.append(avg_concentration)
        standard_errors.append(SE_concentration)
windspeed_values = np.array(avg_windspeeds)
total_concentrations = np.array(total_concentrations)
standard_errors = 2 * np.array(standard_errors)           # now ±2 SE
counting_errors_CAS = 2 * np.array(median_absolute_errors_per_bin)  # now ±2σ counting error
def linear_model(x, m, b):
    return m * x + b
popt, pcov = curve_fit(linear_model, windspeed_values, total_concentrations)
m_fit, b_fit = popt
perr = np.sqrt(np.diag(pcov))
m_err, b_err = perr
residuals = total_concentrations - linear_model(windspeed_values, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((total_concentrations - np.mean(total_concentrations))**2)
r_squared = 1 - (ss_res / ss_tot)
r_value = np.sign(m_fit) * np.sqrt(r_squared)
x_fit = np.linspace(min(windspeed_values), max(windspeed_values), 100)
y_fit = linear_model(x_fit, *popt)
plt.figure(figsize=(8, 6))
plt.errorbar(windspeed_values, total_concentrations,
             yerr=standard_errors, fmt='o', color='black',
             ecolor='black', elinewidth=1.5, capsize=5, capthick=2,
             label="±2 Standard Errors", zorder=3)
plt.errorbar(windspeed_values - 0.2, total_concentrations,
             yerr=counting_errors_CAS,
             fmt='s', 
             markersize=4,
             markerfacecolor='#8c510a',
             markeredgecolor='black',
             ecolor='#8c510a',
             elinewidth=4,
             capsize=8,
             capthick=3,
             label="±2σ CDP Instrument Error",
             zorder=2)


plt.plot(x_fit, y_fit, '-', color='black', linewidth=2.5,
         label=f'Fit: y = ({m_fit:.3f}±{m_err:.3f})x + {b_fit:.3f}\nR² = {r_squared:.2f}, R = {r_value:.2f}')
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=20, fontweight='bold')
plt.ylabel("Total Wind Speed Bin \nConcentration (cm$^{-3}$)", fontsize=20, fontweight='bold')
plt.title("CDP Below Cloud Base \n January-June 2021", fontsize=20, fontweight='bold')
plt.legend(fontsize=11, title_fontsize=14, loc='upper left', frameon=False)
plt.tight_layout()
plt.xlim(0, 12)
plt.ylim(0, 1)
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')
plt.show()
#%%
save_dir = "/home/disk/eos4/kathem24/activate/data/2021/CDP/1Hz"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "CDP_2021_wind_bin_concentration_plot_data.pkl")
plot_data = {
    "windspeed_bins": windspeed_bins,
    "bin_edges": bin_edges,
    "common_bins": common_bins,
    "bin_widths_um": bin_widths_um,

    "avg_windspeeds": avg_windspeeds,
    "windspeed_values": windspeed_values,
    "total_concentrations": total_concentrations,

    "standard_errors_2SE": standard_errors,
    "counting_errors_CAS_2sigma": counting_errors_CAS,
    "median_absolute_errors_per_bin": median_absolute_errors_per_bin,

    "m_fit": m_fit,
    "b_fit": b_fit,
    "m_err": m_err,
    "b_err": b_err,
    "r_squared": r_squared,
    "r_value": r_value,
    "x_fit": x_fit,
    "y_fit": y_fit,

    "grouped_distributions": grouped_distributions,
    "mean_windspeeds": mean_windspeeds,
}

with open(save_path, "wb") as f:
    pickle.dump(plot_data, f)
print(f"Saved plot data to: {save_path}")
#%%
# import pickle
# load_path = "/home/disk/eos4/kathem24/activate/data/2020/CDP/CDP_FMAS2020_wind_bin_concentration_plot_data.pkl"
# with open(load_path, "rb") as f:
#     plot_data = pickle.load(f)

# windspeed_values = plot_data["windspeed_values"]
# total_concentrations = plot_data["total_concentrations"]
# standard_errors = plot_data["standard_errors_2SE"]
# counting_errors_CAS = plot_data["counting_errors_CAS_2sigma"]

# m_fit = plot_data["m_fit"]
# b_fit = plot_data["b_fit"]
# m_err = plot_data["m_err"]
# b_err = plot_data["b_err"]
# r_squared = plot_data["r_squared"]
# r_value = plot_data["r_value"]
# x_fit = plot_data["x_fit"]
# y_fit = plot_data["y_fit"]
# plt.figure(figsize=(8, 6))

# plt.errorbar(
#     windspeed_values, total_concentrations,
#     yerr=standard_errors,
#     fmt='o', color='black',
#     ecolor='black', elinewidth=1.5,
#     capsize=5, capthick=2,
#     label="±2 Standard Errors",
#     zorder=3
# )

# plt.errorbar(
#     windspeed_values - 0.2, total_concentrations,
#     yerr=counting_errors_CAS,
#     fmt='s',
#     markersize=4,
#     markerfacecolor='#8c510a',
#     markeredgecolor='black',
#     ecolor='#8c510a',
#     elinewidth=4,
#     capsize=8,
#     capthick=3,
#     label="±2σ CAS Instrument Error",
#     zorder=2
# )

# plt.plot(
#     x_fit, y_fit, '-',
#     color='black', linewidth=2.5,
#     label=f'Fit: y = ({m_fit:.3f}±{m_err:.3f})x + {b_fit:.3f}\nR² = {r_squared:.2f}, R = {r_value:.2f}'
# )

# plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=20, fontweight='bold')
# plt.ylabel("Total Wind Speed Bin \nConcentration (cm$^{-3}$)", fontsize=20, fontweight='bold')
# plt.title("CDP Below Cloud Base \n FMAS 2020", fontsize=20, fontweight='bold')
# plt.legend(fontsize=13, title_fontsize=14, loc='upper left', frameon=False)
# plt.tight_layout()
# plt.xlim(0, 12)
# plt.ylim(0, 1)
# plt.xticks(fontsize=18, fontweight='bold')
# plt.yticks(fontsize=18, fontweight='bold')
# plt.show()
#%%
print("CDP concentration per-bin points:")
for i, (x, y, se2, ce2) in enumerate(zip(windspeed_values,
                                         total_concentrations,
                                         standard_errors,
                                         counting_errors_CAS)):
    print(f"  Bin{i}: WS={x:.2f} m/s, Conc={y:.3f} cm⁻³, ±2SE={se2:.3f}, ±2σCountErr≈{ce2:.3f}")
def linear_model(x, m, b):
    return m * x + b
mask = np.isfinite(windspeed_values) & np.isfinite(total_concentrations)
x = windspeed_values[mask]
y = total_concentrations[mask]
yerr = standard_errors[mask]  
popt, pcov = curve_fit(linear_model, x, y)
m_fit, b_fit = popt
perr = np.sqrt(np.diag(pcov))
m_err, b_err = perr
residuals = y - linear_model(x, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y - np.mean(y))**2)
r_squared = 1 - (ss_res / ss_tot)
r_value = np.sign(m_fit) * np.sqrt(r_squared)
x_fit = np.linspace(min(x), max(x), 100)
y_fit = linear_model(x_fit, *popt)
print("\nFit results:")
print(f"Slope = {m_fit:.4f} ± {m_err:.4f}")
print(f"Intercept = {b_fit:.4f} ± {b_err:.4f}")
print(f"R² = {r_squared:.3f}")
print(f"R  = {r_value:.3f}")
# %%
#total mass against wind speed 
grouped_mass_values = {i: [] for i in range(len(windspeed_bins))}
mean_windspeeds_mass = {i: [] for i in range(len(windspeed_bins))}
for mass_entry in filtered_dry_mass_inf:
    date = mass_entry['Date']
    BCB_start = mass_entry['BCB_start']
    BCB_stop = mass_entry['BCB_stop']
    windspeed_entry = df_combined[
        (df_combined['Date'] == date) & 
        (df_combined['BCB_start'] == BCB_start) & 
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    if windspeed_entry.empty:
        continue  
    windspeed = windspeed_entry['Windspeed'].values[0]
    mass_value = mass_entry['Dry Mass (µg/m³)']
    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:
            grouped_mass_values[idx].append(mass_value)
            mean_windspeeds_mass[idx].append(windspeed)
            break

# %%
# Compute average wind speed and mass per bin
avg_windspeeds_mass = []
total_mass_values = []
for idx, mass_list in grouped_mass_values.items():
    if mass_list:
        avg_windspeed = np.mean(mean_windspeeds_mass[idx])  
        avg_mass = np.mean(mass_list)
        avg_windspeeds_mass.append(avg_windspeed)
        total_mass_values.append(avg_mass)
windspeed_values_mass = np.array(avg_windspeeds_mass)
total_mass_values = np.array(total_mass_values)
popt_mass, _ = curve_fit(linear_model, windspeed_values_mass, total_mass_values)
m_fit_mass, b_fit_mass = popt_mass
residuals_mass = total_mass_values - linear_model(windspeed_values_mass, *popt_mass)
ss_res_mass = np.sum(residuals_mass**2)
ss_tot_mass = np.sum((total_mass_values - np.mean(total_mass_values))**2)
r_squared_mass = 1 - (ss_res_mass / ss_tot_mass)

# %%
plt.figure(figsize=(8, 6))
for idx in range(len(windspeed_bins)):
    plt.scatter(windspeed_values_mass[idx], total_mass_values[idx], 
                color="blue", s=100, edgecolor='black', zorder=3)
x_fit_mass = np.linspace(min(windspeed_values_mass), max(windspeed_values_mass), 100)
y_fit_mass = linear_model(x_fit_mass, *popt_mass)
plt.plot(x_fit_mass, y_fit_mass, 'r-', label=f'Fit: y = {m_fit_mass:.3f}x + {b_fit_mass:.3f}, R² = {r_squared_mass:.2f}')
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=16, fontweight='bold')
plt.ylabel("Total Dry Mass (µg/m³)", fontsize=16, fontweight='bold')
plt.title("CDP Below Cloud Base January-June 2021", fontsize=16, fontweight='bold')
legend_labels_mass = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[idx], markersize=10, 
               label=f"{windspeed_values_mass[idx]:.1f} m/s") for idx in range(len(windspeed_bins))
]
plt.legend(handles=legend_labels_mass + [plt.Line2D([0], [0], color='red', 
               label=f'Fit: y = {m_fit_mass:.3f}x + {b_fit_mass:.3f}, R² = {r_squared_mass:.2f}')], 
           title="Wind Speed Bins", title_fontsize=14, fontsize=13)
plt.tight_layout()
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.show()
print(f"Slope (m): {m_fit_mass:.3f}")
print(f"Intercept (b): {b_fit_mass:.3f}")
print(f"R² value: {r_squared_mass:.2f}")
#%%
windspeed_bins = [
    (0, 2.5),
    (2.501, 3.5),
    (3.501, 5),
    (5.001, 7),
    (7.001, 9),
    (9.001, np.inf)
]
avg_windspeeds_mass = []
total_mass_values = []
standard_errors_mass = []
for idx, mass_list in grouped_mass_values.items():
    if mass_list:
        avg_windspeed = np.mean(mean_windspeeds_mass[idx]) 
        avg_mass = np.mean(mass_list) 
        std_mass = np.std(mass_list, ddof=1) 
        N_mass = len(mass_list)  
        SE_mass = std_mass / np.sqrt(N_mass) 

        avg_windspeeds_mass.append(avg_windspeed)
        total_mass_values.append(avg_mass)
        standard_errors_mass.append(SE_mass)
windspeed_values_mass = np.array(avg_windspeeds_mass)
total_mass_values = np.array(total_mass_values)
standard_errors_mass = np.array(standard_errors_mass)
def linear_model(x, m, b):
    return m * x + b

popt_mass, _ = curve_fit(linear_model, windspeed_values_mass, total_mass_values)
m_fit_mass, b_fit_mass = popt_mass
residuals_mass = total_mass_values - linear_model(windspeed_values_mass, *popt_mass)
ss_res_mass = np.sum(residuals_mass**2)
ss_tot_mass = np.sum((total_mass_values - np.mean(total_mass_values))**2)
r_squared_mass = 1 - (ss_res_mass / ss_tot_mass)
r_value_mass = np.sign(m_fit_mass) * np.sqrt(r_squared_mass)
plt.figure(figsize=(8, 6))
plt.errorbar(windspeed_values_mass, total_mass_values, 
             yerr=standard_errors_mass, fmt='o', color='blue', 
             markersize=10, capsize=5, capthick=2, label="CDP", 
             ecolor='black', elinewidth=1.5, zorder=3)
x_fit_mass = np.linspace(min(windspeed_values_mass), max(windspeed_values_mass), 100)
y_fit_mass = linear_model(x_fit_mass, *popt_mass)
plt.plot(x_fit_mass, y_fit_mass, 'r-', label=f'Fit: y = {m_fit_mass:.3f}x + {b_fit_mass:.3f}\nR² = {r_squared_mass:.2f}, R = {r_value_mass:.2f}')
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=16, fontweight='bold')
plt.ylabel("Total Dry Mass (µg/m³)", fontsize=16, fontweight='bold')
plt.title("CDP Below Cloud Base January-June 2021", fontsize=16, fontweight='bold')
plt.legend(fontsize=13, title_fontsize=14, loc='best', frameon=True)
plt.tight_layout()
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.ylim(0,35)
plt.xlim()
plt.show()
print(f"Slope (m): {m_fit_mass:.3f}")
print(f"Intercept (b): {b_fit_mass:.3f}")
print(f"R² value: {r_squared_mass:.2f}")
print(f"R value (Pearson correlation): {r_value_mass:.3f}")
#%%
#2(sigma) uncertainty
plt.figure(figsize=(8, 6))
plt.errorbar(windspeed_values_mass, total_mass_values,
             yerr=2 * standard_errors_mass, fmt='none',
             ecolor='lightgray', elinewidth=6, capsize=0, label='95% (±2σ)', zorder=1)

plt.errorbar(windspeed_values_mass, total_mass_values, 
             yerr=standard_errors_mass, fmt='o', color='blue', 
             markersize=10, capsize=5, capthick=2, label="CDP", 
             ecolor='black', elinewidth=1.5, zorder=3)
x_fit_mass = np.linspace(min(windspeed_values_mass), max(windspeed_values_mass), 100)
y_fit_mass = linear_model(x_fit_mass, *popt_mass)
plt.plot(x_fit_mass, y_fit_mass, 'r-', label=f'Fit: y = {m_fit_mass:.3f}x + {b_fit_mass:.3f}\nR² = {r_squared_mass:.2f}, R = {r_value_mass:.2f}')
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=20, fontweight='bold')
plt.ylabel("Total Dry Mass (µg/m³)", fontsize=20, fontweight='bold')
plt.title("CDP Below Cloud Base January-June 2021", fontsize=20, fontweight='bold')
plt.legend(fontsize=13, title_fontsize=14, loc='best', frameon=True)
plt.tight_layout()
plt.xticks(fontsize=20, fontweight='bold')
plt.yticks(fontsize=20, fontweight='bold')
plt.ylim(0, 35)
plt.xlim(0, 12)
plt.show()
#%%
windspeed_bins = [
    (0, 2.5),
    (2.501, 3.5),
    (3.501, 5),
    (5.001, 7),
    (7.001, 9),
    (9.001, np.inf)
]
avg_windspeeds_mass = []
total_mass_values = []
standard_errors_mass = []
for idx, mass_list in grouped_mass_values.items():
    if mass_list:
        avg_windspeed = np.mean(mean_windspeeds_mass[idx])  
        avg_mass = np.mean(mass_list) 
        std_mass = np.std(mass_list, ddof=1) 
        N_mass = len(mass_list) 
        SE_mass = std_mass / np.sqrt(N_mass)
        avg_windspeeds_mass.append(avg_windspeed)
        total_mass_values.append(avg_mass)
        standard_errors_mass.append(SE_mass)
windspeed_values_mass = np.array(avg_windspeeds_mass)
total_mass_values = np.array(total_mass_values)
standard_errors_mass = np.array(standard_errors_mass)
def linear_model(x, m, b):
    return m * x + b

popt_mass, pcov_mass = curve_fit(linear_model, windspeed_values_mass, total_mass_values)
m_fit_mass, b_fit_mass = popt_mass
perr_mass = np.sqrt(np.diag(pcov_mass))
m_err_mass, b_err_mass = perr_mass
residuals_mass = total_mass_values - linear_model(windspeed_values_mass, *popt_mass)
ss_res_mass = np.sum(residuals_mass**2)
ss_tot_mass = np.sum((total_mass_values - np.mean(total_mass_values))**2)
r_squared_mass = 1 - (ss_res_mass / ss_tot_mass)
r_value_mass = np.sign(m_fit_mass) * np.sqrt(r_squared_mass)
plt.figure(figsize=(8, 6))

plt.errorbar(windspeed_values_mass, total_mass_values, 
             yerr=standard_errors_mass, fmt='o', color='#4daf4a',  # green
             markersize=10, capsize=5, capthick=2, label="CDP", 
             ecolor='black', elinewidth=1.5, zorder=3)

x_fit_mass = np.linspace(min(windspeed_values_mass), max(windspeed_values_mass), 100)
y_fit_mass = linear_model(x_fit_mass, *popt_mass)

plt.plot(x_fit_mass, y_fit_mass, '-', color='#984ea3', linewidth=2.5,  # purple
         label=f'Fit: y = ({m_fit_mass:.3f}±{m_err_mass:.3f})x + {b_fit_mass:.3f}\nR² = {r_squared_mass:.2f}, R = {r_value_mass:.2f}')

plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=16, fontweight='bold')
plt.ylabel("Total Dry Mass (µg/m³)", fontsize=16, fontweight='bold')
plt.title("CDP Below Cloud Base January-June 2021", fontsize=16, fontweight='bold')

plt.legend(fontsize=13, loc='upper left', frameon=False)
plt.tight_layout()
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.ylim(0, 35)
plt.xlim(0, 12)
plt.show()
print(f"Slope (m): {m_fit_mass:.3f} ± {m_err_mass:.3f}")
print(f"Intercept (b): {b_fit_mass:.3f}")
print(f"R² value: {r_squared_mass:.2f}")
print(f"R value (Pearson correlation): {r_value_mass:.3f}")
plt.figure(figsize=(8, 6))
plt.errorbar(windspeed_values_mass, total_mass_values,
             yerr=2 * standard_errors_mass, fmt='none',
             ecolor='lightgray', elinewidth=6, capsize=0, label='95% (±2σ)', zorder=1)
plt.errorbar(windspeed_values_mass, total_mass_values, 
             yerr=standard_errors_mass, fmt='o', color='#4daf4a',
             markersize=10, capsize=5, capthick=2, label="CDP", 
             ecolor='black', elinewidth=1.5, zorder=3)
plt.plot(x_fit_mass, y_fit_mass, '-', color='#984ea3', linewidth=2.5,
         label=f'Fit: y = ({m_fit_mass:.3f}±{m_err_mass:.3f})x + {b_fit_mass:.3f}\nR² = {r_squared_mass:.2f}, R = {r_value_mass:.2f}')

plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=20, fontweight='bold')
plt.ylabel("Total Dry Mass (µg/m³)", fontsize=20, fontweight='bold')
plt.title("CDP Below Cloud Base \nJanuary-June 2021", fontsize=20, fontweight='bold')
plt.legend(fontsize=16, loc='upper left', frameon=False)
plt.tight_layout()
plt.xticks(fontsize=20, fontweight='bold')
plt.yticks(fontsize=20, fontweight='bold')
plt.ylim(0, 25)
plt.xlim(0, 12)
plt.show()
#%%
windspeed_bins = [
    (0, 2.5),
    (2.501, 3.5),
    (3.501, 5),
    (5.001, 7),
    (7.001, 9),
    (9.001, np.inf)
]
bin_centers_um = np.array([
    2.25, 2.75, 3.25, 3.75, 4.5, 5.75,
    6.85, 7.55, 9.05, 11.4, 13.8, 17.5,
    22.5, 27.5, 32.5, 37.5, 42.5, 47.5
])
bin_widths_um = np.diff(np.concatenate(([2], bin_centers_um + np.diff(bin_centers_um, prepend=0)/2)))
radii_um = bin_centers_um / 2
windspeed_values_mass = []
total_mass_values = []
standard_errors_mass = []

for entry in filtered_dry_mass_inf:  
    date = entry['Date']
    start = entry['BCB_start']
    stop = entry['BCB_stop']
    total_mass = entry['Dry Mass (µg/m³)']
    ws_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == start) &
        (df_combined['BCB_stop'] == stop)
    ]
    if ws_entry.empty or np.isnan(total_mass):
        continue

    windspeed = ws_entry['Windspeed'].values[0]
    windspeed_values_mass.append(windspeed)
    total_mass_values.append(total_mass)
    if 'Standard_Error' in entry:
        standard_errors_mass.append(entry['Standard_Error'])
    else:
        standard_errors_mass.append(0) 
windspeed_values_mass = np.array(windspeed_values_mass)
total_mass_values = np.array(total_mass_values)
standard_errors_mass = np.array(standard_errors_mass)
#%%
sample_area_cm2 = 0.00323  # CDP sample area in cm²
plane_speed_cm_s = 1.2e4
sampling_time_s = 198  # seconds
T = sampling_time_s
V = sample_area_cm2 * plane_speed_cm_s  # cm³/s
bin_centers_um = np.array([2.25, 2.75, 3.25, 3.75, 4.5, 5.75,
                           6.85, 7.55, 9.05, 11.4, 13.8, 17.5,
                           22.5, 27.5, 32.5, 37.5, 42.5, 47.5])
bin_widths_um = np.diff(np.concatenate(([2], bin_centers_um + np.diff(bin_centers_um, prepend=0)/2)))
radii_um = bin_centers_um / 2
rho_salt_ug_cm3 = 2.2e6  # µg/cm³
eta = (4/3) * np.pi * rho_salt_ug_cm3  # µg/cm³
grouped_mass_errors = {i: [] for i in range(len(windspeed_bins))}

for entry in dry_exponential_fits_10:
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    n0 = entry['Dry_Intercept_n0']
    D = entry['Dry_E_folding_D']
    ws_entry = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    if ws_entry.empty or np.isnan(n0) or np.isnan(D):
        continue
    windspeed = ws_entry['Windspeed'].values[0]
    n_i = n0 * np.exp(-bin_centers_um / D)  # cm⁻³/μm
    radii_cm = radii_um * 1e-4
    bin_widths_cm = bin_widths_um * 1e-4
    n_i = n0 * np.exp(-bin_centers_um / D)
    print(f"\nDate: {date}, BCB_start: {BCB_start}, Windspeed: {windspeed:.2f}")
    print(f"  n0: {n0:.4e}, D: {D:.4f}")
    print(f"  n_i (first 5): {n_i[:5]}")
    print(f"  bin_widths_cm (first 5): {bin_widths_cm[:5]}")
    print(f"  radii_cm (first 5): {radii_cm[:5]}")

    term = n_i * bin_widths_cm * (radii_cm ** 6)
    print(f"  term (first 5): {term[:5]}")
    print(f"  sum(term): {np.nansum(term):.4e}")

    if np.nansum(term) == 0:
        print(" WARNING: Term sum is zero — likely cause of mass_error = 0")
    summation = np.nansum(term)
    mass_error = eta * np.sqrt(summation / (T * V))  # µg/m³


    for idx, (low, high) in enumerate(windspeed_bins):
        if low <= windspeed < high:
            grouped_mass_errors[idx].append(mass_error)
            break
median_mass_errors_per_bin = []
for idx, errs in grouped_mass_errors.items():
    if errs:
        median_mass_errors_per_bin.append(np.nanmedian(errs))
    else:
        median_mass_errors_per_bin.append(np.nan)
print("\nMass Counting Errors per Wind Speed Bin:")
for idx, (bounds, err) in enumerate(zip(windspeed_bins, counting_errors_mass)):
    label = f"{bounds[0]}–{bounds[1] if bounds[1] != np.inf else '∞'} m/s"
    if not np.isnan(err):
        print(f"  Bin {idx} ({label}): {err:.4f} µg/m³")
    else:
        print(f"  Bin {idx} ({label}): NaN (no data)")
# %%
counting_errors_mass = np.array(median_mass_errors_per_bin)

# %%
def linear_model(x, m, b):
    return m * x + b

popt_mass, pcov_mass = curve_fit(linear_model, windspeed_values_mass, total_mass_values)
m_fit_mass, b_fit_mass = popt_mass
m_err_mass, b_err_mass = np.sqrt(np.diag(pcov_mass))

residuals = total_mass_values - linear_model(windspeed_values_mass, *popt_mass)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((total_mass_values - np.mean(total_mass_values))**2)
r_squared_mass = 1 - (ss_res / ss_tot)
r_value_mass = np.sign(m_fit_mass) * np.sqrt(r_squared_mass)

x_fit_mass = np.linspace(min(windspeed_values_mass), max(windspeed_values_mass), 100)
y_fit_mass = linear_model(x_fit_mass, *popt_mass)

plt.figure(figsize=(8, 6))
plt.errorbar(windspeed_values_mass, total_mass_values,
             yerr=standard_errors_mass,
             fmt='o', color='#4daf4a',  # green markers
             markersize=10, capsize=5, capthick=2,
             ecolor='black', elinewidth=1.5,
             label="CAS Standard Error", zorder=3)

plt.errorbar(windspeed_values_mass - 0.15,
             total_mass_values,
             yerr=counting_errors_mass,
             fmt='s', color='brown',  
             markersize=6,
             capsize=6, capthick=2, elinewidth=3,
             label="CAS Instrument Counting Error", zorder=2)
plt.plot(x_fit_mass, y_fit_mass, '-', color='black', linewidth=2.5,
         label=f'Fit: y = ({m_fit_mass:.3f}±{m_err_mass:.3f})x + {b_fit_mass:.3f}\n'
               f'R² = {r_squared_mass:.2f}, R = {r_value_mass:.2f}')
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=20, fontweight='bold')
plt.ylabel("Total Dry Mass (µg/m³)", fontsize=20, fontweight='bold')
plt.title("CAS Below Cloud Base\nJanuary–June 2022", fontsize=20, fontweight='bold')
plt.legend(fontsize=14, loc='upper left', frameon=False)
plt.tight_layout()
plt.xlim(0, 12)
plt.ylim(0, 25)
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')
plt.show()
print(f"Slope (m): {m_fit_mass:.3f} ± {m_err_mass:.3f}")
print(f"Intercept (b): {b_fit_mass:.3f}")
print(f"R² value: {r_squared_mass:.2f}")
print(f"R value (Pearson correlation): {r_value_mass:.3f}")
#%%
windspeed_bins = [
    (0, 2.5), (2.501, 3.5), (3.501, 5),
    (5.001, 7), (7.001, 9), (9.001, np.inf)
]

bin_centers_um = np.array([2.25, 2.75, 3.25, 3.75, 4.5, 5.75,
                           6.85, 7.55, 9.05, 11.4, 13.8, 17.5,
                           22.5, 27.5, 32.5, 37.5, 42.5, 47.5])
edges_um = np.empty(len(bin_centers_um) + 1)
edges_um[0] = 2.0 
edges_um[1:-1] = 0.5 * (bin_centers_um[:-1] + bin_centers_um[1:])
edges_um[-1] = bin_centers_um[-1] + 0.5 * (bin_centers_um[-1] - bin_centers_um[-2])
bin_widths_um = np.diff(edges_um)                  # strictly positive (µm)
radii_cm = (bin_centers_um / 2.0) * 1e-4          # µm -> cm
sample_area_cm2 = 0.00323
plane_speed_cm_s = 1.2e4
T = 198                                           # s
V = sample_area_cm2 * plane_speed_cm_s            # cm^3/s
rho_salt_ug_cm3 = 2.2e6                           # µg/cm^3
eta_cm = (4.0/3.0) * np.pi * rho_salt_ug_cm3      # µg/cm^3
eta_m = eta_cm * 1e6                               # convert to µg/m^3 for final output
grouped_mass_errors = {i: [] for i in range(len(windspeed_bins))}
for entry in dry_mass_data_inf: 
    date = entry['Date']
    BCB_start = entry['BCB_start']
    BCB_stop = entry['BCB_stop']
    n0 = entry.get('Dry Intercept (N0)')
    D  = entry.get('Dry Slope (D)')               

    ws_row = df_combined[
        (df_combined['Date'] == date) &
        (df_combined['BCB_start'] == BCB_start) &
        (df_combined['BCB_stop'] == BCB_stop)
    ]
    if ws_row.empty or np.isnan(n0) or np.isnan(D):
        continue
    windspeed = ws_row['Windspeed'].values[0]
    n_i = n0 * np.exp(-bin_centers_um / D)       

    term_cm3 = (n_i * bin_widths_um) * (radii_cm ** 6)  # cm^3
    summation_cm3 = np.nansum(term_cm3)

    if summation_cm3 <= 0 or not np.isfinite(summation_cm3):
        mass_err_ug_m3 = 0.0
    else:
        factor = np.sqrt(summation_cm3 / (T * V))
        # result in µg/m^3
        mass_err_ug_m3 = eta_m * factor

    for idx, (lo, hi) in enumerate(windspeed_bins):
        if lo <= windspeed < hi:
            grouped_mass_errors[idx].append(mass_err_ug_m3)
            break
median_mass_errors_per_bin = [np.nanmedian(errs) if errs else np.nan
                              for errs in grouped_mass_errors.values()]

print("\n Mass Counting Errors per Wind Speed Bin (unfiltered):")
for idx, (bounds, err) in enumerate(zip(windspeed_bins, median_mass_errors_per_bin)):
    label = f"{bounds[0]}–{bounds[1] if bounds[1] != np.inf else '∞'} m/s"
    if not np.isnan(err):
        print(f"  Bin {idx} ({label}): {err:.4f} µg/m³")
    else:
        print(f"  Bin {idx} ({label}): NaN (no data)")
#%%
counting_errors_mass = np.array(median_mass_errors_per_bin)
print("Counting errors used for plotting:", counting_errors_mass)

# %%
def linear_model(x, m, b):
    return m * x + b

popt_mass, pcov_mass = curve_fit(linear_model, windspeed_values_mass, total_mass_values)
m_fit_mass, b_fit_mass = popt_mass
m_err_mass, b_err_mass = np.sqrt(np.diag(pcov_mass))

residuals = total_mass_values - linear_model(windspeed_values_mass, *popt_mass)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((total_mass_values - np.mean(total_mass_values))**2)
r_squared_mass = 1 - (ss_res / ss_tot)
r_value_mass = np.sign(m_fit_mass) * np.sqrt(r_squared_mass)

x_fit_mass = np.linspace(min(windspeed_values_mass), max(windspeed_values_mass), 100)
y_fit_mass = linear_model(x_fit_mass, *popt_mass)
plt.figure(figsize=(8, 6))
offset = 0.4

plt.errorbar(windspeed_values_mass, total_mass_values,
             yerr=standard_errors_mass,
             fmt='o', color='black',
             markersize=10, capsize=5, capthick=2,
             ecolor='black', elinewidth=1.5,
             label="CAS Standard Error", zorder=3)
plt.errorbar(windspeed_values_mass + offset, total_mass_values,
             yerr=counting_errors_mass,
             fmt='o', color='brown',
             markersize=10, capsize=5, capthick=2,
             ecolor='black', elinewidth=1.5,
             label="CAS Error in Total Mass", zorder=2)
plt.plot(x_fit_mass, y_fit_mass, '-', color='black', linewidth=2.5,
         label=f'Fit: y = ({m_fit_mass:.3f}±{m_err_mass:.3f})x + {b_fit_mass:.3f}\n'
               f'R² = {r_squared_mass:.2f}, R = {r_value_mass:.2f}')
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=20, fontweight='bold')
plt.ylabel("Total Dry Mass (µg/m³)", fontsize=20, fontweight='bold')
plt.title("CDP Below Cloud Base\nJanuary–June 2021", fontsize=20, fontweight='bold')
plt.legend(fontsize=12, loc='upper left', frameon=False)
plt.tight_layout()
plt.xlim(0, 12)
plt.ylim(0, 25)
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')
plt.show()
print(f"Slope (m): {m_fit_mass:.3f} ± {m_err_mass:.3f}")
print(f"Intercept (b): {b_fit_mass:.3f}")
print(f"R² value: {r_squared_mass:.2f}")
print(f"R value (Pearson correlation): {r_value_mass:.3f}")

# %%

windspeed_bins = [(0,2.5),(2.501,3.5),(3.501,5),(5.001,7),(7.001,9),(9.001,np.inf)]
bin_centers_um = np.array([2.25, 2.75, 3.25, 3.75, 4.5, 5.75,
                           6.85, 7.55, 9.05, 11.4, 13.8, 17.5,
                           22.5, 27.5, 32.5, 37.5, 42.5, 47.5])
edges_um = np.empty(len(bin_centers_um) + 1)
edges_um[0] = 2.0
edges_um[1:-1] = 0.5*(bin_centers_um[:-1] + bin_centers_um[1:])
edges_um[-1] = bin_centers_um[-1] + 0.5*(bin_centers_um[-1] - bin_centers_um[-2])
bin_widths_um = np.diff(edges_um)
radii_cm = (bin_centers_um/2.0)*1e-4  # µm -> cm
sample_area_cm2 = 0.00323  # CDP sample area in cm²
plane_speed_cm_s = 1.2e4
T = 198
V = sample_area_cm2 * plane_speed_cm_s
rho_salt_ug_cm3 = 2.2e6
eta_cm = (4/3)*np.pi*rho_salt_ug_cm3
eta_m = eta_cm*1e6  # -> µg/m^3
try:
    source_mass_entries = filtered_dry_mass_inf
except NameError:
    source_mass_entries = dry_mass_data_inf
grouped_mass_errors = {i: [] for i in range(len(windspeed_bins))}

for entry in dry_mass_data_inf:  
    n0 = entry.get('Dry Intercept (N0)')
    D  = entry.get('Dry Slope (D)')
    if n0 is None or D is None or np.isnan(n0) or np.isnan(D):
        continue
    ws_row = df_combined[
        (df_combined['Date']==entry['Date']) &
        (df_combined['BCB_start']==entry['BCB_start']) &
        (df_combined['BCB_stop']==entry['BCB_stop'])
    ]
    if ws_row.empty: 
        continue
    ws = float(ws_row['Windspeed'].values[0])
    n_i = n0*np.exp(-bin_centers_um/D)                     # cm^-3 per µm
    term_cm3 = (n_i*bin_widths_um) * (radii_cm**6)         # cm^3
    S = np.nansum(term_cm3)
    if S<=0 or not np.isfinite(S):
        mass_err = 0.0
    else:
        mass_err = eta_m * np.sqrt(S/(T*V))                # µg/m^3

    for k,(lo,hi) in enumerate(windspeed_bins):
        if lo <= ws < hi:
            grouped_mass_errors[k].append(mass_err)
            break

counting_errors_mass = np.array([
    np.nanmedian(vals) if len(vals)>0 else np.nan
    for vals in grouped_mass_errors.values()
])
grouped_mass_vals   = {i: [] for i in range(len(windspeed_bins))}
grouped_windspeeds  = {i: [] for i in range(len(windspeed_bins))}

for entry in source_mass_entries:
    mass_val = entry.get('Dry Mass (µg/m³)')
    if mass_val is None or not np.isfinite(mass_val):
        continue

    ws_row = df_combined[
        (df_combined['Date']==entry['Date']) &
        (df_combined['BCB_start']==entry['BCB_start']) &
        (df_combined['BCB_stop']==entry['BCB_stop'])
    ]
    if ws_row.empty: 
        continue
    ws = float(ws_row['Windspeed'].values[0])

    for k,(lo,hi) in enumerate(windspeed_bins):
        if lo <= ws < hi:
            grouped_mass_vals[k].append(mass_val)
            grouped_windspeeds[k].append(ws)
            break

avg_ws, avg_mass, se_mass = [], [], []
for k in range(len(windspeed_bins)):
    vals = np.array(grouped_mass_vals[k], dtype=float)
    wss  = np.array(grouped_windspeeds[k], dtype=float)
    if vals.size == 0:
        continue
    avg_ws.append(np.nanmean(wss))
    avg_mass.append(np.nanmean(vals))
    if vals.size >= 2:
        se_mass.append(np.nanstd(vals, ddof=1)/np.sqrt(vals.size))
    else:
        se_mass.append(np.nan)

avg_ws  = np.array(avg_ws, dtype=float)
avg_mass= np.array(avg_mass, dtype=float)
se_mass = np.array(se_mass, dtype=float)
print("CDP mass per-bin points:")
for x,y,se,ce in zip(avg_ws, avg_mass, se_mass, counting_errors_mass[np.isfinite(counting_errors_mass)]):
    print(f"  WS={x:.2f} m/s, Mass={y:.2f} µg/m³, SE={se:.2f}, CountErr≈{ce:.2f}")
def linear_model(x,m,b): return m*x + b
mask = np.isfinite(avg_ws) & np.isfinite(avg_mass)
x = avg_ws[mask]; y = avg_mass[mask]; yerr = 2 * se_mass[mask] 

popt, pcov = curve_fit(linear_model, x, y)
m_fit, b_fit = popt
m_err, b_err = np.sqrt(np.diag(pcov))
res = y - linear_model(x, *popt)
r2 = 1 - (np.sum(res**2) / np.sum((y - y.mean())**2))
R  = np.sign(m_fit)*np.sqrt(max(r2,0))
x_fit = np.linspace(x.min(), x.max(), 100)
y_fit = linear_model(x_fit, *popt)
plt.figure(figsize=(8,6))
plt.errorbar(x, y, yerr=yerr, fmt='o', color='black',
             ecolor='black', elinewidth=1.5, capsize=5, capthick=2,
             label="±2 Standard Errors"
, zorder=3)
offset = 0.35
count_err_for_points = 2 * counting_errors_mass[~np.isnan(counting_errors_mass)][:len(x)]  # ±2σ mass error
plt.errorbar(x+offset, y, yerr=count_err_for_points, fmt='o', color='#8c510a',
             ecolor='black', elinewidth=1.5, capsize=5, capthick=2,
             label="±2σ CDP Counting Error"
, zorder=2)

plt.plot(x_fit, y_fit, '-', color='black', linewidth=2.5,
         label=f"Fit: y = ({m_fit:.3f}±{m_err:.3f})x + {b_fit:.3f}\nR² = {r2:.2f}, R = {R:.2f}")

plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=20, fontweight='bold')
plt.ylabel("Total Dry Mass (µg/m³)", fontsize=20, fontweight='bold')
plt.title("CDP Below Cloud Base\nJanuary–June 2021", fontsize=20, fontweight='bold')
plt.legend(fontsize=9, frameon=False, loc='upper left')
plt.xlim(0, 12); plt.ylim(2, 25)
plt.xticks(fontsize=18, fontweight='bold'); plt.yticks(fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()
print("\n=== CDP Mass Fit ===")
print(f"Slope (m): {m_fit:.3f} ± {m_err:.3f}")
print(f"Intercept (b): {b_fit:.3f}")
print(f"R² value: {r2:.2f}, R = {R:.2f}")
# %%
save_dir = "/home/disk/eos4/kathem24/activate/data/2021/CDP/1Hz"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "CDP_2021_wind_bin_mass_plot_data.pkl")
mass_plot_data = {
    "windspeed_bins": windspeed_bins,

    "bin_centers_um": bin_centers_um,
    "edges_um": edges_um,
    "bin_widths_um": bin_widths_um,
    "radii_cm": radii_cm,

    "sample_area_cm2": sample_area_cm2,
    "plane_speed_cm_s": plane_speed_cm_s,
    "T": T,
    "V": V,
    "rho_salt_ug_cm3": rho_salt_ug_cm3,
    "eta_cm": eta_cm,
    "eta_m": eta_m,

    "grouped_mass_errors": grouped_mass_errors,
    "counting_errors_mass": counting_errors_mass,
    "grouped_mass_vals": grouped_mass_vals,
    "grouped_windspeeds": grouped_windspeeds,

    "avg_ws": avg_ws,
    "avg_mass": avg_mass,
    "se_mass": se_mass,

    "x": x,
    "y": y,
    "yerr_2SE": yerr,
    "count_err_for_points_2sigma": count_err_for_points,

    "m_fit": m_fit,
    "b_fit": b_fit,
    "m_err": m_err,
    "b_err": b_err,
    "r2": r2,
    "R": R,
    "x_fit": x_fit,
    "y_fit": y_fit,
}
with open(save_path, "wb") as f:
    pickle.dump(mass_plot_data, f)
print(f"Saved CDP mass plot data to: {save_path}")
# %%
# import pickle

# load_path = "/home/disk/eos4/kathem24/activate/data/2020/CDP/CDP_FMAS2020_wind_bin_mass_plot_data.pkl"

# with open(load_path, "rb") as f:
#     mass_plot_data = pickle.load(f)

# x = mass_plot_data["x"]
# y = mass_plot_data["y"]
# yerr = mass_plot_data["yerr_2SE"]
# count_err_for_points = mass_plot_data["count_err_for_points_2sigma"]

# m_fit = mass_plot_data["m_fit"]
# b_fit = mass_plot_data["b_fit"]
# m_err = mass_plot_data["m_err"]
# b_err = mass_plot_data["b_err"]
# r2 = mass_plot_data["r2"]
# R = mass_plot_data["R"]
# x_fit = mass_plot_data["x_fit"]
# y_fit = mass_plot_data["y_fit"]
# plt.figure(figsize=(8,6))

# plt.errorbar(
#     x, y, yerr=yerr,
#     fmt='o', color='black',
#     ecolor='black', elinewidth=1.5,
#     capsize=5, capthick=2,
#     label="±2 Standard Errors",
#     zorder=3
# )

# offset = 0.35

# plt.errorbar(
#     x + offset, y,
#     yerr=count_err_for_points,
#     fmt='o', color='#8c510a',
#     ecolor='black', elinewidth=1.5,
#     capsize=5, capthick=2,
#     label="±2σ CDP Counting Error",
#     zorder=2
# )

# plt.plot(
#     x_fit, y_fit, '-',
#     color='black', linewidth=2.5,
#     label=f"Fit: y = ({m_fit:.3f}±{m_err:.3f})x + {b_fit:.3f}\nR² = {r2:.2f}, R = {R:.2f}"
# )

# plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=20, fontweight='bold')
# plt.ylabel("Total Dry Mass (µg/m³)", fontsize=20, fontweight='bold')
# plt.title("CDP Below Cloud Base\nFMAS 2020", fontsize=20, fontweight='bold')
# plt.legend(fontsize=11, frameon=False, loc='upper left')
# plt.xlim(0, 12)
# plt.ylim(0, 31)
# plt.xticks(fontsize=18, fontweight='bold')
# plt.yticks(fontsize=18, fontweight='bold')
# plt.tight_layout()
# plt.show()