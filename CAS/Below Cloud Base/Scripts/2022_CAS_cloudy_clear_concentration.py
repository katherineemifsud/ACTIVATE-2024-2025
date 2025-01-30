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


bin_log=[C_0, C_01, C_02, C_03, C_04, C_05, C_06, C_07, C_08, C_09, C_10, C_11, C_12, C_13, C_14, C_15, C_16, 
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
bin_name = ['CAS_Bin00', 'CAS_Bin01', 'CAS_Bin02', 'CAS_Bin03', 'CAS_Bin04', 'CAS_Bin05', 'CAS_Bin06', 'CAS_Bin07', 'CAS_Bin08',
            'CAS_Bin09', 'CAS_Bin10', 'CAS_Bin11', 'CAS_Bin12', 'CAS_Bin13', 'CAS_Bin14', 'CAS_Bin15', 'CAS_Bin16', 'CAS_Bin17', 
            'CAS_Bin18', 'CAS_Bin19', 'CAS_Bin20', 'CAS_Bin21', 'CAS_Bin22', 'CAS_Bin23', 'CAS_Bin24', 'CAS_Bin25', 'CAS_Bin26',
            'CAS_Bin27', 'CAS_Bin28', 'CAS_Bin29']

datasets = []

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
clear_means = [] 
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
for date in dates_CAS:
    dataset = {'Date': date, 'Clear Means': [], 'Cloud Means': []}  # Initialize a dataset dictionary
    datestr = date.replace('-', '')
    fname_CAS = sorted(glob.glob(f'/home/disk/eos4/kathem24/activate/data/cloudaerospect/2022csv/ACTIVATE-LARGE-CAS_HU25_{datestr}_R*.csv'), reverse=True)
    
    print(f"Date: {date}")
    print(f"Files: {list(set(fname_CAS))}")
    
    file_path = fname_CAS[0]
    print(file_path)
# %%
