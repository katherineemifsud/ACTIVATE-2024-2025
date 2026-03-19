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
from scipy.integrate import quad
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D
from collections import Counter
from scipy.spatial import distance
#%%
mass_path = "/home/disk/p/kathem24/activate/ACTIVATE-2024-2025/CAS/Below Cloud Base/Scripts/Dry_mass_BCB2022_lessthan100massREAL.csv"
df_mass_CAS = pd.read_csv(mass_path)
print(df_mass_CAS.columns)
# %%
#computing mean and median CAS mass 
mass_col = "Dry Mass (µg/m³)"
mean_mass = df_mass_CAS[mass_col].mean()
median_mass = df_mass_CAS[mass_col].median()
print(f"Mean CAS mass: {mean_mass:.3f} µg/m³")
print(f"Median CAS mass: {median_mass:.3f} µg/m³")
# %%
mass_col = "Dry Mass (µg/m³)"
max_mass = df_mass_CAS[mass_col].max()
print(f"Maximum CAS mass: {max_mass:.3f} µg/m³")
# %%
