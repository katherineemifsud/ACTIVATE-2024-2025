#%%
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pathlib
import statistics
import math
from matplotlib.colors import BoundaryNorm
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
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy.ma as ma
import matplotlib.patheffects as path_effects
from scipy.interpolate import interp1d
from scipy.stats import linregress
import matplotlib.colors as mcolors
import pickle
import re
from collections import defaultdict
import glob
import os
import sys
#%%
mass_path = "/home/disk/eos4/kathem24/activate/data/CAS/filtered_dry_mass_inf.csv"
df_mass = pd.read_csv(mass_path)
print("Total number of legs:", len(df_mass))
# %%
#monthly trend of dry mass
df = df_mass.copy()
df = df[df["Date"].str.startswith("2022-")].copy()
df["Month"] = df["Date"].str[5:7].astype(int)
df = df[df["Month"].between(1, 6)]
df_sorted = df.sort_values(["Date", "BCB_start"], kind="mergesort").reset_index(drop=True)
mass = df_sorted["Dry Mass (µg/m³)"].astype(float).values
x = np.arange(len(df_sorted))
plt.figure(figsize=(12, 4.8))
plt.plot(x, mass, '-')
plt.yscale("log")
plt.grid(alpha=0.3)
plt.xlabel("Leg index (sorted by Date, then BCB_start)", fontsize=13, fontweight="bold")
plt.ylabel("Dry GCCN Mass (µg/m³)", fontsize=13, fontweight="bold")
plt.title("Dry Mass Timeline (Jan–Jun 2022)\nLegs ordered by Date then BCB_start",
          fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
#%%
#monthly mass trend coded with color seperation
month_name = {
    1: "January",
    2: "February",
    3: "March",
    5: "May",
    6: "June"
}
plt.figure(figsize=(12, 4.8))
for m in sorted(df_sorted["Month"].unique()):
    if m not in month_name:
        continue
    m_mask = (df_sorted["Month"].values == m)
    plt.plot(
        x[m_mask],
        mass[m_mask],
        '-',
        label=f"{month_name[m]}"
    )
plt.yscale("log")  
plt.grid(alpha=0.3)
plt.ylabel("Dry GCCN Mass (µg/m³)", fontsize=16, fontweight="bold")
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.title("CAS BCB Dry Mass (January–June 2022)",
          fontsize=18, fontweight="bold")
plt.legend(ncol=2, fontsize=10)
plt.yticks(fontsize=14, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
#%%
#monthly mean mass trend coded with color seperation
plt.figure(figsize=(12, 4.8))
for m in sorted(df_sorted["Month"].unique()): 
    if m not in month_name:
        continue
    m_mask = (df_sorted["Month"].values == m)
    mean_mass = np.mean(mass[m_mask])
    plt.plot(
        x[m_mask],
        mass[m_mask],
        '-',
        label=f"{month_name[m]} (mean: {mean_mass:.2f} µg/m³)"
    )
plt.yscale("log")
plt.grid(alpha=0.3)
plt.ylabel("Dry GCCN Mass (µg/m³)", fontsize=16, fontweight="bold")
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.title("CAS BCB Dry Mass January–June 2022\nMonthly Means", fontsize=18, fontweight="bold")
plt.legend(ncol=2, fontsize=10)
plt.yticks(fontsize=14, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
#%%
#monthly median trend coded with color seperation
plt.figure(figsize=(12, 4.8))
for m in sorted(df_sorted["Month"].unique()): 
    if m not in month_name:
        continue
    m_mask = (df_sorted["Month"].values == m)
    median_mass = np.median(mass[m_mask])
    plt.plot(
        x[m_mask],
        mass[m_mask],
        '-',
        label=f"{month_name[m]} (median: {median_mass:.2f} µg/m³)"
    )
plt.yscale("log")
plt.grid(alpha=0.3)
plt.ylabel("Dry GCCN Mass (µg/m³)", fontsize=16, fontweight="bold")
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.title("CAS BCB Dry Mass January–June 2022\nMonthly Medians", fontsize=18, fontweight="bold")
plt.legend(ncol=2, fontsize=10)
plt.yticks(fontsize=14, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
#%%
#monthly trend of Dry Slope (D)
df = df_mass.copy()
df = df[df["Date"].str.startswith("2022-")].copy()
df["Month"] = df["Date"].str[5:7].astype(int)
df = df[df["Month"].between(1, 6)]
df_sorted = df.sort_values(["Date", "BCB_start"], kind="mergesort").reset_index(drop=True)
slope = df_sorted["Dry Slope (D)"].astype(float).values
x = np.arange(len(df_sorted))
plt.figure(figsize=(12, 4.8))
plt.plot(x, slope, '-')
plt.grid(alpha=0.3)
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.ylabel("Slope (D)", fontsize=16, fontweight="bold")
plt.title("CAS BCB Slope January–June 2022",
          fontsize=18, fontweight="bold")
plt.tight_layout()
plt.show()
# %%
#monthly Dry Slope (D) trend coded with color seperation
plt.figure(figsize=(12, 4.8))
for m in sorted(df_sorted["Month"].unique()):
    if m not in month_name:
        continue
    m_mask = (df_sorted["Month"].values == m)
    plt.plot(
        x[m_mask],
        slope[m_mask],
        '-',
        label=f"{month_name[m]}"
    )
plt.grid(alpha=0.3)
plt.ylabel("Slope (D)", fontsize=16, fontweight="bold")
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.title("CAS BCB Slope (D) January–June 2022",
          fontsize=18, fontweight="bold")
plt.legend(ncol=2, fontsize=10)
plt.yticks(fontsize=14, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
# %%
#monthly mean Dry Slope (D) trend coded with color seperation
plt.figure(figsize=(12, 4.8))
for m in sorted(df_sorted["Month"].unique()): 
    if m not in month_name:
        continue
    m_mask = (df_sorted["Month"].values == m)
    mean_slope = np.mean(slope[m_mask])
    plt.plot(
        x[m_mask],
        slope[m_mask],
        '-',
        label=f"{month_name[m]} (mean: {mean_slope:.2f})"
    )
plt.yscale("log")
plt.grid(alpha=0.3)
plt.ylabel("Dry GCCN Mass (µg/m³)", fontsize=16, fontweight="bold")
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.title("CAS BCB Dry Slope (D) January–June 2022\nMonthly Means", fontsize=18, fontweight="bold")
plt.legend(ncol=2, fontsize=10)
plt.yticks(fontsize=14, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
# %%
#monthly median Dry Slope (D) trend coded with color seperation
plt.figure(figsize=(12, 4.8))
for m in sorted(df_sorted["Month"].unique()): 
    if m not in month_name:
        continue
    m_mask = (df_sorted["Month"].values == m)
    median_slope = np.median(slope[m_mask])
    plt.plot(
        x[m_mask],
        slope[m_mask],
        '-',
        label=f"{month_name[m]} (median: {median_slope:.2f})"
    )
plt.yscale("log")
plt.grid(alpha=0.3)
plt.ylabel("Dry GCCN Mass (µg/m³)", fontsize=16, fontweight="bold")
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.title("CAS BCB Dry Slope (D) January–June 2022\nMonthly Medians", fontsize=18, fontweight="bold")
plt.legend(ncol=2, fontsize=10)
plt.yticks(fontsize=14, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
# %%
