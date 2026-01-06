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
gccn_path = "/home/disk/eos4/kathem24/activate/data/CAS/total_Y_concentration_cm3.csv"
df_gccn = pd.read_csv(gccn_path)
print("Total number of legs:", len(df_gccn))
# %%
#monthly trend of gccn concentration
df = df_gccn.copy()
df = df[df["Date"].str.startswith("2022-")].copy()
df["Month"] = df["Date"].str[5:7].astype(int)
df = df[df["Month"].between(1, 6)]
df_sorted = df.sort_values(["Date", "BCB_start"], kind="mergesort").reset_index(drop=True)
gccn_concentration = df_sorted['Total_Y_Concentration_cm3'].astype(float).values
x = np.arange(len(df_sorted))
plt.figure(figsize=(12, 4.8))
plt.plot(x, gccn_concentration, '-')
plt.yscale("log")
plt.grid(alpha=0.3)
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.ylabel("Total GCCN Concentration (cm⁻³)", fontsize=16, fontweight="bold")
plt.title("CAS BCB Total GCCN Concentration\nJanuary–June 2022", fontsize=18, fontweight="bold")
plt.tight_layout()
plt.show()

# %%
#monthly gccn trend coded with color seperation
month_name = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
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
        gccn_concentration[m_mask],
        '-',
        label=f"{month_name[m]}"
    )
plt.yscale("log")  
plt.grid(alpha=0.3)
plt.ylabel("Total GCCN Concentration (cm⁻³)", fontsize=16, fontweight="bold")
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.title("CAS BCB Total GCCN Concentration (January–June 2022)",
          fontsize=18, fontweight="bold")
plt.legend(ncol=2, fontsize=10)
plt.yticks(fontsize=14, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
# %%
#monthly mean gccn concentration trend coded with color seperation
plt.figure(figsize=(12, 4.8))
for m in sorted(df_sorted["Month"].unique()): 
    if m not in month_name:
        continue
    m_mask = (df_sorted["Month"].values == m)
    mean_concentration = np.mean(gccn_concentration[m_mask])
    plt.plot(
        x[m_mask],
        gccn_concentration[m_mask],
        '-',
        label=f"{month_name[m]} (mean: {mean_concentration:.2f} cm⁻³)"
    )
plt.yscale("log")
plt.grid(alpha=0.3)
plt.ylabel("Total GCCN Concentration (cm⁻³)", fontsize=16, fontweight="bold")
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.title("CAS BCB Total GCCN Concentration January–June 2022\nMonthly Means", fontsize=18, fontweight="bold")
plt.legend(ncol=2, fontsize=10)
plt.yticks(fontsize=14, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
# %%
