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
dfp = df_sorted.reset_index(drop=True).copy()
dfp["Date_dt"] = pd.to_datetime(dfp["Date"])
x = np.arange(len(dfp))
tick_pos = dfp.groupby(dfp["Date_dt"].dt.date).head(1).index.to_numpy()
tick_lab = dfp.loc[tick_pos, "Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
#%%
#monthly mean gccn concentration trend coded with color seperation and date x-axis
dfp = df_sorted.copy()
dfp["Date_dt"] = pd.to_datetime(dfp["Date"])
sort_cols = ["Date_dt"]
if "Min_start" in dfp.columns:
    sort_cols.append("Min_start")
dfp = dfp.sort_values(sort_cols).reset_index(drop=True)
x = np.arange(len(dfp))
gccn_arr = np.asarray(gccn_concentration)
date_first = dfp.groupby(dfp["Date_dt"].dt.date).head(1)
tick_pos = date_first.index.to_numpy()
tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
fig_w = max(22, 0.55 * len(tick_pos)) 
fig, ax = plt.subplots(figsize=(fig_w, 6.2))
for m in sorted(dfp["Month"].unique()):
    if m not in month_name:
        continue
    m_mask = (dfp["Month"].values == m)
    mean_concentration = np.nanmean(gccn_arr[m_mask])
    ax.plot(
        x[m_mask], gccn_arr[m_mask],
        '-', linewidth=1.5,
        label=f"{month_name[m]} (mean: {mean_concentration:.2f} cm⁻³)"
    )
for p in tick_pos:
    ax.axvline(p, color="k", alpha=0.06, linewidth=1)
ax.set_yscale("log")
ax.grid(alpha=0.3)
ax.set_ylabel("Total GCCN Concentration (cm⁻³)", fontsize=16, fontweight="bold")
ax.set_xlabel("Leg Date (every flight day)", fontsize=16, fontweight="bold")
ax.set_title("CAS BCB Total GCCN Concentration January–June 2022\nMonthly Means", fontsize=18, fontweight="bold")
ax.legend(ncol=2, fontsize=10, loc="upper right")
ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_lab, rotation=60, ha="right", fontsize=7, fontweight="bold")
labels = ax.get_xticklabels()
for i, lab in enumerate(labels):
    lab.set_text("\n" * (i % 4) + lab.get_text())
ax.set_xticklabels([lab.get_text() for lab in labels])
target_dates = {"2022-06-05": 0, 
                "2022-06-07": 3} 

labels = ax.get_xticklabels()
for lab in labels:
    txt = lab.get_text().replace("\n", "")
    if txt in target_dates:
        lab.set_text("\n" * target_dates[txt] + txt)

ax.set_xticklabels([lab.get_text() for lab in labels])
fig.subplots_adjust(bottom=0.40)
fig.tight_layout()
plt.show()
# %%
#keeping the same plot and code but adding a black thick triangle for the mean of each 
#month and a thick black circle for the median of each month for gccn concentration
dfp = df_sorted.copy()
dfp["Date_dt"] = pd.to_datetime(dfp["Date"])
sort_cols = ["Date_dt"]
if "Min_start" in dfp.columns:
    sort_cols.append("Min_start")
dfp = dfp.sort_values(sort_cols).reset_index(drop=True)
x = np.arange(len(dfp))
gccn_arr = np.asarray(gccn_concentration)
date_first = dfp.groupby(dfp["Date_dt"].dt.date).head(1)
tick_pos = date_first.index.to_numpy()
tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
fig_w = max(22, 0.55 * len(tick_pos))
fig, ax = plt.subplots(figsize=(fig_w, 6.2))
legend_handles = []

for m in sorted(dfp["Month"].unique()):
    if m not in month_name:
        continue

    m_mask = (dfp["Month"].values == m)
    mean_concentration = np.nanmean(gccn_arr[m_mask])
    median_concentration = np.nanmedian(gccn_arr[m_mask])
    line, = ax.plot(
        x[m_mask], gccn_arr[m_mask],
        '-', linewidth=1.5
    )
    c = line.get_color()
    mean_x = x[m_mask][len(x[m_mask]) // 2]
    ax.plot(mean_x, mean_concentration, marker="^", color=c,
            markersize=12, markeredgewidth=1.5, linestyle="None")
    ax.plot(mean_x + 5, median_concentration, marker="o", color=c,
            markersize=12, markeredgewidth=1.5, linestyle="None")
    legend_handles.extend([
        Line2D([0], [0], color=c, lw=2, label=month_name[m]),
        Line2D([0], [0], marker="^", color=c, lw=0, markersize=10,
               label=f"{month_name[m]} mean = {mean_concentration:.2f} cm⁻³"),
        Line2D([0], [0], marker="o", color=c, lw=0, markersize=10,
               label=f"{month_name[m]} median = {median_concentration:.2f} cm⁻³"),
    ])
for p in tick_pos:
    ax.axvline(p, color="k", alpha=0.06, linewidth=1)
ax.set_yscale("log")
plt.yticks(fontsize=16, fontweight="bold")
ax.grid(alpha=0.3)
ax.set_ylabel("Total GCCN Concentration (cm⁻³)", fontsize=20, fontweight="bold")
ax.set_xlabel("Flight Date", fontsize=20, fontweight="bold")
ax.set_title("CAS BCB  January–June 2022\nMonthly Trend",
             fontsize=20, fontweight="bold")
ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_lab, rotation=60, ha="right", fontsize=7, fontweight="bold")
labels = ax.get_xticklabels()
for i, lab in enumerate(labels):
    base = lab.get_text()
    lab.set_text("\n" * (i % 4) + base)
ax.set_xticklabels([lab.get_text() for lab in labels])
target_dates = {"2022-06-05": 0, "2022-06-07": 3}
labels = ax.get_xticklabels()
for lab in labels:
    txt = lab.get_text().replace("\n", "")
    if txt in target_dates:
        lab.set_text("\n" * target_dates[txt] + txt)
ax.set_xticklabels([lab.get_text() for lab in labels])
ax.legend(
    handles=legend_handles,
    ncol=2,
    fontsize=9,
    loc="lower right",
    frameon=True
)
fig.subplots_adjust(bottom=0.40)
fig.tight_layout()
plt.show()
# %%
