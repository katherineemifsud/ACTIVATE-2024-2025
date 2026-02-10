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
mass_path = "/home/disk/eos4/kathem24/activate/data/CDP/2022/csv/filtered_dry_mass_inf_CDP.csv"
df_mass_CDP = pd.read_csv(mass_path)
print("Total number of legs:", len(df_mass_CDP))
# %%
#monthly trend of dry mass
df_CDP= df_mass_CDP.copy()
df_CDP= df_CDP[df_CDP["Date"].str.startswith("2022-")].copy()
df_CDP["Month"] = df_CDP["Date"].str[5:7].astype(int)
df_CDP = df_CDP[df_CDP["Month"].between(1, 6)]
df_sorted_cdp = df_CDP.sort_values(["Date", "BCB_start"], kind="mergesort").reset_index(drop=True)
mass_cdp = df_sorted_cdp["Dry Mass (µg/m³)"].astype(float).values
x = np.arange(len(df_sorted_cdp))
plt.figure(figsize=(12, 4.8))
plt.plot(x, mass_cdp, '-')
plt.yscale("log")
plt.grid(alpha=0.3)
plt.xlabel("Leg index (sorted by Date, then BCB_start)", fontsize=13, fontweight="bold")
plt.ylabel("CDP Mass (µg/m³)", fontsize=13, fontweight="bold")
plt.title("Mass Timeline (Jan–Jun 2022)\nLegs ordered by Date then BCB_start",
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
for m in sorted(df_sorted_cdp["Month"].unique()):
    if m not in month_name:
        continue
    m_mask = (df_sorted_cdp["Month"].values == m)
    plt.plot(
        x[m_mask],
        mass_cdp[m_mask],
        '-',
        label=f"{month_name[m]}"
    )
plt.yscale("log")  
plt.grid(alpha=0.3)
plt.ylabel("CDP GCCN Mass (µg/m³)", fontsize=16, fontweight="bold")
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.title("CDP BCB Mass (January–June 2022)",
          fontsize=18, fontweight="bold")
plt.legend(ncol=2, fontsize=10)
plt.yticks(fontsize=14, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
#%%
#monthly mean mass trend coded with color seperation
plt.figure(figsize=(12, 4.8))
for m in sorted(df_sorted_cdp["Month"].unique()): 
    if m not in month_name:
        continue
    m_mask = (df_sorted_cdp["Month"].values == m)
    mean_mass = np.mean(mass_cdp[m_mask])
    plt.plot(
        x[m_mask],
        mass_cdp[m_mask],
        '-',
        label=f"{month_name[m]} (mean: {mean_mass:.2f} µg/m³)"
    )
plt.yscale("log")
plt.grid(alpha=0.3)
plt.ylabel("CDP GCCN Mass (µg/m³)", fontsize=16, fontweight="bold")
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.title("CDP BCB Mass January–June 2022\nMonthly Means", fontsize=18, fontweight="bold")
plt.legend(ncol=2, fontsize=10)
plt.yticks(fontsize=14, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
#%%
#monthly median trend coded with color seperation
plt.figure(figsize=(12, 4.8))
for m in sorted(df_sorted_cdp["Month"].unique()): 
    if m not in month_name:
        continue
    m_mask = (df_sorted_cdp["Month"].values == m)
    median_mass = np.median(mass_cdp[m_mask])
    plt.plot(
        x[m_mask],
        mass_cdp[m_mask],
        '-',
        label=f"{month_name[m]} (median: {median_mass:.2f} µg/m³)"
    )
plt.yscale("log")
plt.grid(alpha=0.3)
plt.ylabel("GCCN Mass (µg/m³)", fontsize=16, fontweight="bold")
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.title("CDP BCB Dry Mass January–June 2022\nMonthly Medians", fontsize=18, fontweight="bold")
plt.legend(ncol=2, fontsize=10)
plt.yticks(fontsize=14, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
#%%
dfp_cdp = df_sorted_cdp.reset_index(drop=True).copy()
dfp_cdp["Date_dt"] = pd.to_datetime(dfp_cdp["Date"])
x = np.arange(len(dfp_cdp))
tick_pos = dfp_cdp.groupby(dfp_cdp["Date_dt"].dt.date).head(1).index.to_numpy()
tick_lab = dfp_cdp.loc[tick_pos, "Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
#%%
#monthly mean mass trend coded with color seperation and date x-axis
dfp_cdp = df_sorted_cdp.copy()
dfp_cdp["Date_dt"] = pd.to_datetime(dfp_cdp["Date"])
sort_cols = ["Date_dt"]
if "Min_start" in dfp_cdp.columns:
    sort_cols.append("Min_start")

dfp_cdp = dfp_cdp.sort_values(sort_cols).reset_index(drop=True)
x = np.arange(len(dfp_cdp))
mass_arr = np.asarray(mass_cdp)
date_first = dfp_cdp.groupby(dfp_cdp["Date_dt"].dt.date).head(1)
tick_pos = date_first.index.to_numpy()
tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
fig_w = max(22, 0.55 * len(tick_pos)) 
fig, ax = plt.subplots(figsize=(fig_w, 6.2))

for m in sorted(dfp_cdp["Month"].unique()):
    if m not in month_name:
        continue
    m_mask = (dfp_cdp["Month"].values == m)
    mean_mass = np.nanmean(mass_arr[m_mask])

    ax.plot(
        x[m_mask], mass_arr[m_mask],
        '-', linewidth=1.5,
        label=f"{month_name[m]} (mean: {mean_mass:.2f} µg/m³)"
    )
for p in tick_pos:
    ax.axvline(p, color="k", alpha=0.06, linewidth=1)

ax.set_yscale("log")
ax.grid(alpha=0.3)
ax.set_ylabel("GCCN Mass (µg/m³)", fontsize=16, fontweight="bold")
ax.set_xlabel("Leg Date (every flight day)", fontsize=16, fontweight="bold")
ax.set_title("CDP BCB Mass January–June 2022\nMonthly Means", fontsize=18, fontweight="bold")
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
#%%
#keeping the same plot and code but adding a black thick triangle for the mean of each 
#month and a thick black circle for the median of each month
dfp_cdp = df_sorted_cdp.copy()
dfp_cdp["Date_dt"] = pd.to_datetime(dfp_cdp["Date"])
sort_cols = ["Date_dt"]
if "Min_start" in dfp_cdp.columns:
    sort_cols.append("Min_start")
dfp_cdp = dfp_cdp.sort_values(sort_cols).reset_index(drop=True)
x = np.arange(len(dfp_cdp))
mass_arr = np.asarray(mass_cdp)
date_first = dfp_cdp.groupby(dfp_cdp["Date_dt"].dt.date).head(1)
tick_pos = date_first.index.to_numpy()
tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
fig_w = max(22, 0.55 * len(tick_pos))
fig, ax = plt.subplots(figsize=(fig_w, 6.2))
legend_handles = []

for m in sorted(dfp_cdp["Month"].unique()):
    if m not in month_name:
        continue

    m_mask = (dfp_cdp["Month"].values == m)
    mean_mass = np.nanmean(mass_arr[m_mask])
    median_mass = np.nanmedian(mass_arr[m_mask])
    line, = ax.plot(
        x[m_mask], mass_arr[m_mask],
        '-', linewidth=1.5
    )
    c = line.get_color()
    mean_x = x[m_mask][len(x[m_mask]) // 2]
    ax.plot(mean_x, mean_mass, marker="^", color=c,
            markersize=12, markeredgewidth=1.5, linestyle="None")
    ax.plot(mean_x + 5, median_mass, marker="o", color=c,
            markersize=12, markeredgewidth=1.5, linestyle="None")
    legend_handles.extend([
        Line2D([0], [0], color=c, lw=2, label=month_name[m]),
        Line2D([0], [0], marker="^", color=c, lw=0, markersize=10,
               label=f"{month_name[m]} mean = {mean_mass:.2f} µg/m³"),
        Line2D([0], [0], marker="o", color=c, lw=0, markersize=10,
               label=f"{month_name[m]} median = {median_mass:.2f} µg/m³"),
    ])
for p in tick_pos:
    ax.axvline(p, color="k", alpha=0.06, linewidth=1)
ax.set_yscale("log")
plt.yticks(fontsize=16, fontweight="bold")
ax.grid(alpha=0.3)
ax.set_ylabel("GCCN Mass (µg/m³)", fontsize=20, fontweight="bold")
ax.set_xlabel("Flight Date", fontsize=20, fontweight="bold")
ax.set_title("CDP BCB Mass January–June 2022\nMonthly Trend",
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
#%%
#doing the EXACT same code as above but not plotting any mass means greater than 100 µg/m³
dfp_cdp = df_sorted_cdp.copy()
dfp_cdp["mass"] = np.asarray(mass_cdp, dtype=float)

dfp_cdp["Date_dt"] = pd.to_datetime(dfp_cdp["Date"])
sort_cols = ["Date_dt"]
if "Min_start" in dfp_cdp.columns:
    sort_cols.append("Min_start")

dfp_cdp = dfp_cdp.sort_values(sort_cols).reset_index(drop=True)

x = np.arange(len(dfp_cdp))

mass_filt_cdp = dfp_cdp["mass"].to_numpy().copy()
fig_w = max(22, 0.55 * len(tick_pos))
fig, ax = plt.subplots(figsize=(fig_w, 6.2))

mass_thr = 100.0
mass_filt_cdp[mass_filt_cdp > mass_thr] = np.nan
mass_filt_cdp[mass_filt_cdp > mass_thr] = np.nan
legend_handles = []
for m in sorted(dfp_cdp["Month"].unique()):
    if m not in month_name:
        continue

    m_mask = (dfp_cdp["Month"].values == m)
    mean_mass = np.nanmean(mass_filt_cdp[m_mask])
    median_mass = np.nanmedian(mass_filt_cdp[m_mask])
    if not np.isfinite(mean_mass):
        continue
    line, = ax.plot(
        x[m_mask], mass_filt_cdp[m_mask],
        '-', linewidth=1.5
    )
    c = line.get_color()
    mean_x = x[m_mask][len(x[m_mask]) // 2]
    ax.plot(mean_x, mean_mass, marker="^", color=c,
            markersize=12, markeredgewidth=1.5, linestyle="None")
    ax.plot(mean_x + 5, median_mass, marker="o", color=c,
            markersize=12, markeredgewidth=1.5, linestyle="None")
    legend_handles.extend([
        Line2D([0], [0], color=c, lw=2, label=month_name[m]),
        Line2D([0], [0], marker="^", color=c, lw=0, markersize=10,
               label=f"{month_name[m]} mean = {mean_mass:.2f} µg/m³"),
        Line2D([0], [0], marker="o", color=c, lw=0, markersize=10,
               label=f"{month_name[m]} median = {median_mass:.2f} µg/m³"),
    ])
for p in tick_pos:
    ax.axvline(p, color="k", alpha=0.06, linewidth=1)
ax.set_yscale("log")
ax.grid(alpha=0.3)
ax.set_ylabel("GCCN Mass (µg/m³)", fontsize=20, fontweight="bold")
ax.set_xlabel("Flight Date", fontsize=20, fontweight="bold")
ax.set_title(f"CAS BCB Mass January–June 2022\nMonthly Trend (legs ≤ {mass_thr:.0f} µg/m³)",
             fontsize=20, fontweight="bold")
plt.yticks(fontsize=16, fontweight="bold")
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
ax.legend(handles=legend_handles, ncol=2, fontsize=9, loc="lower right", frameon=True)
fig.subplots_adjust(bottom=0.40)
fig.tight_layout()
plt.show()

#%%
#monthly trend of Dry Slope (D)
df_cdp = df_mass_CDP.copy()
df_cdp = df_cdp[df_cdp["Date"].str.startswith("2022-")].copy()
df_cdp["Month"] = df_cdp["Date"].str[5:7].astype(int)
df_cdp = df_cdp[df_cdp["Month"].between(1, 6)]
df_sorted_cdp = df_cdp.sort_values(["Date", "BCB_start"], kind="mergesort").reset_index(drop=True)
slope = df_sorted_cdp["Dry Slope (D)"].astype(float).values
x = np.arange(len(df_sorted_cdp))
plt.figure(figsize=(12, 4.8))
plt.plot(x, slope, '-')
plt.grid(alpha=0.3)
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.ylabel("Slope (D)", fontsize=16, fontweight="bold")
plt.title("CDP BCB Slope January–June 2022",
          fontsize=18, fontweight="bold")
plt.tight_layout()
plt.show()
# %%
#monthly Dry Slope (D) trend coded with color seperation
plt.figure(figsize=(12, 4.8))
for m in sorted(df_sorted_cdp["Month"].unique()):
    if m not in month_name:
        continue
    m_mask = (df_sorted_cdp["Month"].values == m)
    plt.plot(
        x[m_mask],
        slope[m_mask],
        '-',
        label=f"{month_name[m]}"
    )
plt.grid(alpha=0.3)
plt.ylabel("Slope (D)", fontsize=16, fontweight="bold")
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.title("CDP BCB Slope (D) January–June 2022",
          fontsize=18, fontweight="bold")
plt.legend(ncol=2, fontsize=10)
plt.yticks(fontsize=14, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
# %%
#monthly mean Dry Slope (D) trend coded with color seperation
plt.figure(figsize=(12, 4.8))
for m in sorted(df_sorted_cdp["Month"].unique()): 
    if m not in month_name:
        continue
    m_mask = (df_sorted_cdp["Month"].values == m)
    mean_slope = np.mean(slope[m_mask])
    plt.plot(
        x[m_mask],
        slope[m_mask],
        '-',
        label=f"{month_name[m]} (mean: {mean_slope:.2f})"
    )
plt.yscale("log")
plt.grid(alpha=0.3)
plt.ylabel("Slope (D)", fontsize=16, fontweight="bold")
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.title("CDP BCB Dry Slope (D) January–June 2022\nMonthly Means", fontsize=18, fontweight="bold")
plt.legend(ncol=2, fontsize=10)
plt.yticks(fontsize=14, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
# %%
#monthly median Dry Slope (D) trend coded with color seperation
plt.figure(figsize=(12, 4.8))
for m in sorted(df_sorted_cdp["Month"].unique()): 
    if m not in month_name:
        continue
    m_mask = (df_sorted_cdp["Month"].values == m)
    median_slope = np.median(slope[m_mask])
    plt.plot(
        x[m_mask],
        slope[m_mask],
        '-',
        label=f"{month_name[m]} (median: {median_slope:.2f})"
    )
plt.yscale("log")
plt.grid(alpha=0.3)
plt.ylabel("Slope (D)", fontsize=16, fontweight="bold")
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.title("CDP BCB Dry Slope (D) January–June 2022\nMonthly Medians", fontsize=18, fontweight="bold")
plt.legend(ncol=2, fontsize=10)
plt.yticks(fontsize=14, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
# %%
dfp_cdp = df_sorted_cdp.reset_index(drop=True).copy()
dfp_cdp["Date_dt"] = pd.to_datetime(dfp_cdp["Date"])
x = np.arange(len(dfp_cdp))
tick_pos = dfp_cdp.groupby(dfp_cdp["Date_dt"].dt.date).head(1).index.to_numpy()
tick_lab = dfp_cdp.loc[tick_pos, "Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
#%%
#monthly mean slope trend coded with color seperation and date x-axis
dfp_cdp = df_sorted_cdp.copy()
dfp_cdp["Date_dt"] = pd.to_datetime(dfp_cdp["Date"])
sort_cols = ["Date_dt"]
if "Min_start" in dfp_cdp.columns:
    sort_cols.append("Min_start")

dfp_cdp = dfp_cdp.sort_values(sort_cols).reset_index(drop=True)
x = np.arange(len(dfp_cdp))
slope_arr = np.asarray(slope)
date_first = dfp_cdp.groupby(dfp_cdp["Date_dt"].dt.date).head(1)
tick_pos = date_first.index.to_numpy()
tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
fig_w = max(22, 0.55 * len(tick_pos)) 
fig, ax = plt.subplots(figsize=(fig_w, 6.2))

for m in sorted(dfp_cdp["Month"].unique()):
    if m not in month_name:
        continue
    m_mask = (dfp_cdp["Month"].values == m)
    mean_slope = np.nanmean(slope_arr[m_mask])

    ax.plot(
        x[m_mask], slope_arr[m_mask],
        '-', linewidth=1.5,
        label=f"{month_name[m]} (mean: {mean_slope:.2f} µg/m³)"
    )
for p in tick_pos:
    ax.axvline(p, color="k", alpha=0.06, linewidth=1)

ax.set_yscale("log")
ax.grid(alpha=0.3)
ax.set_ylabel("Slope (D)", fontsize=16, fontweight="bold")
ax.set_xlabel("Flight Date", fontsize=16, fontweight="bold")
ax.set_title("CDP BCB Slope (D) January–June 2022\nMonthly Trend", fontsize=18, fontweight="bold")
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
#keeping the same plot and code but adding a thick triangle for the mean of each 
#month and a thick circle for the median of each month for slope
dfp_cdp = df_sorted_cdp.copy()
dfp_cdp["Date_dt"] = pd.to_datetime(dfp_cdp["Date"])
sort_cols = ["Date_dt"]
if "Min_start" in dfp_cdp.columns:
    sort_cols.append("Min_start")
dfp_cdp = dfp_cdp.sort_values(sort_cols).reset_index(drop=True)
x = np.arange(len(dfp_cdp))
slope_arr = np.asarray(slope)
date_first = dfp_cdp.groupby(dfp_cdp["Date_dt"].dt.date).head(1)
tick_pos = date_first.index.to_numpy()
tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
fig_w = max(22, 0.55 * len(tick_pos))
fig, ax = plt.subplots(figsize=(fig_w, 6.2))
legend_handles = []

for m in sorted(dfp_cdp["Month"].unique()):
    if m not in month_name:
        continue

    m_mask = (dfp_cdp["Month"].values == m)
    mean_slope = np.nanmean(slope_arr[m_mask])
    median_slope = np.nanmedian(slope_arr[m_mask])
    line, = ax.plot(
        x[m_mask], slope_arr[m_mask],
        '-', linewidth=1.5
    )
    c = line.get_color()
    mean_x = x[m_mask][len(x[m_mask]) // 2]
    ax.plot(mean_x, mean_slope, marker="^", color=c,
            markersize=12, markeredgewidth=1.5, linestyle="None")
    ax.plot(mean_x + 5, median_slope, marker="o", color=c,
            markersize=12, markeredgewidth=1.5, linestyle="None")
    legend_handles.extend([
        Line2D([0], [0], color=c, lw=2, label=month_name[m]),
        Line2D([0], [0], marker="^", color=c, lw=0, markersize=10,
               label=f"{month_name[m]} mean = {mean_slope:.2f} µg/m³"),
        Line2D([0], [0], marker="o", color=c, lw=0, markersize=10,
               label=f"{month_name[m]} median = {median_slope:.2f} µg/m³"),
    ])
for p in tick_pos:
    ax.axvline(p, color="k", alpha=0.06, linewidth=1)
ax.set_yscale("log")
plt.yticks(fontsize=16, fontweight="bold")
ax.grid(alpha=0.3)
ax.set_ylabel("Slope (D)", fontsize=20, fontweight="bold")
ax.set_xlabel("Flight Date", fontsize=20, fontweight="bold")
ax.set_title("CDP BCB Slope January–June 2022\nMonthly Trend",
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
    ncol=3,
    fontsize=9,
    loc="upper right",
    frameon=True
)
fig.subplots_adjust(bottom=0.40)
fig.tight_layout()
plt.show()
# %%
##keeping the same slope plot and code but adding a thick triangle for the mean of each 
#month and a thick circle for the median of each month but removing the same legs we removed for mass
dfp_cdp = df_sorted_cdp.copy()
dfp_cdp["Date_dt"] = pd.to_datetime(dfp_cdp["Date"])
sort_cols = ["Date_dt"]
if "Min_start" in dfp_cdp.columns:
    sort_cols.append("Min_start")
dfp_cdp = dfp_cdp.sort_values(sort_cols).reset_index(drop=True)
x = np.arange(len(dfp_cdp))
slope_arr = np.asarray(slope)
slope_filt_cdp = slope_arr.astype(float).copy()
slope_filt_cdp[mass_arr_cdp > mass_thr] = np.nan
date_first = dfp_cdp.groupby(dfp_cdp["Date_dt"].dt.date).head(1)
tick_pos = date_first.index.to_numpy()
tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
fig_w = max(22, 0.55 * len(tick_pos))
fig, ax = plt.subplots(figsize=(fig_w, 6.2))
legend_handles = []
for m in sorted(dfp_cdp["Month"].unique()):
    if m not in month_name:
        continue

    m_mask = (dfp_cdp["Month"].values == m)
    mean_slope = np.nanmean(slope_filt_cdp[m_mask])
    median_slope = np.nanmedian(slope_filt_cdp[m_mask])
    line, = ax.plot(
        x[m_mask], slope_filt_cdp[m_mask],
        '-', linewidth=1.5
    )
    c = line.get_color()
    mean_x = x[m_mask][len(x[m_mask]) // 2]
    ax.plot(mean_x, mean_slope, marker="^", color=c,
            markersize=12, markeredgewidth=1.5, linestyle="None")
    ax.plot(mean_x + 5, median_slope, marker="o", color=c,
            markersize=12, markeredgewidth=1.5, linestyle="None")
    legend_handles.extend([
        Line2D([0], [0], color=c, lw=2, label=month_name[m]),
        Line2D([0], [0], marker="^", color=c, lw=0, markersize=10,
               label=f"{month_name[m]} mean = {mean_slope:.2f} µg/m³"),
        Line2D([0], [0], marker="o", color=c, lw=0, markersize=10,
               label=f"{month_name[m]} median = {median_slope:.2f} µg/m³"),
    ])
for p in tick_pos:
    ax.axvline(p, color="k", alpha=0.06, linewidth=1)
ax.set_yscale("log")
plt.yticks(fontsize=16, fontweight="bold")
ax.grid(alpha=0.3)
ax.set_ylabel("Slope (D)", fontsize=20, fontweight="bold")
ax.set_xlabel("Flight Date", fontsize=20, fontweight="bold")
ax.set_title(f"CDP BCB Slope January–June 2022\nMonthly Trend (legs ≤ {mass_thr:.0f} µg/m³)",
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
    ncol=3,
    fontsize=9,
    loc="upper right",
    frameon=True
)
fig.subplots_adjust(bottom=0.40)
fig.tight_layout()
plt.show()
#%%
mass_thr = 100.0
dfp_cdp = df_sorted_cdp.copy()
dfp_cdp["Date_dt"] = pd.to_datetime(dfp_cdp["Date"])
dfp_cdp = dfp_cdp.sort_values(["Date_dt", "BCB_start"]).reset_index(drop=True)
slope_arr = pd.to_numeric(dfp_cdp["Dry Slope (D)"], errors="coerce").to_numpy()
mass_arr  = pd.to_numeric(dfp_cdp["Dry Mass (µg/m³)"], errors="coerce").to_numpy()
slope_filt_cdp = slope_arr.copy()
slope_filt_cdp[mass_arr > mass_thr] = np.nan
print("Total legs:", len(dfp_cdp))
print("Mass >100 count:", np.sum(mass_arr > 100))
print("Slope masked:", np.sum(~np.isfinite(slope_filt_cdp)))
#%%
dfp_cdp = df_sorted_cdp.copy()
dfp_cdp["Date_dt"] = pd.to_datetime(dfp_cdp["Date"])
sort_cols = ["Date_dt"]
if "Min_start" in dfp_cdp.columns:
    sort_cols.append("Min_start")
dfp_cdp = dfp_cdp.sort_values(sort_cols).reset_index(drop=True)
x = np.arange(len(dfp_cdp))
mass_thr = 100.0
slope_arr = pd.to_numeric(dfp_cdp["Dry Slope (D)"], errors="coerce").to_numpy(dtype=float)
mass_arr  = pd.to_numeric(dfp_cdp["Dry Mass (µg/m³)"], errors="coerce").to_numpy(dtype=float) 
slope_filt_cdp = slope_arr.copy()
slope_filt_cdp[mass_arr > mass_thr] = np.nan
date_first = dfp_cdp.groupby(dfp_cdp["Date_dt"].dt.date).head(1)
tick_pos = date_first.index.to_numpy()
tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
fig_w = max(22, 0.55 * len(tick_pos))
fig, ax = plt.subplots(figsize=(fig_w, 6.2))
legend_handles = []
for m in sorted(dfp_cdp["Month"].unique()):
    if m not in month_name:
        continue
    m_mask = (dfp_cdp["Month"].values == m)
    mean_slope = np.nanmean(slope_filt_cdp[m_mask])
    median_slope = np.nanmedian(slope_filt_cdp[m_mask])
    line, = ax.plot(x[m_mask], slope_filt_cdp[m_mask], '-', linewidth=1.5)
    c = line.get_color()
    mean_x = x[m_mask][len(x[m_mask]) // 2]
    ax.plot(mean_x, mean_slope, marker="^", color=c,
            markersize=12, markeredgewidth=1.5, linestyle="None")
    ax.plot(mean_x + 5, median_slope, marker="o", color=c,
            markersize=12, markeredgewidth=1.5, linestyle="None")
    legend_handles.extend([
        Line2D([0], [0], color=c, lw=2, label=month_name[m]),
        Line2D([0], [0], marker="^", color=c, lw=0, markersize=10,
               label=f"{month_name[m]} mean = {mean_slope:.2f}"),
        Line2D([0], [0], marker="o", color=c, lw=0, markersize=10,
               label=f"{month_name[m]} median = {median_slope:.2f}"),
    ])
for p in tick_pos:
    ax.axvline(p, color="k", alpha=0.06, linewidth=1)
ax.set_yscale("log")
plt.yticks(fontsize=16, fontweight="bold")
ax.grid(alpha=0.3)
ax.set_ylabel("Slope (D)", fontsize=20, fontweight="bold")
ax.set_xlabel("Flight Date", fontsize=20, fontweight="bold")
ax.set_title(f"CDP BCB Slope January–June 2022\nMonthly Trend (legs ≤ {mass_thr:.0f} µg/m³)",
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
ax.legend(handles=legend_handles, ncol=3, fontsize=9, loc="upper right", frameon=True)
fig.subplots_adjust(bottom=0.40)
fig.tight_layout()
plt.show()

# %%
#combining CDP and CAS mass monthly trend plots into one figure
def prep_trend(df_sorted, mass, mass_thr=100.0, mass_low=None):

    dfp = df_sorted.copy()

    dfp["mass"] = np.asarray(mass, dtype=float)

    dfp["Date_dt"] = pd.to_datetime(dfp["Date"])
    sort_cols = ["Date_dt"]
    if "Min_start" in dfp.columns:
        sort_cols.append("Min_start")

    dfp = dfp.sort_values(sort_cols).reset_index(drop=True)

    x = np.arange(len(dfp))

    mass_filt = dfp["mass"].to_numpy().copy()

    mass_filt[mass_filt > mass_thr] = np.nan
    if mass_low is not None:
        mass_filt[mass_filt < mass_low] = np.nan

    date_first = dfp.groupby(dfp["Date_dt"].dt.date).head(1)
    tick_pos = date_first.index.to_numpy()
    tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()

    return dfp, x, mass_filt, tick_pos, tick_lab


#%%
month_colors = {
    1: "#0072B2",
    2: "#E69F00",
    3: "#009E73",
    5: "#CC79A7",
    6: "#56B4E9",
}
mass_thr = 100.0
mass_low = 1e-3
dfp_cas, _, mass_cas_f, _, _ = prep_trend(df_sorted_cas, mass_cas, mass_thr, mass_low)
dfp_cdp, _, mass_cdp_f, _, _ = prep_trend(df_sorted_cdp, mass_cdp, mass_thr, mass_low)
all_dates = pd.concat([
    dfp_cas["Date_dt"],
    dfp_cdp["Date_dt"]
]).drop_duplicates().sort_values().reset_index(drop=True)

date_to_x = {d: i for i, d in enumerate(all_dates)}
x_cas = dfp_cas["Date_dt"].map(date_to_x).to_numpy()
x_cdp = dfp_cdp["Date_dt"].map(date_to_x).to_numpy()
x_cas = x_cas.astype(float)
x_cdp = x_cdp.astype(float)
for d in np.unique(dfp_cas["Date_dt"]):
    idx = np.where(dfp_cas["Date_dt"] == d)[0]
    if len(idx) > 1:
        x_cas[idx] += np.linspace(-0.3, 0.3, len(idx))

for d in np.unique(dfp_cdp["Date_dt"]):
    idx = np.where(dfp_cdp["Date_dt"] == d)[0]
    if len(idx) > 1:
        x_cdp[idx] += np.linspace(-0.3, 0.3, len(idx))


tick_pos = np.arange(len(all_dates))
tick_lab = all_dates.dt.strftime("%m-%d").to_numpy()
fig_w = max(22, 0.55 * len(tick_pos))
fig, ax = plt.subplots(figsize=(fig_w, 6.2))
legend_month_handles = []
seen_months = set()
for m in sorted(dfp_cas["Month"].unique()):
    if m not in month_name:
        continue
    cas_mask = (dfp_cas["Month"].values == m)
    cdp_mask = (dfp_cdp["Month"].values == m)
    c = month_colors[m]
    ax.plot(x_cas[cas_mask], mass_cas_f[cas_mask], "-", lw=1.5, color=c)
    ax.plot(x_cdp[cdp_mask], mass_cdp_f[cdp_mask], "--", lw=1.5, color=c)
    median_cas = np.nanmedian(mass_cas_f[cas_mask])
    median_cdp = np.nanmedian(mass_cdp_f[cdp_mask])
    median_x = np.nanmedian(x_cas[cas_mask])
    ax.plot(median_x, median_cas, marker="s", color="k", markersize=10, linestyle="None")
    ax.plot(median_x + 0.15, median_cdp, marker="^", color="k", markersize=10, linestyle="None")
    if m not in seen_months:
        legend_month_handles.append(Line2D([0], [0], color=c, lw=2, label=month_name[m]))
        seen_months.add(m)

legend_instrument_handles = [
    Line2D([0], [0], color="k", lw=2, ls="-",  label="CAS (solid)"),
    Line2D([0], [0], color="k", lw=2, ls="--", label="CDP (dashed)"),
    Line2D([0], [0], marker="s", color="k", lw=0, markersize=9, label="CAS monthly median"),
    Line2D([0], [0], marker="^", color="k", lw=0, markersize=9, label="CDP monthly median"),
]

handles = legend_month_handles + legend_instrument_handles
for p in tick_pos:
    ax.axvline(p, color="k", alpha=0.06, linewidth=1)
ax.set_yscale("log")
ax.set_ylim(bottom=10**(-1.3))
plt.yticks(fontsize=13, fontweight="bold")
ax.grid(alpha=0.3)
ax.set_ylabel("GCCN Mass (µg/m³)", fontsize=13, fontweight="bold")
ax.set_xlabel("Flight Date", fontsize=13, fontweight="bold")
ax.set_title("GCCN Mass seasonal trend over the Western Atlantic", fontsize=13, fontweight="bold")
ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_lab, rotation=60, ha="right", fontsize=8, fontweight="bold")
ax.legend(handles=handles, ncol=2, fontsize=9, loc="lower right", frameon=True)
fig.subplots_adjust(bottom=0.40)
fig.tight_layout()
plt.show()
#%%
#adding one mean whisker for each month for CAS and CDP mass trend plot
month_colors = {
    1: "#0072B2",
    2: "#E69F00",
    3: "#009E73",
    5: "#CC79A7",
    6: "#56B4E9",
}
mass_thr = 100.0
mass_low = 1e-3
dfp_cas, _, mass_cas_f, _, _ = prep_trend(df_sorted_cas, mass_cas, mass_thr, mass_low)
dfp_cdp, _, mass_cdp_f, _, _ = prep_trend(df_sorted_cdp, mass_cdp, mass_thr, mass_low)
all_dates = pd.concat([
    dfp_cas["Date_dt"],
    dfp_cdp["Date_dt"]
]).drop_duplicates().sort_values().reset_index(drop=True)

date_to_x = {d: i for i, d in enumerate(all_dates)}
x_cas = dfp_cas["Date_dt"].map(date_to_x).to_numpy()
x_cdp = dfp_cdp["Date_dt"].map(date_to_x).to_numpy()
x_cas = x_cas.astype(float)
x_cdp = x_cdp.astype(float)
for d in np.unique(dfp_cas["Date_dt"]):
    idx = np.where(dfp_cas["Date_dt"] == d)[0]
    if len(idx) > 1:
        x_cas[idx] += np.linspace(-0.3, 0.3, len(idx))

for d in np.unique(dfp_cdp["Date_dt"]):
    idx = np.where(dfp_cdp["Date_dt"] == d)[0]
    if len(idx) > 1:
        x_cdp[idx] += np.linspace(-0.3, 0.3, len(idx))


tick_pos = np.arange(len(all_dates))
tick_lab = all_dates.dt.strftime("%m-%d").to_numpy()
fig_w = max(22, 0.55 * len(tick_pos))
fig, ax = plt.subplots(figsize=(fig_w, 6.2))
legend_month_handles = []
seen_months = set()
for m in sorted(dfp_cas["Month"].unique()):
    if m not in month_name:
        continue
    cas_mask = (dfp_cas["Month"].values == m)
    cdp_mask = (dfp_cdp["Month"].values == m)
    c = month_colors[m]
    ax.plot(x_cas[cas_mask], mass_cas_f[cas_mask], "-", lw=1.5, color=c)
    ax.plot(x_cdp[cdp_mask], mass_cdp_f[cdp_mask], "--", lw=1.5, color=c)
    median_cas = np.nanmedian(mass_cas_f[cas_mask])
    median_cdp = np.nanmedian(mass_cdp_f[cdp_mask])
    mean_cas   = np.nanmean(mass_cas_f[cas_mask])
    mean_cdp   = np.nanmean(mass_cdp_f[cdp_mask])

    median_x = np.nanmedian(x_cas[cas_mask])
    ax.plot(median_x, median_cas, marker="s", color="k", markersize=10, linestyle="None")
    ax.vlines(median_x, ymin=min(median_cas, mean_cas),
            ymax=max(median_cas, mean_cas), color="k", lw=1.5)
    ax.plot(median_x + 0.15, median_cdp, marker="^", color="k", markersize=10, linestyle="None")
    ax.vlines(median_x + 0.15, ymin=min(median_cdp, mean_cdp),
            ymax=max(median_cdp, mean_cdp), color="k", lw=1.5)

    if m not in seen_months:
        legend_month_handles.append(Line2D([0], [0], color=c, lw=2, label=month_name[m]))
        seen_months.add(m)

legend_instrument_handles = [
    Line2D([0], [0], color="k", lw=2, ls="-",  label="CAS (solid)"),
    Line2D([0], [0], color="k", lw=2, ls="--", label="CDP (dashed)"),
    Line2D([0], [0], marker="s", color="k", lw=0, markersize=9, label="CAS monthly median"),
    Line2D([0], [0], marker="^", color="k", lw=0, markersize=9, label="CDP monthly median"),
    Line2D([0], [0], color="k", lw=1.5, label="Monthly mean (whisker)")

]

handles = legend_month_handles + legend_instrument_handles
for p in tick_pos:
    ax.axvline(p, color="k", alpha=0.06, linewidth=1)
ax.set_yscale("log")
ax.set_ylim(bottom=10**(-1.3))
plt.yticks(fontsize=13, fontweight="bold")
ax.grid(alpha=0.3)
ax.set_ylabel("GCCN Mass (µg/m³)", fontsize=13, fontweight="bold")
ax.set_xlabel("Flight Date", fontsize=13, fontweight="bold")
ax.set_title("GCCN Mass seasonal trend over the Western Atlantic", fontsize=13, fontweight="bold")
ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_lab, rotation=60, ha="right", fontsize=8, fontweight="bold")
ax.legend(handles=handles, ncol=2, fontsize=9, loc="lower right", frameon=True)
fig.subplots_adjust(bottom=0.40)
fig.tight_layout()
plt.show()

# %%
# #combinging CDP and CAS slope monthly trend plots into one figure
# mass_thr = 100.0
# dfp_cas = df_sorted_cas.copy()
# dfp_cas["Date_dt"] = pd.to_datetime(dfp_cas["Date"])
# sort_cols = ["Date_dt"]
# if "Min_start" in dfp_cas.columns:
#     sort_cols.append("Min_start")
# dfp_cas = dfp_cas.sort_values(sort_cols).reset_index(drop=True)
# x_cas = np.arange(len(dfp_cas))
# slope_cas = pd.to_numeric(dfp_cas["Dry Slope (D)"], errors="coerce").to_numpy(dtype=float)
# mass_cas  = pd.to_numeric(dfp_cas["Dry Mass (µg/m³)"], errors="coerce").to_numpy(dtype=float)
# slope_filt_cas = slope_cas.copy()
# slope_filt_cas[mass_cas > mass_thr] = np.nan
# date_first = dfp_cas.groupby(dfp_cas["Date_dt"].dt.date).head(1)
# tick_pos = date_first.index.to_numpy()
# tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
# dfp_cdp = df_sorted_cdp.copy()
# dfp_cdp["Date_dt"] = pd.to_datetime(dfp_cdp["Date"])
# sort_cols = ["Date_dt"]
# if "Min_start" in dfp_cdp.columns:
#     sort_cols.append("Min_start")
# dfp_cdp = dfp_cdp.sort_values(sort_cols).reset_index(drop=True)
# x_cdp = np.arange(len(dfp_cdp))
# slope_cdp = pd.to_numeric(dfp_cdp["Dry Slope (D)"], errors="coerce").to_numpy(dtype=float)
# mass_cdp  = pd.to_numeric(dfp_cdp["Dry Mass (µg/m³)"], errors="coerce").to_numpy(dtype=float)
# slope_filt_cdp = slope_cdp.copy()
# slope_filt_cdp[mass_cdp > mass_thr] = np.nan
# fig_w = max(22, 0.55 * len(tick_pos))
# fig, ax = plt.subplots(figsize=(fig_w, 6.2))
# legend_month_handles = []
# seen_months = set()
# months_all = sorted(set(dfp_cas["Month"].unique()).union(set(dfp_cdp["Month"].unique())))
# for m in months_all:
#     if m not in month_name:
#         continue

#     cas_mask = (dfp_cas["Month"].values == m)
#     cdp_mask = (dfp_cdp["Month"].values == m)
#     if "month_colors" in globals() and m in month_colors:
#         c = month_colors[m]
#     else:
#         c = None 
#     if np.any(cas_mask):
#         ax.plot(x_cas[cas_mask], slope_filt_cas[cas_mask], "-", lw=1.5, color=c)

#         mean_cas = np.nanmean(slope_filt_cas[cas_mask])
#         med_cas  = np.nanmedian(slope_filt_cas[cas_mask])
#         mean_x_cas = x_cas[cas_mask][len(x_cas[cas_mask]) // 2]

#         ax.plot(mean_x_cas, mean_cas, marker="^", color=c, markersize=12,
#                 markeredgewidth=1.5, linestyle="None")
#         ax.plot(mean_x_cas + 5, med_cas, marker="o", color=c, markersize=12,
#                 markeredgewidth=1.5, linestyle="None")
#     if np.any(cdp_mask):
#         ax.plot(x_cdp[cdp_mask], slope_filt_cdp[cdp_mask], "--", lw=1.5, color=c)

#         mean_cdp = np.nanmean(slope_filt_cdp[cdp_mask])
#         med_cdp  = np.nanmedian(slope_filt_cdp[cdp_mask])
#         mean_x_cdp = x_cdp[cdp_mask][len(x_cdp[cdp_mask]) // 2]

#         ax.plot(mean_x_cdp, mean_cdp, marker="v", color=c, markersize=12,
#                 markeredgewidth=1.5, linestyle="None")
#         ax.plot(mean_x_cdp + 5, med_cdp, marker="D", color=c, markersize=11,
#                 markeredgewidth=1.5, linestyle="None")
#     if m not in seen_months:
#         legend_month_handles.append(Line2D([0], [0], color=c, lw=2, label=month_name[m]))
#         seen_months.add(m)
# for p in tick_pos:
#     ax.axvline(p, color="k", alpha=0.06, linewidth=1)

# ax.set_yscale("linear")
# plt.yticks(fontsize=16, fontweight="bold")
# ax.grid(alpha=0.3)
# ax.set_ylabel("Slope (D)", fontsize=20, fontweight="bold")
# ax.set_xlabel("Flight Date", fontsize=20, fontweight="bold")
# ax.set_title(f"BCB Slope January–June 2022\nCAS vs CDP (legs ≤ {mass_thr:.0f} µg/m³)",
#              fontsize=20, fontweight="bold")
# ax.set_xticks(tick_pos)
# ax.set_xticklabels(tick_lab, rotation=60, ha="right", fontsize=7, fontweight="bold")
# labels = ax.get_xticklabels()
# for i, lab in enumerate(labels):
#     base = lab.get_text()
#     lab.set_text("\n" * (i % 4) + base)
# ax.set_xticklabels([lab.get_text() for lab in labels])

# target_dates = {"2022-06-05": 0, "2022-06-07": 3}
# labels = ax.get_xticklabels()
# for lab in labels:
#     txt = lab.get_text().replace("\n", "")
#     if txt in target_dates:
#         lab.set_text("\n" * target_dates[txt] + txt)
# ax.set_xticklabels([lab.get_text() for lab in labels])
# legend_instrument_handles = [
#     Line2D([0], [0], color="k", lw=2, ls="-",  label="CAS line (solid)"),
#     Line2D([0], [0], color="k", lw=2, ls="--", label="CDP line (dashed)"),
#     Line2D([0], [0], marker="^", color="k", lw=0, markersize=9, label="CAS mean"),
#     Line2D([0], [0], marker="o", color="k", lw=0, markersize=9, label="CAS median"),
#     Line2D([0], [0], marker="v", color="k", lw=0, markersize=9, label="CDP mean"),
#     Line2D([0], [0], marker="D", color="k", lw=0, markersize=8, label="CDP median"),
# ]
# handles = legend_month_handles + legend_instrument_handles
# ax.legend(handles=handles, ncol=3, fontsize=9, loc="upper right", frameon=True)
# fig.subplots_adjust(bottom=0.40)
# fig.tight_layout()
# plt.show()

# # %%
# # CAS vs CDP MASS monthly trend

# month_colors = {
#     1: "#0072B2",  # Jan - blue
#     2: "#E69F00",  # Feb - orange
#     3: "#009E73",  # Mar - bluish green
#     5: "#CC79A7",  # May - purple/magenta
#     6: "#56B4E9",  # Jun - sky blue
# }

# mass_thr = 100.0
# mass_low = 1e-3
# p_lo, p_hi = 10, 90
# dfp_cas, x_cas, mass_cas_f, tick_pos, tick_lab = prep_trend(df_sorted_cas, mass_cas, mass_thr, mass_low)
# dfp_cdp, x_cdp, mass_cdp_f, _, _ = prep_trend(df_sorted_cdp, mass_cdp, mass_thr, mass_low)

# fig_w = max(22, 0.55 * len(tick_pos))
# fig, ax = plt.subplots(figsize=(fig_w, 6.2))

# legend_month_handles = []
# seen_months = set()

# for m in sorted(dfp_cas["Month"].unique()):
#     if m not in month_name:
#         continue

#     cas_mask = (dfp_cas["Month"].values == m)
#     cdp_mask = (dfp_cdp["Month"].values == m)
#     c = month_colors[m]
#     ax.plot(x_cas[cas_mask], mass_cas_f[cas_mask], "-",  lw=1.5, color=c)
#     ax.plot(x_cdp[cdp_mask], mass_cdp_f[cdp_mask], "--", lw=1.5, color=c)
#     mean_cas = np.nanmean(mass_cas_f[cas_mask])
#     mean_cdp = np.nanmean(mass_cdp_f[cdp_mask])
#     mean_x = x_cas[cas_mask][len(x_cas[cas_mask]) // 2]
#     ax.plot(mean_x,     mean_cas, marker="s", color="k", markersize=10, linestyle="None", zorder=6)
#     ax.plot(mean_x + 2, mean_cdp, marker="^", color="k", markersize=10, linestyle="None", zorder=6)
#     vals_cas = mass_cas_f[cas_mask]
#     if np.sum(np.isfinite(vals_cas)) >= 3 and np.isfinite(mean_cas):
#         qlo_cas, qhi_cas = np.nanpercentile(vals_cas, [p_lo, p_hi])
#         ax.errorbar(
#             mean_x, mean_cas,
#             yerr=[[mean_cas - qlo_cas], [qhi_cas - mean_cas]],
#             fmt="none", ecolor="k", elinewidth=2.5, capsize=6, capthick=2.5, zorder=10
#         )
#     vals_cdp = mass_cdp_f[cdp_mask]
#     if np.sum(np.isfinite(vals_cdp)) >= 3 and np.isfinite(mean_cdp):
#         qlo_cdp, qhi_cdp = np.nanpercentile(vals_cdp, [p_lo, p_hi])
#         ax.errorbar(
#             mean_x + 2, mean_cdp,
#             yerr=[[mean_cdp - qlo_cdp], [qhi_cdp - mean_cdp]],
#             fmt="none", ecolor="k", elinewidth=2.5, capsize=6, capthick=2.5, zorder=10
#         )
#     if m not in seen_months:
#         legend_month_handles.append(Line2D([0], [0], color=c, lw=2, label=month_name[m]))
#         seen_months.add(m)
# for p in tick_pos:
#     ax.axvline(p, color="k", alpha=0.06, linewidth=1)
# ax.set_yscale("log")
# plt.yticks(fontsize=16, fontweight="bold")
# ax.grid(alpha=0.3)
# ax.set_ylabel("GCCN Mass (µg/m³)", fontsize=20, fontweight="bold")
# ax.set_xlabel("Flight Date", fontsize=20, fontweight="bold")
# ax.set_title("BCB Mass January–June 2022\nMonthly Trend", fontsize=20, fontweight="bold")
# ax.set_xticks(tick_pos)
# ax.set_xticklabels(tick_lab, rotation=60, ha="right", fontsize=7, fontweight="bold")
# legend_instrument_handles = [
#     Line2D([0], [0], color="k", lw=2, ls="-",  label="CAS (solid)"),
#     Line2D([0], [0], color="k", lw=2, ls="--", label="CDP (dashed)"),
#     Line2D([0], [0], marker="s", color="k", lw=0, markersize=9, label="CAS monthly mean"),
#     Line2D([0], [0], marker="^", color="k", lw=0, markersize=9, label="CDP monthly mean"),
#     Line2D([0], [0], color="k", lw=2.5, label=f"{p_lo}–{p_hi}th percentile"),
# ]
# handles = legend_month_handles + legend_instrument_handles
# ax.legend(handles=handles, ncol=2, fontsize=9, loc="lower right", frameon=True)
# fig.subplots_adjust(bottom=0.40)
# fig.tight_layout()
# plt.show()
#%%
#trying the new slope style to match our new mass style
month_colors = {
    1: "#0072B2",
    2: "#E69F00",
    3: "#009E73",
    5: "#CC79A7",
    6: "#56B4E9",
}

mass_thr = 100.0
dfp_cas, _, slope_cas_f, _, _ = prep_trend(df_sorted_cas,
                                          pd.to_numeric(df_sorted_cas["Dry Slope (D)"], errors="coerce"),
                                          mass_thr)

dfp_cdp, _, slope_cdp_f, _, _ = prep_trend(df_sorted_cdp,
                                          pd.to_numeric(df_sorted_cdp["Dry Slope (D)"], errors="coerce"),
                                          mass_thr)
all_dates = pd.concat([
    dfp_cas["Date_dt"],
    dfp_cdp["Date_dt"]
]).drop_duplicates().sort_values().reset_index(drop=True)

date_to_x = {d: i for i, d in enumerate(all_dates)}

x_cas = dfp_cas["Date_dt"].map(date_to_x).to_numpy().astype(float)
x_cdp = dfp_cdp["Date_dt"].map(date_to_x).to_numpy().astype(float)
for d in np.unique(dfp_cas["Date_dt"]):
    idx = np.where(dfp_cas["Date_dt"] == d)[0]
    if len(idx) > 1:
        x_cas[idx] += np.linspace(-0.3, 0.3, len(idx))

for d in np.unique(dfp_cdp["Date_dt"]):
    idx = np.where(dfp_cdp["Date_dt"] == d)[0]
    if len(idx) > 1:
        x_cdp[idx] += np.linspace(-0.3, 0.3, len(idx))
tick_pos = np.arange(len(all_dates))
tick_lab = all_dates.dt.strftime("%m-%d").to_numpy()
fig_w = max(22, 0.55 * len(tick_pos))
fig, ax = plt.subplots(figsize=(fig_w, 6.2))
legend_month_handles = []
seen_months = set()
for m in sorted(dfp_cas["Month"].unique()):
    if m not in month_name:
        continue
    cas_mask = dfp_cas["Month"].values == m
    cdp_mask = dfp_cdp["Month"].values == m
    c = month_colors[m]
    ax.plot(x_cas[cas_mask], slope_cas_f[cas_mask], "-", lw=1.5, color=c)
    ax.plot(x_cdp[cdp_mask], slope_cdp_f[cdp_mask], "--", lw=1.5, color=c)
    median_cas = np.nanmedian(slope_cas_f[cas_mask])
    median_cdp = np.nanmedian(slope_cdp_f[cdp_mask])
    mean_cas   = np.nanmean(slope_cas_f[cas_mask])
    mean_cdp   = np.nanmean(slope_cdp_f[cdp_mask])
    median_x = np.nanmedian(x_cas[cas_mask])
    ax.plot(median_x, median_cas, marker="s", color="k", markersize=10, linestyle="None")
    ax.vlines(median_x, min(median_cas, mean_cas), max(median_cas, mean_cas), color="k", lw=1.5)
    ax.plot(median_x + 0.15, median_cdp, marker="^", color="k", markersize=10, linestyle="None")
    ax.vlines(median_x + 0.15, min(median_cdp, mean_cdp), max(median_cdp, mean_cdp), color="k", lw=1.5)

    if m not in seen_months:
        legend_month_handles.append(Line2D([0], [0], color=c, lw=2, label=month_name[m]))
        seen_months.add(m)
legend_instrument_handles = [
    Line2D([0], [0], color="k", lw=2, ls="-",  label="CAS (solid)"),
    Line2D([0], [0], color="k", lw=2, ls="--", label="CDP (dashed)"),
    Line2D([0], [0], marker="s", color="k", lw=0, markersize=9, label="CAS monthly median"),
    Line2D([0], [0], marker="^", color="k", lw=0, markersize=9, label="CDP monthly median"),
    Line2D([0], [0], color="k", lw=1.5, label="Monthly mean (whisker)")
]

handles = legend_month_handles + legend_instrument_handles
for p in tick_pos:
    ax.axvline(p, color="k", alpha=0.06, linewidth=1)

ax.grid(alpha=0.3)
ax.set_ylabel("Slope (D)", fontsize=13, fontweight="bold")
ax.set_xlabel("Flight Date", fontsize=13, fontweight="bold")
ax.set_title("GCCN Slope seasonal trend over the Western Atlantic", fontsize=13, fontweight="bold")

ax.set_xticks(tick_pos)
ax.set_ylim(top=6)
plt.yticks(fontsize=13, fontweight="bold")
ax.set_xticklabels(tick_lab, rotation=60, ha="right", fontsize=8, fontweight="bold")
ax.legend(handles=handles, ncol=2, fontsize=10, loc="upper right", frameon=True)
fig.subplots_adjust(bottom=0.40)
fig.tight_layout()
plt.show()

# %%
# #plotting mass and wind speed together 
# # Make sure these are numeric ints in BOTH dfs (prevents hidden string/float issues)
# wind_cas = dfp_cas_w["Windspeed"].to_numpy(float)
# good = np.isfinite(mass_cas_f) & np.isfinite(wind_cas)

# mass_cas_fw = mass_cas_f.copy()
# mass_cas_fw[~good] = np.nan

# wind_cas_fw = wind_cas.copy()
# wind_cas_fw[~good] = np.nan
# wind_cas = dfp_cas_w["Windspeed"].to_numpy(float)
# wind_cdp = dfp_cdp_w["Windspeed"].to_numpy(float)

# valid_cas = np.isfinite(mass_cas_f) & np.isfinite(wind_cas)
# valid_cdp = np.isfinite(mass_cdp_f) & np.isfinite(wind_cdp)

# mass_cas_fw = mass_cas_f.copy()
# wind_cas_fw = wind_cas.copy()
# mass_cas_fw[~valid_cas] = np.nan
# wind_cas_fw[~valid_cas] = np.nan

# mass_cdp_fw = mass_cdp_f.copy()
# wind_cdp_fw = wind_cdp.copy()
# mass_cdp_fw[~valid_cdp] = np.nan
# wind_cdp_fw[~valid_cdp] = np.nan



# # %%
# mass_thr = 100.0
# mass_low = 1e-3
# dfp_cas, x_cas, mass_cas_f, tick_pos, tick_lab = prep_trend(df_sorted_cas, mass_cas, mass_thr, mass_low)
# dfp_cdp, x_cdp, mass_cdp_f, _, _ = prep_trend(df_sorted_cdp, mass_cdp, mass_thr, mass_low)
# dfp_cas_w = attach_wind_to_mass_df(dfp_cas, df_wind_CDP)
# dfp_cdp_w = attach_wind_to_mass_df(dfp_cdp, df_wind_CDP)
# print("CAS kept legs:", np.sum(valid_cas), "out of", len(valid_cas))
# print("CDP kept legs:", np.sum(valid_cdp), "out of", len(valid_cdp))
# wind_cas = dfp_cas_w["Windspeed"].to_numpy(dtype=float)
# wind_cdp = dfp_cdp_w["Windspeed"].to_numpy(dtype=float)
# good_cas = np.isfinite(mass_cas_f)
# good_cdp = np.isfinite(mass_cdp_f)
# wind_cas_f = wind_cas.copy()
# wind_cas_f[~good_cas] = np.nan
# wind_cdp_f = wind_cdp.copy()
# wind_cdp_f[~good_cdp] = np.nan
# p_lo, p_hi = 10, 90
# def stats_median_and_prc(v, p_lo=10, p_hi=90):
#     v = np.asarray(v, dtype=float)
#     v = v[np.isfinite(v)]
#     if v.size == 0:
#         return np.nan, np.nan, np.nan
#     median = np.nanmedian(v)
#     plo  = np.nanpercentile(v, p_lo)
#     phi  = np.nanpercentile(v, p_hi)
#     return median, plo, phi

# def month_summary(dfp, y, month_col="Month", p_lo=10, p_hi=90):
#     out = {}
#     months = sorted(pd.unique(dfp[month_col]))
#     for m in months:
#         mask = (dfp[month_col].values == m)
#         median, plo, phi = stats_median_and_prc(np.asarray(y)[mask], p_lo, p_hi)
#         out[m] = (median, plo, phi)
#     return out
# cas_mass_sum = month_summary(dfp_cas, mass_cas_fw, "Month", p_lo, p_hi)
# cdp_mass_sum = month_summary(dfp_cdp, mass_cdp_fw, "Month", p_lo, p_hi)

# cas_wind_sum = month_summary(dfp_cas, wind_cas_fw, "Month", p_lo, p_hi)
# cdp_wind_sum = month_summary(dfp_cdp, wind_cdp_fw, "Month", p_lo, p_hi)
# months_all = sorted(set(cas_mass_sum.keys()).union(cdp_mass_sum.keys()))
# fig, ax = plt.subplots(figsize=(7.6, 5.8))
# for m in months_all:
#     if m not in month_name:
#         continue
#     c = month_colors.get(m, "k")
#     w_median_cas, w_p10_cas, w_p90_cas = cas_wind_sum.get(m, (np.nan, np.nan, np.nan))
#     w_median_cdp, w_p10_cdp, w_p90_cdp = cdp_wind_sum.get(m, (np.nan, np.nan, np.nan))

#     m_median_cas, m_p10_cas, m_p90_cas = cas_mass_sum.get(m, (np.nan, np.nan, np.nan))
#     m_median_cdp, m_p10_cdp, m_p90_cdp = cdp_mass_sum.get(m, (np.nan, np.nan, np.nan))
#     if np.isfinite(w_median_cas) and np.isfinite(m_median_cas):
#         ax.errorbar(
#             w_median_cas, m_median_cas,
#             xerr=[[w_median_cas - w_p10_cas], [w_p90_cas - w_median_cas]],
#             yerr=[[m_median_cas - m_p10_cas], [m_p90_cas - m_median_cas]],
#             fmt="s", color=c, ecolor=c,
#             markersize=9, markeredgewidth=1.5,
#             elinewidth=1.8, capsize=4, linestyle="None"
#         )

#     if np.isfinite(w_median_cdp) and np.isfinite(m_median_cdp):
#         ax.errorbar(
#             w_median_cdp, m_median_cdp,
#             xerr=[[w_median_cdp - w_p10_cdp], [w_p90_cdp - w_median_cdp]],
#             yerr=[[m_median_cdp - m_p10_cdp], [m_p90_cdp - m_median_cdp]],
#             fmt="^", color=c, ecolor=c,
#             markersize=10, markeredgewidth=1.5,
#             elinewidth=1.8, capsize=4, linestyle="None"
#         )
# legend_handles = [
#     Line2D([0],[0], marker="s", color="k", lw=0, markersize=9, label="CAS monthly median"),
#     Line2D([0],[0], marker="^", color="k", lw=0, markersize=10, label="CDP monthly median"),
#     Line2D([0],[0], color="none", lw=0, label=f"Error bars: P{p_lo}–P{p_hi} (x and y)")
# ]
# for m in months_all:
#     if m in month_name:
#         legend_handles.append(Line2D([0],[0], marker="o", color=month_colors.get(m,"k"),
#                                      lw=0, markersize=7, label=month_name[m]))
# ax.set_yscale("log")
# ax.grid(alpha=0.3)
# ax.set_xlabel("Monthly median wind speed (m/s)", fontsize=15, fontweight="bold")
# ax.set_ylabel("Monthly median GCCN mass (µg/m³)", fontsize=15, fontweight="bold")
# ax.set_title("BCB January–June 2022\nMonthly GCCN Mass vs Wind Speed", fontsize=15, fontweight="bold")
# ax.legend(
#     handles=legend_handles,
#     ncol=2,
#     fontsize=9,
#     frameon=True,
#     loc="lower right"
# )

# ax.tick_params(axis="both", labelsize=15)
# for lab in ax.get_xticklabels() + ax.get_yticklabels():
#     lab.set_fontweight("bold")

# plt.tight_layout()
# plt.show()
# # %%
# mass_thr = 100.0
# mass_low = 1e-3
# dfp_cas, x_cas, mass_cas_f, tick_pos, tick_lab = prep_trend(df_sorted_cas, mass_cas, mass_thr, mass_low)
# dfp_cdp, x_cdp, mass_cdp_f, _, _ = prep_trend(df_sorted_cdp, mass_cdp, mass_thr, mass_low)
# dfp_cas_w = attach_wind_to_mass_df(dfp_cas, df_wind_CDP)  
# dfp_cdp_w = attach_wind_to_mass_df(dfp_cdp, df_wind_CDP)
# wind_cas = dfp_cas_w["Windspeed"].to_numpy(dtype=float)
# wind_cdp = dfp_cdp_w["Windspeed"].to_numpy(dtype=float)
# valid_cas = np.isfinite(mass_cas_f) & np.isfinite(wind_cas)
# valid_cdp = np.isfinite(mass_cdp_f) & np.isfinite(wind_cdp)
# mass_cas_fw = mass_cas_f.copy()
# wind_cas_fw = wind_cas.copy()
# mass_cas_fw[~valid_cas] = np.nan
# wind_cas_fw[~valid_cas] = np.nan
# mass_cdp_fw = mass_cdp_f.copy()
# wind_cdp_fw = wind_cdp.copy()
# mass_cdp_fw[~valid_cdp] = np.nan
# wind_cdp_fw[~valid_cdp] = np.nan
# print("CAS kept legs:", valid_cas.sum(), "out of", len(valid_cas))
# print("CDP kept legs:", valid_cdp.sum(), "out of", len(valid_cdp))
# assert len(x_cas) == len(mass_cas_fw) == len(wind_cas_fw)
# assert len(x_cdp) == len(mass_cdp_fw) == len(wind_cdp_fw)
# fig_w = max(22, 0.55 * len(tick_pos))
# fig, ax1 = plt.subplots(figsize=(fig_w, 6.2))
# ax2 = ax1.twinx()
# for m in sorted(dfp_cas["Month"].unique()):
#     if m not in month_name:
#         continue
#     mask = (dfp_cas["Month"].values == m)
#     c = month_colors.get(m, "k")
#     ax1.plot(x_cas[mask], mass_cas_fw[mask], "-", lw=1.5, color=c, alpha=0.9)
# for m in sorted(dfp_cdp["Month"].unique()):
#     if m not in month_name:
#         continue
#     mask = (dfp_cdp["Month"].values == m)
#     c = month_colors.get(m, "k")
#     ax1.plot(x_cdp[mask], mass_cdp_fw[mask], "--", lw=1.5, color=c, alpha=0.9)
# ax1.set_yscale("log")
# ax1.set_ylabel("GCCN mass (µg/m³)", fontsize=16, fontweight="bold")
# ax1.grid(alpha=0.3)
# ax2.plot(x_cas, wind_cas_fw, color="k", lw=1.6, alpha=0.75)
# ax2.set_ylabel("Wind speed (m/s)", fontsize=16, fontweight="bold")
# ax1.tick_params(axis="y", labelsize=15)
# for lab in ax1.get_yticklabels():
#     lab.set_fontweight("bold")
# ax2.tick_params(axis="y", labelsize=15)
# for lab in ax2.get_yticklabels():
#     lab.set_fontweight("bold")
# for p in tick_pos:
#     ax1.axvline(p, color="k", alpha=0.06, linewidth=1)
# ax1.set_xticks(tick_pos)
# ax1.set_xticklabels(tick_lab, rotation=60, ha="right", fontsize=7, fontweight="bold")
# legend_handles = []
# for m in sorted(set(dfp_cas["Month"].unique()).union(dfp_cdp["Month"].unique())):
#     if m in month_name:
#         legend_handles.append(
#             Line2D([0], [0], color=month_colors.get(m, "k"), lw=2, label=month_name[m])
#         )

# legend_handles.extend([
#     Line2D([0], [0], color="k", lw=2, linestyle="-",  label="CAS mass"),
#     Line2D([0], [0], color="k", lw=2, linestyle="--", label="CDP mass"),
#     Line2D([0], [0], color="k", lw=2, linestyle="-",  label="Wind (masked to mass+wind-valid legs)"),
# ])

# ax1.legend(handles=legend_handles, ncol=3, fontsize=9, loc="lower right", frameon=True)
# ax1.set_xlabel("Flight Date", fontsize=16, fontweight="bold")
# ax1.set_title("BCB January–June 2022\nGCCN Mass with Wind Speed", fontsize=16, fontweight="bold")
# fig.subplots_adjust(bottom=0.40)
# fig.tight_layout()
# plt.show()


# # %%
# def month_summary_median_simple(dfp, y):
#     out = {}
#     for m in sorted(dfp["Month"].unique()):
#         mask = (dfp["Month"].values == m)
#         vals = np.asarray(y)[mask]
#         vals = vals[np.isfinite(vals)]
#         if len(vals) == 0:
#             out[m] = (np.nan, np.nan, np.nan)
#         else:
#             out[m] = (
#                 np.median(vals),
#                 np.percentile(vals, 10),
#                 np.percentile(vals, 90)
#             )
#     return out
# #plotting slope vs wind speed monthly trend
# wind_cas = dfp_cas_w["Windspeed"].to_numpy(dtype=float)
# wind_cdp = dfp_cdp_w["Windspeed"].to_numpy(dtype=float)
# slope_cas = pd.to_numeric(dfp_cas_w["Dry Slope (D)"], errors="coerce").to_numpy(dtype=float)
# slope_cdp = pd.to_numeric(dfp_cdp_w["Dry Slope (D)"], errors="coerce").to_numpy(dtype=float)
# valid_cas = np.isfinite(mass_cas_f) & np.isfinite(wind_cas) & np.isfinite(slope_cas)
# valid_cdp = np.isfinite(mass_cdp_f) & np.isfinite(wind_cdp) & np.isfinite(slope_cdp)
# slope_cas_fw = slope_cas.copy()
# wind_cas_fw  = wind_cas.copy()
# slope_cas_fw[~valid_cas] = np.nan
# wind_cas_fw[~valid_cas]  = np.nan
# slope_cdp_fw = slope_cdp.copy()
# wind_cdp_fw  = wind_cdp.copy()
# slope_cdp_fw[~valid_cdp] = np.nan
# wind_cdp_fw[~valid_cdp]  = np.nan
# print("CAS kept legs:", np.sum(valid_cas), "out of", len(valid_cas))
# print("CDP kept legs:", np.sum(valid_cdp), "out of", len(valid_cdp))
# cas_slope_sum = month_summary_median_simple(dfp_cas_w, slope_cas_fw)
# cdp_slope_sum = month_summary_median_simple(dfp_cdp_w, slope_cdp_fw)
# cas_wind_sum  = month_summary_median_simple(dfp_cas_w, wind_cas_fw)
# cdp_wind_sum  = month_summary_median_simple(dfp_cdp_w, wind_cdp_fw)
# months_all = sorted(set(cas_slope_sum.keys()).union(cdp_slope_sum.keys()))
# fig, ax = plt.subplots(figsize=(7.6, 5.8))
# for m in months_all:
#     if m not in month_name:
#         continue

#     c = month_colors.get(m, "k")
#     w_median_cas, w_p10_cas, w_p90_cas = cas_wind_sum.get(m, (np.nan, np.nan, np.nan))
#     w_median_cdp, w_p10_cdp, w_p90_cdp = cdp_wind_sum.get(m, (np.nan, np.nan, np.nan))
#     s_median_cas, s_p10_cas, s_p90_cas = cas_slope_sum.get(m, (np.nan, np.nan, np.nan))
#     s_median_cdp, s_p10_cdp, s_p90_cdp = cdp_slope_sum.get(m, (np.nan, np.nan, np.nan))
#     if np.isfinite(w_median_cas) and np.isfinite(s_median_cas):
#         ax.errorbar(
#             w_median_cas, s_median_cas,
#             xerr=[[w_median_cas - w_p10_cas], [w_p90_cas - w_median_cas]],
#             yerr=[[s_median_cas - s_p10_cas], [s_p90_cas - s_median_cas]],
#             fmt="s", color=c, ecolor=c,
#             markersize=9, markeredgewidth=1.5,
#             elinewidth=1.8, capsize=4, linestyle="None"
#         )
#     if np.isfinite(w_median_cdp) and np.isfinite(s_median_cdp):
#         ax.errorbar(
#             w_median_cdp, s_median_cdp,
#             xerr=[[w_median_cdp - w_p10_cdp], [w_p90_cdp - w_median_cdp]],
#             yerr=[[s_median_cdp - s_p10_cdp], [s_p90_cdp - s_median_cdp]],
#             fmt="^", color=c, ecolor=c,
#             markersize=10, markeredgewidth=1.5,
#             elinewidth=1.8, capsize=4, linestyle="None"
#         )

# legend_handles = [
#     Line2D([0],[0], marker="s", color="k", lw=0, markersize=9, label="CAS monthly median"),
#     Line2D([0],[0], marker="^", color="k", lw=0, markersize=10, label="CDP monthly median"),
#     Line2D([0],[0], color="none", lw=0, label=f"Error bars: P{p_lo}–P{p_hi} (x and y)")
# ]
# for m in months_all:
#     if m in month_name:
#         legend_handles.append(Line2D([0],[0], marker="o", color=month_colors.get(m,"k"),
#                                      lw=0, markersize=7, label=month_name[m]))
# ax.grid(alpha=0.3)
# ax.set_xlabel("Monthly median wind speed (m/s)", fontsize=15, fontweight="bold")
# ax.set_ylabel("Monthly median slope (D)", fontsize=15, fontweight="bold")
# ax.set_title(f"BCB January–June 2022\nMonthly Slope vs Wind Speed",
#              fontsize=15, fontweight="bold")
# ax.legend(handles=legend_handles, ncol=2, fontsize=9, frameon=True, loc="best")
# ax.tick_params(axis="both", labelsize=15)
# for lab in ax.get_xticklabels() + ax.get_yticklabels():
#     lab.set_fontweight("bold")
# plt.tight_layout()
# plt.show()

# %%
# mass_thr = 100.0
# mass_low = 1e-3
# dfp_cas, x_cas, mass_cas_f, tick_pos, tick_lab = prep_trend(df_sorted_cas, mass_cas, mass_thr, mass_low)
# dfp_cdp, x_cdp, mass_cdp_f, _, _               = prep_trend(df_sorted_cdp, mass_cdp, mass_thr, mass_low)
# dfp_cas_w = attach_wind_to_mass_df(dfp_cas, df_wind_CDP)
# dfp_cdp_w = attach_wind_to_mass_df(dfp_cdp, df_wind_CDP)
# wind_cas = dfp_cas_w["Windspeed"].to_numpy(dtype=float)
# wind_cdp = dfp_cdp_w["Windspeed"].to_numpy(dtype=float)
# slope_cas = pd.to_numeric(dfp_cas_w["Dry Slope (D)"], errors="coerce").to_numpy(dtype=float)
# slope_cdp = pd.to_numeric(dfp_cdp_w["Dry Slope (D)"], errors="coerce").to_numpy(dtype=float)
# valid_cas = np.isfinite(mass_cas_f) & np.isfinite(wind_cas) & np.isfinite(slope_cas)
# valid_cdp = np.isfinite(mass_cdp_f) & np.isfinite(wind_cdp) & np.isfinite(slope_cdp)
# slope_cas_fw = slope_cas.copy()
# wind_cas_fw  = wind_cas.copy()
# slope_cas_fw[~valid_cas] = np.nan
# wind_cas_fw[~valid_cas]  = np.nan
# slope_cdp_fw = slope_cdp.copy()
# wind_cdp_fw  = wind_cdp.copy()
# slope_cdp_fw[~valid_cdp] = np.nan
# wind_cdp_fw[~valid_cdp]  = np.nan
# print("CAS kept legs:", valid_cas.sum(), "out of", len(valid_cas))
# print("CDP kept legs:", valid_cdp.sum(), "out of", len(valid_cdp))
# assert len(x_cas) == len(slope_cas_fw) == len(wind_cas_fw)
# assert len(x_cdp) == len(slope_cdp_fw) == len(wind_cdp_fw)
# fig_w = max(22, 0.55 * len(tick_pos))
# fig, ax1 = plt.subplots(figsize=(fig_w, 6.2))
# ax2 = ax1.twinx()
# for m in sorted(dfp_cas_w["Month"].unique()):
#     if m not in month_name:
#         continue
#     mask = (dfp_cas_w["Month"].values == m)
#     c = month_colors.get(m, "k")
#     ax1.plot(x_cas[mask], slope_cas_fw[mask], "-", lw=1.5, color=c, alpha=0.9)
# for m in sorted(dfp_cdp_w["Month"].unique()):
#     if m not in month_name:
#         continue
#     mask = (dfp_cdp_w["Month"].values == m)
#     c = month_colors.get(m, "k")
#     ax1.plot(x_cdp[mask], slope_cdp_fw[mask], "--", lw=1.5, color=c, alpha=0.9)
# ax1.set_ylabel("Slope (D)", fontsize=16, fontweight="bold")
# ax1.grid(alpha=0.3)
# ax2.plot(x_cas, wind_cas_fw, color="k", lw=1.6, alpha=0.75)
# ax2.set_ylabel("Wind speed (m/s)", fontsize=16, fontweight="bold")
# ax1.tick_params(axis="y", labelsize=15)
# for lab in ax1.get_yticklabels():
#     lab.set_fontweight("bold")
# ax2.tick_params(axis="y", labelsize=15)
# for lab in ax2.get_yticklabels():
#     lab.set_fontweight("bold")
# for p in tick_pos:
#     ax1.axvline(p, color="k", alpha=0.06, linewidth=1)
# ax1.set_xticks(tick_pos)
# ax1.set_xticklabels(tick_lab, rotation=60, ha="right", fontsize=7, fontweight="bold")
# legend_handles = []
# for m in sorted(set(dfp_cas_w["Month"].unique()).union(dfp_cdp_w["Month"].unique())):
#     if m in month_name:
#         legend_handles.append(Line2D([0], [0], color=month_colors.get(m, "k"), lw=2, label=month_name[m]))
# legend_handles.extend([
#     Line2D([0], [0], color="k", lw=2, linestyle="-",  label="CAS slope"),
#     Line2D([0], [0], color="k", lw=2, linestyle="--", label="CDP slope"),
#     Line2D([0], [0], color="k", lw=2, linestyle="-",  label="Wind (masked to mass+wind+slope-valid legs)"),
# ])
# ax1.legend(
#     handles=legend_handles,
#     ncol=3,
#     fontsize=9,
#     frameon=True,
#     loc="upper center",
#     bbox_to_anchor=(0.5, -0.25)
# )

# fig.subplots_adjust(bottom=0.45)

# %%
