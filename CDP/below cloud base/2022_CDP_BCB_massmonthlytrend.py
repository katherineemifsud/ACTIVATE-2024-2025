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
#plotting mass and wind speed together 
#3 panel paper figure with gccn concentration, slope, and mass for both CAS and CDP with monthly means and medians with singular legend
month_colors = {
    1: "#0072B2",
    2: "#E69F00",
    3: "#009E73",
    5: "#CC79A7",
    6: "#56B4E9",
}
mass_thr = 100.0
mass_low = 1e-3

# MASS
dfp_m_cas, _, mass_cas_f, _, _ = prep_trend(df_sorted_cas, mass_cas, mass_thr, mass_low)
dfp_m_cdp, _, mass_cdp_f, _, _ = prep_trend(df_sorted_cdp, mass_cdp, mass_thr, mass_low)

# SLOPE
dfp_s_cas, _, slope_cas_f, _, _ = prep_trend(
    df_sorted_cas,
    pd.to_numeric(df_sorted_cas["Dry Slope (D)"], errors="coerce"),
    mass_thr
)
dfp_s_cdp, _, slope_cdp_f, _, _ = prep_trend(
    df_sorted_cdp,
    pd.to_numeric(df_sorted_cdp["Dry Slope (D)"], errors="coerce"),
    mass_thr
)

# GCCN CONCENTRATION
dfp_g_cas, _, gccn_cas_f, _, _ = prep_trend(df_sorted,     gccn_concentration,     mass_thr=np.inf)
dfp_g_cdp, _, gccn_cdp_f, _, _ = prep_trend(df_sorted_CDP, gccn_concentration_CDP, mass_thr=np.inf)
all_dates = pd.concat([
    dfp_m_cas["Date_dt"], dfp_m_cdp["Date_dt"],
    dfp_s_cas["Date_dt"], dfp_s_cdp["Date_dt"],
    dfp_g_cas["Date_dt"], dfp_g_cdp["Date_dt"],
]).drop_duplicates().sort_values().reset_index(drop=True)

date_to_x = {d: i for i, d in enumerate(all_dates)}
tick_pos = np.arange(len(all_dates))
tick_lab = all_dates.dt.strftime("%m-%d").to_numpy()

def make_x(dfp, jitter=0.3):
    x = dfp["Date_dt"].map(date_to_x).to_numpy().astype(float)
    for d in np.unique(dfp["Date_dt"]):
        idx = np.where(dfp["Date_dt"] == d)[0]
        if len(idx) > 1:
            x[idx] += np.linspace(-jitter, jitter, len(idx))
    return x
xm_cas = make_x(dfp_m_cas); xm_cdp = make_x(dfp_m_cdp)
xs_cas = make_x(dfp_s_cas); xs_cdp = make_x(dfp_s_cdp)
xg_cas = make_x(dfp_g_cas); xg_cdp = make_x(dfp_g_cdp)
fig_w = max(22, 0.55 * len(tick_pos))
fig, ax_mass = plt.subplots(
    1, 1, figsize=(fig_w, 12), sharex=True
)

def plot_monthly(ax, dfp_cas, dfp_cdp, x_cas, x_cdp, y_cas, y_cdp, yscale=None, ylim=None,
                 title="", ylabel="", show_xticks=False):
    legend_month_handles = []
    seen_months = set()

    for m in sorted(dfp_cas["Month"].unique()):
        if m not in month_name:
            continue

        cas_mask = dfp_cas["Month"].values == m
        cdp_mask = dfp_cdp["Month"].values == m
        c = month_colors[m]

        ax.plot(x_cas[cas_mask], y_cas[cas_mask], "-",  lw=1.5, color=c)
        ax.plot(x_cdp[cdp_mask], y_cdp[cdp_mask], "--", lw=1.5, color=c)

        median_cas = np.nanmedian(y_cas[cas_mask])
        mean_cas   = np.nanmean(y_cas[cas_mask])
        median_cdp = np.nanmedian(y_cdp[cdp_mask])
        mean_cdp   = np.nanmean(y_cdp[cdp_mask])

        median_x = np.nanmedian(x_cas[cas_mask])

        ax.plot(median_x, median_cas, marker="s", color="k", markersize=14, linestyle="None")
        ax.vlines(median_x, min(median_cas, mean_cas), max(median_cas, mean_cas), color="k", lw=3.5)

        ax.plot(median_x + 0.15, median_cdp, marker="^", color="k", markersize=14, linestyle="None")
        ax.vlines(median_x + 0.15, min(median_cdp, mean_cdp), max(median_cdp, mean_cdp), color="k", lw=3.5)

        if m not in seen_months:
            legend_month_handles.append(Line2D([0], [0], color=c, lw=2, label=month_name[m]))
            seen_months.add(m)
    for p in tick_pos:
        ax.axvline(p, color="k", alpha=0.06, linewidth=1)

    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=15, fontweight="bold")

    if yscale is not None:
        ax.set_yscale(yscale)
    if ylim is not None:
        ax.set_ylim(**ylim) if isinstance(ylim, dict) else ax.set_ylim(ylim)
    ax.tick_params(axis="y", labelsize=13)
    for t in ax.get_yticklabels():
        t.set_fontweight("bold")
    if show_xticks:
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab, rotation=60, ha="right", fontsize=11, fontweight="bold")

    return legend_month_handles
# MASS
legend_month_handles = plot_monthly(
    ax_mass,
    dfp_m_cas, dfp_m_cdp, xm_cas, xm_cdp, mass_cas_f, mass_cdp_f,
    yscale="log",
    ylim={"bottom": 10**(-1.3)},
    title="GCCN Mass seasonal trend over the Western Atlantic",
    ylabel="GCCN Mass (µg/m³)",
    show_xticks=True
)

# SLOPE
# _ = plot_monthly(
#     ax_slope,
#     dfp_s_cas, dfp_s_cdp, xs_cas, xs_cdp, slope_cas_f, slope_cdp_f,
#     yscale=None,
#     ylim={"top": 4},
#     title="GCCN Slope parameter seasonal trend over the Western Atlantic",
#     ylabel="Slope Parameter D (µm)",
#     show_xticks=False
# )

# GCCN
# _ = plot_monthly(
#     ax_gccn,
#     dfp_g_cas, dfp_g_cdp, xg_cas, xg_cdp, gccn_cas_f, gccn_cdp_f,
#     yscale="log",
#     ylim={"bottom": 10**(-2.3)},
#     title="GCCN concentration seasonal trend over the Western Atlantic",
#     ylabel="Total GCCN Concentration (cm⁻³)",
#     show_xticks=True
# )

ax_mass.set_xlabel("Flight Date", fontsize=15, fontweight="bold")
legend_instrument_handles = [
    Line2D([0], [0], color="k", lw=2, ls="-",  label="CAS (solid)"),
    Line2D([0], [0], color="k", lw=2, ls="--", label="CDP (dashed)"),
    Line2D([0], [0], marker="s", color="k", lw=0, markersize=9, label="CAS monthly median"),
    Line2D([0], [0], marker="^", color="k", lw=0, markersize=9, label="CDP monthly median"),
    Line2D([0], [0], color="k", lw=1.5, label="Monthly mean (whisker)")
]
handles = legend_month_handles + legend_instrument_handles
fig.legend(handles=handles, ncol=1, fontsize=16, frameon=True,
           loc="lower right", bbox_to_anchor=(1.15, 0.45))

fig.tight_layout()
fig.align_ylabels()
plt.show()
# %%
month_colors = {
    1: "#0072B2",
    2: "#E69F00",
    3: "#009E73",
    5: "#CC79A7",
    6: "#56B4E9",
}
mass_thr = 100.0
mass_low = 1e-3
dfp_m_cas, _, mass_cas_f, _, _ = prep_trend(df_sorted_cas, mass_cas, mass_thr, mass_low)
dfp_m_cdp, _, mass_cdp_f, _, _ = prep_trend(df_sorted_cdp, mass_cdp, mass_thr, mass_low)
all_dates = pd.concat([
    dfp_m_cas["Date_dt"], dfp_m_cdp["Date_dt"],
]).drop_duplicates().sort_values().reset_index(drop=True)

date_to_x = {d: i for i, d in enumerate(all_dates)}
tick_pos = np.arange(len(all_dates))
tick_lab = all_dates.dt.strftime("%m-%d").to_numpy()

def make_x(dfp, jitter=0.3):
    x = dfp["Date_dt"].map(date_to_x).to_numpy().astype(float)
    for d in np.unique(dfp["Date_dt"]):
        idx = np.where(dfp["Date_dt"] == d)[0]
        if len(idx) > 1:
            x[idx] += np.linspace(-jitter, jitter, len(idx))
    return x

xm_cas = make_x(dfp_m_cas)
xm_cdp = make_x(dfp_m_cdp)

fig_w = max(22, 0.55 * len(tick_pos))
fig, ax_mass = plt.subplots(1, 1, figsize=(fig_w, 12), sharex=True)

def plot_monthly(ax, dfp_cas, dfp_cdp, x_cas, x_cdp, y_cas, y_cdp, yscale=None, ylim=None,
                 title="", ylabel="", show_xticks=False):
    legend_month_handles = []
    seen_months = set()

    for m in sorted(dfp_cas["Month"].unique()):
        if m not in month_name:
            continue

        cas_mask = dfp_cas["Month"].values == m
        cdp_mask = dfp_cdp["Month"].values == m
        c = month_colors[m]

        ax.plot(x_cas[cas_mask], y_cas[cas_mask], "-",  lw=1.5, color=c)
        ax.plot(x_cdp[cdp_mask], y_cdp[cdp_mask], "--", lw=1.5, color=c)

        median_cas = np.nanmedian(y_cas[cas_mask])
        mean_cas   = np.nanmean(y_cas[cas_mask])
        median_cdp = np.nanmedian(y_cdp[cdp_mask])
        mean_cdp   = np.nanmean(y_cdp[cdp_mask])

        median_x = np.nanmedian(x_cas[cas_mask])

        ax.plot(median_x, median_cas, marker="s", color="k", markersize=15, linestyle="None")
        ax.vlines(median_x, min(median_cas, mean_cas), max(median_cas, mean_cas), color="k", lw=3.5)

        ax.plot(median_x + 0.15, median_cdp, marker="^", color="k", markersize=15, linestyle="None")
        ax.vlines(median_x + 0.15, min(median_cdp, mean_cdp), max(median_cdp, mean_cdp), color="k", lw=3.5)

        if m not in seen_months:
            legend_month_handles.append(Line2D([0], [0], color=c, lw=2, label=month_name[m]))
            seen_months.add(m)

    for p in tick_pos:
        ax.axvline(p, color="k", alpha=0.06, linewidth=1)

    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=18, fontweight="bold")

    if yscale is not None:
        ax.set_yscale(yscale)
    if ylim is not None:
        ax.set_ylim(**ylim) if isinstance(ylim, dict) else ax.set_ylim(ylim)

    ax.tick_params(axis="y", labelsize=18)
    for t in ax.get_yticklabels():
        t.set_fontweight("bold")

    if show_xticks:
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab, rotation=60, ha="right", fontsize=17, fontweight="bold")

    return legend_month_handles
legend_month_handles = plot_monthly(
    ax_mass,
    dfp_m_cas, dfp_m_cdp, xm_cas, xm_cdp, mass_cas_f, mass_cdp_f,
    yscale="log",
    ylim={"bottom": 10**(-1.3)},
    title="GCCN Mass seasonal trend over the Western Atlantic",
    ylabel="GCCN Mass (µg m$^{-3}$)",
    show_xticks=True
)
dfp_w = df_wind.copy()
dfp_w["Date_dt"] = pd.to_datetime(dfp_w["Date"])
if "Month" not in dfp_w.columns:
    dfp_w["Month"] = dfp_w["Date_dt"].dt.month

sort_cols = ["Date_dt"]
if "BCB_start" in dfp_w.columns:
    sort_cols.append("BCB_start")
elif "Min_start" in dfp_w.columns:
    sort_cols.append("Min_start")
dfp_w = dfp_w.sort_values(sort_cols).reset_index(drop=True)

wind_arr = pd.to_numeric(dfp_w["Windspeed"], errors="coerce").to_numpy()
xw = dfp_w["Date_dt"].map(date_to_x).to_numpy().astype(float)
jitter = 0.3
for d in np.unique(dfp_w["Date_dt"]):
    idx = np.where(dfp_w["Date_dt"].values == d)[0]
    if len(idx) > 1:
        xw[idx] += np.linspace(-jitter, jitter, len(idx))
ok = np.isfinite(xw) & np.isfinite(wind_arr)
xw = xw[ok]
wind_arr = wind_arr[ok]
dfp_w = dfp_w.loc[ok].reset_index(drop=True)
ax_wind = ax_mass.twinx()
ax_wind.set_ylabel("Wind Speed (m s$^{-1}$)", fontsize=18, fontweight="bold")
ax_wind.tick_params(axis="y", labelsize=18)
for t in ax_wind.get_yticklabels():
    t.set_fontweight("bold")
    ax_wind.set_ylim(0, 24)
for m in sorted(dfp_w["Month"].unique()):
    if m not in month_name:
        continue
    m_mask = (dfp_w["Month"].values == m)
    if not np.any(m_mask):
        continue
    c = month_colors[m]
    ax_wind.plot(xw[m_mask], wind_arr[m_mask], "-", lw=3, color="navy", alpha=0.6)

ax_mass.set_xlabel("Flight Date", fontsize=18, fontweight="bold")

legend_instrument_handles = [
    Line2D([0], [0], color="k", lw=2, ls="-",  label="CAS (solid)"),
    Line2D([0], [0], color="k", lw=2, ls="--", label="CDP (dashed)"),
    Line2D([0], [0], marker="s", color="k", lw=0, markersize=14, label="CAS monthly median"),
    Line2D([0], [0], marker="^", color="k", lw=0, markersize=14, label="CDP monthly median"),
    Line2D([0], [0], color="k", lw=1.5, label="Monthly mean (whisker)"),
    Line2D([0], [0], color="k", lw=2, alpha=0.6, label="Wind Speed (m s$^{-1}$)"),
]

handles = legend_month_handles + legend_instrument_handles
fig.legend(handles=handles, ncol=1, fontsize=18, frameon=True,
           loc="lower right", bbox_to_anchor=(1.19, 0.45))

fig.tight_layout()
plt.show()
# fig.savefig("mass_wind_trend.pdf", bbox_inches="tight")
# %%
#turning the above code into a scatterplot color coded by month
# Add a small instrument offset so CAS and CDP points do not sit
# directly on top of each other
xm_cas = make_x(dfp_m_cas) - 0.08
xm_cdp = make_x(dfp_m_cdp) + 0.08


def plot_monthly(
    ax,
    dfp_cas,
    dfp_cdp,
    x_cas,
    x_cdp,
    y_cas,
    y_cdp,
    yscale=None,
    ylim=None,
    title="",
    ylabel="",
    show_xticks=False
):
    legend_month_handles = []

    # Use months appearing in either instrument
    all_months = sorted(
        set(dfp_cas["Month"].dropna().astype(int).unique()) |
        set(dfp_cdp["Month"].dropna().astype(int).unique())
    )

    for m in all_months:
        if m not in month_name or m not in month_colors:
            continue

        c = month_colors[m]

        cas_mask = dfp_cas["Month"].to_numpy() == m
        cdp_mask = dfp_cdp["Month"].to_numpy() == m

        # CAS observations: circles
        if np.any(cas_mask):
            ax.scatter(
                x_cas[cas_mask],
                y_cas[cas_mask],
                marker="o",
                s=85,
                facecolor=c,
                edgecolor="k",
                linewidth=0.5,
                alpha=0.85,
                zorder=3
            )

        # CDP observations: squares
        if np.any(cdp_mask):
            ax.scatter(
                x_cdp[cdp_mask],
                y_cdp[cdp_mask],
                marker="s",
                s=85,
                facecolor=c,
                edgecolor="k",
                linewidth=0.5,
                alpha=0.85,
                zorder=3
            )

        # Determine a central x-position for the monthly statistics
        month_x_values = []

        if np.any(cas_mask):
            month_x_values.extend(x_cas[cas_mask])

        if np.any(cdp_mask):
            month_x_values.extend(x_cdp[cdp_mask])

        if len(month_x_values) > 0:
            monthly_x = np.nanmedian(month_x_values)

            # CAS monthly median and mean whisker
            cas_month_values = y_cas[cas_mask]
            cas_valid = np.isfinite(cas_month_values)

            if np.any(cas_valid):
                median_cas = np.nanmedian(cas_month_values)
                mean_cas = np.nanmean(cas_month_values)
                cas_summary_x = monthly_x - 0.15

                ax.plot(
                    cas_summary_x,
                    median_cas,
                    marker="o",
                    markerfacecolor="none",
                    markeredgecolor="k",
                    markeredgewidth=2.5,
                    markersize=15,
                    linestyle="None",
                    zorder=5
                )

                ax.vlines(
                    cas_summary_x,
                    min(median_cas, mean_cas),
                    max(median_cas, mean_cas),
                    color="k",
                    linewidth=3.5,
                    zorder=4
                )

            # CDP monthly median and mean whisker
            cdp_month_values = y_cdp[cdp_mask]
            cdp_valid = np.isfinite(cdp_month_values)

            if np.any(cdp_valid):
                median_cdp = np.nanmedian(cdp_month_values)
                mean_cdp = np.nanmean(cdp_month_values)
                cdp_summary_x = monthly_x + 0.15

                ax.plot(
                    cdp_summary_x,
                    median_cdp,
                    marker="s",
                    markerfacecolor="none",
                    markeredgecolor="k",
                    markeredgewidth=2.5,
                    markersize=15,
                    linestyle="None",
                    zorder=5
                )

                ax.vlines(
                    cdp_summary_x,
                    min(median_cdp, mean_cdp),
                    max(median_cdp, mean_cdp),
                    color="k",
                    linewidth=3.5,
                    zorder=4
                )

        # Month legend entry
        legend_month_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=c,
                markeredgecolor="k",
                markersize=11,
                label=month_name[m]
            )
        )

    for p in tick_pos:
        ax.axvline(
            p,
            color="k",
            alpha=0.06,
            linewidth=1,
            zorder=0
        )

    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=18, fontweight="bold")

    if yscale is not None:
        ax.set_yscale(yscale)

    if ylim is not None:
        if isinstance(ylim, dict):
            ax.set_ylim(**ylim)
        else:
            ax.set_ylim(ylim)

    ax.tick_params(axis="y", labelsize=18)

    for t in ax.get_yticklabels():
        t.set_fontweight("bold")

    if show_xticks:
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(
            tick_lab,
            rotation=60,
            ha="right",
            fontsize=17,
            fontweight="bold"
        )

    return legend_month_handles
# %%
legend_month_handles = plot_monthly(
    ax_mass,
    dfp_m_cas,
    dfp_m_cdp,
    xm_cas,
    xm_cdp,
    mass_cas_f,
    mass_cdp_f,
    yscale="log",
    ylim={"bottom": 10**(-1.3)},
    title="GCCN Mass seasonal trend over the Western Atlantic",
    ylabel="GCCN Mass (µg m$^{-3}$)",
    show_xticks=True
)
# %%
from matplotlib.lines import Line2D

month_colors = {
    1: "#0072B2",
    2: "#E69F00",
    3: "#009E73",
    5: "#CC79A7",
    6: "#56B4E9",
}

mass_thr = 100.0
mass_low = 1e-3


# -------------------------------------------------------------
# Prepare CAS and CDP mass data
# -------------------------------------------------------------
dfp_m_cas, _, mass_cas_f, _, _ = prep_trend(
    df_sorted_cas,
    mass_cas,
    mass_thr,
    mass_low
)

dfp_m_cdp, _, mass_cdp_f, _, _ = prep_trend(
    df_sorted_cdp,
    mass_cdp,
    mass_thr,
    mass_low
)

# Make sure the plotted arrays are NumPy arrays
mass_cas_f = np.asarray(mass_cas_f, dtype=float)
mass_cdp_f = np.asarray(mass_cdp_f, dtype=float)

dfp_m_cas = dfp_m_cas.copy()
dfp_m_cdp = dfp_m_cdp.copy()

dfp_m_cas["Date_dt"] = pd.to_datetime(dfp_m_cas["Date_dt"])
dfp_m_cdp["Date_dt"] = pd.to_datetime(dfp_m_cdp["Date_dt"])

# Make sure Month exists and is numeric
dfp_m_cas["Month"] = dfp_m_cas["Date_dt"].dt.month
dfp_m_cdp["Month"] = dfp_m_cdp["Date_dt"].dt.month


# -------------------------------------------------------------
# Common flight-date x-axis
# -------------------------------------------------------------
all_dates = pd.concat(
    [
        dfp_m_cas["Date_dt"],
        dfp_m_cdp["Date_dt"],
    ],
    ignore_index=True
).drop_duplicates().sort_values().reset_index(drop=True)

date_to_x = {
    date: i
    for i, date in enumerate(all_dates)
}

tick_pos = np.arange(len(all_dates))
tick_lab = all_dates.dt.strftime("%m-%d").to_numpy()


def make_x(dfp, jitter=0.30):
    x = (
        dfp["Date_dt"]
        .map(date_to_x)
        .to_numpy(dtype=float)
    )

    for date in dfp["Date_dt"].drop_duplicates():
        idx = np.where(
            dfp["Date_dt"].to_numpy() == date
        )[0]

        if len(idx) > 1:
            x[idx] += np.linspace(
                -jitter,
                jitter,
                len(idx)
            )

    return x


# Offset the instruments slightly so their points are visible
xm_cas = make_x(dfp_m_cas) - 0.06
xm_cdp = make_x(dfp_m_cdp) + 0.06


# -------------------------------------------------------------
# Create a NEW figure and axis
# -------------------------------------------------------------
plt.close("all")

fig_w = max(22, 0.55 * len(tick_pos))

fig, ax_mass = plt.subplots(
    figsize=(fig_w, 12)
)


# -------------------------------------------------------------
# CAS/CDP monthly scatter plotting function
# -------------------------------------------------------------
def plot_monthly(
    ax,
    dfp_cas,
    dfp_cdp,
    x_cas,
    x_cdp,
    y_cas,
    y_cdp,
    yscale=None,
    ylim=None,
    title="",
    ylabel="",
    show_xticks=False
):
    y_cas = np.asarray(y_cas, dtype=float)
    y_cdp = np.asarray(y_cdp, dtype=float)

    legend_month_handles = []

    all_months = sorted(
        set(dfp_cas["Month"].dropna().astype(int).unique()) |
        set(dfp_cdp["Month"].dropna().astype(int).unique())
    )

    for m in all_months:

        if m not in month_name:
            continue

        if m not in month_colors:
            continue

        color = month_colors[m]

        cas_mask = (
            dfp_cas["Month"].to_numpy(dtype=int) == m
        )

        cdp_mask = (
            dfp_cdp["Month"].to_numpy(dtype=int) == m
        )

        # Also remove nonfinite x/y values
        cas_valid = (
            cas_mask
            & np.isfinite(x_cas)
            & np.isfinite(y_cas)
        )

        cdp_valid = (
            cdp_mask
            & np.isfinite(x_cdp)
            & np.isfinite(y_cdp)
        )

        # CAS = filled circles
        if np.any(cas_valid):
            ax.scatter(
                x_cas[cas_valid],
                y_cas[cas_valid],
                marker="o",
                s=95,
                facecolor=color,
                edgecolor="black",
                linewidth=0.7,
                alpha=0.90,
                zorder=4
            )

        # CDP = filled squares
        if np.any(cdp_valid):
            ax.scatter(
                x_cdp[cdp_valid],
                y_cdp[cdp_valid],
                marker="s",
                s=95,
                facecolor=color,
                edgecolor="black",
                linewidth=0.7,
                alpha=0.90,
                zorder=4
            )

        # -----------------------------------------------------
        # Monthly mean/median statistics
        # -----------------------------------------------------
        month_x_values = []

        if np.any(cas_valid):
            month_x_values.extend(
                x_cas[cas_valid].tolist()
            )

        if np.any(cdp_valid):
            month_x_values.extend(
                x_cdp[cdp_valid].tolist()
            )

        if len(month_x_values) > 0:

            monthly_x = np.nanmedian(month_x_values)

            # CAS monthly median and mean whisker
            if np.any(cas_valid):

                median_cas = np.nanmedian(
                    y_cas[cas_valid]
                )

                mean_cas = np.nanmean(
                    y_cas[cas_valid]
                )

                cas_summary_x = monthly_x - 0.18

                ax.vlines(
                    cas_summary_x,
                    min(median_cas, mean_cas),
                    max(median_cas, mean_cas),
                    color="black",
                    linewidth=3.5,
                    zorder=6
                )

                ax.plot(
                    cas_summary_x,
                    median_cas,
                    marker="o",
                    markerfacecolor="white",
                    markeredgecolor="black",
                    markeredgewidth=2.5,
                    markersize=15,
                    linestyle="None",
                    zorder=7
                )
            if np.any(cdp_valid):

                median_cdp = np.nanmedian(
                    y_cdp[cdp_valid]
                )

                mean_cdp = np.nanmean(
                    y_cdp[cdp_valid]
                )

                cdp_summary_x = monthly_x + 0.18

                ax.vlines(
                    cdp_summary_x,
                    min(median_cdp, mean_cdp),
                    max(median_cdp, mean_cdp),
                    color="black",
                    linewidth=3.5,
                    zorder=6
                )

                ax.plot(
                    cdp_summary_x,
                    median_cdp,
                    marker="s",
                    markerfacecolor="white",
                    markeredgecolor="black",
                    markeredgewidth=2.5,
                    markersize=15,
                    linestyle="None",
                    zorder=7
                )
        legend_month_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=11,
                label=month_name[m]
            )        )

    for p in tick_pos:
        ax.axvline(
            p,
            color="black",
            alpha=0.06,
            linewidth=1,
            zorder=0        )

    ax.grid(
        alpha=0.30,
        zorder=0    )

    ax.set_title(
        title,
        fontsize=18,
        fontweight="bold"    )

    ax.set_ylabel(
        ylabel,
        fontsize=18,
        fontweight="bold"    )

    if yscale is not None:
        ax.set_yscale(yscale)

    if ylim is not None:

        if isinstance(ylim, dict):
            ax.set_ylim(**ylim)

        else:
            ax.set_ylim(ylim)

    ax.tick_params(
        axis="y",
        labelsize=18    )

    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    if show_xticks:

        ax.set_xticks(tick_pos)

        ax.set_xticklabels(
            tick_lab,
            rotation=60,
            ha="right",
            fontsize=17,
            fontweight="bold"        )

    return legend_month_handles
legend_month_handles = plot_monthly(
    ax_mass,
    dfp_m_cas,
    dfp_m_cdp,
    xm_cas,
    xm_cdp,
    mass_cas_f,
    mass_cdp_f,
    yscale="log",
    ylim={"bottom": 10**(-1.3)},
    title="GCCN Mass seasonal trend over the Western Atlantic",
    ylabel="GCCN Mass (µg m$^{-3}$)",
    show_xticks=True)
dfp_w = df_wind.copy()

dfp_w["Date_dt"] = pd.to_datetime(
    dfp_w["Date"])
dfp_w["Month"] = dfp_w["Date_dt"].dt.month

sort_cols = ["Date_dt"]

if "BCB_start" in dfp_w.columns:
    sort_cols.append("BCB_start")
elif "Min_start" in dfp_w.columns:
    sort_cols.append("Min_start")
dfp_w = (
    dfp_w
    .sort_values(sort_cols)
    .reset_index(drop=True))
wind_arr = pd.to_numeric(
    dfp_w["Windspeed"],
    errors="coerce"
).to_numpy(dtype=float)
xw = (
    dfp_w["Date_dt"]
    .map(date_to_x)
    .to_numpy(dtype=float))
jitter = 0.30
for date in dfp_w["Date_dt"].drop_duplicates():

    idx = np.where(
        dfp_w["Date_dt"].to_numpy() == date
    )[0]
    if len(idx) > 1:
        xw[idx] += np.linspace(
            -jitter,
            jitter,
            len(idx)        )
wind_valid = (
    np.isfinite(xw)
    & np.isfinite(wind_arr))
xw = xw[wind_valid]
wind_arr = wind_arr[wind_valid]
dfp_w = (
    dfp_w.loc[wind_valid]
    .reset_index(drop=True))
wind_order = np.argsort(xw)
xw = xw[wind_order]
wind_arr = wind_arr[wind_order]
ax_wind = ax_mass.twinx()
ax_wind.set_ylabel(
    "Wind Speed (m s$^{-1}$)",
    fontsize=18,
    fontweight="bold")
ax_wind.tick_params(
    axis="y",
    labelsize=18)
ax_wind.set_ylim(0, 24)
for tick in ax_wind.get_yticklabels():
    tick.set_fontweight("bold")
ax_wind.plot(
    xw,
    wind_arr,
    linestyle="-",
    linewidth=3,
    color="navy",
    alpha=0.60,
    zorder=2)
ax_mass.set_zorder(
    ax_wind.get_zorder() + 1)
ax_mass.patch.set_visible(False)
ax_mass.set_xlabel(
    "Flight Date",
    fontsize=18,
    fontweight="bold")
ax_mass.set_xlim(
    -0.6,
    len(tick_pos) - 0.4)
legend_instrument_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="gray",
        markeredgecolor="black",
        markersize=11,
        label="CAS observations"    ),

    Line2D(
        [0],
        [0],
        marker="x",
        color="none",
        markerfacecolor="gray",
        markeredgecolor="black",
        markersize=11,
        label="CDP observations"    ),

    Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=2.5,
        markersize=14,
        label="CAS monthly median"    ),
    Line2D(
        [0],
        [0],
        marker="x",
        color="none",
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=2.5,
        markersize=14,
        label="CDP monthly median"    ),

    Line2D(
        [0],
        [0],
        color="black",
        linewidth=3.5,
        label="Monthly mean–median difference"
    ),
    Line2D(
        [0],
        [0],
        color="navy",
        linewidth=3,
        alpha=0.60,
        label="Wind Speed"
    ),]
handles = (
    legend_month_handles
    + legend_instrument_handles)
fig.legend(
    handles=handles,
    ncol=1,
    fontsize=18,
    frameon=True,
    loc="center left",
    bbox_to_anchor=(0.82, 0.50))
fig.subplots_adjust(
    right=0.80,
    bottom=0.20)
plt.show()
# %%
