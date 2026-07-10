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
import pickle
#%%
with open("FCDP_slope_massLE1002020.pkl", "rb") as f:
    filtered_slope_mass100gone = pickle.load(f)
print("Total number of legs:", len(filtered_slope_mass100gone))
# %%
#monthly trend of dry mass
df_fcdp = pd.DataFrame(filtered_slope_mass100gone).copy()
df_fcdp = df_fcdp[df_fcdp["Date"].astype(str).str.startswith("2020-")].copy()
df_fcdp["Month"] = df_fcdp["Date"].astype(str).str[5:7].astype(int)
df_fcdp = df_fcdp[df_fcdp["Month"].isin([2, 3, 8, 9])].copy()
df_fcdp_sorted = df_fcdp.sort_values(
    ["Date", "BCB_start"],
    kind="mergesort"
).reset_index(drop=True)
mass_fcdp = df_fcdp_sorted["Dry Mass (µg/m³)"].astype(float).values
x = np.arange(len(df_fcdp_sorted))
plt.figure(figsize=(12, 4.8))
plt.plot(x, mass_fcdp, '-')
plt.yscale("log")
plt.grid(alpha=0.3)
plt.xlabel("Leg index (sorted by Date, then BCB_start)", fontsize=13, fontweight="bold")
plt.ylabel("Dry GCCN Mass (µg/m³)", fontsize=13, fontweight="bold")
plt.title("Dry Mass Timeline (FMAS 2020)\nLegs ordered by Date then BCB_start",
          fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
#%%
df_fcdp = pd.DataFrame(filtered_slope_mass100gone).copy()
df_fcdp = df_fcdp[
    df_fcdp["Date"].astype(str).str.startswith("2020-")
].copy()
df_fcdp["Month"] = (
    df_fcdp["Date"]
    .astype(str)
    .str[5:7]
    .astype(int)
)
df_fcdp = df_fcdp[
    df_fcdp["Month"].isin([2, 3, 8, 9])
].copy()
bad_mass_mask = (
    pd.to_numeric(
        df_fcdp["Dry Mass (µg/m³)"],
        errors="coerce"
    ) < 0.001
)
print("Removing these low-mass legs:")
print(
    df_fcdp.loc[
        bad_mass_mask,
        ["Date", "BCB_start", "BCB_stop", "Dry Mass (µg/m³)"]
    ]
)
df_fcdp = df_fcdp.loc[~bad_mass_mask].copy()
#%%
df_fcdp_sorted = df_fcdp.sort_values(
    ["Date", "BCB_start"],
    kind="mergesort"
).reset_index(drop=True)
mass_fcdp = (
    df_fcdp_sorted["Dry Mass (µg/m³)"]
    .astype(float)
    .values
)
x = np.arange(len(df_fcdp_sorted))
plt.figure(figsize=(12, 4.8))
plt.plot(x, mass_fcdp, '-')
plt.yscale("log")
plt.grid(alpha=0.3)
plt.xlabel(
    "Leg index (sorted by Date, then BCB_start)",
    fontsize=13,
    fontweight="bold"
)
plt.ylabel(
    "Dry GCCN Mass (µg/m³)",
    fontsize=13,
    fontweight="bold"
)
plt.title(
    "Dry Mass Timeline (FMAS 2020)\nLegs ordered by Date then BCB_start",
    fontsize=14,
    fontweight="bold"
)
plt.tight_layout()
plt.show()
#%%
#monthly mass trend coded with color seperation
month_name = {
    2: "February",
    3: "March",
    8: "August",
    9: "September"
}

plt.figure(figsize=(12, 4.8))

for m in sorted(df_fcdp_sorted["Month"].unique()):
    if m not in month_name:
        continue

    m_mask = (df_fcdp_sorted["Month"].values == m)

    plt.plot(
        x[m_mask],
        mass_fcdp[m_mask],
        '-o',
        label=month_name[m],
        markersize=6
    )

plt.yscale("log")
plt.grid(alpha=0.3)
plt.ylabel("GCCN Mass (µg/m³)", fontsize=16, fontweight="bold")
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.title("FCDP BCB Mass FMAS 2020", fontsize=18, fontweight="bold")
plt.legend(ncol=2, fontsize=10)
plt.yticks(fontsize=14, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
#%%
#monthly mean mass trend coded with color seperation
plt.figure(figsize=(12, 4.8))
for m in sorted(df_fcdp_sorted["Month"].unique()): 
    if m not in month_name:
        continue
    m_mask = (df_fcdp_sorted["Month"].values == m)
    mean_mass = np.mean(mass_fcdp[m_mask])
    plt.plot(
        x[m_mask],
        mass_fcdp[m_mask],
        '-',
        label=f"{month_name[m]} (mean: {mean_mass:.2f} µg/m³)"
    )
plt.yscale("log")
plt.grid(alpha=0.3)
plt.ylabel("GCCN Mass (µg/m³)", fontsize=16, fontweight="bold")
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.title("FCDP BCB Mass FMAS 2020\nMonthly Means", fontsize=18, fontweight="bold")
plt.legend(ncol=2, fontsize=10)
plt.yticks(fontsize=14, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
#%%
#monthly median trend coded with color seperation
plt.figure(figsize=(12, 4.8))
for m in sorted(df_fcdp_sorted["Month"].unique()): 
    if m not in month_name:
        continue
    m_mask = (df_fcdp_sorted["Month"].values == m)
    median_mass = np.median(mass_fcdp[m_mask])
    plt.plot(
        x[m_mask],
        mass_fcdp[m_mask],
        '-',
        label=f"{month_name[m]} (median: {median_mass:.2f} µg/m³)"
    )
plt.yscale("log")
plt.grid(alpha=0.3)
plt.ylabel("GCCN Mass (µg/m³)", fontsize=16, fontweight="bold")
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.title("FCDP BCB Mass FMAS 2020\nMonthly Medians", fontsize=18, fontweight="bold")
plt.legend(ncol=2, fontsize=10)
plt.ylim(0.01, 1000)
plt.yticks(fontsize=14, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
#%%
dfp_fcdp = df_fcdp_sorted.reset_index(drop=True).copy()
dfp_fcdp["Date_dt"] = pd.to_datetime(dfp_fcdp["Date"])
x = np.arange(len(dfp_fcdp))
tick_pos = dfp_fcdp.groupby(dfp_fcdp["Date_dt"].dt.date).head(1).index.to_numpy()
tick_lab = dfp_fcdp.loc[tick_pos, "Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
#%%
#monthly mean mass trend coded with color seperation and date x-axis
dfp_fcdp = df_fcdp_sorted.copy()
dfp_fcdp["Date_dt"] = pd.to_datetime(dfp_fcdp["Date"])
sort_cols = ["Date_dt"]
if "BCB_start" in dfp_fcdp.columns:
    sort_cols.append("BCB_start")
dfp_fcdp = dfp_fcdp.sort_values(sort_cols).reset_index(drop=True)
mass_arr = dfp_fcdp["Dry Mass (µg/m³)"].astype(float).values
x = np.arange(len(dfp_fcdp))
date_first = dfp_fcdp.groupby(dfp_fcdp["Date_dt"].dt.date).head(1)
tick_pos = date_first.index.to_numpy()
tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
fig_w = max(14, 0.55 * len(tick_pos))
fig, ax = plt.subplots(figsize=(fig_w, 6.2))
for m in sorted(dfp_fcdp["Month"].unique()):
    if m not in month_name:
        continue
    m_mask = dfp_fcdp["Month"].values == m
    mean_mass = np.nanmean(mass_arr[m_mask])
    ax.plot(
        x[m_mask],
        mass_arr[m_mask],
        '-o',
        linewidth=1.5,
        markersize=5,
        label=f"{month_name[m]} (mean: {mean_mass:.2f} µg/m³)"
    )
for p in tick_pos:
    ax.axvline(p, color="k", alpha=0.06, linewidth=1)
ax.set_yscale("log")
ax.grid(alpha=0.3)
ax.set_ylabel("GCCN Mass (µg/m³)", fontsize=16, fontweight="bold")
ax.set_xlabel("Leg Date (every flight day)", fontsize=16, fontweight="bold")
ax.set_title("FCDP BCB Mass FMAS 2020\nMonthly Means",
             fontsize=18, fontweight="bold")
ax.legend(ncol=2, fontsize=10, loc="upper right")
ax.set_xticks(tick_pos)
ax.set_xticklabels(
    tick_lab,
    rotation=75,
    ha="right",
    fontsize=8,
    fontweight="bold"
)
labels = ax.get_xticklabels()
for i, lab in enumerate(labels):
    lab.set_text("\n" * (i % 4) + lab.get_text())
ax.set_xticklabels([lab.get_text() for lab in labels])
fig.subplots_adjust(bottom=0.40)
fig.tight_layout()
plt.show()
#%%
#keeping the same plot and code but adding a black thick triangle for the mean of each 
#month and a thick black circle for the median of each month
dfp_fcdp = df_fcdp_sorted.copy()
dfp_fcdp["Date_dt"] = pd.to_datetime(dfp_fcdp["Date"])
sort_cols = ["Date_dt"]
if "BCB_start" in dfp_fcdp.columns:
    sort_cols.append("BCB_start")
dfp_fcdp = dfp_fcdp.sort_values(sort_cols).reset_index(drop=True)
x = np.arange(len(dfp_fcdp))
mass_arr = np.asarray(dfp_fcdp["Dry Mass (µg/m³)"].astype(float))
date_first = dfp_fcdp.groupby(dfp_fcdp["Date_dt"].dt.date).head(1)
tick_pos = date_first.index.to_numpy()
tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
fig_w = max(18, 0.9 * len(tick_pos))
fig, ax = plt.subplots(figsize=(fig_w, 6.2))
legend_handles = []

for m in sorted(dfp_fcdp["Month"].unique()):
    if m not in month_name:
        continue

    m_mask = (dfp_fcdp["Month"].values == m)
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
ax.set_title("FCDP BCB Mass FMAS 2020",
             fontsize=20, fontweight="bold")
ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_lab, rotation=75, ha="right", fontsize=8, fontweight="bold")
labels = ax.get_xticklabels()
for i, lab in enumerate(labels):
    base = lab.get_text()
    lab.set_text("\n" * (i % 4) + base)
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
#monthly trend of Dry Slope (D)
df_fcdp = pd.DataFrame(filtered_slope_mass100gone).copy()

df_fcdp = df_fcdp[df_fcdp["Date"].astype(str).str.startswith("2020-")].copy()
df_fcdp["Month"] = df_fcdp["Date"].astype(str).str[5:7].astype(int)
df_fcdp = df_fcdp[df_fcdp["Month"].isin([2, 3, 8, 9])].copy()
df_sorted_fcdp = df_fcdp.sort_values(
    ["Date", "BCB_start"],
    kind="mergesort"
).reset_index(drop=True)
slope_fcdp = df_sorted_fcdp["Dry Slope (D)"].astype(float).values
x = np.arange(len(df_sorted_fcdp))
plt.figure(figsize=(12, 4.8))
plt.plot(x, slope_fcdp, '-o', markersize=4)
plt.grid(alpha=0.3)
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.ylabel("Slope (D)", fontsize=16, fontweight="bold")
plt.title("FCDP BCB Slope FMAS 2020",
          fontsize=18, fontweight="bold")
plt.tight_layout()
plt.show()
# %%
#monthly Dry Slope (D) trend coded with color seperation
plt.figure(figsize=(12, 4.8))
for m in sorted(df_sorted_fcdp["Month"].unique()):
    if m not in month_name:
        continue
    m_mask = (df_sorted_fcdp["Month"].values == m)
    plt.plot(
        x[m_mask],
        slope_fcdp[m_mask],
        '-',
        label=f"{month_name[m]}"
    )
plt.grid(alpha=0.3)
plt.ylabel("Slope (D)", fontsize=16, fontweight="bold")
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.title("FCDP BCB Slope (D) FMAS 2020",
          fontsize=18, fontweight="bold")
plt.legend(ncol=2, fontsize=10)
plt.yticks(fontsize=14, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
# %%
#monthly mean Dry Slope (D) trend coded with color seperation
plt.figure(figsize=(12, 4.8))
for m in sorted(df_sorted_fcdp["Month"].unique()): 
    if m not in month_name:
        continue
    m_mask = (df_sorted_fcdp["Month"].values == m)
    mean_slope = np.mean(slope_fcdp[m_mask])
    plt.plot(
        x[m_mask],
        slope_fcdp[m_mask],
        '-',
        label=f"{month_name[m]} (mean: {mean_slope:.2f})"
    )
plt.yscale("log")
plt.grid(alpha=0.3)
plt.ylabel("Dry GCCN Mass (µg/m³)", fontsize=16, fontweight="bold")
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.title("FCDP BCB Dry Slope (D) FMAS 2020\nMonthly Means", fontsize=18, fontweight="bold")
plt.legend(ncol=2, fontsize=10)
plt.yticks(fontsize=14, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
# %%
#monthly median Dry Slope (D) trend coded with color seperation
plt.figure(figsize=(12, 4.8))
for m in sorted(df_sorted_fcdp["Month"].unique()): 
    if m not in month_name:
        continue
    m_mask = (df_sorted_fcdp["Month"].values == m)
    median_slope = np.median(slope_fcdp[m_mask])
    plt.plot(
        x[m_mask],
        slope_fcdp[m_mask],
        '-',
        label=f"{month_name[m]} (median: {median_slope:.2f})"
    )
plt.yscale("log")
plt.grid(alpha=0.3)
plt.ylabel("Dry Slope (D)", fontsize=16, fontweight="bold")
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.title("FCDP BCB Dry Slope (D) FMAS 2020\nMonthly Medians", fontsize=18, fontweight="bold")
plt.legend(ncol=2, fontsize=10)
plt.yticks(fontsize=14, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
# %%
dfp_fcdp = df_sorted_fcdp.reset_index(drop=True).copy()
dfp_fcdp["Date_dt"] = pd.to_datetime(dfp_fcdp["Date"])
x = np.arange(len(dfp_fcdp))
tick_pos = dfp_fcdp.groupby(dfp_fcdp["Date_dt"].dt.date).head(1).index.to_numpy()
tick_lab = dfp_fcdp.loc[tick_pos, "Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
#%%
#monthly mean slope trend coded with color seperation and date x-axis
dfp_fcdp = df_sorted_fcdp.copy()
dfp_fcdp["Date_dt"] = pd.to_datetime(dfp_fcdp["Date"])
sort_cols = ["Date_dt"]
if "BCB_start" in dfp_fcdp.columns:
    sort_cols.append("BCB_start")

dfp_fcdp = dfp_fcdp.sort_values(sort_cols).reset_index(drop=True)
x = np.arange(len(dfp_fcdp))
slope_arr = np.asarray(slope_fcdp)
date_first = dfp_fcdp.groupby(dfp_fcdp["Date_dt"].dt.date).head(1)
tick_pos = date_first.index.to_numpy()
tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
fig_w = max(22, 0.55 * len(tick_pos)) 
fig, ax = plt.subplots(figsize=(fig_w, 6.2))

for m in sorted(dfp_fcdp["Month"].unique()):
    if m not in month_name:
        continue
    m_mask = (dfp_fcdp["Month"].values == m)
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
ax.set_ylabel("Slope (D) (µg/m³)", fontsize=16, fontweight="bold")
ax.set_xlabel("Flight Date", fontsize=16, fontweight="bold")
ax.set_title("FCDP BCB Slope (D) FMAS 2020\nMonthly Trend", fontsize=18, fontweight="bold")
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
# %%
#keeping the same plot and code but adding a thick triangle for the mean of each 
#month and a thick circle for the median of each month for slope
dfp_fcdp = df_sorted_fcdp.copy()
dfp_fcdp["Date_dt"] = pd.to_datetime(dfp_fcdp["Date"])
sort_cols = ["Date_dt"]
if "BCB_start" in dfp_fcdp.columns:
    sort_cols.append("BCB_start")
dfp_fcdp = dfp_fcdp.sort_values(sort_cols).reset_index(drop=True)
x = np.arange(len(dfp_fcdp))
slope_arr = np.asarray(slope_fcdp)
date_first = dfp_fcdp.groupby(dfp_fcdp["Date_dt"].dt.date).head(1)
tick_pos = date_first.index.to_numpy()
tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
fig_w = max(22, 0.55 * len(tick_pos))
fig, ax = plt.subplots(figsize=(fig_w, 6.2))
legend_handles = []

for m in sorted(dfp_fcdp["Month"].unique()):
    if m not in month_name:
        continue

    m_mask = (dfp_fcdp["Month"].values == m)
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
ax.set_title("FCDP BCB Slope FMAS 2020\nMonthly Trend",
             fontsize=20, fontweight="bold")
ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_lab, rotation=60, ha="right", fontsize=7, fontweight="bold")
labels = ax.get_xticklabels()
for i, lab in enumerate(labels):
    base = lab.get_text()
    lab.set_text("\n" * (i % 4) + base)
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
dfp_fcdp = df_sorted_fcdp.copy()
dfp_fcdp["Date_dt"] = pd.to_datetime(dfp_fcdp["Date"])
sort_cols = ["Date_dt"]
if "BCB_start" in dfp_fcdp.columns:
    sort_cols.append("BCB_start")
dfp_fcdp = dfp_fcdp.sort_values(sort_cols).reset_index(drop=True)
x = np.arange(len(dfp_fcdp))
slope_arr = pd.to_numeric(dfp_fcdp["Dry Slope (D)"], errors="coerce").to_numpy(dtype=float)
date_first = dfp_fcdp.groupby(dfp_fcdp["Date_dt"].dt.date).head(1)
tick_pos = date_first.index.to_numpy()
tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
fig_w = max(18, 0.9 * len(tick_pos))
fig, ax = plt.subplots(figsize=(fig_w, 6.2))
legend_handles = []
for m in sorted(dfp_fcdp["Month"].unique()):
    if m not in month_name:
        continue
    m_mask = dfp_fcdp["Month"].values == m
    mean_slope = np.nanmean(slope_arr[m_mask])
    median_slope = np.nanmedian(slope_arr[m_mask])
    line, = ax.plot(
        x[m_mask],
        slope_arr[m_mask],
        '-o',
        linewidth=1.5,
        markersize=4
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
               label=f"{month_name[m]} mean = {mean_slope:.2f}"),
        Line2D([0], [0], marker="o", color=c, lw=0, markersize=10,
               label=f"{month_name[m]} median = {median_slope:.2f}"),
    ])
for p in tick_pos:
    ax.axvline(p, color="k", alpha=0.06, linewidth=1)
ax.grid(alpha=0.3)
ax.set_ylabel("Slope (D)", fontsize=20, fontweight="bold")
ax.set_xlabel("Flight Date", fontsize=20, fontweight="bold")
ax.set_title("FCDP BCB Slope FMAS 2020",
             fontsize=20, fontweight="bold")
ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_lab, rotation=75, ha="right", fontsize=8, fontweight="bold")
labels = ax.get_xticklabels()
for i, lab in enumerate(labels):
    base = lab.get_text()
    lab.set_text("\n" * (i % 4) + base)
ax.set_xticklabels([lab.get_text() for lab in labels])
ax.legend(handles=legend_handles, ncol=3, fontsize=9, loc="upper right", frameon=True)
fig.subplots_adjust(bottom=0.40)
fig.tight_layout()
plt.show()

#%%
#concentration 
with open("FCDP_concentration_massLE1002020.pkl", "rb") as f:
    total_concentration_cm3_mass100gone = pickle.load(f)
print("Total number of legs:", len(total_concentration_cm3_mass100gone))

# %%
# FCDP concentration monthly trend

dfp_fcdp = pd.DataFrame(total_concentration_cm3_mass100gone).copy()
dfp_fcdp = dfp_fcdp[dfp_fcdp["Date"].astype(str).str.startswith("2020-")].copy()
dfp_fcdp["Month"] = dfp_fcdp["Date"].astype(str).str[5:7].astype(int)
dfp_fcdp = dfp_fcdp[dfp_fcdp["Month"].isin([2, 3, 8, 9])].copy()
dfp_fcdp["Date_dt"] = pd.to_datetime(dfp_fcdp["Date"])
sort_cols = ["Date_dt"]
if "BCB_start" in dfp_fcdp.columns:
    sort_cols.append("BCB_start")
dfp_fcdp = dfp_fcdp.sort_values(sort_cols).reset_index(drop=True)
x = np.arange(len(dfp_fcdp))
conc_arr = pd.to_numeric(
    dfp_fcdp["Total_Y_Concentration_cm3"],
    errors="coerce"
).to_numpy(dtype=float)
date_first = dfp_fcdp.groupby(dfp_fcdp["Date_dt"].dt.date).head(1)
tick_pos = date_first.index.to_numpy()
tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
fig_w = max(18, 0.9 * len(tick_pos))
fig, ax = plt.subplots(figsize=(fig_w, 6.2))
legend_handles = []
for m in sorted(dfp_fcdp["Month"].unique()):
    if m not in month_name:
        continue
    m_mask = dfp_fcdp["Month"].values == m
    mean_conc = np.nanmean(conc_arr[m_mask])
    median_conc = np.nanmedian(conc_arr[m_mask])
    line, = ax.plot(
        x[m_mask],
        conc_arr[m_mask],
        '-o',
        linewidth=1.5,
        markersize=4
    )
    c = line.get_color()
    mean_x = x[m_mask][len(x[m_mask]) // 2]
    ax.plot(mean_x, mean_conc, marker="^", color=c,
            markersize=12, markeredgewidth=1.5, linestyle="None")
    ax.plot(mean_x + 5, median_conc, marker="o", color=c,
            markersize=12, markeredgewidth=1.5, linestyle="None")
    legend_handles.extend([
        Line2D([0], [0], color=c, lw=2, label=month_name[m]),
        Line2D([0], [0], marker="^", color=c, lw=0, markersize=10,
               label=f"{month_name[m]} mean = {mean_conc:.2f} cm⁻³"),
        Line2D([0], [0], marker="o", color=c, lw=0, markersize=10,
               label=f"{month_name[m]} median = {median_conc:.2f} cm⁻³"),
    ])

for p in tick_pos:
    ax.axvline(p, color="k", alpha=0.06, linewidth=1)
ax.grid(alpha=0.3)
ax.set_ylabel("FCDP Concentration (cm⁻³)", fontsize=20, fontweight="bold")
ax.set_xlabel("Flight Date", fontsize=20, fontweight="bold")
ax.set_title("FCDP Concentration FMAS 2020",
             fontsize=20, fontweight="bold")
ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_lab, rotation=75, ha="right", fontsize=8, fontweight="bold")
labels = ax.get_xticklabels()
for i, lab in enumerate(labels):
    base = lab.get_text()
    lab.set_text("\n" * (i % 4) + base)
ax.set_xticklabels([lab.get_text() for lab in labels])
ax.legend(handles=legend_handles, ncol=3, fontsize=9, loc="upper right", frameon=True)
fig.subplots_adjust(bottom=0.40)
fig.tight_layout()
plt.show()
# %%
df_mass_slope = pd.DataFrame(filtered_slope_mass100gone).copy()
df_conc = pd.DataFrame(total_concentration_cm3_mass100gone).copy()

# Convert Date to string first
for df in [df_mass_slope, df_conc]:
    df["Date"] = df["Date"].astype(str)

# Identify the bad low-mass leg
bad_mass_mask = (
    pd.to_numeric(
        df_mass_slope["Dry Mass (µg/m³)"],
        errors="coerce"
    ) < 0.001
)

print("Removing:")
print(
    df_mass_slope.loc[
        bad_mass_mask,
        ["Date", "BCB_start", "BCB_stop", "Dry Mass (µg/m³)"]
    ]
)

# Save the exact leg identifiers
bad_leg_keys = set(
    zip(
        df_mass_slope.loc[bad_mass_mask, "Date"],
        df_mass_slope.loc[bad_mass_mask, "BCB_start"],
        df_mass_slope.loc[bad_mass_mask, "BCB_stop"]
    )
)

# Remove it from mass and slope
df_mass_slope = df_mass_slope.loc[~bad_mass_mask].copy()

# Remove the same leg from concentration
conc_bad_mask = [
    (date, start, stop) in bad_leg_keys
    for date, start, stop in zip(
        df_conc["Date"],
        df_conc["BCB_start"],
        df_conc["BCB_stop"]
    )
]
df_conc = df_conc.loc[~np.array(conc_bad_mask)].copy()
#%%
for df in [df_mass_slope, df_conc]:
    df["Date"] = df["Date"].astype(str)
    df["Month"] = df["Date"].str[5:7].astype(int)
    df["Date_dt"] = pd.to_datetime(df["Date"])
    df.sort_values(["Date_dt", "BCB_start"], kind="mergesort", inplace=True)
    df.reset_index(drop=True, inplace=True)
dfp = df_mass_slope.copy()
x = np.arange(len(dfp))
mass_arr = pd.to_numeric(dfp["Dry Mass (µg/m³)"], errors="coerce").to_numpy(dtype=float)
slope_arr = pd.to_numeric(dfp["Dry Slope (D)"], errors="coerce").to_numpy(dtype=float)
df_conc = df_conc.sort_values(["Date_dt", "BCB_start"], kind="mergesort").reset_index(drop=True)
conc_arr = pd.to_numeric(df_conc["Total_Y_Concentration_cm3"], errors="coerce").to_numpy(dtype=float)
month_name = {
    2: "February",
    3: "March",
    8: "August",
    9: "September"
}
date_first = dfp.groupby(dfp["Date_dt"].dt.date).head(1)
tick_pos = date_first.index.to_numpy()
tick_lab = date_first["Date_dt"].dt.strftime("%m-%d").to_numpy()
fig, axes = plt.subplots(
    3, 1,
    figsize=(18, 9),
    sharex=True
)
plot_info = [
    {
        "ax": axes[0],
        "data": mass_arr,
        "ylabel": "FCDP Mass\n(µg m$^{-3}$)",
        "title": "FCDP mass seasonal trend over the Western Atlantic",
        "log": True,
        "fmt": "{:.2f}"
    },
    {
        "ax": axes[1],
        "data": slope_arr,
        "ylabel": "Slope Parameter D\n(µm)",
        "title": "FCDP slope parameter seasonal trend over the Western Atlantic",
        "log": False,
        "fmt": "{:.2f}"
    },
    {
        "ax": axes[2],
        "data": conc_arr,
        "ylabel": "Total FCDP Concentration\n(cm$^{-3}$)",
        "title": "FCDP total FCDP concentration seasonal trend over the Western Atlantic",
        "log": True,
        "fmt": "{:.2f}"
    }
]
legend_handles = []
for panel_i, info in enumerate(plot_info):
    ax = info["ax"]
    arr = info["data"]
    for m in sorted(dfp["Month"].unique()):
        if m not in month_name:
            continue
        m_mask = dfp["Month"].values == m
        line, = ax.plot(
            x[m_mask],
            arr[m_mask],
            '-o',
            linewidth=1.5,
            markersize=3,
            label=month_name[m]
        )
        c = line.get_color()
        monthly_mean = np.nanmean(arr[m_mask])
        monthly_median = np.nanmedian(arr[m_mask])
        mean_x = x[m_mask][len(x[m_mask]) // 2]
        ax.plot(
            mean_x,
            monthly_median,
            marker="s",
            color="black",
            markersize=9,
            linestyle="None"
        )
        ax.plot(
            mean_x + 1,
            monthly_mean,
            marker="^",
            color="black",
            markersize=9,
            linestyle="None"
        )
        monthly_std = np.nanstd(arr[m_mask])

        ax.errorbar(
            mean_x + 1,
            monthly_mean,
            yerr=monthly_std,
            fmt="none",
            ecolor="black",
            elinewidth=1.2,
            capsize=3
        )

        if panel_i == 0:
            legend_handles.append(
                Line2D([0], [0], color=c, lw=2, label=month_name[m])
            )
    for p in tick_pos:
        ax.axvline(p, color="k", alpha=0.06, linewidth=1)

    if info["log"]:
        ax.set_yscale("log")

    ax.grid(alpha=0.3)
    ax.set_ylabel(info["ylabel"], fontsize=13, fontweight="bold")
    ax.set_title(info["title"], fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", labelsize=11)
axes[-1].set_xlabel("Flight Date", fontsize=15, fontweight="bold")
axes[-1].set_xticks(tick_pos)
axes[-1].set_xticklabels(
    tick_lab,
    rotation=60,
    ha="right",
    fontsize=9,
    fontweight="bold"
)
legend_handles.extend([
    Line2D([0], [0], marker="s", color="black", lw=0,
           markersize=9, label="Monthly median"),
    Line2D([0], [0], marker="^", color="black", lw=0,
           markersize=9, label="Monthly mean"),
    Line2D([0], [0], color="black", lw=1.2,
           label="Monthly mean ± std")
])
fig.legend(
    handles=legend_handles,
    loc="center left",
    bbox_to_anchor=(0.91, 0.5),
    fontsize=11,
    frameon=True
)
fig.subplots_adjust(right=0.88, hspace=0.25, bottom=0.15)
plt.show()
# %%
fcdp_plot_data = {
    "dfp": dfp,
    "x": x,
    "mass_arr": mass_arr,
    "slope_arr": slope_arr,
    "conc_arr": conc_arr,
    "tick_pos": tick_pos,
    "tick_lab": tick_lab,
    "month_name": month_name,
}
with open("FCDP_3panel_seasonaltrend_FMAS2020.pkl", "wb") as f:
    pickle.dump(fcdp_plot_data, f)
print("Saved FCDP 3-panel plot data.")
# %%
with open("CAS_3panel_seasonaltrend_FMAS2020.pkl", "rb") as f:
    cas = pickle.load(f)
with open("CDP_3panel_seasonaltrend_FMAS2020.pkl", "rb") as f:
    cdp = pickle.load(f)
month_name = {
    2: "February",
    3: "March",
    8: "August",
    9: "September"
}
month_colors = {
    2: "tab:blue",
    3: "tab:orange",
    8: "tab:green",
    9: "tab:red"
}
def prep_dataset(d):
    df = d["dfp"].copy()
    df["Date_dt"] = pd.to_datetime(df["Date"])
    df["Date_only"] = df["Date_dt"].dt.date
    sort_cols = ["Date_dt"]
    if "BCB_start" in df.columns:
        sort_cols.append("BCB_start")
    df = df.sort_values(sort_cols).reset_index(drop=True)
    arrays = [
        np.asarray(d["mass_arr"], dtype=float),
        np.asarray(d["slope_arr"], dtype=float),
        np.asarray(d["conc_arr"], dtype=float)
    ]
    n = min(len(df), len(arrays[0]), len(arrays[1]), len(arrays[2]))
    df = df.iloc[:n].copy()
    arrays = [a[:n] for a in arrays]
    return df, arrays
cas_df, cas_arrays = prep_dataset(cas)
cdp_df, cdp_arrays = prep_dataset(cdp)
all_dates = sorted(
    set(cas_df["Date_only"].unique()).union(set(cdp_df["Date_only"].unique()))
)

date_to_x = {d: i for i, d in enumerate(all_dates)}
def make_x_positions(df, side_offset):
    x_pos = np.zeros(len(df), dtype=float)
    for date, group in df.groupby("Date_only", sort=False):
        base_x = date_to_x[date]
        n = len(group)
        if n == 1:
            offsets = np.array([side_offset])
        else:
            offsets = np.linspace(-0.28, 0.28, n) + side_offset

        x_pos[group.index.to_numpy()] = base_x + offsets
    return x_pos
cas_x = make_x_positions(cas_df, side_offset=-0.08)
cdp_x = make_x_positions(cdp_df, side_offset=0.08)
tick_pos = np.arange(len(all_dates))
tick_lab = [pd.Timestamp(d).strftime("%m-%d") for d in all_dates]
ylabels = [
    "GCCN Mass\n(µg m$^{-3}$)",
    "Slope Parameter D\n(µm)",
    "Total Concentration\n(cm$^{-3}$)"
]
titles = [
    "2020 CAS and CDP GCCN mass seasonal trend over the Western Atlantic",
    "2020 CAS and CDP GCCN slope parameter seasonal trend over the Western Atlantic",
    "2020 CAS and CDP total GCCN concentration seasonal trend over the Western Atlantic"
]

logscale = [True, False, True]
fig, axes = plt.subplots(3, 1, figsize=(20, 9), sharex=True)
for panel_i, ax in enumerate(axes):

    cas_arr = cas_arrays[panel_i]
    cdp_arr = cdp_arrays[panel_i]

    for m in sorted(month_name):

        c = month_colors[m]

        cas_mask = cas_df["Month"].values == m
        cdp_mask = cdp_df["Month"].values == m

        if np.any(cas_mask):
            ax.plot(
                cas_x[cas_mask],
                cas_arr[cas_mask],
                "-o",
                color=c,
                linewidth=1.5,
                markersize=3
            )

            mean_x = np.nanmean(cas_x[cas_mask])
            cas_mean = np.nanmean(cas_arr[cas_mask])
            cas_median = np.nanmedian(cas_arr[cas_mask])
            cas_std = np.nanstd(cas_arr[cas_mask])

            ax.plot(mean_x, cas_median, marker="s", color="black",
                    markersize=9, linestyle="None")

            ax.errorbar(mean_x, cas_mean, yerr=cas_std, fmt="none",
                        ecolor="black", elinewidth=1.4, capsize=3)

        if np.any(cdp_mask):
            ax.plot(
                cdp_x[cdp_mask],
                cdp_arr[cdp_mask],
                "--o",
                color=c,
                linewidth=1.5,
                markersize=3
            )
            mean_x = np.nanmean(cdp_x[cdp_mask])
            cdp_mean = np.nanmean(cdp_arr[cdp_mask])
            cdp_median = np.nanmedian(cdp_arr[cdp_mask])
            cdp_std = np.nanstd(cdp_arr[cdp_mask])

            ax.plot(mean_x, cdp_median, marker="^", color="black",
                    markersize=9, linestyle="None")

            ax.errorbar(mean_x, cdp_mean, yerr=cdp_std, fmt="none",
                        ecolor="black", elinewidth=1.4, capsize=3)

    for p in tick_pos:
        ax.axvline(p, color="k", alpha=0.06, linewidth=1)

    if logscale[panel_i]:
        ax.set_yscale("log")

    ax.grid(alpha=0.3)
    ax.set_ylabel(ylabels[panel_i], fontsize=12, fontweight="bold")
    ax.set_title(titles[panel_i], fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", labelsize=11)

axes[-1].set_xlabel("Flight Date", fontsize=15, fontweight="bold")
axes[-1].set_xticks(tick_pos)
axes[-1].set_xticklabels(
    tick_lab,
    rotation=60,
    ha="right",
    fontsize=9,
    fontweight="bold"
)

legend_handles = []

for m in sorted(month_name):
    legend_handles.append(
        Line2D([0], [0], color=month_colors[m], lw=2, label=month_name[m])
    )

legend_handles.extend([
    Line2D([0], [0], color="black", lw=2, linestyle="-", label="CAS (solid)"),
    Line2D([0], [0], color="black", lw=2, linestyle="--", label="CDP (dashed)"),
    Line2D([0], [0], marker="s", color="black", lw=0,
           markersize=9, label="CAS monthly median"),
    Line2D([0], [0], marker="^", color="black", lw=0,
           markersize=9, label="CDP monthly median"),
    Line2D([0], [0], color="black", lw=1.4,
           label="Monthly mean (whisker)")
])

fig.legend(
    handles=legend_handles,
    loc="center left",
    bbox_to_anchor=(0.91, 0.5),
    fontsize=11,
    frameon=True
)

fig.subplots_adjust(right=0.88, hspace=0.35, bottom=0.18)
plt.show()
# %%
#save this combined CAS and CDP 3-panel plot data for future use
combined_plot_data = {
    "cas_df": cas_df,
    "cas_x": cas_x,
    "cas_mass_arr": cas_arrays[0],
    "cas_slope_arr": cas_arrays[1],
    "cas_conc_arr": cas_arrays[2],
    "cdp_df": cdp_df,
    "cdp_x": cdp_x,
    "cdp_mass_arr": cdp_arrays[0],
    "cdp_slope_arr": cdp_arrays[1],
    "cdp_conc_arr": cdp_arrays[2],
    "tick_pos": tick_pos,
    "tick_lab": tick_lab,
    "month_name": month_name,
    "month_colors": month_colors,
}
with open("CAS_CDP_3panel_seasonaltrend_FMAS2020.pkl", "wb") as f:
    pickle.dump(combined_plot_data, f)
print("Saved combined CAS and CDP 3-panel plot data.")
# %%
#CAS, CDP, and FCDP 3-panel plot comparison
with open("CAS_3panel_seasonaltrend_FMAS2020.pkl", "rb") as f:
    cas = pickle.load(f)
with open("CDP_3panel_seasonaltrend_FMAS2020.pkl", "rb") as f:
    cdp = pickle.load(f)
with open("FCDP_3panel_seasonaltrend_FMAS2020.pkl", "rb") as f:
    fcdp = pickle.load(f)
month_name = {
    2: "February",
    3: "March",
    8: "August",
    9: "September"
}
month_colors = {
    2: "tab:blue",
    3: "tab:orange",
    8: "tab:green",
    9: "tab:red"
}
def prep_dataset(d):
    df = d["dfp"].copy()
    df["Date_dt"] = pd.to_datetime(df["Date"])
    df["Date_only"] = df["Date_dt"].dt.date
    sort_cols = ["Date_dt"]
    if "BCB_start" in df.columns:
        sort_cols.append("BCB_start")
    df = df.sort_values(sort_cols).reset_index(drop=True)
    arrays = [
        np.asarray(d["mass_arr"], dtype=float),
        np.asarray(d["slope_arr"], dtype=float),
        np.asarray(d["conc_arr"], dtype=float)
    ]
    n = min(len(df), len(arrays[0]), len(arrays[1]), len(arrays[2]))
    df = df.iloc[:n].copy()
    arrays = [a[:n] for a in arrays]
    return df, arrays
cas_df, cas_arrays = prep_dataset(cas)
cdp_df, cdp_arrays = prep_dataset(cdp)
fcdp_df, fcdp_arrays = prep_dataset(fcdp)
all_dates = sorted(
    set(cas_df["Date_only"].unique()).union(set(cdp_df["Date_only"].unique()))
)

date_to_x = {d: i for i, d in enumerate(all_dates)}
def make_x_positions(df, side_offset):
    x_pos = np.zeros(len(df), dtype=float)
    for date, group in df.groupby("Date_only", sort=False):
        base_x = date_to_x[date]
        n = len(group)
        if n == 1:
            offsets = np.array([side_offset])
        else:
            offsets = np.linspace(-0.28, 0.28, n) + side_offset

        x_pos[group.index.to_numpy()] = base_x + offsets
    return x_pos
cas_x = make_x_positions(cas_df, side_offset=-0.08)
cdp_x = make_x_positions(cdp_df, side_offset=0.08)
fcdp_x = make_x_positions(fcdp_df, side_offset=0.08)
tick_pos = np.arange(len(all_dates))
tick_lab = [pd.Timestamp(d).strftime("%m-%d") for d in all_dates]
ylabels = [
    "GCCN Mass\n(µg m$^{-3}$)",
    "Slope Parameter D\n(µm)",
    "Total Concentration\n(cm$^{-3}$)"
]
titles = [
    "2020 CAS, CDP, and FCDP GCCN mass seasonal trend over the Western Atlantic",
    "2020 CAS, CDP, and FCDP GCCN slope parameter seasonal trend over the Western Atlantic",
    "2020 CAS, CDP, and FCDP total GCCN concentration seasonal trend over the Western Atlantic"
]

logscale = [True, False, True]
fig, axes = plt.subplots(3, 1, figsize=(20, 9), sharex=True)
for panel_i, ax in enumerate(axes):

    cas_arr = cas_arrays[panel_i]
    cdp_arr = cdp_arrays[panel_i]
    fcdp_arr = fcdp_arrays[panel_i]

    for m in sorted(month_name):

        c = month_colors[m]

        cas_mask = cas_df["Month"].values == m
        cdp_mask = cdp_df["Month"].values == m
        fcdp_mask = fcdp_df["Month"].values == m

        if np.any(cas_mask):
            ax.plot(
                cas_x[cas_mask],
                cas_arr[cas_mask],
                "-o",
                color=c,
                linewidth=1.5,
                markersize=3
            )

            mean_x = np.nanmean(cas_x[cas_mask])
            cas_mean = np.nanmean(cas_arr[cas_mask])
            cas_median = np.nanmedian(cas_arr[cas_mask])
            cas_std = np.nanstd(cas_arr[cas_mask])

            ax.plot(mean_x, cas_median, marker="s", color="black",
                    markersize=9, linestyle="None")

            ax.errorbar(mean_x, cas_mean, yerr=cas_std, fmt="none",
                        ecolor="black", elinewidth=1.4, capsize=3)

        if np.any(cdp_mask):
            ax.plot(
                cdp_x[cdp_mask],
                cdp_arr[cdp_mask],
                "--o",
                color=c,
                linewidth=1.5,
                markersize=3
            )
            mean_x = np.nanmean(cdp_x[cdp_mask])
            cdp_mean = np.nanmean(cdp_arr[cdp_mask])
            cdp_median = np.nanmedian(cdp_arr[cdp_mask])
            cdp_std = np.nanstd(cdp_arr[cdp_mask])

            ax.plot(mean_x, cdp_median, marker="^", color="black",
                    markersize=9, linestyle="None")

            ax.errorbar(mean_x, cdp_mean, yerr=cdp_std, fmt="none",
                        ecolor="black", elinewidth=1.4, capsize=3)

        if np.any(fcdp_mask):
            ax.plot(
                fcdp_x[fcdp_mask],
                fcdp_arr[fcdp_mask],
                "-.",
                color=c,
                linewidth=1.5,
                markersize=3
            )
            mean_x = np.nanmean(fcdp_x[fcdp_mask])
            fcdp_mean = np.nanmean(fcdp_arr[fcdp_mask])
            fcdp_median = np.nanmedian(fcdp_arr[fcdp_mask])
            fcdp_std = np.nanstd(fcdp_arr[fcdp_mask])

            ax.plot(mean_x, fcdp_median, marker="d", color="black",
                    markersize=9, linestyle="None")

            ax.errorbar(mean_x, fcdp_mean, yerr=fcdp_std, fmt="none",
                        ecolor="black", elinewidth=1.4, capsize=3)

    for p in tick_pos:
        ax.axvline(p, color="k", alpha=0.06, linewidth=1)

    if logscale[panel_i]:
        ax.set_yscale("log")

    ax.grid(alpha=0.3)
    ax.set_ylabel(ylabels[panel_i], fontsize=12, fontweight="bold")
    ax.set_title(titles[panel_i], fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", labelsize=11)

axes[-1].set_xlabel("Flight Date", fontsize=15, fontweight="bold")
axes[-1].set_xticks(tick_pos)
axes[-1].set_xticklabels(
    tick_lab,
    rotation=60,
    ha="right",
    fontsize=9,
    fontweight="bold"
)

legend_handles = []

for m in sorted(month_name):
    legend_handles.append(
        Line2D([0], [0], color=month_colors[m], lw=2, label=month_name[m])
    )

legend_handles.extend([
    Line2D([0], [0], color="black", lw=2, linestyle="-", label="CAS (solid)"),
    Line2D([0], [0], color="black", lw=2, linestyle="--", label="CDP (dashed)"),
    Line2D([0], [0], color="black", lw=2, linestyle="-.", label="FCDP (dash-dot)"),
    Line2D([0], [0], marker="s", color="black", lw=0,
           markersize=9, label="CAS monthly median"),
    Line2D([0], [0], marker="^", color="black", lw=0,
           markersize=9, label="CDP monthly median"),
    Line2D([0], [0], marker="d", color="black", lw=0,
           markersize=9, label="FCDP monthly median"),
    Line2D([0], [0], color="black", lw=1.4,
           label="Monthly mean (whisker)")
])

fig.legend(
    handles=legend_handles,
    loc="center left",
    bbox_to_anchor=(0.91, 0.5),
    fontsize=11,
    frameon=True
)
fig.subplots_adjust(right=0.88, hspace=0.35, bottom=0.18)
plt.show()
# %%
#save this combined CAS and CDP and FCDP 3-panel plot data for future use
combined_plot_data = {
    "cas_df": cas_df,
    "cas_x": cas_x,
    "cas_mass_arr": cas_arrays[0],
    "cas_slope_arr": cas_arrays[1],
    "cas_conc_arr": cas_arrays[2],
    "cdp_df": cdp_df,
    "cdp_x": cdp_x,
    "cdp_mass_arr": cdp_arrays[0],
    "cdp_slope_arr": cdp_arrays[1],
    "cdp_conc_arr": cdp_arrays[2],
    "tick_pos": tick_pos,
    "tick_lab": tick_lab,
    "month_name": month_name,
    "month_colors": month_colors,
    "fcdp_df": fcdp_df,
    "fcdp_x": fcdp_x,
    "fcdp_mass_arr": fcdp_arrays[0],
    "fcdp_slope_arr": fcdp_arrays[1],
    "fcdp_conc_arr": fcdp_arrays[2],
}
with open("CAS_CDP_FCDP_3panel_seasonaltrend_FMAS2020.pkl", "wb") as f:
    pickle.dump(combined_plot_data, f)
print("Saved combined CAS, CDP, and FCDP 3-panel plot data.")
# %%
