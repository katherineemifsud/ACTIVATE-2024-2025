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
with open("CAS_slope_massLE1002021.pkl", "rb") as f:
    filtered_slope_mass100gone = pickle.load(f)
print("Total number of legs:", len(filtered_slope_mass100gone))
# %%
#monthly trend of dry mass
df_cas = pd.DataFrame(filtered_slope_mass100gone).copy()
df_cas = df_cas[df_cas["Date"].astype(str).str.startswith("2021-")].copy()
df_cas["Month"] = df_cas["Date"].astype(str).str[5:7].astype(int)
df_cas = df_cas[df_cas["Month"].isin([11, 12])].copy()
df_cas_sorted = df_cas.sort_values(
    ["Date", "BCB_start"],
    kind="mergesort"
).reset_index(drop=True)
mass_cas = df_cas_sorted["Dry Mass (µg/m³)"].astype(float).values
x = np.arange(len(df_cas_sorted))
plt.figure(figsize=(12, 4.8))
plt.plot(x, mass_cas, '-')
plt.yscale("log")
plt.grid(alpha=0.3)
plt.xlabel("Leg index (sorted by Date, then BCB_start)", fontsize=13, fontweight="bold")
plt.ylabel("Dry GCCN Mass (µg/m³)", fontsize=13, fontweight="bold")
plt.title("Dry Mass Timeline November-December 2021\nLegs ordered by Date then BCB_start",
          fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
#%%
#monthly mass trend coded with color seperation
month_name = {
    11: "November",
    12: "December",
}

plt.figure(figsize=(12, 4.8))

for m in sorted(df_cas_sorted["Month"].unique()):
    if m not in month_name:
        continue

    m_mask = (df_cas_sorted["Month"].values == m)

    plt.plot(
        x[m_mask],
        mass_cas[m_mask],
        '-o',
        label=month_name[m],
        markersize=6
    )

plt.yscale("log")
plt.grid(alpha=0.3)
plt.ylabel("GCCN Mass (µg/m³)", fontsize=16, fontweight="bold")
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.title("CAS BCB Mass November-December 2021", fontsize=18, fontweight="bold")
plt.legend(ncol=2, fontsize=10)
plt.yticks(fontsize=14, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
#%%
#monthly mean mass trend coded with color seperation
plt.figure(figsize=(12, 4.8))
for m in sorted(df_cas_sorted["Month"].unique()): 
    if m not in month_name:
        continue
    m_mask = (df_cas_sorted["Month"].values == m)
    mean_mass = np.mean(mass_cas[m_mask])
    plt.plot(
        x[m_mask],
        mass_cas[m_mask],
        '-',
        label=f"{month_name[m]} (mean: {mean_mass:.2f} µg/m³)"
    )
plt.yscale("log")
plt.grid(alpha=0.3)
plt.ylabel("GCCN Mass (µg/m³)", fontsize=16, fontweight="bold")
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.title("CAS BCB Mass November-December 2021\nMonthly Means", fontsize=18, fontweight="bold")
plt.legend(ncol=2, fontsize=10)
plt.yticks(fontsize=14, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
#%%
#monthly median trend coded with color seperation
plt.figure(figsize=(12, 4.8))
for m in sorted(df_cas_sorted["Month"].unique()): 
    if m not in month_name:
        continue
    m_mask = (df_cas_sorted["Month"].values == m)
    median_mass = np.median(mass_cas[m_mask])
    plt.plot(
        x[m_mask],
        mass_cas[m_mask],
        '-',
        label=f"{month_name[m]} (median: {median_mass:.2f} µg/m³)"
    )
plt.yscale("log")
plt.grid(alpha=0.3)
plt.ylabel("GCCN Mass (µg/m³)", fontsize=16, fontweight="bold")
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.title("CAS BCB Mass November-December 2021\nMonthly Medians", fontsize=18, fontweight="bold")
plt.legend(ncol=2, fontsize=10)
plt.yticks(fontsize=14, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
#%%
dfp_cas = df_cas_sorted.reset_index(drop=True).copy()
dfp_cas["Date_dt"] = pd.to_datetime(dfp_cas["Date"])
x = np.arange(len(dfp_cas))
tick_pos = dfp_cas.groupby(dfp_cas["Date_dt"].dt.date).head(1).index.to_numpy()
tick_lab = dfp_cas.loc[tick_pos, "Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
#%%
#monthly mean mass trend coded with color seperation and date x-axis
dfp_cas = df_cas_sorted.copy()
dfp_cas["Date_dt"] = pd.to_datetime(dfp_cas["Date"])
sort_cols = ["Date_dt"]
if "BCB_start" in dfp_cas.columns:
    sort_cols.append("BCB_start")
dfp_cas = dfp_cas.sort_values(sort_cols).reset_index(drop=True)
mass_arr = dfp_cas["Dry Mass (µg/m³)"].astype(float).values
x = np.arange(len(dfp_cas))
date_first = dfp_cas.groupby(dfp_cas["Date_dt"].dt.date).head(1)
tick_pos = date_first.index.to_numpy()
tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
fig_w = max(14, 0.55 * len(tick_pos))
fig, ax = plt.subplots(figsize=(fig_w, 6.2))
for m in sorted(dfp_cas["Month"].unique()):
    if m not in month_name:
        continue
    m_mask = dfp_cas["Month"].values == m
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
ax.set_title("CAS BCB Mass November-December 2021\nMonthly Means",
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
dfp_cas = df_cas_sorted.copy()
dfp_cas["Date_dt"] = pd.to_datetime(dfp_cas["Date"])
sort_cols = ["Date_dt"]
if "BCB_start" in dfp_cas.columns:
    sort_cols.append("BCB_start")
dfp_cas = dfp_cas.sort_values(sort_cols).reset_index(drop=True)
x = np.arange(len(dfp_cas))
mass_arr = np.asarray(dfp_cas["Dry Mass (µg/m³)"].astype(float))
date_first = dfp_cas.groupby(dfp_cas["Date_dt"].dt.date).head(1)
tick_pos = date_first.index.to_numpy()
tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
fig_w = max(18, 0.9 * len(tick_pos))
fig, ax = plt.subplots(figsize=(fig_w, 6.2))
legend_handles = []

for m in sorted(dfp_cas["Month"].unique()):
    if m not in month_name:
        continue

    m_mask = (dfp_cas["Month"].values == m)
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
ax.set_title("CAS BCB Mass November-December 2021",
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
    frameon=True)
fig.subplots_adjust(bottom=0.40)
fig.tight_layout()
plt.show()
#%%
#monthly trend of Dry Slope (D)
df_cas = pd.DataFrame(filtered_slope_mass100gone).copy()

df_cas = df_cas[df_cas["Date"].astype(str).str.startswith("2021-")].copy()
df_cas["Month"] = df_cas["Date"].astype(str).str[5:7].astype(int)
df_cas = df_cas[df_cas["Month"].isin([11, 12])].copy()
df_sorted_cas = df_cas.sort_values(
    ["Date", "BCB_start"],
    kind="mergesort"
).reset_index(drop=True)
slope_cas = df_sorted_cas["Dry Slope (D)"].astype(float).values
x = np.arange(len(df_sorted_cas))
plt.figure(figsize=(12, 4.8))
plt.plot(x, slope_cas, '-o', markersize=4)
plt.grid(alpha=0.3)
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.ylabel("Slope (D)", fontsize=16, fontweight="bold")
plt.title("CAS BCB Slope November-December 2021",
          fontsize=18, fontweight="bold")
plt.tight_layout()
plt.show()
# %%
#monthly Dry Slope (D) trend coded with color seperation
plt.figure(figsize=(12, 4.8))
for m in sorted(df_sorted_cas["Month"].unique()):
    if m not in month_name:
        continue
    m_mask = (df_sorted_cas["Month"].values == m)
    plt.plot(
        x[m_mask],
        slope_cas[m_mask],
        '-',
        label=f"{month_name[m]}"
    )
plt.grid(alpha=0.3)
plt.ylabel("Slope (D)", fontsize=16, fontweight="bold")
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.title("CAS BCB Slope (D) November-December 2021",
          fontsize=18, fontweight="bold")
plt.legend(ncol=2, fontsize=10)
plt.yticks(fontsize=14, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
# %%
#monthly mean Dry Slope (D) trend coded with color seperation
plt.figure(figsize=(12, 4.8))
for m in sorted(df_sorted_cas["Month"].unique()): 
    if m not in month_name:
        continue
    m_mask = (df_sorted_cas["Month"].values == m)
    mean_slope = np.mean(slope_cas[m_mask])
    plt.plot(
        x[m_mask],
        slope_cas[m_mask],
        '-',
        label=f"{month_name[m]} (mean: {mean_slope:.2f})"
    )
plt.yscale("log")
plt.grid(alpha=0.3)
plt.ylabel("Dry GCCN Mass (µg/m³)", fontsize=16, fontweight="bold")
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.title("CAS BCB Dry Slope (D) Novemeber-December\nMonthly Means", fontsize=18, fontweight="bold")
plt.legend(ncol=2, fontsize=10)
plt.yticks(fontsize=14, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
# %%
#monthly median Dry Slope (D) trend coded with color seperation
plt.figure(figsize=(12, 4.8))
for m in sorted(df_sorted_cas["Month"].unique()): 
    if m not in month_name:
        continue
    m_mask = (df_sorted_cas["Month"].values == m)
    median_slope = np.median(slope_cas[m_mask])
    plt.plot(
        x[m_mask],
        slope_cas[m_mask],
        '-',
        label=f"{month_name[m]} (median: {median_slope:.2f})"
    )
plt.yscale("log")
plt.grid(alpha=0.3)
plt.ylabel("Dry Slope (D)", fontsize=16, fontweight="bold")
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.title("CAS BCB Dry Slope (D) November-December 2021\nMonthly Medians", fontsize=18, fontweight="bold")
plt.legend(ncol=2, fontsize=10)
plt.yticks(fontsize=14, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
# %%
dfp_cas = df_sorted_cas.reset_index(drop=True).copy()
dfp_cas["Date_dt"] = pd.to_datetime(dfp_cas["Date"])
x = np.arange(len(dfp_cas))
tick_pos = dfp_cas.groupby(dfp_cas["Date_dt"].dt.date).head(1).index.to_numpy()
tick_lab = dfp_cas.loc[tick_pos, "Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
#%%
#monthly mean slope trend coded with color seperation and date x-axis
dfp_cas = df_sorted_cas.copy()
dfp_cas["Date_dt"] = pd.to_datetime(dfp_cas["Date"])
sort_cols = ["Date_dt"]
if "BCB_start" in dfp_cas.columns:
    sort_cols.append("BCB_start")

dfp_cas = dfp_cas.sort_values(sort_cols).reset_index(drop=True)
x = np.arange(len(dfp_cas))
slope_arr = np.asarray(slope_cas)
date_first = dfp_cas.groupby(dfp_cas["Date_dt"].dt.date).head(1)
tick_pos = date_first.index.to_numpy()
tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
fig_w = max(22, 0.55 * len(tick_pos)) 
fig, ax = plt.subplots(figsize=(fig_w, 6.2))

for m in sorted(dfp_cas["Month"].unique()):
    if m not in month_name:
        continue
    m_mask = (dfp_cas["Month"].values == m)
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
ax.set_title("CAS BCB Slope (D) November-December 2021\nMonthly Trend", fontsize=18, fontweight="bold")
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
dfp_cas = df_sorted_cas.copy()
dfp_cas["Date_dt"] = pd.to_datetime(dfp_cas["Date"])
sort_cols = ["Date_dt"]
if "BCB_start" in dfp_cas.columns:
    sort_cols.append("BCB_start")
dfp_cas = dfp_cas.sort_values(sort_cols).reset_index(drop=True)
x = np.arange(len(dfp_cas))
slope_arr = np.asarray(slope_cas)
date_first = dfp_cas.groupby(dfp_cas["Date_dt"].dt.date).head(1)
tick_pos = date_first.index.to_numpy()
tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
fig_w = max(22, 0.55 * len(tick_pos))
fig, ax = plt.subplots(figsize=(fig_w, 6.2))
legend_handles = []

for m in sorted(dfp_cas["Month"].unique()):
    if m not in month_name:
        continue

    m_mask = (dfp_cas["Month"].values == m)
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
ax.set_title("CAS BCB Slope November-December 2021\nMonthly Trend",
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
    frameon=True)
fig.subplots_adjust(bottom=0.40)
fig.tight_layout()
plt.show()
# %%
dfp_cas = df_sorted_cas.copy()
dfp_cas["Date_dt"] = pd.to_datetime(dfp_cas["Date"])
sort_cols = ["Date_dt"]
if "BCB_start" in dfp_cas.columns:
    sort_cols.append("BCB_start")
dfp_cas = dfp_cas.sort_values(sort_cols).reset_index(drop=True)
x = np.arange(len(dfp_cas))
slope_arr = pd.to_numeric(dfp_cas["Dry Slope (D)"], errors="coerce").to_numpy(dtype=float)
date_first = dfp_cas.groupby(dfp_cas["Date_dt"].dt.date).head(1)
tick_pos = date_first.index.to_numpy()
tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
fig_w = max(18, 0.9 * len(tick_pos))
fig, ax = plt.subplots(figsize=(fig_w, 6.2))
legend_handles = []
for m in sorted(dfp_cas["Month"].unique()):
    if m not in month_name:
        continue
    m_mask = dfp_cas["Month"].values == m
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
ax.set_title("CAS BCB Slope November-December 2021",
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
with open("CAS_concentration_massLE1002021.pkl", "rb") as f:
    total_concentration_cm3_mass100gone = pickle.load(f)
print("Total number of legs:", len(total_concentration_cm3_mass100gone))

# %%
# CAS concentration monthly trend

dfp_cas = pd.DataFrame(total_concentration_cm3_mass100gone).copy()
dfp_cas = dfp_cas[dfp_cas["Date"].astype(str).str.startswith("2021-")].copy()
dfp_cas["Month"] = dfp_cas["Date"].astype(str).str[5:7].astype(int)
dfp_cas = dfp_cas[dfp_cas["Month"].isin([11, 12])].copy()
dfp_cas["Date_dt"] = pd.to_datetime(dfp_cas["Date"])
sort_cols = ["Date_dt"]
if "BCB_start" in dfp_cas.columns:
    sort_cols.append("BCB_start")
dfp_cas = dfp_cas.sort_values(sort_cols).reset_index(drop=True)
x = np.arange(len(dfp_cas))
conc_arr = pd.to_numeric(
    dfp_cas["Total_Y_Concentration_cm3"],
    errors="coerce"
).to_numpy(dtype=float)
date_first = dfp_cas.groupby(dfp_cas["Date_dt"].dt.date).head(1)
tick_pos = date_first.index.to_numpy()
tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
fig_w = max(18, 0.9 * len(tick_pos))
fig, ax = plt.subplots(figsize=(fig_w, 6.2))
legend_handles = []
for m in sorted(dfp_cas["Month"].unique()):
    if m not in month_name:
        continue
    m_mask = dfp_cas["Month"].values == m
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
ax.set_ylabel("GCCN Concentration (cm⁻³)", fontsize=20, fontweight="bold")
ax.set_xlabel("Flight Date", fontsize=20, fontweight="bold")
ax.set_title("CAS BCB Concentration November-December 2021", fontsize=20, fontweight="bold")
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
    11: "November",
    12: "December"
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
        "ylabel": "GCCN Mass\n(µg m$^{-3}$)",
        "title": "CAS GCCN mass seasonal trend over the Western Atlantic",
        "log": True,
        "fmt": "{:.2f}"
    },
    {
        "ax": axes[1],
        "data": slope_arr,
        "ylabel": "Slope Parameter D\n(µm)",
        "title": "CAS GCCN slope parameter seasonal trend over the Western Atlantic",
        "log": False,
        "fmt": "{:.2f}"
    },
    {
        "ax": axes[2],
        "data": conc_arr,
        "ylabel": "Total GCCN Concentration\n(cm$^{-3}$)",
        "title": "CAS total GCCN concentration seasonal trend over the Western Atlantic",
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
    fontweight="bold")
legend_handles.extend([
    Line2D([0], [0], marker="s", color="black", lw=0,
           markersize=9, label="Monthly median"),
    Line2D([0], [0], marker="^", color="black", lw=0,
           markersize=9, label="Monthly mean"),
    Line2D([0], [0], color="black", lw=1.2,
           label="Monthly mean ± std")])
fig.legend(
    handles=legend_handles,
    loc="center left",
    bbox_to_anchor=(0.91, 0.5),
    fontsize=11,
    frameon=True)
fig.subplots_adjust(right=0.88, hspace=0.25, bottom=0.15)
plt.show()
# %%
cas_plot_data = {
    "dfp": dfp,
    "x": x,
    "mass_arr": mass_arr,
    "slope_arr": slope_arr,
    "conc_arr": conc_arr,
    "tick_pos": tick_pos,
    "tick_lab": tick_lab,
    "month_name": month_name,
}
with open("CAS_3panel_seasonaltrend_2021.pkl", "wb") as f:
    pickle.dump(cas_plot_data, f)
print("Saved CAS 3-panel plot data.")
# %%