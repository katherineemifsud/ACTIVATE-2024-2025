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
gccn_path = "/home/disk/eos4/kathem24/activate/data/CDP/2022/csv/total_Y_concentration_cm3_CDP.csv"
df_gccn_CDP = pd.read_csv(gccn_path)
print("Total number of legs:", len(df_gccn_CDP))
# %%
#monthly trend of gccn concentration
df_CDP = df_gccn_CDP.copy()
df_CDP = df_CDP[df_CDP["Date"].str.startswith("2022-")].copy()
df_CDP["Month"] = df_CDP["Date"].str[5:7].astype(int)
df_CDP = df_CDP[df_CDP["Month"].between(1, 6)]
df_sorted_CDP = df_CDP.sort_values(["Date", "BCB_start"], kind="mergesort").reset_index(drop=True)
gccn_concentration_CDP = df_sorted_CDP['Total_Y_Concentration_cm3'].astype(float).values
x = np.arange(len(df_sorted_CDP))
plt.figure(figsize=(12, 4.8))
plt.plot(x, gccn_concentration_CDP, '-')
plt.yscale("log")
plt.grid(alpha=0.3)
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.ylabel("Total GCCN Concentration (cm⁻³)", fontsize=16, fontweight="bold")
plt.title("CDP BCB Total GCCN Concentration\nJanuary–June 2022", fontsize=18, fontweight="bold")
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
for m in sorted(df_sorted_CDP["Month"].unique()):
    if m not in month_name:
        continue
    m_mask = (df_sorted_CDP["Month"].values == m)
    plt.plot(
        x[m_mask],
        gccn_concentration_CDP[m_mask],
        '-',
        label=f"{month_name[m]}"
    )
plt.yscale("log")  
plt.grid(alpha=0.3)
plt.ylabel("Total GCCN Concentration (cm⁻³)", fontsize=16, fontweight="bold")
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.title("CDP BCB Total GCCN Concentration (January–June 2022)",
          fontsize=18, fontweight="bold")
plt.legend(ncol=2, fontsize=10)
plt.yticks(fontsize=14, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
# %%
#monthly mean gccn concentration trend coded with color seperation
plt.figure(figsize=(12, 4.8))
for m in sorted(df_sorted_CDP["Month"].unique()): 
    if m not in month_name:
        continue
    m_mask = (df_sorted_CDP["Month"].values == m)
    mean_concentration = np.mean(gccn_concentration_CDP[m_mask])
    plt.plot(
        x[m_mask],
        gccn_concentration_CDP[m_mask],
        '-',
        label=f"{month_name[m]} (mean: {mean_concentration:.2f} cm⁻³)"
    )
plt.yscale("log")
plt.grid(alpha=0.3)
plt.ylabel("Total GCCN Concentration (cm⁻³)", fontsize=16, fontweight="bold")
plt.xlabel("Leg index", fontsize=16, fontweight="bold")
plt.title("CDP BCB Total GCCN Concentration January–June 2022\nMonthly Means", fontsize=18, fontweight="bold")
plt.legend(ncol=2, fontsize=10)
plt.yticks(fontsize=14, fontweight="bold")
plt.xticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
# %%
dfp_CDP = df_sorted_CDP.reset_index(drop=True).copy()
dfp_CDP["Date_dt"] = pd.to_datetime(dfp_CDP["Date"])
x = np.arange(len(dfp_CDP))
tick_pos = dfp_CDP.groupby(dfp_CDP["Date_dt"].dt.date).head(1).index.to_numpy()
tick_lab = dfp_CDP.loc[tick_pos, "Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
#%%
#monthly mean gccn concentration trend coded with color seperation and date x-axis
dfp_CDP = df_sorted_CDP.copy()
dfp_CDP["Date_dt"] = pd.to_datetime(dfp_CDP["Date"])
sort_cols = ["Date_dt"]
if "Min_start" in dfp_CDP.columns:
    sort_cols.append("Min_start")
dfp_CDP = dfp_CDP.sort_values(sort_cols).reset_index(drop=True)
x = np.arange(len(dfp_CDP))
gccn_arr = np.asarray(gccn_concentration_CDP)
date_first = dfp_CDP.groupby(dfp_CDP["Date_dt"].dt.date).head(1)
tick_pos = date_first.index.to_numpy()
tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
fig_w = max(22, 0.55 * len(tick_pos)) 
fig, ax = plt.subplots(figsize=(fig_w, 6.2))
for m in sorted(dfp_CDP["Month"].unique()):
    if m not in month_name:
        continue
    m_mask = (dfp_CDP["Month"].values == m)
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
ax.set_title("CDP BCB Total GCCN Concentration January–June 2022\nMonthly Means", fontsize=18, fontweight="bold")
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
dfp_CDP = df_sorted_CDP.copy()
dfp_CDP["Date_dt"] = pd.to_datetime(dfp_CDP["Date"])
sort_cols = ["Date_dt"]
if "Min_start" in dfp_CDP.columns:
    sort_cols.append("Min_start")
dfp_CDP = dfp_CDP.sort_values(sort_cols).reset_index(drop=True)
x = np.arange(len(dfp_CDP))
gccn_arr = np.asarray(gccn_concentration_CDP)
date_first = dfp_CDP.groupby(dfp_CDP["Date_dt"].dt.date).head(1)
tick_pos = date_first.index.to_numpy()
tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
fig_w = max(22, 0.55 * len(tick_pos))
fig, ax = plt.subplots(figsize=(fig_w, 6.2))
legend_handles = []

for m in sorted(dfp_CDP["Month"].unique()):
    if m not in month_name:
        continue

    m_mask = (dfp_CDP["Month"].values == m)
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
ax.set_title("CDP BCB  January–June 2022\nMonthly Trend",
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
#combining all the code into one plot for both CAS and CDP BCB gccn monthly trend plots
month_colors = {
    1: "#0072B2",  # Jan - blue
    2: "#E69F00",  # Feb - orange
    3: "#009E73",  # Mar - bluish green
    5: "#CC79A7",  # May - purple/magenta
    6: "#56B4E9",  # Jun - sky blue
}
def prep_df(dfp):
    dfp = dfp.copy()
    dfp["Date_dt"] = pd.to_datetime(dfp["Date"])
    sort_cols = ["Date_dt"]
    if "Min_start" in dfp.columns:
        sort_cols.append("Min_start")
    dfp = dfp.sort_values(sort_cols).reset_index(drop=True)
    x = np.arange(len(dfp))
    date_first = dfp.groupby(dfp["Date_dt"].dt.date).head(1)
    tick_pos = date_first.index.to_numpy()
    tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
    return dfp, x, tick_pos, tick_lab
def month_stats(arr, mask, p_lo=10, p_hi=90):
    vals = np.asarray(arr)[mask]
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    mean = np.mean(vals)
    p10  = np.percentile(vals, p_lo)
    p90  = np.percentile(vals, p_hi)
    return mean, p10, p90   
p_lo, p_hi = 10, 90
dfp_cas, x_cas, tick_pos, tick_lab = prep_df(df_sorted)
dfp_cdp, x_cdp, _, _ = prep_df(df_sorted_CDP)

cas_arr = np.asarray(gccn_concentration, dtype=float)
cdp_arr = np.asarray(gccn_concentration_CDP, dtype=float)
fig_w = max(22, 0.55 * len(tick_pos))
fig, ax = plt.subplots(figsize=(fig_w, 6.2))
legend_handles = []
for m in sorted(dfp_cas["Month"].unique()):
    if m not in month_name:
        continue
    mask = (dfp_cas["Month"].values == m)
    c = month_colors.get(m, None)
    ax.plot(x_cas[mask], cas_arr[mask], "-", lw=1.5, color=c, alpha=0.9)
for m in sorted(dfp_cdp["Month"].unique()):
    if m not in month_name:
        continue
    mask = (dfp_cdp["Month"].values == m)
    c = month_colors.get(m, None)
    ax.plot(x_cdp[mask], cdp_arr[mask], "--", lw=1.5, color=c, alpha=0.9)
for m in sorted(set(dfp_cas["Month"].unique()).union(set(dfp_cdp["Month"].unique()))):
    if m not in month_name:
        continue
    c = month_colors.get(m, "k")
    legend_handles.append(Line2D([0], [0], color=c, lw=2, label=month_name[m]))
legend_handles.extend([
    Line2D([0], [0], color="k", lw=2, linestyle="-",  label="CAS legs"),
    Line2D([0], [0], color="k", lw=2, linestyle="--", label="CDP legs"),
])
months_all = sorted(set(dfp_cas["Month"].unique()).union(set(dfp_cdp["Month"].unique())))
for m in months_all:
    if m not in month_name:
        continue
    cas_mask = (dfp_cas["Month"].values == m)
    cas_mean, cas_p25, cas_p75 = month_stats(cas_arr, cas_mask, p_lo, p_hi)
    cdp_mask = (dfp_cdp["Month"].values == m)
    cdp_mean, cdp_p25, cdp_p75 = month_stats(cdp_arr, cdp_mask, p_lo, p_hi)
    if np.any(cas_mask) and np.isfinite(cas_mean):
        cas_x = x_cas[cas_mask][len(x_cas[cas_mask]) // 2]
        ax.errorbar(
            cas_x, cas_mean,
            yerr=[[cas_mean - cas_p25], [cas_p75 - cas_mean]],
            fmt="s", color="k", ecolor="k",
            markersize=9, markeredgewidth=1.8,
            elinewidth=2.2, capsize=5, linestyle="None", zorder=6
        )

    if np.any(cdp_mask) and np.isfinite(cdp_mean):
        cdp_x = x_cdp[cdp_mask][len(x_cdp[cdp_mask]) // 2]
        ax.errorbar(
            cdp_x, cdp_mean,
            yerr=[[cdp_mean - cdp_p25], [cdp_p75 - cdp_mean]],
            fmt="^", color="k", ecolor="k",
            markersize=10, markeredgewidth=1.8,
            elinewidth=2.2, capsize=5, linestyle="None", zorder=6
        )
legend_handles.extend([
    Line2D([0], [0], marker="s", color="k", lw=0, markersize=9,
           label=f"CAS monthly median ± P{p_lo}–P{p_hi}"),
    Line2D([0], [0], marker="^", color="k", lw=0, markersize=10,
           label=f"CDP monthly median ± P{p_lo}–P{p_hi}"),
])
for p in tick_pos:
    ax.axvline(p, color="k", alpha=0.06, linewidth=1)
ax.set_yscale("log")
plt.yticks(fontsize=16, fontweight="bold")
ax.grid(alpha=0.3)
ax.set_ylabel("Total Concentration (cm⁻³)", fontsize=20, fontweight="bold")
ax.set_xlabel("Flight Date", fontsize=20, fontweight="bold")
ax.set_title("BCB January–June 2022\nMonthly Trend",
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
#print out the monthly mean values for both CAS and CDP
for m in months_all:
    if m not in month_name:
        continue
    cas_mask = (dfp_cas["Month"].values == m)
    cas_mean, _, _ = month_stats(cas_arr, cas_mask, p_lo, p_hi)
    cdp_mask = (dfp_cdp["Month"].values == m)
    cdp_mean, _, _ = month_stats(cdp_arr, cdp_mask, p_lo, p_hi)
    print(f"{month_name[m]} - CAS mean: {cas_mean:.2f} cm⁻³, CDP mean: {cdp_mean:.2f} cm⁻³")

# %%
month_colors = {
    1: "#0072B2",  # Jan - blue
    2: "#E69F00",  # Feb - orange
    3: "#009E73",  # Mar - bluish green
    5: "#CC79A7",  # May - purple/magenta
    6: "#56B4E9",  # Jun - sky blue
}
#%%
def prep_df(dfp):
    dfp = dfp.copy()
    dfp["Date_dt"] = pd.to_datetime(dfp["Date"])
    sort_cols = ["Date_dt"]
    if "Min_start" in dfp.columns:
        sort_cols.append("Min_start")
    dfp = dfp.sort_values(sort_cols).reset_index(drop=True)
    x = np.arange(len(dfp))
    date_first = dfp.groupby(dfp["Date_dt"].dt.date).head(1)
    tick_pos = date_first.index.to_numpy()
    tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
    return dfp, x, tick_pos, tick_lab
def month_stats_median(arr, mask, p_lo=25, p_hi=75):
    vals = np.asarray(arr)[mask]
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    med = np.median(vals)
    p10 = np.percentile(vals, p_lo)
    p90 = np.percentile(vals, p_hi)
    return med, p10, p90
  
p_lo, p_hi = 10, 90
dfp_cas, x_cas, tick_pos, tick_lab = prep_df(df_sorted)
dfp_cdp, x_cdp, _, _ = prep_df(df_sorted_CDP)

cas_arr = np.asarray(gccn_concentration, dtype=float)
cdp_arr = np.asarray(gccn_concentration_CDP, dtype=float)
fig_w = max(22, 0.55 * len(tick_pos))
fig, ax = plt.subplots(figsize=(fig_w, 6.2))
legend_handles = []
for m in sorted(dfp_cas["Month"].unique()):
    if m not in month_name:
        continue
    mask = (dfp_cas["Month"].values == m)
    c = month_colors.get(m, None)
    ax.plot(x_cas[mask], cas_arr[mask], "-", lw=1.5, color=c, alpha=0.9)
for m in sorted(dfp_cdp["Month"].unique()):
    if m not in month_name:
        continue
    mask = (dfp_cdp["Month"].values == m)
    c = month_colors.get(m, None)
    ax.plot(x_cdp[mask], cdp_arr[mask], "--", lw=1.5, color=c, alpha=0.9)
for m in sorted(set(dfp_cas["Month"].unique()).union(set(dfp_cdp["Month"].unique()))):
    if m not in month_name:
        continue
    c = month_colors.get(m, "k")
    legend_handles.append(Line2D([0], [0], color=c, lw=2, label=month_name[m]))
legend_handles.extend([
    Line2D([0], [0], color="k", lw=2, linestyle="-",  label="CAS legs"),
    Line2D([0], [0], color="k", lw=2, linestyle="--", label="CDP legs"),
])
months_all = sorted(set(dfp_cas["Month"].unique()).union(set(dfp_cdp["Month"].unique())))

for m in months_all:
    if m not in month_name:
        continue

    cas_mask = (dfp_cas["Month"].values == m)
    cdp_mask = (dfp_cdp["Month"].values == m)
    cas_med, cas_p10, cas_p90 = month_stats_median(cas_arr, cas_mask, p_lo, p_hi)
    cdp_med, cdp_p10, cdp_p90 = month_stats_median(cdp_arr, cdp_mask, p_lo, p_hi)
    if np.any(cas_mask) and np.isfinite(cas_med):
        cas_x = x_cas[cas_mask][len(x_cas[cas_mask]) // 2]
        ax.errorbar(
            cas_x, cas_med,
            yerr=[[cas_med - cas_p10], [cas_p90 - cas_med]],
            fmt="s", color="k", ecolor="k",
            markersize=9, markeredgewidth=1.8,
            elinewidth=2.2, capsize=5, linestyle="None", zorder=6
        )

    if np.any(cdp_mask) and np.isfinite(cdp_med):
        cdp_x = x_cdp[cdp_mask][len(x_cdp[cdp_mask]) // 2]
        ax.errorbar(
            cdp_x, cdp_med,
            yerr=[[cdp_med - cdp_p10], [cdp_p90 - cdp_med]],
            fmt="^", color="k", ecolor="k",
            markersize=10, markeredgewidth=1.8,
            elinewidth=2.2, capsize=5, linestyle="None", zorder=6
        )

legend_handles.extend([
    Line2D([0], [0], marker="s", color="k", lw=0, markersize=9,
           label=f"CAS monthly median ± P{p_lo}–P{p_hi}"),
    Line2D([0], [0], marker="^", color="k", lw=0, markersize=10,
           label=f"CDP monthly median ± P{p_lo}–P{p_hi}"),
])
for p in tick_pos:
    ax.axvline(p, color="k", alpha=0.06, linewidth=1)
ax.set_yscale("log")
plt.yticks(fontsize=16, fontweight="bold")
ax.grid(alpha=0.3)
ax.set_ylabel("Total Concentration (cm⁻³)", fontsize=20, fontweight="bold")
ax.set_xlabel("Flight Date", fontsize=20, fontweight="bold")
ax.set_title("BCB January–June 2022\nMonthly Trend",
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
#print out the monthly mean values for both CAS and CDP
for m in months_all:
    if m not in month_name:
        continue
    cas_mask = (dfp_cas["Month"].values == m)
    cas_median, _, _ = month_stats_median(cas_arr, cas_mask, p_lo, p_hi)
    cdp_mask = (dfp_cdp["Month"].values == m)
    cdp_median, _, _ = month_stats_median(cdp_arr, cdp_mask, p_lo, p_hi)
    print(f"{month_name[m]} - CAS median: {cas_median:.2f} cm⁻³, CDP median: {cdp_median:.2f} cm⁻³")


# %%
#plotting gccn with windspeed. 

p_lo, p_hi = 10, 90
def add_month_col(dfp):
    dfp = dfp.copy()
    dfp["Date_dt"] = pd.to_datetime(dfp["Date"])
    if "Month" not in dfp.columns:
        dfp["Month"] = dfp["Date_dt"].dt.month
    return dfp

def stats_mean_and_prc(values, p_lo=10, p_hi=90):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    mean = np.nanmean(v)
    plo = np.nanpercentile(v, p_lo)
    phi = np.nanpercentile(v, p_hi)
    return mean, plo, phi

def month_summary(dfp, y, month_col="Month", p_lo=10, p_hi=90):
    out = {}
    months = sorted(pd.unique(dfp[month_col]))
    for m in months:
        mask = (dfp[month_col].values == m)
        mean, plo, phi = stats_mean_and_prc(np.asarray(y)[mask], p_lo, p_hi)
        out[m] = (mean, plo, phi)
    return out
dfp_cas = add_month_col(df_sorted)
dfp_cdp = add_month_col(df_sorted_CDP)
dfp_wind = add_month_col(df_wind_CDP)

cas_arr = np.asarray(gccn_concentration, dtype=float)
cdp_arr = np.asarray(gccn_concentration_CDP, dtype=float)
wind_arr = pd.to_numeric(dfp_wind["Windspeed"], errors="coerce").to_numpy()
cas_sum  = month_summary(dfp_cas,  cas_arr,  "Month", p_lo, p_hi)
cdp_sum  = month_summary(dfp_cdp,  cdp_arr,  "Month", p_lo, p_hi)
wind_sum = month_summary(dfp_wind, wind_arr, "Month", p_lo, p_hi)
months_all = sorted(set(cas_sum.keys()).union(cdp_sum.keys()).intersection(wind_sum.keys()))
x_mean, x_lo, x_hi = [], [], []
ycas_mean, ycas_lo, ycas_hi = [], [], []
ycdp_mean, ycdp_lo, ycdp_hi = [], [], []
months_plot = []
for m in months_all:
    if m not in month_name: 
        continue
    w_mean, w_p10, w_p90 = wind_sum[m]
    cas_mean, cas_p10, cas_p90 = cas_sum.get(m, (np.nan, np.nan, np.nan))
    cdp_mean, cdp_p10, cdp_p90 = cdp_sum.get(m, (np.nan, np.nan, np.nan))
    if not np.isfinite(w_mean):
        continue
    if not (np.isfinite(cas_mean) or np.isfinite(cdp_mean)):
        continue

    months_plot.append(m)
    x_mean.append(w_mean); x_lo.append(w_mean - w_p10); x_hi.append(w_p90 - w_mean) 
    ycas_mean.append(cas_mean); ycas_lo.append(cas_mean - cas_p10); ycas_hi.append(cas_p90 - cas_mean)
    ycdp_mean.append(cdp_mean); ycdp_lo.append(cdp_mean - cdp_p10); ycdp_hi.append(cdp_p90 - cdp_mean)

x_mean = np.array(x_mean); xerr = np.vstack([x_lo, x_hi])
ycas_mean = np.array(ycas_mean); ycas_err = np.vstack([ycas_lo, ycas_hi])
ycdp_mean = np.array(ycdp_mean); ycdp_err = np.vstack([ycdp_lo, ycdp_hi])
fig, ax = plt.subplots(figsize=(7.5, 5.8))
for i, m in enumerate(months_plot):
    c = month_colors.get(m, "k")
    if np.isfinite(ycas_mean[i]):
        ax.errorbar(
            x_mean[i], ycas_mean[i],
            xerr=xerr[:, i].reshape(2,1),
            yerr=ycas_err[:, i].reshape(2,1),
            fmt="s", color=c, ecolor=c,
            markersize=9, markeredgewidth=1.5,
            elinewidth=1.8, capsize=4, linestyle="None",
            label=None
        )
    if np.isfinite(ycdp_mean[i]):
        ax.errorbar(
            x_mean[i], ycdp_mean[i],
            xerr=xerr[:, i].reshape(2,1),
            yerr=ycdp_err[:, i].reshape(2,1),
            fmt="^", color=c, ecolor=c,
            markersize=10, markeredgewidth=1.5,
            elinewidth=1.8, capsize=4, linestyle="None",
            label=None
        )
legend_handles = [
    Line2D([0],[0], marker="s", color="k", lw=0, markersize=9, label="CAS monthly mean"),
    Line2D([0],[0], marker="^", color="k", lw=0, markersize=10, label="CDP monthly mean"),
    Line2D([0],[0], color="none", lw=0, label=f"Error bars: P{p_lo}–P{p_hi} (x and y)")
]
for m in months_plot:
    c = month_colors.get(m, "k")
    legend_handles.append(Line2D([0],[0], marker="o", color=c, lw=0, markersize=7, label=month_name[m]))
ax.set_yscale("log")
plt.yticks(fontsize=15, fontweight="bold")
plt.xticks(fontsize=15, fontweight="bold")
ax.grid(alpha=0.3)
ax.set_xlabel("Monthly mean wind speed (m/s)", fontsize=15, fontweight="bold")
ax.set_ylabel("Monthly mean GCCN concentration (cm⁻³)", fontsize=15, fontweight="bold")
ax.set_title("BCB January-June 2022\nMonthly GCCN vs Wind Speed", fontsize=15, fontweight="bold")
ax.legend(handles=legend_handles, ncol=2, fontsize=9, frameon=True)
plt.tight_layout()
plt.show()
#%%

# %%
def prep_df_for_x(dfp):
    dfp = dfp.copy()
    dfp["Date_dt"] = pd.to_datetime(dfp["Date"])
    if "Month" not in dfp.columns:
        dfp["Month"] = dfp["Date_dt"].dt.month
    sort_cols = ["Date_dt"]
    for c in ["BCB_start", "Min_start"]:
        if c in dfp.columns:
            sort_cols.append(c)
            break
    dfp = dfp.sort_values(sort_cols).reset_index(drop=True)
    x = np.arange(len(dfp))
    date_first = dfp.groupby(dfp["Date_dt"].dt.date).head(1)
    tick_pos = date_first.index.to_numpy()
    tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
    return dfp, x, tick_pos, tick_lab
dfp_cas, x_cas, tick_pos, tick_lab = prep_df_for_x(df_sorted)
dfp_cdp, x_cdp, _, _ = prep_df_for_x(df_sorted_CDP)
dfp_wind, x_wind, _, _ = prep_df_for_x(df_wind_CDP)
cas_arr = np.asarray(gccn_concentration, dtype=float)
cdp_arr = np.asarray(gccn_concentration_CDP, dtype=float)
wind_arr = pd.to_numeric(dfp_wind["Windspeed"], errors="coerce").to_numpy()
fig_w = max(22, 0.55 * len(tick_pos))
fig, ax1 = plt.subplots(figsize=(fig_w, 6.2))
ax2 = ax1.twinx()
for m in sorted(dfp_cas["Month"].unique()):
    if m not in month_name: 
        continue
    mask = (dfp_cas["Month"].values == m)
    c = month_colors.get(m, "k")
    ax1.plot(x_cas[mask], cas_arr[mask], "-", lw=1.5, color=c, alpha=0.9)

for m in sorted(dfp_cdp["Month"].unique()):
    if m not in month_name:
        continue
    mask = (dfp_cdp["Month"].values == m)
    c = month_colors.get(m, "k")
    ax1.plot(x_cdp[mask], cdp_arr[mask], "--", lw=1.5, color=c, alpha=0.9)
ax1.set_yscale("log")
ax1.tick_params(axis="y", labelsize=15)
for label in ax1.get_yticklabels():
    label.set_fontweight("bold")
ax1.set_ylabel("GCCN concentration (cm⁻³)", fontsize=16, fontweight="bold")
ax1.grid(alpha=0.3)
ax2.plot(x_wind, wind_arr, color="k", lw=1.6, alpha=0.75)
ax2.set_ylabel("Wind speed (m/s)", fontsize=16, fontweight="bold")
for p in tick_pos:
    ax1.axvline(p, color="k", alpha=0.06, linewidth=1)
ax1.set_xticks(tick_pos)
ax1.set_xticklabels(tick_lab, rotation=60, ha="right", fontsize=7, fontweight="bold")
labels = ax1.get_xticklabels()
for i, lab in enumerate(labels):
    base = lab.get_text()
    lab.set_text("\n" * (i % 4) + base)
ax1.set_xticklabels([lab.get_text() for lab in labels])
target_dates = {"2022-06-05": 0, "2022-06-07": 3}
labels = ax1.get_xticklabels()
for lab in labels:
    txt = lab.get_text().replace("\n", "")
    if txt in target_dates:
        lab.set_text("\n" * target_dates[txt] + txt)
ax1.set_xticklabels([lab.get_text() for lab in labels])
legend_handles = []
for m in sorted(set(dfp_cas["Month"].unique()).union(dfp_cdp["Month"].unique())):
    if m not in month_name:
        continue
    c = month_colors.get(m, "k")
    legend_handles.append(Line2D([0],[0], color=c, lw=2, label=month_name[m]))

legend_handles.extend([
    Line2D([0],[0], color="k", lw=2, linestyle="-",  label="CAS GCCN"),
    Line2D([0],[0], color="k", lw=2, linestyle="--", label="CDP GCCN"),
    Line2D([0],[0], color="k", lw=2, linestyle="-",  label="Wind speed (right axis)"),
])

ax1.legend(handles=legend_handles, ncol=3, fontsize=9, loc="lower right", frameon=True)
ax1.set_xlabel("Flight Date", fontsize=16, fontweight="bold")
ax1.set_title("BCB January–June 2022\nGCCN with Wind Speed",
              fontsize=16, fontweight="bold")
fig.subplots_adjust(bottom=0.40)
plt.yticks(fontsize=15, fontweight="bold")
fig.tight_layout()
plt.show()

# %%
df_gccn_full = df_sorted_cas.copy()
df_gccn_full["GCCN"] = gccn_concentration
df_master = dfp_cas_w.copy()
df_master["Mass"] = mass_cas_f

df_master = df_master.merge(
    df_gccn_full[["Date","BCB_start","BCB_stop","GCCN"]],
    on=["Date","BCB_start","BCB_stop"],
    how="left"
)
df_master = df_master[
    np.isfinite(df_master["Mass"]) &
    np.isfinite(df_master["Windspeed"]) &
    np.isfinite(df_master["GCCN"])
].reset_index(drop=True)

# %%
valid_legs = set(
    zip(dfp_cdp_w["Date"], dfp_cdp_w["BCB_start"], dfp_cdp_w["BCB_stop"])
)
mask = [
    (d, s, e) in valid_legs 
    for d, s, e in zip(df_sorted_CDP["Date"], df_sorted_CDP["BCB_start"], df_sorted_CDP["BCB_stop"])
]
df_gccn_matched = df_sorted_CDP[mask].reset_index(drop=True)
gccn_concentration_CDP_matched = df_gccn_matched["Total_Y_Concentration_cm3"].astype(float).to_numpy()
print("Mass+Wind legs:", len(dfp_cdp_w))
print("Original GCCN legs:", len(df_sorted_CDP))
print("Matched GCCN legs:", len(df_gccn_matched))

# %%
df_gccn_CAS = df_sorted.copy()
df_gccn_CAS["GCCN"] = gccn_concentration
df_gccn_CDP = df_sorted_CDP.copy()
df_gccn_CDP["GCCN"] = gccn_concentration_CDP
cas_valid = set(zip(dfp_cas_w["Date"], dfp_cas_w["BCB_start"], dfp_cas_w["BCB_stop"]))
cdp_valid = set(zip(dfp_cdp_w["Date"], dfp_cdp_w["BCB_start"], dfp_cdp_w["BCB_stop"]))
mask_cas = [(d,s,e) in cas_valid for d,s,e in zip(df_gccn_CAS["Date"], df_gccn_CAS["BCB_start"], df_gccn_CAS["BCB_stop"])]
df_gccn_CAS_m = df_gccn_CAS[mask_cas].reset_index(drop=True)
mask_cdp = [(d,s,e) in cdp_valid for d,s,e in zip(df_gccn_CDP["Date"], df_gccn_CDP["BCB_start"], df_gccn_CDP["BCB_stop"])]
df_gccn_CDP_m = df_gccn_CDP[mask_cdp].reset_index(drop=True)
print("CAS GCCN matched legs:", len(df_gccn_CAS_m))
print("CDP GCCN matched legs:", len(df_gccn_CDP_m))
key_cas = df_gccn_CAS_m.set_index(["Date","BCB_start","BCB_stop"]).index
dfp_cas_w = dfp_cas_w.set_index(["Date","BCB_start","BCB_stop"]).loc[key_cas].reset_index()
key_cdp = df_gccn_CDP_m.set_index(["Date","BCB_start","BCB_stop"]).index
dfp_cdp_w = dfp_cdp_w.set_index(["Date","BCB_start","BCB_stop"]).loc[key_cdp].reset_index()
assert len(df_gccn_CAS_m) == len(dfp_cas_w)
assert len(df_gccn_CDP_m) == len(dfp_cdp_w)
assert all(df_gccn_CAS_m["Date"].values == dfp_cas_w["Date"].values)
assert all(df_gccn_CDP_m["Date"].values == dfp_cdp_w["Date"].values)
cas_arr  = df_gccn_CAS_m["GCCN"].to_numpy(float)
cdp_arr  = df_gccn_CDP_m["GCCN"].to_numpy(float)
wind_cas = dfp_cas_w["Windspeed"].to_numpy(float)
wind_cdp = dfp_cdp_w["Windspeed"].to_numpy(float)

# %%
p_lo, p_hi = 10, 90
def add_month_col(dfp):
    dfp = dfp.copy()
    dfp["Date_dt"] = pd.to_datetime(dfp["Date"])
    if "Month" not in dfp.columns:
        dfp["Month"] = dfp["Date_dt"].dt.month
    return dfp
def stats_mean_and_prc(values, p_lo=10, p_hi=90):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    mean = np.nanmean(v)
    plo = np.nanpercentile(v, p_lo)
    phi = np.nanpercentile(v, p_hi)
    return mean, plo, phi

def month_summary(dfp, y, month_col="Month", p_lo=10, p_hi=90):
    out = {}
    months = sorted(pd.unique(dfp[month_col]))
    for m in months:
        mask = (dfp[month_col].values == m)
        mean, plo, phi = stats_mean_and_prc(np.asarray(y)[mask], p_lo, p_hi)
        out[m] = (mean, plo, phi)
    return out
dfp_cas = add_month_col(dfp_cas_w)
dfp_cdp = add_month_col(dfp_cdp_w)
dfp_wind = dfp_cdp
cas_sum  = month_summary(dfp_cas, cas_arr, "Month", p_lo, p_hi)
cdp_sum  = month_summary(dfp_cdp, cdp_arr, "Month", p_lo, p_hi)
wind_sum = month_summary(dfp_wind, wind_cdp, "Month", p_lo, p_hi)
months_all = sorted(set(cas_sum.keys()).union(cdp_sum.keys()).intersection(wind_sum.keys()))

x_mean, x_lo, x_hi = [], [], []
ycas_mean, ycas_lo, ycas_hi = [], [], []
ycdp_mean, ycdp_lo, ycdp_hi = [], [], []
months_plot = []

for m in months_all:
    if m not in month_name:
        continue

    w_mean, w_p10, w_p90 = wind_sum[m]
    cas_mean, cas_p10, cas_p90 = cas_sum.get(m, (np.nan,np.nan,np.nan))
    cdp_mean, cdp_p10, cdp_p90 = cdp_sum.get(m, (np.nan,np.nan,np.nan))

    if not np.isfinite(w_mean):
        continue
    if not (np.isfinite(cas_mean) or np.isfinite(cdp_mean)):
        continue

    months_plot.append(m)
    x_mean.append(w_mean); x_lo.append(w_mean - w_p10); x_hi.append(w_p90 - w_mean)
    ycas_mean.append(cas_mean); ycas_lo.append(cas_mean - cas_p10); ycas_hi.append(cas_p90 - cas_mean)
    ycdp_mean.append(cdp_mean); ycdp_lo.append(cdp_mean - cdp_p10); ycdp_hi.append(cdp_p90 - cdp_mean)

x_mean = np.array(x_mean); xerr = np.vstack([x_lo, x_hi])
ycas_mean = np.array(ycas_mean); ycas_err = np.vstack([ycas_lo, ycas_hi])
ycdp_mean = np.array(ycdp_mean); ycdp_err = np.vstack([ycdp_lo, ycdp_hi])

fig, ax = plt.subplots(figsize=(7.5, 5.8))

for i, m in enumerate(months_plot):
    c = month_colors.get(m, "k")

    if np.isfinite(ycas_mean[i]):
        ax.errorbar(x_mean[i], ycas_mean[i],
                    xerr=xerr[:, i].reshape(2,1),
                    yerr=ycas_err[:, i].reshape(2,1),
                    fmt="s", color=c, ecolor=c, markersize=9,
                    markeredgewidth=1.5, elinewidth=1.8, capsize=4)

    if np.isfinite(ycdp_mean[i]):
        ax.errorbar(x_mean[i], ycdp_mean[i],
                    xerr=xerr[:, i].reshape(2,1),
                    yerr=ycdp_err[:, i].reshape(2,1),
                    fmt="^", color=c, ecolor=c, markersize=10,
                    markeredgewidth=1.5, elinewidth=1.8, capsize=4)

legend_handles = [
    Line2D([0],[0], marker="s", color="k", lw=0, markersize=9, label="CAS monthly mean"),
    Line2D([0],[0], marker="^", color="k", lw=0, markersize=10, label="CDP monthly mean"),
    Line2D([0],[0], color="none", lw=0, label=f"Error bars: P{p_lo}–P{p_hi}")
]
for m in months_plot:
    legend_handles.append(Line2D([0],[0], marker="o", color=month_colors[m],
                                 lw=0, markersize=7, label=month_name[m]))

ax.set_yscale("log")
ax.grid(alpha=0.3)
ax.set_xlabel("Monthly mean wind speed (m/s)", fontsize=15, fontweight="bold")
ax.set_ylabel("Monthly mean GCCN concentration (cm⁻³)", fontsize=15, fontweight="bold")
ax.set_title("BCB January–June 2022\nMonthly GCCN vs Wind Speed", fontsize=15, fontweight="bold")
ax.legend(handles=legend_handles, ncol=2, fontsize=9, frameon=True, loc="lower right")

ax.tick_params(axis="both", labelsize=15)
for lab in ax.get_xticklabels() + ax.get_yticklabels():
    lab.set_fontweight("bold")

plt.tight_layout()
plt.show()
def prep_df_for_x(dfp):
    dfp = dfp.copy()
    dfp["Date_dt"] = pd.to_datetime(dfp["Date"])
    if "Month" not in dfp.columns:
        dfp["Month"] = dfp["Date_dt"].dt.month
    dfp = dfp.sort_values(["Date_dt","BCB_start"]).reset_index(drop=True)
    x = np.arange(len(dfp))
    date_first = dfp.groupby(dfp["Date_dt"].dt.date).head(1)
    tick_pos = date_first.index.to_numpy()
    tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
    return dfp, x, tick_pos, tick_lab
dfp_cas, x_cas, tick_pos, tick_lab = prep_df_for_x(dfp_cas_w)
dfp_cdp, x_cdp, _, _ = prep_df_for_x(dfp_cdp_w)

fig_w = max(22, 0.55 * len(tick_pos))
fig, ax1 = plt.subplots(figsize=(fig_w, 6.2))
ax2 = ax1.twinx()
for m in sorted(dfp_cas["Month"].unique()):
    if m not in month_name: continue
    mask = (dfp_cas["Month"].values == m)
    c = month_colors[m]
    ax1.plot(x_cas[mask], cas_arr[mask], "-", lw=1.5, color=c, alpha=0.9)

for m in sorted(dfp_cdp["Month"].unique()):
    if m not in month_name: continue
    mask = (dfp_cdp["Month"].values == m)
    c = month_colors[m]
    ax1.plot(x_cdp[mask], cdp_arr[mask], "--", lw=1.5, color=c, alpha=0.9)
ax2.plot(x_cdp, wind_cdp, color="k", lw=1.6, alpha=0.75)
ax1.set_yscale("log")
ax1.set_ylabel("GCCN concentration (cm⁻³)", fontsize=16, fontweight="bold")
ax2.set_ylabel("Wind speed (m/s)", fontsize=16, fontweight="bold")
ax1.grid(alpha=0.3)
ax1.tick_params(axis="y", labelsize=15)
ax2.tick_params(axis="y", labelsize=15)
for lab in ax1.get_yticklabels()+ax2.get_yticklabels():
    lab.set_fontweight("bold")
for p in tick_pos:
    ax1.axvline(p, color="k", alpha=0.06)
ax1.set_xticks(tick_pos)
ax1.set_xticklabels(tick_lab, rotation=60, ha="right", fontsize=7, fontweight="bold")
legend_handles = []
for m in sorted(set(dfp_cas["Month"].unique()).union(dfp_cdp["Month"].unique())):
    if m in month_name:
        legend_handles.append(Line2D([0],[0], color=month_colors[m], lw=2, label=month_name[m]))

legend_handles.extend([
    Line2D([0],[0], color="k", lw=2, linestyle="-",  label="CAS GCCN"),
    Line2D([0],[0], color="k", lw=2, linestyle="--", label="CDP GCCN"),
    Line2D([0],[0], color="k", lw=2, linestyle="-",  label="Wind speed"),
])

ax1.legend(handles=legend_handles, ncol=3, fontsize=9, loc="lower right", frameon=True)
ax1.set_xlabel("Flight Date", fontsize=16, fontweight="bold")
ax1.set_title("BCB January–June 2022\nGCCN with Wind Speed", fontsize=16, fontweight="bold")
fig.subplots_adjust(bottom=0.40)
fig.tight_layout()
plt.show()