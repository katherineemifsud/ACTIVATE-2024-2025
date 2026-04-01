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
from scipy.stats import pearsonr
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
#%%
#new gccn plotting code cas and cdp following new slope and mass 
month_colors = {
    1: "#0072B2",
    2: "#E69F00",
    3: "#009E73",
    5: "#CC79A7",
    6: "#56B4E9",
}
dfp_cas, _, gccn_cas_f, _, _ = prep_trend(df_sorted,
                                         gccn_concentration,
                                         mass_thr=np.inf)

dfp_cdp, _, gccn_cdp_f, _, _ = prep_trend(df_sorted_CDP,
                                         gccn_concentration_CDP,
                                         mass_thr=np.inf)

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
    ax.plot(x_cas[cas_mask], gccn_cas_f[cas_mask], "-", lw=1.5, color=c)
    ax.plot(x_cdp[cdp_mask], gccn_cdp_f[cdp_mask], "--", lw=1.5, color=c)
    median_cas = np.nanmedian(gccn_cas_f[cas_mask])
    median_cdp = np.nanmedian(gccn_cdp_f[cdp_mask])
    mean_cas   = np.nanmean(gccn_cas_f[cas_mask])
    mean_cdp   = np.nanmean(gccn_cdp_f[cdp_mask])
    median_x = np.nanmedian(x_cas[cas_mask])
    ax.plot(median_x, median_cas, marker="s", color="k", markersize=10, linestyle="None")
    ax.vlines(median_x, min(median_cas, mean_cas), max(median_cas, mean_cas),
              color="k", lw=1.5)
    ax.plot(median_x + 0.15, median_cdp, marker="^", color="k", markersize=10, linestyle="None")
    ax.vlines(median_x + 0.15, min(median_cdp, mean_cdp), max(median_cdp, mean_cdp),
              color="k", lw=1.5)

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
plt.yticks(fontsize=13, fontweight="bold")
ax.set_ylim(bottom=10**(-2.3))
ax.grid(alpha=0.3)
ax.set_ylabel("Total GCCN Concentration (cm⁻³)", fontsize=13, fontweight="bold")
ax.set_xlabel("Flight Date", fontsize=13, fontweight="bold")
ax.set_title("GCCN seasonal trend over the Western Atlantic", fontsize=13, fontweight="bold")
ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_lab, rotation=60, ha="right", fontsize=8, fontweight="bold")
ax.legend(handles=handles, ncol=2, fontsize=10, loc="lower right", frameon=True)
fig.subplots_adjust(bottom=0.40)
fig.tight_layout()
plt.show()
#%%
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
fig, (ax_mass, ax_slope, ax_gccn) = plt.subplots(
    3, 1, figsize=(fig_w, 12), sharex=True
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
    ax.set_title(title, fontsize=17, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=17, fontweight="bold")

    if yscale is not None:
        ax.set_yscale(yscale)
    if ylim is not None:
        ax.set_ylim(**ylim) if isinstance(ylim, dict) else ax.set_ylim(ylim)
    ax.tick_params(axis="y", labelsize=16)
    for t in ax.get_yticklabels():
        t.set_fontweight("bold")
    if show_xticks:
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab, rotation=60, ha="right", fontsize=14, fontweight="bold")

    return legend_month_handles
# MASS
legend_month_handles = plot_monthly(
    ax_mass,
    dfp_m_cas, dfp_m_cdp, xm_cas, xm_cdp, mass_cas_f, mass_cdp_f,
    yscale="log",
    ylim={"bottom": 10**(-1.3)},
    title="GCCN mass seasonal trend over the Western Atlantic",
    ylabel="GCCN Mass \n(µg m$^{-3}$)",
    show_xticks=False
)

# SLOPE
_ = plot_monthly(
    ax_slope,
    dfp_s_cas, dfp_s_cdp, xs_cas, xs_cdp, slope_cas_f, slope_cdp_f,
    yscale=None,
    ylim={"top": 4},
    title="GCCN slope parameter seasonal trend over the Western Atlantic",
    ylabel="Slope Parameter D\n (µm)",
    show_xticks=False
)

# GCCN
_ = plot_monthly(
    ax_gccn,
    dfp_g_cas, dfp_g_cdp, xg_cas, xg_cdp, gccn_cas_f, gccn_cdp_f,
    yscale="log",
    ylim={"bottom": 10**(-2.3)},
    title="GCCN concentration seasonal trend over the Western Atlantic",
    ylabel="Total GCCN Concentration\n (cm⁻³)",
    show_xticks=True
)

ax_gccn.set_xlabel("Flight Date", fontsize=17, fontweight="bold")
legend_instrument_handles = [
    Line2D([0], [0], color="k", lw=2, ls="-",  label="CAS (solid)"),
    Line2D([0], [0], color="k", lw=2, ls="--", label="CDP (dashed)"),
    Line2D([0], [0], marker="s", color="k", lw=0, markersize=12, label="CAS monthly median"),
    Line2D([0], [0], marker="^", color="k", lw=0, markersize=12, label="CDP monthly median"),
    Line2D([0], [0], color="k", lw=1.5, label="Monthly mean (whisker)")
]
handles = legend_month_handles + legend_instrument_handles
fig.legend(handles=handles, ncol=1, fontsize=17, frameon=True,
           loc="lower right", bbox_to_anchor=(1.19, 0.45))

fig.tight_layout()
fig.align_ylabels()
plt.show()
#save to pdf
fig.savefig("gccn3panel_seasonal_trend.pdf", bbox_inches="tight")
#%%
#monthly correlation table for CAS and CDP with gccn concentration, slope, and mass
#across all months
months = sorted(dfp_m_cas["Month"].unique())

def monthly_stat(dfp, arr, m):
    mask = dfp["Month"].values == m
    return np.nanmedian(arr[mask]) 

for name, cas_arr, cdp_arr, dfp_cas, dfp_cdp in [
    ("Mass",  mass_cas_f,  mass_cdp_f,  dfp_m_cas, dfp_m_cdp),
    ("Slope", slope_cas_f, slope_cdp_f, dfp_s_cas, dfp_s_cdp),
    ("GCCN",  gccn_cas_f,  gccn_cdp_f,  dfp_g_cas, dfp_g_cdp),
]:

    cas_month = []
    cdp_month = []

    for m in months:
        cas_month.append(monthly_stat(dfp_cas, cas_arr, m))
        cdp_month.append(monthly_stat(dfp_cdp, cdp_arr, m))

    cas_month = np.array(cas_month)
    cdp_month = np.array(cdp_month)

    ok = np.isfinite(cas_month) & np.isfinite(cdp_month)
    r, _ = pearsonr(cas_month[ok], cdp_month[ok])

    print(f"\n{name}")
    print("CAS monthly:", cas_month)
    print("CDP monthly:", cdp_month)
    print(f"Trend r = {r:.3f}, R² = {r*r:.3f}")

#%%
df_full = pd.DataFrame({
    "Mass (µg m$^{-3}$)": [f"{0.555:.3f} ({0.308:.3f})"],
    "Slope Parameter D (µm)": [f"{0.836:.3f} ({0.699:.3f})"],
    r"$\mathbf{N_d}$ (cm$^{-3}$)" + "\n": [f"{0.913:.3f} ({0.833:.3f})"],
})

nrows, ncols = df_full.shape
fig_w = max(10, 1.8 * ncols)
fig_h = max(2.5, 0.8 * (nrows + 1))

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.axis("off")

tbl = ax.table(
    cellText=df_full.values,
    colLabels=df_full.columns,
    cellLoc="center",
    colLoc="center",
    bbox=[0, 0.15, 1.0, 0.7]   # [left, bottom, width, height]
)

tbl.auto_set_column_width(col=list(range(ncols)))
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1.0, 1.6)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("black")
    if r == 0:  # header
        cell.set_text_props(weight="bold")
        cell.set_linewidth(1.2)
        cell.set_facecolor("#F2F2F2")
    else:
        cell.set_linewidth(0.8)
for c in range(ncols):
    cell = tbl[(0, c)]
    cell.set_height(cell.get_height() * 1.35)

ax.set_title(r"January–June 2022 Instrument Correlations (r($\mathbf{R^2}$))",
             fontsize=16, fontweight="bold", pad=12)
plt.savefig("full_trend_correlations_table.pdf", bbox_inches="tight")
plt.show()
#%%
#individual monthly instrument correlation 

month_name = {1:"January", 2:"February", 3:"March", 5:"May", 6:"June"}
months_order = [1, 2, 3, 5, 6]

def per_month_daily_corr(dfp_cas, y_cas, dfp_cdp, y_cdp):
    cas = pd.DataFrame({
        "Date_dt": pd.to_datetime(dfp_cas["Date_dt"]),
        "Month": dfp_cas["Month"].astype(int),
        "y": np.asarray(y_cas, dtype=float),
    })
    cdp = pd.DataFrame({
        "Date_dt": pd.to_datetime(dfp_cdp["Date_dt"]),
        "Month": dfp_cdp["Month"].astype(int),
        "y": np.asarray(y_cdp, dtype=float),
    })
    cas_day = cas.groupby(["Month", "Date_dt"], as_index=False)["y"].median().rename(columns={"y":"CAS"})
    cdp_day = cdp.groupby(["Month", "Date_dt"], as_index=False)["y"].median().rename(columns={"y":"CDP"})
    m = cas_day.merge(cdp_day, on=["Month", "Date_dt"], how="inner")

    out = {}
    for mon in months_order:
        g = m[m["Month"] == mon]
        x = g["CAS"].to_numpy(float)
        y = g["CDP"].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]; y = y[ok]

        if len(x) < 2:
            out[mon] = (np.nan, np.nan, len(x))
        else:
            r, _ = pearsonr(x, y)
            out[mon] = (r, r*r, len(x))
    return out
corr_mass  = per_month_daily_corr(dfp_m_cas, mass_cas_f,  dfp_m_cdp, mass_cdp_f)
corr_slope = per_month_daily_corr(dfp_s_cas, slope_cas_f, dfp_s_cdp, slope_cdp_f)
corr_gccn  = per_month_daily_corr(dfp_g_cas, gccn_cas_f,  dfp_g_cdp, gccn_cdp_f)
for mon in months_order:
    mon_str = month_name[mon]
    rm, r2m, nm = corr_mass[mon]
    rs, r2s, ns = corr_slope[mon]
    rg, r2g, ng = corr_gccn[mon]
    def fmt(r, r2, n):
        return "NA" if not np.isfinite(r) else f"{r:.3f} ({r2:.3f})  N={n}"

    print(f"\n{mon_str}")
    print("  Mass :", fmt(rm, r2m, nm))
    print("  Slope:", fmt(rs, r2s, ns))
    print("  GCCN :", fmt(rg, r2g, ng))

months_order = [1,2,3,5,6]
month_name = {1:"January", 2:"February", 3:"March", 5:"May", 6:"June"}

def cell(cdict, mon):
    r, r2, n = cdict[mon]
    return "NA" if not np.isfinite(r) else f"{r:.3f} ({r2:.3f})"

table_df = pd.DataFrame({
    "Month": [month_name[m] for m in months_order],
    "Mass (µg m$^{-3}$)":  [cell(corr_mass,  m) for m in months_order],
    "Slope Parameter D (µm)": [cell(corr_slope, m) for m in months_order],
    r"$\mathbf{N_d}$ (cm$^{-3}$)"
:  [cell(corr_gccn,  m) for m in months_order],
})

fig, ax = plt.subplots(figsize=(10.5, 3.6))
ax.axis("off")

tbl = ax.table(
    cellText=table_df.values,
    colLabels=table_df.columns,
    cellLoc="center",
    colLoc="center",
    bbox=[0, 0.05, 1.15, 0.85]

)

tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.2, 1.6)

for (r, c), cell_obj in tbl.get_celld().items():
    if r == 0:
        cell_obj.set_text_props(weight="bold")
        cell_obj.set_linewidth(1.2)
    else:
        cell_obj.set_linewidth(0.7)

ax.set_title("Monthly Instrument Correlations (r(R²))", fontsize=18, fontweight="bold")
plt.show()
plt.savefig("monthly_instrument_correlations.pdf", bbox_inches="tight")

#%%

df = pd.DataFrame({
    "Month": ["January","February","March","May","June"],
    "CAS $N_d$\nMean (cm⁻³)":   [0.313,0.465,0.652,0.618,0.785],
    "CAS $N_d$\nMedian (cm⁻³)": [0.248,0.404,0.460,0.306,0.779],
    "CDP $N_d$\nMean (cm⁻³)":   [0.080,0.123,0.472,0.151,0.288],
    "CDP $N_d$\nMedian (cm⁻³)": [0.032,0.032,0.161,0.087,0.273],
    "CAS D\nMean (µm)":  [1.177,0.818,0.912,0.525,0.625],
    "CAS D\nMedian (µm)": [0.868,0.774,0.777,0.529,0.679],
    "CDP D\nMean (µm)":  [1.450,1.180,1.463,0.897,1.231],
    "CDP D\nMedian (µm)": [1.319,1.129,1.343,0.856,1.256],
    "CAS Mass\nMean (µg/m³)":   [9.021,8.985,15.476,4.950,9.702],
    "CAS Mass\nMedian (µg/m³)": [6.014,7.525,10.897,2.273,9.559],
    "CDP Mass\nMean (µg/m³)":   [5.602,5.103,21.487,4.599,22.692],
    "CDP Mass\nMedian (µg/m³)": [2.297,1.946,6.003,2.248,16.652],
})

df_disp = pd.DataFrame({
    "Month": df["Month"],
    "CAS " + r"$\mathbf{N_d}$" + "\n(cm$^{-3}$)":  [f"{m:.3f} ({md:.3f})" for m, md in zip(df["CAS $N_d$\nMean (cm⁻³)"],   df["CAS $N_d$\nMedian (cm⁻³)"])],
    "CDP " + r"$\mathbf{N_d}$" + "\n(cm$^{-3}$)":  [f"{m:.3f} ({md:.3f})" for m, md in zip(df["CDP $N_d$\nMean (cm⁻³)"],   df["CDP $N_d$\nMedian (cm⁻³)"])],
    "CAS Slope Parameter \nD (µm)":   [f"{m:.3f} ({md:.3f})" for m, md in zip(df["CAS D\nMean (µm)"],      df["CAS D\nMedian (µm)"])],
    "CDP Slope Parameter \nD (µm)":   [f"{m:.3f} ({md:.3f})" for m, md in zip(df["CDP D\nMean (µm)"],      df["CDP D\nMedian (µm)"])],
    "CAS Mass\n (µg m$^{-3}$) ":[f"{m:.3f} ({md:.3f})" for m, md in zip(df["CAS Mass\nMean (µg/m³)"],   df["CAS Mass\nMedian (µg/m³)"])],
    "CDP Mass\n (µg m$^{-3}$)":[f"{m:.3f} ({md:.3f})" for m, md in zip(df["CDP Mass\nMean (µg/m³)"],   df["CDP Mass\nMedian (µg/m³)"])],
})
nrows, ncols = df_disp.shape
fig_w = max(12, 1.5 * ncols)
fig_h = max(3, 0.65 * (nrows+1))
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.axis("off")
tbl = ax.table(
    cellText=df_disp.values,
    colLabels=df_disp.columns,
    cellLoc="center",
    colLoc="center",
    bbox=[0, 0.05, 1.15, 0.9]

)
tbl.auto_set_column_width(col=list(range(ncols)))

tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.0, 1.35)

for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("black")
    if r == 0:
        cell.set_text_props(weight="bold")
        cell.set_linewidth(1.2)
        cell.set_facecolor("#F2F2F2")
    else:
        cell.set_linewidth(0.6)
for c in range(ncols):
    cell = tbl[(0, c)]
    cell.set_height(cell.get_height() * 1.4)
ax.set_title("Monthly Summary Statistics (mean(median))",
             fontsize=15, fontweight="bold")
plt.show()
plt.savefig("monthly_trend_summary.pdf", bbox_inches="tight")
#%%
#combining the two tables
table_df = pd.DataFrame({
    "Month": ["January", "February", "March", "May", "June", "All months"],
    "Mass (µg m$^{-3}$)": [
        "0.693 (0.481)",
        "0.182 (0.033)",
        "-0.255 (0.065)",
        "0.538 (0.289)",
        "0.897 (0.805)",
        "0.555 (0.308)",
    ],
    "Slope Parameter D \n(µm)": [
        "0.771 (0.594)",
        "0.611 (0.373)",
        "0.089 (0.008)",
        "-0.658 (0.434)",
        "0.912 (0.832)",
        "0.836 (0.699)",
    ],
    r"$\mathbf{N_d}$ (cm$^{-3}$)": [
        "0.753 (0.566)",
        "0.334 (0.111)",
        "0.026 (0.001)",
        "0.583 (0.340)",
        "0.984 (0.969)",
        "0.913 (0.833)",
    ],
})
nrows, ncols = table_df.shape
fig_w = max(10, 1.8 * ncols)
fig_h = max(2.8, 0.75 * (nrows + 1))

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.axis("off")

tbl = ax.table(
    cellText=table_df.values,
    colLabels=table_df.columns,
    cellLoc="center",
    colLoc="center",
    bbox=[0, 0.08, 1.0, 0.82]
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.1, 1.6)
for (r, c), cell_obj in tbl.get_celld().items():
    cell_obj.set_edgecolor("black")
    if r == 0:
        cell_obj.set_text_props(weight="bold")
        cell_obj.set_linewidth(1.2)
        cell_obj.set_facecolor("#F2F2F2")
    else:
        cell_obj.set_linewidth(0.7)
for c in range(ncols):
    hcell = tbl[(0, c)]
    hcell.set_height(hcell.get_height() * 1.35)
all_months_row = len(table_df) 
for c in range(ncols):
    tbl[(all_months_row, c)].set_text_props(weight="bold")
ax.set_title("Monthly Correlations between CAS and CDP (r(R²))", fontsize=18, fontweight="bold")
plt.savefig("monthly_plus_overall_correlations.pdf", bbox_inches="tight")
plt.show()
#%%
#mass versus conc. with color coded slope with 1 plot for CAS and 1 for CDP 2 panel figure.
from matplotlib.colors import Normalize
KEYS = ["Date", "BCB_start", "BCB_stop"]
def build_plot_df(df_mass, mass_arr, df_gccn, gccn_arr, slope_col="Dry Slope (D)", mass_thr=100.0):
    dm = df_mass.copy()
    dm["mass"] = np.asarray(mass_arr, dtype=float)
    dm["slope"] = pd.to_numeric(dm[slope_col], errors="coerce")
    dm = dm[KEYS + ["mass", "slope"]].drop_duplicates(KEYS)

    dg = df_gccn.copy()
    dg["gccn"] = np.asarray(gccn_arr, dtype=float)
    dg = dg[KEYS + ["gccn"]].drop_duplicates(KEYS)

    out = dm.merge(dg, on=KEYS, how="inner")
    out = out[np.isfinite(out["mass"]) & np.isfinite(out["gccn"]) & np.isfinite(out["slope"])]
    out = out[(out["mass"] > 0) & (out["gccn"] > 0)]
    out = out[out["mass"] <= mass_thr]
    return out

df_plot_cas = build_plot_df(df_sorted_cas, mass_cas, df_sorted,     gccn_concentration,     mass_thr=100.0)
df_plot_cdp = build_plot_df(df_sorted_cdp, mass_cdp, df_sorted_CDP, gccn_concentration_CDP, mass_thr=100.0)

print("CAS plotted legs:", len(df_plot_cas))
print("CDP plotted legs:", len(df_plot_cdp))
all_d = np.concatenate([df_plot_cas["slope"].to_numpy(), df_plot_cdp["slope"].to_numpy()])
vmin, vmax = np.nanpercentile(all_d, [2, 98])
norm = Normalize(vmin=vmin, vmax=vmax)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
sc1 = ax1.scatter(df_plot_cas["gccn"], df_plot_cas["mass"],
                  c=df_plot_cas["slope"], norm=norm,
                  s=55, edgecolor="k", linewidth=0.4, alpha=0.9)

sc2 = ax2.scatter(df_plot_cdp["gccn"], df_plot_cdp["mass"],
                  c=df_plot_cdp["slope"], norm=norm,
                  s=55, edgecolor="k", linewidth=0.4, alpha=0.9)
cas_med_g = np.nanmedian(df_plot_cas["gccn"].to_numpy())
cas_med_m = np.nanmedian(df_plot_cas["mass"].to_numpy())
cas_med_s = np.nanmedian(df_plot_cas["slope"].to_numpy())
cdp_med_g = np.nanmedian(df_plot_cdp["gccn"].to_numpy())
cdp_med_m = np.nanmedian(df_plot_cdp["mass"].to_numpy())
cdp_med_s = np.nanmedian(df_plot_cdp["slope"].to_numpy())
ax1.scatter(cas_med_g, cas_med_m, marker="s", s=260,
            facecolor="k", edgecolor="k", linewidth=1.2, zorder=10)
ax2.scatter(cdp_med_g, cdp_med_m, marker="^", s=260,
            facecolor="k", edgecolor="k", linewidth=1.2, zorder=10)
median_handle = Line2D([0], [0], marker="^", linestyle="None",
                       markerfacecolor="k", markeredgecolor="k",
                       markersize=12, label="Median (GCCN, Mass)")

cas_text = f"CAS median D = {cas_med_s:.2f} µm"
cdp_text = f"CDP median D = {cdp_med_s:.2f} µm"
cas_handle = Line2D([0], [0], marker="s", linestyle="None",
                    markerfacecolor="k", markeredgecolor="k",
                    markersize=12, label=cas_text)
cdp_handle = Line2D([0], [0], marker="^", linestyle="None",
                    markerfacecolor="k", markeredgecolor="k",
                    markersize=12, label=cdp_text)
fig.legend(handles=[cas_handle, cdp_handle], loc="center left",
           bbox_to_anchor=(1.06, 0.5), frameon=False, fontsize=14)
fig.tight_layout(rect=[0, 0, 0.88, 1])

for ax, title in [(ax1, "CAS"), (ax2, "CDP")]:
    ax.set_xscale("log")
    ax.set_yscale("log")    
    ax1.set_ylim(bottom=10**(-2.5))
    ax2.set_ylim(bottom=10**(-2.5))
    ax1.set_ylim(top=10**(2.2))
    ax2.set_ylim(top=10**(2.2))

    ax.grid(alpha=0.25)
    ax.set_title(title, fontsize=17, fontweight="bold")
ax1.set_xlabel("GCCN Concentration (cm⁻³)", fontsize=17, fontweight="bold")
ax2.set_xlabel("GCCN Concentration (cm⁻³)", fontsize=17, fontweight="bold")
ax1.set_ylabel("GCCN Mass (µg m$^{-3}$)", fontsize=17, fontweight="bold")
cbar = fig.colorbar(sc2, ax=[ax1, ax2], fraction=0.045, pad=0.0)
cbar.ax.set_position([1, 0.15, 0.02, 0.7])
cbar.set_label("GCCN Slope Parameter D (µm)", fontsize=17, fontweight="bold")
fig.tight_layout()
for ax in [ax1, ax2]:
    ax.tick_params(axis="both", labelsize=17)
    
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
for label in cbar.ax.get_yticklabels():
    label.set_fontweight("bold")
plt.show()
#save figure as pdf
fig.savefig("CAS_CDP_slope_conc_mass_scatter.pdf", bbox_inches="tight")
# %%
# #plotting gccn with windspeed. 

# p_lo, p_hi = 10, 90
# def add_month_col(dfp):
#     dfp = dfp.copy()
#     dfp["Date_dt"] = pd.to_datetime(dfp["Date"])
#     if "Month" not in dfp.columns:
#         dfp["Month"] = dfp["Date_dt"].dt.month
#     return dfp

# def stats_mean_and_prc(values, p_lo=10, p_hi=90):
#     v = np.asarray(values, dtype=float)
#     v = v[np.isfinite(v)]
#     if v.size == 0:
#         return np.nan, np.nan, np.nan
#     mean = np.nanmean(v)
#     plo = np.nanpercentile(v, p_lo)
#     phi = np.nanpercentile(v, p_hi)
#     return mean, plo, phi

# def month_summary(dfp, y, month_col="Month", p_lo=10, p_hi=90):
#     out = {}
#     months = sorted(pd.unique(dfp[month_col]))
#     for m in months:
#         mask = (dfp[month_col].values == m)
#         mean, plo, phi = stats_mean_and_prc(np.asarray(y)[mask], p_lo, p_hi)
#         out[m] = (mean, plo, phi)
#     return out
# dfp_cas = add_month_col(df_sorted)
# dfp_cdp = add_month_col(df_sorted_CDP)
# dfp_wind = add_month_col(df_wind_CDP)

# cas_arr = np.asarray(gccn_concentration, dtype=float)
# cdp_arr = np.asarray(gccn_concentration_CDP, dtype=float)
# wind_arr = pd.to_numeric(dfp_wind["Windspeed"], errors="coerce").to_numpy()
# cas_sum  = month_summary(dfp_cas,  cas_arr,  "Month", p_lo, p_hi)
# cdp_sum  = month_summary(dfp_cdp,  cdp_arr,  "Month", p_lo, p_hi)
# wind_sum = month_summary(dfp_wind, wind_arr, "Month", p_lo, p_hi)
# months_all = sorted(set(cas_sum.keys()).union(cdp_sum.keys()).intersection(wind_sum.keys()))
# x_mean, x_lo, x_hi = [], [], []
# ycas_mean, ycas_lo, ycas_hi = [], [], []
# ycdp_mean, ycdp_lo, ycdp_hi = [], [], []
# months_plot = []
# for m in months_all:
#     if m not in month_name: 
#         continue
#     w_mean, w_p10, w_p90 = wind_sum[m]
#     cas_mean, cas_p10, cas_p90 = cas_sum.get(m, (np.nan, np.nan, np.nan))
#     cdp_mean, cdp_p10, cdp_p90 = cdp_sum.get(m, (np.nan, np.nan, np.nan))
#     if not np.isfinite(w_mean):
#         continue
#     if not (np.isfinite(cas_mean) or np.isfinite(cdp_mean)):
#         continue

#     months_plot.append(m)
#     x_mean.append(w_mean); x_lo.append(w_mean - w_p10); x_hi.append(w_p90 - w_mean) 
#     ycas_mean.append(cas_mean); ycas_lo.append(cas_mean - cas_p10); ycas_hi.append(cas_p90 - cas_mean)
#     ycdp_mean.append(cdp_mean); ycdp_lo.append(cdp_mean - cdp_p10); ycdp_hi.append(cdp_p90 - cdp_mean)

# x_mean = np.array(x_mean); xerr = np.vstack([x_lo, x_hi])
# ycas_mean = np.array(ycas_mean); ycas_err = np.vstack([ycas_lo, ycas_hi])
# ycdp_mean = np.array(ycdp_mean); ycdp_err = np.vstack([ycdp_lo, ycdp_hi])
# fig, ax = plt.subplots(figsize=(7.5, 5.8))
# for i, m in enumerate(months_plot):
#     c = month_colors.get(m, "k")
#     if np.isfinite(ycas_mean[i]):
#         ax.errorbar(
#             x_mean[i], ycas_mean[i],
#             xerr=xerr[:, i].reshape(2,1),
#             yerr=ycas_err[:, i].reshape(2,1),
#             fmt="s", color=c, ecolor=c,
#             markersize=9, markeredgewidth=1.5,
#             elinewidth=1.8, capsize=4, linestyle="None",
#             label=None
#         )
#     if np.isfinite(ycdp_mean[i]):
#         ax.errorbar(
#             x_mean[i], ycdp_mean[i],
#             xerr=xerr[:, i].reshape(2,1),
#             yerr=ycdp_err[:, i].reshape(2,1),
#             fmt="^", color=c, ecolor=c,
#             markersize=10, markeredgewidth=1.5,
#             elinewidth=1.8, capsize=4, linestyle="None",
#             label=None
#         )
# legend_handles = [
#     Line2D([0],[0], marker="s", color="k", lw=0, markersize=9, label="CAS monthly mean"),
#     Line2D([0],[0], marker="^", color="k", lw=0, markersize=10, label="CDP monthly mean"),
#     Line2D([0],[0], color="none", lw=0, label=f"Error bars: P{p_lo}–P{p_hi} (x and y)")
# ]
# for m in months_plot:
#     c = month_colors.get(m, "k")
#     legend_handles.append(Line2D([0],[0], marker="o", color=c, lw=0, markersize=7, label=month_name[m]))
# ax.set_yscale("log")
# plt.yticks(fontsize=15, fontweight="bold")
# plt.xticks(fontsize=15, fontweight="bold")
# ax.grid(alpha=0.3)
# ax.set_xlabel("Monthly mean wind speed (m/s)", fontsize=15, fontweight="bold")
# ax.set_ylabel("Monthly mean GCCN concentration (cm⁻³)", fontsize=15, fontweight="bold")
# ax.set_title("BCB January-June 2022\nMonthly GCCN vs Wind Speed", fontsize=15, fontweight="bold")
# ax.legend(handles=legend_handles, ncol=2, fontsize=9, frameon=True)
# plt.tight_layout()
# plt.show()
# #%%

# # %%
# def prep_df_for_x(dfp):
#     dfp = dfp.copy()
#     dfp["Date_dt"] = pd.to_datetime(dfp["Date"])
#     if "Month" not in dfp.columns:
#         dfp["Month"] = dfp["Date_dt"].dt.month
#     sort_cols = ["Date_dt"]
#     for c in ["BCB_start", "Min_start"]:
#         if c in dfp.columns:
#             sort_cols.append(c)
#             break
#     dfp = dfp.sort_values(sort_cols).reset_index(drop=True)
#     x = np.arange(len(dfp))
#     date_first = dfp.groupby(dfp["Date_dt"].dt.date).head(1)
#     tick_pos = date_first.index.to_numpy()
#     tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
#     return dfp, x, tick_pos, tick_lab
# dfp_cas, x_cas, tick_pos, tick_lab = prep_df_for_x(df_sorted)
# dfp_cdp, x_cdp, _, _ = prep_df_for_x(df_sorted_CDP)
# dfp_wind, x_wind, _, _ = prep_df_for_x(df_wind_CDP)
# cas_arr = np.asarray(gccn_concentration, dtype=float)
# cdp_arr = np.asarray(gccn_concentration_CDP, dtype=float)
# wind_arr = pd.to_numeric(dfp_wind["Windspeed"], errors="coerce").to_numpy()
# fig_w = max(22, 0.55 * len(tick_pos))
# fig, ax1 = plt.subplots(figsize=(fig_w, 6.2))
# ax2 = ax1.twinx()
# for m in sorted(dfp_cas["Month"].unique()):
#     if m not in month_name: 
#         continue
#     mask = (dfp_cas["Month"].values == m)
#     c = month_colors.get(m, "k")
#     ax1.plot(x_cas[mask], cas_arr[mask], "-", lw=1.5, color=c, alpha=0.9)

# for m in sorted(dfp_cdp["Month"].unique()):
#     if m not in month_name:
#         continue
#     mask = (dfp_cdp["Month"].values == m)
#     c = month_colors.get(m, "k")
#     ax1.plot(x_cdp[mask], cdp_arr[mask], "--", lw=1.5, color=c, alpha=0.9)
# ax1.set_yscale("log")
# ax1.tick_params(axis="y", labelsize=15)
# for label in ax1.get_yticklabels():
#     label.set_fontweight("bold")
# ax1.set_ylabel("GCCN concentration (cm⁻³)", fontsize=16, fontweight="bold")
# ax1.grid(alpha=0.3)
# ax2.plot(x_wind, wind_arr, color="k", lw=1.6, alpha=0.75)
# ax2.set_ylabel("Wind speed (m/s)", fontsize=16, fontweight="bold")
# for p in tick_pos:
#     ax1.axvline(p, color="k", alpha=0.06, linewidth=1)
# ax1.set_xticks(tick_pos)
# ax1.set_xticklabels(tick_lab, rotation=60, ha="right", fontsize=7, fontweight="bold")
# labels = ax1.get_xticklabels()
# for i, lab in enumerate(labels):
#     base = lab.get_text()
#     lab.set_text("\n" * (i % 4) + base)
# ax1.set_xticklabels([lab.get_text() for lab in labels])
# target_dates = {"2022-06-05": 0, "2022-06-07": 3}
# labels = ax1.get_xticklabels()
# for lab in labels:
#     txt = lab.get_text().replace("\n", "")
#     if txt in target_dates:
#         lab.set_text("\n" * target_dates[txt] + txt)
# ax1.set_xticklabels([lab.get_text() for lab in labels])
# legend_handles = []
# for m in sorted(set(dfp_cas["Month"].unique()).union(dfp_cdp["Month"].unique())):
#     if m not in month_name:
#         continue
#     c = month_colors.get(m, "k")
#     legend_handles.append(Line2D([0],[0], color=c, lw=2, label=month_name[m]))

# legend_handles.extend([
#     Line2D([0],[0], color="k", lw=2, linestyle="-",  label="CAS GCCN"),
#     Line2D([0],[0], color="k", lw=2, linestyle="--", label="CDP GCCN"),
#     Line2D([0],[0], color="k", lw=2, linestyle="-",  label="Wind speed (right axis)"),
# ])

# ax1.legend(handles=legend_handles, ncol=3, fontsize=9, loc="lower right", frameon=True)
# ax1.set_xlabel("Flight Date", fontsize=16, fontweight="bold")
# ax1.set_title("BCB January–June 2022\nGCCN with Wind Speed",
#               fontsize=16, fontweight="bold")
# fig.subplots_adjust(bottom=0.40)
# plt.yticks(fontsize=15, fontweight="bold")
# fig.tight_layout()
# plt.show()

# # %%
# df_gccn_full = df_sorted_cas.copy()
# df_gccn_full["GCCN"] = gccn_concentration
# df_master = dfp_cas_w.copy()
# df_master["Mass"] = mass_cas_f

# df_master = df_master.merge(
#     df_gccn_full[["Date","BCB_start","BCB_stop","GCCN"]],
#     on=["Date","BCB_start","BCB_stop"],
#     how="left"
# )
# df_master = df_master[
#     np.isfinite(df_master["Mass"]) &
#     np.isfinite(df_master["Windspeed"]) &
#     np.isfinite(df_master["GCCN"])
# ].reset_index(drop=True)

# # %%
# valid_legs = set(
#     zip(dfp_cdp_w["Date"], dfp_cdp_w["BCB_start"], dfp_cdp_w["BCB_stop"])
# )
# mask = [
#     (d, s, e) in valid_legs 
#     for d, s, e in zip(df_sorted_CDP["Date"], df_sorted_CDP["BCB_start"], df_sorted_CDP["BCB_stop"])
# ]
# df_gccn_matched = df_sorted_CDP[mask].reset_index(drop=True)
# gccn_concentration_CDP_matched = df_gccn_matched["Total_Y_Concentration_cm3"].astype(float).to_numpy()
# print("Mass+Wind legs:", len(dfp_cdp_w))
# print("Original GCCN legs:", len(df_sorted_CDP))
# print("Matched GCCN legs:", len(df_gccn_matched))

# # %%
# df_gccn_CAS = df_sorted.copy()
# df_gccn_CAS["GCCN"] = gccn_concentration
# df_gccn_CDP = df_sorted_CDP.copy()
# df_gccn_CDP["GCCN"] = gccn_concentration_CDP
# cas_valid = set(zip(dfp_cas_w["Date"], dfp_cas_w["BCB_start"], dfp_cas_w["BCB_stop"]))
# cdp_valid = set(zip(dfp_cdp_w["Date"], dfp_cdp_w["BCB_start"], dfp_cdp_w["BCB_stop"]))
# mask_cas = [(d,s,e) in cas_valid for d,s,e in zip(df_gccn_CAS["Date"], df_gccn_CAS["BCB_start"], df_gccn_CAS["BCB_stop"])]
# df_gccn_CAS_m = df_gccn_CAS[mask_cas].reset_index(drop=True)
# mask_cdp = [(d,s,e) in cdp_valid for d,s,e in zip(df_gccn_CDP["Date"], df_gccn_CDP["BCB_start"], df_gccn_CDP["BCB_stop"])]
# df_gccn_CDP_m = df_gccn_CDP[mask_cdp].reset_index(drop=True)
# print("CAS GCCN matched legs:", len(df_gccn_CAS_m))
# print("CDP GCCN matched legs:", len(df_gccn_CDP_m))
# key_cas = df_gccn_CAS_m.set_index(["Date","BCB_start","BCB_stop"]).index
# dfp_cas_w = dfp_cas_w.set_index(["Date","BCB_start","BCB_stop"]).loc[key_cas].reset_index()
# key_cdp = df_gccn_CDP_m.set_index(["Date","BCB_start","BCB_stop"]).index
# dfp_cdp_w = dfp_cdp_w.set_index(["Date","BCB_start","BCB_stop"]).loc[key_cdp].reset_index()
# assert len(df_gccn_CAS_m) == len(dfp_cas_w)
# assert len(df_gccn_CDP_m) == len(dfp_cdp_w)
# assert all(df_gccn_CAS_m["Date"].values == dfp_cas_w["Date"].values)
# assert all(df_gccn_CDP_m["Date"].values == dfp_cdp_w["Date"].values)
# cas_arr  = df_gccn_CAS_m["GCCN"].to_numpy(float)
# cdp_arr  = df_gccn_CDP_m["GCCN"].to_numpy(float)
# wind_cas = dfp_cas_w["Windspeed"].to_numpy(float)
# wind_cdp = dfp_cdp_w["Windspeed"].to_numpy(float)

# # %%
# p_lo, p_hi = 10, 90
# def add_month_col(dfp):
#     dfp = dfp.copy()
#     dfp["Date_dt"] = pd.to_datetime(dfp["Date"])
#     if "Month" not in dfp.columns:
#         dfp["Month"] = dfp["Date_dt"].dt.month
#     return dfp
# def stats_mean_and_prc(values, p_lo=10, p_hi=90):
#     v = np.asarray(values, dtype=float)
#     v = v[np.isfinite(v)]
#     if v.size == 0:
#         return np.nan, np.nan, np.nan
#     mean = np.nanmean(v)
#     plo = np.nanpercentile(v, p_lo)
#     phi = np.nanpercentile(v, p_hi)
#     return mean, plo, phi

# def month_summary(dfp, y, month_col="Month", p_lo=10, p_hi=90):
#     out = {}
#     months = sorted(pd.unique(dfp[month_col]))
#     for m in months:
#         mask = (dfp[month_col].values == m)
#         mean, plo, phi = stats_mean_and_prc(np.asarray(y)[mask], p_lo, p_hi)
#         out[m] = (mean, plo, phi)
#     return out
# dfp_cas = add_month_col(dfp_cas_w)
# dfp_cdp = add_month_col(dfp_cdp_w)
# dfp_wind = dfp_cdp
# cas_sum  = month_summary(dfp_cas, cas_arr, "Month", p_lo, p_hi)
# cdp_sum  = month_summary(dfp_cdp, cdp_arr, "Month", p_lo, p_hi)
# wind_sum = month_summary(dfp_wind, wind_cdp, "Month", p_lo, p_hi)
# months_all = sorted(set(cas_sum.keys()).union(cdp_sum.keys()).intersection(wind_sum.keys()))

# x_mean, x_lo, x_hi = [], [], []
# ycas_mean, ycas_lo, ycas_hi = [], [], []
# ycdp_mean, ycdp_lo, ycdp_hi = [], [], []
# months_plot = []

# for m in months_all:
#     if m not in month_name:
#         continue

#     w_mean, w_p10, w_p90 = wind_sum[m]
#     cas_mean, cas_p10, cas_p90 = cas_sum.get(m, (np.nan,np.nan,np.nan))
#     cdp_mean, cdp_p10, cdp_p90 = cdp_sum.get(m, (np.nan,np.nan,np.nan))

#     if not np.isfinite(w_mean):
#         continue
#     if not (np.isfinite(cas_mean) or np.isfinite(cdp_mean)):
#         continue

#     months_plot.append(m)
#     x_mean.append(w_mean); x_lo.append(w_mean - w_p10); x_hi.append(w_p90 - w_mean)
#     ycas_mean.append(cas_mean); ycas_lo.append(cas_mean - cas_p10); ycas_hi.append(cas_p90 - cas_mean)
#     ycdp_mean.append(cdp_mean); ycdp_lo.append(cdp_mean - cdp_p10); ycdp_hi.append(cdp_p90 - cdp_mean)

# x_mean = np.array(x_mean); xerr = np.vstack([x_lo, x_hi])
# ycas_mean = np.array(ycas_mean); ycas_err = np.vstack([ycas_lo, ycas_hi])
# ycdp_mean = np.array(ycdp_mean); ycdp_err = np.vstack([ycdp_lo, ycdp_hi])

# fig, ax = plt.subplots(figsize=(7.5, 5.8))

# for i, m in enumerate(months_plot):
#     c = month_colors.get(m, "k")

#     if np.isfinite(ycas_mean[i]):
#         ax.errorbar(x_mean[i], ycas_mean[i],
#                     xerr=xerr[:, i].reshape(2,1),
#                     yerr=ycas_err[:, i].reshape(2,1),
#                     fmt="s", color=c, ecolor=c, markersize=9,
#                     markeredgewidth=1.5, elinewidth=1.8, capsize=4)

#     if np.isfinite(ycdp_mean[i]):
#         ax.errorbar(x_mean[i], ycdp_mean[i],
#                     xerr=xerr[:, i].reshape(2,1),
#                     yerr=ycdp_err[:, i].reshape(2,1),
#                     fmt="^", color=c, ecolor=c, markersize=10,
#                     markeredgewidth=1.5, elinewidth=1.8, capsize=4)

# legend_handles = [
#     Line2D([0],[0], marker="s", color="k", lw=0, markersize=9, label="CAS monthly mean"),
#     Line2D([0],[0], marker="^", color="k", lw=0, markersize=10, label="CDP monthly mean"),
#     Line2D([0],[0], color="none", lw=0, label=f"Error bars: P{p_lo}–P{p_hi}")
# ]
# for m in months_plot:
#     legend_handles.append(Line2D([0],[0], marker="o", color=month_colors[m],
#                                  lw=0, markersize=7, label=month_name[m]))

# ax.set_yscale("log")
# ax.grid(alpha=0.3)
# ax.set_xlabel("Monthly mean wind speed (m/s)", fontsize=15, fontweight="bold")
# ax.set_ylabel("Monthly mean GCCN concentration (cm⁻³)", fontsize=15, fontweight="bold")
# ax.set_title("BCB January–June 2022\nMonthly GCCN vs Wind Speed", fontsize=15, fontweight="bold")
# ax.legend(handles=legend_handles, ncol=2, fontsize=9, frameon=True, loc="lower right")

# ax.tick_params(axis="both", labelsize=15)
# for lab in ax.get_xticklabels() + ax.get_yticklabels():
#     lab.set_fontweight("bold")

# plt.tight_layout()
# plt.show()
# def prep_df_for_x(dfp):
#     dfp = dfp.copy()
#     dfp["Date_dt"] = pd.to_datetime(dfp["Date"])
#     if "Month" not in dfp.columns:
#         dfp["Month"] = dfp["Date_dt"].dt.month
#     dfp = dfp.sort_values(["Date_dt","BCB_start"]).reset_index(drop=True)
#     x = np.arange(len(dfp))
#     date_first = dfp.groupby(dfp["Date_dt"].dt.date).head(1)
#     tick_pos = date_first.index.to_numpy()
#     tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
#     return dfp, x, tick_pos, tick_lab
# dfp_cas, x_cas, tick_pos, tick_lab = prep_df_for_x(dfp_cas_w)
# dfp_cdp, x_cdp, _, _ = prep_df_for_x(dfp_cdp_w)

# fig_w = max(22, 0.55 * len(tick_pos))
# fig, ax1 = plt.subplots(figsize=(fig_w, 6.2))
# ax2 = ax1.twinx()
# for m in sorted(dfp_cas["Month"].unique()):
#     if m not in month_name: continue
#     mask = (dfp_cas["Month"].values == m)
#     c = month_colors[m]
#     ax1.plot(x_cas[mask], cas_arr[mask], "-", lw=1.5, color=c, alpha=0.9)

# for m in sorted(dfp_cdp["Month"].unique()):
#     if m not in month_name: continue
#     mask = (dfp_cdp["Month"].values == m)
#     c = month_colors[m]
#     ax1.plot(x_cdp[mask], cdp_arr[mask], "--", lw=1.5, color=c, alpha=0.9)
# ax2.plot(x_cdp, wind_cdp, color="k", lw=1.6, alpha=0.75)
# ax1.set_yscale("log")
# ax1.set_ylabel("GCCN concentration (cm⁻³)", fontsize=16, fontweight="bold")
# ax2.set_ylabel("Wind speed (m/s)", fontsize=16, fontweight="bold")
# ax1.grid(alpha=0.3)
# ax1.tick_params(axis="y", labelsize=15)
# ax2.tick_params(axis="y", labelsize=15)
# for lab in ax1.get_yticklabels()+ax2.get_yticklabels():
#     lab.set_fontweight("bold")
# for p in tick_pos:
#     ax1.axvline(p, color="k", alpha=0.06)
# ax1.set_xticks(tick_pos)
# ax1.set_xticklabels(tick_lab, rotation=60, ha="right", fontsize=7, fontweight="bold")
# legend_handles = []
# for m in sorted(set(dfp_cas["Month"].unique()).union(dfp_cdp["Month"].unique())):
#     if m in month_name:
#         legend_handles.append(Line2D([0],[0], color=month_colors[m], lw=2, label=month_name[m]))

# legend_handles.extend([
#     Line2D([0],[0], color="k", lw=2, linestyle="-",  label="CAS GCCN"),
#     Line2D([0],[0], color="k", lw=2, linestyle="--", label="CDP GCCN"),
#     Line2D([0],[0], color="k", lw=2, linestyle="-",  label="Wind speed"),
# ])

# ax1.legend(handles=legend_handles, ncol=3, fontsize=9, loc="lower right", frameon=True)
# ax1.set_xlabel("Flight Date", fontsize=16, fontweight="bold")
# ax1.set_title("BCB January–June 2022\nGCCN with Wind Speed", fontsize=16, fontweight="bold")
# fig.subplots_adjust(bottom=0.40)
# fig.tight_layout()
# plt.show()