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
with open("CAS_3panel_seasonaltrend_FMAS2020.pkl", "rb") as f:
    cas_plot_data = pickle.load(f)

dfp = cas_plot_data["dfp"]
x = cas_plot_data["x"]
mass_arr = cas_plot_data["mass_arr"]
slope_arr = cas_plot_data["slope_arr"]
conc_arr = cas_plot_data["conc_arr"]
tick_pos = cas_plot_data["tick_pos"]
tick_lab = cas_plot_data["tick_lab"]
month_name = cas_plot_data["month_name"]
# %%
fig, axes = plt.subplots(3, 1, figsize=(18, 9), sharex=True)
plot_info = [
    (axes[0], mass_arr, "GCCN Mass\n(µg m$^{-3}$)", 
     "CAS GCCN mass seasonal trend over the Western Atlantic", True),
    (axes[1], slope_arr, "Slope Parameter D\n(µm)", 
     "CAS GCCN slope parameter seasonal trend over the Western Atlantic", False),
    (axes[2], conc_arr, "Total GCCN Concentration\n(cm$^{-3}$)", 
     "CAS total GCCN concentration seasonal trend over the Western Atlantic", True),
]
legend_handles = []
for panel_i, (ax, arr, ylabel, title, logscale) in enumerate(plot_info):
    for m in sorted(dfp["Month"].unique()):
        if m not in month_name:
            continue
        m_mask = dfp["Month"].values == m
        line, = ax.plot(
            x[m_mask],
            arr[m_mask],
            "-o",
            linewidth=1.5,
            markersize=3
        )
        c = line.get_color()
        monthly_mean = np.nanmean(arr[m_mask])
        monthly_median = np.nanmedian(arr[m_mask])
        monthly_std = np.nanstd(arr[m_mask])
        mean_x = x[m_mask][len(x[m_mask]) // 2]

        ax.plot(mean_x, monthly_median, marker="s", color="black",
                markersize=9, linestyle="None")

        ax.plot(mean_x + 1, monthly_mean, marker="^", color="black",
                markersize=9, linestyle="None")

        ax.errorbar(mean_x + 1, monthly_mean, yerr=monthly_std,
                    fmt="none", ecolor="black", elinewidth=1.2, capsize=3)

        if panel_i == 0:
            legend_handles.append(Line2D([0], [0], color=c, lw=2, label=month_name[m]))

    for p in tick_pos:
        ax.axvline(p, color="k", alpha=0.06, linewidth=1)
    if logscale:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
axes[-1].set_xlabel("Flight Date", fontsize=15, fontweight="bold")
axes[-1].set_xticks(tick_pos)
axes[-1].set_xticklabels(tick_lab, rotation=60, ha="right",
                         fontsize=9, fontweight="bold")
legend_handles.extend([
    Line2D([0], [0], marker="s", color="black", lw=0, markersize=9, label="Monthly median"),
    Line2D([0], [0], marker="^", color="black", lw=0, markersize=9, label="Monthly mean"),
    Line2D([0], [0], color="black", lw=1.2, label="Monthly mean ± std"),
])
fig.legend(handles=legend_handles, loc="center left",
           bbox_to_anchor=(0.91, 0.5), fontsize=11, frameon=True)
fig.subplots_adjust(right=0.88, hspace=0.25, bottom=0.15)
plt.show()
# %%
df_dry_mass_inf = pd.read_csv("Dry_mass_BCB2020_lessthan100massREAL_CDP.csv")
print("Total number of legs:", len(df_dry_mass_inf))
print(df_dry_mass_inf.columns)
df_mass = pd.read_csv("Dry_mass_BCB2020_lessthan100massREAL_CDP.csv")
print("Total number of legs:", len(df_mass))
# %%
# CDP BCB mass monthly trend
month_name = {
    2: "February",
    3: "March",
    8: "August",
    9: "September"
}
df_cdp = df_mass.copy()
df_cdp = df_cdp[df_cdp["Date"].astype(str).str.startswith("2020-")].copy()
df_cdp["Month"] = df_cdp["Date"].astype(str).str[5:7].astype(int)
df_cdp = df_cdp[df_cdp["Month"].isin([2, 3, 8, 9])].copy()
df_cdp_sorted = df_cdp.sort_values(
    ["Date", "BCB_start"],
    kind="mergesort"
).reset_index(drop=True)
dfp_cdp = df_cdp_sorted.copy()
dfp_cdp["Date_dt"] = pd.to_datetime(dfp_cdp["Date"])
sort_cols = ["Date_dt"]
if "BCB_start" in dfp_cdp.columns:
    sort_cols.append("BCB_start")
dfp_cdp = dfp_cdp.sort_values(sort_cols).reset_index(drop=True)
x = np.arange(len(dfp_cdp))
mass_arr = pd.to_numeric(dfp_cdp["Dry Mass (µg/m³)"], errors="coerce").to_numpy(dtype=float)
date_first = dfp_cdp.groupby(dfp_cdp["Date_dt"].dt.date).head(1)
tick_pos = date_first.index.to_numpy()
tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
fig_w = max(18, 0.9 * len(tick_pos))
fig, ax = plt.subplots(figsize=(fig_w, 6.2))
legend_handles = []
for m in sorted(dfp_cdp["Month"].unique()):
    if m not in month_name:
        continue
    m_mask = dfp_cdp["Month"].values == m
    mean_mass = np.nanmean(mass_arr[m_mask])
    median_mass = np.nanmedian(mass_arr[m_mask])
    line, = ax.plot(
        x[m_mask],
        mass_arr[m_mask],
        '-',
        linewidth=1.5
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
ax.set_title("CDP BCB Mass FMAS 2020",
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
# %%
#slope
df_cdp = df_cdp_sorted.copy()
dfp_cdp["Date_dt"] = pd.to_datetime(dfp_cdp["Date"])
sort_cols = ["Date_dt"]
if "BCB_start" in dfp_cdp.columns:
    sort_cols.append("BCB_start")
dfp_cdp = dfp_cdp.sort_values(sort_cols).reset_index(drop=True)
x = np.arange(len(dfp_cdp))
slope_arr = pd.to_numeric(dfp_cdp["Dry Slope (D)"], errors="coerce").to_numpy(dtype=float)
date_first = dfp_cdp.groupby(dfp_cdp["Date_dt"].dt.date).head(1)
tick_pos = date_first.index.to_numpy()
tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
fig_w = max(18, 0.9 * len(tick_pos))
fig, ax = plt.subplots(figsize=(fig_w, 6.2))
legend_handles = []
for m in sorted(dfp_cdp["Month"].unique()):
    if m not in month_name:
        continue
    m_mask = dfp_cdp["Month"].values == m
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
ax.set_title("CDP BCB Slope FMAS 2020",
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
#concentration
with open("CDP_concentration_massLE1002020.pkl", "rb") as f:
    total_concentration_cm3_mass100gone = pickle.load(f)

print("Total CDP concentration legs:", len(total_concentration_cm3_mass100gone))
# %%
month_name = {
    2: "February",
    3: "March",
    8: "August",
    9: "September"
}

dfp_cdp = pd.DataFrame(total_concentration_cm3_mass100gone).copy()

dfp_cdp = dfp_cdp[dfp_cdp["Date"].astype(str).str.startswith("2020-")].copy()
dfp_cdp["Month"] = dfp_cdp["Date"].astype(str).str[5:7].astype(int)
dfp_cdp = dfp_cdp[dfp_cdp["Month"].isin([2, 3, 8, 9])].copy()

dfp_cdp["Date_dt"] = pd.to_datetime(dfp_cdp["Date"])

sort_cols = ["Date_dt"]
if "BCB_start" in dfp_cdp.columns:
    sort_cols.append("BCB_start")

dfp_cdp = dfp_cdp.sort_values(sort_cols).reset_index(drop=True)

x = np.arange(len(dfp_cdp))

conc_arr = pd.to_numeric(
    dfp_cdp["Total_Y_Concentration_cm3"],
    errors="coerce"
).to_numpy(dtype=float)

date_first = dfp_cdp.groupby(dfp_cdp["Date_dt"].dt.date).head(1)
tick_pos = date_first.index.to_numpy()
tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()

fig_w = max(18, 0.9 * len(tick_pos))
fig, ax = plt.subplots(figsize=(fig_w, 6.2))

legend_handles = []

for m in sorted(dfp_cdp["Month"].unique()):
    if m not in month_name:
        continue

    m_mask = dfp_cdp["Month"].values == m

    mean_conc = np.nanmean(conc_arr[m_mask])
    median_conc = np.nanmedian(conc_arr[m_mask])

    line, = ax.plot(
        x[m_mask],
        conc_arr[m_mask],
        "-",
        linewidth=1.5
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
ax.set_yscale("log")
plt.yticks(fontsize=16, fontweight="bold")
ax.grid(alpha=0.3)
ax.set_ylabel("GCCN Concentration (cm⁻³)", fontsize=20, fontweight="bold")
ax.set_xlabel("Flight Date", fontsize=20, fontweight="bold")
ax.set_title("CDP BCB Concentration FMAS 2020",
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
# %%
# CDP 3-panel seasonal trend
df_mass_slope = df_mass.copy()
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
        "ylabel": "GCCN Mass\n(µg m$^{-3}$)",
        "title": "CDP GCCN mass seasonal trend over the Western Atlantic",
        "log": True,
        "fmt": "{:.2f}"
    },
    {
        "ax": axes[1],
        "data": slope_arr,
        "ylabel": "Slope Parameter D\n(µm)",
        "title": "CDP GCCN slope parameter seasonal trend over the Western Atlantic",
        "log": False,
        "fmt": "{:.2f}"
    },
    {
        "ax": axes[2],
        "data": conc_arr,
        "ylabel": "Total GCCN Concentration\n(cm$^{-3}$)",
        "title": "CDP total GCCN concentration seasonal trend over the Western Atlantic",
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
        monthly_std = np.nanstd(arr[m_mask])
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
cdp_plot_data = {
    "dfp": dfp,
    "x": x,
    "mass_arr": mass_arr,
    "slope_arr": slope_arr,
    "conc_arr": conc_arr,
    "tick_pos": tick_pos,
    "tick_lab": tick_lab,
    "month_name": month_name,
}
with open("CDP_3panel_seasonaltrend_FMAS2020.pkl", "wb") as f:
    pickle.dump(cdp_plot_data, f)
print("Saved CDP 3-panel plot data.")
# %%
#plot CDP and CAS together 

# %%
