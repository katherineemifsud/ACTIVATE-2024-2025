#%%
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import datetime
import pathlib
import statistics
import mputil
import pickle
import shutil
from collections import defaultdict
import glob
import os
import re
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
import matplotlib.colors as mcolors
#%%
BASE_DIR = (
    "/home/disk/p/kathem24/activate/"
    "ACTIVATE-2024-2025/CAS/Below Cloud Base/Scripts")
with open(
    f"{BASE_DIR}/CAS_GCCN_concentration_flight_split_2020.pkl",
    "rb"
) as f:
    cas_conc_data = pickle.load(f)
with open(
    f"{BASE_DIR}/CAS_GCCN_mass_flight_split_2020.pkl",
    "rb"
) as f:
    cas_mass_data = pickle.load(f)
#%%
with open(
    f"{BASE_DIR}/CDP_GCCN_concentration_flight_split_2020.pkl",
    "rb"
) as f:
    cdp_conc_data = pickle.load(f)
with open(
    f"{BASE_DIR}/CDP_GCCN_mass_flight_split_2020.pkl",
    "rb"
) as f:
    cdp_mass_data = pickle.load(f)
print("All four pickle files loaded successfully.")
#%%
CAS_concentration_by_flight = (
    cas_conc_data["average_gccn_per_flight"])
CAS_mass_by_flight = (
    cas_mass_data["average_mass_per_flight"])
CDP_concentration_by_flight = (
    cdp_conc_data["average_gccn_per_flight"])
CDP_mass_by_flight = (
    cdp_mass_data["average_mass_per_flight"])
#%%
print("CAS concentration flights:",
      len(CAS_concentration_by_flight))
print("CAS mass flights:",
      len(CAS_mass_by_flight))
print("CDP concentration flights:",
      len(CDP_concentration_by_flight))
print("CDP mass flights:",
      len(CDP_mass_by_flight))
#%%
CAS_common_dates = sorted(
    set(CAS_mass_by_flight.keys()) &
    set(CAS_concentration_by_flight.keys()))
CAS_dates = []
CAS_mass = []
CAS_concentration = []
for date in CAS_common_dates:
    mass = CAS_mass_by_flight[date]
    concentration = CAS_concentration_by_flight[date]
    if np.isfinite(mass) and np.isfinite(concentration):
        CAS_dates.append(date)
        CAS_mass.append(mass)
        CAS_concentration.append(concentration)
CAS_mass = np.array(CAS_mass)
CAS_concentration = np.array(CAS_concentration)
CDP_common_dates = sorted(
    set(CDP_mass_by_flight.keys()) &
    set(CDP_concentration_by_flight.keys()))
CDP_dates = []
CDP_mass = []
CDP_concentration = []
for date in CDP_common_dates:
    mass = CDP_mass_by_flight[date]
    concentration = CDP_concentration_by_flight[date]
    if np.isfinite(mass) and np.isfinite(concentration):
        CDP_dates.append(date)
        CDP_mass.append(mass)
        CDP_concentration.append(concentration)
CDP_mass = np.array(CDP_mass)
CDP_concentration = np.array(CDP_concentration)
print("Matched CAS flights:", len(CAS_dates))
print("Matched CDP flights:", len(CDP_dates))
# %%
#%%
fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 6))
axes[0].scatter(
    CAS_mass,
    CAS_concentration,
    s=80,
    alpha=0.8,
    edgecolor="black", color='black')
axes[0].set_xlabel(
    "Flight-Mean Mass (µg m⁻³)",
    fontsize=18,
    fontweight="bold")
axes[0].set_ylabel(
    "Flight-Mean Number Concentration\n (cm⁻³)",
    fontsize=18,
    fontweight="bold")
axes[0].set_title(
    "CAS 2020",
    fontsize=18,
    fontweight="bold")
axes[0].tick_params(
    axis="both",
    which="major",
    labelsize=17,
    width=2,
    length=6)
axes[0].grid(
    linestyle="--",
    alpha=0.4)
axes[1].scatter(
    CDP_mass,
    CDP_concentration,
    s=80,
    alpha=0.8,
    edgecolor="blue", color='blue')
axes[1].set_xlabel(
    "Flight-Mean Mass (µg m⁻³)",
    fontsize=18,
    fontweight="bold")
axes[1].set_ylabel(
    "Flight-Mean Number Concentration \n(cm⁻³)",
    fontsize=18,
    fontweight="bold")
axes[1].set_title(
    "CDP 2020",
    fontsize=18,
    fontweight="bold")
axes[1].tick_params(
    axis="both",
    which="major",
    labelsize=17,
    width=2,
    length=6)
axes[1].grid(
    linestyle="--",
    alpha=0.4)
plt.tight_layout()
plt.show()
# %%
#combine CAS and CDP 
base_path = ("/home/disk/p/kathem24/activate/ACTIVATE-2024-2025/"
    "CAS/Below Cloud Base/Scripts")
cas_concentration_file = (
    f"{base_path}/CAS_GCCN_concentration_leg_level_2020.pkl")
cas_mass_file = (f"{base_path}/CAS_GCCN_mass_leg_level_2020.pkl")
cdp_concentration_file = (f"{base_path}/CDP_GCCN_concentration_leg_level_2020.pkl")
cdp_mass_file = (f"{base_path}/CDP_GCCN_mass_leg_level_2020.pkl")
with open(cas_concentration_file, "rb") as f:
    cas_concentration = pickle.load(f)
with open(cas_mass_file, "rb") as f:
    cas_mass = pickle.load(f)
with open(cdp_concentration_file, "rb") as f:
    cdp_concentration = pickle.load(f)
with open(cdp_mass_file, "rb") as f:cdp_mass = pickle.load(f)
print("CAS concentration legs:",len(cas_concentration))
print("CAS mass legs:", len(cas_mass))
print("CDP concentration legs:",len(cdp_concentration))
print("CDP mass legs:", len(cdp_mass))
# %%
# %%
# COMBINED CAS + CDP GCCN MASS
CAS_mass_flight_totals = defaultdict(
    lambda: {
        'Total_GCCN_Mass': 0,
        'Leg_Count': 0})
for entry in cas_mass:
    date = entry['Date']
    total_mass = entry['Dry Mass (µg/m³)']
    if np.isfinite(total_mass):
        CAS_mass_flight_totals[date][
            'Total_GCCN_Mass'
        ] += total_mass
        CAS_mass_flight_totals[date][
            'Leg_Count'
        ] += 1
CAS_average_mass_per_flight = {}
for date, flight_data in CAS_mass_flight_totals.items():
    if flight_data['Leg_Count'] > 0:
        CAS_average_mass_per_flight[date] = (
            flight_data['Total_GCCN_Mass'] /
            flight_data['Leg_Count'])
CDP_mass_flight_totals = defaultdict(
    lambda: {'Total_GCCN_Mass': 0,
        'Leg_Count': 0})
for entry in cdp_mass:
    date = entry['Date']
    total_mass = entry['Dry Mass (µg/m³)']
    if np.isfinite(total_mass):
        CDP_mass_flight_totals[date][
            'Total_GCCN_Mass'
        ] += total_mass
        CDP_mass_flight_totals[date][
            'Leg_Count'
        ] += 1
CDP_average_mass_per_flight = {}
for date, flight_data in CDP_mass_flight_totals.items():
    if flight_data['Leg_Count'] > 0:
        CDP_average_mass_per_flight[date] = (
            flight_data['Total_GCCN_Mass'] /
            flight_data['Leg_Count'])
common_mass_dates = sorted(
    set(CAS_average_mass_per_flight) &
    set(CDP_average_mass_per_flight))
average_mass_per_flight = {}
for date in common_mass_dates:
    CAS_mass_mean = (
        CAS_average_mass_per_flight[date]    )
    CDP_mass_mean = (
        CDP_average_mass_per_flight[date] )
    average_mass_per_flight[date] = np.mean([
        CAS_mass_mean,
        CDP_mass_mean])
print("Combined CAS + CDP Average "
    "GCCN Mass per Flight:")
for date, avg_mass in average_mass_per_flight.items():
    print(f"{date}: "
        f"{avg_mass:.2f} µg/m³")
print("Number of combined mass flights:",
    len(average_mass_per_flight))
# %%
CAS_GCCN_flight_totals = defaultdict(
    lambda: {'Total_GCCN_Concentration': 0,
        'Leg_Count': 0})
for entry in cas_concentration:
    date = entry['Date']
    total_gccn = (entry['Total_GCCN_Concentration'])
    if np.isfinite(total_gccn):
        CAS_GCCN_flight_totals[date][
            'Total_GCCN_Concentration'
        ] += total_gccn
        CAS_GCCN_flight_totals[date][
            'Leg_Count'
        ] += 1
CAS_average_gccn_per_flight = {}
for date, flight_data in CAS_GCCN_flight_totals.items():
    if flight_data['Leg_Count'] > 0:
        CAS_average_gccn_per_flight[date] = (
            flight_data[
                'Total_GCCN_Concentration'
            ] /
            flight_data['Leg_Count'] )
CDP_GCCN_flight_totals = defaultdict(
    lambda: {
        'Total_GCCN_Concentration': 0,
        'Leg_Count': 0})
for entry in cdp_concentration:
    date = entry['Date']
    total_gccn = (
        entry['Total_GCCN_Concentration']    )
    if np.isfinite(total_gccn):
        CDP_GCCN_flight_totals[date][
            'Total_GCCN_Concentration'
        ] += total_gccn
        CDP_GCCN_flight_totals[date][
            'Leg_Count'
        ] += 1
CDP_average_gccn_per_flight = {}
for date, flight_data in CDP_GCCN_flight_totals.items():
    if flight_data['Leg_Count'] > 0:
        CDP_average_gccn_per_flight[date] = (
            flight_data[
                'Total_GCCN_Concentration'
            ] /
            flight_data['Leg_Count'])
common_gccn_dates = sorted(
    set(CAS_average_gccn_per_flight) &
    set(CDP_average_gccn_per_flight))
average_gccn_per_flight = {}
for date in common_gccn_dates:
    CAS_gccn_mean = (
        CAS_average_gccn_per_flight[date])
    CDP_gccn_mean = (
        CDP_average_gccn_per_flight[date])
    average_gccn_per_flight[date] = np.mean([
        CAS_gccn_mean,
        CDP_gccn_mean])
print("Combined CAS + CDP Average GCCN "
    "Concentration per Flight:")
for date, avg_gccn in average_gccn_per_flight.items():
    print(
        f"{date}: "
        f"{avg_gccn:.3f} cm⁻³"  )
print("Number of combined concentration flights:",
    len(average_gccn_per_flight))
#%%
# Save combined CAS + CDP flight-mean GCCN mass
with open("CAS_CDP_GCCN_mass_flight_mean_2020.pkl",
    "wb"
) as f:
    pickle.dump(average_mass_per_flight,
        f)
print("Saved combined CAS + CDP "
    "flight-mean GCCN mass.")
print("Number of flights:",
    len(average_mass_per_flight))
#%%
with open(
    "CAS_CDP_GCCN_concentration_flight_mean_2020.pkl", "wb"
) as f:
    pickle.dump(average_gccn_per_flight, f)
print("Saved combined CAS + CDP "
    "flight-mean GCCN concentration.")
print("Number of flights:", len(average_gccn_per_flight))
# %%
#plot concentration vs mass for combined CAS + CDP flights
common_dates = sorted(
    set(average_mass_per_flight.keys()) &
    set(average_gccn_per_flight.keys()))
print("Number of matched flights:", len(common_dates))
flight_mass = np.array([
    average_mass_per_flight[date]
    for date in common_dates
], dtype=float)
flight_concentration = np.array([
    average_gccn_per_flight[date]
    for date in common_dates
], dtype=float)
valid = (
    np.isfinite(flight_mass) &
    np.isfinite(flight_concentration))
flight_mass = flight_mass[valid]
flight_concentration = flight_concentration[valid]
valid_dates = np.array(common_dates)[valid]
fig, ax = plt.subplots(
    figsize=(7, 6))
ax.scatter(
    flight_mass,
    flight_concentration,
    s=55,
    alpha=0.8,
    edgecolor="purple", color='purple')
ax.set_xlabel("Mean Mass per Flight \n(µg m$^{-3}$)",fontsize=17, fontweight="bold")
ax.set_ylabel("Mean Number Concentration per Flight \n(cm$^{-3}$)", fontsize=17, fontweight="bold")
ax.set_title("Combined CAS and CDP\nFMAS 2020",fontsize=18, fontweight="bold")
ax.tick_params(axis="both", labelsize=17, width=2, length=6)
plt.tight_layout()
plt.show()
# %%
#linear regression
from scipy.stats import linregress
common_dates = sorted(set(average_mass_per_flight.keys()) &
    set(average_gccn_per_flight.keys()))
print("Number of matched flights:", len(common_dates))
flight_mass = np.array([
    average_mass_per_flight[date]
    for date in common_dates], dtype=float)
flight_concentration = np.array([
    average_gccn_per_flight[date]
    for date in common_dates], dtype=float)
valid = (np.isfinite(flight_mass) &
    np.isfinite(flight_concentration))
flight_mass = flight_mass[valid]
flight_concentration = flight_concentration[valid]
valid_dates = np.array(common_dates)[valid]
regression = linregress(
    flight_mass,
    flight_concentration)
slope = regression.slope
intercept = regression.intercept
r_value = regression.rvalue
r_squared = r_value**2
p_value = regression.pvalue
std_err = regression.stderr
print("\nLinear regression results:")
print(f"Slope: {slope:.5f}")
print(f"Intercept: {intercept:.5f}")
print(f"R: {r_value:.3f}")
print(f"R²: {r_squared:.3f}")
print(f"p-value: {p_value:.4g}")
print(f"Slope standard error: {std_err:.5f}")
x_fit = np.linspace(
    np.min(flight_mass),
    np.max(flight_mass),
    200)
y_fit = (
    slope * x_fit +
    intercept)
fig, ax = plt.subplots(
    figsize=(7, 6))
ax.scatter(
    flight_mass,
    flight_concentration,
    s=55,
    alpha=0.8,
    edgecolor="purple",
    color="purple")
ax.plot( x_fit,
    y_fit,
    color="black",
    linewidth=2)
ax.text(
    0.05,
    0.95,
    f"$R^2$ = {r_squared:.2f}\n"
    f"$p$ = {p_value:.3g}",
    transform=ax.transAxes,
    fontsize=14,
    verticalalignment="top")
ax.set_xlabel(
    "Mean Mass per Flight \n(µg m$^{-3}$)",
    fontsize=17,
    fontweight="bold")
ax.set_ylabel(
    "Mean Number Concentration per Flight \n(cm$^{-3}$)",
    fontsize=17,
    fontweight="bold")
ax.set_title(
    "Combined CAS and CDP\nFMAS 2020",
    fontsize=18,
    fontweight="bold")
ax.tick_params(
    axis="both",
    labelsize=17,
    width=2,
    length=6)
plt.tight_layout()
plt.show()
# %%
#CAS mass against CDP mass
common_mass_dates = sorted(
    set(CAS_average_mass_per_flight.keys()) &
    set(CDP_average_mass_per_flight.keys()))
print("Number of matched flights:", len(common_mass_dates))
CAS_flight_mass = np.array([
    CAS_average_mass_per_flight[date]
    for date in common_mass_dates], dtype=float)
CDP_flight_mass = np.array([
    CDP_average_mass_per_flight[date]
    for date in common_mass_dates
], dtype=float)
valid = (
    np.isfinite(CAS_flight_mass) &
    np.isfinite(CDP_flight_mass))
CAS_flight_mass = CAS_flight_mass[valid]
CDP_flight_mass = CDP_flight_mass[valid]
valid_dates = np.array(common_mass_dates)[valid]
print("Number of valid matched flights:", len(valid_dates))
fig, ax = plt.subplots(
    figsize=(7, 6))
ax.scatter(
    CAS_flight_mass,
    CDP_flight_mass,
    s=55,
    alpha=0.8,
    edgecolor="purple",
    color="purple")
ax.set_xlabel(
    "CAS Mean Mass per Flight\n(µg m$^{-3}$)",
    fontsize=17,
    fontweight="bold")
ax.set_ylabel(
    "CDP Mean Mass per Flight\n(µg m$^{-3}$)",
    fontsize=17,
    fontweight="bold")
ax.set_title(
    "CAS vs CDP Mass\nFMAS 2020",
    fontsize=18,
    fontweight="bold")
ax.tick_params(
    axis="both",
    labelsize=17,
    width=2,
    length=6)
plt.tight_layout()
plt.show()
# %%
#linear regression for CAS mass vs CDP mass
from scipy.stats import linregress
common_mass_dates = sorted(
    set(CAS_average_mass_per_flight.keys()) &
    set(CDP_average_mass_per_flight.keys()))
print("Number of matched flights:", len(common_mass_dates))
CAS_flight_mass = np.array([
    CAS_average_mass_per_flight[date]
    for date in common_mass_dates], dtype=float)
CDP_flight_mass = np.array([
    CDP_average_mass_per_flight[date]
    for date in common_mass_dates], dtype=float)
valid = (
    np.isfinite(CAS_flight_mass) &
    np.isfinite(CDP_flight_mass))
CAS_flight_mass = CAS_flight_mass[valid]
CDP_flight_mass = CDP_flight_mass[valid]
valid_dates = np.array(common_mass_dates)[valid]
regression = linregress(
    CAS_flight_mass,
    CDP_flight_mass)
slope = regression.slope
intercept = regression.intercept
r_value = regression.rvalue
r_squared = r_value**2
p_value = regression.pvalue
std_err = regression.stderr
print("\nLinear regression results:")
print(f"Slope: {slope:.5f}")
print(f"Intercept: {intercept:.5f}")
print(f"R: {r_value:.3f}")
print(f"R²: {r_squared:.3f}")
print(f"p-value: {p_value:.4g}")    
# %%
#variabilty across flights 
def calculate_flight_variability(
    leg_data,
    value_key
):
    """
    For each flight:
      mean = mean across BCB legs
      s = standard deviation across BCB legs
      relative_s = 100 * s / mean

    Uses sample standard deviation (ddof=1).
    """
    values_by_flight = defaultdict(list)

    for entry in leg_data:

        date = entry["Date"]
        value = entry[value_key]

        if np.isfinite(value):
            values_by_flight[date].append(value)

    flight_stats = {}

    for date, values in values_by_flight.items():

        values = np.asarray(
            values,
            dtype=float        )
        mean_value = np.mean(values)
        if len(values) > 1:
            s = np.std(
                values,
                ddof=1
            )
        else:
            s = np.nan

        if (
            np.isfinite(s)
            and mean_value != 0
        ):
            relative_s = (
                100 * s / mean_value
            )
        else:
            relative_s = np.nan

        flight_stats[date] = {
            "mean": mean_value,
            "s": s,
            "relative_s_percent": relative_s,
            "n_legs": len(values)        }
    return flight_stats
# %%
CAS_mass_stats = calculate_flight_variability(
    cas_mass,
    "Dry Mass (µg/m³)"
)

CDP_mass_stats = calculate_flight_variability(
    cdp_mass,
    "Dry Mass (µg/m³)"
)

CAS_concentration_stats = calculate_flight_variability(
    cas_concentration,
    "Total_GCCN_Concentration"
)

CDP_concentration_stats = calculate_flight_variability(
    cdp_concentration,
    "Total_GCCN_Concentration"
)
# %%
def summarize_variability(
    name,
    flight_stats
):

    means = np.array([
        stats["mean"]
        for stats in flight_stats.values()
        if np.isfinite(stats["mean"])
    ])

    s_values = np.array([
        stats["s"]
        for stats in flight_stats.values()
        if np.isfinite(stats["s"])
    ])

    relative_s = np.array([
        stats["relative_s_percent"]
        for stats in flight_stats.values()
        if np.isfinite(
            stats["relative_s_percent"]
        )
    ])
    median_s = np.median(s_values)
    median_relative_s = np.median(
        relative_s
    )
    across_flight_s = np.std(
        means,
        ddof=1    )
    ratio = (
        100 * median_s /
        across_flight_s    )

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print(
        "Number of flight means:",
        len(means)    )

    print(
        "Flights with >=2 BCB legs:",
        len(s_values)    )

    print(
        f"Median within-flight s: "
        f"{median_s:.3f}"    )

    print(
        f"Median s/mean: "
        f"{median_relative_s:.1f}%"    )

    print(
        f"SD of flight means across dataset: "
        f"{across_flight_s:.3f}"    )

    print(
        f"Median within-flight s / "
        f"across-flight SD: "
        f"{ratio:.1f}%"    )

    print(
        f"Range of flight means: "
        f"{np.min(means):.3f} to "
        f"{np.max(means):.3f}"    )
# %%
summarize_variability(
    "CAS GCCN Mass",
    CAS_mass_stats)

summarize_variability(
    "CDP GCCN Mass",
    CDP_mass_stats)

summarize_variability(
    "CAS GCCN Number Concentration",
    CAS_concentration_stats)

summarize_variability(
    "CDP GCCN Number Concentration",
    CDP_concentration_stats)
# %%
def prepare_CAS_CDP_arrays(
    CAS_stats,
    CDP_stats):
    common_dates = sorted(
        set(CAS_stats.keys()) &
        set(CDP_stats.keys())    )
    CAS_mean = np.array([
        CAS_stats[date]["mean"]
        for date in common_dates    ])
    CDP_mean = np.array([
        CDP_stats[date]["mean"]
        for date in common_dates    ])
    CAS_s = np.array([
        CAS_stats[date]["s"]
        for date in common_dates    ])
    CDP_s = np.array([
        CDP_stats[date]["s"]
        for date in common_dates    ])
    return (        common_dates,
        CAS_mean,
        CDP_mean,
        CAS_s,
        CDP_s    )
# %%
(    mass_dates,
    CAS_mass_mean,
    CDP_mass_mean,
    CAS_mass_s,
    CDP_mass_s
) = prepare_CAS_CDP_arrays(
    CAS_mass_stats,
    CDP_mass_stats)
# %%
(    concentration_dates,
    CAS_concentration_mean,
    CDP_concentration_mean,
    CAS_concentration_s,
    CDP_concentration_s
) = prepare_CAS_CDP_arrays(
    CAS_concentration_stats,
    CDP_concentration_stats)
# %%
fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 6))
axes[0].scatter(
    CAS_mass_mean,
    CDP_mass_mean,
    s=70,
    color="purple",
    alpha=0.8,
    zorder=3)
valid_mass_s = (
    np.isfinite(CAS_mass_s) &
    np.isfinite(CDP_mass_s))
axes[0].errorbar(
    CAS_mass_mean[valid_mass_s],
    CDP_mass_mean[valid_mass_s],
    xerr=CAS_mass_s[valid_mass_s],
    yerr=CDP_mass_s[valid_mass_s],
    fmt="none",
    ecolor="gray",
    elinewidth=1.0,
    capsize=2,
    alpha=0.4,
    zorder=2
)
axes[0].set_xlabel(
    r"CAS Flight-Mean $M_{d>2}$"
    "\n($\mu$g m$^{-3}$)",
    fontsize=17,
    fontweight="bold")
axes[0].set_ylabel(
    r"CDP Flight-Mean $M_{d>2}$"
    "\n($\mu$g m$^{-3}$)",
    fontsize=17,
    fontweight="bold")
axes[0].set_title(
    "(a) GCCN Mass",
    fontsize=18,
    fontweight="bold")
axes[1].scatter(
    CAS_concentration_mean,
    CDP_concentration_mean,
    s=70,
    color="purple",
    alpha=0.8,
    zorder=3)
valid_concentration_s = (
    np.isfinite(CAS_concentration_s) &
    np.isfinite(CDP_concentration_s))
axes[1].errorbar(
    CAS_concentration_mean[
        valid_concentration_s    ],
    CDP_concentration_mean[
        valid_concentration_s    ],
    xerr=CAS_concentration_s[
        valid_concentration_s    ],
    yerr=CDP_concentration_s[
        valid_concentration_s    ],
    fmt="none",
    ecolor="gray",
    elinewidth=1.0,
    capsize=2,
    alpha=0.4,
    zorder=2
)
axes[1].set_xlabel(
    r"CAS Flight-Mean $N_{d>2}$"
    "\n(cm$^{-3}$)",
    fontsize=17,
    fontweight="bold")
axes[1].set_ylabel(
    r"CDP Flight-Mean $N_{d>2}$"
    "\n(cm$^{-3}$)",
    fontsize=17,
    fontweight="bold")
axes[1].set_title(
    "(b) GCCN Number Concentration",
    fontsize=18,
    fontweight="bold")
for ax in axes:
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=15,
        width=2,
        length=6    )
    axes[0].set_xlim(0, 23)
axes[0].set_ylim(0, 55)
axes[1].set_xlim(0, 2.4)
axes[1].set_ylim(0, 1.1)
fig.suptitle(
    "In-Flight GCCN Variability\n"
    "FMAS 2020",
    fontsize=19,
    fontweight="bold")
plt.tight_layout()
plt.show()
# %%
#%%
# WITHIN-FLIGHT VARIABILITY OF COMBINED CAS + CDP GCCN MASS
cdp_mass_lookup = {}
for entry in cdp_mass:
    key = (
        entry['Date'],
        entry['BCB_start'],
        entry['BCB_stop']    )
    mass = entry['Dry Mass (µg/m³)']
    if np.isfinite(mass):
        cdp_mass_lookup[key] = mass
combined_mass_by_flight = defaultdict(list)
for entry in cas_mass:
    key = (
        entry['Date'],
        entry['BCB_start'],
        entry['BCB_stop']    )
    cas_value = entry['Dry Mass (µg/m³)']
    if (
        key in cdp_mass_lookup
        and np.isfinite(cas_value)    ):
        cdp_value = cdp_mass_lookup[key]
        combined_leg_mass = np.mean([
            cas_value,
            cdp_value        ])
        date = entry['Date']
        combined_mass_by_flight[date].append(
            combined_leg_mass        )
mass_std_per_flight = {}
for date, leg_values in combined_mass_by_flight.items():
    leg_values = np.array(
        leg_values,
        dtype=float    )
    if len(leg_values) >= 2:
        mass_std_per_flight[date] = np.std(
            leg_values,
            ddof=1        )
print(
    "Flights with combined CAS+CDP "
    "within-flight mass SD:",
    len(mass_std_per_flight))
# %%
#%%
print("\n--- CHECK COMBINED FLIGHT MEANS ---")
for date, leg_values in combined_mass_by_flight.items():
    if date in average_mass_per_flight:
        reconstructed_mean = np.mean(
            leg_values        )
        saved_mean = (
            average_mass_per_flight[date]        )
        difference = (
            reconstructed_mean -
            saved_mean        )
        print(
            date,
            f"saved = {saved_mean:.3f}",
            f"matched-leg = {reconstructed_mean:.3f}",
            f"difference = {difference:.3f}"        )
# %%
#%%
avg_high_mass = np.mean(high_mass_values)
avg_low_mass = np.mean(low_mass_values)
mass_high_low_difference = (
    avg_high_mass - avg_low_mass)
mass_std_values = np.array(
    list(mass_std_per_flight.values()),
    dtype=float)
median_mass_std = np.median(mass_std_values)
mean_mass_std = np.mean(mass_std_values)
mass_separation_ratio = (
    mass_high_low_difference /
    median_mass_std)
print("\n--- GCCN MASS COMPARISON ---")
print(
    f"Mean High GCCN mass: "
    f"{avg_high_mass:.2f} µg/m³")
print(
    f"Mean Low GCCN mass: "
    f"{avg_low_mass:.2f} µg/m³")

print(
    f"Difference between High and Low means: "
    f"{mass_high_low_difference:.2f} µg/m³")
print(
    f"Median within-flight SD: "
    f"{median_mass_std:.2f} µg/m³")
print(
    f"Mean within-flight SD: "
    f"{mean_mass_std:.2f} µg/m³")

print(
    f"High-Low difference / median within-flight SD: "
    f"{mass_separation_ratio:.2f}")
# %%
#%%
cdp_conc_lookup = {}
for entry in cdp_concentration:
    key = (
        entry['Date'],
        entry['BCB_start'],
        entry['BCB_stop']    )
    conc = entry['Total_GCCN_Concentration']
    if np.isfinite(conc):
        cdp_conc_lookup[key] = conc
combined_conc_by_flight = defaultdict(list)
for entry in cas_concentration:
    key = (
        entry['Date'],
        entry['BCB_start'],
        entry['BCB_stop']    )
    cas_value = entry['Total_GCCN_Concentration']
    if (
        key in cdp_conc_lookup
        and np.isfinite(cas_value)
    ):

        cdp_value = cdp_conc_lookup[key]

        combined_leg_conc = np.mean([
            cas_value,
            cdp_value        ])
        date = entry['Date']
        combined_conc_by_flight[date].append(
            combined_leg_conc        )
conc_std_per_flight = {}
for date, leg_values in combined_conc_by_flight.items():
    leg_values = np.array(
        leg_values,
        dtype=float    )
    if len(leg_values) >= 2:
        conc_std_per_flight[date] = np.std(
            leg_values,
            ddof=1        )
print(
    "Flights with combined CAS+CDP "
    "within-flight concentration SD:",
    len(conc_std_per_flight))
# %%
gccn_values = np.array(
    list(average_gccn_per_flight.values()))
gccn_threshold = np.percentile(
    gccn_values,
    50)
high_gccn_flights = {}
low_gccn_flights = {}
for date, avg_gccn in average_gccn_per_flight.items():
    if avg_gccn >= gccn_threshold:
        high_gccn_flights[date] = avg_gccn
    else:
        low_gccn_flights[date] = avg_gccn
high_gccn_values = np.array(
    list(high_gccn_flights.values()))
low_gccn_values = np.array(
    list(low_gccn_flights.values()))
avg_high_gccn = np.mean(
    high_gccn_values)
avg_low_gccn = np.mean(
    low_gccn_values)
gccn_high_low_difference = (
    avg_high_gccn -
    avg_low_gccn)
conc_std_values = np.array(
    list(conc_std_per_flight.values()),
    dtype=float)
median_conc_std = np.median(
    conc_std_values)
mean_conc_std = np.mean(
    conc_std_values)
conc_separation_ratio = (
    gccn_high_low_difference /
    median_conc_std)
print("\n--- GCCN NUMBER CONCENTRATION COMPARISON ---")
print(
    f"Mean High GCCN concentration: "
    f"{avg_high_gccn:.3f} cm⁻³")
print(
    f"Mean Low GCCN concentration: "
    f"{avg_low_gccn:.3f} cm⁻³")
print(
    f"Difference between High and Low means: "
    f"{gccn_high_low_difference:.3f} cm⁻³")
print(
    f"Median within-flight SD: "
    f"{median_conc_std:.3f} cm⁻³")
print(
    f"Mean within-flight SD: "
    f"{mean_conc_std:.3f} cm⁻³")
print(
    f"High-Low difference / median within-flight SD: "
    f"{conc_separation_ratio:.2f}")
# %%
