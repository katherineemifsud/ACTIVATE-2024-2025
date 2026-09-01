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
    f"{BASE_DIR}/CAS_GCCN_concentration_flight_split_2022.pkl",
    "rb"
) as f:
    cas_conc_data = pickle.load(f)
with open(
    f"{BASE_DIR}/CAS_GCCN_mass_flight_split_2022.pkl",
    "rb"
) as f:
    cas_mass_data = pickle.load(f)
with open(
    f"{BASE_DIR}/CDP_GCCN_concentration_flight_split_2022.pkl",
    "rb"
) as f:
    cdp_conc_data = pickle.load(f)
with open(
    f"{BASE_DIR}/CDP_GCCN_mass_flight_split_2022.pkl",
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
    "CAS",
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
    "CDP",
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
    f"{base_path}/CAS_GCCN_concentration_leg_level_2022.pkl")
cas_mass_file = (f"{base_path}/CAS_GCCN_mass_leg_level_2022.pkl")
cdp_concentration_file = (f"{base_path}/CDP_GCCN_concentration_leg_level_2022.pkl")
cdp_mass_file = (f"{base_path}/CDP_GCCN_mass_leg_level_2022.pkl")
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
with open("CAS_CDP_GCCN_mass_flight_mean_2022.pkl",
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
    "CAS_CDP_GCCN_concentration_flight_mean_2022.pkl", "wb"
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
ax.set_title("Combined CAS and CDP\nJanuary-June 2022",fontsize=18, fontweight="bold")
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
    "Combined CAS and CDP\nJanuary-June 2022",
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
    "CAS vs CDP Mass\nJanuary-June 2022",
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
