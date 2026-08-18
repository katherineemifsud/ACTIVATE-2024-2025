#%%
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pathlib
import statistics
import glob
import os
import re
import math
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde, linregress
import pickle
from scipy.integrate import quad
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D
from collections import Counter
from scipy.spatial import distance
#%%
#load altitudes and CDP concentration 
BASE_DIR = (
    "/home/disk/p/kathem24/activate/"
    "ACTIVATE-2024-2025/CDP/below cloud base")
with open(
    f"{BASE_DIR}/BCB_altitude_per_leg_2022.pkl",
    "rb"
) as f:
    altitude_per_leg = pickle.load(f)
with open(
    f"{BASE_DIR}/CDP_concentration_uncertainty_massLE1002022.pkl",
    "rb"
) as f:
    CDP_concentration_uncertainty_massLE100 = pickle.load(f)
# %%
altitude_lookup = {
    (entry["Date"],
        int(entry["BCB_start"]),
        int(entry["BCB_stop"])
    ): entry["Mean_Altitude_m"]
    for entry in altitude_per_leg}
# %%
matched_altitude_concentration = []
for concentration_entry in (
    CDP_concentration_uncertainty_massLE100):
    leg_key = (
        concentration_entry["Date"],
        int(concentration_entry["BCB_start"]),
        int(concentration_entry["BCB_stop"]) )
    if leg_key not in altitude_lookup:
        continue
    matched_altitude_concentration.append({
        "Date": concentration_entry["Date"],
        "BCB_start": int(
            concentration_entry["BCB_start"] ),
        "BCB_stop": int(
            concentration_entry["BCB_stop"] ),
        "Mean_Altitude_m": altitude_lookup[leg_key],
        "Total_Y_Concentration_cm3": (
            concentration_entry[
                "Total_Y_Concentration_cm3"
            ])})
print("Number of concentration legs:",
    len(CDP_concentration_uncertainty_massLE100))
print("Number successfully matched:",
    len(matched_altitude_concentration))
print("\nFirst matched entry:")
print(matched_altitude_concentration[0])
# %%
altitudes = np.asarray([
    entry["Mean_Altitude_m"]
    for entry in matched_altitude_concentration
], dtype=float)
concentrations = np.asarray([
    entry["Total_Y_Concentration_cm3"]
    for entry in matched_altitude_concentration
], dtype=float)
valid = (np.isfinite(altitudes) &
    np.isfinite(concentrations))
CDP_altitudes = altitudes.copy()
CDP_concentrations = concentrations.copy()
CDP_valid = valid.copy()
plt.figure(figsize=(10, 8))
plt.scatter(altitudes[valid],
    concentrations[valid],
    edgecolor="black",
    alpha=0.7)
plt.xlabel("Mean altitude (m)",
    fontsize=16,
    fontweight="bold")
plt.ylabel("Total GCCN concentration (cm$^{-3}$)",
    fontsize=16,
    fontweight="bold")
plt.xticks(fontsize=14,
    fontweight="bold")
plt.yticks(fontsize=14,
    fontweight="bold")
plt.title(
    "CDP GCCN concentration versus BCB altitude\n"
    "January–June 2022",
    fontsize=18,
    fontweight="bold")
plt.tight_layout()
plt.show()
# %%
#load altitudes and CAS concentration 
ALTITUDE_DIR = (
    "/home/disk/p/kathem24/activate/"
    "ACTIVATE-2024-2025/CDP/below cloud base")
CAS_DIR = (
    "/home/disk/p/kathem24/activate/"
    "ACTIVATE-2024-2025/CDP/below cloud base")
altitude_file = (
    f"{ALTITUDE_DIR}/BCB_altitude_per_leg_2022.pkl")
cas_concentration_file = (
    f"{CAS_DIR}/CAS_concentration_uncertainty_massLE1002022.pkl")
print("Altitude file exists:", os.path.exists(altitude_file))
print("CAS file exists:", os.path.exists(cas_concentration_file))
with open(altitude_file, "rb") as f:
    altitude_per_leg = pickle.load(f)
with open(cas_concentration_file, "rb") as f:
    CAS_concentration_uncertainty_massLE100 = pickle.load(f)
print("Altitude legs:", len(altitude_per_leg))
# %%
altitude_lookup = {
    (entry["Date"],
        int(entry["BCB_start"]),
        int(entry["BCB_stop"])
    ): entry["Mean_Altitude_m"]
    for entry in altitude_per_leg}
# %%
matched_altitude_concentration = []
for concentration_entry in (
    CAS_concentration_uncertainty_massLE100):
    leg_key = (
        concentration_entry["Date"],
        int(concentration_entry["BCB_start"]),
        int(concentration_entry["BCB_stop"]) )
    if leg_key not in altitude_lookup:
        continue
    matched_altitude_concentration.append({
        "Date": concentration_entry["Date"],
        "BCB_start": int(
            concentration_entry["BCB_start"] ),
        "BCB_stop": int(
            concentration_entry["BCB_stop"] ),
        "Mean_Altitude_m": altitude_lookup[leg_key],
        "Total_Y_Concentration_cm3": (
            concentration_entry[
                "Total_Y_Concentration_cm3"
            ])})
print("Number of concentration legs:",
    len(CAS_concentration_uncertainty_massLE100))
print("Number successfully matched:",
    len(matched_altitude_concentration))
print("\nFirst matched entry:")
print(matched_altitude_concentration[0])
# %%
altitudes = np.asarray([
    entry["Mean_Altitude_m"]
    for entry in matched_altitude_concentration
], dtype=float)
concentrations = np.asarray([
    entry["Total_Y_Concentration_cm3"]
    for entry in matched_altitude_concentration
], dtype=float)
valid = (np.isfinite(altitudes) &
    np.isfinite(concentrations))
CAS_altitudes = altitudes.copy()
CAS_concentrations = concentrations.copy()
CAS_valid = valid.copy()
plt.figure(figsize=(10, 8))
plt.scatter(altitudes[valid],
    concentrations[valid],
    edgecolor="black",
    alpha=0.7)
plt.xlabel("Mean altitude (m)",
    fontsize=16,
    fontweight="bold")
plt.ylabel("Total GCCN concentration (cm$^{-3}$)",
    fontsize=16,
    fontweight="bold")
plt.xticks(fontsize=14,
    fontweight="bold")
plt.yticks(fontsize=14,
    fontweight="bold")
plt.title(
    "CAS GCCN concentration versus BCB altitude\n"
    "January–June 2022",
    fontsize=18,
    fontweight="bold")
plt.tight_layout()
plt.show()
# %%
#combined CDP and CAS scatterplot with altitude
plt.figure(figsize=(10, 8))
plt.scatter(CAS_altitudes[CAS_valid],
    CAS_concentrations[CAS_valid],
    color="black",
    edgecolor="black",
    alpha=0.7,
    label="CAS")
plt.scatter(CDP_altitudes[CDP_valid],
    CDP_concentrations[CDP_valid],
    color="blue",
    edgecolor="black",
    alpha=0.7,
    label="CDP")
plt.xlabel("Mean altitude (m)",
    fontsize=20,
    fontweight="bold")
plt.ylabel("Total GCCN concentration (cm$^{-3}$)",
    fontsize=20,
    fontweight="bold")
plt.xticks(fontsize=16,
    fontweight="bold")
plt.yticks(fontsize=16,
    fontweight="bold")
plt.title("GCCN concentration versus BCB altitude\n"
    "January–June 2022",
    fontsize=20,
    fontweight="bold")
plt.legend(fontsize=20,
    frameon=False)
plt.tight_layout()
plt.show()
# %%
#equation for fits 
from scipy.stats import linregress
# %%
# Linear fits of GCCN concentration versus altitude
CAS_altitude_km = (
    CAS_altitudes[CAS_valid] / 1000.0)
CAS_concentration_valid = (
    CAS_concentrations[CAS_valid])
CAS_fit = linregress(
    CAS_altitude_km,
    CAS_concentration_valid)
CAS_slope = CAS_fit.slope
CAS_intercept = CAS_fit.intercept
CAS_slope_error = CAS_fit.stderr
CAS_r = CAS_fit.rvalue
CAS_r2 = CAS_r**2
CAS_p = CAS_fit.pvalue
CDP_altitude_km = (
    CDP_altitudes[CDP_valid] / 1000.0)
CDP_concentration_valid = (
    CDP_concentrations[CDP_valid])
CDP_fit = linregress(
    CDP_altitude_km,
    CDP_concentration_valid)
CDP_slope = CDP_fit.slope
CDP_intercept = CDP_fit.intercept
CDP_slope_error = CDP_fit.stderr
CDP_r = CDP_fit.rvalue
CDP_r2 = CDP_r**2
CDP_p = CDP_fit.pvalue
# %%
# Expected concentration change from surface to 700 m
height_km = 0.700
CAS_change_700 = (
    CAS_slope * height_km)
CAS_change_700_error = (
    CAS_slope_error * height_km)
CAS_surface = CAS_intercept
CAS_at_700 = (
    CAS_intercept +
    CAS_slope * height_km)
CDP_change_700 = (
    CDP_slope * height_km)
CDP_change_700_error = (
    CDP_slope_error * height_km)
CDP_surface = CDP_intercept
CDP_at_700 = (
    CDP_intercept +
    CDP_slope * height_km)
# %%
#stats 
# %%
print("\nCAS RESULTS")
print(
    f"Slope b = "
    f"{CAS_slope:.3f} ± "
    f"{CAS_slope_error:.3f} "
    f"cm^-3 km^-1")
print(
    f"Intercept a = "
    f"{CAS_intercept:.3f} cm^-3")
print(
    f"R = {CAS_r:.3f}")
print(
    f"R^2 = {CAS_r2:.3f}")
print(
    f"p-value = {CAS_p:.4f}")
print(
    f"Predicted surface concentration = "
    f"{CAS_surface:.3f} cm^-3")
print(
    f"Predicted concentration at 700 m = "
    f"{CAS_at_700:.3f} cm^-3")

print(
    f"Change from surface to 700 m = "
    f"{CAS_change_700:.3f} ± "
    f"{CAS_change_700_error:.3f} cm^-3")
print("\nCDP RESULTS")
print(
    f"Slope b = "
    f"{CDP_slope:.3f} ± "
    f"{CDP_slope_error:.3f} "
    f"cm^-3 km^-1")
print(
    f"Intercept a = "
    f"{CDP_intercept:.3f} cm^-3")
print(
    f"R = {CDP_r:.3f}")
print(
    f"R^2 = {CDP_r2:.3f}")
print(
    f"p-value = {CDP_p:.4f}")
print(
    f"Predicted surface concentration = "
    f"{CDP_surface:.3f} cm^-3")
print(
    f"Predicted concentration at 700 m = "
    f"{CDP_at_700:.3f} cm^-3")
print(
    f"Change from surface to 700 m = "
    f"{CDP_change_700:.3f} ± "
    f"{CDP_change_700_error:.3f} cm^-3")
# %%
#update figure 
plt.figure(figsize=(10, 8))
plt.scatter(
    CAS_altitudes[CAS_valid],
    CAS_concentrations[CAS_valid],
    color="black",
    edgecolor="black",
    alpha=0.7,
    label="CAS")
plt.scatter(
    CDP_altitudes[CDP_valid],
    CDP_concentrations[CDP_valid],
    color="blue",
    edgecolor="black",
    alpha=0.7,
    label="CDP")
x_fit_m = np.linspace(
    0,
    max(
        np.nanmax(
            CAS_altitudes[CAS_valid]
        ),
        np.nanmax(
            CDP_altitudes[CDP_valid]
        )    ),    500)
x_fit_km = x_fit_m / 1000.0
CAS_y_fit = (
    CAS_intercept +
    CAS_slope * x_fit_km)
plt.plot(
    x_fit_m,
    CAS_y_fit,
    color="black",
    linewidth=2,
    linestyle="-")
CDP_y_fit = (
    CDP_intercept +
    CDP_slope * x_fit_km)
plt.plot(
    x_fit_m,
    CDP_y_fit,
    color="blue",
    linewidth=2,
    linestyle="-")
plt.xlabel(
    "Mean altitude (m)",
    fontsize=20,
    fontweight="bold")
plt.ylabel(
    "Total GCCN concentration (cm$^{-3}$)",
    fontsize=20,
    fontweight="bold")
plt.xticks(
    fontsize=16,
    fontweight="bold")
plt.yticks(
    fontsize=16,
    fontweight="bold")
plt.title(
    "GCCN concentration versus BCB altitude\n"
    "January–June 2022",
    fontsize=20,
    fontweight="bold")
plt.legend(
    fontsize=20,
    frameon=False)
plt.tight_layout()
plt.ylim(bottom=-1)
plt.show()
# %%
