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
from scipy.stats import gaussian_kde
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
