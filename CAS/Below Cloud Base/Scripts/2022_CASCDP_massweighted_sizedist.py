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
# %%
CAS_FILE = (
    "/home/disk/p/kathem24/activate/"
    "ACTIVATE-2024-2025/CAS/Below Cloud Base/Scripts/"
    "CAS_mass_weighted_dry_distribution_massLE1002022.pkl")
with open(CAS_FILE, "rb") as f:
    CAS_mass_distribution_data = pickle.load(f)
print("Loaded CAS mass-weighted distribution.")
print("Saved keys:")
print(CAS_mass_distribution_data.keys())
#%%
CDP_FILE = (
    "/home/disk/p/kathem24/activate/"
    "ACTIVATE-2024-2025/CDP/below cloud base/"
    "CDP_mass_weighted_dry_distribution_massLE1002022.pkl")
with open(CDP_FILE, "rb") as f:
    CDP_mass_distribution_data = pickle.load(f)
print("Loaded CDP mass-weighted distribution.")
print("Saved keys:")
print(CDP_mass_distribution_data.keys())
# %%
# %%
plt.figure(figsize=(10, 8))
plt.plot(CAS_mass_distribution_data["Diameter (µm)"],
    CAS_mass_distribution_data[
        "Mean dM/dlog10D (µg/m³)"],
    color="black",
    linewidth=3,
    label="CAS")
plt.fill_between(CAS_mass_distribution_data["Diameter (µm)"],
    CAS_mass_distribution_data[
        "Mean dM/dlog10D (µg/m³)"
    ] - CAS_mass_distribution_data[
        "2SEM dM/dlog10D (µg/m³)"],
    CAS_mass_distribution_data[
        "Mean dM/dlog10D (µg/m³)"
    ] + CAS_mass_distribution_data[
        "2SEM dM/dlog10D (µg/m³)"],
    color="black",
    alpha=0.2)
plt.plot(CDP_mass_distribution_data["Diameter (µm)"],
    CDP_mass_distribution_data[
        "Mean dM/dlog10D (µg/m³)"],
    color="blue",
    linewidth=3,
    label="CDP")
plt.fill_between(CDP_mass_distribution_data["Diameter (µm)"],
    CDP_mass_distribution_data[
        "Mean dM/dlog10D (µg/m³)"
    ] - CDP_mass_distribution_data[
        "2SEM dM/dlog10D (µg/m³)"
    ],
    CDP_mass_distribution_data[
        "Mean dM/dlog10D (µg/m³)"
    ] + CDP_mass_distribution_data[
        "2SEM dM/dlog10D (µg/m³)"],
    color="blue",
    alpha=0.2)
plt.xlabel("Dry diameter (µm)",
    fontsize=20,
    fontweight="bold")
plt.ylabel(r"$dM/d\log_{10}D$ ($\mu$g m$^{-3}$)",
    fontsize=20,
    fontweight="bold")
plt.title("Mass-Weighted Dry Size Distributions\n January-June 2022",
    fontsize=20,
    fontweight="bold")
plt.xscale("log")
plt.xticks(fontsize=18,
    fontweight="bold")
plt.yticks(fontsize=18,
    fontweight="bold")
plt.legend(fontsize=18,
    frameon=False)
plt.xlim(2, 50)
plt.tight_layout()
plt.show()
# %%
