#%%
import numpy as np
import pandas as pd
import csv
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pathlib
import statistics
import mputil
import shutil
import glob
import os
import re
import math
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
from scipy.spatial import distance
#%%
with open("CAS_ddry_massLE100.pkl", "rb") as f:
    filtered_master_BCB_ddry_massOK = pickle.load(f)
print("Loaded CAS filtered legs:", len(filtered_master_BCB_ddry_massOK))

with open ("CDP_ddry_massLE100.pkl", "rb") as f:
    master_ddry_CDP_mass100 = pickle.load(f)
print("Loaded CDP filtered legs:", len(master_ddry_CDP_mass100))
# %%
data = np.load("CAS_average_drysize_nomass100.npz")
common_bins = data["bins"]
average_dN_dD_dry = data["average"]
data = np.load("CDP_average_drysize_mass100.npz")
common_bins_CDP = data["bins"]
average_dN_dD_dry_CDP = data["average"]
plt.figure(figsize=(8, 6))
plt.plot(common_bins, average_dN_dD_dry,
         color="black", linewidth=2,
         label="CAS")
plt.plot(common_bins_CDP, average_dN_dD_dry_CDP,
         color="blue", linewidth=2,
         label="CDP")
plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=15, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=15, fontweight="bold")
plt.yscale("log")
plt.ylim(1e-4, 1e0)
plt.xlim(0, 45)
plt.xticks(fontweight="bold", fontsize=15)
plt.yticks(fontweight="bold", fontsize=15)
plt.title("Average Dry Size Distributions",
          fontsize=15, fontweight="bold")
plt.legend(fontsize=16, loc="center left", bbox_to_anchor=(1.02, 0.5))
plt.show()
#%%
#average distriutions with error bars
def mean_and_2sem_from_legs(legs, bins, x_key="ddry", y_key="dN/dDdry", require_positive=True):
    nlegs = len(legs)
    nbins = len(bins)
    Y = np.full((nlegs, nbins), np.nan, dtype=float)

    for i, e in enumerate(legs):
        x = np.asarray(e[x_key], dtype=float)
        y = np.asarray(e[y_key], dtype=float)

        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 2:
            continue

        f = interp1d(x[valid], y[valid], kind="linear", bounds_error=False, fill_value=np.nan)
        yi = f(bins)

        if require_positive:
            yi[~(np.isfinite(yi) & (yi > 0))] = np.nan

        Y[i, :] = yi

    N = np.sum(np.isfinite(Y), axis=0)
    mean_y = np.nanmean(Y, axis=0)

    std_y = np.nanstd(Y, axis=0, ddof=1)
    sem_y = std_y / np.sqrt(N)
    sem_y[N < 2] = np.nan

    sem2_y = 2 * sem_y
    return mean_y, sem2_y, N
with open("CAS_ddry_massLE100.pkl", "rb") as f:
    filtered_master_BCB_ddry_massOK = pickle.load(f)
print("Loaded CAS legs:", len(filtered_master_BCB_ddry_massOK))
with open("CDP_ddry_massLE100.pkl", "rb") as f:
    master_ddry_CDP_mass100 = pickle.load(f)
print("Loaded CDP legs:", len(master_ddry_CDP_mass100))
common_bins = np.linspace(2, 25, 35)
cas_mean, cas_2sem, cas_N = mean_and_2sem_from_legs(filtered_master_BCB_ddry_massOK, common_bins)
cdp_mean, cdp_2sem, cdp_N = mean_and_2sem_from_legs(master_ddry_CDP_mass100, common_bins)
plt.figure(figsize=(10, 6))
plt.errorbar(common_bins, cas_mean, yerr=cas_2sem, fmt="-", lw=2,
             color="black", capsize=3, label="CAS ±2 SEM (~95% CI)")
plt.errorbar(common_bins, cdp_mean, yerr=cdp_2sem, fmt="-", lw=2,
             color="blue", capsize=3, label="CDP ±2 SEM (~95% CI)")
plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=15, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=15, fontweight="bold")
plt.yscale("log")
plt.ylim(1e-4, 1e0)
plt.xlim(0, 45)
plt.xticks(fontweight="bold", fontsize=15)
plt.yticks(fontweight="bold", fontsize=15)
plt.title("Average Dry Size Distributions", fontsize=15, fontweight="bold")
plt.legend(fontsize=13, loc="center left", bbox_to_anchor=(1.02, 0.5))
plt.tight_layout(rect=[0, 0, 0.82, 1])
plt.show()
# %%
