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
import pickle
#%%
cas_pkl = "/home/disk/p/kathem24/activate/ACTIVATE-2024-2025/CDP/below cloud base/CAS_ddry_massLE100.pkl"
with open(cas_pkl, "rb") as f:
    filtered_master_BCB_ddry_massOK = pickle.load(f)
print("Loaded CAS filtered legs:", len(filtered_master_BCB_ddry_massOK))
#%%
BASE_DIR = "/home/disk/p/kathem24/activate/ACTIVATE-2024-2025/CDP/below cloud base"
cdp_pkl = os.path.join(BASE_DIR, "CDP_ddry_massLE100.pkl")
print("Trying to load:", cdp_pkl)
print("Exists?", os.path.exists(cdp_pkl))
with open(cdp_pkl, "rb") as f:
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
plt.xlim(0, 28)
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

        f = interp1d(x[valid], y[valid],
                     kind="linear",
                     bounds_error=False,
                     fill_value=np.nan)

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
cas_npz = np.load("CAS_average_drysize_nomass100.npz")
common_bins = cas_npz["bins"]
cas_mean = cas_npz["average"]
cdp_npz = np.load("CDP_average_drysize_mass100.npz")
cdp_bins = cdp_npz["bins"]
cdp_mean0 = cdp_npz["average"]
cdp_mean = np.interp(common_bins, cdp_bins, cdp_mean0, left=np.nan, right=np.nan)
cas_mean2, cas_2sem, cas_N = mean_and_2sem_from_legs(filtered_master_BCB_ddry_massOK, common_bins)
cdp_mean2, cdp_2sem, cdp_N = mean_and_2sem_from_legs(master_ddry_CDP_mass100, common_bins)
plt.figure(figsize=(10, 6))
plt.errorbar(common_bins, cas_mean, yerr=cas_2sem, fmt="-", lw=2,
             color="black", capsize=3, label="CAS ±2 SEM (~95% CI)")
plt.errorbar(common_bins, cdp_mean, yerr=cdp_2sem, fmt="-", lw=2,
             color="blue", capsize=3, label="CDP ±2 SEM (~95% CI)")
plt.yscale("log")
plt.ylim(1e-4, 1e0)
plt.xlim(0, 28)
plt.xticks(fontweight="bold", fontsize=16)
plt.yticks(fontweight="bold", fontsize=16)
plt.xlabel("Dry Bin Center Diameter (μm)", fontsize=17, fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=17, fontweight="bold")
plt.title("Average Dry Size Distributions", fontsize=17, fontweight="bold",  pad=20)
plt.legend(fontsize=13, loc="center left", bbox_to_anchor=(1.02, 0.5))
plt.tight_layout(rect=[0, 0, 0.82, 1])
plt.tight_layout()
plt.subplots_adjust(right=0.75)
out = "2022_CASCDP_averagedrydist.pdf"
plt.savefig(out, dpi=300, bbox_inches="tight")
print("Saved:", out)
import os
out = "2022_CASCDP_averagedrydist.pdf"
plt.savefig(out, bbox_inches="tight")  # dpi not needed for pdf
print("Python CWD:", os.getcwd())
print("Absolute:", os.path.abspath(out))
print("Exists?", os.path.exists(out), "bytes:", os.path.getsize(out) if os.path.exists(out) else None)
plt.show()
# %%
#windspeed relationhips 
data = np.load("CAS_CDP_wind_mass_concentration_windspeed.npz")

windspeed_values = data["windspeed_values"]
total_concentrations = data["total_concentrations"]
standard_errors = data["standard_errors"]
counting_errors_CAS = data["counting_errors_CAS"]

windspeed_values_CDP = data["windspeed_values_CDP"]
total_concentrations_CDP = data["total_concentrations_CDP"]
standard_errors_CDP = data["standard_errors_CDP"]
counting_errors_CDP = data["counting_errors_CDP"]

cas_x = data["cas_x"]
cas_y = data["cas_y"]
cas_se = data["cas_se"]
cas_ce = data["cas_ce"]

cdp_x = data["cdp_x"]
cdp_y = data["cdp_y"]
cdp_se = data["cdp_se"]
cdp_ce = data["cdp_ce"]

x_fit_CAS = data["x_fit_CAS"]
y_fit_CAS = data["y_fit_CAS"]
x_fit_CDP = data["x_fit_CDP"]
y_fit_CDP = data["y_fit_CDP"]

xfit_cas = data["xfit_cas"]
yfit_cas = data["yfit_cas"]
xfit_cdp = data["xfit_cdp"]
yfit_cdp = data["yfit_cdp"]