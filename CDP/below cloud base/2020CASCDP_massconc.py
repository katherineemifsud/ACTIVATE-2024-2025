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
import shutil
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy.ma as ma
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
import pickle
from scipy.integrate import quad
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D
from collections import Counter
from scipy.spatial import distance
# %%
import pickle

base_path = "/home/disk/p/kathem24/activate/ACTIVATE-2024-2025/CDP/below cloud base"

cas_slope_file = f"{base_path}/CAS_slope_massLE1002020.pkl"
cas_ddry_file  = f"{base_path}/CAS_ddry_massLE1002020.pkl"

cdp_slope_file = f"{base_path}/CDP_slope_massLE1002020.pkl"
cdp_ddry_file  = f"{base_path}/CDP_ddry_massLE1002020.pkl"

with open(cas_slope_file, "rb") as f:
    cas_slope = pickle.load(f)

with open(cas_ddry_file, "rb") as f:
    cas_ddry = pickle.load(f)
with open(cdp_slope_file, "rb") as f:
    cdp_slope = pickle.load(f)

with open(cdp_ddry_file, "rb") as f:
    cdp_ddry = pickle.load(f)
print("CAS slope:", len(cas_slope))
print("CAS ddry :", len(cas_ddry))
print("CDP slope:", len(cdp_slope))
print("CDP ddry :", len(cdp_ddry))
# %%
print("CAS slope keys:")
print(cas_slope[0].keys())
print("\nCAS ddry keys:")
print(cas_ddry[0].keys())
print("\nCDP slope keys:")
print(cdp_slope[0].keys())
print("\nCDP ddry keys:")
print(cdp_ddry[0].keys())
# %%
def leg_key(entry):
    return (entry["Date"], int(entry["BCB_start"]), int(entry["BCB_stop"]))
def calc_total_concentration(entry):
    ddry = np.array(entry["ddry"], dtype=float)
    dNdDdry = np.array(entry["dN/dDdry"], dtype=float)
    widths = np.array(entry["ddry_bin_widths"], dtype=float)
    good = np.isfinite(ddry) & np.isfinite(dNdDdry) & np.isfinite(widths)
    if np.sum(good) == 0:
        return np.nan

    return np.sum(dNdDdry[good] * widths[good])
def match_mass_slope_conc(slope_data, ddry_data):
    conc_lookup = {
        leg_key(e): calc_total_concentration(e)
        for e in ddry_data
    }

    conc = []
    mass = []
    slope = []

    for e in slope_data:
        key = leg_key(e)

        if key not in conc_lookup:
            continue

        c = conc_lookup[key]
        m = e["Dry Mass (µg/m³)"]
        d = e["Dry Slope (D)"]

        if np.isfinite(c) and np.isfinite(m) and np.isfinite(d) and c > 0 and m > 0:
            conc.append(c)
            mass.append(m)
            slope.append(d)

    return np.array(conc), np.array(mass), np.array(slope)

cas_conc, cas_mass, cas_D = match_mass_slope_conc(cas_slope, cas_ddry)
cdp_conc, cdp_mass, cdp_D = match_mass_slope_conc(cdp_slope, cdp_ddry)
print("Matched CAS points:", len(cas_conc))
print("Matched CDP points:", len(cdp_conc))
print(f"CAS median D: {np.median(cas_D):.2f}")
print(f"CDP median D: {np.median(cdp_D):.2f}")
# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
vmin = min(np.nanmin(cas_D), np.nanmin(cdp_D))
vmax = max(np.nanmax(cas_D), np.nanmax(cdp_D))
sc0 = axes[0].scatter(
    cas_conc, cas_mass,
    c=cas_D,
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
    s=55,
    edgecolor="black",
    linewidth=0.4,
    alpha=0.85
)

axes[0].scatter(
    np.median(cas_conc),
    np.median(cas_mass),
    marker="s",
    s=180,
    color="black",
    zorder=10
)
axes[0].set_title("CAS 2020", fontsize=18, fontweight="bold")
axes[0].set_xlabel(r"GCCN Concentration (cm$^{-3}$)", fontsize=15, fontweight="bold")
axes[0].set_ylabel(r"GCCN Mass ($\mu$g m$^{-3}$)", fontsize=15, fontweight="bold")
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].grid(True, alpha=0.3)
sc1 = axes[1].scatter(
    cdp_conc, cdp_mass,
    c=cdp_D,
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
    s=55,
    edgecolor="black",
    linewidth=0.4,
    alpha=0.85
)

axes[1].scatter(
    np.median(cdp_conc),
    np.median(cdp_mass),
    marker="^",
    s=180,
    color="black",
    zorder=10
)
axes[1].set_title("CDP 2020", fontsize=18, fontweight="bold")
axes[1].set_xlabel(r"GCCN Concentration (cm$^{-3}$)", fontsize=15, fontweight="bold")
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].grid(True, alpha=0.3)
cax = fig.add_axes([0.83, 0.16, 0.018, 0.74])
cbar = fig.colorbar(sc1, cax=cax)
cbar.set_label(r"GCCN Slope Parameter D ($\mu$m)", fontsize=15, fontweight="bold", labelpad=18)
cbar.ax.tick_params(labelsize=12)
legend_elements = [
    Line2D([0], [0], marker="s", color="black", linestyle="None",
           markersize=10, label=f"CAS median D = {np.median(cas_D):.2f} µm"),
    Line2D([0], [0], marker="^", color="black", linestyle="None",
           markersize=10, label=f"CDP median D = {np.median(cdp_D):.2f} µm")
]
fig.legend(
    handles=legend_elements,
    loc="center left",
    bbox_to_anchor=(0.9, 0.5),
    fontsize=13,
    frameon=False
)
for ax in axes:
    ax.tick_params(axis="both", labelsize=13)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
plt.subplots_adjust(right=0.82, wspace=0.07)
plt.show()
# %%
