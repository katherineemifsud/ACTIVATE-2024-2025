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
#%%
sys.modules.setdefault('numpy.core', np)
sys.modules.setdefault('numpy.core.multiarray', np.core.multiarray)
sys.modules.setdefault('numpy._core', np)
sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)

base = "/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions/Feb15BigSed"

# --- load n0_r.pkl ---
with open(os.path.join(base, "n0_r.pkl"), "rb") as f:
    n0_data = pickle.load(f)
print("n0_r.pkl contents:")
if isinstance(n0_data, (list, tuple)):
    print("Length:", len(n0_data))
    for i, item in enumerate(n0_data):
        print(f"  [{i}] type={type(item)}")
r_dry, n0_r = n0_data
print("  r_dry:", r_dry.shape, r_dry[:5])
print("  n0_r :", n0_r.shape, n0_r[:5])

# --- load R.pkl ---
with open(os.path.join(base, "R.pkl"), "rb") as f:
    R_data = pickle.load(f)
print("\nR.pkl contents:")
if isinstance(R_data, (list, tuple)):
    print("Length:", len(R_data))
    for i, item in enumerate(R_data):
        print(f"  [{i}] type={type(item)}")
time, rain_t = R_data
print("  time  :", time.shape, time[:5])
print("  rain_t:", rain_t.shape, rain_t[:5])
# %%
#big sedimentation precip time series 
plt.figure(figsize=(8, 5))

for i in range(rain_t.shape[0]):
    plt.plot(time, rain_t[i, :], lw=1, alpha=0.7)

plt.xlabel("Time (s)", fontweight="bold", fontsize=16)
plt.ylabel("Accumulated Rainfall (mm)", fontweight="bold", fontsize=16)
plt.title("BCB February 15\nBin Microphysics\nAll Legs", fontweight="bold", fontsize=18)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.yticks(fontweight="bold", fontsize=14)
plt.xticks(fontweight="bold", fontsize=14)
plt.show()
#%%

for i in range(rain_t.shape[0]):
    plt.figure(figsize=(6, 4))
    plt.plot(time, rain_t[i, :], lw=2)
    plt.title(f"BCB February 15\nBin Microphysics\nLeg {i+1}", fontweight="bold", fontsize=18)
    plt.xlabel("Time (s)", fontweight="bold", fontsize=16)
    plt.ylabel("Accumulated Rainfall (mm)", fontweight="bold", fontsize=16)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.yticks(fontweight="bold", fontsize=14)
    plt.xticks(fontweight="bold", fontsize=14)
    plt.show()
#%%
#plotting size distributions
for i in range(n0_r.shape[0]):
    plt.figure(figsize=(6, 4))
    plt.plot(r_dry, n0_r[i, :], lw=2)
    plt.title(f"February 15, 2022\nDry size distribution\nLeg {i+1}", fontweight="bold", fontsize=18)
    plt.xlabel("Dry radius (m)", fontweight="bold", fontsize=16)
    plt.ylabel("Number Concentration (m⁻⁴)", fontweight="bold", fontsize=16)
    plt.yscale("log")
    plt.yticks(fontweight="bold", fontsize=14)
    plt.xticks(fontweight="bold", fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
#%%
#cumulative size distributions 
dr = np.diff(r_dry)
dr = np.append(dr, dr[-1]) 
for i in range(n0_r.shape[0]):
    cumulative = np.cumsum((n0_r[i, ::-1] * dr[::-1]))[::-1]
    plt.figure(figsize=(6, 4))
    plt.plot(r_dry, cumulative, lw=2)
    plt.title(f"February 15, 2022\nCumulative dry size distribution\nLeg {i+1}",
              fontweight="bold", fontsize=18)
    plt.xlabel("Dry radius (m)", fontweight="bold", fontsize=16)
    plt.ylabel("Cumulative Number \nConcentration (m⁻³)", fontweight="bold", fontsize=16)
    plt.yscale("log")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

#%%
total_rain_mm = []

for i in range(rain_t.shape[0]):
    t = time  
    precip = rain_t[i, :]
    dt = np.median(np.diff(t))
    total_mm = np.sum(precip * dt) 
    total_rain_mm.append(total_mm)

total_rain_mm = np.array(total_rain_mm)
#%%
logx = np.log10(gccn_total_cm3)
logy = np.log10(total_rain_mm)
slope, intercept, r, p, _ = linregress(logx, logy)
fit_y = 10 ** (intercept + slope * logx)
plt.figure(figsize=(6, 4))
colors = plt.cm.viridis(np.linspace(0, 1, len(gccn_total_cm3)))

for i, (x, y, c) in enumerate(zip(gccn_total_cm3, total_rain_mm, colors), start=1):
    plt.scatter(x, y, s=80, color=c, edgecolor="k", linewidth=0.7, label=f"Leg {i}")
x_sorted = np.sort(gccn_total_cm3)
y_fit_sorted = 10 ** (intercept + slope * np.log10(x_sorted))
plt.plot(x_sorted, y_fit_sorted, "r--", lw=2,
         label=f"Fit: slope={slope:.2f}, R={r:.2f}")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("GCCN concentration \n(cm$^{-3}$, D > 1 µm)", fontweight="bold", fontsize=16)
plt.ylabel("Total accumulated \nrainfall (mm)", fontweight="bold", fontsize=16)
plt.title("Below Cloud Base\nFebruary 15, 2022\nGCCN vs. Rainfall per Leg", fontweight="bold", fontsize=14)
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.yticks(fontweight="bold", fontsize=14)
plt.xticks(fontweight="bold", fontsize=14)
plt.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
#%%
#converting to um radius and cm3 number concentration

r_dry_um = r_dry * 1e6              # m → µm
n0_r_um = n0_r / 1e6                # (#/m⁴) → (#/m³/µm)

# --- Mask for GCCN (r > 0.5 µm, i.e. D > 1 µm) ---
mask = r_dry_um > 0.5

# --- Integrate to get total GCCN concentration per leg ---
# n0_r_um [#/m³/µm] × dr [µm] → [#/m³]
gccn_total_m3 = np.trapz(n0_r_um[:, mask], r_dry_um[mask], axis=1)

# convert m⁻³ → cm⁻³ (1 m³ = 10⁶ cm³)
gccn_total_cm3 = gccn_total_m3 / 1e6

# --- Print GCCN for each leg ---
for i, val in enumerate(gccn_total_cm3, start=1):
    print(f"Leg {i:02d}: GCCN (>1 µm D) = {val:.3e} cm⁻³")

# =====================================================================
# --- Integrate rainfall time series per leg ---
total_rain_mm = []
for i in range(rain_t.shape[0]):
    precip = rain_t[i, :]           # [mm/s] or equivalent rate
    dt = np.median(np.diff(time))   # time step (s)
    total_mm = np.sum(precip * dt)  # total accumulated rainfall (mm)
    total_rain_mm.append(total_mm)
total_rain_mm = np.array(total_rain_mm)

# =====================================================================
# --- Log–log fit and scatter plot ---
logx = np.log10(gccn_total_cm3)
logy = np.log10(total_rain_mm)
slope, intercept, r, p, _ = linregress(logx, logy)

# --- Plot ---
plt.figure(figsize=(6, 4))
colors = plt.cm.viridis(np.linspace(0, 1, len(gccn_total_cm3)))

for i, (x, y, c) in enumerate(zip(gccn_total_cm3, total_rain_mm, colors), start=1):
    plt.scatter(x, y, s=80, color=c, edgecolor="k", linewidth=0.7, label=f"Leg {i}")

# Fit line
x_sorted = np.sort(gccn_total_cm3)
y_fit_sorted = 10 ** (intercept + slope * np.log10(x_sorted))
plt.plot(x_sorted, y_fit_sorted, "r--", lw=2,
         label=f"Fit: slope = {slope:.2f}, R = {r:.2f}")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("GCCN concentration (cm$^{-3}$, D > 1 µm)", fontweight="bold", fontsize=16)
plt.ylabel("Total accumulated rainfall (mm)", fontweight="bold", fontsize=16)
plt.title("Below Cloud Base – Feb 15 2022\nGCCN vs Rainfall per Leg", fontweight="bold", fontsize=14)
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.xticks(fontweight="bold", fontsize=14)
plt.yticks(fontweight="bold", fontsize=14)
plt.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
#%%

# %%
