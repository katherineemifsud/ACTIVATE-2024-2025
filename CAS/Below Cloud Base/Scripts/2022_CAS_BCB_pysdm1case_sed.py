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

base = "/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions/Feb15Bins"
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
plt.title("BCB February 15\nBin Microphysics Heavy Sedimentation\nAll Legs", fontweight="bold", fontsize=18)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.yticks(fontweight="bold", fontsize=14)
plt.xticks(fontweight="bold", fontsize=14)
plt.show()
#%%

for i in range(rain_t.shape[0]):
    plt.figure(figsize=(6, 4))
    plt.plot(time, rain_t[i, :], lw=2)
    plt.title(f"BCB February 15\nBin Microphysics Heavy Sedimentation\nLeg {i+1}", fontweight="bold", fontsize=18)
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
# total accumulated rainfall per leg (in mm)
total_rain_mm = rain_t[:, -1]

for i, val in enumerate(total_rain_mm, start=1):
    print(f"Leg {i:02d}: Total accumulated rainfall = {val:.4f} mm")
# %%

dr = np.diff(r_dry)
dr = np.append(dr, dr[-1])
dlnr = np.diff(np.log(r_dry))
dlnr = np.append(dlnr, dlnr[-1])

i = 0  # pick one leg
raw_sum     = np.sum(n0_r[i])                     # if already per-bin (#/m3)
linear_sum  = np.sum(n0_r[i] * dr)                # if #/m4
log_sum     = np.sum(n0_r[i] * r_dry * dlnr)      # if #/m4 (log-spaced bins)

print(f"Raw sum     = {raw_sum:.3e} /m³ (assuming per-bin)")
print(f"Linear int  = {linear_sum:.3e} /m³ (assuming #/m⁴)")
print(f"Log int     = {log_sum:.3e} /m³ (assuming #/m⁴ log bins)")

# %%
# Check log-spacing uniformity
log_r = np.log10(r_dry)
diffs = np.diff(log_r)
print("Mean Δlog10(r):", np.mean(diffs))
print("Std(Δlog10(r)) :", np.std(diffs))
print("First 5 Δlog10(r):", diffs[:5])
print("Last 5 Δlog10(r):", diffs[-5:])
plt.figure(figsize=(6,3))
plt.plot(diffs, '.-')
plt.xlabel("Bin index")
plt.ylabel("Δ log10(r)")
plt.title("Spacing between bins in log10(r)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Calculate total and GCCN number concentrations per leg
dlnr = np.log(r_dry[1] / r_dry[0])
total_m3 = np.sum(n0_r * r_dry * dlnr, axis=1)

mask = r_dry > 0.5e-6  # D > 1 µm
gccn_m3 = np.sum(n0_r[:, mask] * r_dry[mask] * dlnr, axis=1)

for i, (tot, gccn) in enumerate(zip(total_m3, gccn_m3), start=1):
    frac = gccn / tot if tot > 0 else np.nan
    print(f"Leg {i:02d}: Total={tot:.3e} m^-3, GCCN={gccn:.3e} m^-3, GCCN/Total={frac:.2e}")

# %%

dlnr = np.log(r_dry[1] / r_dry[0])

total_m3 = np.sum(n0_r * r_dry * dlnr, axis=1)
mask = r_dry > 0.5e-6  # D > 1 µm
gccn_m3 = np.sum(n0_r[:, mask] * r_dry[mask] * dlnr, axis=1)
total_rain_mm = rain_t[:, -1]
for i, (gccn, rain) in enumerate(zip(gccn_m3, total_rain_mm), start=1):
    print(f"Leg {i:02d}: GCCN={gccn:.3e} m^-3, Rain={rain:.4f} mm")
plt.figure(figsize=(6, 4.5))
colors = plt.cm.viridis(np.linspace(0, 1, len(gccn_m3)))

for i, (gccn, rain, c) in enumerate(zip(gccn_m3, total_rain_mm, colors), start=1):
    plt.scatter(gccn, rain, s=80, edgecolor='k', color=c, label=f"Leg {i}")
logx = np.log10(gccn_m3)
logy = np.log10(total_rain_mm)
slope, intercept, r_value, p_value, std_err = linregress(logx, logy)
fit_y = 10 ** (intercept + slope * logx)
x_sorted = np.sort(gccn_m3)
y_fit_sorted = 10 ** (intercept + slope * np.log10(x_sorted))
plt.plot(x_sorted, y_fit_sorted, "r--", lw=2,
         label=f"Fit: slope={slope:.2f}, R={r_value:.2f}")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("GCCN concentration (m$^{-3}$, D > 1 µm)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated rainfall (mm)", fontsize=16, fontweight="bold")
plt.title("Below Cloud Base\n Feb 15, 2022\nGCCN vs Rainfall", fontsize=16, fontweight="bold")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.grid(alpha=0.3)
plt.yticks(fontweight="bold", fontsize=14)
plt.xticks(fontweight="bold", fontsize=14)  
plt.tight_layout()
plt.show()

# %%
#Less sedimentation case for comparison
sys.modules.setdefault('numpy.core', np)
sys.modules.setdefault('numpy.core.multiarray', np.core.multiarray)
sys.modules.setdefault('numpy._core', np)
sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)

base = "/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions/Feb15Binsgap"
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
#less sedimentation precip time series 
plt.figure(figsize=(8, 5))

for i in range(rain_t.shape[0]):
    plt.plot(time, rain_t[i, :], lw=1, alpha=0.7)

plt.xlabel("Time (s)", fontweight="bold", fontsize=16)
plt.ylabel("Accumulated Rainfall (mm)", fontweight="bold", fontsize=16)
plt.title("BCB February 15\nBin Microphysics Less Sedimentation\nAll Legs", fontweight="bold", fontsize=18)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.yticks(fontweight="bold", fontsize=14)
plt.xticks(fontweight="bold", fontsize=14)
plt.show()
# %%
for i in range(rain_t.shape[0]):
    plt.figure(figsize=(6, 4))
    plt.plot(time, rain_t[i, :], lw=2)
    plt.title(f"BCB February 15\nBin Microphysics Less Sedimentation\nLeg {i+1}", fontweight="bold", fontsize=18)
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
# %%
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
# %%
# total accumulated rainfall per leg (in mm)
total_rain_mm = rain_t[:, -1]

for i, val in enumerate(total_rain_mm, start=1):
    print(f"Leg {i:02d}: Total accumulated rainfall = {val:.4f} mm")
# %%

dr = np.diff(r_dry)
dr = np.append(dr, dr[-1])
dlnr = np.diff(np.log(r_dry))
dlnr = np.append(dlnr, dlnr[-1])

i = 0  # pick one leg
raw_sum     = np.sum(n0_r[i])                     # if already per-bin (#/m3)
linear_sum  = np.sum(n0_r[i] * dr)                # if #/m4
log_sum     = np.sum(n0_r[i] * r_dry * dlnr)      # if #/m4 (log-spaced bins)

print(f"Raw sum     = {raw_sum:.3e} /m³ (assuming per-bin)")
print(f"Linear int  = {linear_sum:.3e} /m³ (assuming #/m⁴)")
print(f"Log int     = {log_sum:.3e} /m³ (assuming #/m⁴ log bins)")

# %%
# Check log-spacing uniformity
log_r = np.log10(r_dry)
diffs = np.diff(log_r)
print("Mean Δlog10(r):", np.mean(diffs))
print("Std(Δlog10(r)) :", np.std(diffs))
print("First 5 Δlog10(r):", diffs[:5])
print("Last 5 Δlog10(r):", diffs[-5:])
plt.figure(figsize=(6,3))
plt.plot(diffs, '.-')
plt.xlabel("Bin index")
plt.ylabel("Δ log10(r)")
plt.title("Spacing between bins in log10(r)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Calculate total and GCCN number concentrations per leg
dlnr = np.log(r_dry[1] / r_dry[0])
total_m3 = np.sum(n0_r * r_dry * dlnr, axis=1)

mask = r_dry > 0.5e-6  # D > 1 µm
gccn_m3 = np.sum(n0_r[:, mask] * r_dry[mask] * dlnr, axis=1)

for i, (tot, gccn) in enumerate(zip(total_m3, gccn_m3), start=1):
    frac = gccn / tot if tot > 0 else np.nan
    print(f"Leg {i:02d}: Total={tot:.3e} m^-3, GCCN={gccn:.3e} m^-3, GCCN/Total={frac:.2e}")

# %%

dlnr = np.log(r_dry[1] / r_dry[0])

total_m3 = np.sum(n0_r * r_dry * dlnr, axis=1)
mask = r_dry > 0.5e-6  # D > 1 µm
gccn_m3 = np.sum(n0_r[:, mask] * r_dry[mask] * dlnr, axis=1)
total_rain_mm = rain_t[:, -1]
for i, (gccn, rain) in enumerate(zip(gccn_m3, total_rain_mm), start=1):
    print(f"Leg {i:02d}: GCCN={gccn:.3e} m^-3, Rain={rain:.4f} mm")
plt.figure(figsize=(6, 4.5))
colors = plt.cm.viridis(np.linspace(0, 1, len(gccn_m3)))

for i, (gccn, rain, c) in enumerate(zip(gccn_m3, total_rain_mm, colors), start=1):
    plt.scatter(gccn, rain, s=80, edgecolor='k', color=c, label=f"Leg {i}")
logx = np.log10(gccn_m3)
logy = np.log10(total_rain_mm)
slope, intercept, r_value, p_value, std_err = linregress(logx, logy)
fit_y = 10 ** (intercept + slope * logx)
x_sorted = np.sort(gccn_m3)
y_fit_sorted = 10 ** (intercept + slope * np.log10(x_sorted))
plt.plot(x_sorted, y_fit_sorted, "r--", lw=2,
         label=f"Fit: slope={slope:.2f}, R={r_value:.2f}")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("GCCN concentration (m$^{-3}$, D > 1 µm)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated rainfall (mm)", fontsize=16, fontweight="bold")
plt.title("Below Cloud Base\n Feb 15, 2022\nGCCN vs Rainfall", fontsize=16, fontweight="bold")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.grid(alpha=0.3)
plt.yticks(fontweight="bold", fontsize=14)
plt.xticks(fontweight="bold", fontsize=14)  
plt.tight_layout()
plt.show()
# %%
