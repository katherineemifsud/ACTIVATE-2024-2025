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
#no turbulence base 385 g m^-2 LWP
sys.modules.setdefault('numpy.core', np)
sys.modules.setdefault('numpy.core.multiarray', np.core.multiarray)
sys.modules.setdefault('numpy._core', np)
sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)

base = "/home/disk/eos4/kathem24/activate/data/CAS/Full base"

def inspect_pickle(fname):
    with open(os.path.join(base, fname), "rb") as f:
        data = pickle.load(f)

    print(f"\n{fname} contents:")
    if isinstance(data, (list, tuple)):
        print("Length:", len(data))
        for i, item in enumerate(data):
            print(f"  [{i}] type={type(item)}")
    else:
        print("Type:", type(data))
    return data
n0_data = inspect_pickle("n0_r.pkl")
R_data  = inspect_pickle("R (1).pkl")
r_dry = n0_data[0]
n0_r  = n0_data[1]
extra = n0_data[2] if len(n0_data) > 2 else None
print("\nExtracted:")
print("  r_dry:", np.shape(r_dry))
print("  n0_r :", np.shape(n0_r))
print("  extra data:", type(extra), "\n")
time = R_data[0]
rain_t = R_data[1]
extra_R = R_data[2] if len(R_data) > 2 else None
print("  time  :", np.shape(time))
print("  rain_t:", np.shape(rain_t))
print("  extra R data:", type(extra_R))
#%%
with open(os.path.join(base, "R (1).pkl"), "rb") as f:
    R_data = pickle.load(f)

R_data = np.asarray(R_data)
print(type(R_data))
print(R_data.shape)
print(R_data)
#%%
mean_precip = np.mean(R_data, axis=1)
std_precip = np.std(R_data, axis=1)
case_idx = np.arange(R_data.shape[0])

plt.figure(figsize=(8, 5))
plt.plot(case_idx, mean_precip, lw=2)
plt.fill_between(case_idx, mean_precip - std_precip, mean_precip + std_precip, alpha=0.25)

plt.xlabel("Case index", fontweight="bold", fontsize=16)
plt.ylabel("Accumulated Rain (mm)", fontweight="bold", fontsize=16)
plt.title("BCB January - June 2022\nBase 385 g m$^{-2}$ LWP\nAccumulated Rainfall",
          fontweight="bold", fontsize=18)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.xticks(fontsize=14, fontweight="bold")
plt.yticks(fontsize=14, fontweight="bold")
plt.show()
# %%
# Calculate total and GCCN number concentrations per leg
total_m3 = np.sum(n0_r, axis=1)
mask = r_dry > 1e-6
gccn_m3_base = np.sum(n0_r[:, mask], axis=1)
for i, (tot, gccn) in enumerate(zip(total_m3, gccn_m3_base), start=1):
    frac = gccn / tot
    print(f"Leg {i:02d}: Total={tot:.3e} m^-3, GCCN={gccn:.3e} m^-3, GCCN/Total={frac:.2e}")

# %%
# GCCN versus accumulated rain
mask = r_dry > 1e-6   # radius > 1 µm -> diameter > 2 µm
gccn_m3_base = np.sum(n0_r[:, mask], axis=1)   
R_data = np.asarray(R_data)
accum_rain_base = np.mean(R_data, axis=1)      
accum_rain_base_full = np.array(accum_rain_base, dtype=float)
gccn_m3_base_full = np.array(gccn_m3_base, dtype=float)
for i, (gccn, rain) in enumerate(zip(gccn_m3_base, accum_rain_base), start=1):
    print(f"Leg {i:02d}: GCCN={gccn:.3e} m^-3, Rain={rain:.3f} mm")
plt.figure(figsize=(6, 4.5))
colors = plt.cm.viridis(np.linspace(0, 1, len(gccn_m3_base)))
for i, (gccn, rain, c) in enumerate(zip(gccn_m3_base, accum_rain_base, colors), start=1):
    plt.scatter(gccn, rain, s=80, edgecolor='k', color=c)
valid = (gccn_m3_base > 0) & (accum_rain_base > 0)
logx = np.log10(gccn_m3_base[valid])
logy = np.log10(accum_rain_base[valid])
slope, intercept, r_value, p_value, std_err = linregress(logx, logy)
x_sorted = np.sort(gccn_m3_base[valid])
y_fit_sorted = 10 ** (intercept + slope * np.log10(x_sorted))
plt.plot(x_sorted, y_fit_sorted, "r--", lw=2,
         label=f"Fit: slope={slope:.2f}, R={r_value:.2f}")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("GCCN concentration (m$^{-3}$)", fontsize=16, fontweight="bold")
plt.ylabel("Final accumulated rain (mm)", fontsize=16, fontweight="bold")
plt.title("BCB January - June 2022\nBase 385 g m$^{-2}$ LWP\nFinal Accumulated Rainfall",
          fontweight="bold", fontsize=18)
plt.grid(alpha=0.3)
plt.yticks(fontweight="bold", fontsize=16)
plt.xticks(fontweight="bold", fontsize=16)
plt.tight_layout()
plt.show()
#%%
#mass analysis
mass_path = "/home/disk/eos4/kathem24/activate/data/CAS/filtered_dry_mass_inf.csv"
df_mass = pd.read_csv(mass_path)
print("CSV rows:", len(df_mass))
print("Rain full len:", len(accum_rain_base_full))
print("GCCN full len:", len(gccn_m3_base_full))
all_mass = df_mass.to_dict(orient="records")
all_mass_sorted = sorted(all_mass, key=lambda x: (x["Date"], x["BCB_start"]))
all_mass_values = [entry["Dry Mass (µg/m³)"] for entry in all_mass_sorted]
mass_thr = 100  # µg/m³
mass_full = np.array(all_mass_values, dtype=float)
rain_full = np.array(accum_rain_base_full, dtype=float)
gccn_full = np.array(gccn_m3_base_full, dtype=float)
assert len(mass_full) == len(rain_full) == len(gccn_full), \
    f"mass={len(mass_full)} rain_full={len(rain_full)} gccn_full={len(gccn_full)}"
keep = (
    np.isfinite(mass_full) &
    np.isfinite(rain_full) &
    (mass_full > 0) &
    (rain_full > 0) &
    (mass_full <= mass_thr)
)
neg_rain_idx = np.where(rain_full < 0)[0]
zero_rain_idx = np.where(rain_full == 0)[0]

print("Negative rain cases:", len(neg_rain_idx))
print("Exactly zero rain cases:", len(zero_rain_idx))

if len(neg_rain_idx) > 0:
    print("\nIndices with NEGATIVE rain:")
    print(neg_rain_idx)

if len(zero_rain_idx) > 0:
    print("\nIndices with ZERO rain:")
    print(zero_rain_idx)

print("\nSmallest 10 rain values:")
print(np.sort(rain_full)[:10])
all_mass_values  = mass_full[keep].tolist()
accum_rain_base  = rain_full[keep]
gccn_m3_base= gccn_full[keep]
print("Kept legs:", keep.sum())
print("Dropped legs:", (~keep).sum())
print("Now lengths -> mass:", len(all_mass_values), "rain:", len(accum_rain_base), "gccn:", len(gccn_m3_base))

#%%
mass = np.array(all_mass_values)
rain = accum_rain_base 
for i, (m, r) in enumerate(zip(mass, rain), start=1):
    print(f"Leg {i:02d}: Mass={m:.2f} µg/m³, Rain={r:.3f} mm")
plt.figure(figsize=(6, 4.5))
colors = plt.cm.plasma(np.linspace(0, 1, len(mass)))

for i, (m, r, c) in enumerate(zip(mass, rain, colors), start=1):
    plt.scatter(m, r, s=80, edgecolor='k', color=c, label=f"Leg {i}")
logx = np.log10(mass)
logy = np.log10(rain)
slope, intercept, r_value, p_value, std_err = linregress(logx, logy)
R = r_value
R2 = R**2
print(f"Correlation R = {R:.4f}")
print(f"Coefficient of Determination R² = {R2:.4f}")
x_sorted = np.sort(mass)
y_fit_sorted = 10 ** (intercept + slope * np.log10(x_sorted))
plt.plot(x_sorted, y_fit_sorted, "r--", lw=2,
         label=f"Fit: slope={slope:.2f}, R={r_value:.2f}")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Dry GCCN Mass (µg/m³)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")
plt.title("BCB January - June 2022\n Base 385 g m$^{-2}$ LWP\nAccumulated Rainfall", 
          fontweight="bold", fontsize=18)
# plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.grid(alpha=0.3)
plt.yticks(fontweight="bold", fontsize=14)
plt.xticks(fontweight="bold", fontsize=14)
plt.tight_layout()
plt.show()
#%%
mass = np.array(all_mass_values)
rain = accum_rain_base   
plt.figure(figsize=(6, 4.5))
colors = plt.cm.plasma(np.linspace(0, 1, len(mass)))
for i, (m, r, c) in enumerate(zip(mass, rain, colors), start=1):
    plt.scatter(m, r, s=80, edgecolor='k', color=c)
logx = np.log10(mass)
logy = np.log10(rain)
slope, intercept, r_value, p_value, std_err = linregress(logx, logy)
R = r_value
R2 = R**2
print(f"FULL DATA: R = {R:.4f}, R² = {R2:.4f}")

x_sorted = np.sort(mass)
y_fit_sorted = 10 ** (intercept + slope * np.log10(x_sorted))

plt.plot(x_sorted, y_fit_sorted, "r--", lw=2,
         label=f"Full Fit: R={R:.2f}")
lower = 0.01
upper = 100.0
mask_range = (mass >= lower) & (mass <= upper)
mass_filt = mass[mask_range]
rain_filt = rain[mask_range]
logx_filt = np.log10(mass_filt)
logy_filt = np.log10(rain_filt)
slope_filt, intercept_filt, r_value_filt, p_value_filt, std_err_filt = linregress(logx_filt, logy_filt)
R_filt = r_value_filt
R2_filt = R_filt**2
print(f"FILTERED (0.1–100): R = {R_filt:.4f}, R² = {R2_filt:.4f}")
print(f"Points used: {len(mass_filt)}")
x_sorted_filt = np.sort(mass_filt)
y_fit_sorted_filt = 10 ** (intercept_filt + slope_filt * np.log10(x_sorted_filt))
plt.plot(x_sorted_filt, y_fit_sorted_filt, "g--", lw=3,
         label=f"Filtered Fit: R={R_filt:.2f}")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Dry GCCN Mass (µg/m³)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")
plt.title("BCB January - June 2022\nBase 385 g m$^{-2}$ LWP\nAccumulated Rainfall",
          fontsize=18, fontweight="bold")
plt.grid(alpha=0.3)
plt.yticks(fontweight="bold", fontsize=14)
plt.xticks(fontweight="bold", fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()


#%%
#plotting slope D vs rain directly
# all_entries = dry_mass_data_inf
# all_sorted = sorted(all_entries, key=lambda x: (x['Date'], x['BCB_start']))
# all_mass_values = np.array([entry['Dry Mass (µg/m³)'] for entry in all_sorted])
# all_slopes      = np.array([entry['Dry Slope (D)']       for entry in all_sorted])
# all_intercepts  = np.array([entry['Dry Intercept (N0)']  for entry in all_sorted])
# slope_D = all_slopes
# rain_mm = accum_rain_base   
# plt.figure(figsize=(6, 4.5))
# colors = plt.cm.cool(np.linspace(0, 1, len(slope_D)))
# for i, (D, r, c) in enumerate(zip(slope_D, rain_mm, colors), start=1):
#     plt.scatter(D, r, s=80, edgecolor='k', color=c)
# log_rain = np.log10(rain_mm)
# slope_lr, intercept_lr, r_val2, p_val2, _ = linregress(slope_D, log_rain)
# D_sorted = np.sort(slope_D)
# rain_fit2 = 10 ** (intercept_lr + slope_lr * D_sorted)
# plt.plot(D_sorted, rain_fit2, "r--", lw=2,
#          label=f"Fit: slope={slope_lr:.2f}, R={r_val2:.2f}")
# plt.yscale("log")
# plt.xlabel("Dry Slope D (µm)", fontsize=16, fontweight="bold")
# plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")
# plt.title("BCB January - June 2022\n Base 385 g m$^{-2}$ LWP", 
#           fontweight="bold", fontsize=18)
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.show()
#%%
#plotting slope D vs rain directly less than 100 ug/m3
# use the already-filtered 440-leg dataset
all_entries = [entry for entry, k in zip(all_mass_sorted, keep) if k]
slope_D = np.array([entry["Dry Slope (D)"] for entry in all_entries], dtype=float)
rain_mm = accum_rain_base
plt.figure(figsize=(6, 4.5))
colors = plt.cm.cool(np.linspace(0, 1, len(slope_D)))
for i, (D, r, c) in enumerate(zip(slope_D, rain_mm, colors), start=1):
    plt.scatter(D, r, s=80, edgecolor='k', color=c)
log_rain = np.log10(rain_mm)
slope_lr, intercept_lr, r_val2, p_val2, _ = linregress(slope_D, log_rain)
D_sorted = np.sort(slope_D)
rain_fit2 = 10 ** (intercept_lr + slope_lr * D_sorted)
plt.plot(D_sorted, rain_fit2, "r--", lw=2,
         label=f"Fit: slope={slope_lr:.2f}, R={r_val2:.2f}")
plt.yscale("log")
plt.xlabel("Dry Slope D (µm)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")
plt.title("BCB January - June 2022\n Base 385 g m$^{-2}$ LWP\nLess than 100 µg/m³ Mass",
          fontweight="bold", fontsize=18)
#plt.title("BCB January - June 2022\n Base 385 g m$^{-2}$ LWP",
        #   fontweight="bold", fontsize=18)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()



#%%
#mass versus slope D
plt.figure(figsize=(6, 4.5))
colors = plt.cm.inferno(np.linspace(0, 1, len(mass)))
for i, (m, D, c) in enumerate(zip(mass, slope_D, colors), start=1):
    plt.scatter(m, D, s=80, edgecolor='k', color=c)
plt.xscale('log')
plt.xlabel("Dry GCCN Mass (µg/m³)", fontsize=16, fontweight="bold")
plt.ylabel("Dry Slope D (µm)", fontsize=16, fontweight="bold")
# plt.title("BCB January - June 2022\n Base 385 g m$^{-2}$ LWP", 
#           fontweight="bold", fontsize=18)
plt.title("BCB January - June 2022\n Base 385 g m$^{-2}$ LWP\nLess than 100 µg/m³ Mass", 
          fontweight="bold", fontsize=18)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
#mass versus slope correlation coefficient
log_mass = np.log10(mass)
slope_coeff, intercept_coeff, r_val3, p_val3, _ = linregress(log_mass, slope_D)
print(f"Correlation between log10(Mass) and Slope D:")
print(f"  R = {r_val3:.4f}, R² = {r_val3**2:.4f}")
#%%
#mass versus gccn
plt.figure(figsize=(6, 4.5))
colors = plt.cm.magma(np.linspace(0, 1, len(mass)))
for i, (m, g, c) in enumerate(zip(mass, gccn_m3_base, colors), start=1):
    plt.scatter(m, g, s=80, edgecolor='k', color=c) 
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Dry GCCN Mass (µg/m³)", fontsize=16, fontweight="bold")
plt.ylabel("GCCN Concentration (m$^{-3}$)", fontsize=16, fontweight="bold")
plt.title("BCB January - June 2022\n Base 385 g m$^{-2}$ LWP\nLess than 100 µg/m³ Mass", 
          fontweight="bold", fontsize=18)
# plt.title("BCB January - June 2022\n Base 385 g m$^{-2}$ LWP", 
#           fontweight="bold", fontsize=18)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
#mass versus gccn correlation coefficient
log_mass = np.log10(mass)
log_gccn = np.log10(gccn_m3_base)
slope_coeff2, intercept_coeff2, r_val4, p_val4, _ = linregress(log_mass, log_gccn)
print(f"Correlation between log10(Mass) and log10(GCCN):")
print(f"  R = {r_val4:.4f}, R² = {r_val4**2:.4f}")
 
# %%
