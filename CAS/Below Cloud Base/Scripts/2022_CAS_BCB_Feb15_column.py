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
#no turbulence 
sys.modules.setdefault('numpy.core', np)
sys.modules.setdefault('numpy.core.multiarray', np.core.multiarray)
sys.modules.setdefault('numpy._core', np)
sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)

base = "/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions/Feb15columnparcel"

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
R_data  = inspect_pickle("R.pkl")
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
LWP = rain_t   # because rain_t is actually LWP(t)
precip_accum = np.max(LWP, axis=1)[:, None] - LWP 
precip_masked = np.where(time >= 800, precip_accum, np.nan)
plt.figure(figsize=(8, 5))
for i in range(precip_masked.shape[0]):
    plt.plot(time, precip_masked[i], lw=1.5, alpha=0.85)

plt.xlabel("Time (s)", fontweight="bold", fontsize=16)
plt.ylabel("Accumulated Rain (mm)", fontweight="bold", fontsize=16)
plt.title("BCB February 15 — Column Simulation\nAccumulated Precipitation (max(LWP) − LWP)", 
          fontweight="bold", fontsize=18)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.legend([f"Leg {i+1}" for i in range(precip_masked.shape[0])],
           bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.xticks(fontsize=14, fontweight="bold")
plt.yticks(fontsize=14, fontweight="bold")
plt.show()

#%%
for i in range(LWP.shape[0]):
    plt.figure(figsize=(6, 4))
    plt.plot(time, LWP[i, :], lw=2) 
    plt.title(f"BCB February 15\nColumn Parcel Model\nLeg {i+1}", 
              fontweight="bold", fontsize=18)
    plt.xlabel("Time (s)", fontweight="bold", fontsize=16)
    plt.ylabel("Liquid Water Path (kg m$^{-2}$)", 
               fontweight="bold", fontsize=16) 
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
    plt.ylabel("Number Concentration (m⁻³)", fontweight="bold", fontsize=16)
    plt.yscale("log")
    plt.ylim(1, 1e8)
    plt.xscale("log")
    plt.yticks(fontweight="bold", fontsize=14)
    plt.xticks(fontweight="bold", fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

#%%
#cumulative size distributions 
dr = np.diff(r_dry)
for i in range(n0_r.shape[0]):
    cumulative = np.cumsum(n0_r[i, ::-1])[::-1]
    plt.figure(figsize=(6, 4))
    plt.plot(r_dry, cumulative, lw=2)
    plt.title(f"February 15, 2022\nCumulative dry size distribution\nLeg {i+1}",
              fontweight="bold", fontsize=18)
    plt.xlabel("Dry radius (m)", fontweight="bold", fontsize=16)
    plt.ylabel("Cumulative Number \nConcentration (m⁻³)", fontweight="bold", fontsize=16)
    plt.yscale("log")
    plt.ylim(1, 1e8)
    plt.xscale("log")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
# %%
dr = np.diff(r_dry)
dr = np.append(dr, dr[-1])
dlnr = np.diff(np.log(r_dry))
dlnr = np.append(dlnr, dlnr[-1])
i = 0 
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
total_m3 = np.sum(n0_r, axis=1)
mask = r_dry > 0.5e-6
gccn_m3 = np.sum(n0_r[:, mask], axis=1)
for i, (tot, gccn) in enumerate(zip(total_m3, gccn_m3), start=1):
    frac = gccn / tot
    print(f"Leg {i:02d}: Total={tot:.3e} m^-3, GCCN={gccn:.3e} m^-3, GCCN/Total={frac:.2e}")

# %%
mask = r_dry > 0.5e-6  # radius > 0.5 µm → diameter > 1 µm
gccn_m3 = np.sum(n0_r[:, mask], axis=1)
accum_rain = np.max(LWP, axis=1) - LWP[:, -1]  # units: kg m^-2 = mm
for i, (gccn, rain) in enumerate(zip(gccn_m3, accum_rain), start=1):
    print(f"Leg {i:02d}: GCCN={gccn:.3e} m^-3, Rain={rain:.3f} mm")
plt.figure(figsize=(6, 4.5))
colors = plt.cm.viridis(np.linspace(0, 1, len(gccn_m3)))
for i, (gccn, rain, c) in enumerate(zip(gccn_m3, accum_rain, colors), start=1):
    plt.scatter(gccn, rain, s=80, edgecolor='k', color=c, label=f"Leg {i}")
logx = np.log10(gccn_m3)
logy = np.log10(accum_rain)
slope, intercept, r_value, p_value, std_err = linregress(logx, logy)
x_sorted = np.sort(gccn_m3)
y_fit_sorted = 10 ** (intercept + slope * np.log10(x_sorted))
plt.plot(x_sorted, y_fit_sorted, "r--", lw=2,
         label=f"Fit: slope={slope:.2f}, R={r_value:.2f}")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("GCCN concentration (m$^{-3}$)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")  # Correct units!
plt.title("BCB February 15 \nGCCN vs Accumulated Rain", fontsize=16, fontweight="bold")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.grid(alpha=0.3)
plt.yticks(fontweight="bold", fontsize=14)
plt.xticks(fontweight="bold", fontsize=14)
plt.tight_layout()
plt.show()
# %% 
#turbulence 
sys.modules.setdefault('numpy.core', np)
sys.modules.setdefault('numpy.core.multiarray', np.core.multiarray)
sys.modules.setdefault('numpy._core', np)
sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)

base = "/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions/Feb15columturbulence"

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
Rturb_data  = inspect_pickle("R_turb.pkl")
r_dry = n0_data[0]
n0_r  = n0_data[1]
extra = n0_data[2] if len(n0_data) > 2 else None
print("\nExtracted:")
print("  r_dry:", np.shape(r_dry))
print("  n0_r :", np.shape(n0_r))
print("  extra data:", type(extra), "\n")
time_turb = Rturb_data[0]
rain_turb = Rturb_data[1]
extra_Rturb = Rturb_data[2] if len(Rturb_data) > 2 else None
print("  time  :", np.shape(time_turb))
print("  rain_t:", np.shape(rain_turb))
print("  extra R data:", type(extra_Rturb))
#%%
LWP_turb = rain_turb 
precip_accum_turb = np.max(LWP_turb, axis=1)[:, None] - LWP_turb
precip_masked_turb = np.where(time_turb >= 800, precip_accum_turb, np.nan)
plt.figure(figsize=(8, 5))
for i in range(precip_masked_turb.shape[0]):
    plt.plot(time_turb, precip_masked_turb[i], lw=1.5, alpha=0.85)

plt.xlabel("Time (s)", fontweight="bold", fontsize=16)
plt.ylabel("Accumulated Rain (mm)", fontweight="bold", fontsize=16)
plt.title("BCB February 15 \nTurbulence Column Simulation\nAccumulated Precipitation", 
          fontweight="bold", fontsize=18)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.legend([f"Leg {i+1}" for i in range(precip_masked.shape[0])],
           bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.xticks(fontsize=14, fontweight="bold")
plt.yticks(fontsize=14, fontweight="bold")
plt.show()

#%%
for i in range(LWP.shape[0]):
    plt.figure(figsize=(6, 4))
    plt.plot(time, LWP[i, :], lw=2) 
    plt.title(f"BCB February 15\nColumn Parcel Model\nLeg {i+1}", 
              fontweight="bold", fontsize=18)
    plt.xlabel("Time (s)", fontweight="bold", fontsize=16)
    plt.ylabel("Liquid Water Path (kg m$^{-2}$)", 
               fontweight="bold", fontsize=16) 
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
    plt.ylabel("Number Concentration (m⁻³)", fontweight="bold", fontsize=16)
    plt.yscale("log")
    plt.ylim(1, 1e8)
    plt.xscale("log")
    plt.yticks(fontweight="bold", fontsize=14)
    plt.xticks(fontweight="bold", fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

#%%
#cumulative size distributions 
dr = np.diff(r_dry)
for i in range(n0_r.shape[0]):
    cumulative = np.cumsum(n0_r[i, ::-1])[::-1]
    plt.figure(figsize=(6, 4))
    plt.plot(r_dry, cumulative, lw=2)
    plt.title(f"February 15, 2022\nCumulative dry size distribution\nLeg {i+1}",
              fontweight="bold", fontsize=18)
    plt.xlabel("Dry radius (m)", fontweight="bold", fontsize=16)
    plt.ylabel("Cumulative Number \nConcentration (m⁻³)", fontweight="bold", fontsize=16)
    plt.yscale("log")
    plt.ylim(1, 1e8)
    plt.xscale("log")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
# %%
# Calculate total and GCCN number concentrations per leg
total_m3 = np.sum(n0_r, axis=1)
mask = r_dry > 0.5e-6
gccn_m3 = np.sum(n0_r[:, mask], axis=1)
for i, (tot, gccn) in enumerate(zip(total_m3, gccn_m3), start=1):
    frac = gccn / tot
    print(f"Leg {i:02d}: Total={tot:.3e} m^-3, GCCN={gccn:.3e} m^-3, GCCN/Total={frac:.2e}")

# %%
mask = r_dry > 0.5e-6  # radius > 0.5 µm → diameter > 1 µm
gccn_m3 = np.sum(n0_r[:, mask], axis=1)
accum_rain = np.max(LWP, axis=1) - LWP[:, -1]  # units: kg m^-2 = mm
for i, (gccn, rain) in enumerate(zip(gccn_m3, accum_rain), start=1):
    print(f"Leg {i:02d}: GCCN={gccn:.3e} m^-3, Rain={rain:.3f} mm")
plt.figure(figsize=(6, 4.5))
colors = plt.cm.viridis(np.linspace(0, 1, len(gccn_m3)))
for i, (gccn, rain, c) in enumerate(zip(gccn_m3, accum_rain, colors), start=1):
    plt.scatter(gccn, rain, s=80, edgecolor='k', color=c, label=f"Leg {i}")
logx = np.log10(gccn_m3)
logy = np.log10(accum_rain)
slope, intercept, r_value, p_value, std_err = linregress(logx, logy)
x_sorted = np.sort(gccn_m3)
y_fit_sorted = 10 ** (intercept + slope * np.log10(x_sorted))
plt.plot(x_sorted, y_fit_sorted, "r--", lw=2,
         label=f"Fit: slope={slope:.2f}, R={r_value:.2f}")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("GCCN concentration (m$^{-3}$)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")  # Correct units!
plt.title("BCB February 15\n Turbulence simulation \nGCCN vs Accumulated Rain", fontsize=16, fontweight="bold")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.grid(alpha=0.3)
plt.yticks(fontweight="bold", fontsize=14)
plt.xticks(fontweight="bold", fontsize=14)
plt.tight_layout()
plt.show()
# %%
#comparing both simulations
plt.figure(figsize=(8, 5))
for i in range(precip_masked.shape[0]):
    plt.plot(time, precip_masked[i], color='red', lw=1.5, alpha=0.7)
for i in range(precip_masked_turb.shape[0]):
    plt.plot(time_turb, precip_masked_turb[i], color='blue', lw=1.5, alpha=0.7)
plt.xlabel("Time (s)", fontweight="bold", fontsize=16)
plt.ylabel("Accumulated Rain (mm)", fontweight="bold", fontsize=16)
plt.title("BCB Feb 15 Column Parcel\nAccumulated Rain: Turbulence vs No-Turbulence", 
          fontweight="bold", fontsize=18)
plt.grid(alpha=0.3)
plt.plot([], [], color='red', lw=3, label="No Turbulence")
plt.plot([], [], color='blue', lw=3, label="Turbulence")
plt.legend(loc="upper left", fontsize=12)
plt.xticks(fontsize=14, fontweight="bold")
plt.yticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# %%
#mass analysis
feb15_mass = [entry for entry in dry_mass_data_inf 
              if entry['Date'] == '2022-02-15']

print("Number of Feb 15 entries:", len(feb15_mass))  # should be 14
feb15_mass_sorted = sorted(feb15_mass, key=lambda x: x['BCB_start'])
feb15_mass_values = [entry['Dry Mass (µg/m³)'] for entry in feb15_mass_sorted]
for i, mass in enumerate(feb15_mass_values, start=1):
    print(f"Leg {i}: {mass:.2f} µg/m³")

# %%
#turbulence
mass = np.array(feb15_mass_values)
rain = accum_rain 
for i, (m, r) in enumerate(zip(mass, rain), start=1):
    print(f"Leg {i:02d}: Mass={m:.2f} µg/m³, Rain={r:.3f} mm")
plt.figure(figsize=(6, 4.5))
colors = plt.cm.plasma(np.linspace(0, 1, len(mass)))

for i, (m, r, c) in enumerate(zip(mass, rain, colors), start=1):
    plt.scatter(m, r, s=80, edgecolor='k', color=c, label=f"Leg {i}")
logx = np.log10(mass)
logy = np.log10(rain)
slope, intercept, r_value, p_value, std_err = linregress(logx, logy)
x_sorted = np.sort(mass)
y_fit_sorted = 10 ** (intercept + slope * np.log10(x_sorted))
plt.plot(x_sorted, y_fit_sorted, "r--", lw=2,
         label=f"Fit: slope={slope:.2f}, R={r_value:.2f}")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Dry GCCN Mass (µg/m³)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")
plt.title("BCB February 15\nTurbulence Simulation\nDry Mass vs Accumulated Rain",
          fontsize=16, fontweight="bold")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.grid(alpha=0.3)
plt.yticks(fontweight="bold", fontsize=14)
plt.xticks(fontweight="bold", fontsize=14)
plt.tight_layout()
plt.show()
#%%
#looking at removing mass effect 
feb15_slopes = [entry['Dry Slope (D)'] for entry in feb15_mass_sorted]
feb15_intercepts = [entry['Dry Intercept (N0)'] for entry in feb15_mass_sorted]
for i, (D, n0) in enumerate(zip(feb15_slopes, feb15_intercepts), start=1):
    print(f"Leg {i:02d}: Slope D={D:.3f} µm, N0={n0:.3f} cm^-3 µm^-1")
predicted_rain = 10 ** (intercept + slope * np.log10(mass))
residuals = rain - predicted_rain
from scipy.stats import linregress
slope_val, intercept_val, R_slope, p_slope, _ = linregress(feb15_slopes, residuals)
print("\nResidual correlation with Dry Slope D:")
print("  R =", R_slope, "p =", p_slope)
plt.figure(figsize=(6,4))
plt.scatter(feb15_slopes, residuals, s=90, edgecolor='k', c='dodgerblue')
plt.axhline(0, color='k', linestyle='--')
plt.xlabel("Dry Slope D (µm)", fontsize=14, fontweight="bold")
plt.ylabel("Residual Rain (mm)", fontsize=14, fontweight="bold")
plt.title("Residual Rain vs GCCN Slope\n(After Removing Mass Effect)",
          fontsize=14, fontweight="bold")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
#%%
bin_centers = r_dry  # m
num_bins = len(bin_centers)
corr_R = []
corr_p = []
valid_bins = [] 
threshold = 1e2 
for i in range(num_bins):
    conc = n0_r[:, i]
    if np.mean(conc) > threshold: 
        slope_i, intercept_i, R_i, p_i, _ = linregress(conc, residuals)
        corr_R.append(R_i)
        corr_p.append(p_i)
        valid_bins.append(bin_centers[i])
corr_R = np.array(corr_R)
corr_p = np.array(corr_p)
valid_bins = np.array(valid_bins)
print("\nTop 5 real bins correlated with residual rainfall:")
top = np.argsort(np.abs(corr_R))[::-1][:5]
for idx in top:
    print(f"Radius = {valid_bins[idx]*1e6:.2f} µm | R={corr_R[idx]:.3f}   p={corr_p[idx]:.3f}")
plt.figure(figsize=(7,5))
plt.plot(valid_bins*1e6, corr_R, '-o', lw=2)
plt.axhline(0, color='k', linestyle='--')
plt.xlabel("Dry radius (µm)", fontsize=16, fontweight="bold")
plt.ylabel("Correlation R", fontsize=16, fontweight="bold")
plt.title("Residual Rainfall vs GCCN Number by Size\n(Filtered Meaningful Bins Only)",
          fontsize=16, fontweight="bold")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
#%%

# %%
#no turbulence 
mass_no_turb = np.array(feb15_mass_values)  # must match leg order
rain_no_turb = accum_rain  # from R.pkl (no turbulence)
for i, (m, r) in enumerate(zip(mass_no_turb, rain_no_turb), start=1):
    print(f"Leg {i:02d}: Mass={m:.2f} µg/m³, Rain={r:.3f} mm")
plt.figure(figsize=(6, 4.5))
colors = plt.cm.plasma(np.linspace(0, 1, len(mass_no_turb)))

for i, (m, r, c) in enumerate(zip(mass_no_turb, rain_no_turb, colors), start=1):
    plt.scatter(m, r, s=80, edgecolor='k', color=c, label=f"Leg {i}")
logx = np.log10(mass_no_turb)
logy = np.log10(rain_no_turb)
slope, intercept, r_value, p_value, std_err = linregress(logx, logy)
x_sorted = np.sort(mass_no_turb)
y_fit_sorted = 10 ** (intercept + slope * np.log10(x_sorted))
plt.plot(x_sorted, y_fit_sorted, "r--", lw=2,
         label=f"Fit: slope={slope:.2f}, R={r_value:.2f}")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Dry GCCN Mass (µg/m³)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")
plt.title("BCB February 15\n No Turbulence\nDry Mass vs Accumulated Rain",
          fontsize=16, fontweight="bold")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.grid(alpha=0.3)
plt.yticks(fontweight="bold", fontsize=14)
plt.xticks(fontweight="bold", fontsize=14)
plt.tight_layout()
plt.show()

# %%
