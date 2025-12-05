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
for i in range(LWP_turb.shape[0]):
    plt.figure(figsize=(6, 4))
    plt.plot(time_turb, LWP_turb[i, :], lw=2) 
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
accum_rain = np.max(LWP_turb, axis=1) - LWP_turb[:, -1]  # units: kg m^-2 = mm
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
#plotting slope D vs rain directly
feb15_mass = [entry for entry in dry_mass_data_inf 
              if entry['Date'] == '2022-02-15']
feb15_mass_sorted = sorted(feb15_mass, key=lambda x: x['BCB_start'])
feb15_mass_values = np.array([entry['Dry Mass (µg/m³)'] for entry in feb15_mass_sorted])
feb15_slopes      = np.array([entry['Dry Slope (D)']       for entry in feb15_mass_sorted])
feb15_intercepts  = np.array([entry['Dry Intercept (N0)']  for entry in feb15_mass_sorted])
for i, (m, D) in enumerate(zip(feb15_mass_values, feb15_slopes), start=1):
    print(f"Leg {i:02d}: Mass={m:.2f} µg/m³, Slope D={D:.3f} µm")
slope_D = feb15_slopes
rain_mm = rain
plt.figure(figsize=(6, 4.5))
colors = plt.cm.cool(np.linspace(0, 1, len(slope_D)))
for i, (D, r, c) in enumerate(zip(slope_D, rain_mm, colors), start=1):
    plt.scatter(D, r, s=80, edgecolor='k', color=c, label=f"Leg {i}")
log_rain = np.log10(rain_mm)
slope_lr, intercept_lr, r_val2, p_val2, _ = linregress(slope_D, log_rain)

D_sorted = np.sort(slope_D)
rain_fit2 = 10 ** (intercept_lr + slope_lr * D_sorted)
plt.plot(D_sorted, rain_fit2, "r--", lw=2,
         label=f"Fit: slope={slope_lr:.2f}, R={r_val2:.2f}")
plt.yscale("log")
plt.xlabel("Dry Slope D (µm)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")
plt.title("BCB February 15\nSlope vs Accumulated Rain",
          fontsize=16, fontweight="bold")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
#%%
#multiple linear regression for mass and slope D
from sklearn.linear_model import LinearRegression
X = np.column_stack((np.log10(mass), feb15_slopes))
y = rain  # mm
model = LinearRegression().fit(X, y)
intercept = model.intercept_
coef_logM, coef_D = model.coef_
print(f"Intercept:        {intercept:.4f} mm")
print(f"β_log10(Mass):    {coef_logM:.4f} mm per log10(µg/m³)")
print(f"β_Slope (D):      {coef_D:.4f} mm per µm slope")
R2 = model.score(X, y)
print(f"R² = {R2:.4f}")
#%%
#p-values
from scipy.stats import t
y_pred = model.predict(X)
resid = y - y_pred
n = len(y)
p = X.shape[1] 
s2 = np.sum(resid**2) / (n - p - 1)
XTX_inv = np.linalg.inv(X.T @ X)
var_b = s2 * XTX_inv.diagonal()  # variances of betas
se_b = np.sqrt(var_b)            # std errors
t_stats = np.array([coef_logM, coef_D]) / se_b[1:]  # skip intercept
p_vals = 2 * (1 - t.cdf(np.abs(t_stats), df=n - p - 1))
print("\n🔍 p-values:")
print(f"  p for log10(Mass): {p_vals[0]:.4f}")
print(f"  p for Slope D:     {p_vals[1]:.4f}")

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
#no turbulence low Na and low no
sys.modules.setdefault('numpy.core', np)
sys.modules.setdefault('numpy.core.multiarray', np.core.multiarray)
sys.modules.setdefault('numpy._core', np)
sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)

base = "/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions/Feb15difflwp"

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
lown0_data = inspect_pickle("n0_r_lowNa.pkl")
lowR_data  = inspect_pickle("R_lowNa.pkl")
lowr_dry = lown0_data[0]
lown0_r  = lown0_data[1]
extra = lown0_data[2] if len(lown0_data) > 2 else None
print("\nExtracted:")
print("  r_dry:", np.shape(lowr_dry))
print("  n0_r :", np.shape(lown0_r))
print("  extra data:", type(extra), "\n")
timelow = lowR_data[0]
rainlow = lowR_data[1]
extra_R = lowR_data[2] if len(lowR_data) > 2 else None
print("  time  :", np.shape(timelow))
print("  rain_t:", np.shape(rainlow))
print("  extra R data:", type(extra_R))
#%%
lownaa = rainlow   # because rainlow is actually LWP(t)
precip_accum = np.max(lownaa, axis=1)[:, None] - lownaa 
precip_masked = np.where(timelow >= 800, precip_accum, np.nan)
plt.figure(figsize=(8, 5))
for i in range(precip_masked.shape[0]):
    plt.plot(timelow, precip_masked[i], lw=1.5, alpha=0.85)

plt.xlabel("Time (s)", fontweight="bold", fontsize=16)
plt.ylabel("Accumulated Rain (mm)", fontweight="bold", fontsize=16)
plt.title("BCB February 15\n Non-turbulent Low Na\nAccumulated Precipitation", 
          fontweight="bold", fontsize=18)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.legend([f"Leg {i+1}" for i in range(precip_masked.shape[0])],
           bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.xticks(fontsize=14, fontweight="bold")
plt.yticks(fontsize=14, fontweight="bold")
plt.show()
# %%
for i in range(lownaa.shape[0]):
    plt.figure(figsize=(6, 4))
    plt.plot(timelow, lownaa[i, :], lw=2) 
    plt.title(f"BCB February 15\nNon-turbulent Low Na\nLeg {i+1}", 
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
for i in range(lown0_r.shape[0]):
    plt.figure(figsize=(6, 4))
    plt.plot(lowr_dry, lown0_r[i, :], lw=2)
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
dr = np.diff(lowr_dry)
for i in range(lown0_r.shape[0]):
    cumulative = np.cumsum(lown0_r[i, ::-1])[::-1]
    plt.figure(figsize=(6, 4))
    plt.plot(lowr_dry, cumulative, lw=2)
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
#%%
# Calculate total and GCCN number concentrations per leg
lowtotal_m3 = np.sum(lown0_r, axis=1)
lowmask = lowr_dry > 0.5e-6
lowgccn_m3 = np.sum(lown0_r[:, lowmask], axis=1)
for i, (tot, gccn) in enumerate(zip(lowtotal_m3, lowgccn_m3), start=1):
    frac = gccn / tot
    print(f"Leg {i:02d}: Total={tot:.3e} m^-3, GCCN={gccn:.3e} m^-3, GCCN/Total={frac:.2e}")

# %%
lowmask = lowr_dry > 0.5e-6  # radius > 0.5 µm → diameter > 1 µm
lowgccn_m3 = np.sum(lown0_r[:, lowmask], axis=1)
accum_rain = np.max(lownaa, axis=1) - lownaa[:, -1]  # units: kg m^-2 = mm
for i, (gccn, rain) in enumerate(zip(lowgccn_m3, accum_rain), start=1):
    print(f"Leg {i:02d}: GCCN={gccn:.3e} m^-3, Rain={rain:.3f} mm")
plt.figure(figsize=(6, 4.5))
colors = plt.cm.viridis(np.linspace(0, 1, len(lowgccn_m3)))
for i, (gccn, rain, c) in enumerate(zip(lowgccn_m3, accum_rain, colors), start=1):
    plt.scatter(gccn, rain, s=80, edgecolor='k', color=c, label=f"Leg {i}")
logx = np.log10(lowgccn_m3)
logy = np.log10(accum_rain)
slope, intercept, r_value, p_value, std_err = linregress(logx, logy)
x_sorted = np.sort(lowgccn_m3)
y_fit_sorted = 10 ** (intercept + slope * np.log10(x_sorted))
plt.plot(x_sorted, y_fit_sorted, "r--", lw=2,
         label=f"Fit: slope={slope:.2f}, R={r_value:.2f}")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("GCCN concentration (m$^{-3}$)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")  # Correct units!
plt.title("BCB February 15 \nNon-turbulent Low Na\nGCCN vs Accumulated Rain", fontsize=16, fontweight="bold")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.grid(alpha=0.3)
plt.yticks(fontweight="bold", fontsize=14)
plt.xticks(fontweight="bold", fontsize=14)
plt.tight_layout()
plt.show()

# %%
mass = np.array(feb15_mass_values)
lowrain=accum_rain
for i, (m, r) in enumerate(zip(mass, lowrain), start=1):
    print(f"Leg {i:02d}: Mass={m:.2f} µg/m³, Rain={r:.3f} mm")
plt.figure(figsize=(6, 4.5))
colors = plt.cm.plasma(np.linspace(0, 1, len(mass)))

for i, (m, r, c) in enumerate(zip(mass, lowrain, colors), start=1):
    plt.scatter(m, r, s=80, edgecolor='k', color=c, label=f"Leg {i}")
logx = np.log10(mass)
logy = np.log10(lowrain)
slope, intercept, r_value, p_value, std_err = linregress(logx, logy)
x_sorted = np.sort(mass)
y_fit_sorted = 10 ** (intercept + slope * np.log10(x_sorted))
plt.plot(x_sorted, y_fit_sorted, "r--", lw=2,
         label=f"Fit: slope={slope:.2f}, R={r_value:.2f}")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Dry GCCN Mass (µg/m³)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")
plt.title("BCB February 15\nNon-turbulent Low Na\nDry Mass vs Accumulated Rain",
          fontsize=16, fontweight="bold")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.grid(alpha=0.3)
plt.yticks(fontweight="bold", fontsize=14)
plt.xticks(fontweight="bold", fontsize=14)
plt.tight_layout()
plt.show()
# %%
#plotting slope D vs rain directly
feb15_mass = [entry for entry in dry_mass_data_inf 
              if entry['Date'] == '2022-02-15']
feb15_mass_sorted = sorted(feb15_mass, key=lambda x: x['BCB_start'])
feb15_mass_values = np.array([entry['Dry Mass (µg/m³)'] for entry in feb15_mass_sorted])
feb15_slopes      = np.array([entry['Dry Slope (D)']       for entry in feb15_mass_sorted])
feb15_intercepts  = np.array([entry['Dry Intercept (N0)']  for entry in feb15_mass_sorted])
for i, (m, D) in enumerate(zip(feb15_mass_values, feb15_slopes), start=1):
    print(f"Leg {i:02d}: Mass={m:.2f} µg/m³, Slope D={D:.3f} µm")
slope_D = feb15_slopes
rain_mm = lowrain
plt.figure(figsize=(6, 4.5))
colors = plt.cm.cool(np.linspace(0, 1, len(slope_D)))
for i, (D, r, c) in enumerate(zip(slope_D, rain_mm, colors), start=1):
    plt.scatter(D, r, s=80, edgecolor='k', color=c, label=f"Leg {i}")
log_rain = np.log10(rain_mm)
slope_lr, intercept_lr, r_val2, p_val2, _ = linregress(slope_D, log_rain)

D_sorted = np.sort(slope_D)
rain_fit2 = 10 ** (intercept_lr + slope_lr * D_sorted)
plt.plot(D_sorted, rain_fit2, "r--", lw=2,
         label=f"Fit: slope={slope_lr:.2f}, R={r_val2:.2f}")
plt.yscale("log")
plt.xlabel("Dry Slope D (µm)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")
plt.title("BCB February 15\nNon-turbulent Low Na\nSlope vs Accumulated Rain",
          fontsize=16, fontweight="bold")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
#%%
#multiple linear regression for mass and slope D
from sklearn.linear_model import LinearRegression
X = np.column_stack((np.log10(mass), feb15_slopes))
y = lowrain  # mm
model = LinearRegression().fit(X, y)
intercept = model.intercept_
coef_logM, coef_D = model.coef_
print(f"Intercept:        {intercept:.4f} mm")
print(f"β_log10(Mass):    {coef_logM:.4f} mm per log10(µg/m³)")
print(f"β_Slope (D):      {coef_D:.4f} mm per µm slope")
R2 = model.score(X, y)
print(f"R² = {R2:.4f}")
#%%
#no turbulence high Na and high no
sys.modules.setdefault('numpy.core', np)
sys.modules.setdefault('numpy.core.multiarray', np.core.multiarray)
sys.modules.setdefault('numpy._core', np)
sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)

base = "/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions/Feb15difflwp"

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
highn0_data = inspect_pickle("n0_r_hiNa.pkl")
highR_data  = inspect_pickle("R_hiNa.pkl")
highr_dry = highn0_data[0]
highn0_r  = highn0_data[1]
extra = highn0_data[2] if len(highn0_data) > 2 else None
print("\nExtracted:")
print("  r_dry:", np.shape(highr_dry))
print("  n0_r :", np.shape(highn0_r))
print("  extra data:", type(extra), "\n")
timehigh = highR_data[0]
rainhigh = highR_data[1]
extra_R = highR_data[2] if len(highR_data) > 2 else None
print("  time  :", np.shape(timehigh))
print("  rain_t:", np.shape(rainhigh))
print("  extra R data:", type(extra_R))
#%%
LWPhigh = rainhigh   # because rainhigh is actually LWP(t)
precip_accum = np.max(LWPhigh, axis=1)[:, None] - LWPhigh 
precip_masked = np.where(timehigh >= 800, precip_accum, np.nan)
plt.figure(figsize=(8, 5))
for i in range(precip_masked.shape[0]):
    plt.plot(timehigh, precip_masked[i], lw=1.5, alpha=0.85)

plt.xlabel("Time (s)", fontweight="bold", fontsize=16)
plt.ylabel("Accumulated Rain (mm)", fontweight="bold", fontsize=16)
plt.title("BCB February 15\n Non-turbulent High Na\nAccumulated Precipitation", 
          fontweight="bold", fontsize=18)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.legend([f"Leg {i+1}" for i in range(precip_masked.shape[0])],
           bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.xticks(fontsize=14, fontweight="bold")
plt.yticks(fontsize=14, fontweight="bold")
plt.show()
# %%
for i in range(LWPhigh.shape[0]):
    plt.figure(figsize=(6, 4))
    plt.plot(timehigh, LWPhigh[i, :], lw=2) 
    plt.title(f"BCB February 15\nNon-turbulent High Na\nLeg {i+1}", 
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
for i in range(highn0_r.shape[0]):
    plt.figure(figsize=(6, 4))
    plt.plot(highr_dry, highn0_r[i, :], lw=2)
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
dr = np.diff(highr_dry)
for i in range(highn0_r.shape[0]):
    cumulative = np.cumsum(highn0_r[i, ::-1])[::-1]
    plt.figure(figsize=(6, 4))
    plt.plot(highr_dry, cumulative, lw=2)
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
#%%
# Calculate total and GCCN number concentrations per leg
hightotal_m3 = np.sum(highn0_r, axis=1)
highmask = highr_dry > 0.5e-6
highgccn_m3 = np.sum(highn0_r[:, highmask], axis=1)
for i, (tot, gccn) in enumerate(zip(hightotal_m3, highgccn_m3), start=1):
    frac = gccn / tot
    print(f"Leg {i:02d}: Total={tot:.3e} m^-3, GCCN={gccn:.3e} m^-3, GCCN/Total={frac:.2e}")

# %%
highmask = highr_dry > 0.5e-6  # radius > 0.5 µm → diameter > 1 µm
highgccn_m3 = np.sum(highn0_r[:, highmask], axis=1)
accum_rain = np.max(LWPhigh, axis=1) - LWPhigh[:, -1]  # units: kg m^-2 = mm
for i, (gccn, rain) in enumerate(zip(highgccn_m3, accum_rain), start=1):
    print(f"Leg {i:02d}: GCCN={gccn:.3e} m^-3, Rain={rain:.3f} mm")
plt.figure(figsize=(6, 4.5))
colors = plt.cm.viridis(np.linspace(0, 1, len(highgccn_m3)))
for i, (gccn, rain, c) in enumerate(zip(highgccn_m3, accum_rain, colors), start=1):
    plt.scatter(gccn, rain, s=80, edgecolor='k', color=c, label=f"Leg {i}")
logx = np.log10(highgccn_m3)
logy = np.log10(accum_rain)
slope, intercept, r_value, p_value, std_err = linregress(logx, logy)
x_sorted = np.sort(highgccn_m3)
y_fit_sorted = 10 ** (intercept + slope * np.log10(x_sorted))
plt.plot(x_sorted, y_fit_sorted, "r--", lw=2,
         label=f"Fit: slope={slope:.2f}, R={r_value:.2f}")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("GCCN concentration (m$^{-3}$)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")  # Correct units!
plt.title("BCB February 15 \nNon-turbulent High Na\nGCCN vs Accumulated Rain", fontsize=16, fontweight="bold")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.grid(alpha=0.3)
plt.yticks(fontweight="bold", fontsize=14)
plt.xticks(fontweight="bold", fontsize=14)
plt.tight_layout()
plt.show()

# %%
mass = np.array(feb15_mass_values)
highrain=accum_rain
for i, (m, r) in enumerate(zip(mass, highrain), start=1):
    print(f"Leg {i:02d}: Mass={m:.2f} µg/m³, Rain={r:.3f} mm")
plt.figure(figsize=(6, 4.5))
colors = plt.cm.plasma(np.linspace(0, 1, len(mass)))

for i, (m, r, c) in enumerate(zip(mass, highrain, colors), start=1):
    plt.scatter(m, r, s=80, edgecolor='k', color=c, label=f"Leg {i}")
logx = np.log10(mass)
logy = np.log10(highrain)
slope, intercept, r_value, p_value, std_err = linregress(logx, logy)
x_sorted = np.sort(mass)
y_fit_sorted = 10 ** (intercept + slope * np.log10(x_sorted))
plt.plot(x_sorted, y_fit_sorted, "r--", lw=2,
         label=f"Fit: slope={slope:.2f}, R={r_value:.2f}")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Dry GCCN Mass (µg/m³)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")
plt.title("BCB February 15\nNon-turbulent High Na\nDry Mass vs Accumulated Rain",
          fontsize=16, fontweight="bold")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.grid(alpha=0.3)
plt.yticks(fontweight="bold", fontsize=14)
plt.xticks(fontweight="bold", fontsize=14)
plt.tight_layout()
plt.show()
# %%
#plotting slope D vs rain directly
feb15_mass = [entry for entry in dry_mass_data_inf 
              if entry['Date'] == '2022-02-15']
feb15_mass_sorted = sorted(feb15_mass, key=lambda x: x['BCB_start'])
feb15_mass_values = np.array([entry['Dry Mass (µg/m³)'] for entry in feb15_mass_sorted])
feb15_slopes      = np.array([entry['Dry Slope (D)']       for entry in feb15_mass_sorted])
feb15_intercepts  = np.array([entry['Dry Intercept (N0)']  for entry in feb15_mass_sorted])
for i, (m, D) in enumerate(zip(feb15_mass_values, feb15_slopes), start=1):
    print(f"Leg {i:02d}: Mass={m:.2f} µg/m³, Slope D={D:.3f} µm")
slope_D = feb15_slopes
rain_mm = highrain
plt.figure(figsize=(6, 4.5))
colors = plt.cm.cool(np.linspace(0, 1, len(slope_D)))
for i, (D, r, c) in enumerate(zip(slope_D, rain_mm, colors), start=1):
    plt.scatter(D, r, s=80, edgecolor='k', color=c, label=f"Leg {i}")
log_rain = np.log10(rain_mm)
slope_lr, intercept_lr, r_val2, p_val2, _ = linregress(slope_D, log_rain)

D_sorted = np.sort(slope_D)
rain_fit2 = 10 ** (intercept_lr + slope_lr * D_sorted)
plt.plot(D_sorted, rain_fit2, "r--", lw=2,
         label=f"Fit: slope={slope_lr:.2f}, R={r_val2:.2f}")
plt.yscale("log")
plt.xlabel("Dry Slope D (µm)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")
plt.title("BCB February 15\nNon-turbulent High Na\nSlope vs Accumulated Rain",
          fontsize=16, fontweight="bold")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
#%%
#multiple linear regression for mass and slope D
from sklearn.linear_model import LinearRegression
X = np.column_stack((np.log10(mass), feb15_slopes))
y = highrain  # mm
model = LinearRegression().fit(X, y)
intercept = model.intercept_
coef_logM, coef_D = model.coef_
print(f"Intercept:        {intercept:.4f} mm")
print(f"β_log10(Mass):    {coef_logM:.4f} mm per log10(µg/m³)")
print(f"β_Slope (D):      {coef_D:.4f} mm per µm slope")
R2 = model.score(X, y)
print(f"R² = {R2:.4f}")

# %%
#no turbulence low LWP
sys.modules.setdefault('numpy.core', np)
sys.modules.setdefault('numpy.core.multiarray', np.core.multiarray)
sys.modules.setdefault('numpy._core', np)
sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)

base = "/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions/Feb15difflwp"
base1 = "/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions/Feb15columnparcel"
def inspect_pickle_base(fname):
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

def inspect_pickle_base1(fname):
    with open(os.path.join(base1, fname), "rb") as f:
        data = pickle.load(f)

    print(f"\n{fname} contents:")
    if isinstance(data, (list, tuple)):
        print("Length:", len(data))
        for i, item in enumerate(data):
            print(f"  [{i}] type={type(item)}")
    else:
        print("Type:", type(data))
    return data
n0_data = inspect_pickle_base1("n0_r.pkl")
lowlwpR_data  = inspect_pickle_base("R_lowLWP.pkl")
lowlwpr_dry = n0_data[0]
lowlwpn0_r  = n0_data[1]
extra = n0_data[2] if len(n0_data) > 2 else None
print("\nExtracted:")
print("  r_dry:", np.shape(lowlwpr_dry))
print("  n0_r :", np.shape(lowlwpn0_r))
print("  extra data:", type(extra), "\n")
timelowlwp = lowlwpR_data[0]
rainlowlwp = lowlwpR_data[1]
extra_R = lowlwpR_data[2] if len(lowlwpR_data) > 2 else None
print("  time  :", np.shape(timelowlwp))
print("  rain_t:", np.shape(rainlowlwp))
print("  extra R data:", type(extra_R))
#%%
LWPlow = rainlowlwp   # because rainlowlwp is actually LWP(t)
precip_accum = np.max(LWPlow, axis=1)[:, None] - LWPlow 
precip_masked = np.where(timelowlwp >= 800, precip_accum, np.nan)
plt.figure(figsize=(8, 5))
for i in range(precip_masked.shape[0]):
    plt.plot(timelowlwp, precip_masked[i], lw=1.5, alpha=0.85)

plt.xlabel("Time (s)", fontweight="bold", fontsize=16)
plt.ylabel("Accumulated Rain (mm)", fontweight="bold", fontsize=16)
plt.title("BCB February 15\n Non-turbulent Low LWP\nAccumulated Precipitation", 
          fontweight="bold", fontsize=18)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.legend([f"Leg {i+1}" for i in range(precip_masked.shape[0])],
           bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.xticks(fontsize=14, fontweight="bold")
plt.yticks(fontsize=14, fontweight="bold")
plt.show()
# %%
for i in range(LWPlow.shape[0]):
    plt.figure(figsize=(6, 4))
    plt.plot(timelowlwp, LWPlow[i, :], lw=2) 
    plt.title(f"BCB February 15\nNon-turbulent Low LWP\nLeg {i+1}", 
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
for i in range(lowlwpn0_r.shape[0]):
    plt.figure(figsize=(6, 4))
    plt.plot(lowlwpr_dry, lowlwpn0_r[i, :], lw=2)
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
dr = np.diff(lowlwpr_dry)
for i in range(lowlwpn0_r.shape[0]):
    cumulative = np.cumsum(lowlwpn0_r[i, ::-1])[::-1]
    plt.figure(figsize=(6, 4))
    plt.plot(lowlwpr_dry, cumulative, lw=2)
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
#%%
# Calculate total and GCCN number concentrations per leg
lowtotal_m3 = np.sum(lowlwpn0_r, axis=1)
lowmask = lowlwpr_dry > 0.5e-6
lowgccn_m3 = np.sum(lowlwpn0_r[:, lowmask], axis=1)
for i, (tot, gccn) in enumerate(zip(lowtotal_m3, lowgccn_m3), start=1):
    frac = gccn / tot
    print(f"Leg {i:02d}: Total={tot:.3e} m^-3, GCCN={gccn:.3e} m^-3, GCCN/Total={frac:.2e}")

# %%
lowerhmask = lowlwpr_dry > 0.5e-6  # radius > 0.5 µm → diameter > 1 µm
lowgccn_m3 = np.sum(lowlwpn0_r[:, lowmask], axis=1)
accum_rain = np.max(LWPlow, axis=1) - LWPlow[:, -1]  # units: kg m^-2 = mm
for i, (gccn, rain) in enumerate(zip(lowgccn_m3, accum_rain), start=1):
    print(f"Leg {i:02d}: GCCN={gccn:.3e} m^-3, Rain={rain:.3f} mm")
plt.figure(figsize=(6, 4.5))
colors = plt.cm.viridis(np.linspace(0, 1, len(lowgccn_m3)))
for i, (gccn, rain, c) in enumerate(zip(lowgccn_m3, accum_rain, colors), start=1):
    plt.scatter(gccn, rain, s=80, edgecolor='k', color=c, label=f"Leg {i}")
logx = np.log10(highgccn_m3)
logy = np.log10(accum_rain)
slope, intercept, r_value, p_value, std_err = linregress(logx, logy)
x_sorted = np.sort(highgccn_m3)
y_fit_sorted = 10 ** (intercept + slope * np.log10(x_sorted))
plt.plot(x_sorted, y_fit_sorted, "r--", lw=2,
         label=f"Fit: slope={slope:.2f}, R={r_value:.2f}")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("GCCN concentration (m$^{-3}$)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")  # Correct units!
plt.title("BCB February 15 \nNon-turbulent Low LWP\nGCCN vs Accumulated Rain", fontsize=16, fontweight="bold")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.grid(alpha=0.3)
plt.yticks(fontweight="bold", fontsize=14)
plt.xticks(fontweight="bold", fontsize=14)
plt.tight_layout()
plt.show()

# %%
mass = np.array(feb15_mass_values)
lowerrain=accum_rain
for i, (m, r) in enumerate(zip(mass, lowerrain), start=1):
    print(f"Leg {i:02d}: Mass={m:.2f} µg/m³, Rain={r:.3f} mm")
plt.figure(figsize=(6, 4.5))
colors = plt.cm.plasma(np.linspace(0, 1, len(mass)))

for i, (m, r, c) in enumerate(zip(mass, lowerrain, colors), start=1):
    plt.scatter(m, r, s=80, edgecolor='k', color=c, label=f"Leg {i}")
logx = np.log10(mass)
logy = np.log10(lowerrain)
slope, intercept, r_value, p_value, std_err = linregress(logx, logy)
x_sorted = np.sort(mass)
y_fit_sorted = 10 ** (intercept + slope * np.log10(x_sorted))
plt.plot(x_sorted, y_fit_sorted, "r--", lw=2,
         label=f"Fit: slope={slope:.2f}, R={r_value:.2f}")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Dry GCCN Mass (µg/m³)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")
plt.title("BCB February 15\nNon-turbulent Low LWP\nDry Mass vs Accumulated Rain",
          fontsize=16, fontweight="bold")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.grid(alpha=0.3)
plt.yticks(fontweight="bold", fontsize=14)
plt.xticks(fontweight="bold", fontsize=14)
plt.tight_layout()
plt.show()
# %%
#plotting slope D vs rain directly
feb15_mass = [entry for entry in dry_mass_data_inf 
              if entry['Date'] == '2022-02-15']
feb15_mass_sorted = sorted(feb15_mass, key=lambda x: x['BCB_start'])
feb15_mass_values = np.array([entry['Dry Mass (µg/m³)'] for entry in feb15_mass_sorted])
feb15_slopes      = np.array([entry['Dry Slope (D)']       for entry in feb15_mass_sorted])
feb15_intercepts  = np.array([entry['Dry Intercept (N0)']  for entry in feb15_mass_sorted])
for i, (m, D) in enumerate(zip(feb15_mass_values, feb15_slopes), start=1):
    print(f"Leg {i:02d}: Mass={m:.2f} µg/m³, Slope D={D:.3f} µm")
slope_D = feb15_slopes
rain_mm = lowerrain
plt.figure(figsize=(6, 4.5))
colors = plt.cm.cool(np.linspace(0, 1, len(slope_D)))
for i, (D, r, c) in enumerate(zip(slope_D, rain_mm, colors), start=1):
    plt.scatter(D, r, s=80, edgecolor='k', color=c, label=f"Leg {i}")
log_rain = np.log10(rain_mm)
slope_lr, intercept_lr, r_val2, p_val2, _ = linregress(slope_D, log_rain)

D_sorted = np.sort(slope_D)
rain_fit2 = 10 ** (intercept_lr + slope_lr * D_sorted)
plt.plot(D_sorted, rain_fit2, "r--", lw=2,
         label=f"Fit: slope={slope_lr:.2f}, R={r_val2:.2f}")
plt.yscale("log")
plt.xlabel("Dry Slope D (µm)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")
plt.title("BCB February 15\nNon-turbulent Low LWP\nSlope vs Accumulated Rain",
          fontsize=16, fontweight="bold")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
#%%
#multiple linear regression for mass and slope D
from sklearn.linear_model import LinearRegression
X = np.column_stack((np.log10(mass), feb15_slopes))
y = lowerrain  # mm
model = LinearRegression().fit(X, y)
intercept = model.intercept_
coef_logM, coef_D = model.coef_
print(f"Intercept:        {intercept:.4f} mm")
print(f"β_log10(Mass):    {coef_logM:.4f} mm per log10(µg/m³)")
print(f"β_Slope (D):      {coef_D:.4f} mm per µm slope")
R2 = model.score(X, y)
print(f"R² = {R2:.4f}")


# %%
#comparing all non-turbulent cases
mask = r_dry > 0.5e-6
gccn_full = np.sum(n0_r[:, mask], axis=1)
rain_full = np.max(LWP, axis=1) - LWP[:, -1]

mask_low = lowr_dry > 0.5e-6
gccn_lowNa = np.sum(lown0_r[:, mask_low], axis=1)
rain_lowNa = np.max(lownaa, axis=1) - lownaa[:, -1]

mask_high = highr_dry > 0.5e-6
gccn_highNa = np.sum(highn0_r[:, mask_high], axis=1)
rain_highNa = np.max(LWPhigh, axis=1) - LWPhigh[:, -1]

mask_lowLWP = lowlwpr_dry > 0.5e-6
gccn_lowLWP = np.sum(lowlwpn0_r[:, mask_lowLWP], axis=1)
rain_lowLWP = np.max(LWPlow, axis=1) - LWPlow[:, -1]

datasets = {
    "Base LWP 385 g/m$^2$": (gccn_full, rain_full, "tab:blue"),
    "Base LWP & 10 /cc": (gccn_lowNa, rain_lowNa, "black"),
    "Base LWP & 100 /cc": (gccn_highNa, rain_highNa, "grey"),
    "LWP 194 g/m$^2$": (gccn_lowLWP, rain_lowLWP, "tab:purple")
}
plt.figure(figsize=(7, 5))

for label, (gccn, rain, color) in datasets.items():
    
    plt.scatter(gccn, rain, s=80, edgecolor='k', color=color, alpha=0.8, label=label)
    logx = np.log10(gccn)
    logy = np.log10(rain)
    slope, intercept, r, p, _ = linregress(logx, logy)
    x_sorted = np.sort(gccn)
    y_fit = 10 ** (intercept + slope * np.log10(x_sorted))
    plt.plot(x_sorted, y_fit, "--", lw=2, color=color,
             label=f"{label} fit: slope={slope:.2f}, R={r:.2f}")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("GCCN concentration (m$^{-3}$)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")
plt.title("BCB Feb 15\n Non-turbulent\n GCCN concentration vs Accumulated Rain", fontsize=16, fontweight="bold")
plt.grid(alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
plt.tight_layout()
plt.xticks(fontweight="bold", fontsize=15)
plt.yticks(fontweight="bold", fontsize=15)
plt.show()


# %%
# Full set mass
mass_full = np.array(feb15_mass_values)

datasets_mass = {
    "Base LWP 385 g/m$^2$":  (mass_full,   rain_full,   "tab:blue"),
    "Base LWP & 10 /cc":   (mass_full,   rain_lowNa,  "black"),
    "Base LWP & 100 /cc":  (mass_full,   rain_highNa, "grey"),
    "LWP 194 g/m$^2$":  (mass_full,   rain_lowLWP, "tab:purple")
}

plt.figure(figsize=(7, 5))

for label, (mass_arr, rain_arr, color) in datasets_mass.items():
    
    plt.scatter(mass_arr, rain_arr, s=90, edgecolor='k',
                color=color, alpha=0.85, label=label)

    logx = np.log10(mass_arr)
    logy = np.log10(rain_arr)
    slope, intercept, r, p, _ = linregress(logx, logy)
    x_sorted = np.sort(mass_arr)
    y_fit = 10 ** (intercept + slope * np.log10(x_sorted))
    plt.plot(x_sorted, y_fit, "--", lw=2, color=color,
             label=f"{label} fit: slope={slope:.2f}, R={r:.2f}")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Dry GCCN Mass (µg/m³)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")
plt.title("BCB Feb 15\n Non-turbulent\n Dry Mass vs Accumulated Rain", fontsize=16, fontweight="bold")
plt.grid(alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
plt.xticks(fontweight="bold", fontsize=15)
plt.yticks(fontweight="bold", fontsize=15)
plt.tight_layout()
plt.show()


# %%
