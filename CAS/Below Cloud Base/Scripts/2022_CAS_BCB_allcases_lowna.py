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
#no turbulence all cases
sys.modules.setdefault('numpy.core', np)
sys.modules.setdefault('numpy.core.multiarray', np.core.multiarray)
sys.modules.setdefault('numpy._core', np)
sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)

base = "/home/disk/eos4/kathem24/activate/data/CAS/Full low Na"

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
n0_data = inspect_pickle("n0_r_lowNa (1).pkl")
R_data  = inspect_pickle("R_lowNa (1).pkl")
r_dry_lowna = n0_data[0]
n0_r_lowna  = n0_data[1]
extra = n0_data[2] if len(n0_data) > 2 else None
print("\nExtracted:")
print("  r_dry:", np.shape(r_dry_lowna))
print("  n0_r :", np.shape(n0_r_lowna))
print("  extra data:", type(extra), "\n")
time_lowna = R_data[0]
rain_t_lowna = R_data[1]
extra_R = R_data[2] if len(R_data) > 2 else None
print("  time  :", np.shape(time_lowna))
print("  rain_t:", np.shape(rain_t_lowna))
print("  extra R data:", type(extra_R))
#%%
LWP_lowna = rain_t_lowna   # because rain_t_lowna is actually LWP(t)
precip_accum = np.max(LWP_lowna, axis=1)[:, None] - LWP_lowna    
precip_masked = np.where(time_lowna >= 800, precip_accum, np.nan)
plt.figure(figsize=(8, 5))
for i in range(precip_masked.shape[0]):
    plt.plot(time_lowna, precip_masked[i], lw=1.5, alpha=0.85)

plt.xlabel("Time (s)", fontweight="bold", fontsize=16)
plt.ylabel("Accumulated Rain (mm)", fontweight="bold", fontsize=16)
plt.title("BCB January - June 2022\n 385 g m$^{-2}$ LWP & low Na\nAccumulated Rainfall", 
          fontweight="bold", fontsize=18)
plt.grid(alpha=0.3)
plt.tight_layout()
# plt.legend([f"Leg {i+1}" for i in range(precip_masked.shape[0])],
#            bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.xticks(fontsize=14, fontweight="bold")
plt.yticks(fontsize=14, fontweight="bold")
plt.show()

#%%
# for i in range(LWP.shape[0]):
#     plt.figure(figsize=(6, 4))
#     plt.plot(time_highna, LWP[i, :], lw=2) 
#     plt.title(f"BCB February 15\nColumn Parcel Model\nLeg {i+1}", 
#               fontweight="bold", fontsize=18)
#     plt.xlabel("Time (s)", fontweight="bold", fontsize=16)
#     plt.ylabel("Liquid Water Path (kg m$^{-2}$)", 
#                fontweight="bold", fontsize=16) 
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     plt.yticks(fontweight="bold", fontsize=14)
#     plt.xticks(fontweight="bold", fontsize=14)
#     plt.show()
#%%
# #plotting size distributions
# for i in range(n0_r.shape[0]):
#     plt.figure(figsize=(6, 4))
#     plt.plot(r_dry, n0_r[i, :], lw=2)
#     plt.title(f"February 15, 2022\nDry size distribution\nLeg {i+1}", fontweight="bold", fontsize=18)
#     plt.xlabel("Dry radius (m)", fontweight="bold", fontsize=16)
#     plt.ylabel("Number Concentration (m⁻³)", fontweight="bold", fontsize=16)
#     plt.yscale("log")
#     plt.ylim(1, 1e8)
#     plt.xscale("log")
#     plt.yticks(fontweight="bold", fontsize=14)
#     plt.xticks(fontweight="bold", fontsize=14)
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     plt.show()

#%%
#cumulative size distributions 
# dr = np.diff(r_dry)
# for i in range(n0_r.shape[0]):
#     cumulative = np.cumsum(n0_r[i, ::-1])[::-1]
#     plt.figure(figsize=(6, 4))
#     plt.plot(r_dry, cumulative, lw=2)
#     plt.title(f"February 15, 2022\nCumulative dry size distribution\nLeg {i+1}",
#               fontweight="bold", fontsize=18)
#     plt.xlabel("Dry radius (m)", fontweight="bold", fontsize=16)
#     plt.ylabel("Cumulative Number \nConcentration (m⁻³)", fontweight="bold", fontsize=16)
#     plt.yscale("log")
#     plt.ylim(1, 1e8)
#     plt.xscale("log")
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     plt.show()
# %%
# Calculate total and GCCN number concentrations per leg
total_m3 = np.sum(n0_r_lowna, axis=1)
mask = r_dry_lowna > 0.5e-6
gccn_m3 = np.sum(n0_r_lowna[:, mask], axis=1)
for i, (tot, gccn) in enumerate(zip(total_m3, gccn_m3), start=1):
    frac = gccn / tot
    print(f"Leg {i:02d}: Total={tot:.3e} m^-3, GCCN={gccn:.3e} m^-3, GCCN/Total={frac:.2e}")

# %%
# GCCN versus accumulated rain
mask = r_dry_lowna > 0.5e-6  # radius > 0.5 µm → diameter > 1 µm
gccn_m3 = np.sum(n0_r_lowna[:, mask], axis=1)
accum_rain_lowna = np.max(LWP_lowna, axis=1) - LWP_lowna[:, -1]  # units: kg m^-2 = mm
for i, (gccn, rain) in enumerate(zip(gccn_m3, accum_rain_lowna), start=1):
    print(f"Leg {i:02d}: GCCN={gccn:.3e} m^-3, Rain={rain:.3f} mm")
plt.figure(figsize=(6, 4.5))
colors = plt.cm.viridis(np.linspace(0, 1, len(gccn_m3)))
for i, (gccn, rain, c) in enumerate(zip(gccn_m3, accum_rain_lowna, colors), start=1):
    plt.scatter(gccn, rain, s=80, edgecolor='k', color=c, label=f"Leg {i}")
logx = np.log10(gccn_m3)
logy = np.log10(accum_rain_lowna)
slope, intercept, r_value, p_value, std_err = linregress(logx, logy)
#print correlation results
R = r_value
R2 = R**2
print(f"Correlation R = {R:.4f}")
print(f"Coefficient of Determination R² = {R2:.4f}")
x_sorted = np.sort(gccn_m3)
y_fit_sorted = 10 ** (intercept + slope * np.log10(x_sorted))
plt.plot(x_sorted, y_fit_sorted, "r--", lw=2,
         label=f"Fit: slope={slope:.2f}, R={r_value:.2f}")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("GCCN concentration (m$^{-3}$)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")  # Correct units!
plt.title("BCB January - June 2022\n 385 g m$^{-2}$ LWP & low Na\nAccumulated Rainfall", 
          fontweight="bold", fontsize=18)
# plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.grid(alpha=0.3)
plt.yticks(fontweight="bold", fontsize=16)
plt.xticks(fontweight="bold", fontsize=16)
plt.tight_layout()
plt.show()
#%%
#making a PDF of gccn number concentration per leg
gccn_cm3_model = np.array(gccn_m3, dtype=float) / 1e6
gccn_cm3_model = gccn_cm3_model[np.isfinite(gccn_cm3_model) & (gccn_cm3_model > 0)]
cas_total_cm3 = np.array(
    [e['Total_Y_Concentration_cm3'] for e in total_concentration_cm3
     if np.isfinite(e['Total_Y_Concentration_cm3']) and e['Total_Y_Concentration_cm3'] > 0],
    dtype=float
)
print("Counts:")
print("  Model GCCN legs:", len(gccn_cm3_model))
print("  CAS total-Y legs:", len(cas_total_cm3))
xmin = min(gccn_cm3_model.min(), cas_total_cm3.min())
xmax = max(gccn_cm3_model.max(), cas_total_cm3.max())
bins = np.linspace(xmin, xmax, 30)
use_log_bins = True
if use_log_bins:
    bins = np.logspace(np.log10(xmin), np.log10(xmax), 30)
plt.figure(figsize=(8, 6))
plt.hist(cas_total_cm3, bins=bins, density=True, alpha=0.45,
         edgecolor="k", label='CAS Total "Y" (cm$^{-3}$)')

plt.hist(gccn_cm3_model, bins=bins, density=True, alpha=0.45,
         edgecolor="k", label='Model GCCN (cm$^{-3}$)')
plot_kde = False
if plot_kde:
    sns.kdeplot(cas_total_cm3, bw_adjust=1.0, label='CAS KDE')
    sns.kdeplot(gccn_cm3_model, bw_adjust=1.0, label='Model KDE')
if use_log_bins:
    plt.xscale("log")
plt.xlabel("Number concentration (cm$^{-3}$)", fontsize=14, fontweight="bold")
plt.ylabel("Probability density", fontsize=14, fontweight="bold")
plt.title("PDF comparison: CAS vs Model (same units)", fontsize=14, fontweight="bold")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


#%%
#mass analysis
all_mass = dry_mass_data_inf

print("Total number of legs:", len(all_mass))  # should be 456
all_mass_sorted = sorted(all_mass, key=lambda x: (x['Date'], x['BCB_start']))
all_mass_values = [entry['Dry Mass (µg/m³)'] for entry in all_mass_sorted]
for i, mass in enumerate(all_mass_values, start=1):
    print(f"Leg {i}: {mass:.2f} µg/m³")

#%%
mass = np.array(all_mass_values)
rain_lowna = accum_rain_lowna 
for i, (m, r) in enumerate(zip(mass, rain_lowna), start=1):
    print(f"Leg {i:02d}: Mass={m:.2f} µg/m³, Rain={r:.3f} mm")
plt.figure(figsize=(6, 4.5))
colors = plt.cm.plasma(np.linspace(0, 1, len(mass)))

for i, (m, r, c) in enumerate(zip(mass, rain_lowna, colors), start=1):
    plt.scatter(m, r, s=80, edgecolor='k', color=c, label=f"Leg {i}")
logx = np.log10(mass)
logy = np.log10(rain_lowna)
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
plt.title("BCB January - June 2022\n 385 g m$^{-2}$ LWP & low Na\nAccumulated Rainfall", 
          fontweight="bold", fontsize=18)
# plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.grid(alpha=0.3)
plt.yticks(fontweight="bold", fontsize=14)
plt.xticks(fontweight="bold", fontsize=14)
plt.tight_layout()
plt.show()
#%%
#looking at mass as a function of month 

all_mass = dry_mass_data_inf
rain_all = np.asarray(accum_rain_lowna, dtype=float)  # same order as all_mass
mass_all = np.array([e['Dry Mass (µg/m³)'] for e in all_mass], dtype=float)
months = np.array([int(e['Date'][5:7]) for e in all_mass], dtype=int)
for m in sorted(np.unique(months)):
    mask = (months == m)
    mass_m = mass_all[mask]
    rain_m = rain_all[mask]
    good = np.isfinite(mass_m) & np.isfinite(rain_m) & (mass_m > 0) & (rain_m > 0)
    mass_m, rain_m = mass_m[good], rain_m[good]
    if len(mass_m) < 3:
        print(f"Month {m}: only {len(mass_m)} usable legs.")
        continue
    slope, intercept, r_value, p_value, std_err = linregress(np.log10(mass_m), np.log10(rain_m))
    R2 = r_value**2
    plt.figure(figsize=(6.5, 5))
    colors = plt.cm.plasma(np.linspace(0, 1, len(mass_m)))
    for i, (x, y, c) in enumerate(zip(mass_m, rain_m, colors), start=1):
        plt.scatter(x, y, s=70, edgecolor='k', linewidth=0.6, color=c)

    x_sorted = np.sort(mass_m)
    y_fit = 10 ** (intercept + slope * np.log10(x_sorted))
    plt.plot(x_sorted, y_fit, "r--", lw=2, label=f"slope={slope:.2f}, R²={R2:.2f}")

    plt.xscale("log"); plt.yscale("log")
    plt.grid(alpha=0.3)
    plt.xlabel("Dry GCCN Mass (µg/m³)", fontsize=14, fontweight="bold")
    plt.ylabel("Accumulated Rain (mm)", fontsize=14, fontweight="bold")
    plt.title(f"Month {m}: Mass vs Accumulated Rain (low Na)", fontsize=15, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.show()
#%%
all_mass = dry_mass_data_inf
mass_all = np.array([e['Dry Mass (µg/m³)'] for e in all_mass], dtype=float)
date_all = np.array([e['Date'] for e in all_mass])
unique_dates = sorted(set(date_all))  # string-sort works for YYYY-MM-DD
daily_mean = []
daily_median = []
daily_se = []
daily_n = []
daily_month = []
for d in unique_dates:
    mask = (date_all == d)
    vals = mass_all[mask]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    n = len(vals)

    daily_n.append(n)
    daily_month.append(int(d[5:7]))

    if n == 0:
        daily_mean.append(np.nan)
        daily_median.append(np.nan)
        daily_se.append(np.nan)
    else:
        daily_mean.append(np.mean(vals))
        daily_median.append(np.median(vals))
        daily_se.append(np.std(vals, ddof=1)/np.sqrt(n) if n > 1 else np.nan)

daily_mean = np.array(daily_mean, float)
daily_median = np.array(daily_median, float)
daily_se = np.array(daily_se, float)
daily_month = np.array(daily_month, int)
x = np.arange(len(unique_dates))
plt.figure(figsize=(11, 4.8))
plt.errorbar(x, daily_mean, yerr=daily_se, fmt='o', capsize=3, label="Daily mean ± SE")
plt.plot(x, daily_median, 's--', label="Daily median")
plt.grid(alpha=0.3)
plt.ylabel("Dry GCCN Mass (µg/m³)", fontsize=14, fontweight="bold")
plt.title("Dry GCCN Mass by Flight Day (multiple legs per day)\nMean ± SE across legs; median shown",
          fontsize=15, fontweight="bold")
step = max(1, len(unique_dates)//12)  # ~12 labels
tick_idx = np.arange(0, len(unique_dates), step)
plt.xticks(tick_idx, [unique_dates[i] for i in tick_idx], rotation=45, ha='right', fontsize=10)

plt.legend()
plt.tight_layout()
plt.show()

#%%
months = sorted(set(daily_month.tolist()))
month_mean_of_days = []
month_se_of_days = []
month_n_days = []

for m in months:
    vals = daily_mean[daily_month == m]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    nd = len(vals)
    month_n_days.append(nd)
    if nd == 0:
        month_mean_of_days.append(np.nan)
        month_se_of_days.append(np.nan)
    else:
        month_mean_of_days.append(np.mean(vals))
        month_se_of_days.append(np.std(vals, ddof=1)/np.sqrt(nd) if nd > 1 else np.nan)

plt.figure(figsize=(7.5, 4.8))
x = np.arange(len(months))
plt.errorbar(x, month_mean_of_days, yerr=month_se_of_days, fmt='o-', capsize=4)

plt.xticks(x, [str(m) for m in months], fontweight="bold")
plt.xlabel("Month", fontsize=14, fontweight="bold")
plt.ylabel("Mean daily Dry Mass (µg/m³)", fontsize=14, fontweight="bold")
plt.title("Monthly Dry Mass (from daily means)", fontsize=15, fontweight="bold")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

for m, nd in zip(months, month_n_days):
    print(f"Month {m}: {nd} flight days")

#%%
#correlation only between 0.1 to 1000 µg/m³
lower = 0.01    # µg/m³
upper = 100.0  # µg/m³
mask_range = (mass >= lower) & (mass <= upper)
mass_filt = mass[mask_range]
rain_filt = rain_lowna[mask_range]
logx_filt = np.log10(mass_filt)
logy_filt = np.log10(rain_filt)
slope_filt, intercept_filt, r_value_filt, p_value_filt, std_err_filt = linregress(logx_filt, logy_filt)

R = r_value_filt
R2 = R**2
print(f"Filtered correlation R = {R:.4f}")
print(f"Filtered R² = {R2:.4f}")
print(f"Number of points used = {len(mass_filt)}")
x_sorted_filt = np.sort(mass_filt)
y_fit_sorted_filt = 10 ** (intercept_filt + slope_filt * np.log10(x_sorted_filt))

plt.plot(x_sorted_filt, y_fit_sorted_filt, "b--", lw=3,
         label=f"Filtered Fit: R={R:.2f}")
plt.legend()
#%%
mass = np.array(all_mass_values)
rain_lowna = accum_rain_lowna
plt.figure(figsize=(6, 4.5))
colors = plt.cm.plasma(np.linspace(0, 1, len(mass)))
for i, (m, r, c) in enumerate(zip(mass, rain_lowna, colors), start=1):
    plt.scatter(m, r, s=80, edgecolor='k', color=c)
logx = np.log10(mass)
logy = np.log10(rain_lowna)
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
rain_filt = rain_lowna[mask_range]
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
plt.title("BCB January - June 2022\n 385 g m$^{-2}$ LWP & low Na\nAccumulated Rainfall",
          fontsize=18, fontweight="bold")
plt.grid(alpha=0.3)
plt.yticks(fontweight="bold", fontsize=14)
plt.xticks(fontweight="bold", fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()


#%%
#plotting slope D vs rain directly
all_entries = dry_mass_data_inf
all_sorted = sorted(all_entries, key=lambda x: (x['Date'], x['BCB_start']))
all_mass_values = np.array([entry['Dry Mass (µg/m³)'] for entry in all_sorted])
all_slopes      = np.array([entry['Dry Slope (D)']       for entry in all_sorted])
all_intercepts  = np.array([entry['Dry Intercept (N0)']  for entry in all_sorted])
slope_D = all_slopes
rain_mm = accum_rain_lowna
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
plt.title("BCB January - June 2022\n 385 g m$^{-2}$ LWP & low Na\nAccumulated Rainfall", 
          fontweight="bold", fontsize=18)
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
plt.title("BCB January - June 2022\n 385 g m$^{-2}$ LWP & low Na", 
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
for i, (m, g, c) in enumerate(zip(mass, gccn_m3, colors), start=1):
    plt.scatter(m, g, s=80, edgecolor='k', color=c) 
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Dry GCCN Mass (µg/m³)", fontsize=16, fontweight="bold")
plt.ylabel("GCCN Concentration (m$^{-3}$)", fontsize=16, fontweight="bold")
plt.title("BCB January - June 2022\n 385 g m$^{-2}$ LWP & low Na", 
          fontweight="bold", fontsize=18)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
#mass versus gccn correlation coefficient
log_mass = np.log10(mass)
log_gccn = np.log10(gccn_m3)
slope_coeff2, intercept_coeff2, r_val4, p_val4, _ = linregress(log_mass, log_gccn)
print(f"Correlation between log10(Mass) and log10(GCCN):")
print(f"  R = {r_val4:.4f}, R² = {r_val4**2:.4f}")
#%%
#slope D versus gccn
plt.figure(figsize=(6, 4.5))
colors = plt.cm.cividis(np.linspace(0, 1, len(slope_D)))
for i, (D, g, c) in enumerate(zip(slope_D, gccn_m3, colors), start=1):
    plt.scatter(D, g, s=80, edgecolor='k', color=c)
plt.yscale('log')
plt.xlabel("Dry Slope D (µm)", fontsize=16, fontweight="bold")
plt.ylabel("GCCN Concentration (m$^{-3}$)", fontsize=16, fontweight="bold")
plt.title("BCB January - June 2022\n 385 g m$^{-2}$ LWP & low Na", 
          fontweight="bold", fontsize=18)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
#slope D versus gccn correlation coefficient
slope_coeff3, intercept_coeff3, r_val5, p_val5, _ = linregress(slope_D, log_gccn)
print(f"Correlation between Slope D and log10(GCCN):")
print(f"  R = {r_val5:.4f}, R² = {r_val5**2:.4f}")

#%%
#multiple linear regression for mass and slope D
from sklearn.linear_model import LinearRegression
X = np.column_stack((np.log10(mass), all_slopes))
y = rain  
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
#actual versus predicted rain using mass 
log_mass = np.log10(mass).reshape(-1, 1)
slope_D  = slope_D.reshape(-1, 1)
log_rain = np.log10(accum_rain).reshape(-1, 1)
X = np.hstack([log_mass, slope_D])
y = log_rain
model = LinearRegression()
model.fit(X, y)
intercept = model.intercept_[0]
beta_mass = model.coef_[0][0]
beta_slope = model.coef_[0][1]
y_pred = model.predict(X)
R2 = model.score(X, y)

print("Multiple Linear Regression Results:")
print(f"Intercept:        {10**intercept:.4f} mm (in rain units)")
print(f"β_log10(Mass):    {beta_mass:.4f}")
print(f"β_Slope (D):      {beta_slope:.4f}")
print(f"R² = {R2:.4f}")
plt.figure(figsize=(6, 5))
plt.scatter(10**y, 10**y_pred, s=60, edgecolor='k', alpha=0.8)
min_val = min(10**y.min(), 10**y_pred.min())
max_val = max(10**y.max(), 10**y_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

plt.xscale('log')
plt.yscale('log')
plt.xlabel("Actual Rain (mm)", fontsize=16, fontweight="bold")
plt.ylabel("Predicted Rain (mm)", fontsize=16, fontweight="bold")
plt.title(f"BCB January - June 2022\n 385 g m$^{-2}$ LWP\nR² = {R2:.3f}", 
          fontweight="bold", fontsize=18)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.yticks(fontweight="bold", fontsize=16)
plt.xticks(fontweight="bold", fontsize=16)  
plt.show()
#%%
# Remove slope effect from rain
residual_rain = log_rain - beta_slope * slope_D

plt.figure(figsize=(6, 5))
plt.scatter(log_mass, 10**residual_rain, s=60, edgecolor='k', alpha=0.8)
slope_m, intercept_m, r_m, _, _ = linregress(log_mass[:,0], residual_rain[:,0])
xs = np.linspace(log_mass.min(), log_mass.max(), 200)
ys = 10**(intercept_m + slope_m * xs)
plt.plot(xs, ys, "r--", lw=2)
plt.yscale('log')
plt.xlabel("log10(Mass)", fontsize=16, fontweight="bold")
plt.ylabel("Rain Residual (mm)", fontsize=16, fontweight="bold")
plt.title("Rain vs Mass (adjusted for slope)", fontsize=16, fontweight="bold")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
#%%
# Remove mass effect from rain
residual_rain2 = log_rain - beta_mass * log_mass
plt.figure(figsize=(6, 5))
plt.scatter(slope_D, 10**residual_rain2, s=60, edgecolor='k', alpha=0.8)
slope_s, intercept_s, r_s, _, _ = linregress(slope_D[:,0], residual_rain2[:,0])
xs = np.linspace(slope_D.min(), slope_D.max(), 200)
ys = 10**(intercept_s + slope_s * xs)
plt.plot(xs, ys, "r--", lw=2)
plt.yscale('log')
plt.xlabel("Dry Slope D (µm)", fontsize=16, fontweight="bold")
plt.ylabel("Rain Residual (mm)", fontsize=16, fontweight="bold")
plt.title("Rain vs Slope (adjusted for mass)", fontsize=16, fontweight="bold")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
#%%
#using mass, slope, and gccn for prediction
log_mass = np.log10(mass).reshape(-1, 1)
log_gccn = np.log10(gccn_m3).reshape(-1, 1)
slope = slope_D.reshape(-1, 1)
log_rain = np.log10(accum_rain).reshape(-1, 1)
X = np.hstack([log_mass, slope, log_gccn])
model = LinearRegression().fit(X, log_rain)
a = model.intercept_[0]
b_mass = model.coef_[0][0]
b_slope = model.coef_[0][1]
b_gccn = model.coef_[0][2]
R2 = model.score(X, log_rain)

print("Log Rain Model:")
print(f"Rain = 10^({a:.4f} + {b_mass:.4f} log10(Mass) + {b_slope:.4f} * Slope + {b_gccn:.4f} log10(GCCN))")
print("R² =", R2)

mass_med  = np.median(mass)
slope_med = np.median(slope_D)
gccn_med  = np.median(gccn_m3)
mass_range = np.logspace(np.log10(min(mass)), np.log10(max(mass)), 200)
logR_mass_curve = (
    a + 
    b_mass * np.log10(mass_range) +
    b_slope * slope_med +
    b_gccn * np.log10(gccn_med)
)

pred_mass_curve = 10 ** logR_mass_curve
plt.figure(figsize=(6, 5))
plt.plot(mass_range, pred_mass_curve, 'k-', lw=2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Mass (µg m⁻³)", fontsize=16, fontweight='bold')
plt.ylabel("Predicted Rain (mm)", fontsize=16, fontweight='bold')
plt.title("Predicted Rain vs Mass\nSlope + GCCN fixed", fontsize=16, fontweight='bold')
plt.grid(alpha=0.3)
plt.yticks(fontweight="bold", fontsize=16)
plt.xticks(fontweight="bold", fontsize=16)
plt.tight_layout()
plt.show()
slope_range = np.linspace(min(slope_D), max(slope_D), 200)
logR_slope_curve = (
    a + 
    b_mass * np.log10(mass_med) +
    b_slope * slope_range +
    b_gccn * np.log10(gccn_med)
)

pred_slope_curve = 10 ** logR_slope_curve
plt.figure(figsize=(6, 5))
plt.plot(slope_range, pred_slope_curve, 'k-', lw=2)
plt.yscale('log')
plt.xlabel("Dry Slope D (µm)", fontsize=16, fontweight='bold')
plt.ylabel("Predicted Rain (mm)", fontsize=16, fontweight='bold')
plt.title("Predicted Rain vs Slope\nMass + GCCN fixed", fontsize=16, fontweight='bold')
plt.grid(alpha=0.3)
plt.yticks(fontweight="bold", fontsize=16)
plt.xticks(fontweight="bold", fontsize=16)
plt.tight_layout()
plt.show()

gccn_range = np.logspace(np.log10(min(gccn_m3)), np.log10(max(gccn_m3)), 200)

logR_gccn_curve = (
    a +
    b_mass * np.log10(mass_med) +
    b_slope * slope_med +
    b_gccn * np.log10(gccn_range)
)

pred_gccn_curve = 10 ** logR_gccn_curve
plt.figure(figsize=(6, 5))
plt.plot(gccn_range, pred_gccn_curve, 'k-', lw=2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("GCCN concentration (m⁻³)", fontsize=16, fontweight='bold')
plt.ylabel("Predicted Rain (mm)", fontsize=16, fontweight='bold')
plt.title("Predicted Rain vs GCCN\nMass + Slope fixed", fontsize=16, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.yticks(fontweight="bold", fontsize=16)
plt.xticks(fontweight="bold", fontsize=16)
plt.show()
lm = np.log10(mass).flatten()
ls = slope_D.flatten()
lg = np.log10(gccn_m3).flatten()
print("corr(Mass, GCCN) =", np.corrcoef(lm, lg)[0,1])
print("corr(Mass, Slope) =", np.corrcoef(lm, ls)[0,1])
print("corr(GCCN, Slope) =", np.corrcoef(lg, ls)[0,1])
# %%
