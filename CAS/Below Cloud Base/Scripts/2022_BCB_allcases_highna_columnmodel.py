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

base = "/home/disk/eos4/kathem24/activate/data/CAS/Full high Na"

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
n0_data = inspect_pickle("n0_r_hiNa (1).pkl")
R_data  = inspect_pickle("R_hiNa (1).pkl")
r_dry_highna = n0_data[0]
n0_r_highna  = n0_data[1]
extra = n0_data[2] if len(n0_data) > 2 else None
print("\nExtracted:")
print("  r_dry:", np.shape(r_dry_highna))
print("  n0_r :", np.shape(n0_r_highna))
print("  extra data:", type(extra), "\n")
time_highna = R_data[0]
rain_t_highna = R_data[1]
extra_R = R_data[2] if len(R_data) > 2 else None
print("  time  :", np.shape(time_highna))
print("  rain_t:", np.shape(rain_t_highna))
print("  extra R data:", type(extra_R))
#%%
LWP_highna = rain_t_highna   # because rain_t_highna is actually LWP(t)
precip_accum = np.max(LWP_highna, axis=1)[:, None] - LWP_highna 
precip_masked = np.where(time_highna >= 800, precip_accum, np.nan)
plt.figure(figsize=(8, 5))
for i in range(precip_masked.shape[0]):
    plt.plot(time_highna, precip_masked[i], lw=1.5, alpha=0.85)

plt.xlabel("Time (s)", fontweight="bold", fontsize=16)
plt.ylabel("Accumulated Rain (mm)", fontweight="bold", fontsize=16)
plt.title("BCB January - June 2022\n 385 g m$^{-2}$ LWP & high Na\nAccumulated Rainfall", 
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
total_m3 = np.sum(n0_r_highna, axis=1)
mask = r_dry_highna > 1e-6
gccn_m3 = np.sum(n0_r_highna[:, mask], axis=1)
for i, (tot, gccn) in enumerate(zip(total_m3, gccn_m3), start=1):
    frac = gccn / tot
    print(f"Leg {i:02d}: Total={tot:.3e} m^-3, GCCN={gccn:.3e} m^-3, GCCN/Total={frac:.2e}")

# %%
# GCCN versus accumulated rain
mask = r_dry_highna > 1e-6  # radius > 1 µm → diameter > 2 µm
gccn_m3 = np.sum(n0_r_highna[:, mask], axis=1)
accum_rain_highna = np.max(LWP_highna, axis=1) - LWP_highna[:, -1] 
accum_rain_highna_full = np.array(accum_rain_highna, dtype=float)   # <-- save the original 456
gccn_m3_highna_full = np.array(gccn_m3, dtype=float) # units: kg m^-2 = mm
for i, (gccn, rain) in enumerate(zip(gccn_m3, accum_rain_highna), start=1):
    print(f"Leg {i:02d}: GCCN={gccn:.3e} m^-3, Rain={rain:.3f} mm")
plt.figure(figsize=(6, 4.5))
colors = plt.cm.viridis(np.linspace(0, 1, len(gccn_m3)))
for i, (gccn, rain, c) in enumerate(zip(gccn_m3, accum_rain_highna, colors), start=1):
    plt.scatter(gccn, rain, s=80, edgecolor='k', color=c, label=f"Leg {i}")
logx = np.log10(gccn_m3)
logy = np.log10(accum_rain_highna)
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
plt.title("BCB January - June 2022\n 385 g m$^{-2}$ LWP & high Na\nAccumulated Rainfall", 
          fontweight="bold", fontsize=18)
# plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.grid(alpha=0.3)
plt.yticks(fontweight="bold", fontsize=16)
plt.xticks(fontweight="bold", fontsize=16)
plt.tight_layout()
plt.show()
#%%
#mass analysis
# all_mass = dry_mass_data_inf

# print("Total number of legs:", len(all_mass))  # should be 456
# all_mass_sorted = sorted(all_mass, key=lambda x: (x['Date'], x['BCB_start']))
# all_mass_values = [entry['Dry Mass (µg/m³)'] for entry in all_mass_sorted]
# for i, mass in enumerate(all_mass_values, start=1):
#     print(f"Leg {i}: {mass:.2f} µg/m³")
#%%
#mass analysis
mass_path = "/home/disk/eos4/kathem24/activate/data/CAS/filtered_dry_mass_inf.csv"
df_mass = pd.read_csv(mass_path)
print("CSV rows:", len(df_mass))
print("Rain full len:", len(accum_rain_highna_full))
print("GCCN full len:", len(gccn_m3_highna_full))
all_mass = df_mass.to_dict(orient="records")
all_mass_sorted = sorted(all_mass, key=lambda x: (x["Date"], x["BCB_start"]))
all_mass_values = [entry["Dry Mass (µg/m³)"] for entry in all_mass_sorted]
mass_thr = 100  # µg/m³
mass_full = np.array(all_mass_values, dtype=float)
rain_full = np.array(accum_rain_highna_full, dtype=float)
gccn_full = np.array(gccn_m3_highna_full, dtype=float)
assert len(mass_full) == len(rain_full) == len(gccn_full), \
    f"mass={len(mass_full)} rain_full={len(rain_full)} gccn_full={len(gccn_full)}"
keep = (
    np.isfinite(mass_full) &
    np.isfinite(rain_full) &
    (mass_full > 0) &
    (rain_full > 0) &
    (mass_full <= mass_thr)
)
# --- Diagnose problematic rain values ---
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
accum_rain_highna  = rain_full[keep]
gccn_m3_highna          = gccn_full[keep]
print("Kept legs:", keep.sum())
print("Dropped legs:", (~keep).sum())
print("Now lengths -> mass:", len(all_mass_values), "rain:", len(accum_rain_highna), "gccn:", len(gccn_m3_highna))


#%%
mass = np.array(all_mass_values)
rain_highna = accum_rain_highna 
for i, (m, r) in enumerate(zip(mass, rain_highna), start=1):
    print(f"Leg {i:02d}: Mass={m:.2f} µg/m³, Rain={r:.3f} mm")
plt.figure(figsize=(6, 4.5))
colors = plt.cm.plasma(np.linspace(0, 1, len(mass)))

for i, (m, r, c) in enumerate(zip(mass, rain_highna, colors), start=1):
    plt.scatter(m, r, s=80, edgecolor='k', color=c, label=f"Leg {i}")
logx = np.log10(mass)
logy = np.log10(rain_highna)
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
plt.title("BCB January - June 2022\n 385 g m$^{-2}$ LWP & high Na\nLess than 100 µg/m³ mass\nAccumulated Rainfall", 
          fontweight="bold", fontsize=18)
#plt.title("BCB January - June 2022\n 385 g m$^{-2}$ LWP & high Na\nAccumulated Rainfall", 
         # fontweight="bold", fontsize=18)
# plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.grid(alpha=0.3)
plt.yticks(fontweight="bold", fontsize=14)
plt.xticks(fontweight="bold", fontsize=14)
plt.tight_layout()
plt.show()
#%%
#correlation only between 0.1 to 1000 µg/m³
lower = 0.01    # µg/m³
upper = 100.0  # µg/m³
mask_range = (mass >= lower) & (mass <= upper)
mass_filt = mass[mask_range]
rain_filt = rain_highna[mask_range]
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
rain_highna = accum_rain_highna
plt.figure(figsize=(6, 4.5))
colors = plt.cm.plasma(np.linspace(0, 1, len(mass)))
for i, (m, r, c) in enumerate(zip(mass, rain_highna, colors), start=1):
    plt.scatter(m, r, s=80, edgecolor='k', color=c)
logx = np.log10(mass)
logy = np.log10(rain_highna)
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
rain_filt = rain_highna[mask_range]
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
plt.title("BCB January - June 2022\n 385 g m$^{-2}$ LWP & high Na\nAccumulated Rainfall",
          fontsize=18, fontweight="bold")
plt.grid(alpha=0.3)
plt.yticks(fontweight="bold", fontsize=14)
plt.xticks(fontweight="bold", fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()


#%%
# #plotting slope D vs rain directly
# all_entries = dry_mass_data_inf
# all_sorted = sorted(all_entries, key=lambda x: (x['Date'], x['BCB_start']))
# all_mass_values = np.array([entry['Dry Mass (µg/m³)'] for entry in all_sorted])
# all_slopes      = np.array([entry['Dry Slope (D)']       for entry in all_sorted])
# all_intercepts  = np.array([entry['Dry Intercept (N0)']  for entry in all_sorted])
# slope_D = all_slopes
# rain_mm = accum_rain_highna
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
# plt.title("BCB January - June 2022\n 385 g m$^{-2}$ LWP & high Na\nAccumulated Rainfall", 
#           fontweight="bold", fontsize=18)
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.show()
#%%
all_entries = [entry for entry, k in zip(all_mass_sorted, keep) if k]
slope_D = np.array([entry["Dry Slope (D)"] for entry in all_entries], dtype=float)
rain_mm = accum_rain_highna
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
plt.title("BCB January - June 2022\n Base 385 g m$^{-2}$ LWP\n High Na\nLess than 100 µg/m³ Mass",
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
plt.title("BCB January - June 2022\n 385 g m$^{-2}$ LWP", 
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
for i, (m, g, c) in enumerate(zip(mass, gccn_m3_highna, colors), start=1):
    plt.scatter(m, g, s=80, edgecolor='k', color=c) 
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Dry GCCN Mass (µg/m³)", fontsize=16, fontweight="bold")
plt.ylabel("GCCN Concentration (m$^{-3}$)", fontsize=16, fontweight="bold")
plt.title("BCB January - June 2022\n 385 g m$^{-2}$ LWP & high Na", 
          fontweight="bold", fontsize=18)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
#mass versus gccn correlation coefficient
log_mass = np.log10(mass)
log_gccn = np.log10(gccn_m3_highna)
slope_coeff2, intercept_coeff2, r_val4, p_val4, _ = linregress(log_mass, log_gccn)
print(f"Correlation between log10(Mass) and log10(GCCN):")
print(f"  R = {r_val4:.4f}, R² = {r_val4**2:.4f}")
#%%
# #monthly gccn trend coded with color seperation
# # all_mass_sorted is your date-ordered list of dicts
# assert len(gccn_m3) == len(all_mass_sorted), \
#     f"Lengths don't match: model={len(gccn_m3)} vs mass={len(all_mass_sorted)}"
# months = np.array([int(e["Date"][5:7]) for e in all_mass_sorted], dtype=int)
# x = np.arange(len(all_mass_sorted))
# gccn_m3 = np.asarray(gccn_m3, dtype=float)
# month_name = {
#     1: "January",
#     2: "February",
#     3: "March",
#     5: "May",
#     6: "June"
# }
# gccn_cm3 = gccn_m3 * 1e-6  # m⁻³ → cm⁻³
# plt.figure(figsize=(12, 4.8))

# for m in sorted(np.unique(months)):
#     if m not in month_name:
#         continue

#     m_mask = (months == m)

#     vals = gccn_cm3[m_mask]
#     good = np.isfinite(vals) & (vals > 0)
#     mean_val = np.mean(vals[good]) if np.any(good) else np.nan

#     plt.plot(
#         x[m_mask],
#         gccn_cm3[m_mask],
#         '-',
#         label=f"{month_name[m]} (mean: {mean_val:.2e} cm⁻³)"
#     )

# plt.yscale("log")
# plt.grid(alpha=0.3)
# plt.ylabel("GCCN Concentration (cm⁻³)", fontsize=16, fontweight="bold")
# plt.xlabel("Leg index", fontsize=16, fontweight="bold")
# plt.title("No Turbulence High Na Total GCCN Concentration\n January–June 2022 Monthly Means",
#           fontsize=18, fontweight="bold")
# plt.legend(ncol=2, fontsize=10)
# plt.yticks(fontsize=14, fontweight="bold")
# plt.xticks(fontsize=14, fontweight="bold")
# plt.tight_layout()
# plt.show()
#%%
all_entries_highna = [entry for entry, k in zip(all_mass_sorted, keep) if k]
dfp = pd.DataFrame(all_entries_highna).copy()
dfp["Date_dt"] = pd.to_datetime(dfp["Date"])
dfp["Month"] = dfp["Date_dt"].dt.month
sort_cols = ["Date_dt"]
if "BCB_start" in dfp.columns:
    sort_cols.append("BCB_start")
dfp = dfp.sort_values(sort_cols).reset_index(drop=True)
x = np.arange(len(dfp))
gccn_arr = np.asarray(gccn_m3_highna, dtype=float) * 1e-6  # m^-3 -> cm^-3
month_name = {
    1: "January",
    2: "February",
    3: "March",
    5: "May",
    6: "June"
}
date_first = dfp.groupby(dfp["Date_dt"].dt.date).head(1)
tick_pos = date_first.index.to_numpy()
tick_lab = date_first["Date_dt"].dt.strftime("%Y-%m-%d").to_numpy()
fig_w = max(22, 0.55 * len(tick_pos))
fig, ax = plt.subplots(figsize=(fig_w, 6.2))
legend_handles = []
for m in sorted(dfp["Month"].unique()):
    if m not in month_name:
        continue

    m_mask = (dfp["Month"].values == m)
    vals = gccn_arr[m_mask]
    good = np.isfinite(vals) & (vals > 0)
    mean_val = np.nanmean(vals[good]) if np.any(good) else np.nan
    median_val = np.nanmedian(vals[good]) if np.any(good) else np.nan

    line, = ax.plot(x[m_mask], vals, "-", linewidth=1.5)
    c = line.get_color()
    xm = x[m_mask]
    if len(xm) > 0:
        mean_x = xm[len(xm) // 2]
        ax.plot(mean_x, mean_val, marker="^", color=c,
                markersize=12, markeredgewidth=1.5, linestyle="None")
        ax.plot(mean_x + 5, median_val, marker="o", color=c,
                markersize=12, markeredgewidth=1.5, linestyle="None")

    legend_handles.extend([
        Line2D([0], [0], color=c, lw=2, label=month_name[m]),
        Line2D([0], [0], marker="^", color=c, lw=0, markersize=10,
               label=f"{month_name[m]} mean = {mean_val:.2e} cm⁻³"),
        Line2D([0], [0], marker="o", color=c, lw=0, markersize=10,
               label=f"{month_name[m]} median = {median_val:.2e} cm⁻³"),
    ])
for p in tick_pos:
    ax.axvline(p, color="k", alpha=0.06, linewidth=1)
ax.set_yscale("log")
ax.grid(alpha=0.3)
ax.set_ylabel("GCCN Concentration (cm⁻³)", fontsize=20, fontweight="bold")
ax.set_xlabel("Flight Date", fontsize=20, fontweight="bold")
ax.set_title("No Turbulence 385 g m$^{-2}$ LWP Total GCCN Concentration High Na\n Less than 100 µg/m³ mass\nJanuary–June 2022 Monthly Trend",
          fontsize=18, fontweight="bold")
ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_lab, rotation=60, ha="right", fontsize=7, fontweight="bold")
labels = ax.get_xticklabels()
for i, lab in enumerate(labels):
    base = lab.get_text()
    lab.set_text("\n" * (i % 4) + base)
ax.set_xticklabels([lab.get_text() for lab in labels])
ax.legend(handles=legend_handles, ncol=2, fontsize=9, loc="lower right", frameon=True)
fig.subplots_adjust(bottom=0.40)
fig.tight_layout()
plt.yticks(fontsize=14, fontweight="bold")
plt.show()
#%%
#slope D versus gccn
plt.figure(figsize=(6, 4.5))
colors = plt.cm.cividis(np.linspace(0, 1, len(slope_D)))
for i, (D, g, c) in enumerate(zip(slope_D, gccn_m3, colors), start=1):
    plt.scatter(D, g, s=80, edgecolor='k', color=c)
plt.yscale('log')
plt.xlabel("Dry Slope D (µm)", fontsize=16, fontweight="bold")
plt.ylabel("GCCN Concentration (m$^{-3}$)", fontsize=16, fontweight="bold")
plt.title("BCB January - June 2022\n 385 g m$^{-2}$ LWP & high Na", 
          fontweight="bold", fontsize=18)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
#slope D versus gccn correlation coefficient
slope_coeff3, intercept_coeff3, r_val5, p_val5, _ = linregress(slope_D, log_gccn)
print(f"Correlation between Slope D and log10(GCCN):")
print(f"  R = {r_val5:.4f}, R² = {r_val5**2:.4f}")