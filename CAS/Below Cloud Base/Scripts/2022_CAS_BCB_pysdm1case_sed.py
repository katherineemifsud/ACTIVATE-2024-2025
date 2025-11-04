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
# convert n0_r from #/m⁴ to #/m³/µm -> divide by 1e6
n0_r_corrected = n0_r / 1e6
mask = r_dry > 0.5e-6
gccn_total_m3 = np.trapz(n0_r_corrected[:, mask], r_dry[mask], axis=1)
gccn_total_cm3 = gccn_total_m3 / 1e6

for i, val in enumerate(gccn_total_cm3, start=1):
    print(f"Leg {i:02d}: GCCN (>1 µm D) = {val:.3e} cm⁻³")
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
# %%
#converting from radius in m to diameter in um
d_centers_um = r_centers_m * 2 * 1e6
for i, d in enumerate(d_centers_um, 1):
    print(f"Bin {i:3d}: {d:.3f} µm")
d_centers_um = np.array(d_centers_um)
#%%
#converting from #/m4 to #/cm3/um
r_edges_m  = np.logspace(-7, -5, 101)
r_centers_m = np.sqrt(r_edges_m[:-1] * r_edges_m[1:])
d_centers_um = r_centers_m * 2 * 1e6
conversion_factor = 1e-12 
N_per_cm3_per_um = N_per_m3_per_m * conversion_factor
plt.figure(figsize=(7,4))
plt.semilogx(d_centers_um, N_per_cm3_per_um, lw=2)
plt.axvline(x=2.0, color='red', linestyle='--', linewidth=2,
            label='2 µm')
plt.xlabel('Dry Diameter (µm)', fontsize=17, fontweight='bold')
plt.ylabel('Number Concentration\n(cm$^{-3}$ µm$^{-1}$)',
           fontsize=17, fontweight='bold')
plt.yscale('log')
plt.title('CAS BCB February 15, 2022\n Leg 1\n Dry Size Distribution', fontsize=17, fontweight='bold')
plt.grid(True, which='both', ls='--', alpha=0.4)
plt.tight_layout()
plt.show()
#%%
#checking bin centers 

print("First 10 bin centers (radius in meters):")
for i, r in enumerate(r_centers_m[:10], 1):
    print(f"  Bin {i:2d}: {r:.3e} m")
print("\nFirst 10 bin centers (diameter in micrometers):")
for i, d in enumerate(d_centers_um[:100], 1):
    print(f"  Bin {i:2d}: {d:.3f} µm")

print(f"\nTotal bins: {len(d_centers_um)}")

#%%
# Bin edges in diameter (µm) from your radius edges
d_edges_um = r_edges_m * 2 * 1e6
d_widths_um = np.diff(d_edges_um)
total_conc = np.nansum(N_per_cm3_per_um * d_widths_um)
mask_2um = d_centers_um >= 2.0
conc_above_2um = np.nansum(N_per_cm3_per_um[mask_2um] *
                           d_widths_um[mask_2um])

print(f"Total concentration = {total_conc:.2f} cm^-3")
print(f"Concentration ≥2 µm = {conc_above_2um:.2f} cm^-3")
plt.figure(figsize=(7,4))
plt.semilogx(d_centers_um, N_per_cm3_per_um, lw=2)
plt.axvline(x=2.0, color='red', linestyle='--', linewidth=2,
            label='2 µm')
plt.xlabel('Dry Diameter (µm)', fontsize=17, fontweight='bold')
plt.ylabel('Number Concentration\n(cm$^{-3}$ µm$^{-1}$)',
           fontsize=17, fontweight='bold')
plt.yscale('log')
plt.title('CAS BCB January 11, 2022\nLeg 1\nDry Size Distribution',
          fontsize=17, fontweight='bold')
plt.grid(True, which='both', ls='--', alpha=0.4)
txt = (f"Total Concentration: {total_conc:.2f} cm$^{{-3}}$\n"
       f"Concentration >2 µm diameter: {conc_above_2um:.2f} cm$^{{-3}}$")
plt.text(0.58, 0.95, txt, transform=plt.gca().transAxes,
         fontsize=7, fontweight='bold', va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
plt.tight_layout()
plt.show()
#%%
r_edges_m = np.logspace(-7, -5, 101)
r_centers_m = np.sqrt(r_edges_m[:-1] * r_edges_m[1:])
d_centers_um_model = r_centers_m * 2 * 1e6  
dry_spec = np.asarray(trial_data["dry spectrum"]).squeeze() 
mean_dry = np.nanmean(dry_spec, axis=0)
N_model = mean_dry * 1e-12 
print("=== Jason's Model Dry Spectrum ===")
print(f"Min diameter: {d_centers_um_model.min():.3f} µm")
print(f"Max diameter: {d_centers_um_model.max():.3f} µm")
print(f"Nonzero bins: {(N_model > 0).sum()} / {len(N_model)}")
print("\nDiameter (µm)    N_model (#/cm³/µm)")
for d, n in zip(d_centers_um_model[::5], N_model[::5]):  
    print(f"{d:10.3f}        {n:10.3e}")
#%%
precip_rate = np.asarray(trial_data["surface precipitation"]).squeeze()  # [kg/m²/s ≡ mm/s]
time_s = np.asarray(trial_data["t"]).squeeze()                           # [s]
dt = np.median(np.diff(time_s))  # should be 10 s
print(f"Time step: {dt} s, Length: {len(time_s)} points, Duration: {time_s[-1]/60:.1f} min")
precip_rate_mmhr = precip_rate * 3600.0  # 1 s → 3600 s/hr
accum_precip_mm = np.cumsum(precip_rate * dt)  # result in mm
fig, ax1 = plt.subplots(figsize=(8,4))
color1 = "tab:blue"
ax1.plot(time_s/60, precip_rate_mmhr, color=color1, lw=2)
ax1.set_xlabel("Time (minutes)", fontsize=14, fontweight="bold")
ax1.set_ylabel("Precipitation Rate (mm hr$^{-1}$)", color=color1,
               fontsize=14, fontweight="bold")
ax1.tick_params(axis="y", labelcolor=color1)
ax1.grid(True, which="both", ls="--", alpha=0.4)
ax2 = ax1.twinx()
color2 = "tab:red"
ax2.plot(time_s/60, accum_precip_mm, color=color2, lw=2)
ax2.set_ylabel("Accumulated Precipitation (mm)", color=color2,
               fontsize=14, fontweight="bold")
ax2.tick_params(axis="y", labelcolor=color2)
plt.title("Model Surface Precipitation Time Series", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.show()
print(f"Total accumulated precipitation after 1 hour: {accum_precip_mm[-1]:.3f} mm")
#%%
surface_precip = np.asarray(trial_data["surface precipitation"]).squeeze()
print("Surface precip shape:", surface_precip.shape)
print("Min:", np.nanmin(surface_precip), "Max:", np.nanmax(surface_precip))
print("Mean:", np.nanmean(surface_precip))
print("Number of nonzero points:", np.count_nonzero(surface_precip))
nonzero_idx = np.where(surface_precip > 0)[0]
if len(nonzero_idx) == 0:
    print("⚠️ No nonzero surface precipitation values found.")
else:
    print("\nFirst few nonzero entries (index, value):")
    for i in nonzero_idx[:10]:
        print(f"  t={i:4d}  value={surface_precip[i]:.3e}")
#%%

def _shim_numpy_for_pickles():
    sys.modules.setdefault('numpy.core', np)
    sys.modules.setdefault('numpy.core.multiarray', np.core.multiarray)
    sys.modules.setdefault('numpy._core', np)
    sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)

_shim_numpy_for_pickles()

base_path = "/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions/Feb15Ensemble"
all_accum = []
time_series = []
for file_name in sorted(os.listdir(base_path)):
    if not file_name.endswith(".pickle"):
        continue

    file_path = os.path.join(base_path, file_name)
    leg_label = file_name.replace(".pickle", "")  # e.g. "0_1"
    
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    precip_rate = np.asarray(data["surface precipitation"]).squeeze()  # [kg/m²/s] = [mm/s]
    time_s = np.asarray(data["t"]).squeeze()

    if precip_rate.size == 0 or time_s.size < 2:
        print(f"⚠️ Empty or invalid precip data in {file_name}, skipping.")
        continue

    dt = np.median(np.diff(time_s))
    accum_precip_mm = np.cumsum(precip_rate * dt)
    total_precip_mm = accum_precip_mm[-1]

    all_accum.append((leg_label, total_precip_mm))
    time_series.append((time_s, precip_rate, leg_label, total_precip_mm))

    print(f"{leg_label}: Total accumulation = {total_precip_mm:.3e} mm")
plt.figure(figsize=(9, 5))
for time_s, precip_rate, leg_label, total_mm in time_series:
    plt.plot(time_s/60, precip_rate*3600, lw=1.8,
             label=f"{leg_label} ({total_mm:.1e} mm total)")

plt.xlabel("Time (minutes)", fontsize=15, fontweight="bold")
plt.ylabel("Precipitation Rate (mm hr$^{-1}$)", fontsize=15, fontweight="bold")
plt.title("Surface Precipitation \n February 15 (High Precip)", fontsize=16, fontweight="bold")
plt.yscale("log")
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend(fontsize=9, ncol=2, loc="upper right")
plt.tight_layout()
plt.show()
print("\n=== Summary of total accumulated precipitation ===")
for leg_label, total in all_accum:
    print(f"{leg_label}: {total:.3e} mm")
#%%
def _shim_numpy_for_pickles():
    sys.modules.setdefault('numpy.core', np)
    sys.modules.setdefault('numpy.core.multiarray', np.core.multiarray)
    sys.modules.setdefault('numpy._core', np)
    sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
_shim_numpy_for_pickles()
base_path = "/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions/Feb15Ensemble"
title_prefix = "February 15, 2022"  
pat = re.compile(r'^(\d+)_([0-9]+)\.pickle$')
groups = defaultdict(list) 
for fn in os.listdir(base_path):
    m = pat.match(fn)
    if not m: 
        continue
    main_idx = int(m.group(1))
    sub_idx  = int(m.group(2))
    groups[main_idx].append((sub_idx, os.path.join(base_path, fn)))
for k in groups:
    groups[k].sort(key=lambda x: x[0])

def load_precip(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    pr = np.asarray(data["surface precipitation"]).squeeze()
    t  = np.asarray(data["t"]).squeeze()
    if pr.size == 0 or t.size < 2:
        return None, None, None
    order = np.argsort(t)
    t = t[order]
    pr = pr[order]
    dt = float(np.median(np.diff(t)))
    accum_mm = np.cumsum(pr * dt)         # mm
    total_mm = float(accum_mm[-1])
    return t, pr, total_mm
summary = [] 
for main_idx in sorted(groups.keys()):
    entries = groups[main_idx]
    if not entries:
        continue
    plt.figure(figsize=(9, 5))
    plotted_any = False
    for sub_idx, path in entries:
        t, pr, total_mm = load_precip(path)
        if t is None:
            print(f"⚠️ Skipping {os.path.basename(path)} (empty or invalid).")
            continue
        plt.plot(t/60.0, pr*3600.0, lw=1.8, label=f"{main_idx}_{sub_idx}  ({total_mm:.1e} mm)")
        summary.append((main_idx, sub_idx, total_mm))
        plotted_any = True
    if not plotted_any:
        plt.close()
        continue
    plt.xlabel("Time (minutes)", fontsize=15, fontweight="bold")
    plt.ylabel("Precipitation Rate (mm hr$^{-1}$)", fontsize=15, fontweight="bold")
    plt.title(f"Surface Precipitation — {title_prefix}\n Leg {main_idx}", fontsize=16, fontweight="bold")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(fontsize=9, ncol=2, loc="upper right")
    plt.tight_layout()
    plt.show()
print("\n=== Summary of total accumulated precipitation (mm) ===")
for main_idx in sorted(groups.keys()):
    rows = [(m,s,tot) for (m,s,tot) in summary if m == main_idx]
    if not rows:
        continue
    print(f"\n Leg {main_idx}:")
    for _, sub_idx, total_mm in sorted(rows, key=lambda x: x[1]):
        print(f"  {main_idx}_{sub_idx}: {total_mm:.3e} mm")
    total_main = np.nansum([tot for _,_,tot in rows])
    print(f"  ---- Leg {main_idx} sum: {total_main:.3e} mm")
#%%
n0_10 = 0.812
D_10  = 0.828
def exponential(x, n0, D):
    return n0 * np.exp(-x / D)
x_fit = np.linspace(2, 10, 200)
y_fit = exponential(x_fit, n0_10, D_10)
plt.figure(figsize=(6, 5))
plt.semilogy(x_fit, y_fit, color="black", lw=1.2, alpha=0.8)
plt.xlabel("Dry Bin Centers Diameter (µm)", fontsize=13, fontweight="bold")
plt.ylabel("CAS Number Concentration (cm$^{-3}$ µm$^{-1}$)", fontsize=13, fontweight="bold")
plt.title("CAS Below Cloud Base – Feb 15 2022 (Leg 1)\nFitted Dry Size Distribution (≤10 µm)",
          fontsize=13, fontweight="bold")
plt.xlim(0, 10)
plt.ylim(1e-7, 1e1)
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.tight_layout()
plt.show()
#%%
n0_10 = 0.812
D_10  = 0.828
def exponential(D, n0, D_e):
    return n0 * np.exp(-D / D_e)
x_fit = np.linspace(2, 10, 200)
y_fit = exponential(x_fit, n0_10, D_10)
r_edges_m = np.logspace(-7, -5, 101)
r_centers_m = np.sqrt(r_edges_m[:-1] * r_edges_m[1:])
d_centers_um_model = r_centers_m * 2 * 1e6
dry_spec = np.asarray(trial_data["dry spectrum"]).squeeze()
mean_dry = np.nanmean(dry_spec, axis=0)
N_model = mean_dry * 1e-12
cas_bins = np.array([
    2.25, 2.75, 3.25, 3.75, 4.5, 5.75, 6.85, 7.55, 9.05,
    11.4, 13.8, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5
])
cas_bins_10 = cas_bins[cas_bins <= 10]
model_interp = np.interp(cas_bins_10, d_centers_um_model, N_model, left=np.nan, right=np.nan)
plt.figure(figsize=(6,5))
plt.semilogy(x_fit, y_fit, color="black", lw=1.2, alpha=0.8, label="Observational Exponential Fit (≤10 µm)")
plt.semilogy(cas_bins_10, model_interp, "o-", color="tab:blue", lw=1.5, label="Model Dry Distribution (interp. to CAS bins)")
plt.xlabel("Dry Bin Centers Diameter (µm)", fontsize=13, fontweight="bold")
plt.ylabel("CAS Number Concentration (cm$^{-3}$ µm$^{-1}$)", fontsize=13, fontweight="bold")
plt.title("CAS Below Cloud Base – Feb 15 2022 (Leg 1)\nFitted vs Model Dry Size Distribution (≤10 µm)",
          fontsize=13, fontweight="bold")
plt.xlim(0, 10)
plt.ylim(1e-7, 1e1)
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend(fontsize=9)
plt.tight_layout()
plt.show()
#%%
fits_feb15 = [
    {"n0": 0.812, "D": 0.828},
    {"n0": 1.073, "D": 0.757},
    {"n0": 5.232, "D": 0.574},
    {"n0": 1.819, "D": 0.798},
    {"n0": 2.263, "D": 0.744},
    {"n0": 1.146, "D": 0.944},
    {"n0": 0.672, "D": 0.944},
    {"n0": 1.919, "D": 0.546},
    {"n0": 1.704, "D": 0.790},
    {"n0": 2.018, "D": 0.934},
    {"n0": 2.269, "D": 0.846},
    {"n0": 1.554, "D": 0.958},
    {"n0": 2.728, "D": 0.570},
    {"n0": 1.058, "D": 0.709},
]

def exponential(D, n0, D_e):
    return n0 * np.exp(-D / D_e)
cas_bins = np.array([
    2.25, 2.75, 3.25, 3.75, 4.5, 5.75, 6.85, 7.55, 9.05,
    11.4, 13.8, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5
])
cas_bins_10 = cas_bins[cas_bins <= 10] 
for leg_idx, fit in enumerate(fits_feb15):
    model_path = f"/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions/Feb15/{leg_idx} (1).pickle"
    with open(model_path, "rb") as f:
        trial_data = pickle.load(f)
    r_edges_m = np.logspace(-7, -5, 101)
    r_centers_m = np.sqrt(r_edges_m[:-1] * r_edges_m[1:])
    d_centers_um_model = r_centers_m * 2 * 1e6
    dry_spec = np.asarray(trial_data["dry spectrum"]).squeeze()
    mean_dry = np.nanmean(dry_spec, axis=0)
    N_model = mean_dry * 1e-12  # [#/cm³/µm]
    N_model_interp = np.interp(cas_bins_10, d_centers_um_model, N_model, left=np.nan, right=np.nan)
    n0, D_e = fit["n0"], fit["D"]
    N_fit_interp = exponential(cas_bins_10, n0, D_e)
    plt.figure(figsize=(6, 5))
    plt.semilogy(cas_bins_10, N_model_interp, "o-", color="tab:blue", lw=2,
                 label=f"Model (interpolated to CAS bins)")
    plt.semilogy(cas_bins_10, N_fit_interp, "r--", lw=2,
                 label=f"Exponential Fit\n$n_0$={n0:.3f}, D={D_e:.3f} µm")

    plt.axvline(2.0, color="k", ls="--", lw=1)
    plt.xlim(2, 10)
    plt.ylim(1e-7, 1e1)
    plt.xlabel("Dry Diameter (µm)", fontsize=13, fontweight="bold")
    plt.ylabel("Number Concentration (cm$^{-3}$ µm$^{-1}$)", fontsize=13, fontweight="bold")
    plt.title(f"Feb 15 2022 – Leg {leg_idx+1}\nModel vs Exponential Fit (≤10 µm)",
              fontsize=13, fontweight="bold")
    plt.legend(fontsize=9)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()
#%%
fits_feb15 = [
    {"n0": 0.812, "D": 0.828},
    {"n0": 1.073, "D": 0.757},
    {"n0": 5.232, "D": 0.574},
    {"n0": 1.819, "D": 0.798},
    {"n0": 2.263, "D": 0.744},
    {"n0": 1.146, "D": 0.944},
    {"n0": 0.672, "D": 0.944},
    {"n0": 1.919, "D": 0.546},
    {"n0": 1.704, "D": 0.790},
    {"n0": 2.018, "D": 0.934},
    {"n0": 2.269, "D": 0.846},
    {"n0": 1.554, "D": 0.958},
    {"n0": 2.728, "D": 0.570},
    {"n0": 1.058, "D": 0.709},
]

def exponential(D, n0, D_e):
    return n0 * np.exp(-D / D_e)

r_edges_m = np.logspace(-7, -5, 101)
r_centers_m = np.sqrt(r_edges_m[:-1] * r_edges_m[1:])
d_centers_um = r_centers_m * 2 * 1e6
base_path = "/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions/one month size distributions"

num_legs = len(fits_feb15)

for leg_idx, fit in enumerate(fits_feb15):
    file_path = os.path.join(base_path, f"{leg_idx} (1).pickle")
    if not os.path.exists(file_path):
        print(f"⚠️ Missing file: {file_path}")
        continue

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    dry_spec = np.asarray(data["dry spectrum"]).squeeze()
    if dry_spec.size == 0:
        print(f"⚠️ Empty dry spectrum in {leg_idx} (1).pickle, skipping.")
        continue

    mean_dry = np.nanmean(dry_spec, axis=0)
    N_model = mean_dry * 1e-12  # #/cm³/µm
    n0, D_e = fit["n0"], fit["D"]
    N_fit = exponential(d_centers_um, n0, D_e)

    plt.figure(figsize=(7, 5))
    plt.semilogy(d_centers_um, N_model, color="tab:blue", lw=2,
                 label=f"Model Dry Spectrum (Leg {leg_idx+1})")
    plt.semilogy(d_centers_um, N_fit, "r--", lw=2,
                 label=f"Exponential Fit\n$n_0$={n0:.3f}, D={D_e:.3f} µm")

    plt.axvline(2.0, color="k", ls="--", lw=1)
    plt.xlim(0.2, 20)
    plt.ylim(1e-7, 1e1)
    plt.xlabel("Dry Diameter (µm)", fontsize=13, fontweight="bold")
    plt.ylabel("Number Concentration (cm$^{-3}$ µm$^{-1}$)", fontsize=13, fontweight="bold")
    plt.title(f"Feb 15 2022 – Leg {leg_idx+1}\nModel vs Exponential Fit (≤10 µm)",
              fontsize=13, fontweight="bold")
    plt.legend(fontsize=9)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

# %%
n0, D_e = 0.812, 0.828 

def exponential(D_um, n0, D_e):
    return n0 * np.exp(-D_um / D_e)
r_edges_m = np.logspace(-7, -5, 101)                     # 100 bins
r_centers_m = np.sqrt(r_edges_m[:-1] * r_edges_m[1:])    # radius centers
d_centers_m = r_centers_m * 2
d_centers_um = d_centers_m * 1e6                         # µm centers
file_path = "/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions/one month size distributions/0 (1).pickle"

with open(file_path, "rb") as f:
    data = pickle.load(f)

dry_spec = np.asarray(data["dry spectrum"]).squeeze()  # [#/m⁴]
mean_dry = np.nanmean(dry_spec, axis=0)
N_model = mean_dry * 1e-12  # (#/m⁴) * (1e-6 m³/cm³) * (1e6 µm/m) = 1e-12
D = d_centers_um
N_rev = np.flip(N_model)
D_rev = np.flip(D)
N_cum_rev = np.zeros_like(N_rev)
for i in range(1, len(D_rev)):
    N_cum_rev[i] = N_cum_rev[i-1] + 0.5 * (N_rev[i] + N_rev[i-1]) * abs(D_rev[i] - D_rev[i-1])

N_model_cum = np.flip(N_cum_rev)
print(f"Leg 0: raw N_model range = {np.nanmin(N_model):.3e} – {np.nanmax(N_model):.3e}")
print(f"Leg 0: cumulative N range = {np.nanmin(N_model_cum):.3e} – {np.nanmax(N_model_cum):.3e}")
D_fit = np.linspace(2, 10, 300)
N_fit = exponential(D_fit, n0, D_e)
plt.figure(figsize=(7, 5))
plt.semilogy(D, N_model_cum, color="tab:blue", lw=2, label="Cumulative Model (Right→Left)")
plt.semilogy(D_fit, N_fit, "r--", lw=2, label=f"Exponential Fit\n$n_0$={n0:.3f}, D={D_e:.3f} µm")

plt.axvline(2.0, color="k", ls="--", lw=1)
plt.xlim(2, 10)
plt.ylim(1e-6, 1e3)
plt.xlabel("Dry Diameter (µm)", fontsize=14, fontweight="bold")
plt.ylabel("Cumulative Number (cm$^{-3}$)", fontsize=14, fontweight="bold")
plt.title("Feb 15 2022 – Leg 0\nCumulative Model (Right→Left) vs Exponential Fit (≤ 10 µm)",
          fontsize=13, fontweight="bold")
plt.legend(fontsize=10)
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.tight_layout()
plt.show()
#%%
def _shim_numpy_for_pickles():
    sys.modules.setdefault('numpy.core', np)
    sys.modules.setdefault('numpy.core.multiarray', np.core.multiarray)
    sys.modules.setdefault('numpy._core', np)
    sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)

_shim_numpy_for_pickles()
base = "/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions/Feb15Ensemble"
fits_feb15 = [
    {"n0": 0.812, "D": 0.828},
    {"n0": 1.073, "D": 0.757},
    {"n0": 5.232, "D": 0.574},
    {"n0": 1.819, "D": 0.798},
    {"n0": 2.263, "D": 0.744},
    {"n0": 1.146, "D": 0.944},
    {"n0": 0.672, "D": 0.944},
    {"n0": 1.919, "D": 0.546},
    {"n0": 1.704, "D": 0.790},
    {"n0": 2.018, "D": 0.934},
    {"n0": 2.269, "D": 0.846},
    {"n0": 1.554, "D": 0.958},
    {"n0": 2.728, "D": 0.570},
    {"n0": 1.058, "D": 0.709},
]

def exponential(D, n0, D_e):
    return n0 * np.exp(-D / D_e)
pat = re.compile(r'^(\d+)_([0-9]+)\.pickle$')
r_edges_m = np.logspace(-7, -5, 101)
r_centers_m = np.sqrt(r_edges_m[:-1] * r_edges_m[1:])
d_centers_um = r_centers_m * 2 * 1e6
mask = d_centers_um <= 10
D_model = d_centers_um[mask]
for fn in sorted(os.listdir(base)):
    m = pat.match(fn)
    if not m:
        continue

    leg_idx = int(m.group(1))
    sub_idx = int(m.group(2))
    if leg_idx >= len(fits_feb15):
        continue  # skip extra legs if beyond fit list

    model_path = os.path.join(base, fn)

    with open(model_path, "rb") as f:
        trial_data = pickle.load(f)

    dry_spec = np.asarray(trial_data["dry spectrum"]).squeeze()
    mean_dry = np.nanmean(dry_spec, axis=0)
    N_model_full = mean_dry * 1e-12
    N_model = N_model_full[mask]
    n0, D_e = fits_feb15[leg_idx]["n0"], fits_feb15[leg_idx]["D"]
    D_fit = np.linspace(2, 10, 200)
    N_fit = exponential(D_fit, n0, D_e)
    plt.figure(figsize=(6, 5))
    plt.semilogy(D_model, N_model, color="tab:blue", lw=2,
                 label=f"Model Spectrum (Leg {leg_idx}_{sub_idx})")
    plt.semilogy(D_fit, N_fit, "r--", lw=2,
                 label=f"Exponential Fit\n$n_0$={n0:.3f}, D={D_e:.3f} µm")
    plt.axvline(2.0, color="k", ls="--", lw=1)
    plt.xlim(2, 10)
    plt.ylim(1e-7, 1e1)
    plt.xlabel("Dry Diameter (µm)", fontsize=13, fontweight="bold")
    plt.ylabel("Number Concentration (cm$^{-3}$ µm$^{-1}$)", fontsize=13, fontweight="bold")
    plt.title(f"Feb 15 2022 – Leg {leg_idx}_{sub_idx}\nModel vs Exponential Fit (≤10 µm)",
              fontsize=13, fontweight="bold")
    plt.legend(fontsize=9)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

# %%
fits_feb15 = [
    {"n0": 0.812, "D": 0.828},
    {"n0": 1.073, "D": 0.757},
    {"n0": 5.232, "D": 0.574},
    {"n0": 1.819, "D": 0.798},
    {"n0": 2.263, "D": 0.744},
    {"n0": 1.146, "D": 0.944},
    {"n0": 0.672, "D": 0.944},
    {"n0": 1.919, "D": 0.546},
    {"n0": 1.704, "D": 0.790},
    {"n0": 2.018, "D": 0.934},
    {"n0": 2.269, "D": 0.846},
    {"n0": 1.554, "D": 0.958},
    {"n0": 2.728, "D": 0.570},
    {"n0": 1.058, "D": 0.709},
]

def exponential(D_um, n0, D_e):
    return n0 * np.exp(-D_um / D_e)
r_edges_m = np.logspace(-7, -5, 101)
r_centers_m = np.sqrt(r_edges_m[:-1] * r_edges_m[1:])
d_centers_m = r_centers_m * 2
d_centers_um = d_centers_m * 1e6

base_path = "/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions/one month size distributions"

for leg_idx, fit in enumerate(fits_feb15):
    file_path = os.path.join(base_path, f"{leg_idx} (1).pickle")
    if not os.path.exists(file_path):
        print(f"⚠️ Missing file: {file_path}")
        continue

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    dry_spec = np.asarray(data["dry spectrum"]).squeeze()  # [#/m⁴]
    if dry_spec.size == 0:
        print(f"⚠️ Empty dry spectrum in leg {leg_idx}")
        continue
    mean_dry = np.nanmean(dry_spec, axis=0)
    N_model = mean_dry * 1e-12
    D = d_centers_um
    N_rev = np.flip(N_model)
    D_rev = np.flip(D)
    N_cum_rev = np.zeros_like(N_rev)
    N_cum_rev[1:] = np.cumsum(
        0.5 * (N_rev[1:] + N_rev[:-1]) * np.abs(np.diff(D_rev))
    )
    N_model_cum = np.flip(N_cum_rev)

    print(f"Leg {leg_idx+1}: cumulative max = {np.nanmax(N_model_cum):.3e}")
    n0, D_e = fit["n0"], fit["D"]
    D_fit = np.linspace(2, 10, 300)
    N_fit = exponential(D_fit, n0, D_e)
    plt.figure(figsize=(7, 5))
    plt.semilogy(D, N_model_cum, color="tab:blue", lw=2,
                 label="Cumulative Model Fit")
    plt.semilogy(D_fit, N_fit, "r--", lw=2,
                 label=f"Dry Exponential Fit\n$n_0$={n0:.3f}, D={D_e:.3f} µm")

    plt.axvline(2.0, color="k", ls="--", lw=1)
    plt.xlim(2, 10)
    plt.ylim(1e-6, 1e3)
    plt.yticks(fontsize=16, fontweight="bold")
    plt.xticks(fontsize=16, fontweight="bold")
    plt.xlabel("Dry Diameter (µm)", fontsize=19, fontweight="bold")
    plt.ylabel("Cumulative Number (cm$^{-3}$)", fontsize=19, fontweight="bold")
    plt.title(f"Below Cloud Base\nFeb 15, 2022 – Leg {leg_idx+1}",
              fontsize=19, fontweight="bold")
    plt.legend(fontsize=13)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

# %%
fits_feb15 = [
    {"n0": 0.812, "D": 0.828},
    {"n0": 1.073, "D": 0.757},
    {"n0": 5.232, "D": 0.574},
    {"n0": 1.819, "D": 0.798},
    {"n0": 2.263, "D": 0.744},
    {"n0": 1.146, "D": 0.944},
    {"n0": 0.672, "D": 0.944},
    {"n0": 1.919, "D": 0.546},
    {"n0": 1.704, "D": 0.790},
    {"n0": 2.018, "D": 0.934},
    {"n0": 2.269, "D": 0.846},
    {"n0": 1.554, "D": 0.958},
    {"n0": 2.728, "D": 0.570},
    {"n0": 1.058, "D": 0.709},
]

def N_exp(D_um, n0, De):
    return n0 * np.exp(-D_um / De)
def cumulative_right_to_left(D_um, N_diff):
    D_rev = np.flip(D_um)
    N_rev = np.flip(N_diff)
    cum_rev = np.zeros_like(N_rev)
    cum_rev[1:] = np.cumsum(0.5 * (N_rev[1:] + N_rev[:-1]) * np.abs(np.diff(D_rev)))
    return np.flip(cum_rev)
r_edges_m = np.logspace(-7, -5, 101)
r_centers_m = np.sqrt(r_edges_m[:-1] * r_edges_m[1:])
D_um = (r_centers_m * 2) * 1e6 
base_path = "/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions/one month size distributions"
for leg_idx, fit in enumerate(fits_feb15):
    file_path = os.path.join(base_path, f"{leg_idx} (1).pickle")
    if not os.path.exists(file_path):
        print(f"⚠️ Missing file: {file_path}")
        continue
    with open(file_path, "rb") as f:
        trial = pickle.load(f)

    dry_spec = np.asarray(trial["dry spectrum"]).squeeze() 
    if dry_spec.size == 0:
        print(f"⚠️ Empty dry spectrum in leg {leg_idx}")
        continue
    N_model = (np.nanmean(dry_spec, axis=0) / 2) * 1e-12
    N_model_cum = cumulative_right_to_left(D_um, N_model)
    n0, De = fit["n0"], fit["D"]
    N_fit_diff = N_exp(D_um, n0, De)
    N_fit_cum  = cumulative_right_to_left(D_um, N_fit_diff)
    m10 = D_um <= 10
    Dx   = D_um[m10]
    Mcum = N_model_cum[m10]
    Fcum = N_fit_cum[m10]
    ymin = max(1e-6, np.nanmin([Mcum[Mcum>0].min() if np.any(Mcum>0) else 1e-6,
                                Fcum[Fcum>0].min() if np.any(Fcum>0) else 1e-6]))
    ymax = np.nanmax([Mcum.max(), Fcum.max(), 1e0]) * 1.2

    plt.figure(figsize=(7,5))
    plt.semilogy(Dx, Mcum, lw=2, label="Cumulative Model Distribution")
    plt.semilogy(Dx, Fcum, "r--", lw=2,
                 label=f"Cumulative Dry Exponential Fit\n$n_0$={n0:.3f}, D={De:.3f} µm")
    plt.axvline(2.0, color="k", ls="--", lw=1)
    plt.xlim(2, 10)
    plt.ylim(ymin, ymax)
    plt.xlabel("Dry Diameter (µm)", fontsize=14, fontweight="bold")
    plt.ylabel("Cumulative Number (cm$^{-3}$)", fontsize=14, fontweight="bold")
    plt.title(f"Below Cloud Base\nFeb 15, 2022 – Leg {leg_idx+1}",
              fontsize=13, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()
# %%
#leaving in jasons original units

fits_feb15 = [
    {"n0": 0.812, "D": 0.828},
    {"n0": 1.073, "D": 0.757},
    {"n0": 5.232, "D": 0.574},
    {"n0": 1.819, "D": 0.798},
    {"n0": 2.263, "D": 0.744},
    {"n0": 1.146, "D": 0.944},
    {"n0": 0.672, "D": 0.944},
    {"n0": 1.919, "D": 0.546},
    {"n0": 1.704, "D": 0.790},
    {"n0": 2.018, "D": 0.934},
    {"n0": 2.269, "D": 0.846},
    {"n0": 1.554, "D": 0.958},
    {"n0": 2.728, "D": 0.570},
    {"n0": 1.058, "D": 0.709},
]

# Jason’s grid (radius in meters)
r_edges_m   = np.logspace(-7, -5, 101)
r_centers_m = np.sqrt(r_edges_m[:-1] * r_edges_m[1:])
dr_m        = np.diff(r_edges_m)  # IMPORTANT: variable bin widths

def cumulative_right_to_left_from_centers(r_edges, N_r_centers):
    """
    Compute cumulative number (right→left) from density N_r(r) [#/m^3/m]
    given at bin centers, using bin widths Δr from edges.
    Returns #/m^3 at centers.
    """
    dr = np.diff(r_edges)
    # integrate bin-by-bin from large→small: N * Δr
    contrib = (N_r_centers * dr)[::-1]
    cum = np.cumsum(contrib)
    return cum[::-1]

def N_exp_diff_radius(r_m, n0_cm_um, De_um):
    """Differential exponential in radius (SI): N_r [#/m^3/m]."""
    n0_SI = n0_cm_um * 1e12   # (#/cm^3/µm) -> (#/m^3/m)
    De_m  = De_um * 1e-6      # µm -> m
    return 2.0 * n0_SI * np.exp(-2.0 * r_m / De_m)

def N_exp_cum_radius_analytic(r_m, n0_cm_um, De_um):
    """
    Analytic cumulative from r to ∞:
      ∫_r^∞ 2 n0_SI exp(-2x/De_m) dx = n0_SI * De_m * exp(-2r/De_m)
    Units: #/m^3.
    """
    n0_SI = n0_cm_um * 1e12
    De_m  = De_um * 1e-6
    return n0_SI * De_m * np.exp(-2.0 * r_m / De_m)

base_path = "/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions/one month size distributions"

for leg_idx, fit in enumerate(fits_feb15):
    file_path = os.path.join(base_path, f"{leg_idx} (1).pickle")
    if not os.path.exists(file_path):
        print(f"⚠️ Missing file: {file_path}")
        continue

    with open(file_path, "rb") as f:
        trial = pickle.load(f)

    dry_spec = np.asarray(trial["dry spectrum"]).squeeze()  # [#/m^3/m] over radius bins
    if dry_spec.size == 0:
        print(f"⚠️ Empty spectrum in leg {leg_idx+1}")
        continue

    mean_dry = np.nanmean(dry_spec, axis=0)  # [#/m^3/m]
    if not np.any(np.isfinite(mean_dry)):
        print(f"⚠️ No finite data in leg {leg_idx+1}")
        continue

    # ✅ Correct cumulative (right→left) with Δr
    N_model_cum = cumulative_right_to_left_from_centers(r_edges_m, np.nan_to_num(mean_dry, nan=0.0))  # [#/m^3]

    # ✅ Exponential cumulative (analytic)
    n0, De = fit["n0"], fit["D"]
    N_fit_cum = N_exp_cum_radius_analytic(r_centers_m, n0, De)  # [#/m^3]

    # y-lims
    pos = np.concatenate([
        N_model_cum[(N_model_cum > 0) & np.isfinite(N_model_cum)],
        N_fit_cum[(N_fit_cum > 0) & np.isfinite(N_fit_cum)]
    ])
    if pos.size == 0:
        print(f"⚠️ No positive data for leg {leg_idx+1}")
        continue
    ymin, ymax = max(1e-6, pos.min()*0.8), pos.max()*1.2

    # Plot in Jason's native units (radius, meters)
    plt.figure(figsize=(7,4))
    plt.semilogx(r_centers_m, N_model_cum, lw=2, label="Cumulative Dry Spectrum (Model)")
    plt.semilogx(r_centers_m, N_fit_cum, "r--", lw=2,
                 label=f"Cumulative Exponential Fit\nn₀={n0:.3f} #/cm³/µm, Dₑ={De:.3f} µm")
    plt.xlabel("Dry Radius (m)", fontsize=14, fontweight="bold")
    plt.ylabel("Cumulative Number (m$^{-3}$)", fontsize=14, fontweight="bold")
    plt.yscale("log")
    plt.xscale('linear')
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.title(f"Below Cloud Base — Feb 15, 2022 — Leg {leg_idx+1}", fontsize=13, fontweight="bold")
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

#%%
n0 = 0.812
De = 0.828
D = np.linspace(0, 10, 400)
N_diff = n0 * np.exp(-D / De)
N_cum = n0 * De * np.exp(-D / De)
plt.figure(figsize=(6,4))
plt.semilogy(D, N_diff, 'r--', lw=2, label='$N(D)$')
plt.semilogy(D, N_cum, 'b-',  lw=2, label='Cumulative $N_{cum}(D)$')
plt.xlabel("Dry Diameter (µm)", fontsize=13, fontweight='bold')
plt.ylabel("Number (cm$^{-3}$ µm$^{-1}$ or cm$^{-3}$)", fontsize=13, fontweight='bold')
plt.title("Exponential Fit", fontsize=13, fontweight='bold')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()
plt.figure(figsize=(6,4))
plt.plot(D, N_diff, 'r--', lw=2, label='$N(D)$')
plt.plot(D, N_cum, 'b-',  lw=2, label='Cumulative $N_{cum}(D)$')
plt.xlabel("Dry Diameter (µm)", fontsize=13, fontweight='bold')
plt.ylabel("Number", fontsize=13, fontweight='bold')
plt.title("Exponential Fit", fontsize=13, fontweight='bold')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()
# %%
#Jason's dry spectrum, keeping 100 bin radius but converting units to #/cm3/um
r_edges_m = np.logspace(-7, -5, 101)
r_centers_m = np.sqrt(r_edges_m[:-1] * r_edges_m[1:])
r_centers_um = r_centers_m * 1e6  # convert m → µm
base_path = "/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions/one month size distributions"
for leg_idx in range(14):
    file_path = os.path.join(base_path, f"{leg_idx} (1).pickle")
    if not os.path.exists(file_path):
        print(f"⚠️ Missing file: {file_path}")
        continue

    with open(file_path, "rb") as f:
        trial = pickle.load(f)

    dry_spec = np.asarray(trial["dry spectrum"]).squeeze()  # [#/m⁴]
    if dry_spec.size == 0:
        print(f"⚠️ Empty dry spectrum in leg {leg_idx}")
        continue
    mean_dry = np.nanmean(dry_spec, axis=0)
    N_model = mean_dry * 1e-12
    plt.figure(figsize=(7, 5))
    plt.semilogy(r_centers_um, N_model, lw=2, color="tab:blue",
                 label=f"Leg {leg_idx+1} Dry Spectrum")

    plt.axvline(2.0, color="k", ls="--", lw=1)
    plt.xlim(r_centers_um.min(), r_centers_um.max())  # full 0.1–20 µm
    plt.ylim(1e-7, 1e1)
    plt.xlabel("Dry Radius (µm)", fontsize=14, fontweight="bold")
    plt.ylabel("Number Concentration (cm$^{-3}$ µm$^{-1}$)", fontsize=14, fontweight="bold")
    plt.title(f"Below Cloud Base – Feb 15 2022\nModel Dry Spectrum (Leg {leg_idx+1})",
              fontsize=13, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

# %%
#checking to see which bins are populated 
leg_idx = 2
file_path = f"/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions/one month size distributions/{leg_idx} (1).pickle"
with open(file_path, "rb") as f:
    trial = pickle.load(f)
dry_spec = np.asarray(trial["dry spectrum"]).squeeze()
mean_dry = np.nanmean(dry_spec, axis=0)
r_edges_m = np.logspace(-7, -5, 101)
r_centers_m = np.sqrt(r_edges_m[:-1] * r_edges_m[1:])
r_centers_um = r_centers_m * 1e6

N_model = mean_dry * 1e-12
nonzero_bins = np.count_nonzero(N_model)
print(f"Number of nonzero bins: {nonzero_bins} / {len(N_model)}")
print(f"Radius range: {r_centers_um.min():.2f} – {r_centers_um.max():.2f} µm")
print(f"Nonzero values stop near: {r_centers_um[np.where(N_model>0)[-1][-1]]:.2f} µm")

# %%
#using cumulative distribution for model, but continue jasons spectra 

def cumulative_right_to_left(r_um, N_diff):
    r_rev = np.flip(r_um)
    N_rev = np.flip(N_diff)
    N_cum_rev = np.zeros_like(N_rev)
    N_cum_rev[1:] = np.cumsum(0.5 * (N_rev[1:] + N_rev[:-1]) * np.abs(np.diff(r_rev)))
    return np.flip(N_cum_rev)
r_edges_m = np.logspace(-7, -5, 101)
r_centers_m = np.sqrt(r_edges_m[:-1] * r_edges_m[1:])
r_centers_um = r_centers_m * 1e6  # convert m → µm
base_path = "/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions/one month size distributions"
for leg_idx in range(14):
    file_path = os.path.join(base_path, f"{leg_idx} (1).pickle")
    if not os.path.exists(file_path):
        print(f"⚠️ Missing file: {file_path}")
        continue

    with open(file_path, "rb") as f:
        trial = pickle.load(f)

    dry_spec = np.asarray(trial["dry spectrum"]).squeeze()  # [#/m⁴]
    if dry_spec.size == 0:
        print(f"⚠️ Empty dry spectrum in leg {leg_idx}")
        continue
    mean_dry = np.nanmean(dry_spec, axis=0)
    N_model = mean_dry * 1e-12
    N_model_cum = cumulative_right_to_left(r_centers_um, N_model)
    plt.figure(figsize=(7, 5))
    plt.semilogy(r_centers_um, N_model_cum, lw=2, color="tab:blue",
                 label=f"Leg {leg_idx+1} Cumulative Dry Spectrum (Right→Left)")

    plt.axvline(2.0, color="k", ls="--", lw=1)
    plt.xlim(r_centers_um.min(), r_centers_um.max())   # full 0.1–10 µm radius
    plt.ylim(1e-6, 1e3)
    plt.xlabel("Dry Radius (µm)", fontsize=14, fontweight="bold")
    plt.ylabel("Cumulative Number (cm$^{-3}$)", fontsize=14, fontweight="bold")
    plt.title(f"Below Cloud Base – Feb 15 2022\nModel Cumulative Dry Spectrum (Leg {leg_idx+1})",
              fontsize=13, fontweight="bold")
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

# %%
#adding the exponentials to jasons spectra
fits_feb15 = [
    {"n0": 0.812, "D": 0.828},
    {"n0": 1.073, "D": 0.757},
    {"n0": 5.232, "D": 0.574},
    {"n0": 1.819, "D": 0.798},
    {"n0": 2.263, "D": 0.744},
    {"n0": 1.146, "D": 0.944},
    {"n0": 0.672, "D": 0.944},
    {"n0": 1.919, "D": 0.546},
    {"n0": 1.704, "D": 0.790},
    {"n0": 2.018, "D": 0.934},
    {"n0": 2.269, "D": 0.846},
    {"n0": 1.554, "D": 0.958},
    {"n0": 2.728, "D": 0.570},
    {"n0": 1.058, "D": 0.709},
]
def N_exp(r_um, n0, D_e):
    return n0 * np.exp(-r_um / D_e)

def cumulative_right_to_left_edges(r_edges_um, N_centers):
    """
    Integrate N(r) [#/cm³/µm] over true bin widths Δr (from edges).
    """
    dr_um = np.diff(r_edges_um)
    contrib = (N_centers * dr_um)[::-1]
    cum = np.cumsum(contrib)
    return cum[::-1]

r_edges_m = np.logspace(-7, -5, 101)
r_centers_m = np.sqrt(r_edges_m[:-1] * r_edges_m[1:])
r_centers_um = r_centers_m * 1e6  # m → µm
base_path = "/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions/one month size distributions"
for leg_idx, fit in enumerate(fits_feb15):
    file_path = os.path.join(base_path, f"{leg_idx} (1).pickle")
    if not os.path.exists(file_path):
        print(f"⚠️ Missing file: {file_path}")
        continue
    with open(file_path, "rb") as f:
        trial = pickle.load(f)
    dry_spec = np.asarray(trial["dry spectrum"]).squeeze()
    if dry_spec.size == 0:
        print(f"⚠️ Empty dry spectrum in leg {leg_idx}")
        continue
    mean_dry = np.nanmean(dry_spec, axis=0)
    N_model = mean_dry * 1e-12
    N_model_cum = cumulative_right_to_left(r_centers_um, N_model)
    n0, D_e = fit["n0"], fit["D"]
    N_fit_diff = N_exp(r_centers_um, n0, D_e)
    N_fit_cum  = cumulative_right_to_left(r_centers_um, N_fit_diff)
    plt.figure(figsize=(7, 5))
    plt.semilogy(r_centers_um, N_model_cum, lw=2, color="tab:blue",
                 label=f"Cumulative Model Distribution (Leg {leg_idx+1})")
    plt.semilogy(r_centers_um, N_fit_cum, "r--", lw=2,
                 label=f"Cumulative Dry Exponential Fit\n$n_0$={n0:.3f}, D={D_e:.3f} µm")

    plt.axvline(1.0, color="k", ls="--", lw=1, label="1 µm radius")
    plt.yticks(fontsize=16, fontweight="bold")
    plt.xticks(fontsize=16, fontweight="bold")
    plt.xlim(0.1, 10)
    plt.ylim(1e-6, 1e3)
    plt.xlabel("Dry Radius (µm)", fontsize=19, fontweight="bold")
    plt.ylabel("Cumulative Number (cm$^{-3}$)", fontsize=19, fontweight="bold")
    plt.title(f"Below Cloud Base \nFeb 15, 2022 (Leg {leg_idx+1})",
              fontsize=19, fontweight="bold")
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.show()
#%%
def _shim_numpy_for_pickles():
    sys.modules.setdefault('numpy.core', np)
    sys.modules.setdefault('numpy.core.multiarray', np.core.multiarray)
    sys.modules.setdefault('numpy._core', np)
    sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)

_shim_numpy_for_pickles()
base_path = "/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions/Feb15Ensemble"
title_prefix = "Feb 15 2022 – Ensemble"
save_dir = None
pat = re.compile(r"^(\d+)_([0-9]+)\.pickle$") 
grouped_files = defaultdict(list)
all_files = sorted(f for f in os.listdir(base_path) if f.endswith(".pickle"))
for fn in all_files:
    m = pat.match(fn)
    if not m:
        continue
    leg = int(m.group(1))
    sub = int(m.group(2))
    grouped_files[leg].append((sub, os.path.join(base_path, fn), fn))
if not grouped_files:
    print(f"⚠️ No NN_MM.pickle files found in:\n{base_path}")
else:
    total_segments = sum(len(v) for v in grouped_files.values())
    print(f"✅ Found {total_segments} segments across {len(grouped_files)} legs in {base_path}")
    for leg in sorted(grouped_files):
        segs = sorted(s for s,_,_ in grouped_files[leg])
        print(f"   Leg {leg}: segments {segs}")

def load_series(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    pr = np.asarray(data["surface precipitation"]).squeeze()  # kg/m²/s (= mm/s)
    t  = np.asarray(data["t"]).squeeze()
    if pr.size == 0 or t.size < 2:
        return None, None, None
    order = np.argsort(t)
    t, pr = t[order], pr[order]
    dt = float(np.median(np.diff(t)))
    acc = np.cumsum(pr * dt)  # mm
    return t, pr, float(acc[-1])
summary = []
for leg, entries in sorted(grouped_files.items()):
    entries.sort(key=lambda x: x[0])  # by sub index
    plt.figure(figsize=(8,5))
    leg_total = 0.0
    plotted_any = False

    for sub, path, fn in entries:
        t, pr, tot = load_series(path)
        if t is None:
            print(f"⚠️ Skipping {fn} (empty/invalid).")
            continue
        leg_total += tot
        plt.plot(t/60.0, pr*3600.0, lw=1.6, label=f"{leg}_{sub} ({tot:.1e} mm)")
        plotted_any = True

    if not plotted_any:
        plt.close()
        continue
    summary.append((leg, leg_total))
    plt.xlabel("Time (minutes)", fontsize=14, fontweight="bold")
    plt.ylabel("Precipitation Rate (mm hr$^{-1}$)", fontsize=14, fontweight="bold")
    plt.title(f"{title_prefix} – Leg {leg}\nTotal across segments = {leg_total:.2e} mm",
              fontsize=13, fontweight="bold")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(fontsize=9, ncol=2, loc="upper right")
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, f"precip_leg_{leg}.png")
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"🖼️ Saved: {out}")
    else:
        plt.show()
if summary:
    print("\n=== Total accumulated precipitation per main leg (mm) ===")
    for leg, total in sorted(summary):
        print(f"Leg {leg:02d}: {total:.3e} mm")
else:
    print("\n⚠️ No plots produced (no matching files or all invalid).")

# %%
#surface precip and gccn concentration 
from scipy.stats import linregress
data = {
    "Leg": np.arange(1, 15),
    "GCCN": [7.151e-02, 6.848e-02, 1.090e-01, 1.406e-01, 1.359e-01,
             1.547e-01,  9.059e-02, 3.172e-02, 1.273e-01, 2.641e-01,
              2.146e-01,2.199e-01, 5.532e-02, 5.300e-02],
    "Rain": [1.371e-03, 1.390e-03, 9.841e-04, 2.925e-03, 2.208e-03, 3.777e-03,
             1.698e-03,3.510e-04, 2.365e-03,6.034e-03,3.954e-03, 4.777e-03, 
             3.089e-04, 9.343e-04]
}
 
df = pd.DataFrame(data)
logx = np.log10(df["GCCN"])
logy = np.log10(df["Rain"])
slope, intercept, r_value, p_value, std_err = linregress(logx, logy)
r2 = r_value ** 2
fit_x = np.logspace(np.log10(df["GCCN"].min()), np.log10(df["GCCN"].max()), 200)
fit_y = 10 ** (intercept + slope * np.log10(fit_x))
plt.figure(figsize=(8,6))
palette = sns.color_palette("tab20", len(df))

for i, row in df.iterrows():
    plt.scatter(row["GCCN"], row["Rain"], 
                color=palette[i], s=80, edgecolor="k", 
                label=f"Leg {int(row['Leg'])}")

plt.plot(fit_x, fit_y, "r--", lw=2,
         label=f"Fit: slope={slope:.2f}, r={r_value:.2f}, r²={r2:.2f}")

plt.xscale("log")
plt.yscale("log")
plt.yticks(fontsize=16, fontweight="bold")
plt.xticks(fontsize=16, fontweight="bold")
plt.xlabel("Total GCCN Concentration (cm$^{-3}$)", fontsize=17, fontweight="bold")
plt.ylabel("Total Accumulated Rainfall (mm hr$^{-1}$)", fontsize=17, fontweight="bold")
plt.title("Below Cloud Base\nFeb 15, 2022\nRainfall vs GCCN Concentration", fontsize=17, fontweight="bold")
plt.legend(fontsize=10, ncol=3, loc="lower right")
plt.tight_layout()
plt.show()
print(f"Slope = {slope:.3f}")
print(f"r = {r_value:.3f}")
print(f"r² = {r2:.3f}")
#%%
from scipy.integrate import cumulative_trapezoid 
def _shim_numpy_for_pickles():
    sys.modules.setdefault('numpy.core', np)
    sys.modules.setdefault('numpy.core.multiarray', np.core.multiarray)
    sys.modules.setdefault('numpy._core', np)
    sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)

_shim_numpy_for_pickles()
leg_idx = 0
sub_idx = 1  
n0 = 0.812  
D  = 0.828  
r = np.logspace(-7, -5, 101)  
r_ = np.sqrt(r[:-1] * r[1:])   

base_path = "/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions/Feb15Ensemble"
file_path = os.path.join(base_path, f"{leg_idx}_{sub_idx}.pickle")

with open(file_path, "rb") as f:
    trial = pickle.load(f)

spec_all  = np.asarray(trial["dry spectrum"]).squeeze()  
spec_mean = np.nanmean(spec_all, axis=0)                 
cum_model = -cumulative_trapezoid(spec_mean[::-1], r_[::-1], initial=0)[::-1]
N_fit_diff = n0 * 1e6 * np.exp(-r_ / (D * 1e-6 / 2)) 

plt.figure(figsize=(7,5))
plt.semilogy(r_, cum_model, lw=2, color="tab:blue", label="Cumulative Dry Spectrum (Model)")
plt.semilogy(r_, N_fit_diff, "r--", lw=2, label=f"Exponential Fit\nn₀={n0:.3f}, Dₑ={D:.3f} µm")
plt.xlim(0, 1e-5)
plt.xlabel("Dry Radius (m)", fontsize=16, fontweight="bold")
plt.ylabel("Cumulative Number (m$^{-3}$ m$^{-1}$)", fontsize=16, fontweight="bold")
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.tight_layout()
plt.title(f"Feb 15 2022 – Leg {leg_idx}_{sub_idx}")
plt.show()
#%%
from scipy.integrate import cumulative_trapezoid

def _shim_numpy_for_pickles():
    sys.modules.setdefault('numpy.core', np)
    sys.modules.setdefault('numpy.core.multiarray', np.core.multiarray)
    sys.modules.setdefault('numpy._core', np)
    sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)

_shim_numpy_for_pickles()
leg_idx = 0
n0, D = 0.812, 0.828 
r = np.logspace(-7, -5, 101)
r_ = np.sqrt(r[:-1] * r[1:])

base = "/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions/Feb15Ensemble"
subs = sorted(
    int(fn.split("_")[1].split(".")[0])
    for fn in os.listdir(base)
    if fn.startswith(f"{leg_idx}_") and fn.endswith(".pickle")
)
for sub_idx in subs:
    path = os.path.join(base, f"{leg_idx}_{sub_idx}.pickle")
    with open(path, "rb") as f:
        trial = pickle.load(f)
    spec_all  = np.asarray(trial["dry spectrum"]).squeeze()
    spec_mean = np.nanmean(spec_all, axis=0)
    cum_model = -cumulative_trapezoid(spec_mean[::-1], r_[::-1], initial=0)[::-1]
    N_fit_diff = n0 * 1e6 * np.exp(-r_ / (D * 1e-6 / 2))
    plt.figure(figsize=(7,5))
    plt.semilogy(r_, cum_model, lw=2, color="tab:blue", label="Cumulative Dry Spectrum (Model)")
    plt.semilogy(r_, N_fit_diff, "r--", lw=2, label=f"Exponential Fit\nn₀={n0:.3f}, Dₑ={D:.3f} µm")
    plt.xlim(0, 1e-5)
    plt.xlabel("Dry Radius (m)", fontsize=16, fontweight="bold")
    plt.ylabel("Cumulative Number (m$^{-3}$ m$^{-1}$)", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.title(f"Feb 15 2022 – Leg {leg_idx}_{sub_idx}")
    plt.show()

# %%
#all legs (correct)
try:
    from scipy.integrate import cumulative_trapezoid as trapz_func
except ImportError:
    from scipy.integrate import cumtrapz as trapz_func
def _shim_numpy_for_pickles():
    sys.modules.setdefault('numpy.core', np)
    sys.modules.setdefault('numpy.core.multiarray', np.core.multiarray)
    sys.modules.setdefault('numpy._core', np)
    sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
_shim_numpy_for_pickles()
fits_feb15 = [
    {"n0": 0.812, "D": 0.828}, {"n0": 1.073, "D": 0.757},
    {"n0": 5.232, "D": 0.574}, {"n0": 1.819, "D": 0.798},
    {"n0": 2.263, "D": 0.744}, {"n0": 1.146, "D": 0.944},
    {"n0": 0.672, "D": 0.944}, {"n0": 1.919, "D": 0.546},
    {"n0": 1.704, "D": 0.790}, {"n0": 2.018, "D": 0.934},
    {"n0": 2.269, "D": 0.846}, {"n0": 1.554, "D": 0.958},
    {"n0": 2.728, "D": 0.570}, {"n0": 1.058, "D": 0.709},
]
r_edges_m = np.logspace(-7, -5, 101)         
r_centers = np.sqrt(r_edges_m[:-1] * r_edges_m[1:]) 
gccn_mask = r_centers > 0.5e-6               

def N_fit_diff_radius(r_m, n0_cm3_per_um, D_um):
    return n0_cm3_per_um * 1e6 * np.exp(-r_m / (D_um * 1e-6 / 2.0)) 
base_path = "/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions/Feb15Ensemble"
overlay_per_leg = True 
pat = re.compile(r"^(\d+)_([0-9]+)\.pickle$")
by_leg = {}
for fn in sorted(f for f in os.listdir(base_path) if f.endswith(".pickle")):
    m = pat.match(fn)
    if not m: 
        continue
    leg = int(m.group(1))
    sub = int(m.group(2))
    by_leg.setdefault(leg, []).append((sub, os.path.join(base_path, fn), fn))
all_gccn = [] 
all_accum = []

for leg in sorted(by_leg.keys()):
    if leg >= len(fits_feb15):
        continue
    n0, D = fits_feb15[leg]["n0"], fits_feb15[leg]["D"]
    entries = sorted(by_leg[leg], key=lambda x: x[0])
    if overlay_per_leg:
        plt.figure(figsize=(7,5))
        plt.title(f"Feb 15 2022 – Leg {leg} (Ensemble segments)", fontsize=16, fontweight="bold")
        plt.xlabel("Dry Radius (m)", fontsize=15, fontweight="bold")
        plt.ylabel("Cumulative Number (m$^{-3}$ m$^{-1}$)", fontsize=15, fontweight="bold")
        plt.grid(True, which="both", ls="--", alpha=0.4)

    leg_print_header_done = False

    for sub, path, fn in entries:
        with open(path, "rb") as f:
            trial = pickle.load(f)
        spec_all = np.asarray(trial["dry spectrum"]).squeeze() 
        if spec_all.size == 0:
            print(f"⚠️ Empty dry spectrum in {fn}")
            continue
        spec_mean = np.nanmean(spec_all, axis=0)               
        cum_model = -trapz_func(spec_mean[::-1], r_centers[::-1], initial=0)[::-1] 
        N_fit_diff = N_fit_diff_radius(r_centers, n0, D)       
        N_gccn_m3  = np.trapz(N_fit_diff[gccn_mask], r_centers[gccn_mask])  
        N_gccn_cm3 = N_gccn_m3 / 1e6
        precip = np.asarray(trial.get("surface precipitation", [])).squeeze()
        t      = np.asarray(trial.get("t", [])).squeeze()
        if precip.size >= 2 and t.size >= 2:
            order = np.argsort(t); t = t[order]; precip = precip[order]
            dt = float(np.median(np.diff(t)))
            total_precip_mm = float(np.sum(precip * dt))
        else:
            total_precip_mm = np.nan
        if not overlay_per_leg:
            plt.figure(figsize=(7,5))
            plt.semilogy(r_centers, cum_model, lw=2, color="tab:blue", label="Cumulative Dry Spectrum (Model)")
            plt.semilogy(r_centers, N_fit_diff, "r--", lw=2, label=f"Exp Fit  n₀={n0:.3f}, Dₑ={D:.3f} µm")
            plt.axvline(0.5e-6, color="k", ls="--", lw=1, label="r = 0.5 µm (D=1 µm)")
            plt.xlim(0, 1e-5); plt.ylim(1e-7, 1e1)
            plt.xlabel("Dry Radius (m)", fontsize=16, fontweight="bold")
            plt.ylabel("Cumulative / Differential (m$^{-3}$ m$^{-1}$)", fontsize=16, fontweight="bold")
            plt.title(f"Leg {leg}_{sub}: GCCN(>1 µm D)={N_gccn_cm3:.2e} cm⁻³; Rain={total_precip_mm:.2e} mm",
                      fontsize=13, fontweight="bold")
            plt.legend(fontsize=10)
            plt.grid(True, which="both", ls="--", alpha=0.4)
            plt.tight_layout()
            plt.show()
        else:
            if not leg_print_header_done:
                print(f"\nLeg {leg} segments:")
                leg_print_header_done = True
            print(f"  {leg}_{sub}: GCCN(>1 µm D) = {N_gccn_cm3:.3e} cm⁻³   Rain = {total_precip_mm:.3e} mm")

            plt.semilogy(r_centers, cum_model, lw=1.6, label=f"{leg}_{sub} (rain {total_precip_mm:.1e} mm)")

        all_gccn.append(N_gccn_cm3)
        all_accum.append(total_precip_mm)
    if overlay_per_leg:
        N_fit_diff_all = N_fit_diff_radius(r_centers, n0, D)
        plt.semilogy(r_centers, N_fit_diff_all, "r--", lw=2,
                     label=f"Exp Fit  n₀={n0:.3f}, Dₑ={D:.3f} µm")
        plt.axvline(0.5e-6, color="k", ls="--", lw=1, label="r = 0.5 µm (D=1 µm)")
        plt.xlim(0, 1e-5); plt.ylim(1e-7, 1e1)
        plt.legend(fontsize=9, ncol=2, loc="upper right")
        plt.tight_layout()
        plt.show()
print("\n=== Summary of GCCN and rainfall ===")
for i, (gccn, rain) in enumerate(zip(all_gccn, all_accum)):
    print(f"Leg {i+1:02d}: GCCN = {gccn:.3e} cm⁻³, Rain = {rain:.3e} mm")

# %%
#scatterplot of  gccn verus rain

def _shim_numpy_for_pickles():
    sys.modules.setdefault('numpy.core', np)
    sys.modules.setdefault('numpy.core.multiarray', np.core.multiarray)
    sys.modules.setdefault('numpy._core', np)
    sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
_shim_numpy_for_pickles()
fits_feb15 = [
    {"n0": 0.812, "D": 0.828}, {"n0": 1.073, "D": 0.757},
    {"n0": 5.232, "D": 0.574}, {"n0": 1.819, "D": 0.798},
    {"n0": 2.263, "D": 0.744}, {"n0": 1.146, "D": 0.944},
    {"n0": 0.672, "D": 0.944}, {"n0": 1.919, "D": 0.546},
    {"n0": 1.704, "D": 0.790}, {"n0": 2.018, "D": 0.934},
    {"n0": 2.269, "D": 0.846}, {"n0": 1.554, "D": 0.958},
    {"n0": 2.728, "D": 0.570}, {"n0": 1.058, "D": 0.709},
]
r_edges_m = np.logspace(-7, -5, 101)                   # 100 radius edges [m]
r_centers = np.sqrt(r_edges_m[:-1] * r_edges_m[1:])     # 100 radius centers [m]
gccn_mask = r_centers > 0.5e-6                          # r > 0.5 µm (D > 1 µm)

def N_fit_diff_radius(r_m, n0_cm3_per_um, D_um):
    return n0_cm3_per_um * 1e6 * np.exp(-r_m / (D_um * 1e-6 / 2.0))
base_path = "/home/disk/eos4/kathem24/activate/data/CAS/one month size distributions/Feb15Ensemble"
pat = re.compile(r"^(\d+)_([0-9]+)\.pickle$") 
by_leg = {}
for fn in sorted(f for f in os.listdir(base_path) if f.endswith(".pickle")):
    m = pat.match(fn)
    if not m:
        continue
    leg = int(m.group(1))
    sub = int(m.group(2))
    by_leg.setdefault(leg, []).append((sub, os.path.join(base_path, fn), fn))
all_gccn = []     
all_accum = []
per_leg = {}      

for leg in sorted(by_leg.keys()):
    if leg >= len(fits_feb15):
        continue
    n0, D = fits_feb15[leg]["n0"], fits_feb15[leg]["D"]
    entries = sorted(by_leg[leg], key=lambda x: x[0])  # sort by sub

    per_leg.setdefault(leg, [])
    print(f"\nLeg {leg} segments:")

    for sub, path, fn in entries:
        with open(path, "rb") as f:
            trial = pickle.load(f)
        N_fit_diff = N_fit_diff_radius(r_centers, n0, D)        # [#/m^3 per m]
        N_gccn_m3  = np.trapz(N_fit_diff[gccn_mask], r_centers[gccn_mask])
        N_gccn_cm3 = N_gccn_m3 / 1e6
        precip = np.asarray(trial.get("surface precipitation", [])).squeeze()
        t      = np.asarray(trial.get("t", [])).squeeze()
        if precip.size >= 2 and t.size >= 2:
            order = np.argsort(t)
            t = t[order]; precip = precip[order]
            dt = float(np.median(np.diff(t)))
            total_precip_mm = float(np.sum(precip * dt))
        else:
            total_precip_mm = np.nan

        all_gccn.append(N_gccn_cm3)
        all_accum.append(total_precip_mm)
        per_leg[leg].append((N_gccn_cm3, total_precip_mm, sub))

        print(f"  {leg}_{sub}: GCCN(>1 µm D) = {N_gccn_cm3:.3e} cm⁻³   Rain = {total_precip_mm:.3e} mm")
print("\n=== Summary of GCCN and rainfall ===")
for i, (gccn, rain) in enumerate(zip(all_gccn, all_accum)):
    print(f"Leg {i+1:02d}: GCCN = {gccn:.3e} cm⁻³, Rain = {rain:.3e} mm")
for leg in sorted(per_leg.keys()):
    rows = per_leg[leg]
    if not rows:
        continue
    x_gccn, y_rain, subs = zip(*rows)
    x_gccn = np.asarray(x_gccn)
    y_rain = np.asarray(y_rain)

    plt.figure(figsize=(6, 4.5))
    plt.scatter(x_gccn, y_rain, s=60, edgecolor='k', linewidth=0.7)
    for gx, ry, s in rows:
        plt.annotate(f"{leg}_{s}", (gx, ry), textcoords="offset points", xytext=(5, 4), fontsize=8)

    plt.xscale('log'); plt.yscale('log')
    plt.xlabel("GCCN (cm$^{-3}$) for D > 1 µm", fontsize=12, fontweight="bold")
    plt.ylabel("Rain Accumulation (mm)", fontsize=12, fontweight="bold")
    plt.title(f"Leg {leg}: GCCN vs Rain (each point = {leg}_sub)", fontsize=12, fontweight="bold")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

# %%
#error bars
for leg in sorted(per_leg.keys()):
    rows = per_leg[leg]
    if not rows:
        continue
    x_gccn, y_rain, subs = zip(*rows)
    x_gccn = np.asarray(x_gccn)
    y_rain = np.asarray(y_rain)
    gccn_val = float(x_gccn[0])
    mean_rain = np.nanmean(y_rain)
    std_rain  = np.nanstd(y_rain, ddof=1)
    n_rain    = np.isfinite(y_rain).sum()
    se_rain   = std_rain / np.sqrt(n_rain) if n_rain > 0 else np.nan
    ci95_rain = 1.96 * se_rain

    plt.figure(figsize=(6, 4.5))
    plt.scatter(x_gccn, y_rain, s=60, edgecolor='k', linewidth=0.7, label="Ensembles")
    for gx, ry, s in rows:
        plt.annotate(f"{leg}_{s}", (gx, ry), textcoords="offset points", xytext=(5, 4), fontsize=8)
    plt.errorbar(gccn_val, mean_rain, yerr=std_rain, fmt='o', ms=8, color='crimson',
                 ecolor='crimson', elinewidth=2, capsize=4, label='Mean ± 1σ')
    plt.errorbar(gccn_val, mean_rain, yerr=ci95_rain, fmt='none', ecolor='crimson',
                 elinewidth=1, capsize=4, alpha=0.5, label='95% CI')

    plt.xscale('log'); plt.yscale('log')
    plt.xlabel("GCCN (cm$^{-3}$) for D > 1 µm", fontsize=12, fontweight="bold")
    plt.ylabel("Rain Accumulation (mm)", fontsize=12, fontweight="bold")
    plt.title(f"Leg {leg}: Ensemble Rain Variability\nMean ± 1σ (and 95% CI)", fontsize=12, fontweight="bold")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
    print(f"Leg {leg:02d}: mean rain = {mean_rain:.3e} mm, σ = {std_rain:.3e}, 95% CI = ±{ci95_rain:.3e}")

# %%
#boxplot
legs = sorted(per_leg.keys())
rain_data = [np.array([r for _, r, _ in per_leg[leg]]) for leg in legs]
gccn_vals = [per_leg[leg][0][0] for leg in legs]  # one GCCN per leg

plt.figure(figsize=(8,5))
plt.boxplot(rain_data, positions=np.arange(len(legs)), patch_artist=True,
            boxprops=dict(facecolor="skyblue", alpha=0.6),
            medianprops=dict(color="darkred", linewidth=1.5))
plt.xticks(np.arange(len(legs)), [f"{leg}\n({gccn:.1e})" for leg, gccn in zip(legs, gccn_vals)],
           rotation=45, ha="right")
plt.yscale("log")
plt.ylabel("Rain Accumulation (mm)", fontsize=12, fontweight="bold")
plt.xlabel("Main Leg (GCCN in cm$^{-3}$ for D > 1 µm)", fontsize=12, fontweight="bold")
plt.title("Rain Distribution Across Ensembles per Leg", fontsize=13, fontweight="bold")
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.tight_layout()
plt.show()

# %%
#signal to noise ratio 
legs = sorted(per_leg.keys())
means = []
stds = []
ns = []
groups = []

for leg in legs:
    rain_vals = np.array([r for _, r, _ in per_leg[leg]])
    means.append(np.nanmean(rain_vals))
    stds.append(np.nanstd(rain_vals, ddof=1))
    ns.append(np.isfinite(rain_vals).sum())
    groups.append(rain_vals[np.isfinite(rain_vals)])

means = np.array(means)
stds = np.array(stds)
ns = np.array(ns)
num = np.sum((ns - 1) * stds**2)
den = np.sum(ns - 1)
pooled_within_std = np.sqrt(num / den) if den > 0 else np.nan
std_between = np.nanstd(means, ddof=1)
SNR = std_between / pooled_within_std
print("\n=== Signal-to-Noise Ratio Analysis ===")
print(f"Between-leg std (signal): {std_between:.3e} mm")
print(f"Pooled within-leg std (noise): {pooled_within_std:.3e} mm")
print(f"SNR = {SNR:.2f}")

#Between-leg std (signal) the standard deviation of the mean rainfall across all main legs
# captures how different the mean rainfall is from leg to leg (i.e., the “GCCN effect”).

# Pooled within-leg std (noise) combines the ensemble spread within each leg
# measures the average “random noise” caused by stochastic variability across ensembles.

#SNR = std between-leg means / pooled within-leg std




# %%
