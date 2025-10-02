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
import glob
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
import matplotlib.colors as mcolors
import pickle
import os
import sys
#%%

sys.modules['numpy._core'] = np
sys.modules['numpy._core.multiarray'] = np.core.multiarray

data_dir = "/home/disk/eos4/kathem24/activate/data/CAS/Full size distributions/"
all_files = sorted(glob.glob(os.path.join(data_dir, "*.pickle")),
                   key=lambda p: int(os.path.basename(p).split(".")[0]))

results = [] 

for fname in all_files:
    with open(fname, "rb") as f:
        d = pickle.load(f)
    surf = d.get('surface precipitation', None)
    if isinstance(surf, np.ndarray):
        surf_shape = surf.shape
    elif isinstance(surf, dict):
        surf_shape = list(surf.keys())
    else:
        surf_shape = None

    dry = d['dry spectrum']
    dry_arr = np.asarray(dry).squeeze()
    mean_spectrum = dry_arr.mean(axis=0)

    print(f"{os.path.basename(fname)} -> "
          f"surf shape: {surf_shape}, dry spectrum shape: {dry_arr.shape}")
    results.append({
        "file": os.path.basename(fname),
        "mean_spectrum": mean_spectrum,
        "total_number_cm3": np.nansum(mean_spectrum)  
    })

# %%

#using Jason's spectrum of np.logspace (-7, -5, 101) and bin edges in radius. his units are 
#/m4
r_edges_m  = np.logspace(-7, -5, 101)
r_centers_m = np.sqrt(r_edges_m[:-1] * r_edges_m[1:])
dr_m       = np.diff(r_edges_m)
d_centers_um = 2.0 * r_centers_m * 1e6  # diameter [µm]

plt.figure(figsize=(7,4))

for fname in all_files:
    with open(fname, "rb") as f:
        d = pickle.load(f)

    dry_spec = np.asarray(d['dry spectrum']).squeeze()   # (time, bin)
    N_bins_m3 = dry_spec * dr_m[None, :]
    N_bins_cm3 = N_bins_m3 / 1e6
    mean_N_cm3 = np.nanmean(N_bins_cm3, axis=0)


    plt.semilogx(d_centers_um, mean_N_cm3,
             alpha=0.6,    # <-- was 0.2 (increase toward 1 for full opacity)
             lw=1.5)       # <-- was 0.8 (thicker line)


plt.xlabel('Dry Diameter (µm)', fontsize=17, fontweight='bold')
plt.ylabel('Number Concentration \nper bin (cm$^{-3}$)', fontsize=17, fontweight='bold')
plt.title('January– June\nBelow Cloud Base\nAll Dry Distributions', fontsize=17, fontweight='bold')
plt.yscale('log')
plt.yticks(fontweight='bold', fontsize=17)
plt.xticks(fontweight='bold', fontsize=17)
plt.tight_layout()
plt.show()

#%%
gccn_threshold = 2.0  # µm
gccn_mask = d_centers_um >= gccn_threshold

gccn_totals = []      # cm^-3
totals_all  = []      # cm^-3

for fname in all_files:
    with open(fname, "rb") as f:
        d = pickle.load(f)
    dry_spec = np.asarray(d['dry spectrum']).squeeze()
    N_bins_m3 = dry_spec * dr_m[None, :]
    N_bins_cm3 = N_bins_m3 / 1e6
    mean_N_cm3 = np.nanmean(N_bins_cm3, axis=0)

    totals_all.append(np.nansum(mean_N_cm3))
    gccn_totals.append(np.nansum(mean_N_cm3[gccn_mask]))
print("len(totals_all) =", len(totals_all))
print("First 10 totals:", totals_all[:10])
print("Last 10 totals :", totals_all[-10:])
print("min / max:", np.nanmin(totals_all), np.nanmax(totals_all))

print(f"Mean total number conc (all bins): {np.mean(totals_all):.3f} cm^-3")
print(f"Mean GCCN conc (≥{gccn_threshold} µm): {np.mean(gccn_totals):.4e} cm^-3")
print(f"Fraction of total in GCCN: {np.mean(np.array(gccn_totals)/np.array(totals_all)):.4%}")


# %%
#only greater than 2um d
r_edges_m   = np.logspace(-7, -5, 101)
r_centers_m = np.sqrt(r_edges_m[:-1] * r_edges_m[1:])
dr_m        = np.diff(r_edges_m)
d_centers_um = 2.0 * r_centers_m * 1e6   # diameter [µm]

mask = d_centers_um >= 2.0
d_gccn = d_centers_um[mask]

plt.figure(figsize=(7,4))

for fname in all_files:
    with open(fname, "rb") as f:
        d = pickle.load(f)

    dry_spec = np.asarray(d['dry spectrum']).squeeze()
    N_bins_m3 = dry_spec * dr_m[None, :]
    N_bins_cm3 = N_bins_m3 / 1e6
    mean_N_cm3 = np.nanmean(N_bins_cm3, axis=0)

    plt.semilogx(d_gccn,
             mean_N_cm3[mask],
             alpha=0.6,   # higher opacity (was 0.2)
             lw=1.5)      # thicker line (was 0.8)

plt.xlabel('Dry Diameter (µm)', fontsize=16, fontweight='bold')
plt.ylabel('Number Concentration \nper bin (cm$^{-3}$)', fontsize=16, fontweight='bold')
plt.title('CAS Below Cloud Base\nAll Dry Distributions\n>2um diameter', fontsize=17, fontweight='bold')
plt.yscale('log')
plt.xlim(2, d_centers_um.max())   # explicitly set min x to 2 µm
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.grid(True, which='both', ls='--', alpha=0.4)
plt.tight_layout()
plt.show()

# %%

gccn_threshold = 2.0  # µm
mask = d_centers_um >= gccn_threshold

gccn_totals = []   # to collect GCCN total conc for each file

for fname in all_files:
    with open(fname, "rb") as f:
        d = pickle.load(f)

    dry_spec = np.asarray(d['dry spectrum']).squeeze()      # (time, bins)
    N_bins_m3 = dry_spec * dr_m[None, :]                    # integrate over radius
    N_bins_cm3 = N_bins_m3 / 1e6                             # convert to cm^-3
    mean_N_cm3 = np.nanmean(N_bins_cm3, axis=0)              # time mean per bin

    gccn_conc = np.nansum(mean_N_cm3[mask])                  # total ≥2 µm
    gccn_totals.append(gccn_conc)

    print(f"{os.path.basename(fname):>8s} : {gccn_conc:8.4f} cm^-3")
    gccn_arr = np.array(gccn_totals)
    print("\nSummary for ≥2 µm dry diameter:")
    print(f"Mean   : {gccn_arr.mean():.4f} cm^-3")
    print(f"Std dev: {gccn_arr.std(ddof=1):.4f} cm^-3")
   

# %%

gccn_threshold = 2.0  # µm
mask = d_centers_um >= gccn_threshold

rows = []  

for fname in all_files:
    with open(fname, "rb") as f:
        d = pickle.load(f)

    dry_spec = np.asarray(d['dry spectrum']).squeeze()
    N_bins_m3 = dry_spec * dr_m[None, :]
    N_bins_cm3 = N_bins_m3 / 1e6
    mean_N_cm3 = np.nanmean(N_bins_cm3, axis=0)
    total_conc = np.nansum(mean_N_cm3)
    gccn_conc = np.nansum(mean_N_cm3[mask])
    gccn_std  = np.nanstd(mean_N_cm3[mask], ddof=1)

    rows.append({
        "File name"                                : os.path.basename(fname),
        "Mean concentration ≥2 µm (cm^-3)"         : gccn_conc,
        "Std dev in ≥2 µm bins (cm^-3)"            : gccn_std,
        "Total concentration (all bins, cm^-3)"    : total_conc
    })
df = pd.DataFrame(rows)
# df.to_csv("gccn_summary.csv", index=False)

# print("Saved CSV with", len(df), "rows to gccn_summary.csv")

# %%
plt.hist(df["Mean concentration ≥2 µm (cm^-3)"], bins=30)
plt.yscale('log')
plt.title('CAS Below Cloud Base \n Total GCCN Concentration (≥2 µm)', fontsize=17, fontweight='bold')
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.xlabel("GCCN concentration (cm$^{-3}$)\n >= 2 µm", fontsize=17, fontweight='bold')
plt.ylabel("Number of legs", fontsize=17, fontweight='bold')

# %%
rho_w = 1000.0 
dt     = 10.0   

rows = [] 

for fname in all_files:
    with open(fname, "rb") as f:
        d = pickle.load(f)

    # pull arrays
    z    = np.asarray(d['z'])
    rho  = np.asarray(d['rhod'])                  # (nz, nt)
    q_r  = np.asarray(d['rain water mixing ratio'])  # g/kg or kg/kg
    v_r  = np.asarray(d['rain averaged terminal velocity'])  # m/s
    surf = np.asarray(d['surface precipitation'])   # model diagnostic [mm/s]
    if np.nanmax(q_r) < 0.5:   # heuristic: probably g/kg
        q_r = q_r / 1000.0
    z_prof = z[:,0] if z.ndim == 2 else z
    sfc_idx = int(np.where(z_prof >= 0)[0][0])
    rain_mm_s = (rho[sfc_idx,:] * q_r[sfc_idx,:] * v_r[sfc_idx,:]) / rho_w * 1000.0
    our_total_mm   = np.nansum(rain_mm_s) * dt
    model_total_mm = np.nansum(surf) * dt

    rows.append({
        "File"           : os.path.basename(fname),
        "Our_total_mm"   : our_total_mm,
        "Model_total_mm" : model_total_mm,
        "Ratio(our/model)": (our_total_mm/model_total_mm
                             if model_total_mm>0 else np.nan)
    })

df_precip = pd.DataFrame(rows)

# # Save to CSV for easy inspection
# df_precip.to_csv("surface_precip_summary.csv", index=False)

# # Quick summary
# print(df_precip.describe())
# print("Saved surface_precip_summary.csv with",
#       len(df_precip), "rows")

# %%
