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
import matplotlib.colors as mcolors
import pickle
import sys
#%%
sys.modules['numpy._core'] = np
sys.modules['numpy._core.multiarray'] = np.core.multiarray

trial_case = "/home/disk/eos4/kathem24/activate/data/CAS/Full size distributions/0.pickle"
with open(trial_case, "rb") as f:
    trial_data = pickle.load(f)

print("Trial Data Type:", type(trial_data))
if isinstance(trial_data, dict):
    print("Trial Data Keys:", trial_data.keys())
# %%
trial_rain = trial_data['surface precipitation']

print("Trial Rain Type:", type(trial_rain))

if isinstance(trial_rain, np.ndarray):
    print("Trial Rain Shape:", trial_rain.shape)

elif isinstance(trial_rain, dict):
    print("Keys inside 'surface precipitation':", trial_rain.keys())

dry_spec = trial_data['dry spectrum']
print(type(dry_spec))
print(getattr(dry_spec, 'shape', None))
dry_spec = trial_data['dry spectrum'].squeeze() 
print(dry_spec.shape)
mean_spectrum = dry_spec.mean(axis=0)  

# %%

#using Jason's spectrum of np.logspace (-7, -5, 101) and bin edges in radius. his units are 
#/m4

r_edges_m = np.logspace(-7, -5, 101)                    
r_centers_m = np.sqrt(r_edges_m[:-1] * r_edges_m[1:])  
dr_m = np.diff(r_edges_m)                              

# Convert to DIAMETER in microns for plotting and GCCN thresholding
d_centers_um = 2.0 * r_centers_m * 1e6                  # diameter [µm]

#Convert spectrum (1/m^4) to per-bin number conc (m^-3), then to cm^-3 ----
N_bins_m3 = dry_spec * dr_m[None, :]                    
N_bins_cm3 = N_bins_m3 / 1e6
mean_N_cm3 = np.nanmean(N_bins_cm3, axis=0)          

plt.figure(figsize=(7,4))
plt.semilogx(d_centers_um, mean_N_cm3)
plt.xlabel('Dry Diameter (µm)')
plt.ylabel('Number Concentration per bin (cm$^{-3}$)')
plt.title('January 11, 2022\nLeg 1\n Dry Size Distribution')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.show()

#%%

# GCCN total (example: diameter ≥ 2 µm) ----
gccn_mask = d_centers_um >= 2.0
gccn_conc_cm3 = np.nansum(mean_N_cm3[gccn_mask])        # cm^-3
print(f"Estimated GCCN concentration (≥2 µm): {gccn_conc_cm3:.3e} cm^-3")
print("Any nonzero in largest 10 bins?:", np.any(mean_N_cm3[-10:] > 0))
print("Total number conc (all bins):", np.nansum(mean_N_cm3), "cm^-3")

# %%
for d, n in zip(d_centers_um, mean_N_cm3):
    print(f"{d:6.3f} µm : {n:8.3f} cm^-3 per bin")

# %%
mean_N = mean_N_cm3
nonzero_bins = np.where(mean_N > 0)[0]
print("Largest occupied bin index:", nonzero_bins.max())
print("Diameter at that bin:", d_centers_um[nonzero_bins.max()])

# %%

r_edges = np.logspace(-7, -5, 101) 

print("First 5 edges:", r_edges[:5])
print("Last 5 edges:",  r_edges[-5:])
print("Min edge:", r_edges.min(), " Max edge:", r_edges.max())

# %%
# %%
