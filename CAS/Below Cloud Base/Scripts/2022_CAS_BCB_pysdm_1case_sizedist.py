#%%
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import datetime
import pathlib
import statistics
import mputil
import shutil
import glob
import os
import re
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
#%%
import sys
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

# %%
dry_spec = trial_data['dry spectrum']
print(type(dry_spec))
print(getattr(dry_spec, 'shape', None))

# %%

dry_spec = trial_data['dry spectrum'].squeeze() 
print(dry_spec.shape)

# %%
mean_spectrum = dry_spec.mean(axis=0)   # → shape (100,)

# %%

#    0.01–10 µm covers accumulation through giant CCN
bin_centers = np.logspace(np.log10(0.01), np.log10(10), 100)  # shape (100,)
dry_spec = trial_data['dry spectrum'].squeeze()  # shape (59, 100)
mean_spectrum = dry_spec.mean(axis=0)
plt.figure(figsize=(6,4))
plt.semilogx(bin_centers, mean_spectrum, color='steelblue')
plt.xlabel('Dry Diameter (µm)')
plt.ylabel('Number Concentration (cm$^{-3}$)')
plt.title('Temporary GCCN Dry Size Distribution')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.show()
gccn_mask = bin_centers >= 2.0
gccn_conc = mean_spectrum[gccn_mask].sum()
print("Estimated GCCN concentration (temporary grid):", gccn_conc)

# %%
dry_spec = trial_data['dry spectrum'].squeeze()
print("Min/Max of spectrum:", dry_spec.min(), dry_spec.max())
print("Fraction > 0:", np.count_nonzero(dry_spec)/dry_spec.size)
mean_spectrum = dry_spec.mean(axis=0)
print("Any bins non-zero at high index positions?:",
      np.any(mean_spectrum[-10:] > 0))

# %%
