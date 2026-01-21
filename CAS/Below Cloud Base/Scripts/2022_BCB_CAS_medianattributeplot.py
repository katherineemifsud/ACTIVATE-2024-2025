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
# %%
# Median attribute table for BCB GCCN study Jan-Jun 2022 full data
months = [1, 2, 3, 5, 6]
month_name = {1:"Jan", 2:"Feb", 3:"Mar", 5:"May", 6:"Jun"}
months_colors = {
    1: "tab:blue",
    2: "tab:orange",
    3: "tab:green",
    5: "tab:red",
    6: "tab:purple"
}
med_wind = {1: 5.57, 2: 4.85, 3: 6.03, 5: 4.83, 6: 3.61}   
med_slope = {1: 0.87, 2: 0.77, 3: 0.78, 5: 0.53, 6: 0.68}    
med_mass  = {1: 7.07, 2: 7.62, 3: 11.57, 5: 2.20, 6: 9.56}      
med_gccn  = {1: 0.25, 2: 0.40, 3: 0.46, 5: 0.31, 6: 0.78}   

df_med = pd.DataFrame({
    "Wind (m/s)":   [med_wind[m] for m in months],
    "Slope (D)":    [med_slope[m] for m in months],
    "Mass (µg/m³)": [med_mass[m]  for m in months],
    "GCCN (cm⁻³)":  [med_gccn[m]  for m in months],
}, index=months)

df_med.index.name = "Month"
df_med
df_norm = (df_med - df_med.min()) / (df_med.max() - df_med.min())
markers = {
    "Wind (m/s)": "D",
    "Slope (D)": "s",
    "Mass (µg/m³)": "o",
    "GCCN (cm⁻³)": "^"
}
var_colors = {
    "Wind (m/s)": "tab:blue",
    "Slope (D)": "tab:orange",
    "Mass (µg/m³)": "tab:green",
    "GCCN (cm⁻³)": "tab:red"
}

x = np.arange(len(months))
xlab = [month_name[m] for m in months]

fig, ax = plt.subplots(figsize=(8.8, 5.4))

for col in df_norm.columns:
    ax.plot(
        x, df_norm[col].to_numpy(),
        marker=markers[col],
        linewidth=2.5,
        markersize=9,
        color=var_colors[col],
        label=col
    )

ax.set_xticks(x)
ax.set_xticklabels(xlab, fontsize=14, fontweight="bold")
ax.set_ylabel("Normalized monthly median (unitless)", fontsize=14, fontweight="bold")
ax.set_title("CAS BCB Monthly Medians (Normalized)\n January–June 2022", fontsize=16, fontweight="bold")
ax.grid(alpha=0.3)
plt.yticks(fontsize=14, fontweight="bold")
ax.legend(ncol=2, fontsize=11, frameon=True)
plt.tight_layout()
plt.show()

# %%
# Median attribute table for BCB GCCN study Jan-Jun 2022 less than 100 ug/m³ mass
months = [1, 2, 3, 5, 6]
month_name = {1:"Jan", 2:"Feb", 3:"Mar", 5:"May", 6:"Jun"}
months_colors = {
    1: "tab:blue",
    2: "tab:orange",
    3: "tab:green",
    5: "tab:red",
    6: "tab:purple"
}
med_wind = {1: 5.57, 2: 4.85, 3: 6.03, 5: 4.83, 6: 3.61}   
med_slope = {1: 0.85, 2: 0.77, 3: 0.77, 5: 0.53, 6: 0.68}    
med_mass  = {1: 6.01, 2: 7.52, 3: 10.90, 5: 2.20, 6: 9.56}      
med_gccn  = {1: 0.25, 2: 0.40, 3: 0.46, 5: 0.31, 6: 0.78}   

df_med = pd.DataFrame({
    "Wind (m/s)":   [med_wind[m] for m in months],
    "Slope (D)":    [med_slope[m] for m in months],
    "Mass (µg/m³)": [med_mass[m]  for m in months],
    "GCCN (cm⁻³)":  [med_gccn[m]  for m in months],
}, index=months)

df_med.index.name = "Month"
df_med
df_norm = (df_med - df_med.min()) / (df_med.max() - df_med.min())
markers = {
    "Wind (m/s)": "D",
    "Slope (D)": "s",
    "Mass (µg/m³)": "o",
    "GCCN (cm⁻³)": "^"
}
var_colors = {
    "Wind (m/s)": "tab:blue",
    "Slope (D)": "tab:orange",
    "Mass (µg/m³)": "tab:green",
    "GCCN (cm⁻³)": "tab:red"
}

x = np.arange(len(months))
xlab = [month_name[m] for m in months]

fig, ax = plt.subplots(figsize=(8.8, 5.4))

for col in df_norm.columns:
    ax.plot(
        x, df_norm[col].to_numpy(),
        marker=markers[col],
        linewidth=2.5,
        markersize=9,
        color=var_colors[col],
        label=col
    )

ax.set_xticks(x)
ax.set_xticklabels(xlab, fontsize=14, fontweight="bold")
ax.set_ylabel("Normalized monthly median (unitless)", fontsize=14, fontweight="bold")
ax.set_title("CAS BCB Monthly Medians (Normalized)\n (legs ≤ 100 µg/m³)\nJanuary–June 2022", fontsize=16, fontweight="bold")
ax.grid(alpha=0.3)
plt.yticks(fontsize=14, fontweight="bold")
ax.legend(ncol=2, fontsize=11, frameon=True)
plt.tight_layout()
plt.show()
# %%
# Mean attribute table for BCB GCCN study Jan-Jun 2022 full data
months = [1, 2, 3, 5, 6]
month_name = {1:"Jan", 2:"Feb", 3:"Mar", 5:"May", 6:"Jun"}
months_colors = {
    1: "tab:blue",
    2: "tab:orange",
    3: "tab:green",
    5: "tab:red",
    6: "tab:purple"
}
mean_wind = {1: 5.47, 2: 5.17, 3: 5.96, 5: 5.83, 6: 3.95}   
mean_slope = {1: 1.18, 2: 0.82, 3: 0.91, 5: 0.52, 6: 0.63}    
mean_mass  = {1: 65.27, 2: 14.03, 3: 20.75, 5: 4.84, 6: 9.70}      
mean_gccn  = {1: 0.31, 2: 0.47, 3: 0.65, 5: 0.62, 6: 0.78}   

df_mean = pd.DataFrame({
    "Wind (m/s)":   [mean_wind[m] for m in months],
    "Slope (D)":    [mean_slope[m] for m in months],
    "Mass (µg/m³)": [mean_mass[m]  for m in months],
    "GCCN (cm⁻³)":  [mean_gccn[m]  for m in months],
}, index=months)

df_mean.index.name = "Month"
df_mean
df_norm = (df_mean - df_mean.min()) / (df_mean.max() - df_mean.min())
markers = {
    "Wind (m/s)": "D",
    "Slope (D)": "s",
    "Mass (µg/m³)": "o",
    "GCCN (cm⁻³)": "^"
}
var_colors = {
    "Wind (m/s)": "tab:blue",
    "Slope (D)": "tab:orange",
    "Mass (µg/m³)": "tab:green",
    "GCCN (cm⁻³)": "tab:red"
}

x = np.arange(len(months))
xlab = [month_name[m] for m in months]

fig, ax = plt.subplots(figsize=(8.8, 5.4))

for col in df_norm.columns:
    ax.plot(
        x, df_norm[col].to_numpy(),
        marker=markers[col],
        linewidth=2.5,
        markersize=9,
        color=var_colors[col],
        label=col
    )

ax.set_xticks(x)
ax.set_xticklabels(xlab, fontsize=14, fontweight="bold")
ax.set_ylabel("Normalized monthly mean (unitless)", fontsize=14, fontweight="bold")
ax.set_title("CAS BCB Monthly Means (Normalized)\n January–June 2022", fontsize=16, fontweight="bold")
ax.grid(alpha=0.3)
plt.yticks(fontsize=14, fontweight="bold")
ax.legend(ncol=2, fontsize=11, frameon=True)
plt.tight_layout()
plt.show()
# %%
# Mean attribute table for BCB GCCN study Jan-Jun 2022 less than 100 ug/m³ mass
months = [1, 2, 3, 5, 6]
month_name = {1:"Jan", 2:"Feb", 3:"Mar", 5:"May", 6:"Jun"}
months_colors = {
    1: "tab:blue",
    2: "tab:orange",
    3: "tab:green",
    5: "tab:red",
    6: "tab:purple"
}
mean_wind = {1: 5.47, 2: 5.17, 3: 5.96, 5: 5.83, 6: 3.95}  
mean_slope = {1: 0.91, 2: 0.80, 3: 0.88, 5: 0.52, 6: 0.63}    
mean_mass  = {1: 9.02, 2: 8.98, 3: 15.48, 5: 4.84, 6: 9.70}      
mean_gccn  = {1: 0.31, 2: 0.47, 3: 0.65, 5: 0.62, 6: 0.78} 

df_mean = pd.DataFrame({
    "Wind (m/s)":   [mean_wind[m] for m in months],
    "Slope (D)":    [mean_slope[m] for m in months],
    "Mass (µg/m³)": [mean_mass[m]  for m in months],
    "GCCN (cm⁻³)":  [mean_gccn[m]  for m in months],
}, index=months)

df_mean.index.name = "Month"
df_mean
df_norm = (df_mean - df_mean.min()) / (df_mean.max() - df_mean.min())
markers = {
    "Wind (m/s)": "D",
    "Slope (D)": "s",
    "Mass (µg/m³)": "o",
    "GCCN (cm⁻³)": "^"
}
var_colors = {
    "Wind (m/s)": "tab:blue",
    "Slope (D)": "tab:orange",
    "Mass (µg/m³)": "tab:green",
    "GCCN (cm⁻³)": "tab:red"
}

x = np.arange(len(months))
xlab = [month_name[m] for m in months]

fig, ax = plt.subplots(figsize=(8.8, 5.4))

for col in df_norm.columns:
    ax.plot(
        x, df_norm[col].to_numpy(),
        marker=markers[col],
        linewidth=2.5,
        markersize=9,
        color=var_colors[col],
        label=col
    )

ax.set_xticks(x)
ax.set_xticklabels(xlab, fontsize=14, fontweight="bold")
ax.set_ylabel("Normalized monthly mean (unitless)", fontsize=14, fontweight="bold")
ax.set_title("CAS BCB Monthly Means (Normalized)\n (legs ≤ 100 µg/m³)\nJanuary–June 2022", fontsize=16, fontweight="bold")
ax.grid(alpha=0.3)
plt.yticks(fontsize=14, fontweight="bold")
ax.legend(ncol=2, fontsize=11, frameon=True)
plt.tight_layout()
plt.show()
# %%
