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
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy.ma as ma
import glob
import os
import re
import pickle
import math
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from scipy.optimize import curve_fit
import seaborn as sns
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
import pickle
from scipy.integrate import quad
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D
from collections import Counter
from scipy.spatial import distance
# %%
load_path = "/home/disk/eos4/kathem24/activate/data/2020/CDP/1Hz/CDP_FMAS2020_wind_bin_concentration_plot_data.pkl"
with open(load_path, "rb") as f:
    plot_data = pickle.load(f)

windspeed_values = plot_data["windspeed_values"]
total_concentrations = plot_data["total_concentrations"]
standard_errors = plot_data["standard_errors_2SE"]
counting_errors_CAS = plot_data["counting_errors_CAS_2sigma"]

m_fit = plot_data["m_fit"]
b_fit = plot_data["b_fit"]
m_err = plot_data["m_err"]
b_err = plot_data["b_err"]
r_squared = plot_data["r_squared"]
r_value = plot_data["r_value"]
x_fit = plot_data["x_fit"]
y_fit = plot_data["y_fit"]
plt.figure(figsize=(8, 6))

plt.errorbar(
    windspeed_values, total_concentrations,
    yerr=standard_errors,
    fmt='o', color='black',
    ecolor='black', elinewidth=1.5,
    capsize=5, capthick=2,
    label="±2 Standard Errors",
    zorder=3
)

plt.errorbar(
    windspeed_values - 0.2, total_concentrations,
    yerr=counting_errors_CAS,
    fmt='s',
    markersize=4,
    markerfacecolor='#8c510a',
    markeredgecolor='black',
    ecolor='#8c510a',
    elinewidth=4,
    capsize=8,
    capthick=3,
    label="±2σ CAS Instrument Error",
    zorder=2
)

plt.plot(
    x_fit, y_fit, '-',
    color='black', linewidth=2.5,
    label=f'Fit: y = ({m_fit:.3f}±{m_err:.3f})x + {b_fit:.3f}\nR² = {r_squared:.2f}, R = {r_value:.2f}'
)

plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=20, fontweight='bold')
plt.ylabel("Total Wind Speed Bin \nConcentration (cm$^{-3}$)", fontsize=20, fontweight='bold')
plt.title("CDP Below Cloud Base \n FMAS 2020", fontsize=20, fontweight='bold')
plt.legend(fontsize=13, title_fontsize=14, loc='upper left', frameon=False)
plt.tight_layout()
plt.xlim(0, 12)
plt.ylim(0, 0.3)
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')
plt.show()
#%%
load_path = "/home/disk/eos4/kathem24/activate/data/2020/CDP/1Hz/CDP_FMAS2020_wind_bin_mass_plot_data.pkl"

with open(load_path, "rb") as f:
    mass_plot_data = pickle.load(f)

x = mass_plot_data["x"]
y = mass_plot_data["y"]
yerr = mass_plot_data["yerr_2SE"]
count_err_for_points = mass_plot_data["count_err_for_points_2sigma"]

m_fit = mass_plot_data["m_fit"]
b_fit = mass_plot_data["b_fit"]
m_err = mass_plot_data["m_err"]
b_err = mass_plot_data["b_err"]
r2 = mass_plot_data["r2"]
R = mass_plot_data["R"]
x_fit = mass_plot_data["x_fit"]
y_fit = mass_plot_data["y_fit"]
plt.figure(figsize=(8,6))

plt.errorbar(
    x, y, yerr=yerr,
    fmt='o', color='black',
    ecolor='black', elinewidth=1.5,
    capsize=5, capthick=2,
    label="±2 Standard Errors",
    zorder=3
)

offset = 0.35

plt.errorbar(
    x + offset, y,
    yerr=count_err_for_points,
    fmt='o', color='#8c510a',
    ecolor='black', elinewidth=1.5,
    capsize=5, capthick=2,
    label="±2σ CDP Counting Error",
    zorder=2
)

plt.plot(
    x_fit, y_fit, '-',
    color='black', linewidth=2.5,
    label=f"Fit: y = ({m_fit:.3f}±{m_err:.3f})x + {b_fit:.3f}\nR² = {r2:.2f}, R = {R:.2f}"
)

plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=20, fontweight='bold')
plt.ylabel("Total Dry Mass (µg/m³)", fontsize=20, fontweight='bold')
plt.title("CDP Below Cloud Base\nFMAS 2020", fontsize=20, fontweight='bold')
plt.legend(fontsize=11, frameon=False, loc='upper left')
plt.xlim(0, 12)
plt.ylim(2, 45)
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()
#%%

load_path = "/home/disk/eos4/kathem24/activate/data/2020/CAS/CAS_FMAS2020_wind_bin_mass_plot_data.pkl"

with open(load_path, "rb") as f:
    mass_plot_data = pickle.load(f)

x = mass_plot_data["x"]
y = mass_plot_data["y"]
yerr = mass_plot_data["yerr_2SE"]
count_err_for_points = mass_plot_data["count_err_for_points_2sigma"]

m_fit = mass_plot_data["m_fit"]
b_fit = mass_plot_data["b_fit"]
m_err = mass_plot_data["m_err"]
b_err = mass_plot_data["b_err"]
r2 = mass_plot_data["r2"]
R = mass_plot_data["R"]
x_fit = mass_plot_data["x_fit"]
y_fit = mass_plot_data["y_fit"]
plt.figure(figsize=(8,6))

plt.errorbar(
    x, y, yerr=yerr,
    fmt='o', color='black',
    ecolor='black', elinewidth=1.5,
    capsize=5, capthick=2,
    label="±2 Standard Errors",
    zorder=3
)

offset = 0.35

plt.errorbar(
    x + offset, y,
    yerr=count_err_for_points,
    fmt='o', color='#8c510a',
    ecolor='black', elinewidth=1.5,
    capsize=5, capthick=2,
    label="±2σ CAS Counting Error",
    zorder=2
)

plt.plot(
    x_fit, y_fit, '-',
    color='black', linewidth=2.5,
    label=f"Fit: y = ({m_fit:.3f}±{m_err:.3f})x + {b_fit:.3f}\nR² = {r2:.2f}, R = {R:.2f}"
)

plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=20, fontweight='bold')
plt.ylabel("Total Dry Mass (µg/m³)", fontsize=20, fontweight='bold')
plt.title("CAS Below Cloud Base\nFMAS 2020", fontsize=20, fontweight='bold')
plt.legend(fontsize=11, frameon=False, loc='upper left')
plt.xlim(0, 12)
plt.ylim(0, 31)
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
load_path = "/home/disk/eos4/kathem24/activate/data/2020/CAS/CAS_FMAS2020_wind_bin_concentration_plot_data.pkl"
with open(load_path, "rb") as f:
    plot_data = pickle.load(f)

windspeed_values = plot_data["windspeed_values"]
total_concentrations = plot_data["total_concentrations"]
standard_errors = plot_data["standard_errors_2SE"]
counting_errors_CAS = plot_data["counting_errors_CAS_2sigma"]

m_fit = plot_data["m_fit"]
b_fit = plot_data["b_fit"]
m_err = plot_data["m_err"]
b_err = plot_data["b_err"]
r_squared = plot_data["r_squared"]
r_value = plot_data["r_value"]
x_fit = plot_data["x_fit"]
y_fit = plot_data["y_fit"]
plt.figure(figsize=(8, 6))

plt.errorbar(
    windspeed_values, total_concentrations,
    yerr=standard_errors,
    fmt='o', color='black',
    ecolor='black', elinewidth=1.5,
    capsize=5, capthick=2,
    label="±2 Standard Errors",
    zorder=3
)

plt.errorbar(
    windspeed_values - 0.2, total_concentrations,
    yerr=counting_errors_CAS,
    fmt='s',
    markersize=4,
    markerfacecolor='#8c510a',
    markeredgecolor='black',
    ecolor='#8c510a',
    elinewidth=4,
    capsize=8,
    capthick=3,
    label="±2σ CAS Instrument Error",
    zorder=2
)

plt.plot(
    x_fit, y_fit, '-',
    color='black', linewidth=2.5,
    label=f'Fit: y = ({m_fit:.3f}±{m_err:.3f})x + {b_fit:.3f}\nR² = {r_squared:.2f}, R = {r_value:.2f}'
)

plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=20, fontweight='bold')
plt.ylabel("Total Wind Speed Bin \nConcentration (cm$^{-3}$)", fontsize=20, fontweight='bold')
plt.title("CAS Below Cloud Base \n FMAS 2020", fontsize=20, fontweight='bold')
plt.legend(fontsize=13, title_fontsize=14, loc='upper left', frameon=False)
plt.tight_layout()
plt.xlim(0, 12)
plt.ylim(0, 1)
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')
plt.show()
# %%
#mass 
cas_path = "/home/disk/eos4/kathem24/activate/data/2020/CAS/CAS_FMAS2020_wind_bin_mass_plot_data.pkl"
with open(cas_path, "rb") as f:
    cas = pickle.load(f)
cdp_path = "/home/disk/eos4/kathem24/activate/data/2020/CDP/1Hz/CDP_FMAS2020_wind_bin_mass_plot_data.pkl"

with open(cdp_path, "rb") as f:
    cdp = pickle.load(f)

plt.figure(figsize=(12, 6))
box_width = 0.6
plt.bar(
    cas["x"],
    2 * cas["yerr_2SE"],
    bottom=cas["y"] - cas["yerr_2SE"],
    width=box_width,
    color="gray",
    edgecolor="black",
    alpha=0.6,
    label="CAS ±2 SEM (~95% CI)",
    zorder=1
)
plt.bar(
    cdp["x"],
    2 * cdp["yerr_2SE"],
    bottom=cdp["y"] - cdp["yerr_2SE"],
    width=box_width,
    color="lightblue",
    edgecolor="blue",
    alpha=0.5,
    label="CDP ±2 SEM (~95% CI)",
    zorder=2
)
plt.errorbar(
    cas["x"],
    cas["y"],
    yerr=cas["count_err_for_points_2sigma"],
    fmt="none",
    ecolor="black",
    elinewidth=3,
    capsize=0,
    label="CAS ±2 SEM Instrument Error in total (both)",
    zorder=5
)
plt.errorbar(
    cdp["x"],
    cdp["y"],
    yerr=cdp["count_err_for_points_2sigma"],
    fmt="none",
    ecolor="blue",
    elinewidth=3,
    capsize=0,
    label="CDP ±2 SEM Instrument Error in total (both)",
    zorder=5
)
plt.plot(
    cas["x"],
    cas["y"],
    "o",
    color="black",
    label="CAS Mean",
    zorder=6
)
plt.plot(
    cdp["x"],
    cdp["y"],
    "o",
    color="blue",
    label="CDP Mean",
    zorder=6
)
plt.plot(
    cas["x_fit"],
    cas["y_fit"],
    "-",
    color="black",
    linewidth=2.5,
    zorder=4
)
plt.plot(
    cdp["x_fit"],
    cdp["y_fit"],
    "-",
    color="blue",
    linewidth=2.5,
    zorder=4
)
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=20, fontweight="bold")
plt.ylabel("Wind Speed Binned\nMass (µg m$^{-3}$)", fontsize=20, fontweight="bold")
plt.title("GCCN Mass as a function of wind speed\n FMAS 2020", fontsize=19, fontweight="bold")

plt.xlim(0, 12)
plt.ylim(0, 50)
plt.xticks(fontsize=18, fontweight="bold")
plt.yticks(fontsize=18, fontweight="bold")
plt.legend(fontsize=12, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
plt.tight_layout()
plt.show()
# %%
#CAS, CDP, and FCDPc oncentration
cas_path = "/home/disk/eos4/kathem24/activate/data/2020/CAS/CAS_FMAS2020_wind_bin_concentration_plot_data.pkl"
with open(cas_path, "rb") as f:
    cas = pickle.load(f)
cdp_path = "/home/disk/eos4/kathem24/activate/data/2020/CDP/1Hz/CDP_FMAS2020_wind_bin_concentration_plot_data.pkl"
with open(cdp_path, "rb") as f:
    cdp = pickle.load(f)
plt.figure(figsize=(11, 6))
box_width = 0.6
plt.bar(
    cas["windspeed_values"],
    2 * cas["standard_errors_2SE"],
    bottom=cas["total_concentrations"] - cas["standard_errors_2SE"],
    width=box_width,
    color="gray",
    edgecolor="black",
    alpha=0.6,
    label="CAS ±2 SEM (~95% CI)",
    zorder=1
)
plt.bar(
    cdp["windspeed_values"],
    2 * cdp["standard_errors_2SE"],
    bottom=cdp["total_concentrations"] - cdp["standard_errors_2SE"],
    width=box_width,
    color="lightblue",
    edgecolor="blue",
    alpha=0.5,
    label="CDP ±2 SEM (~95% CI)",
    zorder=2
)
plt.errorbar(
    cas["windspeed_values"],
    cas["total_concentrations"],
    yerr=cas["counting_errors_CAS_2sigma"],
    fmt="none",
    ecolor="black",
    elinewidth=3,
    capsize=0,
    label="CAS ±2 SEM Instrument Error in total (both)",
    zorder=5
)
plt.errorbar(
    cdp["windspeed_values"],
    cdp["total_concentrations"],
    yerr=cdp["counting_errors_CAS_2sigma"],
    fmt="none",
    ecolor="blue",
    elinewidth=3,
    capsize=0,
    label="CDP ±2 SEM Instrument Error in total (both)",
    zorder=5
)
plt.plot(
    cas["windspeed_values"],
    cas["total_concentrations"],
    "o",
    color="black",
    label="CAS Mean",
    zorder=6
)
plt.plot(
    cdp["windspeed_values"],
    cdp["total_concentrations"],
    "o",
    color="blue",
    label="CDP Mean",
    zorder=6
)
plt.plot(
    cas["x_fit"],
    cas["y_fit"],
    "-",
    color="black",
    linewidth=2.5,
    zorder=4
)
plt.plot(
    cdp["x_fit"],
    cdp["y_fit"],
    "-",
    color="blue",
    linewidth=2.5,
    zorder=4
)
plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=20, fontweight="bold")
plt.ylabel("Wind Speed Binned\nConcentration (cm$^{-3}$)", fontsize=20, fontweight="bold")
plt.title("GCCN Concentration as a function of wind speed\nFMAS 2020", fontsize=20, fontweight="bold")
plt.xlim(0, 12)
plt.ylim(0, 0.9)
plt.xticks(fontsize=18, fontweight="bold")
plt.yticks(fontsize=18, fontweight="bold")
plt.legend(fontsize=12, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
plt.tight_layout()
plt.show()
#%%
load_path = "/home/disk/eos4/kathem24/activate/data/2020/FCDP/FCDP_FMAS2020_wind_bin_mass_plot_data.pkl"
with open(load_path, "rb") as f:
    mass_plot_data = pickle.load(f)

x = mass_plot_data["x"]
y = mass_plot_data["y"]
yerr = mass_plot_data["yerr_2SE"]
count_err_for_points = mass_plot_data["count_err_for_points_2sigma"]

m_fit = mass_plot_data["m_fit"]
b_fit = mass_plot_data["b_fit"]
m_err = mass_plot_data["m_err"]
b_err = mass_plot_data["b_err"]
r2 = mass_plot_data["r2"]
R = mass_plot_data["R"]
x_fit = mass_plot_data["x_fit"]
y_fit = mass_plot_data["y_fit"]
plt.figure(figsize=(8,6))

plt.errorbar(
    x, y, yerr=yerr,
    fmt='o', color='black',
    ecolor='black', elinewidth=1.5,
    capsize=5, capthick=2,
    label="±2 Standard Errors",
    zorder=3
)

offset = 0.35

plt.errorbar(
    x + offset, y,
    yerr=count_err_for_points,
    fmt='o', color='#8c510a',
    ecolor='black', elinewidth=1.5,
    capsize=5, capthick=2,
    label="±2σ CDP Counting Error",
    zorder=2
)

plt.plot(
    x_fit, y_fit, '-',
    color='black', linewidth=2.5,
    label=f"Fit: y = ({m_fit:.3f}±{m_err:.3f})x + {b_fit:.3f}\nR² = {r2:.2f}, R = {R:.2f}"
)

plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=20, fontweight='bold')
plt.ylabel("Total Dry Mass (µg/m³)", fontsize=20, fontweight='bold')
plt.title("FCDP Below Cloud Base\nFMAS 2020", fontsize=20, fontweight='bold')
plt.legend(fontsize=11, frameon=False, loc='upper left')
plt.xlim(0, 12)
plt.ylim(0, 45)
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()
#%%
load_path = "/home/disk/eos4/kathem24/activate/data/2020/FCDP/FCDP_FMAS2020_wind_bin_concentration_plot_data.pkl"
with open(load_path, "rb") as f:
    plot_data = pickle.load(f)

windspeed_values = plot_data["windspeed_values"]
total_concentrations = plot_data["total_concentrations"]
standard_errors = plot_data["standard_errors_2SE"]
counting_errors_CAS = plot_data["counting_errors_CAS_2sigma"]

m_fit = plot_data["m_fit"]
b_fit = plot_data["b_fit"]
m_err = plot_data["m_err"]
b_err = plot_data["b_err"]
r_squared = plot_data["r_squared"]
r_value = plot_data["r_value"]
x_fit = plot_data["x_fit"]
y_fit = plot_data["y_fit"]
plt.figure(figsize=(8, 6))

plt.errorbar(
    windspeed_values, total_concentrations,
    yerr=standard_errors,
    fmt='o', color='black',
    ecolor='black', elinewidth=1.5,
    capsize=5, capthick=2,
    label="±2 Standard Errors",
    zorder=3
)

plt.errorbar(
    windspeed_values - 0.2, total_concentrations,
    yerr=counting_errors_CAS,
    fmt='s',
    markersize=4,
    markerfacecolor='#8c510a',
    markeredgecolor='black',
    ecolor='#8c510a',
    elinewidth=4,
    capsize=8,
    capthick=3,
    label="±2σ CAS Instrument Error",
    zorder=2
)

plt.plot(
    x_fit, y_fit, '-',
    color='black', linewidth=2.5,
    label=f'Fit: y = ({m_fit:.3f}±{m_err:.3f})x + {b_fit:.3f}\nR² = {r_squared:.2f}, R = {r_value:.2f}'
)

plt.xlabel("Wind Speed (m s$^{-1}$)", fontsize=20, fontweight='bold')
plt.ylabel("Total Wind Speed Bin \nConcentration (cm$^{-3}$)", fontsize=20, fontweight='bold')
plt.title("FCDP Below Cloud Base \n FMAS 2020", fontsize=20, fontweight='bold')
plt.legend(fontsize=13, title_fontsize=14, loc='upper left', frameon=False)
plt.tight_layout()
plt.xlim(0, 12)
plt.ylim(0, 1)
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')
plt.show()
# %%
cas_path = "/home/disk/eos4/kathem24/activate/data/2020/CAS/CAS_FMAS2020_wind_bin_concentration_plot_data.pkl"
with open(cas_path, "rb") as f:
    cas = pickle.load(f)

cdp_path = "/home/disk/eos4/kathem24/activate/data/2020/CDP/1Hz/CDP_FMAS2020_wind_bin_concentration_plot_data.pkl"
with open(cdp_path, "rb") as f:
    cdp = pickle.load(f)

fcdp_path = "/home/disk/eos4/kathem24/activate/data/2020/FCDP/FCDP_FMAS2020_wind_bin_concentration_plot_data.pkl"
with open(fcdp_path, "rb") as f:
    fcdp = pickle.load(f)
plt.figure(figsize=(11, 6))
box_width = 0.6
plt.bar(
    cas["windspeed_values"],
    2 * cas["standard_errors_2SE"],
    bottom=cas["total_concentrations"] - cas["standard_errors_2SE"],
    width=box_width,
    color="gray",
    edgecolor="black",
    alpha=0.6,
    label="CAS ±2 SEM (~95% CI)",
    zorder=1
)
plt.bar(
    cdp["windspeed_values"],
    2 * cdp["standard_errors_2SE"],
    bottom=cdp["total_concentrations"] - cdp["standard_errors_2SE"],
    width=box_width,
    color="lightblue",
    edgecolor="blue",
    alpha=0.5,
    label="CDP ±2 SEM (~95% CI)",
    zorder=2
)
plt.bar(
    fcdp["windspeed_values"],
    2 * fcdp["standard_errors_2SE"],
    bottom=fcdp["total_concentrations"] - fcdp["standard_errors_2SE"],
    width=box_width,
    color="mistyrose",
    edgecolor="red",
    alpha=0.5,
    label="FCDP ±2 SEM (~95% CI)",
    zorder=3
)
plt.errorbar(
    cas["windspeed_values"],
    cas["total_concentrations"],
    yerr=cas["counting_errors_CAS_2sigma"],
    fmt="none",
    ecolor="black",
    elinewidth=3,
    capsize=0,
    label="CAS ±2σ Instrument Error",
    zorder=5
)

plt.errorbar(
    cdp["windspeed_values"],
    cdp["total_concentrations"],
    yerr=cdp["counting_errors_CAS_2sigma"],
    fmt="none",
    ecolor="blue",
    elinewidth=3,
    capsize=0,
    label="CDP ±2σ Instrument Error",
    zorder=5
)

plt.errorbar(
    fcdp["windspeed_values"],
    fcdp["total_concentrations"],
    yerr=fcdp["counting_errors_CAS_2sigma"],
    fmt="none",
    ecolor="red",
    elinewidth=3,
    capsize=0,
    label="FCDP ±2σ Instrument Error",
    zorder=5
)
plt.plot(
    cas["windspeed_values"],
    cas["total_concentrations"],
    "o",
    color="black",
    label="CAS Mean",
    zorder=6
)
plt.plot(
    cdp["windspeed_values"],
    cdp["total_concentrations"],
    "o",
    color="blue",
    label="CDP Mean",
    zorder=6
)

plt.plot(
    fcdp["windspeed_values"],
    fcdp["total_concentrations"],
    "o",
    color="red",
    label="FCDP Mean",
    zorder=6
)
plt.plot(
    cas["x_fit"],
    cas["y_fit"],
    "-",
    color="black",
    linewidth=2.5,
    zorder=4
)
plt.plot(
    cdp["x_fit"],
    cdp["y_fit"],
    "-",
    color="blue",
    linewidth=2.5,
    zorder=4
)
plt.plot(
    fcdp["x_fit"],
    fcdp["y_fit"],
    "-",
    color="red",
    linewidth=2.5,
    zorder=4
)
plt.xlabel(
    "Wind Speed (m s$^{-1}$)",
    fontsize=20,
    fontweight="bold"
)
plt.ylabel(
    "Wind Speed Binned\nConcentration (cm$^{-3}$)",
    fontsize=20,
    fontweight="bold"
)
plt.title(
    "GCCN Concentration as a function of wind speed\nFMAS 2020",
    fontsize=20,
    fontweight="bold"
)
plt.xlim(0, 12)
plt.ylim(0, 1)
plt.xticks(fontsize=18, fontweight="bold")
plt.yticks(fontsize=18, fontweight="bold")
plt.legend(
    fontsize=12,
    frameon=False,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5)
)
plt.tight_layout()
plt.show()
# %%
#all instrument mass
cas_path = "/home/disk/eos4/kathem24/activate/data/2020/CAS/CAS_FMAS2020_wind_bin_mass_plot_data.pkl"
with open(cas_path, "rb") as f:
    cas = pickle.load(f)
cdp_path = "/home/disk/eos4/kathem24/activate/data/2020/CDP/1Hz/CDP_FMAS2020_wind_bin_mass_plot_data.pkl"
with open(cdp_path, "rb") as f:
    cdp = pickle.load(f)
fcdp_path = "/home/disk/eos4/kathem24/activate/data/2020/FCDP/FCDP_FMAS2020_wind_bin_mass_plot_data.pkl"
with open(fcdp_path, "rb") as f:
    fcdp = pickle.load(f)
plt.figure(figsize=(12, 6))
box_width = 0.6
plt.bar(
    cas["x"],
    2 * cas["yerr_2SE"],
    bottom=cas["y"] - cas["yerr_2SE"],
    width=box_width,
    color="gray",
    edgecolor="black",
    alpha=0.6,
    label="CAS ±2 SEM (~95% CI)",
    zorder=1
)
plt.bar(
    cdp["x"],
    2 * cdp["yerr_2SE"],
    bottom=cdp["y"] - cdp["yerr_2SE"],
    width=box_width,
    color="lightblue",
    edgecolor="blue",
    alpha=0.5,
    label="CDP ±2 SEM (~95% CI)",
    zorder=2
)
plt.bar(
    fcdp["x"],
    2 * fcdp["yerr_2SE"],
    bottom=fcdp["y"] - fcdp["yerr_2SE"],
    width=box_width,
    color="mistyrose",
    edgecolor="red",
    alpha=0.5,
    label="FCDP ±2 SEM (~95% CI)",
    zorder=3
)
plt.errorbar(
    cas["x"],
    cas["y"],
    yerr=cas["count_err_for_points_2sigma"],
    fmt="none",
    ecolor="black",
    elinewidth=3,
    capsize=0,
    label="CAS ±2σ Instrument Error",
    zorder=5
)
plt.errorbar(
    cdp["x"],
    cdp["y"],
    yerr=cdp["count_err_for_points_2sigma"],
    fmt="none",
    ecolor="blue",
    elinewidth=3,
    capsize=0,
    label="CDP ±2σ Instrument Error",
    zorder=5
)
plt.errorbar(
    fcdp["x"],
    fcdp["y"],
    yerr=fcdp["count_err_for_points_2sigma"],
    fmt="none",
    ecolor="red",
    elinewidth=3,
    capsize=0,
    label="FCDP ±2σ Instrument Error",
    zorder=5
)
plt.plot(
    cas["x"],
    cas["y"],
    "o",
    color="black",
    label="CAS Mean",
    zorder=6
)
plt.plot(
    cdp["x"],
    cdp["y"],
    "o",
    color="blue",
    label="CDP Mean",
    zorder=6
)
plt.plot(
    fcdp["x"],
    fcdp["y"],
    "o",
    color="red",
    label="FCDP Mean",
    zorder=6
)
plt.plot(
    cas["x_fit"],
    cas["y_fit"],
    "-",
    color="black",
    linewidth=2.5,
    zorder=4
)
plt.plot(
    cdp["x_fit"],
    cdp["y_fit"],
    "-",
    color="blue",
    linewidth=2.5,
    zorder=4
)

plt.plot(
    fcdp["x_fit"],
    fcdp["y_fit"],
    "-",
    color="red",
    linewidth=2.5,
    zorder=4
)
plt.xlabel(
    "Wind Speed (m s$^{-1}$)",
    fontsize=20,
    fontweight="bold"
)
plt.ylabel(
    "Wind Speed Binned\nMass (µg m$^{-3}$)",
    fontsize=20,
    fontweight="bold"
)
plt.title(
    "GCCN Mass as a function of wind speed\nFMAS 2020",
    fontsize=19,
    fontweight="bold"
)
plt.xlim(0, 12)
plt.ylim(0, 50)
plt.xticks(fontsize=18, fontweight="bold")
plt.yticks(fontsize=18, fontweight="bold")
plt.legend(
    fontsize=12,
    frameon=False,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5)
)
plt.tight_layout()
plt.show()
# %%
