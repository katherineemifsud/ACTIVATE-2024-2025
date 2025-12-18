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
#all cases no turbulence gccn concentration vs accumulated rain
# Low Na + turbulence
accum_rain_lownaturb = (
    np.max(LWP_lownaturb, axis=1) - LWP_lownaturb[:, -1]
).copy()

# Base + turbulence
accum_rain_baseturb = (
    np.max(LWP_turb, axis=1) - LWP_turb[:, -1]
).copy()
turb_cases = {
    "Base + Turb": {
        "r": r_dry_turb,
        "n0": n0_r_turb,
        "rain": accum_rain_baseturb,
        "color": "tab:blue"
    },
    "Low Na + Turb": {
        "r": r_dry_lownaturb,
        "n0": n0_r_lownaturb,
        "rain": accum_rain_lownaturb,
        "color": "tab:green",
        "marker": "s"
    }
}
for k, v in turb_cases.items():
    print(k, len(v["rain"]))
plt.figure(figsize=(7, 5))

for label, case in turb_cases.items():

    mask = case["r"] > 0.5e-6   # radius > 0.5 µm
    gccn = np.sum(case["n0"][:, mask], axis=1)
    rain = case["rain"]

    # safety mask
    valid = (gccn > 0) & (rain > 0)
    gccn = gccn[valid]
    rain = rain[valid]

    logx = np.log10(gccn)
    logy = np.log10(rain)

    slope, intercept, R, _, _ = linregress(logx, logy)

    xfit = np.logspace(np.log10(gccn.min()), np.log10(gccn.max()), 200)
    yfit = 10 ** (intercept + slope * np.log10(xfit))

    plt.plot(
        xfit, yfit,
        "--",
        lw=3,
        color=case["color"],
        label=f"{label} (s={slope:.2f}, R={R:.2f})"
    )

    print(f"{label}: slope={slope:.2f}, R={R:.2f}, R²={R**2:.2f}")
plt.xscale("log")
plt.yscale("log")

plt.xlabel("GCCN concentration (m$^{-3}$)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")

plt.title(
    "BCB January–June 2022\n385 g m$^{-2}$ LWP\nGCCN vs Accumulated Rainfall (Turbulence)",
    fontsize=18,
    fontweight="bold"
)

plt.legend(fontsize=12)
plt.grid(alpha=0.3)

plt.xticks(fontsize=14, fontweight="bold")
plt.yticks(fontsize=14, fontweight="bold")

plt.tight_layout()
plt.show()
#%%

plt.figure(figsize=(7, 5))
for label, case in turb_cases.items():
    mask = case["r"] > 0.5e-6
    gccn = np.sum(case["n0"][:, mask], axis=1)
    rain = case["rain"]
    valid = (gccn > 0) & (rain > 0)
    gccn = gccn[valid]
    rain = rain[valid]
    plt.scatter(
        gccn, rain,
        s=40,
        color=case["color"],
        alpha=0.85,
        edgecolor="k",
        linewidth=0.4
    )
    logx = np.log10(gccn)
    logy = np.log10(rain)

    slope, intercept, R, _, _ = linregress(logx, logy)
    xfit = np.logspace(np.log10(gccn.min()), np.log10(gccn.max()), 200)
    yfit = 10 ** (intercept + slope * np.log10(xfit))

    plt.plot(
        xfit, yfit,
        "--",
        lw=3,
        color=case["color"]
    )

    print(f"{label}: slope={slope:.2f}, R={R:.2f}, R²={R**2:.2f}")
legend_elements = [
    Line2D([0], [0],
           marker='o',
           color='tab:blue',
           linestyle='None',
           markersize=8,
           markeredgecolor='k',
           label='Base + Turb legs'),

    Line2D([0], [0],
           color='tab:blue',
           linestyle='--',
           linewidth=3,
           label='Base + Turb fit'),

    Line2D([0], [0],
           marker='s',
           color='tab:green',
           linestyle='None',
           markersize=8,
           markeredgecolor='k',
           label='Low Na + Turb legs'),

    Line2D([0], [0],
           color='tab:green',
           linestyle='--',
           linewidth=3,
           label='Low Na + Turb fit')
]

plt.legend(
    handles=legend_elements,
    fontsize=11,
    frameon=True,
    fancybox=False,
    framealpha=0.9,
    edgecolor='k',
    loc='lower right'
)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("GCCN concentration (m$^{-3}$)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")
plt.title(
    "BCB January–June 2022\n385 g m$^{-2}$ LWP\nGCCN vs Accumulated Rainfall Turbulence",
    fontsize=18,
    fontweight="bold"
)
plt.grid(alpha=0.3)
plt.xticks(fontsize=14, fontweight="bold")
plt.yticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
# %%
#saving this as a .csv
# rows = []

# for label, case in turb_cases.items():

#     mask = case["r"] > 0.5e-6
#     gccn = np.sum(case["n0"][:, mask], axis=1)
#     rain = case["rain"]

#     valid = (gccn > 0) & (rain > 0)
#     gccn = gccn[valid]
#     rain = rain[valid]

#     for i, (g, r) in enumerate(zip(gccn, rain), start=1):
#         rows.append({
#             "Case": label,
#             "Leg": i,
#             "GCCN_m3": g,
#             "Accumulated_Rain_mm": r,
#             "log10_GCCN": np.log10(g),
#             "log10_Rain": np.log10(r)
#         })

# df_gccn_rain_turb = pd.DataFrame(rows)

# print("Total rows saved:", len(df_gccn_rain_turb))
# print(df_gccn_rain_turb.head())
# save_path = (
#     "/home/disk/eos4/kathem24/activate/data/CAS/"
#     "gccn_rain_BCB_385g_LWP_turbulence.csv"
# )

# df_gccn_rain_turb.to_csv(save_path, index=False)

# print("Saved to:")
# print(save_path)


# %%
#mass vs accumulated rain for all cases no turbulence

mass = np.array(all_mass_values)  # identical for all cases 
mass_cases = {
    "Base": {
        "mass": mass,
        "rain": accum_rainbaseturb,
        "color": "tab:blue",
        "marker": "o"
    },
    "Low Na": {
        "mass": mass,
        "rain": accum_rain_lownaturb,
        "color": "tab:green",
        "marker": "s"
    }
}
for k, v in mass_cases.items():
    print(k, len(v["mass"]), len(v["rain"]))
plt.figure(figsize=(7, 5))

for label, case in mass_cases.items():

    m = case["mass"]
    r = case["rain"]

    mask = (m > 0) & (r > 0)  # safety
    m = m[mask]
    r = r[mask]

    logx = np.log10(m)
    logy = np.log10(r)

    slope, intercept, R, _, _ = linregress(logx, logy)

    xfit = np.logspace(np.log10(m.min()), np.log10(m.max()), 200)
    yfit = 10 ** (intercept + slope * np.log10(xfit))

    plt.plot(
        xfit, yfit,
        "--",
        lw=3,
        color=case["color"],
        label=f"{label} (s={slope:.2f}, R={R:.2f})"
    )

    print(f"{label}: slope={slope:.2f}, R={R:.2f}, R²={R**2:.2f}")
plt.xscale("log")
plt.yscale("log")

plt.xlabel("Dry GCCN Mass (µg m$^{-3}$)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")

plt.title(
    "BCB January–June 2022\n385 g m$^{-2}$ LWP\nDry GCCN Mass vs Accumulated Rainfall\nNo Turbulence",
    fontsize=18,
    fontweight="bold"
)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# %%
#mass vs accumulated rain for all cases no turbulence with scatter points
# After computing each accum_rain
           
mass = np.array(all_mass_values)  # identical for all cases 
mass_cases = {
    "Base": {
        "mass": mass,
        "rain": accum_rainbaseturb,
        "color": "tab:blue",
        "marker": "o"
    },
    "Low Na": {
        "mass": mass,
        "rain": accum_rain_lownaturb,
        "color": "tab:green",
        "marker": "s"
    }
}
for k, v in mass_cases.items():
    print(k, len(v["mass"]), len(v["rain"]))
plt.figure(figsize=(7, 5))  
for label, case in mass_cases.items():

    m = case["mass"]
    r = case["rain"]

    plt.scatter(
        m, r,
        s=90,
        marker=case["marker"],
        color=case["color"],
        edgecolor="k",
        alpha=0.85,
        label=f"{label}"
    )

    logx = np.log10(m)
    logy = np.log10(r)

    slope, intercept, R, _, _ = linregress(logx, logy)

    xfit = np.logspace(np.log10(m.min()), np.log10(m.max()), 200)
    yfit = 10 ** (intercept + slope * np.log10(xfit))

    plt.plot(
        xfit, yfit,
        "--",
        lw=3,
        color=case["color"],
        label=f"{label} fit (s={slope:.2f}, R={R:.2f})"
    )

    print(f"{label}: slope={slope:.2f}, R={R:.2f}, R²={R**2:.2f}")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Dry GCCN Mass (µg m$^{-3}$)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")
plt.title(
    "BCB January–June 2022\n385 g m$^{-2}$ LWP\nDry GCCN Mass vs Accumulated Rainfall\nTurbulence",
    fontsize=18,
    fontweight="bold"
)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# %%
# #saving mass vs accumulated rain turbulence as .csv
# rows = []

# for label, case in mass_cases.items():
#     m = case["mass"]
#     r = case["rain"]
#     valid = (m > 0) & (r > 0)
#     m = m[valid]
#     r = r[valid]

#     for i, (mass_val, rain_val) in enumerate(zip(m, r), start=1):
#         rows.append({
#             "Case": label,
#             "Leg": i,
#             "Dry_GCCN_Mass_ug_m3": mass_val,
#             "Accumulated_Rain_mm": rain_val,
#             "log10_Dry_GCCN_Mass": np.log10(mass_val),
#             "log10_Accumulated_Rain": np.log10(rain_val)
#         })

# df_mass_rain_turb = pd.DataFrame(rows)
# save_path = (
#     "/home/disk/eos4/kathem24/activate/data/CAS/"
#     "mass_rain_BCB_385g_LWP_turbulence.csv"
# )

# df_mass_rain_turb.to_csv(save_path, index=False)
# print("Saved to:")
# print(save_path)
# df_check = pd.read_csv(save_path)
# print(df_check.groupby("Case").size())
# print(df_check.head())

