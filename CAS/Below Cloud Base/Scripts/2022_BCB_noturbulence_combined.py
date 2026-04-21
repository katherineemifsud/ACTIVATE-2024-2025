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
#mass vs accumulated rain for all cases no turbulence with scatter points
# After computing each accum_rain
        
mass_cases = {
    "Base": {
        "mass": mass,
        "rain": accum_rain_base,
        "color": "tab:blue",
        "marker": "o"
    },
    "Low Na": {
        "mass": mass,
        "rain": accum_rain_lowna,
        "color": "tab:green",
        "marker": "s"
    },
    "High Na": {
        "mass": mass,
        "rain": accum_rain_highna,
        "color": "tab:red",
        "marker": "^"
    },
    "100 g m$^{-2}$ LWP": {
        "mass": mass,
        "rain": accum_rain_lowLWP,
        "color": "tab:orange",
        "marker": "D"
    }
}
for k, v in mass_cases.items():
    print(k, len(v["mass"]), len(v["rain"]))
for label, case in mass_cases.items():
    print(label, "mass len =", len(case["mass"]), "rain len =", len(case["rain"]))
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

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Dry GCCN Mass (µg m$^{-3}$)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")
plt.title(
    "BCB January–June 2022\n385 g m$^{-2}$ LWP\nLess than 100 µg m$^{-3}$ GCCN\nNo Turbulence",
    fontsize=18,
    fontweight="bold"
)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# %%
#gccn vs accumulated rain for all cases no turbulence with scatter points
# After computing each accum_rain
        
gccn_cases = {
    "Base": {
        "gccn": gccn_m3_base,
        "rain": accum_rain_base,
        "color": "tab:blue",
        "marker": "o"
    },
    "Low Na": {
        "gccn": gccn_m3_lowna,
        "rain": accum_rain_lowna,
        "color": "tab:green",
        "marker": "s"
    },
    "High Na": {
        "gccn": gccn_m3_highna,
        "rain": accum_rain_highna,
        "color": "tab:red",
        "marker": "^"
    },
    "100 g m$^{-2}$ LWP": {
        "gccn": gccn_m3_lowLWP,
        "rain": accum_rain_lowLWP,
        "color": "tab:orange",
        "marker": "D"
    }
}
for k, v in gccn_cases.items():
    print(k, len(v["gccn"]), len(v["rain"]))
plt.figure(figsize=(7, 5))  
for label, case in gccn_cases.items():

    m = case["gccn"]
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

    # logx = np.log10(m)
    # logy = np.log10(r)

    # slope, intercept, R, _, _ = linregress(logx, logy)

    # xfit = np.logspace(np.log10(m.min()), np.log10(m.max()), 200)
    # yfit = 10 ** (intercept + slope * np.log10(xfit))

    # plt.plot(
    #     xfit, yfit,
    #     "--",
    #     lw=3,
    #     color=case["color"],
    #     label=f"{label} fit (s={slope:.2f}, R={R:.2f})"
    # )

    # print(f"{label}: slope={slope:.2f}, R={R:.2f}, R²={R**2:.2f}")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("GCCN concentration (m$^{-3}$)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")
plt.title(
    "BCB January–June 2022\n385 g m$^{-2}$ LWP\nLess than 100 µg m$^{-3}$ GCCN\nNo Turbulence",
    fontsize=18,
    fontweight="bold"
)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# %%