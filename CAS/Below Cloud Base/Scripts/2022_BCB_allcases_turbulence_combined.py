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
#gccn vs accumulated rain for all cases turbulence with scatter points
        
gccn_cases_turb = {
    "Base": {
        "gccn": gccn_m3_baseturb,
        "rain": accum_rain_baseturb,
        "color": "tab:blue",
        "marker": "o"
    },
    "Low Na": {
        "gccn": gccn_m3_lownaturb,
        "rain": accum_rain_lownaturb,
        "color": "tab:green",
        "marker": "s"
    },
    "High Na": {
        "gccn": gccn_m3_highnaturb,
        "rain": accum_rain_highnaturb,
        "color": "tab:red",
        "marker": "^"
    },
    "100 g m$^{-2}$ LWP": {
        "gccn": gccn_m3_lowLWPturb,
        "rain": accum_rain_lowLWPturb,
        "color": "tab:orange",
        "marker": "D"
    },
    "High Turbulence Base": {
        "gccn": gccn_m3_baseturb2,
        "rain": accum_rain_baseturb2,
        "color": "tab:purple",
        "marker": "v"
    },
    "High Turbulence Low Na": {
        "gccn": gccn_m3_lowna_turb2,
        "rain": accum_rain_lowna_turb2,
        "color": "tab:brown",
        "marker": "<"
    },
    "High Turbulence High Na": {
        "gccn": gccn_m3_highnaturb2,
        "rain": accum_rain_highnaturb2,
        "color": "tab:pink",
        "marker": ">"
    },
    "High Turbulence 100 g m$^{-2}$ LWP": {
        "gccn": gccn_m3_lowLWPturb2,
        "rain": accum_rain_lowLWPturb2,
        "color": "tab:gray",
        "marker": "P"
    }
}
for k, v in gccn_cases_turb.items():
    print(k, len(v["gccn"]), len(v["rain"]))
plt.figure(figsize=(7, 5))  
for label, case in gccn_cases_turb.items():

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

plt.xscale("log")
plt.yscale("log")
plt.xlabel("GCCN concentration (m$^{-3}$)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")
plt.title(
    "BCB January–June 2022\n385 g m$^{-2}$ LWP\nLess than 100 µg m$^{-3}$ GCCN\nTurbulence",
    fontsize=18,
    fontweight="bold"
)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.yticks(fontsize=16, fontweight="bold")
plt.xticks(fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()
# %%
#thinning the data for better visualization, use every 5th point for each of the 4 cases
plt.figure(figsize=(7, 5))
for label, case in gccn_cases_turb.items():

    m = case["gccn"]
    r = case["rain"]

    thin_m = m[::5]
    thin_r = r[::5]

    plt.scatter(
        thin_m, thin_r,
        s=90,
        marker=case["marker"],
        color=case["color"],
        edgecolor="k",
        alpha=0.85,
        label=f"{label}"
    )
plt.xscale("log")
plt.yscale("log")
plt.xlabel("GCCN concentration (m$^{-3}$)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")
plt.title(
    "BCB January–June 2022\n385 g m$^{-2}$ LWP\nLess than 100 µg m$^{-3}$ GCCN\nTurbulence (95 legs)",
    fontsize=18,
    fontweight="bold"
)
plt.legend(fontsize=12)
plt.yticks(fontsize=16, fontweight="bold")
plt.xticks(fontsize=16, fontweight="bold")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# %%
#mass vs accumulated rain for all cases turbulence

mass = np.array(all_mass_values)  
mass_cases = {
    "Base": {
        "mass": mass,
        "rain": accum_rain_baseturb,
        "color": "tab:blue",
        "marker": "o"
    },
    "Low Na": {
        "mass": mass,
        "rain": accum_rain_lownaturb,
        "color": "tab:green",
        "marker": "s"
    },
    "High Na": {
        "mass": mass,
        "rain": accum_rain_highnaturb,
        "color": "tab:red",
        "marker": "^"
    },
    "100 g m$^{-2}$ LWP": {
        "mass": mass,
        "rain": accum_rain_lowLWP,
        "color": "tab:orange",
        "marker": "D"
    },
    "High Turbulence Base": {
        "mass": mass,
        "rain": accum_rain_baseturb2,
        "color": "tab:purple",
        "marker": "v"
    },

    "High Turbulence Low Na": {
        "mass": mass,
        "rain": accum_rain_lowna_turb2,
        "color": "tab:brown",
        "marker": "<"
    },
    "High Turbulence High Na": {
        "mass": mass,
        "rain": accum_rain_highnaturb2,
        "color": "tab:pink",
        "marker": ">"
    },
    "High Turbulence 100 g m$^{-2}$ LWP": {
        "mass": mass,
        "rain": accum_rain_lowLWPturb2,
        "color": "tab:gray",
        "marker": "P"
    }
}
for k, v in mass_cases.items():
    print(k, len(v["mass"]), len(v["rain"]))
plt.figure(figsize=(7, 5))

for label, case in mass_cases.items():

    m = case["mass"]
    r = case["rain"]

    mask = (m > 0) & (r > 0) 
    m = m[mask]
    r = r[mask]

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
#mass vs accumulated rain for all cases turbulence with scatter points
          
mass = np.array(all_mass_values)  
mass_cases_turb = {
    "Base": {
        "mass": mass,
        "rain": accum_rain_baseturb,
        "color": "tab:blue",
        "marker": "o"
    },
    "Low Na": {
        "mass": mass,
        "rain": accum_rain_lownaturb,
        "color": "tab:green",
        "marker": "s"
    },
    "High Na": {
        "mass": mass,
        "rain": accum_rain_highnaturb,
        "color": "tab:red",
        "marker": "^" 
    },
    "100 g m$^{-2}$ LWP": {
        "mass": mass,
        "rain": accum_rain_lowLWPturb,
        "color": "tab:orange",
        "marker": "D"
    }, 
    "High Turbulence Base": {
        "mass": mass,
        "rain": accum_rain_baseturb2,
        "color": "tab:purple",
        "marker": "v"
    },  
    "High Turbulence Low Na": {
        "mass": mass,
        "rain": accum_rain_lowna_turb2,
        "color": "tab:brown",
        "marker": "<"
    },
    "High Turbulence High Na": {
        "mass": mass,
        "rain": accum_rain_highnaturb2,
        "color": "tab:pink",
        "marker": ">"
    },
    "High Turbulence 100 g m$^{-2}$ LWP": {
        "mass": mass,
        "rain": accum_rain_lowLWPturb2,
        "color": "tab:gray",
        "marker": "P"
    }
}
for k, v in mass_cases_turb.items():
    print(k, len(v["mass"]), len(v["rain"]))
plt.figure(figsize=(7, 5))  
for label, case in mass_cases_turb.items():

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
    "BCB January–June 2022\n385 g m$^{-2}$ LWP\n Less than 100 µg/m³ mass\nTurbulence",
    fontsize=18,
    fontweight="bold"
)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
#%%
#thinning the data for better visualization, use every 5th point for each of the 4 cases
plt.figure(figsize=(7, 5))
for label, case in mass_cases_turb.items():
    m = case["mass"]
    r = case["rain"]

    mask = (m > 0) & (r > 0) 
    m = m[mask]
    r = r[mask]

    thin_m = m[::5]
    thin_r = r[::5]

    plt.scatter(
        thin_m, thin_r,
        s=90,
        marker=case["marker"],
        color=case["color"],
        edgecolor="k",
        alpha=0.85,
        label=f"{label}"
    )
plt.xscale("log")
plt.yscale("log")
plt.xlabel("GCCN Mass (µg m$^{-3}$)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")
plt.title(
    "BCB January–June 2022\n385 g m$^{-2}$ LWP\n Less than 100 µg/m³ mass\nTurbulence (95 legs)",
    fontsize=18,
    fontweight="bold"
)
plt.legend(fontsize=11, frameon=True)
plt.grid(alpha=0.3)
plt.xticks(fontsize=14, fontweight="bold")
plt.yticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
# %%
#combining turbulence and non turbulence mass vs rain data into one figure for easier comparison using the same color for each case but different markers for turb vs no turb
plt.figure(figsize=(7, 5))
for label, case in mass_cases.items():
    m = case["mass"]
    r = case["rain"]

    mask = (m > 0) & (r > 0)
    m = m[mask]
    r = r[mask]

    plt.scatter(
        m, r,
        s=90,
        marker=case["marker"],
        color=case["color"],
        edgecolor="k",
        alpha=0.85,
        label=f"{label} (Turbulence)"
    )   
    non_turb_label = label.replace(" + Turb", "")
    non_turb_case = mass_cases.get(non_turb_label)
    if non_turb_case:
        non_turb_m = non_turb_case["mass"]
        non_turb_r = non_turb_case["rain"]

        non_turb_mask = (non_turb_m > 0) & (non_turb_r > 0)
        non_turb_m = non_turb_m[non_turb_mask]
        non_turb_r = non_turb_r[non_turb_mask]

        plt.scatter(
            non_turb_m, non_turb_r,
            s=90,
            marker="X",  
            color=case["color"],
            edgecolor="k",
            alpha=0.85,
            label=f"{non_turb_label} (No Turbulence)"
        )   
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Dry GCCN Mass (µg m$^{-3}$)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")
plt.title(
    "BCB January–June 2022\n385 g m$^{-2}$ LWP\nGCCN Mass vs Accumulated Rainfall",
    fontsize=18,
    fontweight="bold"
)
plt.legend(
    ncol=2,
    fontsize=6
)
plt.grid(alpha=0.3)
plt.xticks(fontsize=14, fontweight="bold")
plt.yticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
#%%
#line plot
cb_colors = {
    "Base": "#0072B2",                 # blue
    "Low Na": "#009E73",               # bluish green
    "High Na": "#CC79A7",              # magenta
    "100 g m$^{-2}$ LWP": "#E69F00"    # orange
}
def base_case_name(label):
    if "Low Na" in label:
        return "Low Na"
    if "High Na" in label:
        return "High Na"
    if "100 g m$^{-2}$ LWP" in label:
        return "100 g m$^{-2}$ LWP"
    return "Base"

for cases in [mass_cases, mass_cases_turb]:
    for label in cases:
        base = base_case_name(label)
        cases[label]["color"] = cb_colors[base]
turb_styles = {
    "No Turbulence": {"linestyle": "-",  "marker": None},
    "Turbulence":    {"linestyle": "--", "marker": None},
    "High Turbulence": {
        "linestyle": "-",
        "marker": "o",       # or "s", "^", "x" — see note below
        "markersize": 6,
        "markevery": 2
    }
}


#%%
def log_binned_median(x, y, nbins=20, min_per_bin=5):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[valid]
    y = y[valid]

    if x.size == 0:
        return np.array([]), np.array([])
    bins = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), nbins + 1)

    x_med, y_med = [], []
    for i in range(nbins):
        msk = (x >= bins[i]) & (x < bins[i + 1])
        if np.sum(msk) >= min_per_bin:
            x_med.append(np.nanmedian(x[msk]))
            y_med.append(np.nanmedian(y[msk]))

    return np.asarray(x_med), np.asarray(y_med)
plt.figure(figsize=(7, 5))

def split_turb_label(label):
    if label.startswith("High Turbulence "):
        return label.replace("High Turbulence ", ""), "High Turbulence"
    return label, "Turbulence"

# --- No turbulence ---
for label, case in mass_cases.items():
    m = np.asarray(case["mass"], dtype=float)
    r = np.asarray(case["rain"], dtype=float)
    assert len(m) == len(r), f"No-turb mismatch for {label}: mass={len(m)} rain={len(r)}"

    x_med, y_med = log_binned_median(m, r, nbins=20, min_per_bin=5)
    if x_med.size == 0:
        continue

    plt.plot(
        x_med, y_med,
        color=case["color"],
        linewidth=3,
        linestyle="-",
        alpha=0.95,
        label=f"{label} (No Turbulence)"
    )

# --- Turbulence + High turbulence (both are inside mass_cases_turb) ---
for label, case in mass_cases_turb.items():
    m = np.asarray(case["mass"], dtype=float)
    r = np.asarray(case["rain"], dtype=float)
    assert len(m) == len(r), f"Turb mismatch for {label}: mass={len(m)} rain={len(r)}"

    base_label, turb_tag = split_turb_label(label)
    is_high = (turb_tag == "High Turbulence")

    x_med, y_med = log_binned_median(m, r, nbins=20, min_per_bin=5)
    if x_med.size == 0:
        continue

    plt.plot(
        x_med, y_med,
        color=case["color"],
        linewidth=3,
        linestyle="--" if not is_high else "-",
        marker="o" if is_high else None,
        markersize=6 if is_high else None,
        markevery=2 if is_high else None,
        alpha=0.95,
        label=f"{base_label} ({turb_tag})"
    )

plt.xscale("log")
plt.yscale("log")
plt.xlabel("GCCN Mass (µg m$^{-3}$)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")
plt.title(
    "BCB January–June 2022\n385 g m$^{-2}$ LWP\nLog-binned Median Curves\nLess than 100 µg/m³ mass",
    fontsize=18,
    fontweight="bold"
)
plt.legend(ncol=1, fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1))
plt.grid(alpha=0.3)
plt.xticks(fontsize=14, fontweight="bold")
plt.yticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
#%%
# %%
#combining turbulence and non turbulence gccn concentration vs rain data into one figure for easier comparison using the same color for each case but different markers for turb vs no turb
cb_colors = {
    "Base": "#0072B2",        # blue
    "Low Na": "#009E73",       # bluish green
    "High Na": "#CC79A7",      # purple/magenta
    "100 g m$^{-2}$ LWP": "#E69F00"  # orange
}
def base_case_name(label):
    if "Low Na" in label:
        return "Low Na"
    if "High Na" in label:
        return "High Na"
    if "100 g m$^{-2}$ LWP" in label:
        return "100 g m$^{-2}$ LWP"
    return "Base"

for cases in (gccn_cases, gccn_cases_turb):
    for label in cases:
        cases[label]["color"] = cb_colors[base_case_name(label)]

#%%
def log_binned_median(x, y, nbins=20, min_per_bin=5):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[valid]
    y = y[valid]
    if x.size == 0:
        return np.array([]), np.array([])
    bins = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), nbins + 1)
    x_med, y_med = [], []
    for i in range(nbins):
        msk = (x >= bins[i]) & (x < bins[i + 1])
        if np.sum(msk) >= min_per_bin:
            x_med.append(np.nanmedian(x[msk]))
            y_med.append(np.nanmedian(y[msk]))

    return np.asarray(x_med), np.asarray(y_med)
def split_turb_label(label):
    if label.startswith("High Turbulence "):
        return label.replace("High Turbulence ", ""), "High Turbulence"
    return label, "Turbulence"

plt.figure(figsize=(7, 5))
for label, case in gccn_cases.items():
    g = np.asarray(case["gccn"], dtype=float)
    r = np.asarray(case["rain"], dtype=float)
    assert len(g) == len(r), f"No-turb mismatch for {label}: gccn={len(g)} rain={len(r)}"
    x_med, y_med = log_binned_median(g, r, nbins=20, min_per_bin=5)
    if x_med.size == 0:
        continue
    plt.plot(
        x_med, y_med,
        color=case["color"],
        linewidth=3,
        linestyle="-",
        alpha=0.95,
        label=f"{label} (No Turbulence)"
    )
for label, case in gccn_cases_turb.items():
    g = np.asarray(case["gccn"], dtype=float)
    r = np.asarray(case["rain"], dtype=float)
    assert len(g) == len(r), f"Turb mismatch for {label}: gccn={len(g)} rain={len(r)}"

    base_label, turb_tag = split_turb_label(label)
    is_high = (turb_tag == "High Turbulence")

    x_med, y_med = log_binned_median(g, r, nbins=20, min_per_bin=5)
    if x_med.size == 0:
        continue

    plt.plot(
        x_med, y_med,
        color=case["color"],            # SAME color for all 3 cases
        linewidth=3,
        linestyle="--" if not is_high else "-",  # dashed vs solid
        marker="o" if is_high else None,          # marker ONLY for high turb
        markersize=6 if is_high else None,
        markevery=2 if is_high else None,
        alpha=0.95,
        label=f"{base_label} ({turb_tag})"
    )

plt.xscale("log")
plt.yscale("log")
plt.xlabel("GCCN concentration (m$^{-3}$)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")
plt.title(
    "BCB January–June 2022\n"
    "Base 385 g m$^{-2}$ LWP\n"
    "Less than 100 µg/m$^3$ mass",
    fontsize=18,
    fontweight="bold"
)
plt.legend(
    ncol=1,
    fontsize=9,
    loc="upper left",
    bbox_to_anchor=(1.02, 1)
)
plt.grid(alpha=0.3)
plt.xticks(fontsize=14, fontweight="bold")
plt.yticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
# %%
#2 panel paper figure for mass vs rain and gccn vs rain turbulence vs no turbulence
# fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
# ax1, ax2 = axes

# for label, case in gccn_cases.items():
#     g = np.asarray(case["gccn"], dtype=float)
#     r = np.asarray(case["rain"], dtype=float)
#     x_med, y_med = log_binned_median(g, r)
#     if x_med.size == 0:
#         continue
#     ax1.plot(x_med, y_med, color=case["color"], lw=3, ls="-",
#              label=f"{label} (No Turb)")

# for label, case in gccn_cases_turb.items():
#     g = np.asarray(case["gccn"], dtype=float)
#     r = np.asarray(case["rain"], dtype=float)
#     x_med, y_med = log_binned_median(g, r)
#     if x_med.size == 0:
#         continue
#     ax1.plot(x_med, y_med, color=case["color"], lw=3, ls="--",
#              label=f"{label} (Turb)")

# ax1.set_xscale("log")
# ax1.set_yscale("log")
# ax1.set_xlabel("GCCN concentration (m$^{-3}$)", fontsize=16, fontweight="bold")
# ax1.set_ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")
# ax1.set_title("(a) GCCN Concentration", fontsize=18, fontweight="bold")
# ax1.grid(alpha=0.3)

# for label, case in mass_cases.items():
#     m = np.asarray(case["mass"], dtype=float)
#     r = np.asarray(case["rain"], dtype=float)
#     x_med, y_med = log_binned_median(m, r)
#     if x_med.size == 0:
#         continue
#     ax2.plot(x_med, y_med, color=case["color"], lw=3, ls="-")

# for label, case in mass_cases_turb.items():
#     m = np.asarray(case["mass"], dtype=float)
#     r = np.asarray(case["rain"], dtype=float)
#     x_med, y_med = log_binned_median(m, r)
#     if x_med.size == 0:
#         continue
#     ax2.plot(x_med, y_med, color=case["color"], lw=3, ls="--")

# ax2.set_xscale("log")
# ax2.set_yscale("log")
# ax2.set_xlabel("GCCN Mass (µg m$^{-3}$)", fontsize=16, fontweight="bold")
# ax2.set_title("(b) GCCN Mass", fontsize=18, fontweight="bold")
# ax2.grid(alpha=0.3)
# handles, labels = ax1.get_legend_handles_labels()
# fig.legend(
#     handles, labels,
#     loc="center left",
#     bbox_to_anchor=(1.02, 0.5),
#     fontsize=11,
#     frameon=False
# )
# fig.suptitle(
#     "BCB January–June 2022\nBase 385 g m$^{-2}$ LWP\nLog-binned Median Curves",
#     fontsize=25,
#     fontweight="bold",
#     y=1.05
# )
# for ax in axes:
#     ax.tick_params(labelsize=15, width=2, length=6)
#     for label in ax.get_xticklabels() + ax.get_yticklabels():
#         label.set_fontweight("bold")

# plt.tight_layout()
# plt.show()
# plt.tight_layout(rect=[0, 0, 0.82, 1])

# %%
def split_turb_label(label):
    if label.startswith("High Turbulence "):
        return label.replace("High Turbulence ", ""), "High Turbulence"
    return label, "Turbulence"

fig, axes = plt.subplots(1, 2, figsize=(10, 7), sharey=True)
ax1, ax2 = axes
for label, case in gccn_cases.items():
    g = np.asarray(case["gccn"], dtype=float)
    r = np.asarray(case["rain"], dtype=float)
    x_med, y_med = log_binned_median(g, r)
    if x_med.size == 0:
        continue
    ax1.plot(
        x_med, y_med,
        color=case["color"], lw=3, ls="-",
        label=f"{label} (No Turbulence)"
    )

for label, case in gccn_cases_turb.items():
    g = np.asarray(case["gccn"], dtype=float)
    r = np.asarray(case["rain"], dtype=float)
    x_med, y_med = log_binned_median(g, r)
    if x_med.size == 0:
        continue

    base_label, turb_tag = split_turb_label(label)
    is_high = (turb_tag == "High Turbulence")

    ax1.plot(
        x_med, y_med,
        color=case["color"], lw=3,
        ls="--" if not is_high else "-",
        marker="o" if is_high else None,
        markersize=6 if is_high else None,
        markevery=2 if is_high else None,
        label=f"{base_label} ({turb_tag})"
    )

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("GCCN concentration (m$^{-3}$)", fontsize=13, fontweight="bold")
ax1.set_ylabel("Accumulated Rain (mm)", fontsize=13, fontweight="bold")
ax1.set_title("(a) GCCN Concentration", fontsize=13, fontweight="bold")
ax1.grid(alpha=0.3)
for label, case in mass_cases.items():
    m = np.asarray(case["mass"], dtype=float)
    r = np.asarray(case["rain"], dtype=float)
    x_med, y_med = log_binned_median(m, r)
    if x_med.size == 0:
        continue
    ax2.plot(
        x_med, y_med,
        color=case["color"], lw=3, ls="-"
    )

for label, case in mass_cases_turb.items():
    m = np.asarray(case["mass"], dtype=float)
    r = np.asarray(case["rain"], dtype=float)
    x_med, y_med = log_binned_median(m, r)
    if x_med.size == 0:
        continue

    base_label, turb_tag = split_turb_label(label)
    is_high = (turb_tag == "High Turbulence")

    ax2.plot(
        x_med, y_med,
        color=case["color"], lw=3,
        ls="--" if not is_high else "-",
        marker="o" if is_high else None,
        markersize=6 if is_high else None,
        markevery=2 if is_high else None
    )

ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("GCCN Mass (µg m$^{-3}$)", fontsize=13, fontweight="bold")
ax2.set_title("(b) GCCN Mass", fontsize=13, fontweight="bold")
ax2.grid(alpha=0.3)
handles, labels = ax1.get_legend_handles_labels()
uniq = dict(zip(labels, handles))
fig.legend(
    uniq.values(), uniq.keys(),
    loc="center left",
    bbox_to_anchor=(0.63, 0.25),
    fontsize=6,
    frameon=False
)

fig.suptitle(
    "Precipiation as a function of mass and concentration\nLog-binned Median Curves",
    fontsize=14,
    fontweight="bold",
    y=0.98
)

for ax in axes:
    ax.tick_params(labelsize=13, width=2, length=6)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
plt.tight_layout(rect=[0, 0, 0.82, 1])
plt.show()

# %%
