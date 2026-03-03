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
# #thinning the data for better visualization, use every 5th point for each of the 4 cases
# plt.figure(figsize=(7, 5))
# for label, case in gccn_cases_turb.items():

#     m = case["gccn"]
#     r = case["rain"]

#     thin_m = m[::5]
#     thin_r = r[::5]

#     plt.scatter(
#         thin_m, thin_r,
#         s=90,
#         marker=case["marker"],
#         color=case["color"],
#         edgecolor="k",
#         alpha=0.85,
#         label=f"{label}"
#     )
# plt.xscale("log")
# plt.yscale("log")
# plt.xlabel("GCCN concentration (m$^{-3}$)", fontsize=16, fontweight="bold")
# plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")
# plt.title(
#     "BCB January–June 2022\n385 g m$^{-2}$ LWP\nLess than 100 µg m$^{-3}$ GCCN\nTurbulence (95 legs)",
#     fontsize=18,
#     fontweight="bold"
# )
# plt.legend(fontsize=12)
# plt.yticks(fontsize=16, fontweight="bold")
# plt.xticks(fontsize=16, fontweight="bold")
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.show()
# %%
# #mass vs accumulated rain for all cases turbulence

# mass = np.array(all_mass_values)  
# mass_cases = {
#     "Base": {
#         "mass": mass,
#         "rain": accum_rain_baseturb,
#         "color": "tab:blue",
#         "marker": "o"
#     },
#     "Low Na": {
#         "mass": mass,
#         "rain": accum_rain_lownaturb,
#         "color": "tab:green",
#         "marker": "s"
#     },
#     "High Na": {
#         "mass": mass,
#         "rain": accum_rain_highnaturb,
#         "color": "tab:red",
#         "marker": "^"
#     },
#     "100 g m$^{-2}$ LWP": {
#         "mass": mass,
#         "rain": accum_rain_lowLWP,
#         "color": "tab:orange",
#         "marker": "D"
#     },
#     "High Turbulence Base": {
#         "mass": mass,
#         "rain": accum_rain_baseturb2,
#         "color": "tab:purple",
#         "marker": "v"
#     },

#     "High Turbulence Low Na": {
#         "mass": mass,
#         "rain": accum_rain_lowna_turb2,
#         "color": "tab:brown",
#         "marker": "<"
#     },
#     "High Turbulence High Na": {
#         "mass": mass,
#         "rain": accum_rain_highnaturb2,
#         "color": "tab:pink",
#         "marker": ">"
#     },
#     "High Turbulence 100 g m$^{-2}$ LWP": {
#         "mass": mass,
#         "rain": accum_rain_lowLWPturb2,
#         "color": "tab:gray",
#         "marker": "P"
#     }
# }
# for k, v in mass_cases.items():
#     print(k, len(v["mass"]), len(v["rain"]))
# plt.figure(figsize=(7, 5))

# for label, case in mass_cases.items():

#     m = case["mass"]
#     r = case["rain"]

#     mask = (m > 0) & (r > 0) 
#     m = m[mask]
#     r = r[mask]

# plt.xscale("log")
# plt.yscale("log")

# plt.xlabel("Dry GCCN Mass (µg m$^{-3}$)", fontsize=16, fontweight="bold")
# plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")

# plt.title(
#     "BCB January–June 2022\n385 g m$^{-2}$ LWP\nDry GCCN Mass vs Accumulated Rainfall\nTurbulence",
#     fontsize=18,
#     fontweight="bold"
# )
# plt.legend(fontsize=12)
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.show()
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
# #thinning the data for better visualization, use every 5th point for each of the 4 cases
# plt.figure(figsize=(7, 5))
# for label, case in mass_cases_turb.items():
#     m = case["mass"]
#     r = case["rain"]

#     mask = (m > 0) & (r > 0) 
#     m = m[mask]
#     r = r[mask]

#     thin_m = m[::5]
#     thin_r = r[::5]

#     plt.scatter(
#         thin_m, thin_r,
#         s=90,
#         marker=case["marker"],
#         color=case["color"],
#         edgecolor="k",
#         alpha=0.85,
#         label=f"{label}"
#     )
# plt.xscale("log")
# plt.yscale("log")
# plt.xlabel("GCCN Mass (µg m$^{-3}$)", fontsize=16, fontweight="bold")
# plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")
# plt.title(
#     "BCB January–June 2022\n385 g m$^{-2}$ LWP\n Less than 100 µg/m³ mass\nTurbulence (95 legs)",
#     fontsize=18,
#     fontweight="bold"
# )
# plt.legend(fontsize=11, frameon=True)
# plt.grid(alpha=0.3)
# plt.xticks(fontsize=14, fontweight="bold")
# plt.yticks(fontsize=14, fontweight="bold")
# plt.tight_layout()
# plt.show()
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
#removing the garbage crashing legs AND mass 

bad_idx = sorted(set([
    6, 29, 35, 38, 43, 45, 52, 66, 95, 114, 119, 133, 134, 173, 179, 182, 196,
    199, 202, 204, 206, 215, 217, 218, 234, 247, 265, 266, 267, 269, 270,
    275, 277, 282, 286, 287, 288, 289, 294, 297, 304, 312, 327, 328, 345,
    349, 372, 374, 376, 377, 391, 399, 401, 402, 406, 408, 426, 435, 436,
    437, 440, 444
]))

mass_thr = 100.0

mass_full = np.asarray(mass_full, dtype=float).ravel()
N = len(mass_full)

bad_mask = np.zeros(N, dtype=bool)
bad_mask[bad_idx] = True
good_base = (
    (~bad_mask) &
    np.isfinite(mass_full) &
    (mass_full > 0) &
    (mass_full <= mass_thr)
)

print("Total legs:", N)
print("Bad-index legs removed:", bad_mask.sum())
print("Mass>thr removed:", int(np.sum((mass_full > mass_thr) & (~bad_mask))))
print("Final kept (after bad+mass):", int(np.sum(good_base)))
def apply_global_filters(mass_full, rain_full, gccn_full, good_mask):
    mass_full = np.asarray(mass_full, dtype=float).ravel()
    rain_full = np.asarray(rain_full, dtype=float).ravel()
    if gccn_full is None:
        gccn_ok = np.ones_like(mass_full, dtype=bool)
    else:
        gccn_full = np.asarray(gccn_full, dtype=float).ravel()
        gccn_ok = np.isfinite(gccn_full)

    assert len(mass_full) == len(rain_full) == len(good_mask), \
        f"Length mismatch: mass={len(mass_full)} rain={len(rain_full)} good={len(good_mask)}"

    ok = good_mask & np.isfinite(rain_full) & (rain_full > 0) & gccn_ok
    return mass_full[ok], rain_full[ok]
mass_base,    rain_base    = apply_global_filters(mass_full, accum_rain_base_full,    gccn_m3_base_full,    good_base)
mass_lowna,   rain_lowna   = apply_global_filters(mass_full, accum_rain_lowna_full,   gccn_m3_lowna_full,   good_base)
mass_highna,  rain_highna  = apply_global_filters(mass_full, accum_rain_highna_full,  gccn_m3_highna_full,  good_base)
mass_lowLWP,  rain_lowLWP  = apply_global_filters(mass_full, accum_rain_lowLWP_full,  gccn_m3_lowLWP_full,  good_base)
mass_baseturb,   rain_baseturb   = apply_global_filters(mass_full, accum_rain_baseturb_full,   gccn_m3_baseturb_full,   good_base)
mass_lownaturb,  rain_lownaturb  = apply_global_filters(mass_full, accum_rain_lownaturb_full,  gccn_m3_lownaturb_full,  good_base)
mass_highnaturb, rain_highnaturb = apply_global_filters(mass_full, accum_rain_highnaturb_full, gccn_m3_highnaturb_full, good_base)
mass_lowLWPturb, rain_lowLWPturb = apply_global_filters(mass_full, accum_rain_lowLWPturb_full, gccn_m3_lowLWPturb_full, good_base)
mass_baseturb2,   rain_baseturb2   = apply_global_filters(mass_full, accum_rain_baseturb_full2,   gccn_m3_baseturb_full2,   good_base)
mass_lownaturb2,  rain_lownaturb2  = apply_global_filters(mass_full, accum_rain_lowna_full2,  gccn_m3_lownaturb_full2,  good_base)
mass_highnaturb2, rain_highnaturb2 = apply_global_filters(mass_full, accum_rain_highnaturb2_full, gccn_m3_highnaturb2_full, good_base)
mass_lowLWPturb2, rain_lowLWPturb2 = apply_global_filters(mass_full, accum_rain_lowLWPturb2_full, gccn_m3_lowLWPturb2_full, good_base)
cb_colors = {
    "Base": "#0072B2",
    "Low Na": "#009E73",
    "High Na": "#CC79A7",
    "100 g m$^{-2}$ LWP": "#E69F00"
}

mass_cases = {
    "Base":              {"mass": mass_base,    "rain": rain_base,    "color": cb_colors["Base"],              "marker": "o"},
    "Low Na":            {"mass": mass_lowna,   "rain": rain_lowna,   "color": cb_colors["Low Na"],            "marker": "s"},
    "High Na":           {"mass": mass_highna,  "rain": rain_highna,  "color": cb_colors["High Na"],           "marker": "^"},
    "100 g m$^{-2}$ LWP":{"mass": mass_lowLWP,  "rain": rain_lowLWP,  "color": cb_colors["100 g m$^{-2}$ LWP"],"marker": "D"},
}

mass_cases_turb = {
    "Base + Turb":               {"mass": mass_baseturb,   "rain": rain_baseturb,   "color": cb_colors["Base"]},
    "Low Na + Turb":             {"mass": mass_lownaturb,  "rain": rain_lownaturb,  "color": cb_colors["Low Na"]},
    "High Na + Turb":            {"mass": mass_highnaturb, "rain": rain_highnaturb, "color": cb_colors["High Na"]},
    "100 g m$^{-2}$ LWP + Turb": {"mass": mass_lowLWPturb, "rain": rain_lowLWPturb, "color": cb_colors["100 g m$^{-2}$ LWP"]},

    "High Turbulence Base":               {"mass": mass_baseturb2,   "rain": rain_baseturb2,   "color": cb_colors["Base"]},
    "High Turbulence Low Na":             {"mass": mass_lownaturb2,  "rain": rain_lownaturb2,  "color": cb_colors["Low Na"]},
    "High Turbulence High Na":            {"mass": mass_highnaturb2, "rain": rain_highnaturb2, "color": cb_colors["High Na"]},
    "High Turbulence 100 g m$^{-2}$ LWP": {"mass": mass_lowLWPturb2, "rain": rain_lowLWPturb2, "color": cb_colors["100 g m$^{-2}$ LWP"]},
}
plt.figure(figsize=(7, 5))
for label, case in mass_cases.items():
    m = np.asarray(case["mass"], dtype=float).ravel()
    r = np.asarray(case["rain"], dtype=float).ravel()
    ok = np.isfinite(m) & np.isfinite(r) & (m > 0) & (r > 0)
    plt.scatter(
        m[ok], r[ok],
        s=90,
        marker=case["marker"],
        color=case["color"],
        edgecolor="k",
        alpha=0.85,
        label=f"{label} (No Turbulence)"
    )
for label, case in mass_cases_turb.items():
    m = np.asarray(case["mass"], dtype=float).ravel()
    r = np.asarray(case["rain"], dtype=float).ravel()
    ok = np.isfinite(m) & np.isfinite(r) & (m > 0) & (r > 0)

    is_high = label.startswith("High Turbulence")
    plt.scatter(
        m[ok], r[ok],
        s=90,
        marker=("o" if is_high else "X"),
        color=case["color"],
        edgecolor="k",
        alpha=0.85,
        label=f"{label} ({'High Turbulence' if is_high else 'Turbulence'})"
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
plt.legend(ncol=2, fontsize=6)
plt.grid(alpha=0.3)
plt.xticks(fontsize=14, fontweight="bold")
plt.yticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
def log_binned_median(x, y, nbins=20, min_per_bin=5):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
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
    if label.startswith("High Turbulence"):
        return label.replace("High Turbulence ", "").replace("High Turbulence", "").strip(), "High Turbulence"
    return label.replace(" + Turb", ""), "Turbulence"

plt.figure(figsize=(7, 5))
for label, case in mass_cases.items():
    x_med, y_med = log_binned_median(case["mass"], case["rain"], nbins=20, min_per_bin=5)
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
for label, case in mass_cases_turb.items():
    base_label, turb_tag = split_turb_label(label)
    is_high = (turb_tag == "High Turbulence")

    x_med, y_med = log_binned_median(case["mass"], case["rain"], nbins=20, min_per_bin=5)
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
#%%
#removing the garbage legs 
bad_idx = sorted(set([
    6, 29, 35, 38, 43, 45, 52, 66, 95, 114, 119, 133, 134, 173, 179, 182, 196,
    199, 202, 204, 206, 215, 217, 218, 234, 247, 265, 266, 267, 269, 270,
    275, 277, 282, 286, 287, 288, 289, 294, 297, 304, 312, 327, 328, 345,
    349, 372, 374, 376, 377, 391, 399, 401, 402, 406, 408, 426, 435, 436,
    437, 440, 444
]))

mass_thr = 100.0

mass_full = np.asarray(mass_full, dtype=float).ravel()
N = len(mass_full)

bad_mask = np.zeros(N, dtype=bool)
bad_mask[bad_idx] = True

good_base = (
    (~bad_mask) &
    np.isfinite(mass_full) &
    (mass_full > 0) &
    (mass_full <= mass_thr)
)

print("Total legs:", N)
print("Bad-index legs removed:", bad_mask.sum())
print("Mass>thr removed:", int(np.sum((mass_full > mass_thr) & (~bad_mask))))
print("Final kept (after bad+mass):", int(np.sum(good_base)))
def apply_global_filters_gccn(mass_full, rain_full, gccn_full, good_mask):
    mass_full = np.asarray(mass_full, dtype=float).ravel()
    rain_full = np.asarray(rain_full, dtype=float).ravel()
    gccn_full = np.asarray(gccn_full, dtype=float).ravel()

    assert len(mass_full) == len(rain_full) == len(gccn_full) == len(good_mask), \
        f"Length mismatch: mass={len(mass_full)} rain={len(rain_full)} gccn={len(gccn_full)} good={len(good_mask)}"

    ok = (
        good_mask &
        np.isfinite(rain_full) & (rain_full > 0) &
        np.isfinite(gccn_full) & (gccn_full > 0)
    )
    return gccn_full[ok], rain_full[ok]
gccn_base,   rain_base    = apply_global_filters_gccn(mass_full, accum_rain_base_full,    gccn_m3_base_full,    good_base)
gccn_lowna,  rain_lowna   = apply_global_filters_gccn(mass_full, accum_rain_lowna_full,   gccn_m3_lowna_full,   good_base)
gccn_highna, rain_highna  = apply_global_filters_gccn(mass_full, accum_rain_highna_full,  gccn_m3_highna_full,  good_base)
gccn_lowLWP, rain_lowLWP  = apply_global_filters_gccn(mass_full, accum_rain_lowLWP_full,  gccn_m3_lowLWP_full,  good_base)
gccn_baseturb,   rain_baseturb   = apply_global_filters_gccn(mass_full, accum_rain_baseturb_full,   gccn_m3_baseturb_full,   good_base)
gccn_lownaturb,  rain_lownaturb  = apply_global_filters_gccn(mass_full, accum_rain_lownaturb_full,  gccn_m3_lownaturb_full,  good_base)
gccn_highnaturb, rain_highnaturb = apply_global_filters_gccn(mass_full, accum_rain_highnaturb_full, gccn_m3_highnaturb_full, good_base)
gccn_lowLWPturb, rain_lowLWPturb = apply_global_filters_gccn(mass_full, accum_rain_lowLWPturb_full, gccn_m3_lowLWPturb_full, good_base)
gccn_baseturb2,   rain_baseturb2   = apply_global_filters_gccn(mass_full, accum_rain_baseturb_full2,   gccn_m3_baseturb_full2,   good_base)
gccn_lownaturb2,  rain_lownaturb2  = apply_global_filters_gccn(mass_full, accum_rain_lowna_full2,  gccn_m3_lownaturb_full2,  good_base)
gccn_highnaturb2, rain_highnaturb2 = apply_global_filters_gccn(mass_full, accum_rain_highnaturb2_full, gccn_m3_highnaturb2_full, good_base)
gccn_lowLWPturb2, rain_lowLWPturb2 = apply_global_filters_gccn(mass_full, accum_rain_lowLWPturb2_full, gccn_m3_lowLWPturb2_full, good_base)
cb_colors = {
    "Base": "#0072B2",
    "Low Na": "#009E73",
    "High Na": "#CC79A7",
    "100 g m$^{-2}$ LWP": "#E69F00"
}

gccn_cases = {
    "Base":               {"gccn": gccn_base,   "rain": rain_base,   "color": cb_colors["Base"]},
    "Low Na":             {"gccn": gccn_lowna,  "rain": rain_lowna,  "color": cb_colors["Low Na"]},
    "High Na":            {"gccn": gccn_highna, "rain": rain_highna, "color": cb_colors["High Na"]},
    "100 g m$^{-2}$ LWP": {"gccn": gccn_lowLWP, "rain": rain_lowLWP, "color": cb_colors["100 g m$^{-2}$ LWP"]},
}

gccn_cases_turb = {
    "Base + Turb":               {"gccn": gccn_baseturb,   "rain": rain_baseturb,   "color": cb_colors["Base"]},
    "Low Na + Turb":             {"gccn": gccn_lownaturb,  "rain": rain_lownaturb,  "color": cb_colors["Low Na"]},
    "High Na + Turb":            {"gccn": gccn_highnaturb, "rain": rain_highnaturb, "color": cb_colors["High Na"]},
    "100 g m$^{-2}$ LWP + Turb": {"gccn": gccn_lowLWPturb, "rain": rain_lowLWPturb, "color": cb_colors["100 g m$^{-2}$ LWP"]},

    "High Turbulence Base":               {"gccn": gccn_baseturb2,   "rain": rain_baseturb2,   "color": cb_colors["Base"]},
    "High Turbulence Low Na":             {"gccn": gccn_lownaturb2,  "rain": rain_lownaturb2,  "color": cb_colors["Low Na"]},
    "High Turbulence High Na":            {"gccn": gccn_highnaturb2, "rain": rain_highnaturb2, "color": cb_colors["High Na"]},
    "High Turbulence 100 g m$^{-2}$ LWP": {"gccn": gccn_lowLWPturb2, "rain": rain_lowLWPturb2, "color": cb_colors["100 g m$^{-2}$ LWP"]},
}
def log_binned_median(x, y, nbins=20, min_per_bin=5):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
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
    g = np.asarray(case["gccn"], dtype=float).ravel()
    r = np.asarray(case["rain"], dtype=float).ravel()
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
    g = np.asarray(case["gccn"], dtype=float).ravel()
    r = np.asarray(case["rain"], dtype=float).ravel()
    assert len(g) == len(r), f"Turb mismatch for {label}: gccn={len(g)} rain={len(r)}"

    base_label, turb_tag = split_turb_label(label)
    is_high = (turb_tag == "High Turbulence")

    x_med, y_med = log_binned_median(g, r, nbins=20, min_per_bin=5)
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
plt.xlabel("GCCN concentration (m$^{-3}$)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")
plt.title(
    "BCB January–June 2022\n"
    "Base 385 g m$^{-2}$ LWP\n"
    "Less than 100 µg/m$^3$ mass",
    fontsize=18,
    fontweight="bold"
)
plt.legend(ncol=1, fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1))
plt.grid(alpha=0.3)
plt.xticks(fontsize=14, fontweight="bold")
plt.yticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# %%
#2 panel paper figure for mass vs rain and gccn vs rain turbulence vs no turbulence
def split_turb_label(label):
    if label.startswith("High Turbulence "):
        return label.replace("High Turbulence ", ""), "High Turbulence"
    if label.startswith("High Turbulence"):
        return label.replace("High Turbulence", "").strip(), "High Turbulence"
    return label.replace(" + Turb", ""), "Turbulence"


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
ax1.set_ylim(bottom=1e-2)
ax1.set_xlabel("GCCN concentration (m$^{-3}$)", fontsize=15, fontweight="bold")
ax1.set_ylabel("Accumulated Rain (mm)", fontsize=15, fontweight="bold")
ax1.set_title("(a) GCCN Concentration", fontsize=15, fontweight="bold")
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
ax2.set_xlabel("GCCN Mass (µg m$^{-3}$)", fontsize=15, fontweight="bold")
ax2.set_title("(b) GCCN Mass", fontsize=15, fontweight="bold")
ax2.grid(alpha=0.3)
handles, labels = ax1.get_legend_handles_labels()
uniq = dict(zip(labels, handles))
fig.legend(
    uniq.values(), uniq.keys(),
    loc="center left",
    bbox_to_anchor=(0.8, 0.5),
    fontsize=11,
    frameon=False
)

fig.suptitle(
    "Precipiation as a function of mass and concentration\nLog-binned Median Curves",
    fontsize=15,
    fontweight="bold",
    y=0.98
)
for ax in axes:
    ax.tick_params(labelsize=15, width=2, length=6)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
plt.tight_layout(rect=[0, 0, 0.82, 1])
plt.show()
#save as a pdf
fig.savefig("model_noscatter.pdf", dpi=300)
# %%
#adding full scatter around base no turb
def split_turb_label(label):
    if label.startswith("High Turbulence "):
        return label.replace("High Turbulence ", ""), "High Turbulence"
    if label.startswith("High Turbulence"):
        return label.replace("High Turbulence", "").strip(), "High Turbulence"
    return label.replace(" + Turb", ""), "Turbulence"


fig, axes = plt.subplots(1, 2, figsize=(10, 7), sharey=True)
ax1, ax2 = axes
for label, case in gccn_cases.items():
    g = np.asarray(case["gccn"], dtype=float).ravel()
    r = np.asarray(case["rain"], dtype=float).ravel()
    x_med, y_med = log_binned_median(g, r)
    if x_med.size == 0:
        continue
    ax1.plot(x_med, y_med, color=case["color"], lw=3, ls="-",
            label=f"{label} (No Turbulence)")
    

for label, case in gccn_cases_turb.items():
    g = np.asarray(case["gccn"], dtype=float).ravel()
    r = np.asarray(case["rain"], dtype=float).ravel()
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
for label, case in mass_cases.items():
    m = np.asarray(case["mass"], dtype=float).ravel()
    r = np.asarray(case["rain"], dtype=float).ravel()
    x_med, y_med = log_binned_median(m, r)
    if x_med.size == 0:
        continue
    ax2.plot(x_med, y_med, color=case["color"], lw=3, ls="-")

for label, case in mass_cases_turb.items():
    m = np.asarray(case["mass"], dtype=float).ravel()
    r = np.asarray(case["rain"], dtype=float).ravel()
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
ax1.set_xscale("log")
ax2.set_xscale("log")
xlim1 = ax1.get_xlim()
xlim2 = ax2.get_xlim()
g_base = np.asarray(gccn_cases["Base"]["gccn"], dtype=float).ravel()
r_base = np.asarray(gccn_cases["Base"]["rain"], dtype=float).ravel()
msk_g = (g_base >= xlim1[0]) & (g_base <= xlim1[1])
ax1.scatter(
    g_base[msk_g], r_base[msk_g],
    s=18, alpha=0.25,
    color=gccn_cases["Base"]["color"],
    edgecolor="none",
    zorder=1,
    label="_nolegend_"
)

m_base = np.asarray(mass_cases["Base"]["mass"], dtype=float).ravel()
r_base2 = np.asarray(mass_cases["Base"]["rain"], dtype=float).ravel()
msk_m = (m_base >= xlim2[0]) & (m_base <= xlim2[1])
ax2.scatter(
    m_base[msk_m], r_base2[msk_m],
    s=18, alpha=0.45,
    color=mass_cases["Base"]["color"],
    edgecolor="none",
    zorder=1,
    label="_nolegend_"
)
ax1.set_xlim(xlim1)
ax2.set_xlim(xlim2)
ax1.set_yscale("log")
ax2.set_yscale("log")
ax1.set_ylim(bottom=1e-2)
ax1.set_xlabel("GCCN concentration (m$^{-3}$)", fontsize=17, fontweight="bold")
ax1.set_ylabel("Accumulated Rain (mm)", fontsize=17, fontweight="bold")
ax1.set_title("(a) GCCN Concentration", fontsize=17, fontweight="bold")
ax1.grid(alpha=0.3)
ax2.set_xlabel("GCCN Mass (µg m$^{-3}$)", fontsize=17, fontweight="bold")
ax2.set_title("(b) GCCN Mass", fontsize=17, fontweight="bold")
ax2.grid(alpha=0.3)
handles, labels = ax1.get_legend_handles_labels()
uniq = dict(zip(labels, handles))
fig.legend(
    uniq.values(), uniq.keys(),
    loc="center left",
    bbox_to_anchor=(0.8, 0.5),
    fontsize=11,
    frameon=False
)

fig.suptitle(
    "Precipiation as a function of mass and concentration\nLog-binned Median Curves",
    fontsize=17,
    fontweight="bold",
    y=0.98
)

for ax in axes:
    ax.tick_params(labelsize=15, width=2, length=6)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")

plt.tight_layout(rect=[0, 0, 0.82, 1])
plt.show()
# %%
##adding high na no turb scatter too
def split_turb_label(label):
    if label.startswith("High Turbulence "):
        return label.replace("High Turbulence ", ""), "High Turbulence"
    if label.startswith("High Turbulence"):
        return label.replace("High Turbulence", "").strip(), "High Turbulence"
    return label.replace(" + Turb", ""), "Turbulence"


fig, axes = plt.subplots(1, 2, figsize=(10, 7), sharey=True)
ax1, ax2 = axes
for label, case in gccn_cases.items():
    g = np.asarray(case["gccn"], dtype=float).ravel()
    r = np.asarray(case["rain"], dtype=float).ravel()
    x_med, y_med = log_binned_median(g, r)
    if x_med.size == 0:
        continue
    ax1.plot(
        x_med, y_med,
        color=case["color"], lw=3, ls="-",
        label=f"{label} (No Turbulence)"
    )

for label, case in gccn_cases_turb.items():
    g = np.asarray(case["gccn"], dtype=float).ravel()
    r = np.asarray(case["rain"], dtype=float).ravel()
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

for label, case in mass_cases.items():
    m = np.asarray(case["mass"], dtype=float).ravel()
    r = np.asarray(case["rain"], dtype=float).ravel()
    x_med, y_med = log_binned_median(m, r)
    if x_med.size == 0:
        continue
    ax2.plot(x_med, y_med, color=case["color"], lw=3, ls="-")

for label, case in mass_cases_turb.items():
    m = np.asarray(case["mass"], dtype=float).ravel()
    r = np.asarray(case["rain"], dtype=float).ravel()
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
ax1.set_xscale("log")
ax2.set_xscale("log")

xlim1 = ax1.get_xlim()
xlim2 = ax2.get_xlim()
g_base = np.asarray(gccn_cases["Base"]["gccn"], dtype=float).ravel()
r_base = np.asarray(gccn_cases["Base"]["rain"], dtype=float).ravel()
msk_g = (g_base >= xlim1[0]) & (g_base <= xlim1[1])
ax1.scatter(
    g_base[msk_g], r_base[msk_g],
    s=18, alpha=0.25,
    color=gccn_cases["Base"]["color"],
    edgecolor="none",
    zorder=1,
    label="_nolegend_"
)
g_high = np.asarray(gccn_cases["High Na"]["gccn"], dtype=float).ravel()
r_high = np.asarray(gccn_cases["High Na"]["rain"], dtype=float).ravel()
msk_gh = (g_high >= xlim1[0]) & (g_high <= xlim1[1])
ax1.scatter(
    g_high[msk_gh], r_high[msk_gh],
    s=18, alpha=0.25,
    color=gccn_cases["High Na"]["color"],
    edgecolor="none",
    zorder=1,
    label="_nolegend_"
)
m_base = np.asarray(mass_cases["Base"]["mass"], dtype=float).ravel()
r_base2 = np.asarray(mass_cases["Base"]["rain"], dtype=float).ravel()
msk_m = (m_base >= xlim2[0]) & (m_base <= xlim2[1])
ax2.scatter(
    m_base[msk_m], r_base2[msk_m],
    s=18, alpha=0.45,
    color=mass_cases["Base"]["color"],
    edgecolor="none",
    zorder=1,
    label="_nolegend_"
)
m_high = np.asarray(mass_cases["High Na"]["mass"], dtype=float).ravel()
r_high2 = np.asarray(mass_cases["High Na"]["rain"], dtype=float).ravel()
msk_mh = (m_high >= xlim2[0]) & (m_high <= xlim2[1])
ax2.scatter(
    m_high[msk_mh], r_high2[msk_mh],
    s=18, alpha=0.45,
    color=mass_cases["High Na"]["color"],
    edgecolor="none",
    zorder=1,
    label="_nolegend_"
)
ax1.set_xlim(xlim1)
ax2.set_xlim(xlim2)
ax1.set_yscale("log")
ax2.set_yscale("log")
ax1.set_ylim(bottom=1e-2)

ax1.set_xlabel("GCCN concentration (m$^{-3}$)", fontsize=17, fontweight="bold")
ax1.set_ylabel("Accumulated Rain (mm)", fontsize=17, fontweight="bold")
ax1.set_title("(a) GCCN Concentration", fontsize=17, fontweight="bold")
ax1.grid(alpha=0.3)

ax2.set_xlabel("GCCN Mass (µg m$^{-3}$)", fontsize=17, fontweight="bold")
ax2.set_title("(b) GCCN Mass", fontsize=17, fontweight="bold")
ax2.grid(alpha=0.3)

handles, labels = ax1.get_legend_handles_labels()
uniq = dict(zip(labels, handles))
fig.legend(
    uniq.values(), uniq.keys(),
    loc="center left",
    bbox_to_anchor=(0.8, 0.5),
    fontsize=11,
    frameon=False
)

fig.suptitle(
    "Precipiation as a function of mass and concentration\nLog-binned Median Curves",
    fontsize=15,
    fontweight="bold",
    y=0.98
)
for ax in axes:
    ax.tick_params(labelsize=15, width=2, length=6)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")

plt.tight_layout(rect=[0, 0, 0.82, 1])
plt.show()
#save figure as pdf
fig.savefig("model_withscatter.pdf", dpi=300, bbox_inches="tight")
# %%
##adding low na no turb scatter too
def split_turb_label(label):
    if label.startswith("High Turbulence "):
        return label.replace("High Turbulence ", ""), "High Turbulence"
    if label.startswith("High Turbulence"):
        return label.replace("High Turbulence", "").strip(), "High Turbulence"
    return label.replace(" + Turb", ""), "Turbulence"


fig, axes = plt.subplots(1, 2, figsize=(10, 7), sharey=True)
ax1, ax2 = axes
for label, case in gccn_cases.items():
    g = np.asarray(case["gccn"], dtype=float).ravel()
    r = np.asarray(case["rain"], dtype=float).ravel()
    x_med, y_med = log_binned_median(g, r)
    if x_med.size == 0:
        continue
    ax1.plot(
        x_med, y_med,
        color=case["color"], lw=3, ls="-",
        label=f"{label} (No Turbulence)"
    )
for label, case in gccn_cases_turb.items():
    g = np.asarray(case["gccn"], dtype=float).ravel()
    r = np.asarray(case["rain"], dtype=float).ravel()
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

for label, case in mass_cases.items():
    m = np.asarray(case["mass"], dtype=float).ravel()
    r = np.asarray(case["rain"], dtype=float).ravel()
    x_med, y_med = log_binned_median(m, r)
    if x_med.size == 0:
        continue
    ax2.plot(x_med, y_med, color=case["color"], lw=3, ls="-")

for label, case in mass_cases_turb.items():
    m = np.asarray(case["mass"], dtype=float).ravel()
    r = np.asarray(case["rain"], dtype=float).ravel()
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

ax1.set_xscale("log")
ax2.set_xscale("log")
xlim1 = ax1.get_xlim()
xlim2 = ax2.get_xlim()

for k in ["Base", "High Na", "Low Na"]:
    g = np.asarray(gccn_cases[k]["gccn"], dtype=float).ravel()
    r = np.asarray(gccn_cases[k]["rain"], dtype=float).ravel()
    msk = np.isfinite(g) & np.isfinite(r) & (g > 0) & (r > 0) & (g >= xlim1[0]) & (g <= xlim1[1])
    ax1.scatter(
        g[msk], r[msk],
        s=18, alpha=0.25,
        color=gccn_cases[k]["color"],
        edgecolor="none",
        zorder=1,
        label="_nolegend_"
    )

for k in ["Base", "High Na", "Low Na"]:
    m = np.asarray(mass_cases[k]["mass"], dtype=float).ravel()
    r = np.asarray(mass_cases[k]["rain"], dtype=float).ravel()
    msk = np.isfinite(m) & np.isfinite(r) & (m > 0) & (r > 0) & (m >= xlim2[0]) & (m <= xlim2[1])
    ax2.scatter(
        m[msk], r[msk],
        s=18, alpha=0.45,
        color=mass_cases[k]["color"],
        edgecolor="none",
        zorder=1,
        label="_nolegend_"
    )
ax1.set_xlim(xlim1)
ax2.set_xlim(xlim2)
ax1.set_yscale("log")
ax2.set_yscale("log")

ax1.set_ylim(bottom=1e-2)
ax1.set_xlabel("GCCN concentration (m$^{-3}$)", fontsize=17, fontweight="bold")
ax1.set_ylabel("Accumulated Rain (mm)", fontsize=17, fontweight="bold")
ax1.set_title("(a) GCCN Concentration", fontsize=17, fontweight="bold")
ax1.grid(alpha=0.3)
ax2.set_xlabel("GCCN Mass (µg m$^{-3}$)", fontsize=17, fontweight="bold")
ax2.set_title("(b) GCCN Mass", fontsize=17, fontweight="bold")
ax2.grid(alpha=0.3)
for ax in axes:
    ax.tick_params(axis="both", labelsize=17, width=2, length=6)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
from collections import OrderedDict

handles, labels = ax1.get_legend_handles_labels()
uniq = OrderedDict()
for h, lab in zip(handles, labels):
    if lab == "_nolegend_":
        continue
    if lab not in uniq:
        uniq[lab] = h
no_h, no_l = [], []
tb_h, tb_l = [], []
hi_h, hi_l = [], []

for lab, h in uniq.items():
    if "(No Turbulence)" in lab:
        no_h.append(h); no_l.append(lab.replace(" (No Turbulence)", ""))
    elif "(Turbulence)" in lab and "(High Turbulence)" not in lab:
        tb_h.append(h); tb_l.append(lab.replace(" (Turbulence)", ""))
    elif "(High Turbulence)" in lab:
        hi_h.append(h); hi_l.append(lab.replace(" (High Turbulence)", ""))
leg1 = fig.legend(
    no_h, no_l,
    title = r"$\mathbf{\epsilon = 0\ cm^2\ s^{-3}}$",
    loc="center left",
    bbox_to_anchor=(0.89, 0.60),
    fontsize=14,
    frameon=False,
)
leg1.get_title().set_fontweight("bold")

leg2 = fig.legend(
    tb_h, tb_l,
    title = r"$\mathbf{\epsilon = 10\ cm^2\ s^{-3}}$",
    loc="center left",
    bbox_to_anchor=(0.89 + 0.22, 0.60),
    fontsize=14,
    frameon=False,
)
leg2.get_title().set_fontweight("bold")

leg3 = fig.legend(
    hi_h, hi_l,
    title=r"$\mathbf{\epsilon = 50\ cm^2\ s^{-3}}$",
    loc="center left",
    bbox_to_anchor=(0.89 + 0.44, 0.60),
    fontsize=14,
    frameon=False,
)
leg3.get_title().set_fontweight("bold")
plt.show()
# save figure as pdf
fig.savefig("model_withscatter.pdf", dpi=300, bbox_inches="tight")
# %%
