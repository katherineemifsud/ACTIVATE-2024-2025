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
#gccn vs accumulated rain for all cases no turbulence with scatter points
# After computing each accum_rain
        
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
# # saving this as a .csv
# rows = []

# for label, case in turb_cases.items():

#     mask = case["r"] > 1e-6
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
#%%
# #thinning the data for better visualization, use every 5th point for each of the 4 cases
# plt.figure(figsize=(7, 5))
# for label, case in turb_cases.items():
#     mask = case["r"] > 1e-6   
#     gccn = np.sum(case["n0"][:, mask], axis=1)
#     rain = case["rain"]

#     valid = (gccn > 0) & (rain > 0) & np.isfinite(gccn) & np.isfinite(rain)
#     gccn = gccn[valid]
#     rain = rain[valid]
#     thin_gccn = gccn[::5]
#     thin_rain = rain[::5]

#     plt.scatter(
#         thin_gccn, thin_rain,
#         s=90,
#         marker=case.get("marker", "o"),
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
#     "BCB January–June 2022\n385 g m$^{-2}$ LWP\nGCCN vs Accumulated Rainfall\nTurbulence (95 legs)",
#     fontsize=18, fontweight="bold"
# )
# plt.legend(fontsize=11, frameon=True)
# plt.grid(alpha=0.3)
# plt.xticks(fontsize=14, fontweight="bold")
# plt.yticks(fontsize=14, fontweight="bold")
# plt.tight_layout()
# plt.show()  

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
    #     label=f"{label} (s={slope:.2f}, R={R:.2f})"
    # )

    # print(f"{label}: slope={slope:.2f}, R={R:.2f}, R²={R**2:.2f}")
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
plt.figure(figsize=(7, 5))
for label, case in mass_cases.items():
    m = np.asarray(case["mass"])
    r = np.asarray(case["rain"])
    assert len(m) == len(r), f"No-turb mismatch for {label}: mass={len(m)} rain={len(r)}"
    valid = np.isfinite(m) & np.isfinite(r) & (m > 0) & (r > 0)
    m = m[valid]
    r = r[valid]
    plt.plot(
    m, r,
    marker=case["marker"],
    color=case["color"],
    linewidth=2.5,
    linestyle="-",         
    alpha=0.85,
    label=f"{label} (No Turbulence)"
)
for label, case in mass_cases_turb.items():
    m = np.asarray(case["mass"])
    r = np.asarray(case["rain"])

    assert len(m) == len(r), f"Turb mismatch for {label}: mass={len(m)} rain={len(r)}"
    valid = np.isfinite(m) & np.isfinite(r) & (m > 0) & (r > 0)
    m = m[valid]
    r = r[valid]
    plt.plot(
    m, r,
    marker=case["marker"],
    color=case["color"],
    linewidth=2.5,
    linestyle="--",        
    alpha=0.85,
    label=f"{label} (Turbulence)"
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
plt.legend(ncol=2, fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
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
for label, case in mass_cases_turb.items():
    m = np.asarray(case["mass"], dtype=float)
    r = np.asarray(case["rain"], dtype=float)
    assert len(m) == len(r), f"Turb mismatch for {label}: mass={len(m)} rain={len(r)}"
    x_med, y_med = log_binned_median(m, r, nbins=20, min_per_bin=5)
    if x_med.size == 0:
        continue
    plt.plot(
        x_med, y_med,
        color=case["color"],
        linewidth=3,
        linestyle="--",
        alpha=0.95,
        label=f"{label} (Turbulence)"
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
plt.legend(ncol=2, fontsize=9)
plt.grid(alpha=0.3)
plt.xticks(fontsize=14, fontweight="bold")
plt.yticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
# %%
#combining turbulence and non turbulence gccn concentration vs rain data into one figure for easier comparison using the same color for each case but different markers for turb vs no turb
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
    x_med, y_med = log_binned_median(g, r, nbins=20, min_per_bin=5)
    if x_med.size == 0:
        continue
    plt.plot(
        x_med, y_med,
        color=case["color"],
        linewidth=3,
        linestyle="--",
        alpha=0.95,
        label=f"{label} (Turbulence)"
    )
plt.xscale("log")
plt.yscale("log")
plt.xlabel("GCCN concentration (m$^{-3}$)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")
plt.title(
    "BCB January–June 2022\n"
    "385 g m$^{-2}$ LWP\n"
    "Log-binned Median Curves\n"
    "Less than 100 µg/m$^3$ mass",
    fontsize=18,
    fontweight="bold"
)
plt.legend(ncol=2, fontsize=9)
plt.grid(alpha=0.3)
plt.xticks(fontsize=16, fontweight="bold")
plt.yticks(fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()


# %%
