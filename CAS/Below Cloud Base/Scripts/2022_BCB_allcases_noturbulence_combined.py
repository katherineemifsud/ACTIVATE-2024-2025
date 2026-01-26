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
# #all cases no turbulence gccn concentration vs accumulated rain
# cases = {
#     "Base": {
#         "r": r_dry,
#         "n0": n0_r,
#         "LWP": LWP,
#         "color": "tab:blue",
#         "marker": "o"
#     },
#     "Low Na": {
#         "r": r_dry_lowna,
#         "n0": n0_r_lowna,
#         "LWP": LWP_lowna,
#         "color": "tab:green",
#         "marker": "s"
#     },
#     "High Na": {
#         "r": r_dry_highna,
#         "n0": n0_r_highna,
#         "LWP": LWP_highna,
#         "color": "tab:red",
#         "marker": "^"
#     },
#     "100 g m$^{-2}$ LWP": {
#         "r": r_dry_lowLWP,
#         "n0": n0_r_lowLWP,
#         "LWP": LWP_lowLWP,
#         "color": "tab:orange",
#         "marker": "D"
#     }
# }
# plt.figure(figsize=(7, 5))
# for label, case in cases.items():
#     mask = case["r"] > 1e-6     # radius > 1 µm
#     gccn = np.sum(case["n0"][:, mask], axis=1)
#     rain = np.max(case["LWP"], axis=1) - case["LWP"][:, -1]
#     plt.scatter(
#         gccn, rain,
#         s=90,
#         marker=case["marker"],
#         color=case["color"],
#         edgecolor="k",
#         alpha=0.85,
#         label=f"{label}"
#     )
#     # logx = np.log10(gccn)
#     # logy = np.log10(rain)
#     # slope, intercept, r, _, _ = linregress(logx, logy)

#     # xfit = np.logspace(np.log10(gccn.min()), np.log10(gccn.max()), 100)
#     # yfit = 10 ** (intercept + slope * np.log10(xfit))

#     # plt.plot(
#     #     xfit, yfit,
#     #     color=case["color"],
#     #     linestyle="--",
#     #     linewidth=2,
#     #     label=f"{label} fit (s={slope:.2f}, R={r:.2f})"
#     # )

#     # print(f"{label}: R = {r:.3f}, R² = {r**2:.3f}")
# plt.xscale("log")
# plt.yscale("log")
# plt.xlabel("GCCN concentration (m$^{-3}$)", fontsize=16, fontweight="bold")
# plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")
# plt.title(
#     "BCB January–June 2022\n385 g m$^{-2}$ LWP\nGCCN vs Accumulated Rainfall\nNo Turbulence",
#     fontsize=18,
#     fontweight="bold"
# )
# plt.legend(fontsize=11, frameon=True)
# plt.grid(alpha=0.3)
# plt.xticks(fontsize=14, fontweight="bold")
# plt.yticks(fontsize=14, fontweight="bold")
# plt.tight_layout()
# plt.show()
# # %%
# #thinning the data for better visualization, use every 5th point for each of the 4 cases
# plt.figure(figsize=(7, 5))
# for label, case in cases.items():
#     mask = case["r"] > 1e-6     # radius > 1 µm
#     gccn = np.sum(case["n0"][:, mask], axis=1)
#     rain = np.max(case["LWP"], axis=1) - case["LWP"][:, -1]

#     # Thinning the data by taking every 5th point
#     thin_gccn = gccn[::5]
#     thin_rain = rain[::5]

#     plt.scatter(
#         thin_gccn, thin_rain,
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
#     "BCB January–June 2022\n385 g m$^{-2}$ LWP\nGCCN vs Accumulated Rainfall\nNo Turbulence (95 legs)",
#     fontsize=18,
#     fontweight="bold"
# )
# plt.legend(fontsize=11, frameon=True)
# plt.grid(alpha=0.3)
# plt.xticks(fontsize=14, fontweight="bold")
# plt.yticks(fontsize=14, fontweight="bold")
# plt.tight_layout()
# plt.show()
# # %%
# #saving this as a .csv
# rows = []

# for case_name, case in cases.items():

#     mask = case["r"] > 1e-6   # radius > 0.5 µm
#     gccn = np.sum(case["n0"][:, mask], axis=1)
#     rain = np.max(case["LWP"], axis=1) - case["LWP"][:, -1]

#     for leg, (g, r) in enumerate(zip(gccn, rain), start=1):
#         rows.append({
#             "Case": case_name,
#             "Leg": leg,
#             "GCCN_m3": g,
#             "Accumulated_Rain_mm": r,
#             "log10_GCCN": np.log10(g),
#             "log10_Rain": np.log10(r)
#         })

# df_gccn_rain = pd.DataFrame(rows)
# save_path = "/home/disk/eos4/kathem24/activate/data/CAS/gccn_rain_no_turbulence_alldata.csv"
# df_gccn_rain.to_csv(save_path, index=False)

# print(f"Saved {len(df_gccn_rain)} rows to:")
# print(save_path)
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
    "BCB January–June 2022\n385 g m$^{-2}$ LWP\nLess than 100 µg m$^{-3}$ GCCN\nNo Turbulence",
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
for label, case in mass_cases.items():

    m = case["mass"]
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
plt.xlabel("Dry GCCN Mass (µg m$^{-3}$)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")
plt.title(
    "BCB January–June 2022\n385 g m$^{-2}$ LWP\nLess than 100 µg m$^{-3}$ GCCN\nNo Turbulence (95 legs)",
    fontsize=18,
    fontweight="bold"
)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# %%
# #saving this as a .csv
# rows = []
# mass_cases = {
#     "Base": {
#         "mass": mass,
#         "rain": accum_rain_base
#     },
#     "Low Na": {
#         "mass": mass,
#         "rain": accum_rain_lowna
#     },
#     "High Na": {
#         "mass": mass,
#         "rain": accum_rain_highna
#     },
#     "100 g m$^{-2}$ LWP": {
#         "mass": mass,
#         "rain": accum_rain_lowLWP
#     }
# }

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

# df_mass_rain_noturb = pd.DataFrame(rows)

# print("Total rows saved:", len(df_mass_rain_noturb))
# print(df_mass_rain_noturb.head())
# save_path = (
#     "/home/disk/eos4/kathem24/activate/data/CAS/"
#     "mass_rain_BCB_385g_LWP_noTurbulence.csv"
# )

# df_mass_rain_noturb.to_csv(save_path, index=False)

# print("Saved to:")
# print(save_path)
# df_check = pd.read_csv(save_path)
# print(df_check.groupby("Case").size())


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
#thinning the data for better visualization, use every 5th point for each of the 4 cases
plt.figure(figsize=(7, 5))
for label, case in gccn_cases.items():

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
    "BCB January–June 2022\n385 g m$^{-2}$ LWP\nLess than 100 µg m$^{-3}$ GCCN\nNo Turbulence (95 legs)",
    fontsize=18,
    fontweight="bold"
)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# %%
