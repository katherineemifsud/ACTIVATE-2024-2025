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
accum_rain_lownaturb = (
    np.max(LWP_lownaturb, axis=1) - LWP_lownaturb[:, -1]
).copy()
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
    },
    "High Na + Turb": {
        "r": r_dry_highnaturb,
        "n0": n0_r_highnaturb,
        "rain": accum_rain_highnaturb,
        "color": "tab:red",
        "marker": "^"
    },
    "100 g m$^{-2}$ LWP + Turb": {
        "r": r_dry_lowLWP,
        "n0": n0_r_lowLWP,
        "rain": accum_rain_lowLWP,
        "color": "tab:orange",
        "marker": "D"
}
}
for k, v in turb_cases.items():
    print(k, len(v["rain"]))
plt.figure(figsize=(7, 5))

for label, case in turb_cases.items():

    mask = case["r"] > 1e-6 
    gccn = np.sum(case["n0"][:, mask], axis=1)
    rain = case["rain"]
    valid = (gccn > 0) & (rain > 0)
    gccn = gccn[valid]
    rain = rain[valid]

    # logx = np.log10(gccn)
    # logy = np.log10(rain)

    # slope, intercept, R, _, _ = linregress(logx, logy)

    # xfit = np.logspace(np.log10(gccn.min()), np.log10(gccn.max()), 200)
    # yfit = 10 ** (intercept + slope * np.log10(xfit))

    # plt.plot(
    #     xfit, yfit,
    #     "--",
    #     lw=3,
    #     color=case["color"],
    #     label=f"{label} (s={slope:.2f}, R={R:.2f})"
    # )

    # print(f"{label}: slope={slope:.2f}, R={R:.2f}, R²={R**2:.2f}")
plt.figure(figsize=(7, 5))

for label, case in turb_cases.items():
    mask = case["r"] > 1e-6
    gccn = np.sum(case["n0"][:, mask], axis=1)
    rain = case["rain"]

    valid = (gccn > 0) & (rain > 0) & np.isfinite(gccn) & np.isfinite(rain)
    gccn = gccn[valid]
    rain = rain[valid]

    plt.scatter(
        gccn, rain,
        s=90,
        marker=case.get("marker", "o"), 
        color=case["color"],
        edgecolor="k",
        alpha=0.85,
        label=label
    )

plt.xscale("log")
plt.yscale("log")
plt.xlabel("GCCN concentration (m$^{-3}$)", fontsize=16, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=16, fontweight="bold")
plt.title(
    "BCB January–June 2022\n385 g m$^{-2}$ LWP\nGCCN vs Accumulated Rainfall\nTurbulence",
    fontsize=18, fontweight="bold"
)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.xticks(fontsize=14, fontweight="bold")
plt.yticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

#%%

# %%
# saving this as a .csv
rows = []

for label, case in turb_cases.items():

    mask = case["r"] > 1e-6
    gccn = np.sum(case["n0"][:, mask], axis=1)
    rain = case["rain"]

    valid = (gccn > 0) & (rain > 0)
    gccn = gccn[valid]
    rain = rain[valid]

    for i, (g, r) in enumerate(zip(gccn, rain), start=1):
        rows.append({
            "Case": label,
            "Leg": i,
            "GCCN_m3": g,
            "Accumulated_Rain_mm": r,
            "log10_GCCN": np.log10(g),
            "log10_Rain": np.log10(r)
        })

df_gccn_rain_turb = pd.DataFrame(rows)

print("Total rows saved:", len(df_gccn_rain_turb))
print(df_gccn_rain_turb.head())
save_path = (
    "/home/disk/eos4/kathem24/activate/data/CAS/"
    "gccn_rain_BCB_385g_LWP_turbulence.csv"
)

df_gccn_rain_turb.to_csv(save_path, index=False)

print("Saved to:")
print(save_path)
#%%
#thinning the data for better visualization, use every 5th point for each of the 4 cases
plt.figure(figsize=(7, 5))
for label, case in turb_cases.items():
    mask = case["r"] > 1e-6   
    gccn = np.sum(case["n0"][:, mask], axis=1)
    rain = case["rain"]

    valid = (gccn > 0) & (rain > 0) & np.isfinite(gccn) & np.isfinite(rain)
    gccn = gccn[valid]
    rain = rain[valid]
    thin_gccn = gccn[::5]
    thin_rain = rain[::5]

    plt.scatter(
        thin_gccn, thin_rain,
        s=90,
        marker=case.get("marker", "o"),
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
    "BCB January–June 2022\n385 g m$^{-2}$ LWP\nGCCN vs Accumulated Rainfall\nTurbulence (95 legs)",
    fontsize=18, fontweight="bold"
)
plt.legend(fontsize=11, frameon=True)
plt.grid(alpha=0.3)
plt.xticks(fontsize=14, fontweight="bold")
plt.yticks(fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()  

# %%
#mass vs accumulated rain for all cases no turbulence

mass = np.array(all_mass_values)  
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
#mass vs accumulated rain for all cases no turbulence with scatter points
          
mass = np.array(all_mass_values)  
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
    "BCB January–June 2022\n385 g m$^{-2}$ LWP\nDry GCCN Mass vs Accumulated Rainfall\nTurbulence",
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
    "BCB January–June 2022\n385 g m$^{-2}$ LWP\n GCCN Mass vs Accumulated Rainfall\nTurbulence (95 legs)",
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
#saving mass vs accumulated rain turbulence as .csv
rows = []

for label, case in mass_cases.items():
    m = case["mass"]
    r = case["rain"]
    valid = (m > 0) & (r > 0)
    m = m[valid]
    r = r[valid]

    for i, (mass_val, rain_val) in enumerate(zip(m, r), start=1):
        rows.append({
            "Case": label,
            "Leg": i,
            "Dry_GCCN_Mass_ug_m3": mass_val,
            "Accumulated_Rain_mm": rain_val,
            "log10_Dry_GCCN_Mass": np.log10(mass_val),
            "log10_Accumulated_Rain": np.log10(rain_val)
        })

df_mass_rain_turb = pd.DataFrame(rows)
save_path = (
    "/home/disk/eos4/kathem24/activate/data/CAS/"
    "mass_rain_BCB_385g_LWP_turbulence.csv"
)

df_mass_rain_turb.to_csv(save_path, index=False)
print("Saved to:")
print(save_path)
df_check = pd.read_csv(save_path)
print(df_check.groupby("Case").size())



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
# %%
#combining turbulence and non turbulence gccn concentration vs rain data into one figure for easier comparison using the same color for each case but different markers for turb vs no turb
plt.figure(figsize=(7, 5))
for label, case in turb_cases.items():
    mask = case["r"] > 1e-6
    gccn = np.sum(case["n0"][:, mask], axis=1)
    rain = case["rain"]
    valid = (gccn > 0) & (rain > 0) & np.isfinite(gccn) & np.isfinite(rain)
    gccn = gccn[valid]
    rain = rain[valid]

    plt.scatter(
        gccn, rain,
        s=90,
        marker=case.get("marker", "o"), 
        color=case["color"],
        edgecolor="k",
        alpha=0.85,
        label=f"{label} (Turbulence)"
    )   
    non_turb_label = label.replace(" + Turb", "")
    non_turb_case = turb_cases.get(non_turb_label)
    if non_turb_case:
        non_turb_mask = non_turb_case["r"] > 1e-6
        non_turb_gccn = np.sum(non_turb_case["n0"][:, non_turb_mask], axis=1)
        non_turb_rain = non_turb_case["rain"]

        valid_non_turb = (non_turb_gccn > 0) & (non_turb_rain > 0) & np.isfinite(non_turb_gccn) & np.isfinite(non_turb_rain)
        non_turb_gccn = non_turb_gccn[valid_non_turb]
        non_turb_rain = non_turb_rain[valid_non_turb]

        plt.scatter(
            non_turb_gccn, non_turb_rain,
            s=90,
            marker="X", 
            color=case["color"],
            edgecolor="k",
            alpha=0.85,
            label=f"{non_turb_label} (No Turbulence)"
        )
non_turb_cases = {
    "Base": {
        "r": r_dry,
        "n0": n0_r,
        "rain": accum_rain_base,      
        "color": "tab:blue",
        "marker": "o"
    },
    "Low Na": {
        "r": r_dry_lowna,
        "n0": n0_r_lowna,
        "rain": accum_rain_lowna,
        "color": "tab:green",
        "marker": "s"
    },
    "High Na": {
        "r": r_dry_highna,
        "n0": n0_r_highna,
        "rain": accum_rain_highna,
        "color": "tab:red",
        "marker": "^"
    },
    "100 g m$^{-2}$ LWP": {
        "r": r_dry_lowLWP,
        "n0": n0_r_lowLWP,
        "rain": accum_rain_lowLWP,
        "color": "tab:orange",
        "marker": "D"
    }
}
plt.figure(figsize=(7, 5))
for label, case in turb_cases.items():
    base_label = label.replace(" + Turb", "").replace(" + Turbulence", "")

    mask = case["r"] > 1e-6
    gccn = np.sum(case["n0"][:, mask], axis=1)
    rain = case["rain"]
    valid = (gccn > 0) & (rain > 0) & np.isfinite(gccn) & np.isfinite(rain)
    plt.scatter(
        gccn[valid], rain[valid],
        s=90,
        marker=case.get("marker", "o"),
        color=case["color"],
        edgecolor="k",
        alpha=0.85,
        label=f"{base_label} (Turb)"
    )
    non_case = non_turb_cases.get(base_label)
    if non_case is not None:
        mask0 = non_case["r"] > 1e-6
        gccn0 = np.sum(non_case["n0"][:, mask0], axis=1)
        rain0 = non_case["rain"]
        valid0 = (gccn0 > 0) & (rain0 > 0) & np.isfinite(gccn0) & np.isfinite(rain0)
        plt.scatter(
            gccn0[valid0], rain0[valid0],
            s=90,
            marker="X",  
            color=non_case["color"],
            edgecolor="k",
            alpha=0.85,
            label=f"{base_label} (No Turb)"
        )

plt.xscale("log")
plt.yscale("log")
plt.xlabel("GCCN concentration (m$^{-3}$)", fontsize=18, fontweight="bold")
plt.ylabel("Accumulated Rain (mm)", fontsize=18, fontweight="bold")
plt.title("BCB January–June 2022\n385 g m$^{-2}$ LWP\nGCCN vs Accumulated Rainfall",
          fontsize=18, fontweight="bold")
plt.legend(ncol=2, fontsize=8)
plt.xticks(fontsize=16, fontweight="bold")
plt.yticks(fontsize=16, fontweight="bold")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %%
