#%%
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import datetime
import pathlib
import statistics
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy.ma as ma
import glob
import os
import re
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
#%%
#import the saved average size distributions for each of the three years and the three instruments

#CAS 2020-2022
average_distribution_paths = {
    "CAS": {
        2020: "/home/disk/eos4/kathem24/activate/data/2020/CAS/Average_Dry_Size_Distribution_beforemass.csv",
        2021: "/home/disk/eos4/kathem24/activate/data/2021/CAS/Average_Dry_Size_Distribution_beforemass.csv",
        2022: "/home/disk/eos4/kathem24/activate/data/2021/CAS/Average_Dry_Size_Distribution_beforemass2022CAS.csv"
    },
#CDP 2020-2022
    "CDP": {
        2020: "/home/disk/eos4/kathem24/activate/data/2020/CDP/1Hz/Average_Dry_Size_Distribution_beforemass_CDP.csv",
        2021: "/home/disk/eos4/kathem24/activate/data/2021/CDP/1Hz/Average_Dry_Size_Distribution_beforemass_CDP.csv",
        2022: "/home/disk/eos4/kathem24/activate/data/2021/CDP/1Hz/Average_Dry_Size_Distribution_beforemass_2022.csv"
    },
#FCDP 2020-2022
    "FCDP": {
        2020: "/home/disk/eos4/kathem24/activate/data/2020/FCDP/Average_Dry_Size_Distribution_beforemass_FCDP.csv",
        2021: "/home/disk/eos4/kathem24/activate/data/2021/FCDP/Average_Dry_Size_Distribution_beforemass_FCDP.csv",
        2022: "/home/disk/eos4/kathem24/activate/data/2022/FCDP/Average_Dry_Size_Distribution_beforemass_FCDP.csv"
    }
}
# %%
average_distributions = {}
for instrument, yearly_paths in average_distribution_paths.items():
    average_distributions[instrument] = {}
    for year, file_path in yearly_paths.items():
        if not os.path.exists(file_path):
            print(f"File not found: {instrument} {year}")
            print(file_path)
            continue
        distribution_df = pd.read_csv(file_path)
        required_columns = [
            "Dry_Diameter_um",
            "Average_dN_dD_dry",
            "N_profiles"]
        missing_columns = [
            column for column in required_columns
            if column not in distribution_df.columns]
        if missing_columns:
            print(
                f"Skipping {instrument} {year}. "
                f"Missing columns: {missing_columns}")
            continue
        distribution_df["Dry_Diameter_um"] = pd.to_numeric(
            distribution_df["Dry_Diameter_um"],
            errors="coerce")
        
        distribution_df["Average_dN_dD_dry"] = pd.to_numeric(
            distribution_df["Average_dN_dD_dry"],
            errors="coerce")
        
        distribution_df["N_profiles"] = pd.to_numeric(
            distribution_df["N_profiles"],
            errors="coerce")
        average_distributions[instrument][year] = distribution_df
        print(
            f"Loaded {instrument} {year}: "
            f"{len(distribution_df)} diameter bins") 
# %%
def combine_instrument_years(
    average_distributions,
    instrument,
    years=(2020, 2021, 2022)):
    common_bins = None
    combined_sum = None
    combined_count = None
    for year in years:
        df = average_distributions[instrument][year]
        diameter = df[
            "Dry_Diameter_um"
        ].to_numpy(dtype=float)
        average = df[
            "Average_dN_dD_dry"
        ].to_numpy(dtype=float)
        count = pd.to_numeric(
            df["N_profiles"],
            errors="coerce"
        ).fillna(0).to_numpy(dtype=float)
        if common_bins is None:
            common_bins = diameter.copy()
            combined_sum = np.zeros_like(
                common_bins,
                dtype=float )
            combined_count = np.zeros_like(
                common_bins,
                dtype=float )

        else:
            if not np.allclose(
                diameter,
                common_bins,
                equal_nan=True
            ):
                raise ValueError(
                    f"{instrument} {year} has different common bins." )

        valid = (
            np.isfinite(average)
            & np.isfinite(diameter)
            & np.isfinite(count)
            & (count > 0))
        combined_sum[valid] += (
            average[valid] * count[valid] )
        combined_count[valid] += count[valid]
    combined_average = np.divide(
        combined_sum,
        combined_count,
        out=np.full_like(
            combined_sum,
            np.nan,
            dtype=float),
        where=combined_count > 0)
    combined_df = pd.DataFrame({
        "Dry_Diameter_um": common_bins,
        "Average_dN_dD_dry": combined_average,
        "N_profiles": combined_count.astype(int)})
    return combined_df
# %%
combined_CAS = combine_instrument_years(
    average_distributions,
    "CAS")
combined_CDP = combine_instrument_years(
    average_distributions,
    "CDP")
combined_FCDP = combine_instrument_years(
    average_distributions,
    "FCDP")
# %%
plt.figure(figsize=(9, 7))
plt.plot(
    combined_CAS["Dry_Diameter_um"],
    combined_CAS["Average_dN_dD_dry"],
    linewidth=2.5,
    color="black",
    label="CAS")
plt.plot(
    combined_CDP["Dry_Diameter_um"],
    combined_CDP["Average_dN_dD_dry"],
    linewidth=2.5,
    color="blue",
    label="CDP")
plt.plot(
    combined_FCDP["Dry_Diameter_um"],
    combined_FCDP["Average_dN_dD_dry"],
    linewidth=2.5,
    color="green",
    label="FCDP")
plt.xlabel("Dry Bin Center Diameter (μm)",
    fontsize=20,
    fontweight="bold")
plt.ylabel(r"Number Concentration (cm$^{-3}$ $\mu$m$^{-1}$)", fontsize=20, fontweight="bold")
plt.yscale("log")
plt.xlim(2, 25)
plt.ylim(10**-4, 10**0)
plt.xticks(fontsize=18, fontweight="bold")
plt.yticks(fontsize=18, fontweight="bold")
plt.title("Average Below Cloud Base Dry Size Distributions\n2020–2022", fontsize=20,fontweight="bold")
plt.legend(fontsize=15)
plt.tight_layout()
plt.show()
# %%
#mass versus concentration for each instrument 
base_root = "/home/disk/p/kathem24/activate/ACTIVATE-2024-2025"
slope_paths = {
    "CAS": {
        2020: os.path.join(base_root, "CAS/below cloud base/CAS_slope_massLE1002020.pkl"),
        2021: os.path.join(base_root, "CAS/below cloud base/CAS_slope_massLE1002021.pkl"),
        2022: os.path.join(base_root, "CAS/below cloud base/CAS_slope_massLE100.pkl")
    },
    "CDP": {
        2020: os.path.join(base_root, "CDP/below cloud base/CDP_slope_massLE1002020.pkl"),
        2021: os.path.join(base_root, "CDP/below cloud base/CDP_slope_massLE1002021.pkl"),
        2022: os.path.join(base_root, "CDP/below cloud base/CDP_slope_massLE100.pkl")
    },
    "FCDP": {
        2020: os.path.join(base_root, "FCDP/below cloud base/FCDP_slope_massLE1002020.pkl"),
        2021: os.path.join(base_root, "FCDP/below cloud base/FCDP_slope_massLE1002021.pkl"),
        2022: os.path.join(base_root, "FCDP/below cloud base/FCDP_slope_massLE1002022.pkl")
    }
}
# %%
base_dir = "/home/disk/p/kathem24/activate/ACTIVATE-2024-2025/CDP/below cloud base"
data_paths = {
    "CAS": {
        2020: {
            "slope": os.path.join(base_dir, "CAS_slope_massLE1002020.pkl"),
            "concentration": os.path.join(base_dir, "CAS_concentration_massLE1002020.pkl"),
            "mass": os.path.join(base_dir, "Dry_mass_BCB2020_lessthan100massREAL.csv")
        },
        2021: {
            "slope": os.path.join(base_dir, "CAS_slope_massLE1002021.pkl"),
            "concentration": os.path.join(base_dir, "CAS_concentration_massLE1002021.pkl"),
            "mass": os.path.join(base_dir, "Dry_mass_BCB2021_lessthan100massREAL.csv")
        },
        2022: {
            "slope": os.path.join(base_dir, "CAS_slope_massLE100.pkl"),
            "concentration": None,
            "mass": os.path.join(base_dir, "Dry_mass_BCB2022_lessthan100massREAL.csv")
        }
    },
    "CDP": {
        2020: {
            "slope": os.path.join(base_dir, "CDP_slope_massLE1002020.pkl"),
            "concentration": os.path.join(base_dir, "CDP_concentration_massLE1002020.pkl"),
            "mass": os.path.join(base_dir, "Dry_mass_BCB2020_lessthan100massREAL_CDP.csv")
        },
        2021: {
            "slope": os.path.join(base_dir, "CDP_slope_massLE1002021.pkl"),
            "concentration": os.path.join(base_dir, "CDP_concentration_massLE1002021.pkl"),
            "mass": os.path.join(base_dir, "Dry_mass_BCB2021_lessthan100massREAL_CDP.csv")
        },
        2022: {
    "slope": os.path.join(base_dir, "CDP_slope_massLE100.pkl"),
    "concentration": os.path.join(base_dir, "CDP_concentration_massLE1002022.pkl"),
    "mass": os.path.join(base_dir, "Dry_mass_CDP_legs_mass100.csv")
        },
    },
    "FCDP": {
        2020: {
            "slope": os.path.join(base_dir, "FCDP_slope_massLE1002020.pkl"),
            "concentration": os.path.join(base_dir, "FCDP_concentration_massLE1002020.pkl"),
            "mass": os.path.join(base_dir, "Dry_mass_BCB2020_lessthan100massREAL_FCDP.csv")
        },
        2021: {
            "slope": os.path.join(base_dir, "FCDP_slope_massLE1002021.pkl"),
            "concentration": os.path.join(base_dir, "FCDP_concentration_massLE1002021.pkl"),
            "mass": os.path.join(base_dir, "Dry_mass_BCB2021_lessthan100massREAL_FCDP.csv")
        },
        2022: {
            "slope": os.path.join(base_dir, "FCDP_slope_massLE1002022.pkl"),
            "concentration": os.path.join(base_dir, "FCDP_concentration_massLE1002022.pkl"),
            "mass": os.path.join(base_dir, "Dry_mass_BCB2022_lessthan100massREAL_FCDP.csv")
        }
    }
}
# %%
