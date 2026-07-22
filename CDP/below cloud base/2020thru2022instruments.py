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
search_patterns = [
    "/home/disk/eos4/kathem24/**/*.csv",
    "/home/disk/p/kathem24/**/*.csv"
]

found_files = []

for pattern in search_patterns:
    found_files.extend(
        glob.glob(pattern, recursive=True)
    )

average_distribution_files = [
    file_path
    for file_path in found_files
    if (
        "Average_Dry_Size_Distribution_beforemass" 
        in os.path.basename(file_path)
    )
]

for file_path in sorted(average_distribution_files):
    print(file_path)

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
            "N_profiles"
        ]

        missing_columns = [
            column for column in required_columns
            if column not in distribution_df.columns
        ]

        if missing_columns:
            print(
                f"Skipping {instrument} {year}. "
                f"Missing columns: {missing_columns}"
            )
            continue

        # Make sure the imported columns are numeric
        distribution_df["Dry_Diameter_um"] = pd.to_numeric(
            distribution_df["Dry_Diameter_um"],
            errors="coerce"
        )

        distribution_df["Average_dN_dD_dry"] = pd.to_numeric(
            distribution_df["Average_dN_dD_dry"],
            errors="coerce"
        )

        distribution_df["N_profiles"] = pd.to_numeric(
            distribution_df["N_profiles"],
            errors="coerce"
        )

        average_distributions[instrument][year] = distribution_df

        print(
            f"Loaded {instrument} {year}: "
            f"{len(distribution_df)} diameter bins"
        )
# %%
