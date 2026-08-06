#%%
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pathlib
import statistics
import glob
import os
import re
import math
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from scipy.optimize import curve_fit
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
# %%
CAS_FILE = (
    "/home/disk/p/kathem24/activate/"
    "ACTIVATE-2024-2025/CAS/Below Cloud Base/Scripts/"
    "CAS_mass_weighted_dry_distribution_massLE1002022.pkl")
with open(CAS_FILE, "rb") as f:
    CAS_mass_distribution_data = pickle.load(f)
print("Loaded CAS mass-weighted distribution.")
print("Saved keys:")
print(CAS_mass_distribution_data.keys())