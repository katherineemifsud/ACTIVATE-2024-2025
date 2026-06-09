#%%
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Path to your Excel file on the server
file_path = "/home/disk/eos4/kathem24/activate/FSM_FY26targetUS.xlsx"

# Read the relevant sheet and columns
df = pd.read_excel(file_path, sheet_name="Sheet1", engine="openpyxl")
df = df[['FY26 FSM Name ', 'State']].dropna(subset=['FY26 FSM Name ', 'State'])

# Count unique FSMs per state
state_counts = df.groupby('State')['FY26 FSM Name '].nunique().reset_index(name='FSM Count')
state_counts = state_counts.sort_values('FSM Count', ascending=False)

# --- Plot ---
plt.figure(figsize=(12, 6))
plt.bar(state_counts['State'], state_counts['FSM Count'], color='skyblue', edgecolor='black')
plt.xticks(rotation=90)
plt.ylabel("Number of FSMs", fontsize=13, fontweight='bold')
plt.xlabel("State", fontsize=13, fontweight='bold')
plt.title("FY26 FSM Count by State", fontsize=15, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# %%
