#%%
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
#%%
with open(
    "Figure5_original_model_data.pkl",
    "rb"
) as f:
    fig5 = pickle.load(f)
gccn_cases = fig5["gccn_cases"]
gccn_cases_turb = fig5["gccn_cases_turb"]
mass_cases = fig5["mass_cases"]
mass_cases_turb = fig5["mass_cases_turb"]
gccn_medians = fig5["gccn_medians"]
gccn_turb_medians = fig5["gccn_turb_medians"]
mass_medians = fig5["mass_medians"]
mass_turb_medians = fig5["mass_turb_medians"]
print("Loaded original Figure 5 data.")

# Mass concentrations supplied by Rob
#
# M_2 = sea-salt mass for particles with d > 2 microns
#
# Rain rates are approximate values read from JL08 Fig. 6
# at Nc ~ 53 cm^-3, which is the closest value to the
# approximately 50 cm^-3 case in our 100 g m^-2 LWP model.

JL08_wind = np.array([
    1.0,
    2.0,
    4.0,
    8.0,
    16.0])
JL08_mass = np.array([
    0.2,
    0.9,
    3.0,
    10.0,
    40.0])
JL08_rain = np.array([
    5.5e-4,
    7.0e-4,
    1.2e-3,
    6.1e-3,
    4.6e-2])
print("\nJensen and Lee (2008) comparison data:")
for wind, mass, rain in zip(
    JL08_wind,
    JL08_mass,
    JL08_rain):
    print(
        f"u = {wind:4.1f} m/s   "
        f"M2 = {mass:5.1f} ug/m3   "
        f"R = {rain:.4f} mm/h"    )
def split_turb_label(label):
    if label.startswith("High Turbulence "):
        return (
            label.replace(
                "High Turbulence ",
                ""
            ),
            "High Turbulence"        )
    if label.startswith("High Turbulence"):
        return (
            label.replace(
                "High Turbulence",
                ""
            ).strip(),
            "High Turbulence"        )
    return (
        label.replace(
            " + Turb",
            ""
        ),
        "Turbulence"    )
fig, axes = plt.subplots(
    1,
    2,
    figsize=(10, 7),
    sharey=True)
ax1, ax2 = axes
for label, med in gccn_medians.items():
    x_med = np.asarray(
        med["x_med"],
        dtype=float    )
    y_med = np.asarray(
        med["y_med"],
        dtype=float    )
    if x_med.size == 0:
        continue
    ax1.plot(
        x_med,
        y_med,
        color=gccn_cases[label]["color"],
        lw=3,
        ls="-",
        label=f"{label} (No Turbulence)"    )
for label, med in gccn_turb_medians.items():
    x_med = np.asarray(
        med["x_med"],
        dtype=float    )
    y_med = np.asarray(
        med["y_med"],
        dtype=float    )
    if x_med.size == 0:
        continue
    base_label, turb_tag = split_turb_label(
        label    )
    is_high = (
        turb_tag == "High Turbulence"    )
    ax1.plot(
        x_med,
        y_med,
        color=gccn_cases_turb[label]["color"],
        lw=3,
        ls="--" if not is_high else "-",
        marker="o" if is_high else None,
        markersize=6 if is_high else None,
        markevery=2 if is_high else None,
        label=f"{base_label} ({turb_tag})"    )
for label, med in mass_medians.items():
    x_med = np.asarray(
        med["x_med"],
        dtype=float    )
    y_med = np.asarray(
        med["y_med"],
        dtype=float    )
    if x_med.size == 0:
        continue
    ax2.plot(
        x_med,
        y_med,
        color=mass_cases[label]["color"],
        lw=3,
        ls="-"    )
for label, med in mass_turb_medians.items():
    x_med = np.asarray(
        med["x_med"],
        dtype=float    )
    y_med = np.asarray(
        med["y_med"],
        dtype=float    )
    if x_med.size == 0:
        continue
    base_label, turb_tag = split_turb_label(
        label    )
    is_high = (
        turb_tag == "High Turbulence"    )
    ax2.plot(
        x_med,
        y_med,
        color=mass_cases_turb[label]["color"],
        lw=3,
        ls="--" if not is_high else "-",
        marker="o" if is_high else None,
        markersize=6 if is_high else None,
        markevery=2 if is_high else None    )
ax1.set_xscale("log")
ax2.set_xscale("log")
xlim1 = ax1.get_xlim()
xlim2 = ax2.get_xlim()
for k in [
    "Base",
    "High Na",
    "Low Na"]:

    g = np.asarray(
        gccn_cases[k]["gccn"],
        dtype=float
    ).ravel()

    r = np.asarray(
        gccn_cases[k]["rain"],
        dtype=float
    ).ravel()

    msk = (
        np.isfinite(g) &
        np.isfinite(r) &
        (g > 0) &
        (r > 0) &
        (g >= xlim1[0]) &
        (g <= xlim1[1])    )
    ax1.scatter(
        g[msk],
        r[msk],
        s=18,
        alpha=0.25,
        color=gccn_cases[k]["color"],
        edgecolor="none",
        zorder=1,
        label="_nolegend_"    )
for k in [
    "Base",
    "High Na",
    "Low Na"]:
    m = np.asarray(
        mass_cases[k]["mass"],
        dtype=float
    ).ravel()
    r = np.asarray(
        mass_cases[k]["rain"],
        dtype=float
    ).ravel()
    msk = (
        np.isfinite(m) &
        np.isfinite(r) &
        (m > 0) &
        (r > 0) &
        (m >= xlim2[0]) &
        (m <= xlim2[1])    )
    ax2.scatter(
        m[msk],
        r[msk],
        s=18,
        alpha=0.45,
        color=mass_cases[k]["color"],
        edgecolor="none",
        zorder=1,
        label="_nolegend_"    )
JL08_handle = ax2.plot(
    JL08_mass,
    JL08_rain,
    color="black",
    linestyle=":",
    linewidth=2.5,
    marker="D",
    markersize=7,
    markerfacecolor="white",
    markeredgecolor="black",
    markeredgewidth=1.5,
    zorder=10,
    label="Jensen & Lee (2008)"
)[0]
ax1.set_xlim(xlim1)
# Make sure the 0.2 ug/m3 JL08 point is visible
new_mass_min = min(
    xlim2[0],
    JL08_mass.min() * 0.8)
new_mass_max = max(
    xlim2[1],
    JL08_mass.max() * 1.15)
ax2.set_xlim(
    new_mass_min,
    new_mass_max)
ax1.set_yscale("log")
ax2.set_yscale("log")
# JL08 has values below 1e-3,
# so lower revised figure to 1e-4
ax1.set_ylim(
    bottom=1e-4)
ax1.set_xlabel(
    "GCCN concentration (m$^{-3}$)",
    fontsize=17,
    fontweight="bold")
ax1.set_ylabel(
    "Accumulated Rain (mm)",
    fontsize=17,
    fontweight="bold")
ax1.set_title(
    "(a) GCCN Concentration",
    fontsize=17,
    fontweight="bold")
ax1.grid(
    alpha=0.3)
ax2.set_xlabel(
    "GCCN Mass (µg m$^{-3}$)",
    fontsize=17,
    fontweight="bold")
ax2.set_title(
    "(b) GCCN Mass",
    fontsize=17,
    fontweight="bold")
ax2.grid(
    alpha=0.3)
for ax in axes:
    ax.tick_params(
        axis="both",
        labelsize=17,
        width=2,
        length=6    )
    for t in (
        ax.get_xticklabels() +
        ax.get_yticklabels()    ):
        t.set_fontweight("bold")
handles, labels = (
    ax1.get_legend_handles_labels())
uniq = OrderedDict()
for h, lab in zip(
    handles,
    labels):
    if lab == "_nolegend_":
        continue
    if lab not in uniq:
        uniq[lab] = h
no_h, no_l = [], []
tb_h, tb_l = [], []
hi_h, hi_l = [], []
for lab, h in uniq.items():
    if "(No Turbulence)" in lab:
        no_h.append(h)
        no_l.append(
            lab.replace(
                " (No Turbulence)",
                ""            )        )
    elif (
        "(Turbulence)" in lab and
        "(High Turbulence)" not in lab    ):
        tb_h.append(h)
        tb_l.append(
            lab.replace(
                " (Turbulence)",
                ""
            )        )
    elif "(High Turbulence)" in lab:
        hi_h.append(h)
        hi_l.append(
            lab.replace(
                " (High Turbulence)",
                ""            )        )
leg1 = fig.legend(
    no_h,
    no_l,
    title=(
        r"$\mathbf{\epsilon = 0\ "
        r"cm^2\ s^{-3}}$"    ),
    loc="center left",
    bbox_to_anchor=(
        0.89,
        0.60    ),
    fontsize=14,
    frameon=False)
leg1.get_title().set_fontweight(
    "bold")
leg2 = fig.legend(
    tb_h,
    tb_l,
    title=(
        r"$\mathbf{\epsilon = 50\ "
        r"cm^2\ s^{-3}}$"    ),
    loc="center left",
    bbox_to_anchor=(
        1.11,
        0.60    ),
    fontsize=14,
    frameon=False)
leg2.get_title().set_fontweight(
    "bold")
leg3 = ax2.legend(
    handles=[JL08_handle],
    labels=[
        "Jensen & Lee (2008)"
    ],
    loc="lower right",
    fontsize=11,
    frameon=False)
plt.show()
# fig.savefig(
#     "Figure5_with_JensenLee2008.pdf",
#     dpi=300,
#     bbox_inches="tight"
# )
# %%
# %%
model_no_turb = mass_medians[
    "100 g m$^{-2}$ LWP"]
model_mass_no_turb = np.asarray(
    model_no_turb["x_med"],
    dtype=float)
model_rain_no_turb = np.asarray(
    model_no_turb["y_med"],
    dtype=float)
model_turb = mass_turb_medians[
    "100 g m$^{-2}$ LWP + Turb"]
model_mass_turb = np.asarray(
    model_turb["x_med"],
    dtype=float)
model_rain_turb = np.asarray(
    model_turb["y_med"],
    dtype=float)
def loglog_interp(x_new, x, y):
    x_new = np.asarray(
        x_new,
        dtype=float    )
    x = np.asarray(
        x,
        dtype=float    )
    y = np.asarray(
        y,
        dtype=float    )
    valid = (
        np.isfinite(x) &
        np.isfinite(y) &
        (x > 0) &
        (y > 0)    )
    x = x[valid]
    y = y[valid]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    result = np.full(
        x_new.shape,
        np.nan,
        dtype=float    )
    inside = (
        (x_new >= np.min(x)) &
        (x_new <= np.max(x))    )
    result[inside] = 10 ** np.interp(
        np.log10(x_new[inside]),
        np.log10(x),
        np.log10(y)    )

    return result
model_at_JL08_no_turb = loglog_interp(
    JL08_mass,
    model_mass_no_turb,
    model_rain_no_turb)
model_at_JL08_turb = loglog_interp(
    JL08_mass,
    model_mass_turb,
    model_rain_turb)
ratio_no_turb = (
    model_at_JL08_no_turb /
    JL08_rain)
ratio_turb = (
    model_at_JL08_turb /
    JL08_rain)
for i in range(
    len(JL08_mass)
):

    print(
        f"\nM2 = {JL08_mass[i]:.1f} "
        f"ug m^-3"    )
    print(
        f"JL08: "
        f"{JL08_rain[i]:.5f} mm h^-1"    )
    if np.isfinite(
        model_at_JL08_no_turb[i]    ):
        print(
            f"Model, eps=0: "
            f"{model_at_JL08_no_turb[i]:.5f} mm"        )
        print(
            f"Model/JL08: "
            f"{ratio_no_turb[i]:.2f}"        )
    else:
        print(
            "Model, eps=0: "
            "outside interpolation range"        )
    if np.isfinite(
        model_at_JL08_turb[i]    ):
        print(
            f"Model, eps=50: "
            f"{model_at_JL08_turb[i]:.5f} mm"        )
        print(
            f"Model/JL08: "
            f"{ratio_turb[i]:.2f}"        )
    else:
        print(
            "Model, eps=50: "
            "outside interpolation range"        )
# %%
