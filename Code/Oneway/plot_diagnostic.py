# This program is to plot out the growth rate and phase speed of the most unstable mode
# import package
import h5py;
import numpy as np;
from pathlib import Path
from matplotlib import pyplot as plt;

# load data
FPATH_INPUT  = "/work/b11209013/2025_Research/MSI/oneway/origin/diagnose.h5"
FPATH_OUTPUT = Path("/home/b11209013/2025_Research/MSI/Fig/oneway/origin/")
FPATH_OUTPUT.mkdir(parents=True, exist_ok=True) # create directory if not exist

with h5py.File(FPATH_INPUT, 'r') as f:
    λ      = np.array(f.get("λ"))
    growth = np.array(f.get("growth_rate"))
    speed  = np.array(f.get("phase_speed"))


# find the unstable modes
idx_max    = np.nanargmax(growth, axis=1)
growth_max = growth[np.arange(λ.size), idx_max]
speed_max  = speed[np.arange(λ.size), idx_max]

# plot figures
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "font.size": 14,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.linewidth": 1.2,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "legend.frameon": False,
    "legend.fontsize": 13,
    "mathtext.default": "regular",  # keep ascii clean; toggle to 'regular' for non-TeX
})

def polish_axes(ax):
    ax.tick_params(direction="in", length=6, width=1.1, top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

# ----------------------------
# Figure 1: Growth rate
# ----------------------------

x = 40000.0 / λ

fig, axes = plt.subplots(2, 1, figsize=(10.5, 12.4), sharex=True)
(ax1, ax2) = axes

# ----------------------------
# (1) Growth Rate of the Most Unstable Mode
# ----------------------------
ax1.plot(x, growth_max/2, label="Max growth over modes", lw=2.5, color="black")

# Optional shading for stable region (commented by default)
# ax1.fill_between(x, 0, growth_max, where=(growth_max <= 0),
#                  alpha=0.08, color="gray", label="Stable (≤ 0)")

ax1.set_xlim(0, 30)
ax1.set_ylim(0, 0.13)
ax1.set_xticks(np.linspace(0, 30, 7))
ax1.set_yticks(np.linspace(0, 0.12, 7))
ax1.set_ylabel("Growth Rate (1/day)")
ax1.set_title("Growth Rate of the Most Unstable Mode")

polish_axes(ax1)
ax1.legend(loc="upper right", ncols=1)

# ----------------------------
# (2) Phase Speed of All Modes
# ----------------------------
# Broadcast x for scatter plot
X = np.broadcast_to(x[:, None], speed.shape)
ax2.scatter(X, speed, s=8, alpha=0.75, edgecolors="none", label="All modes")

# Highlight most unstable mode (growth > 0)
ax2.scatter(x, speed_max,
            s=36, facecolors="none", edgecolors="black", linewidths=1.2,
            label="Most-unstable (growth > 0)")

ax2.set_xlim(0, 30)
ax2.set_ylim(-5, 60)
ax2.set_xticks(np.linspace(0, 30, 7))
ax2.set_yticks(np.linspace(0, 60, 7))
ax2.set_xlabel("Non-dimensional Wavelength (2π/40000 km)")
ax2.set_ylabel("Phase Speed (m s$^{-1}$)")
ax2.set_title("Phase Speed of All Modes")

polish_axes(ax2)
ax2.legend(loc="upper right")

# ----------------------------
# Final adjustments
# ----------------------------
fig.tight_layout(h_pad=2.0)  # increase vertical spacing between subplots
fig.savefig(FPATH_OUTPUT / "diagnostic.png", dpi=300)
plt.close(fig)