# This program is to plot out the growth rate and phase speed of the most unstable mode
# import package
import h5py;
import numpy as np;
from pathlib import Path
from matplotlib import pyplot as plt;

# load data
FPATH_INPUT  = "/home/b11209013/2025_Research/MSI/File/Full/diagnose.h5"
FPATH_OUTPUT = Path("/home/b11209013/2025_Research/MSI/Fig/Full/")
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
fig, ax = plt.subplots(figsize=(10.5, 6.2))
ax.plot(x, growth_max, label="Max growth over modes", lw=2.5, color="black")

# Optionally shade stable region for context
# ax.fill_between(x, 0, growth_max, where=(growth_max <= 0), alpha=0.08, color="gray", label="Stable (≤ 0)")

ax.set_xlim(0, 30)
ax.set_ylim(0, 0.13)
ax.set_xticks(np.linspace(0, 30, 7))
ax.set_yticks(np.linspace(0, 0.12, 7))

ax.set_xlabel("Non-dimensional Wavelength (2π/40000 km)")
ax.set_ylabel("Growth Rate (1/day)")
ax.set_title("Growth Rate of the Most Unstable Mode")

polish_axes(ax)
ax.legend(loc="upper right", ncols=1)
fig.tight_layout()
fig.savefig(FPATH_OUTPUT / "growth_rate.png")
plt.close(fig)

# ----------------------------
# Figure 2: Phase speed (all modes + highlight most-unstable positive-growth)
# ----------------------------
fig, ax = plt.subplots(figsize=(10.5, 6.2))

# Plot all modes lightly for context (vectorized, no Python loops):
# We broadcast x (Nλ,) to (Nλ, Nmodes) for a single scatter call.
X = np.broadcast_to(x[:, None], speed.shape)
ax.scatter(X, speed, s=8, alpha=0.25, edgecolors="none", label="All modes")

# Overlay the most-unstable mode (only where growth>0)
ax.scatter(x, speed_max,
           s=36, facecolors="none", edgecolors="black", linewidths=1.2,
           label="Most-unstable (growth > 0)")

ax.set_xlim(0, 30)
ax.set_ylim(-5, 60)
ax.set_xticks(np.linspace(0, 30, 7))
ax.set_yticks(np.linspace(0, 60, 7))

ax.set_xlabel("Non-dimensional Wavelength (2π/40000 km)")
ax.set_ylabel("Phase Speed (m s$^{-1}$)")
ax.set_title("Phase Speed of All Modes")

polish_axes(ax)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(FPATH_OUTPUT / "phase_speed.png")
plt.close(fig)