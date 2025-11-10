# This program is to plot the phase relation among T1, T2, q

######################
# 1. import package
######################

import h5py
import numpy as np
from matplotlib import pyplot as plt

from matplotlib.animation import FuncAnimation, FFMpegWriter

######################
# 2. Load dataset
######################

FPATH_INPUT  = "/work/b11209013/2025_Research/MSI/Full/Origin/"
FPATH_SIM    = "/work/b11209013/2025_Research/MSI/Sim_stuff/"
FPATH_OUTPUT = "/home/b11209013/2025_Research/MSI/Fig/"

# Load state file
with h5py.File(FPATH_INPUT + "state.h5", "r") as f:
    k     = np.array(f.get("wavenumber"))
    state = np.array(f.get("state vector"))

# Load inverse matrix
with h5py.File(FPATH_SIM+"inv_mat.h5", "r") as f:
    inv_mat = np.array(f.get("inverse matrix"))


# setup index
target_wl = 8640.0
target_k  = 2.0 * np.pi * 4320.0 / target_wl
kidx      = np.argmin(np.abs(k - target_k))

state     = state[kidx]
inv_mat   = inv_mat[:,kidx]
print("state shape: ", state.shape)
# select specific series
T1 = state[2]; T2 = state[3]; q = state[4]

T1 = T1*inv_mat
T2 = T2*inv_mat
q  = q*inv_mat

#######################
# 3. Plot
#######################

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

# ----------------------------
# Figure 1: Growth rate
# ----------------------------

fig = plt.figure(figsize=(10.5, 6.2))

plt.plot(np.linspace(-40000, 40000, 80001), T1.real, color="k", label="T1")
plt.plot(np.linspace(-40000, 40000, 80001), T2.real, color="r", label="T2")
plt.plot(np.linspace(-40000, 40000, 80001), q.real, color="b", label="q")
plt.xlim(-4320, 4320)
plt.grid()
plt.tight_layout()
plt.legend()
plt.savefig(FPATH_OUTPUT+"Full/Origin/phase.png", dpi=500)
plt.close()
