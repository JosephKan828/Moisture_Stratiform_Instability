# This program is to compare modal growth rate of original model and those with radiation

#######################
# 1. Load package
#######################
import h5py
import numpy as np
import matplotlib.pyplot as plt

from glob import glob

#######################
# 2. Load dataset
#######################
FPATH_INPUT  = "/work/b11209013/2025_Research/MSI/Full/"
FPATH_OUTPUT = "/home/b11209013/2025_Research/MSI/Fig/"

# load diagnose of original model
with h5py.File(FPATH_INPUT + "Origin/diagnose.h5", 'r') as f:
    λ      = np.array(f.get("λ"))
    growth = np.array(f.get("growth_rate"))
    speed  = np.array(f.get("phase_speed"))
x = 40000.0 / λ
# load diagnose of radiation model
RAD_FILEs = sorted(glob(FPATH_INPUT + "Rad/diagnose_rad_both_*.h5"))[:3]

scaling_factor = ["0.001", "0.005", "0.01"]

RAD_growth = np.zeros((λ.size, 6, len(RAD_FILEs)))
RAD_speed  = np.zeros((λ.size, 6, len(RAD_FILEs)))

for i, f in enumerate(RAD_FILEs):
    with h5py.File(f, 'r') as h:
        RAD_growth[:, :, i] = np.array(h.get("growth_rate"))
        RAD_speed[:, :, i]  = np.array(h.get("phase_speed"))

#######################
# 3. Compute the growth rate of most unstable mode
#######################

# find the unstable modes for original
idx_max    = np.nanargmax(growth, axis=1)
growth_max = growth[np.arange(λ.size), idx_max]
speed_max  = speed[np.arange(λ.size), idx_max]

# find the unstable modes for radiation
idx_max_rad = np.nanargmax(RAD_growth, axis=1)

growth_max_rad = np.empty((np.arange(λ.size).size, len(RAD_FILEs)))
speed_max_rad  = np.empty((np.arange(λ.size).size, len(RAD_FILEs)))

for i in range(len(RAD_FILEs)):
    growth_max_rad[:,i] = RAD_growth[np.arange(λ.size),idx_max_rad[:,i],i]
    speed_max_rad[:,i]  = RAD_speed[np.arange(λ.size),idx_max_rad[:,i],i]

plt.plot(x, growth_max, label="Original")
for i in range(len(RAD_FILEs)):
    plt.plot(x, growth_max_rad[:,i], label=f"Scaling: {scaling_factor[i]}")
plt.xlim(0,30)
plt.ylim(0,None)
plt.legend()
plt.savefig(FPATH_OUTPUT + "GrowthRate_Comparision.png", dpi=300)
plt.close()

plt.scatter(np.broadcast_to(x[:, None], speed.shape), speed, label="Original", s=1)
for i in range(len(RAD_FILEs)):
    plt.scatter(np.broadcast_to(x[:, None], speed.shape), RAD_speed[:,:,i], label=f"Scaling: {scaling_factor[i]}", s=1)
plt.ylim(0,None)
plt.legend()
plt.savefig(FPATH_OUTPUT + "PhaseSpeed_Comparision.png", dpi=300)
plt.close()