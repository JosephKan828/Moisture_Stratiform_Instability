import numpy as np
from matplotlib import pyplot as plt
import h5py

with h5py.File("/home/b11209013/2025_Research/MSI/File/Full/state_rad.h5", "r") as f:
    state = np.array(f.get("state vector"))
    k = np.array(f.get("wavenumber"))

q = state[:, 4, :]

λ  = 8640
kλ = 2*np.pi*4320/λ
kidx = np.argmin(np.abs(k - kλ))

q_target = q[kidx,:]

plt.plot(q_target[:100])
plt.savefig("moisture_evo.png", dpi=300)