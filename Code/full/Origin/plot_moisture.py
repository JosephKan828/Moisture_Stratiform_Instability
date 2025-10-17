import numpy as np
from matplotlib import pyplot as plt
import h5py

with h5py.File("/home/b11209013/2025_Research/MSI/File/Full/state.h5", "r") as f:
    state = np.array(f.get("state vector"))
    k = np.array(f.get("wavenumber"))


with h5py.File("/home/b11209013/2025_Research/MSI/File/Sim_stuff/inv_mat.h5", "r") as f:
    inv_mat = np.array(f.get("inverse matrix"))
print(inv_mat.shape)
q = state[:, 4, :]

λ  = 8640
kλ = 2*np.pi*4320/λ
kidx = np.argmin(np.abs(k - kλ))

q_target = q[kidx,:].reshape(-1,1) @ inv_mat[:,kidx].reshape(1,-1)

plt.contourf(q_target.real[:200,:])
plt.colorbar()
plt.savefig("moisture_evo.png", dpi=300)