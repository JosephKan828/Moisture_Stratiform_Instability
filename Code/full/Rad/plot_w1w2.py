import numpy as np
from matplotlib import pyplot as plt
import h5py

with h5py.File("/home/b11209013/2025_Research/MSI/File/Full/state_rad.h5", "r") as f:
    state = np.array(f.get("state vector"))
    k = np.array(f.get("wavenumber"))

λ  = 8640
kλ = 2*np.pi*4320/λ
kidx = np.argmin(np.abs(k - kλ))

w1 = state[kidx, 0, :]
w2 = state[kidx, 1, :]
q  = state[kidx, 4, :]
L  = state[kidx, 5, :]

J1 = 2*L + 0.7*q
J2 = -0.7*q

R1 = 5.61*q
R2 = 3.36*q

# plt.plot(w1[:50], label="w1")
# plt.plot(w2[:50], label="w2")
plt.plot(J1[:600], label="J1")
plt.plot(J2[:600], label="J2")
# plt.plot(q[:100], label="q")
plt.plot(R1[:600], label="R1")
plt.plot(R2[:600], label="R2")
plt.legend()
plt.savefig("w1w2q_evo.png", dpi=300)
plt.close()
plt.plot(q[:600], label="q")
plt.savefig("q_evo.png", dpi=300)