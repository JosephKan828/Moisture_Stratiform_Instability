#!/usr/bin/env python
# coding: utf-8

# ## Plot animation of temperature and vertical motion animation

# ### import package

import sys;
import h5py;
import numpy as np;
from tqdm import tqdm;
from matplotlib import pyplot as plt;
from matplotlib.colors import TwoSlopeNorm
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter;

sys.path.append("/home/b11209013/Package/")
import Plot_Style as ps; # type: ignore

scaling_factor = float(sys.argv[1])

#################################
# 1. Read reconstructed data
#################################

with h5py.File(f"/work/b11209013/2025_Research/MSI/Full/Rad/reconstruction_rad_{scaling_factor}.h5", "r") as f:
    w = np.transpose(np.array(f.get("w")).astype(np.complex64).real, axes=(1, 0, 2));          # (nt, nz, nx)
    T = np.transpose(np.array(f.get("T")).astype(np.complex64).real, axes=(1, 0, 2));          # (nt, nz, nx)
    J = np.transpose(np.array(f.get("J")).astype(np.complex64).real, axes=(1, 0, 2));          # (nt, nz, nx)
    x = np.array(f.get("x")).astype(np.complex64).real;          # (nx,)
    z = np.array(f.get("z")).astype(np.complex64).real;          # (nz,)
    t = np.array(f.get("t")).astype(np.complex64).real;          # (nt,)

xmin, xmax = float(np.min(x)), float(np.max(x))
zmin, zmax = float(np.min(z)), float(np.max(z))

#################################
# 2. generate animation
#################################
tmax = float(np.max(t)) # set the upper bound of time coordinate

# setup levels for different field
Tmax = np.nanmax(np.abs(T)); Tnorm = TwoSlopeNorm(vcenter=0, vmin=-Tmax, vmax=Tmax)
Jmax = np.nanmax(np.abs(J)); Jnorm = TwoSlopeNorm(vcenter=0, vmin=-Jmax, vmax=Jmax)

fps=40

print("Start generating T profile animation")

# generate figure for temperature
fig,ax = plt.subplots(figsize=(10.5, 6.2))

## tmeperature evolution
qmA = ax.pcolormesh(
    x, z, T[0].T,
    cmap="RdBu_r", norm=Tnorm, shading="nearest"
)
cbarA = fig.colorbar(qmA, ax=ax, pad=0.02)
cbarA.ax.tick_params(labelsize=16)
cbarA.set_label(f"[ K ]", fontsize=18)

ax.set_ylabel("Z [ km ]", fontsize=18)
ax.set_xlabel("X [ 100 km ]", fontsize=18)
ax.set_xticks(np.linspace(-4_000_000, 4_000_000, 5))
ax.set_xticklabels(["-40","-20","0","20","40"], fontsize=16)
ax.set_yticks(np.linspace(0, 14_000, 8))
ax.set_yticklabels(["0","2","4","6","8","10","12","14"],fontsize=16)
ax.set_xlim(xmin, xmax)
ax.set_ylim(zmin, zmax)
titleA = ax.set_title(
    r"$T^\prime$"+f" (λ = 8640 km)   t = {t[0]:.1f}/{tmax:.1f}",
    fontsize=18
)

plt.tight_layout(h_pad=2.0)

# Update both panels each frame (fast path: set_array with raveled data)
def update(i):
    qmA.set_array(T[i].T.ravel())
    titleA.set_text(
        r"$T^\prime$"+f" (λ = 8640 km)   t = {t[i]:.1f}/{tmax:.1f}"
    )
    return (qmA, titleA)

ani = FuncAnimation(fig, update, frames=t.size, blit=False, interval=1000.0/fps)
ani.save(
    f"/work/b11209013/2025_Research/MSI/Animation/Full/Rad/T_prof_evo_{scaling_factor}.mp4",
    writer=FFMpegWriter(fps=fps, extra_args=["-pix_fmt", "yuv420p"]),
    dpi=500
)
plt.close(fig)

print("T animation output finished")

# generate figure for heating
fig, ax = plt.subplots(figsize=(10.5, 6.2))

## tmeperature evolution
qmA = ax.pcolormesh(
    x, z, J[0].T,
    cmap="BrBG_r", norm=Tnorm, shading="nearest"
)
cbarA = fig.colorbar(qmA, ax=ax, pad=0.02)
cbarA.ax.tick_params(labelsize=16)
cbarA.set_label(r"[ $K day^{-1}$ ]", fontsize=18)

ax.set_ylabel("Z [ km ]", fontsize=18)
ax.set_xlabel("X [ 100 km ]", fontsize=18)
ax.set_xticks(np.linspace(-4_000_000, 4_000_000, 5))
ax.set_xticklabels(
    ["-40","-20","0","20","40"],
    fontsize=16)
ax.set_yticks(np.linspace(0, 14_000, 8))
ax.set_yticklabels(
    ["0","2","4","6","8","10","12","14"],
    fontsize=16)
ax.set_xlim(xmin, xmax)
ax.set_ylim(zmin, zmax)
titleA = ax.set_title(
    r"$J^\prime$"+f" (λ = 8640 km)   t = {t[0]:.1f}/{tmax:.1f}",
    fontsize=18
)

plt.tight_layout(h_pad=2.0)

# Update both panels each frame (fast path: set_array with raveled data)
def update(i):
    qmA.set_array(J[i].T.ravel())
    titleA.set_text(
        r"$J^\prime$"+f" (λ = 8640 km)   t = {t[i]:.1f}/{tmax:.1f}"
    )
    return (qmA, titleA)

ani = FuncAnimation(fig, update, frames=t.size, blit=False, interval=1000.0/fps)
ani.save(
    f"/work/b11209013/2025_Research/MSI/Animation/Full/Rad/J_prof_evo_{scaling_factor}.mp4",
    writer=FFMpegWriter(fps=fps, extra_args=["-pix_fmt", "yuv420p"]),
    dpi=500
)
plt.close(fig)

print("J animation output finished")
