#!/usr/bin/env python
# coding: utf-8

# ## Plot animation of temperature and vertical motion animation

# ### import package

import sys;
import h5py;
import numpy as np;
from tqdm import tqdm;
from matplotlib import pyplot as plt;
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter;

sys.path.append("/home/b11209013/Package/")
import Plot_Style as ps; # type: ignore

with h5py.File("/home/b11209013/2025_Research/MSI/File/Sim_stuff/reconstruction_test.h5", "r") as f:
    w = np.transpose(np.array(f.get("w")).astype(np.complex64).real, axes=(1, 0, 2));          # (nt, nz, nx)
    T = np.transpose(np.array(f.get("T")).astype(np.complex64).real, axes=(1, 0, 2));          # (nt, nz, nx)
    J = np.transpose(np.array(f.get("J")).astype(np.complex64).real, axes=(1, 0, 2));          # (nt, nz, nx)
    x = np.array(f.get("x")).astype(np.complex64).real;          # (nx,)
    z = np.array(f.get("z")).astype(np.complex64).real;          # (nz,)
    t = np.array(f.get("t")).astype(np.complex64).real;          # (nt,)

xmin, xmax = float(np.min(x)), float(np.max(x))
zmin, zmax = float(np.min(z)),  float(np.max(z))

def make_movie(data_3d, x_cells, z_cells, cmap, title_prefix, units, out_path,
                        fps=40, bitrate=8000):
    """
    data_3d: (nt, nz, nx) float32
    """
    nt, nz, nx = data_3d.shape

    # Symmetric range for anomalies
    vmax = float(np.nanmax(np.abs(data_3d)))
    norm = Normalize(vmin=-vmax, vmax=+vmax)

    fig, ax = plt.subplots(figsize=(16, 9))

    # Create once
    qm = ax.pcolormesh(
        x_cells, z_cells, data_3d[0].T,
        cmap=cmap, norm=norm, shading="nearest"  # "nearest" is fastest
    )
    cbar = fig.colorbar(qm, ax=ax, pad=0.02)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(f"[ {units} ]", fontsize=20)

    # Axes cosmetics
    ax.set_xlabel("X [ 100 km ]", fontsize=20)
    ax.set_ylabel("Z [ km ]", fontsize=20)
    ax.set_xticks(np.linspace(-4_000_000, 4_000_000, 5))
    ax.set_xticklabels(["-40","-20","0","20","40"], fontsize=16)
    ax.set_yticks(np.linspace(0, 14_000, 8))
    ax.set_yticklabels(["0","2","4","6","8","10","12","14"], fontsize=16)
    ax.set_xlim(np.min(x_cells), np.max(x_cells))
    ax.set_ylim(np.min(z_cells), np.max(z_cells))

    tmax = float(np.max(t))

    # Fast per-frame update: update only the underlying array
    # QuadMesh.set_array expects a flat array of the *face colors*; for
    # pcolormesh with 2D data, pass raveled data. Matplotlib handles mapping.
    def update(i):
        qm.set_array(data_3d[i].ravel(order="C").T)
        ax.set_title(f"{title_prefix} (λ=8640 km)  {t[i]:.1f}/{tmax}; Max={np.max(data_3d[i]):.2f}", fontsize=24)
        return (qm,)

    ani = FuncAnimation(fig, update, frames=nt, blit=False)
    ani.save(out_path,
                writer=FFMpegWriter(fps=fps, bitrate=bitrate, extra_args=["-pix_fmt","yuv420p"]))
    plt.close(fig)

make_movie(
    T, x, z,
    cmap="RdBu_r", title_prefix=r"$T^\prime$", units="K",
    out_path="/home/b11209013/2025_Research/MSI/Fig/Full/Temp_prof_test.mp4",
)

make_movie(
    J, x, z,
    cmap="BrBG_r", title_prefix=r"$J^\prime$", units="K/day",
    out_path="/home/b11209013/2025_Research/MSI/Fig/Full/heat_prof_test.mp4",
)