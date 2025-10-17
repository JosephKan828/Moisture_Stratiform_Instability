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

with h5py.File("/home/b11209013/2025_Research/MSI/File/Full/Origin/reconstruction.h5", "r") as f:
    w = np.transpose(np.array(f.get("w")).astype(np.complex64).real, axes=(1, 0, 2));          # (nt, nz, nx)
    T = np.transpose(np.array(f.get("T")).astype(np.complex64).real, axes=(1, 0, 2));          # (nt, nz, nx)
    J = np.transpose(np.array(f.get("J")).astype(np.complex64).real, axes=(1, 0, 2));          # (nt, nz, nx)
    x = np.array(f.get("x")).astype(np.complex64).real;          # (nx,)
    z = np.array(f.get("z")).astype(np.complex64).real;          # (nz,)
    t = np.array(f.get("t")).astype(np.complex64).real;          # (nt,)

xmin, xmax = float(np.min(x)), float(np.max(x))
zmin, zmax = float(np.min(z)),  float(np.max(z))

def make_dual_movie(
    dataA_3d, dataB_3d,          # (nt, nz, nx) each
    x_cells, z_cells,
    cmaps=("RdBu_r", "BrBG_r"),
    titles=(r"$T^\prime$", r"$J^\prime$"),
    units=("K", r"K day$^{-1}$"),
    out_path="/tmp/dual.mp4",
    t=None, lam_km=8640.0,
    fps=40, bitrate=8000,
    figsize=(10.5, 12.4),        # matches your stacked figure
):
    # Shapes & time
    nt, nz, nx = dataA_3d.shape
    assert dataB_3d.shape == (nt, nz, nx), "dataA and dataB must have identical shapes"
    if t is None:
        t = np.arange(nt, dtype=float)
    tmax = float(np.max(t))

    # Independent symmetric color ranges (per field)
    vmaxA = float(np.nanmax(np.abs(dataA_3d)))
    vmaxB = float(np.nanmax(np.abs(dataB_3d)))
    normA = Normalize(vmin=-vmaxA, vmax=+vmaxA)
    normB = Normalize(vmin=-vmaxB, vmax=+vmaxB)

    # Figure & axes (2 x 1)
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=False)
    axA, axB = axes

    # Panel A
    qmA = axA.pcolormesh(
        x_cells, z_cells, dataA_3d[0].T,
        cmap=cmaps[0], norm=normA, shading="nearest"
    )
    cbarA = fig.colorbar(qmA, ax=axA, pad=0.02)
    cbarA.ax.tick_params(labelsize=16)
    cbarA.set_label(f"[ {units[0]} ]", fontsize=18)

    axA.set_ylabel("Z [ km ]", fontsize=18)
    axA.set_xlabel("X [ 100 km ]", fontsize=18)
    axA.set_xticks(np.linspace(-4_000_000, 4_000_000, 5))
    axA.set_xticklabels(["-40","-20","0","20","40"], fontsize=16)
    axA.set_yticks(np.linspace(0, 14_000, 8))
    axA.set_yticklabels(["0","2","4","6","8","10","12","14"], fontsize=16)
    axA.set_xlim(np.min(x_cells), np.max(x_cells))
    axA.set_ylim(np.min(z_cells), np.max(z_cells))
    titleA = axA.set_title(
        f"{titles[0]}  (λ = {lam_km:.0f} km)   t = {t[0]:.1f}/{tmax:.1f}   Max = {np.nanmax(dataA_3d[0]):.2f}",
        fontsize=18
    )

    # Panel B
    qmB = axB.pcolormesh(
        x_cells, z_cells, dataB_3d[0].T,
        cmap=cmaps[1], norm=normB, shading="nearest"
    )
    cbarB = fig.colorbar(qmB, ax=axB, pad=0.02)
    cbarB.ax.tick_params(labelsize=16)
    cbarB.set_label(f"[ {units[1]} ]", fontsize=18)

    axB.set_ylabel("Z [ km ]", fontsize=18)
    axB.set_xlabel("X [ 100 km ]", fontsize=18)
    axB.set_xticks(np.linspace(-4_000_000, 4_000_000, 5))
    axB.set_xticklabels(["-40","-20","0","20","40"], fontsize=16)
    axB.set_yticks(np.linspace(0, 14_000, 8))
    axB.set_yticklabels(["0","2","4","6","8","10","12","14"], fontsize=16)
    axB.set_xlim(np.min(x_cells), np.max(x_cells))
    axB.set_ylim(np.min(z_cells), np.max(z_cells))
    titleB = axB.set_title(
        f"{titles[1]}  (λ = {lam_km:.0f} km)   t = {t[0]:.1f}/{tmax:.1f}   Max = {np.nanmax(dataB_3d[0]):.2f}",
        fontsize=18
    )

    # Optional polish if available
    for ax in (axA, axB):
        try:
            polish_axes(ax)
        except NameError:
            pass

    fig.tight_layout(h_pad=2.0)

    # Update both panels each frame (fast path: set_array with raveled data)
    def update(i):
        qmA.set_array(dataA_3d[i].T.ravel())
        titleA.set_text(
            f"{titles[0]}  (λ = {lam_km:.0f} km)   t = {t[i]:.1f}/{tmax:.1f}   Max = {np.nanmax(dataA_3d[i]):.2f}"
        )
        qmB.set_array(dataB_3d[i].T.ravel())
        titleB.set_text(
            f"{titles[1]}  (λ = {lam_km:.0f} km)   t = {t[i]:.1f}/{tmax:.1f}   Max = {np.nanmax(dataB_3d[i]):.2f}"
        )
        return (qmA, titleA, qmB, titleB)

    ani = FuncAnimation(fig, update, frames=nt, blit=False, interval=1000.0/fps)
    ani.save(
        out_path,
        writer=FFMpegWriter(fps=fps, bitrate=bitrate, extra_args=["-pix_fmt", "yuv420p"])
    )
    plt.close(fig)


# ===== Example call: combine T and J into one video =====
make_dual_movie(
    dataA_3d=T,
    dataB_3d=J,
    x_cells=x, z_cells=z,
    cmaps=("RdBu_r", "BrBG_r"),
    titles=(r"$T^\prime$", r"$J^\prime$"),
    units=("K", r"K day$^{-1}$"),
    out_path="/home/b11209013/2025_Research/MSI/Fig/Full/Origin/T_and_J.mp4",
    t=t, lam_km=8640.0, fps=40, bitrate=8000
)