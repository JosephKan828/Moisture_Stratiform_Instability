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


def polish_axes(ax):
    ax.tick_params(direction="in", length=6, width=1.1, top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

def make_movie(
    data,          # (nt, nz, nx) each
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
