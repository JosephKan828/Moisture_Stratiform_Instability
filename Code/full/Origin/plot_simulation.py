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


# ### Functions

def outer_time_einsum(A, B, out=None, dtype=None):
    """
    A: (x, t), B: (z,)
    Returns C: (x, z, t) with C[x,z,t] = A[x,t] * B[z]
    """
    if dtype is None:
        dtype = np.result_type(A, B)
    x, t = A.shape
    z    = B.size
    if out is None:
        out = np.empty((x, z, t), dtype=dtype, order='C')

    A_ = np.ascontiguousarray(A, dtype=dtype)   # contiguous, right dtype
    B_ = np.ascontiguousarray(B, dtype=dtype).squeeze()

    # einsum writes directly into `out`
    np.einsum('xt,z->xzt', A_, B_, out=out, optimize=True)
    return out


# ### Import files

fpath = "/home/b11209013/2025_Research/MSI/File/"; # file path

# load inverse matrix
with h5py.File(fpath+"Sim_stuff/inv_mat.h5","r") as f:
    inv_mat = np.array(f.get("inverse matrix")); # shape: (x, k)
    k       = np.array(f.get("wavenumber"));     # shape: (k,)
    x       = np.array(f.get("x"));              # shape: (x,)

# load vertical mode
with h5py.File(fpath+"Sim_stuff/vertical_mode.h5","r") as f:
    G1      = np.array(f.get("G1")); # shape: (1, z)
    G2      = np.array(f.get("G2")); # shape: (1, z)
    z       = np.array(f.get("z"));  # shape: (z,)

G1 = np.asarray(G1, dtype=np.float32, order="C").squeeze()
G2 = np.asarray(G2, dtype=np.float32, order="C").squeeze()

# load background field
with h5py.File(fpath+"Sim_stuff/background.h5","r") as f:
    ρ0      = np.array(f.get("ρ0")); # shape: (z,)

# load state vector
with h5py.File(fpath+"Full/state.h5","r") as f:
    state   = np.array(f.get("state vector")); # shape: (k, v, t)
    t       = np.array(f.get("time"));         # shape: (t,)
    var     = np.array(f.get("variables"));    # shape: (v,1)


# ### Reconstruct specific wave length

# target λ (units: km)
λ    = 8640;
kn   = 2*np.pi*4320/λ;          # corresponding non-dimensional wavenumber 
kidx = np.argmin(np.abs(k-kn)); # index for specific λ

# limit domain of plotting
lft_bnd = np.argmin(np.abs(x+4320000));
rgt_bnd = np.argmin(np.abs(x-4320000));

# reconstruct physical temperature and vertical motion 
# w_pc = np.einsum("vt,x->vxt", state[kidx,:2,:] , inv_mat[:,kidx], optimize=True)[:,lft_bnd:rgt_bnd+1,:].real;
t1_pc = (state[kidx,2,:][:,None]@inv_mat[:,kidx][None,:]).real;
t2_pc = (state[kidx,3,:][:,None]@inv_mat[:,kidx][None,:]).real;

q_pc  = (state[kidx,4,:][:,None]@inv_mat[:,kidx][None,:]).real;
L_pc  = (state[kidx,5,:][:,None]@inv_mat[:,kidx][None,:]).real;

t1_pc = np.asarray(t1_pc, dtype=np.float32, order="C")
t2_pc = np.asarray(t2_pc, dtype=np.float32, order="C")
q_pc  = np.asarray(q_pc , dtype=np.float32, order="C")
L_pc  = np.asarray(L_pc , dtype=np.float32, order="C")

U_pc  = L_pc + 0.7*q_pc

J1 = L_pc+U_pc ; J2 = L_pc-U_pc



# construct vertical profile
# w1 = outer_time_einsum(w_pc[0], G1) / ρ0[None,:,None];
# w2 = outer_time_einsum(w_pc[1], G2) / ρ0[None,:,None];

# Vectorized vertical projection: (z,) × (nt,nx) -> (nt,z,nx)

print(G1.shape)
t1_prof = np.einsum('z,tx->tzx', G1, t1_pc, optimize=True)
t2_prof = np.einsum('z,tx->tzx', G2, t2_pc, optimize=True)

J1_prof = np.einsum('z,tx->tzx', G1, J1,    optimize=True)
J2_prof = np.einsum('z,tx->tzx', G2, J2,    optimize=True)

rho = np.asarray(ρ0, dtype=np.float32).reshape(1, -1, 1)

C = (-0.0065 + 9.81/1004.5)

temp32 = C*(t1_prof + t2_prof) / rho
heat32 = C*(J1_prof + J2_prof) / rho

x_sub = x[lft_bnd:rgt_bnd+1]
Temp_prof = temp32[:, :, lft_bnd:rgt_bnd+1].astype(np.float32, copy=False)
heat_prof = heat32[:, :, lft_bnd:rgt_bnd+1].astype(np.float32, copy=False)

# Temp_prof = np.empty((len(t),len(z),len(x)));
# heat_prof = np.empty((len(t),len(z),len(x)));

# for i in range(len(t)):
#     t1_prof = G1.T @ t1_pc[i][None,:]*(-0.0065+9.81/1004.5);
#     t2_prof = G2.T @ t2_pc[i][None,:]*(-0.0065+9.81/1004.5);

#     J1_prof = G1.T @ J1[i][None,:] *(-0.0065+9.81/1004.5);
#     J2_prof = G2.T @ J2[i][None,:] *(-0.0065+9.81/1004.5);

#     Temp_prof[i,...] = (t1_prof+t2_prof)/ρ0[:,None];
#     heat_prof[i,...] = (J1_prof+J2_prof)/ρ0[:,None];    

# ### Plot the animation



# assume Temp_prof has shape (nt, nz, nx) with (z, x) increasing
# temp32 = np.asarray(Temp_prof, dtype=np.float32, order="C")

# symmetric color range is often nicer for anomalies
def to_edges(centers):
    centers = np.asarray(centers, dtype=np.float64)
    edges = np.empty(centers.size, dtype=centers.dtype)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0]  = centers[0]  - (edges[1] - centers[0])
    edges[-1] = centers[-1] + (centers[-1] - edges[-2])
    return edges

xmin, xmax = float(x_sub.min()), float(x_sub.max())
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
        x_cells, z_cells, data_3d[0],
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
        qm.set_array(data_3d[i].ravel(order="C"))
        ax.set_title(f"{title_prefix} (λ=8640 km)  {t[i]}/{tmax}", fontsize=24)
        return (qm,)

    ani = FuncAnimation(fig, update, frames=nt, blit=False)
    ani.save(out_path,
                writer=FFMpegWriter(fps=fps, bitrate=bitrate, extra_args=["-pix_fmt","yuv420p"]))
    plt.close(fig)

x_sub = x[lft_bnd:rgt_bnd+1]
# x_edges = to_edges(x_sub)
# z_edges = to_edges(z)

make_movie(
    Temp_prof, x_sub, z,
    cmap="RdBu_r", title_prefix=r"$T^\prime$", units="K",
    out_path="/home/b11209013/2025_Research/MSI/Fig/Full/Temp_prof.mp4",
)

make_movie(
    heat_prof, x_sub, z,
    cmap="BrBG_r", title_prefix=r"$J^\prime$", units="K/day",
    out_path="/home/b11209013/2025_Research/MSI/Fig/Full/heat_prof.mp4",
)