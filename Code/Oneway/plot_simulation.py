#!/usr/bin/env python
# coding: utf-8

# ## Plot animation of temperature and vertical motion animation

# ### import package

# In[1]:


import sys;
import h5py;
import numpy as np;
from tqdm import tqdm;
from matplotlib import pyplot as plt;
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter;

sys.path.append("/home/b11209013/Package/")
import Plot_Style as ps;


# ### Functions

# In[2]:


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

# In[3]:


fpath = "/home/b11209013/2025_Research/MSI/File/"; # file path

# load inverse matrix
with h5py.File(fpath+"inv_mat.h5","r") as f:
    inv_mat = np.array(f.get("inverse matrix")); # shape: (x, k)
    k       = np.array(f.get("wavenumber"));     # shape: (k,)
    x       = np.array(f.get("x"));              # shape: (x,)

# load vertical mode
with h5py.File(fpath+"vertical_mode.h5","r") as f:
    G1      = np.array(f.get("G1")); # shape: (1, z)
    G2      = np.array(f.get("G2")); # shape: (1, z)
    z       = np.array(f.get("z"));  # shape: (z,)

# load background field
with h5py.File(fpath+"bg_field.h5","r") as f:
    ρ0      = np.array(f.get("ρ0")); # shape: (z,)

# load state vector
with h5py.File(fpath+"oneway_state.h5","r") as f:
    state   = np.array(f.get("state vector")); # shape: (k, v, t)
    t       = np.array(f.get("time"));         # shape: (t,)
    var     = np.array(f.get("variables"));    # shape: (v,1)


# ### Reconstruct specific wave length

# In[ ]:


# target λ (units: km)
λ    = 8640;
kn   = 2*np.pi*4320/λ;          # corresponding non-dimensional wavenumber 
kidx = np.argmin(np.abs(k-kn)); # index for specific λ

# limit domain of plotting
lft_bnd = np.argmin(np.abs(x+4320000));
rgt_bnd = np.argmin(np.abs(x-4320000));

# reconstruct physical temperature and vertical motion 
# w_pc = np.einsum("vt,x->vxt", state[kidx,:2,:] , inv_mat[:,kidx], optimize=True)[:,lft_bnd:rgt_bnd+1,:].real;
t1_pc = (state[kidx,1,:][:,None]@inv_mat[:,kidx][None,:]).real;
t2_pc = (state[kidx,2,:][:,None]@inv_mat[:,kidx][None,:]).real;

# construct vertical profile
# w1 = outer_time_einsum(w_pc[0], G1) / ρ0[None,:,None];
# w2 = outer_time_einsum(w_pc[1], G2) / ρ0[None,:,None];

Temp_prof = np.empty((len(t),len(z),len(x)));

for i in range(len(t)):
    t1_prof = G1.T @ t1_pc[i][None,:]*(-0.0065+9.81/1004.5);
    t2_prof = G2.T @ t2_pc[i][None,:]*(-0.0065+9.81/1004.5);

    Temp_prof[i,...] = (t1_prof+t2_prof)/ρ0[:,None];    

# ### Plot the animation

# In[5]:
from matplotlib.colors import Normalize

# assume Temp_prof has shape (nt, nz, nx) with (z, x) increasing
temp32 = np.asarray(Temp_prof, dtype=np.float32, order="C")

# symmetric color range is often nicer for anomalies
amax  = float(np.nanmax(np.abs(temp32)))
levels = np.linspace(-amax, amax, 41)
norm   = Normalize(vmin=-amax, vmax=amax)

tmax = np.max(t)

x_sub = x[lft_bnd:rgt_bnd+1]  # x for plotting

for i in tqdm(range(len(t))):
    fig, ax = plt.subplots(figsize=(16, 9))

    # Filled contours
    cf = ax.contourf(
        x_sub, z, temp32[i][:, lft_bnd:rgt_bnd+1], levels=levels, cmap="RdBu_r",
        norm=norm, extend="both"
    )
    # (optional) zero contour line for reference
    # ax.contour(x_sub, z, temp32[i], levels=[0.0], colors="k", linewidths=0.8)

    ax.set_xlim(-4_000_000, 4_000_000)
    ax.set_ylim(0, 15_000)
    ax.set_xticks(np.linspace(-4_000_000, 4_000_000, 5),
                    ["-40","-20","0","20","40"], fontsize=20)
    ax.set_yticks(np.linspace(2_000, 14_000, 7),
                    ["2","4","6","8","10","12","14"], fontsize=20)
    ax.set_xlabel("X [ 100 km ]", fontsize=24)
    ax.set_ylabel("Z [ km ]", fontsize=24)
    ax.set_title(r"$T^\prime$ ($\lambda$=8640km) "+f"{t[i]}/{tmax}", fontsize=32)

    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label("T [K]")

    plt.savefig(f"/data92/b11209013/MSI/Figure/Oneway/t={t[i]}.png", dpi=100)
    plt.close(fig)