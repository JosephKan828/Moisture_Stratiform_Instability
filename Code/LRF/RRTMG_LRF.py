#!/usr/bin/env python
# coding: utf-8

# ## LRF with RRTMG

# ### Import packages

# In[7]:


import sys;
import json;
import h5py;
import numpy as np;
import netCDF4 as nc;
import climlab as cl; #type: ignore

import metpy.calc as mpcalc
from metpy.units import units

from copy import deepcopy;
from matplotlib import pyplot as plt;

sys.path.append("/home/b11209013/Package/"); # path to my packages
import Plot_Style as ps; # type: ignore

ps.apply_custom_plot_style();

# Load data
with h5py.File("/work/b11209013/2025_Research/MSI/Rad_Stuff/mean_state.h5", "r") as f:
    q_mean = np.array(f.get("q"))[::-1];
    t_mean = np.array(f.get("t"))[::-1];

# ### Construct RRTMG

# constructing RRTMG
levs = np.linspace(100.0, 1000.0, 37)
lev_lim = np.argmin(np.abs(levs-300.0))

nlev = len(levs)

state = cl.column_state(num_lev=nlev, water_depth=1)
state["Tatm"][:] = t_mean
state["Ts"][:] = state["Tatm"][-1]

# water vapor profile
q = q_mean;

rad_model = cl.radiation.RRTMG(
    name="Radiation Model",
    state=state,
    specific_humidity=q,
    albedo=0.3,
);

rad_model.compute_diagnostics();

LW_ref = rad_model.diagnostics["TdotLW"].copy();
SW_ref = rad_model.diagnostics["TdotSW"].copy();

# ### perturb moisture and temperature

# In[9]:


LRF = {
    "q_lw": np.zeros((nlev, nlev)),
    "q_sw": np.zeros((nlev, nlev)),
    "t_lw": np.zeros((nlev, nlev)),
    "t_sw": np.zeros((nlev, nlev)),
    }


for l in range(nlev):
    q_perturb = deepcopy(q);
    pert = q_perturb[l]*0.01
    q_perturb[l] += pert; # perturb specific humidity by 0.01 kg/kg

    rad_perturb = cl.radiation.RRTMG(
        name="Radiation Model",
        state=state,
        specific_humidity=q_perturb,
        albedo=0.3,
    );

    rad_perturb.compute_diagnostics();

    LRF["q_lw"][l] = (rad_perturb.diagnostics["TdotLW"] - LW_ref) / pert*1e-3;
    LRF["q_sw"][l] = (rad_perturb.diagnostics["TdotSW"] - SW_ref) / pert*1e-3;

    del q_perturb, rad_perturb;

for l in range(nlev):
    perturb_state = deepcopy(state);
    perturb_state["Tatm"][l] += 1;
    perturb_state["Ts"][:] = perturb_state["Tatm"][-1];

    rad_perturb = cl.radiation.RRTMG(
        name="Radiation Model",
        state=perturb_state,
        specific_humidity=q,
        albedo=0.3,
    );

    rad_perturb.compute_diagnostics();

    LRF["t_lw"][l] = (rad_perturb.diagnostics["TdotLW"] - LW_ref) / 1;
    LRF["t_sw"][l] = (rad_perturb.diagnostics["TdotSW"] - SW_ref) / 1;

    del perturb_state, rad_perturb;


# ### Plot out profile

# In[27]:


# for key in LRF.keys():
#     temp_resp = np.zeros((nlev, nlev));

#     temp_resp[lev_lim:, lev_lim:] = LRF[key][lev_lim:, lev_lim:];
#     LRF[key] = temp_resp;

# Save files
with h5py.File("/work/b11209013/2025_Research/MSI/Rad_Stuff/LRF.h5", "w") as f: 
    f.create_dataset("q_lw", data = LRF["q_lw"].T)
    f.create_dataset("q_sw", data = LRF["q_sw"].T)
    f.create_dataset("t_lw", data = LRF["t_lw"].T)
    f.create_dataset("t_sw", data = LRF["t_sw"].T)


plt.figure(figsize=(16,12))
plt.pcolormesh(levs,levs,LRF["q_lw"].T, vmin=-2, vmax=2, cmap='RdBu_r');
plt.xlim(1000,100)
plt.ylim(1000,100)
plt.xlabel("Perturbation level (hPa)")
plt.ylabel("Response level (hPa)")
plt.colorbar()
plt.savefig("/home/b11209013/2025_Research/MSI/Fig/Rad/RRTMG_LRF_q_lw.png", dpi=300);
# plt.show()
plt.close()

plt.figure(figsize=(16,12))
plt.pcolormesh(levs,levs,LRF["q_sw"].T, vmin=-2, vmax=2, cmap='RdBu_r');
plt.xlim(1000,100)
plt.ylim(1000,100)
plt.xlabel("Perturbation level (hPa)")
plt.ylabel("Response level (hPa)")
plt.colorbar()
plt.savefig("/home/b11209013/2025_Research/MSI/Fig/Rad/RRTMG_LRF_q_sw.png", dpi=300);
# plt.show()
plt.close()

plt.figure(figsize=(16,12))
plt.pcolormesh(levs,levs,LRF["t_lw"].T, vmin=-1, vmax=1, cmap='RdBu_r');
plt.xlim(1000,100)
plt.ylim(1000,100)
plt.xlabel("Perturbation level (hPa)")
plt.ylabel("Response level (hPa)")
plt.colorbar()
plt.savefig("/home/b11209013/2025_Research/MSI/Fig/Rad/RRTMG_LRF_t_lw.png", dpi=300);
# plt.show()
plt.close()


plt.figure(figsize=(16,12))
plt.pcolormesh(levs,levs,LRF["t_sw"].T, vmin=-1, vmax=1, cmap='RdBu_r');
plt.xlim(1000,100)
plt.ylim(1000,100)
plt.xlabel("Perturbation level (hPa)")
plt.ylabel("Response level (hPa)")
plt.colorbar()
plt.savefig("/home/b11209013/2025_Research/MSI/Fig/Rad/RRTMG_LRF_t_sw.png", dpi=300);
# plt.show()
plt.close()
