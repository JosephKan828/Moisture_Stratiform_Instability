# =============================================
# Construct Linear Response Function with RRTMG
# =============================================

# #####################
# Import Package
# #####################

import h5py
import numpy as np
import climlab as clab
import matplotlib

from copy import deepcopy
from matplotlib import pyplot as plt

# matplotlib.use( "TkAgg" )
# #####################
# Load data
# #####################

WORK_DIR: str = "/work/b11209013/2025_Research/MSI/Rad_Stuff/"

# Load mean state of moisture and temperature
with h5py.File( WORK_DIR+"mean_state.h5", "r" ) as f:
    q_mean: np.ndarray = np.array( f.get("q") )[::-1]
    t_mean: np.ndarray = np.array( f.get("t") )[::-1]

print( "Data loaded" )

# ############################
# Construct Referenced Profile
# ############################

# vertical levels
levels  : np.ndarray = np.linspace( 100, 1000, 37 )
n_levels: int        = len( levels )

# construct column state
state = clab.column_state(
    lev = levels,
    water_depth = 1.0
)

## Set temperature profile
state["Tatm"][:] = t_mean
state["Ts"][:]   = state["Tatm"][-1]

# Calculate reference radiative heating profile
ref_radiation = clab.radiation.RRTMG( #type: ignore
    name              = "reference profile",
    state             = state,
    specific_humidity = q_mean,
    albedo            = 0.3
)

ref_radiation.compute_diagnostics()

LW_ref: np.ndarray = np.array( ref_radiation.diagnostics[ "TdotLW" ].copy() )
SW_ref: np.ndarray = np.array( ref_radiation.diagnostics[ "TdotSW" ].copy() )

print( "Reference profile constructed" )

# ##################################
# Construct Linear Response Function
# ##################################

# Pre-allocate LRF dictionary
LRF: dict[str, np.ndarray] = {
    "q_lw": np.zeros( ( n_levels, n_levels ) ),
    "q_sw": np.zeros( ( n_levels, n_levels ) ),
    "t_lw": np.zeros( ( n_levels, n_levels ) ),
    "t_sw": np.zeros( ( n_levels, n_levels ) )
}

# Pre-allocate perturbed radiative heating profile
rad_pert_profile: dict[str, np.ndarray] = {
    "q_lw": np.zeros( ( n_levels, n_levels ) ),
    "q_sw": np.zeros( ( n_levels, n_levels ) ),
    "t_lw": np.zeros( ( n_levels, n_levels ) ),
    "t_sw": np.zeros( ( n_levels, n_levels ) )
}

# perturbed radiative heating
for l in range(n_levels):
    # perturb specific humidity
    q_pert        : np.ndarray = deepcopy( q_mean )
    q_perturbation: float      = q_pert[l] * 0.01
    q_pert[l]                 += q_perturbation

    rad_perturb = clab.radiation.RRTMG( #type: ignore
        name              = "perturbed profile",
        state             = state,
        specific_humidity = q_pert,
        albedo            = 0.3
    )

    rad_perturb.compute_diagnostics()

    lw_pert: np.ndarray = np.array(rad_perturb.diagnostics[ "TdotLW" ].copy())
    sw_pert: np.ndarray = np.array(rad_perturb.diagnostics[ "TdotSW" ].copy())
    
    rad_pert_profile["q_lw"][l, :] = lw_pert
    rad_pert_profile["q_sw"][l, :] = sw_pert

    LRF["q_lw"][l, :] = (lw_pert - LW_ref) / q_perturbation * 1e-3
    LRF["q_sw"][l, :] = (sw_pert - SW_ref) / q_perturbation * 1e-3

    del q_pert, q_perturbation, rad_perturb, lw_pert, sw_pert

    # perturb temperature
    state_pert = deepcopy( state )
    state_pert["Tatm"][l] += 1
    state_pert["Ts"][:] = state_pert["Tatm"][-1]

    rad_perturb = clab.radiation.RRTMG( #type: ignore
        name              = "perturbed profile",
        state             = state_pert,
        specific_humidity = q_mean,
        albedo            = 0.3
    )

    rad_perturb.compute_diagnostics()

    rad_pert_profile["t_lw"][l, :] = np.array(rad_perturb.diagnostics[ "TdotLW" ].copy())
    rad_pert_profile["t_sw"][l, :] = np.array(rad_perturb.diagnostics[ "TdotSW" ].copy())

    LRF["t_lw"][l, :] = (rad_pert_profile["t_lw"][l, :] - LW_ref) / 1
    LRF["t_sw"][l, :] = (rad_pert_profile["t_sw"][l, :] - SW_ref) / 1

    del state_pert, rad_perturb

print( "LRF constructed" )

# ############################
# Plot Linear Response Function
# ############################

fig, ax = plt.subplots( 
    2, 2,
    sharex=True, sharey=True,
    figsize=(20,24) 
    )

pc = ax[0, 0].pcolormesh(
    levels, levels, LRF["q_lw"].T,
    cmap="RdBu_r", vmin=-1, vmax=1
)
ax[0, 0].set_xlim( 1000, 100 )
ax[0, 0].set_ylim( 1000, 100 )
ax[0, 0].set_yticks(np.linspace(1000, 100, 10))
ax[0, 0].set_yticklabels(
    ["1000","900","800","700","600","500","400","300","200","100"],
    fontsize=20
    )
ax[0, 0].set_title( "LW from q", fontsize=24 )
ax[0, 0].grid(linestyle="--")

pc = ax[0, 1].pcolormesh(
    levels, levels, LRF["q_sw"].T,
    cmap="RdBu_r", vmin=-1, vmax=1
)
ax[0, 1].set_xlim( 1000, 100 )
ax[0, 1].set_ylim( 1000, 100 )
ax[0, 1].set_title( "SW from q", fontsize=24 )
ax[0, 1].grid(linestyle="--")

pc = ax[1, 0].pcolormesh(
    levels, levels, LRF["t_lw"].T,
    cmap="RdBu_r", vmin=-1, vmax=1
)
ax[1, 0].set_xlim( 1000, 100 )
ax[1, 0].set_ylim( 1000, 100 )
ax[1, 0].set_xticks(np.linspace(1000, 100, 10))
ax[1, 0].set_xticklabels(
    ["1000","900","800","700","600","500","400","300","200","100"],
    fontsize=20
)
ax[1, 0].set_yticks(np.linspace(1000, 100, 10))
ax[1, 0].set_yticklabels(
    ["1000","900","800","700","600","500","400","300","200","100"],
    fontsize=20
)
ax[1, 0].set_title( "LW from T", fontsize=24 )
ax[1, 0].grid(linestyle="--")

pc = ax[1, 1].pcolormesh(
    levels, levels, LRF["t_sw"].T,
    cmap="RdBu_r", vmin=-1, vmax=1
)
ax[1, 1].set_xticks(np.linspace(1000, 100, 10))
ax[1, 1].set_xticklabels(
    ["1000","900","800","700","600","500","400","300","200","100"],
    fontsize=20
)
ax[1, 1].set_xlim( 1000, 100 )
ax[1, 1].set_ylim( 1000, 100 )
ax[1, 1].set_title( "SW from T", fontsize=24 )
ax[1, 1].grid(linestyle="--")

cbar = fig.colorbar( pc, ax=ax, orientation="horizontal", shrink=0.8, aspect=50, pad=0.05 )
cbar.set_ticks( [-1, -0.5, 0, 0.5, 1] )
cbar.set_ticklabels( ["-1", "-0.5", "0", "0.5", "1"], fontsize=20 )

plt.savefig( "/home/b11209013/2025_Research/MSI/Fig/Rad/qt_LRF.png", dpi=600, bbox_inches="tight" )
plt.close()

print( "LRF plot saved" )

# ######################
# Plot difference
# ######################

rad_diff: dict[ str, np.ndarray ] = {
    "q_lw": np.zeros( ( n_levels, n_levels ) ),
    "q_sw": np.zeros( ( n_levels, n_levels ) ),
    "t_lw": np.zeros( ( n_levels, n_levels ) ),
    "t_sw": np.zeros( ( n_levels, n_levels ) )
}

for l in range( n_levels ):
    q_pert = np.zeros( n_levels )
    q_pert[l] = q_mean[l] * 0.01

    t_pert = np.zeros( n_levels )
    t_pert[l] += 1

    q_lw_rad_approx = ( LRF["q_lw"].T @ q_pert[:, None] ).squeeze() + LW_ref
    q_sw_rad_approx = ( LRF["q_sw"].T @ q_pert[:, None] ).squeeze() + SW_ref

    t_lw_rad_approx = ( LRF["t_lw"].T @ t_pert[:, None] ).squeeze() + LW_ref
    t_sw_rad_approx = ( LRF["t_sw"].T @ t_pert[:, None] ).squeeze() + SW_ref

    rad_diff["q_lw"][l, :] = q_lw_rad_approx - rad_pert_profile["q_lw"][l, :]
    rad_diff["q_sw"][l, :] = q_sw_rad_approx - rad_pert_profile["q_sw"][l, :]

    rad_diff["t_lw"][l, :] = t_lw_rad_approx - rad_pert_profile["t_lw"][l, :]
    rad_diff["t_sw"][l, :] = t_sw_rad_approx - rad_pert_profile["t_sw"][l, :]

    del q_pert, t_pert, q_lw_rad_approx, q_sw_rad_approx, t_lw_rad_approx, t_sw_rad_approx

fig, axs = plt.subplots( 4, 10, figsize=(20, 20), sharey=True )

axs = axs.flatten()

for i in range( n_levels ):
    axs[i].plot( rad_diff["q_lw"][i, :], levels, color="b" )
    axs[i].plot( rad_diff["q_sw"][i, :], levels, color="r" )

    axs[i].plot( rad_diff["t_lw"][i, :], levels, color="b", linestyle="--" )
    axs[i].plot( rad_diff["t_sw"][i, :], levels, color="r", linestyle="--" )
    axs[i].set_xlim(-0.01, 0.01)
    axs[i].set_ylim(1000, 100)

plt.savefig( "/home/b11209013/2025_Research/MSI/Fig/Rad/rad_diff.png", dpi=600, bbox_inches="tight" )
plt.close()

print( "Rad diff plot saved" )

# ###################
# Save LRF
# ###################
with h5py.File( WORK_DIR + "LRF.h5", "w" ) as f:
    for k, v in LRF.items():
        f.create_dataset( k, data=v.T )