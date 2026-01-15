# ================================================================================
# Regress radiative profile associated with vertical motion
# ================================================================================

# #####################
# Import Package
# #####################

import h5py
import numpy as np
import matplotlib.pyplot as plt

from metpy.calc import pressure_to_height_std
from metpy.units import units
from scipy.interpolate import interp1d

def main() -> None:

    # ###############
    # Load data
    # ###############

    RAD_DIR: str = "/work/b11209013/2025_Research/MSI/Rad_Stuff/"
    SIM_DIR: str = "/work/b11209013/2025_Research/MSI/Sim_stuff/"

    # Load linear response function
    with h5py.File(
        f"{RAD_DIR}w_LRF.h5", "r"
    ) as f:
        EOF   : np.ndarray = np.array( f.get( "EOF" ) )    # shape: (nmode, nlev)
        LW_LRF: np.ndarray = np.array( f.get( "LRF_lw" ) )
        SW_LRF: np.ndarray = np.array( f.get( "LRF_sw" ) )

    nmode: int = EOF.shape[0]

    # Load background profile
    with h5py.File(
        f"{SIM_DIR}background.h5", "r"
    ) as f:
        ρ0: np.ndarray = np.array( f.get( "ρ0" ) )

    # ###########################
    # Interpolate vertical layers
    # ###########################

    # Design vertical coordinates
    levels  : np.ndarray = np.linspace( 1000, 100, 37 )
    levels_z: np.ndarray = np.asarray(
        pressure_to_height_std( levels*units.hPa ) * 1000.0
                                      )

    # Interpolate background density profile
    ρ0_itp  : np.ndarray = interp1d(
        np.linspace(0, 14000, 141),
        ρ0, kind="linear",
        fill_value="extrapolate"
    )(levels_z)

    # Calculate vertical normal modes
    G1      : np.ndarray = np.asarray(
        ( np.pi/2 * np.sin( np.pi*levels_z/np.max(levels_z) ) )
    )
    G2      : np.ndarray = np.asarray(
        ( np.pi/2 * np.sin( 2*np.pi*levels_z/np.max(levels_z) ) )
    )

    # #######################################
    # Decompose vertical normal mode into PCs
    # #######################################

    # Convert vertical normal mode into pressure velocity
    G1_ω : np.ndarray = np.asarray( -9.81*G1/86400.0 )[ None, : ]
    G2_ω : np.ndarray = np.asarray( -9.81*G2/86400.0 )[ None, : ]

    # Calculate PCs
    G1ω_pcs: np.ndarray = np.asarray(
        G1_ω @ EOF.T @ np.linalg.inv( EOF @ EOF.T )
    ).T
    G2ω_pcs: np.ndarray = np.asarray(
        G2_ω @ EOF.T @ np.linalg.inv( EOF @ EOF.T )
    ).T # Shape: (nmodes, nsamples)

    # ##########################################
    # Predict radiative heating and coefficients
    # ##########################################

    # Predict radiative heating profile
    lw_G1: np.ndarray = np.asarray(
        EOF.T @ ( LW_LRF @ G1ω_pcs )
    ).T
    lw_G2: np.ndarray = np.asarray(
        EOF.T @ ( LW_LRF @ G2ω_pcs )
    ).T
    sw_G1: np.ndarray = np.asarray(
        EOF.T @ ( SW_LRF @ G1ω_pcs )
    ).T
    sw_G2: np.ndarray = np.asarray(
        EOF.T @ ( SW_LRF @ G2ω_pcs )
    ).T

    # calculate coefficients
    bases : np.ndarray = np.asarray(
        np.stack( ( G1, G2 ), axis=0 )*0.0033
    )

    lw_G1_coeff: np.ndarray = np.asarray(
        ( ρ0_itp*lw_G1 ) @ bases.T @ np.linalg.inv( bases @ bases.T )
    ).squeeze()
    lw_G2_coeff: np.ndarray = np.asarray(
        ( ρ0_itp*lw_G2 ) @ bases.T @ np.linalg.inv( bases @ bases.T )
    ).squeeze()
    sw_G1_coeff: np.ndarray = np.asarray(
        ( ρ0_itp*sw_G1 ) @ bases.T @ np.linalg.inv( bases @ bases.T )
    ).squeeze()
    sw_G2_coeff: np.ndarray = np.asarray(
        ( ρ0_itp*sw_G2 ) @ bases.T @ np.linalg.inv( bases @ bases.T )
    ).squeeze()

    # ############################
    # Verification of coefficients
    # ############################

    fig, axs = plt.subplots(
        1, 2, figsize=( 12, 8 ),
        sharey = True
    )

    axs[0].plot(
        ( ρ0_itp * lw_G1 ).squeeze(), levels,
        color="#085993", linewidth=2, linestyle="-",
        label="LW Exact"
    )
    axs[0].plot(
        ( lw_G1_coeff[0]*bases[0] + lw_G1_coeff[1]*bases[1] ).squeeze(),
        levels,
        color="#085993", linewidth=2, linestyle="--",
        label="LW Approx."
    )
    axs[0].plot(
        ( ρ0_itp*sw_G1 ).squeeze(), levels,
        color="#009C24", linewidth=2, linestyle="-",
        label="SW Exact"
    )
    axs[0].plot(
        ( sw_G1_coeff[0]*bases[0] + sw_G1_coeff[1]*bases[1] ).squeeze(),
        levels,
        color="#009C24", linewidth=2, linestyle="--",
        label="SW Approx."
    )
    axs[0].axvline( 0.0, color="k", linewidth=2, linestyle="--" )
    axs[0].spines["right"].set_visible( False )
    axs[0].spines["top"].set_visible( False )
    axs[0].tick_params( axis="both", which="major", labelsize=18 )
    # axs[0].set_xlim( -90, 90 )
    axs[0].set_ylim( 1000, 100 )
    axs[0].set_title(
        "$w_1$", fontsize=18
    )

    axs[1].plot(
        ( ρ0_itp * lw_G2 ).squeeze(), levels,
        color="#085993", linewidth=2, linestyle="-"
    )
    axs[1].plot(
        ( lw_G2_coeff[0]*bases[0] + lw_G2_coeff[1]*bases[1] ).squeeze(),
        levels,
        color="#085993", linewidth=2, linestyle="--"
    )
    axs[1].plot(
        ( ρ0_itp*sw_G2 ).squeeze(), levels,
        color="#009C24", linewidth=2, linestyle="-"
    )
    axs[1].plot(
        ( sw_G2_coeff[0]*bases[0] + sw_G2_coeff[1]*bases[1] ).squeeze(),
        levels,
        color="#009C24", linewidth=2, linestyle="--"
    )
    axs[1].axvline( 0.0, color="k", linewidth=2, linestyle="--" )
    axs[1].spines["right"].set_visible( False )
    axs[1].spines["top"].set_visible( False )
    axs[1].tick_params( axis="both", which="major", labelsize=18 )
    axs[1].set_ylim( 1000, 100 )
    axs[1].set_title(
        "$w_2$", fontsize=18
    )

    fig.supxlabel( r"ρ0 $\times$ Radiative Heating", fontsize=20 )
    fig.supylabel( "Level [ hPa ]", fontsize=20 )
    fig.legend( frameon=False, fontsize=18, loc="upper center", ncol=4 )


    plt.savefig(
        f"/home/b11209013/2025_Research/MSI/Fig/Rad/w_regression_{EOF.shape[0]}modes.png",
        dpi=600, bbox_inches="tight"
    )
    plt.close()

    # ###################
    # Save Coefficients
    # ###################

    with h5py.File(
        f"{RAD_DIR}w_coeff_{EOF.shape[0]}modes.h5", "w"
    ) as f:
        f.create_dataset( "Rw1_lw", data=lw_G1_coeff )
        f.create_dataset( "Rw1_sw", data=sw_G1_coeff )
        f.create_dataset( "Rw2_lw", data=lw_G2_coeff )
        f.create_dataset( "Rw2_sw", data=sw_G2_coeff )

if __name__ == "__main__":
    main()
