# ================================================================================
# Regress radiative profile associated with moisture, temperature
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

    # ##################
    # Load data
    # ##################

    RAD_DIR: str = "/work/b11209013/2025_Research/MSI/Rad_Stuff/"
    SIM_DIR: str = "/work/b11209013/2025_Research/MSI/Sim_stuff/"

    # Load linear response function for q and T
    with h5py.File( RAD_DIR + "LRF.h5", "r" ) as f:
        q_lw_lrf: np.ndarray = np.array( f.get( "q_lw" ) )[ ::-1, ::-1 ] # 1000 -> 100
        q_sw_lrf: np.ndarray = np.array( f.get( "q_sw" ) )[ ::-1, ::-1 ]
        t_lw_lrf: np.ndarray = np.array( f.get( "t_lw" ) )[ ::-1, ::-1 ]
        t_sw_lrf: np.ndarray = np.array( f.get( "t_sw" ) )[ ::-1, ::-1 ]

    # Load moisture perturbaiton
    q_pert: np.ndarray = np.loadtxt( RAD_DIR + "mean_corr_moisture.txt" ) # 1000 -> 100

    # Load background density profile
    with h5py.File( SIM_DIR + "background.h5", "r" ) as f:
        rho0: np.ndarray = np.array( f.get( "ρ0" ) )
        T0  : np.ndarray = np.array( f.get( "T0" ) )

    # #########################################
    # Interpolate moisture perturbation profile
    # #########################################
    levels: np.ndarray = np.linspace( 1000, 100, 37 )

    q_pert_itp: np.ndarray = interp1d(
        [1000.0, 925.0, 850.0, 700.0, 500.0, 250.0, 200.0, 100.0],
        q_pert, kind = "linear",
        fill_value="extrapolate", bounds_error=False,  #type: ignore
        )( levels )

    # #########################################
    # Design vertical modes
    # #########################################
    levels_z: np.ndarray = pressure_to_height_std( levels*units.hPa ) * 1000.0

    G1: np.ndarray = np.asarray( ( np.pi/2 * np.sin( np.pi*levels_z/np.max(levels_z) ) )[ :, None ] )
    G2: np.ndarray = np.asarray( ( np.pi/2 * np.sin( 2*np.pi*levels_z/np.max(levels_z) ) )[ :, None ] )

    # ###########################
    # Interpolate density profile
    # ###########################

    rho_itp: np.ndarray = interp1d(
        np.linspace( 0.0, 14000.0, 141 ), rho0, kind = "linear",
        fill_value="extrapolate", bounds_error=False,  #type: ignore
        )( levels_z )

    # ########################################
    # Compute Radiative Heating corresponding to q and T
    # ########################################

    # Compute Moisture-induced part
    unit_factor: float = 1e-3*2.5e6/1004.5

    q_lw_rad: np.ndarray = ( q_lw_lrf @ q_pert_itp ).reshape( 1, -1 )*unit_factor
    q_sw_rad: np.ndarray = ( q_sw_lrf @ q_pert_itp ).reshape( 1, -1 )*unit_factor

    # Compute Temperature-induced part
    T1_lw_rad: np.ndarray = ( t_lw_lrf @ ( G1*0.0033 / rho_itp[:, None] ) ).T
    T2_lw_rad: np.ndarray = ( t_lw_lrf @ ( G2*0.0033 / rho_itp[:, None] ) ).T

    T1_sw_rad: np.ndarray = ( t_sw_lrf @ ( G1*0.0033 / rho_itp[:, None] ) ).T
    T2_sw_rad: np.ndarray = ( t_sw_lrf @ ( G2*0.0033 / rho_itp[:, None] ) ).T

    # ############################################
    # Decompose radiative heating for unit impulse
    # ############################################

    bases: np.ndarray = np.array( np.concatenate( ( G1, G2 ), axis=-1 ).T * 0.0033 )

    Rq_lw: np.ndarray = ( ( rho_itp[ None, : ]*q_lw_rad ) @ bases.T @ np.linalg.inv( bases @ bases.T ) ).squeeze()
    Rq_sw: np.ndarray = ( ( rho_itp[ None, : ]*q_sw_rad ) @ bases.T @ np.linalg.inv( bases @ bases.T ) ).squeeze()

    RT1_lw: np.ndarray = ( ( rho_itp[ None, : ]*T1_lw_rad ) @ bases.T @ np.linalg.inv( bases @ bases.T ) ).squeeze()
    RT2_lw: np.ndarray = ( ( rho_itp[ None, : ]*T2_lw_rad ) @ bases.T @ np.linalg.inv( bases @ bases.T ) ).squeeze()

    RT1_sw: np.ndarray = ( ( rho_itp[ None, : ]*T1_sw_rad ) @ bases.T @ np.linalg.inv( bases @ bases.T ) ).squeeze()
    RT2_sw: np.ndarray = ( ( rho_itp[ None, : ]*T2_sw_rad ) @ bases.T @ np.linalg.inv( bases @ bases.T ) ).squeeze()

    with h5py.File(
        f"{RAD_DIR}qt_coeff.h5", "w"
    ) as f:
        f.create_dataset( "Rq_lw" , data=Rq_lw )
        f.create_dataset( "Rq_sw" , data=Rq_sw )
        f.create_dataset( "RT1_lw", data=RT1_lw )
        f.create_dataset( "RT1_sw", data=RT1_sw )
        f.create_dataset( "RT2_lw", data=RT2_lw )
        f.create_dataset( "RT2_sw", data=RT2_sw )

    # ##############################################
    # Verification for the coefficients
    # ##############################################

    fig, axs = plt.subplots( 1, 3, figsize=(18, 10), sharey=True )

    axs[0].plot(
        ( rho_itp[ None, : ]*q_lw_rad ).squeeze(), levels,
        color="#085993", linewidth=2, linestyle="-"
    )
    axs[0].plot(
        ( Rq_lw[0]*bases[0] + Rq_lw[1]*bases[1] ).squeeze(), levels,
        color="#085993", linewidth=2, linestyle="--"
    )
    axs[0].plot(
        ( rho_itp[ None, : ]*q_sw_rad ).squeeze(), levels,
        color="#009C24", linewidth=2, linestyle="-"
    )
    axs[0].plot(
        ( Rq_sw[0]*bases[0] + Rq_sw[1]*bases[1] ).squeeze(), levels,
        color="#009C24", linewidth=2, linestyle="--"
    )
    axs[0].axvline( 0.0, color="k", linewidth=2, linestyle="--" )
    axs[0].spines["right"].set_visible(False)
    axs[0].spines["top"].set_visible(False)
    axs[0].set_xticks( np.linspace( -1, 1, 5 ) )
    axs[0].set_xticklabels(
        ["-1.0", "-0.5", "0.0", "0.5", "1.0"],
        fontsize=18
        )
    axs[0].set_yticks( [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100] )
    axs[0].set_yticklabels(
        ["1000", "900", "800", "700", "600", "500", "400", "300", "200", "100"],
        fontsize=18
        )
    axs[0].set_xlim( -1.02, 1.02 )
    axs[0].set_ylim( 1000, 100 )
    axs[0].set_title(
        "Moisture", fontsize=18
    )

    axs[1].plot(
        ( rho_itp[ None, : ]*T1_lw_rad ).squeeze(), levels,
        color="#085993", linewidth=2, linestyle="-"
    )
    axs[1].plot(
        ( RT1_lw[0]*bases[0] + RT1_lw[1]*bases[1] ).squeeze(), levels,
        color="#085993", linewidth=2, linestyle="--"
    )
    axs[1].plot(
        ( rho_itp[ None, : ]*T1_sw_rad ).squeeze(), levels,
        color="#009C24", linewidth=2, linestyle="-"
    )
    axs[1].plot(
        ( RT1_sw[0]*bases[0] + RT1_sw[1]*bases[1] ).squeeze(), levels,
        color="#009C24", linewidth=2, linestyle="--"
    )
    axs[1].axvline( 0.0, color="k", linewidth=2, linestyle="--" )
    axs[1].spines["right"].set_visible(False)
    axs[1].spines["top"].set_visible(False)
    axs[1].set_xticks( np.linspace( -0.00025, 0.00025, 3 ) )
    axs[1].set_xticklabels(
        ["-0.00025", "0.0", "0.00025"],
        fontsize=18
        )
    axs[1].set_xlim( -0.00028, 0.00028 )
    axs[1].set_ylim( 1000, 100 )
    axs[1].set_title(
        f"$T_1$", fontsize=18
    )

    axs[2].plot(
        ( rho_itp[ None, : ]*T2_lw_rad ).squeeze(), levels,
        color="#085993", linewidth=2, linestyle="-",
        label="LW Exact"
    )
    axs[2].plot(
        ( RT2_lw[0]*bases[0] + RT2_lw[1]*bases[1] ).squeeze(), levels,
        color="#085993", linewidth=2, linestyle="--",
        label="LW Approx."
    )
    axs[2].plot(
        ( rho_itp[ None, : ]*T2_sw_rad ).squeeze(), levels,
        color="#009C24", linewidth=2, linestyle="-",
        label="SW Exact"
    )
    axs[2].plot(
        ( RT2_sw[0]*bases[0] + RT2_sw[1]*bases[1] ).squeeze(), levels,
        color="#009C24", linewidth=2, linestyle="--",
        label="SW Approx."
    )
    axs[2].axvline( 0.0, color="k", linewidth=2, linestyle="--" )
    axs[2].spines["right"].set_visible(False)
    axs[2].spines["top"].set_visible(False)
    axs[2].set_xticks( np.linspace( -0.0003, 0.0003, 3 ) )
    axs[2].set_xticklabels(
        ["-0.0003", "0.0", "0.0003"],
        fontsize=18
        )
    axs[2].set_xlim( -0.00033, 0.00033 )
    axs[2].set_ylim( 1000, 100 )
    axs[2].set_title(
        f"$T_2$", fontsize=18
    )
    fig.legend( frameon=False, fontsize=18, loc="upper center", ncol=4 )

    plt.savefig("/home/b11209013/2025_Research/MSI/Fig/Rad/qt_regression.png", dpi=600, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
