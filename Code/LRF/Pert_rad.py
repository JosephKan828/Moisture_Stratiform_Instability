# This program is to compute corresponding radiative heating to unit mid-tropospheric moisture perturbation

# Import package
import h5py;
import numpy as np;
import matplotlib.pyplot as plt;
from metpy.calc import pressure_to_height_std
from metpy.units import units
from scipy.interpolate import interp1d

def main():
    # Load data
    fpath = "/home/b11209013/2025_Research/MSI/File/";

    ## Load LRF
    with h5py.File(fpath+"Rad/LRF.h5", "r" ) as f:

        q_lw = np.array(f.get("q_lw"))
        q_sw = np.array(f.get("q_sw"))
        t_lw = np.array(f.get("t_lw"))
        t_sw = np.array(f.get("t_sw"))


    # vertical profile
    with h5py.File(fpath+"Sim_stuff/vertical_mode.h5","r") as f:
        G1      = np.array(f.get("G1")).squeeze(); # shape: (1, z)
        G2      = np.array(f.get("G2")).squeeze(); # shape: (1, z)
        z       = np.array(f.get("z"));            # shape: (z,)

    plt.plot(G1, z)
    plt.plot(G2, z)
    plt.savefig("/home/b11209013/2025_Research/MSI/Fig/vertical_mode.png")
    plt.close()
    
    ## design vertical coordinates
    levels = np.linspace(100, 1000, 37)
    z_std = np.array(pressure_to_height_std((levels).astype(int) * units.hPa)) *1000.0

    ## Load moisture correlation
    mean_moist = np.loadtxt(fpath+"Rad/mean_corr_moisture.txt")
    # mean_temp  = np.loadtxt(fpath+"Rad/mean_corr_temperature.txt")

    mean_moist_itp = np.interp(
            np.linspace(100, 1000, 37), np.array([250, 500, 700, 850, 925, 1000]),
            mean_moist);
    
    G1_itp = np.interp(z_std, z, G1)*0.0065;
    G2_itp = np.interp(z_std, z, G2)*0.0065;    

    # mean_temp_itp = interp1d(
    #         np.array([250, 500, 700, 850, 925, 1000])[::-1], mean_temp, kind='linear', fill_value="extrapolate"
    #         )(np.linspace(100, 1000, 37)[::-1])[::-1]

    plt.figure(figsize=(6, 8))
    plt.plot(mean_moist_itp, np.linspace(100, 1000, 37))
    plt.ylim(1000, 100)
    plt.xlabel("Correlation")
    plt.ylabel("Levels")
    plt.savefig("/home/b11209013/2025_Research/MSI/Fig/Rad/corr_moist.png")
    # plt.show()
    plt.close()

    lw_moist_pert = np.array(q_lw) @ np.array(mean_moist_itp[:, None]);
    sw_moist_pert = np.array(q_sw) @ np.array(mean_moist_itp[:, None]);
    lw_g1_pert = np.array(t_lw) @ np.array(G1_itp[:, None]);
    sw_g1_pert = np.array(t_sw) @ np.array(G1_itp[:, None]);
    lw_g2_pert = np.array(t_lw) @ np.array(G2_itp[:, None]);
    sw_g2_pert = np.array(t_sw) @ np.array(G2_itp[:, None]);

    plt.figure(figsize=(6, 8))
    plt.plot(lw_moist_pert, np.linspace(100, 1000, 37))
    plt.ylim(1000, 100)
    plt.xlabel("LW anomalies")
    plt.ylabel("Levels")
    plt.savefig("/home/b11209013/2025_Research/MSI/Fig/Rad/LW_moist_pert.png")
    # plt.show()
    plt.close()


    plt.figure(figsize=(6, 8))
    plt.plot(sw_moist_pert, np.linspace(100, 1000, 37))
    plt.ylim(1000, 100)
    plt.xlabel("SW anomalies")
    plt.ylabel("Levels")
    plt.savefig("/home/b11209013/2025_Research/MSI/Fig/Rad/SW_moist_pert.png")
    # plt.show()
    plt.close()

    plt.figure(figsize=(6, 8))
    plt.plot(lw_g1_pert, np.linspace(100, 1000, 37))
    plt.ylim(1000, 100)
    plt.xlabel("LW anomalies by unit T1")
    plt.ylabel("Levels")
    plt.savefig("/home/b11209013/2025_Research/MSI/Fig/Rad/LW_G1_pert.png")
    # plt.show()
    plt.close()


    plt.figure(figsize=(6, 8))
    plt.plot(sw_g1_pert, np.linspace(100, 1000, 37))
    plt.ylim(1000, 100)
    plt.xlabel("SW anomalies by unit T1")
    plt.ylabel("Levels")
    plt.savefig("/home/b11209013/2025_Research/MSI/Fig/Rad/SW_G1_pert.png")
    # plt.show()
    plt.close()

    plt.figure(figsize=(6, 8))
    plt.plot(lw_g2_pert, np.linspace(100, 1000, 37))
    plt.ylim(1000, 100)
    plt.xlabel("LW anomalies by unit T2")
    plt.ylabel("Levels")
    plt.savefig("/home/b11209013/2025_Research/MSI/Fig/Rad/LW_G2_pert.png")
    # plt.show()
    plt.close()


    plt.figure(figsize=(6, 8))
    plt.plot(sw_g2_pert, np.linspace(100, 1000, 37))
    plt.ylim(1000, 100)
    plt.xlabel("SW anomalies by unit T2")
    plt.ylabel("Levels")
    plt.savefig("/home/b11209013/2025_Research/MSI/Fig/Rad/SW_G2_pert.png")
    # plt.show()
    plt.close()

    with h5py.File(fpath+"Rad/Rad_anom.h5", "w" ) as f:
        f.create_dataset("LW_moist_pert", data=lw_moist_pert);
        f.create_dataset("SW_moist_pert", data=sw_moist_pert);
        f.create_dataset("LW_g1_pert", data=lw_g1_pert);
        f.create_dataset("SW_g1_pert", data=sw_g1_pert);
        f.create_dataset("LW_g2_pert", data=lw_g2_pert);
        f.create_dataset("SW_g2_pert", data=sw_g2_pert);

if __name__ == "__main__":
    main();


