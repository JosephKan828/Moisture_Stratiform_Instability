# This program is to compute corresponding radiative heating to unit mid-tropospheric moisture perturbation

# Import package
import h5py;
import numpy as np;
import matplotlib.pyplot as plt;
from scipy.interpolate import interp1d

def main():
    # Load data
    fpath = "/home/b11209013/2025_Research/MSI/File/";

    ## Load LRF
    with h5py.File(fpath+"/LRF.h5", "r" ) as f:

        q_lw = np.array(f.get("q_lw"))
        q_sw = np.array(f.get("q_sw"))
        t_lw = np.array(f.get("t_lw"))
        t_sw = np.array(f.get("t_sw"))

    ## Load moisture correlation
    mean_moist = np.loadtxt(fpath+"Rad/mean_corr_moisture.txt")
    mean_temp  = np.loadtxt(fpath+"Rad/mean_corr_temperature.txt")



    mean_moist_itp = np.interp(
            np.linspace(100, 1000, 37), np.array([250, 500, 700, 850, 925, 1000]),
            mean_moist);
    
    mean_temp_itp = interp1d(
            np.array([250, 500, 700, 850, 925, 1000])[::-1], mean_temp, kind='linear', fill_value="extrapolate"
            )(np.linspace(100, 1000, 37)[::-1])[::-1]

    plt.figure(figsize=(6, 8))
    plt.plot(mean_moist_itp, np.linspace(100, 1000, 37))
    plt.ylim(1000, 100)
    plt.xlabel("Correlation")
    plt.ylabel("Levels")
    plt.savefig("/home/b11209013/2025_Research/MSI/Fig/Rad/corr_moist.png")
    # plt.show()
    plt.close()

    plt.figure(figsize=(6, 8))
    plt.plot(mean_temp_itp, np.linspace(100, 1000, 37))
    plt.ylim(1000, 100)
    plt.xlabel("Correlation")
    plt.ylabel("Levels")
    plt.savefig("/home/b11209013/2025_Research/MSI/Fig/Rad/corr_temp.png")
    # plt.show()
    plt.close()

    lw_moist_pert = np.array(q_lw.T) @ np.array(mean_moist_itp[:, None]);
    sw_moist_pert = np.array(q_sw.T) @ np.array(mean_moist_itp[:, None]);
    lw_temp_pert = np.array(t_lw.T) @ np.array(mean_temp_itp[:, None]);
    sw_temp_pert = np.array(t_sw.T) @ np.array(mean_temp_itp[:, None]);

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
    plt.plot(lw_temp_pert, np.linspace(100, 1000, 37))
    plt.ylim(1000, 100)
    plt.xlabel("LW anomalies")
    plt.ylabel("Levels")
    plt.savefig("/home/b11209013/2025_Research/MSI/Fig/Rad/LW_temp_pert.png")
    # plt.show()
    plt.close()


    plt.figure(figsize=(6, 8))
    plt.plot(sw_temp_pert, np.linspace(100, 1000, 37))
    plt.ylim(1000, 100)
    plt.xlabel("SW anomalies")
    plt.ylabel("Levels")
    plt.savefig("/home/b11209013/2025_Research/MSI/Fig/Rad/SW_temp_pert.png")
    # plt.show()
    plt.close()

    with h5py.File(fpath+"Rad/Rad_anom.h5", "w" ) as f:
        f.create_dataset("LW_moist_pert", data=lw_moist_pert);
        f.create_dataset("SW_moist_pert", data=sw_moist_pert);
        f.create_dataset("LW_temp_pert", data=lw_temp_pert);
        f.create_dataset("SW_temp_pert", data=sw_temp_pert);

if __name__ == "__main__":
    main();


