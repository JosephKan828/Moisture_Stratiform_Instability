# This program is to compute corresponding radiative heating to unit mid-tropospheric moisture perturbation

# Import package
import h5py;
import numpy as np;
import matplotlib.pyplot as plt;


def main():
    # Load data
    fpath = "/home/b11209013/2025_Research/MSI/File/";

    ## Load LRF
    with h5py.File(fpath+"/LRF.h5", "r" ) as f:
        
        q_lw = f.get("q_lw")[...];
        q_sw = f.get("q_sw")[...];

    ## Load moisture correlation
    mean_corr = np.loadtxt(fpath+"/mean_corr_moisture.txt")

    mean_corr_itp = np.interp(
            np.linspace(100, 1000, 37), np.array([250, 500, 700, 850, 925, 1000]),
            mean_corr
            );


    lw_pert = np.array(q_lw) @ np.array(mean_corr_itp[:, None]);
    sw_pert = np.array(q_sw) @ np.array(mean_corr_itp[:, None]);


    plt.plot(lw_pert, np.linspace(100, 1000, 37))
    plt.ylim(1000, 100)
    plt.show()

if __name__ == "__main__":
    main();


