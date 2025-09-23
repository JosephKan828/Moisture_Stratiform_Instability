# This program is to regress the two vertical modes
# Import package
import h5py;
import numpy as np;

from metpy.units import units;
from metpy.calc import pressure_to_height_std;
from matplotlib import pyplot as plt;

def main():
    # ==== 1. load data ==== #
    fpath = "/home/b11209013/2025_Research/MSI/File/Rad_anom.h5";

    with h5py.File(fpath, "r") as f:
        lw = f.get("LW_pert")[:].squeeze();
        sw = f.get("SW_pert")[:].squeeze();

    # ==== 2. set up vertical normal modes ==== #
    levs = np.linspace(100, 1000, 37);

    lev_lim = np.argmin(np.abs(levs-300));

    z = np.array(pressure_to_height_std(levs*units.hPa))*1000.0;
    G1 = np.pi / 2.0 * np.sin(np.pi*z/z.max());
    G2 = np.pi / 2.0 * np.sin(2*np.pi*z/z.max());

    modes = np.stack([G1[lev_lim:], G2[lev_lim:]]);

    lw_coeff = (np.array(lw[lev_lim:]) @ np.array(modes.T)) @ np.linalg.inv(np.array(modes)@np.array(modes.T));
    sw_coeff = (np.array(sw[lev_lim:]) @ np.array(modes.T)) @ np.linalg.inv(np.array(modes)@np.array(modes.T));

    plt.plot(lw, levs);
    plt.plot(lw_coeff[0]*G1+lw_coeff[1]*G2, levs);
    plt.ylim(1000, 100)
    plt.show()

    plt.plot(sw, levs);
    plt.plot(sw_coeff[0]*G1+sw_coeff[1]*G2, levs);
    plt.ylim(1000, 100)
    plt.show()




if __name__ == "__main__":
    main();
