# This program is to regress the two vertical modes
# Import package
import h5py;
import numpy as np;

from metpy.units import units;
from metpy.calc import pressure_to_height_std;
from matplotlib import pyplot as plt;
from scipy.interpolate import interp1d;

def main():
    # ==== 1. load data ==== #
    fpath = "/home/b11209013/2025_Research/MSI/File/Rad_anom.h5";

    with h5py.File(fpath, "r") as f:
        lw = np.array(f.get("LW_pert")).squeeze();
        sw = np.array(f.get("SW_pert")).squeeze();

    with h5py.File("/home/b11209013/2025_Research/MSI/File/bg_field.h5", "r") as f:
        rho0 = np.array(f.get("ρ0")).squeeze();
        z_itp = np.array(f.get("z")).squeeze()[::-1];

    # ==== 2. set up vertical normal modes ==== #
    levs = np.linspace(100, 1000, 37);

    lev_lim = np.argmin(np.abs(levs-300));

    z         = np.array(pressure_to_height_std(levs*units.hPa))*1000.0;
    z300      = z[lev_lim];

    G1 = np.pi / 2.0 * np.sin(np.pi*z_itp/z_itp.max())* 0.0065;
    G2 = np.pi / 2.0 * np.sin(2*np.pi*z_itp/z_itp.max())* 0.0065;

    modes = np.stack([G1[z_itp<=z300], G2[z_itp<=z300]], axis=0);

    # ==== 3. interpolate lw and sw to z_itp_lim ==== #
    lw_itp_func = interp1d(z, lw, kind="linear", fill_value="extrapolate");
    sw_itp_func = interp1d(z, sw, kind="linear", fill_value="extrapolate");

    lw_itp = lw_itp_func(z_itp);
    sw_itp = sw_itp_func(z_itp);

    lw_coeff = (np.array((rho0*lw_itp)[z_itp<=z300]) @ np.array(modes.T)) @ np.linalg.inv(np.array(modes)@np.array(modes.T));
    sw_coeff = (np.array((rho0*sw_itp)[z_itp<=z300]) @ np.array(modes.T)) @ np.linalg.inv(np.array(modes)@np.array(modes.T));

    plt.figure(figsize=(12, 16))
    plt.plot(rho0*lw_itp, z_itp, color="k", label="LW");
    plt.plot(rho0*sw_itp, z_itp, color="b", label="SW");
    plt.plot(lw_coeff[0]*G1+lw_coeff[1]*G2, z_itp, "k--", label="LW Regress");
    plt.plot(sw_coeff[0]*G1+sw_coeff[1]*G2, z_itp, "b--", label="SW Regress");
    plt.legend(fontsize=18);
    plt.xlabel("ρ0 * Radiative Heating Anomaly (K kg m$^{-3}$)", fontsize=20);
    plt.ylabel("Height (m)", fontsize=20);
    plt.xticks(fontsize=18);
    plt.yticks(fontsize=18);
    plt.xlim(-0.7, 0.7);
    plt.ylim(0, 15000);
    plt.title("Regression of Radiative Heating Anomaly", fontsize=28);
    plt.ylim(0, 15000);
    plt.text(-0.6, 500, "LW Regression Coeff.: \n"+f"{lw_coeff}", fontsize=18);
    plt.text(-0.6, 5000, "SW Regression Coeff.: \n"+f"{sw_coeff}", fontsize=18);
    plt.grid()
    plt.savefig("/home/b11209013/2025_Research/MSI/Fig/Regress_rad.png", dpi=300);
    plt.close();

if __name__ == "__main__":
    main();
