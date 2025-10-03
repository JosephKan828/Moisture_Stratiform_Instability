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
    fpath = "/home/b11209013/2025_Research/MSI/File/Rad/Rad_anom.h5";

    with h5py.File(fpath, "r") as f:
        lw_q = np.array(f.get("LW_moist_pert")).squeeze();
        sw_q = np.array(f.get("SW_moist_pert")).squeeze();
        lw_t = np.array(f.get("LW_temp_pert")).squeeze();
        sw_t = np.array(f.get("SW_temp_pert")).squeeze();

    with h5py.File("/home/b11209013/2025_Research/MSI/File/bg_field.h5", "r") as f:
        rho0 = np.array(f.get("ρ0")).squeeze();
        z_itp = np.array(f.get("z")).squeeze()[::-1];

    # ==== 2. set up vertical normal modes ==== #
    levs = np.linspace(100, 1000, 37);

    lev_lim = np.argmin(np.abs(levs-300));

    z         = np.asarray(pressure_to_height_std(levs * units.hPa).to('m').m)
    z300      = z[lev_lim];

    G1 = np.pi / 2.0 * np.sin(np.pi*z_itp/z_itp.max())* 0.0065;
    G2 = np.pi / 2.0 * np.sin(2*np.pi*z_itp/z_itp.max())* 0.0065;

    modes = np.stack([G1[z_itp<=z300], G2[z_itp<=z300]], axis=0);

    # ==== 3. interpolate lw and sw to z_itp_lim ==== #
    lw_itp_q = interp1d(z, lw_q, kind="linear", fill_value="extrapolate")(z_itp); #type: ignore
    sw_itp_q = interp1d(z, sw_q, kind="linear", fill_value="extrapolate")(z_itp); #type: ignore

    lw_itp_t = interp1d(z, lw_t, kind="linear", fill_value="extrapolate")(z_itp); #type: ignore
    sw_itp_t = interp1d(z, sw_t, kind="linear", fill_value="extrapolate")(z_itp); #type: ignore

    lw_q_coeff = (np.array((rho0*lw_itp_q)[z_itp<=z300]) @ np.array(modes.T)) @ np.linalg.inv(np.array(modes)@np.array(modes.T));
    sw_q_coeff = (np.array((rho0*sw_itp_q)[z_itp<=z300]) @ np.array(modes.T)) @ np.linalg.inv(np.array(modes)@np.array(modes.T));

    lw_t_coeff = (np.array((rho0*lw_itp_t)[z_itp<=z300]) @ np.array(modes.T)) @ np.linalg.inv(np.array(modes)@np.array(modes.T));
    sw_t_coeff = (np.array((rho0*sw_itp_t)[z_itp<=z300]) @ np.array(modes.T)) @ np.linalg.inv(np.array(modes)@np.array(modes.T));

    plt.figure(figsize=(12, 16))
    plt.plot(rho0*lw_itp_q, z_itp, color="k", label="LW");
    plt.plot(rho0*sw_itp_q, z_itp, color="b", label="SW");
    plt.plot(lw_q_coeff[0]*G1+lw_q_coeff[1]*G2, z_itp, "k--", label="LW Regress");
    plt.plot(sw_q_coeff[0]*G1+sw_q_coeff[1]*G2, z_itp, "b--", label="SW Regress");
    plt.legend(fontsize=18);
    plt.xlabel("ρ0 * Radiative Heating Anomaly (K kg m$^{-3}$)", fontsize=20);
    plt.ylabel("Height (m)", fontsize=20);
    plt.xticks(fontsize=18);
    plt.yticks(fontsize=18);
    plt.xlim(-0.7, 0.7);
    plt.ylim(0, 15000);
    plt.title("Regression of Radiative Heating Anomaly (Moist)", fontsize=28);
    plt.ylim(0, 15000);
    plt.text(-0.6, 500, "LW Regression Coeff.: \n"+f"{lw_q_coeff}", fontsize=18);
    plt.text(-0.6, 5000, "SW Regression Coeff.: \n"+f"{sw_q_coeff}", fontsize=18);
    plt.grid()
    plt.savefig("/home/b11209013/2025_Research/MSI/Fig/Rad/Regress_q_rad.png", dpi=300);
    plt.close();

    fig, ax1 = plt.subplots(figsize=(12, 16))
    ax1.plot(rho0*lw_itp_t, z_itp, color="k", label="LW Regress")
    ax1.plot(lw_t_coeff[0]*G1+lw_t_coeff[1]*G2, z_itp, "k--", label="LW Regress");
    ax1.set_xlabel("ρ0 * Radiative Heating Anomaly (K kg m$^{-3}$)", fontsize=20);
    ax1.set_ylabel("Height (m)", fontsize=20);
    ax1.set_xlim(-0.1, 0.1);
    ax1.set_ylim(0, 15000);
    ax1.tick_params(axis='x', labelsize=18);
    ax1.tick_params(axis='y', labelsize=18);
    # ax1.set_title("Regression of Radiative Heating Anomaly (Temp)", fontsize=28);
    ax1.text(0.025, 3000, "LW Regression Coeff.: \n"+f"{lw_t_coeff}", fontsize=18);
    ax1.grid()
    ax1.legend(loc="upper left", fontsize=18);

    ax2 = ax1.twiny()
    ax2.plot(rho0*sw_itp_t, z_itp, color="b", label="SW Regress")
    ax2.plot(sw_t_coeff[0]*G1+sw_t_coeff[1]*G2, z_itp, "b--", label="SW Regress");
    # ax2.set_xlabel("ρ0 * Radiative Heating Anomaly (K kg m$^{-3}$)", fontsize=20);
    ax2.set_xlim(-0.0004, 0.0004);
    ax2.text(0.0001, 2000, "SW Regression Coeff.: \n"+f"{sw_t_coeff}", fontsize=18);
    ax2.tick_params(axis='x', color="b", labelsize=18);
    ax2.legend(loc="upper right", fontsize=18);

    plt.suptitle("Regression of Radiative Heating Anomaly (Temp)", fontsize=28);

    plt.savefig("/home/b11209013/2025_Research/MSI/Fig/Rad/Regress_t_rad.png", dpi=300);
    plt.close();

if __name__ == "__main__":
    main();
