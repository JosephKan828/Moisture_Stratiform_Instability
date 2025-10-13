# This program is to regress the two vertical modes
# Import package
import h5py;
import numpy as np;

from metpy.units import units;
from metpy.calc import pressure_to_height_std;
import matplotlib as mpl
from matplotlib import pyplot as plt;
from matplotlib.ticker import AutoMinorLocator
from scipy.interpolate import interp1d;

def set_scientific_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.titlesize": 28,
        "axes.linewidth": 1.2,
        "axes.titlesize": 24,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 16,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "grid.color": "0.85",
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
        "font.family": "DejaVu Sans",
    })

def add_common_formatting(ax, xlabel, ylabel, xlim, ylim):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    # zero-line for reference
    ax.axvline(0, lw=1.0, color="0.35", alpha=0.8)

def annotate_coeff(ax, label, coeff_array, xy_axes=(0.03, 0.15)):
    """Place text in axes coordinates to avoid overlap when limits change."""
    txt = f"{label}:\n{coeff_array}"
    ax.text(
        xy_axes[0], xy_axes[1], txt,
        transform=ax.transAxes, fontsize=16,
        va="bottom", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.8")
    )

def main():
    # ==== 1. load data ==== #
    fpath = "/home/b11209013/2025_Research/MSI/File/Rad/Rad_anom.h5";

    with h5py.File(fpath, "r") as f:
        lw_q  = np.array(f.get("LW_moist_pert")).squeeze();
        sw_q  = np.array(f.get("SW_moist_pert")).squeeze();
        lw_g1 = np.array(f.get("LW_g1_pert")).squeeze();
        sw_g1 = np.array(f.get("SW_g1_pert")).squeeze();
        lw_g2 = np.array(f.get("LW_g2_pert")).squeeze();
        sw_g2 = np.array(f.get("SW_g2_pert")).squeeze();

    with h5py.File("/home/b11209013/2025_Research/MSI/File/Sim_stuff/background.h5", "r") as f:
        rho0 = np.array(f.get("ρ0")).squeeze()[::-1];
        z_itp = np.array(f.get("z")).squeeze()[::-1];
    
    # ==== 2. set up vertical normal modes ==== #
    levs = np.linspace(100, 1000, 37);

    # lev_lim = np.argmin(np.abs(levs-300));

    z         = np.asarray(pressure_to_height_std(levs * units.hPa).to('m').m)
    # z300      = z[lev_lim];

    G1 = np.pi / 2.0 * np.sin(np.pi*z_itp/z_itp.max())* (-0.0065 + 9.8/1004.5);
    G2 = np.pi / 2.0 * np.sin(2*np.pi*z_itp/z_itp.max())* (-0.0065 + 9.8/1004.5);

    modes = np.stack([G1, G2], axis=0);

    # ==== 3. interpolate lw and sw to z_itp_lim ==== #
    lw_itp_q  = interp1d(z, lw_q, kind="linear", fill_value="extrapolate")(z_itp); #type: ignore
    sw_itp_q  = interp1d(z, sw_q, kind="linear", fill_value="extrapolate")(z_itp); #type: ignore

    lw_itp_g1 = interp1d(z, lw_g1, kind="linear", fill_value="extrapolate")(z_itp); #type: ignore
    sw_itp_g1 = interp1d(z, sw_g1, kind="linear", fill_value="extrapolate")(z_itp); #type: ignore

    lw_itp_g2 = interp1d(z, lw_g2, kind="linear", fill_value="extrapolate")(z_itp); #type: ignore
    sw_itp_g2 = interp1d(z, sw_g2, kind="linear", fill_value="extrapolate")(z_itp); #type: ignore
    
    lw_q_coeff = (np.array((rho0*lw_itp_q)) @ np.array(modes.T)) @ np.linalg.inv(np.array(modes)@np.array(modes.T));
    sw_q_coeff = (np.array((rho0*sw_itp_q)) @ np.array(modes.T)) @ np.linalg.inv(np.array(modes)@np.array(modes.T));

    lw_g1_coeff = (np.array((rho0*lw_itp_g1)) @ np.array(modes.T)) @ np.linalg.inv(np.array(modes)@np.array(modes.T));
    sw_g1_coeff = (np.array((rho0*sw_itp_g1)) @ np.array(modes.T)) @ np.linalg.inv(np.array(modes)@np.array(modes.T));

    lw_g2_coeff = (np.array((rho0*lw_itp_g2)) @ np.array(modes.T)) @ np.linalg.inv(np.array(modes)@np.array(modes.T));
    sw_g2_coeff = (np.array((rho0*sw_itp_g2)) @ np.array(modes.T)) @ np.linalg.inv(np.array(modes)@np.array(modes.T));

    set_scientific_style()

    ###########################
    # ==== 4. plot results ==== #
    ###########################

    # plot LW and SW regression results for moisture perturbation
    fig = plt.figure(figsize=(12.0, 16.0))  # narrower width for profile plots
    ax = fig.add_subplot(111)

    ax.plot(rho0*lw_itp_q, z_itp, color="black", lw=2.0, label="LW")
    ax.plot(rho0*sw_itp_q, z_itp, color="#1f77b4", lw=2.0, label="SW")
    ax.plot(lw_q_coeff[0]*G1 + lw_q_coeff[1]*G2, z_itp, "k--", lw=2.0, label="LW (regress)")
    ax.plot(sw_q_coeff[0]*G1 + sw_q_coeff[1]*G2, z_itp, "--", color="#1f77b4", lw=2.0, label="SW (regress)")

    add_common_formatting(
        ax,
        xlabel=r"$\rho_0 \times$ Radiative Heating Anomaly (K kg m$^{-3}$)",
        ylabel="Height (m)",
        xlim=(-0.7, 0.7),
        ylim=(0, 15000),
    )

    annotate_coeff(ax, "LW coeff", lw_q_coeff, xy_axes=(0.05, 0.78))
    annotate_coeff(ax, "SW coeff", sw_q_coeff, xy_axes=(0.05, 0.60))

    # place legend outside to reduce occlusion
    leg = ax.legend(loc="upper left", frameon=False)
    plt.suptitle("Regression of Radiative Heating Anomaly (Moist)")

    out1_png = "/home/b11209013/2025_Research/MSI/Fig/Rad/Regress_q_rad.png"
    out1_pdf = "/home/b11209013/2025_Research/MSI/Fig/Rad/Regress_q_rad.pdf"
    fig.savefig(out1_png)
    fig.savefig(out1_pdf)
    plt.close(fig)
    

    # plot LW and SW regression results for g1 perturbation
    fig, ax1 = plt.subplots(figsize=(12.0, 16.0))

    ax1.plot(rho0*lw_itp_g1, z_itp, color="black", lw=2.0, label="LW")
    ax1.plot(lw_g1_coeff[0]*G1 + lw_g1_coeff[1]*G2, z_itp, "k--", lw=2.0, label="LW (regress)")

    add_common_formatting(
        ax1,
        xlabel=r"$\rho_0 \times$ LW (K kg m$^{-3}$)",
        ylabel="Height (m)",
        xlim=(-8e-4, 8e-4),
        ylim=(0, 15000),
    )
    annotate_coeff(ax1, "LW coeff", lw_g1_coeff, xy_axes=(0.05, 0.78))
    ax1.legend(loc="upper left", frameon=False)

    # twin axis for SW on top
    ax2 = ax1.twiny()
    ax2.plot(rho0*sw_itp_g1, z_itp, color="#1f77b4", lw=2.0, label="SW")
    ax2.plot(sw_g1_coeff[0]*G1 + sw_g1_coeff[1]*G2, z_itp, "--", color="#1f77b4", lw=2.0, label="SW (regress)")

    # keep top axis formatting minimal, with its own limits
    ax2.set_xlim(-6e-6, 6e-6)
    ax2.tick_params(axis="x", labelsize=18, pad=4)
    ax2.set_xlabel(r"$\rho_0 \times$ SW (K kg m$^{-3}$)")
    # Add a zero-line on the top axis view as well
    ax2.axvline(0, lw=1.0, color="0.35", alpha=0.8)
    # annotation for SW on the right upper corner
    annotate_coeff(ax2, "SW coeff", sw_g1_coeff, xy_axes=(0.05, 0.60))
    ax2.legend(loc="upper right", frameon=False)

    fig.suptitle("Regression of Radiative Heating Anomaly (T1)")
    out2_png = "/home/b11209013/2025_Research/MSI/Fig/Rad/Regress_g1_rad.png"
    out2_pdf = "/home/b11209013/2025_Research/MSI/Fig/Rad/Regress_g1_rad.pdf"
    fig.savefig(out2_png)
    fig.savefig(out2_pdf)
    plt.close(fig)

    # ================== Figure 3: g2 with twin x-axes ==================
    fig, ax1 = plt.subplots(figsize=(12.0, 16.0))

    ax1.plot(rho0*lw_itp_g2, z_itp, color="black", lw=2.0, label="LW")
    ax1.plot(lw_g2_coeff[0]*G1 + lw_g2_coeff[1]*G2, z_itp, "k--", lw=2.0, label="LW (regress)")

    add_common_formatting(
        ax1,
        xlabel=r"$\rho_0 \times$ LW (K kg m$^{-3}$)",
        ylabel="Height (m)",
        xlim=(-1.2e-3, 1.2e-3),
        ylim=(0, 15000),
    )
    annotate_coeff(ax1, "LW coeff", lw_g2_coeff, xy_axes=(0.05, 0.78))
    ax1.legend(loc="upper left", frameon=False)

    ax2 = ax1.twiny()
    ax2.plot(rho0*sw_itp_g2, z_itp, color="#1f77b4", lw=2.0, label="SW")
    ax2.plot(sw_g2_coeff[0]*G1 + sw_g2_coeff[1]*G2, z_itp, "--", color="#1f77b4", lw=2.0, label="SW (regress)")

    ax2.set_xlim(-1e-5, 1e-5)
    ax2.tick_params(axis="x", labelsize=18, pad=4)
    ax2.set_xlabel(r"$\rho_0 \times$ SW (K kg m$^{-3}$)")
    ax2.axvline(0, lw=1.0, color="0.35", alpha=0.8)
    annotate_coeff(ax2, "SW coeff", sw_g2_coeff, xy_axes=(0.05, 0.60))
    ax2.legend(loc="upper right", frameon=False)

    fig.suptitle("Regression of Radiative Heating Anomaly (T2)")
    out3_png = "/home/b11209013/2025_Research/MSI/Fig/Rad/Regress_g2_rad.png"
    out3_pdf = "/home/b11209013/2025_Research/MSI/Fig/Rad/Regress_g2_rad.pdf"
    fig.savefig(out3_png)
    fig.savefig(out3_pdf)
    plt.close(fig)

if __name__ == "__main__":
    main();
