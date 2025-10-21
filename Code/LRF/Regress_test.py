import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


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

with h5py.File("/work/b11209013/2025_Research/MSI/Sim_stuff/background.h5", "r") as f:
    rho0 = np.array(f.get("ρ0")).squeeze()[::-1];
    z_itp = np.array(f.get("z")).squeeze()[::-1];

G1 = np.pi/2 * np.sin(np.pi * z_itp/z_itp.max()) * 0.0033
G2 = np.pi/2 * np.sin(2 * np.pi * z_itp/z_itp.max()) * 0.0033

unit_moisture = (5.61 * G1 + 3.36 * G2) / 0.402 / rho0
unit_G1       = (-0.042 * G1 + -0.0087 * G2) / rho0
unit_G2       = (-0.011 * G1 + -0.069 * G2) / rho0


# ==== 1. unit moisture response ==== #
set_scientific_style()

fig = plt.figure(figsize=(12.0, 16.0))  # narrower width for profile plots
ax = fig.add_subplot(111)

ax.plot(unit_moisture, z_itp, label="Moisture", color="tab:blue")
add_common_formatting(ax, 
    "Radiative heating [K/day]",
    "Height [m]",
    (-0.15, 0.15),
    (0, 15000),
    )
plt.title("Radiative Heating Response to Unit Moisture Perturbation")
plt.savefig("/home/b11209013/2025_Research/MSI/Fig/Rad/unit_moisture.png", dpi=300);
plt.close()

fig = plt.figure(figsize=(12.0, 16.0))  # narrower width for profile plots
ax = fig.add_subplot(111)

ax.plot(unit_G1, z_itp, label="T1", color="tab:blue")
add_common_formatting(ax, 
    "Radiative heating [K/day]",
    "Height [m]",
    (-0.0005, 0.0005),
    (0, 15000),
    )
plt.title("Radiative Heating Response to Unit T1")
plt.savefig("/home/b11209013/2025_Research/MSI/Fig/Rad/unit_G1.png", dpi=300);
plt.close()

fig = plt.figure(figsize=(12.0, 16.0))  # narrower width for profile plots
ax = fig.add_subplot(111)

ax.plot(unit_G2, z_itp, label="T2", color="tab:blue")
add_common_formatting(ax, 
    "Radiative heating [K/day]",
    "Height [m]",
    (-0.001, 0.001),
    (0, 15000),
    )
plt.title("Radiative Heating Response to Unit T2")
plt.savefig("/home/b11209013/2025_Research/MSI/Fig/Rad/unit_G2.png", dpi=300);
plt.close()