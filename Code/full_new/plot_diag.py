# This program is to plot out the growth rate and phase speed of the most unstable mode
# import package
import h5py;
import numpy as np;

from matplotlib import pyplot as plt;

# load data
file = "/home/b11209013/2025_Research/MSI/File/diagnose.h5"; 

with h5py.File(file, 'r') as f:
    λ = np.array(f.get("λ"));
    growth = np.array(f.get("growth_rate"));
    speed = np.array(f.get("phase_speed"));

pos_σ = np.array(np.where(growth>0));

unstable_λ = λ[pos_σ[0]];
unstable_speed = speed[pos_σ[0], pos_σ[1]];

# plot
plt.figure(figsize=(16,9));
plt.plot(40000/λ, np.max(growth, axis=1), label="Growth Rate", color='black');
plt.xticks(np.linspace(0, 30, 7), fontsize=18);
plt.yticks(np.linspace(0, 0.12, 7), fontsize=18);
plt.xlim(0, 30);
plt.ylim(0, 0.13);
plt.xlabel("Non-dimensional Wavelength (2π/40000km)", fontsize=24);
plt.ylabel("Growth Rate (1/day)", fontsize=24);
plt.title("Growth Rate of the Most Unstable Mode", fontsize=28);
plt.grid();
plt.savefig("/home/b11209013/2025_Research/MSI/Fig/Full_growth_rate.png", dpi=300);
plt.close()

plt.figure(figsize=(16,9));
for i in range(speed.shape[1]):
    plt.scatter(40000/λ, speed[:,i], label=f"Mode {i+1}", s=10, color='black');
for i in range(speed.shape[1]):
    plt.scatter(40000/unstable_λ, unstable_speed, marker="o", s=40, facecolors='none', edgecolors='black');
plt.xticks(np.linspace(0, 30, 7), fontsize=18);
plt.yticks(np.linspace(0, 60, 8), fontsize=18);
plt.xlim(0, 30)
plt.ylim(-5, 60)
plt.xlabel("Non-dimensional Wavelength (2π/40000km)", fontsize=24);
plt.ylabel("Phase Speed (m/s)", fontsize=24);
plt.title("Phase Speed of All Modes", fontsize=28);
plt.grid();
plt.savefig("/home/b11209013/2025_Research/MSI/Fig/Full_phase_speed.png", dpi=300);
plt.close()