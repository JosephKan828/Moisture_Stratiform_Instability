# This program is to plot the phase relation among T1, T2, q

######################
# 1. import package
######################

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

######################
# 2. Load dataset
######################

FPATH_INPUT  = "/work/b11209013/2025_Research/MSI/Full/Origin/"
FPATH_SIM    = "/work/b11209013/2025_Research/MSI/Sim_stuff/"
FPATH_OUTPUT = "/home/b11209013/2025_Research/MSI/Fig/"

# Load state file
with h5py.File(FPATH_INPUT + "state.h5", "r") as f:
   k     = np.array(f.get("wavenumber"))
   state = np.array(f.get("state vector"))

# Load inverse matrix
with h5py.File(FPATH_SIM+"inv_mat.h5", "r") as f:
   inv_mat = np.array(f.get("inverse matrix"))

# setup index
target_wl = 8640.0
target_k  = 2.0 * np.pi * 4320.0 / target_wl
kidx      = np.argmin(np.abs(k - target_k))

state     = state[kidx]
inv_mat   = inv_mat[:,kidx]

# select specific series
T1 = state[2]; T2 = state[3]; q = state[4]

print(T1.shape, T2.shape, q.shape)

T1 = T1[:,None]*inv_mat[None,:]
T2 = T2[:,None]*inv_mat[None,:]
q  = q[:,None]*inv_mat[None,:]

#######################
# 3. Plot
#######################

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "font.size": 14,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.linewidth": 1.2,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "legend.frameon": False,
    "legend.fontsize": 13,
    "mathtext.default": "regular",  # keep ascii clean; toggle to 'regular' for non-TeX
})
########################
# User config / inputs #
########################

lam_km      = 8640.0                                       # wavelength label
out_path    = FPATH_OUTPUT + "Full/Origin/phase_anim.mp4"  # output movie
fps         = 40
bitrate     = 8000

# If you already have time array, set it here.
# Otherwise we just use frame index as "time".
t_arr = np.arange(T1.shape[0], dtype=float)

# x-grid for plotting: same as your static figure
nt, nx = T1.shape
x_full = np.linspace(-40000.0, 40000.0, nx)

ymax = np.nanmax(np.abs(np.array([T1.real, T2.real, q.real])))

pad = 0.01*ymax
ax_ylim = [-ymax-pad, ymax+pad]
########################
# Precompute constants #
########################

tmax = float(t_arr[-1])

# xticks in "100 km" units like before
xticks_raw  = np.linspace(-4000, 4000, 5)
xticks_labs = ["-40","-20","0","20","40"]

#################################
# Set up the initial figure     #
#################################

fig, ax = plt.subplots(figsize=(10.5, 6.2))

# grab frame 0
T1_now = np.asarray(T1[0]).real
T2_now = np.asarray(T2[0]).real
q_now  = np.asarray(q [0]).real

# plot initial lines and KEEP the handles so we can update .set_ydata()
line1, = ax.plot(x_full, T1_now, color="k", lw=1.5, label=r"$T_1$")
line2, = ax.plot(x_full, T2_now, color="r", lw=1.5, label=r"$T_2$")
line3, = ax.plot(x_full, q_now,  color="b", lw=1.5, label=r"$q$")

# axis labels
ax.set_xlabel("X [ 100 km ]", fontsize=18)
ax.set_ylabel("Amplitude [ arbitrary units ]", fontsize=18)

# ticks / limits
ax.set_xticks(xticks_raw)
ax.set_xticklabels(xticks_labs, fontsize=16)
ax.set_xlim(-4320, 4320)
ax.tick_params(axis="y", labelsize=16)

ax.set_ylim(*ax_ylim)

# dashed horizontal guides only
ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)

# legend
ax.legend(frameon=False, fontsize=16, loc="upper right", handlelength=2.5)

# title template
max_all0 = np.nanmax(np.abs([T1_now, T2_now, q_now]))
title = ax.set_title(
    r"$\lambda$ = "
    f"{lam_km:.0f} km   "
    r"$t$ = "
    f"{t_arr[0]:.1f}/{tmax:.1f}   "
    r"$\max(|T_1|,|T_2|,|q|)$ = "
    f"{max_all0:.2f}",
    fontsize=18
)

# polish_axes equivalent
ax.tick_params(direction="in", length=6, width=1.1,
                top=True, right=True, labelsize=16)
for spine in ax.spines.values():
    spine.set_linewidth(1.1)

fig.tight_layout()

#################################
# Update function per frame     #
#################################

def update(frame_i):
    # get new data at this time
    T1_now = np.asarray(T1[frame_i]).real
    T2_now = np.asarray(T2[frame_i]).real
    q_now  = np.asarray(q [frame_i]).real

    # update y-data of the 3 lines
    line1.set_ydata(T1_now)
    line2.set_ydata(T2_now)
    line3.set_ydata(q_now)

    # update title
    max_all = np.nanmax(np.abs([T1_now, T2_now, q_now]))
    title.set_text(
        r"$\lambda$ = "
        f"{lam_km:.0f} km   "
        r"$t$ = "
        f"{t_arr[frame_i]:.1f}/{tmax:.1f}   "
        r"$\max(|T_1|,|T_2|,|q|)$ = "
        f"{max_all:.2f}"
    )

    # return artists that changed (not strictly required if blit=False,
    # but nice to be explicit)
    return (line1, line2, line3, title)

#################################
# Build and save animation      #
#################################

ani = FuncAnimation(
    fig,
    update,
    frames=nt,
    interval=1000.0 / fps,   # ms between frames
    blit=False               # line plots + title are cheap, keep False for simplicity
)

writer = FFMpegWriter(fps=fps, bitrate=bitrate,
                        extra_args=["-pix_fmt", "yuv420p"])
ani.save(out_path, writer=writer)

plt.close(fig)