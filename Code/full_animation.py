# This program is to plot out the animation of the evolutions
import os;
import numpy as np;
import h5py as h5;
from matplotlib import pyplot as plt;
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

def outer_time_einsum(A, B, out=None, dtype=None):
    """
    A: (x, t), B: (z,)
    Returns C: (x, z, t) with C[x,z,t] = A[x,t] * B[z]
    """
    if dtype is None:
        dtype = np.result_type(A, B)
    x, t = A.shape
    z    = B.size
    if out is None:
        out = np.empty((x, z, t), dtype=dtype, order='C')

    A_ = np.ascontiguousarray(A, dtype=dtype)   # contiguous, right dtype
    B_ = np.ascontiguousarray(B, dtype=dtype)

    # einsum writes directly into `out`
    np.einsum('xt,z->xzt', A_, B_, out=out, optimize=True)
    return out

def main():
    # load file
    fpath = "/home/b11209013/2025_Research/MSI/File/";

    with h5.File(fpath+"background.h5", "r") as f:

        bg_list = list(f.keys())


        background = { 
                key: f[key][:]
                for key in bg_list                 
                };

    with h5.File(fpath+"state_vector.h5", "r") as f:
        state_vec = f["state vector"][:]; # shape: (kn, var, t)
        kn        = f["wavenumber"][:];
        t         = f["time"][:];
        x         = f["x"][:];

    with h5.File(fpath+"vertical_mode.h5", "r") as f:
        z  = f["z"][:];
        G1 = f["G1"][:].squeeze();
        G2 = f["G2"][:].squeeze();


    with h5.File(fpath+"inverse_mat.h5", "r") as f:
        inv_mat = f["inverse matrix"][:]; # shape: (x, kn)
    
    # setup left and right boundary of x
    lft_bnd    = np.argmin(np.abs(x+4320000));
    rgt_bnd    = np.argmin(np.abs(x-4320000));

    # select specific wavenumber
    idx_8640   = np.argmin(np.abs(kn-2*np.pi*4320/8640));

    state_8640 = state_vec[idx_8640,:,:];
    inv_8640   = inv_mat[:,idx_8640];

    temp_PC    = np.einsum("vt,x->vxt", state_8640, inv_8640, optimize=True)[2:4,...][:,lft_bnd:rgt_bnd+1,:];

    temp_PC    = np.real(temp_PC);

    # reconstruct the vertical profile 
    PC1_prof  = outer_time_einsum(temp_PC[0].squeeze(), G1*(-0.0065+9.81/1004.5));

    PC2_prof  = outer_time_einsum(temp_PC[1].squeeze(), G2*(-0.0065+9.81/1004.5));

    Temp_prof = np.transpose((PC1_prof + PC2_prof)/background["rho"][None,:,None], (2,1,0));

    # Plot the animation for temperature evolution
    vmin = np.nanmin(Temp_prof);
    vmax = np.nanmax(Temp_prof);   
    
    fig, ax = plt.subplots(figsize=(16,9))

    prof = ax.pcolormesh( 
            x[lft_bnd:rgt_bnd+1], z, Temp_prof[0],
            cmap="RdBu_r", vmin=vmin, vmax=vmax,
            shading="auto", animated=True
            );
    ax.set_xticks(np.linspace(-4000000, 4000000, 5), ["-40", "-20", "0", "20", "40"], fontsize=20);
    ax.set_yticks(np.linspace(2000, 14000, 7), ["2", "4", "6", "8", "10", "12", "14"], fontsize=20);
    ax.set_xlim(-4000000, 4000000);
    ax.set_ylim(0, 15000);
    ax.set_xlabel("X [ 100 km ]", fontsize=24);
    ax.set_ylabel("Z [ km ]", fontsize=24);
    ax.set_title(r"$T^\prime$ ($\lambda$=8640km)", fontsize=32);

    cbar=fig.colorbar(prof,ax=ax,pad=0.02);
    cbar.set_label("T [K]");

    time_txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize=20)
    
    def init(t=t, prof=prof, temp=Temp_prof[0], time_txt=time_txt):
        prof.set_array(temp.ravel());
        
        time_txt.set_text(f"t = {t[0]}");

        return prof, time_txt

    def update(i, prof=prof, temp=Temp_prof):
        prof.set_array(temp[i].ravel())

        time_txt.set_text(f"t = {t[i]}")

        return prof, time_txt

    ani = FuncAnimation(
            fig, update, frames=len(t), init_func=init,
            interval=50, blit=False
            )

    ani.save("/home/b11209013/2025_Research/MSI/Fig/temp_prof.mp4", dpi=150, writer=FFMpegWriter(fps=50))

if __name__ == "__main__":
    main();
