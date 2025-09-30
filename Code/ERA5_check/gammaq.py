# This program is to compute relations of J1, J2, q

##############################
# 1. import package
##############################

import os;
import numpy as np;
import xarray as xr;
import matplotlib.pyplot as plt;

##############################
# 2. main function
##############################

def main():
    # ==== 1. read data ==== #
    base_path = "/work/b11209013/2024_Research/ERA5/"

    # load Q1
    with xr.open_dataset(os.path.join(base_path, "Q1/Q1.nc")) as ds:
        q1 = ds["Q1"].sel(lon=slice(160, 260)).isel(time=slice(None, 4000)).mean(dim="lat")*86400/1004.5;

    # load q
    with xr.open_dataset(os.path.join(base_path, "q/q.nc")) as ds:
        print(ds.plev.values)
        q = ds["q"].sel(lon=slice(160, 260)).isel(time=slice(None, 4000), plev=1).mean(dim="lat")*1000;

    # load z grid
    with xr.open_dataset(os.path.join(base_path, "z/z_2006.nc")) as ds:
        z = ds["z"].sel(lat=slice(-10, 10)).mean(dim={"time","lat","lon"});

    q1 -= q1.mean(dim={"time", "lon"});
    q  -= q.mean(dim={"time", "lon"});

    # ==== 2. compute vertical modes ==== #
    G1 = np.pi/2 * np.sin(np.pi*z/z.max());
    G2 = np.pi/2 * np.sin(2*np.pi*z/z.max());

    vert_mode = np.stack([G1, G2], axis=0); # (2, nz)

    # ==== 3. project Q1 onto vertical modes ==== #
    q1_trans = np.transpose(q1.values, (1, 0, 2)).reshape(q1.shape[1], -1); # (nz, nt*nlon)

    q1_proj = (q1_trans.T @ vert_mode.T) @ np.linalg.inv(vert_mode @ vert_mode.T); # (nt*nlon, 2)
    q1_proj = q1_proj.T.reshape(2, q1.shape[0], q1.shape[2]); # (2, nt, nlon)


    # ==== 4. plot out relations between q1 and q ==== #

    plt.figure(figsize=(8, 6))
    plt.scatter(q.values.flatten(), q1_proj[0].flatten(), color="b", s=1, alpha=0.5)
    plt.xlabel("q (g/kg)")
    plt.ylabel("J1 (K/day)")
    plt.title("Relation between J1 and q")
    plt.axhline(0, color='k', linestyle='--')
    plt.axvline(0, color='k', linestyle='--')
    plt.savefig("/home/b11209013/2025_Research/MSI/Fig/J1_q_relation.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(q.values.flatten(), q1_proj[1].flatten(), color="b", s=1, alpha=0.5)
    plt.xlabel("q (g/kg)")
    plt.ylabel("J2 (K/day)")
    plt.title("Relation between J2 and q")
    plt.axhline(0, color='k', linestyle='--')
    plt.axvline(0, color='k', linestyle='--')
    plt.savefig("/home/b11209013/2025_Research/MSI/Fig/J2_q_relation.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(q1_proj[0].flatten(), q1_proj[1].flatten(), color="b", s=1, alpha=0.5)
    plt.xlabel("J1 (K/day)")
    plt.ylabel("J2 (K/day)")
    plt.title("Relation between J2 and J1")
    plt.axhline(0, color='k', linestyle='--')
    plt.axvline(0, color='k', linestyle='--')
    plt.savefig("/home/b11209013/2025_Research/MSI/Fig/J1_J2_relation.png", dpi=300)
    plt.close()


###############################
# 3. execute main function
###############################

if __name__ == "__main__":
    main();