# This program is to compute the tropical mean profile of temperature and moisture from ERA5;
# Import package
import h5py;
import numpy as np;
import xarray as xr;

def main():
    # ==== 1. load data ==== #
    fpath = "/work/DATA/Satellite/CloudSat/ERA5/"

    with xr.open_dataset(fpath+"q.nc") as f:
        f = f.sel(lon=slice(160, 260), lat=slice(-10,10));

        q_mean = f["q"].mean(dim={"time","lon","lat"}).values;

    with xr.open_dataset(fpath+"t.nc") as f:
        f = f.sel(lon=slice(160, 260), lat=slice(-10,10));

        t_mean = f["t"].mean(dim={"time","lon","lat"}).values;



    # ==== 2. save file ==== #
    with h5py.File("/home/b11209013/2025_Research/MSI/File/Rad/mean_state.h5", "w") as f:
        f.create_dataset("q", data=q_mean);
        f.create_dataset("t", data=t_mean);

if __name__ == "__main__":
    main();
