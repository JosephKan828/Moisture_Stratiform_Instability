#!/usr/bin/env python
# coding: utf-8

# ## Regress the profile against mid troposphere moisture

# ### Import package

import numpy as np;
import xarray as xr;

from metpy.calc import pressure_to_height_std
from metpy.units import units
from matplotlib import pyplot as plt;


# ### Load data
# file path
fpath="/work/b11209013/2024_Research/ERA5/q/q.nc";

# load data
ds = xr.open_dataset(fpath);

lev = ds["level"];
q = ds["q"].sel(lat=slice(-10,10), lon=slice(160,260)) * 1000.0;

ds.close();


# ### Find mean moisture profile
q_mean = q.mean(dim=["time","lat","lon"], skipna=True);
q_anom = q - q_mean;
q500a  = q.sel(level='500');

cov    = xr.cov(q_anom, q500a, dim="time");
var500 = q500a.var(dim="time");

beta = cov / var500;

mean_corr = np.array(beta.mean(["lat", "lon"]));

z_std = np.array(pressure_to_height_std((lev.values).astype(int) * units.hPa)) *1000.0

np.savetxt("/work/b11209013/2025_Research/MSI/Rad_Stuff/mean_corr_moisture.txt", mean_corr);

# Plot figure

plt.figure()

plt.plot(mean_corr, lev)
plt.ylim(1000, 100)
plt.savefig("/home/b11209013/2025_Research/MSI/Fig/Rad/corr_moist.png")
