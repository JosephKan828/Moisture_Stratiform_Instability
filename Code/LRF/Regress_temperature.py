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
fpath="/data92/b11209013/ERA5/File/t.nc";

# load data
ds = xr.open_dataset(fpath);

lev = ds["plev"];
t = ds["t"].sel(lat=slice(-10,10), lon=slice(160,260));

ds.close();


# ### Find mean moisture profile
t_mean = t.mean(dim=["time","lat","lon"], skipna=True);
t_anom = t - t_mean;
t500a  = t.sel(plev=500);

cov    = xr.cov(t_anom, t500a, dim="time");
var500 = t500a.var(dim="time");

beta = cov / var500;

plt.plot(beta.mean(["lat", "lon"]))
# plt.colorbar()
plt.show()

mean_corr = np.array(beta.mean(["lat", "lon"]));

z_std = np.array(pressure_to_height_std((lev.values).astype(int) * units.hPa)) *1000.0

np.savetxt("/work/b11209013/2025_Research/MSI/Rad_Stuff/mean_corr_temperature.txt", mean_corr);

G1 = np.pi/2 * np.sin(np.pi * z_std/z_std.max()) * 0.0033
G2 = np.pi/2 * np.sin(2 * np.pi * z_std/z_std.max()) * 0.0033

coeff_1 = np.inner(mean_corr, G1) / np.inner(G1, G1)
coeff_2 = np.inner(mean_corr, G2) / np.inner(G2, G2)

print("coeff_1:", coeff_1)
print("coeff_2:", coeff_2)