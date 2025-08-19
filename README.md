# Moisture Stratiform Instability

> Repo Author: Yu-Chuan Kan\
> Affiliation: Dept. of Atmospheric Sciences, National Taiwan University, Taipei, Taiwan \
> Reference: Kuang, Z., 2008: A Moisture-Stratiform Instability for Convectively Coupled Waves. J. Atmos. Sci., 65, 834–854, https://doi.org/10.1175/2007JAS2444.1.

## Table of Content

* [Background](#bg)
* [Toy Model](#model)

## Background <a name="bg"></a>

This mechanism is to setup a theoretical framework for explaining the self-maintaining and self-amplifying of gravity-wave-mode (GWM) disturbances.\
With constructing a toy model, this study shows several key results:
1. Gravity-wave-mode disturbances can be simulated by only the first two vertical modes.
2. 2nd-mode temperature profile and mid-tropospheric moisture deficit are important in the GWM growth.

## Toy Model <a name="model"></a>

The model setup needs to consider:
1. Response to convective heating.
2. Convection Parameterisation.

### Response to Convective Heating 

Primitive equations is stemmed from linear, inviscid anelastic 2D primitive equation. We assume background state is resting, and written equations depending on the wavenumber:

$$
\begin{cases}
    & \left( \frac{\partial}{\partial t} + ϵ \right) \left( \bar{\rho}w^\prime \right)_{zz} = -k^2 \bar{\rho}g \frac{T^\prime}{\bar{T}} \\
    \\
    & \frac{\partial}{\partial t}T^\prime + w^\prime \left( \frac{d\bar{T}}{dz} + \frac{g}{C_p} \right) = J^\prime
\end{cases}
$$
