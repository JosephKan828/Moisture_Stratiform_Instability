#!/bin/sh

# =====================================================================
# This program is to iterate reconstruction.jl over several wavenumbers
# =====================================================================

JULIA_SCRIPT="simulation.jl"

rad_scales=(0.001 0.005 0.01 0.03 0.05 0.1)

for rad_scale in ${rad_scales[@]}; do

	echo ${rad_scale} "started"
	julia ${JULIA_SCRIPT} ${rad_scale}
	echo ${rad_scale} "done"

done
