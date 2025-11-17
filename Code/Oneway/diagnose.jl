# diagnose_oneway_combined.jl
# ------------------------------------------------------------
# Single-way wave solution diagnostics (modal + nonmodal)
# - Eigenvalue growth rates (all modes)
# - Phase speeds (all modes)
# - Max growth rate per k
# - Nonmodal amplification via matrix exponential
# ------------------------------------------------------------

using Pkg
Pkg.activate("/work/b11209013/external/JuliaENV/atmo")

using HDF5
using LinearAlgebra
using Statistics

using Base.Threads;

include("/home/b11209013/2025_Research/MSI/src/LinearModel.jl")
using .LinearModel

# -----------------------------
# Parameters
# -----------------------------
params = default_params("conv_only")

# -----------------------------
# Diagnostics per k
# -----------------------------
function diagnostics(kn::Float64)
    mat = coeff_matrix(kn; param=params, mode="oneway");
    eigval = eigvals(mat);
    σ = real.(eigval);
    c = -imag.(eigval)./kn * (4320000/86400);
    return σ, c
end

# -----------------------------
# Main
# -----------------------------

# design wavenumber
λ = collect(540.0:540.0:43200.0)         # wavelength [km]
k = @. 2.0 * π * 4320.0 / λ              # nondimensional wavenumber

Nk, Nv = length(k), 4

growth = zeros(Float64, Nv, Nk)
speed  = zeros(Float64, Nv, Nk)

@threads for i in eachindex(k)
    σ, c = diagnostics(k[i]);
    growth[:, i] = σ;
    speed[:, i]  = c;
end

# save file
h5open("/work/b11209013/2025_Research/MSI/oneway/origin/diagnose.h5", "w") do f
    write(f, "λ", λ);
    write(f, "k", k);
    write(f, "growth_rate", growth);
    write(f, "phase_speed", speed);

    attributes(f["λ"])["standard_name"] = "wavelength";
    attributes(f["λ"])["units"] = "km";

    attributes(f["k"])["standard_name"] = "non-dimensional wavenumber";
    attributes(f["k"])["units"] = "km/km";

    attributes(f["growth_rate"])["standard_name"] = "modal growth rate";
    attributes(f["growth_rate"])["units"] = "1/day";

    attributes(f["phase_speed"])["standard_name"] = "phase speed";
    attributes(f["phase_speed"])["units"] = "m/s";
end