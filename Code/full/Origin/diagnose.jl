# This program is to compute the growth rate and phase speed of dynamic system
# using package
using Pkg; Pkg.activate("/work/b11209013/external/JuliaENV/atmo");
using HDF5;
using Base.Threads;
using LinearAlgebra, Statistics;

include("/home/b11209013/2025_Research/MSI/src/LinearModel.jl")
using .LinearModel

# setup parameters
params = default_params("conv_only");

function diagnostics(kn::Float64)
    mat = coeff_matrix(kn; param=params);
    eigval = eigvals(coeff_matrix(kn; param=params));
    σ = real.(eigval);
    c = -imag.(eigval)./kn * (4320000/86400);
    return σ, c
end

# main code
λ :: Vector{Float64} = collect(540:540:43200);
k :: Vector{Float64} = @. 2.0*π*4320.0 / λ;

Nk, Nv = length(k), 6;

growth = zeros(Float64, Nv, Nk)
speed  = zeros(Float64, Nv, Nk)

@threads for i in eachindex(k)
    σ, c = diagnostics(k[i]);
    growth[:, i] = σ;
    speed[:, i]  = c;
end

# save file
h5open("/work/b11209013/2025_Research/MSI/Full/Origin/diagnose.h5", "w") do f
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