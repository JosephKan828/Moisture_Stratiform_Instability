# This program is to reconstruct coefficient matrix and eigenvalues

using Pkg; Pkg.activate("/work/b11209013/external/JuliaENV/atmo")
using HDF5
using Tullio
using LinearAlgebra, Statistics

include("/home/b11209013/2025_Research/MSI/src/Galerkin_Fourier.jl")
using .Galerkin_Fourier

#####################
# Import necessary data
#####################

rad_scaling = 0.1

FPATH_INPUT :: String = "/work/b11209013/2025_Research/MSI/Full/Rad/state_rad_"*string(rad_scaling)*".h5"
FPATH_SIM   :: String = "/work/b11209013/2025_Research/MSI/Sim_stuff/"
FPATH_OUTPUT:: String = "/work/b11209013/2025_Research/MSI/Full/Rad/"

# load state vector
state, t, k, vars = h5open(FPATH_INPUT, "r") do f
    state = read(f, "state vector");
    t     = read(f, "time");
    k     = read(f, "wavenumber");
    vars  = read(f, "variables");
    return state, t, k, vars
end

# Load inverse matrix
inv_mat, x = h5open(joinpath(FPATH_SIM, "inv_mat.h5"), "r") do f
    inv_mat = read(f, "inverse matrix");
    x       = read(f, "x");

    return inv_mat, x
end

# Load Vertical mode
G1, G2, z = h5open(joinpath(FPATH_SIM, "vertical_mode.h5"), "r") do f
    G1 = read(f, "G1");
    G2 = read(f, "G2");
    z  = read(f, "z");

    return G1, G2, z
end

# Load background field
ρ0 = h5open(joinpath(FPATH_SIM, "background.h5"), "r") do f
    ρ0 = read(f, "ρ0");

    return ρ0
end

#####################
# Setup parameters
#####################
# Target wavelength and wavenumber
λ    :: Float64 = 8640.0; # wavelength in km
k0   :: Float64 = 2.0*π*4320.0 / λ; # non-dimensional wavenumber in km/km
kidx :: Int = findfirst(isequal(k0), k); # index for k0

# find left and right boundary index to limit domain
lft, rgt = findfirst(x .>= -4320000), findfirst(x .>= 4320000);

#####################
# Compute PC 
#####################

# Compute coefficient for each variable
Fourier_basis::Vector{ComplexF64} = inv_mat[kidx, lft:rgt]          # length X

@show size(state)

Fourier_coeff = Dict(
    var => Fourier_Reconstruct(
        state[:,i,kidx], Fourier_basis
        ) 
    for (i,var) in enumerate(vars)
)

# rewrite q and L to J1 and J2
Fourier_coeff["Rq1"] = @. 5.61*Fourier_coeff["q"]
Fourier_coeff["Rq2"] = @. 3.36*Fourier_coeff["q"]
Fourier_coeff["J1"] = @. 2*Fourier_coeff["L"] + 0.7*Fourier_coeff["q"]
Fourier_coeff["J2"] = @. -0.7*Fourier_coeff["q"]

# Compute profile for each variable
w1_prof, w2_prof = Galerkin_Reconstruct(
    Fourier_coeff["w1"], Fourier_coeff["w2"],
    -0.0065, ρ0, G1, G2
)
T1_prof, T2_prof = Galerkin_Reconstruct(
    Fourier_coeff["T1"], Fourier_coeff["T2"],
    -0.0065, ρ0, G1, G2
)
J1_prof, J2_prof = Galerkin_Reconstruct(
    Fourier_coeff["J1"], Fourier_coeff["J2"],
    -0.0065, ρ0, G1, G2
)

R1_prof, R2_prof = Galerkin_Reconstruct(
    Fourier_coeff["Rq1"], Fourier_coeff["Rq2"],
    -0.0065, ρ0, G1, G2
)

# --- combine w ---
w = @. w1_prof + w2_prof
T = @. T1_prof + T2_prof
J = @. J1_prof + J2_prof
R = @. R1_prof + R2_prof

w_r = real.(w)  # (T,X,Z)
T_r = real.(T)  # (T,X,Z)
J_r = real.(J)  # (T,X,Z)
R_r = real.(R)  # (T,X,Z)

println(size(w))

h5open(joinpath(FPATH_OUTPUT, "reconstruction_rad_"*string(rad_scaling)*".h5"), "w") do f
    write(f, "w", w_r)
    write(f, "T", T_r)
    write(f, "J", J_r)
    write(f, "R", R_r)
    write(f, "x", x[lft:rgt])
    write(f, "z", z)
    write(f, "t", t)
end 