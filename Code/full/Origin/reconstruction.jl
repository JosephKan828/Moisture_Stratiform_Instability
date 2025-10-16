# This program is to reconstruct coefficient matrix and eigenvalues

using Pkg; Pkg.activate("/work/b11209013/external/JuliaENV/atmo")
using HDF5
using Tullio
using LinearAlgebra, Statistics

#####################
# Import necessary data
#####################

FPATH_INPUT :: String = "/home/b11209013/2025_Research/MSI/File/Full/state.h5"
FPATH_SIM   :: String = "/home/b11209013/2025_Research/MSI/File/Sim_stuff/"
FPATH_OUTPUT:: String = "/home/b11209013/2025_Research/MSI/File/Full/"

# load state vector
state, t, k = h5open(FPATH_INPUT, "r") do f
    state = read(f, "state vector");
    t     = read(f, "time");
    k     = read(f, "wavenumber");
    return state, t, k
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
# reconstruct wind field
@views w1_coeff = state[:,1,kidx]
@views w2_coeff = state[:,2,kidx]
@views T1_coeff = state[:,3,kidx]
@views T2_coeff = state[:,4,kidx]
@views q_coeff  = state[:,5,kidx]
@views L_coeff  = state[:,6,kidx]
@views b  = inv_mat[kidx, lft:rgt]          # length X

T = length(w1_coeff)
X = length(b)
Z = length(z)

W1 = Array{eltype(state)}(undef, T, X)
# --- preallocate (T, X) work arrays once ---
W2 = similar(W1)
T1 = similar(W1)
T2 = similar(W1)
Q  = similar(W1)
Lh = similar(W1)
J1 = similar(W1)
J2 = similar(W1)

# --- outer products via BLAS (no extra allocs) ---
# treat vectors as (T×1)*(1×X)
mul!(W1, reshape(w1_coeff, T,1), reshape(b, 1,X))  # (T,X)
mul!(W2, reshape(w2_coeff, T,1), reshape(b, 1,X))
mul!(T1, reshape(T1_coeff, T,1), reshape(b, 1,X))
mul!(T2, reshape(T2_coeff, T,1), reshape(b, 1,X))
mul!(Q , reshape(q_coeff , T,1), reshape(b, 1,X))
mul!(Lh, reshape(L_coeff , T,1), reshape(b, 1,X))

# --- fuse arithmetic (no U temp) ---
@. J1 = 2Lh + 0.7Q
@. J2 = -0.7Q

# --- pre-scale vertical basis once ---
# ρ0 is a function of z only; fold constants into basis
G1s = (0.0033) .* (G1 ./ ρ0)   # length Z
G2s = (0.0033) .* (G2 ./ ρ0)

# --- allocate 3D outputs ---
w1_prof = Array{eltype(state)}(y, Z, T, X)
w2_prof = similar(w1_prof)
T1_prof = similar(w1_prof)
T2_prof = similar(w1_prof)
J1_prof = similar(w1_prof)
J2_prof = similar(w1_prof)

# --- “lift” (T,X) -> (Z,T,X) with explicit einsum (threaded) ---
@tullio w1_prof[z,t,x] = G1[z]  * W1[t,x]
@tullio w2_prof[z,t,x] = G2[z]  * W2[t,x]
@tullio T1_prof[z,t,x] = G1s[z] * T1[t,x]
@tullio T2_prof[z,t,x] = G2s[z] * T2[t,x]
@tullio J1_prof[z,t,x] = G1s[z] * J1[t,x]
@tullio J2_prof[z,t,x] = G2s[z] * J2[t,x]

# --- combine w ---
w = @. w1_prof + w2_prof
T = @. T1_prof + T2_prof
J = @. J1_prof + J2_prof

w = real.(permutedims(w, (2,3,1)))  # (T,X,Z)
T = real.(permutedims(T, (2,3,1)))  # (T,X,Z)
J = real.(permutedims(J, (2,3,1)))  # (T,X,Z)

println(size(w))

h5open(joinpath(FPATH_SIM, "reconstruction.h5"), "w") do f
    write(f, "w", w)
    write(f, "T", T)
    write(f, "J", J)
    write(f, "x", x[lft:rgt])
    write(f, "z", z)
    write(f, "t", t)
end 