using Pkg; Pkg.activate("/work/b11209013/external/JuliaENV/atmo");

using HDF5;
using FFTW, LinearAlgebra, Statistics;
using LazyGrids;
using Dates;
using Base.Threads;

BLAS.set_num_threads(12);

#####################
# setting 
#####################

const Γ ::Float64 = 6.5/1000.0; 
const Rd::Float64 = 287.5;
const Cp::Float64 = 1004.5; 
const g ::Float64 = 9.81;

# Horizontal spatial setting
Δx       ::Float64         = 1e3;                # Horizontal space interval: 1 km
Lx       ::Float64         = 4e7;                # left/right boundary of horizontal space.
x        ::Vector{Float64} = collect(-Lx:Δx:Lx); # x-axis
Nx       ::Int64           = length(x);          # length of x-dimension
x_scaling::Float64         = 4_320_000.0;        # scaling factor for x

lft_idx  :: Int64 = argmin(abs.(x.+x_scaling));   # left boundary index for our interest
rgt_idx  :: Int64 = argmin(abs.(x.+x_scaling));   # right boundary index for our interest

# vertical spatial setting
Δz       ::Float64         = 1e2;                # vertical space interval
Lz       ::Float64         = 14e3;               # Depth of troposphere
z        ::Vector{Float64} = collect(0.0:Δz:Lz); # z-axis
Nz       ::Int64           = length(z);          # Number of vertical coordinate

# Temporal setting
Δt ::Float64         = 1e-1;              # Time interval: 0.1 day
Lt ::Float64         = 1e2;               # simulation duration: 100 days
t  ::Vector{Float64} = collect(0.0:Δt:Lt) # time axis
Nt ::Int64           = length(t);         # length of time 

# Vertical modes
G1 :: Matrix{Float64} = reshape(π/2*sin.(π*z/Lz), Nz, 1);  # first vertical mode
G2 :: Matrix{Float64} = reshape(π/2*sin.(2π*z/Lz), Nz, 1); # second vertical mode

# temperature profile (units: K)
T0 :: Vector{Float64} = @. 300.0 - Γ*z;

# pressure profile (units: Pa)
p0 :: Vector{Float64} = @. 101325*(1.0 - Γ/300*z)^(g/Rd/Γ);

# density profile (units: kg/m^3)
ρ0 :: Vector{Float64} = @. p0 / Rd / T0;

λ ::Vector{Float64} = collect(540:540:40000); # λ from 540 km ~ 40000 km
k ::Vector{Float64} = 2π*4320.0./λ;           # wavenumber is in angular and non-dimensional
Nk::Int64           = length(k);
# Initial state vector
init :: Matrix{ComplexF64} = randn(6, length(k))*0.1;
## initial_state_vec: row1: w1; row2: w2; row3: T1; row4: T2; row5: q; row6: L;

inv_mat ::Matrix{ComplexF64} = zeros(length(k), Nx);

@inbounds for (i, kj) in enumerate(k)
    inv_mat[i,:] = @. exp(im*kj*x/x_scaling)
end ;

#####################
# Save files 
#####################

fpath::String = "/home/b11209013/2025_Research/MSI/File/";

# domain setting
h5open(fpath*"Sim_stuff/domain.h5","w") do f
    write(f, "x", x)
    write(f, "z", z)
    write(f, "t", t)

    attributes(f["x"])["units"] = "m"
    attributes(f["x"])["standard_name"] = "x-axis"

    attributes(f["z"])["units"] = "m"
    attributes(f["z"])["standard_name"] = "z-axis"

    attributes(f["t"])["units"] = "day"
    attributes(f["t"])["standard_name"] = "time"
end

# save inverse matrix
h5open(fpath*"Sim_stuff/inv_mat.h5","w") do f
    write(f, "inverse matrix", inv_mat)
    write(f, "wavenumber", k)
    write(f, "x", x)
    attributes(f["x"])["units"] = "m"
    attributes(f["wavenumber"])["units"] = "None"
end

# save vertical mode
h5open(fpath*"Sim_stuff/vertical_mode.h5","w") do f
    write(f, "G1", G1)
    write(f, "G2", G2)
    write(f, "z", z)

    attributes(f["z"])["units"] = "m"
    attributes(f["z"])["standard_name"] = "z-axis"

    attributes(f["G1"])["standard_name"] = "1st vertical mode"
    attributes(f["G2"])["standard_name"] = "2nd vertical mode"
end

# save background field
h5open(fpath*"Sim_stuff/background.h5","w") do f
    write(f, "ρ0", ρ0)
    write(f, "p0", p0)
    write(f, "T0", T0)
    write(f, "z", z)

    attributes(f["ρ0"])["units"] = "kg m^-3"
    attributes(f["ρ0"])["standard_name"] = "density"

    attributes(f["p0"])["units"] = "Pa"
    attributes(f["p0"])["standard_name"] = "pressure"

    attributes(f["T0"])["units"] = "K"
    attributes(f["T0"])["standard_name"] = "temperature"
end
