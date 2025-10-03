using Pkg; Pkg.activate("/work/b11209013/external/JuliaENV/atmo");

using HDF5;
using FFTW, LinearAlgebra, Statistics;
using LazyGrids;
using Dates;
using Base.Threads;

BLAS.set_num_threads(12);

struct ModelParams
    a1::Float64 
    a2::Float64
    b1::Float64 
    b2::Float64    
    c1::Float64 
    c2::Float64
    d1::Float64
    d2::Float64
    m1::Float64 
    m2::Float64
    r0::Float64 
    rq::Float64
    F ::Float64 
    f ::Float64
    τL::Float64 
    ϵ ::Float64    
end

param = ModelParams(1.4, 0.0, 1.0, 2.0, 1.0, 0.5, 1.1, -1.0, 0.3, 1.0, 1.0, 0.7, 4.0, 0.5, 1/12, 0.1)
A     = 1.0 - 2.0*param.f + (param.b2-param.b1)/param.F;
B     = 1.0 + (param.b2+param.b1)/param.F - A*param.r0;

const Γ ::Float64 = 6.5/1000.0; 
const Rd::Float64 = 287.5;
const Cp::Float64 = 1004.5; 
const g ::Float64 = 9.81;

# coefficient matrix for dynamical system
function coeff_matrix(
        kn::Float64;
        ϵ=param.ϵ  , c1=param.c1, c2=param.c2,
        rq=param.rq, r0=param.r0, a1=param.a1,
        a2=param.a2, d1=param.d1, d2=param.d2,
        τL=param.τL, f=param.f  , A=A        , B=B
)
    α ::ComplexF64 = 1.5*rq*(d1-d2);
    β ::ComplexF64 = -rq*(d1-d2);
    γ ::ComplexF64 = -(d1*(1+r0)+d2*(1-r0));
    
    mat :: Array{ComplexF64}= [
        -ϵ 0.0 (kn*c1)^2.0 0.0 0.0 0.0;
        0.0 -ϵ 0.0 (kn*c2)^2.0 0.0 0.0;
        -1.0 0.0 -1.5*rq 0.0 rq 1+r0;
        0.0 -1.0 1.5*rq 0.0 -rq 1-r0;
        a1 a2 α 0.0 β γ;
        f/B/τL (1-f)/B/τL -1.5*A*rq/B/τL 0.0 A*rq/B/τL -1.0/τL;
    ]

    return mat
end

# integration function
function integration(state, t, k, init, coeff_mat)
    Nt,Nv,Nk = size(state); # acquire number of time, variable, and wavenumber
    Δt       = t[2]-t[1]
    
    @threads for j in eachindex(k)
        A = coeff_matrix(k[j]) # coefficient matrix for each wavenumber
        Φ = exp(Δt*A) # matrix exponential for the coefficient matrix with Δt
        
        state[1,:,j] .= init[:,j]; # set initial condition as the first time step
        for i in 2:Nt
            @views state[i,:,j] = Φ*state[i-1,:,j]
        end
    end
    return state
end

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

# preallocate state vector
state_vec ::Array{ComplexF64} = zeros(Nt,6,length(k));

state_vec = integration(state_vec, t, k, init, coeff_matrix
);

fpath::String = "/home/b11209013/2025_Research/MSI/File/";

# save state vector
h5open(fpath*"Full/state.h5","w") do f
    write(f, "state vector", state_vec)
    write(f, "time", t)
    write(f, "wavenumber", k)
    write(f, "variables", ["w1" "w2" "T1" "T2" "q" "L"])

    attributes(f["time"])["units"] = "day"
    attributes(f["wavenumber"])["units"] = "None"
end

# save inverse matrix
h5open(fpath*"inv_mat.h5","w") do f
    write(f, "inverse matrix", inv_mat)
    write(f, "wavenumber", k)
    write(f, "x", x)

    attributes(f["x"])["units"] = "m"
    attributes(f["wavenumber"])["units"] = "None"
end

# save vertical mode
h5open(fpath*"vertical_mode.h5","w") do f
    write(f, "G1", G1)
    write(f, "G2", G2)
    write(f, "z", z)

    attributes(f["z"])["units"] = "m"
    attributes(f["z"])["standard_name"] = "z-axis"

    attributes(f["G1"])["standard_name"] = "1st vertical mode"
    attributes(f["G2"])["standard_name"] = "2nd vertical mode"
end

# save background field
h5open(fpath*"bg_field.h5","w") do f
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
