#####################
# Full model simulation
# Activate environment and using packages
#####################
using Pkg; Pkg.activate("/work/b11209013/external/JuliaENV/atmo");

using HDF5;
using FFTW, LinearAlgebra, Statistics;
using LazyGrids;
using Dates;
using Base.Threads;

BLAS.set_num_threads(12);

#####################
# Model parameters and functions
#####################
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

#####################
# Import necessary data
#####################
FPATH_SIM ::String = "/home/b11209013/2025_Research/MSI/File/Sim_stuff/";

# background field
ρ0, p0, T0, z_bg = h5open(FPATH_SIM * "background.h5", "r") do f
    read(f, "ρ0"), read(f, "p0"), read(f, "T0"), read(f, "z")
end

# vertical mode
G1, G2 = h5open(FPATH_SIM * "vertical_mode.h5", "r") do f
    read(f, "G1"), read(f, "G2")
end

# domain setting
x, z, t = h5open(FPATH_SIM * "domain.h5", "r") do f
    read(f, "x"), read(f, "z"), read(f, "t")
end

# inverse matrix for projection
k = h5open(FPATH_SIM * "inv_mat.h5", "r") do f
    read(f, "wavenumber")
end

#####################
# Horizontal spatial setting
#####################
Nt :: Int64 = length(t);          # number of time steps


# Initial state vector
init :: Matrix{ComplexF64} = randn(6, length(k))*0.1;
## initial_state_vec: row1: w1; row2: w2; row3: T1; row4: T2; row5: q; row6: L;

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
