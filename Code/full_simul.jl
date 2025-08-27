using Pkg;
Pkg.activate("/work/b11209013/external/JuliaENV/atmo/");

using FFTW, LinearAlgebra, Statistics;
using LazyGrids;
using HDF5;
using Dates;
using Base.Threads;

BLAS.set_num_threads(12);

function sys_matrix(
    k :: Float64;
    ϵ=ϵ, c1=c1, c2=c2,
    a1=a1, a2=a2, d1=d1, d2=d2, rq=rq, r0=r0,
    A=A, B=B, f=f, τL=τL
    )

    α :: ComplexF64 = 1.5*rq*(d1+d2);
    β :: ComplexF64 = -rq*(d1-d2);
    γ :: ComplexF64 = -1*((d1-d2)*r0 + (d1+d2));

    mat :: Matrix{ComplexF64} = [
    -ϵ 0.0 (c1*k)^2 0.0 0.0 0.0;
    0.0 -ϵ 0.0 (c2*k)^2 0.0 0.0;
    -1.0 0.0 -1.5*rq 0.0 rq 1+r0;
    0.0 -1.0 1.5*rq 0.0 -rq 1-r0;
    a1 a2 α 0.0 β γ;
    f/B/τL (1-f)/B/τL -1.5*A*rq/B/τL 0.0 A*rq/B/τL -1.0/τL
                                ];
    return mat
end

function propagate_uniform_step!(state_vec, t, kn, initial_state_vec, sys_matrix)
    Nt = length(t); Nk = length(kn)
    n  = size(initial_state_vec, 1)
    @assert size(state_vec) == (Nt, n, Nk)
    @assert Nt ≥ 1
    @assert Nt == 1 || all(isapprox.(diff(t), diff(t)[1])) "t must be uniform"

    Δt = Nt == 1 ? zero(eltype(t)) : t[2] - t[1]
    t0 = t[1]

    @threads for j in eachindex(kn)
        A  = sys_matrix(kn[j])::Matrix{ComplexF64}
        Φ  = exp(Δt * A)                      # one matrix exp per k

        # start state at t0 (supports nonzero t0) via eigen scaling; 1 solve + 1 GEMV
        F  = eigen(A); V, λ = F.vectors, F.values
        y0 = V \ @view(initial_state_vec[:, j])
        y0 .= exp.(t0 .* λ) .* y0
        x  = V * y0

        @views state_vec[1, :, j] = x
        for i in 2:Nt
            x = Φ * x                         # gemv each step
            @views state_vec[i, :, j] = x
        end
    end
    return state_vec
end

const b1 :: Float64 = 1.0; const b2 :: Float64 = 2.0;

const a1 :: Float64 = 1.4; const a2 :: Float64 = 0.0;

const d1 :: Float64 = 1.1; const d2 :: Float64 = -1.0;

const m1 :: Float64 = 0.3; const m2 :: Float64 = 1.0;

const r0 :: Float64 = 1.0; const rq :: Float64 = 0.7;

const F  :: Float64 = 4.0; const f  :: Float64 = 0.5;

const c1 :: Float64 = 1.0; const c2 :: Float64 = 0.5;

const τL :: Float64 = 1/12; const ϵ  :: Float64 = 0.1;

const Γ  :: Float64 = 6.5/1000.0;

const Rd :: Float64 = 287.5; const Cp :: Float64 = 1004.5;
const g  :: Float64 = 9.81;

const A  :: Float64 = 1.0 - 2.0*f + (b2-b1)/F; const B  :: Float64 = 1.0 + (b2+b1)/F - A*r0;

# spatial domain
## Horizontal
Δx       :: Float64         = 1e3; # spatial interval: 1 km;
x_scale  :: Float64         = 4_320_000.0;
Lx       :: Float64         = 4e7; # Maximum length: 40000 km;
x        :: Vector{Float64} = collect(-Lx:Δx:Lx);
x_nondim :: Vector{Float64} = x ./ x_scale;
Nx       :: Int64           = length(x); # length of x-domain (80001,);

lft_bnd :: Int64 = argmin(abs.(x.+x_scale));
rgt_bnd :: Int64 = argmin(abs.(x.-x_scale));

## Vertical
Δz :: Float64         = 1e2;  # Vertical spacing: 100 m;
Lz :: Float64         = 15e3; # Depth of troposphere;
z  :: Vector{Float64} = collect(0.0:Δz:Lz);
Nz :: Int64           = length(z);

## Vertical modes
# G_j(z) = π/2 sin(jπz/H_T), where j=1,2
G1 :: Matrix{Float64} = reshape(π/2*sin.(π*z/Lz), Nz, 1); # first vertical mode
G2 :: Matrix{Float64} = reshape(π/2*sin.(2π*z/Lz), Nz, 1);

# temporal domain
Δt :: Float64         = 1e-1; # Time interval: 0.1 day;
Lt :: Float64         = 1e2; # Final time stamp: 100 days;
t  :: Vector{Float64} = collect(0.0:Δt:Lt);
Nt :: Int64           = length(t);

# wavenumber setting
λ  :: Vector{Float64} = collect(540:540:40000);
kn :: Vector{Float64} = 2*π*4320.0./λ;

# Initial field in frequency domain
initial_state_vec :: Matrix{ComplexF64} = randn(6, length(kn))*0.1;
## initial_state_vec: row1: w1; row2: w2; row3: T1; row4: T2; row5: q; row6: L;

# inverse matrix
exp_mat :: Matrix{ComplexF64} = zeros(length(kn), Nx);

for (i, kj) in enumerate(kn)
    exp_mat[i,:] = @. exp(im*kj*x_nondim)
end ;

# setup background profile
T0 :: Vector{Float64} = @. 300.0 - Γ*z;
p0 :: Vector{Float64} = @. 101325*(1.0 - Γ/300*z)^(g/Rd/Γ);
ρ0 :: Vector{Float64} = @. p0 / Rd / T0

# Compute in frequency domain
state_vec :: Array{ComplexF64,3} = zeros(Nt, 6, length(kn));

state_vec = propagate_uniform_step!(state_vec, t, kn, initial_state_vec, sys_matrix);

ts = Dates.format(now(), "yyyymmdd_HHMMSS");


h5open("/home/b11209013/2025_Research/MSI/File/vertical_mode.h5", "w") do h5

    write(h5, "z", z)
    write(h5, "G1", G1)
    write(h5, "G2", G2)

end

h5open("/home/b11209013/2025_Research/MSI/File/state_vector.h5", "w") do h5
    write(h5, "state vector", state_vec)
    write(h5, "time", t)
    write(h5, "wavenumber", kn)
    write(h5, "x", x);
end

h5open("/home/b11209013/2025_Research/MSI/File/background.h5", "w") do h5
    write(h5, "T", T0)
    write(h5, "p", p0)
    write(h5, "rho", ρ0)
end

h5open("/home/b11209013/2025_Research/MSI/File/inverse_mat.h5", "w") do h5
    write(h5, "inverse matrix", exp_mat)
end

# # choose specific Wavenumber
# final_choose :: Array{ComplexF64,3} = zeros(size(state_vec));
# final_choose[:,:,idx_8640] = state_vec[:,:,idx_8640];
# 
# # reconstruct
# recon :: Array{Float64,3} = zeros(Nt,6,Nx);
# 
# @inbounds for i in 1:Nt
#     recon[i,:,:] = real.(final_choose[i,:,:]*exp_mat);
# end
# 
# temp_prof :: Array{Float64,3} = zeros(Nt, Nz, length(x[lft_bnd:rgt_bnd]));
# 
# @inbounds for i in 1:Nt
#     tot_prof = G1*reshape(recon[i,3,:],1,Nx) .+ G2*reshape(recon[i,4,:],1,Nx)
# 
#     temp_prof[i,:,:] = (tot_prof[:,lft_bnd:rgt_bnd]*(-Γ+g/Cp))./ρ0
# end
# 
# open("/home/b11209013/2025_Research/MSI/File/full_temp.json", "w") do f
#     JSON.print(f,temp_prof)
# end
