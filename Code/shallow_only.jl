# This program is to simulate system based on the Kuang 2008
# using package
using Pkg;
Pkg.activate("/work/b11209013/external/JuliaENV/atmo/");

using Plots; gr();
using FFTW, LinearAlgebra, Statistics;
using LazyGrids;

# ==== 1. set parameter ==== #
const b1 :: Float64 = 1.0;
const b2 :: Float64 = 2.0;

const a1 :: Float64 = 1.4;
const a2 :: Float64 = 0.0;

const d1 :: Float64 = 1.1;
const d2 :: Float64 = -1.0;

const m1 :: Float64 = 0.3;
const m2 :: Float64 = 1.0;

const γ0 :: Float64 = 0.0;
const γq :: Float64 = 0.7;

const F  :: Float64 = 4.0;
const f  :: Float64 = 0.5;

const τL :: Float64 = 1/12;
const c1 :: Float64 = 1.0;
const c2 :: Float64 = 0.5;

const ϵ  :: Float64 = 0.1;

const Γ  :: Float64 = 6.5/1000.0;

const Rd :: Float64 = 287.5;
const Cp :: Float64 = 1004.5;
const g  :: Float64 = 9.81;

# ==== 2. setup domain ==== #
## Space domain
Δx :: Float64 = 3e3; Lx :: Float64 = 4e7;
x  :: Array{Float64,1} = range(-Lx, Lx, step=Δx);
Nx :: Int64            = length(x);
k  :: Array{Float64,1} = rfftfreq(Nx);

Δz :: Float64 = 1e2; Lz :: Float64 = 1.5e4;
z  :: Array{Float64,1} = range(0, Lz, step=Δz);

zz, xx = ndgrid(z, x);

## Time domain
Δt :: Float64 = 1e-2; Lt :: Float64 = 1e3;
t  :: Array{Float64,1} = range(0.0, Lt, step=Δt);

## background temperature and density
T0 :: Array{Float64,2} = @. 300.0 - Γ*zz;
ρ0 :: Array{Float64,2} = @. (101325/Rd/T0)*(1.0-Γ*zz/300)^(g/Rd/Γ);

# ==== 3. Initial condition ==== #
G1 :: Array{Float64,1} = π/2*sin.(π*z/Lz);
G2 :: Array{Float64,1} = π/2*sin.(2*π*z/Lz);

T1 :: Array{Float64,1} = @. 1.0*sin(20*π*x/Lx);
T2 :: Array{Float64,1} = @. 1.0*cos(20*π*x/Lx);

q  :: Array{Float64,1} = @. 1.0*cos(40*π*x/Lx);

T̂1 :: Array{ComplexF64,1} = rfft(T1);
T̂2 :: Array{ComplexF64,1} = rfft(T2);
q̂  :: Array{ComplexF64,1} = rfft(q);

@show length(k);
@show length(T̂1);

initial :: Array{ComplexF64,2} = vcat([T̂1 T̂2 q̂])';

# ==== 4. integration ==== #
# preallocate operating matrix
function sys_matrix(
    k :: Float64;
    c1=c1, c2=c2, F=F, b1=b1, γq=γq,
    m1=m1, m2=m2
    )

    α :: Float64 = F/b1;

    A :: Array{ComplexF64,2} = [
    -im*k*c1-ϵ    im*k*c2*α        α*γq    ;
       0.0       -im*k*c2-ϵ         -γq    ;  
       0.0      im*k*m1*c2*α    (m1*α-m2)γq]

    return A

end

Final :: Array{ComplexF64,2} = similar(initial);

@inbounds for (j, kj) in enumerate(k)
    Final[:,j] .= exp(sys_matrix(kj)*10)*initial[:,j] 
end

idx_8640 :: Int64 = argmin(abs.(k .- ));

@show real.(eigvals(sys_matrix(Lx / 8640000)));

T̂1_Final :: Vector{ComplexF64} = zeros(length(T̂1));
T̂2_Final :: Vector{ComplexF64} = zeros(length(T̂2));
q̂_Final :: Vector{ComplexF64} = zeros(length(q̂));

T̂1_Final[idx_8640] = Final[1,idx_8640];
T̂2_Final[idx_8640] = Final[2,idx_8640];
q̂_Final[idx_8640]  = Final[3,idx_8640];

T1_final :: Vector{Float64} = real(irfft(T̂1_Final, length(T1)));
T2_final :: Vector{Float64} = real(irfft(T̂2_Final, length(T2)));
q_final  :: Vector{Float64} = real(irfft(q̂_Final, length(q)));

plt=plot(x,T1_final, xlim=(-5000000,5000000))
plot!(plt, x, T1)
display(plt)


if !isinteractive()
    println("Press Enter ..."); readline()
end
