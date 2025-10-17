# This program is to compute the growth rate and phase speed of dynamic system
# using package
using Pkg; Pkg.activate("/work/b11209013/external/JuliaENV/atmo");
using HDF5;
using LinearAlgebra, Statistics;

# setup universal parameter
const g  = 9.81;               # gravitational acceleration (m/s^2)
const cp = 1004.0;            # specific heat at constant pressure (J/kg/K)
const Rd = 287.0;             # gas constant for dry air (J/kg/K)
const Γ  = 6.5/1000.0;        # lapse rate (K/m)

# setup model parameter
struct ModelParams
    a1   :: Float64
    a2   :: Float64
    b1   :: Float64
    b2   :: Float64
    c1   :: Float64
    c2   :: Float64
    d1   :: Float64
    d2   :: Float64
    m1   :: Float64
    m2   :: Float64
    r0   :: Float64
    rq   :: Float64
    F    :: Float64
    f    :: Float64
    τL   :: Float64
    ϵ    :: Float64
    RT11 :: Float64
    RT12 :: Float64
    RT21 :: Float64
    RT22 :: Float64
    Rq1  :: Float64
    Rq2  :: Float64
end

param = ModelParams(
    1.4, 0.0, 1.0, 2.0, 1.0, 0.5, 1.1, -1.0, 0.3, 1.0,
    1.0, 0.7, 4.0, 0.5, 1/12, 0.1, -0.042, -0.0087, -0.011, -0.069, 5.61, 3.36 # radiation from temperature and moisture
    # 1.0, 0.7, 4.0, 0.5, 1/12, 0.1, 0.0, 0.0, 0.0, 0.0, -1.9, 4.73 # radiation from moisture only
);
A :: Float64 = 1.0 - 2.0*param.f + (param.b2 - param.b1)/param.F;
B :: Float64 = 1.0 + (param.b2 + param.b1)/param.F - A*param.r0;

# functions
## functions for coefficient matrix
function coeff_matrix(
    kn :: Float64;
    ϵ  = param.ϵ , c1 = param.c1, c2 = param.c2,
    rq = param.rq, r0 = param.r0, a1 = param.a1, a2 = param.a2,
    d1 = param.d1, d2 = param.d2, τL = param.τL, f  = param.f,
    A  =A        , B  = B, RT11=param.RT11, RT12=param.RT12,
    RT21=param.RT21, RT22=param.RT22, Rq1=param.Rq1, Rq2=param.Rq2
)
    α :: ComplexF64 = -d1*(-1.5*rq+RT11) - d2*(1.5*rq+RT12);
    β :: ComplexF64 = -d1*RT21 - d2*RT22;
    γ :: ComplexF64 = -d1*(rq+Rq1) - d2*(-rq+Rq2);
    δ :: ComplexF64 = -d1*(1+r0) - d2*(1-r0)

    mat :: Matrix{ComplexF64} = [
        -ϵ        0.0      (kn*c1)^2.0      0.0        0.0      0.0  ;
        0.0        -ϵ           0.0     (kn*c2)^2.0    0.0      0.0  ;
        -1.0      0.0      -1.5*rq+RT11     RT21      rq+Rq1   1+r0  ;
        0.0      -1.0       1.5*rq+RT12     RT22     -rq+Rq2   1-r0  ;
        a1         a2            α           β          γ        δ   ;
        f/B/τL (1-f)/B/τL -1.5*A*rq/B/τL    0.0     A*rq/B/τL -1.0/τL;
    ];

    return mat
end

function growth_rate(kn::Float64)
    mat = coeff_matrix(kn);
    eigvals = eigen(mat).values;
    σ = real.(eigvals);
    return σ
end

function phase_speed(kn::Float64)
    mat = coeff_matrix(kn);
    eigvals = eigen(mat).values;
    c = -imag.(eigvals)./kn * (4320000/86400);
    return c
end

# main code
λ :: Vector{Float64} = collect(540:540:43200);
k :: Vector{Float64} = @. 2.0*π*4320.0 / λ;

modal_growth :: Matrix{Float64} = hcat((@. growth_rate(k))...);
speed        :: Matrix{Float64} = hcat((@. phase_speed(k))...);


# save file
h5open("/home/b11209013/2025_Research/MSI/File/Full/diagnose_rad_both.h5", "w") do f
    write(f, "λ", λ);
    write(f, "k", k);
    write(f, "growth_rate", modal_growth);
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