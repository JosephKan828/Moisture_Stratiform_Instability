# This program is to simulate system based on the Kuang 2008
# using package
using Pkg;
Pkg.activate("/work/b11209013/external/JuliaENV/atmo/");

using Plots, Measures; gr();
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

const r0 :: Float64 = 1.0;
const rq :: Float64 = 0.7;

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

const A  :: Float64 = 1.0 - 2.0*f + (b2-b1)/F;
const B  :: Float64 = 1.0 + (b2+b1)/F - A*r0

# ==== 2. setup domain ==== #
## Space domain
Δx :: Float64 = 1e3; Lx :: Float64 = 4e7;
x  :: Array{Float64,1} = range(-Lx, Lx, step=Δx);
Nx :: Int64            = length(x);
k  :: Array{Float64,1} = rfftfreq(Nx, 1/Δx)*2*π;

# ==== 3. construct system ODE matrix ==== #
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

function stab_test(kn::Float64)

    mat :: Matrix{ComplexF64} = sys_matrix(kn);

    unit_vec :: Matrix{ComplexF64} = ones(ComplexF64, 6,1);
    sys_mat  :: Matrix{ComplexF64} = exp(mat);
    pred_vec :: Matrix{ComplexF64} = exp(mat)*unit_vec;

    amp_factor :: Matrix{ComplexF64} = (pred_vec'*pred_vec) * inv(unit_vec'*unit_vec)

    return real.(amp_factor[1,1])
end

λ_target :: Vector{Float64} = collect(2000:500:40000);

stab :: Vector{Float64} = Array{Float64}(undef, length(λ_target));

@inbounds for (i, λ) in enumerate(λ_target)
    kn :: Float64 = 2*π*4320/λ;

    stab[i] = stab_test(kn);
end

# ==== 4. plot the growth rate ==== #
default(
    size=(1600,900),
    bottom_margin=10mm,
    left_margin=10mm,
    top_margin=10mm,
    right_margin=10mm,
    titlefont=font(24,"times"),
    guidefont=font(18,"times"),
    tickfont=font(16,"times"),
)
plt = plot(
    40000.0./λ_target, stab,
    seriestype=:scatter, color="gray", label=false
    );
plot!(plt, collect(0:25), ones(length(0:25)),
    color="blue", ls=:dot, label=false, lw=3
    );
xlims!(plt, (0, 25));
xlabel!(plt, "Nondimensional Wavenumber");
ylabel!(plt, "Growth Rate")
title!(plt, "Growth Rate vs. Wavenumber");
savefig(plt, "/home/b11209013/2025_Research/MSI/Fig/Full_GrowthRate.png");
display(plt);

if !isinteractive()
    println("Press Enter ..."); readline();
end

