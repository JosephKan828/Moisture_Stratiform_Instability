using Pkg; Pkg.activate("/work/b11209013/external/JuliaENV/atmo");

using HDF5;
using Plots; gr();
using LinearAlgebra, Statistics;
using LazyGrids;

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
    Γ ::Float64 
    Rd::Float64
    Cp::Float64 
    g ::Float64    
end

param = ModelParams(1.4, 0.0, 1.0, 2.0, 1.0, 0.5, 1.1, -1.0, 0.3, 1.0, 1.0, 0.7, 4.0, 0.5, 1/12, 0.1, 6.5/1000.0, 287.5, 1004.5, 9.81)
A     = 1.0 - 2.0*param.f + (param.b2-param.b1)/param.F;
B     = 1.0 + (param.b2+param.b1)/param.F - A*param.r0;

function coeff_matrix(
        kn::Float64;
        ϵ=param.ϵ, c1=param.c1, c2=param.c2,
        rq=param.rq, r0=param.r0, a1=param.a1,
        a2=param.a2, d1=param.d1, d2=param.d2,
        τL=param.τL, f=param.f, A=A, B=B
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

function modal(kn::Float64)
    mat :: Array{ComplexF64} = coeff_matrix(kn);

    growth_rate :: Array{Float64} = real.(eigvals(mat))

    return maximum(growth_rate)
end

function nonmodal(kn::Float64)
    mat :: Matrix{ComplexF64} = coeff_matrix(kn);

    unit_vec :: Matrix{ComplexF64} = ones(ComplexF64, 6,1);
    sys_mat  :: Matrix{ComplexF64} = exp(mat);
    pred_vec :: Matrix{ComplexF64} = exp(mat)*unit_vec;

    amp_factor :: Matrix{ComplexF64} = (pred_vec'*pred_vec) * inv(unit_vec'*unit_vec)

    return real.(amp_factor[1,1])
end

λ :: Array{Float64} = collect(540:540:43200)
k :: Array{Float64} = @. 2*π*4320.0 / λ;

modal_stab    :: Array{Float64} = modal.(k);
nonmodal_stab :: Array{Float64} = nonmodal.(k);

h5open("/home/b11209013/2025_Research/MSI/File/stability.h5", "w") do f
    write(f, "nonmodal", nonmodal_stab)
    write(f, "modal", modal_stab)
    write(f, "kn", k)

    attributes(f["nonmodal"])["standard_name"] = "nonmodal growth rate"
    attributes(f["modal"])["standard_name"] = "modal growth rate"
    attributes(f["modal"])["units"] = "1/day"
    attributes(f["kn"])["standard_name"] = "angular wavenumber"
    attributes(f["kn"])["units"] = "None"
end


