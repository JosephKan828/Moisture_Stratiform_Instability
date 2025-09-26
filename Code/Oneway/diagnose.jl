# diagnose_oneway_combined.jl
# ------------------------------------------------------------
# Single-way wave solution diagnostics (modal + nonmodal)
# - Eigenvalue growth rates (all modes)
# - Phase speeds (all modes)
# - Max growth rate per k
# - Nonmodal amplification via matrix exponential
# ------------------------------------------------------------

using Pkg
Pkg.activate("/work/b11209013/external/JuliaENV/atmo")

using HDF5
using LinearAlgebra
using Statistics

# -----------------------------
# Parameters
# -----------------------------
struct ModelParams
    b1::Float64
    b2::Float64
    c1::Float64
    c2::Float64
    m1::Float64
    m2::Float64
    γ0::Float64
    γq::Float64
    F ::Float64
    f ::Float64
    τL::Float64
    ϵ ::Float64
end

const param = ModelParams(1.0, 2.0, 1.0, 0.5, 0.3, 1.0, 0.0, 0.7, 4.0, 0.5, 1/12, 0.1)

# -----------------------------
# Coefficient matrix
# -----------------------------
function coeff_matrix(
    kn::Float64;
    ϵ=param.ϵ, b1=param.b1, b2=param.b2,
    c1=param.c1, c2=param.c2, γq=param.γq,
    γ0=param.γ0, m1=param.m1, m2=param.m2,
    τL=param.τL, f=param.f, F=param.F,
)

    α :: ComplexF64 = -1.5*b2*γq + F*f*(im*kn*c1 + ϵ) - 1.5*F*(1 - f)*γq
    β :: ComplexF64 = F*(1 - f)*(im*kn*c2 + ϵ)
    γ :: ComplexF64 = (b2 + F*(1 - f)) * γq
    δ :: ComplexF64 = -F*f - F*(1 - f)*γ0 - (b1 + b2*γ0)

    relax :: ComplexF64 = τL * (b1 + b2*γ0)

    mat :: Matrix{ComplexF64} = [
        (-im*kn*c1 - ϵ)    0.0                0.0        1.0          ;
         1.5*γq           (-im*kn*c2 - ϵ)   -γq         γ0           ;
         1.5*m2*γq         0.0              -m2*γq      m1 + m2*γ0   ;
         α/relax           β/relax           γ/relax     δ/relax      ;
    ]
    return mat
end

# -----------------------------
# Diagnostics per k
# -----------------------------
# All modal growth rates (real parts of eigenvalues), length-4 vector
function growth_rate_all(kn::Float64)
    λ = eigen(coeff_matrix(kn)).values
    return real.(λ)
end

# Phase speed from imaginary parts; sign convention: c = -Im(λ)/k
# Convert to m/s via factor 4,320,000/86,400 = 50 (matches your script)
function phase_speed_all(kn::Float64)
    λ = eigen(coeff_matrix(kn)).values
    return -imag.(λ) ./ kn .* (4320000/86400)  # = *50.0
end

# Max modal growth (scalar) — renamed to avoid clashing with Base.max
function max_growth_at_k(kn::Float64)
    λ = eigen(coeff_matrix(kn)).values
    return maximum(real.(λ))
end

# Simple nonmodal amplification via matrix exponential acting on an all-ones vector
# Returns a scalar amplification factor (squared-norm ratio)
function nonmodal_amp(kn::Float64)
    M = coeff_matrix(kn)
    u = ones(ComplexF64, 4, 1)
    U = exp(M)                    # matrix exponential
    v = U * u
    amp = (v' * v) * inv(u' * u)  # ||v||^2 / ||u||^2
    return real(amp[1, 1])
end

# -----------------------------
# Main
# -----------------------------
function main(; outpath::AbstractString = "/home/b11209013/2025_Research/MSI/File/diagnose_oneway.h5")
    # Grid
    λ = collect(540.0:540.0:43200.0)         # wavelength [km]
    k = @. 2.0 * π * 4320.0 / λ              # nondimensional wavenumber

    # Modal spectra (4×N)
    growth_rate = hcat((@. growth_rate_all(k))...)   # 4 × N
    phase_speed = hcat((@. phase_speed_all(k))...)   # 4 × N

    # Max growth (1×N as Vector)
    max_growth = max_growth_at_k.(k)                 # N

    # Nonmodal amplification (1×N as Vector)
    nonmodal = nonmodal_amp.(k)                      # N

    # Save
    h5open(outpath, "w") do f
        write(f, "λ", λ)
        write(f, "k", k)
        write(f, "growth_rate", growth_rate)
        write(f, "phase_speed", phase_speed)
        write(f, "max_growth", max_growth)
        write(f, "nonmodal", nonmodal)

        attributes(f["λ"])["standard_name"] = "wavelength"
        attributes(f["λ"])["units"] = "km"

        attributes(f["k"])["standard_name"] = "non-dimensional wavenumber"
        attributes(f["k"])["units"] = "1"  # dimensionless (was "km/km")

        attributes(f["growth_rate"])["standard_name"] = "modal growth rate (all modes)"
        attributes(f["growth_rate"])["units"] = "1/day"

        attributes(f["phase_speed"])["standard_name"] = "phase speed (all modes)"
        attributes(f["phase_speed"])["units"] = "m/s"

        attributes(f["max_growth"])["standard_name"] = "maximum modal growth rate"
        attributes(f["max_growth"])["units"] = "1/day"

        attributes(f["nonmodal"])["standard_name"] = "nonmodal amplification (||e^M u||^2 / ||u||^2)"
        attributes(f["nonmodal"])["units"] = "1"
    end

    println("✅ Wrote diagnostics to: ", outpath)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

