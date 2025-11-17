module LinearModel

using LinearAlgebra
export ModelParams, default_params, coeff_matrix

# ============================================
# 1. Model parameter structure
# ============================================
"""
    ModelParams

Holds physical and empirical parameters for the linear dynamical model.
All fields are `Float64` for numerical consistency.
"""
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


# ============================================
# 2. Default parameter sets
# ============================================
"""
    default_params(exptype::String="conv_only") -> ModelParams

Returns a parameter set according to the experiment type.

# Arguments
- `exptype`: one of  
    `"conv_only"` — Convection only  
    `"conv_radiation_full"` — Convection + (moisture + temperature) radiation  
    `"conv_radiation_moist"` — Convection + (moisture-only) radiation
"""
function default_params(exptype::String="conv_only", rad_scaling::Float64=0.00)

    if exptype == "conv_only"
        # --- Convection only ---
        return ModelParams(
            1.4, 0.0, 1.0, 2.0, 1.0, 0.5, 1.1, -1.0, 0.3, 1.0,
            1.0, 0.7, 4.0, 0.5, 1/12, 0.1,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )

    elseif exptype == "conv_radiation_full"
        # --- Convection + (moisture + temperature) Radiation ---
        return ModelParams(
            1.4, 0.0, 1.0, 2.0, 1.0, 0.5, 1.1, -1.0, 0.3, 1.0,
            1.0, 0.7, 4.0, 0.5, 1/12, 0.1,
            -0.042*rad_scaling, -0.0087*rad_scaling, -0.011*rad_scaling,
            -0.069*rad_scaling, 5.61*rad_scaling, 3.36*rad_scaling
        )

    elseif exptype == "conv_radiation_moist"
        # --- Convection + (moisture-only) Radiation ---
        return ModelParams(
            1.4, 0.0, 1.0, 2.0, 1.0, 0.5, 1.1, -1.0, 0.3, 1.0,
            1.0, 0.7, 4.0, 0.5, 1/12, 0.1,
            0.0, 0.0, 0.0, 0.0, 5.61*rad_scaling, 3.36*rad_scaling
        )

    else
        error("Invalid experiment type: '$exptype'. Choose from 'conv_only', 'conv_radiation_full', or 'conv_radiation_moist'.")
    end
end


# ============================================
# 3. Coefficient matrix generator
# ============================================
"""
    coeff_matrix(kn; param=default_params())

Compute the 6×6 complex coefficient matrix for a given wavenumber `kn`.

# Keyword arguments
- `param`: `ModelParams` struct (default = `default_params()`)

# Returns
`Matrix{ComplexF64}` of size (6,6)
"""
function coeff_matrix(kn::Float64; param::ModelParams = default_params(), mode="full")
    if mode == "full"

        # Precompute auxiliary constants
        A = 1.0 - 2.0*param.f + (param.b2 - param.b1)/param.F
        B = 1.0 + (param.b2 + param.b1)/param.F - A*param.r0

        α = -param.d1*(-1.5*param.rq + param.RT11) - param.d2*(1.5*param.rq + param.RT12)
        β = -param.d1*param.RT21 - param.d2*param.RT22
        γ = -param.d1*(param.rq + param.Rq1) - param.d2*(-param.rq + param.Rq2)
        δ = -param.d1*(1 + param.r0) - param.d2*(1 - param.r0)

        mat = ComplexF64[
            -param.ϵ               0.0                (kn*param.c1)^2          0.0                 0.0                  0.0;
            0.0                  -param.ϵ            0.0               (kn*param.c2)^2          0.0                  0.0;
            -1.0                   0.0               -1.5*param.rq+param.RT11  param.RT21           param.rq+param.Rq1   1+param.r0;
            0.0                  -1.0                1.5*param.rq+param.RT12  param.RT22          -param.rq+param.Rq2   1-param.r0;
            param.a1              param.a2           α                        β                    γ                    δ;
            param.f/B/param.τL   (1-param.f)/B/param.τL  -1.5*A*param.rq/B/param.τL  0.0   A*param.rq/B/param.τL   -1/param.τL
        ]

        return mat

    elseif mode == "oneway"
    γ0 = (1 - param.r0) / (1 + param.r0)
    γq = (2 * param.rq) / (1 + param.r0)

    # denominator D = b1 + b2 * γ0
    D = param.b1 + param.b2 * γ0

    # --- helper coefficients from d/dt (f T1 + (1-f) T2) ---
    # α0 = 1.5 γq (1-f) - f (ϵ + i k c1)
    α0 = param.f * (-1im * kn * param.c1 - param.ϵ) +
         1.5 * γq * (1 - param.f)

    # β0 = -(1-f) (ϵ + i k c2)
    β0 = (1 - param.f) * (-1im * kn * param.c2 - param.ϵ)

    # δ0 = -(1-f) γq
    δ0 = -(1 - param.f) * γq

    # --- coefficients in J1_eq = coeff_T1*T1 + coeff_T2*T2 + coeff_q*q + coeff_J1*J1 ---
    coeff_T1 = (-1.5 * param.b2 * γq - param.F * α0) / D
    coeff_T2 = (-param.F * β0) / D
    coeff_q  = ( param.b2 * γq - param.F * δ0) / D
    coeff_J1 = (-param.F * param.f) / D

    # --- bottom row of the matrix: dJ1/dt = α*T1 + β*T2 + γ*q + δ*J1 ---
    α = coeff_T1 / param.τL
    β = coeff_T2 / param.τL
    γ = coeff_q  / param.τL
    δ = (coeff_J1 - 1) / param.τL

    mat = ComplexF64[
        -1im * kn * param.c1 - param.ϵ      0.0                           0.0           1.0;
         1.5 * γq                           -1im * kn * param.c2 - param.ϵ   -γq         0.0;
         1.5 * param.m2 * γq                0.0                           -param.m2*γq  param.m1;
         α                                  β                             γ             δ
    ]

    return mat
end

end

end # module
