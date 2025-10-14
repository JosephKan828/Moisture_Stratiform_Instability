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
function default_params(exptype::String="conv_only")
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
            -0.042, -0.0087, -0.011, -0.069, 5.61, 3.36
        )

    elseif exptype == "conv_radiation_moist"
        # --- Convection + (moisture-only) Radiation ---
        return ModelParams(
            1.4, 0.0, 1.0, 2.0, 1.0, 0.5, 1.1, -1.0, 0.3, 1.0,
            1.0, 0.7, 4.0, 0.5, 1/12, 0.1,
            0.0, 0.0, 0.0, 0.0, 5.61, 3.36
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
function coeff_matrix(kn::Float64; param::ModelParams = default_params())
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
end

end # module
