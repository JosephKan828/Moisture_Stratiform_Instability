module Galerkin_Fourier

using LinearAlgebra, Tullio
export Fourier_Reconstruct, Galerkin_Reconstruct

# ============================================
# Reconstruct based on Fourier basis
# ============================================

function Fourier_Reconstruct(
    state   :: Vector{ComplexF64},
    inv_mat :: Vector{ComplexF64},
)      
    T = size(state,1)
    X = size(inv_mat,1)

    Fourier_coeff = Array{eltype(state)}(undef, T, X)

    mul!(Fourier_coeff, reshape(state, T,1), reshape(inv_mat, 1,X))  # (T,X)

    return Fourier_coeff
end


# ============================================
# Reconstruct based on Galerkin basis
# ============================================
function Galerkin_Reconstruct(
    Fourier_coeff1   :: Matrix{ComplexF64},
    Fourier_coeff2   :: Matrix{ComplexF64},
    Γ               :: Float64,
    ρ0              :: Vector{Float64},
    G1              :: Matrix{Float64},
    G2              :: Matrix{Float64},
)
    G1s = @. (Γ + 9.81/1004.5) * G1 / ρ0
    G2s = @. (Γ + 9.81/1004.5) * G2 / ρ0

    prof1 = Array{eltype(Fourier_coeff1)}(undef, length(G1), size(Fourier_coeff1,1), size(Fourier_coeff1,2))
    prof2 = similar(prof1)

    @tullio prof1[z,t,x] = G1s[z]  * Fourier_coeff1[t,x]
    @tullio prof2[z,t,x] = G2s[z]  * Fourier_coeff2[t,x]

    return prof1, prof2
end

end