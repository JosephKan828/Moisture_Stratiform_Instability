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

include("/home/b11209013/2025_Research/MSI/src/LinearModel.jl")
using .LinearModel

BLAS.set_num_threads(12);

params = default_params("conv_only");

# integration function
# function integration(state, t, k, init, coeff_mat)
#     Nt,Nv,Nk = size(state); # acquire number of time, variable, and wavenumber
#     Δt       = t[2]-t[1]
    
#     @threads for j in eachindex(k)
#         A = coeff_matrix(k[j]) # coefficient matrix for each wavenumber
#         Φ = exp(Δt*A) # matrix exponential for the coefficient matrix with Δt
        
#         state[1,:,j] .= init[:,j]; # set initial condition as the first time step
#         for i in 2:Nt
#             @views state[i,:,j] = Φ*state[i-1,:,j]
#         end
#     end
#     return state
# end

function integration(state, t, k, init, coeff_mat)
    Nt,Nv,Nk = size(state); # acquire number of time, variable, and wavenumber
    Δt       = t[2]-t[1]
    
    Φs = @. exp(Δt*coeff_matrix(k; param=params))

    @threads for j in eachindex(k)
        # A = coeff_matrix(k[j]) # coefficient matrix for each wavenumber
        # Φ = exp(Δt*A) # matrix exponential for the coefficient matrix with Δt
        
        Φ = Φs[j]
        @views state[1,:,j] .= init[:,j]; # set initial condition as the first time stepא
        tmp = similar(state[1, :, j])
        for i in 2:Nt
            @views mul!(tmp, Φ, state[i-1, :, j])
            @views state[i, :, j] .= tmp
        end
        # for i in 2:Nt
            # @views state[i,:,j] = Φ*state[i-1,:,j]
        # end
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
