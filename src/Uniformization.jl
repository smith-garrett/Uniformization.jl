module Uniformization

using LinearAlgebra: I, eigen, diag, inv, nullspace
using Distributions: Poisson, pdf, Categorical, probs, params, ncategories
export probs

include("ctmc.jl")
export FullRateMatrix, TransientRateMatrix
export uniformize, standard_uniformization, discrete_observation_times, erlangization

end
