module Uniformization

using LinearAlgebra: I, eigen, diag, inv, nullspace
using Distributions: Poisson, pdf, cdf, quantile

include("ctmc.jl")
export FullRateMatrix, TransientRateMatrix
export uniformize, standard_uniformization, discrete_observation_times, erlangization

end
