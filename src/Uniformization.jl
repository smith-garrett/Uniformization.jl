module Uniformization

using LinearAlgebra: I, eigen, diag, inv, nullspace
using Distributions: Poisson, pdf, cdf, quantile

include("algorithms.jl")
export FullRateMatrix
export uniformize, standard_uniformization, discrete_observation_times, erlangization

end
