module Uniformization

using LinearAlgebra: I, eigen, diag, inv, nullspace
using Distributions: Poisson, pdf

include("ctmc.jl")
export TransitionRateMatrix, uniformize, standard_uniformization, discrete_observation_times
export erlangization

end
