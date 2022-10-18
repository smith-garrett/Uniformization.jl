module Uniformization

using LinearAlgebra: I, eigen, diag, inv
using Distributions: Poisson, pdf

include("ctmc.jl")
export TransitionRateMatrix

end
