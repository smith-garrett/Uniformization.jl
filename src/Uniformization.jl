module Uniformization

using LinearAlgebra: I, eigen, diag
using Distributions: Poisson, pdf

include("ctmc.jl")
export TransitionRateMatrix

end
