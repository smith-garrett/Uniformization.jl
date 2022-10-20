# Tools for solving continuous-time Markov chains (CTMCs) via uniformization
# Borrows heavily from tpapp/ContinuousTimeMarkov


function issquare(Q)
    s1, s2 = size(Q)
    s1 == s2 ? true : false
end

function getdiagonal(Q)
    return sum(Q, dims=1)
end

function setdiagonal!(Q)
    for i in 1:size(Q, 1)
        @inbounds Q[i,i] = zero(eltype(Q))
    end
    d = getdiagonal(Q)
    for i in eachindex(d)
        @inbounds Q[i,i] = -d[i]
    end
    return Q
end

function off_diag_nonnegative(Q)
    for idx in CartesianIndices(Q)
        if idx[1] == idx[2]
            continue
        elseif Q[idx] < zero(eltype(Q))
            return false
        end
    end
    return true
end

struct TransitionRateMatrix{E, T <: AbstractMatrix{E}} <: AbstractMatrix{E}
    matrix::T
    max_rate::E

    function TransitionRateMatrix(Q)
        if !(issquare(Q))
            error("Matrix is not square.")
        end
        if !(off_diag_nonnegative(Q))
            error("Matrix contains off-diagonal elements that aren't positive.")
        end

        newQ = setdiagonal!(copy(Q))
        max_rate = maximum(abs, diag(newQ))
        new{eltype(newQ), typeof(newQ)}(newQ, max_rate)
    end
end

Base.size(m::TransitionRateMatrix) = size(m.matrix)
Base.getindex(m::TransitionRateMatrix, I...) = Base.getindex(m.matrix, I...)
Base.IndexStyle(::Type{TransitionRateMatrix{E,T}}) where {E,T} = IndexStyle(T)

function make_dtmc(Q::TransitionRateMatrix, λ=Q.max_rate)
    ndim = size(Q, 1)
    return I(ndim) + Q ./ λ
end

function stationary_distribution(Q::TransitionRateMatrix)
    soln = nullspace(Q.matrix)
    return soln ./ sum(soln)
end
"""
    solve(Q, p0, t)

Approximate 𝐩(t) = exp(t𝐐)𝐩(0) using uniformization. The parameter λ controls the rate of
transitions occurring in the approximated process. Higher λ leads to a better approximation.
"""
function uniformize(Q::TransitionRateMatrix, method::Function, p0, t, λ=Q.max_rate, args...)
    @assert t ≥ zero(t) "Time t must be positive."
    @assert size(p0, 1) == size(Q, 1) "Initial condition p0 must be the same size as Q."
    method(Q, t, λ, args...) * p0
end

"""
    standard_uniformization(Q::TransitionRateMatrix, p0, t, λ=Q.max_rate)

Approximate 𝐩(t) = exp(t𝐐)𝐩(0) using standard uniformization. The upper bound of the
truncation is determined automatically on the fly. Matrix powers are calculated
incrementally. Still much less efficient than discrete_observation_times and erlangization.
"""
function standard_uniformization(Q::TransitionRateMatrix, t, λ=Q.max_rate, ϵ=10e-9)
    P = make_dtmc(Q, λ)
    Ppower = deepcopy(P)
    sm = zeros(size(Q))
    δ = 0.0
    k = 0
    # Automatically determine the upper bound for the approximation
    while (1 - δ) ≥ ϵ
        pr = pdf(Poisson(λ * t), k)
        if k == 0
            sm .+= pr .* I(size(Q, 1))
        elseif k ==1
            sm .+= pr .* P
        else
            Ppower *= P
            sm .+= pr .* Ppower
        end
        k += 1
        δ += pr
    end
    return sm
end

"""
    discrete_observation_times(Q::TransitionRateMatrix, p0, t, λ=Q.max_rate)

Approximate 𝐩(t) = exp(t𝐐)𝐩(0) using P₄ of Yoon & Shanthikumar (1989, p. 181). The default λ
is usually much too small for a good approximation. Powers of two seem to work well.

"""
function discrete_observation_times(Q::TransitionRateMatrix, t, λ=Q.max_rate, args...)
    P = make_dtmc(Q, λ)
    return P^floor(Int, λ * t)
end

"""
    erlangization(Q::TransitionRateMatrix, p0, t, λ=Q.max_rate)

Approximate 𝐩(t) = exp(t𝐐)𝐩(0) using P₃ of Yoon & Shanthikumar (1989, p. 179), originally
from Ross (1987).
"""
function erlangization(Q::TransitionRateMatrix, t, λ=Q.max_rate, args...)
    P = inv(I(size(Q, 1)) - Q ./ λ)
    return P^floor(Int, λ * t)
end

#kronecker(n1, n2) = n1 == n2

#function make_P2(Q::TransitionRateMatrix, Δ)
#    P = zeros(size(Q))
#    diags = abs.(diag(Q))
#    for idx in CartesianIndices(Q)
#        P[idx] = (1 - exp(-diags[idx[1]] * Δ)) * (Q[idx] / diags[idx[1]]) + exp(-diags[idx[1]] * Δ) * kronecker(idx[1], idx[2])
#    end
#    return P
#end

# To do:
# - Implement a version of the uniformization fn. as a subtype of discrete multivariate distribution
# - Make iterative solver for whole trajectory, using the previous solution point as the
# initial condition for the next.


