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

# Define an abstract type for all transition rate matrices
abstract struct AbstractRateMatrix{E} <: AbstractMatrix{E} end

function getmaxrate(Q)
    maximum(abs, diag(q))
end

# Full transition rate matrix; columns must sum to 0
struct TransitionRateMatrix{E, T <: AbstractMatrix{E}} <: AbstractRateMatrix{E}
    matrix::T

    function TransitionRateMatrix(Q)
        if !(issquare(Q))
            error("Matrix is not square.")
        end
        if !(off_diag_nonnegative(Q))
            error("Matrix contains off-diagonal elements that aren't positive.")
        end

        newQ = setdiagonal!(copy(Q))
        new{eltype(newQ), typeof(newQ)}(newQ, max_rate)
    end
end

# Transient rate matrix
struct TransientRateMatrix{E, T <: AbstractMatrix{E}} <: AbstractRateMatrix{E}
    matrix::T

    function TransitionRateMatrix(Q)
        if !(issquare(Q))
            error("Matrix is not square.")
        end
        if !(off_diag_nonnegative(Q))
            error("Matrix contains off-diagonal elements that aren't positive.")
        end

        new{eltype(Q), typeof(Q)}(Q)
    end
end
Base.size(m::AbstractRateMatrix) = size(m.matrix)
Base.getindex(m::AbstractRateMatrix, I...) = Base.getindex(m.matrix, I...)
Base.IndexStyle(::Type{AbstractRateMatrix{E,T}}) where {E,T} = IndexStyle(T)

function make_dtmc(Q, λ=2^10)
    ndim = size(Q, 1)
    return I(ndim) + Q ./ λ
end

function stationary_distribution(Q)
    soln = nullspace(Q.matrix)
    return soln ./ sum(soln)
end

"""
    uniformize(Q, p0, λ=2^10, t=0.0, method::Function=discrete_observation_times, args...)

Approximate 𝐩(t) = exp(t𝐐)𝐩(0) using uniformization. The parameter λ controls the rate of
transitions occurring in the approximated process. Higher λ leads to a better approximation.
Returns a (normalized) Distributions.Categorical distribution over the states at time 𝑡.
"""
function uniformize(Q, p0, λ=2^10, t=0.0,
                    method::Function=discrete_observation_times, args...)
    @assert t ≥ zero(t) "Time t must be positive."
    @assert size(p0, 1) == size(Q, 1) "Initial condition p0 must be the same size as Q."
    res = method(Q, λ, t, args...) * p0
    Categorical(res ./ sum(res))
end

"""
    standard_uniformization(Q::TransitionRateMatrix, λ=2^10, t=0.0)

Approximate 𝐑(t) = exp(t𝐐) using standard uniformization, where the Rᵢⱼ are the probability
of starting at state 𝑗 and ending at state 𝑖 at time 𝑡. The upper bound of the truncation is
determined automatically on the fly. Matrix powers are calculated incrementally. Still much
less efficient than discrete_observation_times and erlangization.
"""
function standard_uniformization(Q λ=2^10, t=0.0, ϵ=10e-9)
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
    discrete_observation_times(Q::TransitionRateMatrix, λ=2^10, t=0.0)

Approximate 𝐑(t) = exp(t𝐐) using P₄ of Yoon & Shanthikumar (1989, p. 181), where the Rᵢⱼ are
the probability of starting at state 𝑗 and ending at state 𝑖 at time 𝑡. The default λ is
usually much too small for a good approximation. Powers of two seem to work well.

"""
function discrete_observation_times(Q::TransitionRateMatrix, λ=2^10, t=0.0, args...)
    P = make_dtmc(Q, λ)
    return P^floor(Int, λ * t)
end

"""
    erlangization(Q::TransitionRateMatrix, λ=2^10, t=0.0)

Approximate 𝐑(t) = exp(t𝐐) using P₃ of Yoon & Shanthikumar (1989, p. 179), originally from
Ross (1987), where the Rᵢⱼ are the probability of starting at state 𝑗 and ending at state 𝑖
at time 𝑡.
"""
function erlangization(Q::TransitionRateMatrix, λ=2^10, t=0.0, args...)
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
# - Implement a version of the uniformization fn. as a subtype of discrete multivariate
# distribution
# - Make iterative solver for whole trajectory, using the previous solution point as the
# initial condition for the next.


