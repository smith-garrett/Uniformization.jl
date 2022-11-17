# Tools for solving continuous-time Markov chains (CTMCs) via uniformization
# Borrows heavily from tpapp/ContinuousTimeMarkov


function issquare(Q)
    s1, s2 = size(Q)
    s1 == s2
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
abstract type AbstractRateMatrix{E} <: AbstractMatrix{E} end


function getmaxrate(Q)
    maximum(abs, diag(Q))
end


# Full transition rate matrix; columns must sum to 0
struct FullRateMatrix{E, T <: AbstractMatrix{E}} <: AbstractRateMatrix{E}
    matrix::T

    function FullRateMatrix(Q)
        if !issquare(Q)
            error("Matrix is not square.")
        end
        if !off_diag_nonnegative(Q)
            error("Matrix contains off-diagonal elements that aren't positive.")
        end

        newQ = setdiagonal!(copy(Q))
        new{eltype(newQ), typeof(newQ)}(newQ)
    end
end


function transient_diag_correct(Q)
    QQ = deepcopy(Q)
    for i in 1:size(QQ, 1)
        @inbounds QQ[i,i] = zero(eltype(QQ))
    end
    all(abs.(diag(Q)) .>= vec(sum(QQ, dims=1)))
end


# Transient rate matrix
struct TransientRateMatrix{E, T <: AbstractMatrix{E}} <: AbstractRateMatrix{E}
    matrix::T

    function TransientRateMatrix(Q)
        if !issquare(Q)
            error("Matrix is not square.")
        end
        if !off_diag_nonnegative(Q)
            error("Matrix contains off-diagonal elements that aren't positive.")
        end
        if !all(diag(Q) .< zero(eltype(Q)))
            error("Diagonal can only contain negative entries.")
        end
        if !transient_diag_correct(Q)
            error("Diagonal elements must be greater than or equal to the column sums.")
        end

        new{eltype(Q), typeof(Q)}(Q)
    end
end


Base.size(m::AbstractRateMatrix) = size(m.matrix)
Base.getindex(m::AbstractRateMatrix, I...) = Base.getindex(m.matrix, I...)
#Base.IndexStyle(::Type{AbstractRateMatrix{E}}) where {E,T} = IndexStyle(T)


"""
    make_dtmc(Q, λ=2^10)

Convert a transition rate matrix 𝐐 for a continuous-time Markov chain to a transition
probability matrix 𝐏 for the corresponding discrete-time Markov chain.
"""
function make_dtmc(Q, λ=2^10)
    return I + Q ./ λ
end

"""
    make_dtmc!(Q, λ=2^10)

Convert a transition rate matrix 𝐐 for a continuous-time Markov chain to a transition
probability matrix 𝐏 for the corresponding discrete-time Markov chain, in place.
"""
function make_dtmc!(Q, λ=2^10)
    Q ./= λ
    Q += I
end

"""
    stationary_distribution(Q)

Calculate the stationary distribution of a full transition rate matrix 𝐐 by solving 𝐐𝐱 = 0.
"""
function stationary_distribution(Q)
    soln = nullspace(Q.matrix)
    return soln ./ sum(soln)
end


"""
    stationary_distribution(Q::TransientRateMatrix)

Return the stationary distribution of Q, which is a vector of zeros for transient rate
matrices.
"""
stationary_distribution(Q::TransientRateMatrix) = zeros(eltype(Q), size(Q, 1))


"""
    uniformize(Q::FullRateMatrix, p0, k=2^10, t=0.0,
               method::Function=discrete_observation_times, args...)

Approximate 𝐩(t) = exp(t𝐐)𝐩(0) using uniformization. The parameter k controls the rate of
transitions occurring in the approximated process. Higher k leads to a better approximation.
Returns a (normalized) distribution over the states at time 𝑡.
"""
function uniformize(Q::FullRateMatrix, p0, k=2^10, t=0.0,
                    method::Function=discrete_observation_times, args...)
    @assert t ≥ zero(t) "Time t must be positive."
    @assert size(p0, 1) == size(Q, 1) "Initial condition p0 must be the same size as Q."
    res = method(Q, k, t, args...) * p0
    res ./ sum(res)
end


"""
    uniformize(Q::TransientRateMatrix, p0, k=2^10, t=0.0,
               method::Function=discrete_observation_times, args...)

Approximate 𝐩(t) = exp(t𝐐)𝐩(0) using uniformization. The parameter k controls the rate of
transitions occurring in the approximated process. Higher k leads to a better approximation.
Returns a non-normalized distribution over the states at time 𝑡.
"""
function uniformize(Q::TransientRateMatrix, p0, k=2^10, t=0.0,
                    method::Function=discrete_observation_times, args...)
    @assert t ≥ zero(t) "Time t must be positive."
    @assert size(p0, 1) == size(Q, 1) "Initial condition p0 must be the same size as Q."
    method(Q, k, t, args...) * p0
end


"""
    standard_uniformization(Q, k=2^10, t=0.0)

Approximate 𝐑(t) = exp(t𝐐) using standard uniformization, where the Rᵢⱼ are the probability
of starting at state 𝑗 and ending at state 𝑖 at time 𝑡. The upper bound of the truncation is
determined automatically on the fly. Matrix powers are calculated incrementally. Still much
less efficient than discrete_observation_times and erlangization.
"""
function standard_uniformization(Q, k=2^10, t=0.0, ϵ=10e-9)
    λ = k / t
    P = copy(Q)
    P = make_dtmc!(P, λ)
    Ppower = copy(P)
    sm = zeros(size(Q))
    δ = 0.0
    n = 0
    # Automatically determine the upper bound for the approximation
    while (1 - δ) ≥ ϵ
        pr = t == zero(t) ? pdf(Poisson(0), n) : pdf(Poisson(λ * t), n)
        if n == 0
            sm .+= pr .* I(size(Q, 1))
        elseif n ==1
            sm .+= pr .* P
        else
            Ppower *= P
            sm .+= pr .* Ppower
        end
        n += 1
        δ += pr
    end
    return sm
end


"""
    discrete_observation_times(Q, k=2^10, t=0.0)

Approximate 𝐑(t) = exp(t𝐐) using P₄ of Yoon & Shanthikumar (1989, p. 181), where the Rᵢⱼ are
the probability of starting at state 𝑗 and ending at state 𝑖 at time 𝑡. The k parameter
should be set to a power of two for efficiency.
"""
#function discrete_observation_times(Q, λ=2^10, t=0.0, args...)
function discrete_observation_times(Q, k=2^10, t=0.0, args...)
    η = getmaxrate(Q)
    # Making sure λ is big enough, Yoon & Shanthikumar, p. 195
    λ = k ≥ t * η ? k / t : t * η
    P = copy(Q)
    P = make_dtmc!(P, λ)
    return P^k
end


"""
    erlangization(Q, k=2^10, t=0.0)

Approximate 𝐑(t) = exp(t𝐐) using P₃ of Yoon & Shanthikumar (1989, p. 179), originally from
Ross (1987), where the Rᵢⱼ are the probability of starting at state 𝑗 and ending at state 𝑖
at time 𝑡. The k parameter should be set to a power of two for efficiency.
"""
function erlangization(Q, k=2^10, t=0.0, args...)
    λ = k / t
    P = inv(I - Q ./ λ)
    return P^k
end

# To do:
# - Make iterative solver for whole trajectory, using the previous solution point as the
# initial condition for the next.


