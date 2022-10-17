# Tools for solving continuous-time Markov chains (CTMCs) via uniformization
# Borrows heavily from tpapp/ContinuousTimeMarkov


function is_square(Q)
    s1, s2 = size(Q)
    s1 == s2 ? true : false
end

function get_diagonal(Q)
    return sum(Q, dims=1)
end

function set_diagonal!(Q)
    for i in 1:size(Q, 1)
        @inbounds Q[i,i] = zero(eltype(Q))
    end
    d = get_diagonal(Q)
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

struct TransitionRateMatrix{E, T <: AbstractMatrix} <: AbstractMatrix{E}
    matrix::T
    function TransitionRateMatrix(Q::T) where {E, T <: AbstractMatrix{E}}
        if !(is_square(Q))
            error("Matrix is not square")
        end
        if !(off_diag_nonnegative(Q))
            error("Matrix contains off-diagonal elements that aren't positive.")
        end
        newQ = set_diagonal!(copy(Q))
        new{E, T}(newQ)
    end
end

#TransitionRateMatrix(Q::TransitionRateMatrix) = TransitionRateMatrix(Q.matrix)

Base.size(m::TransitionRateMatrix) = size(m.matrix)
Base.getindex(m::TransitionRateMatrix, I...) = Base.getindex(m.matrix, I...)
Base.IndexStyle(::Type{TransitionRateMatrix{E,T}}) where {E,T} = IndexStyle(T)

function make_dtmc(Q::TransitionRateMatrix)
    max_rate = maximum(abs.(diag(Q)))
    ndim = size(Q, 1)
    return I(ndim) + Q ./ max_rate
end

function stationary_distribution(Q::TransitionRateMatrix)
    eigvals, eigvecs = eigen(Q)
    idx = findall(x -> iszero.(x), eigvals)
    if length(idx) != 1
        error("More than one stationary state found.")
    end
    return eigvecs[:,idx] ./ sum(eigvecs[:,idx])
end

function solve(Q::TransitionRateMatrix, p0, t)
    max_rate = maximum(abs.(diag(Q)))
    P = make_dtmc(Q)
    #probs = zeros(length(p0))
    sm = zeros(size(Q))
    for i = 0:25
        #probs .+= exp(-max_rate * t) * (max_rate * t)^i / factorial(i) .* (P^i * p0) 
        #probs ./= sum(probs)
        #sm .+= exp(-max_rate * t) * (max_rate * t)^i / factorial(i) .* P^i
        sm .+= pdf(Poisson(max_rate * t), i) .* P^i
    end
    #return probs
    return sm * p0
end


