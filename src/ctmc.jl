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

#struct TransitionRateMatrix{E, T <: AbstractMatrix} <: AbstractMatrix{E}
#    matrix::T
#    #max_rate::E
#    function TransitionRateMatrix(Q::T) where {E, T <: AbstractMatrix{E}}
#        if !(is_square(Q))
#            error("Matrix is not square")
#        end
#        if !(off_diag_nonnegative(Q))
#            error("Matrix contains off-diagonal elements that aren't positive.")
#        end
#        newQ = set_diagonal!(copy(Q))
#        new{E, T}(newQ, max_rate)
#    end
#end

struct TransitionRateMatrix{E, T <: AbstractMatrix{E}} <: AbstractMatrix{E}
    matrix::T
    max_rate::E

    function TransitionRateMatrix(Q)
        if !(is_square(Q))
            error("Matrix is not square.")
        end
        if !(off_diag_nonnegative(Q))
            error("Matrix contains off-diagonal elements that aren't positive.")
        end

        newQ = set_diagonal!(copy(Q))
        max_rate = maximum(abs, diag(newQ))
        new{eltype(newQ), typeof(newQ)}(newQ, max_rate)
    end
end

Base.size(m::TransitionRateMatrix) = size(m.matrix)
Base.getindex(m::TransitionRateMatrix, I...) = Base.getindex(m.matrix, I...)
Base.IndexStyle(::Type{TransitionRateMatrix{E,T}}) where {E,T} = IndexStyle(T)

function make_dtmc(Q::TransitionRateMatrix)
    ndim = size(Q, 1)
    return I(ndim) + Q ./ Q.max_rate
end

function stationary_distribution(Q::TransitionRateMatrix)
    eigvals, eigvecs = eigen(Q)
    idx = findall(x -> iszero.(x), eigvals)
    if length(idx) != 1
        error("More than one stationary state found.")
    end
    return eigvecs[:,idx] ./ sum(eigvecs[:,idx])
end

#function solve(Q::TransitionRateMatrix, p0, t)
#    max_rate = maximum(abs.(diag(Q)))
#    P = make_dtmc(Q)
#    #probs = zeros(length(p0))
#    sm = zeros(size(Q))
#    for i = 0:25
#        #probs .+= exp(-max_rate * t) * (max_rate * t)^i / factorial(i) .* (P^i * p0) 
#        #probs ./= sum(probs)
#        #sm .+= exp(-max_rate * t) * (max_rate * t)^i / factorial(i) .* P^i
#        sm .+= pdf(Poisson(max_rate * t), i) .* P^i
#    end
#    #return probs
#    return sm * p0
#end


"""
    solve(Q, p0, t)

Solve for exp(Qt) * p0 using the method P₄ from Yoon & Shanthikumar (1989).
"""

function solve(Q::TransitionRateMatrix, p0, t, λ)
    P = make_dtmc(Q)
    return P^(floor(t * λ)) * p0
end

function solve(Q::TransitionRateMatrix, p0, t)
    solve(Q, p0, t, Q.max_rate)
end
#function get_delta(max_rate, ϵ=1e-12)
#    return -max_rate^-1 * log(1 - ϵ)
#end

#kronecker(n1, n2) = n1 == n2

#function make_P2(Q::TransitionRateMatrix, Δ)
#    P = zeros(size(Q))
#    diags = abs.(diag(Q))
#    for idx in CartesianIndices(Q)
#        P[idx] = (1 - exp(-diags[idx[1]] * Δ)) * (Q[idx] / diags[idx[1]]) + exp(-diags[idx[1]] * Δ) * kronecker(idx[1], idx[2])
#    end
#    return P
#end

#function make_P3(Q::TransitionRateMatrix)
#    max_rate = maximum(abs.(diag(Q)))
#    P3 = inv(I(size(Q, 1)) - Q ./ max_rate)
#end



