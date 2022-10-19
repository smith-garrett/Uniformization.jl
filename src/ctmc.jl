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
#    eigvals, eigvecs = eigen(Q.matrix)
#    idx = findall(x -> iszero.(x), eigvals)
#    if length(idx) != 1
#        error("More than one stationary state found.")
#    end
#    return eigvecs[:,idx] ./ sum(eigvecs[:,idx])
    soln = nullspace(Q.matrix)
    return soln ./ sum(soln)
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
#function solve(Q::TransitionRateMatrix, p0, t, λ=Q.max_rate)
#function solve(Q::TransitionRateMatrix, p0, t, k=2^8)
function solve(Q::TransitionRateMatrix, p0, t, λ=Q.max_rate, ϵ=10e-9)
    @assert t ≥ zero(t) "Time t must be positive."
    @assert size(p0, 1) == size(Q, 1) "Initial condition p0 must be the same size as Q."
    P = make_dtmc(Q, λ)
    sm = zeros(size(Q))
    δ = 1.0
    k = 0
    # Automatically determine the upper bound for the approximation
    while δ ≥ ϵ
        pr = pdf(Poisson(Q.max_rate * t), k)
        #sm .+= pr .* P^k
        if k == 0
            sm .+= pr .* I(size(Q, 1))
        elseif k ==1
            sm .+= pr .* P
        else
            P *= P
            sm .+= pr .* P
        end
        δ -= pr
        k += 1
    end
    #res = sm * p0
    return sm * p0  #res ./ sum(res)
end

# Doesn't seem to work well
function P4(Q::TransitionRateMatrix, p0, t, λ=Q.max_rate)
    P = make_dtmc(Q, λ)
    return P^floor(λ * t) * p0
end

function P3(Q::TransitionRateMatrix, p0, t, λ=Q.max_rate)
    P = inv(I(size(Q, 1)) - Q ./ λ)
    return P^floor(λ * t) * p0
end

#function solve(Q::TransitionRateMatrix, p0, t)
#    solve(Q, p0, t, Q.max_rate)
#end

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



