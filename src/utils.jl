# Utility functions for Uniformization.jl

"""
    issquare(Q)

Test whether the matrix Q is square.
"""
function issquare(Q)
    s1, s2 = size(Q)
    return s1 == s2
end


"""
    getdiagonal(Q)

Get the row sums of Q to be used for setting the diagonal of a transition rate matrix.
"""
function getdiagonal(Q)
    return sum(Q, dims=1)
end


"""
    setdiagonal!(Q)

Set the diagonal of Q to be the row sums. Modifies Q.
"""
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


"""
    off_diag_nonnegative(Q)

Test whether the off-diagonal elements of Q are non-negative, as is required for transition
rate matrices.
"""
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


"""
    getmaxrate(Q)

Return the largest rate from the diagonal of the matrix Q.
"""
function getmaxrate(Q)
    return maximum(abs, diag(Q))
end


#"""
#    transient_diag_correct(Q)
#
#Test whether the diagonal of a transient rate matrix is correct.
#"""
#function transient_diag_correct(Q)
#    QQ = deepcopy(Q)
#    for i in 1:size(QQ, 1)
#        @inbounds QQ[i,i] = zero(eltype(QQ))
#    end
#    all(abs.(diag(Q)) .>= vec(sum(QQ, dims=1)))
#end


