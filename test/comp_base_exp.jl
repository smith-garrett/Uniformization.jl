@testset "Comparison to base exp()" begin
    mat = [-1.0 1 0; 1 -2 1; 0 1 -1]
    Q = TransitionRateMatrix(mat)
    k = 2^12
    tvals = [0.0, 1.0]
    p0 = [1.0, 0, 0]
    rt = 1e-4
    for t in tvals
        expQ = exp(mat * t)
        pvec = expQ * p0
        # Standard uniformization is not very accurate
        @test all(isapprox.(standard_uniformization(Q, 2^14, t; p0=p0), pvec, atol=1e-2))

        discobs = discrete_observation_times(Q, k, t)
        @test all(isapprox.(discobs, expQ, rtol=rt))
        @test all(isapprox.(discobs * p0, pvec, rtol=rt))

        erl = erlangization(Q, k, t)
        @test all(isapprox.(erl, expQ, rtol=rt))
        @test all(isapprox.(erl * p0, pvec, rtol=rt))
    end
end
