@testset "Basic tests" begin
    badmatrix = [-1.0 1; -2 1]
    @test Uniformization.off_diag_nonnegative(badmatrix) == false
    fixable_matrix = [-1.0 1; 2 1]
    corrected = [-2.0 1; 2 -1]
    @test all(Uniformization.setdiagonal!(fixable_matrix) .== corrected)
    @test Uniformization.issquare(fixable_matrix)
    @test Uniformization.issquare([1 2 3; 4 5 6]) == false
end

