sess = Session()

@testset "fem_to_gauss_points" begin
m = 10
n = 10
h = 0.1
u = rand((m+1)*(n+1))
u0 = fem_to_gauss_points(u, m, n, h)
@test run(sess, fem_to_gauss_points(constant(u), m, n, h)) ≈ u0
end