# Solves the Poisson equation using the Mixed finite element method 
using Revise
using AdFem
using DelimitedFiles
using SparseArrays
using PyPlot


λ = 1.0
μ = 1.0
n = 50
mmesh = Mesh(n, n, 1/n, degree = BDM1)
a = 1/2μ
b = -λ/(2μ*(2μ+2λ))

TestCase = [
    (
        (x,y)->begin;x*y*(x - 1)*(y - 1);end,
        (x,y)->begin;x*y*(x - 1)*(y - 1);end,
        (x,y)->begin;2.0*x^2 + 8.0*x*y - 6.0*x + 6.0*y^2 - 10.0*y + 2.0;end,
        (x,y)->begin;6.0*x^2 + 8.0*x*y - 10.0*x + 2.0*y^2 - 6.0*y + 2.0;end,
    ),
    (
        (x,y)->begin;x^2*y^2*(x - 1)*(y^2 - 1);end,
        (x,y)->begin;x^2*y^2*(x - 1)*(y^2 - 1);end,
        (x,y)->begin;12.0*x^3*y^2 - 2.0*x^3 + 24.0*x^2*y^3 - 12.0*x^2*y^2 - 12.0*x^2*y + 2.0*x^2 + 18.0*x*y^4 - 16.0*x*y^3 - 18.0*x*y^2 + 8.0*x*y - 6.0*y^4 + 6.0*y^2;end,
        (x,y)->begin;36.0*x^3*y^2 - 6.0*x^3 + 24.0*x^2*y^3 - 36.0*x^2*y^2 - 12.0*x^2*y + 6.0*x^2 + 6.0*x*y^4 - 16.0*x*y^3 - 6.0*x*y^2 + 8.0*x*y - 2.0*y^4 + 2.0*y^2;end,
    ),
    (
        (x,y)->begin;x^2*y^2*(x - 1)*(y^2 - 1);end,
        (x,y)->begin;x*y*(x - 1)*(y - 1);end,
        (x,y)->begin;12.0*x^3*y^2 - 2.0*x^3 - 12.0*x^2*y^2 + 2.0*x^2 + 18.0*x*y^4 - 18.0*x*y^2 + 8.0*x*y - 4.0*x - 6.0*y^4 + 6.0*y^2 - 4.0*y + 2.0;end,
        (x,y)->begin;24.0*x^2*y^3 - 12.0*x^2*y + 6.0*x^2 - 16.0*x*y^3 + 8.0*x*y - 6.0*x + 2.0*y^2 - 2.0*y;end,
    )

]

for k = 1:length(TestCase)
    @info "Running TestCase $k..."
    ufunc, vfunc, gfunc, hfunc = TestCase[k]

    A = compute_fem_bdm_mass_matrix(a*ones(get_ngauss(mmesh)), b*ones(get_ngauss(mmesh)), mmesh)
    B = compute_fem_bdm_div_matrix(mmesh)
    C = compute_fem_bdm_skew_matrix(mmesh)

    D = [A B' C'
        B spzeros(2mmesh.nelem, 3mmesh.nelem)
        C spzeros(mmesh.nelem, 3mmesh.nelem)]

    gD = bcedge(mmesh) 
    t1 = eval_f_on_gauss_pts(gfunc, mmesh)
    t2 = eval_f_on_gauss_pts(hfunc, mmesh)
    f1 = compute_fvm_source_term(t1, mmesh)
    f2 = compute_fvm_source_term(t2, mmesh)

    rhs = [zeros(2mmesh.ndof); f1; f2; zeros(mmesh.nelem)]

    sol = D\rhs
    u = sol[2mmesh.ndof+1:2mmesh.ndof+2mmesh.nelem]
    close("all")
    figure(figsize=(15, 10))
    subplot(231)
    title("Reference")
    xy = fvm_nodes(mmesh)
    x, y = xy[:,1], xy[:,2]
    uf = ufunc.(x, y)
    visualize_scalar_on_fvm_points(uf, mmesh)
    subplot(232)
    title("Numerical")
    visualize_scalar_on_fvm_points(u[1:mmesh.nelem], mmesh)
    subplot(233)
    title("Absolute Error")
    visualize_scalar_on_fvm_points( abs.(u[1:mmesh.nelem] - uf) , mmesh)

    subplot(234)
    xy = fvm_nodes(mmesh)
    x, y = xy[:,1], xy[:,2]
    uf = vfunc.(x, y)
    visualize_scalar_on_fvm_points(uf, mmesh)
    subplot(235)
    visualize_scalar_on_fvm_points(u[mmesh.nelem+1:end], mmesh)
    subplot(236)
    visualize_scalar_on_fvm_points( abs.(u[mmesh.nelem+1:end] - uf) , mmesh)
    savefig("varying_elasticity$k.png")
end