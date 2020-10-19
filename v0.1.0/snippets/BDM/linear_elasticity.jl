# Solves the Poisson equation using the Mixed finite element method 
using Revise
using AdFem
using DelimitedFiles
using SparseArrays
using PyPlot


TestCase = [
    (
        (x,y)->begin;x*y*(x - 1)*(y - 1);end,
        (x,y)->begin;x*y*(x - 1)*(y - 1);end,
        (x,y)->begin;x^2 + 2*x*y - 2*x + 2*y^2 - 3*y + 1/2;end,
        (x,y)->begin;2*x^2 + 2*x*y - 3*x + y^2 - 2*y + 1/2;end,
    ),
    (
        (x,y)->begin;x^2*y^2*(x - 1)*(y^2 - 1);end,
        (x,y)->begin;x^2*y^2*(x - 1)*(y^2 - 1);end,
        (x,y)->begin;6*x^3*y^2 - x^3 + 6*x^2*y^3 - 6*x^2*y^2 - 3*x^2*y + x^2 + 6*x*y^4 - 4*x*y^3 - 6*x*y^2 + 2*x*y - 2*y^4 + 2*y^2;end,
        (x,y)->begin;12*x^3*y^2 - 2*x^3 + 6*x^2*y^3 - 12*x^2*y^2 - 3*x^2*y + 2*x^2 + 3*x*y^4 - 4*x*y^3 - 3*x*y^2 + 2*x*y - y^4 + y^2;end,
    ),
    (
        (x,y)->begin;x^2*y^2*(x - 1)*(y^2 - 1);end,
        (x,y)->begin;x*y*(x - 1)*(y - 1);end,
        (x,y)->begin;6*x^3*y^2 - x^3 - 6*x^2*y^2 + x^2 + 6*x*y^4 - 6*x*y^2 + 2*x*y - x - 2*y^4 + 2*y^2 - y + 1/2;end,
        (x,y)->begin;6*x^2*y^3 - 3*x^2*y + 2*x^2 - 4*x*y^3 + 2*x*y - 2*x + y^2 - y;end,
    )
]

for k = 1:length(TestCase)
    @info "Running TestCase $k..."
    ufunc, vfunc, gfunc, hfunc = TestCase[k]

    n = 50
    mmesh = Mesh(n, n, 1/n, degree = BDM1)


    A = compute_fem_bdm_mass_matrix(mmesh)
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
    savefig("elasticity$k.png")
    end