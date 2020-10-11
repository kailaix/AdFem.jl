# Solves the Poisson equation using the Mixed finite element method 
using Revise
using PoreFlow
using DelimitedFiles
using SparseArrays
using PyPlot


n = 50
mmesh = Mesh(n, n, 1/n, degree = BDM1)

testCase = [
    ((x, y)->begin
        x * (1-x) * y * (1-y)
    end, 
    (x, y)->begin
        -2x*(1-x) -2y*(1-y)
    end),  # test case 1
    ((x, y)->begin
        x^2 * (1-x) * y * (1-y)^2
    end, 
    (x, y)->begin
        2*x^2*y*(1 - x) + 2*x^2*(1 - x)*(2*y - 2) - 4*x*y*(1 - y)^2 + 2*y*(1 - x)*(1 - y)^2
    end) ,# test case 2
    (
        (x, y)->x*y, 
        (x, y)->0.0
    ), # test case 3
    (
        (x, y)->x^2 * y^2 + 1/(1+x^2), 
        (x, y)->2*x^2 + 8*x^2/(x^2 + 1)^3 + 2*y^2 - 2/(x^2 + 1)^2
    ), # test case 4
]

for k = 1:4
    @info "TestCase $k..."
    ufunc, ffunc = testCase[k]

    α = ones(get_ngauss(mmesh))
    β = zeros(get_ngauss(mmesh))

    A = compute_fem_bdm_mass_matrix(α, β, mmesh)[1:mmesh.ndof, 1:mmesh.ndof]
    B = compute_fem_bdm_div_matrix(mmesh)
    C = [A -B'
        -B spzeros(mmesh.nelem, mmesh.nelem)]

    bdedge = bcedge(mmesh)
    t1 = eval_f_on_boundary_edge(ufunc, bdedge, mmesh)
    g = compute_fem_traction_term1(t1, bdedge, mmesh) 
    t2 = eval_f_on_gauss_pts(ffunc, mmesh)
    f = compute_fvm_source_term(t2, mmesh)
    rhs = [-g; f]

    sol = C\rhs
    u = sol[mmesh.ndof+1:end]
    close("all")
    figure(figsize=(15, 5))
    subplot(131)
    xy = fvm_nodes(mmesh)
    x, y = xy[:,1], xy[:,2]
    uf = ufunc.(x, y)
    visualize_scalar_on_fvm_points(uf, mmesh)
    subplot(132)
    visualize_scalar_on_fvm_points(u, mmesh)
    subplot(133)
    visualize_scalar_on_fvm_points( abs.(u - uf) , mmesh)
    savefig("bdm$k.png")
end