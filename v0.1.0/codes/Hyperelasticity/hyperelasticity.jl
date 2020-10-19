using Revise
using AdFem
using LinearAlgebra
using PyPlot
using DelimitedFiles


function calc_residual_and_jacobian(θ, u)
    ψ, J = neo_hookean(u, μ, λ, mmesh)
    B = compute_fem_source_term(zeros(get_ngauss(mmesh)), -0.5*ones(get_ngauss(mmesh)), mmesh)
    ψ = ψ - B
    J, ψ = impose_Dirichlet_boundary_conditions(J, ψ, bdnode, zeros(length(bdnode)))
    ψ, J
end


mmesh = Mesh(8, 8 , 1/8)

E, nu = 10.0, 0.3
μ, λ = E/(2*(1 + nu)) * ones(get_ngauss(mmesh)), E*nu/((1 + nu)*(1 - 2*nu)) * ones(get_ngauss(mmesh))
bdnode = bcnode((x,y)->(x<1e-5) || (x>1-1e-5), mmesh)
bdnode = [bdnode; bdnode .+ mmesh.ndof]
nr = newton_raphson(calc_residual_and_jacobian, zeros(2mmesh.ndof), missing)

ADCME.options.newton_raphson.verbose = true
sess = Session(); init(sess)
nr = run(sess, nr)

close("all")
# visualize_displacement(nr.x, mmesh)
u = nr.x[1:mmesh.nnode]
v = nr.x[mmesh.ndof+1:mmesh.ndof + mmesh.nnode]
visualize_scalar_on_fem_points(sqrt.(u.^2+v.^2), mmesh)
savefig("test_AdFem.png")