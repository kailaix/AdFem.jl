using Revise
using PoreFlow
using LinearAlgebra
using PyPlot
using DelimitedFiles
using MAT 

function calc_residual_and_jacobian(θ, u)
    E = θ[1:get_ngauss(mmesh)]
    nu = θ[get_ngauss(mmesh)+1:end]
    μ = E / (2*(1 + nu))
    λ = E*nu/((1 + nu)*(1 - 2*nu))

    ψ, J = neo_hookean(u, μ, λ, mmesh)
    B = compute_fem_source_term(zeros(get_ngauss(mmesh)), -0.5*ones(get_ngauss(mmesh)), mmesh)
    ψ = ψ - B
    J, ψ = impose_Dirichlet_boundary_conditions(J, ψ, bdnode, zeros(length(bdnode)))
    ψ, J
end


mmesh = Mesh(joinpath(PDATA, "twoholes_large.stl"))
mmesh = Mesh(mmesh.nodes * 10, mmesh.elems)


θ = placeholder([10.0 * ones(get_ngauss(mmesh)); 0.3 * ones(get_ngauss(mmesh))])
bdnode = bcnode((x,y)->(x<1e-5) || (x>0.49-1e-5), mmesh)
bdnode = [bdnode; bdnode .+ mmesh.ndof]
nr = newton_raphson_with_grad(calc_residual_and_jacobian, zeros(2mmesh.ndof), θ)

ADCME.options.newton_raphson.verbose = true
sess = Session(); init(sess)
x = run(sess, nr)

matwrite("fenics/fwd1.mat", Dict("x"=>x))

close("all")
figure(figsize = (15, 5))
subplot(121)
u = x[1:mmesh.nnode]
v = x[mmesh.ndof+1:mmesh.ndof + mmesh.nnode]
visualize_scalar_on_fem_points(sqrt.(u.^2+v.^2), mmesh)
subplot(122)
visualize_displacement(x, mmesh)
savefig("fenics/fwd1.png")