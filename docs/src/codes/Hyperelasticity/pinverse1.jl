using Revise
using PoreFlow
using LinearAlgebra
using PyPlot
using DelimitedFiles
using Optim
using MAT

function calc_residual_and_jacobian(θ, u)
    E = θ[1] * constant(ones(get_ngauss(mmesh)))
    nu = 0.3 * constant(ones(get_ngauss(mmesh)))
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


θ = Variable([1.0; 0.0])
bdnode = bcnode((x,y)->(x<1e-5) || (x>0.49-1e-5), mmesh)
bdnode = [bdnode; bdnode .+ mmesh.ndof]
nr = newton_raphson_with_grad(calc_residual_and_jacobian, zeros(2mmesh.ndof), θ)

U = matread("fenics/fwd1.mat")["x"]

loss = sum((nr - U)^2)
g = tf.convert_to_tensor(gradients(loss, θ))

sess = Session(); init(sess)

BFGS!(sess, loss, g, θ)

# close("all")
# gradview(sess, θ, loss, [5.0])
# savefig("gradtest.png")