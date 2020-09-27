using Revise
using PoreFlow
using LinearAlgebra
using PyPlot
using DelimitedFiles
using MAT 
using Optim

function f(x, y)
    10/(1+x^2) + x * y + 10*y^2
end


function calc_residual_and_jacobian(θ, u)
    E = squeeze(fc(gauss_nodes(mmesh), [20,20,20,1], θ)) + 10.0
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

# E = eval_f_on_gauss_pts(f, mmesh)
θ = Variable(fc_init([2, 20, 20, 20, 1]))
nu = constant(0.3 * ones(get_ngauss(mmesh)))
bdnode = bcnode((x,y)->(x<1e-5) || (x>0.49-1e-5), mmesh)
bdnode = [bdnode; bdnode .+ mmesh.ndof]
nr = newton_raphson_with_grad(calc_residual_and_jacobian, zeros(2mmesh.ndof), θ)
u = matread("fenics/fwd2.mat")["x"]
loss = sum((nr-u)^2) * 1e10
g = gradients(loss, θ)
sess = Session(); init(sess)

_loss = Float64[]
cb = (x, iter, loss)->begin 
    global _loss
    push!(_loss, loss)
    printstyled("[#iter $iter] loss=$loss\n", color=:green)
    nu = run(sess, squeeze(fc(gauss_nodes(mmesh), [20,20,20,1], x)) + 10.0)
    close("all")
    visualize_scalar_on_gauss_points(nu, mmesh)
    if mod(iter, 10)==1
        matwrite("fenics/bwd2-$iter.mat", Dict("iter"=>iter,"loss"=>_loss, "E"=>nu))
        savefig("fenics/bwd2_nn$iter.png")
    end
end

# run(sess, loss)
loss_ = BFGS!(sess, loss, g, θ, callback = cb)

