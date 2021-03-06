using ADCME
using AdFem

include("../chip_unstructured_solver.jl")
include("../chip_unstructured_geometry.jl")

k_mold = 0.014531
k_chip_ref = 2.60475
k_air = 0.64357

function k_exact(x, y)
    k_mold + 1000 * k_chip_ref * (x-0.49)^2 / (1 + x^2)
end

function k_nn(x, y, θ) # exact solution + nn: change to nn solution?
    constant(@. k_exact(x, y)) +  fc(x, [20,20,20,1], θ)
end

nu = 0.47893  # equal to 1/Re
power_source = 82.46295  #82.46295 = 1.0e6 divide by air rho cp   #0.0619 = 1.0e6 divide by chip die rho cp
buoyance_coef = 299102.83

u_std = 0.001
p_std = 0.000001225
T_infty = 300

NT = 14    # number of iterations for Newton's method

xy = mesh.nodes 
xy2 = zeros(mesh.nedge, 2)
for i = 1:mesh.nedge
    xy2[i,:] = (mesh.nodes[mesh.edges[i,1], :] + mesh.nodes[mesh.edges[i,2], :])/2
end
xy = [xy;xy2]

x, y = xy[chip_fem_idx, 1], xy[chip_fem_idx, 2]
k_chip_exact = constant( @. k_exact(x, y) )

heat_source_fem = zeros(ndof)
heat_source_fem[chip_fem_top_idx] .= power_source
heat_source_gauss = dof_to_gauss_points(heat_source_fem, mesh)
heat_source = compute_fem_source_term1(constant(heat_source_gauss), mesh)

B = constant(compute_interaction_matrix(mesh))
Laplace = constant(compute_fem_laplace_matrix1(nu * constant(ones(ngauss)), mesh))

# apply Dirichlet to velocity and temperature; set left bottom two points to zero to fix rank deficient problem for pressure
bd = [bd; bd .+ ndof; 
      fvm_bd .+ 2*ndof; 
      bd .+ (2*ndof+nelem)]

# add solid region into boundary condition for u, v, p, i.e. exclude solid when solving Navier Stokes
bd = [bd; solid_fem_idx; solid_fem_idx .+ ndof; solid_fvm_idx .+ 2*ndof]

S0 = constant(zeros(nelem+3*ndof))
S_data = matread("diagnose1+/data.mat")["V"]

ntest = 100
max_iter = 20
dist_test = zeros(ntest)
loss_test = zeros(ntest)



for t in 1:ntest
    @info t
# begin of FOR LOOP
# generate one set of NN weights and biases

    n = 901 
    θ = Variable(randn(n))
    d = 4.0 * rand() + 1.0 # for diagnose1+, sample distance between 1 and 5
    θ = θ / norm(θ) * d
    k_chip = k_nn(x, y, θ)

    S = solve_navier_stokes(S0, NT, k_chip)
    S_computed = S[end, :]

    loss =  mean((S_computed .- S_data)^2)
    loss = loss * 1e10

    # ---------------------------------------------------
    _loss = Float64[]
    cb = (vs, iter, loss)->begin 
        _loss
        push!(_loss, loss)
        printstyled("[#t $t] [#iter $iter] loss=$loss\n", color=:green)
        if mod(iter, 5)==1
            matwrite(string("diagnose1+/test$t","_loss$iter.mat"), Dict("iter"=>iter,"loss"=>_loss, "k_chip"=>vs[1], "d"=>d[1]))
        end
    end

    sess = Session(); init(sess)
    loss = BFGS!(sess, loss, max_iter; vars = [k_chip], callback = cb)

    # record distance d; loss after a fixed number of iterations (converge or not?)
    dist_test[t] = d
    loss_test[t] = loss[end, 1]

    matwrite("diagnose1+/summary$t.mat", Dict("dist"=>dist_test,"loss"=>loss_test))
# end of FOR LOOP
end

# ---------------------------------------------------
