using Revise
using ADCME
using PyCall
using PyPlot
using Random
using Distributions
using PoreFlow
using MAT
using Random; Random.seed!(233)
np = pyimport("numpy")
include("common.jl")

sampler = :sample_poisson_mixture_model_
m = 15
noise_level = 0.1
solver = :solve_poisson

#------------------------------------------------------------------------------------------
ADCME.options.sparse.auto_reorder = false
n = m
h = 0.1
obs_id = rand(interior_node("all", m, n, h), 10)
bdnode = bcnode("all", m, n, h)

#------------------------------------------------------------------------------------------
"""
Example 

sol = solve_poisson(ones(2), obs=true)
sess = Session(); init(sess)
@time sol_ = run(sess, sol)
"""
function solve_poisson(c; jac=false, obs = false)
    c = constant(c)
    xy = gauss_nodes(m, n, h)
    κ = exp(c[1] + c[2] * xy[:,1])
    κ = compute_space_varying_tangent_elasticity_matrix(κ, m, n, h)
    K = compute_fem_stiffness_matrix1(κ, m, n, h)
    K, _ = fem_impose_Dirichlet_boundary_condition1(K, bdnode, m, n, h)
    rhs = compute_fem_source_term1(ones(4m*n), m, n, h)
    sol = K\rhs
    if obs 
        sol = sol[obs_id]
    end
    @info sol 
    if jac
        J = gradients(sol, c)
        J = set_shape(J, (length(sol), 2))
        J = independent(J)
        return sol, J 
    else 
        return sol 
    end
end

function solve_elasticity(c; jac=false, obs = false)
    c = constant(c)
    E, ν = c[1], c[2]
    H = E/(1+ν)/(1-2ν)*tensor([
             1-ν ν 0.0
             ν 1-ν 0.0
             0.0 0.0 (1-2ν)/2
         ])
    K = compute_fem_stiffness_matrix(H, m, n, h)
    bdedge = bcedge("right", m, n, h)
    bdnode = bcedge("left", m, n, h)
    compute_fem_traction_term([ones()])
end


#------------------------------------------------------------------------------------------
# PDE Flow Operator
mutable struct PoissonFlow <: FlowOp
    dim::Int64 
    o::LinearFlow
end
function PoissonFlow(J, sol, c0)
    b = - J * c0 + sol
    PoissonFlow(2, LinearFlow(J, b))
end
forward(fo::PoissonFlow, x) = ADCME.forward(fo.o, x)
backward(fo::PoissonFlow, x) = ADCME.backward(fo.o, x)

#------------------------------------------------------------------------------------------

sampler = eval(sampler)
solver = eval(solver)

sess = Session()
b0, A0 = generate_A_and_b(sess, ones(2))
s0 = sample_observation(solver, sess, 128, sampler)

#------------------------------------------------------------------------------------------
flow1 = [AffineHalfFlow(2, mod(i,2)==1, missing, x->mlp(x, i, 1)) for i = 0:4]
flow2 = [AffineConstantFlow(2, shift=false)]
flows0 = [flow1;flow2]
flows = [PoissonFlow(A0, vec(b0), ones(2));flows0]

prior = ADCME.MultivariateNormalDiag(loc=zeros(2))
model = NormalizingFlowModel(prior, flows)
model0 = NormalizingFlowModel(prior, flows0)

x = placeholder(s0)
zs, prior_logpdf, logdet = model(x)
log_pdf = prior_logpdf + logdet
loss = -sum(log_pdf)

model_samples = rand(model, 128*8)
model_samples0 = rand(model0, 128*8)
sess = Session(); init(sess)

# BFGS!(sess, loss*1e10, 500)

# z = run(sess, model_samples0[end])
# scatter(z[:,1], z[:,2], marker = "+", s=5, label="Estimated")
# if !isnothing(sampler)
#     t0 = sampler(1024)
#     scatter(t0[:,1], t0[:,2], marker = ".", s=5, label="Reference")
# end

# figure()
# x0 = LinRange(0,2,100)|>Array
# x0, y0 = np.meshgrid(x0, x0)
# p = ADCME.pdf(model0, [x0[:] y0[:]])
# p = reshape(run(sess, p), 100, 100)
# pcolormesh(x0, y0, p)

DIR = "solver-$solver-sampler-$sampler-m-$m-noise_level-$noise_level"
if !isdir(DIR)
    mkdir(DIR)
end

for i = 1:100
    BFGS!(sess, loss*1e10, 100)

    close("all")
    z = run(sess, model_samples0[end])
    scatter(z[:,1], z[:,2], marker = "+", s=5, label="Estimated")
    if !isnothing(sampler)
        t0 = sampler(1024)
        scatter(t0[:,1], t0[:,2], marker = ".", s=5, label="Reference")
    end
    legend()
    xlabel("x"); ylabel("y")
    savefig(joinpath(DIR, "scatter$i.png"))

    close("all")
    x0 = LinRange(0,2,100)|>Array
    x0, y0 = np.meshgrid(x0, x0)
    p = ADCME.pdf(model0, [x0[:] y0[:]])
    p = reshape(run(sess, p), 100, 100)
    pcolormesh(x0, y0, p)
    colorbar()
    xlabel("x"); ylabel("y")
    savefig(joinpath(DIR, "pdf$i.png"))

    matwrite(joinpath(DIR, "$i.mat"), Dict("z"=>z, "p"=>p))
end