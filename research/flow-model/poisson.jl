using Revise
using ADCME
using PyCall
using PyPlot
using Random
using Distributions
using PoreFlow
using Random; Random.seed!(233)
include("common.jl")

adaptive = false
sampler = :sample_poisson_mixture_model_
m = 5

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
        return sol, J 
    else 
        return sol 
    end
end

"""
Example:
sol = solve_poisson_batch(ones(128,2))
sess = Session(); init(sess)
@time sol_, J_ = run(sess, sol)
visualize_scalar_on_fem_points(sol_[1,:], m, n, h)
"""
function solve_poisson_batch(c; jac = false, obs = false)
    c = constant(c)
    if !jac 
        sol = map(x->solve_poisson(x; jac=false, obs=obs), c)
    else 
        sol, J = tf.map_fn(x->solve_poisson(x; jac=true, obs=obs), c, dtype=(tf.float64, tf.float64), back_prop=false)
    end
end

function generate_A_and_b(sess, c0)
    sol, A = solve_poisson(c0, jac=true, obs=true)
    sol, A = run(sess, [sol, A])
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

mutable struct AdaptivePoissonFlow <: FlowOp
    dim::Int64 
    o::LinearFlow
end
function AdaptivePoissonFlow(J, sol, c)
    sol, J = solve_poisson(c; jac = true, obs = true)
    b = - J * c + sol
    AdaptivePoissonFlow(2, LinearFlow(J, b))
end
forward(fo::AdaptivePoissonFlow, x) = ADCME.forward(fo.o, x)
backward(fo::AdaptivePoissonFlow, x) = ADCME.backward(fo.o, x)


#------------------------------------------------------------------------------------------
# Observation Sampler
function sample_poisson_mixture_model_(n)
    d = MixtureModel(MvNormal[
        MvNormal([0.0;2.0], [0.5 0.0;0.0 0.5]),
        MvNormal([-2;-2.0], [0.5 0.0;0.0 0.5]),
        MvNormal([2;-2.0], [0.5 0.0;0.0 0.5])])
    v = Array(rand(d, 10000)')[:,1:2]
    v = v * 0.1 .+ [1.0 1.0]
end

function sample_moons_(n)
    v = sample_moons(n)
    v = v * 0.1 .+ [1.0 1.0]
end


function sample_dirichlet_(n)
    v = sample_dirichlet(n)
    v = v * 0.1 .+ [1.0 1.0]
end

function sample_observation(sess, n, sampler)
    v = sampler(n)
    sol = solve_poisson_batch(v, obs=true)
    run(sess, sol)
end


sampler = eval(sampler)

#------------------------------------------------------------------------------------------
sess = Session()
b0, A0 = generate_A_and_b(sess, ones(2))
s0 = sample_observation(sess, 128, sampler)

#------------------------------------------------------------------------------------------
c0 = ones(2)
if adaptive 
    c0 = Variable(zeros(2))
end
PDEFlow = adaptive ? AdaptivePoissonFlow : PoissonFlow
if PDEFlow==AdaptivePoissonFlow
    @info "Using adaptive Flow Operator"
end

flow1 = [AffineHalfFlow(2, mod(i,2)==1, missing, x->mlp(x, i, 1)) for i = 0:4]
flow2 = [AffineConstantFlow(2, shift=false)]
flows0 = [flow1;flow2]
flows = [PDEFlow(A0, vec(b0), c0);flows0]

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

BFGS!(sess, loss*1e10, 500)
run(sess, c0)

z = run(sess, model_samples0[end])
scatter(z[:,1], z[:,2], marker = "+", s=5, label="Estimated")
t0 = sampler(1024)
scatter(t0[:,1], t0[:,2], marker = ".", s=5, label="Reference")