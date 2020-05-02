using Revise
using ADCME
using PyCall
using PyPlot
using Random
using Distributions
using PoreFlow
using MAT
using Random; 
np = pyimport("numpy")
include("common.jl")

sampler = :sample_elasticity_spd_mixture_model_
n = 15
noise_level = 0.1
solver = :solve_elasticity_spd

#------------------------------------------------------------------------------------------
ADCME.options.sparse.auto_reorder = false
h = 0.1
if solver==:solve_poisson
    global m= n
elseif solver==:solve_elasticity || solver==:solve_elasticity_spd
    global m = 2n
end

# c is a vector of 9, SPD
function solve_elasticity_spd(c; jac=false, obs = false)
    if isa(c, Array)
        c = constant(c)
    end
    H = 1e11*reshape(c, (3, 3))
    K = compute_fem_stiffness_matrix(H, m, n, h)
    bdedge = bcedge("right", m, n, h)
    bdnode = bcnode("left", m, n, h)
    K, _ = fem_impose_Dirichlet_boundary_condition(K, bdnode, m, n, h)
    F = compute_fem_traction_term([1e6*ones(4*m*n) 1e6*ones(4*m*n)], bdedge, m, n, h)
    sol = K\F

    Random.seed!(233)
    obs_id = rand(interior_node("all", m, n, h), 10)
    
    if obs 
        sol = sol[obs_id]
    end
    @info sol 
    if jac
        J = gradients(sol, c)
        J = set_shape(J, (length(sol), 9))
        J = independent(J)
        return sol, J 
    else 
        return sol 
    end
end


function sample_elasticity_spd_mixture_model_(n)
    out = zeros(n, 9)
    for i = 1:n 
        H = [3.20988  1.7284   0.0     
            1.7284   3.20988  0.0     
            0.0      0.0      0.740741]
        σ = 0.1*[abs(randn()) 0.0 abs.(sample_mixture_gaussian_4(1))][:]
        N = [1+σ[1] 1+σ[2] 0.0
            1+σ[2] 1+σ[3] 0.0
            0.0 0.0 1 + σ[4]]
        out[i,:] = H[:] .*  N[:]
    end
    abs.(out )
end


solver = eval(solver)
sampler = eval(sampler)


H = [3.20988  1.7284   0.0     
    1.7284   3.20988  0.0     
    0.0      0.0      0.740741]
# sol = solve_elasticity_spd(H[:])
# sess = Session(); init(sess)
# sol_ = run(sess, sol)
# # visualize_displacement(sol_, m, n, h)

# # # error()


# # inverse modeling of elasticity
# obs_id = rand(interior_node("all", m, n, h), 10)
# obs_id = [obs_id; obs_id .+ (m+1)*(n+1)]
# c = spd(Variable(diagm(0=>ones(3))))[:]
# sol = solve_elasticity_spd(c)
# loss = sum((sol[obs_id] - sol_[obs_id])^2) * 1e20
# init(sess)
# BFGS!(sess, loss)
# error()


#------------------------------------------------------------------------------------------
# PDE Flow Operator
# transform to (0, infty) and (0, c2)
mutable struct CholFlow <: FlowOp
    dim::Int64 
end
function CholFlow() 
    CholFlow(9)
end
function forward(fo::CholFlow, x) 
    cholesky_factorize_logdet = load_op_and_grad("$(@__DIR__)/../../deps/CholeskyOp/build/libCholeskyOp.dylib", "cholesky_logdet"; multiple=true)
    z, logdet = cholesky_factorize_logdet(x)
    z, logdet
end
function backward(fo::CholFlow, z)
    x = cholesky_outproduct(z)
    return x, nothing
end

mutable struct PoissonFlow <: FlowOp
    dim::Int64 
    o::LinearFlow
end
function PoissonFlow(J, sol, c0)
    b = - J * c0 + sol
    PoissonFlow(-1, LinearFlow(J, b))
end
forward(fo::PoissonFlow, x) = ADCME.forward(fo.o, x)
backward(fo::PoissonFlow, x) = ADCME.backward(fo.o, x)


# #------------------------------------------------------------------------------------------

sess = Session(); init(sess)
b0, A0 = generate_A_and_b(sess, H[:])


# #------------------------------------------------------------------------------------------
# dim = 6
# ADCME.options.training.training = placeholder(true)
# flows = create_transform(dim)
# prior = ADCME.MultivariateNormalDiag(loc=zeros(dim))

# # pretraining 
# s0 = zeros(128, dim)
# for i = 1:128
#     s0[i,:] = [1.;1.;1.;0.;0.;0.]
# end
# model = NormalizingFlowModel(prior, flows)
# x = placeholder(s0)
# zs, prior_logpdf, logdet = model(x)
# log_pdf = prior_logpdf + logdet
# loss = -sum(log_pdf)
# model_samples0 = rand(model, 128*8)

# opt = AdamOptimizer().minimize(loss)
# sess = Session(); init(sess)

# for i = 1:5000
#     loss_, _ = run(sess, [loss, opt])
#     @info i, loss_
# end

# z_init = run(sess, model_samples0[end], ADCME.options.training.training=>false)
# error()
# #------------------------------------------------------------------------------------------

dim = 6
ADCME.options.training.training = placeholder(true)
flows6 = create_transform(dim, use_batch_norm =true)
prior = ADCME.MultivariateNormalDiag(loc=zeros(6))
model6 =  NormalizingFlowModel(prior, flows6)

flows = [CholFlow();flows6...]
model = NormalizingFlowModel(prior, flows)
model_samples0 = rand(model, 128*8)


# pretraining 
s0 = zeros(128, 9)
for i = 1:128
    s0[i,:] = H[:]
end
x = placeholder(s0)
zs, prior_logpdf, logdet = model(x)
log_pdf = prior_logpdf + logdet
loss = -sum(log_pdf)

model_samples0 = rand(model, 128)

lr = placeholder(1e-6)
opt = AdamOptimizer(lr,name="pretrain").minimize(loss)
init(sess)

XX = zeros(128, 9)
for i = 1:128
    XX[i,:] = H[:]
end
for k = 1:10000
    X0 =  XX 
    loss_, _ = run(sess, [loss, opt], feed_dict = Dict(lr=> 1e-6, x=>X0))
    if mod(k,1000)==0
        p = run(sess, model_samples0[end], ADCME.options.training.training=>false)
        close("all")
        for i = 1:3
            for j = 1:3
                subplot(3,3,3*(j-1)+i)
                hist(p[:,3*(j-1)+i],bins=20)
            end
        end
        savefig("test$k.png")
    end
    @info k, loss_
end

# BFGS!(sess, loss)
