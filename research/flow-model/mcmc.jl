
using Revise
using PoreFlow
using ADCME
using Random
using Mamba
using ProgressMeter

n = 10
m = 10
h = 0.1
Random.seed!(233)
obs_id = rand(interior_node("all", m, n, h), 10)
obs_id = [obs_id; obs_id .+ (m+1)*(n+1)]
function solve_elasticity(c)
    c = constant(c)
    E, ν = c[1], c[2]
    H = 1e11*E/(1+ν)/(1-2ν)*tensor([
             1-ν ν 0.0
             ν 1-ν 0.0
             0.0 0.0 (1-2ν)/2
         ])
    K = compute_fem_stiffness_matrix(H, m, n, h)
    bdedge = bcedge("right", m, n, h)
    bdnode = bcnode("left", m, n, h)
    K, _ = fem_impose_Dirichlet_boundary_condition(K, bdnode, m, n, h)
    F = compute_fem_traction_term([zeros(4*m*n) 1e6*ones(4*m*n)], bdedge, m, n, h)
    sol = K\F
    sol = sol[obs_id]
    return sol 
end

# sol = solve_elasticity([2.0;0.35])
# sess = Session(); init(sess)
# sol_ = run(sess, sol)
# # visualize_displacement(sol_*1e3, m, n, h)

# error()
function invsigma(x)
    return log(x/(1-x))
end

c = placeholder(rand(2))
sol = solve_elasticity(c)


#------------------------------------------------------------------------------------------
# RWMC
function logf(x)
    b0 = sigmoid(x[1])*10
    b1 = sigmoid(x[2])*0.499
    r = sol_ - run(sess, sol, c=>[b0;b1])
    -sum(r.^2)/(2 * 1e-10) 
end
sess = Session(); init(sess)
N = 500
burnin = 100
sim = Chains(N, 2, names=["E", "mu"])
θ = RWMVariate([invsigma(2.0/10),invsigma(0.35/0.499)], [0.1,0.1], logf, proposal = SymUniform)

@showprogress for i = 1:N 
    sample!(θ)
    sim[i, :, 1] = [ sigmoid(θ[1])*10; sigmoid(θ[2])*0.499]
end

describe(sim)



#------------------------------------------------------------------------------------------
# HMC

r = sol_ - sol
lf = -sum(r^2)/(2 * 1e-10)
lfg = gradients(lf, c)


function logfgrad(x)
    b0 = sigmoid(x[1])*10
    b1 = sigmoid(x[2])*0.499
    r, dr = run(sess, [lf,lfg], c=>[b0;b1])
    r, dr
end
sess = Session(); init(sess)
N = 500
burnin = 100
sim = Chains(N, 2, names=["E", "mu"])

ε = 0.1
L = 50
θ = HMCVariate([invsigma(2.0/10),invsigma(0.35/0.499)], 
    ε, L, logfgrad)

@showprogress for i = 1:N 
    sample!(θ)
    sim[i, :, 1] = [ sigmoid(θ[1])*10; sigmoid(θ[2])*0.499]
end

describe(sim)