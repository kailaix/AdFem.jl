
using Revise
using AdFem
using ADCME
using Random
using Mamba
using ProgressMeter
using PyPlot 
using DelimitedFiles

n = 10
m = 15
h = 0.1

σ0 = 0.1
N = 500

if length(ARGS)==2
    σ0 = parse(Float64, ARGS[1])
    N = parse(Int64, ARGS[2])
end
DIR = "sigma$(σ0)-$N-$(randstring(10))"
if !isdir(DIR)
    mkdir(DIR)
end


Random.seed!(233)
obs_id = rand(interior_node("all", m, n, h), 10)
obs_id = [obs_id; obs_id .+ (m+1)*(n+1)]
function solve_elasticity(c)
    c = constant(c)
    E, ν = c[1], c[2]
    H = 1e6*E/(1+ν)/(1-2ν)*tensor([
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

sol = solve_elasticity([2.0;0.35])
sess = Session(); init(sess)
sol_ = run(sess, sol)
# # visualize_displacement(sol_*1e3, m, n, h)

# error()
function invsigma(x)
    return log(x/(1-x))
end

c = placeholder(rand(2))
sol = solve_elasticity(c)


function visualize(sim)
    close("all")
    figure(figsize=(10, 5))
    subplot(121)
    title("E")
    p = hist(sim[:,1], bins=50, density=true)
    PyPlot.plot(ones(100)*2.0, LinRange(0, maximum(p[1]), 100), "--")
    subplot(122)
    title("\$\\mu\$")
    p = hist(sim[:,2], bins=50, density=true)
    PyPlot.plot(ones(100)*0.35, LinRange(0, maximum(p[1]), 100), "--")
    savefig("$DIR/hist.png")
    writedlm("$DIR/data.txt", sim)
end

#------------------------------------------------------------------------------------------
# RWMC
function logf(x)
    b0 = sigmoid(x[1])*10
    b1 = sigmoid(x[2])*0.499
    r = sol_ - run(sess, sol, c=>[b0;b1])
    -sum(r.^2)/(2 * σ0^2) 
end
sess = Session(); init(sess)

burnin = div(N, 5)
sim = Chains(N, 2, names=["E", "mu"])
θ = RWMVariate([invsigma(2.0/10),invsigma(0.35/0.499)], [0.1,0.1], logf, proposal = SymUniform)

@showprogress for i = 1:N 
    sample!(θ)
    sim[i, :, 1] = [ sigmoid(θ[1])*10; sigmoid(θ[2])*0.499]
end
visualize(sim.value[burnin+1:end,:,1])



# #------------------------------------------------------------------------------------------
# # HMC

# r = sol_ - sol
# lf = -sum(r^2)/(2σ0^2)
# lfg = gradients(lf, c)


# function logfgrad(x)
#     b0 = sigmoid(x[1])*10
#     b1 = sigmoid(x[2])*0.499
#     r, dr = run(sess, [lf,lfg], c=>[b0;b1])
#     r, dr
# end
# sess = Session(); init(sess)
# N = 500
# burnin = 100
# sim = Chains(N, 2, names=["E", "mu"])

# ε = 0.1
# L = 50
# θ = HMCVariate([invsigma(2.0/10),invsigma(0.35/0.499)], 
#     ε, L, logfgrad)

# @showprogress for i = 1:N 
#     sample!(θ)
#     sim[i, :, 1] = [ sigmoid(θ[1])*10; sigmoid(θ[2])*0.499]
# end

# describe(sim)