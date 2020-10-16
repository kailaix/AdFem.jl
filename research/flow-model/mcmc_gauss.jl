
using Revise
using AdFem
using ADCME
using Random
using ProgressMeter
using PyPlot 
using DelimitedFiles
using Distributions

np = pyimport("numpy")
n = 10
m = 15
h = 0.1

xy = Dict(
    1=>(1.95,2.05,0.32,0.38),
    2=>(1.8,2.2,0.2,0.45),
    3=>(1.7,2.2,0.18,0.5),
    4=>(1.5,2.3,0.15,0.5),
    5=>(1.2,3.0,0.0,0.5)
)
for (ii,σ0) in enumerate([0.01 0.05 0.1 0.2 0.5])
xmin,xmax, ymin,ymax = xy[ii]

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

c = Variable(0.1*rand(2))
sol = solve_elasticity(c)

loss = sum((sol-sol_)^2)
sess = Session(); init(sess)
BFGS!(sess, loss)

y = sol_
hs = run(sess, sol)
H = run(sess, gradients(sol, c))
R = 0.1^2*diagm(0=>ones(length(sol_)))
s = run(sess, c)
Q = diagm(0=>ones(2))

μ, V = uqnlin(y, hs, H, R, s, Q)
d = Distributions.MvNormal(μ, (V+V')/2)

x0 = LinRange(xmin, xmax, 100)
y0 = LinRange(ymin, ymax, 100)
X, Y = np.meshgrid(x0, y0)
Z = hcat([[pdf(d, [X[i, j];Y[i, j]]) for i = 1:100] for j = 1:100]...)
mcmc = readdlm("simdata/sigma$σ0-100000.txt")

close("all")
contour(X, Y, Z, levels=4)
scatter(mcmc[:,1], mcmc[:,2], s=2, c="red", label="RW-MCMC", alpha=0.5)
legend()
xlabel("E")
ylabel("\$\\nu\$")
xlim(xmin, xmax)
ylim(ymin, ymax)
savefig("sigma$σ0.png")
end