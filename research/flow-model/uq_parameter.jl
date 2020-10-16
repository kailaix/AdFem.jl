using ADCME
using PyPlot
using AdFem


Random.seed!(233)
idx = fem_randidx(100, m, n, h)

function poisson1d(c)
    m = 40
    n = 40
    h = 0.1
    bdnode = bcnode("all", m, n, h)
    c = constant(c)
    xy = gauss_nodes(m, n, h)
    κ = exp(c[1] + c[2] * xy[:,1] + c[3]*xy[:,2])
    κ = compute_space_varying_tangent_elasticity_matrix(κ, m, n, h)
    K = compute_fem_stiffness_matrix1(κ, m, n, h)
    K, _ = fem_impose_Dirichlet_boundary_condition1(K, bdnode, m, n, h)
    rhs = compute_fem_source_term1(ones(4m*n), m, n, h)
    rhs[bdnode] .= 0.0
    sol = K\rhs
    sol[idx]
end

c = Variable(rand(3))
y = poisson1d(c)
Γ = gradients(y, c) 
Γ = reshape(Γ, (100, 3))

# generate data 
sess = Session(); init(sess)
run(sess, assign(c, [1.0;2.0;3.0]))
obs = run(sess, y) + 1e-3 * randn(100)

# Inverse modeling 
loss = sum((y - obs)^2)
init(sess)
BFGS!(sess, loss)

y = obs 
H = run(sess, Γ)
R = (2e-3)^2 * diagm(0=>ones(100))
X = run(sess, c)
Q = diagm(0=>ones(3))
m, V = uqlin(y, H, R, X, Q)
plot([1;2;3], [1.;2.;3.], "o", label="Reference")
errorbar([1;2;3],m + run(sess, c), yerr=2diag(V), label="Estimated")
legend()