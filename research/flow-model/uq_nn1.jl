using Revise
using ADCME
using PyPlot
using AdFem


m = 40
n = 40
h = 1/n
bdnode = bcnode("all", m, n, h)
xy = gauss_nodes(m, n, h)
xy_fem = fem_nodes(m, n, h)

function poisson1d(κ)
    κ = compute_space_varying_tangent_elasticity_matrix(κ, m, n, h)
    K = compute_fem_stiffness_matrix1(κ, m, n, h)
    K, _ = fem_impose_Dirichlet_boundary_condition1(K, bdnode, m, n, h)
    rhs = compute_fem_source_term1(ones(4m*n), m, n, h)
    rhs[bdnode] .= 0.0
    sol = K\rhs
end

κ = @. 0.1 + sin(xy[:,1]) + (xy[:,2]-1)^2 * xy[:,1] + log(1+xy[:,2])
y = poisson1d(κ)

sess = Session(); init(sess)
SOL = run(sess, y)


# inverse modeling 
κnn = squeeze(abs(ae(xy, [20,20,20,1])))
y = poisson1d(κnn)

using Random; Random.seed!(233)
idx = fem_randidx(100, m, n, h)
obs = y[idx]
OBS = SOL[idx]
loss = sum((obs-OBS)^2)

init(sess)
BFGS!(sess, loss, 200)

figure(figsize=(10,4))
subplot(121)
visualize_scalar_on_fem_points(SOL, m, n, h)
subplot(122)
visualize_scalar_on_fem_points(run(sess, y), m, n, h)
plot(xy_fem[idx,1], xy_fem[idx,2], "o", c="red", label="Observation")
legend()


figure(figsize=(10,4))
subplot(121)
visualize_scalar_on_gauss_points(κ, m, n, h)
title("Exact \$K(x, y)\$")
subplot(122)
visualize_scalar_on_gauss_points(run(sess, κnn), m, n, h)
title("Estimated \$K(x, y)\$")


H = gradients(obs, κnn) 
H = run(sess, H)
y = OBS 
hs = run(sess, obs)
R = (1e-1)^2*diagm(0=>ones(length(obs)))
s = run(sess, κnn)
Q = (1e-2)^2*diagm(0=>ones(length(κnn)))
μ, Σ = uqnlin(y, hs, H, R, s, Q)

σ = diag(Σ)
figure(figsize=(10,4))
subplot(121)
visualize_scalar_on_gauss_points(σ, m, n, h)
title("Standard Deviation")
subplot(122)
visualize_scalar_on_gauss_points(abs.(run(sess, κnn)-κ), m, n, h)
title("Absolute Error")
