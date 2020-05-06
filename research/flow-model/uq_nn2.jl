using Revise
using ADCME
using PyPlot
using PoreFlow


m = 20
n = 20
h = 1/n
bdnode = bcnode("all", m, n, h)
xy = gauss_nodes(m, n, h)
xy_fem = fem_nodes(m, n, h)

not_bdnode = ones(Bool, (m+1)*(n+1))
not_bdnode[bdnode] .= false
not_bdnode = findall(not_bdnode)


function poisson1d(κ)
    κ = compute_space_varying_tangent_elasticity_matrix(κ, m, n, h)
    K = compute_fem_stiffness_matrix1(κ, m, n, h)
    K, _ = fem_impose_Dirichlet_boundary_condition1(K, bdnode, m, n, h)
    rhs = compute_fem_source_term1(ones(4m*n), m, n, h)
    rhs[bdnode] .= 0.0
    sol = K\rhs
end

function κfc(x, θ)
    xc = fem_to_fvm(x, m, n, h)
    xc = reshape(repeat(xc, 1, 4), (-1,))
    @assert length(xc)==4*m*n
    0.1 + abs(squeeze(fc(reshape(xc, (-1,1)), [20,20,20,1], θ)))
end

function param(x, θ)
    xc = fem_to_fvm(x, m, n, h)
    xc = reshape(repeat(xc, 1, 4), (-1,))
    return 0.1 + 1/(1+xc^2) + 100xc^2
end

function poisson1d_nn(fn)    
    θ = missing
    function f(θ, u)
        κ = fn(u, θ)
        K = compute_space_varying_tangent_elasticity_matrix(κ, m, n, h)
        K = compute_fem_stiffness_matrix1(K, m, n, h)
        K, _ = fem_impose_Dirichlet_boundary_condition1(K, bdnode, m, n, h)
        rhs = K * u 
        rhs = rhs[not_bdnode] - compute_fem_source_term1(ones(4m*n), m, n, h)[not_bdnode]
        rhs, gradients(rhs, u)
    end
    if fn==κfc
        θ = Variable(ae_init([1, 20, 20, 20, 1]))
    end
    u = newton_raphson_with_grad(f, constant(zeros((m+1)*(n+1))) ,θ)
    κnn = fn(u, θ)
    return u, κnn, θ 
end

function jacobian(u, θ)
    κ = fn(u, θ)
    K = compute_space_varying_tangent_elasticity_matrix(κ, m, n, h)
    K = compute_fem_stiffness_matrix1(K, m, n, h)
    K, _ = fem_impose_Dirichlet_boundary_condition1(K, bdnode, m, n, h)
    rhs = K * u 
    rhs = rhs[not_bdnode] - compute_fem_source_term1(ones(4m*n), m, n, h)[not_bdnode]
    J = gradients(rhs, u)
    F = gradients(rhs, κ)
    -J\F
end

ADCME.options.newton_raphson.verbose=true
ADCME.options.newton_raphson.rtol = 1e-4
ADCME.options.newton_raphson.tol = 1e-4
y, κ, _ = poisson1d_nn(param)

sess = Session(); init(sess)
SOL = run(sess, y)


umax = maximum(SOL)
xc = LinRange(0, umax, 100)|>Array
u1 = @. 0.1 + 1/(1+xc^2) + 100xc^2
close("all")
plot(xc, u1)
title("K(u)")
savefig("nn2_ku_plot.png")

y, κnn, θ  = poisson1d_nn(κfc)


using Random; Random.seed!(233)
idx = fem_randidx(50, m, n, h)
obs = y[idx]
OBS = SOL[idx]
loss = sum((obs-OBS)^2)

init(sess)
BFGS!(sess, loss, 200)


close("all")
figure(figsize=(10,4))
subplot(121)
visualize_scalar_on_fem_points(SOL, m, n, h)
subplot(122)
visualize_scalar_on_fem_points(run(sess, y), m, n, h)
plot(xy_fem[idx,1], xy_fem[idx,2], "o", c="red", label="Observation")
legend()
savefig("nn2_sol_compare.png")


close("all")
figure(figsize=(10,4))
subplot(121)
visualize_scalar_on_gauss_points(run(sess, κ), m, n, h)
title("Exact \$K(x, y)\$")
subplot(122)
visualize_scalar_on_gauss_points(run(sess, κnn), m, n, h)
title("Estimated \$K(x, y)\$")
savefig("nn2_sol_compare.png")

u0 = constant(LinRange(0,umax,100)|>Array )
z = 1 + abs(squeeze(fc(reshape(u0, (-1,1)), [20,20,20,1], θ)))
plot(run(sess, u0), run(sess, z), "--")
plot(xc, u1, "-")
savefig("nn2_ucurve.png")


H = jacobian(y, θ)
H = run(sess, H)[idx,:]
y = OBS 
hs = run(sess, obs)
s = run(sess, κnn)

R = (1e-2)^2*diagm(0=>ones(length(obs)))
Q = (1)^2*diagm(0=>ones(length(κnn)))
μ, Σ = uqnlin(y, hs, H, R, s, Q)

error()
σ = diag(Σ)
figure(figsize=(10,4))
subplot(121)
visualize_scalar_on_gauss_points(σ, m, n, h)
title("Standard Deviation")
subplot(122)
visualize_scalar_on_gauss_points(abs.(run(sess, κnn)-κ), m, n, h)
title("Absolute Error")
