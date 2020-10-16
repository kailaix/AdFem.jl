using ADCME
using PyPlot
using AdFem
using MAT


m = 20
n = 20
h = 1/n
bdnode = bcnode("all", m, n, h)
xy = gauss_nodes(m, n, h)
xy_fem = fem_nodes(m, n, h)

not_bdnode = ones(Bool, (m+1)*(n+1))
not_bdnode[bdnode] .= false
not_bdnode = findall(not_bdnode)


σ0 = 0.0
prior = 0.01
if length(ARGS)==2
    global σ0 = parse(Float64, ARGS[1])
    global prior = parse(Float64, ARGS[2])
end

DIR = "uq_nn2-$σ0-$prior"
rm(DIR, force=true, recursive=true)
mkdir(DIR)


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
    κ = κfc(u, θ)
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

y, κnn, θ  = poisson1d_nn(κfc)

using Random; Random.seed!(233)
idx = fem_randidx(50, m, n, h)
obs = y[idx]
OBS = SOL[idx] .* ( 1. .+ σ0*randn(length(idx)))
loss = sum((obs-OBS)^2)

init(sess)
BFGS!(sess, loss, 200)


close("all")
figure(figsize=(10,4))
subplot(121)
visualize_scalar_on_fem_points(SOL, m, n, h)
subplot(122)
Yvalue = run(sess, y)
visualize_scalar_on_fem_points(Yvalue, m, n, h)
plot(xy_fem[idx,1], xy_fem[idx,2], "o", c="red", label="Observation")
legend()
savefig("$DIR/sol_compare.png")


close("all")
figure(figsize=(10,4))
subplot(121)
κvalue = run(sess, κ)
visualize_scalar_on_gauss_points(κvalue, m, n, h)
title("Exact \$K(x, y)\$")
subplot(122)
κnnvalue = run(sess, κnn)
visualize_scalar_on_gauss_points(κnnvalue, m, n, h)
title("Estimated \$K(x, y)\$")
savefig("$DIR/K_compare.png")

u0 = constant(LinRange(0,umax,100)|>Array )
z = 0.1 + abs(squeeze(fc(reshape(u0, (-1,1)), [20,20,20,1], θ)))
close("all")
plot(run(sess, u0), run(sess, z), "--")
plot(xc, u1, "-")
savefig("$DIR/ucurve.png")


u_est = run(sess, y)
H = jacobian(y, θ)
H = run(sess, H)[idx,:]
y_ = OBS 
hs = run(sess, obs)
s = run(sess, κnn)

R = (prior)^2*diagm(0=>ones(length(obs)))
Q = (1e-2)^2*diagm(0=>ones(length(κnn)))
μ, Σ = uqnlin(y_, hs, H, R, s, Q)
# error()
σ = diag(Σ)

close("all")
u1 = fem_to_fvm(u_est, m, n, h)
z1 = μ[1:4:end]
σ1 = 2σ[1:4:end]
Is = sortperm(u1)
u1 = u1[Is]
z1 = z1[Is]
σ1 = σ1[Is]


xc = LinRange(0, umax, 100)|>Array
u2 = @. 0.1 + 1/(1+xc^2) + 100xc^2
plot(xc, u2, "-", linewidth=2, label="Referencee")

plot(run(sess, u0), run(sess, z), "--", linewidth=2, label="Estimated")
plot(u1, z1, "--", linewidth=2, label="Posterior Mean")

fill_between(u1, z1-σ1, z1+σ1, alpha=0.3, color="orange", label="Uncertainty Region")
xlim(minimum(u1), maximum(u1))
legend()
xlabel("\$u\$")
ylabel("\$K(u)\$")
savefig("$DIR/errorbar.png")

matwrite("$DIR/data.mat", 
    Dict(
        "SOL"=>SOL,
        "Yvalue"=>Yvalue,
        "kappavalue"=>κvalue,
        "u_est"=>u_est,
        "H"=>H,
        "yobs"=>y_, 
        "hs"=>hs,
        "s"=>s,
        "sigma"=>σ,
        "mu"=>μ
    ))