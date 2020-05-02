
using Revise
using PoreFlow
using ADCME
using Random
using PyPlot
using Mamba
using ProgressMeter
using DelimitedFiles

reset_default_graph()

n = 10
m = 10
h = 0.1

σ0 = 0.1
if length(ARGS)==1
    global σ0 = parse(Float64, ARGS[1])
end

DIR = "vae-sigma$(σ0)"

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

# c = placeholder(rand(2))
# sol = solve_elasticity(c)
rm(DIR, force=true, recursive=true)
mkdir(DIR)

#------------------------------------------------------------------------------------------
function encoder(x, n_hidden, n_output, rate)
    local μ, σ
    variable_scope("encoder") do 
        y = dense(x, n_hidden, activation = "tanh")
        y = dropout(y, rate, ADCME.options.training.training)
        y = dense(y, n_hidden, activation = "elu")
        y = dropout(y, rate, ADCME.options.training.training)
        y = dense(y, 2n_output)
        μ = y[:, 1:n_output]
        σ = 1e-6 + softplus(y[:,n_output+1:end])
    end
    return μ, σ
end

function decoder(z, n_hidden, n_output, rate)
    local y 
    variable_scope("decoder") do 
        y = dense(z, n_hidden, activation="tanh")
        y = dropout(y, rate, ADCME.options.training.training)
        y = dense(y, n_hidden, activation="elu")
        y = dropout(y, rate, ADCME.options.training.training)
        y = dense(y, n_output)
        y = sigmoid(y) .* [10.0,0.499]
    end
    return y 
end

function autoencoder(xh, x, dim_img, dim_z, n_hidden, rate)
    μ, σ = encoder(xh, n_hidden, dim_z, rate)
    z = μ + σ .* tf.random_normal(size(μ), 0, 1, dtype=tf.float64)
    c = decoder(z, n_hidden, dim_img, rate)

    marginal_likelihood =  sum(-1/(2σ0^2) * (x - map(solve_elasticity, c))^2, dims=2)
    KL_divergence = 0.5 * sum(μ^2 + σ^2 - log(1e-8 + σ^2) - 1, dims=2)

    marginal_likelihood = mean(marginal_likelihood)
    KL_divergence = mean(KL_divergence)

    ELBO = marginal_likelihood - KL_divergence
    loss = -ELBO 
    return c, loss, -marginal_likelihood, KL_divergence
end

n_hidden = 500
rate = 0.1
dim_z = 20
dim_img = 2
batch_size = 32
ADCME.options.training.training = placeholder(true)
SOL = repeat(sol_', batch_size, 1)
x = placeholder( SOL )
xh = x
c, loss, ml, KL_divergence = autoencoder(xh, x, dim_img, dim_z, n_hidden, rate)
opt = AdamOptimizer(1e-5).minimize(loss)

sess = Session(); init(sess)

losses = Float64[]
mls = Float64[]
kls = Float64[]
function visualize(i)
    if i<100
        return
    end
    sim = zeros(batch_size*1000,2)
    @showprogress for i = 1:1000
        sim[batch_size*(i-1)+1:batch_size*i,:] = run(sess, c, feed_dict = Dict(ADCME.options.training.training=>false))
    end

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
    savefig("$DIR/dist$i.png")
    close("all")
    semilogy(losses, label="Total Loss")
    semilogy(mls, label="Negative Marginal Likelihood")
    semilogy(kls, label="KL Divergence")
    legend()
    savefig("$DIR/loss$i.png")

    writedlm("$DIR/data$i.png", sim)
end
# BFGS!(sess, loss)
for i = 1:10000
    run(sess, opt, x=> SOL + σ0 * randn(batch_size, length(sol_)))
    if mod(i,10)==1
        c_, loss_, ml_, kl_ = run(sess, [ c, loss, ml, KL_divergence], ADCME.options.training.training=>false)
        println("iter $i: L_tot = $(loss_), L_likelihood = $(ml_), L_KL = $(kl_), c = $(c_[1,:])")
        if i>1 && loss_ < minimum(losses)
            visualize(i)
        end
        push!(losses, loss_)
        push!(mls, ml_)
        push!(kls, kl_)
    end
end

