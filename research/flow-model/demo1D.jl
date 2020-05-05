using Revise
using ADCME
using PyPlot
using PoreFlow
using Distributions
using ProgressMeter
using DelimitedFiles
reset_default_graph()

#-----------------------------------------------------------------------------
# hyperparameters

σ0 = 0.01
dim_z = 10
batch_size = 64
model = "beta"

if length(ARGS)==2
    global model = ARGS[1]
    global σ0 = parse(Float64, ARGS[2])
    # global dim_z = parse(Int64, ARGS[3])
    # global batch_size = parse(Int64, ARGS[4])
end

# default distribution for the hidden parameter
x0 = LinRange(0,1.,100)

if model=="beta"
    global beta = Beta(5,2)
elseif model=="arcsin"
    global beta = Arcsine(0,1)
elseif model == "lognormal"
    global beta = LogNormal() 
elseif model == "mixture"
    global beta = MixtureModel(Normal[
        Normal(0.3, 0.1),
        Normal(0.7, 0.1)], [0.3, 0.7])
    global x0 = LinRange(0, 100.,10000.)
end 
# beta = Arcsine(0,1)
# beta = LogNormal() # use LinRange(0,100.,10000)


y0 = pdf.(beta, x0)
DIR = "demo1D-$(σ0)-$dim_z-$batch_size-$model"

rm(DIR, recursive=true, force=true)
mkdir(DIR)


#-----------------------------------------------------------------------------
# Solving a Poisson's equation 
# ∇⋅(exp(1+cx)∇u) = 1
# u = 0 on boundary 

function poisson1d(c)
    m = 20
    n = 10
    h = 0.1
    bdnode = bcnode("all", m, n, h)
    c = constant(c)
    xy = gauss_nodes(m, n, h)
    κ = exp(1.0 + c * xy[:,1])
    κ = compute_space_varying_tangent_elasticity_matrix(κ, m, n, h)
    K = compute_fem_stiffness_matrix1(κ, m, n, h)
    K, _ = fem_impose_Dirichlet_boundary_condition1(K, bdnode, m, n, h)
    rhs = compute_fem_source_term1(ones(4m*n), m, n, h)
    rhs[bdnode] .= 0.0
    sol = K\rhs
    S = reshape(sol, (n+1, m+1))
    S[n÷2+1, m÷2+1]
end

function poisson1ds(c)
    c = constant(c)
    s = map(poisson1d, c)
    # c^2
end

#-----------------------------------------------------------------------------
# autoencoder 
function encoder(x)
    ae(reshape(x, :, 1), [20,20,20,2dim_z], "encoder")
end

function decoder(s)
    σ, μ = s[:,1:dim_z], s[:, dim_z+1:end]
    out = tf.random_normal([size(s,1);dim_z], dtype=tf.float64) .* σ + μ 
    out = ae(out, [20,20,20,1], "decoder") |> abs
    out, σ, μ
end


#-----------------------------------------------------------------------------
# construct an autoencoder

x = placeholder(Float64, shape=[batch_size, 1])
z = encoder(x)
θ, σ, μ = decoder(z)

y = reshape(poisson1ds(θ[:]), (-1,1))
loss_ = sum((y-x)^2/2σ0^2, dims=2) + 1/2 * sum(σ^2 - log(1e-6 + σ^2) + μ^2, dims=2)
loss = mean(loss_)

θtrue = placeholder(Float64, shape=batch_size)
xdata = poisson1ds(θtrue)
function generate_data()
    noise = rand(beta, batch_size)
    reshape(run(sess, xdata, θtrue=>noise), :, 1)
end


opt = AdamOptimizer().minimize(loss)
sess = Session(); init(sess)


losses = Float64[]
for i = 1:10000
    l,_ = run(sess, [loss, opt], x=>generate_data())
    push!(losses, l)
    if mod(i, 10)==1
        θ0 = []
        @showprogress for j = 1:20
            t = run(sess, θ, x=>generate_data())
            push!(θ0, t)
        end
        θ0 = Float64.(vcat(θ0...))
        close("all")
        hist(θ0, bins=20, density=true)
        plot(x0, y0, c="g", label="Reference")
        # xlim(0,1)
        savefig("$DIR/hist$i.png")
        writedlm("$DIR/theta$i.txt", θ0)
        writedlm("$DIR/loss$i.txt", losses)
        # close("all")
        # semilogy(losses)
        # savefig("$DIR/loss.png")
    end
    println("iteration $i, loss = $l")
end
