using Revise
using ADCME
using PyPlot
using Distributions
reset_default_graph()

# beta = Beta(5,2)
# beta = Arcsine(0,1)
# beta = LogNormal() # use LinRange(0,100.,10000)
beta = MixtureModel(Normal[
    Normal(0.3, 0.1),
    Normal(0.7, 0.1)], [0.3, 0.7])
x0 = LinRange(0,1.,100)
y0 = pdf.(beta, x0)
plot(x0, y0)

σ0 = 0.01
dim_z = 10
batch_size = 64
function encoder(x)
    ae(reshape(x, :, 1), [20,20,20,2dim_z], "encoder")
end

function decoder(s)
    σ, μ = s[:,1:dim_z], s[:, dim_z+1:end]
    out = tf.random_normal([size(s,1);dim_z], dtype=tf.float64) .* σ + μ 
    out = ae(out, [20,20,20,1], "decoder") |> abs
    out, σ, μ
end

x = placeholder(Float64, shape=[batch_size, 1])
z = encoder(x)
θ, σ, μ = decoder(z)

loss_ = sum((θ^2-x)^2/2σ0^2, dims=2) + 1/2 * sum(σ^2 - log(1e-6 + σ^2) + μ^2, dims=2)
loss = mean(loss_)

opt = AdamOptimizer().minimize(loss)
sess = Session(); init(sess)

for i = 1:10000
    run(sess, opt, x=>rand(beta, batch_size, 1).^2)
    if mod(i, 500)==1
        θ0 = []
        for j = 1:500
            global l, t = run(sess, [loss, θ], x=>rand(beta, batch_size, 1).^2)
            push!(θ0, t)
        end
        θ0 = Float64.(vcat(θ0...))
        close("all")
        # θ0 = θ0[0.0 .< θ0 .< 1.0]
        hist(θ0, bins=20, density=true)
        plot(x0, y0, c="g", label="Reference")
        # xlim(0,1)
        println("iteration $i, loss = $l")
    end
end
