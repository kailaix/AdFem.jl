using Revise
using ADCME
using PyCall
using PyPlot
using Random
using Distributions

reset_default_graph()

# `nmoons` is adapted from https://github.com/wildart/nmoons
function nmoons(::Type{T}, n::Int=100, c::Int=2;
    shuffle::Bool=false, ε::Real=0.1, d::Int = 2,
    translation::Vector{T}=zeros(T, d),
    rotations::Dict{Pair{Int,Int},T} = Dict{Pair{Int,Int},T}(),
    seed::Union{Int,Nothing}=nothing) where {T <: Real}
    rng = seed === nothing ? Random.GLOBAL_RNG : MersenneTwister(Int(seed))
    ssize = floor(Int, n/c)
    ssizes = fill(ssize, c)
    ssizes[end] += n - ssize*c
    @assert sum(ssizes) == n "Incorrect partitioning"
    pi = convert(T, π)
    R(θ) = [cos(θ) -sin(θ); sin(θ) cos(θ)]
    X = zeros(d,0)
    for (i, s) in enumerate(ssizes)
    circ_x = cos.(range(zero(T), pi, length=s)).-1.0
    circ_y = sin.(range(zero(T), pi, length=s))
    C = R(-(i-1)*(2*pi/c)) * hcat(circ_x, circ_y)'
    C = vcat(C, zeros(d-2, s))
    dir = zeros(d)-C[:,end] # translation direction
    X = hcat(X, C .+ dir.*translation)
    end
    y = vcat([fill(i,s) for (i,s) in enumerate(ssizes)]...)
    if shuffle
        idx = randperm(rng, n)
        X, y = X[:, idx], y[idx]
    end
    # Add noise to the dataset
    if ε > 0.0
        X += randn(rng, size(X)).*convert(T,ε/d)
    end
    # Rotate dataset
    for ((i,j),θ) in rotations
        X[[i,j],:] .= R(θ)*view(X,[i,j],:)
    end
    return X, y
end

function sample_moons(n)
    X, _ = nmoons(Float64, n, 2, ε=0.05, d=2, translation=[0.25, -0.25])
    return Array(X')
end

function sample_dirichlet(n)
    d = Dirichlet([1.0;1.0;1.0])
    v = Array(rand(d, 10000)')[:,1:2]
end

function sample_mixture_model(n)
    
    d = MixtureModel(MvNormal[
        MvNormal([0.0;2.0], [0.5 0.0;0.0 0.5]),
        MvNormal([-2;-2.0], [0.5 0.0;0.0 0.5]),
        MvNormal([2;-2.0], [0.5 0.0;0.0 0.5])])
    v = Array(rand(d, 10000)')[:,1:2]
end


A0 = 0.1*[
    1.0 2.0
    -1.0 2.0
    3.0 1.0
]
b0 = 0.1*[1.0 2.0 3.0]

flow1 = [AffineHalfFlow(2, mod(i,2)==1, missing, x->mlp(x, i, 1)) for i = 0:4]
flow2 = [AffineConstantFlow(2, shift=false)]
flows = [flow1;flow2]


flows = [LinearFlow(A0, vec(b0));flows]


prior = ADCME.MultivariateNormalDiag(loc=zeros(2))
model = NormalizingFlowModel(prior, flows)


s0 = sample_mixture_model(128) * A0' .+ b0
x = placeholder(s0)
zs, prior_logpdf, logdet = model(x)
log_pdf = prior_logpdf + logdet
loss = -sum(log_pdf)


model_samples = rand(model, 128*8)

sess = Session(); init(sess)
BFGS!(sess, loss*1e10, 500)

z = run(sess, model_samples[end]) 
x = s0


figure(figsize=(12,3))

subplot(131)
scatter(x[:,1], x[:,2], c="b", s=5, label="data")
scatter(z[:,1], z[:,2],  marker="+", c="r", s=5, label="prior --> posterior")
axis("scaled"); xlabel("x"); ylabel("y")#


subplot(132)
scatter(x[:,1], x[:,3], c="b", s=5, label="data")
scatter(z[:,1], z[:,3], marker="+",  c="r", s=5, label="prior --> posterior")
axis("scaled"); xlabel("x"); ylabel("y")#


subplot(133)
scatter(x[:,3], x[:,2], c="b", s=5, label="data")
scatter(z[:,3], z[:,2], marker="+",  c="r", s=5, label="prior --> posterior")
axis("scaled"); xlabel("x"); ylabel("y")#

figure()

scatter3D(x[:,1], x[:,2], x[:,3], c="b", s=5, label="data")
scatter3D(z[:,1], z[:,2], z[:,3], c="r", s=5, label="prior --> posterior")
xlabel("x"); ylabel("y"); zlabel("z"); 
