using Revise
using ADCME
using PyCall
using PyPlot
using Random
using Distributions

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

function mlp(x, k, id; nout = 1)
    x = constant(x)
    variable_scope("layer$k$id") do
        x = dense(x, 24, activation="leaky_relu")
        x = dense(x, 24, activation="leaky_relu")
        x = dense(x, 24, activation="leaky_relu")
        x = dense(x, nout)
    end
    return x
end



#------------------------------------------------------------------------------------------
"""
Example:
sol = solve_batch_pde(solve_poisson, ones(128,2))
sess = Session(); init(sess)
@time sol_, J_ = run(sess, sol)
visualize_scalar_on_fem_points(sol_[1,:], m, n, h)
"""
function solve_batch_pde(solver, c; jac = false, obs = false)
    c = constant(c)
    if !jac 
        sol = map(x->solver(x; jac=false, obs=obs), c)
    else 
        sol, J = tf.map_fn(x->solver(x; jac=true, obs=obs), c, dtype=(tf.float64, tf.float64), back_prop=false)
    end
end

function sample_poisson_mixture_model_(n)
    d = MixtureModel(MvNormal[
        MvNormal([0.0;2.0], [0.5 0.0;0.0 0.5]),
        MvNormal([-2;-2.0], [0.5 0.0;0.0 0.5]),
        MvNormal([2;-2.0], [0.5 0.0;0.0 0.5])])
    v = Array(rand(d, 10000)')[:,1:2]
    v = v * noise_level .+ [1.0 1.0]
end


function sample_elasticity_mixture_model_(n)
    d = MixtureModel(MvNormal[
        MvNormal([0.0;2.0], [0.5 0.0;0.0 0.5]),
        MvNormal([-2;-2.0], [0.5 0.0;0.0 0.5]),
        MvNormal([2;-2.0], [0.5 0.0;0.0 0.5])])
    v = Array(rand(d, 10000)')[:,1:2] .* [1.0 0.5] 
    v = v * noise_level * 0.4 .+ [2.0 0.35]
    v[v[:,2] .< 0.0, :] .= 0.0
    v[v[:,2] .> 0.499, :] .= 0.499
    v
end

function sample_moons_(n)
    v = sample_moons(n)
    v = v * noise_level .+ [1.0 1.0]
end


function sample_dirichlet_(n)
    v = sample_dirichlet(n)
    v = v * noise_level .+ [1.0 1.0]
end

function sample_observation(solver, sess, n, sampler)
    @info "Sampler: ", sampler
    if isnothing(sampler)
        sol = run(sess, solver(ones(2), obs=true))
        out = zeros(n, length(sol))
        for i = 1:n 
            out[i,:] = sol .* (1 .+ noise_level*randn(length(sol)))
        end
        return sol 
    end
    v = sampler(n)
    sol = solve_batch_pde(solver, v, obs=true)
    init(sess)
    run(sess, sol)
end

function generate_A_and_b(sess, c0)
    sol, A = solver(c0, jac=true, obs=true)
    init(sess)
    sol, A = run(sess, [sol, A])
end
