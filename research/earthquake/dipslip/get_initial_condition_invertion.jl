using Revise
using ADCME
using ADCMEKit
using NNFEM 
using PyPlot
using ProgressMeter 
using MAT
using ADCMEKit
using Clustering



close("all")
include("load_domain_function.jl")

NT = 100
Δt = 30/NT
# domain = load_crack_domain()
domain = load_crack_domain(mesh="crack_wider.msh", c1=(3.5, 0.0), c2=(4.0,0.5))
# domain = load_crack_domain(option="viscoelasticity")

# visualize_mesh(domain)
H = domain.elements[1].mat[1].H
EBC_func = nothing
FBC_func = nothing
body_func = nothing
globaldata = GlobalData(missing, missing, missing, missing, domain.neqs, EBC_func, FBC_func, body_func)
# globaldata, domain = LinearStaticSolver(globaldata, domain)
Fext = compute_external_force(globaldata, domain)
d = LinearStaticSolver(globaldata, domain, domain.state, H, Fext)
sess = Session(); init(sess)
domain.state = run(sess, d)

matwrite("data/dippingfault_viscosity_inversion.mat", Dict(
  "d" => domain.state
)) 
#=
figure()
subplot(211)
visualize_scalar_on_scoped_body(domain.state[1:domain.nnodes], zeros(size(domain.state)...), domain)
subplot(212)
visualize_scalar_on_scoped_body(domain.state[domain.nnodes+1:end], zeros(size(domain.state)...), domain)
=#

strain = getStrain(domain)
stress = getStress(domain)
state = domain.state

# error()

domain = load_crack_domain(mesh="crack_wider.msh", option="mixed", c1=(3.5, 0.0), c2=(4.0,0.5))

ts = GeneralizedAlphaSolverTime(Δt, NT)
ubd, abd = compute_boundary_info(domain, globaldata, ts)
Fext = compute_external_force(domain, globaldata, ts)

d0 = state
v0 = zeros(2domain.nnodes)
a0  = zeros(2domain.nnodes)
# σ0 = zeros(getNGauss(domain),3)
# ϵ0 = zeros(getNGauss(domain),3)
σ0 = stress
ϵ0 = strain

gnodes = getGaussPoints(domain)

kr = kmeans(gnodes', 20)
A = kr.assignments
# for i = 1:20
#   scatter(gnodes[A .== i,1], gnodes[A .== i,2])
# end
# gca().invert_yaxis()

vs = Variable(1.5*ones(20))

ETA = constant(zeros(getNGauss(domain)))
for i = 1:20
  mask = zeros(getNGauss(domain)) 
  mask[A.==i] .= 1.0
  global ETA += vs[i] * mask 
end 
ETA *= 1e10

μ = zeros(getNGauss(domain))
λ = zeros(getNGauss(domain))
η = zeros(getNGauss(domain))
k = 0
for i = 1:domain.neles
  e = domain.elements[i]
  for mat in e.mat
    global k += 1 
    μ[k] = mat.μ
    λ[k] = mat.λ
    η[k] = mat.η 
  end
end

# η = 1e10 + X2 * gnodes[:,1] + X3 * gnodes[:,2]
# η = @. 1e10 *( 2 + abs(gnodes[:,1] - maximum(gnodes[:,1])/2)/maximum(gnodes[:,1]) - (gnodes[:,2])/maximum(gnodes[:,2]))
# η = 1e10 *( 2 .- (gnodes[:,2])/maximum(gnodes[:,2]))
# η = 1e10 *Variable( 1.5 ) * ones(getNGauss(domain))
# η = 1e10 *( Variable(1.5) + Variable(1.5) * (gnodes[:,2])/maximum(gnodes[:,2]))
η = ETA 

d, v, a, σ, ϵ = ViscoelasticitySolver(
  globaldata, domain, d0, v0, a0, σ0, ϵ0, Δt, NT, μ, λ, η, Fext, ubd, abd
)

d_ = matread("data/viscoelasticity_linear.mat")["d"]

loss = sum((d-d_)^2)
sess = Session(); init(sess)

@info run(sess, loss)
BFGS!(sess, loss, 30)
# lineview(sess, pl, loss, [20.0], [log(1e10)])

# d_ = run(sess, d)
# matwrite("data/viscoelasticity_linear.mat", Dict("d"=>d_))

figure()
η_ = run(sess, η[1:9:end])
visualize_scalar_on_scoped_body(η_, zeros(domain.nnodes*2), domain)

figure()
η2 = 1e10 *( 2 .- (gnodes[:,2])/maximum(gnodes[:,2]))
η2 = η2[1:9:end]
visualize_scalar_on_scoped_body(η2, zeros(domain.nnodes*2), domain)

#=
d_, σ_ = run(sess, [d, σ])
domain.history["state"] = []
domain.history["stress"] = []
for i = 1:NT+1
  push!(domain.history["state"], d_[i,:])
  push!(domain.history["stress"], σ_[i,:,:])
end




###################### check surface ########################
idx = []
k = 0
pts = getGaussPoints(domain)
xcoord = Float64[]
for e in domain.elements
    global k 
    if sum(map(x->x≈0.0, e.coords[:,2]))≠2
        k += length(e.weights)
        continue 
    end
    l = argmin(pts[k+1:k+length(e.weights), 2])
    push!(idx, k + l)
    push!(xcoord, pts[k+l, 1])
    k += length(e.weights)
end
ii = sortperm(xcoord)
xcoord = xcoord[ii]
idx = idx[ii]s

σ_surface = σ_[:, idx, :]
s = zeros(size(σ_surface, 1), size(σ_surface, 2))
for i = 1:size(σ_surface, 1)
    for j = 1:size(σ_surface, 2)
        s[i,j] = NNFEM.postprocess_stress(σ_surface[i,j,:], "vonMises")
    end
end

figure(figsize=(8,3))
semilogy(s[1, :], "-", label="x0")
semilogy(s[NT+1, :], "--", label="x")
legend()
title("Stress")
savefig("Stress_test.png")

figure(figsize=(8,3))
semilogy(s[:, 13], "-")
title("Stress")


slip_idx = findall(domain.nodes[:,2] .≈ 0.0)
coords = domain.nodes[slip_idx,1]
ii = sortperm(coords)
coords = coords[ii]
x_ = d_[:, slip_idx[ii]]
y_ = d_[:, slip_idx[ii] .+ domain.nnodes]
# plot(coords, x_)

x0 = d0[slip_idx[ii]]
y0 = d0[slip_idx[ii] .+ domain.nnodes]

figure(figsize=(8,3))
subplot(211)
plot((x_' .- x0)[:,1:10:end])
subplot(212)
plot(x0, label="x")
# plot(x_[1, :] .- x0, "-", label="x0")
# plot(x_[NT+1, :] .- x0, "--", label="x")
legend()

figure(figsize=(8,3))
subplot(211)
S = 1:20:NT
for i = 1:length(S)
  plot(-(y_' .- y0)[:,S[i]],"C$i",  label="$i")
end
legend()

subplot(212)
S = 1:20:NT
for i = 1:length(S)
  plot(-y_'[:,S[i]],"C$i",  label="$i")
end
plot(-y0, label="y")
# plot(y_[1, :] .- y0, "-", label="x0")
# plot(y_[NT+1, :] .- y0, "--", label="x")
# plot(y0, ":", label="static")
legend()
savefig("disp.png")

figure()
plot(y_[:, 10])
# plot(y_[1, :] .- y0, "-", label="x0")
# plot(y_[NT+1, :] .- y0, "--", label="x")
# plot(y0, ":", label="static")
legend()
=#

