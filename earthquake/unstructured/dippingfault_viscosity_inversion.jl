using Revise
using ADCME
# using ADCMEKit
using NNFEM 
using PyPlot
using ProgressMeter 
using Printf
using MAT
using Clustering

close("all")
include("load_domain_function.jl")

NT = 100
Δt = 20/NT

#################### solve static elastic response  ####################
domain = load_crack_domain()
H = domain.elements[1].mat[1].H
EBC_func = nothing
FBC_func = nothing
body_func = nothing
globaldata = GlobalData(missing, missing, missing, missing, domain.neqs, EBC_func, FBC_func, body_func)
Fext = compute_external_force(globaldata, domain)
d = LinearStaticSolver(globaldata, domain, domain.state, H, Fext)
sess = Session(); init(sess)
domain.state = run(sess, d)

#################### solve viscoelastic relaxation  ####################
strain = getStrain(domain)
stress = getStress(domain)
state = domain.state

domain = load_crack_domain(option="mixed")
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
  
gnodes = getGaussPoints(domain)
η = 1e10 *( constant(1.5) + constant(1.5) * (gnodes[:,2])/maximum(gnodes[:,2]))

d, v, a, σ, ϵ = ViscoelasticitySolver(
  globaldata, domain, d0, v0, a0, σ0, ϵ0, Δt, NT, μ, λ, η, Fext, ubd, abd
)


# domain.history["state"] = []
# domain.history["stress"] = []
# for i = 1:NT+1
#   push!(domain.history["state"], d_[i,:])
#   push!(domain.history["stress"], σ_[i,:,:])
# end

# ############################ Save Training Data ###########################
sess = Session(); init(sess)
d_, σ_, η_ = run(sess, [d, σ, η])
matwrite("data/dippingfault_viscosity_inversion.mat", Dict(
  "sigma"=> σ_, 
  "d" => d_,
  "eta" => η_,
)) 
# error()

#################### Get surface observation ####################

## get surface slip
slip_idx = findall(domain.nodes[:,2] .≈ 0.0)
coords = domain.nodes[slip_idx,1]
ii = sortperm(coords)
sorted_slip_idx = slip_idx[ii]
coords = coords[ii]
# x_ = d_[:, slip_idx[ii]]
# y_ = d_[:, slip_idx[ii] .+ domain.nnodes]

# x0 = d0[slip_idx[ii]]
# y0 = d0[slip_idx[ii] .+ domain.nnodes]

# ## get surface stress
# idx = []
# k = 0
# pts = getGaussPoints(domain)
# xcoord = Float64[]
# for e in domain.elements
#     global k 
#     if sum(map(x->x≈0.0, e.coords[:,2]))≠2
#         k += length(e.weights)
#         continue 
#     end
#     l = argmin(pts[k+1:k+length(e.weights), 2])
#     push!(idx, k + l)
#     push!(xcoord, pts[k+l, 1])
#     k += length(e.weights)
# end
# ii = sortperm(xcoord)
# xcoord = xcoord[ii]
# idx = idx[ii]

# σ_surface = σ_[:, idx, :]
# s = zeros(size(σ_surface, 1), size(σ_surface, 2))
# for i = 1:size(σ_surface, 1)
#     for j = 1:size(σ_surface, 2)
#         s[i,j] = NNFEM.postprocess_stress(σ_surface[i,j,:], "vonMises")
#     end
# end

########################### Inversion ###########################

domain = load_crack_domain()
H = domain.elements[1].mat[1].H
EBC_func = nothing
FBC_func = nothing
body_func = nothing
globaldata = GlobalData(missing, missing, missing, missing, domain.neqs, EBC_func, FBC_func, body_func)
Fext = compute_external_force(globaldata, domain)
d = LinearStaticSolver(globaldata, domain, domain.state, H, Fext)
sess = Session(); init(sess)
domain.state = run(sess, d)

strain = getStrain(domain)
stress = getStress(domain)
state = domain.state

domain = load_crack_domain(option="mixed")

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

# create Variable eta 
gnodes = getGaussPoints(domain)
kr = kmeans(gnodes', 20)
A = kr.assignments

figure()
for i = 1:20
  scatter(gnodes[A .== i,1], gnodes[A .== i,2])
end
gca().invert_yaxis()


vs = Variable(1.5*ones(20))
ETA = constant(zeros(getNGauss(domain)))
for i = 1:20
  mask = zeros(getNGauss(domain)) 
  mask[A.==i] .= 1.0
  global ETA += vs[i] * mask 
end 
ETA *= 1e10
η = ETA 

μ = zeros(getNGauss(domain))
λ = zeros(getNGauss(domain))
k = 0
for i = 1:domain.neles
  e = domain.elements[i]
  for mat in e.mat
    global k += 1 
    μ[k] = mat.μ
    λ[k] = mat.λ
  end
end

d, v, a, σ, ϵ = ViscoelasticitySolver(
  globaldata, domain, d0, v0, a0, σ0, ϵ0, Δt, NT, μ, λ, η, Fext, ubd, abd
)

dat = matread("data/dippingfault_viscosity_inversion.mat")
d_, σ_, η_ = dat["d"], dat["sigma"], dat["eta"]

# xy_id = [sorted_slip_idx; sorted_slip_idx .+ domain.nnodes]
# loss = sum((d[:, xy_id]-d_[:, xy_id])^2)
loss = sum((d-d_)^2)
sess = Session(); init(sess)

@info run(sess, loss)
BFGS!(sess, loss, 20)

figure(figsize=(8,3))
subplot(121)
η_est = run(sess, η[1:9:end])
visualize_scalar_on_scoped_body(η_est, zeros(domain.nnodes*2), domain)
title("Estimate")
subplot(122)
η_ref = η_[1:9:end]
visualize_scalar_on_scoped_body(η_ref, zeros(domain.nnodes*2), domain)
title("Reference")

#################### Plot results ####################

# figure()
# p = visualize_scalar_on_scoped_body(σ_[:, 1:domain.nnodes, 2], d_, domain, scale_factor=1.0)
# # p = visualize_von_mises_stress_on_scoped_body(d_, domain, scale_factor=10.0)
# # p = visualize_total_deformation_on_scoped_body(d_, domain, scale_factor=10.0)
# saveanim(p, "ad_solver.gif")
# close("all")

