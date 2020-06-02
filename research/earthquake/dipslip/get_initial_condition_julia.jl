using Revise
using ADCME
using ADCMEKit
using NNFEM 
using PyPlot
using ProgressMeter 


close("all")
include("load_domain_function.jl")

NT = 100
Δt = 30/NT
domain = load_crack_domain()


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


figure()
subplot(211)
visualize_scalar_on_scoped_body(domain.state[1:domain.nnodes], zeros(size(domain.state)...), domain)
subplot(212)
visualize_scalar_on_scoped_body(domain.state[domain.nnodes+1:end], zeros(size(domain.state)...), domain)


strain = getStrain(domain)
stress = getStress(domain)
state = domain.state

# error()

domain = load_crack_domain(option="mixed")
# domain = load_crack_domain()

d0 = state[:]
v0 = zeros(2domain.nnodes)
a0  = zeros(2domain.nnodes)
# σ0 = zeros(getNGauss(domain),3)
# ϵ0 = zeros(getNGauss(domain),3)
σ0 = stress
ϵ0 = strain
domain.state = state[:]
p = 1
for i = 1:domain.neles
  for k = 1:length(domain.elements[i].weights)
    domain.elements[i].mat[k].σ0 = stress[p,:]
    domain.elements[i].mat[k].ε0 = strain[p,:]
    # @info stress[p,:] - H * strain[p,:]
    global p += 1
  end
end


# d, v, a, σ, ϵ = ViscoelasticitySolver(
#   globaldata, domain, d0, v0, a0, σ0, ϵ0, Δt, NT, μ, λ, η, Fext, ubd, abd
# )

Dstate = state[:]
globaldata = GlobalData(d0[domain.dof_to_eq], Dstate[domain.dof_to_eq], v0[domain.dof_to_eq], a0[domain.dof_to_eq], domain.neqs, EBC_func, FBC_func, nothing)
assembleMassMatrix!(globaldata, domain)
@showprogress for i = 1:NT
    global globaldata, domain = GeneralizedAlphaSolverStep(globaldata, domain, Δt, maxiterstep=1)
end

# d_, σ_ = run(sess, [d, σ])
# domain.history["state"] = []
# domain.history["stress"] = []
# for i = 1:NT+1
#   push!(domain.history["state"], d_[i,:])
#   push!(domain.history["stress"], σ_[i,:,:])
# end

d_ = hcat(domain.history["state"]...)'|>Array
σ_ = vcat([reshape(x, 1, size(x,1), size(x,2)) for x in domain.history["stress"]]...)
# figure()
# p = visualize_scalar_on_scoped_body(σ_[:, 1:domain.nnodes, 2], d_, domain, scale_factor=1.0)
# # p = visualize_von_mises_stress_on_scoped_body(d_, domain, scale_factor=10.0)
# # p = visualize_total_deformation_on_scoped_body(d_, domain, scale_factor=10.0)
# saveanim(p, "ad_solver.gif")
# close("all")


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
idx = idx[ii]

σ_surface = σ_[:, idx, :]
s = zeros(size(σ_surface, 1), size(σ_surface, 2))
for i = 1:size(σ_surface, 1)
    for j = 1:size(σ_surface, 2)
        s[i,j] = NNFEM.postprocess_stress(σ_surface[i,j,:], "vonMises")
    end
end

figure(figsize=(8,3))
semilogy(s[1, :], "-", label="x0")
semilogy(s[NT, :], "--", label="x")
legend()
title("Stress")
savefig("Stress_test.png")

figure(figsize=(8,3))
semilogy(s[:, 13], "-")
title("Stress")


# visualize_total_deformation_on_scoped_body(Array(d_), domain; scale_factor=100)

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

