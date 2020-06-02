using Revise
using ADCME
using ADCMEKit
using NNFEM 
using PyPlot
using ProgressMeter 
using Printf
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


figure()
subplot(211)
visualize_scalar_on_scoped_body(domain.state[1:domain.nnodes], zeros(size(domain.state)...), domain)
autoscale(enable=true, axis="both", tight=true)
xticks([])
xlabel("")
title(L"$u_x$")
subplot(212)
visualize_scalar_on_scoped_body(domain.state[domain.nnodes+1:end], zeros(size(domain.state)...), domain)
autoscale(enable=true, axis="both", tight=true)
title(L"$u_y$")
savefig("figures/dippingfault_disp_elasticity.png")

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

d, v, a, σ, ϵ = ViscoelasticitySolver(
  globaldata, domain, d0, v0, a0, σ0, ϵ0, Δt, NT, μ, λ, η, Fext, ubd, abd
)

sess = Session(); init(sess)

d_, σ_ = run(sess, [d, σ])
domain.history["state"] = []
domain.history["stress"] = []
for i = 1:NT+1
  push!(domain.history["state"], d_[i,:])
  push!(domain.history["stress"], σ_[i,:,:])
end

#################### visulalize results ####################

## get surface slip
slip_idx = findall(domain.nodes[:,2] .≈ 0.0)
coords = domain.nodes[slip_idx,1]
ii = sortperm(coords)
coords = coords[ii]
x_ = d_[:, slip_idx[ii]]
y_ = d_[:, slip_idx[ii] .+ domain.nnodes]

x0 = d0[slip_idx[ii]]
y0 = d0[slip_idx[ii] .+ domain.nnodes]

figure(figsize=(6,3))
subplot(211)
plot(coords, (x_' .- x0)[:,1:20:end])
title(L"u_x - u_{x0}")
autoscale(enable=true, axis="x", tight=true)
xticks([])
subplot(212)
plot(coords, x_[1:20:end, :]')
title(L"u_x")
autoscale(enable=true, axis="x", tight=true)
tight_layout()
xlabel("x")
savefig("figures/dippingfault_ux_viscosity.png")

figure(figsize=(6,3))
subplot(211)
plot(coords, -(y_' .- y0)[:,1:20:end])
title(L"u_y - u_{y0}")
autoscale(enable=true, axis="x", tight=true)
xticks([])
subplot(212)
plot(coords, -y_[1:20:end, :]')
title(L"u_y")
autoscale(enable=true, axis="x", tight=true)
tight_layout()
xlabel("x")
savefig("figures/dippingfault_uy_viscosity.png")


figure(figsize=(5,3))
subplot(211)
pl1, = plot(coords, x0, "o-", markersize = 3)
autoscale(enable=true, axis="x", tight=true)
t = title("time = 0")
xticks([])
ylabel(L"u_x")
subplot(212)
pl2, = plot(coords, y0, "o-", markersize = 3)
autoscale(enable=true, axis="x", tight=true)
xlabel("x")
ylabel(L"u_y")
tight_layout()
function update(i)
  pl1.set_data(coords, x_[i, :])
  pl2.set_data(coords, -y_[i, :])
  t.set_text("time = $(round(i*Δt, digits=1))")
end
p = animate(update, collect(1:2:NT))
saveanim(p, "figures/dippingfault_viscosity.gif")


## get surface stress
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
plot(s[1:20:end, :]')
title("von Mises stress")
savefig("figures/dippingfault_vonMises_stress.png")

figure()
visualize_boundary(domain)
# visualize_mesh(domain)
visualize_scalar_on_scoped_body(η[1:9:end], zeros(domain.nnodes*2), domain)
gca().invert_yaxis()
# autoscale(enable=true, axis="both", tight=true)
scatter(domain.nodes[slip_idx,1], domain.nodes[slip_idx,2], color="m", marker=".", label="Tractrion Free")
legend()
title(L"Viscosity ($\eta$) model")
savefig("figures/dippingfault_mesh.png")



# figure()
# p = visualize_scalar_on_scoped_body(σ_[:, 1:domain.nnodes, 2], d_, domain, scale_factor=1.0)
# # p = visualize_von_mises_stress_on_scoped_body(d_, domain, scale_factor=10.0)
# # p = visualize_total_deformation_on_scoped_body(d_, domain, scale_factor=10.0)
# saveanim(p, "ad_solver.gif")
# close("all")

