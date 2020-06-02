using Revise
using ADCME
using ADCMEKit
using NNFEM 
using PyPlot
using ProgressMeter 
using Printf
using MAT
using Clustering
using JLD2

close("all")

dat = matread("data/dippingfault_viscosity_inversion.mat")
σ_ = dat["sigma"]
d_ = dat["d"]
η_ = dat["eta"]
d0 = dat["d0"]
v0 = dat["v0"]
a0 = dat["a0"]
ϵ0 = dat["epsilon0"]
σ0 = dat["sigma0"]
NT = dat["NT"]
Δt = dat["dt"]

@load "data/dippingfault_viscosity_inversion.jld2" domain
#################### Get surface observation ####################

## get surface slip
slip_idx = findall(domain.nodes[:,2] .≈ 0.0)
coords = domain.nodes[slip_idx,1]
ii = sortperm(coords)
sorted_slip_idx = slip_idx[ii]
coords = coords[ii]

########################### Inversion ###########################
# create Variable eta 
gnodes = getGaussPoints(domain)
using Random; Random.seed!(0)
kr = kmeans(gnodes', 20)
A = kr.assignments

figure()
for i = 1:20
  scatter(gnodes[A .== i,1], gnodes[A .== i,2])
end
gca().invert_yaxis()
autoscale(enable=true, axis="both", tight=true)
title("Patch division for inversion")
ylabel("y")
xlabel("x")
savefig("figures/dipslip-patch.png")


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

globaldata = GlobalData(missing, missing, missing, missing, domain.neqs, nothing, nothing, nothing)
ts = GeneralizedAlphaSolverTime(Δt, NT)
ubd, abd = compute_boundary_info(domain, globaldata, ts)
Fext = compute_external_force(domain, globaldata, ts)

d, v, a, σ, ϵ = ViscoelasticitySolver(
  globaldata, domain, d0, v0, a0, σ0, ϵ0, Δt, NT, μ, λ, η, Fext, ubd, abd
)

dat = matread("data/dippingfault_viscosity_inversion.mat")
d_ = dat["d"]
y_id = [sorted_slip_idx; sorted_slip_idx .+ domain.nnodes]
loss = sum((d[:, y_id]-d_[:, y_id])^2)
# loss = sum((d - d_)^2)
sess = Session(); init(sess)

figure()
cb = (vs, iter, loss)->begin 
    η_est = vs[1]
    if mod(iter, 10) == 0
        clf()
        visualize_scalar_on_scoped_body(η_est, zeros(domain.nnodes*2), domain, vmin=minimum(η_), vmax=maximum(η_))
        title("Iter = $iter")
        autoscale(enable=true, axis="both", tight=true)
        savefig("figures_tmp/dippingfault_viscosity_inversion_$(lpad(iter,5,"0")).png",bbox_size="tight")
    end
    printstyled("[#iter $iter] loss = $loss\n", color = :green)
end

@show run(sess, loss)
BFGS!(sess, loss, 200, callback=cb, vars=[η[1:9:end]])

figure()
η_est = run(sess, η[1:9:end])
visualize_scalar_on_scoped_body(η_est, zeros(domain.nnodes*2), domain, vmin=minimum(η_), vmax=maximum(η_))
autoscale(enable=true, axis="both", tight=true)
title("Inverted viscosity result")
savefig("figures/dipslip-inv-linear-visco-model.png")

figure()
η_ref = η_[1:9:end]
visualize_scalar_on_scoped_body(η_ref, zeros(domain.nnodes*2), domain)
autoscale(enable=true, axis="both", tight=true)
title(L"True viscosity ($\eta$) model")
savefig("figures/dipslip-linear-visco-model.png")