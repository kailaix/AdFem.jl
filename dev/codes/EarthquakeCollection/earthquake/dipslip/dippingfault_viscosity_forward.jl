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
include("load_domain_function.jl")

NT = 100
Δt = 30/NT


mesh = "crack_wider.msh"
c1 = (3.5, 0.0)
c2 = (4.0, 0.5)
slip_scale = 5.0 * sqrt(2)

#################### solve static elastic response  ####################
domain = load_crack_domain(mesh=mesh, c1=c1, c2=c2, slip_scale=slip_scale)
# domain = load_crack_domain()

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

domain = load_crack_domain(mesh=mesh, c1=c1, c2=c2, slip_scale=slip_scale, option="mixed")
# domain = load_crack_domain(option="mixed")

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
η = 1e10 *( constant(3.0) - constant(1.5) * (gnodes[:,2])/maximum(gnodes[:,2]))

d, v, a, σ, ϵ = ViscoelasticitySolver(
  globaldata, domain, d0, v0, a0, σ0, ϵ0, Δt, NT, μ, λ, η, Fext, ubd, abd
)

sess = Session(); init(sess)
d_, σ_, η_ = run(sess, [d, σ, η])
matwrite("data/dippingfault_viscosity_inversion.mat", Dict(
  "sigma"=> σ_, 
  "d" => d_,
  "eta" => η_,
  "d0" => d0, 
  "v0" => v0,
  "a0" => a0,
  "epsilon0"=> ϵ0,
  "sigma0" => σ0,
  "NT" => NT, 
  "dt" => Δt 
))
@save "data/dippingfault_viscosity_inversion.jld2" domain