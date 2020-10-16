using MAT
using PyPlot
using LinearAlgebra
using ADCME
using AdFem
using Distributions

sigma = 0.01
q = (1e-1)^2
m = n = 20
h = 0.1


for sigma in [0.0 0.01]

r = (1e-2)^2 

d = matread("simdata/nn2-data-$sigma.mat")
y_, hs, H, s, u_est, kappa = d["yobs"], d["hs"], d["H"], d["s"], d["u_est"], d["kappavalue"]

R = r*diagm(0=>ones(length(hs)))
Q = q*diagm(0=>ones(length(s)))
μ, Σ = uqnlin(y_, hs, H, R, s, Q)
σ = diag(Σ)



umax = maximum(u_est)
xc = LinRange(0, umax, 100)|>Array
u2 = @. 0.1 + 1/(1+xc^2) + 100xc^2
figure()
plot(xc, u2, "-", linewidth=2, label="Referencee")

# k = s[1:4:end]
u1 = fem_to_fvm(u_est, m, n, h)
# Is = sortperm(u1)
# plot(u1[Is], k[Is], "+--")



z1 = μ[1:4:end]
σ1 = 2σ[1:4:end]
u1 = u1[Is]
z1 = z1[Is]
σ1 = σ1[Is]
plot(u1, z1, "--", linewidth=2, label="Posterior Mean")
fill_between(u1, z1-σ1, z1+σ1, alpha=0.3, color="orange", label="Uncertainty Region")

xlim(minimum(u1), maximum(u1))

xlabel("\$u\$")
ylabel("\$K(u)\$")
legend()
savefig("nn2-uq$sigma-1.png")


close("all")
plot(u1, σ1, "+")
xlabel("\$u\$")
ylabel("Standard Deviation")
savefig("nn2-uq$sigma-2.png")

end