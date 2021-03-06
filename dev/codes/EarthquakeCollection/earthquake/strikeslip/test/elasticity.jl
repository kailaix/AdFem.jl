# exp(-t)*(2-x)*x*(1-y)
using Revise
using ADCME
using AdFem 
using PyPlot

n = 20
m = 2n
NT = 200
ρ = 0.0
Δt = 1/NT 
h = 1/n
x = zeros((m+1)*(n+1))
y = zeros((m+1)*(n+1))
for i = 1:m+1
    for j = 1:n+1
        idx = (j-1)*(m+1)+i 
        x[idx] = (i-1)*h 
        y[idx] = (j-1)*h 
    end
end
bd = bcnode("all", m, n, h)

u1 = (x,y,t)->exp(-t)*(0.5-x)^2*(2-y)^2
u2 = (x,y,t)->exp(-t)*(0.5-x^2)^2*(2-sin(y))^2

ts = Δt * ones(NT)
dt = αscheme_time(ts, ρ = ρ )
F = zeros(NT, 2(m+1)*(n+1))
for i = 1:NT 
    t = dt[i] 
    f1 = (x,y)->-9.87654320987654*x*(0.5 - x^2)*(2 - sin(y))*exp(-t)*cos(y) + (0.5 - x)^2*(2 - y)^2*exp(-t) - 0.740740740740741*(0.5 - x)^2*exp(-t) - 3.20987654320988*(2 - y)^2*exp(-t)
  	f2 = (x,y)->-2.96296296296296*x^2*(2 - sin(y))^2*exp(-t) + (0.5 - x^2)^2*(2 - sin(y))^2*exp(-t) - 3.20987654320988*(0.5 - x^2)^2*(2 - sin(y))*exp(-t)*sin(y) - 3.20987654320988*(0.5 - x^2)^2*exp(-t)*cos(y)^2 + 1.48148148148148*(0.5 - x^2)*(2 - sin(y))^2*exp(-t) - (0.740740740740741*x - 0.37037037037037)*(2*y - 4)*exp(-t) - (2*x - 1.0)*(1.72839506172839*y - 3.45679012345679)*exp(-t)
    fval1 = eval_f_on_gauss_pts(f1, m, n, h)
  	fval2 = eval_f_on_gauss_pts(f2, m, n, h)
    F[i,:] = compute_fem_source_term(fval1, fval2, m, n, h)
end

abd = zeros(NT, (m+1)*(n+1)*2)
for i = 1:NT 
    t = dt[i]
    abd[i,:] = [(@. u1(x, y, t)); (@. u2(x, y, t))] 
end
abd = constant(abd[:, [bd; bd .+ (m+1)*(n+1)]])

E = 1.0
ν = 0.35
H = E/(1+ν)/(1-2ν)*[
  1-ν ν 0
  ν 1-ν 0
  0 0 (1-2ν)/2
]
M = constant(compute_fem_mass_matrix(m, n, h))
K = constant(compute_fem_stiffness_matrix(H, m, n, h))




a0 = [(@. u1(x, y, 0.0)); (@. u2(x, y, 0.0))] 
u0 = -[(@. u1(x, y, 0.0)); (@. u2(x, y, 0.0))] 
d0 = [(@. u1(x, y, 0.0)); (@. u2(x, y, 0.0))] 


function solver(A, rhs, i)
    A, Abd = fem_impose_Dirichlet_boundary_condition_experimental(A, bd, m, n, h)
    rhs = rhs - Abd * abd[i]
    rhs = scatter_update(rhs, [bd; bd .+ (m+1)*(n+1)], abd[i]) 
    return A\rhs
end
d, u, a = αscheme(M, spzero(2(m+1)*(n+1)), K, F, d0, u0, a0, ts; extsolve=solver, ρ = ρ  )


sess = Session()
d_, u_, a_ = run(sess, [d, u, a])


function plot_traj(idx)
    figure(figsize=(12,3))
    subplot(131)
    plot((0:NT)*Δt, u1.(x[idx], y[idx],(0:NT)*Δt), "b-", label="x-Acceleration")
    plot((0:NT)*Δt, a_[:,idx], "y--", markersize=2)
  	plot((0:NT)*Δt, u2.(x[idx], y[idx],(0:NT)*Δt), "r-", label="y-Acceleration")
    plot((0:NT)*Δt, a_[:,idx+(m+1)*(n+1)], "c--", markersize=2)
    legend()
    xlabel("Time")
    ylabel("Value")

    subplot(132)
    plot((0:NT)*Δt, u1.(x[idx], y[idx],(0:NT)*Δt), "b-", label="x-Displacement")
    plot((0:NT)*Δt, d_[:,idx], "y--", markersize=2)
  	plot((0:NT)*Δt, u2.(x[idx], y[idx],(0:NT)*Δt), "r-", label="y-Displacement")
    plot((0:NT)*Δt, d_[:,idx+(m+1)*(n+1)], "c--", markersize=2)
    legend()
    xlabel("Time")
    ylabel("Value")

    subplot(133)
    plot((0:NT)*Δt, -u1.(x[idx], y[idx],(0:NT)*Δt), "b-", label="x-Velocity")
    plot((0:NT)*Δt, u_[:,idx], "y--", markersize=2)
  	plot((0:NT)*Δt, -u2.(x[idx], y[idx],(0:NT)*Δt), "r-", label="y-Velocity")
    plot((0:NT)*Δt, u_[:,idx+(m+1)*(n+1)], "c--", markersize=2)
    legend()
    xlabel("Time")
    ylabel("Value")

    tight_layout()
end

idx2 = (n÷3)*(m+1) + m÷2
plot_traj(idx2)