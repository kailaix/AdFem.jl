using Revise
using AdFem
using PyPlot
using LinearAlgebra
using Statistics
using MAT 
SMALL_SIZE = 20
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

matplotlib.rc("font", size=SMALL_SIZE)          # controls default text sizes
matplotlib.rc("axes", titlesize=SMALL_SIZE)     # fontsize of the axes title
matplotlib.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
matplotlib.rc("xtick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc("ytick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
matplotlib.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

function f(x, y)
    0.0001*(10/(1+x^2) + x * y + 10*y^2)
end




Δt = 0.01
mmesh = Mesh(30, 30, 1/30)

xy = gauss_nodes(mmesh)
x, y = xy[:,1], xy[:, 2]
init_nu = 0.001*(1. .+ 7.  .- 4. *x + 11. *y)
E = eval_f_on_gauss_pts(f, mmesh)

E2 = matread("fenics/bwd2-61.mat")["nu"]
E3 = matread("pixel/bwd3-16.mat")["nu"]

close("all")
figure(figsize=(32, 5))
subplot(141)
title("Exact")
visualize_scalar_on_gauss_points(E, mmesh)
gca().set_rasterized(true)
subplot(142)
title("Difference (DNN)")
visualize_scalar_on_gauss_points(abs.(E-E2), mmesh, vmin=0.0, vmax = 2e-4)
xlabel("x")
subplot(143)
title("Difference (Discretization on Gauss Points)")
visualize_scalar_on_gauss_points(abs.(E-E3), mmesh, vmin=0.0, vmax = 2e-4)
xlabel("x")

subplot(144)
title("Initial Guess for the Discretization Method")
visualize_scalar_on_gauss_points(init_nu, mmesh)
xlabel("x")

tight_layout()
savefig("Burgers.png")
savefig("Burgers.pdf")
