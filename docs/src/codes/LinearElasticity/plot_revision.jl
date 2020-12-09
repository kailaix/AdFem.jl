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
    1/(1+x^2) + x * y + y^2
end
    
mmesh = Mesh(50, 50, 1/50, degree=2)

E = eval_f_on_gauss_pts(f, mmesh)

E2 = matread("fenics/data2-inverse10001.mat")["E"]
E3 = matread("pixel/data2-inverse981.mat")["E"]


close("all")
figure(figsize=(30, 8))
subplot(131)
title("Exact")
visualize_scalar_on_gauss_points(E, mmesh)
gca().set_rasterized(true)
subplot(132)
title("Difference (DNN)")
visualize_scalar_on_gauss_points(abs.(E-E2), mmesh)
xlabel("x")
subplot(133)
title("Difference (Discretization on Gauss Points)")
visualize_scalar_on_gauss_points(abs.(E-E3), mmesh)
xlabel("x")
tight_layout()
savefig("LinearElasticity.png")
savefig("LinearElasticity.pdf")
