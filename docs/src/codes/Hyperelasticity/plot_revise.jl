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

mmesh = Mesh(joinpath(PDATA, "twoholes_large.stl"))
mmesh = Mesh(mmesh.nodes * 10, mmesh.elems)


function f(x, y)
    10/(1+x^2) + x * y + 10*y^2
end


E = eval_f_on_gauss_pts(f, mmesh)

E2 = matread("fenics/bwd2-10001.mat")["E"]
E3 = matread("fenics/bwd3-10001.mat")["E"]
E4 = matread("fenics/bwd4-2471.mat")["E"]

loss2 = matread("fenics/bwd2-10001.mat")["loss"]
loss3 = matread("fenics/bwd3-10001.mat")["loss"]
loss4 = matread("fenics/bwd4-2471.mat")["loss"]

close("all")
figure(figsize=(25, 5))
subplot(141)
title("Exact")
visualize_scalar_on_gauss_points(E, mmesh)
gca().set_rasterized(true)
subplot(142)
title("Difference (DNN)")
visualize_scalar_on_gauss_points(abs.(E-E2), mmesh)
xlabel("x")
subplot(143)
title("Difference \n (Discretization on \n Gauss Points)")
visualize_scalar_on_gauss_points(abs.(E-E3), mmesh)
xlabel("x")
subplot(144)
title("Difference \n (Discretization on \n Each Finite Element)")
visualize_scalar_on_gauss_points(abs.(E-E4), mmesh)
xlabel("x")
tight_layout()
savefig("Hyperelasticity.png")
savefig("Hyperelasticity.pdf")