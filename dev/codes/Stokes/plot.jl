using Revise
using PoreFlow
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
mmesh = Mesh(mmesh.nodes * 10, mmesh.elems, -1, 2)


function f(x, y)
    0.0001*(10/(1+x^2) + x * y + 10*y^2)
end

E = eval_f_on_gauss_pts(f, mmesh)

E2 = matread("fenics/bwd2-601.mat")["nu"]
E3 = matread("fenics/bwd3-601.mat")["nu"]
E4 = matread("fenics/bwd4-481.mat")["nu"]

loss2 = matread("fenics/bwd2-601.mat")["loss"]
loss3 = matread("fenics/bwd3-601.mat")["loss"]
loss4 = matread("fenics/bwd4-481.mat")["loss"]

close("all")
figure(figsize=(30, 18))
subplot(331)
title("Exact")
visualize_scalar_on_gauss_points(E, mmesh)
gca().set_rasterized(true)
subplot(332)
title("DNN")
visualize_scalar_on_gauss_points(E2, mmesh)
xlabel("")
subplot(333)
title("Error")
visualize_scalar_on_gauss_points(abs.(E-E2), mmesh)
xlabel("")
subplot(334)
semilogy(loss2, label="DNN")
semilogy(loss3, label="Discretization (Gauss Points)")
semilogy(loss4, label="Discretization (Piecewise Constant)")
legend(loc="upper left", bbox_to_anchor=(0.1, -0.3))
xlabel("Iterations")
ylabel("Loss")
grid("on")
grid(b=true, which="minor")
plt.minorticks_on()
subplot(335)
title("Discretization (Gauss Points)")
visualize_scalar_on_gauss_points(E3, mmesh)
xlabel("")
subplot(336)
title("Error")
visualize_scalar_on_gauss_points(abs.(E-E3), mmesh)
xlabel("")
subplot(338)
title("Discretization (Piecewise Constant)")
visualize_scalar_on_gauss_points(E4, mmesh)
subplot(339)
title("Error")
visualize_scalar_on_gauss_points(abs.(E-E4), mmesh)
savefig("Stokes.png")
savefig("Stokes.pdf")

close("all")
visualize_mesh(mmesh)
tight_layout()
savefig("StokesMesh.png")
savefig("StokesMesh.pdf")