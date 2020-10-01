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

function f(x, y)
    1/(1+x^2) + x * y + y^2
end
    
mmesh = Mesh(50, 50, 1/50, degree=2)

E = eval_f_on_gauss_pts(f, mmesh)

E2 = matread("fenics/data2-inverse10001.mat")["E"]
E3 = matread("fenics/data3-inverse10001.mat")["E"]
loss2 = matread("fenics/data2-inverse10001.mat")["loss"]
loss3 = matread("fenics/data3-inverse10001.mat")["loss"]

close("all")
figure(figsize=(25, 10))
subplot(231)
title("Exact")
visualize_scalar_on_gauss_points(E, mmesh)
gca().set_rasterized(true)
subplot(232)
title("DNN")
visualize_scalar_on_gauss_points(E2, mmesh)
xlabel("")
subplot(233)
title("Error")
visualize_scalar_on_gauss_points(abs.(E-E2), mmesh)
xlabel("")
subplot(234)
semilogy(loss2, label="DNN")
semilogy(loss3, label="Discrete")
legend()
xlabel("Iterations")
ylabel("Loss")
grid("on")
grid(b=true, which="minor")
plt.minorticks_on()
subplot(235)
title("Discrete")
visualize_scalar_on_gauss_points(E3, mmesh)
subplot(236)
title("Error")
visualize_scalar_on_gauss_points(abs.(E-E3), mmesh)
savefig("LinearElasticity.png")
savefig("LinearElasticity.pdf")

close("all")
visualize_mesh(mmesh)
tight_layout()
savefig("LinearElasticityMesh.png")
savefig("LinearElasticityMesh.pdf")