using Revise
using PoreFlow
using ADCME 
using PyPlot 
matplotlib.use("agg")
mesh = Mesh(joinpath(PDATA, "subduction2d.mat"))
visualize_mesh(mesh)
savefig("mesh.png")

