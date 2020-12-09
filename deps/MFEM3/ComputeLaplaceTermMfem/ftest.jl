using Revise
using AdFem
using PyPlot 
using LinearAlgebra
using DelimitedFiles

mmesh = Mesh3(10, 10, 10, 1/10)

u = readdlm("val.txt")[:]
visualize_scalar_on_fem_points(u, mmesh, filename = "laplace.png")
