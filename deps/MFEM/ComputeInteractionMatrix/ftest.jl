using LinearAlgebra
using PoreFlow
using DelimitedFiles

A = readdlm("fenics/A.txt")
A1 = readdlm("fenics/B.txt")

A = [A A1]
mesh = Mesh(8, 8, 1. /8)
B = compute_interaction_matrix(mesh)

@show norm(A - B)
