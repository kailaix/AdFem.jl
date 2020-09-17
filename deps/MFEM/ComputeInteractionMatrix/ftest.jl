using LinearAlgebra
using PoreFlow
using DelimitedFiles

A = readdlm("fenics/A.txt")

mesh = Mesh(8, 8, 1. /8)
B = compute_interaction_matrix(mesh)

@show norm(A - B)
