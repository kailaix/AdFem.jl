using LinearAlgebra
using AdFem
using DelimitedFiles

A = readdlm("fenics/A.txt")
A1 = readdlm("fenics/B.txt")

A = [A A1]
mesh = Mesh(8, 8, 1. /8, degree=1)
B = compute_interaction_matrix(mesh)

@show norm(A - B)


mesh = Mesh(8, 8, 1. / 8, degree=2)

A = readdlm("fenics/A2.txt")
A1 = readdlm("fenics/B2.txt")
E = Int64.(readdlm("fenics/edges.txt"))
Edof = get_edge_dof(E, mesh)
A = [A A1]
B = Array(compute_interaction_matrix(mesh))

B[:, mesh.nnode + 1:mesh.ndof] = B[:, mesh.nnode + 1:mesh.ndof][:, Edof]
B[:, mesh.ndof + mesh.nnode + 1:2mesh.ndof] = B[:, mesh.ndof + mesh.nnode + 1:2mesh.ndof][:, Edof]
@show norm(A - B)
