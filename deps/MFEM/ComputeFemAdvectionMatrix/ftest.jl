using ADCME
using LinearAlgebra
using PoreFlow
using DelimitedFiles

A = readdlm("fenics/A.txt")
mesh = Mesh(8,8,1/8)
ρ = ones(get_ngauss(mesh))
B = compute_fem_advection_matrix1(2*ρ, 3*ρ, mesh)
sess = Session(); init(sess)
B0 = run(sess, B)
B0 = Array(B0)
@show norm(A - B0)


mesh = Mesh(8, 8, 1.0/8, degree=2)
A = readdlm("fenics/A2.txt")
E = Int64.(readdlm("fenics/edges.txt"))
Edof = get_edge_dof(E, mesh)

ρ = ones(get_ngauss(mesh))
B = compute_fem_advection_matrix1(2*ρ, 3*ρ, mesh)
sess = Session(); init(sess)
B0 = run(sess, B)
B0 = Array(B0)
E = Int64.(readdlm("fenics/edges.txt"))
Edof = get_edge_dof(E, mesh)
DOF = [1:mesh.nnode; Edof .+ mesh.nnode]
B0 = B0[DOF, DOF]

@show norm(A - B0)