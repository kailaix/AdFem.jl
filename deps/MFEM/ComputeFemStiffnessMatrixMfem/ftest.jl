using ADCME
using LinearAlgebra
using PoreFlow
using DelimitedFiles

A = readdlm("fenics/A.txt")
mesh = Mesh(8, 8, 1/8)
K = zeros(get_ngauss(mesh), 3, 3)
for i = 1:get_ngauss(mesh)
    K[i, :, :] = [1.0 0.0 0.0
                  0.0 1.0 0.0
                  0.0 0.0 0.5]
end
B1 = compute_fem_stiffness_matrix(K, mesh)

K = constant(K)
B = compute_fem_stiffness_matrix(K, mesh)

sess = Session(); init(sess)
B0 = run(sess, B)
B0 = Array(B0)
@info norm(B0-B1)

DOF = zeros(Int64, 2*mesh.ndof)
for i = 1:mesh.ndof
    DOF[2*i-1] = i
    DOF[2*i] = i + mesh.ndof 
end
B0 = B0[DOF, DOF]

@show norm(A - B0)


A = readdlm("fenics/A2.txt")
mesh = Mesh(8, 8, 1.0 / 8,degree=2)
K = zeros(get_ngauss(mesh), 3, 3)
for i = 1:get_ngauss(mesh)
    K[i, :, :] = [1.0 0.0 0.0
                  0.0 1.0 0.0
                  0.0 0.0 0.5]
end
K = constant(K)
B = compute_fem_stiffness_matrix(K, mesh)
sess = Session(); init(sess)
B0 = run(sess, B)
B0 = Array(B0)

E = Int64.(readdlm("fenics/edges.txt"))
Edof = get_edge_dof(E, mesh)
DOF = [1:mesh.nnode; Edof .+ mesh.nnode]
DOF = [DOF; DOF .+ mesh.ndof]
B0 = B0[DOF, DOF]

@show norm(A - B0)