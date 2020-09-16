using ADCME
using LinearAlgebra
using PoreFlow
using DelimitedFiles

A = readdlm("fenics/A.txt")
mesh = Mesh(8,8,1/8)
ρ = ones(get_ngauss(mesh))
B = compute_fem_mass_matrix1(ρ, mesh)
sess = Session(); init(sess)
B0 = run(sess, B)
B0 = Array(B0)
@show norm(A - B0)