using Revise
using PoreFlow

mesh = Mesh(8,8,1/8)
ρ = ones(get_ngauss(mesh))
B = compute_fem_advection_matrix1(2*ρ, 3*ρ, mesh)

ρ = constant(ones(get_ngauss(mesh)))
B2 = compute_fem_advection_matrix1(2*ρ, 3*ρ, mesh)

sess = Session(); init(sess)
maximum(abs.(run(sess, B2) - B))