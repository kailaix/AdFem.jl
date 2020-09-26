using Revise
using PoreFlow
using PyPlot 
using LinearAlgebra


mmesh = Mesh(10, 10, 0.1)
nu = ones(get_ngauss(mmesh))
u = rand(mmesh.ndof)
f = compute_fem_laplace_term1(u, nu, mmesh)
f1 = compute_fem_laplace_term1(constant(u), nu, mmesh)

K = compute_fem_laplace_matrix1(nu, mmesh)
f0 = K * u
@info norm(f0 - f)

sess = Session(); init(sess)
f1 = run(sess, f1)
@info norm(f0-f1)