using Revise
using AdFem

mesh = Mesh3(10,10,10,0.1)
nu = constant(rand(get_ngauss(mesh)))
u = constant(rand(mesh.ndof))

f = compute_fem_laplace_term1(u, nu, mesh)

sess = Session(); init(sess)
run(sess, f)