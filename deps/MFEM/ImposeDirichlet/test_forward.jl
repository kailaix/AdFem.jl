using Revise
using AdFem
using LinearAlgebra
using SparseArrays

A = rand(10, 10)
bd = rand(1:10, 5)
bdval = rand(5)
rhs = rand(10)
A1, rhs1 = impose_Dirichlet_boundary_conditions(A, rhs, bd, bdval)
A2, rhs2 = impose_Dirichlet_boundary_conditions(constant(sparse(A)), rhs, bd, bdval)

sess = Session(); init(sess)
@info maximum(abs.(Array(run(sess, A2)) - A1))
@info maximum(abs.(run(sess, rhs2) - rhs1))