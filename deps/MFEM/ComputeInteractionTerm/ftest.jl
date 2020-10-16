using ADCME
using LinearAlgebra
using AdFem
using DelimitedFiles

p = readdlm("fenics/x.txt")[:]
f = readdlm("fenics/f.txt")[:]

mesh = Mesh(8, 8, 1. /8)
f0 = compute_interaction_term(p, mesh)
sess = Session(); init(sess)
f1 = run(sess, f0)

@show norm(f - f1)


p = readdlm("fenics/x2.txt")[:]
f = readdlm("fenics/f2.txt")[:]

mesh = Mesh(8, 8, 1. /8, degree=2)
f0 = compute_interaction_term(p, mesh)
sess = Session(); init(sess)
f1 = run(sess, f0)

E = Int64.(readdlm("fenics/edges.txt"))
Edof = get_edge_dof(E, mesh)
DOF = [1:mesh.nnode; Edof .+ mesh.nnode]
f1 = f1[[DOF; DOF .+ mesh.ndof]]
@show norm(f - f1)
