include("common.jl")

Δt = 0.01
mmesh = Mesh(30, 30, 1/30)
bdnode = bcnode(mmesh)
bdnode = [bdnode; bdnode .+ mmesh.ndof]
nu = Variable(0.001) * ones(get_ngauss(mmesh))
nodes = fem_nodes(mmesh)
u = @. sin(2π * nodes[:,1])
v = @. sin(2π * nodes[:,2])
u0 = [u;v]
u0[bdnode] .= 0.0
us = solve_burgers(u0, 10, nu)


U = matread("fenics/fwd1.mat")["U"]
loss = sum((us - U)^2)
sess = Session(); init(sess)
# @info run(sess, loss)
BFGS!(sess, loss)
