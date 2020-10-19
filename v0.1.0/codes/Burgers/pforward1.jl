include("common.jl")

Δt = 0.01
mmesh = Mesh(30, 30, 1/30)
bdnode = bcnode(mmesh)
bdnode = [bdnode; bdnode .+ mmesh.ndof]
nu = constant(0.0001ones(get_ngauss(mmesh)))
nodes = fem_nodes(mmesh)
u = @. sin(2π * nodes[:,1])
v = @. sin(2π * nodes[:,2])
u0 = [u;v]
u0[bdnode] .= 0.0
us = solve_burgers(u0, 10, nu)

sess = Session(); init(sess)
U = run(sess, us)


matwrite("fenics/fwd1.mat", Dict("U"=>U))
close("all")
figure(figsize=(10,3))
subplot(121)
title("u displacement")
visualize_scalar_on_fem_points(U[end,1:mmesh.nnode], mmesh)
subplot(122)
title("v displacement")
visualize_scalar_on_fem_points(U[end,mmesh.ndof + 1:mmesh.ndof + mmesh.nnode], mmesh)
savefig("fenics/fwd1.png")

close("all")
visualize_vector_on_fem_points(U[end,1:mmesh.nnode], U[end,mmesh.ndof + 1:mmesh.ndof + mmesh.nnode], mmesh)
savefig("fenics/fwd1_quiver.png")