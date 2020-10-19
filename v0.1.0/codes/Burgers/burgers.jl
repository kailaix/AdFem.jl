

ADCME.options.newton_raphson.verbose = true
Δt = 0.01
mmesh = Mesh(30, 30, 1/30)
bdnode = bcnode(mmesh)
bdnode = [bdnode; bdnode .+ mmesh.ndof]
nu = constant(0.0001ones(get_ngauss(mmesh)))
nodes = fem_nodes(mmesh)
u = @. sin(2π * nodes[:,1])
v = @. cos(2π * nodes[:,2])
u0 = [u;v]
u0[bdnode] .= 0.0
us = solve_burgers(u0, 10)


sess = Session(); init(sess)
U = run(sess, us)
figure(figsize=(10,5))
close("all")
subplot(121)
visualize_scalar_on_fem_points(U[end,1:mmesh.nnode], mmesh)
subplot(122)
visualize_scalar_on_fem_points(U[end,mmesh.ndof + 1:mmesh.ndof + mmesh.nnode], mmesh)
savefig("test_mfem.png")

close("all")
visualize_vector_on_fem_points(U[end,1:mmesh.nnode], U[end,mmesh.ndof + 1:mmesh.ndof + mmesh.nnode], mmesh)
savefig("mfem_quiver.png")
# function ff(x)
#     unext = constant(x[1:mmesh.ndof])
#     vnext = constant(x[mmesh.ndof+1:end])
#     r, J = calc_residual_and_jacobian(unext, vnext, u, v, mmesh)
#     run(sess, r), run(sess, J)
# end

# ff(rand(2mmesh.ndof))
# sess = Session(); init(sess)
# test Jacobian 
# test_jacobian(ff, rand(2mmesh.ndof))