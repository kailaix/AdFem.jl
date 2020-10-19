include("common.jl")


function f(x, y)
    0.0001*(10/(1+x^2) + x * y + 10*y^2)
end


Δt = 0.01
mmesh = Mesh(30, 30, 1/30)
bdnode = bcnode(mmesh)
bdnode = [bdnode; bdnode .+ mmesh.ndof]

nu = constant(eval_f_on_gauss_pts(f, mmesh))

nodes = fem_nodes(mmesh)
u = @. sin(2π * nodes[:,1])
v = @. sin(2π * nodes[:,2])
u0 = [u;v]
u0[bdnode] .= 0.0
us = solve_burgers(u0, 10, nu)

sess = Session(); init(sess)
U = run(sess, us)


matwrite("fenics/fwd2.mat", Dict("U"=>U))
close("all")
figure(figsize=(10,3))
subplot(121)
title("u displacement")
visualize_scalar_on_fem_points(U[end,1:mmesh.nnode], mmesh)
subplot(122)
title("v displacement")
visualize_scalar_on_fem_points(U[end,mmesh.ndof + 1:mmesh.ndof + mmesh.nnode], mmesh)
savefig("fenics/fwd2.png")

close("all")
visualize_vector_on_fem_points(U[end,1:mmesh.nnode], U[end,mmesh.ndof + 1:mmesh.ndof + mmesh.nnode], mmesh)
savefig("fenics/fwd2_quiver.png")


close("all")
visualize_scalar_on_gauss_points(run(sess, nu), mmesh)
savefig("fenics/nu.png")