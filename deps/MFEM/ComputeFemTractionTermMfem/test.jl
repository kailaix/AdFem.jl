using Revise
using AdFem
using LinearAlgebra
using PyPlot

n = 20
mesh = Mesh(n, n, 1/n, degree=2)
bd = bcedge(mesh)
# bdedge = []
# for i = 1:size(bd, 1)
#     a, b = bd[i,:]
#     if mesh.nodes[a,2]<1e-5 && mesh.nodes[b, 2]<1e-5
#         push!(bdedge, [a b])
#     end
# end
# bdedge = vcat(bdedge...)
bdedge = []
for i = 1:size(bd, 1)
    a, b = bd[i,:]
    if mesh.nodes[a,2]>1-1e-5 && mesh.nodes[b, 2]>1-1e-5
        push!(bdedge, [a b])
    end
end
bdedge = vcat(bdedge...)


# nodes = get_bdedge_integration_pts(bdedge, mesh)
# close("all")
# visualize_mesh(mesh)
# # scatter(mesh.nodes[bdedge[:,1],1], mesh.nodes[bdedge[:,1],2])
# # scatter(mesh.nodes[bdedge[:,2],1], mesh.nodes[bdedge[:,2],2], marker = "x")
# scatter(nodes[:,1], nodes[:,2], s = 2)
# savefig("test_quadrature.png")

# nx, ny = AdFem._traction_get_nodes(bdedge, mesh)
# close("all")
# visualize_mesh(mesh)
# scatter(nx[:,1], ny[:,1])
# scatter(nx[:,2], ny[:,2], marker = "+")
# scatter(nx[:,3], ny[:,3], marker = "x")
# savefig("test_vertex.png")


t = eval_f_on_boundary_edge((x, y)->  1.0, bdedge, mesh)
rhs = compute_fem_traction_term1(t, bdedge, mesh)
sum(rhs)
# close("all")
# plot(rhs)
# savefig("test_traction.png")