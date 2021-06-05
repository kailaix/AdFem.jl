using AdFem
using PyPlot
mmesh = Mesh(10, 10, 1/10)
close("all")
visualize_mesh(mmesh)
savefig("mesh1.png")

mmesh = Mesh_v2(10, 10, 1/10)
close("all")
visualize_mesh(mmesh)
savefig("mesh2.png")

mmesh = Mesh_v3(10, 10, 1/10)
close("all")
visualize_mesh(mmesh)
savefig("mesh3.png")