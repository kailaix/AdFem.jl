using Revise
using AdFem

mmesh3 = Mesh3(10,10,10,0.1)
xy = gauss_nodes(mmesh3)

x, y, z = xy[:,1], xy[:,2], xy[:,3]
f = @. (1-x) * x * (1-y) * y * (1+x^2+z^2)

F = compute_fem_source_term1(f, mmesh3)
M = compute_fem_mass_matrix1((@. 1+x^2+z^2), mmesh3)

bd = bcnode(mmesh3)
M, F = impose_Dirichlet_boundary_conditions(M, F, bd, zeros(length(bd)))
s = M\F

sess = Session(); init(sess)
SOL = run(sess, s)

f = vtk(mmesh3, "test")
f["u"] = SOL 
save(f)

xy = fem_nodes(mmesh3)
x, y, z = xy[:,1], xy[:,2], xy[:,3]

f = vtk(mmesh3, "test2")
f["u"] =  @. (1-x) * x * (1-y) * y
save(f)