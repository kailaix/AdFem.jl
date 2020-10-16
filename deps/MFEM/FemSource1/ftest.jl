using AdFem
using LinearAlgebra

for d in [1, 2]
    mesh = Mesh(8, 8, 1/8, degree=d)
    c0 = rand(get_ngauss(mesh))
    c = constant(c0)
    
    out = compute_fem_source_term1(c, mesh)
    out0 = compute_fem_source_term1(c0, mesh)
    C = compute_fem_mass_matrix1(c, mesh)
    o = C * ones(mesh.ndof)
    sess = Session(); init(sess)
    @info maximum(abs.(run(sess, out)-run(sess, o)))
    @info maximum(abs.(run(sess, c)-c0))
end
