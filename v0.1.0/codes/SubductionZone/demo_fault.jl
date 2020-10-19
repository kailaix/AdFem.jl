include("common.jl")

mmesh = Mesh(joinpath(PDATA,"crack.stl"))
c1=(0.50, 0.0)
c2=(1.5, 0.5)
fault_idx, fault_coords, fault_normal_vec = get_fault_info(mmesh, c1, c2)

close("all")
visualize_mesh(mmesh)
plot(mmesh.nodes[fault_idx,1], mmesh.nodes[fault_idx,2], ".")
savefig("mesh1.png")


mmesh, conn = fix_mesh(mmesh, fault_idx, fault_redblack)
cn1 = conn[:,1]
cn2 = conn[:,2]
close("all")
visualize_mesh(mmesh)
v = 0.01*(rand(length(cn1)).-0.5)
u = 0.01*(rand(length(cn1)).-0.5)
scatter(mmesh.nodes[cn1,1] + u, mmesh.nodes[cn1,2] + v, s = 2, color="red")
scatter(mmesh.nodes[cn2,1], mmesh.nodes[cn2,2], s = 2, color="blue")

savefig("mesh.png")

bdnode = bcnode((x, y)->(x<1e-5 || x>2-1e-5), mmesh)

Conn = compute_slip_boundary_condition( conn, mmesh)
NT = 100
Δt = 0.004
λ = 1.0
μ = 1.0
invη = constant(ones(get_ngauss(mmesh)))
sn = size(conn,1)
slipvec = constant(0.1*[2ones(NT, sn) ones(NT, sn)]) # acceleration
gravity = 0.0

U, Sigma, Varepsilon = subduction_solver(invη, gravity, Conn, slipvec, bdnode, mmesh)

sess = Session(); init(sess)
Uval, Stress, Strain = run(sess, [U, Sigma, Varepsilon])
p = visualize_displacement(Uval[1:5:end,:], mmesh)
saveanim(p, "subduction_rectangle.gif")


loss = sum(U^2)
close("all")
u = Uval[end,:]
visualize_scalar_on_fem_points(u[mmesh.ndof+1:2mmesh.ndof],mmesh)
savefig("y_displacement.png")

close("all")
p = visualize_scalar_on_gauss_points(Stress[end,:,1], mmesh)
savefig("fault_stress.png")