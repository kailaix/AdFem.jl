include("common.jl")

mmesh = Mesh("rectangle.stl")
c1=(1.0, 0.0)
c2=(1.0, 1.0)
fault_idx, fault_coords, fault_normal_vec = get_fault_info(mmesh, c1, c2)


mmesh, conn = fix_mesh(mmesh, fault_idx, rectangle_redblack)
close("all")
visualize_mesh(mmesh)
v = 0.05*(rand(mmesh.nnode).-0.5)
u = 0.05*(rand(mmesh.nnode).-0.5)
scatter(mmesh.nodes[:,1] + u, mmesh.nodes[:,2] + v, s = 2)
# plot(fault_coords[1,:], fault_coords[2,:], "r.", label="fault")
# legend()
savefig("mesh.png")

bdnode = bcnode((x, y)->(x<1e-5 || x>2-1e-5), mmesh)

Conn = compute_slip_boundary_condition( conn, mmesh)
NT = 100
Δt = 0.004
λ = 1.0
μ = 1.0
invη = ones(get_ngauss(mmesh))
sn = size(conn,1)
slipvec = [zeros(NT, sn) ones(NT, sn)] # acceleration
gravity = 0.0

U, Sigma, Varepsilon = subduction_solver(invη, gravity, Conn, slipvec, bdnode, mmesh)

sess = Session(); init(sess)
Uval, Stress, Strain = run(sess, [U, Sigma, Varepsilon])
p = visualize_displacement(Uval[1:5:end,:], mmesh)
saveanim(p, "subduction.gif")

close("all")
u = Uval[end,:]
# u = (@. sqrt(u[1:mmesh.ndof]^2 + u[mmesh.ndof+1:2mmesh.ndof]^2))
visualize_scalar_on_fem_points(u[mmesh.ndof+1:2mmesh.ndof],mmesh)
savefig("stress_yy.png")
# node = [0.0 0.0
#         1.0 0.0
#         0.0 1.0
#         5.0 0.0
#         6.0 0.0
#         5.0 1.0]

# elem = [1 2 3
#         4 5 6]

# mmesh = Mesh(node, elem)
# close("all")
# visualize_mesh(mmesh)
# savefig("mesh.png")