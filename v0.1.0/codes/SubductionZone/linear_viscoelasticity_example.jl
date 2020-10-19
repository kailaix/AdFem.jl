include("common.jl")

mmesh = Mesh(10, 10, 1/10, degree=1)
NT = 500
Δt = 2.0/NT
ηmax = 1
ηmin = 0.5

bdedge = bcedge((x1, y1, x2, y2)->(x1>1-1e-5 && x2>1-1e-5), mmesh)
bdnode = bcnode((x1, y1)->(y1<1e-5), mmesh)

invη = 50 * constant(ones(get_ngauss(mmesh)))
μ = 1.0
λ = 1.0


Forces = zeros(NT, 2mmesh.ndof)
T = eval_f_on_boundary_edge((x,y)->0.1, bdedge, mmesh)
T = [-T T]
rhs = compute_fem_traction_term(T, bdedge, mmesh)
for i = 1:NT
  Forces[i, :] = rhs
end
Forces = constant(Forces)

U, Sigma, Varepsilon = viscoelasticity_solver(invη, μ, λ, Forces, mmesh)

sess = Session(); init(sess)
Uval, Sigmaval, Varepsilonval = run(sess, [U, Sigma, Varepsilon])
p = visualize_displacement(Uval[1:5:end,:], mmesh)
saveanim(p, "space_u.gif")

