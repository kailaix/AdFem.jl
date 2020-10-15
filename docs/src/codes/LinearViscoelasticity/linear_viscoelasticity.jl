using Revise
using PoreFlow
using PyCall
using LinearAlgebra
using PyPlot
using SparseArrays
using MAT
np = pyimport("numpy")

stepsize = 1
if length(ARGS)==1
  global stepsize = parse(Int64, ARGS[1])
end
@info stepsize


## alpha-scheme
β = 1/4; γ = 1/2
a = b = 0.1    # damping 

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

fn_G = invη->begin 
  G = tensor([1/Δt+2/3*μ*invη -μ/3*invη 0.0
    -μ/3*invη 1/Δt+2/3*μ*invη 0.0
    0.0 0.0 1/Δt+μ*invη])
  invG = inv(G)
end
invG = map(fn_G, invη)
S = tensor([2μ/Δt+λ/Δt λ/Δt 0.0
    λ/Δt 2μ/Δt+λ/Δt 0.0
    0.0 0.0 μ/Δt])
H = invG*S


M = compute_fem_mass_matrix1(mmesh)
Zero = spzero(mmesh.ndof, mmesh.ndof)
M = [M Zero;Zero M]

K = compute_fem_stiffness_matrix(H, mmesh)
C = a*M + b*K # damping matrix 
L = M + γ*Δt*C + β*Δt^2*K
bddof = [bdnode; bdnode .+ mmesh.ndof]
L, _ = impose_Dirichlet_boundary_conditions(L, zeros(2mmesh.ndof), bddof, zeros(length(bddof)))
L = factorize(L)

a = TensorArray(NT+1); a = write(a, 1, zeros(2mmesh.ndof))
v = TensorArray(NT+1); v = write(v, 1, zeros(2mmesh.ndof))
d = TensorArray(NT+1); d = write(d, 1, zeros(2mmesh.ndof))
U = TensorArray(NT+1); U = write(U, 1, zeros(2mmesh.ndof))
Sigma = TensorArray(NT+1); Sigma = write(Sigma, 1, zeros(get_ngauss(mmesh), 3))
Varepsilon = TensorArray(NT+1); Varepsilon = write(Varepsilon, 1, zeros(get_ngauss(mmesh), 3))


Forces = zeros(NT, 2mmesh.ndof)
T = eval_f_on_boundary_edge((x,y)->0.1, bdedge, mmesh)
T = [-T T]
rhs = compute_fem_traction_term(T, bdedge, mmesh)
for i = 1:NT
  Forces[i, :] = rhs
end
Forces = constant(Forces)

function condition(i, tas...)
  i <= NT
end

function body(i, tas...)
  a_, v_, d_, U_, Sigma_, Varepsilon_ = tas
  a = read(a_, i)
  v = read(v_, i)
  d = read(d_, i)
  U = read(U_, i)
  Sigma = read(Sigma_, i)
  Varepsilon = read(Varepsilon_, i)

  res = batch_matmul(invG/Δt, Sigma)
  F = compute_strain_energy_term(res, mmesh) - K * U
  rhs = Forces[i] - F

  td = d + Δt*v + Δt^2/2*(1-2β)*a 
  tv = v + (1-γ)*Δt*a 
  rhs = rhs - C*tv - K*td
  rhs = scatter_update(rhs, bddof, zeros(length(bddof)))


  ## alpha-scheme
  a = L\rhs # bottleneck  
  d = td + β*Δt^2*a 
  v = tv + γ*Δt*a 
  U_new = d

  Varepsilon_new = eval_strain_on_gauss_pts(U_new, mmesh)
  Sigma_new = update_stress_viscosity(Varepsilon_new, Varepsilon, Sigma, invη, μ*ones(get_ngauss(mmesh)), λ*ones(get_ngauss(mmesh)), Δt)

  i+1, write(a_, i+1, a), write(v_, i+1, v), write(d_, i+1, d), write(U_, i+1, U_new),
        write(Sigma_, i+1, Sigma_new), write(Varepsilon_, i+1, Varepsilon_new)
end


i = constant(1, dtype=Int32)
_, _, _, _, u, sigma, varepsilon = while_loop(condition, body, 
                  [i, a, v, d, U, Sigma, Varepsilon])

U = stack(u)
Sigma = stack(sigma)
Varepsilon = stack(varepsilon)

sess = Session(); init(sess)
Uval,Sigmaval, Varepsilonval = run(sess, [U, Sigma, Varepsilon])

# p = visualize_von_mises_stress(Sigmaval[1:5:end,:,:], m, n, h); saveanim(p, "space_s.gif")
p = visualize_displacement(Uval[1:5:end,:], mmesh)
saveanim(p, "space_u.gif")

# visualize_von_mises_stress(Sigmaval[end,:,:], m, n, h); savefig("space_s.pdf")
# visualize_displacement(Uval[end,:], m, n, h); savefig("space_u.pdf")

# visualize_inv_eta(run(sess, invη), "true")
