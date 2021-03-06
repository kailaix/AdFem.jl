using Revise
using ADCME
using ADCMEKit
using NNFEM 
using PyPlot
using ProgressMeter 
using AdFem
using SparseArrays
#----------------------------------------------------------------------------
# step 1: create a computational domain 
# fixed on the bottom and push from right 


m = 40
n = 20
h = 1/n
NT = 50
Δt = 0.01/NT

bdedge = bcedge("right", m, n, h)
bdnode = bcnode("lower", m, n, h)

λ = constant(8.641975308641973e9)
μ = constant(3.7037037037037034e9)
invη = 1/1e5*constant(ones(4*m*n))


M = 2000*compute_fem_mass_matrix1(m, n, h)
Zero = spzeros((m+1)*(n+1), (m+1)*(n+1))
M = SparseTensor([M Zero;Zero M])

## alpha-scheme
β = 1/4; γ = 1/2

# invη is a 4*m*n array 
function make_matrix(invη)
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
  K = compute_fem_stiffness_matrix(H, m, n, h)
  L = M + β*Δt^2*K
  # L = pl[1]*L
  L, Lbd = fem_impose_Dirichlet_boundary_condition_experimental(L, bdnode, m, n, h)

  return K, L, S, invG
end

function simulate()


  a = TensorArray(NT+1); a = write(a, 1, zeros(2(m+1)*(n+1))|>constant)
  v = TensorArray(NT+1); v = write(v, 1, zeros(2(m+1)*(n+1))|>constant)
  d = TensorArray(NT+1); d = write(d, 1, zeros(2(m+1)*(n+1))|>constant)
  U = TensorArray(NT+1); U = write(U, 1, zeros(2(m+1)*(n+1))|>constant)
  Sigma = TensorArray(NT+1); Sigma = write(Sigma, 1, zeros(4*m*n, 3)|>constant)
  Varepsilon = TensorArray(NT+1); Varepsilon = write(Varepsilon, 1,zeros(4*m*n, 3)|>constant)


  Forces = zeros(NT, 2(m+1)*(n+1))
  for i = 1:NT
    T = eval_f_on_boundary_edge((x,y)->1e6, bdedge, m, n, h)

    T = [-T zeros(size(T)...)]
    rhs = compute_fem_traction_term(T, bdedge, m, n, h)

    Forces[i, :] = rhs
  end
  Forces = constant(Forces)

  K, L, S, invG = make_matrix(invη)

  L = factorize(L)

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
    F = compute_strain_energy_term(res, m, n, h) - K * U
    rhs = Forces[i] -  F

    td = d + Δt*v + Δt^2/2*(1-2β)*a 
    tv = v + (1-γ)*Δt*a 
    rhs = rhs - K*td
    rhs = scatter_update(rhs, constant([bdnode; bdnode.+(m+1)*(n+1)]), constant(zeros(2*length(bdnode))))


    ## alpha-scheme
    a = L\rhs # bottleneck  
    d = td + β*Δt^2*a 
    v = tv + γ*Δt*a 
    U_new = d

    Varepsilon_new = eval_strain_on_gauss_pts(U_new, m, n, h)

    res2 = batch_matmul(invG * S, Varepsilon_new-Varepsilon)
    Sigma_new = res +  res2

    i+1, write(a_, i+1, a), write(v_, i+1, v), write(d_, i+1, d), write(U_, i+1, U_new),
          write(Sigma_, i+1, Sigma_new), write(Varepsilon_, i+1, Varepsilon_new)
  end



  i = constant(1, dtype=Int32)
  _, _, _, _, u, sigma, varepsilon = while_loop(condition, body, 
                    [i, a, v, d, U, Sigma, Varepsilon])

  U = stack(u)
  Sigma = stack(sigma)
  Varepsilon = stack(varepsilon)

  U = set_shape(U, (NT+1, size(U,2)))
  Sigma = set_shape(Sigma, (NT+1, 4*m*n, 3))

  return U, Sigma
end

U, sigma = simulate()
sess = Session(); init(sess)
U, sigma = run(sess, [U, sigma])


##

elements = []
# prop = Dict("name"=> "PlaneStrain", "rho"=> 2000, "E"=> 1e10, "nu"=> 0.35)
prop = Dict("name"=> "ViscoelasticityMaxwell", "rho"=> 2000, "E"=> 1e10, "nu"=> 0.35, "eta"=>1e7)

coords = zeros((m+1)*(n+1), 2)
for j = 1:n
    for i = 1:m
        idx = (m+1)*(j-1)+i 
        elnodes = [idx; idx+1; idx+1+m+1; idx+m+1]
        ngp = 2
        nodes = [
        (i-1)*h (j-1)*h
        i*h (j-1)*h
        i*h j*h
        (i-1)*h j*h
        ]
        coords[elnodes, :] = nodes
        push!(elements, SmallStrainContinuum(nodes, elnodes, prop, ngp))
    end
end

# fixed on the bottom, push on the right
EBC = zeros(Int64, (m+1)*(n+1), 2)
FBC = zeros(Int64, (m+1)*(n+1), 2)
g = zeros((m+1)*(n+1), 2)
f = zeros((m+1)*(n+1), 2)

for i = 1:m+1
    EBC[i+n*(m+1),:] .= -1
end

Edge_Traction_Data = Array{Int64}[]
for i = 1:n
  elem = elements[(i-1)*m + m]
  for k = 1:4
    if elem.coords[k,1]>2-0.001 && elem.coords[k+1>4 ? 1 : k+1,1]>2-0.001
      push!(Edge_Traction_Data, [(i-1)*m + m, k, 1])
    end
  end
end
Edge_Traction_Data = hcat(Edge_Traction_Data...)'|>Array

ndims = 2
domain = Domain(coords, elements, ndims, EBC, g, FBC, f, Edge_Traction_Data)
######

domain.history["state"] = []
domain.history["stress"] = []
for i = 1:NT+1
  push!(domain.history["state"], U[i,:])
  push!(domain.history["stress"], sigma[i,:,:])
end
# visualize_von_mises_stress_on_scoped_body(d_, domain, scale_factor=10.0)
p = visualize_total_deformation_on_scoped_body(U, domain, scale_factor=10.0)
saveanim(p, "adfem.gif")
close("all")