using Revise
using PoreFlow
using PyPlot 
using LinearAlgebra
using ADCME 
using Statistics
using SparseArrays


function get_fault_info(mmesh, c1, c2)
    # c1=(0.5, 0.0)
    # c2=(1.5, 0.5)
    vec = [c2[1]-c1[1], c2[2]-c1[2]]
    k = vec[2]/(vec[1]+1e-8)
    normal_vec = [-vec[2] vec[1]]./norm(vec)
    nodes = mmesh.nodes
    ids = Int64[]
    coords = Any[]
    dist(x0, y0) = abs(k * (x0-c1[1]) - (y0-c1[2])) / sqrt(k^2 + 1)
    for i = 1:size(nodes, 1)
        x, y = nodes[i,:]
        if dist(x, y) < 1e-3 && (c1[1] <= x <= c2[1])
            push!(ids, i)
            push!(coords, [x, y])
        end
    end
   return ids, hcat(coords...), normal_vec
end

"""
isredblack(elem_idx, node_idx, mmesh)::Bool
"""
function fix_mesh(mmesh, red_fault_idx, isredblack)
    nodes = mmesh.nodes
    elems = mmesh.elems

    extra_nodes = [nodes;nodes[red_fault_idx,:]]
    black_fault_idx =collect( mmesh.nnode .+ (1:length(red_fault_idx)) )
    black_fault_idx_map = Dict()
    for i = 1:length(red_fault_idx)
        black_fault_idx_map[red_fault_idx[i]] = black_fault_idx[i]
    end

    conn = zeros(Int64, length(red_fault_idx), 2)
    for i = 1:length(red_fault_idx)
        conn[i,:] = [red_fault_idx[i] black_fault_idx[i]]
    end

    is_fault_node = Set(red_fault_idx)
    for i = 1:mmesh.nelem
        candidate = elems[i,:]
        for j = 1:3
            if elems[i,j] in is_fault_node
                if isredblack(i, elems[i,j], mmesh)
                    continue 
                else 
                    candidate[j] = black_fault_idx_map[elems[i,j]]
                end
            end
        end
        elems[i,:] = candidate
    end
    Mesh(extra_nodes, elems), conn
end

function rectangle_redblack(elem_idx, node_idx, mmesh)
    node = mean(mmesh.nodes[mmesh.elems[elem_idx,:],:], dims=1)
    x = node[1,1]
    if x<1.0
        return true 
    else
        return false
    end
end


function fault_redblack(elem_idx, node_idx, mmesh)
    node = mean(mmesh.nodes[mmesh.elems[elem_idx,:],:], dims=1)
    x, y = node 
    return y-0.5x+0.25>0
end



function viscoelasticity_solver(invη, μ, λ, Forces, mmesh)

    ## alpha-scheme
    β = 1/4; γ = 1/2
    a = b = 0.1    # damping 

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

    U, Sigma, Varepsilon
end


function compute_slip_boundary_condition( conn, mmesh)
    ii = Int64[]
    jj = Int64[]
    vv = Float64[]
    k = 0
    # u - u' = du 
    # v - v' = dv
    for i = 1:size(conn, 1)
        k += 1
        push!(ii, k)
        push!(jj, conn[i,1])
        push!(vv, 1.0)
        push!(ii, k)
        push!(jj, conn[i,2])
        push!(vv, -1.0)
    end

    for i = 1:size(conn, 1)
        k += 1
        push!(ii, k)
        push!(jj, conn[i,1] + mmesh.ndof)
        push!(vv, 1.0)
        push!(ii, k)
        push!(jj, conn[i,2] + mmesh.ndof)
        push!(vv, -1.0)
    end
    sparse(ii, jj, vv, 2size(conn,1), 2mmesh.ndof)
end

function subduction_solver(invη, gravity, Conn, slipvec, bdnode, mmesh)
    slipvec = constant(slipvec)
    Conn = constant(Conn)
    invη = constant(invη)
    @assert size(slipvec,1)==NT 
    @assert size(slipvec,2)==size(Conn,1)
    @assert size(Conn,2) == 2mmesh.ndof
    @assert isa(gravity, Real)
    @assert size(invη)==(get_ngauss(mmesh),)


    l_length = size(Conn, 1)
    ## alpha-scheme
    β = 1/4; γ = 1/2
    a = b = 0.1    # damping 

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
    L = [L Conn'
        Conn spzero(l_length, l_length)]
    bddof = [bdnode; bdnode .+ mmesh.ndof]
    L, _ = impose_Dirichlet_boundary_conditions(L, zeros(2mmesh.ndof+l_length), bddof, zeros(length(bddof)))
    L = factorize(L)

    a = TensorArray(NT+1); a = write(a, 1, zeros(2mmesh.ndof))
    v = TensorArray(NT+1); v = write(v, 1, zeros(2mmesh.ndof))
    d = TensorArray(NT+1); d = write(d, 1, zeros(2mmesh.ndof))
    U = TensorArray(NT+1); U = write(U, 1, zeros(2mmesh.ndof))
    Sigma = TensorArray(NT+1); Sigma = write(Sigma, 1, zeros(get_ngauss(mmesh), 3))
    Varepsilon = TensorArray(NT+1); Varepsilon = write(Varepsilon, 1, zeros(get_ngauss(mmesh), 3))

    Forces = compute_fem_source_term(zeros(get_ngauss(mmesh)), -ones(get_ngauss(mmesh))*gravity, mmesh)

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
      rhs = Forces - F

      td = d + Δt*v + Δt^2/2*(1-2β)*a 
      tv = v + (1-γ)*Δt*a 
      rhs = rhs - C*tv - K*td
      rhs = scatter_update(rhs, bddof, zeros(length(bddof)))


      ## alpha-scheme
      rhs = [rhs; slipvec[i]]
      a = L\rhs # bottleneck  
      a = a[1:2mmesh.ndof]
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

    U, Sigma, Varepsilon
end