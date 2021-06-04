export neo_hookean, compute_absorbing_boundary_condition_matrix

@doc raw"""
    neo_hookean(u::Union{Array{Float64, 1}, PyObject}, 
        μ::Union{Array{Float64, 1}, PyObject},
        λ::Union{Array{Float64, 1}, PyObject},
        mesh::Mesh)
    neo_hookean(u::Array{Float64, 1}, μ::Array{Float64, 1}, λ::Array{Float64, 1}, mesh::Mesh)

Computes the elastic stored energy density for  a common neo-Hookean stored energy model.

$$\psi =  \frac{\mu}{2} (I_{c} - 2) - \frac{\mu}{2} \ln(J) + \frac{\lambda}{8}\ln(J)^{2}$$

Here the deformation gradient $F$ is defined by 

$$F = I + \nabla u$$

the right Cauchy-Green tensor $C$ is given by 

$$C = F^{T} F$$

and the scalars $J$ and $I_c$

$$\begin{split}J     &= \det(C), \\
I_{c} &= {\rm trace}(C).\end{split}$$

$\mu$ and $\lambda$ are Lame parameters. 
"""
function neo_hookean(u::Union{Array{Float64, 1}, PyObject}, 
      μ::Union{Array{Float64, 1}, PyObject},
      λ::Union{Array{Float64, 1}, PyObject},
      mesh::Mesh)
    @assert length(μ)==length(λ)==get_ngauss(mesh)
    @assert length(u) == 2mesh.ndof
    neo_hookean_ = load_op_and_grad(AdFem.libmfem,"neo_hookean", multiple=true)
    u,mu,lamb = convert_to_tensor(Any[u,μ,λ], [Float64,Float64,Float64])
    psi, indices, vv = neo_hookean_(u,mu,lamb)
    J = RawSparseTensor(indices, vv, 2mesh.ndof, 2mesh.ndof)
    set_shape(psi, (2mesh.ndof,)), J
end



function neo_hookean(u::Array{Float64, 1}, μ::Array{Float64, 1}, λ::Array{Float64, 1}, mesh::Mesh)
    @assert length(μ)==length(λ)==get_ngauss(mesh)
    @assert length(u) == 2mesh.ndof
    N = get_ngauss(mesh)*(2*mesh.elem_ndof)^2
    psi = zeros(2mesh.ndof)
    indices = zeros(Int64, 2*N)
    vv = zeros(N)
    @eval ccall((:NH_forward_Julia, $LIBMFEM), Cvoid,
         (Ptr{Cdouble}, Ptr{Clonglong}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}), 
            $psi, $indices, $vv, $u, $μ, $λ)
    J = sparse(indices[1:2:end] .+ 1, indices[2:2:end] .+ 1, vv, 2mesh.ndof, 2mesh.ndof)
    psi, J
end


function compute_absorbing_boundary_condition_matrix(
    ρ::Union{PyObject, Array{Float64, 1}},
    cs::Union{PyObject, Array{Float64, 1}},
    cp::Union{PyObject, Array{Float64, 1}},
    bdedge::Array{Int64, 2}, mmesh::Mesh)
    @assert length(ρ)==length(cs)==length(cp)==4*size(bdedge, 1)
    ρ, cs, cp = convert_to_tensor([ρ, cs, cp], [Float64, Float64, Float64])
    n = get_edge_normal(bdedge, mmesh)
    m = [n[:,2] -n[:,1]]
    n1 = constant(repeat(n[:,1], 1, 4)'[:])
    n2 = constant(repeat(n[:,2], 1, 4)'[:])
    m1 = constant(repeat(m[:,1], 1, 4)'[:])
    m2 = constant(repeat(m[:,2], 1, 4)'[:])
    a11 = ρ*cp*n1^2+ρ*cs*m1^2
    a12 = ρ*cp*n1*n2+ρ*cs*m1*m2
    a21 = ρ*cp*n1*n2+ρ*cs*m1*m2
    a22 = ρ*cp*n2^2+ρ*cs*m2^2
    A11 = compute_fem_boundary_mass_matrix1(a11, bdedge, mmesh)
    A12 = compute_fem_boundary_mass_matrix1(a12, bdedge, mmesh)
    A21 = compute_fem_boundary_mass_matrix1(a21, bdedge, mmesh)
    A22 = compute_fem_boundary_mass_matrix1(a22, bdedge, mmesh)
    K = [A11 A21
        A12 A22]
end