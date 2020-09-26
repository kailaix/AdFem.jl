export neo_hookean

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
    neo_hookean_ = load_op_and_grad(PoreFlow.libmfem,"neo_hookean", multiple=true)
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
