export neo_hookean

@doc raw"""
    neo_hookean(u::Union{Array{Float64, 1}, PyObject}, 
        μ::Union{Array{Float64, 1}, PyObject},
        λ::Union{Array{Float64, 1}, PyObject},
        mesh::Mesh)

Computes the Neo-Hookean hyperelasticity invariances and their sensitivity matrix. 

$$F = I + \nabla \mathbf{u}, C = F^T F$$

The two invariances are given by 

$$\begin{aligned}I_c &= \mu\text{tr}(C) \\ J_c &= \lambda \log \text{det} C\end{aligned}$$

Returns $I_c$, $\frac{\partial I_c}{\partial \mathbf{u}}$, $J_c$, $\frac{\partial J_c}{\partial \mathbf{u}}$
"""
function neo_hookean(u::Union{Array{Float64, 1}, PyObject}, 
      μ::Union{Array{Float64, 1}, PyObject},
      λ::Union{Array{Float64, 1}, PyObject},
      mesh::Mesh)
    @assert length(μ)==length(λ)==get_ngauss(mesh)
    @assert length(u) == 2mesh.ndof
    neo_hookean_ = load_op_and_grad(PoreFlow.libmfem,"neo_hookean", multiple=true)
    u,mu,lamb = convert_to_tensor(Any[u,μ,λ], [Float64,Float64,Float64])
    ic, jc, idx1, vv1, idx2, vv2 = neo_hookean_(u,mu,lamb)
    DIc = RawSparseTensor(idx1, vv1, 2mesh.ndof, 2mesh.ndof)
    DJc = RawSparseTensor(idx2, vv2, 2mesh.ndof, 2mesh.ndof)
    set_shape(ic, (2mesh.ndof,)), DIc, set_shape(jc, (2mesh.ndof,)), DJc
end



function neo_hookean(u::Array{Float64, 1}, μ::Array{Float64, 1}, λ::Array{Float64, 1}, mesh::Mesh)
    @assert length(μ)==length(λ)==get_ngauss(mesh)
    @assert length(u) == 2mesh.ndof
    ic = zeros(2*mesh.ndof)
    jc = zeros(2*mesh.ndof)
    N = get_ngauss(mesh)*(2*mesh.elem_ndof)^2
    idi = zeros(Int64, 2*N)
    iv = zeros(N)
    jdi = zeros(Int64, 2*N)
    jv = zeros(N)
    @eval ccall((:NH_forward_Julia, $LIBMFEM), Cvoid,
         (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Clonglong}, Ptr{Cdouble}, Ptr{Clonglong}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}), 
            $ic, $jc, $idi, $iv, $jdi, $jv, $u, $μ, $λ)
    DIc = sparse(idi[1:2:end] .+ 1, idi[2:2:end] .+ 1, iv, 2mesh.ndof, 2mesh.ndof)
    DJc = sparse(jdi[1:2:end] .+ 1, jdi[2:2:end] .+ 1, jv, 2mesh.ndof, 2mesh.ndof)
    ic, DIc, jc, DJc
end