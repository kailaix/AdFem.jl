
export fast_αscheme
function fast_αscheme(
    m::Int64, n::Int64, h::Float64,
    M::Union{SparseTensor, SparseMatrixCSC}, 
    C::Union{SparseTensor, SparseMatrixCSC}, 
    K::Union{SparseTensor, SparseMatrixCSC}, 
    body_force::Union{Array{Float64}, PyObject}, 
    d0::Union{Array{Float64, 1}, PyObject}, 
    v0::Union{Array{Float64, 1}, PyObject}, 
    a0::Union{Array{Float64, 1}, PyObject}, 
    Δt::Float64,
    bdnode::Union{Array{Int64, 1}, PyObject, Missing} = missing, 
    bd_d::Union{Array{Float64, 2}, PyObject, Missing} = missing, 
    bd_v::Union{Array{Float64, 2}, PyObject, Missing} = missing, 
    bd_a::Union{Array{Float64, 2}, PyObject, Missing} = missing;
    ρ::Float64 = 1.0)
    nt = size(body_force, 1)
    if !ismissing(bdnode)
        if ismissing(bd_a)
            error("Boundary acceleration at Dirichlet nodes must be provided.")
        end
        for v in [bd_d, bd_v, bd_a]
            if !ismissing(v) && (size(v,1)!= nt || size(v,2)!= 2length(bdnode))
                error("Invalid shape for $v. Expected $(nt)×$(2length(bdnode)) but got $(size(v,1))×$(size(v,2))")
            end
        end
    end
    !ismissing(bd_d) && (bd_d = convert_to_tensor(bd_d, dtype=Float64))
    !ismissing(bd_v) && (bd_v = convert_to_tensor(bd_v, dtype=Float64))
    !ismissing(bd_a) && (bd_a = convert_to_tensor(bd_a, dtype=Float64))
    
    αm = (2ρ-1)/(ρ+1)
    αf = ρ/(1+ρ)
    γ = 1/2-αm+αf 
    β = 0.25*(1-αm+αf)^2
    d = length(d0)

    M = isa(M, SparseMatrixCSC) ? constant(M) : M
    C = isa(C, SparseMatrixCSC) ? constant(C) : C
    K = isa(K, SparseMatrixCSC) ? constant(K) : K
    body_force, d0, v0, a0, Δt = convert_to_tensor([body_force, d0, v0, a0, Δt], [Float64, Float64, Float64, Float64, Float64])

    A = (1-αm)*M + (1-αf)*C*Δt*γ + (1-αf)*K*β*Δt^2
    if !ismissing(bdnode)
        A, Abd = fem_impose_Dirichlet_boundary_condition(A, bdnode, m, n, h)
    else
        Abd = missing 
    end
    A = factorize(A)

    function equ(dc, vc, ac, dt, body_force, i)
        dn = dc + dt*vc + dt^2/2*(1-2β)*ac 
        vn = vc + dt*((1-γ)*ac)

        df = (1-αf)*dn + αf*dc
        vf = (1-αf)*vn + αf*vc 
        am = αm*ac 

        rhs = body_force - (M*am + C*vf + K*df)
        if !ismissing(Abd)
            rhs -= Abd * bd_a[i]
            rhs = scatter_update(rhs, [bdnode; bdnode .+ (m+1)*(n+1)], bd_a[i]) 
        end

        A\rhs
    end

    function condition(i, tas...)
        return i<=nt
    end
    function body(i, tas...)
        dc_arr, vc_arr, ac_arr = tas
        dc = read(dc_arr, i)
        vc = read(vc_arr, i)
        ac = read(ac_arr, i)
        y = equ(dc, vc, ac, Δt, body_force[i], i)
        dn = dc + Δt*vc + Δt^2/2*((1-2β)*ac+2β*y)
        vn = vc + Δt*((1-γ)*ac+γ*y)

        if !ismissing(bd_v)
            vn = scatter_update(vn, [bdnode; bdnode .+ (m+1)*(n+1)], bd_v[i]) 
        end

        if !ismissing(bd_d)
            dn = scatter_update(dn, [bdnode; bdnode .+ (m+1)*(n+1)], bd_d[i]) 
        end

        i+1, write(dc_arr, i+1, dn), write(vc_arr, i+1, vn), write(ac_arr, i+1, y)
    end

    dM = TensorArray(nt+1); vM = TensorArray(nt+1); aM = TensorArray(nt+1)
    dM = write(dM, 1, d0)
    vM = write(vM, 1, v0)
    aM = write(aM, 1, a0)
    i = constant(1, dtype=Int32)
    _, d, v, a = while_loop(condition, body, [i,dM, vM, aM])
    set_shape(stack(d), (nt+1, length(d0))), set_shape(stack(v), (nt+1, length(v0))), set_shape(stack(a), (nt+1, length(a0)))
end