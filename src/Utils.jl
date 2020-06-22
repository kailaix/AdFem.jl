export femidx, fvmidx, get_edge_normal,
plot_u,bcnode,bcedge, layer_model, gauss_nodes, fem_nodes, fvm_nodes, subdomain, interior_node,
cholesky_outproduct, cholesky_factorize, cholesky_logdet, fem_randidx, gauss_randidx, fem_to_fvm


"""
    femidx(d::Int64, m::Int64)

Returns the FEM index of the dof `d`. Basically, `femidx` is the inverse of 

```
(i,j) → d = (j-1)*(m+1) + i
```
"""
function femidx(i::Int64, m::Int64)
    ii = mod(i, m+1)
    ii = ii == 0 ? m+1 : ii 
    jj = div(i-1, m+1) + 1
    return ii, jj
end

"""
    fvmidx(d::Int64, m::Int64)

Returns the FVM index of the dof `d`. Basically, `femidx` is the inverse of 

```
(i,j) → d = (j-1)*m + i
```
"""
function fvmidx(i::Int64, m::Int64)
    ii = mod(i, m)
    ii = ii == 0 ? m : ii 
    jj = div(i-1, m) + 1
    return ii, jj
end

function plot_u(u::Array{Float64}, m::Int64, n::Int64, h::Float64)
    x = LinRange(0,m*h,m+1)
    y = LinRange(0,n*h,n+1)
    X, Y = np.meshgrid(x, y)
    U = reshape(u[1:(m+1)*(n+1)], m+1, n+1)'|>Array
    V = reshape(u[(m+1)*(n+1)+1:end], m+1, n+1)'|>Array
    X, Y, U, V
end

@doc raw"""
    layer_model(u::Array{Float64, 1}, m::Int64, n::Int64, h::Float64)

Convert the vertical profile of a quantity to a layer model. 
The input `u` is a length $n$ vector, the output is a length $4mn$ vector, representing the $4mn$ Gauss points. 
"""
function layer_model(u::Array{Float64, 1}, m::Int64, n::Int64, h::Float64)
    repeat(reshape(u, :, 1), 1, 4m)'[:]
end


@doc raw"""
    layer_model(u::PyObject, m::Int64, n::Int64, h::Float64)

A differential kernel for [`layer_model`](@ref). 
"""
function layer_model(u::PyObject, m::Int64, n::Int64, h::Float64)
    reshape(repeat(u, 1, 4m), (-1,))
end


"""
    get_edge_normal(edge::Array{Int64,1}, m::Int64, n::Int64, h::Float64)   

Returns the normal vector given edge `edge`.
"""
function get_edge_normal(edge::Array{Int64,1}, m::Int64, n::Int64, h::Float64)
    i1, j1 = femidx(edge[1], m)
    i2, j2 = femidx(edge[2], m)
    if i1==i2==1
        return [-1.0;0.0]
    elseif i1==i2==m+1
        return [1.0;0.0]
    elseif j1==j2==1
        return [0.0;-1.0]
    elseif j1==j2==n+1
        return [0.0;1.0]
    else
        error("Invalid edge index $edge")
    end
end

function get_edge_normal(edges::Array{Int64,2}, m::Int64, n::Int64, h::Float64)
    out = zeros(size(edges,1), 2)
    for i = 1:size(edges,1)
        out[i,:] = get_edge_normal(edges[i,:], m, n, h)
    end
end

"""
    bcedge(desc::String, m::Int64, n::Int64, h::Float64)

Returns the edge indices for description. See [`bcnode`](@ref)
"""
function bcedge(desc::String, m::Int64, n::Int64, h::Float64)
    desc=="all" && (desc="left|right|upper|lower")
    descs = strip.(split(desc, '|'))
    edges = bcedge_.(descs, m, n, h)
    unique(vcat(edges...), dims=1)
end



function bcedge_(desc::AbstractString, m::Int64, n::Int64, h::Float64)
    nodes = bcnode_(desc, m, n, h)
    [nodes[1:end-1] nodes[2:end]]
end



"""
    bcnode(desc::String, m::Int64, n::Int64, h::Float64)

Returns the node indices for the description. Multiple descriptions can be concatented via `|`

```
                upper
        |------------------|
left    |                  | right
        |                  |
        |__________________|

                lower
```

# Example
```julia
bcnode("left|upper", m, n, h)
```
"""
function bcnode(desc::String, m::Int64, n::Int64, h::Float64)
    desc=="all" && (desc="left|right|upper|lower")
    descs = strip.(split(desc, '|'))
    nodes = bcnode_.(descs, m, n, h)
    unique(vcat(nodes...))
end

function bcnode_(desc::AbstractString, m::Int64, n::Int64, h::Float64)
    nodes = Int64[]
    if desc=="upper" || desc=="top"
        for i = 1:m+1
            push!(nodes, i)
        end
    elseif desc=="lower" || desc == "bottom"
        for i = 1:m+1
            push!(nodes, i+n*(m+1))
        end
    elseif desc=="left"
        for j = 1:n+1
            push!(nodes, 1+(j-1)*(m+1))
        end
    elseif desc=="right"
        for j = 1:n+1
            push!(nodes, m+1+(j-1)*(m+1))
        end
    else 
        error("$desc is not a valid specification. Only `upper` (top), `lower` (bottom), `left`, `right`, `all` or their combination via `|`, is accepted.")
    end
    return nodes
end

"""
    interior_node(desc::String, m::Int64, n::Int64, h::Float64)

In contrast to [`bcnode`](@ref), `interior_node` returns the nodes that are not specified by `desc`, including thosee on the boundary.
"""
function interior_node(desc::String, m::Int64, n::Int64, h::Float64)
    bc = bcnode(desc, m, n, h)
    collect(setdiff(Set(1:(m+1)*(n+1)), bc))
end


"""
    fem_to_fvm(u::Union{PyObject, Array{Float64}}, m::Int64, n::Int64, h::Float64)

Interpolates the nodal values of `u` to cell values. 
"""
function fem_to_fvm(u::Union{PyObject, Array{Float64}}, m::Int64, n::Int64, h::Float64)
    idx1 = zeros(Int64, m*n)
    idx2 = zeros(Int64, m*n)
    idx3 = zeros(Int64, m*n)
    idx4 = zeros(Int64, m*n)
    for i = 1:m 
        for j = 1:n 
            idx1[(j-1)*m+i] = (j-1)*(m+1)+i
            idx2[(j-1)*m+i] = (j-1)*(m+1)+i+1
            idx3[(j-1)*m+i] = j*(m+1)+i
            idx4[(j-1)*m+i] = j*(m+1)+i+1
        end
    end
    (u[idx1] + u[idx2] + u[idx3] + u[idx4])/4
end

@doc raw"""
    gauss_nodes(m::Int64, n::Int64, h::Float64)

Returns the node matrix of Gauss points for all elements. The matrix has a size $4mn\times 2$
"""
function gauss_nodes(m::Int64, n::Int64, h::Float64)
    xs = zeros(4*m*n, 2)
    k = 1
    for i = 1:m 
        for j = 1:n 
            idx = (j-1)*m + i 
            x1 = (i-1)*h 
            y1 = (j-1)*h
            for p = 1:2
                for q = 1:2
                    k = (idx-1)*4 + 2*(q-1) + p
                    ξ = pts[p]; η = pts[q]
                    x = x1 + ξ*h; y = y1 + η*h
                    xs[k,:] = [x;y]
                    k += 1
                end
            end
        end
    end
    xs
end

@doc raw"""
    fem_nodes(m::Int64, n::Int64, h::Float64)

Returns the FEM node matrix of size $(m+1)(n+1)\times 2$
"""
function fem_nodes(m::Int64, n::Int64, h::Float64)
    xs = zeros((m+1)*(n+1), 2)
    k = 1
    for i = 1:m+1
        for j = 1:n+1
            idx = (j-1)*(m+1) + i 
            x1 = (i-1)*h 
            y1 = (j-1)*h
            xs[idx,:] = [x1;y1]
            k += 1
        end
    end
    xs
end

@doc raw"""
    fvm_nodes(m::Int64, n::Int64, h::Float64)

Returns the FVM node matrix of size $(m+1)(n+1)\times 2$
"""
function fvm_nodes(m::Int64, n::Int64, h::Float64)
    xs = zeros(m*n, 2)
    k = 1
    for i = 1:m
        for j = 1:n
            idx = (j-1)*m + i 
            x1 = (i-1/2)*h 
            y1 = (j-1/2)*h
            xs[idx,:] = [x1;y1]
            k += 1
        end
    end
    xs
end

@doc raw"""
    fem_randidx(N::Int64, m::Int64, n::Int64, h::Float64)

Returns $N$ random index 
"""
function fem_randidx(N::Int64, m::Int64, n::Int64, h::Float64)
    idx = Set([])
    for k = 1:N 
        while true
            i = rand(2:m)
            j = rand(2:n)
            if j*(m+1)+i in idx 
                continue 
            else
                push!(idx, j*(m+1)+i)
                break 
            end
        end
    end
    sort([k for k in idx])
end

"""
    subdomain(f::Function, m::Int64, n::Int64, h::Float64)

Returns the subdomain defined by `f(x, y)==true`.
"""
function subdomain(f::Function, m::Int64, n::Int64, h::Float64)
    nodes = fem_nodes(m, n, h)
    idx = f.(nodes[:,1], nodes[:,2])
    findall(idx)
end

@doc raw"""
    cholesky_outproduct(L::Union{Array{<:Real,2}, PyObject})

Returns 
$$A = LL'$$
where `L` (length=6) is a vectorized form of $L$
$$L = \begin{matrix}
l_1 & 0 & 0\\ 
l_4 & l_2 & 0 \\ 
l_5 & l_6 & l_3
\end{matrix}$$
and `A` (length=9) is also a vectorized form of $A$
"""
function cholesky_outproduct(A::Union{Array{<:Real,2}, PyObject})
    @assert size(A,2)==6
    op_ = load_op_and_grad("$(@__DIR__)/../deps/build/libporeflow","cholesky_backward_op")
    A = convert_to_tensor([A], [Float64]); A = A[1]
    L = op_(A)
end

@doc raw"""
    cholesky_factorize(A::Union{Array{<:Real,2}, PyObject})

Returns the cholesky factor of `A`. See [`cholesky_outproduct`](@ref) for details. 
"""
function cholesky_factorize(A::Union{Array{<:Real,2}, PyObject})
    @assert size(A,2)==9
    op_ = load_op_and_grad("$(@__DIR__)/../deps/build/libporeflow","cholesky_forward_op")
    A = convert_to_tensor([A], [Float64]); A = A[1]
    L = op_(A)
end


@doc raw"""
    cholesky_logdet(A::Union{Array{<:Real,2}, PyObject})

Returns the cholesky factor of `A` as well as the log determinant. See [`cholesky_outproduct`](@ref) for details. 
"""
function cholesky_logdet(A::Union{Array{<:Real,2}, PyObject})
    op_ = load_op_and_grad("$(@__DIR__)/../depsCholeskyOp/build/libCholeskyOp/build","cholesky_logdet")
    A = convert_to_tensor(A, dtype=Float64)
    L, J = op_(A)
end