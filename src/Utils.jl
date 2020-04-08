export femidx, fvmidx, get_edge_normal,
plot_u,bcnode,bcedge, layer_model


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
Abstract    bcnode(desc::String, m::Int64, n::Int64, h::Float64)

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
    if desc=="upper"
        for i = 1:m+1
            push!(nodes, i)
        end
    elseif desc=="lower"
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
    end
    return nodes
end


