export femidx, fvmidx, get_edge_normal,
plot_u,bcnode,bcedge
function femidx(i, m)
    ii = mod(i, m+1)
    ii = ii == 0 ? m+1 : ii 
    jj = div(i-1, m+1) + 1
    return ii, jj
end

function fvmidx(i, m)
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
    nodes = []
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


