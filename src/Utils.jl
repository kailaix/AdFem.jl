export femidx, fvmidx, get_edge_normal
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