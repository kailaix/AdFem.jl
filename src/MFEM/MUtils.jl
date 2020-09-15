export PDATA 

"""
    PDATA

The folder where built-in meshes are stored. 
"""
PDATA = joinpath(@__DIR__, "..", "..",  "deps", "MFEM", "MeshData")

"""
    Mesh(stlfile::String)

Reads a `stl` file and extracts element, coordinates and boundaries.

# Example
```julia
mesh = Mesh(joinpath(PDATA, "twoholes.stl"))
```
"""
function Mesh(stlfile::String)
    cnt = strip.(readlines(stlfile))
    nodes_ = Set{Tuple{Float64, Float64}}()
    nelems = 0
    for i = 1:length(cnt)
        c = cnt[i]
        if length(c)>6 && c[1:6]=="vertex"
            r = split(c)
            push!(nodes_, (parse(Float64, r[2]), parse(Float64, r[3])))
            nelems += 1
        end
    end
    n = length(nodes_)
    nodes = zeros(n, 2)
    nodes_dict = Dict{Tuple{Float64, Float64}, Int64}()
    for (k, nd) in enumerate(nodes_)
        nodes_dict[nd] = k
        nodes[k,:] = [nd[1]; nd[2]]
    end
    elems = zeros(Int64, nelems)
    k = 1
    for i = 1:length(cnt)
        c = cnt[i]
        if length(c)>6 && c[1:6]=="vertex"
            r = split(c)
            elems[k] = nodes_dict[(parse(Float64, r[2]), parse(Float64, r[3]))]
            k = k + 1
        end
    end
    elems = reshape(elems, 3, :)'|>Array
    return Mesh(nodes, elems)
end