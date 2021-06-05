export CrackMesh
@doc raw"""
    CrackMesh(m::Int64, n::Int64, h::Float64, k::Int64 = 1)

Creates a crack mesh. 

```julia
mmesh = CrackMesh(20, 10, 0.1, 4)
visualize_mesh(mmesh)
```

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/AdFem/crackmesh.PNG?raw=true)

To access the underlying [`Mesh`](@ref) object, use `mmesh.mesh`.
"""
mutable struct CrackMesh
    m::Int64 
    n::Int64 
    h::Float64
    k::Int64
    lower::Array{Int64,1}
    upper::Array{Int64,1}
    bdedge::Array{Int64,2}
    mesh::Mesh
    function CrackMesh(m::Int64, n::Int64, h::Float64, k::Int64 = 1)
        @assert mod(n, 2)==0
        @assert 2*k<=m
        coords = zeros((m+1)*(n+1)+m+1-2k,2)
        elem = zeros(Int64, 2m*n,3)
        s = 1
        for j = 1:n+1
            for i = 1:m+1
                x = (i-1)*h 
                y = (j-1)*h 
                coords[s, :] = [x;y]
                s += 1
            end
        end
        lower = [n÷2*(m+1)+k]
        upper = [n÷2*(m+1)+k]
        for i = k+1:m+1-k
            x = (i-1)*h 
            y = n/2*h
            coords[s, :] = [x;y]
            s += 1 
            push!(lower, n÷2*(m+1)+i)
            push!(upper, i-k+(m+1)*(n+1))
        end
        push!(lower, lower[end]+1)
        push!(upper, lower[end]+1)

        s = 1
        for i = 1:m
            for j = 1:n
                if j==(n÷2)+1 && i==k
                    elem[s,:] = [j*(m+1)+i; (j-1)*(m+1)+i; 1 + (m+1)*(n+1)]
                    elem[s+1,:] = [j*(m+1)+i; 1 + (m+1)*(n+1); j*(m+1)+i+1]
                elseif j==(n÷2)+1 && i==m-k+1
                    elem[s,:] = [j*(m+1)+i; m+1-2k+(m+1)*(n+1); (j-1)*(m+1)+i+1]
                    elem[s+1,:] = [j*(m+1)+i; (j-1)*(m+1)+i+1; j*(m+1)+i+1]
                elseif j==(n÷2)+1 && (i>k) && (i<m-k+1)
                    elem[s,:] = [j*(m+1)+i; i-k + (m+1)*(n+1); i-k+1 + (m+1)*(n+1)]
                    elem[s+1,:] = [j*(m+1)+i; i-k+1 + (m+1)*(n+1); j*(m+1)+i+1]
                else 
                    elem[s,:] = [j*(m+1)+i; (j-1)*(m+1)+i; (j-1)*(m+1)+i+1]
                    elem[s+1,:] = [j*(m+1)+i; (j-1)*(m+1)+i+1; j*(m+1)+i+1]
                end 
                s += 2
            end
        end

        mmesh = Mesh(m, n, h)
        bdedge = bcedge(mmesh)
        mmesh = Mesh(coords, elem, -1, 1, -1)
        new(m, n, h, k, lower, upper, bdedge, mmesh)
    end
end

function visualize_mesh(mmesh::CrackMesh)
    m, n, h = mmesh.m, mmesh.n, mmesh.h
    mmesh2 = copy(mmesh.mesh)
    mmesh2.nodes[(m+1)*(n+1)+1:end,2] .+= 0.3*h
    visualize_mesh(mmesh2)
end