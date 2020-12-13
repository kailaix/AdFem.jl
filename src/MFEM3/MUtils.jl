export bcface, bcnode

@doc raw"""
    bcface(mmesh::Mesh3)

Returns the boundary faces as a $n_b \times 3$ matrix. Each row is the three vertices for the triangle face. 
"""
function bcface(mmesh::Mesh3)
    fdict = Dict{Tuple{Int64, Int64, Int64}, Int64}()
    for i = 1:mmesh.nface
        fdict[(mmesh.faces[i,1], mmesh.faces[i,2], mmesh.faces[i,3])] = i 
    end 
    bface = Dict([i=>0 for i = 1:mmesh.nface])
    for i = 1:mmesh.nelem
        el = mmesh.elems[i,:]
        bface[fdict[Tuple(sort(el[[1;2;3]]))]] += 1
        bface[fdict[Tuple(sort(el[[1;2;4]]))]] += 1
        bface[fdict[Tuple(sort(el[[1;3;4]]))]] += 1
        bface[fdict[Tuple(sort(el[[2;3;4]]))]] += 1
    end
    bf = []
    for k = 1:mmesh.nface
        if bface[k] == 1
            push!(bf, mmesh.faces[k,:])
        end
    end
    Array(hcat(bf...)')
end

"""
    bcnode(mmesh::Mesh3)
"""
function bcnode(mmesh::Mesh3)
    bf = bcface(mmesh)
    s = Set(bf[:])
    return [x for x in s]
end