
function get_face_dof(bdface, mmesh)
end

function bcface(mmesh::Mesh3)
    fdict = Dict{Tuple{Int64, Int64, Int64}, Int64}()
    for i = 1:mmesh.nface
        fdict[(fdict[i,1], fdict[i,2], fdict[i,3])] = i 
    end 
    bface = Dict([i=>0 for i = 1:mmesh.nface])
    for i = 1:mmesh.nelem
        el = mmesh.elems[i,:]
        bface[Tuple(el[[1;2;3]])] += 1
        bface[Tuple(el[[1;2;4]])] += 1
        bface[Tuple(el[[1;3;4]])] += 1
        bface[Tuple(el[[2;3;4]])] += 1
    end
end

function bcnode(mmesh::Mesh3)
    bdedge = bcedge(mmesh)
    if by_dof && mmesh.elem_type == P2
        edgedof = get_edge_dof(bdedge, mmesh) .+ mmesh.nnode
        [collect(Set(bdedge[:])); edgedof]
    elseif by_dof && mmesh.elem_type == BDM1
        bd = get_edge_dof(bdedge, mmesh) 
        [bd; bd .+ mmesh.nedge]
    else
        collect(Set(bdedge[:]))
    end
end