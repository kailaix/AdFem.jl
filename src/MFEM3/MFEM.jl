export Mesh3, get_ngauss, get_volume

@doc raw"""
`Mesh` holds data structures for an unstructured mesh. 

- `nodes`: a $n_v \times 2$ coordinates array
- `edges`: a $n_{\text{edge}} \times 2$ integer array for edges 
- `elems`: a $n_e \times 3$ connectivity matrix, 1-based. 
- `nnode`, `nedge`, `nelem`: number of nodes, edges, and elements 
- `ndof`: total number of degrees of freedoms 
- `conn`: connectivity matrix, `nelems × 3` or `nelems × 6`, depending on whether a linear element or a quadratic element is used. 
- `lorder`: order of quadrature rule for line integrals 
- `elem_type`: type of the element (P1, P2 or BDM1)

"""
mutable struct Mesh3
    nodes::Array{Float64, 2}
    edges::Array{Int64, 2}
    faces::Array{Int64, 2}
    elems::Array{Int64, 2}
    nnode::Int64
    nedge::Int64
    nface::Int64
    nelem::Int64
    ndof::Int64
    conn::Array{Int64, 2}
    lorder::Int64
    elem_ndof::Int64
    elem_type::FiniteElementType
end

"""
    Mesh3(coords::Array{Float64, 2}, elems::Array{Int64, 2}, 
        order::Int64 = -1, degree::Union{FiniteElementType,Int64} = 1, lorder::Int64 = -1)

- `degree`: 1 for P1 element, 2 for P2 element 
- `order`: Integration order for elements 
- `lorder`: Integration order for faces
"""
function Mesh3(coords::Array{Float64, 2}, elems::Array{Int64, 2}, 
    order::Int64 = -1, degree::Union{FiniteElementType,Int64} = 1, lorder::Int64 = -1)
    @assert length(size(coords))==2 && size(coords,2)==3
    @assert length(size(elems))==2 && size(elems,2)==4
    if degree==P1 
        degree = 1
    elseif degree==P2 
        degree = 2
    end

    if order==-1
        if degree == 1
            order = 2
        elseif degree == 2
            order = 4
        end
    end
    
    nnode = size(coords, 1)
    nelem = size(elems,1)
    nedges = zeros(Int64, 1)
    c = coords'[:]
    e = Int32.(elems'[:].- 1) 
    edges_ptr = @eval ccall((:init_nnfem_mesh3, $LIBMFEM3), Ptr{Clonglong}, (Ptr{Cdouble}, Cint, 
            Ptr{Cint}, Cint, Cint, Cint, Ptr{Clonglong}), $c, Int32(size($coords, 1)), $e, Int32(size($elems,1)), 
            Int32($order), Int32($degree), $nedges)
    nedges = nedges[1]
    edges = unsafe_wrap(Array{Int64,1}, edges_ptr, (2nedges,), own=true)
    edges = reshape(edges, nedges, 2)
    elem_dof = Int64(@eval ccall((:mfem_get_elem_ndof3, $LIBMFEM3), Cint, ()))
    conn = zeros(Int64, elem_dof * size(elems, 1))
    @eval ccall((:mfem_get_connectivity3, $LIBMFEM3), Cvoid, (Ptr{Clonglong}, ), $conn)
    ndof = Int64(@eval ccall((:mfem_get_ndof3, $LIBMFEM3), Cint, ()))
    conn = reshape(conn, elem_dof, size(elems, 1))'|>Array
    @eval ccall((:mfem_get_element_to_vertices3, $LIBMFEM3), Cvoid, (Ptr{Clonglong}, ), $elems)

    

    elem_type = missing
    if degree==1
        elem_type = P1
    elseif degree==2
        elem_type = P2 
    end

    # get faces 
    fset = Set{Tuple{Int64, Int64, Int64}}([])
    for i = 1:nelem
        cc = elems[i,:]
        push!(fset, Tuple(sort(cc[[1;2;3]])))
        push!(fset, Tuple(sort(cc[[1;2;4]])))
        push!(fset, Tuple(sort(cc[[1;3;4]])))
        push!(fset, Tuple(sort(cc[[2;3;4]])))
    end

    faces = vcat([[x[1] x[2] x[3]] for x in fset]...)
    nfaces = size(faces, 1)
    Mesh3(coords, edges,  faces, elems, nnode, nedges, nfaces, nelem, ndof, conn, lorder, elem_dof, elem_type)
end

Base.:copy(mesh::Mesh3) = Mesh3(copy(mesh.nodes),
                            copy(mesh.edges),
                            copy(mesh.faces),
                            copy(mesh.elems),
                            copy(mesh.nnode),
                            copy(mesh.nedge),
                            copy(mesh.nelem),
                            copy(mesh.nface),
                            copy(mesh.ndof),
                            copy(mesh.conn),
                            copy(mesh.lorder),
                            copy(mesh.elem_ndof),
                            copy(mesh.elem_type))

@doc raw"""
    Mesh3(m::Int64, n::Int64, l::Int64, 
        h::Float64; 
        order::Int64 = 2, 
        degree::Union{FiniteElementType, Int64} = 1, 
        lorder::Int64 = -1)

Constructs a mesh of a rectangular domain. The rectangle is split into $m\times n$ cells, and each cell is further split into two triangles. 
`order` specifies the quadrature rule order. `degree` determines the degree for finite element basis functions.
"""
function Mesh3(m::Int64, n::Int64, l::Int64, h::Float64; order::Int64 = -1, 
            degree::Union{FiniteElementType, Int64} = 1, lorder::Int64 = -1)
    coords = zeros((m+1)*(n+1)*(l+1), 3)
    elems = zeros(Int64, 5*m*n*l, 4)
    function ID(i, j, k)
        (k-1)*(n+1)*(m+1) + (j-1)*(m+1) + i 
    end
    TE1 = [
        [1; 2; 3; 5],
        [2; 3; 4; 8],
        [3; 5; 7; 8],
        [2; 3; 5; 8],
        [2; 5; 6; 8]
    ]
    TE2 = [
        [1; 2; 4; 6],
        [1; 5; 6; 7],
        [4; 6; 7; 8],
        [1; 4; 6; 7],
        [1; 3; 4; 7]
    ]
    
    s = 1
    for k = 1:l+1
        for j = 1:m+1 
            for i = 1:n+1
                x = (i-1)*h 
                y = (j-1)*h 
                z = (k-1)*h 
                coords[s,:] = [x;y;z]
                s += 1
            end
        end
    end
    K = 0
    for i = 1:n
        for j = 1:m 
            for k = 1:l 
                IDX = [
                    ID(i, j, k)
                    ID(i+1, j, k)
                    ID(i, j+1, k)
                    ID(i+1, j+1, k)
                    ID(i, j, k+1)
                    ID(i+1, j, k+1)
                    ID(i, j+1, k+1)
                    ID(i+1, j+1, k+1)
                ]
                for s = 1:5
                    if (i+j+k)%2==0
                        elems[s + K,:] = IDX[TE1[s]]
                    else
                        elems[s + K,:] = IDX[TE2[s]]
                    end
                end
                K += 5
            end
        end
    end
   
    Mesh3(coords, elems, order, degree, lorder)
end


"""
    Mesh3(filename::String; file_format::Union{String, Missing} = missing, 
    order::Int64 = 2, degree::Union{FiniteElementType, Int64} = 1, lorder::Int64 = 2)
"""
function Mesh3(filename::String; file_format::Union{String, Missing} = missing, 
    order::Int64 = 2, degree::Union{FiniteElementType, Int64} = 1, lorder::Int64 = 2)
    if splitext(filename)[2] == ".mat"
        d = matread(filename)
        return Mesh3(Float64.(d["nodes"]), Int64.(d["elems"]), order, degree, lorder)
    end
    meshio = get_meshio()
    if !ismissing(file_format)
        mesh = meshio.read(filename, file_format = file_format)
    else
    mesh = meshio.read(filename)
    end
    elem = []
    for (mkr, dat) in mesh.cells
        if mkr == "tetra"
            push!(elem, dat)
        end
    end
    elem = vcat(elem...)
    if length(elem)==0
        error("No triangles found in the mesh file.")
    end
    Mesh3(Float64.(mesh.points), Int64.(elem) .+ 1, order, degree, lorder)
end


# """
#     Mesh(; order::Int64 = -1, 
#     degree::Union{FiniteElementType, Int64} = 1, lorder::Int64 = -1)

# Creates a mesh with a reference triangle. 

# ![](https://raw.githubusercontent.com/ADCMEMarket/ADCMEImages/master/AdFem/mapping.png)

# """
# function Mesh(; order::Int64 = -1, 
#     degree::Union{FiniteElementType, Int64} = 1, lorder::Int64 = -1)
#     coords = [
#         1.0 0.0
#         0.0 1.0
#         0.0 0.0
#     ]
#     elems = [
#         1 2 3
#     ]
#     Mesh(coords, elems, order, degree, lorder)
# end

"""
    get_ngauss(mesh::Mesh3) 

Return the total number of Gauss points. 
"""
function get_ngauss(mesh::Mesh3)
    return Int64(@eval ccall((:mfem_get_ngauss3, $LIBMFEM3), Cint, ()))
end

"""
    get_volume(mesh::Mesh3) 

Return the areas of triangles as an array. 
"""
function get_volume(mesh::Mesh3)
    a = zeros(size(mesh.elems,1))
    @eval ccall((:mfem_get_area3, $LIBMFEM3), Cvoid, (Ptr{Cdouble}, ), $a)
    a
end


"""
    gauss_nodes(mesh::Mesh3)
"""
function gauss_nodes(mesh::Mesh3)
    ngauss = get_ngauss(mesh)
    x = zeros(ngauss)
    y = zeros(ngauss)
    z = zeros(ngauss)
    @eval ccall((:mfem_get_gauss3, $LIBMFEM3), Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}), $x, $y, $z)
    [x y z]
end

"""
    fem_nodes(mesh::Mesh3)
"""
function fem_nodes(mesh::Mesh3)
    mesh.nodes
end

"""
    fvm_nodes(mesh::Mesh3)
"""
function fvm_nodes(mesh::Mesh3)
    nnode = size(mesh.nodes, 1)
    nelem = size(mesh.elems, 1)
    out = zeros(nelem, 2)
    for i = 1:nelem
        idx = mesh.elems[i, :]
        out[i, 1] = mean(mesh.nodes[idx, 1])
        out[i, 2] = mean(mesh.nodes[idx, 2])
    end
    return out
end

function _edge_dict(mesh::Mesh3)
    D = Dict{Tuple{Int64, Int64}, Int64}()
    for i = 1:mesh.nedge 
        D[(mesh.edges[i,1], mesh.edges[i,2])] = i 
        D[(mesh.edges[i,2], mesh.edges[i,1])] = i 
    end
    D
end