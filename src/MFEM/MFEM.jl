export Mesh, get_ngauss, get_area, Mesh_v2, Mesh_v3

macro exported_enum(name, args...)
    esc(quote
        @enum($name, $(args...))
        export $name
        $([:(export $arg) for arg in args]...)
        end)
end

@exported_enum FiniteElementType P1 P2 BDM1

@doc raw"""
`Mesh` holds data structures for an unstructured mesh. 

- `nodes`: a $n_v \times 2$ coordinates array
- `edges`: a $n_{\text{edge}} \times 2$ integer array for edges 
- `elems`: a $n_e \times 3$ connectivity matrix, 1-based. 
- `nnode`, `nedge`, `nelem`: number of nodes, edges, and elements 
- `ndof`: total number of degrees of freedoms 
- `conn`: connectivity matrix, `nelems × 3` or `nelems × 6`, depending on whether a linear element or a quadratic element is used. 
- `lorder`: order of quadrature rule for line integrals (default = 6, 4 gauss points per line segment)
- `elem_type`: type of the element (P1, P2 or BDM1)

# Constructors 

```
Mesh(m::Int64, n::Int64, h::Float64; order::Int64 = -1, 
            degree::Union{FiniteElementType, Int64} = 1, lorder::Int64 = -1, 
            version::Int64 = 1)
```

Constructs a mesh of a rectangular domain. The rectangle is split into $m\times n$ cells, and each cell is further split into two triangles. 
`order` specifies the quadrature rule order. `degree` determines the degree for finite element basis functions.

!!! info 
    AdFem provides three types of triangulations for a rectangular domain. The different types of meshes can be used to validate numerical schemes.
    For example, we can change to different meshes to verify that bugs of our program do not originate from mesh types. 
    
    ![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/AdFem/mesh_types.png?raw=true)

# Integration Order : `order` and `lorder`

TODO
"""
mutable struct Mesh
    nodes::Array{Float64, 2}
    edges::Array{Int64, 2}
    elems::Array{Int64, 2}
    nnode::Int64
    nedge::Int64
    nelem::Int64
    ndof::Int64
    conn::Array{Int64, 2}
    lorder::Int64
    elem_ndof::Int64
    elem_type::FiniteElementType
end

function Mesh(coords::Array{Float64, 2}, elems::Array{Int64, 2}, order::Int64 = -1, 
        degree::Union{FiniteElementType,Int64} = 1, lorder::Int64 = -1)
    if degree==P1 
        degree = 1
    elseif degree==P2 
        degree = 2
    end
    if !(degree in [1, 2]) && degree!=BDM1
        error("Only degree = 1 or 2 is supported.")
    end

    if order==-1
        if degree == 1 || degree==BDM1
            order = 2
        elseif degree == 2
            order = 4
        end
    end

    if lorder==-1
        if degree == 1 || degree==BDM1
            lorder = 6
        elseif degree == 2
            lorder = 6
        end
    end

    if degree==BDM1 
        degree = -1
    end
    nnode = size(coords, 1)
    nelem = size(elems,1)
    nedges = zeros(Int64, 1)
    c = [coords zeros(size(coords, 1))]'[:]
    e = Int32.(elems'[:].- 1) 
    edges_ptr = @eval ccall((:init_nnfem_mesh, $LIBMFEM), Ptr{Clonglong}, (Ptr{Cdouble}, Cint, 
            Ptr{Cint}, Cint, Cint, Cint, Cint, Ptr{Clonglong}), $c, Int32(size($coords, 1)), $e, Int32(size($elems,1)), 
            Int32($order), Int32($lorder), Int32($degree), $nedges)
    nedges = nedges[1]
    edges = unsafe_wrap(Array{Int64,1}, edges_ptr, (2nedges,), own=true)
    edges = reshape(edges, nedges, 2)
    elem_dof = Int64(@eval ccall((:mfem_get_elem_ndof, $LIBMFEM), Cint, ()))
    conn = zeros(Int64, elem_dof * size(elems, 1))
    @eval ccall((:mfem_get_connectivity, $LIBMFEM), Cvoid, (Ptr{Clonglong}, ), $conn)
    ndof = Int64(@eval ccall((:mfem_get_ndof, $LIBMFEM), Cint, ()))
    conn = reshape(conn, elem_dof, size(elems, 1))'|>Array
    @eval ccall((:mfem_get_element_to_vertices, $LIBMFEM), Cvoid, (Ptr{Clonglong}, ), $elems)

    

    elem_type = missing
    if degree==1
        elem_type = P1
    elseif degree==2
        elem_type = P2 
    elseif degree==-1
        elem_type = BDM1
    end
    Mesh(coords, edges,  elems, nnode, nedges, nelem, ndof, conn, lorder, elem_dof, elem_type)
end

Base.:copy(mesh::Mesh) = Mesh(copy(mesh.nodes),
                            copy(mesh.edges),
                            copy(mesh.elems),
                            copy(mesh.nnode),
                            copy(mesh.nedge),
                            copy(mesh.nelem),
                            copy(mesh.ndof),
                            copy(mesh.conn),
                            copy(mesh.lorder),
                            copy(mesh.elem_ndof),
                            mesh.elem_type)


function Mesh(m::Int64, n::Int64, h::Float64; order::Int64 = -1, 
            degree::Union{FiniteElementType, Int64} = 1, lorder::Int64 = -1, 
            version::Int64 = 1)
    coords = zeros((m+1)*(n+1), 2)
    elems = zeros(Int64, 2*m*n, 3)
    for i = 1:n 
        for j = 1:m 
            e = 2*((i-1)*m + j - 1)+1
            if version == 1
                elems[e, :] = [(i-1)*(m+1)+j; (i-1)*(m+1)+j+1; i*(m+1)+j ]
                elems[e+1, :] = [(i-1)*(m+1)+j+1; i*(m+1)+j; i*(m+1)+j+1]
            elseif version == 2
                elems[e, :] = [(i-1)*(m+1)+j; i*(m+1)+j+1; i*(m+1)+j ]
                elems[e+1, :] = [(i-1)*(m+1)+j; (i-1)*(m+1)+j+1; i*(m+1)+j+1]
            elseif version == 3
                if rand()>0.5
                    elems[e, :] = [(i-1)*(m+1)+j; (i-1)*(m+1)+j+1; i*(m+1)+j ]
                    elems[e+1, :] = [(i-1)*(m+1)+j+1; i*(m+1)+j; i*(m+1)+j+1]
                else
                    elems[e, :] = [(i-1)*(m+1)+j; i*(m+1)+j+1; i*(m+1)+j ]
                    elems[e+1, :] = [(i-1)*(m+1)+j; (i-1)*(m+1)+j+1; i*(m+1)+j+1]
                end
            end

        end
    end
    k = 1
    for i = 1:n+1
        for j = 1:m+1
            x = (j-1)*h 
            y = (i-1)*h
            coords[k, :] = [x;y]
            k += 1
        end
    end
    Mesh(coords, elems, order, degree, lorder)
end

Mesh_v2(args...;kwargs...) = Mesh(args...; version = 2, kwargs...)
Mesh_v3(args...;kwargs...) = Mesh(args...; version = 3, kwargs...)


"""
    Mesh(; order::Int64 = -1, 
    degree::Union{FiniteElementType, Int64} = 1, lorder::Int64 = -1)

Creates a mesh with a reference triangle. 

![](https://raw.githubusercontent.com/ADCMEMarket/ADCMEImages/master/AdFem/mapping.png)

"""
function Mesh(; order::Int64 = -1, 
    degree::Union{FiniteElementType, Int64} = 1, lorder::Int64 = -1)
    coords = [
        1.0 0.0
        0.0 1.0
        0.0 0.0
    ]
    elems = [
        1 2 3
    ]
    Mesh(coords, elems, order, degree, lorder)
end

function Base.:length(mesh::Mesh)
    return maximum(mesh.conn)
end

"""
    get_ngauss(mesh::Mesh) 

Return the total number of Gauss points. 
"""
function get_ngauss(mesh::Mesh)
    return Int64(@eval ccall((:mfem_get_ngauss, $LIBMFEM), Cint, ()))
end

"""
    get_ngauss(mesh::Mesh) 

Return the areas of triangles as an array. 
"""
function get_area(mesh::Mesh)
    a = zeros(size(mesh.elems,1))
    @eval ccall((:mfem_get_area, $LIBMFEM), Cvoid, (Ptr{Cdouble}, ), $a)
    a
end


"""
    gauss_nodes(mesh::Mesh)
"""
function gauss_nodes(mesh::Mesh)
    ngauss = get_ngauss(mesh)
    x = zeros(ngauss)
    y = zeros(ngauss)
    @eval ccall((:mfem_get_gauss, $LIBMFEM), Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}), $x, $y)
    [x y]
end

"""
    fem_nodes(mesh::Mesh)
"""
function fem_nodes(mesh::Mesh)
    mesh.nodes
end

"""
    fvm_nodes(mesh::Mesh)
"""
function fvm_nodes(mesh::Mesh)
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

function _edge_dict(mesh::Mesh)
    D = Dict{Tuple{Int64, Int64}, Int64}()
    for i = 1:mesh.nedge 
        D[(mesh.edges[i,1], mesh.edges[i,2])] = i 
        D[(mesh.edges[i,2], mesh.edges[i,1])] = i 
    end
    D
end