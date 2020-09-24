export Mesh, get_ngauss, get_area

@doc raw"""
`Mesh` holds data structures for an unstructured mesh. 

- `nodes`: a $n_v \times 2$ coordinates array
- `edges`: a $n_{\text{edge}} \times 2$ integer array for edges 
- `elems`: a $n_e \times 3$ connectivity matrix, 1-based. 
- `nnode`, `nedge`, `nelem`: number of nodes, edges, and elements 
- `ndof`: total number of degrees of freedoms 
- `conn`: connectivity matrix, `nelems × 3` or `nelems × 6`, depending on whether a linear element or a quadratic element is used. 

Internally, the mesh `mmesh` is represented by a collection of `NNFEM_Element` object with some other attributes
```c++
int nelem; // total number of elements
int nnode; // total number of nodes
int ngauss; // total number of Gauss points
int ndof; // total number of dofs
int order; // quadrature integration order
int degree; // Degree of Polynomials, 1 - P1 element, 2 - P2 element 
int elem_ndof; // 3 for P1, 6 for P2
MatrixXd GaussPts; // coordinates of Gauss quadrature points
std::vector<NNFEM_Element*> elements; // array of elements
```

The `NNFEM_Element` has data
```c++
VectorXd h;   // basis functions, elem_ndof × ng  
VectorXd hx;  // x-directional basis functions, elem_ndof × ng  
VectorXd hy;  // y-directional basis functions, elem_ndof × ng  
MatrixXd hs;  // shape functions for linear element, 3 × ng
VectorXd w;   // weight vectors, ng  
double area;  // area of the triangle
MatrixXd coord; // coordinates array, 3 × 2
int nnode; // total number of nodes 
int ngauss; // total number of Gauss points
int dof[6]; // global indices for both nodes and edges, note that edge indices are offset by `nv`
int node[3]; // global indices of local vertices
int edge[3]; // global indices of local edges
int ndof; // DOF, 3 for P1 element, 6 for P2 element 
```
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
    function Mesh(coords::Array{Float64, 2}, elems::Array{Int64, 2}, order::Int64 = -1, degree::Int64 = 1)
        if !(degree in [1, 2])
            error("Only degree = 1 or 2 is supported.")
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
        nedges = nnode + nelem - 1
        edges = zeros(Int64, nedges*2)
        c = [coords zeros(size(coords, 1))]'[:]
        e = Int32.(elems'[:].- 1) 
        @eval ccall((:init_nnfem_mesh, $LIBMFEM), Cvoid, (Ptr{Cdouble}, Cint, 
                Ptr{Cint}, Cint, Cint, Cint, Ptr{Clonglong}), $c, Int32(size($coords, 1)), $e, Int32(size($elems,1)), 
                Int32($order), Int32($degree), $edges)
        edges = reshape(edges, nedges, 2)
        elem_dof = Int64(@eval ccall((:mfem_get_elem_ndof, $LIBMFEM), Cint, ()))
        conn = zeros(Int64, elem_dof * size(elems, 1))
        @eval ccall((:mfem_get_connectivity, $LIBMFEM), Cvoid, (Ptr{Cint}, ), $conn)
        ndof = Int64(@eval ccall((:mfem_get_ndof, $LIBMFEM), Cint, ()))
        conn = reshape(conn, elem_dof, size(elems, 1))'|>Array
        elems = conn[:, 1:3]
        new(coords, edges,  elems, nnode, nedges, nelem, ndof, conn)
    end
end

@doc raw"""
    Mesh(m::Int64, n::Int64, h::Float64; order::Int64 = 2, degree::Int64 = 1)

Constructs a mesh of a rectangular domain. The rectangle is split into $m\times n$ cells, and each cell is further split into two triangles. 
`order` specifies the quadrature rule order. `degree` determines the degree for finite element basis functions.
"""
function Mesh(m::Int64, n::Int64, h::Float64; order::Int64 = -1, degree::Int64 = 1)
    coords = zeros((m+1)*(n+1), 2)
    elems = zeros(Int64, 2*m*n, 3)
    for i = 1:n 
        for j = 1:m 
            e = 2*((i-1)*m + j - 1)+1
            elems[e, :] = [(i-1)*(m+1)+j; (i-1)*(m+1)+j+1; i*(m+1)+j ]
            elems[e+1, :] = [(i-1)*(m+1)+j+1; i*(m+1)+j; i*(m+1)+j+1]
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
    Mesh(coords, elems, order, degree)
end

function Base.:length(mesh::Mesh)
    return maximum(mesh.conn)
end

"""
    get_ngauss(mesh::Mesh) 

Return the total number of Gauss points. 
"""
function get_ngauss(mesh::Mesh)
    return @eval ccall((:mfem_get_ngauss, $LIBMFEM), Cint, ())
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