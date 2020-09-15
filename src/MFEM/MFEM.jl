export install_mfem, Mesh, get_ngauss, get_area


LIBMFEM = joinpath(@__DIR__, "..", "..",  "deps", "MFEM", "build", get_library_name("nnfem_mfem"))
libmfem = missing 

@doc raw"""
`Mesh` holds data structures for an unstructured mesh. 

- `nodes`: a $n_v \times 2$ coordinates array
- `elems`: a $n_e \times 3$ connectivity matrix, 1-based. 

Internally, the mesh `mmesh` is represented by a collection of `NNFEM_Element` object with some other attributes
```c++
int nelem; // total number of elements
int nnode; // total number of nodes
int ngauss; // total number of Gauss points
int order; // quadrature integration order
MatrixXd GaussPts; // coordinates of Gauss quadrature points
std::vector<NNFEM_Element*> elements; // array of elements
```

The `NNFEM_Element` has data
```c++
VectorXd h;   // basis functions, 3 × ng  
VectorXd hx;  // x-directional basis functions, 3 × ng  
VectorXd hy;  // y-directional basis functions, 3 × ng  
VectorXd w;   // weight vectors, ng  
double area;  // area of the triangle
MatrixXd coord; // coordinates array, 3 × 2
int node[3]; // global index of the vertices
int nnode; // total number of nodes 
int ngauss; // total number of Gauss points
```
"""
mutable struct Mesh
    nodes::Array{Float64, 2}
    elems::Array{Int64, 2}
    function Mesh(coords::Array{Float64, 2}, elems::Array{Int64, 2}, order::Int64 = 2)
        c = [coords zeros(size(coords, 1))]'[:]
        e = Int32.(elems'[:].- 1) 
        global libmfem = tf.load_op_library(LIBMFEM) # load for tensorflow first
        @eval ccall((:init_nnfem_mesh, $LIBMFEM), Cvoid, (Ptr{Cdouble}, Cint, 
                Ptr{Cint}, Cint, Cint), $c, Int32(size($coords, 1)), $e, Int32(size($elems,1)), 
                Int32($order))
        new(coords, elems)
    end
end

@doc raw"""
    Mesh(m::Int64, n::Int64, h::Float64; order::Int64 = 2)

Constructs a mesh of a rectangular domain. The rectangle is split into $m\times n$ cells, and each cell is further split into two triangles. 
`order` specifies the quadrature rule order. 
"""
function Mesh(m::Int64, n::Int64, h::Float64; order::Int64 = 2)
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
    Mesh(coords, elems, order)
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