export install_mfem, Mesh, get_ngauss, get_area


LIBMFEM = joinpath(@__DIR__, "..",  "deps", "MFEM", "build", get_library_name("nnfem_mfem"))
libmfem = missing 

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

function get_ngauss(mesh::Mesh)
    return @eval ccall((:mfem_get_ngauss, $LIBMFEM), Cint, ())
end

function get_area(mesh::Mesh)
    a = zeros(size(mesh.elems,1))
    @eval ccall((:mfem_get_area, $LIBMFEM), Cvoid, (Ptr{Cdouble}, ), $a)
    a
end

function gauss_nodes(mesh::Mesh)
    ngauss = get_ngauss(mesh)
    x = zeros(ngauss)
    y = zeros(ngauss)
    @eval ccall((:mfem_get_gauss, $LIBMFEM), Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}), $x, $y)
    [x y]
end


function fem_nodes(mesh::Mesh)
    mesh.nodes
end