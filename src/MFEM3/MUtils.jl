export bcface, bcnode, vtk, pvd, save

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

@doc """
    vtk(mmesh::Mesh3, name::String = "default_vtk_file")

Returns a handle to vtk file `name`. Users can add data to the file via 

```
f = vtk(mmesh)
f["nodal_data"] = d1 # length(d1) == mmesh.nnode
f["cell_data"] = d2 # length(d2) == mmesh.nelem
f["scalar_data"] = d3 # d3 is a scalar
```

Once the data has been added, users can save the file to harddisk via 

```
outfiles = save(f)
```
"""
function vtk(mmesh::Mesh3, name::String = "default_vtk_file")
    cells = WriteVTK.MeshCell[]
    for i = 1:mmesh.nelem
        push!(cells, MeshCell(VTKCellTypes.VTK_TETRA, mmesh.elems[i,:]))
    end
    vtk_grid(name, mmesh.nodes[:,1], mmesh.nodes[:,2], mmesh.nodes[:,3], cells)
end

function save(f::Union{WriteVTK.CollectionFile,WriteVTK.DatasetFile})
    vtk_save(f)
end

@doc """
    pvd(name::String = "default_pvd_file"; append = true)

Returns a handle to pvd file `name`

```
f = pdv()
f[t1] = vtk1 # vtk1 is a vtk file
f[t2] = vtk2
...
```

Once the data has been added, users can save the file to harddisk via 

```
outfiles = save(f)
```
"""
function pvd(name::String = "default_pvd_file"; append = true)
    pvd = paraview_collection(name; append=append)
end

"""
    compute_pml_term(u::Union{Array{Float64,1}, PyObject},
        βprime::Union{Array{Float64,1}, PyObject},
        λ::Union{Array{Float64,1}, PyObject},μ::Union{Array{Float64,1}, PyObject}, 
        nv::Union{Array{Float64,2}, PyObject}, mmesh::Mesh3)
"""
function compute_pml_term(u::Union{Array{Float64,1}, PyObject},
    βprime::Union{Array{Float64,1}, PyObject},
    λ::Union{Array{Float64,1}, PyObject},μ::Union{Array{Float64,1}, PyObject}, 
    nv::Union{Array{Float64,2}, PyObject}, mmesh::Mesh3)
    @assert length(u)==3mmesh.ndof
    @assert length(βprime)==length(λ)==length(μ)==get_ngauss(mmesh)
    @assert size(nv)==(get_ngauss(mmesh), 3)
    compute_pml_elastic_term_t_ = load_op_and_grad(libmfem3,"compute_pml_elastic_term_t", multiple=true)
    u,betap,e,nu,nv = convert_to_tensor(Any[u,βprime,λ,μ,nv], [Float64,Float64,Float64,Float64,Float64])
    out = compute_pml_elastic_term_t_(u,betap,e,nu,nv)
    set_shape(out[1], (3mmesh.ndof, )), set_shape(out[2], (3mmesh.ndof, )), set_shape(out[3], (3mmesh.ndof, )), set_shape(out[4], (3mmesh.ndof, ))
end