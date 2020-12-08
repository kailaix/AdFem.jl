"""
    visualize_mesh(mmesh::Mesh3; filename::Union{Missing, String} = missing)

Visualizes the unstructured mesh `mmesh`. When `filename` is provided, a screenshot is saved to `filename`
"""
function visualize_mesh(mmesh::Mesh3; 
        filename::Union{Missing, String} = missing, 
        scalars::Union{Nothing, Array{Float64, 1}} = nothing)
    points = mmesh.nodes 
    cells = Int32.(mmesh.elems) .- 1
    cell_dict = Dict(10=>cells)
    mesh = pv.UnstructuredGrid(cell_dict, points)
    p = pv.Plotter()
    p.add_mesh(mesh, show_edges=true, color="white", scalars = scalars)
    p.add_mesh(pv.PolyData(mesh.points), color="red",
        point_size=10, render_points_as_spheres=true)
    p.show_axes()
    if ismissing(filename)
        p.show()
    else 
        p.show(screenshot=filename)
    end
    p
end

"""
    visualize_scalar_on_fem_points(u::Array{Float64,1}, mmesh::Mesh3; filename::Union{Missing, String} = missing)

Visualizes the unstructured mesh `mmesh` with nodes colored with values `u`. When `filename` is provided, a screenshot is saved to `filename`
"""
function visualize_scalar_on_fem_points(u::Array{Float64,1}, mmesh::Mesh3; filename::Union{Missing, String} = missing)
    @assert length(u) == mmesh.nnode 
    visualize_mesh(mmesh; filename = filename, scalars = u)
end


"""
    visualize_scalar_on_fvm_points(u::Array{Float64,1}, mmesh::Mesh3; filename::Union{Missing, String} = missing)

Visualizes the unstructured mesh `mmesh` with elements colored with values `u`. When `filename` is provided, a screenshot is saved to `filename`
"""
function visualize_scalar_on_fvm_points(u::Array{Float64,1}, mmesh::Mesh3; filename::Union{Missing, String} = missing)
    @assert length(u) == mmesh.nelem
    visualize_mesh(mmesh; filename = filename, scalars = u)
end



"""
    visualize_scalar_on_fvm_points(u::Array{Float64,1}, mmesh::Mesh3; filename::Union{Missing, String} = missing)

Visualizes the unstructured mesh `mmesh` with elements colored with values `u`. When `filename` is provided, a screenshot is saved to `filename`
"""
function visualize_scalar_on_gauss_points(u::Array{Float64,1}, mmesh::Mesh3; filename::Union{Missing, String} = missing)
    @assert length(u) == get_ngauss(mmesh)
    ngauss_per_element = length(u)÷mmesh.nelem
    z = zeros(mmesh.nelem)
    for i = 1:ngauss_per_element
        z += u[i:ngauss_per_element:end]
    end
    z/=ngauss_per_element
    visualize_scalar_on_fvm_points(z, mmesh; filename = filename)
end