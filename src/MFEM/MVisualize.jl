export visualize_mesh
function visualize_mesh(mesh::Mesh)
    nodes, elems = mesh.nodes, mesh.elems
    patches = PyObject[]
    for i = 1:size(elems,1)
        e = elems[i,:]
        p = plt.Polygon(nodes[e,:],edgecolor="k",lw=1,fc=nothing,fill=false)
        push!(patches, p)
    end
    p = matplotlib.collections.PatchCollection(patches, match_original=true)
    gca().add_collection(p)
    axis("scaled")
    xlabel("x")
    ylabel("y")
    gca().invert_yaxis()
end

function visualize_scalar_on_fem_points(u::Array{Float64,1}, mesh::Mesh, args...;kwargs...)
        # plots a finite element mesh
    function plot_fem_mesh(nodes_x, nodes_y)
        for i = 1:size(mesh.elems, 1)
            x = nodes_x[mesh.elems[i,:]]
            y = nodes_y[mesh.elems[i,:]]
            plt.fill(x, y, edgecolor="black", fill=false)
        end
    end

    # FEM data
    nodes_x = mesh.nodes[:,1]
    nodes_y = mesh.nodes[:,2]
    nodal_values = u
    elements_tris = []
    for i = 1:size(mesh.elems, 1)
        push!(elements_tris, mesh.elems[i,:] .- 1)
    end

    # create an unstructured triangular grid instance
    triangulation = matplotlib.tri.Triangulation(nodes_x, nodes_y, elements_tris)

    # plot the finite element mesh
    plot_fem_mesh(nodes_x, nodes_y)

    # plot the contours
    plt.tricontourf(triangulation, nodal_values)

    # show
    colorbar()
    axis("scaled")
end