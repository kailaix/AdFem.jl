
using NNFEM 
function load_crack_domain()
    nodes, elements = meshread("crack.msh")

    scale_factor = 10000
    nodes *= scale_factor

    # visualize_mesh(nodes, elements)
    ids = Int64[]
    for i = 1:size(nodes, 1)
        x, y = nodes[i,:]
        if  abs(y-0.5*(x-0.5*scale_factor))<0.01*scale_factor && (0.5*scale_factor < x <= 1.55*scale_factor)
            push!(ids, i)
            # @info x, y
            # plot([x], [y], ".g")
        end
    end

    new_nodes = []
    adic = Dict{Int64, Int64}()
    id1 = Set{Int64}([]) # crack lower surface
    id2 = Set{Int64}([]) # crack upper surface
    id3 = Set{Int64}([]) # free traction
    id4 = Set{Int64}([]) # other fixed boundaries 


    for i = 1:size(elements, 1)
        e = elements[i,:]
        if !any([j in ids for j in e])
            continue
        end

        @info i 
        x, y = mean(nodes[e,:], dims=1)[:]
        if y-0.5*(x-0.5*scale_factor)<0
            # plot([x], [y], ".r")
        else
            for j = 1:4
                for e[j] in ids
                    push!(id1, e[j])
                end
            end
            continue
        end

        ## add nodes from the uppper side of crack
        for j = 1:4 
            if e[j] in ids 
                if haskey(adic,e[j])
                    elements[i,j] = adic[e[j]]
                else
                    push!(new_nodes, nodes[e[j],:])
                    elements[i,j] = size(new_nodes,1) + size(nodes,1)
                    adic[e[j]] = size(new_nodes,1) + size(nodes,1)
                end
                push!(id2, adic[e[j]])
            end
        end
            
    end

    # savefig("marker.png")

    new_nodes = hcat(new_nodes...)
    nodes = [nodes;new_nodes']


    id1 = collect(id1)
    id2 = collect(id2) 

    for j = 1:size(elements, 1)
        e = elements[j,:]
        for i in e 
            x, y = nodes[i, :]
            if x<0.01*scale_factor || x>2*scale_factor-0.01*scale_factor || y>=1*scale_factor-0.01*scale_factor
                push!(id4, i)
            end
            if y<0.01*scale_factor && x>=0.01*scale_factor && x<=2*scale_factor-0.01*scale_factor
                push!(id3, i)
            end
        end

    end

    id3 = collect(id3)
    id4 = collect(id4)

    elems = []
    prop = Dict("name"=> "PlaneStrain", "rho"=> 2700, "E"=> 3.4e10, "nu"=> 0.2339)
    for j = 1:size(elements,1)
        elnodes = elements[j,:]
        nodes_ = nodes[elnodes, :]
        ngp = 3
        push!(elems, SmallStrainContinuum(nodes_, elnodes, prop, ngp))
    end


    # fixed on the bottom, left, and right
    EBC = zeros(Int64, size(nodes, 1), 2)
    FBC = zeros(Int64, size(nodes, 1), 2)
    g = zeros(size(nodes, 1), 2)
    f = zeros(size(nodes, 1), 2)

    EBC[id1,:] .= -1
    EBC[id2,:] .= -1
    EBC[id4,:] .= -1
    g[id1,:] .= [10 5] # m
    g[id2,:] .= [-10 -5] # m

    ndims = 2
    domain = Domain(nodes, elems, ndims, EBC, g, FBC, f)
end