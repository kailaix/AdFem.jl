
using NNFEM 


viscoelasticity_idx = []


# function load_crack_domain(;option::String = "elasticity", xmax::Float64 = 8.0, ymax::Float64 = 4.0, c1=(4.0, 0.0), c2=(4.0,0.5))
#     nodes, elements = meshread("crack_vertical.msh")
function load_crack_domain(;option::String = "elasticity", xmax::Float64 = 8.0, ymax::Float64 = 4.0, c1=(3.5, 0.0), c2=(4.0,0.5))
    nodes, elements = meshread("crack_wider.msh")
    slope = (c2[2]-c1[2])/(c2[1]-c1[1])
    # slope = 0
    scale_factor = 1000
    nodes *= scale_factor

    # visualize_mesh(nodes, elements)
    ids = Int64[]
    for i = 1:size(nodes, 1)
        x, y = nodes[i,:]
        if  abs(y-slope*(x-c1[1]*scale_factor))<0.05*scale_factor && (c1[1]*0.99*scale_factor < x <= c2[1]*1.01*scale_factor)
        # if  abs(c1[2]*scale_factor <= y <= c2[2]*scale_factor) && abs(x-c1[1]*scale_factor)<0.05*scale_factor
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
        if y-slope*(x-c1[1]*scale_factor)<0
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
            if x<0.001*scale_factor || x>xmax*scale_factor-0.001*scale_factor || y>=ymax*scale_factor-0.001*scale_factor
                push!(id4, i)
            end
            if y<0.001*scale_factor && x>=0.001*scale_factor && x<=xmax*scale_factor-0.001*scale_factor
                push!(id3, i)
            end
        end

    end

    id3 = collect(id3)
    id4 = collect(id4)

    elems = []
    # prop = Dict("name"=> "PlaneStrain", "rho"=> 2700, "E"=> 3.4e10, "nu"=> 0.2339)
    if option=="viscoelasticity"
        prop = Dict("name"=> "ViscoelasticityMaxwell", "rho"=> 2000, "E"=> 1e10, "nu"=> 0.35, "eta"=>1e20)
    elseif option=="elasticity"
        prop = Dict("name"=> "PlaneStrain", "rho"=> 2000, "E"=> 1e10, "nu"=> 0.35)
    end
    global viscoelasticity_idx = []
    g_idx = 0
    for j = 1:size(elements,1)
        elnodes = elements[j,:]
        nodes_ = nodes[elnodes, :]
        ngp = 3
        if option=="mixed"
            if mean(nodes_[:,2])<c2[2]*scale_factor
                prop = Dict("name"=> "ViscoelasticityMaxwell", "rho"=> 2000, "E"=> 1e10, "nu"=> 0.35, "eta"=>1e20)
            else 
                prop = Dict("name"=> "ViscoelasticityMaxwell", "rho"=> 2000, "E"=> 1e10, "nu"=> 0.35, "eta"=>1e10)
                # prop = Dict("name"=> "PlaneStrainViscoelasticityProny", "rho"=> 2000, "E"=> 1e10, "nu"=> 0.35, "eta"=>1e10, "c"=>0.99, "tau"=>1e-3)
                # prop = Dict("name"=> "PlaneStrainViscoelasticityProny", "rho"=> 2000, "E"=> 1e10, "nu"=> 0.35, "eta"=>1e20, "c"=>0.99, "tau"=>10.0)
                # prop = Dict("name"=> "ViscoelasticityMaxwell", "rho"=> 2000, "E"=> 1e10, "nu"=> 0.35, "eta"=>1e20)
                # prop = Dict("name"=> "PlaneStrain", "rho"=> 2000, "E"=> 1e8, "nu"=> 0.35)

            end
        end
        push!(elems, SmallStrainContinuum(nodes_, elnodes, prop, ngp))

        if option=="mixed"
            if mean(nodes_[:,2])<c2[2]*scale_factor
            else 
                for k = 1:length(elems[end].weights)
                    push!(viscoelasticity_idx, g_idx + k)
                end
            end
        end
        g_idx += length(elems[end].weights)
    end


    # fixed on the bottom, left, and right
    EBC = zeros(Int64, size(nodes, 1), 2)
    FBC = zeros(Int64, size(nodes, 1), 2)
    g = zeros(size(nodes, 1), 2)
    f = zeros(size(nodes, 1), 2)

    EBC[id1,:] .= -1
    EBC[id2,:] .= -1
    EBC[id4,:] .= -1
    g[id1,:] .= [5 5] # m
    g[id2,:] .= [-5 -5] # m
    # g[id1,:] .= [0. 0.5] # m
    # g[id2,:] .= [0. -0.5] # m

    ndims = 2
    domain = Domain(nodes, elems, ndims, EBC, g, FBC, f)
end


function make_patch()
end