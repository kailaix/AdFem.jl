using Revise
using NNFEM
using Statistics 
using PyPlot
using ADCME
using ADCMEKit

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

close("all")
visualize_mesh(nodes, elements)
gca().invert_yaxis()

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

plot(nodes[id1, 1] .+ 10, nodes[id1, 2] .+ 5, ".", label="Shifted")
plot(nodes[id2, 1] .- 10, nodes[id2, 2] .- 5, ".", label="Shifted opposite")
plot(nodes[id3, 1], nodes[id3, 2], "x", label="Traction free")
plot(nodes[id4, 1], nodes[id4, 2], "+", label="Fixed")
legend()

savefig("mesh.png")

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

Dstate = zeros(domain.neqs)
state = zeros(domain.neqs)
velo = zeros(domain.neqs)
acce = zeros(domain.neqs)
gt = nothing
ft = nothing
globdat = GlobalData(state, Dstate, velo, acce, domain.neqs, gt, ft)
assembleMassMatrix!(globdat, domain)


#===========
Simulation 
===========#
NT = 100
Δt = 30.0/NT
Solver = GeneralizedAlphaSolver
SolverTime = Symbol(Solver, "Time")
@eval  ts = $SolverTime(Δt, NT)
ubd, abd = compute_boundary_info(domain, globdat, ts)
d0, v0, a0 = SolverInitial(Δt, globdat, domain)

H = elems[1].mat[1].H
Hs = zeros(getNGauss(domain), 3, 3)
for i = 1:size(Hs,1)
    Hs[i,:,:] = H 
end
@eval d, v, a = $Solver(globdat, domain, d0, v0, a0, Δt, NT, Hs)

sess = Session(); init(sess)

d_ = run(sess, d)
# error()

surface_x = []
surface_y = []
x_tmp = []
visited = []
for j = 1:size(elements, 1)
    e = elements[j,:]
    for i in e 
        x, y = nodes[i, :]
        if y<0.01*scale_factor && x>=0.01*scale_factor && x<=2*scale_factor-0.01*scale_factor
            if i in visited
                push!(surface_x, d_[:, i])
                push!(surface_y, d_[:, domain.nnodes+i])
                push!(x_tmp, x)
            end
            push!(visited, i)
        end
    end
end
surface_x = hcat(surface_x[sortperm(x_tmp)]...)
surface_y = hcat(surface_y[sortperm(x_tmp)]...)

# figure()
figure(figsize=(8,3))
# subplot(211)
# for i = 1:NT÷10:NT
#     plot(surface_x[i, :])
# end
plot(surface_x[NT, :], "--", label="x")
# legend()
# subplot(212)
# for i = 1:NT÷10:NT
#     plot(-surface_y[i, :])
# end
plot(-surface_y[NT, :], "-.", label="y")
legend()
title("Displacement")
savefig("surface_disp.png")

# error()
p = visualize_total_deformation_on_scoped_body(d_, domain; scale_factor=10.0)
saveanim(p, "displacement.gif")
p = visualize_scalar_on_scoped_body(d_[:,1:domain.nnodes], d_, domain; scale_factor=10.0)
saveanim(p, "displacement_x.gif")
p = visualize_scalar_on_scoped_body(d_[:,domain.nnodes+1:end], d_, domain; scale_factor=10.0)
saveanim(p, "displacement_y.gif")

#saveanim(p, "displacement.gif")

# v_ = run(sess, v)
# p = visualize_total_deformation_on_scoped_body(v_, domain; scale_factor=10.0)
# p = visualize_scalar_on_scoped_body(v_[:,1:domain.nnodes], v_, domain; scale_factor=10.0)
# p = visualize_scalar_on_scoped_body(v_[:,domain.nnodes+1:end], v_, domain; scale_factor=10.0)

# a_ = run(sess, a)
# p = visualize_total_deformation_on_scoped_body(a_, domain; scale_factor=10.0)
# p = visualize_scalar_on_scoped_body(a_[:,1:domain.nnodes], a_, domain; scale_factor=10.0) ###
# p = visualize_scalar_on_scoped_body(a_[:,domain.nnodes+1:end], a_, domain; scale_factor=10.0)


# a_ = run(sess, a)
# close("all")
# p = visualize_scalar_on_scoped_body(a_[:,1:domain.nnodes], a_, domain; scale_factor=10.0)


# p = visualize_von_mises_stress_on_scoped_body(d_, domain; scale_factor=10.0)

# saveanim(p, "displacement.gif")
