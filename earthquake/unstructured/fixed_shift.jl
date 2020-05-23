using Revise
using NNFEM
using Statistics 
using PyPlot
using ADCMEKit

nodes, elements = meshread("crack.msh")

visualize_mesh(nodes, elements)

ids = Int64[]
for i = 1:size(nodes, 1)
    x, y = nodes[i,:]
    if  abs(y-0.5*(x-0.5))<0.01 && (0.5 < x <= 1.55)
        push!(ids, i)
        @info x, y
        plot([x], [y], ".g")
    end
end

new_nodes = []
adic = Dict{Int64, Int64}()
id1 = Set{Int64}([])
id2 = Set{Int64}([])
id3 = Set{Int64}([])

for i = 1:size(elements, 1)
    e = elements[i,:]
    if !any([j in ids for j in e])
        continue
    end

    @info i 
    x, y = mean(nodes[e,:], dims=1)[:]
    if y-0.5*(x-0.5)<0
        plot([x], [y], ".r")
    else
        for j = 1:4
            for e[j] in ids
                push!(id1, e[j])
            end
        end
        continue
    end

    for j = 1:4
        if e[j] in ids 
            push!(id2, e[j])
            if haskey(adic,e[j])
                elements[i,j] = adic[e[j]]
            else
                push!(new_nodes, nodes[e[j],:])
                elements[i,j] = size(new_nodes,1) + size(nodes,1)
                adic[e[j]] = size(new_nodes,1) + size(nodes,1)
            end
        end
    end
        
end

savefig("marker.png")

new_nodes = hcat(new_nodes...)
nodes = [nodes;new_nodes']

close("all")
visualize_mesh(nodes, elements)
gca().invert_yaxis()

id1 = collect(id1) # moving 
id2 = collect(id2) # fixed 

for j = 1:size(elements, 1)
    e = elements[j,:]
    for i in e 
        x, y = nodes[i, :]
        if x<0.01 || x>2-0.01 || y>=1-0.01
            push!(id2, i)
        end
        if y<0.01 && x>=0.01 && x<=2-0.01
            push!(id3, i)
        end
    end

end

id3 = collect(id3) # free traction


plot(nodes[id1, 1] .+0.01, nodes[id1, 2] .+ 0.005, ".", label="Shifted")
plot(nodes[id2, 1], nodes[id2, 2], "+", label="Fixed")
plot(nodes[id3, 1], nodes[id3, 2], "x", label="Traction free")
legend()

savefig("mesh.png")

elems = []
prop = Dict("name"=> "PlaneStrain", "rho"=> 1.0, "E"=> 2.0, "nu"=> 0.35)
for j = 1:size(elements,1)
    elnodes = elements[j,:]
    nodes_ = nodes[elnodes, :]
    ngp = 3
    push!(elems, SmallStrainContinuum(nodes_, elnodes, prop, ngp))
end


# fixed on the bottom, push on the right
EBC = zeros(Int64, size(nodes, 1), 2)
FBC = zeros(Int64, size(nodes, 1), 2)
g = zeros(size(nodes, 1), 2)
f = zeros(size(nodes, 1), 2)

EBC[id1,:] .= -1
EBC[id2,:] .= -1
g[id1,:] .= [0.01 0.005]


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
Δt = 1.0/NT
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
p = visualize_displacement(d_, domain)
saveanim(p, "displacement.gif")
