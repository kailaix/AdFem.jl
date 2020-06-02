using Revise
using ADCME
using ADCMEKit
using NNFEM 
using PyPlot
using ProgressMeter 

#----------------------------------------------------------------------------
# step 1: create a computational domain 
# fixed on the bottom and push from right 


m = 40
n = 20
h = 1/n
NT = 50
Δt = 0.01/NT

elements = []
prop = Dict("name"=> "PlaneStrain", "rho"=> 2000, "E"=> 1e10, "nu"=> 0.35)
# prop = Dict("name"=> "ViscoelasticityMaxwell", "rho"=> 2000, "E"=> 1e10, "nu"=> 0.35, "eta"=>1e5)

coords = zeros((m+1)*(n+1), 2)
for j = 1:n
    for i = 1:m
        idx = (m+1)*(j-1)+i 
        elnodes = [idx; idx+1; idx+1+m+1; idx+m+1]
        ngp = 2
        nodes = [
        (i-1)*h (j-1)*h
        i*h (j-1)*h
        i*h j*h
        (i-1)*h j*h
        ]
        coords[elnodes, :] = nodes
        push!(elements, SmallStrainContinuum(nodes, elnodes, prop, ngp))
    end
end

# fixed on the bottom, push on the right
EBC = zeros(Int64, (m+1)*(n+1), 2)
FBC = zeros(Int64, (m+1)*(n+1), 2)
g = zeros((m+1)*(n+1), 2)
f = zeros((m+1)*(n+1), 2)

for i = 1:m+1
    EBC[i+n*(m+1),:] .= -1
end

Edge_Traction_Data = Array{Int64}[]
for i = 1:n
  elem = elements[(i-1)*m + m]
  for k = 1:4
    if elem.coords[k,1]>2-0.001 && elem.coords[k+1>4 ? 1 : k+1,1]>2-0.001
      push!(Edge_Traction_Data, [(i-1)*m + m, k, 1])
    end
  end
end
Edge_Traction_Data = hcat(Edge_Traction_Data...)'|>Array

ndims = 2
domain = Domain(coords, elements, ndims, EBC, g, FBC, f, Edge_Traction_Data)
globaldata = example_global_data(domain)

# visualize_boundary(domain)
#----------------------------------------------------------------------------
# step 2: Simulation

Dstate = zeros(domain.neqs) 
state = zeros(domain.neqs)
velo = zeros(domain.neqs)
acce = zeros(domain.neqs)
EBC_func = nothing 
FBC_func = nothing 
Body_func = nothing 
function Edge_func(x, y, t, idx)
  return [-1e6*ones(length(x)) zeros(length(x))] 
end


globaldata = GlobalData(state, Dstate, velo, acce, domain.neqs, EBC_func, FBC_func,Body_func, Edge_func)
assembleMassMatrix!(globaldata, domain)

@showprogress for i = 1:NT
    global globaldata, domain = GeneralizedAlphaSolverStep(globaldata, domain, Δt)
end

#----------------------------------------------------------------------------
# step 3: visualize 
d_ = hcat(domain.history["state"]...)'|>Array
# visualize_von_mises_stress_on_scoped_body(d_, domain, scale_factor=10.0)
p = visualize_total_deformation_on_scoped_body(d_, domain, scale_factor=10.0)
saveanim(p, "linear.gif")
close("all")

#----------------------------------------------------------------------------
# step 4: verification with PoreFlow   
