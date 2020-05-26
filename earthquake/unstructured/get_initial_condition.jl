using Revise
using ADCME
using ADCMEKit
using NNFEM 
using PyPlot
using ProgressMeter 

include("load_domain_function.jl")

NT = 100
Δt = 30/NT
domain = load_crack_domain()
globaldata = example_global_data(domain)
assembleMassMatrix!(globaldata, domain)
updateDomainStateBoundary!(domain, globdat)
Hs = domain.elements[1].mat[1].H
d0 = globaldata.state[:]

globaldata, domain = ImplicitStaticSolver(globaldata, domain, d0, 0.0, 0, Hs, Fext)

ImplicitStaticSolver(globaldata, domain)
visualize_scalar_on_scoped_body(d0, d0, domain)