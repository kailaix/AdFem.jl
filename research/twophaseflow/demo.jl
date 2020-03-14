using Revise
using FwiFlow
using PoreFlow
using ADCME
using PyPlot


function computeφ(u)
    ε = compute_fvm_mechanics(u, m, n, h) 
end