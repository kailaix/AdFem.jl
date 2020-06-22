# implicit scheme for advection-diffusion

using Revise
using PoreFlow
using ADCME
using PyPlot
using Statistics
using LinearAlgebra
using ADCMEKit

m = 40
n = 20
h = 1/n
NT = 100
Δt = 1/NT 

function solve_stokes(η)
    hmat = compute_space_varying_tangent_elasticity_matrix(η)    
    K = compute_fem_stiffness_matrix(hmat, m, n, h)
    B = constant(compute_interaction_matrix(m, n, h))
    Z = [K -B'
    B spzero(size(B,1))]
    bd = bcnode("lower", m, n, h)
    Z, _ = fem_impose_Dirichlet_boundary_condition(Z, bd, m, n, h)
    T = ones(4*m*n)
    F1 = compute_fem_source_term1(T, m, n, h)
    F0 = zero(F1)
    rhs = [F1;F0;zeros(m*n)]

    sol = Z\rhs
end