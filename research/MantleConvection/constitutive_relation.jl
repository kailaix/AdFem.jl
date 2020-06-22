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
