using PoreFlow
using Test
using PyPlot
using PyCall
using LinearAlgebra

sess = Session(); 

include("invkernel.jl")