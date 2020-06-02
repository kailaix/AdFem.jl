using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
using SparseArrays
Random.seed!(233)

function push_matrices(A,B)
    ii1, jj1, vv1 = find(A)
    ii2, jj2, vv2 = find(B)
    d = size(A, 1)
    push_matrices_ = load_op_and_grad("$(@__DIR__)/build/libPushMatrices","push_matrices")
    ii1,jj1,vv1,ii2,jj2,vv2,d = convert_to_tensor([ii1,jj1,vv1,ii2,jj2,vv2,d], [Int64,Int64,Float64,Int64,Int64,Float64,Int64])
    push_matrices_(ii1,jj1,vv1,ii2,jj2,vv2,d)
end

function visco_solve(rhs,vv,op)
    visco_solve_ = load_op_and_grad("$(@__DIR__)/build/libViscoSolve","visco_solve")
    rhs,vv, op = convert_to_tensor([rhs,vv, op], [Float64,Float64, Int64])
    visco_solve_(rhs,vv, op)
end