using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

function cholesky_logdet(A)
    op_ = load_op_and_grad("./build/libCholeskyOp","cholesky_logdet")
    A = convert_to_tensor(A, dtype=Float64)
    op_(A)
end

# TODO: specify your input parameters
A = zeros(100, 9)
L = zeros(100, 6)
for i = 1:100
    a = rand(3,3)
    a = a*a'
    A[i,:] = a[:]
end
u, J = cholesky_logdet(A)
sess = Session(); init(sess)
u_ = run(sess, u)
for i = 1:100
    l = u_[i,:]
    L = [
        l[1] 0 0 
        l[4]    l[2] 0
          l[5] l[6]      l[3]
    ]
    @info L*L'-reshape(A[i,:],3,3)
end
