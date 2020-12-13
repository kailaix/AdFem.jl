sess = Session(); init(sess)
@testset "pcl_impose_Dirichlet_boundary_conditions" begin 
# using Revise; using AdFem; using PyPlot; using SparseArrays
# sess = Session(); init(sess)

    A = sprand(10,10,0.6)
    II, JJ, VV = findnz(A)
    nz = length(VV)
    indices = SparseTensor(A).o.indices 

    IND = run(sess, indices).+1

    x = placeholder(rand(nz))
    B = RawSparseTensor(indices, x, 10, 10)
    rhs = rand(10)
    bdnode = [1;4;6]
    B1, rhs1 = impose_Dirichlet_boundary_conditions(B, rhs, bdnode, zeros(3))
    n = length(run(sess, B1.o.values))

    loss = sum(B1.o.values^2)
    g = gradients(loss, x)
    J = pcl_impose_Dirichlet_boundary_conditions(IND, bdnode, n)

    function test_f(x0)
        G = run(sess, g, x=>x0)
        H = pcl_linear_op(J, pcl_square_sum(n))
        return G, H 
    end

    err1, err2 = test_hessian(test_f, rand(nz); showfig = true)
    @test all(err1.<err2)
end 

@testset "pcl_compute_fem_laplace_matrix1" begin 
    mmesh = Mesh(10,10,0.1)
    κ = placeholder(rand(get_ngauss(mmesh)))
    A = compute_fem_laplace_matrix1(κ, mmesh)
    loss = sum((A.o.values - rand(mmesh.elem_ndof^2 * get_ngauss(mmesh)))^2)
    g = gradients(loss, κ)

    function test_f(x0)
        G = run(sess, g, κ=>x0)
        J = pcl_compute_fem_laplace_matrix1(mmesh)
        H = pcl_linear_op(J, pcl_square_sum(mmesh.elem_ndof^2 * get_ngauss(mmesh)))
        return G, H 
    end

    err1, err2 = test_hessian(test_f, rand(get_ngauss(mmesh)); showfig = true)
    @test all(err1.<err2)
end 

