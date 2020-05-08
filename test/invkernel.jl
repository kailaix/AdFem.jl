@testset "compute_fvm_tpfa_matrix" begin 
    m = 20
    n = 10
    h = 0.1
    kvalue = rand(m*n)
    bc = bcedge("all", m, n, h)
    pval = rand(size(bc, 1))
    K, rhs = compute_fvm_tpfa_matrix(kvalue,bc,pval,m,n,h)
    K1, rhs1 = compute_fvm_tpfa_matrix(constant(kvalue),bc,pval,m,n,h)
    K2 = run(sess, K1)
    rhs2 = run(sess, rhs1)
    @test rhs≈rhs2
    @test K ≈ K2

    m = 20
    n = 10
    h = 0.1
    kvalue = rand(m*n)
    K = compute_fvm_tpfa_matrix(kvalue,m,n,h)
    K1 = compute_fvm_tpfa_matrix(constant(kvalue),m,n,h)
    K2 = run(sess, K1)
    @test K ≈ K2
end