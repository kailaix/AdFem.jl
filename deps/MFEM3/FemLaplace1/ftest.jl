using AdFem

mmesh = Mesh3(10, 10, 10, 0.1)
xy = gauss_nodes(mmesh)
κ = @. 1/(a+xy[:,1]^2 + xy[:,2]^2)
K = compute_fem_laplace_matrix1(κ, mmesh)
rhs = compute_fem_source_term1(ones(get_ngauss(mmesh)), mmesh)
bcnode(mmesh)
sol = K\rhs 

idx = falses(mmesh.ndof)
s = 1
for k = 1:11
    for j = 1:11
        for i = 1:11
            idx[s] = true 
            global s = s + 1
        end
    end
end
sol[s]