using AdFem

mmesh = Mesh3(20, 20, 20, 1/20)
xy = gauss_nodes(mmesh)
a = Variable(0.1)
κ = 1/(a+xy[:,1].^2 + xy[:,2].^2 + xy[:,3].^2)
K = compute_fem_laplace_matrix1(κ, mmesh)
rhs = compute_fem_source_term1(ones(get_ngauss(mmesh)), mmesh)
bd = bcnode(mmesh)
K, rhs = impose_Dirichlet_boundary_conditions(K, rhs, bd, zeros(length(bd)))
sol = K\rhs
sess = Session(); init(sess)
SOL = run(sess, sol)


############## Visualization 
using PyPlot
using DelimitedFiles 
idx = falses(mmesh.ndof)
s = 1
for k = 1:21
    for j = 1:21
        for i = 1:21
            if k==11 && j==11
                idx[s] = true 
            end
            global s = s + 1
        end
    end
end
close("all")
x0 = LinRange(0,1,21)
plot(x0, readdlm("val.txt")[:], label = "FEniCS")
plot(x0, SOL[idx], "o", label = "AdFem")

legend()
xlabel("x")
ylabel("\$u(x, 0.5, 0.5)\$")
savefig("section_plot_mfem3d.png")