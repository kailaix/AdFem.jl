using Revise
using AdFem
using PyPlot 
using SparseArrays


cmesh = CrackMesh(50,50,1/50,15)
mmesh = cmesh.mesh 


NT = 500
Δt = 2/NT 

Ca = spzeros(2cmesh.nslip, 2mmesh.ndof)
ls = zeros(2cmesh.nslip)
for i = 1:cmesh.nslip
    u = cmesh.upper[i+1] 
    l = cmesh.lower[i+1] 
    Ca[i, u] = 1.0
    Ca[i, l] = -1.0
    ls[i] = 0.0

    u = cmesh.upper[i+1] + mmesh.ndof 
    l = cmesh.lower[i+1] + mmesh.ndof 
    Ca[i+cmesh.nslip, u] = 1.0
    Ca[i+cmesh.nslip, l] = -1.0
    ls[i+cmesh.nslip] = 0.0
end
Ca = constant(Ca)


upbd = bcnode((x,y)->y>1.0-0.001, mmesh)
downbd = bcnode((x,y)->y<0.001, mmesh)

ν = 0.1
E = 1.0
D = compute_plane_strain_matrix(E, ν)
K = constant(compute_fem_stiffness_matrix(D, mmesh))

rhs = zeros(2mmesh.ndof)
K, rhs = impose_Dirichlet_boundary_conditions(
    K, rhs, 
    [upbd; downbd; upbd .+ mmesh.ndof; downbd .+ mmesh.ndof],
    [0.001*ones(length(upbd)); zeros(length(downbd)+length(upbd)+length(downbd))]
)

A = [K Ca';Ca spzero(2cmesh.nslip)]
rhs = [rhs; constant(ls)]
u = A\rhs

u = u[1:2mmesh.ndof]

sess = Session(); init(sess)

U = run(sess, u)


# close("all")
# visualize_displacement(100*U, mmesh)
# savefig("mesh.png")

Se = compute_von_mises_stress_term(D, U, mmesh)
close("all")
visualize_scalar_on_gauss_points(Se, mmesh)
savefig("stress.png")