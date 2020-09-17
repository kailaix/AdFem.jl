using ADCME
using LinearAlgebra 
using PoreFlow
using PyPlot 
using SparseArrays

function Q_exact(x, y)
    # (2*x^2*y*(x - 1)*(2*y - 1) + 1.0*(y^2 + 1)^2*(x*y*(x - 1) + x*y*(y - 1) + x*(x - 1)*(y - 1) + y*(x - 1)*(y - 1)) - (y^2 + 1)*(2*x*(x - 1)*(x + (x^2 + 1)*(y^2 + 1)) + 2*y*(x + (x^2 + 1)*(y^2 + 1))*(y - 1) + y*(2*x - 1)*(y - 1)*(2*x*(y^2 + 1) + 1)))/(y^2 + 1)^2
    # (2*x^2*y*(x - 1)*(2*y - 1) + x*(y^2 + 1)^2*(-y^2*(2*x - 1)*(y - 1)^2 + 1.0*y*(x - 1) + 1.0*(x - 1)*(y - 1)) - (y^2 + 1)*(2*x*(x - 1)*(x + (x^2 + 1)*(y^2 + 1)) + 2*y*(x + (x^2 + 1)*(y^2 + 1))*(y - 1) + y*(2*x - 1)*(y - 1)*(2*x*(y^2 + 1) + 1)))/(y^2 + 1)^2
    (2*x^2*y*(x - 1)*(2*y - 1)*(x + y + 1) - x*y^2*(2*x - 1)*(y - 1)^2*(y^2 + 1)^2*(x + y + 1) + x*(x - 1)*(2*y - 1)*(y^2 + 1)^2 - (y^2 + 1)*(x + y + 1)*(2*x*(x - 1)*(x + (x^2 + 1)*(y^2 + 1)) + 2*y*(x + (x^2 + 1)*(y^2 + 1))*(y - 1) + y*(2*x - 1)*(y - 1)*(2*x*(y^2 + 1) + 1)))/((y^2 + 1)^2*(x + y + 1))
end

function T_exact(x, y)
    x * (1-x) * y * (1-y)
end

function u_exact(x, y)
    x * y * (1-y)
end

function v_exact(x, y)
    1 / (1 + x + y)
end

function k_exact(x, y)
    1 + x^2 + x / (1 + y^2)
end

# function k_nn(xy) # xy shape N (=m*n) x 2
#     out = fc(xy, [20,20,20,1])^2 + 0.1 # N x 1 
#     squeeze(out)
# end

#---------------------------------------------
# grid setup
m = 20
n = 20
h = 1/n 

mesh = Mesh(m, n, h)
nnode = size(mesh.nodes, 1)
nelem = size(mesh.elems, 1)

#---------------------------------------------
# discretized governing equation 
#  J * u^{n+1} - J * u^n = - v ⋅ grad u + K * u^{n+1} + F^{n+1}
#---------------------------------------------
xy = mesh.nodes
x, y = xy[:,1], xy[:,2]
T0 = @. T_exact(x, y)
u = @. u_exact(x,y)
v = @. v_exact(x,y)
Q = @. Q_exact(x,y)
k = @. k_exact(x, y)
# k0 = @. k_nn(xy)

# ---------------------------------------------

bd = Int64[]
for j = 1:m+1
    push!(bd, j)
    push!(bd, n*(m+1)+j)
end
for i = 2:n
    push!(bd, (i-1)*(m+1)+1)
    push!(bd, (i-1)*(m+1)+m+1)
end

ugauss = fem_to_gauss_points(u, mesh)
vgauss = fem_to_gauss_points(v, mesh)
kgauss = fem_to_gauss_points(k, mesh)
Qgauss = fem_to_gauss_points(Q, mesh)

Advection = constant(compute_fem_advection_matrix1(constant(ugauss), constant(vgauss), mesh))
Laplace = constant(compute_fem_laplace_matrix1(constant(kgauss), mesh))
A = Advection + Laplace
A, _ = fem_impose_Dirichlet_boundary_condition1(A, bd, mesh)
b = constant(compute_fem_source_term1(constant(Qgauss), mesh))
b = scatter_update(b, bd, zeros(length(bd)))
sol = A\b

# ---------------------------------------------------
# create a session and run 
sess = Session(); init(sess)
T_computed = run(sess, sol)

#---------------------------------------------
# visualize numerical solution and exact solution
figure(figsize=(10,4))
subplot(121)
visualize_scalar_on_fem_points(T_computed, mesh, with_mesh=true)
subplot(122)
visualize_scalar_on_fem_points(T0, mesh, with_mesh=true)
savefig("forward_solution_unstructured.png")
close("all")