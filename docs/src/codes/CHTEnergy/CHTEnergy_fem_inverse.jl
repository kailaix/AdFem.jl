using ADCME
using LinearAlgebra 
using MAT
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

function k_nn(x, y)
    out = fc([x y], [20,20,20,1])^2 + 0.1 # N x 1 
    squeeze(out)
end

#---------------------------------------------
# grid setup
m = 20
n = 20
h = 1/n 

#---------------------------------------------
xy = fem_nodes(m, n, h)
x, y = xy[:,1], xy[:,2]
T0 = @. T_exact(x, y)
u = @. u_exact(x,y)
v = @. v_exact(x,y)
Q = @. Q_exact(x,y)
k = @. k_nn(x, y); k=stack(k)
# k0 = @. k_nn(xy)

# ---------------------------------------------

bd = bcnode("all", m, n, h)

ugauss = fem_to_gauss_points(u, m, n, h)
vgauss = fem_to_gauss_points(v, m, n, h)
kgauss = fem_to_gauss_points(k, m, n, h)
Qgauss = fem_to_gauss_points(Q, m, n, h)

Advection = constant(compute_fem_advection_matrix1(constant(ugauss), constant(vgauss), m, n, h))
Laplace = compute_fem_laplace_matrix1(kgauss, m, n, h)
A = Advection + Laplace
A, _ = fem_impose_Dirichlet_boundary_condition1(A, bd, m, n, h)
b = constant(compute_fem_source_term1(constant(Qgauss), m, n, h))
b = scatter_update(b, bd, zeros(length(bd)))
T_computed = A\b

T_data = matread("chtenergy_data.mat")["T"]
# loss = mean((T_computed .- T_data)^2)

sample_size = 40
idx = rand(1:(m+1)*(n+1), sample_size)
observed_data = T_data[idx]
loss = mean((T_computed[idx] .- observed_data)^2)

noise = false
noise_level = 0.01
if noise
    noise_ratio = (1 - noise_level) .+ 2 * noise_level * rand(Float64, size(observed_data)) # uniform on (1-noise_level, 1+noise_level)
    observed_data = observed_data .* noise_ratio
end

loss = loss * 1e10
# ---------------------------------------------------
# create a session and run 
max_iter = 100
sess = Session(); init(sess)
loss_ = BFGS!(sess, loss, max_iter)
figure(); semilogy(loss_); savefig("chtenergy_loss.png")

figure(figsize=(14,4));
subplot(131)
visualize_scalar_on_fem_points(k_exact.(x,y), m, n, h); title("conductivity exact")
subplot(132)
visualize_scalar_on_fem_points(run(sess, k), m, n, h); title("conductivity prediction")
subplot(133)
visualize_scalar_on_fem_points(k_exact.(x,y).-run(sess, k), m, n, h); title("conductivity difference")
savefig("chtenergy_k.png")
