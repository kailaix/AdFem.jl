using ADCME
using LinearAlgebra 
using MAT
using PoreFlow
using PyPlot 
using SparseArrays

function Q_exact(x, y, t)
    (2*x^2*y*(x - 1)*(2*y - 1)*(x + y + 1) + x*y*(y - 1)*(y^2 + 1)^2*(-x - y*(2*x - 1)*(y - 1) + 1)*(x + y + 1) + x*(x - 1)*(2*y - 1)*(y^2 + 1)^2 - (y^2 + 1)*(x + y + 1)*(2*x*(x - 1)*(x + (x^2 + 1)*(y^2 + 1)) + 2*y*(x + (x^2 + 1)*(y^2 + 1))*(y - 1) + y*(2*x - 1)*(y - 1)*(2*x*(y^2 + 1) + 1)))*exp(-t)/((y^2 + 1)^2*(x + y + 1))
end

function T_exact(x, y, t)
    x * (1-x) * y * (1-y) * exp(-t)
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
NT = 10
t_final = 1.0
dt = t_final/NT 
#---------------------------------------------
# discretized governing equation 
#  J * u^{n+1} - J * u^n = - v ⋅ grad u + K * u^{n+1} + F^{n+1}
#---------------------------------------------
xy = fem_nodes(m, n, h)
x, y = xy[:,1], xy[:,2]
T0 = @. T_exact(x, y, 0.0)
u = @. u_exact(x,y)
v = @. v_exact(x,y)
k = @. k_exact(x, y)

bd = bcnode("all", m, n, h)

ugauss = fem_to_gauss_points(u, m, n, h)
vgauss = fem_to_gauss_points(v, m, n, h)
kgauss = fem_to_gauss_points(k, m, n, h)

# precompute source term 
Q = zeros(NT, (m+1)*(n+1))
for i = 1:NT 
    t = i*dt 
    f_value = eval_f_on_gauss_pts((x,y)->Q_exact(x, y, t), m, n, h)
    Q[i,:] = compute_fem_source_term1(f_value, m, n, h)
end
Q = constant(Q)

Mass = constant(compute_fem_mass_matrix1(m, n, h))
Advection = constant(compute_fem_advection_matrix1(constant(ugauss), constant(vgauss), m, n, h))
Laplace = constant(compute_fem_laplace_matrix1(kgauss, m, n, h))
A = 1/dt * Mass + Advection + Laplace
A, _ = fem_impose_Dirichlet_boundary_condition1(A, bd, m, n, h)

function solve_chtenergy_one_step(T, source)
    Tgauss = fem_to_gauss_points(T, m, n, h)
    b = constant(compute_fem_source_term1(constant(Tgauss / dt), m, n, h)) + source
    b = scatter_update(b, bd, zeros(length(bd)))
    sol = A\b
end


# ---------------------------------------------
function condition(i, T_arr)
    i<=NT+1
end

function body(i, T_arr)
    T = read(T_arr, i-1)
    T_new = solve_chtenergy_one_step(T, Q[i-1])
    T_arr = write(T_arr, i, T_new)
    i+1, T_arr
end

i = constant(2, dtype=Int32)

T_arr = TensorArray(NT+1)
T_arr = write(T_arr, 1, T0)
_, T_arr = while_loop(condition, body, [i, T_arr])
T_arr = set_shape(stack(T_arr), (NT+1, (m+1)*(n+1)))

# ---------------------------------------------------
# create a session and run 
sess = Session(); init(sess)
T_computed = run(sess, T_arr)

matwrite("chtenergy_transient_data.mat", Dict("T"=>T_computed) )

T_analytical = @. T_exact(x, y, t_final)

#---------------------------------------------
# visualize numerical solution and exact solution
figure(figsize=(10,4))
subplot(121)
visualize_scalar_on_fem_points(T_computed[end,:], m, n, h)
subplot(122)
visualize_scalar_on_fem_points(T_analytical, m, n, h)
