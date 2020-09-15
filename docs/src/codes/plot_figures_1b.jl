using ADCME
using LinearAlgebra
using MAT
using PoreFlow
using PyPlot
using SparseArrays


SMALL_SIZE = 22
MEDIUM_SIZE = 24
BIGGER_SIZE = 24

plt.rc("font", size=SMALL_SIZE)          # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Example 1: 

function u_exact(x,y)
    if y == 0.0
        return 1.0
    else
        return 0.0    
    end
end

function v_exact(x,y)
    0.0
end

function p_exact(x,y)
    0.0
end

function ffunc(x, y)
    0.0
end
function gfunc(x, y)
    0.0
end

function hfunc(x,y)
    0.0
end

function nu_exact(x, y)
    1 + 6 * x^2 + x / (1 + 2 * y^2)
end

function nu_nn(x, y)
    1.5 + fc([x y], [20,20,20,1])|>squeeze
end

m = 20
n = 20
h = 1/n
# nu = 0.01


F1 = compute_fem_source_term1(eval_f_on_gauss_pts(ffunc, m, n, h), m, n, h)
F2 = compute_fem_source_term1(eval_f_on_gauss_pts(gfunc, m, n, h), m, n, h)
H = h^2*eval_f_on_fvm_pts(hfunc, m, n, h)
B = constant(compute_interaction_matrix(m, n, h))

# compute F
xy = fem_nodes(m, n, h)
x, y = xy[:,1], xy[:,2]
u0 = @. u_exact(x,y)
v0 = @. v_exact(x,y)

xy = fvm_nodes(m, n, h)
x, y = xy[:,1], xy[:,2]
p0 = @. p_exact(x,y)
# nu_gauss = Variable(ones(4*m*n))
# nu_fvm = Variable(ones(m*n))
nu_fvm = matread("sess_figures/ex1/ones50.mat")["Variablecolon0"]
nu_fvm = constant(nu_fvm)
nu_gauss = reshape(repeat(nu_fvm, 1, 4), (-1,))

Laplace = compute_fem_laplace_matrix1(nu_gauss, m, n, h)


# Laplace = constant(nu*compute_fem_laplace_matrix1(m, n, h))
function compute_residual(S)
    u, v, p = S[1:(m+1)*(n+1)], S[(m+1)*(n+1)+1:2(m+1)*(n+1)], S[2(m+1)*(n+1)+1:2(m+1)*(n+1)+m*n]
    G = eval_grad_on_gauss_pts([u;v], m, n, h)
    ugauss = fem_to_gauss_points(u, m, n, h)
    vgauss = fem_to_gauss_points(v, m, n, h)
    ux, uy, vx, vy = G[:,1,1], G[:,1,2], G[:,2,1], G[:,2,2]

    interaction = compute_interaction_term(p, m, n, h) # julia kernel needed
    f1 = compute_fem_source_term1(ugauss.*ux, m, n, h)
    f2 = compute_fem_source_term1(vgauss.*uy, m, n, h)
    f3 = -interaction[1:(m+1)*(n+1)]
    f4 = Laplace*u 
    f5 = -F1
    F = f1 + f2 + f3 + f4 + f5 

    g1 = compute_fem_source_term1(ugauss.*vx, m, n, h)
    g2 = compute_fem_source_term1(vgauss.*vy, m, n, h)
    g3 = -interaction[(m+1)*(n+1)+1:end]    
    g4 = Laplace*v 
    g5 = -F2
    G = g1 + g2 + g3 + g4 + g5

    H0 = -B * [u;v] + H

    R = [F;G;H0]
    return R
end

function compute_jacobian(S)
    u, v, p = S[1:(m+1)*(n+1)], S[(m+1)*(n+1)+1:2(m+1)*(n+1)], S[2(m+1)*(n+1)+1:2(m+1)*(n+1)+m*n]
    G = eval_grad_on_gauss_pts([u;v], m, n, h)
    ugauss = fem_to_gauss_points(u, m, n, h)
    vgauss = fem_to_gauss_points(v, m, n, h)
    ux, uy, vx, vy = G[:,1,1], G[:,1,2], G[:,2,1], G[:,2,2]

    M1 = constant(compute_fem_mass_matrix1(ux, m, n, h))
    M2 = constant(compute_fem_advection_matrix1(constant(ugauss), constant(vgauss), m, n, h)) # a julia kernel needed
    M3 = Laplace
    Fu = M1 + M2 + M3 

    Fv = constant(compute_fem_mass_matrix1(uy, m, n, h))

    N1 = constant(compute_fem_mass_matrix1(vy, m, n, h))
    N2 = constant(compute_fem_advection_matrix1(constant(ugauss), constant(vgauss), m, n, h))
    N3 = Laplace
    Gv = N1 + N2 + N3 

    Gu = constant(compute_fem_mass_matrix1(vx, m, n, h))

    J0 = [Fu Fv
          Gu Gv]
    J = [J0 -B'
        -B spdiag(zeros(size(B,1)))]
end

NT = 5
# S = zeros(m*n+2(m+1)*(n+1), NT+1)
# S[1:m+1, 1] = ones(m+1,1)

bd = bcnode("all", m, n, h)
# bd = [bd; bd .+ (m+1)*(n+1); ((1:m) .+ 2(m+1)*(n+1))]
bd = [bd; bd .+ (m+1)*(n+1); 2*(m+1)*(n+1)+(n-1)*m+1:2*(m+1)*(n+1)+(n-1)*m+2] # only apply Dirichlet to velocity


function solve_steady_cavityflow_one_step(S)
    residual = compute_residual(S)
    J = compute_jacobian(S)
    
    J, _ = fem_impose_Dirichlet_boundary_condition1(J, bd, m, n, h)
    # residual[bd] .= 0.0
    residual = scatter_update(residual, bd, zeros(length(bd)))

    d = J\residual
    S_new = S - d
    return S_new
end


function condition(i, S_arr)
    i <= NT + 1
end

function body(i, S_arr)
    S = read(S_arr, i-1)
    op = tf.print("i=",i)
    i = bind(i, op)
    S_new = solve_steady_cavityflow_one_step(S)
    S_arr = write(S_arr, i, S_new)
    return i+1, S_arr
end

S_arr = TensorArray(NT+1)
S_arr = write(S_arr, 1, [u0; v0; p0])

i = constant(2, dtype=Int32)

_, S = while_loop(condition, body, [i, S_arr])
S = set_shape(stack(S), (NT+1, 2*(m+1)*(n+1)+m*n))

S_true = matread("sess_figures/ex1/steady_cavity_data.mat")["V"]
loss = mean((S[end,1:2*(m+1)*(n+1)] - S_true[end,1:2*(m+1)*(n+1)])^2)
loss = loss * 1e10

# vmax = maximum(nu_exact.(x,y))
# vmin = minimum(nu_exact.(x,y))

sess = Session(); init(sess)
# ADCME.load(sess, "sess_figures/ex1/ones50.mat")
nu_fvm = matread("sess_figures/ex1/ones50.mat")["Variablecolon0"]


vmax = 6.8
vmin = -0.5

figure();visualize_scalar_on_fvm_points(nu_exact.(x,y), m, n, h, vmax=vmax, vmin=vmin); tight_layout()
savefig("sess_figures/ex1/ones/viscosity_exact.png")

# figure();visualize_scalar_on_gauss_points(run(sess, nu_gauss), m, n, h); #gca().invert_yaxis()
# savefig("sess_figures/ex1/ones/viscosity_prediction.png")

figure();visualize_scalar_on_gauss_points(run(sess, nu_gauss), m, n, h, vmax=vmax, vmin=vmin); tight_layout() ;#gca().invert_yaxis()
savefig("sess_figures/ex1/ones/viscosity_prediction.png")

# figure();visualize_scalar_on_fvm_points(nu_exact.(x,y).-run(sess, nu_gauss)[1:4:end], m, n, h); 
# savefig("sess_figures/ex1/ones/viscosity_error.png")

figure();visualize_scalar_on_fvm_points(nu_exact.(x,y).-nu_fvm, m, n, h); tight_layout()
savefig("sess_figures/ex1/ones/viscosity_error.png")


vmax = 41
vmin = -39
figure();visualize_scalar_on_fvm_points(S_true[end, 2*(m+1)*(n+1)+1:end], m, n, h, vmax=vmax, vmin=vmin); tight_layout()
savefig("sess_figures/ex1/ones/pressure_data.png")

figure();visualize_scalar_on_fvm_points(run(sess, S[end,2*(m+1)*(n+1)+1:end]), m, n, h, vmax=vmax, vmin=vmin); tight_layout()
savefig("sess_figures/ex1/ones/pressure_prediction.png")

figure();visualize_scalar_on_fvm_points(S_true[end, 2*(m+1)*(n+1)+1:end] - run(sess, S[end,2*(m+1)*(n+1)+1:end]), m, n, h); tight_layout()
savefig("sess_figures/ex1/ones/pressure_error.png")

figure();visualize_scalar_on_fvm_points_nocontour(S_true[end, 2*(m+1)*(n+1)+1:end] - run(sess, S[end,2*(m+1)*(n+1)+1:end]), m, n, h); tight_layout()
savefig("sess_figures/ex1/ones/pressure_error_nocontour.png")


function visualize_scalar_on_fvm_points_nocontour(φ::Array{Float64, 1}, m::Int64, n::Int64, h::Float64)
    φ = reshape(φ, m, n)'|>Array
    m_ = mean(φ)
    s = std(φ)
    vmin, vmax = m_ - 2s, m_ + 2s
    x = (1:m)*h .- 0.5h
    y = (1:n)*h .- 0.5h
    ln = pcolormesh(x, y, φ, vmin= vmin, vmax=vmax)
    colorbar()
    # c = contour(φ[1,:,:], 10, cmap="jet", vmin=vmin,vmax=vmax)
    axis("scaled")
    xlabel("x")
    ylabel("y")
    ln = gca().pcolormesh(x, y, φ, vmin= vmin, vmax=vmax)
    # c = gca().contour(x, y, φ, 10, cmap="jet", vmin=vmin,vmax=vmax)
    gca().invert_yaxis()
end
