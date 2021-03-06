using Revise
using AdFem
using PyCall
using LinearAlgebra
using PyPlot
using SparseArrays
using MAT
using ADCMEKit
np = pyimport("numpy")
using PyPlot
using SpecialFunctions
include("viscosity_accel/viscosity_accel.jl")

ADCME.options.sparse.auto_reorder = false
# matplotlib.use("Agg")
close("all")


# simulation parameter setup
n = 20
NT = 300
ρ = 0.1 # design variable in α-schemes
m = 4n 
h = 1 / n 
Δt = 3000. / NT 
density = 100.

mode = "data"
# mode = "inv" 

# coordinates
xo = zeros((m + 1) * (n + 1))
yo = zeros((m + 1) * (n + 1))
for i = 1:m + 1
    for j = 1:n + 1
        idx = (j - 1) * (m + 1) + i 
        xo[idx] = (i - 1) * h 
        yo[idx] = (j - 1) * h 
    end
end

# Dirichlet boundary condition on three sides, and Neumann boundary condition (traction-free) on the top
bdnode = bcnode("left | lower | right", m, n, h)

# viscoelasticity η and shear modulus μ
ηf = (x, y)->begin 
    if y <= 0.25
        return 100.
    elseif y <= 0.6
        return 2.
    else 
      return 1
    end
end

if mode == "data"
    η = constant(eval_f_on_gauss_pts(ηf, m, n, h))
    # η = ones(4*m*n)
else
  
  ### 2D inversion
    # η = Variable(eval_f_on_gauss_pts(ηf, m, n, h))

  ### 1D layer inversion
    η_ = [100. * ones(5); ones(15)]
    global v_var = [constant(ones(5)); Variable(2.5ones(n - 5))]
    η = v_var .* η_

    ## invert log η
    # v_var = Variable(log(2.5) * ones(n))
    # η = exp(v_var)

    ## inv 1/η
    # η_ = [10000. * ones(5); ones(15)]
    # global v_var = [constant(ones(5)); Variable(1. / 2.5 * ones(n - 5))]
    # η = 1/v_var .* η_

    # v_var = Variable(2.5 * ones(n))
    # η = v_var

    global η = layer_model(η, m, n, h)

  ### uniform model inversion
  # global v_var = Variable(2.5)*ones(n)
  # η = v_var
  # global η = layer_model(η, m, n, h)

  ## debug
  # global η = placeholder(eval_f_on_gauss_pts(ηf, m, n, h)) 
end 


## DEBUG
# pl = placeholder([1.0])

μ = 0.001 * constant(ones(4m * n))

### uniform model inversion
# μ_ = Variable(2.0)
# μ = μ_ * 0.001 * constant(ones(4m*n))

# linear elasticity matrix 
coef = 2μ * η / (η + μ * Δt) 
# mapH = c->begin 
#     c * diagm(0 => ones(2))
# end
# H = map(mapH, coef)
H = compute_space_varying_tangent_elasticity_matrix(coef, m, n, h)

# H = reshape([coef constant(zeros(4*m*n)) coef constant(zeros(4*m*n))], (4*m*n,2,2))

Δt = Δt * ones(NT)

# generalized mass matrix and stiffness matrix 
M = density * constant(compute_fem_mass_matrix1(m, n, h))
K = compute_fem_stiffness_matrix1(H, m, n, h)
C = spzero((m + 1) * (n + 1))

C = 0.1 * M + 0.1 * K 

# fixed displacement 
db = zeros((m + 1) * (n + 1))
for j = 1:n + 1
    idx = (j - 1) * (m + 1) + 1
    if j <= div(n, 4)
        db[idx] = 1.
    else
        db[idx] = (1 - (j - div(n, 4)) / (3div(n, 4)))
        # db[idx] = 1.
    end
end

# cast the problem into homogeneous Dirichlet problem
idof = ones(Bool, (m + 1) * (n + 1))
idof[bdnode] .= false
idof = findall(idof)

d0 = db
v0 = zeros((m + 1) * (n + 1))
# v0[bcnode("left", m, n, h)] .= 1.0
a0 = vector(idof, M[idof, idof] \ ((-K * db)[idof]), (m + 1) * (n + 1))

function antiplane_visco_αscheme(M::Union{SparseTensor,SparseMatrixCSC}, 
  K::Union{SparseTensor,SparseMatrixCSC}, 
  d0::Union{Array{Float64,1},PyObject}, 
  v0::Union{Array{Float64,1},PyObject}, 
  a0::Union{Array{Float64,1},PyObject}, 
  Δt::Array{Float64}; 
  ρ::Float64 = 1.0)
    nt = length(Δt)

    αm = (2ρ - 1) / (ρ + 1)
    αf = ρ / (1 + ρ)
    γ = 1 / 2 - αm + αf 
    β = 0.25 * (1 - αm + αf)^2
    d = length(d0)

    M = isa(M, SparseMatrixCSC) ? constant(M) : M
    K = isa(K, SparseMatrixCSC) ? constant(K) : K
    d0, v0, a0, Δt = convert_to_tensor([d0, v0, a0, Δt], [Float64, Float64, Float64, Float64])

    Kterm = (1 - αf) * K * β * Δt[1]^2
    Kterm = fem_impose_Dirichlet_boundary_condition1(Kterm, bdnode, m, n, h)[1]

    A = (1 - αm) * M + Kterm
    A =  fem_impose_Dirichlet_boundary_condition1(A, bdnode, m, n, h)[1]

  # for visco_solve
  # ii, jj, vv = find(Kterm)
  # opp = push_matrices(A, Kterm)
    
    A = factorize(A)

    function equ(dc, vc, ac, dt, εc, σc, i)
        dn = dc + dt * vc + dt^2 / 2 * (1 - 2β) * ac 
        vn = vc + dt * ((1 - γ) * ac)

        df = (1 - αf) * dn + αf * dc
        vf = (1 - αf) * vn + αf * vc 
        am = αm * ac 

        σ_dt = (1 - αf) * dt
        Σ = 2 * repeat(μ .* η / (η + μ * σ_dt), 1, 2) .* εc - repeat(η / (η + μ * σ_dt), 1, 2) .* σc
        Force = compute_strain_energy_term1(Σ, m, n, h)

        rhs = - (M * am + C * vf + K * df) + Force
        rhs = scatter_update(rhs, bdnode, zeros(length(bdnode))) 

    # @info "visco_solve"
    # op = tf.print(norm(visco_solve(rhs,vv,opp)-A\rhs))
    # rhs = bind(rhs, op)
    # visco_solve(rhs,vv,opp)
    # @info "A \ rhs"
        A \ rhs
    end

    function condition(i, tas...)
        return i <= nt
    end

    function body(i, tas...)
        dc_arr, vc_arr, ac_arr, σc_arr = tas
        dc = read(dc_arr, i)
        vc = read(vc_arr, i)
        ac = read(ac_arr, i)
        σc = read(σc_arr, i)
        εc = eval_strain_on_gauss_pts1(dc, m, n, h)

        y = equ(dc, vc, ac, Δt[i], εc, σc, i)
        dn = dc + Δt[i] * vc + Δt[i]^2 / 2 * ((1 - 2β) * ac + 2β * y)
        vn = vc + Δt[i] * ((1 - γ) * ac + γ * y)

        εn = eval_strain_on_gauss_pts1(dn, m, n, h)
        σn = 2 * repeat(μ .* η / (η + μ * Δt[i]), 1, 2) .* (εn - εc) + repeat(η / (η + μ * Δt[i]), 1, 2) .* σc

        i + 1, write(dc_arr, i + 1, dn), write(vc_arr, i + 1, vn), write(ac_arr, i + 1, y), write(σc_arr, i + 1, σn)
    end

    dM = TensorArray(nt + 1); vM = TensorArray(nt + 1); aM = TensorArray(nt + 1); σM = TensorArray(nt + 1);
    dM = write(dM, 1, d0)
    vM = write(vM, 1, v0)
    aM = write(aM, 1, a0)

    ε0 = eval_strain_on_gauss_pts1(d0, m, n, h)
    σ0 = batch_matmul(H, ε0)
    σM = write(σM, 1, σ0)

    i = constant(1, dtype = Int32)
    _, d, v, a = while_loop(condition, body, [i,dM, vM, aM, σM], parallel_iterations = 1)
    set_shape(stack(d), (nt + 1, length(a0))), set_shape(stack(v), (nt + 1, length(a0))), set_shape(stack(a), (nt + 1, length(a0)))
end

d, v, a = antiplane_visco_αscheme(M, K, d0, v0, a0, Δt, ρ = ρ)


## DEBUG
## αscheme from ADCME 
# function solver(A, rhs)
#   A, _ = fem_impose_Dirichlet_boundary_condition1(A, bdnode, m, n, h)
#   rhs = scatter_update(rhs, bdnode, zeros(length(bdnode)))
#   A\rhs
# end
# d, v, a = αscheme(M, C, K, zeros(NT, (m+1)*(n+1)), d0, v0, a0, Δt; solve = solver)

function observation(d, v)
    idx = 10:m 
    idx_plus = idx .+ 1
    idx_minus = idx .- 1
    idx_t = div(NT, 6):NT + 1
    dobs = d[idx_t, idx]
    vobs = v[idx_t, idx]
    strain_rate_obs = (v[idx_t, idx_plus] - v[idx_t, idx_minus]) / 2h 
    return dobs, vobs, strain_rate_obs
end

dobs, vobs, strain_rate_obs = observation(d, v)

sess = Session(); init(sess)

if mode == "data"
    @time disp, vel, strain_rate = run(sess, [dobs, vobs, strain_rate_obs]) 
    matwrite("data-visc.mat", Dict("vel" => vel, "strain_rate" => strain_rate, "disp" => disp))
    # visulization()
    @info "Data Generated."
end

cb = (vs, iter, loss)->begin 
    if mod(iter, 10) == 0 || iter < 10
        # clf()
        # plot(vs[1])
        x_tmp, y_tmp, z_tmp = visualize_scalar_on_gauss_points(vs[2], m, n, h)
        clf()
        pcolormesh(x_tmp, y_tmp, z_tmp', vmax=3, vmin = 0, rasterized=true)
        colorbar(shrink=0.2)
        axis("scaled")
        xlabel("x")
        ylabel("y")
        gca().invert_yaxis()
        title("Iter = $iter")
        savefig("figures2/inv_$(lpad(iter,5,"0")).png", bbox_size="tight")
        matwrite("results2/inv_$(lpad(iter,5,"0")).mat", Dict("var" => vs[1], "eta" => vs[2]))
    end
    printstyled("[#iter $iter] eta = $(vs[1])\n", color = :green)
end

if mode != "data"
    data = matread("data-visc.mat")
    global disp, vel, strain_rate =  data["disp"], data["vel"], data["strain_rate"]
    # global loss = 1e10 * sum((vel - vobs)^2)
    global loss = 1e10 * sum((disp - dobs)^2)
    @info run(sess, loss)
    global loss_ = BFGS!(sess, loss * 1e10, vars = [v_var, η], callback = cb)
end

## DEBUG
# η_ = [10000. * ones(5); ones(15)]
# v_var = [constant(ones(5)); Variable(2.5ones(n - 5))]
# η0 = v_var .* η_
# η0 = layer_model(η0, m, n, h)
# sess = Session();init(sess)
# @show run(sess, loss)
# η0 = run(sess, η0)
# lineview(sess, η, loss, eval_f_on_gauss_pts(ηf, m, n, h), η0)
# gradview(sess, η, loss, η0)
# gradview(sess, pl, loss, [1.0])

# @time run(sess, loss)

## plot model

x, y, z = visualize_scalar_on_gauss_points(run(sess, η), m, n, h)
figure()
pcolormesh(x, y, z', vmax=3, vmin = 0, rasterized=true)
colorbar(shrink=0.2)
axis("scaled")
xlabel("x")
ylabel("y")
gca().invert_yaxis()
if mode == "data"
  title(L"""Viscosity Model
  (top elastic layer: $η=100$)""")
  savefig("viscosity-model.png", bbox_size="tight")
else
  title("Inverted Model")
  savefig("viscosity-inversion.png", bbox_size="tight")
end

d_, v_, a_ = run(sess, [d, v, a])

figure()
pl, = plot([], [], "o-", markersize = 3)
t = title("0")
xi = (0:m) * h 
xlim(-h, (m + 1) * h)
xlabel("Distance")
ylim(-0.1, 1.1)
ylabel("Displacement")
tight_layout()
function update(i)
    pl.set_data(xi[:], d_[i,1:m + 1])
    t.set_text("time = $(i * Δt[1])")
end
p = animate(update, [NT÷6:10:NT+1;])
# p = animate(update, [NT÷2:5:NT+1])
# saveanim(p, "displacement.gif")

# figure()
# pl, = plot([], [], "o-", markersize = 3)
# t = title("time = 0")
# xi = (0:m)*h 
# xlim(-h, (m+1)*h)
# xlabel("Distance")
# ylim(-0.0001, 0.0005)
# ylabel("Velocity")
# tight_layout()
# function update(i)
#   pl.set_data(xi[:], v_[i,1:m+1])
#   t.set_text("time = $(i*Δt[1])")
# end
# p = animate(update, [NT÷6:10:NT+1;])
# saveanim(p, "velocity.gif")

# figure()
# pl, = plot([], [], "o-", markersize = 3)
# t = title("time = 0")
# xi = (0:m)*h 
# xlim(-h, (m+1)*h)
# xlabel("Distance")
# ylim(-0.001, 0.002)
# ylabel("Strain Rate")
# tight_layout()
# function update(i)
#   pl.set_data(xi[1:end-1], (v_[i,2:m+1]-v_[i,1:m])/h)
#   t.set_text("time = $(i*Δt[1])")
# end
# p = animate(update, [NT÷6:10:NT+1;])
# saveanim(p, "strain_rate.gif")
