using Revise
using AdFem
using PyPlot


β = 1/4; γ = 1/2
a = b = 0.1
mmesh = Mesh_v3(50,50, 1.0/50)
NT = 500
Δt = 2/NT 
bdedge = bcedge(mmesh)

ρ = ones(get_ngauss(mmesh))

M = compute_fem_mass_matrix1(ρ, mmesh)
Z = spzero(mmesh.ndof, mmesh.ndof)
M = [M Z;Z M]

ν = 0.0ones(get_ngauss(mmesh))
E = 0.1*ones(get_ngauss(mmesh))

D = compute_plane_strain_matrix(E, ν)
K = compute_fem_stiffness_matrix(D, mmesh)
# K = constant(compute_fem_laplace_matrix1(ones(get_ngauss(mmesh)), mmesh))
# K = [K Z;Z K]
# Nbd = size(bdedge, 1)
# ρe = ones(4Nbd)
# cs = ones(4Nbd)
# cp = ones(4Nbd)
# C = -compute_absorbing_boundary_condition_matrix(ρe, cs, cp, bdedge, mmesh)

C = 0*(a*M + b*K)

function ricker(;dt = 0.05, f0 = 5.0)
    nw = 2/(f0*dt)
    nc = floor(Int, nw/2)
    t = dt*collect(-nc:1:nc)
    b = (π*f0*t).^2
    w = @. (1 - 2b)*exp(-b)
end

F = zeros(NT, 2mmesh.ndof)
r = ricker()
xy = gauss_nodes(mmesh)
S = @. exp(-100*((xy[:,1]-0.5)^2+(xy[:,2]-0.5)^2))
for i = 1:min(length(r), NT)
    f1 = S * r[i]
    # f1 = zeros(get_ngauss(mmesh))
    # f1[3900] = r[i]
    F[i,:] = compute_fem_source_term(-f1, -f1, mmesh)
end



F = constant(F)
en = ExplicitNewmark(M, C, K, Δt)
u = TensorArray(NT+1)
u = write(u, 1, zeros(2mmesh.ndof))
u = write(u, 2, zeros(2mmesh.ndof))

function condition(i, arr...)
    i <= NT+1
end
function body(i, arr...)
    u = arr[1]
    u0 = read(u, i-2)
    u1 = read(u, i-1)
    u = write(u, i, step(en, u0, u1, F[i-1]))
    i+1, u 
end
i = constant(3, dtype = Int32)
_, out = while_loop(condition, body, [i, u])
U = stack(out)


sess = Session(); init(sess)
uarr = run(sess, U)

m = mean(uarr[:, 1:mmesh.ndof])
s = std(uarr[:, 1:mmesh.ndof])
vmin = -3s 
vmax = 3s
function update(i)
    clf()
    visualize_scalar_on_fem_points(uarr[i,1:mmesh.ndof], mmesh, vmin = vmin, vmax = vmax)
    title("i = $i")
end
p = animate(update, 1:(NT÷20):NT)
saveanim(p, "udir.gif")

m = mean(uarr[:, mmesh.ndof+1:end])
s = std(uarr[:, mmesh.ndof+1:end])
vmin = -3s 
vmax = 3s
function update(i)
    clf()
    visualize_scalar_on_fem_points(uarr[i,mmesh.ndof+1:end], mmesh, vmin = vmin, vmax = vmax)
    title("i = $i")
end
p = animate(update, 1:(NT÷20):NT)
saveanim(p, "vdir.gif")

