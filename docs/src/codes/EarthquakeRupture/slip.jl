using Revise
using AdFem
using PyPlot 
using SparseArrays 

cmesh = CrackMesh(50,50,1/50,24)
mmesh = cmesh.mesh 
β = 1/4; γ = 1/2
a = b = 0.1

NT = 500
Δt = 2/NT 

Ca = spzeros(cmesh.nslip, 2mmesh.ndof)
ls = zeros(cmesh.nslip)
for i = 1:cmesh.nslip
    u = cmesh.upper[i+1] 
    l = cmesh.lower[i+1] 
    Ca[i, u] = 1.0
    Ca[i, l] = -1.0
    ls[i] = cmesh.h/5
end
Ca = constant(Ca)



bdedge = cmesh.bdedge

ρ = ones(get_ngauss(mmesh))

M = compute_fem_mass_matrix1(ρ, mmesh)
Z = spzero(mmesh.ndof, mmesh.ndof)
M = [M Z;Z M]

ν = 0.1ones(get_ngauss(mmesh))
E = ones(get_ngauss(mmesh))

D = compute_plane_strain_matrix(E, ν)
K = compute_fem_stiffness_matrix(D, mmesh)

Nbd = size(bdedge, 1)
ρe = ones(4Nbd)
mu = 1/(2*(1+0.3))
lambda = 0.3/(1+0.3)/(1-2*0.3) 
cs = ones(4Nbd) * sqrt(mu)
cp = ones(4Nbd) * sqrt(lambda + 2mu)
C = compute_absorbing_boundary_condition_matrix(ρe, cs, cp, bdedge, mmesh)


function ricker(;dt = 0.05, f0 = 5.0)
    nw = 2/(f0*dt)
    nc = floor(Int, nw/2)
    t = dt*collect(-nc:1:nc)
    b = (π*f0*t).^2
    w = @. (1 - 2b)*exp(-b)
end

F = zeros(NT, 2mmesh.ndof+cmesh.nslip)
r = ricker()
xy = gauss_nodes(mmesh)
S = @. exp(-20*((xy[:,1]-0.5)^2+(xy[:,2]-0.5)^2))

for i = 1:min(length(r), NT)
    f1 = S * r[i]
    F[i,:] = [compute_fem_source_term(zero(f1), zero(f1), mmesh);ls]
end



F = constant(F)

    
en = ExplicitNewmark(M, C, K, Δt, factorize_A = false)
en.A = factorize([en.A Ca'
        Ca spzero(cmesh.nslip)])
en.C = [en.C spzero(2mmesh.ndof, cmesh.nslip)
        spzero(cmesh.nslip, 2mmesh.ndof) spzero(cmesh.nslip)]
en.B = [en.B spzero(2mmesh.ndof, cmesh.nslip)
    spzero(cmesh.nslip, 2mmesh.ndof) spzero(cmesh.nslip)]



u = TensorArray(NT+1)
u = write(u, 1, zeros(2mmesh.ndof+cmesh.nslip))
u = write(u, 2, zeros(2mmesh.ndof+cmesh.nslip))

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
vmin = -5s
vmax = 5s
function update(i)
    clf()
    visualize_scalar_on_fem_points(uarr[i,1:mmesh.ndof], mmesh, vmin = vmin, vmax = vmax)
    title("i = $i")
end
p = animate(update, 1:(NT÷20):NT)
saveanim(p, "udir.gif")

m = mean(uarr[:, mmesh.ndof+1:2mmesh.ndof])
s = std(uarr[:, mmesh.ndof+1:2mmesh.ndof])
vmin = m - 2s 
vmax = m + 2s
function update(i)
    clf()
    visualize_scalar_on_fem_points(uarr[i,mmesh.ndof+1:2mmesh.ndof], mmesh, vmin = vmin, vmax = vmax)
    title("i = $i")
end
p = animate(update, 1:(NT÷20):NT)
saveanim(p, "vdir.gif")

