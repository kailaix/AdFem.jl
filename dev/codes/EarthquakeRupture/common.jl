using Revise
using AdFem
using PyPlot

function ricker(;dt = 0.005, f0 = 3.0)
    nw = 2/(f0*dt)
    nc = floor(Int, nw/2)
    t = dt*collect(-nc:1:nc)
    b = (π*f0*t).^2
    w = @. (1 - 2b)*exp(-b)
end

n = 50
mmesh = Mesh(n, n, 1/n)
NT = 400
Δt = 2/NT 

xy = gauss_nodes(mmesh)
xg, yg = xy[:,1], xy[:,2]
xy = fem_nodes(mmesh)
x, y = xy[:,1], xy[:,2]

β = zeros(mmesh.ndof)
β_g = zeros(get_ngauss(mmesh))
βprime = zeros(get_ngauss(mmesh))
λ = constant(ones(get_ngauss(mmesh)))
μ = constant(0.5ones(get_ngauss(mmesh)))

nv = [ones(get_ngauss(mmesh)) zeros(get_ngauss(mmesh))]

R = 5000
pf = n->R*n^3
pfprime = n->3*R*n^2


for i = 1:get_ngauss(mmesh)
    n = 0.0

    if yg[i]<0.2
        nv[i,:] = [0.0;-1.0]
        n = max(n, abs(0.2-yg[i]))
    elseif yg[i]>0.8
        nv[i,:] = [0.0;1.0]
        n = max(abs(0.8-yg[i]), n)
    end

    if xg[i]<0.2
        nv[i,:] = [-1.0;0.0]
        n = max(n, abs(0.2-xg[i]))
    elseif xg[i]>0.8
        nv[i,:] = [1.0;0.0]
        n = max(n, abs(0.8-xg[i]))
    end

    if xg[i]<0.2 && yg[i]<0.2
        nv[i,:] = [xg[i]-0.2 yg[i]-0.2]
        ns = norm(nv[i,:])
        nv[i,:] = nv[i,:]/ns
    end
    if xg[i]<0.2 && yg[i]>0.8
        nv[i,:] = [xg[i]-0.2 yg[i]-0.8]
        ns = norm(nv[i,:])
        nv[i,:] = nv[i,:]/ns
    end
    if xg[i]>0.8 && yg[i]>0.8
        nv[i,:] = [xg[i]-0.8 yg[i]-0.8]
        ns = norm(nv[i,:])
        nv[i,:] = nv[i,:]/ns
    end
    if xg[i]>0.8 && yg[i]<0.2
        nv[i,:] = [xg[i]-0.8 yg[i]-0.2]
        ns = norm(nv[i,:])
        nv[i,:] = nv[i,:]/ns
    end
      

    if (0.2<xg[i]<0.8) && (0.2<yg[i]<0.8)
        continue 
    end
    β_g[i] = pf(n)
    βprime[i] = pfprime(n)
end


for i = 1:mmesh.ndof
    n = 0.0

    if x[i]<0.2
        n = abs(0.2-x[i])
    elseif x[i]>0.8
        n = abs(0.8-x[i])
    end

    if y[i]<0.2
        n = max(n, abs(0.2-y[i]))
    elseif y[i]>0.8
        n = max(abs(0.8-y[i]), n)
    end

    if (0.2<x[i]<0.8) && (0.2<y[i]<0.8)
        continue 
    end

    β[i] = pf(n)
end

nv = constant(nv)
β_g = constant(β_g)
βprime = constant(βprime)
β = constant(β)
β = [β;β]