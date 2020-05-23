export visualize_pressure, visualize_displacement, 
    visualize_stress, visualize_scattered_displacement, 
    visualize_von_mises_stress, visualize_saturation,
    visualize_potential, visualize_scalar_on_gauss_points,
    visualize_scalar_on_fem_points

""" 
    visualize_pressure(U::Array{Float64, 2}, m::Int64, n::Int64, h::Float64)

Visualizes pressure. `U` is the solution vector. 
"""
function visualize_pressure(U::Array{Float64, 2}, m::Int64, n::Int64, h::Float64)
    # fig, ax = subplots()
    close("all")
    x = (1:m)*h
    y = (1:n)*h
    vmin = mean(U) - 2std(U)
    vmax = mean(U) + 2std(U)
    U = reshape(U, size(U,1), m, n)
    U = permutedims(U, [1,3,2])
    ln = pcolormesh(x, y, U[1,:,:], vmin=vmin, vmax=vmax,rasterized=true)
    t = title("snapshot = 000")
    colorbar()
    axis("scaled")
    xlabel("x")
    ylabel("y")
    levels = LinRange(vmin, vmax, 10) |> Array
    c = contour(x, y, U[1,:,:], levels, cmap="jet", vmin=vmin, vmax=vmax)
    gca().invert_yaxis()
    function update(frame)
        gca().clear()
        k = string(frame-1)
        k = repeat("0", 3-length(k))*k 
        title("snapshot = $k")
        pcolormesh(x, y, U[frame,:,:], vmin=vmin, vmax=vmax,rasterized=true)
        contour(x, y, U[frame,:,:], levels, cmap="jet", vmin=vmin, vmax=vmax)
        xlabel("x")
        ylabel("y")
        gca().invert_yaxis()
    end
    anim = animate(update, 1:size(U,1))
end


function visualize_pressure(U::Array{Float64,1}, m::Int64, n::Int64, h::Float64)
    # fig, ax = subplots()
    close("all")
    x = (1:m)*h
    y = (1:n)*h
    vmin = mean(U) - 2std(U)
    vmax = mean(U) + 2std(U)
    U = reshape(U,  m, n)
    U = Array(U')
    ln = pcolormesh(x, y, U, vmin=vmin, vmax=vmax,rasterized=true)
    colorbar()
    axis("scaled")
    xlabel("x")
    ylabel("y")
    levels = LinRange(vmin, vmax, 10) |> Array
    c = contour(x, y, U[:,:], levels, cmap="jet", vmin=vmin, vmax=vmax)
    gca().invert_yaxis()
end


@doc raw""" 
    visualize_stress(K::Array{Float64, 2}, U::Array{Float64, 2}, m::Int64, n::Int64, h::Float64; name::String="")

Visualizes displacement. `U` is the solution vector, `K` is the elasticity matrix ($3\times 3$).
"""
function visualize_stress(K::Array{Float64, 2}, U::Array{Float64, 1}, m::Int64, n::Int64, h::Float64; name::String="")
    close("all")
    NT = size(U,1)
    x1 = LinRange(0.5h,m*h,m)|>collect
    y1 = LinRange(0.5h,n*h,n)|>collect
    S = zeros(n, m)
    s = compute_von_mises_stress_term(K, U, m, n, h)
    for i = 1:m 
        for j = 1:n 
            S[j,i] = sum(s[4*(m*(j-1)+i-1)+1:4*(m*(j-1)+i)])/4.0
        end
    end
    μ = mean(S); σ = std(S)
    vmin = μ - 2σ
    vmax = μ + 2σ

    levels = LinRange(vmin, vmax, 10) |> Array
    pcolormesh(x1,y1,S[:,:], vmin=vmin,vmax=vmax,rasterized=true)
    colorbar()
    contour(x1, y1, S[:,:], levels, cmap="jet")
    axis("scaled")
    xlabel("x")
    ylabel("y")
    gca().invert_yaxis()
end


function visualize_stress(K::Array{Float64, 2}, U::Array{Float64, 1}, m::Int64, n::Int64, h::Float64; name::String="")
    close("all")
    NT = size(U,1)
    x1 = LinRange(0.5h,m*h,m)|>collect
    y1 = LinRange(0.5h,n*h,n)|>collect
    S = zeros(NT, n, m)
    for k = 1:NT
        s = compute_von_mises_stress_term(K, U, m, n, h)
        for i = 1:m 
            for j = 1:n 
                S[k,j,i] = sum(s[4*(m*(j-1)+i-1)+1:4*(m*(j-1)+i)])/4.0
            end
        end
    end
    μ = mean(S); σ = std(S)
    vmin = μ - 2σ
    vmax = μ + 2σ

    levels = LinRange(vmin, vmax, 10) |> Array
    pcolormesh(x1,y1,S[1,:,:], vmin=vmin,vmax=vmax,rasterized=true)
    colorbar()
    contour(x1, y1, S[1,:,:], levels, cmap="jet")
    axis("scaled")
    xlabel("x")
    ylabel("y")
    gca().invert_yaxis()
end


"""
    visualize_stress(Se::Array{Float64, 2}, m::Int64, n::Int64, h::Float64; name::String="")

Visualizes the Von Mises stress. `Se` is the Von Mises at the cell center. 
"""
function visualize_stress(Se::Array{Float64, 2}, m::Int64, n::Int64, h::Float64; name::String="", kwargs...)
    NT = size(Se,2)

    x1 = LinRange(0.5h,m*h,m)|>collect
    y1 = LinRange(0.5h,n*h,n)|>collect
    X1, Y1 = np.meshgrid(x1,y1)

    

    if size(Se, 1)==4*m*n 
        Sep = zeros(m*n, NT)
        for k = 1:NT 
            Sep[:,k] = mean([Se[1:4:end, k], Se[2:4:end, k], Se[3:4:end, k], Se[4:4:end, k]])
        end
        Se = Sep
    end

    S = zeros(size(Se, 2), n, m)
    for k = 1:size(Se, 2)
        S[k,:,:] = reshape(Se[:,k], m, n)'
    end
    μ = mean(S); σ = std(S)
    vmin = μ - 2σ
    vmax = μ + 2σ

    x = (1:m)*h
    y = (1:n)*h
    close("all")
    ln = pcolormesh(x, y, S[1,:,:], vmin= vmin, vmax=vmax)
    
    colorbar()
    # c = contour(φ[1,:,:], 10, cmap="jet", vmin=vmin,vmax=vmax)
    t = title("t = 0")
    axis("scaled")
    xlabel("x")
    ylabel("y")
    gca().invert_yaxis()
    function update(i)
        gca().clear()
        # t.set_text("t = $(round(frame * Δt, digits=3))")
        # ln.set_array(φ[frame,:,:]'[:])
        # c.set_array(φ[frame,:,:]'[:])
        ln = gca().pcolormesh(x, y, S[i,:,:], vmin= vmin, vmax=vmax)
        c = gca().contour(x, y, S[i,:,:], 10, cmap="jet", vmin=vmin,vmax=vmax)
        xlabel("x")
        ylabel("y")

        k = string(i-1)
        k = repeat("0", 3-length(k))*k 
        title("snapshot = $k")
        gca().invert_yaxis()
    end
    anim = animate(update, 1:size(S,1))
end

"""
    visualize_von_mises_stress(Se::Array{Float64, 2}, m::Int64, n::Int64, h::Float64; name::String="")

Visualizes the Von Mises stress. 
"""
function visualize_von_mises_stress(Se::Array{Float64, 3}, m::Int64, n::Int64, h::Float64; name::String="", kwargs...)
    S = zeros(size(Se, 1), m*n)
    for i = 1:size(Se, 1)
        S[i,:] = compute_von_mises_stress_term(Se[i,:,:], m, n, h)
    end
    visualize_stress(S'|>Array, m, n, h; name = name, kwargs...)
end

function visualize_von_mises_stress(Se::Array{Float64, 2}, m::Int64, n::Int64, h::Float64)
    S = compute_von_mises_stress_term(Se, m, n, h)

    x1 = LinRange(0.5h,m*h,m)|>collect
    y1 = LinRange(0.5h,n*h,n)|>collect
    X1, Y1 = np.meshgrid(x1,y1)

    μ = mean(S); σ = std(S)
    vmin = μ - 2σ
    vmax = μ + 2σ


    S = reshape(S, m, n)'
    x = (1:m)*h
    y = (1:n)*h
    close("all")
    ln = pcolormesh(x, y, S, vmin= vmin, vmax=vmax, rasterized=true)
    
    colorbar()
    c = contour(x, y, S, cmap="jet")
    axis("scaled")
    xlabel("x")
    ylabel("y")
    gca().invert_yaxis()

end


function visualize_saturation(s2::Array{Float64,3}, m::Int64, n::Int64, h::Float64)
    # fig, ax = subplots()
    close("all")
    x = (1:m)*h
    y = (1:n)*h
    ln = pcolormesh(x, y, s2[1,:,:], vmin=0.0, vmax=1.0)
    t = title("snapshot = 000")
    colorbar()
    axis("scaled")
    xlabel("x")
    ylabel("y")
    function update(frame)
        k = string(frame-1)
        k = repeat("0", 3-length(k))*k 
        t.set_text("snapshot = $k")
        ln.set_array(s2[frame,1:end-1,1:end-1]'[:])
    end
    anim = animate(update, 1:size(s2,1))
end


function visualize_saturation(s2::Array{Float64,2}, m::Int64, n::Int64, h::Float64)
    # fig, ax = subplots()
    close("all")
    x = (1:m)*h
    y = (1:n)*h
    ln = pcolormesh(x, y, s2, vmin=0.0, vmax=1.0)
    colorbar()
    axis("scaled")
    xlabel("x")
    ylabel("y")
end



@doc raw"""
    visualize_potential(φ::Array{Float64, 3}, m::Int64, n::Int64, h::Float64)

Generates scattered potential animation for the potential $\phi\in \mathbb{R}^{(NT+1)\times n \times m}$.
"""
function visualize_potential(φ::Array{Float64, 3}, m::Int64, n::Int64, h::Float64)
    m_ = mean(φ)
    s = std(φ)
    close("all")
    vmin, vmax = m_ - 2s, m_ + 2s
    x = (1:m)*h
    y = (1:n)*h
    ln = pcolormesh(x, y, φ[1,:,:], vmin= vmin, vmax=vmax)
    colorbar()
    # c = contour(φ[1,:,:], 10, cmap="jet", vmin=vmin,vmax=vmax)
    t = title("t = 0")
    axis("scaled")
    xlabel("x")
    ylabel("y")
    function update(i)
        gca().clear()
        # t.set_text("t = $(round(frame * Δt, digits=3))")
        # ln.set_array(φ[frame,:,:]'[:])
        # c.set_array(φ[frame,:,:]'[:])
        ln = gca().pcolormesh(x, y, φ[i,:,:], vmin= vmin, vmax=vmax)
        c = gca().contour(x, y, φ[i,:,:], 10, cmap="jet", vmin=vmin,vmax=vmax)
        xlabel("x")
        ylabel("y")
        k = string(i-1)
        k = repeat("0", 3-length(k))*k 
        title("snapshot = $k")
    end
    anim = animate(update, 1:size(φ,1))
end

@doc raw"""
    visualize_displacement(u::Array{Float64, 2}, m::Int64, n::Int64, h::Float64)

Generates scattered plot animation for displacement $u\in \mathbb{R}^{(NT+1)\times 2(m+1)(n+1)}$.
"""
function visualize_displacement(u::Array{Float64, 2}, m::Int64, n::Int64, h::Float64)
    X = zeros(m+1, n+1)
    Y = zeros(m+1, n+1)
    for i = 1:m+1
        for j = 1:n+1
            X[i, j] = (i-1)*h 
            Y[i, j] = (j-1)*h 
        end
    end
    function disp(u)
        U1 = reshape(u[1:(m+1)*(n+1)], m+1, n+1)
        U2 = reshape(u[(m+1)*(n+1)+1:2(m+1)*(n+1)], m+1, n+1)
        U1 = X + U1 
        U2 = Y + U2
        U1, U2 
    end
    close("all")
    U1, U2 = disp(u[1,:])
    s = scatter(U1[:], U2[:], s=5)
    xmin = minimum(u[:,1:(m+1)*(n+1)])
    xmax = maximum(u[:,1:(m+1)*(n+1)]) + m*h
    ymin = minimum(u[:,(m+1)*(n+1)+1:2(m+1)*(n+1)])
    ymax = maximum(u[:,(m+1)*(n+1)+1:2(m+1)*(n+1)]) + n*h
    t = title("snapshot = 000")
    xlabel("x")
    ylabel("y")
    axis("equal")
    xlim(xmin.-h, xmax.+h)
    ylim(ymin.-h, ymax.+h)
    gca().invert_yaxis()
    function update(i)
        U1, U2 = disp(u[i,:])
        s.set_offsets([U1[:] U2[:]])

        k = string(i-1)
        k = repeat("0", 3-length(k))*k 
        t.set_text("snapshot = $k")
        xlim(xmin.-h, xmax.+h)
        ylim(ymin.-h, ymax.+h)
        gca().invert_yaxis()
    end
    animate(update, 1:size(u,1))
end

function visualize_displacement(u::Array{Float64, 1}, m::Int64, n::Int64, h::Float64)
    X = zeros(m+1, n+1)
    Y = zeros(m+1, n+1)
    for i = 1:m+1
        for j = 1:n+1
            X[i, j] = (i-1)*h 
            Y[i, j] = (j-1)*h 
        end
    end
    function disp(u)
        U1 = reshape(u[1:(m+1)*(n+1)], m+1, n+1)
        U2 = reshape(u[(m+1)*(n+1)+1:2(m+1)*(n+1)], m+1, n+1)
        U1 = X + U1 
        U2 = Y + U2
        U1, U2 
    end
    close("all")
    U1, U2 = disp(u)
    s = scatter(U1[:], U2[:], s=5)
    xmin = minimum(u[1:(m+1)*(n+1)])
    xmax = maximum(u[1:(m+1)*(n+1)]) + m*h
    ymin = minimum(u[(m+1)*(n+1)+1:2(m+1)*(n+1)])
    ymax = maximum(u[(m+1)*(n+1)+1:2(m+1)*(n+1)]) + n*h
    xlim(xmin.-h, xmax.+h)
    ylim(ymin.-h, ymax.+h)
    gca().invert_yaxis()
    xlabel("x")
    ylabel("y")
    axis("equal")
end

@doc raw"""
    visualize_scalar_on_gauss_points(u::Array{Float64,1}, m::Int64, n::Int64, h::Float64, args...;kwargs...)

Visualizes the scalar `u` using pcolormesh. Here `u` is a length $4mn$ vector and the values are defined on the Gauss points
"""
function visualize_scalar_on_gauss_points(u::Array{Float64,1}, m::Int64, n::Int64, h::Float64, args...;kwargs...)
    # close("all")
    z = zeros(2m, 2n)
    x = zeros(2m)
    y = zeros(2n)

    for i = 1:m 
        x[2*(i-1)+1] = pts[1] * h + (i-1)*h 
        x[2*(i-1)+2] = pts[2] * h + (i-1)*h 
    end

    for j = 1:n
        y[2*(j-1)+1] = pts[1] * h + (j-1)*h 
        y[2*(j-1)+2] = pts[2] * h + (j-1)*h 
    end

    for j = 1:n 
    end

    for i = 1:m 
        for j = 1:n 
            idx = (j-1)*m+i
            z[2(i-1)+1, 2(j-1)+1] = u[4*(idx-1)+1]
            z[2(i-1)+2, 2(j-1)+1] = u[4*(idx-1)+2]
            z[2(i-1)+1, 2(j-1)+2] = u[4*(idx-1)+3]
            z[2(i-1)+2, 2(j-1)+2] = u[4*(idx-1)+4]
        end
    end

    vmin = mean(z) - 2std(z)
    vmax = mean(z) + 2std(z)
    pcolormesh(x, y, z', vmin=vmin, vmax=vmax,rasterized=true, args...; kwargs...)
    colorbar()
    axis("scaled")
    xlabel("x")
    ylabel("y")
    levels = LinRange(vmin, vmax, 10) |> Array
    c = contour(x, y, z', levels, cmap="jet", vmin=vmin, vmax=vmax)
    gca().invert_yaxis()
    return x, y, z 
end

@doc raw"""
    visualize_scalar_on_fem_points(u::Array{Float64,1}, m::Int64, n::Int64, h::Float64, args...;kwargs...)

Visualizes the scalar `u` using pcolormesh. Here `u` is a length $(m+1)(n+1)$ vector and the values are defined on the FEM points
"""
function visualize_scalar_on_fem_points(u::Array{Float64,1}, m::Int64, n::Int64, h::Float64, args...;kwargs...)
    x = Array((0:m)*h)
    y = Array((0:n)*h)
    z = reshape(u, m+1, n+1)

    vmin = mean(z) - 2std(z)
    vmax = mean(z) + 2std(z)
    pcolormesh(x, y, z', vmin=vmin, vmax=vmax,rasterized=true, args...; kwargs...)
    colorbar()
    axis("scaled")
    xlabel("x")
    ylabel("y")
    levels = LinRange(vmin, vmax, 10) |> Array
    c = contour(x, y, z', levels, cmap="jet", vmin=vmin, vmax=vmax)
    gca().invert_yaxis()
    return x, y, z 
end