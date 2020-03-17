export visualize_pressure, visualize_displacement, 
    visualize_stress, visualize_scattered_displacement, 
    visualize_von_mises_stress, visualize_saturation,
    visualize_potential, visualize_displacement

""" 
    visualize_pressure(U::Array{Float64, 2}, m::Int64, n::Int64, h::Float64; name::String="")

Visualizes pressure. `U` is the solution vector. 
"""
function visualize_pressure(U::Array{Float64, 2}, m::Int64, n::Int64, h::Float64; name::String="")
    
    vmin = minimum(U[2(m+1)*(n+1)+1:end,:])
    vmax = maximum(U[2(m+1)*(n+1)+1:end,:])
    NT = size(U,2)
    x1 = LinRange(0.0,m*h,m)|>collect
    y1 = LinRange(0.0,n*h,n)|>collect
    for k in Int64.(round.(LinRange(1, NT, 20)))
        close("all")
        pcolormesh(x1, y1, reshape(U[2(m+1)*(n+1)+1:end, k], m, n)', vmin=vmin,vmax=vmax)
        # scatter( (1:m)*h .-0.5*h, div(n,3) * ones(n) * h .-0.5h, marker="^", color="r")
        colorbar()
        k_ = string(k)
        k_ = repeat("0", 3-length(k_))*k_
        title("snapshot = $k_")
        contour(x1, y1, reshape(U[2(m+1)*(n+1)+1:end, k], m, n)', 10, cmap="jet")
        axis("equal")
        gca().invert_yaxis()
        savefig("__p$k_.png")
    end
    run(`convert -delay 10 -loop 0 __p*.png disp_p$name.gif`)
    rfiles = [x for x in readdir(".") if occursin("__p", x)]
    rm.(rfiles)
end

""" 
    visualize_displacement(U::Array{Float64, 2}, m::Int64, n::Int64, h::Float64; name::String="")

Visualizes displacement. `U` is the solution vector. 
"""
function visualize_displacement(U::Array{Float64, 2}, m::Int64, n::Int64, 
        h::Float64; name::String = "")
    vmin = minimum(U[1:(m+1)*(n+1),:])
    vmax = maximum(U[1:(m+1)*(n+1),:])
    NT = size(U,2)
    x1 = LinRange(0.0,m*h,m+1)|>collect
    y1 = LinRange(0.0,n*h,n+1)|>collect
    for k in Int64.(round.(LinRange(1, NT, 20)))
        close("all")
        pcolormesh(x1, y1, reshape(U[1:(m+1)*(n+1), k], m+1, n+1)', vmin=vmin,vmax=vmax)
        colorbar()
        contour(x1, y1, reshape(U[1:(m+1)*(n+1), k], m+1, n+1)', 10, cmap="jet", vmin=vmin,vmax=vmax)
        # scatter((1:m+1)*h .-h, div(n,3) * ones(n+1) * h, marker="^", color="r")
        k_ = string(k)
        k_ = repeat("0", 3-length(k_))*k_
        title("snapshot = $k_")
        axis("equal")
        gca().invert_yaxis()
        savefig("__u$k_.png")
    end
    run(`convert -delay 10 -loop 0 __u*.png disp_u$name.gif`)
    rfiles = [x for x in readdir(".") if occursin("__u", x)]
    rm.(rfiles)


    vmin = minimum(U[(m+1)*(n+1)+1:2*(m+1)*(n+1),:])
    vmax = maximum(U[(m+1)*(n+1)+1:2*(m+1)*(n+1),:])
    for k in Int64.(round.(LinRange(1, NT, 20)))
        close("all")
        pcolormesh(x1, y1, reshape(U[(m+1)*(n+1)+1:2*(m+1)*(n+1), k], m+1, n+1)', vmin=vmin,vmax=vmax)
        colorbar()
        contour(x1, y1, reshape(U[(m+1)*(n+1)+1:2*(m+1)*(n+1), k], m+1, n+1)', 10, cmap="jet", vmin=vmin,vmax=vmax)
        # scatter((1:m+1)*h .-h, div(n,3) * ones(n+1) * h, marker="^", color="r")
        k_ = string(k)
        k_ = repeat("0", 3-length(k_))*k_
        title("snapshot = $k_")
        axis("equal")
        gca().invert_yaxis()
        savefig("__v$k_.png")
    end
    run(`convert -delay 10 -loop 0 __v*.png disp_v$name.gif`)

    close("all")
    rfiles = [x for x in readdir(".") if occursin("__v", x)]
    rm.(rfiles)
end

@doc raw""" 
    visualize_stress(K::Array{Float64, 2}, U::Array{Float64, 2}, m::Int64, n::Int64, h::Float64; name::String="")

Visualizes displacement. `U` is the solution vector, `K` is the elasticity matrix ($3\times 3$).
"""
function visualize_stress(K::Array{Float64, 2}, U::Array{Float64, 2}, m::Int64, n::Int64, h::Float64; name::String="")
    close("all")
    NT = size(U,2)
    x1 = LinRange(0.5h,m*h,m)|>collect
    y1 = LinRange(0.5h,n*h,n)|>collect
    S = zeros(20, n, m)
    for (ix,k) in enumerate(Int64.(round.(LinRange(1, NT, 20))))
        s = compute_von_mises_stress_term(K, U[:,k], m, n, h)
        for i = 1:m 
            for j = 1:n 
                S[ix,j,i] = sum(s[4*(m*(j-1)+i-1)+1:4*(m*(j-1)+i)])/4.0
            end
        end
    end
    μ = mean(S); σ = std(S)
    vmin = μ - 2σ
    vmax = μ + 2σ


    Z = S[1,:,:]
    pcolormesh(x1,y1,Z, vmin=vmin,vmax=vmax)
    colorbar()
    title("snapshot = 1")
    contour(x1, y1, Z, 10, cmap="jet")
    axis("scaled")
    gca().invert_yaxis()
    
    function update(ix)
        gca().clear()
        Z = S[ix,:,:]
        pcolormesh(x1,y1,Z, vmin=vmin,vmax=vmax)
        k_ = string(ix)
        k_ = repeat("0", 3-length(k_))*k_
        title("snapshot = $k_")
        contour(x1, y1, Z, 10, cmap="jet")
        axis("scaled")
        gca().invert_yaxis()
    end

    p = animate(update, 1:20)
    saveanim(p, "disp_s$name.gif")
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

function visualize_scattered_displacement(U::Array{Float64, 2}, m::Int64, n::Int64, 
            h::Float64; name::String = "", xlim_=nothing, ylim_=nothing)

    NT = size(U, 2)
    x = []
    y = []
    for j= 1:n+1
        for i = 1:m+1
        push!(x, (i-1)*h)
        push!(y, (j-1)*h)
        end
    end
            
    for (i,k) in enumerate(Int64.(round.(LinRange(1, NT, 20))))
        close("all")
        scatter(x+U[1:(m+1)*(n+1), k], y+U[(m+1)*(n+1)+1:2(m+1)*(n+1), k])
        xlabel("x")
        ylabel("y")
        k = string(i)
        k = repeat("0", 3-length(k))*k 
        title("Iteration = $i")
        # if !isnothing(xlim_)
             
        # end
        axis("equal")
        if !isnothing(xlim_)
            xlim(xlim_...)
        end
        if !isnothing(ylim_)
            ylim(ylim_...)
        end
        
        gca().invert_yaxis()
        k_ = string(i)
        k_ = repeat("0", 3-length(k_))*k_
        savefig("__Scattered$k_.png")
    end
    run(`convert -delay 20 -loop 0 __Scattered*.png disp_scattered_u$name.gif`)
    rfiles = [x for x in readdir(".") if occursin("__Scattered", x)]
    rm.(rfiles)
end


function visualize_saturation(s2, m, n, h)
    # fig, ax = subplots()
    close("all")
    x = (1:m)*h
    y = (1:n)*h
    ln = pcolormesh(x, y, s2[1,:,:], vmin=0.0, vmax=1.0)
    t = title("t = 0")
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
        U2 = reshape(u[(m+1)*(n+1)+1:end], m+1, n+1)
        U1 = X + U1 
        U2 = Y + U2
        U1, U2 
    end
    close("all")
    U1, U2 = disp(u[1,:])
    s = scatter(U1[:], U2[:], s=1)
    xlim(-h, h+(m+1)*h)
    ylim(-h, h+(n+1)*h)
    gca().invert_yaxis()
    t = title("t = 0")
    xlabel("x")
    ylabel("y")
    axis("equal")
    function update(i)
        U1, U2 = disp(u[i,:])
        s.set_offsets([U1[:] U2[:]])

        k = string(i-1)
        k = repeat("0", 3-length(k))*k 
        t.set_text("snapshot = $k")
    end
    animate(update, 1:size(u,1))
end