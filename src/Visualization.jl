export visualize_pressure, visualize_displacement, visualize_stress

""" 
    visualize_pressure(U::Array{Float64, 2}, m::Int64, n::Int64, h::Float64; name::String="")

Visualizes pressure. `U` is the solution vector. 
"""
function visualize_pressure(U::Array{Float64, 2}, m::Int64, n::Int64, h::Float64; name::String="")
    
    vmin = minimum(U[2(m+1)*(n+1)+1:end,:])
    vmax = maximum(U[2(m+1)*(n+1)+1:end,:])
    NT = size(U,2)
    for k in Int64.(round.(LinRange(1, NT, 20)))
        close("all")
        pcolormesh((1:n)*h .-0.5h,(1:m)*h .-0.5h, reshape(U[2(m+1)*(n+1)+1:end, k], m, n)', vmin=vmin,vmax=vmax)
        # scatter( (1:m)*h .-0.5*h, div(n,3) * ones(n) * h .-0.5h, marker="^", color="r")
        colorbar()
        k_ = string(k)
        k_ = repeat("0", 3-length(k_))*k_
        title("snapshot = $k_")
        contour((1:n)*h .-0.5h,(1:m)*h .-0.5h, reshape(U[2(m+1)*(n+1)+1:end, k], m, n)', 10, cmap="jet")
        savefig("__p$k_.png")
    end
    run(`convert -delay 10 -loop 0 __p*.png disp_p$name.gif`)
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
    for k in Int64.(round.(LinRange(1, NT, 20)))
        close("all")
        pcolormesh((1:n+1)*h .-h,(1:m+1)*h .-h, reshape(U[1:(m+1)*(n+1), k], m+1, n+1)', vmin=vmin,vmax=vmax)
        colorbar()
        contour((1:n+1)*h .-h,(1:m+1)*h .-h, reshape(U[1:(m+1)*(n+1), k], m+1, n+1)', 10, cmap="jet", vmin=vmin,vmax=vmax)
        # scatter((1:m+1)*h .-h, div(n,3) * ones(n+1) * h, marker="^", color="r")
        k_ = string(k)
        k_ = repeat("0", 3-length(k_))*k_
        title("snapshot = $k_")
        savefig("__u$k_.png")
    end
    run(`convert -delay 10 -loop 0 __u*.png disp_u$name.gif`)


    vmin = minimum(U[(m+1)*(n+1)+1:2*(m+1)*(n+1),:])
    vmax = maximum(U[(m+1)*(n+1)+1:2*(m+1)*(n+1),:])
    for k in Int64.(round.(LinRange(1, NT, 20)))
        close("all")
        pcolormesh((1:n+1)*h .-h,(1:m+1)*h .-h, reshape(U[(m+1)*(n+1)+1:2*(m+1)*(n+1), k], m+1, n+1)', vmin=vmin,vmax=vmax)
        colorbar()
        contour((1:n+1)*h .-h,(1:m+1)*h .-h, reshape(U[(m+1)*(n+1)+1:2*(m+1)*(n+1), k], m+1, n+1)', 10, cmap="jet", vmin=vmin,vmax=vmax)
        # scatter((1:m+1)*h .-h, div(n,3) * ones(n+1) * h, marker="^", color="r")
        k_ = string(k)
        k_ = repeat("0", 3-length(k_))*k_
        title("snapshot = $k_")
        savefig("__v$k_.png")
    end
    run(`convert -delay 10 -loop 0 __v*.png disp_v$name.gif`)

    close("all")
end

@doc raw""" 
    visualize_stress(K::Array{Float64, 2}, U::Array{Float64, 2}, m::Int64, n::Int64, h::Float64; name::String="")

Visualizes displacement. `U` is the solution vector, `K` is the elasticity matrix ($3\times 3$).
"""
function visualize_stress(K::Array{Float64, 2}, U::Array{Float64, 2}, m::Int64, n::Int64, h::Float64; name::String="")
    NT = size(U,2)

    x1 = LinRange(0.5h,n*h,n)|>collect
    y1 = LinRange(0.5h,m*h,m)|>collect
    X1, Y1 = np.meshgrid(x1,y1)

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

    for (ix,k) in enumerate(Int64.(round.(LinRange(1, NT, 20))))
        Z = S[ix,:,:]
        close("all")
        pcolormesh(X1,Y1,Z, vmin=vmin,vmax=vmax)
        colorbar()
        k_ = string(k)
        k_ = repeat("0", 3-length(k_))*k_
        title("snapshot = $k_")
        contour(X1, Y1, Z, 10, cmap="jet")
        savefig("__s$k_.png")
    end
    run(`convert -delay 10 -loop 0 __s*.png disp_s$name.gif`)
end


"""
    visualize_stress(Se::Array{Float64, 2}, m::Int64, n::Int64, h::Float64; name::String="")

Visualizes the Von Mises stress. `Se` is the Von Mises at the cell center. 
"""
function visualize_stress(Se::Array{Float64, 2}, m::Int64, n::Int64, h::Float64; name::String="")
    NT = size(Se,2)

    x1 = LinRange(0.5h,n*h,n)|>collect
    y1 = LinRange(0.5h,m*h,m)|>collect
    X1, Y1 = np.meshgrid(x1,y1)

    S = zeros(20, n, m)

    for (ix,k) in enumerate(Int64.(round.(LinRange(1, NT, 20))))
        S[ix,:,:] = reshape(Se[:,k], m, n)'
    end
    μ = mean(S); σ = std(S)
    vmin = μ - 2σ
    vmax = μ + 2σ

    for (ix,k) in enumerate(Int64.(round.(LinRange(1, NT, 20))))
        Z = S[ix,:,:]
        close("all")
        pcolormesh(X1,Y1,Z, vmin=vmin,vmax=vmax)
        colorbar()
        k_ = string(k)
        k_ = repeat("0", 3-length(k_))*k_
        title("snapshot = $k_")
        contour(X1, Y1, Z, 10, cmap="jet")
        savefig("__s$k_.png")
    end
    run(`convert -delay 10 -loop 0 __s*.png disp_s$name.gif`)
end