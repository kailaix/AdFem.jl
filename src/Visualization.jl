export visualize_pressure, visualize_velocity, visualize_displacement, visualize_stress
function visualize_pressure(U::Array{Float64, 2}, m::Int64, n::Int64, h::Float64; name::String="")
    
    vmin = minimum(U[2(m+1)*(n+1)+1:end,:])
    vmax = maximum(U[2(m+1)*(n+1)+1:end,:])
    NT = size(U,2)
    for k in Int64.(round.(LinRange(1, NT, 20)))
        close("all")
        pcolormesh((1:n)*h .-0.5h,(1:m)*h .-0.5h, reshape(U[2(m+1)*(n+1)+1:end, k], m, n)', vmin=vmin,vmax=vmax)
        colorbar()
        k_ = string(k)
        k_ = repeat("0", 3-length(k_))*k_
        title("snapshot = $k_")
        contour((1:n)*h .-0.5h,(1:m)*h .-0.5h, reshape(U[2(m+1)*(n+1)+1:end, k], m, n)', 10, cmap="jet")
        savefig("__p$k_.png")
    end
    run(`convert -delay 10 -loop 0 __p*.png disp_p$name.gif`)
end

function visualize_velocity(U::Array{Float64, 2}, m::Int64, n::Int64, h::Float64; fmt::String="png")
    umin = minimum(U[1:(m+1)*(n+1), :])
    umax = maximum(U[1:(m+1)*(n+1), :])
    vmin = minimum(U[(m+1)*(n+1)+1:2(m+1)*(n+1), :])
    vmax = maximum(U[(m+1)*(n+1)+1:2(m+1)*(n+1), :])
    V1 = []; V2 = []
    NT = size(U,2)
    for k in Int64.(round.(LinRange(1, NT, 9)))
        push!(V1, reshape(U[1:(m+1)*(n+1), k], m+1, n+1)')
        push!(V2, reshape(U[(m+1)*(n+1)+1:2(m+1)*(n+1), k], m+1, n+1)')
    end

    close("all")
    figure(1)
    rc("axes", titlesize=30)
    rc("axes", labelsize=30)
    rc("xtick", labelsize=28)
    rc("ytick", labelsize=28)
    rc("legend", fontsize=30)
    fig1,axs = subplots(3,3, figsize=[20,16], sharex=true, sharey=true)
    ims = Array{Any}(undef, 9)
    for iPrj = 1:3
        for jPrj = 1:3
            ims[(iPrj-1)*3+jPrj] = axs[iPrj,jPrj].imshow(V1[(iPrj-1)*3+jPrj], extent=[0,m*h,n*h,0], vmin=umin, vmax=umax);
            if jPrj == 1 || jPrj == 1
                axs[iPrj,jPrj].set_ylabel("Depth (m)")
            end
            if iPrj == 3 || iPrj == 3
                axs[iPrj,jPrj].set_xlabel("Distance (m)")
            end
        end
    end
    fig1.subplots_adjust(wspace=0.02, hspace=0.18)
    cbar_ax = fig1.add_axes([0.91, 0.08, 0.01, 0.82])
    cb1 = fig1.colorbar(ims[1], cax=cbar_ax)
    cb1.set_label("u") 
    savefig("u.$fmt")
    close("all")
    
    figure(2)
    rc("axes", titlesize=30)
    rc("axes", labelsize=30)
    rc("xtick", labelsize=28)
    rc("ytick", labelsize=28)
    rc("legend", fontsize=30)
    fig1,axs = subplots(3,3, figsize=[20,16], sharex=true, sharey=true)
    ims = Array{Any}(undef, 9)
    for iPrj = 1:3
        for jPrj = 1:3
            ims[(iPrj-1)*3+jPrj] = axs[iPrj,jPrj].imshow(V2[(iPrj-1)*3+jPrj], extent=[0,m*h,n*h,0], vmin=vmin, vmax=vmax);
            if jPrj == 1 || jPrj == 1
                axs[iPrj,jPrj].set_ylabel("Depth (m)")
            end
            if iPrj == 3 || iPrj == 3
                axs[iPrj,jPrj].set_xlabel("Distance (m)")
            end
        end
    end
    fig1.subplots_adjust(wspace=0.02, hspace=0.18)
    cbar_ax = fig1.add_axes([0.91, 0.08, 0.01, 0.82])
    cb1 = fig1.colorbar(ims[1], cax=cbar_ax)
    cb1.set_label("v")
    savefig("v.$fmt")
    close("all")
end

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
        k_ = string(k)
        k_ = repeat("0", 3-length(k_))*k_
        title("snapshot = $k_")
        savefig("__v$k_.png")
    end
    run(`convert -delay 10 -loop 0 __v*.png disp_v$name.gif`)

    close("all")
end

function visualize_stress(K::Array{Float64, 2}, U::Array{Float64, 2}, m::Int64, n::Int64, 
    h::Float64; fmt::String="png", scale::Float64=1e-6)
    x = LinRange(0, m*h, 50)|>collect
    y = LinRange(-n*h, 0, 50)|>collect
    X, Y = np.meshgrid(x, y)
    Xv = Float64[]; Yv = Float64[]
    for i = 1:m
        for j = 1:n 
            for p = 1:2
                for q = 1:2
                    η = pts[p]; ξ = pts[q]
                    push!(Xv, (i-1)*h+ξ*h)
                    push!(Yv, -(j-1)*h+η*h)
                end
            end
        end
    end
    T = []
    NT = size(U,2)
    for k in Int64.(round.(LinRange(1, NT, 9)))
        push!(T, compute_principal_stress_term(K, U[:, k], m, n, h))
    end
    vmax = maximum([maximum(t) for t in T])
    vmin = minimum([minimum(t) for t in T])
    function helper(ax, T)
        Z = interpolate.griddata((Xv, Yv), T, (X, Y))
        return ax.pcolormesh(X, Y, Z)
    end

    close("all")
    rc("axes", titlesize=30)
    rc("axes", labelsize=30)
    rc("xtick", labelsize=28)
    rc("ytick", labelsize=28)
    rc("legend", fontsize=30)
    fig1,axs = subplots(3,3, figsize=[20,16], sharex=true, sharey=true)
    ims = Array{Any}(undef, 9)
    for iPrj = 1:3
        for jPrj = 1:3
            ims[(iPrj-1)*3+jPrj] = helper(axs[iPrj,jPrj], T[(iPrj-1)*3+jPrj])
            if jPrj == 1 || jPrj == 1
                axs[iPrj,jPrj].set_ylabel("Depth (m)")
            end
            if iPrj == 3 || iPrj == 3
                axs[iPrj,jPrj].set_xlabel("Distance (m)")
            end
        end
    end
    fig1.subplots_adjust(wspace=0.02, hspace=0.18)
    cbar_ax = fig1.add_axes([0.91, 0.08, 0.01, 0.82])
    cb1 = fig1.colorbar(ims[1], cax=cbar_ax)
    cb1.set_label("Displacement") 
    savefig("stress.$fmt")
    close("all")
end