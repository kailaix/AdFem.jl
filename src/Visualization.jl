export visualize_pressure, visualize_velocity
function visualize_pressure(U::Array{Float64, 2}, m::Int64, n::Int64, h::Float64; fmt::String="png")
    close("all")
    vmin = minimum(U[2(m+1)*(n+1)+1:end,:])
    vmax = maximum(U[2(m+1)*(n+1)+1:end,:])
    V = []
    NT = size(U,2)
    for k in Int64.(round.(LinRange(1, NT, 9)))
        push!(V, reshape(U[2(m+1)*(n+1)+1:end, k], m, n)')
    end
    rc("axes", titlesize=30)
    rc("axes", labelsize=30)
    rc("xtick", labelsize=28)
    rc("ytick", labelsize=28)
    rc("legend", fontsize=30)
    fig1,axs = subplots(3,3, figsize=[30,15], sharex=true, sharey=true)
    ims = Array{Any}(undef, 9)
    for iPrj = 1:3
        for jPrj = 1:3
            ims[(iPrj-1)*3+jPrj] = axs[iPrj,jPrj].imshow(V[(iPrj-1)*3+jPrj])#, extent=[0,m*h,n*h,0], vmin=vmin, vmax=vmax);
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
    cb1.set_label("Pressure")    
    savefig("p.$fmt")
end

function visualize_velocity(U::Array{Float64, 2}, m::Int64, n::Int64, h::Float64; fmt::String="png")
    umin = minimum(U[1:(m+1)*(n+1), :])
    umin = maximum(U[1:(m+1)*(n+1), :])
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
    fig1,axs = subplots(3,3, figsize=[30,15], sharex=true, sharey=true)
    ims = Array{Any}(undef, 9)
    for iPrj = 1:3
        for jPrj = 1:3
            ims[(iPrj-1)*3+jPrj] = axs[iPrj,jPrj].imshow(V1[(iPrj-1)*3+jPrj])#, extent=[0,m*h,n*h,0], vmin=vmin, vmax=vmax);
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
    
    figure(2)
    rc("axes", titlesize=30)
    rc("axes", labelsize=30)
    rc("xtick", labelsize=28)
    rc("ytick", labelsize=28)
    rc("legend", fontsize=30)
    fig1,axs = subplots(3,3, figsize=[30,15], sharex=true, sharey=true)
    ims = Array{Any}(undef, 9)
    for iPrj = 1:3
        for jPrj = 1:3
            ims[(iPrj-1)*3+jPrj] = axs[iPrj,jPrj].imshow(V1[(iPrj-1)*3+jPrj])#, extent=[0,m*h,n*h,0], vmin=vmin, vmax=vmax);
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
end