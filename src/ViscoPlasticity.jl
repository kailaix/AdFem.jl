export compute_viscoplasticity_stress_and_stiffness_matrix
@doc raw"""
     
"""
function compute_Maxwell_viscoplasticity_stress_and_stiffness_matrix(
    u::Array{Float64}, ε0::Array{Float64}, σ0::Array{Float64}, 
    K::Float64, k::Float64, η::Float64, Δt::Float64, m::Int64, n::Int64, h::Float64
)
    I = Int64[]; J = Int64[]; V = Float64[]
    function add(ii, jj, kk)
        for  i = 1:length(ii)
            for j = 1:length(jj)
                push!(I,ii[i])
                push!(J,jj[j])
                push!(V,kk[i,j])
            end
        end
    end
    ε = eval_strain_on_gauss_pts(u, m, n, h)
    out_rhs = zeros(2(m+1)*(n+1))
    out_α = zeros(4*m*n)
    out_σ = zeros(4*m*n, 3)

    B = zeros(4, 3, 8)
    for i = 1:2
        for j = 1:2
            ξ = pts[i]; η = pts[j]
            B[(i-1)*2+j,:,:] = [
                -1/h*(1-η) 1/h*(1-η) -1/h*η 1/h*η 0.0 0.0 0.0 0.0
                0.0 0.0 0.0 0.0 -1/h*(1-ξ) -1/h*ξ 1/h*(1-ξ) 1/h*ξ
                -1/h*(1-ξ) -1/h*ξ 1/h*(1-ξ) 1/h*ξ -1/h*(1-η) 1/h*(1-η) -1/h*η 1/h*η
            ]
        end
    end

    for i = 1:m 
        for j = 1:n 
            idx = [(j-1)*(m+1)+i;(j-1)*(m+1)+i+1;j*(m+1)+i;j*(m+1)+i+1]
            idx = [idx; idx .+ (m+1)*(n+1)]
            for p = 1:2
                for q = 1:2
                    k = (p-1)*2 + q
                    gauss_k = 4*((j-1)*m+i-1)+k
                    Bk = B[k, :, :]
                    σ, dΔσΔε, out_α[gauss_k] = _planestressplasticity_get_stress_and_sensitivity(
                        ε[gauss_k,:], ε0[gauss_k,:], σ0[gauss_k,:], α[gauss_k], K, σY, H
                    )
                    out_rhs[idx] += (σ'*Bk)[:]*0.25*h^2 # 1 x 8
                    out_σ[gauss_k,:] = σ
                    add(idx, idx, Bk'*dΔσΔε*Bk*0.25*h^2)
                end
            end
        end
    end
    return out_rhs, sparse(I, J, V, (m+1)*(n+1)*2, (m+1)*(n+1)*2), out_α, out_σ
end

