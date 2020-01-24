export compute_planestressplasticity_stress_and_stiffness_matrix
@doc raw"""
    planestressplasticity_get_stress_and_sensitivity(
        ε::Array{Float64}, ε0::Array{Float64}, σ0::Array{Float64}, 
        α::Array{Float64}, K::Float64, σY::Float64, H::Array{Float64}
    )

Returns 
- The stress $\sigma$
- The sensitivity $\frac{d\sigma}{d\varepsilon}$
- New internal variable $\alpha_1$ 

Yield function is defined as 
$$y = f(\sigma) - (\sigma_Y+K\alpha)$$
where $\alpha$ is the internal variable per Gauss points. `f` is the J2 function. 

$\varepsilon$ is the proposed strain. $\varepsilon_0$ and $\sigma_0$ are strain and stress from the last step. 
"""
function compute_planestressplasticity_stress_and_stiffness_matrix(
    u::Array{Float64}, ε0::Array{Float64}, σ0::Array{Float64}, 
    α::Array{Float64}, K::Float64, σY::Float64, H::Array{Float64}, m::Int64, n::Int64, h::Float64
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


function _f(σ, α, σY, K)
    return sqrt(σ[1]^2-σ[1]*σ[2]+σ[2]^2+3*σ[3]^2)-σY-K*α
end

function _fσ(σ)
    σ1, σ2, σ3 = σ[1], σ[2], σ[3]
    J2 = sqrt(σ1^2-σ1*σ2+σ2^2+3*σ3^2)
    z = [(σ1 - σ2/2)/J2;
        (-σ1/2 + σ2)/J2;
        3*σ3/J2]
end

function _fσσ(σ)
    σ1, σ2, σ3 = σ[1], σ[2], σ[3]
    J2 = sqrt(σ1^2-σ1*σ2+σ2^2+3*σ3^2)
    [     (-σ1 + σ2/2)*(σ1 - σ2/2)/J2^3 + 1/J2 (σ1/2 - σ2)*(σ1 - σ2/2)/J2^3 - 1/(2*J2)                                   -3*σ3*(σ1 - σ2/2)/J2^3;
    (-σ1 + σ2/2)*(-σ1/2 + σ2)/J2^3 - 1/(2*J2)    (-σ1/2 + σ2)*(σ1/2 - σ2)/J2^3 + 1/J2                                  -3*σ3*(-σ1/2 + σ2)/J2^3;
    3*σ3*(-σ1 + σ2/2)/J2^3                                                        3*σ3*(σ1/2 - σ2)/J2^3 -9*σ3^2/J2^3 + 3/J2]
end

function _planestressplasticity_get_stress_and_sensitivity(
    ε::Array{Float64}, ε0::Array{Float64}, σ0::Array{Float64}, 
    α0::Float64, K::Float64, σY::Float64, H::Array{Float64}
)
    local dΔσdΔε
    σ = σ0 + H*(ε-ε0) 
    Δγ = 0.0
    r2 = _f(σ, α0, σY, K)
    if r2<0
        # @info "elasticity"
        dΔσdΔε = H
    else
        # @info "plasticity"
        function compute(σ, Δγ)
            α = α0 + Δγ
            r1 = σ + Δγ * H* _fσ(σ) - σ0 - H*ε + H*ε0 
            r2 = _f(σ, α, σY, K)
            J = [UniformScaling(1.0)+Δγ*H*_fσσ(σ) H*_fσ(σ)
                reshape(_fσ(σ),1,3) -K]
            return [r1;r2], J
        end

        function compute_sensitivity(σ, Δγ)
            α = α0 + Δγ
            J = [UniformScaling(1.0)+Δγ*H*_fσσ(σ) H*_fσ(σ)
                reshape(_fσ(σ),1,3) -K]
            δ = J\[H;zeros(1,3)]
            return δ[1:3,:]
        end
        res0, _ = compute(σ, Δγ)
        for i = 1:100
            res, J = compute(σ, Δγ)
            δ = -J\res
            σ += δ[1:3]; Δγ += δ[4]
            if norm(res)/norm(res0) < 1e-6 || norm(res) < 1e-6
                break
            end
        end

        dΔσdΔε = compute_sensitivity(σ, Δγ)
    end
    α = α0+Δγ
    return σ[:], dΔσdΔε, α
end



