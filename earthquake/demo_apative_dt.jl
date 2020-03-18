include("utils.jl")

m = 100
n = 50

right = bcnode("right", m, n, h)
left = bcnode("left", m, n, h)
ind = [right;left]
ρ = 3 
μ = 30
a = ones(n+1) * 0.01
b = ones(n+1) * 0.02
# b[1:n÷2] .= 0.03
# b[n÷2:end] .= 0.1
Dc = 0.03
v0 = 1e-3
f0 = 0.6
Ψ0 = ones(n+1) * 1.01 * f0
NT = 50
σn = ones(n+1) * 50
vs = 3.4e3
η = μ/(2*vs)
# η = 1e-13
h = μ * Dc * π / (σn * (b-a))
Δt = h / vs
maxΔt = 1e3
tol = 1e-5

# right = bcnode("right", m, n, h)
# left = bcnode("left", m, n, h)
# ind = [right;left]
# ρ = 1. 
# μ = 1.
# η = 1.
# a = constant(ones(n+1) )
# b = constant(2*ones(n+1) )
# # b[1:n÷2] .= 0.03
# # b[n÷2:end] .= 0.1
# Dc = 1.
# v0 = 1.e-3
# f0 = 1.
# Ψ0 = ones(n+1) * 1. * f0
# Δt = 10.
# NT = 100
# σn = 10. *ones(n+1) 


K_ = compute_fem_stiffness_matrix1(diagm(0=>ones(2)), m, n, h)
K_[ind, :] .= 0.0
K_[ind, ind] = spdiagm(0=>ones(length(ind)))
K_ = dropzeros(K_)
K = constant(K_)

u_bd = zeros(NT+1, n+1)
for i = 1:NT+1
    u_bd[i, :] .= (i-1)*Δt*v0
end
u_bd = constant(u_bd)

τ_id = Int64[]
for i = 1:n 
    elem = (i-1)*m + 1
    push!(τ_id, elem*4 + 1)
end
push!(τ_id, τ_id[end]+2)

U0 = zeros((m+1)*(n+1))
# U1 = v0 * Δt * ones((m+1)*(n+1))

# U1 = zeros(m+1, n+1)
U1 = ones(m+1, n+1) .+ v0*Δt
# for i = 1:n+1
#     U1[end,i] = v0 * Δt
# end
# U1 = U1[:]
# U1 = K_\U1


function simulate()
    function one_step(u3, u3_old, Ψ, i)
        while true
            ## --- RK Stage 1 ---
            vS1 = (u3[left] - u3_old[left])/Δt
            f = a * tf.asinh(vS1/2v0*exp(Ψ/a))
            ΔψS1 = -vS1/Dc*(f-f0+(b-a)*log(vS1/v0))
            Ψ_tmp = ΔψS1 * Δt/2 + Ψ
            u3_bd_left = u3[left] + Δt/2 * vS1 /2
            u3_bd_right = u3[right] + Δt/2 * v0 
            u0 = vector([right;left], [u3_bd_left; u3_bd_right], (m+1)*(n+1))
            u3_tmp = K\u0
            ε = eval_strain_on_gauss_pts1(u3_tmp, m, n, h)
            σ = μ * ε
            τ = σ[τ_id,1] # σzx
            u3_bd_tmp = rate_state_friction(a, u3_tmp[left], v0, Ψ_tmp, σn, τ, η, Δt)

            ## --- RK Stage 2 ---
            vS2 = (u3_bd_tmp - u3_tmp[left])/Δt
            f = a * tf.asinh(vS2/2v0*exp(Ψ_tmp/a))
            ΔψS2 = -vS2/Dc*(f-f0+(b-a)*log(vS2/v0)) 
            Ψ_tmp = ( -ΔψS1 + 2*ΔψS2 ) * Δt + Ψ
            u3_bd_left = u3[left] +  Δt * ( -vS1 + 2*vS2 ) /2
            u3_bd_right = u3[right] + Δt * v0 
            u0 = vector([right;left], [u3_bd_left; u3_bd_right], (m+1)*(n+1))
            u3_tmp = K\u0        
            ε = eval_strain_on_gauss_pts1(u3_tmp, m, n, h)
            σ = μ * ε
            τ = σ[τ_id,1] # σzx
            u3_bd_tmp = rate_state_friction(a, u3_tmp[left], v0, Ψ_tmp, σn, τ, η, Δt)

            # --- RK Stage 3 ---
            vS3 = (u3_bd_tmp - u3_tmp[left])/Δt
            f = a * tf.asinh(vS3/2v0*exp(Ψ_tmp/a))
            ΔψS3 = -vS3/Dc*(f-f0+(b-a)*log(vS3/v0))

            uRK2 = u3 + Δt/2 * (vS1 + vS3)
            uRK3 = u3 + Δt/6 * (vS1 + 4*vS2 + vS3)
            ψRK2 = ψ + Δt/2 * (ΔψS1 + ΔψS2) 
            ψRK3 = ψ + Δt/6 * (ΔψS1 + 4*ΔψS2 + ΔψS3)

            error = max(norm(uRK2 - uRK3, Inf), norm(ψRK2 - ψRK3, Inf))

            if error < tol
                u3_bd = uRK3/2
                ψ = ψRK3
                u0 = vector([right;left], [u3_bd; u3_bd_right], (m+1)*(n+1))
                u3 = K\u0
                σ = μ * ε
                τ = σ[τ_id,1]
                u3_bd = rate_state_friction(a, u3[left], v0, Ψ, σn, τ, η, Δt)
                u3[left] = u3_bd ##??

                ## Find suggestion for next time step length
                if error != 0
                    dt = min(min(0.9*Δt*(tol/error)^(1/3), maxΔt), 5*Δt);
                end
                break
            end

            ## Find suggestion for next time step length
            Δt = min(0.9*Δt*(tol/error)^(1/3), maxΔt);
        end

        # rate and state friction
        ε = eval_strain_on_gauss_pts1(u3, m, n, h)
        σ = μ * ε
        σzx = σ[τ_id,1]
        v3 = (u3[left] - u3_old[left])/Δt

        f = a * tf.asinh(v3/2v0*exp(Ψ/a))
        Ψ_new = -v3/Dc*(f-f0+(b-a)*log(v3/v0)) * Δt + Ψ
        Ψ_new = tf.cond(tf.equal(i,2), ()->constant(Ψ0), ()->Ψ_new)
        op = tf.print(i, Ψ_new)
        i = bind(i, op)

        u3_bd = rate_state_friction(a, u3[left], v0, Ψ_new, σn, σzx, η, Δt)
    
        # op = tf.print(i, u3_bd, u3[left])
        # i = bind(i, op)

        u0 = vector([right;left], [u_bd[i+1]; u3_bd], (m+1)*(n+1))

        # Ψ_new = Ψ
        # u0 = vector([bcnode("left", m, n, h); bcnode("right", m, n, h)], [constant(zeros(n+1)); u_bd[i+1]], (m+1)*(n+1))
        
        
        u3_new = K\u0
    
        return u3_new, Ψ_new
    end
    
    function body(i, ta_u3, ta_Ψ)
        u3 = read(ta_u3, i)
        u3_old = read(ta_u3, i-1)
        Ψ = read(ta_Ψ, i)
        
        u3_new, Ψ_new = one_step(u3, u3_old, Ψ, i)

        # op = tf.print(u3_new)
        # i = bind(i, op)

        ta_u3 = write(ta_u3, i+1, u3_new)
        ta_Ψ = write(ta_Ψ, i+1, Ψ_new)
        op = tf.print(i)
        i = bind(i, op)
        return i+1, ta_u3, ta_Ψ
    end
    
    function condition(i, ta_u3, ta_Ψ)
        i<=NT
    end

    ta_u3, ta_Ψ = TensorArray(NT+1), TensorArray(NT+1)
    i = constant(2, dtype=Int32)
    ta_u3 = write(ta_u3, 1, constant(U0))
    ta_u3 = write(ta_u3, 2, constant(U1))
    ta_Ψ = write(ta_Ψ, 1, constant(Ψ0))
    ta_Ψ = write(ta_Ψ, 2, constant(Ψ0))

    _, u3, _ = while_loop(condition,body, [i, ta_u3, ta_Ψ])
    u3 = stack(u3)
    return u3
end


u3 = simulate()
sess = Session(); init(sess)
U3 = run(sess, u3);

close("all")
plot(U3[:,left[length(left)÷2]])
nothing