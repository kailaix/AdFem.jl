include("utils.jl")

m = 1
n = 21

a = constant(0.01*Array(LinRange(1,2,n+1)))
b = constant(0.02*Array(LinRange(1,2,n+1)))
σn = 50*Array(LinRange(1,2,n+1))
τ = 30*Array(LinRange(1,2,n+1))
η = 4.7
v0 = 1e-6
f0 = 0.6
Ψ0 = Array(LinRange(1,2,n+1)) * 1.01 * f0
Dc = 0.01

u_left_bd_S2 = Array(LinRange(1,2,n+1))
v_bd_S2 = rate_state_friction(a, u_left_bd_S2, v0, Ψ0, σn, τ, η, 1.0) # t+ Δt + δΔt


# du is defined on the left side
function StateRate(Ψ, v)
    v = abs(v)
    s = v/(2v0)
    o = exp(-Ψ/a)
    f = a * (Ψ/a + log(s+sqrt(s^2+o^2)))
    # f = a* tf.asinh(v/(2v0)*exp(Ψ/a))
    Δψ = -v/Dc*(f-f0+(b-a)*log(v/v0))
end

dpsi_S3  = StateRate(Ψ0, v_bd_S2) # t + Δt


sess = Session(); 
run(sess, v_bd_S2)

# run(sess, dpsi_S3)