export antiplane_viscosity, mantle_viscosity
@doc raw"""
    antiplane_viscosity(ε::Union{PyObject, Array{Float64}}, σ::Union{PyObject, Array{Float64}}, 
    μ::Union{PyObject, Float64}, η::Union{PyObject, Float64}, Δt::Float64)

Calculates the stress at time $t_{n+1}$ given the strain at $t_{n+1}$ and stress at $t_{n}$. The governing equation is 
```math
\dot\sigma + \frac{\mu}{\eta}\sigma = 2\mu \dot\epsilon
```
The discretization form is 
```math
\sigma^{n+1} = \frac{1}{\frac{1}{\Delta t}+\frac{\mu}{\eta}}(2\mu\dot\epsilon^{n+1} + \frac{\sigma^n}{\Delta t})
```
"""
function antiplane_viscosity(dε::Union{PyObject, Array{Float64}}, σ::Union{PyObject, Array{Float64}}, 
        μ::Union{PyObject, Float64, Array{Float64}}, η::Union{PyObject, Float64, Array{Float64}}, Δt::Union{PyObject,Float64})
    dε, σ, μ, η = convert_to_tensor([dε, σ, μ,η], [Float64, Float64, Float64, Float64])
    if length(size(μ))==0
        η/(η/Δt + μ)*(2μ*dε+σ/Δt)
    elseif length(size(μ))==1
        repeat(η/(η/Δt + μ),1,2).*(2repeat(μ, 1, 2).*dε+σ/Δt)
    end
end

@doc raw"""
    mantle_viscosity(u::Union{Array{Float64}, PyObject},
        T::Union{Array{Float64}, PyObject}, m::Int64, n::Int64, h::Float64;
        σ_yield::Union{Float64, PyObject} = 300e6, 
        ω::Union{Float64, PyObject}, 
        η_min::Union{Float64, PyObject} = 1e18, 
        η_max::Union{Float64, PyObject} = 1e23, 
        E::Union{Float64, PyObject} = 9.0, 
        C::Union{Float64, PyObject} = 1000., N::Union{Float64, PyObject} = 2.)


```math
\eta = \eta_{\min} + \min\left( \frac{\sigma_{\text{yield}}}{2\sqrt{\epsilon_{II}}}, \omega\min(\eta_{\max}, \eta) \right)
```
with  
```math
\epsilon_{II} = \frac{1}{2} \epsilon(u)\qquad \eta = C e^{E(0.5-T)} (\epsilon_{II})^{(1-n)/2n}
```

Here $\epsilon_{II}$ is the second invariant of the strain rate tensor, $C > 0$ is a viscosity pre-factor, $E > 0$ is the non-dimensional activation energy,
$n > 0$ is the nonlinear exponent, $η_\min$, $η_\max$ act as minimum and maximum bounds for the effective viscosity, and $σ_{\text{yield}} > 0$ is the yield
stress. Moreover
The viscosity of the mantle is governed by the high-temperature creep of silicates, for which laboratory experiments show that the creep
strength is temperature-, pressure-, compositional- and stress-dependent. 

The output is a length $4mn$ vector. 
"""
function mantle_viscosity(u::Union{Array{Float64}, PyObject},
     T::Union{Array{Float64}, PyObject}, m::Int64, n::Int64, h::Float64;
     σ_yield::Union{Float64, PyObject} = 300e6, 
     ω::Union{Float64, PyObject}, 
     η_min::Union{Float64, PyObject} = 1e18, 
     η_max::Union{Float64, PyObject} = 1e23, 
     E::Union{Float64, PyObject} = 9.0, 
     C::Union{Float64, PyObject} = 1000., N::Union{Float64, PyObject} = 2.)
    u, T, σ_yield, ω, η_min, η_max, E, C, N = convert_to_tensor(
        [u, T, σ_yield, ω, η_min, η_max, E, C, N], [Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64]
    )
    ε = eval_strain_on_gauss_pts(u, m, n, h)
    εII = 0.5 * sum(ε .* ε, dims = 2)
    η = C * exp(E*(0.5 - T))*εII^((1-N)/2N)
    η_min + min(σ_yield/2/sqrt(εII), ω*min(η_max, η))
end