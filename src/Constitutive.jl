export antiplane_viscosity
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