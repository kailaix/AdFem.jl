export compute_fem_viscoelasticity_strain_energy_term

@doc raw"""
    compute_fem_viscoelasticity_strain_energy_term(ε0, σ0, ε, A, B, m, n, h)

Given the constitutive relation
```math
\sigma^{n+1} = S \sigma^n + H (\varepsilon^{n+1}-\varepsilon^n),
```
this function computes 
```math
\int_A {\sigma:\delta \varepsilon}\mathrm{d} x = \underbrace{\int_A { B \varepsilon^{n+1}:\delta \varepsilon}\mathrm{d} x}  + \underbrace{ \int_A { A \sigma^{n+1}:\delta \varepsilon}\mathrm{d} x - \int_A { B \varepsilon^{n+1}:\delta \varepsilon}\mathrm{d} x }_f
```
and returns $f$
"""
function compute_fem_viscoelasticity_strain_energy_term(ε0::Union{Array{Float64, 2}, PyObject}, 
    σ0::Union{Array{Float64, 2}, PyObject}, 
    S::Union{Array{Float64, 2}, PyObject}, H::Union{Array{Float64, 2}, PyObject}, 
    m::Int64, n::Int64, h::Float64) 
    ε0, σ0, S, H = convert_to_tensor([ε0, σ0, S, H], [Float64, Float64, Float64, Float64])
    f1 = -compute_strain_energy_term(ε0 * H, m, n, h)
    f2 = compute_strain_energy_term(σ0 * S, m, n, h)
    return f1 + f2
end

