export PoreData

const K_CONST = 9.869233e-13
@doc raw"""
`PoreData` is a collection of physical parameters for coupled geomechanics and flow simulation

- `M`: Biot modulus
- `b`: Biot coefficient
- `ρb`: Bulk density
- `kp`: Permeability
- `E`: Young modulus
- `ν`: Poisson ratio
- `μ`: Fluid viscosity
- `Pi`: Initial pressure
- `Bf`: formation volume, $B_f=\frac{\rho_{f,0}}{\rho_f}$
"""
mutable struct PoreData
    M::Float64
    b::Float64
    ρb::Float64
    kp::Float64 # Darcy
    E::Float64
    ν::Float64
    μ::Float64 # Poise
    Pi::Float64
    Bf::Float64
end

function PoreData()
    PoreData(
        0.1,
        1.0,
        2.4,
        50e6 * K_CONST, 
        350e6,
        0.0,
        8.9e-4, # viscosity of water
        2.125e6,
        1.0
    )
end