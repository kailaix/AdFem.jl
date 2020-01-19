export PoreData
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
    kp::Float64
    E::Float64
    ν::Float64
    μ::Float64
    Pi::Float64
    Bf::Float64
end

function PoreData()
    PoreData(
        1.0,
        1.0,
        2400,
        50,
        350e6,
        0.35,
        1.0,
        2.125e6,
        1.0
    )
end