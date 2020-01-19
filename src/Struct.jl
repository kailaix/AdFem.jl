export PoreData

@doc raw"""
`PoreData` is a collection of physical parameters for coupled geomechanics and flow simulation

- `M`: Biot modulus
- `b`: Biot coefficient
- `ρb`: Bulk density
- `ρf`: Fluid density
- `kp`: Permeability
- `E`: Young modulus
- `ν`: Poisson ratio
- `μ`: Fluid viscosity
- `Pi`: Initial pressure
- `Bf`: formation volume, $B_f=\frac{\rho_{f,0}}{\rho_f}$
- `g`: Gravity acceleration
"""
@with_kw mutable struct PoreData
    M::Float64 = 0.1
    b::Float64 = 1.0
    ρb::Float64 = 2400.
    ρf::Float64 = 1000.
    kp::Float64 = 50e6 * 9.869233e-13 # Darcy
    E::Float64 = 350e6
    ν::Float64 = 0.0
    μ::Float64 = 8.9e-4 # viscosity of water
    Pi::Float64 = 2.125e6
    Bf::Float64 = 1.0
    g::Float64 = 9.807
end