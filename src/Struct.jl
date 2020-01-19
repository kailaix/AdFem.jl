export PoreData
"""
`PoreData` is a collection of physical parameters for coupled geomechanics and flow simulation

- `M`: Biot modulus
- `b`: Biot coefficient
- `ρb`: Bulk density
- `kp`: Permeability
- `E`: Young modulus
- `ν`: Poisson ratio
- `μ`: Fluid viscosity
- `Pi`: Initial pressure
- `Pbc`: Boundary pressure
- `Bf`: ???
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
    Pbc::Float64
    Bf::Float64
end