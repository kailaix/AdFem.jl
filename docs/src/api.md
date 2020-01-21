# API

## Data Structures
```@docs
PoreData
```

## Matrix Assembling Functions
```@docs
compute_fem_stiffness_matrix
compute_interaction_matrix
compute_fluid_tpfa_matrix
compute_fem_mass_matrix
```

## Vector Assembling Functions
```@docs
compute_fem_source_term
compute_fvm_source_term
compute_fvm_mechanics_term
compute_fem_normal_traction_term
compute_principal_stress_term
```

## Misc

```@docs
trim_fem
eval_f_on_gauss_pts
trim_coupled
compute_elasticity_tangent
coupled_impose_pressure
```
