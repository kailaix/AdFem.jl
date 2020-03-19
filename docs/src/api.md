# API

## Data Structures
```@docs
PoreData
```

## Matrix Assembling Functions
```@docs
compute_fem_stiffness_matrix
compute_interaction_matrix
compute_fvm_tpfa_matrix
compute_fem_mass_matrix
compute_fvm_mass_matrix
compute_fem_mass_matrix1
compute_fem_stiffness_matrix1
```

## Vector Assembling Functions
```@docs
compute_fem_source_term
compute_fvm_source_term
compute_fvm_mechanics_term
compute_fem_normal_traction_term
compute_von_mises_stress_term
compute_fem_source_term1
compute_fem_flux_term1
compute_strain_energy_term
compute_fem_viscoelasticity_strain_energy_term
```

## Evaluation Functions
```@docs
eval_f_on_gauss_pts
eval_f_on_boundary_node
eval_f_on_boundary_edge
eval_strain_on_gauss_pts
```

## Visualization 
```@docs
visualize_pressure
visualize_displacement
visualize_stress
visualize_von_mises_stress
```

## Misc

```@docs
fem_impose_Dirichlet_boundary_condition
fem_impose_Dirichlet_boundary_condition1
trim_coupled
compute_elasticity_tangent
coupled_impose_pressure
compute_vel
bcnode
bcedge
```
