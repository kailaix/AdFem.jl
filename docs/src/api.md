# API

## Data Structures
```@docs
PoreData
Mesh
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
compute_fvm_advection_matrix
compute_fem_laplace_matrix1
compute_fem_laplace_matrix
compute_fem_advection_matrix1
```

## Vector Assembling Functions
```@docs
compute_fem_source_term
compute_fvm_source_term
compute_fvm_mechanics_term
compute_fem_normal_traction_term
compute_fem_traction_term
compute_von_mises_stress_term
compute_fem_source_term1
compute_fem_flux_term1
compute_strain_energy_term
compute_strain_energy_term1
compute_fem_viscoelasticity_strain_energy_term
compute_fvm_advection_term
compute_interaction_term
compute_fem_laplace_term1
compute_fem_traction_term1
```

## Evaluation Functions
```@docs
eval_f_on_gauss_pts
eval_f_on_dof_pts
eval_f_on_boundary_node
eval_f_on_boundary_edge
eval_strain_on_gauss_pts
eval_strain_on_gauss_pts1
eval_f_on_fvm_pts
eval_f_on_fem_pts
eval_grad_on_gauss_pts1
eval_grad_on_gauss_pts
```

## Boundary Conditions
```@docs
fem_impose_Dirichlet_boundary_condition
fem_impose_Dirichlet_boundary_condition1
impose_Dirichlet_boundary_conditions
```


## Visualization 
In `visualize_scalar_on_XXX_points`, the first argument is the data matrix. When the data matrix is 1D, one snapshot is plotted. When the data matrix is 2D, it is understood as multiple snapshots at different time steps (each row is a snapshot). When the data matrix is 3D, it is understood as `time step × height × width`. 

```@docs
visualize_mesh
visualize_pressure
visualize_displacement
visualize_stress
visualize_von_mises_stress
visualize_scalar_on_gauss_points
visualize_scalar_on_fem_points
visualize_scalar_on_fvm_points
visualize_vector_on_fem_points
```

## Modeling Tools
```@docs
layer_model
compute_vel
compute_plane_strain_matrix
compute_plane_stress_matrix
compute_space_varying_tangent_elasticity_matrix
mantle_viscosity
antiplane_viscosity
```

## Mesh
```@docs
get_edge_dof
get_area
get_ngauss
bcnode
bcedge
interior_node
femidx
fvmidx
subdomain
gauss_nodes
fem_nodes
fvm_nodes
```

## Misc

```@docs
trim_coupled
coupled_impose_pressure
cholesky_factorize
cholesky_outproduct
fem_to_fvm
fem_to_gauss_points
dof_to_gauss_points
```
