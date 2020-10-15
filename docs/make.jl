# using PyCall 
# using Pkg; Pkg.add("PyPlot")

push!(LOAD_PATH, "../src/")
using Documenter, PoreFlow
makedocs(sitename="AdFem", modules=[PoreFlow],
    pages = Any[
        "index.md",
        "Tutorial"=>Any["gallery.md"],
        "Forward Computation"=>Any["coupled.md", "staticelasticity.md", "plasticity.md", 
            "viscoelasticity.md", "viscoelasticity_earth.md", "earthquake.md", 
            "heatequation.md","elastodynamics.md", "twophaseflow.md", "NavierStokes2.md",
            "SteadyStateNavierStokes.md", "CHTCoupled.md", 
            "fwd_mixed_poisson.md",
            "fwd_linear_elasticity.md",
            "fwd_stress_based_viscoelasticity.md"],
        "Inverse Modeling"=>Any["inverse.md", "inv_viscoelasticity.md", "coupled_viscoelasticity.md",
            "inv_twophaseflow.md", "inv_viscoelasticity_nonparametric.md", "inv_viscoelasticity_earth.md"],
        "Advanced Topics"=>Any["mfem_tutorial.md", "mfem_mesh.md", "dev_unstructured.md", "BDMElement.md"],
        "api.md"
    ],
    authors = "Kailai Xu"
)

deploydocs(
    repo = "github.com/kailaix/PoreFlow.jl.git",
)