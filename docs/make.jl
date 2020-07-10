# using PyCall 
# using Pkg; Pkg.add("PyPlot")

push!(LOAD_PATH, "../src/")
using Documenter, PoreFlow
makedocs(sitename="PoreFlow", modules=[PoreFlow],
pages = Any[
    "index.md",
    "gallery.md",
    "Examples"=>Any["coupled.md", "staticelasticity.md", "plasticity.md", 
        "viscoelasticity.md", "viscoelasticity_earth.md", "earthquake.md", 
        "heatequation.md","elastodynamics.md", "twophaseflow.md", "NavierStokes2.md"],
    "Inverse Modeling"=>Any["inverse.md", "inv_viscoelasticity.md", "coupled_viscoelasticity.md",
            "inv_twophaseflow.md", "inv_viscoelasticity_nonparametric.md", "inv_viscoelasticity_earth.md"],
    "api.md"
],
authors = "Kailai Xu")

deploydocs(
    repo = "github.com/kailaix/PoreFlow.jl.git",
)