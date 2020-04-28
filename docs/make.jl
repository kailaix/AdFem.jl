# using PyCall 
# using Pkg; Pkg.add("PyPlot")

push!(LOAD_PATH, "../src/")
using Documenter, PoreFlow
makedocs(sitename="PoreFlow", modules=[PoreFlow],
pages = Any[
    "index.md",
    "Examples"=>Any["coupled.md", "staticelasticity.md", "plasticity.md", 
        "viscoelasticity.md", "viscoelasticity_antiplane.md", "earthquake.md", "heatequation.md","elastodynamics.md", "twophaseflow.md"],
    "Inverse Modeling"=>Any["inverse.md", "inv_viscoelasticity.md", "coupled_viscoelasticity.md",
            "inv_twophaseflow.md", "inv_viscoelasticity_nonparametric.md"],
    "api.md"
],
authors = "Kailai Xu")

deploydocs(
    repo = "github.com/kailaix/PoreFlow.jl.git",
)