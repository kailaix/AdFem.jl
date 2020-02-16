# using PyCall 
# using Pkg; Pkg.add("PyPlot")

push!(LOAD_PATH, "../src/")
using Documenter, PoreFlow
makedocs(sitename="PoreFlow", modules=[PoreFlow],
pages = Any[
    "index.md",
    "Examples"=>Any["coupled.md", "staticelasticity.md", "plasticity.md", "viscoelasticity.md", "heatequation.md","elastodynamics.md"],
    "Inverse Modeling"=>Any["inverse.md", "inv_viscoelasticity.md", "coupled_viscoelasticity.md"],
    "api.md"
],
authors = "Kailai Xu")

deploydocs(
    repo = "github.com/kailaix/PoreFlow.jl.git",
)