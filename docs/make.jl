# using PyCall 
# using Pkg; Pkg.add("PyPlot")

push!(LOAD_PATH, "../src/")
using Documenter, PoreFlow
makedocs(sitename="PoreFlow", modules=[PoreFlow],
pages = Any[
    "coupled.md",
    "plasticity.md",
    "api.md"
],
authors = "Kailai Xu")

deploydocs(
    repo = "github.com/kailaix/PoreFlow.jl.git",
)