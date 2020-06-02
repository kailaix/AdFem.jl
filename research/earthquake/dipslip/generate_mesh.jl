using Revise
using NNFEM

init_gmsh()
pts = [
    addPoint(0,0, 1.0),
    addPoint(8,0, 1.0),
    addPoint(8,4, 1.0),
    addPoint(0,4, 1.0),
    addPoint(3.13, 0.0, 0.8),
    addPoint(4.0, 0.5, 0.8)
]
rectangle = addPlaneSurface(
    [addCurveLoop([
        addLine(1,2),
        addLine(2,3),
        addLine(3,4),
        addLine(4,1)
    ])]
)
# meshsize("")
line = addLine(5,6)
embedLine([line], rectangle)
filename = finalize_gmsh()
cp(filename, "test.msh")