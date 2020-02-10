using ADCME
function compile(DIR)
    PWD = pwd()
    cd(DIR)
    rm("build", force=true, recursive=true)
    mkdir("build")
    cd("build")
    ADCME.cmake()
    ADCME.make()
    cd(PWD)
end


compile("$(@__DIR__)/deps/DirichletBD")
compile("$(@__DIR__)/deps/FemStiffness")
compile("$(@__DIR__)/deps/Strain")
compile("$(@__DIR__)/deps/StrainEnergy")
compile("$(@__DIR__)/deps/SpatialFemStiffness")