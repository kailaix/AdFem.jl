using ADCME

install_adept()
install_mfem()
install_had()

change_directory(joinpath(@__DIR__, "build"))
require_file("CMakeCache.txt") do 
    ADCME.cmake()
end
require_library("adfem") do 
    ADCME.make()
end

change_directory(joinpath(@__DIR__, "MFEM", "build"))
require_file("CMakeCache.txt") do 
    ADCME.cmake()
end
require_library("admfem") do 
    ADCME.make()
end

run(`$(ADCME.PIP) install pyvista`)

