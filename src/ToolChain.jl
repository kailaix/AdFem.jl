function precompile_adfem()
    PWD = pwd()
    DEPS = joinpath(@__DIR__, "..", "deps")
    change_directory(DEPS)
    ADCME.precompile()

    install_adept()
    install_mfem()
    install_had()

    change_directory(joinpath(DEPS, "build"))
    require_file("CMakeCache.txt") do 
        ADCME.cmake()
    end
    require_library("adfem") do 
        ADCME.make()
    end

    change_directory(joinpath(DEPS, "MFEM", "build"))
    require_file("CMakeCache.txt") do 
        ADCME.cmake()
    end
    require_library("admfem") do 
        ADCME.make()
    end

    change_directory(joinpath(DEPS, "MFEM3", "build"))
    require_file("CMakeCache.txt") do 
        ADCME.cmake()
    end
    require_library("admfem") do 
        ADCME.make()
    end


    PIP = get_pip()
    try 
        run(`$(PIP) install pyvista`)
    catch 
        @warn "pyvista installation was not successful. 3D plots functionalities are disabled."
    end
    change_directory(PWD)

end