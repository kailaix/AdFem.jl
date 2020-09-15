using ADCME

function install_mfem()
    PWD = pwd()
    change_directory()
    http_file("https://bit.ly/mfem-4-1", "mfem-4-1.tgz")
    uncompress("mfem-4-1.tgz", "mfem-4.1")
    str = String(read("mfem-4.1/CMakeLists.txt"))
    str = replace(str, "add_library(mfem \${SOURCES} \${HEADERS} \${MASTER_HEADERS})"=>"""add_library(mfem SHARED \${SOURCES} \${HEADERS} \${MASTER_HEADERS})
set_property(TARGET mfem PROPERTY POSITION_INDEPENDENT_CODE ON)""")
    open("mfem-4.1/CMakeLists.txt", "w") do io 
        write(io, str)
    end
    change_directory("mfem-4.1/build")
    require_file("build.ninja") do
        ADCME.cmake(CMAKE_ARGS = ["-DCMAKE_INSTALL_PREFIX=$(joinpath(ADCME.LIBDIR, ".."))", "SHARED=YES", "STATIC=NO"])
    end
    require_library("mfem") do 
        ADCME.make()
    end
    require_file(joinpath(ADCME.LIBDIR, get_library_name("mfem"))) do 
        run_with_env(`$(ADCME.NINJA) install`)
    end
    cd(PWD)
end

install_adept()
install_mfem()

change_directory(joinpath(@__DIR__, "build"))
require_file("build.ninja") do 
    ADCME.cmake()
end
require_library("poreflow") do 
    ADCME.make()
end

change_directory(joinpath(@__DIR__, "MFEM", "build"))
require_file("build.ninja") do 
    ADCME.cmake()
end
require_library("nnfem_mfem") do 
    ADCME.make()
end


