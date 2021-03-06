cmake_minimum_required(VERSION 3.5)
project(TF_CUSTOM_OP)
set (CMAKE_CXX_STANDARD 11)

message("JULIA=${JULIA}")

IF(JULIA)
ELSE()
find_program(JULIA "julia")
ENDIF()

execute_process(COMMAND ${JULIA} -e "import ADCME; print(ADCME.__STR__)" OUTPUT_VARIABLE JL_OUT)
list(GET JL_OUT 0 BINDIR)
list(GET JL_OUT 1 LIBDIR)
list(GET JL_OUT 2 TF_INC)
list(GET JL_OUT 3 TF_ABI)
list(GET JL_OUT 4 PREFIXDIR)
list(GET JL_OUT 5 CC)
list(GET JL_OUT 6 CXX)
list(GET JL_OUT 7 CMAKE)
list(GET JL_OUT 8 MAKE)
list(GET JL_OUT 9 GIT)
list(GET JL_OUT 10 PYTHON)
list(GET JL_OUT 11 TF_LIB_FILE)
list(GET JL_OUT 12 LIBCUDA)
list(GET JL_OUT 13 CUDA_INC)

message("Python path=${PYTHON}")
message("LIBDIR=${LIBDIR}")
message("TF_INC=${TF_INC}")
message("TF_ABI=${TF_ABI}")
message("TF_LIB_FILE=${TF_LIB_FILE}")


if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.0)
  set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=${TF_ABI} ${CMAKE_CXX_FLAGS}")
endif()

set(CMAKE_BUILD_TYPE Release)
if(MSVC)
set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG")
else()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

if(MSVC)
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W0" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W0")
  endif()
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
  add_definitions(-DNOMINMAX)
endif()


include_directories(${TF_INC} ${PREFIXDIR} ${CUDA_INC} ${EIGEN_INC} ${LIBDIR}/../include ${PREFIXDIR}/had ${PREFIXDIR}/Adept-2/include)

find_library(ADEPT_LIB_FILE adept HINTS ${LIBDIR})
message("ADEPT_LIB_FILE=${ADEPT_LIB_FILE}")
find_library(LIBOPENBLAS openblas HINTS ${LIBDIR})


add_library(adfem SHARED Advection/Advection.cpp CholeskyOp/CholeskyOp.cpp ComputeVel/ComputeVel.cpp 
  DirichletBd/DirichletBd.cpp FemStiffness/FemStiffness.cpp FemStiffness1/UnivariateFemStiffness.cpp 
  SpatialFemStiffness/SpatialFemStiffness.cpp SpatialVaryingTangentElastic/SpatialVaryingTangentElastic.cpp Strain/StrainOp.cpp 
  Strain1/StrainOpUnivariate.cpp StrainEnergy/StrainEnergy.cpp StrainEnergy1/StrainEnergyUnivariate.cpp 
  TpfaOp/TpfaOp.cpp VolumetricStrain/VolumetricStrain.cpp ImplicitAdvection/ImplicitAdvection.cpp FemSource/FemSource.cpp
  FemGrad/FemGrad.cpp FemMass/FemMass.cpp FemAdvection/FemAdvection.cpp FemToGaussPoints/FemToGaussPoints.cpp
  FemLaplace/FemLaplace.cpp InteractionTerm/InteractionTerm.cpp FemLaplaceTerm/FemLaplaceTerm.cpp
  ViscoelasticityStressUpdate/ViscoelasticityStressUpdate.cpp)
set_property(TARGET adfem PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(adfem ${TF_LIB_FILE} ${ADEPT_LIB_FILE} ${LIBOPENBLAS})
set_target_properties(adfem PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build  RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/build)
