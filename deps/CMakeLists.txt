cmake_minimum_required(VERSION 3.5)
project(TF_CUSTOM_OP)
set (CMAKE_CXX_STANDARD 11)

execute_process(COMMAND julia -e "import ADCME; print(ADCME.__STR__)" OUTPUT_VARIABLE JL_OUT)
list(GET JL_OUT 0 BINDIR)
list(GET JL_OUT 1 LIBDIR)
list(GET JL_OUT 2 TF_INC)
list(GET JL_OUT 3 TF_ABI)
list(GET JL_OUT 4 EIGEN_INC)
list(GET JL_OUT 5 CC)
list(GET JL_OUT 6 CXX)
list(GET JL_OUT 7 CMAKE)
list(GET JL_OUT 8 MAKE)
list(GET JL_OUT 9 GIT)
list(GET JL_OUT 10 PYTHON)
list(GET JL_OUT 11 TF_LIB_FILE)

message("Python path=${PYTHON}")
message("EIGEN_INC=${EIGEN_INC}")
message("TF_INC=${TF_INC}")
message("TF_ABI=${TF_ABI}")
message("TF_LIB_FILE=${TF_LIB_FILE}")


if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.0)
  set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=${TF_ABI} ${CMAKE_CXX_FLAGS}")
endif()

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

include_directories(${TF_INC} ${HEADERS} ${EIGEN_INC})
link_directories(${TF_LIB} ${LIBDIR}/Adept-2/adept/.libs)
include_directories(${LIBDIR}/Adept-2/include)


set(LIBDIR_NAME VolumetricStrain ComputeVel DirichletBd
    FemStiffness FemStiffness1 SpatialFemStiffness
    SpatialVaryingTangentElastic Strain Strain1
    StrainEnergy StrainEnergy1 CholeskyOp TpfaOp)
set(LIB_NAME VolumetricStrain ComputeVel DirichletBd
    FemStiffness UnivariateFemStiffness SpatialFemStiffness
    SpatialVaryingTangentElastic StrainOp StrainOpUnivariate
    StrainEnergy StrainEnergyUnivariate CholeskyOp TpfaOp)

list(LENGTH "LIBDIR_NAME" LIBLENGTH)
message("Total number of libraries to make: ${LIBLENGTH}")
MATH(EXPR LIBLENGTH "${LIBLENGTH}-1")
foreach(IDX RANGE 0 ${LIBLENGTH})
  list(GET LIBDIR_NAME ${IDX} _LIB_DIR)
  list(GET LIB_NAME ${IDX} _LIB_NAME)
  message("Compiling ${IDX}th library: ${_LIB_DIR}==>${_LIB_NAME}")
  file(MAKE_DIRECTORY ${_LIB_DIR}/build)
  add_library(${_LIB_NAME} SHARED ${_LIB_DIR}/${_LIB_NAME}.cpp)
  set_property(TARGET ${_LIB_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_target_properties(${_LIB_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/${_LIB_DIR}/build)
  target_link_libraries(${_LIB_NAME} ${TF_LIB_FILE})
endforeach(IDX)

target_link_libraries(CholeskyOp adept)
