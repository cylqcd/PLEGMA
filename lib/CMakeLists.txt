# cmake-format: off

# this allows simplified running of clang-tidy
if(${CMAKE_BUILD_TYPE} STREQUAL "DEVEL")
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

#build up git version add -debug to GITVERSION if we build with debug options enabled
string(REGEX MATCH [Dd][Ee][Bb][Uu][Gg] DEBUG_BUILD ${CMAKE_BUILD_TYPE})
if(DEBUG_BUILD)
  if(GITVERSION)
    set(GITVERSION ${GITVERSION}-debug)
  else()
    set(GITVERSION debug)
  endif()
endif()

set (PLEGMA_OBJS 
  PLEGMA_Correlator.cu
  PLEGMA.cu
  PLEGMA_Propagator.cu
  PLEGMA_Vector.cu
  PLEGMA_Gauge.cu
  PLEGMA_Field.cu
  PLEGMA_Su3field.cu
  PLEGMA_Random.cu
  kernels/PLEGMA_baryons_NtoN.cu
  kernels/PLEGMA_seqSourceNucleon.cu
  kernels/PLEGMA_threep_local.cu
  kernels/PLEGMA_threep_noe.cu
  kernels/PLEGMA_threep_oneD.cu
  kernels/PLEGMA_threep_twoD.cu
  kernels/PLEGMA_threep_threeD_part1.cu
  kernels/PLEGMA_threep_threeD_part2.cu
  kernels/PLEGMA_threep_threeD_part3.cu
  kernels/PLEGMA_threep_threeD_part4.cu
  kernels/PLEGMA_threep_wilsonLine.cu
  kernels/PLEGMA_threep_staple.cu
  kernels/PLEGMA_threep_qgq.cu	
  PLEGMA_QLoops.cu
  PLEGMA_FT.cu
  PLEGMA_BLAS.cu
  PLEGMA_Fmunu.cu
  PLEGMA_U1Gauge.cu	
)
if(PLEGMA_UDSC_BARYONS)
  LIST(APPEND PLEGMA_OBJS
    kernels/PLEGMA_baryons_udsc.cu
    kernels/PLEGMA_baryons_udsc_arrays.cu
    kernels/PLEGMA_heavy_light_tetraquarks.cu
    kernels/PLEGMA_heavy_light_tetraquarks_stochastic.cu
    kernels/PLEGMA_heavy_light_tetraquarks_open_contractions.cu
    kernels/PLEGMA_heavy_light_tetraquarks_arrays.cu
    kernels/PLEGMA_heavy_light_tetraquarks_arrays_stoch.cu
    kernels/PLEGMA_bcud_tetraquarks.cu
    kernels/PLEGMA_bcud_tetraquarks_stochastic.cu
    kernels/PLEGMA_bcud_tetraquarks_arrays.cu
    kernels/PLEGMA_bcud_tetraquarks_arrays_stoch.cu
  )
endif()
if(PLEGMA_SCATTERING_CONTRACTIONS)
  LIST(APPEND PLEGMA_OBJS
    PLEGMA_ScattCorrelator.cu
    kernels/PLEGMA_scattreductionsT.cu
    kernels/PLEGMA_scattreductionsV.cu
  )
endif()
    

if(PLEGMA_LIGHT_BARYONS)
  LIST(APPEND PLEGMA_OBJS
    kernels/PLEGMA_baryons_NtoR.cu
    kernels/PLEGMA_baryons_RtoN.cu
    kernels/PLEGMA_baryons_RtoR.cu
    kernels/PLEGMA_baryons_deltas.cu
  )
endif()

## split source into cu and cpp files
FOREACH(item ${PLEGMA_OBJS})
  STRING(REGEX MATCH ".+\\.cu$" item_match ${item})
  IF(item_match)
	  LIST(APPEND PLEGMA_CU_OBJS ${item})
  ENDIF(item_match)
ENDFOREACH(item ${PLEGMA_OBJS})

LIST(REMOVE_ITEM PLEGMA_OBJS ${PLEGMA_CU_OBJS})

# PLEGMA_CU_OBJS should contain all cuda files now PLEGMA_OBJS all c, cpp, fortran sources
# if we have a git version make version.cpp depend on git head so that it is rebuild if the git sha changed

if(PLEGMA_OBJS)
  add_library(plegma_cpp OBJECT ${PLEGMA_OBJS})

  target_compile_definitions(plegma_cpp PUBLIC -DQUDA_HASH="${HASH}")
  set_target_properties(plegma_cpp PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

endif()
if(${CMAKE_BUILD_TYPE} STREQUAL "DEVEL")
  if(GITVERSION)
    find_path(
      PLEGMA_GITDIR NAME HEAD
      PATHS ${CMAKE_SOURCE_DIR}/.git/logs
      NO_DEFAULT_PATH)
    include(AddFileDependencies)
    if(PLEGMA_GITDIR)
      add_file_dependencies(version.cpp ${PLEGMA_GITDIR}/HEAD)
    endif()
  endif()
  mark_as_advanced(PLEGMA_GITDIR)
endif()

# QUDA_CU_OBJS shoudl contain all cuda files now
# PLEGMA_OBJS all c, cpp, fortran sources

if(PLEGMA_BUILD_SHAREDLIB)
  if (PLEGMA_OBJS)
    set_target_properties(plegma_cpp PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
  endif()
  add_library(plegma SHARED)
else()
  add_library(plegma STATIC)
endif()
add_library(PLEGMA::plegma ALIAS plegma)

# add target specific files
if(${PLEGMA_TARGET_TYPE} STREQUAL "CUDA")
  include(targets/cuda/target_cuda.cmake)
endif()
if(${PLEGMA_TARGET_TYPE} STREQUAL "HIP")
  include(targets/hip/target_hip.cmake)
endif()
if(${PLEGMA_TARGET_TYPE} STREQUAL "SYCL")
  include(targets/sycl/target_sycl.cmake)
endif()

# make one library
target_sources(plegma PRIVATE ${PLEGMA_CU_OBJS})
#<TARGET_OBJECTS:plegma_cpp> 
#	                      ${PLEGMA_CU_OBJS})

include(CheckLinkerFlag)
check_linker_flag(CXX "-Wl,--compress-debug-sections=zlib" PLEGMA_LINKER_COMPRESS)
target_link_options(plegma PRIVATE $<$<CONFIG:DEVEL>:$<${PLEGMA_LINKER_COMPRESS}:-Wl,--compress-debug-sections=zlib>>)


# set up QUDA compile options, put them before target specific options
target_compile_options(
  plegma BEFORE PRIVATE $<$<COMPILE_LANGUAGE:CXX,C>:
               $<IF:$<CONFIG:RELEASE>,-w,-Wall -Wextra>
               $<$<CONFIG:STRICT>:-Werror>
  >)

target_compile_definitions(
  plegma PRIVATE $<$<CONFIG:DEVEL>:DEVEL> $<$<CONFIG:HOSTDEBUG>:HOST_DEBUG> $<$<CONFIG:DEBUG>:HOST_DEBUG>
               $<$<CONFIG:DEBUG>:DEVICE_DEBUG> $<$<CONFIG:SANITIZE>:HOST_DEBUG>)
target_include_directories(plegma PRIVATE .)
target_include_directories(plegma PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})


# include_directories
target_include_directories(plegma SYSTEM PRIVATE ${QUDA_HOME}/lib/)
#PLEGMA: 
target_include_directories(plegma SYSTEM PRIVATE ${PLEGMA_GSLHOME}/include)
target_include_directories(plegma SYSTEM PRIVATE ${PLEGMA_HDF5HOME}/include)

target_include_directories(plegma SYSTEM PRIVATE ${PLEGMA_HDF5HOME}/src)
target_include_directories(plegma SYSTEM PRIVATE ${PLEGMA_LIMEHOME}/include)
if(PLEGMA_ARPACK)
	target_include_directories(SYSTEM PRIVATE ${PLEGMA_ARPACKHOME}/PARPACK/SRC/MPI)
	target_include_directories(SYSTEM PRIVATE ${PLEGMA_ARPACKHOME}/PARPACK/UTIL/MPI)
	target_include_directories(SYSTEM PRIVATE ${PLEGMA_ARPACKHOME}/SRC)
	target_include_directories(SYSTEM PRIVATE ${PLEGMA_ARPACKHOME}/UTIL)
endif(PLEGMA_ARPACK)



LIST(APPEND PLEGMA ${PLEGMA_GSL_LIB})
LIST(APPEND PLEGMA ${PLEGMA_HDF5_LIB})
LIST(APPEND PLEGMA ${PLEGMA_LIME_LIB})
if(PLEGMA_ARPACK)
  LIST(APPEND PLEGMA ${PLEGMA_PARPACK_LIB})
  LIST(APPEND PLEGMA ${PLEGMA_ARPACK_LIB})
endif(PLEGMA_ARPACK)

if(PLEGMA_OPENBLAS)
  LIST(APPEND PLEGMA ${PLEGMA_OPENBLAS_GOMP})
  LIST(APPEND PLEGMA ${PLEGMA_OPENBLAS_LIB})
endif(PLEGMA_OPENBLAS)
  
if(PLEGMA_MKL)
  LIST(APPEND PLEGMA ${PLEGMA_MKL_LIB_IOMP5})
  LIST(APPEND PLEGMA ${PLEGMA_MKL_LIB_GFORTRAN})
  LIST(APPEND PLEGMA ${PLEGMA_MKL_LIB_LP64})
  LIST(APPEND PLEGMA ${PLEGMA_MKL_LIB_THREAD})
  LIST(APPEND PLEGMA ${PLEGMA_MKL_LIB_CORE})
endif(PLEGMA_MKL)



# add dependencies for linking
# if CUDA_cuda_LIBRARY (driver api) is not found look for stubs
#FIND_LIBRARY(CUDA_cuda_LIBRARY cuda HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib/ ${CUDA_TOOLKIT_ROOT_DIR}/lib/stubs)

target_link_libraries(plegma PUBLIC ${CMAKE_THREAD_LIBS_INIT} ${QUDA_LIB} ${QUDA_LIBS}) # ${CUDA_cuda_LIBRARY})
add_dependencies(plegma BoostPP)

if(QUDA_QMP)
  target_link_libraries(PLEGMA ${QMP_LIB} ${MPI_CXX_LIBRARIES})
endif()

if(QUDA_MPI)
  target_link_libraries(PLEGMA ${MPI_CXX_LIBRARIES})
endif()

if(QUDA_MAGMA)
  target_link_libraries(PLEGMA ${MAGMA})
endif()

if(QUDA_ARPACK)
  target_link_libraries(PLEGMA ${PARPACK_OR_ARPACK})
endif()


# until we define an install step copy the include directory to the build directory
ADD_CUSTOM_COMMAND(TARGET plegma POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
# some hackery to prevent having old shared / static builds of quda messing with the current build
ADD_CUSTOM_COMMAND(TARGET plegma PRE_LINK COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/libPLEGMA.a ${CMAKE_CURRENT_BINARY_DIR}/libPLEGMA.so)

install(TARGETS plegma EXPORT PLEGMATargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        INCLUDES DESTINATION include)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include)
include(CMakePackageConfigHelpers)
write_basic_package_version_file( 
  "${CMAKE_CURRENT_BINARY_DIR}/qudaConfigVersion.cmake" 
  VERSION ${QUDA_VERSION}
  COMPATIBILITY AnyNewerVersion
)

export(EXPORT PLEGMATargets FILE "${CMAKE_CURRENT_BINARY_DIR}/PLEGMATargets.cmake" NAMESPACE quda::)
set(ConfigPackageLocation lib/cmake/quda/)
install(EXPORT PLEGMATargets NAMESPACE quda:: DESTINATION ${ConfigPackageLocation})


add_custom_target(gen ${PYTHON_EXECUTABLE} ${PLEGMA_QUDA_HOME}/lib/generate/gen.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating dslash_core files" )

add_custom_target(mpi_nvtx ${PYTHON_EXECUTABLE} ${PLEGMA_QUDA_HOME}/lib/generate/wrap.py -g -o ${PLEGMA_QUDA_HOME}/lib/nvtx_pmpi.c ${PLEGMA_QUDA_HOME}/lib/generate/nvtx.w
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating mpi_nvtx wrapper" )

