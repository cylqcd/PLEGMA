set (QUDA_OBJS 
  PLEGMA_Correlator.cu
  PLEGMA.cu
  PLEGMA_Propagator.cu
  PLEGMA_Vector.cu
  PLEGMA_Gauge.cu
  PLEGMA_Field.cu
  PLEGMA_Su3field.cu
  kernels/PLEGMA_baryons_NtoN.cu
  kernels/PLEGMA_seqSourceNucleon.cu
  kernels/PLEGMA_contractPropOpProp_local.cu
  kernels/PLEGMA_contractPropOpProp_noe.cu
  kernels/PLEGMA_contractPropOpProp_oneD.cu
  kernels/PLEGMA_contractPropOpProp_wilsonLine.cu
  PLEGMA_QLoops.cu
  PLEGMA_FT.cu
)

if(PLEGMA_ALL_BARYONS)
  LIST(APPEND QUDA_OBJS
    kernels/PLEGMA_baryons_NtoR.cu
    kernels/PLEGMA_baryons_RtoN.cu
    kernels/PLEGMA_baryons_RtoR.cu
    kernels/PLEGMA_baryons_deltas.cu
  )
endif()

## split source into cu and cpp files
FOREACH(item ${QUDA_OBJS})
  STRING(REGEX MATCH ".+\\.cu$" item_match ${item})
  IF(item_match)
    LIST(APPEND QUDA_CU_OBJS ${item})
  ENDIF(item_match)
ENDFOREACH(item ${QUDA_OBJS})

LIST(REMOVE_ITEM QUDA_OBJS ${QUDA_CU_OBJS})

# QUDA_CU_OBJS shoudl contain all cuda files now
# QUDA_OBJS all c, cpp, fortran sources

# include_directories
include_directories(${QUDA_HOME}/lib/)
#PLEGMA: 
include_directories(code_pieces)
include_directories(SYSTEM ${PLEGMA_GSLHOME}/include)
include_directories(SYSTEM ${PLEGMA_HDF5HOME}/src)
include_directories(SYSTEM ${PLEGMA_LIMEHOME}/include)
if(PLEGMA_ARPACK)
  include_directories(SYSTEM ${PLEGMA_ARPACKHOME}/PARPACK/SRC/MPI)
  include_directories(SYSTEM ${PLEGMA_ARPACKHOME}/PARPACK/UTIL/MPI)
  include_directories(SYSTEM ${PLEGMA_ARPACKHOME}/SRC)
  include_directories(SYSTEM ${PLEGMA_ARPACKHOME}/UTIL)
endif(PLEGMA_ARPACK)
#PLEGMA: 

set(PLEGMA_LIB ${QUDA_CU_OBJS})

# generate a cmake object library for all cpp files first
if(QUDA_OBJS)
  add_library(PLEGMA_cpp OBJECT ${QUDA_OBJS})

  # add some definitions that cause issues with cmake 3.7 and nvcc only to cpp files 
  target_compile_definitions(PLEGMA_cpp PUBLIC -DQUDA_HASH="${HASH}")
  if(PLEGMA_BUILD_SHAREDLIB)
    set_target_properties(PLEGMA_cpp PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
  endif()

  LIST(APPEND PLEGMA_LIB $<TARGET_OBJECTS:PLEGMA_cpp>)

endif()

# make one library
if(PLEGMA_BUILD_SHAREDLIB)
    cuda_add_library(PLEGMA SHARED ${PLEGMA_LIB})
else()
  cuda_add_library(PLEGMA STATIC ${PLEGMA_LIB})
endif()
target_include_directories(PLEGMA PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)


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
# if CUDA_CUDA_LIBRARY (driver api) is not found look for stubs
FIND_LIBRARY(CUDA_CUDA_LIBRARY cuda HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib/ ${CUDA_TOOLKIT_ROOT_DIR}/lib/stubs)

target_link_libraries(PLEGMA ${CMAKE_THREAD_LIBS_INIT} ${QUDA_LIB} ${QUDA_LIBS} ${CUDA_CUDA_LIBRARY})

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

# malloc.cpp uses both the driver and runtime api
# So we need to find the CUDA_CUDA_LIBRARY (driver api) or the stub version
find_library(CUDA_CUDA_LIBRARY cuda HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib/ ${CUDA_TOOLKIT_ROOT_DIR}/lib/stubs)
target_link_libraries(PLEGMA ${CUDA_CUDA_LIBRARY})

# until we define an install step copy the include directory to the build directory
ADD_CUSTOM_COMMAND(TARGET PLEGMA POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
# some hackery to prevent having old shared / static builds of quda messing with the current build
ADD_CUSTOM_COMMAND(TARGET PLEGMA PRE_LINK COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/libPLEGMA.a ${CMAKE_CURRENT_BINARY_DIR}/libPLEGMA.so)

install(TARGETS PLEGMA EXPORT PLEGMATargets
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
