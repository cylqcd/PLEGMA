# ######################################################################################################################
# malloc.cpp uses both the driver and runtime api
# So we need to find the CUDA_cuda_LIBRARY (driver api) or the stub version
find_library(CUDA_cuda_LIBRARY cuda HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib/ ${CUDA_TOOLKIT_ROOT_DIR}/lib/stubs)
target_link_libraries(plegma PUBLIC ${CUDA_cuda_LIBRARY})
# CUDA specific part of CMakeLists
include(CheckLanguage)
check_language(CUDA)

set(PLEGMA_TARGET_CUDA ON)

# using dirs in LD_LIBRARY_PATh for searching for libraries
string(REPLACE ":" ";" LIBRARY_DIRS $ENV{LD_LIBRARY_PATH})

set(DEFAULT_GPU_ARCH sm_70)
set(GPU_ARCH ${DEFAULT_GPU_ARCH} CACHE STRING "set the GPU architecture (sm_20, sm_21, sm_30, sm_35, sm_37, sm_50, sm_52, sm_60, sm_70)")
set_property(CACHE GPU_ARCH PROPERTY STRINGS sm_20 sm_21 sm_30 sm_35 sm_37 sm_50 sm_52 sm_60 sm_70)

# Use CUDA textures
set(PLEGMA_TEXTURE TRUE CACHE BOOL "Wheater to use or not CUDA textures")
mark_as_advanced(PLEGMA_TEXTURE)
if(PLEGMA_TEXTURE)
  add_definitions(-DPLEGMA_TEXTURE)
else()
  
endif()

#######################################################################
# everything below here is processing the setup
#######################################################################
# we need to check for some packages
find_package(PythonInterp)

if(${CMAKE_VERSION} VERSION_GREATER 3.7.99)
  find_package(CUDAWrapper)
  set(USING_CUDA_LANG_SUPPORT True)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED True)
else()
  set(CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "Host side compiler used by NVCC")
  mark_as_advanced(CUDA_HOST_COMPILER)
  find_package(CUDA REQUIRED)
  set(USING_CUDA_LANG_SUPPORT False)
endif()

# solve compiler issues with CUDA and tuples
if( (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.5) AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)  AND (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER 9.0) AND (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 9.2))
  message(FATAL_ERROR "This library will have compilation problems with CUDA 9.1 and gcc 6")
endif()
if( (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6) AND (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER 9.1) AND (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 9.3) )
  message(FATAL_ERROR "This library will have compilation problems with CUDA 9.2 and gcc < 6")
endif()

LIST(APPEND CUDA_LIBS ${CUDA_cufft_LIBRARY} ${CUDA_curand_LIBRARY})
LIST(APPEND CUDA_LIBS ${CUDA_cublas_LIBRARY})
LIST(APPEND CUDA_LIBS ${CUDA_nvrtc_LIBRARY})
LIST(APPEND CUDA_LIBS ${CUDA_nvToolsExt_LIBRARY})

find_package(LibDL)
LIST(APPEND CUDA_LIBS ${LIBDL_LIBRARIES})

add_definitions(-DMULTI_GPU)

include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
include_directories(lib/kernels)


# GPU ARCH
STRING(REGEX REPLACE sm_ "" COMP_CAP ${GPU_ARCH})
SET(COMP_CAP "${COMP_CAP}0")
add_definitions(-D__COMPUTE_CAPABILITY__=${COMP_CAP})


# NVCC FLAGS independet off build type
if(NOT USING_CUDA_LANG_SUPPORT)
  set(CUDA_NVCC_FLAGS "-std c++17 -arch=${GPU_ARCH} -ftz=true -prec-div=false -prec-sqrt=false")
else()
  set(CUDA_NVCC_FLAGS "-ftz=true -prec-div=false -prec-sqrt=false")
  set(CMAKE_CUDA_FLAGS "-arch=${GPU_ARCH}" CACHE STRING "Flags used by the CUDA compiler" FORCE)
endif()

# some clang warnings shouds be warning even when turning warnings into errors
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CLANG_NOERROR "-Wno-error=unused-private-field")
# this is a hack to get colored diagnostics back when using Ninja and clang
    if(CMAKE_GENERATOR MATCHES "Ninja")
      set(CLANG_FORCE_COLOR "-fcolor-diagnostics")
    endif()
endif()

## define CUDA flags when CMake < 3.8
set(CUDA_NVCC_FLAGS_DEVEL ${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets -Xcompiler -Wno-unknown-pragmas,-Wno-unused-function,-Wno-unused-local-typedef,-Wno-unused-private-field -O3 -lineinfo CACHE STRING
    "Flags used by the CUDA compiler during regular development builds."
    FORCE )
set(CUDA_NVCC_FLAGS_STRICT ${CUDA_NVCC_FLAGS_DEVEL} CACHE STRING
    "Flags used by the CUDA compiler during strict jenkins builds."
    FORCE )
set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS} -O3 -w CACHE STRING
    "Flags used by the C++ compiler during release builds."
    FORCE )
set(CUDA_NVCC_FLAGS_HOSTDEBUG ${CUDA_NVCC_FLAGS} -g -lineinfo -DHOST_DEBUG CACHE STRING
    "Flags used by the C++ compiler during host-debug builds."
    FORCE )
set(CUDA_NVCC_FLAGS_DEVICEDEBUG ${CUDA_NVCC_FLAGS} -G CACHE STRING
    "Flags used by the C++ compiler during device-debug builds."
    FORCE )
set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS} -g -DHOST_DEBUG -G CACHE STRING
    "Flags used by the C++ compiler during full (host+device) debug builds."
    FORCE )

## define CUDA flags when CMake >= 3.8
set(CMAKE_CUDA_FLAGS_DEVEL "${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets -Xcompiler -Wno-unknown-pragmas,-Wno-unused-function,-Wno-unused-local-typedef,-Wno-unused-private-field -O3 -lineinfo" CACHE STRING
    "Flags used by the CUDA compiler during regular development builds."
    FORCE )
set(CMAKE_CUDA_FLAGS_STRICT "${CMAKE_CUDA_FLAGS_DEVEL}" CACHE STRING
    "Flags used by the CUDA compiler during strict jenkins builds."
    FORCE )
set(CMAKE_CUDA_FLAGS_RELEASE "${CUDA_NVCC_FLAGS} -O3 -w" CACHE STRING
    "Flags used by the CUDA compiler during release builds."
    FORCE )
set(CMAKE_CUDA_FLAGS_HOSTDEBUG "${CUDA_NVCC_FLAGS} -g -lineinfo -DHOST_DEBUG" CACHE STRING
    "Flags used by the C++ compiler during host-debug builds."
    FORCE )
set(CMAKE_CUDA_FLAGS_DEVICEDEBUG "${CUDA_NVCC_FLAGS} -G" CACHE STRING
    "Flags used by the C++ compiler during device-debug builds."
    FORCE )
set(CMAKE_CUDA_FLAGS_DEBUG "${CUDA_NVCC_FLAGS} -g -DHOST_DEBUG -G" CACHE STRING
    "Flags used by the C++ compiler during full (host+device) debug builds."
    FORCE )


mark_as_advanced(CUDA_NVCC_FLAGS_DEVEL)
mark_as_advanced(CUDA_NVCC_FLAGS_STRICT)
mark_as_advanced(CUDA_NVCC_FLAGS_HOSTDEBUG)
mark_as_advanced(CUDA_NVCC_FLAGS_DEVICEDEBUG)
mark_as_advanced(CMAKE_CUDA_FLAGS_DEVEL)
mark_as_advanced(CMAKE_CUDA_FLAGS_STRICT)
mark_as_advanced(CMAKE_CUDA_FLAGS_HOSTDEBUG)
mark_as_advanced(CMAKE_CUDA_FLAGS_DEVICEDEBUG)

# make one library
#if(PLEGMA_BUILD_SHAREDLIB)
#   cuda_add_library(plegma SHARED ${PLEGMA_LIB})
#else()
#  cuda_add_library(plegma STATIC ${PLEGMA_LIB})
#endif()
target_include_directories(plegma PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_include_directories(plegma PUBLIC ${QUDA_HOME}/include/targets/cuda)
