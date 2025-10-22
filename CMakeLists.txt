# basic setup for cmake
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 17)
include(CheckIncludeFileCXX)
include(CheckSymbolExists)

if(${CMAKE_VERSION} VERSION_GREATER "3.12.0") 
  cmake_policy(SET CMP0074 NEW)
endif()

# disable in source builds
# this is only a temporary fix, but for now we need it as cmake will
# otherwise overwrite the existing makefiles
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
# add a directory for cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
enable_language(C)
enable_language(CXX)

project("PLEGMA" VERSION 0.0.1 LANGUAGES)

# TODO: add authomatic download and compilation of QUDA
set(QUDA_HOME "" CACHE PATH "path to QUDA")
if("${QUDA_HOME}" STREQUAL "")
  message( FATAL_ERROR "QUDA_HOME must be defined" )
endif()
FIND_LIBRARY(QUDA_LIB quda ${QUDA_HOME}/lib)
include_directories(SYSTEM ${QUDA_HOME}/include)

set(QUDA_SRC "" CACHE PATH "path to QUDA src directory")
if("${QUDA_SRC}" STREQUAL "")
  message( FATAL_ERROR "QUDA_SRC must be defined" )
endif()

find_package(Git)
if(GIT_FOUND)
  execute_process(COMMAND ${GIT_EXECUTABLE} show
                  WORKING_DIRECTORY ${QUDA_SRC}
                  RESULT_VARIABLE IS_GIT_REPOSIITORY
                  OUTPUT_QUIET ERROR_QUIET)
  if(${IS_GIT_REPOSIITORY} EQUAL 0)
    execute_process(COMMAND ${GIT_EXECUTABLE} merge-base --is-ancestor b08233a7d1cee0145e56751ae04c5542ce98b3b5 HEAD
                    WORKING_DIRECTORY ${QUDA_SRC}
		    RESULT_VARIABLE QUDA_INCLUDES_IT
                    OUTPUT_VARIABLE GITTAG
		    OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(${QUDA_INCLUDES_IT} EQUAL 0)
      add_compile_definitions(QUDA_INCLUDES_COMMIT_b08233a)
    endif()

    execute_process(COMMAND ${GIT_EXECUTABLE} merge-base --is-ancestor 775a033a84b4f085e8cf68812b89f989e6a5a816 HEAD
                    WORKING_DIRECTORY ${QUDA_SRC}
		    RESULT_VARIABLE QUDA_INCLUDES_IT
                    OUTPUT_VARIABLE GITTAG
		    OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(${QUDA_INCLUDES_IT} EQUAL 0)
      add_compile_definitions(QUDA_INCLUDES_COMMIT_775a033)
    endif()

    execute_process(COMMAND ${GIT_EXECUTABLE} merge-base --is-ancestor 55782743d1ee8d99af21e1038be11a7023650d7d HEAD
    	            WORKING_DIRECTORY ${QUDA_SRC}
		    RESULT_VARIABLE QUDA_INCLUDES_IT
                    OUTPUT_VARIABLE GITTAG
		    OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(${QUDA_INCLUDES_IT} EQUAL 0)
      add_compile_definitions(QUDA_INCLUDES_COMMIT_55782743)
    endif()


    execute_process(COMMAND ${GIT_EXECUTABLE} merge-base --is-ancestor 1dec1db7a57e4fe00b14e04ba2088ca5039f8760 HEAD
                    WORKING_DIRECTORY ${QUDA_SRC}
		    RESULT_VARIABLE QUDA_INCLUDES_IT
                    OUTPUT_VARIABLE GITTAG
		    OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(${QUDA_INCLUDES_IT} EQUAL 0)
      add_compile_definitions(QUDA_INCLUDES_COMMIT_1dec1db)
    endif()
  endif()
endif(GIT_FOUND)

include(ExternalProject)

find_package(Boost)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
else()
  message(WARNING
    "Boost was not found; we will download what needed.")
  set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)
  ExternalProject_Add(BoostPP
    GIT_REPOSITORY https://github.com/boostorg/preprocessor
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    LOG_DOWNLOAD ON
    EXCLUDE_FROM_ALL TRUE
  )
  ExternalProject_Get_Property(BoostPP DOWNLOAD_DIR)
  include_directories(${DOWNLOAD_DIR}/BoostPP/include)
endif()

set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS DEVEL RELEASE STRICT DEBUG HOSTDEBUG DEVICEDEBUG)

# using dirs in LD_LIBRARY_PATh for searching for libraries
string(REPLACE ":" ";" LIBRARY_DIRS $ENV{LD_LIBRARY_PATH})

set(DEFAULT_GPU_ARCH sm_60)
set(GPU_ARCH ${DEFAULT_GPU_ARCH} CACHE STRING "set the GPU architecture (sm_20, sm_21, sm_30, sm_35, sm_37, sm_50, sm_52, sm_60, sm_70, sm_80)")
set_property(CACHE GPU_ARCH PROPERTY STRINGS sm_20 sm_21 sm_30 sm_35 sm_37 sm_50 sm_52 sm_60 sm_70 sm_80)

# max threads
set(PLEGMA_MAX_THREADS 256 CACHE INT "maximum number of threads per block in tuner")
mark_as_advanced(PLEGMA_MAX_THREADS)
add_definitions(-DMAX_THREADS=${PLEGMA_MAX_THREADS})

# perform tuning based on quda
set(PLEGMA_NO_TUNING FALSE CACHE BOOL "force not to perform kernel tuning in PLEGMA")
mark_as_advanced(PLEGMA_NO_TUNING)
if(PLEGMA_NO_TUNING)
  add_definitions(-DPLEGMA_NO_TUNING)
else()
  
endif()

# Use CUDA textures
set(PLEGMA_TEXTURE TRUE CACHE BOOL "Wheater to use or not CUDA textures")
mark_as_advanced(PLEGMA_TEXTURE)
if(PLEGMA_TEXTURE)
  add_definitions(-DPLEGMA_TEXTURE)
else()
  
endif()

# compile contractions for nucleon fix sink 3pf
set(PLEGMA_NUCLEON_3PF_FIX_SINK FALSE CACHE BOOL "compile contractions for nucleon 3pf fix sink method")
if(PLEGMA_NUCLEON_3PF_FIX_SINK)
  add_definitions(-DPLEGMA_NUCLEON_3PF_FIX_SINK)
endif()

# compile contractions for all baryons
set(PLEGMA_LIGHT_BARYONS FALSE CACHE BOOL "compile contractions for all light baryons")
if(PLEGMA_LIGHT_BARYONS)
  add_definitions(-DPLEGMA_LIGHT_BARYONS)
endif()
set(PLEGMA_SCATTERING_CONTRACTIONS FALSE CACHE BOOL "compile contractions for pi-N scattering I=3/2")
if (PLEGMA_SCATTERING_CONTRACTIONS)
  add_definitions(-DPLEGMA_SCATTERING_CONTRACTIONS)
endif()
set(PLEGMA_UDSC_BARYONS FALSE CACHE BOOL "compile contractions for all u,d,s and c baryons")
if(PLEGMA_UDSC_BARYONS)
  add_definitions(-DPLEGMA_UDSC_BARYONS)
endif()

# flag to allow for pinned memory ghost buffers
set(PLEGMA_PINNED_GHOST_BUFFERS FALSE CACHE BOOL "Enable pinned ghost buffer exchanged. Experimental use it if you are fully aware about the issues")
if(PLEGMA_PINNED_GHOST_BUFFERS)
  add_definitions(-DHAVE_PINNED_GHOST)
endif()

set(PLEGMA_EIGENSOLVER FALSE CACHE BOOL "Build eigensolvers interface. Options (ARPACK, PRIMME, QUDAEIG)")
if(PLEGMA_EIGENSOLVER)
  add_definitions(-DHAVE_EIGENSOLVER)
  #ARPACK || PRIMME
  set(PLEGMA_ARPACK OFF CACHE BOOL "build arpack interface")
  set(PLEGMA_PRIMME OFF CACHE BOOL "build PRIMME interface")
  set(PLEGMA_QUDAEIG OFF CACHE BOOL "build interface for QUDA eigensolver")

  if(PLEGMA_ARPACK AND PLEGMA_PRIMME)
    message(SEND_ERROR "Please specify either ARPACK or PRIMME eigensolver, not both")
  endif()
  if(PLEGMA_ARPACK AND PLEGMA_QUDAEIG)
    message(SEND_ERROR "Please specify either ARPACK or QUDAEIG eigensolver, not both")
  endif()
  if(PLEGMA_QUDAEIG AND PLEGMA_PRIMME)
    message(SEND_ERROR "Please specify either ARPACK or PRIMME eigensolver, not both")
  endif()

  if(PLEGMA_ARPACK)
    add_definitions(-DHAVE_ARPACK)
    set(PLEGMA_ARPACKHOME "" CACHE PATH "path to ARPACK")
  endif(PLEGMA_ARPACK)

  if(PLEGMA_PRIMME)
    add_definitions(-DHAVE_PRIMME)
    set(PLEGMA_PRIMMEHOME "" CACHE PATH "path to PRIMME")
  endif(PLEGMA_PRIMME)

  if(PLEGMA_QUDAEIG)
     add_definitions(-DQUDAEIG)
  endif(PLEGMA_QUDAEIG)
endif(PLEGMA_EIGENSOLVER)

set(PLEGMA_2FLAVMG ON CACHE BOOL "compile 2-flavour MG routines. This requires modification of QUDA to add extra preconditioner pointers")
if(PLEGMA_2FLAVMG)
  add_definitions(-DPLEGMA_2FLAVMG)
endif()

# Locations for GSL HDF5 LIME
#set(PLEGMA_GSLHOME "" CACHE PATH "path to GSL")
set(PLEGMA_LIMEHOME "" CACHE PATH "path to LIME")
set(PLEGMA_HDF5HOME "" CACHE PATH "path to HDF5")
set(PLEGMA_HDF5_THREAD_SAFE OFF CACHE BOOL "Is HDF5 compiled in a thread-safe mode?")
    
# Switch between MKL and OpenBlas BLAS
set(PLEGMA_MKL OFF CACHE BOOL "build PLEGMA with MKL")
set(PLEGMA_OPENBLAS OFF CACHE BOOL "build PLEGMA with OPENBLAS")

if(PLEGMA_MKL AND PLEGMA_OPENBLAS)
  message(SEND_ERROR "Please specify either MKL or OpenBLAS libraries, not both")
endif()

# Locations for OPENBLAS or MKL
if(${PLEGMA_OPENBLAS})
  set(PLEGMA_OPENBLASHOME "" CACHE PATH "path to OPENBLAS source")
  set(PLEGMA_OPENBLAS_LIBDIR "" CACHE PATH "path to OPENBLAS lib")
endif()

if(${PLEGMA_MKL})
  set(PLEGMA_MKL_LIBDIR "${MKLROOT}/lib/intel64" CACHE PATH "path to MKL library dir")
  set(PLEGMA_MKL_INCDIR "${MKLROOT}/include" CACHE PATH "path to MKL include dir")
endif()

#######################################################################
# build PLEGMA as shared library
set(PLEGMA_BUILD_SHAREDLIB OFF CACHE BOOL "build PLEGMA as shared library")
mark_as_advanced(PLEGMA_BUILD_SHAREDLIB)
if(${PLEGMA_BUILD_SHAREDLIB})
  if( NOT( ${QUDA_LIB} MATCHES "[.]so$") )
    message( FATAL_ERROR "QUDA_LIB must be a shared library " ${QUDA_LIB})
  endif()
endif()

#######################################################################
# options that are not exposed at all because only one option exists
set(CPU_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})

# Compile and run tests
set(PLEGMA_RUN_TESTS FALSE CACHE BOOL "Compile and run tests with googletest")
mark_as_advanced(PLEGMA_RUN_TESTS)


#######################################################################
# everything below here is processing the setup
#######################################################################
# we need to check for some packages
find_package(PythonInterp)

if(${CMAKE_VERSION} VERSION_GREATER 3.7.99)
  find_package(CUDAWrapper)
  set(USING_CUDA_LANG_SUPPORT True)
  set(CMAKE_CUDA_STANDARD 14)
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

# We need threads
find_package(Threads REQUIRED)

# do all the build definitions
#  
#-C.K. Modified here to take OpenBlas and MKL switches into account  
if(${PLEGMA_OPENBLAS})
  add_definitions(-DHAVE_OPENBLAS)
  FIND_LIBRARY(PLEGMA_OPENBLAS_LIB openblas ${LIBRARY_DIRS} ${PLEGMA_OPENBLAS_LIBDIR})
  FIND_LIBRARY(PLEGMA_OPENBLAS_GOMP gomp ${LIBRARY_DIRS})
  include_directories(SYSTEM ${PLEGMA_OPENBLASHOME}/include)
endif()

if(${PLEGMA_MKL})
  add_definitions(-DHAVE_MKL)
  FIND_LIBRARY(PLEGMA_MKL_LIB_GFORTRAN gfortran /usr/lib64)
  FIND_LIBRARY(PLEGMA_MKL_LIB_IOMP5 iomp5 /opt/intel/lib/intel64)
  FIND_LIBRARY(PLEGMA_MKL_LIB_LP64 mkl_intel_lp64 ${PLEGMA_MKL_LIBDIR})
  FIND_LIBRARY(PLEGMA_MKL_LIB_THREAD mkl_intel_thread ${PLEGMA_MKL_LIBDIR})
  FIND_LIBRARY(PLEGMA_MKL_LIB_CORE mkl_core ${PLEGMA_MKL_LIBDIR})  
  include_directories(SYSTEM ${PLEGMA_MKL_INCDIR})
endif()

if(PLEGMA_EIGENSOLVER)
  if(PLEGMA_ARPACK)
    FIND_LIBRARY(PLEGMA_GFORTRAN_LIB gfortran ${LIBRARY_DIRS})  
    FIND_LIBRARY(PLEGMA_MPIFH_LIB mpi_mpifh ${LIBRARY_DIRS})  
    FIND_LIBRARY(PLEGMA_PARPACK_LIB parpack  ${PLEGMA_ARPACKHOME}/lib)
    include_directories(SYSTEM ${PLEGMA_ARPACKHOME}/PARPACK/SRC/MPI)
    include_directories(SYSTEM ${PLEGMA_ARPACKHOME}/PARPACK/UTIL/MPI)

    FIND_LIBRARY(PLEGMA_ARPACK_LIB arpack ${PLEGMA_ARPACKHOME}/lib)
    include_directories(SYSTEM ${PLEGMA_ARPACKHOME}/SRC)
    include_directories(SYSTEM ${PLEGMA_ARPACKHOME}/UTIL)
  endif(PLEGMA_ARPACK)

  if(PLEGMA_PRIMME)
    FIND_LIBRARY(PLEGMA_GFORTRAN_LIB gfortran ${LIBRARY_DIRS})
    FIND_LIBRARY(PLEGMA_PRIMME_LIB primme ${PLEGMA_PRIMMEHOME}/lib)
    include_directories(SYSTEM ${PLEGMA_PRIMMEHOME}/include)
  endif(PLEGMA_PRIMME)
endif(PLEGMA_EIGENSOLVER)

#add_definitions(-DHAVE_GSL)
#FIND_LIBRARY(PLEGMA_GSL_LIB gsl ${PLEGMA_GSLHOME}/lib)
#include_directories(SYSTEM ${PLEGMA_GSLHOME}/include)

# if PLEGMA_HDF5HOME has been specified, give this as a hint to
# find_package via the *environment* variable HDF5_ROOT
if(PLEGMA_HDF5HOME)
  set(HDF5_ROOT ${PLEGMA_HDF5HOME})
endif()
# we want parallel HDF5
set(HDF5_PREFER_PARALLEL TRUE)
find_package(HDF5)
if(HDF5_FOUND)
  add_definitions(-DH5_HAVE_PARALLEL)
  include_directories(${HDF5_INCLUDE_DIRS})
  FIND_LIBRARY(PLEGMA_Z_LIB z ${LIBRARY_DIRS})	
#link_libraries(${HDF5_C_LIBRARY_hdf5})
  message(STATUS "HDF5 version ${HDF5_VERSION} found, include dir: ${HDF5_INCLUDE_DIRS}, libraries ${HDF5_C_LIBRARY_hdf5}")
 if(NOT HDF5_IS_PARALLEL)
   message(FATAL_ERROR "HDF5 must be parallel")
 endif()
else()
  message(FATAL_ERROR "HDF5 could not be located, did you try specifying -DPLEGMA_HDF5HOME?")
endif()

if(PLEGMA_HDF5_THREAD_SAFE)
  add_definitions(-DHDF5_THREAD_SAFE)
endif()

add_definitions(-DHAVE_LIME)
FIND_LIBRARY(PLEGMA_LIME_LIB lime ${PLEGMA_LIMEHOME}/lib)
include_directories(SYSTEM ${PLEGMA_LIMEHOME}/include)

LIST(APPEND CUDA_LIBS ${CUDA_cufft_LIBRARY} ${CUDA_curand_LIBRARY})
LIST(APPEND CUDA_LIBS ${CUDA_cublas_LIBRARY})
LIST(APPEND CUDA_LIBS ${CUDA_nvrtc_LIBRARY})
LIST(APPEND CUDA_LIBS ${CUDA_nvToolsExt_LIBRARY})

find_package(LibDL)
LIST(APPEND CUDA_LIBS ${LIBDL_LIBRARIES})

add_definitions(-DMULTI_GPU)

set(MPI_HOME_MANUAL "" CACHE PATH "Path to MPI for manual setting")
mark_as_advanced(MPI_HOME_MANUAL)

if(NOT "${MPI_HOME_MANUAL}" STREQUAL "")
    include_directories(SYSTEM ${MPI_HOME_MANUAL}/include) 
    FIND_LIBRARY(MPICH_LIBRARY_MANUAL mpich ${MPI_HOME_MANUAL}/lib)
    FIND_LIBRARY(MPI_LIBRARY_MANUAL mpi ${MPI_HOME_MANUAL}/lib)
    if( NOT ( "${MPI_LIBRARY_MANUAL}" STREQUAL "") )
      #message(FATAL_ERROR "${MPI_LIBRARY_MANUAL}")
      set(MPI_CXX_LIBRARIES ${MPI_LIBRARY_MANUAL}) 
    elseif( NOT ( "${MPICH_LIBRARY_MANUAL}" STREQUAL "" ) )
      #message(FATAL_ERROR "2")
      set(MPI_CXX_LIBRARIES ${MPICH_LIBRARY_MANUAL})
    else()
      message(FATAL_ERROR "Could not locate any MPI library")
    endif()
else()
    find_package(MPI)
    if( MPI_FOUND )
      include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
      if("${MPI_CXX_INCLUDE_PATH}" STREQUAL "")
        message(FATAL_ERROR "MPI_CXX_INCLUDE_PATH could not be specified, please specify MPI_HOME_MANUAL manually")
      endif()
   else()
      message(FATAL_ERROR "Could not locate any MPI library")
    endif()
endif()

add_definitions(-DMPI_COMMS)


# COMPILER OPTIONS and BUILD types
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
include_directories(include)
include_directories(utils)
include_directories(lib)
include_directories(lib/kernels)


# GPU ARCH
STRING(REGEX REPLACE sm_ "" COMP_CAP ${GPU_ARCH})
SET(COMP_CAP "${COMP_CAP}0")
add_definitions(-D__COMPUTE_CAPABILITY__=${COMP_CAP})


# NVCC FLAGS independet off build type
if(NOT USING_CUDA_LANG_SUPPORT)
  set(CUDA_NVCC_FLAGS "-std c++11 -arch=${GPU_ARCH} -ftz=true -prec-div=false -prec-sqrt=false")
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

#define CXX FLAGS
set(CMAKE_CXX_FLAGS_DEVEL  "${OpenMP_CXX_FLAGS} -O3 -Wall ${CLANG_FORCE_COLOR}" CACHE STRING
"Flags used by the C++ compiler during regular development builds.")
set(CMAKE_CXX_FLAGS_STRICT  "${OpenMP_CXX_FLAGS} -O3 -Wall -Werror ${CLANG_NOERROR}" CACHE STRING
"Flags used by the C++ compiler during strict jenkins builds.")
set(CMAKE_CXX_FLAGS_RELEASE "${OpenMP_CXX_FLAGS} -O3 -w" CACHE STRING
    "Flags used by the C++ compiler during release builds.")
set(CMAKE_CXX_FLAGS_HOSTDEBUG "${OpenMP_CXX_FLAGS} -Wall -Wno-unknown-pragmas -g -fno-inline -DHOST_DEBUG ${CLANG_FORCE_COLOR}" CACHE STRING
    "Flags used by the C++ compiler during host-debug builds.")
set(CMAKE_CXX_FLAGS_DEVICEDEBUG "${OpenMP_CXX_FLAGS} -Wall -Wno-unknown-pragmas ${CLANG_FORCE_COLOR}" CACHE STRING
    "Flags used by the C++ compiler during device-debug builds.")
set(CMAKE_CXX_FLAGS_DEBUG "${OpenMP_CXX_FLAGS} -Wall -Wno-unknown-pragmas -g -fno-inline -DHOST_DEBUG ${CLANG_FORCE_COLOR}" CACHE STRING
    "Flags used by the C++ compiler during full (host+device) debug builds.")

OPTION(USE_OpenMP "Use OpenMP" ON)
IF(USE_OpenMP)
    FIND_PACKAGE(OpenMP)
    IF( OPENMP_FOUND )
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
   	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler ${OpenMP_CXX_FLAGS}")
    ENDIF()
ENDIF()





#define C FLAGS
set(CMAKE_C_FLAGS_DEVEL "-Wall -O3"  CACHE STRING
    "Flags used by the C compiler during regular development builds.")
set(CMAKE_C_FLAGS_STRICT "-Wall -O3 -Werror -Wno-error=unused-private-field"  CACHE STRING
    "Flags used by the C compiler during strict jenkins builds.")
set(CMAKE_C_FLAGS_RELEASE "-Wall -O3 -w" CACHE STRING
    "Flags used by the C compiler during release builds.")
set(CMAKE_C_FLAGS_HOSTDEBUG "-Wall -Wno-unknown-pragmas -g -fno-inline -DHOST_DEBUG" CACHE STRING
    "Flags used by the C compiler during host-debug builds.")
set(CMAKE_C_FLAGS_DEVICEDEBUG "-Wall" CACHE STRING
"Flags used by the C compiler during device-debug builds.")
set(CMAKE_C_FLAGS_DEBUG "-Wall -g -fno-inline -DHOST_DEBUG" CACHE STRING
    "Flags used by the C compiler during full (host+device) debug builds.")

#define FORTRAN FLAGS
set(CMAKE_F_FLAGS -std=c99 CACHE STRING "Fortran Flags")

set(PLEGMA_ALIGNMENT 64 CACHE STRING "Alignment size for allocations done with mealign, default = 64 bytes")
mark_as_advanced(PLEGMA_ALIGNMENT)
add_definitions(-DPLEGMA_ALIGNMENT=${PLEGMA_ALIGNMENT})

check_symbol_exists(memalign malloc.h HAVE_MEMALIGN)
if(HAVE_MEMALIGN)
  add_definitions(-DPLEGMA_HAVE_MEMALIGN)
else()
  message(WARNING "Could not find memalign, alignment PLEGMA_ALIGNMENT=${PLEGMA_ALIGNMENT} will be ignored")
endif()

# make the compiler flags an advanced option for all user defined build types (cmake defined build types are advanced by default )
mark_as_advanced(CMAKE_CXX_FLAGS_DEVEL)
mark_as_advanced(CMAKE_CXX_FLAGS_STRICT)
mark_as_advanced(CMAKE_CXX_FLAGS_HOSTDEBUG)
mark_as_advanced(CMAKE_CXX_FLAGS_DEVICEDEBUG)

mark_as_advanced(CUDA_NVCC_FLAGS_DEVEL)
mark_as_advanced(CUDA_NVCC_FLAGS_STRICT)
mark_as_advanced(CUDA_NVCC_FLAGS_HOSTDEBUG)
mark_as_advanced(CUDA_NVCC_FLAGS_DEVICEDEBUG)
mark_as_advanced(CMAKE_CUDA_FLAGS_DEVEL)
mark_as_advanced(CMAKE_CUDA_FLAGS_STRICT)
mark_as_advanced(CMAKE_CUDA_FLAGS_HOSTDEBUG)
mark_as_advanced(CMAKE_CUDA_FLAGS_DEVICEDEBUG)

mark_as_advanced(CMAKE_C_FLAGS_DEVEL)
mark_as_advanced(CMAKE_C_FLAGS_STRICT)
mark_as_advanced(CMAKE_C_FLAGS_HOSTDEBUG)
mark_as_advanced(CMAKE_C_FLAGS_DEVICEDEBUG)

mark_as_advanced(CMAKE_F_FLAGS)

### add tests and quda library
add_subdirectory(lib)
add_subdirectory(tests)
### add PLEGMA directory
add_subdirectory(utils)
add_subdirectory(plegma)

