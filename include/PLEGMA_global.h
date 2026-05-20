#pragma once

//======== External libraries =========//
#include <mpi.h>
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <typeinfo>
#include <typeindex>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <numeric>
#include <array>
#include <algorithm>
#include <quda.h>
#include <quda_internal.h>
#include <cublas_v2.h>
#ifdef __GNUG__ // gnu C++ compiler
#include <cxxabi.h>
#include <stdlib.h>
#endif
#include <malloc.h>
#include <new> // for availability of std::bad_alloc exception
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <assert.h>
#include <fstream>
#include <map>
#include <iterator>
#include <thread>
#include <memory>
#include <chrono>
using namespace std::chrono_literals;

//======== Constant values =========//
#define PI 3.141592653589793

#define N_COLS     3
#define N_SPINS    4

//======== Preprocessor macros =========//
#include <global/PLEGMA_macros.hpp>
#include <tune_quda.h>
#include <comm_quda.h>
// The following CCCL headers must be included before 'using namespace quda'
// to avoid double4 ambiguity: CUDA 13 defines quda::double4 = ::double4_32a,
// but CCCL (thrust/cub/cuda::std) specialize templates for ::double4.
// After 'using namespace quda', bare 'double4' becomes ambiguous between
// ::double4 and quda::double4. Pre-including ensures the specializations are
// processed before that ambiguity exists (#pragma once prevents re-processing).
#include <curand_kernel.h>
// Pre-include CCCL headers (cub/util_type.cuh pulls in cuda_fp8.h/cuda_fp6.h;
// thrust and cuda::std include further FP type specializations).
// Must be OUTSIDE __CUDACC__ guard: driver_types.h (included by CUDA runtime)
// includes cuda_fp8.h in CUDA 13, so g++-compiled .cpp files also trigger
// the double4 ambiguity unless these are pre-included before 'using namespace quda'.
#include <thrust/type_traits/is_trivially_relocatable.h>
#include <cub/util_type.cuh>
#include <cuda/std/tuple>
using namespace quda;

namespace plegma {
  //======== PLEGMA_printf, PLEGMA_error, PLEGMA_warning =========//
#include <global/PLEGMA_prints.hpp>

  //======== Enumerations =========//
#include <global/PLEGMA_enums.h>

  //======== Templated types and functions =========//
#include <global/PLEGMA_templates.h>

  //======== Some custom data struct =========//
#include <global/PLEGMA_structs.h>

  //======== Class for reading options from command line or file =========//
#include <global/PLEGMA_Options.h>

  //======== Global constants on host and device =========//
#include <global/PLEGMA_global_constants.h>


static inline void PLEGMA_memset(void *ptr, int value, size_t count){
   return qudaMemset(ptr,value,count);
}
static inline void PLEGMA_memcpy(void *dst, const void *src, size_t count,qudaMemcpyKind kind){
   return qudaMemcpy(dst,src,count,kind);
}
}
using namespace plegma; // TODO: This one shouldn't be here.. But helps avoiding missing namespace.
