#pragma once

//======== External libraries =========//
#include <mpi.h>
//#include <cuda.h>
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
//#include <cublas_v2.h>
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
#include <random_quda.h>
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
