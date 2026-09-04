#pragma once

//======== External libraries =========//
#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>
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
#ifdef HAVE_OPENBLAS
#include <cblas.h>
#endif
#include <quda.h>
#include <quda_internal.h>
#include <cublas_v2.h>
#include <tune_key.h>
#include <tune_quda.h>
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

namespace plegma {

using quda::Topology;
using quda::MsgHandle;

using quda::comm_abort;
using quda::comm_barrier;
using quda::comm_broadcast;
using quda::comm_hostname;
using quda::comm_rank;
using quda::comm_rank_from_coords;

using quda::getLastTuneKey;
using quda::saveTuneCache;

  //======== PLEGMA_printf, PLEGMA_error, PLEGMA_warning =========//
#include <global/PLEGMA_prints.hpp>

inline void checkCudaError()
{
  cudaError_t error = cudaGetLastError();

#ifdef HOST_DEBUG
  if (error == cudaSuccess) {
    error = cudaDeviceSynchronize();
  }
#endif

  if (error != cudaSuccess) {
    std::fprintf(
        stderr,
        "CUDA error: %s\n",
        cudaGetErrorString(error)
    );
    quda::comm_abort(1);
  }
}

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


}
using namespace plegma; // TODO: This one shouldn't be here.. But helps avoiding missing namespace.
