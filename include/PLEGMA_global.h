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

//======== Constant values =========//
#define PI 3.141592653589793

#define N_COLS     3
#define N_SPINS    4

//======== Preprocessor macros =========//
#include <global/PLEGMA_macros.hpp>

namespace plegma {
  //======== PLEGMA_printf, PLEGMA_error, PLEGMA_warning =========//
#include <global/PLEGMA_prints.hpp>

  //======== Enumerations =========//
#include <global/PLEGMA_enums.h>

  //======== Templated types and functions =========//
#include <global/PLEGMA_templates.h>

  //======== Some custom data struct =========//
#include <global/PLEGMA_structs.h>

  //======== Global constants on host and device =========//
#include <global/PLEGMA_global_constants.h>


}
using namespace plegma; // TODO: This one shouldn't be here.. But helps avoiding missing namespace.
