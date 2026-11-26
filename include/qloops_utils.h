#pragma once

#include <cuda_runtime.h>
#include <quda_api.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// CUDA error check after a kernel / device operation.
// ---------------------------------------------------------------------------
static inline void CUDA_SYNC(const char* where)
{
    qudaDeviceSynchronize();
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        fprintf(stderr, "[CUDA] error after %s: %s (%d)\n", where, cudaGetErrorString(e), (int) e);
        fflush(stderr);
        abort();
    }
}

// ---------------------------------------------------------------------------
// PRIMME preset-method lookup (only compiled when PRIMME is available).
// ---------------------------------------------------------------------------
#ifdef HAVE_PRIMME
static constexpr int Nmethods = 16;
static const char* const methods[] = {
    "PRIMME_DEFAULT_METHOD",
    "PRIMME_DYNAMIC",
    "PRIMME_DEFAULT_MIN_TIME",
    "PRIMME_DEFAULT_MIN_MATVECS",
    "PRIMME_Arnoldi",
    "PRIMME_GD",
    "PRIMME_GD_plusK",
    "PRIMME_GD_Olsen_plusK",
    "PRIMME_JD_Olsen_plusK",
    "PRIMME_RQI",
    "PRIMME_JDQR",
    "PRIMME_JDQMR",
    "PRIMME_JDQMR_ETol",
    "PRIMME_STEEPEST_DESCENT",
    "PRIMME_LOBPCG_OrthoBasis",
    "PRIMME_LOBPCG_OrthoBasis_Window"
};
static primme_preset_method getMethod(std::string str)
{
    for (int i = 0; i < Nmethods; i++)
        if (str == methods[i])
            return static_cast<primme_preset_method>(i);
    PLEGMA_warning("Method provided %s is not in PRIMME, available methods are", str.c_str());
    for (int i = 0; i < Nmethods; i++)
        PLEGMA_printf(methods[i]);
    PLEGMA_exit(-1);
    return static_cast<primme_preset_method>(0);
}
#endif
