// Microbenchmark: compare threep_local (propagator inputs, 1 gamma) against
// oneEnd_trick / contractG5 (vector inputs, all 16 gammas).
//
// To make the comparison fair, we build the propagator by replicating the
// same vector into all 12 (spin, color) slots, so both sides see the same
// per-rank lattice volume and the same numerical Float type.
//
// Reports per-call wall time using MPI_Wtime, after a CUDA sync.
//
// Run, e.g.:
//   mpirun -n 1 ./bench_threep_vs_oet --load-gauge ../conf80.txt --rng-seed 7
//
// Build target: bench_threep_vs_oet (added in CMakeLists.txt).

#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <cuda_runtime.h>
#include <quda_api.h>

#include <mpi.h>
#include <string>
#include <vector>
#include <cstdio>

#include "higher_derivative_patterns.h"
#include "PLEGMA_QLoops_patterns.h"
#include "PLEGMA_QLoops_fast.h"
#include "PLEGMA_Correlator_patterns.h"
#include "PLEGMA_global.h"

using namespace plegma;
using namespace quda;

static std::vector<std::string> listOpt = {
    "verbosity",
    "load-gauge",
    "rng-seed",
    "corr-file-format",
};

static inline void CUDA_SYNC()
{
    qudaDeviceSynchronize();
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "[CUDA] %s\n", cudaGetErrorString(e));
        abort();
    }
}

template <typename Fn>
static double time_call(int reps, Fn &&fn)
{
    // warmup
    fn();
    CUDA_SYNC();
    MPI_Barrier(MPI_COMM_WORLD);

    double t0 = MPI_Wtime();
    for (int i = 0; i < reps; ++i)
        fn();
    CUDA_SYNC();
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    return (t1 - t0) / reps;
}

int main(int argc, char **argv)
{
    initializeOptions(argc, argv, true, listOpt);
    initializePLEGMA();

    // Load gauge (needed for QUDA init; not used directly by either kernel).
    PLEGMA_Gauge<double> gauge;
    gauge.readFile(latfile, LIME_FORMAT);
    gauge.calculatePlaq();
    initGaugeQuda(gauge, true);
    plaqQuda();

    // We benchmark in float, matching the higher-derivative pattern path.
    using F = float;

    // ---- Random source vector ----
    PLEGMA_Vector<F> vec(BOTH);
    vec.randInit(rng_seed);
    vec.stochastic_Z(4);

    // ---- Build a propagator by copying the same vector into all 12 slots ----
    PLEGMA_Propagator<F> prop(BOTH);
    for (int nu = 0; nu < N_SPINS; ++nu)
        for (int c2 = 0; c2 < N_COLS; ++c2)
            prop.absorb(vec, nu, c2);

    // ---- Pattern + correlator setup (single gamma channel: G4) ----
    ThrpPattern pat;
    pat.gamma = G4;
    pat.n_deriv = 0;
    pat.isZfac = false;

    site src{ std::array<int, N_DIMS>{{0, 0, 0, 0}} };
    int Q2max_local = 0;  // Q2=0 -> position-space-style minimal FT cost
    int totalT = HGC_totalL[DIM_T];

    PLEGMA_Correlator_patterns<F> corr(MOMENTUM_SPACE, src, Q2max_local, totalT);

    // QLoops accumulator (matches what contractG5_patterns uses).
    PLEGMA_QLoops_patterns<F> qLoops(BOTH);

    // ---- Configure metadata up front so timed calls only do contraction ----
    corr.configurePatternMetadata(make_pattern_group(pat),
                                  make_pattern_dataset(pat),
                                  make_pattern_description(pat),
                                  pat.isZfac);
    qLoops.configurePatternMetadata(make_pattern_group(pat),
                                    "loop",
                                    make_pattern_description(pat));

    std::vector<GAMMAS> gammas = {pat.gamma};
    int signProps = -1;

    const int REPS = 5;

    // ---------- 1) threep_local (1 gamma, propagator x propagator) ----------
    double t_threep = time_call(REPS, [&]() {
        threep_local<true, F, F, F>(corr, prop, prop, signProps, gammas, false);
    });

    // ---------- 2) oneEnd_trick (vector x vector, 16 gamma channels) -------
    qLoops.clearAccumBuffs();
    double t_oet = time_call(REPS, [&]() {
        qLoops.oneEnd_trick(vec, vec, F(1), /*accum=*/true);
    });

    // ---------- 3) raw contractG5 kernel only (no unload, no BLAS) ---------
    double t_g5 = time_call(REPS, [&]() {
        qLoops.contractG5(vec, vec);
    });

    // ---------- 4) optimized oneEnd_trick: device-side accumulator --------
    PLEGMA_QLoops_fast<F> qLoops_fast(BOTH);
    qLoops_fast.allocAccumDevice();
    qLoops_fast.clearAccumDevice();
    double t_fast = time_call(REPS, [&]() {
        qLoops_fast.oneEnd_trick_fast(vec, vec, F(1), /*accum=*/true);
    });

    if (HGC_fullRank == 0) {
        printf("\n========== threep_local vs oneEnd_trick benchmark ==========\n");
        printf("Float type:       %s\n", sizeof(F) == 4 ? "float" : "double");
        printf("Local volume:     %ld sites\n", (long)HGC_localVolume);
        printf("Reps:             %d (timed, plus 1 warmup)\n", REPS);
        printf("\n");
        printf("threep_local (prop, 1 gamma)              : %10.4f ms / call\n", t_threep * 1e3);
        printf("oneEnd_trick (vec,  16 gammas)            : %10.4f ms / call\n", t_oet    * 1e3);
        printf("contractG5   (vec,  16 gammas, kernel only): %10.4f ms / call\n", t_g5    * 1e3);
        printf("oneEnd_trick_fast (device accumulator)    : %10.4f ms / call\n", t_fast  * 1e3);
        printf("\n");
        printf("ratio threep_local / oneEnd_trick        : %8.3f\n", t_threep / t_oet);
        printf("ratio threep_local / contractG5          : %8.3f\n", t_threep / t_g5);
        printf("ratio oneEnd_trick / contractG5          : %8.3f  (unload + cBLAS overhead)\n",
               t_oet / t_g5);
        printf("ratio oneEnd_trick / oneEnd_trick_fast   : %8.3f  (speedup from device accumulator)\n",
               t_oet / t_fast);
        printf("ratio oneEnd_trick_fast / contractG5     : %8.3f  (residual scale-axpy overhead)\n",
               t_fast / t_g5);
        printf("============================================================\n\n");
    }

    finalize();
    return 0;
}
