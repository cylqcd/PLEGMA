// tests/test_oneD_prec.cu
//
// Precision comparison for oneD and twoD three-point correlators.
//
// Takes two pre-computed, pre-flowed propagators (one stored at float
// precision, one at double precision) together with a single shared
// double-precision flowed gauge field, contracts each with
// contractNucleonThrp_oneD and contractNucleonThrp_twoD, and writes
// both results to HDF5.  Prop difference metrics are printed so that
// the propagator-level deviation is visible alongside the correlator files.
//
// Options:
//   --prop-float   <path>   LIME file of the float-spinor flowed propagator
//   --prop-double  <path>   LIME file of the double-spinor flowed propagator
//   --gauge        <path>   LIME file of the shared double-precision flowed gauge
//   --threep-filename <pfx>   output prefix (default: "./threep")
//   --src-filename  <path>  PLEGMA source-list file (optional)
//   --corr-file-format      HDF5_FORMAT or ASCII_FORMAT
//   --corr-space            MOMENTUM_SPACE or POSITION_SPACE
//   --maxQsq        <n>     maximum |q|^2 (default 64)
//   --verbosity     <n>

#include "PLEGMA_global.h"
#include "quda.h"

#include <PLEGMA.h>
#include <PLEGMA_utils.h>

#include <mpi.h>
#include <string>
#include <vector>
#include <cmath>

using namespace plegma;
using namespace quda;

static std::vector<double> runtime;
#define TIME(fnc)                                                              \
  runtime.push_back(MPI_Wtime());                                              \
  fnc;                                                                         \
  PLEGMA_printf("TIME for " #fnc " %f sec\n", MPI_Wtime() - runtime.back()); \
  runtime.pop_back()

// ---------------------------------------------------------------------------
// Print per-column and aggregate difference metrics between two propagators.
// ---------------------------------------------------------------------------
static void compare_props(const char *label,
                           PLEGMA_Propagator<float> &ref,
                           PLEGMA_Propagator<float> &test)
{
  double norm_ref_sq  = 0.0;
  double norm_diff_sq = 0.0;

  PLEGMA_printf("[%s] per-column relative differences:\n", label);
  for (int isc = 0; isc < 12; ++isc) {
    const int spin = isc / 3, col = isc % 3;

    PLEGMA_Vector<float> va, vb, vd;
    va.absorb(ref,  spin, col);
    vb.absorb(test, spin, col);
    vd.copy(va);
    vd.add(vb, std::complex<float>(-1.0f, 0.0f));

    const double na = va.norm();
    const double nd = vd.norm();
    PLEGMA_printf("  (%d,%d): |ref|=%.6e  |test|=%.6e  rel_diff=%.6e\n",
                  spin, col, na, vb.norm(), (na > 0.0) ? nd / na : nd);

    norm_ref_sq  += na * na;
    norm_diff_sq += nd * nd;
  }

  const double norm_ref  = std::sqrt(norm_ref_sq);
  const double norm_diff = std::sqrt(norm_diff_sq);
  PLEGMA_printf("[%s] aggregate: |ref|=%.16e  |diff|=%.16e  rel=%.16e\n\n",
                label, norm_ref, norm_diff,
                (norm_ref > 0.0) ? norm_diff / norm_ref : norm_diff);
}

// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
  std::vector<std::string> listOpt = {
      "verbosity",
      "src-filename",
      "threep-filename",
      "corr-file-format",
      "corr-space",
      "maxQsq"
  };
  initializeOptions(argc, argv, false, listOpt);

  // Paths to the two input propagators and the shared gauge.
  std::string prop_float_path  = "./prop_float.lime";
  std::string prop_double_path = "./prop_double.lime";
  std::string gauge_path       = "./gauge.lime";

  HGC_options->set("prop-float",  "LIME path of float-spinor flowed propagator",  verbosity, prop_float_path);
  HGC_options->set("prop-double", "LIME path of double-spinor flowed propagator", verbosity, prop_double_path);
  HGC_options->set("gauge",       "LIME path of double-precision flowed gauge",   verbosity, gauge_path);

  initializePLEGMA();
  {
    const site &src = sourcePositions[0];
    PLEGMA_printf("\n=== Source at (%d,%d,%d,%d) ===\n",
                  src[0], src[1], src[2], src[3]);

    // ------------------------------------------------------------------
    // Load propagators (SECOND_SIDE ghost needed for twoD)
    // ------------------------------------------------------------------
    PLEGMA_Propagator<float> prop_f(BOTH, plegma::SECOND_SIDE);
    PLEGMA_printf("\n--- Loading float-spinor propagator: %s ---\n", prop_float_path.c_str());
    TIME(prop_f.readLIME(prop_float_path));

    PLEGMA_Propagator<double> prop_d(BOTH, plegma::SECOND_SIDE);
    PLEGMA_printf("\n--- Loading double-spinor propagator: %s ---\n", prop_double_path.c_str());
    TIME(prop_d.readLIME(prop_double_path));

    // ------------------------------------------------------------------
    // Propagator comparison (downcast prop_d to float for diff)
    // ------------------------------------------------------------------
    {
      PLEGMA_Propagator<float> prop_d_f(BOTH);
      prop_d_f.copy(prop_d);
      compare_props("prop_float vs prop_double", prop_f, prop_d_f);
    }

    // ------------------------------------------------------------------
    // Load flowed gauge (SECOND_SIDE ghost needed for twoD)
    // ------------------------------------------------------------------
    PLEGMA_Gauge<double> gauge_d(BOTH, plegma::SECOND_SIDE);
    PLEGMA_printf("--- Loading gauge: %s ---\n", gauge_path.c_str());
    TIME(gauge_d.readFile(gauge_path, LIME_FORMAT));

    PLEGMA_Gauge<float> gauge_f(BOTH, plegma::SECOND_SIDE);
    gauge_f.copy(gauge_d);

    // ------------------------------------------------------------------
    // oneD contractions
    // ------------------------------------------------------------------
    const std::string out_prefix = threep_filename;

    PLEGMA_printf("\n--- oneD: float-spinor prop (float accumulator) ---\n");
    PLEGMA_Correlator<float> corr_f_1D(corr_space, src, maxQsq);
    TIME(corr_f_1D.contractNucleonThrp_oneD(prop_f, prop_f, gauge_f, 0,
                                            {G1, G2, G3, G4, G5G4}));
    TIME(corr_f_1D.writeFile(out_prefix + "_float_oneD.h5", corr_file_format));

    PLEGMA_printf("\n--- oneD: double-spinor prop (double accumulator) ---\n");
    PLEGMA_Correlator<double> corr_d_1D(corr_space, src, maxQsq);
    TIME(corr_d_1D.contractNucleonThrp_oneD(prop_d, prop_d, gauge_d, 0,
                                            {G1, G2, G3, G4, G5G4}));
    TIME(corr_d_1D.writeFile(out_prefix + "_double_oneD.h5", corr_file_format));

    // ------------------------------------------------------------------
    // twoD contractions
    // ------------------------------------------------------------------
    PLEGMA_printf("\n--- twoD: float-spinor prop (float accumulator) ---\n");
    PLEGMA_Correlator<float> corr_f_2D(corr_space, src, maxQsq);
    TIME(corr_f_2D.contractNucleonThrp_twoD(prop_f, prop_f, gauge_f, 0,
                                            {G1, G2, G3, G4, G5G4}));
    TIME(corr_f_2D.writeFile(out_prefix + "_float_twoD.h5", corr_file_format));

    PLEGMA_printf("\n--- twoD: double-spinor prop (double accumulator) ---\n");
    PLEGMA_Correlator<double> corr_d_2D(corr_space, src, maxQsq);
    TIME(corr_d_2D.contractNucleonThrp_twoD(prop_d, prop_d, gauge_d, 0,
                                            {G1, G2, G3, G4, G5G4}));
    TIME(corr_d_2D.writeFile(out_prefix + "_double_twoD.h5", corr_file_format));

    PLEGMA_printf("\n=== Output files ===\n");
    PLEGMA_printf("  %s_float_oneD.h5\n",  out_prefix.c_str());
    PLEGMA_printf("  %s_double_oneD.h5\n", out_prefix.c_str());
    PLEGMA_printf("  %s_float_twoD.h5\n",  out_prefix.c_str());
    PLEGMA_printf("  %s_double_twoD.h5\n", out_prefix.c_str());
  }

  finalize();
  return 0;
}
