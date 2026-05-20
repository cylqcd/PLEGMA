#include "PLEGMA_fiveD.h"
#include "PLEGMA_global.h"
#include "PLEGMA_oneD.cuh"
#include "QUDA_wilson_flow.h"
#include "enum_quda.h"
#include "quda.h"

#include <PLEGMA.h>
#include <PLEGMA_utils.h>

#include <mpi.h>
#include <string>
#include <vector>

using namespace plegma;
using namespace quda;

// Simple scoped timer
static std::vector<double> runtime;
#define TIME(fnc)                                                              \
  runtime.push_back(MPI_Wtime());                                              \
  fnc;                                                                         \
  PLEGMA_printf("TIME for " #fnc " %f sec\n", MPI_Wtime() - runtime.back());   \
  runtime.pop_back()

extern quda::GaugeField *gaugePrecise;
extern quda::GaugeField *gaugeSmeared;

static double rel_diff(PLEGMA_Vector<double> &a, PLEGMA_Vector<double> &b) {
  PLEGMA_Vector<double> tmp(BOTH);
  tmp.copy(a, DEVICE);
  tmp.add(b, std::complex<double>(-1.0, 0.0)); // tmp = a - b

  double na = a.norm();
  double nd = tmp.norm();
  return (na > 0.0) ? nd / na : nd;
}

static double rel_diff_prop(PLEGMA_Propagator<double> &a,
                            PLEGMA_Propagator<double> &b) {
  PLEGMA_Propagator<double> tmp(BOTH);
  tmp.copy(a, DEVICE);
  tmp.add(b, std::complex<double>(-1.0, 0.0)); // tmp = a - b

  double na = a.norm();
  double nd = tmp.norm();
  return (na > 0.0) ? nd / na : nd;
}

static void print_norm_prop(const char *name, PLEGMA_Propagator<double> &p) {
  PLEGMA_printf("%s norm = %.16e\n", name, p.norm());
}

static void print_norm(const char *name, PLEGMA_Vector<double> &v) {
  PLEGMA_printf("%s norm = %.16e\n", name, v.norm());
}

int main(int argc, char **argv) {
  std::vector<std::string> listOpt = {"verbosity", "load-gauge", "src-filename",
                                      "corr-file-format", "maxQsq"};

  initializeOptions(argc, argv, true, listOpt);
  initializePLEGMA();

  {
    const int n_steps = 10;
    const double epsilon = 0.01;
    const double t0_start = 0.0;
    const bool antiperiodic = false;

    // -----------------------------
    // Gauge
    // -----------------------------
    PLEGMA_Gauge<double> gauge(BOTH);
    PLEGMA_Gauge<float> gauge_flowed(BOTH,plegma::THIRD_SIDE);

    gauge.readFile(latfile, LIME_FORMAT);
    gauge.calculatePlaq();
    gauge_flowed.copy(gauge);

    // initGaugeQuda(gauge, antiperiodic);
    // plaqQuda();

    PLEGMA_printf("After initGaugeQuda: gaugePrecise=%p gaugeSmeared=%p\n",
                  (void *)gaugePrecise, (void *)gaugeSmeared);

    // -----------------------------
    // Dirac operator
    // -----------------------------
    updateOptions(LIGHT);

    // const QudaDslashType test_dslash = QUDA_TWISTED_MASS_DSLASH;

    // QUDA_dirac *D = new QUDA_dirac(test_dslash);
    // D->print();

    // Separate invert param for fermionFlow_PLEGMA
    // QudaInvertParam inv_flow = newQudaInvertParam();
    // setInvertParam(inv_flow);
    // inv_flow.dslash_type = test_dslash;

    // -----------------------------
    // One stochastic source
    // -----------------------------
    // PLEGMA_Vector<float> src, out;
    // PLEGMA_Vector<float> src_flowed(BOTH);
    // PLEGMA_Vector<float> out_before(BOTH);
    // PLEGMA_Vector<float> out_after(BOTH);
    // PLEGMA_Vector<float> out_reload(BOTH);
    PLEGMA_Propagator<double> prop_q(BOTH,plegma::THIRD_SIDE);
    PLEGMA_Propagator<double> SeqProp(BOTH,plegma::THIRD_SIDE);
    // prop_q.readLIME("/onyx/qdata/ckummer/toBolun/results/prop1f.lime");
    // SeqProp.readLIME("/onyx/qdata/ckummer/toBolun/results/prop1f.lime");
    prop_q.readLIME("/onyx/qdata/ckummer/toBolun_twoProp/propL.lime");
    SeqProp.readLIME("/onyx/qdata/ckummer/toBolun_twoProp/propR.lime");

    // src.randInit(rng_seed);
    // src.stochastic_Z(4);
    // src.communicateSideGhost(-1, DIR_BOTH, DO_ALL);
    // src.writeHDF5("src");

    // for (int isc = 0; isc < 12; ++isc) {
    //   prop_q.absorb(src, isc / 3, isc % 3);
    // }

    prop_q.communicateSideGhost(-1, DIR_BOTH, DO_ALL);
    SeqProp.communicateSideGhost(-1, DIR_BOTH, DO_ALL);
    site &source = sourcePositions[0];
    PLEGMA_Correlator<float> corr3(corr_space, source, maxQsq);
    PLEGMA_Correlator<float> corr3_oneD(corr_space, source, maxQsq);
    PLEGMA_Correlator<float> corr3_twoD(corr_space, source, maxQsq);
    //PLEGMA_Correlator<float> corr3_threeD(corr_space, source, maxQsq);

    // TIME(corr3.contractNucleonThrp_local(prop_q, SeqProp, 0,
    //                                          {G1, G2, G3, G4, G5G4}));
    // TIME(corr3.writeFile("./threep_local.t24.h5", corr_file_format));
    TIME(contractNucleonThrp_fiveD_patterns(
        prop_q, SeqProp, gauge_flowed, corr_space, source, maxQsq,
        "threep_patterns.LR.mutiGPU.h5"));
    // TIME(corr3_oneD.contractNucleonThrp_oneD(prop_q, SeqProp, gauge_flowed, 0,
    //                                          {G1, G2, G3, G4, G5G4}));
    // TIME(corr3_oneD.writeFile("./threep_oneD.t24.h5", corr_file_format));
    // TIME(corr3_twoD.contractNucleonThrp_twoD(prop_q, SeqProp, gauge_flowed, 0,
    //                                          {G1, G2, G3, G4, G5G4}));
    // TIME(corr3_twoD.writeFile("./threep_twoD.t24.h5", corr_file_format));
    //TIME(corr3_threeD.contractNucleonThrp_threeD(prop_q, SeqProp, gauge_flowed, 0,
    //                                         {G1, G2, G3, G4, G5G4}));
    //TIME(corr3_threeD.writeFile("./threep_threeD.t24.h5", corr_file_format));

    finalizeGaugeQuda();
  }

  finalize();
  return 0;
}
