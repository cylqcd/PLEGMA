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

int main(int argc, char **argv) {
  std::vector<std::string> listOpt = {"verbosity", "load-gauge", "src-filename",
                                      "corr-file-format", "maxQsq"};

  initializeOptions(argc, argv, true, listOpt);
  initializePLEGMA();

  {
    // -----------------------------
    // Gauge
    // -----------------------------
    PLEGMA_Gauge<double> gauge(BOTH);
    PLEGMA_Gauge<float> gauge_flowed(BOTH,plegma::THIRD_SIDE);

    gauge.readFile(latfile, LIME_FORMAT);
    gauge.calculatePlaq();
    gauge_flowed.copy(gauge);

    // -----------------------------
    // Dirac operator
    // -----------------------------
    updateOptions(LIGHT);

    PLEGMA_Propagator<float> prop_q(BOTH,plegma::THIRD_SIDE);
    PLEGMA_Propagator<float> SeqProp(BOTH,plegma::THIRD_SIDE);
    prop_q.readLIME("./prop_q.t24.lime");
    SeqProp.readLIME("./SeqProp.t24.lime");

    prop_q.communicateSideGhost(-1, DIR_BOTH, DO_ALL);
    site &source = sourcePositions[0];
    PLEGMA_Correlator<float> corr3_threeD(corr_space, source, maxQsq, 10 + 1);

    corr3_threeD.contractNucleonThrp_threeD(prop_q, SeqProp, gauge_flowed, 0,
                                             {G1, G2, G3, G4, G5G4});
    corr3_threeD.writeFile("./threep_threeD.t24.h5", corr_file_format);

    finalizeGaugeQuda();
  }

  finalize();
  return 0;
}
