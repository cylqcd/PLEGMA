#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <cuda_profiler_api.h>

using namespace plegma;
using namespace quda;

int main(int argc, char **argv)
{
  initializeOptions(argc, argv); // Put list of options later
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//

  //=========================================================================================================//
  initializePLEGMA();

  // Allocation done on BOTH, DEVICE and HOST
  {
    cudaProfilerStart();
    PLEGMA_Gauge<double> gauge(BOTH);
    // Reading from Lime file and loading to device
    gauge.readFile(latfile, LIME_FORMAT);

    // Computing plaquette on device in three different way for crosschecking
    gauge.calculatePlaq();
    gauge.calculatePlaqCorners();
    gauge.calculatePlaqShifts();
    gauge.calculatePlaqClover();
    gauge.calculatePlaqStaples();

    // Loading to QUDA and computing plaquette also there
    initGaugeQuda(gauge, false, QUDA_SU3_LINKS);
    plaqQuda();
    cudaProfilerStop();
  }
  finalize();
 
  return 0;
}
