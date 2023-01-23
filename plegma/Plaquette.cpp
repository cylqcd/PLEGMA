#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#ifdef __NVCC__
#include <cuda_profiler_api.h>
#elif defined (__HIP__)
#include <hip/hip_runtime_api.h>
#endif

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
#ifdef __NVCC__
    cudaProfilerStart();
#elif defined (__HIP__)
    hipProfilerStart();
#endif
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
#ifdef __NVCC__
    cudaProfilerStop();
#elif defined (__HIP__)
    hipProfilerStop();
#endif
  }
  finalize();
 
  return 0;
}
