#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

int main(int argc, char **argv)
{
  initializeOptions(argc, argv); // Put list of options later
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//

  //=========================================================================================================//
  initializePLEGMA();

  // Allocation done on BOTH, DEVICE and HOST
  PLEGMA_Gauge<double> gauge(BOTH);
  // Reading from Lime file and loading to device
  gauge.readFromLime(latfile.c_str());

  // Computing plaquette on device in three different way for crosschecking
  gauge.calculatePlaq();
  gauge.calculatePlaqCorners();
  gauge.calculatePlaqShifts();

  // Loading to QUDA and computing plaquette also there
  initGaugeQuda(gauge, false, QUDA_SU3_LINKS);
  plaqQuda();
  
  finalize();
 
  return 0;
}
