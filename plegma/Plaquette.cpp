#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

int main(int argc, char **argv)
{
  initialize(argc, argv);

  // Allocation done on BOTH, DEVICE and HOST
  PLEGMA_Gauge<double> pGauge(BOTH);

  // Reading from Lime file and loading to device
  pGauge.readFromLime(latfile.c_str());
  pGauge.load();
  
  // Compuiting plaquette on device in three different way for crosschecking
  pGauge.calculatePlaq();
  pGauge.calculatePlaqCorners();
  pGauge.calculatePlaqShifts();

  finalize();
 
  return 0;
}
