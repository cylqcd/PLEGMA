#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

int main(int argc, char **argv)
{
  initialize(argc, argv);

  // Allocation done on BOTH, DEVICE and HOST
  PLEGMA_Gauge<double> gauge(BOTH);

  // Reading from Lime file and loading to device
  gauge.readFromLime(latfile.c_str());
  gauge.load();
  
  // Compuiting plaquette on device in three different way for crosschecking
  gauge.calculatePlaq();
  gauge.calculatePlaqCorners();
  gauge.calculatePlaqShifts();

  // Loading to QUDA and computing plaquette also there
  initGaugeQuda(gauge, false, QUDA_SU3_LINKS);
  double plq[3]; // total, spatial and temporal plaquette
  plaqQuda(plq);
  printfQuda("TEST: Calculated plaquette in QUDA: %f (sp: %f, T: %f)\n", plq[0], plq[1], plq[2]);
  
  finalize();
 
  return 0;
}
