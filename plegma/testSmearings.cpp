#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

extern int device;
//extern char latfile[];

int main(int argc, char **argv)
{
  initialize(argc, argv, false);
  int dimStout=3;
  int dimAPE=3;
  
  HGC_options->set("dim-stout", "Directions to do the Stout smearing. Allowed options (3,4). Default 3", verbosity, dimStout);
  HGC_options->set("dim-APE", "Directions to do the APE smearing. Allowed options (3,4) Default 3", verbosity, dimAPE);

  HGC_options->close();
  // Allocation done on BOTH, DEVICE and HOST
  PLEGMA_Gauge<double> gauge;

  // Reading from Lime file and loading to device
  gauge.readFromLime(latfile.c_str());
  gauge.load();
  gauge.calculatePlaq();


  PLEGMA_Gauge<double> smearedGauge;
  smearedGauge.stoutSmearing(gauge, nsmearStout, alphaStout, dimStout);
  PLEGMA_printf("Plaquette using stout is:");
  smearedGauge.calculatePlaq();
  smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, dimAPE);
  PLEGMA_printf("Plaquette using APE is:");
  smearedGauge.calculatePlaq();

  finalize();

  return 0;
}
