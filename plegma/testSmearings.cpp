#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

extern int device;
extern char latfile[];

int main(int argc, char **argv)
{
  initialize(argc, argv);

  // Allocation done on BOTH, DEVICE and HOST
  PLEGMA_Gauge<double> gauge;

  // Reading from Lime file and loading to device
  gauge.readFromLime(latfile.c_str());
  gauge.load();
  gauge.calculatePlaq();

  PLEGMA_Gauge<double> smearedGauge;
  smearedGauge.stoutSmearing(gauge, 10, 0.1, 4);
  smearedGauge.calculatePlaq();
  smearedGauge.APEsmearing(gauge, 10, 0.1, 3);
  smearedGauge.calculatePlaq();

  finalize();

  return 0;
}
