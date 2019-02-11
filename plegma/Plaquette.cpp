#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

int main(int argc, char **argv)
{
  initialize(argc, argv);

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setGaugeParam(gauge_param);

  //-Read the gauge field in lime format
  GaugeBuffer<double> gauge;
  readLimeGauge(gauge.get_ptr(), latfile.c_str(), &gauge_param, procs);

  // The gauge is loaded in a format suitable for QUDA. We need to re-map it
  mapEvenOddToNormalGauge(gauge.get_ptr(),gauge_param, dims);

  // Allocation done on BOTH, DEVICE and HOST
  PLEGMA_Gauge<double> pGauge(BOTH);

  pGauge.pack(gauge.get_ptr());
  pGauge.load();
  pGauge.calculatePlaq();
  pGauge.calculatePlaqCorners();
  pGauge.calculatePlaqShifts();

  finalize();
 
  return 0;
}
