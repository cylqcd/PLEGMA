#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

extern int device;
extern char latfile[];

int main(int argc, char **argv)
{
  PLEGMA_params params;
  initialize(argc, argv, &params);

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setGaugeParam(gauge_param);

  //-Read the gauge field in lime format
  GaugeBuffer<double> gauge(params);
  readLimeGauge(gauge.get_ptr(), latfile, &gauge_param, params.procs);

  // The gauge is loaded in a format suitable for QUDA. We need to re-map it
  mapEvenOddToNormalGauge(gauge.get_ptr(),params.lL);

  // Allocation done on BOTH, DEVICE and HOST
  PLEGMA_Gauge<double> pGauge(BOTH);

  pGauge.pack(gauge.get_ptr());
  pGauge.load();
  pGauge.stoutSmearing(pGauge, 10, 0.1, 4);
  pGauge.calculatePlaqShifts();

  finalize();

  return 0;
}
