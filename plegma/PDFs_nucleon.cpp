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
  mapEvenOddToNormalGauge(gauge.get_ptr(),gauge_param,params.lL);

  // Allocation done on BOTH, DEVICE and HOST
  PLEGMA_Gauge<double> pGauge(BOTH);
  pGauge.pack(gauge.get_ptr());
  pGauge.load();
  //  std::vector<int> indDiag = {0, 4, 8, 9, 13, 17, 18, 22, 26, 27, 31, 35};
  //pGauge.setUnitMatrix(indDiag);
  PLEGMA_Su3field<double> su3(BOTH);
  PLEGMA_Su3field<double> WL(BOTH);
  PLEGMA_Su3field<double> tmp(BOTH);
  su3.absorbDir_device(pGauge, 0);
  WL.setUnit( (std::vector<int>) {0,4,8});
  for(int i = 0 ; i < GK_totalL[0];i++ )
    WL.wilsonLineUpdate(su3, tmp, 4+0); // build Wilson line in the +x direction
  //  pGauge.calculatePlaqShifts();

  finalize();

  return 0;
}
