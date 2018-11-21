#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

extern int device;
extern char latfile[];

int main(int argc, char **argv)
{
  PLEGMA_params params;
  read_command_line(argc, argv, &params);
  
  // initialize QMP/MPI, QUDA comms grid and RNG 
  initComms(argc, argv, params.procs);

  // initialize the QUDA library
  initQuda(device);
  print_info();

  // initialize PLEGMA info
  initialize(&params);
  print_status();

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
  pGauge.calculatePlaqShifts();

  // finalize the QUDA library
  saveTuneCache(false);
  endQuda();
    
  // finalize the communications layer
  finalizeComms();

  return 0;
}
