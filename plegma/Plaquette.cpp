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

  double *gauge[4];
  int *lL = params.lL;
  size_t V = lL[0]*lL[1]*lL[2]*lL[3];
  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = (double*) malloc(V*gaugeSiteSize*sizeof(double));
  }

  //-Read the gauge field in lime format
  readLimeGauge(gauge, latfile, &gauge_param, params.procs);

  // The gauge is loaded in a format suitable for QUDA. We need to re-map it
  mapEvenOddToNormalGauge(gauge,gauge_param,lL[0],lL[1],lL[2],lL[3]);

  // Allocation done on BOTH, DEVICE and HOST
  PLEGMA_Gauge<double> pGauge(BOTH);

  pGauge.packGauge(gauge);
  pGauge.loadGauge();
  pGauge.calculatePlaq();

  for(int i = 0 ; i < 4 ; i++){
    free(gauge[i]);
  }

  // finalize the QUDA library
  saveTuneCache(false);
  endQuda();
    
  // finalize the communications layer
  finalizeComms();

  return 0;
}
