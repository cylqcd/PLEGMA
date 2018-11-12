#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <quda_params.h>
#include <quda_solver.h>

using namespace plegma;
using namespace quda;

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

  // Setting the QUDA params as read from command line
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setGaugeParam(gauge_param);
  
  double *gauge[4];
  double *gauge_APE[4];
  int *lL = params.lL;
  size_t V = lL[0]*lL[1]*lL[2]*lL[3];
  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = (double*) malloc(V*gaugeSiteSize*sizeof(double));
    gauge_APE[dir] = (double*) malloc(V*gaugeSiteSize*sizeof(double));
  }
  //-Read the gauge field in lime format
  readLimeGauge(gauge, latfile, &gauge_param, params.procs);

  // This gauge will be used for the inversions.
  // We need to apply the anti-periodic boundaries.
  applyBoundaryCondition(gauge, V/2 ,&gauge_param);

  // Load the gauge field into QUDA
  loadGaugeQuda((void*)gauge, &gauge_param);

  //-Read the smeared gauge field in lime format
  // TODO: create locally the smeared gauge
  readLimeGauge(gauge_APE, latfile_smeared, &gauge_param, params.procs);

  //-Loadin the gauge into QUDA for being used in case of QUDA-smearing
  gauge_param.type = QUDA_SMEARED_LINKS;
  loadGaugeQuda((void*)gauge_APE, &gauge_param);
  mapEvenOddToNormalGauge(gauge_APE,gauge_param,lL[0],lL[1],lL[2],lL[3]);

  // ensuring mu positive
  if(mu<0) mu*=-1.;
  QUDA_solver solverUP(mu);

  // ensuring mu negative
  if(mu>0) mu*=-1.;
  QUDA_solver solverDN(mu);

  

  
  delete &solverUP;
  delete &solverDN;
  
  for(int i = 0 ; i < 4 ; i++){
    free(gauge[i]);
    free(gauge_APE[i]);
  }
  
  // finalize the QUDA library
  endQuda();
  
  // finalize the communications layer
  finalizeComms();

  return 0;
}

