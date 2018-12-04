#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <contractQuda.h>
#include <quda_params.h>

#include <quda_params.h>
#include <invert_quda.h>
#include <quda_solver.h>

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
  // This gauge will be used for the inversions.
  // We need to apply the anti-periodic boundaries.
  applyBoundaryCondition(gauge.get_ptr(), params.lL, &gauge_param);
  initGaugeQuda((void*)gauge.get_ptr(), gauge_param);  
  mapEvenOddToNormalGauge(gauge.get_ptr(),gauge_param,params.lL);

  // Allocation done on BOTH, DEVICE and HOST
  PLEGMA_Gauge<double> pGauge(BOTH);
  pGauge.pack(gauge.get_ptr());
  pGauge.load();

  // ensuring mu negative
  if(mu>0) mu*=-1.;
  QUDA_solver *solverDN = new QUDA_solver(mu);
  PLEGMA_Vector<double> source(DEVICE);
  PLEGMA_Vector<double> phi(BOTH);
  PLEGMA_Vector<double> tmp(BOTH,&pGauge);
  PLEGMA_QLoops<double> loops(BOTH,true);
  QudaInvertParam inv_params = solverDN->getInvParams();
  // just put units to the whole for debugging
  source.setUnit((std::vector<int>) {0,1,2,3,4,5,6,7,8,9,10,11});
  solverDN->solve(phi,source);
  // for convention reasons for quark loops we put the normalization factors of the fields later in the analysis
  phi.scaleVector(1./(2.*inv_params.kappa)); 
  
  loops.oneEnd_trick(phi,phi,tmp,-1.,true); //standard one-end trick
  std::string prefix = "/onyx/noether/h/khadjiyiannakou/runs/";
  loops.write_ASCII(prefix+"std_local_loops.0000.dat", prefix+"std_oneD_loops.0000.dat", prefix+"std_oneDC_loops.0000.dat");

  delete solverDN;
  finalize();

  return 0;
}
