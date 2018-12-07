#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <contractQuda.h>
#include <quda_params.h>

#include <quda_params.h>
#include <invert_quda.h>
#include <quda_solver.h>
#include <PLEGMA_Qdirac.h>

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
  PLEGMA_QLoops<double> loops_std(BOTH,true);
  QudaInvertParam inv_params = solverDN->getInvParams();
  // just put units to the whole for debugging
  source.setUnit((std::vector<int>) {0,1,2,3,4,5,6,7,8,9,10,11});
  solverDN->solve(phi,source);
  // for convention reasons for quark loops we put the normalization factors of the fields later in the analysis
  phi.scaleVector(1./(2.*inv_params.kappa)); 

  
  loops_std.oneEnd_trick(phi,phi,tmp,-1.,true); //standard one-end trick
  std::string prefix = "/onyx/noether/h/khadjiyiannakou/runs/";
  loops_std.write_ASCII(prefix+"std_local_loops.0000.dat", prefix+"std_oneD_loops.0000.dat", prefix+"std_oneDC_loops.0000.dat");

  PLEGMA_QLoops<double> loops_gen(BOTH,true);
  PLEGMA_Vector<double> phi_r(BOTH);
  PLEGMA_Qdirac *D = nullptr;
  if(inv_params.dslash_type == QUDA_TWISTED_CLOVER_DSLASH)
    D = new PLEGMA_Qdirac(QUDA_CLOVER_WILSON_DSLASH);
  else if (inv_params.dslash_type == QUDA_TWISTED_MASS_DSLASH)
    D =	new PLEGMA_Qdirac(QUDA_WILSON_DSLASH);
  else
    errorQuda("Only QUDA_TWISTED_CLOVER_DSLASH and QUDA_TWISTED_MASS_DSLASH are allowed for the one-end trick");

  D->apply<M>(phi_r,phi);
  phi_r.apply_gamma5();
  loops_gen.oneEnd_trick(phi, phi_r, tmp, +1., true); //generalized one-end trick
  loops_gen.write_ASCII(prefix+"gen_local_loops.0000.dat", prefix+"gen_oneD_loops.0000.dat", prefix+"gen_oneDC_loops.0000.dat");

  delete D;
  delete solverDN;
  finalize();

  return 0;
}
