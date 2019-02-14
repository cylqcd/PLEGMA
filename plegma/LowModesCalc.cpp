#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <eigSolver.h>

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

  applyBoundaryCondition(gauge.get_ptr(), params.lL, &gauge_param);
  initGaugeQuda((void*)gauge.get_ptr(), gauge_param);

#if defined(HAVE_EIGENSOLVER)
  EigSolverParams eigParam;
  eigParam.NeV = 50;
  eigParam.isACC = true;
  eigParam.PolyDeg = 300;
  eigParam.amin = 5e-04;
  eigParam.amax = 4.5;
  eigParam.spectrumPart = "SR";
  eigParam.tol =1e-05;//1e-05;
  eigParam.maxIters = 100000;
#if defined(HAVE_ARPACK)
  eigParam.NkV = 80;
  eigParam.logFile = "/home/khadjiyiannakou_tmp//khadjiyiannakou/runs/arpack.log";
#elif defined(HAVE_PRIMME)
  eigParam.printLevel = 4;
  eigParam.primme_method=PRIMME_JD_Olsen_plusK;
#else
  errorQuda("No arpack or primme is compiled");
#endif
 
  EigSolver eigSol(eigParam, QUDA_TWISTED_CLOVER_DSLASH , true);
  eigSol.dumpEvalsVdagG5V("/home/khadjiyiannakou_tmp//khadjiyiannakou/runs/eigVals_VdagG5V.dat");
  PLEGMA_Vector<double> in,out;
  in.setUnit((std::vector<int>) {0,1,2,3,4,5,6,7,8,9,10,11});
  eigSol.projectVector(out,in);
  std::complex<double> aka = out.dot(out);

#else
  errorQuda("No eigenSolver is compiled");
#endif

  finalize();
  return 0;
}

  // PLEGMA_Vector<double> source(DEVICE);
  // source.setUnit((std::vector<int>) {0,1,2,3,4,5,6,7,8,9,10,11});
  // std::complex<double> a(2.,0.);
  // std::complex<double> b(3.,0.);
  // std::complex<double> c(2.,0.);
  // plegma::axpbypcz(4*3*GK_localVolume,reinterpret_cast<double(&)[2]>(a), source.D_elem(), reinterpret_cast<double(&)[2]>(b), source.D_elem(), reinterpret_cast<double(&)[2]>(c), source.D_elem());
  // std::complex<double> aka = source.dot(source);
  // finalize();
