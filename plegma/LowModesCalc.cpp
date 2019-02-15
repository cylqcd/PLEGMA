#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

int main(int argc, char **argv)
{
  initialize(argc, argv);

  // Reading from Lime file and loading to device
  PLEGMA_Gauge<double> gauge;
  gauge.readFromLime(latfile.c_str());
  gauge.load();
  gauge.calculatePlaq();

  // Loading to QUDA and computing plaquette also there
  initGaugeQuda(gauge, true);
  plaqQuda();

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
  eigParam.primme_method=PRIMME_RQI;
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
