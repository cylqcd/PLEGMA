#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

#ifdef HAVE_PRIMME
int Nmethods = 16;
std::string methods[] = { "PRIMME_DEFAULT_METHOD","PRIMME_DYNAMIC","PRIMME_DEFAULT_MIN_TIME",
			  "PRIMME_DEFAULT_MIN_MATVECS", "PRIMME_Arnoldi", "PRIMME_GD",
			 "PRIMME_GD_plusK", "PRIMME_GD_Olsen_plusK", "PRIMME_JD_Olsen_plusK", "PRIMME_RQI",
			 "PRIMME_JDQR", "PRIMME_JDQMR", "PRIMME_JDQMR_ETol", "PRIMME_STEEPEST_DESCENT",
			  "PRIMME_LOBPCG_OrthoBasis", "PRIMME_LOBPCG_OrthoBasis_Window"};
static primme_preset_method getMethod(std::string str){
  for(int i = 0; i < Nmethods; i++)
    if(str == methods[i])
      return static_cast<primme_preset_method>(i);
  PLEGMA_warning("Method provided %s is not in PRIMME, available methods are",str.c_str());
  for(int i = 0; i < Nmethods; i++)
    PLEGMA_printf(methods[i].c_str());
  PLEGMA_exit(-1);
  return static_cast<primme_preset_method>(0);
}
#endif


int main(int argc, char **argv)
{
  initialize(argc, argv);
  int NeV = 10;
  bool isACC = true;
  int PolyDeg = 100;
  double amin=1e-04, amax=4.5;
  std::string spectrumPart = "SR";
  double tol = 1e-05;
  int maxIters = 100000;
  std::string outputFile = "./eigsVdagG5V.dat";
#if defined(HAVE_ARPACK)
  int NkV = 2*NeV;
  std::string logFile = "./logfile.out";
#elif defined(HAVE_PRIMME)
  std::string method = "PRIMME_Arnoldi";
  int printLevel = 4;
#else
  PLEGMA_error("Not implemented");
#endif

  
  HGC_options->set("NeV", "Number of eigenpairs to compute. (Default 10)", verbosity, NeV);
#ifdef HAVE_ARPACK
  HGC_options->set("NkV", "Number of vectors for the Krylov subspace. (Default 2*NeV)", verbosity, NkV);
  HGC_options->set("logFile", "Path for the logfile of the eigensolver. (Default ./logfile.out)", verbosity, logFile);
#elif HAVE_PRIMME
  HGC_options->set("printLevel", "Print Level for the PRIMEE eigenSolver. (Default 4)", verbosity, printLevel);
  HGC_options->set("method-PRIMME", "The method for eigensolver from PRIMME see manual for all. (Default PRIMME_Arnoldi)", verbosity, method);
#endif
  HGC_options->set("isACC", "If we want to use Polynomial accel. (Default true)", verbosity, isACC);
  HGC_options->set("PolyDeg", "The degree of the Polynomial. (Default 100)", verbosity, PolyDeg);
  HGC_options->set("amin", "The low bound of the Polynomial called amin. (Default 1e-4)", verbosity, amin);
  HGC_options->set("amax", "The upper bound of the Polynomial called amax. (Default 4.5)", verbosity, amax);
  HGC_options->set("spectrumPart", "Which part of the spectrum you want to compute. Options (SR, LR). (Default SR)", verbosity, spectrumPart);
  HGC_options->set("tol", "Tolerance for the eigensolver. At least all eigenpairs converge to this tol. (Default 1e-05)", verbosity, tol);
  HGC_options->set("maxIters", "Tolerance for the eigensolver. At least all eigenpairs converge to this tol. (Default 100000)", verbosity, maxIters);
  HGC_options->set("outputFile", "Path to dump the eigenvalues and vdag g5 v. (Default ./eigsVdagG5V.dat)",verbosity, outputFile);
  HGC_options->close();
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
  eigParam.NeV = NeV;
  eigParam.isACC = isACC;
  eigParam.PolyDeg = PolyDeg;
  eigParam.amin = amin;
  eigParam.amax = amax;
  eigParam.spectrumPart = spectrumPart;
  eigParam.tol =tol;
  eigParam.maxIters = maxIters;
#if defined(HAVE_ARPACK)
  eigParam.NkV = NkV;
  eigParam.logFile = logFile;
#elif defined(HAVE_PRIMME)
  eigParam.printLevel = printLevel;
  eigParam.primme_method=getMethod(method);
#else
  PLEGMA_error("No arpack or primme is compiled");
#endif
 
  EigSolver eigSol(eigParam, dslash_type , true);
  eigSol.dumpEvalsVdagG5V(outputFile);
  PLEGMA_Vector<double> in,out;
  in.setUnit((std::vector<int>) {0,1,2,3,4,5,6,7,8,9,10,11});
  eigSol.projectVector(out,in);
  std::complex<double> aka = out.dot(out);

#else
  PLEGMA_error("No eigenSolver is compiled");
#endif

  finalize();
  return 0;
}
