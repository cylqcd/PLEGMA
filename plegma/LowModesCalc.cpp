#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

#ifdef HAVE_PRIMME
int Nmethods = 16;
static std::string methods[] = { "PRIMME_DEFAULT_METHOD","PRIMME_DYNAMIC","PRIMME_DEFAULT_MIN_TIME",
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

static std::vector<std::string> listOpt = {"verbosity", "load-gauge", "Eig-isACC", "Eig-PolyDeg", "Eig-amin",
					   "Eig-amax", "Eig-spectrumPart", "Eig-tol", "Eig-maxIters"
#ifdef HAVE_ARPACK
					   ,"Eig-NkV", "Eig-logFile"
#elif HAVE_PRIMME
					   "Eig-printLevel", "Eig-method-PRIMME"
#endif
};

int main(int argc, char **argv)
{
  initialize(argc, argv, true, listOpt);
  std::string Eig_outputFile = "./eigsVdagG5V.dat";
  HGC_options->set("Eig-outputFile", "Path to dump the eigenvalues and vdag g5 v",verbosity, Eig_outputFile);
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
  eigParam.NeV = Eig_NeV;
  eigParam.isACC = Eig_isACC;
  eigParam.PolyDeg = Eig_PolyDeg;
  eigParam.amin = Eig_amin;
  eigParam.amax = Eig_amax;
  eigParam.spectrumPart = Eig_spectrumPart;
  eigParam.tol =Eig_tol;
  eigParam.maxIters = Eig_maxIters;
#if defined(HAVE_ARPACK)
  eigParam.NkV = Eig_NkV;
  eigParam.logFile = Eig_logFile;
#elif defined(HAVE_PRIMME)
  eigParam.printLevel = Eig_printLevel;
  eigParam.primme_method=getMethod(Eig_method);
#else
  PLEGMA_error("No arpack or primme is compiled");
#endif
 
  EigSolver eigSol(eigParam, dslash_type , true);
  eigSol.dumpEvalsVdagG5V(Eig_outputFile);
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
