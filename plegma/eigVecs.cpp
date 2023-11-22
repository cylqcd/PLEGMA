#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;

static std::vector<double> runtime;
#define TIME(fnc)  runtime.push_back(MPI_Wtime()); fnc;                 \
  PLEGMA_printf("TIME for "#fnc" %f sec\n", MPI_Wtime()-runtime.back()); \
  runtime.pop_back() 

static std::vector<std::string> listOpt = {"verbosity", "load-gauge", "Eig-isACC", "Eig-PolyDeg", "Eig-amin",
					   "Eig-amax", "Eig-spectrumPart", "Eig-tol", "Eig-maxIters", "Eig-NeV",
#if  defined(HAVE_ARPACK) || defined(QUDAEIG)
					   "Eig-NkV", "Eig-logFile"
#elif defined(HAVE_PRIMME)
					   "Eig-printLevel", "Eig-method-PRIMME"
#endif
};

int main(int argc, char **argv)
{
    initializeOptions(argc, argv, true, listOpt);
    //================ Add your options in this between initializeOptions and initializePLEGMA ================//
    //std::string Eig_outputFile = "./eigsVdagG5V.dat";
    //HGC_options->set("Eig-outputFile", "Path to dump the eigenvalues and vdag g5 v",verbosity, Eig_outputFile);
    bool isReadEigenVecs = false, isWriteEigenVecs = true;
    std::string fnameEigenVecsPrefix="";
    HGC_options->set("readEigenVectors", "Where we want to read EigenVectors from file", verbosity, isReadEigenVecs);
    HGC_options->set("writeEigenVectors", "Where we want to read EigenVectors from file", verbosity, isWriteEigenVecs);
    HGC_options->set("prefixEigenVecsFile", "Path with prefix for the filenames of the eigenvectors", verbosity, fnameEigenVecsPrefix);
    #ifdef QUDAEIG
    int batched_rotate = 1;
    HGC_options->set("batched-rotate", "The size of the batch during Ritz rotation", verbosity, batched_rotate);
    #endif
    //=========================================================================================================//
    initializePLEGMA();
    // Reading from Lime file and loading to device
    PLEGMA_Gauge<double> gauge;
    gauge.readFile(latfile, LIME_FORMAT);
    gauge.calculatePlaq();

    // Loading to QUDA and computing plaquette also there
    initGaugeQuda(gauge, true);
    plaqQuda();

    #if defined(HAVE_EIGENSOLVER)
        EigSolver *eigSol = nullptr;

        EigSolverParams eigParam;
        eigParam.NeV = Eig_NeV;
        eigParam.isACC = Eig_isACC;
        eigParam.PolyDeg = Eig_PolyDeg;
        eigParam.amin = Eig_amin;
        eigParam.amax = Eig_amax;
        eigParam.spectrumPart = Eig_spectrumPart;
        eigParam.tol = Eig_tol;
        eigParam.maxIters = Eig_maxIters;
    #ifdef QUDAEIG
        eigParam.batched_rotate = batched_rotate;
    #endif
    #if defined(HAVE_ARPACK) || defined(QUDAEIG)
        eigParam.NkV = Eig_NkV;
        eigParam.logFile = Eig_logFile;
    #elif defined(HAVE_PRIMME)
        eigParam.printLevel = Eig_printLevel;
        eigParam.primme_method=getMethod(Eig_method);
    #else
        PLEGMA_error("No arpack or primme is compiled");
    #endif
 
    TIME(eigSol = new EigSolver(eigParam, dslash_type, isReadEigenVecs, isWriteEigenVecs, fnameEigenVecsPrefix, true));


#else
  PLEGMA_error("No eigenSolver is compiled");
#endif

  finalize();
  return 0;
}
