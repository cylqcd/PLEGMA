//-------------------//
// PLEGMA Parameters //
//-------------------//

#ifdef ALLOCATE
#define define(var,...) var EQUAL(__VA_ARGS__)
#else
#define define(var,...) extern var 
#endif


// Main paramters -- read by plegmaOptions
define(int dims[N_DIMS], {8,8,8,16});
define(int procs[N_DIMS], {1,1,1,1});
define(std::string latfile);
define(int verbosity, 1);

define(int nsmearAPE, 20);
define(double alphaAPE, 0.5);
define(int nsmearGauss, 50);
define(double alphaGauss, 0.2);
define(int nsmearStout, 5);
define(double alphaStout, 0.2);
define(double xiMomSm,0.6); 
define(std::vector<GAMMAS> gammas, {}); 
define(int numSourcePositions, 1);
define(std::string pathListGaugeConfs);
define(std::vector<std::string> listGaugeConfs);
define(std::string pathListSourcePositions);
define(std::vector<site> sourcePositions, {});
define(int maxQsq, 64);
define(FILE_FORMAT corr_file_format, HDF5_FORMAT);
define(CORR_SPACE corr_space, MOMENTUM_SPACE);
define(std::string twop_filename, "./twop");
define(std::string threep_filename, "./threep");
define(std::vector<int> tSinks, {});
define(std::vector<std::string> Projs, {});
define(int rng_seed, 123456);
define(std::string inputLIGHT,"");
define(std::string inputST,"");
define(std::string inputCH,"");

define(int Eig_NeV, 10);
define(bool Eig_isACC, true);
define(int Eig_PolyDeg, 100);
define(double Eig_amin, 1e-04);
define(double Eig_amax, 4.5);
define(std::string Eig_spectrumPart, "SR");
define(double Eig_tol, 1e-05);
define(int Eig_maxIters, 100000);
#if defined(HAVE_ARPACK)
define(int Eig_NkV, 2*Eig_NeV);
define(std::string Eig_logFile, "./logfile.out");
#elif defined(HAVE_PRIMME)
define(std::string Eig_method, "PRIMME_Arnoldi");
define(int Eig_printLevel, 4);
#endif
