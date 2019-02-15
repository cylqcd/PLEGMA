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

define(int numSourcePositions, 1);
define(std::string pathListSourcePositions);
define(int (*sourcePositions)[N_DIMS], NULL);
define(int maxQsq, 64);
define(FILE_WRITE_FORMAT corr_file_format, HDF5_FORM);
define(CORR_SPACE corr_space, MOMENTUM_SPACE);
define(std::string twop_filename, "./twop");
define(std::string threep_filename, "./threep");
define(int numTSink);
define(std::string pathListTSink);
define(int numProj);
define(std::string pathListProj);

