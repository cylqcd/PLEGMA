//-------------------//
// PLEGMA Parameters //
//-------------------//

#include <boost/preprocessor/control/if.hpp>
#ifdef ALLOCATE
#define NOTHING(...)
#define EQUAL_CAT(...) =  __VA_ARGS__
#define EQUAL(...) BOOST_PP_IF(IS_EMPTY(__VA_ARGS__),		\
			       NOTHING,				\
			       EQUAL_CAT) (__VA_ARGS__)
#define define(var,...) var EQUAL(__VA_ARGS__)
#else
#define define(var,...) extern var 
#endif

// Main paramters -- read by basicOptions
define(int dims[N_DIMS], {8,8,8,16});
define(int procs[N_DIMS], {1,1,1,1});
define(std::string latfile);
define(int verbose, 1);

// Additional paramters -- read by extraOptions
define(int nsmearAPE, 20);
define(double alphaAPE, 0.5);
define(int nsmearGauss, 50);
define(double alphaGauss, 0.2);

define(int numSourcePositions, 1);
define(std::string pathListSourcePositions);
define(int maxQsq, 64);
define(int corr_file_format, 1);
define(int corr_space, 1);
define(std::string twop_filename);
define(std::string threep_filename);
define(int numTSink);
define(std::string pathListTSink);
define(int numProj);
define(std::string pathListProj);

