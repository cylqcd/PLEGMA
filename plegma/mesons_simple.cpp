#include <PLEGMA.h>
#include <PLEGMA_utils.h>

std::vector<double> runtime;
#define TIME(fnc)  runtime.push_back(MPI_Wtime()); fnc;			\
  PLEGMA_printf("TIME for "#fnc" %f sec\n", MPI_Wtime()-runtime.back()); \
  runtime.pop_back()

std::vector<std::thread> threads;
//#define THREAD(fnc) threads.push_back(std::thread([=]() { TIME(fnc); }))
#define THREAD(fnc) saveTuneCache(false); TIME(fnc)

using namespace plegma;
using namespace quda;
static std::vector<std::string> listOpt = { "verbosity", "load-gauge", "nsmear-APE", "alpha-APE", "nsmear-gauss", "alpha-gauss",
					    "nsrc", "src-filename", "maxQsq", "twop-filename", "corr-file-format", "corr-space", "tSinks", "threep-filename"};
  
int main(int argc, char **argv) {
  initializeOptions(argc, argv, true, listOpt);
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
  //=========================================================================================================//
  // Only multiple of three accepted
  
  initializePLEGMA();

  {
    {
      // Reading from Lime file and loading to device
      PLEGMA_Gauge<double> gauge;
      gauge.readFile(latfile, LIME_FORMAT);
      gauge.load();
      gauge.calculatePlaq();
      
      // Loading to QUDA and computing plaquette also there
      initGaugeQuda(gauge, true);
      plaqQuda();
      
    }

    updateOptions(LIGHT);
    TIME(QUDA_solver solver(mu));

    std::string given_twop_filename = twop_filename;
    std::string given_threep_filename = threep_filename;
    
    for(int isource = 0; isource < numSourcePositions; isource++){
      site& source = sourcePositions[isource];
      PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
		    isource, source[0], source[1], source[2], source[3]);
      //      updateOptions(srcInputFile + std::to_string(isource), listOpt, add_options);


      auto computePropagator = [&](PLEGMA_Propagator<float>& prop){
				 // ensuring mu value
				 for(int isc = 0 ; isc < 12 ; isc++){
				   PLEGMA_Vector<double> vectorInOut;
				   vectorInOut.pointSource(source, isc/3, isc%3, DEVICE);
				   // Inverting
				   PLEGMA_printf("Going to invert LIGHT for component %d\n", isc);
				   TIME(solver.solve(vectorInOut, vectorInOut));
				   PLEGMA_Vector<float> vectorAuxF;
				   vectorAuxF.copy(vectorInOut);
				   prop.absorb(vectorAuxF, isc/3, isc%3);

				 }
			       };

      char * src_string;
      asprintf(&src_string, "_sx%02dsy%02dsz%02dst%03d", source[0], source[1], source[2], source[3]);
      twop_filename = given_twop_filename + src_string;
      threep_filename = given_threep_filename + src_string;
      free(src_string);

      PLEGMA_Propagator<float> propUP;
      TIME(computePropagator(propUP));
      propUP.rotateToPhysicalBase_device(mu/abs(mu));
      PLEGMA_Correlator<float> corr(corr_space, source, maxQsq);
      TIME(corr.contractMesonsNew(propUP, propUP));
      corr.setDatasets((std::vector<std::string>) {"twop_meson_uu"});
      propUP.writeFile(twop_filename, corr_file_format);
      THREAD(corr.writeFile(twop_filename, corr_file_format));
    }
    while(not threads.empty()) {threads.back().join(); threads.pop_back();}
  }
    
  finalize();
  return 0;
}
  
