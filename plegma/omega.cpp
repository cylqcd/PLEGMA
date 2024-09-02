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
					    "nsrc", "src-filename", "maxQsq", "twop-filename", "corr-file-format", "corr-space"// ,
};
  
int main(int argc, char **argv) {
  initializeOptions(argc, argv, true, listOpt);
  initializePLEGMA();

  {
    PLEGMA_Gauge<double> smearedGauge(BOTH);
    {
      // Reading from Lime file and loading to device
      PLEGMA_Gauge<double> gauge;
      gauge.readFile(latfile, LIME_FORMAT);
      gauge.calculatePlaq();
      
      // Loading to QUDA and computing plaquette also there
      initGaugeQuda(gauge, true);
      plaqQuda();

      // Smearing
      TIME(smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3));
      PLEGMA_printf("Plaquette after smearing:\n");
      smearedGauge.calculatePlaq();
    }

    updateOptions(LIGHT);
    TIME(QUDA_solver solver(mu));

    
    std::string given_twop_filename = twop_filename;
    
    for(int isource = 0; isource < numSourcePositions; isource++){
      site& source = sourcePositions[isource];
      PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
		    isource, source[0], source[1], source[2], source[3]);

      auto computePropagator = [&](PLEGMA_Propagator<float>& prop, bool finalize) {
	PLEGMA_Gauge3D<double> smearedGauge3D;
	smearedGauge3D.absorb(smearedGauge, source[DIM_T]);
	
	PLEGMA_Vector<double> vectorInOut;
	PLEGMA_Vector<double> vectorAuxD;
	PLEGMA_Vector<float> vectorAuxF;
	
	for(int isc = 0 ; isc < 12 ; isc++){
	  { // Smearing the source
	    PLEGMA_Vector3D<double> vector1, vector2;
	    vector1.pointSource(source, isc/3, isc%3, DEVICE);
	    TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nsmearGauss, alphaGauss));
	    vectorInOut.absorb(vector2,source[DIM_T]);
	  }
	  PLEGMA_printf("Going to invert %f for component %d\n", mu, isc);
	  TIME(solver.solve(vectorInOut, vectorInOut));
	  
	  { // Smearing the solution
	    TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nsmearGauss, alphaGauss));
	    vectorAuxF.copy(vectorAuxD);
	    prop.absorb(vectorAuxF, isc/3, isc%3);
	  }
	}
	if(finalize) {
	  prop.rotateToPhysicalBase_device(mu/abs(mu));
	  prop.applyBoundaries_device(source[DIM_T]);
	}
      };
      
      char * src_string;
      asprintf(&src_string, "_sx%02dsy%02dsz%02dst%03d", source[0], source[1], source[2], source[3]);
      twop_filename = given_twop_filename + src_string + ".h5";
      free(src_string);

      if(access( twop_filename.c_str(), F_OK ) != -1) {
	continue;
      }
      
      PLEGMA_Propagator<float> prop;

      TIME(computePropagator(prop, true));
      
      PLEGMA_Correlator<float> corr(corr_space, source, maxQsq);
	      
      PLEGMA_Propagator<float> none(NONE);
      TIME(corr.contractBaryonsUDSC(none, none, prop, none, false, false));
      
      char * group;
      asprintf(&group, "baryons_s[%+1.6e]", mu);
      corr.setGroups(group);
      free(group);
      THREAD(corr.writeFile(twop_filename, corr_file_format));

      TIME(corr.contractMesonsNew(prop, prop));
      
      char *dset;
      asprintf(&dset, "twop_mesons_s[%+1.6e]", mu);
      corr.setDatasets((std::vector<std::string>) {dset});
      free(dset);
      THREAD(corr.writeFile(twop_filename, corr_file_format));

      TIME(corr.contractMesonsOpen(prop, prop));
      
      asprintf(&dset, "twop_mesons_open_s[%+1.6e]", mu);
      corr.setDatasets((std::vector<std::string>) {dset});
      free(dset);
      THREAD(corr.writeFile(twop_filename, corr_file_format));

    }
    while(not threads.empty()) {threads.back().join(); threads.pop_back();}
  }

  //finalize();
  return 0;
}

