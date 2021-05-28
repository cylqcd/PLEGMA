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
    // Reading from Lime file and loading to device
    PLEGMA_Gauge<double> gauge;
    if ( latfile == "unit" ) {
      gauge.setUnit((std::vector<int>) {0,4,8, 9,13,17, 18,22,26, 27,31,35});
      gauge.unload();//I think this is not necessary
    }
    else {
      gauge.readFile(latfile, LIME_FORMAT);
      gauge.load();
    }
    gauge.calculatePlaq();
    
    // Loading to QUDA and computing plaquette also there
    initGaugeQuda(gauge, true);
    plaqQuda();
    
#if 0
    PLEGMA_Vector3D<double> tmp3;
    PLEGMA_Vector<double> tmp, tmp43;
    tmp.randInit(rng_seed);PLEGMA_printf("1\n");
    tmp3.randInit(rng_seed);PLEGMA_printf("2\n");
    
    tmp.stochastic_Z(4);PLEGMA_printf("3\n");
    tmp3.stochastic_Z(4);PLEGMA_printf("4\n");
    
    tmp43.absorb(tmp3,0);PLEGMA_printf("5\n");
    tmp.unload();PLEGMA_printf("6\n");
    tmp43.unload();PLEGMA_printf("7\n");

    tmp3.unload();
    PLEGMA_printf("norm (4D: %g) (3D: %g, %g)\n",tmp.norm(),tmp3.norm(), tmp43.norm());
    double* h_p = tmp.H_elem();
    double* h_p3 = tmp3.H_elem();
    for(int x=0;x<dims[0];x++)for(int y=0;y<dims[1];y++)for(int z=0;z<dims[2];z++)for(int t=0;t<dims[3];t++)for(int s=0;s<12;s++){
              PLEGMA_printf("(x,y,z,t,s) = (%d,%d,%d,%d,%d): 4D:%g %g , 3D:%g %g\n",x,y,z,t,s, *h_p,*(h_p+1), *h_p3, *(h_p3+1));
              h_p+=2;if(t==0)h_p3+=2;
            }
    tmp3.absorb(tmp,0);
    PLEGMA_printf("norm2 (4D: %g) (3D: %g)\n",tmp.norm(),tmp3.norm());
#endif
    updateOptions(LIGHT);
    TIME(QUDA_solver solver(mu));

    std::string given_twop_filename = twop_filename;
    std::string given_threep_filename = threep_filename;
    
    for(int isource = 0; isource < numSourcePositions; isource++){
      site& source = sourcePositions[isource];
      PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
		    isource, source[0], source[1], source[2], source[3]);
      //      updateOptions(srcInputFile + std::to_string(isource), listOpt, add_options);


      auto computePropagator = [&](PLEGMA_Propagator<double>& prop){
				 // ensuring mu value
				 for(int isc = 0 ; isc < 12 ; isc++){
				   PLEGMA_Vector<double> vectorInOut;
				   vectorInOut.pointSource(source, isc/3, isc%3, DEVICE);
				   // Inverting
				   PLEGMA_printf("Going to invert LIGHT for component %d\n", isc);
				   TIME(solver.solve(vectorInOut, vectorInOut));
				   PLEGMA_Vector<double> vectorAuxF;
				   vectorAuxF.copy(vectorInOut);
				   prop.absorb(vectorAuxF, isc/3, isc%3);
				   // Test oneD
				   /*
				   PLEGMA_Vector<double> vectorTd, vectorTdt;
				   for(int dir=0;dir<N_DIMS;dir++){
				     vectorTdt.covD(vectorInOut,gauge,dir);
				     vectorTdt.apply_gamma(static_cast<GAMMAS>(dir%N_DIMS+1));
				     vectorTd.add(vectorTdt,(dir<N_DIMS)?0.5:-0.5);
				   }
				   vectorTdt.copy(vectorInOut);
				   vectorTdt.apply_gamma5();
				   std::complex<double> Isingle(0,1);
				   vectorTdt.cscale(mu*Isingle);
				   vectorTd.add(vectorTdt,1.0);
				   vectorTd.writeHDF5(given_twop_filename+std::to_string(isc));
				   */
				 }
			       };

      char * src_string;
      asprintf(&src_string, "_sx%02dsy%02dsz%02dst%03d", source[0], source[1], source[2], source[3]);
      twop_filename = given_twop_filename + src_string;
      threep_filename = given_threep_filename + src_string;
      free(src_string);

      PLEGMA_Propagator<double> propUP;
      TIME(computePropagator(propUP));
      propUP.rotateToPhysicalBase_device(mu/abs(mu));
      
      PLEGMA_Correlator<double> corr(corr_space, source, maxQsq);
      TIME(corr.contractMesonsNew(propUP, propUP));
      corr.setDatasets((std::vector<std::string>) {"twop_meson_uu"});
      // or
      //TIME(corr.contractMesons(propUP, propUP));
      
      THREAD(propUP.writeFile(twop_filename, corr_file_format));
      THREAD(corr.writeFile(twop_filename, corr_file_format));
    }
    while(not threads.empty()) {threads.back().join(); threads.pop_back();}
  }
    
  finalize();
  return 0;
}
  
