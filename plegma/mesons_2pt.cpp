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
					    "nsrc", "src-filename", "maxQsq", "twop-filename", "corr-file-format", "corr-space"};
  
int main(int argc, char **argv) {
  initializeOptions(argc, argv, true, listOpt);
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
  double mu_s;
  double mu_ud = mu;
  double mu_ud_factor[QUDA_MAX_MG_LEVEL];
  for(int i=0;i<QUDA_MAX_MG_LEVEL;i++) mu_ud_factor[i] = mu_factor[i];
  int nsmearGauss_s = nsmearGauss/2;
  int startSource = 0;
  int endSource = numSourcePositions;
  auto add_options = [&](Options& options) {
    options.set("mu-s", "List of mu_s to run for the strange quark in baryons", verbosity, mu_s);
    options.set("nsmear-gauss-s", "Number of Gaussian smearing step for the strange quark propagator", verbosity, nsmearGauss_s);
    options.set("start-src", "The index of the source position where to start the calculation", verbosity, startSource);
    options.set("end-src", "The index of the source position where to stop the calculation", verbosity, endSource);
		     };
  add_options(*HGC_options);
  //=========================================================================================================//
  // Only multiple of three accepted
  
  initializePLEGMA();

  {
    PLEGMA_Gauge<double> smearedGauge(BOTH);
    PLEGMA_Gauge<float> contractGauge(BOTH);
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
      
      // Smearing
      TIME(smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3));
      PLEGMA_printf("Plaquette after smearing:\n");
      smearedGauge.calculatePlaq();

      // Gauge for contractions
      contractGauge.copy(gauge);
      // apply boundary conditions since is needed for the covariant derivative
      applyBoundaryConditions(contractGauge,true);
    }
    
    updateOptions(LIGHT);
    TIME(QUDA_solver solver(mu));

    std::string given_twop_filename = twop_filename;
    for(int isource = startSource; isource < endSource; isource++){
      site& source = sourcePositions[isource];
      PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
                    isource, source[0], source[1], source[2], source[3]);
      
      PLEGMA_Gauge3D<double> smearedGauge3D;
      smearedGauge3D.absorb(smearedGauge, source[DIM_T]);

      auto computePropagator = [&](PLEGMA_Propagator<float>& prop_SS, PLEGMA_Propagator<float>& prop_SL,
				   double run_mu, WHICHFLAVOR fl, int nSmear0, int nSmear1) {
				 // ensuring mu value
				 if(mu != run_mu) {
				   updateOptions(fl);
				   mu = run_mu;
				   solver.UpdateSolver();
				 }
				 for(int isc = 0 ; isc < 12 ; isc++){
				   PLEGMA_Vector<double> vectorInOut;
				   { // Smearing the source
				     PLEGMA_Vector3D<double> vector1, vector2;
				     vector1.pointSource(source, isc/3, isc%3, DEVICE);
				     TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nSmear0, alphaGauss));
				     vectorInOut.absorb(vector2,source[DIM_T]);
				   }
				   // Inverting
				   PLEGMA_printf("Going to invert %s for component %d\n",
						 fl==LIGHT ? "LIGHT" : (fl == STRANGE ? "STRANGE" : "CHARM"), isc);
				   TIME(solver.solve(vectorInOut, vectorInOut));
				   if(prop_SL.getAllocation() != NONE) {
				     PLEGMA_Vector<float> vectorAuxF;
				     vectorAuxF.copy(vectorInOut);
				     prop_SL.absorb(vectorAuxF, isc/3, isc%3);
				   }
				   { // Smearing the solution
				     PLEGMA_Vector<double> vectorAuxD;
				     PLEGMA_Vector<float> vectorAuxF;
				     TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nSmear1, alphaGauss));
				     vectorAuxF.copy(vectorAuxD);
				     prop_SS.absorb(vectorAuxF, isc/3, isc%3);
				   }
				 }
			       };
      
      char * src_string;
      asprintf(&src_string, "_sx%02dsy%02dsz%02dst%03d", source[0], source[1], source[2], source[3]);
      twop_filename = given_twop_filename + src_string;
      free(src_string);
      
      PLEGMA_Propagator<float> propUP_wrong_smear;
      PLEGMA_Propagator<float> propUP;
      PLEGMA_Propagator<float> propST_wrong_smear;
      PLEGMA_Propagator<float> propST;
      { // Whithin this scope we keep track also of the propagator non smeared on the sink
	//PLEGMA_Propagator<float> propUP_SL(tSinks.size()>0 ? BOTH:NONE, FIRST_CORNER);
	//PLEGMA_Propagator<float> propST_SL(tSinks.size()>0 ? BOTH:NONE, FIRST_CORNER);
	PLEGMA_Propagator<float> propUP_SL(BOTH, FIRST_CORNER);
	PLEGMA_Propagator<float> propST_SL(BOTH, FIRST_CORNER);

	TIME(computePropagator(propUP, propUP_SL, mu_ud, LIGHT, nsmearGauss, nsmearGauss));
	//TIME(computePropagator(propST, propST_SL, mu_s, STRANGE, nsmearGauss_s, nsmearGauss_s));
	// Correcting the smearing
	for(int isc = 0 ; isc < 12 ; isc++){
	  PLEGMA_Vector<double> vectorAuxD, vectorAuxD2;
	  PLEGMA_Vector<float> vectorAuxF;
	  vectorAuxF.absorb(propUP_SL, isc/3, isc%3);
	  vectorAuxD2.copy(vectorAuxF);
	  TIME(vectorAuxD.gaussianSmearing(vectorAuxD2, smearedGauge, nsmearGauss_s, alphaGauss));
	  vectorAuxF.copy(vectorAuxD);
	  propUP_wrong_smear.absorb(vectorAuxF, isc/3, isc%3);
	}
	// Correcting the smearing
	/*
	for(int isc = 0 ; isc < 12 ; isc++){
	  PLEGMA_Vector<double> vectorAuxD, vectorAuxD2;
	  PLEGMA_Vector<float> vectorAuxF;
	  vectorAuxF.absorb(propST_SL, isc/3, isc%3);
	  vectorAuxD2.copy(vectorAuxF);
	  TIME(vectorAuxD.gaussianSmearing(vectorAuxD2, smearedGauge, nsmearGauss, alphaGauss));
	  vectorAuxF.copy(vectorAuxD);
	  propST_wrong_smear.absorb(vectorAuxF, isc/3, isc%3);
	  }
	*/
      }
      propUP_wrong_smear.rotateToPhysicalBase_device(mu_ud/abs(mu_ud));
      //propST_wrong_smear.rotateToPhysicalBase_device(mu_s/abs(mu_s));
      propUP.rotateToPhysicalBase_device(mu_ud/abs(mu_ud));
      //propST.rotateToPhysicalBase_device(mu_s/abs(mu_s));
      propUP_wrong_smear.applyBoundaries_device(source[3]);
      //propST_wrong_smear.applyBoundaries_device(source[3]);
      propUP.applyBoundaries_device(source[3]);
      //propST.applyBoundaries_device(source[3]);

      {
	PLEGMA_Correlator<float> corr(corr_space, source, maxQsq);
	TIME(corr.contractMesonsAll(propUP, propUP));
	corr.setDatasets((std::vector<std::string>) {"twop_meson_uu"});
	THREAD(corr.writeFile(twop_filename, corr_file_format));
	/*
	TIME(corr.contractMesonsNew(propUP, propST));
	corr.setDatasets((std::vector<std::string>) {"twop_meson_us"});
	THREAD(corr.writeFile(twop_filename, corr_file_format));
	
	TIME(corr.contractMesonsNew(propUP_wrong_smear, propST));
	corr.setDatasets((std::vector<std::string>) {"twop_meson_u's"});
	THREAD(corr.writeFile(twop_filename, corr_file_format));
	
	TIME(corr.contractMesonsNew(propUP, propST_wrong_smear));
	corr.setDatasets((std::vector<std::string>) {"twop_meson_us'"});
	THREAD(corr.writeFile(twop_filename, corr_file_format));

      	TIME(corr.contractMesonsNew(propUP_wrong_smear, propST_wrong_smear));
	corr.setDatasets((std::vector<std::string>) {"twop_meson_u's'"});
	THREAD(corr.writeFile(twop_filename, corr_file_format));
	*/
      }
    }
    while(not threads.empty()) {threads.back().join(); threads.pop_back();}
  }
    
  finalize();
  return 0;
}
  
