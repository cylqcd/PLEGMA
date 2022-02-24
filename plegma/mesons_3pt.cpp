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
  double mu_s;
  double mu_ud = mu;
  double mu_ud_factor[QUDA_MAX_MG_LEVEL];
  for(int i=0;i<QUDA_MAX_MG_LEVEL;i++) mu_ud_factor[i] = mu_factor[i];
  int nsmearGauss_s = nsmearGauss/2;
  int startSource = 0;
  std::string srcInputFile = "./input.src";
  std::vector<int> sourceMom = {0,0,0};
    auto add_options = [&](Options& options) {
    options.set("mu-s", "List of mu_s to run for the strange quark in baryons", verbosity, mu_s);
    options.set("nsmear-gauss-s", "Number of Gaussian smearing step for the strange quark propagator", verbosity, nsmearGauss_s);
    options.set("src-input-file", "Use the file to update option at every source. The file searched is [src-input-file]+str(n) where n is the source (0, 1, ...)", verbosity, srcInputFile);
    options.set("start-src", "The index of the source position where to start the calculation", verbosity, startSource);
    options.set("source-mom", "The list of momenta components at the source. Every three makes a momentum", verbosity, sourceMom);
		     };
  add_options(*HGC_options);
  //=========================================================================================================//
  // Only multiple of three accepted
  assert(sourceMom.size()%3==0);
  
  initializePLEGMA();

  {
    PLEGMA_Gauge<double> smearedGauge(BOTH);
    PLEGMA_Gauge<float> contractGauge(BOTH);
    {
      // Reading from Lime file and loading to device
      PLEGMA_Gauge<double> gauge;
      gauge.readFile(latfile, LIME_FORMAT);
      gauge.load();
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
    std::string given_threep_filename = threep_filename;
    
    for(int isource = startSource; isource < numSourcePositions; isource++){
      site& source = sourcePositions[isource];
      PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
		    isource, source[0], source[1], source[2], source[3]);
      updateOptions(srcInputFile + std::to_string(isource), listOpt, add_options);

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
      threep_filename = given_threep_filename + src_string;
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
	TIME(computePropagator(propST, propST_SL, mu_s, STRANGE, nsmearGauss_s, nsmearGauss_s));
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
	for(int isc = 0 ; isc < 12 ; isc++){
	  PLEGMA_Vector<double> vectorAuxD, vectorAuxD2;
	  PLEGMA_Vector<float> vectorAuxF;
	  vectorAuxF.absorb(propST_SL, isc/3, isc%3);
	  vectorAuxD2.copy(vectorAuxF);
	  TIME(vectorAuxD.gaussianSmearing(vectorAuxD2, smearedGauge, nsmearGauss, alphaGauss));
	  vectorAuxF.copy(vectorAuxD);
	  propST_wrong_smear.absorb(vectorAuxF, isc/3, isc%3);
	}
	
	for(size_t its = 0; its < tSinks.size(); its++){
	  int tsinkMtsource = tSinks[its];
	  if(tsinkMtsource >= HGC_totalL[3])
	    PLEGMA_error("Provided tsink=%d is >= than temporal extent",tsinkMtsource);
	  int signPer = (tsinkMtsource+source[3]) >= HGC_totalL[3] ? -1 : +1;
	  int global_fixSinkTime = (tsinkMtsource + source[3])%HGC_totalL[3]; 

	  // 3D propagators at t_sink
	  PLEGMA_Propagator3D<float> propUP3D;
	  PLEGMA_Propagator3D<float> propST3D;
	  PLEGMA_Gauge3D<double> smearedGauge3D_sink;
	  propUP3D.absorb(propUP, global_fixSinkTime);
	  propST3D.absorb(propST, global_fixSinkTime);
	  smearedGauge3D_sink.absorb(smearedGauge, global_fixSinkTime);

	  std::vector<GAMMAS> gammas = {ONE,G1,G2,G3,G4,G5,G5G1,G5G2,G5G3,G5G4,S12,S13,S23,S41,S42,S43};
	  for(size_t imom = 0; imom < sourceMom.size()/3; imom++){
	    std::vector<int> sinkMom = {sourceMom[imom*3],sourceMom[imom*3+1],sourceMom[imom*3+2]};

	    auto computeThreep = [&](double run_mu, PLEGMA_Propagator3D<float>& prop, PLEGMA_Propagator<float> &propF, int nSmear, WHICHFLAVOR fl, std::string name) {
	      char * mom_string;
	      asprintf(&mom_string, "_mx%+dmy%+dsz%+d", sinkMom[0], sinkMom[1], sinkMom[2]);
	      std::string filename = threep_filename + mom_string + "_dt" + std::to_string(tsinkMtsource)+name;
	      free(mom_string);
	      
  	      int signProps = -run_mu/abs(run_mu);		   
	      PLEGMA_Propagator<float> seqProp(BOTH, FIRST_CORNER);
	      // ensuring mu positive
	      if(mu != run_mu) {
		updateOptions(fl);
		mu = run_mu;
		solver.UpdateSolver();
	      }
				     
	      for(int nu = 0 ; nu < 4 ; nu++)
		for(int c2 = 0 ; c2 < 3 ; c2++){
		  PLEGMA_Vector<double> vectorInOut;
		  {
		    PLEGMA_Vector3D<double> vectorAuxD1, vectorAuxD2;
		    PLEGMA_Vector3D<float> vectorAuxF;
		    vectorAuxF.absorb(prop, nu, c2);
		    vectorAuxF.apply_gamma(G5);
		    vectorAuxF.mulMomentumPhases(sinkMom,+1); // put momentum at the sink
		    std::complex<float> Isingle(0,1);
		    float phase = 2.*PI*(((float) sinkMom[0] * source[0])/HGC_totalL[0]
					 + ((float) sinkMom[1] * source[1])/HGC_totalL[1]
					 + ((float) sinkMom[2] * source[2])/HGC_totalL[2]);
		    vectorAuxF.cscale(std::exp<float>(-phase*Isingle)); // put momentum from the point source
		    vectorAuxD1.copy(vectorAuxF);
		    TIME(vectorAuxD2.gaussianSmearing(vectorAuxD1,smearedGauge3D_sink, nSmear, alphaGauss));
		    vectorInOut.absorb(vectorAuxD2, global_fixSinkTime);
		  }
		  double norm = vectorInOut.norm();
		  vectorInOut.scale(1/norm);
		  TIME(solver.solve(vectorInOut, vectorInOut));
		  vectorInOut.scale(norm);
		  PLEGMA_Vector<float> vectorAuxF;
		  vectorAuxF.copy(vectorInOut);
		  seqProp.absorb(vectorAuxF, nu, c2);
		}
	      seqProp.apply_gamma(G5);
	      seqProp.conjugate();
				     
	      PLEGMA_Correlator<float> corr(corr_space, source, maxQsq, tsinkMtsource+1);
	  
	      // LOCAL contractions
	      TIME(corr.contractNucleonThrp_local(seqProp, propF, signProps, gammas));
	      if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) corr.H_elem()[iv] *= signPer;      
	      THREAD(corr.writeFile(filename, corr_file_format));
				     
	      // ONED contractions
	      TIME(corr.contractNucleonThrp_oneD(seqProp, propF, contractGauge, signProps, gammas));
	      if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) corr.H_elem()[iv] *= signPer;
	      THREAD(corr.writeFile( filename, corr_file_format));
				     
	      // noe contractions
	      TIME(corr.contractNucleonThrp_noe(seqProp, propF, contractGauge, signProps));
	      if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) corr.H_elem()[iv] *= signPer;
	      THREAD(corr.writeFile( filename, corr_file_format));
	      
	      // TWOD contractions
	      TIME(corr.contractNucleonThrp_twoD(seqProp, propF, contractGauge, signProps, gammas));
	      if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) corr.H_elem()[iv] *= signPer;
	      THREAD(corr.writeFile( filename, corr_file_format));
	    };
	    TIME(computeThreep(-mu_ud, propUP3D, propUP_SL, nsmearGauss, LIGHT, "_up_pion"));
	    TIME(computeThreep(-mu_ud, propST3D, propUP_SL, nsmearGauss, LIGHT, "_up_kaon"));
	    TIME(computeThreep(-mu_s, propUP3D, propST_SL, nsmearGauss_s, STRANGE, "_st_kaon"));
	  }
	}
      }
      propUP_wrong_smear.rotateToPhysicalBase_device(mu_ud/abs(mu_ud));
      propST_wrong_smear.rotateToPhysicalBase_device(mu_s/abs(mu_s));
      propUP.rotateToPhysicalBase_device(mu_ud/abs(mu_ud));
      propST.rotateToPhysicalBase_device(mu_s/abs(mu_s));
      propUP_wrong_smear.applyBoundaries_device(source[3]);
      propST_wrong_smear.applyBoundaries_device(source[3]);
      propUP.applyBoundaries_device(source[3]);
      propST.applyBoundaries_device(source[3]);

      {
	PLEGMA_Correlator<float> corr(corr_space, source, maxQsq);
	TIME(corr.contractMesonsNew(propUP, propUP));
	corr.setDatasets((std::vector<std::string>) {"twop_meson_uu"});
	THREAD(corr.writeFile(twop_filename, corr_file_format));

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
      }
    }
    while(not threads.empty()) {threads.back().join(); threads.pop_back();}
  }
    
  finalize();
  return 0;
}
  
