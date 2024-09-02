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
					    "nsrc", "src-filename", "maxQsq", "twop-filename", "corr-file-format", "corr-space", "tSinks","Projs", "threep-filename", "momlist-filename"};
  
int main(int argc, char **argv) {
  initializeOptions(argc, argv, true, listOpt);
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
  double mu_s;
  double mu_ud = mu;
  double mu_ud_factor[QUDA_MAX_MG_LEVEL];
  for(int i=0;i<QUDA_MAX_MG_LEVEL;i++) mu_ud_factor[i] = mu_factor[i];
  int nsmearGauss_s = nsmearGauss/2;
  int startSource = 0;
  std::string prOrNt = "neutron";
  std::string srcInputFile = "./input.src";
  auto add_options = [&](Options& options) {
    options.set("mu-s", "List of mu_s to run for the strange quark in baryons", verbosity, mu_s);
    options.set("nsmear-gauss-s", "Number of Gaussian smearing step for the strange quark propagator", verbosity, nsmearGauss_s);
    options.set("whichParticle", "Which particle we want to do the 3pf. Options (proton, neutron)", verbosity, prOrNt);
    options.set("src-input-file", "Use the file to update option at every source. The file searched is [src-input-file]+str(n) where n is the source (0, 1, ...)", verbosity, srcInputFile);
    options.set("start-src", "The index of the source position where to start the calculation", verbosity, startSource);
		     };
  add_options(*HGC_options);
  if(prOrNt != "proton" && prOrNt != "neutron") PLEGMA_error("This exec is only for nucleon, %s is not allowed",prOrNt.c_str());
  //=========================================================================================================//
  initializePLEGMA();

  {
    PLEGMA_Gauge<double> smearedGauge(BOTH);
    PLEGMA_Gauge<float> contractGauge(BOTH);
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

      auto computePropagator = [&](PLEGMA_Propagator<float>& prop_SS, PLEGMA_Propagator<float>& prop_SL,
				   double run_mu, WHICHFLAVOR fl, int nSmear, bool finalize) {
				 PLEGMA_Gauge3D<double> smearedGauge3D;
				 smearedGauge3D.absorb(smearedGauge, source[DIM_T]);

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
				     TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nSmear, alphaGauss));
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
				     TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nSmear, alphaGauss));
				     vectorAuxF.copy(vectorAuxD);
				     prop_SS.absorb(vectorAuxF, isc/3, isc%3);
				   }
				 }
				 if(finalize) {
				   prop_SS.rotateToPhysicalBase_device(run_mu/abs(run_mu));
				   prop_SS.applyBoundaries_device(source[DIM_T]);
				 }
			       };

      char * src_string;
      asprintf(&src_string, "_sx%02dsy%02dsz%02dst%03d", source[0], source[1], source[2], source[3]);
      twop_filename = given_twop_filename + src_string + ".h5";
      threep_filename = given_threep_filename + src_string;
      free(src_string);
      
      PLEGMA_Propagator<float> propUP;
      PLEGMA_Propagator<float> propST;
      
      { // Whithin this scope we keep track also of the propagator non smeared on the sink
	PLEGMA_Propagator<float> propUP_SL(tSinks.size()>0 ? BOTH:NONE, FIRST_VERTEX);
	PLEGMA_Propagator<float> propST_SL(tSinks.size()>0 ? BOTH:NONE, FIRST_VERTEX);

	// If twop_filename exists we hold the computation of the light props
	if(access( twop_filename.c_str(), F_OK ) == -1) {
	  TIME(computePropagator(propUP, propUP_SL, mu_ud, LIGHT, nsmearGauss, false));
	  TIME(computePropagator(propST, propST_SL, mu_s, LIGHT, nsmearGauss_s, false));
	}
	
	{
	  PLEGMA_Correlator<float> corr(corr_space, source, maxQsq);
	  TIME(corr.contractMesonsNew(propUP, propUP));
	  char *dset;
	  asprintf(&dset, "pion", mu_ud);
	  corr.setDatasets((std::vector<std::string>) {dset});
	  free(dset);
	  THREAD(corr.writeFile(twop_filename, corr_file_format));
	  
	  TIME(corr.contractMesonsNew(propUP, propST));
	  asprintf(&dset, "kaon", mu_ud, mu_s);
	  corr.setDatasets((std::vector<std::string>) {dset});
	  free(dset);
	  THREAD(corr.writeFile(twop_filename, corr_file_format));
	}
      
#ifdef PLEGMA_NUCLEON_3PF_FIX_SINK
	for(size_t its = 0; its < tSinks.size(); its++){
	  int tsinkMtsource = tSinks[its];
	  if(tsinkMtsource >= HGC_totalL[3])
	    PLEGMA_error("Provided tsink=%d is >= than temporal extent",tsinkMtsource);
	  int signPer = (tsinkMtsource+source[3]) >= HGC_totalL[3] ? -1 : +1;
	  int global_fixSinkTime = (tsinkMtsource + source[3])%HGC_totalL[3]; 


	  WHICHPARTICLE nucleon = get_particle(prOrNt); 
	  std::vector<GAMMAS> gammas = {ONE,G1,G2,G3,G4,G5,G5G1,G5G2,G5G3,G5G4,S12,S13,S23,S41,S42,S43};
	  for(size_t imom = 0; imom < momenta.size(); imom++){
	    auto sinkMom = momenta[imom];
	    
	    auto computeThreep = [&](double run_mu, PLEGMA_Propagator<float>& prop1, PLEGMA_Propagator<float>& prop2, int signProps, std::string fl, int nsmear) {
				   
	      char * mom_string;
	      asprintf(&mom_string, "_mx%+01dmy%+01dmz%+01d", sinkMom[0], sinkMom[1], sinkMom[2]);
	      std::string filename = threep_filename + mom_string + "_dt" + std::to_string(tsinkMtsource) + "_" + fl + ".h5";
	      free(mom_string);
	      
	      if(access( filename.c_str(), F_OK ) != -1) {
		PLEGMA_printf("File %s already exists. Skipping...", filename.c_str());
		return;
	      }
	      
	      PLEGMA_Propagator<float> seqProp(BOTH, FIRST_VERTEX);
	      // ensuring mu positive
	      if(mu != run_mu) {
		updateOptions(LIGHT);
		mu = run_mu;
		solver.UpdateSolver();
	      }
	      
	      {
		// 3D propagators at t_sink
		PLEGMA_Propagator3D<float> prop13D;
		prop13D.absorb(prop1, global_fixSinkTime);

		PLEGMA_Gauge3D<double> smearedGauge3D_sink;
		smearedGauge3D_sink.absorb(smearedGauge, global_fixSinkTime);

	      for(int nu = 0 ; nu < 4 ; nu++)
		for(int c2 = 0 ; c2 < 3 ; c2++){
		  PLEGMA_Vector<double> vectorInOut;
		  {
		    PLEGMA_Vector3D<double> vectorAuxD1,vectorAuxD2;
		    PLEGMA_Vector3D<float> vectorAuxF;

		    vectorAuxF.absorb(prop13D, nu, c2);
		    vectorAuxF.apply_gamma(G5);
		    
		    vectorAuxF.mulMomentumPhases(std::vector<int>(std::begin(sinkMom), std::end(sinkMom)),-1); // put momentum at the sink
		    std::complex<float> Isingle(0,1);
		    float phase = 2.*PI*(((float) sinkMom[0] * source[0])/HGC_totalL[0]
					 + ((float)sinkMom[1] * source[1])/HGC_totalL[1]
					 + ((float)sinkMom[2] * source[2])/HGC_totalL[2]);
		    vectorAuxF.cscale(std::exp<float>(+phase*Isingle)); // put momentum from the point source

		    vectorAuxD1.copy(vectorAuxF);
		    TIME(vectorAuxD2.gaussianSmearing(vectorAuxD1,smearedGauge3D_sink, nsmear, alphaGauss));
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
	      }
	      seqProp.apply_gamma(G5);
	      seqProp.conjugate();
				     
	      PLEGMA_Correlator<float> corr(corr_space, source, maxQsq, tsinkMtsource+1);
	      
	      // LOCAL contractions
	      TIME(corr.contractNucleonThrp_local(seqProp, prop2, signProps, gammas));
	      if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) corr.H_elem()[iv] *= signPer;      
	      THREAD(corr.writeFile(filename, corr_file_format));
				     
	      // ONED contractions
	      TIME(corr.contractNucleonThrp_oneD(seqProp, prop2, contractGauge, signProps, gammas));
	      if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) corr.H_elem()[iv] *= signPer;
	      THREAD(corr.writeFile( filename, corr_file_format));

	      // TWOD contractions
	      TIME(corr.contractNucleonThrp_twoD(seqProp, prop2, contractGauge, signProps, gammas));
	      if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) corr.H_elem()[iv] *= signPer;
	      THREAD(corr.writeFile( filename, corr_file_format));

	      // THREED contractions
	      TIME(corr.contractNucleonThrp_threeD(seqProp, prop2, contractGauge, signProps, gammas));
	      if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) corr.H_elem()[iv] *= signPer;
	      THREAD(corr.writeFile( filename, corr_file_format));

	      // noe contractions
	      TIME(corr.contractNucleonThrp_noe(seqProp, prop2, contractGauge, signProps));
	      if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) corr.H_elem()[iv] *= signPer;
	      THREAD(corr.writeFile( filename, corr_file_format));

	      
	    };

	    // TODO check sign
	    TIME(computeThreep(-mu_ud, propUP, propUP_SL, +1, "pion", nsmearGauss));
	    TIME(computeThreep(-mu_ud, propST, propUP_SL, +1, "kaon_up", nsmearGauss));
	    TIME(computeThreep(-mu_s, propUP, propST_SL, +1, "kaon_st", nsmearGauss_s));
	  }
	}
#endif
      }
      // If twop_filename exists we skip the rest
      if(access( twop_filename.c_str(), F_OK ) != -1) {
	PLEGMA_printf("File %s already exists. Skipping...", twop_filename.c_str());
	continue;
      }
      
    }
    while(not threads.empty()) {threads.back().join(); threads.pop_back();}
  }

  //finalize();
  return 0;
}

