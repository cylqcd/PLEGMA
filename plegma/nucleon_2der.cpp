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
					    "nsrc", "src-filename", "maxQsq", "twop-filename", "corr-file-format", "corr-space", "tSinks","Projs", "threep-filename"};
  
int main(int argc, char **argv) {
  initializeOptions(argc, argv, true, listOpt);
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
  double mu_ud = mu;
  double mu_ud_factor[QUDA_MAX_MG_LEVEL];
  for(int i=0;i<QUDA_MAX_MG_LEVEL;i++) mu_ud_factor[i] = mu_factor[i];
  int startSource = 0;
  std::string prOrNt = "neutron";
  std::string srcInputFile = "./input.src";
  std::string mom_path = "%+d_%+d_%+d.txt";
  std::string mom_path2 = "%+d_%+d_%+d.txt";
  std::string mom_path3 = "%+d_%+d_%+d.txt";

  auto add_options = [&](Options& options) {
    options.set("whichParticle", "Which particle we want to do the 3pf. Options (proton, neutron)", verbosity, prOrNt);
    options.set("src-input-file", "Use the file to update option at every source. The file searched is [src-input-file]+str(n) where n is the source (0, 1, ...)", verbosity, srcInputFile);
    options.set("start-src", "The index of the source position where to start the calculation", verbosity, startSource);
    options.set("mompath", "Path to the list of momenta", verbosity, mom_path);  
    options.set("mompath2", "Path to the list of momenta", verbosity, mom_path2);  
    options.set("mompath3", "Path to the list of momenta", verbosity, mom_path3);  
		     };
  add_options(*HGC_options);
  if(prOrNt != "proton" && prOrNt != "neutron") PLEGMA_error("This exec is only for nucleon, %s is not allowed",prOrNt.c_str());
  //=========================================================================================================//
  initializePLEGMA();

  {
    PLEGMA_Gauge<double> smearedGauge(BOTH);
    PLEGMA_Gauge<float> contractGauge(BOTH, FIRST_VERTEX);
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

      for(int imom=0; imom<12; imom++) {

      std::vector<int> sinkMom = {0,0,0};
      sinkMom[(imom/4)%3] = (imom%2)*2-1;       //-+-+ in positions 1,2,3
      sinkMom[(imom/4+1)%3] = ((imom/2)%2)*2-1; //--++ in positions 2,3,1

      momenta.clear();
      char * tmp;
      asprintf(&tmp, mom_path.c_str(), sinkMom[0],sinkMom[1],sinkMom[2]);
      pathListMomenta = tmp;
      free(tmp);
      readMomentaList();
      
      std::vector<std::vector<int>> momenta2; 
      std::vector<std::vector<int>> momenta3;
      for(int im=0;im<momenta.size();im++) {
	momentum v1 = momenta[im];
	std::vector<int> v2 = {v1[0],v1[1],v1[2]};
	std::vector<int> v3 = {sinkMom[0]-v1[0],sinkMom[1]-v1[1],sinkMom[2]-v1[2]};
	momenta2.push_back(v2);
	momenta3.push_back(v3);
      }  

      momenta.clear();
      asprintf(&tmp, mom_path2.c_str(), sinkMom[0],sinkMom[1],sinkMom[2]);
      pathListMomenta = tmp;
      free(tmp);
      readMomentaList();
      
      std::vector<std::vector<int>> momenta4;
      for(int im=0;im<momenta.size();im++) {
	momentum v1 = momenta[im];
	std::vector<int> v3 = {sinkMom[0]-v1[0],sinkMom[1]-v1[1],sinkMom[2]-v1[2]};
	momenta4.push_back(v3);
      }  

      momenta.clear();
      asprintf(&tmp, mom_path3.c_str(), sinkMom[0],sinkMom[1],sinkMom[2]);
      pathListMomenta = tmp;
      free(tmp);
      readMomentaList();
      
      std::vector<std::vector<int>> momenta5;
      for(int im=0;im<momenta.size();im++) {
	momentum v1 = momenta[im];
	std::vector<int> v3 = {sinkMom[0]-v1[0],sinkMom[1]-v1[1],sinkMom[2]-v1[2]};
	momenta5.push_back(v3);
      }  
      
      PLEGMA_printf("\n ### Sink Momentum px %d, py %d, pz %d\n", sinkMom[0],sinkMom[1],sinkMom[2]);

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
      asprintf(&src_string, "_sx%02dsy%02dsz%02dst%03d_px%+dpy%+dpz%+d",
	       source[0], source[1], source[2], source[3], sinkMom[0], sinkMom[1], sinkMom[2]);
      twop_filename = given_twop_filename + src_string + ".h5";
      threep_filename = given_threep_filename + src_string;
      free(src_string);
      
      PLEGMA_Propagator<float> propUP;
      PLEGMA_Propagator<float> propDN;
      { // Whithin this scope we keep track also of the propagator non smeared on the sink
	PLEGMA_Propagator<float> propUP_SL(tSinks.size()>0 ? BOTH:NONE, FIRST_VERTEX);
	PLEGMA_Propagator<float> propDN_SL(tSinks.size()>0 ? BOTH:NONE, FIRST_VERTEX);

	bool computed_light = false;
	// If twop_filename exists we hold the computation of the light props
	if(access( twop_filename.c_str(), F_OK ) == -1) {
	  TIME(computePropagator(propUP, propUP_SL, mu_ud, LIGHT, nsmearGauss, false));
	  TIME(computePropagator(propDN, propDN_SL, -mu_ud, LIGHT, nsmearGauss, false));
	  computed_light = true;
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
	  for(size_t iproj = 0; iproj < Projs.size(); iproj++){
	    auto computeThreep = [&](double run_mu, PLEGMA_Propagator<float>& prop1, PLEGMA_Propagator<float>& prop2, int signProps, PLEGMA_Propagator<float> &propF, PLEGMA_Propagator<float> &propF2, std::string fl) {
	      std::string filename = threep_filename + "_" + Projs[iproj] + "_dt" + std::to_string(tsinkMtsource) + "_" + fl + ".h5";
	      if(access( filename.c_str(), F_OK ) != -1) {
		PLEGMA_printf("File %s already exists. Skipping...", filename.c_str());
		return;
	      }
	      if(not computed_light) {
		TIME(computePropagator(propUP, propUP_SL, mu_ud, LIGHT, nsmearGauss, false));
		TIME(computePropagator(propDN, propDN_SL, -mu_ud, LIGHT, nsmearGauss, false));
		computed_light = true;
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
		PLEGMA_Propagator3D<float> prop23D;
		prop13D.absorb(prop1, global_fixSinkTime);
		prop23D.absorb(prop2, global_fixSinkTime);
		PLEGMA_Gauge3D<double> smearedGauge3D_sink;
		smearedGauge3D_sink.absorb(smearedGauge, global_fixSinkTime);

	      for(int nu = 0 ; nu < 4 ; nu++)
		for(int c2 = 0 ; c2 < 3 ; c2++){
		  PLEGMA_Vector<double> vectorInOut;
		  {
		    PLEGMA_Vector3D<double> vectorAuxD1,vectorAuxD2;
		    PLEGMA_Vector3D<float> vectorAuxF;
		    if(&prop1 != &prop2)
		      vectorAuxF.seqSourceNucleon(prop13D, prop23D, get_projector(Projs[iproj]), nucleon, nu, c2);
		    else
		      vectorAuxF.seqSourceNucleon(prop13D, get_projector(Projs[iproj]), nucleon, nu, c2);
					 
		    vectorAuxF.mulMomentumPhases(sinkMom,-1); // put momentum at the sink
		    std::complex<float> Isingle(0,1);
		    float phase = 2.*PI*(((float) sinkMom[0] * sourcePositions[isource][0])/HGC_totalL[0]
					 + ((float)sinkMom[1] * sourcePositions[isource][1])/HGC_totalL[1]
					 + ((float)sinkMom[2] * sourcePositions[isource][2])/HGC_totalL[2]);
		    vectorAuxF.cscale(std::exp<float>(+phase*Isingle)); // put momentum from the point source
		    vectorAuxF.conjugate();
		    vectorAuxF.apply_gamma(G5);
		    vectorAuxD1.copy(vectorAuxF);
		    TIME(vectorAuxD2.gaussianSmearing(vectorAuxD1,smearedGauge3D_sink, nsmearGauss, alphaGauss));
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
				     
	      PLEGMA_Correlator<float> corr(corr_space, source, 0, tsinkMtsource+1);
	      corr.setFixMomList(momenta3);
	      PLEGMA_Correlator<float> corr2(corr_space, source, 0, tsinkMtsource+1);
	      corr2.setFixMomList(momenta4);
	      PLEGMA_Correlator<float> corr3(corr_space, source, 0, tsinkMtsource+1);
	      corr3.setFixMomList(momenta5);
	  
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
	      TIME(corr2.contractNucleonThrp_twoD(seqProp, propF, contractGauge, signProps, gammas));
	      if(signPer < 0) for(size_t iv = 0 ; iv < corr2.getTotalSize()*2; iv++) corr2.H_elem()[iv] *= signPer;
	      THREAD(corr2.writeFile( filename, corr_file_format));

	      // THREED contractions
	      TIME(corr3.contractNucleonThrp_threeD(seqProp, propF, contractGauge, signProps, gammas));
	      if(signPer < 0) for(size_t iv = 0 ; iv < corr3.getTotalSize()*2; iv++) corr3.H_elem()[iv] *= signPer;
	      THREAD(corr3.writeFile( filename, corr_file_format));

	      // LOCAL contractions
	      TIME(corr.contractNucleonThrp_local(seqProp, propF2, signProps, gammas));
	      if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) corr.H_elem()[iv] *= signPer;      
	      corr.setDatasets((std::vector<std::string>) {"threep_OS"});
	      THREAD(corr.writeFile(filename, corr_file_format));
				     
	      // ONED contractions
	      TIME(corr.contractNucleonThrp_oneD(seqProp, propF2, contractGauge, signProps, gammas));
	      if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) corr.H_elem()[iv] *= signPer;
	      corr.setDatasets((std::vector<std::string>) {"threep_OS"});
	      THREAD(corr.writeFile( filename, corr_file_format));
				     
	      // noe contractions
	      TIME(corr.contractNucleonThrp_noe(seqProp, propF2, contractGauge, signProps));
	      if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) corr.H_elem()[iv] *= signPer;
	      corr.setDatasets((std::vector<std::string>) {"threep_OS"});
	      THREAD(corr.writeFile( filename, corr_file_format));
	    };
	    if(nucleon == PROTON) {
	      TIME(computeThreep(-mu_ud, propUP, propDN, +1, propUP_SL, propDN_SL, "up"));
	      TIME(computeThreep( mu_ud, propUP, propUP, -1, propDN_SL, propUP_SL, "dn"));
	    } else {
	      TIME(computeThreep( mu_ud, propDN, propUP, -1, propDN_SL, propUP_SL, "dn"));
	      TIME(computeThreep(-mu_ud, propDN, propDN, +1, propUP_SL, propDN_SL, "up"));
	    }
	  }
	}
#endif
      }
      // If twop_filename exists we skip the rest
      if(access( twop_filename.c_str(), F_OK ) != -1) {
	PLEGMA_printf("File %s already exists. Skipping...", twop_filename.c_str());
	continue;
      }
      
      propUP.rotateToPhysicalBase_device(+1);
      propDN.rotateToPhysicalBase_device(-1);
      propUP.applyBoundaries_device(source[3]);
      propDN.applyBoundaries_device(source[3]);
      
      {
	PLEGMA_Correlator<float> corr(corr_space, source, maxQsq);
	TIME(corr.contractMesonsNew(propUP, propDN));
	char *dset;
	asprintf(&dset, "twop_mesons_new_u[%+1.1e]d[%+1.1e]", mu_ud, -1*mu_ud);
	corr.setDatasets((std::vector<std::string>) {dset});
	free(dset);
	THREAD(corr.writeFile(twop_filename, corr_file_format));

	TIME(corr.contractMesonsNew(propUP, propUP));
	asprintf(&dset, "twop_mesons_new_u[%+1.1e]u[%+1.1e]", mu_ud, mu_ud);
	corr.setDatasets((std::vector<std::string>) {dset});
	free(dset);
	THREAD(corr.writeFile(twop_filename, corr_file_format));

	TIME(corr.contractMesonsNew(propDN, propDN));
	asprintf(&dset, "twop_mesons_new_d[%+1.1e]d[%+1.1e]", -mu_ud, -mu_ud);
	corr.setDatasets((std::vector<std::string>) {dset});
	free(dset);
	THREAD(corr.writeFile(twop_filename, corr_file_format));

	
	TIME(corr.contractBaryons(propUP, propDN));
	THREAD(corr.writeFile(twop_filename, corr_file_format));
      }


#ifdef PLEGMA_UDSC_BARYONS
      PLEGMA_Propagator<float> none(NONE);
      PLEGMA_Correlator<float> corr(corr_space, source, 0);
      corr.setFixMomList(momenta2);

      TIME(corr.contractBaryonsUDSC(propUP, propDN, none, none));
      char * group;
      
      asprintf(&group, "baryons_u[%+1.1e]d[%+1.1e]", mu_ud, -1*mu_ud);
      corr.setGroups(group);
      free(group);
      THREAD(corr.writeFile(twop_filename, corr_file_format));
#endif
    }
    }
    while(not threads.empty()) {threads.back().join(); threads.pop_back();}
  }

  //finalize();
  return 0;
}

