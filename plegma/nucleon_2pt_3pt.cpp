#include <PLEGMA.h>
#include <PLEGMA_utils.h>

double runtime;
#define TIME(fnc)  runtime = MPI_Wtime(); fnc; runtime = MPI_Wtime()-runtime; \
  PLEGMA_printf("TIME for "#fnc" %lf sec\n", runtime)

std::vector<std::thread> threads;
//#define THREAD(fnc) threads.push_back(std::thread([=]() { TIME(fnc); }))
#define THREAD(fnc) saveTuneCache(false); TIME(fnc)

using namespace plegma;
using namespace quda;
static std::vector<std::string> listOpt = { "verbosity", "load-gauge", "nsmear-APE", "alpha-APE", "nsmear-gauss", "alpha-gauss",
					    "nsrc", "src-filename", "maxQsq", "twop-filename", "corr-file-format", "corr-space", "tSinks","Projs", "threep-filename"};
  
int main(int argc, char **argv)
{
  initializeOptions(argc, argv, true, listOpt);
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
  std::vector<double> mu_s;
  std::vector<double> mu_c;
  double mu_ud = mu;
  double mu_ud_factor[QUDA_MAX_MG_LEVEL];
  for(int i=0;i<QUDA_MAX_MG_LEVEL;i++) mu_ud_factor[i] = mu_factor[i];
  int nsmearGauss_s = nsmearGauss/2;
  int nsmearGauss_c = 0;
  std::string prOrNt = "neutron";
  HGC_options->set("mu-s", "List of mu_s to run for the strange quark in baryons", verbosity, mu_s);
  HGC_options->set("mu-c", "List of mu_c to run for the charm quark in baryons", verbosity, mu_c);
  HGC_options->set("nsmear-gauss-s", "Number of Gaussian smearing step for the strange quark propagator", verbosity, nsmearGauss_s);
  HGC_options->set("nsmear-gauss-c", "Number of Gaussian smearing step for the charm quark propagator", verbosity, nsmearGauss_c);
  HGC_options->set("whichParticle", "Which particle we want to do the 3pf. Options (proton, neutron)", verbosity, prOrNt);
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
    
    for(int isource = 0 ; isource < numSourcePositions; isource++){
      PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
		    isource, sourcePositions[isource][0], sourcePositions[isource][1],
		    sourcePositions[isource][2], sourcePositions[isource][3]);

      PLEGMA_Propagator<float> propUP;
      PLEGMA_Propagator<float> propDN;
      { // Whithin this scope we keep track also of the propagator non smeared on the sink
	PLEGMA_Propagator<float> propUP_SL;
	PLEGMA_Propagator<float> propDN_SL;
	
	// ensuring mu positive
	if(mu != mu_ud) {
	  updateOptions(LIGHT);
	  mu = mu_ud;
	  solver.UpdateSolver();
	}
	for(int isc = 0 ; isc < 12 ; isc++){
	  PLEGMA_Vector<double> vectorInOut, vectorAuxD;
	  PLEGMA_Vector<float> vectorAuxF;
	  vectorAuxD.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
	  TIME(vectorInOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss, sourcePositions[isource][DIM_T]));
	  PLEGMA_printf("Going to invert UP for component %d\n", isc);
	  TIME(solver.solve(vectorInOut, vectorInOut));
	  vectorAuxF.copy(vectorInOut);
	  propUP_SL.absorb(vectorAuxF, isc/3, isc%3);
	  TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nsmearGauss, alphaGauss));
	  vectorAuxF.copy(vectorAuxD);
	  propUP.absorb(vectorAuxF, isc/3, isc%3);
	}
	
	// ensuring mu negative
	if(mu != -mu_ud) {
	  updateOptions(LIGHT);
	  mu = -mu_ud;
	  solver.UpdateSolver();
	}
	for(int isc = 0 ; isc < 12 ; isc++){
	  PLEGMA_Vector<double> vectorInOut, vectorAuxD;
	  PLEGMA_Vector<float> vectorAuxF;
	  vectorAuxD.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
	  TIME(vectorInOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss, sourcePositions[isource][DIM_T]));
	  PLEGMA_printf("Going to invert DN for component %d\n", isc);
	  TIME(solver.solve(vectorInOut, vectorInOut));
	  vectorAuxF.copy(vectorInOut);
	  propDN_SL.absorb(vectorAuxF, isc/3, isc%3);
	  TIME(vectorAuxD.gaussianSmearing(vectorInOut,smearedGauge, nsmearGauss, alphaGauss));
	  vectorAuxF.copy(vectorAuxD);
	  propDN.absorb(vectorAuxF, isc/3, isc%3);
	}
#ifdef PLEGMA_NUCLEON_3PF_FIX_SINK
	for(size_t its = 0; its < tSinks.size(); its++){
	  int tsinkMtsource = tSinks[its];
	  if(tsinkMtsource >= HGC_totalL[3])
	    PLEGMA_error("Provided tsink=%d is >= than temporal extent",tsinkMtsource);
	  int signPer = (tsinkMtsource+sourcePositions[isource][3]) >= HGC_totalL[3] ? -1 : +1;
	  int global_fixSinkTime = (tsinkMtsource + sourcePositions[isource][3])%HGC_totalL[3]; 

	  // 3D propagators at t_sink
	  PLEGMA_Propagator3D<float> propUP3D;
	  PLEGMA_Propagator3D<float> propDN3D;
	  propUP3D.absorb(propUP, global_fixSinkTime);
	  propDN3D.absorb(propDN, global_fixSinkTime);

	  WHICHPARTICLE nucleon = get_particle(prOrNt); 
	  std::vector<GAMMAS> gammas = {ONE,G1,G2,G3,G4,G5,G5G1,G5G2,G5G3,G5G4,S12,S13,S23,S41,S42,S43};
	  for(size_t iproj = 0; iproj < Projs.size(); iproj++){
	    for(int flav = 0; flav < 2; flav++){
	      int signProps = (nucleon == PROTON) ? ((flav==0) ? +1: -1) : ((flav==0) ? -1 : +1);
	      std::string fl = (nucleon == PROTON) ? ((flav==0) ? "up":"dn") : ((flav==0) ? "dn":"up");
	      PLEGMA_Propagator<float> &propF = (nucleon == PROTON) ? ((flav==0) ? propUP_SL:propDN_SL):
		((flav==0) ? propDN_SL:propUP_SL);
	      std::string filename = threep_filename + "_" + Projs[iproj] + "_dt" + std::to_string(tsinkMtsource) + "_" + fl;
	      
	      PLEGMA_Propagator<float> seqProp;
	      if(flav==0) {
		// ensuring mu positive
		if(mu != mu_ud) {
		  updateOptions(LIGHT);
		  mu = mu_ud;
		  solver.UpdateSolver();
		}
	      } else {
		// ensuring mu negative
		if(mu != -mu_ud) {
		  updateOptions(LIGHT);
		  mu = -mu_ud;
		  solver.UpdateSolver();
		}
	      }
	      
	      for(int nu = 0 ; nu < 4 ; nu++)
		for(int c2 = 0 ; c2 < 3 ; c2++){
		  PLEGMA_Vector<double> vectorInOut, vectorAuxD;
		  PLEGMA_Vector<float> vectorAuxF;
		  if(flav==0) {
		    if(nucleon == PROTON)
		      vectorAuxF.seqSourceNucleon(propUP3D, propDN3D, get_projector(Projs[iproj]),
						  nucleon, global_fixSinkTime, nu, c2);
		    else
		      vectorAuxF.seqSourceNucleon(propDN3D, propUP3D, get_projector(Projs[iproj]),
						  nucleon, global_fixSinkTime, nu, c2);
		  } else {
		    if(nucleon == PROTON)
		      vectorAuxF.seqSourceNucleon(propUP3D, get_projector(Projs[iproj]), nucleon,
						  global_fixSinkTime, nu, c2);
		    else
		      vectorAuxF.seqSourceNucleon(propDN3D, get_projector(Projs[iproj]), nucleon,
						  global_fixSinkTime, nu, c2);
		  }
		  // put a momentum in the sink later
		  vectorAuxF.conjugate();
		  vectorAuxF.apply_gamma(G5);
		  vectorAuxD.copy(vectorAuxF);
		  // TODO: gaussian smearing only on the t_sink
		  TIME(vectorInOut.gaussianSmearing(vectorAuxD,smearedGauge, nsmearGauss, alphaGauss, global_fixSinkTime));
		  double norm = vectorInOut.norm();
		  vectorInOut.cscale(1/norm);
		  TIME(solver.solve(vectorInOut, vectorInOut));
		  vectorInOut.cscale(norm);
		  vectorAuxF.copy(vectorInOut);
		  seqProp.absorb(vectorAuxF, nu, c2);
		}
	      seqProp.apply_gamma(G5);
	      seqProp.conjugate();
	      
	      PLEGMA_Correlator<float> corr(corr_space, sourcePositions[isource], maxQsq, tsinkMtsource+1);
	  
	      // LOCAL contractions
	      corr.contractNucleonThrp_local(seqProp, propF, signProps, gammas);
	      if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) corr.H_elem()[iv] *= signPer;      
	      THREAD(corr.writeFile(filename, corr_file_format));
	      
	      // ONED contractions
	      corr.contractNucleonThrp_oneD(seqProp, propF, contractGauge, signProps, gammas);
	      if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) corr.H_elem()[iv] *= signPer;
	      THREAD(corr.writeFile( filename, corr_file_format));

	      // noe contractions
	      corr.contractNucleonThrp_noe(seqProp, propF, contractGauge, signProps);
	      if(signPer < 0) for(size_t iv = 0 ; iv < corr.getTotalSize()*2; iv++) corr.H_elem()[iv] *= signPer;
	      THREAD(corr.writeFile( filename, corr_file_format));
	    }
	  }
	}
#endif
      }
      propUP.rotateToPhysicalBase_device(+1);
      propDN.rotateToPhysicalBase_device(-1);
      propUP.applyBoundaries_device(sourcePositions[isource][3]);
      propDN.applyBoundaries_device(sourcePositions[isource][3]);

      {
	PLEGMA_Correlator<float> corr(corr_space, sourcePositions[isource], maxQsq);
	TIME(corr.contractMesons(propUP, propDN));
	
	char *dset1, *dset2;
	asprintf(&dset1, "twop_mesons_u[%+1.1e]d[%+1.1e]", mu_ud, -1*mu_ud);
	asprintf(&dset2, "twop_mesons_d[%+1.1e]u[%+1.1e]", -1*mu_ud, mu_ud);
	corr.setDatasets((std::vector<std::string>) {dset1, dset2});
	free(dset1); free(dset2);
	THREAD(corr.writeFile(twop_filename, corr_file_format));
	
	TIME(corr.contractBaryons(propUP, propDN));
	THREAD(corr.writeFile(twop_filename, corr_file_format));
      }
      
      // Storing only the smaller and then computing on the fly the other
      int nSmaller = std::min(mu_s.size(),mu_c.size());
      char cSmaller = (nSmaller==(int)mu_s.size()) ? 's' : 'c';
      
      PLEGMA_Propagator<float> propS[nSmaller];
      for(int ismall=0; ismall < nSmaller; ismall++) {
	for(int i=0;i<QUDA_MAX_MG_LEVEL;i++) mu_factor[i] = 1;
	updateOptions((cSmaller=='s') ? STRANGE : CHARM);
	mu = (cSmaller=='s') ? mu_s[ismall] : mu_c[ismall];
	int nsmear = (cSmaller=='s') ? nsmearGauss_s : nsmearGauss_c;
	solver.UpdateSolver();
	
	for(int isc = 0 ; isc < 12 ; isc++){
	  PLEGMA_Vector<double> vectorInOut, vectorAuxD;
	  PLEGMA_Vector<float> vectorAuxF;
	  vectorAuxD.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
	  TIME(vectorInOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmear, alphaGauss, sourcePositions[isource][DIM_T]));
	  
	  PLEGMA_printf("Going to invert %f for component %d\n", mu, isc);
	  TIME(solver.solve(vectorInOut, vectorInOut));
	  TIME(vectorAuxD.gaussianSmearing(vectorInOut,smearedGauge, nsmear, alphaGauss));
	  vectorAuxF.copy(vectorAuxD);
	  propS[ismall].absorb(vectorAuxF, isc/3, isc%3);
	}
	propS[ismall].rotateToPhysicalBase_device(mu/abs(mu));
	propS[ismall].applyBoundaries_device(sourcePositions[isource][3]);
      }
      
      int nLarger = (cSmaller!='s') ? mu_s.size() : mu_c.size();
      if(nLarger > 0) {
	PLEGMA_Propagator<float> propL;
	for(int ilarge=0; ilarge < nLarger; ilarge++) {
	  updateOptions((cSmaller!='s') ? STRANGE : CHARM);
	  mu = (cSmaller!='s') ? mu_s[ilarge] : mu_c[ilarge];
	  int nsmear = (cSmaller!='s') ? nsmearGauss_s : nsmearGauss_c;
	  solver.UpdateSolver();
	  
	  for(int isc = 0 ; isc < 12 ; isc++){
	    PLEGMA_Vector<double> vectorInOut, vectorAuxD;
	    PLEGMA_Vector<float> vectorAuxF;
	    vectorAuxD.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
	    TIME(vectorInOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmear, alphaGauss, sourcePositions[isource][DIM_T]));
	    
	    PLEGMA_printf("Going to invert %f for component %d\n", mu, isc);
	    TIME(solver.solve(vectorInOut, vectorInOut));
	    TIME(vectorAuxD.gaussianSmearing(vectorInOut,smearedGauge, nsmear, alphaGauss));
	    vectorAuxF.copy(vectorAuxD);
	    propL.absorb(vectorAuxF, isc/3, isc%3);
	  }
	  propL.rotateToPhysicalBase_device(mu/abs(mu));
	  propL.applyBoundaries_device(sourcePositions[isource][3]);

	  if(nSmaller>0) {
	    for(int ismall=0; ismall < nSmaller; ismall++) {
	      PLEGMA_Propagator<float> &propST = (cSmaller=='s') ? propS[ismall] : propL;
	      PLEGMA_Propagator<float> &propCH = (cSmaller=='c') ? propS[ismall] : propL;
	      PLEGMA_Correlator<float> corr(corr_space, sourcePositions[isource], maxQsq);
	      bool only_st = (ismall>0 && cSmaller=='s') || (ilarge>0 && cSmaller!='s');
	      bool only_ch = (ismall>0 && cSmaller=='c') || (ilarge>0 && cSmaller!='c');
#ifdef PLEGMA_UDSC_BARYONS
	      TIME(corr.contractBaryonsUDSC(propUP, propDN, propST, propCH, only_st, only_ch));
	      char * group;
	    
	      asprintf(&group, "baryons_u[%+1.1e]d[%+1.1e]s[%+1.1e]c[%+1.1e]%s%s", mu_ud, -1*mu_ud, mu_s[cSmaller=='s'? ismall:ilarge], mu_c[cSmaller=='c'? ismall:ilarge],
		       only_st ? "_only-s" : "", only_ch ? "_only-c" : "");
	      corr.setGroups(group);
	      free(group);
	      THREAD(corr.writeFile(twop_filename, corr_file_format));
#endif
	      TIME(corr.contractMesons(propST, propCH));
	      char *dset1, *dset2;
	      asprintf(&dset1, "twop_mesons_s[%+1.1e]c[%+1.1e]", mu_s[cSmaller=='s'? ismall:ilarge], mu_c[cSmaller=='c'? ismall:ilarge]);
	      asprintf(&dset2, "twop_mesons_c[%+1.1e]s[%+1.1e]", mu_c[cSmaller=='c'? ismall:ilarge], mu_s[cSmaller=='s'? ismall:ilarge]);
	      corr.setDatasets((std::vector<std::string>) {dset1, dset2});
	      free(dset1); free(dset2);
	      THREAD(corr.writeFile(twop_filename, corr_file_format));

	      if(!only_ch) {
		TIME(corr.contractMesons(propUP, propST));
		asprintf(&dset1, "twop_mesons_u[%+1.1e]s[%+1.1e]", mu_ud, mu_s[cSmaller=='s'? ismall:ilarge]);
		asprintf(&dset2, "twop_mesons_s[%+1.1e]u[%+1.1e]", mu_s[cSmaller=='s'? ismall:ilarge], mu_ud);
		corr.setDatasets((std::vector<std::string>) {dset1, dset2});
		free(dset1); free(dset2);
		THREAD(corr.writeFile(twop_filename, corr_file_format));
	      
		TIME(corr.contractMesons(propDN, propST));
		asprintf(&dset1, "twop_mesons_d[%+1.1e]s[%+1.1e]", -1*mu_ud, mu_s[cSmaller=='s'? ismall:ilarge]);
		asprintf(&dset2, "twop_mesons_s[%+1.1e]d[%+1.1e]", mu_s[cSmaller=='s'? ismall:ilarge], -1*mu_ud);
		corr.setDatasets((std::vector<std::string>) {dset1, dset2});
		free(dset1); free(dset2);
		THREAD(corr.writeFile(twop_filename, corr_file_format));
	      }

	      if(!only_st) {
		TIME(corr.contractMesons(propUP, propCH));
		asprintf(&dset1, "twop_mesons_u[%+1.1e]c[%+1.1e]", mu_ud, mu_c[cSmaller=='c'? ismall:ilarge]);
		asprintf(&dset2, "twop_mesons_c[%+1.1e]u[%+1.1e]", mu_c[cSmaller=='c'? ismall:ilarge], mu_ud);
		corr.setDatasets((std::vector<std::string>) {dset1, dset2});
		free(dset1); free(dset2);
		THREAD(corr.writeFile(twop_filename, corr_file_format));
	      
		TIME(corr.contractMesons(propDN, propCH));
		asprintf(&dset1, "twop_mesons_d[%+1.1e]c[%+1.1e]", -1*mu_ud, mu_c[cSmaller=='c'? ismall:ilarge]);
		asprintf(&dset2, "twop_mesons_c[%+1.1e]d[%+1.1e]", mu_c[cSmaller=='c'? ismall:ilarge], -1*mu_ud);
		corr.setDatasets((std::vector<std::string>) {dset1, dset2});
		free(dset1); free(dset2);
		THREAD(corr.writeFile(twop_filename, corr_file_format));
	      }
	    }
	  } else {
	    PLEGMA_Propagator<float> none(NONE);
	    PLEGMA_Propagator<float> &propST = (cSmaller=='s') ? none : propL;
	    PLEGMA_Propagator<float> &propCH = (cSmaller=='c') ? none : propL;
	    PLEGMA_Correlator<float> corr(corr_space, sourcePositions[isource], maxQsq);
	    bool only_st = (ilarge>0 && cSmaller!='s');
	    bool only_ch = (ilarge>0 && cSmaller!='c');
#ifdef PLEGMA_UDSC_BARYONS
	    TIME(corr.contractBaryonsUDSC(propUP, propDN, propST, propCH, only_st, only_ch));
	    char * group;

	    if(cSmaller=='s') {
	      asprintf(&group, "baryons_u[%+1.1e]d[%+1.1e]c[%+1.1e]%s", mu_ud, -1*mu_ud, mu_c[ilarge], only_ch ? "_only-c" : "");
	    } else {
	      asprintf(&group, "baryons_u[%+1.1e]d[%+1.1e]s[%+1.1e]%s", mu_ud, -1*mu_ud, mu_s[ilarge], only_st ? "_only-s" : "");
	    }
	    corr.setGroups(group);
	    free(group);
	    THREAD(corr.writeFile(twop_filename, corr_file_format));
#endif
	    if(!only_ch && !only_st) {
	      char *dset1, *dset2;
	      TIME(corr.contractMesons(propUP, (cSmaller=='s') ? propCH : propST));
	      if(cSmaller=='s') {
		asprintf(&dset1, "twop_mesons_u[%+1.1e]c[%+1.1e]", mu_ud, mu_c[ilarge]);
		asprintf(&dset2, "twop_mesons_c[%+1.1e]u[%+1.1e]", mu_c[ilarge], mu_ud);
	      } else {
		asprintf(&dset1, "twop_mesons_u[%+1.1e]s[%+1.1e]", mu_ud, mu_s[ilarge]);
		asprintf(&dset2, "twop_mesons_s[%+1.1e]u[%+1.1e]", mu_s[ilarge], mu_ud);
	      }

	      corr.setDatasets((std::vector<std::string>) {dset1, dset2});
	      free(dset1); free(dset2);
	      THREAD(corr.writeFile(twop_filename, corr_file_format));
	      
	      TIME(corr.contractMesons(propDN, (cSmaller=='s') ? propCH : propST));
	      if(cSmaller=='s') {
		asprintf(&dset1, "twop_mesons_d[%+1.1e]c[%+1.1e]", -1*mu_ud, mu_c[ilarge]);
		asprintf(&dset2, "twop_mesons_c[%+1.1e]d[%+1.1e]", mu_c[ilarge], -1*mu_ud);
	      } else {
		asprintf(&dset1, "twop_mesons_d[%+1.1e]s[%+1.1e]", -1*mu_ud, mu_s[ilarge]);
		asprintf(&dset2, "twop_mesons_s[%+1.1e]d[%+1.1e]", mu_s[ilarge], -1*mu_ud);
	      }
	      corr.setDatasets((std::vector<std::string>) {dset1, dset2});
	      free(dset1); free(dset2);
	      THREAD(corr.writeFile(twop_filename, corr_file_format));
	    }
	  }
	}
      } else {
#ifdef PLEGMA_UDSC_BARYONS
	PLEGMA_Propagator<float> none(NONE);
	PLEGMA_Correlator<float> corr(corr_space, sourcePositions[isource], maxQsq);
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

  finalize();
  return 0;
}

