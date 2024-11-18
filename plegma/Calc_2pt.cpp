#include <PLEGMA.h>
#include <PLEGMA_utils.h>

std::vector<double> runtime;
#define TIME(fnc)  runtime.push_back(MPI_Wtime()); fnc;			\
  PLEGMA_printf("TIME for "#fnc" %f sec\n", MPI_Wtime()-runtime.back()); \
  runtime.pop_back()

std::vector<std::thread> threads;
//#define THREAD(fnc) threads.push_back(std::thread([=]() { TIME(fnc); }))
#define THREAD(fnc) TIME(fnc)

using namespace plegma;
using namespace quda;
static std::vector<std::string> listOpt = { "verbosity", "load-gauge", "nsmear-APE", "alpha-APE", "nsmear-gauss", "alpha-gauss",
					    "nsrc", "src-filename", "maxQsq", "twop-filename", "corr-file-format", "corr-space"};

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
  bool run_ud = true;
  HGC_options->set("run-ud", "Whether to run or not light quark flavors", verbosity, run_ud);
  HGC_options->set("mu-s", "List of mu_s to run for the strange quark in baryons", verbosity, mu_s);
  HGC_options->set("mu-c", "List of mu_c to run for the charm quark in baryons", verbosity, mu_c);
  HGC_options->set("nsmear-gauss-s", "Number of Gaussian smearing step for the strange quark propagator", verbosity, nsmearGauss_s);
  HGC_options->set("nsmear-gauss-c", "Number of Gaussian smearing step for the charm quark propagator", verbosity, nsmearGauss_c);
  //=========================================================================================================//
  initializePLEGMA();

  twop_filename += std::string("_") + ((nsmearGauss>0) ? "SS" : "LL") +
    "_gN" + std::to_string(nsmearGauss) + "a" + convNumToStr(alphaGauss) +
    "_aN" + std::to_string(nsmearAPE) + "a" + convNumToStr(alphaAPE);

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
    if(run_ud) {
      updateOptions(LIGHT);
      mu = mu_ud;
    } else if(mu_s.size()>0) {
      updateOptions(STRANGE);
      mu = mu_s[0];
    } else {
      updateOptions(CHARM);
      mu = mu_c[0];
    }
    TIME(QUDA_solver solver(mu,12));
    std::vector<std::thread> threads;

    
    for(int isource = 0 ; isource < numSourcePositions; isource++){
      PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
		    isource, sourcePositions[isource][0], sourcePositions[isource][1],
		    sourcePositions[isource][2], sourcePositions[isource][3]);

      site& source = sourcePositions[isource];
      PLEGMA_Propagator<float> propUP(run_ud ? BOTH : NONE);
      PLEGMA_Propagator<float> propDN(run_ud ? BOTH : NONE);

      PLEGMA_Gauge3D<double> smearedGauge3D;
      smearedGauge3D.absorb(smearedGauge, source[DIM_T]);

      auto computePropagator = [&](PLEGMA_Propagator<float>& prop, const double run_mu, WHICHFLAVOR fl, int nSmear) {
				 // ensuring mu value
				 if(mu != run_mu) {
				   updateOptions(fl);
				   mu = run_mu;
				   solver.UpdateSolver();
				 }

				 std::vector<PLEGMA_Vector<double>> vectorInOut(12);
				 for(int isc = 0 ; isc < 12 ; isc++){
				    // Smearing the source
				    PLEGMA_Vector3D<double> vector1, vector2;
				    vector1.pointSource(source, isc/3, isc%3, DEVICE);
				    TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nSmear, alphaGauss));
				    (vectorInOut[isc]).absorb(vector2,source[DIM_T]);
				 }
				   // Inverting
				 PLEGMA_printf("Going to invert %s for all component\n",
						 fl==LIGHT ? "LIGHT" : (fl == STRANGE ? "STRANGE" : "CHARM"));
				 TIME(solver.solve(vectorInOut, vectorInOut));
                                 for(int isc = 0 ; isc < 12 ; isc++)
				 { // Smearing the solution
				     PLEGMA_Vector<double> vectorAuxD;
				     PLEGMA_Vector<float> vectorAuxF;
				     TIME(vectorAuxD.gaussianSmearing(vectorInOut[isc], smearedGauge, nSmear, alphaGauss));
				     vectorAuxF.copy(vectorAuxD);
				     prop.absorb(vectorAuxF, isc/3, isc%3);
				 }  
				 prop.rotateToPhysicalBase_device(run_mu/abs(run_mu));
				 prop.applyBoundaries_device(source[DIM_T]);
			       };

      if (run_ud) {
	TIME(computePropagator(propUP, mu_ud, LIGHT, nsmearGauss));

	TIME(computePropagator(propDN, -mu_ud, LIGHT, nsmearGauss));

	PLEGMA_Correlator<float> corr(corr_space, source, maxQsq);
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
	double run_mu = (cSmaller=='s') ? mu_s[ismall] : mu_c[ismall];
	int nsmear = (cSmaller=='s') ? nsmearGauss_s : nsmearGauss_c;
	
	TIME(computePropagator(propS[ismall], run_mu, (cSmaller=='s') ? STRANGE : CHARM, nsmear));
      }
      
      int nLarger = (cSmaller!='s') ? mu_s.size() : mu_c.size();
      if(nLarger > 0) {
	PLEGMA_Propagator<float> propL;
	for(int ilarge=0; ilarge < nLarger; ilarge++) {
	  double run_mu = (cSmaller!='s') ? mu_s[ilarge] : mu_c[ilarge];
	  int nsmear = (cSmaller!='s') ? nsmearGauss_s : nsmearGauss_c;
	  TIME(computePropagator(propL, run_mu, (cSmaller!='s') ? STRANGE : CHARM, nsmear));
	  
	  if(nSmaller>0) {
	    for(int ismall=0; ismall < nSmaller; ismall++) {
	      PLEGMA_Propagator<float> &propST = (cSmaller=='s') ? propS[ismall] : propL;
	      PLEGMA_Propagator<float> &propCH = (cSmaller=='c') ? propS[ismall] : propL;
	      PLEGMA_Correlator<float> corr(corr_space, source, maxQsq);
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

	      if(!only_ch && run_ud) {
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

	      if(!only_st && run_ud) {
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
	    PLEGMA_Correlator<float> corr(corr_space, source, maxQsq);
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
	    if(!only_ch && !only_st && run_ud) {
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
      } else if(run_ud) {
#ifdef PLEGMA_UDSC_BARYONS
	PLEGMA_Propagator<float> none(NONE);
	PLEGMA_Correlator<float> corr(corr_space, source, maxQsq);
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

