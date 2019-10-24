#include <PLEGMA.h>
#include <PLEGMA_utils.h>

using namespace plegma;
using namespace quda;
static std::vector<std::string> listOpt = { "verbosity", "load-gauge", "nsmear-APE", "alpha-APE", "nsmear-gauss", "alpha-gauss",
					    "nsrc", "src-filename", "maxQsq", "twop-baryons-filename", "twop-mesons-filename", "corr-file-format", "corr-space"};
  
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
  HGC_options->set("run-ud", "Wheater to run or not light quark flavors", verbosity, run_ud);
  HGC_options->set("mu-s", "List of mu_s to run for the strange quark in baryons", verbosity, mu_s);
  HGC_options->set("mu-c", "List of mu_c to run for the charm quark in baryons", verbosity, mu_c);
  HGC_options->set("nsmear-gauss-s", "Number of Gaussian smearing step for the strange quark propagator", verbosity, nsmearGauss_s);
  HGC_options->set("nsmear-gauss-c", "Number of Gaussian smearing step for the charm quark propagator", verbosity, nsmearGauss_c);
  //=========================================================================================================//
  initializePLEGMA();
  double start_time, tmp_time;
  {
    PLEGMA_Gauge<double> smearedGauge(BOTH);
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
      smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3);
      PLEGMA_printf("Plaquette after smearing:\n");
      smearedGauge.calculatePlaq();
    }
    QUDA_solver solver(mu);
    
    for(int isource = 0 ; isource < numSourcePositions; isource++){
      PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
		    isource, sourcePositions[isource][0], sourcePositions[isource][1],
		    sourcePositions[isource][2], sourcePositions[isource][3]);

      PLEGMA_Propagator<float> propUP(run_ud ? BOTH : NONE);
      tmp_time = 0;
      // ensuring mu positive
      if (run_ud) {
	if(mu != mu_ud) {
	  for(int i=0;i<QUDA_MAX_MG_LEVEL;i++) mu_factor[i] = mu_ud_factor[i];
	  mu = mu_ud;
	  solver.UpdateSolver();
	}
	for(int isc = 0 ; isc < 12 ; isc++){
	  PLEGMA_Vector<double> vectorInOut, vectorAuxD;
	  PLEGMA_Vector<float> vectorAuxF;
	  vectorAuxD.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
	  start_time = MPI_Wtime();
	  vectorInOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
	  tmp_time += MPI_Wtime()-start_time;
	  PLEGMA_printf("Going to invert UP for component %d\n", isc);
	  solver.solve(vectorInOut, vectorInOut);
	  start_time = MPI_Wtime();
	  vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nsmearGauss, alphaGauss);
	  tmp_time += MPI_Wtime()-start_time;
	  vectorAuxF.copy(vectorAuxD);
	  propUP.absorb(vectorAuxF, isc/3, isc%3);
	}
      }
      
      PLEGMA_Propagator<float> propDN(run_ud ? BOTH : NONE);
      // ensuring mu negative
      if (run_ud) {
	if(mu != -1*mu_ud) {
	  mu = -1*mu_ud;
	  solver.UpdateSolver();
	}
	for(int isc = 0 ; isc < 12 ; isc++){
	  PLEGMA_Vector<double> vectorInOut, vectorAuxD;
	  PLEGMA_Vector<float> vectorAuxF;
	  vectorAuxD.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
	  start_time = MPI_Wtime();
	  vectorInOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmearGauss, alphaGauss);
	  tmp_time += MPI_Wtime()-start_time;
	  PLEGMA_printf("Going to invert DN for component %d\n", isc);
	  solver.solve(vectorInOut, vectorInOut);
	  start_time = MPI_Wtime();
	  vectorAuxD.gaussianSmearing(vectorInOut,smearedGauge, nsmearGauss, alphaGauss);
	  tmp_time += MPI_Wtime()-start_time;
	  vectorAuxF.copy(vectorAuxD);
	  propDN.absorb(vectorAuxF, isc/3, isc%3);
	}
	PLEGMA_printf("Smearing time %lf sec\n",tmp_time);
	propUP.rotateToPhysicalBase_device(+1);
	propDN.rotateToPhysicalBase_device(-1);
	propUP.applyBoundaries_device(sourcePositions[isource][3]);
	propDN.applyBoundaries_device(sourcePositions[isource][3]);

	PLEGMA_Correlator<float> corrm(corr_space, maxQsq);
	PLEGMA_Correlator<float> corrb(corr_space, maxQsq);
	start_time = MPI_Wtime();
	corrm.contractMesons(propUP, propDN, sourcePositions[isource]);
	tmp_time = MPI_Wtime()-start_time;
	PLEGMA_printf("Contraction time for mesons %lf sec\n",tmp_time);
	
	char *dset1, *dset2;
	asprintf(&dset1, "twop_mesons_u[%+1.1e]d[%+1.1e]", mu_ud, -1*mu_ud);
	asprintf(&dset2, "twop_mesons_d[%+1.1e]u[%+1.1e]", -1*mu_ud, mu_ud);
	corrm.setDatasets((std::vector<std::string>) {dset1, dset2});
	free(dset1); free(dset2);
	corrm.writeFile(twop_m_filename.c_str(), corr_file_format, true);
	
	start_time = MPI_Wtime();
	corrb.contractBaryons(propUP, propDN, sourcePositions[isource]);
	tmp_time = MPI_Wtime()-start_time;
	PLEGMA_printf("Contraction time for baryons %lf sec\n",tmp_time);
	corrb.writeFile(twop_b_filename.c_str(), corr_file_format, true);
      }
      
      // Storing only the smaller and then computing on the fly the other
      int nSmaller = MIN(mu_s.size(),mu_c.size());
      char cSmaller = (nSmaller==(int)mu_s.size()) ? 's' : 'c';

      PLEGMA_Propagator<float> propS[nSmaller];
      for(int ismall=0; ismall < nSmaller; ismall++) {
	for(int i=0;i<QUDA_MAX_MG_LEVEL;i++) mu_factor[i] = 1;
	mu = (cSmaller=='s') ? mu_s[ismall] : mu_c[ismall];
	int nsmear = (cSmaller=='s') ? nsmearGauss_s : nsmearGauss_c;
	solver.UpdateSolver();
	
	for(int isc = 0 ; isc < 12 ; isc++){
	  PLEGMA_Vector<double> vectorInOut, vectorAuxD;
	  PLEGMA_Vector<float> vectorAuxF;
	  vectorAuxD.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
	  vectorInOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmear, alphaGauss);
	  
	  PLEGMA_printf("Going to invert %f for component %d\n", mu, isc);
	  solver.solve(vectorInOut, vectorInOut);
	  vectorAuxD.gaussianSmearing(vectorInOut,smearedGauge, nsmear, alphaGauss);
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
	  mu = (cSmaller!='s') ? mu_s[ilarge] : mu_c[ilarge];
	  int nsmear = (cSmaller!='s') ? nsmearGauss_s : nsmearGauss_c;
	  solver.UpdateSolver();
	  
	  for(int isc = 0 ; isc < 12 ; isc++){
	    PLEGMA_Vector<double> vectorInOut, vectorAuxD;
	    PLEGMA_Vector<float> vectorAuxF;
	    vectorAuxD.pointSource(sourcePositions[isource], isc/3, isc%3, DEVICE);
	    vectorInOut.gaussianSmearing(vectorAuxD, smearedGauge, nsmear, alphaGauss);
	    
	    PLEGMA_printf("Going to invert %f for component %d\n", mu, isc);
	    solver.solve(vectorInOut, vectorInOut);
	    vectorAuxD.gaussianSmearing(vectorInOut,smearedGauge, nsmear, alphaGauss);
	    vectorAuxF.copy(vectorAuxD);
	    propL.absorb(vectorAuxF, isc/3, isc%3);
	  }
	  propL.rotateToPhysicalBase_device(mu/abs(mu));
	  propL.applyBoundaries_device(sourcePositions[isource][3]);

	  if(nSmaller>0) {
	    for(int ismall=0; ismall < nSmaller; ismall++) {
	      PLEGMA_Propagator<float> &propST = (cSmaller=='s') ? propS[ismall] : propL;
	      PLEGMA_Propagator<float> &propCH = (cSmaller=='c') ? propS[ismall] : propL;
	      PLEGMA_Correlator<float> corrb(corr_space, maxQsq);
	      PLEGMA_Correlator<float> corrm(corr_space, maxQsq);
	      bool only_st = (ismall>0 && cSmaller=='s') || (ilarge>0 && cSmaller!='s');
	      bool only_ch = (ismall>0 && cSmaller=='c') || (ilarge>0 && cSmaller!='c');
#ifdef PLEGMA_UDSC_BARYONS
	      corrb.contractBaryonsUDSC(propUP, propDN, propST, propCH, sourcePositions[isource], HGC_totalL[DIM_T], only_st, only_ch);
	      char * group;
	    
	      asprintf(&group, "baryons_u[%+1.1e]d[%+1.1e]s[%+1.1e]c[%+1.1e]%s%s", mu_ud, -1*mu_ud, mu_s[cSmaller=='s'? ismall:ilarge], mu_c[cSmaller=='c'? ismall:ilarge],
		       only_st ? "_only-s" : "", only_ch ? "_only-c" : "");
	      corrb.setGroups(group);
	      free(group);
	      corrb.writeFile(twop_filename.c_str(), corr_file_format, true);
#endif
	      corrm.contractMesons(propST, propCH, sourcePositions[isource]);
	      char *dset1, *dset2;
	      asprintf(&dset1, "twop_mesons_s[%+1.1e]c[%+1.1e]", mu_s[cSmaller=='s'? ismall:ilarge], mu_c[cSmaller=='c'? ismall:ilarge]);
	      asprintf(&dset2, "twop_mesons_c[%+1.1e]s[%+1.1e]", mu_c[cSmaller=='c'? ismall:ilarge], mu_s[cSmaller=='s'? ismall:ilarge]);
	      corrm.setDatasets((std::vector<std::string>) {dset1, dset2});
	      free(dset1); free(dset2);
	      corrm.writeFile(twop_filename.c_str(), corr_file_format, true);

	      if(!only_ch) {
		corrm.contractMesons(propUP, propST, sourcePositions[isource]);
		asprintf(&dset1, "twop_mesons_u[%+1.1e]s[%+1.1e]", mu_ud, mu_s[cSmaller=='s'? ismall:ilarge]);
		asprintf(&dset2, "twop_mesons_s[%+1.1e]u[%+1.1e]", mu_s[cSmaller=='s'? ismall:ilarge], mu_ud);
		corrm.setDatasets((std::vector<std::string>) {dset1, dset2});
		free(dset1); free(dset2);
		corrm.writeFile(twop_m_filename.c_str(), corr_file_format, true);
	      
		corrm.contractMesons(propDN, propST, sourcePositions[isource]);
		asprintf(&dset1, "twop_mesons_d[%+1.1e]s[%+1.1e]", -1*mu_ud, mu_s[cSmaller=='s'? ismall:ilarge]);
		asprintf(&dset2, "twop_mesons_s[%+1.1e]d[%+1.1e]", mu_s[cSmaller=='s'? ismall:ilarge], -1*mu_ud);
		corrm.setDatasets((std::vector<std::string>) {dset1, dset2});
		free(dset1); free(dset2);
		corrm.writeFile(twop_m_filename.c_str(), corr_file_format, true);
	      }

	      if(!only_st) {
		corrm.contractMesons(propUP, propCH, sourcePositions[isource]);
		asprintf(&dset1, "twop_mesons_u[%+1.1e]c[%+1.1e]", mu_ud, mu_c[cSmaller=='c'? ismall:ilarge]);
		asprintf(&dset2, "twop_mesons_c[%+1.1e]u[%+1.1e]", mu_c[cSmaller=='c'? ismall:ilarge], mu_ud);
		corrm.setDatasets((std::vector<std::string>) {dset1, dset2});
		free(dset1); free(dset2);
		corrm.writeFile(twop_m_filename.c_str(), corr_file_format, true);
	      
		corrm.contractMesons(propDN, propCH, sourcePositions[isource]);
		asprintf(&dset1, "twop_mesons_d[%+1.1e]c[%+1.1e]", -1*mu_ud, mu_c[cSmaller=='c'? ismall:ilarge]);
		asprintf(&dset2, "twop_mesons_c[%+1.1e]d[%+1.1e]", mu_c[cSmaller=='c'? ismall:ilarge], -1*mu_ud);
		corrm.setDatasets((std::vector<std::string>) {dset1, dset2});
		free(dset1); free(dset2);
		corrm.writeFile(twop_m_filename.c_str(), corr_file_format, true);
	      }
	    }
	  } else {
	    PLEGMA_Propagator<float> none(NONE);
	    PLEGMA_Propagator<float> &propST = (cSmaller=='s') ? none : propL;
	    PLEGMA_Propagator<float> &propCH = (cSmaller=='c') ? none : propL;
	    PLEGMA_Correlator<float> corrb(corr_space, maxQsq);
	    PLEGMA_Correlator<float> corrm(corr_space, maxQsq);
	    bool only_st = (ilarge>0 && cSmaller!='s');
	    bool only_ch = (ilarge>0 && cSmaller!='c');
#ifdef PLEGMA_UDSC_BARYONS
	    corrb.contractBaryonsUDSC(propUP, propDN, propST, propCH, sourcePositions[isource], HGC_totalL[DIM_T], only_st, only_ch);
	    char * group;

	    if(cSmaller=='s') {
	      asprintf(&group, "baryons_u[%+1.1e]d[%+1.1e]c[%+1.1e]%s", mu_ud, -1*mu_ud, mu_c[ilarge], only_ch ? "_only-c" : "");
	    } else {
	      asprintf(&group, "baryons_u[%+1.1e]d[%+1.1e]s[%+1.1e]%s", mu_ud, -1*mu_ud, mu_s[ilarge], only_st ? "_only-s" : "");
	    }
	    corrb.setGroups(group);
	    free(group);
	    corrb.writeFile(twop_b_filename.c_str(), corr_file_format, true);
#endif
	    if(!only_ch && !only_st) {
	      char *dset1, *dset2;
	      corrm.contractMesons(propUP, (cSmaller=='s') ? propCH : propST, sourcePositions[isource]);
	      if(cSmaller=='s') {
		asprintf(&dset1, "twop_mesons_u[%+1.1e]c[%+1.1e]", mu_ud, mu_c[ilarge]);
		asprintf(&dset2, "twop_mesons_c[%+1.1e]u[%+1.1e]", mu_c[ilarge], mu_ud);
	      } else {
		asprintf(&dset1, "twop_mesons_u[%+1.1e]s[%+1.1e]", mu_ud, mu_s[ilarge]);
		asprintf(&dset2, "twop_mesons_s[%+1.1e]u[%+1.1e]", mu_s[ilarge], mu_ud);
	      }

	      corrm.setDatasets((std::vector<std::string>) {dset1, dset2});
	      free(dset1); free(dset2);
	      corrm.writeFile(twop_m_filename.c_str(), corr_file_format, true);
	      
	      corrm.contractMesons(propDN, (cSmaller=='s') ? propCH : propST, sourcePositions[isource]);
	      if(cSmaller=='s') {
		asprintf(&dset1, "twop_mesons_d[%+1.1e]c[%+1.1e]", -1*mu_ud, mu_c[ilarge]);
		asprintf(&dset2, "twop_mesons_c[%+1.1e]d[%+1.1e]", mu_c[ilarge], -1*mu_ud);
	      } else {
		asprintf(&dset1, "twop_mesons_d[%+1.1e]s[%+1.1e]", -1*mu_ud, mu_s[ilarge]);
		asprintf(&dset2, "twop_mesons_s[%+1.1e]d[%+1.1e]", mu_s[ilarge], -1*mu_ud);
	      }
	      corrm.setDatasets((std::vector<std::string>) {dset1, dset2});
	      free(dset1); free(dset2);
	      corrm.writeFile(twop_m_filename.c_str(), corr_file_format, true);
	    }
	  }
	}
      } else if(run_ud) {
#ifdef PLEGMA_UDSC_BARYONS
	PLEGMA_Propagator<float> none(NONE);
	PLEGMA_Correlator<float> corrb(corr_space, maxQsq);
	corrb.contractBaryonsUDSC(propUP, propDN, none, none, sourcePositions[isource]);
	char * group;
	
	asprintf(&group, "baryons_u[%+1.1e]d[%+1.1e]", mu_ud, -1*mu_ud);
	corrb.setGroups(group);
	free(group);
	corrb.writeFile(twop_b_filename.c_str(), corr_file_format);
#endif
      }
    }
  }
  
  finalize();
  return 0;
}

