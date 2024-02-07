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
					    "nsrc", "src-filename", "maxQsq", "twop-filename", "corr-file-format", "corr-space", "xiMomSm","sinkMom"};

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
  std::string srcInputFile = "./input.src";
  auto add_options = [&](Options& options) {
    options.set("run-ud", "Whether to run or not light quark flavors", verbosity, run_ud);
    options.set("mu-s", "List of mu_s to run for the strange quark in baryons", verbosity, mu_s);
    options.set("mu-c", "List of mu_c to run for the charm quark in baryons", verbosity, mu_c);
    options.set("nsmear-gauss-s", "Number of Gaussian smearing step for the strange quark propagator", verbosity, nsmearGauss_s);
    //options.set("src-input-file", "Use the file to update option at every source. The file searched is [src-input-file]+str(n) where n is the source (0, 1, ...)", verbosity, srcInputFile);
    options.set("nsmear-gauss-c", "Number of Gaussian smearing step for the charm quark propagator", verbosity, nsmearGauss_c);
		     };
 add_options(*HGC_options);
   //=========================================================================================================//
  initializePLEGMA();

  std::vector<int> sinkMom = {0,0,0};

  std::string given_twop_filename = twop_filename;
  {
    PLEGMA_Gauge<double> smearedGauge(BOTH);
    double max_mom = 0;
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

      //Momentum smearing: put the momentum phase to smeared gauge field
      std::complex<double> momSmScale[N_DIMS];
      std::complex<double> I(0,1);
      for(int i = 0 ; i < N_DIMS; i++) {
	momSmScale[i] = std::exp(-(xiMomSm*2.*PI*sinkMom[i]/HGC_totalL[i])*I);
	max_mom = max_mom>pow(xiMomSm*sinkMom[i],2) ? max_mom:pow(xiMomSm*sinkMom[i],2);
      }
      if(max_mom>0)
	smearedGauge.scaleDirWise(momSmScale);
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
    TIME(QUDA_solver solver(mu));
    std::vector<std::thread> threads;

    
    for(int isource = 0 ; isource < numSourcePositions; isource++){
      PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
		    isource, sourcePositions[isource][0], sourcePositions[isource][1],
		    sourcePositions[isource][2], sourcePositions[isource][3]);

      site& source = sourcePositions[isource];
      PLEGMA_Propagator<float> propUP(run_ud ? BOTH : NONE);
      PLEGMA_Propagator<float> propDN(run_ud ? BOTH : NONE);

      char * src_string;
      asprintf(&src_string, "_sx%02dsy%02dsz%02dst%03d", source[0], source[1], source[2], source[3]);
      twop_filename = given_twop_filename + std::string("_") + ((nsmearGauss>0) ? "SS" : "LL") +
	"_gN" + std::to_string(nsmearGauss) + "a" + convNumToStr(alphaGauss) +
	"_aN" + std::to_string(nsmearAPE) + "a" + convNumToStr(alphaAPE) + src_string;
      free(src_string);

      if(max_mom > 0) {
	asprintf(&src_string, "_mx%02dmy%02dmz%02d_xi%0.2f", sinkMom[0], sinkMom[1], sinkMom[2], xiMomSm);
	twop_filename = twop_filename + src_string;
	free(src_string);
      }
      twop_filename = twop_filename+".h5";
      
      if(access( twop_filename.c_str(), F_OK ) != -1)
	continue;

      {
	PLEGMA_Gauge3D<double> smearedGauge3D;
	smearedGauge3D.absorb(smearedGauge, source[DIM_T]);
	
	auto computePropagator = [&](PLEGMA_Propagator<float>& prop, const double run_mu, WHICHFLAVOR fl, int nSmear) {
				   // ensuring mu value
				   if(mu != run_mu) {
				     updateOptions(fl);
				     mu = run_mu;
				     solver.UpdateSolver();
				 }
				   for(int isc = 0 ; isc < 12 ; isc++){
				     PLEGMA_Vector<double> vectorInOut;
<<<<<<< HEAD
=======
				     vectorInOut.zero_where(BOTH);
>>>>>>> origin/hip_lumi_working
				     { // Smearing the source
				       PLEGMA_Vector3D<double> vector1, vector2;
				       vector1.pointSource(source, isc/3, isc%3, DEVICE);
				       TIME(vector2.gaussianSmearing(vector1, smearedGauge3D, nSmear, alphaGauss));
				       vectorInOut.absorb(vector2,source[DIM_T]);
				     }
				     // Inverting
				     PLEGMA_printf("Going to invert %s for component %d\n",
						   fl==LIGHT ? "LIGHT" : (fl == STRANGE ? "STRANGE" : "CHARM"), isc);
				     vectorInOut.unload();
<<<<<<< HEAD
				     vectorInOut.writeHDF5("testinputbooster");
				     vectorInOut.load();
				     TIME(solver.solve(vectorInOut, vectorInOut));
				     vectorInOut.unload();
                                     vectorInOut.writeHDF5("testoutputbooster");
=======
				     vectorInOut.writeHDF5("testinput");
				     vectorInOut.load();
				     TIME(solver.solve(vectorInOut, vectorInOut));
				     vectorInOut.unload();
                                     vectorInOut.writeHDF5("testoutput");
>>>>>>> origin/hip_lumi_working
				     vectorInOut.load();

				     { // Smearing the solution
				       PLEGMA_Vector<double> vectorAuxD;
				       PLEGMA_Vector<float> vectorAuxF;
				       TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nSmear, alphaGauss));
				       vectorAuxF.copy(vectorAuxD);
				       prop.absorb(vectorAuxF, isc/3, isc%3);
				   }
				   }  
				   prop.rotateToPhysicalBase_device(run_mu/abs(run_mu));
				   prop.applyBoundaries_device(source[DIM_T]);
				 };
	
	TIME(computePropagator(propUP, mu_ud, LIGHT, nsmearGauss));
	TIME(computePropagator(propDN, -mu_ud, LIGHT, nsmearGauss));
      }	
      
      PLEGMA_Correlator<float> corr(corr_space, source, maxQsq);
      TIME(corr.contractMesonsNew(propUP, propDN));
      char *dset1, *dset2;
      asprintf(&dset1, "twop_mesons_new_u[%+1.1e]d[%+1.1e]", mu_ud, -1*mu_ud);
      asprintf(&dset2, "twop_mesons_new_d[%+1.1e]u[%+1.1e]", -1*mu_ud, mu_ud);
      corr.setDatasets((std::vector<std::string>) {dset1});
      free(dset1); free(dset2);
      THREAD(corr.writeFile(twop_filename, corr_file_format));
      
      TIME(corr.contractBaryons(propUP, propDN));
      THREAD(corr.writeFile(twop_filename, corr_file_format));
      
#ifdef PLEGMA_UDSC_BARYONS
<<<<<<< HEAD
=======
#if 0
>>>>>>> origin/hip_lumi_working
      PLEGMA_Propagator<float> none(NONE);
      TIME(corr.contractBaryonsUDSC(propUP, propDN, none, none));
      
      char * group;
      asprintf(&group, "baryons_u[%+1.1e]d[%+1.1e]", mu_ud, -1*mu_ud);
      corr.setGroups(group);
      free(group);
      THREAD(corr.writeFile(twop_filename, corr_file_format));
#endif
<<<<<<< HEAD
=======
#endif
>>>>>>> origin/hip_lumi_working
    }
    while(not threads.empty()) {threads.back().join(); threads.pop_back();}
  }

  //finalize();
  return 0;
}

