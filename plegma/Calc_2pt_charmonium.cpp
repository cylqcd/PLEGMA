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
//   int nsmearStout = 0;
//   double alphaStout = 0.129;
//   std::vector<double> mu_s;
     double kappa_ud = kappa;
     double kappa_c;
//   double mu_ud = mu;
//   double mu_ud_factor[QUDA_MAX_MG_LEVEL];
//   for(int i=0;i<QUDA_MAX_MG_LEVEL;i++) mu_ud_factor[i] = mu_factor[i];
//   int nsmearGauss_s = nsmearGauss/2;
     int nsmearGauss_c = nsmearGauss;
//   bool run_ud = true;
//   HGC_options->set("nsmear-stout", "Number of stout smearing step for the configuration",verbosity,nsmearStout);
//   HGC_options->set("alpha-stout", "Coefficient for the stout smearing for the configuration",verbosity,alphaStout);
//   HGC_options->set("run-ud", "Whether to run or not light quark flavors", verbosity, run_ud);
//   HGC_options->set("mu-s", "List of mu_s to run for the strange quark in baryons", verbosity, mu_s);
   HGC_options->set("kappa-c", "List of kappa_c to run for the charm quark in mesons", verbosity, kappa_c);
//   HGC_options->set("nsmear-gauss-s", "Number of Gaussian smearing step for the strange quark propagator", verbosity, nsmearGauss_s);
     HGC_options->set("nsmear-gauss-c", "Number of Gaussian smearing step for the charm quark propagator", verbosity, nsmearGauss_c);
  //=========================================================================================================//
  
  initializePLEGMA();

  twop_filename += std::string("_") + ((nsmearGauss>0) ? "SS" : "LL") +
    "_gN" + std::to_string(nsmearGauss) + "a" + convNumToStr(alphaGauss) +
    "_aN" + std::to_string(nsmearAPE) + "a" + convNumToStr(alphaAPE) + 
    "_kappa"+ std::to_string(kappa_c);

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
      TIME(smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3));
      PLEGMA_printf("Smeared Plaquette with APE 3D:\n");
      smearedGauge.calculatePlaq();
    }
    updateOptions(LIGHT);
    TIME(QUDA_solver solver(0));
    std::vector<std::thread> threads;


    
    for(int isource = 0 ; isource < numSourcePositions; isource++){
      PLEGMA_printf("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
		    isource, sourcePositions[isource][0], sourcePositions[isource][1],
		    sourcePositions[isource][2], sourcePositions[isource][3]);

      site& source = sourcePositions[isource];
      

      PLEGMA_Gauge3D<double> smearedGauge3D;
      smearedGauge3D.absorb(smearedGauge, source[DIM_T]);
      
      
      
      
      
          auto computePropagator = [&](PLEGMA_Propagator<float>& prop,double run_kappa, WHICHFLAVOR fl, int nSmear) {
        
            // ensuring kappa value
            if(kappa != run_kappa) {
            updateOptions(fl);
            kappa = run_kappa;
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
            { // Smearing the solution
                PLEGMA_Vector<double> vectorAuxD;
                PLEGMA_Vector<float> vectorAuxF;
                TIME(vectorAuxD.gaussianSmearing(vectorInOut, smearedGauge, nSmear, alphaGauss));
                vectorAuxF.copy(vectorAuxD);
                prop.absorb(vectorAuxF, isc/3, isc%3);
            }
            }
//              prop.applyBoundaries_device(source[DIM_T]);
            };
      
      
           
            
      PLEGMA_Propagator<float> propLT;
      PLEGMA_Propagator<float> propCH;
      char * group;
     
      TIME(computePropagator(propLT, kappa_ud, LIGHT, nsmearGauss));
      TIME(computePropagator(propCH, kappa_c, CHARM, nsmearGauss_c));

      
      PLEGMA_Correlator<float> corr(corr_space, source, maxQsq);
      
      TIME(corr.contractMesonsNew(propCH,propCH));
      asprintf(&group, "charmonium");
      corr.setGroups(group);
      free(group);
      THREAD(corr.writeFile(twop_filename, corr_file_format));
      
      
      
     TIME(corr.contractMesonsNew(propLT,propCH));
      asprintf(&group, "D_meson");
      corr.setGroups(group);
      free(group);
      THREAD(corr.writeFile(twop_filename, corr_file_format));
	
//       TIME(corr.contractBaryons(prop, prop));
//       THREAD(corr.writeFile(twop_filename, corr_file_format));
    }
    
    while(not threads.empty()) {threads.back().join(); threads.pop_back();}
  }

  finalize();
  return 0;
}

