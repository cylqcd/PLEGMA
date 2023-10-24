#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <omp.h>

using namespace plegma;
using namespace quda;

std::vector<double> runtime;
#define TIME(fnc)  runtime.push_back(MPI_Wtime()); fnc;                 \
  PLEGMA_printf("TIME for "#fnc" %f sec\n", MPI_Wtime()-runtime.back()); \
  runtime.pop_back()

std::vector<std::thread> threads;
//#define THREAD(fnc) threads.push_back(std::thread([=]() { TIME(fnc); }))
#define THREAD(fnc) TIME(fnc)

extern int device;
static std::vector<std::string> listOpt = {"verbosity", "load-gauge","nsrc","src-filename","tSinks","twop-filename", "maxQsq"};
// Note here sinkMom is used as the momentum insertion in the sequential souce, probably has to be renamed to seqMom

int main(int argc, char **argv)
{
  initializeOptions(argc, argv, true, listOpt);
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
  int n_stochastic_samples;
  int nroots=2;
  int rand_seed1=1234;
  HGC_options->set("seed1", "Seed for initialization of stochastic sources for the oet", verbosity, rand_seed1);
  std::vector<double> mus;
  HGC_options->set("extra-mu", "List of additional mu to run", verbosity, mus);

  //=========================================================================================================//
  initializePLEGMA();

  mus.insert(mus.begin(), mu);
  int nmus = mus.size();

  {

    //Storing only the smeared gauge

    // Reading from Lime file and loading to device
    PLEGMA_Gauge<double> gauge;
    gauge.readFile(latfile, LIME_FORMAT);
    gauge.calculatePlaq();

    // Loading to QUDA and computing plaquette also there
    initGaugeQuda(gauge, true);
    plaqQuda();

    updateOptions(LIGHT);
    TIME(QUDA_solver solver(mu));

    PLEGMA_Vector<double> vector_stoc;
    vector_stoc.randInit(rand_seed1);
  
    PLEGMA_printf("Start producing stochastic vectors and propagators\n");

    for(size_t its = 0; its < tSinks.size(); its++){
      int tsink = tSinks[its];
      PLEGMA_printf("\n ### Calculations for stochastic source %d - %02d begin now ###\n\n",
                    its, tsink);

      //We draw a different random vector for every source position
      vector_stoc.stochastic_Z(nroots);
      
      //Dilution
      PLEGMA_Vector<double> vectortmp1;
      vectortmp1.absorbTimeslice(vector_stoc, tsink);
      vector_stoc.dilutespin(vectortmp1,0);

      //Store zero momentum oet propagators
      PLEGMA_Propagator<double> propUP;
      PLEGMA_Propagator<double> propDN;

      for(int imu=0; imu<nmus; imu++){
	mu = mus[imu];
	char * src_string;
	asprintf(&src_string, "_%.8f_id%02d_st%03d", mu, its, tsink);
	std::string outfilename = twop_filename + src_string + ".h5";
	free(src_string);
	
	if(access( outfilename.c_str(), F_OK ) != -1) {
	  PLEGMA_printf("File %s already exists. Skipping...", outfilename.c_str());
	  continue;
	}
      for(int fl=0; fl<2; fl++){

	 if(fl==1) {
	   mu *= -1;
	 }
	 solver.UpdateSolver();
	 
         for (int spinindex=0; spinindex<4; ++spinindex){
           if (spinindex==0)
             vectortmp1.copy(vector_stoc);
	   else
	     vectortmp1.diluteSpinDisplace(vector_stoc,spinindex,0);
           TIME(solver.solve(vectortmp1, vectortmp1));
	   if(fl==0)
	     propUP.absorb(vectortmp1, spinindex, 0);
	   else
	     propDN.absorb(vectortmp1, spinindex, 0);
         }
      }

      propUP.rotateToPhysicalBase_device(+1);
      propUP.applyBoundaries_device(tsink);
      propDN.rotateToPhysicalBase_device(-1);
      propDN.applyBoundaries_device(tsink);

      PLEGMA_Correlator<double> corr(corr_space, site({0,0,0,tsink}), maxQsq);
      TIME(corr.contractMesonsNew(propUP, propUP, false));
      corr.setDatasets((std::vector<std::string>) {"uu"});
      TIME(corr.writeHDF5( outfilename ));
      TIME(corr.contractMesonsOpen(propUP, propUP, false));
      corr.setDatasets((std::vector<std::string>) {"uu_open"});
      TIME(corr.writeHDF5( outfilename ));
      TIME(corr.contractMesonsNew(propDN, propDN, false));
      corr.setDatasets((std::vector<std::string>) {"dd"});
      TIME(corr.writeHDF5( outfilename ));
      TIME(corr.contractMesonsOpen(propDN, propDN, false));
      corr.setDatasets((std::vector<std::string>) {"dd_open"});
      TIME(corr.writeHDF5( outfilename ));
      TIME(corr.contractMesonsNew(propUP, propDN, false));
      corr.setDatasets((std::vector<std::string>) {"ud"});
      TIME(corr.writeHDF5( outfilename ));
      TIME(corr.contractMesonsOpen(propUP, propDN, false));
      corr.setDatasets((std::vector<std::string>) {"ud_open"});
      TIME(corr.writeHDF5( outfilename ));
    }
    }
  } 
  finalize();

  return 0;
}
