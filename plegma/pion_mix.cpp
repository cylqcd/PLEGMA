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
      
      char * src_string;
      asprintf(&src_string, "_id%02d_st%03d", its, tsink);
      std::string outfilename = twop_filename + src_string + ".h5";
      free(src_string);

      //We draw a different random vector for every source position
      vector_stoc.stochastic_Z(nroots);
      
      if(access( outfilename.c_str(), F_OK ) != -1) {
	PLEGMA_printf("File %s already exists. Skipping...", outfilename.c_str());
	continue;
      }

      //Store zero momentum oet propagators
      PLEGMA_Propagator<double> prop1;
      std::vector<std::shared_ptr<PLEGMA_Propagator<double>>> props;
      PLEGMA_Correlator<double> corr(corr_space, site({0,0,0,tsink}), maxQsq);

      {
	//Dilution
	PLEGMA_Vector<double> vectortmp1;
	vectortmp1.absorbTimeslice(vector_stoc, tsink);
	vector_stoc.dilutespin(vectortmp1,0);

	for(int imu=0; imu<nmus; imu++){
	  props.push_back(std::make_shared<PLEGMA_Propagator<double>>(HOST));
	  mu = mus[imu];
	  solver.UpdateSolver();
	
	  for (int spinindex=0; spinindex<4; ++spinindex){
	    if (spinindex==0)
	      vectortmp1.copy(vector_stoc);
	    else
	      vectortmp1.diluteSpinDisplace(vector_stoc,spinindex,0);
	    TIME(solver.solve(vectortmp1, vectortmp1));
	    prop1.absorb(vectortmp1, spinindex, 0);
	  }
	  prop1.rotateToPhysicalBase_device(mu>0? +1:-1);
	  prop1.applyBoundaries_device(tsink);
	  
	  char * mu_string;
	  asprintf(&mu_string, "%+.4e_%+.4e", mus[imu], mus[imu]);
	  std::string dataset = mu_string;
	  free(mu_string);
	  
	  TIME(corr.contractMesonsNew(prop1, prop1, false));
	  corr.setDatasets((std::vector<std::string>) {dataset});
	  TIME(corr.writeHDF5( outfilename ));
	  TIME(corr.contractMesonsOpen(prop1, prop1, false));
	  corr.setDatasets((std::vector<std::string>) {dataset+"_open"});
	  TIME(corr.writeHDF5( outfilename ));

	  prop1.unload();
	  props[imu]->copy(prop1, HOST);
	}
      }
      
      PLEGMA_Propagator<double> prop2;
      
      for(int imu1=0; imu1<nmus; imu1++){
	prop1.copy(*props[imu1], HOST);
	prop1.load();
	for(int imu2=imu1+1; imu2<nmus; imu2++){
	  prop2.copy(*props[imu2], HOST);
	  prop2.load();
	  
	  char * mu_string;
	  asprintf(&mu_string, "%+.4e_%+.4e", mus[imu1], mus[imu2]);
	  std::string dataset = mu_string;
	  free(mu_string);
	  
	  TIME(corr.contractMesonsNew(prop1, prop2, false));
	  corr.setDatasets((std::vector<std::string>) {dataset});
	  TIME(corr.writeHDF5( outfilename ));
	  TIME(corr.contractMesonsOpen(prop1, prop2, false));
	  corr.setDatasets((std::vector<std::string>) {dataset+"_open"});
	  TIME(corr.writeHDF5( outfilename ));
	}
      }
    }
  }
  finalize();

  return 0;
}
