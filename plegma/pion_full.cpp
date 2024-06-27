#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <omp.h>

using namespace plegma;
using namespace quda;

static std::vector<double> runtime;
#define TIC()  runtime.push_back(MPI_Wtime())
#define TOC(str)  PLEGMA_printf("TIME for %s %f sec\n", str, MPI_Wtime()-runtime.back()); \
  runtime.pop_back()
#define TIME(fnc)  TIC(); fnc;	TOC(#fnc)

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
    bool in_use_mg = use_mg;
    QudaInverterType in_inv_type = inv_type;
    double in_tol_hq = tol_hq;
    double in_reliable_delta = reliable_delta;
    int in_niter = niter;

    PLEGMA_Vector<double> vector_stoc;
    PLEGMA_Vector<double> vectorsol;
    vector_stoc.randInit(rand_seed1);

    PLEGMA_Propagator<double> props[nmus];

    PLEGMA_printf("Start producing stochastic vectors and propagators\n");

    for(size_t its = 0; its < tSinks.size(); its++){
      int tsink = tSinks[its];
      PLEGMA_printf("\n ### Calculations for stochastic source %d - %02d begin now ###\n\n",
                    its, tsink);
      
      TIC();
      //We draw a different random vector for every source position
      vector_stoc.stochastic_Z(nroots);
      
      char * src_string;
      asprintf(&src_string, "_id%03d_st%03d_rs%06d", its, tsink, rand_seed1);
      std::string outfilename = twop_filename + src_string + ".h5";
      free(src_string);

      if(access( outfilename.c_str(), F_OK ) != -1) {
	PLEGMA_printf("File %s already exists. Skipping...", outfilename.c_str());
	continue;
      }

      PLEGMA_Correlator<double> corr(corr_space, site({0,0,0,tsink}), maxQsq);

      TIC();
      //Dilution
      vectorsol.absorbTimeslice(vector_stoc, tsink);
      vector_stoc.dilutespin(vectorsol,0);

      for(int imu=0; imu<nmus; imu++){
	mu = mus[imu];
	if(mu*mu>0.01) {
	  use_mg = false;
	  inv_type = get_solver_type("cgnr");
	  tol_hq = 1e-13;
	  reliable_delta = 1e-10;
	  niter = 500;
	} else {
	  use_mg = in_use_mg;
	  inv_type = in_inv_type;
	  tol_hq = in_tol_hq;
	  reliable_delta = in_reliable_delta;
	  niter = in_niter;
	}
	solver.UpdateSolver();
	
	for (int spinindex=0; spinindex<4; ++spinindex){
	  if (spinindex==0)
	    vectorsol.copy(vector_stoc);
	  else
	    vectorsol.diluteSpinDisplace(vector_stoc,spinindex,0);

	  // NOTE!!! Here propagator are made to match those produced by pion_defl_IR_opt.cpp.
	  // So they are not rotated and they have a G5 applied to the source.
	  vectorsol.apply_gamma(G5);
	  TIME(solver.solve(vectorsol, vectorsol));
	  props[imu].absorb(vectorsol, spinindex, 0);
	}
      }
      TOC("building the propagators");

      TIC();
      for(int imu1=0; imu1<nmus; imu1++){
	char * mu_string;
	asprintf(&mu_string, "%+.4e_%+.4e_full", mus[imu1], mus[imu1]);
	std::string dataset = mu_string;
	free(mu_string);
	
	TIME(corr.contractMesonsOpen(props[imu1], props[imu1], false));
	corr.setDatasets((std::vector<std::string>) {dataset});
	TIME(corr.writeHDF5( outfilename ));
	
	for(int imu2=imu1+1; imu2<nmus; imu2++){
	  
	  asprintf(&mu_string, "%+.4e_%+.4e_full", mus[imu1], mus[imu2]);
	  dataset = mu_string;
	  free(mu_string);
	  
	  TIME(corr.contractMesonsOpen(props[imu1], props[imu2], false));
	  corr.setDatasets((std::vector<std::string>) {dataset});
	  TIME(corr.writeHDF5( outfilename ));
	}	
      }
      TOC("contraction and IO");
      TOC("one stochastic source");
    }
  }
  finalize();

  return 0;
}
