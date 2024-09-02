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

#define abs(a) (((a)>=0) ? (a):-(a))

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
  std::vector<double> mul;
  HGC_options->set("extra-mul", "List of additional mu to run", verbosity, mul);
  std::vector<double> muh;
  HGC_options->set("muh", "List of additional mu to run", verbosity, muh);

  //=========================================================================================================//
  initializePLEGMA();

  mul.insert(mul.begin(), mu);
  int nmul = mul.size();
  int nmuh = muh.size();
  int nts = tSinks.size();

  std::vector<double> mus;
  for(int i=0; i<nmul; i++) {
    mus.push_back(mul[i]);
  }
  for(int i=0; i<nmuh; i++) {
    mus.push_back(muh[i]);
  }
  int nmus = mus.size();
  
  srand(rand_seed1);
  std::vector<int> seeds;
  for(int i=0; i<nts; i++) seeds.push_back(rand());
  seeds[0] = rand_seed1;
  
  double min_mu=mu;
  for(int i=0; i<nmus; i++) {
    if(min_mu>abs(mus[i])){
      min_mu=abs(mus[i]);
    }
  }
  double mu_setup=min_mu;
  {

    // Reading from Lime file and loading to device
    PLEGMA_Gauge<double> gauge;
    gauge.readFile(latfile, LIME_FORMAT);
    gauge.calculatePlaq();
        
    //PLEGMA_GaugeU1<double> gaugeU1;
    //gauge.readFile(qedfile, LIME_FORMAT);
    
    PLEGMA_Gauge<double> contractGauge(BOTH);
    // Gauge for contractions
    contractGauge.copy(gauge);
    // apply boundary conditions since is needed for the covariant derivative
    applyBoundaryConditions(contractGauge,true);
    
    // Loading to QUDA and computing plaquette also there
    initGaugeQuda(gauge, true);
    plaqQuda();

    updateOptions(LIGHT);
    PLEGMA_printf("\n ### Running setup for mu=%.4e ###\n\n",min_mu);
    mu = min_mu;
    TIME(QUDA_solver solver(mu));

    PLEGMA_Propagator<double> props[nmus];
      
    for(int its =0; its < nts; its++){

      int tsink = tSinks[its];
      PLEGMA_Correlator<double> corr(corr_space, site({0,0,0,tsink}), maxQsq);
      PLEGMA_Correlator<double> corr3D(corr_space, site({0,0,0,tsink}), maxQsq, 1);
      
      char * src_string;
      asprintf(&src_string, "_id%02d_st%03d", its, tsink);
      std::string outfilename = twop_filename + src_string + ".h5";
      free(src_string);
      
      if(access( outfilename.c_str(), F_OK ) != -1) {
	PLEGMA_printf("\nFile %s already exists. Skipping...\n", outfilename.c_str());
	continue;
      }
      
      for(int imu=0; imu<nmus; imu++){
	mu = mus[imu];
	PLEGMA_printf("\n ### Calculations for stochastic source its=%d(%02d), mu=%+.4e  begin now ###\n\n",
		      its, tsink, mus[imu]);
	PLEGMA_printf("\n ### Updating solver ###\n\n");
	TIME(solver.UpdateSolver());


	//Dilution
	PLEGMA_Vector<double> vector_stoc, vectortmp1;
	PLEGMA_Propagator3D<double> prop3D;
	PLEGMA_Vector3D<double> vect3D;
	//We draw a different random vector for every source position
	vector_stoc.randInit(seeds[its]);
	vector_stoc.stochastic_Z(nroots);
	vectortmp1.absorbTimeslice(vector_stoc, tsink);
	vector_stoc.dilutespin(vectortmp1,0);

	for (int spinindex=0; spinindex<4; ++spinindex){
	  if (spinindex==0)
	    vectortmp1.copy(vector_stoc);
	  else
	    vectortmp1.diluteSpinDisplace(vector_stoc,spinindex,0);
	  vect3D.absorb(vectortmp1, tsink, spinindex);
	  TIME(solver.solve(vectortmp1, vectortmp1));
	  props[imu].absorb(vectortmp1, spinindex, 0);
	  TIME(solver.solve(vectortmp1, vectortmp1));
	  props[imu].absorb(vectortmp1, spinindex, 1);
	  vectortmp1.absorb(props[imu], spinindex, 0);
	  vectortmp1.apply_gamma5();
	  TIME(solver.solve(vectortmp1, vectortmp1));
	  props[imu].absorb(vectortmp1, spinindex, 2);
	}
	
	props[imu].rotateToPhysicalBase_device(mu>0? +1:-1);
	props[imu].applyBoundaries_device(tsink);
	prop3D.absorb(props[imu], tsink);
	
	char * mu_string;
	asprintf(&mu_string, "%+.4e_%+.4e", mus[imu], mus[imu]);
	std::string dataset = mu_string;
	free(mu_string);
	  
	TIME(corr.contractMesonsSIB(props[imu], props[imu]));
	corr.setDatasets((std::vector<std::string>) {dataset});
	TIME(corr.writeHDF5( outfilename ));
	TIME(corr.contractMesonsOpenSIB(props[imu], props[imu]));
	corr.setDatasets((std::vector<std::string>) {dataset+"_open"});
	TIME(corr.writeHDF5( outfilename ));
	TIME(corr3D.contractLoopSIB(vect3D, prop3D));
	corr3D.setDatasets((std::vector<std::string>) {dataset+"_loop"});
	TIME(corr3D.writeHDF5( outfilename ));

      }
      
      PLEGMA_Propagator<double> prop2;
      
      for(int imu1=0; imu1<nmus; imu1++){
	for(int imu2=0; imu2<nmus; imu2++){
	  if(not (mus[imu1]==-mus[imu2] or (imu1<nmul and imu2>=nmul) or (imu2<nmul and imu1>=nmul))){ continue; }
	  char * mu_string;
	  asprintf(&mu_string, "%+.4e_%+.4e", mus[imu1], mus[imu2]);
	  std::string dataset = mu_string;
	  free(mu_string);
	  
	  TIME(corr.contractMesonsSIB(props[imu1], props[imu2]));
	  corr.setDatasets((std::vector<std::string>) {dataset});
	  TIME(corr.writeHDF5( outfilename ));
	  TIME(corr.contractMesonsOpenSIB(props[imu1], props[imu2]));
	  corr.setDatasets((std::vector<std::string>) {dataset+"_open"});
	  TIME(corr.writeHDF5( outfilename ));
	}
      }
    }
  }
  finalize();

  return 0;
}
