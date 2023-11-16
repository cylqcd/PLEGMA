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
  std::vector<double> dmus;
  HGC_options->set("dmu", "dmu used for LIBE", verbosity, dmus);
  std::vector<double> dks;
  HGC_options->set("dkappa", "dkappa used for LIBE", verbosity, dks);
  std::vector<double> des;
  HGC_options->set("dqed", "dqed used for LIBE", verbosity, des);
  std::string qedfile;
  HGC_options->set("qed-filename", "The path to the QED field", verbosity, qedfile);
  int dnts = 2;
  HGC_options->set("nts_inner", "Number of timeslices to run in the inner loop", verbosity, dnts);

  //=========================================================================================================//
  initializePLEGMA();

  mul.insert(mul.begin(), mu);
  int nmul = mul.size();
  int nmuh = muh.size();
  int nts = tSinks.size();
  int ndmus = dmus.size();
  int ndks = dks.size();
  int ndms = ndmus+ndks;
  double kappa0 = kappa;
  double mass0 = mass;

  std::vector<double> mus;
  for(int i=0; i<nmul; i++) {
    mus.push_back(mul[i]);
  }
  for(int i=0; i<nmuh; i++) {
    mus.push_back(muh[i]);
  }
  int nmus = mus.size();
  if(nmus != nmul+nmuh) {
    PLEGMA_error("Foo\n");
  }
  
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
  double min_mu2=min_mu;
  for(int i=0; i<ndmus; i++) {
    if(min_mu2>abs(min_mu*(1+dmus[i]))){
      min_mu2=abs(min_mu*(1+dmus[i]));
    }
  }
  double mu_setup=min_mu2;
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
    PLEGMA_printf("\n ### Running setup for mu=%.4e ###\n\n",min_mu2);
    mu = min_mu2;
    TIME(QUDA_solver solver(mu));

    int its2=0;
    while(its2 < nts){
      std::vector<std::shared_ptr<PLEGMA_Propagator<double>>> props;
      props.reserve(nmus*dnts);

      for(int imu=0; imu<nmus; imu++){
	if(kappa0 == -1.0) {
	  mass = mass0;
	} else {
	  kappa = kappa0;
	}
	mu = mus[imu];
	PLEGMA_printf("\n ### Updating solver ###\n\n");
	TIME(solver.UpdateSolver());

	for(int dits = 0; dits < dnts; dits++){
	  int its = its2+dits;
	  if (its>=nts) {
	    props.push_back(std::make_shared<PLEGMA_Propagator<double>>(NONE));
	    break;
	  }
	  int tsink = tSinks[its];
	  PLEGMA_printf("\n ### Calculations for stochastic source its=%d(%02d), mu=%+.4e  begin now ###\n\n",
		      its, tsink, mus[imu]);

	  char * src_string;
	  asprintf(&src_string, "_id%02d_st%03d", its, tsink);
	  std::string outfilename = twop_filename + src_string + ".h5";
	  free(src_string);

	
	  PLEGMA_Propagator<double> prop1;
	
	  //Dilution
	  PLEGMA_Vector<double> vector_stoc, vectortmp1;
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
	    TIME(solver.solve(vectortmp1, vectortmp1));
	    prop1.absorb(vectortmp1, spinindex, 0);
	  }
	  PLEGMA_Propagator<double> prop0;
	  prop0.copy(prop1);

	  prop1.rotateToPhysicalBase_device(mu>0? +1:-1);
	  prop1.applyBoundaries_device(tsink);
	
	  char * mu_string;
	  asprintf(&mu_string, "%+.4e_%+.4e", mus[imu], mus[imu]);
	  std::string dataset = mu_string;
	  free(mu_string);
	  
	  PLEGMA_Correlator<double> corr(corr_space, site({0,0,0,tsink}), maxQsq);
	  TIME(corr.contractMesonsNew(prop1, prop1, false));
	  corr.setDatasets((std::vector<std::string>) {dataset});
	  TIME(corr.writeHDF5( outfilename ));
	  TIME(corr.contractMesonsOpen(prop1, prop1, false));
	  corr.setDatasets((std::vector<std::string>) {dataset+"_open"});
	  TIME(corr.writeHDF5( outfilename ));
	  TIME(corr.contractMesons1ps(prop0, prop0, contractGauge, false));
	  corr.setDatasets((std::vector<std::string>) {dataset+"_1ps"});
	  TIME(corr.writeHDF5( outfilename ));

	  TIME(prop1.unload());
	  props.push_back(std::make_shared<PLEGMA_Propagator<double>>(HOST));
	  props[imu*dnts+dits]->copy(prop1, HOST);
	}
      }
	

      for(int dits = 0; dits < dnts; dits++){
	int its = its2+dits;
	if (its>=nts) break;
	int tsink = tSinks[its];
      
	char * src_string;
	asprintf(&src_string, "_id%02d_st%03d", its, tsink);
	std::string outfilename = twop_filename + src_string + ".h5";
	free(src_string);
      
	PLEGMA_Correlator<double> corr(corr_space, site({0,0,0,tsink}), maxQsq);
      
	PLEGMA_Propagator<double> prop1;
	PLEGMA_Propagator<double> prop2;
      
	for(int imu1=0; imu1<nmus; imu1++){
	  prop1.copy(*props[imu1*dnts+dits], HOST);
	  TIME(prop1.load());
	  for(int imu2=imu1+1; imu2<nmus; imu2++){
	    if(not (mus[imu1]==-mus[imu2] or (imu1<nmul and imu2>=nmul))){ continue; }
	    
	    prop2.copy(*props[imu2*dnts+dits], HOST);
	    TIME(prop2.load());
	  
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

      for(int idm=0; idm<ndms; idm++ ) {

	std::vector<std::shared_ptr<PLEGMA_Propagator<double>>> dprops;
	dprops.reserve(nmus*dnts);
      
	double dmu = (idm<ndmus) ? dmus[idm]:0;
	double dk = (idm>=ndmus) ? dks[idm-ndmus]:0;
      
	for(int imu=0; imu<nmus; imu++){
	  mu = mus[imu]+mus[imu]*dmu;
	  if(kappa0 == -1.0) {
	    mass = mass0+dk;
	  } else {
	    kappa = kappa0+dk;
	  }
	  PLEGMA_printf("\n ### Updating solver ###\n\n");
	  TIME(solver.UpdateSolver());    

	  for(int dits = 0; dits < dnts; dits++){
	    int its = its2+dits;
	    if (its>=nts) {
	      dprops.push_back(std::make_shared<PLEGMA_Propagator<double>>(NONE));
	      break;
	    }
	    int tsink = tSinks[its];
	    PLEGMA_printf("\n ### Calculations for stochastic source its=%d(%02d), mu=%+.4e, dmu=%+.4e, dk=%+.4e  begin now ###\n\n",
			  its, tsink, mus[imu], dmu, dk);
      
	    char * src_string;
	    asprintf(&src_string, "_id%02d_st%03d", its, tsink);
	    std::string outfilename = twop_filename + src_string + ".h5";
	    free(src_string);

	    PLEGMA_Propagator<double> prop1;
	
	    //Dilution
	    PLEGMA_Vector<double> vector_stoc,vectortmp1;
	    vector_stoc.randInit(seeds[its]);
	    vector_stoc.stochastic_Z(nroots);
	    vectortmp1.absorbTimeslice(vector_stoc, tsink);
	    vector_stoc.dilutespin(vectortmp1,0);

	    for (int spinindex=0; spinindex<4; ++spinindex){
	      if (spinindex==0)
		vectortmp1.copy(vector_stoc);
	      else
		vectortmp1.diluteSpinDisplace(vector_stoc,spinindex,0);
	      TIME(solver.solve(vectortmp1, vectortmp1));
	      prop1.absorb(vectortmp1, spinindex, 0);
	    }
	    PLEGMA_Propagator<double> prop0;
	    prop0.copy(prop1);
	    prop1.rotateToPhysicalBase_device(mu>0? +1:-1);
	    prop1.applyBoundaries_device(tsink);

	    /*
	    char * mu_string;
	    asprintf(&mu_string, "%+.4e_%+.4e_dmu%+.4e_dk%+.4e_both", mus[imu], mus[imu],dmu,dk);
	    std::string dataset = mu_string;
	    free(mu_string);
	  
	    PLEGMA_Correlator<double> corr(corr_space, site({0,0,0,tsink}), maxQsq);
	    TIME(corr.contractMesonsNew(prop1, prop1, false));
	    corr.setDatasets((std::vector<std::string>) {dataset});
	    TIME(corr.writeHDF5( outfilename ));
	    TIME(corr.contractMesonsOpen(prop1, prop1, false));
	    corr.setDatasets((std::vector<std::string>) {dataset+"_open"});
	    TIME(corr.writeHDF5( outfilename ));
	    TIME(corr.contractMesons1ps(prop0, prop0, contractGauge, false));
	    corr.setDatasets((std::vector<std::string>) {dataset+"_1ps"});
	    TIME(corr.writeHDF5( outfilename ));
	    */	    
	    TIME(prop1.unload());
	    dprops.push_back(std::make_shared<PLEGMA_Propagator<double>>(HOST));
	    dprops[imu*dnts+dits]->copy(prop1, HOST);
	  }
	}

	for(int dits = 0; dits < dnts; dits++){
	  int its = its2+dits;
	  if (its>=nts) break;
	  int tsink = tSinks[its];
      
	  char * src_string;
	  asprintf(&src_string, "_id%02d_st%03d", its, tsink);
	  std::string outfilename = twop_filename + src_string + ".h5";
	  free(src_string);
      
	  PLEGMA_Correlator<double> corr(corr_space, site({0,0,0,tsink}), maxQsq);
      
	  PLEGMA_Propagator<double> prop1;
	  PLEGMA_Propagator<double> prop2;

	  /*
	  for(int imu1=0; imu1<nmus; imu1++){
	    prop1.copy(*dprops[imu1*dnts+dits], HOST);
	    TIME(prop1.load());
	    for(int imu2=imu1+1; imu2<nmus; imu2++){
	      if(not (mus[imu1]==-mus[imu2] or (imu1<nmul and imu2>=nmul))){ continue; }
	      
	      prop2.copy(*dprops[imu2*dnts+dits], HOST);
	      TIME(prop2.load());
	  
	      char * mu_string;
	      asprintf(&mu_string, "%+.4e_%+.4e_dmu%+.4e_dk%+.4e_both", mus[imu1], mus[imu2],dmu,dk);
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
	  */
	  for(int imu1=0; imu1<nmus; imu1++){
	    prop1.copy(*props[imu1*dnts+dits], HOST);
	    TIME(prop1.load());
	    for(int imu2=0; imu2<nmus; imu2++){
	      if(not (mus[imu1]==-mus[imu2] or (imu1<nmul and imu2>=nmul) or (imu2<nmul and imu1>=nmul))){ continue; }
	      
	      prop2.copy(*dprops[imu2*dnts+dits], HOST);
	      TIME(prop2.load());
	  
	      char * mu_string;
	      asprintf(&mu_string, "%+.4e_%+.4e_dmu%+.4e_dk%+.4e", mus[imu1], mus[imu2],dmu,dk);
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
      its2+=dnts;
    }
  }
  finalize();

  return 0;
}
