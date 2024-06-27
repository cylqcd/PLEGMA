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

#define sign(a) (((a)>=0) ? +1:-1)

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
  double des;
  HGC_options->set("dqed", "dqed used for LIBE", verbosity, des);
  std::string qedfile;
  HGC_options->set("qed-filename", "The path to the QED field", verbosity, qedfile);
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

  if(link_recon!=QUDA_RECONSTRUCT_NO or link_recon_sloppy!=QUDA_RECONSTRUCT_NO or link_recon_precondition!=QUDA_RECONSTRUCT_NO) {
    PLEGMA_error("QED requires QUDA_RECONSTRUCT_NO\n");
  }
  
  {

    // Reading from Lime file and loading to device
    PLEGMA_Gauge<double> gauge;
    gauge.readFile(latfile, LIME_FORMAT);
    gauge.calculatePlaq();

    // Loading to QUDA and computing plaquette also there
    initGaugeQuda(gauge, true);
    plaqQuda();
    
    updateOptions(LIGHT);
    PLEGMA_printf("\n ### Running setup for mu=%.4e ###\n\n",min_mu);
    mu = min_mu;
    TIME(QUDA_solver solver(mu));

    std::vector<std::shared_ptr<PLEGMA_Propagator<double>>> props;
    std::vector<std::shared_ptr<PLEGMA_Propagator<double>>> props2;
    PLEGMA_Propagator<double> prop;
    //PLEGMA_Propagator3D<double> prop3D;
    PLEGMA_GaugeU1<double> gaugeU1;
    //PLEGMA_Vector3D<double> vect3D;
    for(int imu=0; imu<nmus; imu++){
      if(des!=0)
	props.push_back(std::make_shared<PLEGMA_Propagator<double>>(HOST));
      props2.push_back(std::make_shared<PLEGMA_Propagator<double>>(HOST));
    }

   
    for(int its =0; its < nts; its++){
	
      int tsink = tSinks[its];

      char * src_string;
      asprintf(&src_string, "_id%02d_st%03d", its, tsink);
      std::string outfilename = twop_filename + src_string + ".h5";
      free(src_string);

      if(access( outfilename.c_str(), F_OK ) != -1) {
	PLEGMA_printf("\nFile %s already exists. Skipping...\n", outfilename.c_str());
	continue;
      }

      if(des!=0) {
	asprintf(&src_string, "%04d", its);
	std::string U1_conf = qedfile + src_string;
	PLEGMA_printf("\n ### Going to read %s ###\n\n", U1_conf.c_str());
	gaugeU1.readFile(U1_conf, LIME_FORMAT);
	free(src_string);
      }

      
      for(int isgn=0; isgn<3; isgn++) {

	if(des==0 and isgn!=1) continue;

	double phase = (isgn-1)*des;
	if(isgn!=1){
	  PLEGMA_Gauge<double> gauge2;
	  gauge2.copy(gauge);
	  gaugeU1.calculatePlaq(phase);
	  gauge2.qedPhase(gaugeU1, phase);
	  gauge2.calculatePlaq();
	  initGaugeQuda(gauge2, true);
	  plaqQuda();
	  //applyBoundaryConditions(gauge2,true);
	} else {
	  initGaugeQuda(gauge, true);
	}
	PLEGMA_printf("\n ### Running on U1(%d) with de=%.4e ###\n\n",its, phase);
	
	for(int imu=0; imu<nmus; imu++){
	  mu = mus[imu];
	  PLEGMA_printf("\n ### Calculations for stochastic source its=%d(%02d), mu=%+.4e de=%+.4e begin now ###\n\n", its, tsink, mus[imu], phase);
	  PLEGMA_printf("\n ### Updating solver ###\n\n");
	  solver.UpdateSolver();

	  PLEGMA_Vector<double> vector_stoc, vectortmp;
	  
	  //Dilution
	  //We draw a different random vector for every source position
	  vector_stoc.randInit(seeds[its]);
	  vector_stoc.stochastic_Z(nroots);
	  vectortmp.absorbTimeslice(vector_stoc, tsink);
	  vector_stoc.dilutespin(vectortmp,0);
	  
	  for (int spinindex=0; spinindex<4; ++spinindex){
	    if (spinindex==0)
	      vectortmp.copy(vector_stoc);
	    else
	      vectortmp.diluteSpinDisplace(vector_stoc,spinindex,0);
	    //if (imu==0)
	    //  vect3D.absorb(vectortmp, tsink, spinindex);
	    
	    TIME(solver.solve(vectortmp, vectortmp));
	    if(des!=0) {
	      vectortmp.unload();
	      props[imu]->absorb_host(vectortmp, spinindex, isgn);
	    }

	    if(isgn==1) {
	      PLEGMA_Vector<double> vectortmp2;
	      vectortmp2.copy(vectortmp);
	      prop.absorb(vectortmp, spinindex, 0);
	      TIME(solver.solve(vectortmp, vectortmp));
	      prop.absorb(vectortmp, spinindex, 1);
	      //vectortmp.absorb(prop, spinindex, 0);
	      vectortmp2.apply_gamma5();
	      TIME(solver.solve(vectortmp, vectortmp2));
	      prop.absorb(vectortmp, spinindex, 2);
	    }
	  }
	  if(isgn==1) {
	    prop.rotateToPhysicalBase_device(mus[imu]>0? +1:-1);
	    prop.applyBoundaries_device(tsink);
	    prop.unload();
	    props2[imu]->copy(prop, HOST);
	  }
	}
      }

      {
      PLEGMA_Correlator<double> corr(corr_space, site({0,0,0,tsink}), maxQsq);
      //PLEGMA_Correlator<double> corr3D(corr_space, site({0,0,0,tsink}), maxQsq, 1);
      PLEGMA_Propagator<double> prop2;

      if(des!=0) {
      for(int imu1=0; imu1<nmus; imu1++){
	char * mu_string;
	asprintf(&mu_string, "%+.4e_%+.4e", mus[imu1], mus[imu1]);
	std::string dataset = mu_string;
	free(mu_string);
	  
	prop.copy(*props[imu1], HOST);
	prop.load();
	prop.rotateToPhysicalBase_device(mus[imu1]>0? +1:-1);
	prop.applyBoundaries_device(tsink);
	TIME(corr.contractMesonsLIBE(prop, prop));
	corr.setDatasets((std::vector<std::string>) {dataset+"_qed"});
	TIME(corr.writeHDF5( outfilename ));
	TIME(corr.contractMesonsOpenLIBE(prop, prop));
	corr.setDatasets((std::vector<std::string>) {dataset+"_qed_open"});
	TIME(corr.writeHDF5( outfilename ));
	//prop3D.absorb(prop, tsink);
	//TIME(corr3D.contractLoopSIB(vect3D, prop3D));
	//corr3D.setDatasets((std::vector<std::string>) {dataset+"_qed_loop"});
	//TIME(corr3D.writeHDF5( outfilename ));

	for(int imu2=imu1; imu2<nmus; imu2++){
	  if(not (mus[imu1]==-mus[imu2] or (imu1<nmul and imu2>=nmul))){ continue; }
	  char * mu_string;
	  asprintf(&mu_string, "%+.4e_%+.4e", mus[imu1], mus[imu2]);
	  std::string dataset = mu_string;
	  free(mu_string);
	  
	  prop2.copy(*props[imu2], HOST);
	  prop2.load();
	  prop2.rotateToPhysicalBase_device(mus[imu2]>0? +1:-1);
	  prop2.applyBoundaries_device(tsink);
	  TIME(corr.contractMesonsLIBE(prop, prop2));
	  corr.setDatasets((std::vector<std::string>) {dataset+"_qed"});
	  TIME(corr.writeHDF5( outfilename ));
	  TIME(corr.contractMesonsOpenLIBE(prop, prop2));
	  corr.setDatasets((std::vector<std::string>) {dataset+"_qed_open"});
	  TIME(corr.writeHDF5( outfilename ));
	}
      }
      }
      for(int imu1=0; imu1<nmus; imu1++){
	char * mu_string;
	asprintf(&mu_string, "%+.4e_%+.4e", mus[imu1], mus[imu1]);
	std::string dataset = mu_string;
	free(mu_string);
	  
	prop.copy(*props2[imu1], HOST);
	prop.load();
	TIME(corr.contractMesonsSIB(prop, prop));
	corr.setDatasets((std::vector<std::string>) {dataset+"_sib"});
	TIME(corr.writeHDF5( outfilename ));
	TIME(corr.contractMesonsOpenSIB(prop, prop));
	corr.setDatasets((std::vector<std::string>) {dataset+"_sib_open"});
	TIME(corr.writeHDF5( outfilename ));
	//prop3D.absorb(prop, tsink);
	//TIME(corr3D.contractLoopSIB(vect3D, prop3D));
	//corr3D.setDatasets((std::vector<std::string>) {dataset+"_sib_loop"});
	//TIME(corr3D.writeHDF5( outfilename ));
	
	for(int imu2=imu1; imu2<nmus; imu2++){
	  if(not (mus[imu1]==-mus[imu2] or (imu1<nmul and imu2>=nmul))){ continue; }
	  char * mu_string;
	  asprintf(&mu_string, "%+.4e_%+.4e", mus[imu1], mus[imu2]);
	  std::string dataset = mu_string;
	  free(mu_string);
	  
	  prop2.copy(*props2[imu2], HOST);
	  prop2.load();
	  TIME(corr.contractMesonsSIB(prop, prop2));
	  corr.setDatasets((std::vector<std::string>) {dataset+"_sib"});
	  TIME(corr.writeHDF5( outfilename ));
	  TIME(corr.contractMesonsOpenSIB(prop, prop2));
	  corr.setDatasets((std::vector<std::string>) {dataset+"_sib_open"});
	  TIME(corr.writeHDF5( outfilename ));
	}
      }
      }
    }
  }
  finalize();

  return 0;
}
