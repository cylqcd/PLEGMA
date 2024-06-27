#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <PLEGMA_Hprobing.h>
#include <stdio.h>
using namespace plegma;
using namespace quda;

static std::vector<double> runtime;
#define TIME(fnc)  runtime.push_back(MPI_Wtime()); fnc;                 \
  PLEGMA_printf("TIME for "#fnc" %f sec\n", MPI_Wtime()-runtime.back()); \
  runtime.pop_back()

static std::vector<std::string> listOpt = {"verbosity", "load-gauge", "Eig-isACC", "Eig-PolyDeg", "Eig-amin",
					   "Eig-amax", "Eig-spectrumPart", "Eig-tol", "Eig-maxIters", "Eig-NeV",
#if defined(HAVE_ARPACK) || defined(QUDAEIG)
					   "Eig-NkV", "Eig-logFile",
#endif
					   "nsrc", "maxQsq", "rng-seed", "corr-file-format", "twop-filename", "tSinks"
};


int main(int argc, char **argv)
{
  initializeOptions(argc, argv, true, listOpt); // Put list of options later
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
  std::string loopsPrefix="./";
  HGC_options->set("output-path", "Path to the directory to dump results", verbosity, loopsPrefix);
  std::string Eig_outputFile = "./eigsVdagG5V.dat";
  HGC_options->set("Eig-outputFile", "Path to dump the eigenvalues and vdag g5 v if low-modes-recon is enabled",verbosity, Eig_outputFile);
  bool isReadEigenVecs = false, isWriteEigenVecs = false, isDeviceEigenVecs = false;
  std::string fnameEigenVecsPrefix="";
  HGC_options->set("deviceEigenVectors", "Where we want to store EigenVectors on device", verbosity, isDeviceEigenVecs);
  HGC_options->set("readEigenVectors", "Where we want to read EigenVectors from file", verbosity, isReadEigenVecs);
  HGC_options->set("writeEigenVectors", "Where we want to read EigenVectors from file", verbosity, isWriteEigenVecs);
  HGC_options->set("prefixEigenVecsFile", "Path with prefix for the filenames of the eigenvectors", verbosity, fnameEigenVecsPrefix);
#ifdef QUDAEIG
  int batched_rotate = 1;
  HGC_options->set("batched-rotate", "The size of the batch during Ritz rotation", verbosity, batched_rotate);
#endif
  int nroots=2;
  int rand_seed1=1234;
  HGC_options->set("seed1", "Seed for initialization of stochastic sources for the oet", verbosity, rand_seed1);
  std::vector<double> mus;
  HGC_options->set("extra-mu", "List of additional mu to run", verbosity, mus);
  bool postProj = false;
  HGC_options->set("postProj", "Whether we project also after inverting", verbosity, postProj);

  //=========================================================================================================//
  TIME(initializePLEGMA());

  mus.insert(mus.begin(), mu);
  int nmus = mus.size();

  // Reading from Lime file and loading to device
  PLEGMA_Gauge<double> gauge;
  TIME(gauge.readFile(latfile, LIME_FORMAT));
  gauge.calculatePlaq();

  // Loading to QUDA and computing plaquette also there
  TIME(initGaugeQuda(gauge, true));
  plaqQuda();

  // apply boundary conditions since is needed for the covariant derivative
  // this needs to be done after initGaugeQuda otherwise causes troubles
  applyBoundaryConditions(gauge,true);

  // In case we need LMR we need to compute the eigenvectors
  EigSolver *eigSol = nullptr;
  if(Eig_NeV>0) {
#if defined(HAVE_EIGENSOLVER)
    EigSolverParams eigParam;
    eigParam.NeV = Eig_NeV;
    eigParam.isACC = Eig_isACC;
    eigParam.littleD = true;
    eigParam.fastio = true;
    eigParam.deviceAlloc = isDeviceEigenVecs;
    eigParam.PolyDeg = Eig_PolyDeg;
    eigParam.amin = Eig_amin;
    eigParam.amax = Eig_amax;
    eigParam.spectrumPart = Eig_spectrumPart;
    eigParam.tol = Eig_tol;
    eigParam.maxIters = Eig_maxIters;
#ifdef QUDAEIG
    eigParam.batched_rotate = batched_rotate;
#endif
#if defined(HAVE_ARPACK) || defined(QUDAEIG)
    eigParam.NkV = Eig_NkV;
    eigParam.logFile = Eig_logFile;
#else
    PLEGMA_error("No arpack or primme is compiled");
#endif
    TIME(eigSol = new EigSolver(eigParam, dslash_type, isReadEigenVecs, isWriteEigenVecs, fnameEigenVecsPrefix, true));

    char * src_string;
    asprintf(&src_string, "_exact_exact_nvec%03d", Eig_NeV);
    std::string outfilename = twop_filename + src_string + ".h5";
    free(src_string);

    if(access( outfilename.c_str(), F_OK ) == -1) {
      {
	HDF5 writer(outfilename);
	std::vector<hsize_t> shape = {Eig_NeV, Eig_NeV, 2};
	writer.write_dataset("littleD", (double*) eigSol->getLittleD(), shape);
      }
    
      PLEGMA_Correlator<double> corr(corr_space, site({0,0,0,0}), maxQsq);
      TIME(corr.contractEigVecs(eigSol->getEigVecs(), Eig_NeV, eigSol->getSize_per_Vec()*2, isDeviceEigenVecs));
      TIME(corr.writeHDF5( outfilename ));
    }

    std::size_t foundPos = latfile.find("conf.");
    if(foundPos == std::string::npos) PLEGMA_error("Cannot find (conf.) in configuration path to get confID");
    std::string confID = latfile.substr(foundPos+5,latfile.length());
    //TIME(eigSol->qLoops_exact(gauge,loopsPrefix,confID,corr_file_format));
#else
    PLEGMA_error("No eigenSolver is compiled");
#endif
  }
  
  TIME(QUDA_solver solver(mu));

  PLEGMA_Vector<double> vector_stoc;
  vector_stoc.randInit(rand_seed1);
  
  PLEGMA_printf("Start producing stochastic vectors and propagators\n");

  for(size_t its = 0; its < tSinks.size(); its++){
    int tsink = tSinks[its];
    PLEGMA_printf("\n ### Calculations for stochastic source %d - %02d begin now ###\n\n",
		  its, tsink);
      
    char * src_string;
    asprintf(&src_string, "_nev%03d_id%02d_st%03d", Eig_NeV, its, tsink);
    std::string outfilename = twop_filename + src_string + ".h5";
    free(src_string);
    
    //We draw a different random vector for every source position
    vector_stoc.stochastic_Z(nroots);
      
    if(access( outfilename.c_str(), F_OK ) != -1) {
      PLEGMA_printf("File %s already exists. Skipping...", outfilename.c_str());
      continue;
    }

    //Store zero momentum oet propagators
    PLEGMA_Propagator<double> prop1, prop2;
    std::vector<std::shared_ptr<PLEGMA_Propagator<double>>> props;
    std::vector<std::shared_ptr<PLEGMA_Propagator<double>>> props2;
    PLEGMA_Correlator<double> corr(corr_space, site({0,0,0,tsink}), maxQsq);

    {
      //Dilution
      PLEGMA_Vector<double> vectorsrc[4], vectorsol;
      vectorsrc[0].absorbTimeslice(vector_stoc, tsink);
      vector_stoc.dilutespin(vectorsrc[0],0);

      double spinVals[4*Eig_NeV*2];
      
      for(int imu=0; imu<nmus; imu++){
	props.push_back(std::make_shared<PLEGMA_Propagator<double>>(HOST));
	props2.push_back(std::make_shared<PLEGMA_Propagator<double>>(HOST));
	mu = mus[imu];
	TIME(solver.UpdateSolver());

	for (int spinindex=0; spinindex<4; ++spinindex){

	  if(imu==0) {
	    if (spinindex==0)
	      vectorsrc[spinindex].copy(vector_stoc);
	    else
	      vectorsrc[spinindex].diluteSpinDisplace(vector_stoc,spinindex,0);
	    
	    if(Eig_NeV>0) {
	      TIME(eigSol->projectVector(vectorsrc[spinindex], spinVals + spinindex*Eig_NeV*2,tsink,spinindex));
	    }
	    vectorsrc[spinindex].apply_gamma(G5);
	  }
	  TIME(solver.solve(vectorsol, vectorsrc[spinindex]));
	  if(postProj and Eig_NeV>0) {
	    TIME(eigSol->projectVector(vectorsol));
	  }
	  prop1.absorb(vectorsol, spinindex, 0);

	  if (spinindex==0)
	    vectorsol.copy(vector_stoc);
	  else
	    vectorsol.diluteSpinDisplace(vector_stoc,spinindex,0);
	  vectorsol.apply_gamma(G5);
	  TIME(solver.solve(vectorsol, vectorsol));
	  prop2.absorb(vectorsol, spinindex, 0);
	  
	}
	char * mu_string;
	asprintf(&mu_string, "%+.4e_%+.4e", mus[imu], mus[imu]);
	std::string dataset = mu_string;
	free(mu_string);

	if(Eig_NeV>0) {
	  PLEGMA_Propagator<double> tmp;
	  std::complex<double> evals[Eig_NeV];
	  for(int i=0; i<Eig_NeV; i++) {
	    evals[i] = eigSol->getLittleD()[i*(Eig_NeV+1)];
	  }
	  TIME(tmp.buildExactPropagator(spinVals, (double*) evals, eigSol->getEigVecs(), Eig_NeV, eigSol->getSize_per_Vec()*2, isDeviceEigenVecs));
	  
	  TIME(corr.contractMesonsOpenDefl(tmp, prop1));
	  corr.setDatasets((std::vector<std::string>) {dataset+"_stoch_exact_summed_open"});
	  TIME(corr.writeHDF5( outfilename ));	  

	  TIME(corr.contractMesonsOpenSIR(tmp, tmp));
	  corr.setDatasets((std::vector<std::string>) {dataset+"_stoch_stoch_IR"});
	  TIME(corr.writeHDF5( outfilename ));	  
	}
	
	asprintf(&mu_string, "%+.4e_%+.4e_stoch_stoch", mus[imu], mus[imu]);
	dataset = mu_string;
	free(mu_string);
	  
	TIME(corr.contractMesonsOpen(prop1, prop1, false));
	corr.setDatasets((std::vector<std::string>) {dataset+"_UV"});
	TIME(corr.writeHDF5( outfilename ));

	prop1.unload();
	props[imu]->copy(prop1, HOST);

      	TIME(corr.contractMesonsOpen(prop2, prop2, false));
	corr.setDatasets((std::vector<std::string>) {dataset+"_full"});
	TIME(corr.writeHDF5( outfilename ));

	prop2.unload();
	props2[imu]->copy(prop2, HOST);
      }
    }
    
    for(int imu1=0; imu1<nmus; imu1++){
      prop1.copy(*props[imu1], HOST);
      prop1.load();
      for(int imu2=imu1+1; imu2<nmus; imu2++){
	prop2.copy(*props[imu2], HOST);
	prop2.load();
	  
	char * mu_string;
	asprintf(&mu_string, "%+.4e_%+.4e_stoch_stoch", mus[imu1], mus[imu2]);
	std::string dataset = mu_string;
	free(mu_string);
	  
	TIME(corr.contractMesonsOpen(prop1, prop2, false));
	corr.setDatasets((std::vector<std::string>) {dataset+"_UV"});
	TIME(corr.writeHDF5( outfilename ));
      }
    }
    for(int imu1=0; imu1<nmus; imu1++){
      prop1.copy(*props2[imu1], HOST);
      prop1.load();
      for(int imu2=imu1+1; imu2<nmus; imu2++){
	prop2.copy(*props2[imu2], HOST);
	prop2.load();
	  
	char * mu_string;
	asprintf(&mu_string, "%+.4e_%+.4e_stoch_stoch", mus[imu1], mus[imu2]);
	std::string dataset = mu_string;
	free(mu_string);
	  
	TIME(corr.contractMesonsOpen(prop1, prop2, false));
	corr.setDatasets((std::vector<std::string>) {dataset+"_full"});
	TIME(corr.writeHDF5( outfilename ));
      }
    }
  }
  finalize();

  return 0;
}
