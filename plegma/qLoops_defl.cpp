#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <PLEGMA_BLAS.h>
#include <stdio.h>
using namespace plegma;
using namespace quda;

static std::vector<double> runtime;
#define TIC()  runtime.push_back(MPI_Wtime())
#define TOC(str)  PLEGMA_printf("TIME for %s %f sec\n", str, MPI_Wtime()-runtime.back()); \
  runtime.pop_back()

#define TIME(fnc)  TIC(); fnc;	TOC(#fnc)
  //  PLEGMA_printf("TIME for "#fnc" %f sec\n", MPI_Wtime()-runtime.back()); \
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
  
  std::vector<double> nevs;
  HGC_options->set("extra-nev", "List of additional nev to run", verbosity, nevs);

  int k_probing = 0;
  bool spinColorDil = false;
  HGC_options->set("k-probing", "Hierarchical probing, with distance D=2**k (Options:0,1,2,3,...) (0 means No probing)",verbosity,k_probing);
  int hadamLow=0;
  int Nhadam = (k_probing>0) ? 2*std::pow(2,N_DIMS*(k_probing-1)) : 1;
  int hadamHgh = Nhadam;
  std::string loopsPrefix="./";
  HGC_options->set("output-path", "Path to the directory to dump results", verbosity, loopsPrefix);
  // Usefull if one wants to go up to high distance in probing and walltime does not allow to do all Hadamard vectors
  HGC_options->set("hadamard-low", "From which Hadamard vector to start (Options:[0,max))", verbosity, hadamLow);
  HGC_options->set("hadamard-high", "Up to which Hadamard vector to stop (Options: 0>= , <=max) (default max)", verbosity, hadamHgh);
  if((k_probing>0) && (hadamLow<0 || hadamHgh<0)) PLEGMA_error("Negative values for number of Hadamard vector not allowed");
  if((k_probing>0) && (hadamLow>hadamHgh))  PLEGMA_error("hadamard-high should be > hadamard-low");
  if((k_probing>0) && (hadamHgh>Nhadam)) PLEGMA_error("hadamard-high should be <= from max number of Hadamard vectors");
  HGC_options->set("spin-color-dil", "Whether we want spin color dilution",verbosity,spinColorDil);
  int Nsc = spinColorDil ? N_SPINS*N_COLS : 1;
  
  bool oneDLoops = true;
  bool accumFlag = true;
  bool twoDLoops = false;
  int NdumpStep = 1;
  HGC_options->set("oneD-loops", "Whether we want to use covariant derivative for the quark loops calculation", verbosity, oneDLoops);
  HGC_options->set("twoD-loops", "Whether we want to use two covariant derivative for the quark loops calculation", verbosity, twoDLoops);
  HGC_options->set("accum-loops", "Accumulate loops over the stochastic source vectors", verbosity, accumFlag);
  HGC_options->set("dump-step", "If accumulation is ON, Every how many stochastic vector to dump results", verbosity, NdumpStep);

  //=========================================================================================================//
  TIME(initializePLEGMA());

  mus.insert(mus.begin(), mu);
  int nmus = mus.size();
  nevs.insert(nevs.begin(), Eig_NeV);
  for(int inev=1; inev < nevs.size(); inev++){ // Loop over NeV
    if(nevs[inev]>nevs[0]) PLEGMA_error("extra-nev can only be smaller");
  }

  // Reading from Lime file and loading to device
  PLEGMA_Gauge<double> gauge;
  TIME(gauge.readFile(latfile, LIME_FORMAT));
  gauge.calculatePlaq();

  // Loading to QUDA and computing plaquette also there
  TIME(initGaugeQuda(gauge, true));
  plaqQuda();

  // apply boundary conditions since is needed for the covariant derivative
  // this needs to be done after initGaugeQuda otherwise causes troubles
  // applyBoundaryConditions(gauge,true);

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

    std::size_t foundPos = latfile.find("conf.");
    if(foundPos == std::string::npos) PLEGMA_error("Cannot find (conf.) in configuration path to get confID");
    std::string confID = latfile.substr(foundPos+5,latfile.length());
    
    for(int inev=0; inev < nevs.size(); inev++){ // Loop over NeV  
      int nev = nevs[inev];
      TIME(eigSol->qLoops_exact(gauge,loopsPrefix,confID,nev,HDF5_FORMAT,maxQsq,oneDLoops,twoDLoops));
    }
#else
    PLEGMA_error("No eigenSolver is compiled");
#endif
  }



  if(oneDLoops) gauge.communicateGhost();
   checkCudaError();

   PLEGMA_FT<double> *ft[2]={nullptr};
   ft[0] = new PLEGMA_FT<double>(maxQsq, 3);
   if(twoDLoops) ft[1] = new PLEGMA_FT<double>(0, 3);
   
   PLEGMA_QLoops<double> qloops_std(BOTH,NO_GHOSTS,true,oneDLoops,twoDLoops);
   PLEGMA_QLoops<double> qloops_gen(BOTH,NO_GHOSTS,true,oneDLoops,twoDLoops);

   PLEGMA_Vector<double> *tmp[16] = {nullptr};
   PLEGMA_QLoops<double> *qLtmp = nullptr;
   if(oneDLoops) tmp[0] = new PLEGMA_Vector<double>(DEVICE);
   if(twoDLoops){
     for(int i = 1 ; i < 16; i++) tmp[i] = new PLEGMA_Vector<double>(DEVICE);
     qLtmp = new PLEGMA_QLoops<double>(DEVICE,FIRST_SIDE,true); // this we need to do the shifts where needed
   }

   double* ptr_vecs = p.deviceAlloc ? d_eigVecs : h_eigVecs;
   PLEGMA_Vector<double> phi;
   checkCudaError();





   
   qloops_std.dumpLoops(ft, outfilename + "_std", confID, format);
   qloops_gen.dumpLoops(ft, outfilename + "_gen", confID, format);
   
   delete ft[0];
   if(oneDLoops) delete tmp[0];
   if(twoDLoops){
     for(int i = 1 ; i < 16; i++) delete tmp[i];
     delete qLtmp;
     delete ft[1];
   }








  

  PLEGMA_Vector<double> vector_stoc;
  vector_stoc.randInit(rand_seed1);
  
  PLEGMA_Propagator<double> props[nmus];

  double *eigVecs = eigSol->getEigVecs();
  double spinVals[4*Eig_NeV*2];
  double *spinVals_d, *source_d, *evecs_d;
  size_t V4 = HGC_localVolume;
  size_t V3 = V4/HGC_localL[3];
  cudaMalloc((void**)&spinVals_d, 4*2*Eig_NeV*sizeof(double));
  cudaMalloc((void**)&source_d, 3*2*V3*sizeof(double));
  cudaMalloc((void**)&evecs_d, 3*2*Eig_NeV*V3*sizeof(double));
  checkCudaError();
  
  PLEGMA_printf("Start producing stochastic vectors and propagators\n");

  for(size_t its = 0; its < tSinks.size(); its++){
    int tsink = tSinks[its];
    PLEGMA_printf("\n ### Calculations for stochastic source %d - %02d begin now ###\n\n",
		  its, tsink);
      
    TIC();
    //We draw a different random vector for every source position
    vector_stoc.stochastic_Z(nroots);
    bool proj_done=false;
    
    for(int inev=0; inev < nevs.size(); inev++){ // Loop over NeV

      int nev = nevs[inev];
      
      char * src_string;
      asprintf(&src_string, "_nev%03d_id%03d_st%03d_rs%06d", nev, its, tsink, rand_seed1);
      std::string outfilename = twop_filename + src_string + ".h5";
      free(src_string);
      
      if(access( outfilename.c_str(), F_OK ) != -1) {
	PLEGMA_printf("File %s already exists. Skipping...", outfilename.c_str());
	continue;
      }
      
      PLEGMA_Correlator<double> corr(corr_space, site({0,0,0,tsink}), maxQsq);

      {
	//Dilution
	PLEGMA_Vector<double> vec;

	double aP[2]={1.,0.}, b[2]={0.,0.};
	size_t size_per_Vec = eigSol->getSize_per_Vec();

	// Projecting the source
	if(not proj_done) {
	  TIC();
	  int my_it = tsink - HGC_procPosition[3] * HGC_localL[3];
	  bool is_myIt = (my_it >= 0) && ( my_it < HGC_localL[3] );

	  if(not is_myIt) {
	    memset(spinVals, 0, 4*Eig_NeV*2*sizeof(double));
	  } else {
	    // Copy the non-zero part of the source
	    for(int c1 = 0 ; c1 < N_COLS ; c1++){
	      double *dst = source_d + c1*V3*2;
	      double *src = vector_stoc.D_elem() + c1*V4*2 + my_it*V3*2;
	      cudaMemcpy(dst, src, 2*V3*sizeof(double), cudaMemcpyDeviceToDevice);
	    }
	    checkCudaError();
	      
	    for (int spinindex=0; spinindex<4; ++spinindex){

	      // Copy the needed part of the evecs
	      for(int iv = 0 ; iv < Eig_NeV ; iv++){
		for(int c1 = 0 ; c1 < N_COLS ; c1++){
		  double *dst = evecs_d + iv*3*V3*2  + c1*V3*2;
		  double *src = eigVecs + iv*4*3*V4*2 + spinindex*3*V4*2 + c1*V4*2 + my_it*V3*2;
		  cudaMemcpy(dst, src, 2*V3*sizeof(double), cudaMemcpyDeviceToDevice);
		}
	      }
	      checkCudaError();
 
	      cuBLAS::gemv(DAGGER, 3*V3, Eig_NeV, aP, evecs_d, source_d, b, spinVals_d+2*Eig_NeV*spinindex);
	      checkCudaError();
	    }
	    cudaMemcpy(spinVals, spinVals_d, 4*Eig_NeV*2*sizeof(double), cudaMemcpyDeviceToHost);
	    checkCudaError();
	  }
	  MPI_Allreduce(MPI_IN_PLACE,spinVals,4*Eig_NeV*2,MPI_DOUBLE,MPI_SUM,HGC_fullComm);
	  proj_done=true;
	  TOC("projecting the source");
	}
	checkCudaError();

	std::complex<double> evals[nev], tmp[nev];
	for(int i=0; i<nev; i++) {
	  evals[i] = eigSol->getLittleD()[i*(Eig_NeV+1)];
	}

	
	// Building propagators
	TIC();
	for(int imu=0; imu<nmus; imu++){
	  mu = mus[imu];

	  for(int i=0; i<nev; i++) {
	    evals[i].imag(mu);
	  }

	  for (int spinindex=0; spinindex<4; ++spinindex){
	    std::complex<double> *vals = (std::complex<double> *) (spinVals+spinindex*nev*2);
	    for(int i=0; i<nev; i++) {
	      tmp[i] = vals[i]/evals[i];
	    }
	    cudaMemcpy(spinVals_d, tmp, 2*nev*sizeof(double), cudaMemcpyHostToDevice);
	    cuBLAS::gemv(NOTRANS, size_per_Vec, nev, aP, eigVecs, spinVals_d, b, vec.D_elem());
	    props[imu].absorb(vec, spinindex, 0);
	  }
	}
	TOC("building propagators");
      
      }

      TIC();
      for(int imu1=0; imu1<nmus; imu1++){
	char * mu_string;
	asprintf(&mu_string, "%+.4e_%+.4e_IR", mus[imu1], mus[imu1]);
	std::string dataset = mu_string;
	free(mu_string);
	
	TIME(corr.contractMesonsOpen(props[imu1], props[imu1], false));
	corr.setDatasets((std::vector<std::string>) {dataset});
	TIME(corr.writeHDF5( outfilename ));
	  
	for(int imu2=imu1+1; imu2<nmus; imu2++){
	  char * mu_string;
	  asprintf(&mu_string, "%+.4e_%+.4e_IR", mus[imu1], mus[imu2]);
	  std::string dataset = mu_string;
	  free(mu_string);
	  
	  TIME(corr.contractMesonsOpen(props[imu1], props[imu2], false));
	  corr.setDatasets((std::vector<std::string>) {dataset});
	  TIME(corr.writeHDF5( outfilename ));
	}
      }
      TOC("contraction and IO");
    }
    TOC("one stoch source");
  }
  cudaFree(spinVals_d);
  cudaFree(source_d);
  cudaFree(evecs_d);

  finalize();

  return 0;
}
