#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <PLEGMA_Hprobing.h>
#include <stdio.h>
using namespace plegma;
using namespace quda;

#ifdef HAVE_PRIMME
int Nmethods = 16;
static std::string methods[] = { "PRIMME_DEFAULT_METHOD","PRIMME_DYNAMIC","PRIMME_DEFAULT_MIN_TIME",
			  "PRIMME_DEFAULT_MIN_MATVECS", "PRIMME_Arnoldi", "PRIMME_GD",
			 "PRIMME_GD_plusK", "PRIMME_GD_Olsen_plusK", "PRIMME_JD_Olsen_plusK", "PRIMME_RQI",
			 "PRIMME_JDQR", "PRIMME_JDQMR", "PRIMME_JDQMR_ETol", "PRIMME_STEEPEST_DESCENT",
			  "PRIMME_LOBPCG_OrthoBasis", "PRIMME_LOBPCG_OrthoBasis_Window"};
static primme_preset_method getMethod(std::string str){
  for(int i = 0; i < Nmethods; i++)
    if(str == methods[i])
      return static_cast<primme_preset_method>(i);
  PLEGMA_warning("Method provided %s is not in PRIMME, available methods are",str.c_str());
  for(int i = 0; i < Nmethods; i++)
    PLEGMA_printf(methods[i].c_str());
  PLEGMA_exit(-1);
  return static_cast<primme_preset_method>(0);
}
#endif


static std::vector<std::string> listOpt = {"verbosity", "load-gauge", "Eig-isACC", "Eig-PolyDeg", "Eig-amin",
					   "Eig-amax", "Eig-spectrumPart", "Eig-tol", "Eig-maxIters", "Eig-NeV",
#if defined(HAVE_ARPACK) || defined(QUDAEIG)
					   "Eig-NkV", "Eig-logFile",
#elif defined(HAVE_PRIMME)
					   "Eig-printLevel", "Eig-method-PRIMME",
#endif
					   "nsrc", "maxQsq", "rng-seed", "corr-file-format"
};


static void dumpLoops(PLEGMA_QLoops<double> &qLoops, PLEGMA_FT<double> *ft[2],
		      std::string filenamePrefix, std::string confID, FILE_FORMAT format, int isc=-1){
  using sv=std::vector<std::string>;
  std::string fname_base;
  if(format != ASCII_FORMAT && format != HDF5_FORMAT) PLEGMA_error("This executable can write only in ascii and hdf5 format");
  std::string suffix = (format == ASCII_FORMAT)? ".dat": ".h5";
  if(isc >= 0)
    fname_base = join(sv({"Conf"+confID,"Ns"+std::to_string(isc)}),"/");
  else
    fname_base = join(sv({"Conf"+confID}),"/");
  

  qLoops.load(qLoops.H_loc());
  ft[0]->apply(qLoops,FT_GEMV);
  std::string fnameUl = fname_base + join(sv({"localLoops","loop"}),"/");
  ft[0]->writeFile( (format == HDF5_FORMAT)? filenamePrefix+suffix+fnameUl: filenamePrefix+findAndReplace(fnameUl,'/','_')+suffix, format);


  if(qLoops.IsOneD())
    for(int mu = 0 ; mu < N_DIMS ; mu++){
      std::string fnameOneD = fname_base + join(sv({"oneD","dir"+std::to_string(mu),"loop"}),"/");
      std::string fnameOneDC = fname_base + join(sv({"oneDC","dir"+std::to_string(mu),"loop"}),"/");
      qLoops.load(qLoops.H_oneD()[mu]);
      ft[0]->apply(qLoops,FT_GEMV);
      ft[0]->scale(0.25); // put the 1/4 of the symmetric covariant derivative
      ft[0]->writeFile((format == HDF5_FORMAT)? filenamePrefix+suffix+fnameOneD: filenamePrefix+findAndReplace(fnameOneD,'/','_')+suffix, format);

      qLoops.load(qLoops.H_oneDC()[mu]);
      ft[0]->apply(qLoops,FT_GEMV);
      ft[0]->scale(0.25);
      ft[0]->writeFile((format == HDF5_FORMAT)? filenamePrefix+suffix+fnameOneDC: filenamePrefix+findAndReplace(fnameOneDC,'/','_')+suffix, format);      
    }

  int count=0;
  if(qLoops.IsTwoD()){
    for(auto munu : qLoops.get_twoD_index()){
      int mu=std::get<0>(munu), nu=std::get<1>(munu);
      std::string fnameTwoD = fname_base + join(sv({"twoD","dirs"+std::to_string(mu)+std::to_string(nu),"loop"}),"/");
      qLoops.load(qLoops.H_twoD()[count]);
      ft[1]->apply(qLoops,FT_GEMV);
      if(mu != 3 && nu != 3) ft[1]->scale(0.25);
      else ft[1]->scale(0.125);
      ft[1]->writeFile((format == HDF5_FORMAT)? filenamePrefix+suffix+fnameTwoD: filenamePrefix+findAndReplace(fnameTwoD,'/','_')+suffix, format);
      count++;
    }
  }  
}

int main(int argc, char **argv)
{
  initializeOptions(argc, argv, true, listOpt); // Put list of options later
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
  int k_probing = 0;
  bool spinColorDil = false;
  bool lowModesRecon = false;
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
  HGC_options->set("low-modes-recon", "Whether we want to use low modes of the operator to reconstruct part of the quark loop",verbosity,lowModesRecon);
  if(!lowModesRecon) Eig_NeV=0;
  std::string Eig_outputFile = "./eigsVdagG5V.dat";
  HGC_options->set("Eig-outputFile", "Path to dump the eigenvalues and vdag g5 v if low-modes-recon is enabled",verbosity, Eig_outputFile);
  bool isReadEigenVecs = false, isWriteEigenVecs = false;
  std::string fnameEigenVecsPrefix="";
  HGC_options->set("readEigenVectors", "Where we want to read EigenVectors from file", verbosity, isReadEigenVecs);
  HGC_options->set("writeEigenVectors", "Where we want to read EigenVectors from file", verbosity, isWriteEigenVecs);
  HGC_options->set("prefixEigenVecsFile", "Path with prefix for the filenames of the eigenvectors", verbosity, fnameEigenVecsPrefix);
#ifdef QUDAEIG
  int batched_rotate = 1;
  HGC_options->set("batched-rotate", "The size of the batch during Ritz rotation", verbosity, batched_rotate);
#endif

  bool oneDLoops = true;
  bool accumFlag = true;
  bool twoDLoops = false;
  int NdumpStep = 1;
  HGC_options->set("oneD-loops", "Whether we want to use covariant derivative for the quark loops calculation", verbosity, oneDLoops);
  HGC_options->set("twoD-loops", "Whether we want to use two covariant derivative for the quark loops calculation", verbosity, twoDLoops);
  HGC_options->set("accum-loops", "Accumulate loops over the stochastic source vectors", verbosity, accumFlag);
  HGC_options->set("dump-step", "If accumulation is ON, Every how many stochastic vector to dump results", verbosity, NdumpStep);
  bool debugMode = false;
  HGC_options->set("debug-mode", "If debug mode is enabled, run 1 source with units everywhere for check", verbosity, debugMode);

  //=========================================================================================================//
  initializePLEGMA();

  if(!accumFlag) NdumpStep =1;
  if(accumFlag && (NdumpStep<1)) PLEGMA_error("dump-step should be >= 1");

  if(Eig_NeV <= 0 && lowModesRecon){
    PLEGMA_warning("You enabled Low modes reconstruction but NeV is <= 0. Switching off Low modes reconstruction");
    lowModesRecon = false;
  }
  
  if(!lowModesRecon && Eig_NeV > 0){
    PLEGMA_warning("The Low modes reconstruction is off forcing number of eigenvalues to zero");
    Eig_NeV = 0;
  }

  
  // Reading from Lime file and loading to device
  PLEGMA_Gauge<double> gauge;
  gauge.readFile(latfile, LIME_FORMAT);
  gauge.calculatePlaq();

  // Loading to QUDA and computing plaquette also there
  initGaugeQuda(gauge, true);
  plaqQuda();

  // apply boundary conditions since is needed for the covariant derivative
  // this needs to be done after initGaugeQuda otherwise causes troubles
  applyBoundaryConditions(gauge,true);

  // In case we need LMR we need to compute the eigenvectors
#if defined(HAVE_EIGENSOLVER)
  EigSolver *eigSol = nullptr;  
#endif
  if(lowModesRecon){
#if defined(HAVE_EIGENSOLVER)
    EigSolverParams eigParam;
    eigParam.NeV = Eig_NeV;
    eigParam.isACC = Eig_isACC;
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
#elif defined(HAVE_PRIMME)
    eigParam.printLevel = Eig_printLevel;
    eigParam.primme_method=getMethod(Eig_method);
#else
    PLEGMA_error("No arpack or primme is compiled");
#endif
    eigSol = new EigSolver(eigParam, dslash_type, isReadEigenVecs, isWriteEigenVecs, fnameEigenVecsPrefix, true);
    eigSol->dumpEvalsVdagG5V(Eig_outputFile);
#else
    PLEGMA_error("No eigenSolver is compiled");
#endif
  }

  PLEGMA_FT<double> *ft[2]={nullptr,nullptr};
  ft[0] = new PLEGMA_FT<double>(maxQsq, 3);
  if(twoDLoops) ft[1] = new PLEGMA_FT<double>(0, 3);
  
  PLEGMA_QLoops<double> qloops_std(BOTH,NO_GHOSTS,true,oneDLoops,twoDLoops);
  PLEGMA_QLoops<double> qloops_gen(BOTH,NO_GHOSTS,true,oneDLoops,twoDLoops);
  
  // // ensuring mu negative
  if(mu>0) mu*=-1.;
  QUDA_solver *solverDN = new QUDA_solver(mu);
  QudaInvertParam inv_params = solverDN->getInvParams();
  
  QUDA_dirac *D = nullptr;
  if(inv_params.dslash_type == QUDA_TWISTED_CLOVER_DSLASH)
    D = new QUDA_dirac(QUDA_CLOVER_WILSON_DSLASH);
  else if (inv_params.dslash_type == QUDA_TWISTED_MASS_DSLASH)
    D = new QUDA_dirac(QUDA_WILSON_DSLASH);
  else
    PLEGMA_error("Only QUDA_TWISTED_CLOVER_DSLASH and QUDA_TWISTED_MASS_DSLASH are allowed for the one-end trick");

  PLEGMA_Vector<double> phi;
  PLEGMA_Vector<double> phi_r;
  PLEGMA_Vector<double> *tmp[16] = {nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
				    nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr};
  PLEGMA_QLoops<double> *qLtmp = nullptr;
  if(oneDLoops) tmp[0] = new PLEGMA_Vector<double>(DEVICE);
  if(twoDLoops){
    for(int i = 1 ; i < 16; i++) tmp[i] = new PLEGMA_Vector<double>(DEVICE);
    qLtmp = new PLEGMA_QLoops<double>(DEVICE,FIRST_SIDE,true); // this we need to do the shifts where needed
  }
    
  PLEGMA_Vector<double> source(DEVICE);
  if(!debugMode) source.randInit(rng_seed);
  PLEGMA_Vector<double> *sourceDil = nullptr;
  if(k_probing>0 || spinColorDil) sourceDil = new PLEGMA_Vector<double>(DEVICE);
    
  if(oneDLoops) gauge.communicateGhost();

  // if LMR is enabled do the exact part
#if defined(HAVE_EIGENSOLVER)
  if(lowModesRecon)
    for(int i=0; i< eigSol->getEigVals().size(); i++){
      double eigVal = std::get<0>(eigSol->getEigVals()[i]);
      long int iorder = std::get<3>(eigSol->getEigVals()[i]);
      double *eigVec = eigSol->getEigVecs() + iorder*eigSol->getSize_per_Vec()*2;
      cudaMemcpy(phi.D_elem(), eigVec, eigSol->getBytes_per_Vec(), cudaMemcpyHostToDevice);
      checkQudaError();
      if(oneDLoops || twoDLoops) qloops_std.oneEnd_trick(phi,phi,tmp,qLtmp,gauge,-1./eigVal,true); //standard one-end trick
      else qloops_std.oneEnd_trick(phi,phi,-1./eigVal,true); //standard one-end trick

      D->apply<M>(phi_r,phi);
      phi_r.apply_gamma5();
      if(oneDLoops || twoDLoops) qloops_gen.oneEnd_trick(phi, phi_r, tmp,qLtmp, gauge, +1./eigVal, true); //generalized one-end trick
      else qloops_gen.oneEnd_trick(phi, phi_r, +1./eigVal, true); //generalized one-end trick
    }
#endif

  std::size_t foundPos = latfile.find("conf.");
  std::size_t foundPos2 = latfile.find("conf_lgfix.");
  if(foundPos == std::string::npos && foundPos2 == std::string::npos)
    PLEGMA_error("Cannot find (conf.) or (conf_lgfix.) in configuration path to get confID");
  std::string confID;
  if(foundPos != std::string::npos)
    confID = latfile.substr(foundPos+5,latfile.length());
  else if(foundPos2 != std::string::npos)
    confID = latfile.substr(foundPos2+5,latfile.length());
  else
    PLEGMA_error("Cannot happen to reach this error");

#if defined(HAVE_EIGENSOLVER)
  if(lowModesRecon){
    dumpLoops(qloops_std, ft, loopsPrefix + "/exact_part_std", confID, corr_file_format);
    dumpLoops(qloops_gen, ft, loopsPrefix + "/exact_part_gen", confID, corr_file_format);
  }
#endif

  PLEGMA_Hprobing *hprop = nullptr;
  if(k_probing>0) hprop = new PLEGMA_Hprobing(k_probing);

  qloops_std.clearAccumBuffs();
  qloops_gen.clearAccumBuffs();
  std::vector<int> indDof = {0,1,2,3,4,5,6,7,8,9,10,11};
  for(int isrc = 0; isrc < numSourcePositions; isrc++){ // numSourcePosition is actually stochastic source position but anyway
    if(debugMode) source.setUnit(indDof);
    else source.stochastic_Z(4); // hardcoded 4 roots of one
    for(int ih = hadamLow; ih < hadamHgh; ih++){
	for(int isc = 0; isc < Nsc; isc++){
	  if(spinColorDil){ sourceDil->dilutespincolor(source,isc/N_COLS,isc%N_COLS);}
	  if(spinColorDil && k_probing>0){ sourceDil->applyHpropColoring4D(*sourceDil,*hprop,ih,indDof);}
	  else if(!spinColorDil && k_probing>0){ sourceDil->applyHpropColoring4D(source,*hprop,ih,indDof);}
	  if(spinColorDil || k_probing>0) solverDN->solve(phi,*sourceDil); else solverDN->solve(phi,source);
	  // for convention reasons for quark loops we put the normalization factors of the fields later in the analysis
	  phi.scale(1./(2.*inv_params.kappa));
	  double t1=MPI_Wtime();
	  // !!!!!!!!!!!!!!!!!!!! if we use deflation here I think we need to project solution vector but check
#if defined(HAVE_EIGENSOLVER)
	  if(lowModesRecon)
	    eigSol->projectVector(phi); // In place application of deflation projector operator on solution vector
#endif
	  if(oneDLoops || twoDLoops) qloops_std.oneEnd_trick(phi,phi,tmp,qLtmp,gauge,-1.,true); //standard one-end trick
	  else qloops_std.oneEnd_trick(phi,phi,-1.,true); //standard one-end trick

	  D->apply<M>(phi_r,phi);
	  phi_r.apply_gamma5();
	  if(oneDLoops || twoDLoops) qloops_gen.oneEnd_trick(phi, phi_r, tmp,qLtmp, gauge, +1., true); //generalized one-end trick
	  else qloops_gen.oneEnd_trick(phi, phi_r, +1., true); //generalized one-end trick
	  double t2=MPI_Wtime();
	  PLEGMA_printf("Contraction time is %f\n",t2-t1);
	} // for loop isc
      } // for loop ih

    double t1=MPI_Wtime();
    if((isrc+1)%NdumpStep == 0){
      dumpLoops(qloops_std, ft, loopsPrefix + "/stoch_part_std", confID, corr_file_format,isrc);
      dumpLoops(qloops_gen, ft, loopsPrefix + "/stoch_part_gen", confID, corr_file_format,isrc);
    }
    double t2=MPI_Wtime();
    PLEGMA_printf("FT and dump data time is %f\n",t2-t1);
    
    if(!accumFlag){ // In case we do not accumulate we clear the buffers
      qloops_std.clearAccumBuffs();
      qloops_gen.clearAccumBuffs();
    }
  } // for loop isrc

  if(k_probing>0) delete hprop;
  if(k_probing>0 || spinColorDil) delete sourceDil;
#if defined(HAVE_EIGENSOLVER)
  if(lowModesRecon) delete eigSol;
#endif

  delete ft[0];
  if(oneDLoops) delete tmp[0];
  if(twoDLoops){
    for(int i = 1 ; i < 16; i++) delete tmp[i];
    delete qLtmp;
    delete ft[1];
  }
  
  delete D;
  delete solverDN;
  finalize();

  return 0;
}
