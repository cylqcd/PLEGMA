#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <PLEGMA_Hprobing.h>
#include <stdio.h>
using namespace plegma;
using namespace quda;

static std::vector<std::string> listOpt = {"verbosity", "load-gauge", "Eig-isACC", "Eig-PolyDeg", "Eig-amin",
					   "Eig-amax", "Eig-spectrumPart", "Eig-tol", "Eig-maxIters", "Eig-NeV"
#ifdef HAVE_ARPACK
					   ,"Eig-NkV", "Eig-logFile"
#elif HAVE_PRIMME
					   "Eig-printLevel", "Eig-method-PRIMME"
#endif
					   ,"nsrc", "maxQsq", "rng-seed", "corr-file-format"
};


static void dumpLoops(PLEGMA_FT<double> **ft,
 		      std::string filenamePrefix, std::string confID, FILE_FORMAT format){
  for(int idir=0; idir < 3; idir++)
    for(int i =0; i < HGC_totalL[0]; i++)
      ft[idir*HGC_totalL[0]+i]->writeFile(filenamePrefix + "_dir" + std::to_string(idir) + "_z" + std::to_string(i) + "_" + confID + ".dat" ,format);
}

int main(int argc, char **argv)
{
  initializeOptions(argc, argv, true, listOpt); // Put list of options later
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//
  int k_probing = 0;
  bool spinColorDil = false;
  bool lowModesRecon = false;
  bool asymProbing = false;
  HGC_options->set("asymmetric-probing", "Whether we want asymmetric probing",verbosity,asymProbing);
  int muAsymProb = 1;
  HGC_options->set("mu-asym-probing", "Multiplicative factor probing length",verbosity,muAsymProb);
  if(muAsymProb!=1 && !asymProbing)
    PLEGMA_error("The multiplicative factor is not 1 while the asymmetric probing will be not employed");
  if(!(ceil(log2(muAsymProb))==floor(log2(muAsymProb))))
    PLEGMA_error("The multiplicative factor of the probing length with asymmetric probing has to be a power of 2");
  
  HGC_options->set("k-probing", "Hierarchical probing, with distance D=2**k (Options:0,1,2,3,...) (0 means No probing)",verbosity,k_probing);
  int hadamLow=0;
  int Nhadam =  (k_probing>0) ? 2*std::pow(2,(N_DIMS-1)*(k_probing-1)+(k_probing-1+log2(muAsymProb))) : 1;
  PLEGMA_printf("Number of Hadamard vectors %d",Nhadam);
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

  int nsmearStoutWL = 30;
  double alphaStoutWL = 0.129;
  HGC_options->set("nsmear-stoutWL", "Number of stout smearing step for the Wilson line",verbosity,nsmearStoutWL);
  HGC_options->set("alpha-stoutWL", "Coefficient for the stout smearing for the Wilson line",verbosity,alphaStoutWL);

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
  int NdumpStep = 1;
  HGC_options->set("dump-step", "If accumulation is ON, Every how many stochastic vector to dump results", verbosity, NdumpStep);
  if(NdumpStep<1) PLEGMA_error("dump-step should be >= 1");
  bool debugMode = false;
  HGC_options->set("debug-mode", "If debug mode is enabled, run 1 source with units everywhere for check", verbosity, debugMode);

  if(Eig_NeV <= 0 && lowModesRecon){
    PLEGMA_warning("You enabled Low modes reconstruction but NeV is <= 0. Switching off Low modes reconstruction");
    lowModesRecon = false;
  }
  
  if(!lowModesRecon && Eig_NeV > 0){
    PLEGMA_warning("The Low modes reconstruction is off forcing number of eigenvalues to zero");
    Eig_NeV = 0;
  }

  double mu_h=0.;
  HGC_options->set("mu-heavy", "Twisted mass mu of the heavy quark. If different from zero the eigenvalues will be shifted accordingly", verbosity, mu_h);
  //=========================================================================================================//
  initializePLEGMA();

  
  // Reading from Lime file and loading to device
  PLEGMA_Gauge<double> gauge;
  gauge.readFile(latfile, LIME_FORMAT);
  gauge.load();
  gauge.calculatePlaq();

  // Loading to QUDA and computing plaquette also there
  initGaugeQuda(gauge, true);
  plaqQuda();

  // apply stout smearing
  PLEGMA_Gauge<double> gaugeStout;
  gaugeStout.stoutSmearing(gauge,nsmearStoutWL, alphaStoutWL, 3);
  PLEGMA_printf("Smeared Plaquette with stout 3D for Wilson Line:");
  gaugeStout.calculatePlaq();
  
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
#if defined(HAVE_ARPACK)
    eigParam.NkV = Eig_NkV;
    eigParam.logFile = Eig_logFile;
#elif defined(HAVE_PRIMME)
    eigParam.printLevel = Eig_printLevel;
    eigParam.primme_method=getMethod(Eig_method);
#else
    PLEGMA_error("No arpack or primme is compiled");
#endif
    eigSol = new EigSolver(eigParam, dslash_type , isReadEigenVecs, isWriteEigenVecs, fnameEigenVecsPrefix, true);
    eigSol->dumpEvalsVdagG5V(Eig_outputFile);
#else
    PLEGMA_error("No eigenSolver is compiled");
#endif
  }
  if(!(HGC_totalL[0] == HGC_totalL[1] && HGC_totalL[1] == HGC_totalL[2])) PLEGMA_error("Spatial total volume should be symmetric for this to work");

  int sizeFT=3*HGC_totalL[0]; 
  PLEGMA_FT<double> **ft_std = new PLEGMA_FT<double>*[sizeFT];
  PLEGMA_FT<double> **ft_gen = new PLEGMA_FT<double>*[sizeFT];
  for(int i = 0 ; i< sizeFT; i++){
    ft_std[i] = new PLEGMA_FT<double>(maxQsq,3,true);
    ft_gen[i] = new PLEGMA_FT<double>(maxQsq,3,true);
  }
  
  PLEGMA_QLoops<double> qloops_std(BOTH,NO_GHOSTS,true);
  PLEGMA_QLoops<double> qloops_gen(BOTH,NO_GHOSTS,true);
  
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
    
  PLEGMA_Vector<double> source(DEVICE);
  if(!debugMode) source.randInit(rng_seed);
  PLEGMA_Vector<double> *sourceDil = nullptr;
  if(k_probing>0 || spinColorDil) sourceDil = new PLEGMA_Vector<double>(DEVICE);

  // if LMR is enabled do the exact part
#if defined(HAVE_EIGENSOLVER)
  if(lowModesRecon)
    for(int i=0; i< eigSol->getEigVals().size(); i++){
      double eigVal = std::get<0>(eigSol->getEigVals()[i]);
      if(mu_h>0.) eigVal += 4.*inv_params.kappa*inv_params.kappa*(mu_h*mu_h-mu*mu);
      long int iorder = std::get<3>(eigSol->getEigVals()[i]);
      double *eigVec = eigSol->getEigVecs() + iorder*eigSol->getSize_per_Vec()*2;
      cudaMemcpy(phi.D_elem(), eigVec, eigSol->getBytes_per_Vec(), cudaMemcpyHostToDevice);
      checkCudaError();      
      phi_r.copy(phi);
      qloops_std.oneEnd_trick_wilsonLine(phi,phi_r,-1./eigVal,gaugeStout,ft_std);
      D->apply<M>(phi_r,phi);
      phi_r.apply_gamma5();
      qloops_gen.oneEnd_trick_wilsonLine(phi,phi_r,+1./eigVal,gaugeStout,ft_gen);
    }
#endif

  std::size_t foundPos = latfile.find("conf.");
  if(foundPos == std::string::npos) PLEGMA_error("Cannot find (conf.) in configuration path to get confID");
  std::string confID = latfile.substr(foundPos+5,latfile.length());

#if defined(HAVE_EIGENSOLVER)
  if(lowModesRecon){
    dumpLoops(ft_std, loopsPrefix + "/exact_part_std_", confID, corr_file_format);
    dumpLoops(ft_gen, loopsPrefix + "/exact_part_gen_", confID, corr_file_format);

    for(int i = 0 ; i < sizeFT; i++){
      ft_std[i]->zero();
      ft_gen[i]->zero();
    }
  }
#endif

  
  if(std::abs(mu_h)>0) {
    if(mu_h>0) mu_h*=-1.;	
    mu = mu_h;
    PLEGMA_printf("Updating the solver with mu value %f\n",mu);
    solverDN->UpdateSolver();
  }

  
  PLEGMA_Hprobing *hprop = nullptr;
  if(k_probing>0) hprop = new PLEGMA_Hprobing(k_probing);

  std::vector<int> indDof = {0,1,2,3,4,5,6,7,8,9,10,11};
  for(int isrc = 0; isrc < numSourcePositions; isrc++){ // numSourcePosition is actually stochastic source position but anyway
    if(debugMode) source.setUnit(indDof);
    else source.stochastic_Z(4); // hardcoded 4 roots of one
    for(int wilsDir= asymProbing? 0 : -1; wilsDir<asymProbing?3 : 0; wilsDir++){
      if(asymProbing && k_probing>0){
	delete hprop;
	hprop = new PLEGMA_Hprobing(k_probing,wilsDir,muAsymProb);
      }
    for(int ih = hadamLow; ih < hadamHgh; ih++){
	for(int isc = 0; isc < Nsc; isc++){
	  if(spinColorDil){ sourceDil->dilutespincolor(source,isc/N_COLS,isc%N_COLS);}
	  if(spinColorDil && k_probing>0){ sourceDil->applyHpropColoring4D(*sourceDil,*hprop,ih,indDof);}
	  else if(!spinColorDil && k_probing>0){ sourceDil->applyHpropColoring4D(source,*hprop,ih,indDof);}
	  if(spinColorDil || k_probing>0) solverDN->solve(phi,*sourceDil); else solverDN->solve(phi,source);
	  // for convention reasons for quark loops we put the normalization factors of the fields later in the analysis
	  phi.scaleVector(1./(2.*inv_params.kappa));
	  double t1=MPI_Wtime();
#if defined(HAVE_EIGENSOLVER)
	  if(lowModesRecon)
	    eigSol->projectVector(phi); // In place application of deflation projector operator on solution vector
#endif
	  phi_r.copy(phi);
	  qloops_std.oneEnd_trick_wilsonLine(phi,phi_r,-1,gaugeStout,ft_std,wilsDir);
	  
	  D->apply<M>(phi_r,phi);
	  phi_r.apply_gamma5();
	  qloops_gen.oneEnd_trick_wilsonLine(phi,phi_r,+1,gaugeStout,ft_gen,wilsDir);
	  double t2=MPI_Wtime();
	  PLEGMA_printf("Contraction time is %f\n",t2-t1);
	} // for loop isc
      } // for loop ih
    } // for loop wilsDir 
    double t1=MPI_Wtime();
    if((isrc+1)%NdumpStep == 0){
      dumpLoops(ft_std, loopsPrefix + "/stoch_part_Src" + std::to_string(isrc) + "_std_", confID, corr_file_format);
      dumpLoops(ft_gen, loopsPrefix + "/stoch_part_Src" + std::to_string(isrc) + "_gen_", confID, corr_file_format);
    }
    double t2=MPI_Wtime();
    PLEGMA_printf("FT and dump data time is %f\n",t2-t1);
    
  } // for loop isrc

  if(k_probing>0) delete hprop;
  if(k_probing>0 || spinColorDil) delete sourceDil;
#if defined(HAVE_EIGENSOLVER)
  if(lowModesRecon) delete eigSol;
#endif

  for(int i = 0 ; i< sizeFT; i++){
    delete ft_std[i];
    delete ft_gen[i];
  }
  delete[] ft_std;
  delete[] ft_gen;
    
  delete D;
  delete solverDN;
  finalize();

  return 0;
}
