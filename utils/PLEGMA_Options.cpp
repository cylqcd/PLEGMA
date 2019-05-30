#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <comm_quda.h>
#include <functional>

const std::vector<std::string> listAvailOptPLEGMA = {"verbosity", "load-gauge", "nsmear-APE", "alpha-APE", "nsmear-gauss", "alpha-gauss",
						     "nsmear-stout", "alpha-stout", "nsrc", "src-filename", "maxQsq", "twop-filename",
						     "threep-filename",  "corr-file-format", "corr-space", "tSinks","Projs", "Eig-NeV"
#ifdef HAVE_ARPACK
						     ,"Eig-NkV", "Eig-logFile"
#elif HAVE_PRIMME
						     ,"Eig-printLevel", "Eig-method-PRIMME"
#endif
						     ,"Eig-isACC", "Eig-PolyDeg", "Eig-amin", "Eig-amax", "Eig-spectrumPart", "Eig-tol", "Eig-maxIters"
						     , "rng-seed"
};

static inline bool isInList(std::vector<std::string> list,std::string str){
  if(std::find(list.begin(),list.end(),str) != list.end() ) return true;
  else return false;
}


void plegmaOptions(Options &opt, std::vector<std::string> list){
  bool isFound;
  std::string tmpString;
  if(isInList(list,"verbosity")) opt.set("verbosity","Set verbosity level, 0 minimal, 1 verbose, 2 debug, 3 debug all", 0, verbosity);
  
  if(verbosity==1) {
    PLEGMA_printf("\nParameters read by plegmaOptions:\n");
    PLEGMA_printf( "procs %d %d %d %d\n",  procs[0], procs[1], procs[2], procs[3]);
  } else if(verbosity>1) {
    PLEGMA_printf("\nAll parameters available in plegmaOptions with read or default value:\n");
    PLEGMA_printf( "verbosity %d\n", verbosity);
    PLEGMA_printf( "procs %d %d %d %d\n",  procs[0], procs[1], procs[2], procs[3]);
  }

  opt.setForced("dims","Set local dimensions (X Y Z T), e.g. 8 8 8 16", verbosity, dims[0], dims[1], dims[2], dims[3]);
  if(!opt.getIsHelp()) for(int i=0; i<4; i++) if( (dims[i] <= 0 || dims[i] > 512) ) PLEGMA_error("Error with dim %d: dims should be > 0 and < 512\n", i);

  if(isInList(list,"load-gauge")) opt.set("load-gauge", "Path to the gauge field", verbosity, latfile);

  // smearing ----------------------------------------------------------------------------------------------
  if(isInList(list,"nsmear-APE")) opt.set("nsmear-APE", "Number of APE smearing step", verbosity, nsmearAPE);
  if(isInList(list,"alpha-APE")) opt.set("alpha-APE", "Coefficient for the APE smearing", verbosity, alphaAPE);
  if(isInList(list,"nsmear-gauss")) opt.set("nsmear-gauss", "Number of Gaussian smearing step", verbosity, nsmearGauss);
  if(isInList(list,"alpha-gauss")) opt.set("alpha-gauss", "Coefficient for the Gaussian smearing", verbosity, alphaGauss);
  if(isInList(list,"nsmear-stout")) opt.set("nsmear-stout", "Number of stout smearing step", verbosity, nsmearStout);
  if(isInList(list,"alpha-stout")) opt.set("alpha-stout", "Coefficient for the stout smearing", verbosity, alphaStout);
  // sources----------------------------------------------------------------------------------------------
  if(isInList(list,"nsrc")) opt.set("nsrc", "Number of source positions or stochastic vectors", verbosity, numSourcePositions);
  if(isInList(list,"src-filename")){
    isFound = opt.set("src-filename", "Filename of source positions", verbosity, pathListSourcePositions);
    if(isFound) readSourceList();
  }
  if(isInList(list,"rng-seed")) opt.set("rng-seed", "A seed for the random number generator", verbosity, rng_seed);
  // Correlators ------------------------------------------------------------------------------------------
  if(isInList(list,"maxQsq")) opt.set("maxQsq", "Maximum Qsq for the Fourier Transform", verbosity, maxQsq);
  if(isInList(list,"twop-filename")) opt.set("twop-filename", "File name for two-point functions, extension will be added", verbosity, twop_filename);
  if(isInList(list,"threep-filename")) opt.set("threep-filename", "File name for three-point functions, extension will be added", verbosity, threep_filename);

  if(isInList(list,"corr-file-format")){
    tmpString = get_file_format_str(corr_file_format);
    isFound=opt.set("corr-file-format", "Correlators file format to use in data writing, options (ascii, hdf5, lime)", verbosity, tmpString);
    if(isFound) corr_file_format = get_file_format(tmpString.c_str());
  }

  if(isInList(list,"corr-space")){
    tmpString = get_space_str(corr_space);
    isFound=opt.set("corr-space", "Correlators space format to use in data writing, options (momentum, position)", verbosity, tmpString);
    if(isFound) corr_space = get_space(tmpString.c_str());
  }

  if(isInList(list, "tSinks")) opt.set("tSinks", "List with the source-sink time separations to do", verbosity, tSinks);
  if(isInList(list, "Projs")) opt.set("Projs", "List of the projectors to be used", verbosity, Projs);
  // Eigensolver ------------------------------------------------------------------------------------------
  if(isInList(list, "Eig-NeV")) opt.set("Eig-NeV", "Number of eigenpairs to compute", verbosity, Eig_NeV);
#ifdef HAVE_ARPACK
  if(isInList(list, "Eig-NkV")) opt.set("Eig-NkV", "Number of vectors for the Krylov subspace", verbosity, Eig_NkV);
  if(isInList(list, "Eig-logFile")) opt.set("Eig-logFile", "Path for the logfile of the eigensolver", verbosity, Eig_logFile);
#elif HAVE_PRIMME
  if(isInList(list, "Eig-printLevel")) opt.set("Eig-printLevel", "Print Level for the PRIMEE eigenSolver", verbosity, Eig_printLevel);
  if(isInList(list, "Eig-method-PRIMME")) opt.set("Eig-method-PRIMME", "The method for eigensolver from PRIMME see manual for all", verbosity, Eig_method);
#endif
  if(isInList(list, "Eig-isACC")) opt.set("Eig-isACC", "If we want to use Polynomial acceleration", verbosity, Eig_isACC);
  if(isInList(list, "Eig-PolyDeg")) opt.set("Eig-PolyDeg", "The degree of the Polynomial", verbosity, Eig_PolyDeg);
  if(isInList(list, "Eig-amin")) opt.set("Eig-amin", "The low bound of the Polynomial called amin", verbosity, Eig_amin);
  if(isInList(list, "Eig-amax")) opt.set("Eig-amax", "The upper bound of the Polynomial called amax", verbosity, Eig_amax);
  if(isInList(list, "Eig-spectrumPart")) opt.set("Eig-spectrumPart", "Which part of the spectrum you want to compute. Options (SR, LR)", verbosity, Eig_spectrumPart);
  if(isInList(list, "Eig-tol")) opt.set("Eig-tol", "Tolerance for the eigensolver. At least all eigenpairs converge to this tol", verbosity, Eig_tol);
  if(isInList(list, "Eig-maxIters")) opt.set("Eig-maxIters", "Tolerance for the eigensolver. At least all eigenpairs converge to this tol", verbosity, Eig_maxIters);

}

template<typename T> static inline void default_map_MG(std::map<int,T> &tpl, T def){
  tpl.clear();
  for(int i=0; i<QUDA_MAX_MG_LEVEL; i++) tpl.insert(std::make_pair(i,def));
}

template<typename T> static inline void map_to_array_MG(std::map<int,T> &tpl, T *arr, T noSm, T noBig, std::string err){
  typename std::map<int,T>::iterator it = tpl.begin();
  while(it != tpl.end()){
    int lvl=it->first;
    T val = it->second;
    if(lvl < 0 || lvl >= QUDA_MAX_MG_LEVEL) PLEGMA_error("ERROR: invalid multigrid level %d", lvl);
    if(val < noSm || val > noBig) PLEGMA_error(err.c_str());
    arr[lvl]=val;
    it++;
  }
}

template<typename T> static inline void map_to_array_MG(std::map<int,std::string> &tpl, T *arr, std::function<T(const char*)> func){
  typename std::map<int,std::string>::iterator it = tpl.begin();
  while(it != tpl.end()){
    int lvl=it->first;
    T val = func((it->second).c_str());
    if(lvl < 0 || lvl >= QUDA_MAX_MG_LEVEL) PLEGMA_error("ERROR: invalid multigrid level %d", lvl);
    arr[lvl]=val;
    it++;
  }
}

void qudaOptions(Options &opt){
  std::string tmpString;
  bool tmpBool;
  bool isFound;

  if(verbosity==1) {
    PLEGMA_printf("\nParameters read by qudaOptions:\n");
  } else if(verbosity>1) {
    PLEGMA_printf("\nAll parameters available in qudaOptions with read or default value:\n");
  }

  tmpString = get_prec_str(prec);
  isFound=opt.set("Q-prec", "Precision in the GPU, options (double,single,half)", verbosity, tmpString);
  if(isFound)prec = get_prec(tmpString.c_str());

  tmpString = get_prec_str(prec);
  isFound=opt.set("Q-prec-sloppy", "Sloppy precision in the GPU, options (double,single,half)", verbosity, tmpString);
  if(isFound)prec_sloppy = get_prec(tmpString.c_str());
  if (prec_sloppy == QUDA_INVALID_PRECISION) prec_sloppy = prec;
  
  tmpString = get_prec_str(prec);
  isFound=opt.set("Q-prec-precondition", "Preconditioner precision in the GPU, options (double,single,half)", verbosity, tmpString);
  if(isFound)prec_precondition = get_prec(tmpString.c_str());
  if (prec_precondition == QUDA_INVALID_PRECISION) prec_precondition = prec_sloppy;

  tmpString = get_prec_str(prec);
  isFound=opt.set("Q-prec-null", "NUll-vector precision in the GPU, options (double,single,half)", verbosity, tmpString);
  if(isFound)prec_null = get_prec(tmpString.c_str());
  if (prec_null == QUDA_INVALID_PRECISION) prec_null = prec_precondition;

  tmpString = get_recon_str(link_recon);
  isFound=opt.set("Q-recon", "Type of link reconstruction, options (8,9,12,13,18)", verbosity, tmpString);
  if(isFound)link_recon  = get_recon(tmpString.c_str());

  tmpString = get_recon_str(link_recon);
  isFound=opt.set("Q-recon-sloppy", "Type of link reconstruction for sloppy, options (8,9,12,13,18)", verbosity, tmpString);
  if(isFound)link_recon_sloppy  = get_recon(tmpString.c_str());
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) link_recon_sloppy = link_recon;

  tmpString = get_recon_str(link_recon);
  isFound=opt.set("Q-recon-precondition", "Type of link reconstruction for precon, options (8,9,12,13,18)", verbosity, tmpString);
  if(isFound)link_recon_precondition  = get_recon(tmpString.c_str());
  if (link_recon_precondition == QUDA_RECONSTRUCT_INVALID) link_recon_precondition = link_recon_sloppy;

  tmpString = get_dslash_str(dslash_type);
  isFound=opt.set("Q-dslash-type", "Set the dslash type, options for now (twisted-mass/twisted-clover)",verbosity, tmpString);
  if(isFound) {
    if((tmpString != "twisted-mass") && (tmpString != "twisted-clover"))PLEGMA_error("Error: only twisted-mass or twisted-clover are allowed for now");
    dslash_type =  get_dslash_type(tmpString.c_str());
  }

  tmpBool = (dagger == QUDA_DAG_YES);
  isFound=opt.set("Q-dagger", "In case you want the dagger operator", verbosity, tmpBool);
  if(isFound && tmpBool) dagger = QUDA_DAG_YES;

  tmpBool = kernel_pack_t;
  isFound=opt.set("Q-kernel-pack-t", "Kernel packing in the T direction", verbosity, tmpBool);
  if(isFound && tmpBool) kernel_pack_t = true;

  tmpString = get_flavor_str(twist_flavor);
  isFound=opt.set("Q-flavor", "Twisted mass type of flavor, options for now (singlet)", verbosity,tmpString);
  if(isFound) twist_flavor = get_flavor_type(tmpString.c_str());

  opt.set("Q-niter", "Maximum number of iterations for the solvers, options (1,...,1e6)",verbosity,niter);
  if(niter<1 || niter>1e6) PLEGMA_error("Error: Invalid number [%d] for max number of iterations\n",niter);

  opt.set("Q-ngcrkrylov", "The number of inner iterations to use for GCR or BiCGstab-l, options (1,...,1e6)",verbosity,gcrNkrylov);
  if(gcrNkrylov<1 || gcrNkrylov>1e6) PLEGMA_error("Error: Invalid number [%d] for gcrNkrylov iterations\n",gcrNkrylov);

  opt.set("Q-pipeline", "The pipeline length for fused operations in GCR or BiCGstab-l, options(0,...,8)", verbosity, pipeline);
  if(pipeline < 0 || pipeline > 8) PLEGMA_error("Error: Invalid number [%d] for pipeline length\n",pipeline);

  opt.set("Q-solution-pipeline", "The pipeline length for fused solution accumulation, options (0,..,16)", verbosity, solution_accumulator_pipeline);
  if (solution_accumulator_pipeline < 0 || solution_accumulator_pipeline > 16)  PLEGMA_error("Error: Invalid number [%d] for solution pipeline length\n",solution_accumulator_pipeline);

  tmpString = get_solver_str(inv_type);
  isFound=opt.set("Q-inv-type", "The type of solver to use, options (cg,bicgstab,gcr)", verbosity, tmpString);
  if(isFound) inv_type = get_solver_type(tmpString.c_str());

  tmpString = get_solver_str(precon_type);
  isFound=opt.set("Q-precon-type", "The type of precon solver to use, options (mr,none)", verbosity, tmpString);
  if(isFound)precon_type = get_solver_type(tmpString.c_str());

  opt.set("Q-kappa", "Kappa value of the Dirac operator", verbosity, kappa);
  opt.set("Q-mu", "Twisted mass value", verbosity, mu);
  opt.set("Q-csw", "The coefficient of the clover term", verbosity, csw);

  tmpString = get_verbosity_str(verbosity_level);
  opt.set("Q-verbosity", "The verbosity of QUDA", verbosity, tmpString);
  if(isFound) verbosity_level = get_verbosity_type(tmpString.c_str());

  tmpString = get_mass_normalization_str(normalization);
  isFound=opt.set("Q-mass-normalization", "Normalization of the dirac operator,options (kappa,mass,asym-mass)", verbosity, tmpString);
  if(isFound) normalization = get_mass_normalization_type(tmpString.c_str());

  tmpString = get_matpc_str(matpc_type);
  isFound=opt.set("Q-matpc", "Operator preconditioning type, options (even-even, odd-odd, even-even-asym, odd-odd-asym)", verbosity, tmpString);
  if(isFound) matpc_type = get_matpc_type(tmpString.c_str());

  tmpString = get_solve_str(solve_type);
  isFound=opt.set("Q-solve-type", "The way to solve the system, options (direct, direct-pc, normop, normop-pc, normerr, normerr-pc)", verbosity, tmpString);
  if(isFound) solve_type = get_solve_type(tmpString.c_str());

  opt.set("Q-tol", "The L2 residual tolerance", verbosity, tol);
  opt.set("Q-tolhq", "Set heavy-quark residual tolerance", verbosity, tol_hq);
  opt.set("Q-reliable-delta", "The delta factor for the reliable updates", verbosity, reliable_delta);

  //=================================== Multigrid related =======================//
  opt.set("Q-mg-levels", "The number of multigrid levels to do. One level has no meaning", verbosity, mg_levels);
  
  std::map<int,int> tpl_int_int;
  std::map<int,site> tpl_int_site;
  std::map<int,std::string> tpl_int_string;
  std::map<int,double> tpl_int_double;

  default_map_MG(tpl_int_int, 24);
  isFound=opt.set("Q-mg-nvec", "Number of null-space vectors for multigrid, usage (level,nvec)", verbosity, tpl_int_int);
  map_to_array_MG<int>(tpl_int_int, nvec, 1, 128, "ERROR: invalid number of vectors");

  default_map_MG(tpl_int_int, 0);
  isFound=opt.set("Q-mg-nu-pre", "Number of pre-smoother applications, 0-20", verbosity, tpl_int_int);
  map_to_array_MG<int>(tpl_int_int, nu_pre, 0, 128, "ERROR: invalid pre-smoother applications");

  default_map_MG(tpl_int_int, 4);
  isFound=opt.set("Q-mg-nu-post", "Number of post-smoother applications, 0-20", verbosity, tpl_int_int);
  map_to_array_MG<int>(tpl_int_int, nu_post, 0, 128, "ERROR: invalid post-smoother applications");
  
  default_map_MG(tpl_int_string, (std::string) "cg");
  isFound=opt.set("Q-mg-setup-inv", "The inverter to use for the setup of multigrid, usage(level,inv)", verbosity, tpl_int_string);
  map_to_array_MG<QudaInverterType>(tpl_int_string, setup_inv, get_solver_type);

  opt.set("Q-mg-setup-tol", "The tolerance to use for the setup of multigrid", verbosity, setup_tol);
  opt.set("Q-mg-omega", "The over/under relaxation factor for the smoother of multigrid", verbosity, omega);

  default_map_MG(tpl_int_string, (std::string) "mr");
  isFound=opt.set("Q-mg-smoother", "The smoother to use for multigrid, usage(level,inv)", verbosity, tpl_int_string);
  map_to_array_MG<QudaInverterType>(tpl_int_string, smoother_type, get_solver_type);

  default_map_MG(tpl_int_site, (site) (std::array<int,4>) {2,2,2,2});
  tpl_int_site[0] = site({4,4,4,4});
  isFound=opt.set("Q-mg-block-size", "Set the geometric block size for the each multigrid level's transfer operator", verbosity, tpl_int_site);
  { // custom map_to_array for site tuple. If needed more often create function.
    auto it = tpl_int_site.begin();
    while(it != tpl_int_site.end()){
      int lvl=it->first;
      mg_block_volume[lvl] = 1;
      site val = it->second;
      if(lvl < 0 || lvl >= QUDA_MAX_MG_LEVEL) PLEGMA_error("ERROR: invalid multigrid level %d", lvl);
      for(int j=0; j<N_DIMS; j++) {
	mg_block_size[lvl][j]=val.x[j];
	mg_block_volume[lvl]*=val.x[j];
      }
      it++;
    }
  }

  default_map_MG(tpl_int_double, 1.0);
  isFound=opt.set("Q-mg-mu-factor", "Set the multiplicative factor for the twisted mass mu parameter on each level, usage(level,float)", verbosity, tpl_int_double);
  map_to_array_MG<double>(tpl_int_double, mu_factor, 0, 100, "ERROR: invalid mu factor for the multigrid");

  default_map_MG(tpl_int_string, (std::string) "summarize");
  isFound=opt.set("Q-mg-verbosity", "The verbosity to use on each level of the multigrid, usage(level, verb)", verbosity, tpl_int_string);
  map_to_array_MG<QudaVerbosity>(tpl_int_string, mg_verbosity, get_verbosity_type);
    
  default_map_MG(tpl_int_string, (std::string) "gcr");
  isFound=opt.set("Q-mg-coarse-solver", "The solver to wrap the V cycle on each level, usage(level,inv)", verbosity, tpl_int_string);
  map_to_array_MG<QudaInverterType>(tpl_int_string, coarse_solver, get_solver_type);

  default_map_MG(tpl_int_double, 0.25);
  isFound=opt.set("Q-mg-coarse-solver-tol", "The coarse solver tolerance for each level, usage(level,float(0,1))", verbosity, tpl_int_double);
  map_to_array_MG<double>(tpl_int_double, coarse_solver_tol, 0, 1, "ERROR: invalid tolerance for the MG coarse solver");

  default_map_MG(tpl_int_int, 100);
  isFound=opt.set("Q-mg-coarse-solver-maxiter", "The coarse solver maxiter for each level, usage(level,int)", verbosity, tpl_int_int);
  map_to_array_MG<int>(tpl_int_int, coarse_solver_maxiter, 0, 10000, "ERROR: invalid max number of MG setup coarse solver max iter");

  default_map_MG(tpl_int_string, (std::string) "false");
  isFound=opt.set("Q-mg-schwarz-type", "Whether to use Schwarz preconditioning (requires MR smoother and GCR setup solver), usage(level,add/mul)", verbosity,tpl_int_string);
  map_to_array_MG<QudaSchwarzType>(tpl_int_string, schwarz_type, get_schwarz_type);

  default_map_MG(tpl_int_int, 1);
  isFound=opt.set("Q-mg-schwarz-cycle", "The number of Schwarz cycles to apply per smoother application, usage(level,int)", verbosity, tpl_int_int);
  map_to_array_MG<int>(tpl_int_int, schwarz_cycle, 0, 127, "ERROR: invalid Schwarz cycle value requested");

  default_map_MG(tpl_int_double, 0.25);
  isFound=opt.set("Q-mg-smoother-tol", "The smoother tolerance to use for each multigrid, usage(level,float(0,1))", verbosity, tpl_int_double);
  map_to_array_MG<double>(tpl_int_double, smoother_tol, 0, 1, "ERROR: invalid tolerance for the MG smoother");

  default_map_MG(tpl_int_int, 1);
  isFound=opt.set("Q-mg-setup-iters", "The number of setup iterations to use for the multigrid, usage (level,int)", verbosity, tpl_int_int);
  map_to_array_MG<int>(tpl_int_int, num_setup_iter, 0, 10, "ERROR: invalid max number of MG setup iterations refresh");
  
  opt.set("Q-mg-pre-orth", "If orthonormalize the vector before inverting in the setup of multigrid", verbosity, pre_orthonormalize);
  opt.set("Q-mg-post-orth", "If orthonormalize the vector after inverting in the setup of multigrid", verbosity, post_orthonormalize);
}
