#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <comm_quda.h>

static void printBasicOptions(){
  _PRINT_("dims, %d %d %d %d\n",dims[0],dims[1],dims[2],dims[3]);
  _PRINT_("procs, %d %d %d %d\n",procs[0],procs[1],procs[2],procs[3]);
  _PRINT_("load-gauge, %s\n", latfile.c_str());
  _PRINT_("verbose, %d\n", verbose);
}

static void printQudaSolverOptions(){
  _PRINT_("Q-prec, %d\n",prec);
  _PRINT_("Q-prec-sloppy, %d\n",prec_sloppy);
  _PRINT_("Q-prec-precondition, %d\n",prec_precondition);
  _PRINT_("Q-prec-null, %d\n",prec_null);
  _PRINT_("Q-recon, %d\n",link_recon);
  _PRINT_("Q-recon-sloppy, %d\n",link_recon_sloppy);
  _PRINT_("Q-recon-precondition, %d\n",link_recon_precondition);
  _PRINT_("Q-dslash-type, %d\n", dslash_type);
  _PRINT_("Q-dagger, %d\n", dagger);
  _PRINT_("Q-kernel-pack-t, %d\n",kernel_pack_t);
  _PRINT_("Q-flavor, %d\n",twist_flavor);
  _PRINT_("Q-niter, %d\n",niter);
  _PRINT_("Q-ngcrkrylov, %d\n",gcrNkrylov);
  _PRINT_("Q-pipeline, %d\n",pipeline);
  _PRINT_("Q-solution-pipeline, %d\n",solution_accumulator_pipeline);
  _PRINT_("Q-inv-type, %d\n",inv_type);
  _PRINT_("Q-precon-type, %d\n",precon_type);
  _PRINT_("Q-kappa, %f\n",kappa);
  _PRINT_("Q-mu, %f\n",mu);
  _PRINT_("Q-csw, %f\n",csw);
  _PRINT_("Q-mass-normalization, %d\n",normalization);
  _PRINT_("Q-matpc, %d\n",matpc_type);
  _PRINT_("Q-solve-type, %d\n", solve_type);
  _PRINT_("Q-tol, %+e\n", tol);
  _PRINT_("Q-tolhq, %+e\n", tol_hq);
  _PRINT_("Q-reliable-delta, %+f\n", reliable_delta);
  _PRINT_("Q-mg-levels, %d\n", mg_levels);
  for(int i = 0; i < mg_levels-1; i++)
    _PRINT_("Q-mg-nvec, %d %d\n",i,nvec[i]);
  _PRINT_("Q-mg-nu-pre, %d\n", nu_pre);
  _PRINT_("Q-mg-nu-post, %d\n", nu_post);
  for(int i = 0; i < mg_levels-1; i++)
    _PRINT_("Q-mg-setup-inv, %d %d\n",i,setup_inv[i]);
  _PRINT_("Q-mg-setup-tol, %+e\n", setup_tol);
  _PRINT_("Q-mg-omega, %f\n", omega);
  for(int i = 0; i < mg_levels-1; i++)
    _PRINT_("Q-mg-smoother, %d %d\n", i, smoother_type[i]);
  for(int i = 0; i < mg_levels-1; i++)
    _PRINT_("Q-mg-block-size, %d \t %d %d %d %d\n", i, mg_block_size[i][0], mg_block_size[i][1], mg_block_size[i][2], mg_block_size[i][3]);
  for(int i = 0; i < mg_levels-1; i++)
    _PRINT_("Q-mg-mu-factor, %d %f\n", i, mu_factor[i]);
  for(int i = 0; i < mg_levels-1; i++)
    _PRINT_("Q-mg-verbosity, %d %d\n", i, mg_verbosity[i]);
  for(int i = 0; i < mg_levels-1; i++)
    _PRINT_("Q-mg-coarse-solver, %d %d\n", i, coarse_solver[i]);
  for(int i = 0; i < mg_levels-1; i++)
    _PRINT_("Q-mg-coarse-solver-tol, %d %f\n",i,coarse_solver_tol[i]);
  for(int i = 0; i < mg_levels-1; i++)
    _PRINT_("Q-mg-coarse-solver-maxiter, %d %d\n", i, coarse_solver_maxiter[i]);
  for(int i = 0; i < mg_levels-1; i++)
    _PRINT_("Q-mg-schwarz-type, %d %d\n", i, schwarz_type[i]);
  for(int i = 0; i < mg_levels-1; i++)
    _PRINT_("Q-mg-schwarz-cycle, %d %d\n", i, schwarz_cycle[i]);
  for(int i = 0; i < mg_levels-1; i++)
    _PRINT_("Q-mg-smoother-tol, %d %f\n",i,smoother_tol[i]);
  for(int i = 0; i < mg_levels-1; i++)
    _PRINT_("Q-mg-setup-iters, %d %d\n", i, num_setup_iter[i]);
  _PRINT_("Q-mg-pre-orth, %d\n", pre_orthonormalize);
  _PRINT_("Q-mg-post-orth, %d\n", post_orthonormalize);
}

void basicOptions(Options &opt){
  bool isFound, debug=false;
  opt.set("verbose","Set verbosity level, 0 minimal, 1 verbose, 2 debug, 3 debug all", false, verbose);
  if(verbose > 2) debug=true;
  
  opt.setForced("dims","Set local dimensions (X Y Z T), e.g. 8 8 8 16", debug, dims[0], dims[1], dims[2], dims[3]);
  if(!opt.getIsHelp()) for(int i=0; i<4; i++) if( (dims[i] <= 0 || dims[i] > 512) ) _ERROR_("Error with dim %d: dims should be > 0 and < 512\n", i);

  opt.setForced("procs","Set number of processors (X Y Z T), e.g. 1 1 1 1", debug, procs[0], procs[1], procs[2], procs[3]);
  if(!opt.getIsHelp()) for(int i=0; i<4; i++) if( procs[i] <= 0 ) _ERROR_("Error with dim %d: Negative proc or not divisor of dim\n", i);

  isFound=opt.set("load-gauge", "Path to the gauge field, default (empty string)", debug, latfile);

  if(verbose > 1 && !opt.getIsHelp()) printBasicOptions();
}

template<typename T> static inline void map_to_array_MG(std::map<int,T> &tpl, T *arr, T def, T noSm, T noBig, std::string err){
  for(int i=0; i<QUDA_MAX_MG_LEVEL; i++) arr[i] = def;
  if(tpl.empty()) return;
  typename std::map<int,T>::iterator it = tpl.begin();
  while(it != tpl.end()){
    int lvl=it->first;
    T val = it->second;
    if(lvl < 0 || lvl >= QUDA_MAX_MG_LEVEL) _ERROR_("ERROR: invalid multigrid level %d", lvl);
    if(val < noSm || val > noBig) _ERROR_(err.c_str());
    arr[lvl]=val;
    it++;
  }
}

template<typename T> static inline void map_to_array_MG(std::map<int,std::string> &tpl, T *arr, T def, std::function<T(const char*)> func){
  for(int i=0; i<QUDA_MAX_MG_LEVEL; i++) arr[i] = def;
  if(tpl.empty()) return;
  typename std::map<int,std::string>::iterator it = tpl.begin();
  while(it != tpl.end()){
    int lvl=it->first;
    T val = func((it->second).c_str());
    if(lvl < 0 || lvl >= QUDA_MAX_MG_LEVEL) _ERROR_("ERROR: invalid multigrid level %d", lvl);
    arr[lvl]=val;
    it++;
  }
}

void qudaSolverOptions(Options &opt){
  std::string tmpString;
  bool tmpBool;
  bool isFound, debug=false;
  if(verbose > 2) debug=true;
    
  isFound=opt.set("Q-prec", "Precision in the GPU, options (double,single,half), default (single)", debug, tmpString);
  if(isFound)prec = get_prec(tmpString.c_str());

  isFound=opt.set("Q-prec-sloppy", "Sloppy precision in the GPU, options (double,single,half), default (Q-prec)", debug, tmpString);
  if(isFound)prec_sloppy = get_prec(tmpString.c_str());

  isFound=opt.set("Q-prec-precondition", "Preconditioner precision in the GPU, options (double,single,half),default (Q-prec)", debug, tmpString);
  if(isFound)prec_precondition = get_prec(tmpString.c_str());

  isFound=opt.set("Q-prec-null", "NUll-vector precision in the GPU, options (double,single,half), default (Q-prec)", debug, tmpString);
  if(isFound)prec_null = get_prec(tmpString.c_str());

  isFound=opt.set("Q-recon", "Type of link reconstruction, options (8,9,12,13,18), default (18 no reconstruction)", debug, tmpString);
  if(isFound)link_recon  = get_recon(tmpString.c_str());

  isFound=opt.set("Q-recon-sloppy", "Type of link reconstruction for sloppy, options (8,9,12,13,18), default (Q-recon)", debug, tmpString);
  if(isFound)link_recon_sloppy  = get_recon(tmpString.c_str());

  isFound=opt.set("Q-recon-precondition", "Type of link reconstruction for precon, options (8,9,12,13,18), default (Q-recon)", debug, tmpString);
  if(isFound)link_recon_precondition  = get_recon(tmpString.c_str());

  isFound=opt.set("Q-dslash-type", "Set the dslash type, options for now (twisted-mass/twisted-clover)",debug, tmpString);
  if(isFound) {
    if((tmpString != "twisted-mass") && (tmpString != "twisted-clover"))_ERROR_("Error: only twisted-mass or twisted-clover are allowed for now");
    dslash_type =  get_dslash_type(tmpString.c_str());
  }

  isFound=opt.set("Q-dagger", "In case you want the dagger operator, default (false)", debug, tmpBool);
  if(isFound && tmpBool) dagger = QUDA_DAG_YES;

  isFound=opt.set("Q-kernel-pack-t", "Kernel packing in the T direction, default (false)", debug, tmpBool);
  if(isFound && tmpBool) kernel_pack_t = true;

  isFound=opt.set("Q-flavor", "Twisted mass type of flavor, options for now (singlet), default (singlet)", debug,tmpString);
  if(isFound) twist_flavor = get_flavor_type(tmpString.c_str());

  opt.set("Q-niter", "Maximum number of iterations for the solvers, options (1,...,1e6), default (10)",debug,niter);
  if(niter<1 || niter>1e6) _ERROR_("Error: Invalid number [%d] for max number of iterations\n",niter);

  opt.set("Q-ngcrkrylov", "The number of inner iterations to use for GCR or BiCGstab-l, options (1,...,1e6), (default 10)",debug,gcrNkrylov);
  if(gcrNkrylov<1 || gcrNkrylov>1e6) _ERROR_("Error: Invalid number [%d] for gcrNkrylov iterations\n",gcrNkrylov);

  opt.set("Q-pipeline", "The pipeline length for fused operations in GCR or BiCGstab-l, options(0,...,8), (default 0, no pipelining)", debug, pipeline);
  if(pipeline < 0 || pipeline > 8) _ERROR_("Error: Invalid number [%d] for pipeline length\n",pipeline);

  opt.set("Q-solution-pipeline", "The pipeline length for fused solution accumulation, options (0,..,16), (default 0, no pipelining)", debug, solution_accumulator_pipeline);
  if (solution_accumulator_pipeline < 0 || solution_accumulator_pipeline > 16)  _ERROR_("Error: Invalid number [%d] for solution pipeline length\n",solution_accumulator_pipeline);

  isFound=opt.set("Q-inv-type", "The type of solver to use, options (cg,bicgstab,gcr), default (cg)", debug, tmpString);
  if(isFound) inv_type = get_solver_type(tmpString.c_str());

  isFound=opt.set("Q-precon-type", "The type of precon solver to use, options (mr,none), default (none)", debug, tmpString);
  if(isFound)precon_type = get_solver_type(tmpString.c_str());

  opt.set("Q-kappa", "Kappa value of the Dirac operator", debug, kappa);
  opt.set("Q-mu", "Twisted mass value", debug, mu);
  opt.set("Q-csw", "The coefficient of the clover term", debug, csw);

  opt.set("Q-verbosity", "The verbosity of QUDA, (default summarize)", debug, tmpString);
  if(isFound) verbosity_level = get_verbosity_type(tmpString.c_str());

  isFound=opt.set("Q-mass-normalization", "Normalization of the dirac operator,options (kappa,mass,asym-mass), default (kappa)", debug, tmpString);
  if(isFound) normalization = get_mass_normalization_type(tmpString.c_str());

  isFound=opt.set("Q-matpc", "Operator preconditioning type, options (even-even, odd-odd, even-even-asym, odd-odd-asym), default(even-even)", debug, tmpString);
  if(isFound) matpc_type = get_matpc_type(tmpString.c_str());

  isFound=opt.set("Q-solve-type", "The way to solve the system, options (direct, direct-pc, normop, normop-pc, normerr, normerr-pc), default (direct-pc)", debug, tmpString);
  if(isFound) solve_type = get_solve_type(tmpString.c_str());

  opt.set("Q-tol", "The L2 residual tolerance, default (1e-09)", debug, tol);
  opt.set("Q-tolhq", "Set heavy-quark residual tolerance, default (0.1)", debug, tol_hq);
  opt.set("Q-reliable-delta", "The delta factor for the reliable updates, default (0.1)", debug, reliable_delta);

  //=================================== Multigrid related =======================//
  opt.set("Q-mg-levels", "The number of multigrid levels to do. One level has no meaning. (default 2)", debug, mg_levels);
  
  std::map<int,int> tpl_int_int;
  std::map<int,std::string> tpl_int_string;
  std::map<int,double> tpl_int_double;

  isFound=opt.set("Q-mg-nvec", "Number of null-space vectors for multigrid, usage (level,nvec)", debug, tpl_int_int);
  map_to_array_MG<int>(tpl_int_int, nvec, 24, 1, 128, "ERROR: invalid number of vectors");

  isFound=opt.set("Q-mg-nu-pre", "Number of pre-smoother applications, 0-20", debug, nu_pre);
  if(isFound) if (nu_pre < 0 || nu_pre > 20) _ERROR_("ERROR: invalid pre-smoother applications value (nu_pre=%d)\n", nu_pre);

  isFound=opt.set("Q-mg-nu-post", "Number of post-smoother applications, 0-20", debug, nu_post);
  if(isFound) if (nu_post < 0 || nu_post > 20) _ERROR_("ERROR: invalid post-smoother applications value (nu_post=%d)\n", nu_post);
  
  tpl_int_string.clear();
  isFound=opt.set("Q-mg-setup-inv", "The inverter to use for the setup of multigrid, usage(level,inv), (default bicgstab)", debug, tpl_int_string);
  map_to_array_MG<QudaInverterType>(tpl_int_string, setup_inv, QUDA_BICGSTAB_INVERTER, get_solver_type);

  opt.set("Q-mg-setup-tol", "The tolerance to use for the setup of multigrid, (default 5e-6)", debug, setup_tol);
  opt.set("Q-mg-omega", "The over/under relaxation factor for the smoother of multigrid (default 0.85)", debug, omega);

  tpl_int_string.clear();
  isFound=opt.set("Q-mg-smoother", "The smoother to use for multigrid, usage(level,inv), (default mr)", debug, tpl_int_string);
  map_to_array_MG<QudaInverterType>(tpl_int_string, smoother_type, QUDA_MR_INVERTER, get_solver_type);

  std::vector<int> intVec;
  isFound=opt.set("Q-mg-block-size", "Set the geometric block size for the each multigrid level's transfer operator (default 4 4 4 4)", debug, intVec);
  if(isFound){
    if(intVec.size()%5 != 0) _ERROR_("Error: For Q-mg-block-size format is (lvl,X,Y,Z,T)\n");
    int nl=intVec.size()/5;
    if(mg_levels != (nl+1)) _ERROR_("Error: Check that Q-mg-levels agrees with the number of levels provided in the blocks\n");
    for(int i = 0; i < nl; i++)
      for(int j = 0; j < 4; j++)
	mg_block_size[intVec[i*5]][j]=intVec[i*5+1+j];
  }

  tpl_int_double.clear();
  isFound=opt.set("Q-mg-mu-factor", "Set the multiplicative factor for the twisted mass mu parameter on each level, usage(level,float), (default 1)", debug, tpl_int_double);
  map_to_array_MG<double>(tpl_int_double, mu_factor, 1, 0, 100, "ERROR: invalid mu factor for the multigrid");

  tpl_int_string.clear();
  isFound=opt.set("Q-mg-verbosity", "The verbosity to use on each level of the multigrid, usage(level, verb), (default summarize)", debug, tpl_int_string);
  map_to_array_MG<QudaVerbosity>(tpl_int_string, mg_verbosity, QUDA_SUMMARIZE, get_verbosity_type);
    
  //==============================================//
  // KH: From now on does not exist in the QUDA 0.9. Do we need them?
  tpl_int_string.clear();
  isFound=opt.set("Q-mg-coarse-solver", "The solver to wrap the V cycle on each level, usage(level,inv), (default gcr, only for levels 1+)", debug, tpl_int_string);
  map_to_array_MG<QudaInverterType>(tpl_int_string, coarse_solver, QUDA_GCR_INVERTER, get_solver_type);

  tpl_int_double.clear();
  isFound=opt.set("Q-mg-coarse-solver-tol", "The coarse solver tolerance for each level, usage(level,float(0,1)), (default 0.25, only for levels 1+)", debug, tpl_int_double);
  map_to_array_MG<double>(tpl_int_double, coarse_solver_tol, 0.25, 0, 1, "ERROR: invalid tolerance for the MG coarse solver");

  tpl_int_int.clear();
  isFound=opt.set("Q-mg-coarse-solver-maxiter", "The coarse solver maxiter for each level, usage(level,int), (default 100)", debug, tpl_int_int);
  map_to_array_MG<int>(tpl_int_int, coarse_solver_maxiter, 100, 0, 10000, "ERROR: invalid max number of MG setup coarse solver max iter");

  tpl_int_string.clear();
  isFound=opt.set("Q-mg-schwarz-type", "Whether to use Schwarz preconditioning (requires MR smoother and GCR setup solver), usage(level,add/mul) (default false)", debug,tpl_int_string);
  map_to_array_MG<QudaSchwarzType>(tpl_int_string, schwarz_type, QUDA_INVALID_SCHWARZ, get_schwarz_type);

  tpl_int_int.clear();
  isFound=opt.set("Q-mg-schwarz-cycle", "The number of Schwarz cycles to apply per smoother application, usage(level,int), (default=1)", debug, tpl_int_int);
  map_to_array_MG<int>(tpl_int_int, schwarz_cycle, 1, 0, 127, "ERROR: invalid Schwarz cycle value requested");

  tpl_int_double.clear();
  isFound=opt.set("Q-mg-smoother-tol", "The smoother tolerance to use for each multigrid, usage(level,float(0,1)), (default 0.25)", debug, tpl_int_double);
  map_to_array_MG<double>(tpl_int_double, smoother_tol, 0.25, 0, 1, "ERROR: invalid tolerance for the MG smoother");

  tpl_int_int.clear();
  isFound=opt.set("Q-mg-setup-iters", "The number of setup iterations to use for the multigrid, usage (level,int), (default 1)", debug, tpl_int_int);
  map_to_array_MG<int>(tpl_int_int, num_setup_iter, 1, 0, 10, "ERROR: invalid max number of MG setup iterations refresh");
  
  opt.set("Q-mg-pre-orth", "If orthonormalize the vector before inverting in the setup of multigrid (default false)", debug, pre_orthonormalize);
  opt.set("Q-mg-post-orth", "If orthonormalize the vector after inverting in the setup of multigrid (default false)", debug, post_orthonormalize);

  // Cross-referencing in case have not been set
  if (prec_sloppy == QUDA_INVALID_PRECISION) prec_sloppy = prec;
  if (prec_precondition == QUDA_INVALID_PRECISION) prec_precondition = prec_sloppy;
  if (prec_null == QUDA_INVALID_PRECISION) prec_null = prec_precondition;
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) link_recon_sloppy = link_recon;
  if (link_recon_precondition == QUDA_RECONSTRUCT_INVALID) link_recon_precondition = link_recon_sloppy;

  if(verbose > 1 && !opt.getIsHelp()) printQudaSolverOptions();
}

  //  isFound=opt.set("Q-prec-refine", "Sloppy precision for refinement in the GPU, options (double,single,half),default (invalid)", visualize, tmpString);
  //if(isFound)prec_refinement_sloppy = get_prec(tmpString.c_str());

  //  tpl_int_string.clear();
  // isFound=opt.set("Q-mg-coarse-solve-type", "The type of solve to do on each level, usage(level,solve), (direct, direct-pc) (default = solve_type)", visualize, tpl_int_string);
  // if(isFound) map_to_array_MG<QudaSolveType>(tpl_int_string, coarse_solve_type, get_solve_type);

  // tpl_int_string.clear();
  // isFound=opt.set("Q-mg-smoother-solve-type", "The type of solve to do on smoother, usage(level,solve), (direct, direct-pc) (default = direct-pc)", visualize, tpl_int_string);
  // if(isFound) map_to_array_MG<QudaSolveType>(tpl_int_string, smoother_solve_type, get_solve_type);

  // tpl_int_int.clear();
  // isFound=opt.set("Q-mg-setup-maxiter", "The maximum number of solver iterations to use when relaxing on a null space vector, usage (level,int), (default 500)", visualize, tpl_int_int);
  // if(isFound) map_to_array_MG<int>(tpl_int_int, setup_maxiter, 0, 10000, "ERROR: invalid max number of MG setup iterations");

  // tpl_int_int.clear();
  // isFound=opt.set("Q-mg-setup-maxiter-refresh", "The maximum number of solver iterations to use when refreshing the pre-existing null space vectors, usage(level,int), (default 100)", visualize, tpl_int_int);
  // if(isFound) map_to_array_MG<int>(tpl_int_int, setup_maxiter_refresh, 0, 10000, "ERROR: invalid max number of MG setup iterations refresh");

  //  isFound=opt.set("Q-mg-smoother-halo-prec", "The smoother halo precision (applies to all levels - defaults to null_precision)", visualize, tmpString );
  //  if(isFound) smoother_halo_prec = get_prec(tmpString.c_str());

  
