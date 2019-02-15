#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <comm_quda.h>

void plegmaOptions(Options &opt){
  bool isFound;
  opt.set("verbosity","Set verbosity level, 0 minimal, 1 verbose, 2 debug, 3 debug all", 0, verbosity);
  
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

  isFound=opt.set("load-gauge", "Path to the gauge field, default (empty string)", verbosity, latfile);
}

template<typename T> static inline void map_to_array_MG(std::map<int,T> &tpl, T *arr, T def, T noSm, T noBig, std::string err){
  for(int i=0; i<QUDA_MAX_MG_LEVEL; i++) arr[i] = def;
  if(tpl.empty()) return;
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

template<typename T> static inline void map_to_array_MG(std::map<int,std::string> &tpl, T *arr, T def, std::function<T(const char*)> func){
  for(int i=0; i<QUDA_MAX_MG_LEVEL; i++) arr[i] = def;
  if(tpl.empty()) return;
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

  if(verbosity>0) {
    PLEGMA_printf("\nParameters read by qudaOptions:\n");
  } else if(verbosity>1) {
    PLEGMA_printf("\nAll parameters available in qudaOptions with read or default value:\n");
  }
    
  isFound=opt.set("Q-prec", "Precision in the GPU, options (double,single,half), default (single)", verbosity, tmpString);
  if(isFound)prec = get_prec(tmpString.c_str());

  isFound=opt.set("Q-prec-sloppy", "Sloppy precision in the GPU, options (double,single,half), default (Q-prec)", verbosity, tmpString);
  if(isFound)prec_sloppy = get_prec(tmpString.c_str());

  isFound=opt.set("Q-prec-precondition", "Preconditioner precision in the GPU, options (double,single,half),default (Q-prec)", verbosity, tmpString);
  if(isFound)prec_precondition = get_prec(tmpString.c_str());

  isFound=opt.set("Q-prec-null", "NUll-vector precision in the GPU, options (double,single,half), default (Q-prec)", verbosity, tmpString);
  if(isFound)prec_null = get_prec(tmpString.c_str());

  isFound=opt.set("Q-recon", "Type of link reconstruction, options (8,9,12,13,18), default (18 no reconstruction)", verbosity, tmpString);
  if(isFound)link_recon  = get_recon(tmpString.c_str());

  isFound=opt.set("Q-recon-sloppy", "Type of link reconstruction for sloppy, options (8,9,12,13,18), default (Q-recon)", verbosity, tmpString);
  if(isFound)link_recon_sloppy  = get_recon(tmpString.c_str());

  isFound=opt.set("Q-recon-precondition", "Type of link reconstruction for precon, options (8,9,12,13,18), default (Q-recon)", verbosity, tmpString);
  if(isFound)link_recon_precondition  = get_recon(tmpString.c_str());

  isFound=opt.set("Q-dslash-type", "Set the dslash type, options for now (twisted-mass/twisted-clover)",verbosity, tmpString);
  if(isFound) {
    if((tmpString != "twisted-mass") && (tmpString != "twisted-clover"))PLEGMA_error("Error: only twisted-mass or twisted-clover are allowed for now");
    dslash_type =  get_dslash_type(tmpString.c_str());
  }

  isFound=opt.set("Q-dagger", "In case you want the dagger operator, default (false)", verbosity, tmpBool);
  if(isFound && tmpBool) dagger = QUDA_DAG_YES;

  isFound=opt.set("Q-kernel-pack-t", "Kernel packing in the T direction, default (false)", verbosity, tmpBool);
  if(isFound && tmpBool) kernel_pack_t = true;

  isFound=opt.set("Q-flavor", "Twisted mass type of flavor, options for now (singlet), default (singlet)", verbosity,tmpString);
  if(isFound) twist_flavor = get_flavor_type(tmpString.c_str());

  opt.set("Q-niter", "Maximum number of iterations for the solvers, options (1,...,1e6), default (10)",verbosity,niter);
  if(niter<1 || niter>1e6) PLEGMA_error("Error: Invalid number [%d] for max number of iterations\n",niter);

  opt.set("Q-ngcrkrylov", "The number of inner iterations to use for GCR or BiCGstab-l, options (1,...,1e6), (default 10)",verbosity,gcrNkrylov);
  if(gcrNkrylov<1 || gcrNkrylov>1e6) PLEGMA_error("Error: Invalid number [%d] for gcrNkrylov iterations\n",gcrNkrylov);

  opt.set("Q-pipeline", "The pipeline length for fused operations in GCR or BiCGstab-l, options(0,...,8), (default 0, no pipelining)", verbosity, pipeline);
  if(pipeline < 0 || pipeline > 8) PLEGMA_error("Error: Invalid number [%d] for pipeline length\n",pipeline);

  opt.set("Q-solution-pipeline", "The pipeline length for fused solution accumulation, options (0,..,16), (default 0, no pipelining)", verbosity, solution_accumulator_pipeline);
  if (solution_accumulator_pipeline < 0 || solution_accumulator_pipeline > 16)  PLEGMA_error("Error: Invalid number [%d] for solution pipeline length\n",solution_accumulator_pipeline);

  isFound=opt.set("Q-inv-type", "The type of solver to use, options (cg,bicgstab,gcr), default (cg)", verbosity, tmpString);
  if(isFound) inv_type = get_solver_type(tmpString.c_str());

  isFound=opt.set("Q-precon-type", "The type of precon solver to use, options (mr,none), default (none)", verbosity, tmpString);
  if(isFound)precon_type = get_solver_type(tmpString.c_str());

  opt.set("Q-kappa", "Kappa value of the Dirac operator", verbosity, kappa);
  opt.set("Q-mu", "Twisted mass value", verbosity, mu);
  opt.set("Q-csw", "The coefficient of the clover term", verbosity, csw);

  opt.set("Q-verbosity", "The verbosity of QUDA, (default summarize)", verbosity, tmpString);
  if(isFound) verbosity_level = get_verbosity_type(tmpString.c_str());

  isFound=opt.set("Q-mass-normalization", "Normalization of the dirac operator,options (kappa,mass,asym-mass), default (kappa)", verbosity, tmpString);
  if(isFound) normalization = get_mass_normalization_type(tmpString.c_str());

  isFound=opt.set("Q-matpc", "Operator preconditioning type, options (even-even, odd-odd, even-even-asym, odd-odd-asym), default(even-even)", verbosity, tmpString);
  if(isFound) matpc_type = get_matpc_type(tmpString.c_str());

  isFound=opt.set("Q-solve-type", "The way to solve the system, options (direct, direct-pc, normop, normop-pc, normerr, normerr-pc), default (direct-pc)", verbosity, tmpString);
  if(isFound) solve_type = get_solve_type(tmpString.c_str());

  opt.set("Q-tol", "The L2 residual tolerance, default (1e-09)", verbosity, tol);
  opt.set("Q-tolhq", "Set heavy-quark residual tolerance, default (0.1)", verbosity, tol_hq);
  opt.set("Q-reliable-delta", "The delta factor for the reliable updates, default (0.1)", verbosity, reliable_delta);

  //=================================== Multigrid related =======================//
  opt.set("Q-mg-levels", "The number of multigrid levels to do. One level has no meaning. (default 2)", verbosity, mg_levels);
  
  std::map<int,int> tpl_int_int;
  std::map<int,std::string> tpl_int_string;
  std::map<int,double> tpl_int_double;

  isFound=opt.set("Q-mg-nvec", "Number of null-space vectors for multigrid, usage (level,nvec)", verbosity, tpl_int_int);
  map_to_array_MG<int>(tpl_int_int, nvec, 24, 1, 128, "ERROR: invalid number of vectors");

  isFound=opt.set("Q-mg-nu-pre", "Number of pre-smoother applications, 0-20", verbosity, nu_pre);
  if(isFound) if (nu_pre < 0 || nu_pre > 20) PLEGMA_error("ERROR: invalid pre-smoother applications value (nu_pre=%d)\n", nu_pre);

  isFound=opt.set("Q-mg-nu-post", "Number of post-smoother applications, 0-20", verbosity, nu_post);
  if(isFound) if (nu_post < 0 || nu_post > 20) PLEGMA_error("ERROR: invalid post-smoother applications value (nu_post=%d)\n", nu_post);
  
  tpl_int_string.clear();
  isFound=opt.set("Q-mg-setup-inv", "The inverter to use for the setup of multigrid, usage(level,inv), (default bicgstab)", verbosity, tpl_int_string);
  map_to_array_MG<QudaInverterType>(tpl_int_string, setup_inv, QUDA_BICGSTAB_INVERTER, get_solver_type);

  opt.set("Q-mg-setup-tol", "The tolerance to use for the setup of multigrid, (default 5e-6)", verbosity, setup_tol);
  opt.set("Q-mg-omega", "The over/under relaxation factor for the smoother of multigrid (default 0.85)", verbosity, omega);

  tpl_int_string.clear();
  isFound=opt.set("Q-mg-smoother", "The smoother to use for multigrid, usage(level,inv), (default mr)", verbosity, tpl_int_string);
  map_to_array_MG<QudaInverterType>(tpl_int_string, smoother_type, QUDA_MR_INVERTER, get_solver_type);

  std::vector<int> intVec;
  isFound=opt.set("Q-mg-block-size", "Set the geometric block size for the each multigrid level's transfer operator (default 4 4 4 4)", verbosity, intVec);
  if(isFound){
    if(intVec.size()%5 != 0) PLEGMA_error("Error: For Q-mg-block-size format is (lvl,X,Y,Z,T)\n");
    int nl=intVec.size()/5;
    if(mg_levels != (nl+1)) PLEGMA_error("Error: Check that Q-mg-levels agrees with the number of levels provided in the blocks\n");
    for(int i = 0; i < nl; i++)
      for(int j = 0; j < 4; j++)
	mg_block_size[intVec[i*5]][j]=intVec[i*5+1+j];
  }

  tpl_int_double.clear();
  isFound=opt.set("Q-mg-mu-factor", "Set the multiplicative factor for the twisted mass mu parameter on each level, usage(level,float), (default 1)", verbosity, tpl_int_double);
  map_to_array_MG<double>(tpl_int_double, mu_factor, 1, 0, 100, "ERROR: invalid mu factor for the multigrid");

  tpl_int_string.clear();
  isFound=opt.set("Q-mg-verbosity", "The verbosity to use on each level of the multigrid, usage(level, verb), (default summarize)", verbosity, tpl_int_string);
  map_to_array_MG<QudaVerbosity>(tpl_int_string, mg_verbosity, QUDA_SUMMARIZE, get_verbosity_type);
    
  //==============================================//
  // KH: From now on does not exist in the QUDA 0.9. Do we need them?
  tpl_int_string.clear();
  isFound=opt.set("Q-mg-coarse-solver", "The solver to wrap the V cycle on each level, usage(level,inv), (default gcr, only for levels 1+)", verbosity, tpl_int_string);
  map_to_array_MG<QudaInverterType>(tpl_int_string, coarse_solver, QUDA_GCR_INVERTER, get_solver_type);

  tpl_int_double.clear();
  isFound=opt.set("Q-mg-coarse-solver-tol", "The coarse solver tolerance for each level, usage(level,float(0,1)), (default 0.25, only for levels 1+)", verbosity, tpl_int_double);
  map_to_array_MG<double>(tpl_int_double, coarse_solver_tol, 0.25, 0, 1, "ERROR: invalid tolerance for the MG coarse solver");

  tpl_int_int.clear();
  isFound=opt.set("Q-mg-coarse-solver-maxiter", "The coarse solver maxiter for each level, usage(level,int), (default 100)", verbosity, tpl_int_int);
  map_to_array_MG<int>(tpl_int_int, coarse_solver_maxiter, 100, 0, 10000, "ERROR: invalid max number of MG setup coarse solver max iter");

  tpl_int_string.clear();
  isFound=opt.set("Q-mg-schwarz-type", "Whether to use Schwarz preconditioning (requires MR smoother and GCR setup solver), usage(level,add/mul) (default false)", verbosity,tpl_int_string);
  map_to_array_MG<QudaSchwarzType>(tpl_int_string, schwarz_type, QUDA_INVALID_SCHWARZ, get_schwarz_type);

  tpl_int_int.clear();
  isFound=opt.set("Q-mg-schwarz-cycle", "The number of Schwarz cycles to apply per smoother application, usage(level,int), (default=1)", verbosity, tpl_int_int);
  map_to_array_MG<int>(tpl_int_int, schwarz_cycle, 1, 0, 127, "ERROR: invalid Schwarz cycle value requested");

  tpl_int_double.clear();
  isFound=opt.set("Q-mg-smoother-tol", "The smoother tolerance to use for each multigrid, usage(level,float(0,1)), (default 0.25)", verbosity, tpl_int_double);
  map_to_array_MG<double>(tpl_int_double, smoother_tol, 0.25, 0, 1, "ERROR: invalid tolerance for the MG smoother");

  tpl_int_int.clear();
  isFound=opt.set("Q-mg-setup-iters", "The number of setup iterations to use for the multigrid, usage (level,int), (default 1)", verbosity, tpl_int_int);
  map_to_array_MG<int>(tpl_int_int, num_setup_iter, 1, 0, 10, "ERROR: invalid max number of MG setup iterations refresh");
  
  opt.set("Q-mg-pre-orth", "If orthonormalize the vector before inverting in the setup of multigrid (default false)", verbosity, pre_orthonormalize);
  opt.set("Q-mg-post-orth", "If orthonormalize the vector after inverting in the setup of multigrid (default false)", verbosity, post_orthonormalize);

  // Cross-referencing in case have not been set
  if (prec_sloppy == QUDA_INVALID_PRECISION) prec_sloppy = prec;
  if (prec_precondition == QUDA_INVALID_PRECISION) prec_precondition = prec_sloppy;
  if (prec_null == QUDA_INVALID_PRECISION) prec_null = prec_precondition;
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) link_recon_sloppy = link_recon;
  if (link_recon_precondition == QUDA_RECONSTRUCT_INVALID) link_recon_precondition = link_recon_sloppy;

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

  
