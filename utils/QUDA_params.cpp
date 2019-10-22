#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include <util_quda.h>
#include <PLEGMA_utils.h>

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

namespace quda {
  extern void setTransferGPU(bool);
}

static inline int dimPartitioned(int dim)
{
  return (procs[dim] > 1);
}

void
infoQuda()
{
  PLEGMA_printf("running the following test:\n");
    
  PLEGMA_printf("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension\n");
  PLEGMA_printf("%s   %s             %s            %s            %d/%d/%d          %d\n",
		get_prec_str(prec).c_str(),get_prec_str(prec_sloppy).c_str(),
		get_recon_str(link_recon).c_str(), 
		get_recon_str(link_recon_sloppy).c_str(), dims[0], dims[1], dims[2], dims[3]);     

  PLEGMA_printf("MG parameters\n");
  PLEGMA_printf(" - number of levels %d\n", mg_levels);
  for (int i=0; i<mg_levels-1; i++) {
    PLEGMA_printf(" - level %d number of null-space vectors %d\n", i+1, nvec[i]);
    PLEGMA_printf(" - level %d number of pre-smoother applications %d\n", i+1, nu_pre[i]);
    PLEGMA_printf(" - level %d number of post-smoother applications %d\n", i+1, nu_post[i]);
  }

  PLEGMA_printf("Outer solver paramers\n");
  PLEGMA_printf(" - pipeline = %d\n", pipeline);

  PLEGMA_printf("Grid partition info:     X  Y  Z  T\n"); 
  PLEGMA_printf("                         %d  %d  %d  %d\n", 
	     dimPartitioned(0),
	     dimPartitioned(1),
	     dimPartitioned(2),
	     dimPartitioned(3)); 
  return ;  
}

QudaPrecision &cpu_prec = prec;
QudaPrecision &cuda_prec = prec;
QudaPrecision &cuda_prec_sloppy = prec_sloppy;
QudaPrecision &cuda_prec_precondition = prec_precondition;

void setGaugeParam(QudaGaugeParam &gauge_param) {
  for(int i=0; i<4; i++)
    gauge_param.X[i] = dims[i];

  gauge_param.anisotropy = anisotropy;
  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_ANTI_PERIODIC_T;
  
  gauge_param.cpu_prec = cpu_prec;

  gauge_param.cuda_prec = cuda_prec;
  gauge_param.reconstruct = link_recon;

  gauge_param.cuda_prec_sloppy = cuda_prec_sloppy;
  gauge_param.reconstruct_sloppy = link_recon_sloppy;

  gauge_param.cuda_prec_precondition = cuda_prec_precondition;
  gauge_param.reconstruct_precondition = link_recon_precondition;

  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  gauge_param.ga_pad = 0;
  // For multi-GPU, ga_pad must be large enough to store a time-slice
#ifdef MULTI_GPU
  int x_face_size = gauge_param.X[1]*gauge_param.X[2]*gauge_param.X[3]/2;
  int y_face_size = gauge_param.X[0]*gauge_param.X[2]*gauge_param.X[3]/2;
  int z_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[3]/2;
  int t_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[2]/2;
  int pad_size =MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  gauge_param.ga_pad = pad_size;    
#endif
}

#ifdef QUDA_INCLUDES_COMMIT_775a033
void setEigParam(QudaEigParam &mg_eig_param, int level)
{
  mg_eig_param.eig_type = mg_eig_type[level];
  mg_eig_param.spectrum = mg_eig_spectrum[level];
  if ((mg_eig_type[level] == QUDA_EIG_TR_LANCZOS || mg_eig_type[level] == QUDA_EIG_IR_LANCZOS)
      && !(mg_eig_spectrum[level] == QUDA_SPECTRUM_LR_EIG || mg_eig_spectrum[level] == QUDA_SPECTRUM_SR_EIG)) {
    PLEGMA_error("Only real spectrum type (LR or SR) can be passed to the a Lanczos type solver");
  }

  mg_eig_param.nEv = mg_eig_nEv[level];
  mg_eig_param.nKr = mg_eig_nKr[level];
  mg_eig_param.nConv = mg_eig_nConv[level];
  mg_eig_param.require_convergence = mg_eig_require_convergence[level];

  mg_eig_param.tol = mg_eig_tol[level];
  mg_eig_param.check_interval = mg_eig_check_interval[level];
  mg_eig_param.max_restarts = mg_eig_max_restarts[level];
  mg_eig_param.cuda_prec_ritz = cuda_prec;

  mg_eig_param.compute_svd = QUDA_BOOLEAN_NO;
  mg_eig_param.use_norm_op = mg_eig_use_normop[level];
  mg_eig_param.use_dagger = mg_eig_use_dagger[level] ;

  mg_eig_param.use_poly_acc = mg_eig_use_poly_acc[level];
  mg_eig_param.poly_deg = mg_eig_poly_deg[level];
  mg_eig_param.a_min = mg_eig_amin[level];
  mg_eig_param.a_max = mg_eig_amax[level];

  strcpy(mg_eig_param.vec_infile, (eig_vec_infile+(eig_vec_infile!=""?("_"+std::to_string(level)):"")).c_str());
  strcpy(mg_eig_param.vec_outfile, (eig_vec_outfile+(eig_vec_outfile!=""?("_"+std::to_string(level)):"")).c_str());

}
#endif

void setMultigridParam(QudaMultigridParam &mg_param) {
  QudaInvertParam &inv_param = *mg_param.invert_param;

  if (kappa == -1.0) {
    inv_param.mass = mass;
    inv_param.kappa = 1.0 / (2.0 * (1 + 3/anisotropy + mass));
  } else {
    inv_param.kappa = kappa;
    inv_param.mass = 0.5/kappa - (1.0 + 3.0/anisotropy);
  }

  PLEGMA_printf("Kappa = %.8f Mass = %.8f\n", inv_param.kappa, inv_param.mass);

  inv_param.Ls = 1;

  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;
  inv_param.cuda_prec_precondition = cuda_prec_precondition;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || 
      dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.clover_cpu_prec = cpu_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec_sloppy = cuda_prec_sloppy;
    inv_param.clover_cuda_prec_precondition = cuda_prec_precondition;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
    inv_param.clover_coeff = csw*inv_param.kappa;
  }

  inv_param.input_location = QUDA_CUDA_FIELD_LOCATION;
  inv_param.output_location = QUDA_CUDA_FIELD_LOCATION;

  inv_param.dslash_type = dslash_type;

  if (dslash_type == QUDA_TWISTED_MASS_DSLASH || 
      dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.mu = mu;
    inv_param.twist_flavor = twist_flavor;
    inv_param.Ls = (inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) ? 
      2 : 1;
    
    if (twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) {
      PLEGMA_printf("Twisted-mass doublet non supported (yet)\n");
      exit(0);
    }
  }
  
  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = normalization;

  // do we want to use an even-odd preconditioned solve or not
  if(isEven) inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
  else inv_param.matpc_type = QUDA_MATPC_ODD_ODD;

  inv_param.solution_type = QUDA_MAT_SOLUTION;

  inv_param.solve_type = QUDA_DIRECT_SOLVE;


  
  mg_param.invert_param = &inv_param;
  mg_param.n_level = mg_levels;
  for (int i=0; i<mg_param.n_level; i++) {
    for (int j=0; j<QUDA_MAX_DIM; j++) {
      // if not defined use 4
      mg_param.geo_block_size[i][j] = mg_block_size[i][j] ? 
	mg_block_size[i][j] : 4;      
    }
    mg_param.verbosity[i] = mg_verbosity[i];
    mg_param.setup_inv_type[i] = setup_inv[i];
    mg_param.num_setup_iter[i] = num_setup_iter[i];
    mg_param.setup_tol[i] = setup_tol;
    mg_param.spin_block_size[i] = 1;
    mg_param.n_vec[i] = nvec[i] == 0 ? 24 : nvec[i]; // default to 24 vectors if not set
    mg_param.precision_null[i] = prec_null; // precision to store the null-space basis
    mg_param.nu_pre[i] = nu_pre[i];
    mg_param.nu_post[i] = nu_post[i];
    mg_param.mu_factor[i] = mu_factor[i];
    
    mg_param.cycle_type[i] = QUDA_MG_CYCLE_RECURSIVE;
 
    // set the coarse solver wrappers including bottom solver
    mg_param.coarse_solver[i] = coarse_solver[i];
    mg_param.coarse_solver_tol[i] = coarse_solver_tol[i];
    mg_param.coarse_solver_maxiter[i] = coarse_solver_maxiter[i];
    mg_param.coarse_solver_ca_basis_size[i] = mg_coarse_solver_ca_basis_size[i];
    mg_param.smoother[i] = smoother_type[i];

    // set the smoother / bottom solver tolerance (for MR smoothing this will be ignored)
    mg_param.smoother_tol[i] = smoother_tol[i];

    // set to QUDA_DIRECT_PC_SOLVE for to enable even/odd 
    // preconditioning on the smoother
    mg_param.smoother_solve_type[i] = QUDA_DIRECT_PC_SOLVE; // EVEN-ODD

    // set to QUDA_ADDITIVE_SCHWARZ for Additive Schwarz precondioned smoother 
    // (presently only impelemented for MR)
    mg_param.smoother_schwarz_type[i] = schwarz_type[i];

    // if using Schwarz preconditioning then use local reductions only
    mg_param.global_reduction[i] = 
      (schwarz_type[i] == QUDA_INVALID_SCHWARZ) ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;

    // set number of Schwarz cycles to apply
    mg_param.smoother_schwarz_cycle[i] = schwarz_cycle[i];
    
    // set to QUDA_MAT_SOLUTION to inject a full field into coarse grid
    // set to QUDA_MATPC_SOLUTION to inject single parity field into coarse grid

    // if we are using an outer even-odd preconditioned solve, then we
    // use single parity injection into the coarse grid
    mg_param.coarse_grid_solution_type[i] = 
      solve_type == QUDA_DIRECT_PC_SOLVE ? QUDA_MATPC_SOLUTION : QUDA_MAT_SOLUTION;

    mg_param.omega[i] = omega; // over/under relaxation factor

    mg_param.location[i] = QUDA_CUDA_FIELD_LOCATION;
  }

  // only coarsen the spin on the first restriction
  mg_param.spin_block_size[0] = 2;

  mg_param.setup_type = setup_type;
  mg_param.pre_orthonormalize = pre_orthonormalize ? QUDA_BOOLEAN_YES :  QUDA_BOOLEAN_NO;
  mg_param.post_orthonormalize = post_orthonormalize ? QUDA_BOOLEAN_YES :  QUDA_BOOLEAN_NO;

  mg_param.compute_null_vector = generate_nullspace ? 
    QUDA_COMPUTE_NULL_VECTOR_YES : QUDA_COMPUTE_NULL_VECTOR_NO;
  mg_param.generate_all_levels = generate_all_levels ? 
    QUDA_BOOLEAN_YES :  QUDA_BOOLEAN_NO;

  mg_param.run_verify = verify_results ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;

  
#ifdef QUDA_INCLUDES_COMMIT_b08233a
  mg_param.run_low_mode_check = QUDA_BOOLEAN_NO; 
  mg_param.run_oblique_proj_check = QUDA_BOOLEAN_NO;
#endif

#ifdef QUDA_INCLUDES_COMMIT_775a033
  for(size_t level=0;level<mg_levels; level++)
    mg_param.use_eig_solver[level]=mg_eig[level];
#endif

#ifdef QUDA_INCLUDES_COMMIT_55782743
  mg_param.preserve_deflation = preserve_deflation;
#endif
#ifndef QUDA_INCLUDES_COMMIT_55782743
  if(preserve_deflation) PLEGMA_error("The current version of QUDA does not include the preserve_deflation feature");
#endif
  // set file i/o parameters
#ifdef QUDA_INCLUDES_COMMIT_1dec1db
  for (int i=0; i<mg_param.n_level; i++) {
    strcpy(mg_param.vec_infile[i], (vec_infile+(vec_infile!=""?("_"+std::to_string(i)):"")).c_str());
    strcpy(mg_param.vec_outfile[i], (vec_outfile+(vec_outfile!=""?("_"+std::to_string(i)):"")).c_str());
  }
#else
  strcpy(mg_param.vec_infile, vec_infile.c_str());
  strcpy(mg_param.vec_outfile, vec_outfile.c_str());
#endif
  
  // these need to be set for now but are actually ignored by the MG setup
  // needed to make it pass the initialization test
  inv_param.inv_type = QUDA_GCR_INVERTER;
  inv_param.tol = tol;
  inv_param.maxiter = niter;
  inv_param.reliable_delta = 1e-10;
  inv_param.gcrNkrylov = 10;

  inv_param.verbosity = QUDA_SUMMARIZE;
  inv_param.verbosity_precondition = QUDA_SUMMARIZE;
}

void setInvertParam(QudaInvertParam &inv_param) {

  if (kappa == -1.0) {
    inv_param.mass = mass;
    inv_param.kappa = 1.0 / (2.0 * (1 + 3/anisotropy + mass));
  } else {
    inv_param.kappa = kappa;
    inv_param.mass = 0.5/kappa - (1.0 + 3.0/anisotropy);
  }
  
  PLEGMA_printf("Kappa = %.8f Mass = %.8f\n", inv_param.kappa, inv_param.mass);


  inv_param.Ls = 1;

  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;

  inv_param.cuda_prec_precondition = cuda_prec_precondition;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param.gamma_basis = QUDA_UKQCD_GAMMA_BASIS;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || 
      dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.clover_cpu_prec = cpu_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec_sloppy = cuda_prec_sloppy;
    inv_param.clover_cuda_prec_precondition = cuda_prec_precondition;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
    inv_param.clover_coeff = csw*inv_param.kappa;
  }

  inv_param.input_location = QUDA_CUDA_FIELD_LOCATION;
  inv_param.output_location = QUDA_CUDA_FIELD_LOCATION;

  inv_param.dslash_type = dslash_type;

  if (dslash_type == QUDA_TWISTED_MASS_DSLASH || 
      dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.mu = mu;
    inv_param.twist_flavor = twist_flavor;
    inv_param.Ls = (inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) ? 
      2 : 1;

    if (twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) {
      PLEGMA_printf("Twisted-mass doublet non supported (yet)\n");
      exit(0);
    }
  }

  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = normalization;

  // do we want full solution or single-parity solution
  inv_param.solution_type = QUDA_MAT_SOLUTION ;

  // do we want to use an even-odd preconditioned solve or not
  inv_param.solve_type = solve_type;
  if(isEven) { 
    inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
    PLEGMA_printf("### Running for the Even-Even Operator\n");
  }
  else {
    PLEGMA_printf("### Running for the Odd-Odd Operator\n");
    inv_param.matpc_type = QUDA_MATPC_ODD_ODD;
  }

  inv_param.inv_type = solver_type;

  inv_param.verbosity_precondition = mg_verbosity[0];

  inv_param.inv_type_precondition = use_mg? QUDA_MG_INVERTER : QUDA_INVALID_INVERTER;
  inv_param.pipeline = pipeline;
  inv_param.gcrNkrylov = gcrNkrylov;
  inv_param.tol = tol;

  // require both L2 relative and heavy quark residual to determine 
  // convergence
  inv_param.residual_type = 
    static_cast<QudaResidualType>(QUDA_L2_RELATIVE_RESIDUAL);
  // specify a tolerance for the residual for heavy quark residual
  inv_param.tol_hq = tol_hq; 
  inv_param.compute_true_res = true;
  
  // these can be set individually
  for (int i=0; i<inv_param.num_offset; i++) {
    inv_param.tol_offset[i] = inv_param.tol;
    inv_param.tol_hq_offset[i] = inv_param.tol_hq;
  }
  inv_param.maxiter = niter;
  inv_param.reliable_delta = reliable_delta; 

  // domain decomposition preconditioner parameters
  inv_param.schwarz_type = QUDA_ADDITIVE_SCHWARZ;
  inv_param.precondition_cycle = 1;
  inv_param.tol_precondition = 1e-1;
  inv_param.maxiter_precondition = 1;
  inv_param.omega = 1.0;

  inv_param.verbosity = verbosity_level;
}


#ifdef QUDA_INCLUDES_COMMIT_775a033
void setEigMultigridParam(QudaMultigridParam &mg_param, QudaEigParam *mg_eig_param){
  for (size_t level=0; level<mg_param.n_level; level++) {
    if (mg_eig[level]==QUDA_BOOLEAN_YES) {
      mg_eig_param[level] = newQudaEigParam();
      setEigParam(mg_eig_param[level], level);
      mg_param.eig_param[level] = &mg_eig_param[level];
    } else {
      mg_param.eig_param[level] = nullptr;
    }
  }
}
#endif
  
