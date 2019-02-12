//-----------------//
// QUDA Parameters //
//-----------------//
#include <utils/MACRO_IS_EMPTY.hpp>

#include <boost/preprocessor/control/if.hpp>
#ifdef ALLOCATE
#define NOTHING(...)
#define EQUAL_CAT(...) =  __VA_ARGS__
#define EQUAL(...) BOOST_PP_IF(IS_EMPTY(__VA_ARGS__),		\
			       NOTHING,				\
			       EQUAL_CAT) (__VA_ARGS__)
#define define(var,...) var EQUAL(__VA_ARGS__)
#else
#define define(var,...) extern var 
#endif

#ifdef MULTI_GPU
define(int device, -1);
#else
define(int device, 0);
#endif

define(bool qudaInitialized, false);

define(QudaVerbosity verbosity_level, QUDA_SUMMARIZE);
define(QudaDslashType dslash_type, QUDA_TWISTED_CLOVER_DSLASH);
define(QudaReconstructType link_recon, QUDA_RECONSTRUCT_NO);
define(QudaReconstructType link_recon_sloppy, QUDA_RECONSTRUCT_INVALID);
define(QudaReconstructType link_recon_precondition, QUDA_RECONSTRUCT_INVALID);
define(QudaPrecision prec, QUDA_DOUBLE_PRECISION);
define(QudaPrecision  prec_sloppy, QUDA_INVALID_PRECISION);
define(QudaPrecision  prec_precondition, QUDA_INVALID_PRECISION);
define(QudaPrecision prec_null, QUDA_INVALID_PRECISION);

// Dirac operator options
define(double mass, 0.1);  // mass of Dirac operator
define(double kappa, -1.0); // kappa of Dirac operator
define(double mu, 0.1); // mu of twisted mass operator
define(double csw, 0.1);
define(double anisotropy, 1.0);
define(QudaTwistFlavorType twist_flavor, QUDA_TWIST_SINGLET);
define(bool compute_clover, false);
define(QudaMassNormalization normalization, QUDA_KAPPA_NORMALIZATION); // mass normalization of Dirac operators
define(QudaDagType dagger, QUDA_DAG_NO);
define(bool isEven, true);

// Solver options
define(double tol, 1e-9); // tolerance for inverter
define(double tol_hq, 0.1); // heavy-quark tolerance for inverter
define(double reliable_delta, 1e-4);
define(int niter,100);
define(QudaMatPCType matpc_type, QUDA_MATPC_EVEN_EVEN);
define(QudaSolveType solve_type, QUDA_DIRECT_PC_SOLVE);
define(bool verify_results, false);
define(bool kernel_pack_t, false);
define(int pipeline, 0); // length of pipeline for fused operations in GCR or BiCGstab-l
define(int solution_accumulator_pipeline, 0);
define(QudaInverterType inv_type, QUDA_MG_INVERTER);
define(QudaInverterType precon_type, QUDA_INVALID_INVERTER);

// MG options
define(int gcrNkrylov, 10); // number of inner iterations for GCR, or l for BiCGstab-l
define(int nvec[QUDA_MAX_MG_LEVEL], {});
define(int mg_levels, 2);
define(bool generate_nullspace, true);
define(bool generate_all_levels, true);
define(int nu_pre, 0);
define(int nu_post, 4);
define(int mg_block_size[QUDA_MAX_MG_LEVEL][QUDA_MAX_DIM], {});
define(double mu_factor[QUDA_MAX_MG_LEVEL], {});
define(QudaVerbosity mg_verbosity[QUDA_MAX_MG_LEVEL], {});
define(QudaInverterType setup_inv[QUDA_MAX_MG_LEVEL],{});
define(int num_setup_iter[QUDA_MAX_MG_LEVEL], {});
define(double setup_tol, 5e-6);
define(QudaSetupType setup_type, QUDA_NULL_VECTOR_SETUP);
define(bool pre_orthonormalize, false);
define(bool post_orthonormalize, true);
define(double omega, 0.85);
define(QudaInverterType coarse_solver[QUDA_MAX_MG_LEVEL], {});
define(QudaInverterType smoother_type[QUDA_MAX_MG_LEVEL], {});
define(double coarse_solver_tol[QUDA_MAX_MG_LEVEL], {});
define(double smoother_tol[QUDA_MAX_MG_LEVEL], {});
define(int coarse_solver_maxiter[QUDA_MAX_MG_LEVEL], {});
define(QudaSchwarzType schwarz_type[QUDA_MAX_MG_LEVEL], {});
define(int schwarz_cycle[QUDA_MAX_MG_LEVEL], {});
define(std::string vec_infile, "");
define(std::string vec_outfile, "");

#undef define
