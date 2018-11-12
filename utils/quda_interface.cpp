#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <quda_params.h>
#include <invert_quda.h>
#include <quda_solver.h>

using namespace std;
using namespace quda;

// HACK definition missing in quda headers, but available in the library
// define (static) checkGaugeParam() and checkInvertParam()
#define CHECK_PARAM
#include "check_params.h"
#undef CHECK_PARAM

// HACK definition missing in quda headers, but available in the library
// void quda::setDiracPreParam(DiracParam &diracParam, QudaInvertParam *inv_param, const bool pc, bool comms);
namespace quda{
  void createDirac(Dirac *&d, Dirac *&dSloppy, Dirac *&dPre, QudaInvertParam &param, const bool pc_solve);
}

static TimeProfile profileQUDA("QUDA"); 

Solver *solveU, *solveD;
void *mg_preconditionerUP, *mg_preconditionerDN;
QudaInvertParam inv_param;

static void initRand()
{
  int rank = 0;

#if defined(QMP_COMMS)
  rank = QMP_get_node_number();
#elif defined(MPI_COMMS)
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

  srand(17*rank + 137);
}

void initComms(int argc, char **argv, const int *commDims)
{
#if defined(QMP_COMMS)
  QMP_thread_level_t tl;
  QMP_init_msg_passing(&argc, &argv, QMP_THREAD_SINGLE, &tl);

  // FIXME? - tests crash without this
  QMP_declare_logical_topology(commDims, 4);

#elif defined(MPI_COMMS)
#ifdef PTHREADS
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
#else
  MPI_Init(&argc, &argv);
#endif

#endif
  initCommsGridQuda(4, commDims, NULL, NULL);
  initRand();
}

void finalizeComms()
{
#if defined(QMP_COMMS)
  QMP_finalize_msg_passing();
#elif defined(MPI_COMMS)
  MPI_Finalize();
#endif
}

QUDA_solver::QUDA_solver(double mu) {
  char *profiler_name;
  asprintf(&profiler_name, "Solver profiler mu=%f", mu);
  profiler = new TimeProfile(profiler_name);
  profiler->TPSTART(QUDA_PROFILE_TOTAL);
  
  mg_inv_param = newQudaInvertParam();
  mg_param = newQudaMultigridParam();
  mg_param.invert_param = &mg_inv_param;
  setMultigridParam(mg_param);
  
  inv_param = newQudaInvertParam();
  setInvertParam(inv_param);
  pushVerbosity(inv_param.verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) 
    printQudaInvertParam(&inv_param);
  checkInvertParam(&inv_param);

   // Constructing the clover term
  // This line ensures that the clover inverse is constructed
  if (mg_param.smoother_solve_type[0] == QUDA_DIRECT_PC_SOLVE || 
      solve_type == QUDA_DIRECT_PC_SOLVE) {
    inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;
  }

  if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH) 
    loadCloverQuda(NULL, NULL, &inv_param);
  
  inv_param.solve_type = solve_type; // restore actual solve_type we want to do
  
  // TODO: add support for other solvers
  if(inv_param.solve_type != QUDA_DIRECT_PC_SOLVE) 
    errorQuda("initSolver: This function works only with Direct solve and even odd preconditioning");
  
  if(inv_param.inv_type != QUDA_GCR_INVERTER) 
    errorQuda("initSolver: This function works only with GCR method");

  if(inv_param.gamma_basis != QUDA_UKQCD_GAMMA_BASIS) 
    errorQuda("initSolver: This function works only with ukqcd gamma basis\n");
  if(inv_param.dirac_order != QUDA_DIRAC_ORDER) 
    errorQuda("initSolver: This function works only with colors inside the spins\n");

  inv_param.mu = mu;
  mg_param.invert_param->mu = mu;
  mg_preconditioner = newMultigridQuda(&mg_param);

  bool pc_solution = false;
  bool pc_solve = true;
  bool mat_solution = ((inv_param.solution_type == QUDA_MAT_SOLUTION) || 
		       (inv_param.solution_type == QUDA_MATPC_SOLUTION));
  bool direct_solve = true;

  inv_param.secs = 0;
  inv_param.gflops = 0;
  inv_param.iter = 0;

  D = NULL;
  DSloppy = NULL;
  DPre = NULL;

  // create the dirac operator
  createDirac(D, DSloppy, DPre, inv_param, pc_solve);

  // Create Operators
  M = new DiracM(*D);
  MSloppy = new DiracM(*DSloppy);
  MPre = new DiracM(*DPre);

  // Create Solvers
  inv_param.preconditioner = mg_preconditioner;
  solverParam = new SolverParam(inv_param);
  solver = Solver::create(*solverParam, *M, *MSloppy, 
			 *MPre, *profiler);
}

QUDA_solver::~QUDA_solver(){
  destroyMultigridQuda(mg_preconditioner);
  delete solver;
}

template<typename Float>
void QUDA_solver::solve(PLEGMA_Vector<Float> &vectorOut, PLEGMA_Vector<Float> &vectorIn){
  bool flag_eo;
  if( inv_param.matpc_type == QUDA_MATPC_EVEN_EVEN )
    flag_eo = true;
  else if(inv_param.matpc_type == QUDA_MATPC_ODD_ODD )
    flag_eo = false;

    bool pc_solution = false;
  ColorSpinorParam cpuParam(NULL, inv_param, GK_localL, pc_solution, 
			    inv_param.input_location);
  ColorSpinorParam cudaParam(cpuParam, inv_param);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  cudaColorSpinorField b(cudaParam), x(cudaParam);

  ColorSpinorField *in = NULL;
  ColorSpinorField *out = NULL;
  vectorIn.copyToQUDA(&b,flag_eo);
  D->prepare(in,out,x,b,inv_param.solution_type);
  (*solver)(*out, *in);
  D->reconstruct(x,b,inv_param.solution_type);
  vectorOut.copyFromQUDA(&x,flag_eo);
  if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION || 
      inv_param.mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
    vectorOut.scaleVector(2*inv_param.kappa);
  }
}

template void QUDA_solver::solve(PLEGMA_Vector<float> &vectorOut, PLEGMA_Vector<float> &vectorIn);
template void QUDA_solver::solve(PLEGMA_Vector<double> &vectorOut, PLEGMA_Vector<double> &vectorIn);
