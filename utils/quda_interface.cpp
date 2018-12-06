#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <quda_params.h>
#include <invert_quda.h>
#include <quda_solver.h>
#include <PLEGMA_Qdirac.h>

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
  void setDiracParam(DiracParam &diracParam, QudaInvertParam *inv_param, const bool pc);
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

void initGaugeQuda(void* gauge, QudaGaugeParam gauge_param) {
  loadGaugeQuda(gauge, &gauge_param);

  if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH)  {
    QudaInvertParam inv_param = newQudaInvertParam();
    setInvertParam(inv_param);
    checkInvertParam(&inv_param);

    inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;
    loadCloverQuda(NULL, NULL, &inv_param);
  }
}

void finalizeGaugeQuda() {
  freeGaugeQuda();
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || 
      dslash_type == QUDA_TWISTED_CLOVER_DSLASH) freeCloverQuda();
}

void updateGaugeQuda(void* gauge, QudaGaugeParam gauge_param) {
  finalizeGaugeQuda();
  initGaugeQuda(gauge, gauge_param);
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
  checkInvertParam(&inv_param);
  
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

  ColorSpinorParam cpuParam(NULL, inv_param, GK_localL, pc_solution,
			    inv_param.input_location);
  ColorSpinorParam cudaParam(cpuParam, inv_param);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  b = new cudaColorSpinorField(cudaParam);
  x = new cudaColorSpinorField(cudaParam);
}

QUDA_solver::~QUDA_solver(){
  destroyMultigridQuda(mg_preconditioner);
  delete solver;
  delete profiler;
  delete b;
  delete x;
  delete M;
  delete MSloppy;
  delete MPre;
  delete D;
  delete DSloppy;
  delete DPre;
}

cudaColorSpinorField *QUDA_solver::solve(cudaColorSpinorField * rhs){
  ColorSpinorField *in = NULL;
  ColorSpinorField *out = NULL;
  D->prepare(in,out,*x,*rhs,inv_param.solution_type);
  (*solver)(*out, *in);
  D->reconstruct(*x,*rhs,inv_param.solution_type);
  return x;
}

template<typename Float>
cudaColorSpinorField *QUDA_solver::solve(PLEGMA_Vector<Float> &vectorIn){
  bool flag_eo;
  if( inv_param.matpc_type == QUDA_MATPC_EVEN_EVEN )
    flag_eo = true;
  else if(inv_param.matpc_type == QUDA_MATPC_ODD_ODD )
    flag_eo = false;

  vectorIn.copyToQUDA(b,flag_eo);
  return solve(b);
}

template<typename Float>
void QUDA_solver::solve(PLEGMA_Vector<Float> &vectorOut, PLEGMA_Vector<Float> &vectorIn){
  bool flag_eo;
  if( inv_param.matpc_type == QUDA_MATPC_EVEN_EVEN )
    flag_eo = true;
  else if(inv_param.matpc_type == QUDA_MATPC_ODD_ODD )
    flag_eo = false;

  x = solve(vectorIn); 
  vectorOut.copyFromQUDA( x, flag_eo);
  if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION || 
      inv_param.mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
    vectorOut.scaleVector(2*inv_param.kappa);
  }
}

template void QUDA_solver::solve(PLEGMA_Vector<float> &vectorOut, PLEGMA_Vector<float> &vectorIn);
template void QUDA_solver::solve(PLEGMA_Vector<double> &vectorOut, PLEGMA_Vector<double> &vectorIn);


template cudaColorSpinorField *QUDA_solver::solve(PLEGMA_Vector<float> &vectorIn);
template cudaColorSpinorField *QUDA_solver::solve(PLEGMA_Vector<double> &vectorIn);

//######################### Quda Dirac operator class ################################

PLEGMA_Qdirac::PLEGMA_Qdirac(QudaDslashType dslashType):
  D(nullptr), in(nullptr), out(nullptr){
  if(dslashType != QUDA_WILSON_DSLASH
     && dslashType != QUDA_CLOVER_WILSON_DSLASH
     && dslashType != QUDA_TWISTED_MASS_DSLASH
     && dslashType != QUDA_TWISTED_CLOVER_DSLASH) errorQuda("Error dslashType is not allowed in PLEGMA");

  inv_param = newQudaInvertParam();
  setInvertParam(inv_param);
  inv_param.dslash_type = dslashType; // change to the desired dslash type
  setDiracParam(dParam, &inv_param, false);
  if (dParam.gauge == nullptr) errorQuda("Gauge field not allocated");
  if (dParam. clover == nullptr && ((inv_param.dslash_type == QUDA_CLOVER_WILSON_DSLASH) || (inv_param.dslash_type == QUDA_TWISTED_CLOVER_DSLASH))) errorQuda("Clover field not allocated");
  D = Dirac::create(dParam);

  ColorSpinorParam cpuParam(nullptr, inv_param, GK_localL, false,
			    inv_param.input_location);
  ColorSpinorParam cudaParam(cpuParam, inv_param);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  in = new cudaColorSpinorField(cudaParam);
  out = new cudaColorSpinorField(cudaParam);
  if(in->SiteSubset() != QUDA_FULL_SITE_SUBSET || out->SiteSubset() != QUDA_FULL_SITE_SUBSET)
    errorQuda("cudaColorSpinorField should be a full vector for this class");
}

PLEGMA_Qdirac::~PLEGMA_Qdirac(){
  delete D;
  delete in;
  delete out;
}

template<typename Float>
void PLEGMA_Qdirac::applyM(PLEGMA_Vector<Float> &Pout, PLEGMA_Vector<Float> &Pin){
  Pin.copyToQUDA(in,false);
  D->M(*out,*in);
  Pout.copyFromQUDA(out,false);
}
template void PLEGMA_Qdirac::applyM(PLEGMA_Vector<float> &Pout, PLEGMA_Vector<float> &Pin);
template void PLEGMA_Qdirac::applyM(PLEGMA_Vector<double> &Pout, PLEGMA_Vector<double> &Pin);

template<typename Float>
void PLEGMA_Qdirac::applyMdag(PLEGMA_Vector<Float> &Pout, PLEGMA_Vector<Float> &Pin){
  Pin.copyToQUDA(in,false);
  D->Mdag(*out,*in);
  Pout.copyFromQUDA(out,false);
}
template void PLEGMA_Qdirac::applyMdag(PLEGMA_Vector<float> &Pout, PLEGMA_Vector<float> &Pin);
template void PLEGMA_Qdirac::applyMdag(PLEGMA_Vector<double> &Pout, PLEGMA_Vector<double> &Pin);


template<typename Float>
void PLEGMA_Qdirac::applyMdagM(PLEGMA_Vector<Float> &Pout, PLEGMA_Vector<Float> &Pin){
  Pin.copyToQUDA(in,false);
  D->MdagM(*out,*in);
  Pout.copyFromQUDA(out,false);
}
template void PLEGMA_Qdirac::applyMdagM(PLEGMA_Vector<float> &Pout, PLEGMA_Vector<float> &Pin);
template void PLEGMA_Qdirac::applyMdagM(PLEGMA_Vector<double> &Pout, PLEGMA_Vector<double> &Pin);

template<typename Float>
void PLEGMA_Qdirac::applyMMdag(PLEGMA_Vector<Float> &Pout, PLEGMA_Vector<Float> &Pin){
  Pin.copyToQUDA(in,false);
  D->MMdag(*out,*in);
  Pout.copyFromQUDA(out,false);
}
template void PLEGMA_Qdirac::applyMMdag(PLEGMA_Vector<float> &Pout, PLEGMA_Vector<float> &Pin);
template void PLEGMA_Qdirac::applyMMdag(PLEGMA_Vector<double> &Pout, PLEGMA_Vector<double> &Pin);

void PLEGMA_Qdirac::switch_mu(double mu){
  delete D;
  D=nullptr;
  inv_param.mu = mu;
  setDiracParam(dParam, &inv_param, false);
  D = Dirac::create(dParam);
}

void PLEGMA_Qdirac::switch_kappa(double kappa){
  delete D;
  D=nullptr;
  inv_param.kappa = kappa;
  setDiracParam(dParam, &inv_param, false);
  D = Dirac::create(dParam);
}

