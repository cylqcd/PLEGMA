#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <invert_quda.h>
#include <PLEGMA_BLAS.h>
#include <multigrid.h>
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

void initGaugeQuda(PLEGMA_Gauge<double> &gauge, bool antiperiodic, QudaLinkType type) {
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setGaugeParam(gauge_param);
  gauge_param.type = type;

  double* buf[N_DIMS];
  for(int i=0; i<N_DIMS; i++) hostMalloc(buf[i], gauge.Bytes_total()/N_DIMS);

  gauge.unload();
  unpackGaugeToEvenOdd(buf, gauge);
  if(antiperiodic) {
    applyAntiperiodicBoundary(buf);
    gauge_param.t_boundary = QUDA_ANTI_PERIODIC_T;
  }
  else {
    gauge_param.t_boundary = QUDA_PERIODIC_T;
  }
  loadGaugeQuda(buf, &gauge_param);

  if (type == QUDA_WILSON_LINKS && dslash_type == QUDA_TWISTED_CLOVER_DSLASH)  {
    QudaInvertParam inv_param = newQudaInvertParam();
    setInvertParam(inv_param);
    checkInvertParam(&inv_param);

    inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;
    loadCloverQuda(NULL, NULL, &inv_param);
  }
  for(int i=0; i<N_DIMS; i++) hostFree(buf[i], gauge.Bytes_total()/N_DIMS);
}

void finalizeGaugeQuda() {
  freeGaugeQuda();
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || 
      dslash_type == QUDA_TWISTED_CLOVER_DSLASH) freeCloverQuda();
}

void updateGaugeQuda(PLEGMA_Gauge<double> &gauge, bool antiperiodic, QudaLinkType type) {
  finalizeGaugeQuda();
  initGaugeQuda(gauge,antiperiodic,type);
}

void plaqQuda() {
  double plq[3]; // total, spatial and temporal plaquette
  plaqQuda(plq);
  PLEGMA_printf("TEST: Calculated plaquette in QUDA: %f (sp: %f, T: %f)\n", plq[0], plq[1], plq[2]);
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
    PLEGMA_error("initSolver: This function works only with Direct solve and even odd preconditioning");
  
  if(inv_param.inv_type != QUDA_GCR_INVERTER) 
    PLEGMA_error("initSolver: This function works only with GCR method");

  if(inv_param.gamma_basis != QUDA_UKQCD_GAMMA_BASIS) 
    PLEGMA_error("initSolver: This function works only with ukqcd gamma basis\n");
  if(inv_param.dirac_order != QUDA_DIRAC_ORDER) 
    PLEGMA_error("initSolver: This function works only with colors inside the spins\n");

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

  ColorSpinorParam cpuParam(NULL, inv_param, HGC_localL, pc_solution,
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

struct MG_MGParam{
  typedef MGParam* MG::*type;
  friend type get(MG_MGParam);
};

struct MG_Coarse{
  typedef MG* MG::*type;
  friend type get(MG_Coarse);
};

template<typename Tag,typename Tag::type M>
struct Rob {
  friend typename Tag::type get(Tag){ return M;}
};
  
template struct Rob<MG_MGParam,&MG::param_coarse>;
template struct Rob<MG_Coarse,&MG::coarse>;

static void updateMultigridParam(MG* mg, MGParam &current, QudaMultigridParam param, int level = 0)
{
  current.nu_pre = param.nu_pre[level];
  current.nu_post = param.nu_post[level];
  current.smoother_tol = param.smoother_tol[level];
  current.cycle_type = param.cycle_type[level];
  current.global_reduction = param.global_reduction[level];
  current.omega = param.omega[level];
  current.smoother = param.smoother[level];
  current.mg_global.mu_factor[level] = param.mu_factor[level];
  
  if(level < mg_levels-1){
    current.mg_global.mu_factor[level+1] = param.mu_factor[level+1];
    updateMultigridParam(mg->*get(MG_Coarse()),*(mg->*get(MG_MGParam())),param,level+1);} 
}

void QUDA_solver::UpdateSolver()
{
  PLEGMA_printf("Updating multigrid parameters\n");
  setMultigridParam(mg_param);
 
  setInvertParam(inv_param);
  checkInvertParam(&inv_param);

  updateMultigridParam(((multigrid_solver*) mg_preconditioner)->mg,*(((multigrid_solver*) mg_preconditioner)->mgParam),mg_param);
  updateMultigridQuda(mg_preconditioner,&mg_param);
 

  bool pc_solution = false;
  bool pc_solve = true;

  delete D;
  delete DSloppy;
  delete DPre;

  D = NULL;
  DSloppy = NULL;
  DPre = NULL;
  
  createDirac(D, DSloppy, DPre, inv_param, pc_solve);

  delete solver;
  delete M;
  delete MSloppy;
  delete MPre;

  solver = NULL;
  M = NULL;
  MSloppy = NULL;
  MPre = NULL;
  
  // Create Operators
  M = new DiracM(*D);
  MSloppy = new DiracM(*DSloppy);
  MPre = new DiracM(*DPre);

  // Create Solvers
  delete solverParam;
  solverParam = NULL;
  solverParam = new SolverParam(inv_param);
  
  solver = Solver::create(*solverParam, *M, *MSloppy, 
  			 *MPre, *profiler);

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

QUDA_dirac::QUDA_dirac(QudaDslashType dslashType):
  D(nullptr), in(nullptr), out(nullptr){
  if(dslashType != QUDA_WILSON_DSLASH
     && dslashType != QUDA_CLOVER_WILSON_DSLASH
     && dslashType != QUDA_TWISTED_MASS_DSLASH
     && dslashType != QUDA_TWISTED_CLOVER_DSLASH) PLEGMA_error("Error dslashType is not allowed in PLEGMA");

  inv_param = newQudaInvertParam();
  setInvertParam(inv_param);
  inv_param.dslash_type = dslashType; // change to the desired dslash type
  setDiracParam(dParam, &inv_param, false);
  if (dParam.gauge == nullptr) PLEGMA_error("Gauge field not allocated");
  if (dParam. clover == nullptr && ((inv_param.dslash_type == QUDA_CLOVER_WILSON_DSLASH) || (inv_param.dslash_type == QUDA_TWISTED_CLOVER_DSLASH))) PLEGMA_error("Clover field not allocated");
  D = Dirac::create(dParam);

  ColorSpinorParam cpuParam(nullptr, inv_param, HGC_localL, false,
			    inv_param.input_location);
  ColorSpinorParam cudaParam(cpuParam, inv_param);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  in = new cudaColorSpinorField(cudaParam);
  out = new cudaColorSpinorField(cudaParam);
  if(in->SiteSubset() != QUDA_FULL_SITE_SUBSET || out->SiteSubset() != QUDA_FULL_SITE_SUBSET)
    PLEGMA_error("cudaColorSpinorField should be a full vector for this class");
}

QUDA_dirac::~QUDA_dirac(){
  delete D;
  delete in;
  delete out;
}

template<APP_TYPE type> void QUDA_dirac::apply(){
  switch (type){
  case(M): D->M(*out,*in); break;
  case(Mdag): D->Mdag(*out,*in); break;
  case(MdagM): D->MdagM(*out,*in);  break;
  case(MMdag): D->MMdag(*out,*in); break;
  }
}

template<APP_TYPE type,typename Float>
void QUDA_dirac::apply(PLEGMA_Vector<Float> &Pout, PLEGMA_Vector<Float> &Pin, QudaMassNormalization normType){
  Pin.copyToQUDA(in);
  apply<type>();
  Pout.copyFromQUDA(out);
  if (normType == QUDA_MASS_NORMALIZATION || normType == QUDA_ASYMMETRIC_MASS_NORMALIZATION) Pout.scaleVector(1./(2*inv_param.kappa));
}

template void QUDA_dirac::apply<M>(PLEGMA_Vector<float> &Pout, PLEGMA_Vector<float> &Pin, QudaMassNormalization normType);
template void QUDA_dirac::apply<M>(PLEGMA_Vector<double> &Pout, PLEGMA_Vector<double> &Pin, QudaMassNormalization normType);
template void QUDA_dirac::apply<Mdag>(PLEGMA_Vector<float> &Pout, PLEGMA_Vector<float> &Pin, QudaMassNormalization normType);
template void QUDA_dirac::apply<Mdag>(PLEGMA_Vector<double> &Pout, PLEGMA_Vector<double> &Pin, QudaMassNormalization normType);
template void QUDA_dirac::apply<MdagM>(PLEGMA_Vector<float> &Pout, PLEGMA_Vector<float> &Pin, QudaMassNormalization normType);
template void QUDA_dirac::apply<MdagM>(PLEGMA_Vector<double> &Pout, PLEGMA_Vector<double> &Pin, QudaMassNormalization normType);
template void QUDA_dirac::apply<MMdag>(PLEGMA_Vector<float> &Pout, PLEGMA_Vector<float> &Pin, QudaMassNormalization normType);
template void QUDA_dirac::apply<MMdag>(PLEGMA_Vector<double> &Pout, PLEGMA_Vector<double> &Pin, QudaMassNormalization normType);

template<APP_TYPE type,typename Float>
void QUDA_dirac::apply(Float *dout, Float *din, QudaMassNormalization normType){
  plegma::copyToQUDA(in,din);
  apply<type>();
  plegma::copyFromQUDA(dout,out);
  if (normType == QUDA_MASS_NORMALIZATION || normType == QUDA_ASYMMETRIC_MASS_NORMALIZATION)
    cuBLAS::scal<Float>(N_SPINS*N_COLS*HGC_localVolume, (Float) (1./(2*inv_param.kappa)), dout);
}

template void QUDA_dirac::apply<M>(float *dout, float *din, QudaMassNormalization normType);
template void QUDA_dirac::apply<M>(double *dout, double *din, QudaMassNormalization normType);
template void QUDA_dirac::apply<Mdag>(float *dout, float *din, QudaMassNormalization normType);
template void QUDA_dirac::apply<Mdag>(double *dout, double *din, QudaMassNormalization normType);
template void QUDA_dirac::apply<MdagM>(float *dout, float *din, QudaMassNormalization normType);
template void QUDA_dirac::apply<MdagM>(double *dout, double *din, QudaMassNormalization normType);
template void QUDA_dirac::apply<MMdag>(float *dout, float *din, QudaMassNormalization normType);
template void QUDA_dirac::apply<MMdag>(double *dout, double *din, QudaMassNormalization normType);


void QUDA_dirac::switchMu(double mu){
  delete D;
  D=nullptr;
  inv_param.mu = mu;
  setDiracParam(dParam, &inv_param, false);
  D = Dirac::create(dParam);
}

void QUDA_dirac::switchKappa(double kappa){
  delete D;
  D=nullptr;
  inv_param.kappa = kappa;
  setDiracParam(dParam, &inv_param, false);
  D = Dirac::create(dParam);
}

