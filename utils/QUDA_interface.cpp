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
  // TODO: QMP not supported right now
#if defined(QMP_COMMS)
  QMP_thread_level_t tl;
  QMP_init_msg_passing(&argc, &argv, QMP_THREAD_SINGLE, &tl);

  QMP_declare_logical_topology(commDims, 4);
#elif defined(MPI_COMMS)
  MPI_Init(&argc, &argv);
#endif

  initCommsGridQuda(4, commDims,NULL, NULL);
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

void gFixingLandauOVR_QUDA(PLEGMA_Gauge<double> &gaugeOut,PLEGMA_Gauge<double> &gaugeIn, double overelaxPar,double tolerance,
			   int maxiter, int verbosePerSteps, int reunit_interval, int stop_theta){
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setGaugeParam(gauge_param);
  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.make_resident_gauge = 0;
  gaugeIn.unload();
  double* buf[N_DIMS];
  for(int i=0; i<N_DIMS; i++) hostMalloc(buf[i], gaugeIn.Bytes_total()/N_DIMS);
  unpackGaugeToEvenOdd(buf, gaugeIn);
  computeGaugeFixingOVRQuda(buf,4,maxiter,verbosePerSteps,overelaxPar,tolerance,reunit_interval,stop_theta,&gauge_param,nullptr);
  packGaugeToNormal(gaugeOut,buf);
  gaugeOut.load();
  for(int i=0; i<N_DIMS; i++) hostFree(buf[i], gaugeIn.Bytes_total()/N_DIMS);
  PLEGMA_printf("Landau Gauge Fixed plaquette is: ");
  gaugeOut.calculatePlaq();
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
  profiler = new TimeProfile(("Solver profiler mu="+to_string(mu)).c_str());
  profiler->TPSTART(QUDA_PROFILE_TOTAL);

  if(use_mg){
    mg_inv_param = newQudaInvertParam();
    mg_param = newQudaMultigridParam();
    mg_param.invert_param = &mg_inv_param;
    setMultigridParam(mg_param);
    checkMultigridParam(&mg_param);
    if(HGC_verbosity > 2) printQudaMultigridParam(&mg_param);
    mg_param.invert_param->mu = mu;

#ifdef QUDA_INCLUDES_COMMIT_775a033
    mg_eig_param = new QudaEigParam[mg_param.n_level];
    setEigMultigridParam(mg_param,mg_eig_param);
#endif
    mg_preconditioner = newMultigridQuda(&mg_param);
  }
  
  inv_param = newQudaInvertParam();

  if(use_mg) inv_param.preconditioner = mg_preconditioner;

  setInvertParam(inv_param);
  checkInvertParam(&inv_param);
  if(HGC_verbosity > 2) {
    printQudaInvertParam(&inv_param);
  }
  if(inv_param.gamma_basis != QUDA_UKQCD_GAMMA_BASIS) 
    PLEGMA_error("initSolver: This function works only with ukqcd gamma basis\n");
  if(inv_param.dirac_order != QUDA_DIRAC_ORDER) 
    PLEGMA_error("initSolver: This function works only with colors inside the spins\n");

  inv_param.mu = mu;
    
  bool pc_solution = false;
  bool pc_solve = true;

  inv_param.secs = 0;
  inv_param.gflops = 0;
  inv_param.iter = 0;

  D = NULL;
  DSloppy = NULL;
  DPre = NULL;

  // create the dirac operator
  createDirac(D, DSloppy, DPre, inv_param, pc_solve);

  // Create Operators
  M = (inv_param.inv_type == QUDA_CG_INVERTER || inv_param.inv_type ==  QUDA_CA_CG_INVERTER) ? static_cast<DiracMatrix*>(new DiracMdagM(*D)) : static_cast<DiracMatrix*>(new DiracM(*D));
  MSloppy = (inv_param.inv_type == QUDA_CG_INVERTER || inv_param.inv_type ==  QUDA_CA_CG_INVERTER) ? static_cast<DiracMatrix*>(new DiracMdagM(*DSloppy)) : static_cast<DiracMatrix*>(new DiracM(*DSloppy));
  MPre = (inv_param.inv_type == QUDA_CG_INVERTER || inv_param.inv_type ==  QUDA_CA_CG_INVERTER) ? static_cast<DiracMatrix*>(new DiracMdagM(*DPre)) : static_cast<DiracMatrix*>(new DiracM(*DPre));

  // Create Solvers
  solverParam = new SolverParam(inv_param);
  solver = Solver::create(*solverParam, *M, *MSloppy, 
			  *MPre, *profiler);

  ColorSpinorParam cpuParam(NULL, inv_param, HGC_localL, pc_solution,
			    inv_param.input_location);
  ColorSpinorParam cudaParam(cpuParam, inv_param);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  b = new cudaColorSpinorField(cudaParam);
  x = new cudaColorSpinorField(cudaParam);
  profiler->TPSTOP(QUDA_PROFILE_TOTAL);
  profiler->Print();
  profiler->TPRESET();
}

QUDA_solver::~QUDA_solver(){
  if(use_mg){
  destroyMultigridQuda(mg_preconditioner);
#ifdef QUDA_INCLUDES_COMMIT_775a033
  delete mg_eig_param;
#endif
  }
  delete solver;
  delete solverParam;
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

struct MG_Transfer{
  typedef Transfer* MG::*type;
  friend type get(MG_Transfer);
};

struct MG_CoarseParam{
  typedef MGParam* MG::*type;
  friend type get(MG_CoarseParam);
};

struct MG_Coarse{
  typedef MG* MG::*type;
  friend type get(MG_Coarse);
};

template<typename Tag,typename Tag::type M>
struct Rob {
  friend typename Tag::type get(Tag){ return M;}
};
  
template struct Rob<MG_Transfer,&MG::transfer>;
template struct Rob<MG_CoarseParam,&MG::param_coarse>;
template struct Rob<MG_Coarse,&MG::coarse>;

inline bool changeBlock(int* blockOut, int* blockIn) {
  bool changed = false;
  for (int i=0; i<QUDA_MAX_DIM; i++) {
    if(blockIn[i]!=blockOut[i]) {
      changed=true;
      blockOut[i]=blockIn[i];
    }
  }
  return changed;
}

static void updateMultigridParam(MG* mg, MGParam* current, QudaMultigridParam* param, int level = 0)
{
  current->nu_pre = param->nu_pre[level];
  current->nu_post = param->nu_post[level];
  current->smoother_tol = param->smoother_tol[level];
  current->cycle_type = param->cycle_type[level];
  current->global_reduction = param->global_reduction[level];
  current->omega = param->omega[level];
  current->smoother = param->smoother[level];
  
  if(level < mg_levels-1 && level < QUDA_MAX_MG_LEVEL-1){
    MG* &coarse = mg->*get(MG_Coarse());
    MGParam* &coarseParam = mg->*get(MG_CoarseParam());
    if(changeBlock(coarseParam->geoBlockSize, param->geo_block_size[level+1])) {
      delete coarse;
      coarse=nullptr;
      delete coarseParam;
      coarseParam=nullptr;
      delete (mg->*get(MG_Transfer()));
      mg->*get(MG_Transfer())=nullptr;
      return;
    }
    if(coarseParam->Nvec != param->n_vec[level+1]) {
      delete coarse;
      coarse=nullptr;
      delete coarseParam;
      coarseParam=nullptr;
      return;
    }
    updateMultigridParam(coarse, coarseParam, param, level+1);
  }
}

void QUDA_solver::UpdateSolver()
{
  profiler->TPSTART(QUDA_PROFILE_TOTAL);
  delete solver;
  delete solverParam;
  delete M;
  delete MSloppy;
  delete MPre;
  delete D; D = NULL;
  delete DSloppy; DSloppy = NULL;
  delete DPre; DPre = NULL;

  if(use_mg){
  PLEGMA_printf("Updating multigrid parameters\n");
  setMultigridParam(mg_param);}

  setInvertParam(inv_param);
  checkInvertParam(&inv_param);

  if(use_mg){
  multigrid_solver* mg = (multigrid_solver*) mg_preconditioner;
  if(changeBlock(mg->mgParam->geoBlockSize, mg_param.geo_block_size[0]) ||
     mg->mgParam->Nvec != mg_param.n_vec[0]) {
    destroyMultigridQuda(mg_preconditioner);
    mg_preconditioner = newMultigridQuda(&mg_param);
  } else {
    updateMultigridParam(mg->mg, mg->mgParam, &mg_param);
    updateMultigridQuda(mg_preconditioner, &mg_param);
  }}
  
  bool pc_solve = true;
  createDirac(D, DSloppy, DPre, inv_param, pc_solve);

  // Create Operators
  M = (inv_param.inv_type == QUDA_CG_INVERTER || inv_param.inv_type ==  QUDA_CA_CG_INVERTER) ? static_cast<DiracMatrix*>(new DiracMdagM(*D)) : static_cast<DiracMatrix*>(new DiracM(*D));
  MSloppy = (inv_param.inv_type == QUDA_CG_INVERTER || inv_param.inv_type ==  QUDA_CA_CG_INVERTER) ? static_cast<DiracMatrix*>(new DiracMdagM(*DSloppy)) : static_cast<DiracMatrix*>(new DiracM(*DSloppy));
  MPre = (inv_param.inv_type == QUDA_CG_INVERTER || inv_param.inv_type ==  QUDA_CA_CG_INVERTER) ? static_cast<DiracMatrix*>(new DiracMdagM(*DPre)) : static_cast<DiracMatrix*>(new DiracM(*DPre));

  
  // Create Solvers
  solverParam = new SolverParam(inv_param);
  
  solver = Solver::create(*solverParam, *M, *MSloppy, 
  			 *MPre, *profiler);

  profiler->TPSTOP(QUDA_PROFILE_TOTAL);
  profiler->Print();
  profiler->TPRESET();
}

cudaColorSpinorField *QUDA_solver::solve(cudaColorSpinorField * rhs){
  profiler->TPSTART(QUDA_PROFILE_TOTAL);
  ColorSpinorField *in = NULL;
  ColorSpinorField *out = NULL;
  D->prepare(in,out,*x,*rhs,inv_param.solution_type);
  (*solver)(*out, *in);
  D->reconstruct(*x,*rhs,inv_param.solution_type);
  profiler->TPSTOP(QUDA_PROFILE_TOTAL);
  profiler->Print();
  profiler->TPRESET();
  return x;
}


template<typename Float>
cudaColorSpinorField *QUDA_solver::solve(PLEGMA_Vector<Float> &vectorIn){
  bool flag_eo=false;
  if( inv_param.matpc_type == QUDA_MATPC_EVEN_EVEN )
    flag_eo = true;

  vectorIn.copyToQUDA(b,flag_eo);
  return solve(b);
}

template<typename Float>
void QUDA_solver::solve(PLEGMA_Vector<Float> &vectorOut, PLEGMA_Vector<Float> &vectorIn){
  bool flag_eo = false;
  if( inv_param.matpc_type == QUDA_MATPC_EVEN_EVEN )
    flag_eo = true;

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

template<typename Float>
void QUDA_solver::runOneIter(PLEGMA_Vector<Float> &vectorOut, PLEGMA_Vector<Float> &vectorIn){
  int maxiter = solverParam->maxiter;
  solverParam->maxiter = 1;
  solve(vectorOut, vectorIn);
  solverParam->maxiter = maxiter;
}

template void QUDA_solver::runOneIter(PLEGMA_Vector<float> &vectorOut, PLEGMA_Vector<float> &vectorIn);
template void QUDA_solver::runOneIter(PLEGMA_Vector<double> &vectorOut, PLEGMA_Vector<double> &vectorIn);


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

