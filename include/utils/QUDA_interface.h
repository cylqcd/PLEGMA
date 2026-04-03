#pragma once
#include <PLEGMA.h>
#include <invert_quda.h>
#include <dirac_quda.h>
using namespace plegma;

namespace quda {  
  class QUDA_solver {

  protected:
    TimeProfile *profiler;
    Solver *solver;
    SolverParam *solverParam;
    void *mg_preconditioner;
    QudaInvertParam inv_param;
    QudaInvertParam mg_inv_param;
    QudaMultigridParam mg_param;
    Dirac *D, *DSloppy, *DPre;
    DiracMatrix *M, *MSloppy, *MPre;
    ColorSpinorField *b, *x;
#ifdef QUDA_INCLUDES_COMMIT_775a033
    QudaEigParam *mg_eig_param;
#endif

  public:
    QudaInvertParam getInvParams() const{return inv_param;}
    SolverParam* getSolverParam() const{return solverParam;}
    void UpdateSolver();
    QUDA_solver(double mu);
    virtual ~QUDA_solver();
    ColorSpinorField* solve(ColorSpinorField * rhs);
    template<typename Float>
    ColorSpinorField* solve(PLEGMA_Vector<Float> &vectorIn);
    template<typename Float>
    void solve(PLEGMA_Vector<Float> &out, PLEGMA_Vector<Float> &in);
    template<typename Float>
    void runOneIter(PLEGMA_Vector<Float> &out, PLEGMA_Vector<Float> &in);
  };

  enum APP_TYPE {M,Mdag,MdagM,MMdag};
  class QUDA_dirac {
  private:
    DiracParam dParam;
    QudaInvertParam inv_param;
    Dirac *D;
    ColorSpinorField *in, *out;
    template<APP_TYPE type> void apply();
  public:
     //only QUDA_WILSON_DSLASH, QUDA_CLOVER_WILSON_DSLASH, QUDA_TWISTED_MASS_DSLASH, QUDA_TWISTED_CLOVER_DSLASH
    QUDA_dirac(QudaDslashType dslashType);
    virtual ~QUDA_dirac();
    void print(){dParam.print();}
    void switchMu(double mu);
    void switchKappa(double kappa);
    template<APP_TYPE type, typename Float> void apply(PLEGMA_Vector<Float> &out, PLEGMA_Vector<Float> &in, QudaMassNormalization normType = QUDA_KAPPA_NORMALIZATION); // Default it is without any normalization
    template<APP_TYPE type, typename Float> void apply(Float *dout, Float *din, QudaMassNormalization normType = QUDA_KAPPA_NORMALIZATION); // Default is without any normalization // Note that dout and din are device pointers
  };
}

