#include <PLEGMA.h>
#include <invert_quda.h>
#include <dirac_quda.h>

#ifndef _QUDA_INTERFACE_H
#define _QUDA_INTERFACE_H
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
    DiracM *M, *MSloppy, *MPre;
    cudaColorSpinorField *b, *x;
    
  public:
    QudaInvertParam getInvParams() const{return inv_param;}
    QUDA_solver(double mu);
    virtual ~QUDA_solver();
    cudaColorSpinorField* solve(cudaColorSpinorField * rhs);
    template<typename Float>
    cudaColorSpinorField* solve(PLEGMA_Vector<Float> &vectorIn);
    template<typename Float>
    void solve(PLEGMA_Vector<Float> &out, PLEGMA_Vector<Float> &in);
  };

  enum APP_TYPE {M,Mdag,MdagM,MMdag};
  class QUDA_dirac {
  private:
    DiracParam dParam;
    QudaInvertParam inv_param;
    Dirac *D;
    cudaColorSpinorField *in, *out;
  public:
     //only QUDA_WILSON_DSLASH, QUDA_CLOVER_WILSON_DSLASH, QUDA_TWISTED_MASS_DSLASH, QUDA_TWISTED_CLOVER_DSLASH
    QUDA_dirac(QudaDslashType dslashType);
    virtual ~QUDA_dirac();
    void print(){dParam.print();}
    void switchMu(double mu);
    void switchKappa(double kappa);
    template<APP_TYPE type, typename Float> void apply(PLEGMA_Vector<Float> &out, PLEGMA_Vector<Float> &in, QudaMassNormalization normType = QUDA_KAPPA_NORMALIZATION); // Default is does not put any normalization
  };
}
#endif
