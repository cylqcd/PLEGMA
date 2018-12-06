#include <PLEGMA.h>
#include <dirac_quda.h>

#ifndef _PLEGMA_QDIRAC_H
#define _PLEGMA_QDIRAC_H
using namespace plegma;

namespace quda {
  
  class PLEGMA_Qdirac {
  private:
    DiracParam dParam;
    QudaInvertParam inv_param;
    Dirac *D;
    cudaColorSpinorField *in, *out;
  public:
     //only QUDA_WILSON_DSLASH, QUDA_CLOVER_WILSON_DSLASH, QUDA_TWISTED_MASS_DSLASH, QUDA_TWISTED_CLOVER_DSLASH
    PLEGMA_Qdirac(QudaDslashType dslashType);
    virtual ~PLEGMA_Qdirac();
    void print(){dParam.print();}
    void switch_mu(double mu);
    void switch_kappa(double kappa);
    template<typename Float> void applyM(PLEGMA_Vector<Float> &out, PLEGMA_Vector<Float> &in);
    template<typename Float> void applyMdag(PLEGMA_Vector<Float> &out, PLEGMA_Vector<Float> &in);
    template<typename Float> void applyMdagM(PLEGMA_Vector<Float> &out, PLEGMA_Vector<Float> &in);
    template<typename Float> void applyMMdag(PLEGMA_Vector<Float> &out, PLEGMA_Vector<Float> &in);
  };

}

#endif
