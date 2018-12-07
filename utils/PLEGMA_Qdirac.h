#include <PLEGMA.h>
#include <dirac_quda.h>

#ifndef _PLEGMA_QDIRAC_H
#define _PLEGMA_QDIRAC_H
using namespace plegma;
enum APP_TYPE {M,Mdag,MdagM,MMdag};
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
    template<APP_TYPE type, typename Float> void apply(PLEGMA_Vector<Float> &out, PLEGMA_Vector<Float> &in, QudaMassNormalization normType = QUDA_KAPPA_NORMALIZATION); // Default is does not put any normalization
  };

}

#endif
