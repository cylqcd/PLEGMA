#include <PLEGMA_Field.h>
#include <dirac_quda.h>
#ifndef _PLEGMA_QLOOPS
#define _PLEGMA_QLOOPS

namespace plegma{

  //enum LOOP_TYPE{S_ULTRALOCAL, G_ULTRALOCAL, S_ONED, G_ONED, S_ONEDC, G_ONEDC, N_LOOPS};
  //const int nCompL[N_LOOPS] ={N_SPINS*N_SPINS, N_SPINS*N_SPINS, N_DIMS*N_SPINS*N_SPINS,
  //			      N_DIMS*N_SPINS*N_SPINS, N_DIMS*N_SPINS*N_SPINS, N_DIMS*N_SPINS*N_SPINS};
    
  template<typename Float>
    class PLEGMA_QLoops : public PLEGMA_Field<Float> {
  private:
    Float *h_loc;
    Float *h_oneD[N_DIMS];
    Float *h_oneDC[N_DIMS];

    bool isOneD;
    quda::GaugeCovDev *cov;
  public:
    PLEGMA_QLoops(ALLOCATION_FLAG alloc_flag, quda::GaugeCovDev *cov = NULL);
    ~PLEGMA_QLoops();
    
    void oneEnd_trick(quda::cudaColorSpinorField &x_l, quda::cudaColorSpinorField &x_r,
		      quda::cudaColorSpinorField &tmp, Float val = 1, bool accum = true);
  };

}
#endif
