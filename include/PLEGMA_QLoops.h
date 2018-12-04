#include <PLEGMA_Field.h>
#include <PLEGMA_Vector.h>
#include <string>

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
  public:
    PLEGMA_QLoops(ALLOCATION_FLAG alloc_flag, bool isOneD=false);
    ~PLEGMA_QLoops();

    void oneEnd_trick(PLEGMA_Vector<Float> &x_l, PLEGMA_Vector<Float> &x_r,
		      Float val , bool accum );
    void oneEnd_trick(PLEGMA_Vector<Float> &x_l, PLEGMA_Vector<Float> &x_r,
		      PLEGMA_Vector<Float> &v_covD, Float val , bool accum );
    void contractG5(PLEGMA_Vector<Float> &x_l, PLEGMA_Vector<Float> &x_r, ACCUM_TYPE acc_type);
    void contractG5(PLEGMA_Vector<Float> &x_l, PLEGMA_Vector<Float> &x_r);

    void write_ASCII(std::string filename_local);
    void write_ASCII(std::string filename_local, std::string filename_oneD, std::string filename_oneDC);
    
  };

}
#endif
