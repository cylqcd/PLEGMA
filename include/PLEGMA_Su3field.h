#include <PLEGMA_Field.h>
#include <vector>

#ifndef _PLEGMA_SU3FIELD_H
#define _PLEGMA_SU3FIELD_H

namespace plegma {
  template<typename Float> class PLEGMA_Gauge;
  ////////////////////////
  // CLASS: PLEGMA_Su3field //
  ////////////////////////
  
  template<typename Float>
    class PLEGMA_Su3field : public PLEGMA_Field<Float> {
  public:
    PLEGMA_Su3field(ALLOCATION_FLAG alloc_flag=BOTH, GHOST_FLAG ghost_flag=FIRST_SIDE);
    ~PLEGMA_Su3field(){;}

    void absorbDir_device(PLEGMA_Gauge<Float> &u,int dir);
    void absorbDir_host(PLEGMA_Gauge<Float> &u,int dir);

    void path(std::vector<int> &steps, PLEGMA_Su3field<Float> **u, PLEGMA_Su3field<Float> &tmp); // this avoids allocation and deallocation
    void path(std::vector<int> &steps, PLEGMA_Su3field<Float> **u);

    void U_plus_eq_aU(PLEGMA_Su3field<Float> &B, Float c);
    void Udag(PLEGMA_Su3field<Float> &x);
    void Udag();
    void UxU(PLEGMA_Su3field<Float> &x, PLEGMA_Su3field<Float> &y);
    void UxUdag(PLEGMA_Su3field<Float> &x, PLEGMA_Su3field<Float> &y);
    void staples(PLEGMA_Su3field<Float> **u, int dir, PLEGMA_Su3field<Float> &tmp1, PLEGMA_Su3field<Float> &tmp2, Float rho, int D3D4);
    void traceHerExpMap(PLEGMA_Su3field<Float> &A);
    void wilsonLineUpdate(PLEGMA_Su3field<Float> &inOut, PLEGMA_Su3field<Float> &tmp, int dirOr, bool reverse=false);
    void su3Projection();
  };
}


#endif
