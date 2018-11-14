#include <PLEGMA_Field.h>
#include <vector>

#ifndef _PLEGMA_SU3MATRIX_H
#define _PLEGMA_SU3MATRIX_H

namespace plegma {
  template<typename Float> class PLEGMA_Gauge;
  ////////////////////////
  // CLASS: PLEGMA_Su3matrix //
  ////////////////////////
  
  template<typename Float>
    class PLEGMA_Su3matrix : public PLEGMA_Field<Float> {
  public:
    PLEGMA_Su3matrix(ALLOCATION_FLAG alloc_flag);
    PLEGMA_Su3matrix(PLEGMA_Gauge<Float> &u, int dir);
    ~PLEGMA_Su3matrix(){;}

    void absorbDir_device(PLEGMA_Gauge<Float> &u,int dir);
    void absorbDir_host(PLEGMA_Gauge<Float> &u,int dir);

    void path(std::vector<int> &steps, PLEGMA_Gauge<Float> &u, PLEGMA_Su3matrix<Float> &tmp); // this avoids allocation and deallocation
    void path(std::vector<int> &steps, PLEGMA_Gauge<Float> &u);

    void Udag(PLEGMA_Su3matrix<Float> &x);
    void UxU(PLEGMA_Su3matrix<Float> &x, PLEGMA_Su3matrix<Float> &y);
    void UxUdag(PLEGMA_Su3matrix<Float> &x, PLEGMA_Su3matrix<Float> &y);
  };
}


#endif
