#include <PLEGMA_Field.h>

#ifndef _PLEGMA_FMUNU_H
#define  _PLEGMA_FMUNU_H

namespace plegma {
  // forward declaration
  template<typename Float>  class PLEGMA_Gauge;
  /////////////////////////
  // Class: PLEGMA_Fmunu //
  /////////////////////////
  template<typename Float>
    class PLEGMA_Fmunu : public PLEGMA_Field<Float> {
  public:
    PLEGMA_Fmunu(ALLOCATION_FLAG alloc_flag=BOTH, GHOST_FLAG ghost_flag=FIRST_SIDE);
    ~PLEGMA_Fmunu(){;}

    /**
       @brief Computes the field strength tensor from the clover leaves 
       @param PLEGMA_Gauge<Float> &gauge, The gauge field 
     **/
    void compute_leaves(PLEGMA_Gauge<Float> &gauge);

    int munuToIndx(std::pair<int,int> munu){
      int count=0;
      // Upper triangular with zero diagonal
      for(int m = 0 ; m < N_DIMS ; m++)
	for(int n = m+1; n < N_DIMS; n++){
	  if(std::get<0>(munu) == m && std::get<1>(munu) == n) return count;
	  else count += 1;
	}
      PLEGMA_error("Cannot access elements (%d,%d)\n",std::get<0>(munu),std::get<1>(munu));
      return -1;
    }
    std::pair<int,int> indxToIndx(int indx){
      int count=0;
      // Upper triangular with zero diagonal
      for(int m = 0 ; m < N_DIMS ; m++)
        for(int n = m+1; n < N_DIMS; n++){
          if(indx==count) return std::make_pair(m,n);
          else count += 1;
        }
      PLEGMA_error("Cannot access elements index=%d to mu,nu\n",indx);
      return std::make_pair(-1,-1);
    }
  };
}

#endif
