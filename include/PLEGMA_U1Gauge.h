#include <PLEGMA_Field.h>

#ifndef _PLEGMA_U1GAUGE_H
#define  _PLEGMA_U1GAUGE_H

namespace plegma {
  // forward declaration

  //////////////////////////
  // Class: PLEGMA_U1Gauge//
  //////////////////////////
  template<typename Float>
    class PLEGMA_U1Gauge : public PLEGMA_Field<Float> {
  public:
    PLEGMA_U1Gauge(ALLOCATION_FLAG alloc_flag=BOTH, GHOST_FLAG ghost_flag=FIRST_SIDE);
    ~PLEGMA_U1Gauge(){;}
    /**
       @brief Computes the plaquette with U1 gauge fields
     **/
    Float calculatePlaq();
    /**
       @brief Creates a constant field 
       @param int mu: The direction of the potential
       @param int nu: Given mu which direction gives the linear increase to the potential 
     **/
    void constField(int mu, int nu, Float exparg, int xnu_0=0);
    /**
       @brief This function ensures that every plq has the same value. 
       @param int mu: Transverse direction to the modification
       @param int nu: Transverse direction to the modification, if nu < 0 then put zeros
     **/
    void modifyBoundaries(int mu, int nu, Float exparg);
  };
  
}

#endif
