#include <PLEGMA_Field.h>
#include <color_spinor_field.h>

#ifndef _PLEGMA_VECTOR_H
#define _PLEGMA_VECTOR_H

namespace plegma {

  // forward declaration
  template<typename Float>  class PLEGMA_Gauge;
  template<typename Float>  class PLEGMA_Propagator;
  template<typename Float>  class PLEGMA_Propagator3D;

  /////////////////////////
  // Class: PLEGMA_Vector //
  /////////////////////////
  
  template<typename Float>
    class PLEGMA_Vector : public PLEGMA_Field<Float> {
  public:
    PLEGMA_Vector(ALLOCATION_FLAG alloc_flag=BOTH, GHOST_FLAG ghost_flag=FIRST_SIDE);
    ~PLEGMA_Vector(){;}
    
    void copyToQUDA( quda::ColorSpinorField *cudaVector, bool isEv = false);
    void copyFromQUDA( quda::ColorSpinorField *cudaVector, bool isEv = false);
    void gaussianSmearing(PLEGMA_Vector<Float> &vecIn, PLEGMA_Gauge<Float> &gaugeAPE);
    void scaleVector(Float a);
    void copy(PLEGMA_Vector<float> &vecIn);
    void copy(PLEGMA_Vector<double> &vecIn);
    void norm2Host();
    void norm2Device();
    /**
       @brief Absorbs elements nu, c2 from a 4D propagator at specific global time and puts it in a 4D vector
       @param PLEGMA_Propagator<Float> prop, The 4D propagator
       @param int global_it, The global time slice which we want to extract, the rest of the time-slices will become zero in the output
       @param int nu, The spin index we want to extract
       @param int c2, The color index we want to extract
       @return void
     **/
    void absorb(PLEGMA_Propagator<Float> &prop, int global_it, int nu , int c2);
    
    /**
       @brief Absorbs elements nu, c2 from a 3D propagator and puts it at a specific global time of the 4D vector
       @param PLEGMA_Propagator3D<Float> prop, The 3D propagator
       @param int global_it, The global time slice where data will be inserted, the rest of the time-slices will become zero in the 4D vector
       @param int nu, The spin index we want to extract
       @param int c2, The color index we want to extract
       @return void
     **/    
    void absorb(PLEGMA_Propagator3D<Float> &prop, int global_it, int nu , int c2);

    /**
       @brief Absorbs elements nu, c2 from a 4D propagator to a 4D vector
       @param PLEGMA_Propagator<Float> prop, The 4D propagator
       @param int nu, The spin index we want to extract
       @param int c2, The color index we want to extract
       @return void
     **/    
    void absorb(PLEGMA_Propagator<Float> &prop, int nu , int c2);
    
    void pointSource(int *sourceposition, 
        int spin, int color, ALLOCATION_FLAG alloc_flag);
    void pointSource(int *sourceposition, int spin, int color);
    void write(char* filename);
    void conjugate();
    void apply_gamma5();
    void covD(PLEGMA_Vector<Float> &vecIn, PLEGMA_Gauge<Float> &gauge, int dirOr);
  };

  template<typename Float> void copyToQUDA(quda::ColorSpinorField *cudaVector, Float* delem, bool isEv = false); // delem is a device pointer
  template<typename Float> void copyFromQUDA(Float* delem, quda::ColorSpinorField *cudaVector, bool isEv = false);

  /////////////////////////////////////
  // CLASS: PLEGMA_Vector3D ///////////
  ////////////////////////////////////
  
  template<typename Float>
    class PLEGMA_Vector3D : public PLEGMA_Field<Float> {
  public:
    PLEGMA_Vector3D(ALLOCATION_FLAG alloc_flag=BOTH, GHOST_FLAG ghost_flag=NO_GHOSTS);
    ~PLEGMA_Vector3D(){;}

    /**
       @brief Absorbs elements nu, c2 from a 3D propagator to a 3D vector
       @param PLEGMA_Propagator3D<Float> prop, The 3D propagator
       @param int nu, The spin index we want to extract
       @param int c2, The color index we want to extract
       @return void
     **/    
    void absorb(PLEGMA_Propagator3D<Float> &prop, int nu, int c2);

    /**
       @brief Absorbs elements nu, c2 from a 4D propagator at specific global time and puts it in a 3D vector
       @param PLEGMA_Propagator<Float> prop, The 4D propagator
       @param int global_it, The global time slice which we want to extract
       @param int nu, The spin index we want to extract
       @param int c2, The color index we want to extract
       @return void
     **/
    void absorb(PLEGMA_Propagator<Float> &prop, int global_it, int nu, int c2);
  };
}

#endif
