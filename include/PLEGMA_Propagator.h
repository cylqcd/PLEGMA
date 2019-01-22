#include <PLEGMA_Field.h>

#ifndef _PLEGMA_PROPAGATOR_H
#define _PLEGMA_PROPAGATOR_H

namespace plegma {
  // forward declaration
  template<typename Float>  class PLEGMA_Vector;
  template<typename Float>  class PLEGMA_Vector3D;
  template<typename Float>  class PLEGMA_Propagator3D;
  /////////////////////////////
  // CLASS: PLEGMA_Propagator //
  /////////////////////////////
  template<typename Float>
    class PLEGMA_Propagator : public PLEGMA_Field<Float> {
    
  public:
    PLEGMA_Propagator(ALLOCATION_FLAG alloc_flag=BOTH, GHOST_FLAG ghost_flag=FIRST_SIDE);
    ~PLEGMA_Propagator(){;}
    
    void conjugate();
    void apply_gamma(GAMMAS gMat, LEFTRIGHT LR = LEFT);
    void apply_gamma5();
    void absorbVectorToHost(PLEGMA_Vector<Float> &vec, 
			    int nu, int c2);
    /**
       @brief Absorbs elements from a 4D vector to a 4D propagator at nu, c2
       @param PLEGMA_Vector<Float> vec, The 4D vector
       @param int nu, The spin index where to insert data 
       @param int c2, The color index where to insert data
       @return void
     **/    
    void absorb(PLEGMA_Vector<Float> &vec, int nu, int c2);

    /**
       @brief Absorbs elements from a 4D vector to a 4D propagator at nu, c2 but at specific global time-slice 
       @param PLEGMA_Vector<Float> vec, The 4D vector
       @param int global_it, The global time slice where data will be inserted, the rest of the time-slices will become zero in the 4D propagator
       @param int nu, The spin index where to insert data 
       @param int c2, The color index where to insert data
       @return void
     **/    
    void absorb(PLEGMA_Vector<Float> &vec, int global_it, int nu, int c2);

    /**
       @brief Absorbs all  elements from a 3D propagator and puts it at a specific global time of the 4D propagator
       @param PLEGMA_Propagator3D<Float> prop, The 3D propagator
       @param int global_it, The global time slice where data which will be inserted, the rest of the time-slices will become zero in the 4D propagator
       @return void
     **/    
    void absorb(PLEGMA_Propagator3D<Float> &prop, int global_it);

    /**
       @brief Absorbs a 3D vector to a 4D Propagator at a specific global time and nu, c2 element
       @param PLEGMA_Vector3D<Float> vec, The 3D vector
       @param int global_it, The global time slice where data which will be inserted, the rest of the time-slices will become zero in the 4D propagator
       @param int nu, The spin index where to insert data 
       @param int c2, The color index where to insert data       
       @return void
     **/    
    void absorb(PLEGMA_Vector3D<Float> &vec, int global_it, int nu, int c2);
    
    void applyBoundaries_device(int t0);
    void rotateToPhysicalBase_host(int sign);
    void rotateToPhysicalBase_device(int sign);
  };

  ///////////////////////////////
  // CLASS: PLEGMA_Propagator3D //
  /////////////////////////////// 
  
  template<typename Float>
    class PLEGMA_Propagator3D : public PLEGMA_Field<Float> {
    
  public:
    PLEGMA_Propagator3D(ALLOCATION_FLAG alloc_flag=BOTH, GHOST_FLAG ghost_flag=NO_GHOSTS);
    ~PLEGMA_Propagator3D(){;}
    
    void absorbTimeSliceFromHost(PLEGMA_Propagator<Float> &prop, 
				 int timeslice);

    /**
       @brief Absorbs a time-slice from a 4D vector to a 3D Propagator at nu, c2 element
       @param PLEGMA_Vector<Float> vec, The 4D vector
       @param int global_it, The global time slice from where data will be extracted from the the 4D vector
       @param int nu, The spin index where to insert data 
       @param int c2, The color index where to insert data       
       @return void
     **/    
    void absorb(PLEGMA_Vector<Float> &vec, int global_it, int nu, int c2);

    /**
       @brief Absorbs a time-slice from a 4D propagator to a 3D Propagator
       @param PLEGMA_Vector<Float> prop, The 4D propagator
       @param int global_it, The global time slice from where data will be extracted from the the 4D prop
       @return void
     **/    
    void absorb(PLEGMA_Propagator<Float> &prop, int global_it);

    /**
       @brief Absorbs a 3D vector to a 3D Propagator at nu, c2 element
       @param PLEGMA_Vector3D<Float> vec, The 3D vector
       @param int nu, The spin index where to insert data 
       @param int c2, The color index where to insert data       
       @return void
     **/    
    void absorb(PLEGMA_Vector3D<Float> &vec, int nu, int c2);

  };
}

#endif
