#include <PLEGMA_Field.h>
#include <PLEGMA_Gauge.h>

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
    class PLEGMA_Propagator : virtual public PLEGMA_Field<Float> {
    
  public:
    PLEGMA_Propagator(ALLOCATION_FLAG alloc_flag=BOTH, GHOST_FLAG ghost_flag=FIRST_SIDE);
    ~PLEGMA_Propagator(){;}
    
    void apply_gamma(GAMMAS gMat, LEFTRIGHT LR = LEFT);
    void apply_gamma5();
    void absorbVectorToHost(PLEGMA_Vector<Float> &vec, 
			    int nu, int c2);

    // This is the generic absorb. Other specializations follow
    using PLEGMA_Field<Float>::absorb;
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
       @brief Absorbs a 3D vector to a 4D Propagator at a specific global time and nu, c2 element
       @param PLEGMA_Vector3D<Float> vec, The 3D vector
       @param int global_it, The global time slice where data which will be inserted, the rest of the time-slices will become zero in the 4D propagator
       @param int nu, The spin index where to insert data 
       @param int c2, The color index where to insert data       
       @return void
     **/    
    void absorb(PLEGMA_Vector3D<Float> &vec, int global_it, int nu, int c2);

    /**
       @brief Applies N times Gaussian(Wuppertal) smearing operator on all time-slices of a vector. NOTE: works also for Vector3D
       @param PLEGMA_Vector<Float> &vecIn, The 4D input vector (Exchange of boundaries happens inside the function)
       @param PLEGMA_Gauge<Float> &gauge, The gauge field that will be used in the Gaussian smearing operator (Exchange of boundaries happens inside the function)
       @param int nsmearGauss, The number of times to apply the operator (if zero copies inVec to outVec)
       @param Float alphaGauss, alpha parameter of the Gaussian smearing
    **/
    void gaussianSmearing(PLEGMA_Propagator<Float> &propIn, PLEGMA_Gauge<Float> &gauge, int nsmearGauss, Float alphaGauss);

    void applyBoundaries_device(int t0);
    void rotateToPhysicalBase_host(int sign);
    void rotateToPhysicalBase_device(int sign);
  };

  ///////////////////////////////
  // CLASS: PLEGMA_Propagator3D //
  /////////////////////////////// 
  
  template<typename Float>
  class PLEGMA_Propagator3D : public PLEGMA_Field3D<Float>, public PLEGMA_Propagator<Float> {
  public:
    PLEGMA_Propagator3D(ALLOCATION_FLAG alloc_flag=BOTH, GHOST_FLAG ghost_flag=NO_GHOSTS) : 
      PLEGMA_Field<Float>(alloc_flag, PROPAGATOR3D, ghost_flag) { }
    ~PLEGMA_Propagator3D(){ }
    
    void absorbTimeSliceFromHost(PLEGMA_Propagator<Float> &prop, 
				 int timeslice);
    
    // This is the generic absorb. Other specializations follow
    using PLEGMA_Field3D<Float>::absorb;
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
