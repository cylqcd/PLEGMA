#include <PLEGMA_Field.h>

#ifndef _PLEGMA_PROPAGATOR_H
#define _PLEGMA_PROPAGATOR_H

namespace plegma {
  // forward declaration
  template<typename Float>  class PLEGMA_Vector;

  /////////////////////////////
  // CLASS: PLEGMA_Propagator //
  /////////////////////////////
  
  template<typename Float>
    class PLEGMA_Propagator : public PLEGMA_Field<Float> {
    
  public:
    PLEGMA_Propagator(ALLOCATION_FLAG alloc_flag);
    ~PLEGMA_Propagator(){;}
    
    void conjugate();
    void apply_gamma5();
    
    void absorbVectorToHost(PLEGMA_Vector<Float> &vec, 
			    int nu, int c2);
    void absorbVectorToDevice(PLEGMA_Vector<Float> &vec, 
			      int nu, int c2);
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
    PLEGMA_Propagator3D(ALLOCATION_FLAG alloc_flag);
    ~PLEGMA_Propagator3D(){;}
    
    void absorbTimeSliceFromHost(PLEGMA_Propagator<Float> &prop, 
				 int timeslice);
    void absorbTimeSlice(PLEGMA_Propagator<Float> &prop, 
			 int timeslice);
    void absorbVectorTimeSlice(PLEGMA_Vector<Float> &vec, 
			       int timeslice, int nu, int c2);
    void broadcast(int tsink);
  };
}

#endif
