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
    PLEGMA_Vector(ALLOCATION_FLAG alloc_flag);
    ~PLEGMA_Vector(){;}
    
    void packVector(Float *vector);
    void unpackVector();
    void unpackVector(Float *vector);
    void loadVector();
    void unloadVector();
    void ghostToHost();
    void cpuExchangeGhost();
    void ghostToDevice();
    
    void download(); // take the vector from device to host
    void copyToQUDA( quda::ColorSpinorField *cudaVector, bool isEv = false);
    void copyFromQUDA( quda::ColorSpinorField *cudaVector, bool isEv = false);
    void gaussianSmearing(PLEGMA_Vector<Float> &vecIn, PLEGMA_Gauge<Float> &gaugeAPE);
    void scaleVector(Float a);
    void copy(PLEGMA_Vector<float> &vecIn);
    void copy(PLEGMA_Vector<double> &vecIn);
    void norm2Host();
    void norm2Device();
    void copyPropagator3D(PLEGMA_Propagator3D<Float> &prop, 
			  int timeslice, int nu , int c2);
    void copyPropagator(PLEGMA_Propagator<Float> &prop, 
			int nu , int c2);
    void write(char* filename);
    void conjugate();
    void apply_gamma5();
  };
}

#endif
