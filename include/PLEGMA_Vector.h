#include <PLEGMA_Field.h>
#include <color_spinor_field.h>

#ifndef _PLEGMA_VECTOR_H
#define _PLEGMA_VECTOR_H

namespace plegma {

  // forward declaration
  template<typename Float>  class PLEGMA_Gauge;
  template<typename Float>  class PLEGMA_Gauge3D;
  template<typename Float>  class PLEGMA_Vector3D;
  template<typename Float>  class PLEGMA_Propagator;
  template<typename Float>  class PLEGMA_Propagator3D;
  template<typename Float>  class PLEGMA_Su3field;
  /////////////////////////
  // Class: PLEGMA_Vector //
  /////////////////////////
  
  template<typename Float>
  class PLEGMA_Vector : virtual public PLEGMA_Field<Float> {
  public:
    PLEGMA_Vector(ALLOCATION_FLAG alloc_flag=BOTH, GHOST_FLAG ghost_flag=FIRST_SIDE);
    ~PLEGMA_Vector(){;}
    
    void copyToQUDA( quda::ColorSpinorField *cudaVector, bool isEv = false);
    void copyFromQUDA( quda::ColorSpinorField *cudaVector, bool isEv = false);

    /**
       @brief Applies N times Gaussian(Wuppertal) smearing operator on all time-slices of a vector. NOTE: works also for Vector3D
       @param PLEGMA_Vector<Float> &vecIn, The 4D input vector (Exchange of boundaries happens inside the function)
       @param PLEGMA_Gauge<Float> &gauge, The gauge field that will be used in the Gaussian smearing operator (Exchange of boundaries happens inside the function)
       @param int nsmearGauss, The number of times to apply the operator (if zero copies inVec to outVec)
       @param Float alphaGauss, alpha parameter of the Gaussian smearing
     **/
    void gaussianSmearing(PLEGMA_Vector<Float> &vecIn, PLEGMA_Gauge<Float> &gauge, int nsmearGauss, Float alphaGauss);

    // This is the generic absorb. Other specializations follow
    using PLEGMA_Field<Float>::absorb;
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
    
    void dilutespin(PLEGMA_Vector<Float> &vecIn, int spin);

    void dilutecolor(PLEGMA_Vector<Float> &vecIn, int color);
    
    void dilutespincolor(PLEGMA_Vector<Float> &vecIn, int spin, int color);

    /**
       @brief Moves a spincomponent to another one, used for creating sources in spin dilution
       @param PLEGMA_Vector<Float> vecIn input vector (assumed to be non-zero only at one spin component
       @param int spin1 target spin index
       @param int spin2 original spin index
     **/
    void diluteSpinDisplace(PLEGMA_Vector<Float> &vecIn, int spin1, int spin2);
    
    void pointSource(const site& sourceposition, int spin, int color, ALLOCATION_FLAG alloc_flag=EVERY);
    void apply_gamma5();
    void apply_gamma(GAMMAS gMat, LEFTRIGHT LR = LEFT);
    void rotateToPhysicalBasis(PLEGMA_Vector<Float> &vecIn, int sgn);
    void apply_gamma_scatt( GAMMAS_SCATT gMat, LEFTRIGHT LR = LEFT);
    /**
       @brief Performs the similarity transformation of gamma matrices from tmLQCD to QUDA-UKQCD and vice versa
     **/
    void rotate_uk_ch();
    void covD(PLEGMA_Vector<Float> &vecIn, PLEGMA_Gauge<Float> &gauge, int dirOr);
    void mulGV(PLEGMA_Vector<Float> &vecIn, PLEGMA_Su3field<Float> &u);
  };

  template<typename Float> void copyToQUDA(quda::ColorSpinorField *cudaVector, Float* delem, bool isEv = false); // delem is a device pointer
  template<typename Float> void copyFromQUDA(Float* delem, quda::ColorSpinorField *cudaVector, bool isEv = false);

  /////////////////////////////////////
  // CLASS: PLEGMA_Vector3D ///////////
  ////////////////////////////////////
  
  template<typename Float>
  class PLEGMA_Vector3D : public PLEGMA_Field3D<Float>, public PLEGMA_Vector<Float> {
  public:
    PLEGMA_Vector3D(ALLOCATION_FLAG alloc_flag=BOTH, GHOST_FLAG ghost_flag=FIRST_SIDE) :
      PLEGMA_Field<Float>(alloc_flag, VECTOR3D, ghost_flag){ }

    ~PLEGMA_Vector3D(){ }
    
    // This is the generic absorb. Other specializations follow
    using PLEGMA_Field3D<Float>::absorb;
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

    void gaussianSmearing(PLEGMA_Vector3D<Float> &vecIn, PLEGMA_Gauge3D<Float> &gauge, int nsmearGauss, Float alphaGauss) {
      this->activeTimeSlice = vecIn.activeTimeSlice;
      return ((PLEGMA_Vector<Float>*) this)->gaussianSmearing(vecIn,gauge,nsmearGauss,alphaGauss);
    }

    void pointSource(const site& sourceposition, int spin, int color, ALLOCATION_FLAG alloc_flag=EVERY){
      int my_it = sourceposition[DIM_T] - HGC_procPosition[DIM_T] * HGC_localL[DIM_T];
      this->activeTimeSlice = (my_it >= 0) && ( my_it < HGC_localL[DIM_T] );
      return ((PLEGMA_Vector<Float>*) this)->pointSource(sourceposition,spin,color,alloc_flag);
    }
    
    std::vector<Float> rms(std::vector<int> listR2, const site& sourceposition) const;

    void seqSourceNucleon(PLEGMA_Propagator3D<Float> &prop1, PLEGMA_Propagator3D<Float> &prop2, WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_nu, int c_c2);
    void seqSourceNucleon(PLEGMA_Propagator3D<Float> &prop, WHICHPROJECTOR proj, WHICHPARTICLE particle, int c_nu, int c_c2);
  };
}

#endif
