#include <PLEGMA_Field.h>
#include "complex"

#ifndef _PLEGMA_GAUGEU1_H
#define _PLEGMA_GAUGEU1_H

namespace plegma {
  template<typename Float> class PLEGMA_Su3field;
  ////////////////////////
  // CLASS: PLEGMA_GaugeU1 //
  ////////////////////////
  
  template<typename Float>
  class PLEGMA_GaugeU1 : virtual public PLEGMA_Field<Float> {
  public:
    PLEGMA_GaugeU1(ALLOCATION_FLAG alloc_flag=BOTH, GHOST_FLAG ghost_flag=FIRST_CORNER);
    ~PLEGMA_GaugeU1(){;}
    Float calculatePlaq(Float phase);
   /* 
    void absorbDir_device(PLEGMA_Su3field<Float> &su,int dir);
    void absorbDir_host(PLEGMA_Su3field<Float> &su,int dir);


    //PLEGMA_topocharge.cuh
    Float calculateTopo(TOPO_CHARGE_DEF charge_def);

    //PLEGMA_WFlow.cuh
    void GFlow_step( PLEGMA_GaugeU1<Float> &Z, double eps );
    void applyGradientFlow( PLEGMA_GaugeU1<Float> &Z, int N, double eps);
    void unitarize();

    void scaleDirWise(std::complex<Float> scale[N_DIMS]);
    void momPhase(Float phase[N_DIMS],int mom[N_DIMS]);
    void gFixingLandau(PLEGMA_GaugeU1<Float> &uIn,Float overelaxPar=0.2,Float tolerance=1.0e-12,int maxIter=10000, int seedOverRelax=123456);
    void gluonField(PLEGMA_GaugeU1<Float> &uIn);
  */
  };
}

#endif
