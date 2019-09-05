#include <PLEGMA_Field.h>
#include "complex"

#ifndef _PLEGMA_GAUGE_H
#define _PLEGMA_GAUGE_H

namespace plegma {
  template<typename Float> class PLEGMA_Su3field;
  ////////////////////////
  // CLASS: PLEGMA_Gauge //
  ////////////////////////
  
  template<typename Float>
    class PLEGMA_Gauge : public PLEGMA_Field<Float> {
  public:
    PLEGMA_Gauge(ALLOCATION_FLAG alloc_flag=BOTH, GHOST_FLAG ghost_flag=FIRST_CORNER);
    ~PLEGMA_Gauge(){;}
    
    void absorbDir_device(PLEGMA_Su3field<Float> &su,int dir);
    void absorbDir_host(PLEGMA_Su3field<Float> &su,int dir);
    void stoutSmearing(PLEGMA_Gauge<Float> &uin, int nSmear, double rho, int D3D4);

    void APEsmearing(PLEGMA_Gauge<Float> &uin, int nSmear, double alpha, int D3D4);

    Float calculatePlaq();

    //for checks
    Float calculatePlaqCorners();
    Float calculatePlaqShifts();
    Float calculatePlaqClover();
    Float calculatePlaqStaples();

    //PLEGMA_topocharge.cuh
    Float calculateTopo(TOPO_CHARGE_DEF charge_def);

    //PLEGMA_WFlow.cuh
    void GFlow_step( PLEGMA_Gauge<Float> &Z, double eps );
    void applyGradientFlow( PLEGMA_Gauge<Float> &Z, int N, double eps);
    void unitarize();
    void print_fields( int sid );

    void scaleDirWise(std::complex<Float> scale[N_DIMS]);
    void momPhase(Float phase[N_DIMS],int mom[N_DIMS]);
  };
}

#endif
