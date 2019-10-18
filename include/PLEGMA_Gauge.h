#include <PLEGMA_Field.h>
#include "complex"

#ifndef _PLEGMA_GAUGE_H
#define _PLEGMA_GAUGE_H

namespace plegma {
  template<typename Float> class PLEGMA_Su3field;
  ////////////////////////
  // CLASS: PLEGMA_Gauge //
  ////////////////////////
   /**
     @brief A child class of PLEGMA_Field with specialization for Gauge field
   **/  
  template<typename Float>
    class PLEGMA_Gauge : public PLEGMA_Field<Float> {
  public:
    /**
       @brief Constructor of the PLEGMA_Gauge. Calls the constructor of its parent
       @param[in] alloc_flag: see "allocation"
       @param[in] ghost_flag: choose the type of halo exchange
    **/
    PLEGMA_Gauge(ALLOCATION_FLAG alloc_flag=BOTH, GHOST_FLAG ghost_flag=FIRST_CORNER);
    ~PLEGMA_Gauge(){;}
    /**
       @brief Absorbs a SU3 field to specific direction of the gauge field on device
       @param[in] su: the input SU3 field
       @param[in] dir: the direction of the gauge field
    **/
    void absorbDir_device(PLEGMA_Su3field<Float> &su,int dir);
    /**
       @brief Absorbs a SU3 field to specific direction of the gauge field on host
       @param[in] su: the input SU3 field
       @param[in] dir: the direction of the gauge field
    **/
    void absorbDir_host(PLEGMA_Su3field<Float> &su,int dir);
    /**
       @brief Stout smearing as introduced hep-lat/0311018
       @param[in] uin: Input gauge field to be smeared
       @param[in] nSmear: number of smearing steps
       @param[in] rho: stout smearing parameter
       @param[in] D3D4: Either to 3 or 4 directions do the smearing
    **/
    void stoutSmearing(PLEGMA_Gauge<Float> &uin, int nSmear, double rho, int D3D4);
    /**
       @brief APE smearing for the SU3 links
       @param[in] uin: Input gauge field to be smeared
       @param[in] nSmear: number of smearing steps
       @param[in] alpha: APE smearing parameter
       @param[in] D3D4: Either to 3 or 4 directions do the smearing
    **/
    void APEsmearing(PLEGMA_Gauge<Float> &uin, int nSmear, double alpha, int D3D4);

    /**
       @brief Calculation of the plaquettes summed over all the directions
       @return the plaquette value normalized
     **/
    Float calculatePlaq();
    /**
       @bried Calculation of plaquette using corners ghost. Implemented for sanity check
       @return the plaquette value normalized
     **/
    Float calculatePlaqCorners();
    /**
       @bried Calculation of plaquette using the shift routines. Implemented for sanity check
       @return the plaquette value normalized
     **/
    Float calculatePlaqShifts();
    /**
       @bried Calculation of plaquette using the clover leaves. Implemented for sanity check
       @return the plaquette value normalized
     **/
    Float calculatePlaqClover();
    /**
       @bried Calculation of plaquette using the staples computations. Implemented for sanity check
       @return the plaquette value normalized
     **/
    Float calculatePlaqStaples();
    /**
       @brief Computes the topological charge for several definitions
       @param[in] charge_def: Options (Plaquette, Clover)
       @return the value of the topological charge
     **/
    Float calculateTopo(TOPO_CHARGE_DEF charge_def);
    /**
       @brief Performs the Wilson flow per step on gauge configuration with Wilson action and forth order integrator
       @param[in] Z: Input gauge field
       @param[in] eps: epsilon used for the integration
     **/
    void GFlow_step( PLEGMA_Gauge<Float> &Z, double eps );
    /**
       @brief Performs the Wilson flow calling GFlow_step to do many steps (N*eps=flow time)
       @param[in] Z: Input gauge field
       @param[in] N: number of steps in the integration
       @param[in] eps: epsilon used for the integration
     **/
    void applyGradientFlow( PLEGMA_Gauge<Float> &Z, int N, double eps);
    /**
       @brief Enforce that the SU3 field is obeys unitarity in every direction
     **/
    void unitarize();
    /**
       @brief Scales the Gauge field with a different value in each direction
       @param[in] scale: array with four values for each direction
     **/
    void scaleDirWise(std::complex<Float> scale[N_DIMS]);
    /**
       @brief Landau gauge fixing using stochastic overelaxation
       @param[in] uIn: Input gauge field
       @param[in] overelaxPar: Propability to use g<- g^2
       @param[in] tolerance: Tolerance to reach maximum of the functional 1/V \sum_x \sum_\mu Re Tr[U]
       @param[in] maxIter: Maximum number of iteration 
       @param[in] seedOverRelax: Seed for the random number generator needed for the overelaxation
     **/    
    void gFixingLandau(PLEGMA_Gauge<Float> &uIn,Float overelaxPar=0.2,Float tolerance=1.0e-8,int maxIter=10000, int seedOverRelax=123456);
    /**
       @bried Computes the gluon field given a gauge field with definition A_mu = \frac{1/2i} [ (U_\mu - U_\mu^\dag) - \frac{1}{3} Tr[U_\mu - U_\mu^\dag]]. Note that g_0 is not included
     **/
    void gluonField(PLEGMA_Gauge<Float> &uIn);
  };
}

#endif
