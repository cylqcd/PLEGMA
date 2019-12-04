#pragma once
#include <PLEGMA_Correlator.h>

namespace plegma {
  enum VRED {V_2=2,V_3=3,V_4=4};
  
  // forward declaration
  template<typename Float>  class PLEGMA_Vector;
  template<typename Float>  class PLEGMA_Propagator;

  /////////////////
  // This PLEGMA_ScattCorrelator Class allows to store a generic number of d.o.f
  // (divided into spin and color index) per momentum. It  
  // will contain all the needed contractions V1, V2, V3, V4, T1, T2 plus
  // other functionalities ( print utilities inherited from Correlators, ....)
  ////////////////
  //
  // N.B. shape is the d.o.f per site -> shape of dataset
  //      other possible variables (for instance, a list of gammas) -> n_datasets*n_groups
 
  template<typename Float>
  class PLEGMA_ScattCorrelator : public PLEGMA_Correlator<Float>  {
  protected:
    //////////////
    // from PLEGMA_Correlator:
    //////////////
    //   // Allocation
    //   bool isAlloc;
    //   PLEGMA_Field<Float>* corr_pos_space;
    //   PLEGMA_FT<Float>* corr_mom_space;
    //   Float* corr;

    //   // Correlator info
    //   CORR_SPACE corr_space;
    //   int Q2_max;
    //   std::vector<int> fixMomVec ;
    //   size_t vol_size; COMMENT: number_of_momenta*time_ext
    //   std::vector<int> shape;

    //   // Allocated 
    //   int site_size; COMMENT:site_size = n_datasets * n_groups * prod(shape) (slowest to fastest running index)
    //   std::array<int,4> source_position;

    //   // Writing informations
    //   std::vector<std::string> datasets;
    //   std::vector<std::string> groups;
    //   std::string description;
    //////////////
    
    std::string shape_labels;    //     index_struct = "sssc" (because spin first)
    int shape_size;               //     prod(shape)    = 4*4*4*3
    
  public:
    // these constructors does NOT ALLOCATE the memory PLEGMA_ScattCorrelator here, because
    // the dimension is not provided. It will be allocated when used.
    PLEGMA_ScattCorrelator(CORR_SPACE CorrSpace, int Q2_max):
      PLEGMA_Correlator<Float>(CorrSpace,Q2_max) { ; }

    PLEGMA_ScattCorrelator(CORR_SPACE CorrSpace, std::vector<int> fixMomVec):
      PLEGMA_Correlator<Float>(CorrSpace,fixMomVec) { ; }

    ~PLEGMA_ScattCorrelator(){;}

    //functions that return values of protected variables
    std::string Shape_labels() const{ return shape_labels;}

    void V1( PLEGMA_Vector<Float> Phi, std::vector<Float> Gammas, PLEGMA_Propagator<Float> S);
    void V2( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS> &Gammas, PLEGMA_Propagator<Float> &S1,  PLEGMA_Propagator<Float> &S2 );
    void V3( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS> &Gammas, PLEGMA_Propagator<Float> &S);
    void V4( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS> &Gammas, PLEGMA_Propagator<Float> &S1,  PLEGMA_Propagator<Float> &S2 );

  };
}
