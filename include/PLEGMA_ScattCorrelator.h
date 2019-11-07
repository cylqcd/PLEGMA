#include <PLEGMA_Correlator.h>
#include <PLEGMA_Vector.h>
#include <PLEGMA_Propagator.h>

#pragma once

namespace plegma {
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
    //   size_t vol_size;
    //   std::vector<int> shape;

    //   // Allocated site_size = n_datasets * n_groups * prod(shape) (slowest to fastest running index)
    //   int site_size;
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
    PLEGMA_ScattCorrelator(CORR_SPACE CorrSpace, in Q2_max);
    PLEGMA_ScattCorrelator(CORR_SPACE CorrSpace, std::vector<int> fixMomVec);

    ~PLEGMA_ScattCorrelator();

    //functions that return values of protected variables
    std::string Shape_labels() const{ return shape_labels;}

    // COMMENT: probably source_pos needed. Not clear to me now, we will see...
    void V1( PLEGMA_Vector<Float> Phi, std::vector<Float> Gammas, PLEGMA_Propagator<Float> S);
    void V2( PLEGMA_Vector<Float> Phi, std::vector<Float> Gammas, PLEGMA_Propagator<Float> S1,  PLEGMA_Propagator<Float> S2);
    void V3( PLEGMA_Vector<Float> Phi, std::vector<Float> Gammas, PLEGMA_Propagator<Float> S);
    void V4( PLEGMA_Vector<Float> Phi, std::vector<Float> Gammas, PLEGMA_Propagator<Float> S1,  PLEGMA_Propagator<Float> S2 );


  };
}
