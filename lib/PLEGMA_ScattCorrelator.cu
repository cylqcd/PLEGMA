#include <PLEGMA_ScattCorrelator.h>

using namespace plegma;
//--------------------------------//
//  class PLEGMA_ScattCorrelator  //
//--------------------------------//

// From PLEGMA_Correlator
//
// - initialize() -> check if n_datasets,n_groups,shape change realloc PLEGMA_FT
// - finalize() -> delete corr_*_space;

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::V3( PLEGMA_Vector<Float> &Phi, std::vector<Float> Gammas, PLEGMA_Propagator<Float> &S) {

  if( corr_space==POSITION_SPACE )
    PLEGMA_Error("Not implemented yet\n");

  int n_gammas= Gammas.size()%16==0 ? Gammas.size()/16 : 0;

  if(n_gammas==0)
    PLEGMA_Error("size of Gammas must be multiple of 16\n");
  
  if(!isAlloc || site_size!=n_gammas*4*3){
    datasets={};
    groups={};
    shape={n_gammas,4,3};
    initialize();
  }
}
