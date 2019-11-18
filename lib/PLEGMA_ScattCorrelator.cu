#include <PLEGMA_ScattCorrelator.h>
#include <PLEGMA_scattreductions.cuh>

using namespace plegma;
//--------------------------------//
//  class PLEGMA_ScattCorrelator  //
//--------------------------------//

// From PLEGMA_Correlator
//
// - initialize() -> check if n_datasets,n_groups,shape change realloc PLEGMA_FT
// - finalize() -> delete corr_*_space;

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::V3( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS> &Gammas, PLEGMA_Propagator<Float> &S, int source[4]) {

  if( this->corr_space==POSITION_SPACE )
    PLEGMA_error("Not implemented yet\n");

  int n_gammas= this->Gammas.size();

  if(n_gammas==0||n_gammas>16)
    PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  
  if(!this->isAlloc || this->site_size!=n_gammas*N_SPINS*N_COLS){
    this->datasets={};
    this->groups={};
    this->shape={n_gammas,N_SPINS,N_COLS};
    this->initialize();
  }

  //N.B. source[0] not used
  this->setSource(source);
  V3_k( *this, Phi, Gammas, S);
  
}
