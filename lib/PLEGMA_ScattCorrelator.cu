#include <PLEGMA_ScattCorrelator.h>
#include <PLEGMA_Vector.h>
#include <PLEGMA_Propagator.h>
//#include <PLEGMA_scattreductions.cuh>
#include <PLEGMA_scattreductions_new.cuh>

using namespace plegma;

//--------------------------------//
//  class PLEGMA_ScattCorrelator  //
//--------------------------------//

// From PLEGMA_Correlator
//
// - initialize() -> check if n_datasets,n_groups,shape change realloc PLEGMA_FT
// - finalize() -> delete corr_*_space;


template<typename Float>
void PLEGMA_ScattCorrelator<Float>::V3( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS> &Gammas, PLEGMA_Propagator<Float> &S) { 
  if( this->corr_space==POSITION_SPACE )
    PLEGMA_error("Not implemented yet\n");

  int n_gammas= Gammas.size();

  if(n_gammas<=0||n_gammas>16)
    PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  
  if(!this->isAlloc || this->site_size!=n_gammas*N_SPINS*N_COLS){
    this->datasets={"dataset_v3"};
    this->groups={"group_v3"};
    this->shape={n_gammas,N_SPINS,N_COLS};
    this->initialize();
  }


  int source[4]={0,0,0,0};
  const VRED V=V_3;
  this->setSource(source);
  
  V_reductions<V,Float,Float,Float>( *this, Phi, Gammas, S);
  //V3_k( *this, Phi, Gammas, S);
  
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::V4( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS> &Gammas, PLEGMA_Propagator<Float> &S1, PLEGMA_Propagator<Float> &S2) {

  if( this->corr_space==POSITION_SPACE )
    PLEGMA_error("Not implemented yet\n");

  int n_gammas= Gammas.size();

  if(n_gammas<=0||n_gammas>16)
    PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  
  if(!this->isAlloc || this->site_size!=n_gammas*N_SPINS*N_SPINS*N_SPINS*N_COLS){
    this->datasets={"dataset_v3"};
    this->groups={"group_v3"};
    this->shape={n_gammas,N_SPINS,N_SPINS,N_SPINS,N_COLS};
    this->initialize();
  }


  int source[4]={0,0,0,0};
  this->setSource(source);

  const VRED V=V_4;
  V_reductions<V,Float,Float,Float>( *this, Phi, Gammas, S1, S2);
  
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::V2( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS> &Gammas, PLEGMA_Propagator<Float> &S1, PLEGMA_Propagator<Float> &S2) {

  if( this->corr_space==POSITION_SPACE )
    PLEGMA_error("Not implemented yet\n");

  int n_gammas= Gammas.size();

  if(n_gammas<=0||n_gammas>16)
    PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  
  if(!this->isAlloc || this->site_size!=n_gammas*N_SPINS*N_SPINS*N_SPINS*N_COLS){
    this->datasets={"dataset_v2"};
    this->groups={"group_v2"};
    this->shape={n_gammas,N_SPINS,N_SPINS,N_SPINS,N_COLS};
    this->initialize();
  }


  int source[4]={0,0,0,0};
  this->setSource(source);

  const VRED V=V_2;
  V_reductions<V,Float,Float,Float>( *this, Phi, Gammas, S1, S2);
  
}

template class PLEGMA_ScattCorrelator<float>;
template class PLEGMA_ScattCorrelator<double>;
