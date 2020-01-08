#include <PLEGMA_ScattCorrelator.h>
#include <PLEGMA_Vector.h>
#include <PLEGMA_Propagator.h>
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
PLEGMA_ScattCorrelator<Float>::PLEGMA_ScattCorrelator(CORR_SPACE CorrSpace, int Q2_max):
  PLEGMA_Correlator<Float>(CorrSpace,Q2_max) { ; }

template<typename Float>
PLEGMA_ScattCorrelator<Float>::PLEGMA_ScattCorrelator(CORR_SPACE CorrSpace, std::vector<int> fixMomVec):
  PLEGMA_Correlator<Float>(CorrSpace,fixMomVec) { ; }

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
  
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::V4( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS> &Gammas, PLEGMA_Propagator<Float> &S1, PLEGMA_Propagator<Float> &S2) {

  if( this->corr_space==POSITION_SPACE )
    PLEGMA_error("Not implemented yet\n");

  int n_gammas= Gammas.size();

  if(n_gammas<=0||n_gammas>16)
    PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  
  if(!this->isAlloc || this->site_size!=n_gammas*N_SPINS*N_SPINS*N_SPINS*N_COLS){
    this->datasets={"dataset_v4"};
    this->groups={"group_v4"};
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

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::T1(std::vector<GAMMAS> &Gammas_i, std::vector<GAMMAS> &Gammas_f, PLEGMA_Propagator<Float> &S1, PLEGMA_Propagator<Float> &S2, PLEGMA_Propagator<Float> &S3) {

  if( this->corr_space==POSITION_SPACE )
    PLEGMA_error("Not implemented yet\n");

  int n_gammas_i= Gammas_i.size();
  int n_gammas_f= Gammas_f.size();
  
  if(n_gammas_i<=0||n_gammas_i>16)
    PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  
  if(n_gammas_f<=0||n_gammas_f>16)
    PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  
  if(!this->isAlloc || this->site_size!=n_gammas_i*n_gammas_f*N_SPINS*N_SPINS){
    this->datasets={"dataset_t1"};
    this->groups={"group_t1"};
    this->shape={n_gammas_i,n_gammas_f, N_SPINS,N_SPINS};
    this->initialize();
  }

  int source[4]={0,0,0,0};
  const TRED T=T_1;
  this->setSource(source);
  
  T_reductions<T,Float,Float>( *this, Gammas_i, Gammas_f, S1, S2, S3);
  
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::T2( std::vector<GAMMAS> &Gammas_i, std::vector<GAMMAS> &Gammas_f, PLEGMA_Propagator<Float> &S1, PLEGMA_Propagator<Float> &S2, PLEGMA_Propagator<Float> &S3) {

  if( this->corr_space==POSITION_SPACE )
    PLEGMA_error("Not implemented yet\n");

  int n_gammas_i= Gammas_i.size();
  int n_gammas_f= Gammas_f.size();

  if(n_gammas_i<=0||n_gammas_i>16)
    PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  
  if(n_gammas_f<=0||n_gammas_f>16)
    PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  
  if(!this->isAlloc || this->site_size!=n_gammas_i*n_gammas_f*N_SPINS*N_SPINS){
    this->datasets={"dataset_t2"};
    this->groups={"group_t2"};
    this->shape={n_gammas_i,n_gammas_f, N_SPINS,N_SPINS};
    this->initialize();
  }


  int source[4]={0,0,0,0};
  const TRED T=T_2;
  this->setSource(source);
  
  T_reductions<T,Float,Float>( *this, Gammas_i, Gammas_f, S1, S2, S3);
}

template class PLEGMA_ScattCorrelator<float>;
template class PLEGMA_ScattCorrelator<double>;
