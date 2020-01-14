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
    this->shape_labels="gsc";
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
    this->datasets={"dataset_v3"};
    this->groups={"group_v3"};
    this->shape={n_gammas,N_SPINS,N_SPINS,N_SPINS,N_COLS};
    this->shape_labels="gsssc";
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
    this->shape_labels="gsssc";
    this->initialize();
  }


  int source[4]={0,0,0,0};
  this->setSource(source);

  const VRED V=V_2;
  V_reductions<V,Float,Float,Float>( *this, Phi, Gammas, S1, S2);
  
}

template<typename Float>
template <int s_free>
void PLEGMA_ScattCorrelator<Float>::absorb_fromV24( PLEGMA_ScattCorrelator<Float> &srcV2, int alfa, int beta){

  if( this->corr_space==POSITION_SPACE )
    PLEGMA_error("Not implemented yet\n");
  if( srcV2.corr_space==POSITION_SPACE )
    PLEGMA_error("Not implemented yet\n");

  //check site_size
  std::string expstr ("gsssc");
  if( srcV2.shape.size() != 5 || srcV2.Shape_labels().compare(expstr)!=0 )
    PLEGMA_error("The shape of src object must be of the type gsssc\n");
  if( srcV2.n_datasets()!=1 || srcV2.n_groups()!=1 )
    PLEGMA_error("1 dataset and 1 group only\n");

  //check index ranges
  if( s_free<0 || s_free>=3 ){
    PLEGMA_error("s_free %d out of range (0, 1 or 2)\n",s_free);
  }
  if( alfa<0 || alfa>=N_SPINS ) PLEGMA_error("%d out of range\n", alfa);
  if( beta<0 || beta>=N_SPINS ) PLEGMA_error("%d out of range\n", beta);

  
  //allocate
  int n_gammas = srcV2.shape[0];

  if(srcV2.fixMomVec.empty()) this->Q2_max = srcV2.Q2_max;
    else this->fixMomVec = srcV2.fixMomVec;
    
  this->datasets={""};
  this->groups={""};
  this->shape={n_gammas,N_SPINS,N_COLS};
  this->shape_labels="gsc";
  this->initialize();
  

  int source[4]={0,0,0,0};
  this->setSource( source );

  const unsigned short N_S1C=N_SPINS*N_COLS;
  const unsigned short N_S2C=N_SPINS*N_SPINS*N_COLS;
  const unsigned short N_S3C=N_SPINS*N_SPINS*N_SPINS*N_COLS;
  const unsigned short N_GS3C=n_gammas*N_SPINS*N_SPINS*N_SPINS*N_COLS;
  const unsigned short N_GS1C=n_gammas*N_SPINS*N_COLS;
  
  size_t VOL_SIZE = srcV2.getVolSize();
  Float* dest = this->corr;
  Float* src = srcV2->corr;
  for(int v=0; v < VOL_SIZE; v++)
    for(int g=0; g < n_gammas; g++)
      #pragma unroll
      for(int s=0; s < N_SPINS; s++)
	#pragma unroll
	for(int c=0; c < N_COLS; c++)
	  if( s_free == 0){
	    dest[v*N_GS1C+g*N_S1C+s*N_COLS+c] =
	      src[v*N_GS3C+g*N_S3C+s*N_S2C+alfa*N_S1C+beta*N_COLS+c];
	  } else if ( s_free == 1 ){
	    dest[v*N_GS1C+g*N_S1C+s*N_COLS+c] =
	      src[v*N_GS3C+g*N_S3C+alfa*N_S2C+s*N_S1C+beta*N_COLS+c];
	  } else {
	    dest[v*N_GS1C+g*N_S1C+s*N_COLS+c] =
	      src[v*N_GS3C+g*N_S3C+alfa*N_S2C+beta*N_S1C+s*N_COLS+c];
	  }
}

//V3_a^l*G_ab*V3_b^l
//template<typename Float>
//void PLEGMA_ScattCorrelator<Float>::V3V3reduction( PLEGMA_ScattCorrelator<Float> &srcV2, int alfa, int beta){

template class PLEGMA_ScattCorrelator<float>;
template class PLEGMA_ScattCorrelator<double>;
