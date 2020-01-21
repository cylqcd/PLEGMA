#include <PLEGMA_ScattCorrelator.h>
#include <PLEGMA_Vector.h>
#include <PLEGMA_Propagator.h>
#include <PLEGMA_scattreductions.cuh>
#include <PLEGMA_utils.h>

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
    this->datasets={"dataset_v4"};
    this->groups={"group_v4"};
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
template <int diagramm_index>
void PLEGMA_ScattCorrelator<Float>::B_diagramms(std::vector<GAMMAS> &Gammas_i1, PLEGMA_ScattCorrelator<Float> &srcV3, PLEGMA_ScattCorrelator<Float> &srcV2) {

  if( (diagramm_index != 1) || (diagramm_index !=2 ) )
    PLEGMA_error("diagramm_index %d out of range (1 or 2)\n",diagramm_index);

  if( this ->corr_space == POSITION_SPACE )
    PLEGMA_error("Not implemented yet\n");
  int n_gammas_i1 = Gammas_i1.size();
  if ((n_gammas_i1 <= 0) || (n_gammas_i1 >16)){
   PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  } 
  if( srcV2.corr_space==POSITION_SPACE )
    PLEGMA_error("Not implemented yet\n");

  //check site_size
  std::string expstr1 ("gsssc");
  if( srcV2.shape.size() != 5 || srcV2.Shape_labels().compare(expstr1)!=0 )
    PLEGMA_error("The shape of srcV2 object must be of the type gsssc\n");
  if( srcV2.n_datasets()!=1 || srcV2.n_groups()!=1 )
    PLEGMA_error("1 dataset and 1 group only\n");

  //allocate
  int n_gammas_f1 = srcV2.shape[0];

  if( srcV3.corr_space==POSITION_SPACE )
    PLEGMA_error("Not implemented yet\n");

  //check site_size
  std::string expstr2 ("gsc");
  if( srcV3.shape.size() != 3 || srcV3.Shape_labels().compare(expstr2)!=0 )
    PLEGMA_error("The shape of srcV3 object must be of the type gsc\n");
  if( srcV3.n_datasets()!=1 || srcV3.n_groups()!=1 )
    PLEGMA_error("1 dataset and 1 group only\n");

  //allocate
  int n_gammas_f2 = srcV3.shape[0];

  std::string dataset_diagrammString = "dataset_B" + std::to_string(diagramm_index) + "diagramm";
  std::string group_diagrammString = "group_B" + std::to_string(diagramm_index) + "diagramm";

  if(!this->isAlloc || this->site_size!=n_gammas_i1*n_gammas_f1*n_gammas_f2*N_SPINS*N_SPINS){
    this->datasets={dataset_diagrammString};
    this->groups={group_diagrammString};
    this->shape={n_gammas_i1,n_gammas_f1,n_gammas_f2,N_SPINS,N_SPINS};
    this->shape_labels="gggss";
    this->initialize();
  }

  int NG3SPIN2=n_gammas_i1*n_gammas_f1*n_gammas_f2*N_SPINS*N_SPINS ;
  int NG2SPIN2=n_gammas_f1*n_gammas_f2*N_SPINS*N_SPINS ;
  int NG1SPIN2=n_gammas_f2*N_SPINS*N_SPINS ;
  int NSPIN2= N_SPINS*N_SPINS ;
  int NG1NSPIN1NCOL1_V3=n_gammas_f2*N_SPINS*N_COLS;
  int NG1NSPIN1NCOL1_V2R=n_gammas_f1*N_SPINS*N_COLS;


  int source[4]={0,0,0,0};
  this->setSource(source);
  for (int alfa=0 ; alfa < N_SPINS; ++alfa){
    for (int beta=0; beta < N_SPINS; ++beta){
      std::vector<int> mom={0,0,0};
      PLEGMA_ScattCorrelator<Float> temporaryV3(MOMENTUM_SPACE, mom );
      if (diagramm_index == 1){
        temporaryV3.absorb_fromV24<2>( srcV2, beta, alfa);
      }
      else{
        temporaryV3.absorb_fromV24<0>( srcV2, alfa, beta);
      }
      #pragma unroll
      for (int loop_gammai1=0 ; loop_gammai1 < n_gammas_i1 ; ++loop_gammai1 ){
        #pragma unroll
        for ( int loop_gammaf1=0; loop_gammaf1 < n_gammas_f1 ; ++loop_gammaf1 ){
          #pragma unroll
          for ( int loop_gammaf2=0; loop_gammaf2 < n_gammas_f2 ; ++loop_gammaf2 ){
            size_t VOL_SIZE = srcV3.getVolSize();
            Float* dest = this->corr;
            Float* src1 = srcV3.corr;
            Float* src2 = temporaryV3.corr;
            for(int v=0; v < VOL_SIZE; v++){
              V_M_V<Float>( &src1[2*(v*NG1NSPIN1NCOL1_V3+loop_gammaf2*N_SPINS*N_COLS)],
                     &src2[2*(v*NG1NSPIN1NCOL1_V2R+loop_gammaf1*N_SPINS*N_COLS)], 
                     Gammas_i1[loop_gammai1], 
                     &dest[2*(v*NG3SPIN2+loop_gammai1*NG2SPIN2+loop_gammaf1*NG1SPIN2+loop_gammaf2*NSPIN2+alfa*N_SPINS+beta)] );
            }
          }
        }
      }
    }
  }
}

template<typename Float>
template <int diagramm_index>
void PLEGMA_ScattCorrelator<Float>::W_diagramms(std::vector<GAMMAS> &Gammas_i1, PLEGMA_ScattCorrelator<Float> &srcV3, PLEGMA_ScattCorrelator<Float> &srcV2) {

  if( (diagramm_index != 1) || (diagramm_index !=2 ) ||  (diagramm_index != 3) ||  (diagramm_index != 4)   )
    PLEGMA_error("diagramm_index %d out of range (1,2,3 or 4)\n",diagramm_index);

  if( this ->corr_space == POSITION_SPACE )
    PLEGMA_error("Not implemented yet\n");
  int n_gammas_i1 = Gammas_i1.size();
  if ((n_gammas_i1 <= 0) || (n_gammas_i1 >16)){
   PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  }
  if( srcV2.corr_space==POSITION_SPACE )
    PLEGMA_error("Not implemented yet\n");

  //check site_size
  std::string expstr1 ("gsssc");
  if( srcV2.shape.size() != 5 || srcV2.Shape_labels().compare(expstr1)!=0 )
    PLEGMA_error("The shape of srcV2 object must be of the type gsssc\n");
  if( srcV2.n_datasets()!=1 || srcV2.n_groups()!=1 )
    PLEGMA_error("1 dataset and 1 group only\n");

  //allocate
  int n_gammas_f1 = srcV2.shape[0];

  if( srcV3.corr_space==POSITION_SPACE )
    PLEGMA_error("Not implemented yet\n");

  //check site_size
  std::string expstr2 ("gsc");
  if( srcV3.shape.size() != 3 || srcV3.Shape_labels().compare(expstr2)!=0 )
    PLEGMA_error("The shape of srcV3 object must be of the type gsc\n");
  if( srcV3.n_datasets()!=1 || srcV3.n_groups()!=1 )
    PLEGMA_error("1 dataset and 1 group only\n");

  //allocate
  int n_gammas_f2 = srcV3.shape[0];

  std::string dataset_diagrammString = "dataset_W" + std::to_string(diagramm_index) + "diagramm";
  std::string group_diagrammString = "group_W" + std::to_string(diagramm_index) + "diagramm";

  if(!this->isAlloc || this->site_size!=n_gammas_i1*n_gammas_f1*n_gammas_f2*N_SPINS*N_SPINS){
    this->datasets={dataset_diagrammString};
    this->groups={group_diagrammString};
    this->shape={n_gammas_i1,n_gammas_f1,n_gammas_f2,N_SPINS,N_SPINS};
    this->shape_labels="gggss";
    this->initialize();
  }

  int NG3SPIN2=n_gammas_i1*n_gammas_f1*n_gammas_f2*N_SPINS*N_SPINS ;
  int NG2SPIN2=n_gammas_f1*n_gammas_f2*N_SPINS*N_SPINS ;
  int NG1SPIN2=n_gammas_f2*N_SPINS*N_SPINS ;
  int NSPIN2= N_SPINS*N_SPINS ;
  int NG1NSPIN1NCOL1_V3=n_gammas_f2*N_SPINS*N_COLS;
  int NG1NSPIN1NCOL1_V2R=n_gammas_f1*N_SPINS*N_COLS;


  int source[4]={0,0,0,0};
  this->setSource(source);
  std::vector<Float> temporary(24);
  for (int alfa=0 ; alfa < N_SPINS; ++alfa){
    for (int beta=0; beta < N_SPINS; ++beta){
      std::vector<int> mom={0,0,0};
      PLEGMA_ScattCorrelator<Float> temporaryV3(MOMENTUM_SPACE, mom );
      if ( diagramm_index == 1 ){
        temporaryV3.absorb_fromV24<2>( srcV2, beta, alfa);
      }
      else if ( (diagramm_index == 2) || (diagramm_index == 3) ){
        temporaryV3.absorbspinmatrix_fromV24<1>( srcV2, alfa );
      }
      else{
        temporaryV3.absorb_fromV24<0>( srcV2, alfa, beta);
      }
      #pragma unroll
      for (int loop_gammai1=0 ; loop_gammai1 < n_gammas_i1 ; ++loop_gammai1 ){
        #pragma unroll
        for ( int loop_gammaf1=0; loop_gammaf1 < n_gammas_f1 ; ++loop_gammaf1 ){
          #pragma unroll
          for ( int loop_gammaf2=0; loop_gammaf2 < n_gammas_f2 ; ++loop_gammaf2 ){
            size_t VOL_SIZE = srcV3.getVolSize();
            Float* dest = this->corr;
            Float* src1 = srcV3.corr;
            Float* src2 = temporaryV3.corr;
            for(int v=0; v < VOL_SIZE; v++){
              if ( (diagramm_index == 1) || (diagramm_index == 4)) {
                V_M_V<Float>( &src1[2*(v*NG1NSPIN1NCOL1_V3+loop_gammaf2*N_SPINS*N_COLS)],
                              &src2[2*(v*NG1NSPIN1NCOL1_V2R+loop_gammaf1*N_SPINS*N_COLS)],
                              Gammas_i1[loop_gammai1],
                              &dest[2*(v*NG3SPIN2+loop_gammai1*NG2SPIN2+loop_gammaf1*NG1SPIN2+loop_gammaf2*NSPIN2+alfa*N_SPINS+beta)] );
              }
              else {
                V_TR_MM<Float,1>(&src2[2*(v*NG1NSPIN1NCOL1_V3+loop_gammaf2*N_SPINS*N_COLS)], 
                                 Gammas_i1[loop_gammai1], temporary);
                dest[2*(v*NG3SPIN2+loop_gammai1*NG2SPIN2+loop_gammaf1*NG1SPIN2+loop_gammaf2*NSPIN2+alfa*N_SPINS+beta)+0]=0.0;
                dest[2*(v*NG3SPIN2+loop_gammai1*NG2SPIN2+loop_gammaf1*NG1SPIN2+loop_gammaf2*NSPIN2+alfa*N_SPINS+beta)+1]=0.0;
                for (int coloridx=0; coloridx<3; ++coloridx ){
                  dest[2*(v*NG3SPIN2+loop_gammai1*NG2SPIN2+loop_gammaf1*NG1SPIN2+loop_gammaf2*NSPIN2+alfa*N_SPINS+beta)+0]+=temporary[2*alfa*N_COLS+2*color.idx+0]*src1[2*(v*NG1NSPIN1NCOL1_V3+loop_gammaf2*N_SPINS*N_COLS+beta*N_COLS+color.idx)+0]-temporary[2*alfa*N_COLS+2*coloridx+1]*src1[2*(v*NG1NSPIN1NCOL1_V3+loop_gammaf2*N_SPINS*N_COLS+beta*N_COLS+coloridx)+1];
                  dest[2*(v*NG3SPIN2+loop_gammai1*NG2SPIN2+loop_gammaf1*NG1SPIN2+loop_gammaf2*NSPIN2+alfa*N_SPINS+beta)+1]+=temporary[2*alfa*N_COLS+2*color.idx+0]*src1[2*(v*NG1NSPIN1NCOL1_V3+loop_gammaf2*N_SPINS*N_COLS+beta*N_COLS+color.idx)+1]+temporary[2*alfa*N_COLS+2*coloridx+1]*src1[2*(v*NG1NSPIN1NCOL1_V3+loop_gammaf2*N_SPINS*N_COLS+beta*N_COLS+coloridx)+0];
                }
              }
            }
          }
        }
      }
    }
  }
}



// template<typename Float>
// void PLEGMA_ScattCorrelator<Float>::T1(std::vector<GAMMAS> &Gammas_i, std::vector<GAMMAS> &Gammas_f, PLEGMA_Propagator<Float> &S1, PLEGMA_Propagator<Float> &S2, PLEGMA_Propagator<Float> &S3) {

//   if( this->corr_space==POSITION_SPACE )
//     PLEGMA_error("Not implemented yet\n");

//   int n_gammas_i= Gammas_i.size();
//   int n_gammas_f= Gammas_f.size();
  
//   if(n_gammas_i<=0||n_gammas_i>16)
//     PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  
//   if(n_gammas_f<=0||n_gammas_f>16)
//     PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  
//   if(!this->isAlloc || this->site_size!=n_gammas_i*n_gammas_f*N_SPINS*N_SPINS){
//     this->datasets={"dataset_t1"};
//     this->groups={"group_t1"};
//     this->shape={n_gammas_i,n_gammas_f, N_SPINS,N_SPINS};
//     this->initialize();
//   }

//   int source[4]={0,0,0,0};
//   const TRED T=T_1;
//   this->setSource(source);
  
//   T_reductions<T,Float,Float>( *this, Gammas_i, Gammas_f, S1, S2, S3);
  
// }

// template<typename Float>
// void PLEGMA_ScattCorrelator<Float>::T2( std::vector<GAMMAS> &Gammas_i, std::vector<GAMMAS> &Gammas_f, PLEGMA_Propagator<Float> &S1, PLEGMA_Propagator<Float> &S2, PLEGMA_Propagator<Float> &S3) {

//   if( this->corr_space==POSITION_SPACE )
//     PLEGMA_error("Not implemented yet\n");

//   int n_gammas_i= Gammas_i.size();
//   int n_gammas_f= Gammas_f.size();

//   if(n_gammas_i<=0||n_gammas_i>16)
//     PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  
//   if(n_gammas_f<=0||n_gammas_f>16)
//     PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  
//   if(!this->isAlloc || this->site_size!=n_gammas_i*n_gammas_f*N_SPINS*N_SPINS){
//     this->datasets={"dataset_t2"};
//     this->groups={"group_t2"};
//     this->shape={n_gammas_i,n_gammas_f, N_SPINS,N_SPINS};
//     this->initialize();
//   }


//   int source[4]={0,0,0,0};
//   const TRED T=T_2;
//   this->setSource(source);
  
//   T_reductions<T,Float,Float>( *this, Gammas_i, Gammas_f, S1, S2, S3);
// }

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::absorb_fromV24_checks( PLEGMA_ScattCorrelator<Float> &srcV2, int alfa, int beta){
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
  if( alfa<0 || alfa>=N_SPINS ) PLEGMA_error("%d out of range\n", alfa);
  if( beta<0 || beta>=N_SPINS ) PLEGMA_error("%d out of range\n", beta);

  
  //allocate
  int n_gammas = srcV2.shape[0];

  if(srcV2.fixMomVec.empty()) this->Q2_max = srcV2.Q2_max;
    else this->fixMomVec = srcV2.fixMomVec;
    
  this->datasets={"absorbvectorfromV24"};
  this->groups={"absorbvectorfromV24"};
  this->shape={n_gammas,N_SPINS,N_COLS};
  this->shape_labels="gsc";
  this->initialize();
  

  int source[4]={0,0,0,0};
  this->setSource( source );
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::absorbspinmatrix_fromV24_checks( PLEGMA_ScattCorrelator<Float> &srcV2, int alfa){
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
  if( alfa<0 || alfa>=N_SPINS ) PLEGMA_error("%d out of range\n", alfa);


  //allocate
  int n_gammas = srcV2.shape[0];

  if(srcV2.fixMomVec.empty()) this->Q2_max = srcV2.Q2_max;
    else this->fixMomVec = srcV2.fixMomVec;

  this->datasets={"absorbmatrixv24"};
  this->groups={"absorbmatrixv24"};
  this->shape={n_gammas,N_SPINS,N_COLS};
  this->shape_labels="gssc";
  this->initialize();


  int source[4]={0,0,0,0};
  this->setSource( source );
}



//V3_a^l*G_ab*V3_b^l
//template<typename Float>
//void PLEGMA_ScattCorrelator<Float>::V3V3reduction( PLEGMA_ScattCorrelator<Float> &srcV2, int alfa, int beta){

template class PLEGMA_ScattCorrelator<float>;
template class PLEGMA_ScattCorrelator<double>;


