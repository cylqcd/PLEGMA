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
PLEGMA_ScattCorrelator<Float>::PLEGMA_ScattCorrelator(CORR_SPACE CorrSpace, std::vector<std::vector<int>> fixMomsVec):
  PLEGMA_Correlator<Float>(CorrSpace,fixMomsVec) { ; }

template<typename Float>
bool PLEGMA_ScattCorrelator<Float>::is_V24() {
  std::string exp_shape("gsssc");

  if( this->corr_space==POSITION_SPACE )
    return false;

  //check site_size
  if( this->shape.size() != 5 || this->shape_labels.compare(exp_shape)!=0 )
    return false;
  if( this->n_datasets()!=1 || this->n_groups()!=1 )
    return false;

  return true;
}
template<typename Float>
bool PLEGMA_ScattCorrelator<Float>::is_V3() {
  std::string exp_shape("gsc");

  if( this->corr_space==POSITION_SPACE )
    return false;

  //check site_size
  if( this->shape.size() != 3 || this->shape_labels.compare(exp_shape)!=0 )
    return false;
  if( this->n_datasets()!=1 || this->n_groups()!=1 )
    return false;

  return true;
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::V3( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS_SCATT> &Gammas, PLEGMA_Propagator<Float> &S) { 
  if( this->corr_space==POSITION_SPACE )
    PLEGMA_error("Not implemented yet\n");

  int n_gammas= Gammas.size();

  if(n_gammas<=0||n_gammas>16)
    PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  this->GList=Gammas;
  
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
void PLEGMA_ScattCorrelator<Float>::V4( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS_SCATT> &Gammas, PLEGMA_Propagator<Float> &S1, PLEGMA_Propagator<Float> &S2) {

  if( this->corr_space==POSITION_SPACE )
    PLEGMA_error("Not implemented yet\n");

  int n_gammas= Gammas.size();

  if(n_gammas<=0||n_gammas>16)
    PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  this->GList=Gammas;
  
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
void PLEGMA_ScattCorrelator<Float>::V2( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS_SCATT> &Gammas, PLEGMA_Propagator<Float> &S1, PLEGMA_Propagator<Float> &S2) {

  if( this->corr_space==POSITION_SPACE )
    PLEGMA_error("Not implemented yet\n");

  int n_gammas= Gammas.size();

  if(n_gammas<=0||n_gammas>16)
    PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  this->GList=Gammas;
  
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
void print_groups_names( momList &moms, std::vector<GAMMAS_SCATT> &G_i1, GAMMAS_SCATT G_i2 , std::vector<GAMMAS_SCATT> &G_f1, std::vector<GAMMAS_SCATT> &G_f2, std::vector<std::string> &out){
  std::string tmp;
  out.clear();
  for(auto &mom : moms.print() )
    for( auto &g1 : G_i1 )
      for( auto &g2 : G_f1 )
	for( auto &g3 : G_f2 ){
	  tmp = mom + "/" + GAMMAS_STR[g1] + "/" + GAMMAS_STR[G_i2] + "/" + GAMMAS_STR[g2] + "/" + GAMMAS_STR[g3];
	  out.push_back(tmp);
	}
}
void print_groups_names( momList &moms, std::vector<GAMMAS_SCATT> &G_i1, std::vector<GAMMAS_SCATT> &G_i2 , std::vector<GAMMAS_SCATT> &G_f1, std::vector<GAMMAS_SCATT> &G_f2, std::vector<std::string> &out){
  std::string tmp;
  out.clear();
  for(auto &mom : moms.print() )
    for( auto &g1 : G_i1 )
      for( auto &g2 : G_i2 )
	for( auto &g3 : G_f1 )
	  for( auto &g4 : G_f2 ){
	    tmp = mom + "/" + GAMMAS_STR[g1] + "/" + GAMMAS_STR[g2] + "/" + GAMMAS_STR[g3] + "/" + GAMMAS_STR[g4];
	    out.push_back(tmp);
	  }
}


template<typename Float>
void PLEGMA_ScattCorrelator<Float>::B_diagramms(momList &moms, PLEGMA_ScattCorrelator<Float> &srcV3, PLEGMA_ScattCorrelator<Float> &srcV2, GAMMAS_SCATT G_i2, std::vector<GAMMAS_SCATT> &Gammas_i1, std::string &outfile) {

  if( this->corr_space == POSITION_SPACE )
    PLEGMA_error("Not implemented yet\n");

  //allocate
  //int n_gammas_f1 = srcV2.shape[0];
  this->datasets={"B1"};
  print_groups_names(moms, Gammas_i1, G_i2, srcV2.GList, srcV3.GList, this->groups);
  this->shape={N_SPINS,N_SPINS};
  this->shape_labels="ss";
  this->initialize();
  
  if( this->vol_size != HGC_localL[3] )
    PLEGMA_error("PLEGMA_SC for writing must have N_moms=1. Detected vol_size=%d\n",this->vol_size);
  
  if( this->site_size != moms.size()*Gammas_i1.size()*srcV2.GList.size()*srcV3.GList.size()*N_SPINS*N_SPINS )
    PLEGMA_error("I did some mistakes. vol_size*site_size=%d; expected= (mom=%d),(Gi1=%d),(Gf2=%d),(Gf1=%d)%d\n",
		 this->vol_size*this->site_size,moms.size(),Gammas_i1.size(),srcV2.GList.size(),srcV3.GList.size(),
		 moms.size()*Gammas_i1.size()*srcV2.GList.size()*srcV3.GList.size()*HGC_localL[3]*N_SPINS*N_SPINS);

  
  std::vector<std::array<int,3>> imap=moms.index_map();
  int offset=Gammas_i1.size()*srcV2.GList.size()*srcV3.GList.size()*N_SPINS*N_SPINS*2;

  //write B1
  for(int i_m=0; i_m<imap.size(); i_m++)
    srcV3.V3V2reduction( Gammas_i1, imap[i_m], srcV2, this->corr + offset*i_m, 2, true);

  this->writeHDF5(outfile);

  //write B2
  this->datasets={"B2"};
  for(int i_m=0; i_m<imap.size(); i_m++)
    srcV3.V3V2reduction( Gammas_i1, imap[i_m], srcV2, this->corr + offset*i_m, 0);

  this->writeHDF5(outfile);
  
}


template<typename Float>
void PLEGMA_ScattCorrelator<Float>::W_diagramms(momList &moms, PLEGMA_ScattCorrelator<Float> &srcV3, PLEGMA_ScattCorrelator<Float> &srcV2, GAMMAS_SCATT G_i2, std::vector<GAMMAS_SCATT> &Gammas_i1, std::string &outfile, int diagramm_index){

  if( this->corr_space == POSITION_SPACE )
    PLEGMA_error("Not implemented yet\n");


  if( (diagramm_index != 1) && (diagramm_index !=2 ) &&  (diagramm_index != 3) &&  (diagramm_index != 4)   )
    PLEGMA_error("diagramm_index %d out of range (1,2,3 or 4)\n",diagramm_index);

  this->datasets={"W"+std::to_string(diagramm_index)};
  print_groups_names(moms, Gammas_i1, G_i2, srcV2.GList, srcV3.GList, this->groups);
  this->shape={N_SPINS,N_SPINS};
  this->shape_labels="ss";
  this->initialize();
  if( this->vol_size != HGC_localL[3] )
    PLEGMA_error("PLEGMA_SC for writing must have N_moms=1\n");
 
  if( this->site_size != moms.size()*Gammas_i1.size()*srcV2.GList.size()*srcV3.GList.size()*N_SPINS*N_SPINS )
    PLEGMA_error("I did some mistakes. vol_size*site_size=%d; expected= (mom=%d),(Gi1=%d),(Gf2=%d),(Gf1=%d)%d\n",
		 this->vol_size*this->site_size,moms.size(),Gammas_i1.size(),srcV2.GList.size(),srcV3.GList.size(),
		 moms.size()*Gammas_i1.size()*srcV2.GList.size()*srcV3.GList.size()*HGC_localL[3]*N_SPINS*N_SPINS);


  std::vector<std::array<int,3>> imap=moms.index_map();
  int offset=Gammas_i1.size()*srcV2.GList.size()*srcV3.GList.size()*N_SPINS*N_SPINS*2;

  for(int i_m=0; i_m<imap.size(); i_m++){
    //write W1
    if (diagramm_index == 1){
      srcV3.V3V2reduction( Gammas_i1, imap[i_m], srcV2, this->corr + offset*i_m, 2, true, true );}
    //write W2
    else if (diagramm_index == 2){
      srcV3.V3V2reduction_matrix( Gammas_i1, imap[i_m], srcV2, this->corr + offset*i_m, 1, false);}
    //write W3
    else if (diagramm_index == 3){
      srcV3.V3V2reduction_matrix( Gammas_i1, imap[i_m], srcV2, this->corr + offset*i_m, 1, false, true);}
    //write W4
    else {
      srcV3.V3V2reduction( Gammas_i1, imap[i_m], srcV2, this->corr + offset*i_m, 0, false, true);}
  }
  this->writeHDF5(outfile);

}
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::V3V2reduction_matrix(std::vector<GAMMAS_SCATT> &Gammas_i1, std::array<int,3> &indexmap, PLEGMA_ScattCorrelator<Float> &srcV2, Float *dest, int index_abs, bool transp, bool transpgamma, int n_gammas_i2, int g0) {
  int n_gammas_i1 = Gammas_i1.size();
  if ((n_gammas_i1 <= 0) || (n_gammas_i1 >16)){
   PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  }

  if( !srcV2.is_V24() ) PLEGMA_error("SrcV2 object does not seem a V2like object\n");
  if( !this->is_V3() ) PLEGMA_error("SrcV3 object does not seem a V3like object\n");

  int n_gammas_f1 = srcV2.getGList().size();
  int n_gammas_f2 = this->GList.size();

  int Nmoms_f1 = srcV2.getVolSize()/HGC_localL[3];
  int Nmoms_f2 = this->vol_size/HGC_localL[3];

  
  PLEGMA_ScattCorrelator<Float> V3aux = (!srcV2.getFixMomList().empty()) ?
    PLEGMA_ScattCorrelator<Float>(MOMENTUM_SPACE, srcV2.fixMomList) :
    ( (!srcV2.fixMomVec.empty()) ? PLEGMA_ScattCorrelator<Float>(MOMENTUM_SPACE, srcV2.fixMomVec) : PLEGMA_ScattCorrelator<Float>(MOMENTUM_SPACE, srcV2.Q2_max) );


  Float* srcf2 = this->corr;//V3
  Float* srcf1;//V2

  int TIME=HGC_localL[3];
  //
  const int f2_MGSC2=Nmoms_f2*n_gammas_f2*N_SPINS*N_COLS*2;
  const int f2_GSC2=n_gammas_f2*N_SPINS*N_COLS*2;
  const int N_SC2=N_SPINS*N_COLS*2;
  const int N_SSC2=N_SPINS*N_SPINS*N_COLS*2;
  //
  const int f1_MGSSC2=Nmoms_f1*n_gammas_f1*N_SPINS*N_SPINS*N_COLS*2;
  const int f1_GSSC2=n_gammas_f1*N_SPINS*N_SPINS*N_COLS*2;
  //
  const int d_GGTSS2=n_gammas_f1*n_gammas_f2*TIME*N_SPINS*N_SPINS*2;
  const int d_GTSS2=n_gammas_f2*TIME*N_SPINS*N_SPINS*2;
  const int d_TSS2=TIME*N_SPINS*N_SPINS*2;
  const int d_SS2=N_SPINS*N_SPINS*2;

  int i_mom_f1 = indexmap[1];
  int i_mom_f2 = indexmap[2];


  Float *temporary_colorvector=(Float *)malloc(sizeof(Float)*6);
  for (int alfa=0 ; alfa < N_SPINS; ++alfa){
    for (int beta=0; beta < N_SPINS; ++beta){
      int spins = (transp) ? (beta*N_SPINS+alfa)*2 : (alfa*N_SPINS+beta)*2;
      switch(index_abs){
        case 0: V3aux.absorbspinmatrix_fromV24<0>( srcV2, alfa ); break;
        case 1: V3aux.absorbspinmatrix_fromV24<1>( srcV2, alfa ); break;
        case 2: V3aux.absorbspinmatrix_fromV24<2>( srcV2, alfa ); break;
      }
 
      srcf1=V3aux.getCorr();
      for (int g1=0 ; g1 < n_gammas_i1 ; ++g1 ){//pi
	for (int g2=0 ; g2 < n_gammas_f1 ; ++g2 ){//pf1
	  for (int g3=0 ; g3 < n_gammas_f2 ; ++g3 ){//pf2
            for(int t=0; t < TIME; t++){
              V_TR_MM<Float>(srcf1+t*f1_MGSSC2 + i_mom_f1*f1_GSSC2+g2*N_SSC2,
                             Gammas_i1[g1],
                             transpgamma,
                             temporary_colorvector);
              dest[(g1*n_gammas_i2+g0)*d_GGTSS2+g2*d_GTSS2+g3*d_TSS2+t*d_SS2+spins+0]=0.;
              dest[(g1*n_gammas_i2+g0)*d_GGTSS2+g2*d_GTSS2+g3*d_TSS2+t*d_SS2+spins+1]=0.;
             
              for (int coloridx=0; coloridx<3; ++coloridx){
                dest[(g1*n_gammas_i2+g0)*d_GGTSS2+g2*d_GTSS2+g3*d_TSS2+t*d_SS2+spins+0]+=
                   +temporary_colorvector[2*coloridx+0]*srcf2[t*f2_MGSC2+i_mom_f2*f2_GSC2+g3*N_SC2+beta*N_COLS*2+2*coloridx+0]
                   -temporary_colorvector[2*coloridx+1]*srcf2[t*f2_MGSC2+i_mom_f2*f2_GSC2+g3*N_SC2+beta*N_COLS*2+2*coloridx+1];
                dest[(g1*n_gammas_i2+g0)*d_GGTSS2+g2*d_GTSS2+g3*d_TSS2+t*d_SS2+spins+1]+=
                   +temporary_colorvector[2*coloridx+1]*srcf2[t*f2_MGSC2+i_mom_f2*f2_GSC2+g3*N_SC2+beta*N_COLS*2+2*coloridx+0]
                   +temporary_colorvector[2*coloridx+0]*srcf2[t*f2_MGSC2+i_mom_f2*f2_GSC2+g3*N_SC2+beta*N_COLS*2+2*coloridx+1];

              }              
            }
          }
        }
      }
    }
  }
  free(temporary_colorvector);
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::Z_diagramms(
                                                momList &moms, 
                                                std::array<PLEGMA_ScattCorrelator<Float>,4> (&srcV3),
                                                std::array<PLEGMA_ScattCorrelator<Float>,4> (&srcV2),
                                                std::vector<GAMMAS_SCATT> &Gammas_i2, 
                                                std::vector<GAMMAS_SCATT> &Gammas_i1, 
                                                std::string &outfile, 
                                                int diagramm_index ){

  if( (diagramm_index != 1) && (diagramm_index !=2 ) &&  (diagramm_index != 3) &&  (diagramm_index != 4)   )
    PLEGMA_error("diagramm_index %d out of range (1,2,3 or 4)\n",diagramm_index);

  if( this->corr_space == POSITION_SPACE )
    PLEGMA_error("Not implemented yet\n");

  const int tot_size= moms.size()*Gammas_i1.size()*Gammas_i2.size()*srcV2[0].GList.size()*srcV3[0].GList.size()*HGC_localL[3]*N_SPINS*N_SPINS*2;
  const int d_GGGGTSS2= tot_size/moms.size();
  const int d_GGGGTSS = d_GGGGTSS2/2;

  this->datasets={"Z"+std::to_string(diagramm_index)};
  print_groups_names(moms, Gammas_i1, Gammas_i2, srcV2[0].GList, srcV3[0].GList, this->groups);
  this->shape={N_SPINS,N_SPINS};
  this->shape_labels="ss";
  this->initialize();
  if( this->vol_size != HGC_localL[3] )
    PLEGMA_error("PLEGMA_SC for writing must have N_moms=1\n");

  if( this->site_size*2 != tot_size/HGC_localL[3] )
    PLEGMA_error("I did some mistakes. vol_size*site_size=%d; expected= (mom=%d),(Gi1=%d),(Gi2=%d),(Gf2=%d),(Gf1=%d)%d\n",
		 this->vol_size*this->site_size,moms.size(),Gammas_i1.size(),Gammas_i2.size(),
		 srcV2[0].GList.size(),srcV3[0].GList.size(),tot_size/2);

  Float * temporary= (Float *)malloc(sizeof(Float)*d_GGGGTSS2);


  std::vector<std::array<int,3>> imap=moms.index_map();

  //write Z1

  Float *dest = this->corr;

  memset(this->corr,0,tot_size*sizeof(Float));
  for(int i_m=0; i_m<imap.size(); i_m++){
    for (int g2=0; g2<Gammas_i2.size();++g2 ){
      GAMMAS_SCATT gammai2= Gammas_i2[g2];
      memset(temporary,0,d_GGGGTSS2*sizeof(Float));
      for (int n=0; n<4; ++n){
        int kappa= gammaInd_scatt_host[gammai2][n][0]; 
        int lambda=  gammaInd_scatt_host[gammai2][n][1];
        Float g[2];
        g[1]=gamma_scatt_host[gammai2][n][1];
        g[0]=gamma_scatt_host[gammai2][n][0];
        if (diagramm_index==1){
          srcV3[lambda].V3V2reduction( Gammas_i1, imap[i_m], srcV2[kappa], temporary, 1,false, true, Gammas_i2.size(),g2 );
        }
        else if (diagramm_index ==2){
          srcV3[lambda].V3V2reduction_matrix( Gammas_i1, imap[i_m], srcV2[kappa], temporary, 0,false, true, Gammas_i2.size(),g2 );
        }
        else if (diagramm_index ==3){
          srcV3[lambda].V3V2reduction_matrix( Gammas_i1, imap[i_m], srcV2[kappa], temporary, 1,false, true, Gammas_i2.size(),g2 );
        }
        else {
          srcV3[lambda].V3V2reduction( Gammas_i1, imap[i_m], srcV2[kappa], temporary, 0,false, true, Gammas_i2.size(),g2 );
        }

        x_pe_cy(dest+i_m*d_GGGGTSS2, g, temporary, d_GGGGTSS);

      }
    }
  }

  free(temporary);

  this->writeHDF5(outfile);

}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::V3V2reduction(std::vector<GAMMAS_SCATT> &Gammas_i1, std::array<int,3> &indexmap, PLEGMA_ScattCorrelator<Float> &srcV2, Float *dest, int index_abs, bool transp, bool transpgamma, int n_gammas_i2, int g0) {
  int n_gammas_i1 = Gammas_i1.size();
  if ((n_gammas_i1 <= 0) || (n_gammas_i1 >16)){
   PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  } 

  if( !srcV2.is_V24() ) PLEGMA_error("SrcV2 object does not seem a V2like object\n");
  if( !this->is_V3() ) PLEGMA_error("SrcV3 object does not seem a V3like object\n");
 
  int n_gammas_f1 = srcV2.getGList().size();
  int n_gammas_f2 = this->GList.size();

  int Nmoms_f1 = srcV2.getVolSize()/HGC_localL[3];
  int Nmoms_f2 = this->vol_size/HGC_localL[3];

  PLEGMA_ScattCorrelator<Float> V3aux = (!srcV2.getFixMomList().empty()) ?
    PLEGMA_ScattCorrelator<Float>(MOMENTUM_SPACE, srcV2.fixMomList) :
    ( (!srcV2.fixMomVec.empty()) ? PLEGMA_ScattCorrelator<Float>(MOMENTUM_SPACE, srcV2.fixMomVec) : PLEGMA_ScattCorrelator<Float>(MOMENTUM_SPACE, srcV2.Q2_max) );
  
  Float* srcf2 = this->corr;//V3
  Float* srcf1; //V2
  
  int TIME=HGC_localL[3];
  //
  const int f2_MGSC2=Nmoms_f2*n_gammas_f2*N_SPINS*N_COLS*2;
  const int f2_GSC2=n_gammas_f2*N_SPINS*N_COLS*2;
  const int N_SC2=N_SPINS*N_COLS*2;
  //
  const int f1_MGSC2=Nmoms_f1*n_gammas_f1*N_SPINS*N_COLS*2;
  const int f1_GSC2=n_gammas_f1*N_SPINS*N_COLS*2;
  //
  const int d_GGTSS2=n_gammas_f1*n_gammas_f2*TIME*N_SPINS*N_SPINS*2;
  const int d_GTSS2=n_gammas_f2*TIME*N_SPINS*N_SPINS*2;
  const int d_TSS2=TIME*N_SPINS*N_SPINS*2;
  const int d_SS2=N_SPINS*N_SPINS*2;
 
  int i_mom_f1 = indexmap[1];
  int i_mom_f2 = indexmap[2];
  for (int alfa=0; alfa < N_SPINS; ++alfa ){
    for (int beta=0; beta < N_SPINS; ++beta ){
      int spins = (transp) ? (beta*N_SPINS+alfa)*2 : (alfa*N_SPINS+beta)*2;
      switch(index_abs){
      case 0: V3aux.absorb_fromV24<0>( srcV2, alfa, beta ); break;
      case 1: V3aux.absorb_fromV24<1>( srcV2, alfa, beta ); break;
      case 2: V3aux.absorb_fromV24<2>( srcV2, alfa, beta ); break;
      }
      srcf1=V3aux.getCorr();
      for (int g1=0 ; g1 < n_gammas_i1 ; ++g1 ){//pi
	for (int g2=0 ; g2 < n_gammas_f1 ; ++g2 ){//pf1
	  for (int g3=0 ; g3 < n_gammas_f2 ; ++g3 ){//pf2
	    for(int t=0; t < TIME; ++t){
	      V_M_V<Float>( srcf2 + t*f2_MGSC2 + i_mom_f2*f2_GSC2 + g3*N_SC2,
	                    srcf1 + t*f1_MGSC2 + i_mom_f1*f1_GSC2 + g2*N_SC2, 
		            Gammas_i1[g1], 
                            transpgamma,
		            dest + (g1*n_gammas_i2 + g0)*d_GGTSS2 + g2*d_GTSS2 + g3*d_TSS2 + t*d_SS2 + spins );
	    }
          }
        }
      }
    }
  }
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::D_diagramms(
                                                momList &moms,
                                                PLEGMA_Propagator<Float> (&S1),
                                                PLEGMA_Propagator<Float> (&S2),
                                                PLEGMA_Propagator<Float> (&S3),                                                                          std::vector<GAMMAS_SCATT> &Gammas_i1,
                                                std::vector<GAMMAS_SCATT> &Gammas_f1,
                                                std::vector<GAMMAS_SCATT> &Gammas_i2,
                                                std::vector<GAMMAS_SCATT> &Gammas_f2,
                                                std::string &outfile){

  const int tot_size= moms.size()*Gammas_i1.size()*Gammas_f1.size()*Gammas_i2.size()*&Gammas_f2.size()*HGC_localL[3]*N_SPINS*N_SPINS*2;

  this->datasets={"D"};
  print_groups_names(moms, Gammas_i1, Gammas_f1, Gammas_i2, Gammas_f2, this->groups);
  this->shape={N_SPINS,N_SPINS};
  this->shape_labels="ss";
  this->initialize();
  if( this->vol_size != HGC_localL[3] )
    PLEGMA_error("PLEGMA_SC for writing must have N_moms=1\n");

  if( this->site_size*2 != tot_size/HGC_localL[3] )
    PLEGMA_error("I did some mistakes. vol_size*site_size=%d; expected= (mom=%d),(Gi1=%d),(Gi2=%d),(Gf2=%d),(Gf1=%d)%d\n",
                 this->vol_size*this->site_size,moms.size(),Gammas_f1.size(),Gammas_i1.size(),
                                                            Gammas_f2.size(),Gammas_i2.size(),tot_size/2);


  PLEGMA_ScattCorrelator<float> reductionsT1(MOMENTUM_SPACE, moms.uniq_p(0));
  PLEGMA_ScattCorrelator<float> reductionsT2(MOMENTUM_SPACE, moms.uniq_p(0));

  reductionsT1.T1( Gammas_i1, Gammas_f1, S1, S2, S3);
  reductionsT2.T2( Gammas_i1, Gammas_f1, S1, S2, S3);

  int Nmom_T1=reductionsT1.getVolSize()/HGC_localL[3];
  const int i_MGGTSS2=Nmom_T1*Gammas_i1.size()*Gammas_f1.size()*HGC_localL[3]*N_SPINS*N_SPINS*2;
  const int d_SS2;


  Float *srcT1 = reductionsT1.getCorr();
  Float *srcT2 = reductionsT2.getCorr();

  Float *dest = this->corr;

  for (int f2g=0; f2g < Gammas_f2.size(); ++f2g ){
    for (int i2g=0; i2g < Gammas_i2.size(); ++i2g ){
      GAMMAS_SCATT_SCATT gammaf2= Gammas_f2[f2g];
      GAMMAS_SCATT_SCATT gammai2= Gammas_i2[i2g];
      for (int n=0; n<4; ++n){
        int alfa =   gammaInd_scatt_host[gammaf2][n][0];
        int alfa0=   gammaInd_scatt_host[gammaf2][n][1];

        int beta=    gammaInd_scatt_host[gammai2][n][0];
        int beta0=   gammaInd_scatt_host[gammai2][n][1];
             
        Float gf[2];
        Float gi[2];
        gi[1]=gamma_scatt_host[gammai2][n][1];
        gi[0]=gamma_scatt_host[gammai2][n][0];
             
        gf[1]=gamma_scatt_host[gammaf2][n][1];
        gf[0]=gamma_scatt_host[gammaf2][n][0];
          
        for (int internalind=0; internalind < i_MGGTV; ++internalind){
          dest[(f2g*Gammas_i2.size()+i2g)*i_MGGTSS2+internalind*d_SS2+(alfa*N_SPINS+beta)*2+0]=
                +gf[0]*T1[internalind*d_SS2+(alfa0*N_SPINS+beta0)*2+0]*gi[0]*4.
                -gf[1]*T1[internalind*d_SS2+(alfa0*N_SPINS+beta0)*2+1]*gi[0]*4.
                -gf[1]*T1[internalind*d_SS2+(alfa0*N_SPINS+beta0)*2+0]*gi[1]*4.
                -gf[0]*T1[internalind*d_SS2+(alfa0*N_SPINS+beta0)*2+1]*gi[1]*4.
                +gf[0]*T2[internalind*d_SS2+(alfa0*N_SPINS+beta0)*2+0]*gi[0]*2.
                -gf[1]*T2[internalind*d_SS2+(alfa0*N_SPINS+beta0)*2+1]*gi[0]*2.
                -gf[1]*T2[internalind*d_SS2+(alfa0*N_SPINS+beta0)*2+0]*gi[1]*2.
                -gf[0]*T2[internalind*d_SS2+(alfa0*N_SPINS+beta0)*2+1]*gi[1]*2.;

          dest[(f2g*Gammas_i2.size()+i2g)*i_MGGTSS2+internalind*d_SS2+(alfa*N_SPINS+beta)*2+1]=
                -gf[1]*T1[internalind*d_SS2+(alfa0*N_SPINS+beta0)*2+1]*gi[1]*4.
                +gf[1]*T1[internalind*d_SS2+(alfa0*N_SPINS+beta0)*2+0]*gi[0]*4.
                +gf[0]*T1[internalind*d_SS2+(alfa0*N_SPINS+beta0)*2+1]*gi[0]*4.
                +gf[0]*T1[internalind*d_SS2+(alfa0*N_SPINS+beta0)*2+0]*gi[1]*4.
                -gf[1]*T2[internalind*d_SS2+(alfa0*N_SPINS+beta0)*2+1]*gi[1]*2.
                +gf[1]*T2[internalind*d_SS2+(alfa0*N_SPINS+beta0)*2+0]*gi[0]*2.
                +gf[0]*T2[internalind*d_SS2+(alfa0*N_SPINS+beta0)*2+1]*gi[0]*2.
                +gf[0]*T2[internalind*d_SS2+(alfa0*N_SPINS+beta0)*2+0]*gi[1]*2.;

        }
      }
    }
  } 
}
// template<typename Float>
// void PLEGMA_ScattCorrelator<Float>::T1(std::vector<GAMMAS_SCATT> &Gammas_i, std::vector<GAMMAS_SCATT> &Gammas_f, PLEGMA_Propagator<Float> &S1, PLEGMA_Propagator<Float> &S2, PLEGMA_Propagator<Float> &S3) {

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
// void PLEGMA_ScattCorrelator<Float>::T2( std::vector<GAMMAS_SCATT> &Gammas_i, std::vector<GAMMAS_SCATT> &Gammas_f, PLEGMA_Propagator<Float> &S1, PLEGMA_Propagator<Float> &S2, PLEGMA_Propagator<Float> &S3) {

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
void PLEGMA_ScattCorrelator<Float>::contract_GxV2_checks( PLEGMA_ScattCorrelator<Float> &srcV2){
  if( this->corr_space==POSITION_SPACE )
    PLEGMA_error("Not implemented yet\n");
  if(!srcV2.is_V24()) PLEGMA_error("srcV2 seems not to have V2like shape\n");
  
  if(!srcV2.fixMomList.empty()){
    if( this->fixMomList != srcV2.fixMomList ) PLEGMA_error("src and dest must have same momenta\n");
  }
  else if(!srcV2.fixMomVec.empty()){
    if( this->fixMomVec != srcV2.fixMomVec ) PLEGMA_error("src and dest must have same momenta\n");
  } else {
    if(this->Q2_max != srcV2.Q2_max) PLEGMA_error("src and dest must have same momenta\n");
  }

  //allocate
  int n_gammas = srcV2.GList.size();
  this->GList=srcV2.GList;
  this->datasets={"absorbvectorfromV24"};
  this->groups={"absorbvectorfromV24"};
  this->shape={n_gammas,N_SPINS,N_COLS};
  this->shape_labels="gsc";
  this->initialize();
  

  int source[4]={0,0,0,0};
  this->setSource( source );
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::absorb_fromV24_checks( PLEGMA_ScattCorrelator<Float> &srcV2, int alfa, int beta){
  if( this->corr_space==POSITION_SPACE )
    PLEGMA_error("Not implemented yet\n");
  if(!srcV2.is_V24()) PLEGMA_error("srcV2 seems not to have V2like shape\n");
  
  //check index ranges
  if( alfa<0 || alfa>=N_SPINS ) PLEGMA_error("%d out of range\n", alfa);
  if( beta<0 || beta>=N_SPINS ) PLEGMA_error("%d out of range\n", beta);

  
  //allocate
  int n_gammas = srcV2.shape[0];

  if(!srcV2.fixMomList.empty()){
    if( this->fixMomList != srcV2.fixMomList ) PLEGMA_error("src and dest must have same momenta. fixMomList detected.\n");
  }
  else if(!srcV2.fixMomVec.empty()){
    if( this->fixMomVec != srcV2.fixMomVec ) PLEGMA_error("src and dest must have same momenta. fixMomVec detected.\n");
  } else {
    if(this->Q2_max != srcV2.Q2_max) PLEGMA_error("src and dest must have same momenta. Q2_max detected.\n");
  }
  
  this->GList=srcV2.GList;
  this->datasets={"absorbvectorfromV24"};
  this->groups={"absorbvectorfromV24"};
  this->shape={n_gammas,N_SPINS,N_COLS};
  this->shape_labels="gsc";
  this->initialize();
  
  size_t exp_size = srcV2.vol_size*n_gammas*N_SPINS*N_COLS;
  if( this->getTotalSize() != exp_size )
    PLEGMA_error("total size %d != expected_size %d\n",this->getTotalSize(),exp_size);

  int source[4]={0,0,0,0};
  this->setSource( source );
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::absorbspinmatrix_fromV24_checks( PLEGMA_ScattCorrelator<Float> &srcV2, int alfa){
  if( this->corr_space==POSITION_SPACE )
    PLEGMA_error("Not implemented yet\n");
  if(!srcV2.is_V24()) PLEGMA_error("srcV2 seems not to have V2like shape\n");
  
  //check index ranges
  if( alfa<0 || alfa>=N_SPINS ) PLEGMA_error("%d out of range\n", alfa);


  //allocate
  int n_gammas = srcV2.shape[0];

  if(!srcV2.fixMomList.empty()){
    if( this->fixMomList != srcV2.fixMomList ) PLEGMA_error("src and dest must have same momenta\n");
  }
  else if(!srcV2.fixMomVec.empty()){
    if( this->fixMomVec != srcV2.fixMomVec ) PLEGMA_error("src and dest must have same momenta\n");
  } else {
    if(this->Q2_max != srcV2.Q2_max) PLEGMA_error("src and dest must have same momenta\n");
  }
  
  this->GList=srcV2.GList;
  this->datasets={"absorbmatrixv24"};
  this->groups={"absorbmatrixv24"};
  this->shape={n_gammas,N_SPINS,N_SPINS,N_COLS};
  this->shape_labels="gssc";
  this->initialize();

  size_t exp_size = srcV2.vol_size*n_gammas*N_SPINS*N_SPINS*N_COLS;
  if( this->getTotalSize() != exp_size )
    PLEGMA_error("total size %d != expected_size %d\n",this->getTotalSize(),exp_size);
  int source[4]={0,0,0,0};
  this->setSource( source );
}



//V3_a^l*G_ab*V3_b^l
//template<typename Float>
//void PLEGMA_ScattCorrelator<Float>::V3V3reduction( PLEGMA_ScattCorrelator<Float> &srcV2, int alfa, int beta){

template class PLEGMA_ScattCorrelator<float>;
template class PLEGMA_ScattCorrelator<double>;


