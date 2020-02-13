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
void PLEGMA_ScattCorrelator<Float>::V3( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS_SCATT> &Gammas, PLEGMA_Propagator<Float> &S){ 
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

//create a list with the structure of hdf5 file for 4pt. Groups order is the same of arguments order. The printed momenta are p_i1, p_i2, p_f1, p_f2.
void print_groups_names_4pt( momList &moms, std::vector<GAMMAS_SCATT> &extG_i1, std::vector<GAMMAS_SCATT> &extG_f1, std::vector<GAMMAS_SCATT> &G_i1, std::vector<GAMMAS_SCATT> &G_i2 , std::vector<GAMMAS_SCATT> &G_f1, std::vector<GAMMAS_SCATT> &G_f2, std::vector<std::string> &out){
  std::string tmp;
  out.clear();
  for(auto &mom : moms.print() )
    for( auto &g1e : extG_i1 )
      for( auto &g3e : extG_f1 )
	for( auto &g1 : G_i1 )
	  for( auto &g2 : G_i2 )
	    for( auto &g3 : G_f1 )
	      for( auto &g4 : G_f2 ){
		tmp = mom + "/" + GAMMAS_SCATT_STR[g1]+"-"+GAMMAS_SCATT_STR[g1e] + "/" + GAMMAS_SCATT_STR[g2] + "/"
		  + GAMMAS_SCATT_STR[g3]+"-"+GAMMAS_SCATT_STR[g3e]+ "/" + GAMMAS_SCATT_STR[g4];
		out.push_back(tmp);
	      }
}

//create a list with the structure of hdf5 file for 2pt. Groups order is the same of arguments order. The printed momentum is the total one (p_f1+p_f2).
void print_groups_names_2pt( std::vector<std::vector<int>> &moms, std::vector<GAMMAS_SCATT> &extG_i, std::vector<GAMMAS_SCATT> &extG_f, std::vector<GAMMAS_SCATT> &G_i, std::vector<GAMMAS_SCATT> &G_f,  std::vector<std::string> &out){
  std::string tmp;
  out.clear();
  for(auto &mom : moms )
    for( auto &g1e : extG_i )
      for( auto &g2e : extG_f )
	for( auto &g1 : G_i )
	  for( auto &g2 : G_f ){
	    tmp = "ptot="+std::to_string(mom[0])+"_"+std::to_string(mom[1])+"_"+std::to_string(mom[2])+ "/" + GAMMAS_SCATT_STR[g1]+"-"+GAMMAS_SCATT_STR[g1e] + "/" + GAMMAS_SCATT_STR[g2]+"-"+GAMMAS_SCATT_STR[g2e];
	    out.push_back(tmp);
	  }
}

//create a list with the structure of hdf5 file for 3pt. Groups order is the same of arguments order. The printed momentum is (p_i1, p_i2, p_f1+p_f2).
void print_groups_names_3pt( momList &moms, std::vector<GAMMAS_SCATT> &extG_i1, std::vector<GAMMAS_SCATT> &extG_f, std::vector<GAMMAS_SCATT> &G_i1 , std::vector<GAMMAS_SCATT> &G_i2, std::vector<GAMMAS_SCATT> &G_f,  std::vector<std::string> &out){
  std::string tmp;
  out.clear();
  for(auto &mom : moms.print_3pt() )
    for( auto &g1e : extG_i1 )
      for( auto &g3e : extG_f )
	for( auto &g1 : G_i1 )
	  for( auto &g2 : G_i2 )
	    for( auto &g3 : G_f ){
	      tmp = mom + "/" + GAMMAS_SCATT_STR[g1]+"-"+GAMMAS_SCATT_STR[g1e] + "/" + GAMMAS_SCATT_STR[g2] + "/" + GAMMAS_SCATT_STR[g3]+"-"+GAMMAS_SCATT_STR[g3e];
	      out.push_back(tmp);
	    }
}


template<typename Float>
void PLEGMA_ScattCorrelator<Float>::B_diagramms(momList &moms, PLEGMA_ScattCorrelator<Float> &srcV3, PLEGMA_ScattCorrelator<Float> &srcV2, GAMMAS_SCATT G_i2, std::vector<GAMMAS_SCATT> &Gammas_i1, std::string &outfile) {

  if( this->corr_space == POSITION_SPACE )
    PLEGMA_error("Not implemented yet\n");

  //allocate
  //int n_gammas_f1 = srcV2.shape[0];
  std::vector<GAMMAS_SCATT> aux_gammas_i2={G_i2};
  this->datasets={"B1"};
  print_groups_names_4pt(moms, Gammas_i1, aux_gammas_i2, srcV2.GList, srcV3.GList, this->groups);
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
  std::vector<std::vector<int>> mom_i1_list=moms.pi1();
  
  int offset=Gammas_i1.size()*srcV2.GList.size()*srcV3.GList.size()*HGC_localL[3]*N_SPINS*N_SPINS*2;
 
  //write B1
  for(int i_m=0; i_m<imap.size(); i_m++){
    const Float phase=2*M_PI/(Float)HGC_totalL[0]* mom_i1_list[i_m][0]*this->source_position[0]+
                      2*M_PI/(Float)HGC_totalL[1]* mom_i1_list[i_m][1]*this->source_position[1]+
                      2*M_PI/(Float)HGC_totalL[2]* mom_i1_list[i_m][2]*this->source_position[2];
    const Float tmpreim[2]={cos(phase),sin(phase)};
    srcV3.V3V2reduction( Gammas_i1, imap[i_m], srcV2, this->corr + offset*i_m, 2, true);
    x_e_cx<Float>( this->corr + offset*i_m,  tmpreim, offset/2);

  }

  this->writeHDF5(outfile);

  //write B2
  this->datasets={"B2"};
  for(int i_m=0; i_m<imap.size(); i_m++){
    const Float phase=2*M_PI/(Float)HGC_totalL[0]* mom_i1_list[i_m][0]*this->source_position[0]+
                      2*M_PI/(Float)HGC_totalL[1]* mom_i1_list[i_m][1]*this->source_position[1]+
                      2*M_PI/(Float)HGC_totalL[2]* mom_i1_list[i_m][2]*this->source_position[2];
    const Float tmpreim[2]={cos(phase),sin(phase)};

    srcV3.V3V2reduction( Gammas_i1, imap[i_m], srcV2, this->corr + offset*i_m, 0);
    x_e_cx<Float>( this->corr + offset*i_m,  tmpreim, offset/2);

  }
  this->writeHDF5(outfile);
  
  
}


template<typename Float>
void PLEGMA_ScattCorrelator<Float>::W_diagramms(momList &moms, PLEGMA_ScattCorrelator<Float> &srcV3, PLEGMA_ScattCorrelator<Float> &srcV2, GAMMAS_SCATT G_i2, std::vector<GAMMAS_SCATT> &Gammas_i1, std::string &outfile, int diagramm_index){

  if( this->corr_space == POSITION_SPACE )
    PLEGMA_error("Not implemented yet\n");


  if( (diagramm_index != 1) && (diagramm_index !=2 ) &&  (diagramm_index != 3) &&  (diagramm_index != 4)   )
    PLEGMA_error("diagramm_index %d out of range (1,2,3 or 4)\n",diagramm_index);

  this->datasets={"W"+std::to_string(diagramm_index)};

  std::vector<GAMMAS_SCATT> aux_gammas_i2={G_i2};
  print_groups_names_4pt(moms, Gammas_i1, aux_gammas_i2, srcV2.GList, srcV3.GList, this->groups);
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

  std::vector<std::vector<int>> mom_i1_list=moms.pi1();
  int offset=Gammas_i1.size()*srcV2.GList.size()*srcV3.GList.size()*HGC_localL[3]*N_SPINS*N_SPINS*2;

  for(int i_m=0; i_m<imap.size(); i_m++){
    //write W1
    if (diagramm_index == 1){
      srcV3.V3V2reduction( Gammas_i1, imap[i_m], srcV2, this->corr + offset*i_m, 2, true, true );
    }
    //write W2
    else if (diagramm_index == 2){
      srcV3.V3V2reduction_matrix( Gammas_i1, imap[i_m], srcV2, this->corr + offset*i_m, 1, false);
    }
    //write W3
    else if (diagramm_index == 3){
      srcV3.V3V2reduction_matrix( Gammas_i1, imap[i_m], srcV2, this->corr + offset*i_m, 1, false, true);
    }
    //write W4
    else {
      srcV3.V3V2reduction( Gammas_i1, imap[i_m], srcV2, this->corr + offset*i_m, 0, false, true);
    }
    const Float phase=2*M_PI/(Float)HGC_totalL[0]* mom_i1_list[i_m][0]*this->source_position[0]+
                      2*M_PI/(Float)HGC_totalL[1]* mom_i1_list[i_m][1]*this->source_position[1]+
                      2*M_PI/(Float)HGC_totalL[2]* mom_i1_list[i_m][2]*this->source_position[2];
    const Float tmpreim[2]={cos(phase),sin(phase)};
    x_e_cx<Float>( this->corr + offset*i_m,  tmpreim, offset/2);

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
                                                std::vector<GAMMAS_SCATT> &Gammas_ext_i,
                                                std::vector<GAMMAS_SCATT> &Gammas_ext_f,
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
  print_groups_names_4pt(moms, Gammas_i1, Gammas_i2, srcV2[0].GList, srcV3[0].GList, this->groups);
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

//here pi2 and Gamma_i2 are looped outside in the building of the sequential propagator. The T reduction contains ptot.
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::T_diagramms( momList &moms, PLEGMA_ScattCorrelator<Float> &T1, PLEGMA_ScattCorrelator<Float> &T3,
						 PLEGMA_ScattCorrelator<Float> &T5, GAMMAS_SCATT &G_i2,
						 std::vector<GAMMAS_SCATT> &extGammas_i1, std::vector<GAMMAS_SCATT> &extGammas_f, std::string &outfile){

  std::vector<std::vector<int>> moms_tot=moms.uniq_p(3);
  std::vector<GAMMAS_SCATT> aux_gammas_i2={G_i2,};
  
  if(!(T1.fixMomList.empty())){
    if(T1.fixMomList!=moms_tot||T3.fixMomList!=moms_tot||T5.fixMomList!=moms_tot)
      PLEGMA_error("T1,T2 or T3 have not the expected mom list\n");
  }
  else if(!(T1.fixMomVec.empty())){
    if(T1.fixMomVec!=moms_tot[0]||T3.fixMomVec!=moms_tot[0]||T5.fixMomVec!=moms_tot[0])
      PLEGMA_error("T1,T2 or T3 have not the expected mom list\n");
  }
  else
    PLEGMA_error("T1,T2 or T3 wrong mom list\n");
  
  int n_gammas_i1=T1.GList.size();
  int n_gammas_f1=T1.GList2.size();
  int n_gammas_extf=extGammas_f.size();
  
  const int tot_size= moms_tot.size()*extGammas_i1.size()*n_gammas_extf*n_gammas_i1*aux_gammas_i2.size()*n_gammas_f1*HGC_localL[3]*N_SPINS*N_SPINS*2;

  this->datasets={"T"};
  print_groups_names_3pt( moms, extGammas_i1, extGammas_f, T1.GList, aux_gammas_i2, T1.GList2, this->groups);
  this->shape={N_SPINS,N_SPINS};
  this->shape_labels="ss";
  this->initialize();

  if( this->vol_size != HGC_localL[3] )
    PLEGMA_error("PLEGMA_SC for writing must have N_moms=1\n");

  if( this->site_size*2 != tot_size/HGC_localL[3] )
    PLEGMA_error("I did some mistakes. vol_size*site_size=%d; expected= (mom=%d),(Gi1=%d),(extG1=%d),(Gi2=%d),(Gf1=%d),(extGf=%d),%d\n", this->vol_size*this->site_size,moms_tot.size(),
		 n_gammas_i1, extGammas_i1, aux_gammas_i2.size(), n_gammas_f1, n_gammas_extf, tot_size/2);

  
  const int i_GGGT = n_gammas_i1*aux_gammas_i2.size()*n_gammas_f1*HGC_localL[3];
  const int i_SS2 = N_SPINS*N_SPINS*2;
  const int i_GGGTSS2 = i_GGGT*i_SS2;
  const int d_GGGGGTSS2 = extGammas_i1.size()*n_gammas_extf*d_GGGTSS2;

  Float *srcTs[3] = {T1.getCorr(),T3.getCorr(),T5.getCorr()};

  Float *dest = this->corr;
  Float *temp = (Float *)malloc(sizeof(Float)*i_SS2);
    
  for(int i_mom; i_mom<moms_tot.size(); ++i_mom){
    for(int out_idx=0; out_idx<i_GGGT; ++out_idx){

      memset(temp,0,i_SS2*sizeof(Float));

      for(int ts=0; ts<3; ++ts)
	for(int int_idx=0; int_idx<i_SS2; ++int_idx)
	  temp[int_idx] += srcTs[ts][ i_mom*i_GGGTSS2 + out_idx*i_SS2 + int_idx ]*2.; //2T1+2T3+2T5 for all moms x gamma_i1 x gamma_i2 x gamma_f 

      //multiply second spin index with extGammas_i1
      for(int ext_gi1=0; ext_gi1 < extGammas_i1.size(); ++ext_gi1){
	//multiply first spin index with extGammas_f
	for(int ext_gf=0; ext_gf < extGammas_f.size(); ++ext_gf){
	  M_e_GNG<Float>( dest + i_mom*d_GGGGGTSS2 + (ext_gi1*n_gammas_extf+ext_gf)*i_GGGTSS2 + out_idx*d_SS2,
			  extGammas_f[ext_gf], extGammas_i1[ext_gi1], temp);
	}
      }
    }
  }

  free( temp );
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
                    PLEGMA_ScattCorrelator<Float> &srcT1,
                    PLEGMA_ScattCorrelator<Float> &srcT2,
                    std::vector<GAMMAS_SCATT> &Gammas_ext_i,
                    std::vector<GAMMAS_SCATT> &Gammas_ext_f,
                    std::string &outfile){

  //Gamma_i1, Gamma_f1 are the gammas in front of the unpaired Wilson quark
  //At the source
  std::vector<GAMMAS_SCATT> Gammas_i1=srcT1.getGList();
  //At the sink
  std::vector<GAMMAS_SCATT> Gammas_f1=srcT1.getGList2();

  this->datasets={"D"};
  //Antonino: I think we have to adjust this a bit
  print_groups_names_2pt( Gammas_ext_i, Gammas_ext_f, Gammas_ext_i, Gammas_i1, Gammas_f1, this->groups);
  this->shape={N_SPINS,N_SPINS};
  this->shape_labels="ss";
  this->initialize();

  if( this->vol_size != HGC_localL[3] )
    PLEGMA_error("PLEGMA_SC for writing must have N_moms=1\n");

  int Nmom_T1;
  //Determining the number of momentas we have
  if(!(srcT1.fixMomList.empty())){
    if(srcT1.fixMomList!=srcT2.fixMomList)
      PLEGMA_error("T1,T2 have not the the same mom list\n");
    Nmom_T1=srcT1.fixMomList.size();
  }
  else if(!(srcT1.fixMomVec.empty())){
    if(srcT1.fixMomVec!=srcT2.fixMomVec)
      PLEGMA_error("T1,T2 have not the same mom vector\n");
    Nmom_T1=1;
  }
  else
    PLEGMA_error("T1,T2 wrong mom list\n");

  //total size of destination
  const int tot_size= Nmom_T1*Gammas_i1.size()*Gammas_f1.size()*Gammas_ext_i.size()*Gammas_ext_f.size()*HGC_localL[3]*N_SPINS*N_SPINS*2;

  if( this->site_size*2 != tot_size/HGC_localL[3] )
    PLEGMA_error("I did some mistakes. vol_size*site_size=%d; expected= (mom=%d),(Gi1=%d),(Gi2=%d),(Gf2=%d),(Gf1=%d)%d\n",
                 this->vol_size*this->site_size,Nmom_T1,Gammas_f1.size(),Gammas_i1.size(),
                                                            Gammas_ext_f.size(),Gammas_ext_i.size(),tot_size/2);
  const int i_GGGGTSS2=tot_size/Nmom_T1;
  const int i_SS2=N_SPINS*N_SPINS*2;

  const int i_Gi = Gammas_ext_i.size();
  const int i_Gf = Gammas_ext_f.size();

  const int i_GGT = Gammas_i1.size()*Gammas_f1.size()*HGC_localL[3];
  const int i_GGTSS2 = Gammas_i1.size()*Gammas_f1.size()*HGC_localL[3]*N_SPINS*N_SPINS*2;


  Float *srcT1_corr = srcT1.getCorr();
  Float *srcT2_corr = srcT2.getCorr();

  Float *dest = this->corr;

  Float tmp_4t12t2[24];
  for (int i_mom=0; i_mom< Nmom_T1; ++i_mom){
    for (int f2g=0; f2g < i_Gi; ++f2g ){
      for (int i2g=0; i2g < i_Gf; ++i2g ){
        GAMMAS_SCATT gammaf2= Gammas_ext_f[f2g];
        GAMMAS_SCATT gammai2= Gammas_ext_i[i2g];

        for (int internalind=0; internalind < i_GGT; ++internalind){
          for (int i=0; i<i_SS2 ; ++i){
            tmp_4t12t2[i]=4*srcT1_corr[(i_mom*i_GGT+internalind)*i_SS2+i]+2*srcT2_corr[(i_mom*i_GGT+internalind)*i_SS2+i];
          }
          //Doing the gamma multiplication for the final indices
          M_e_GNG<Float>(&dest[i_mom*i_GGGGTSS2+(i2g*i_Gi+f2g)*i_GGTSS2+internalind*i_SS2],
                         gammai2,
                         gammaf2,
                         tmp_4t12t2);

        }
      } 
    }
  } 
  this->writeHDF5(outfile);
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::T1(std::vector<GAMMAS_SCATT> &Gammas_i, std::vector<GAMMAS_SCATT> &Gammas_f, PLEGMA_Propagator<Float> &S1, PLEGMA_Propagator<Float> &S2, PLEGMA_Propagator<Float> &S3) {
  
  if( this->corr_space==POSITION_SPACE )
    PLEGMA_error("Not implemented yet\n");
  
  int n_gammas_i= Gammas_i.size();
  int n_gammas_f= Gammas_f.size();
  
  if(n_gammas_i<=0||n_gammas_i>16)
    PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  this->GList=Gammas_i;
  
  if(n_gammas_f<=0||n_gammas_f>16)
    PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  this->GList2=Gammas_f;
  
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
void PLEGMA_ScattCorrelator<Float>::T2( std::vector<GAMMAS_SCATT> &Gammas_i, std::vector<GAMMAS_SCATT> &Gammas_f, PLEGMA_Propagator<Float> &S1, PLEGMA_Propagator<Float> &S2, PLEGMA_Propagator<Float> &S3){
  
  if( this->corr_space==POSITION_SPACE )
    PLEGMA_error("Not implemented yet\n");
   
  int n_gammas_i= Gammas_i.size();
  int n_gammas_f= Gammas_f.size();
   
  if(n_gammas_i<=0||n_gammas_i>16)
    PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  this->GList=Gammas_i;
  
  if(n_gammas_f<=0||n_gammas_f>16)
    PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  this->GList2=Gammas_f;
  
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


