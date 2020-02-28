#include <PLEGMA_ScattCorrelator.h>
#include <PLEGMA_Vector.h>
#include <PLEGMA_Propagator.h>
#include <PLEGMA_scattreductions.cuh>
#include <PLEGMA_scattreductionsPiPi.cuh>
#include <PLEGMA_utils.h>
using namespace plegma;

//--------------------------------//
//  class PLEGMA_ScattCorrelator  //
//--------------------------------//
// From PLEGMA_Correlator
//
// - initialize() -> check if nDatasets,nGroups,shape change realloc PLEGMA_FT
// - finalize() -> delete corr_*_space;

template<typename Float>
PLEGMA_ScattCorrelator<Float>::PLEGMA_ScattCorrelator(site source, int Q2_max, int totalT):
  PLEGMA_Correlator<Float>(MOMENTUM_SPACE,source,Q2_max,totalT) { ; }

template<typename Float>
PLEGMA_ScattCorrelator<Float>::PLEGMA_ScattCorrelator(site source, std::vector<int> fixMomVec, int totalT):
  PLEGMA_Correlator<Float>(MOMENTUM_SPACE,source,0,totalT) { this->setFixMomVec(fixMomVec); }

template<typename Float>
PLEGMA_ScattCorrelator<Float>::PLEGMA_ScattCorrelator(site source, std::vector<std::vector<int>> fixMomsList, int totalT):
  PLEGMA_Correlator<Float>(MOMENTUM_SPACE,source,0,totalT) { this->setFixMomList(fixMomsList); }

template<typename Float>
bool PLEGMA_ScattCorrelator<Float>::is_V24() {
  std::string exp_shape("gsssc");

  //check getSiteSize()
  if( this->shape.size() != 5 || this->shape_labels.compare(exp_shape)!=0 )
    return false;
  if( this->nDatasets()!=1 || this->nGroups()!=1 )
    return false;

  return true;
}
template<typename Float>
bool PLEGMA_ScattCorrelator<Float>::is_V3() {
  std::string exp_shape("gsc");

  //check getSiteSize()
  if( this->shape.size() != 3 || this->shape_labels.compare(exp_shape)!=0 )
    return false;
  if( this->nDatasets()!=1 || this->nGroups()!=1 )
    return false;

  return true;
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::V3( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS_SCATT> &Gammas, PLEGMA_Propagator<Float> &S){ 

  int n_gammas= Gammas.size();

  if(n_gammas<=0||n_gammas>16)
    PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  this->GList=Gammas;
  
  this->datasets={"dataset_v3"};
  this->groups={"group_v3"};
  this->shape={n_gammas,N_SPINS,N_COLS};
  this->shape_labels="gsc";
  this->initialize();
  
  const VRED V=V_3;
  
  V_reductions<V,Float,Float,Float>( *this, Phi, Gammas, S);
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::V4( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS_SCATT> &Gammas, PLEGMA_Propagator<Float> &S1, PLEGMA_Propagator<Float> &S2) {

  int n_gammas= Gammas.size();

  if(n_gammas<=0||n_gammas>16)
    PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  this->GList=Gammas;
  
  this->datasets={"dataset_v4"};
  this->groups={"group_v4"};
  this->shape={n_gammas,N_SPINS,N_SPINS,N_SPINS,N_COLS};
  this->shape_labels="gsssc";
  this->initialize();

  const VRED V=V_4;
  V_reductions<V,Float,Float,Float>( *this, Phi, Gammas, S1, S2);
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::V2( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS_SCATT> &Gammas, PLEGMA_Propagator<Float> &S1, PLEGMA_Propagator<Float> &S2) {

  int n_gammas= Gammas.size();

  if(n_gammas<=0||n_gammas>16)
    PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  this->GList=Gammas;
  
  this->datasets={"dataset_v2"};
  this->groups={"group_v2"};
  this->shape={n_gammas,N_SPINS,N_SPINS,N_SPINS,N_COLS};
  this->shape_labels="gsssc";
  this->initialize();

  const VRED V=V_2;
  V_reductions<V,Float,Float,Float>( *this, Phi, Gammas, S1, S2);
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::PhiPhi( PLEGMA_Vector<Float> &Phi_0, std::vector<GAMMAS_SCATT> &Gammas,  PLEGMA_Vector<Float> &Phi_1) {

  int n_gammas = Gammas.size();

  if(n_gammas<=0||n_gammas>16)
    PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  this->GList = Gammas;
  
  this->datasets={"PhixGxPhi"};
  this->groups={"group_PhixGxPhi"};
  this->shape={n_gammas};
  this->shape_labels="g";
  this->initialize();

  PhixGxPhi_k<Float,Float>( *this, Phi_0, Gammas, Phi_1);
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
void print_groups_names_2pt( std::string prefix, std::vector<std::vector<int>> &moms, std::vector<GAMMAS_SCATT> &extG_i, std::vector<GAMMAS_SCATT> &extG_f, std::vector<GAMMAS_SCATT> &G_i, std::vector<GAMMAS_SCATT> &G_f,  std::vector<std::string> &out){
  std::string tmp;
  out.clear();
  for(auto &mom : moms )
    for( auto &g1e : extG_i )
      for( auto &g2e : extG_f )
	for( auto &g1 : G_i )
	  for( auto &g2 : G_f ){
	    tmp = prefix + std::to_string(mom[0])+"_"+std::to_string(mom[1])+"_"+std::to_string(mom[2])+ "/" + GAMMAS_SCATT_STR[g1]+"-"+GAMMAS_SCATT_STR[g1e] + "/" + GAMMAS_SCATT_STR[g2]+"-"+GAMMAS_SCATT_STR[g2e];
	    out.push_back(tmp);
	  }
}

//create a list with the structure of hdf5 file for 2pt. Groups order is the same of arguments order. The printed momentum is the total one (p_f1+p_f2).
void print_groups_names_2pt( std::string prefix, std::vector<std::vector<int>> &moms,  std::vector<GAMMAS_SCATT> &G_i, std::vector<GAMMAS_SCATT> &G_f,  std::vector<std::string> &out){
  std::string tmp;
  out.clear();
  for(auto &mom : moms )
    for( auto &g1 : G_i )
      for( auto &g2 : G_f ){
	tmp = prefix + std::to_string(mom[0])+"_"+std::to_string(mom[1])+"_"+std::to_string(mom[2])+ "/" + GAMMAS_SCATT_STR[g1] + "/" + GAMMAS_SCATT_STR[g2];
	out.push_back(tmp);
      }
}

//create a list with the structure of hdf5 file for the pion-pion loop. Groups order is moms(from moms_red), G_i2, G_f2, extG_i1, extG_f1, G_i1, G_f1 (as in NucleonsGroups)
void print_groups_names_4pt( momList &moms_red, std::vector<std::string> &groupN, std::vector<GAMMAS_SCATT> &G_i2, std::vector<GAMMAS_SCATT> &G_f2, std::vector<std::string> &out){
  if(!moms_red.check_eq(0)) PLEGMA_error("Mmmmmh pi2 must be equal in moms\n");

  std::string tmp;
  std::string delimiter = "/";
 
  out.clear();
  for(auto &mom : moms_red.print() )
    for( auto &g1 : G_i2 )
      for( auto &g2 : G_f2 )
	for( auto &Nstring : groupN ){ //N_string => pf1=p0_p1_p2/G_i1/G_f1
	  std::string aux = Nstring.substr( Nstring.find(delimiter)+delimiter.length() ); //aux => G_i1/G_f1 
	  std::string str_G_i1 = aux.substr( 0, aux.find(delimiter)); // str_G_i1 => extG_i1-G_i1
	  std::string str_G_f1 = aux.substr( aux.find(delimiter)+delimiter.length() ); // str_G_f1 => extG_f1-G_f1
	  tmp = mom + "/" + str_G_i1 + "/" + GAMMAS_SCATT_STR[g1] + "/" + str_G_f1 + "/" + GAMMAS_SCATT_STR[g2];
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
void PLEGMA_ScattCorrelator<Float>::B_diagramms(momList &moms, PLEGMA_ScattCorrelator<Float> &srcV3, PLEGMA_ScattCorrelator<Float> &srcV2, std::vector<GAMMAS_SCATT> Gammas_ext_i, std::vector<GAMMAS_SCATT> Gammas_ext_f, GAMMAS_SCATT G_i2, std::vector<GAMMAS_SCATT> &Gammas_i1, std::string &outfile) {

  //allocate
  //int n_gammas_f1 = srcV2.shape[0];
  std::vector<GAMMAS_SCATT> aux_gammas_i2={G_i2};
  this->datasets={"B1"};
  print_groups_names_4pt(moms, Gammas_ext_i, Gammas_ext_f, Gammas_i1, aux_gammas_i2, srcV2.GList, srcV3.GList, this->groups);
  this->shape={N_SPINS,N_SPINS};
  this->shape_labels="ss";
  this->initialize();
  
  if( this->getVolSize() != HGC_localL[3] )
    PLEGMA_error("PLEGMA_SC for writing must have N_moms=1. Detected getVolSize()=%d\n",this->getVolSize());
  
  if( this->getSiteSize() != moms.size()*Gammas_ext_i.size()*Gammas_ext_f.size()*Gammas_i1.size()*srcV2.GList.size()*srcV3.GList.size()*N_SPINS*N_SPINS )
    PLEGMA_error("I did some mistakes. getVolSize()*getSiteSize()=%d; expected= (mom=%d),(Gi1=%d),(Gf2=%d),(Gf1=%d)%d\n",
		 this->getVolSize()*this->getSiteSize(),moms.size(),Gammas_i1.size(),srcV2.GList.size(),srcV3.GList.size(),
		 moms.size()*Gammas_i1.size()*srcV2.GList.size()*srcV3.GList.size()*HGC_localL[3]*N_SPINS*N_SPINS);

  const int d_SS2= N_SPINS*N_SPINS*2;
  const int d_G_ext_f = Gammas_ext_f.size();
  const int d_G_ext_i = Gammas_ext_i.size();
  const int d_GGGT     = Gammas_i1.size()*srcV2.GList.size()*srcV3.GList.size()*HGC_localL[3];
  const int d_GGGTSS2  = Gammas_i1.size()*srcV2.GList.size()*srcV3.GList.size()*HGC_localL[3]*N_SPINS*N_SPINS*2;
  const int d_GGGGGTSS2= Gammas_ext_i.size()*Gammas_ext_f.size()*Gammas_i1.size()*srcV2.GList.size()*srcV3.GList.size()*HGC_localL[3]*N_SPINS*N_SPINS*2;
  
  std::vector<std::array<int,3>> imap=moms.index_map();
  std::vector<std::vector<int>> mom_i1_list=moms.pi1();
  
  const int offset=d_GGGTSS2; 
  Float *temporary=(Float *)malloc(sizeof(Float)*offset);

 
  //write B1
  for(int i_m=0; i_m<imap.size(); i_m++){
    memset(temporary, 0, d_GGGTSS2*sizeof(Float));
    // const Float phase=2*M_PI/(Float)HGC_totalL[0]* mom_i1_list[i_m][0]*this->source_position[0]+
    //                   2*M_PI/(Float)HGC_totalL[1]* mom_i1_list[i_m][1]*this->source_position[1]+
    //                   2*M_PI/(Float)HGC_totalL[2]* mom_i1_list[i_m][2]*this->source_position[2];
    // const Float tmpreim[2]={cos(phase),sin(phase)};
    srcV3.V3V2reduction( Gammas_i1, imap[i_m], srcV2, temporary, 2, true);
    // x_e_cx<Float>( temporary, tmpreim, offset/2 );

    for (int g_i_ind=0; g_i_ind < d_G_ext_i ; ++ g_i_ind){
      for (int g_f_ind=0; g_f_ind < d_G_ext_f ; ++ g_f_ind){    
        GAMMAS_SCATT gammaf2= Gammas_ext_f[g_i_ind];
        GAMMAS_SCATT gammai2= Gammas_ext_i[g_f_ind];

        for (int internalind=0; internalind < d_GGGT; ++internalind){
          //Doing the gamma multiplication for the final indices
          M_e_GNG<Float>(this->H_elem()+i_m*d_GGGGGTSS2+(g_i_ind*d_G_ext_f+g_f_ind)*d_GGGTSS2+internalind*d_SS2,
                         gammaf2,
                         gammai2,
                         &temporary[internalind*d_SS2]);

        }
      }
    }
  }

  this->writeHDF5(outfile);

  //write B2
  this->datasets={"B2"};
  for(int i_m=0; i_m<imap.size(); i_m++){
    memset(temporary, 0, d_GGGTSS2*sizeof(Float));
    // const Float phase=2*M_PI/(Float)HGC_totalL[0]* mom_i1_list[i_m][0]*this->source_position[0]+
    //                   2*M_PI/(Float)HGC_totalL[1]* mom_i1_list[i_m][1]*this->source_position[1]+
    //                   2*M_PI/(Float)HGC_totalL[2]* mom_i1_list[i_m][2]*this->source_position[2];
    // const Float tmpreim[2]={cos(phase),sin(phase)};

    srcV3.V3V2reduction( Gammas_i1, imap[i_m], srcV2, temporary, 0);
    // x_e_cx<Float>( temporary,  tmpreim, offset/2);
    for (int g_i_ind=0; g_i_ind < d_G_ext_i ; ++ g_i_ind){
      for (int g_f_ind=0; g_f_ind < d_G_ext_f ; ++ g_f_ind){
        GAMMAS_SCATT gammaf2= Gammas_ext_f[g_i_ind];
        GAMMAS_SCATT gammai2= Gammas_ext_i[g_f_ind];

        for (int internalind=0; internalind < d_GGGT; ++internalind){
          //Doing the gamma multiplication for the final indices
          M_e_GNG<Float>(this->H_elem()+ i_m*d_GGGGGTSS2+(g_i_ind*d_G_ext_f+g_f_ind)*d_GGGTSS2+internalind*d_SS2,
                         gammaf2,
                         gammai2,
                         &temporary[internalind*d_SS2]);

        }
      }
    }

  }
  this->writeHDF5(outfile);
  
  free(temporary); 
}


template<typename Float>
void PLEGMA_ScattCorrelator<Float>::W_diagramms(momList &moms, PLEGMA_ScattCorrelator<Float> &srcV3, PLEGMA_ScattCorrelator<Float> &srcV2, std::vector<GAMMAS_SCATT> Gammas_ext_source, std::vector<GAMMAS_SCATT> Gammas_ext_sink, GAMMAS_SCATT G_i2, std::vector<GAMMAS_SCATT> &Gammas_i1, std::string &outfile, int diagramm_index){

  if( (diagramm_index != 1) && (diagramm_index !=2 ) &&  (diagramm_index != 3) &&  (diagramm_index != 4)   )
    PLEGMA_error("diagramm_index %d out of range (1,2,3 or 4)\n",diagramm_index);

  this->datasets={"W"+std::to_string(diagramm_index)};

  std::vector<GAMMAS_SCATT> aux_gammas_i2={G_i2};
  print_groups_names_4pt(moms, Gammas_ext_source, Gammas_ext_sink,  Gammas_i1, aux_gammas_i2, srcV2.GList, srcV3.GList, this->groups);
  this->shape={N_SPINS,N_SPINS};
  this->shape_labels="ss";
  this->initialize();

  if( this->getVolSize() != HGC_localL[3] )
    PLEGMA_error("PLEGMA_SC for writing must have N_moms=1\n");
 
  if( this->getSiteSize() != moms.size()*Gammas_ext_source.size()*Gammas_ext_sink.size()*Gammas_i1.size()*srcV2.GList.size()*srcV3.GList.size()*N_SPINS*N_SPINS )
    PLEGMA_error("I did some mistakes. getVolSize()*getSiteSize()=%d; expected= (mom=%d),(Gexti=%d),(Gextf=%d),(Gi1=%d),(Gf2=%d),(Gf1=%d)%d\n",
		 this->getVolSize()*this->getSiteSize(),moms.size(),Gammas_ext_source.size(),Gammas_ext_sink.size(),Gammas_i1.size(),srcV2.GList.size(),srcV3.GList.size(),
		 moms.size()*Gammas_i1.size()*srcV2.GList.size()*srcV3.GList.size()*HGC_localL[3]*N_SPINS*N_SPINS);

  std::vector<std::array<int,3>> imap=moms.index_map();
  std::vector<std::vector<int>> mom_i1_list=moms.pi1();


  const int d_SS2= N_SPINS*N_SPINS*2;
  const int d_G_ext_f = Gammas_ext_sink.size();
  const int d_G_ext_i = Gammas_ext_source.size();
  const int d_GGGT     = Gammas_i1.size()*srcV2.GList.size()*srcV3.GList.size()*HGC_localL[3];
  const int d_GGGTSS2  = Gammas_i1.size()*srcV2.GList.size()*srcV3.GList.size()*HGC_localL[3]*N_SPINS*N_SPINS*2;
  const int d_GGGGGTSS2= Gammas_ext_source.size()*Gammas_ext_sink.size()*Gammas_i1.size()*srcV2.GList.size()*srcV3.GList.size()*HGC_localL[3]*N_SPINS*N_SPINS*2;

  const int offset= d_GGGTSS2; 
  Float *temporary=(Float *)malloc(sizeof(Float)*offset);


  for(int i_m=0; i_m<imap.size(); i_m++){
    memset(temporary, 0, d_GGGTSS2*sizeof(Float));
    //write W1
    if (diagramm_index == 1){
      srcV3.V3V2reduction( Gammas_i1, imap[i_m], srcV2, temporary, 2, true, true );
    }
    //write W2
    else if (diagramm_index == 2){
      srcV3.V3V2reduction_matrix( Gammas_i1, imap[i_m], srcV2, temporary, 1, false);
    }
    //write W3
    else if (diagramm_index == 3){
      srcV3.V3V2reduction_matrix( Gammas_i1, imap[i_m], srcV2, temporary, 1, false, true);
    }
    //write W4
    else {
      srcV3.V3V2reduction( Gammas_i1, imap[i_m], srcV2, temporary, 0, false, true);
    }
    // const Float phase=2*M_PI/(Float)HGC_totalL[0]* mom_i1_list[i_m][0]*this->source_position[0]+
    //                   2*M_PI/(Float)HGC_totalL[1]* mom_i1_list[i_m][1]*this->source_position[1]+
    //                   2*M_PI/(Float)HGC_totalL[2]* mom_i1_list[i_m][2]*this->source_position[2];
    // const Float tmpreim[2]={cos(phase),sin(phase)};
    // x_e_cx<Float>( temporary,  tmpreim, offset/2);
    for (int g_i_ind=0; g_i_ind < d_G_ext_i ; ++ g_i_ind){
      for (int g_f_ind=0; g_f_ind < d_G_ext_f ; ++ g_f_ind){
        GAMMAS_SCATT gammaf2= Gammas_ext_sink[g_i_ind];
        GAMMAS_SCATT gammai2= Gammas_ext_source[g_f_ind];

        for (int internalind=0; internalind < d_GGGT; ++internalind){
          //Doing the gamma multiplication for the final indices
          M_e_GNG<Float>(this->H_elem()+ i_m*d_GGGGGTSS2+(g_i_ind*d_G_ext_f+g_f_ind)*d_GGGTSS2+internalind*d_SS2,
                         gammaf2,
                         gammai2,
                         &temporary[internalind*d_SS2]);

        }
      }
    }
  }
  this->writeHDF5(outfile);
  free(temporary);

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
  int Nmoms_f2 = this->getVolSize()/HGC_localL[3];

  
  PLEGMA_ScattCorrelator<Float> V3aux(srcV2.getSource(), srcV2.getMomList(), srcV2.getTotalT());

  Float* srcf2 = this->H_elem();//V3
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
 
      srcf1=V3aux.H_elem();
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

  const int tot_size= moms.size()*Gammas_ext_i.size()*Gammas_ext_f.size()*Gammas_i1.size()*Gammas_i2.size()*srcV2[0].GList.size()*srcV3[0].GList.size()*HGC_localL[3]*N_SPINS*N_SPINS*2;
  const int d_GGGGGGTSS2= tot_size/moms.size();
  const int d_GGGGTSS2= tot_size/moms.size()/Gammas_ext_i.size()/Gammas_ext_f.size();
  const int d_GGGGTSS = d_GGGGTSS2/2;
  const int d_GGGGT   = Gammas_i1.size()*Gammas_i2.size()*srcV2[0].GList.size()*srcV3[0].GList.size()*HGC_localL[3];
  const int d_G_ext_i = Gammas_ext_i.size();
  const int d_G_ext_f = Gammas_ext_f.size();
  const int d_SS2 = N_SPINS*N_SPINS*2 ;

  this->datasets={"Z"+std::to_string(diagramm_index)};
  print_groups_names_4pt(moms, Gammas_ext_i, Gammas_ext_f, Gammas_i1, Gammas_i2, srcV2[0].GList, srcV3[0].GList, this->groups);
  this->shape={N_SPINS,N_SPINS};
  this->shape_labels="ss";
  this->initialize();


  if( this->getVolSize() != HGC_localL[3] )
    PLEGMA_error("PLEGMA_SC for writing must have N_moms=1\n");

  if( this->getSiteSize()*2 != tot_size/HGC_localL[3] )
    PLEGMA_error("I did some mistakes. getVolSize()*getSiteSize()=%d; expected= (mom=%d),(Gi1=%d),(Gi2=%d),(Gf2=%d),(Gf1=%d)%d\n",
		 this->getVolSize()*this->getSiteSize(),moms.size(),Gammas_i1.size(),Gammas_i2.size(),
		 srcV2[0].GList.size(),srcV3[0].GList.size(),tot_size/2);

  Float * temporary= (Float *)malloc(sizeof(Float)*d_GGGGTSS2);
  Float * temporary2 =  (Float *)malloc(sizeof(Float)*d_GGGGTSS2);


  std::vector<std::array<int,3>> imap=moms.index_map();

  std::vector<std::vector<int>> mom_i1_list=moms.pi1();

  //write Z1

  Float *dest = this->H_elem();

  memset(this->H_elem(),0,tot_size*sizeof(Float));

  for(int i_m=0; i_m<imap.size(); i_m++){
    memset(temporary2, 0, d_GGGGTSS2*sizeof(Float));
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
        x_pe_cy(temporary2, g, temporary, d_GGGGTSS);

      }

    }

    // const Float phase=2*M_PI/(Float)HGC_totalL[0]* mom_i1_list[i_m][0]*this->source_position[0]+
    //                   2*M_PI/(Float)HGC_totalL[1]* mom_i1_list[i_m][1]*this->source_position[1]+
    //                   2*M_PI/(Float)HGC_totalL[2]* mom_i1_list[i_m][2]*this->source_position[2];
    // const Float tmpreim[2]={cos(phase),sin(phase)};
    // x_e_cx<Float>( temporary2,  tmpreim, d_GGGGTSS2/2);


    for (int g_i_ind=0; g_i_ind < d_G_ext_i ; ++ g_i_ind){
      for (int g_f_ind=0; g_f_ind < d_G_ext_f ; ++ g_f_ind){ 
        GAMMAS_SCATT gammaf2= Gammas_ext_f[g_i_ind];
        GAMMAS_SCATT gammai2= Gammas_ext_i[g_f_ind];

        for (int internalind=0; internalind < d_GGGGT; ++internalind){
          //Doing the gamma multiplication for the final indices
          M_e_GNG<Float>(&dest[i_m*d_GGGGGGTSS2+(g_i_ind*d_G_ext_f+g_f_ind)*d_GGGGTSS2+internalind*d_SS2],
                         gammaf2,
                         gammai2,
                         &temporary2[internalind*d_SS2]);

        }
      }
    }
  }


  free(temporary);
  free(temporary2);
  this->writeHDF5(outfile);

}

//here pi2 is looped outside in the building of the stocastic propagator. NB for moms_red I expect that pi2 is the same! Phi_0[s] is the stocastic propagator at zero momentum and spin s, Phi_1 with momentum pi2
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::P_diagramms( std::vector<int> mom_pi2, std::vector<GAMMAS_SCATT> &G_i2, std::vector<GAMMAS_SCATT> &G_f2, std::array<PLEGMA_Vector<Float>,4> &Phi_0, std::array<PLEGMA_Vector<Float>,4> &Phi_1, std::string &outfile){

  //++++++++ PION-PION +++++++++
 
  //size of temporal output for pion loop
  const int tot_size = G_i2.size()*G_f2.size()*HGC_localL[3]*2;
  std::vector<std::vector<int>> aux_mom = {mom_pi2 ,};

  print_groups_names_2pt("p_tot=", aux_mom, G_i2, G_f2, this->groups);
  this->datasets={"P"};
  this->shape={1,};
  this->shape_labels="";
  this->initialize();

  if( this->getVolSize() != HGC_localL[3] )
    PLEGMA_error("PLEGMA_SC for writing must have N_moms=1\n");

  if( this->getSiteSize()*2 != tot_size/HGC_localL[3] )
    PLEGMA_error("I did some mistakes. getVolSize()*getSiteSize()=%d; expected= (mom=1),(Gi2=%d),(Gf2=%d)%d\n", this->getVolSize()*this->getSiteSize(),
		 G_i2.size(), G_f2.size(), tot_size/2);

  memset( this->H_elem() , 0, tot_size*sizeof(Float) );
  
  //useful consts
  const int TIME = HGC_localL[3];
  const int n_gammas_f2 = G_f2.size();
  const int d_GT2 = G_f2.size()*TIME*2;

  //aux PLEGMA_SC for PhixGxPhi multiplications
  PLEGMA_ScattCorrelator<Float> pipi_aux(this->getSource(), mom_pi2, this->getTotalT());

  //loop over G_i2
  for(int gi2=0; gi2<G_i2.size(); ++gi2){
    for(int nz_e=0; nz_e<4; ++nz_e){
      int alfa = gammaInd_scatt_host[G_i2[gi2]][nz_e][0]; 
      int beta = gammaInd_scatt_host[G_i2[gi2]][nz_e][1];
      Float g[2];
      g[1] = gamma_scatt_host[G_i2[gi2]][nz_e][1];
      g[0] = gamma_scatt_host[G_i2[gi2]][nz_e][0];

      //PhixGf2xPhi
      pipi_aux.PhiPhi( Phi_0[beta], G_f2, Phi_1[alfa]); //T x N_moms x n_gammas_f2

      for( int time=0; time<TIME; ++time)
	for( int gf2=0; gf2<n_gammas_f2; ++gf2)
	  x_pe_cy( this->H_elem() + gi2*d_GT2 + gf2*2*TIME + time*2, g, pipi_aux.H_elem() + time*n_gammas_f2*2 + gf2*2, 1);
    }
  }
  this->writeHDF5(outfile);
}


//here pi2 is looped outside in the building of the stocastic propagator. NB for moms_red I expect that pi2 is the same! Phi_0[s] is the stocastic propagator at zero momentum and spin s, Phi_1 with momentum pi2
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::M_diagramms( momList &moms, momList &moms_red, PLEGMA_ScattCorrelator<Float> &CorrNucleon, std::vector<GAMMAS_SCATT> &G_i2, std::vector<GAMMAS_SCATT> &G_f2, std::array<PLEGMA_Vector<Float>,4> &Phi_0, std::array<PLEGMA_Vector<Float>,4> &Phi_1, std::string &outfile){

  //++++++++ PION-PION +++++++++

  //extract moms
  if(!moms_red.check_eq(0)) PLEGMA_error("Mmmmmh something is not going as expected\n");
  std::vector<int> mom_pi2 = moms_red.pi(0)[0]; 
  std::vector<std::vector<int>> moms_pf2 = moms_red.uniq_p(2);

  //size of temporal output for pion loop
  const int pp_size = moms_pf2.size()*G_i2.size()*G_f2.size()*HGC_localL[3]*2;
  Float *temp_pp = (Float*)malloc( pp_size*sizeof(Float) );
  memset( temp_pp, 0, pp_size*sizeof(Float) );
  
  //useful consts
  const int TIME = HGC_localL[3];
  const int n_gammas_f2 = G_f2.size();
  const int d_GGT2 = G_i2.size()*G_f2.size()*TIME*2;
  const int d_GT2 = G_f2.size()*TIME*2;
  const int i_MG2 = moms_pf2.size()*G_f2.size()*2;
  const int d_GG = G_i2.size()*G_f2.size();

  //aux PLEGMA_SC for PhixGxPhi multiplications
  PLEGMA_ScattCorrelator pipi_aux(this->getSource(), moms_pf2, this->getTotalT());

  //loop over G_i2
  for(int gi2=0; gi2<G_i2.size(); ++gi2){
    for(int nz_e=0; nz_e<4; ++nz_e){
      int alfa = gammaInd_scatt_host[G_i2[gi2]][nz_e][0]; 
      int beta = gammaInd_scatt_host[G_i2[gi2]][nz_e][1];
      Float g[2];
      g[1] = gamma_scatt_host[G_i2[gi2]][nz_e][1];
      g[0] = gamma_scatt_host[G_i2[gi2]][nz_e][0];

      //PhixGf2xPhi
      pipi_aux.PhiPhi( Phi_0[beta], G_f2, Phi_1[alfa]); //T x N_moms x n_gammas_f2

      for( int i_pf2=0; i_pf2<moms_pf2.size(); ++i_pf2 )
	for( int time=0; time<TIME; ++time)
	  for( int gf2=0; gf2<n_gammas_f2; ++gf2)
	    x_pe_cy( temp_pp + i_pf2*d_GGT2 + gi2*d_GT2 + gf2*2*TIME + time*2, g, pipi_aux.H_elem() + time*i_MG2 + i_pf2*n_gammas_f2*2 + gf2*2, 1);
    }
  }

  //++++++++ NUCLEON-NUCLEON +++++++++

  //extract vector p_f1
  std::vector<std::vector<int>> moms_pf1_red = moms_red.uniq_p(1); //list of pf1 momenta needed here
  std::vector<std::vector<int>> moms_pf1 = moms.uniq_p(1); //list of pf1 in Nucleons PLEGMA_SC
  std::vector<int> i_pf1s = moms.u_posix( 1, moms_pf1_red ); //list of positions of moms_pf1_red momenta in moms_pf1 array
  std::vector<std::array<int,3>> map = moms_red.index_map();
  
  //Data in CorrNucleon
  Float *temp_NN = CorrNucleon.H_elem();//N_moms_pf1*extG_i1*extG_f1*G_i1*G_f1*T*S*S;

  //Groups structure of CorrNucleon (only the gamma structure)
  std::vector<std::string> groupsN = CorrNucleon.getGroups();//N_moms_pf1*extG_i1*extG_f1*G_i1*G_f1
  int N_GGGG = groupsN.size()/moms_pf1.size();
  groupsN.resize(N_GGGG);


  //++++++++++ NN x PIPI ++++++++++++

  //initialize output
  print_groups_names_4pt( moms_red, groupsN, G_i2, G_f2, this->groups);
  this->datasets={"M"};
  this->shape={N_SPINS,N_SPINS};
  this->shape_labels="ss";
  this->initialize();

  //output size and checks
  const int tot_size = moms_red.size()*N_GGGG*d_GGT2*N_SPINS*N_SPINS;
 
  if( this->getVolSize() != HGC_localL[3] )
    PLEGMA_error("PLEGMA_SC for writing must have N_moms=1\n");

  if( this->getSiteSize()*2 != tot_size/HGC_localL[3] )
    PLEGMA_error("I did some mistakes. getVolSize()*getSiteSize()=%d; expected= (mom=%d),(Gi2=%d),(Gf2=%d),(nucleonGGGG=%d)%d\n", this->getVolSize()*this->getSiteSize(), moms_red.size(),
		 G_i2.size(), G_f2.size(), N_GGGG, tot_size/2);

  //useful consts
  const int o_SS2 = N_SPINS*N_SPINS*2;
  const int o_TSS2 = TIME*o_SS2;
  const int o_GGGGTSS2 = N_GGGG*o_TSS2;
  const int o_GGGGGGTSS2 = d_GG*o_GGGGTSS2;


  //put output to zero
  memset( this->H_elem(), 0, tot_size*sizeof(Float)); 

  //for each momentum in moms_red
  for( int i_mom=0; i_mom<moms_red.size(); ++i_mom){
    int i_pf1 = i_pf1s[map[i_mom][1]]; //position of pf1 in moms_pf1 (tempNN)
    int i_pf2 = map[i_mom][2]; //position of pf2 in tempPP
    // const Float phase = 2*M_PI/(Float)HGC_totalL[0]*(moms_pf2[i_pf2][0]-mom_pi2[0])*this->source_position[0]+
    //                     2*M_PI/(Float)HGC_totalL[1]*(moms_pf2[i_pf2][1]-mom_pi2[1])*this->source_position[1]+
    //                     2*M_PI/(Float)HGC_totalL[2]*(moms_pf2[i_pf2][2]-mom_pi2[2])*this->source_position[2]; //not pi1 becayse i*pf1*sourcepos already multiplied in Nucleons results
    // const Float tmpreim[2]={cos(phase),sin(phase)};


    for( int inner1=0; inner1<d_GG; ++inner1 ){ //for G_i2xG_f2
      for( int inner2=0; inner2<N_GGGG; ++inner2 ){ //for extG_i1 x extG_f1 x G_i1 x G_f1
	for( int t=0; t<TIME; ++t){ //for time 
	  x_pe_cy( this->H_elem() + i_mom*o_GGGGGGTSS2 + inner1*o_GGGGTSS2 + inner2*o_TSS2 + t*o_SS2,
		   temp_pp + i_pf2*d_GGT2 + (inner1*TIME + t)*2,
		   temp_NN + i_pf1*o_GGGGTSS2 + inner2*o_TSS2 + t*o_SS2,
		   N_SPINS*N_SPINS);
	}
      }
    }
    //multiplication by the p_i1 phase
    // x_e_cx<Float>( this->H_elem() + i_mom*o_GGGGGGTSS2, tmpreim, o_GGGGGGTSS2/2);
  }

  free( temp_pp );
  this->writeHDF5(outfile);
}

//Nucleon correlator. This function should be called outside the p_i2 loop, with Ts computed using the entire list of unique p_f1s. N.B: we multiply the output by exp(i * x_sourcepos * p_f1);
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::N_diagramms( PLEGMA_ScattCorrelator<Float> &T1, PLEGMA_ScattCorrelator<Float> &T2, std::vector<GAMMAS_SCATT> &extG_i1, std::vector<GAMMAS_SCATT> &extG_f1, std::string &outfile){

  //checks between T1 T2
  if(T1.getMomList()!=T2.getMomList())
    PLEGMA_error("T1,T2 have not the the same mom list\n");
  
  if((T1.GList!=T2.GList)||(T1.GList2!=T2.GList2))
    PLEGMA_error("T1,T2 wrong gamma list\n");


  //extract array mom
  std::vector<std::vector<int>> moms_pf1 = T1.getMomList();

  //size of final output for NN
  const int n_gammas_i1 = T1.GList.size();
  const int n_gammas_f1 = T1.GList2.size();
  const int n_gammas_extf1 = extG_f1.size();
  const int tot_size = moms_pf1.size()*extG_i1.size()*extG_f1.size()*n_gammas_i1*n_gammas_f1*HGC_localL[3]*N_SPINS*N_SPINS*2;
  const int src_size = HGC_localL[3]*moms_pf1.size()*n_gammas_i1*n_gammas_f1*N_SPINS*N_SPINS*2;
  
  //initialize output
  print_groups_names_2pt("pf1=", moms_pf1, extG_i1, extG_f1, T1.GList, T1.GList2, this->groups);
  this->datasets={"N"};
  this->shape={N_SPINS,N_SPINS};
  this->shape_labels="ss";
  this->initialize();

  //checks on memory
  if( this->getVolSize() != HGC_localL[3] )
    PLEGMA_error("PLEGMA_SC for writing must have N_moms=1\n");

  if( this->getSiteSize()*2 != tot_size/HGC_localL[3] )
    PLEGMA_error("I did some mistakes. getVolSize()*getSiteSize()=%d; expected= (mom=%d),(extGi1=%d),(extGf1=%d),(Gi1=%d),(Gf1=%d),%d\n", this->getVolSize()*this->getSiteSize(), moms_pf1.size(),
		 extG_i1.size(), extG_f1.size(), n_gammas_i1, n_gammas_f1, tot_size/2);

  //put memory = 0
  memset( this->H_elem(), 0, tot_size*sizeof(Float));

  //define temp array
  Float *temp = (Float *)malloc(src_size*sizeof(Float));
  for( int idx=0; idx<src_size; ++idx)
    temp[idx] = T1.H_elem()[idx] + T2.H_elem()[idx];

  //define constants
  const int s_SS2 = N_SPINS*N_SPINS*2;
  const int s_GGSS2 = n_gammas_i1*n_gammas_f1*s_SS2;
  const int s_MGGSS2 = moms_pf1.size()*s_GGSS2;
  const int d_TSS2 = HGC_localL[3]*s_SS2;
  const int d_GGTSS2 = n_gammas_i1*n_gammas_f1*d_TSS2;
  const int d_GGGGTSS2 = extG_i1.size()*extG_f1.size()*d_GGTSS2;

  
  for( int i_mom=0; i_mom<moms_pf1.size(); ++i_mom){
    // const Float phase=2*M_PI/(Float)HGC_totalL[0]*(moms_pf1[i_mom][0])*this->source_position[0]+
    //                   2*M_PI/(Float)HGC_totalL[1]*(moms_pf1[i_mom][1])*this->source_position[1]+
    //                   2*M_PI/(Float)HGC_totalL[2]*(moms_pf1[i_mom][2])*this->source_position[2];
    // const Float tmpreim[2]={cos(phase),sin(phase)};

    for( int t=0; t<HGC_localL[3]; ++t){
      for( int i_gg=0; i_gg<n_gammas_i1*n_gammas_f1; ++i_gg ){
	
	//multiply second spin index with extGammas_i1
	for(int ext_gi1=0; ext_gi1 < extG_i1.size(); ++ext_gi1){
	  //multiply first spin index with extGammas_f
	  for(int ext_gf1=0; ext_gf1 < extG_f1.size(); ++ext_gf1){
	    M_e_GNG<Float>( this->H_elem() + i_mom*d_GGGGTSS2 + (ext_gi1*n_gammas_extf1+ext_gf1)*d_GGTSS2 + i_gg*d_TSS2 + t*s_SS2,
			    extG_f1[ext_gf1], extG_i1[ext_gi1], temp + t*s_MGGSS2 + i_mom*s_GGSS2 + i_gg*s_SS2);
	  }
	}
      }
    }
    // x_e_cx<Float>( this->H_elem() + i_mom*d_GGGGTSS2,  tmpreim, d_GGGGTSS2/2);
  }

  free( temp );
  this->writeHDF5(outfile);
  
}


//here pi2 and Gamma_i2 are looped outside in the building of the sequential propagator. NB for moms I expect that pi2 is the same! The T reduction contains ptot.
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::T_diagramms( momList &moms, PLEGMA_ScattCorrelator<Float> &T1, PLEGMA_ScattCorrelator<Float> &T3,
						 PLEGMA_ScattCorrelator<Float> &T5, GAMMAS_SCATT &G_i2,
						 std::vector<GAMMAS_SCATT> &extGammas_i1, std::vector<GAMMAS_SCATT> &extGammas_f, std::string &outfile){

  std::vector<std::vector<int>> moms_tot=moms.uniq_p(3);
  std::vector<GAMMAS_SCATT> aux_gammas_i2={G_i2,};

  if(!moms.check_eq(0)) PLEGMA_error("Mmmmmh something is not going as expected\n");
  std::vector<int> p_i2=moms.pi(0)[0];
  
  if(T1.getMomList()!=moms_tot||T3.getMomList()!=moms_tot||T5.getMomList()!=moms_tot)
    PLEGMA_error("T1,T2 or T3 have not the expected mom list\n");
  
  int n_gammas_i1=T1.GList.size();
  int n_gammas_f1=T1.GList2.size();
  int n_gammas_extf=extGammas_f.size();
  
  const int tot_size= moms_tot.size()*extGammas_i1.size()*n_gammas_extf*n_gammas_i1*aux_gammas_i2.size()*n_gammas_f1*HGC_localL[3]*N_SPINS*N_SPINS*2;

  this->datasets={"T"};
  print_groups_names_3pt( moms, extGammas_i1, extGammas_f, T1.GList, aux_gammas_i2, T1.GList2, this->groups);
  this->shape={N_SPINS,N_SPINS};
  this->shape_labels="ss";
  this->initialize();

  if( this->getVolSize() != HGC_localL[3] )
    PLEGMA_error("PLEGMA_SC for writing must have N_moms=1\n");

  if( this->getSiteSize()*2 != tot_size/HGC_localL[3] )
    PLEGMA_error("I did some mistakes. getVolSize()*getSiteSize()=%d; expected= (mom=%d),(Gi1=%d),(extG1=%d),(Gi2=%d),(Gf1=%d),(extGf=%d),%d\n", this->getVolSize()*this->getSiteSize(),moms_tot.size(),
		 n_gammas_i1, extGammas_i1.size(), aux_gammas_i2.size(), n_gammas_f1, n_gammas_extf, tot_size/2);

  
  const int i_GGGT = n_gammas_i1*aux_gammas_i2.size()*n_gammas_f1*HGC_localL[3];
  const int i_SS2 = N_SPINS*N_SPINS*2;
  const int i_GGGTSS2 = i_GGGT*i_SS2;
  const int d_GGGGGTSS2 = extGammas_i1.size()*n_gammas_extf*i_GGGTSS2;

  Float *srcTs[3] = {T1.H_elem(),T3.H_elem(),T5.H_elem()};

  Float *dest = this->H_elem();
  Float *temp = (Float *)malloc(sizeof(Float)*i_SS2);
    
  for(int i_mom=0; i_mom<moms_tot.size(); ++i_mom){
    
    // const Float phase=2*M_PI/(Float)HGC_totalL[0]*(moms_tot[i_mom][0]-p_i2[0])*this->source_position[0]+
    //                   2*M_PI/(Float)HGC_totalL[1]*(moms_tot[i_mom][1]-p_i2[1])*this->source_position[1]+
    //                   2*M_PI/(Float)HGC_totalL[2]*(moms_tot[i_mom][2]-p_i2[2])*this->source_position[2];
    // const Float tmpreim[2]={cos(phase),sin(phase)};

    for(int out_idx=0; out_idx<i_GGGT; ++out_idx){

      memset(temp,0,i_SS2*sizeof(Float));

      for(int ts=0; ts<3; ++ts)
	for(int int_idx=0; int_idx<i_SS2; ++int_idx)
	  temp[int_idx] += srcTs[ts][ i_mom*i_GGGTSS2 + out_idx*i_SS2 + int_idx ]*2.; //2T1+2T3+2T5 for all moms x gamma_i1 x gamma_i2 x gamma_f 

      //multiply second spin index with extGammas_i1
      for(int ext_gi1=0; ext_gi1 < extGammas_i1.size(); ++ext_gi1){
	//multiply first spin index with extGammas_f
	for(int ext_gf=0; ext_gf < extGammas_f.size(); ++ext_gf){
	  M_e_GNG<Float>( dest + i_mom*d_GGGGGTSS2 + (ext_gi1*n_gammas_extf+ext_gf)*i_GGGTSS2 + out_idx*i_SS2,
			  extGammas_f[ext_gf], extGammas_i1[ext_gi1], temp);
	}
      }
    }
    // x_e_cx<Float>( dest+i_mom*d_GGGGGTSS2,  tmpreim, d_GGGGGTSS2/2);
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
  int Nmoms_f2 = this->getVolSize()/HGC_localL[3];

  PLEGMA_ScattCorrelator<Float> V3aux(srcV2.getSource(), srcV2.getMomList(), srcV2.getTotalT());
  
  Float* srcf2 = this->H_elem();//V3
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
      srcf1=V3aux.H_elem();
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
                    PLEGMA_ScattCorrelator<Float> &T1,
                    PLEGMA_ScattCorrelator<Float> &T2,
                    std::vector<GAMMAS_SCATT> &extG_i,
                    std::vector<GAMMAS_SCATT> &extG_f,
                    std::string &outfile){

  
  //checks between T1 T2
  if(T1.getMomList()!=T2.getMomList())
    PLEGMA_error("T1,T2 have not the the same mom list\n");

  if((T1.GList!=T2.GList)||(T1.GList2!=T2.GList2))
    PLEGMA_error("T1,T2 wrong gamma list\n");

  //extract array mom
  std::vector<std::vector<int>> moms_tot = T1.getMomList();
  
  //size of final output for NN
  const int n_gammas_i = T1.GList.size();
  const int n_gammas_f = T1.GList2.size();
  const int n_gammas_extf = extG_f.size();
  const int tot_size = moms_tot.size()*extG_i.size()*extG_f.size()*n_gammas_i*n_gammas_f*HGC_localL[3]*N_SPINS*N_SPINS*2;
  const int src_size = HGC_localL[3]*moms_tot.size()*n_gammas_i*n_gammas_f*N_SPINS*N_SPINS*2;

  
  //initialize output
  print_groups_names_2pt("pf1=", moms_tot, extG_i, extG_f, T1.GList, T1.GList2, this->groups);
  this->datasets={"D"};
  this->shape={N_SPINS,N_SPINS};
  this->shape_labels="ss";
  this->initialize();

  //checks on memory
  if( this->getVolSize() != HGC_localL[3] )
    PLEGMA_error("PLEGMA_SC for writing must have N_moms=1\n");

  if( this->getSiteSize()*2 != tot_size/HGC_localL[3] )
    PLEGMA_error("I did some mistakes. getVolSize()*getSiteSize()=%d; expected= (mom=%d),(extGi1=%d),(extGf1=%d),(Gi1=%d),(Gf1=%d),%d\n", this->getVolSize()*this->getSiteSize(), moms_tot.size(),
		 extG_i.size(), extG_f.size(), n_gammas_i, n_gammas_f, tot_size/2);

  //put memory = 0
  memset( this->H_elem(), 0, tot_size*sizeof(Float));

  //define temp array
  Float *temp = (Float *)malloc(src_size*sizeof(Float));
  for( int idx=0; idx<src_size; ++idx)
    temp[idx] = 4*T1.H_elem()[idx] + 2*T2.H_elem()[idx];

  //define constants
  const int s_SS2 = N_SPINS*N_SPINS*2;
  const int s_GGSS2 = n_gammas_i*n_gammas_f*s_SS2;
  const int s_MGGSS2 = moms_tot.size()*s_GGSS2;
  const int d_TSS2 = HGC_localL[3]*s_SS2;
  const int d_GGTSS2 = n_gammas_i*n_gammas_f*d_TSS2;
  const int d_GGGGTSS2 = extG_i.size()*extG_f.size()*d_GGTSS2;

  for( int i_mom=0; i_mom<moms_tot.size(); ++i_mom){
    // const Float phase=2*M_PI/(Float)HGC_totalL[0]*(moms_tot[i_mom][0])*this->source_position[0]+
    //                   2*M_PI/(Float)HGC_totalL[1]*(moms_tot[i_mom][1])*this->source_position[1]+
    //                   2*M_PI/(Float)HGC_totalL[2]*(moms_tot[i_mom][2])*this->source_position[2];
    // const Float tmpreim[2]={cos(phase),sin(phase)};

    for( int t=0; t<HGC_localL[3]; ++t){
      for( int i_gg=0; i_gg<n_gammas_i*n_gammas_f; ++i_gg ){
	
	//multiply second spin index with extGammas_i1
	for(int ext_gi=0; ext_gi< extG_i.size(); ++ext_gi){
	  //multiply first spin index with extGammas_f
	  for(int ext_gf=0; ext_gf < extG_f.size(); ++ext_gf){
	    M_e_GNG<Float>( this->H_elem() + i_mom*d_GGGGTSS2 + (ext_gi*n_gammas_extf+ext_gf)*d_GGTSS2 + i_gg*d_TSS2 + t*s_SS2,
			    extG_f[ext_gf], extG_i[ext_gi], temp + t*s_MGGSS2 + i_mom*s_GGSS2 + i_gg*s_SS2);
         
	  }
	}
      }
    }
    // x_e_cx<Float>( this->H_elem() + i_mom*d_GGGGTSS2,  tmpreim, d_GGGGTSS2/2);
  }

  free( temp );
  this->writeHDF5(outfile);
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::T1(std::vector<GAMMAS_SCATT> &Gammas_i, std::vector<GAMMAS_SCATT> &Gammas_f, PLEGMA_Propagator<Float> &S1, PLEGMA_Propagator<Float> &S2, PLEGMA_Propagator<Float> &S3) {
  
  int n_gammas_i= Gammas_i.size();
  int n_gammas_f= Gammas_f.size();
  
  if(n_gammas_i<=0||n_gammas_i>16)
    PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  this->GList=Gammas_i;
  
  if(n_gammas_f<=0||n_gammas_f>16)
    PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  this->GList2=Gammas_f;
  
  this->datasets={"dataset_t1"};
  this->groups={"group_t1"};
  this->shape={n_gammas_i,n_gammas_f, N_SPINS,N_SPINS};
  this->initialize();
  
  const TRED T=T_1;
  T_reductions<T,Float,Float>( *this, Gammas_i, Gammas_f, S1, S2, S3);
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::T2( std::vector<GAMMAS_SCATT> &Gammas_i, std::vector<GAMMAS_SCATT> &Gammas_f, PLEGMA_Propagator<Float> &S1, PLEGMA_Propagator<Float> &S2, PLEGMA_Propagator<Float> &S3){
  
  int n_gammas_i= Gammas_i.size();
  int n_gammas_f= Gammas_f.size();
   
  if(n_gammas_i<=0||n_gammas_i>16)
    PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  this->GList=Gammas_i;
  
  if(n_gammas_f<=0||n_gammas_f>16)
    PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  this->GList2=Gammas_f;
  
  this->datasets={"dataset_t2"};
  this->groups={"group_t2"};
  this->shape={n_gammas_i,n_gammas_f, N_SPINS,N_SPINS};
  this->initialize();

  const TRED T=T_2;
  T_reductions<T,Float,Float>( *this, Gammas_i, Gammas_f, S1, S2, S3);
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::contract_GxV2_checks( PLEGMA_ScattCorrelator<Float> &srcV2){
  if(!srcV2.is_V24()) PLEGMA_error("srcV2 seems not to have V2like shape\n");
  
  if( this->getMomList() != srcV2.getMomList() ) PLEGMA_error("src and dest must have same momenta\n");

  //allocate
  int n_gammas = srcV2.GList.size();
  this->GList=srcV2.GList;
  this->datasets={"absorbvectorfromV24"};
  this->groups={"absorbvectorfromV24"};
  this->shape={n_gammas,N_SPINS,N_COLS};
  this->shape_labels="gsc";
  this->initialize();
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::absorb_fromV24_checks( PLEGMA_ScattCorrelator<Float> &srcV2, int alfa, int beta){
  if(!srcV2.is_V24()) PLEGMA_error("srcV2 seems not to have V2like shape\n");
  
  //check index ranges
  if( alfa<0 || alfa>=N_SPINS ) PLEGMA_error("%d out of range\n", alfa);
  if( beta<0 || beta>=N_SPINS ) PLEGMA_error("%d out of range\n", beta);

  
  //allocate
  int n_gammas = srcV2.shape[0];

  if( this->getMomList() != srcV2.getMomList() ) PLEGMA_error("src and dest must have same momenta. getMomList() detected.\n");
  
  this->GList=srcV2.GList;
  this->datasets={"absorbvectorfromV24"};
  this->groups={"absorbvectorfromV24"};
  this->shape={n_gammas,N_SPINS,N_COLS};
  this->shape_labels="gsc";
  this->initialize();
  
  size_t exp_size = srcV2.getVolSize()*n_gammas*N_SPINS*N_COLS;
  if( this->getTotalSize() != exp_size )
    PLEGMA_error("total size %d != expected_size %d\n",this->getTotalSize(),exp_size);
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::absorbspinmatrix_fromV24_checks( PLEGMA_ScattCorrelator<Float> &srcV2, int alfa){
  if(!srcV2.is_V24()) PLEGMA_error("srcV2 seems not to have V2like shape\n");
  
  //check index ranges
  if( alfa<0 || alfa>=N_SPINS ) PLEGMA_error("%d out of range\n", alfa);


  //allocate
  int n_gammas = srcV2.shape[0];

  if( this->getMomList() != srcV2.getMomList() ) PLEGMA_error("src and dest must have same momenta\n");
  
  this->GList=srcV2.GList;
  this->datasets={"absorbmatrixv24"};
  this->groups={"absorbmatrixv24"};
  this->shape={n_gammas,N_SPINS,N_SPINS,N_COLS};
  this->shape_labels="gssc";
  this->initialize();

  size_t exp_size = srcV2.getVolSize()*n_gammas*N_SPINS*N_SPINS*N_COLS;
  if( this->getTotalSize() != exp_size )
    PLEGMA_error("total size %d != expected_size %d\n",this->getTotalSize(),exp_size);
}



//V3_a^l*G_ab*V3_b^l
//template<typename Float>
//void PLEGMA_ScattCorrelator<Float>::V3V3reduction( PLEGMA_ScattCorrelator<Float> &srcV2, int alfa, int beta){

template class PLEGMA_ScattCorrelator<float>;
template class PLEGMA_ScattCorrelator<double>;


