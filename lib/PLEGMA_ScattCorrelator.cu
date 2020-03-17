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

//##################
//#  Constructors  #
//##################

template<typename Float>
PLEGMA_ScattCorrelator<Float>::PLEGMA_ScattCorrelator(site source, int Q2_max, int totalT):
  PLEGMA_Correlator<Float>(MOMENTUM_SPACE,source,Q2_max,totalT) { ; }

template<typename Float>
PLEGMA_ScattCorrelator<Float>::PLEGMA_ScattCorrelator(site source, std::vector<int> fixMomVec, int totalT):
  PLEGMA_Correlator<Float>(MOMENTUM_SPACE,source,0,totalT) { this->setFixMomVec(fixMomVec); }

template<typename Float>
PLEGMA_ScattCorrelator<Float>::PLEGMA_ScattCorrelator(site source, std::vector<std::vector<int>> fixMomsList, int totalT):
  PLEGMA_Correlator<Float>(MOMENTUM_SPACE,source,0,totalT) { this->setFixMomList(fixMomsList); }


//#########################
//#  Auxiliary functions  #
//#########################
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::setOffsets( ){
  int g_count=0;
  ranges.clear();
      
  for( auto &l : labels ){
    switch(l){
    case('p'): if(pList.empty()||N_p==0) PLEGMA_error( "p detected with PList empty\n"); else ranges.push_back(N_p); break;
    case('t'): assert(this->corr_mom_space); ranges.push_back( this->corr_mom_space->DimT() ); break;
    case('m'): assert(this->corr_mom_space); ranges.push_back( this->corr_mom_space->Nmoms() ); break;
    case('g'): assert(g_count<GList.size()); ranges.push_back( GList[g_count].size() ); g_count++; break;
    case('s'): ranges.push_back( N_SPINS ); break;
    case('c'): ranges.push_back( N_COLS ); break;
    default: PLEGMA_error( "Label %c not recognized\n", l );
    }	
  }

  offsets.clear();
  for( int ir=1; ir<ranges.size(); ++ir ){
    int offset = std::accumulate(ranges.begin()+ir, ranges.end(), 2, std::multiplies<int>());
    offsets.push_back(offset);
  }
  offsets.push_back(2);
}
  
template<typename Float>
bool PLEGMA_ScattCorrelator<Float>::check_reduction( VRED V ) {
  std::string exp_shape = ( V==V_3 ) ? "tmgsc" : "tmgsssc";
  
  //check getSiteSize()
  if( this->shape.size()+2 != exp_shape.length() || this->labels.compare(exp_shape)!=0 )
    return false;
  if( this->nDatasets()!=1 || this->nGroups()!=1 )
    return false;

  //Only 1 gamma list
  if( this->GList.size() != 1 )
    return false;

  //PList useless
  if( !this->pList.empty() )
    return false;
  return true;
}

template<typename Float>
bool PLEGMA_ScattCorrelator<Float>::check_reduction( TRED T ) {
  std::string exp_shape = "tmggss";
  //check getSiteSize()
  if( this->shape.size()+2 != exp_shape.length() || this->labels.compare(exp_shape)!=0 )
    return false;
  if( this->nDatasets()!=1 || this->nGroups()!=1 )
    return false;

  //2 gamma list
  if( this->GList.size() != 2 )
    return false;

  //PList useless
  if( !this->pList.empty() )
    return false;
  return true;
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::contract_GxV2_checks( PLEGMA_ScattCorrelator<Float> &srcV2){
  if(!srcV2.check_reduction(V_2)) PLEGMA_error("srcV2 seems not to have V2like shape\n");
  
  if( this->getMomList() != srcV2.getMomList() ) PLEGMA_error("src and dest must have same momenta\n");

 
  int n_gammas = srcV2.GList[0].size();

   //check momenta
  if( this->getMomList() != srcV2.getMomList() ) PLEGMA_error("src and dest must have same momenta. getMomList() detected.\n");

  //allocate
  this->GList = srcV2.GList;
  this->datasets={"absorbvectorfromV24"};
  this->groups={"absorbvectorfromV24"};
  this->shape={n_gammas,N_SPINS,N_COLS};
  this->labels="tmgsc";
  this->initialize();
  this->setOffsets();
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::absorb_fromV24_checks( PLEGMA_ScattCorrelator<Float> &srcV2, int alfa, int beta){
  if(!srcV2.check_reduction(V_2)) PLEGMA_error("srcV2 seems not to have V2like shape\n");
  
  //check index ranges
  if( alfa<0 || alfa>=N_SPINS ) PLEGMA_error("%d out of range\n", alfa);
  if( beta<0 || beta>=N_SPINS ) PLEGMA_error("%d out of range\n", beta);

    
  //allocate
  int n_gammas = srcV2.GList[0].size();

  //check momenta
  if( this->getMomList() != srcV2.getMomList() ) PLEGMA_error("src and dest must have same momenta. getMomList() detected.\n");
  
  this->GList = srcV2.GList;
  this->datasets={"absorbvectorfromV24"};
  this->groups={"absorbvectorfromV24"};
  this->shape={n_gammas,N_SPINS,N_COLS};
  this->labels="tmgsc";
  this->initialize();
  this->setOffsets();
  
  size_t exp_size = srcV2.getVolSize()*n_gammas*N_SPINS*N_COLS;
  //check size
  if( this->getTotalSize() != exp_size )
    PLEGMA_error("total size %d != expected_size %d\n",this->getTotalSize(),exp_size);
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::absorbspinmatrix_fromV24_checks( PLEGMA_ScattCorrelator<Float> &srcV2, int alfa){
  if(!srcV2.check_reduction(V_2)) PLEGMA_error("srcV2 seems not to have V2like shape\n");
  
  //check index ranges
  if( alfa<0 || alfa>=N_SPINS ) PLEGMA_error("%d out of range\n", alfa);


  //allocate
  int n_gammas = srcV2.GList[0].size();

  //check momentum
  if( this->getMomList() != srcV2.getMomList() ) PLEGMA_error("src and dest must have same momenta\n");
  
  this->GList = srcV2.GList;
  this->datasets={"absorbmatrixv24"};
  this->groups={"absorbmatrixv24"};
  this->shape = {n_gammas,N_SPINS,N_SPINS,N_COLS};
  this->labels="tmgssc";
  this->initialize();
  this->setOffsets();
  
  size_t exp_size = srcV2.getVolSize()*n_gammas*N_SPINS*N_SPINS*N_COLS;
  if( this->getTotalSize() != exp_size )
    PLEGMA_error("total size %d != expected_size %d\n",this->getTotalSize(),exp_size);
}


//#####################
//#  Main reductions  #
//#####################

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::V3( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS_SCATT> &Gammas, PLEGMA_Propagator<Float> &S){ 

  int n_gammas= Gammas.size();
  this->GList.clear();
  
  if(n_gammas<=0||n_gammas>16)
    PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  this->GList.push_back(Gammas);
  
  this->datasets={"dataset_v3"};
  this->groups={"group_v3"};
  this->shape={n_gammas,N_SPINS,N_COLS};
  this->labels="tmgsc";
  this->initialize();
  this->setOffsets();
  
  for(int i=0; i<3; ++i)
    assert(this->source[i]==0);
  
  V_reductions<Float,Float,Float>(V_3, *this, Phi, Gammas, S);
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::V4( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS_SCATT> &Gammas, PLEGMA_Propagator<Float> &S1, PLEGMA_Propagator<Float> &S2) {

  int n_gammas= Gammas.size();
  this->GList.clear();
  
  if(n_gammas<=0||n_gammas>16)
    PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  this->GList.push_back(Gammas);
  
  this->datasets={"dataset_v4"};
  this->groups={"group_v4"};
  this->shape={n_gammas,N_SPINS,N_SPINS,N_SPINS,N_COLS};
  this->labels="tmgsssc";
  this->initialize();
  this->setOffsets();

  for(int i=0; i<3; ++i)
    assert(this->source[i]==0);

  V_reductions<Float,Float,Float>( V_4, *this, Phi, Gammas, S1, S2);
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::V2( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS_SCATT> &Gammas, PLEGMA_Propagator<Float> &S1, PLEGMA_Propagator<Float> &S2) {

  int n_gammas= Gammas.size();
  this->GList.clear();
  
  if(n_gammas<=0||n_gammas>16)
    PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  this->GList.push_back(Gammas);
  
  this->datasets={"dataset_v2"};
  this->groups={"group_v2"};
  this->shape={n_gammas,N_SPINS,N_SPINS,N_SPINS,N_COLS};
  this->labels="tmgsssc";
  this->initialize();
  this->setOffsets();

  for(int i=0; i<3; ++i)
    assert(this->source[i]==0);

  V_reductions<Float,Float,Float>(V_2, *this, Phi, Gammas, S1, S2);
}


template<typename Float>
void PLEGMA_ScattCorrelator<Float>::T1(std::vector<GAMMAS_SCATT> &Gammas_i, std::vector<GAMMAS_SCATT> &Gammas_f, PLEGMA_Propagator<Float> &S1, PLEGMA_Propagator<Float> &S2, PLEGMA_Propagator<Float> &S3) {
  
  int n_gammas_i= Gammas_i.size();
  int n_gammas_f= Gammas_f.size();
  this->GList.clear();
  
  if(n_gammas_i<=0||n_gammas_i>16)
    PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  this->GList.push_back(Gammas_i);
  
  if(n_gammas_f<=0||n_gammas_f>16)
    PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  this->GList.push_back(Gammas_f);
  
  this->datasets={"dataset_t1"};
  this->groups={"group_t1"};
  this->labels="tmggss";
  this->shape={n_gammas_i,n_gammas_f, N_SPINS,N_SPINS};
  this->initialize();
  this->setOffsets();

  for(int i=0; i<3; ++i)
    assert(this->source[i]==0);

  T_reductions<Float,Float>( T_1, *this, Gammas_i, Gammas_f, S1, S2, S3);
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::T2( std::vector<GAMMAS_SCATT> &Gammas_i, std::vector<GAMMAS_SCATT> &Gammas_f, PLEGMA_Propagator<Float> &S1, PLEGMA_Propagator<Float> &S2, PLEGMA_Propagator<Float> &S3){
  
  int n_gammas_i= Gammas_i.size();
  int n_gammas_f= Gammas_f.size();
  this->GList.clear();
   
  if(n_gammas_i<=0||n_gammas_i>16)
    PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  this->GList.push_back(Gammas_i);
  
  if(n_gammas_f<=0||n_gammas_f>16)
    PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  this->GList.push_back(Gammas_f);
  
  this->datasets={"dataset_t2"};
  this->groups={"group_t2"};
  this->labels="tmggss";
  this->shape={n_gammas_i,n_gammas_f, N_SPINS,N_SPINS};
  this->initialize();
  this->setOffsets();

  for(int i=0; i<3; ++i)
    assert(this->source[i]==0);

  T_reductions<Float,Float>( T_2, *this, Gammas_i, Gammas_f, S1, S2, S3);
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::PhiPhi( PLEGMA_Vector<Float> &Phi_0, std::vector<GAMMAS_SCATT> &Gammas,  PLEGMA_Vector<Float> &Phi_1) {

  int n_gammas = Gammas.size();
  this->GList.clear();
 
  if(n_gammas<=0||n_gammas>16)
    PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  this->GList.push_back(Gammas);
  
  this->datasets={"PhixGxPhi"};
  this->groups={"group_PhixGxPhi"};
  this->shape={n_gammas};
  this->labels="tmg";
  this->initialize();
  this->setOffsets();

  for(int i=0; i<3; ++i)
    assert(this->source[i]==0);

  PhixGxPhi_k<Float,Float>( *this, Phi_0, Gammas, Phi_1);
}


//###############################
//#  Combination of reductions  #
//###############################

//Contracts V3^a_c x Gammas_cd x V2^a_{abd}. Gammas_i list of gammas between V3, V2. 
//called by aux PLEGMA_ScattCorrelator with shape PTGGGGGGSS
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::V3V2reduction( PLEGMA_ScattCorrelator<Float> &srcV3,PLEGMA_ScattCorrelator<Float> &srcV2, int index_abs, bool transp, int g0, bool transpgamma, Float* factor) {

  //checks
  std::string exp_shape="ptmggggggss";
  if( this->labels!=exp_shape )
    PLEGMA_error("V3V2reduction Expected shape ptmggggggss not the one detected\n");

  if( !srcV2.check_reduction(V_2) ) PLEGMA_error("SrcV2 object does not seem a V2like object\n");
  if( !srcV3.check_reduction(V_3) ) PLEGMA_error("SrcV3 object does not seem a V3like object\n");
  if( srcV2.getGList()[0] != this->GList[4] ) PLEGMA_error("G_f1 doesn't match\n");
  if( srcV3.getGList()[0] != this->GList[5] ) PLEGMA_error("G_f2 doesn't match\n");

  int n_gammas_exti = this->GList[0].size();
  int n_gammas_extf = this->GList[1].size();
  int n_gammas_i1 = this->GList[2].size();
  int n_gammas_i2 = this->GList[3].size();
  int n_gammas_f1 = this->GList[4].size();
  int n_gammas_f2 = this->GList[5].size();
  int TIME = this->localT();
  
  int Nmoms_f1 = srcV2.Nmoms();
  int Nmoms_f2 = srcV3.Nmoms();

  //PLEGMA_ScattCorrelator<Float> V3aux(srcV2.getSource(), srcV2.getMomList(), srcV2.getTotalT());
  Float V3aux[N_SPINS*N_COLS*2];
  std::vector<std::array<int,3>> imap = this->pList.index_map();

  Float temp[2*N_SPINS*N_SPINS];
  
  for(int i_m=0; i_m<imap.size(); i_m++){
    int i_mom_f1 = imap[i_m][1];
    int i_mom_f2 = imap[i_m][2];
    for(int t=0; t < TIME; ++t){
      for (int g_exti=0; g_exti < n_gammas_exti ; ++ g_exti){
	for (int g_extf=0; g_extf < n_gammas_extf ; ++ g_extf){    
	  GAMMAS_SCATT egammai = this->GList[0][g_exti];
	  GAMMAS_SCATT egammaf = this->GList[1][g_extf];
	  for (int g1=0 ; g1 < n_gammas_i1 ; ++g1 ){//pi
	    for (int g2=0 ; g2 < n_gammas_f1 ; ++g2 ){//pf1
	      for (int g3=0 ; g3 < n_gammas_f2 ; ++g3 ){//pf2
		for (int alfa=0; alfa < N_SPINS; ++alfa ){
		  for (int beta=0; beta < N_SPINS; ++beta ){
		    int spins = (transp) ? (beta*N_SPINS+alfa)*2 : (alfa*N_SPINS+beta)*2;
		    switch(index_abs){
		    case 0: absorb_fromV24<0,Float>( V3aux, srcV2.Corr(t,i_mom_f1,g2), alfa, beta ); break;
		    case 1: absorb_fromV24<1,Float>( V3aux, srcV2.Corr(t,i_mom_f1,g2), alfa, beta ); break;
		    case 2: absorb_fromV24<2,Float>( V3aux, srcV2.Corr(t,i_mom_f1,g2), alfa, beta ); break;
		    }
		    V_M_V<Float>( srcV3.Corr(t,i_mom_f2,g3), V3aux,
				  this->GList[2][g1], transpgamma, temp + spins);
		  }//beta
		}//alfa
		//PLEGMA_printf("DEBUG: --- loop (%d,%d,%d,%d,%d,%d,%d) ---  V3V2reduction V_M_V done\n",i_m,t,g_exti,g_extf,g1,g2,g3);
		if(factor!=NULL){
		  Float aux;
		  for(int ss=0; ss<N_SPINS*N_SPINS; ++ss){
		    aux = temp[2*ss+0]*factor[0] - temp[2*ss+1]*factor[1];
		    temp[2*ss+1] = temp[2*ss+0]*factor[1] + temp[2*ss+1]*factor[0];
		    temp[2*ss+0] = aux;
		  }
		}
		
		//multiplication with external gammas NB written here! mod in M_pe_GNG
		M_pe_GNG<Float>( this->Corr(i_m,t,0,g_exti,g_extf,g1,g0,g2,g3),
				egammaf, egammai, temp );
		//PLEGMA_printf("DEBUG: --- loop (%d,%d,%d,%d,%d,%d,%d) ---  V3V2reduction M_pe_GNG done\n",i_m,t,g_exti,g_extf,g1,g2,g3);
	      }//Gf2
	    }//Gf1
	  }//Gi1	
	}//Gextf
      }//Gexti
    }//time
  }//mom
}


template<typename Float>
void PLEGMA_ScattCorrelator<Float>::V3V2reduction_matrix( PLEGMA_ScattCorrelator<Float> &srcV3, PLEGMA_ScattCorrelator<Float> &srcV2, int index_abs, bool transp,  int g0, bool transpgamma, Float* factor) {

  //checks
  std::string exp_shape="ptmggggggss";
  if( this->labels!=exp_shape )
    PLEGMA_error("V3V2reduction_matrix Expected shape ptmggggggss not the one detected (%s)\n",+this->labels.c_str());
  
  if( !srcV2.check_reduction(V_2) ) PLEGMA_error("SrcV2 object does not seem a V2like object\n");
  if( !srcV3.check_reduction(V_3) ) PLEGMA_error("SrcV3 object does not seem a V3like object\n");
  if( srcV2.getGList()[0] != this->GList[4] ) PLEGMA_error("G_f1 doesn't match\n");
  if( srcV3.getGList()[0] != this->GList[5] ) PLEGMA_error("G_f2 doesn't match\n");

  int n_gammas_exti = this->GList[0].size();
  int n_gammas_extf = this->GList[1].size();
  int n_gammas_i1 = this->GList[2].size();
  int n_gammas_i2 = this->GList[3].size();
  int n_gammas_f1 = this->GList[4].size();
  int n_gammas_f2 = this->GList[5].size();
  int TIME = this->localT();
  
  int Nmoms_f1 = srcV2.Nmoms();
  int Nmoms_f2 = srcV3.Nmoms();

  //PLEGMA_ScattCorrelator<Float> V3aux(srcV2.getSource(), srcV2.getMomList(), srcV2.getTotalT());
  Float V3aux[N_SPINS*N_SPINS*N_COLS*2];
  
  std::vector<std::array<int,3>> imap = this->pList.index_map();


  Float temp_colorvector[N_COLS*2];
  Float temp[N_SPINS*N_SPINS*2];
  //PLEGMA_printf("DEBUG: V3V2reduction_matrix - prel done\n");

  for(int i_m=0; i_m<imap.size(); i_m++){
    int i_mom_f1 = imap[i_m][1];
    int i_mom_f2 = imap[i_m][2];
    for(int t=0; t < TIME; ++t){
      for (int g_exti=0; g_exti < n_gammas_exti ; ++ g_exti){
	for (int g_extf=0; g_extf < n_gammas_extf ; ++ g_extf){    
	  GAMMAS_SCATT egammai = this->GList[0][g_exti];
	  GAMMAS_SCATT egammaf = this->GList[1][g_extf];
	  for (int g1=0 ; g1 < n_gammas_i1 ; ++g1 ){//pi
	    for (int g2=0 ; g2 < n_gammas_f1 ; ++g2 ){//pf1
	      for (int g3=0 ; g3 < n_gammas_f2 ; ++g3 ){//pf2
		for (int alfa=0; alfa < N_SPINS; ++alfa ){
		  for (int beta=0; beta < N_SPINS; ++beta ){
		    int spins = (transp) ? (beta*N_SPINS+alfa)*2 : (alfa*N_SPINS+beta)*2;
		    switch(index_abs){
		    case 0: absorbspinmatrix_fromV24<0,Float>( V3aux, srcV2.Corr(t,i_mom_f1,g2), alfa ); break;
		    case 1: absorbspinmatrix_fromV24<1,Float>( V3aux, srcV2.Corr(t,i_mom_f1,g2), alfa ); break;
		    case 2: absorbspinmatrix_fromV24<2,Float>( V3aux, srcV2.Corr(t,i_mom_f1,g2), alfa ); break;
		    }
		    //color vector from Tr[G_i1 V2]
		    V_TR_MM<Float>( V3aux, this->GList[2][g1], transpgamma, temp_colorvector);
		    temp[spins]=0.;
		    temp[spins+1]=0.;

		    //colorvector x V3
		    for (int coloridx=0; coloridx<3; ++coloridx){
		      temp[spins+0] +=
			+temp_colorvector[2*coloridx+0]*srcV3.Corr(t,i_mom_f2,g3,beta,coloridx)[0]
			-temp_colorvector[2*coloridx+1]*srcV3.Corr(t,i_mom_f2,g3,beta,coloridx)[1];
		      temp[spins+1] +=
			+temp_colorvector[2*coloridx+1]*srcV3.Corr(t,i_mom_f2,g3,beta,coloridx)[0]
			+temp_colorvector[2*coloridx+0]*srcV3.Corr(t,i_mom_f2,g3,beta,coloridx)[1];
		    }//coloridx
		  }//beta
		}//alfa
		//PLEGMA_printf("DEBUG: --- loop (%d,%d,%d,%d,%d,%d,%d) ---  V3V2reduction_matrix V_M_V done\n",i_m,t,g_exti,g_extf,g1,g2,g3);
		if(factor!=NULL){
		  Float aux;
		  for(int ss=0; ss<N_SPINS*N_SPINS; ++ss){
		    aux = temp[2*ss+0]*factor[0] - temp[2*ss+1]*factor[1];
		    temp[2*ss+1] = temp[2*ss+0]*factor[1] + temp[2*ss+1]*factor[0];
		    temp[2*ss+0] = aux;
		  }
		}

		//multiplication with external gammas NB: written here!!
		M_pe_GNG<Float>( this->Corr(i_m,t,0,g_exti,g_extf,g1,g0,g2,g3),
				egammaf, egammai, temp );
		//PLEGMA_printf("DEBUG: --- loop (%d,%d,%d,%d,%d,%d,%d) ---  V3V2reduction_matrix M_pe_GNG done\n",i_m,t,g_exti,g_extf,g1,g2,g3);
		
	      }//Gf2
	    }//Gf1
	  }//Gi1	
	}//Gextf
      }//Gexti
    }//time
  }//mom
}


//#########################
//#  Initialize diagrams  #
//#########################

//2pt --> "P", 2Gammas
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::initialize_diagram( momList &momenta, std::vector<GAMMAS_SCATT> &G_i2, std::vector<GAMMAS_SCATT> &G_f2, std::string name_of_diagram, bool inM){

  assert( name_of_diagram=="P" );

  //Gamma list
  this->GList.clear();
  std::vector<GAMMAS_SCATT> tmpG =  apply_gamma5_scatt_gamma(G_i2,RIGHT);
  this->GList.push_back( tmpG );
  tmpG = apply_gamma5_scatt_gamma(G_f2,RIGHT);
  this->GList.push_back( tmpG );

  //Description
  std::string tmp="";
  // for( auto &gi2: G_i2 )
  //   for( auto &gf2: G_f2 )
  //     tmp += GAMMAS_SCATT_STR[gi2]+"_"+GAMMAS_SCATT_STR[gf2]+", ";
  this->description=tmp;

  //momList
  this->setPList( momenta );
  std::vector<std::vector<int>> momlist = (inM) ? momenta.uniq_p(2) : momenta.uniq_p(0);
  this->N_p = momlist.size();

  //Groups
  this->groups.clear();  
  tmp="";
  for( auto &mom : momlist ){
    tmp = "pi2=" + std::to_string(mom[0])+"_"+std::to_string(mom[1])+"_"+std::to_string(mom[2])+"_";
    tmp += "pf2=" + std::to_string(mom[0])+"_"+std::to_string(mom[1])+"_"+std::to_string(mom[2]);
    this->groups.push_back(tmp);
  }

  //Dataset
  this->datasets={name_of_diagram,};

  //Shape
  this->shape = { (int)(this->GList[0].size())*(int)(this->GList[1].size()) };

  //initialize
  this->initialize();
  assert( this->nGroups()==this->N_p );
  assert( this->Nmoms()==1 );
  
  if( this->getVolSize() != this->localT() )
    PLEGMA_error("PLEGMA_SC for writing must have N_moms=1\n");

  int tot_size = this->N_p * this->localT() * this->GList[0].size() * this->GList[1].size();
  if( this->getSiteSize() != tot_size/this->localT() )
     PLEGMA_error("I did some mistakes. getVolSize()*getSiteSize()=%d; expected= (mom=1),(Gi2=%d),(Gf2=%d)%d\n", this->getVolSize()*this->getSiteSize(),
   		 this->GList[0].size(), this->GList[1].size(), tot_size);

  //Offsets
  this->labels="ptmgg";
  this->setOffsets();

  this->clear_output(true);
}

//2pt --> "N","D", 4Gammas
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::initialize_diagram( momList &momenta, std::vector<GAMMAS_SCATT> &eG_i, std::vector<GAMMAS_SCATT> &eG_f, std::vector<GAMMAS_SCATT> &G_i1, std::vector<GAMMAS_SCATT> &G_f1, std::string name_of_diagram){

  assert( name_of_diagram=="N" || name_of_diagram=="D");
  //Gamma list
  this->GList.clear();
  this->GList.push_back( eG_i );
  this->GList.push_back( eG_f );
  this->GList.push_back( G_i1 );
  this->GList.push_back( G_f1 );

  //Description
  std::string tmp="";
  // for( auto &egi: eG_i )
  //   for( auto &egf: eG_f )
  //     for( auto &gi1: G_i1 )
  // 	for( auto &gf1: G_f1 )
  // 	  tmp += GAMMAS_SCATT_STR[gi1]+"-"+GAMMAS_SCATT_STR[egi]+"_"+GAMMAS_SCATT_STR[gf1]+"-"+GAMMAS_SCATT_STR[egf]+", ";
  this->description=tmp;
    
  //momList
  this->setPList( momenta );
  std::vector<std::vector<int>> momlist = (name_of_diagram=="N") ? momenta.uniq_p(1) : momenta.uniq_p(3);
  this->N_p = momlist.size();

  //Groups
  this->groups.clear();  
  tmp="";
  std::string prefix1 = (name_of_diagram=="N") ? "pi1=" : "pi=";
  std::string prefix2 = (name_of_diagram=="N") ? "pf1=" : "pf=";
  for( auto &mom : momlist ){
    tmp = prefix1 + std::to_string(mom[0])+"_"+std::to_string(mom[1])+"_"+std::to_string(mom[2])+"_";
    tmp += prefix2 + std::to_string(mom[0])+"_"+std::to_string(mom[1])+"_"+std::to_string(mom[2]);
    this->groups.push_back(tmp);
  }

  //Dataset
  this->datasets={name_of_diagram,};

  //Shape
  this->shape={(int)(this->GList[0].size())*
	       (int)(this->GList[1].size())*
	       (int)(this->GList[2].size())*
	       (int)(this->GList[3].size()),
	       N_SPINS*N_SPINS};

  //initialize
  this->initialize();
  assert( this->nGroups()==this->N_p );
  assert( this->Nmoms()==1 );
  
  if( this->getVolSize() != this->localT() )
    PLEGMA_error("PLEGMA_SC for writing must have N_moms=1\n");

  //Offsets
  this->labels="ptmggggss";
  this->setOffsets();

  int tot_size = this->N_p * this->localT() * this->GList[0].size() * this->GList[1].size() * this->GList[2].size() * this->GList[3].size() *N_SPINS*N_SPINS;
  if( this->getSiteSize() != tot_size/this->localT() )
    PLEGMA_error("I did some mistakes. getVolSize()*getSiteSize()=%d; expected= (mom=%d),(extGi1=%d),(extGf1=%d),(Gi1=%d),(Gf1=%d),%d\n", this->getVolSize()*this->getSiteSize(),
		 this->N_p, this->GList[0].size(), this->GList[1].size(), this->GList[2].size(), this->GList[3].size(), tot_size);

  this->clear_output(true);
}

//3pt --> "T", 5Gammas
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::initialize_diagram( momList &momenta, std::vector<GAMMAS_SCATT> &eG_i, std::vector<GAMMAS_SCATT> &eG_f, std::vector<GAMMAS_SCATT> &G_i1, std::vector<GAMMAS_SCATT> &G_i2, std::vector<GAMMAS_SCATT> &G_f, std::string name_of_diagram){

  assert( name_of_diagram=="T" );

 //Gamma list
  this->GList.clear();
  this->GList.push_back( eG_i );
  this->GList.push_back( eG_f );
  this->GList.push_back( G_i1 );
  this->GList.push_back( G_i2 );
  this->GList.push_back( G_f );

  //Description
  std::string tmp="";
  // for( auto &egi: eG_i )
  //   for( auto &egf: eG_f )
  //     for( auto &gi1: G_i1 )
  // 	for( auto &gi2: G_i2 )
  // 	  for( auto &gf: G_f )
  // 	   tmp += GAMMAS_SCATT_STR[gi1]+"-"+GAMMAS_SCATT_STR[egi]+"_"+GAMMAS_SCATT_STR[gi2]+"_"+GAMMAS_SCATT_STR[gf]+"-"+GAMMAS_SCATT_STR[egf]+", ";
  this->description=tmp;
  
  //momList
  this->setPList( momenta );
  std::vector<std::vector<int>> momlist = momenta.uniq_p(3);
  this->N_p = momlist.size();

  //Groups
  this->groups = momenta.print_3pt();

  //Dataset
  this->datasets = {name_of_diagram,};

  //Shape
  this->shape={(int)(this->GList[0].size())*
	       (int)(this->GList[1].size())*
	       (int)(this->GList[2].size())*
	       (int)(this->GList[3].size())*
	       (int)(this->GList[4].size()),
	       N_SPINS*N_SPINS};

  //initialize
  this->initialize();
  assert( this->nGroups()==this->N_p );
  assert( this->Nmoms()==1 );

  int exp_site_size = this->N_p * this->GList[0].size() * this->GList[1].size() * this->GList[2].size() * this->GList[3].size() * this->GList[4].size() * N_SPINS * N_SPINS;
  int exp_site_size1 = this->nGroups();
  for( auto &s: this->shape )
    exp_site_size1 *= s;
  
  if( this->getVolSize() != this->localT() )
    PLEGMA_error("PLEGMA_SC for writing must have N_moms=1\n");

  if( this->getSiteSize() != exp_site_size || exp_site_size!=exp_site_size1 )
     PLEGMA_error("I did some mistakes. getSiteSize()=%d; exp_site_size= (mom=%d,ngroups=%d),(extGi1=%d),(extGf1=%d),(Gi1=%d),(Gi2=%d),(Gf1=%d) %d, exp_site_size1=%d\n", this->getSiteSize(), this->N_p,
		  this->nGroups(), this->GList[0].size(), this->GList[1].size(), this->GList[2].size(), this->GList[3].size(), this->GList[4].size(), exp_site_size,  exp_site_size1);

  //Offsets
  this->labels="ptmgggggss";
  this->setOffsets();

  this->clear_output(true);
}

//4pt --> "B1","B2","W1","W2","W3","W4","Z1","Z2","Z3","Z4","M",  6Gammas
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::initialize_diagram( momList &momenta, std::vector<GAMMAS_SCATT> &eG_i, std::vector<GAMMAS_SCATT> &eG_f, std::vector<GAMMAS_SCATT> &G_i1, std::vector<GAMMAS_SCATT> &G_i2, std::vector<GAMMAS_SCATT> &G_f1, std::vector<GAMMAS_SCATT> &G_f2, std::string name_of_diagram){
  
  char letter = name_of_diagram.at(0);
  assert( (letter=='M') || (letter=='B') || (letter=='W') || (letter=='Z') );
  if( letter != 'M' ){
    char number = name_of_diagram.at(1);
    if( letter=='B' ) assert( (number>'0') && (number<'3') );
    else assert( (number>'0') && (number<'5') );
  }
  
  //Gamma list
  this->GList.clear();
  this->GList.push_back( eG_i );
  this->GList.push_back( eG_f );
  this->GList.push_back( G_i1 );
  std::vector<GAMMAS_SCATT> tmpG = (letter == 'Z') ? apply_gamma5_scatt_gamma(G_i2,RIGHT) : G_i2;
  this->GList.push_back( tmpG );
  this->GList.push_back( G_f1 );
  tmpG = (letter == 'Z') ? apply_gamma5_scatt_gamma(G_f2,LEFT) : G_f2;
  this->GList.push_back( tmpG );

  //Description
  std::string tmp="";
  for( auto &egi: eG_i )
    for( auto &egf: eG_f )
      for( auto &gi1: G_i1 )
	for( auto &gi2: G_i2 )
	  for( auto &gf1: G_f1 )
	    for( auto &gf2: G_f2 )
	      tmp += GAMMAS_SCATT_STR[gi1]+"-"+GAMMAS_SCATT_STR[egi]+"_"+GAMMAS_SCATT_STR[gi2]+"_"+GAMMAS_SCATT_STR[gf1]+"-"+GAMMAS_SCATT_STR[egf] + "_" + GAMMAS_SCATT_STR[gf2] + ", ";
  //this->description=tmp;
  
  //momList
  this->setPList( momenta );
  this->N_p = momenta.size();

  //Groups
  this->groups = momenta.print();

  //Dataset
  this->datasets = {name_of_diagram};

  //Shape
  this->shape={(int)(this->GList[0].size())*
	       (int)(this->GList[1].size())*
	       (int)(this->GList[2].size())*
	       (int)(this->GList[3].size())*
	       (int)(this->GList[4].size())*
	       (int)(this->GList[5].size()),
	       N_SPINS*N_SPINS};

  //initialize
  this->initialize();
  assert( this->nGroups()==this->N_p );
  assert( this->Nmoms()==1 );
  
  if( this->getVolSize() != this->localT() )
    PLEGMA_error("PLEGMA_SC for writing must have N_moms=1\n");

  // if( this->getSiteSize() != moms.size()*Gammas_ext_i.size()*Gammas_ext_f.size()*Gammas_i1.size()*srcV2.GList.size()*srcV3.GList.size()*N_SPINS*N_SPINS )
  //   PLEGMA_error("I did some mistakes. getVolSize()*getSiteSize()=%d; expected= (mom=%d),(Gi1=%d),(Gf2=%d),(Gf1=%d)%d\n",
  // 		 this->getVolSize()*this->getSiteSize(),moms.size(),Gammas_i1.size(),srcV2.GList.size(),srcV3.GList.size(),
  // 		 moms.size()*Gammas_i1.size()*srcV2.GList.size()*srcV3.GList.size()*this->localT()*N_SPINS*N_SPINS);

  //Offsets
  this->labels="ptmggggggss";
  this->setOffsets();

  this->clear_output(true);
}

//4pt pi1



//P_diagrams
//


//T_diagrams
// const Float phase=2*M_PI/(Float)HGC_totalL[0]*(moms_tot[i_mom][0]-p_i2[0])*this->source[0]+
//                     2*M_PI/(Float)HGC_totalL[1]*(moms_tot[i_mom][1]-p_i2[1])*this->source[1]+
//                     2*M_PI/(Float)HGC_totalL[2]*(moms_tot[i_mom][2]-p_i2[2])*this->source[2];
//   const Float tmpreim[2]={cos(phase),sin(phase)};
// const int tot_size= moms_tot.size()*extGammas_i1.size()*n_gammas_extf*n_gammas_i1*aux_gammas_i2.size()*n_gammas_f1*this->localT()*N_SPINS*N_SPINS*2;


//Ddiagrams
// const Float phase=2*M_PI/(Float)HGC_totalL[0]*(moms_tot[i_mom][0])*this->source[0]+
//                   2*M_PI/(Float)HGC_totalL[1]*(moms_tot[i_mom][1])*this->source[1]+
//                   2*M_PI/(Float)HGC_totalL[2]*(moms_tot[i_mom][2])*this->source[2];
// const Float tmpreim[2]={cos(phase),sin(phase)};

//##############
//#  Diagrams  #
//##############
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::B_diagramms(PLEGMA_ScattCorrelator<Float> &srcV3, PLEGMA_ScattCorrelator<Float> &srcV2, int ig_i2, int diagram_index, bool accum) {

  if((diagram_index!=1)&&(diagram_index!=2)) PLEGMA_error("diagram_index for B 1 or 2, detected: %d\n", diagram_index);
  if(ig_i2 >= this->GList[3].size()) PLEGMA_error("ig_i2 = %d but Gi2 list size is %d\n", ig_i2, this->GList[3].size() );

  this->clear_output(!accum, 6, ig_i2); 
  
  int aux_idx = (diagram_index==1) ? 2 : 0;

  this->V3V2reduction( srcV3, srcV2, aux_idx, (diagram_index==1), ig_i2);

}


template<typename Float>
void PLEGMA_ScattCorrelator<Float>::W_diagramms(PLEGMA_ScattCorrelator<Float> &srcV3, PLEGMA_ScattCorrelator<Float> &srcV2, int ig_i2, int diagramm_index, bool accum){

  if( (diagramm_index != 1) && (diagramm_index !=2 ) &&  (diagramm_index != 3) &&  (diagramm_index != 4)   )
    PLEGMA_error("diagramm_index %d out of range (1,2,3 or 4)\n",diagramm_index);
  if(ig_i2 >= this->GList[3].size()) PLEGMA_error("ig_i2 = %d but Gi2 list size is %d\n", ig_i2, this->GList[3].size() );

  this->clear_output(!accum, 6, ig_i2); 

  
  //write W1
  if (diagramm_index == 1){
    this->V3V2reduction( srcV3, srcV2, 2, true, ig_i2, true );
  }
  else if (diagramm_index == 2){
    this->V3V2reduction_matrix( srcV3, srcV2, 1, false, ig_i2);
  }
  //write W3
  else if (diagramm_index == 3){
    this->V3V2reduction_matrix( srcV3, srcV2, 1, false, ig_i2, true);
  }
  //write W4
  else {
    this->V3V2reduction( srcV3, srcV2, 0, false, ig_i2, true);
  }
}



template<typename Float>
void PLEGMA_ScattCorrelator<Float>::Z_diagramms(std::array<PLEGMA_ScattCorrelator<Float>,4> (&srcV3),
                                                std::array<PLEGMA_ScattCorrelator<Float>,4> (&srcV2),
						int diagramm_index, bool accum ){

  if( (diagramm_index != 1) && (diagramm_index !=2 ) &&  (diagramm_index != 3) &&  (diagramm_index != 4)   )
    PLEGMA_error("diagramm_index %d out of range (1,2,3 or 4)\n",diagramm_index);

  this->clear_output(!accum); 

  for (int g2=0; g2<this->GList[3].size(); ++g2 ){
    GAMMAS_SCATT gammai2 = this->GList[3][g2];
    for (int n=0; n<4; ++n){
      int kappa = gammaInd_scatt_host[gammai2][n][0]; 
      int lambda =  gammaInd_scatt_host[gammai2][n][1];
      Float g[2];
      g[1] = gamma_scatt_host[gammai2][n][1];
      g[0] = gamma_scatt_host[gammai2][n][0];

      //Z1
      if (diagramm_index==1){
	this->V3V2reduction( srcV3[lambda], srcV2[kappa], 1, false, g2, true, g);
      }
      //Z2
      else if (diagramm_index ==2){
	this->V3V2reduction_matrix( srcV3[lambda], srcV2[kappa], 0, false, g2, true, g);
      }
      //Z3
      else if (diagramm_index ==3){
	this->V3V2reduction_matrix( srcV3[lambda], srcV2[kappa], 1, false, g2, true, g);
      }
      //Z4
      else {
	this->V3V2reduction( srcV3[lambda], srcV2[kappa], 0, false, g2, true, g);
      }

    }//n -> nonzero elems of Gi2
  }//loop over G_i2 matrix

}

//here pi2 is looped outside in the building of the stocastic propagator. NB for moms_red I expect that pi2 is the same! Phi_0[s] is the stocastic propagator at zero momentum and spin s, Phi_1 with momentum pi2
//GList only 2 gammas G_i2, G_f2
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::P_diagramms( std::array<PLEGMA_Vector<Float>,4> &Phi_0, std::array<PLEGMA_Vector<Float>,4> &Phi_1, bool accum, int pi){
  
  this->clear_output(!accum); 

  std::vector<std::vector<int>> momlist = this->pList.uniq_p(pi);
  assert( this->N_p == momlist.size() );
  //++++++++ PION-PION +++++++++

  //mom_pi2 can be 1 mom or a list of moms
  
  //aux PLEGMA_SC for PhixGxPhi multiplications
  site source=site({0,0,0,this->getSource()[3]});
  PLEGMA_ScattCorrelator<Float> pipi_aux(source, momlist, this->getTotalT());

  int N_moms = pipi_aux.Nmoms();
  int n_gammas_i2 = this->GList[0].size();
  int n_gammas_f2 = this->GList[1].size();
  int TIME = this->localT();

  //small check
  if( pi==0 ) assert( N_moms == 1 );
  else assert( pi==2 );
	
  //loop over G_i2
  for(int gi2=0; gi2<n_gammas_i2; ++gi2){
    GAMMAS_SCATT G_i2=this->GList[0][gi2];
    for(int nz_e=0; nz_e<4; ++nz_e){
      int alfa = gammaInd_scatt_host[G_i2][nz_e][0]; 
      int beta = gammaInd_scatt_host[G_i2][nz_e][1];
      Float g[2];
      g[1] = gamma_scatt_host[G_i2][nz_e][1];
      g[0] = gamma_scatt_host[G_i2][nz_e][0];

      //PhixGf2xPhi
      pipi_aux.PhiPhi( Phi_0[beta], this->GList[1], Phi_1[alfa]); //T x N_moms x n_gammas_f2

      for( int im=0; im<N_moms; ++im)
	for( int t=0; t<TIME; ++t)
	  for( int gf2=0; gf2<n_gammas_f2; ++gf2)
	    x_pe_cy( this->Corr(im,t,0,gi2,gf2), g, pipi_aux.Corr(t,im,gf2), 1);
    }//nonzero elems G_i2
  }//loop over G_i2 matrix
  
}


//here pi2 is looped outside in the building of the stocastic propagator. NB for moms_red I expect that pi2 is the same! Phi_0[s] is the stocastic propagator at zero momentum and spin s, Phi_1 with momentum pi2
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::M_diagramms( PLEGMA_ScattCorrelator<Float> &CorrNucleon, std::array<PLEGMA_Vector<Float>,4> &Phi_0, std::array<PLEGMA_Vector<Float>,4> &Phi_1, bool accum){

  //extract moms
  if(!this->pList.check_eq(0)) PLEGMA_error("Mmmmmh something is not going as expected\n");
  std::vector<int> mom_pi2 = this->pList.pi(0)[0]; 
  std::vector<std::vector<int>> moms_pf2 = this->pList.uniq_p(2);
  //extract vector p_f1
  std::vector<std::vector<int>> moms_pf1_red = this->pList.uniq_p(1); //list of pf1 momenta needed here
  std::vector<std::vector<int>> moms_pf1 = CorrNucleon.pList.uniq_p(1); //list of pf1 in Nucleons PLEGMA_SC
  std::vector<int> i_pf1s = CorrNucleon.pList.u_posix( 1, moms_pf1_red ); //list of positions of moms_pf1_red momenta in moms_pf1 array
  std::vector<std::array<int,3>> map = this->pList.index_map();


  //++++++++ PION-PION +++++++++

  //aux PLEGMA_SC for PhixGxPhi multiplications
  PLEGMA_ScattCorrelator pipi_aux(this->getSource(), this->getMomList(), this->getTotalT());

  pipi_aux.initialize_diagram( this->pList, this->GList[3], this->GList[5], "P", true ); //false m is pi2, true is pf2

  pipi_aux.P_diagramms( Phi_0, Phi_1, false, 2); // pf2, t, 1, gi2, gf2

  
  //++++++++++ NN x PIPI ++++++++++++

  //put output to zero
  this->clear_output(!accum); 

  int n_extgammas_i = CorrNucleon.GList[0].size();
  int n_extgammas_f = CorrNucleon.GList[1].size();
  int n_gammas_i1 = CorrNucleon.GList[2].size();
  int n_gammas_i2 = pipi_aux.GList[0].size();
  int n_gammas_f1 = CorrNucleon.GList[3].size();
  int n_gammas_f2 = pipi_aux.GList[1].size();
  int TIME = this->localT();
  
  //for each momentum in moms_red
  for( int i_mom=0; i_mom < this->pList.size(); ++i_mom){
    int i_pf1 = i_pf1s[map[i_mom][1]]; //position of pf1 in moms_pf1 (tempNN)
    int i_pf2 = map[i_mom][2]; //position of pf2 in pionpion
    for( int t=0; t<TIME; ++t){
      for( int gei=0; gei<n_extgammas_i; ++gei ){ 
	for( int gef=0; gef<n_extgammas_f; ++gef ){
	  for( int gi1=0; gi1<n_gammas_i1; ++gi1 ){
	    for( int gi2=0; gi2<n_gammas_i2; ++gi2 ){
	      for( int gf1=0; gf1<n_gammas_f1; ++gf1 ){
		for( int gf2=0; gf2<n_gammas_f2; ++gf2){
		  
		  x_pe_cy( this->Corr(i_mom,t,0,gei,gef,gi1,gi2,gf1,gf2), pipi_aux.Corr(i_pf2,t,0,gi2,gf2),
			   CorrNucleon.Corr(i_pf1,t,0,gei,gef,gi1,gf1), N_SPINS*N_SPINS);
		}//G_f2
	      }//G_f1
	    }//G_i2
	  }//G_i1
	}//G_ext_f
      }//G_ext_i
    }//time
  }//mom
}

//Nucleon correlator. This function should be called outside the p_i2 loop, with Ts computed using the entire list of unique p_f1s. N.B: we multiply the output by exp(i * x_sourcepos * p_f1);
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::N_diagramms( PLEGMA_ScattCorrelator<Float> &T1, PLEGMA_ScattCorrelator<Float> &T2, bool accum){

  //checks between T1 T2
  if(!T1.check_reduction(T_1)) PLEGMA_error("srcT1 seems not to have T1like shape\n");
  if(!T2.check_reduction(T_2)) PLEGMA_error("srcT2 seems not to have T1like shape\n");

  if( T1.getMomList()!=T2.getMomList() || T1.getMomList()!=this->pList.uniq_p(1) )
    PLEGMA_error("T1,T2 have not the the same mom list of N\n");

  for(int i=0; i<2; ++i)
    if((T1.GList[i]!=T2.GList[i])||(T1.GList[i]!=this->GList[i+2]))
      PLEGMA_error("T1,T2 wrong gamma list\n");
  
  //extract array mom
  std::vector<std::vector<int>> moms_pf1 = this->pList.uniq_p(1);
  assert( this->N_p == moms_pf1.size() );
    
  //size of final output for NN
  int n_extgammas_i = this->GList[0].size();
  int n_extgammas_f = this->GList[1].size();
  int n_gammas_i1 = this->GList[2].size();
  int n_gammas_f1 = this->GList[3].size();
  int TIME = this->localT();  

  Float temp[N_SPINS*N_SPINS*2];
  
  //put output to zero
  this->clear_output(!accum);

  for( int i_mom=0; i_mom<moms_pf1.size(); ++i_mom){
    for( int t=0; t<TIME; ++t){
      for( int gei=0; gei<n_extgammas_i; ++gei ){ 
	for( int gef=0; gef<n_extgammas_f; ++gef ){
	  GAMMAS_SCATT extG_i1 = this->GList[0][gei];
	  GAMMAS_SCATT extG_f1 = this->GList[1][gef];
	  for( int gi1=0; gi1<n_gammas_i1; ++gi1 ){
	    for( int gf1=0; gf1<n_gammas_f1; ++gf1 ){
	      for(int spin=0; spin<N_SPINS*N_SPINS*2; ++spin)
		temp[spin] = T1.Corr(t,i_mom,gi1,gf1)[spin] + T2.Corr(t,i_mom,gi1,gf1)[spin];

	      //change in pe_GNG
	      M_pe_GNG<Float>( this->Corr(i_mom,t,0,gei,gef,gi1,gf1), extG_f1, extG_i1, temp );
	    }
	  }
	}
      }
    }
  }

}



//here pi2 and Gamma_i2 are looped outside in the building of the sequential propagator. NB for moms I expect that pi2 is the same! The T reduction contains ptot.
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::T_diagramms( PLEGMA_ScattCorrelator<Float> &T1, PLEGMA_ScattCorrelator<Float> &T3, PLEGMA_ScattCorrelator<Float> &T5, int ig_i2, bool accum){

  //checks between T1 T3 T5
  if(!T1.check_reduction(T_1)) PLEGMA_error("srcT1 seems not to have T1like shape\n");
  if(!T3.check_reduction(T_1)) PLEGMA_error("srcT3 seems not to have T1like shape\n");
  if(!T5.check_reduction(T_2)) PLEGMA_error("srcT5 seems not to have T2like shape\n");
      
  if(!this->pList.check_eq(0)) PLEGMA_error("Mmmmmh something is not going as expected\n");
  //PLEGMA_printf("DEBUG: Tdia checks ok\n");

  std::vector<std::vector<int>> moms_tot = this->pList.uniq_p(3);
  assert( this->N_p == moms_tot.size() );

  if( moms_tot != T1.getMomList() ) PLEGMA_error("T1 has not the the same mom list of T\n");
  if( moms_tot != T3.getMomList() ) PLEGMA_error("T3 has not the the same mom list of T\n");
  if( moms_tot != T5.getMomList() ) PLEGMA_error("T5 has not the the same mom list of T\n");
  //PLEGMA_printf("DEBUG: Tdia moms checks ok\n");
  
  for(int i=0; i<2; ++i)
    if( (T1.GList[i]!=T3.GList[i]) || (T1.GList[i]!=T5.GList[i]) || (T1.GList[i]!=this->GList[(i+1)*2]) )
      PLEGMA_error("T1,T3,T5 and T wrong gamma list\n");
  

  //n gammas
  int n_gammas_f = this->GList[4].size();
  int n_gammas_i2 = this->GList[3].size();
  int n_gammas_i1 = this->GList[2].size();
  int n_extgammas_f = this->GList[1].size();
  int n_extgammas_i = this->GList[0].size();
  int TIME = this->localT();
  PLEGMA_printf("DEBUG: T_diagram n_gammas: %d,%d,%d,%d,%d. ig2=%d \n",n_extgammas_i,n_extgammas_f,n_gammas_i1,n_gammas_i2,n_gammas_f, ig_i2);
	      
  Float temp[N_SPINS*N_SPINS*2];

  this->clear_output(!accum, 6, ig_i2); 
    
  for(int i_mom=0; i_mom<this->N_p; ++i_mom){
    for(int t=0; t<TIME; ++t){
      for( int gei=0; gei<n_extgammas_i; ++gei ){ 
	for( int gef=0; gef<n_extgammas_f; ++gef ){
	  GAMMAS_SCATT eGamma_i = this->GList[0][gei];
	  GAMMAS_SCATT eGamma_f = this->GList[1][gef];
	  for( int gi1=0; gi1<n_gammas_i1; ++gi1 ){
	    for( int gf=0; gf<n_gammas_f; ++gf ){
	      for(int spin=0; spin<N_SPINS*N_SPINS*2; ++spin){
		temp[spin] = (T1.Corr(t,i_mom,gi1,gf)[spin] + T3.Corr(t,i_mom,gi1,gf)[spin] + T5.Corr(t,i_mom,gi1,gf)[spin])*2;
	      }
	      //PLEGMA_printf("DEBUG: --- loop (%d,%d,%d,%d,%d,%d,%d) --- Tdia temp spin ok\n",i_mom,t,gei,gef,gi1,ig_i2,gf);
	      M_pe_GNG<Float>( this->Corr(i_mom,t,0,gei,gef,gi1,ig_i2,gf), eGamma_f, eGamma_i, temp);
				  
	    }//G_f
	  }//G_i1
	}//G_ext_f
      }//G_ext_i
    }//time
  }//mom
}


template<typename Float>
void PLEGMA_ScattCorrelator<Float>::D_diagramms( PLEGMA_ScattCorrelator<Float> &T1, PLEGMA_ScattCorrelator<Float> &T2, bool accum){
 //checks between T1 T2
  //PLEGMA_printf("DEBUG: Enter D_diagrams\n");
  if(!T1.check_reduction(T_1)) PLEGMA_error("srcT1 seems not to have T1like shape\n");
  if(!T2.check_reduction(T_2)) PLEGMA_error("srcT2 seems not to have T1like shape\n");
  //PLEGMA_printf("DEBUG: checks on Tshapes done\n");
  
  if( T1.getMomList()!=T2.getMomList() || T1.getMomList()!=this->pList.uniq_p(3) )
    PLEGMA_error("T1,T2 have not the the same mom list of D\n");
  //PLEGMA_printf("DEBUG: checks on T momList done\n");
 
  for(int i=0; i<2; ++i)
    if((T1.GList[i]!=T2.GList[i])||(T1.GList[i]!=this->GList[i+2]))
      PLEGMA_error("T1,T2 wrong gamma list\n");
  //PLEGMA_printf("DEBUG: checks on GLists done\n");

  
  //extract array mom
  std::vector<std::vector<int>> moms_tot = this->pList.uniq_p(3);
  assert( this->N_p == moms_tot.size() );
    
  //size of final output for DD
  int n_extgammas_i = this->GList[0].size();
  int n_extgammas_f = this->GList[1].size();
  int n_gammas_i = this->GList[2].size();
  int n_gammas_f = this->GList[3].size();
  int TIME = this->localT();  

  Float temp[N_SPINS*N_SPINS*2];
  PLEGMA_printf("DEBUG: Preparation done\n");
  
  //put output to zero
  this->clear_output(!accum); 
  //PLEGMA_printf("DEBUG: clear output done\n");


  for( int i_mom=0; i_mom<moms_tot.size(); ++i_mom){
    for( int t=0; t<TIME; ++t){
      for( int gei=0; gei<n_extgammas_i; ++gei ){ 
	for( int gef=0; gef<n_extgammas_f; ++gef ){
	  GAMMAS_SCATT extG_i1 = this->GList[0][gei];
	  GAMMAS_SCATT extG_f1 = this->GList[1][gef];
	  for( int gi=0; gi<n_gammas_i; ++gi ){
	    for( int gf=0; gf<n_gammas_f; ++gf ){
	      for(int spin=0; spin<N_SPINS*N_SPINS*2; ++spin)
		temp[spin] = 4*T1.Corr(t,i_mom,gi,gf)[spin] + 2*T2.Corr(t,i_mom,gi,gf)[spin];
	      //PLEGMA_printf("DEBUG: --- loop(%d,%d,%d,%d,%d,%d) --- temp[spin] done\n",i_mom,t,gei,gef,gi,gf);
	      //change in pe_GNG
	      M_pe_GNG<Float>( this->Corr(i_mom,t,0,gei,gef,gi,gf), extG_f1, extG_i1, temp );
	      //PLEGMA_printf("DEBUG: --- loop(%d,%d,%d,%d,%d,%d) --- M_pe_GNG done\n",i_mom,t,gei,gef,gi,gf);
	    }
	  }
	}
      }
    }
  }
}


//#####################
//#  Other functions  #
//#####################

//this must be used only if the source is the one used in PLEGMA_ScattCorrelator
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::applyBoundaryConditions( bool antiperiodic ) {
  //PLEGMA_printf("DEBUG: applyBoundaryConditins started\n");
  if(!antiperiodic) return;

  std::size_t n_t = this->labels.find("t");
  assert(n_t!=std::string::npos);

  //PLEGMA_printf("DEBUG: t found\n");
  
  int TIME = this->localT();
  int in_dofs = std::accumulate(ranges.begin()+n_t+1, ranges.end(), 2, std::multiplies<int>());
  int out_dofs = ranges[0]*offsets[0]/TIME/in_dofs;
  int maxT = this->endT() - this->startT();

  //PLEGMA_printf("DEBUG: prel. done\n");
  //PLEGMA_printf("DEBUG: TIME:%d, in_dofs:%d, out_dofs:%d, maxT:%d\n", TIME, in_dofs, out_dofs, maxT);
  
  for( int t=0; t<TIME; ++t){
    int t_local = (t>=maxT) ? (this->source[DIM_T]%HGC_localL[DIM_T]) + t - maxT : t;
    int t_global = HGC_procPosition[DIM_T] * HGC_localL[DIM_T] + t_local;
    if( t_global < this->source[DIM_T] ){
      for( int o_dofs=0; o_dofs<out_dofs; ++o_dofs){
	for( int i_dofs=0; i_dofs<in_dofs; ++i_dofs){
	  *(this->H_elem() + o_dofs*TIME*in_dofs + t*in_dofs  + i_dofs) = -*(this->H_elem() + o_dofs*TIME*in_dofs + t*in_dofs  + i_dofs);
	  //PLEGMA_printf("DEBUG: --- loop (%d,%d,%d) --- multiplication done\n",t,o_dofs,i_dofs);
	}
      }
    }
  }
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::apply_phase( std::vector<std::vector<int>> mom_list ){
  int tot_size = this->getTotalSize();
  int N_moms = mom_list.size();
  int in_dofs = tot_size/N_moms;
  
  assert( N_moms==this->ranges[0] );
  assert( N_moms==this->N_p );
 
  for(int i_m=0; i_m<N_moms; i_m++){
    Float phase=2*M_PI/(Float)HGC_totalL[0]* mom_list[i_m][0]*this->source[0]+
                      2*M_PI/(Float)HGC_totalL[1]* mom_list[i_m][1]*this->source[1]+
                      2*M_PI/(Float)HGC_totalL[2]* mom_list[i_m][2]*this->source[2];
    Float tmpreim[2]={cos(phase),sin(phase)};
    x_e_cx<Float>( this->Corr(i_m), tmpreim, in_dofs);
  }
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::clear_output(bool tozero, int n_index, int i){

  if(!tozero)
    return;
  
  assert(n_index<this->labels.length());

  int out_dofs = std::accumulate(ranges.begin(),ranges.begin()+n_index, 1, std::multiplies<int>());
  int in_dofs = this->offsets[n_index];

  for(int oi=0; oi<out_dofs; ++oi)
    memset( this->H_elem()+(oi*ranges[n_index]+i)*in_dofs, 0, in_dofs*sizeof(Float) );
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::clear_output(bool tozero){
  if(!tozero)
    return;
  int tot_size = 2*this->getTotalSize();
  memset( this->H_elem(), 0, tot_size*sizeof(Float) );
}


//V3_a^l*G_ab*V3_b^l
//template<typename Float>
//void PLEGMA_ScattCorrelator<Float>::V3V3reduction( PLEGMA_ScattCorrelator<Float> &srcV2, int alfa, int beta){

template class PLEGMA_ScattCorrelator<float>;
template class PLEGMA_ScattCorrelator<double>;


