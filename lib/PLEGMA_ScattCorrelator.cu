#include <PLEGMA_ScattCorrelator.h>
#include <PLEGMA_Vector.h>
#include <PLEGMA_Propagator.h>
#include <PLEGMA_scattreductions.cuh>
#include <PLEGMA_scattreductionsPiPi.cuh>
#include <PLEGMA_utils.h>
#include <omp.h>
#include  <memory>
#include <comm_quda.h>
#include <communicator_quda.h>
using namespace plegma;
using namespace quda;

Communicator &get_current_communicator();



bool gammas_isSym( std::vector<GAMMAS_SCATT> &Gammas ){
  bool res = true;
  for( auto &g : Gammas )
    if ( gammaTranspSign_scatt[g] == -1 ) res=false;
  return res;
}

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

template<typename Float>
PLEGMA_ScattCorrelator<Float>::PLEGMA_ScattCorrelator(site source, momList &listmom, int totalT): PLEGMA_Correlator<Float>(MOMENTUM_SPACE,source,0,totalT), plist(listmom) {
  auto mlist = this->plist.tolist();
  this->setFixMomList( mlist );
}

//#########################
//#  Auxiliary functions  #
//#########################
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::setOffsets( ){
  int g_count=0;
  ranges.clear();
      
  for( auto &l : this->labels ){
    switch(l){
    case('t'): assert(this->corr_mom_space); ranges.push_back( this->corr_mom_space->DimT() ); break;
    case('m'): assert(this->corr_mom_space); ranges.push_back( this->corr_mom_space->Nmoms() ); break;
    case('g'): assert(g_count<GList.size()); ranges.push_back( GList[g_count].size() ); g_count++; break;
    case('s'): ranges.push_back( N_SPINS ); break;
    case('c'): ranges.push_back( N_COLS ); break;
    case('d'): ranges.push_back( N_DIMS ); break;
    case('l'): ranges.push_back( N_DIMS*(N_DIMS-1)) ; break;
    default: PLEGMA_error( "Label %c not recognized\n", l );
   }	
  }

  offsets.clear();
  for( int ir=1; ir<ranges.size(); ++ir ){
    int offset = std::accumulate(ranges.begin()+ir, ranges.end(), 2, std::multiplies<int>());
    offsets.push_back(offset);
  }
  offsets.push_back(2);

  assert(this->offsets[0]*this->ranges[0]==2*this->getTotalSize());
  
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

  return true;
}


//#####################
//#  Main reductions  #
//#####################

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::V3( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS_SCATT> &Gammas, PLEGMA_Propagator<Float> &S, bool conj_v){ 

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
  if(conj_v)
    V_reductions<true,0,1,Float,Float,Float>(V_3, *this, Phi, Gammas, S);
  else
    V_reductions<false,0,1,Float,Float,Float>(V_3, *this, Phi, Gammas, S);
}


template<typename Float>
void PLEGMA_ScattCorrelator<Float>::V5( PLEGMA_Vector<Float> &Phi1,  PLEGMA_Vector<Float> &Phi2, bool conj_v){

  this->GList.clear();


  this->datasets={"dataset_v5"};
  this->groups={"group_v5"};
  this->shape={N_SPINS,N_SPINS,N_COLS};
  this->labels="tmssc";
  this->initialize();
  this->setOffsets();

  for(int i=0; i<3; ++i)
    assert(this->source[i]==0);
  if(conj_v)
    V_reductions<true,0,1,Float,Float,Float>(V_5, *this, Phi1, Phi2);
  else
    V_reductions<false,0,1,Float,Float,Float>(V_5, *this, Phi1, Phi2);
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::V6( PLEGMA_Vector<Float> &Phi1,  PLEGMA_Vector<Float> &Phi2, PLEGMA_Propagator<Float> &S, bool conj_v){

  this->GList.clear();


  this->datasets={"dataset_v6"};
  this->groups={"group_v6"};
  this->shape={N_SPINS,N_SPINS,N_SPINS,N_SPINS,N_COLS};
  this->labels="tmssssc";
  this->initialize();
  this->setOffsets();

  for(int i=0; i<3; ++i)
    assert(this->source[i]==0);
  if(conj_v)
    V_reductions<true,0,1,Float,Float,Float>(V_6, *this, Phi1, Phi2, S);
  else
    V_reductions<false,0,1,Float,Float,Float>(V_6, *this, Phi1, Phi2, S);
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::V6_RED( PLEGMA_Vector<Float> &Phi1, PLEGMA_Vector<Float> &Phi2, std::vector<GAMMAS_SCATT> &Gammas, PLEGMA_Propagator<Float> &S1, int C1, int C2, bool conj_v) {

  int n_gammas= Gammas.size();
  this->GList.clear();

  if(n_gammas<=0||n_gammas>16)
    PLEGMA_error("provide at list 1 Gamma matrix and no more than 16(temporary)\n");
  this->GList.push_back(Gammas);

  this->datasets={"dataset_v6_red"};
  this->groups={"group_v6_red"};
  this->shape={n_gammas,N_SPINS,N_SPINS,N_COLS};
  this->labels="tmgssc";
  this->initialize();
  this->setOffsets();

  for(int i=0; i<3; ++i)
    assert(this->source[i]==0);

  if(conj_v ){
    if ((C1==0) && (C2==1)){
       V_reductions<true,0,1,Float,Float,Float>( V_6_RED, *this, Phi1, Phi2, Gammas, S1);
    }
    else if ((C1==1) && (C2==2)){
       V_reductions<true,1,2,Float,Float,Float>( V_6_RED, *this, Phi1, Phi2, Gammas, S1);
    }
  }
  else{
   if ((C1==0) && (C2==1)){
       V_reductions<false,0,1,Float,Float,Float>( V_6_RED, *this, Phi1, Phi2, Gammas, S1);
    }
    else if ((C1==1) && (C2==2)){
       V_reductions<false,1,2,Float,Float,Float>( V_6_RED, *this, Phi1, Phi2, Gammas, S1);
    }
  }
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::V4( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS_SCATT> &Gammas, PLEGMA_Propagator<Float> &S1, PLEGMA_Propagator<Float> &S2, bool conj_v) {

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

  if(conj_v)
    V_reductions<true,0,1,Float,Float,Float>( V_4, *this, Phi, Gammas, S1, S2);
  else
    V_reductions<false,0,1,Float,Float,Float>( V_4, *this, Phi, Gammas, S1, S2);
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::V2( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS_SCATT> &Gammas, PLEGMA_Propagator<Float> &S1, PLEGMA_Propagator<Float> &S2, bool conj_v) {

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

  if(conj_v)
    V_reductions<true,0,1,Float,Float,Float>(V_2, *this, Phi, Gammas, S1, S2);
  else
    V_reductions<false,0,1,Float,Float,Float>(V_2, *this, Phi, Gammas, S1, S2);
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
//This routine filters the source time-slice from a PLEGMA_ScattCorrelator
//object: i.e. it return all the momenta, gamma, spin, real-imag components
//at the source-time slice. We particulary use it for diagrams with loop at
//the source, here now with zero momentum and with one gamma structure
//
//To do: (1)check it
//       (2)generalize it to the case, where the ScattCorrelator object is not
//          as long as the time-extent of the Lattice
//       (3)genarlize it to a other time-slices
template<typename Float>
Float *PLEGMA_ScattCorrelator<Float>::get_source_time_slice(){
  const int  size_of_glist =this->GList.size();
  int size_timeslice= this->Nmoms();
  for (int i=0; i< size_of_glist; ++i){
    size_timeslice *= this->GList[i].size();
  }
  std::size_t n_t = this->labels.find("s");
  if (n_t!=std::string::npos){
    size_timeslice *= 32;
  }
  else{
    size_timeslice *= 2;
  }
  Float *ptr=((Float *)malloc(sizeof(Float)*size_timeslice));
  const int t_source_local= this->source[DIM_T]%HGC_localL[DIM_T];
  memcpy(ptr, this->H_elem()+t_source_local*size_timeslice, sizeof(Float)*size_timeslice); 
  int coords[4];
  for(int i = 0 ; i < N_DIMS; i++) coords[i] = this->getSource()[i] / HGC_localL[i];
  int rankHas = comm_rank_from_coords(coords);

  int mpiErr = MPI_Bcast(ptr, size_timeslice, MPI_Type<Float>(), rankHas, HGC_fullComm);
  if(mpiErr != MPI_SUCCESS) PLEGMA_error("MPI_Bcast failed with error %d\n", mpiErr);
  MPI_Barrier(HGC_fullComm);
//  PLEGMA_printf("DEBUG ptr global %e %e\n",ptr[0],ptr[1]);
  return ptr;
}

template<typename Float>
Float *PLEGMA_ScattCorrelator<Float>::get_time_slice(int global_time_index){
  const int  size_of_glist =this->GList.size();
  int size_timeslice= this->Nmoms();
  printf("Nmoms %d \n", this->Nmoms());
  printf("Global time index %d\n",global_time_index);
  for (int i=0; i< size_of_glist; ++i){
    size_timeslice *= this->GList[i].size();
  }
  printf("Size timeslice %d\n",size_timeslice);
  std::size_t n_t = this->labels.find("s");
  if (n_t!=std::string::npos){
    size_timeslice *= 32;
  }
  else{
    size_timeslice *= 2;
  }
  Float *ptr=((Float *)malloc(sizeof(Float)*size_timeslice));
  const int t_source_local= global_time_index%HGC_localL[DIM_T];
  memcpy(ptr, this->H_elem()+t_source_local*size_timeslice, sizeof(Float)*size_timeslice);
  printf("ptr %e\n",ptr[0]);
  int coords[4];
  for(int i = 0 ; i < (N_DIMS-1); i++) coords[i] = 0;
  coords[N_DIMS-1]=global_time_index / HGC_localL[N_DIMS-1];
  int rankHas = quda::comm_rank_from_coords(coords);
  printf("rankHas %d\n",rankHas);
  MPI_Barrier(HGC_fullComm);
  int mpiErr = MPI_Bcast(ptr, size_timeslice, MPI_Type<Float>(), rankHas, HGC_fullComm);
  if(mpiErr != MPI_SUCCESS) PLEGMA_error("MPI_Bcast failed with error %d\n", mpiErr);
  printf("ptrafter %e\n",ptr[0]);
  fflush(stdout);
  MPI_Barrier(HGC_fullComm);
  return ptr;
}

//This routine sum over the time direction a particular PLEGMA_ScattCorrelator object
template<typename Float>
std::shared_ptr<Float> PLEGMA_ScattCorrelator<Float>::average_all_time_slices(){
  const int  size_of_glist =this->GList.size();
  int size_timeslice= this->Nmoms();
  for (int i=0; i< size_of_glist; ++i){
    size_timeslice *= this->GList[i].size();
  }
  std::size_t n_t = this->labels.find("s");
  if (n_t!=std::string::npos){
    size_timeslice *= 32;
  }
  else{
    size_timeslice *= 2;
  }
  Float *localsum=(Float *)malloc(sizeof(Float)*size_timeslice);
  std::shared_ptr<Float> ptr((Float *)malloc(sizeof(Float)*size_timeslice), free);
  for(int j=0; j<size_timeslice; ++j)
      localsum[j]=0; 
  int TIME = this->localT();
  for (int i=0; i<TIME; ++i){
    for(int j=0; j<size_timeslice; ++j)
      localsum[j]+=this->Corr(i)[j];
  }
  for (int j=0; j<size_timeslice; ++j)
    ptr.get()[j]=localsum[j];

  MPI_Allreduce( localsum, ptr.get(), size_timeslice, MPI_Type(localsum[0]), MPI_SUM, HGC_timeComm);
  MPI_Barrier(MPI_COMM_WORLD);
  free(localsum);
  return ptr;
}



//###############################
//#  Combination of reductions  #
//###############################

//Contracts V3^a_c x Gammas_cd x V2^a_{abd}. Gammas_i list of gammas between V3, V2. 
//called by aux PLEGMA_ScattCorrelator with shape PTGGGGGGSS
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::V3V2reduction( PLEGMA_ScattCorrelator<Float> &srcV3,PLEGMA_ScattCorrelator<Float> &srcV2, int index_abs, bool transp, int g0, bool transpgamma_i1, Float* factor, bool transpgamma_f1, bool oet) {

  //checks
  std::string exp_shape="tmggggggss";
  if( this->labels!=exp_shape )
    PLEGMA_error("V3V2reduction Expected shape ptmggggggss not the one detected\n");

  if( !srcV2.check_reduction(V_2) ) PLEGMA_error("SrcV2 object does not seem a V2like object\n");
  if( !srcV3.check_reduction(V_3) ) PLEGMA_error("SrcV3 object does not seem a V3like object\n");
  if( srcV2.getGList()[0] != this->GList[4] ) PLEGMA_error("G_f1 doesn't match\n");
  //if( srcV3.getGList()[0] != this->GList[5] ) PLEGMA_error("G_f2 doesn't match\n");

  int n_gammas_exti = this->GList[0].size();
  int n_gammas_extf = this->GList[1].size();
  int n_gammas_i1 = this->GList[2].size();
  int n_gammas_i2 = this->GList[3].size();
  int n_gammas_f1 = this->GList[4].size();
  int n_gammas_f2 = this->GList[5].size();
  int TIME = this->localT();
  
  int Nmoms_f1 = srcV2.Nmoms();
  int Nmoms_i2_f2 = srcV3.Nmoms();

  //PLEGMA_ScattCorrelator<Float> V3aux(srcV2.getSource(), srcV2.getMomList(), srcV2.getTotalT());
  auto imap = this->pList().index_map();

  #pragma omp parallel for
  for(int i_m=0; i_m<imap.size(); i_m++){
    Float V3aux[N_SPINS*N_COLS*2];
    Float temp[2*N_SPINS*N_SPINS];
    int i_mom_f1 = imap[i_m][1];
    int i_mom_i2_f2;
    if (oet==true) {
      i_mom_i2_f2= imap[i_m][0];
    }else{
      i_mom_i2_f2= imap[i_m][2];
    }
    for(int t=0; t < TIME; ++t){
      for (int g1=0 ; g1 < n_gammas_i1 ; ++g1 ){//pi
        for (int g2=0 ; g2 < n_gammas_f1 ; ++g2 ){//pf1
	  int n_gammas_i2_f2;
	  if (oet==true){
	    n_gammas_i2_f2=n_gammas_i2;
	  } else {
	    n_gammas_i2_f2=n_gammas_f2;
	  }
          for (int g3=0 ; g3 < n_gammas_i2_f2 ; ++g3 ){//pf2
            for (int alfa=0; alfa < N_SPINS; ++alfa ){
	      for (int beta=0; beta < N_SPINS; ++beta ){
	        int spins = (transp) ? (beta*N_SPINS+alfa)*2 : (alfa*N_SPINS+beta)*2;
	        switch(index_abs){
		  case 0: absorb_fromV24<0,Float>( V3aux, srcV2.Corr(t,i_mom_f1,g2), alfa, beta ); break;
		  case 1: absorb_fromV24<1,Float>( V3aux, srcV2.Corr(t,i_mom_f1,g2), alfa, beta ); break;
		  case 2: absorb_fromV24<2,Float>( V3aux, srcV2.Corr(t,i_mom_f1,g2), alfa, beta ); break;
                }
		//if true multiply by sigma_T(G_f1)
		if(transpgamma_f1){
                  for(int sc=0; sc<N_SPINS*N_COLS*2; ++sc)
		    V3aux[sc] *= gammaTranspSign_scatt[this->GList[4][g2]];
		}
		V_M_V<Float>( srcV3.Corr(t,i_mom_i2_f2,g3), V3aux,
		              this->GList[2][g1], transpgamma_i1, temp + spins);
              }//beta
            }//alfa
            if(factor!=NULL){
	      Float aux;
	      for(int ss=0; ss<N_SPINS*N_SPINS; ++ss){
	        aux = temp[2*ss+0]*factor[0] - temp[2*ss+1]*factor[1];
		temp[2*ss+1] = temp[2*ss+0]*factor[1] + temp[2*ss+1]*factor[0];
		temp[2*ss+0] = aux;
              }
            }

            for (int g_exti=0; g_exti < n_gammas_exti ; ++ g_exti){
              for (int g_extf=0; g_extf < n_gammas_extf ; ++ g_extf){
                GAMMAS_SCATT egammai = this->GList[0][g_exti];
                GAMMAS_SCATT egammaf = this->GList[1][g_extf];
		
	        //multiplication with external gammas NB written here! mod in M_pe_GNG
		if (oet==true){
                  M_pe_GNG<Float>( this->Corr(t,i_m,g_exti,g_extf,g1,g3,g2,g0),
                                 egammaf, egammai, temp );
		} else{
		  M_pe_GNG<Float>( this->Corr(t,i_m,g_exti,g_extf,g1,g0,g2,g3),
		                 egammaf, egammai, temp );

		}
	      }//Gextf
	    }//Gexti
	  }//Gf2	
	}//Gf1
      }//Gi1
    }//time
  }//mom
}


template<typename Float>
void PLEGMA_ScattCorrelator<Float>::V3V2reduction_matrix( PLEGMA_ScattCorrelator<Float> &srcV3, PLEGMA_ScattCorrelator<Float> &srcV2, int index_abs, bool transp,  int g0, bool transpgamma_i1, Float* factor, bool transpgamma_f1, bool oet) {

  //checks
  std::string exp_shape="tmggggggss";
  if( this->labels!=exp_shape )
    PLEGMA_error("V3V2reduction_matrix Expected shape tmggggggss not the one detected (%s)\n",+this->labels.c_str());
  
  if( !srcV2.check_reduction(V_2) ) PLEGMA_error("SrcV2 object does not seem a V2like object\n");
  if( !srcV3.check_reduction(V_3) ) PLEGMA_error("SrcV3 object does not seem a V3like object\n");
  if( srcV2.getGList()[0] != this->GList[4] ) PLEGMA_error("G_f1 doesn't match\n");
  //if( srcV3.getGList()[0] != this->GList[5] ) PLEGMA_error("G_f2 doesn't match\n");

  int n_gammas_exti = this->GList[0].size();
  int n_gammas_extf = this->GList[1].size();
  int n_gammas_i1 = this->GList[2].size();
  int n_gammas_i2 = this->GList[3].size();
  int n_gammas_f1 = this->GList[4].size();
  int n_gammas_f2 = this->GList[5].size();
  int TIME = this->localT();
  
  int Nmoms_f1 = srcV2.Nmoms();
  int Nmoms_f2 = srcV3.Nmoms();
    
  auto imap = this->pList().index_map();


  #pragma omp parallel for
  for(int i_m=0; i_m<imap.size(); i_m++){
    int i_mom_f1 = imap[i_m][1];
    int i_mom_i2_f2;
    if (oet==true){
     i_mom_i2_f2= imap[i_m][0];
    }else {
     i_mom_i2_f2= imap[i_m][2];
    }
    Float V3aux[N_SPINS*N_SPINS*N_COLS*2];
    Float temp_colorvector[N_COLS*2];
    Float temp[N_SPINS*N_SPINS*2];

    for(int t=0; t < TIME; ++t){  
      for (int g1=0 ; g1 < n_gammas_i1 ; ++g1 ){//pi
        for (int g2=0 ; g2 < n_gammas_f1 ; ++g2 ){//pf1
	  int n_gammas_i2_f2;
	  if (oet==true){
	    n_gammas_i2_f2=n_gammas_i2;
	  } else{
	    n_gammas_i2_f2=n_gammas_f2;
	  }
          for (int g3=0 ; g3 < n_gammas_i2_f2 ; ++g3 ){//pf2
            for (int alfa=0; alfa < N_SPINS; ++alfa ){
              for (int beta=0; beta < N_SPINS; ++beta ){
                int spins = (transp) ? (beta*N_SPINS+alfa)*2 : (alfa*N_SPINS+beta)*2;
	        switch(index_abs){
		  case 0: absorbspinmatrix_fromV24<0,Float>( V3aux, srcV2.Corr(t,i_mom_f1,g2), alfa ); break;
		  case 1: absorbspinmatrix_fromV24<1,Float>( V3aux, srcV2.Corr(t,i_mom_f1,g2), alfa ); break;
		  case 2: absorbspinmatrix_fromV24<2,Float>( V3aux, srcV2.Corr(t,i_mom_f1,g2), alfa ); break;
		}
		//if true multiply by sigma_T(G_f1)
		if(transpgamma_f1){
		  for(int ssc=0; ssc<N_SPINS*N_SPINS*N_COLS*2; ++ssc)
		    V3aux[ssc] *= gammaTranspSign_scatt[this->GList[4][g2]];
		}

		//color vector from Tr[G_i1 V2]
		V_TR_MM<Float>( V3aux, this->GList[2][g1], transpgamma_i1, temp_colorvector);
		temp[spins]=0.;
		temp[spins+1]=0.;

		//colorvector x V3
		for (int coloridx=0; coloridx<3; ++coloridx){
		  temp[spins+0] +=
	            +temp_colorvector[2*coloridx+0]*srcV3.Corr(t,i_mom_i2_f2,g3,beta,coloridx)[0]
	            -temp_colorvector[2*coloridx+1]*srcV3.Corr(t,i_mom_i2_f2,g3,beta,coloridx)[1];
		  temp[spins+1] +=
	            +temp_colorvector[2*coloridx+1]*srcV3.Corr(t,i_mom_i2_f2,g3,beta,coloridx)[0]
		    +temp_colorvector[2*coloridx+0]*srcV3.Corr(t,i_mom_i2_f2,g3,beta,coloridx)[1];
		}//coloridx
	      }//beta
	    }//alfa
	    if(factor!=NULL){
              Float aux;
	      for(int ss=0; ss<N_SPINS*N_SPINS; ++ss){
	        aux = temp[2*ss+0]*factor[0] - temp[2*ss+1]*factor[1];
	        temp[2*ss+1] = temp[2*ss+0]*factor[1] + temp[2*ss+1]*factor[0];
		temp[2*ss+0] = aux;
	      }
	    }

            for (int g_exti=0; g_exti < n_gammas_exti ; ++ g_exti){
	      for (int g_extf=0; g_extf < n_gammas_extf ; ++ g_extf){    
	        GAMMAS_SCATT egammai = this->GList[0][g_exti];
	        GAMMAS_SCATT egammaf = this->GList[1][g_extf];


		//multiplication with external gammas NB: written here!!
		if (oet==true){
                  M_pe_GNG<Float>( this->Corr(t,i_m,g_exti,g_extf,g1,g3,g2,g0),
                                egammaf, egammai, temp );
		}else{
		  M_pe_GNG<Float>( this->Corr(t,i_m,g_exti,g_extf,g1,g0,g2,g3),
				egammaf, egammai, temp );
		}
		
	      }//Gextf
	    }//Gexti
	  }//Gf2	
	}//Gf1
      }//Gi1
    }//time
  }//mom
}


template<typename Float>
void PLEGMA_ScattCorrelator<Float>::V5V6reduction(PLEGMA_ScattCorrelator<Float> &srcV6, std::shared_ptr<Float> &Phi0, std::vector<PLEGMA_Vector<Float>*> &Phi_1, int input_mom_f2, int index_abs_V6, int index_abs_V5, bool transp, bool transpgamma_i1, bool transpgamma_f1, Float *factor) {


    static const int eps_host[6][3]= {{0,1,2},
                                      {2,0,1},
                                      {1,2,0},
                                      {2,1,0},
				      {0,2,1},
                                      {1,0,2}};
    
    static const int sgn_eps_host[6]= { +1,+1,+1,-1,-1,-1 };
  
    int Nmoms_f1 = srcV6.Nmoms();

    std::string exp_shape="tmggggggss";
    if( this->labels!=exp_shape )
      PLEGMA_error("V3V2reduction Expected shape ptmggggggss not the one detected\n");

    if( srcV6.getGList()[0] != this->GList[4] ) PLEGMA_error("G_f1 doesn't match\n");
  
    //if( srcV3.getGList()[0] != this->GList[5] ) PLEGMA_error("G_f2 doesn't match\n");

    int n_gammas_exti = this->GList[0].size();
    int n_gammas_extf = this->GList[1].size();
    int n_gammas_i1 = this->GList[2].size();
    int n_gammas_i2 = this->GList[3].size();
    int n_gammas_f1 = this->GList[4].size();
    int n_gammas_f2 = this->GList[5].size();
    int TIME = this->localT();
    int TIME_src= srcV6.localT();

    const int NS2C=2*N_SPINS*N_SPINS*N_COLS;
    const int NS1C=2*N_SPINS*N_COLS;

    if (TIME != TIME_src){
      PLEGMA_error("W diagram and V6 reduction expected to have the same time extent\n");
    }

    site actualSource=this->source;

    auto imap = this->pList().index_map();

    std::vector<std::shared_ptr<Float>> Phi1; 
    for (int t=0; t<HGC_totalL[DIM_T]; ++t){
      Phi1.push_back(Phi_1[t]->getPointSource(actualSource,HOST));
    }


    for(int t=0; t < TIME; ++t){
      int global_time_index = t + HGC_procPosition[3] * HGC_localL[3];

      for (int g2=0 ; g2 < n_gammas_i2 ; ++g2 ){//gi2
        Float phi0Aux[N_SPINS*N_COLS*2];
        GAMMAS_SCATT gamma5 = G_5;
        GAMMAS_SCATT gamma5_t_gammai2 = apply_g5( this->GList[3][g2], LEFT);
        V_MVM<Float>( Phi0.get(), gamma5, gamma5_t_gammai2, phi0Aux );

        for (int g4=0 ; g4 < n_gammas_f2 ; ++g4 ){//gf2

          Float phi1Aux[N_SPINS*N_COLS*2];
          GAMMAS_SCATT gamma5_t_gammaf2 = apply_g5( this->GList[5][g4], LEFT);
          V_MVM<Float>( Phi1[global_time_index].get(), gamma5, gamma5_t_gammaf2, phi1Aux );

          for (int g1=0 ; g1 < n_gammas_i1 ; ++g1 ){//gi1
            for (int g3=0 ; g3 < n_gammas_f1 ; ++g3 ){//gf1
              Float V5Aux[NS2C];
	      for (int ii=0;ii<NS2C;++ii)
	        V5Aux[ii]=0;

	      for (int alfa=0; alfa < N_SPINS; ++alfa ){                
	        for (int beta=0; beta < N_SPINS; ++beta ){

	          for (unsigned short eps1_nz=0; eps1_nz<6; eps1_nz++ ){
	            unsigned short m=eps_host[eps1_nz][0];
		    unsigned short a=eps_host[eps1_nz][1];
		    unsigned short b=eps_host[eps1_nz][2];
		    int eps1_sgn=sgn_eps_host[eps1_nz];
		    Float tmp[2];
		    tmp[0]= (phi0Aux[2*(alfa*N_COLS+a)]*phi1Aux[2*(beta*N_COLS+b)]-phi0Aux[2*(alfa*N_COLS+a)+1]*phi1Aux[2*(beta*N_COLS+b)+1])*(Float)eps1_sgn;
		    tmp[1]= (-phi0Aux[2*(alfa*N_COLS+a)+1]*phi1Aux[2*(beta*N_COLS+b)]-phi0Aux[2*(alfa*N_COLS+a)]*phi1Aux[2*(beta*N_COLS+b)+1])*(Float)eps1_sgn;
		    V5Aux[2*((alfa*N_SPINS+beta)*N_COLS+m)]   = V5Aux[2*((alfa*N_SPINS+beta)*N_COLS+m)] + tmp[0];
		    V5Aux[2*((alfa*N_SPINS+beta)*N_COLS+m)+1] = V5Aux[2*((alfa*N_SPINS+beta)*N_COLS+m)+1] + tmp[1];
		  }
                }
	      }

              #pragma omp parallel for
              for(int i_m=0; i_m<imap.size(); i_m++){
	        Float V5Aux2[NS1C];
                Float V6Aux[NS1C];
                Float temp[2*N_SPINS*N_SPINS];
	        for (int ii=0;ii<NS1C;++ii){
                  V6Aux[ii]=0;
                  V5Aux2[ii]=0;
                }
                int i_mom_f1 = imap[i_m][1];
                int i_mom_f2 = imap[i_m][2];
                if ((i_mom_f2==input_mom_f2)){
		  
		  for (int alfa=0;alfa<N_SPINS;++alfa){
		    for (int beta=0; beta<N_SPINS;++beta){

		      int spins = (transp) ? (beta*N_SPINS+alfa)*2 : (alfa*N_SPINS+beta)*2;

		      switch(index_abs_V6){
			case 0: absorb_fromV56<0,Float>( V6Aux, srcV6.Corr(t,i_mom_f1,g3), alfa ); break;
			case 1: absorb_fromV56<1,Float>( V6Aux, srcV6.Corr(t,i_mom_f1,g3), alfa ); break;
		      }
		      switch(index_abs_V5){
			case 0: absorb_fromV56<0,Float>( V5Aux2, V5Aux, beta ); break;
			case 1: absorb_fromV56<1,Float>( V5Aux2, V5Aux, beta ); break;
		      }

		      //if true multiply by sigma_T(G_f1)
		      if(transpgamma_f1){
			for(int sc=0; sc<N_SPINS*N_COLS*2; ++sc){
			  V6Aux[sc] *= gammaTranspSign_scatt[this->GList[4][g2]];
			}
		      }
		      V_M_V<Float>( V5Aux2, V6Aux,
			  this->GList[2][g1], transpgamma_i1, temp + spins);
		    }//beta
		  }//alfa
		  if(factor!=NULL){
		    Float aux;
		    for(int ss=0; ss<N_SPINS*N_SPINS; ++ss){
		      aux = temp[2*ss+0]*factor[0] - temp[2*ss+1]*factor[1];
		      temp[2*ss+1] = temp[2*ss+0]*factor[1] + temp[2*ss+1]*factor[0];
		      temp[2*ss+0] = aux;
		    }
		  }
		  for (int g_exti=0; g_exti < n_gammas_exti ; ++ g_exti){
		    for (int g_extf=0; g_extf < n_gammas_extf ; ++ g_extf){
		      GAMMAS_SCATT egammai = this->GList[0][g_exti];
		      GAMMAS_SCATT egammaf = this->GList[1][g_extf];

		      //multiplication with external gammas NB written here! mod in M_pe_GNG
		      M_pe_GNG<Float>( this->Corr(t,i_m,g_exti,g_extf,g1,g2,g3,g4),
			  egammaf, egammai, temp );
		    }//Gextf
		  }//Gexti
		}//Gf1
	      }//Gi1
	    }//Gf2
	  }//Gi2
	}//t
      }//if mom
    }//mom
}


template<typename Float>
void PLEGMA_ScattCorrelator<Float>::V5V6reduction_matrix(PLEGMA_ScattCorrelator<Float> &srcV6, std::shared_ptr<Float> &Phi0, std::vector<PLEGMA_Vector<Float>*> &Phi_1, int input_mom_f2, bool transp, bool transpgamma_i1, bool transpgamma_f1, Float *factor) {


    static const int eps_host[6][3]= {{0,1,2},
                                      {2,0,1},
                                      {1,2,0},
                                      {2,1,0},
                                      {0,2,1},
                                      {1,0,2}};
    
    static const int sgn_eps_host[6]= { +1,+1,+1,-1,-1,-1 };
  
    int Nmoms_f1 = srcV6.Nmoms();

    std::string exp_shape="tmggggggss";
    if( this->labels!=exp_shape )
      PLEGMA_error("V3V2reduction Expected shape ptmggggggss not the one detected\n");

    if( srcV6.getGList()[0] != this->GList[4] ) PLEGMA_error("G_f1 doesn't match\n");
  
    //if( srcV3.getGList()[0] != this->GList[5] ) PLEGMA_error("G_f2 doesn't match\n");

    int n_gammas_exti = this->GList[0].size();
    int n_gammas_extf = this->GList[1].size();
    int n_gammas_i1 = this->GList[2].size();
    int n_gammas_i2 = this->GList[3].size();
    int n_gammas_f1 = this->GList[4].size();
    int n_gammas_f2 = this->GList[5].size();
    int TIME = this->localT();
    int TIME_src= srcV6.localT();

    if (TIME != TIME_src){
      PLEGMA_error("W diagram and V6 reduction expected to have the same time extent\n");
    }
    
    site actualSource=this->source;
    
    auto imap = this->pList().index_map();

    std::vector<std::shared_ptr<Float>> Phi1;
    for (int t=0; t<HGC_totalL[DIM_T]; ++t){
      Phi1.push_back(Phi_1[t]->getPointSource(actualSource,HOST));
    }


    for (int t=0; t<TIME; ++t){
      int global_time_index = t + HGC_procPosition[3] * HGC_localL[3];

      for (int g2=0 ; g2 < n_gammas_i2 ; ++g2 ){//gi2
        Float phi0Aux[N_SPINS*N_COLS*2];
        GAMMAS_SCATT gamma5 = G_5;
        GAMMAS_SCATT gamma5_t_gammai2 = apply_g5( this->GList[3][g2], LEFT);
        V_MVM<Float>( Phi0.get(), gamma5, gamma5_t_gammai2, phi0Aux );
        for (int g4=0 ; g4 < n_gammas_f2 ; ++g4 ){//gf2
          Float phi1Aux[N_SPINS*N_COLS*2];
          GAMMAS_SCATT gamma5_t_gammaf2 = apply_g5( this->GList[5][g4], LEFT);
	  V_MVM<Float>( Phi1[global_time_index].get(), gamma5, gamma5_t_gammaf2, phi1Aux );

          for (int g1=0 ; g1 < n_gammas_i1 ; ++g1 ){//gi1i
	    for (int g3=0 ; g3 < n_gammas_f1 ; ++g3 ){//gf1

              Float V5Aux[2*N_SPINS*N_SPINS*N_COLS];
              for (int i=0;i<2*N_SPINS*N_SPINS*N_COLS;++i){
	        V5Aux[i]=0.;
	      }
	      Float V5Aux2[2*N_COLS];
              for (int i=0;i<2*N_COLS;++i){
                V5Aux2[i]=0.;
              }

              for (int alfa=0; alfa < N_SPINS; ++alfa ){                
	        for (int beta=0; beta < N_SPINS; ++beta ){
                  for (unsigned short eps1_nz=0; eps1_nz<6; eps1_nz++ ){
	            unsigned short m=eps_host[eps1_nz][0];
		    unsigned short a=eps_host[eps1_nz][1];
		    unsigned short b=eps_host[eps1_nz][2];
		    int eps1_sgn=sgn_eps_host[eps1_nz];
		    Float tmp[2];
		    tmp[0]= ( phi0Aux[2*(alfa*N_COLS+a)]*phi1Aux[2*(beta*N_COLS+b)]-phi0Aux[2*(alfa*N_COLS+a)+1]*phi1Aux[2*(beta*N_COLS+b)+1])*(Float)eps1_sgn;
                    tmp[1]= (-phi0Aux[2*(alfa*N_COLS+a)+1]*phi1Aux[2*(beta*N_COLS+b)]-phi0Aux[2*(alfa*N_COLS+a)]*phi1Aux[2*(beta*N_COLS+b)+1])*(Float)eps1_sgn;
		    V5Aux[2*((alfa*N_SPINS+beta)*N_COLS+m)]   = V5Aux[2*((alfa*N_SPINS+beta)*N_COLS+m)] + tmp[0];
                    V5Aux[2*((alfa*N_SPINS+beta)*N_COLS+m)+1] = V5Aux[2*((alfa*N_SPINS+beta)*N_COLS+m)+1] + tmp[1];
	          }
	        }
	      }
              V_TR_MM<Float>( V5Aux, this->GList[2][g1],transpgamma_i1, V5Aux2);
              #pragma omp parallel for
              for(int i_m=0; i_m<imap.size(); i_m++){
	        Float V6Aux[2*N_SPINS*N_SPINS*N_COLS];
                Float temp[2*N_SPINS*N_SPINS];
                int i_mom_f1 = imap[i_m][1];
                int i_mom_f2 = imap[i_m][2];
                if ((i_mom_f2==input_mom_f2) ){
                  for (int alfa=0;alfa<N_SPINS;++alfa){
                    for (int beta=0;beta<N_SPINS;++beta){
		      for (int coloridx=0;coloridx<N_COLS;++coloridx){
		           V6Aux[(alfa*N_SPINS+beta)*2*N_COLS+2*coloridx+0]=srcV6.Corr(t,i_mom_f1,g3,alfa,beta,coloridx)[0];
                           V6Aux[(alfa*N_SPINS+beta)*2*N_COLS+2*coloridx+1]=srcV6.Corr(t,i_mom_f1,g3,alfa,beta,coloridx)[1];
		      }
		    }
		  }

                  //if true multiply by sigma_T(G_f1)

                  if(transpgamma_f1){
                    for(int sc=0; sc<N_SPINS*N_SPINS*N_COLS*2; ++sc)
                      V6Aux[sc] *= gammaTranspSign_scatt[this->GList[4][g3]];
                  }

		  for (int alfa=0;alfa<N_SPINS;++alfa){
		    for (int beta=0; beta<N_SPINS;++beta){
	              int spins = (transp) ? (beta*N_SPINS+alfa)*2 : (alfa*N_SPINS+beta)*2;
		      temp[spins]=0.;
		      temp[spins+1]=0.;
		      for (int coloridx=0; coloridx<N_COLS; ++coloridx){ 
		        temp[spins+0]+=
			  +V5Aux2[2*coloridx+0]*V6Aux[(alfa*N_SPINS+beta)*2*N_COLS+2*coloridx+0]
			  -V5Aux2[2*coloridx+1]*V6Aux[(alfa*N_SPINS+beta)*2*N_COLS+2*coloridx+1];
			temp[spins+1]+=
                          +V5Aux2[2*coloridx+0]*V6Aux[(alfa*N_SPINS+beta)*2*N_COLS+2*coloridx+1]
                          +V5Aux2[2*coloridx+1]*V6Aux[(alfa*N_SPINS+beta)*2*N_COLS+2*coloridx+0];

		      }//n_col
		    }//beta
		  }//alfa


		  if(factor!=NULL){
		    Float aux;
		    for(int ss=0; ss<N_SPINS*N_SPINS; ++ss){
		      aux = temp[2*ss+0]*factor[0] - temp[2*ss+1]*factor[1];
		      temp[2*ss+1] = temp[2*ss+0]*factor[1] + temp[2*ss+1]*factor[0];
		      temp[2*ss+0] = aux;
		    }
		  }
		  for (int g_exti=0; g_exti < n_gammas_exti ; ++ g_exti){
		    for (int g_extf=0; g_extf < n_gammas_extf ; ++ g_extf){
		      GAMMAS_SCATT egammai = this->GList[0][g_exti];
		      GAMMAS_SCATT egammaf = this->GList[1][g_extf];
		      
		      //multiplication with external gammas NB written here! mod in M_pe_GNG
		      M_pe_GNG<Float>( this->Corr(t,i_m,g_exti,g_extf,g1,g2,g3,g4),
                                egammaf, egammai, temp );
		    }//Gextf
                  }//Gexti
                }//if mom
              }//imom
            }//Gf1
          }//Gi1 
        }//Gf2 
      }//Gi2
    }//t
}


//#########################
//#  Initialize diagrams  #
//#########################

//1pt --> "L", 1Gammas
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::initialize_diagram( std::vector<GAMMAS_SCATT> &G_f2, std::string name_of_diagram){

  assert( name_of_diagram=="L" );

  //Gamma list
  this->GList.clear();
  this->GList.push_back( G_f2 ) ;

  //Description
  std::vector<std::vector<GAMMAS_SCATT>> tmpvector= {G_f2};
  std::string tmp="";
  for (int i=0; i<tmpvector.size(); ++i){
    tmp+="{";
    std::vector<GAMMAS_SCATT> elements=tmpvector[i];
    for( int j=0; j<elements.size();++j ){
      if (j==(elements.size()-1)){
        tmp+= GAMMAS_SCATT_STR[elements[j]];
      }
      else{
        tmp+= GAMMAS_SCATT_STR[elements[j]]+",";
      }
    }
    if (i==(tmpvector.size()-1)){
      tmp+="}";
    }
    else{
      tmp+="},";
    }
  }
  this->description = tmp;

  //Groups
  this->groups = {""};

  //Dataset
  this->datasets = {name_of_diagram,};

  //Shape
  this->shape = { (int)(this->GList[0].size()) };

  //initialize
  this->initialize();

  //Offsets
  this->labels="tmg";
  this->setOffsets();

  this->clear_output(true);
}


//2pt --> "P", 2Gammas
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::initialize_diagram( std::vector<GAMMAS_SCATT> &G_i2, std::vector<GAMMAS_SCATT> &G_f2, std::string name_of_diagram){

  assert(( name_of_diagram=="P") ||  (name_of_diagram=="P0DN") ||  (name_of_diagram=="P0UP") ||  (name_of_diagram=="PPDN") ||  (name_of_diagram=="PPUP" ) );

  //Gamma list
  this->GList.clear();
  //this->GList.push_back( apply_gamma5_scatt_gamma(G_i2,RIGHT) );
  //this->GList.push_back( apply_gamma5_scatt_gamma(G_f2,LEFT) );
  this->GList.push_back( G_i2 );
  this->GList.push_back( G_f2 );
  
  //Description
  std::vector<std::vector<GAMMAS_SCATT>> tmpvector= {G_i2, G_f2};
  std::string tmp="";
  for (int i=0; i<tmpvector.size(); ++i){
    tmp+="{";
    std::vector<GAMMAS_SCATT> elements=tmpvector[i];
    for( int j=0; j<elements.size();++j ){
      if (j==(elements.size()-1)){
        tmp+= GAMMAS_SCATT_STR[elements[j]];
      }
      else{
        tmp+= GAMMAS_SCATT_STR[elements[j]]+",";
      }
    }
    if (i==(tmpvector.size()-1)){
      tmp+="}";
    }
    else{
      tmp+="},";
    }
  }
  tmp+="/S1/S2/";

  this->description = tmp;
  
  //Groups
  this->groups = {""};
  
  //Dataset
  this->datasets = {name_of_diagram,};

  //Shape
  this->shape = { (int)(this->GList[0].size())*(int)(this->GList[1].size()) };

  //initialize
  this->initialize();

  //Offsets
  this->labels="tmgg";
  this->setOffsets();

  this->clear_output(true);
}
//3pt ->  PJP
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::initialize_diagram( std::vector<GAMMAS_SCATT> &G_i2, std::vector<GAMMAS_SCATT> &G_f2,std::vector<GAMMAS_SCATT> &G_c,std::string name_of_diagram){

  assert(( name_of_diagram=="PJP") || (name_of_diagram=="PJP_STL") || (name_of_diagram=="PJP_STD") || (name_of_diagram=="PJP_TWOD")  );

  //Gamma list
  this->GList.clear();
  //this->GList.push_back( apply_gamma5_scatt_gamma(G_i2,RIGHT) );
  //this->GList.push_back( apply_gamma5_scatt_gamma(G_f2,LEFT) );
  this->GList.push_back( G_i2 );
  this->GList.push_back( G_f2 );
  this->GList.push_back( G_c );


  //Description
  std::vector<std::vector<GAMMAS_SCATT>> tmpvector= {G_i2, G_f2, G_c};
  std::string tmp="";
  for (int i=0; i<tmpvector.size(); ++i){
    tmp+="{";
    std::vector<GAMMAS_SCATT> elements=tmpvector[i];
    for( int j=0; j<elements.size();++j ){
      if (j==(elements.size()-1)){
        tmp+= GAMMAS_SCATT_STR[elements[j]];
      }
      else{
        tmp+= GAMMAS_SCATT_STR[elements[j]]+",";
      }
    }
    if (i==(tmpvector.size()-1)){
      tmp+="}";
    }
    else{
      tmp+="},";
    }
  }
  if (name_of_diagram=="PJP_STD"){
    tmp+="/x,y,z,t / ";
  }
  else if (name_of_diagram=="PJP_TWOD"){
    tmp+="xy,xz,xt,yx,yz,yt,zx,zy,zt,tx,ty,tz / ";
  }
  tmp+="/S1/S2/";

  this->description = tmp;

  //Groups
  if (name_of_diagram=="PJP"){
    this->groups = {""};
  }
  else{
   char *temporary;
   asprintf(&temporary,"/pi=");
   this->groups ={this->pList().to_string({0},{"pi="})[0], };
   free(temporary);
  }

  //Dataset
  this->datasets = {name_of_diagram,};

  //Shape
  if (name_of_diagram == "PJP_STD"){
    this->shape = { (int)(this->GList[0].size())*(int)(this->GList[1].size())*(int)(this->GList[2].size()),(int)N_DIMS };
  } else if (name_of_diagram == "PJP_TWOD"){
    this->shape = { (int)(this->GList[0].size())*(int)(this->GList[1].size())*(int)(this->GList[2].size()),(int)(N_DIMS*(N_DIMS-1)) };
  }
  else{
    this->shape = { (int)(this->GList[0].size())*(int)(this->GList[1].size())*(int)(this->GList[2].size()) };
  }

  //initialize
  this->initialize();

  //Offsets
  if (name_of_diagram == "PJP_STD"){
    this->labels="tmgggd";
  }
  else if (name_of_diagram == "PJP_TWOD"){
    this->labels="tmgggl";
  }
  else{
    this->labels="tmggg";
  }
  this->setOffsets();

  this->clear_output(true);
}


//2pt --> "N","D", 4Gammas
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::initialize_diagram( std::vector<GAMMAS_SCATT> &eG_i, std::vector<GAMMAS_SCATT> &eG_f, std::vector<GAMMAS_SCATT> &G_i1, std::vector<GAMMAS_SCATT> &G_f1, std::string name_of_diagram){

  //assert( name_of_diagram=="N" || name_of_diagram=="D");
  //Gamma list
  this->GList.clear();
  this->GList.push_back( eG_i );
  this->GList.push_back( eG_f );
  this->GList.push_back( G_i1 );
  this->GList.push_back( G_f1 );

  //Description
  std::vector<std::vector<GAMMAS_SCATT>> tmpvector= {eG_i, eG_f, G_i1, G_f1};
  std::string tmp="";
  for (int i=0; i<tmpvector.size(); ++i){
    tmp+="{";
    std::vector<GAMMAS_SCATT> elements=tmpvector[i];
    for( int j=0; j<elements.size();++j ){
      if (j==(elements.size()-1)){
        tmp+= GAMMAS_SCATT_STR[elements[j]];
      }
      else{
        tmp+= GAMMAS_SCATT_STR[elements[j]]+",";
      }
    }
    if (i==(tmpvector.size()-1)){
      tmp+="}";
    }
    else{
      tmp+="},";
    }
  }
  tmp+="/S1/S2/";

  this->description = tmp;

  //momList

  //Groups
  this->groups={"",};

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
  
  //Offsets
  this->labels="tmggggss";
  this->setOffsets();
             

  this->clear_output(true);
}

//3pt --> "NJN","D", 5Gammas
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::initialize_diagram( std::vector<GAMMAS_SCATT> &eG_i, std::vector<GAMMAS_SCATT> &eG_f, std::vector<GAMMAS_SCATT> &G_i1, std::vector<GAMMAS_SCATT> &G_f1, std::vector<GAMMAS_SCATT> &G_c, std::string name_of_diagram){

  //assert( name_of_diagram=="N" || name_of_diagram=="D");
  //Gamma list
  this->GList.clear();
  this->GList.push_back( eG_i );
  this->GList.push_back( eG_f );
  this->GList.push_back( G_i1 );
  this->GList.push_back( G_f1 );
  this->GList.push_back( G_c );


  //Description
  std::vector<std::vector<GAMMAS_SCATT>> tmpvector= {eG_i, eG_f, G_i1, G_f1, G_c};
  std::string tmp="";
  for (int i=0; i<tmpvector.size(); ++i){
    tmp+="{";
    std::vector<GAMMAS_SCATT> elements=tmpvector[i];
    for( int j=0; j<elements.size();++j ){
      if (j==(elements.size()-1)){
        tmp+= GAMMAS_SCATT_STR[elements[j]];
      }
      else{
        tmp+= GAMMAS_SCATT_STR[elements[j]]+",";
      }
    }
    if (i==(tmpvector.size()-1)){
      tmp+="}";
    }
    else{
      tmp+="},";
    }
  }
  tmp+="/S1/S2/";

  this->description = tmp;

  //momList

  //Groups
  this->groups={"",};

  //Dataset
  this->datasets={name_of_diagram,};

  //Shape
  this->shape={(int)(this->GList[0].size())*
               (int)(this->GList[1].size())*
               (int)(this->GList[2].size())*
               (int)(this->GList[3].size())*
               (int)(this->GList[4].size()),
               N_SPINS*N_SPINS};

  //initialize
  this->initialize();

  //Offsets
  this->labels="tmgggggss";
  this->setOffsets();


  this->clear_output(true);
}



//3pt --> "T", 5Gammas, "T1" 5Gammas +1 fake (cause V3V2reductions works only for 4pt)
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::initialize_diagram( std::vector<GAMMAS_SCATT> &eG_i, std::vector<GAMMAS_SCATT> &eG_f, std::vector<GAMMAS_SCATT> &G_i1, std::vector<GAMMAS_SCATT> &G_i2, std::vector<GAMMAS_SCATT> &G_f, std::string isospin, std::string name_of_diagram){

 //assert( (name_of_diagram=="T") || (name_of_diagram=="T1") );

 //Gamma list
  this->GList.clear();
  this->GList.push_back( eG_i );
  this->GList.push_back( eG_f );
  this->GList.push_back( G_i1 );
  std::size_t found = name_of_diagram.find("Tseq");
  if (found==std::string::npos){
    std::vector<GAMMAS_SCATT> fake_glist={ID,};
    this->GList.push_back( fake_glist );
  }
  this->GList.push_back( G_i2 );//or Gf1
  this->GList.push_back( G_f );//or Gf2

  //Description
  std::vector<std::vector<GAMMAS_SCATT>> tmpvector= {eG_i, eG_f, G_i1, G_i2, G_f};
  std::string tmp="";
  for (int i=0; i<tmpvector.size(); ++i){
    tmp+="{";
    std::vector<GAMMAS_SCATT> elements=tmpvector[i];
    for( int j=0; j<elements.size();++j ){
      if (j==(elements.size()-1)){
        tmp+= GAMMAS_SCATT_STR[elements[j]];
      }
      else{
        tmp+= GAMMAS_SCATT_STR[elements[j]]+",";
      }
    }
    if (i==(tmpvector.size()-1)){
      tmp+="}";
    }
    else{
      tmp+="},";
    }
  }
  this->description=tmp;
  
  //momList
  assert( this->pList().check_eq(0) );

  //Groups
  char *temporary;
  asprintf(&temporary,"%s/pi2=",isospin.c_str());
  this->groups ={this->pList().to_string({0},{temporary})[0], };
  free(temporary);
  
  //Dataset
  this->datasets = {name_of_diagram,};

  //Shape
  int ngammas=1;
  for( auto& g : this->GList )
    ngammas*=g.size();
    
  this->shape = { ngammas, N_SPINS*N_SPINS };

  //initialize
  this->initialize();
 
  //Offsets
  this->labels=( found==std::string::npos ) ? "tmggggggss" : "tmgggggss";
  this->setOffsets();

  this->clear_output(true);
}

//4pt --> "B1","B2","W1","W2","W3","W4","Z1","Z2","Z3","Z4","M","TpiNsink" "D" 6Gammas
//note that here D stands for disconnected
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::initialize_diagram( std::vector<GAMMAS_SCATT> &eG_i, std::vector<GAMMAS_SCATT> &eG_f, std::vector<GAMMAS_SCATT> &G_i1, std::vector<GAMMAS_SCATT> &G_i2, std::vector<GAMMAS_SCATT> &G_f1, std::vector<GAMMAS_SCATT> &G_f2, std::string isospin, std::string name_of_diagram, bool oet){
  
  char letter = name_of_diagram.at(0);
  assert( (letter=='M') || (letter=='B') || (letter=='W') || (letter=='Z') || (letter=='D') );
  if( (letter != 'M') && (letter != 'D') ){
    char number = name_of_diagram.at(1);
    assert(name_of_diagram.length()>1);
    char *namecopy;
    asprintf(&namecopy,"%s",name_of_diagram.c_str()+1);
    int diagram_index=atoi(namecopy); 
    free(namecopy);
    
    if( letter=='B' ) assert( (diagram_index>0) && (diagram_index<21) );
    if( letter=='W' ) assert( (diagram_index>0) && (diagram_index<37) );
    if( letter=='Z' ) assert( (diagram_index>0) && (diagram_index<21) );
  }

  //Gamma list
  this->GList.clear();
  this->GList.push_back( eG_i );
  this->GList.push_back( eG_f );
  this->GList.push_back( G_i1 );
  //std::vector<GAMMAS_SCATT> tmpG = (letter == 'Z') ? apply_gamma5_scatt_gamma(G_i2,RIGHT) : G_i2;
  this->GList.push_back( G_i2 );
  this->GList.push_back( G_f1 );
  //tmpG = (letter == 'Z') ? apply_gamma5_scatt_gamma(G_f2,LEFT) : G_f2;
  this->GList.push_back( G_f2 );


  //Description
  std::vector<std::vector<GAMMAS_SCATT>> tmpvector= {eG_i, eG_f, G_i1, G_i2, G_f1, G_f2};
  std::string tmp="";
  for (int i=0; i<tmpvector.size(); ++i){
    tmp+="{";
    std::vector<GAMMAS_SCATT> elements=tmpvector[i];
    for( int j=0; j<elements.size();++j ){
      if (j==(elements.size()-1)){
        tmp+= GAMMAS_SCATT_STR[elements[j]];
      }
      else{
        tmp+= GAMMAS_SCATT_STR[elements[j]]+",";
      }
    }
    if (i==(tmpvector.size()-1)){
      tmp+="}";
    }
    else{
      tmp+="},";
    }
  }
  tmp+="/S1/S2/";

  this->description = tmp;
  
  //Groups
  char *temporary;
  if (oet ==true){
    asprintf(&temporary,"%s/pf2=",isospin.c_str());
    this->groups = {this->pList().to_string({2},{temporary})[0],};
  }
  else{
    asprintf(&temporary,"%s/pi2=",isospin.c_str());
    this->groups = {this->pList().to_string({0},{temporary})[0],};
  }
  free(temporary);

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
  
  //Offsets
  this->labels="tmggggggss";
  this->setOffsets();

  this->clear_output(true);
}

//2pt with insertion --> "B1","B2","W1","W2","W3","W4","Z1","Z2","Z3","Z4","M","TpiNsink" "D" 7Gammas
//note that here D stands for disconnected
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::initialize_diagram( std::vector<GAMMAS_SCATT> &eG_i, std::vector<GAMMAS_SCATT> &eG_f, std::vector<GAMMAS_SCATT> &G_i1, std::vector<GAMMAS_SCATT> &G_i2, std::vector<GAMMAS_SCATT> &G_f1, std::vector<GAMMAS_SCATT> &G_f2, std::vector<GAMMAS_SCATT> &G_c, std::string isospin, std::string name_of_diagram){


  //Gamma list
  this->GList.clear();
  this->GList.push_back( eG_i );
  this->GList.push_back( eG_f );
  this->GList.push_back( G_i1 );
  //std::vector<GAMMAS_SCATT> tmpG = (letter == 'Z') ? apply_gamma5_scatt_gamma(G_i2,RIGHT) : G_i2;
  this->GList.push_back( G_i2 );
  this->GList.push_back( G_f1 );
  //tmpG = (letter == 'Z') ? apply_gamma5_scatt_gamma(G_f2,LEFT) : G_f2;
  this->GList.push_back( G_f2 );
  this->GList.push_back( G_c );


  //Description
  std::vector<std::vector<GAMMAS_SCATT>> tmpvector= {eG_i, eG_f, G_i1, G_i2, G_f1, G_f2, G_c};
  std::string tmp="";
  for (int i=0; i<tmpvector.size(); ++i){
    tmp+="{";
    std::vector<GAMMAS_SCATT> elements=tmpvector[i];
    for( int j=0; j<elements.size();++j ){
      if (j==(elements.size()-1)){
        tmp+= GAMMAS_SCATT_STR[elements[j]];
      }
      else{
        tmp+= GAMMAS_SCATT_STR[elements[j]]+",";
      }
    }
    if (i==(tmpvector.size()-1)){
      tmp+="}";
    }
    else{
      tmp+="},";
    }
  }
  tmp+="/S1/S2/";

  this->description = tmp;

  //Groups
  char *temporary;
  asprintf(&temporary,"%s/pi2=",isospin.c_str());
  this->groups = {this->pList().to_string({0},{temporary})[0],};
  
  free(temporary);

  //Dataset
  this->datasets = {name_of_diagram};

  //Shape
  this->shape={(int)(this->GList[0].size())*
               (int)(this->GList[1].size())*
               (int)(this->GList[2].size())*
               (int)(this->GList[3].size())*
               (int)(this->GList[4].size())*
               (int)(this->GList[5].size())*
	       (int)(this->GList[6].size()),
               N_SPINS*N_SPINS};

  //initialize
  this->initialize();

  //Offsets
  this->labels="tmgggggggss";
  this->setOffsets();

  this->clear_output(true);
}



//##############
//#  Diagrams  #
//##############
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::D1ii_diagrams(PLEGMA_ScattCorrelator<Float> &srcV3, PLEGMA_ScattCorrelator<Float> &srcV2, Float *loopcontribution, const int ig_i2, const int diagram_index, bool accum) {

  if(ig_i2 >= this->GList[3].size()) PLEGMA_error("ig_i2 = %d but Gi2 list size is %d\n", ig_i2, this->GList[3].size() );

  this->clear_output(!accum, 5, ig_i2);

  //Diagrams (1,5), (2,6), (3,7) and (4,8) are structurally the same the 
  //only difference between them is the type of loop(pipi_aux): UP and 
  //DN in the former respectively in the latter

  float signofFactor=-1.; 
  Float factor[2]={signofFactor*loopcontribution[0],signofFactor*loopcontribution[1]};//-1 from eqs. (20),(23), ....
  
  switch( diagram_index ){
    case 1:
      this->V3V2reduction(        srcV3, srcV2, 0, false, ig_i2, true, factor, false);//checked FP
      break;
    case 2:
      this->V3V2reduction(        srcV3, srcV2, 2, false, ig_i2, true, factor, true);//checked FP
      break;
    case 3:
      this->V3V2reduction_matrix( srcV3, srcV2, 1, false, ig_i2, true, factor, false);//checked FP
      break;
    case 4:
      this->V3V2reduction_matrix( srcV3, srcV2, 0, false, ig_i2, false, factor, true);//checked FP
      break;
    case 9:
      this->V3V2reduction(        srcV3, srcV2, 2,  true, ig_i2, false, factor, true);//checked FP
      break;
    case 10:
      this->V3V2reduction_matrix( srcV3, srcV2, 0, false, ig_i2, false, factor, true);//checked FP
      break;
    case 13:
      this->V3V2reduction(        srcV3, srcV2, 2, false, ig_i2, true,  factor, false);//checked FP
      break;
    case 14:
      this->V3V2reduction_matrix( srcV3, srcV2, 0, false, ig_i2, false, factor, false);//checked FP
      break;
    case 15:
      this->V3V2reduction(        srcV3, srcV2, 2, true, ig_i2,  true,  factor, false);//checked FP
      break;
    case 16:
      this->V3V2reduction_matrix( srcV3, srcV2, 1, false, ig_i2, false, factor, false);//checked FP
      break;
    default:
      PLEGMA_error("This value of D1ii diagram index does not exists, please check your inputs in piNdiagrams.cpp");
  }

}
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::B_diagrams(PLEGMA_ScattCorrelator<Float> &srcV3, PLEGMA_ScattCorrelator<Float> &srcV2, int ig_i2, int diagram_index, bool accum, bool oet) {

  if(ig_i2 >= this->GList[3].size()) PLEGMA_error("ig_i2 = %d but Gi2 list size is %d\n", ig_i2, this->GList[3].size() );

  this->clear_output(!accum, 5, ig_i2); 

  Float factor[2]={-1.,0.};//-1 from eqs. (20),(23), ....
  
  switch( diagram_index ){
    case 1:
      this->V3V2reduction(        srcV3, srcV2, 2, true,  ig_i2, false, factor, true, oet);  //checked FP
      break;
    case 2:
      this->V3V2reduction(        srcV3, srcV2, 0, false, ig_i2, false, factor, true, oet);  //checked FP
      break;
    case 3:
      this->V3V2reduction(        srcV3, srcV2, 0, false, ig_i2, true,  factor, false, oet); //checked FP
      break;
    case 4:
      this->V3V2reduction(        srcV3, srcV2, 2, false, ig_i2, true,  factor, true, oet); //checked FP
      break;
    case 5:
      this->V3V2reduction_matrix( srcV3, srcV2, 1, false, ig_i2, true,  factor, false, oet); //checked FP
      break;
    case 6:
      this->V3V2reduction_matrix( srcV3, srcV2, 0, false, ig_i2, false, factor, true, oet); //checked FP
      break;
    case 7:
      this->V3V2reduction(        srcV3, srcV2, 2, true,  ig_i2, false, factor, true, oet);//checked FP
      break;
    case 8:
      this->V3V2reduction(        srcV3, srcV2, 0, false, ig_i2, false, factor, true, oet);//checked FP
      break;
    case 9:
      this->V3V2reduction(        srcV3, srcV2, 2, true,  ig_i2, true,  factor, false, oet);//checked FP
      break;
    case 10:
      this->V3V2reduction(        srcV3, srcV2, 1, false, ig_i2, true,  factor, true, oet);//checked FP
      break;
    case 11:
      this->V3V2reduction_matrix( srcV3, srcV2, 1, false, ig_i2, false, factor, false, oet);//checked FP
      break;
    case 12:
      this->V3V2reduction_matrix( srcV3, srcV2, 0, false, ig_i2, true, factor, true, oet);//checked FP
      break;
    case 13:
      this->V3V2reduction(        srcV3, srcV2, 1, false, ig_i2, true,  factor, false, oet);//checked FP
      break;
    case 14:
      this->V3V2reduction_matrix( srcV3, srcV2, 0, false, ig_i2, true,  factor, false, oet);//checked FP
      break;
    case 15:
      this->V3V2reduction(        srcV3, srcV2, 0, false, ig_i2, true,  factor, false, oet);//checked FP
      break;
    case 16:
      this->V3V2reduction_matrix( srcV3, srcV2, 1, false, ig_i2, true,  factor, false, oet);//checked FP
      break;
    case 17:
      this->V3V2reduction(        srcV3, srcV2, 2, false, ig_i2, true,  factor, false, oet); //checked FP
      break;
    case 18:
      this->V3V2reduction_matrix( srcV3, srcV2, 0, false, ig_i2, false, factor, false, oet); //checked FP
      break;
    case 19:
      this->V3V2reduction(        srcV3, srcV2, 2, true , ig_i2, true,  factor, false, oet); //checked FP
      break;
    case 20:
      this->V3V2reduction_matrix( srcV3, srcV2, 1, false, ig_i2, false, factor, false, oet);//checked FP
      break;
    default:
      PLEGMA_error("This value of B diagram index does not exists, please check your inputs in piNdiagrams.cpp");
  }
  //Note that the last true or false indicates the of transp(G_f1).

}


template<typename Float>
void PLEGMA_ScattCorrelator<Float>::W_diagrams_oet(PLEGMA_ScattCorrelator<Float> &srcV6, std::shared_ptr<Float> &Phi0, std::vector<PLEGMA_Vector<Float>*> &Phi1, int input_mom_f2, int diagram_index, bool accum){

  this->clear_output(!accum);
  Float factor[2]={-1,0};

  switch( diagram_index ){
    case 1:
      this->V5V6reduction_matrix(srcV6, Phi0, Phi1, input_mom_f2,  false, true, true, factor);
      break;
    case 2: 
      this->V5V6reduction(srcV6, Phi0, Phi1,  input_mom_f2, 1, 0, false, false, true, factor);
      break;
    case 3:
      this->V5V6reduction(srcV6, Phi0, Phi1,  input_mom_f2, 1, 0, false, false, false, factor);
      break;
    case 4:
      this->V5V6reduction_matrix(srcV6, Phi0, Phi1, input_mom_f2, false, true, false, factor);
      break;
    case 5:
    case 6:
      this->V5V6reduction(srcV6, Phi0, Phi1,  input_mom_f2, 1, 0, false, true, false, factor);
      break;
    case 7:
    case 8:
      this->V5V6reduction(srcV6, Phi0, Phi1,  input_mom_f2, 1, 1, false, true, false, factor);
      break;
    case 9:
      this->V5V6reduction_matrix(srcV6, Phi0, Phi1, input_mom_f2, false, true, false, factor);
      break;
    case 10:
      this->V5V6reduction_matrix(srcV6, Phi0, Phi1, input_mom_f2, false, true, true, factor);
      break;
    case 11:
      this->V5V6reduction(srcV6, Phi0, Phi1,  input_mom_f2, 1, 0, false, false, false, factor);
      break;
    case 12:
      this->V5V6reduction(srcV6, Phi0, Phi1,  input_mom_f2, 1, 0, false, false, true, factor);
      break;
    case 13:
      this->V5V6reduction_matrix(srcV6, Phi0, Phi1, input_mom_f2, false, false, false, factor);
      break;
    case 14:
      this->V5V6reduction_matrix(srcV6, Phi0, Phi1, input_mom_f2, false, false, true, factor);
      break;
    case 15:
      this->V5V6reduction(srcV6, Phi0, Phi1,  input_mom_f2, 1, 1, false, false, false, factor);
      break;
    case 16:
      this->V5V6reduction(srcV6, Phi0, Phi1,  input_mom_f2, 1, 1, false, false, true, factor);
      break;
    case 17:
    case 18:
      this->V5V6reduction_matrix(srcV6, Phi0, Phi1, input_mom_f2, false, false, false, factor);
      break;
    case 19:
    case 20:
      this->V5V6reduction(srcV6, Phi0, Phi1,  input_mom_f2, 1, 1, false, false, false, factor);
      break;
    case 21:
      this->V5V6reduction(srcV6, Phi0, Phi1,  input_mom_f2, 1, 0, false, true, false, factor);
      break;
    case 22:
      this->V5V6reduction(srcV6, Phi0, Phi1,  input_mom_f2, 1, 0, false, true, true, factor);
      break;
    case 23:
      this->V5V6reduction(srcV6, Phi0, Phi1,  input_mom_f2, 1, 1, false, true, false, factor);
      break;
    case 24:
      this->V5V6reduction(srcV6, Phi0, Phi1,  input_mom_f2, 1, 1, false, true, true, factor);
      break;
    case 25:
      this->V5V6reduction_matrix(srcV6, Phi0, Phi1, input_mom_f2,  false, false, true, factor);
      break;
    case 26:
      this->V5V6reduction(srcV6, Phi0, Phi1, input_mom_f2, 1, 1,  false, false, true, factor);
      break;
    case 27:
      this->V5V6reduction_matrix(srcV6, Phi0, Phi1, input_mom_f2,  false, false, false, factor);
      break;
    case 28:
      this->V5V6reduction(srcV6, Phi0, Phi1, input_mom_f2, 1, 1,  false, false, false, factor);
      break;
    case 29:
      this->V5V6reduction(srcV6, Phi0, Phi1, input_mom_f2, 1, 0,  false, true, true, factor);
      break;
    case 30:
      this->V5V6reduction(srcV6, Phi0, Phi1, input_mom_f2, 1, 1,  false, true, true, factor);
      break;
    case 31:
      this->V5V6reduction(srcV6, Phi0, Phi1, input_mom_f2, 1, 0,  false, true, false, factor);
      break;
    case 32:
      this->V5V6reduction(srcV6, Phi0, Phi1, input_mom_f2, 1, 1,  false, true, false, factor);
      break;
    case 33:
      this->V5V6reduction_matrix(srcV6, Phi0, Phi1, input_mom_f2,  false, true, false, factor);
      break;
    case 34:
      this->V5V6reduction(srcV6, Phi0, Phi1, input_mom_f2, 1, 0,  false, false, false, factor);
      break;
    case 35:
      this->V5V6reduction_matrix(srcV6, Phi0, Phi1, input_mom_f2,  false, true, false, factor);
      break;
    case 36:
      this->V5V6reduction(srcV6, Phi0, Phi1, input_mom_f2, 1, 0,  false, false, false, factor);
      break;
    default:
      PLEGMA_error("This value of W diagram oet index does not exists, please check your inputs in piNdiagrams_oet.cpp");
  }
}


template<typename Float>
void PLEGMA_ScattCorrelator<Float>::W_diagrams(PLEGMA_ScattCorrelator<Float> &srcV3, PLEGMA_ScattCorrelator<Float> &srcV2, int ig_i2, int diagram_index, bool accum){

  //if( (diagramm_index != 1) && (diagramm_index !=2 ) &&  (diagramm_index != 3) &&  (diagramm_index != 4)   )
  //  PLEGMA_error("diagramm_index %d out of range (1,2,3 or 4)\n",diagramm_index);
  if(ig_i2 >= this->GList[3].size()) PLEGMA_error("ig_i2 = %d but Gi2 list size is %d\n", ig_i2, this->GList[3].size() );

  this->clear_output(!accum, 5, ig_i2); 

  Float factor[2]={-1.,0.};//-1 from eqs. (28),(31),(34),(37), ....

  switch( diagram_index ){
    case 1:
      this->V3V2reduction(        srcV3, srcV2, 2, true,  ig_i2, true,  factor, true); //checked FP,last true because of transp(G_f1)
      break;
    case 2:
      this->V3V2reduction_matrix( srcV3, srcV2, 1, false, ig_i2, false, factor, true);//checked FP,last true because of transp(G_f1)
      break;
    case 3:
      this->V3V2reduction_matrix( srcV3, srcV2, 1, false, ig_i2, true,  factor, true);//checked FP,last true because of transp(G_f1)
      break;
    case 4:
      this->V3V2reduction(        srcV3, srcV2, 0, false, ig_i2, true,  factor, true);//checked FP,last true because of transp(G_f1)
      break;
    case 5:
    case 12:
      this->V3V2reduction_matrix( srcV3, srcV2, 0, false, ig_i2, true,  factor, false);//checked FP
      break;
    case 6:
    case 11:
      this->V3V2reduction_matrix( srcV3, srcV2, 1, false, ig_i2, true,  factor, false);//checked FP
      break;
    case 7:
    case 10:
      this->V3V2reduction(        srcV3, srcV2, 1, false, ig_i2, true,  factor, false);//checked FP
      break;
    case 8:
    case 9:
      this->V3V2reduction(        srcV3, srcV2, 0, false, ig_i2, true,  factor, false);//checked FP
      break;
    case 13:
      this->V3V2reduction(        srcV3, srcV2, 0, false, ig_i2, false, factor, true); //checked FP,last true because of transp(G_f1)
      break;
    case 14:
      this->V3V2reduction(        srcV3, srcV2, 2, true,  ig_i2, false, factor, true); //checked FP,last true because of transp(G_f1)
      break;
    case 15:
      this->V3V2reduction(        srcV3, srcV2, 2, true,  ig_i2, false, factor, true); //checked FP,last true because of transp(G_f1)
      break;
    case 16:
      this->V3V2reduction(        srcV3, srcV2, 0, false, ig_i2, false, factor, true); //checked FP,last true because of transp(G_f1)
      break;
    case 17:
      this->V3V2reduction(        srcV3, srcV2, 2, false, ig_i2, false, factor, false); //checked FP
      break;
    case 18:
      this->V3V2reduction(        srcV3, srcV2, 2, true,  ig_i2, false, factor, false); //checked FP
      break;
    case 19:
      this->V3V2reduction(        srcV3, srcV2, 1, false, ig_i2, false, factor, false); //checked FP
      break;
    case 20:
      this->V3V2reduction(        srcV3, srcV2, 0, false, ig_i2, false, factor, false); //checked FP
      break;
    case 21:
      this->V3V2reduction_matrix( srcV3, srcV2, 1, false, ig_i2, false, factor, true); //checked FP,last true because of transp(G_f1)
      break;
    case 22:
      this->V3V2reduction_matrix( srcV3, srcV2, 1, false, ig_i2, true,  factor, true); //checked FP,last true because of transp(G_f1)
      break;
    case 23:
      this->V3V2reduction(        srcV3, srcV2, 2, true,  ig_i2, true,  factor, true); //checked FP,last true because of transp(G_f1)
      break;
    case 24:
      this->V3V2reduction(        srcV3, srcV2, 0, false, ig_i2, true,  factor, true);//checked FP,last true because of transp(G_f1)
      break;
    case 25:
      this->V3V2reduction(        srcV3, srcV2, 1, false, ig_i2, false, factor, false); //checked FP
      break;
    case 26:
      this->V3V2reduction(        srcV3, srcV2, 2, false, ig_i2, false, factor, false); //checked FP
      break;
    case 27:
      this->V3V2reduction(        srcV3, srcV2, 0, false, ig_i2, false, factor, false); //checked FP
      break;
    case 28:
      this->V3V2reduction(        srcV3, srcV2, 2, true,  ig_i2, false, factor, false); //checked FP
      break;
    case 29:
      this->V3V2reduction_matrix( srcV3, srcV2, 0, false, ig_i2, false,  factor, false);//checked FP
      break;
    case 30:
      this->V3V2reduction(        srcV3, srcV2, 2, false, ig_i2, true,  factor, false);//checked FP
      break;
    case 31:
      this->V3V2reduction_matrix( srcV3, srcV2, 1, false, ig_i2, false, factor, false);//checked FP
      break;
    case 32:
      this->V3V2reduction(        srcV3, srcV2, 2, true,  ig_i2, true,  factor, false);//checked FP
      break;
    case 33:
      this->V3V2reduction(        srcV3, srcV2, 2, false, ig_i2, true,  factor, false);//checked FP
      break;
    case 34:
      this->V3V2reduction_matrix( srcV3, srcV2, 0, false, ig_i2, false, factor, false);//checked FP
      break;
    case 35:
      this->V3V2reduction(        srcV3, srcV2, 2, true,  ig_i2, true,  factor, false);//checked FP
      break;
    case 36:
      this->V3V2reduction_matrix( srcV3, srcV2, 1, false, ig_i2, false, factor, false);//checked FP
      break;
    default:
      PLEGMA_error("This value of W diagram index does not exists, please check your inputs in piNdiagrams.cpp");
  }//switch (diagram_index)

}

//void PLEGMA_ScattCorrelator<Float>::V3V2reduction( PLEGMA_ScattCorrelator<Float> &srcV3,PLEGMA_ScattCorrelator<Float> &srcV2, int index_abs, bool transp, int g0, bool transpgamma_i1, Float* factor, bool transpgamma_f1, bool oet) {

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::Z_diagrams_without_dilution_check(PLEGMA_ScattCorrelator<Float> &srcV3,
                                                PLEGMA_ScattCorrelator<Float> &srcV2, int i_g_i2,
                                                std::string proporder, int diagram_number, bool transp_i1, bool transp_f1, bool accum ){
  this->clear_output(!accum, 5, i_g_i2);
  Float factor[2]={-1.,0.};//-1 from eqs. (28),(31),(34),(37), ....
  if (proporder== "PSS"){
    switch (diagram_number){
     case 1:
       this->V3V2reduction( srcV3, srcV2, 1, false, i_g_i2, transp_i1, factor,transp_f1); 
       break;
     case 2:
       this->V3V2reduction_matrix( srcV3, srcV2, 0, false, i_g_i2, transp_i1, factor,transp_f1);
       break;
    }
  }
  else if (proporder =="SPS"){
    switch (diagram_number){
    case 1:
       this->V3V2reduction( srcV3, srcV2, 2, true, i_g_i2, transp_i1, factor,transp_f1);
       break;
     case 2:
       this->V3V2reduction( srcV3, srcV2, 0, false, i_g_i2, transp_i1, factor,transp_f1);
       break;
    }
  }
  else if (proporder =="SSP"){
    switch (diagram_number){
    case 1:
       this->V3V2reduction_matrix( srcV3, srcV2, 1, false, i_g_i2, transp_i1, factor,transp_f1);
       break;
    case 2:
       this->V3V2reduction( srcV3, srcV2, 0, false, i_g_i2, transp_i1, factor,transp_f1);
       break;
    }
  }
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::Z_diagrams_without_dilution(PLEGMA_ScattCorrelator<Float> &srcV3, 
                                                PLEGMA_ScattCorrelator<Float> &srcV2, int i_g_i2,
                                                int diagramm_index, bool accum ){
  //if( (diagramm_index != 1) && (diagramm_index !=2 ) &&  (diagramm_index != 3) &&  (diagramm_index != 4)   )
  //  PLEGMA_error("diagramm_index %d out of range (1,2,3 or 4)\n",diagramm_index);

  this->clear_output(!accum, 5, i_g_i2);

  Float factor[2]={-1.,0.};//-1 from eqs. (28),(31),(34),(37), ....
  switch (diagramm_index){
  case 1:
    this->V3V2reduction( srcV3, srcV2, 1, false, i_g_i2, true, factor,false); //checked FP
    break;
  case 2:
    this->V3V2reduction_matrix( srcV3, srcV2, 0, false, i_g_i2, true, factor, false);//checked FP
    break;
  case 3:
    this->V3V2reduction_matrix( srcV3, srcV2, 1, false, i_g_i2, true, factor, false);//checked FP
    break;
  case 4:
    this->V3V2reduction( srcV3, srcV2, 0, false, i_g_i2, true, factor, false);//checked FP
    break;
  case 5:
    this->V3V2reduction( srcV3, srcV2, 0, false, i_g_i2, true, factor, false);//checked FP
    break;
  case 6:
    this->V3V2reduction( srcV3, srcV2, 1, false, i_g_i2, true, factor, false);//checked FP
    break;
  case 7:
    this->V3V2reduction_matrix( srcV3, srcV2, 1, false, i_g_i2, true, factor, false);//checked FP
    break;
  case 8:
    this->V3V2reduction_matrix( srcV3, srcV2, 0, false, i_g_i2, true, factor, false);//checked FP
    break;
  case 9:
    this->V3V2reduction( srcV3, srcV2, 2, true, i_g_i2, false, factor, true); //checked FP
    break;
  case 10:
    this->V3V2reduction( srcV3, srcV2, 0, false, i_g_i2, false, factor, true); //checked FP
    break;
  case 11:
    this->V3V2reduction( srcV3, srcV2, 2, true, i_g_i2,  true, factor, false); //checked FP
    break;
  case 12:
    this->V3V2reduction( srcV3, srcV2, 2, false,i_g_i2, true, factor, false); //checked FP
    break;
  case 13:
    this->V3V2reduction_matrix( srcV3, srcV2, 1, false, i_g_i2, false, factor, false); //checked FP
    break;
  case 14:
    this->V3V2reduction_matrix( srcV3, srcV2, 0, false, i_g_i2, false, factor, false); //checked FP
    break;
  case 15:
    this->V3V2reduction( srcV3, srcV2, 0, false, i_g_i2, false, factor, true);//checked FP
  //this->V3V2reduction( srcV3[lambda], srcV2[kappa], 0, false, g2, false, g, true); //checked FP
    break;
  case 16:
    this->V3V2reduction( srcV3, srcV2, 2, true, i_g_i2, false, factor, true);//checked FP
  //this->V3V2reduction( srcV3[lambda], srcV2[kappa], 2, true, g2,  false, g, true); //checked FP
    break;
  case 17:
    this->V3V2reduction( srcV3, srcV2, 2, true, i_g_i2, true, factor, false);//checked FP
  //this->V3V2reduction( srcV3[lambda], srcV2[kappa], 2, true, g2, true, g, false);//checked FP
    break;
  case 18:
    this->V3V2reduction_matrix( srcV3, srcV2, 1, false, i_g_i2, false, factor, false);//checked FP
  //this->V3V2reduction_matrix( srcV3[lambda], srcV2[kappa], 1, false, g2, false, g, false);//checked FP
    break;
  case 19:
    this->V3V2reduction( srcV3, srcV2, 2, false, i_g_i2, true, factor, false);//checked FP
  //this->V3V2reduction( srcV3[lambda], srcV2[kappa], 2, false, g2, true, g, false); //checked FP
    break;
  case 20:
    this->V3V2reduction_matrix( srcV3, srcV2, 0, false, i_g_i2, false, factor, false);//checked FP
  //this->V3V2reduction_matrix( srcV3[lambda], srcV2[kappa], 0, false, g2, false, g, false);//checked FP
    break;
  default:
    PLEGMA_error("This value of Z diagram index does not exists, please check your inputs in piNdiagrams.cpp");
  }//switch (Diagram index)
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::Z_diagrams(std::array<PLEGMA_ScattCorrelator<Float>,4> (&srcV3),
                                                std::array<PLEGMA_ScattCorrelator<Float>,4> (&srcV2),
						int diagramm_index, bool accum ){

  if( (diagramm_index <1) || (diagramm_index >20 ) )
    PLEGMA_error("diagramm_index %d out of range (1..20)\n",diagramm_index);

  this->clear_output(!accum); 

  for (int g2=0; g2<this->GList[3].size(); ++g2 ){
    //GAMMAS_SCATT gammai2 = this->GList[3][g2];
    GAMMAS_SCATT gammai2_t_gamma5 = apply_g5( this->GList[3][g2], RIGHT);
    for (int n=0; n<4; ++n){
      int kappa = gammaInd_scatt[gammai2_t_gamma5][n][0]; 
      int lambda =gammaInd_scatt[gammai2_t_gamma5][n][1];
      Float g[2];
      g[1] = -gamma_scatt[gammai2_t_gamma5][n][1]; //-1 from eqs. (41),(44),(47),(50)
      g[0] = -gamma_scatt[gammai2_t_gamma5][n][0]; //-1 from eqs. (41),(44),(47),(50)

      switch (diagramm_index){
        case 1:
	  this->V3V2reduction( srcV3[lambda], srcV2[kappa], 1, false, g2, true, g, false); //checked FP
          break;
        case 2:          
	  this->V3V2reduction_matrix( srcV3[lambda], srcV2[kappa], 0, false, g2, true, g, false);//checked FP
          break;
        case 3:
	  this->V3V2reduction_matrix( srcV3[lambda], srcV2[kappa], 1, false, g2, true, g, false);//checked FP
          break;
        case 4:
	  this->V3V2reduction( srcV3[lambda], srcV2[kappa], 0, false, g2, true, g, false);//checked FP
          break;
        case 5:
          this->V3V2reduction( srcV3[lambda], srcV2[kappa], 0, false, g2, true, g, false);//checked FP
          break;
        case 6:
          this->V3V2reduction( srcV3[lambda], srcV2[kappa], 1, false, g2, true, g, false);//checked FP
          break;
        case 7:
          this->V3V2reduction_matrix( srcV3[lambda], srcV2[kappa], 1, false, g2, true, g, false);//checked FP
          break;
        case 8:
          this->V3V2reduction_matrix( srcV3[lambda], srcV2[kappa], 0, false, g2, true, g, false);//checked FP
          break;
        case 9:
          this->V3V2reduction( srcV3[lambda], srcV2[kappa], 2, true, g2, false, g, true); //checked FP
          break;
        case 10:
          this->V3V2reduction( srcV3[lambda], srcV2[kappa], 0, false, g2, false, g, true); //checked FP
          break;
        case 11:
          this->V3V2reduction( srcV3[lambda], srcV2[kappa], 2, true, g2,  true, g, false); //checked FP
          break;
        case 12:
          this->V3V2reduction( srcV3[lambda], srcV2[kappa], 2, false, g2, true, g, false); //checked FP 
          break;
        case 13:
          this->V3V2reduction_matrix( srcV3[lambda], srcV2[kappa], 1, false, g2, false, g, false); //checked FP
          break;
        case 14:
          this->V3V2reduction_matrix( srcV3[lambda], srcV2[kappa], 0, false, g2, false, g, false); //checked FP
          break;
        case 15:
          this->V3V2reduction( srcV3[lambda], srcV2[kappa], 0, false, g2, false, g, true); //checked FP
          break;
        case 16:
          this->V3V2reduction( srcV3[lambda], srcV2[kappa], 2, true, g2,  false, g, true); //checked FP
          break;
        case 17:
          this->V3V2reduction( srcV3[lambda], srcV2[kappa], 2, true, g2, true, g, false);//checked FP
          break;
        case 18:
          this->V3V2reduction_matrix( srcV3[lambda], srcV2[kappa], 1, false, g2, false, g, false);//checked FP
          break;
        case 19:
          this->V3V2reduction( srcV3[lambda], srcV2[kappa], 2, false, g2, true, g, false); //checked FP
          break;
        case 20:
          this->V3V2reduction_matrix( srcV3[lambda], srcV2[kappa], 0, false, g2, false, g, false);//checked FP
          break;
        default:
          PLEGMA_error("This value of Z diagram index does not exists, please check your inputs in piNdiagrams.cpp");
      }//switch (Diagram index)

    }//n -> nonzero elems of Gi2
  }//loop over G_i2 matrix

}

//here pi2 is looped outside in the building of the stocastic propagator. NB for moms_red I expect that pi2 is the same! Phi_0[s] is the stocastic propagator at zero momentum and spin s, Phi_1 with momentum pi2
//GList only 2 gammas G_i2, G_f2
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::P_diagrams( std::vector<PLEGMA_Vector<Float>*> &Phi_0, std::vector<PLEGMA_Vector<Float>*> &Phi_1, int i_pi2, bool accum){

  assert(i_pi2<this->pList().size());

  std::vector<std::vector<int>> momlist(this->pList().pi(0));
  if(i_pi2!=-1)
    momlist = std::vector<std::vector<int>>(1,this->pList().pi(0)[i_pi2]);

  if(i_pi2==-1)
    this->clear_output(!accum);
  else
    this->clear_output(!accum,1,i_pi2);
 
  //++++++++ MESON-MESON +++++++++

  //mom_pi2 can be 1 mom or a list of moms

  //aux PLEGMA_SC for PhixGxPhi multiplications
  site source=site({0,0,0,this->getSource()[3]});
  PLEGMA_ScattCorrelator<Float> pipi_aux(source, momlist, this->getTotalT());

  int N_moms = pipi_aux.Nmoms();
  assert( N_moms==1 || i_pi2==-1 );
  int n_gammas_i2 = this->GList[0].size();
  int n_gammas_f2 = this->GList[1].size();
  int TIME = this->localT();

  //loop over G_i2
  for(int gi2=0; gi2<n_gammas_i2; ++gi2){
    //GAMMAS_SCATT G_i2=this->GList[0][gi2];
    GAMMAS_SCATT G_i2= apply_g5( this->GList[0][gi2], RIGHT );
    for(int nz_e=0; nz_e<4; ++nz_e){
      int alfa = gammaInd_scatt[G_i2][nz_e][0];
      int beta = gammaInd_scatt[G_i2][nz_e][1];
      Float g[2];
      g[1] = -gamma_scatt[G_i2][nz_e][1]; //-1 from eq.(13)
      g[0] = -gamma_scatt[G_i2][nz_e][0]; //-1 from eq.(13)

      PLEGMA_Vector<Float> phi0beta;
      PLEGMA_Vector<Float> phi1alfa;
      phi0beta.copy(*Phi_0[beta],HOST);
      phi0beta.load();
      phi1alfa.copy(*Phi_1[alfa],HOST);
      phi1alfa.load();


      //PhixGf2xPhi
      //pipi_aux.PhiPhi( Phi_0[beta], this->GList[1], Phi_1[alfa]); //T x N_moms x n_gammas_f2
      std::vector<GAMMAS_SCATT> tmpGf2 = apply_gamma5_scatt_gamma( this->GList[1], LEFT);
      pipi_aux.PhiPhi( phi0beta, tmpGf2, phi1alfa); //T x N_moms x n_gammas_f2
    
      if(i_pi2==-1){
        for( int im=0; im<N_moms; ++im)
          for( int t=0; t<TIME; ++t)
            for( int gf2=0; gf2<n_gammas_f2; ++gf2)
              x_pe_cy( this->Corr(t,im,gi2,gf2), g, pipi_aux.Corr(t,im,gf2), 1);
      }
      else{
        for( int t=0; t<TIME; ++t)
          for( int gf2=0; gf2<n_gammas_f2; ++gf2)
            x_pe_cy( this->Corr(t,i_pi2,gi2,gf2), g, pipi_aux.Corr(t,0,gf2), 1);
      }
    }//nonzero elems G_i2
  }//loop over G_i2 matrix

}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::P_diagrams( PLEGMA_Vector<Float> &Phi_0, PLEGMA_Vector<Float> &Phi_1, int i_pi2, bool accum){

  assert(i_pi2<this->pList().size());

  std::vector<std::vector<int>> momlist(this->pList().pi(0));
  if(i_pi2!=-1)
    momlist = std::vector<std::vector<int>>(1,this->pList().pi(0)[i_pi2]);

  if(i_pi2==-1)
    this->clear_output(!accum);
  else
    this->clear_output(!accum,1,i_pi2);

  //++++++++ MESON-MESON +++++++++

  //mom_pi2 can be 1 mom or a list of moms

  //aux PLEGMA_SC for PhixGxPhi multiplications
  site source=site({0,0,0,this->getSource()[3]});
  PLEGMA_ScattCorrelator<Float> pipi_aux(source, momlist, this->getTotalT());

  int N_moms = pipi_aux.Nmoms();
  assert( N_moms==1 || i_pi2==-1 );
  int n_gammas_i2 = this->GList[0].size();
  int n_gammas_f2 = this->GList[1].size();
  int TIME = this->localT();

  PLEGMA_Vector<Float> vectortmp(BOTH);



  //loop over G_f2
  for(int gi2=0; gi2<n_gammas_i2; ++gi2){
    //GAMMAS_SCATT G_i2=this->GList[0][gi2];
    GAMMAS_SCATT G_i2= apply_g5( this->GList[0][gi2], RIGHT );
    vectortmp.copy(Phi_1);
    vectortmp.apply_gamma_scatt(G_i2,RIGHT);

    std::vector<GAMMAS_SCATT> tmpGf2 = apply_gamma5_scatt_gamma( this->GList[1], LEFT);
    double norm1=Phi_0.norm();
    double norm2=Phi_1.norm();
    pipi_aux.PhiPhi( Phi_0, tmpGf2, Phi_1); //T x N_moms x n_gammas_f2
    Float g[2];
    g[0]=-1;//eq 13
    g[1]=0;

    if(i_pi2==-1){
      for( int im=0; im<N_moms; ++im)
        for( int t=0; t<TIME; ++t)
          for( int gf2=0; gf2<n_gammas_f2; ++gf2){
              x_pe_cy( this->Corr(t,im,gi2,gf2), g, pipi_aux.Corr(t,im,gf2), 1);
	  }
    }
    else{
      for( int t=0; t<TIME; ++t)
        for( int gf2=0; gf2<n_gammas_f2; ++gf2){
          x_pe_cy( this->Corr(t,i_pi2,gi2,gf2), g, pipi_aux.Corr(t,0,gf2), 1);
	}
    }
  }//loop over G_i2 matrix

}



//here pi2 is looped outside in the building of the stocastic propagator. NB for moms_red I expect that pi2 is the same! 
//Phi_0[r] is the stochastic vector at zero momentum
//Phi_1[r] is the stochastic vector at zero momentum
//GList only 1 gamma G_i
//if i_pi2 == -1 we compute the loop for all momenta in this->pList()
//if i_pi2 != -1 we compute the loop only for the i_pi2 momentum in this->pList() 
//Note that the arguments are pointers to PLEGMA_Vectors on the host, they
//have to be loaded to the device to start the contractions
//Note that here we explicitely assume you want to compute pi0 loops <uu>-<dd> with gamma5 insertion
//by setting the real part of the loop to zero explicitely and multiply the imaginary part by 2.
//it is assumed therefore that we call it with the propagator as the first argument and with the
//stochastic source as the second one.
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::Loop_diagrams( PLEGMA_Vector<Float>* &Phi_0, PLEGMA_Vector<Float>* &Phi_1, int i_pi2, bool dn, bool accum){

  assert(i_pi2<this->pList().size());
  
  std::vector<std::vector<int>> momlist(this->pList().pi(0));
  if(i_pi2!=-1)
    momlist = std::vector<std::vector<int>>(1,this->pList().pi(0)[i_pi2]);
 
  if(i_pi2==-1)
    this->clear_output(!accum);
  else    
    this->clear_output(!accum,1,i_pi2); 

  //mom_pi2 can be 1 mom or a list of moms
  
  //aux PLEGMA_SC for PhixGxPhi multiplications
  site source=site({0,0,0,0});
  PLEGMA_ScattCorrelator<Float> pipi_aux(source, momlist, this->getTotalT());

  int N_moms = pipi_aux.Nmoms();
  assert( N_moms==1 || i_pi2==-1 );
  int n_gammas_f2 = this->GList[0].size();
  int TIME = this->localT();

  //loop over G_i2
  PLEGMA_Vector<Float> phi0;
  PLEGMA_Vector<Float> phi1;
  phi0.copy(*Phi_0,HOST);
  phi0.load();
  if (dn){
   phi0.apply_gamma5();
  }
  phi1.copy(*Phi_1,HOST);
  phi1.load();
  if (dn){
   phi1.apply_gamma5();
  }
      
  //PhixGf2xPhi
  pipi_aux.PhiPhi( phi0, this->GList[0], phi1); //T x N_moms x n_gammas_f2

  Float factor=-2.;
  if(i_pi2==-1){
    for( int im=0; im<N_moms; ++im){
      for( int t=0; t<TIME; ++t){
        for( int gf2=0; gf2<n_gammas_f2; ++gf2){
          x_pe_sy( this->Corr(t,im,gf2), factor, pipi_aux.Corr(t,im,gf2), 1); //the sign minus -1 is coming from the fermion loop
          this->Corr(t,im,gf2)[0]=0;
        }
      }
    }
  }
  else{
    for( int t=0; t<TIME; ++t){
      for( int gf2=0; gf2<n_gammas_f2; ++gf2){
        x_pe_sy( this->Corr(t,i_pi2,gf2), factor,  pipi_aux.Corr(t,i_pi2,gf2), 1);//the sign minus -1 is coming from the fermion loop
        this->Corr(t,i_pi2,gf2)[0]=0;
      }
    }
  }	
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::M_diagrams( PLEGMA_ScattCorrelator<Float> &CorrNucleon, PLEGMA_ScattCorrelator<Float> &CorrPion, Float *CorrNucleonTimeSlice, bool accum){

  //extract moms
  assert(this->pList().check_eq(0));

  //extract vector p_f1
  std::vector<std::vector<int>> moms_pf1_red = this->pList().uniq_p(1); //list of pf1 momenta needed here
  std::vector<std::vector<int>> moms_pf1 = CorrNucleon.pList().pi(0); //list of pf1 in Nucleons PLEGMA_SC
  std::vector<int> i_pf1s = CorrNucleon.pList().u_posix( 0, moms_pf1_red ); //list of positions of moms_pf1_red momenta in moms_pf1 array
  auto map = this->pList().index_map();
  //++++++++++ NN x PIPI ++++++++++++

  //put output to zero
  this->clear_output(!accum);

  int n_extgammas_i = CorrNucleon.GList[0].size();
  int n_extgammas_f = CorrNucleon.GList[1].size();
  int n_gammas_i1 = CorrNucleon.GList[2].size();
  int n_gammas_i2 = CorrPion.GList[0].size();
  int n_gammas_f1 = CorrNucleon.GList[3].size();
  int n_gammas_c  = CorrPion.GList[2].size();

  int n_gammas_f2 = CorrPion.GList[1].size();
  int TIME = this->localT();
  int GEIGEFGIF=n_extgammas_i*n_extgammas_f*n_gammas_i1*n_gammas_f1;
  int GEFGIGF=n_extgammas_f*n_gammas_i1*n_gammas_f1;
  int GIGF=n_gammas_i1*n_gammas_f1;

  //for each momentum in moms_red
  #pragma omp parallel for
  for( int i_mom=0; i_mom < this->pList().size(); ++i_mom){
    int i_pf1 = i_pf1s[map[i_mom][1]]; //position of pf1 in moms_pf1 (tempNN) 
    for( int t=0; t<TIME; ++t){
      for( int gei=0; gei<n_extgammas_i; ++gei ){
        for( int gef=0; gef<n_extgammas_f; ++gef ){
          for( int gi1=0; gi1<n_gammas_i1; ++gi1 ){
            for( int gi2=0; gi2<n_gammas_i2; ++gi2 ){
              for( int gf1=0; gf1<n_gammas_f1; ++gf1 ){
                for( int gf2=0; gf2<n_gammas_f2; ++gf2){
                  //Float *pion_pointer=pipi_aux.Corr(t,i_pf2,gi2,gf2);
                  //Float pion_contribution[2];
                  //pion_contribution[0]=-1.* pion_pointer[0];
                  //pion_contribution[1]=-1.* pion_pointer[1];
                  //x_pe_cy( this->Corr(t,i_mom,gei,gef,gi1,gi2,gf1,gf2), pion_contribution,
                  //       CorrNucleon.Corr(t,i_pf1,gei,gef,gi1,gf1), N_SPINS*N_SPINS);
                  for (int gc=0; gc< n_gammas_c; ++gc){
	            int index=i_pf1*GEIGEFGIF+gei*GEFGIGF+gef*GIGF+gi1*n_gammas_f1+gf1;
                    x_pe_cy( this->Corr(t,i_mom,gei,gef,gi1,gi2,gf1,gf2,gc),
                             CorrPion.Corr(t,i_mom,gi2,gf2,gc),
                             &CorrNucleonTimeSlice[index],
                             N_SPINS*N_SPINS);
                  } //gc
                }//G_f2
              }//G_f1
            }//G_i2
          }//G_i1
        }//G_ext_f
      }//G_ext_i
    }//time
  }//mom
}



template<typename Float>
void PLEGMA_ScattCorrelator<Float>::M_diagrams( PLEGMA_ScattCorrelator<Float> &CorrNucleon, PLEGMA_Vector<Float> &Phi_0, PLEGMA_Vector<Float> &Phi_1, bool accum){

  //extract moms
  assert(this->pList().check_eq(0));

  std::vector<int> mom_pi2 = this->pList().pi(0)[0];
  //This is working only when the unique list of pc and pf1 are the same!!!!!!!!!! 
  //They taking the same set of momenta
  std::vector<std::vector<int>> moms_pf2 = this->pList().uniq_p(2);
  //extract vector p_f1
  std::vector<std::vector<int>> moms_pf1_red = this->pList().uniq_p(1); //list of pf1 momenta needed here
  std::vector<std::vector<int>> moms_pf1 = CorrNucleon.pList().pi(0); //list of pf1 in Nucleons PLEGMA_SC
  std::vector<int> i_pf1s = CorrNucleon.pList().u_posix( 0, moms_pf1_red ); //list of positions of moms_pf1_red momenta in moms_pf1 array
  auto map = this->pList().index_map();
  auto map_minus = this->pList().index_map_minus();

  //++++++++ PION-PION +++++++++

  double normcheck;
  normcheck=Phi_0.norm();
  PLEGMA_printf("Phi0 norm %e\n", normcheck);
  normcheck=Phi_1.norm();
  PLEGMA_printf("Phi1 norm %e\n", normcheck);

  //aux PLEGMA_SC for PhixGxPhi multiplications
  momList auxmlist(1, {moms_pf2,}, {0,});
  PLEGMA_ScattCorrelator pipi_aux(this->getSource(), auxmlist);

  pipi_aux.initialize_diagram( this->GList[3], this->GList[5], "P"); //false m is pi2, true is pf2

  pipi_aux.P_diagrams( Phi_0, Phi_1, -1, false); // pf2, t, 1, gi2, gf2 //-1 from eq. (13) is inside P_diagram


  //++++++++++ NN x PIPI ++++++++++++

  //put output to zero
  this->clear_output(!accum);

  int n_extgammas_i = CorrNucleon.GList[0].size();
  int n_extgammas_f = CorrNucleon.GList[1].size();
  int n_gammas_i1 = CorrNucleon.GList[2].size();
  int n_gammas_i2 = pipi_aux.GList[0].size();
  int n_gammas_f1 = CorrNucleon.GList[3].size();
  int n_gammas_c ;
  if (CorrNucleon.GList.size() >4){ 
    n_gammas_c = CorrNucleon.GList[4].size();
  }
  int n_gammas_f2 = pipi_aux.GList[1].size();
  int TIME = this->localT();
  Float *sinktimeslice;
  int globalSinkTimeSlice;
  if (CorrNucleon.GList.size() >4){
    globalSinkTimeSlice=(this->source[3]+this->getTotalT()-1)%HGC_totalL[3];
    sinktimeslice=pipi_aux.get_time_slice(globalSinkTimeSlice);
    //pipi_aux.writeHDF5("testcase");
    //PLEGMA_printf("zero component %e %d %d total %d \n",sinktimeslice[0], globalSinkTimeSlice, this->localT(), this->getTotalT());
  }


  //for each momentum in moms_red
  #pragma omp parallel for
  for( int i_mom=0; i_mom < this->pList().size(); ++i_mom){
    int i_pf1 = i_pf1s[map[i_mom][1]]; //position of pf1 in moms_pf1 (tempNN)
    int i_pf2 = map[i_mom][2]; //position of pf2 in pionpion
    if (CorrNucleon.GList.size() >4){//ensure always zero momentum at the sink
      i_pf2=map_minus[i_mom][1];
    }
    for( int t=0; t<TIME; ++t){
      for( int gei=0; gei<n_extgammas_i; ++gei ){
        for( int gef=0; gef<n_extgammas_f; ++gef ){
          for( int gi1=0; gi1<n_gammas_i1; ++gi1 ){
            for( int gi2=0; gi2<n_gammas_i2; ++gi2 ){
              for( int gf1=0; gf1<n_gammas_f1; ++gf1 ){
                for( int gf2=0; gf2<n_gammas_f2; ++gf2){
                  //Float *pion_pointer=pipi_aux.Corr(t,i_pf2,gi2,gf2);
                  //Float pion_contribution[2];
                  //pion_contribution[0]=-1.* pion_pointer[0];
                  //pion_contribution[1]=-1.* pion_pointer[1];
                  //x_pe_cy( this->Corr(t,i_mom,gei,gef,gi1,gi2,gf1,gf2), pion_contribution,
                  //       CorrNucleon.Corr(t,i_pf1,gei,gef,gi1,gf1), N_SPINS*N_SPINS);
	          if (CorrNucleon.GList.size() >4){
	            for (int gc=0; gc< n_gammas_c; ++gc){
                      x_pe_cy( this->Corr(t,i_mom,gei,gef,gi1,gi2,gf1,gf2,gc),
                               &sinktimeslice[i_pf2*n_gammas_i2*n_gammas_f2*2+gi2*n_gammas_f2*2+gf2*2],
                               CorrNucleon.Corr(t,i_mom,gei,gef,gi1,gf1,gc),
                               N_SPINS*N_SPINS);
		    } //gc

		  }
		  else{
                    x_pe_cy( this->Corr(t,i_mom,gei,gef,gi1,gi2,gf1,gf2),
                             pipi_aux.Corr(t,i_pf2,gi2,gf2),
                             CorrNucleon.Corr(t,i_pf1,gei,gef,gi1,gf1),
                             N_SPINS*N_SPINS);
		  }
		  
                }//G_f2
              }//G_f1
            }//G_i2
          }//G_i1
        }//G_ext_f
      }//G_ext_i
    }//time
  }//mom
  if  (CorrNucleon.GList.size() >4){
    free(sinktimeslice);
  }
}

//here pi2 is looped outside in the building of the stocastic propagator. NB for moms_red I expect that pi2 is the same! Phi_0[s] is the stocastic propagator at zero momentum and spin s, Phi_1 with momentum pi2
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::M_diagrams( PLEGMA_ScattCorrelator<Float> &CorrNucleon, std::vector<PLEGMA_Vector<Float>*> &Phi_0, std::vector<PLEGMA_Vector<Float>*> &Phi_1, bool accum){

  //extract moms
  assert(this->pList().check_eq(0));
  
  std::vector<int> mom_pi2 = this->pList().pi(0)[0]; 
  std::vector<std::vector<int>> moms_pf2 = this->pList().uniq_p(2);
  //extract vector p_f1
  std::vector<std::vector<int>> moms_pf1_red = this->pList().uniq_p(1); //list of pf1 momenta needed here
  std::vector<std::vector<int>> moms_pf1 = CorrNucleon.pList().pi(0); //list of pf1 in Nucleons PLEGMA_SC
  std::vector<int> i_pf1s = CorrNucleon.pList().u_posix( 0, moms_pf1_red ); //list of positions of moms_pf1_red momenta in moms_pf1 array
  auto map = this->pList().index_map();


  //++++++++ PION-PION +++++++++

  //aux PLEGMA_SC for PhixGxPhi multiplications
  momList auxmlist(1, {moms_pf2,}, {0,});
  PLEGMA_ScattCorrelator pipi_aux(this->getSource(), auxmlist, this->getTotalT());

  pipi_aux.initialize_diagram( this->GList[3], this->GList[5], "P"); //false m is pi2, true is pf2

  pipi_aux.P_diagrams( Phi_0, Phi_1, -1, false); // pf2, t, 1, gi2, gf2 //-1 from eq. (13) is inside P_diagram

  
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
  #pragma omp parallel for
  for( int i_mom=0; i_mom < this->pList().size(); ++i_mom){
    int i_pf1 = i_pf1s[map[i_mom][1]]; //position of pf1 in moms_pf1 (tempNN)
    int i_pf2 = map[i_mom][2]; //position of pf2 in pionpion
    for( int t=0; t<TIME; ++t){
      for( int gei=0; gei<n_extgammas_i; ++gei ){ 
	for( int gef=0; gef<n_extgammas_f; ++gef ){
	  for( int gi1=0; gi1<n_gammas_i1; ++gi1 ){
	    for( int gi2=0; gi2<n_gammas_i2; ++gi2 ){
	      for( int gf1=0; gf1<n_gammas_f1; ++gf1 ){
		for( int gf2=0; gf2<n_gammas_f2; ++gf2){
                  //Float *pion_pointer=pipi_aux.Corr(t,i_pf2,gi2,gf2);
                  //Float pion_contribution[2];
                  //pion_contribution[0]=-1.* pion_pointer[0];
                  //pion_contribution[1]=-1.* pion_pointer[1];
		  //x_pe_cy( this->Corr(t,i_mom,gei,gef,gi1,gi2,gf1,gf2), pion_contribution,
		  //	   CorrNucleon.Corr(t,i_pf1,gei,gef,gi1,gf1), N_SPINS*N_SPINS);
		  x_pe_cy( this->Corr(t,i_mom,gei,gef,gi1,gi2,gf1,gf2),
			   pipi_aux.Corr(t,i_pf2,gi2,gf2),
		  	   CorrNucleon.Corr(t,i_pf1,gei,gef,gi1,gf1),
			   N_SPINS*N_SPINS);
		}//G_f2
	      }//G_f1
	    }//G_i2
	  }//G_i1
	}//G_ext_f
      }//G_ext_i
    }//time
  }//mom
}


//T diagramm pion nucleon at the sink
//V3 should have momentum list p_f2
//V2 should have momentum list p_f1 
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::T_diagrams_piNsink( PLEGMA_ScattCorrelator<Float> &srcV3, PLEGMA_ScattCorrelator<Float> &srcV2, int diagram_index, bool accum){


  //checks between srcV2 srcV3
  if(!srcV2.check_reduction(V_2)) PLEGMA_error("srcV2 seems not to have V2like shape\n");
  if(!srcV3.check_reduction(V_3)) PLEGMA_error("srcV3 seems not to have V3like shape\n");

  this->clear_output(!accum);

  Float factor[2]={0.,0.};
  switch (diagram_index) {
    case 1:
      factor[0]=-2.;// -1 from eqs. (51-56)...
      this->V3V2reduction_matrix( srcV3, srcV2, 1,  false, 0, false, factor, true );
      break;
    case 3:
      factor[0]=-2.;// -1 from eqs. (51-56)...
      this->V3V2reduction( srcV3, srcV2, 2, true, 0, false, factor, true );
      break;
    case 5:
      factor[0]=-2.;// -1 from eqs. (51-56)...
      this->V3V2reduction( srcV3, srcV2, 0, false, 0, false, factor, true );
      break;
    case 7:
      factor[0]=-1.;// -1 from eqs. (51-56)...
      this->V3V2reduction( srcV3, srcV2, 1, false, 0, true, factor, false);
      break;
    case 8:
      factor[0]=-1.;// -1 from eqs. (51-56)...
      this->V3V2reduction_matrix( srcV3, srcV2, 0, false, 0, true, factor, false);
      break;
    case 9:
      factor[0]=-1.;// -1 from eqs. (51-56)...
      this->V3V2reduction( srcV3, srcV2, 0, false, 0, true, factor, false);
      break;
    case 10:
      factor[0]=-1.;// -1 from eqs. (51-56)...
      this->V3V2reduction_matrix( srcV3, srcV2, 1, false, 0, true, factor, false);
      break;
    case 11:
      factor[0]=-1.;// -1 from eqs. (51-56)...
      this->V3V2reduction( srcV3, srcV2, 0, false, 0, false, factor, true);
      break;
    case 12:
      factor[0]=-1.;// -1 from eqs. (51-56)...
      this->V3V2reduction( srcV3, srcV2, 2, true, 0, false, factor, true);
      break;
    case 13:
      factor[0]=-2.;// -1 from eqs. (51-56)...
      this->V3V2reduction( srcV3, srcV2, 1, false, 0, false, factor, false);
      break;
    case 15:
      factor[0]=-2.;// -1 from eqs. (51-56)...
      this->V3V2reduction( srcV3, srcV2, 2, true, 0, true, factor, false);
      break;
    case 17:
      factor[0]=-2.;// -1 from eqs. (51-56)...
      this->V3V2reduction_matrix( srcV3, srcV2, 1, false, 0, false, factor, true);
      break;
    case 19:
      factor[0]=-1.;// -1 from eqs. (51-56)...
      this->V3V2reduction( srcV3, srcV2, 2, false, 0, true, factor, false);
      break;
    case 20:
      factor[0]=-1.;// -1 from eqs. (51-56)...
      this->V3V2reduction_matrix( srcV3, srcV2, 0, false, 0, false, factor, false);
      break;
    case 21:
      factor[0]=-1.;// -1 from eqs. (51-56)...
      this->V3V2reduction( srcV3, srcV2, 2, true, 0, true, factor, false);
      break;
    case 22:
      factor[0]=-1.;// -1 from eqs. (51-56)...
      this->V3V2reduction_matrix( srcV3, srcV2, 1, false, 0, false, factor, false);
      break;
    case 23:
      factor[0]=-2.;// -1 from eqs. (51-56)...
      this->V3V2reduction( srcV3, srcV2, 1, false, 0, false, factor, true);
      break;
    case 25:
      factor[0]=-2.;// -1 from eqs. (51-56)...
      this->V3V2reduction( srcV3, srcV2, 0, false, 0, false, factor, true);
      break;
    default:
      PLEGMA_error("This value of T-piNsink diagram index does not exists, please check your inputs in piNdiagrams.cpp");
  }
}
//T diagramm pion nucleon at the sink
//V3 should have momentum list p_f2
//V2 should have momentum list p_f1
/*
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::T_diagrams_piNsink( PLEGMA_ScattCorrelator<Float> &srcV3, PLEGMA_ScattCorrelator<Float> &srcV2, bool accum){


  //checks between srcV2 srcV3
  if(!srcV2.check_reduction(V_2)) PLEGMA_error("srcV2 seems not to have V2like shape\n");
  if(!srcV3.check_reduction(V_3)) PLEGMA_error("srcV3 seems not to have V3like shape\n");

  this->clear_output(!accum);

  if( gammas_isSym( this->GList[2] ) ){ //Symm G_i1
    Float factor[2] = {-2.,0.};// -1 from eqs. (51-56)
    this->V3V2reduction_matrix( srcV3, srcV2, 1,  false, 0, false, factor, true ); //last true because of transp(G_f1)
    
    this->V3V2reduction( srcV3, srcV2, 2, true, 0, false, factor, true ); //last true because of transp(G_f1)
    
    this->V3V2reduction( srcV3, srcV2, 0, false, 0, false, factor, true ); //last true because of transp(G_f1)
  }
  else{
    Float factor[2] = {-1.,0.};// -1 from eqs. (51-56)
    this->V3V2reduction_matrix( srcV3, srcV2, 1,  false, 0, false, factor, true ); //last true because of transp(G_f1)
    this->V3V2reduction_matrix( srcV3, srcV2, 1,  false, 0, true, factor, true );  //last true because of transp(G_f1)
    
    this->V3V2reduction( srcV3, srcV2, 2, true, 0, false, factor, true ); //last true because of transp(G_f1)
    this->V3V2reduction( srcV3, srcV2, 2, true, 0, true, factor, true );  //last true because of transp(G_f1)

    this->V3V2reduction( srcV3, srcV2, 0, false, 0, false, factor, true ); //last true because of transp(G_f1)
    this->V3V2reduction( srcV3, srcV2, 0, false, 0, true, factor, true  ); //last true because of transp(G_f1)
  }

}
*/
//LT diagrams, Loop at the sink multiplied by T diagramm at the
//T is build up from a T1 and a T2 reduction
//source
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::LT_diagrams( PLEGMA_ScattCorrelator<Float> &T1, PLEGMA_ScattCorrelator<Float> &T2, PLEGMA_ScattCorrelator<Float> &Loop, bool accum){

  //checks between T1 T2
  if(!T1.check_reduction(T_1)) PLEGMA_error("srcT1 seems not to have T1like shape\n");
  if(!T2.check_reduction(T_2)) PLEGMA_error("srcT2 seems not to have T2like shape\n");

  if( T1.getMomList()!=T2.getMomList() || T1.getMomList()!=this->pList().pi(1) )
    PLEGMA_error("T1,T2 have not the the same mom list of N\n");
  assert( Nmoms()==this->pList().pi(0).size() );

  for(int i=0; i<2; ++i)
    if((T1.GList[i]!=T2.GList[i])||(T1.GList[i]!=this->GList[2*i+2]))
      PLEGMA_error("T1,T2 wrong gamma list\n");

  //size of final output for NN
  int n_extgammas_i = this->GList[0].size();
  int n_extgammas_f = this->GList[1].size();
  int n_gammas_i1 = this->GList[2].size();
  int n_gammas_i2 = this->GList[3].size();
  int n_gammas_f1 = this->GList[4].size();
  int n_gammas_f2 = this->GList[5].size();
  int TIME = this->localT();


  //put output to zero
  this->clear_output(!accum);
  auto imap = this->pList().index_map();

  #pragma omp parallel for
  for(int i_m=0; i_m<imap.size(); i_m++){
    int i_mom_i2 = imap[i_m][0];
    int i_mom_f1 = imap[i_m][1];
    int i_mom_f2 = imap[i_m][2];
//    #pragma omp critical
//    {
//      PLEGMA_printf("IMOM thread: %d im %d i2 %d f1 %d f2 %d\n", omp_get_thread_num(), i_m, i_mom_i2, i_mom_f1, i_mom_f2 );
//    }
    for( int t=0; t<TIME; ++t){
      for (int gf2=0; gf2<n_gammas_f2; ++gf2){
        Float temp[N_SPINS*N_SPINS*2];
        for( int gi1=0; gi1<n_gammas_i1; ++gi1 ){
          for( int gf1=0; gf1<n_gammas_f1; ++gf1 ){
            Float *loop_pointer=Loop.Corr(t,i_mom_f2,gf2);
            Float loop_contribution[2];
            loop_contribution[0]= loop_pointer[0];
            loop_contribution[1]= loop_pointer[1];

            for(int spin=0; spin<N_SPINS*N_SPINS*2; ++spin)
              temp[spin] = (T1.Corr(t,i_mom_f1,gi1,gf1)[spin] + T2.Corr(t,i_mom_f1,gi1,gf1)[spin]);
            for(int spin=0; spin<N_SPINS*N_SPINS; ++spin){
              Float realpart,imagpart;
              realpart=temp[2*spin]*loop_contribution[0]-temp[2*spin+1]*loop_contribution[1];
              imagpart=temp[2*spin]*loop_contribution[1]+temp[2*spin+1]*loop_contribution[0];
              temp[2*spin+0]=realpart;
              temp[2*spin+1]=imagpart;
            }

            for( int gei=0; gei<n_extgammas_i; ++gei ){
              for( int gef=0; gef<n_extgammas_f; ++gef ){
                GAMMAS_SCATT extG_i1 = this->GList[0][gei];
                GAMMAS_SCATT extG_f1 = this->GList[1][gef];
                //change in pe_GNG
                M_pe_GNG<Float>( this->Corr(t,i_m,gei,gef,gi1,0,gf1,0), extG_f1, extG_i1, temp );
              } //Gextf
            } //G_exti
          } //G_f1
        } //G_i1
      } //G_gf2
    } //mom
  } //T

}

//Nucleon correlator. This function should be called outside the p_i2 loop, with Ts computed using the entire list of unique p_f1s. N.B: we multiply the output by exp(i * x_sourcepos * p_f1);
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::N_diagrams( PLEGMA_ScattCorrelator<Float> &T1, PLEGMA_ScattCorrelator<Float> &T2, bool accum){

  //checks between T1 T2
  if(!T1.check_reduction(T_1)) PLEGMA_error("srcT1 seems not to have T1like shape\n");
  if(!T2.check_reduction(T_2)) PLEGMA_error("srcT2 seems not to have T1like shape\n");

  if( T1.getMomList()!=T2.getMomList() || T1.getMomList()!=this->pList().pi(0) )
    PLEGMA_error("T1,T2 have not the the same mom list of N\n");
  assert( Nmoms()==this->pList().pi(0).size() );
  
  for(int i=0; i<2; ++i)
    if((T1.GList[i]!=T2.GList[i])||(T1.GList[i]!=this->GList[i+2]))
      PLEGMA_error("T1,T2 wrong gamma list\n");
        
  //size of final output for NN
  int n_extgammas_i = this->GList[0].size();
  int n_extgammas_f = this->GList[1].size();
  int n_gammas_i1 = this->GList[2].size();
  int n_gammas_f1 = this->GList[3].size();
  int TIME = this->localT();  

  
  //put output to zero
  this->clear_output(!accum);
  for( int t=0; t<TIME; ++t){
    #pragma omp parallel for
    for( int i_mom=0; i_mom<this->Nmoms(); ++i_mom){
      Float temp[N_SPINS*N_SPINS*2];
      for( int gi1=0; gi1<n_gammas_i1; ++gi1 ){
        for( int gf1=0; gf1<n_gammas_f1; ++gf1 ){
	  int coeffT = gammaTranspSign_scatt[this->GList[3][gf1]]*gammaTranspSign_scatt[this->GList[2][gi1]];
	  for(int spin=0; spin<N_SPINS*N_SPINS*2; ++spin)
	    temp[spin] = coeffT*(T1.Corr(t,i_mom,gi1,gf1)[spin] + T2.Corr(t,i_mom,gi1,gf1)[spin]);

          for( int gei=0; gei<n_extgammas_i; ++gei ){ 
	    for( int gef=0; gef<n_extgammas_f; ++gef ){
	      GAMMAS_SCATT extG_i1 = this->GList[0][gei];
	      GAMMAS_SCATT extG_f1 = this->GList[1][gef];
	      //change in pe_GNG
	      M_pe_GNG<Float>( this->Corr(t,i_mom,gei,gef,gi1,gf1), extG_f1, extG_i1, temp );
	    } //G_extf
	  } //G_exti
	} //G_f1
      } //G_i1
    } //mom
  } //T

}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::T_diagrams_oet(PLEGMA_ScattCorrelator<Float> &reductionsVT, std::shared_ptr<Float> &Phi0,  int diagramindex, bool accum){

  //put output to zero
  this->clear_output(!accum);
    this->clear_output(!accum);
  Float factor[2]={1,0};

  switch( diagramindex ){
    case 1:
      this->V24pointSourceReduction(reductionsVT, Phi0, 1,  false, false, false, factor);
      break;
    case 2:
      this->V24pointSourceReduction(reductionsVT, Phi0, 2,  true, false, false, factor);
      break;
    case 3:
      this->V24pointSourceReduction(reductionsVT, Phi0, 0,  false, false,  false, factor);
      break;
//    case 7:
//      this->V24pointSourceReduction_matrix(reductionsVT, Phi0, 1,  false, true, factor);
//      break;
//    case 9:
//      this->V24pointSourceReduction(reductionsVT, Phi0, 0,  false, true, factor);
//      break;
    case 11:
      this->V24pointSourceReduction(reductionsVT, Phi0, 1,  false, true, true, factor);
      break;
    case 12:
      this->V24pointSourceReduction_matrix(reductionsVT, Phi0, 0,  false, true, true, factor);
      break;
    case 13:
      this->V24pointSourceReduction(reductionsVT, Phi0, 2,  true, true, false, factor);
      break;
    case 14:
      this->V24pointSourceReduction_matrix(reductionsVT, Phi0, 1,  false, false, false, factor);
      break;
//  case 15:
//    this->V24pointSourceReduction(reductionsVT, Phi0, 2,  true, true, factor);
//    break;
//  case 17:
//    this->V24pointSourceReduction_matrix(reductionsVT, Phi0, 1,  false, false, factor);
//    break;
//  case 19:
//    this->V24pointSourceReduction(reductionsVT, Phi0, 1, false, false, factor);
//    break;
    case 21:
      this->V24pointSourceReduction(reductionsVT, Phi0, 1, false, true, false, factor);
      break;
    case 22:
      this->V24pointSourceReduction_matrix(reductionsVT, Phi0, 0, false, true, false, factor);
      break;
    case 23:
      this->V24pointSourceReduction_matrix(reductionsVT, Phi0, 0, false, true, false, factor);
      break;
    case 24:
      this->V24pointSourceReduction(reductionsVT, Phi0, 0, false, true, false, factor);
      break;
    case 25:
      this->V24pointSourceReduction(reductionsVT, Phi0, 2, true, false, true, factor);
      break;
    case 26:
      this->V24pointSourceReduction(reductionsVT, Phi0, 0, false, false, true, factor);
      break;
    default:
      PLEGMA_error("This value of T-piN oet diagram index does not exists, please check your inputs in piNdiagrams_oet.cpp");
  }
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::V24pointSourceReduction_matrix( PLEGMA_ScattCorrelator<Float> &reductionsVT,  std::shared_ptr<Float> &Phi0, int index_abs, bool transp, bool transpgamma_i1, bool transpgamma_f1, Float* factor) {
    
  if( this->pList().pi(1) != reductionsVT.getMomList() ) PLEGMA_error("T1 has not the the same mom list of T\n");

    //n gammas
  int n_gammas_f = this->GList[5].size();
  int n_gammas_i2= this->GList[4].size();
  int n_gammas_i1 = this->GList[2].size();
  int n_extgammas_f = this->GList[1].size();
  int n_extgammas_i = this->GList[0].size();
  int TIME = this->localT();

  #pragma omp parallel for
  for(int i_mom=0; i_mom<this->Nmoms(); ++i_mom){
    Float temp_colorvector[N_COLS*2];
    Float temp[N_SPINS*N_SPINS*2];
    Float V3aux[N_SPINS*N_SPINS*N_COLS*2];
    Float stochAux[N_SPINS*N_COLS*2];
    GAMMAS_SCATT gamma5 = G_5;
    for(int t=0; t<TIME; ++t){
      for (int gi2=0 ; gi2< n_gammas_i2; ++gi2){
        GAMMAS_SCATT gamma5_t_gammai2 = apply_g5( this->GList[4][gi2], LEFT);
        V_MVM<Float>( Phi0.get(), gamma5, gamma5_t_gammai2, stochAux );
        for (int i=0; i<N_SPINS*N_COLS;++i){//Taking the complex conjugate
          stochAux[2*i+1]=-1*stochAux[2*i+1];
        }
        for( int gi1=0; gi1<n_gammas_i1; ++gi1 ){
          for( int gf=0; gf<n_gammas_f; ++gf ){
            for ( int alpha=0;alpha<N_SPINS;++alpha){

              switch(index_abs){
                case 0: absorbspinmatrix_fromV24<0,Float>( V3aux, reductionsVT.Corr(t,i_mom,gf), alpha ); break;
                case 1: absorbspinmatrix_fromV24<1,Float>( V3aux, reductionsVT.Corr(t,i_mom,gf), alpha ); break;
                case 2: absorbspinmatrix_fromV24<2,Float>( V3aux, reductionsVT.Corr(t,i_mom,gf), alpha ); break;
              }

              if(transpgamma_f1){
                for(int sc=0; sc<N_SPINS*N_COLS*2; ++sc)
                  V3aux[sc] *= gammaTranspSign_scatt[this->GList[5][gf]];
              }

              //color vector from Tr[G_i1 V2]
              V_TR_MM<Float>( V3aux, this->GList[2][gi1], transpgamma_i1, temp_colorvector);
              for ( int beta=0; beta< N_SPINS; ++beta){
                int spins = transp ? (beta*N_SPINS+alpha)*2 : (alpha*N_SPINS+beta)*2;
		temp[spins]=0.;
                temp[spins+1]=0.;

                //colorvector x V3
                for (int coloridx=0; coloridx<3; ++coloridx){
                  temp[spins+0] +=
                    +temp_colorvector[2*coloridx+0]*stochAux[2*beta*N_COLS+2*coloridx]
                    -temp_colorvector[2*coloridx+1]*stochAux[2*beta*N_COLS+2*coloridx+1];
                  temp[spins+1] +=
                    +temp_colorvector[2*coloridx+1]*stochAux[2*beta*N_COLS+2*coloridx]
                    +temp_colorvector[2*coloridx+0]*stochAux[2*beta*N_COLS+2*coloridx+1];
		}//coloridx
              }
            }

            for( int gei=0; gei<n_extgammas_i; ++gei ){
              for( int gef=0; gef<n_extgammas_f; ++gef ){
                GAMMAS_SCATT eGamma_i = this->GList[0][gei];
                GAMMAS_SCATT eGamma_f = this->GList[1][gef];
                M_pe_GNG<Float>( this->Corr(t,i_mom,gei,gef,gi1,0,gi2,gf), eGamma_f, eGamma_i, temp);

              }//G_ext_f
            }//G_ext_i
          }//G_f
        }//G_i
      }//G_i2
    }//time
  }//mom
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::V24pointSourceReduction( PLEGMA_ScattCorrelator<Float> &reductionsVT, std::shared_ptr<Float> &Phi0, int index_abs, bool transp, bool transpgamma_i1, bool transpgamma_f1, Float* factor) {

  if( this->pList().pi(1) != reductionsVT.getMomList() ) PLEGMA_error("T1 has not the the same mom list of T\n");

    //n gammas
  int n_gammas_f = this->GList[5].size();
  int n_gammas_i2= this->GList[4].size();
  int n_gammas_i1 = this->GList[2].size();
  int n_extgammas_f = this->GList[1].size();
  int n_extgammas_i = this->GList[0].size();
  int TIME = this->localT();	

  for(int t=0; t<TIME; ++t){
    #pragma omp parallel for
    for(int i_mom=0; i_mom<this->Nmoms(); ++i_mom){

      Float temp[N_SPINS*N_SPINS*2];
      Float V3aux[N_SPINS*N_COLS*2];
      Float stochAux[N_SPINS*N_COLS*2];
      GAMMAS_SCATT gamma5 = G_5;
      for (int gi2=0 ; gi2< n_gammas_i2; ++gi2){
        GAMMAS_SCATT gamma5_t_gammai2 = apply_g5( this->GList[4][gi2], LEFT);
        V_MVM<Float>( Phi0.get(), gamma5, gamma5_t_gammai2, stochAux );
	for (int i=0; i<N_SPINS*N_COLS;++i){
	  stochAux[2*i+1]=-1*stochAux[2*i+1];
	}
        for( int gi1=0; gi1<n_gammas_i1; ++gi1 ){
          for( int gf=0; gf<n_gammas_f; ++gf ){
	    for ( int alpha=0;alpha<N_SPINS;++alpha){
	      for ( int beta=0; beta< N_SPINS; ++beta){
	        int spins = transp ? (beta*N_SPINS+alpha)*2 : (alpha*N_SPINS+beta)*2;
	        switch(index_abs){
                  case 1: absorb_fromV24<1,Float>( V3aux, reductionsVT.Corr(t,i_mom,gf), alpha, beta ); break;
                  case 2: absorb_fromV24<2,Float>( V3aux, reductionsVT.Corr(t,i_mom,gf), alpha, beta); break;
                  case 0: absorb_fromV24<0,Float>( V3aux, reductionsVT.Corr(t,i_mom,gf), alpha, beta ); break;
	        }

		if(transpgamma_f1){
                  for(int sc=0; sc<N_SPINS*N_COLS*2; ++sc)
                    V3aux[sc] *= gammaTranspSign_scatt[this->GList[5][gf]];
                }

                V_M_V<Float>( stochAux, V3aux,
                              this->GList[2][gi1], transpgamma_i1, temp + spins);
	      }
	    }
            for( int gei=0; gei<n_extgammas_i; ++gei ){
              for( int gef=0; gef<n_extgammas_f; ++gef ){
                GAMMAS_SCATT eGamma_i = this->GList[0][gei];
                GAMMAS_SCATT eGamma_f = this->GList[1][gef];
                M_pe_GNG<Float>( this->Corr(t,i_mom,gei,gef,gi1,0,gi2,gf), eGamma_f, eGamma_i, temp);

              }//G_ext_f
            }//G_ext_i
          }//G_f
        }//G_i
      }//G_i2
    }//mom
  }//time
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::contractMesonThrp_local(PLEGMA_Vector<Float> &bwdProp,
                       PLEGMA_Vector<Float> &fwdProp,
                       std::vector<GAMMAS_SCATT> gammas){
  this->shape = {(int) gammas.size()};
  this->datasets = {"threep"};
  this->groups =  {"Local"};
  this->description = getGammasString_scatt(gammas);
  this->initialize();

  if(gammas.size() == 0) PLEGMA_error("List of gammas provided is empty");

  PhixGxPhi_k<Float,Float>(*this,fwdProp, gammas, bwdProp);

}



//here pi2 and Gamma_i2 are looped outside in the building of the sequential propagator. NB for moms I expect that pi2 is the same! The T reduction contains ptot.
//Here we assume that the nucleon interpolator is anti-symmetric and the delta interpolator is symmetric
//This is true for delta++ and I=3/2, I_3=3/2 pion-nucleon scattering
//Please take care of your signs!
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::T_diagrams( PLEGMA_ScattCorrelator<Float> &T1, PLEGMA_ScattCorrelator<Float> &T3, PLEGMA_ScattCorrelator<Float> &T5, int ig_i2, bool accum){


  //checks between T1 T3 T5
  if(!T1.check_reduction(T_1)) PLEGMA_error("srcT1 seems not to have T1like shape\n");
  if(!T3.check_reduction(T_1)) PLEGMA_error("srcT3 seems not to have T1like shape\n");
  if(!T5.check_reduction(T_2)) PLEGMA_error("srcT5 seems not to have T2like shape\n");
      
  assert(this->pList().check_eq(0));

  if( this->pList().pi(1) != T1.getMomList() ) PLEGMA_error("T1 has not the the same mom list of T\n");
  if( T1.getMomList() != T3.getMomList() ) PLEGMA_error("T3 has not the the same mom list of T\n");
  if( T1.getMomList() != T5.getMomList() ) PLEGMA_error("T5 has not the the same mom list of T\n");
  assert(Nmoms() == T1.Nmoms());
  
  for(int i=0; i<2; ++i)
    if( (T1.GList[i]!=T3.GList[i]) || (T1.GList[i]!=T5.GList[i]) || (T1.GList[i]!=this->GList[(i+1)*2]) )
      PLEGMA_error("T1,T3,T5 and T wrong gamma list\n");
  
  //n gammas
  int n_gammas_f = this->GList[4].size();
  int n_gammas_i1 = this->GList[2].size();
  int n_extgammas_f = this->GList[1].size();
  int n_extgammas_i = this->GList[0].size();
  int TIME = this->localT();
	      
  this->clear_output(!accum, 5, ig_i2); 

  for(int t=0; t<TIME; ++t){
    #pragma omp parallel for
    for(int i_mom=0; i_mom<this->Nmoms(); ++i_mom){
      Float temp[N_SPINS*N_SPINS*2];
      for( int gi1=0; gi1<n_gammas_i1; ++gi1 ){
        for( int gf=0; gf<n_gammas_f; ++gf ){
	  for(int spin=0; spin<N_SPINS*N_SPINS*2; ++spin){
            temp[spin] = (+1.*T1.Corr(t,i_mom,gi1,gf)[spin]*(gammaTranspSign_scatt[this->GList[4][gf]]+1.) 
                          +1.*T3.Corr(t,i_mom,gi1,gf)[spin]*(gammaTranspSign_scatt[this->GList[4][gf]]+1.)*(gammaTranspSign_scatt[this->GList[2][gi1]])
                          +1.*T5.Corr(t,i_mom,gi1,gf)[spin]*(gammaTranspSign_scatt[this->GList[4][gf]]+1.));
	  }
          for( int gei=0; gei<n_extgammas_i; ++gei ){ 
	    for( int gef=0; gef<n_extgammas_f; ++gef ){
	      GAMMAS_SCATT eGamma_i = this->GList[0][gei];
	      GAMMAS_SCATT eGamma_f = this->GList[1][gef];
              M_pe_GNG<Float>( this->Corr(t,i_mom,gei,gef,gi1,ig_i2,gf), eGamma_f, eGamma_i, temp);
	      
	    }//G_ext_f
	  }//G_ext_i
	}//G_f
      }//G_i
    }//mom
  }//time
}


template<typename Float>
void PLEGMA_ScattCorrelator<Float>::D_diagrams( PLEGMA_ScattCorrelator<Float> &T1, PLEGMA_ScattCorrelator<Float> &T2, bool accum){
  //checks between T1 T2
  if(!T1.check_reduction(T_1)) PLEGMA_error("srcT1 seems not to have T1like shape\n");
  if(!T2.check_reduction(T_2)) PLEGMA_error("srcT2 seems not to have T1like shape\n");
  
  if( T1.getMomList()!=T2.getMomList() || T1.getMomList()!=this->pList().pi(0) )
    PLEGMA_error("T1,T2 have not the the same mom list of D\n");
 
  for(int i=0; i<2; ++i)
    if((T1.GList[i]!=T2.GList[i])||(T1.GList[i]!=this->GList[i+2]))
      PLEGMA_error("T1,T2 wrong gamma list\n");

  
  //extract array mom
  assert( this->Nmoms() == T1.Nmoms() );
    
  //size of final output for DD
  int n_extgammas_i = this->GList[0].size();
  int n_extgammas_f = this->GList[1].size();
  int n_gammas_i = this->GList[2].size();
  int n_gammas_f = this->GList[3].size();
  int TIME = this->localT();  
 
  //put output to zero
  this->clear_output(!accum); 

  for( int t=0; t<TIME; ++t){
    #pragma omp parallel for
    for( int i_mom=0; i_mom<this->Nmoms(); ++i_mom){
      Float temp[N_SPINS*N_SPINS*2];
      for( int gi=0; gi<n_gammas_i; ++gi ){
        for( int gf=0; gf<n_gammas_f; ++gf ){
	  int coeffT1=gammaTranspSign_scatt[this->GList[3][gf]]+1+gammaTranspSign_scatt[this->GList[3][gf]]*gammaTranspSign_scatt[this->GList[2][gi]]+gammaTranspSign_scatt[this->GList[2][gi]];
	  int coeffT2=gammaTranspSign_scatt[this->GList[3][gf]]*gammaTranspSign_scatt[this->GList[2][gi]]+gammaTranspSign_scatt[this->GList[2][gi]];
	  //TMP CHECKS
	  assert(coeffT1==4);
	  assert(coeffT2==2);
	  for(int spin=0; spin<N_SPINS*N_SPINS*2; ++spin)
            temp[spin] = coeffT1*T1.Corr(t,i_mom,gi,gf)[spin] + coeffT2*T2.Corr(t,i_mom,gi,gf)[spin];
          for( int gei=0; gei<n_extgammas_i; ++gei ){ 
	    for( int gef=0; gef<n_extgammas_f; ++gef ){
	      GAMMAS_SCATT extG_i1 = this->GList[0][gei];
	      GAMMAS_SCATT extG_f1 = this->GList[1][gef];
	      M_pe_GNG<Float>( this->Corr(t,i_mom,gei,gef,gi,gf), extG_f1, extG_i1, temp );
	    } //G_extf
	  } //G_exti
	} //G_f
      } //G_i
    } //mom
  } //T
}

//This routine converts a T reduction into a diagramm format
//In particular: adds the necessary external gamma structure
//and perform the correct ordering
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::convertTreductiontoDiagram( PLEGMA_ScattCorrelator<Float> &T2, int ig_i2, bool accum, bool transp_i1, bool transp_f1){
  //checks between T2
  if(!T2.check_reduction(T_2)) PLEGMA_error("srcT2 seems not to have T1like shape\n");

  if( T2.getMomList()!=this->pList().pi(1) )
    PLEGMA_error("T2 must have a mom list\n");

  //extract array mom
  assert( this->Nmoms() == T2.Nmoms() );

  //size of final output for DD
  int n_extgammas_i = this->GList[0].size();
  int n_extgammas_f = this->GList[1].size();
  int n_gammas_i1 = this->GList[2].size();
  int n_gammas_f;
  if (ig_i2 == -1){
    n_gammas_f = this->GList[3].size();
  }
  else{
    n_gammas_f = this->GList[4].size();
  }

  int TIME = this->localT();

  //put output to zero
  if (ig_i2 == -1){
    this->clear_output(!accum);
  }
  else{
    this->clear_output(!accum,5,ig_i2);
  }
  for( int t=0; t<TIME; ++t){
    #pragma omp parallel for
    for( int i_mom=0; i_mom<this->Nmoms(); ++i_mom){
      Float temp[N_SPINS*N_SPINS*2];
      for( int gi1=0; gi1<n_gammas_i1; ++gi1 ){
        for( int gf=0; gf<n_gammas_f; ++gf ){
          int coeffT = 1;
          if( transp_i1 == true)
             coeffT*=gammaTranspSign_scatt[this->GList[2][gi1]];
          if( transp_f1 == true)
             coeffT*=gammaTranspSign_scatt[this->GList[4][gf]];
          for(int spin=0; spin<N_SPINS*N_SPINS*2; ++spin)
            temp[spin] = coeffT*T2.Corr(t,i_mom,gi1,gf)[spin];

          for( int gei=0; gei<n_extgammas_i; ++gei ){
            for( int gef=0; gef<n_extgammas_f; ++gef ){
              GAMMAS_SCATT extG_i1 = this->GList[0][gei];
              GAMMAS_SCATT extG_f1 = this->GList[1][gef];
              if (ig_i2 == -1){
                M_pe_GNG<Float>( this->Corr(t,i_mom,gei,gef,gi1,gf), extG_f1, extG_i1, temp );
              }
              else{
                M_pe_GNG<Float>( this->Corr(t,i_mom,gei,gef,gi1,ig_i2,gf), extG_f1, extG_i1, temp );
              }
            } //G_extf
          } //G_exti
        } //G_f
      } //G_i
    } //mom
  } //T
}

//####################(
//#  Other functions  #
//#####################

//this must be used only if the source is the one used in PLEGMA_ScattCorrelator

//attract_look_up_table contains the global coordinates with respect to the
//coherent sources applied
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::applyBoundaryConditions( bool antiperiodic, int n_coherent_source, int *attract_look_up_table ) {

  if(!antiperiodic) return;

  std::size_t n_t = this->labels.find("t");
  assert(n_t!=std::string::npos);

  int TIME = this->localT();
  if (TIME==0) return;
  int in_dofs = std::accumulate(ranges.begin()+n_t+1, ranges.end(), 2, std::multiplies<int>());
  int out_dofs = ranges[0]*offsets[0]/TIME/in_dofs;
  int maxT = this->endT() - this->startT();

  if (n_coherent_source >1 && attract_look_up_table==NULL) PLEGMA_error("attract_look_up_table must be created before using this function\n");

  for( int t=0; t<TIME; ++t){
    int t_local = (t>=maxT) ? (this->source[DIM_T]%HGC_localL[DIM_T]) + t - maxT : t;
    int t_global = HGC_procPosition[DIM_T] * HGC_localL[DIM_T] + t_local;
    int source_num= n_coherent_source > 1 ? attract_look_up_table[t_global] : this->source[DIM_T];
    if( t_global < source_num ){
      for( int o_dofs=0; o_dofs<out_dofs; ++o_dofs){
        for( int i_dofs=0; i_dofs<in_dofs; ++i_dofs){
          *(this->H_elem() + o_dofs*TIME*in_dofs + t*in_dofs  + i_dofs) = -*(this->H_elem() + o_dofs*TIME*in_dofs + t*in_dofs  + i_dofs);
        }
      }
    }
  }
}


template<typename Float>
void PLEGMA_ScattCorrelator<Float>::apply_phase(){
  std::size_t n_m = this->labels.find("m");
  assert(n_m!=std::string::npos);
  int N_moms = this->Nmoms();
  int in_dofs = std::accumulate(ranges.begin()+n_m+1, ranges.end(), 2, std::multiplies<int>());
  int out_dofs = ranges[0]*offsets[0]/N_moms/in_dofs;

  std::vector<std::vector<int>> mom_list=this->pList().pi1(); 

  #pragma omp parallel for
  for( int i_m=0; i_m<N_moms; ++i_m){
      Float phase=2*M_PI/(Float)HGC_totalL[0]* mom_list[i_m][0]*this->source[0]+
	2*M_PI/(Float)HGC_totalL[1]* mom_list[i_m][1]*this->source[1]+
	2*M_PI/(Float)HGC_totalL[2]* mom_list[i_m][2]*this->source[2];
      Float tmpreim[2]={cos(phase),sin(phase)};
      for( int o_dofs=0; o_dofs<out_dofs; ++o_dofs)
	x_e_cx<Float>( this->H_elem() + (o_dofs*N_moms+i_m)*in_dofs, tmpreim, in_dofs/2);
  }

}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::normalize_nstoch(int n_stoch){
  int in_dofs=ranges[0]*offsets[0];
  x_e_sx<Float>( this->H_elem() , 1./n_stoch, in_dofs/2);
}


template<typename Float>
void PLEGMA_ScattCorrelator<Float>::absorbGammai2Gammaf2momentumf2(PLEGMA_ScattCorrelator<Float> &srcCorr, int  i_gamma_i2, int i_gamma_f2, int i_pf1, bool forcetozero ){

  int TIME = this->localT();
  if (TIME==0) return;

  int Nlist= this->pList().N_list();

//  std::vector<std::vector<int>> moms_pinsertion_red = this->pList().uniq_p(2); //list of pf1 momenta needed here
  std::vector<std::vector<int>> moms_pinsertion_red = this->pList().uniq_p(Nlist-1); //list of pf1 momenta needed here


  std::vector<std::vector<int>> moms_pinsertion = srcCorr.pList().uniq_p(0); //list of pf1 in Nucleons PLEGMA_SC

  std::vector<int> i_pinsertions = srcCorr.pList().u_posix( 0, moms_pinsertion_red); //list of positions of moms_pf1_red momenta in moms_pf1 array


  int n_gammas_i2 = this->GList[0].size();
  int n_gammas_f2 = this->GList[1].size();
  int n_gammas_c  = this->GList[2].size();

  int Nmoms_c = srcCorr.Nmoms();


  auto imap = this->pList().index_map();



  std::size_t n_s1 = this->labels.find("d");
  std::size_t n_s2 = this->labels.find("l");
//  int LIM=TIME*Nmoms_c*n_gammas_c;
//  for (int i=0;i<LIM;++i){ 
//    printf("Source %d %e %e\n",i, srcCorr.H_elem()[2*i+0],srcCorr.H_elem()[2*i+1]);
//  }
  if ((n_s1==std::string::npos) && (n_s2==std::string::npos)) {
  for(int i_m=0; i_m<imap.size(); i_m++){
    int i_mom_f1 = imap[i_m][1];
    if (i_mom_f1!=i_pf1){
      continue;
    }
    int i_pc = i_pinsertions[imap[i_m][Nlist-1]]; //position of pf1 in moms_pf1 (tempNN)
    for(int t=0; t < TIME; ++t){
      for (int g1=0 ; g1 < n_gammas_i2 ; ++g1 ){//pi
        if (i_gamma_i2 != g1)
         continue;
        for (int g2=0 ; g2 < n_gammas_f2 ; ++g2 ){//pf1
          if (i_gamma_f2 != g2)
           continue;
          for (int g3=0; g3 < n_gammas_c; ++g3 ){
            //printf("REAL %e\n",srcCorr.H_elem()[0]);
	    //printf("IMAG %e\n",srcCorr.H_elem()[2*t*Nmoms_c*n_gammas_c+2*i_pc*n_gammas_c+2*g3+1]);
            this->Corr(t,i_m,g1,g2,g3)[0]=srcCorr.H_elem()[2*t*Nmoms_c*n_gammas_c+2*i_pc*n_gammas_c+2*g3+0];
            this->Corr(t,i_m,g1,g2,g3)[1]=srcCorr.H_elem()[2*t*Nmoms_c*n_gammas_c+2*i_pc*n_gammas_c+2*g3+1];//srcCorr.H_elem(t, i_pc, g3)[1];

          }
        }
      }
    }
  }
  }
  else{
  int derivLoopLength;
  if (n_s1==std::string::npos){
    derivLoopLength=N_DIMS*(N_DIMS-1);
  }
  else {
    derivLoopLength=N_DIMS;
  }
  for(int i_m=0; i_m<imap.size(); i_m++){
    int i_mom_f1 = imap[i_m][1];
    if (i_mom_f1!=i_pf1){
      continue;
    }
    int i_pc = i_pinsertions[imap[i_m][1]]; //position of pf1 in moms_pf1 (tempNN)
    for(int t=0; t < TIME; ++t){
      for (int g1=0 ; g1 < n_gammas_i2 ; ++g1 ){//pi
        if (i_gamma_i2 != g1)
         continue;
        for (int g2=0 ; g2 < n_gammas_f2 ; ++g2 ){//pf1
          if (i_gamma_f2 != g2)
           continue;
	  for (int dir=0; dir<derivLoopLength; ++dir){
            for (int g3=0; g3 < n_gammas_c; ++g3 ){

              //printf("REAL %e\n",srcCorr.H_elem()[2*t*Nmoms_c*N_DIMS*n_gammas_c+2*i_pc*n_gammas_c*N_DIMS+2*dir*n_gammas_c+2*g3+0]);
	      //printf("IMAG %e\n",srcCorr.H_elem()[2*t*Nmoms_c*N_DIMS*n_gammas_c+2*i_pc*n_gammas_c*N_DIMS+2*dir*n_gammas_c+2*g3+1]);
              this->Corr(t,i_m,g1,g2,g3,dir)[0]=srcCorr.H_elem()[2*t*Nmoms_c*derivLoopLength*n_gammas_c+2*i_pc*n_gammas_c*derivLoopLength+2*dir*n_gammas_c+2*g3+0];
              this->Corr(t,i_m,g1,g2,g3,dir)[1]=srcCorr.H_elem()[2*t*Nmoms_c*derivLoopLength*n_gammas_c+2*i_pc*n_gammas_c*derivLoopLength+2*dir*n_gammas_c+2*g3+1];//srcCorr.H_elem(t, i_pc, g3)[1];
	    }
          }
        }
      }
    }
  }

  }

}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::absorbSourceSinkSpinMom(PLEGMA_ScattCorrelator<Float> &srcCorr, int alpha, int beta, int pf1, bool forcetozero){

  std::vector<std::vector<int>> moms_pinsertion_red = this->pList().uniq_p(2); //list of pf1 momenta needed here
  std::vector<std::vector<int>> moms_pinsertion = srcCorr.pList().uniq_p(0); //list of pf1 in Nucleons PLEGMA_SC
  std::vector<int> i_pinsertions = srcCorr.pList().u_posix( 0, moms_pinsertion_red); //list of positions of moms_pf1_red momenta in moms_pf1 array

//  int n_gammas_exti = this->GList[0].size();
//  int n_gammas_extf = this->GList[1].size();
  int n_gammas_i1 = this->GList[2].size();
  int n_gammas_f1 = this->GList[3].size();
  int n_gammas_c = this->GList[4].size();
  int TIME = this->localT();
  if (TIME==0) return;

  int Nmoms_c = srcCorr.Nmoms();

  //PLEGMA_ScattCorrelator<Float> V3aux(srcV2.getSource(), srcV2.getMomList(), srcV2.getTotalT());
  auto imap = this->pList().index_map();

  #pragma omp parallel for
  for(int i_m=0; i_m<imap.size(); i_m++){ 
    int i_mom_f1 = imap[i_m][1];
    if (i_mom_f1!=pf1){
      continue;
    }
    int i_pc = i_pinsertions[imap[i_m][2]]; //position of pf1 in moms_pf1 (tempNN)

    for(int t=0; t < TIME; ++t){
      for (int g1=0 ; g1 < n_gammas_i1 ; ++g1 ){//pi
        for (int g2=0 ; g2 < n_gammas_f1 ; ++g2 ){//pf1
          for (int g3=0; g3 < n_gammas_c; ++g3 ){
            this->Corr(t,i_m,0,0,g1,g2,g3,alpha,beta)[0]=srcCorr.H_elem()[2*t*Nmoms_c*n_gammas_c+2*i_pc*n_gammas_c+2*g3+0];
            this->Corr(t,i_m,0,0,g1,g2,g3,alpha,beta)[1]=srcCorr.H_elem()[2*t*Nmoms_c*n_gammas_c+2*i_pc*n_gammas_c+2*g3+1];//srcCorr.H_elem(t, i_pc, g3)[1];
          }
	}
      }
    }
  }
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::absorbTimeslice(PLEGMA_ScattCorrelator<Float> &srcCorr, int global_it, bool forcetozero){

//  if( this->pList().pi(1) != srcCorr.getMomList() ) PLEGMA_error("ScattCorrelator has not the the same mom list of srcCorr\n");

//  if(this->GList.size() !=srcCorr.GList.size())  PLEGMA_error("ScattCorrelator has not the the same length of GList list of srcCorr\n");
//  for(int i=0; i<(this->GList.size()); ++i)
//    if((this->GList[i]!=srcCorr.GList[i]))
//      PLEGMA_error("ScattCorrelator, srcCorr wrong gamma list\n");


  if(global_it >= HGC_totalL[3]) PLEGMA_error("The global time slice you provided exceed the temporal extent\n");

  int my_it = global_it - comm_coord(3) * HGC_localL[3];

  bool is_myIt = (my_it >= 0) && ( my_it < HGC_localL[3] );

  if (forcetozero == true && !this->labels.empty()){
    int tot_size = 2*this->getTotalSize();
    memset( this->H_elem(), 0, tot_size*sizeof(Float) );
  }


  if (is_myIt){
    int TIME;
    if (this->labels.empty()){
      this->labels=srcCorr.labels;
     
      for (auto &elem : srcCorr.GList){
	this->GList.push_back(elem);
      }
      this->datasets=srcCorr.datasets;
      this->groups=srcCorr.groups;
      this->shape=srcCorr.shape;
      this->initialize();
      this->setOffsets();
      TIME=0;

    }
    else {
      TIME=this->localT();
    }

    std::size_t n_t_src = srcCorr.labels.find("t");
    
    int TIME_src= srcCorr.localT();

    int in_dofs_src = std::accumulate(srcCorr.ranges.begin()+n_t_src+1, srcCorr.ranges.end(), 2, std::multiplies<int>());
    int out_dofs_src = srcCorr.ranges[0]*srcCorr.offsets[0]/TIME_src/in_dofs_src;
 
    for( int o_dofs=0; o_dofs<out_dofs_src; ++o_dofs){
      for( int i_dofs=0; i_dofs<in_dofs_src; ++i_dofs){
        *(this->H_elem() + o_dofs*TIME*in_dofs_src + my_it*in_dofs_src  + i_dofs) = *(srcCorr.H_elem() + o_dofs*TIME*in_dofs_src + my_it*in_dofs_src  + i_dofs);  	
      }
    }

    int coords[4];
    for(int i = 0 ; i < (N_DIMS-1); i++) coords[i] = 0;
    coords[3]= global_it / HGC_localL[3];
    int rankHas = quda::comm_rank_from_coords(coords);
    int mpiErr = MPI_Bcast(this->H_elem(), in_dofs_src*out_dofs_src , MPI_Type<Float>(), rankHas, HGC_fullComm);
    if(mpiErr != MPI_SUCCESS) PLEGMA_error("MPI_Bcast failed with error %d\n", mpiErr);
    
  }
  comm_barrier();

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

// multiply data per sign coming from transposition of one Gamma matrix   (G.T=sign*G)
// signs depend on GList[gi][:] and that axis is multiplied accordingly
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::apply_sign_transp(int gi){
  std::size_t n_g = this->labels.find('g');
  assert(n_g!=std::string::npos);
  n_g += gi;

  int N_gammas = this->GList[gi].size();
  int in_dofs  = std::accumulate(ranges.begin()+n_g+1, ranges.end(), 2, std::multiplies<int>());
  int out_dofs = ranges[0]*offsets[0]/N_gammas/in_dofs;
  auto sign_arr = gammaTranspSign_scatt;
  
  for( int o_dofs=0; o_dofs<out_dofs; ++o_dofs){
    for( int i_g=0; i_g < N_gammas; ++i_g ){
      Float sign[2]={ (Float)sign_arr[this->GList[gi][i_g]], 0. };
      x_e_cx<Float>( this->H_elem() + (o_dofs*N_gammas+i_g)*in_dofs, sign, in_dofs/2 );
    }
  }
}

// multiply data per sign coming from adjoint of one Gamma matrix   (g4*G.T.conj()*g4=sign*G)
// signs depend on GList[gi][:] and that axis is multiplied accordingly
template<typename Float>
void PLEGMA_ScattCorrelator<Float>::apply_sign_adj(int gi){
  std::size_t n_g = this->labels.find('g');
  assert(n_g!=std::string::npos);
  n_g += gi;

  int N_gammas = this->GList[gi].size();
  int in_dofs  = std::accumulate(ranges.begin()+n_g+1, ranges.end(), 2, std::multiplies<int>());
  int out_dofs = ranges[0]*offsets[0]/N_gammas/in_dofs;
  auto sign_arr = gammaAdjointSign_scatt;

  PLEGMA_printf("Apply_sign_adj: gi=%d, N_gammas=%d, indofs=%d, outdofs=%d\n",gi,N_gammas, in_dofs, out_dofs);
  PLEGMA_printf("Apply_sign_adj: gi=%d, begin_arr= [",gi);
  
  for( int i_g=0; i_g < N_gammas; ++i_g ){
    Float sign[2] = { (Float)sign_arr[this->GList[gi][i_g]], 0. };
    PLEGMA_printf("(%s->%f %f),", GAMMAS_SCATT_STR[this->GList[gi][i_g]].c_str(), sign[0], sign[1] );
    for( int o_dofs=0; o_dofs<out_dofs; ++o_dofs){
      x_e_cx<Float>( this->H_elem() + (o_dofs*N_gammas+i_g)*in_dofs, sign, in_dofs/2);
    }
  }
  PLEGMA_printf("Apply_sign_adj end\n");
}

template<typename Float>
void PLEGMA_ScattCorrelator<Float>::apply_sign(std::string name_of_diagram ){

  if ( name_of_diagram == "D" ){
    Float overall_sign[2] = {-1., 0.}; // -1 epsilon in adjoint interp
    x_e_cx<Float>( this->H_elem(), overall_sign, this->getTotalSize());
    this->apply_sign_adj(0);     // adjoint G_ei (Delta-extsource)
    this->apply_sign_adj(2);     // adjoint G_i1 (Delta-source)
  }
  else if( name_of_diagram == "N"){
    Float overall_sign[2] = {-1., 0.}; // -1 epsilon in adjoint interp
    x_e_cx<Float>( this->H_elem(), overall_sign, this->getTotalSize());
    this->apply_sign_adj(0);     // adjoint G_ei (Nucleon-extsource)
    this->apply_sign_adj(2);     // adjoint G_i1 (Nucleon-source)
  }
  else if( name_of_diagram == "P"){
    //Float overall_sign[2] = {1. ,0.}; // i at sink, -i at source
    //x_e_cx<Float>( this->H_elem(), overall_sign, this->getTotalSize());
    this->apply_sign_adj(0);     // adjoint G_i2 (Pion-source)
  }
  else if( name_of_diagram == "NJNP"){
    //Float overall_sign[2] = {1. ,0.}; // i at sink, -i at source
    //x_e_cx<Float>( this->H_elem(), overall_sign, this->getTotalSize());
    this->apply_sign_adj(3);     // adjoint G_i2 (Pion-source)
  }
  else if( name_of_diagram == "T"){
    Float overall_sign[2] = {0.,1.}; //-i from pion at source, -1 coming from epsilon in adj interp of N
    x_e_cx<Float>( this->H_elem(), overall_sign, this->getTotalSize());
    this->apply_sign_adj(0);     // adjoint G_ei (Nucleon-extsource)
    this->apply_sign_adj(2);     // adjoint G_i1 (Nucleon-source)
    this->apply_sign_adj(3);     // adjoint G_i2 (Pion-source)
  }
  else if(  name_of_diagram == "T1"){
    Float overall_sign[2] = {0.,-1.}; //i from pion at sink, -1 comig from epsilon in adj interp of D
    x_e_cx<Float>( this->H_elem(), overall_sign, this->getTotalSize());
    this->apply_sign_adj(0);     // adjoint G_ei (Delta-extsource)
    this->apply_sign_adj(2);     // adjoint G_i1 (Delta-source)
  }
  else if( name_of_diagram == "4pt"){
    // ??????? is N called after or before phase multiplication of M !!!!!!! In the following like we call signs for N after the M_diagram call.
    Float overall_sign = -1.; // -1 coming from epsilon in adj interp of N 
    x_e_sx<Float>( this->H_elem(), overall_sign, this->getTotalSize());
    apply_sign_adj(0);     // adjoint G_ei (Nucleon-extsource)
    apply_sign_adj(2);     // adjoint G_i1 (Nucleon-source)
    apply_sign_adj(3);     // adjoint G_i2 (Pion-source)
  }
  else if( name_of_diagram == "NPJP"){
    // ??????? is N called after or before phase multiplication of M !!!!!!! In the following like we call signs for N after the M_diagram call.
    Float overall_sign = -1.; // -1 coming from epsilon in adj interp of N
    x_e_sx<Float>( this->H_elem(), overall_sign, this->getTotalSize());
    apply_sign_adj(0);     // adjoint G_ei (Nucleon-extsource)
    apply_sign_adj(2);     // adjoint G_i1 (Nucleon-source)
    apply_sign_adj(3);     // adjoint G_i2 (Pion-source)
  }
  else if( name_of_diagram == "L"){
    Float overall_sign[2] = {0.,1.}; //i from pion interpolating operator
    x_e_cx<Float>( this->H_elem(), overall_sign, this->getTotalSize());
  }
  else{
    PLEGMA_error("Error! %s not recognized!\n",name_of_diagram.c_str());
  }
}

template class PLEGMA_ScattCorrelator<float>;
template class PLEGMA_ScattCorrelator<double>;


