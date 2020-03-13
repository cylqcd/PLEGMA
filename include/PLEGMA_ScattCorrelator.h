#pragma once
#include <PLEGMA_Correlator.h>
#include <PLEGMA_gammas.h>
#include <utils/PLEGMA_scatt_utils.h>

namespace plegma {
  enum VRED {V_2=2,V_3=3,V_4=4};
  enum TRED {T_1=1,T_2=2};  
  // forward declaration
  template<typename Float>  class PLEGMA_Vector;
  template<typename Float>  class PLEGMA_Propagator;

  
  /////////////////
  // This PLEGMA_ScattCorrelator Class allows to store a generic number of d.o.f
  // (divided into spin and color index) per momentum. It  
  // will contain all the needed contractions V1, V2, V3, V4, T1, T2 plus
  // other functionalities ( print utilities inherited from Correlators, ....)
  ////////////////
  //
  // N.B. shape is the d.o.f per site -> shape of dataset
  //      other possible variables (for instance, a list of gammas) -> n_datasets*n_groups
 
  template<typename Float>
  class PLEGMA_ScattCorrelator : public PLEGMA_Correlator<Float>  {
  protected:
    //////////////
    // from PLEGMA_Correlator:
    //////////////
    //   // Allocation
    //   bool isAlloc;
    //   PLEGMA_Field<Float>* corr_pos_space;
    //   PLEGMA_FT<Float>* corr_mom_space;
    //   Float* corr;

    //   // Correlator info
    //   CORR_SPACE corr_space;
    //   int Q2_max;
    //   std::vector<int> fixMomVec ;
    //   size_t vol_size;
    //   std::vector<int> shape;

    //   // Allocated site_size = n_datasets * n_groups * prod(shape) (slowest to fastest running index)
    //   int site_size;
    //   std::array<int,4> source_position;

    //   // Writing informations
    //   std::vector<std::string> datasets;
    //   std::vector<std::string> groups;
    //   std::string description;
    //////////////

    std::vector<std::vector<GAMMAS_SCATT>> GList;
    momList pList;
    int N_p=0;

    std::string labels;    //     index_struct = "gsssc" (because spin first)
    std::vector<int> offsets;
    std::vector<int> ranges;
    
    void absorb_fromV24_checks( PLEGMA_ScattCorrelator<Float> &srcV2, int alfa, int beta);
    void absorbspinmatrix_fromV24_checks( PLEGMA_ScattCorrelator<Float> &srcV2, int alfa);
    void contract_GxV2_checks( PLEGMA_ScattCorrelator<Float> &srcV2);



  public:
    // these constructors does NOT ALLOCATE the memory PLEGMA_ScattCorrelator here, because
    // the dimension is not provided. It will be allocated when used.
    PLEGMA_ScattCorrelator(site source, int Q2_max, int totalT=HGC_totalL[DIM_T]);

    PLEGMA_ScattCorrelator(site source, std::vector<int> fixMomVec, int totalT=HGC_totalL[DIM_T]);

    PLEGMA_ScattCorrelator(site source, std::vector<std::vector<int>> fixMomsVec, int totalT=HGC_totalL[DIM_T]);

    ~PLEGMA_ScattCorrelator(){;}

    //functions that return values of protected variables
    std::string getLabels() const{ return labels; }
    std::vector<std::vector<GAMMAS_SCATT>> getGList(){ return GList; }

    //momenta
    // in 4pt functions the order of momenta is determined by momList
    // in 2pt functions by the order of momList.uniq_p( i ) where i is the momentum you take
    // in 3pt function the order of momenta is deretmined by momList.uniq_p(tot)
    
    void setPList( momList &list_p ){
      this->pList = list_p;
    }

    int Nmoms(){
      assert(this->corr_mom_space);
      return this->corr_mom_space->Nmoms();
    }
      
    //checks
    void setOffsets( );

    Float* Corr(  std::initializer_list<int> idx ) const {
      if(offsets.size()<idx.size())
	PLEGMA_error("Number of indices (%d) greater than size of correlator\n",idx.size());
      int check=std::inner_product( idx.begin(), idx.end(), offsets.begin(), 0);
      if( check >= this->getTotalSize() ){
	int i=0;
	for(auto id: idx){ PLEGMA_printf("(%d/%d)-",id,ranges[i]); i++; }
	PLEGMA_error("\n Error check:%d >= totsize:%d\n",check,this->getTotalSize());
      }
      return this->H_elem() + check;
    }

    bool check_reduction( VRED V );
    bool check_reduction( TRED T );

    //reductions
    void V2( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS_SCATT> &Gammas, PLEGMA_Propagator<Float> &S1,  PLEGMA_Propagator<Float> &S2 );
    void V3( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS_SCATT> &Gammas, PLEGMA_Propagator<Float> &S );
    void V4( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS_SCATT> &Gammas, PLEGMA_Propagator<Float> &S1,  PLEGMA_Propagator<Float> &S2 );
    void T1( std::vector<GAMMAS_SCATT> &Gammas_i, std::vector<GAMMAS_SCATT> &Gammas_f, PLEGMA_Propagator<Float> &S1, PLEGMA_Propagator<Float> &S2, PLEGMA_Propagator<Float> &S3 );
    void T2( std::vector<GAMMAS_SCATT> &Gammas_i, std::vector<GAMMAS_SCATT> &Gammas_f, PLEGMA_Propagator<Float> &S1, PLEGMA_Propagator<Float> &S2, PLEGMA_Propagator<Float> &S3 );
    void PhiPhi( PLEGMA_Vector<Float> &Phi_0, std::vector<GAMMAS_SCATT> &Gammas,  PLEGMA_Vector<Float> &Phi_1 );


    //manipulation
    template <int s_free>
    void absorb_fromV24( PLEGMA_ScattCorrelator<Float> &srcV2like, int alfa, int beta );
    template <int s_fixed>
    void absorbspinmatrix_fromV24( PLEGMA_ScattCorrelator<Float> &srcV2like, int alfa );
    
    void V3V2reduction( PLEGMA_ScattCorrelator<Float> &srcV3,PLEGMA_ScattCorrelator<Float> &srcV2, int index_abs, bool transp, int g0, bool transpgamma=false, Float* factor=NULL );

    void V3V2reduction_matrix( PLEGMA_ScattCorrelator<Float> &srcV3, PLEGMA_ScattCorrelator<Float> &srcV2, int index_abs, bool transp,  int g0, bool transpgamma=false, Float* factor=NULL );
    

    //initialize_diagrams
    void initialize_diagram( momList &momenta, std::vector<GAMMAS_SCATT> &G_i2, std::vector<GAMMAS_SCATT> &G_f2, std::string name_of_diagram, bool inM=false );//P
    void initialize_diagram( momList &momenta, std::vector<GAMMAS_SCATT> &eG_i, std::vector<GAMMAS_SCATT> &eG_f,
			     std::vector<GAMMAS_SCATT> &G_i1, std::vector<GAMMAS_SCATT> &G_f1, std::string name_of_diagram );//N,D
    void initialize_diagram( momList &momenta, std::vector<GAMMAS_SCATT> &eG_i, std::vector<GAMMAS_SCATT> &eG_f,
			     std::vector<GAMMAS_SCATT> &G_i1, std::vector<GAMMAS_SCATT> &G_i2, std::vector<GAMMAS_SCATT> &G_f, std::string name_of_diagram );//T
    void initialize_diagram( momList &momenta, std::vector<GAMMAS_SCATT> &eG_i, std::vector<GAMMAS_SCATT> &eG_f,
			     std::vector<GAMMAS_SCATT> &G_i1, std::vector<GAMMAS_SCATT> &G_i2, std::vector<GAMMAS_SCATT> &G_f1, std::vector<GAMMAS_SCATT> &G_f2, std::string name_of_diagram );//B,W,Z,M

    
    //diagrams
    void B_diagramms( PLEGMA_ScattCorrelator<Float> &srcV3, PLEGMA_ScattCorrelator<Float> &srcV2, int ig_i2, int diagram_index, bool accum=false );
    void W_diagramms( PLEGMA_ScattCorrelator<Float> &srcV3, PLEGMA_ScattCorrelator<Float> &srcV2, int ig_i2, int diagramm_index, bool accum=false );
    void Z_diagramms( std::array<PLEGMA_ScattCorrelator<Float>,4> (&srcV3), std::array<PLEGMA_ScattCorrelator<Float>,4> (&srcV2),int diagramm_index, bool accum=false );
    void M_diagramms( PLEGMA_ScattCorrelator<Float> &CorrNucleon, std::array<PLEGMA_Vector<Float>,4> &Phi_0, std::array<PLEGMA_Vector<Float>,4> &Phi_1, bool accum=false );

    void T_diagramms( PLEGMA_ScattCorrelator<Float> &T1, PLEGMA_ScattCorrelator<Float> &T3, PLEGMA_ScattCorrelator<Float> &T5, int ig_i2, bool accum=false );

    void P_diagramms( std::array<PLEGMA_Vector<Float>,4> &Phi_0, std::array<PLEGMA_Vector<Float>,4> &Phi_1, bool accum=false, int pi=0 );
    void N_diagramms( PLEGMA_ScattCorrelator<Float> &T1, PLEGMA_ScattCorrelator<Float> &T2, bool accum=false );
    void D_diagramms( PLEGMA_ScattCorrelator<Float> &T1, PLEGMA_ScattCorrelator<Float> &T2, bool accum=false );

    //others
    void applyBoundaryConditions( bool antiperiodic );
    void apply_phase( std::vector<std::vector<int>> mom_list );
    void clear_output( bool tozero, int n_index, int i );
    void clear_output( bool tozero );

  };


}

using namespace plegma;

//template functions must be defined here
/*
template<typename Float>
template <int s_free>
void PLEGMA_ScattCorrelator<Float>::contract_GxV2( PLEGMA_ScattCorrelator<Float> &srcV2, GAMMAS_SCATT &G, bool transp){
  if( s_free<0 || s_free>=3 )
    PLEGMA_error("s_free %d out of range (0, 1 or 2)\n",s_free);
  this->contract_GxV2_checks(srcV2);

  int n_gammas = srcV2.GList.size();
  const unsigned short N_S1C=N_SPINS*N_COLS;
  const unsigned short N_S2C=N_SPINS*N_SPINS*N_COLS;
  const unsigned short N_S3C=N_SPINS*N_SPINS*N_SPINS*N_COLS;
  const unsigned short N_GS3C=n_gammas*N_SPINS*N_SPINS*N_SPINS*N_COLS;
  const unsigned short N_GS1C=n_gammas*N_SPINS*N_COLS;
    
  size_t VOL_SIZE = srcV2.getVolSize();
  Float* dest = this->H_elem();
  Float* src = srcV2.H_elem();
  
  for(int v=0; v < VOL_SIZE; v++)
    for(int g=0; g < n_gammas; g++)
      for(int s=0; s < N_SPINS; s++)
      	for(int c=0; c < N_COLS; c++){
	  dest[(v*N_GS1C+g*N_S1C+s*N_COLS+c)*2] = 0.;
	  dest[(v*N_GS1C+g*N_S1C+s*N_COLS+c)*2 + 1] = 0.;
	  for(int e_nz=0; e_nz<4; e_nz++){
	    int alfa = (transp) ? gammaInd_scatt_host[G][e_nz][1] : gammaInd_scatt_host[G][e_nz][0];
	    int beta = (transp) ? gammaInd_scatt_host[G][e_nz][0] : gammaInd_scatt_host[G][e_nz][1];
	    std::complex<Float> g(gamma_scatt_host[G][e_nz][0],gamma_scatt_host[G][e_nz][1]);
	    std::complex<Float> a;
	    Float* aux;
	    
	    if( s_free == 0){
	      aux = src + (v*N_GS3C+g*N_S3C+s*N_S2C+alfa*N_S1C+beta*N_COLS+c)*2;
	    } else if ( s_free == 1 ){
	      aux = src + (v*N_GS3C+g*N_S3C+alfa*N_S2C+s*N_S1C+beta*N_COLS+c)*2;
	    } else {
	      aux = src + (v*N_GS3C+g*N_S3C+alfa*N_S2C+beta*N_S1C+s*N_COLS+c)*2;
	    }
	    a = {*aux,*(aux+1)};
	    g = g*a;
	    dest[(v*N_GS1C+g*N_S1C+s*N_COLS+c)*2] += g.real();
	    dest[(v*N_GS1C+g*N_S1C+s*N_COLS+c)*2 + 1] += g.imag();
	  }
	}
	}*/

template<typename Float>
template <int s_free>
void PLEGMA_ScattCorrelator<Float>::absorb_fromV24( PLEGMA_ScattCorrelator<Float> &srcV2,
						    int alfa, int beta){
  this->absorb_fromV24_checks(srcV2, alfa, beta);
  if( s_free<0 || s_free>=3 )
    PLEGMA_error("s_free %d out of range (0, 1 or 2)\n",s_free);
    
  int n_gammas = srcV2.GList[0].size();
  int n_momenta = srcV2.Nmoms();

  for(int t=0; t < this->localT(); ++t)
    for(int m=0; m < Nmoms(); ++m)
      for(int g=0; g < n_gammas; ++g)
	for(int s=0; s < N_SPINS; ++s)
	  for(int c=0; c < N_COLS; ++c)
	    for(int ri=0; ri<2; ++ri)
	      if( s_free == 0){
		this->Corr({t,m,g,s,c})[ri]  = srcV2.Corr({t,m,g,s,alfa,beta,c})[ri];
	      } else if ( s_free == 1 ){
		this->Corr({t,m,g,s,c})[ri]  = srcV2.Corr({t,m,g,alfa,s,beta,c})[ri];
	      } else {
		this->Corr({t,m,g,s,c})[ri]  = srcV2.Corr({t,m,g,alfa,beta,s,c})[ri];
	      } 
}

//template functions must be defined here
template<typename Float>
template <int s_fixed>
void PLEGMA_ScattCorrelator<Float>::absorbspinmatrix_fromV24( PLEGMA_ScattCorrelator<Float> &srcV2, 
                                                              int alfa ){
  this->absorbspinmatrix_fromV24_checks(srcV2, alfa);
  if( s_fixed<0 || s_fixed>=3 )
    PLEGMA_error("s_fixed %d out of range (0, 1 or 2)\n",s_fixed);

  int n_gammas = GList[0].size();
   
  for(int t=0; t < this->localT(); ++t)
    for(int m=0; m < Nmoms(); ++m)
      for(int g=0; g < n_gammas; ++g)
	for(int s1=0; s1 < N_SPINS; ++s1)
	  for( int s2=0; s2 < N_SPINS; ++s2)
	    for(int c=0; c < N_COLS; ++c)
	      for(int ri=0; ri<2; ++ri)
		if( s_fixed == 2){
		  this->Corr({t,m,g,s1,s2,c})[ri] = srcV2.Corr({t,m,g,s1,s2,alfa,c})[ri];
		} else if ( s_fixed == 1 ){
		  this->Corr({t,m,g,s1,s2,c})[ri] = srcV2.Corr({t,m,g,s1,alfa,s2,c})[ri];
		} else {
		  this->Corr({t,m,g,s1,s2,c})[ri] = srcV2.Corr({t,m,g,alfa,s1,s2,c})[ri];
		}
}
