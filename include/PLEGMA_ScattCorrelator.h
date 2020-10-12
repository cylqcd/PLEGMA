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

    std::vector<std::vector<GAMMAS_SCATT>> GList;
    momList plist;

    std::string labels;    //     index_struct = "gsssc" (because spin first)
    std::vector<int> offsets;
    std::vector<int> ranges;
    
  public:
    // these constructors does NOT ALLOCATE the memory PLEGMA_ScattCorrelator here, because
    // the dimension is not provided. It will be allocated when used.
    PLEGMA_ScattCorrelator(site source, int Q2_max, int totalT=HGC_totalL[DIM_T]);

    PLEGMA_ScattCorrelator(site source, std::vector<int> fixMomVec, int totalT=HGC_totalL[DIM_T]);

    PLEGMA_ScattCorrelator(site source, std::vector<std::vector<int>> fixMomsVec, int totalT=HGC_totalL[DIM_T]);

    PLEGMA_ScattCorrelator(site source, momList &listmom, int totalT=HGC_totalL[DIM_T]);

    ~PLEGMA_ScattCorrelator(){;}

    //functions that return values of protected variables
    std::string getLabels() const{ return labels; }
    std::vector<std::vector<GAMMAS_SCATT>> getGList(){ return GList; }

    //momenta
    // in 4pt functions the order of momenta is determined by momList
    // in 2pt functions by the order of momList.uniq_p( i ) where i is the momentum you take
    // in 3pt function the order of momenta is deretmined by momList.uniq_p(tot)
    momList& pList(){
	return plist;
    }
    
    int Nmoms(){
      assert(this->corr_mom_space);
      return this->corr_mom_space->Nmoms();
    }
      
    //checks
    void setOffsets( );

    template<int N>
    __inline__ long sum_offsets(int iN) const {
      return iN*offsets[N];
    }

    template<int N, typename ... Args>
    __inline__ long sum_offsets(int iN, Args... others) const {
      return iN*offsets[N] + sum_offsets<N+1>(others...);
    }

    template<typename ... Args>
    __inline__ Float* Corr( Args... is ) const {
      return this->H_elem() + sum_offsets<0>(is...);
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
    void V3V2reduction( PLEGMA_ScattCorrelator<Float> &srcV3,PLEGMA_ScattCorrelator<Float> &srcV2, int index_abs, bool transp, int g0, bool transp_i1=false, Float* factor=NULL, bool transp_f1=false );

    void V3V2reduction_matrix( PLEGMA_ScattCorrelator<Float> &srcV3, PLEGMA_ScattCorrelator<Float> &srcV2, int index_abs, bool transp,  int g0, bool transp_i1=false, Float* factor=NULL, bool transp_f1=false );
    

    //initialize_diagrams
    void initialize_diagram( std::vector<GAMMAS_SCATT> &G_i2, std::vector<GAMMAS_SCATT> &G_f2, std::string name_of_diagram );//P
    void initialize_diagram( std::vector<GAMMAS_SCATT> &eG_i, std::vector<GAMMAS_SCATT> &eG_f,
			     std::vector<GAMMAS_SCATT> &G_i1, std::vector<GAMMAS_SCATT> &G_f1, std::string name_of_diagram );//N,D
    void initialize_diagram( std::vector<GAMMAS_SCATT> &eG_i, std::vector<GAMMAS_SCATT> &eG_f,
			     std::vector<GAMMAS_SCATT> &G_i1, std::vector<GAMMAS_SCATT> &G_i2, std::vector<GAMMAS_SCATT> &G_f, std::string name_of_diagram );//T
    void initialize_diagram( std::vector<GAMMAS_SCATT> &eG_i, std::vector<GAMMAS_SCATT> &eG_f,
			     std::vector<GAMMAS_SCATT> &G_i1, std::vector<GAMMAS_SCATT> &G_i2, std::vector<GAMMAS_SCATT> &G_f1, std::vector<GAMMAS_SCATT> &G_f2, std::string name_of_diagram );//B,W,Z,M

    
    //diagrams
    void B_diagramms( PLEGMA_ScattCorrelator<Float> &srcV3, PLEGMA_ScattCorrelator<Float> &srcV2, int ig_i2, int diagram_index, bool accum=false );
    void W_diagramms( PLEGMA_ScattCorrelator<Float> &srcV3, PLEGMA_ScattCorrelator<Float> &srcV2, int ig_i2, int diagramm_index, bool accum=false );
    void Z_diagramms( std::array<PLEGMA_ScattCorrelator<Float>,4> (&srcV3), std::array<PLEGMA_ScattCorrelator<Float>,4> (&srcV2),int diagramm_index, bool accum=false );
    void M_diagramms( PLEGMA_ScattCorrelator<Float> &CorrNucleon, std::array<PLEGMA_Vector<Float>,4> &Phi_0, std::array<PLEGMA_Vector<Float>,4> &Phi_1, bool accum=false );

    void T_diagramms( PLEGMA_ScattCorrelator<Float> &T1, PLEGMA_ScattCorrelator<Float> &T3, PLEGMA_ScattCorrelator<Float> &T5, int ig_i2, bool accum=false );


    void T_diagramms_piNsink(PLEGMA_ScattCorrelator<Float> &srcV2, PLEGMA_ScattCorrelator<Float> &srcV3, bool accum=false);

    void P_diagramms( std::array<PLEGMA_Vector<Float>,4> &Phi_0, std::array<PLEGMA_Vector<Float>,4> &Phi_1, int i_pi2, bool accum=false );
    void N_diagramms( PLEGMA_ScattCorrelator<Float> &T1, PLEGMA_ScattCorrelator<Float> &T2, bool accum=false );
    void D_diagramms( PLEGMA_ScattCorrelator<Float> &T1, PLEGMA_ScattCorrelator<Float> &T2, bool accum=false );
    void convertTreductiontoDiagram( PLEGMA_ScattCorrelator<Float> &T2, bool accum=false );


    //others
    void applyBoundaryConditions( bool antiperiodic );
    void apply_phase( );
    void normalize_nstoch(int n_stoch);
    void clear_output( bool tozero, int n_index, int i );
    void clear_output( bool tozero );
    void apply_sign_transp(int gi);
    void apply_sign_adj(int gi);
    void apply_sign( std::string name_of_diagram );

  };

  template <int s_free, typename Float>
  void absorb_fromV24( Float dest[N_SPINS*N_COLS*2], Float* src, int alfa, int beta ){
    if( s_free == 0){
      for(int s=0; s < N_SPINS; ++s)
	for(int c=0; c < N_COLS; ++c)
	  for(int ri=0; ri<2; ++ri)
	    dest[(s*N_COLS+c)*2+ri]  = src[(((s*N_SPINS+alfa)*N_SPINS+beta)*N_COLS+c)*2+ri];
    } else if ( s_free == 1 ){
      for(int s=0; s < N_SPINS; ++s)
	for(int c=0; c < N_COLS; ++c)
	  for(int ri=0; ri<2; ++ri)
	    dest[(s*N_COLS+c)*2+ri]  = src[(((alfa*N_SPINS+s)*N_SPINS+beta)*N_COLS+c)*2+ri];
    } else {
      for(int s=0; s < N_SPINS; ++s)
	for(int c=0; c < N_COLS; ++c)
	  for(int ri=0; ri<2; ++ri)
	    dest[(s*N_COLS+c)*2+ri]  = src[(((alfa*N_SPINS+beta)*N_SPINS+s)*N_COLS+c)*2+ri];
    } 
  }

  template <int s_fixed, typename Float>
  void absorbspinmatrix_fromV24( Float dest[N_SPINS*N_SPINS*N_COLS*2], Float* src, int alfa ){
    if( s_fixed == 2){
      for(int s1=0; s1 < N_SPINS; ++s1)
	for( int s2=0; s2 < N_SPINS; ++s2)
	  for(int c=0; c < N_COLS; ++c)
	    for(int ri=0; ri<2; ++ri)
	      dest[((s1*N_SPINS+s2)*N_COLS+c)*2+ri]  = src[(((s1*N_SPINS+s2)*N_SPINS+alfa)*N_COLS+c)*2+ri];
    } else if ( s_fixed == 1 ){
      for(int s1=0; s1 < N_SPINS; ++s1)
	for( int s2=0; s2 < N_SPINS; ++s2)
	  for(int c=0; c < N_COLS; ++c)
	    for(int ri=0; ri<2; ++ri)
	      dest[((s1*N_SPINS+s2)*N_COLS+c)*2+ri]  = src[(((s1*N_SPINS+alfa)*N_SPINS+s2)*N_COLS+c)*2+ri];
    } else {
      for(int s1=0; s1 < N_SPINS; ++s1)
	for( int s2=0; s2 < N_SPINS; ++s2)
	  for(int c=0; c < N_COLS; ++c)
	    for(int ri=0; ri<2; ++ri)
	      dest[((s1*N_SPINS+s2)*N_COLS+c)*2+ri]  = src[(((alfa*N_SPINS+s1)*N_SPINS+s2)*N_COLS+c)*2+ri];
    }
  }

  
}
