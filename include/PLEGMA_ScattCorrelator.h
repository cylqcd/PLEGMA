#pragma once
#include <PLEGMA_Correlator.h>
#include <PLEGMA_gammas.h>
#include <utils/PLEGMA_scatt_utils.h>

namespace plegma {
  enum VRED {V_2=2,V_3=3,V_4=4,V_5=5,V_6=6,V_6_RED=7};
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
    /**
     *
     *  @brief performs V2 type reduction produces three spinor and and color indices tensor from a fermion vector and two fermion 
     *         propagator. It is used for forming diagrams to 2 hadron 2 pt correlation function where the sink to sink propagator 
     *         is replaced by a stochastic one.
     *         Formula:
     *         V2_{Gamma}^{alfa0,alfa1,alfa2}_{n}=\eps_{a,b,c}\eps_{l,m,n}Gamma^{beta0,beta1}*phi^{beta0}_{c}*S1^{beta1,alfa0}_{b,m}*S2^{alfa1,alfa2}_{a,l}
     *  @params PLEGMA_Vector<Float> &Phi When the propagator that should be replaced is a DD type, then this should be gamma_5 *
     *                               stochastic source, when it is UU type then is should be the stochastic propagator itself
     *                               without gamma_5 multiplication
     *  @params std::vector<GAMMAS_SCATT> &Gammas list of gammas with which the stochastic vector (argument above) will be multiplied
     *  @params PLEGMA_Propagator<Float> &S1 fermion propagator
     *  @params PLEGMA_Propagator<Float> &S2 fermion propagator
     *  @params bool conj_v=false perform /or not to perform a conjugation on the fermion vector
     *          in most cases we do set this to false
     *
     **/
    void V2( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS_SCATT> &Gammas, PLEGMA_Propagator<Float> &S1,  PLEGMA_Propagator<Float> &S2,bool conj_v=false );
    /**
     *
     *  @brief performs V3 type reduction produces one spin and one color indices tensor from a fermion vector and a fermion propagator
     *         It is used for forming diagrams for 2 hadron 2pt correlation function where the sink to sink propagator is replaced by
     *         a stochastic one
     *         Formula:
     *         V3_{Gamma}^{beta}_{b}=conj(phi)^{alpha0}_{a}*Gamma_{alpha0,alpha1}*S^{alpha1,beta}_{a,b}
     *  @params PLEGMA_Vector<Float> &Phi When the propagator that should be replaced is a DD type, then this should be gamma_5 *
     *                                    stochastic propagator, when it is UU type then is should be the stochastic source itself
     *                                    without gamma_5 multiplication
     *  @params std::vector<GAMMAS_SCATT> &Gammas list of gammas with which the stochastic vector (argument above) will be multiplied
     *  @params PLEGMA_Propagator<Float> &S1 fermion propagator
     *  @params bool conj_v=false perform /or not to perform a conjugation on the fermion vector
     *                      in most cases we do set this to true
     *
     **/
    void V3( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS_SCATT> &Gammas, PLEGMA_Propagator<Float> &S,bool conj_v=true );
    /**
     *
     *  @brief performs V4 type reduction produces three spin and one color indices tensor from a fermion vector and two fermion
     *         propagators. It is used for forming diagrams for 2 hadron 2pt correlation function where the sink to sink 
     *         propagator is replaced by a stochastic one
     *         Formula
     *         V4_{Gamma}^{alpha0,alpha1,alpha2}_{l}=\eps_{abc}\eps_{lmn}\phi^{alpha0}_{a}S1^{beta1,alpha1}_{b,m}\Gamma{beta0,beta1}*
     *                                               S2^{beta0,alpha2}
     *  @params PLEGMA_Vector<Float> &Phi When the propagator that should be replaced is a DD type, then this should be gamma_5 *
     *                                    stochastic propagator, when it is UU type then is should be the stochastic source itself
     *                                    without gamma_5 multiplication
     *  @params std::vector<GAMMAS_SCATT> &Gammas list of gammas with which the stochastic vector (argument above) will be multiplied
     *  @params PLEGMA_Propagator<Float> &S1 fermion propagator
     *  @params PLEGMA_Propagator<Float> &S2 fermion propagator
     *  @params bool conj_v=false perform /or not to perform a conjugation on the fermion vector 
     *                      in most cases we do set this to false
     *
     *
     **/
    void V4( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS_SCATT> &Gammas, PLEGMA_Propagator<Float> &S1,  PLEGMA_Propagator<Float> &S2, bool conj_v=false );
    /**
     *
     *  @brief performs V5 type reduction produces two spin and one color indices tensor from two fermion vectors
     *         It is used for forming diagrams for 2 hadron 2pt correlation function with one-end-trick
     *         Formula
     *         V5^{\kappa_1,\kappa_2}_{m}=\eps_{mab}{\phi}^{a*}_{\kappa_1}{\xi}^{b*}_{\kappa_2}
     *  @params PLEGMA_Vector<Float> &Phi1
     *  @params PLEGMA_Vector<Float> &Phi2
     *  @params bool conj_v=false perform /or not to perform a conjugation on the fermion vectors
     *                      in most cases we do set this to true
     *
     **/
    void V5( PLEGMA_Vector<Float> &Phi1, PLEGMA_Vector<Float> &Phi2, bool conj_v=true );
    /**
     *
     *  @brief performs V6 type reduction produces four spin and one color indices tensor from two fermion vectors
     *         and one fermion propagator
     *         Formula
     *         V6^{alpha;beta;gamma,delta}_{m}=\eps_{abc}{\phi}^{a}_{alpha}{\xi}^{b}_{\beta}S^{c,m}_{\gamma,\delta}
     * @params PLEGMA_Vector<Float> &Phi1
     * @params PLEGMA_Vector<Float> &Phi2
     * @params PLEGMA_Propagator<Float> &S1
     * @params bool conj_v=false perform /or not to perform a conjugation on the fermion vectors
     *  in most cases we do set this to false
     *
     **/
    void V6( PLEGMA_Vector<Float> &Phi1, PLEGMA_Vector<Float> &Phi2, PLEGMA_Propagator<Float> &S, bool conj_v=false );
    /** 
     *
     *  @brief performs V6 type reduction produces two spin and one color indices tensor from two fermion vectors
     *         and one fermion propagator
     *         Formula
     *         C1=0,C2=1
     *         V6^{gamma,delta}_{m}=\eps_{abc}{\phi}^{a}_{alpha}\Gamma_{\alpha,\beta}{\xi}^{b}_{\beta}S^{c,m}_{\gamma,\delta}
     *         C1=1,C2=2
     *         V6^{alpha,delta}_{m}=\eps_{abc}{\phi}^{a}_{alpha}{\xi}^{b}_{\beta}\Gamma_{\beta,\gamma}S^{c,m}_{\gamma,\delta}
     **/
    void V6_RED( PLEGMA_Vector<Float> &Phi1, PLEGMA_Vector<Float> &Phi2, std::vector<GAMMAS_SCATT> &Gammas, PLEGMA_Propagator<Float> &S1, int C1=0, int C2=1, bool conj_v=false);
    /**
     *  @brief performs T1 type reduction to compute baryon 2pt functions
     *  T1_{alpha,beta}=\epsilon_{a,b,c}\epsilon_{l,m,n}S1^{c,l}_{alpha,alpha0}\Gamma_{i}_{alpha0,alpha1}S2^{b,m}_{beta0,alpha1}\Gamma_{f}_{beta0,beta1}S3^{a,n}_{beta1,beta} 
     *  @param std::vector<GAMMAS_SCATT> &Gammas_i: list of gammas at the source for the contractions
     *  @param std::vector<GAMMAS_SCATT> &Gammas_f: list of gammas at the sink for the contractions
     *  @param PLEGMA_Propagator<Float> &S1: propagator to the sink spin index
     *  @param PLEGMA_Propagator<Float> &S2: propagator 2: it will be transposed in the code
     *  @param PLEGMA_Propagator<Float> &S3: propagator from the source spin index
     *
    **/
    void T1( std::vector<GAMMAS_SCATT> &Gammas_i, std::vector<GAMMAS_SCATT> &Gammas_f, PLEGMA_Propagator<Float> &S1, PLEGMA_Propagator<Float> &S2, PLEGMA_Propagator<Float> &S3 );
    /**
     *  @brief performs T2 type reduction to compute baryon 2pt functions
     *  T2_{alpha,beta}=\epsilon_{a,b,c}\epsilon_{l,m,n}S1^{c,l}_{alpha,beta}\Gamma_{i}_{alpha0,alpha1}S2^{b,m}_{beta0,alpha1}\Gamma_{f}_{beta0,beta1}S3^{a,n}_{beta1,alpha0} 
     *  @param std::vector<GAMMAS_SCATT> &Gammas_i: list of gammas at the source for the contractions
     *  @param std::vector<GAMMAS_SCATT> &Gammas_f: list of gammas at the sink for the contractions
     *  @param PLEGMA_Propagator<Float> &S1: propagator to the sink spin from the source spin index
     *  @param PLEGMA_Propagator<Float> &S2: propagator 2: it will be transposed in the code
     *  @param PLEGMA_Propagator<Float> &S3: propagator 3: won't be transposed
    **/
    void T2( std::vector<GAMMAS_SCATT> &Gammas_i, std::vector<GAMMAS_SCATT> &Gammas_f, PLEGMA_Propagator<Float> &S1, PLEGMA_Propagator<Float> &S2, PLEGMA_Propagator<Float> &S3 );
    void PhiPhi( PLEGMA_Vector<Float> &Phi_0, std::vector<GAMMAS_SCATT> &Gammas,  PLEGMA_Vector<Float> &Phi_1 );

    //manipulation
    //Note that in the case of oet B, W diagram g0 refers not to gamma_i2 but to gamma_f2
    void V3V2reduction( PLEGMA_ScattCorrelator<Float> &srcV3,PLEGMA_ScattCorrelator<Float> &srcV2, int index_abs, bool transp, int g0, bool transp_i1=false, Float* factor=NULL, bool transp_f1=false, bool oet=false );

    void V5V6reduction(PLEGMA_ScattCorrelator<Float> &srcV6, 
	                      std::shared_ptr<Float> &Phi0, 
			      std::vector<PLEGMA_Vector<Float>*> &Phi_1,
			      int input_mom_f2, 
			      int index_abs_V6, int index_abs_V5, 
			      bool transp=false, bool transpgamma_i1=false, bool transgamma_f1=false, 
			      Float *factor=NULL);

    void V3V2reduction_matrix( PLEGMA_ScattCorrelator<Float> &srcV3, PLEGMA_ScattCorrelator<Float> &srcV2, int index_abs, bool transp,  int g0, bool transp_i1=false, Float* factor=NULL, bool transp_f1=false, bool oet=false ); 

    void V5V6reduction_matrix( PLEGMA_ScattCorrelator<Float> &srcV6, 
	                             std::shared_ptr<Float> &Phi0, 
				     std::vector<PLEGMA_Vector<Float>*> &Phi_1,
				     int input_mom_f2, 
				     bool transp=false, bool transpgamma_i1=false, bool transpgamma_f1=false, Float *factor=NULL );


    //manipulation for coherent source implementation

    void absorbTimeslice(PLEGMA_ScattCorrelator<Float> &srcCorr, int global_it, bool forcetozero=false); 

    //manipulation to construct N like diagram from NjN
    void absorbSourceSinkSpinMom(PLEGMA_ScattCorrelator<Float> &srcCorr, int alpha, int beta, int pf1, bool forcetozero=false);

    //manipulation to construct P like diagram from pi j pi
    void absorbGammai2Gammaf2momentumf2(PLEGMA_ScattCorrelator<Float> &srcCorr, int  i_gamma_i2, int i_gamma_f2, int i_pf1, bool forcetozero=false );

    void contractMesonThrp_local(PLEGMA_Vector<Float> &bwdProp,
                                 PLEGMA_Vector<Float> &fwdProp,
                                 std::vector<GAMMAS_SCATT> gammas);



    //initialize_diagrams
    void initialize_diagram( std::vector<GAMMAS_SCATT> &G_f2, std::string name_of_diagram );//L

    void initialize_diagram( std::vector<GAMMAS_SCATT> &G_i2, std::vector<GAMMAS_SCATT> &G_f2, std::string name_of_diagram );//P
    void initialize_diagram( std::vector<GAMMAS_SCATT> &G_i2, std::vector<GAMMAS_SCATT> &G_f2, std::vector<GAMMAS_SCATT> &G_c, std::string name_of_diagram );//P

    void initialize_diagram( std::vector<GAMMAS_SCATT> &eG_i, std::vector<GAMMAS_SCATT> &eG_f,
			     std::vector<GAMMAS_SCATT> &G_i1, std::vector<GAMMAS_SCATT> &G_f1, std::string name_of_diagram );//N,D
    void initialize_diagram( std::vector<GAMMAS_SCATT> &eG_i, std::vector<GAMMAS_SCATT> &eG_f,
                             std::vector<GAMMAS_SCATT> &G_i1, std::vector<GAMMAS_SCATT> &G_f1, std::vector<GAMMAS_SCATT> &G_c, std::string name_of_diagram );//NjN

    void initialize_diagram( std::vector<GAMMAS_SCATT> &eG_i, std::vector<GAMMAS_SCATT> &eG_f,
			     std::vector<GAMMAS_SCATT> &G_i1, std::vector<GAMMAS_SCATT> &G_i2, std::vector<GAMMAS_SCATT> &G_f, std::string isospin, std::string name_of_diagram );//T
    void initialize_diagram( std::vector<GAMMAS_SCATT> &eG_i, std::vector<GAMMAS_SCATT> &eG_f,
			     std::vector<GAMMAS_SCATT> &G_i1, std::vector<GAMMAS_SCATT> &G_i2, std::vector<GAMMAS_SCATT> &G_f1, std::vector<GAMMAS_SCATT> &G_f2, std::string isospin, std::string name_of_diagram, bool oet=false );//B,W,Z,M
    void initialize_diagram( std::vector<GAMMAS_SCATT> &eG_i, std::vector<GAMMAS_SCATT> &eG_f,
                             std::vector<GAMMAS_SCATT> &G_i1, std::vector<GAMMAS_SCATT> &G_i2, std::vector<GAMMAS_SCATT> &G_f1, std::vector<GAMMAS_SCATT> &G_f2, std::vector<GAMMAS_SCATT> &G_c, std::string isospin, std::string name_of_diagram );//BWZM for Npi J Npi


    
    //diagrams
    void B_diagrams( PLEGMA_ScattCorrelator<Float> &srcV3, PLEGMA_ScattCorrelator<Float> &srcV2, int ig_i2, int diagram_index, bool accum=false, bool oet=false );
    void W_diagrams( PLEGMA_ScattCorrelator<Float> &srcV3, PLEGMA_ScattCorrelator<Float> &srcV2, int ig_i2, int diagramm_index, bool accum=false );
    void W_diagrams_oet(PLEGMA_ScattCorrelator<Float> &srcV6, std::shared_ptr<Float> &Phi0, std::vector<PLEGMA_Vector<Float>*> &Phi_1, int input_mom_f2, int diagram_index, bool accum=false);

    void Z_diagrams( std::array<PLEGMA_ScattCorrelator<Float>,4> (&srcV3), std::array<PLEGMA_ScattCorrelator<Float>,4> (&srcV2),int diagramm_index, bool accum=false );
    void Z_diagrams_without_dilution( PLEGMA_ScattCorrelator<Float> &srcV3, PLEGMA_ScattCorrelator<Float> &srcV2, int ig_i2,  int diagramm_index, bool accum=false );
    void Z_diagrams_without_dilution_check(PLEGMA_ScattCorrelator<Float> &srcV3,
                                                PLEGMA_ScattCorrelator<Float> &srcV2, int i_g_i2,
                                                std::string proporder, int diagram_number, bool transp_i1, bool transp_f1, bool accum );

    void M_diagrams( PLEGMA_ScattCorrelator<Float> &CorrNucleon, std::vector<PLEGMA_Vector<Float>*> &Phi_0, std::vector<PLEGMA_Vector<Float>*> &Phi_1, bool accum=false );

    void M_diagrams( PLEGMA_ScattCorrelator<Float> &CorrNucleon, PLEGMA_Vector<Float> &Phi_0, PLEGMA_Vector<Float> &Phi_1, bool accum=false );

    void M_diagrams( PLEGMA_ScattCorrelator<Float> &CorrNucleon, PLEGMA_ScattCorrelator<Float> &CorrPion, Float *data, bool accum=false );

    void LT_diagrams( PLEGMA_ScattCorrelator<Float> &T1reduction, PLEGMA_ScattCorrelator<Float> &T2reduction, PLEGMA_ScattCorrelator<Float> &Loop, bool accum=false );

    void D1ii_diagrams(PLEGMA_ScattCorrelator<Float> &srcV3, PLEGMA_ScattCorrelator<Float> &srcV2, Float * loopcontribution,  const int ig_i2, const int diagram_index, bool accum=false);

    void T_diagrams( PLEGMA_ScattCorrelator<Float> &T1, PLEGMA_ScattCorrelator<Float> &T3, PLEGMA_ScattCorrelator<Float> &T5, int ig_i2, bool accum=false );

    void T_diagrams_oet(PLEGMA_ScattCorrelator<Float> &reductionsVT, std::shared_ptr<Float> &Phi0, int diagramindex, bool accum=false);

    void V24pointSourceReduction( PLEGMA_ScattCorrelator<Float> &reductionsVT, std::shared_ptr<Float> &Phi0, int index_abs, bool transp, bool transpgamma_i1, bool transpgamma_f1, Float* factor);
    void V24pointSourceReduction_matrix( PLEGMA_ScattCorrelator<Float> &reductionsVT,  std::shared_ptr<Float> &Phi0, int index_abs, bool transp, bool transpgamma_i1, bool transpgamma_f1, Float* factor);

    void T_diagrams_piNsink(PLEGMA_ScattCorrelator<Float> &srcV2, PLEGMA_ScattCorrelator<Float> &srcV3,int diagram_index, bool accum=false);

    void Loop_diagrams( PLEGMA_Vector<Float>* &Phi_0, PLEGMA_Vector<Float>* &Phi_1, int i_pi2,bool dn=false, bool accum=false);
    void P_diagrams( std::vector<PLEGMA_Vector<Float>*> &Phi_0, std::vector<PLEGMA_Vector<Float>*> &Phi_1, int i_pi2, bool accum=false );

    void P_diagrams( PLEGMA_Vector<Float> &Phi_0, PLEGMA_Vector<Float> &Phi_1, int i_pi2, bool accum=false);

    void N_diagrams( PLEGMA_ScattCorrelator<Float> &T1, PLEGMA_ScattCorrelator<Float> &T2, bool accum=false );
    void D_diagrams( PLEGMA_ScattCorrelator<Float> &T1, PLEGMA_ScattCorrelator<Float> &T2, bool accum=false );
    void convertTreductiontoDiagram( PLEGMA_ScattCorrelator<Float> &T2, int ig_i2, bool accum=false, bool transp_i1=false, bool transp_f1=false );

    //others
    //
    std::shared_ptr<Float> average_all_time_slices( );
    Float *get_source_time_slice( );
    Float *get_time_slice( int global_time_index );

    void multiply_by_time_slice(std::shared_ptr<Float>&);
    void applyBoundaryConditions( bool antiperiodic, int n_coherent_source=1, int *attract_look_up_table=NULL );
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

  template <int s_free, typename Float>
  void absorb_fromV56( Float dest[N_SPINS*N_COLS*2], Float* src, int alfa ){
    if( s_free == 0){
      for(int s=0; s < N_SPINS; ++s)
        for(int c=0; c < N_COLS; ++c)
          for(int ri=0; ri<2; ++ri)
            dest[(s*N_COLS+c)*2+ri]  = src[((s*N_SPINS+alfa)*N_COLS+c)*2+ri];
    } else {
      for(int s=0; s < N_SPINS; ++s)
        for(int c=0; c < N_COLS; ++c)
          for(int ri=0; ri<2; ++ri)
            dest[(s*N_COLS+c)*2+ri]  = src[((alfa*N_SPINS+s)*N_COLS+c)*2+ri];
    }
  }
}
