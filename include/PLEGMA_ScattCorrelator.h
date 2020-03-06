#pragma once
#include <PLEGMA_Correlator.h>
#include <PLEGMA_gammas.h>

namespace plegma {
  enum VRED {V_2=2,V_3=3,V_4=4};
  enum TRED {T_1=1,T_2=2};  
  // forward declaration
  template<typename Float>  class PLEGMA_Vector;
  template<typename Float>  class PLEGMA_Propagator;
  class momList;
  
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

    //std::vector<std::vector<GAMMAS_SCATT>> GList;
    std::vector<GAMMAS_SCATT> GList;
    std::vector<GAMMAS_SCATT> GList2;
    std::string shape_labels;    //     index_struct = "gsssc" (because spin first)

    
    // std::vector<int> offsets;
    // std::vector<int> ranges;
    
    // void setOffsets( int np=0 ){
    //   int g_count=0;
    //   ranges.clear();
      
    //   for( auto &l : labels ){
    // 	switch(l){
    // 	case("p"): if(np==0) PLEGMA_error( "p detected with np=0\n") else ranges.push_back(np); break;
    // 	case("t"): ranges.push_back( corr_mom_space->DimT() ); break;
    // 	case("m"): ranges.push_back( corr_mom_space->Nmoms() ); break;
    // 	case("g"): ranges.push_back( GList[g_count].size() ); g_count++; break;
    // 	case("s"): ranges.push_back( N_SPINS ); break;
    // 	case("c"): ranges.push_back( N_COLS ); break;
    // 	default: PLEGMA_error( "Label %c not recognized\n", l );
    // 	}	
    //   }

    //   offsets.clear();
    //   for( int ir=1; ir<=ranges.size(); ++ir ){
    // 	int offset = std::accumulate(ranges.begin()+ir, ranges.end(), 2, std::multiplies<int>());
    // 	offsets.push_back(offset);
    //   }
    // }
  
    // Float* corr(  std::initializer_list<int> idx ) const {
    //   if(offsets.size()<idx.size())
    // 	PLEGMA_error("Number of indices (%d) greater than size of correlator\n",idx.size());
      
    //   return H_elem() + std::inner_product( idx.begin(), idx.end(), offsets.begin(), 0.0);
    // }
    
    void absorb_fromV24_checks( PLEGMA_ScattCorrelator<Float> &srcV2, int alfa, int beta);
    void absorbspinmatrix_fromV24_checks( PLEGMA_ScattCorrelator<Float> &srcV2, int alfa);
    void contract_GxV2_checks( PLEGMA_ScattCorrelator<Float> &srcV2);



  public:
    // these constructors does NOT ALLOCATE the memory PLEGMA_ScattCorrelator here, because
    // the dimension is not provided. It will be allocated when used.
    PLEGMA_ScattCorrelator(site source, int Q2_max, int totalT=HGC_totalL[DIM_T]);
//      PLEGMA_Correlator<Float>(CorrSpace,Q2_max) { ; }
//
    PLEGMA_ScattCorrelator(site source, std::vector<int> fixMomVec, int totalT=HGC_totalL[DIM_T]);
//      PLEGMA_Correlator<Float>(CorrSpace,fixMomVec) { ; }
    PLEGMA_ScattCorrelator(site source, std::vector<std::vector<int>> fixMomsVec, int totalT=HGC_totalL[DIM_T]);

    ~PLEGMA_ScattCorrelator(){;}

    //functions that return values of protected variables
    std::string getLabels() const{ return shape_labels; }
    //std::vector<std::vector<GAMMAS_SCATT>> getGList(){ return GList; }
    std::vector<GAMMAS_SCATT> getGList(){ return GList; }
    std::vector<GAMMAS_SCATT> getGList2(){ return GList2; }
    
    //checks
    bool check_reduction( VRED V );
    bool check_reduction( TRED T );

    //reductions
    void V2( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS_SCATT> &Gammas, PLEGMA_Propagator<Float> &S1,  PLEGMA_Propagator<Float> &S2);
    void V3( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS_SCATT> &Gammas, PLEGMA_Propagator<Float> &S);
    void V4( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS_SCATT> &Gammas, PLEGMA_Propagator<Float> &S1,  PLEGMA_Propagator<Float> &S2);
    void T1( std::vector<GAMMAS_SCATT> &Gammas_i, std::vector<GAMMAS_SCATT> &Gammas_f, PLEGMA_Propagator<Float> &S1, PLEGMA_Propagator<Float> &S2, PLEGMA_Propagator<Float> &S3);
    void T2( std::vector<GAMMAS_SCATT> &Gammas_i, std::vector<GAMMAS_SCATT> &Gammas_f, PLEGMA_Propagator<Float> &S1, PLEGMA_Propagator<Float> &S2, PLEGMA_Propagator<Float> &S3);
    void PhiPhi( PLEGMA_Vector<Float> &Phi_0, std::vector<GAMMAS_SCATT> &Gammas,  PLEGMA_Vector<Float> &Phi_1);


    //manipulation
    template <int s_free>
    void absorb_fromV24( PLEGMA_ScattCorrelator<Float> &srcV2like, int alfa, int beta);

    template <int s_fixed>
    void absorbspinmatrix_fromV24( PLEGMA_ScattCorrelator<Float> &srcV2like, int alfa);

    template <int s_fixed>
    void contract_GxV2( PLEGMA_ScattCorrelator<Float> &srcV2like, GAMMAS_SCATT &G, bool transp=false );

    void V3V2reduction(std::vector<GAMMAS_SCATT> &Gammas_i1, std::array<int,3> &indexmap, PLEGMA_ScattCorrelator<Float> &srcV2, Float *dest, int index_abs, bool transp=false, bool transpgamma=false, int n_gammas_i2=1, int g0=0);

    void V3V2reduction_matrix(std::vector<GAMMAS_SCATT> &Gammas_i1, std::array<int,3> &indexmap, PLEGMA_ScattCorrelator<Float> &srcV2, Float *dest, int index_abs, bool transp=false, bool transpgamma=false, int n_gammas_i2=1, int g0=0) ;

    //diagrams
    void B_diagramms(momList &moms, PLEGMA_ScattCorrelator<Float> &srcV3, PLEGMA_ScattCorrelator<Float> &srcV2, std::vector<GAMMAS_SCATT> Gammas_ext_source, std::vector<GAMMAS_SCATT> Gammas_ext_sink, GAMMAS_SCATT G_i2, std::vector<GAMMAS_SCATT> &Gammas_i1, std::string &outfile, int diagram_index);

    void W_diagramms(momList &moms, PLEGMA_ScattCorrelator<Float> &srcV3, PLEGMA_ScattCorrelator<Float> &srcV2, std::vector<GAMMAS_SCATT> Gammas_ext_source, std::vector<GAMMAS_SCATT> Gammas_ext_sink, GAMMAS_SCATT G_i2, std::vector<GAMMAS_SCATT> &Gammas_i1, std::string &outfile, int diagramm_index);

    void Z_diagramms(momList &moms, std::array<PLEGMA_ScattCorrelator<Float>,4> (&srcV3), std::array<PLEGMA_ScattCorrelator<Float>,4>(&srcV2), std::vector<GAMMAS_SCATT> &Gamma_ext_source, std::vector<GAMMAS_SCATT> &Gamma_ext_sink, std::vector<GAMMAS_SCATT> &Gammas_i2, std::vector<GAMMAS_SCATT> &Gammas_i1, std::string &outfile, int diagramm_index);

    void T_diagramms( momList &moms, PLEGMA_ScattCorrelator<Float> &T1, PLEGMA_ScattCorrelator<Float> &T3, PLEGMA_ScattCorrelator<Float> &T5, GAMMAS_SCATT &G_i2, std::vector<GAMMAS_SCATT> &extGammas_i1, std::vector<GAMMAS_SCATT> &extGammas_f, std::string &outfile);

    void M_diagramms( momList &moms, momList &moms_red, PLEGMA_ScattCorrelator<Float> &CorrNucleon, std::vector<GAMMAS_SCATT> &G_i2, std::vector<GAMMAS_SCATT> &G_f2, std::array<PLEGMA_Vector<Float>,4> &Phi_0, std::array<PLEGMA_Vector<Float>,4> &Phi_1, std::string &outfile);

    void D_diagramms( PLEGMA_ScattCorrelator<Float> &T1, PLEGMA_ScattCorrelator<Float> &T2, std::vector<GAMMAS_SCATT> &extG_i, std::vector<GAMMAS_SCATT> &extG_f, std::string &outfile);

    void N_diagramms( PLEGMA_ScattCorrelator<Float> &T1, PLEGMA_ScattCorrelator<Float> &T2, std::vector<GAMMAS_SCATT> &extG_i1, std::vector<GAMMAS_SCATT> &extG_f1, std::string &outfile);

    void P_diagramms( std::vector<int> mom_pi2, std::vector<GAMMAS_SCATT> &G_i2, std::vector<GAMMAS_SCATT> &G_f2, std::array<PLEGMA_Vector<Float>,4> &Phi_0, std::array<PLEGMA_Vector<Float>,4> &Phi_1, std::string &outfile);

  };


}

using namespace plegma;

//template functions must be defined here
template<typename Float>
template <int s_free>
void PLEGMA_ScattCorrelator<Float>::contract_GxV2( PLEGMA_ScattCorrelator<Float> &srcV2, GAMMAS_SCATT &G, bool transp){
  if( s_free<0 || s_free>=3 )
    PLEGMA_error("s_free %d out of range (0, 1 or 2)\n",s_free);
  this->contract_GxV2_checks(srcV2);

  int n_gammas=srcV2.GList.size();
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
}

template<typename Float>
template <int s_free>
void PLEGMA_ScattCorrelator<Float>::absorb_fromV24( PLEGMA_ScattCorrelator<Float> &srcV2,
						    int alfa, int beta){
  this->absorb_fromV24_checks(srcV2, alfa, beta);
  if( s_free<0 || s_free>=3 )
    PLEGMA_error("s_free %d out of range (0, 1 or 2)\n",s_free);
    
  int n_gammas = srcV2.shape[0];

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
      	for(int c=0; c < N_COLS; c++)
	  for(int ri=0; ri<2; ri++)
	    if( s_free == 0){
	      dest[(v*N_GS1C+g*N_S1C+s*N_COLS+c)*2+ri] =
		src[(v*N_GS3C+g*N_S3C+s*N_S2C+alfa*N_S1C+beta*N_COLS+c)*2+ri];
	    } else if ( s_free == 1 ){
	      dest[(v*N_GS1C+g*N_S1C+s*N_COLS+c)*2+ri] =
		src[(v*N_GS3C+g*N_S3C+alfa*N_S2C+s*N_S1C+beta*N_COLS+c)*2+ri];
	    } else {
	      dest[(v*N_GS1C+g*N_S1C+s*N_COLS+c)*2+ri] =
		src[(v*N_GS3C+g*N_S3C+alfa*N_S2C+beta*N_S1C+s*N_COLS+c)*2+ri];
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

  int n_gammas = srcV2.shape[0];

  const unsigned short N_S1C=N_SPINS*N_COLS;
  const unsigned short N_S2C=N_SPINS*N_SPINS*N_COLS;
  const unsigned short N_S3C=N_SPINS*N_SPINS*N_SPINS*N_COLS;
  const unsigned short N_GS3C=n_gammas*N_SPINS*N_SPINS*N_SPINS*N_COLS;
  const unsigned short N_GS2C=n_gammas*N_SPINS*N_SPINS*N_COLS;

  size_t VOL_SIZE = srcV2.getVolSize();
  Float* dest = this->H_elem();
  Float* src = srcV2.H_elem();
  for(int v=0; v < VOL_SIZE; v++)
    for(int g=0; g < n_gammas; g++)
      for(int s1=0; s1 < N_SPINS; s1++)
	for( int s2=0; s2 < N_SPINS; s2++)
	  for(int c=0; c < N_COLS; c++)
	    for(int ri=0; ri<2; ri++)
	      if( s_fixed == 2){
		dest[(v*N_GS2C+g*N_S2C+s1*N_S1C+s2*N_COLS+c)*2+ri] =
		  src[(v*N_GS3C+g*N_S3C+s1*N_S2C+s2*N_S1C+alfa*N_COLS+c)*2+ri];
	      } else if ( s_fixed == 1 ){
		dest[(v*N_GS2C+g*N_S2C+s1*N_S1C+s2*N_COLS+c)*2+ri] =
		  src[(v*N_GS3C+g*N_S3C+s1*N_S2C+alfa*N_S1C+s2*N_COLS+c)*2+ri];
	      } else {
		dest[(v*N_GS2C+g*N_S1C+s1*N_S1C+s2*N_COLS+c)*2+ri] =
		  src[(v*N_GS3C+g*N_S3C+alfa*N_S2C+s1*N_S1C+s2*N_COLS+c)*2+ri];	
	      }
}
