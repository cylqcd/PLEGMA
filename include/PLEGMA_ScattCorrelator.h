#pragma once
#include <PLEGMA_Correlator.h>

namespace plegma {
  class momList {
    std::vector<std::array<int,3>> p_i2;
    std::vector<std::array<int,3>> p_f1;
    std::vector<std::array<int,3>> p_f2;
    std::vector<std::array<int,3>> *ps[3]={&p_i2,&p_f1,&p_f2};
  public:
    momList() {;}
    momList( std::vector<int> &mom_list ){
      if(mom_list.size()%9!=0) PLEGMA_error("n x 9 integers expected\n");
      for(int i=0; i<mom_list.size(); i=i+9)
	for(int j=0; j<3; j++){
	  std::array<int,3> p={mom_list[i+j*3],mom_list[i+j*3+1],mom_list[i+j*3+2]};
	  (*ps[j]).push_back(p);
      }
    }
    void add_mom( std::vector<int> &mom ){
      if(mom.size()%9!=0) PLEGMA_error("9 integers expected\n");; return;
      for(int j=0; j<3; j++){
	std::array<int,3> p={mom[j*3],mom[j*3+1],mom[j*3+2]};
	(*ps[j]).push_back(p);
      }
    }
    void add_mom( std::array<int,3> &p1, std::array<int,3> &p2, std::array<int,3> &p3 ){
      p_i2.push_back(p1);
      p_f1.push_back(p2);
      p_f2.push_back(p3);
    }
      
    std::vector<std::array<int,3>> uniq_p(int p_i){
      std::vector<std::array<int,3>> out=(*ps[p_i]);
      std::sort(out.begin(),out.end());
      auto new_end = std::unique(out.begin(),out.end());
      out.resize(new_end-out.begin());
      return out;
    }

    momList extract( std::array<int,3> &mom, int p_i ){
      momList out;
      for(int j=0; j<p_i2.size(); j++)
	if( (*ps[p_i])[j]==mom ) out.add_mom( p_i2[j], p_f1[j], p_f2[j]);
      return out;
    }

    std::vector<int> u_posix( int p_i, std::vector<std::array<int,3>> &moms){
      int aux;
      std::vector<int> res;
      std::vector<std::array<int,3>> uniq_pi=uniq_p(p_i);

      for( auto& mom: moms ){
	auto momf = std::find(uniq_pi.begin(), uniq_pi.end(), mom);
	aux = (momf==uniq_pi.end()) ? -1 : momf-uniq_pi.begin();
	res.push_back(aux);
      }
      return res;
    }
    std::vector<int> u_posix( int p_i ){
      int aux;
      std::vector<int> res;
      std::vector<std::array<int,3>> uniq_pi=uniq_p(p_i);

      for( auto& mom: (*this->ps[p_i]) ){
	auto momf = std::find(uniq_pi.begin(), uniq_pi.end(), mom);
	aux = (momf==uniq_pi.end()) ? -1 : momf-uniq_pi.begin();
	res.push_back(aux);
      }
      return res;
    }

    std::vector<std::array<int,3>> index_map(){
      std::vector<std::array<int,3>> res;
      std::vector<int> aux1= u_posix(0);
      std::vector<int> aux2= u_posix(1);
      std::vector<int> aux3= u_posix(2);
      //std::array<int,3> aux;
      for(int i=0; i<p_i2.size(); i++)
	res.push_back({aux1[i],aux2[i],aux3[i]});
      return res;
    }

    void print(){
      for(int n=0; n<p_i2.size(); n++)
	std::cout<< "p_i2=" << p_i2[n][0] << "_" << p_i2[n][1] << "_" << p_i2[n][2] << "_" <<
	  "p_f1=" << p_f1[n][0] << "_" << p_f1[n][1] << "_" << p_f1[n][2] << "_" <<
	  "p_f2=" << p_f2[n][0] << "_" << p_f2[n][1] << "_" << p_f2[n][2] << "\n";
    }
  };

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
    std::vector<GAMMAS> GList;
    std::vector<std::vector<int>> fixMomList;
    std::string shape_labels;    //     index_struct = "gsssc" (because spin first)
    size_t shape_size;               //     prod(shape)    = n_gammas*4*4*4*3

    void absorb_fromV24_checks( PLEGMA_ScattCorrelator<Float> &srcV2, int alfa, int beta);
    void absorbspinmatrix_fromV24_checks( PLEGMA_ScattCorrelator<Float> &srcV2, int alfa );



  public:
    // these constructors does NOT ALLOCATE the memory PLEGMA_ScattCorrelator here, because
    // the dimension is not provided. It will be allocated when used.
    PLEGMA_ScattCorrelator(CORR_SPACE CorrSpace, int Q2_max);
//      PLEGMA_Correlator<Float>(CorrSpace,Q2_max) { ; }
//
    PLEGMA_ScattCorrelator(CORR_SPACE CorrSpace, std::vector<int> fixMomVec);
//      PLEGMA_Correlator<Float>(CorrSpace,fixMomVec) { ; }
    PLEGMA_ScattCorrelator(CORR_SPACE CorrSpace, std::vector<std::vector<int>> fixMomsVec);

    ~PLEGMA_ScattCorrelator(){;}

    //functions that return values of protected variables
    std::string Shape_labels() const{ return shape_labels; }
    std::vector<std::vector<int>> getFixMomList(){ return fixMomList; }
    std::vector<GAMMAS> getGList(){ return GList; }

    //reductions
    void V2( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS> &Gammas, PLEGMA_Propagator<Float> &S1,  PLEGMA_Propagator<Float> &S2 );
    void V3( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS> &Gammas, PLEGMA_Propagator<Float> &S);
    void V4( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS> &Gammas, PLEGMA_Propagator<Float> &S1,  PLEGMA_Propagator<Float> &S2 );
    void T1( std::vector<GAMMAS> &Gammas_i, std::vector<GAMMAS> &Gammas_f, PLEGMA_Propagator<Float> &S1, PLEGMA_Propagator<Float> &S2, PLEGMA_Propagator<Float> &S3);
    void T2( std::vector<GAMMAS> &Gammas_i, std::vector<GAMMAS> &Gammas_f, PLEGMA_Propagator<Float> &S1, PLEGMA_Propagator<Float> &S2, PLEGMA_Propagator<Float> &S3);

    //manipulation
    template <int s_free>
    void absorb_fromV24( PLEGMA_ScattCorrelator<Float> &srcV2like, int alfa, int beta);
    template <int s_fixed>
    void absorbspinmatrix_fromV24( PLEGMA_ScattCorrelator<Float> &srcV2like, int alfa);

    friend void V3V2reduction(std::vector<GAMMAS> &, std::vector<std::array<int,3>> &, std::vector<std::array<int,3>> &, PLEGMA_ScattCorrelator<Float> &, PLEGMA_ScattCorrelator<Float> &, Float *, int, bool transp=false, int offset=0);

    void B_diagramms(std::vector<GAMMAS> &Gammas_i1, PLEGMA_ScattCorrelator<Float> &srcV3, PLEGMA_ScattCorrelator<Float> &srcV2, int diagramm_index);

    void W_diagramms(std::vector<GAMMAS> &Gammas_i1, PLEGMA_ScattCorrelator<Float> &srcV3, PLEGMA_ScattCorrelator<Float> &srcV2, int diagramm_index);

    void Z_diagramms(std::vector<GAMMAS> &Gammas_i1, std::vector<GAMMAS> &Gammas_i2, std::vector<PLEGMA_ScattCorrelator<Float>> &srcV3, std::vector<PLEGMA_ScattCorrelator<Float>> &srcV2, int diagramm_index);

  };

}



//template functions must be defined here
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
  Float* dest = this->corr;
  Float* src = srcV2.corr;
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
  const unsigned short N_GS1C=n_gammas*N_SPINS*N_COLS;

  size_t VOL_SIZE = srcV2.getVolSize();
  Float* dest = this->corr;
  Float* src = srcV2.corr;
  for(int v=0; v < VOL_SIZE; v++)
    for(int g=0; g < n_gammas; g++)
      for(int s1=0; s1 < N_SPINS; s1++)
	for( int s2=0; s2 < N_SPINS; s2++)
	  for(int c=0; c < N_COLS; c++)
	    for(int ri=0; ri<2; ri++)
	      if( s_fixed == 0){
		dest[(v*N_GS1C+g*N_S2C+s1*N_S1C+s2*N_COLS+c)*2+ri] =
		  src[(v*N_GS3C+g*N_S3C+s1*N_S2C+s2*N_S1C+alfa*N_COLS+c)*2+ri];
	      } else if ( s_fixed == 1 ){
		dest[(v*N_GS1C+g*N_S2C+s1*N_S1C+s2*N_COLS+c)*2+ri] =
		  src[(v*N_GS3C+g*N_S3C+s1*N_S2C+alfa*N_S1C+s2*N_COLS+c)*2+ri];
	      } else {
		dest[(v*N_GS1C+g*N_S1C+s1*N_S1C+s2*N_COLS+c)*2+ri] =
		  src[(v*N_GS3C+g*N_S3C+alfa*N_S2C+s1*N_S1C+s2*N_COLS+c)*2+ri];
		
	      }
}
