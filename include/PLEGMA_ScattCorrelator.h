#pragma once
#include <PLEGMA_Correlator.h>
#include <PLEGMA_gammas.h>

namespace plegma {
  class momList {
    std::vector<std::vector<int>> p_i2;
    std::vector<std::vector<int>> p_f1;
    std::vector<std::vector<int>> p_f2;
    std::vector<std::vector<int>> p_tot;
    std::vector<std::vector<int>> *ps[4]={&p_i2,&p_f1,&p_f2,&p_tot};

  public:
    momList() {;}
    
    momList( std::vector<int> &mom_list ){

      if(mom_list.size()%9!=0) PLEGMA_error("n x 9 integers expected\n");
      for(int i=0; i<mom_list.size(); i=i+9){
	std::vector<int> p;
	for(int j=0; j<3; j++){
	  p={mom_list[i+j*3],mom_list[i+j*3+1],mom_list[i+j*3+2]};
	  (*ps[j]).push_back(p);
	}
	int idx=(int)(i/9);
	p = {p_f1[idx][0]+p_f2[idx][0],p_f1[idx][1]+p_f2[idx][1],p_f1[idx][2]+p_f2[idx][2]};
	p_tot.push_back(p);
      }
    }
    
    momList( std::string input_file){
      std::ifstream file;
      int tmp;
      std::vector<int> mom_list;

      file.open(input_file);

      while(file>>tmp) //reads one string at a time 
        mom_list.push_back(tmp); //add it to data vector 
      file.close();

      if(mom_list.size()%9!=0) PLEGMA_error("n x 9 integers expected\n");
      for(int i=0; i<mom_list.size(); i=i+9){
        std::vector<int> p;
	for(int j=0; j< 3; j++){
          p={mom_list[i+j*3],mom_list[i+j*3+1],mom_list[i+j*3+2]};
          (*ps[j]).push_back(p);
        }
	int idx=(int)(i/9);
	p = {p_f1[idx][0]+p_f2[idx][0],p_f1[idx][1]+p_f2[idx][1],p_f1[idx][2]+p_f2[idx][2]};
	p_tot.push_back(p);
      }
    }

    int size(){ return p_i2.size(); }

    void add_mom( std::vector<int> &mom ){
      if(mom.size()%9!=0) PLEGMA_error("9 integers expected\n");
      std::vector<int> p;
      for(int j=0; j<3; j++){
	p={mom[j*3],mom[j*3+1],mom[j*3+2]};
	(*ps[j]).push_back(p);
      }
      int idx=p_i2.size()-1;
      p = {p_f1[idx][0]+p_f2[idx][0],p_f1[idx][1]+p_f2[idx][1],p_f1[idx][2]+p_f2[idx][2]};
      p_tot.push_back(p);
    }
    
    void add_mom( std::vector<int> &p1, std::vector<int> &p2, std::vector<int> &p3 ){
      if(p1.size()!=3||p2.size()!=3||p3.size()!=3) PLEGMA_error("3dim vectors expected\n");
      p_i2.push_back(p1);
      p_f1.push_back(p2);
      p_f2.push_back(p3);
      std::vector<int> p={p2[0]+p3[0],p2[1]+p3[1],p2[2]+p3[2]};
      p_tot.push_back(p);
    }
      
    std::vector<std::vector<int>> uniq_p(int p_i){
      std::vector<std::vector<int>> out=(*ps[p_i]);
      std::sort(out.begin(),out.end());
      auto new_end = std::unique(out.begin(),out.end());
      out.resize(new_end-out.begin());
      return out;
    }

    momList extract( std::vector<int> &mom, int p_i ){
      momList out;
      for(int j=0; j<p_i2.size(); j++)
	if( (*ps[p_i])[j]==mom ) out.add_mom( p_i2[j], p_f1[j], p_f2[j]);
      return out;
    }

    std::vector<int> u_posix( int p_i, std::vector<std::vector<int>> &moms){
      int aux;
      std::vector<int> res;
      std::vector<std::vector<int>> uniq_pi=uniq_p(p_i);

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
      std::vector<std::vector<int>> uniq_pi=uniq_p(p_i);

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
    
    std::vector<std::vector<int>> pi1(){
      std::vector<std::vector<int>> p_i1;
      for(int n=0; n<p_i2.size(); n++){
	std::vector<int> tmp={p_f1[n][0]+p_f2[n][0]-p_i2[n][0],p_f1[n][1]+p_f2[n][1]-p_i2[n][1],p_f1[n][2]+p_f2[n][2]-p_i2[n][2]};
	p_i1.push_back(tmp);
      }
      return p_i1;
    }
    
    std::vector<std::string> print_3pt(){
      std::vector<std::vector<int>> p_tot_u = uniq_p(3);
      std::vector<std::string> out;
      std::string tmp;

      for(int n=0; n < p_tot.size(); n++){
	std::vector<int> p_i1={p_tot_u[n][0]-p_i2[0][0],p_tot_u[n][1]-p_i2[0][1],p_tot_u[n][2]-p_i2[0][2]};
        tmp ="pi1="+std::to_string(p_i1[0])+"_"+std::to_string(p_i1[1])+"_"+std::to_string(p_i1[2])+"_";
	tmp += "pi2="+std::to_string(p_i2[0][0])+"_"+std::to_string(p_i2[0][1])+"_"+std::to_string(p_i2[0][2])+"_";
	tmp += "ptot="+std::to_string(p_tot_u[n][0])+"_"+std::to_string(p_tot_u[n][1])+"_"+std::to_string(p_tot_u[n][2]);
	out.push_back(tmp);
      }
      return out;
    }

    std::vector<std::string> print(){
      std::vector<std::vector<int>> p_i1=pi1();
      std::vector<std::string> out;
      std::string tmp;
      for(int n=0; n<p_i2.size(); n++){
        tmp ="pi1="+std::to_string(p_i1[n][0])+"_"+std::to_string(p_i1[n][1])+"_"+std::to_string(p_i1[n][2])+"_";
	tmp += "pi2="+std::to_string(p_i2[n][0])+"_"+std::to_string(p_i2[n][1])+"_"+std::to_string(p_i2[n][2])+"_";
	tmp += "pf1="+std::to_string(p_f1[n][0])+"_"+std::to_string(p_f1[n][1])+"_"+std::to_string(p_f1[n][2])+"_";
	tmp += "pf2="+std::to_string(p_f2[n][0])+"_"+std::to_string(p_f2[n][1])+"_"+std::to_string(p_f2[n][2]);
	out.push_back(tmp);
      }
      return out;
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
    std::vector<GAMMAS_SCATT> GList;
    std::vector<GAMMAS_SCATT> GList2;
    std::string shape_labels;    //     index_struct = "gsssc" (because spin first)
    size_t shape_size;               //     prod(shape)    = n_gammas*4*4*4*3

    void absorb_fromV24_checks( PLEGMA_ScattCorrelator<Float> &srcV2, int alfa, int beta);
    void absorbspinmatrix_fromV24_checks( PLEGMA_ScattCorrelator<Float> &srcV2, int alfa);
    void contract_GxV2_checks( PLEGMA_ScattCorrelator<Float> &srcV2);



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
    std::vector<GAMMAS_SCATT> getGList(){ return GList; }
    std::vector<GAMMAS_SCATT> getGList2(){ return GList2; }

    //checks
    bool is_V24();
    bool is_V3();

    //reductions
    void V2( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS_SCATT> &Gammas, PLEGMA_Propagator<Float> &S1,  PLEGMA_Propagator<Float> &S2);
    void V3( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS_SCATT> &Gammas, PLEGMA_Propagator<Float> &S);
    void V4( PLEGMA_Vector<Float> &Phi, std::vector<GAMMAS_SCATT> &Gammas, PLEGMA_Propagator<Float> &S1,  PLEGMA_Propagator<Float> &S2);
    void T1( std::vector<GAMMAS_SCATT> &Gammas_i, std::vector<GAMMAS_SCATT> &Gammas_f, PLEGMA_Propagator<Float> &S1, PLEGMA_Propagator<Float> &S2, PLEGMA_Propagator<Float> &S3);
    void T2( std::vector<GAMMAS_SCATT> &Gammas_i, std::vector<GAMMAS_SCATT> &Gammas_f, PLEGMA_Propagator<Float> &S1, PLEGMA_Propagator<Float> &S2, PLEGMA_Propagator<Float> &S3);


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
    void B_diagramms(momList &moms, PLEGMA_ScattCorrelator<Float> &srcV3, PLEGMA_ScattCorrelator<Float> &srcV2, std::vector<GAMMAS_SCATT> Gammas_ext_source, std::vector<GAMMAS_SCATT> Gammas_ext_sink, GAMMAS_SCATT G_i2, std::vector<GAMMAS_SCATT> &Gammas_i1, std::string &outfile);

    void W_diagramms(momList &moms, PLEGMA_ScattCorrelator<Float> &srcV3, PLEGMA_ScattCorrelator<Float> &srcV2, std::vector<GAMMAS_SCATT> Gammas_ext_source, std::vector<GAMMAS_SCATT> Gammas_ext_sink, GAMMAS_SCATT G_i2, std::vector<GAMMAS_SCATT> &Gammas_i1, std::string &outfile, int diagramm_index);

    void Z_diagramms(momList &moms, std::array<PLEGMA_ScattCorrelator<Float>,4> (&srcV3), std::array<PLEGMA_ScattCorrelator<Float>,4>(&srcV2), std::vector<GAMMAS_SCATT> &Gamma_ext_source, std::vector<GAMMAS_SCATT> &Gamma_ext_sink, std::vector<GAMMAS_SCATT> &Gammas_i2, std::vector<GAMMAS_SCATT> &Gammas_i1, std::string &outfile, int diagramm_index);

    void T_diagramms( momList &moms, PLEGMA_ScattCorrelator<Float> &T1, PLEGMA_ScattCorrelator<Float> &T3, PLEGMA_ScattCorrelator<Float> &T5, GAMMAS_SCATT &G_i2, std::vector<GAMMAS_SCATT> &extGammas_i1, std::vector<GAMMAS_SCATT> &extGammas_f, std::string &outfile);


    void D_diagramms( PLEGMA_ScattCorrelator<Float> (&srcT1), PLEGMA_ScattCorrelator<Float> (&srcT2), std::vector<GAMMAS_SCATT> &Gammas_i2, std::vector<GAMMAS_SCATT> &Gammas_f2, std::string &outfile);
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
  Float* dest = this->corr;
  Float* src = srcV2.corr;
  
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
  const unsigned short N_GS2C=n_gammas*N_SPINS*N_SPINS*N_COLS;

  size_t VOL_SIZE = srcV2.getVolSize();
  Float* dest = this->corr;
  Float* src = srcV2.corr;
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
