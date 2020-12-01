#pragma once
namespace plegma {
  class momList {
  private:
    const int NLIST;
  protected:
    std::vector<std::vector<std::vector<int>>> ps;
    std::vector<int> i_tot;  
  public:
    momList( int nlist=0, std::initializer_list<int> li_tot={} ) : NLIST(nlist), i_tot(li_tot), ps( std::vector<std::vector<std::vector<int>>>(NLIST) ) {;}

    momList( int nlist, std::vector<int> &mom_list, std::initializer_list<int> li_tot ) : NLIST(nlist), i_tot(li_tot), ps( std::vector<std::vector<std::vector<int>>>(NLIST) ) {
      read_momList( mom_list );
    }

    momList( int nlist, std::string &input_file, std::initializer_list<int> li_tot ) : NLIST(nlist), i_tot(li_tot), ps( std::vector<std::vector<std::vector<int>>>(NLIST) ) {
      std::ifstream file;
      int tmp;
      std::vector<int> mom_list;
  
      file.open(input_file);

      while(file>>tmp) //reads one string at a time 
	mom_list.push_back(tmp); //add it to data vector 
      file.close();

      read_momList( mom_list );
    }

    momList( int nlist, std::initializer_list< std::vector<std::vector<int>> > list_ps, std::initializer_list<int> li_tot ) : NLIST(nlist), i_tot(li_tot), ps(list_ps){
      assert( list_ps.size() == NLIST );
    }

    void read_momList( std::vector<int> &mom_list ){
      assert( mom_list.size()%(3*NLIST)==0 );
      for(int i=0; i<mom_list.size(); i=i+3*NLIST){
	for(int j=0; j<NLIST; j++){
	  ps[j].push_back( std::vector<int>({mom_list[i+j*3],mom_list[i+j*3+1],mom_list[i+j*3+2]}) );
	}
      }
    }

    int N_list(){ return NLIST; }
    
    std::vector<std::vector<int>> p_tot( ){
      assert(!i_tot.empty());
      std::vector<std::vector<int>> ptot;
    
      for( int i=0; i<this->size(); ++i){
	std::vector<int> paux={0,0,0};

	for( int k=0; k<3; ++k ){
	  for( int j : i_tot ){
	    assert( j<NLIST );
	    paux[k] += ps[j][i][k];
	  }
	}
	ptot.push_back( paux );
      }
      return ptot;
    }
  
    int  size(){ return ps[0].size(); }

    bool empty(){ return ps[0].empty(); }

    void add_mom( std::vector<int> &mom ){
      assert( mom.size() == 3*NLIST );
      std::vector<int> pt={0,0,0};
      for(int j=0; j<NLIST; j++){
	(ps[j]).push_back( std::vector<int>({mom[j*3],mom[j*3+1],mom[j*3+2]}) );
      }
    }

    void add_mom( std::vector<std::vector<int>> &p ){
      for( auto &mom : p )
	assert(mom.size()==3);
      assert( p.size()==NLIST );
    
      for( int j=0; j<NLIST; ++j )
	ps[j].push_back(p[j]);

    }

    std::vector<std::vector<int>> uniq_p(int p_i){
      assert(p_i<=NLIST);
    
      std::vector<std::vector<int>> out = (p_i==NLIST) ? p_tot(): ps[p_i];
      std::sort(out.begin(),out.end());
      auto new_end = std::unique(out.begin(),out.end());
      out.resize(new_end-out.begin());
      return out;
    }

    bool check_eq( int p_i ){
      assert(p_i<NLIST);
      bool res = true;
      std::vector<int> el0 = ps[p_i][0];
    
      for( auto& mom: ps[p_i] )
	if( el0 != mom )
	  res = false;
      return res;
    }

    momList extract( std::vector<int> mom, int p_i ){
      momList out(NLIST);
      out.i_tot = this->i_tot;

      assert( p_i<NLIST );
    
      for(int i=0; i<this->size(); ++i){
	if( this->ps[p_i][i]==mom ){
	  std::vector<std::vector<int>> tmp(NLIST);
	  for(int j=0; j<NLIST; ++j)
	    tmp[j] = ps[j][i];
	  out.add_mom( tmp );
	}
      }
      return out;
    }

    std::vector<int> u_posix( int p_i, std::vector<std::vector<int>> &moms){
      int aux;
      std::vector<int> res;
      std::vector<std::vector<int>> uniq_pi = uniq_p(p_i);

      for( auto& mom: moms ){
	auto momf = std::find(uniq_pi.begin(), uniq_pi.end(), mom);
	aux = (momf==uniq_pi.end()) ? -1 : momf-uniq_pi.begin();
	res.push_back(aux);
      }
      return res;
    }

    std::vector<int> u_posix( int p_i ){
      assert(p_i<NLIST);
      auto &moms = ps[p_i];
    
      return this->u_posix( p_i, moms );
    }

    std::vector<std::vector<int>> index_map(){
      std::vector<std::vector<int>> res;

      std::vector<std::vector<int>> auxs;
      for(int j=0; j<NLIST; j++)
	auxs.push_back( u_posix(j) );

      for(int i=0; i<this->size(); i++){
	std::vector<int> tmp;
	for(int j=0; j<NLIST; j++)
	  tmp.push_back(auxs[j][i]);
	res.push_back(tmp);
      }
    
      return res;
    }
    
    std::vector<std::vector<int>> tolist( std::initializer_list<int> p_i ){
      std::vector<std::vector<int>> out;
      for(int i=0; i<this->size(); ++i){
	out.push_back(std::vector<int>());
	for(int j : p_i){
	  assert( j<NLIST );
	  for(int k=0; k<3; ++k)
	    out.back().push_back( ps[j][i][k] );
	}
      }
      
      return out;
    }

    std::vector<std::vector<int>> tolist(){
      std::vector<std::vector<int>> out;
      for(int i=0; i<this->size(); ++i){
	out.push_back(std::vector<int>());
	for(int j=0; j<NLIST; ++j)
	  for(int k=0; k<3; ++k)
	    out.back().push_back( ps[j][i][k] );
      }
            
      return out;
    }

    std::vector<std::vector<int>> pi1(){
      assert(!i_tot.empty());
    
      if(NLIST==1)
	return ps[0];
    
      //build i_left
      std::vector<int> i_left;
      for(int j=0; j<NLIST; ++j)
	i_left.push_back(j);

      std::vector<int>::iterator pend = i_left.end();
      for( int j : i_tot ){
	assert(j<NLIST);
	pend = std::remove( i_left.begin(), pend, j);
      }
      i_left.resize( pend - i_left.begin() );

      std::vector<std::vector<int>> p_i1 = this->p_tot();

      if(!i_left.empty()){
	for(int i=0; i<this->size(); ++i){
	  for( int k=0; k<3; ++k){
	    for( int j : i_left ){
	      assert(j<NLIST);
	      p_i1[i][k] -= ps[j][i][k];
	    }
	  }
	}
      }
    
      return p_i1;
    }

    std::vector<std::vector<int>> pi(int p_i){
      assert(p_i<NLIST);
      return ps[p_i];
    }

    //0,1,2,...,NLIST-1 for ps[i], NLIST for ptot, -1 for p0
    std::vector<std::string> to_string( std::vector<int> p_i, std::initializer_list<std::string> prefix ){
      assert( prefix.size() == p_i.size() );
      auto p_i1 = pi1();
      auto ptot = p_tot();
      std::vector<std::string> out;
      std::string tmp;
    
      for(int n=0; n<this->size(); ++n){
	int j=0;
	tmp = "";
	for( auto &pre : prefix ){
	  auto &mom = (p_i[j]==-1) ? p_i1 : ((p_i[j]==NLIST) ? ptot : ps[p_i[j]]);
	  tmp += pre + std::to_string(mom[n][0])+"_"+std::to_string(mom[n][1])+"_"+std::to_string(mom[n][2]);
	  if(j!=prefix.size()-1) tmp+="_";
	  j++;
	}
	out.push_back(tmp);
      }
      return out;
    }
  
  };
}

/**
 *
 *  @brief vector(spin x color)  matrix(spin x spin)  vector(spin x color) 
 *          multiplication for piN scattering project resulting in complex
 *          number: V1*gamma*V2
 *  @params Float * V1 pointer to a float array of size 2*N_COLS*N_SPINS
 *  @params Float * V2 pointer to a float array of size 2*N_COLS*N_SPINS
 *  @params GAMMAS_SCATT gamma enumerator specifies the gamma matrix
 *  @params bool transp transp==false then V1(a)Gamma(a,b)V2(b) is returned
 *                      transp==true  then V1(a)Gamma(b,a)V2(b) is returned 
 *  @params Float * Dest pointer to 2 Float number (complex)
 **/
template<typename Float>
__inline__ void V_M_V( Float * V1, Float * V2, GAMMAS_SCATT gamma, bool transp, Float *Dest ){
   *(Dest+0)=0.;
   *(Dest+1)=0.;
   #pragma unroll
   for(int nz_e = 0 ; nz_e < 4 ; nz_e++){  
     int beta0= (!transp) ? gammaInd_scatt[gamma][nz_e][0] : gammaInd_scatt[gamma][nz_e][1];
     int beta1= (!transp) ? gammaInd_scatt[gamma][nz_e][1] : gammaInd_scatt[gamma][nz_e][0];
     #pragma unroll
     for (int nz_c = 0; nz_c < 3; nz_c++) {
       *(Dest+0)+= +V1[2*(beta0*N_COLS+nz_c)+0]*gamma_scatt[gamma][nz_e][0]*V2[2*(beta1*N_COLS+nz_c)+0]
                   -V1[2*(beta0*N_COLS+nz_c)+0]*gamma_scatt[gamma][nz_e][1]*V2[2*(beta1*N_COLS+nz_c)+1]
                   -V1[2*(beta0*N_COLS+nz_c)+1]*gamma_scatt[gamma][nz_e][0]*V2[2*(beta1*N_COLS+nz_c)+1]
                   -V1[2*(beta0*N_COLS+nz_c)+1]*gamma_scatt[gamma][nz_e][1]*V2[2*(beta1*N_COLS+nz_c)+0];
       *(Dest+1)+= -V1[2*(beta0*N_COLS+nz_c)+1]*gamma_scatt[gamma][nz_e][1]*V2[2*(beta1*N_COLS+nz_c)+1]
                   +V1[2*(beta0*N_COLS+nz_c)+1]*gamma_scatt[gamma][nz_e][0]*V2[2*(beta1*N_COLS+nz_c)+0]
                   +V1[2*(beta0*N_COLS+nz_c)+0]*gamma_scatt[gamma][nz_e][1]*V2[2*(beta1*N_COLS+nz_c)+0]
                   +V1[2*(beta0*N_COLS+nz_c)+0]*gamma_scatt[gamma][nz_e][0]*V2[2*(beta1*N_COLS+nz_c)+1];
     }
   }
}

/**
 *  @brief tensor*matrix multiplication  
 *         for piN scattering project returns a color vector
 *  @params Float * V1 pointer to a Float array of size 2*N_COLS*N_SPINS*N_SPINS
 *  @params GAMMAS_SCATT gamma enumerator specifies the gamma matrix
 *  @params bool transp if transp==false Gamma(a,b)*V1(b,a) is returned
 *                      if transp==true  Gamma(a,b)*V1(a,b) is returned
 *  @params Float *Dest pointer to array of Float with size 2*N_COLS
 **/
template<typename Float>
__inline__ void V_TR_MM( Float * V1, GAMMAS_SCATT gamma,bool transp, Float *Dest ){
  #pragma unroll
  for (int nz_c = 0 ; nz_c < 3 ; nz_c++){
    *(Dest+2*nz_c+0) = 0;
    *(Dest+2*nz_c+1) = 0;
  }
  #pragma unroll
  for (int nz_c=0; nz_c < 3 ; nz_c++){
    #pragma unroll
    for(int nz_e_inner = 0 ; nz_e_inner < 4 ; nz_e_inner++){
      int beta0=(!transp) ? gammaInd_scatt[gamma][nz_e_inner][0] : gammaInd_scatt[gamma][nz_e_inner][1];
      int beta1=(!transp) ? gammaInd_scatt[gamma][nz_e_inner][1] : gammaInd_scatt[gamma][nz_e_inner][0];
      *(Dest+2*nz_c+0)+=+gamma_scatt[gamma][nz_e_inner][0]*V1[2*(beta1*N_SPINS*N_COLS+beta0*N_COLS+nz_c)+0]
                        -gamma_scatt[gamma][nz_e_inner][1]*V1[2*(beta1*N_SPINS*N_COLS+beta0*N_COLS+nz_c)+1];
      *(Dest+2*nz_c+1)+=+gamma_scatt[gamma][nz_e_inner][1]*V1[2*(beta1*N_SPINS*N_COLS+beta0*N_COLS+nz_c)+0]
                        +gamma_scatt[gamma][nz_e_inner][0]*V1[2*(beta1*N_SPINS*N_COLS+beta0*N_COLS+nz_c)+1];
    }
  }
  
}


template<typename Float>
__inline__ void M_pe_GNG( Float *dest, const GAMMAS_SCATT Gamma_f, const GAMMAS_SCATT Gamma_i, const Float *source, bool forcezero=false ){
  const int N2=N_SPINS*N_SPINS*2;
  if(forcezero)
    for (int i=0; i < N2; ++i)
      dest[i]=0.;
  for (int n_gamma_f=0; n_gamma_f<4; ++n_gamma_f) {    
    const int alfa =   gammaInd_scatt[Gamma_f][n_gamma_f][0];
    const int alfa0=   gammaInd_scatt[Gamma_f][n_gamma_f][1];
    Float gf[2];
    gf[1]=gamma_scatt[Gamma_f][n_gamma_f][1];
    gf[0]=gamma_scatt[Gamma_f][n_gamma_f][0];
    for (int n_gamma_i=0; n_gamma_i<4; ++n_gamma_i){
      const int beta=    gammaInd_scatt[Gamma_i][n_gamma_i][1];
      const int beta0=   gammaInd_scatt[Gamma_i][n_gamma_i][0];
      Float gi[2];
      gi[1]=gamma_scatt[Gamma_i][n_gamma_i][1];
      gi[0]=gamma_scatt[Gamma_i][n_gamma_i][0];
      dest[(alfa*N_SPINS+beta)*2+0]+=
                +gi[0]*source[(alfa0*N_SPINS+beta0)*2+0]*gf[0]
                -gi[1]*source[(alfa0*N_SPINS+beta0)*2+1]*gf[0]
                -gi[1]*source[(alfa0*N_SPINS+beta0)*2+0]*gf[1]
                -gi[0]*source[(alfa0*N_SPINS+beta0)*2+1]*gf[1];
      dest[(alfa*N_SPINS+beta)*2+1]+=
                -gi[1]*source[(alfa0*N_SPINS+beta0)*2+1]*gf[1]
                +gi[1]*source[(alfa0*N_SPINS+beta0)*2+0]*gf[0]
                +gi[0]*source[(alfa0*N_SPINS+beta0)*2+1]*gf[0]
                +gi[0]*source[(alfa0*N_SPINS+beta0)*2+0]*gf[1];
    }
  }
}

template<typename Float> void x_pe_cy( Float *dest, Float *floatcomplex, Float *temporary, int size );
template<typename Float> void x_e_cx( Float *dest, const Float floatcomplex[2], int size );
template<typename Float> void x_e_sx( Float *dest, const Float floatreal, int size );

GAMMAS_SCATT apply_g5(GAMMAS_SCATT source, LEFTRIGHT LR);
std::vector<GAMMAS_SCATT> apply_gamma5_scatt_gamma( std::vector<GAMMAS_SCATT> &source, LEFTRIGHT LR);


