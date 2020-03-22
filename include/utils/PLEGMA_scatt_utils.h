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

    momList extract( std::vector<int> &mom, int p_i ){
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
    std::vector<std::string> print( std::vector<int> p_i, std::initializer_list<std::string> prefix ){
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

template<typename Float> void V_M_V( Float * V1, Float * V2, GAMMAS_SCATT gamma, bool transp, Float *Dest);
template<typename Float> void V_TR_MM( Float *V1, GAMMAS_SCATT gamma, bool transp, Float *Dest );
template<typename Float> void x_pe_cy( Float *dest, Float *floatcomplex, Float *temporary, int size );
template<typename Float> void x_e_cx( Float *dest, const Float floatcomplex[2], int size );
template<typename Float> void M_pe_GNG( Float *dest, const GAMMAS_SCATT Gamma_f, const GAMMAS_SCATT Gamma_i, const Float *source, bool forcezero=false) ;
std::vector<GAMMAS_SCATT> apply_gamma5_scatt_gamma( std::vector<GAMMAS_SCATT> &source, LEFTRIGHT LR);

