#pragma once

template<typename Float> void V_M_V( Float * V1, Float * V2, GAMMAS_SCATT gamma, bool transp, Float *Dest);
template<typename Float> void V_TR_MM( Float *V1, GAMMAS_SCATT gamma, bool transp, Float *Dest );
template<typename Float> void x_pe_cy( Float *dest, Float *floatcomplex, Float *temporary, int size );
template<typename Float> void x_e_cx( Float *dest, const Float floatcomplex[2], int size );
template<typename Float> void M_e_GNG( Float *dest, const GAMMAS_SCATT Gamma_f, const GAMMAS_SCATT Gamma_i, const Float *source) ;
std::vector<GAMMAS_SCATT> apply_gamma5_scatt_gamma( std::vector<GAMMAS_SCATT> &source, LEFTRIGHT LR);

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

    bool check_eq( int p_i ){
      bool res = true;
      std::vector<int> el0 = (*ps[p_i])[0];
      
      for( auto& mom: *(this->ps[p_i]) )
	if( el0 != mom )
	  res = false;
      return res;
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

    std::vector<std::vector<int>> pi(int p_i){
      std::vector<std::vector<int>> pi=*(this->ps[p_i]);
      return pi;
    }

    std::vector<std::string> print_3pt(bool piNsink){
      std::vector<std::vector<int>> p_tot_u = uniq_p(3);
      std::vector<std::string> out;
      std::string tmp;

      for(int n=0; n < p_tot_u.size(); n++){
	std::vector<int> p_i1={p_tot_u[n][0]-p_i2[0][0],p_tot_u[n][1]-p_i2[0][1],p_tot_u[n][2]-p_i2[0][2]};
        if (piNsink){
          tmp ="pf1="+std::to_string(p_i1[0])+"_"+std::to_string(p_i1[1])+"_"+std::to_string(p_i1[2])+"_";
          tmp += "pf2="+std::to_string(p_i2[0][0])+"_"+std::to_string(p_i2[0][1])+"_"+std::to_string(p_i2[0][2])+"_";
        }
        else{
          tmp ="pi1="+std::to_string(p_i1[0])+"_"+std::to_string(p_i1[1])+"_"+std::to_string(p_i1[2])+"_";
	  tmp += "pi2="+std::to_string(p_i2[0][0])+"_"+std::to_string(p_i2[0][1])+"_"+std::to_string(p_i2[0][2])+"_";
        }
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
}
