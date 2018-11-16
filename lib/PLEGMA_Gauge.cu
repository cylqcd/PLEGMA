#include <PLEGMA_Gauge.h>
#include <PLEGMA_Su3field.h>
#include <PLEGMA_plaquette.cuh>
#include <PLEGMA_su3field.cuh>
using namespace plegma;

//--------------------------//
// class PLEGMA_Gauge //
//--------------------------//

template<typename Float>
PLEGMA_Gauge<Float>::PLEGMA_Gauge(ALLOCATION_FLAG alloc_flag): 
  PLEGMA_Field<Float>(alloc_flag, GAUGE){ ; }

template<typename Float>
void PLEGMA_Gauge<Float>::packGauge(double **p_gauge){
  
  for(int dir = 0 ; dir < N_DIMS ; dir++)
    for(int iv = 0 ; iv < GK_localVolume ; iv++)
      for(int c1 = 0 ; c1 < N_COLS ; c1++)
	for(int c2 = 0 ; c2 < N_COLS ; c2++)
	  for(int part = 0 ; part < 2 ; part++){
	    PLEGMA_Field<Float>::h_elem[dir*N_COLS*N_COLS*GK_localVolume*2 + 
		       c1*N_COLS*GK_localVolume*2 + 
		       c2*GK_localVolume*2 + 
		       iv*2 + part] = 
	      (Float) p_gauge[dir][iv*N_COLS*N_COLS*2 + 
				   c1*N_COLS*2 + c2*2 + part];
	  }
}

template<typename Float>
void PLEGMA_Gauge<Float>::packGaugeToBackup(void **gauge){
  double **p_gauge = (double**) gauge;
  if(PLEGMA_Field<Float>::h_elem_backup != NULL){
    for(int dir = 0 ; dir < N_DIMS ; dir++)
    for(int iv = 0 ; iv < GK_localVolume ; iv++)
    for(int c1 = 0 ; c1 < N_COLS ; c1++)
    for(int c2 = 0 ; c2 < N_COLS ; c2++)
    for(int part = 0 ; part < 2 ; part++){
      PLEGMA_Field<Float>::h_elem_backup[dir*N_COLS*N_COLS*GK_localVolume*2 + 
			c1*N_COLS*GK_localVolume*2 + 
			c2*GK_localVolume*2 + 
			iv*2 + part] = 
	(Float) p_gauge[dir][iv*N_COLS*N_COLS*2 + 
			     c1*N_COLS*2 + 
			     c2*2 + part];
    }
  }
  else{
    errorQuda("Error you can call this method only if you allocate memory for h_elem_backup");
  }

}

template<typename Float>
void PLEGMA_Gauge<Float>::justDownloadGauge(){
  cudaMemcpy(PLEGMA_Field<Float>::h_elem,PLEGMA_Field<Float>::d_elem,PLEGMA_Field<Float>::bytes_total_length, 
	     cudaMemcpyDeviceToHost);
  checkCudaError();
}

template<typename Float>
void PLEGMA_Gauge<Float>::loadGauge(){
  cudaMemcpy(PLEGMA_Field<Float>::d_elem,PLEGMA_Field<Float>::h_elem,PLEGMA_Field<Float>::bytes_total_length, 
	     cudaMemcpyHostToDevice );
  checkCudaError();
}

template<typename Float>
void PLEGMA_Gauge<Float>::loadGaugeFromBackup(){
  if(PLEGMA_Field<Float>::h_elem_backup != NULL){
    cudaMemcpy(PLEGMA_Field<Float>::d_elem,PLEGMA_Field<Float>::h_elem_backup, PLEGMA_Field<Float>::bytes_total_length, 
	       cudaMemcpyHostToDevice );
    checkCudaError();
  }
  else{
    errorQuda("Error you can call this method only if you allocate memory for h_elem_backup");
  }
}


template<typename Float>
void PLEGMA_Gauge<Float>::calculatePlaq(){
  
  this->ghostToHost();
  this->cpuExchangeGhost();
  this->ghostToDevice();
  
  gaugeTex<Float> tex;
  tex.tex = this->createTexObject();
  printfQuda("Calculated plaquette is %f\n",calculatePlaquette<Float>(tex));
  this->destroyTexObject(tex.tex);
}

template<typename Float>
void PLEGMA_Gauge<Float>::calculatePlaqShifts(){
  PLEGMA_Su3field<Float> res(BOTH);
  PLEGMA_Su3field<Float> tmp(BOTH);
  PLEGMA_Su3field<Float> *u_s[4];
  for(int idir = 0; idir < 4 ; idir++){
    u_s[idir] = new PLEGMA_Su3field<Float>(BOTH);
    u_s[idir]->absorbDir_device(*this,idir);
  }
  
  Float resV=0;
  for(int dir1 = 0; dir1 < 4; dir1++)
    for(int dir2 = dir1+1; dir2 < 4; dir2++){
      int spath[] = {dir1,dir2,4+dir1,4+dir2};
      std::vector<int> vspath(spath,spath+4);
      res.path(vspath, u_s, tmp);
      resV += sumRtraceU<Float,Float>(res);
    }
  printfQuda("Calculated plaquette is %f\n",resV/(GK_totalVolume*N_COLS*6));

  for(int idir = 0; idir < 4 ; idir++)
    delete u_s[idir];

}

template<typename Float>
void PLEGMA_Gauge<Float>::absorbDir_device(PLEGMA_Su3field<Float> &su,int dir){
  cudaMemcpy(this->d_elem + dir*(su.Field_length())*(su.Total_length())*2 , su.D_elem(),
  	     su.Bytes_total(), cudaMemcpyDeviceToDevice);
  checkCudaError();
}

template<typename Float>
void PLEGMA_Gauge<Float>::absorbDir_host(PLEGMA_Su3field<Float> &su,int dir){
  memcpy(this->h_elem + dir*(su.Field_length())*(su.Total_length())*2, su.H_elem(),
	 su.Bytes_total());
}


template<typename Float>
void PLEGMA_Gauge<Float>::stoutSmearing(PLEGMA_Gauge<Float> &uin, int nSmear, double rho, int D3D4){
  if(nSmear < 1){
    cudaMemcpy(this->D_elem(), uin.D_elem(), this->Bytes_total(), cudaMemcpyDeviceToDevice);
    checkCudaError();
    return;
  }
  PLEGMA_Su3field<Float> tmp1(BOTH);
  PLEGMA_Su3field<Float> tmp2(BOTH);

  PLEGMA_Su3field<Float> *u_s1[D3D4];
  PLEGMA_Su3field<Float> *u_s2[D3D4];
  for(int idir = 0; idir < D3D4 ; idir++){
    u_s1[idir] = new PLEGMA_Su3field<Float>(BOTH);
    u_s1[idir]->absorbDir_device(*this,idir);
    u_s2[idir] = new PLEGMA_Su3field<Float>(BOTH);
  }

  for(int i = 0; i < nSmear; i++){
    for(int idir = 0 ; idir < D3D4; idir++){
      u_s2[idir]->staples(u_s1, idir, tmp1, tmp2, rho, D3D4);
      tmp1.UxUdag(*(u_s2[idir]), *(u_s1[idir]));
      tmp2.traceHerExpMap(tmp1);
      u_s2[idir]->UxU(tmp2, *(u_s1[idir]));
    }
    for(int idir = 0 ; idir < D3D4; idir++) u_s1[idir]->exchangeRefs(*(u_s2[idir]));
  }

  for(int idir = 0 ; idir < D3D4; idir++) this->absorbDir_device(*(u_s1[idir]), idir);
  if(D3D4 == 3){
    int offset = 3*(tmp1.Field_length())*(tmp1.Total_length())*2;
    cudaMemcpy(this->D_elem() + offset, uin.D_elem() + offset, tmp1.Bytes_total(), cudaMemcpyDeviceToDevice );
    checkCudaError();
  }
  
  for(int idir = 0; idir < D3D4 ; idir++){
    delete u_s1[idir];
    delete u_s2[idir];
  }
}

template class PLEGMA_Gauge<float>;
template class PLEGMA_Gauge<double>;
