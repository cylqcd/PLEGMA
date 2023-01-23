#include <PLEGMA_U1Gauge.h>
#include <kernels/PLEGMA_plaquette.cuh>
#include <kernels/PLEGMA_u1gauge_utils.cuh>

using namespace plegma;

template<typename Float>
PLEGMA_U1Gauge<Float>::PLEGMA_U1Gauge(ALLOCATION_FLAG alloc_flag, GHOST_FLAG ghost_flag): 
  PLEGMA_Field<Float>(alloc_flag, U1GAUGE, ghost_flag){ ; }

template<typename Float>
Float PLEGMA_U1Gauge<Float>::calculatePlaq(){
  this->communicateGhost(-1,DIR_BOTH,FIRST_SIDE);
  auto tex = toTexture<u1gaugeTex>(*this);
  Float plaq = calculatePlaquette<Float,Float,u1gaugeTex<Float>>(*tex);
  if(HGC_verbosity>0) PLEGMA_printf("Calculated plaquette is %f\n",plaq);
  return plaq;
}

template<typename Float>
void PLEGMA_U1Gauge<Float>::constField(int mu, int nu, Float exparg, int xnu_0){
  constFieldU1(toField2<u1gauge2>(*this),mu,nu,exparg,xnu_0);
}

template<typename Float>
void PLEGMA_U1Gauge<Float>::modifyBoundaries(int mu, int nu, Float exparg){
  this->unload();
#ifdef MULTI_GPU
  bool last_node_in_mu = (commCoords(mu) == commDim(mu)-1) ? true : false;
#else
  bool last_node_in_mu = true;
#endif
  if(last_node_in_mu){
    int x[4];
    for(x[3]=0; x[3] < HGC_localL[3]; x[3]++)
      for(x[2]=0; x[2] < HGC_localL[2]; x[2]++)
	for(x[1]=0; x[1] < HGC_localL[1]; x[1]++)
	  for(x[0]=0; x[0] < HGC_localL[0]; x[0]++){
	    if(x[mu] == HGC_localL[mu]-1){
	      size_t idx=((x[3]*HGC_localL[2]+x[2])*HGC_localL[1]+x[1])*HGC_localL[0]+x[0];
	      size_t r=mu*HGC_localVolume+idx;
	      if(nu > 0){
		Float val = exparg*HGC_totalL[mu]*(HGC_procPosition[nu]*HGC_localL[nu] + x[nu]);
		this->h_elem[2*r+0]=cos(val); this->h_elem[2*r+1]=sin(val);
	      }
	      else{
		this->h_elem[2*r+0]=0.; this->h_elem[2*r+1]=0.;
	      }
	    }
	  } 
  }
  this->load();
}

template class PLEGMA_U1Gauge<float>;
template class PLEGMA_U1Gauge<double>;
