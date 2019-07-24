#include <PLEGMA_Gauge.h>
#include <PLEGMA_Su3field.h>
#include <PLEGMA_plaquette.cuh>
#include <PLEGMA_plaquetteCorners.cuh>
#include <PLEGMA_su3field.cuh>
#include <PLEGMA_gauge_utils.cuh>
#include <PLEGMA_field_utils.cuh>
#include <PLEGMA_io.h>
#include <PLEGMA_topocharge.cuh>
#include <PLEGMA_WFlow.cuh>

using namespace plegma;

//--------------------------//
// class PLEGMA_Gauge //
//--------------------------//

template<typename Float>
PLEGMA_Gauge<Float>::PLEGMA_Gauge(ALLOCATION_FLAG alloc_flag, GHOST_FLAG ghost_flag): 
  PLEGMA_Field<Float>(alloc_flag, GAUGE, ghost_flag){ ; }


template<typename Float>
Float PLEGMA_Gauge<Float>::calculateTopo( TOPO_CHARGE_DEF charge_def ){
  gaugeTex<Float> tex;
  this->communicateGhost(-1,FIRST_CORNER);

  tex.tex = this->createTexObject();
  Float Q = calcTopoCharge<Float>(tex, charge_def);
  if(HGC_verbosity>0) PLEGMA_printf("Calculated topological charge is %.14f\n",Q);
  this->destroyTexObject(tex.tex);
  return Q;
}

template<typename Float>
Float PLEGMA_Gauge<Float>::calculatePlaq(){
  gaugeTex<Float> tex;
  this->communicateGhost(-1,FIRST_SIDE);
  tex.tex = this->createTexObject();
  Float plaq = calculatePlaquette<Float>(tex);
  if(HGC_verbosity>0) PLEGMA_printf("Calculated plaquette is %f\n",plaq);
  this->destroyTexObject(tex.tex);
  return plaq;
}

template<typename Float>
Float PLEGMA_Gauge<Float>::calculatePlaqClovDef(){
  gaugeTex<Float> tex;
  this->communicateGhost(-1,FIRST_CORNER);
  tex.tex = this->createTexObject();
  Float plaq = calcPlaqClovDef<Float,Float>(tex);
  if(HGC_verbosity>0) PLEGMA_printf("Calculated plaquette with clover is %.14f\n",plaq);
  this->destroyTexObject(tex.tex);
  return plaq;
}

template<typename Float>
Float PLEGMA_Gauge<Float>::calculatePlaqShiftDef(){
  Float resV=0;
  PLEGMA_Su3field<Float> Plaq(BOTH);
  PLEGMA_Su3field<Float> Plaq1(BOTH);
  
  for(int dir1 = 0; dir1 < 4; dir1++)
    for(int dir2 = dir1+1; dir2 < 4; dir2++){
      plaquette_s( Plaq, *this, dir1, dir2);
      
      resV += sumRtraceU<Float,Float>( Plaq );
    }
  
  Float plaqShifts = resV/(HGC_totalVolume*N_COLS*6);
  gaugeTex<Float> tex;
  this->communicateGhost(-1,FIRST_SIDE);
  tex.tex = this->createTexObject();
  Float plaqRef = calculatePlaquette<Float>(tex);
  this->destroyTexObject(tex.tex);
  PLEGMA_printf("TEST: Calculated plaquette with shifts is %.14f; diff with reference: %e\n", plaqShifts, plaqShifts-plaqRef);

  return plaqShifts;
}

template<typename Float>
Float PLEGMA_Gauge<Float>::calculatePlaqStaplesDef(){
  
  this->communicateGhost(-1,FIRST_CORNER);

  Float plaq = calcPlaqStaplesDef<Float>( this->D_elem() );

  if(HGC_verbosity>0) PLEGMA_printf("Calculated plaquette using staples is %f\n",plaq);
  return plaq;
}

template<typename Float>
void PLEGMA_Gauge<Float>::calculatePlaqCorners(){
  gaugeTex<Float> tex;
  this->communicateGhost(-1,FIRST_CORNER);
  tex.tex = this->createTexObject();
  Float plaqCorners = calculatePlaquetteCorners<Float>(tex);
  Float plaqRef = calculatePlaquette<Float>(tex);
  PLEGMA_printf("TEST: Calculated plaquette with corners is %f; diff with reference: %e\n",plaqCorners, plaqCorners-plaqRef);
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
  Float plaqShifts = resV/(HGC_totalVolume*N_COLS*6);

  gaugeTex<Float> tex;
  this->communicateGhost(-1,FIRST_SIDE);
  tex.tex = this->createTexObject();
  Float plaqRef = calculatePlaquette<Float>(tex);
  this->destroyTexObject(tex.tex);
  PLEGMA_printf("TEST: Calculated plaquette with shifts is %f; diff with reference: %e\n", plaqShifts, plaqShifts-plaqRef);

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

  PLEGMA_Su3field<Float> *ref;
  
  for(int idir = 0; idir < D3D4 ; idir++){
    u_s1[idir] = new PLEGMA_Su3field<Float>(BOTH);
    u_s1[idir]->absorbDir_device(uin,idir);
    u_s2[idir] = new PLEGMA_Su3field<Float>(BOTH);
  }

  for(int i = 0; i < nSmear; i++){
    for(int idir = 0 ; idir < D3D4; idir++){
      u_s2[idir]->staples(u_s1, idir, tmp1, tmp2, rho, D3D4);
      tmp1.UxUdag(*(u_s2[idir]), *(u_s1[idir]));
      tmp2.traceHerExpMap(tmp1);
      u_s2[idir]->UxU(tmp2, *(u_s1[idir]));
    }
    for(int idir = 0 ; idir < D3D4; idir++){
      ref=u_s2[idir];
      u_s2[idir]=u_s1[idir];
      u_s1[idir]=ref;
    }
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

template<typename Float>
void PLEGMA_Gauge<Float>::scaleDirWise(std::complex<Float> scale[N_DIMS]){
  scale_dir_wise(PLEGMA_Field<Float>::d_elem, (Float*) scale);
  this->communicateGhost();
}

template<typename Float>
void PLEGMA_Gauge<Float>::momPhase(Float phase[N_DIMS],int mom[N_DIMS]){
  std::complex<Float> scale[N_DIMS];
  for(int d=0; d<N_DIMS; d++) {
    Float theta = 2.0*PI*((Float)mom[d])*phase[d]/((Float) HGC_totalL[d]);
    scale[d] = {cos(theta), sin(theta)};
  }
  scaleDirWise(scale);
}

template<typename Float>
void PLEGMA_Gauge<Float>::APEsmearing(PLEGMA_Gauge<Float> &uin, int nSmear, double alpha, int D3D4){
  if(nSmear < 1){
    cudaMemcpy(this->D_elem(), uin.D_elem(), this->Bytes_total(), cudaMemcpyDeviceToDevice);
    checkCudaError();
    return;
  }
  PLEGMA_Su3field<Float> tmp1(BOTH);
  PLEGMA_Su3field<Float> tmp2(BOTH);

  PLEGMA_Su3field<Float> *u_s1[D3D4];
  PLEGMA_Su3field<Float> *u_s2[D3D4];

  PLEGMA_Su3field<Float> *ref;
  
  for(int idir = 0; idir < D3D4 ; idir++){
    u_s1[idir] = new PLEGMA_Su3field<Float>(BOTH);
    u_s1[idir]->absorbDir_device(uin,idir);
    u_s2[idir] = new PLEGMA_Su3field<Float>(BOTH);
  }

  for(int i = 0; i < nSmear; i++){
    for(int idir = 0 ; idir < D3D4; idir++){
      u_s2[idir]->staples(u_s1, idir, tmp1, tmp2, alpha, D3D4);
      xpby(*(u_s2[idir]), *(u_s2[idir]), *(u_s1[idir]), (Float) 1. );
      u_s2[idir]->su3Projection();
    }
    for(int idir = 0 ; idir < D3D4; idir++){
      ref=u_s2[idir];
      u_s2[idir]=u_s1[idir];
      u_s1[idir]=ref;
    }
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

template<typename FloatG>
void PLEGMA_Gauge<FloatG>::GFlow_step( PLEGMA_Gauge<FloatG> &Z, double eps )
{

  //1step RK
  GFlow_substep<FloatG,double>( this->D_elem(), Z.D_elem(), eps/(4.0), 0.0 );// 1/4epsZ_0, W1
  //communicate ghost W1
  this->communicateGhost();
  
  //2step RK
  GFlow_substep<FloatG,double>( this->D_elem(), Z.D_elem(), eps*(8.0/9.0), -(17.0/9.0) );// 8/9epsZ_1-17/36epsZ_0, W2
  //communicate ghost W2
  this->communicateGhost();

  //3step RK
  GFlow_substep<FloatG,double>( this->D_elem(), Z.D_elem(), eps*(3.0/4.0), (-1.0) );// Z_2=8/9epsZ_1-17/36epsZ_0, W3
  //communicate ghost W3
  this->communicateGhost();

}

template<typename Float>
void PLEGMA_Gauge<Float>::applyGradientFlow( PLEGMA_Gauge<Float> &Z, int N, double eps ){
  this->communicateGhost();
  for(int i=0; i<N; i++){
    this->GFlow_step( Z, eps );
  }
  this->unload();
}

template<typename Float>
void PLEGMA_Gauge<Float>::unitarize(){
  unitarize_dev( this->D_elem() );
  this->communicateGhost();
}

template<typename Float>
void PLEGMA_Gauge<Float>::print_fields( int sid ){
  print_fields_k( this->D_elem(), sid );
}

template class PLEGMA_Gauge<float>;
template class PLEGMA_Gauge<double>;
