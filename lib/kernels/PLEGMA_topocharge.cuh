#pragma once
#include "PLEGMA_kernel_utils.cuh"
#include <PLEGMA_global.h>
using namespace plegma;

//####################################################################################
//###############                  Clover definition                   ###############
//####################################################################################

//device function that computes the clover term! save a clover extracted from gaugetex (dir1,dir2,sid) into a C matrix
//COMMENTS: check with clover-definded plaquette OK!
template<typename FloatG>
__device__ void clover_term( Float2<FloatG> C[N_COLS][N_COLS], gaugeTex<FloatG> &gaugeTex, int dir1, int dir2, int sid ) {

  Float2<FloatG> G1[N_COLS][N_COLS], G2[N_COLS][N_COLS],
    G3[N_COLS][N_COLS], G4[N_COLS][N_COLS], P[N_COLS][N_COLS];

  init_to_zero( C );//C=0
  
  // term trace[U^{i}(id) * U^{j}(id+i) * U^{i+}(id+j) * U^{j+}(id)]
  gaugeTex.get(G1,dir1,sid);
  gaugeTex.template get<Plus>(G2,dir2,sid,dir1);

  mul_G_G(G3,G1,G2);

  gaugeTex.template get<Plus>(G1,dir1,sid,dir2);
  gaugeTex.get(G2,dir2,sid);
      
  mul_Gdag_Gdag(G4,G1,G2);

  mul_G_G( P, G3, G4);
  G_plus_aG( C, P, 1.);//C=C+P
	
  // term trace[U^{i}(id) * U^{j}(id+i) * U^{i+}(id+j) * U^{j+}(id)]
  gaugeTex.get(G1,dir2,sid);
  gaugeTex.template get<MinusPlus>(G2,dir1,sid,dir1,dir2);
      
  mul_G_Gdag(G3,G1,G2);
      
  gaugeTex.template get<Minus>(G1,dir2,sid,dir1);
  gaugeTex.template get<Minus>(G2,dir1,sid,dir1);
      
  mul_Gdag_G(G4,G1,G2);

  mul_G_G( P, G3, G4);
  G_plus_aG( C, P, 1.);

  // term trace[U^{i}(id) * U^{j}(id+i) * U^{i+}(id+j) * U^{j+}(id)]
  gaugeTex.template get<Minus>(G1,dir1,sid,dir1);
  gaugeTex.template get<MinusMinus>(G2,dir2,sid,dir1,dir2);

  mul_Gdag_Gdag(G3,G1,G2); // flops = N_COLS*N_COLS*N_COLS*2
  
  gaugeTex.template get<MinusMinus>(G1,dir1,sid,dir1,dir2);
  gaugeTex.template get<Minus>(G2,dir2,sid,dir2);
      
  mul_G_G(G4,G1,G2); // flops = N_COLS*N_COLS*N_COLS*2
      
  mul_G_G( P, G3, G4);
  G_plus_aG( C, P, 1.);
  
  // term trace[U^{i}(id) * U^{j}(id+i) * U^{i+}(id+j) * U^{j+}(id)]
  gaugeTex.template get<Minus>(G1,dir2,sid,dir2);
  gaugeTex.template get<Minus>(G2,dir1,sid,dir2);
    
  mul_Gdag_G(G3,G1,G2); // flops = N_COLS*N_COLS*N_COLS*2

  gaugeTex.template get<PlusMinus>(G1,dir2,sid,dir1,dir2);
  gaugeTex.get(G2,dir1,sid);
      
  mul_G_Gdag(G4,G1,G2); // flops = N_COLS*N_COLS*N_COLS*2
      
  mul_G_G( P, G3, G4);
  G_plus_aG( C, P, 1.);
  
}


//kernel for computing top_charge based on clover definition
template< typename FloatG,  typename Float >
static __global__ void calcTopChClovDef_kernel(gaugeTex<FloatG> gaugeTex, Float *partial_Q) {
  __shared__ Float shared_cache[THREADS_PER_BLOCK];
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  int cacheIndex = threadIdx.x;

  if (sid < gaugeTex.volume()) {
    int dir0[3]={3,3,3}, dir1[3]={0,1,2}, dir2[3]={1,2,0}, dir3[3]={2,0,1};
    Float2<FloatG> clov1[N_COLS][N_COLS], clov2[N_COLS][N_COLS];
   
    FloatG trace = 0. ;
    FloatG tr_aux = 0. ;
    #pragma unroll
    for(int i=0; i<3; i++) {
      clover_term( clov1, gaugeTex, dir0[i], dir1[i], sid );
      clover_term( clov2, gaugeTex, dir2[i], dir3[i], sid );

      tr_aux = trace_mul_ImG_ImG<FloatG,FloatG>( clov1, clov2 );
      
      trace += tr_aux;
    }
    shared_cache[cacheIndex] = trace/16.; //Because the clover is defined as 1/4 Im( clover_path )
  }
  else {
    shared_cache[cacheIndex] = 0.;
  }

  reduce(shared_cache, 1);

  if(cacheIndex == 0 && partial_Q!=NULL){
    partial_Q[blockIdx.x] = shared_cache[0];   // write result back to global memory
  }
}


//####################################################################################
//###############                 Plaquette definition                 ###############
//####################################################################################

template<typename FloatG>
__device__ void plaquette( Float2<FloatG> C[N_COLS][N_COLS], gaugeTex<FloatG> &gaugeTex, int dir1, int dir2, int sid ) {

  Float2<FloatG> G1[N_COLS][N_COLS], G2[N_COLS][N_COLS],
    G3[N_COLS][N_COLS], G4[N_COLS][N_COLS];

  //G3 = U^{dir1}(id) * U^{dir2}(id+dir1)
  gaugeTex.get(G1,dir1,sid);
  gaugeTex.template get<Plus>(G2,dir2,sid,dir1);
  
  mul_G_G(G3,G1,G2);
  
  //G4 = U^{dir1+}(id+dir2) * U^{dir2+}(id)
  gaugeTex.template get<Plus>(G1,dir1,sid,dir2);
  gaugeTex.get(G2,dir2,sid);
  
  mul_Gdag_Gdag(G4,G1,G2);
  mul_G_G(C,G3,G4);
}


//kernel for computing top_charge based on plaquette definition
template< typename FloatG,  typename Float >
static __global__ void calcTopChPlaqDef_kernel(gaugeTex<FloatG> gaugeTex, Float *partial_Q) {
  __shared__ Float shared_cache[THREADS_PER_BLOCK];
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  int cacheIndex = threadIdx.x;

  int dir0=3, dir1[3]={0,1,2}, dir2[3]={1,2,0}, dir3[3]={2,0,1};
  Float2<FloatG> plaq1[N_COLS][N_COLS], plaq2[N_COLS][N_COLS];

  FloatG trace = 0. ;

  if (sid < gaugeTex.volume()) {
    #pragma unroll
    for(int i=0; i<3; i++) {
      plaquette( plaq1, gaugeTex, dir0, dir1[i], sid );
      plaquette( plaq2, gaugeTex, dir2[i], dir3[i], sid );

      trace += trace_mul_ImG_ImG<FloatG,FloatG>( plaq1, plaq2 );
    }
    shared_cache[cacheIndex] = trace; //Plaquette is Im( plaquette_path )
  }
  else {
    shared_cache[cacheIndex] = 0.;
  }

  reduce(shared_cache, 1);

  if(cacheIndex == 0 && partial_Q!=NULL){
    partial_Q[blockIdx.x] = shared_cache[0];   // write result back to global memory  
  }
}


//####################################################################################
//###############                 Calculate TopoCharge                 ###############
//####################################################################################

template<typename Float, typename FloatG>
static Float calcTopoCharge(gaugeTex<FloatG> gaugeTex, TOPO_CHARGE_DEF charge_def ){
  Float Q = 0.;
  Float globalQ = 0.;
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (gaugeTex.volume() + blockDim.x -1)/blockDim.x , 1 , 1);
  Float *h_partial_Q = NULL;
  Float *d_partial_Q = NULL;
  h_partial_Q = (Float*) malloc(gridDim.x * sizeof(Float) );
  if(h_partial_Q == NULL) errorQuda("Error allocate memory for host partial plaq");
  d_partial_Q=(Float*)device_malloc(gridDim.x * sizeof(Float));

  switch(charge_def){
  case PLAQUETTE:
    calcTopChPlaqDef_kernel<FloatG,Float><<<gridDim,blockDim>>>( gaugeTex, d_partial_Q );
    break;
  case CLOVER:
    calcTopChClovDef_kernel<FloatG,Float><<<gridDim,blockDim>>>( gaugeTex, d_partial_Q );
    break;
    }
  checkQudaError();

  qudaMemcpy(h_partial_Q, d_partial_Q , gridDim.x * sizeof(Float) , qudaMemcpyDeviceToHost);
  device_free(d_partial_Q);
  checkQudaError();

  for(int i = 0 ; i < gridDim.x ; i++){
    Q += h_partial_Q[i];
  }
  free(h_partial_Q);

  MPI_Allreduce(&Q , &globalQ , 1 , MPI_Type(Q) , MPI_SUM , HGC.fullComm);  
  return globalQ/PI/PI/4.; // 8*( 3 indipendent ijkt index order ) /( 32 pi**2)
}


//kernel for computing the mean plaquette based on clover definition (for xchecks)
template< typename FloatG,  typename Float >
static __global__ void calcPlaqClovDef_kernel(gaugeTex<FloatG> gaugeTex, Float *partial_plaq) {
  __shared__ Float shared_cache[THREADS_PER_BLOCK];
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  int cacheIndex = threadIdx.x;

  if (sid < gaugeTex.volume()) {
    Float2<FloatG> clov_tmp[N_COLS][N_COLS];
    Float trace = 0. ;

    #pragma unroll
    for(int dir1=0; dir1<N_DIMS-1; dir1++) {
      #pragma unroll
      for(int dir2=dir1+1; dir2<N_DIMS; dir2++) {
	clover_term( clov_tmp, gaugeTex, dir1, dir2, sid);
	trace += real_trace<Float,FloatG>( clov_tmp );
      }
    }
    shared_cache[cacheIndex] = trace/4.;
  }
  else {
    shared_cache[cacheIndex] = 0.;
  }
  reduce(shared_cache, 1);

  if(cacheIndex == 0 && partial_plaq!=NULL){
    partial_plaq[blockIdx.x] = shared_cache[0];   // write result back to global memory  
  }

}

template<typename Float, typename FloatG>
static Float calcPlaqClovDef(gaugeTex<FloatG> gaugeTex){
  Float Plaq = 0.;
  Float globalPlaq = 0.;
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (gaugeTex.volume() + blockDim.x -1)/blockDim.x , 1 , 1);
  Float *h_partial_Plaq = NULL;
  Float *d_partial_Plaq = NULL;
  h_partial_Plaq = (Float*) malloc(gridDim.x * sizeof(Float) );
  if(h_partial_Plaq == NULL) errorQuda("Error allocate memory for host partial plaq");
  d_partial_Plaq=(Float*)device_malloc(gridDim.x * sizeof(Float));

  calcPlaqClovDef_kernel<FloatG,Float><<<gridDim,blockDim>>>( gaugeTex, d_partial_Plaq );

  qudaMemcpy(h_partial_Plaq, d_partial_Plaq , gridDim.x * sizeof(Float) , qudaMemcpyDeviceToHost);
  device_free(d_partial_Plaq);
  checkQudaError();

  for(int i = 0 ; i < gridDim.x ; i++)
    Plaq += h_partial_Plaq[i];
  free(h_partial_Plaq);

  MPI_Allreduce(&Plaq , &globalPlaq , 1 , MPI_Type(Plaq) , MPI_SUM , HGC.fullComm);  
  return globalPlaq/(HGC.totalVolume*N_COLS*6); // 6*N_sites Plaquettes(+ 3 colors to normalize the trace )
}
