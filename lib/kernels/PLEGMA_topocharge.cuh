#pragma once
#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_global.h>
using namespace plegma;

//####################################################################################
//###############                  Clover definition                   ###############
//####################################################################################

//device function that computes the clover term! save a clover extracted from gaugetex (dir1,dir2,sid) into a C matrix
template<typename FloatG>
__device__ void clover( Float2<FloatG> C[N_COLS][N_COLS], gaugeTex<FloatG> &gaugeTex, int dir1, int dir2, int sid ) {

  Float2<FloatG> G1[N_COLS][N_COLS], G2[N_COLS][N_COLS],
    G3[N_COLS][N_COLS], G4[N_COLS][N_COLS], P[N_COLS][N_COLS];

  init_to_zero( C );
  
  // term trace[U^{i}(id) * U^{j}(id+i) * U^{i+}(id+j) * U^{j+}(id)]
  gaugeTex.get(G1,dir1,sid);
  gaugeTex.get<Plus>(G2,dir2,sid,dir1);
  mul_G_G(G3,G1,G2);

  gaugeTex.get<Plus>(G1,dir1,sid,dir2);
  gaugeTex.get(G2,dir2,sid);
      
  mul_Gdag_Gdag(G4,G1,G2);

  mul_G_G( P, G3, G4);
  G_plus_aG( C, P, 1.);
	
  // term trace[U^{i}(id) * U^{j}(id+i) * U^{i+}(id+j) * U^{j+}(id)]
  gaugeTex.get<Minus>(G1,dir1,sid,dir1);
  gaugeTex.get(G2,dir2,sid);
      
  mul_G_G(G3,G1,G2);
      
  gaugeTex.get<MinusPlus>(G1,dir1,sid,dir1,dir2);
  gaugeTex.get<Minus>(G2,dir2,sid,dir1);
      
  mul_Gdag_Gdag(G4,G1,G2);

  mul_G_G( P, G3, G4);
  G_plus_aG( C, P, 1.);

  // term trace[U^{i}(id) * U^{j}(id+i) * U^{i+}(id+j) * U^{j+}(id)]
  gaugeTex.get<MinusMinus>(G1,dir1,sid,dir1,dir2);
  gaugeTex.get<Minus>(G2,dir2,sid,dir2);
     
  mul_G_G(G3,G1,G2); // flops = N_COLS*N_COLS*N_COLS*2
      
  gaugeTex.get<Minus>(G1,dir1,sid,dir1);
  gaugeTex.get<MinusMinus>(G2,dir2,sid,dir1,dir2);
      
  mul_Gdag_Gdag(G4,G1,G2); // flops = N_COLS*N_COLS*N_COLS*2
      
  mul_G_G( P, G3, G4);
  G_plus_aG( C, P, 1.);
  
  // term trace[U^{i}(id) * U^{j}(id+i) * U^{i+}(id+j) * U^{j+}(id)]
  gaugeTex.get<Minus>(G1,dir1,sid,dir2);
  gaugeTex.get<PlusMinus>(G2,dir2,sid,dir1,dir2);
    
  mul_G_G(G3,G1,G2); // flops = N_COLS*N_COLS*N_COLS*2
      
  gaugeTex.get(G1,dir1,sid);
  gaugeTex.get<Minus>(G2,dir2,sid,dir2);
      
  mul_Gdag_Gdag(G4,G1,G2); // flops = N_COLS*N_COLS*N_COLS*2
      
  mul_G_G( P, G3, G4);
  G_plus_aG( C, P, 1.);
  
}

//kernel for filling an Su3field with clovers
template< typename FloatG>
static __global__ void extract_clover_kernel( FloatG *res_dvc_pointer, gaugeTex<FloatG> gaugeTex, int dir0, int dir1) {
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  Float2<FloatG> clov[N_COLS][N_COLS];
  su3_2<FloatG> result(res_dvc_pointer);
  
  if (sid < DGC_localVolume) {
    clover( clov, gaugeTex, dir0, dir1, sid );
    result.set( clov, sid);
  }
}

//function for calling the above kernel
template< typename FloatG>
static void extract_clover( PLEGMA_Su3field<FloatG> &res, PLEGMA_Gauge<FloatG> &U, int dir0, int dir1) {
  gaugeTex<FloatG> tex;
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (HGC_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  
  tex.tex = U.createTexObject();
  extract_clover_kernel<FloatG><<<gridDim,blockDim>>>( res.D_elem(), tex, dir0, dir1);
  U.destroyTexObject(tex.tex);

}

  
//kernel for computing top_charge based on clover definition
template< typename FloatG,  typename Float >
static __global__ void calcTopChClovDef_kernel(gaugeTex<FloatG> gaugeTex, Float *partial_Q) {
  __shared__ Float shared_cache[THREADS_PER_BLOCK];
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  int cacheIndex = threadIdx.x;

  int dir0=3, dir1[3]={0,1,2}, dir2[3]={1,2,0}, dir3[3]={2,0,1};
  Float2<FloatG> clov1[N_COLS][N_COLS], clov2[N_COLS][N_COLS];

  FloatG trace = 0. ;

  if (sid < DGC_localVolume) {
    #pragma unroll
    for(int i=1; i<4; i++) {
      clover( clov1, gaugeTex, dir0, dir1[i], sid );
      clover( clov2, gaugeTex, dir2[i], dir3[i], sid );

      trace += trace_mul_ImG_ImG<FloatG,FloatG>( clov1, clov2 );
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
  gaugeTex.get<Plus>(G2,dir2,sid,dir1);
  
  mul_G_G(G3,G1,G2);
  
  //G4 = U^{dir1+}(id+dir2) * U^{dir2+}(id)
  gaugeTex.get<Plus>(G1,dir1,sid,dir2);
  gaugeTex.get(G2,dir2,sid);
  
  mul_Gdag_Gdag(G4,G1,G2);
  mul_G_G(C,G3,G4);
}


//kernel for extracting an Su3field with plaquette
template< typename FloatG>
static __global__ void extract_plaquette_kernel( FloatG *res_dvc_pointer, gaugeTex<FloatG> gaugeTex, int dir0, int dir1) {
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  Float2<FloatG> plaq[N_COLS][N_COLS];
  su3_2<FloatG> result(res_dvc_pointer);
  
  if (sid < DGC_localVolume) {
    plaquette( plaq, gaugeTex, dir0, dir1, sid );
    result.set( plaq, sid);
  }
}

//function for calling the above kernel
template< typename FloatG>
static void extract_plaquette( PLEGMA_Su3field<FloatG> &res, PLEGMA_Gauge<FloatG> &U, int dir0, int dir1) {
  gaugeTex<FloatG> tex;
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (HGC_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  
  tex.tex = U.createTexObject();
  extract_plaquette_kernel<FloatG><<<gridDim,blockDim>>>( res.D_elem(), tex, dir0, dir1);
  U.destroyTexObject(tex.tex);
}

//old version, could be replaced
template< typename FloatG,  typename Float >
static __global__ void calcTopChPlaqDef_kernel(gaugeTex<FloatG> gaugeTex, Float *partial_Q) {
  __shared__ Float shared_cache[THREADS_PER_BLOCK];
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  int cacheIndex = threadIdx.x;

  if (N_DIMS != 4){
    printf("Is topological charge well defined on an arbitrary number of dim?\n");
  }
  else {
    if (sid < DGC_localVolume) {
      Float2<FloatG> G1[N_COLS][N_COLS], G2[N_COLS][N_COLS],
	G3[N_COLS][N_COLS], G4[N_COLS][N_COLS],
	P1[N_COLS][N_COLS], P2[N_COLS][N_COLS];
      Float trace = 0.;
      int dir0=3, dir1[3]={0,1,2}, dir2[3]={1,2,0}, dir3[3]={2,0,1};
      
      // Loop over xy, xz, xt, yz, yt, zt
#pragma unroll
      for(int i=0; i<3; i++) {
	
	//dir0dir1 
	//G3 = U^{dir0}(id) * U^{dir1}(id+dir0)
	gaugeTex.get(G1,dir0,sid);
	gaugeTex.get<Plus>(G2,dir1[i],sid,dir0);
	  
	mul_G_G(G3,G1,G2);

	//G4 = U^{dir0+}(id+dir1) * U^{dir1+}(id)
	gaugeTex.get<Plus>(G1,dir0,sid,dir1[i]);
	gaugeTex.get(G2,dir1[i],sid);
	  
	mul_Gdag_Gdag(G4,G1,G2);
	mul_G_G(P1,G3,G4);
	  
	//dir2dir3
	//G3 = U^{dir2}(id) * U^{dir3}(id+dir2)
	gaugeTex.get(G1,dir2[i],sid);
	gaugeTex.get<Plus>(G2,dir3[i],sid,dir2[i]);
	  
	mul_G_G(G3,G1,G2);
	  
	//G4 = U^{dir2+}(id+dir3) * U^{dir2+}(id)
	gaugeTex.get<Plus>(G1,dir2[i],sid,dir3[i]);
	gaugeTex.get(G2,dir3[i],sid);
	  
	mul_Gdag_Gdag(G4,G1,G2);
	  
	mul_G_G(P2,G3,G4);
	  
	trace += trace_mul_ImG_ImG<FloatG,FloatG>(P1,P2);
	//trace += real_trace_mul_G_G<Float>(G3,G4);
      } 
    
      shared_cache[cacheIndex] = trace;
    } else {
      shared_cache[cacheIndex] = 0.;
    }
    __syncthreads(); // synchronize threads to be sure that all have written their register trace to share memory
    // for reduction threads per block must be power of 2 ( this is always my case)
    int i = blockDim.x/2;
      
    while (i != 0){
      if(cacheIndex < i)
	shared_cache[cacheIndex] += shared_cache[cacheIndex + i];
      __syncthreads();
      i /= 2;
    }
    
    // now on the first element of the shared memory we have the reduction of block threads
    if(cacheIndex == 0){
      partial_Q[blockIdx.x] = shared_cache[0];   // write result back to global memory  
    }
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
  dim3 gridDim( (HGC_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  Float *h_partial_Q = NULL;
  Float *d_partial_Q = NULL;
  h_partial_Q = (Float*) malloc(gridDim.x * sizeof(Float) );
  if(h_partial_Q == NULL) errorQuda("Error allocate memory for host partial plaq");
  cudaMalloc((void**)&d_partial_Q, gridDim.x * sizeof(Float));

#ifdef TIMING_REPORT
  cudaEvent_t start,stop;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
#endif
  
  switch(charge_def){
  case PLAQUETTE:
    calcTopChPlaqDef_kernel<FloatG,Float><<<gridDim,blockDim>>>( gaugeTex, d_partial_Q );
    break;
  case CLOVER:
    calcTopChClovDef_kernel<FloatG,Float><<<gridDim,blockDim>>>( gaugeTex, d_partial_Q );
    break;
    }

#ifdef TIMING_REPORT
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime,start,stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  printfQuda("Elapsed time for plaquette kernel is %f ms\n",elapsedTime);
#endif

  cudaMemcpy(h_partial_Q, d_partial_Q , gridDim.x * sizeof(Float) , cudaMemcpyDeviceToHost);
  cudaFree(d_partial_Q);
  checkCudaError();

  for(int i = 0 ; i < gridDim.x ; i++)
    Q += h_partial_Q[i];
  free(h_partial_Q);

  MPI_Allreduce(&Q , &globalQ , 1 , MPI_Type(Q) , MPI_SUM , MPI_COMM_WORLD);  
  return globalQ/PI/PI/4; // 8*( 3 indipendent ijkt index order ) /( 32 pi**2)
}


//####################################################################################
//###############                Calc TopoChargeDensity                ###############
//####################################################################################
//
//dir_min
//|
//|
//|
// +<---------------<----------------+
// |                .                ^
// |                .                |
// |                .                |
// |                .                |
// |                .                |
// |                .                |
// |                .                |
// V                .                |
// O---------------->--------------->+____dir_max
//
//Return the rectangle with longest side along the dir_max direction (they can be also negative)  
template<typename Float>
void rectangles( PLEGMA_Su3field<Float> &res, PLEGMA_Su3field<Float> *U[4], int dir_max, int dir_min) {
  int spath[] = {dir_max, dir_max, dir_min, (dir_max+4)%8, (dir_max+4)%8, (dir_min+4)%8};
  std::vector<int> vspath(spath,spath+6);

  res.path(vspath, U);
}

//sum over all 2x1 and 1x2 rectangles; N.B.: the function return the whole product (not only the Im part)  
template<typename Float, typename FloatG>
void extract_improved_clover( PLEGMA_Su3field<Float> &res, PLEGMA_Gauge<FloatG> &U_in, int dir1, int dir2) {
  PLEGMA_Su3field<Float> aux1(BOTH);
  PLEGMA_Su3field<Float> aux2(BOTH);
  PLEGMA_Su3field<Float> *U[4];

  for(int idir = 0; idir < 4 ; idir++){
    U[idir] = new PLEGMA_Su3field<Float>(BOTH);
    U[idir]->absorbDir_device( U_in, idir);
  }

  //longest side along dir1
  rectangles( res, U, dir1, dir2);//++

  res.shift( aux1, dir2 );        //+-
  res += aux1;                    //should be implemented
  
  aux1.shift( aux2, dir1);
  aux2.shift( aux1, dir1);        //--
  res += aux1;                    //should be implemented

  aux1.shift( aux2, dir2+4 );     //-+
  res += aux2;                    //should be implemented

  //longest side along dir2
  rectangles( aux1, U, dir2, dir1+4);//-+
  res += aux1;                    //should be implemented

  aux1.shift( aux2, dir1+4 );     //++
  res += aux2;                    //should be implemented
  
  aux2.shift( aux1, dir2);        
  aux1.shift( aux2, dir2);        //+-
  res += aux2;                    //should be implemented

  aux2.shift( aux1 ,dir1);        //--
  res += aux1;                    //should be implemented
}

/*
template<typename Float, typename FloatG>
static Float calcTopoChargeDensity( PLEGMA_Field<Float> &q, PLEGMA_Gauge<FloatG> &U, TOPO_CHARGE_DEF charge_def ){
  PLEGMA_Su3field<FloatG> Gmn1(BOTH), Gmn2(BOTH);
  FloatG coeff1;
  Float coeff2;
  
  PLEGMA_Field<Float> aux;
  int dir0=3, dir1[3]={0,1,2}, dir2[3]={1,2,0}, dir3[3]={2,0,1};

  q.zero_where(BOTH);

  #pragma unroll
  for(int i=0; i<3; i++){
    switch(charge_def){
    case PLAQUETTE:
      extract_plaquette<FloatG>( Gmn1, U, dir0, dir1[i] );
      extract_plaquette<FloatG>( Gmn2, U, dir2[i], dir3[i] );
      coeff1 = 1.;
      coeff2 = 1./4./PI/PI;
      break;
    case CLOVER:
      extract_clover<FloatG>( Gmn1, U, dir0, dir1[i] ); 
      extract_clover<FloatG>( Gmn2, U, dir2[i], dir3[i] );
      coeff1 = 1./4.;
      coeff2 = 1./4./PI/PI;

      break;
    case IMP_CLOVER:
      extract_improved_clover<FloatG>( Gmn1, U, dir0, dir1[i]);
      extract_improved_clover<FloatG>( imp_clover2, U, dir2[i], dir3[i]);
      coeff1 = 1./8.;
      coeff2 = 1./2./PI/PI;

      break;
    }
    
    //*******to be implemented*******************************
    Gmn1 *= coeff1;
    Gmn2 *= coeff2;
    tr_mul_ImxIm( aux, imp_clover1, imp_clover2);
    q += aux;
    //*******************************************************
  }
  
  q *= coeff2;
  q.unload();
}
*/
