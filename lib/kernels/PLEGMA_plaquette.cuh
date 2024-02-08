#include "PLEGMA_kernel_utils.cuh"
#include "PLEGMA_kernel_tuner.cuh"
#include <type_traits>
using namespace plegma;

template<typename Float, typename FloatG>
static __global__ void calculatePlaquette_device(u1gaugeTex<FloatG> gTex, Float *partial_plaq) {
  extern __shared__ int ext_shared_cache[];
  Float *shared_cache = (Float*)ext_shared_cache;
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  int cacheIndex = threadIdx.x;
  if (sid < gTex.volume()) {
    Float2<FloatG> G1,G2,G3,G4;
    Float sum = 0.;
    // Loop over xy, xz, xt, yz, yt, zt
    //    int x[N_DIMS]=GET_ID(sid);
    #pragma unroll
    for(int dir1=0; dir1<N_DIMS-1; dir1++) {
      #pragma unroll
      for(int dir2=dir1+1; dir2<N_DIMS; dir2++) {
	gTex.get(G1,dir1,sid); gTex.template get<Plus>(G2,dir2,sid,dir1);
	gTex.template get<Plus>(G3,dir1,sid,dir2); gTex.get(G4,dir2,sid);
	Float2<Float> val = G1*G2*conj(G3)*conj(G4);
	//	if(dir1 == 2 && dir2 == 3)
	  //	  printf("%d %d %d %d %f %f\n",x[0],x[1],x[2],x[3],val.x,val.y);
	sum+=val.x*val.x-val.y*val.y;
      }
    }
    shared_cache[cacheIndex] = sum;
  } else {
    shared_cache[cacheIndex] = 0.;
  }
  reduce(shared_cache, 1);

  // now on the first element of the shared memory we have the reduction of block threads
  if(cacheIndex == 0 && partial_plaq!=NULL)
    partial_plaq[blockIdx.x] = shared_cache[0];   // write result back to global memory
}

template<typename Float, typename FloatG>
static __global__ void calculatePlaquette_device(gaugeTex<FloatG> gTex, Float *partial_plaq) {
  extern __shared__ int ext_shared_cache[];
  Float *shared_cache = (Float*)ext_shared_cache;
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  int cacheIndex = threadIdx.x;
  
  if (sid < gTex.volume()) {
    Float2<FloatG> G1[N_COLS][N_COLS], G2[N_COLS][N_COLS],
      G3[N_COLS][N_COLS], G4[N_COLS][N_COLS];    
    Float trace = 0.;

    // Loop over xy, xz, xt, yz, yt, zt
    #pragma unroll
    for(int dir1=0; dir1<N_DIMS-1; dir1++) {
      #pragma unroll
      for(int dir2=dir1+1; dir2<N_DIMS; dir2++) {
	// term trace[U^{i}(id) * U^{j}(id+i) * U^{i+}(id+j) * U^{j+}(id)]
	gTex.get(G1,dir1,sid);
	gTex.template get<Plus>(G2,dir2,sid,dir1);
      
	mul_G_G(G3,G1,G2); // flops = N_COLS*N_COLS*N_COLS*2
      
	gTex.template get<Plus>(G1,dir1,sid,dir2);
	gTex.get(G2,dir2,sid);
      
	mul_Gdag_Gdag(G4,G1,G2); // flops = N_COLS*N_COLS*N_COLS*2
      
	trace += real_trace_mul_G_G<Float>(G3,G4); // flops = N_COLS*N_COLS*(2+1)
      }
    } // tot_flop = (N_DIMS-1)*(N_DIMS)/2 * int_flops
    shared_cache[cacheIndex] = trace;
  } else {
    shared_cache[cacheIndex] = 0.;
  }
  reduce(shared_cache, 1);

  // now on the first element of the shared memory we have the reduction of block threads
  if(cacheIndex == 0 && partial_plaq!=NULL)
    partial_plaq[blockIdx.x] = shared_cache[0];   // write result back to global memory
}

template<typename Float, typename FloatG, typename TG>
static void calculatePlaquette_host(ProfileStruct& ps, TG gTex, Float& plaquette){

  Float *d_partial_plaq = NULL;
  int gridDimX = ps.tp.grid.x;
  d_partial_plaq=(Float*)device_malloc( gridDimX * sizeof(Float));
  calculatePlaquette_device<<<ps.tp.grid,ps.tp.block,ps.tp.shared_bytes>>>(gTex, d_partial_plaq);

  Float *h_partial_plaq = NULL;
  hostMalloc(h_partial_plaq, gridDimX * sizeof(Float) );
  if(h_partial_plaq == NULL) PLEGMA_error("Error allocate memory for host partial plaq");
  qudaMemcpy(h_partial_plaq, d_partial_plaq , gridDimX * sizeof(Float) , qudaMemcpyDeviceToHost);
  device_free(d_partial_plaq);
  checkQudaError();

  plaquette = 0.;
  for(int i = 0 ; i < gridDimX ; i++)
    plaquette += h_partial_plaq[i];
  hostFree(h_partial_plaq, gridDimX * sizeof(Float) );
}

template<typename Float, typename FloatG, typename TG>
static Float calculatePlaquette(TG gTex){

  assert(gTex.is4D); // TODO: For 3D we should not compute the plaquette in T
  PLEGMA_printf("gTex.volume() %d\n", gTex.volume());
  fflush(stdout);
  ProfileStruct ps(gTex.volume(),sizeof(Float));
  PLEGMA_printf("ps.tp.grid.x %d\n", ps.tp.grid.x);
  fflush(stdout);
  Float plaquette;
  std::string nameK;
  int normC;
  if(std::is_same<TG, gaugeTex<FloatG>>::value){
    nameK="calculatePlaquette";
    normC=N_COLS;
  }
  else if(std::is_same<TG, u1gaugeTex<FloatG>>::value){
    nameK="calculateU1Plaquette";
    normC=1;
  }
  else assert(false);
  PLEGMA_printf("Before tuneAndRun\n");
  fflush(stdout);
  tuneAndRun(ps, nameK, calculatePlaquette_host<Float,FloatG,TG>, ps, gTex, plaquette);

  Float globalPlaquette = 0.;
  MPI_Allreduce(&plaquette , &globalPlaquette , 1 , MPI_Type(plaquette) , MPI_SUM , HGC_fullComm);  
  return globalPlaquette/(HGC_totalVolume*normC*6);
}
