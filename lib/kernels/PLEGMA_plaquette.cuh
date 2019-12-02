#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_kernel_tuner.cuh>
using namespace plegma;

template<typename Float, typename FloatG>
static __global__ void calculatePlaquette_device(gaugeTex<FloatG> gaugeTex, Float *partial_plaq) {
  extern __shared__ int ext_shared_cache[];
  Float *shared_cache = (Float*)ext_shared_cache;
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  int cacheIndex = threadIdx.x;
  
  if (sid < DGC_localVolume) {
    Float2<FloatG> G1[N_COLS][N_COLS], G2[N_COLS][N_COLS],
      G3[N_COLS][N_COLS], G4[N_COLS][N_COLS];    
    Float trace = 0.;

    // Loop over xy, xz, xt, yz, yt, zt
    #pragma unroll
    for(int dir1=0; dir1<N_DIMS-1; dir1++) {
      #pragma unroll
      for(int dir2=dir1+1; dir2<N_DIMS; dir2++) {
	// term trace[U^{i}(id) * U^{j}(id+i) * U^{i+}(id+j) * U^{j+}(id)]
	gaugeTex.get(G1,dir1,sid);
	gaugeTex.get<Plus>(G2,dir2,sid,dir1);
      
	mul_G_G(G3,G1,G2); // flops = N_COLS*N_COLS*N_COLS*2
      
	gaugeTex.get<Plus>(G1,dir1,sid,dir2);
	gaugeTex.get(G2,dir2,sid);
      
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

template<typename Float, typename FloatG>
static void calculatePlaquette_host(ProfileStruct& ps, gaugeTex<FloatG> gaugeTex, Float& plaquette){

  Float *d_partial_plaq = NULL;
  int gridDimX = ps.tp.grid.x;
  cudaMalloc((void**)&d_partial_plaq, gridDimX * sizeof(Float));
  calculatePlaquette_device<<<ps.tp.grid,ps.tp.block,ps.tp.shared_bytes>>>(gaugeTex, d_partial_plaq);

  Float *h_partial_plaq = NULL;
  hostMalloc(h_partial_plaq, gridDimX * sizeof(Float) );
  if(h_partial_plaq == NULL) PLEGMA_error("Error allocate memory for host partial plaq");
  cudaMemcpy(h_partial_plaq, d_partial_plaq , gridDimX * sizeof(Float) , cudaMemcpyDeviceToHost);
  cudaFree(d_partial_plaq);
  checkCudaError();

  plaquette = 0.;
  for(int i = 0 ; i < gridDimX ; i++)
    plaquette += h_partial_plaq[i];
  hostFree(h_partial_plaq, gridDimX * sizeof(Float) );
}

template<typename Float, typename FloatG>
static Float calculatePlaquette(gaugeTex<FloatG> gaugeTex){
  
  ProfileStruct ps(HGC_localVolume,sizeof(Float));
  Float plaquette;
  tuneAndRun(ps, "calculatePlaquette", calculatePlaquette_host<Float,FloatG>, ps, gaugeTex, plaquette);

  Float globalPlaquette = 0.;
  MPI_Allreduce(&plaquette , &globalPlaquette , 1 , MPI_Type(plaquette) , MPI_SUM , HGC_fullComm);  
  return globalPlaquette/(HGC_totalVolume*N_COLS*6);
}
