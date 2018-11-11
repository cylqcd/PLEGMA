#include <PLEGMA_kernel_utils.cuh>
using namespace plegma;

template<typename Float, typename FloatG>
static __global__ void calculatePlaquette_kernel(gaugeTex<FloatG> gaugeTex, Float *partial_plaq) {
  __shared__ Float shared_cache[THREADS_PER_BLOCK];
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  int cacheIndex = threadIdx.x;

  if (sid < c_threads) {
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
	gaugeTex.getPlus(G2,dir2,dir1,sid);
      
	mul_G_G(G3,G1,G2);
      
	gaugeTex.getPlus(G1,dir1,dir2,sid);
	gaugeTex.get(G2,dir2,sid);
      
	mul_Gdag_Gdag(G4,G1,G2);
      
	trace += real_trace_mul_G_G<Float>(G3,G4);
      }
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
  if(cacheIndex == 0)
    partial_plaq[blockIdx.x] = shared_cache[0];   // write result back to global memory  
}

template<typename Float, typename FloatG>
static Float calculatePlaquette(gaugeTex<FloatG> gaugeTex){
  Float plaquette = 0.;
  Float globalPlaquette = 0.;
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (GK_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  Float *h_partial_plaq = NULL;
  Float *d_partial_plaq = NULL;
  h_partial_plaq = (Float*) malloc(gridDim.x * sizeof(Float) );
  if(h_partial_plaq == NULL) errorQuda("Error allocate memory for host partial plaq");
  cudaMalloc((void**)&d_partial_plaq, gridDim.x * sizeof(Float));

#ifdef TIMING_REPORT
  cudaEvent_t start,stop;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
#endif
  calculatePlaquette_kernel<Float,FloatG><<<gridDim,blockDim>>>(gaugeTex, d_partial_plaq);
#ifdef TIMING_REPORT
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime,start,stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  printfQuda("Elapsed time for plaquette kernel is %f ms\n",elapsedTime);
#endif

  cudaMemcpy(h_partial_plaq, d_partial_plaq , gridDim.x * sizeof(Float) , cudaMemcpyDeviceToHost);
  cudaFree(d_partial_plaq);
  checkCudaError();

  for(int i = 0 ; i < gridDim.x ; i++)
    plaquette += h_partial_plaq[i];
  free(h_partial_plaq);

  MPI_Allreduce(&plaquette , &globalPlaquette , 1 , MPI_Type(plaquette) , MPI_SUM , MPI_COMM_WORLD);  
  return globalPlaquette/(GK_totalVolume*N_COLS*6);
}
