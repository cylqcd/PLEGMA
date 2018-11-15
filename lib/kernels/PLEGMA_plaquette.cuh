#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_kernel_tuner.cuh>
using namespace plegma;

// structure that contains all arguments necessary
//  to run the plaquette kernel
template<typename Float, typename FloatG>
struct ArgsPlaquette{
  gaugeTex<FloatG> gaugeTex;
  Float *partial_plaq;
};

template<typename Float, typename FloatG>
static __global__ void calculatePlaquette_kernel(ArgsPlaquette<Float,FloatG> args) {
  extern __shared__ int ext_shared_cache[];
  Float *shared_cache = (Float*)ext_shared_cache;
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
	args.gaugeTex.get(G1,dir1,sid);
	args.gaugeTex.getPlus(G2,dir2,dir1,sid);
      
	mul_G_G(G3,G1,G2); // flops = N_COLS*N_COLS*N_COLS*2
      
	args.gaugeTex.getPlus(G1,dir1,dir2,sid);
	args.gaugeTex.get(G2,dir2,sid);
      
	mul_Gdag_Gdag(G4,G1,G2); // flops = N_COLS*N_COLS*N_COLS*2
      
	trace += real_trace_mul_G_G<Float>(G3,G4); // flops = N_COLS*N_COLS*(2+1)
      }
    } // tot_flop = (N_DIMS-1)*(N_DIMS)/2 * int_flops
    shared_cache[cacheIndex] = trace;
  } else {
    shared_cache[cacheIndex] = 0.;
  }
  __syncthreads(); // synchronize threads to be sure that all have written their register trace to share memory
  // for reduction threads per block must be power of 2 ( this is always my case)
  int i = blockDim.x/2;
  int r = blockDim.x%2;
  while (i > 0){
    if(cacheIndex < i){
      shared_cache[cacheIndex] += shared_cache[cacheIndex + i];
      if(r==1 && cacheIndex==i-1)
	 shared_cache[cacheIndex] += shared_cache[cacheIndex + i+1];
    }
    __syncthreads();
    r = i%2;
    i /= 2;
  }

  // now on the first element of the shared memory we have the reduction of block threads
  if(cacheIndex == 0 && args.partial_plaq!=NULL)
    args.partial_plaq[blockIdx.x] = shared_cache[0];   // write result back to global memory  
}

template<typename Float, typename FloatG>
static Float calculatePlaquette(gaugeTex<FloatG> gaugeTex){
  Float plaquette = 0.;
  Float globalPlaquette = 0.;
  Float *d_partial_plaq = NULL;
  
  ArgsPlaquette<Float,FloatG> kernel_args;
  kernel_args.gaugeTex = gaugeTex;

  ProfileStruct kernel_ps;
  kernel_ps.flops = N_DIMS*(N_DIMS-1)/2 * N_COLS*N_COLS*(3+N_COLS*4);
  kernel_ps.outBytes = N_COLS*N_COLS*2*4*2*sizeof(Float) ;
  kernel_ps.inpBytes = (N_COLS*N_COLS*4*2 + 1)*sizeof(Float) ;
  kernel_ps.siteBytes = N_COLS*N_COLS*N_DIMS*2*sizeof(Float) ;
  kernel_ps.volume = GK_localVolume ;
  kernel_ps.stride = GK_strideFull;
  kernel_ps.tuneY = false ;
  kernel_ps.sharedMemory = true ;
  kernel_ps.sharedBytesPerThread = sizeof(Float);
  
  PLEGMA_kernel_tuner<ArgsPlaquette<Float,FloatG>> tuner( calculatePlaquette_kernel<Float,FloatG>, &kernel_args, kernel_ps );
  
#ifdef TIMING_REPORT
  cudaEvent_t start,stop;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
#endif

  kernel_args.partial_plaq = NULL;
  tuner.tune();
  int gridDimX = tuner.getGridDimX();
  cudaMalloc((void**)&d_partial_plaq, gridDimX * sizeof(Float));
  kernel_args.partial_plaq = d_partial_plaq;
  tuner.run();
  
#ifdef TIMING_REPORT
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime,start,stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  printfQuda("Elapsed time for plaquette kernel is %f ms\n",elapsedTime);
#endif

  Float *h_partial_plaq = NULL;
  h_partial_plaq = (Float*) malloc(gridDimX * sizeof(Float) );
  if(h_partial_plaq == NULL) errorQuda("Error allocate memory for host partial plaq");
  cudaMemcpy(h_partial_plaq, d_partial_plaq , gridDimX * sizeof(Float) , cudaMemcpyDeviceToHost);
  cudaFree(d_partial_plaq);
  checkCudaError();

  for(int i = 0 ; i < gridDimX ; i++)
    plaquette += h_partial_plaq[i];
  free(h_partial_plaq);

  MPI_Allreduce(&plaquette , &globalPlaquette , 1 , MPI_Type(plaquette) , MPI_SUM , MPI_COMM_WORLD);  
  return globalPlaquette/(GK_totalVolume*N_COLS*6);
}
