#include <PLEGMA_kernel_utils.cuh>
using namespace plegma;

template< typename Float, typename FloatGauge>
__global__ void scale_dir_wise_kernel(FloatGauge* gauge, Float2<Float> scale[N_DIMS]){

  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= c_threads) return;

  gauge2<FloatGauge> G(gauge);
#pragma unroll
  for(int dir = 0; dir < N_DIMS; dir++) {
#pragma unroll
      for(int c=0; c<N_COLS*N_COLS; c++) {
	G.set(dir, c/N_COLS, c%N_COLS, sid,
	      scale[dir]*G.get(dir, c/N_COLS, c%N_COLS, sid));
      }
  }
}


template<typename Float, typename FloatGauge>
static void scale_dir_wise(FloatGauge* gauge, Float* scale){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (GK_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  Float2<Float> *d_scale;
  cudaMalloc((void**) &d_scale, N_DIMS*sizeof(Float2<Float>));
  cudaMemcpy( d_scale, scale, N_DIMS*sizeof(Float2<Float>),cudaMemcpyHostToDevice);
  scale_dir_wise_kernel<Float,FloatGauge><<<gridDim,blockDim>>>(gauge, d_scale);
  cudaFree(d_scale);
  checkCudaError();
}
