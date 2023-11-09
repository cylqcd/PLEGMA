#include <PLEGMA_kernel_utils.cuh>
using namespace plegma;

template< typename Float, typename FloatGauge>
__global__ void scale_dir_wise_kernel(gauge2<FloatGauge> gauge, Float2<Float> scale[N_DIMS]){

  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= gauge.volume()) return;

#pragma unroll
  for(int dir = 0; dir < N_DIMS; dir++) {
#pragma unroll
      for(int c=0; c<N_COLS*N_COLS; c++) {
	gauge.set(dir, c/N_COLS, c%N_COLS, sid,
	      scale[dir]*gauge.get(dir, c/N_COLS, c%N_COLS, sid));
      }
  }
}


template<typename Float, typename FloatGauge>
static void scale_dir_wise(gauge2<FloatGauge> gauge, Float* scale){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (gauge.volume() + blockDim.x -1)/blockDim.x , 1 , 1);
  Float2<Float> *d_scale;
  cudaMalloc((void**) &d_scale, N_DIMS*sizeof(Float2<Float>));
  cudaMemcpy( d_scale, scale, N_DIMS*sizeof(Float2<Float>),cudaMemcpyHostToDevice);
  scale_dir_wise_kernel<Float,FloatGauge><<<gridDim,blockDim>>>(gauge, d_scale);
  cudaFree(d_scale);
  checkCudaError();
}


template< typename Float, typename FloatGauge>
__global__ void qedPhase_kernel(gauge2<FloatGauge> gauge, gaugeU12<FloatGauge> gaugeU1, Float phase){

  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= gauge.volume()) return;

#pragma unroll
  for(int dir = 0; dir < N_DIMS; dir++) {
    Float theta = phase*gaugeU1.get(dir, sid).x;
    Float2<Float> scale = {cos(theta), sin(theta)};
#pragma unroll
      for(int c=0; c<N_COLS*N_COLS; c++) {
	gauge.set(dir, c/N_COLS, c%N_COLS, sid,
	      scale*gauge.get(dir, c/N_COLS, c%N_COLS, sid));
      }
  }
}


template<typename Float, typename FloatGauge>
static void qedPhase_k(gauge2<FloatGauge> gauge, gaugeU12<FloatGauge> gaugeU1, Float phase){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (gauge.volume() + blockDim.x -1)/blockDim.x , 1 , 1);
  qedPhase_kernel<Float,FloatGauge><<<gridDim,blockDim>>>(gauge, gaugeU1, phase);
  checkCudaError();
}
