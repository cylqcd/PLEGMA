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


template< typename Float>
__global__ void qedPhase_kernel(gauge2<Float> gauge, gaugeU12<Float> gaugeU1, Float phase){

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


template<typename Float>
static void qedPhase_k(gauge2<Float> gauge, gaugeU12<Float> gaugeU1, Float phase){
  ProfileStruct ps(gauge.volume());
  tuneAndRun(ps, "qedPhase_kernel", qedPhase_kernel<Float>, gauge, gaugeU1, phase);
}


template< typename Float>
__global__ void mul_dag_kernel(gauge2<Float> gauge, gauge2<Float> gaugeIn){

  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= gauge.volume()) return;
  
  Float2<Float> G1[N_COLS][N_COLS], G2[N_COLS][N_COLS], G3[N_COLS][N_COLS];    

#pragma unroll
  for(int dir = 0; dir < N_DIMS; dir++) {
    gauge.get(G1,dir,sid);
    gaugeIn.get(G2,dir,sid);
    mul_G_Gdag(G3,G1,G2);
    gauge.set(G3, dir, sid);
  }
}


template<typename Float>
static void mul_dag_k(gauge2<Float> gauge, gauge2<Float> gaugeIn){
  ProfileStruct ps(gauge.volume());
  tuneAndRun(ps, "mul_dag_kernel", mul_dag_kernel<Float>, gauge, gaugeIn);
}
