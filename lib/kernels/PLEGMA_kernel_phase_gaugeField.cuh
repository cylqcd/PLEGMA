#include <PLEGMA_kernel_utils.cuh>
using namespace plegma;

//*** provided the xi parameter and momentum in all four directions,
//we multiply the gauge links by an exponential phase.                                                                                                                                         

template< typename Float, typename FloatGauge>
  __global__ void phase_gauge_field_kernel(FloatGauge* gauge, Float* xi, int* mom){

  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= c_threads) return;

  Float2<FloatGauge> *G = (Float2<FloatGauge> *) gauge;

#pragma unroll
  for(int dir = 0; dir < N_DIMS; dir++)
    {
      Float theta= 2.0*PI*((Float)(mom[dir]))*xi[dir]/((Float)(c_totalL[dir]));
      Float2<Float> phase;
      phase.x=cos(theta);  phase.y=sin(theta);
#pragma unroll
      #pragma unroll
      for(int c1=0;c1<N_COLS*N_COLS;c1++)
        {
          G[(dir*N_COLS*N_COLS+c1)*c_stride+sid]=phase*(G[(dir*N_COLS*N_COLS+c1)*c_stride+sid]);
        }
    }
}


template<typename Float, typename FloatGauge>
  static void phase_gauge_field(FloatGauge* gauge, Float xi[4], int mom[4]){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (GK_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);

#ifdef TIMING_REPORT
  cudaEvent_t start,stop;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
#endif

  Float* xi_d;
  int* mom_d;
  cudaMalloc((void**) &xi_d, 4*sizeof(Float));

  cudaMemcpy( xi_d, xi, 4*sizeof(Float),cudaMemcpyHostToDevice);
  cudaMemcpy( mom_d, mom, 4*sizeof(int),cudaMemcpyHostToDevice);

  phase_gauge_field_kernel<Float,FloatGauge><<<gridDim,blockDim>>>(gauge, xi_d, mom_d);
  checkCudaError();

  #ifdef TIMING_REPORT
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime,start,stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  printfQuda("Elapsed time for phase multiplication is %f ms\n",elapsedTime);
#endif

  cudaFree(xi_d);
  cudaFree(mom_d);
}
