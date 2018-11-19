#include <PLEGMA_kernel_utils.cuh>
using namespace plegma;

template<typename FloatA,typename FloatB>
static __global__ void Udag_kernel(FloatA *A, FloatB *B){
  
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= c_threads) return;

  Float2<FloatA> lA[N_COLS][N_COLS];
  Float2<FloatA> lB[N_COLS][N_COLS];

  su3_2<FloatA> RA(A);
  su3_2<FloatB> RB(B);

  RB.get(lB,sid);
  Gdag(lA,lB);
  RA.set(lA,sid);
}

template<typename FloatA,typename FloatB, typename FloatC>
static __global__ void UxU_kernel(FloatA *A, FloatB *B, FloatC *C){
  
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= c_threads) return;

  Float2<FloatA> lA[N_COLS][N_COLS];
  Float2<FloatA> lB[N_COLS][N_COLS];
  Float2<FloatA> lC[N_COLS][N_COLS];

  su3_2<FloatA> RA(A);
  su3_2<FloatB> RB(B);
  su3_2<FloatC> RC(C);

  RB.get(lB,sid);
  RC.get(lC,sid);
  mul_G_G(lA,lB,lC);
  RA.set(lA,sid);
}


template<typename FloatA,typename FloatB, typename FloatC>
static __global__ void UxUdag_kernel(FloatA *A, FloatB *B, FloatC *C){
  
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= c_threads) return;

  Float2<FloatA> lA[N_COLS][N_COLS];
  Float2<FloatA> lB[N_COLS][N_COLS];
  Float2<FloatA> lC[N_COLS][N_COLS];

  su3_2<FloatA> RA(A);
  su3_2<FloatB> RB(B);
  su3_2<FloatC> RC(C);

  RB.get(lB,sid);
  RC.get(lC,sid);
  mul_G_Gdag(lA,lB,lC);
  RA.set(lA,sid);
}

template<typename Float,typename FloatU>
static __global__ void sum_real_trace_kernel(FloatU *U, Float *partial_plaq){
  __shared__ Float shared_cache[THREADS_PER_BLOCK];
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  int cacheIndex = threadIdx.x;
  if(sid < c_threads){
  Float2<FloatU> lU[N_COLS][N_COLS];
  su3_2<FloatU> RU(U);
  RU.get(lU,sid);
  shared_cache[cacheIndex] = real_trace<Float>(lU);
  }
  else{
    shared_cache[cacheIndex] = 0.;
  }
  __syncthreads();

  //!!!!!!!!!!!!!!!!!!!!!!!!! Change to the one which works with not only powerrs of 2
  int i = blockDim.x/2;
  while (i != 0){
    if(cacheIndex < i)
      shared_cache[cacheIndex] += shared_cache[cacheIndex + i];
    __syncthreads();
    i /= 2;
  }
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
  if(cacheIndex == 0)
    partial_plaq[blockIdx.x] = shared_cache[0];   // write result back to global memory  
}

template<typename FloatA,typename FloatB>
static __global__ void traceHerExpMap_kernel(FloatA *A, FloatB *B){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= c_threads) return;
  Float2<FloatA> lA[N_COLS][N_COLS];
  Float2<FloatB> lB[N_COLS][N_COLS];
  su3_2<FloatA> RA(A);
  su3_2<FloatB> RB(B);
  RB.get(lB,sid);
  Gdag(lA,lB);
  G_plus_aG(lA,lB,-1.);
  FloatA div3=1./3.;
  Float2<FloatA> tr = div3*trace<FloatA,FloatA>(lA);
  Float2<FloatA> I;
  I.x=0.; I.y=0.5;
  lA[0][0] = lA[0][0] - tr;
  lA[1][1] = lA[1][1] - tr;
  lA[2][2] = lA[2][2] - tr;
  scaleG(lA,I);
  exponentiate_iQ(lA);
  RA.set(lA,sid);
}

template<typename FloatA, typename FloatB>
static void traceHerExpMap_kernel(PLEGMA_Su3field<FloatA> &A, PLEGMA_Su3field<FloatB> &B){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (GK_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  traceHerExpMap_kernel<FloatA,FloatB><<<gridDim,blockDim>>>(A.D_elem(), B.D_elem());
  checkCudaError();
}

template<typename FloatA, typename FloatB>
static void Udag_k(PLEGMA_Su3field<FloatA> &A, PLEGMA_Su3field<FloatB> &B){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (GK_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  Udag_kernel<FloatA,FloatB><<<gridDim,blockDim>>>(A.D_elem(), B.D_elem());
  checkCudaError();
}

template<typename FloatA, typename FloatB, typename FloatC>
static void UxU_k(PLEGMA_Su3field<FloatA> &A, PLEGMA_Su3field<FloatB> &B, PLEGMA_Su3field<FloatC> &C){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (GK_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  UxU_kernel<FloatA,FloatB,FloatC><<<gridDim,blockDim>>>(A.D_elem(), B.D_elem(),C.D_elem());
  checkCudaError();
}

template<typename FloatA, typename FloatB, typename FloatC>
static void UxUdag_k(PLEGMA_Su3field<FloatA> &A, PLEGMA_Su3field<FloatB> &B, PLEGMA_Su3field<FloatC> &C){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (GK_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  UxUdag_kernel<FloatA,FloatB,FloatC><<<gridDim,blockDim>>>(A.D_elem(), B.D_elem(),C.D_elem());
  checkCudaError();
}

template<typename Float, typename FloatS>
static Float sumRtraceU(PLEGMA_Su3field<FloatS> &su3M){
  Float sum = 0.;
  Float globalSum = 0.;
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (GK_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  Float *h_partial_sum = NULL;
  Float *d_partial_sum = NULL;
  h_partial_sum = (Float*) malloc(gridDim.x * sizeof(Float) );
  if(h_partial_sum == NULL) errorQuda("Error allocate memory for host partial sum");
  cudaMalloc((void**)&d_partial_sum, gridDim.x * sizeof(Float));

#ifdef TIMING_REPORT
  cudaEvent_t start,stop;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
#endif
  sum_real_trace_kernel<Float,FloatS><<<gridDim,blockDim>>>(su3M.D_elem(), d_partial_sum);
#ifdef TIMING_REPORT
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime,start,stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  printfQuda("Elapsed time for sumuette kernel is %f ms\n",elapsedTime);
#endif

  cudaMemcpy(h_partial_sum, d_partial_sum , gridDim.x * sizeof(Float) , cudaMemcpyDeviceToHost);
  cudaFree(d_partial_sum);
  checkCudaError();

  for(int i = 0 ; i < gridDim.x ; i++)
    sum += h_partial_sum[i];
  free(h_partial_sum);

  MPI_Allreduce(&sum , &globalSum , 1 , MPI_Type(sum) , MPI_SUM , MPI_COMM_WORLD);  
  return globalSum;
}
