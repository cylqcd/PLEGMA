#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_kernel_tuner.cuh>
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

  extern __shared__ int ext_shared_cache[];
  Float *shared_cache = (Float*)ext_shared_cache;
  
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

  reduce(shared_cache,1);
  
  if(cacheIndex == 0 && partial_plaq!=NULL)
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
static void traceHerExpMap_k(PLEGMA_Su3field<FloatA> &A, PLEGMA_Su3field<FloatB> &B){
  ProfileStruct ps(GK_localVolume);
  tuneAndRun(ps,traceHerExpMap_kernel<FloatA,FloatB>, A.D_elem(), B.D_elem());
  checkCudaError();
}

template<typename FloatA, typename FloatB>
static void Udag_k(PLEGMA_Su3field<FloatA> &A, PLEGMA_Su3field<FloatB> &B){
  ProfileStruct ps(GK_localVolume);
  tuneAndRun(ps,Udag_kernel<FloatA,FloatB>,A.D_elem(), B.D_elem());
  checkCudaError();
}

template<typename FloatA, typename FloatB, typename FloatC>
static void UxU_k(PLEGMA_Su3field<FloatA> &A, PLEGMA_Su3field<FloatB> &B, PLEGMA_Su3field<FloatC> &C){
  ProfileStruct ps(GK_localVolume);
  tuneAndRun(ps,UxU_kernel<FloatA,FloatB,FloatC>,A.D_elem(), B.D_elem(),C.D_elem());
  checkCudaError();
}

template<typename FloatA, typename FloatB, typename FloatC>
static void UxUdag_k(PLEGMA_Su3field<FloatA> &A, PLEGMA_Su3field<FloatB> &B, PLEGMA_Su3field<FloatC> &C){
  ProfileStruct ps(GK_localVolume);
  tuneAndRun(ps,UxUdag_kernel<FloatA,FloatB,FloatC>,A.D_elem(), B.D_elem(),C.D_elem());
  checkCudaError();
}

template<typename Float, typename FloatS>
static Float sumRtraceU(PLEGMA_Su3field<FloatS> &su3M){
  Float sum = 0.;
  Float globalSum = 0.;
  Float *h_partial_sum = NULL;
  Float *d_partial_sum = NULL;

  ProfileStruct ps(GK_localVolume,sizeof(FloatS));
  tune(ps,sum_real_trace_kernel<Float,FloatS>,su3M.D_elem(), d_partial_sum);

  int gridDimX = ps.tp.grid.x;
  
  h_partial_sum = (Float*) malloc(gridDimX * sizeof(Float) );
  if(h_partial_sum == NULL) errorQuda("Error allocate memory for host partial sum");
  cudaMalloc((void**)&d_partial_sum, gridDimX * sizeof(Float));

  sum_real_trace_kernel<Float,FloatS><<<ps.tp.grid,ps.tp.block,ps.tp.shared_bytes>>>(su3M.D_elem(), d_partial_sum);

  cudaMemcpy(h_partial_sum, d_partial_sum , gridDimX * sizeof(Float) , cudaMemcpyDeviceToHost);
  cudaFree(d_partial_sum);
  checkCudaError();

  for(int i = 0 ; i < gridDimX ; i++)
    sum += h_partial_sum[i];
  free(h_partial_sum);

  MPI_Allreduce(&sum , &globalSum , 1 , MPI_Type(sum) , MPI_SUM , MPI_COMM_WORLD);  
  return globalSum;
}
