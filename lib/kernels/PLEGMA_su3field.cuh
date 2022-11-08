#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_kernel_tuner.cuh>
#include <malloc_quda.h>
using namespace plegma;

template<typename FloatA>
static __global__ void Udag_kernel(su3_2<FloatA> RA){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= RA.volume()) return;
  Float2<FloatA> lA[N_COLS][N_COLS];
  RA.get(lA,sid);
  Gdag(lA);
  RA.set(lA,sid);
}


template<typename FloatA,typename FloatB>
static __global__ void Udag_kernel(su3_2<FloatA> RA, su3_2<FloatB> RB){
  
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= RA.volume()) return;

  Float2<FloatA> lA[N_COLS][N_COLS];
  Float2<FloatA> lB[N_COLS][N_COLS];
  RB.get(lB,sid);
  Gdag(lA,lB);
  RA.set(lA,sid);
}

template<typename FloatA,typename FloatB, typename FloatC>
static __global__ void U_plus_eq_aU_kernel(su3_2<FloatA> RA, su3_2<FloatB> RB, FloatC c){
  
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= RA.volume()) return;

  Float2<FloatA> lA[N_COLS][N_COLS];
  Float2<FloatB> lB[N_COLS][N_COLS];
  Float2<FloatA> lR[N_COLS][N_COLS];
  RA.get(lA,sid);
  RB.get(lB,sid);
  
  G_plus_aG( lR, lA, lB, c);

  RA.set(lR,sid);
}

template<typename FloatA,typename FloatB, typename FloatC>
static __global__ void UxU_kernel(su3_2<FloatA> RA, su3_2<FloatB> RB, su3_2<FloatC> RC){
  
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= RA.volume()) return;

  Float2<FloatA> lA[N_COLS][N_COLS];
  Float2<FloatA> lB[N_COLS][N_COLS];
  Float2<FloatA> lC[N_COLS][N_COLS];
  RB.get(lB,sid);
  RC.get(lC,sid);
  mul_G_G(lA,lB,lC);
  RA.set(lA,sid);
}


template<typename FloatA,typename FloatB, typename FloatC>
static __global__ void UxUdag_kernel(su3_2<FloatA> RA, su3_2<FloatB> RB, su3_2<FloatC> RC){
  
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= RA.volume()) return;

  Float2<FloatA> lA[N_COLS][N_COLS];
  Float2<FloatA> lB[N_COLS][N_COLS];
  Float2<FloatA> lC[N_COLS][N_COLS];
  RB.get(lB,sid);
  RC.get(lC,sid);
  mul_G_Gdag(lA,lB,lC);
  RA.set(lA,sid);
}

template<typename Float,typename FloatU>
static __global__ void sum_real_trace_kernel(su3_2<FloatU> RU, Float *partial_plaq){

  extern __shared__ int ext_shared_cache[];
  Float *shared_cache = (Float*)ext_shared_cache;
  
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  int cacheIndex = threadIdx.x;
  if(sid < RU.volume()){
  Float2<FloatU> lU[N_COLS][N_COLS];
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
static __global__ void traceHerExpMap_kernel(su3_2<FloatA> RA, su3_2<FloatB> RB){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= RA.volume()) return;
  Float2<FloatA> lA[N_COLS][N_COLS];
  Float2<FloatB> lB[N_COLS][N_COLS];
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

template<typename FloatA, typename FloatB, typename FloatC>
static void U_plus_eq_aU_k( PLEGMA_Su3field<FloatA> &A, PLEGMA_Su3field<FloatB> &B, FloatC c){
  assert(A.checkVolume(B));
  ProfileStruct ps(A.Total_length());
  run(ps, "U_plus_eq_aU_kernel", U_plus_eq_aU_kernel<FloatA,FloatB,FloatC>, toField2<su3_2>(A), toField2<su3_2>(B), c);
  checkQudaError();
}

template<typename FloatA, typename FloatB>
static void traceHerExpMap_k(PLEGMA_Su3field<FloatA> &A, PLEGMA_Su3field<FloatB> &B){
  assert(A.checkVolume(B));
  ProfileStruct ps(A.Total_length());
  tuneAndRun(ps, "traceHerExpMap_kernel", traceHerExpMap_kernel<FloatA,FloatB>, toField2<su3_2>(A), toField2<su3_2>(B));
  checkQudaError();
}

template<typename FloatA, typename FloatB>
static void Udag_k(PLEGMA_Su3field<FloatA> &A, PLEGMA_Su3field<FloatB> &B){
  assert(A.checkVolume(B));
  ProfileStruct ps(A.Total_length());
  tuneAndRun(ps, "Udag_kernel", Udag_kernel<FloatA,FloatB>,toField2<su3_2>(A), toField2<su3_2>(B));
  checkQudaError();
}

template<typename Float>
static void Udag_k(PLEGMA_Su3field<Float> &A){
  ProfileStruct ps(A.Total_length());
  run(ps, "Udag_kernel", Udag_kernel<Float>,toField2<su3_2>(A));
  checkQudaError();
}

template<typename FloatA, typename FloatB, typename FloatC>
static void UxU_k(PLEGMA_Su3field<FloatA> &A, PLEGMA_Su3field<FloatB> &B, PLEGMA_Su3field<FloatC> &C){
  assert(A.checkVolume(B,C));
  ProfileStruct ps(A.Total_length());
  tuneAndRun(ps, "UxU_kernel", UxU_kernel<FloatA,FloatB,FloatC>,toField2<su3_2>(A), toField2<su3_2>(B),toField2<su3_2>(C));
  checkQudaError();
}

template<typename FloatA, typename FloatB, typename FloatC>
static void UxUdag_k(PLEGMA_Su3field<FloatA> &A, PLEGMA_Su3field<FloatB> &B, PLEGMA_Su3field<FloatC> &C){
  assert(A.checkVolume(B,C));
  ProfileStruct ps(A.Total_length());
  tuneAndRun(ps, "UxUdag_kernel", UxUdag_kernel<FloatA,FloatB,FloatC>, toField2<su3_2>(A), toField2<su3_2>(B),toField2<su3_2>(C));
  checkQudaError();
}

template<typename Float, typename FloatS>
static void sum_real_trace_host(ProfileStruct& ps, PLEGMA_Su3field<FloatS> &su3M, Float& sum){
  Float *h_partial_sum = NULL;
  Float *d_partial_sum = NULL;
  sum=0.;
  int gridDimX = ps.tp.grid.x;
  
  hostMalloc(h_partial_sum, gridDimX * sizeof(Float) );
  d_partial_sum=(Float*)device_mallocMalloc(gridDimX * sizeof(Float));
//  d_partial_sum=quda::device_malloc_(__func__, quda::file_name(__FILE__), __LINE__, gridDimX * sizeof(Float));

  sum_real_trace_kernel<Float,FloatS><<<ps.tp.grid,ps.tp.block,ps.tp.shared_bytes>>>(toField2<su3_2>(su3M), d_partial_sum);

  cudaMemcpy(h_partial_sum, d_partial_sum , gridDimX * sizeof(Float) , cudaMemcpyDeviceToHost);
  cudaFree(d_partial_sum);
  checkQudaError();

  for(int i = 0 ; i < gridDimX ; i++)
    sum += h_partial_sum[i];
  hostFree(h_partial_sum, gridDimX * sizeof(Float) );
}

template<typename Float, typename FloatS>
static Float sumRtraceU_k(PLEGMA_Su3field<FloatS> &su3M){
  Float sum = 0.;

  ProfileStruct ps(su3M.Total_length(),sizeof(FloatS));
  tuneAndRun(ps, "sum_real_trace_host", sum_real_trace_host<Float,FloatS>, ps, su3M, sum);

  Float globalSum = 0.;
  MPI_Allreduce(&sum , &globalSum , 1 , MPI_Type(sum) , MPI_SUM , HGC_fullComm);  
  return globalSum;
}
