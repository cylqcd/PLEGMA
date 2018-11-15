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
