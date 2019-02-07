#include <PLEGMA_BLAS.h>
#include <PLEGMA_Random.h>
#include <PLEGMA_kernel_utils.cuh>
#include <PLEGMA_gammas.cuh>
#include <PLEGMA_kernel_tuner.cuh>

using namespace plegma;
using namespace quda;

template<typename FloatOut,typename FloatIn>
static __global__ void castVector_kernel(FloatOut *out, FloatIn *in){
  
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= DGC_localVolume) return;

  #pragma unroll
  for(int mu = 0 ; mu < 4 ; mu++){
    #pragma unroll
    for(int c1 = 0 ; c1 < 3 ; c1++){
      #pragma unroll
      for(int ri = 0 ; ri < 2 ; ri++){
	out[((mu*3+c1)*DGC_stride+sid)*2+ri] = in[((mu*3+c1)*DGC_stride+sid)*2+ri];
      }
    }
  }
}

template<typename FloatOut,typename FloatIn>
static void castVector(FloatOut *out, FloatIn *in){
  ProfileStruct ps(HGC_localVolume);
  tuneAndRun(ps, "castVector_kernel", castVector_kernel<FloatOut,FloatIn>, (FloatOut*) out, (FloatIn*) in);
  checkCudaError();
}


template<LEFTRIGHT LF,typename Float>
static __global__ void apply_gamma_vector_kernel(Float *inOut, GAMMAS r){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  vector2<Float> vec(inOut);
  Float2<Float> Sin[N_SPINS][N_COLS];
  Float2<Float> Sout[N_SPINS][N_COLS]; 
  if (sid >= DGC_localVolume) return;
  vec.get(Sin,sid);
  gammaV<LF>(Sout,Sin,r);
  vec.set(Sout,sid);
}

template<typename Float>
static void apply_gamma_vector(LEFTRIGHT LR,Float *inOut,GAMMAS r){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (HGC_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  switch(LR){
  case(LEFT):
    apply_gamma_vector_kernel<LEFT><<<gridDim,blockDim>>>((Float*) inOut, r);
    break;
  case(RIGHT):
    apply_gamma_vector_kernel<RIGHT><<<gridDim,blockDim>>>((Float*) inOut, r);
    break;
  }
  checkCudaError();
}

template<typename Float>
static __global__ void apply_gamma5_vector_kernel(Float *inOut){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  Float2<Float> *inOut2 = (Float2<Float> *) inOut;
  if (sid >= DGC_localVolume) return;
    
  #pragma unroll
  for(int c1 = 0 ; c1 < N_COLS ; c1++){
    Float2<Float> spinor[4];
    // inline shuffling
    #pragma unroll
    for(int mu = 0 ; mu < N_SPINS ; mu++)
      spinor[(mu+2)%4] = inOut2[(mu*N_COLS+c1)*DGC_stride + sid];
    // replacing
    #pragma unroll
    for(int mu = 0 ; mu < N_SPINS ; mu++)
      inOut2[(mu*N_COLS+c1)*DGC_stride + sid] = spinor[mu];
  }

}

template<typename Float>
void apply_gamma5_vector(Float *inOut){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (HGC_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  apply_gamma5_vector_kernel<<<gridDim,blockDim>>>(inOut);
}



  
template<typename Float>
static __global__ void conjugate_vector_kernel(Float *inOut){

  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= DGC_localVolume) return;

  #pragma unroll
  for(int i = 0 ; i < N_SPINS*N_COLS ; i++)
    inOut[(i*DGC_stride + sid)*2 + 1] *= -1.;
}

template<typename Float>
void conjugate_vector(Float *inOut){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (HGC_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  conjugate_vector_kernel<<<gridDim,blockDim>>>(inOut);
  checkCudaError();
}

template<typename Float>
__inline__ __global__ void scale_vector_kernel(Float a, Float* inOut){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= DGC_localVolume) return;

  #pragma unroll
  for(int i = 0 ; i < N_SPINS*N_COLS ; i++) {
    inOut[(i*DGC_stride + sid)*2 + 0] *= a;
    inOut[(i*DGC_stride + sid)*2 + 1] *= a;
  }
}

template<typename Float>
void scale_vector(Float a, Float* inOut){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (HGC_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);

  scale_vector_kernel<<<gridDim,blockDim>>>( a, inOut);
  checkCudaError();
}

template<typename Float>
void norm2_device(Float norm, Float* in){

}

template<typename FloatIn, typename FloatOut, bool outEvenB, bool outOddB> 
static __global__ void copy_to_QUDA(FloatIn *in, FloatOut *outEven, FloatOut *outOdd){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= DGC_localVolume/2) return;

  // take indices on 4d lattice
  int half_stride = DGC_stride/2;
  int latt_coord = 2*sid;

  int r1,r2,x_id,y_id,z_id,t_id;
  r1 = latt_coord/(DGC_localL[0]);
  r2 = r1/(DGC_localL[1]);
  x_id = latt_coord - r1*(DGC_localL[0]);
  y_id = r1 - r2*(DGC_localL[1]);
  t_id = r2/(DGC_localL[2]);
  z_id = r2 - t_id*(DGC_localL[2]);
  int evenSiteBit = ((x_id+y_id+z_id+t_id) & 1);
  int oddSiteBit  = evenSiteBit ^ 1;
  Float2<FloatOut> *outEven2 = (Float2<FloatOut> *) outEven;
  Float2<FloatOut> *outOdd2 = (Float2<FloatOut> *) outOdd;
  Float2<FloatIn> *in2 = (Float2<FloatIn> *) in;

  #pragma unroll
  for(int mu = 0 ; mu < N_SPINS ; mu++) {
    #pragma unroll
    for(int ic = 0 ; ic < N_COLS ; ic++){
      if(outEvenB) {
	outEven2[(mu*N_COLS + ic)*half_stride + sid] =
	  in2[(mu*N_COLS + ic)*DGC_stride + latt_coord + evenSiteBit];
      } else
	outEven2[(mu*N_COLS + ic)*half_stride + sid] = 0.;

      if(outOddB) {
	outOdd2[(mu*N_COLS + ic)*half_stride + sid] =
	  in2[(mu*N_COLS + ic)*DGC_stride + latt_coord + oddSiteBit];
      } else
	outOdd2[(mu*N_COLS + ic)*half_stride + sid] = 0.;
    }
  }
}

template<typename FloatIn, typename FloatOut> 
static void copy_to_QUDA(FloatIn* in,ColorSpinorField &qudaVec, bool isEven){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (HGC_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  if( qudaVec.SiteSubset() == QUDA_PARITY_SITE_SUBSET ){
    if( isEven )
      copy_to_QUDA<FloatIn,FloatOut,true,false><<<gridDim,blockDim>>>(in,(FloatOut*) qudaVec.V(), NULL);
    else
      copy_to_QUDA<FloatIn,FloatOut,false,true><<<gridDim,blockDim>>>(in, NULL,(FloatOut*) qudaVec.V());
  } else
    copy_to_QUDA<FloatIn,FloatOut,true,true><<<gridDim,blockDim>>>(in, (FloatOut*) qudaVec.Even().V(),(FloatOut*) qudaVec.Odd().V());
}

template<typename FloatIn> 
static void copy_to_QUDA(FloatIn* in, ColorSpinorField &qudaVec, bool isEven){
  if( qudaVec.Precision() == QUDA_SINGLE_PRECISION )
    copy_to_QUDA<FloatIn,float>(in, qudaVec, isEven);
  else if ( qudaVec.Precision() == QUDA_DOUBLE_PRECISION )
    copy_to_QUDA<FloatIn,double>(in, qudaVec, isEven);
  else
    errorQuda("Precision %d not supported", qudaVec.Precision());

  checkCudaError();
}

template<typename FloatOut, typename FloatIn, bool inEvenB, bool inOddB> 
static __global__ void copy_from_QUDA_kernel(FloatOut *out, FloatIn *inEven, FloatIn *inOdd){
  
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= DGC_localVolume/2) return;

  int half_stride = DGC_stride/2;
  int latt_coord = 2*sid;

  int r1,r2,x_id,y_id,z_id,t_id;
  r1 = latt_coord/(DGC_localL[0]);
  r2 = r1/(DGC_localL[1]);
  x_id = latt_coord - r1*(DGC_localL[0]);
  y_id = r1 - r2*(DGC_localL[1]);
  t_id = r2/(DGC_localL[2]);
  z_id = r2 - t_id*(DGC_localL[2]);
  int evenSiteBit = ((x_id+y_id+z_id+t_id) & 1);
  int oddSiteBit  = evenSiteBit ^ 1;
  Float2<FloatIn> *inEven2 = (Float2<FloatIn> *) inEven;
  Float2<FloatIn> *inOdd2 = (Float2<FloatIn> *) inOdd;
  Float2<FloatOut> *out2 = (Float2<FloatOut> *) out;

  #pragma unroll
  for(int mu = 0 ; mu < N_SPINS ; mu++) {
    #pragma unroll
    for(int ic = 0 ; ic < N_COLS ; ic++) {
      if(inEvenB) {
	out2[(mu*N_COLS + ic)*DGC_stride + latt_coord + evenSiteBit] =
	  inEven2[(mu*N_COLS + ic)*half_stride + sid];
      } else
	out2[(mu*N_COLS + ic)*DGC_stride + latt_coord + evenSiteBit] = 0.;

      if(inOddB) {
	out2[(mu*N_COLS + ic)*DGC_stride + latt_coord + oddSiteBit] =
	  inOdd2[(mu*N_COLS + ic)*half_stride + sid];
      } else
	out2[(mu*N_COLS + ic)*DGC_stride + latt_coord + oddSiteBit] = 0.;
    }
  }
}

template<typename FloatOut, typename FloatIn> 
static void copy_from_QUDA(FloatOut* out, ColorSpinorField &qudaVec, bool isEven){
  ProfileStruct ps(HGC_localVolume);
  if( qudaVec.SiteSubset() == QUDA_PARITY_SITE_SUBSET ){
    if( isEven )
      tuneAndRun(ps, "copy_from_QUDA_kernel", copy_from_QUDA_kernel<FloatOut,FloatIn,true,false>,out,(FloatIn*) qudaVec.V(), (FloatIn*) NULL);
    else
      tuneAndRun(ps, "copy_from_QUDA_kernel", copy_from_QUDA_kernel<FloatOut,FloatIn,false,true>, out, (FloatIn*) NULL,(FloatIn*) qudaVec.V());
  } else
    tuneAndRun(ps, "copy_from_QUDA_kernel", copy_from_QUDA_kernel<FloatOut,FloatIn,true,true>, out, (FloatIn*) qudaVec.Even().V(),(FloatIn*) qudaVec.Odd().V());
}

template<typename FloatOut> 
static void copy_from_QUDA(FloatOut* out, ColorSpinorField &qudaVec, bool isEven){
  if( qudaVec.Precision() == QUDA_SINGLE_PRECISION )
    copy_from_QUDA<FloatOut,float>(out, qudaVec, isEven);
  else if ( qudaVec.Precision() == QUDA_DOUBLE_PRECISION )
    copy_from_QUDA<FloatOut,double>(out, qudaVec, isEven);
  else
    errorQuda("Precision %d not supported", qudaVec.Precision());
  
  checkCudaError();
}
